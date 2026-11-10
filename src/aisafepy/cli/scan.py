"""``aisafepy scan``: run Tier-1 deterministic guards over files or stdin.

This is the "grep for secrets and PII, but with the same guards your
pipeline uses" command. It builds a list of :class:`RegexGuard` /
:class:`AhoCorasickGuard` instances from the flags, evaluates each
input against every guard independently, and prints one row per
(file, guard) pair. The exit code is 2 when any decision reaches the
``--fail-on`` threshold, so the command slots into pre-commit hooks and
CI jobs without wrapper scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError
from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import Context

DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
)

_ACTION_RANK: dict[str, int] = {
    "never": 99,
    "allow": 0,
    "transform": 1,
    "escalate": 2,
    "block": 3,
}


@dataclass
class ScanRow:
    """One (input, guard) evaluation result."""

    source: str
    guard: str
    decision: GuardDecision

    def as_dict(self) -> dict[str, Any]:
        d = self.decision
        return {
            "source": self.source,
            "guard": self.guard,
            "action": d.action.value,
            "severity": d.severity,
            "confidence": d.confidence,
            "rationale": d.rationale,
            "matches": _matches(d),
            "latency_ms": round(d.latency_ms, 3),
            "transformed_content": d.transformed_content,
        }


@dataclass
class ScanSummary:
    rows: list[ScanRow] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    files: int = 0
    elapsed_s: float = 0.0

    def worst_action(self) -> str:
        worst = "allow"
        for row in self.rows:
            a = row.decision.action.value
            if _ACTION_RANK.get(a, 0) > _ACTION_RANK.get(worst, 0):
                worst = a
        return worst

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows:
            key = row.decision.action.value
            out[key] = out.get(key, 0) + 1
        return out


# ---- argparse ------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "scan",
        help="run Tier-1 guards (PII, secrets, blocklists, custom regex) over files or stdin",
        description=(
            "Evaluate files, directories, or stdin against deterministic Tier-1 guards. "
            "With no guard flags the built-in PII set is used."
        ),
    )
    p.add_argument(
        "paths",
        nargs="*",
        default=["-"],
        metavar="PATH",
        help="files or directories to scan; '-' (default) reads stdin",
    )
    p.add_argument("--pii", action="store_true", help="enable the built-in PII pattern set")
    p.add_argument(
        "--redact",
        action="store_true",
        help="with --pii, emit a transform (redacted text) instead of a block",
    )
    p.add_argument(
        "--blocklist",
        action="append",
        default=[],
        metavar="TERM",
        help="literal term to block (repeatable)",
    )
    p.add_argument(
        "--blocklist-file",
        metavar="FILE",
        help="file with one blocklist term per line; '#' starts a comment",
    )
    p.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="named custom regex (repeatable); case-insensitive, multiline",
    )
    p.add_argument(
        "--pattern-action",
        choices=["block", "transform"],
        default="block",
        help="action for --pattern matches; transform redacts the span (default: block)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=50_000,
        help="cap on characters fed to each guard; longer inputs use the tail (default 50000)",
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="when scanning directories, only include files matching GLOB (repeatable)",
    )
    p.add_argument(
        "--no-default-ignores",
        action="store_true",
        help=f"do not skip {', '.join(DEFAULT_IGNORES[:4])}, ... when walking directories",
    )
    p.add_argument(
        "--format",
        choices=["table", "json", "jsonl"],
        default="table",
        help="output format (default: table)",
    )
    p.add_argument(
        "--fail-on",
        choices=["block", "escalate", "transform", "never"],
        default="block",
        help="lowest action that produces exit code 2 (default: block)",
    )
    p.add_argument(
        "--show-allowed",
        action="store_true",
        help="also print rows for guards that allowed the input",
    )
    p.set_defaults(handler=run)


# ---- guard construction --------------------------------------------------


def build_guards(args: argparse.Namespace) -> list[Any]:
    """Translate the scan flags into Tier-1 guard instances."""
    guards: list[Any] = []

    terms = list(args.blocklist)
    if args.blocklist_file:
        terms.extend(_read_blocklist_file(args.blocklist_file))
    patterns = _io.parse_key_values(args.pattern)

    use_pii = args.pii or (not terms and not patterns)
    if use_pii:
        guard = RegexGuard.compile_pii(name="pii", redact=args.redact)
        guard.max_input_chars = args.max_chars
        guards.append(guard)
    if terms:
        guards.append(AhoCorasickGuard(name="blocklist", terms=_dedupe(terms)))
    if patterns:
        import re

        compiled: dict[str, Any] = {}
        for name, raw in patterns.items():
            try:
                compiled[name] = re.compile(raw, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                raise CLIError(f"invalid regex for --pattern {name}: {exc}") from exc
        guard = RegexGuard.from_patterns(
            name="custom", patterns=compiled, action_on_match=args.pattern_action
        )
        guard.max_input_chars = args.max_chars
        if args.pattern_action == "transform":
            guard.redact = True
        guards.append(guard)
    return guards


def _read_blocklist_file(path: str) -> list[str]:
    text = _io.read_text(path)
    out: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    if not out:
        raise CLIError(f"blocklist file {path} contains no terms")
    return out


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ---- input discovery -----------------------------------------------------


def iter_inputs(
    paths: Iterable[str],
    *,
    include: list[str],
    ignores: tuple[str, ...],
    stdin: IO[str] | None,
) -> Iterator[tuple[str, str | None, str | None]]:
    """Yield ``(label, text, skip_reason)`` for every scannable input.

    ``text`` is ``None`` when the input was skipped, in which case
    ``skip_reason`` says why. Directories are walked recursively.
    """
    for raw in paths:
        if raw == "-":
            source = stdin
            if source is None:
                import sys

                source = sys.stdin
            yield "<stdin>", source.read(), None
            continue
        p = Path(raw)
        if not p.exists():
            raise CLIError(f"no such file or directory: {p}")
        if p.is_file():
            yield from _read_file(p)
            continue
        for child in sorted(p.rglob("*")):
            if not child.is_file():
                continue
            rel_parts = child.relative_to(p).parts
            if any(part in ignores for part in rel_parts[:-1]):
                continue
            if include and not any(
                fnmatch.fnmatch(child.name, g) or fnmatch.fnmatch(child.as_posix(), g)
                for g in include
            ):
                continue
            yield from _read_file(child)


def _read_file(path: Path) -> Iterator[tuple[str, str | None, str | None]]:
    label = str(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        yield label, None, f"unreadable: {exc.strerror or exc}"
        return
    if _io.is_probably_binary(data):
        yield label, None, "binary file"
        return
    try:
        yield label, data.decode("utf-8"), None
    except UnicodeDecodeError:
        yield label, None, "not valid UTF-8"


# ---- evaluation ----------------------------------------------------------


def evaluate_text(guards: list[Any], text: str, *, source: str, max_chars: int) -> list[ScanRow]:
    """Run every guard against ``text`` and collect the rows."""
    if len(text) > max_chars:
        text = text[-max_chars:]
    ctx = Context(buffer=text)
    rows: list[ScanRow] = []
    for guard in guards:
        decision = _run_guard(guard, ctx)
        rows.append(ScanRow(source=source, guard=guard.name, decision=decision))
    return rows


def _run_guard(guard: Any, ctx: Context) -> GuardDecision:
    """Drive one async guard to completion from synchronous code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(guard(ctx))
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, guard(ctx)).result()


def scan(
    paths: Iterable[str],
    guards: list[Any],
    *,
    max_chars: int,
    include: list[str] | None = None,
    ignores: tuple[str, ...] = DEFAULT_IGNORES,
    stdin: IO[str] | None = None,
) -> ScanSummary:
    """Library-friendly entry point used by :func:`run` and by tests."""
    summary = ScanSummary()
    started = time.perf_counter()
    for label, text, reason in iter_inputs(
        paths, include=include or [], ignores=ignores, stdin=stdin
    ):
        if text is None:
            summary.skipped.append((label, reason or "skipped"))
            continue
        summary.files += 1
        summary.rows.extend(evaluate_text(guards, text, source=label, max_chars=max_chars))
    summary.elapsed_s = time.perf_counter() - started
    return summary


# ---- command -------------------------------------------------------------


def run(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    guards = build_guards(args)
    ignores = () if args.no_default_ignores else DEFAULT_IGNORES
    summary = scan(
        args.paths,
        guards,
        max_chars=args.max_chars,
        include=args.include,
        ignores=ignores,
        stdin=getattr(args, "stdin", None),
    )

    visible = [
        r for r in summary.rows if args.show_allowed or r.decision.action != Action.ALLOW
    ]
    fmt = "json" if args.json and args.format == "table" else args.format

    if fmt == "json":
        _io.dump_json(
            out,
            {
                "files": summary.files,
                "skipped": [{"source": s, "reason": r} for s, r in summary.skipped],
                "counts": summary.counts(),
                "worst_action": summary.worst_action(),
                "rows": [r.as_dict() for r in visible],
                "elapsed_s": round(summary.elapsed_s, 4),
            },
        )
    elif fmt == "jsonl":
        for row in visible:
            _io.dump_json(out, row.as_dict(), indent=None)
    else:
        if visible:
            table = _io.render_table(
                ["source", "guard", "action", "severity", "rationale", "matches"],
                (
                    [
                        r.source,
                        r.guard,
                        r.decision.action.value,
                        r.decision.severity,
                        r.decision.rationale,
                        ", ".join(_matches(r.decision)[:4]),
                    ]
                    for r in visible
                ),
            )
            _io.emit(out, table)
        if not args.quiet:
            for label, reason in summary.skipped:
                _io.emit(err, f"skipped {label}: {reason}")
            counts = ", ".join(f"{k}={v}" for k, v in sorted(summary.counts().items())) or "none"
            _io.emit(
                out,
                f"{_io.human_count('file', summary.files)} scanned, "
                f"{_io.human_count('guard', len(guards))}, decisions: {counts} "
                f"({summary.elapsed_s * 1000:.1f} ms)",
            )

    threshold = _ACTION_RANK[args.fail_on]
    worst = _ACTION_RANK[summary.worst_action()]
    if args.fail_on != "never" and worst >= threshold:
        return EXIT_FLAGGED
    return EXIT_OK


def _matches(decision: GuardDecision) -> list[str]:
    raw = decision.evidence.get("matches", [])
    out: list[str] = []
    for m in raw:
        if isinstance(m, list | tuple) and len(m) == 2:
            out.append(f"{m[0]}:{m[1]}")
        else:
            out.append(str(m))
    return out
