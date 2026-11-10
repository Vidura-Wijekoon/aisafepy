"""``aisafepy policy``: evaluate a runtime :class:`PolicyDocument` from the shell.

Three subcommands share one loader:

``check``
    First-match evaluation, exactly what the runtime does. Prints the
    winning rule and exits 2 when its decision is ``block`` or
    ``escalate``.
``explain``
    Evaluates every rule and reports matched / unmatched, which is the
    fastest way to debug rule precedence while authoring a document.
``show``
    Dumps the loaded document (ids, decisions, severities, selector
    kinds) so you can confirm what the YAML actually produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError
from aisafepy.core.policies import PolicyDocument, PolicyRule

_SELECTOR_KEYS = ("pattern", "contains", "equals", "not_empty")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "policy",
        help="evaluate, explain, or display a PolicyDocument YAML",
        description="Work with runtime policy documents (aisafepy.core.policies).",
    )
    sub = p.add_subparsers(dest="policy_command", metavar="SUBCOMMAND")
    sub.required = True

    check = sub.add_parser("check", help="first-match evaluation against an input")
    _add_input_flags(check)
    check.set_defaults(handler=run_check)

    explain = sub.add_parser("explain", help="evaluate every rule and show which matched")
    _add_input_flags(explain)
    explain.set_defaults(handler=run_explain)

    show = sub.add_parser("show", help="print the loaded document")
    show.add_argument("policy", metavar="POLICY.yaml")
    show.set_defaults(handler=run_show)


def _add_input_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("policy", metavar="POLICY.yaml", help="policy document to load")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="TEXT", help="content to evaluate")
    src.add_argument(
        "--input-file", metavar="FILE", help="read content from FILE ('-' for stdin)"
    )
    p.add_argument(
        "--role",
        choices=["user", "assistant", "tool", "system"],
        default="user",
        help="value for context['role'] (default: user)",
    )
    p.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra context field for selectors using 'field:' (repeatable)",
    )


# ---- shared ---------------------------------------------------------------


def load_document(path: str) -> PolicyDocument:
    """Load a document, turning loader failures into one-line errors."""
    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such policy file: {p}")
    try:
        return PolicyDocument.from_yaml(p)
    except RuntimeError as exc:  # PyYAML missing
        raise CLIError(str(exc)) from exc
    except KeyError as exc:
        raise CLIError(f"policy {p} is missing required key {exc}") from exc
    except Exception as exc:
        raise CLIError(f"could not load policy {p}: {type(exc).__name__}: {exc}") from exc


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the free-form context dict a selector receives."""
    if args.input is not None:
        content = args.input
    else:
        content = _io.read_text(args.input_file, stdin=getattr(args, "stdin", None))
    ctx: dict[str, Any] = {"content": content, "role": args.role}
    ctx.update(_io.parse_key_values(args.field))
    return ctx


def rule_summary(rule: PolicyRule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "decision": rule.decision,
        "severity": rule.severity,
        "rationale": rule.rationale,
        "metadata": rule.metadata,
    }


def _flagged(decision: str) -> bool:
    return decision in ("block", "escalate")


# ---- check ----------------------------------------------------------------


def run_check(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    doc = load_document(args.policy)
    ctx = build_context(args)
    rule = doc.evaluate(ctx)

    if args.json:
        _io.dump_json(
            out,
            {
                "policy": doc.name,
                "version": doc.version,
                "matched": rule_summary(rule) if rule else None,
                "flagged": bool(rule and _flagged(rule.decision)),
            },
        )
    elif rule is None:
        _io.emit(out, f"{doc.name} v{doc.version}: no match (default allow)")
    else:
        _io.emit(
            out,
            f"{doc.name} v{doc.version}: matched rule '{rule.id}' -> "
            f"{rule.decision} [{rule.severity}]: {rule.rationale}",
        )
    return EXIT_FLAGGED if rule and _flagged(rule.decision) else EXIT_OK


# ---- explain --------------------------------------------------------------


def run_explain(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    doc = load_document(args.policy)
    ctx = build_context(args)
    results = [(rule, rule.matches(ctx)) for rule in doc.rules]
    first = next((r for r, hit in results if hit), None)

    if args.json:
        _io.dump_json(
            out,
            {
                "policy": doc.name,
                "version": doc.version,
                "winner": first.id if first else None,
                "rules": [{**rule_summary(r), "matched": hit} for r, hit in results],
            },
        )
    else:
        rows = []
        for rule, hit in results:
            marker = "MATCH" if hit else "-"
            if first is not None and rule is first:
                marker = "MATCH (wins)"
            rows.append([rule.id, rule.decision, rule.severity, marker, rule.rationale])
        _io.emit(out, _io.render_table(["rule", "decision", "severity", "result", "rationale"], rows))
        if not args.quiet:
            outcome = f"{first.decision} via '{first.id}'" if first else "allow (no match)"
            _io.emit(out, f"outcome: {outcome}")
    return EXIT_FLAGGED if first and _flagged(first.decision) else EXIT_OK


# ---- show -----------------------------------------------------------------


def run_show(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    doc = load_document(args.policy)
    raw = _io.load_structured(args.policy)
    raw_rules = {r.get("id"): r for r in (raw or {}).get("rules", []) if isinstance(r, dict)}

    rows = []
    for rule in doc.rules:
        src = raw_rules.get(rule.id, {})
        kind = next((k for k in _SELECTOR_KEYS if k in src), "never-matches")
        detail = src.get(kind, "") if kind != "never-matches" else ""
        rows.append([rule.id, rule.decision, rule.severity, kind, detail, rule.rationale])

    if args.json:
        _io.dump_json(
            out,
            {
                "name": doc.name,
                "version": doc.version,
                "description": doc.description,
                "metadata": doc.metadata,
                "rules": [
                    {**rule_summary(r), "selector": row[3], "selector_arg": row[4]}
                    for r, row in zip(doc.rules, rows, strict=True)
                ],
            },
        )
        return EXIT_OK

    _io.emit(out, f"{doc.name} v{doc.version}")
    if doc.description:
        _io.emit(out, f"  {doc.description}")
    _io.emit(out, f"  {_io.human_count('rule', len(doc.rules))}, first match wins")
    _io.emit(out)
    _io.emit(
        out,
        _io.render_table(
            ["rule", "decision", "severity", "selector", "argument", "rationale"], rows
        ),
    )
    return EXIT_OK
