"""``aisafepy audit``: operate on the hash-chained :class:`AuditLog`.

``verify`` is the command you wire into a nightly job: it walks the
chain and exits 2 if any entry was edited, reordered, or deleted.
``tail`` and ``stats`` are for humans looking at a log; ``append`` lets
shell scripts write governance events (deploys, approvals, rollbacks)
into the same chain the compiler uses.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import IO, Any

from aisafepy.adapt.governance import AuditEntry, AuditLog
from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "audit",
        help="verify, tail, append to, or summarise an append-only audit log",
        description="Operate on aisafepy.adapt.governance.AuditLog JSON Lines files.",
    )
    sub = p.add_subparsers(dest="audit_command", metavar="SUBCOMMAND")
    sub.required = True

    verify = sub.add_parser("verify", help="check the SHA-256 hash chain")
    verify.add_argument("log", metavar="LOG.jsonl")
    verify.set_defaults(handler=run_verify)

    tail = sub.add_parser("tail", help="print the most recent entries")
    tail.add_argument("log", metavar="LOG.jsonl")
    tail.add_argument("-n", type=int, default=20, help="number of entries (default 20)")
    tail.add_argument("--event", help="only entries with this event name")
    tail.add_argument("--actor", help="only entries by this actor")
    tail.set_defaults(handler=run_tail)

    append = sub.add_parser("append", help="append one entry (creates the log if missing)")
    append.add_argument("log", metavar="LOG.jsonl")
    append.add_argument("--event", required=True, help="event name, e.g. guard.promoted")
    append.add_argument("--actor", required=True, help="who or what performed the action")
    append.add_argument(
        "--payload",
        default="{}",
        metavar="JSON",
        help="JSON object with event details (default: {})",
    )
    append.set_defaults(handler=run_append)

    stats = sub.add_parser("stats", help="counts per event and actor, time span")
    stats.add_argument("log", metavar="LOG.jsonl")
    stats.set_defaults(handler=run_stats)


# ---- helpers --------------------------------------------------------------


def _open_existing(path: str) -> AuditLog:
    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such audit log: {p}")
    if not p.is_file():
        raise CLIError(f"not a file: {p}")
    try:
        return AuditLog(path=p)
    except json.JSONDecodeError as exc:
        raise CLIError(f"{p} is not valid JSON Lines: line {exc.lineno}: {exc.msg}") from exc


def locate_break(path: Path) -> tuple[int, str] | None:
    """Return ``(line_number, reason)`` of the first chain break, or ``None``.

    Mirrors :meth:`AuditLog.verify` but keeps going until it can name
    the offending line, which the boolean API does not expose.
    """
    prev: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                return lineno, f"invalid JSON: {exc.msg}"
            claimed = obj.pop("__hash__", None)
            try:
                entry = AuditEntry(
                    id=obj["id"],
                    event=obj["event"],
                    timestamp=obj["timestamp"],
                    actor=obj["actor"],
                    payload=obj["payload"],
                    prev_hash=obj.get("prev_hash"),
                )
            except (KeyError, TypeError) as exc:
                return lineno, f"malformed entry: {exc}"
            if entry.prev_hash != prev:
                return lineno, "prev_hash does not match the previous entry"
            if entry.compute_hash() != claimed:
                return lineno, "entry hash does not match its content"
            prev = claimed
    return None


def _entry_dict(e: AuditEntry) -> dict[str, Any]:
    return {
        "id": e.id,
        "event": e.event,
        "timestamp": e.timestamp,
        "time": _fmt_time(e.timestamp),
        "actor": e.actor,
        "payload": e.payload,
        "prev_hash": e.prev_hash,
    }


def _fmt_time(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(ts))) + "Z"
    except (TypeError, ValueError, OverflowError, OSError):
        return str(ts)


# ---- commands -------------------------------------------------------------


def run_verify(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    log = _open_existing(args.log)
    entries = log.entries()
    problem = locate_break(Path(args.log))
    ok = problem is None and log.verify()

    if args.json:
        _io.dump_json(
            out,
            {
                "log": args.log,
                "entries": len(entries),
                "ok": ok,
                "first_break": (
                    {"line": problem[0], "reason": problem[1]} if problem else None
                ),
            },
        )
    elif ok:
        _io.emit(out, f"OK: {args.log} chain intact ({_io.human_count('entry', len(entries))})")
    else:
        line, reason = problem or (0, "chain verification failed")
        _io.emit(out, f"FAIL: {args.log} line {line}: {reason}")
    return EXIT_OK if ok else EXIT_FLAGGED


def run_tail(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    log = _open_existing(args.log)
    entries = log.entries()
    if args.event:
        entries = [e for e in entries if e.event == args.event]
    if args.actor:
        entries = [e for e in entries if e.actor == args.actor]
    n = max(0, args.n)
    shown = entries[-n:] if n else []

    if args.json:
        _io.dump_json(out, [_entry_dict(e) for e in shown])
        return EXIT_OK
    if not shown:
        _io.emit(out, "no entries")
        return EXIT_OK
    rows = [
        [_fmt_time(e.timestamp), e.event, e.actor, json.dumps(e.payload, default=str), e.id[:12]]
        for e in shown
    ]
    _io.emit(out, _io.render_table(["time", "event", "actor", "payload", "id"], rows, max_width=60))
    return EXIT_OK


def run_append(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        raise CLIError(f"--payload is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CLIError("--payload must be a JSON object")
    log = AuditLog(path=Path(args.log))
    entry = log.append(args.event, actor=args.actor, payload=payload)
    if args.json:
        _io.dump_json(out, _entry_dict(entry))
    elif not args.quiet:
        _io.emit(out, f"appended {entry.event} by {entry.actor} as {entry.id}")
    return EXIT_OK


def run_stats(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    log = _open_existing(args.log)
    entries = log.entries()
    by_event = Counter(e.event for e in entries)
    by_actor = Counter(e.actor for e in entries)
    first = min((e.timestamp for e in entries), default=None)
    last = max((e.timestamp for e in entries), default=None)
    intact = log.verify() if entries else True

    if args.json:
        _io.dump_json(
            out,
            {
                "log": args.log,
                "entries": len(entries),
                "chain_ok": intact,
                "first": first,
                "last": last,
                "by_event": dict(by_event.most_common()),
                "by_actor": dict(by_actor.most_common()),
            },
        )
        return EXIT_OK

    _io.emit(out, f"{args.log}: {_io.human_count('entry', len(entries))}, chain {'ok' if intact else 'BROKEN'}")
    if first is not None and last is not None:
        _io.emit(out, f"  span: {_fmt_time(first)} .. {_fmt_time(last)}")
    if by_event:
        _io.emit(out)
        _io.emit(out, _io.render_table(["event", "count"], by_event.most_common()))
    if by_actor:
        _io.emit(out)
        _io.emit(out, _io.render_table(["actor", "count"], by_actor.most_common()))
    return EXIT_OK if intact else EXIT_FLAGGED
