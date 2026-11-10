"""``aisafepy decisions``: summarise and filter ``GuardDecision`` JSON Lines.

Any sink that writes ``decision.model_dump()`` one per line (a log
handler, an OTel exporter side-channel, the examples in the repo)
produces a file this command understands. ``summarize`` answers the
on-call questions: what is blocking, how often, how slow. ``filter``
narrows a large log to the rows worth reading and re-emits JSONL so the
output pipes into ``jq`` or back into ``summarize``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_OK, CLIError

_ACTIONS = ("allow", "block", "transform", "escalate")
_SEVERITIES = ("info", "low", "medium", "high", "critical")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "decisions",
        help="summarise or filter a JSONL log of GuardDecision records",
        description="Analyse GuardDecision.model_dump() records stored as JSON Lines.",
    )
    sub = p.add_subparsers(dest="decisions_command", metavar="SUBCOMMAND")
    sub.required = True

    summarize = sub.add_parser("summarize", help="counts, latency percentiles, top rationales")
    summarize.add_argument("log", metavar="DECISIONS.jsonl", help="file or '-' for stdin")
    summarize.add_argument("--top", type=int, default=10, help="rationales to list (default 10)")
    summarize.set_defaults(handler=run_summarize)

    filt = sub.add_parser("filter", help="select records and re-emit them as JSONL")
    filt.add_argument("log", metavar="DECISIONS.jsonl", help="file or '-' for stdin")
    filt.add_argument("--action", action="append", choices=_ACTIONS, help="keep these actions")
    filt.add_argument(
        "--severity", action="append", choices=_SEVERITIES, help="keep these severities"
    )
    filt.add_argument("--guard", action="append", metavar="NAME", help="keep these guard names")
    filt.add_argument("--tier", type=int, action="append", help="keep these tiers")
    filt.add_argument("--min-confidence", type=float, default=None)
    filt.add_argument("--max-confidence", type=float, default=None)
    filt.add_argument("--min-latency-ms", type=float, default=None)
    filt.add_argument("--contains", metavar="TEXT", help="rationale contains TEXT (case-insensitive)")
    filt.add_argument("--limit", type=int, default=None, help="stop after N matching records")
    filt.add_argument("--count", action="store_true", help="print only the number of matches")
    filt.set_defaults(handler=run_filter)


# ---- loading ---------------------------------------------------------------


def load_decisions(path: str, *, stdin: IO[str] | None) -> list[dict[str, Any]]:
    """Load decision records, validating the handful of fields we rely on."""
    records: list[dict[str, Any]] = []
    for i, obj in enumerate(_io.iter_jsonl(path, stdin=stdin), start=1):
        action = obj.get("action")
        if action not in _ACTIONS:
            raise CLIError(f"{path}:{i}: 'action' must be one of {_ACTIONS}, got {action!r}")
        records.append(obj)
    return records


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---- summarize -------------------------------------------------------------


def summarize(records: list[dict[str, Any]], *, top: int = 10) -> dict[str, Any]:
    """Compute the summary as a plain dict so it renders to JSON or text."""
    by_action = Counter(r["action"] for r in records)
    by_severity = Counter(str(r.get("severity", "unknown")) for r in records)
    by_guard = Counter(str(r.get("guard_name") or "<unnamed>") for r in records)
    by_tier = Counter(int(_num(r.get("tier"), 0)) for r in records)
    latencies = [_num(r.get("latency_ms")) for r in records]
    blocked = [r for r in records if r["action"] == "block"]
    rationales = Counter(str(r.get("rationale", "")) for r in records if r["action"] != "allow")

    guard_rows = []
    for guard, n in by_guard.most_common():
        mine = [r for r in records if str(r.get("guard_name") or "<unnamed>") == guard]
        blocks = sum(1 for r in mine if r["action"] == "block")
        lat = [_num(r.get("latency_ms")) for r in mine]
        guard_rows.append(
            {
                "guard": guard,
                "decisions": n,
                "blocks": blocks,
                "block_rate": round(blocks / n, 4) if n else 0.0,
                "p95_latency_ms": round(_io.percentile(lat, 95), 3),
            }
        )

    return {
        "records": len(records),
        "by_action": dict(by_action),
        "by_severity": dict(by_severity),
        "by_tier": {str(k): v for k, v in sorted(by_tier.items())},
        "block_rate": round(len(blocked) / len(records), 4) if records else 0.0,
        "latency_ms": {
            "p50": round(_io.percentile(latencies, 50), 3),
            "p95": round(_io.percentile(latencies, 95), 3),
            "p99": round(_io.percentile(latencies, 99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "guards": guard_rows,
        "top_rationales": [{"rationale": k, "count": v} for k, v in rationales.most_common(top)],
    }


def run_summarize(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    records = load_decisions(args.log, stdin=getattr(args, "stdin", None))
    if not records:
        raise CLIError(f"{args.log}: no decision records")
    report = summarize(records, top=args.top)
    if args.json:
        _io.dump_json(out, report)
        return EXIT_OK

    _io.emit(out, f"{_io.human_count('decision', report['records'])} in {args.log}")
    actions = ", ".join(f"{a}={report['by_action'].get(a, 0)}" for a in _ACTIONS)
    _io.emit(out, f"  actions: {actions} (block rate {report['block_rate']:.2%})")
    sev = ", ".join(f"{s}={report['by_severity'].get(s, 0)}" for s in _SEVERITIES)
    _io.emit(out, f"  severity: {sev}")
    lat = report["latency_ms"]
    _io.emit(
        out,
        f"  latency ms: p50={lat['p50']} p95={lat['p95']} p99={lat['p99']} max={lat['max']}",
    )
    _io.emit(out)
    _io.emit(
        out,
        _io.render_table(
            ["guard", "decisions", "blocks", "block_rate", "p95_ms"],
            (
                [g["guard"], g["decisions"], g["blocks"], f"{g['block_rate']:.2%}", g["p95_latency_ms"]]
                for g in report["guards"]
            ),
        ),
    )
    if report["top_rationales"]:
        _io.emit(out)
        _io.emit(
            out,
            _io.render_table(
                ["count", "rationale"],
                ([r["count"], r["rationale"]] for r in report["top_rationales"]),
                max_width=80,
            ),
        )
    return EXIT_OK


# ---- filter ----------------------------------------------------------------


def matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.action and record["action"] not in args.action:
        return False
    if args.severity and str(record.get("severity")) not in args.severity:
        return False
    if args.guard and str(record.get("guard_name")) not in args.guard:
        return False
    if args.tier and int(_num(record.get("tier"), -1)) not in args.tier:
        return False
    conf = _num(record.get("confidence"))
    if args.min_confidence is not None and conf < args.min_confidence:
        return False
    if args.max_confidence is not None and conf > args.max_confidence:
        return False
    if args.min_latency_ms is not None and _num(record.get("latency_ms")) < args.min_latency_ms:
        return False
    return not (args.contains and args.contains.lower() not in str(record.get("rationale", "")).lower())


def run_filter(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    records = load_decisions(args.log, stdin=getattr(args, "stdin", None))
    kept = 0
    for record in records:
        if not matches(record, args):
            continue
        kept += 1
        if not args.count:
            _io.dump_json(out, record, indent=None)
        if args.limit is not None and kept >= args.limit:
            break
    if args.count:
        _io.emit(out, str(kept))
    elif not args.quiet:
        _io.emit(err, f"{kept} of {len(records)} records matched")
    return EXIT_OK
