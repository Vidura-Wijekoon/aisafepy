"""``aisafepy flow``: IFC policies, plans, and audit records from the shell.

``lint``
    Load a YAML policy and run :func:`aisafepy.flow.lint.lint_policy`.
    Exit 2 on errors; ``--strict`` also fails on warnings. Pass the
    tool names your agent exposes with ``--tools`` to get coverage
    findings.
``show``
    Print the declarative form of a policy (sources, requirements,
    deny rules, declassifiers) as a table or JSON.
``plan``
    Validate a planner-emitted action plan against a policy, or execute
    it under IFC enforcement with tools imported from dotted paths and
    print the trace. Exit 2 when the plan is invalid or halts.
``audit``
    Summarise a JSONL file written by :class:`DecisionRecorder`:
    counts per tool and action, mediated approvals, top deny reasons.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "flow",
        help="lint, display, or exercise an IFC policy; summarise audit records",
        description="Work with aisafepy.flow policies, plans, and decision records.",
    )
    sub = p.add_subparsers(dest="flow_command", metavar="SUBCOMMAND")
    sub.required = True

    lint = sub.add_parser("lint", help="run the policy linter")
    lint.add_argument("policy", metavar="POLICY.yaml")
    lint.add_argument(
        "--tools",
        metavar="NAME[,NAME...]",
        help="tool names the agent exposes, for coverage findings",
    )
    lint.add_argument("--strict", action="store_true", help="exit 2 on warnings as well")
    lint.set_defaults(handler=run_lint)

    show = sub.add_parser("show", help="print the declarative form of a policy")
    show.add_argument("policy", metavar="POLICY.yaml")
    show.set_defaults(handler=run_show)

    plan = sub.add_parser("plan", help="validate or execute a plan under a policy")
    plan.add_argument("policy", metavar="POLICY.yaml")
    plan.add_argument("plan", metavar="PLAN.json", help="JSON list of steps (or {'steps': [...]})")
    plan.add_argument(
        "--tools",
        metavar="SPEC[,SPEC...]",
        help="tool callables as module:attr or name=module:attr; required unless --dry-run",
    )
    plan.add_argument("--dry-run", action="store_true", help="validate only, do not execute")
    plan.add_argument(
        "--cfi",
        choices=["TRUSTED", "UNTRUSTED", "QUARANTINED"],
        default="TRUSTED",
        help="starting control-flow integrity (default: TRUSTED)",
    )
    plan.add_argument(
        "--cfi-mode",
        choices=["control", "data"],
        default="control",
        help="how control-flow integrity is derived (see aisafepy.flow.dsl)",
    )
    plan.add_argument("--max-steps", type=int, default=64)
    plan.set_defaults(handler=run_plan_cmd)

    audit = sub.add_parser("audit", help="summarise a DecisionRecorder JSONL file")
    audit.add_argument("records", metavar="RECORDS.jsonl")
    audit.add_argument("--top", type=int, default=10, help="deny reasons to list (default: 10)")
    audit.set_defaults(handler=run_audit)


# ---- shared ---------------------------------------------------------------


def load_policy(path: str) -> Any:
    """Load a YAML policy, mapping loader failures to one-line errors."""
    from aisafepy.flow.serialization import SerializationError, policy_from_yaml

    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such policy file: {p}")
    try:
        return policy_from_yaml(p)
    except SerializationError as exc:
        raise CLIError(f"invalid policy {p}: {exc}") from exc
    except RuntimeError as exc:  # PyYAML missing
        raise CLIError(str(exc)) from exc
    except Exception as exc:
        raise CLIError(f"could not load policy {p}: {type(exc).__name__}: {exc}") from exc


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def import_tools(specs: list[str]) -> dict[str, Callable[..., Any]]:
    """Resolve ``module:attr`` or ``name=module:attr`` specs to callables."""
    tools: dict[str, Callable[..., Any]] = {}
    for spec in specs:
        alias, _, target = spec.rpartition("=") if "=" in spec else ("", "", spec)
        module_path, sep, attr = target.partition(":")
        if not sep or not module_path or not attr:
            raise CLIError(f"tool spec {spec!r} must look like module:attr or name=module:attr")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise CLIError(f"cannot import module {module_path!r} for tool {spec!r}: {exc}") from exc
        obj: Any = module
        for part in attr.split("."):
            if not hasattr(obj, part):
                raise CLIError(f"module {module_path!r} has no attribute {attr!r}")
            obj = getattr(obj, part)
        if not callable(obj):
            raise CLIError(f"{target} is not callable")
        name = alias or getattr(obj, "__name__", attr.rsplit(".", 1)[-1])
        if name in tools:
            raise CLIError(f"duplicate tool name {name!r}")
        tools[name] = obj
    return tools


def load_plan(path: str, *, stdin: IO[str] | None) -> list[dict[str, Any]]:
    data = _io.load_structured(path, stdin=stdin)
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        data = data["steps"]
    if not isinstance(data, list):
        raise CLIError(f"{path}: a plan must be a JSON list of steps")
    return data


# ---- lint -----------------------------------------------------------------


def run_lint(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.flow.lint import format_findings, has_errors, lint_policy

    policy = load_policy(args.policy)
    tools = _split_csv(args.tools) or None
    findings = lint_policy(policy, tools=tools)
    warnings = sum(1 for f in findings if f.severity == "warning")
    failed = has_errors(findings) or (args.strict and warnings > 0)

    if args.json:
        _io.dump_json(
            out,
            {
                "policy": args.policy,
                "ok": not failed,
                "strict": bool(args.strict),
                "findings": [
                    {
                        "code": f.code,
                        "severity": f.severity,
                        "subject": f.subject,
                        "message": f.message,
                    }
                    for f in findings
                ],
            },
        )
    else:
        _io.emit(out, format_findings(findings))
    return EXIT_FLAGGED if failed else EXIT_OK


# ---- show -----------------------------------------------------------------


def run_show(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.flow.serialization import policy_to_dict

    policy = load_policy(args.policy)
    data = policy_to_dict(policy)
    if args.json:
        _io.dump_json(out, {"policy": args.policy, **data})
        return EXIT_OK

    _io.emit(out, f"{args.policy}: mode={data.get('mode', policy.mode)}")
    sources = data.get("sources", {})
    if sources:
        _io.emit(out)
        _io.emit(out, "sources")
        rows = [
            [name, s.get("integrity"), ", ".join(s.get("capabilities", [])) or "-"]
            for name, s in sorted(sources.items())
        ]
        _io.emit(out, _io.render_table(["source", "integrity", "capabilities"], rows))
    requirements = data.get("requirements", {})
    if requirements:
        _io.emit(out)
        _io.emit(out, "requirements")
        rows = [
            [
                name,
                r.get("control_flow_integrity"),
                r.get("arg_max_integrity"),
                ", ".join(r.get("capabilities", [])) or "-",
            ]
            for name, r in sorted(requirements.items())
        ]
        _io.emit(
            out,
            _io.render_table(["tool", "control_flow", "arg_max_integrity", "capabilities"], rows),
        )
    deny_rules = data.get("deny_rules", [])
    if deny_rules:
        _io.emit(out)
        _io.emit(out, "deny rules")
        rows = [
            [
                r.get("tool"),
                r.get("reason"),
                json.dumps(r["when"], sort_keys=True) if r.get("when") else "(python callable)",
            ]
            for r in deny_rules
        ]
        _io.emit(out, _io.render_table(["tool", "reason", "when"], rows, max_width=70))
    declassifications = data.get("declassifications", {})
    if declassifications:
        _io.emit(out)
        _io.emit(out, "declassifiers")
        rows = [[name, d.get("by", "")] for name, d in sorted(declassifications.items())]
        _io.emit(out, _io.render_table(["tool", "justification"], rows, max_width=80))
    return EXIT_OK


# ---- plan -----------------------------------------------------------------


def run_plan_cmd(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.flow.dsl import PlanError, run_plan, validate_plan
    from aisafepy.flow.interpreter import IFCContext, IFCViolationError

    policy = load_policy(args.policy)
    plan = load_plan(args.plan, stdin=getattr(args, "stdin", None))
    specs = _split_csv(args.tools)

    if args.dry_run:
        tool_names: Any = list(import_tools(specs)) if specs else sorted(
            {str(step.get("call")) for step in plan if isinstance(step, dict)}
        )
        problems = validate_plan(plan, tool_names, max_steps=args.max_steps)
        if args.json:
            _io.dump_json(
                out,
                {"plan": args.plan, "ok": not problems, "steps": len(plan), "problems": problems},
            )
        elif problems:
            _io.emit(out, f"{args.plan}: {_io.human_count('problem', len(problems))}")
            for item in problems:
                _io.emit(out, f"  - {item}")
        elif not args.quiet:
            _io.emit(out, f"{args.plan}: {_io.human_count('step', len(plan))}, valid")
        return EXIT_FLAGGED if problems else EXIT_OK

    if not specs:
        raise CLIError("--tools is required to execute a plan (use --dry-run to validate only)")
    tools = import_tools(specs)
    context = IFCContext(control_flow_integrity=args.cfi)
    halted = False
    try:
        trace = run_plan(
            plan,
            tools=tools,
            policy=policy,
            context=context,
            max_steps=args.max_steps,
            cfi_mode=args.cfi_mode,
        )
    except IFCViolationError as exc:
        trace = getattr(exc, "trace", None)
        halted = True
        if trace is None:
            raise CLIError(f"plan denied: {exc.violation.reason}") from exc
    except PlanError as exc:
        raise CLIError(f"invalid plan: {exc}") from exc
    halted = halted or not trace.completed

    if args.json:
        _io.dump_json(out, {"plan": args.plan, "halted": halted, "trace": trace.to_dict()})
        return EXIT_FLAGGED if halted else EXIT_OK

    rows = []
    for step in trace.steps:
        detail = ""
        if step.violation is not None:
            detail = step.violation.reason
        elif step.decision is not None:
            detail = step.decision.rationale
        rows.append([step.index, step.tool, step.action, step.cfi, step.bound or "-", detail])
    _io.emit(out, _io.render_table(["#", "tool", "action", "cfi", "bound", "detail"], rows))
    if halted:
        reason = trace.violation.reason if trace.violation else "halted"
        _io.emit(out, f"halted at step {trace.halted_at}: {reason}")
    elif not args.quiet:
        _io.emit(out, f"completed {_io.human_count('step', len(trace.steps))}")
    return EXIT_FLAGGED if halted else EXIT_OK


# ---- audit ----------------------------------------------------------------


def run_audit(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    records = list(_io.iter_jsonl(args.records, stdin=getattr(args, "stdin", None)))
    by_action: Counter[str] = Counter()
    by_tool: dict[str, Counter[str]] = {}
    deny_reasons: Counter[str] = Counter()
    mediated = 0
    first_ts: float | None = None
    last_ts: float | None = None
    for rec in records:
        kind = rec.get("kind", "decision")
        key = "violation" if kind == "violation" else str(rec.get("action", "allow"))
        tool = str(rec.get("tool", "?"))
        by_action[key] += 1
        by_tool.setdefault(tool, Counter())[key] += 1
        if kind == "violation":
            deny_reasons[str(rec.get("reason", ""))] += 1
        if rec.get("mediated"):
            mediated += 1
        ts = rec.get("timestamp")
        if isinstance(ts, int | float):
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

    summary = {
        "records": args.records,
        "total": len(records),
        "by_action": dict(by_action),
        "by_tool": {t: dict(c) for t, c in sorted(by_tool.items())},
        "top_deny_reasons": [[r, n] for r, n in deny_reasons.most_common(args.top)],
        "mediated_approvals": mediated,
        "span_s": round(last_ts - first_ts, 3) if first_ts is not None and last_ts else 0.0,
    }
    if args.json:
        _io.dump_json(out, summary)
        return EXIT_OK

    _io.emit(out, f"{_io.human_count('record', len(records))} in {args.records}")
    _io.emit(out, "by action: " + (", ".join(f"{k}={v}" for k, v in sorted(by_action.items())) or "-"))
    if mediated:
        _io.emit(out, f"mediated approvals: {mediated}")
    if by_tool:
        _io.emit(out)
        actions = sorted(by_action)
        rows = [[tool, *[counts.get(a, 0) for a in actions]] for tool, counts in sorted(by_tool.items())]
        _io.emit(out, _io.render_table(["tool", *actions], rows))
    if deny_reasons:
        _io.emit(out)
        _io.emit(out, "top deny reasons")
        _io.emit(
            out,
            _io.render_table(
                ["count", "reason"], [[n, r] for r, n in deny_reasons.most_common(args.top)], max_width=90
            ),
        )
    return EXIT_OK
