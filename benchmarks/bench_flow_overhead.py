"""Per-call overhead of the flow IFC interpreter.

Compares a bare function call against :func:`evaluate_call` with 0, 1,
and 5 ``deny_if`` rules, reporting microseconds per tool call. This is
the number to cite when someone asks "what does wrapping my agent in IFC
cost?" for the common case where the call is allowed.

Run::

    python benchmarks/bench_flow_overhead.py
    python benchmarks/bench_flow_overhead.py --json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import base_arg_parser, print_report


def _bare(to, body) -> str:
    return f"{to}:{body}"


def build_policy(n_deny: int):
    from aisafepy.flow import Capability, Policy

    policy = (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .require("send_email", control_flow_integrity="TRUSTED", caps={Capability.WRITE_EXTERNAL})
    )
    for i in range(n_deny):
        policy = policy.deny_if(
            "send_email",
            when=lambda *a, _i=i, **k: f"marker-{_i}" in str(a),
            reason=f"deny-{i}",
        )
    return policy


def measure(iterations: int, n_deny: int) -> float:
    """Return mean microseconds per allowed ``evaluate_call``."""
    from aisafepy.flow.interpreter import IFCContext, evaluate_call
    from aisafepy.flow.taint import lift

    policy = build_policy(n_deny)
    ctx = IFCContext(control_flow_integrity="TRUSTED")
    to = lift("alice@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hello", source="user_prompt", integrity="TRUSTED", capabilities=("write.external",))

    t0 = time.perf_counter()
    for _ in range(iterations):
        evaluate_call(
            tool="send_email", args=(to, body), kwargs={}, policy=policy, context=ctx
        )
    return (time.perf_counter() - t0) / iterations * 1e6


def measure_bare(iterations: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        _bare("alice@example.com", "hello")
    return (time.perf_counter() - t0) / iterations * 1e6


def run(quick: bool):
    iterations = 2000 if quick else 30000
    bare = measure_bare(iterations)
    rows = [("bare call (no IFC)", bare, 0.0)]
    for n in (0, 1, 5):
        us = measure(iterations, n)
        rows.append((f"evaluate_call, {n} deny rules", us, us - bare))
    return iterations, rows


def main(argv: list[str] | None = None) -> int:
    args = base_arg_parser("flow IFC per-call overhead benchmark").parse_args(argv)
    iterations, rows = run(args.quick)
    print_report(
        f"Flow IFC overhead over {iterations} iterations (microseconds per call)",
        ["scenario", "us_per_call", "overhead_us"],
        [[label, f"{us:.2f}", f"{ov:.2f}"] for label, us, ov in rows],
        as_json=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
