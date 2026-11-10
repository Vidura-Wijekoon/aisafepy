"""Budgets, rate limits, loop detection and decision sinks around a toy agent loop.

Everything here is from ``aisafepy.core`` and runs offline. The "agent"
is a scripted sequence of tool calls that drifts into a loop; the
budgets and the progress tracker stop it, and every decision lands in a
JSONL sink. Run with::

    python examples/core_budgets_and_loops.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aisafepy.core import (
    BudgetExceeded,
    BudgetSet,
    DollarBudget,
    InMemorySink,
    IterationBudget,
    JSONLSink,
    MultiSink,
    ProgressTracker,
    RateBudget,
    Step,
    TokenBudget,
    TokenPricing,
    budgeted,
)

SCRIPT = [
    ("search_docs", {"q": "refund policy"}, "Refunds within 30 days."),
    ("open_ticket", {"id": 4411}, "Ticket 4411: customer asks about refund."),
    ("search_docs", {"q": "refund policy"}, "Refunds within 30 days."),
    ("open_ticket", {"id": 4411}, "Ticket 4411: customer asks about refund."),
    ("search_docs", {"q": "refund policy"}, "Refunds within 30 days."),
    ("open_ticket", {"id": 4411}, "Ticket 4411: customer asks about refund."),
    ("search_docs", {"q": "refund policy"}, "Refunds within 30 days."),
]


def main() -> None:
    pricing = TokenPricing()
    pricing.register("demo-model", 1.0, 3.0)  # USD per 1M prompt / completion tokens

    steps = IterationBudget("steps", limit=10)
    tokens = TokenBudget("tokens", limit=5_000)
    spend = DollarBudget("spend", limit=0.01)
    budgets = BudgetSet("agent-run", [steps, tokens, spend])
    rate = RateBudget("tool-calls", max_events=5, per_seconds=60)
    tracker = ProgressTracker(window=8, duplicate_threshold=3, stagnation_window=4)

    with tempfile.TemporaryDirectory() as tmp:
        memory = InMemorySink()
        log_path = Path(tmp) / "decisions.jsonl"
        sink = MultiSink([memory, JSONLSink(log_path)])

        @budgeted(steps)
        def call_tool(name: str, args: dict, observation: str) -> str:
            tokens.consume(400)
            spend.charge("demo-model", 300, 100, pricing=pricing)
            return observation

        for i, (tool, args, observation) in enumerate(SCRIPT, 1):
            if not rate.try_acquire():
                print(f"step {i}: rate limit hit, retry after {rate.retry_after():.0f}s")
                break
            try:
                result = call_tool(tool, args, observation)
            except BudgetExceeded as exc:
                print(f"step {i}: budget exceeded: {exc}")
                break
            loop = tracker.observe(Step(tool=tool, args=args, observation=result))
            print(f"step {i}: {tool}({args}) -> {result[:32]!r}")
            if loop is not None:
                sink.emit(loop, {"step": i})
                print(f"  halted: {loop.rationale}")
                break

        print()
        for snap in budgets.snapshots().values():
            print(f"budget {snap.name:<7} {snap.used:>10.4f} / {snap.limit:<10g} ({snap.fraction:.0%})")
        print(f"decisions captured in memory: {len(memory.actions())}")
        print(f"jsonl bytes written: {log_path.stat().st_size}")


if __name__ == "__main__":
    main()
