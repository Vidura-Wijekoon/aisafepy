"""Example: detect an agent stuck in a tool-call loop."""

from __future__ import annotations

from aisafepy.core.progress import ProgressTracker, Step


def main() -> None:
    tracker = ProgressTracker(duplicate_threshold=3, window=10)
    plan = [
        Step(tool="search", args={"q": "weather"}),
        Step(tool="search", args={"q": "weather"}),
        Step(tool="search", args={"q": "weather"}),
        Step(tool="read", args={"id": 1}),
    ]
    for i, step in enumerate(plan, 1):
        decision = tracker.observe(step)
        if decision is not None:
            print(f"step {i}: LOOP DETECTED")
            print(f"  reason: {decision.evidence['reason']}")
            print(f"  tool/args: {decision.evidence['tool']} / {decision.evidence['args']}")
            return
        print(f"step {i}: OK")


if __name__ == "__main__":
    main()
