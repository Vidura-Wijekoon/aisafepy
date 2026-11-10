"""Throughput of the core budget primitives.

Measures ``Budget.consume`` calls per second single-threaded and across
several threads. Budgets sit on the hot path of every agent step, so
their lock contention under concurrency is worth knowing.

Run::

    python benchmarks/bench_budgets.py
    python benchmarks/bench_budgets.py --json
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import base_arg_parser, print_report


def single_thread(n: int) -> float:
    from aisafepy.core.budgets import TokenBudget

    budget = TokenBudget("single", limit=float(n + 1))
    t0 = time.perf_counter()
    for _ in range(n):
        budget.consume(1.0)
    elapsed = time.perf_counter() - t0
    return n / elapsed if elapsed else 0.0


def multi_thread(n: int, threads: int) -> float:
    from aisafepy.core.budgets import TokenBudget

    budget = TokenBudget("shared", limit=float(n + threads + 1))
    per = n // threads
    barrier = threading.Barrier(threads + 1)

    def worker() -> None:
        barrier.wait()
        for _ in range(per):
            budget.consume(1.0)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    barrier.wait()
    t0 = time.perf_counter()
    for w in workers:
        w.join()
    elapsed = time.perf_counter() - t0
    total = per * threads
    return total / elapsed if elapsed else 0.0


def run(quick: bool):
    n = 20_000 if quick else 500_000
    rows = [("single thread", 1, single_thread(n))]
    for threads in (2, 4, 8):
        rows.append((f"{threads} threads", threads, multi_thread(n, threads)))
    return n, rows


def main(argv: list[str] | None = None) -> int:
    args = base_arg_parser("budget consume() throughput benchmark").parse_args(argv)
    n, rows = run(args.quick)
    print_report(
        f"Budget.consume throughput (n={n})",
        ["scenario", "threads", "consume_per_s"],
        [[label, threads, f"{rate:,.0f}"] for label, threads, rate in rows],
        as_json=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
