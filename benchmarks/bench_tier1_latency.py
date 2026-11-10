"""Latency of the Tier-1 deterministic guards.

Measures per-call latency for the PII regex set, a 1000-term
Aho-Corasick blocklist, and a three-guard :class:`GuardPipeline`, over a
corpus of synthetic texts of varied length including one 50 KB
pathological input. Reports p50 / p95 / p99 in milliseconds and
throughput.

Run::

    python benchmarks/bench_tier1_latency.py
    python benchmarks/bench_tier1_latency.py --json
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Timing, base_arg_parser, print_report, time_guard

_WORDS = ["invoice", "report", "meeting", "quarterly", "budget", "customer", "ticket", "deploy", "release", "pipeline", "latency", "summary", "approval", "vendor", "contract", "schedule", "reminder", "note"]


def synth_texts(n: int, *, seed: int = 7) -> list[str]:
    """Build ``n`` texts of varied length; the last is a 50 KB stressor."""
    rng = random.Random(seed)
    texts: list[str] = []
    for _ in range(max(0, n - 1)):
        length = rng.choice([8, 20, 50, 120, 400])
        texts.append(" ".join(rng.choice(_WORDS) for _ in range(length)))
    texts.append(" ".join(rng.choice(_WORDS) for _ in range(9000))[:50_000])
    return texts


def build_blocklist_terms(n: int, *, seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    terms = {f"forbidden_term_{i}_{rng.randint(1000, 9999)}" for i in range(n)}
    return sorted(terms)


def run(quick: bool) -> list[Timing]:
    from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
    from aisafepy.stream.pipeline import Context, GuardPipeline

    n_texts = 40 if quick else 1000
    n_terms = 50 if quick else 1000
    repeats = 1

    texts = synth_texts(n_texts)
    pii = RegexGuard.compile_pii()
    blocklist = AhoCorasickGuard(name="blocklist", terms=build_blocklist_terms(n_terms))
    custom = RegexGuard.from_patterns(
        "secrets", {"aws": r"AKIA[0-9A-Z]{16}", "bearer": r"Bearer\s+[A-Za-z0-9._-]+"}
    )

    results = [
        time_guard(pii, texts, repeats=repeats, label="RegexGuard.compile_pii"),
        time_guard(blocklist, texts, repeats=repeats, label=f"AhoCorasick[{n_terms}]"),
    ]

    pipeline = GuardPipeline(tier1=[pii, blocklist, custom])

    async def _pipe(ctx: Context):
        return await pipeline.evaluate_chunk(ctx)

    _pipe.name = "GuardPipeline[3 guards]"  # type: ignore[attr-defined]
    results.append(time_guard(_pipe, texts, repeats=repeats, label="GuardPipeline[3 guards]"))
    return results


def main(argv: list[str] | None = None) -> int:
    args = base_arg_parser("Tier-1 guard latency benchmark").parse_args(argv)
    results = run(args.quick)
    print_report(
        "Tier-1 latency (lower is better)",
        ["guard", "n", "p50_ms", "p95_ms", "p99_ms", "throughput/s"],
        [
            [t.label, t.count, f"{t.p50_ms:.3f}", f"{t.p95_ms:.3f}", f"{t.p99_ms:.3f}", f"{t.throughput_per_s:.0f}"]
            for t in results
        ],
        as_json=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
