"""Example: a three-tier streaming guard pipeline.

We compose a Tier-1 regex guard, a Tier-2 stub classifier, and run a
simulated streaming response through it. Replace
``StubClassifierGuard`` with ``ClassifierGuard.from_hf(...)`` for a
real HF model.
"""

from __future__ import annotations

import asyncio

from aisafepy.stream import GuardPipeline
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard


async def main() -> None:
    pipeline = GuardPipeline(
        tier1=[
            RegexGuard.compile_pii(redact=True),
            AhoCorasickGuard(name="secret-blocklist",
                             terms=["api_key=", "BEGIN PRIVATE"]),
        ],
        tier2=[
            StubClassifierGuard(triggers=("HARMFUL", "[BLOCK]")),
        ],
        budget_ms_p95=80,
    )

    chunks = [
        "Sure, I can help with that. ",
        "Reach me at alice@example.com for more details. ",
        "Now here's the [HARMFUL] section: ...",
    ]

    print("=== streaming output ===")
    async for piece in pipeline.guard_stream(chunks, prompt="say hello"):
        if hasattr(piece, "action"):
            print(f"\n[GUARD DECISION] tier={piece.tier} action={piece.action.value}")
            print(f"  rationale: {piece.rationale}")
            print(f"  fallback: {piece.fallback}")
            break
        print(piece, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
