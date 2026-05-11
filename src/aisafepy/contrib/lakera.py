"""Lakera Guard HTTP API wrapper.

Lakera Guard is a hosted classifier API. We call the public endpoint
via ``httpx`` (loaded lazily). The ``api_key`` must be provided
either at construction time or via the ``LAKERA_GUARD_API_KEY``
environment variable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class LakeraGuard:
    name: str = "lakera-guard"
    endpoint: str = "https://api.lakera.ai/v2/guard"
    api_key: str | None = None
    tier: int = 2
    timeout_s: float = 5.0
    breakers: tuple[str, ...] = ("prompt_attack",)
    """Lakera category names whose hits should produce a BLOCK."""

    async def __call__(self, ctx: Context) -> GuardDecision:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            return GuardDecision.allow(
                tier=self.tier,
                rationale="httpx not installed; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )

        api_key = self.api_key or os.environ.get("LAKERA_GUARD_API_KEY")
        if not api_key:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="LAKERA_GUARD_API_KEY not set; failing open",
                guard_name=self.name,
                evidence={"missing_key": True},
            )

        text = ctx.buffer or ctx.chunk or ""
        payload = {
            "messages": [{"role": "user", "content": ctx.prompt or ""},
                         {"role": "assistant", "content": text}]
        }
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"Lakera call failed: {exc!r}; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        flagged = data.get("results", [{}])[0].get("flagged", False)
        categories = data.get("results", [{}])[0].get("categories", {})
        breakers_hit = [c for c in self.breakers if categories.get(c)]
        if flagged or breakers_hit:
            return GuardDecision.block(
                tier=self.tier,
                confidence=float(max(categories.values(), default=0.9)) if categories else 0.9,
                rationale=f"Lakera flagged: {breakers_hit or list(categories.keys())}",
                guard_name=self.name,
                evidence={"data": data},
                severity="high",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale="Lakera: clean",
            guard_name=self.name,
            evidence={"data": data},
            latency_ms=latency_ms,
        )
