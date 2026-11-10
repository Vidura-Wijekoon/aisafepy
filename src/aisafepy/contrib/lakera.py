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
    threshold: float = 0.5
    """Score at or above which a *numeric* category value counts as a hit.

    Lakera's v2 API reports ``categories`` as booleans and puts the
    floats in ``category_scores``; older responses and some proxies
    inline the floats under ``categories``. Both shapes are handled."""

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
        first = (data.get("results") or [{}])[0]
        flagged = bool(first.get("flagged", False))
        categories: dict[str, object] = first.get("categories") or {}
        scores: dict[str, float] = first.get("category_scores") or {}
        breakers_hit = [c for c in self.breakers if self._is_hit(categories.get(c), scores.get(c))]
        if flagged or breakers_hit:
            return GuardDecision.block(
                tier=self.tier,
                confidence=self._confidence(categories, scores),
                rationale=f"Lakera flagged: {breakers_hit or _true_categories(categories)}",
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

    # ---- helpers -------------------------------------------------------

    def _is_hit(self, flag: object, score: float | None) -> bool:
        """Decide whether one category counts as a breaker hit.

        Boolean flags are authoritative. When the API inlines a float
        under ``categories`` (or only ``category_scores`` is present) the
        score is compared against ``threshold`` so that a 0.01 does not
        trip the guard the way bare truthiness would.
        """
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, (int, float)):
            return float(flag) >= self.threshold
        if score is not None:
            return float(score) >= self.threshold
        return False

    def _confidence(self, categories: dict[str, object], scores: dict[str, float]) -> float:
        numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
        if not numeric:
            numeric = [
                float(v)
                for v in categories.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
        if not numeric:
            return 0.9
        return max(0.0, min(1.0, max(numeric)))


def _true_categories(categories: dict[str, object]) -> list[str]:
    return [k for k, v in categories.items() if v is True]
