"""OpenAI Moderation API wrapper.

Wraps ``client.moderations.create(model=..., input=...)`` from the
official ``openai`` SDK (sync ``OpenAI`` or async ``AsyncOpenAI``)
behind the Guard protocol. The client is duck-typed so tests use a
fake and no network is touched at import time.

The moderation endpoint returns per-category booleans and scores. The
guard BLOCKs when any category is flagged by the API or when a
category score exceeds the per-category threshold you configure;
otherwise it ALLOWs with the scores attached as evidence. Transport
errors fail open with evidence, matching the rest of ``contrib``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:  # pragma: no cover - defensive
                pass
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {}


@dataclass
class OpenAIModerationGuard:
    """OpenAI Moderation endpoint as a Tier-2 guard."""

    client: Any
    name: str = "openai-moderation"
    model: str = "omni-moderation-latest"
    tier: int = 2
    timeout_s: float = 5.0
    thresholds: dict[str, float] = field(default_factory=dict)
    """Per-category score thresholds (e.g. ``{"self-harm": 0.2}``).
    Categories absent from the mapping rely on the API's own flag."""
    respect_api_flags: bool = True
    include_prompt: bool = False
    """When True, moderate ``prompt + "\\n" + content`` rather than the
    content alone."""

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = ctx.buffer or ctx.chunk or ""
        if not text.strip():
            return GuardDecision.allow(tier=self.tier, guard_name=self.name, rationale="empty")
        if self.include_prompt and ctx.prompt:
            text = f"{ctx.prompt}\n{text}"
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(self._moderate(text), timeout=self.timeout_s)
        except Exception as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: call failed ({exc!r}); failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        data = _as_dict(response)
        results = data.get("results") or []
        result = _as_dict(results[0]) if results else {}
        flagged = bool(result.get("flagged", False))
        categories = {k: bool(v) for k, v in _as_dict(result.get("categories")).items()}
        scores = {k: float(v) for k, v in _as_dict(result.get("category_scores")).items() if v is not None}

        api_hits = [k for k, v in categories.items() if v] if self.respect_api_flags else []
        threshold_hits = [
            k for k, thr in self.thresholds.items() if scores.get(k, 0.0) >= thr
        ]
        hits = sorted(set(api_hits) | set(threshold_hits))
        evidence = {
            "model": self.model,
            "flagged": flagged,
            "categories": sorted(api_hits),
            "threshold_hits": sorted(threshold_hits),
            "scores": scores,
        }
        if hits or (flagged and self.respect_api_flags):
            confidence = max([scores.get(k, 0.0) for k in hits] or [0.9])
            return GuardDecision.block(
                tier=self.tier,
                confidence=min(1.0, max(0.0, confidence)),
                rationale=f"{self.name}: flagged {hits or ['unspecified']}",
                guard_name=self.name,
                evidence=evidence,
                severity="high" if confidence >= 0.8 else "medium",
                latency_ms=latency_ms,
            )
        top = max(scores.values(), default=0.0)
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - min(1.0, top),
            rationale=f"{self.name}: clean",
            guard_name=self.name,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    async def _moderate(self, text: str) -> Any:
        moderations = getattr(self.client, "moderations", None)
        create = getattr(moderations, "create", None) if moderations is not None else None
        if create is None:
            raise TypeError("client must expose moderations.create(model=..., input=...)")
        result = create(model=self.model, input=text)
        if inspect.isawaitable(result):
            return await result
        return result
