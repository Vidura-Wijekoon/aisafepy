"""Azure AI Content Safety wrapper.

Wraps a ``ContentSafetyClient`` (``azure-ai-contentsafety``) behind the
Guard protocol. The client is duck-typed: anything exposing
``analyze_text(options)`` returning an object with
``categories_analysis`` (a list of ``{category, severity}``) works, so
tests use a fake and the SDK is never imported here.

Azure reports a severity per category (Hate, SelfHarm, Sexual,
Violence) on a 0-7 scale (0-6 in four-level mode). The guard BLOCKs
when any category's severity reaches its configured threshold.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context

DEFAULT_SEVERITY_THRESHOLDS: dict[str, int] = {
    "Hate": 4,
    "SelfHarm": 4,
    "Sexual": 4,
    "Violence": 4,
}

_MAX_SEVERITY = 7


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class AzureContentSafetyGuard:
    """Azure AI Content Safety text analysis as a Tier-2 guard."""

    client: Any
    name: str = "azure-content-safety"
    tier: int = 2
    timeout_s: float = 5.0
    thresholds: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SEVERITY_THRESHOLDS))
    """Per-category severity (0-7) at or above which the guard blocks."""
    max_chars: int = 10_000
    """Azure caps text at 10k characters per request; longer inputs are
    truncated from the front so the most recent output is analysed."""

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = ctx.buffer or ctx.chunk or ""
        if not text.strip():
            return GuardDecision.allow(tier=self.tier, guard_name=self.name, rationale="empty")
        if len(text) > self.max_chars:
            text = text[-self.max_chars :]
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(self._analyze(text), timeout=self.timeout_s)
        except Exception as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: call failed ({exc!r}); failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        analysis = _get(response, "categories_analysis") or _get(response, "categoriesAnalysis") or []
        severities: dict[str, int] = {}
        for item in analysis:
            category = _get(item, "category")
            severity = _get(item, "severity")
            if category is None or severity is None:
                continue
            try:
                severities[str(category)] = int(severity)
            except (TypeError, ValueError):
                continue

        hits = {
            cat: sev
            for cat, sev in severities.items()
            if cat in self.thresholds and sev >= self.thresholds[cat]
        }
        evidence = {"severities": severities, "thresholds": dict(self.thresholds), "hits": hits}
        if hits:
            worst = max(hits.values())
            return GuardDecision.block(
                tier=self.tier,
                confidence=min(1.0, worst / _MAX_SEVERITY),
                rationale=f"{self.name}: severity threshold reached for {sorted(hits)}",
                guard_name=self.name,
                evidence=evidence,
                severity="critical" if worst >= 6 else "high",
                latency_ms=latency_ms,
            )
        top = max(severities.values(), default=0)
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - min(1.0, top / _MAX_SEVERITY),
            rationale=f"{self.name}: below thresholds",
            guard_name=self.name,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    async def _analyze(self, text: str) -> Any:
        analyze = getattr(self.client, "analyze_text", None)
        if analyze is None:
            raise TypeError("client must expose analyze_text(options)")
        options: Any
        try:
            from azure.ai.contentsafety.models import (  # type: ignore[import-not-found]
                AnalyzeTextOptions,
            )

            options = AnalyzeTextOptions(text=text)
        except Exception:
            options = {"text": text}
        result = analyze(options)
        if inspect.isawaitable(result):
            return await result
        return result
