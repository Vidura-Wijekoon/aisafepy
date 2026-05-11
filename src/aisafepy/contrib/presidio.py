"""Microsoft Presidio PII detection wrapper."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class PresidioPIIGuard:
    """Wrap Presidio's ``AnalyzerEngine`` as a Guard.

    Supports both detection-only mode (returns BLOCK with the entity
    list as evidence) and redaction mode (returns TRANSFORM with the
    masked content).
    """

    name: str = "presidio-pii"
    language: str = "en"
    entities: tuple[str, ...] = (
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "US_SSN",
        "PERSON",
    )
    redact: bool = True
    tier: int = 1
    _analyzer: Any = field(default=None, init=False, repr=False)
    _anonymizer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            self._analyzer = AnalyzerEngine()
            try:
                from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]
                self._anonymizer = AnonymizerEngine()
            except ImportError:
                self._anonymizer = None
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "PresidioPIIGuard requires `pip install aisafepy[contrib-presidio]`."
            ) from exc

    async def __call__(self, ctx: Context) -> GuardDecision:
        return await asyncio.to_thread(self._call_sync, ctx)

    def _call_sync(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        try:
            self._load()
        except ModuleNotFoundError as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="presidio unavailable; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )
        text = ctx.buffer or ctx.chunk or ""
        analyzer_results = self._analyzer.analyze(
            text=text,
            entities=list(self.entities),
            language=self.language,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not analyzer_results:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no PII detected",
                guard_name=self.name,
                latency_ms=latency_ms,
            )
        evidence = {
            "entities": [
                {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
                for r in analyzer_results
            ]
        }
        if self.redact and self._anonymizer is not None:
            redacted = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
            return GuardDecision.transform(
                tier=self.tier,
                confidence=0.95,
                rationale=f"PII redacted: {[r.entity_type for r in analyzer_results]}",
                transformed_content=redacted.text,
                guard_name=self.name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=0.95,
            rationale=f"PII detected: {[r.entity_type for r in analyzer_results]}",
            guard_name=self.name,
            evidence=evidence,
            severity="high",
            latency_ms=latency_ms,
        )
