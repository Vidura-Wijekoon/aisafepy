"""Protect AI llm-guard scanner wrapper.

Adapts any llm-guard input or output scanner to the Guard protocol.
The llm-guard package is optional; without it the guard imports fine
but raises on instantiation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class LLMGuardOutputScanner:
    """Wrap a single llm-guard *output* scanner.

    ``scanner_cls_path`` is a dotted path such as
    ``"llm_guard.output_scanners.NoRefusal"``.
    """

    name: str
    scanner_cls_path: str
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    tier: int = 2
    _scanner: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._scanner is not None:
            return
        try:
            import importlib

            module_path, _, cls_name = self.scanner_cls_path.rpartition(".")
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            self._scanner = cls(**self.init_kwargs)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "LLMGuardOutputScanner requires `pip install aisafepy[contrib-llm-guard]`."
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
                rationale="llm-guard unavailable; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )
        text = ctx.buffer or ctx.chunk or ""
        # llm-guard's scanners return (sanitized_text, is_valid, risk_score)
        try:
            sanitized, is_valid, score = self._scanner.scan(ctx.prompt or "", text)
        except Exception as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"llm-guard scan raised; failing open: {exc!r}",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not is_valid:
            return GuardDecision.block(
                tier=self.tier,
                confidence=float(score),
                rationale=f"llm-guard scanner {self.scanner_cls_path} flagged content",
                guard_name=self.name,
                evidence={"sanitized": sanitized, "score": float(score)},
                severity="high",
                latency_ms=latency_ms,
            )
        if sanitized and sanitized != text:
            return GuardDecision.transform(
                tier=self.tier,
                confidence=float(score),
                rationale=f"llm-guard scanner {self.scanner_cls_path} sanitized content",
                transformed_content=sanitized,
                guard_name=self.name,
                evidence={"score": float(score)},
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale=f"llm-guard scanner {self.scanner_cls_path} clean",
            guard_name=self.name,
            latency_ms=latency_ms,
        )
