"""Tier-2 / Tier-3 classifier wrappers.

The :class:`HFClassifierGuard` wraps any Hugging Face sequence
classification model behind the Guard protocol. KV-cache reuse is
left to the user for now (it requires custom forward passes for each
backbone); the wrapper batches at the request boundary and supports
both binary "safe/unsafe" heads and multi-label heads.

When ``transformers`` / ``torch`` are not installed the class can
still be imported (the import is lazy) but instantiation will raise
``ModuleNotFoundError``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class HFClassifierGuard:
    """A Hugging Face classifier as a Guard.

    Parameters
    ----------
    name:
        Identifier surfaced in OTel spans.
    model_id:
        HF repo id, e.g. ``"meta-llama/Llama-Prompt-Guard-2-22M"``.
    threshold:
        Probability threshold on the "unsafe" class above which the
        guard returns ``BLOCK``.
    unsafe_label:
        Index or string label of the unsafe class. Default 1.
    device:
        Torch device. ``None`` chooses ``cuda`` if available, else
        ``cpu``.
    tier:
        Cascade tier; defaults to 2.
    """

    name: str
    model_id: str
    threshold: float = 0.5
    unsafe_label: int | str = 1
    device: str | None = None
    tier: int = 2
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_hf(cls, model_id: str, **kwargs: Any) -> HFClassifierGuard:
        return cls(name=kwargs.pop("name", model_id), model_id=model_id, **kwargs)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "HFClassifierGuard requires `pip install aisafepy[stream]`"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()
        self.device = device

    async def __call__(self, ctx: Context) -> GuardDecision:
        return await asyncio.to_thread(self._call_sync, ctx)

    def _call_sync(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        try:
            self._load()
        except ModuleNotFoundError as exc:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="classifier unavailable; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        text = ctx.buffer or ctx.chunk or ""
        if not text:
            return GuardDecision.allow(tier=self.tier, guard_name=self.name)

        with self._torch.no_grad():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1).cpu().numpy()[0]

        if isinstance(self.unsafe_label, str):
            id2label = getattr(self._model.config, "id2label", {})
            label2id = {v: int(k) for k, v in id2label.items()} if id2label else {}
            unsafe_idx = label2id.get(self.unsafe_label, 1)
        else:
            unsafe_idx = int(self.unsafe_label)

        unsafe_prob = float(probs[unsafe_idx]) if unsafe_idx < len(probs) else 0.0
        latency_ms = (time.perf_counter() - start) * 1000.0

        if unsafe_prob >= self.threshold:
            return GuardDecision.block(
                tier=self.tier,
                confidence=unsafe_prob,
                rationale=f"{self.name}: P(unsafe)={unsafe_prob:.3f} >= {self.threshold}",
                guard_name=self.name,
                evidence={
                    "model": self.model_id,
                    "probs": probs.tolist(),
                    "threshold": self.threshold,
                },
                severity="high" if unsafe_prob > 0.9 else "medium",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - unsafe_prob,
            rationale=f"{self.name}: P(unsafe)={unsafe_prob:.3f} < {self.threshold}",
            guard_name=self.name,
            evidence={"model": self.model_id, "probs": probs.tolist()},
            latency_ms=latency_ms,
        )


@dataclass
class StubClassifierGuard:
    """Deterministic stub classifier for tests and CI.

    Returns ``BLOCK`` when the buffer contains any of the configured
    triggers, otherwise ``ALLOW``. Used in unit tests so they don't
    need to download HF models.
    """

    name: str = "stub"
    triggers: tuple[str, ...] = ("HARMFUL", "[BLOCK]")
    tier: int = 2

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = (ctx.buffer or ctx.chunk or "").upper()
        for trig in self.triggers:
            if trig.upper() in text:
                return GuardDecision.block(
                    tier=self.tier,
                    confidence=0.99,
                    rationale=f"stub trigger matched: {trig}",
                    guard_name=self.name,
                    evidence={"trigger": trig},
                )
        return GuardDecision.allow(
            tier=self.tier,
            rationale="stub: no trigger",
            guard_name=self.name,
        )
