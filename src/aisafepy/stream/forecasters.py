"""Streaming forecasters (Tier 2 / Tier 3).

The interesting research result is that detection should *forecast*
the harmfulness of the most-likely continuation from a *prefix*, not
wait for the full sequence. Three concrete recipes are wrapped here:

* :class:`StreamGuardForecast`. MC-rollout-supervised classifier
  (StreamGuard, arXiv 2604.03962). At training time, completions are
  sampled and scored; the classifier learns to predict the forecasted
  unsafe probability from a prefix.

* :class:`SCMForecast`. Token-level FineHarm-supervised classifier
  with dual heads (SCM, arXiv 2506.09996). The current-token head and
  the early-stop head provide F1 comparable to full detection at
  ~18% of generated tokens.

* :class:`KelpForecast`. The cheapest option (~20M params,
  <0.5 ms/token; Kelp, arXiv 2510.09694) for cases where Tier-2
  latency must be tiny.

All three classes share the same protocol: they expose a Tier-2 (or
Tier-3) Guard that takes a streaming buffer and returns a
``GuardDecision`` whose ``confidence`` reflects the forecast.

When the underlying pre-trained model is not available, the
forecaster falls back to a "no signal" allow with a clear evidence
flag so the cascade keeps working.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class _BaseForecaster:
    name: str
    model_id: str | None = None
    threshold: float = 0.5
    tier: int = 2
    on_time_target: float = 0.90
    """Target fraction of "on-time" detections (caught before the
    harmful tokens are emitted). Used at training time."""
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None or self.model_id is None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError:  # pragma: no cover - depends on env
            return
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device).eval()

    async def __call__(self, ctx: Context) -> GuardDecision:
        return await asyncio.to_thread(self._call_sync, ctx)

    def _call_sync(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        self._load()
        if self._model is None:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: no forecaster model loaded; allowing",
                guard_name=self.name,
                evidence={"no_model": True},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        text = ctx.buffer or ctx.chunk or ""
        torch = self._torch
        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        unsafe_prob = float(probs[-1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        if unsafe_prob >= self.threshold:
            return GuardDecision.block(
                tier=self.tier,
                confidence=unsafe_prob,
                rationale=f"{self.name}: forecast P(unsafe)={unsafe_prob:.3f}",
                guard_name=self.name,
                evidence={"forecast": "unsafe", "prob": unsafe_prob, "model": self.model_id},
                severity="high",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - unsafe_prob,
            rationale=f"{self.name}: forecast P(unsafe)={unsafe_prob:.3f}",
            guard_name=self.name,
            evidence={"forecast": "safe", "prob": unsafe_prob},
            latency_ms=latency_ms,
        )


@dataclass
class StreamGuardForecast(_BaseForecaster):
    """MC-rollout-trained forecaster (StreamGuard, arXiv 2604.03962)."""


@dataclass
class SCMForecast(_BaseForecaster):
    """Dual-head FineHarm forecaster (SCM, arXiv 2506.09996)."""


@dataclass
class KelpForecast(_BaseForecaster):
    """The 20M-param "Kelp" forecaster (arXiv 2510.09694)."""
