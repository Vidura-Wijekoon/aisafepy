"""Meta Llama Guard 4 wrapper.

Llama Guard 4 is a generative classifier. It consumes the
conversation and emits ``unsafe`` / ``safe`` plus an MLCommons taxonomy
category. We wrap the HF model behind the standard Guard protocol.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


@dataclass
class LlamaGuard4:
    name: str = "llama-guard-4"
    model_id: str = "meta-llama/Llama-Guard-4-12B"
    tier: int = 3
    device: str | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "LlamaGuard4 requires `pip install aisafepy[stream]`."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device).eval()
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
                rationale="LlamaGuard4 unavailable; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
            )
        torch = self._torch
        text = ctx.buffer or ctx.chunk or ""
        prompt = (
            f"<|begin_of_text|>[INST] Task: Check if the following text is safe.\n\n"
            f"Text: {text}\n\n"
            f"Provide your assessment in the format 'safe' or 'unsafe' followed by the category. [/INST]"
        )
        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self._model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip().lower()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if decoded.startswith("unsafe"):
            return GuardDecision.block(
                tier=self.tier,
                confidence=0.95,
                rationale=f"LlamaGuard4: {decoded[:120]}",
                guard_name=self.name,
                evidence={"raw": decoded},
                severity="high",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale=f"LlamaGuard4: {decoded[:120]}",
            guard_name=self.name,
            evidence={"raw": decoded},
            latency_ms=latency_ms,
        )
