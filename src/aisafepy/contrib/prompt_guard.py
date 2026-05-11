"""Meta Llama Prompt Guard 2 wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from aisafepy.stream.classifiers import HFClassifierGuard


@dataclass
class PromptGuard2(HFClassifierGuard):
    """Llama Prompt Guard 2 (22M / 86M).

    The 22M model is fast enough for Tier 1 in many setups; the 86M
    model has better detection but lives in Tier 2.
    """

    name: str = "prompt-guard-2"
    model_id: str = "meta-llama/Llama-Prompt-Guard-2-22M"
    tier: int = 2
    threshold: float = 0.5
    unsafe_label: int | str = 1
