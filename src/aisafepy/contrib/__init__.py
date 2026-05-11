"""Thin Guard wrappers around upstream guardrail libraries.

Each module here adapts one external tool (Llama Guard 4, ShieldGemma,
Llama Prompt Guard 2, llm-guard, Presidio, Lakera Guard) to the
:class:`aisafepy.stream.Guard` protocol. None of these modules import
their upstream dependency at module load time. They all defer the
import to first use so that ``from aisafepy import contrib`` is cheap.
"""

from aisafepy.contrib.lakera import LakeraGuard
from aisafepy.contrib.llama_guard import LlamaGuard4
from aisafepy.contrib.llm_guard import LLMGuardOutputScanner
from aisafepy.contrib.presidio import PresidioPIIGuard
from aisafepy.contrib.prompt_guard import PromptGuard2
from aisafepy.contrib.shield_gemma import ShieldGemma

__all__ = [
    "LakeraGuard",
    "LLMGuardOutputScanner",
    "LlamaGuard4",
    "PresidioPIIGuard",
    "PromptGuard2",
    "ShieldGemma",
]
