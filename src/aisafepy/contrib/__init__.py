"""Thin Guard wrappers around upstream guardrail libraries and APIs.

Each module here adapts one external tool (Llama Guard 4, ShieldGemma,
Llama Prompt Guard 2, llm-guard, Presidio, Lakera Guard, the OpenAI
Moderation API, Azure AI Content Safety) to the
:class:`aisafepy.stream.Guard` protocol. None of these modules import
their upstream dependency at module load time. They all defer the
import to first use so that ``from aisafepy import contrib`` is cheap,
and every wrapper fails *open* with an explanatory evidence payload
when its dependency or credential is missing.
"""

from aisafepy.contrib.azure_content_safety import AzureContentSafetyGuard
from aisafepy.contrib.lakera import LakeraGuard
from aisafepy.contrib.llama_guard import LlamaGuard4
from aisafepy.contrib.llm_guard import LLMGuardOutputScanner
from aisafepy.contrib.openai_moderation import OpenAIModerationGuard
from aisafepy.contrib.presidio import PresidioPIIGuard
from aisafepy.contrib.prompt_guard import PromptGuard2
from aisafepy.contrib.shield_gemma import ShieldGemma

__all__ = [
    "AzureContentSafetyGuard",
    "LLMGuardOutputScanner",
    "LakeraGuard",
    "LlamaGuard4",
    "OpenAIModerationGuard",
    "PresidioPIIGuard",
    "PromptGuard2",
    "ShieldGemma",
]
