"""Google ShieldGemma wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from aisafepy.stream.classifiers import HFClassifierGuard


@dataclass
class ShieldGemma(HFClassifierGuard):
    """ShieldGemma 2B / 9B / 27B.

    The 2B variant is a reasonable Tier-2 choice; the 9B/27B variants
    are slow enough to be Tier-3. Override ``model_id`` to pick the
    weight size.
    """

    name: str = "shield-gemma"
    model_id: str = "google/shieldgemma-2b"
    tier: int = 2
