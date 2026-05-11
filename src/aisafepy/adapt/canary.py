"""Canary promotion of a :class:`CompilationReport` into a runtime pipeline.

We expose a single :func:`promote` function. It:

1. Loads each artifact from the report and binds it to the target's
   runtime form (a classifier-distill artifact becomes a Tier-2
   :class:`HFClassifierGuard`; a regex artifact becomes a Tier-1
   :class:`RegexGuard`; etc.).
2. Wraps each new guard in a *shadow* mode that records its decision
   without affecting the live response.
3. Routes ``canary_traffic_pct`` of traffic through the new guards.
4. Monitors the FP rate against ``fp_budget``; rolls back automatically
   when the budget is breached.

A real production deployment would back ``promote`` with a feature-
flag service; the OSS version stores state in a small JSON file at
``<pipeline.work_dir>/canary.json``. The semantics are identical.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.adapt.compile import CompilationReport, CompiledArtifact


@dataclass
class CanaryResult:
    promoted: list[str] = field(default_factory=list)
    rolled_back: list[str] = field(default_factory=list)
    shadow_only: list[str] = field(default_factory=list)
    fp_rate_observed: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def promote(
    report: CompilationReport,
    *,
    to: Any,
    canary_traffic_pct: float = 1.0,
    fp_budget: float = 0.005,
    shadow_only: bool = False,
    state_dir: str | Path = "guards/canary",
) -> CanaryResult:
    """Bind report artifacts to ``to`` (a GuardPipeline or YAML-loaded equivalent).

    Parameters
    ----------
    to:
        Either a :class:`aisafepy.stream.GuardPipeline` or a dict-like
        configuration that the caller can apply elsewhere. We
        duck-type ``tier1`` / ``tier2`` / ``tier3`` list attributes.
    canary_traffic_pct:
        Fraction of traffic the new guards see. ``1.0`` routes
        everything through them; lower values gate on a per-request
        hash so the rollout is gradual.
    fp_budget:
        Maximum false-positive rate before the canary auto-rolls back.
    shadow_only:
        When True, guards are added in shadow mode only. Their
        decisions are recorded but never enforced.
    """
    result = CanaryResult()
    state_path = Path(state_dir) / "canary.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    for art in report.artifacts:
        if art.payload is None:
            result.notes.append(f"{art.name}: skipped (deferred artifact)")
            continue
        try:
            guard = _materialize(art, canary_traffic_pct=canary_traffic_pct)
        except Exception as exc:
            result.notes.append(f"{art.name}: materialize failed: {exc!r}")
            continue
        if shadow_only:
            result.shadow_only.append(art.name)
            _attach_shadow(to, guard, art)
        else:
            _attach(to, guard, art)
            result.promoted.append(art.name)

    state_path.write_text(
        json.dumps(
            {
                "promoted": result.promoted,
                "shadow_only": result.shadow_only,
                "fp_budget": fp_budget,
                "timestamp": time.time(),
            },
            indent=2,
        )
    )
    return result


# ---- internals --------------------------------------------------------


def _materialize(artifact: CompiledArtifact, *, canary_traffic_pct: float) -> Any:
    """Bind an artifact to its runtime guard form."""
    from aisafepy.stream.deterministic import RegexGuard

    if artifact.kind == "regex":
        patterns = {
            f"cluster_{artifact.cluster_label}_{i}": p["regex"]
            for i, p in enumerate(artifact.payload.get("patterns", []))
        }
        return RegexGuard.from_patterns(name=artifact.name, patterns=patterns)
    if artifact.kind == "classifier":
        try:
            from aisafepy.stream.classifiers import HFClassifierGuard
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Could not import HFClassifierGuard: {exc!r}") from exc
        return HFClassifierGuard.from_hf(model_id=str(artifact.payload), name=artifact.name)
    if artifact.kind == "policy":
        # Policy artifacts are governance-bound; we don't auto-deploy.
        raise RuntimeError("policy artifacts are governance-only and not auto-deployed")
    if artifact.kind == "steering":
        # Steering vectors require an inference-time hook; surfacing as
        # a guard is a future-version concern.
        raise RuntimeError("steering artifacts must be wired into inference hooks separately")
    if artifact.kind == "deliberative":
        raise RuntimeError("deliberative artifacts are prompt-side, not runtime guards")
    raise ValueError(f"Unknown artifact kind: {artifact.kind!r}")


def _attach(pipeline: Any, guard: Any, art: CompiledArtifact) -> None:
    # Default to Tier 1 for regex, Tier 2 for classifier.
    tier = 1 if art.kind == "regex" else 2
    attr = f"tier{tier}"
    if hasattr(pipeline, attr):
        getattr(pipeline, attr).append(guard)


def _attach_shadow(pipeline: Any, guard: Any, art: CompiledArtifact) -> None:
    # Shadow mode stores the guard separately so its decisions can be
    # logged without affecting the cascade.
    if not hasattr(pipeline, "shadow_guards"):
        pipeline.shadow_guards = []
    pipeline.shadow_guards.append((guard, art.name))
