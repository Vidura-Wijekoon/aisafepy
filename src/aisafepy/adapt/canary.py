"""Canary promotion of a :class:`CompilationReport` into a runtime pipeline.

The public surface is two things:

* :func:`promote`, the one-call API from the README. It materializes
  each artifact into its runtime guard form, attaches it to the target
  pipeline (live or shadow) and persists a small state file.
* :class:`CanaryController`, the stateful object behind ``promote``
  that owns the full lifecycle: deterministic traffic routing,
  per-guard outcome accounting, false-positive-rate estimation against
  labeled benign traffic, automatic rollback when a guard exceeds its
  FP budget, shadow-to-live promotion, and state persistence so a
  controller can be reloaded by a later process.

False-positive accounting
-------------------------
A guard's FP rate is estimated only over requests the caller labeled
benign (``labeled_benign=True``). A block or escalate on a benign
request is a false positive. Unlabeled traffic contributes to sample
counts but never to the FP estimate, so a controller fed only
unlabeled production traffic will not roll anything back; it will
say so in :meth:`CanaryController.check` notes. The rollback decision
requires at least ``min_samples`` benign samples so a single early
mistake cannot trip a guard.

A real production deployment would back the controller with a
feature-flag service; the OSS version stores state in
``<state_dir>/canary.json``. The semantics are identical.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.adapt.compile import CompilationReport, CompiledArtifact
from aisafepy.core.decisions import Action, GuardDecision

STATE_VERSION = 2
_ROUTE_BUCKETS = 10_000


@dataclass
class CanaryResult:
    promoted: list[str] = field(default_factory=list)
    rolled_back: list[str] = field(default_factory=list)
    shadow_only: list[str] = field(default_factory=list)
    fp_rate_observed: float | None = None
    notes: list[str] = field(default_factory=list)
    fp_rates: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardStats:
    """Per-guard outcome counters maintained by :class:`CanaryController`."""

    name: str
    kind: str
    status: str = "shadow"  # 'shadow' | 'live' | 'rolled_back'
    samples: int = 0
    flagged: int = 0
    benign_samples: int = 0
    benign_fp: int = 0
    last_seen: float | None = None

    @property
    def fp_rate(self) -> float | None:
        if self.benign_samples == 0:
            return None
        return self.benign_fp / self.benign_samples

    @property
    def flag_rate(self) -> float | None:
        if self.samples == 0:
            return None
        return self.flagged / self.samples

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fp_rate"] = self.fp_rate
        d["flag_rate"] = self.flag_rate
        return d


class ShadowGuard:
    """Wrap a guard so it is evaluated but never enforced.

    The inner decision is reported to the controller (which attributes
    it to the inner guard's name) and the wrapper returns an ``allow``
    whose evidence carries the shadowed action. This lets shadow guards
    sit inside a normal tier for pipelines that have no separate shadow
    slot; the cascade sees only allows.
    """

    def __init__(self, inner: Any, controller: CanaryController, *, name: str | None = None):
        self.inner = inner
        self.controller = controller
        self.name = name or f"shadow:{getattr(inner, 'name', type(inner).__name__)}"
        self.tier = getattr(inner, "tier", 1)
        self.inner_name = getattr(inner, "name", type(inner).__name__)

    async def __call__(self, ctx: Any) -> GuardDecision:
        decision = await self.inner(ctx)
        meta = getattr(ctx, "metadata", None) or {}
        self.controller.record(
            decision,
            guard_name=self.inner_name,
            labeled_benign=meta.get("labeled_benign"),
            request_key=meta.get("request_key"),
        )
        return GuardDecision.allow(
            tier=decision.tier,
            guard_name=self.name,
            rationale=f"shadow: {decision.action.value} suppressed",
            evidence={
                "shadow": True,
                "shadow_action": decision.action.value,
                "shadow_confidence": decision.confidence,
                "shadow_rationale": decision.rationale,
            },
            latency_ms=decision.latency_ms,
        )


class CanaryController:
    """Lifecycle manager for canary-deployed guards.

    Parameters
    ----------
    report:
        The compilation report whose artifacts are deployed.
    pipeline:
        A :class:`aisafepy.stream.GuardPipeline` or any object with
        ``tier1`` / ``tier2`` / ``tier3`` list attributes.
    canary_traffic_pct:
        Fraction (0..1) of request keys routed through the canary.
    fp_budget:
        Maximum tolerated false-positive rate on labeled benign traffic.
    state_dir:
        Directory holding ``canary.json``.
    min_samples:
        Minimum labeled-benign samples before a rollback can trigger.
    clock:
        Injectable time source (seconds since epoch) for tests.
    """

    def __init__(
        self,
        report: CompilationReport,
        pipeline: Any,
        *,
        canary_traffic_pct: float = 1.0,
        fp_budget: float = 0.005,
        state_dir: str | Path = "guards/canary",
        min_samples: int = 50,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 <= canary_traffic_pct <= 1.0:
            raise ValueError("canary_traffic_pct must be within [0, 1]")
        if not 0.0 <= fp_budget <= 1.0:
            raise ValueError("fp_budget must be within [0, 1]")
        self.report = report
        self.pipeline = pipeline
        self.canary_traffic_pct = float(canary_traffic_pct)
        self.fp_budget = float(fp_budget)
        self.state_dir = Path(state_dir)
        self.min_samples = int(min_samples)
        self.clock = clock
        self.guards: dict[str, Any] = {}
        self.artifacts: dict[str, CompiledArtifact] = {}
        self.stats: dict[str, GuardStats] = {}
        self.notes: list[str] = []
        self._started_at: float | None = None

    # ---- lifecycle -----------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.state_dir / "canary.json"

    def start(self, *, shadow: bool = False, shadow_in_tier: bool = False) -> CanaryResult:
        """Materialize artifacts and attach them to the pipeline.

        ``shadow=True`` records decisions without enforcing them. With
        ``shadow_in_tier=True`` the shadow guards are inserted into the
        pipeline tiers wrapped in :class:`ShadowGuard` (so they run on
        real traffic automatically); otherwise they are stored on
        ``pipeline.shadow_guards`` for the caller to drive.
        """
        self._started_at = self.clock()
        result = CanaryResult()
        for art in self.report.artifacts:
            if art.payload is None:
                self.notes.append(f"{art.name}: skipped (deferred artifact)")
                continue
            if art.name in self.guards:
                continue
            try:
                guard = _materialize(art, canary_traffic_pct=self.canary_traffic_pct)
            except Exception as exc:
                self.notes.append(f"{art.name}: materialize failed: {exc!r}")
                continue
            self.guards[art.name] = guard
            self.artifacts[art.name] = art
            self.stats[art.name] = GuardStats(name=art.name, kind=art.kind)
            if shadow:
                self._attach_shadow(art.name, in_tier=shadow_in_tier)
                result.shadow_only.append(art.name)
            else:
                self._attach_live(art.name)
                result.promoted.append(art.name)
        result.notes = list(self.notes)
        self.save()
        return result

    def route(self, request_key: str | bytes | int) -> bool:
        """Deterministically decide whether ``request_key`` is in the canary.

        The same key always yields the same answer, so a user or
        session that is in the canary stays in it for the whole rollout
        and experiences a consistent policy.
        """
        if self.canary_traffic_pct >= 1.0:
            return True
        if self.canary_traffic_pct <= 0.0:
            return False
        raw = request_key if isinstance(request_key, bytes) else str(request_key).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % _ROUTE_BUCKETS
        return bucket < round(self.canary_traffic_pct * _ROUTE_BUCKETS)

    def record(
        self,
        decision: GuardDecision,
        *,
        guard_name: str | None = None,
        labeled_benign: bool | None = None,
        request_key: str | None = None,
    ) -> None:
        """Account one decision to a canary guard.

        ``guard_name`` defaults to ``decision.guard_name``. Decisions for
        guards the controller does not manage are ignored. Only block /
        escalate count as "flagged"; a flagged decision on a request
        labeled benign is a false positive.
        """
        name = guard_name or decision.guard_name
        if name is None or name not in self.stats:
            return
        if request_key is not None and not self.route(request_key):
            return
        st = self.stats[name]
        flagged = decision.action in (Action.BLOCK, Action.ESCALATE)
        st.samples += 1
        st.last_seen = self.clock()
        if flagged:
            st.flagged += 1
        if labeled_benign:
            st.benign_samples += 1
            if flagged:
                st.benign_fp += 1

    def fp_rate(self, guard_name: str | None = None) -> float | None:
        """FP rate for one guard, or pooled across all guards when ``None``."""
        if guard_name is not None:
            st = self.stats.get(guard_name)
            return st.fp_rate if st else None
        benign = sum(s.benign_samples for s in self.stats.values())
        if benign == 0:
            return None
        return sum(s.benign_fp for s in self.stats.values()) / benign

    def check(self) -> CanaryResult:
        """Apply the FP budget: roll back any guard over budget.

        Guards with fewer than ``min_samples`` labeled-benign samples are
        left alone and noted. Returns the current overall status.
        """
        result = CanaryResult()
        for name, st in self.stats.items():
            if st.status == "rolled_back":
                result.rolled_back.append(name)
                continue
            rate = st.fp_rate
            if rate is None or st.benign_samples < self.min_samples:
                result.notes.append(
                    f"{name}: insufficient labeled benign samples "
                    f"({st.benign_samples}/{self.min_samples}); no FP verdict"
                )
            elif rate > self.fp_budget:
                self.rollback(name, reason=f"fp_rate {rate:.4f} > budget {self.fp_budget:.4f}")
                result.rolled_back.append(name)
                continue
            if st.status == "live":
                result.promoted.append(name)
            elif st.status == "shadow":
                result.shadow_only.append(name)
        result.fp_rate_observed = self.fp_rate()
        result.fp_rates = {n: s.fp_rate for n, s in self.stats.items() if s.fp_rate is not None}
        result.samples = {n: s.samples for n, s in self.stats.items()}
        result.notes.extend(self.notes)
        self.save()
        return result

    def rollback(self, guard_name: str, *, reason: str = "manual") -> None:
        """Remove a guard from the pipeline (live tiers and shadow slots)."""
        st = self.stats.get(guard_name)
        if st is None:
            raise KeyError(guard_name)
        guard = self.guards.get(guard_name)
        if guard is not None:
            self._detach(guard)
        st.status = "rolled_back"
        self.notes.append(f"{guard_name}: rolled back ({reason})")
        self.save()

    def promote(self, guard_name: str) -> None:
        """Move one shadow guard to live enforcement."""
        st = self.stats.get(guard_name)
        if st is None:
            raise KeyError(guard_name)
        if st.status != "shadow":
            return
        guard = self.guards[guard_name]
        self._detach(guard)
        self._attach_live(guard_name)
        self.save()

    def promote_all(self, *, require_fp_ok: bool = True) -> list[str]:
        """Promote every shadow guard, optionally only those within budget.

        With ``require_fp_ok=True`` (default) a guard is promoted only if
        it has at least ``min_samples`` benign samples and its FP rate
        is within budget. Returns the names promoted.
        """
        promoted: list[str] = []
        for name, st in self.stats.items():
            if st.status != "shadow":
                continue
            if require_fp_ok:
                rate = st.fp_rate
                if rate is None or st.benign_samples < self.min_samples or rate > self.fp_budget:
                    self.notes.append(f"{name}: not promoted (fp_rate={rate}, n={st.benign_samples})")
                    continue
            self.promote(name)
            promoted.append(name)
        return promoted

    # ---- persistence ---------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "timestamp": self.clock(),
            "started_at": self._started_at,
            "fp_budget": self.fp_budget,
            "canary_traffic_pct": self.canary_traffic_pct,
            "min_samples": self.min_samples,
            "promoted": [n for n, s in self.stats.items() if s.status == "live"],
            "shadow_only": [n for n, s in self.stats.items() if s.status == "shadow"],
            "rolled_back": [n for n, s in self.stats.items() if s.status == "rolled_back"],
            "fp_rate_observed": self.fp_rate(),
            "guards": {n: s.to_dict() for n, s in self.stats.items()},
            "notes": list(self.notes),
        }

    def save(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state(), indent=2, default=str), encoding="utf-8")
        return self.state_path

    @classmethod
    def load(
        cls,
        state_dir: str | Path,
        report: CompilationReport,
        pipeline: Any,
        *,
        clock: Callable[[], float] = time.time,
        shadow_in_tier: bool = False,
    ) -> CanaryController:
        """Rebuild a controller from ``<state_dir>/canary.json``.

        Guards are re-materialized from ``report`` and re-attached
        according to their persisted status; counters are restored.
        """
        state_dir = Path(state_dir)
        data = json.loads((state_dir / "canary.json").read_text(encoding="utf-8"))
        ctrl = cls(
            report,
            pipeline,
            canary_traffic_pct=float(data.get("canary_traffic_pct", 1.0)),
            fp_budget=float(data.get("fp_budget", 0.005)),
            state_dir=state_dir,
            min_samples=int(data.get("min_samples", 50)),
            clock=clock,
        )
        ctrl._started_at = data.get("started_at")
        ctrl.notes = list(data.get("notes", []))
        persisted: dict[str, dict[str, Any]] = data.get("guards", {})
        if not persisted:  # version-1 state files
            for n in data.get("promoted", []):
                persisted[n] = {"status": "live"}
            for n in data.get("shadow_only", []):
                persisted[n] = {"status": "shadow"}
        by_name = {a.name: a for a in report.artifacts}
        for name, g in persisted.items():
            art = by_name.get(name)
            if art is None or art.payload is None:
                ctrl.notes.append(f"{name}: in state file but not in report; ignored")
                continue
            try:
                guard = _materialize(art, canary_traffic_pct=ctrl.canary_traffic_pct)
            except Exception as exc:
                ctrl.notes.append(f"{name}: materialize failed on load: {exc!r}")
                continue
            ctrl.guards[name] = guard
            ctrl.artifacts[name] = art
            st = GuardStats(
                name=name,
                kind=str(g.get("kind", art.kind)),
                status=str(g.get("status", "shadow")),
                samples=int(g.get("samples", 0)),
                flagged=int(g.get("flagged", 0)),
                benign_samples=int(g.get("benign_samples", 0)),
                benign_fp=int(g.get("benign_fp", 0)),
                last_seen=g.get("last_seen"),
            )
            ctrl.stats[name] = st
            if st.status == "live":
                ctrl._attach_live(name)
            elif st.status == "shadow":
                ctrl._attach_shadow(name, in_tier=shadow_in_tier)
        return ctrl

    # ---- attachment helpers --------------------------------------------

    def _attach_live(self, name: str) -> None:
        guard = self.guards[name]
        _attach(self.pipeline, guard, self.artifacts[name])
        self.stats[name].status = "live"

    def _attach_shadow(self, name: str, *, in_tier: bool) -> None:
        guard = self.guards[name]
        art = self.artifacts[name]
        if in_tier:
            wrapper = ShadowGuard(guard, self, name=f"shadow:{name}")
            _attach(self.pipeline, wrapper, art)
        else:
            _attach_shadow(self.pipeline, guard, art)
        self.stats[name].status = "shadow"

    def _detach(self, guard: Any) -> None:
        for attr in ("tier1", "tier2", "tier3"):
            tier = getattr(self.pipeline, attr, None)
            if isinstance(tier, list):
                tier[:] = [
                    g
                    for g in tier
                    if g is not guard and not (isinstance(g, ShadowGuard) and g.inner is guard)
                ]
        shadow = getattr(self.pipeline, "shadow_guards", None)
        if isinstance(shadow, list):
            shadow[:] = [g for g in shadow if g is not guard]


def promote(
    report: CompilationReport,
    *,
    to: Any,
    canary_traffic_pct: float = 1.0,
    fp_budget: float = 0.005,
    shadow_only: bool = False,
    state_dir: str | Path = "guards/canary",
    min_samples: int = 50,
    controller: CanaryController | None = None,
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
        hash so the rollout is gradual (see :meth:`CanaryController.route`).
    fp_budget:
        Maximum false-positive rate before the canary auto-rolls back.
    shadow_only:
        When True, guards are added in shadow mode only. Their
        decisions are recorded but never enforced.
    controller:
        Pass an existing controller to keep driving it; otherwise one
        is created and exposed on the result via ``to.__aisafepy_canary__``.

    The returned :class:`CanaryResult` lists what was promoted, what is
    in shadow and, when the controller already has labeled samples, the
    observed FP rate.
    """
    ctrl = controller or CanaryController(
        report,
        to,
        canary_traffic_pct=canary_traffic_pct,
        fp_budget=fp_budget,
        state_dir=state_dir,
        min_samples=min_samples,
    )
    start_result = ctrl.start(shadow=shadow_only)
    checked = ctrl.check()
    result = CanaryResult(
        promoted=checked.promoted,
        rolled_back=checked.rolled_back,
        shadow_only=checked.shadow_only,
        fp_rate_observed=checked.fp_rate_observed,
        notes=list(dict.fromkeys(start_result.notes + checked.notes)),
        fp_rates=checked.fp_rates,
        samples=checked.samples,
    )
    with contextlib.suppress(Exception):  # exotic pipeline objects may forbid attrs
        to.__aisafepy_canary__ = ctrl
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
    if artifact.kind == "hashed_classifier":
        from aisafepy.adapt.compile.hashed import materialize_hashed_classifier

        return materialize_hashed_classifier(artifact)
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
    # Shadow mode stores the guard in the pipeline's shadow slot, where it
    # is evaluated and reported through ``on_decision`` but never enforced.
    # The slot holds plain guards (the artifact name is the guard's name),
    # which is the shape ``GuardPipeline`` evaluates.
    if not hasattr(pipeline, "shadow_guards"):
        pipeline.shadow_guards = []
    pipeline.shadow_guards.append(guard)
