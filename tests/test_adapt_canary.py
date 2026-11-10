from __future__ import annotations

import json
from pathlib import Path

import pytest

from aisafepy.adapt.canary import CanaryController, ShadowGuard, promote
from aisafepy.adapt.compile import CompilationReport, CompiledArtifact
from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context, GuardPipeline


def _regex_artifact(name: str = "regex-0", label: int = 0) -> CompiledArtifact:
    return CompiledArtifact(
        kind="regex",
        name=name,
        payload={"patterns": [{"span": "ignore previous instructions",
                               "regex": r"\bignore\s+previous\s+instructions\b",
                               "precision": 1.0}]},
        cluster_label=label,
        attack_success_rate=1.0,
        n_records=6,
    )


def _report(*names: str) -> CompilationReport:
    arts = [_regex_artifact(n, i) for i, n in enumerate(names or ("regex-0",))]
    arts.append(CompiledArtifact(kind="classifier", name="deferred", payload=None,
                                 cluster_label=9, attack_success_rate=1.0, n_records=3))
    arts.append(CompiledArtifact(kind="policy", name="cedar-0", payload="forbid();",
                                 cluster_label=0, attack_success_rate=1.0, n_records=3,
                                 metadata={"dsl": "cedar"}))
    return CompilationReport(artifacts=arts)


class FakeClock:
    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def _block(name: str) -> GuardDecision:
    return GuardDecision.block(tier=1, confidence=1.0, rationale="x", guard_name=name)


def _allow(name: str) -> GuardDecision:
    return GuardDecision.allow(tier=1, guard_name=name)


# ---- start / attach ---------------------------------------------------------


def test_start_live_attaches_regex_to_tier1_and_skips_unmaterializable(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report(), pipe, state_dir=tmp_path, clock=FakeClock())
    res = ctrl.start()
    assert res.promoted == ["regex-0"]
    assert [g.name for g in pipe.tier1] == ["regex-0"]
    assert any("deferred" in n for n in res.notes)
    assert any("cedar-0" in n and "materialize failed" in n for n in res.notes)
    assert ctrl.state_path.exists()
    assert ctrl.stats["regex-0"].status == "live"


def test_start_shadow_stores_guard_on_pipeline(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report(), pipe, state_dir=tmp_path)
    res = ctrl.start(shadow=True)
    assert res.shadow_only == ["regex-0"]
    assert pipe.tier1 == []
    assert [n for _, n in pipe.shadow_guards] == ["regex-0"]


@pytest.mark.asyncio
async def test_shadow_in_tier_records_but_never_blocks(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report(), pipe, state_dir=tmp_path, min_samples=1)
    ctrl.start(shadow=True, shadow_in_tier=True)
    assert isinstance(pipe.tier1[0], ShadowGuard)
    ctx = Context(buffer="please ignore previous instructions",
                  metadata={"labeled_benign": True})
    d = await pipe.evaluate_chunk(ctx)
    assert d.action.value == "allow"
    assert d.evidence["shadow"] is True and d.evidence["shadow_action"] == "block"
    st = ctrl.stats["regex-0"]
    assert st.samples == 1 and st.flagged == 1 and st.benign_fp == 1
    assert ctrl.fp_rate("regex-0") == 1.0


# ---- routing ------------------------------------------------------------------


def test_route_is_deterministic_and_roughly_proportional(tmp_path: Path):
    ctrl = CanaryController(_report(), GuardPipeline(), canary_traffic_pct=0.25,
                            state_dir=tmp_path)
    keys = [f"user-{i}" for i in range(4000)]
    first = [ctrl.route(k) for k in keys]
    second = [ctrl.route(k) for k in keys]
    assert first == second
    share = sum(first) / len(first)
    assert 0.20 < share < 0.30
    assert ctrl.route(b"bytes-key") == ctrl.route("bytes-key")
    assert ctrl.route(42) == ctrl.route("42")
    full = CanaryController(_report(), GuardPipeline(), canary_traffic_pct=1.0, state_dir=tmp_path)
    assert all(full.route(k) for k in keys[:50])
    none = CanaryController(_report(), GuardPipeline(), canary_traffic_pct=0.0, state_dir=tmp_path)
    assert not any(none.route(k) for k in keys[:50])


def test_invalid_parameters_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        CanaryController(_report(), GuardPipeline(), canary_traffic_pct=1.5, state_dir=tmp_path)
    with pytest.raises(ValueError):
        CanaryController(_report(), GuardPipeline(), fp_budget=-0.1, state_dir=tmp_path)


def test_record_ignores_unknown_guards_and_out_of_canary_keys(tmp_path: Path):
    ctrl = CanaryController(_report(), GuardPipeline(), canary_traffic_pct=0.5,
                            state_dir=tmp_path)
    ctrl.start()
    ctrl.record(_block("nope"))
    assert ctrl.stats["regex-0"].samples == 0
    in_keys = [k for k in (f"k{i}" for i in range(200)) if ctrl.route(k)]
    out_keys = [k for k in (f"k{i}" for i in range(200)) if not ctrl.route(k)]
    ctrl.record(_block("regex-0"), request_key=out_keys[0], labeled_benign=True)
    assert ctrl.stats["regex-0"].samples == 0
    ctrl.record(_block("regex-0"), request_key=in_keys[0], labeled_benign=True)
    assert ctrl.stats["regex-0"].samples == 1


# ---- FP budget / rollback ---------------------------------------------------


def test_check_rolls_back_guard_over_fp_budget(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report("regex-0", "regex-1"), pipe, fp_budget=0.05,
                            min_samples=20, state_dir=tmp_path, clock=FakeClock())
    ctrl.start()
    assert len(pipe.tier1) == 2
    # regex-0: 2 FPs in 40 benign (5%) is within budget; regex-1: 4/40 is not.
    for i in range(40):
        ctrl.record(_block("regex-0") if i < 2 else _allow("regex-0"), labeled_benign=True)
        ctrl.record(_block("regex-1") if i < 4 else _allow("regex-1"), labeled_benign=True)
    res = ctrl.check()
    assert res.promoted == ["regex-0"]
    assert res.rolled_back == ["regex-1"]
    assert [g.name for g in pipe.tier1] == ["regex-0"]
    assert res.fp_rates["regex-1"] == pytest.approx(0.1)
    assert res.fp_rate_observed == pytest.approx(6 / 80)
    assert any("rolled back" in n for n in res.notes)
    state = json.loads(ctrl.state_path.read_text())
    assert state["rolled_back"] == ["regex-1"]
    # A second check keeps the rolled-back guard out and does not double-count.
    res2 = ctrl.check()
    assert res2.rolled_back == ["regex-1"] and res2.promoted == ["regex-0"]


def test_check_requires_min_samples_before_verdict(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report(), pipe, fp_budget=0.0, min_samples=10, state_dir=tmp_path)
    ctrl.start()
    for _ in range(5):
        ctrl.record(_block("regex-0"), labeled_benign=True)
    res = ctrl.check()
    assert res.promoted == ["regex-0"] and not res.rolled_back
    assert any("insufficient labeled benign samples" in n for n in res.notes)
    assert len(pipe.tier1) == 1


def test_unlabeled_traffic_never_triggers_rollback(tmp_path: Path):
    ctrl = CanaryController(_report(), GuardPipeline(), fp_budget=0.0, min_samples=1,
                            state_dir=tmp_path)
    ctrl.start()
    for _ in range(100):
        ctrl.record(_block("regex-0"))
    res = ctrl.check()
    assert res.rolled_back == [] and ctrl.fp_rate() is None
    assert ctrl.stats["regex-0"].flag_rate == 1.0


# ---- shadow -> live --------------------------------------------------------


def test_promote_all_requires_fp_evidence(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report("regex-0", "regex-1"), pipe, fp_budget=0.01,
                            min_samples=10, state_dir=tmp_path)
    ctrl.start(shadow=True)
    for _ in range(10):
        ctrl.record(_allow("regex-0"), labeled_benign=True)
    for _ in range(10):
        ctrl.record(_block("regex-1"), labeled_benign=True)
    promoted = ctrl.promote_all()
    assert promoted == ["regex-0"]
    assert [g.name for g in pipe.tier1] == ["regex-0"]
    assert [n for _, n in pipe.shadow_guards] == ["regex-1"]
    assert ctrl.stats["regex-1"].status == "shadow"
    assert ctrl.promote_all(require_fp_ok=False) == ["regex-1"]
    assert pipe.shadow_guards == []
    assert len(pipe.tier1) == 2


def test_rollback_of_shadow_guard_removes_it(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report(), pipe, state_dir=tmp_path)
    ctrl.start(shadow=True, shadow_in_tier=True)
    ctrl.rollback("regex-0", reason="test")
    assert pipe.tier1 == []
    assert ctrl.stats["regex-0"].status == "rolled_back"
    with pytest.raises(KeyError):
        ctrl.rollback("missing")


# ---- persistence -----------------------------------------------------------


def test_state_round_trips_through_load(tmp_path: Path):
    pipe = GuardPipeline()
    ctrl = CanaryController(_report("regex-0", "regex-1"), pipe, canary_traffic_pct=0.3,
                            fp_budget=0.02, min_samples=5, state_dir=tmp_path,
                            clock=FakeClock())
    ctrl.start()
    for _ in range(5):
        ctrl.record(_block("regex-1"), labeled_benign=True)
    ctrl.check()  # rolls back regex-1
    ctrl.rollback  # noqa: B018 - attribute access only

    pipe2 = GuardPipeline()
    again = CanaryController.load(tmp_path, _report("regex-0", "regex-1"), pipe2)
    assert again.canary_traffic_pct == 0.3 and again.fp_budget == 0.02 and again.min_samples == 5
    assert again.stats["regex-0"].status == "live"
    assert again.stats["regex-1"].status == "rolled_back"
    assert again.stats["regex-1"].benign_fp == 5
    assert [g.name for g in pipe2.tier1] == ["regex-0"]
    assert again.state()["rolled_back"] == ["regex-1"]


def test_load_tolerates_version1_state(tmp_path: Path):
    (tmp_path / "canary.json").write_text(json.dumps({
        "promoted": ["regex-0"], "shadow_only": [], "fp_budget": 0.01, "timestamp": 1.0,
    }))
    pipe = GuardPipeline()
    ctrl = CanaryController.load(tmp_path, _report(), pipe)
    assert ctrl.stats["regex-0"].status == "live"
    assert [g.name for g in pipe.tier1] == ["regex-0"]


# ---- functional API ----------------------------------------------------------


def test_promote_function_keeps_old_contract(tmp_path: Path):
    pipe = GuardPipeline()
    res = promote(_report(), to=pipe, state_dir=tmp_path)
    assert res.promoted == ["regex-0"]
    assert res.fp_rate_observed is None
    assert [g.name for g in pipe.tier1] == ["regex-0"]
    state = json.loads((tmp_path / "canary.json").read_text())
    assert state["promoted"] == ["regex-0"] and state["fp_budget"] == 0.005
    assert isinstance(pipe.__aisafepy_canary__, CanaryController)

    pipe2 = GuardPipeline()
    res2 = promote(_report(), to=pipe2, shadow_only=True, state_dir=tmp_path / "s")
    assert res2.shadow_only == ["regex-0"] and pipe2.tier1 == []
