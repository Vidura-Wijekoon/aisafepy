"""Offline tests for the model-backed Tier-2/Tier-3 guards.

The forecasters and probes wrap Hugging Face models that are never
downloaded in CI. These tests pin the *contract* those classes keep
when no model is available: they import, construct, and fail open with
an explicit evidence flag instead of raising. Where a method takes a
model object we substitute a tiny fake that implements only the calls
the guard makes.
"""

from __future__ import annotations

import sys

import pytest

from aisafepy.stream.classifiers import HFClassifierGuard, StubClassifierGuard
from aisafepy.stream.forecasters import KelpForecast, SCMForecast, StreamGuardForecast
from aisafepy.stream.pipeline import Context, GuardPipeline


@pytest.mark.parametrize("cls", [StreamGuardForecast, SCMForecast, KelpForecast])
@pytest.mark.asyncio
async def test_forecasters_without_model_allow_with_flag(cls):
    guard = cls(name=cls.__name__.lower())
    d = await guard(Context(buffer="some streamed prefix"))
    assert d.action.value == "allow"
    assert d.evidence == {"no_model": True}
    assert d.tier == 2
    assert d.guard_name == cls.__name__.lower()


@pytest.mark.parametrize("cls", [StreamGuardForecast, SCMForecast, KelpForecast])
@pytest.mark.asyncio
async def test_forecasters_with_model_id_but_no_torch_still_allow(cls, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    guard = cls(name="f", model_id="org/model-that-does-not-exist")
    d = await guard(Context(buffer="prefix"))
    assert d.action.value == "allow"
    assert d.evidence.get("no_model") is True


def test_forecaster_defaults_are_sane():
    g = KelpForecast(name="kelp")
    assert g.threshold == 0.5
    assert g.on_time_target == pytest.approx(0.90)
    assert g.model_id is None


class _FakeTensor:
    def __init__(self, probs):
        self._probs = probs

    def cpu(self):
        return self

    def numpy(self):
        import array

        class _Arr:
            def __init__(self, rows):
                self.rows = rows

            def __getitem__(self, i):
                return self.rows[i]

        return _Arr([array.array("d", self._probs)])


class _FakeTorch:
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def no_grad(self):
        return self._NoGrad()

    @staticmethod
    def softmax(logits, dim):
        return logits


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, text, **kw):
        return _FakeInputs(input_ids=[0])


class _FakeModel:
    def __init__(self, probs, id2label=None):
        self._probs = probs

        class Cfg:
            pass

        self.config = Cfg()
        if id2label is not None:
            self.config.id2label = id2label

    def __call__(self, **inputs):
        class Out:
            logits = _FakeTensor(self._probs)

        return Out()


@pytest.mark.asyncio
async def test_hf_classifier_blocks_above_threshold_with_fake_model():
    guard = HFClassifierGuard(name="c", model_id="fake/model", threshold=0.5)
    guard._model = _FakeModel([0.1, 0.9])
    guard._tokenizer = _FakeTokenizer()
    guard._torch = _FakeTorch()
    guard.device = "cpu"
    d = await guard(Context(buffer="bad text"))
    assert d.is_blocked
    assert d.confidence == pytest.approx(0.9)
    assert d.severity == "medium"  # 0.9 is not > 0.9
    assert d.evidence["model"] == "fake/model"


@pytest.mark.asyncio
async def test_hf_classifier_allows_below_threshold_and_reports_confidence():
    guard = HFClassifierGuard(name="c", model_id="fake/model", threshold=0.5)
    guard._model = _FakeModel([0.8, 0.2])
    guard._tokenizer = _FakeTokenizer()
    guard._torch = _FakeTorch()
    guard.device = "cpu"
    d = await guard(Context(buffer="fine"))
    assert d.action.value == "allow"
    assert d.confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_hf_classifier_resolves_string_unsafe_label():
    guard = HFClassifierGuard(name="c", model_id="fake/model", threshold=0.5, unsafe_label="JAILBREAK")
    guard._model = _FakeModel([0.05, 0.95], id2label={0: "BENIGN", 1: "JAILBREAK"})
    guard._tokenizer = _FakeTokenizer()
    guard._torch = _FakeTorch()
    guard.device = "cpu"
    d = await guard(Context(buffer="x"))
    assert d.is_blocked
    assert d.severity == "high"


@pytest.mark.asyncio
async def test_hf_classifier_empty_text_allows_without_inference():
    guard = HFClassifierGuard(name="c", model_id="fake/model")
    guard._model = object()  # would explode if called
    guard._tokenizer = None
    guard._torch = None
    d = await guard(Context(buffer=""))
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_hf_classifier_out_of_range_label_index_is_safe():
    guard = HFClassifierGuard(name="c", model_id="fake/model", unsafe_label=5)
    guard._model = _FakeModel([0.5, 0.5])
    guard._tokenizer = _FakeTokenizer()
    guard._torch = _FakeTorch()
    guard.device = "cpu"
    d = await guard(Context(buffer="x"))
    assert d.action.value == "allow"


def test_hf_classifier_from_hf_pops_name_kwarg():
    g = HFClassifierGuard.from_hf("org/m", name="custom", tier=3, threshold=0.8)
    assert g.name == "custom"
    assert g.model_id == "org/m"
    assert g.tier == 3
    assert g.threshold == 0.8


@pytest.mark.asyncio
async def test_stub_classifier_is_case_insensitive_and_reports_trigger():
    g = StubClassifierGuard(triggers=("harmful",))
    d = await g(Context(buffer="This Is HARMFUL"))
    assert d.is_blocked
    assert d.evidence == {"trigger": "harmful"}
    d2 = await g(Context(buffer="benign"))
    assert d2.action.value == "allow"


@pytest.mark.asyncio
async def test_forecaster_composes_in_pipeline_tier2():
    pipe = GuardPipeline(tier2=[KelpForecast(name="kelp")])
    d = await pipe.evaluate_chunk(Context(buffer="hello"))
    assert d.action.value == "allow"
    assert d.tier == 2


# ---- probes -----------------------------------------------------------------------------


def test_probe_classes_import_and_expose_contract():
    from aisafepy.stream import probes

    assert hasattr(probes, "LinearProbe")
    assert hasattr(probes, "MLPProbe")
    lp = probes.LinearProbe(name="p", layers=(8, 12))
    assert lp.tier == 3
    assert lp.threshold == 0.5
    assert lp.layers == (8, 12)


def test_probe_require_torch_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    from aisafepy.stream import probes

    with pytest.raises(ModuleNotFoundError, match=r"aisafepy\[probes\]"):
        probes._require_torch()
