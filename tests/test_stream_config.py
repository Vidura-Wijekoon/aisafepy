from __future__ import annotations

import textwrap

import pytest
from pydantic import BaseModel

from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.config import (
    ConfigError,
    GuardRegistry,
    PipelineConfig,
    default_registry,
    dump_pipeline,
    dump_yaml,
    guard_to_spec,
    interpolate_env,
    load_pipeline,
    parse_config,
    validate_config,
)
from aisafepy.stream.deterministic import BlocklistGuard, RegexGuard, SchemaGuard
from aisafepy.stream.pipeline import Context, GuardPipeline

YAML_DOC = textwrap.dedent(
    """
    version: 1
    name: demo
    budget_ms_p95: 50
    uncertainty_band: [0.2, 0.8]
    on_budget_exceeded: block
    tier_timeouts_ms: {2: 25}
    tiers:
      1:
        - kind: regex_pii
          redact: true
        - kind: blocklist
          name: ${BLOCKLIST_NAME:-secrets}
          terms: ["BEGIN PRIVATE KEY", "${EXTRA_TERM}"]
      tier2:
        - kind: stub_classifier
          triggers: ["UNSAFE"]
    shadow:
      - kind: stub_classifier
        name: shadow-stub
        triggers: ["SHADOW"]
    """
)


class ToolArgs(BaseModel):
    path: str


def make_guard(*, name: str = "custom", marker: str = "X") -> StubClassifierGuard:
    return StubClassifierGuard(name=name, triggers=(marker,))


# ---- interpolation --------------------------------------------------------


def test_interpolate_env_substitutes_and_defaults():
    env = {"A": "alpha"}
    out = interpolate_env({"x": "${A}", "y": "${B:-beta}", "z": ["${A}-${B:-}"], "n": 3}, env)
    assert out == {"x": "alpha", "y": "beta", "z": ["alpha-"], "n": 3}


def test_interpolate_env_missing_without_default_raises():
    with pytest.raises(ConfigError, match="MISSING"):
        interpolate_env("${MISSING}", {})


# ---- parsing ----------------------------------------------------------------


def test_parse_config_from_yaml_text_with_env():
    cfg = parse_config(YAML_DOC, env={"EXTRA_TERM": "api_key="})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.name == "demo"
    assert cfg.on_budget_exceeded == "block"
    assert set(cfg.tiers) == {1, 2}
    assert cfg.tiers[1][1].params["name"] == "secrets"
    assert cfg.tiers[1][1].params["terms"] == ["BEGIN PRIVATE KEY", "api_key="]
    assert cfg.tier_timeouts_ms == {2: 25.0}
    assert len(cfg.shadow) == 1


def test_parse_config_rejects_bad_tier_and_mode():
    with pytest.raises(ConfigError, match="tiers"):
        parse_config({"tiers": {"4": []}})
    with pytest.raises(ConfigError, match="on_budget_exceeded"):
        parse_config({"on_budget_exceeded": "explode"})
    with pytest.raises(ConfigError, match="uncertainty_band"):
        parse_config({"uncertainty_band": [0.9, 0.1]})


def test_parse_config_names_offending_guard():
    with pytest.raises(ConfigError) as exc_info:
        parse_config({"tiers": {"1": [{"kind": "regex"}, {"name": "no-kind"}]}})
    assert "tiers[1] guard #1" in str(exc_info.value)


def test_parse_config_from_file(tmp_path):
    p = tmp_path / "guards.yaml"
    p.write_text(YAML_DOC, encoding="utf-8")
    cfg = parse_config(p, env={"EXTRA_TERM": "x"})
    assert cfg.name == "demo"
    with pytest.raises(ConfigError, match="not found"):
        parse_config(tmp_path / "missing.yaml")


# ---- validation -------------------------------------------------------------


def test_validate_config_reports_unknown_kind_and_missing_params():
    problems = validate_config(
        {
            "tiers": {
                "1": [{"kind": "nope"}, {"kind": "regex"}],
                "2": [{"kind": "hf_classifier", "model_id": "x"}],
            },
            "shadow": [{"kind": "custom"}],
            "tier_timeouts_ms": {"9": 0},
        }
    )
    joined = "\n".join(problems)
    assert "tiers[1] guard #0: unknown guard kind 'nope'" in joined
    assert "tiers[1] guard #1" in joined and "patterns" in joined
    assert "shadow guard #0" in joined and "target" in joined
    assert "unknown tier 9" in joined
    assert "must be > 0" in joined
    # hf_classifier with model_id is structurally fine and must not be built.
    assert "hf_classifier" not in joined


def test_validate_config_clean_document():
    assert validate_config(YAML_DOC, env={"EXTRA_TERM": "k"}) == []


# ---- loading ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_pipeline_builds_working_cascade(monkeypatch):
    monkeypatch.setenv("EXTRA_TERM", "api_key=")
    pipe = load_pipeline(YAML_DOC)
    assert isinstance(pipe, GuardPipeline)
    assert pipe.name == "demo"
    assert pipe.budget_ms_p95 == 50
    assert pipe.on_budget_exceeded == "block"
    assert pipe.tier_timeouts_ms == {2: 25.0}
    assert [type(g).__name__ for g in pipe.tier1] == ["RegexGuard", "BlocklistGuard"]
    assert pipe.tier1[1].name == "secrets"
    assert pipe.tier2[0].triggers == ("UNSAFE",)
    assert pipe.shadow_guards[0].name == "shadow-stub"

    d = await pipe.evaluate_chunk(Context(buffer="here is api_key=abc"))
    assert d.is_blocked and d.tier == 1
    d = await pipe.evaluate_chunk(Context(buffer="totally UNSAFE"))
    assert d.is_blocked and d.tier == 2


def test_from_yaml_and_from_dict_classmethods(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTRA_TERM", "x")
    p = tmp_path / "p.yml"
    p.write_text(YAML_DOC, encoding="utf-8")
    assert GuardPipeline.from_yaml(p).name == "demo"
    assert GuardPipeline.from_yaml(str(p)).name == "demo"
    assert GuardPipeline.from_dict({"name": "d", "tiers": {}}).name == "d"


def test_unknown_kind_lists_known_kinds():
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline({"tiers": {"1": [{"kind": "does_not_exist"}]}})
    msg = str(exc_info.value)
    assert "tiers[1] guard #0" in msg
    assert "known kinds" in msg and "regex_pii" in msg and "blocklist" in msg


def test_bad_builder_kwargs_surface_as_config_error():
    with pytest.raises(ConfigError, match="regex_pii"):
        load_pipeline({"tiers": {"1": [{"kind": "regex_pii", "bogus": 1}]}})


def test_custom_kind_via_dotted_path():
    pipe = load_pipeline(
        {
            "tiers": {
                "1": [
                    {"kind": "custom", "target": "tests.test_stream_config:make_guard", "marker": "ZZ"},
                    {"kind": "custom", "target": "aisafepy.stream.classifiers.StubClassifierGuard",
                     "name": "cls", "triggers": ["QQ"]},
                ]
            }
        }
    )
    assert pipe.tier1[0].triggers == ("ZZ",)
    assert pipe.tier1[1].name == "cls"
    with pytest.raises(ConfigError, match="cannot import"):
        load_pipeline({"tiers": {"1": [{"kind": "custom", "target": "no.such.module:x"}]}})
    with pytest.raises(ConfigError, match="no attribute"):
        load_pipeline({"tiers": {"1": [{"kind": "custom", "target": "tests.test_stream_config:nope"}]}})


def test_schema_kind_resolves_model():
    pipe = load_pipeline(
        {"tiers": {"1": [{"kind": "schema", "model": "tests.test_stream_config:ToolArgs"}]}}
    )
    guard = pipe.tier1[0]
    assert isinstance(guard, SchemaGuard)
    assert guard.model is ToolArgs


def test_registry_register_decorator_and_required():
    reg = GuardRegistry(include_defaults=False)

    @reg.register("mine", required=("marker",))
    def _build(*, marker: str) -> StubClassifierGuard:
        return StubClassifierGuard(name="mine", triggers=(marker,))

    assert reg.names() == ["mine"]
    assert reg.build({"kind": "mine", "marker": "A"}).triggers == ("A",)
    with pytest.raises(ConfigError, match="missing required"):
        reg.build({"kind": "mine"})
    pipe = load_pipeline({"tiers": {"1": [{"kind": "mine", "marker": "B"}]}}, registry=reg)
    assert pipe.tier1[0].triggers == ("B",)
    assert "regex" in default_registry().names()


# ---- dumping ----------------------------------------------------------------------


def test_round_trip_for_deterministic_guards():
    pipe = GuardPipeline(
        name="rt",
        tier1=[
            RegexGuard.from_patterns("kw", {"k": r"foo\d+"}),
            BlocklistGuard(name="bl", terms=["a", "b"], case_sensitive=True),
            SchemaGuard(name="sch", model=ToolArgs, partial=False),
        ],
        tier2=[StubClassifierGuard(name="stub", triggers=("Z",))],
        tier_timeouts_ms={2: 10},
    )
    cfg = dump_pipeline(pipe)
    regex_spec = cfg["tiers"][1][0]
    assert regex_spec["kind"] == "regex" and regex_spec["name"] == "kw"
    assert regex_spec["patterns"] == {"k": r"foo\d+"} and regex_spec["action_on_match"] == "block"
    assert cfg["tiers"][1][1]["kind"] == "blocklist" and cfg["tiers"][1][1]["case_sensitive"] is True
    assert cfg["tiers"][1][2]["model"] == "tests.test_stream_config:ToolArgs"
    rebuilt = load_pipeline(cfg)
    assert rebuilt.name == "rt"
    assert rebuilt.tier1[0].patterns["k"].pattern == r"foo\d+"
    assert rebuilt.tier1[1].terms == ["a", "b"]
    assert rebuilt.tier1[2].model is ToolArgs and rebuilt.tier1[2].partial is False
    assert rebuilt.tier2[0].triggers == ("Z",)
    assert rebuilt.tier_timeouts_ms == {2: 10.0}
    # to_dict on the instance and YAML dump agree.
    assert pipe.to_dict() == cfg
    assert "kind: regex" in dump_yaml(pipe)


def test_guard_to_spec_prefers_to_spec_and_falls_back_to_custom():
    class WithSpec:
        name = "ws"
        tier = 1

        def to_spec(self):
            return {"kind": "custom", "target": "x:y", "a": 1}

        async def __call__(self, ctx):  # pragma: no cover
            raise NotImplementedError

    class Opaque:
        name = "op"
        tier = 1

        async def __call__(self, ctx):  # pragma: no cover
            raise NotImplementedError

    assert guard_to_spec(WithSpec()) == {"kind": "custom", "target": "x:y", "a": 1}
    spec = guard_to_spec(Opaque())
    assert spec["kind"] == "custom" and spec["name"] == "op" and "Opaque" in spec["type"]
