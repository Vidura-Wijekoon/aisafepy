"""Declarative pipeline configuration.

The README promises ``GuardPipeline.from_yaml("guards.yaml")``. This
module implements it. A configuration document looks like::

    version: 1
    name: support-bot
    budget_ms_p95: 80
    uncertainty_band: [0.3, 0.7]
    on_budget_exceeded: allow
    tier_timeouts_ms: {2: 60, 3: 400}
    tiers:
      1:
        - kind: regex_pii
          redact: true
        - kind: blocklist
          name: secrets
          terms: ["BEGIN PRIVATE KEY", "api_key="]
      2:
        - kind: hf_classifier
          model_id: ${PROMPT_GUARD_MODEL:-meta-llama/Llama-Prompt-Guard-2-22M}
          threshold: 0.6
      3:
        - kind: custom
          target: my_company.guards:LegalReviewGuard
          jurisdiction: EU
    shadow:
      - kind: stub_classifier
        triggers: ["HARMFUL"]

Every guard entry is a *spec*: a ``kind`` naming a builder in the
:class:`GuardRegistry` plus arbitrary keyword arguments for it.
String values support ``${VAR}`` and ``${VAR:-default}`` environment
interpolation so secrets and model paths stay out of the file.

Builders for every guard shipped in the repository are pre-registered.
They import their modules lazily, so loading a config that references
``hf_classifier`` does not import ``torch`` until the guard is built,
and :func:`validate_config` never builds anything at all.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aisafepy.stream.pipeline import GuardPipeline

try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except Exception:  # pragma: no cover - depends on env
    _HAS_YAML = False


class ConfigError(ValueError):
    """Raised for malformed pipeline configuration."""


# ---- environment interpolation ------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string values.

    Raises :class:`ConfigError` when a variable is missing and has no
    default. Non-string leaves are returned unchanged.
    """
    environ = os.environ if env is None else env
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name!r} is not set and has no default")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v, environ) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, environ) for v in value]
    if isinstance(value, tuple):
        return tuple(interpolate_env(v, environ) for v in value)
    return value


# ---- schema ----------------------------------------------------------------


class GuardSpec(BaseModel):
    """One guard entry. ``kind`` selects the builder; every other key is
    forwarded to it as a keyword argument."""

    model_config = ConfigDict(extra="allow")

    kind: str = Field(min_length=1)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class PipelineConfig(BaseModel):
    """Top-level configuration document."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str = "pipeline"
    budget_ms_p95: float = Field(default=80.0, gt=0)
    uncertainty_band: tuple[float, float] = (0.30, 0.70)
    on_budget_exceeded: str = "allow"
    tier_timeouts_ms: dict[int, float] | None = None
    tiers: dict[int, list[GuardSpec]] = Field(default_factory=dict)
    shadow: list[GuardSpec] = Field(default_factory=list)

    @field_validator("on_budget_exceeded")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("allow", "block"):
            raise ValueError("on_budget_exceeded must be 'allow' or 'block'")
        return v

    @field_validator("uncertainty_band")
    @classmethod
    def _check_band(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("uncertainty_band must satisfy 0 <= low <= high <= 1")
        return (float(lo), float(hi))

    @field_validator("tiers", mode="before")
    @classmethod
    def _normalize_tiers(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError("tiers must be a mapping of tier number to a list of guards")
        out: dict[int, Any] = {}
        for key, guards in v.items():
            tier = _tier_number(key)
            if guards is None:
                guards = []
            if not isinstance(guards, list):
                raise ValueError(f"tier {tier}: expected a list of guard specs")
            out[tier] = guards
        return out

    @field_validator("tiers")
    @classmethod
    def _check_tier_numbers(cls, v: dict[int, list[GuardSpec]]) -> dict[int, list[GuardSpec]]:
        bad = [t for t in v if t not in (1, 2, 3)]
        if bad:
            raise ValueError(f"tiers must be 1, 2 or 3 (got {bad})")
        return v


def _tier_number(key: Any) -> int:
    if isinstance(key, int):
        return key
    text = str(key).strip().lower()
    if text.startswith("tier"):
        text = text[4:].strip(" _-")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"invalid tier key {key!r}") from exc


# ---- registry --------------------------------------------------------------

Builder = Callable[..., Any]


class GuardRegistry:
    """Name -> builder mapping used to materialize guard specs."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._builders: dict[str, Builder] = {}
        self._required: dict[str, tuple[str, ...]] = {}
        if include_defaults:
            _register_defaults(self)

    def register(
        self, kind: str, *, required: tuple[str, ...] = ()
    ) -> Callable[[Builder], Builder]:
        """Decorator registering ``fn`` as the builder for ``kind``.

        ``required`` lists parameter names that :func:`validate_config`
        checks for without building the guard.
        """

        def deco(fn: Builder) -> Builder:
            self.add(kind, fn, required=required)
            return fn

        return deco

    def add(self, kind: str, builder: Builder, *, required: tuple[str, ...] = ()) -> None:
        if not kind or not isinstance(kind, str):
            raise ConfigError("guard kind must be a non-empty string")
        self._builders[kind] = builder
        self._required[kind] = tuple(required)

    def names(self) -> list[str]:
        return sorted(self._builders)

    def has(self, kind: str) -> bool:
        return kind in self._builders

    def required_params(self, kind: str) -> tuple[str, ...]:
        return self._required.get(kind, ())

    def build(self, spec: GuardSpec | Mapping[str, Any]) -> Any:
        """Instantiate the guard described by ``spec``."""
        if not isinstance(spec, GuardSpec):
            try:
                spec = GuardSpec.model_validate(dict(spec))
            except ValidationError as exc:
                raise ConfigError(f"invalid guard spec {dict(spec)!r}: {exc}") from exc
        builder = self._builders.get(spec.kind)
        if builder is None:
            raise ConfigError(
                f"unknown guard kind {spec.kind!r}; known kinds: {', '.join(self.names())}"
            )
        missing = [p for p in self.required_params(spec.kind) if p not in spec.params]
        if missing:
            raise ConfigError(f"guard kind {spec.kind!r} is missing required parameter(s) {missing}")
        try:
            return builder(**spec.params)
        except TypeError as exc:
            raise ConfigError(f"guard kind {spec.kind!r}: {exc}") from exc


def _import_target(path: str) -> Any:
    """Resolve ``package.module:attr`` or ``package.module.attr``."""
    if ":" in path:
        mod_name, _, attr = path.partition(":")
    else:
        mod_name, _, attr = path.rpartition(".")
    if not mod_name or not attr:
        raise ConfigError(f"invalid import target {path!r}; expected 'module:attr'")
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import module {mod_name!r} for target {path!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(f"module {mod_name!r} has no attribute {attr!r}") from exc


def _apply_extra(guard: Any, extra: dict[str, Any], *, kind: str) -> Any:
    """Set optional attributes produced by a factory classmethod.

    Lets a spec carry tuning fields (``scope``, ``max_input_chars``,
    ``redact_with`` ...) without the builder having to know every
    dataclass field up front. Unknown attributes are a configuration
    error rather than a silent no-op.
    """
    for key, value in extra.items():
        if not hasattr(guard, key):
            raise ConfigError(f"guard kind {kind!r} has no parameter {key!r}")
        setattr(guard, key, value)
    return guard


def _register_defaults(reg: GuardRegistry) -> None:
    # -- deterministic ------------------------------------------------------
    @reg.register("regex", required=("patterns",))
    def _regex(
        *,
        patterns: dict[str, str],
        name: str = "regex",
        action_on_match: str = "block",
        **kw: Any,
    ) -> Any:
        from aisafepy.stream.deterministic import RegexGuard

        guard = RegexGuard.from_patterns(name=name, patterns=patterns, action_on_match=action_on_match)
        return _apply_extra(guard, kw, kind="regex")

    @reg.register("regex_pii")
    def _regex_pii(*, name: str = "pii", redact: bool = False, **kw: Any) -> Any:
        from aisafepy.stream.deterministic import RegexGuard

        return _apply_extra(RegexGuard.compile_pii(name=name, redact=redact), kw, kind="regex_pii")

    @reg.register("blocklist", required=("terms",))
    def _blocklist(*, terms: list[str], name: str = "blocklist", **kw: Any) -> Any:
        from aisafepy.stream.deterministic import BlocklistGuard

        return BlocklistGuard(name=name, terms=list(terms), **kw)

    @reg.register("aho_corasick", required=("terms",))
    def _aho(*, terms: list[str], name: str = "aho_corasick", **kw: Any) -> Any:
        from aisafepy.stream.deterministic import AhoCorasickGuard

        return AhoCorasickGuard(name=name, terms=list(terms), **kw)

    @reg.register("schema", required=("model",))
    def _schema(*, model: str, name: str = "schema", partial: bool = True) -> Any:
        from aisafepy.stream.deterministic import SchemaGuard

        return SchemaGuard(name=name, model=_import_target(model), partial=partial)

    # -- classifiers ---------------------------------------------------------
    @reg.register("hf_classifier", required=("model_id",))
    def _hf(*, model_id: str, name: str | None = None, **kw: Any) -> Any:
        from aisafepy.stream.classifiers import HFClassifierGuard

        return HFClassifierGuard(name=name or model_id, model_id=model_id, **kw)

    @reg.register("stub_classifier")
    def _stub(*, name: str = "stub", triggers: list[str] | None = None, tier: int = 2) -> Any:
        from aisafepy.stream.classifiers import StubClassifierGuard

        if triggers is None:
            return StubClassifierGuard(name=name, tier=tier)
        return StubClassifierGuard(name=name, triggers=tuple(triggers), tier=tier)

    # -- forecasters ---------------------------------------------------------
    def _forecaster(cls_name: str, default_name: str) -> Builder:
        def build(*, name: str = default_name, **kw: Any) -> Any:
            module = importlib.import_module("aisafepy.stream.forecasters")
            return getattr(module, cls_name)(name=name, **kw)

        return build

    reg.add("streamguard", _forecaster("StreamGuardForecast", "streamguard"))
    reg.add("scm", _forecaster("SCMForecast", "scm"))
    reg.add("kelp", _forecaster("KelpForecast", "kelp"))

    # -- probes ------------------------------------------------------------
    def _probe(cls_name: str) -> Builder:
        def build(*, path: str, name: str | None = None, threshold: float | None = None) -> Any:
            module = importlib.import_module("aisafepy.stream.probes")
            probe = getattr(module, cls_name).load(path)
            if name is not None:
                probe.name = name
            if threshold is not None:
                probe.threshold = threshold
            return probe

        return build

    reg.add("linear_probe", _probe("LinearProbe"), required=("path",))
    reg.add("mlp_probe", _probe("MLPProbe"), required=("path",))

    # -- contrib -------------------------------------------------------------
    def _contrib(module: str, cls_name: str) -> Builder:
        def build(**kw: Any) -> Any:
            mod = importlib.import_module(f"aisafepy.contrib.{module}")
            return getattr(mod, cls_name)(**kw)

        return build

    reg.add("lakera", _contrib("lakera", "LakeraGuard"))
    reg.add("llama_guard", _contrib("llama_guard", "LlamaGuard4"))
    reg.add("shield_gemma", _contrib("shield_gemma", "ShieldGemma"))
    reg.add("prompt_guard", _contrib("prompt_guard", "PromptGuard2"))
    reg.add("presidio", _contrib("presidio", "PresidioPIIGuard"))
    reg.add("llm_guard", _contrib("llm_guard", "LLMGuardOutputScanner"), required=("name",))
    reg.add("openai_moderation", _contrib("openai_moderation", "OpenAIModerationGuard"))
    reg.add("azure_content_safety", _contrib("azure_content_safety", "AzureContentSafetyGuard"))

    # -- stream extras -------------------------------------------------------
    @reg.register("llm_judge", required=("judge",))
    def _judge(*, judge: Any, **kw: Any) -> Any:
        from aisafepy.stream.judge import LLMJudgeGuard

        if isinstance(judge, str):
            judge = _import_target(judge)
            if isinstance(judge, type):
                judge = judge()
        return LLMJudgeGuard(judge=judge, **kw)

    @reg.register("session")
    def _session(**kw: Any) -> Any:
        from aisafepy.stream.session import SessionGuard

        return SessionGuard(**kw)

    # -- escape hatch ----------------------------------------------------------
    @reg.register("custom", required=("target",))
    def _custom(*, target: str, **kw: Any) -> Any:
        factory = _import_target(target)
        return factory(**kw)


_DEFAULT_REGISTRY: GuardRegistry | None = None


def default_registry() -> GuardRegistry:
    """Process-wide registry with all built-in builders. Users may
    ``register`` additional kinds on it."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = GuardRegistry()
    return _DEFAULT_REGISTRY


# ---- loading ---------------------------------------------------------------


def _read_document(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).suffix in (".yaml", ".yml", ".json")
    ):
        path = Path(source)
        if not path.exists():
            raise ConfigError(f"configuration file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        text = str(source)
    if not _HAS_YAML:
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "PyYAML is not installed; install it (`pip install pyyaml`) or supply JSON"
            ) from exc
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[union-attr]
            raise ConfigError(f"invalid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    return data


def parse_config(
    source: str | Path | Mapping[str, Any], *, env: Mapping[str, str] | None = None
) -> PipelineConfig:
    """Read, interpolate and validate a configuration document."""
    raw = interpolate_env(_read_document(source), env)
    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = list(err.get("loc", ()))
        where = ".".join(str(x) for x in loc) or "<root>"
        if loc and loc[0] == "tiers" and len(loc) >= 3:
            where = f"tiers[{loc[1]}] guard #{loc[2]}" + (
                "." + ".".join(str(x) for x in loc[3:]) if len(loc) > 3 else ""
            )
        elif loc and loc[0] == "shadow" and len(loc) >= 2:
            where = f"shadow guard #{loc[1]}" + (
                "." + ".".join(str(x) for x in loc[2:]) if len(loc) > 2 else ""
            )
        parts.append(f"{where}: {err.get('msg')}")
    return "invalid pipeline configuration: " + "; ".join(parts)


def validate_config(
    source: str | Path | Mapping[str, Any] | PipelineConfig,
    *,
    registry: GuardRegistry | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return a list of problems without instantiating any guard.

    Checks structure, that every ``kind`` is registered, and that the
    builder's declared required parameters are present. An empty list
    means the document is loadable (modulo import-time failures inside
    heavy guards).
    """
    reg = registry or default_registry()
    try:
        cfg = source if isinstance(source, PipelineConfig) else parse_config(source, env=env)
    except ConfigError as exc:
        return [str(exc)]
    problems: list[str] = []

    def check(where: str, spec: GuardSpec) -> None:
        if not reg.has(spec.kind):
            problems.append(f"{where}: unknown guard kind {spec.kind!r}")
            return
        missing = [p for p in reg.required_params(spec.kind) if p not in spec.params]
        if missing:
            problems.append(f"{where}: kind {spec.kind!r} is missing required parameter(s) {missing}")

    for tier, specs in sorted(cfg.tiers.items()):
        for i, spec in enumerate(specs):
            check(f"tiers[{tier}] guard #{i}", spec)
    for i, spec in enumerate(cfg.shadow):
        check(f"shadow guard #{i}", spec)
    if cfg.tier_timeouts_ms:
        for t, ms in cfg.tier_timeouts_ms.items():
            if t not in (0, 1, 2, 3):
                problems.append(f"tier_timeouts_ms: unknown tier {t}")
            if ms <= 0:
                problems.append(f"tier_timeouts_ms[{t}]: must be > 0")
    return problems


def load_pipeline(
    source: str | Path | Mapping[str, Any] | PipelineConfig,
    *,
    registry: GuardRegistry | None = None,
    env: Mapping[str, str] | None = None,
) -> GuardPipeline:
    """Build a :class:`GuardPipeline` from YAML text, a path, a dict or a
    parsed :class:`PipelineConfig`."""
    reg = registry or default_registry()
    cfg = source if isinstance(source, PipelineConfig) else parse_config(source, env=env)

    def build_all(where: str, specs: list[GuardSpec]) -> list[Any]:
        out: list[Any] = []
        for i, spec in enumerate(specs):
            try:
                out.append(reg.build(spec))
            except ConfigError as exc:
                raise ConfigError(f"{where} guard #{i}: {exc}") from exc
        return out

    pipeline = GuardPipeline(
        tier1=build_all("tiers[1]", cfg.tiers.get(1, [])),
        tier2=build_all("tiers[2]", cfg.tiers.get(2, [])),
        tier3=build_all("tiers[3]", cfg.tiers.get(3, [])),
        budget_ms_p95=cfg.budget_ms_p95,
        uncertainty_band=cfg.uncertainty_band,
        on_budget_exceeded=cfg.on_budget_exceeded,  # type: ignore[arg-type]
        tier_timeouts_ms=dict(cfg.tier_timeouts_ms) if cfg.tier_timeouts_ms else None,
        shadow_guards=build_all("shadow", cfg.shadow),
        name=cfg.name,
    )
    return pipeline


# ---- dumping ---------------------------------------------------------------


def guard_to_spec(guard: Any) -> dict[str, Any]:
    """Best-effort spec for a guard instance.

    Guards exposing ``to_spec()`` are authoritative. The built-in
    deterministic and stub guards are handled explicitly; anything
    else is recorded as ``{"kind": "custom", "repr": ...}`` which is
    informative but not loadable.
    """
    to_spec = getattr(guard, "to_spec", None)
    if callable(to_spec):
        spec = dict(to_spec())
        spec.setdefault("kind", "custom")
        return spec

    cls = type(guard).__name__
    name = getattr(guard, "name", None)
    if cls == "RegexGuard":
        patterns = {k: p.pattern for k, p in guard.patterns.items()}
        return {
            "kind": "regex",
            "name": name,
            "patterns": patterns,
            "action_on_match": guard.action_on_match,
        }
    if cls in ("BlocklistGuard", "AhoCorasickGuard"):
        return {
            "kind": "blocklist" if cls == "BlocklistGuard" else "aho_corasick",
            "name": name,
            "terms": list(guard.terms),
            "case_sensitive": guard.case_sensitive,
        }
    if cls == "StubClassifierGuard":
        return {"kind": "stub_classifier", "name": name, "triggers": list(guard.triggers),
                "tier": guard.tier}
    if cls == "HFClassifierGuard":
        return {
            "kind": "hf_classifier",
            "name": name,
            "model_id": guard.model_id,
            "threshold": guard.threshold,
            "tier": guard.tier,
        }
    if cls == "SchemaGuard":
        model = guard.model
        return {
            "kind": "schema",
            "name": name,
            "model": f"{model.__module__}:{model.__qualname__}",
            "partial": guard.partial,
        }
    return {"kind": "custom", "name": name, "repr": repr(guard), "type": f"{type(guard).__module__}.{cls}"}


def dump_pipeline(pipeline: GuardPipeline) -> dict[str, Any]:
    """Serialize a pipeline back to the configuration format."""
    return {
        "version": 1,
        "name": pipeline.name,
        "budget_ms_p95": pipeline.budget_ms_p95,
        "uncertainty_band": list(pipeline.uncertainty_band),
        "on_budget_exceeded": pipeline.on_budget_exceeded,
        "tier_timeouts_ms": dict(pipeline.tier_timeouts_ms) if pipeline.tier_timeouts_ms else None,
        "tiers": {
            1: [guard_to_spec(g) for g in pipeline.tier1],
            2: [guard_to_spec(g) for g in pipeline.tier2],
            3: [guard_to_spec(g) for g in pipeline.tier3],
        },
        "shadow": [guard_to_spec(g) for g in pipeline.shadow_guards],
    }


def dump_yaml(pipeline: GuardPipeline) -> str:
    """Serialize a pipeline to a YAML document (requires PyYAML)."""
    if not _HAS_YAML:
        raise RuntimeError("dump_yaml requires PyYAML. Install via `pip install pyyaml`.")
    return yaml.safe_dump(dump_pipeline(pipeline), sort_keys=False)  # type: ignore[union-attr]
