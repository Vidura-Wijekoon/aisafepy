"""Adapters for red-team frameworks and production trace sources.

Each adapter normalizes its native record format into a
:class:`FailureRecord`. The compiler only sees ``FailureRecord``s, so
new sources (Promptfoo, DeepEval, internal QA tools) plug in with a
~30-line shim. The generic :class:`JSONLSource` and :class:`CSVSource`
cover most home-grown formats without any code at all: pass a
``mapping`` from ``FailureRecord`` field names to (dotted) paths in
the native record.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FailureRecord:
    """A normalized red-team / production-failure record."""

    id: str
    input: str
    output: str
    was_violation: bool
    attack_category: str | None = None
    """e.g. 'encoding_bypass', 'tool_poisoning', 'many_shot_override'."""
    severity: str = "medium"
    scorer: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    source: str = "unknown"


@runtime_checkable
class RedTeamSource(Protocol):
    """A stream of ``FailureRecord``s."""

    def __iter__(self) -> Iterator[FailureRecord]:  # pragma: no cover - protocol
        ...


# ---- PyRIT ------------------------------------------------------------


class PyRITSource:
    """Wraps a PyRIT memory store.

    PyRIT writes its results to either a DuckDB or a JSONL store. This
    adapter accepts the path to either and yields one record per
    scored attempt.
    """

    def __init__(
        self,
        memory_db: str | Path | None = None,
        jsonl_path: str | Path | None = None,
        only_violations: bool = True,
    ):
        if memory_db is None and jsonl_path is None:
            raise ValueError("PyRITSource requires memory_db or jsonl_path")
        self.memory_db = Path(memory_db) if memory_db else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.only_violations = only_violations

    def __iter__(self) -> Iterator[FailureRecord]:
        if self.jsonl_path is not None:
            yield from self._from_jsonl()
        else:
            yield from self._from_duckdb()

    def _from_jsonl(self) -> Iterator[FailureRecord]:
        assert self.jsonl_path is not None
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rec = _normalize_pyrit_record(obj, default_id=f"pyrit-{i}")
                if self.only_violations and not rec.was_violation:
                    continue
                yield rec

    def _from_duckdb(self) -> Iterator[FailureRecord]:
        try:
            import duckdb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "PyRITSource(memory_db=...) requires `pip install duckdb`."
            ) from exc
        assert self.memory_db is not None
        conn = duckdb.connect(str(self.memory_db), read_only=True)
        try:
            # Schema varies across PyRIT versions; query defensively.
            rows = conn.execute(
                """
                SELECT
                    conversation_id, original_value, converted_value,
                    response_text, scorer, score, score_value, timestamp, labels
                FROM prompt_request_pieces
                """
            ).fetchall()
        except Exception:
            rows = conn.execute("SELECT * FROM prompt_request_pieces").fetchall()
        finally:
            conn.close()

        for i, row in enumerate(rows):
            obj = dict(zip(("conversation_id", "original_value", "converted_value",
                            "response_text", "scorer", "score", "score_value",
                            "timestamp", "labels"), row, strict=False))
            rec = _normalize_pyrit_record(obj, default_id=f"pyrit-db-{i}")
            if self.only_violations and not rec.was_violation:
                continue
            yield rec


def _normalize_pyrit_record(obj: dict[str, Any], default_id: str) -> FailureRecord:
    return FailureRecord(
        id=str(obj.get("conversation_id") or obj.get("id") or default_id),
        input=str(obj.get("converted_value") or obj.get("original_value") or obj.get("input", "")),
        output=str(obj.get("response_text") or obj.get("output", "")),
        was_violation=_truthy(obj.get("score_value") or obj.get("scorer_result") or obj.get("was_violation")),
        attack_category=obj.get("labels", {}).get("attack_category") if isinstance(obj.get("labels"), dict) else None,
        scorer=obj.get("scorer"),
        score=_as_float(obj.get("score") or obj.get("score_value")),
        timestamp=_as_float(obj.get("timestamp")),
        source="pyrit",
        metadata=obj,
    )


# ---- Garak -----------------------------------------------------------


class GarakReport:
    """Reads a Garak ``report.jsonl`` file."""

    def __init__(self, path: str | Path, only_violations: bool = True):
        self.path = Path(path)
        self.only_violations = only_violations

    def __iter__(self) -> Iterator[FailureRecord]:
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Garak emits per-probe results. We're interested in the
                # ``attempt`` and ``digest`` entries.
                if obj.get("entry_type") not in ("attempt", "digest", None):
                    continue
                rec = FailureRecord(
                    id=str(obj.get("uuid") or f"garak-{i}"),
                    input=str(obj.get("prompt", "")),
                    output=str(obj.get("outputs", [""])[0] if obj.get("outputs") else ""),
                    was_violation=_truthy(obj.get("detector_results", {}).values()),
                    attack_category=str(obj.get("probe_classname") or obj.get("probe_name", "")),
                    scorer=str(obj.get("detector_classname", "")),
                    source="garak",
                    metadata=obj,
                )
                if self.only_violations and not rec.was_violation:
                    continue
                yield rec


# ---- Inspect AI ------------------------------------------------------


class InspectLog:
    """Reads Inspect AI ``*.eval`` log files (a JSON-Lines-like format)."""

    def __init__(self, path: str | Path, only_violations: bool = True):
        self.path = Path(path)
        self.only_violations = only_violations

    def __iter__(self) -> Iterator[FailureRecord]:
        # An ``.eval`` log can be a single JSON file (newer Inspect)
        # or JSONL (older). Try JSON first, then fall back.
        try:
            text = self.path.read_text(encoding="utf-8")
            obj = json.loads(text)
            samples = obj.get("samples", obj.get("results", []))
        except Exception:
            samples = []
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))

        for i, s in enumerate(samples):
            score = s.get("score") or s.get("scores", [{}])[0].get("value")
            rec = FailureRecord(
                id=str(s.get("id") or f"inspect-{i}"),
                input=str(s.get("input", "")),
                output=str(s.get("output", "")),
                was_violation=_truthy(score),
                attack_category=str(s.get("task", "")),
                source="inspect",
                metadata=s,
            )
            if self.only_violations and not rec.was_violation:
                continue
            yield rec


# ---- Production traces (Langfuse) ------------------------------------


class LangfuseTraces:
    """Pull production traces from a Langfuse client.

    The client is duck-typed: any object exposing ``fetch_traces``
    with a ``filter`` keyword works. This avoids hard dependence on
    the ``langfuse`` package.
    """

    def __init__(self, client: Any, filter: dict[str, Any] | None = None, limit: int = 1000):
        self.client = client
        self.filter = filter or {}
        self.limit = limit

    def __iter__(self) -> Iterator[FailureRecord]:
        try:
            traces = self.client.fetch_traces(filter=self.filter, limit=self.limit)
        except TypeError:  # pragma: no cover - depends on client API
            traces = self.client.fetch_traces(self.filter)
        for i, t in enumerate(getattr(traces, "data", traces) or []):
            input_text = getattr(t, "input", "") or ""
            output_text = getattr(t, "output", "") or ""
            metadata = getattr(t, "metadata", {}) or {}
            was_violation = bool(metadata.get("guard_action") not in (None, "allow"))
            yield FailureRecord(
                id=str(getattr(t, "id", f"trace-{i}")),
                input=str(input_text),
                output=str(output_text),
                was_violation=was_violation,
                attack_category=metadata.get("attack_category"),
                source="langfuse",
                metadata=metadata if isinstance(metadata, dict) else {},
            )


# Alias for symmetry with the proposal's name.
ProductionTraces = LangfuseTraces


# ---- Generic mapped sources -----------------------------------------

#: Default field mapping used by :class:`JSONLSource` and :class:`CSVSource`.
#: Keys are ``FailureRecord`` fields; values are dotted paths into the
#: native record. The first path that resolves to a non-``None`` value
#: wins when a list of paths is given.
DEFAULT_MAPPING: dict[str, str | list[str]] = {
    "id": ["id", "uuid", "conversation_id"],
    "input": ["input", "prompt", "question", "query"],
    "output": ["output", "response", "completion", "answer"],
    "was_violation": ["was_violation", "violation", "unsafe", "failed", "label"],
    "attack_category": ["attack_category", "category", "probe", "tactic"],
    "severity": ["severity"],
    "scorer": ["scorer", "judge", "detector"],
    "score": ["score", "confidence"],
    "timestamp": ["timestamp", "created_at", "time"],
}


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted path (``"a.b.0.c"``) against nested dicts/lists.

    Integer segments index into lists. Returns ``default`` when any
    segment is missing. Never raises on malformed input.
    """
    cur = obj
    for seg in path.split("."):
        if isinstance(cur, Mapping):
            if seg not in cur:
                return default
            cur = cur[seg]
        elif isinstance(cur, Sequence) and not isinstance(cur, str | bytes):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return cur


def _resolve(obj: Any, paths: str | list[str] | None) -> Any:
    if paths is None:
        return None
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        v = get_path(obj, p)
        if v is not None:
            return v
    return None


def record_from_mapping(
    obj: Mapping[str, Any],
    *,
    mapping: Mapping[str, str | list[str]] | None = None,
    default_id: str = "",
    source: str = "mapped",
) -> FailureRecord:
    """Build a :class:`FailureRecord` from a native object via a field mapping.

    Unmapped fields fall back to :data:`DEFAULT_MAPPING`. The full
    native object is retained in ``metadata`` for traceability.
    """
    m: dict[str, str | list[str]] = {**DEFAULT_MAPPING, **(dict(mapping) if mapping else {})}
    return FailureRecord(
        id=str(_resolve(obj, m.get("id")) or default_id),
        input=str(_resolve(obj, m.get("input")) or ""),
        output=str(_resolve(obj, m.get("output")) or ""),
        was_violation=_truthy(_resolve(obj, m.get("was_violation"))),
        attack_category=_opt_str(_resolve(obj, m.get("attack_category"))),
        severity=str(_resolve(obj, m.get("severity")) or "medium"),
        scorer=_opt_str(_resolve(obj, m.get("scorer"))),
        score=_as_float(_resolve(obj, m.get("score"))),
        timestamp=_as_float(_resolve(obj, m.get("timestamp"))),
        source=source,
        metadata=dict(obj),
    )


class JSONLSource:
    """Reads any JSON Lines file through a configurable field mapping.

    Example::

        JSONLSource("qa_failures.jsonl",
                    mapping={"input": "request.prompt",
                             "output": "response.text",
                             "was_violation": "judge.unsafe"})
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mapping: Mapping[str, str | list[str]] | None = None,
        only_violations: bool = True,
        source: str = "jsonl",
    ):
        self.path = Path(path)
        self.mapping = dict(mapping) if mapping else None
        self.only_violations = only_violations
        self.source = source

    def __iter__(self) -> Iterator[FailureRecord]:
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, Mapping):
                    continue
                rec = record_from_mapping(
                    obj, mapping=self.mapping, default_id=f"{self.source}-{i}", source=self.source
                )
                if self.only_violations and not rec.was_violation:
                    continue
                yield rec


class CSVSource:
    """Reads a CSV file with a header row through a field mapping.

    Mapping values are column names. Boolean-ish columns (``true``,
    ``yes``, ``1``, ``unsafe``, ``fail``) are parsed with the same
    rules as every other source.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mapping: Mapping[str, str | list[str]] | None = None,
        only_violations: bool = True,
        source: str = "csv",
        delimiter: str = ",",
        encoding: str = "utf-8",
    ):
        self.path = Path(path)
        self.mapping = dict(mapping) if mapping else None
        self.only_violations = only_violations
        self.source = source
        self.delimiter = delimiter
        self.encoding = encoding

    def __iter__(self) -> Iterator[FailureRecord]:
        with self.path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for i, row in enumerate(reader):
                clean = {k: (v if v != "" else None) for k, v in row.items() if k is not None}
                rec = record_from_mapping(
                    clean, mapping=self.mapping, default_id=f"{self.source}-{i}", source=self.source
                )
                if self.only_violations and not rec.was_violation:
                    continue
                yield rec


# ---- Promptfoo -------------------------------------------------------


class PromptfooSource:
    """Reads a promptfoo evaluation output (``promptfoo eval -o out.json``).

    promptfoo marks each test with ``success``; a failed assertion is a
    violation from the guardrail compiler's point of view. The red-team
    plugin id (``metadata.pluginId``) or strategy is used as the attack
    category when present.
    """

    def __init__(self, path: str | Path, only_violations: bool = True):
        self.path = Path(path)
        self.only_violations = only_violations

    def __iter__(self) -> Iterator[FailureRecord]:
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        results = get_path(obj, "results.results")
        if results is None:
            results = obj.get("results") if isinstance(obj, Mapping) else None
        if not isinstance(results, list):
            return
        for i, r in enumerate(results):
            if not isinstance(r, Mapping):
                continue
            vars_ = r.get("vars") or {}
            prompt = (
                get_path(r, "prompt.raw")
                or vars_.get("prompt")
                or vars_.get("query")
                or vars_.get("input")
                or ""
            )
            output = get_path(r, "response.output")
            if output is None:
                output = r.get("output", "")
            grading = r.get("gradingResult") or {}
            success = r.get("success")
            if success is None:
                success = grading.get("pass", True)
            score = r.get("score")
            if score is None:
                score = grading.get("score")
            category = (
                get_path(r, "metadata.pluginId")
                or get_path(r, "metadata.strategyId")
                or get_path(r, "testCase.metadata.pluginId")
            )
            rec = FailureRecord(
                id=str(r.get("id") or f"promptfoo-{i}"),
                input=str(prompt),
                output=str(output if not isinstance(output, Mapping) else json.dumps(output)),
                was_violation=not bool(success),
                attack_category=_opt_str(category),
                scorer=_opt_str(get_path(grading, "componentResults.0.assertion.type")),
                score=_as_float(score),
                source="promptfoo",
                metadata=dict(r),
            )
            if self.only_violations and not rec.was_violation:
                continue
            yield rec


# ---- DeepEval --------------------------------------------------------


class DeepEvalSource:
    """Reads a DeepEval test-run export (``deepeval test run`` JSON).

    Each ``testCases[*]`` entry carries ``input``, ``actualOutput`` and
    a ``metricsData`` list with ``success`` flags. A test case is a
    violation if any metric failed. The failing metric name is kept as
    ``scorer`` and its score as ``score``.
    """

    def __init__(self, path: str | Path, only_violations: bool = True):
        self.path = Path(path)
        self.only_violations = only_violations

    def __iter__(self) -> Iterator[FailureRecord]:
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        cases = obj.get("testCases") if isinstance(obj, Mapping) else None
        if cases is None and isinstance(obj, list):
            cases = obj
        if not isinstance(cases, list):
            return
        for i, c in enumerate(cases):
            if not isinstance(c, Mapping):
                continue
            metrics = c.get("metricsData") or c.get("metrics_data") or []
            failed = [m for m in metrics if isinstance(m, Mapping) and not m.get("success", True)]
            success = c.get("success")
            was_violation = bool(failed) if success is None else not bool(success)
            first = failed[0] if failed else (metrics[0] if metrics else {})
            rec = FailureRecord(
                id=str(c.get("id") or c.get("name") or f"deepeval-{i}"),
                input=str(c.get("input", "")),
                output=str(c.get("actualOutput") or c.get("actual_output") or ""),
                was_violation=was_violation,
                attack_category=_opt_str(first.get("name") if isinstance(first, Mapping) else None),
                scorer=_opt_str(first.get("evaluationModel") if isinstance(first, Mapping) else None),
                score=_as_float(first.get("score") if isinstance(first, Mapping) else None),
                source="deepeval",
                metadata=dict(c),
            )
            if self.only_violations and not rec.was_violation:
                continue
            yield rec


# ---- In-memory, merging, filtering ----------------------------------


class InMemorySource:
    """A ``RedTeamSource`` over an in-memory list. Useful in tests and notebooks."""

    def __init__(self, records: Iterable[FailureRecord]):
        self.records = list(records)

    def __iter__(self) -> Iterator[FailureRecord]:
        yield from self.records

    def __len__(self) -> int:
        return len(self.records)


def merge_sources(*sources: Iterable[FailureRecord]) -> InMemorySource:
    """Concatenate several sources into one. Ids are prefixed with the
    source name when two records would otherwise collide."""
    seen: set[str] = set()
    out: list[FailureRecord] = []
    for src in sources:
        for rec in src:
            rid = rec.id
            if rid in seen:
                rid = f"{rec.source}:{rec.id}"
                n = 1
                while rid in seen:
                    n += 1
                    rid = f"{rec.source}:{rec.id}#{n}"
                rec = replace(rec, id=rid)
            seen.add(rid)
            out.append(rec)
    return InMemorySource(out)


def dedupe_records(
    records: Iterable[FailureRecord],
    *,
    key: Sequence[str] = ("input", "output"),
    normalize: bool = True,
) -> list[FailureRecord]:
    """Drop records whose ``key`` fields repeat an earlier record.

    With ``normalize=True`` the comparison is case- and whitespace-
    insensitive, which collapses the near-identical variants that
    attack generators tend to emit. The first occurrence is kept.
    """
    seen: set[tuple[Any, ...]] = set()
    out: list[FailureRecord] = []
    for rec in records:
        parts = []
        for k in key:
            v = getattr(rec, k, None)
            if normalize and isinstance(v, str):
                v = " ".join(v.lower().split())
            parts.append(v)
        t = tuple(parts)
        if t in seen:
            continue
        seen.add(t)
        out.append(rec)
    return out


@dataclass
class RecordFilter:
    """Composable predicate builder over ``FailureRecord`` streams.

    Example::

        flt = (RecordFilter()
               .categories("jailbreak", "prompt_injection")
               .min_severity("high")
               .min_score(0.8)
               .since(1_700_000_000))
        kept = list(flt.apply(source))
    """

    _predicates: list[Callable[[FailureRecord], bool]] = field(default_factory=list)

    _SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")

    def categories(self, *names: str) -> RecordFilter:
        wanted = {n.lower() for n in names}
        self._predicates.append(
            lambda r: (r.attack_category or "").lower() in wanted
        )
        return self

    def exclude_categories(self, *names: str) -> RecordFilter:
        unwanted = {n.lower() for n in names}
        self._predicates.append(
            lambda r: (r.attack_category or "").lower() not in unwanted
        )
        return self

    def min_severity(self, severity: str) -> RecordFilter:
        order = self._SEVERITY_ORDER
        floor = order.index(severity) if severity in order else 0

        def pred(r: FailureRecord) -> bool:
            rank = order.index(r.severity) if r.severity in order else 0
            return rank >= floor

        self._predicates.append(pred)
        return self

    def min_score(self, score: float) -> RecordFilter:
        self._predicates.append(lambda r: r.score is not None and r.score >= score)
        return self

    def since(self, timestamp: float) -> RecordFilter:
        self._predicates.append(lambda r: r.timestamp is not None and r.timestamp >= timestamp)
        return self

    def violations_only(self, flag: bool = True) -> RecordFilter:
        self._predicates.append(lambda r: r.was_violation == flag)
        return self

    def sources(self, *names: str) -> RecordFilter:
        wanted = set(names)
        self._predicates.append(lambda r: r.source in wanted)
        return self

    def where(self, predicate: Callable[[FailureRecord], bool]) -> RecordFilter:
        self._predicates.append(predicate)
        return self

    def matches(self, record: FailureRecord) -> bool:
        return all(p(record) for p in self._predicates)

    def apply(self, records: Iterable[FailureRecord]) -> Iterator[FailureRecord]:
        for r in records:
            if self.matches(r):
                yield r

    def __call__(self, record: FailureRecord) -> bool:
        return self.matches(record)


# ---- helpers ---------------------------------------------------------


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.lower() in ("true", "violation", "unsafe", "fail", "1", "yes")
    if isinstance(value, Iterable):
        try:
            return any(_truthy(v) for v in value)
        except Exception:
            return False
    return bool(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
