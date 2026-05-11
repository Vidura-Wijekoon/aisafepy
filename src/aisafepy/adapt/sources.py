"""Adapters for red-team frameworks and production trace sources.

Each adapter normalizes its native record format into a
:class:`FailureRecord`. The compiler only sees ``FailureRecord``s, so
new sources (Promptfoo, DeepEval, internal QA tools) plug in with a
~30-line shim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
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


# ---- helpers ---------------------------------------------------------


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
