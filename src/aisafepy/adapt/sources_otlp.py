"""Sources that read AIsafePy's own telemetry back into the compiler.

Every guard decision becomes an OpenTelemetry span with
``aisafepy.guard.*`` attributes (:mod:`aisafepy.core.telemetry`) and,
optionally, a JSONL record (:mod:`aisafepy.core.sinks`). Those two
streams are the cheapest production signal there is: no labelling
pipeline, no red-team run, just what the guards already decided. This
module turns them into :class:`~aisafepy.adapt.sources.FailureRecord`
objects so blocked production traffic can be clustered and compiled
into tighter guards.

:class:`OTLPSpanSource` reads the OTLP/JSON export format
(``resourceSpans[].scopeSpans[].spans[]`` with typed attribute values),
which is what the OTel collector's file exporter, Jaeger's JSON dump
and most vendor "download trace" buttons produce.
:class:`DecisionLogSource` reads the JSONL written by
:class:`aisafepy.core.sinks.JSONLSink`.

A record's ``input`` is the text the guard looked at. Spans only carry
it when the application attached ``aisafepy.input``, ``gen_ai.prompt``
or ``gen_ai.completion``; otherwise the redacted evidence JSON is used
so clustering still has something to work with. Pass ``text_from`` to
override that choice.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from aisafepy.adapt.sources import FailureRecord

GUARD_PREFIX = "aisafepy.guard."
_TEXT_KEYS = ("aisafepy.input", "gen_ai.prompt", "gen_ai.input", "gen_ai.completion", "gen_ai.output")


def unwrap_attribute_value(value: Any) -> Any:
    """Collapse an OTLP ``AnyValue`` into a plain Python value."""
    if not isinstance(value, Mapping):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        try:
            return float(value["doubleValue"])
        except (TypeError, ValueError):
            return value["doubleValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        values = value["arrayValue"].get("values", []) if isinstance(value["arrayValue"], Mapping) else []
        return [unwrap_attribute_value(v) for v in values]
    if "kvlistValue" in value:
        items = value["kvlistValue"].get("values", []) if isinstance(value["kvlistValue"], Mapping) else []
        return {str(i.get("key")): unwrap_attribute_value(i.get("value")) for i in items if isinstance(i, Mapping)}
    if "bytesValue" in value:
        return value["bytesValue"]
    return dict(value)


def span_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    """Return a span's attributes as a flat dict (list or dict input shapes)."""
    raw = span.get("attributes")
    out: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            out[str(k)] = unwrap_attribute_value(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and "key" in item:
                out[str(item["key"])] = unwrap_attribute_value(item.get("value"))
    return out


def iter_otlp_spans(doc: Any) -> Iterator[Mapping[str, Any]]:
    """Yield span dicts from an OTLP/JSON document or a plain list of spans."""
    if isinstance(doc, list):
        for item in doc:
            if isinstance(item, Mapping):
                if "resourceSpans" in item or "scopeSpans" in item:
                    yield from iter_otlp_spans(item)
                else:
                    yield item
        return
    if not isinstance(doc, Mapping):
        return
    if "spans" in doc and isinstance(doc["spans"], list) and "scopeSpans" not in doc:
        yield from (s for s in doc["spans"] if isinstance(s, Mapping))
        return
    for rs in doc.get("resourceSpans", [doc] if "scopeSpans" in doc else []):
        if not isinstance(rs, Mapping):
            continue
        scopes = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
        for ss in scopes:
            if not isinstance(ss, Mapping):
                continue
            for span in ss.get("spans", []):
                if isinstance(span, Mapping):
                    yield span


def _load_doc(path_or_obj: Any) -> Any:
    if isinstance(path_or_obj, (str, Path)):
        return json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    return path_or_obj


def _evidence_text(evidence: Any) -> str:
    if isinstance(evidence, str):
        return evidence
    try:
        return json.dumps(evidence, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(evidence)


def _default_span_text(attrs: Mapping[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return _evidence_text(attrs.get(GUARD_PREFIX + "evidence", ""))


class OTLPSpanSource:
    """Yield ``FailureRecord`` objects from an OTLP/JSON trace export.

    Parameters
    ----------
    path_or_obj:
        File path or an already-parsed document.
    only_violations:
        Skip spans whose ``aisafepy.guard.action`` is ``allow``.
    text_from:
        ``callable(attributes) -> str`` overriding how the record's
        ``input`` is chosen.
    """

    def __init__(
        self,
        path_or_obj: Any,
        *,
        only_violations: bool = True,
        text_from: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        self.path_or_obj = path_or_obj
        self.only_violations = only_violations
        self.text_from = text_from or _default_span_text

    def __iter__(self) -> Iterator[FailureRecord]:
        doc = _load_doc(self.path_or_obj)
        for i, span in enumerate(iter_otlp_spans(doc)):
            attrs = span_attributes(span)
            action = attrs.get(GUARD_PREFIX + "action")
            if action is None:
                continue
            action = str(action)
            was_violation = action != "allow"
            if self.only_violations and not was_violation:
                continue
            guard_name = attrs.get(GUARD_PREFIX + "name")
            decision_id = attrs.get(GUARD_PREFIX + "decision_id")
            span_id = span.get("spanId") or span.get("span_id")
            output = ""
            for key in ("gen_ai.completion", "gen_ai.output"):
                if isinstance(attrs.get(key), str):
                    output = attrs[key]
                    break
            yield FailureRecord(
                id=str(decision_id or span_id or f"otlp-{i}"),
                input=self.text_from(attrs),
                output=output,
                was_violation=was_violation,
                attack_category=str(guard_name) if guard_name else None,
                severity=str(attrs.get(GUARD_PREFIX + "severity", "medium")),
                scorer=str(guard_name) if guard_name else None,
                score=_as_float(attrs.get(GUARD_PREFIX + "confidence")),
                timestamp=_span_timestamp(span),
                source="otlp",
                metadata={
                    "action": action,
                    "rationale": attrs.get(GUARD_PREFIX + "rationale"),
                    "tier": attrs.get(GUARD_PREFIX + "tier"),
                    "guard_name": guard_name,
                    "decision_id": decision_id,
                    "span_name": span.get("name"),
                    "trace_id": span.get("traceId") or span.get("trace_id"),
                    "span_id": span_id,
                    "evidence": attrs.get(GUARD_PREFIX + "evidence"),
                },
            )


def _default_record_text(record: Mapping[str, Any]) -> str:
    extra = record.get("extra") or {}
    if isinstance(extra, Mapping):
        for key in ("input", "text", "prompt", "buffer", "content"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return _evidence_text(record.get("evidence", ""))


class DecisionLogSource:
    """Yield ``FailureRecord`` objects from a ``JSONLSink`` decision log.

    One JSON object per line in the shape produced by
    :func:`aisafepy.core.sinks.decision_to_record`. Blank and malformed
    lines are skipped and counted in :attr:`skipped`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        only_violations: bool = True,
        text_from: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.only_violations = only_violations
        self.text_from = text_from or _default_record_text
        self.skipped = 0

    def __iter__(self) -> Iterator[FailureRecord]:
        self.skipped = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.skipped += 1
                    continue
                if not isinstance(record, Mapping) or "action" not in record:
                    self.skipped += 1
                    continue
                action = str(record["action"])
                was_violation = action != "allow"
                if self.only_violations and not was_violation:
                    continue
                guard_name = record.get("guard_name")
                extra = record.get("extra") or {}
                output = ""
                if isinstance(extra, Mapping) and isinstance(extra.get("output"), str):
                    output = extra["output"]
                yield FailureRecord(
                    id=str(record.get("decision_id") or f"decision-{i}"),
                    input=self.text_from(record),
                    output=output,
                    was_violation=was_violation,
                    attack_category=str(guard_name) if guard_name else None,
                    severity=str(record.get("severity", "medium")),
                    scorer=str(guard_name) if guard_name else None,
                    score=_as_float(record.get("confidence")),
                    timestamp=_as_float(record.get("timestamp")),
                    source="decision_log",
                    metadata={
                        "action": action,
                        "rationale": record.get("rationale"),
                        "tier": record.get("tier"),
                        "guard_name": guard_name,
                        "decision_id": record.get("decision_id"),
                        "evidence": record.get("evidence"),
                        "extra": dict(extra) if isinstance(extra, Mapping) else extra,
                        "schema_version": record.get("schema_version"),
                    },
                )


# ---- helpers ----------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _span_timestamp(span: Mapping[str, Any]) -> float | None:
    for key in ("startTimeUnixNano", "start_time_unix_nano"):
        raw = span.get(key)
        if raw is None:
            continue
        try:
            return int(raw) / 1e9
        except (TypeError, ValueError):
            return None
    return None
