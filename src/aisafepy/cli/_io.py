"""Shared I/O helpers for the ``aisafepy`` command-line interface.

Everything here is deliberately boring: reading text from a path or
stdin, loading JSON / YAML by extension, rendering fixed-width ASCII
tables, and printing JSON with a permissive ``default``. The command
modules stay focused on wiring library APIs to arguments.

Output is kept ASCII-only so the CLI renders correctly on a Windows
console running the cp1252 code page.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any

# ---- exit codes ----------------------------------------------------------

EXIT_OK = 0
"""The command completed and nothing was flagged."""

EXIT_ERROR = 1
"""Usage error, unreadable input, or an unexpected failure."""

EXIT_FLAGGED = 2
"""The command completed but a guard blocked, a chain failed to verify,
or a manifest drifted. Scripts can branch on this code."""


class CLIError(Exception):
    """A user-facing error. The message is printed on one line and the
    process exits with :data:`EXIT_ERROR`."""

    def __init__(self, message: str, *, exit_code: int = EXIT_ERROR):
        super().__init__(message)
        self.exit_code = exit_code


# ---- streams -------------------------------------------------------------


def out_stream(stream: IO[str] | None) -> IO[str]:
    """Return ``stream`` or the current ``sys.stdout``.

    Resolved lazily so pytest's capture and callers that pass their own
    ``io.StringIO`` both work.
    """
    return stream if stream is not None else sys.stdout


def err_stream(stream: IO[str] | None) -> IO[str]:
    """Return ``stream`` or the current ``sys.stderr``."""
    return stream if stream is not None else sys.stderr


def emit(stream: IO[str], text: str = "") -> None:
    """Write one line, replacing characters the stream cannot encode.

    A ``StringIO`` has no encoding and accepts everything; a Windows
    console might not. We never want a stray non-ASCII character in a
    scanned file to crash the report.
    """
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    stream.write(text + "\n")


# ---- reading input -------------------------------------------------------


def read_text(path: str | Path, *, stdin: IO[str] | None = None) -> str:
    """Read UTF-8 text from ``path``; ``"-"`` reads from stdin.

    Raises :class:`CLIError` when the path does not exist or the file is
    not decodable as text.
    """
    if str(path) == "-":
        source = stdin if stdin is not None else sys.stdin
        return source.read()
    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such file: {p}")
    if p.is_dir():
        raise CLIError(f"expected a file, got a directory: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CLIError(f"cannot decode {p} as UTF-8 text: {exc.reason}") from exc


def is_probably_binary(data: bytes) -> bool:
    """Cheap binary sniff: a NUL byte in the first 8 KiB means binary."""
    return b"\x00" in data[:8192]


def load_structured(path: str | Path, *, stdin: IO[str] | None = None) -> Any:
    """Load a JSON or YAML document, choosing the parser by extension.

    ``.yaml`` / ``.yml`` require PyYAML; everything else is parsed as
    JSON. Parse errors are turned into :class:`CLIError`.
    """
    text = read_text(path, stdin=stdin)
    suffix = Path(str(path)).suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise CLIError("reading YAML requires PyYAML: pip install pyyaml") from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CLIError(f"invalid YAML in {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLIError(f"invalid JSON in {path}: {exc.msg} (line {exc.lineno})") from exc


def iter_jsonl(path: str | Path, *, stdin: IO[str] | None = None) -> Iterable[dict[str, Any]]:
    """Yield one parsed object per non-empty line of a JSON Lines file."""
    text = read_text(path, stdin=stdin)
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise CLIError(f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}")
        yield obj


def parse_key_values(items: Sequence[str] | None) -> dict[str, str]:
    """Turn ``["k=v", "a=b"]`` into a dict; reject entries without ``=``."""
    out: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise CLIError(f"expected KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise CLIError(f"empty key in {item!r}")
        out[key] = value
    return out


# ---- rendering -----------------------------------------------------------


def truncate(value: Any, width: int) -> str:
    """Render ``value`` as a single line no longer than ``width``."""
    text = str(value).replace("\n", " ").replace("\r", "")
    if width <= 3:
        return text[:width]
    if len(text) > width:
        return text[: width - 3] + "..."
    return text


def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    max_width: int = 48,
) -> str:
    """Render a plain fixed-width table.

    Columns are sized to their widest cell, capped at ``max_width``;
    longer cells are truncated with an ellipsis made of three dots. The
    header is underlined with hyphens. Only ASCII characters are used
    for the frame.
    """
    materialised = [[truncate(c, max_width) for c in row] for row in rows]
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in materialised:
        for i in range(min(cols, len(row))):
            widths[i] = max(widths[i], len(row[i]))

    def fmt(cells: Sequence[str]) -> str:
        padded = []
        for i in range(cols):
            cell = cells[i] if i < len(cells) else ""
            padded.append(cell.ljust(widths[i]))
        return "  ".join(padded).rstrip()

    lines = [fmt(list(headers)), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in materialised)
    return "\n".join(lines)


def dump_json(stream: IO[str], payload: Any, *, indent: int | None = 2) -> None:
    """Print ``payload`` as JSON. Unknown types fall back to ``str``."""
    emit(stream, json.dumps(payload, indent=indent, default=_json_default, sort_keys=False))


def _json_default(value: Any) -> Any:
    if isinstance(value, set | frozenset):
        return sorted(str(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    return str(value)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile over ``values``; ``0.0`` for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(rank, len(ordered) - 1))]


def human_count(label: str, n: int) -> str:
    """``"1 file"`` / ``"3 files"`` / ``"3 entries"``: pluralisation for summaries."""
    if n == 1:
        return f"{n} {label}"
    if label.endswith("y") and label[-2:-1] not in "aeiou":
        return f"{n} {label[:-1]}ies"
    return f"{n} {label}s"
