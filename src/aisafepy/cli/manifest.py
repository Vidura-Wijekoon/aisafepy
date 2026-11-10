"""``aisafepy manifest``: pin and verify MCP tool manifests.

The MCP tool-poisoning defence in :mod:`aisafepy.flow.adapters.mcp`
only bites when a manifest is pinned. This command group makes that
a one-liner:

1. ``pin`` takes the JSON a server returns for ``tools/list`` (or any
   list of ``{name, description, inputSchema}`` objects), screens each
   description for hidden-instruction patterns, and writes a manifest
   with one SHA-256 per tool.
2. ``verify`` recomputes the hashes from a fresh ``tools/list`` and
   reports added, removed, and drifted tools; drift exits 2.
3. ``diff`` compares two manifests directly.

The manifest format is deliberately tiny so it can be committed next
to the agent configuration and reviewed in pull requests::

    {
      "version": 1,
      "generated_at": 1730000000.0,
      "tools": {"send_email": {"schema_hash": "..."}}
    }
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError
from aisafepy.flow.adapters.mcp import ToolManifestEntry, schema_hash, screen_description

MANIFEST_VERSION = 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "manifest",
        help="pin and verify MCP tool manifests (schema hashes + description screening)",
        description="Defend against MCP tool poisoning and silent schema drift.",
    )
    sub = p.add_subparsers(dest="manifest_command", metavar="SUBCOMMAND")
    sub.required = True

    pin = sub.add_parser("pin", help="hash every tool schema and screen descriptions")
    pin.add_argument("tools", metavar="TOOLS.json", help="tools/list output or a list of tools")
    pin.add_argument("-o", "--output", metavar="MANIFEST.json", help="where to write the manifest")
    pin.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 (and do not write) when any description fails screening",
    )
    pin.add_argument(
        "--extra-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="additional suspicious-description regex (repeatable)",
    )
    pin.set_defaults(handler=run_pin)

    verify = sub.add_parser("verify", help="compare live tools against a pinned manifest")
    verify.add_argument("tools", metavar="TOOLS.json")
    verify.add_argument("manifest", metavar="MANIFEST.json")
    verify.add_argument(
        "--allow-added",
        action="store_true",
        help="do not treat tools missing from the manifest as a failure",
    )
    verify.add_argument(
        "--allow-removed",
        action="store_true",
        help="do not treat pinned tools that disappeared as a failure",
    )
    verify.set_defaults(handler=run_verify)

    diff = sub.add_parser("diff", help="compare two manifests")
    diff.add_argument("left", metavar="A.json")
    diff.add_argument("right", metavar="B.json")
    diff.set_defaults(handler=run_diff)


# ---- parsing ---------------------------------------------------------------


def load_tools(path: str) -> list[dict[str, Any]]:
    """Accept ``[{...}]``, ``{"tools": [...]}`` or ``{"result": {"tools": [...]}}``."""
    data = _io.load_structured(path)
    if isinstance(data, dict):
        if "tools" in data:
            data = data["tools"]
        elif isinstance(data.get("result"), dict) and "tools" in data["result"]:
            data = data["result"]["tools"]
    if not isinstance(data, list):
        raise CLIError(f"{path}: expected a list of tools or an object with a 'tools' key")
    tools: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "name" not in item:
            raise CLIError(f"{path}: tool #{i} is not an object with a 'name'")
        tools.append(item)
    names = [t["name"] for t in tools]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise CLIError(f"{path}: duplicate tool names: {', '.join(dupes)}")
    return tools


def tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return the schema portion that gets hashed.

    We hash the description together with the input schema: a changed
    description is exactly the drift tool poisoning relies on.
    """
    schema = tool.get("inputSchema")
    if schema is None:
        schema = tool.get("parameters")
    if schema is None:
        schema = tool.get("schema", {})
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "inputSchema": schema,
    }


def build_manifest(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "generated_at": time.time(),
        "tools": {
            t["name"]: {"schema_hash": schema_hash(tool_schema(t))} for t in tools
        },
    }


def load_manifest(path: str) -> dict[str, str]:
    """Return ``{tool_name: schema_hash}`` from a manifest file."""
    data = _io.load_structured(path)
    if not isinstance(data, dict) or not isinstance(data.get("tools"), dict):
        raise CLIError(f"{path}: not a manifest (missing 'tools' object)")
    out: dict[str, str] = {}
    for name, entry in data["tools"].items():
        if isinstance(entry, str):
            out[name] = entry
        elif isinstance(entry, dict) and isinstance(entry.get("schema_hash"), str):
            out[name] = entry["schema_hash"]
        else:
            raise CLIError(f"{path}: tool {name!r} has no schema_hash")
    return out


def manifest_entries(path: str) -> list[ToolManifestEntry]:
    """Load a manifest as the dataclass list ``ifc_mcp_server(manifest=...)`` expects."""
    return [ToolManifestEntry(name=n, schema_hash=h) for n, h in load_manifest(path).items()]


def screen_tools(
    tools: list[dict[str, Any]], extra_patterns: list[str]
) -> dict[str, list[str]]:
    """Return ``{tool_name: [pattern, ...]}`` for tools whose description looks poisoned."""
    import re

    extra = []
    for raw in extra_patterns:
        try:
            extra.append(re.compile(raw, re.IGNORECASE | re.MULTILINE))
        except re.error as exc:
            raise CLIError(f"invalid --extra-pattern {raw!r}: {exc}") from exc
    hits: dict[str, list[str]] = {}
    for t in tools:
        description = str(t.get("description", ""))
        found = screen_description(t["name"], description)
        found += [p.pattern for p in extra if p.search(description)]
        if found:
            hits[t["name"]] = found
    return hits


def compare(live: dict[str, str], pinned: dict[str, str]) -> dict[str, list[str]]:
    """Classify tool names into added / removed / drifted / unchanged."""
    added = sorted(set(live) - set(pinned))
    removed = sorted(set(pinned) - set(live))
    common = set(live) & set(pinned)
    drifted = sorted(n for n in common if live[n] != pinned[n])
    unchanged = sorted(n for n in common if live[n] == pinned[n])
    return {"added": added, "removed": removed, "drifted": drifted, "unchanged": unchanged}


# ---- commands --------------------------------------------------------------


def run_pin(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    tools = load_tools(args.tools)
    hits = screen_tools(tools, args.extra_pattern)
    manifest = build_manifest(tools)
    blocked = bool(hits) and args.strict

    if not blocked and args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        import json

        Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.json:
        _io.dump_json(
            out,
            {
                "manifest": manifest,
                "screening_hits": hits,
                "written": (args.output if (args.output and not blocked) else None),
            },
        )
    else:
        rows = [
            [name, entry["schema_hash"][:16] + "...", "SUSPICIOUS" if name in hits else "ok"]
            for name, entry in manifest["tools"].items()
        ]
        _io.emit(out, _io.render_table(["tool", "schema_hash", "description"], rows))
        for name, patterns in hits.items():
            _io.emit(err, f"warning: {name}: description matched {patterns}")
        if not args.quiet:
            if blocked:
                _io.emit(out, "manifest NOT written: --strict and screening hits present")
            elif args.output:
                _io.emit(out, f"wrote {args.output} ({_io.human_count('tool', len(tools))})")
            else:
                _io.emit(out, f"{_io.human_count('tool', len(tools))} pinned (use -o to write)")
    return EXIT_FLAGGED if blocked else EXIT_OK


def run_verify(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    tools = load_tools(args.tools)
    live = {t["name"]: schema_hash(tool_schema(t)) for t in tools}
    pinned = load_manifest(args.manifest)
    result = compare(live, pinned)
    failures = list(result["drifted"])
    if not args.allow_added:
        failures += result["added"]
    if not args.allow_removed:
        failures += result["removed"]
    ok = not failures

    if args.json:
        _io.dump_json(out, {**result, "ok": ok})
    else:
        rows = (
            [[n, "drifted"] for n in result["drifted"]]
            + [[n, "added (not pinned)"] for n in result["added"]]
            + [[n, "removed (pinned but absent)"] for n in result["removed"]]
        )
        if rows:
            _io.emit(out, _io.render_table(["tool", "status"], rows))
        verdict = "OK" if ok else "FAIL"
        _io.emit(
            out,
            f"{verdict}: {len(result['unchanged'])} unchanged, {len(result['drifted'])} drifted, "
            f"{len(result['added'])} added, {len(result['removed'])} removed",
        )
    return EXIT_OK if ok else EXIT_FLAGGED


def run_diff(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    left = load_manifest(args.left)
    right = load_manifest(args.right)
    result = compare(right, left)  # "live" is the right-hand side
    same = not (result["added"] or result["removed"] or result["drifted"])
    if args.json:
        _io.dump_json(out, {**result, "identical": same})
    elif same:
        _io.emit(out, f"identical ({_io.human_count('tool', len(left))})")
    else:
        rows = (
            [[n, "changed"] for n in result["drifted"]]
            + [[n, f"only in {args.right}"] for n in result["added"]]
            + [[n, f"only in {args.left}"] for n in result["removed"]]
        )
        _io.emit(out, _io.render_table(["tool", "status"], rows))
    return EXIT_OK if same else EXIT_FLAGGED
