"""Tests for ``aisafepy pipeline`` (validate, describe, run, registry, bench)."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

from aisafepy.cli import pipeline as pipeline_cmd
from aisafepy.cli._io import CLIError

PIPELINE_YAML = """
version: 1
name: cli-test
budget_ms_p95: 80
tiers:
  1:
    - kind: regex_pii
      name: pii
      redact: true
    - kind: blocklist
      name: markers
      terms: ["FORBIDDEN"]
  2:
    - kind: stub_classifier
      name: stub
      triggers: ["HARMFUL"]
shadow:
  - kind: stub_classifier
    name: shadow-stub
    triggers: ["SUSPICIOUS"]
"""

BAD_YAML = """
version: 1
tiers:
  1:
    - kind: does_not_exist
    - kind: blocklist
"""


def invoke(*argv: str, stdin: str | None = None) -> tuple[int, str, str]:
    """Build a root parser like main.py does, register the module, run the handler."""
    parser = argparse.ArgumentParser(prog="aisafepy")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    sub = parser.add_subparsers(dest="command")
    pipeline_cmd.register(sub)
    args = parser.parse_args(list(argv))
    args.stdin = io.StringIO(stdin) if stdin is not None else None
    out, err = io.StringIO(), io.StringIO()
    try:
        code = args.handler(args, out, err)
    except CLIError as exc:
        err.write(str(exc))
        code = exc.exit_code
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(PIPELINE_YAML, encoding="utf-8")
    return p


# ---- validate ------------------------------------------------------------------


def test_validate_ok(config):
    code, out, _ = invoke("pipeline", "validate", str(config))
    assert code == 0
    assert out.strip().endswith("ok")


def test_validate_reports_problems_and_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(BAD_YAML, encoding="utf-8")
    code, out, _ = invoke("pipeline", "validate", str(bad))
    assert code == 2
    assert "does_not_exist" in out
    code, out, _ = invoke("--json", "pipeline", "validate", str(bad))
    payload = json.loads(out)
    assert payload["ok"] is False and len(payload["problems"]) >= 1


def test_validate_missing_file_is_an_error(tmp_path):
    code, _, err = invoke("pipeline", "validate", str(tmp_path / "nope.yaml"))
    assert code == 1
    assert "no such config" in err


# ---- describe ------------------------------------------------------------------


def test_describe_prints_structure(config):
    code, out, _ = invoke("pipeline", "describe", str(config))
    assert code == 0
    assert "cli-test" in out
    assert "tier 1" in out


def test_describe_json_round_trips_config(config):
    code, out, _ = invoke("--json", "pipeline", "describe", str(config))
    assert code == 0
    payload = json.loads(out)
    assert payload["pipeline"]["name"] == "cli-test"


# ---- run -------------------------------------------------------------------------


def test_run_allows_clean_text(config):
    code, out, _ = invoke("pipeline", "run", str(config), "--input", "hello there")
    assert code == 0
    assert out.startswith("allow")


def test_run_blocks_and_exits_2(config):
    code, out, _ = invoke("pipeline", "run", str(config), "--input", "this is FORBIDDEN")
    assert code == 2
    assert out.startswith("block")
    assert "markers" in out
    assert "rationale:" in out


def test_run_transform_shows_redacted_text(config):
    code, out, _ = invoke("pipeline", "run", str(config), "--input", "mail bob@example.com")
    assert code == 0
    assert "transformed:" in out
    assert "bob@example.com" not in out.split("transformed:")[1].splitlines()[0]


def test_run_json_payload_is_redacted_decision(config):
    code, out, _ = invoke(
        "--json", "pipeline", "run", str(config), "--input", "HARMFUL text", "--role", "user"
    )
    assert code == 2
    payload = json.loads(out)
    assert payload["flagged"] is True
    assert payload["decision"]["action"] == "block"
    assert payload["decision"]["tier"] == 2


def test_run_reads_stdin_and_metadata(config):
    code, out, _ = invoke(
        "pipeline", "run", str(config), "--input-file", "-", "--metadata", "tenant=a",
        stdin="plain text",
    )
    assert code == 0
    assert out.startswith("allow")


def test_run_rejects_bad_metadata(config):
    code, _, err = invoke("pipeline", "run", str(config), "--input", "x", "--metadata", "novalue")
    assert code == 1
    assert "KEY=VALUE" in err


# ---- registry -------------------------------------------------------------------


def test_registry_lists_kinds():
    code, out, _ = invoke("pipeline", "registry")
    assert code == 0
    assert "regex_pii" in out
    assert "blocklist" in out
    code, out, _ = invoke("--json", "pipeline", "registry")
    kinds = json.loads(out)["kinds"]
    assert "terms" in kinds["blocklist"]


# ---- bench ------------------------------------------------------------------------


def test_bench_reports_percentiles(config, tmp_path):
    samples = tmp_path / "samples.txt"
    samples.write_text("hello\n\nFORBIDDEN stuff\nmail a@b.co\n", encoding="utf-8")
    code, out, _ = invoke("pipeline", "bench", str(config), "--input-file", str(samples), "--repeat", "2")
    assert code == 0
    assert "6 evaluations (3 samples x 2)" in out
    assert "p95=" in out
    assert "block=2" in out


def test_bench_json_and_validation(config, tmp_path):
    samples = tmp_path / "samples.txt"
    samples.write_text("one\n", encoding="utf-8")
    code, out, _ = invoke("--json", "pipeline", "bench", str(config), "--input-file", str(samples))
    payload = json.loads(out)
    assert payload["evaluations"] == 5
    assert set(payload["latency_ms"]) >= {"p50", "p95", "p99"}
    assert "guards" in payload["metrics"]
    code, _, err = invoke("pipeline", "bench", str(config), "--input-file", str(samples), "--repeat", "0")
    assert code == 1 and "--repeat" in err
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    code, _, err = invoke("pipeline", "bench", str(config), "--input-file", str(empty))
    assert code == 1 and "no non-empty" in err
