"""Smoke tests for the offline benchmark suite.

Validates the labelled dataset and runs each benchmark's ``main(["--quick"])``
with tiny sizes so a broken script is caught in CI without needing a GPU
or network access. The whole module runs in a couple of seconds.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DATASET = BENCH_DIR / "datasets" / "injection_eval.jsonl"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, BENCH_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's types.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_common():
    return _load("common", "common.py")


def load_bench(name: str):
    return _load(name, f"{name}.py")


# ---- dataset ---------------------------------------------------------------


def test_dataset_structure_and_balance():
    common = load_common()
    records = common.load_dataset(DATASET)
    assert len(records) >= 160

    ids = [r.id for r in records]
    assert len(ids) == len(set(ids)), "ids must be unique"
    assert all(r.label in {"attack", "benign"} for r in records)

    by_label = Counter(r.label for r in records)
    assert by_label["attack"] >= 70
    assert by_label["benign"] >= 60

    attack_families = {r.family for r in records if r.label == "attack"}
    for expected in (
        "instruction_override",
        "role_hijack",
        "system_prompt_exfil",
        "delimiter_injection",
        "tool_poisoning",
        "encoded_payload",
        "obfuscation",
        "many_shot",
        "data_exfil_markdown",
    ):
        assert expected in attack_families, f"missing attack family {expected}"


def test_encoded_payloads_decode_to_claimed_phrase():
    common = load_common()
    records = common.load_dataset(DATASET)
    encoded = [r for r in records if r.extra.get("encoding") in {"base64", "hex"}]
    assert len(encoded) >= 6
    for rec in encoded:
        payload = rec.text.split()[-1]
        claimed = rec.extra["decodes_to"]
        if rec.extra["encoding"] == "base64":
            decoded = base64.b64decode(payload).decode()
        else:
            decoded = bytes.fromhex(payload).decode()
        assert decoded == claimed, f"{rec.id}: {decoded!r} != {claimed!r}"


def test_dataset_rejects_duplicate_ids(tmp_path: Path):
    common = load_common()
    bad = tmp_path / "dupe.jsonl"
    bad.write_text(
        '{"id": "x", "text": "a", "label": "attack", "family": "f"}\n'
        '{"id": "x", "text": "b", "label": "benign", "family": "f"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        common.load_dataset(bad)


# ---- metrics ---------------------------------------------------------------


def test_confusion_metrics():
    common = load_common()
    c = common.confusion([True, True, False, False], [True, False, True, False])
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.precision == 0.5
    assert c.recall == 0.5
    assert c.f1 == 0.5
    assert c.fpr == 0.5


def test_percentiles_and_timing_shape():
    common = load_common()

    class Dummy:
        name = "dummy"

        async def __call__(self, ctx):
            from aisafepy.core.decisions import GuardDecision

            return GuardDecision.allow(tier=1)

    timing = common.time_guard(Dummy(), ["a", "b", "c"], repeats=2)
    assert timing.count == 3
    assert timing.repeats == 2
    assert timing.p95_ms >= timing.p50_ms >= 0.0
    assert timing.throughput_per_s > 0


# ---- each benchmark runs ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["bench_tier1_latency", "bench_detection", "bench_flow_overhead", "bench_budgets"],
)
def test_bench_runs_quick(name: str):
    module = load_bench(name)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = module.main(["--quick"])
    assert rc == 0
    assert buf.getvalue().strip(), "benchmark produced no output"


@pytest.mark.parametrize(
    "name",
    ["bench_tier1_latency", "bench_detection", "bench_flow_overhead", "bench_budgets"],
)
def test_bench_json_output(name: str):
    import json

    module = load_bench(name)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = module.main(["--quick", "--json"])
    assert rc == 0
    json.loads(buf.getvalue().strip().splitlines()[-1])


def test_detection_recall_is_reasonable():
    """The curated Tier-1 guards should catch the obvious override family."""
    module = load_bench("bench_detection")
    _records, _overall, fam_recall = module.run(quick=False)
    caught, total = fam_recall["instruction_override"]
    assert total > 0
    assert caught / total >= 0.5
