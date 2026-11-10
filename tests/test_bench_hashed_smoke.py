"""Smoke test for the hashed-classifier benchmark script."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, BENCH_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_bench_hashed_classifier_quick_runs_and_learns_something():
    bench = _load("bench_hashed_classifier", "bench_hashed_classifier.py")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert bench.main(["--quick", "--json"]) == 0
    payload = json.loads(buf.getvalue())
    rows = {r[0]: r[1] for r in payload["rows"]}
    assert rows["train records"] + rows["test records"] >= 160
    assert rows["non-zero features"] > 0
    # A bag-of-ngrams model should beat chance on this split even with 3 epochs.
    assert float(rows["test f1"]) > 0.5


def test_bench_hashed_classifier_table_output():
    bench = _load("bench_hashed_classifier_tbl", "bench_hashed_classifier.py")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert bench.main(["--quick"]) == 0
    out = buf.getvalue()
    assert "Hashed n-gram classifier" in out
    assert "test f1" in out


def test_stratified_split_keeps_both_classes():
    bench = _load("bench_hashed_classifier_split", "bench_hashed_classifier.py")
    common = _load("common", "common.py")
    records = common.load_dataset()
    train, test = bench.stratified_split(records, test_fraction=0.3, seed=1)
    assert len(train) + len(test) == len(records)
    assert any(r.is_attack for r in test) and any(not r.is_attack for r in test)
    assert not ({r.id for r in train} & {r.id for r in test})
