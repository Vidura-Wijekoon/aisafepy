"""Train the pure-Python hashed n-gram classifier on the labelled dataset.

Splits ``injection_eval.jsonl`` into a stratified train / test set,
trains :class:`aisafepy.stream.hashed_classifier.HashedNGramClassifierGuard`
on the training half, and reports precision / recall / F1 / false-
positive rate on the held-out half together with training time and
per-call latency. This is the number to compare against
``bench_detection.py`` (Tier-1 alone) when deciding whether the
classifier earns its place in Tier 2.

The dataset is small (under 200 records), so the test split is noisy;
treat the metrics as a smoke signal, not a leaderboard entry.

Run::

    python benchmarks/bench_hashed_classifier.py
    python benchmarks/bench_hashed_classifier.py --json
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Record, base_arg_parser, confusion, load_dataset, print_report, time_guard


def stratified_split(records: list[Record], *, test_fraction: float, seed: int) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    train: list[Record] = []
    test: list[Record] = []
    for label in ("attack", "benign"):
        group = [r for r in records if r.label == label]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def run(quick: bool, seed: int = 0) -> dict:
    from aisafepy.stream.hashed_classifier import HashedNGramClassifierGuard

    records = load_dataset()
    train, test = stratified_split(records, test_fraction=0.3, seed=seed)
    guard = HashedNGramClassifierGuard(name="hashed-bench")
    t0 = time.perf_counter()
    report = guard.fit(
        [r.text for r in train if r.is_attack],
        [r.text for r in train if not r.is_attack],
        epochs=3 if quick else 15,
        seed=seed,
    )
    train_seconds = time.perf_counter() - t0

    preds = [guard.predict_proba(r.text) >= guard.threshold for r in test]
    labels = [r.is_attack for r in test]
    conf = confusion(preds, labels)
    timing = time_guard(guard, [r.text for r in test], repeats=1 if quick else 3, label="hashed")
    return {
        "n_train": len(train),
        "n_test": len(test),
        "epochs_run": report.epochs_run,
        "n_features": report.n_features,
        "train_seconds": round(train_seconds, 3),
        "validation": {"precision": round(report.precision, 4), "recall": round(report.recall, 4), "f1": round(report.f1, 4)},
        "test": conf.as_dict(),
        "latency": timing.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("hashed n-gram classifier: train on the labelled dataset and score the holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run(args.quick, seed=args.seed)
    test = result["test"]
    rows = [
        ["train records", result["n_train"]],
        ["test records", result["n_test"]],
        ["epochs run", result["epochs_run"]],
        ["non-zero features", result["n_features"]],
        ["train seconds", result["train_seconds"]],
        ["test precision", test["precision"]],
        ["test recall", test["recall"]],
        ["test f1", test["f1"]],
        ["test fpr", test["fpr"]],
        ["p50 ms / call", result["latency"]["p50_ms"]],
        ["p95 ms / call", result["latency"]["p95_ms"]],
    ]
    print_report("Hashed n-gram classifier on injection_eval.jsonl (30% holdout)", ["metric", "value"], rows, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
