"""Detection quality of Tier-1 guards against the labelled dataset.

Runs a curated blocklist plus a set of injection-oriented regexes over
``injection_eval.jsonl`` and reports per-family recall alongside overall
precision, recall, F1, and false-positive rate. Tier-1 deterministic
guards are deliberately shallow: this benchmark exists to quantify
exactly how shallow, so the value a Tier-2 classifier adds is measurable
rather than assumed.

Use ``--guard package.module:factory`` to plug in a guard from another
module (for example a classifier wrapper) and fold its decisions into
the same report.

Run::

    python benchmarks/bench_detection.py
    python benchmarks/bench_detection.py --json
"""

from __future__ import annotations

import importlib
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    Record,
    base_arg_parser,
    confusion,
    load_dataset,
    print_report,
    run_guard,
)

# Curated Tier-1 signal. Literal phrases an attacker commonly uses, plus a
# few structural regexes. Intentionally does not try to catch encoded or
# homoglyph families: those are what Tier-2 is for, and the per-family
# recall makes that gap visible.
BLOCKLIST_PHRASES = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "disregard the system prompt",
    "forget your prior guidelines",
    "forget everything you were told",
    "developer mode",
    "do anything now",
    "without any restrictions",
    "reveal the system prompt",
    "print your system prompt",
    "repeat the text above",
    "stay in character",
    "unfiltered mode",
    "no content policy",
]

REGEX_PATTERNS = {
    "override": re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    "persona": re.compile(r"\b(you are|act as|pretend you are)\b.*\b(dan|unrestricted|jailbroken)\b", re.I),
    "exfil_prompt": re.compile(r"(repeat|print|reveal|output).{0,40}(system prompt|instructions)", re.I),
    "markdown_exfil": re.compile(r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*(data|token|secret|s=|k=)", re.I),
    "fake_delimiter": re.compile(r"</?(system|assistant|tool|input|context)\s*>", re.I),
}


def _build_default_guards():
    from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard

    blocklist = AhoCorasickGuard(name="phrase-blocklist", terms=BLOCKLIST_PHRASES)
    regexes = RegexGuard.from_patterns("injection-regex", REGEX_PATTERNS)
    return [blocklist, regexes]


def _load_extra_guard(spec: str):
    module_name, _, factory = spec.partition(":")
    if not factory:
        raise SystemExit(f"--guard must be 'module:factory', got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, factory)()


def predict(guards, record: Record) -> bool:
    """A record is predicted "attack" if any guard blocks or escalates it."""
    for guard in guards:
        decision = run_guard(guard, record.text)
        if decision.action.value in ("block", "escalate"):
            return True
    return False


def run(quick: bool, extra_guard: str | None = None):
    records = load_dataset()
    if quick:
        by_family = defaultdict(list)
        for r in records:
            by_family[(r.label, r.family)].append(r)
        records = [r for group in by_family.values() for r in group[:2]]

    guards = _build_default_guards()
    if extra_guard:
        guards.append(_load_extra_guard(extra_guard))

    preds = [predict(guards, r) for r in records]
    labels = [r.is_attack for r in records]
    overall = confusion(preds, labels)

    fam_recall: dict[str, tuple[int, int]] = {}
    for r, pred in zip(records, preds, strict=True):
        if not r.is_attack:
            continue
        caught, total = fam_recall.get(r.family, (0, 0))
        fam_recall[r.family] = (caught + int(pred), total + 1)

    return records, overall, fam_recall


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Tier-1 detection-quality benchmark")
    parser.add_argument("--guard", help="extra guard as 'package.module:factory'")
    args = parser.parse_args(argv)

    records, overall, fam_recall = run(args.quick, args.guard)
    fam_rows = [
        [fam, f"{caught}/{total}", f"{caught / total:.0%}" if total else "-"]
        for fam, (caught, total) in sorted(fam_recall.items())
    ]

    if args.json:
        print_report(
            "detection",
            ["metric", "value"],
            [["records", len(records)], *[[k, v] for k, v in overall.as_dict().items()],
             *[["recall:" + f, f"{c}/{t}"] for f, (c, t) in sorted(fam_recall.items())]],
            as_json=True,
        )
        return 0

    print(f"Detection over {len(records)} records")
    print(
        f"  precision={overall.precision:.3f} recall={overall.recall:.3f} "
        f"f1={overall.f1:.3f} fpr={overall.fpr:.3f} accuracy={overall.accuracy:.3f}"
    )
    print()
    print_report("Per-family recall (attacks only)", ["family", "caught", "recall"], fam_rows, as_json=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
