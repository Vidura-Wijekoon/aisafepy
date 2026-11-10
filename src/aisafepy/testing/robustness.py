"""Robustness reports: how a guard fares against mutated attacks.

A detection rate measured on canonical attack strings overstates what a
guard does in production. :func:`robustness_report` applies each
mutation strategy to every attack *and* every benign sample, then
reports per strategy:

* recall on attacks (what fraction of mutated attacks still got flagged),
* false-positive rate on benign text (whether the transformation itself
  makes the guard over-trigger, which matters for encodings and markup),
* the concrete evading variants, so a failing strategy can be turned
  into a regression test in one copy-paste.

:func:`coverage_matrix` answers the complementary question for a set of
guards: which guard catches which attack family, so a cascade can be
assembled with no family left uncovered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aisafepy.testing.assertions import run_guard
from aisafepy.testing.corpus import Sample
from aisafepy.testing.mutations import DEFAULT_STRATEGIES, Variant, mutate, strategy_key


@dataclass
class Evasion:
    """A mutated attack the guard did not flag."""

    sample_id: str
    family: str | None
    strategy: str
    text: str
    action: str
    rationale: str


@dataclass
class StrategyResult:
    strategy: str
    evades: str
    attacks_total: int = 0
    attacks_caught: int = 0
    benign_total: int = 0
    benign_flagged: int = 0
    not_applicable: int = 0
    evasions: list[Evasion] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.attacks_caught / self.attacks_total if self.attacks_total else 0.0

    @property
    def fpr(self) -> float:
        return self.benign_flagged / self.benign_total if self.benign_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "evades": self.evades,
            "attacks_total": self.attacks_total,
            "attacks_caught": self.attacks_caught,
            "recall": round(self.recall, 4),
            "benign_total": self.benign_total,
            "benign_flagged": self.benign_flagged,
            "fpr": round(self.fpr, 4),
            "not_applicable": self.not_applicable,
            "evasions": len(self.evasions),
        }


@dataclass
class RobustnessReport:
    guard_name: str
    baseline: StrategyResult
    per_strategy: list[StrategyResult]
    flag_actions: tuple[str, ...]

    def weakest_strategies(self, n: int = 3) -> list[StrategyResult]:
        """Strategies with the lowest attack recall (ties broken by FPR, high first)."""
        return sorted(self.per_strategy, key=lambda r: (r.recall, -r.fpr))[:n]

    def evasions(self) -> list[Evasion]:
        out: list[Evasion] = list(self.baseline.evasions)
        for r in self.per_strategy:
            out.extend(r.evasions)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard": self.guard_name,
            "flag_actions": list(self.flag_actions),
            "baseline": self.baseline.to_dict(),
            "strategies": [r.to_dict() for r in self.per_strategy],
            "evasions": [vars(e) for e in self.evasions()],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Robustness report: {self.guard_name}",
            "",
            f"Flagging actions: {', '.join(self.flag_actions)}",
            "",
            "| strategy | attack recall | benign FPR | evasions | evades |",
            "|---|---:|---:|---:|---|",
            _row(self.baseline),
        ]
        for r in self.per_strategy:
            lines.append(_row(r))
        weakest = self.weakest_strategies(3)
        if weakest:
            lines += ["", "## Weakest strategies", ""]
            for r in weakest:
                lines.append(f"- `{r.strategy}`: recall {r.recall:.0%}, {len(r.evasions)} evasions")
        evasions = self.evasions()
        if evasions:
            lines += ["", "## Sample evasions", ""]
            for e in evasions[:10]:
                preview = e.text if len(e.text) <= 100 else e.text[:97] + "..."
                lines.append(f"- `{e.strategy}` on `{e.sample_id}` ({e.action}): {preview!r}")
        return "\n".join(lines)


def _row(r: StrategyResult) -> str:
    return (
        f"| `{r.strategy}` | {r.recall:.0%} ({r.attacks_caught}/{r.attacks_total}) "
        f"| {r.fpr:.0%} ({r.benign_flagged}/{r.benign_total}) | {len(r.evasions)} | {r.evades} |"
    )


def _flagged(decision: Any, flag_actions: Sequence[str]) -> bool:
    return decision.action.value in flag_actions


def robustness_report(
    guard: Any,
    samples: Iterable[Sample],
    *,
    strategies: Iterable[Any] = DEFAULT_STRATEGIES,
    seed: int = 0,
    flag_actions: Sequence[str] = ("block", "escalate"),
    role: str = "user",
    max_evasions_per_strategy: int = 50,
) -> RobustnessReport:
    """Measure ``guard`` against every sample and every mutation of it."""
    samples = list(samples)
    strategies = tuple(strategies)
    name = getattr(guard, "name", type(guard).__name__)
    flags = tuple(flag_actions)

    baseline = StrategyResult(strategy="baseline", evades="(unmutated text)")
    results: dict[str, StrategyResult] = {
        strategy_key(s): StrategyResult(strategy=strategy_key(s), evades=s.evades) for s in strategies
    }

    for sample in samples:
        decision = run_guard(guard, sample.text, role=role)
        _tally(baseline, sample, sample.text, decision, flags, max_evasions_per_strategy)
        variants: list[Variant] = mutate(sample.text, strategies=strategies, seed=seed)
        seen = {v.strategy for v in variants}
        for key, res in results.items():
            if key not in seen:
                res.not_applicable += 1
        for v in variants:
            res = results[v.strategy]
            decision = run_guard(guard, v.text, role=role)
            _tally(res, sample, v.text, decision, flags, max_evasions_per_strategy)

    return RobustnessReport(
        guard_name=name,
        baseline=baseline,
        per_strategy=list(results.values()),
        flag_actions=flags,
    )


def _tally(
    res: StrategyResult,
    sample: Sample,
    text: str,
    decision: Any,
    flags: Sequence[str],
    cap: int,
) -> None:
    flagged = _flagged(decision, flags)
    if sample.is_attack:
        res.attacks_total += 1
        if flagged:
            res.attacks_caught += 1
        elif len(res.evasions) < cap:
            res.evasions.append(
                Evasion(
                    sample_id=sample.id,
                    family=sample.family,
                    strategy=res.strategy,
                    text=text,
                    action=decision.action.value,
                    rationale=decision.rationale,
                )
            )
    else:
        res.benign_total += 1
        if flagged:
            res.benign_flagged += 1


# --- coverage matrix ------------------------------------------------------------


@dataclass
class CoverageMatrix:
    """Per-guard, per-family catch counts."""

    guards: list[str]
    families: list[str]
    caught: dict[str, dict[str, int]]
    totals: dict[str, int]
    benign_flagged: dict[str, int]
    benign_total: int

    def recall(self, guard: str, family: str) -> float:
        t = self.totals.get(family, 0)
        return self.caught[guard].get(family, 0) / t if t else 0.0

    def uncovered_families(self, threshold: float = 0.5) -> list[str]:
        """Families no single guard catches at or above ``threshold`` recall."""
        return [
            f for f in self.families if all(self.recall(g, f) < threshold for g in self.guards)
        ]

    def best_guard(self, family: str) -> str | None:
        if not self.guards:
            return None
        return max(self.guards, key=lambda g: self.recall(g, family))

    def to_dict(self) -> dict[str, Any]:
        return {
            "guards": self.guards,
            "families": self.families,
            "recall": {g: {f: round(self.recall(g, f), 4) for f in self.families} for g in self.guards},
            "benign_fpr": {
                g: round(self.benign_flagged[g] / self.benign_total, 4) if self.benign_total else 0.0
                for g in self.guards
            },
            "uncovered_families": self.uncovered_families(),
        }

    def to_markdown(self) -> str:
        header = "| guard | " + " | ".join(self.families) + " | benign FPR |"
        sep = "|---|" + "---:|" * (len(self.families) + 1)
        rows = [header, sep]
        for g in self.guards:
            cells = [f"{self.recall(g, f):.0%}" for f in self.families]
            fpr = self.benign_flagged[g] / self.benign_total if self.benign_total else 0.0
            rows.append(f"| `{g}` | " + " | ".join(cells) + f" | {fpr:.0%} |")
        return "\n".join(rows)


def coverage_matrix(
    guards: Mapping[str, Any] | Sequence[Any],
    samples: Iterable[Sample],
    *,
    flag_actions: Sequence[str] = ("block", "escalate"),
    role: str = "user",
) -> CoverageMatrix:
    """Which guard catches which attack family on the given samples."""
    if isinstance(guards, Mapping):
        named = dict(guards)
    else:
        named = {getattr(g, "name", f"guard{i}"): g for i, g in enumerate(guards)}
    samples = list(samples)
    families = sorted({s.family for s in samples if s.is_attack and s.family})
    totals = {f: sum(1 for s in samples if s.family == f and s.is_attack) for f in families}
    caught = {g: dict.fromkeys(families, 0) for g in named}
    benign_flagged = dict.fromkeys(named, 0)
    benign_total = sum(1 for s in samples if not s.is_attack)
    for gname, guard in named.items():
        for s in samples:
            decision = run_guard(guard, s.text, role=role)
            if not _flagged(decision, flag_actions):
                continue
            if s.is_attack and s.family:
                caught[gname][s.family] += 1
            elif not s.is_attack:
                benign_flagged[gname] += 1
    return CoverageMatrix(
        guards=list(named),
        families=families,
        caught=caught,
        totals=totals,
        benign_flagged=benign_flagged,
        benign_total=benign_total,
    )
