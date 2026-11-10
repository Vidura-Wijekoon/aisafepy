"""Agent loop / repetition / no-progress detection.

Re-implements the small set of heuristics that every agent framework
ends up re-inventing (browser-use#191, AgentBudget, Modexa's *Agent Loop
Problem*). The detectors are intentionally cheap and side-effect-free:
they classify a stream of "steps" and emit a ``LoopDetected`` decision
when one of the triggers fires.

A "step" here is whatever your agent treats as a unit of work. Usually
a tool call (name + canonicalized arguments) plus optionally a textual
observation. The library does not assume a specific agent framework.

Detectors, cheapest first:

* **max steps**: a hard cap on the total number of steps observed.
* **duplicate tool call**: the same (tool, args) fingerprint seen
  ``duplicate_threshold`` times within the window. Fingerprints can
  ignore volatile arguments (``ignore_args``) and normalize string
  case/whitespace so ``{"q": "Weather "}`` and ``{"q": "weather"}``
  count as the same call.
* **cyclic tool calls**: A-B-A-B or A-B-C-A-B-C patterns.
* **stagnation**: identical observations for ``stagnation_window`` steps.
* **semantic no-progress**: a run of high similarity scores, either
  supplied by the caller (embeddings) or computed here from character
  shingles via :meth:`ProgressTracker.observe_text`.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aisafepy.core.decisions import GuardDecision


class RepetitionReason(str, Enum):
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    """The exact same (tool, args) was issued ``threshold`` times within the window."""

    CYCLIC_TOOL_CALLS = "cyclic_tool_calls"
    """A 2- or 3-step cycle of tool calls was detected."""

    STAGNATION = "stagnation"
    """The agent's observable state has not changed across ``stagnation_window`` steps."""

    SEMANTIC_NO_PROGRESS = "semantic_no_progress"
    """A user-supplied semantic similarity score above ``semantic_threshold``."""

    MAX_STEPS = "max_steps"
    """The hard cap on total steps was reached."""


class LoopDetected(GuardDecision):
    """A specialized ``GuardDecision`` for the loop-detector.

    Built on top of ``GuardDecision`` so it composes with every other
    AIsafePy primitive (e.g. a ``GuardPipeline`` can take a
    ``ProgressTracker``-driven guard as one of its tiers).
    """


@dataclass(frozen=True)
class Step:
    """A canonical step. Free-form ``extra`` is for downstream introspection."""

    tool: str
    args: dict[str, Any]
    observation: str | None = None
    extra: dict[str, Any] | None = None

    def fingerprint(
        self,
        *,
        ignore_args: Iterable[str] = (),
        normalize_strings: bool = False,
    ) -> str:
        """A stable hash over (tool, args) used for cycle detection.

        ``ignore_args`` drops volatile keys (timestamps, request ids)
        before hashing; ``normalize_strings`` lowercases and collapses
        whitespace in string values so trivially different phrasings
        collide.
        """
        args = _canonical_args(self.args, ignore=set(ignore_args), normalize=normalize_strings)
        payload = json.dumps([self.tool, args], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sort_args(args: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize argument order and nested dict ordering for hashing."""
    return _canonical_args(args, ignore=set(), normalize=False)


def _canonical_args(args: dict[str, Any], *, ignore: set[str], normalize: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted(args):
        if k in ignore:
            continue
        v = args[k]
        if isinstance(v, dict):
            out[k] = _canonical_args(v, ignore=ignore, normalize=normalize)
        elif isinstance(v, list | tuple):
            out[k] = [_canonical_value(x, normalize) for x in v]
        else:
            out[k] = _canonical_value(v, normalize)
    return out


def _canonical_value(v: Any, normalize: bool) -> Any:
    if isinstance(v, dict):
        return _canonical_args(v, ignore=set(), normalize=normalize)
    if normalize and isinstance(v, str):
        return " ".join(v.lower().split())
    return v


def shingles(text: str, n: int = 3) -> set[str]:
    """Character ``n``-gram shingles over whitespace-normalized text."""
    t = " ".join(text.lower().split())
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity in ``[0, 1]``; two empty sets are identical (1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ProgressTracker:
    """Stateful detector that classifies each new step as progress / loop / stagnation.

    Parameters
    ----------
    window:
        How many recent steps to remember.
    duplicate_threshold:
        How many copies of the same fingerprint within ``window`` constitutes
        a duplicate-tool-call loop. Default 3. I.e. issuing the same tool
        call three times within ``window`` steps trips the detector.
    cycle_lengths:
        Cycle lengths to scan for. Defaults to (2, 3) which captures the
        most common pathologies (A-B-A-B-A-B; A-B-C-A-B-C).
    stagnation_window:
        How many *consecutive* steps without an observation change before
        the stagnation detector fires.
    semantic_threshold:
        If you pass observation embeddings (or rolling similarities) via
        ``observe_similarity``, this is the threshold above which the
        agent is considered to be "going in circles" semantically.
    max_steps:
        Hard cap on total steps; ``None`` disables it.
    ignore_args:
        Argument keys excluded from fingerprints (e.g. ``{"request_id"}``).
    normalize_strings:
        Lowercase and whitespace-collapse string arguments before
        fingerprinting so near-identical calls are treated as duplicates.
    """

    def __init__(
        self,
        *,
        window: int = 12,
        duplicate_threshold: int = 3,
        cycle_lengths: Sequence[int] = (2, 3),
        stagnation_window: int = 4,
        semantic_threshold: float = 0.95,
        max_steps: int | None = None,
        ignore_args: Iterable[str] = (),
        normalize_strings: bool = False,
    ) -> None:
        if window < max(cycle_lengths) * 2:
            raise ValueError(
                "window must be >= 2 * max(cycle_lengths) so cycle detection has room"
            )
        if duplicate_threshold < 2:
            raise ValueError("duplicate_threshold must be >= 2")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1 when set")
        self.window = window
        self.duplicate_threshold = duplicate_threshold
        self.cycle_lengths = tuple(cycle_lengths)
        self.stagnation_window = stagnation_window
        self.semantic_threshold = semantic_threshold
        self.max_steps = max_steps
        self.ignore_args = frozenset(ignore_args)
        self.normalize_strings = normalize_strings
        self._steps: deque[Step] = deque(maxlen=window)
        self._observations: deque[str | None] = deque(maxlen=stagnation_window)
        self._semantic_scores: deque[float] = deque(maxlen=stagnation_window)
        self._total_steps = 0
        self._last_text: str | None = None
        self._last_reason: RepetitionReason | None = None

    # ---- ingestion -----------------------------------------------------

    def observe(self, step: Step) -> LoopDetected | None:
        """Record a new step. Returns a ``LoopDetected`` if a loop fires."""
        self._steps.append(step)
        self._observations.append(step.observation)
        self._total_steps += 1
        # Run detectors in order of cheapness.
        for detector in (
            self._check_max_steps,
            self._check_duplicate,
            self._check_cycle,
            self._check_stagnation,
            self._check_semantic,
        ):
            decision = detector()
            if decision is not None:
                self._last_reason = RepetitionReason(decision.evidence["reason"])
                return decision
        return None

    def observe_similarity(self, score: float) -> LoopDetected | None:
        """Record a semantic similarity score between consecutive observations.

        Pass scores in [0, 1] where 1.0 means "identical". A run of high
        scores indicates the agent is going in circles even when the
        observation text differs character-by-character.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("similarity must be in [0, 1]")
        self._semantic_scores.append(score)
        decision = self._check_semantic()
        if decision is not None:
            self._last_reason = RepetitionReason.SEMANTIC_NO_PROGRESS
        return decision

    def observe_text(self, observation: str, *, n: int = 3) -> LoopDetected | None:
        """Compare ``observation`` with the previous one by shingle Jaccard.

        A dependency-free stand-in for embedding similarity: character
        3-gram Jaccard is robust to small edits (a changing timestamp,
        a reordered list) that defeat exact-match stagnation detection.
        The first observation has nothing to compare against and
        records no score.
        """
        current = shingles(observation, n)
        previous = self._last_text
        self._last_text = observation
        if previous is None:
            return None
        return self.observe_similarity(jaccard(shingles(previous, n), current))

    # ---- detectors -----------------------------------------------------

    def _fp(self, step: Step) -> str:
        return step.fingerprint(ignore_args=self.ignore_args, normalize_strings=self.normalize_strings)

    def _check_max_steps(self) -> LoopDetected | None:
        if self.max_steps is None or self._total_steps < self.max_steps:
            return None
        latest = self._steps[-1]
        return _loop_decision(
            RepetitionReason.MAX_STEPS,
            tool=latest.tool,
            args=latest.args,
            extra={"max_steps": self.max_steps, "total_steps": self._total_steps},
        )

    def _check_duplicate(self) -> LoopDetected | None:
        if len(self._steps) < self.duplicate_threshold:
            return None
        counts: dict[str, int] = {}
        for s in self._steps:
            fp = self._fp(s)
            counts[fp] = counts.get(fp, 0) + 1
            if counts[fp] >= self.duplicate_threshold:
                latest = self._steps[-1]
                return _loop_decision(
                    RepetitionReason.DUPLICATE_TOOL_CALL,
                    tool=latest.tool,
                    args=latest.args,
                    extra={"fingerprint": fp, "count": counts[fp]},
                )
        return None

    def _check_cycle(self) -> LoopDetected | None:
        fps = [self._fp(s) for s in self._steps]
        for length in self.cycle_lengths:
            # Need at least 2*length steps to see a cycle.
            if len(fps) < length * 2:
                continue
            tail = fps[-length:]
            prev = fps[-2 * length : -length]
            if tail == prev:
                return _loop_decision(
                    RepetitionReason.CYCLIC_TOOL_CALLS,
                    tool=self._steps[-1].tool,
                    args=self._steps[-1].args,
                    extra={"cycle_length": length, "cycle": tail},
                )
        return None

    def _check_stagnation(self) -> LoopDetected | None:
        if len(self._observations) < self.stagnation_window:
            return None
        obs = list(self._observations)
        if all(o == obs[0] and o is not None for o in obs):
            return _loop_decision(
                RepetitionReason.STAGNATION,
                tool=self._steps[-1].tool,
                args=self._steps[-1].args,
                extra={
                    "stagnation_window": self.stagnation_window,
                    "observation": obs[0],
                },
            )
        return None

    def _check_semantic(self) -> LoopDetected | None:
        if len(self._semantic_scores) < self.stagnation_window:
            return None
        scores = list(self._semantic_scores)
        if all(s >= self.semantic_threshold for s in scores):
            return _loop_decision(
                RepetitionReason.SEMANTIC_NO_PROGRESS,
                tool=self._steps[-1].tool if self._steps else "?",
                args=self._steps[-1].args if self._steps else {},
                extra={
                    "scores": scores,
                    "semantic_threshold": self.semantic_threshold,
                },
            )
        return None

    # ---- introspection -------------------------------------------------

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def total_steps(self) -> int:
        """Steps observed since construction or the last :meth:`reset`."""
        return self._total_steps

    def history(self) -> list[Step]:
        return list(self._steps)

    def explain(self) -> dict[str, Any]:
        """Current state of every detector, for debugging and dashboards."""
        fps = [self._fp(s) for s in self._steps]
        counts: dict[str, int] = {}
        for fp in fps:
            counts[fp] = counts.get(fp, 0) + 1
        obs = list(self._observations)
        scores = list(self._semantic_scores)
        return {
            "total_steps": self._total_steps,
            "max_steps": self.max_steps,
            "window_fill": len(self._steps),
            "window": self.window,
            "duplicate": {
                "threshold": self.duplicate_threshold,
                "max_count": max(counts.values(), default=0),
                "distinct_fingerprints": len(counts),
            },
            "cycle": {
                "lengths": list(self.cycle_lengths),
                "recent_fingerprints": fps[-max(self.cycle_lengths) * 2 :] if fps else [],
            },
            "stagnation": {
                "window": self.stagnation_window,
                "identical_run": _identical_run(obs),
            },
            "semantic": {
                "threshold": self.semantic_threshold,
                "scores": scores,
                "all_above": bool(scores) and all(s >= self.semantic_threshold for s in scores),
            },
            "last_reason": self._last_reason.value if self._last_reason else None,
        }

    def reset(self) -> None:
        """Forget all history (e.g. at the start of a new task)."""
        self._steps.clear()
        self._observations.clear()
        self._semantic_scores.clear()
        self._total_steps = 0
        self._last_text = None
        self._last_reason = None


def _identical_run(observations: list[str | None]) -> int:
    """Length of the trailing run of identical, non-None observations."""
    if not observations or observations[-1] is None:
        return 0
    last = observations[-1]
    run = 0
    for o in reversed(observations):
        if o != last:
            break
        run += 1
    return run


def _loop_decision(
    reason: RepetitionReason,
    *,
    tool: str,
    args: dict[str, Any],
    extra: dict[str, Any],
) -> LoopDetected:
    return LoopDetected(
        action="escalate",  # halt the agent and surface to caller
        confidence=0.99,
        tier=0,
        rationale=f"agent loop detected: {reason.value}",
        severity="medium",
        guard_name="aisafepy.core.progress",
        evidence={"reason": reason.value, "tool": tool, "args": args, **extra},
        latency_ms=0.0,
        fallback="The agent appears to be stuck in a loop and was halted.",
    )
