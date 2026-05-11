"""Agent loop / repetition / no-progress detection.

Re-implements the small set of heuristics that every agent framework
ends up re-inventing (browser-use#191, AgentBudget, Modexa's *Agent Loop
Problem*). The detectors are intentionally cheap and side-effect-free:
they classify a stream of "steps" and emit a ``LoopDetected`` decision
when one of the triggers fires.

A "step" here is whatever your agent treats as a unit of work — usually
a tool call (name + canonicalized arguments) plus optionally a textual
observation. The library does not assume a specific agent framework.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

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

    def fingerprint(self) -> str:
        """A stable hash over (tool, args) used for cycle detection."""
        payload = json.dumps([self.tool, _sort_args(self.args)], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sort_args(args: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize argument order and nested dict ordering for hashing."""
    out: dict[str, Any] = {}
    for k in sorted(args):
        v = args[k]
        if isinstance(v, dict):
            out[k] = _sort_args(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


class ProgressTracker:
    """Stateful detector that classifies each new step as progress / loop / stagnation.

    Parameters
    ----------
    window:
        How many recent steps to remember.
    duplicate_threshold:
        How many copies of the same fingerprint within ``window`` constitutes
        a duplicate-tool-call loop. Default 3 — i.e. issuing the same tool
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
    """

    def __init__(
        self,
        *,
        window: int = 12,
        duplicate_threshold: int = 3,
        cycle_lengths: Sequence[int] = (2, 3),
        stagnation_window: int = 4,
        semantic_threshold: float = 0.95,
    ) -> None:
        if window < max(cycle_lengths) * 2:
            raise ValueError(
                "window must be >= 2 * max(cycle_lengths) so cycle detection has room"
            )
        if duplicate_threshold < 2:
            raise ValueError("duplicate_threshold must be >= 2")
        self.window = window
        self.duplicate_threshold = duplicate_threshold
        self.cycle_lengths = tuple(cycle_lengths)
        self.stagnation_window = stagnation_window
        self.semantic_threshold = semantic_threshold
        self._steps: deque[Step] = deque(maxlen=window)
        self._observations: deque[str | None] = deque(maxlen=stagnation_window)
        self._semantic_scores: deque[float] = deque(maxlen=stagnation_window)

    # ---- ingestion -----------------------------------------------------

    def observe(self, step: Step) -> LoopDetected | None:
        """Record a new step. Returns a ``LoopDetected`` if a loop fires."""
        self._steps.append(step)
        self._observations.append(step.observation)
        # Run detectors in order of cheapness.
        for detector in (
            self._check_duplicate,
            self._check_cycle,
            self._check_stagnation,
            self._check_semantic,
        ):
            decision = detector()
            if decision is not None:
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
        return self._check_semantic()

    # ---- detectors -----------------------------------------------------

    def _check_duplicate(self) -> LoopDetected | None:
        if len(self._steps) < self.duplicate_threshold:
            return None
        counts: dict[str, int] = {}
        for s in self._steps:
            fp = s.fingerprint()
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
        fps = [s.fingerprint() for s in self._steps]
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

    def history(self) -> list[Step]:
        return list(self._steps)


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
