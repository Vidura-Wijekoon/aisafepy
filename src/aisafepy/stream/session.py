"""Conversation-level guarding.

Per-message guards are blind to slow-burn attacks: many-shot
jailbreaks that creep up over twenty turns, an attacker probing a
classifier with dozens of near-misses, or a client hammering the
endpoint. :class:`SessionGuard` keeps a small rolling record per
session and escalates on *patterns* across turns:

* ``N`` blocks within the window
* ``M`` escalations within the window
* a decayed cumulative risk score above a threshold
* turn rate above a per-session requests-per-minute limit

State lives in a :class:`SessionStore` (in-memory, TTL + LRU bounded).
The guard reads ``ctx.metadata["session_id"]``. The previous turn's
decision can be fed back either by placing it in
``ctx.metadata["last_decision"]`` or by calling
:meth:`SessionGuard.record` from a pipeline ``on_decision`` hook; the
latter is the recommended wiring because it sees every final decision
automatically.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.stream.pipeline import Context

_RISK_WEIGHTS: dict[str, float] = {
    "allow": 0.0,
    "transform": 0.25,
    "escalate": 0.6,
    "block": 1.0,
}


@dataclass
class SessionEvent:
    timestamp: float
    action: str
    confidence: float
    guard_name: str | None


@dataclass
class SessionState:
    """Mutable per-session record."""

    id: str
    created: float
    last_seen: float
    events: deque[SessionEvent] = field(default_factory=lambda: deque(maxlen=512))
    turns: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    risk: float = 0.0
    risk_updated: float = 0.0
    counters: dict[str, int] = field(
        default_factory=lambda: {"allow": 0, "block": 0, "transform": 0, "escalate": 0, "turns": 0}
    )

    def decayed_risk(self, now: float, half_life_s: float) -> float:
        if half_life_s <= 0 or self.risk == 0.0:
            return self.risk
        dt = max(0.0, now - self.risk_updated)
        return self.risk * math.pow(0.5, dt / half_life_s)

    def add_risk(self, amount: float, now: float, half_life_s: float) -> None:
        self.risk = self.decayed_risk(now, half_life_s) + amount
        self.risk_updated = now

    def count_in_window(self, action: str, now: float, window_s: float) -> int:
        cutoff = now - window_s
        return sum(1 for e in self.events if e.action == action and e.timestamp >= cutoff)

    def turns_in_window(self, now: float, window_s: float) -> int:
        cutoff = now - window_s
        return sum(1 for t in self.turns if t >= cutoff)

    def snapshot(self, now: float, half_life_s: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "created": self.created,
            "last_seen": self.last_seen,
            "age_s": round(now - self.created, 3),
            "risk": round(self.decayed_risk(now, half_life_s), 4),
            "counters": dict(self.counters),
            "recent": [
                {"t": e.timestamp, "action": e.action, "confidence": e.confidence, "guard": e.guard_name}
                for e in list(self.events)[-10:]
            ],
        }


class SessionStore:
    """In-memory session table with TTL expiry and LRU bounding."""

    def __init__(
        self,
        *,
        ttl_s: float = 3600.0,
        max_sessions: int = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.ttl_s = float(ttl_s)
        self.max_sessions = max_sessions
        self._clock = clock or time.time
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._clock()

    def get(self, session_id: str, *, create: bool = True) -> SessionState | None:
        now = self._clock()
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                if now - state.last_seen > self.ttl_s:
                    del self._sessions[session_id]
                    state = None
                else:
                    self._sessions.move_to_end(session_id)
            if state is None:
                if not create:
                    return None
                if len(self._sessions) >= self.max_sessions:
                    self._sessions.popitem(last=False)
                state = SessionState(id=session_id, created=now, last_seen=now, risk_updated=now)
                self._sessions[session_id] = state
            return state

    def touch(self, state: SessionState) -> None:
        state.last_seen = self._clock()

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def prune(self) -> int:
        now = self._clock()
        with self._lock:
            stale = [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl_s]
            for k in stale:
                del self._sessions[k]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)


@dataclass
class SessionGuard:
    """Escalate or block a session based on its recent history.

    Parameters
    ----------
    window_s:
        Window for the block / escalation counters.
    max_blocks, max_escalations:
        Thresholds within the window. ``0`` disables the check.
    risk_threshold:
        Decayed cumulative risk above which the guard trips. Each
        decision contributes ``risk_weights[action] * confidence``.
    risk_half_life_s:
        Half-life of the risk decay.
    max_turns_per_minute:
        Per-session rate limit. ``0`` disables.
    action:
        ``escalate`` (default) or ``block`` when a threshold trips.
    session_key:
        Metadata key holding the session id.
    """

    name: str = "session"
    tier: int = 1
    store: SessionStore = field(default_factory=SessionStore)
    window_s: float = 600.0
    max_blocks: int = 3
    max_escalations: int = 5
    risk_threshold: float = 3.0
    risk_half_life_s: float = 900.0
    max_turns_per_minute: int = 0
    action: str = "escalate"
    session_key: str = "session_id"
    risk_weights: dict[str, float] = field(default_factory=lambda: dict(_RISK_WEIGHTS))

    def __post_init__(self) -> None:
        if self.action not in ("escalate", "block"):
            raise ValueError("action must be 'escalate' or 'block'")
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")

    # ---- feeding history ----------------------------------------------------

    def record(self, session_id: str, decision: GuardDecision) -> SessionState:
        """Record a final decision for ``session_id``.

        Wire this from ``GuardPipeline.on_decision``::

            guard = SessionGuard()
            pipe = GuardPipeline(
                tier1=[guard],
                on_decision=lambda d, ctx: guard.record(ctx.metadata["session_id"], d),
            )
        """
        if decision.evidence.get("shadow") is True:
            state = self.store.get(session_id)
            assert state is not None
            return state
        if decision.guard_name == self.name:
            # Our own trip decisions must not feed back into the risk score.
            state = self.store.get(session_id)
            assert state is not None
            return state
        state = self.store.get(session_id)
        assert state is not None
        now = self.store.now()
        action = decision.action.value
        state.events.append(SessionEvent(now, action, decision.confidence, decision.guard_name))
        state.counters[action] = state.counters.get(action, 0) + 1
        weight = self.risk_weights.get(action, 0.0)
        state.add_risk(weight * max(0.0, decision.confidence), now, self.risk_half_life_s)
        self.store.touch(state)
        return state

    # ---- guard -----------------------------------------------------------------

    async def __call__(self, ctx: Context) -> GuardDecision:
        session_id = ctx.metadata.get(self.session_key)
        if not session_id:
            return GuardDecision.allow(
                tier=self.tier,
                guard_name=self.name,
                rationale="no session id; skipped",
                evidence={"skipped": True},
            )
        session_id = str(session_id)
        last = ctx.metadata.get("last_decision")
        if isinstance(last, GuardDecision):
            self.record(session_id, last)
        state = self.store.get(session_id)
        assert state is not None
        now = self.store.now()
        state.turns.append(now)
        state.counters["turns"] = state.counters.get("turns", 0) + 1
        self.store.touch(state)

        reasons: list[str] = []
        evidence: dict[str, Any] = {"session_id": session_id}
        blocks = state.count_in_window("block", now, self.window_s)
        escalations = state.count_in_window("escalate", now, self.window_s)
        risk = state.decayed_risk(now, self.risk_half_life_s)
        evidence.update({"blocks_in_window": blocks, "escalations_in_window": escalations, "risk": round(risk, 4)})
        if self.max_blocks and blocks >= self.max_blocks:
            reasons.append(f"{blocks} blocks in {int(self.window_s)}s")
        if self.max_escalations and escalations >= self.max_escalations:
            reasons.append(f"{escalations} escalations in {int(self.window_s)}s")
        if self.risk_threshold and risk >= self.risk_threshold:
            reasons.append(f"risk {risk:.2f} >= {self.risk_threshold}")
        if self.max_turns_per_minute:
            rate = state.turns_in_window(now, 60.0)
            evidence["turns_last_minute"] = rate
            if rate > self.max_turns_per_minute:
                reasons.append(f"{rate} turns/min > {self.max_turns_per_minute}")

        if not reasons:
            return GuardDecision.allow(
                tier=self.tier, guard_name=self.name, rationale="session healthy", evidence=evidence
            )
        evidence["reasons"] = reasons
        confidence = min(1.0, 0.7 + 0.1 * len(reasons))
        rationale = f"{self.name}: " + "; ".join(reasons)
        if self.action == "block":
            return GuardDecision.block(
                tier=self.tier,
                confidence=confidence,
                rationale=rationale,
                guard_name=self.name,
                evidence=evidence,
                severity="high",
                fallback="This conversation has been paused for review.",
            )
        return GuardDecision.escalate(
            tier=self.tier,
            confidence=confidence,
            rationale=rationale,
            guard_name=self.name,
            evidence=evidence,
            severity="medium",
        )

    # ---- introspection ---------------------------------------------------------

    def explain(self, session_id: str) -> dict[str, Any]:
        """Return the current state and threshold status for a session."""
        state = self.store.get(session_id, create=False)
        now = self.store.now()
        if state is None:
            return {"id": session_id, "known": False}
        snap = state.snapshot(now, self.risk_half_life_s)
        snap["known"] = True
        snap["thresholds"] = {
            "blocks_in_window": [state.count_in_window("block", now, self.window_s), self.max_blocks],
            "escalations_in_window": [
                state.count_in_window("escalate", now, self.window_s),
                self.max_escalations,
            ],
            "risk": [round(state.decayed_risk(now, self.risk_half_life_s), 4), self.risk_threshold],
            "turns_last_minute": [state.turns_in_window(now, 60.0), self.max_turns_per_minute],
        }
        return snap

    def reset(self, session_id: str) -> bool:
        return self.store.drop(session_id)


def action_of(decision: GuardDecision) -> Action:
    """Small helper for callers that want the enum rather than the string."""
    return decision.action
