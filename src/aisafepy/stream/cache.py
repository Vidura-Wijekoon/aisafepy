"""A decision cache for expensive guards.

Tier-2 classifiers and Tier-3 judges cost tens to hundreds of
milliseconds and real traffic repeats itself: the same system prompt
preamble, the same canned replies, the same retry of a request that
timed out. :class:`DecisionCache` remembers the decision a guard made
for a piece of text so the next identical evaluation is a dictionary
lookup.

Why not cache Tier 1
--------------------

Deterministic guards already run in microseconds; hashing the input to
look it up costs about as much as scanning it, and caching them would
only make a configuration change (a new blocklist term) take effect
late. Cache Tiers 2 and 3, which is what :func:`cached_pipeline` does
by default.

What is and is not cached
-------------------------

A hit returns a *copy* of the stored decision with
``evidence["cached"] = True``, a fresh ``decision_id`` (so audit logs
still have one id per evaluation; pass ``reuse_decision_id=True`` to
keep the original), and the lookup latency. Decisions whose evidence
carries ``error`` or ``timeout`` are never stored: a guard that failed
open once should get another chance, not have its failure replayed for
the TTL.

Keys are a SHA-256 over the NFKC-normalised, whitespace-collapsed text,
the context role, and the guard name. Prompt and metadata are *not*
part of the key by default; set ``key_fields`` to include metadata
keys (a tenant id, a policy version) when the guard's answer depends on
them.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context, Guard, GuardPipeline

_WS = re.compile(r"\s+")


def normalise_for_key(text: str) -> str:
    """NFKC, collapse whitespace, strip. Case is preserved on purpose."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def make_key(
    text: str,
    *,
    role: str = "assistant",
    guard_name: str | None = None,
    extra: Sequence[tuple[str, Any]] = (),
) -> str:
    """Stable cache key for ``text`` under ``role`` for ``guard_name``."""
    h = hashlib.sha256()
    h.update(normalise_for_key(text).encode("utf-8"))
    h.update(b"\x00")
    h.update(role.encode("utf-8"))
    h.update(b"\x00")
    h.update((guard_name or "").encode("utf-8"))
    for k, v in extra:
        h.update(b"\x00")
        h.update(f"{k}={v!r}".encode())
    return h.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    stores: int = 0
    skipped: int = 0
    """Decisions not stored because they carried an error or timeout."""

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "stores": self.stores,
            "skipped": self.skipped,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class _Entry:
    decision: GuardDecision
    expires_at: float


@dataclass
class DecisionCache:
    """Thread-safe LRU cache with per-entry TTL.

    Parameters
    ----------
    max_entries:
        Capacity; the least recently *used* entry is evicted first.
    default_ttl_s:
        Lifetime applied when ``set`` is called without ``ttl_s``.
        ``None`` means entries never expire.
    clock:
        Monotonic clock, injectable for tests.
    """

    max_entries: int = 10_000
    default_ttl_s: float | None = 300.0
    clock: Callable[[], float] = time.monotonic
    _entries: OrderedDict[str, _Entry] = field(default_factory=OrderedDict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stats: CacheStats = field(default_factory=CacheStats, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be >= 1")

    def get(self, key: str) -> GuardDecision | None:
        now = self.clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self._stats.hits += 1
            return entry.decision

    def set(self, key: str, decision: GuardDecision, *, ttl_s: float | None = None) -> bool:
        """Store ``decision``. Returns False (and stores nothing) for error / timeout decisions."""
        if not is_cacheable(decision):
            with self._lock:
                self._stats.skipped += 1
            return False
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        expires_at = float("inf") if ttl is None else self.clock() + ttl
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = _Entry(decision=decision, expires_at=expires_at)
            self._stats.stores += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._stats.evictions += 1
        return True

    def invalidate(self, key: str | None = None) -> int:
        """Drop one key, or everything when ``key`` is None. Returns the count removed."""
        with self._lock:
            if key is None:
                n = len(self._entries)
                self._entries.clear()
                return n
            return 1 if self._entries.pop(key, None) is not None else 0

    def prune(self) -> int:
        """Remove expired entries eagerly. Returns the count removed."""
        now = self.clock()
        with self._lock:
            dead = [k for k, e in self._entries.items() if e.expires_at <= now]
            for k in dead:
                del self._entries[k]
            self._stats.expirations += len(dead)
            return len(dead)

    def stats(self) -> CacheStats:
        with self._lock:
            return replace(self._stats)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            entry = self._entries.get(str(key))
            return entry is not None and entry.expires_at > self.clock()


def is_cacheable(decision: GuardDecision) -> bool:
    """A decision is cacheable unless the guard failed to decide."""
    ev = decision.evidence or {}
    return not (ev.get("error") or ev.get("timeout") or ev.get("budget_exceeded"))


@dataclass
class CachedGuard:
    """Wrap a guard so repeated inputs reuse its decision.

    Parameters
    ----------
    inner:
        The guard to wrap. ``name`` becomes ``cached:<inner.name>`` so
        metrics can tell the two apart; the stored decision keeps the
        inner guard's ``guard_name``.
    cache:
        A shared :class:`DecisionCache`.
    ttl_s:
        Per-guard TTL override.
    reuse_decision_id:
        Keep the original ``decision_id`` on hits instead of minting one.
    key_fields:
        ``ctx.metadata`` keys folded into the cache key.
    """

    inner: Guard
    cache: DecisionCache
    ttl_s: float | None = None
    reuse_decision_id: bool = False
    key_fields: tuple[str, ...] = ()
    name: str = field(init=False)
    tier: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"cached:{getattr(self.inner, 'name', type(self.inner).__name__)}"
        self.tier = int(getattr(self.inner, "tier", 2))

    def key_for(self, ctx: Context) -> str:
        text = ctx.buffer or ctx.chunk or ""
        extra = tuple((k, ctx.metadata.get(k)) for k in self.key_fields) if ctx.metadata else ()
        return make_key(
            text, role=ctx.role, guard_name=getattr(self.inner, "name", None), extra=extra
        )

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        key = self.key_for(ctx)
        hit = self.cache.get(key)
        if hit is not None:
            update: dict[str, Any] = {
                "evidence": {**hit.evidence, "cached": True},
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            }
            if not self.reuse_decision_id:
                update["decision_id"] = uuid.uuid4().hex
            return hit.model_copy(update=update)
        decision = await self.inner(ctx)
        self.cache.set(key, decision, ttl_s=self.ttl_s)
        return decision

    def invalidate(self, text: str | None = None, *, role: str = "assistant") -> int:
        """Forget the cached decision for ``text`` (or everything for this cache)."""
        if text is None:
            return self.cache.invalidate()
        return self.cache.invalidate(
            make_key(text, role=role, guard_name=getattr(self.inner, "name", None))
        )

    def stats(self) -> CacheStats:
        return self.cache.stats()


def cache_guards(
    guards: Iterable[Guard],
    cache: DecisionCache,
    *,
    ttl_s: float | None = None,
    key_fields: tuple[str, ...] = (),
) -> list[Guard]:
    """Wrap every guard in ``guards`` with the same cache; already-cached guards pass through."""
    out: list[Guard] = []
    for g in guards:
        if isinstance(g, CachedGuard):
            out.append(g)
        else:
            out.append(CachedGuard(g, cache, ttl_s=ttl_s, key_fields=key_fields))
    return out


def cached_pipeline(
    pipeline: GuardPipeline,
    cache: DecisionCache,
    *,
    tiers: Iterable[int] = (2, 3),
    ttl_s: float | None = None,
    key_fields: tuple[str, ...] = (),
) -> GuardPipeline:
    """Return a new pipeline whose selected tiers are wrapped with ``cache``.

    The input pipeline is not mutated; untouched tiers are shallow
    copied so appending to the new pipeline never leaks into the old.
    """
    wanted = {int(t) for t in tiers}
    bad = wanted - {1, 2, 3}
    if bad:
        raise ValueError(f"tiers must be within 1..3 (got {sorted(bad)})")
    fields: dict[str, Any] = {}
    for tier in (1, 2, 3):
        attr = f"tier{tier}"
        guards = list(getattr(pipeline, attr))
        if tier in wanted:
            guards = cache_guards(guards, cache, ttl_s=ttl_s, key_fields=key_fields)
        fields[attr] = guards
    if hasattr(pipeline, "shadow_guards"):
        fields["shadow_guards"] = list(pipeline.shadow_guards)
    return replace(pipeline, **fields)
