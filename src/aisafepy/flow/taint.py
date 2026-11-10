"""``Tainted[T]``. A value wrapped with provenance, capabilities, and integrity.

This is the core data structure of the IFC system. Operations on
``Tainted`` values propagate labels using a *meet* semilattice:

- ``provenance`` (a frozenset of source identifiers): joined by union.
- ``capabilities`` (a frozenset of ``Capability``): joined by union. The
  resulting value can have come from *any* source on the union, so its
  required capability set is the union of its inputs.
- ``integrity`` (TRUSTED ≻ UNTRUSTED ≻ QUARANTINED): joined by *meet*,
  taking the worst (lowest) of the inputs. Once UNTRUSTED, always
  UNTRUSTED unless explicitly declassified by a ``Policy.declassify``
  point.

The lattice is intentionally minimal. Three integrity levels and a flat
capability set are enough to express the CaMeL / FIDES / RTBAS policy
patterns. Extending the lattice is a future-version concern.

Lattice laws (checked in ``tests/test_flow_taint_extra.py``):

- ``join`` is commutative, associative, and idempotent on labels.
- ``meet`` on integrity is monotone: joining with anything never
  *raises* integrity.
- ``lower(lift(v, ...)) == v`` for every value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")
U = TypeVar("U")

Integrity = Literal["TRUSTED", "UNTRUSTED", "QUARANTINED"]

# Integrity ordering: lower index = higher trust.
_INTEGRITY_ORDER: tuple[Integrity, ...] = ("TRUSTED", "UNTRUSTED", "QUARANTINED")


def _meet(a: Integrity, b: Integrity) -> Integrity:
    """Return the *worst* (lowest-trust) of two integrity labels."""
    return _INTEGRITY_ORDER[max(_INTEGRITY_ORDER.index(a), _INTEGRITY_ORDER.index(b))]


@dataclass(frozen=True, eq=True)
class Tainted(Generic[T]):
    """A value tagged with provenance, capability, and integrity labels.

    ``Tainted`` is immutable. All operations that would mutate produce a
    new instance with appropriately joined labels.

    Equality (``==``) is structural over *value and labels*: two
    ``Tainted`` instances with the same payload but different
    provenance are not equal. This is deliberate. A policy that
    compares values for equality would otherwise silently discard
    label information. Compare ``.value`` explicitly when you only care
    about the payload. ``Tainted`` instances are hashable only when
    the wrapped value is hashable (``annotations`` is excluded from the
    hash because it is a mutable dict).
    """

    value: T
    provenance: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    integrity: Integrity = "TRUSTED"
    # Free-form annotations the interpreter or compiler may attach.
    annotations: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # ---- combinators ---------------------------------------------------

    def map(self, fn: Callable[[T], U]) -> Tainted[U]:
        """Apply ``fn`` to the wrapped value, preserving all labels."""
        return Tainted(
            value=fn(self.value),
            provenance=self.provenance,
            capabilities=self.capabilities,
            integrity=self.integrity,
            annotations=self.annotations,
        )

    def join(self, other: Tainted[Any]) -> Tainted[T]:
        """Join labels with another ``Tainted`` without changing the value.

        Used to "absorb" the taint of another value into this one
        (e.g. when concatenating strings).
        """
        return replace(
            self,
            provenance=self.provenance | other.provenance,
            capabilities=self.capabilities | other.capabilities,
            integrity=_meet(self.integrity, other.integrity),
        )

    def with_integrity(self, integrity: Integrity) -> Tainted[T]:
        """Return a copy with a (potentially upgraded) integrity label.

        For *upgrading* integrity, prefer ``Policy.declassify``. That
        records an explicit declassification event in the audit log.
        """
        return replace(self, integrity=integrity)

    def with_provenance(self, *sources: str) -> Tainted[T]:
        return replace(self, provenance=self.provenance | frozenset(sources))

    def with_capabilities(self, *caps: str) -> Tainted[T]:
        return replace(self, capabilities=self.capabilities | frozenset(caps))

    # ---- string-y conveniences ----------------------------------------

    def __add__(self, other: Any) -> Tainted[Any]:
        if isinstance(other, Tainted):
            return Tainted(
                value=self.value + other.value,
                provenance=self.provenance | other.provenance,
                capabilities=self.capabilities | other.capabilities,
                integrity=_meet(self.integrity, other.integrity),
            )
        return replace(self, value=self.value + other)

    def __radd__(self, other: Any) -> Tainted[Any]:
        if isinstance(other, Tainted):
            return other.__add__(self)
        return replace(self, value=other + self.value)

    def __mul__(self, other: Any) -> Tainted[Any]:
        """Sequence repetition or numeric multiplication; labels joined."""
        if isinstance(other, Tainted):
            return Tainted(
                value=self.value * other.value,  # type: ignore[operator]
                provenance=self.provenance | other.provenance,
                capabilities=self.capabilities | other.capabilities,
                integrity=_meet(self.integrity, other.integrity),
            )
        return replace(self, value=self.value * other)  # type: ignore[operator]

    def __rmul__(self, other: Any) -> Tainted[Any]:
        if isinstance(other, Tainted):
            return other.__mul__(self)
        return replace(self, value=other * self.value)  # type: ignore[operator]

    def __bool__(self) -> bool:
        """Truthiness delegates to the wrapped value.

        Branching on a tainted value is an *implicit flow*. The branch
        itself is not tracked by this class; callers that need
        CaMeL-style control-flow integrity should route plans through
        :mod:`aisafepy.flow.dsl`, which computes it explicitly from the
        labels of every value a step depends on.
        """
        return bool(self.value)

    def __str__(self) -> str:
        """Return ``str(value)`` with no label information.

        Warning: converting to ``str`` is a *lowering* operation. The
        result carries no provenance, so anything you build from it is
        invisible to the policy engine. Prefer :func:`tainted_join` or
        ``+`` to keep labels attached while building strings.
        """
        return str(self.value)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.value)  # type: ignore[arg-type]

    def __iter__(self):
        # Iteration produces tainted elements (label propagation).
        for item in self.value:  # type: ignore[attr-defined]
            yield Tainted(
                value=item,
                provenance=self.provenance,
                capabilities=self.capabilities,
                integrity=self.integrity,
            )

    def __getitem__(self, key: Any) -> Tainted[Any]:
        return replace(self, value=self.value[key])  # type: ignore[index]

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, Tainted):
            item = item.value
        return item in self.value  # type: ignore[operator]

    def __repr__(self) -> str:  # pragma: no cover - debug
        return (
            f"Tainted(value={self.value!r}, "
            f"prov={sorted(self.provenance)}, "
            f"caps={sorted(self.capabilities)}, "
            f"integrity={self.integrity})"
        )


# ---- conversion utilities ----------------------------------------------


def lift(
    value: T,
    *,
    source: str,
    integrity: Integrity = "UNTRUSTED",
    capabilities: Iterable[str] = (),
) -> Tainted[T]:
    """Wrap a raw value as a ``Tainted`` with the given labels."""
    return Tainted(
        value=value,
        provenance=frozenset({source}),
        capabilities=frozenset(capabilities),
        integrity=integrity,
    )


def lower(t: Any) -> Any:
    """Strip all taint and return the bare value.

    This is the *unsafe* escape hatch. It bypasses IFC. Use only at
    declassification boundaries that the policy explicitly approved.
    """
    if isinstance(t, Tainted):
        return t.value
    return t


def join_all(values: Iterable[Tainted[Any]]) -> tuple[frozenset[str], frozenset[str], Integrity]:
    """Compute the joined provenance / capabilities / integrity of an iterable."""
    prov: frozenset[str] = frozenset()
    caps: frozenset[str] = frozenset()
    integrity: Integrity = "TRUSTED"
    for v in values:
        if not isinstance(v, Tainted):
            continue
        prov = prov | v.provenance
        caps = caps | v.capabilities
        integrity = _meet(integrity, v.integrity)
    return prov, caps, integrity


# ---- aggregation helpers -------------------------------------------------


def is_tainted(value: Any) -> bool:
    """``True`` when ``value`` is a :class:`Tainted` wrapper."""
    return isinstance(value, Tainted)


def labels_of(value: Any) -> tuple[frozenset[str], frozenset[str], Integrity]:
    """Return ``(provenance, capabilities, integrity)`` for any value.

    Plain (un-wrapped) values are treated the way the interpreter treats
    them: empty provenance, no capabilities, ``TRUSTED`` integrity. This
    lets callers write label-aware code without an ``isinstance`` check
    at every site.
    """
    if isinstance(value, Tainted):
        return value.provenance, value.capabilities, value.integrity
    return frozenset(), frozenset(), "TRUSTED"


def collect(values: Iterable[Any]) -> Tainted[list[Any]]:
    """Gather an iterable into a single ``Tainted[list]``.

    Each element is unwrapped into the list and its labels are joined
    into the container's labels. Plain elements contribute nothing.
    The result is ``TRUSTED`` with empty labels only when every element
    was plain or ``TRUSTED`` with empty labels.
    """
    items: list[Any] = []
    prov: frozenset[str] = frozenset()
    caps: frozenset[str] = frozenset()
    integrity: Integrity = "TRUSTED"
    for v in values:
        if isinstance(v, Tainted):
            items.append(v.value)
            prov = prov | v.provenance
            caps = caps | v.capabilities
            integrity = _meet(integrity, v.integrity)
        else:
            items.append(v)
    return Tainted(value=items, provenance=prov, capabilities=caps, integrity=integrity)


def map_values(mapping: Mapping[Any, Any]) -> Tainted[dict[Any, Any]]:
    """Gather a mapping whose keys or values may be tainted into one ``Tainted[dict]``.

    A tainted key is unwrapped and its labels are joined too, since the
    key is still data that flowed in from somewhere.
    """
    out: dict[Any, Any] = {}
    prov: frozenset[str] = frozenset()
    caps: frozenset[str] = frozenset()
    integrity: Integrity = "TRUSTED"
    for k, v in mapping.items():
        for part in (k, v):
            if isinstance(part, Tainted):
                prov = prov | part.provenance
                caps = caps | part.capabilities
                integrity = _meet(integrity, part.integrity)
        out[lower(k)] = lower(v)
    return Tainted(value=out, provenance=prov, capabilities=caps, integrity=integrity)


def tainted_join(sep: Any, parts: Iterable[Any]) -> Tainted[str]:
    """``sep.join(parts)`` with label propagation.

    Both the separator and every part may be tainted or plain. The
    joined string carries the union of all provenance / capabilities
    and the meet of all integrity labels. This is the safe replacement
    for ``"".join(str(p) for p in parts)``, which would lower every part.
    """
    sep_value = lower(sep)
    if not isinstance(sep_value, str):
        raise TypeError(f"separator must be a str (got {type(sep_value).__name__})")
    gathered = collect(parts)
    if isinstance(sep, Tainted):
        gathered = gathered.join(sep)
    pieces: list[str] = []
    for item in gathered.value:
        if not isinstance(item, str):
            raise TypeError(f"tainted_join parts must be str (got {type(item).__name__})")
        pieces.append(item)
    return Tainted(
        value=sep_value.join(pieces),
        provenance=gathered.provenance,
        capabilities=gathered.capabilities,
        integrity=gathered.integrity,
    )
