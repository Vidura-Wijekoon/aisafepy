"""``Tainted[T]`` — a value wrapped with provenance, capabilities, and integrity.

This is the core data structure of the IFC system. Operations on
``Tainted`` values propagate labels using a *meet* semilattice:

- ``provenance`` (a frozenset of source identifiers): joined by union.
- ``capabilities`` (a frozenset of ``Capability``): joined by union — the
  resulting value can have come from *any* source on the union, so its
  required capability set is the union of its inputs.
- ``integrity`` (TRUSTED ≻ UNTRUSTED ≻ QUARANTINED): joined by *meet*,
  taking the worst (lowest) of the inputs. Once UNTRUSTED, always
  UNTRUSTED unless explicitly declassified by a ``Policy.declassify``
  point.

The lattice is intentionally minimal — three integrity levels and a flat
capability set are enough to express the CaMeL / FIDES / RTBAS policy
patterns. Extending the lattice is a future-version concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Generic, Iterable, Literal, TypeVar

T = TypeVar("T")
U = TypeVar("U")

Integrity = Literal["TRUSTED", "UNTRUSTED", "QUARANTINED"]

# Integrity ordering: lower index = higher trust.
_INTEGRITY_ORDER: tuple[Integrity, ...] = ("TRUSTED", "UNTRUSTED", "QUARANTINED")


def _meet(a: Integrity, b: Integrity) -> Integrity:
    """Return the *worst* (lowest-trust) of two integrity labels."""
    return _INTEGRITY_ORDER[max(_INTEGRITY_ORDER.index(a), _INTEGRITY_ORDER.index(b))]


@dataclass(frozen=True)
class Tainted(Generic[T]):
    """A value tagged with provenance, capability, and integrity labels.

    ``Tainted`` is immutable. All operations that would mutate produce a
    new instance with appropriately joined labels.
    """

    value: T
    provenance: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    integrity: Integrity = "TRUSTED"
    # Free-form annotations the interpreter or compiler may attach.
    annotations: dict[str, Any] = field(default_factory=dict)

    # ---- combinators ---------------------------------------------------

    def map(self, fn: Callable[[T], U]) -> "Tainted[U]":
        """Apply ``fn`` to the wrapped value, preserving all labels."""
        return Tainted(
            value=fn(self.value),
            provenance=self.provenance,
            capabilities=self.capabilities,
            integrity=self.integrity,
            annotations=self.annotations,
        )

    def join(self, other: "Tainted[Any]") -> "Tainted[T]":
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

    def with_integrity(self, integrity: Integrity) -> "Tainted[T]":
        """Return a copy with a (potentially upgraded) integrity label.

        For *upgrading* integrity, prefer ``Policy.declassify`` — that
        records an explicit declassification event in the audit log.
        """
        return replace(self, integrity=integrity)

    def with_provenance(self, *sources: str) -> "Tainted[T]":
        return replace(self, provenance=self.provenance | frozenset(sources))

    def with_capabilities(self, *caps: str) -> "Tainted[T]":
        return replace(self, capabilities=self.capabilities | frozenset(caps))

    # ---- string-y conveniences ----------------------------------------

    def __add__(self, other: Any) -> "Tainted[Any]":
        if isinstance(other, Tainted):
            return Tainted(
                value=self.value + other.value,
                provenance=self.provenance | other.provenance,
                capabilities=self.capabilities | other.capabilities,
                integrity=_meet(self.integrity, other.integrity),
            )
        return replace(self, value=self.value + other)

    def __radd__(self, other: Any) -> "Tainted[Any]":
        if isinstance(other, Tainted):
            return other.__add__(self)
        return replace(self, value=other + self.value)

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

    def __getitem__(self, key: Any) -> "Tainted[Any]":
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

    This is the *unsafe* escape hatch — it bypasses IFC. Use only at
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
