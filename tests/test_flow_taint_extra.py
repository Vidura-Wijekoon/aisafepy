"""Lattice laws and the newer ``Tainted`` helpers."""

from __future__ import annotations

import itertools

import pytest

from aisafepy.flow.taint import (
    Tainted,
    _meet,
    collect,
    is_tainted,
    join_all,
    labels_of,
    lift,
    lower,
    map_values,
    tainted_join,
)

INTEGRITIES = ("TRUSTED", "UNTRUSTED", "QUARANTINED")

LABEL_TABLE = [
    Tainted("a"),
    Tainted("b", provenance=frozenset({"web"}), integrity="UNTRUSTED"),
    Tainted("c", provenance=frozenset({"mail"}), capabilities=frozenset({"read.user"})),
    Tainted(
        "d",
        provenance=frozenset({"vault"}),
        capabilities=frozenset({"read.secrets"}),
        integrity="QUARANTINED",
    ),
]


def _labels(t: Tainted) -> tuple[frozenset[str], frozenset[str], str]:
    return t.provenance, t.capabilities, t.integrity


# ---- lattice laws ------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), list(itertools.product(INTEGRITIES, INTEGRITIES)))
def test_meet_is_commutative_and_never_raises_integrity(a, b):
    assert _meet(a, b) == _meet(b, a)
    # Monotone: the result is at least as bad as either input.
    assert INTEGRITIES.index(_meet(a, b)) >= max(INTEGRITIES.index(a), INTEGRITIES.index(b))


@pytest.mark.parametrize(("a", "b", "c"), list(itertools.product(INTEGRITIES, repeat=3)))
def test_meet_is_associative(a, b, c):
    assert _meet(_meet(a, b), c) == _meet(a, _meet(b, c))


@pytest.mark.parametrize("x", INTEGRITIES)
def test_meet_is_idempotent_with_trusted_as_identity(x):
    assert _meet(x, x) == x
    assert _meet(x, "TRUSTED") == x


@pytest.mark.parametrize(("x", "y"), list(itertools.product(LABEL_TABLE, LABEL_TABLE)))
def test_join_is_commutative_on_labels(x, y):
    assert _labels(x.join(y)) == _labels(y.join(x))


@pytest.mark.parametrize(("x", "y", "z"), list(itertools.product(LABEL_TABLE, repeat=3)))
def test_join_is_associative_on_labels(x, y, z):
    assert _labels(x.join(y).join(z)) == _labels(x.join(y.join(z)))


@pytest.mark.parametrize("x", LABEL_TABLE)
def test_join_is_idempotent(x):
    assert _labels(x.join(x)) == _labels(x)


def test_join_all_matches_pairwise_join():
    pairwise = LABEL_TABLE[0]
    for t in LABEL_TABLE[1:]:
        pairwise = pairwise.join(t)
    assert join_all(LABEL_TABLE) == _labels(pairwise)


@pytest.mark.parametrize("value", ["x", 0, 3.5, None, [1, 2], {"k": "v"}, b"bytes"])
def test_lift_lower_round_trip(value):
    assert lower(lift(value, source="s")) == value
    assert lower(value) == value


# ---- equality / hashing -----------------------------------------------------


def test_equality_includes_labels_but_not_annotations():
    plain = Tainted("x")
    assert plain == Tainted("x")
    assert plain != Tainted("x", provenance=frozenset({"web"}))
    assert plain != Tainted("x", integrity="UNTRUSTED")
    assert plain == Tainted("x", annotations={"note": "ignored"})
    assert hash(plain) == hash(Tainted("x", annotations={"note": "ignored"}))
    assert len({plain, Tainted("x")}) == 1


# ---- arithmetic / dunder behaviour ---------------------------------------------------


def test_mul_joins_labels_and_repeats_value():
    a = lift("ab", source="web")
    b = lift(2, source="user", integrity="TRUSTED")
    out = a * b
    assert out.value == "abab"
    assert out.provenance == frozenset({"web", "user"})
    assert out.integrity == "UNTRUSTED"
    assert (3 * a).value == "ababab"
    assert (a * 2).provenance == a.provenance
    assert (lift(4, source="n") * lift(5, source="m")).value == 20


def test_bool_delegates_to_value():
    assert bool(lift("x", source="s"))
    assert not bool(lift("", source="s"))
    assert not lift(0, source="s")


def test_str_lowers_without_labels():
    t = lift("payload", source="web")
    assert str(t) == "payload"
    assert "web" not in str(t)
    assert "web" in repr(t)


def test_iteration_over_string_and_list_keeps_labels():
    s = lift("ab", source="web")
    chars = list(s)
    assert [c.value for c in chars] == ["a", "b"]
    assert all(c.provenance == frozenset({"web"}) for c in chars)
    lst = lift([1, 2], source="db", capabilities=("read.user",))
    assert all(item.capabilities == frozenset({"read.user"}) for item in lst)
    assert len(lst) == 2
    assert 2 in lst
    assert lift(2, source="x") in lst


# ---- helpers ----------------------------------------------------------------------


def test_is_tainted_and_labels_of():
    assert is_tainted(lift(1, source="s"))
    assert not is_tainted(1)
    assert labels_of("plain") == (frozenset(), frozenset(), "TRUSTED")
    t = lift("v", source="web", capabilities=("network",), integrity="QUARANTINED")
    assert labels_of(t) == (frozenset({"web"}), frozenset({"network"}), "QUARANTINED")


def test_collect_joins_labels_and_unwraps():
    out = collect([lift("a", source="web"), "plain", lift("c", source="db", integrity="TRUSTED")])
    assert out.value == ["a", "plain", "c"]
    assert out.provenance == frozenset({"web", "db"})
    assert out.integrity == "UNTRUSTED"
    empty = collect([])
    assert empty.value == [] and empty.integrity == "TRUSTED" and not empty.provenance
    assert collect(["x", "y"]).integrity == "TRUSTED"


def test_map_values_joins_keys_and_values():
    out = map_values(
        {
            "to": lift("a@x", source="mail"),
            lift("k", source="web"): 1,
            "plain": "value",
        }
    )
    assert out.value == {"to": "a@x", "k": 1, "plain": "value"}
    assert out.provenance == frozenset({"mail", "web"})
    assert out.integrity == "UNTRUSTED"


def test_tainted_join_propagates_from_parts_and_separator():
    out = tainted_join(", ", [lift("a", source="web"), "b", lift("c", source="db")])
    assert out.value == "a, b, c"
    assert out.provenance == frozenset({"web", "db"})
    sep = lift("-", source="sep", integrity="QUARANTINED")
    out2 = tainted_join(sep, ["x", "y"])
    assert out2.value == "x-y"
    assert out2.integrity == "QUARANTINED"
    assert "sep" in out2.provenance


def test_tainted_join_type_errors():
    with pytest.raises(TypeError):
        tainted_join(1, ["a"])
    with pytest.raises(TypeError):
        tainted_join(",", ["a", lift(2, source="n")])
