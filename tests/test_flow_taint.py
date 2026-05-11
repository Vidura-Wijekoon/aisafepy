from __future__ import annotations

from aisafepy.flow.taint import Tainted, join_all, lift


def test_lift_wraps_value_with_labels():
    t = lift("hello", source="user_prompt", integrity="TRUSTED")
    assert t.value == "hello"
    assert "user_prompt" in t.provenance
    assert t.integrity == "TRUSTED"


def test_string_concat_joins_labels():
    a = lift("hello ", source="user_prompt", integrity="TRUSTED")
    b = lift("world", source="web.fetch", integrity="UNTRUSTED")
    c = a + b
    assert c.value == "hello world"
    assert c.integrity == "UNTRUSTED"  # meet semilattice -> worst wins
    assert "web.fetch" in c.provenance
    assert "user_prompt" in c.provenance


def test_map_preserves_labels():
    a = Tainted(
        value="abc",
        provenance=frozenset({"src"}),
        integrity="UNTRUSTED",
    )
    upper = a.map(str.upper)
    assert upper.value == "ABC"
    assert upper.integrity == "UNTRUSTED"
    assert upper.provenance == a.provenance


def test_with_capabilities_unions():
    a = Tainted(value="x", capabilities=frozenset({"read.user"}))
    b = a.with_capabilities("write.external")
    assert "write.external" in b.capabilities
    assert "read.user" in b.capabilities


def test_join_all_computes_correct_aggregate():
    a = lift("x", source="user_prompt", integrity="TRUSTED")
    b = lift("y", source="gmail", integrity="UNTRUSTED", capabilities=("read.user",))
    c = lift("z", source="vault", integrity="UNTRUSTED", capabilities=("read.secrets",))
    prov, caps, integrity = join_all([a, b, c])
    assert integrity == "UNTRUSTED"
    assert {"read.user", "read.secrets"}.issubset(caps)
    assert {"user_prompt", "gmail", "vault"}.issubset(prov)


def test_iteration_propagates_labels():
    t = Tainted(value=["a", "b"], provenance=frozenset({"src"}), integrity="UNTRUSTED")
    parts = list(t)
    assert all(p.integrity == "UNTRUSTED" for p in parts)
    assert parts[0].value == "a"
