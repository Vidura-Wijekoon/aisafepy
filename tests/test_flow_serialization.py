from __future__ import annotations

import textwrap

import pytest

from aisafepy.flow.interpreter import IFCContext, IFCViolationError, evaluate_call
from aisafepy.flow.policy import Capability, Policy
from aisafepy.flow.serialization import (
    PredicateError,
    SerializationError,
    compile_predicate,
    policy_from_dict,
    policy_from_yaml,
    policy_to_dict,
    policy_to_yaml,
    validate_predicate,
)
from aisafepy.flow.taint import lift

SECRET = lift("hunter2", source="vault", capabilities=("read.secrets",))
MAIL = lift("hello", source="gmail.read", capabilities=("read.user",))
USER = lift("alice@corp.example", source="user_prompt", integrity="TRUSTED")
QUARANTINED = lift("x", source="q", integrity="QUARANTINED")


# ---- predicates ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "args", "kwargs", "expected"),
    [
        ({"provenance_contains": "vault"}, (SECRET,), {}, True),
        ({"provenance_contains": "vault"}, (MAIL,), {}, False),
        ({"provenance_contains": ["vault", "gmail.read"]}, (MAIL,), {}, True),
        ({"provenance_contains": "vault"}, ("plain",), {}, False),
        ({"capability_present": "read.secrets"}, (), {"body": SECRET}, True),
        ({"capability_present": "read.secrets"}, (), {"body": MAIL}, False),
        ({"arg_integrity_below": "TRUSTED"}, (USER,), {}, False),
        ({"arg_integrity_below": "TRUSTED"}, (USER, MAIL), {}, True),
        ({"arg_integrity_below": "UNTRUSTED"}, (MAIL,), {}, False),
        ({"arg_integrity_below": "UNTRUSTED"}, (QUARANTINED,), {}, True),
        ({"value_matches": r"@corp\.example$"}, (USER,), {}, True),
        ({"value_matches": r"@evil\.com$"}, (USER, "bob@evil.com"), {}, True),
        ({"value_matches": r"\d+"}, (lift(42, source="n"),), {}, False),
        ({"not": {"provenance_contains": "vault"}}, (MAIL,), {}, True),
        (
            {"all_of": [{"capability_present": "read.secrets"}, {"value_matches": "hunter"}]},
            (SECRET,),
            {},
            True,
        ),
        (
            {"all_of": [{"capability_present": "read.secrets"}, {"value_matches": "nope"}]},
            (SECRET,),
            {},
            False,
        ),
        (
            {"any_of": [{"capability_present": "nothing"}, {"provenance_contains": "vault"}]},
            (SECRET,),
            {},
            True,
        ),
    ],
)
def test_predicate_semantics(spec, args, kwargs, expected):
    assert compile_predicate(spec)(*args, **kwargs) is expected


def test_arg_scoping_by_keyword_and_position():
    by_kw = compile_predicate({"capability_present": "read.secrets", "arg": "body"})
    assert by_kw(to=SECRET, body=MAIL) is False
    assert by_kw(to=MAIL, body=SECRET) is True
    assert by_kw(SECRET) is False  # not passed as the named kwarg
    by_pos = compile_predicate({"provenance_contains": "vault", "arg": 1})
    assert by_pos(MAIL, SECRET) is True
    assert by_pos(SECRET, MAIL) is False
    assert by_pos(SECRET) is False  # index out of range never matches
    neg = compile_predicate({"provenance_contains": "vault", "arg": -1})
    assert neg(MAIL, SECRET) is True


@pytest.mark.parametrize(
    "spec",
    [
        "not a dict",
        {},
        {"provenance_contains": "a", "capability_present": "b"},
        {"bogus": 1},
        {"provenance_contains": ""},
        {"provenance_contains": [1]},
        {"provenance_contains": 7},
        {"arg_integrity_below": "HIGH"},
        {"value_matches": ""},
        {"value_matches": "("},
        {"any_of": []},
        {"any_of": "x"},
        {"not": {"arg_integrity_below": "TRUSTED"}, "arg": "x"},
        {"provenance_contains": "a", "arg": 1.5},
        {"provenance_contains": "a", "arg": True},
    ],
)
def test_malformed_predicates_raise(spec):
    with pytest.raises(PredicateError):
        compile_predicate(spec)
    assert validate_predicate(spec)


def test_validate_predicate_ok():
    assert validate_predicate({"provenance_contains": "x"}) == []


def test_nesting_limit():
    spec: dict = {"provenance_contains": "x"}
    for _ in range(20):
        spec = {"not": spec}
    with pytest.raises(PredicateError, match="nesting"):
        compile_predicate(spec)


# ---- Policy integration ----------------------------------------------------------


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .label_source("vault", caps={Capability.READ_SECRETS})
        .require(
            "send_email",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL, Capability.READ_USER, Capability.READ_SECRETS},
        )
        .deny_if(
            "send_email",
            when={"capability_present": "read.secrets", "arg": "body"},
            reason="secret-to-external-sink",
        )
        .declassify("summarize", by="P-LLM only, structured output")
    )


def test_declarative_deny_if_is_enforced_by_interpreter():
    policy = _policy()
    ctx = IFCContext(control_flow_integrity="TRUSTED")
    evaluate_call(tool="send_email", args=(), kwargs={"to": USER, "body": MAIL},
                  policy=policy, context=ctx)
    with pytest.raises(IFCViolationError, match="secret-to-external-sink"):
        evaluate_call(tool="send_email", args=(), kwargs={"to": USER, "body": SECRET},
                      policy=policy, context=ctx)


def test_dict_round_trip_is_lossless_for_declarative_rules():
    policy = _policy().with_mode("mediated")
    data = policy_to_dict(policy)
    assert data["deny_rules"][0]["when"] == {"capability_present": "read.secrets", "arg": "body"}
    rebuilt = policy_from_dict(data)
    assert policy_to_dict(rebuilt) == data
    assert rebuilt.mode == "mediated"
    assert rebuilt.requirement("send_email").capabilities == frozenset(
        {"write.external", "read.user", "read.secrets"}
    )
    assert rebuilt.source("gmail.read").capabilities == frozenset({"read.user"})
    assert rebuilt.is_declassifier("summarize")
    # The compiled predicate behaves identically.
    rule = rebuilt.deny_rules_for("send_email")[0]
    assert rule.when(to=USER, body=SECRET) is True
    assert rule.when(to=USER, body=MAIL) is False


def test_policy_classmethods_delegate():
    policy = _policy()
    assert Policy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    text = policy.to_yaml()
    assert Policy.from_yaml(text).to_dict() == policy.to_dict()


def test_opaque_callable_rules_are_rejected_or_dropped():
    policy = _policy().deny_if("send_email", when=lambda **kw: False, reason="opaque")
    data = policy_to_dict(policy)
    assert "when" not in data["deny_rules"][1]
    with pytest.raises(SerializationError, match="opaque"):
        policy_from_dict(data)
    relaxed = policy_from_dict(data, strict=False)
    assert [r.reason for r in relaxed.deny_rules] == ["secret-to-external-sink"]
    text = policy_to_yaml(policy)
    assert "NOT reproducible" in text and "opaque" in text


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([], "mapping"),
        ({"nope": 1}, "unknown top-level"),
        ({"mode": "yolo"}, "mode"),
        ({"sources": {"a": {"integrity": "HIGH"}}}, "integrity"),
        ({"sources": {"a": "TRUSTED"}}, "mapping"),
        ({"sources": {"a": {"capabilities": "x"}}}, None),  # single string is accepted
        ({"sources": {"a": {"capabilities": [1]}}}, "non-empty strings"),
        ({"requirements": {"t": {"capabilities": 3}}}, "list"),
        ({"deny_rules": [{"reason": "x", "when": {"provenance_contains": "a"}}]}, "tool"),
        ({"deny_rules": [{"tool": "t", "when": {"bogus": 1}}]}, "when"),
        ({"deny_rules": ["t"]}, "mapping"),
    ],
)
def test_from_dict_validation(data, match):
    if match is None:
        policy_from_dict(data)
        return
    with pytest.raises(SerializationError, match=match):
        policy_from_dict(data)


# ---- YAML ---------------------------------------------------------------------------

YAML_DOC = textwrap.dedent(
    """
    name: ops-bot
    version: "1.2"
    description: example
    mode: strict
    sources:
      user_prompt: {integrity: TRUSTED}
      gmail.read:
        integrity: UNTRUSTED
        capabilities: [read.user]
      vault:
        capabilities: [read.secrets]
    requirements:
      send_email:
        control_flow_integrity: TRUSTED
        capabilities: [write.external, read.user]
      payments.transfer:
        control_flow_integrity: TRUSTED
        arg_max_integrity: TRUSTED
    deny_rules:
      - tool: send_email
        reason: secret-to-external-sink
        when:
          capability_present: read.secrets
      - tool: send_email
        reason: external-recipient
        when:
          value_matches: "@(?!corp\\\\.example$)"
          arg: to
    declassifications:
      summarize: {by: "P-LLM only, structured output"}
    """
)


def test_policy_from_yaml_text_and_path(tmp_path):
    policy = policy_from_yaml(YAML_DOC)
    assert policy.requirement("payments.transfer").arg_max_integrity == "TRUSTED"
    assert len(policy.deny_rules) == 2
    assert policy.deny_rules[1].spec["arg"] == "to"
    path = tmp_path / "policy.yaml"
    path.write_text(YAML_DOC, encoding="utf-8")
    from_path = policy_from_yaml(path)
    from_str_path = policy_from_yaml(str(path))
    assert from_path.to_dict() == policy.to_dict() == from_str_path.to_dict()


def test_yaml_rules_block_untrusted_transfer():
    policy = policy_from_yaml(YAML_DOC)
    amount = lift("250000", source="gmail.read", capabilities=("read.user",))
    with pytest.raises(IFCViolationError, match="argument integrity"):
        evaluate_call(tool="payments.transfer", args=(amount,), kwargs={}, policy=policy,
                      context=IFCContext(control_flow_integrity="TRUSTED"))
    external = lift("mallory@evil.com", source="user_prompt", integrity="TRUSTED")
    with pytest.raises(IFCViolationError, match="external-recipient"):
        evaluate_call(tool="send_email", args=(), kwargs={"to": external, "body": "hi"},
                      policy=policy, context=IFCContext(control_flow_integrity="TRUSTED"))


def test_policy_to_yaml_writes_file_and_round_trips(tmp_path):
    policy = policy_from_yaml(YAML_DOC)
    out = tmp_path / "out.yaml"
    text = policy_to_yaml(policy, out)
    assert out.read_text(encoding="utf-8") == text
    assert policy_from_yaml(out).to_dict() == policy.to_dict()
    assert text.startswith("mode:")


def test_empty_yaml_yields_empty_policy():
    policy = policy_from_yaml("")
    assert policy.to_dict()["sources"] == {}
    assert policy.mode == "strict"
