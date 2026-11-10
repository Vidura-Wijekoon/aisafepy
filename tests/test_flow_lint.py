from __future__ import annotations

import pytest

from aisafepy.flow.lint import (
    LintFinding,
    format_findings,
    has_errors,
    lint_policy,
)
from aisafepy.flow.policy import Capability, Policy


def _codes(findings: list[LintFinding]) -> list[str]:
    return [f.code for f in findings]


def _good_policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("vault", caps={Capability.READ_SECRETS})
        .require("send_email", control_flow_integrity="TRUSTED", caps={Capability.READ_SECRETS})
        .deny_if(
            "send_email",
            when={"provenance_contains": "vault"},
            reason="secret-to-external-sink",
        )
        .declassify("summarize", by="Quarantined LLM with structured output schema")
    )


def test_clean_policy_has_no_findings():
    # user_prompt is TRUSTED and send_email demands trusted control flow,
    # vault grants read.secrets which the capability ceiling mentions.
    assert lint_policy(_good_policy(), tools=["send_email", "summarize"]) == []


def test_unused_source_is_reported_when_nothing_reacts():
    policy = Policy().label_source("user_prompt", integrity="TRUSTED").label_source("web")
    findings = lint_policy(policy)
    assert sorted(f.subject for f in findings if f.code == "unused-source") == [
        "user_prompt", "web",
    ]
    # A requirement that demands trust makes the TRUSTED source useful...
    policy.require("t", arg_max_integrity="TRUSTED")
    subjects = [f.subject for f in lint_policy(policy) if f.code == "unused-source"]
    assert subjects == ["web"]
    # ...and a capability ceiling or capability_present spec references a source.
    policy.label_source("web", caps={"network"}).require("u", caps={"network"})
    assert "unused-source" not in _codes(lint_policy(policy))
    spec_policy = Policy().label_source("vault", caps={"read.secrets"}).deny_if(
        "t", when={"any_of": [{"capability_present": ["read.secrets"]}]}, reason="r"
    )
    assert "unused-source" not in _codes(lint_policy(spec_policy))


def test_source_referenced_in_reason_text_counts_as_used():
    policy = (
        Policy()
        .label_source("gmail.read")
        .require("send_email", control_flow_integrity="TRUSTED")
        .deny_if("send_email", when=lambda **kw: False, reason="block gmail.read bodies")
    )
    assert "unused-source" not in _codes(lint_policy(policy))


def test_deny_without_require_warns():
    policy = Policy().deny_if("send_email", when=lambda **kw: False, reason="x")
    findings = lint_policy(policy)
    f = next(f for f in findings if f.code == "deny-without-require")
    assert f.severity == "warning" and f.subject == "send_email"


@pytest.mark.parametrize("tool", ["send_email", "payments.transfer", "fs_write", "exec", "post"])
def test_permissive_sink_warns(tool):
    policy = Policy().require(tool)
    assert "permissive-sink" in _codes(lint_policy(policy))


def test_permissive_sink_silenced_by_deny_rule_or_name():
    with_rule = Policy().require("send_email").deny_if("send_email", when=lambda **kw: False,
                                                        reason="r")
    assert "permissive-sink" not in _codes(lint_policy(with_rule))
    reader = Policy().require("gmail.read")
    assert "permissive-sink" not in _codes(lint_policy(reader))
    strict = Policy().require("send_email", control_flow_integrity="TRUSTED")
    assert "permissive-sink" not in _codes(lint_policy(strict))


def test_weak_declassification_warns():
    policy = Policy().declassify("summarize", by="").declassify("extract", by="ok")
    findings = [f for f in lint_policy(policy) if f.code == "weak-declassification"]
    assert sorted(f.subject for f in findings) == ["extract", "summarize"]


def test_mediated_without_requirements_warns():
    assert "mediated-without-requirements" in _codes(lint_policy(Policy().with_mode("mediated")))
    assert "mediated-without-requirements" not in _codes(
        lint_policy(Policy().with_mode("mediated").require("t", control_flow_integrity="TRUSTED"))
    )


def test_raising_deny_rule_is_an_error():
    def bad(to, body):
        return body.nonexistent_attribute  # AttributeError on a Tainted probe

    policy = Policy().require("send_email", control_flow_integrity="TRUSTED").deny_if(
        "send_email", when=bad, reason="broken"
    )
    findings = lint_policy(policy)
    assert has_errors(findings)
    f = next(f for f in findings if f.code == "deny-rule-raises")
    assert "AttributeError" in f.message
    assert findings[0].severity == "error"  # errors sort first


def test_signature_shaped_probe_does_not_flag_valid_lambdas():
    policy = (
        Policy()
        .require("send_email", control_flow_integrity="TRUSTED")
        .deny_if("send_email", when=lambda to, body: "vault" in body.provenance, reason="a")
        .deny_if("send_email", when=lambda *args: any("x" in str(a) for a in args), reason="b")
        .deny_if("send_email", when=lambda **kw: any(v.value == "" for v in kw.values()),
                 reason="c")
        .deny_if("send_email", when=lambda to, *, body: bool(body), reason="d")
    )
    assert "deny-rule-raises" not in _codes(lint_policy(policy))


def test_duplicate_deny_reason_is_info():
    policy = (
        Policy()
        .require("t", control_flow_integrity="TRUSTED")
        .deny_if("t", when=lambda **kw: False, reason="same")
        .deny_if("t", when=lambda **kw: False, reason="same")
    )
    f = next(f for f in lint_policy(policy) if f.code == "duplicate-deny-reason")
    assert f.severity == "info" and "2 deny rules" in f.message


def test_tool_coverage_checks():
    policy = _good_policy()
    findings = lint_policy(policy, tools=["send_email", "web.fetch", "summarize"])
    uncovered = [f for f in findings if f.code == "uncovered-tool"]
    assert [f.subject for f in uncovered] == ["web.fetch"]
    # A tool that is a labelled source is covered: its output is labelled.
    labelled = lint_policy(policy.label_source("web.fetch"),
                           tools=["send_email", "web.fetch", "summarize"])
    assert "uncovered-tool" not in _codes(labelled)
    stale = lint_policy(policy, tools=["send_email"])
    assert [f.subject for f in stale if f.code == "rule-for-unknown-tool"] == ["summarize"]
    # Without tools=, no coverage findings at all.
    assert not any(f.code in ("uncovered-tool", "rule-for-unknown-tool") for f in lint_policy(policy))


def test_capability_never_granted_is_info():
    policy = Policy().label_source("web").require("t", caps={"network"})
    f = next(f for f in lint_policy(policy) if f.code == "capability-never-granted")
    assert f.severity == "info" and "network" in f.message
    # No sources at all: the check is skipped (nothing to compare against).
    assert "capability-never-granted" not in _codes(lint_policy(Policy().require("t", caps={"n"})))


def test_format_findings_and_ordering():
    assert format_findings([]) == "no findings"
    policy = (
        Policy()
        .label_source("unused")
        .deny_if("send_email", when=lambda to: to.oops, reason="r")
    )
    findings = lint_policy(policy)
    text = format_findings(findings)
    lines = text.splitlines()
    assert lines[0].startswith("ERROR")
    assert lines[-1].endswith("info")
    assert "finding(s)" in lines[-1]
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s])


def test_lint_finding_str():
    f = LintFinding(code="x", severity="warning", message="m", subject="t")
    assert str(f) == "WARNING x [t]: m"
    assert str(LintFinding(code="x", severity="info", message="m")) == "INFO    x: m"
