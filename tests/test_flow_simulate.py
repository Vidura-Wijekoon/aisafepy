"""Tests for the flow policy simulator."""

from __future__ import annotations

from aisafepy.flow import Capability, Policy
from aisafepy.flow.simulate import LITERAL_SOURCE, explain_path, simulate_policy


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("web.fetch", integrity="UNTRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .label_source("vault.read", integrity="TRUSTED", caps={Capability.READ_SECRETS})
        .require("send_email", control_flow_integrity="TRUSTED", arg_max_integrity="TRUSTED", caps={Capability.WRITE_EXTERNAL})
        .require("payments.transfer", control_flow_integrity="TRUSTED", caps={Capability.WRITE_EXTERNAL})
        .require("search", control_flow_integrity="UNTRUSTED")
        .deny_if("post_message", when={"provenance_contains": "vault.read"}, reason="no secrets to chat")
    )


def test_matrix_covers_every_source_tool_cfi_combination():
    report = simulate_policy(_policy())
    assert report.sources[0] == LITERAL_SOURCE
    assert set(report.sources) == {LITERAL_SOURCE, "user_prompt", "web.fetch", "gmail.read", "vault.read"}
    assert report.tools == ["payments.transfer", "post_message", "search", "send_email"]
    assert len(report.results) == len(report.sources) * len(report.tools) * 2


def test_allowed_and_denied_paths_match_the_policy():
    report = simulate_policy(_policy())
    assert report.get("user_prompt", "send_email", "TRUSTED").allowed
    web = report.get("web.fetch", "send_email", "TRUSTED")
    assert not web.allowed and "argument integrity" in web.reason
    gmail = report.get("gmail.read", "payments.transfer", "TRUSTED")
    assert not gmail.allowed and "forbidden capabilities" in gmail.reason
    assert not report.get("user_prompt", "send_email", "UNTRUSTED").allowed
    assert report.get("web.fetch", "search", "UNTRUSTED").allowed
    vault = report.get("vault.read", "post_message", "TRUSTED")
    assert not vault.allowed and "no secrets to chat" in vault.reason
    assert len(report.allowed_paths()) + len(report.denied_paths()) == len(report.results)


def test_risky_paths_flag_untrusted_or_secret_data_into_sinks():
    # payments.transfer permits READ_USER-free untrusted data with no arg integrity floor:
    # web.fetch (UNTRUSTED, no caps) is allowed into a sink. That is the finding.
    report = simulate_policy(_policy())
    risky = {(r.source, r.tool) for r in report.risky_paths(_policy()) if r.cfi == "TRUSTED"}
    assert ("web.fetch", "payments.transfer") in risky
    assert ("user_prompt", "send_email") not in risky
    assert all(r.source != LITERAL_SOURCE for r in report.risky_paths())
    # vault.read into post_message is denied, so not risky; into search it is allowed but
    # search is not a sink.
    assert ("vault.read", "post_message") not in risky
    assert ("vault.read", "search") not in risky


def test_uncovered_tools_with_and_without_policy():
    policy = _policy()
    report = simulate_policy(policy, tools=["send_email", "lookup_weather"])
    assert report.uncovered_tools(policy) == ["lookup_weather"]
    assert report.uncovered_tools() == ["lookup_weather"]


def test_markdown_and_dict_outputs():
    report = simulate_policy(_policy())
    md = report.to_markdown()
    assert "| source |" in md and "`send_email`" in md and "allow" in md and "deny" in md
    assert "Risky allowed paths" in md
    d = report.to_dict()
    assert d["policy_mode"] == "strict"
    assert any(p["allowed"] is False for p in d["paths"])
    assert ["web.fetch", "payments.transfer", "TRUSTED"] in d["risky"]


def test_fixed_signature_lambda_rules_are_noted_not_crashed():
    policy = _policy().deny_if("send_email", when=lambda to, body: False, reason="needs two args")
    report = simulate_policy(policy)
    assert any("fixed signature" in n for n in report.notes)
    assert report.get("user_prompt", "send_email", "TRUSTED").allowed


def test_mediated_policies_are_simulated_strictly_without_mutation():
    policy = _policy().with_mode("mediated")
    report = simulate_policy(policy)
    assert report.policy_mode == "mediated"
    assert not report.get("web.fetch", "send_email", "TRUSTED").allowed
    assert policy.mode == "mediated"


def test_unlabelled_source_defaults_to_untrusted():
    report = simulate_policy(_policy(), sources=["mystery"], tools=["send_email", "search"], include_literal=False)
    assert report.sources == ["mystery"]
    assert not report.get("mystery", "send_email", "TRUSTED").allowed
    assert report.get("mystery", "search", "TRUSTED").allowed


def test_explain_path_narrates_each_check():
    steps = explain_path(_policy(), "gmail.read", "payments.transfer")
    joined = "\n".join(steps)
    assert "labelled UNTRUSTED" in steps[0] and "read.user" in steps[0]
    assert "control flow must be at least TRUSTED: ok" in joined
    assert "arguments must be at least UNTRUSTED: ok" in joined
    assert "DENY (forbidden: ['read.user'])" in joined
    assert steps[-1].startswith("result: DENY")

    ok = explain_path(_policy(), "user_prompt", "send_email")
    assert ok[-1] == "result: ALLOW"
    literal = explain_path(_policy(), LITERAL_SOURCE, "search")
    assert "plain literal" in literal[0] and literal[-1] == "result: ALLOW"
    rule = explain_path(_policy(), "vault.read", "post_message")
    assert any("deny_if 'no secrets to chat': DENY" in s for s in rule)
