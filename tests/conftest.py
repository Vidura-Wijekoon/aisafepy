"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from aisafepy.flow import Capability, Policy


@pytest.fixture
def strict_policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("web.fetch", integrity="UNTRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .label_source("secrets", integrity="UNTRUSTED", caps={Capability.READ_SECRETS})
        .require(
            "send_email",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
        .require(
            "payments.transfer",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
        .deny_if(
            "send_email",
            when=lambda **kw: any(
                "read.secrets" in getattr(v, "provenance", frozenset())
                for v in kw.values()
            ),
            reason="secret-to-external-sink",
        )
    )
