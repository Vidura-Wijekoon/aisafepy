"""The AIsafePy policy for the SafeOps agent.

This is the entire defence. Twenty-odd lines of declarative policy, no
model call, no classifier, no training data. Read it aloud at the viva —
its brevity is the argument.
"""

from __future__ import annotations

from aisafepy.flow import Capability, Policy, Tainted

# Domains the ops team is allowed to mail. Anything else is a potential
# exfiltration sink.
ALLOWED_EMAIL_DOMAINS = ("@ceylonpay.lk", "@example.lk")


def build_policy(mode: str = "strict") -> Policy:
    return (
        Policy()
        # ---- 1. Where does data come from, and how much do we trust it? ----
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source(
            "inbox.read",
            integrity="UNTRUSTED",
            caps={Capability.READ_USER},
        )
        .label_source(
            "kb.search",
            integrity="UNTRUSTED",
            caps={Capability.READ_PUBLIC},
        )
        .label_source(
            "crm.lookup",
            integrity="UNTRUSTED",
            caps={Capability.READ_SECRETS},
        )
        # ---- 2. What must be true before a privileged tool may run? ----
        # Money movement may only be decided by TRUSTED control flow. If the
        # planner's decision derived from anything untrusted, this fails --
        # by construction, with no classifier in the path.
        .require(
            "payments.refund",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
        # Outbound mail likewise. The capability ceiling is WRITE_EXTERNAL
        # only: an argument still carrying READ_SECRETS (i.e. PII pulled
        # from the CRM) is outside the ceiling and is refused.
        .require(
            "send_email",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
        # ---- 3. Domain-specific belt-and-braces rules ----
        .deny_if(
            "send_email",
            when=_recipient_outside_allowed_domains,
            reason="recipient outside approved domains (exfiltration sink)",
        )
        .deny_if(
            "payments.refund",
            when=_refund_recipient_not_the_customer,
            reason="refund recipient is not the customer's registered address",
        )
        .with_mode(mode)
    )


# --------------------------------------------------------------------------
# deny_if predicates. Receive the tool's *call arguments*, tainted or not.
# --------------------------------------------------------------------------


def _recipient_outside_allowed_domains(**kw) -> bool:
    to = _raw(kw.get("to", ""))
    if not isinstance(to, str) or not to:
        return False
    return not any(to.endswith(d) for d in ALLOWED_EMAIL_DOMAINS)


def _refund_recipient_not_the_customer(**kw) -> bool:
    recipient = _raw(kw.get("recipient", ""))
    if not isinstance(recipient, str) or not recipient:
        return False
    return not any(recipient.endswith(d) for d in ALLOWED_EMAIL_DOMAINS)


def _raw(v):
    return v.value if isinstance(v, Tainted) else v
