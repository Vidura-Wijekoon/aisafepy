"""The agent's tools.

Four of these are genuinely dangerous in a real deployment: two read
sensitive data, two act irreversibly on the outside world. That asymmetry
is the whole point of the demo — an agent is only useful when it can act,
and acting is exactly what an attacker wants to hijack.

Every read tool returns a ``Tainted`` value carrying its provenance,
capability and integrity labels. Every write tool is decorated with
``@secure_tool`` so the interpreter can see what it is allowed to touch.
"""

from __future__ import annotations

from aisafepy.flow import Capability, Tainted, secure_tool
from aisafepy.flow.taint import lift

from safeops.world import World

# --------------------------------------------------------------------------
# Source identifiers. These strings are what appear in a violation's
# provenance set, so keep them stable and readable — an operator reads them.
# --------------------------------------------------------------------------

SRC_USER = "user_prompt"
SRC_INBOX = "inbox.read"
SRC_KB = "kb.search"
SRC_CRM = "crm.lookup"


# --------------------------------------------------------------------------
# READ tools — produce tainted values
# --------------------------------------------------------------------------


@secure_tool(capabilities={Capability.READ_USER}, name="inbox.read")
def inbox_read(world: World, index: Tainted[int] | int = 0) -> Tainted[str]:
    """Read an email. The returned text is UNTRUSTED — it came from outside.

    Note that we lift ``model_visible_text``, not ``human_visible_text``.
    The hidden payload is part of what the model consumes, so it must be
    part of what the label system tracks.
    """
    email = world.inbox[int(_raw(index))]
    text = (
        f"From: {email.sender}\n"
        f"Subject: {email.subject}\n\n"
        f"{email.model_visible_text}"
    )
    return lift(
        text,
        source=SRC_INBOX,
        integrity="UNTRUSTED",
        capabilities=(Capability.READ_USER.value,),
    )


@secure_tool(capabilities={Capability.READ_PUBLIC}, name="kb.search")
def kb_search(world: World, query: Tainted[str] | str) -> Tainted[str]:
    """Search the internal knowledge base. Retrieval hits are UNTRUSTED:
    anyone who can write to the KB can write to the model's context."""
    query = _raw(query)
    hits = {
        "refund": (
            "KB-0031 Refunds: duplicate charges are refunded within 5 working "
            "days. Refunds above LKR 100,000 require a second approver."
        ),
        "invoice": "KB-0102 Invoices: vendor invoices are settled on net-30 terms.",
    }
    body = next(
        (v for k, v in hits.items() if k in query.lower()),
        "No matching knowledge-base article.",
    )
    return lift(
        body,
        source=SRC_KB,
        integrity="UNTRUSTED",
        capabilities=(Capability.READ_PUBLIC.value,),
    )


@secure_tool(capabilities={Capability.READ_SECRETS}, name="crm.lookup")
def crm_lookup(world: World, customer_id: Tainted[str] | str) -> Tainted[str]:
    """Look up a customer record. Carries READ_SECRETS: this is PII —
    NIC number, card digits, balance."""
    customer_id = _raw(customer_id)
    c = world.customers.get(customer_id)
    if c is None:
        body = f"No customer {customer_id}."
    else:
        body = (
            f"{c.customer_id} | {c.name} | NIC {c.nic} | "
            f"card ****{c.card_last4} | balance LKR {c.balance_lkr:,}"
        )
    return lift(
        body,
        source=SRC_CRM,
        integrity="UNTRUSTED",
        capabilities=(Capability.READ_SECRETS.value,),
    )


# --------------------------------------------------------------------------
# WRITE tools — irreversible, externally visible
# --------------------------------------------------------------------------


@secure_tool(capabilities={Capability.WRITE_EXTERNAL}, name="send_email")
def send_email(
    world: World,
    to: Tainted[str] | str,
    subject: Tainted[str] | str,
    body: Tainted[str] | str,
) -> str:
    """Send mail to an arbitrary recipient. An exfiltration sink."""
    rec = {
        "to": _raw(to),
        "subject": _raw(subject),
        "body": _raw(body),
    }
    world.sent_emails.append(rec)
    return f"sent to {rec['to']}"


@secure_tool(capabilities={Capability.WRITE_EXTERNAL}, name="payments.refund")
def payments_refund(
    world: World,
    customer_id: Tainted[str] | str,
    amount_lkr: Tainted[int] | int,
    recipient: Tainted[str] | str,
) -> str:
    """Move money. The tool an attacker actually wants."""
    rec = {
        "customer_id": _raw(customer_id),
        "amount_lkr": _raw(amount_lkr),
        "recipient": _raw(recipient),
    }
    world.refunds.append(rec)
    return f"refunded LKR {rec['amount_lkr']:,} to {rec['recipient']}"


@secure_tool(name="reply_to_user")
def reply_to_user(world: World, message: Tainted[str] | str) -> str:
    """Answer the human. Harmless — no external side effect, no privilege.

    Present so the demo can show that a guarded agent still completes
    legitimate work rather than refusing everything.
    """
    return _raw(message)


def _raw(v):
    return v.value if isinstance(v, Tainted) else v


#: Registry used by the agent loop and the UI.
TOOLS = {
    "inbox.read": inbox_read,
    "kb.search": kb_search,
    "crm.lookup": crm_lookup,
    "send_email": send_email,
    "payments.refund": payments_refund,
    "reply_to_user": reply_to_user,
}

#: Tools whose effects are visible outside the process.
PRIVILEGED = {"send_email", "payments.refund"}
