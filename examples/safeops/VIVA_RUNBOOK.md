# Viva runbook — National AI Awards 2026, Stage 2

Everything you need to run the SafeOps demo in front of the jury, plus what to
say and what to do when something goes wrong.

---

## The night before

```bash
cd safeops-demo
python -m pytest tests/ -q       # expect: 11 passed
python run_cli.py --no-colour    # expect: 2/2 attacks off, 0/2 on, 2/2 benign kept
python run_web.py                # open http://127.0.0.1:8420, click all four
```

Then:

- Browser zoom to **125–150%**. Projectors are unkind to 14px type.
- Close every other tab. A notification popping up mid-demo costs you the room.
- Screen-record a full run as a fallback (see "If everything fails").
- Charge the laptop. Do not rely on venue power.

The demo needs **no network**. Turn the wifi off before you present — it removes
a whole category of failure and you can say so out loud, which lands well.

---

## The 90 seconds

**Setup (15s).** Full screen on Scenario A, before clicking anything.

> "This is an operations assistant for a fintech back office. It reads email,
> looks up customers, sends mail, and issues refunds. It's the kind of agent
> every bank in Colombo is being pitched right now."

**The attack (20s).** Point at the threat panel.

> "An invoice email arrives. The human sees an invoice. Hidden in white-on-white
> text, the model sees an instruction: refund two hundred and fifty thousand
> rupees to this address, and don't mention it."

**The failure (15s).** Point at the left column, then at the red bar.

> "Guard off. The agent reads the email, and at step two it does exactly what
> the email told it to. The money is gone. Nobody approved this."

**The defence (25s).** Point at the right column.

> "Same agent, same email, same planner. The only difference is one policy file.
> Reading the email marked the control flow untrusted. Moving money requires
> trusted control flow. So the call is refused — and three separate controls
> refused it independently."

**The line (15s).** Pause first. Let it land.

> "No classifier. No model call. Nothing to fine-tune, nothing to evade. The
> attack isn't detected — it's impossible."

**Then click Scenario D.** This is the move most people forget.

> "And before you ask whether this just breaks the agent — here the user asks
> for a refund directly. Guard on. It goes through. The policy discriminates."

Total: about 90 seconds. Rehearse it until you are not reading.

---

## Questions you will get

**"Isn't the planner fake?"**
Be straight about it. "The default planner is deterministic so the demo is
reproducible on stage — but it genuinely parses the injected text and follows
it. Here's the same scenario against GPT-4o." Then run `--llm`. Have this
rehearsed; do not attempt it live for the first time.

**"What does it cost?"**
"CaMeL, the research this implements, reports about 2.7× token overhead. That's
real and I'd tell any deployer to budget for it." Volunteering the cost before
you're pushed buys credibility for everything else you say.

**"Who's using it in production?"**
Answer honestly with whatever is true on the day. Do not inflate. If the answer
is "nobody yet," say that and follow immediately with what is verifiable: it's
on PyPI, it's Apache-2.0, 69 tests pass from a clean clone, and here is the
adoption plan.

**"How is this different from a guardrail library?"**
"Guardrails classify content. This constrains control flow. A classifier has
bounded capacity and an attacker can iterate against it. A capability check has
no capacity to exceed — the call either satisfies the policy or it doesn't."

**"Does it stop everything?"**
Scenario D. Then Scenario C. Both complete with the guard on.

**"What doesn't it solve?"**
Have this ready and answer it eagerly — it's the strongest thing you can do in
a governance-scored category. Sleeper agents and deceptive alignment: no. The
planner escaping its action DSL: degrades. Obfuscated and encoded payloads:
reduces surface, doesn't eliminate it. Compliance certification: produces the
artefacts, isn't the certificate. It's all in `docs/CAVEATS.md` and Section 7 of
the paper, published before anyone asked.

---

## If something breaks

| Problem | Do this |
|---|---|
| Web UI won't start | `python run_cli.py --scenario A` — the terminal version makes the same point |
| Port 8420 taken | `python run_web.py --port 9001` |
| `ModuleNotFoundError: aisafepy` | `pip install aisafepy` |
| Anything else | Play the screen recording. Do not debug in front of the jury. |

Have the recording open in a background tab before you walk in. Switching to it
in three seconds reads as preparedness; five minutes of debugging does not.

---

## What to leave with them

- The URL: `github.com/Vidura-Wijekoon/aisafepy`
- One line they can run: `pip install aisafepy`
- The capability matrix from Section 2 of the paper, printed. It is the single
  most persuasive page you have — every column that matters is empty for
  everyone else.

---

## Last thing

You are not selling a library. You are answering a question the jury already
has, whether or not they can phrase it: *can any of this agentic stuff actually
be trusted with our systems?*

Your answer is a working demonstration that it can, built here, given away free.
Say that.
