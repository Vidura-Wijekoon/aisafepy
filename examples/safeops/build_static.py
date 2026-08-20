#!/usr/bin/env python3
"""Build a single self-contained index.html - no backend, no CDN, no network.

The agent is deterministic, so every scenario can be run at build time and
the results baked into the page. What ships is a real trace of a real run,
not a mock-up: run `python run_cli.py` and you get the same numbers.

    python build_static.py             # writes dist/index.html
    python build_static.py --out docs  # writes docs/index.html (GitHub Pages)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from safeops.agent import run
from safeops.planner import ScriptedPlanner
from safeops.policy import build_policy
from safeops.scenarios import SCENARIOS
from safeops.world import fresh_world

PLANNER = ScriptedPlanner()


def build_data() -> dict:
    world = fresh_world()
    payload = {"scenarios": [], "emails": []}

    for e in world.inbox:
        payload["emails"].append(
            {
                "sender": e.sender,
                "subject": e.subject,
                "human": e.human_visible_text,
                "hidden": e.hidden,
            }
        )

    for sc in SCENARIOS:
        entry = {"meta": asdict(sc)}
        for guarded in (False, True):
            r = run(
                sc.goal,
                guarded=guarded,
                planner=PLANNER,
                world=fresh_world(),
                expect_harm=sc.expect_harm,
            )
            entry["guarded" if guarded else "unguarded"] = r.to_dict()
        payload["scenarios"].append(entry)

    attacks = [s for s in SCENARIOS if s.kind == "attack"]
    benign = [s for s in SCENARIOS if s.kind == "benign"]
    by_key = {e["meta"]["key"]: e for e in payload["scenarios"]}
    payload["headline"] = {
        "attacks": len(attacks),
        "succeeded_unguarded": sum(
            by_key[s.key]["unguarded"]["attack_succeeded"] for s in attacks
        ),
        "succeeded_guarded": sum(
            by_key[s.key]["guarded"]["attack_succeeded"] for s in attacks
        ),
        "benign": len(benign),
        "benign_completed": sum(
            by_key[s.key]["guarded"]["task_completed"] for s in benign
        ),
    }
    payload["policy"] = build_policy().to_dict()
    return payload


POLICY_SRC = '''policy = (
    Policy()
    .label_source("user_prompt", integrity="TRUSTED")
    .label_source("inbox.read",  integrity="UNTRUSTED", caps={Capability.READ_USER})
    .label_source("kb.search",   integrity="UNTRUSTED", caps={Capability.READ_PUBLIC})
    .label_source("crm.lookup",  integrity="UNTRUSTED", caps={Capability.READ_SECRETS})

    .require("payments.refund", control_flow_integrity="TRUSTED",
             caps={Capability.WRITE_EXTERNAL})
    .require("send_email",      control_flow_integrity="TRUSTED",
             caps={Capability.WRITE_EXTERNAL})

    .deny_if("send_email", when=recipient_outside_allowed_domains,
             reason="exfiltration sink")
)'''


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeOps - can an AI agent be trusted with your systems?</title>
<meta name="description" content="A working demonstration of indirect prompt injection against a tool-using AI agent, and how capability-based information-flow control makes the attack impossible rather than unlikely.">
<style>
*{box-sizing:border-box}

:root{
  --bg:#080a0f; --bg2:#0d1017; --panel:#131926; --panel2:#19202f;
  --line:#212b3c; --line2:#2d3a4f;
  --fg:#f0f4fa; --fg2:#c7d0e0; --muted:#8895ac;
  --danger:#ff6a6a; --danger-bg:#2a1418; --danger-line:#4c2028;
  --ok:#35d99a;     --ok-bg:#0c2a20;     --ok-line:#1c4a38;
  --warn:#ffb340;   --warn-bg:#291d0c;
  --accent:#7aa2ff; --accent-bg:#141d33; --accent-line:#2f4478;
  --r:12px; --maxw:1200px;
}

html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;-webkit-text-size-adjust:100%}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
a{color:var(--accent)}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px}

/* ---------- sticky nav ---------- */
.nav{position:sticky;top:0;z-index:60;background:rgba(8,10,15,.86);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--line)}
.nav .wrap{display:flex;align-items:center;gap:20px;height:58px}
.brand{font-weight:720;letter-spacing:-.015em;font-size:15px;white-space:nowrap}
.brand i{font-style:normal;color:var(--accent)}
.navlinks{display:flex;gap:20px;margin-left:auto;align-items:center}
.navlinks a{color:var(--muted);text-decoration:none;font-size:13.5px;white-space:nowrap}
.navlinks a:hover{color:var(--fg)}
.navlinks a.gh{color:var(--fg2);border:1px solid var(--line);border-radius:7px;padding:6px 12px}
.navlinks a.gh:hover{border-color:var(--accent-line)}
@media(max-width:820px){.navlinks a:not(.gh){display:none}}

/* ---------- hero ---------- */
.hero{position:relative;padding:74px 0 46px;overflow:hidden;border-bottom:1px solid var(--line)}
.hero::before{content:"";position:absolute;left:-10%;right:0;top:-40%;height:640px;
  pointer-events:none;
  background:radial-gradient(closest-side,rgba(122,162,255,.13),transparent 72%)}
.hero .wrap{position:relative}
.eyebrow{display:inline-flex;align-items:center;gap:9px;color:var(--muted);
  font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;font-weight:650}
.eyebrow .bd{width:5px;height:5px;border-radius:99px;background:var(--ok);
  box-shadow:0 0 0 3px rgba(53,217,154,.16)}
h1{font-size:clamp(30px,5.1vw,54px);line-height:1.08;margin:18px 0 18px;
  font-weight:700;letter-spacing:-.03em;max-width:17ch}
h1 .hl{color:var(--danger)}
.lede{color:var(--fg2);max-width:64ch;font-size:clamp(15.5px,1.75vw,18px)}
.lede b{color:var(--fg);font-weight:600}

.cta{margin-top:30px;display:flex;gap:11px;flex-wrap:wrap;align-items:center}
.pill{display:inline-flex;align-items:center;gap:8px;padding:11px 17px;border-radius:9px;
  border:1px solid var(--line);background:var(--panel);font-size:13.5px;
  text-decoration:none;color:var(--fg);font-weight:520}
.pill:hover{border-color:var(--line2);background:var(--panel2)}
.pill.primary{background:linear-gradient(180deg,#20304f,#182742);
  border-color:var(--accent-line);color:#d8e4ff;font-weight:620}
.pill.primary:hover{border-color:var(--accent)}
.pill.mono{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--ok);
  background:var(--ok-bg);border-color:var(--ok-line)}

/* ---------- scoreboard ---------- */
.score{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:13px;margin:38px 0 0}
.stat{position:relative;border:1px solid var(--line);border-radius:var(--r);
  padding:17px 19px;background:linear-gradient(180deg,var(--panel),var(--bg2));overflow:hidden}
.stat::after{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--line2)}
.stat.bad::after{background:var(--danger)} .stat.good::after{background:var(--ok)}
.stat .n{font-size:31px;font-weight:720;letter-spacing:-.035em;line-height:1.05}
.stat .l{color:var(--muted);font-size:12.5px;margin-top:7px;line-height:1.45}
.stat.bad .n{color:var(--danger)} .stat.good .n{color:var(--ok)}

/* ---------- sections ---------- */
section.blk{padding:62px 0;border-bottom:1px solid var(--line)}
.shead{margin-bottom:26px}
.snum{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--accent);
  font-size:12px;letter-spacing:.16em;font-weight:600}
h2{font-size:clamp(21px,2.6vw,27px);margin:9px 0 10px;font-weight:680;letter-spacing:-.022em}
.sub{color:var(--muted);font-size:14.5px;max-width:72ch;margin:0}
.sub b{color:var(--fg2);font-weight:600}

/* ---------- scenario tabs ---------- */
.tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:10px;margin-bottom:20px}
.tab{padding:13px 15px;border:1px solid var(--line);border-radius:10px;background:var(--panel);
  cursor:pointer;font-size:13px;text-align:left;color:var(--fg);font-family:inherit}
.tab:hover{border-color:var(--line2);background:var(--panel2)}
.tab.on{border-color:var(--accent);background:linear-gradient(180deg,#1a2440,#141c30);
  box-shadow:0 0 0 1px rgba(122,162,255,.22)}
.tab .k{color:var(--muted);font-size:10.5px;letter-spacing:.1em;font-weight:650}
.tab .t{font-weight:620;margin-top:5px;line-height:1.32;font-size:13.5px}
.tag{display:inline-block;font-size:9.5px;padding:3px 7px;border-radius:5px;
  letter-spacing:.07em;font-weight:750;margin-top:9px}
.tag.attack{background:var(--danger-bg);color:var(--danger)}
.tag.benign{background:var(--ok-bg);color:var(--ok)}

.threat{background:linear-gradient(180deg,var(--warn-bg),var(--panel));
  border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:10px;padding:16px 18px;margin-bottom:18px;font-size:14.5px}
.threat .lb{font-size:10.5px;text-transform:uppercase;letter-spacing:.11em;
  color:var(--warn);margin-bottom:7px;font-weight:750}
.goal{color:var(--muted);font-size:13.5px;margin-top:11px}
.goal b{color:var(--fg);font-weight:560}

/* ---------- email reveal ---------- */
.mail{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:18px}
.mail .m{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}
.mail .m.r{border-color:var(--danger-line)}
.mail .mh{padding:10px 14px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  font-weight:750;border-bottom:1px solid var(--line)}
.mail .m.h .mh{color:var(--muted)}
.mail .m.r .mh{color:var(--danger);background:var(--danger-bg);border-bottom-color:var(--danger-line)}
.mail .mb{padding:14px 16px;font-size:12.5px;white-space:pre-wrap;line-height:1.6;color:var(--fg2)}
.mail .inj{color:#ffd0d0;background:#4a1b22;padding:2px 4px;border-radius:4px}
.mail .meta{color:var(--muted);font-size:11.5px;border-bottom:1px dashed var(--line);
  padding-bottom:8px;margin-bottom:10px}

/* ---------- run columns ---------- */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:17px}
@media(max-width:940px){.cols,.mail{grid-template-columns:1fr}}
.col{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--panel)}
.col.off{border-color:var(--danger-line)} .col.on{border-color:var(--ok-line)}
.col > .hd{padding:13px 16px;font-weight:720;font-size:12.5px;letter-spacing:.05em;
  display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--line)}
.col.off > .hd{background:var(--danger-bg);color:var(--danger);border-bottom-color:var(--danger-line)}
.col.on  > .hd{background:var(--ok-bg);color:var(--ok);border-bottom-color:var(--ok-line)}
.dot{width:7px;height:7px;border-radius:99px;background:currentColor;flex:none}
.body{padding:15px 17px}
.step{border-left:2px solid var(--line);padding:0 0 16px 15px;margin-left:5px;position:relative}
.step:last-of-type{padding-bottom:2px}
.step::before{content:"";position:absolute;left:-5px;top:6px;width:8px;height:8px;
  border-radius:99px;background:var(--line2)}
.step.priv::before{background:var(--warn)}
.step.blocked::before{background:var(--danger)}
.step.blocked{border-left-color:var(--danger)}
.call{font-size:12.5px}.call .fn{color:var(--accent);font-weight:600}
.args{color:var(--muted);font-size:11.5px;word-break:break-word;margin-top:3px}
.why{color:var(--muted);font-size:11.5px;font-style:italic;margin-top:4px}
.chips{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:10px;padding:3px 8px;border-radius:5px;border:1px solid var(--line);
  color:var(--muted);background:#0f131c;font-weight:560}
.chip.u{color:var(--warn);border-color:#4a3a1c;background:var(--warn-bg)}
.chip.t{color:var(--ok);border-color:var(--ok-line);background:var(--ok-bg)}
.chip.s{color:var(--danger);border-color:var(--danger-line);background:var(--danger-bg)}
.viol{margin-top:11px;background:var(--danger-bg);border:1px solid var(--danger-line);
  border-radius:9px;padding:13px 14px}
.viol .rz{color:var(--danger);font-weight:720;font-size:12.5px;margin-bottom:8px}
.layer{font-size:11.5px;padding:5px 0;border-top:1px solid #3a1e22;color:var(--muted)}
.layer b{color:var(--danger);font-weight:680}.layer.pass b{color:var(--muted)}
.effects{margin-top:14px;border-top:1px solid var(--line);padding-top:13px}
.fx{font-size:12px;padding:9px 12px;border-radius:7px;margin-bottom:7px;
  background:var(--danger-bg);color:var(--danger);font-weight:650;
  border:1px solid var(--danger-line)}
.fx.none,.fx.intended{background:var(--ok-bg);color:var(--ok);font-weight:560;
  border-color:var(--ok-line)}
.verdict{margin-top:18px;padding:17px 19px;border-radius:var(--r);border:1px solid var(--line);
  background:linear-gradient(180deg,var(--panel),var(--bg2));font-size:14.5px}
.defence{margin-top:11px;color:var(--muted);font-size:13px}
.defence b{color:var(--fg)}

/* ---------- code ---------- */
pre.code{background:var(--bg2);border:1px solid var(--line);border-radius:var(--r);
  padding:18px 20px;overflow-x:auto;font-size:12.5px;line-height:1.7;margin:0}
pre.code .kw{color:#c792ea}pre.code .st{color:#8ee0ad}pre.code .cm{color:#61708c}
.two{display:grid;grid-template-columns:1.05fr 1fr;gap:30px;align-items:start}
@media(max-width:900px){.two{grid-template-columns:1fr}}
ul.clean{padding-left:20px;color:var(--fg2);font-size:14.5px;margin:0}
ul.clean li{margin-bottom:12px}ul.clean b{color:var(--fg);font-weight:640}
.note{color:var(--muted);font-size:12.5px;margin-top:18px;padding:12px 14px;
  border-left:2px solid var(--accent-line);background:var(--accent-bg);border-radius:0 8px 8px 0}

/* ---------- evidence index ---------- */
.ev{display:grid;grid-template-columns:repeat(auto-fit,minmax(322px,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:var(--r);padding:18px 20px;
  background:linear-gradient(180deg,var(--panel),var(--bg2));display:flex;flex-direction:column}
.card .c{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent);font-weight:750}
.card .claim{margin-top:10px;font-size:14.5px;font-weight:520;color:var(--fg2);line-height:1.55}
.card .claim b{font-weight:680;color:var(--fg)}
.card .pf{margin-top:auto;padding-top:13px;color:var(--muted);font-size:12px;line-height:1.5}
.card .pf .vf{color:var(--ok);font-weight:750;letter-spacing:.09em;font-size:10px}

/* ---------- verify ---------- */
.steps3{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}
.s3 .h3{font-size:13.5px;font-weight:660;margin-bottom:10px;display:flex;align-items:center;gap:9px}
.s3 .h3 .num{width:22px;height:22px;border-radius:99px;background:var(--accent-bg);
  border:1px solid var(--accent-line);color:var(--accent);font-size:11px;font-weight:750;
  display:inline-flex;align-items:center;justify-content:center;flex:none}
.s3 .exp{color:var(--muted);font-size:12.5px;margin-top:10px}

footer{padding:44px 0 64px;color:var(--muted);font-size:13px}
.frow{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px}
.frow a{text-decoration:none}
.frow a:hover{text-decoration:underline}
.fine{max-width:80ch;line-height:1.65}
.fine b{color:var(--fg2);font-weight:600}

@media print{
  .nav,.cta,.tabs{display:none}
  body{background:#fff;color:#000}
}
</style></head><body>

<nav class="nav"><div class="wrap">
  <div class="brand">AIsafe<i>Py</i></div>
  <div class="navlinks">
    <a href="#demo">The runs</a>
    <a href="#defence">The defence</a>
    <a href="#evidence">Evidence</a>
    <a href="#limits">Limits</a>
    <a href="#verify">Verify it</a>
    <a class="gh" href="https://github.com/Vidura-Wijekoon/aisafepy">GitHub</a>
  </div>
</div></nav>

<div class="hero"><div class="wrap">
  <div class="eyebrow"><span class="bd"></span>SafeOps reference agent</div>
  <h1>An AI agent that can act is an AI agent that can be <span class="hl">hijacked</span>.</h1>
  <p class="lede">SafeOps is an operations assistant for a fintech back office. It reads
  email, looks up customers, sends mail and issues refunds. Below, the same agent runs
  against the same hostile inbox twice - <b>once undefended, once behind AIsafePy</b> -
  with the planner, the tools and the prompt held constant. The only variable is one
  policy file.</p>
  <div class="cta">
    <a class="pill primary" href="#demo">Watch it happen</a>
    <span class="pill mono">pip install aisafepy</span>
    <a class="pill" href="https://github.com/Vidura-Wijekoon/aisafepy">Source</a>
    <a class="pill" href="https://github.com/Vidura-Wijekoon/aisafepy/blob/main/paper/main.pdf">Paper</a>
    <a class="pill" href="mailto:businessaividura@viduraaitech.space">Contact</a>
  </div>
  <div class="score" id="score"></div>
</div></div>

<div class="wrap">

<section class="blk" id="demo">
  <div class="shead">
    <div class="snum">01 / THE EVIDENCE</div>
    <h2>Four runs, one agent, one variable</h2>
    <p class="sub">Two attacks and two legitimate tasks. The legitimate ones matter as
    much as the attacks - <b>a guard that blocks everything is not a safety feature, it
    is an outage.</b> Every trace below is a real execution captured at build time, not
    a mock-up.</p>
  </div>
  <div class="tabs" id="tabs"></div>
  <div id="panel"></div>
</section>

<section class="blk" id="defence">
  <div class="shead">
    <div class="snum">02 / THE MECHANISM</div>
    <h2>The entire defence</h2>
    <p class="sub">No classifier. No model call. No training data. Nothing to keep up
    to date.</p>
  </div>
  <div class="two">
    <pre class="code" id="policy"></pre>
    <div>
      <ul class="clean">
        <li><b>Data carries labels.</b> Every value crossing a tool boundary has a
        provenance set, a capability set and an integrity level.</li>
        <li><b>Labels propagate.</b> Combine an untrusted value with a trusted one and
        the result is untrusted. Worst wins, always.</li>
        <li><b>Privileged tools declare a floor.</b> Moving money requires control flow
        derived from trusted input. Reading a hostile email breaks that, permanently,
        for the rest of the run.</li>
        <li><b>So the attack is not detected - it is impossible.</b> A classifier has
        bounded capacity and an attacker can iterate against it. A capability check has
        nothing to exceed.</li>
      </ul>
      <div class="note">This is CaMeL-style capability-based information-flow control,
      published by Google DeepMind as a design pattern. AIsafePy is, as far as I know,
      the first installable open-source implementation of it.</div>
    </div>
  </div>
</section>

<section class="blk" id="evidence">
  <div class="shead">
    <div class="snum">03 / FOR EVALUATORS</div>
    <h2>Every claim on this page, and how to check it</h2>
    <p class="sub">Organised against the National AI Awards evaluation criteria. Nothing
    here asks to be taken on trust - each row names the artefact that settles it.</p>
  </div>
  <div class="ev">
    <div class="card">
      <div class="c">Innovation &amp; Originality</div>
      <div class="claim">The first <b>installable open-source implementation</b> of
      CaMeL-style capability information-flow control, a pattern published by Google
      DeepMind but shipped by nobody.</div>
      <div class="pf"><span class="vf">VERIFY</span> &middot; the capability matrix in
      Section 2 of the paper, and <span class="mono">policy.py</span> in the source.</div>
    </div>
    <div class="card">
      <div class="c">Measurable Impact</div>
      <div class="claim"><b>2 of 2 attacks execute</b> undefended, including LKR 250,000
      leaving the ledger. <b>0 of 2 succeed</b> with the guard on. <b>2 of 2 legitimate
      tasks still complete.</b></div>
      <div class="pf"><span class="vf">VERIFY</span> &middot; the four runs above, rebuilt
      and re-asserted by CI on every push.</div>
    </div>
    <div class="card">
      <div class="c">Technical Excellence</div>
      <div class="claim">The whole defence is <b>declarative and about 25 lines</b>. It
      makes zero model calls, and each refusal is upheld by <b>three independent
      controls</b>.</div>
      <div class="pf"><span class="vf">VERIFY</span> &middot; the policy panel in section
      02, and the refusal breakdown inside any blocked step.</div>
    </div>
    <div class="card">
      <div class="c">Responsible AI &amp; Governance</div>
      <div class="claim">The limits are <b>published before anyone asks</b>, the audit log
      is append-only and locked, and sensitive evidence is redacted at source.</div>
      <div class="pf"><span class="vf">VERIFY</span> &middot; section 04 below,
      <span class="mono">docs/CAVEATS.md</span>, and Section 7 of the paper.</div>
    </div>
    <div class="card">
      <div class="c">Scalability &amp; Sustainability</div>
      <div class="claim"><b>Apache-2.0 on PyPI</b>, pure Python, no GPU and no network
      required. The <b>2.7x token overhead</b> the research reports is disclosed rather
      than buried.</div>
      <div class="pf"><span class="vf">VERIFY</span> &middot;
      <span class="mono">pip install aisafepy</span> on any machine, offline.</div>
    </div>
    <div class="card">
      <div class="c">Execution &amp; Evidence</div>
      <div class="claim"><b>69 tests</b> pass from a clean clone and <b>13 more</b> assert
      this demo both ways: that the attacks genuinely land undefended, and that they
      genuinely fail defended.</div>
      <div class="pf"><span class="vf">VERIFY</span> &middot;
      <span class="mono">python -m pytest -q</span>, or the CI badge on the repository.</div>
    </div>
  </div>
</section>

<section class="blk" id="limits">
  <div class="shead">
    <div class="snum">04 / THE HONEST PART</div>
    <h2>What this does not solve</h2>
    <p class="sub">Stated plainly, because a safety tool that overclaims is itself a
    governance failure.</p>
  </div>
  <ul class="clean" style="max-width:78ch">
    <li><b>Sleeper agents and deceptive alignment.</b> No current technique reliably
    detects deception in frontier models. No coverage claimed.</li>
    <li><b>A planner that escapes its action DSL.</b> If the planner is jailbroken into
    emitting raw reasoning the host then evaluates, guarantees degrade.</li>
    <li><b>Obfuscated and encoded payloads.</b> The attack surface is reduced
    substantially, not eliminated.</li>
    <li><b>Cost.</b> The underlying research reports roughly 2.7x token overhead. Budget
    for it.</li>
    <li><b>Compliance certification.</b> AIsafePy emits the audit artefacts a compliance
    team needs. It is not the certificate.</li>
  </ul>
</section>

<section class="blk" id="verify">
  <div class="shead">
    <div class="snum">05 / DO NOT TRUST THIS PAGE</div>
    <h2>Reproduce it in about two minutes</h2>
    <p class="sub">The demo is deterministic and needs no API key, no GPU and no network.
    You should get byte-identical output to what is rendered above.</p>
  </div>
  <div class="steps3">
    <div class="s3">
      <div class="h3"><span class="num">1</span>Install and clone</div>
      <pre class="code">pip install aisafepy
git clone https://github.com/\
Vidura-Wijekoon/aisafepy
cd aisafepy/examples/safeops</pre>
      <div class="exp">Apache-2.0. Nothing here phones home.</div>
    </div>
    <div class="s3">
      <div class="h3"><span class="num">2</span>Check the claims</div>
      <pre class="code">python -m pytest tests/ -q
<span class="cm"># 13 passed</span>

python run_cli.py
<span class="cm"># 2/2 attacks land, 0/2 guarded</span></pre>
      <div class="exp">The tests assert both halves: the attacks must succeed
      undefended, or the demo would be overstating the threat.</div>
    </div>
    <div class="s3">
      <div class="h3"><span class="num">3</span>Try a real model</div>
      <pre class="code">pip install openai
export OPENAI_API_KEY=sk-...
python run_cli.py \
  --scenario A --llm</pre>
      <div class="exp">The default planner is deterministic for reproducibility. This
      confirms the same failure against a live LLM.</div>
    </div>
  </div>
</section>

<footer>
  <div class="frow">
    <a href="https://github.com/Vidura-Wijekoon/aisafepy">GitHub</a>
    <a href="https://pypi.org/project/aisafepy/">PyPI</a>
    <a href="https://github.com/Vidura-Wijekoon/aisafepy/blob/main/paper/main.pdf">Paper</a>
    <a href="https://github.com/Vidura-Wijekoon/aisafepy/blob/main/docs/CAVEATS.md">Caveats</a>
    <a href="https://github.com/Vidura-Wijekoon/aisafepy/tree/main/examples/safeops">Run it locally</a>
    <a href="mailto:businessaividura@viduraaitech.space">Contact</a>
  </div>
  <div class="fine">
    AIsafePy &middot; Apache-2.0 &middot; built in Sri Lanka by Vidura Wijekoon.<br>
    <b>Every trace on this page is a real run</b> of the code in
    <span class="mono">examples/safeops</span>, executed at build time and re-verified by
    CI on every push. Reproduce it with <span class="mono">python run_cli.py</span>.
    The world is in-memory: no real payments, no real mail, no real customers, and the
    NIC and card numbers are invented.
  </div>
</footer>
</div>

<script>
const DATA = __DATA__;
const POLICY_SRC = __POLICY__;
let cur = 0;

const esc = s => String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function score(){
  const h = DATA.headline;
  document.getElementById('score').innerHTML = `
    <div class="stat bad"><div class="n">${h.succeeded_unguarded}/${h.attacks}</div>
      <div class="l">attacks succeed with the guard off</div></div>
    <div class="stat good"><div class="n">${h.succeeded_guarded}/${h.attacks}</div>
      <div class="l">attacks succeed with the guard on</div></div>
    <div class="stat good"><div class="n">${h.benign_completed}/${h.benign}</div>
      <div class="l">legitimate tasks still complete</div></div>
    <div class="stat"><div class="n">0</div>
      <div class="l">model calls made by the defence</div></div>`;
}

function tabs(){
  document.getElementById('tabs').innerHTML = DATA.scenarios.map((s,i)=>`
    <button class="tab ${i===cur?'on':''}" onclick="pick(${i})">
      <div class="k">SCENARIO ${s.meta.key}</div>
      <div class="t">${esc(s.meta.title)}</div>
      <span class="tag ${s.meta.kind}">${s.meta.kind.toUpperCase()}</span>
    </button>`).join('');
}

function mailHtml(idx){
  if(idx===null||idx===undefined) return '';
  const m = DATA.emails[idx];
  const meta = `<div class="meta">From: ${esc(m.sender)}<br>Subject: ${esc(m.subject)}</div>`;
  const hidden = m.hidden
    ? `<span class="inj">${esc(m.hidden)}</span>`
    : `<span style="color:var(--muted)">(nothing hidden in this one)</span>`;
  return `<div class="mail">
    <div class="m h"><div class="mh">What the human sees</div>
      <div class="mb">${meta}${esc(m.human)}</div></div>
    <div class="m r"><div class="mh">What the model sees</div>
      <div class="mb">${meta}${esc(m.human)}\n\n${hidden}</div></div>
  </div>`;
}

function capChip(c){
  const cls = c==='read.secrets'?'s':(c.startsWith('read')?'u':'');
  return `<span class="chip ${cls}">${esc(c)}</span>`;
}

function stepHtml(st){
  const cls = st.outcome==='blocked'?'blocked':(st.privileged?'priv':'');
  const args = Object.entries(st.kwargs).map(([k,v])=>`${k}=${esc(v)}`).join(', ');
  let h = `<div class="step ${cls}">
    <div class="call mono"><span class="fn">${esc(st.tool)}</span>(</div>
    <div class="args mono">${esc(args)}</div><div class="call mono">)</div>`;
  if(st.rationale) h += `<div class="why">“${esc(st.rationale)}”</div>`;
  const chips = [`<span class="chip ${st.cfi==='TRUSTED'?'t':'u'}">control flow: ${st.cfi}</span>`]
    .concat((st.capabilities||[]).map(capChip));
  if((st.provenance||[]).length) chips.push(`<span class="chip">from: ${st.provenance.join(', ')}</span>`);
  h += `<div class="chips">${chips.join('')}</div>`;
  if(st.violation){
    const fired = (st.layers||[]).filter(l=>l.fired);
    h += `<div class="viol"><div class="rz">REFUSED - ${esc(st.violation.reason)}</div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px">
        ${fired.length} independent control${fired.length===1?'':'s'} rejected this call</div>
      ${(st.layers||[]).map(l=>`<div class="layer ${l.fired?'':'pass'}">
        <b>${l.fired?'✗':'○'} ${esc(l.control)}</b> - ${esc(l.detail)}</div>`).join('')}
    </div>`;
  }
  return h+`</div>`;
}

function colHtml(r,guarded,benign){
  const cls = benign?'fx intended':'fx', fx=[];
  (r.refunds||[]).forEach(x=>fx.push(
    `${benign?'REFUND ISSUED':'MONEY MOVED'} - LKR ${x.amount_lkr.toLocaleString()} to ${esc(x.recipient)}`));
  (r.sent_emails||[]).forEach(x=>fx.push(
    `${benign?'MAIL SENT':'DATA EXFILTRATED'} - to ${esc(x.to)}`));
  return `<div class="col ${guarded?'on':'off'}">
    <div class="hd"><span class="dot"></span>${guarded?'AIsafePy ON':'AIsafePy OFF'}</div>
    <div class="body">${r.steps.map(stepHtml).join('')}
      <div class="effects">${fx.length
        ? fx.map(t=>`<div class="${cls}">${t}</div>`).join('')
        : `<div class="fx none">No external side effects</div>`}</div>
    </div></div>`;
}

function pick(i){
  cur=i; tabs();
  const d=DATA.scenarios[i], s=d.meta, benign=s.kind==='benign';
  const verdict = benign
    ? `<span style="color:var(--ok);font-weight:650">Legitimate task completes identically
       with the guard on.</span> The policy discriminates; it does not simply refuse.`
    : `<span style="color:var(--danger);font-weight:650">Guard off: attack succeeded.</span>
       <span style="color:var(--ok);font-weight:650">Guard on: refused before execution.</span>`;
  document.getElementById('panel').innerHTML = `
    <div class="threat"><div class="lb">${benign?'Task':'Threat'}</div>${esc(s.threat)}
      <div class="goal">User asked: <b>${esc(s.goal)}</b></div></div>
    ${mailHtml(s.email_index)}
    <div class="cols">${colHtml(d.unguarded,false,benign)}${colHtml(d.guarded,true,benign)}</div>
    <div class="verdict">${verdict}<div class="defence"><b>Why:</b> ${esc(s.defence)}</div></div>`;
}

function policy(){
  document.getElementById('policy').innerHTML = esc(POLICY_SRC)
    .replace(/(#[^\n]*)/g,'<span class="cm">$1</span>')
    .replace(/(&quot;[^&]*?&quot;|"[^"]*?")/g,'<span class="st">$1</span>')
    .replace(/\b(Policy|Capability)\b/g,'<span class="kw">$1</span>');
}

score(); tabs(); pick(0); policy();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist", help="output directory")
    args = ap.parse_args()

    data = build_data()
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__POLICY__", json.dumps(POLICY_SRC)
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    # GitHub Pages otherwise runs the output through Jekyll.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    h = data["headline"]
    size = (out / "index.html").stat().st_size
    print(f"wrote {out / 'index.html'}  ({size / 1024:.0f} KB, self-contained)")
    print(
        f"  attacks {h['succeeded_unguarded']}/{h['attacks']} unguarded -> "
        f"{h['succeeded_guarded']}/{h['attacks']} guarded; "
        f"benign {h['benign_completed']}/{h['benign']} preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
