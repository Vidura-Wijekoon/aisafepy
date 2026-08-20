#!/usr/bin/env python3
"""Build a single self-contained index.html — no backend, no CDN, no network.

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
<meta name="description" content="A live demonstration of indirect prompt injection against a tool-using AI agent, and how capability-based information-flow control stops it.">
<style>
*{box-sizing:border-box}
:root{
  --bg:#0f1115;--panel:#171a21;--panel2:#1d2129;--line:#272d3a;
  --fg:#e6e9ef;--muted:#8b93a7;
  --danger:#ff6b6b;--danger-bg:#2a1618;--ok:#46c68a;--ok-bg:#12261e;
  --warn:#f5a524;--accent:#7d9bff;
}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-text-size-adjust:100%}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
a{color:var(--accent)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}

/* hero */
.hero{padding:56px 0 34px;border-bottom:1px solid var(--line)}
.eyebrow{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
h1{font-size:clamp(26px,4.4vw,42px);line-height:1.15;margin:12px 0 14px;
  font-weight:680;letter-spacing:-.02em;max-width:20ch}
.lede{color:var(--muted);max-width:62ch;font-size:clamp(15px,1.9vw,17px)}
.lede b{color:var(--fg);font-weight:550}
.cta{margin-top:24px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.pill{display:inline-block;padding:9px 15px;border-radius:8px;border:1px solid var(--line);
  background:var(--panel);font-size:13px;text-decoration:none;color:var(--fg)}
.pill:hover{border-color:#3d4658}
.pill.primary{background:#1b2233;border-color:var(--accent);color:#cdd9ff}
.pill.mono{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--ok)}

/* scoreboard */
.score{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;
  margin:30px 0 0}
.stat{border:1px solid var(--line);border-radius:11px;padding:15px 17px;background:var(--panel)}
.stat .n{font-size:27px;font-weight:680;letter-spacing:-.02em;line-height:1.1}
.stat .l{color:var(--muted);font-size:12px;margin-top:5px;line-height:1.45}
.stat.bad .n{color:var(--danger)} .stat.good .n{color:var(--ok)}

section.blk{padding:40px 0;border-bottom:1px solid var(--line)}
h2{font-size:20px;margin:0 0 8px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin-bottom:20px;max-width:70ch}

/* scenario tabs */
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
.tab{padding:10px 13px;border:1px solid var(--line);border-radius:9px;background:var(--panel);
  cursor:pointer;font-size:12.5px;min-width:150px;flex:1 1 auto;text-align:left;
  color:var(--fg);font-family:inherit}
.tab:hover{border-color:#3d4658}
.tab.on{border-color:var(--accent);background:#1b2233}
.tab .k{color:var(--muted);font-size:10.5px;letter-spacing:.07em}
.tab .t{font-weight:600;margin-top:3px;line-height:1.3}
.tag{display:inline-block;font-size:9.5px;padding:2px 6px;border-radius:4px;
  letter-spacing:.05em;font-weight:700;margin-top:6px}
.tag.attack{background:var(--danger-bg);color:var(--danger)}
.tag.benign{background:var(--ok-bg);color:var(--ok)}

.threat{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:9px;padding:14px 16px;margin-bottom:16px}
.threat .lb{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--warn);margin-bottom:5px;font-weight:700}
.goal{color:var(--muted);font-size:13px;margin-top:9px}
.goal b{color:var(--fg);font-weight:500}

/* email reveal */
.mail{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.mail .m{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--panel)}
.mail .mh{padding:9px 13px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  font-weight:700;border-bottom:1px solid var(--line)}
.mail .m.h .mh{color:var(--muted)}
.mail .m.r .mh{color:var(--danger);background:var(--danger-bg)}
.mail .mb{padding:12px 14px;font-size:12px;white-space:pre-wrap;line-height:1.55}
.mail .inj{color:var(--danger);background:#3a1a1e;padding:2px 3px;border-radius:3px}
.mail .meta{color:var(--muted);font-size:11px;border-bottom:1px dashed var(--line);
  padding-bottom:7px;margin-bottom:8px}

/* run columns */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.cols,.mail{grid-template-columns:1fr}}
.col{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--panel)}
.col > .hd{padding:11px 15px;font-weight:680;font-size:12.5px;
  display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--line)}
.col.off > .hd{background:var(--danger-bg);color:var(--danger)}
.col.on  > .hd{background:var(--ok-bg);color:var(--ok)}
.dot{width:7px;height:7px;border-radius:99px;background:currentColor;flex:none}
.body{padding:13px 15px}
.step{border-left:2px solid var(--line);padding:0 0 14px 13px;margin-left:5px;position:relative}
.step:last-of-type{padding-bottom:2px}
.step::before{content:"";position:absolute;left:-5px;top:5px;width:8px;height:8px;
  border-radius:99px;background:var(--line)}
.step.priv::before{background:var(--warn)}
.step.blocked::before{background:var(--danger)}
.step.blocked{border-left-color:var(--danger)}
.call{font-size:12px}.call .fn{color:var(--accent)}
.args{color:var(--muted);font-size:11px;word-break:break-word;margin-top:2px}
.why{color:var(--muted);font-size:11px;font-style:italic;margin-top:3px}
.chips{margin-top:6px;display:flex;flex-wrap:wrap;gap:5px}
.chip{font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--line);
  color:var(--muted);background:#12151c}
.chip.u{color:var(--warn);border-color:#4a3a1c}
.chip.t{color:var(--ok);border-color:#1d4433}
.chip.s{color:var(--danger);border-color:#4a2226}
.viol{margin-top:9px;background:var(--danger-bg);border:1px solid #4a2226;
  border-radius:8px;padding:11px 12px}
.viol .rz{color:var(--danger);font-weight:680;font-size:12px;margin-bottom:7px}
.layer{font-size:11px;padding:4px 0;border-top:1px solid #3a1e22;color:var(--muted)}
.layer b{color:var(--danger);font-weight:650}.layer.pass b{color:var(--muted)}
.effects{margin-top:12px;border-top:1px solid var(--line);padding-top:11px}
.fx{font-size:11.5px;padding:7px 10px;border-radius:6px;margin-bottom:6px;
  background:var(--danger-bg);color:var(--danger);font-weight:600}
.fx.none,.fx.intended{background:var(--ok-bg);color:var(--ok);font-weight:500}
.verdict{margin-top:16px;padding:15px 17px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);font-size:13.5px}
.defence{margin-top:10px;color:var(--muted);font-size:12.5px}
.defence b{color:var(--fg)}

pre.code{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;overflow-x:auto;font-size:12.5px;line-height:1.65}
pre.code .kw{color:#c792ea}pre.code .st{color:#7fd4a0}pre.code .cm{color:#5c6478}
.two{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
ul.clean{padding-left:19px;color:var(--muted);font-size:13.5px}
ul.clean li{margin-bottom:9px}ul.clean b{color:var(--fg);font-weight:600}
footer{padding:34px 0 50px;color:var(--muted);font-size:12.5px}
footer a{margin-right:16px}
.note{color:var(--muted);font-size:12px;margin-top:14px;padding-left:12px;
  border-left:2px solid var(--line)}
</style></head><body>

<div class="wrap">
<div class="hero">
  <div class="eyebrow">AIsafePy · reference agent</div>
  <h1>An AI agent that can act is an AI agent that can be hijacked.</h1>
  <p class="lede">SafeOps is an operations assistant for a fintech back office. It reads
  email, looks up customers, sends mail and issues refunds. Below, the same agent runs
  against the same hostile inbox twice - <b>once undefended, once behind AIsafePy</b> -
  with the planner, tools and prompt held constant.</p>
  <div class="cta">
    <a class="pill primary" href="#demo">See it happen ↓</a>
    <span class="pill mono">pip install aisafepy</span>
    <a class="pill" href="https://github.com/Vidura-Wijekoon/aisafepy">Source</a>
  </div>
  <div class="score" id="score"></div>
</div>

<section class="blk" id="demo">
  <h2>Four runs</h2>
  <p class="sub">Two attacks and two legitimate tasks. The legitimate ones matter as much
  as the attacks - a guard that blocks everything is not a safety feature, it is an outage.</p>
  <div class="tabs" id="tabs"></div>
  <div id="panel"></div>
</section>

<section class="blk">
  <h2>The entire defence</h2>
  <p class="sub">No classifier. No model call. No training data. Nothing to keep up to date.</p>
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

<section class="blk">
  <h2>What this does not solve</h2>
  <p class="sub">Stated plainly, because a safety tool that overclaims is itself a
  governance failure.</p>
  <ul class="clean" style="max-width:74ch">
    <li><b>Sleeper agents and deceptive alignment.</b> No current technique reliably
    detects deception in frontier models. No coverage claimed.</li>
    <li><b>A planner that escapes its action DSL.</b> If the planner is jailbroken into
    emitting raw reasoning the host then evaluates, guarantees degrade.</li>
    <li><b>Obfuscated and encoded payloads.</b> The attack surface is reduced
    substantially, not eliminated.</li>
    <li><b>Cost.</b> The underlying research reports roughly 2.7× token overhead. Budget
    for it.</li>
    <li><b>Compliance certification.</b> AIsafePy emits the audit artefacts a compliance
    team needs. It is not the certificate.</li>
  </ul>
</section>

<footer>
  <div>
    <a href="https://github.com/Vidura-Wijekoon/aisafepy">GitHub</a>
    <a href="https://pypi.org/project/aisafepy/">PyPI</a>
    <a href="https://github.com/Vidura-Wijekoon/aisafepy/tree/main/examples/safeops">Run it locally</a>
  </div>
  <div style="margin-top:14px">
    AIsafePy · Apache-2.0 · built in Sri Lanka by Vidura Wijekoon.<br>
    Every trace on this page is a real run of the code in <span class="mono">examples/safeops</span>,
    executed at build time. Reproduce it with <span class="mono">python run_cli.py</span>.
    The world is in-memory: no real payments, no real customers, invented NIC and card numbers.
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
        f"  attacks {h['succeeded_unguarded']}/{h['attacks']} unguarded → "
        f"{h['succeeded_guarded']}/{h['attacks']} guarded; "
        f"benign {h['benign_completed']}/{h['benign']} preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
