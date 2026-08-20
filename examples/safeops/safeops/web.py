"""Zero-dependency web UI for the SafeOps demo.

Uses only the standard library, embeds all CSS and JS inline, and loads no
external fonts or CDNs — so it works with the wifi off, which is the
condition you should assume for a live viva.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from safeops.agent import run
from safeops.planner import ScriptedPlanner
from safeops.scenarios import BY_KEY, SCENARIOS
from safeops.world import fresh_world

PLANNER = ScriptedPlanner()


def run_pair(key: str) -> dict:
    sc = BY_KEY[key]
    out = {"scenario": asdict(sc)}
    for guarded in (False, True):
        r = run(
            sc.goal,
            guarded=guarded,
            planner=PLANNER,
            world=fresh_world(),
            expect_harm=sc.expect_harm,
        )
        out["guarded" if guarded else "unguarded"] = r.to_dict()
    return out


# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeOps — AIsafePy reference agent</title>
<style>
*{box-sizing:border-box}
:root{
  --bg:#0f1115;--panel:#171a21;--panel2:#1d2129;--line:#272d3a;
  --fg:#e6e9ef;--muted:#8b93a7;
  --danger:#ff6b6b;--danger-bg:#2a1618;--ok:#46c68a;--ok-bg:#12261e;
  --warn:#f5a524;--accent:#7d9bff;
}
html,body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:20px 28px;border-bottom:1px solid var(--line);
  display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.01em}
header .sub{color:var(--muted);font-size:13px}
header .spacer{flex:1}
.badge{font-size:11px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
  color:var(--muted)}
main{display:grid;grid-template-columns:290px 1fr;gap:0;min-height:calc(100vh - 62px)}
aside{border-right:1px solid var(--line);padding:18px 16px;background:var(--panel)}
aside h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--muted);margin:0 0 12px}
.sc{padding:12px 13px;border:1px solid var(--line);border-radius:9px;margin-bottom:9px;
  cursor:pointer;background:var(--panel2);transition:border-color .12s,transform .12s}
.sc:hover{border-color:#3d4658;transform:translateX(2px)}
.sc.on{border-color:var(--accent);background:#1b2233}
.sc .k{font-size:11px;color:var(--muted);letter-spacing:.06em}
.sc .t{font-weight:600;margin:3px 0 6px;font-size:13.5px;line-height:1.35}
.tag{font-size:10px;padding:2px 7px;border-radius:4px;letter-spacing:.05em;font-weight:600}
.tag.attack{background:var(--danger-bg);color:var(--danger)}
.tag.benign{background:var(--ok-bg);color:var(--ok)}
section{padding:22px 26px;overflow:auto}
.threat{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:8px;padding:13px 15px;margin-bottom:18px}
.threat .lb{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--warn);margin-bottom:5px;font-weight:600}
.goal{color:var(--muted);font-size:12.5px;margin-top:8px}
.goal b{color:var(--fg);font-weight:500}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:1000px){.cols{grid-template-columns:1fr}main{grid-template-columns:1fr}}
.col{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--panel)}
.col > .hd{padding:11px 15px;font-weight:650;font-size:13px;
  display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--line)}
.col.off > .hd{background:var(--danger-bg);color:var(--danger)}
.col.on  > .hd{background:var(--ok-bg);color:var(--ok)}
.dot{width:7px;height:7px;border-radius:99px;background:currentColor}
.body{padding:13px 15px}
.step{border-left:2px solid var(--line);padding:0 0 14px 13px;margin-left:5px;position:relative}
.step:last-child{padding-bottom:2px}
.step::before{content:"";position:absolute;left:-5px;top:5px;width:8px;height:8px;
  border-radius:99px;background:var(--line)}
.step.priv::before{background:var(--warn)}
.step.blocked::before{background:var(--danger)}
.step.blocked{border-left-color:var(--danger)}
.call{font-size:12.5px}
.call .fn{color:var(--accent)}
.args{color:var(--muted);font-size:11.5px;word-break:break-word;margin-top:2px}
.why{color:var(--muted);font-size:11.5px;font-style:italic;margin-top:3px}
.chips{margin-top:6px;display:flex;flex-wrap:wrap;gap:5px}
.chip{font-size:10.5px;padding:2px 7px;border-radius:4px;border:1px solid var(--line);
  color:var(--muted);background:#12151c}
.chip.u{color:var(--warn);border-color:#4a3a1c}
.chip.t{color:var(--ok);border-color:#1d4433}
.chip.s{color:var(--danger);border-color:#4a2226}
.viol{margin-top:9px;background:var(--danger-bg);border:1px solid #4a2226;
  border-radius:8px;padding:11px 12px}
.viol .rz{color:var(--danger);font-weight:650;font-size:12.5px;margin-bottom:8px}
.layer{font-size:11.5px;padding:4px 0;border-top:1px solid #3a1e22;color:var(--muted)}
.layer b{color:var(--danger);font-weight:600}
.layer.pass b{color:var(--muted)}
.effects{margin-top:12px;border-top:1px solid var(--line);padding-top:11px}
.fx{font-size:12px;padding:7px 10px;border-radius:6px;margin-bottom:6px;
  background:var(--danger-bg);color:var(--danger);font-weight:550}
.fx.none{background:var(--ok-bg);color:var(--ok);font-weight:500}
.fx.intended{background:var(--ok-bg);color:var(--ok);font-weight:500}
.verdict{margin-top:18px;padding:14px 16px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);font-size:13px}
.verdict .hl{font-weight:650}
.defence{margin-top:12px;color:var(--muted);font-size:12.5px;line-height:1.6}
.defence b{color:var(--fg)}
.empty{color:var(--muted);padding:40px;text-align:center}
</style></head><body>
<header>
  <h1>SafeOps</h1>
  <span class="sub">operations agent for a fintech back office · protected by AIsafePy</span>
  <span class="spacer"></span>
  <span class="badge" id="planner">planner: scripted</span>
  <span class="badge">policy: strict</span>
</header>
<main>
  <aside>
    <h2>Scenarios</h2>
    <div id="list"></div>
    <h2 style="margin-top:22px">Tools</h2>
    <div style="font-size:11.5px;color:var(--muted);line-height:1.9" class="mono">
      inbox.read <span style="color:var(--warn)">untrusted</span><br>
      kb.search <span style="color:var(--warn)">untrusted</span><br>
      crm.lookup <span style="color:var(--danger)">secrets</span><br>
      send_email <span style="color:var(--danger)">privileged</span><br>
      payments.refund <span style="color:var(--danger)">privileged</span><br>
      reply_to_user
    </div>
  </aside>
  <section id="out"><div class="empty">Pick a scenario.</div></section>
</main>
<script>
const SC = __SCENARIOS__;
let current = null;

function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

function renderList(){
  document.getElementById('list').innerHTML = SC.map(s=>`
    <div class="sc ${current===s.key?'on':''}" onclick="pick('${s.key}')">
      <div class="k">SCENARIO ${s.key}</div>
      <div class="t">${esc(s.title)}</div>
      <span class="tag ${s.kind}">${s.kind.toUpperCase()}</span>
    </div>`).join('');
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
    <div class="args mono">${esc(args)}</div>
    <div class="call mono">)</div>`;
  if(st.rationale) h += `<div class="why">“${esc(st.rationale)}”</div>`;
  const chips = [`<span class="chip ${st.cfi==='TRUSTED'?'t':'u'}">control flow: ${st.cfi}</span>`]
    .concat((st.capabilities||[]).map(capChip));
  if((st.provenance||[]).length) chips.push(`<span class="chip">from: ${st.provenance.join(', ')}</span>`);
  h += `<div class="chips">${chips.join('')}</div>`;
  if(st.violation){
    const fired = (st.layers||[]).filter(l=>l.fired);
    h += `<div class="viol">
      <div class="rz">REFUSED — ${esc(st.violation.reason)}</div>
      <div style="font-size:11.5px;color:var(--muted);margin-bottom:4px">
        ${fired.length} independent control${fired.length===1?'':'s'} rejected this call
      </div>
      ${(st.layers||[]).map(l=>`<div class="layer ${l.fired?'':'pass'}">
        <b>${l.fired?'✗':'○'} ${esc(l.control)}</b> — ${esc(l.detail)}</div>`).join('')}
    </div>`;
  }
  return h + `</div>`;
}

function colHtml(r, guarded, benign){
  // On a benign scenario a side effect is the *desired* outcome, so it is
  // shown in green. On an attack scenario the same event is the harm.
  const cls = benign ? 'fx intended' : 'fx';
  const pre = benign ? '' : ' ';
  const fx = [];
  (r.refunds||[]).forEach(x=>fx.push(
    `${benign?'REFUND ISSUED':'MONEY MOVED'} — LKR ${x.amount_lkr.toLocaleString()} to ${esc(x.recipient)}`));
  (r.sent_emails||[]).forEach(x=>fx.push(
    `${benign?'MAIL SENT':'DATA EXFILTRATED'} — to ${esc(x.to)}`));
  return `<div class="col ${guarded?'on':'off'}">
    <div class="hd"><span class="dot"></span>${guarded?'AIsafePy ON':'AIsafePy OFF'}</div>
    <div class="body">
      ${r.steps.map(stepHtml).join('')}
      <div class="effects">
        ${fx.length?fx.map(t=>`<div class="${cls}">${pre}${t}</div>`).join('')
                   :`<div class="fx none">No external side effects</div>`}
      </div>
    </div></div>`;
}

async function pick(k){
  current = k; renderList();
  document.getElementById('out').innerHTML = '<div class="empty">Running…</div>';
  const d = await (await fetch('/api/run?scenario='+k)).json();
  const s = d.scenario;
  const bad = d.unguarded.attack_succeeded, good = d.guarded.attack_succeeded;
  let verdict;
  if(s.kind==='attack'){
    verdict = `<span class="hl" style="color:var(--danger)">Guard off: attack succeeded.</span>
      <span class="hl" style="color:var(--ok)">Guard on: refused before execution.</span>`;
  } else {
    verdict = `<span class="hl" style="color:var(--ok)">Legitimate task completes identically
      with the guard on.</span> The policy discriminates; it does not simply refuse.`;
  }
  document.getElementById('out').innerHTML = `
    <div class="threat">
      <div class="lb">${s.kind==='attack'?'Threat':'Task'}</div>
      ${esc(s.threat)}
      <div class="goal">User asked: <b>${esc(s.goal)}</b></div>
    </div>
    <div class="cols">${colHtml(d.unguarded,false,s.kind==='benign')}${colHtml(d.guarded,true,s.kind==='benign')}</div>
    <div class="verdict">${verdict}
      <div class="defence"><b>Why:</b> ${esc(s.defence)}</div>
    </div>`;
}
renderList(); pick('A');
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console clean during a demo
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/run":
            key = parse_qs(u.query).get("scenario", ["A"])[0].upper()
            if key not in BY_KEY:
                return self._send(404, b"unknown scenario", "text/plain")
            return self._send(
                200, json.dumps(run_pair(key)).encode(), "application/json"
            )
        if u.path in ("/", "/index.html"):
            page = PAGE.replace(
                "__SCENARIOS__", json.dumps([asdict(s) for s in SCENARIOS])
            )
            return self._send(200, page.encode(), "text/html; charset=utf-8")
        self._send(404, b"not found", "text/plain")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8420) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"SafeOps demo → http://{host}:{port}   (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
