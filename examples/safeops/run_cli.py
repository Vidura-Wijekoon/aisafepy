#!/usr/bin/env python3
"""SafeOps — terminal demo.

    python run_cli.py              # all four scenarios, guard off vs on
    python run_cli.py --scenario A # just the injection attack
    python run_cli.py --llm        # use a real LLM planner (needs OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import sys

from safeops.agent import run
from safeops.planner import OpenAIPlanner, ScriptedPlanner
from safeops.scenarios import BY_KEY, SCENARIOS
from safeops.world import fresh_world

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW, BLUE = "\033[31m", "\033[32m", "\033[33m", "\033[34m"


def _c(text: str, colour: str, on: bool) -> str:
    return f"{colour}{text}{RESET}" if on else text


def show(scenario, guarded: bool, planner, colour: bool) -> bool:
    result = run(
        scenario.goal,
        guarded=guarded,
        planner=planner,
        world=fresh_world(),
        expect_harm=scenario.expect_harm,
    )
    label = "AIsafePy ON " if guarded else "AIsafePy OFF"
    head = _c(label, GREEN if guarded else RED, colour)
    print(f"  {head}  {DIM if colour else ''}goal: {scenario.goal}{RESET if colour else ''}")

    for s in result.steps:
        if s.outcome == "blocked" and s.violation:
            print(f"    {_c('BLOCKED', RED, colour)} step {s.n}: {s.tool}")
            print(f"      reason      : {s.violation['reason']}")
            print(f"      provenance  : {s.violation['provenance']}")
            print(f"      capabilities: {s.violation['capabilities']}")
            print(f"      permitted   : {s.violation['required_capabilities']}")
            print(f"      integrity   : {s.violation['integrity']}   cfi: {s.cfi}")
            fired = [x for x in s.layers if x["fired"]]
            if fired:
                print(f"      {len(fired)} independent control(s) refused this call:")
                for x in fired:
                    print(f"        ✗ {x['control']} — {x['detail']}")
        else:
            mark = "!" if s.privileged else "·"
            print(f"    {mark} step {s.n}: {s.tool}({_kw(s.kwargs)})")
            if s.privileged:
                print(f"        {_c('EXECUTED', YELLOW, colour)} cfi={s.cfi} caps={s.capabilities}")

    for r in result.refunds:
        print(f"    {_c('MONEY MOVED', RED, colour)}: LKR {r['amount_lkr']:,} → {r['recipient']}")
    for e in result.sent_emails:
        print(f"    {_c('MAIL SENT', RED, colour)}: → {e['to']}")

    print(f"    {DIM if colour else ''}{result.summary}{RESET if colour else ''}\n")
    return result.attack_succeeded


def _kw(kwargs: dict) -> str:
    parts = []
    for k, v in kwargs.items():
        sv = str(v).replace("\n", " ")
        parts.append(f"{k}={sv[:38] + '…' if len(sv) > 38 else sv}")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="A, B, C or D. Default: all.")
    ap.add_argument("--llm", action="store_true", help="use a real LLM planner")
    ap.add_argument("--no-colour", action="store_true")
    args = ap.parse_args()

    colour = not args.no_colour and sys.stdout.isatty()
    planner = OpenAIPlanner() if args.llm else ScriptedPlanner()

    chosen = [BY_KEY[args.scenario.upper()]] if args.scenario else SCENARIOS

    print(f"\n{BOLD if colour else ''}SafeOps — AIsafePy reference agent{RESET if colour else ''}")
    print(f"{DIM if colour else ''}planner: {planner.name}{RESET if colour else ''}\n")

    harm_unguarded = harm_guarded = 0
    for sc in chosen:
        tag = _c("ATTACK", RED, colour) if sc.kind == "attack" else _c("BENIGN", BLUE, colour)
        print(f"{BOLD if colour else ''}[{sc.key}] {sc.title}{RESET if colour else ''}  {tag}")
        print(f"  {DIM if colour else ''}{sc.threat}{RESET if colour else ''}\n")
        harm_unguarded += show(sc, False, planner, colour)
        harm_guarded += show(sc, True, planner, colour)
        print(f"  → {sc.defence}\n")

    attacks = sum(1 for s in chosen if s.kind == "attack")
    print(f"{BOLD if colour else ''}Result{RESET if colour else ''}")
    print(f"  attacks attempted        : {attacks}")
    print(f"  succeeded, guard OFF     : {harm_unguarded}/{attacks}")
    print(f"  succeeded, guard ON      : {harm_guarded}/{attacks}")
    benign = [s for s in chosen if s.kind == "benign"]
    print(f"  benign tasks still working: {len(benign)}/{len(benign)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
