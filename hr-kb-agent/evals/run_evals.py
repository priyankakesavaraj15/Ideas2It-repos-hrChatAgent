"""Repeatable evaluation of the live HR agent (needs GROQ_API_KEY + network) against evals/cases.json.

Usage:  .venv\\Scripts\\python evals\\run_evals.py [--only CASE_ID ...] [--delay SECONDS]
Pins the demo date (HR_TODAY=2026-10-05) so leave-notice rules give the same results every run.
Writes a Markdown report to evals/results.md (a snapshot of that run only) and exits non-zero on any failure.

Checks applied to every case, in addition to its own "expect" block:
  [citations]   every [id] the answer cites must match evidence retrieved in the conversation
  [tool-errors] no tool may return an error unless the case sets "allow_tool_errors": true
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HR_TODAY"] = "2026-10-05"  # must be set before the agent reads it

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agent import MODEL, ask, root_cause  # noqa: E402

EMPLOYEE_NAMES = {"E001": "Asha Rao", "E002": "Rahul Mehta", "E003": "Priya Nair",
                  "E004": "Karan Singh", "E005": "Meera Iyer"}


def norm(s: str) -> str:
    return (s.lower().replace("‑", "-").replace("’", "'").replace("*", "").replace(",", ""))


def has(text: str, needle: str) -> bool:
    needle = norm(needle)
    if needle.isdigit():  # whole-number match so "6" doesn't match "2026"
        return re.search(rf"(?<!\d){needle}(?!\d)", text) is not None
    return needle in text


def calc_facts(result: dict) -> dict:
    """The calculate_leave fields a case can assert on."""
    return {
        "working_days": result["working_days"], "eligible": result["eligible"], "outcome": result["outcome"],
        "leave_type": result["request"]["leave_type"], "lwp_days": result["balance"]["lwp_days"],
        "excluded_days": len(result["excluded_days"]),
        "failed_rules": sorted({r["policy_id"] for r in result["rules"] if r["status"] == "fail"}),
        "approval_rules": sorted({r["policy_id"] for r in result["rules"] if r["status"] == "approval"}),
    }


def check(case: dict, answer: str, trace: list[dict], citations: list[dict] | None = None) -> list[str]:
    """Returns failure messages, each prefixed with the check that failed."""
    exp, a, fails = case["expect"], norm(answer), []
    citations = citations or []
    called = [s["tool"] for s in trace]

    # [tools]
    for t in exp.get("tools_called", []):
        if t not in called:
            fails.append(f"[tools] expected {t} to be called; called: {called or 'none'}")
    if exp.get("tools_called_any") and not set(exp["tools_called_any"]) & set(called):
        fails.append(f"[tools] expected one of {exp['tools_called_any']}; called: {called or 'none'}")
    for t in exp.get("tools_not_called", []):
        if t in called:
            fails.append(f"[tools] {t} must not be called (it was)")

    # [answer]
    for n in exp.get("answer_contains_all", []):
        if not has(a, n):
            fails.append(f"[answer] missing '{n}'")
    if exp.get("answer_contains_any") and not any(has(a, n) for n in exp["answer_contains_any"]):
        fails.append(f"[answer] contains none of {exp['answer_contains_any']}")
    for n in exp.get("answer_not_contains", []):
        if has(a, n):
            fails.append(f"[answer] must not contain '{n}'")

    # [privacy]
    trace_text = json.dumps(trace)
    for n in exp.get("trace_not_contains", []):
        if n in trace_text:
            fails.append(f"[privacy] trace contains '{n}'")

    # [tool-errors]
    errors = [f"{s['tool']}: {s['result']['error']}" for s in trace if "error" in s["result"]]
    if errors and not case.get("allow_tool_errors"):
        fails.append(f"[tool-errors] unexpected tool error(s): {errors}")
    if "tool_error_contains" in exp and not any(exp["tool_error_contains"] in e for e in errors):
        fails.append(f"[tool-errors] expected an error containing '{exp['tool_error_contains']}'; got {errors or 'none'}")

    # [calc]
    calcs = [s["result"] for s in trace if s["tool"] == "calculate_leave" and "error" not in s["result"]]
    if exp.get("no_successful_calc") and calcs:
        fails.append(f"[calc] expected no successful calculation; got {[calc_facts(c) for c in calcs]}")
    if "calc" in exp:
        if not calcs:
            fails.append("[calc] no successful calculate_leave result")
        else:
            got = calc_facts(calcs[-1])
            for k, v in exp["calc"].items():
                if (sorted(v) if isinstance(v, list) else v) != got[k]:
                    fails.append(f"[calc] {k} = {got[k]!r}, expected {v!r}")

    # [citations]
    for c in citations:
        if not c["valid"]:
            fails.append(f"[citations] cited [{c['id']}] which was never retrieved")
    if exp.get("cites_any") and not {c["id"] for c in citations if c["valid"]} & set(exp["cites_any"]):
        fails.append(f"[citations] expected a valid citation of one of {exp['cites_any']}; "
                     f"got {[c['id'] for c in citations] or 'none'}")
    return fails


async def run_case(case: dict) -> dict:
    mem, thread = MemorySaver(), f"eval-{case['id']}"
    emp = case["employee"]
    out = {"answer": "", "trace": [], "citations": []}
    for q in case["turns"]:  # earlier turns build chat context; checks apply to the final turn
        out = await ask(emp, EMPLOYEE_NAMES[emp], q, thread, mem)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="case ids to run")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between cases (free-tier rate limits)")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (.env). Live evals need it; offline checks: python -m pytest")

    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    rows, n_pass = [], 0
    print(f"Running {len(cases)} cases with {MODEL} (HR_TODAY={os.environ['HR_TODAY']})\n")
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        out = {"answer": "", "trace": [], "citations": []}
        try:
            out = asyncio.run(run_case(case))
            fails = check(case, out["answer"], out["trace"], out["citations"])
        except Exception as e:
            fails = [f"[run] {type(root_cause(e)).__name__}: {root_cause(e)}"]
        ok = not fails
        n_pass += ok
        tools = ", ".join(s["tool"] for s in out["trace"]) or "-"
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<30} [{case['category']}] "
              f"{time.perf_counter() - t0:4.1f}s  tools: {tools}")
        for f in fails:
            print(f"      - {f}")
        if not ok:
            print(f"      answer: {out['answer'][:300]!r}")
        rows.append((case, ok, fails, tools))
        if i < len(cases) - 1:
            time.sleep(args.delay)

    print(f"\n{n_pass}/{len(cases)} passed")
    report = [f"# Eval results (snapshot)\n\nRun {datetime.now():%Y-%m-%d %H:%M} · model `{MODEL}` · "
              f"HR_TODAY {os.environ['HR_TODAY']} · **{n_pass}/{len(cases)} passed**\n\n"
              "This records one run of a live LLM. It is not a guarantee: re-run after any change.\n",
              "| Result | Case | Category | Tools (final turn) | Failures |", "|---|---|---|---|---|"]
    for case, ok, fails, tools in rows:
        report.append(f"| {'✅' if ok else '❌'} | {case['id']} | {case['category']} | {tools} | "
                      f"{'<br>'.join(fails).replace('|', '/')} |")
    (ROOT / "evals" / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    sys.exit(0 if n_pass == len(cases) else 1)


if __name__ == "__main__":
    main()
