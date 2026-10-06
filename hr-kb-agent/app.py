"""Streamlit UI: HR Knowledge Base chat with an evidence panel and auditable leave calculations."""
import asyncio
import logging
import os
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from agent import MODEL, ask, log_sensitive, render_citations, root_cause, today

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
if not logging.getLogger("hr").handlers:  # Streamlit reruns this script; configure once
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_h = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    for h in (file_h, console_h):
        h.setFormatter(fmt)
        logging.getLogger("hr").addHandler(h)
    # Default: operational metadata only. HR_LOG_SENSITIVE=1 adds questions/answers/tool payloads (dev only).
    logging.getLogger("hr").setLevel(logging.DEBUG if log_sensitive() else logging.INFO)
log = logging.getLogger("hr.app")

EXAMPLES = {
    "Policy": ["How many sick days do I get per year?", "What about casual?", "Can I carry forward unused annual leave?"],
    "My data": ["What's my leave balance?", "Who is my manager?"],
    "Leave calculation": ["Can I take annual leave from 2026-10-16 to 2026-10-21?", "What if it was casual leave instead?"],
    "Ambiguous": ["I want to take some leave next month"],
    "Boundaries": ["What is Karan Singh's leave balance?", "What is the CEO's salary?"],
}
STATUS_ICON = {"pass": "✅", "fail": "❌", "approval": "🟡", "info": "ℹ️"}


def friendly_error(exc: BaseException) -> str:
    """User-facing message for common failures (mostly Groq API problems)."""
    name = type(exc).__name__
    return {
        "RateLimitError": "Groq free-tier rate limit reached. Wait about a minute and retry, or set "
                          "`GROQ_MODEL=openai/gpt-oss-20b` in `.env` (separate limit) and restart the app.",
        "APIConnectionError": "Can't reach the Groq API. Check your internet connection or proxy, then retry.",
        "APITimeoutError": "The Groq API timed out. Retry in a moment.",
        "AuthenticationError": "Groq rejected the API key. Check `GROQ_API_KEY` in `.env` and restart.",
        "PermissionDeniedError": "Groq refused the request for this key. Check your Groq account.",
        "NotFoundError": f"Model `{MODEL}` is not available to your key. Set `GROQ_MODEL` in `.env` (see README).",
        "InternalServerError": "Groq had a server error. Retry in a moment.",
    }.get(name, f"{name}: {exc}")


st.set_page_config(page_title="HR Knowledge Base", page_icon="📘", layout="wide")
st.title("📘 HR Knowledge Base")
st.caption("Ask about HR policies, your leave balance, eligibility and leave calculations. "
           "Answers come only from company policy documents and HR records.")

# --- Startup checks: fail with a clear message instead of a stack trace ---
problems = []
if not (ROOT / "hr.db").exists() or not (ROOT / "chroma").exists():
    problems.append("Demo data not found. Run: `.venv\\Scripts\\python seed.py`")
if not os.getenv("GROQ_API_KEY"):
    problems.append("`GROQ_API_KEY` is not set. Copy `.env.example` to `.env` and add your free key from console.groq.com.")
if problems:
    for p in problems:
        st.error(p)
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()  # contextual chat memory (per thread)

with sqlite3.connect(ROOT / "hr.db") as con:
    con.row_factory = sqlite3.Row
    employees = {r["id"]: dict(r) for r in con.execute("SELECT * FROM employees ORDER BY id")}

with st.sidebar:
    st.subheader("Sign in")
    st.warning("Demo only: picking an employee simulates sign-in. This is **not** real authentication.", icon="⚠️")
    emp_id = st.selectbox("Employee", list(employees),
                          format_func=lambda i: f"{employees[i]['name']} ({i}, {employees[i]['dept']})")
    emp = employees[emp_id]
    st.markdown(f"**{emp['role']}**, {emp['dept']}  \nManager: {emp['manager']}  \nJoined: {emp['join_date']}")
    new_chat = st.button("🔄 New chat", use_container_width=True)
    st.divider()
    st.subheader("Try these")
    for group, qs in EXAMPLES.items():
        st.caption(group)
        for q in qs:
            if st.button(q, key=f"ex-{q}", use_container_width=True):
                st.session_state.pending_q = q
    st.divider()
    st.caption(f"Today (demo date): {today()}  \nModel: {MODEL} · temperature 0")
    if log_sensitive():
        st.error("HR_LOG_SENSITIVE=1: questions, answers and HR data are being written to logs/app.log.")

# Reset the chat when the employee changes or "New chat" is clicked
if new_chat or st.session_state.get("emp_id") != emp["id"]:
    st.session_state.emp_id = emp["id"]
    st.session_state.thread_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []
    log.info("chat start thread=%s", st.session_state.thread_id)


def render_calc_audit(res: dict):
    if "error" in res:
        st.warning(f"Calculation not possible: {res['error']}")
        return
    req = res["request"]
    st.markdown(f"**Request:** {req['leave_type']} leave, {req['start_date']} → {req['end_date']} "
                f"(checked on {req['today']})")
    c = st.columns(4)
    c[0].metric("Calendar days", res["calendar_days"])
    c[1].metric("Working days counted", res["working_days"])
    c[2].metric("Balance before", res["balance"]["remaining_before"])
    c[3].metric("Balance after", res["balance"]["remaining_after"],
                delta=f"{res['balance']['lwp_days']} day(s) LWP" if res["balance"]["lwp_days"] else None,
                delta_color="inverse")
    if res["excluded_days"]:
        st.markdown("**Excluded days:** " + ", ".join(f"{d['date']} ({d['reason']})" for d in res["excluded_days"]))
    else:
        st.markdown("**Excluded days:** none")
    rows = "\n".join(f"| {STATUS_ICON[r['status']]} | {r['rule']} | {r['policy_section']} | {r['detail']} |"
                     for r in res["rules"])
    st.markdown("**Rules applied** (✅ pass · ❌ blocks the request · 🟡 needs approval · ℹ️ note)\n\n"
                "| | Rule | Source | Result |\n|---|---|---|---|\n" + rows)
    approvals = [r["detail"] for r in res["rules"] if r["status"] == "approval"]
    notes = [r["detail"] for r in res["rules"][1:] if r["status"] == "info"]
    if res["outcome"] == "not_eligible":
        st.error("**Not eligible**  \n" + "  \n".join(
            f"• {r['detail']} ({r['policy_section']})" for r in res["rules"] if r["status"] == "fail"))
    elif res["outcome"] == "eligible_subject_to_approval":
        st.warning("**Eligible, subject to approval** (not yet approved)  \n" + "  \n".join(f"• {a}" for a in approvals))
    else:
        st.success("**Eligible**: no rule blocks this request and no extra approval is required by policy.")
    if notes:
        st.info("  \n".join(f"ℹ️ {n}" for n in notes))


def render_step(step: dict, cited_ids: set[str]):
    res = step["result"]
    if "error" in res:
        st.warning(f"Tool returned an error: {res['error']}")
    elif step["tool"] == "search_hr_policies":
        for r in res["results"]:
            badge = " · ✅ **cited in answer**" if r["id"] in cited_ids else ""
            st.markdown(f"📄 **{r['section']}** `[{r['id']}]` (distance {r['distance']}){badge}")
            st.caption(r["text"][:400] + ("…" if len(r["text"]) > 400 else ""))
    elif step["tool"] == "get_employee_context":
        for r in res["results"]:
            st.caption(f"👤 {r['text']} (distance {r['distance']})")
    elif step["tool"] == "get_leave_balance":
        st.markdown("| Leave type | Entitled | Used | Remaining |\n|---|---|---|---|\n" + "\n".join(
            f"| {b['leave_type']} | {b['entitled']} | {b['used']} | {b['remaining']} |" for b in res["balances"]))
        st.caption(f"{len(res['requests'])} leave request(s) on record were also returned.")
    elif step["tool"] == "calculate_leave":
        st.caption(f"Outcome: {res['outcome'].replace('_', ' ')}. Full breakdown in the "
                   "**Leave calculation audit** panel.")


def render_evidence(trace: list[dict], citations: list[dict]):
    for s in (s for s in trace if s["tool"] == "calculate_leave"):
        with st.expander("🧮 Leave calculation audit", expanded=True):
            render_calc_audit(s["result"])

    valid = [c for c in citations if c["valid"]]
    invalid = [c for c in citations if not c["valid"]]
    used_policy = any(s["tool"] in ("search_hr_policies", "calculate_leave") and s["ok"] for s in trace)
    label = f"🔍 How I answered ({len(trace)} tool call{'s' if len(trace) != 1 else ''})"
    with st.expander(label + (" ⚠️" if invalid else "")):
        st.caption("Workflow trace: the tools that ran, the inputs the assistant gave them, and what they returned. "
                   "It is not the model's internal reasoning.")
        if not trace:
            st.info("No tools were called in this turn. The reply is a clarifying question or a refusal, "
                    "or it reuses evidence already retrieved earlier in this chat.")
        for i, step in enumerate(trace, 1):
            args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            st.markdown(f"**Step {i}: `{step['tool']}`**({args}) · {step['ms']} ms · {'ok' if step['ok'] else 'error'}")
            render_step(step, {c["id"] for c in valid})
        st.markdown("**Citations:** " + (", ".join(f"✅ {c['section']} `[{c['id']}]`" for c in valid) or "none"))
        if invalid:
            st.error("Cited ids that were never retrieved in this chat: "
                     + ", ".join(f"`[{c['id']}]`" for c in invalid) + ". Treat those statements as unverified.")
        elif used_policy and not valid:
            st.caption("Policy evidence was retrieved but the answer cites none of it.")
        st.caption(f"Employee data can only come from the signed-in employee ({emp['name']}, {emp['id']}): "
                   "the MCP server is bound to that ID for this turn, and no tool accepts another ID.")
        st.markdown("**Final answer**: written from the evidence above only; citations are checked against it.")


def show_assistant(m: dict):
    st.markdown(render_citations(m["content"], m.get("citations", [])))
    if "trace" in m:
        render_evidence(m["trace"], m["citations"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            show_assistant(m)
        else:
            st.markdown(m["content"])

question = st.chat_input("e.g. Can I take annual leave from 2026-11-16 to 2026-11-18?") or st.session_state.pop("pending_q", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking HR records..."):
            try:
                out = asyncio.run(ask(emp["id"], emp["name"], question, st.session_state.thread_id,
                                      st.session_state.memory))
                msg = {"role": "assistant", "content": out["answer"], "trace": out["trace"],
                       "citations": out["citations"]}
            except Exception as e:
                cause = root_cause(e)
                log.error("turn failed thread=%s error=%s", st.session_state.thread_id, type(cause).__name__,
                          exc_info=cause if log_sensitive() else None)
                msg = {"role": "assistant", "content": f"⚠️ {friendly_error(cause)}"}
        show_assistant(msg)
    st.session_state.messages.append(msg)
