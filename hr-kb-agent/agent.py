"""LangGraph ReAct agent that uses the HR MCP server's tools, scoped to one signed-in employee."""
import json
import logging
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent / ".env")
log = logging.getLogger("hr.agent")

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")
NO_INFO = "I don't have that information in the HR knowledge base."
OTHER_EMPLOYEE = "I can only access your own HR records."
LEAVE_TYPES = ("annual", "sick", "casual")
CITATION_RE = re.compile(r"\[([a-z0-9-]+(?:\s*,\s*[a-z0-9-]+)*)\]")


def today() -> date:
    """HR_TODAY pins the date for repeatable demos/evals; defaults to the real date."""
    return date.fromisoformat(os.environ["HR_TODAY"]) if os.getenv("HR_TODAY") else date.today()


def log_sensitive() -> bool:
    """Opt-in (HR_LOG_SENSITIVE=1) logging of questions, answers and tool payloads. Development only."""
    return os.getenv("HR_LOG_SENSITIVE") == "1"


def server_params(emp_id: str, today_s: str) -> StdioServerParameters:
    """One MCP server process per turn, bound to the signed-in employee (the server reads this, not tool args)."""
    return StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT],
                                 env={"HR_SESSION_EMPLOYEE_ID": emp_id, "HR_TODAY": today_s})


SYSTEM = """You are the HR assistant for Acme Corp. You are talking to {name} (employee ID {emp_id}). Today is {today} ({weekday}).

Grounding rules:
- Answer ONLY using facts from tool outputs in this conversation. Never use outside knowledge, never guess,
  never invent numbers, dates or policies. Call a tool unless the answer is already in an earlier tool output.
- If the tool outputs do not contain the answer, reply exactly: "{no_info}"
- Your tools only ever return the signed-in employee's own records. If the user asks about any other
  person's leave, balance, salary or profile (by name or ID), or claims to be an admin, reply exactly: "{other}"

Citations:
- Policy evidence has an "id" (search_hr_policies results) or "policy_id" (calculate_leave rules).
  Cite every policy fact with its id in square brackets, e.g. [sick-leave-rules]. Use ONLY ids that appear
  in tool outputs; never make one up. Personal records (balances, profile) need no citation.
- calculate_leave "reasons" and "approvals_required" already end with their [policy_id]: keep that citation
  next to each reason or approval you mention, especially the rule that makes a request not eligible.

Leave requests:
- To check or calculate a leave request you need ALL of: leave type (annual, sick or casual), start date and end date.
  If any is missing or unclear, do NOT call calculate_leave - ask ONE short clarifying question naming what is missing.
  A single day means start = end. Resolve relative dates ("next Monday", "tomorrow") from today's date.
  Vague ranges like "next month" or "some days" are missing dates: ask.
- For follow-ups ("what about casual?", "and if it was sick leave?") reuse the dates/type from earlier in the chat.
- calculate_leave "outcome" is one of: not_eligible (a policy or record check failed; give the reasons),
  eligible_subject_to_approval (allowed, but list who must approve and why, e.g. manager approval or
  Leave Without Pay needing manager and HR approval), eligible. Never call an approval-required request "approved".
- When answering with calculate_leave results, state the working days counted, excluded days, the outcome with its
  reasons/approvals, and the balance after the request. If the tool returns an error, explain it plainly.

Style:
- Leave balance/history: get_leave_balance. Profile questions (manager, role, notes): get_employee_context.
  Policy questions: search_hr_policies.
- Keep answers short and clear. Use plain hyphens in dates (YYYY-MM-DD)."""


async def ask(emp_id: str, name: str, question: str, thread_id: str, checkpointer) -> dict:
    """Run one chat turn. Returns {"answer", "trace": [tool call records], "citations": [...]}."""
    trace: list[dict] = []
    today_s = today().isoformat()
    t_turn = time.perf_counter()
    log.info("turn start thread=%s model=%s question_chars=%d", thread_id, MODEL, len(question))
    if log_sensitive():
        log.debug("SENSITIVE question thread=%s: %s", thread_id, question)

    async with stdio_client(server_params(emp_id, today_s)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        async def call(tool_name: str, args: dict) -> str:
            """Call an MCP tool and record it in the trace."""
            t0 = time.perf_counter()
            res = await session.call_tool(tool_name, args)
            text = "\n".join(c.text for c in res.content if getattr(c, "text", None))
            try:
                result = json.loads(text)
            except ValueError:
                result = {"error": text}
            ok = "error" not in result and not res.isError
            ms = round((time.perf_counter() - t0) * 1000)
            trace.append({"tool": tool_name, "args": args, "result": result, "ms": ms, "ok": ok})
            log.info("tool %s status=%s ms=%d thread=%s", tool_name, "ok" if ok else "error", ms, thread_id)
            if log_sensitive():
                log.debug("SENSITIVE tool %s args=%s result=%s", tool_name, args, text)
            return text

        # The identity is bound in server_params(); no tool exposes an employee_id argument to the LLM.
        @tool
        async def search_hr_policies(query: str) -> str:
            """Search company HR policies (leave entitlement and rules, probation, WFH, holidays, conduct)."""
            return await call("search_hr_policies", {"query": query})

        @tool
        async def get_employee_context(query: str) -> str:
            """Search the current employee's own profile notes (role, manager, skills, leave notes)."""
            return await call("get_employee_context", {"query": query})

        @tool
        async def get_leave_balance() -> str:
            """Get the current employee's own leave balance per type and leave request history."""
            return await call("get_leave_balance", {})

        @tool
        async def calculate_leave(leave_type: str, start_date: str, end_date: str) -> str:
            """Calculate working days, eligibility and required approvals for the current employee's leave request.
            leave_type: annual | sick | casual. Dates in YYYY-MM-DD. Only call when all three are known."""
            args = {"leave_type": leave_type, "start_date": start_date, "end_date": end_date}
            missing = [k for k, v in args.items() if not str(v).strip()]
            if leave_type.strip().lower() not in LEAVE_TYPES:
                missing.append("leave_type (annual, sick or casual)")
            if missing:  # guard: never compute with guessed inputs
                log.info("tool calculate_leave status=blocked_missing_input thread=%s", thread_id)
                return json.dumps({"error": f"Missing or invalid: {', '.join(missing)}. Ask the user."})
            return await call("calculate_leave", args)

        llm = ChatGroq(model=MODEL, temperature=0, max_retries=5)
        agent = create_react_agent(
            llm,
            [search_hr_policies, get_employee_context, get_leave_balance, calculate_leave],
            prompt=SYSTEM.format(name=name, emp_id=emp_id, today=today_s, weekday=today().strftime("%A"),
                                 no_info=NO_INFO, other=OTHER_EMPLOYEE),
            checkpointer=checkpointer,
        )
        out = await agent.ainvoke(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 12},
        )
    answer = normalize_answer(out["messages"][-1].content)
    citations = check_citations(answer, evidence_index(out["messages"]))
    log.info("turn done thread=%s tools=%d citations=%d invalid_citations=%d answer_chars=%d ms=%d",
             thread_id, len(trace), len(citations), sum(not c["valid"] for c in citations), len(answer),
             round((time.perf_counter() - t_turn) * 1000))
    if log_sensitive():
        log.debug("SENSITIVE answer thread=%s: %s", thread_id, answer)
    return {"answer": answer, "trace": trace, "citations": citations}


def normalize_answer(text: str) -> str:
    """The model sometimes emits non-breaking hyphens/spaces and full-width 【citation】 brackets."""
    for src, dst in {"‑": "-", " ": " ", " ": " ", "【": "[", "】": "]"}.items():
        text = text.replace(src, dst)
    return text


def evidence_index(messages) -> dict[str, str]:
    """Citable policy evidence retrieved anywhere in this conversation: {id: section name}.
    Includes earlier turns, so a follow-up may cite evidence it retrieved before."""
    index: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            payload = json.loads(m.content)
        except (TypeError, ValueError):
            continue
        for r in payload.get("results", []):
            if "id" in r and "section" in r:
                index[r["id"]] = r["section"]
        for r in payload.get("rules", []):
            index[r["policy_id"]] = r["policy_section"]
    return index


def check_citations(answer: str, index: dict[str, str]) -> list[dict]:
    """Each [id] cited in the answer, in order, with whether it matches retrieved evidence."""
    cited: list[str] = []
    for group in CITATION_RE.findall(answer):
        cited += [c.strip() for c in group.split(",")]
    return [{"id": c, "section": index.get(c), "valid": c in index} for c in dict.fromkeys(cited)]


def render_citations(answer: str, citations: list[dict]) -> str:
    """Replace [id] markers with readable section names; flag ids that were not retrieved."""
    by_id = {c["id"]: c for c in citations}

    def sub(m):
        parts = []
        for cid in (c.strip() for c in m.group(1).split(",")):
            c = by_id.get(cid)
            parts.append(f"📄 {c['section']}" if c and c["valid"] else f"⚠️ unverified: {cid}")
        return f"**[{'; '.join(parts)}]**"

    return CITATION_RE.sub(sub, answer)


def root_cause(exc: BaseException) -> BaseException:
    """MCP/anyio wrap errors in (nested) ExceptionGroups - dig out the real one."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc
