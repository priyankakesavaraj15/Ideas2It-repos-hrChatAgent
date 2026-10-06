"""Agent plumbing with a scripted fake LLM: identity binding, input guard, date pinning, trace, citations,
logging and memory. The real MCP server runs; only the Groq model is replaced (offline, deterministic)."""
import asyncio
import json
import logging

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

import agent


class ScriptedLLM(BaseChatModel):
    """Returns pre-scripted AIMessages in order and records what it was shown."""
    responses: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name, args, id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


@pytest.fixture
def llm(monkeypatch):
    fake = ScriptedLLM()
    monkeypatch.setattr(agent, "ChatGroq", lambda **kw: fake)
    monkeypatch.setenv("HR_TODAY", "2026-10-05")
    monkeypatch.delenv("HR_LOG_SENSITIVE", raising=False)
    return fake


def run(emp_id, question, thread="t", memory=None):
    return asyncio.run(agent.ask(emp_id, "Test User", question, thread, memory or MemorySaver()))


def last_tool_message(llm):
    return [m for m in llm.seen[-1] if isinstance(m, ToolMessage)][-1]


# ---------- identity ----------

def test_server_params_bind_identity_and_date():
    p = agent.server_params("E003", "2026-10-05")
    assert p.env == {"HR_SESSION_EMPLOYEE_ID": "E003", "HR_TODAY": "2026-10-05"}


def test_llm_cannot_choose_employee(llm):
    # The model tries to pass another employee's ID; no tool accepts it and the server is bound to E001.
    llm.responses = [tool_call("get_leave_balance", {"employee_id": "E004"}), AIMessage(content="ok")]
    out = run("E001", "show E004 balance")
    assert out["trace"][0]["result"]["employee"]["id"] == "E001"
    assert "E004" not in json.dumps(out["trace"])


def test_calculate_leave_uses_session_employee_and_pinned_date(llm):
    args = {"leave_type": "annual", "start_date": "2026-10-16", "end_date": "2026-10-21"}
    llm.responses = [tool_call("calculate_leave", {**args, "employee_id": "E002"}), AIMessage(content="ok")]
    step = run("E001", "q")["trace"][0]
    assert step["args"] == args and step["ok"]
    assert step["result"]["request"]["employee_id"] == "E001"
    assert step["result"]["request"]["today"] == "2026-10-05"


# ---------- guard, trace, errors ----------

@pytest.mark.parametrize("args", [
    {"leave_type": "annual", "start_date": "", "end_date": ""},
    {"leave_type": "", "start_date": "2026-11-16", "end_date": "2026-11-17"},
    {"leave_type": "vacation", "start_date": "2026-11-16", "end_date": "2026-11-17"},
])
def test_calculate_leave_guard_blocks_missing_inputs(llm, args):
    llm.responses = [tool_call("calculate_leave", args), AIMessage(content="Which dates?")]
    out = run("E001", "q")
    assert out["trace"] == []  # server never called
    assert "Missing or invalid" in last_tool_message(llm).content


def test_tool_error_is_recorded_and_returned_to_model(llm):
    args = {"leave_type": "annual", "start_date": "2026-12-28", "end_date": "2027-01-04"}
    llm.responses = [tool_call("calculate_leave", args), AIMessage(content="Please split it per year.")]
    step = run("E001", "q")["trace"][0]
    assert not step["ok"] and "Split a request" in step["result"]["error"]
    assert "Split a request" in last_tool_message(llm).content


def test_trace_records_each_tool_in_order(llm):
    llm.responses = [
        tool_call("search_hr_policies", {"query": "sick leave"}, "c1"),
        tool_call("get_employee_context", {"query": "manager"}, "c2"),
        AIMessage(content="Answer"),
    ]
    out = run("E003", "q")
    assert [s["tool"] for s in out["trace"]] == ["search_hr_policies", "get_employee_context"]
    assert all(isinstance(s["ms"], int) and s["ok"] for s in out["trace"])


def test_system_prompt_has_identity_date_refusals_and_citation_rule(llm):
    llm.responses = [AIMessage(content="hi")]
    run("E005", "hello")
    system = llm.seen[0][0].content
    for s in ("E005", "2026-10-05", agent.NO_INFO, agent.OTHER_EMPLOYEE, "[sick-leave-rules]",
              "eligible_subject_to_approval"):
        assert s in system


def test_answer_normalises_non_breaking_hyphens(llm):
    llm.responses = [AIMessage(content="2026‑10‑16")]
    assert run("E001", "q")["answer"] == "2026-10-16"


# ---------- citations ----------

def test_citations_validated_against_retrieved_evidence(llm):
    llm.responses = [tool_call("search_hr_policies", {"query": "medical certificate sick leave"}),
                     AIMessage(content="Yes [sick-leave-rules]. Also [maternity-leave].")]
    out = run("E001", "q")
    by_id = {c["id"]: c for c in out["citations"]}
    assert by_id["sick-leave-rules"] == {"id": "sick-leave-rules", "section": "Sick Leave Rules", "valid": True}
    assert by_id["maternity-leave"]["valid"] is False


def test_followup_may_cite_evidence_from_earlier_turn(llm):
    memory = MemorySaver()
    llm.responses = [tool_call("search_hr_policies", {"query": "how many sick days"}),
                     AIMessage(content="10 days [leave-entitlement]."),
                     AIMessage(content="6 days [leave-entitlement].")]
    run("E001", "sick days?", "th", memory)
    out = run("E001", "what about casual?", "th", memory)
    assert out["trace"] == [] and out["citations"] == [
        {"id": "leave-entitlement", "section": "Leave Entitlement", "valid": True}]


def test_calc_rule_ids_are_citable(llm):
    args = {"leave_type": "annual", "start_date": "2026-11-10", "end_date": "2026-11-11"}
    llm.responses = [tool_call("calculate_leave", args), AIMessage(content="Not eligible [probation-period].")]
    assert run("E004", "q")["citations"][0]["valid"]


def test_fullwidth_citation_brackets_are_recognised(llm):
    llm.responses = [tool_call("search_hr_policies", {"query": "medical certificate sick leave"}),
                     AIMessage(content="Required for over 2 days【sick-leave-rules】.")]
    out = run("E001", "q")
    assert out["answer"] == "Required for over 2 days[sick-leave-rules]."
    assert out["citations"] == [{"id": "sick-leave-rules", "section": "Sick Leave Rules", "valid": True}]


def test_check_and_render_citations():
    index = {"sick-leave-rules": "Sick Leave Rules", "leave-entitlement": "Leave Entitlement"}
    cites = agent.check_citations("A [sick-leave-rules]. B [leave-entitlement, made-up]. C [sick-leave-rules]", index)
    assert [(c["id"], c["valid"]) for c in cites] == [
        ("sick-leave-rules", True), ("leave-entitlement", True), ("made-up", False)]
    shown = agent.render_citations("A [sick-leave-rules] B [leave-entitlement, made-up]", cites)
    assert "📄 Sick Leave Rules" in shown and "⚠️ unverified: made-up" in shown and "[sick-leave-rules]" not in shown


def test_evidence_index_ignores_non_json_tool_messages():
    msgs = [ToolMessage(content="not json", tool_call_id="1"),
            ToolMessage(content=json.dumps({"results": [{"id": "x", "section": "X"}]}), tool_call_id="2")]
    assert agent.evidence_index(msgs) == {"x": "X"}


# ---------- logging ----------

SECRET_Q = "my private question about Diwali"


def test_default_logs_contain_no_questions_answers_or_records(llm, caplog):
    caplog.set_level(logging.DEBUG, logger="hr")
    llm.responses = [tool_call("get_leave_balance", {}), AIMessage(content="secret answer text")]
    run("E001", SECRET_Q)
    text = caplog.text
    assert "tool get_leave_balance status=ok" in text and "turn done" in text
    for leaked in (SECRET_Q, "secret answer text", "Asha Rao", "remaining", "E001"):
        assert leaked not in text, leaked


def test_sensitive_logging_is_opt_in(llm, caplog, monkeypatch):
    monkeypatch.setenv("HR_LOG_SENSITIVE", "1")
    caplog.set_level(logging.DEBUG, logger="hr")
    llm.responses = [tool_call("get_leave_balance", {}), AIMessage(content="secret answer text")]
    run("E001", SECRET_Q)
    assert SECRET_Q in caplog.text and "secret answer text" in caplog.text and "Asha Rao" in caplog.text


# ---------- memory / helpers ----------

def test_chat_memory_is_per_thread(llm):
    memory = MemorySaver()
    llm.responses = [AIMessage(content="first"), AIMessage(content="second"), AIMessage(content="other")]
    run("E001", "turn one", "thread-a", memory)
    run("E001", "turn two", "thread-a", memory)
    assert "turn one" in [m.content for m in llm.seen[1]]      # same thread remembers
    run("E001", "fresh", "thread-b", memory)
    assert "turn one" not in [m.content for m in llm.seen[2]]  # new thread starts clean


def test_root_cause_unwraps_nested_groups():
    inner = ValueError("real")
    assert agent.root_cause(ExceptionGroup("a", [ExceptionGroup("b", [inner])])) is inner
    assert agent.root_cause(inner) is inner
