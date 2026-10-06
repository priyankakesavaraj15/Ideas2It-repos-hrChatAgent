"""The MCP server works end to end over stdio, exactly as the agent launches it."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent import SERVER_SCRIPT, server_params

CALLS = {
    "search_hr_policies": {"query": "sick leave"},
    "get_employee_context": {"query": "manager"},
    "get_leave_balance": {},
    "calculate_leave": {"leave_type": "annual", "start_date": "2026-10-16", "end_date": "2026-10-21"},
}


def run_session(params, calls):
    async def run():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            results = {n: json.loads((await s.call_tool(n, a)).content[0].text) for n, a in calls.items()}
            return tools, results
    return asyncio.run(run())


def test_bound_session_lists_and_runs_all_tools():
    tools, results = run_session(server_params("E001", "2026-10-05"), CALLS)
    assert set(tools) == set(CALLS)
    for name, payload in results.items():
        assert "error" not in payload, (name, payload)
    assert results["get_leave_balance"]["employee"]["id"] == "E001"
    assert results["calculate_leave"]["working_days"] == 3
    assert results["calculate_leave"]["request"]["today"] == "2026-10-05"


def test_tool_schemas_expose_no_employee_id():
    tools, _ = run_session(server_params("E001", "2026-10-05"), {})
    for t in tools.values():
        assert "employee_id" not in t.inputSchema.get("properties", {}), t.name


def test_unbound_session_is_denied_employee_data():
    params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])  # no identity bound
    _, results = run_session(params, CALLS)
    assert "results" in results["search_hr_policies"]  # company-wide policy is still available
    for name in ("get_employee_context", "get_leave_balance", "calculate_leave"):
        assert "Access denied" in results[name]["error"], name


def test_extra_employee_id_argument_cannot_switch_identity():
    _, results = run_session(server_params("E001", "2026-10-05"), {"get_leave_balance": {"employee_id": "E004"}})
    assert results["get_leave_balance"]["employee"]["id"] == "E001"
