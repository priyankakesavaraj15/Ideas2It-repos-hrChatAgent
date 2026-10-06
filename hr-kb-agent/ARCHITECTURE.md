# Architecture

```
Streamlit (app.py)                LangGraph agent (agent.py)                 MCP server (mcp_server.py, stdio)
 mock sign-in → emp_id  ──ask()──▶ ReAct loop, Groq LLM, temperature 0      process started per turn with
 chat + evidence panels ◀────────  spawns server bound to emp_id + date ──▶ HR_SESSION_EMPLOYEE_ID, HR_TODAY
                         answer,   records trace, validates citations       search_hr_policies ──▶ Chroma: hr_policies
                         trace,    MemorySaver (thread = random chat id)    get_employee_context ─▶ Chroma: employee_context (filtered)
                         citations                                          get_leave_balance ───▶ SQLite
                                                                            calculate_leave ─────▶ SQLite
```

## Components and why
| Component | Choice | Reason |
|---|---|---|
| UI | Streamlit | One Python file; chat, expanders, metrics built in |
| Agent | LangGraph `create_react_agent` | Tool-calling loop + checkpointer for multi-turn memory with little code |
| LLM | Groq free tier, `openai/gpt-oss-120b`, temperature 0 | Free, fast, reliable tool calling. Configurable via `GROQ_MODEL` |
| Tools | MCP server (FastMCP, `mcp==1.30.0`), stdio | Tools reusable by any MCP client. `mcp` 2.x renamed FastMCP, hence the pin |
| Vector DB | Chroma (persistent, local), built-in `all-MiniLM-L6-v2` ONNX embeddings | Free, no server, no torch |
| Records DB | SQLite | Exact numbers for balances/calculations; zero setup |

Versions are pinned in `requirements.txt` (direct) and `requirements.lock` (full environment), tested on Python 3.13.7 / Windows 11.

## Retrieval (RAG)
- `data/policies.md` is chunked **one chunk per `##` section** (9 chunks), with `section` metadata.
- Employee profiles are chunked **per paragraph** (15 chunks), each tagged with `employee_id` metadata.
  Search is filtered by the session's employee.
- Chroma embeds chunks at insert time (`seed.py`) and the query at search time; top 3 by distance are returned.
- Exact numbers (balances, requests, holidays) are **not** in the vector DB. They come from SQLite.

## Tools (all return JSON)
| Tool | LLM-provided inputs | Bound by the session | Output |
|---|---|---|---|
| `search_hr_policies` | `query` | | `results[]`: `id`, `section`, `text`, `distance` |
| `get_employee_context` | `query` | employee | `results[]`: `text`, `distance` |
| `get_leave_balance` | | employee | `employee`, `year`, `balances[]`, `requests[]` |
| `calculate_leave` | `leave_type`, `start_date`, `end_date` | employee, today | see below |

`calculate_leave` returns `working_days`, `calendar_days`, `excluded_days[]`, `balance` (before, paid, after, `lwp_days`),
`rules[]` (`rule`, `policy_section`, `policy_id`, `status`, `detail`), `eligible`, `outcome`, `reasons[]`,
`approvals_required[]`, `notes[]`. Or it returns `error` for invalid input, a year crossing, a year outside the records,
or a missing balance row.

### Eligibility vs approval (policy semantics)
The tool reports **policy eligibility and required approvals separately. It never approves anything.**

| Rule | Source | Status if triggered |
|---|---|---|
| Working days only (weekends, holidays excluded) | Leave Entitlement | info |
| No annual leave in first 6 months | Probation Period | **fail** |
| Annual > 2 days needs 7 days' notice | Annual Leave Rules | **fail** |
| Annual leave must be approved by the reporting manager | Annual Leave Rules | **approval** (always, for annual) |
| Casual max 3 consecutive working days | Casual Leave Rules | **fail** |
| Medical certificate for sick > 2 days | Sick Leave Rules | info |
| Days beyond balance = Leave Without Pay, subject to manager and HR approval | Leave Without Balance | **approval** |
| Must not overlap approved/pending leave | HR leave records (system check, not policy text) | **fail** |
| Start date before today | HR leave records | info |

`outcome` = `not_eligible` if any rule fails, otherwise `eligible_subject_to_approval` if any approval is required,
otherwise `eligible`. Rule constants live in `mcp_server.py`. Tests check they match the wording in `policies.md`.
Records cover calendar year 2026 only. Requests that cross 31 December must be split per year (Leave Entitlement is per
calendar year). Requests for other years return an error rather than a guess.

## Trust boundary
**Demo (as built):**
1. The Streamlit dropdown sets `emp_id`. This is **mock authentication**: anyone with the URL can pick anyone.
2. `agent.py` starts a fresh MCP server process for each turn with `HR_SESSION_EMPLOYEE_ID=<emp_id>`.
3. The MCP server reads identity **only** from that process environment. Employee-data tools have **no
   `employee_id` parameter**, so neither the LLM nor any other MCP client can ask for another employee. A server
   started without an identity refuses all employee-data tools (policy search still works, as it is company-wide).

The app host is therefore trusted to set the right identity. The LLM is not trusted with identity at all.

**What production would need:** real sign-in (SSO/OIDC) handled server-side. The identity must come from a verified token
(e.g. MCP over HTTP with OAuth 2.1 bearer tokens validated by the MCP server, mapping the token subject to an employee),
not from a UI control or environment variable. It would also need DB-level row filtering by that identity,
authorization for HR/manager roles, TLS, and audit logging of data access.

## Grounding and citations
- Temperature 0. The system prompt forbids outside knowledge and defines exact replies for unsupported questions
  and other-employee requests.
- Every policy evidence item has a stable id (`sick-leave-rules`, from the section heading). The model must cite
  `[id]`. `agent.py` extracts the cited ids and checks each one against an **evidence index built from all tool outputs in
  the conversation** (so a follow-up may cite evidence retrieved in an earlier turn).
  - The UI renders valid citations as section names.
  - It flags unknown ids as **unverified** in the answer and the evidence panel.
  - It notes when policy evidence was retrieved but nothing was cited.
- `calculate_leave` runs only when leave type, start and end dates are all known. The wrapper also rejects empty or
  invalid inputs before the server is called. The model is told to ask one clarifying question instead.
- Limitation: citations prove the cited evidence was retrieved, not that every sentence is supported by it.

## Evidence panel ("How I answered")
It shows the recorded workflow: each tool, the inputs the model chose, the latency, ok/error status, the returned evidence,
and the validated citations. It is **not** the model's internal reasoning. When `calculate_leave` ran, the
**Leave calculation audit** shows dates, working vs calendar days, excluded days, balance before and after, every rule
with ✅/❌/🟡/ℹ️ and its source, and the outcome with reasons, required approvers and notes.

## Logging and privacy
- Default (`logs/app.log` + console) is **operational metadata only**: random chat id, tool name, ok/error status,
  duration, counts (tools, citations, answer length) and error type. No employee IDs, names, questions, answers or
  tool payloads. A test enforces this.
- `HR_LOG_SENSITIVE=1` (opt-in, local debugging only) adds questions, answers, tool arguments and results, and error
  tracebacks. The log file then contains personal HR data. Don't enable it with real data, delete the log
  afterwards, and never share it. The sidebar shows a red warning while it is on.
- `logs/` is git-ignored.

## Chat memory
`MemorySaver` keyed by a random `thread_id`, so follow-ups ("What about casual?") reuse earlier turns. It is new on employee
switch or **New chat**, and is in-memory only (lost on restart).

## Limitations
- Mock authentication. One MCP subprocess per turn (~1 s). Memory not persisted.
- Grounding is enforced by prompt + deterministic tools + citation validation, not guaranteed. The audit panel lets
  users verify the numbers.
- Single records year (2026), one holiday list, no half days, no leave application/approval workflow.
- Groq free tier has rate limits. There is no offline LLM fallback. Live evals vary run to run and need re-running after changes.
