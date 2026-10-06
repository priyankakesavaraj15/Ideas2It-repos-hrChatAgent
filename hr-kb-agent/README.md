# HR Knowledge Base – AI Chat Agent

A chat assistant that answers HR questions **only from company policy documents and HR records**:
policies (RAG over a vector DB), the signed-in employee's own leave balance/profile (SQLite + vector DB),
and auditable leave calculations. Built with LangGraph, an MCP tool server, Chroma, SQLite and Streamlit.

> ⚠️ **Demo authentication only.** Picking an employee in the sidebar *simulates* sign-in. There is no
> password, session token or SSO. See [ARCHITECTURE.md → Trust boundary](ARCHITECTURE.md#trust-boundary).

## Setup (tested: Python 3.13.7, Windows 11, PowerShell)
```powershell
cd hr-kb-agent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # pinned direct deps; or requirements.lock for the exact tested set
copy .env.example .env          # then put your free key from https://console.groq.com in .env
.venv\Scripts\python seed.py    # builds hr.db + chroma/ (first run downloads an ~80 MB embedding model)
.venv\Scripts\streamlit run app.py
```
Opens at http://localhost:8501. `.env` is git-ignored; never commit it.

| `.env` setting | Purpose |
|---|---|
| `GROQ_API_KEY` | required, free Groq key |
| `GROQ_MODEL` | default `openai/gpt-oss-120b` (Groq retires models; pick any tool-calling model your key can use) |
| `HR_TODAY` | recommended for demos: `2026-10-05`. Pins "today" so notice-period results don't change day to day |
| `HR_LOG_SENSITIVE` | leave unset. `1` writes questions, answers and HR records to `logs/app.log` (local debugging only) |

## 3-minute demo flow
Use the **Try these** buttons in the sidebar, with `HR_TODAY=2026-10-05`.

1. **Asha Rao (E001)**, *What's my leave balance?* → balance table from SQLite in **How I answered**.
2. *Can I take annual leave from 2026-10-16 to 2026-10-21?* → **Leave calculation audit**: 6 calendar days,
   3 working days (weekend + Dussehra excluded), every rule with its source,
   **Eligible, subject to approval** by the reporting manager. Balance 6 → 3.
3. *What if it was casual leave instead?* → follow-up reuses the dates; casual rules apply, plain **Eligible**.
4. *How many sick days do I get per year?* → answer cites **[📄 Leave Entitlement]**, checked against retrieved evidence.
   Then *What about casual?* (contextual follow-up).
5. *I want to take some leave next month* → asks a clarifying question; nothing is calculated on guessed dates.
6. *What is Karan Singh's leave balance?* → "I can only access your own HR records." (0 tool calls).
7. *What is the CEO's salary?* → "I don't have that information in the HR knowledge base."
8. Switch to **Karan Singh (E004)**, *Can I take annual leave on 2026-11-10 and 2026-11-11?* → **Not eligible** (probation).
   Optional: **Priya Nair (E003)**, *annual leave 2026-11-16 to 2026-11-18* → 1 day Leave Without Pay, needs manager + HR approval.

### If something goes wrong during the demo
| Symptom | Fix |
|---|---|
| Red box "Demo data not found" / "GROQ_API_KEY is not set" | Run `seed.py` / create `.env`, then refresh |
| "Groq free-tier rate limit reached" or slow answers | Wait ~1 min. The client already retries. Or set `GROQ_MODEL=openai/gpt-oss-20b` (separate limit) and restart |
| "Model ... is not available to your key" | Groq retired it. Pick a current tool-calling model in `GROQ_MODEL` |
| "Can't reach the Groq API" | Network/proxy issue. The tools still work offline: `python -m pytest` shows every rule passing. There is no offline LLM fallback |
| Anything else | `logs/app.log` (tool names, status, timings; no personal data by default) |

Do a warm-up question before presenting (first call starts the MCP server and loads the embedding model).

## Tests (offline, no API key, ~40 s)
```powershell
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```
| File | Covers |
|---|---|
| `tests/test_tools.py` | Each MCP tool called directly. Identity binding (tools refuse with no bound employee, no `employee_id` parameter). Policy retrieval per section. Employee-context isolation **and** relevance. Every `calculate_leave` rule and outcome: probation/notice/casual-cap boundaries, manager approval, LWP as approval (not rejection), overlap with approved/pending/rejected/adjacent requests, year crossing, outside records year, missing balance row, invalid and past dates. Policy text ↔ code constants. Seeded data consistency |
| `tests/test_mcp_protocol.py` | Real MCP server over stdio: tool schemas expose no `employee_id`. An unbound session is denied employee data. Extra arguments can't switch identity |
| `tests/test_agent.py` | Agent with a scripted fake LLM: identity/date binding, missing-input guard, tool error handling, trace, citation validation (incl. follow-ups and full-width brackets), logs contain no personal data unless opted in, per-thread memory |
| `tests/test_eval_checker.py` | The eval checker and `cases.json` schema |

## Evaluation (live LLM: needs `GROQ_API_KEY` and network)
```powershell
.venv\Scripts\python evals\run_evals.py
.venv\Scripts\python evals\run_evals.py --only calc-probation boundary-other-by-name
```
25 cases in `evals/cases.json` across these categories: policy, employee data, leave calculation, follow-up, ambiguous,
tool error, unsupported and access boundary. Every case also checks automatically that all citations match retrieved evidence
and that there were no unexpected tool errors. Leave-calculation cases check the structured result (outcome, failed and
approval rules, LWP days, excluded days), not just the wording. Each failure is labelled with the check that
failed (`[calc]`, `[citations]`, `[tools]`, `[answer]`, `[privacy]`, `[tool-errors]`).

`evals/results.md` records the **most recent run only**. LLM output can vary, so re-run after any change.
Run history: 19/19 before this revision. 23/25 on the first run of the expanded set (it caught full-width citation
brackets and a missing failure citation, both fixed). Then 25/25 with `openai/gpt-oss-120b` on 2026-10-05.

## Files
| File | What it does |
|---|---|
| `app.py` | Streamlit UI: mock sign-in, chat, example buttons, **How I answered** + **Leave calculation audit** panels, friendly API errors |
| `agent.py` | LangGraph ReAct agent (Groq, temperature 0), session-bound MCP server, trace, citation validation, privacy-safe logging |
| `mcp_server.py` | MCP server (stdio), 4 JSON tools: policy search, employee context, balance, leave calculation |
| `seed.py` | Sample employees/leave/holidays → SQLite; policies + profiles chunked → Chroma |
| `data/policies.md` | Sample HR policy document (one chunk per `##` section) |
| `evals/` | `cases.json`, `run_evals.py` |
| `tests/` | Offline pytest suite |
| `requirements.txt` / `requirements.lock` | Pinned direct deps / full tested environment |

Reset demo data any time: `.venv\Scripts\python seed.py`.
