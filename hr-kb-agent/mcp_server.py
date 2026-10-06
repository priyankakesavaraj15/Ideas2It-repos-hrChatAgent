"""HR MCP server (stdio): policy RAG, employee-context RAG, leave balance and leave calculation.

Identity: the server is started once per chat turn by the host (agent.py) with HR_SESSION_EMPLOYEE_ID set to
the signed-in employee. Employee-data tools take NO employee_id argument; they only ever read the bound
employee's records, and refuse if no identity is bound. (See ARCHITECTURE.md "Trust boundary".)

Every tool returns a JSON string so the agent can reason over it and the UI can render it as evidence.
Evidence carries a stable id (e.g. "sick-leave-rules") that the answer cites as [sick-leave-rules].
"""
import calendar
import json
import os
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent
DB_PATH = ROOT / "hr.db"

# Rule constants - must match data/policies.md (tests/test_tools.py checks this)
PROBATION_MONTHS = 6
CASUAL_MAX_CONSECUTIVE = 3
ANNUAL_NOTICE_DAYS = 7
ANNUAL_NOTICE_THRESHOLD = 2  # annual leave longer than this needs notice
SICK_CERT_THRESHOLD = 2  # sick leave longer than this needs a medical certificate
LEAVE_TYPES = ("annual", "sick", "casual")
RECORDS_YEAR = 2026  # balances and holidays in hr.db cover this calendar year only

# Checks that come from HR records rather than a policy section
RECORDS_CHECK = "HR leave records"

mcp = FastMCP("hr-tools")
_chroma = chromadb.PersistentClient(path=str(ROOT / "chroma"))
_policies = _chroma.get_collection("hr_policies")
_context = _chroma.get_collection("employee_context")


def section_id(name: str) -> str:
    """Stable citation id for a policy section, e.g. 'Sick Leave Rules' -> 'sick-leave-rules'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _session_employee() -> str | None:
    return os.getenv("HR_SESSION_EMPLOYEE_ID") or None


def _today() -> date:
    return date.fromisoformat(os.environ["HR_TODAY"]) if os.getenv("HR_TODAY") else date.today()


def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


NO_IDENTITY = _json({"error": "Access denied: no signed-in employee is bound to this MCP session."})


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, min(d.day, calendar.monthrange(d.year + y, m + 1)[1]))


def probation_end(join_date: date) -> date:
    """First day after probation."""
    return _add_months(join_date, PROBATION_MONTHS)


@mcp.tool()
def search_hr_policies(query: str) -> str:
    """Search the HR policy knowledge base (company-wide, not employee-specific).
    Returns the most relevant policy sections, each with a citation id."""
    res = _policies.query(query_texts=[query], n_results=3)
    return _json({"results": [
        {"id": section_id(m["section"]), "section": m["section"], "text": d, "distance": round(dist, 3)}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]})


@mcp.tool()
def get_employee_context(query: str) -> str:
    """Search the signed-in employee's own profile notes (role, manager, skills, leave notes)."""
    emp_id = _session_employee()
    if not emp_id:
        return NO_IDENTITY
    res = _context.query(query_texts=[query], n_results=3, where={"employee_id": emp_id})
    return _json({"employee_id": emp_id, "results": [
        {"text": d, "distance": round(dist, 3)} for d, dist in zip(res["documents"][0], res["distances"][0])
    ]})


@mcp.tool()
def get_leave_balance() -> str:
    """The signed-in employee's leave balance per leave type plus their leave requests this year."""
    emp_id = _session_employee()
    if not emp_id:
        return NO_IDENTITY
    with _db() as con:
        emp = con.execute("SELECT * FROM employees WHERE id=?", (emp_id,)).fetchone()
        if not emp:
            return _json({"error": f"Employee {emp_id} not found."})
        bals = con.execute("SELECT * FROM leave_balances WHERE emp_id=? ORDER BY leave_type", (emp_id,)).fetchall()
        reqs = con.execute("SELECT * FROM leave_requests WHERE emp_id=? ORDER BY start", (emp_id,)).fetchall()
    return _json({
        "employee": {"id": emp_id, "name": emp["name"], "join_date": emp["join_date"]},
        "year": RECORDS_YEAR,
        "balances": [
            {"leave_type": b["leave_type"], "entitled": b["entitled"], "used": b["used"],
             "remaining": b["entitled"] - b["used"]} for b in bals
        ],
        "requests": [
            {"type": r["type"], "start": r["start"], "end": r["end"], "days": r["days"], "status": r["status"]}
            for r in reqs
        ],
    })


def _rule(rule: str, section: str, status: str, detail: str) -> dict:
    """status: pass | fail (blocks the request) | approval (allowed, needs sign-off) | info."""
    return {"rule": rule, "policy_section": section, "policy_id": section_id(section),
            "status": status, "detail": detail}


@mcp.tool()
def calculate_leave(leave_type: str, start_date: str, end_date: str) -> str:
    """Calculate working days for the signed-in employee's leave request (YYYY-MM-DD dates) and check it
    against company policy. leave_type is one of: annual, sick, casual.
    outcome: not_eligible (a rule is broken) | eligible_subject_to_approval | eligible."""
    emp_id = _session_employee()
    if not emp_id:
        return NO_IDENTITY
    leave_type = leave_type.lower().strip()
    if leave_type not in LEAVE_TYPES:
        return _json({"error": f"Invalid leave_type '{leave_type}'. Use one of: {', '.join(LEAVE_TYPES)}."})
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return _json({"error": "Invalid date. Use a real calendar date in YYYY-MM-DD format."})
    if end < start:
        return _json({"error": "end_date is before start_date."})
    if start.year != end.year:
        return _json({"error": "Leave is allocated per calendar year (Leave Entitlement). "
                               "Split a request that crosses 31 December into one request per year."})
    if start.year != RECORDS_YEAR:
        return _json({"error": f"Leave balances and holidays on record cover {RECORDS_YEAR} only, "
                               f"so a {start.year} request cannot be checked."})
    today = _today()

    with _db() as con:
        emp = con.execute("SELECT * FROM employees WHERE id=?", (emp_id,)).fetchone()
        if not emp:
            return _json({"error": f"Employee {emp_id} not found."})
        bal = con.execute(
            "SELECT entitled - used AS remaining FROM leave_balances WHERE emp_id=? AND leave_type=?",
            (emp_id, leave_type),
        ).fetchone()
        if bal is None:
            return _json({"error": f"No {leave_type} leave balance on record for {RECORDS_YEAR}. Contact HR."})
        holidays = {
            r["date"]: r["name"]
            for r in con.execute("SELECT * FROM holidays WHERE date BETWEEN ? AND ?", (start_date, end_date))
        }
        overlaps = con.execute(
            "SELECT * FROM leave_requests WHERE emp_id=? AND status IN ('approved','pending') "
            "AND start <= ? AND end >= ? ORDER BY start", (emp_id, end_date, start_date),
        ).fetchall()

    working, excluded = 0, []
    d = start
    while d <= end:
        if d.weekday() >= 5:
            excluded.append({"date": d.isoformat(), "reason": "weekend"})
        elif d.isoformat() in holidays:
            excluded.append({"date": d.isoformat(), "reason": f"holiday: {holidays[d.isoformat()]}"})
        else:
            working += 1
        d += timedelta(days=1)

    remaining = bal["remaining"]
    notice_days = (start - today).days
    prob_end = probation_end(date.fromisoformat(emp["join_date"]))
    calendar_days = (end - start).days + 1

    rules = [_rule("Count working days only (weekends and company holidays excluded)", "Leave Entitlement",
                   "info", f"{working} working day(s) out of {calendar_days} calendar day(s).")]
    if start < today:
        rules.append(_rule("Request dates relative to today", RECORDS_CHECK, "info",
                           f"Start date {start} is before today ({today}): this checks a past period."))
    rules.append(_rule("Must not overlap existing approved or pending leave", RECORDS_CHECK,
                       "fail" if overlaps else "pass",
                       "Overlaps: " + ", ".join(f"{o['type']} {o['start']} to {o['end']} ({o['status']})"
                                                for o in overlaps) if overlaps else "No overlapping leave on record."))
    if leave_type == "annual":
        on_probation = start < prob_end
        rules.append(_rule(f"No annual leave during the first {PROBATION_MONTHS} months (probation)", "Probation Period",
                           "fail" if on_probation else "pass",
                           f"Joined {emp['join_date']}; probation "
                           f"{'runs' if on_probation else 'ended'} until {prob_end - timedelta(days=1)}."))
        notice_rule = f"Annual leave over {ANNUAL_NOTICE_THRESHOLD} days needs {ANNUAL_NOTICE_DAYS} days' notice"
        if working > ANNUAL_NOTICE_THRESHOLD:
            rules.append(_rule(notice_rule, "Annual Leave Rules", "fail" if notice_days < ANNUAL_NOTICE_DAYS else "pass",
                               f"{notice_days} day(s) notice given (today {today}, start {start})."))
        else:
            rules.append(_rule(notice_rule, "Annual Leave Rules", "pass",
                               f"Not required: request is {working} working day(s)."))
        rules.append(_rule("Annual leave must be approved by the reporting manager", "Annual Leave Rules", "approval",
                           f"Needs approval from {emp['manager']} (reporting manager)."))
    if leave_type == "casual":
        rules.append(_rule(f"Casual leave max {CASUAL_MAX_CONSECUTIVE} consecutive working days", "Casual Leave Rules",
                           "fail" if working > CASUAL_MAX_CONSECUTIVE else "pass",
                           f"Request is {working} working day(s)."))
    if leave_type == "sick":
        needs_cert = working > SICK_CERT_THRESHOLD
        rules.append(_rule(f"Medical certificate for sick leave over {SICK_CERT_THRESHOLD} consecutive working days",
                           "Sick Leave Rules", "info" if needs_cert else "pass",
                           "Medical certificate required." if needs_cert else "Not required."))
    lwp = max(working - remaining, 0)
    rules.append(_rule("Days beyond the remaining balance are Leave Without Pay, subject to manager and HR approval",
                       "Leave Without Balance", "approval" if lwp else "pass",
                       f"Remaining {leave_type} balance {remaining}, requested {working}"
                       + (f"; {lwp} day(s) would be Leave Without Pay and need manager and HR approval." if lwp else ".")))

    eligible = not any(r["status"] == "fail" for r in rules)
    approvals = [r for r in rules if r["status"] == "approval"]
    outcome = "not_eligible" if not eligible else "eligible_subject_to_approval" if approvals else "eligible"
    return _json({
        "request": {"employee_id": emp_id, "leave_type": leave_type, "start_date": start_date,
                    "end_date": end_date, "today": today.isoformat()},
        "calendar_days": calendar_days,
        "working_days": working,
        "excluded_days": excluded,
        "balance": {"remaining_before": remaining, "paid_days": working - lwp,
                    "remaining_after": max(remaining - working, 0), "lwp_days": lwp},
        "rules": rules,
        "eligible": eligible,
        "outcome": outcome,
        "reasons": [f"{r['detail']} [{r['policy_id']}]" for r in rules if r["status"] == "fail"],
        "approvals_required": [f"{r['detail']} [{r['policy_id']}]" for r in approvals],
        "notes": [f"{r['detail']} [{r['policy_id']}]" for r in rules[1:] if r["status"] == "info"],
    })


if __name__ == "__main__":
    mcp.run()
