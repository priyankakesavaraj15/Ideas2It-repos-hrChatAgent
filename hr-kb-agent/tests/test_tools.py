"""Direct tests of every MCP tool function and of the seeded demo data (no LLM)."""
import json
import re
import sqlite3
from datetime import date

import pytest

import seed
from conftest import ROOT


@pytest.fixture
def calc(server, as_employee):
    def run(emp, leave_type, start, end):
        as_employee(emp)
        return json.loads(server.calculate_leave(leave_type, start, end))
    return run


def rule(res, policy_id, status=None):
    found = [r for r in res["rules"] if r["policy_id"] == policy_id and (status is None or r["status"] == status)]
    assert found, f"no {policy_id} rule with status {status}: {res['rules']}"
    return found[0]


def statuses(res, status):
    return {r["policy_id"] for r in res["rules"] if r["status"] == status}


# ---------- policy text <-> code alignment ----------

def policy_text():
    return (ROOT / "data" / "policies.md").read_text(encoding="utf-8")


def test_policy_constants_match_policy_text(server):
    text = policy_text()
    assert f"first {server.PROBATION_MONTHS} months" in text
    assert f"exceed {server.CASUAL_MAX_CONSECUTIVE} consecutive" in text
    assert f"at least {server.ANNUAL_NOTICE_DAYS} calendar days" in text
    assert f"longer than {server.ANNUAL_NOTICE_THRESHOLD} working days" in text
    assert f"more than {server.SICK_CERT_THRESHOLD} consecutive" in text
    assert "approved by the reporting manager" in text
    assert "subject to manager and HR approval" in text
    for t, n in seed.ENTITLED.items():
        assert f"{t.capitalize()} Leave: {n} working days" in text


def test_every_rule_cites_a_real_policy_section_or_records(calc, server):
    sections = {server.section_id(s) for s in re.findall(r"^## (.+)$", policy_text(), flags=re.M)}
    allowed = sections | {server.section_id(server.RECORDS_CHECK)}
    for args in [("E001", "annual", "2026-10-07", "2026-10-09"), ("E003", "casual", "2026-11-02", "2026-11-06"),
                 ("E001", "sick", "2026-09-01", "2026-09-03"), ("E002", "annual", "2026-10-22", "2026-10-23")]:
        for r in calc(*args)["rules"]:
            assert r["policy_id"] in allowed, r


# ---------- seeded data consistency ----------

@pytest.mark.parametrize("emp_id", list(seed.USED))
def test_used_days_match_approved_requests(emp_id):
    con = sqlite3.connect(ROOT / "hr.db")
    for leave_type, used in seed.USED[emp_id].items():
        total = con.execute("SELECT COALESCE(SUM(days),0) FROM leave_requests WHERE emp_id=? AND type=? "
                            "AND status='approved'", (emp_id, leave_type)).fetchone()[0]
        assert total == used, f"{emp_id} {leave_type}: requests {total} != used {used}"


@pytest.mark.parametrize("req", seed.LEAVE_REQUESTS, ids=lambda r: f"{r[0]}-{r[1]}-{r[2]}")
def test_recorded_request_days_match_calculator(calc, req):
    emp, leave_type, start, end, days, _ = req
    res = calc(emp, leave_type, start, end)
    assert res["working_days"] == days
    assert "hr-leave-records" in statuses(res, "fail")  # the request overlaps itself


# ---------- identity boundary ----------

@pytest.mark.parametrize("tool,args", [
    ("get_employee_context", ("manager",)),
    ("get_leave_balance", ()),
    ("calculate_leave", ("annual", "2026-11-16", "2026-11-16")),
])
def test_employee_tools_refuse_without_bound_identity(server, as_employee, tool, args):
    as_employee(None)
    assert "Access denied" in json.loads(getattr(server, tool)(*args))["error"]


def test_employee_tools_take_no_employee_id_argument(server):
    import inspect
    for tool in (server.get_employee_context, server.get_leave_balance, server.calculate_leave):
        assert "employee_id" not in inspect.signature(tool).parameters


def test_policy_search_works_without_identity(server, as_employee):
    as_employee(None)
    assert json.loads(server.search_hr_policies("sick leave"))["results"]


# ---------- search_hr_policies ----------

@pytest.mark.parametrize("query,expected_section", [
    ("how many sick days do I get", "Leave Entitlement"),
    ("carry forward unused annual leave", "Annual Leave Rules"),
    ("medical certificate for sick leave", "Sick Leave Rules"),
    ("new joiner probation annual leave", "Probation Period"),
    ("work from home days per week", "Work From Home"),
    ("harassment reporting", "Code of Conduct"),
])
def test_policy_search_retrieves_right_section(server, query, expected_section):
    res = json.loads(server.search_hr_policies(query))["results"]
    assert len(res) == 3
    assert expected_section in [r["section"] for r in res]
    for r in res:
        assert r["text"].startswith(r["section"]) and isinstance(r["distance"], float)
        assert r["id"] == server.section_id(r["section"])


# ---------- get_employee_context ----------

@pytest.mark.parametrize("emp_id", list(seed.PROFILES))
def test_employee_context_only_returns_own_chunks(server, as_employee, emp_id):
    as_employee(emp_id)
    res = json.loads(server.get_employee_context("who is my manager and what leave did I take"))
    assert res["employee_id"] == emp_id and res["results"]
    for r in res["results"]:
        assert r["text"] in seed.PROFILES[emp_id]


@pytest.mark.parametrize("emp_id,query,expected", [
    ("E004", "who is my manager", "reporting to Vikram Shah"),
    ("E004", "what are my skills", "frontend development in React"),
    ("E002", "notes about my leave", "medical certificate submitted"),
    ("E001", "which team do I lead", "leads the payments platform team"),
    ("E005", "do I work from home", "Works from home on Fridays"),
])
def test_employee_context_top_result_is_relevant(server, as_employee, emp_id, query, expected):
    as_employee(emp_id)
    top = json.loads(server.get_employee_context(query))["results"][0]["text"]
    assert expected in top, top


def test_employee_context_unknown_employee_is_empty(server, as_employee):
    as_employee("E999")
    assert json.loads(server.get_employee_context("manager"))["results"] == []


# ---------- get_leave_balance ----------

def test_leave_balance_values(server, as_employee):
    as_employee("E001")
    res = json.loads(server.get_leave_balance())
    assert res["employee"]["name"] == "Asha Rao" and res["year"] == 2026
    bal = {b["leave_type"]: b for b in res["balances"]}
    assert (bal["annual"]["remaining"], bal["casual"]["remaining"], bal["sick"]["remaining"]) == (6, 3, 8)
    assert len(res["requests"]) == 5


def test_leave_balance_includes_pending_but_does_not_deduct(server, as_employee):
    as_employee("E002")
    res = json.loads(server.get_leave_balance())
    assert {"type": "casual", "start": "2026-10-23", "end": "2026-10-23", "days": 1, "status": "pending"} in res["requests"]
    assert next(b for b in res["balances"] if b["leave_type"] == "casual")["remaining"] == 6


def test_leave_balance_unknown_employee(server, as_employee):
    as_employee("E999")
    assert "error" in json.loads(server.get_leave_balance())


# ---------- calculate_leave: counting ----------

def test_calc_excludes_weekend_and_holiday(calc):
    res = calc("E001", "annual", "2026-10-16", "2026-10-21")
    assert res["calendar_days"] == 6 and res["working_days"] == 3
    assert res["excluded_days"] == [
        {"date": "2026-10-17", "reason": "weekend"},
        {"date": "2026-10-18", "reason": "weekend"},
        {"date": "2026-10-20", "reason": "holiday: Dussehra"},
    ]
    assert res["balance"] == {"remaining_before": 6, "paid_days": 3, "remaining_after": 3, "lwp_days": 0}


@pytest.mark.parametrize("start,end", [("2026-10-17", "2026-10-18"), ("2026-10-02", "2026-10-02")])
def test_calc_no_working_days(calc, start, end):
    res = calc("E001", "casual", start, end)
    assert res["working_days"] == 0 and res["outcome"] == "eligible"


def test_calc_normalises_leave_type(calc):
    assert calc("E001", "  Annual ", "2026-11-16", "2026-11-16")["request"]["leave_type"] == "annual"


# ---------- calculate_leave: outcomes (blocking rules vs approvals) ----------

def test_calc_annual_always_needs_manager_approval(calc):
    res = calc("E001", "annual", "2026-10-16", "2026-10-21")
    assert res["outcome"] == "eligible_subject_to_approval" and res["eligible"]
    assert "Vikram Shah" in rule(res, "annual-leave-rules", "approval")["detail"]
    assert res["reasons"] == [] and len(res["approvals_required"]) == 1


def test_calc_casual_within_rules_is_plain_eligible(calc):
    res = calc("E001", "casual", "2026-11-16", "2026-11-18")
    assert res["outcome"] == "eligible" and res["approvals_required"] == []
    assert rule(res, "casual-leave-rules")["status"] == "pass"


def test_calc_probation_blocks_annual(calc):
    res = calc("E004", "annual", "2026-11-10", "2026-11-11")
    assert res["outcome"] == "not_eligible"
    assert "2026-12-31" in rule(res, "probation-period", "fail")["detail"]


def test_probation_end_boundary(server):
    assert server.probation_end(date(2026, 7, 1)) == date(2027, 1, 1)   # last probation day 2026-12-31
    assert server.probation_end(date(2026, 8, 31)) == date(2027, 2, 28)  # month-end clamps


def test_calc_probation_last_day_still_blocked(calc):
    assert rule(calc("E004", "annual", "2026-12-31", "2026-12-31"), "probation-period")["status"] == "fail"


def test_calc_probation_does_not_block_sick_or_casual(calc):
    for t in ("sick", "casual"):
        res = calc("E004", t, "2026-11-10", "2026-11-10")
        assert res["eligible"] and "probation-period" not in {r["policy_id"] for r in res["rules"]}


@pytest.mark.parametrize("start,end,status", [
    ("2026-10-07", "2026-10-09", "fail"),   # 3 days, 2 days notice
    ("2026-10-12", "2026-10-14", "pass"),   # exactly 7 days notice
    ("2026-10-07", "2026-10-08", "pass"),   # 2 days: notice not required
])
def test_calc_annual_notice(calc, start, end, status):
    res = calc("E001", "annual", start, end)
    notice = next(r for r in res["rules"] if "notice" in r["rule"])
    assert notice["status"] == status


def test_calc_lwp_needs_manager_and_hr_approval_not_rejection(calc):
    res = calc("E003", "annual", "2026-11-16", "2026-11-18")  # 3 days, 2 annual left
    assert res["eligible"] and res["outcome"] == "eligible_subject_to_approval"
    assert res["balance"] == {"remaining_before": 2, "paid_days": 2, "remaining_after": 0, "lwp_days": 1}
    assert "manager and HR approval" in rule(res, "leave-without-balance", "approval")["detail"]
    assert statuses(res, "approval") == {"annual-leave-rules", "leave-without-balance"}


def test_calc_lwp_on_casual_still_blocked_by_cap(calc):
    res = calc("E003", "casual", "2026-11-02", "2026-11-06")  # 5 days, cap 3, 1 casual left
    assert res["outcome"] == "not_eligible"
    assert statuses(res, "fail") == {"casual-leave-rules"}
    assert "leave-without-balance" in statuses(res, "approval") and res["balance"]["lwp_days"] == 4
    assert len(res["reasons"]) == 1 and res["reasons"][0].endswith("[casual-leave-rules]")


def test_calc_insufficient_balance_counts_holidays_correctly(calc):
    res = calc("E001", "annual", "2026-11-02", "2026-11-13")  # 9 working days (Diwali 11-09), 6 left
    assert res["working_days"] == 9 and res["balance"]["lwp_days"] == 3


def test_calc_sick_certificate_is_note_not_blocker(calc):
    res = calc("E001", "sick", "2026-11-02", "2026-11-04")
    assert res["outcome"] == "eligible" and rule(res, "sick-leave-rules")["status"] == "info"
    assert any("Medical certificate" in n for n in res["notes"])


# ---------- calculate_leave: records checks and edge cases ----------

def test_calc_overlap_with_pending_request_blocks(calc):
    res = calc("E002", "annual", "2026-10-22", "2026-10-23")  # pending casual on 2026-10-23
    assert res["outcome"] == "not_eligible"
    assert "casual 2026-10-23 to 2026-10-23 (pending)" in rule(res, "hr-leave-records", "fail")["detail"]


def test_calc_overlap_with_approved_request_blocks(calc):
    res = calc("E001", "casual", "2026-04-24", "2026-04-27")  # approved annual ends 2026-04-24
    assert "annual 2026-04-13 to 2026-04-24 (approved)" in rule(res, "hr-leave-records", "fail")["detail"]


def test_calc_adjacent_dates_do_not_overlap(calc):
    res = calc("E001", "casual", "2026-04-27", "2026-04-28")
    overlap = next(r for r in res["rules"] if "overlap" in r["rule"])
    assert overlap["status"] == "pass" and res["eligible"]


def test_calc_rejected_requests_do_not_count_as_overlap(calc, scratch_db):
    scratch_db.execute("INSERT INTO leave_requests (emp_id,type,start,end,days,status) "
                       "VALUES ('E001','casual','2026-11-20','2026-11-20',1,'rejected')")
    scratch_db.commit()
    assert rule(calc("E001", "casual", "2026-11-20", "2026-11-20"), "hr-leave-records")["status"] == "pass"


def test_calc_past_dates_are_flagged_as_note(calc):
    res = calc("E001", "sick", "2026-09-01", "2026-09-01")
    assert res["outcome"] == "eligible"
    assert any("before today" in n for n in res["notes"])


def test_calc_crossing_year_boundary_is_rejected(calc):
    assert "Split a request" in calc("E001", "annual", "2026-12-28", "2027-01-04")["error"]


def test_calc_outside_records_year_is_rejected(calc):
    assert "cover 2026 only" in calc("E004", "annual", "2027-01-04", "2027-01-05")["error"]


def test_calc_last_day_of_records_year_ok(calc):
    res = calc("E001", "casual", "2026-12-31", "2026-12-31")
    assert res["working_days"] == 1 and res["outcome"] == "eligible"


def test_calc_missing_balance_row_is_an_error_not_lwp(calc, scratch_db):
    scratch_db.execute("DELETE FROM leave_balances WHERE emp_id='E005' AND leave_type='sick'")
    scratch_db.commit()
    assert "No sick leave balance on record" in calc("E005", "sick", "2026-11-16", "2026-11-16")["error"]


@pytest.mark.parametrize("emp,leave_type,start,end,msg", [
    ("E001", "maternity", "2026-11-16", "2026-11-16", "Invalid leave_type"),
    ("E001", "annual", "16/11/2026", "2026-11-16", "Invalid date"),
    ("E001", "annual", "2026-02-30", "2026-03-02", "Invalid date"),
    ("E001", "annual", "2026-11-18", "2026-11-16", "before start_date"),
    ("E999", "annual", "2026-11-16", "2026-11-16", "not found"),
])
def test_calc_invalid_inputs_return_error(calc, emp, leave_type, start, end, msg):
    assert msg in calc(emp, leave_type, start, end)["error"]
