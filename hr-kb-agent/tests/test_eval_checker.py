"""The eval runner's checker itself must be right, or eval results mean nothing."""
import json

from conftest import ROOT
from evals.run_evals import check, has

CALC = {"tool": "calculate_leave", "result": {
    "working_days": 3, "eligible": True, "outcome": "eligible_subject_to_approval",
    "request": {"leave_type": "annual"}, "balance": {"lwp_days": 0},
    "excluded_days": [{}, {}, {}],
    "rules": [{"policy_id": "annual-leave-rules", "status": "approval"},
              {"policy_id": "hr-leave-records", "status": "pass"}]}}
VALID = [{"id": "annual-leave-rules", "section": "Annual Leave Rules", "valid": True}]


def test_number_match_is_whole_number():
    assert has("you have 6 days", "6")
    assert not has("in 2026", "6")


def test_check_passes():
    case = {"expect": {"tools_called": ["calculate_leave"], "answer_contains_all": ["3"],
                       "cites_any": ["annual-leave-rules"], "trace_not_contains": ["E004"],
                       "calc": {"working_days": 3, "outcome": "eligible_subject_to_approval", "excluded_days": 3,
                                "failed_rules": [], "approval_rules": ["annual-leave-rules"]}}}
    assert check(case, "**3** working days [annual-leave-rules]", [CALC], VALID) == []


def test_failures_name_the_check_that_failed():
    case = {"expect": {"tools_called": ["calculate_leave"], "calc": {"outcome": "not_eligible"},
                       "cites_any": ["probation-period"]}}
    fails = check(case, "x", [CALC], [{"id": "made-up", "section": None, "valid": False}])
    assert any(f.startswith("[calc] outcome = 'eligible_subject_to_approval', expected 'not_eligible'") for f in fails)
    assert any(f.startswith("[citations] cited [made-up]") for f in fails)
    assert any(f.startswith("[citations] expected a valid citation") for f in fails)


def test_tool_errors_fail_unless_allowed():
    err = {"tool": "calculate_leave", "result": {"error": "Leave balances and holidays on record cover 2026 only"}}
    case = {"expect": {"tool_error_contains": "cover 2026 only", "no_successful_calc": True}}
    assert any(f.startswith("[tool-errors] unexpected") for f in check(case, "x", [err]))
    assert check({**case, "allow_tool_errors": True}, "x", [err]) == []
    assert any(f.startswith("[calc] expected no successful") for f in check({**case, "allow_tool_errors": True},
                                                                          "x", [err, CALC]))


def test_privacy_and_forbidden_tool():
    case = {"expect": {"tools_not_called": ["calculate_leave"], "trace_not_contains": ["E004"]}}
    leak = {"tool": "calculate_leave", "result": {"employee_id": "E004"}}
    fails = check(case, "x", [leak])
    assert [f.split("]")[0] for f in fails] == ["[tools", "[privacy"]


def test_eval_cases_are_well_formed():
    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    allowed = {"tools_called", "tools_called_any", "tools_not_called", "answer_contains_all",
               "answer_contains_any", "answer_not_contains", "trace_not_contains", "calc", "cites_any",
               "tool_error_contains", "no_successful_calc"}
    calc_keys = {"working_days", "eligible", "outcome", "leave_type", "lwp_days", "excluded_days",
                 "failed_rules", "approval_rules"}
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["turns"] and c["employee"].startswith("E") and set(c["expect"]) <= allowed, c["id"]
        assert set(c["expect"].get("calc", {})) <= calc_keys, c["id"]
    assert {c["category"] for c in cases} >= {"policy", "employee-data", "leave-calculation", "follow-up",
                                              "ambiguous", "tool-error", "unsupported", "access-boundary"}
