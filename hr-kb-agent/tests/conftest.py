"""Shared test setup. Tests run offline: no LLM / network calls."""
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TODAY = "2026-10-05"  # fixed "today" used by date-sensitive tests

# mcp_server opens hr.db / chroma at import time, so make sure demo data exists first.
if not (ROOT / "hr.db").exists() or not (ROOT / "chroma").exists():
    subprocess.run([sys.executable, str(ROOT / "seed.py")], check=True, cwd=ROOT)


@pytest.fixture(scope="session")
def server():
    import mcp_server
    return mcp_server


@pytest.fixture
def as_employee(monkeypatch):
    """Bind the MCP server's session identity (and today's date) the way agent.server_params() does."""
    def bind(emp_id: str | None, today: str = TODAY):
        if emp_id is None:
            monkeypatch.delenv("HR_SESSION_EMPLOYEE_ID", raising=False)
        else:
            monkeypatch.setenv("HR_SESSION_EMPLOYEE_ID", emp_id)
        monkeypatch.setenv("HR_TODAY", today)
    return bind


@pytest.fixture
def scratch_db(server, monkeypatch, tmp_path):
    """A private copy of hr.db the test may modify. Returns an open connection to it."""
    path = tmp_path / "hr.db"
    shutil.copy(ROOT / "hr.db", path)
    monkeypatch.setattr(server, "DB_PATH", path)
    con = sqlite3.connect(path)
    yield con
    con.close()
