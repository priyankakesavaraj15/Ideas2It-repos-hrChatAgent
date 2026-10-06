"""Create the demo SQLite DB (hr.db) and the Chroma vector store (chroma/). Run once."""
import re
import sqlite3
from pathlib import Path

import chromadb

ROOT = Path(__file__).parent
DB_PATH = ROOT / "hr.db"
CHROMA_PATH = ROOT / "chroma"

EMPLOYEES = [
    # id, name, dept, role, join_date, manager, location
    ("E001", "Asha Rao", "Engineering", "Senior Software Engineer", "2021-03-15", "Vikram Shah", "Bengaluru"),
    ("E002", "Rahul Mehta", "Sales", "Account Executive", "2023-08-01", "Neha Kapoor", "Mumbai"),
    ("E003", "Priya Nair", "HR", "HR Business Partner", "2019-11-04", "Anil Kumar", "Chennai"),
    ("E004", "Karan Singh", "Engineering", "Software Engineer", "2026-07-01", "Vikram Shah", "Bengaluru"),  # on probation
    ("E005", "Meera Iyer", "Finance", "Financial Analyst", "2022-01-10", "Suresh Pillai", "Pune"),
]

# emp_id -> {leave_type: used}; entitlements come from the policy (18/10/6)
ENTITLED = {"annual": 18, "sick": 10, "casual": 6}
USED = {
    "E001": {"annual": 12, "sick": 2, "casual": 3},
    "E002": {"annual": 5, "sick": 6, "casual": 0},
    "E003": {"annual": 16, "sick": 0, "casual": 5},
    "E004": {"annual": 0, "sick": 1, "casual": 0},
    "E005": {"annual": 8, "sick": 3, "casual": 2},
}

LEAVE_REQUESTS = [
    # emp_id, type, start, end, days, status
    ("E001", "annual", "2026-04-13", "2026-04-24", 10, "approved"),
    ("E001", "annual", "2026-08-17", "2026-08-18", 2, "approved"),
    ("E001", "sick", "2026-02-09", "2026-02-10", 2, "approved"),
    ("E001", "casual", "2026-06-05", "2026-06-05", 1, "approved"),
    ("E001", "casual", "2026-09-10", "2026-09-11", 2, "approved"),
    ("E002", "annual", "2026-05-04", "2026-05-08", 5, "approved"),
    ("E002", "sick", "2026-03-02", "2026-03-10", 6, "approved"),  # 2026-03-04 is Holi
    ("E002", "casual", "2026-10-23", "2026-10-23", 1, "pending"),
    ("E003", "annual", "2026-06-01", "2026-06-19", 15, "approved"),
    ("E003", "annual", "2026-09-04", "2026-09-04", 1, "approved"),
    ("E003", "casual", "2026-01-12", "2026-01-13", 2, "approved"),
    ("E003", "casual", "2026-03-16", "2026-03-18", 3, "approved"),
    ("E004", "sick", "2026-09-15", "2026-09-15", 1, "approved"),
    ("E005", "annual", "2026-07-06", "2026-07-15", 8, "approved"),
    ("E005", "sick", "2026-02-23", "2026-02-25", 3, "approved"),
    ("E005", "casual", "2026-08-21", "2026-08-24", 2, "approved"),
]

HOLIDAYS = [
    ("2026-01-01", "New Year's Day"),
    ("2026-01-26", "Republic Day"),
    ("2026-03-04", "Holi"),
    ("2026-05-01", "Labour Day"),
    ("2026-08-15", "Independence Day"),
    ("2026-10-02", "Gandhi Jayanti"),
    ("2026-10-20", "Dussehra"),
    ("2026-11-09", "Diwali"),
    ("2026-12-25", "Christmas"),
]

# Free-text employee context, chunked into the vector store (one chunk per paragraph)
PROFILES = {
    "E001": """Asha Rao is a Senior Software Engineer in the Engineering department, based in Bengaluru, reporting to Vikram Shah. She joined on 2021-03-15 and is a confirmed (non-probation) full-time employee.

Skills and work: backend development in Python and Go, leads the payments platform team, mentors two junior engineers.

Leave notes: took a 10-day annual leave in April 2026 for a family trip and 2 days in August 2026. Carried forward 0 days from 2025.""",
    "E002": """Rahul Mehta is an Account Executive in the Sales department, based in Mumbai, reporting to Neha Kapoor. He joined on 2023-08-01 and is a confirmed full-time employee.

Skills and work: manages enterprise accounts in the west region, quarterly sales target owner.

Leave notes: took 6 days of sick leave in March 2026 with a medical certificate submitted. Has a pending casual leave request for 2026-10-23.""",
    "E003": """Priya Nair is an HR Business Partner in the HR department, based in Chennai, reporting to Anil Kumar. She joined on 2019-11-04 and is a confirmed full-time employee.

Skills and work: employee relations, onboarding, policy rollout for the southern offices.

Leave notes: took a long annual leave in June 2026. Has used 16 of 18 annual leave days this year.""",
    "E004": """Karan Singh is a Software Engineer in the Engineering department, based in Bengaluru, reporting to Vikram Shah. He joined on 2026-07-01 and is currently on probation until 2026-12-31.

Skills and work: frontend development in React, part of the payments platform team, onboarding buddy is Asha Rao.

Leave notes: took 1 sick day in September 2026. Not eligible for annual leave until probation ends.""",
    "E005": """Meera Iyer is a Financial Analyst in the Finance department, based in Pune, reporting to Suresh Pillai. She joined on 2022-01-10 and is a confirmed full-time employee.

Skills and work: budgeting, monthly close, vendor payment reconciliation.

Leave notes: took 8 days of annual leave in July 2026. Works from home on Fridays with manager approval.""",
}


def seed_sqlite():
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE employees (id TEXT PRIMARY KEY, name TEXT, dept TEXT, role TEXT,
                                join_date TEXT, manager TEXT, location TEXT);
        CREATE TABLE leave_balances (emp_id TEXT, leave_type TEXT, entitled INTEGER, used INTEGER,
                                     PRIMARY KEY (emp_id, leave_type));
        CREATE TABLE leave_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, type TEXT,
                                     start TEXT, end TEXT, days INTEGER, status TEXT);
        CREATE TABLE holidays (date TEXT PRIMARY KEY, name TEXT);
    """)
    con.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?)", EMPLOYEES)
    con.executemany(
        "INSERT INTO leave_balances VALUES (?,?,?,?)",
        [(e, t, ENTITLED[t], u) for e, used in USED.items() for t, u in used.items()],
    )
    con.executemany(
        "INSERT INTO leave_requests (emp_id, type, start, end, days, status) VALUES (?,?,?,?,?,?)",
        LEAVE_REQUESTS,
    )
    con.executemany("INSERT INTO holidays VALUES (?,?)", HOLIDAYS)
    con.commit()
    con.close()


def chunk_policies(text):
    """One chunk per '## ' section."""
    chunks = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = part.partition("\n")
        chunks.append((title.strip(), f"{title.strip()}\n{body.strip()}"))
    return chunks


def seed_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    for name in ("hr_policies", "employee_context"):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    policies = client.create_collection("hr_policies")
    chunks = chunk_policies((ROOT / "data" / "policies.md").read_text(encoding="utf-8"))
    policies.add(
        ids=[f"policy-{i}" for i in range(len(chunks))],
        documents=[c for _, c in chunks],
        metadatas=[{"section": s} for s, _ in chunks],
    )

    context = client.create_collection("employee_context")
    ids, docs, metas = [], [], []
    for emp_id, profile in PROFILES.items():
        for i, para in enumerate(p.strip() for p in profile.split("\n\n") if p.strip()):
            ids.append(f"{emp_id}-{i}")
            docs.append(para)
            metas.append({"employee_id": emp_id})
    context.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Chroma: {len(chunks)} policy chunks, {len(docs)} employee chunks")


if __name__ == "__main__":
    seed_sqlite()
    seed_chroma()
    print(f"SQLite: {DB_PATH}")
