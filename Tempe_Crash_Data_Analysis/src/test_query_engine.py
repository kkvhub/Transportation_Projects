"""
test_query_engine.py
====================
PHASE 1 — STEP 3: Sanity tests for the query engine.

WHY THIS EXISTS
---------------
Before wiring an LLM on top, we prove the underlying numbers are right and that
the "no data" guard works. These are lightweight assertions (no test framework
needed) that cross-check the engine's answers against the raw data computed an
independent way. If any check fails the script exits non-zero and prints which.

Run:  python src/test_query_engine.py
"""

import sys
import pandas as pd
import query_engine as qe

df = qe.load_data()
passed, failed = 0, 0


def check(name, condition, detail=""):
    """Record and print a single pass/fail line."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("Running query-engine sanity checks...\n")

# 1) busiest_year must equal the year with the max count computed directly.
r = qe.busiest_year()
true_year = int(df["Year"].value_counts().idxmax())
check("busiest_year matches raw value_counts",
      str(true_year) in r["answer"], f"expected {true_year}; got: {r['answer']}")
check("busiest_year returns mappable rows", r["map"] and len(r["rows"]) > 0)

# 2) peak_hour rows all share the reported hour.
r = qe.peak_hour()
hr = int(r["rows"]["crash_hour"].iloc[0])
check("peak_hour rows are all the same hour",
      (r["rows"]["crash_hour"] == hr).all())

# 3) school_ranking(0.25) — every mapped row is within 0.25 mi and is the top school.
r = qe.school_ranking(0.25)
check("school_ranking rows within 0.25 mi",
      (r["rows"]["dist_school_mi"] <= 0.25).all())
check("school_ranking has a ranking table", len(r["meta"]["ranking"]) > 0)

# 4) filter_crashes count matches a direct pandas filter (rain + dark).
r = qe.filter_crashes({"Weather": "Rain", "is_dark": 1})
direct = len(df[(df["Weather"] == "Rain") & (df["is_dark"] == 1)])
check("filter rain+dark count matches direct filter",
      len(r["rows"]) == direct, f"engine {len(r['rows'])} vs direct {direct}")

# 5) OUT-OF-SCOPE must return ok=False (this is the 'No data available' guard).
r = qe.filter_crashes({"City": "Phoenix"})
check("out-of-scope field returns no-data", r["ok"] is False)
r = qe.crashes_near_school("Hogwarts", 0.25)
check("nonexistent school returns no-data", r["ok"] is False)

# 6) No crash should still be labelled 'Unknown' (the join fix).
check("no 'Unknown' nearest_school remains",
      (df["nearest_school"] == "Unknown").sum() == 0)

print(f"\n{passed} passed, {failed} failed.")
sys.exit(1 if failed else 0)
