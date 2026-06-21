"""
test_battery.py
===============
Broad battery of representative questions through the composable router (rule
engine). Asserts the parsed METRIC and whether the question succeeds (ok).
Out-of-scope / junk must return ok=False.
"""

import sys
import nl_router as nlr

# (question, expected_metric, expected_ok)
CASES = [
    ("Which year had the most crashes?", "busiest_year", True),
    ("what year was worst for crashes", "busiest_year", True),
    ("What time of day has the most accidents?", "peak_hour", True),
    ("which hour is most dangerous", "peak_hour", True),
    ("Which day of the week has the most crashes?", "peak_day", True),
    ("Which month has the most crashes?", "peak_month", True),
    ("Which school has the most crashes within 0.25 mi?", "school_ranking", True),
    ("which school has the most serious crashes within half a mile", "school_ranking", True),
    # compositional
    ("crashes within 0.25 mi of Holdeman Elementary with KSI = 1", "count", True),
    ("what time of day do most crashes happen near Holdeman Elementary", "peak_hour", True),
    ("how many crashes near Marcos De Niza High", "count", True),
    ("alcohol crashes near Tempe High School", "count", True),
    ("peak month for crashes near Mcclintock High School", "peak_month", True),
    # filters
    ("how many crashes in the rain", "count", True),
    ("alcohol related crashes at night in 2023", "count", True),
    ("pedestrian crashes", "count", True),
    ("head-on crashes on Rural Rd", "count", True),
    ("crashes in 2020", "count", True),
    ("weekend crashes", "count", True),
    # breakdown
    ("break down crashes by weather", "breakdown", True),
    ("crashes by severity", "breakdown", True),
    # out of scope / junk
    ("How many crashes in Phoenix?", None, False),
    ("predict next year's crashes", None, False),
    ("what is the cost of crashes", None, False),
    ("what car model crashes most", None, False),
    ("", None, False),
    ("hello there", None, False),
    # tricky wording that must NOT misfire
    ("which year had the highest crashes", "busiest_year", True),
    ("crashes at high speed in the rain", "count", True),
    ("crashes on the highway at night", "count", True),
]

passed = failed = 0
print(f"Running question battery ({len(CASES)} cases)...\n")
for q, exp_metric, exp_ok in CASES:
    r = nlr.answer(q, prefer_llm=False)
    got_metric = r.get("intent", {}).get("metric")
    if r["ok"] == exp_ok and got_metric == exp_metric:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {q!r}\n        expected (metric={exp_metric}, ok={exp_ok}) "
              f"got (metric={got_metric}, ok={r['ok']})")

print(f"\n{passed}/{len(CASES)} passed, {failed} failed.")
sys.exit(1 if failed else 0)
