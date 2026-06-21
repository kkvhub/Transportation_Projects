"""
test_phase3.py
==============
Tests for the composable NL router (rule parser + analysis dispatch), run WITHOUT
a Gemini key so they're deterministic. Focus: the spec is parsed correctly and
compositional questions (filter + metric + school) produce correct numbers.
"""

import sys
import nl_router as nlr
import analysis as an

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}  {detail}")


print("Running Phase 3 (composable) checks...\n")

# 1) Metric mapping.
metric_cases = {
    "Which year had the most crashes?": "busiest_year",
    "What time of day is most accidents?": "peak_hour",
    "Which day of the week is worst?": "peak_day",
    "Which month has the most crashes?": "peak_month",
    "Which school has the most crashes within 0.25 mi?": "school_ranking",
    "Break down crashes by weather": "breakdown",
    "alcohol crashes at night in 2023": "count",
    "How many crashes in Mesa?": None,
}
for q, exp in metric_cases.items():
    got = nlr.parse_spec(q)["metric"]
    check(f'metric {exp} <- {q!r}', got == exp, f"got {got}")

# 2) COMPOSITION: KSI filter + school + count (the failing case #1).
sp = nlr.parse_spec("number of crashes in 0.25mi of Holdeman Elementary School with KSI = 1")
check("KSI=1 captured as filter", sp["filters"].get("KSI") == 1, str(sp["filters"]))
check("Holdeman resolved as school", sp["school"] == "Holdeman Elementary School", sp["school"])
r = an.analyze(dict(sp))
check("Holdeman KSI=1 count is 19", r["ok"] and len(r["rows"]) == 19, r["answer"])

# 3) COMPOSITION: peak hour near a school (the failing case #2).
sp2 = nlr.parse_spec("at what time of day most crashes happen in 0.25mi of Holdeman Elementary School")
check("metric is peak_hour", sp2["metric"] == "peak_hour")
check("school still Holdeman", sp2["school"] == "Holdeman Elementary School")
r2 = an.analyze(dict(sp2))
check("peak hour near Holdeman = 17:00", r2["ok"] and "17:00" in r2["answer"], r2["answer"])
check("peak-hour map shows the 697 area crashes", len(r2["rows"]) == 697, len(r2["rows"]))

# 4) Buffer default + nearest override.
check("school question defaults to buffer", nlr.answer(
    "which school has most crashes within 0.25 mi", prefer_llm=False)["intent"]["view"] == "buffer")
check("'nearest' switches view", nlr.parse_spec(
    "which nearest school has most crashes")["view"] == "nearest")

# 5) Ranking by KSI matches the known top (Mi Escuelita, 24).
rr = nlr.answer("which school has the most serious crashes within 0.25 mi", prefer_llm=False)
check("ranking top is Mi Escuelita w/ 24 ksi",
      rr["ok"] and "Mi Escuelita" in rr["answer"] and "24" in rr["answer"], rr["answer"])

# 6) Out-of-scope + guards.
check("Phoenix out of scope", nlr.answer("crashes in Phoenix", prefer_llm=False)["ok"] is False)
check("empty question guarded", nlr.answer("", prefer_llm=False)["ok"] is False)

print(f"\n{passed} passed, {failed} failed.")
sys.exit(1 if failed else 0)
