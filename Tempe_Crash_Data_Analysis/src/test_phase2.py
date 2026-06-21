"""
test_phase2.py
==============
PHASE 2 sanity tests: the many-to-many buffer view and the map renderer.

Proves:
  * the buffer view counts MORE total school-crash memberships than the
    nearest-only view (because overlap is allowed),
  * some crash really is shared by >1 school at 0.25 mi (the double-count case),
  * buffer per-school counts never exceed the true number of crashes in range,
  * the map renderer writes a valid HTML file for a normal result AND for an
    empty (no-data) result.

Run:  python src/test_phase2.py
"""

import sys
from pathlib import Path
import pandas as pd

import query_engine as qe
import buffer_analysis as ba
import map_render as mr

passed, failed = 0, 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}  {detail}")


print("Running Phase 2 sanity checks...\n")

pairs = ba.load_pairs()
crashes = qe.load_data()

# 1) Buffer membership total (0.25 mi) should exceed distinct in-buffer crashes,
#    which is the whole point: overlap means sum of per-school counts > crashes.
p025 = pairs[pairs["dist_mi"] <= 0.25]
total_memberships = len(p025)
distinct_crashes = p025["OBJECTID"].nunique()
check("buffer memberships > distinct crashes (overlap exists)",
      total_memberships > distinct_crashes,
      f"{total_memberships} vs {distinct_crashes}")

# 2) At least one crash is within 0.25 mi of more than one school.
shared = p025["OBJECTID"].value_counts()
check("a crash is shared by >1 school at 0.25 mi", (shared > 1).any(),
      f"max schools per crash = {int(shared.max())}")

# 3) Buffer top-school crash count must be <= distinct crashes in its own buffer
#    (a per-school count can't exceed the crashes actually near that school).
r = ba.school_ranking_buffer(0.25)
top = r["meta"]["ranking"][0]
top_name, top_count = top["school_name"], top["crashes"]
true_in_range = pairs[(pairs["dist_mi"] <= 0.25)
                      & (pairs["school_name"] == top_name)]["OBJECTID"].nunique()
check("buffer top-school count matches its in-range crashes",
      top_count == true_in_range, f"{top_count} vs {true_in_range}")

# 4) Buffer view returns mappable rows.
check("buffer ranking returns mappable rows", r["map"] and len(r["rows"]) > 0)

# 5) Map renderer writes a valid HTML file for a real result.
tmp = Path("/tmp/_t_map.html")
mr.render_result(r, tmp, title="test", school=top_name, radius_mi=0.25)
ok_html = tmp.exists() and tmp.stat().st_size > 1000 and "leaflet" in tmp.read_text().lower()
check("map renderer writes valid HTML", ok_html)

# 6) Map renderer handles an empty (no-data) result without crashing.
tmp2 = Path("/tmp/_t_empty.html")
mr.render_result(qe._no_data(), tmp2, title="empty")
check("map renderer handles empty result", tmp2.exists() and tmp2.stat().st_size > 500)

# 7) Out-of-range radius is rejected cleanly (buffer table only goes to 0.5 mi).
check("radius > 0.5 mi returns no-data", ba.school_ranking_buffer(0.9)["ok"] is False)

print(f"\n{passed} passed, {failed} failed.")
sys.exit(1 if failed else 0)
