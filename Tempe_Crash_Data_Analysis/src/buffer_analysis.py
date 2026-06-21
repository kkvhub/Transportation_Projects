"""
buffer_analysis.py
==================
PHASE 2 — STEP 2: School questions answered in the BUFFER (overlap-allowed) view.

THE TWO VIEWS (read this first)
-------------------------------
There are two legitimate ways to count "crashes near a school":

  * NEAREST view  (query_engine.school_ranking)
      Each crash belongs to exactly ONE school (its closest). No double counting;
      counts partition the dataset. Good for "whose catchment has the most crashes".

  * BUFFER view  (THIS module)
      A crash belongs to EVERY school whose 0.25-mi zone it falls inside. A crash
      near two close schools is counted for BOTH -- the correct answer to
      "how many crashes are physically within 0.25 mi of school X?". Totals here
      can exceed the number of crashes because of intentional overlap.

This module reads data/crash_school_pairs.csv (built by build_buffer_table.py),
which precomputes every crash-school pair within 0.5 mi, then filters to the
requested radius. It returns the SAME standard result dict as query_engine so the
UI/LLM layer treats both views identically:
    {"ok", "answer", "rows", "map", "meta"}

Run a demo:  python src/buffer_analysis.py
"""

from functools import lru_cache
from pathlib import Path
import pandas as pd

import query_engine as qe   # reuse load_data() (the crash table) and result helpers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAIRS = DATA / "crash_school_pairs.csv"


@lru_cache(maxsize=1)
def load_pairs():
    """Load the many-to-many crash<->school pair table once (cached)."""
    if not PAIRS.exists():
        raise FileNotFoundError(
            f"{PAIRS} not found. Run `python src/build_buffer_table.py` first.")
    return pd.read_csv(PAIRS, low_memory=False)


def school_ranking_buffer(radius_mi=0.25, metric="crashes", top_n=10):
    """'Which school has the most crashes within X mi?' -- BUFFER (overlap) view.

    Steps:
      1. take all crash-school pairs with dist_mi <= radius_mi,
      2. attach each pair's KSI flag from the crash table,
      3. group by school and count crashes (or KSI crashes),
      4. answer = the leader; meta['ranking'] = full top-N table.
    A crash within range of several schools contributes to each of them (overlap),
    so this can rank schools differently from the nearest-only view.

    Mapped rows = the actual crash records within radius_mi of the #1 school.
    """
    if radius_mi > 0.5:
        return qe._no_data("Buffer table only precomputes to 0.5 mi. "
                           "Use a radius <= 0.5, or rebuild with a larger MAX_RADIUS_MI.")
    pairs = load_pairs()
    near = pairs[pairs["dist_mi"] <= radius_mi]
    if near.empty:
        return qe._no_data(f"No crashes within {radius_mi} mi of any school.")

    crashes = qe.load_data()[["OBJECTID", "KSI"]]
    near = near.merge(crashes, on="OBJECTID", how="left")   # bring in severity

    grp = near.groupby("school_name")
    table = pd.DataFrame({
        "crashes": grp["OBJECTID"].nunique(),               # distinct crashes per school
        "ksi": grp["KSI"].sum(min_count=1).fillna(0).astype(int),
    })
    sort_col = "ksi" if metric == "ksi" else "crashes"
    table = table.sort_values(sort_col, ascending=False)

    top_school = table.index[0]
    # Pull the real crash rows (with lat/long) for the winning school, for the map.
    top_ids = near[near["school_name"] == top_school]["OBJECTID"].unique()
    rows = qe.load_data()
    rows = rows[rows["OBJECTID"].isin(top_ids)]

    metric_word = "KSI (serious/fatal) crashes" if metric == "ksi" else "crashes"
    answer = (f"[Buffer view] Within {radius_mi} mi, {top_school} has the most "
              f"{metric_word}: {int(table.iloc[0][sort_col]):,}. "
              f"(Overlap allowed -- crashes near several schools count for each.)")
    return qe._ok(answer, rows,
                  meta={"ranking": table.head(top_n).reset_index().to_dict("records")})


def crashes_in_buffer(school_name, radius_mi=0.25):
    """'How many crashes within X mi of <school>?' -- BUFFER view, substring match."""
    if radius_mi > 0.5:
        return qe._no_data("Buffer table only precomputes to 0.5 mi.")
    pairs = load_pairs()
    near = pairs[(pairs["dist_mi"] <= radius_mi)
                 & pairs["school_name"].str.contains(school_name, case=False, na=False)]
    if near.empty:
        return qe._no_data(f'No crashes within {radius_mi} mi of a school matching '
                           f'"{school_name}".')
    name = near["school_name"].mode().iloc[0]
    ids = near["OBJECTID"].unique()
    rows = qe.load_data()
    rows = rows[rows["OBJECTID"].isin(ids)]
    ksi = int(rows["KSI"].sum())
    answer = (f"[Buffer view] {len(ids):,} crashes occurred within {radius_mi} mi of "
              f"{name} ({ksi} serious/fatal). May overlap neighbouring schools' zones.")
    return qe._ok(answer, rows)


def _demo():
    r = school_ranking_buffer(0.25)
    print("OK " if r["ok"] else "NO ", r["answer"], f"[{len(r['rows'])} rows]")
    # Show how the two views differ on the same radius.
    near = qe.school_ranking(0.25)
    print("\nNearest-only top school:", near["meta"]["ranking"][0])
    print("Buffer (overlap) top school:", r["meta"]["ranking"][0])


if __name__ == "__main__":
    _demo()
