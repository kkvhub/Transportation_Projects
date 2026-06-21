"""
query_engine.py
===============
PHASE 1 — STEP 2: The "brain that does the counting".

WHAT THIS MODULE IS (plain English)
-----------------------------------
A small library of Python functions that answer the questions the chatbot
supports. Each function does TWO things:
  1. returns a short human-readable ANSWER string, and
  2. returns the matching crash ROWS (with Latitude/Longitude) so the UI can
     plot them on a map.

WHY IT'S SEPARATE FROM THE LLM
------------------------------
The language model (Phase 3) will only TRANSLATE a user's sentence into a call
to one of these functions. The actual numbers always come from pandas here, so
they are always correct and never "hallucinated". If a question maps to nothing
in here, we return a NO-DATA result -> the bot says "No data available for that."

THE STANDARD RETURN SHAPE
-------------------------
Every public function returns a dict:
    {
      "ok":      True/False,          # False = no data / invalid request
      "answer":  "<text answer>",     # what the bot says
      "rows":    <pandas DataFrame>,  # matching crashes (may be empty); has lat/long
      "map":     True/False,          # whether rows are worth plotting
      "meta":    {...}                # optional extras (e.g. full ranking table)
    }
This uniform shape means the UI and the LLM layer never have to special-case
individual functions.

Run a quick demo:  python src/query_engine.py
"""

from functools import lru_cache
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ENRICHED = DATA / "crashes_enriched.csv"   # produced by build_dataset.py

# Human-friendly labels (source encodes crash_dow 0=Monday).
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# Data loading (cached so the 56k-row CSV is read from disk only once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_data():
    """Load the enriched crash table once and keep it in memory.

    lru_cache means repeated calls return the same in-memory DataFrame instead
    of re-reading the file -- important when the bot answers many questions.
    """
    if not ENRICHED.exists():
        raise FileNotFoundError(
            f"{ENRICHED} not found. Run `python src/build_dataset.py` first.")
    return pd.read_csv(ENRICHED, low_memory=False)


# ---------------------------------------------------------------------------
# Helpers that build the standard return dicts (keeps each function tidy)
# ---------------------------------------------------------------------------
def _ok(answer, rows, do_map=True, meta=None):
    """Build a SUCCESS result. `rows` is the DataFrame to (optionally) map.

    `meta` is an optional dict of extras (e.g. the full table behind a ranking)
    that the UI/LLM layer can use but that aren't part of the headline answer.
    """
    return {"ok": True, "answer": answer, "rows": rows,
            "map": do_map and rows is not None and len(rows) > 0,
            "meta": meta or {}}


def _no_data(reason="No data available for that."):
    """Build the standard 'out of scope / nothing found' result."""
    return {"ok": False, "answer": reason,
            "rows": pd.DataFrame(), "map": False, "meta": {}}


# ---------------------------------------------------------------------------
# TIME-BASED QUESTIONS
# ---------------------------------------------------------------------------
def busiest_year(df=None):
    """'Which year had the most crashes?' -> group by Year, count, take the max."""
    df = load_data() if df is None else df
    counts = df["Year"].value_counts().sort_values(ascending=False)
    year = int(counts.index[0])
    rows = df[df["Year"] == year]                       # rows for the map
    answer = (f"{year} had the most crashes: {counts.iloc[0]:,} "
              f"(out of {len(df):,} total, 2012-2025).")
    return _ok(answer, rows, meta={"by_year": counts.to_dict()})


def peak_hour(df=None):
    """'What time of day is most prevalent?' -> group by crash_hour (0-23)."""
    df = load_data() if df is None else df
    counts = df["crash_hour"].value_counts().sort_values(ascending=False)
    hour = int(counts.index[0])
    rows = df[df["crash_hour"] == hour]
    answer = (f"The most crash-prone time of day is {hour:02d}:00-{hour:02d}:59 "
              f"with {counts.iloc[0]:,} crashes.")
    return _ok(answer, rows, meta={"by_hour": counts.sort_index().to_dict()})


def peak_day_of_week(df=None):
    """'Which day of the week has the most crashes?' -> group by crash_dow."""
    df = load_data() if df is None else df
    counts = df["crash_dow"].value_counts().sort_index()
    busiest = int(counts.idxmax())
    rows = df[df["crash_dow"] == busiest]
    answer = f"{DOW[busiest]} has the most crashes ({counts.max():,})."
    return _ok(answer, rows,
               meta={"by_dow": {DOW[i]: int(counts.get(i, 0)) for i in range(7)}})


def peak_month(df=None):
    """'Which month has the most crashes?' -> group by crash_month."""
    df = load_data() if df is None else df
    counts = df["crash_month"].value_counts().sort_index()
    busiest = int(counts.idxmax())
    rows = df[df["crash_month"] == busiest]
    answer = f"{MONTHS[busiest]} has the most crashes ({counts.max():,})."
    return _ok(answer, rows,
               meta={"by_month": {MONTHS[i]: int(counts.get(i, 0)) for i in range(1, 13)}})


# ---------------------------------------------------------------------------
# SCHOOL-PROXIMITY QUESTIONS
# ---------------------------------------------------------------------------
def school_ranking(radius_mi=0.25, metric="crashes", top_n=10, df=None):
    """'Which school has the most crashes within X miles?'

    1) keep crashes whose nearest-school distance <= radius_mi,
    2) group by nearest_school,
    3) rank by total crashes (metric='crashes') or KSI crashes (metric='ksi'),
    4) answer = the leader; meta['ranking'] = the full top-N table.
    Mapped rows = all crashes for the #1 school within the radius.
    """
    df = load_data() if df is None else df
    near = df[df["dist_school_mi"] <= radius_mi]
    if near.empty:
        return _no_data(f"No crashes found within {radius_mi} mi of any school.")

    grp = near.groupby("nearest_school")
    table = pd.DataFrame({
        "crashes": grp.size(),
        "ksi": grp["KSI"].sum(min_count=1).fillna(0).astype(int),
    })
    sort_col = "ksi" if metric == "ksi" else "crashes"
    table = table.sort_values(sort_col, ascending=False)

    top_school = table.index[0]
    rows = near[near["nearest_school"] == top_school]
    metric_word = "KSI (serious/fatal) crashes" if metric == "ksi" else "crashes"
    answer = (f"Within {radius_mi} mi, {top_school} has the most {metric_word}: "
              f"{int(table.iloc[0][sort_col]):,} "
              f"(of {int(table['crashes'].sum()):,} crashes near schools in that radius).")
    return _ok(answer, rows,
               meta={"ranking": table.head(top_n).reset_index().to_dict("records")})


def crashes_near_school(school_name, radius_mi=0.25, df=None):
    """'How many crashes near <school>?' -- case-insensitive substring match."""
    df = load_data() if df is None else df
    mask = df["nearest_school"].str.contains(school_name, case=False, na=False)
    near = df[mask & (df["dist_school_mi"] <= radius_mi)]
    if near.empty:
        return _no_data(f'No crashes within {radius_mi} mi of a school matching '
                        f'"{school_name}". (Check the name, or widen the radius.)')
    name = near["nearest_school"].mode().iloc[0]
    ksi = int(near["KSI"].sum())
    answer = (f"{len(near):,} crashes occurred within {radius_mi} mi of {name} "
              f"({ksi} were serious/fatal).")
    return _ok(answer, near)


# ---------------------------------------------------------------------------
# GENERAL FILTER + COUNT-BY (covers most "how many ... by ..." questions)
# ---------------------------------------------------------------------------
# Fields the LLM layer is allowed to filter/group on. Anything else -> no-data,
# which is how out-of-scope questions get the "No data available" answer.
EQUALITY_FIELDS = {
    "Year", "crash_hour", "crash_dow", "crash_month", "is_weekend", "is_dark",
    "KSI", "Weather", "SurfaceCondition", "Lightcondition", "Collisionmanner",
    "JunctionRelation", "Unittype_One", "Unittype_Two", "AlcoholUse_Drv1",
    "DrugUse_Drv1", "Gender_Drv1", "Injuryseverity", "nearest_school",
}


def filter_crashes(filters=None, max_school_dist_mi=None, street=None, df=None):
    """Generic filter -> count + mapped rows.

    filters            : dict like {"Weather": "Rain", "is_dark": 1, "Year": 2023}
    max_school_dist_mi : keep only crashes within this distance of a school
    street             : case-insensitive substring match on StreetName
    """
    df = load_data() if df is None else df
    out = df
    filters = filters or {}

    for field, value in filters.items():
        if field not in EQUALITY_FIELDS:
            return _no_data(f'"{field}" is not something I have data on.')
        out = out[out[field].astype(str).str.lower() == str(value).lower()]

    if max_school_dist_mi is not None:
        out = out[out["dist_school_mi"] <= max_school_dist_mi]
    if street:
        out = out[out["StreetName"].str.contains(street, case=False, na=False)]

    if out.empty:
        return _no_data("No crashes match those conditions.")

    parts = [f"{k}={v}" for k, v in filters.items()]
    if street:
        parts.append(f"street contains '{street}'")
    if max_school_dist_mi is not None:
        parts.append(f"within {max_school_dist_mi} mi of a school")
    desc = ", ".join(parts) if parts else "all crashes"
    ksi = int(out["KSI"].sum())
    answer = f"{len(out):,} crashes match ({desc}); {ksi} were serious/fatal."
    return _ok(answer, out)


def count_by(field, df=None):
    """'Break crashes down by <field>' -- generic group-and-count for any allowed field."""
    df = load_data() if df is None else df
    if field not in EQUALITY_FIELDS:
        return _no_data(f'"{field}" is not something I have data on.')
    counts = df[field].value_counts().sort_values(ascending=False)
    top = counts.index[0]
    answer = (f'Most crashes have {field} = "{top}" ({counts.iloc[0]:,}). '
              f"See the breakdown for all {len(counts)} categories.")
    rows = df[df[field].astype(str) == str(top)]
    return _ok(answer, rows, meta={"breakdown": counts.head(25).to_dict()})


def all_crashes(df=None):
    """Return ALL crashes -- used for the landing-page overview map.

    This lets the app show the full crash picture on first load (no question
    needed yet), so the user can see where problems cluster and what to ask.
    """
    df = load_data() if df is None else df
    answer = (f"Showing all {len(df):,} Tempe crashes (2012-2025). "
              f"Ask a question to filter -- e.g. by year, time, school, or weather.")
    return _ok(answer, df, meta={"overview": True})


def _demo():
    """Print a few example answers (used when running this file directly)."""
    examples = [busiest_year(), peak_hour(), peak_day_of_week(),
                school_ranking(0.25),
                filter_crashes({"Weather": "Rain", "is_dark": 1}),
                filter_crashes({"City": "Phoenix"})]   # last is out-of-scope on purpose
    for r in examples:
        tag = "OK " if r["ok"] else "NO "
        extra = f"[{len(r['rows'])} rows]" if r["ok"] else ""
        print(tag, r["answer"], extra)


if __name__ == "__main__":
    _demo()
