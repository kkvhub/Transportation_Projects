"""
analysis.py
===========
PHASE 5 (rework): the COMPOSABLE query engine.

WHY THIS REPLACES THE ONE-ACTION-AT-A-TIME APPROACH
---------------------------------------------------
Earlier, each question mapped to a single function (busiest_year, school_ranking,
...). That couldn't answer compositional questions like:
  * "crashes within 0.25 mi of Holdeman Elementary WITH KSI = 1"   (filter + count)
  * "what time of day do most crashes happen NEAR Holdeman"        (filter + metric)
because it could not combine a spatial/attribute FILTER with an AGGREGATION.

This module fixes that with a two-step model driven by a single "spec" dict:
        build a SUBSET (school buffer + attribute filters)  ->  run a METRIC on it.

THE SPEC (what the router produces)
-----------------------------------
{
  "metric":   "count" | "peak_hour" | "peak_day" | "peak_month" |
              "busiest_year" | "school_ranking" | "breakdown",
  "by_field": <field>      # only for "breakdown"
  "school":   <free-text school name or None>,
  "radius_mi":<float, default 0.25>,
  "view":     "buffer" | "nearest",   # buffer = overlap allowed (default)
  "filters":  {field: value, ...},    # equality filters, e.g. {"KSI": 1, "Year": 2023}
  "street":   <substring or None>,
  "in_scope": True/False              # False -> "No data available"
}

Every public call returns the SAME standard dict the rest of the app expects:
    {"ok", "answer", "rows", "map", "meta"}  (+ meta["spec"] for the UI)

Run a demo:  python src/analysis.py
"""

import re
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd

import query_engine as qe   # reuse load_data(), _ok(), _no_data(), DOW, MONTHS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAIRS = DATA / "crash_school_pairs.csv"           # many-to-many buffer table
SCHOOLS_JSON = DATA / "schools_tempe_clean.json"

# Equality fields the engine will filter on (a question referencing anything else
# is treated as out of scope by the router).
EQUALITY_FIELDS = {
    "Year", "crash_hour", "crash_dow", "crash_month", "is_weekend", "is_dark",
    "KSI", "Weather", "SurfaceCondition", "Lightcondition", "Collisionmanner",
    "JunctionRelation", "Unittype_One", "Unittype_Two", "AlcoholUse_Drv1",
    "DrugUse_Drv1", "Gender_Drv1", "Injuryseverity", "nearest_school",
}


@lru_cache(maxsize=1)
def load_pairs():
    """Load the many-to-many crash<->school pair table (built in Phase 2)."""
    if not PAIRS.exists():
        raise FileNotFoundError(f"{PAIRS} not found. Run build_buffer_table.py first.")
    return pd.read_csv(PAIRS, low_memory=False)


@lru_cache(maxsize=1)
def _school_name_index():
    """Build {search_key -> canonical_name} for fuzzy school lookup.

    For every school we register its full name AND a few shortened variants
    (dropping trailing 'School'/'Academy'/'Elementary'...), so a user typing
    'Holdeman' or 'Holdeman Elementary' still resolves to 'Holdeman Elementary
    School'. Aliases from the dedup step are included too.
    """
    feats = json.load(open(SCHOOLS_JSON))["features"]
    index = {}
    SUFFIXES = [" elementary school", " high school", " middle school",
                " school", " academy", " center", " charter school",
                " elementary", " preparatory", " prep"]
    GENERIC = {"school", "academy", "high", "elementary", "middle", "center",
               "prep", "the", "of", "tempe"}
    for f in feats:
        canonical = f["properties"].get("SchoolName", "")
        names = [canonical] + (f["properties"].get("aliases") or [])
        for nm in names:
            nl = nm.lower().strip()
            keys = {nl}
            for suf in SUFFIXES:
                if nl.endswith(suf):
                    keys.add(nl[: -len(suf)].strip())
            for k in keys:
                if len(k) >= 4 and k not in GENERIC:
                    # Keep the longest canonical if a key collides.
                    index.setdefault(k, canonical)
    return index


def find_school(text):
    """Return the canonical school name mentioned in `text`, or None.

    Picks the LONGEST matching key so 'holdeman elementary school' beats a bare
    'holdeman', and a more specific school wins over a generic token.
    """
    q = (text or "").lower()
    best, best_len = None, 0
    for key, canonical in _school_name_index().items():
        if key in q and len(key) > best_len:
            best, best_len = canonical, len(key)
    return best


def build_subset(spec):
    """Apply the spec's school buffer + attribute filters -> a crash DataFrame."""
    df = qe.load_data()
    school = spec.get("school")
    radius = float(spec.get("radius_mi", 0.25))
    view = spec.get("view", "buffer")

    if school:
        if view == "buffer":
            pr = load_pairs()
            ids = pr[(pr["dist_mi"] <= radius)
                     & (pr["school_name"] == school)]["OBJECTID"].unique()
            sub = df[df["OBJECTID"].isin(ids)]
        else:  # nearest-only partition view
            sub = df[(df["dist_school_mi"] <= radius)
                     & (df["nearest_school"] == school)]
    else:
        sub = df

    for field, value in (spec.get("filters") or {}).items():
        if field not in EQUALITY_FIELDS:
            continue
        col = sub[field]
        try:
            # Numeric columns (KSI is 1.0/0.0, Year is int) -> compare as numbers
            # so a filter value of 1 matches the stored 1.0.
            fv = float(value)
            sub = sub[col.astype(float) == fv]
        except (ValueError, TypeError):
            sub = sub[col.astype(str).str.lower() == str(value).lower()]
    if spec.get("street"):
        sub = sub[sub["StreetName"].str.contains(spec["street"], case=False, na=False)]
    return sub


def _scope_phrase(spec):
    """Human-readable description of the filters, for the answer text."""
    parts = []
    if spec.get("school"):
        parts.append(f"within {spec.get('radius_mi', 0.25)} mi of {spec['school']}")
    for k, v in (spec.get("filters") or {}).items():
        parts.append(f"{k}={v}")
    if spec.get("street"):
        parts.append(f"on {spec['street']}")
    return (" " + ", ".join(parts)) if parts else ""


def analyze(spec):
    """Run the spec: build the subset, then compute the requested metric."""
    if not spec.get("in_scope", True) or not spec.get("metric"):
        return qe._no_data()

    # Resolve a free-text school name to the canonical one (if any was given).
    if spec.get("school"):
        matched = find_school(spec["school"])
        if not matched:
            return qe._no_data(f'I couldn\'t find a school matching '
                               f'"{spec["school"]}" in Tempe.')
        spec["school"] = matched

    metric = spec["metric"]

    # school_ranking ranks schools, so it must NOT pre-filter to one school.
    if metric == "school_ranking":
        return _school_ranking(spec)

    sub = build_subset(spec)
    if len(sub) == 0:
        return qe._no_data("No crashes match those conditions.")
    scope = _scope_phrase(spec)

    if metric == "count":
        ksi = int(sub["KSI"].sum())
        ans = f"{len(sub):,} crashes{scope}; {ksi} were serious/fatal (KSI)."
        return qe._ok(ans, sub, meta={"spec": spec})

    if metric in ("peak_hour", "peak_day", "peak_month", "busiest_year"):
        col = {"peak_hour": "crash_hour", "peak_day": "crash_dow",
               "peak_month": "crash_month", "busiest_year": "Year"}[metric]
        counts = sub[col].value_counts()
        top = counts.idxmax()
        if metric == "peak_hour":
            label = f"{int(top):02d}:00-{int(top):02d}:59"
            meta = {"by_hour": counts.sort_index().to_dict()}
        elif metric == "peak_day":
            label = qe.DOW[int(top)]
            meta = {"by_dow": {qe.DOW[i]: int(counts.get(i, 0)) for i in range(7)}}
        elif metric == "peak_month":
            label = qe.MONTHS[int(top)]
            meta = {"by_month": {qe.MONTHS[i]: int(counts.get(i, 0)) for i in range(1, 13)}}
        else:
            label = str(int(top))
            meta = {"by_year": counts.sort_index().to_dict()}
        meta["spec"] = spec
        word = {"peak_hour": "time of day", "peak_day": "day", "peak_month": "month",
                "busiest_year": "year"}[metric]
        ans = (f"The most crash-prone {word}{scope} is {label} "
               f"({int(counts.max()):,} of {len(sub):,} crashes).")
        return qe._ok(ans, sub, meta=meta)   # map shows ALL matching crashes

    if metric == "breakdown":
        field = spec.get("by_field")
        if field not in EQUALITY_FIELDS:
            return qe._no_data(f'"{field}" is not something I can break down.')
        counts = sub[field].value_counts()
        top = counts.index[0]
        ans = (f'Most crashes{scope} have {field} = "{top}" '
               f"({int(counts.iloc[0]):,}); {len(counts)} categories total.")
        return qe._ok(ans, sub, meta={"breakdown": counts.head(25).to_dict(),
                                      "spec": spec})

    return qe._no_data()


def _school_ranking(spec):
    """Rank schools by crashes (or KSI) within radius, honouring attribute filters.

    Uses the buffer (overlap) view by default; 'nearest' uses the partition view.
    Attribute filters in spec['filters'] (e.g. Year=2023) are applied first.
    """
    radius = float(spec.get("radius_mi", 0.25))
    view = spec.get("view", "buffer")
    rank_metric = spec.get("rank_metric", "crashes")   # 'crashes' or 'ksi'

    # Base = all crashes matching the attribute filters (NO school scope yet).
    base_spec = dict(spec); base_spec["school"] = None
    base = build_subset(base_spec)
    if len(base) == 0:
        return qe._no_data("No crashes match those conditions.")

    if view == "buffer":
        pr = load_pairs()
        near = pr[pr["dist_mi"] <= radius]
        near = near[near["OBJECTID"].isin(base["OBJECTID"])]
        if near.empty:
            return qe._no_data(f"No crashes within {radius} mi of any school.")
        ksi_map = base.set_index("OBJECTID")["KSI"]
        near = near.assign(KSI=near["OBJECTID"].map(ksi_map).fillna(0))
        grp = near.groupby("school_name")
        table = pd.DataFrame({"crashes": grp["OBJECTID"].nunique(),
                              "ksi": grp["KSI"].sum().astype(int)})
        view_word = "Buffer view"
        top_ids = near[near["school_name"] == None]  # placeholder, set below
    else:
        near = base[base["dist_school_mi"] <= radius]
        if near.empty:
            return qe._no_data(f"No crashes within {radius} mi of any school.")
        grp = near.groupby("nearest_school")
        table = pd.DataFrame({"crashes": grp.size(),
                              "ksi": grp["KSI"].sum(min_count=1).fillna(0).astype(int)})
        view_word = "Nearest view"

    sort_col = "ksi" if rank_metric == "ksi" else "crashes"
    table = table.sort_values(sort_col, ascending=False)
    top_school = table.index[0]

    # Rows for the map = crashes near the #1 school.
    if view == "buffer":
        ids = near[near["school_name"] == top_school]["OBJECTID"].unique()
        rows = qe.load_data()
        rows = rows[rows["OBJECTID"].isin(ids)]
    else:
        rows = near[near["nearest_school"] == top_school]

    word = "KSI (serious/fatal) crashes" if rank_metric == "ksi" else "crashes"
    scope = _scope_phrase({k: v for k, v in spec.items() if k != "school"})
    ans = (f"[{view_word}] Within {radius} mi, {top_school} has the most {word}: "
           f"{int(table.iloc[0][sort_col]):,}{(' (' + scope.strip() + ')') if scope.strip() else ''}.")
    spec2 = dict(spec); spec2["school"] = top_school
    return qe._ok(ans, rows, meta={"ranking": table.head(10).reset_index().to_dict("records"),
                                   "spec": spec2})


def _demo():
    examples = [
        {"metric": "count", "school": "Holdeman Elementary", "radius_mi": 0.25,
         "filters": {"KSI": 1}, "view": "buffer", "in_scope": True},
        {"metric": "peak_hour", "school": "Holdeman Elementary", "radius_mi": 0.25,
         "filters": {}, "view": "buffer", "in_scope": True},
        {"metric": "school_ranking", "radius_mi": 0.25, "rank_metric": "ksi",
         "filters": {}, "view": "buffer", "in_scope": True},
    ]
    for sp in examples:
        r = analyze(dict(sp))
        print(("OK " if r["ok"] else "NO "), r["answer"], f"[{len(r['rows'])} rows]")


if __name__ == "__main__":
    _demo()
