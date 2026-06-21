"""
build_buffer_table.py
=====================
PHASE 2 — STEP 1: Build the MANY-TO-MANY crash<->school buffer table.

WHY THIS EXISTS (the double-counting question)
----------------------------------------------
crashes_enriched.csv assigns each crash to its ONE nearest school (a clean,
no-overlap "partition" view). But if two schools sit close together, a crash can
genuinely be inside BOTH of their 0.25-mile zones. The nearest-only view hides
that: it gives the crash to whichever school is marginally closer, so the other
school under-counts.

This script builds the complementary "buffer membership" view: it links each
crash to EVERY school within a generous radius (0.5 mi), not just the closest.
With this table a crash near several schools is intentionally counted for each of
them -- the correct behaviour when the question is "how many crashes are within
0.25 mi of school X?".

We precompute out to 0.5 mi (a superset) so any query radius up to 0.5 mi
(0.1, 0.25, 0.5, ...) can be answered by simply filtering dist_mi <= radius,
without recomputing distances.

OUTPUT
------
data/crash_school_pairs.csv  -> columns: OBJECTID, school_name, dist_mi
                                (one ROW per crash-school pair within 0.5 mi;
                                 ~152k rows, avg ~2.7 schools per crash).

HOW
---
Same haversine distance as build_dataset.py, but instead of taking the single
minimum per crash we keep ALL (crash, school) pairs whose distance <= 0.5 mi.
We process crashes in chunks of 4,000 so the temporary distance matrix stays
small in memory.

Run:  python src/build_buffer_table.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ENRICHED     = DATA / "crashes_enriched.csv"
SCHOOLS_JSON = DATA / "schools_tempe_clean.json"
OUT          = DATA / "crash_school_pairs.csv"

EARTH_RADIUS_MI = 3958.7613
MAX_RADIUS_MI = 0.5     # precompute pairs out to here (superset of any query radius)
CHUNK = 4000            # crashes processed per batch (keeps memory modest)


def main():
    df = pd.read_csv(ENRICHED, low_memory=False)
    schools = json.load(open(SCHOOLS_JSON))["features"]
    s_name = np.array([f["properties"].get("SchoolName", "") for f in schools])
    s_lat  = np.array([f["geometry"]["coordinates"][1] for f in schools])
    s_lon  = np.array([f["geometry"]["coordinates"][0] for f in schools])

    clat = df["Latitude"].to_numpy()
    clon = df["Longitude"].to_numpy()
    oid  = df["OBJECTID"].to_numpy()

    # School coordinates are constant across chunks, so precompute their radians.
    la2 = np.radians(s_lat)[None, :]
    lo2 = np.radians(s_lon)[None, :]

    pair_oid, pair_sidx, pair_dist = [], [], []
    for i in range(0, len(df), CHUNK):
        la1 = np.radians(clat[i:i + CHUNK])[:, None]
        lo1 = np.radians(clon[i:i + CHUNK])[:, None]
        a = (np.sin((la2 - la1) / 2) ** 2
             + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
        d = 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))   # (chunk, n_schools)

        # Keep every (crash, school) cell within the radius -> many-to-many.
        rows, cols = np.where(d <= MAX_RADIUS_MI)
        pair_oid.append(oid[i:i + CHUNK][rows])
        pair_sidx.append(cols)
        pair_dist.append(d[rows, cols])

    pairs = pd.DataFrame({
        "OBJECTID": np.concatenate(pair_oid),
        "school_name": s_name[np.concatenate(pair_sidx)],
        "dist_mi": np.round(np.concatenate(pair_dist), 4),
    })
    pairs.to_csv(OUT, index=False)

    n_crashes = pairs["OBJECTID"].nunique()
    print(f"Saved {len(pairs):,} crash-school pairs (<= {MAX_RADIUS_MI} mi) -> {OUT}")
    print(f"  {n_crashes:,} distinct crashes; "
          f"avg {len(pairs)/n_crashes:.2f} schools per crash within {MAX_RADIUS_MI} mi")


if __name__ == "__main__":
    main()
