"""
build_dataset.py
================
PHASE 1 — STEP 1: Build the single, clean table the chatbot queries.

WHAT THIS SCRIPT DOES (plain English)
-------------------------------------
The raw project has crash facts spread across three files:
  * school_crashes_tempe.csv  -> engineered features (KSI, crash_hour, is_dark, ...)
                                 + crash coordinates, but its nearest-school column
                                 was built with a 1-mile cap, leaving 11,704 "Unknown".
  * crash_data_tempe.csv      -> raw attributes (StreetName, Injuryseverity,
                                 driver age/gender, violation).
  * feature_table_tempe.csv   -> a couple of time features not in the school file
                                 (crash_dow = day of week, crash_month).

This script merges all three on OBJECTID, then RE-COMPUTES the nearest school for
every crash from the *verified* school file (schools_tempe_clean.json, 188 named
schools) with NO distance cap. Because Tempe is dense, every crash ends up within
~1 mile of a named school, so the "Unknown" bucket disappears (11,704 -> 0) while
the real distance is kept so "within 0.25 mi" style questions still work.

It also does light cleaning (stray numeric codes -> "Unknown", impossible ages -> NaN).

OUTPUT
------
data/crashes_enriched.csv  -> ONE row per crash, all queryable fields + a correct
                              nearest_school + dist_school_mi. The only table the
                              query engine (query_engine.py) reads.

HOW THE NEAREST-SCHOOL MATH WORKS
---------------------------------
We use the haversine formula (great-circle distance on a sphere) vectorised with
numpy: for all crashes (rows) against all schools (columns) at once we build a
(56k x 188) distance matrix, then take the column index of the minimum per row.
That index gives both the nearest school's NAME and its DISTANCE in miles.

Run:  python src/build_dataset.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths are relative to the project root (tempe_chatbot/) -------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SCHOOL_CRASHES = DATA / "school_crashes_tempe.csv"   # features + coordinates
RAW_CRASHES    = DATA / "crash_data_tempe.csv"       # raw attributes
FEATURE_TABLE  = DATA / "feature_table_tempe.csv"    # has crash_dow / crash_month
SCHOOLS_JSON   = DATA / "schools_tempe_clean.json"   # 188 verified school points
OUT            = DATA / "crashes_enriched.csv"       # <- what we produce

EARTH_RADIUS_MI = 3958.7613  # mean Earth radius in miles (for haversine)

# Categorical columns where the source contains stray numeric codes (e.g. "51",
# "255", "10"). Any value not in the known-good set is rewritten to "Unknown"
# so groupings/filters stay clean.
CLEAN_CATEGORICALS = {
    "Lightcondition": {"Daylight", "Dark Lighted", "Dark Not Lighted",
                       "Dark Unknown Lighting", "Dusk", "Dawn"},
    "Collisionmanner": {"Rear End", "ANGLE (Front To Side)(Other Than Left Turn)",
                        "Left Turn", "Single Vehicle", "Sideswipe Same Direction",
                        "Sideswipe Opposite Direction", "Head On", "Rear To Side",
                        "Rear To Rear", "U Turn", "Other", "Unknown"},
}


def haversine_nearest(crash_lat, crash_lon, school_lat, school_lon):
    """Return (nearest_index, nearest_distance_miles) for each crash.

    All inputs are 1-D numpy arrays. We broadcast crashes (rows) against schools
    (cols) to build the full distance matrix in one shot, which is fast at our
    size (~56k x 188). nanargmin picks the closest school per crash.
    """
    la1 = np.radians(crash_lat)[:, None]   # shape (n_crashes, 1)
    lo1 = np.radians(crash_lon)[:, None]
    la2 = np.radians(school_lat)[None, :]  # shape (1, n_schools)
    lo2 = np.radians(school_lon)[None, :]

    # haversine: a = sin2(dlat/2) + cos(lat1)cos(lat2) sin2(dlon/2)
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    dist = 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))   # (n_crashes, n_schools)

    idx = np.nanargmin(dist, axis=1)                     # closest school per crash
    nearest_dist = dist[np.arange(len(idx)), idx]
    return idx, nearest_dist


def main():
    # 1) Load the crash tables ------------------------------------------------
    print("Loading source tables...")
    sc = pd.read_csv(SCHOOL_CRASHES, low_memory=False)   # features + coords
    raw = pd.read_csv(RAW_CRASHES, low_memory=False)     # raw attributes
    feat = pd.read_csv(FEATURE_TABLE, low_memory=False)  # crash_dow / crash_month

    # 2) Bring across extra attributes by OBJECTID join -----------------------
    raw_cols = ["OBJECTID", "StreetName", "CrossStreet", "DateTime", "Injuryseverity",
                "Totalinjuries", "Totalfatalities", "Age_Drv1", "Gender_Drv1",
                "Violation1_Drv1"]
    df = sc.merge(raw[raw_cols], on="OBJECTID", how="left")
    df = df.merge(feat[["OBJECTID", "crash_dow", "crash_month"]], on="OBJECTID", how="left")
    print(f"Merged table: {df.shape[0]} rows, {df.shape[1]} columns "
          f"(raw-attribute match rate {df['StreetName'].notna().mean():.1%})")

    # 3) Re-compute nearest school from the VERIFIED school file --------------
    schools = json.load(open(SCHOOLS_JSON))["features"]
    s_name = np.array([f["properties"].get("SchoolName", "") for f in schools])
    s_lat  = np.array([f["geometry"]["coordinates"][1] for f in schools])
    s_lon  = np.array([f["geometry"]["coordinates"][0] for f in schools])

    idx, dist_mi = haversine_nearest(df["Latitude"].to_numpy(),
                                     df["Longitude"].to_numpy(), s_lat, s_lon)
    df["nearest_school"] = s_name[idx]
    df["dist_school_mi"] = np.round(dist_mi, 4)
    df["dist_school_ft"] = np.round(dist_mi * 5280, 1)

    unknown_before = (sc["nearest_school"] == "Unknown").sum()
    print(f'"Unknown" nearest_school: {unknown_before} (old, capped) -> '
          f'{(df["nearest_school"] == "Unknown").sum()} (new). '
          f'Distinct schools referenced: {df["nearest_school"].nunique()}')

    # 4) Light data cleaning --------------------------------------------------
    for col, good in CLEAN_CATEGORICALS.items():            # stray codes -> Unknown
        if col in df.columns:
            bad = ~df[col].isin(good) & df[col].notna()
            n_bad = int(bad.sum())
            if n_bad:
                df.loc[bad, col] = "Unknown"
                print(f'  cleaned {n_bad} stray codes in {col} -> "Unknown"')
    if "Age_Drv1" in df.columns:                            # impossible ages -> NaN
        bad_age = (df["Age_Drv1"] < 14) | (df["Age_Drv1"] > 100)
        df.loc[bad_age, "Age_Drv1"] = np.nan
        print(f"  blanked {int(bad_age.sum())} impossible Age_Drv1 values")

    # 5) Save ----