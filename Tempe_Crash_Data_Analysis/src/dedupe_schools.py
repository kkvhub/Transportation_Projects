"""
dedupe_schools.py
=================
PHASE 5 — Collapse co-located (same-coordinate) school entries to ONE canonical
school per point, keeping the other names as aliases.

WHY (the problem this fixes)
----------------------------
The verified school file has 188 points but only 142 distinct coordinates: 35
locations carry 2-4 entries each (e.g. "ECA - Arizona, Inc." and
"The Early Career Academy" at the exact same spot, or "EVIT - Tempe High School"
and "Tempe High School"). In the buffer view this made one physical campus appear
as several schools with identical crash counts -- noisy and confusing.

Since two entries at the SAME coordinate are spatially indistinguishable (any
crash is equidistant to both), we keep ONE canonical name per coordinate and list
the rest as `aliases`. No spatial information is lost; we only stop double-listing
the same point.

HOW THE CANONICAL NAME IS CHOSEN
--------------------------------
For each coordinate we score the candidate names and keep the "cleanest":
  - drop obvious placeholders ("DRP Placeholder ...") when alternatives exist,
  - penalise corporate/operator names (Inc, LLC, dba, Corporation, Holdings,
    Enterprises) -- these are usually the legal operator, not the school's
    common name,
  - among the rest prefer the shorter name, then alphabetical (stable).
All original names are preserved in properties["aliases"] so nothing is lost and
you can override a choice later.

SAFETY
------
* Backs up the current file to schools_tempe_clean_PREDEDUPE_<date>.json.
* Writes a human-readable review report to data/school_dedup_report.csv.
* Overwrites schools_tempe_clean.json with the 142-point deduped file so the
  existing build scripts keep working unchanged.

After running this, rebuild downstream data:
    python src/dedupe_schools.py
    python src/build_dataset.py
    python src/build_buffer_table.py
"""

import csv
import datetime
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHOOLS = DATA / "schools_tempe_clean.json"
REPORT = DATA / "school_dedup_report.csv"

CORP_TOKENS = ["inc", "llc", "l.l.c", "dba", "corporation", "holdings",
               "enterprises", "incorp", "placeholder", "drp"]

# Words that mark a real school's common name (preferred for the canonical).
DESCRIPTORS = ["elementary", "middle", "high school", "academy", "school",
               "prep", "preparatory", "college", "k-8", "kindergarten",
               "head start"]


def _is_acronymish(name):
    """True for short all-caps-ish tokens like 'TAPBI' or 'MCHS' we'd rather avoid."""
    letters = name.replace(" ", "").replace("-", "")
    return len(name) <= 6 or (letters.isupper() and len(letters) <= 6)


def _score(name):
    """Lower score = better canonical candidate. Compared left-to-right:
    1) has a school-descriptor word (prefer yes), 2) fewer corporate tokens,
    3) not acronym-like, 4) shorter, 5) alphabetical (stable)."""
    low = name.lower()
    has_descriptor = any(d in low for d in DESCRIPTORS)
    corp_hits = sum(1 for t in CORP_TOKENS if t in low)
    return (0 if has_descriptor else 1, corp_hits,
            1 if _is_acronymish(name) else 0, len(name), name)


def pick_canonical(names):
    """Choose the cleanest display name from a list of co-located names."""
    uniq = list(dict.fromkeys(names))             # de-dup exact repeats, keep order
    non_placeholder = [n for n in uniq if "placeholder" not in n.lower()]
    pool = non_placeholder or uniq
    canonical = min(pool, key=_score)
    aliases = [n for n in uniq if n != canonical]
    return canonical, aliases


def main():
    fc = json.load(open(SCHOOLS))
    feats = fc["features"]

    # Group features by rounded coordinate.
    groups = defaultdict(list)
    for f in feats:
        lon, lat = f["geometry"]["coordinates"]
        groups[(round(lat, 6), round(lon, 6))].append(f)

    # Backup before writing anything.
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    shutil.copy(SCHOOLS, DATA / f"schools_tempe_clean_PREDEDUPE_{stamp}.json")

    new_feats, report_rows = [], []
    for (lat, lon), group in groups.items():
        names = [g["properties"].get("SchoolName", "") for g in group]
        canonical, aliases = pick_canonical(names)
        # Reuse the chosen feature's properties; set canonical name + aliases.
        base = dict(group[0]["properties"])
        base["SchoolName"] = canonical
        base["aliases"] = aliases
        new_feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": base})
        if aliases:   # only report the merges
            report_rows.append({"latitude": lat, "longitude": lon,
                                "canonical_name": canonical,
                                "merged_aliases": " | ".join(aliases)})

    json.dump({"type": "FeatureCollection", "features": new_feats},
              open(SCHOOLS, "w"), indent=1)

    with open(REPORT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["latitude", "longitude",
                                           "canonical_name", "merged_aliases"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"{len(feats)} school entries -> {len(new_feats)} unique campuses "
          f"({len(report_rows)} coordinates had duplicates merged).")
    print(f"Review the merges in: {REPORT}")
    print("Now rebuild: python src/build_dataset.py && python src/build_buffer_table.py")


if __name__ == "__main__":
    main()
