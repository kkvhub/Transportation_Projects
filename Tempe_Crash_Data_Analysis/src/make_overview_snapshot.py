"""
make_overview_snapshot.py
=========================
Build the STATIC landing image: all crashes drawn over a light road-network
basemap so it actually reads like a map (not floating dots on white).

WHY STATIC
----------
The interactive 56k-point map froze the browser on first load. The landing only
needs a quick visual, so we pre-render a small PNG (~a few hundred KB) shown
instantly with st.image(). Interactive maps are still produced for each question.

LAYERS (bottom -> top)
----------------------
1. light page background,
2. the Tempe-area road network (thin grey lines) from adot_roads.json -> gives
   the "map" feel and context for where crashes sit,
3. ordinary crashes (small, faint blue),
4. KSI / serious-fatal crashes (red, on top) so danger spots pop.

OUTPUT
------
data/overview_all_crashes.png

Run:  python src/make_overview_snapshot.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import query_engine as qe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROADS = DATA / "adot_roads.json"
OUT = DATA / "overview_all_crashes.png"

KSI_COLOR = "#d62728"
OTHER_COLOR = "#2a6fd6"
BG = "#eef1f4"           # light "land" background
ROAD_COLOR = "#b9c2cc"   # grey roads


def _road_segments(bbox):
    """Return a list of [(x,y),...] line segments for roads inside the bbox.

    Handles both LineString and MultiLineString geometries. We clip to the crash
    bounding box (with a small margin) so the basemap matches the data extent.
    """
    if not ROADS.exists():
        return []
    minx, miny, maxx, maxy = bbox
    segs = []

    def keep(line):
        # keep a line if any vertex falls within the (slightly padded) bbox
        return any(minx <= x <= maxx and miny <= y <= maxy for x, y in line)

    for feat in json.load(open(ROADS))["features"]:
        geom = feat["geometry"]
        lines = ([geom["coordinates"]] if geom["type"] == "LineString"
                 else geom["coordinates"])
        for line in lines:
            pts = [(c[0], c[1]) for c in line]
            if keep(pts):
                segs.append(pts)
    return segs


def main():
    df = qe.load_data()
    ksi = df[df["KSI"] == 1]
    other = df[df["KSI"] != 1]

    # Bounding box of the crashes (with a small margin) for clipping roads.
    margin = 0.01
    bbox = (df["Longitude"].min() - margin, df["Latitude"].min() - margin,
            df["Longitude"].max() + margin, df["Latitude"].max() + margin)

    fig, ax = plt.subplots(figsize=(9, 10), dpi=120)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor("white")

    # 1) roads basemap
    segs = _road_segments(bbox)
    if segs:
        ax.add_collection(LineCollection(segs, colors=ROAD_COLOR,
                                         linewidths=0.5, alpha=0.9, zorder=1))

    # 2) crashes
    ax.scatter(other["Longitude"], other["Latitude"], s=1.0,
               c=OTHER_COLOR, alpha=0.12, linewidths=0, zorder=2, label="Other")
    ax.scatter(ksi["Longitude"], ksi["Latitude"], s=6,
               c=KSI_COLOR, alpha=0.65, linewidths=0, zorder=3,
               label="KSI (serious/fatal)")

    # Frame to the crash extent; correct aspect for latitude.
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1.0 / math.cos(math.radians(df["Latitude"].mean())))
    ax.set_title(f"All Tempe crashes, 2012-2025  (n = {len(df):,})", fontsize=13)
    ax.axis("off")
    leg = ax.legend(loc="upper right", framealpha=0.95, markerscale=2, fontsize=9)
    leg.get_frame().set_edgecolor("#cccccc")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved overview snapshot -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB; "
          f"{len(segs)} road segments)")


if __name__ == "__main__":
    main()
