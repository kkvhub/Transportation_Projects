"""
map_render.py
=============
PHASE 2 (revised in Phase 5): Turn any query result into an interactive map.

WHAT THIS DOES (plain English)
------------------------------
Takes a query result's "rows" (each crash has Latitude/Longitude) and draws every
crash as its OWN point on a Leaflet map -- NO clustering, NO aggregation. Points
stay individually visible at every zoom level, so you can see the exact road
stretches where crashes concentrate. Produces a standalone .html file the UI
embeds.

WHY GeoJSON + CANVAS (instead of clustered markers)
---------------------------------------------------
Earlier we used MarkerCluster, which merges nearby points into a numbered bubble
that re-bins as you zoom -- the user asked to remove that. Plotting tens of
thousands of individual folium markers as separate objects would be huge and slow.
Instead we:
  * pack all points into ONE GeoJSON layer (compact: just coordinates + a few
    props), and
  * render with a Canvas renderer (folium.Map(prefer_canvas=True)), which draws
    thousands of dots fast on a single canvas instead of thousands of DOM nodes.
This shows ALL crashes at ALL zoom levels with good performance.

Colour: red = KSI (serious/fatal), blue = other. Optional school marker + buffer
circle for school questions.

Public function:
    render_result(result, out_html, title=..., school=None, radius_mi=None)

Run a demo:  python src/map_render.py
"""

import json
from pathlib import Path
import folium

import query_engine as qe
import buffer_analysis as ba

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT.parent / "outputs"
SCHOOLS_JSON = DATA / "schools_tempe_clean.json"

MILE_TO_M = 1609.344
KSI_COLOR = "#d62728"      # red
OTHER_COLOR = "#2a6fd6"    # blue
# Hard safety cap so a runaway render can't produce a gigantic file. The full
# dataset (~56k) is well under this, so in practice ALL crashes are shown.
MAX_POINTS = 3500


def _school_coords(name):
    """Look up a school's (lat, lon) by exact name from the verified school file."""
    for f in json.load(open(SCHOOLS_JSON))["features"]:
        if f["properties"].get("SchoolName", "") == name:
            lon, lat = f["geometry"]["coordinates"]
            return lat, lon
    return None


# Above this many points we drop per-point popups and extra properties to keep
# the embedded HTML small/fast (e.g. the ~56k "all crashes" landing overview).
LIGHT_THRESHOLD = 20000


def _rows_to_geojson(rows, light=False):
    """Convert crash rows to a compact GeoJSON FeatureCollection.

    Coordinates are rounded to 5 decimals (~1 m) to shrink the file. When
    `light` is True (very large sets) we store ONLY the severity flag, which
    keeps even ~56k points to a manageable size.
    """
    feats = []
    for r in rows.itertuples(index=False):
        props = {"ksi": int(getattr(r, "KSI", 0) == 1)}
        if not light:
            props.update({
                "Year": getattr(r, "Year", ""),
                "Severity": getattr(r, "Injuryseverity", ""),
                "Street": getattr(r, "StreetName", ""),
                "School": getattr(r, "nearest_school", ""),
            })
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(float(r.Longitude), 5),
                                         round(float(r.Latitude), 5)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def render_result(result, out_html, title="Crash map", school=None,
                  radius_mi=None, max_points=MAX_POINTS):
    """Render a query result's crash rows to an interactive HTML map.

    result    : dict returned by any query function (needs result["rows"]).
    out_html  : path to write the .html file to.
    title     : heading shown on the map.
    school    : optional school NAME to mark + draw a buffer circle around.
    radius_mi : optional buffer radius (miles) for that circle.
    max_points: safety cap on plotted points (default 60k -> covers all crashes).
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    rows = result.get("rows")
    if rows is None or len(rows) == 0:
        # Nothing to map -> a small notice so the UI always has a file to show.
        m = folium.Map(location=[33.414, -111.926], zoom_start=12)
        folium.Marker([33.414, -111.926],
                      popup="No mappable results for this question.").add_to(m)
        m.save(str(out_html))
        return str(out_html)

    total = len(rows)
    sampled = total > max_points
    if sampled:
        rows = rows.sample(max_points, random_state=42)   # representative subset
    center = [rows["Latitude"].mean(), rows["Longitude"].mean()]

    # prefer_canvas=True -> all points drawn on one fast canvas layer.
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap",
                   prefer_canvas=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)

    # One GeoJSON layer with every crash as an individual circle (no clustering).
    # style_function colours each point by severity; the small radius keeps dense
    # corridors readable while still showing each crash.
    light = len(rows) > LIGHT_THRESHOLD
    gj = _rows_to_geojson(rows, light=light)
    popup = None if light else folium.GeoJsonPopup(
        fields=["Year", "Severity", "Street", "School"],
        aliases=["Year", "Severity", "Street", "Nearest school"],
    )
    folium.GeoJson(
        gj,
        name="Crashes",
        marker=folium.CircleMarker(radius=3, weight=0, fill=True, fill_opacity=0.65),
        style_function=lambda feat: {
            "fillColor": KSI_COLOR if feat["properties"]["ksi"] else OTHER_COLOR,
            "color": KSI_COLOR if feat["properties"]["ksi"] else OTHER_COLOR,
            "fillOpacity": 0.65,
        },
        popup=popup,
    ).add_to(m)

    # Optional school marker + buffer circle.
    if school:
        coords = _school_coords(school)
        if coords:
            folium.Marker(coords, tooltip=school,
                          icon=folium.Icon(color="green", icon="education",
                                           prefix="glyphicon")).add_to(m)
            if radius_mi:
                folium.Circle(coords, radius=radius_mi * MILE_TO_M,
                              color="green", fill=True, fill_opacity=0.08,
                              popup=f"{radius_mi} mi buffer").add_to(m)

    note = f" (random sample of {max_points:,} shown)" if sampled else ""
    html = (f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
            f'background:#fff;padding:8px 14px;border:1px solid #999;'
            f'border-radius:6px;font:14px Segoe UI,Arial;box-shadow:0 2px 6px rgba(0,0,0,.2)">'
            f'<b>{title}</b><br>{total:,} crashes{note} '
            f'&nbsp;<span style="color:{KSI_COLOR}">●</span> KSI '
            f'<span style="color:{OTHER_COLOR}">●</span> other</div>')
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=True).add_to(m)

    m.save(str(out_html))
    return str(out_html)


def _demo():
    OUTPUTS.mkdir(exist_ok=True)
    # 1) Landing-style overview: ALL crashes, individual points.
    out_all = render_result(qe.all_crashes(), OUTPUTS / "demo_map_all_crashes.html",
                            title="All Tempe crashes (2012-2025)")
    print("wrote", out_all)
    # 2) Buffer-view school example with the buffer circle.
    res = ba.school_ranking_buffer(0.25)
    top = res["meta"]["ranking"][0]["school_name"]
    out2 = render_result(res, OUTPUTS / "demo_map_school_buffer.html",
                         title=f"Crashes within 0.25 mi of {top}",
                         school=top, radius_mi=0.25)
    print("wrote", out2)


if __name__ == "__main__":
    _demo()
