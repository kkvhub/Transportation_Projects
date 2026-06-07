"""
map_builder.py
==============
Builds the Folium interactive map — the visual centerpiece of this tool.

MAP LAYERS (in draw order, bottom to top):
    1. Base tile layer     : CartoDB Positron (clean, minimal, works offline if cached)
    2. Full corridor       : thin grey polyline showing the complete road
    3. Covered segments    : thick GREEN polylines on covered corridor sections
    4. Gap segments        : thick RED polylines on uncovered sections
    5. Sensor circles      : semi-transparent colored circles showing detection zones
    6. Sensor markers      : colored icons at sensor locations with tooltips
    7. Placement markers   : yellow star markers at recommended new sensor positions

COLOR SCHEME:
    Covered segments   : #2ecc71  (green)
    Gap segments       : #e74c3c  (red)    — thick & prominent
    Sensor markers     : color by sensor type (see SENSOR_COLORS)
    Placement markers  : #f1c40f  (gold/yellow) with star icon

INTERACTION:
    Every marker and segment has a popup (click) and tooltip (hover).
    Layer control lets the audience toggle sensor types on/off.
    The map auto-fits to the corridor extent.
"""

import folium
from folium.plugins import MiniMap, MeasureControl
from gap_analysis import (
    gap_to_latlon_segment,
    coverage_to_latlon_segment,
    chainage_to_latlon,
)


# ---------------------------------------------------------------------------
# COLOR CONSTANTS
# ---------------------------------------------------------------------------

SENSOR_COLORS = {
    "Fixed Camera":           "#3498db",   # blue
    "Radar / Loop":           "#9b59b6",   # purple
    "Bluetooth / WiFi Probe": "#1abc9c",   # teal
    "RSU / V2X":              "#e67e22",   # orange
    "Lidar":                  "#2980b9",   # dark blue
    "Unknown":                "#95a5a6",   # grey
}

PRIORITY_COLORS = {
    "CRITICAL": "#c0392b",   # dark red
    "HIGH":     "#e67e22",   # orange
    "MEDIUM":   "#f39c12",   # amber
    "LOW":      "#27ae60",   # green
}

COVERED_COLOR  = "#2ecc71"   # green
GAP_COLOR      = "#e74c3c"   # red
CORRIDOR_COLOR = "#bdc3c7"   # light grey (full corridor underlay)
PLACEMENT_COLOR = "#f1c40f"  # gold


# ---------------------------------------------------------------------------
# SENSOR TYPE ICON MAPPING
# ---------------------------------------------------------------------------

SENSOR_ICONS = {
    "Fixed Camera":           "camera",
    "Radar / Loop":           "signal",
    "Bluetooth / WiFi Probe": "wifi",
    "RSU / V2X":              "broadcast-tower",
    "Lidar":                  "eye",
}


# ---------------------------------------------------------------------------
# HELPER: Folium CircleMarker HTML popup
# ---------------------------------------------------------------------------

def _sensor_popup_html(s):
    """Build a styled HTML popup for a sensor marker."""
    color  = SENSOR_COLORS.get(s.get("sensor_type", "Unknown"), "#95a5a6")
    iv     = s.get("interval")
    if iv:
        coverage_str = f"{iv[0]:.0f}m – {iv[1]:.0f}m ({iv[1]-iv[0]:.0f}m covered)"
    else:
        coverage_str = "⚠️ Not on corridor"

    aadt = s.get("aadt", "N/A")
    try:
        aadt = f"{int(float(aadt)):,}"
    except (TypeError, ValueError):
        aadt = "N/A"

    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; min-width: 200px;">
        <div style="background:{color}; color:white; padding:6px 10px;
                    border-radius:4px 4px 0 0; font-weight:bold; font-size:13px;">
            {s.get('sensor_id', 'Sensor')}
        </div>
        <div style="padding:8px 10px; border:1px solid #ddd; border-radius:0 0 4px 4px;">
            <b>Type:</b> {s.get('sensor_type', 'Unknown')}<br>
            <b>Range:</b> {s.get('effective_range_m', '?')}m<br>
            <b>AADT:</b> {aadt}<br>
            <b>Traffic:</b> {s.get('traffic_class', 'Unknown')}<br>
            <b>Coverage:</b> {coverage_str}<br>
            {f"<b>Notes:</b> {s.get('notes', '')}" if s.get('notes') else ""}
        </div>
    </div>
    """
    return html


def _gap_popup_html(gap):
    """Build a styled HTML popup for a gap segment."""
    band_color = PRIORITY_COLORS.get(gap["priority_band"], "#e74c3c")
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; min-width: 220px;">
        <div style="background:{band_color}; color:white; padding:6px 10px;
                    border-radius:4px 4px 0 0; font-weight:bold; font-size:13px;">
            Gap #{gap['rank']} — {gap['priority_band']}
        </div>
        <div style="padding:8px 10px; border:1px solid #ddd; border-radius:0 0 4px 4px;">
            <b>Length:</b> {gap['length_m']:.0f} m<br>
            <b>Chainage:</b> {gap['start_m']:.0f}m – {gap['end_m']:.0f}m<br>
            <b>Priority Score:</b> {gap['priority_score']:.1f} / 100<br>
            <b>Traffic Weight:</b> {gap['traffic_weight']:.2f}x<br>
        </div>
    </div>
    """
    return html


def _placement_popup_html(p):
    """Build a styled HTML popup for a new sensor placement recommendation."""
    band_color = PRIORITY_COLORS.get(p["priority_band"], "#f1c40f")
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; min-width: 220px;">
        <div style="background:#f1c40f; color:#333; padding:6px 10px;
                    border-radius:4px 4px 0 0; font-weight:bold; font-size:13px;">
            ⭐ Recommended Sensor #{p['rank']}
        </div>
        <div style="padding:8px 10px; border:1px solid #ddd; border-radius:0 0 4px 4px;">
            <b>Addresses Gap:</b> {p['gap_length_m']:.0f}m ({p['priority_band']})<br>
            <b>Priority Score:</b> {p['priority_score']:.1f} / 100<br>
            <b>Chainage:</b> {p['chainage_m']:.0f}m from start<br>
            <b>Coordinates:</b> {p['lat']:.5f}, {p['lon']:.5f}<br>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# MAIN MAP BUILDER
# ---------------------------------------------------------------------------

def build_map(corridor_result, coverage_result, gaps, placements):
    """
    Assemble the full Folium interactive map.

    Args:
        corridor_result : dict from corridor.get_corridor()
        coverage_result : dict from coverage.compute_coverage()
        gaps            : scored gap list from gap_analysis.score_gaps()
        placements      : placement list from gap_analysis.compute_placements()

    Returns:
        A folium.Map object (call .get_root().render() or ._repr_html_()
        to get the HTML string for display in Streamlit)
    """
    line_wgs84        = corridor_result["line_wgs84"]
    corridor_utm      = corridor_result["line_utm"]
    utm_crs           = corridor_result["utm_crs"]
    covered_intervals = coverage_result["covered_intervals"]
    sensor_details    = coverage_result["sensor_details"]

    # ---- Map initialization ----
    # Center on the corridor midpoint
    mid_lat = sum(c[1] for c in line_wgs84.coords) / len(list(line_wgs84.coords))
    mid_lon = sum(c[0] for c in line_wgs84.coords) / len(list(line_wgs84.coords))

    m = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=14,
        tiles="CartoDB positron",   # clean, minimal basemap
        prefer_canvas=True,
    )

    # ---- Layer groups (so user can toggle them in the layer control) ----
    lg_corridor  = folium.FeatureGroup(name="Corridor Centerline", show=True)
    lg_covered   = folium.FeatureGroup(name="✅ Covered Segments",  show=True)
    lg_gaps      = folium.FeatureGroup(name="🔴 Coverage Gaps",     show=True)
    lg_sensors   = folium.FeatureGroup(name="📡 Sensors",           show=True)
    lg_buffers   = folium.FeatureGroup(name="🔵 Detection Zones",   show=True)
    lg_placement = folium.FeatureGroup(name="⭐ Recommended Positions", show=True)

    # ---- Layer 1: Full corridor underlay (thin grey) ----
    corridor_coords_latlon = [(c[1], c[0]) for c in line_wgs84.coords]
    folium.PolyLine(
        locations=corridor_coords_latlon,
        color=CORRIDOR_COLOR,
        weight=3,
        opacity=0.6,
        tooltip="Road corridor",
    ).add_to(lg_corridor)

    # ---- Layer 2: Covered segments (thick green) ----
    for start_m, end_m in covered_intervals:
        coords = coverage_to_latlon_segment(
            start_m, end_m, corridor_utm, utm_crs, n_points=30
        )
        folium.PolyLine(
            locations=coords,
            color=COVERED_COLOR,
            weight=7,
            opacity=0.75,
            tooltip=f"✅ Covered: {start_m:.0f}m – {end_m:.0f}m ({end_m-start_m:.0f}m)",
        ).add_to(lg_covered)

    # ---- Layer 3: Gap segments (thick red, prominent) ----
    for gap in gaps:
        coords = gap_to_latlon_segment(gap, corridor_utm, utm_crs, n_points=30)
        band_color = PRIORITY_COLORS.get(gap["priority_band"], GAP_COLOR)

        folium.PolyLine(
            locations=coords,
            color=band_color,
            weight=9,       # thicker than covered segments — stands out
            opacity=0.85,
            tooltip=f"🔴 Gap #{gap['rank']} — {gap['length_m']:.0f}m — {gap['priority_band']}",
            popup=folium.Popup(
                folium.IFrame(_gap_popup_html(gap), width=260, height=180),
                max_width=260,
            ),
        ).add_to(lg_gaps)

    # ---- Layer 4: Sensor detection zones (semi-transparent circles) ----
    for s in sensor_details:
        sensor_type = s.get("sensor_type", "Unknown")
        color       = SENSOR_COLORS.get(sensor_type, "#95a5a6")
        lat         = float(s["lat"])
        lon         = float(s["lon"])
        radius_m    = s.get("effective_range_m", 100)

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.12,
            weight=1.5,
            tooltip=f"📡 {s.get('sensor_id','?')} — {sensor_type} — {radius_m}m range",
        ).add_to(lg_buffers)

    # ---- Layer 5: Sensor markers (colored icons) ----
    for s in sensor_details:
        sensor_type = s.get("sensor_type", "Unknown")
        color       = SENSOR_COLORS.get(sensor_type, "grey")
        lat         = float(s["lat"])
        lon         = float(s["lon"])

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"📡 {s.get('sensor_id','?')} ({sensor_type})",
            popup=folium.Popup(
                folium.IFrame(_sensor_popup_html(s), width=260, height=200),
                max_width=260,
            ),
        ).add_to(lg_sensors)

    # ---- Layer 6: Recommended new sensor placements (gold stars) ----
    for p in placements:
        # Gold star marker using a DivIcon so we can use emoji without extra deps
        star_icon = folium.DivIcon(
            html=f"""
            <div style="font-size:22px; text-align:center;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
                        margin-top:-10px; margin-left:-10px;">⭐</div>
            """,
            icon_size=(20, 20),
            icon_anchor=(10, 10),
        )
        folium.Marker(
            location=[p["lat"], p["lon"]],
            icon=star_icon,
            tooltip=f"⭐ Recommended #{p['rank']} — {p['gap_length_m']:.0f}m gap — {p['priority_band']}",
            popup=folium.Popup(
                folium.IFrame(_placement_popup_html(p), width=260, height=170),
                max_width=260,
            ),
        ).add_to(lg_placement)

    # ---- Add all layer groups to the map ----
    lg_corridor.add_to(m)
    lg_covered.add_to(m)
    lg_gaps.add_to(m)
    lg_buffers.add_to(m)
    lg_sensors.add_to(m)
    lg_placement.add_to(m)

    # ---- Plugins ----
    # MiniMap: small overview map in the corner — helpful for orientation
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)

    # MeasureControl: lets the audience measure distances on the map
    MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
    ).add_to(m)

    # Layer control — toggle layers on/off
    folium.LayerControl(collapsed=False).add_to(m)

    # ---- Fit map to corridor bounds ----
    lats = [c[1] for c in line_wgs84.coords]
    lons = [c[0] for c in line_wgs84.coords]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return m


# ---------------------------------------------------------------------------
# LEGEND HTML (injected into the map as a floating div)
# ---------------------------------------------------------------------------

LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 30px; right: 10px;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
    z-index: 9999;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    min-width: 160px;
">
    <b style="font-size:13px;">Map Legend</b><br><br>
    <span style="color:#2ecc71;">━━━</span> Covered segment<br>
    <span style="color:#e74c3c;">━━━</span> Gap (CRITICAL)<br>
    <span style="color:#e67e22;">━━━</span> Gap (HIGH)<br>
    <span style="color:#f39c12;">━━━</span> Gap (MEDIUM)<br>
    <span style="color:#27ae60;">━━━</span> Gap (LOW)<br>
    <span style="color:#bdc3c7;">━━━</span> Corridor centerline<br>
    ⭐ Recommended position<br><br>
    <span style="color:#3498db;">●</span> Fixed Camera &nbsp;
    <span style="color:#9b59b6;">●</span> Radar<br>
    <span style="color:#1abc9c;">●</span> BT/WiFi &nbsp;
    <span style="color:#e67e22;">●</span> RSU/V2X
</div>
"""


def inject_legend(map_html):
    """
    Inject the floating legend div into the rendered map HTML.
    Called after m._repr_html_() to append the legend before </body>.
    """
    return map_html.replace("</body>", LEGEND_HTML + "</body>")
