"""
app.py
======
ITS Sensor Coverage Gap Analyzer
Streamlit application entry point.

This file wires together all the modules:
    sample_data.py  → demo dataset and constants
    corridor.py     → road centerline geometry (OSM or fallback)
    coverage.py     → sensor buffer projection and interval math
    gap_analysis.py → gap detection, scoring, and placement recommendations
    map_builder.py  → Folium interactive map

UI LAYOUT:
    Sidebar  : inputs (data source, corridor definition, parameters)
    Main     : three tabs
               Tab 1 — Interactive Map (Folium)
               Tab 2 — Gap Analysis Table
               Tab 3 — Coverage Statistics

CONFERENCE DEMO FLOW:
    1. Click "Load Detroit Demo Data" in sidebar
    2. Click "Run Analysis" button
    3. Map appears with green/red corridor and sensor markers
    4. Switch to Tab 2 to show ranked gap table
    5. Switch to Tab 3 to show coverage stats
    → Total time from open to wow-moment: ~15 seconds

DEPLOYMENT:
    Streamlit Community Cloud
    Entry point: app.py
    Requirements: requirements.txt
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Local modules
from sample_data import (
    DEMO_SENSORS,
    DEMO_CORRIDOR_WAYPOINTS,
    DEMO_START_ADDRESS,
    DEMO_END_ADDRESS,
    SENSOR_TYPE_DEFAULTS,
)
from corridor    import get_corridor
from coverage    import compute_coverage
from gap_analysis import (
    detect_gaps,
    score_gaps,
    compute_placements,
    compute_summary_stats,
)
from map_builder import build_map, inject_legend, PRIORITY_COLORS


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ITS Sensor Coverage Gap Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — dark engineering theme consistent with IRC compliance checker
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #16213e;
}
/* Headers */
h1, h2, h3 {
    color: #f39c12 !important;
    font-family: 'Segoe UI', sans-serif;
}
/* Metric boxes */
[data-testid="metric-container"] {
    background-color: #0f3460;
    border: 1px solid #f39c12;
    border-radius: 8px;
    padding: 12px;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #16213e;
}
.stTabs [data-baseweb="tab"] {
    color: #f39c12;
    font-weight: 600;
}
/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #0f3460;
}
/* Buttons */
.stButton > button {
    background-color: #f39c12;
    color: #1a1a2e;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 15px;
    width: 100%;
}
.stButton > button:hover {
    background-color: #e67e22;
    color: white;
}
/* Info / warning boxes */
.stAlert {
    border-radius: 6px;
}
/* Divider */
hr {
    border-color: #f39c12;
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
    <span style="font-size:36px;">📡</span>
    <h1 style="margin:4px 0; font-size:28px; letter-spacing:1px;">
        ITS SENSOR COVERAGE GAP ANALYZER
    </h1>
    <p style="color:#bdc3c7; font-size:14px; margin:0;">
        Identify coverage gaps · Prioritize deployments · Optimize sensor placement
    </p>
</div>
<hr>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR — INPUT PANEL
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ---- Data source ----
    st.markdown("### 📂 Sensor Data")
    data_source = st.radio(
        "Choose input method:",
        ["🏙️ Load Detroit Demo Data", "📤 Upload CSV"],
        index=0,
    )

    sensors_df = None

    if data_source == "🏙️ Load Detroit Demo Data":
        sensors_df = DEMO_SENSORS.copy()
        waypoints  = DEMO_CORRIDOR_WAYPOINTS
        st.success(f"✅ {len(sensors_df)} sensors loaded on Woodward Ave, Detroit")

        # Show the demo data so user can explain it
        with st.expander("Preview sensor data"):
            st.dataframe(
                sensors_df[["sensor_id", "sensor_type", "detection_range_m", "aadt"]],
                hide_index=True,
                use_container_width=True,
            )

    else:
        # CSV upload path
        st.markdown("""
        **Required columns:** `sensor_id`, `sensor_type`, `lat`, `lon`

        **Optional columns:** `detection_range_m`, `aadt`, `notes`

        **Sensor types:** Fixed Camera · Radar / Loop ·
        Bluetooth / WiFi Probe · RSU / V2X · Lidar
        """)

        uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])

        if uploaded is not None:
            try:
                sensors_df = pd.read_csv(uploaded)
                required   = ["sensor_id", "sensor_type", "lat", "lon"]
                missing    = [c for c in required if c not in sensors_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    sensors_df = None
                else:
                    st.success(f"✅ {len(sensors_df)} sensors loaded")
            except Exception as e:
                st.error(f"CSV read error: {e}")
                sensors_df = None

        # Corridor waypoints for uploaded data
        st.markdown("### 🗺️ Corridor Waypoints")
        st.markdown("Enter one waypoint per line as `lat,lon`:")
        waypoints_text = st.text_area(
            "Waypoints",
            value="\n".join(f"{lat},{lon}"
                            for lat, lon in DEMO_CORRIDOR_WAYPOINTS),
            height=140,
            label_visibility="collapsed",
        )
        waypoints = []
        for line in waypoints_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                lat_s, lon_s = line.split(",")
                waypoints.append((float(lat_s), float(lon_s)))
            except ValueError:
                st.warning(f"Could not parse: {line}")

    st.markdown("---")

    # ---- Analysis parameters ----
    st.markdown("### 🔧 Analysis Parameters")

    min_gap_m = st.slider(
        "Minimum gap to report (meters)",
        min_value=25,
        max_value=500,
        value=100,
        step=25,
        help="Gaps shorter than this threshold are ignored (likely intentional overlaps)",
    )

    top_n_placements = st.slider(
        "Number of placement recommendations",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    use_osm = st.checkbox(
        "Fetch road geometry from OpenStreetMap",
        value=True,
        help="Uncheck if you have no internet connection. Uses straight-line corridor instead.",
    )

    osm_timeout = st.slider(
        "OSM timeout (seconds)",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
        help="Increase if on slow WiFi. If fetch fails, straight-line fallback is used.",
    ) if use_osm else 10

    st.markdown("---")

    # ---- Run button ----
    run_analysis = st.button(
        "▶ Run Analysis",
        disabled=(sensors_df is None or len(waypoints) < 2),
        use_container_width=True,
    )

    if sensors_df is None:
        st.warning("Load demo data or upload a CSV to enable analysis.")
    elif len(waypoints) < 2:
        st.warning("Need at least 2 corridor waypoints.")

    # ---- CSV template download ----
    st.markdown("---")
    st.markdown("### 📥 CSV Template")
    template_df = pd.DataFrame([{
        "sensor_id": "CAM-001",
        "sensor_type": "Fixed Camera",
        "lat": 42.334,
        "lon": -83.047,
        "detection_range_m": 80,
        "aadt": 25000,
        "notes": "Example sensor",
    }])
    csv_bytes = template_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV Template",
        data=csv_bytes,
        file_name="sensor_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ---- About panel ----
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **ITS Sensor Coverage Gap Analyzer**

        Built for transportation agencies and ITS practitioners to:
        - Audit existing sensor network coverage
        - Identify and prioritize coverage gaps
        - Recommend optimal locations for new sensors

        **Coverage model:** Euclidean buffers projected onto road centerline
        *(not RF propagation — planning-level tool)*

        **Road geometry:** OpenStreetMap via osmnx, with straight-line fallback

        **Priority score:** Gap length × traffic demand weight, normalized 0–100

        *Portfolio tool — kkvhub.github.io*
        """)


# ---------------------------------------------------------------------------
# MAIN PANEL — RESULTS
# ---------------------------------------------------------------------------

if not run_analysis:
    # Landing state — show instructions and sample map image
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#7f8c8d;">
        <span style="font-size:64px;">🗺️</span>
        <h2 style="color:#7f8c8d; margin-top:16px;">
            Configure inputs in the sidebar and click <span style="color:#f39c12;">▶ Run Analysis</span>
        </h2>
        <p>The map will show your corridor with green covered segments,
           red gap segments, and gold star placement recommendations.</p>
        <p style="font-size:13px; margin-top:24px;">
            <b>Demo:</b> Select "Load Detroit Demo Data" for an instant example
            on Woodward Ave, Detroit (Huntington Place corridor).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show a quick explainer of the method
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📡 Step 1 — Sensor Input**
        Load the demo dataset or upload your own CSV with sensor locations,
        types, and optional AADT values.
        """)
    with col2:
        st.markdown("""
        **🗺️ Step 2 — Coverage Modeling**
        Each sensor's detection range is projected onto the road centerline.
        Covered and uncovered intervals are computed automatically.
        """)
    with col3:
        st.markdown("""
        **📊 Step 3 — Gap Analysis**
        Gaps are ranked by length × traffic demand. Optimal placement
        locations (midpoints of each gap) are shown as ⭐ markers.
        """)

else:
    # ---- RUN THE ANALYSIS PIPELINE ----
    with st.spinner("🔄 Fetching road geometry and computing coverage..."):

        # Step 1: Get corridor geometry
        corridor_result = get_corridor(
            waypoints=waypoints,
            use_osm=use_osm,
            timeout=osm_timeout,
        )

        # Step 2: Compute sensor coverage
        coverage_result = compute_coverage(
            sensors_df=sensors_df,
            corridor_utm=corridor_result["line_utm"],
            utm_crs=corridor_result["utm_crs"],
        )

        # Step 3: Detect gaps
        gaps = detect_gaps(
            covered_intervals=coverage_result["covered_intervals"],
            corridor_length_m=coverage_result["corridor_length_m"],
            min_gap_m=min_gap_m,
        )

        # Step 4: Score gaps and compute placements
        gaps       = score_gaps(gaps, coverage_result["sensor_details"],
                                coverage_result["corridor_length_m"])
        placements = compute_placements(gaps, corridor_result["line_utm"],
                                        corridor_result["utm_crs"],
                                        top_n=top_n_placements)

        # Step 5: Summary stats
        stats = compute_summary_stats(gaps, coverage_result)

        # Step 6: Build map
        folium_map = build_map(corridor_result, coverage_result, gaps, placements)
        map_html   = inject_legend(folium_map._repr_html_())

    # ---- Show OSM fallback warning if applicable ----
    if corridor_result.get("warning"):
        st.warning(corridor_result["warning"])

    # ---- Quick stats bar ----
    st.markdown("### 📊 Coverage Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Corridor Length",    f"{stats['corridor_length_m']:.0f} m")
    c2.metric("Coverage",           f"{stats['coverage_pct']}%",
              delta=f"{stats['coverage_pct']-100:.1f}% from full")
    c3.metric("Total Gap",          f"{stats['gap_length_m']:.0f} m")
    c4.metric("Gaps Detected",      stats["num_gaps"])
    c5.metric("Critical / High",
              f"{stats['critical_gaps']} / {stats['high_gaps']}",
              delta="⚠️ Priority" if stats["critical_gaps"] > 0 else None,
              delta_color="inverse")
    c6.metric("Longest Gap",        f"{stats['longest_gap_m']:.0f} m")

    st.markdown("---")

    # ---- Three tabs ----
    tab_map, tab_gaps, tab_stats = st.tabs([
        "🗺️  Interactive Map",
        "📋  Gap Analysis",
        "📈  Coverage Statistics",
    ])

    # ========== TAB 1: MAP ==========
    with tab_map:
        st.markdown(
            f"**Corridor source:** {'OpenStreetMap road network' if corridor_result['source'] == 'osm' else 'Straight-line fallback'}  "
            f"&nbsp;|&nbsp; **{stats['num_sensors']} sensors** &nbsp;|&nbsp; "
            f"**{stats['num_gaps']} gaps** detected &nbsp;|&nbsp; "
            f"**{len(placements)} placements** recommended"
        )

        # Render Folium map in an iframe-like component
        import streamlit.components.v1 as components
        components.html(map_html, height=580, scrolling=False)

        # Map download button
        st.download_button(
            label="⬇️ Download Map (HTML)",
            data=map_html.encode("utf-8"),
            file_name="sensor_coverage_map.html",
            mime="text/html",
            use_container_width=False,
        )

    # ========== TAB 2: GAP TABLE ==========
    with tab_gaps:
        if not gaps:
            st.success("✅ No coverage gaps detected above the minimum threshold. "
                       "Excellent sensor coverage!")
        else:
            st.markdown(f"#### {len(gaps)} gaps detected — sorted by priority")

            # Build display dataframe
            gap_rows = []
            for g in gaps:
                band  = g["priority_band"]
                color = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(band,"⚪")
                gap_rows.append({
                    "Rank":           g["rank"],
                    "Priority":       f"{color} {band}",
                    "Score":          g["priority_score"],
                    "Gap Length (m)": g["length_m"],
                    "Start (m)":      g["start_m"],
                    "End (m)":        g["end_m"],
                    "Traffic Weight": g["traffic_weight"],
                })
            gap_df = pd.DataFrame(gap_rows)
            st.dataframe(
                gap_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Priority Score",
                        help="0–100, higher = more urgent",
                        min_value=0,
                        max_value=100,
                    ),
                    "Gap Length (m)": st.column_config.NumberColumn(
                        format="%d m"
                    ),
                },
            )

            # Gap CSV download
            csv_out = gap_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Gap Report (CSV)",
                data=csv_out,
                file_name="coverage_gaps.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.markdown(f"#### ⭐ Recommended Sensor Placements (top {len(placements)})")

            placement_rows = []
            for p in placements:
                band  = p["priority_band"]
                color = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(band,"⚪")
                placement_rows.append({
                    "Rank":           p["rank"],
                    "Priority":       f"{color} {band}",
                    "Addresses Gap":  f"{p['gap_length_m']:.0f} m",
                    "Chainage (m)":   p["chainage_m"],
                    "Latitude":       p["lat"],
                    "Longitude":      p["lon"],
                })
            st.dataframe(
                pd.DataFrame(placement_rows),
                hide_index=True,
                use_container_width=True,
            )

    # ========== TAB 3: STATISTICS ==========
    with tab_stats:
        st.markdown("#### 📐 Corridor & Coverage Breakdown")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Length breakdown**")
            breakdown_df = pd.DataFrame([
                {"Segment": "✅ Covered",    "Length (m)": stats["covered_length_m"],
                 "Percent": stats["coverage_pct"]},
                {"Segment": "🔴 Gap",        "Length (m)": stats["gap_length_m"],
                 "Percent": stats["gap_pct"]},
                {"Segment": "📏 Total corridor","Length (m)": stats["corridor_length_m"],
                 "Percent": 100.0},
            ])
            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("**Gap priority breakdown**")
            if gaps:
                from collections import Counter
                band_counts = Counter(g["priority_band"] for g in gaps)
                band_df = pd.DataFrame([
                    {"Priority Band": band, "Count": count}
                    for band, count in [
                        ("CRITICAL", band_counts.get("CRITICAL", 0)),
                        ("HIGH",     band_counts.get("HIGH",     0)),
                        ("MEDIUM",   band_counts.get("MEDIUM",   0)),
                        ("LOW",      band_counts.get("LOW",      0)),
                    ]
                ])
                st.dataframe(band_df, hide_index=True, use_container_width=True)
            else:
                st.info("No gaps detected.")

        st.markdown("---")
        st.markdown("#### 📡 Sensor Inventory")

        sensor_rows = []
        for s in coverage_result["sensor_details"]:
            iv = s.get("interval")
            sensor_rows.append({
                "Sensor ID":    s.get("sensor_id", "?"),
                "Type":         s.get("sensor_type", "Unknown"),
                "Range (m)":    s.get("effective_range_m", "?"),
                "Traffic":      s.get("traffic_class", "Unknown"),
                "AADT":         s.get("aadt", "N/A"),
                "Coverage":     f"{iv[0]:.0f}–{iv[1]:.0f}m ({iv[1]-iv[0]:.0f}m)" if iv else "Off corridor",
                "On Corridor":  "✅" if iv else "⚠️ Off corridor",
            })
        st.dataframe(
            pd.DataFrame(sensor_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("""
        **⚠️ Known Limitations**
        - Coverage modeled as Euclidean buffers (not RF propagation or line-of-sight)
        - AADT values are user-supplied; no real-time traffic data integration
        - Corridor limited to ~10km for OSM fetch performance
        - Single road corridor only (no network-wide analysis)

        *This is a planning-level tool. Field verification is required before deployment decisions.*
        """)
