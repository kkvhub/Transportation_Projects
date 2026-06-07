# ITS Sensor Coverage Gap Analyzer

A Streamlit application for auditing ITS sensor network coverage on road corridors.

## What it does

Given a set of existing sensor locations and their detection ranges, this tool:
- Fetches road centerline geometry from OpenStreetMap (with straight-line fallback)
- Models each sensor's coverage as a buffer projected onto the corridor centerline
- Identifies uncovered gaps and ranks them by priority (length × traffic demand)
- Recommends optimal locations for new sensor deployments
- Outputs an interactive Folium map with color-coded coverage and gap segments

## Live demo

[Deploy link — add after Streamlit Cloud deployment]

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **"Load Detroit Demo Data"** for an instant example on Woodward Ave, Detroit.

## Input CSV format

| Column | Required | Description |
|---|---|---|
| sensor_id | Yes | Unique label |
| sensor_type | Yes | Fixed Camera / Radar / Loop / Bluetooth / WiFi Probe / RSU / V2X / Lidar |
| lat | Yes | Decimal degrees |
| lon | Yes | Decimal degrees |
| detection_range_m | No | Override range in meters (uses type default if blank) |
| aadt | No | Annual Average Daily Traffic for priority weighting |
| notes | No | Displayed in map popup |

## Architecture

```
app.py           — Streamlit UI and pipeline orchestration
sample_data.py   — Demo dataset and constants
corridor.py      — OSM road geometry fetch + straight-line fallback
coverage.py      — Sensor buffer projection and interval arithmetic
gap_analysis.py  — Gap detection, priority scoring, placement recommendations
map_builder.py   — Folium interactive map construction
```

## Known limitations

- Coverage modeled as Euclidean buffers (not RF propagation or line-of-sight)
- AADT is user-supplied; no real-time traffic data integration
- Single corridor only; no network-wide analysis
- Corridor limited to ~10km for OSM fetch performance

## Part of

[kkvhub.github.io](https://kkvhub.github.io) — Transportation Analytics Portfolio
