"""
sample_data.py
==============
Hard-coded demo dataset for the ITS America 2026 conference demo.

WHY THIS EXISTS:
    Conference WiFi is unreliable. This dataset lets the presenter click
    "Load Detroit Demo Data" and immediately show a fully-populated map
    without needing to upload a CSV or wait for an OSM fetch.

    Corridor: Woodward Avenue, Detroit — from Huntington Place (conference
    venue) northward to Midtown. A real, recognizable Detroit arterial.

    Sensors: 5 realistic ITS deployments with deliberate coverage gaps
    so the gap detection logic has something interesting to show.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# CORRIDOR WAYPOINTS
# A sequence of (lat, lon) pairs tracing Woodward Ave from the convention
# center area northward. These are used as the fallback corridor geometry
# when OSM fetch is unavailable, and as the default corridor input fields.
# ---------------------------------------------------------------------------

DEMO_CORRIDOR_WAYPOINTS = [
    (42.3294, -83.0458),   # Start: near Huntington Place / Hart Plaza area
    (42.3340, -83.0476),   # Woodward + Larned
    (42.3380, -83.0492),   # Woodward + Congress
    (42.3420, -83.0505),   # Woodward + Grand River
    (42.3470, -83.0519),   # Woodward + Temple
    (42.3520, -83.0530),   # Woodward + Canfield
    (42.3570, -83.0541),   # Woodward + Forest
    (42.3610, -83.0549),   # End: Woodward + West Warren (Midtown edge)
]

DEMO_START_ADDRESS = "Huntington Place, Detroit, MI"
DEMO_END_ADDRESS   = "Woodward Ave & West Warren Ave, Detroit, MI"

# ---------------------------------------------------------------------------
# SENSOR DATASET
# Each row is one deployed sensor on or near the Woodward corridor.
#
# Columns:
#   sensor_id        : unique label shown on map tooltips
#   sensor_type      : one of the four standard ITS types
#   lat, lon         : geographic position of the sensor
#   detection_range_m: override range in meters (if blank, type default used)
#   aadt             : Annual Average Daily Traffic at that location
#                      (used to weight gap priority scores)
#   notes            : optional context shown in map popup
#
# COORDINATE METHOD:
#   All sensor lat/lon values are computed by interpolating directly onto
#   the corridor centerline at specific chainage distances. This guarantees
#   every sensor sits exactly on Woodward Ave and its coverage interval
#   is geometrically accurate.
#
# GAP ENGINEERING:
#   Sensors at chainages 200m, 700m, 1800m, 2200m, 2700m on a 3594m corridor.
#   This creates a CRITICAL ~820m gap (700-1800m) and several MEDIUM gaps,
#   making the demo visually compelling and the ranking meaningful.
# ---------------------------------------------------------------------------

DEMO_SENSORS = pd.DataFrame([
    {
        "sensor_id":         "CAM-001",
        "sensor_type":       "Fixed Camera",
        "lat":               42.331129,
        "lon":               -83.046477,
        "detection_range_m": 80,
        "aadt":              28000,
        "notes":             "Existing MDOT camera, Woodward + Larned St"
    },
    {
        "sensor_id":         "RSU-001",
        "sensor_type":       "RSU / V2X",
        "lat":               42.335450,
        "lon":               -83.048180,
        "detection_range_m": 300,
        "aadt":              25000,
        "notes":             "V2X RSU at signalized intersection, Woodward + Congress"
    },
    {
        "sensor_id":         "BT-001",
        "sensor_type":       "Bluetooth / WiFi Probe",
        "lat":               42.345063,
        "lon":               -83.051358,
        "detection_range_m": 40,
        "aadt":              18000,
        "notes":             "Probe reader at transit stop, Woodward + Temple"
    },
    {
        "sensor_id":         "RAD-001",
        "sensor_type":       "Radar / Loop",
        "lat":               42.348602,
        "lon":               -83.052252,
        "detection_range_m": 50,
        "aadt":              18000,
        "notes":             "Radar detector mid-block, Woodward + Canfield"
    },
    {
        "sensor_id":         "CAM-002",
        "sensor_type":       "Fixed Camera",
        "lat":               42.353044,
        "lon":               -83.053230,
        "detection_range_m": 80,
        "aadt":              15000,
        "notes":             "City of Detroit intersection camera, Woodward + Forest"
    },
])

# ---------------------------------------------------------------------------
# SENSOR TYPE DEFAULTS
# Used when a sensor's detection_range_m is missing or zero.
# Values are the midpoint of realistic ITS deployment specs.
# ---------------------------------------------------------------------------

SENSOR_TYPE_DEFAULTS = {
    "Fixed Camera":          80,    # meters — covers one approach
    "Radar / Loop":          50,    # meters — point or short-range zone
    "Bluetooth / WiFi Probe":40,    # meters — passive probe reader radius
    "RSU / V2X":             400,   # meters — DSRC / C-V2X broadcast range
    "Lidar":                 150,   # meters — forward-facing roadside unit
}

# Default range if the sensor type is not in the lookup above
SENSOR_TYPE_DEFAULT_FALLBACK = 100   # meters

# ---------------------------------------------------------------------------
# TRAFFIC WEIGHT MULTIPLIERS
# Applied to gap priority scores. Higher traffic = higher urgency to fill gap.
# Thresholds are based on US urban arterial AADT classifications.
# ---------------------------------------------------------------------------

TRAFFIC_WEIGHTS = {
    "High":    1.5,   # AADT > 20,000
    "Medium":  1.2,   # AADT 5,000 – 20,000
    "Low":     1.0,   # AADT < 5,000
    "Unknown": 1.1,   # No AADT data provided — slight uplift to flag for review
}

AADT_THRESHOLDS = {
    "High":   20000,
    "Medium":  5000,
}

def get_traffic_class(aadt):
    """
    Convert a numeric AADT value to a traffic class string.
    Returns 'Unknown' if aadt is None, NaN, or zero.
    """
    try:
        aadt = float(aadt)
        if aadt > AADT_THRESHOLDS["High"]:
            return "High"
        elif aadt > AADT_THRESHOLDS["Medium"]:
            return "Medium"
        else:
            return "Low"
    except (TypeError, ValueError):
        return "Unknown"

def get_traffic_weight(aadt):
    """Return the numeric priority multiplier for a given AADT value."""
    return TRAFFIC_WEIGHTS[get_traffic_class(aadt)]
