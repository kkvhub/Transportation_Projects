"""
gap_analysis.py
===============
Detects coverage gaps, scores them by priority, and recommends where to
place new sensors.

THREE OUTPUTS:
    1. Gap list   : every uncovered segment longer than the minimum threshold,
                    with start/end chainage, length, and geographic coordinates
    2. Priority scores: each gap ranked by urgency (length × traffic weight)
    3. Placement recommendations: the optimal lat/lon to deploy the next sensor
                                   in each gap (midpoint of the gap on the corridor)

PRIORITY SCORE FORMULA:
    score = (gap_length_m / max_gap_length_m) × traffic_multiplier × 100

    Normalized to 0–100+ so the audience sees an intuitive number, not an
    arbitrary unit. Traffic multiplier comes from the AADT at the nearest
    sensor (proxy for traffic demand in the gap zone).

    Score bands:
        ≥ 75  → CRITICAL  (red)
        50–74 → HIGH      (orange)
        25–49 → MEDIUM    (yellow)
        < 25  → LOW       (green)
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from corridor import unproject_point


# ---------------------------------------------------------------------------
# GAP DETECTION
# ---------------------------------------------------------------------------

def detect_gaps(covered_intervals, corridor_length_m, min_gap_m=50.0):
    """
    Find uncovered segments of the corridor.

    Args:
        covered_intervals : sorted, merged list of (start_m, end_m) from coverage.py
        corridor_length_m : total corridor length in meters
        min_gap_m         : gaps shorter than this are ignored (they may be
                            intentional overlaps or rounding artifacts)

    Returns:
        List of dicts, each representing one gap:
            start_m   : chainage of gap start
            end_m     : chainage of gap end
            length_m  : gap length in meters
            midpoint_m: chainage of gap midpoint (optimal new sensor location)

    HOW IT WORKS:
        The covered intervals cover some portions of [0, corridor_length].
        The gaps are everything else. We walk the number line from 0 to
        corridor_length and collect the uncovered segments.

        Example:
            corridor = 1000m
            covered  = [(0, 200), (400, 700)]
            gaps     = [(200, 400), (700, 1000)]  → lengths 200m and 300m
    """
    gaps = []

    # Start from the beginning of the corridor
    cursor = 0.0

    for cov_start, cov_end in covered_intervals:
        if cov_start > cursor:
            # There's an uncovered stretch before this coverage interval
            gap_length = cov_start - cursor
            if gap_length >= min_gap_m:
                gaps.append({
                    "start_m":    round(cursor, 1),
                    "end_m":      round(cov_start, 1),
                    "length_m":   round(gap_length, 1),
                    "midpoint_m": round((cursor + cov_start) / 2, 1),
                })
        # Advance cursor past the current coverage interval
        cursor = max(cursor, cov_end)

    # Check for a gap after the last sensor (tail of corridor)
    if cursor < corridor_length_m:
        gap_length = corridor_length_m - cursor
        if gap_length >= min_gap_m:
            gaps.append({
                "start_m":    round(cursor, 1),
                "end_m":      round(corridor_length_m, 1),
                "length_m":   round(gap_length, 1),
                "midpoint_m": round((cursor + corridor_length_m) / 2, 1),
            })

    return gaps


# ---------------------------------------------------------------------------
# TRAFFIC WEIGHT FOR GAPS
# ---------------------------------------------------------------------------

def assign_gap_traffic_weight(gap_start_m, gap_end_m, sensor_details):
    """
    Estimate the traffic demand in a gap zone by finding the nearest sensor
    that has AADT data and using its traffic weight as a proxy.

    If no sensors have AADT data, returns the "Unknown" weight (1.1).

    This is an intentional simplification — in a real deployment, you would
    use spatially interpolated AADT values from a statewide traffic model.
    """
    from sample_data import get_traffic_weight, TRAFFIC_WEIGHTS

    gap_mid = (gap_start_m + gap_end_m) / 2

    best_weight   = TRAFFIC_WEIGHTS["Unknown"]
    best_distance = float("inf")

    for sensor in sensor_details:
        iv = sensor.get("interval")
        if iv is None:
            continue
        sensor_mid   = (iv[0] + iv[1]) / 2
        dist         = abs(sensor_mid - gap_mid)
        aadt         = sensor.get("aadt", None)

        # Only use sensors that have real AADT data
        try:
            aadt_val = float(aadt)
            if aadt_val > 0 and dist < best_distance:
                best_distance = dist
                best_weight   = get_traffic_weight(aadt_val)
        except (TypeError, ValueError):
            pass

    return best_weight


# ---------------------------------------------------------------------------
# PRIORITY SCORING
# ---------------------------------------------------------------------------

def score_gaps(gaps, sensor_details, corridor_length_m):
    """
    Add priority scores to each gap.

    Args:
        gaps              : list of gap dicts from detect_gaps()
        sensor_details    : list of sensor dicts from coverage.py
        corridor_length_m : total corridor length (used for normalization)

    Returns:
        Same list of dicts with added fields:
            traffic_weight  : float multiplier
            raw_score       : length × traffic_weight
            priority_score  : normalized 0–100+ integer
            priority_band   : "CRITICAL" / "HIGH" / "MEDIUM" / "LOW"
    """
    if not gaps:
        return gaps

    # Compute raw scores = length × traffic weight
    for gap in gaps:
        tw = assign_gap_traffic_weight(
            gap["start_m"], gap["end_m"], sensor_details
        )
        gap["traffic_weight"] = round(tw, 2)
        gap["raw_score"]      = round(gap["length_m"] * tw, 1)

    # Normalize: the worst gap (highest raw score) gets score 100
    max_raw = max(g["raw_score"] for g in gaps)

    for gap in gaps:
        if max_raw > 0:
            normalized = (gap["raw_score"] / max_raw) * 100
        else:
            normalized = 0.0
        gap["priority_score"] = round(normalized, 1)
        gap["priority_band"]  = _score_to_band(normalized)

    # Sort by priority score descending (most urgent first)
    gaps.sort(key=lambda g: g["priority_score"], reverse=True)

    # Add rank
    for i, gap in enumerate(gaps):
        gap["rank"] = i + 1

    return gaps


def _score_to_band(score):
    """Convert a 0–100 priority score to a text band label."""
    if score >= 75:
        return "CRITICAL"
    elif score >= 50:
        return "HIGH"
    elif score >= 25:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# OPTIMAL SENSOR PLACEMENT
# ---------------------------------------------------------------------------

def compute_placements(gaps, corridor_utm, utm_crs, top_n=5):
    """
    For each gap, compute the optimal lat/lon to place a new sensor.

    Strategy: midpoint of the gap along the corridor centerline.
    This minimizes the maximum uncovered distance within the gap.

    Args:
        gaps         : scored gap list from score_gaps()
        corridor_utm : Shapely LineString in UTM
        utm_crs      : UTM EPSG string
        top_n        : return only the top N placements (ranked by priority)

    Returns:
        List of dicts with:
            rank         : same rank as the gap
            gap_length_m : length of the gap being addressed
            priority_band: urgency label
            lat, lon     : recommended sensor location
            chainage_m   : distance from corridor start to placement
    """
    placements = []

    for gap in gaps[:top_n]:
        midpoint_m = gap["midpoint_m"]

        # interpolate() returns the point at a given distance along the LineString
        point_utm = corridor_utm.interpolate(midpoint_m)

        # Convert back to WGS84 for Folium
        lat, lon = unproject_point(point_utm.x, point_utm.y, utm_crs)

        placements.append({
            "rank":          gap["rank"],
            "gap_length_m":  gap["length_m"],
            "priority_band": gap["priority_band"],
            "priority_score":gap["priority_score"],
            "lat":           round(lat, 6),
            "lon":           round(lon, 6),
            "chainage_m":    round(midpoint_m, 1),
        })

    return placements


# ---------------------------------------------------------------------------
# CHAINAGE → GEOGRAPHIC COORDINATE (for segment endpoints on the map)
# ---------------------------------------------------------------------------

def chainage_to_latlon(chainage_m, corridor_utm, utm_crs):
    """
    Convert a chainage value (meters from corridor start) to a (lat, lon).
    Used to draw gap segments and coverage segments on the Folium map.
    """
    point_utm = corridor_utm.interpolate(chainage_m)
    lat, lon  = unproject_point(point_utm.x, point_utm.y, utm_crs)
    return lat, lon


def gap_to_latlon_segment(gap, corridor_utm, utm_crs, n_points=20):
    """
    Convert a gap's (start_m, end_m) interval to a list of (lat, lon) points
    tracing the gap along the corridor centerline.

    Using n_points > 2 means the gap segment follows the road curvature
    rather than drawing a straight line across a curve.
    """
    chainages = np.linspace(gap["start_m"], gap["end_m"], n_points)
    coords    = []
    for ch in chainages:
        lat, lon = chainage_to_latlon(ch, corridor_utm, utm_crs)
        coords.append((lat, lon))
    return coords


def coverage_to_latlon_segment(start_m, end_m, corridor_utm, utm_crs, n_points=20):
    """
    Same as gap_to_latlon_segment but for covered intervals.
    Returns a list of (lat, lon) points tracing the covered segment.
    """
    chainages = np.linspace(start_m, end_m, n_points)
    coords    = []
    for ch in chainages:
        lat, lon = chainage_to_latlon(ch, corridor_utm, utm_crs)
        coords.append((lat, lon))
    return coords


# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------

def compute_summary_stats(gaps, coverage_result):
    """
    Compute top-level statistics for the coverage statistics panel.

    Returns a dict of display-ready metrics.
    """
    corridor_length_m = coverage_result["corridor_length_m"]
    coverage_pct      = coverage_result["coverage_pct"]
    covered_intervals = coverage_result["covered_intervals"]

    covered_length = sum(e - s for s, e in covered_intervals)
    gap_length     = corridor_length_m - covered_length

    return {
        "corridor_length_m":  round(corridor_length_m, 0),
        "covered_length_m":   round(covered_length, 0),
        "gap_length_m":       round(gap_length, 0),
        "coverage_pct":       coverage_pct,
        "gap_pct":            round(100 - coverage_pct, 1),
        "num_gaps":           len(gaps),
        "num_sensors":        len(coverage_result["sensor_details"]),
        "longest_gap_m":      round(max((g["length_m"] for g in gaps), default=0), 0),
        "critical_gaps":      sum(1 for g in gaps if g["priority_band"] == "CRITICAL"),
        "high_gaps":          sum(1 for g in gaps if g["priority_band"] == "HIGH"),
    }
