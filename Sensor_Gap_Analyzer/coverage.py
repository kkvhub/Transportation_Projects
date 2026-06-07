"""
coverage.py
===========
Models sensor coverage on the road corridor.

THE CORE IDEA:
    Every sensor has a detection range (radius in meters). We model this as
    a circular buffer around the sensor's location. Where that circle
    overlaps the road centerline, the road is "covered."

    Instead of working in 2D space (which gets complicated), we reduce the
    problem to 1D: we measure "chainage" — distance along the corridor from
    the start (0 meters) to the end (total length). Each sensor produces a
    contiguous interval [start_m, end_m] on this 1D number line.

    Coverage analysis then becomes simple interval arithmetic:
        - Covered  = union of all sensor intervals
        - Gaps     = complement of the covered union within [0, corridor_length]

KNOWN LIMITATION (stated in the UI):
    Buffers are Euclidean circles, not true signal-propagation zones.
    Real-world coverage depends on line-of-sight, terrain, buildings, and
    antenna characteristics. This is a planning-level tool, not an RF model.

PROJECTION:
    All geometry here operates in UTM (meters). Sensors given in WGS84
    (lat/lon) are projected to UTM before buffering.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point
import pyproj
from sample_data import (
    SENSOR_TYPE_DEFAULTS,
    SENSOR_TYPE_DEFAULT_FALLBACK,
    get_traffic_weight,
)


# ---------------------------------------------------------------------------
# SENSOR PROJECTION
# ---------------------------------------------------------------------------

def project_sensor_to_utm(lat, lon, utm_crs):
    """
    Convert a sensor's WGS84 (lat, lon) to UTM (x, y) in meters.
    Returns a Shapely Point in UTM coordinates.
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    utm   = pyproj.CRS(utm_crs)
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    x, y = transformer.transform(lon, lat)   # always_xy=True: lon first, then lat
    return Point(x, y)


# ---------------------------------------------------------------------------
# DETECTION RANGE RESOLUTION
# ---------------------------------------------------------------------------

def resolve_detection_range(row):
    """
    Determine the effective detection range for a sensor row.

    Priority order:
      1. Explicit detection_range_m column in the CSV (if > 0)
      2. Type-based default from SENSOR_TYPE_DEFAULTS lookup
      3. Global fallback (100m) if type is unrecognized
    """
    # Check for an explicit override in the data
    try:
        r = float(row.get("detection_range_m", 0))
        if r > 0:
            return r
    except (TypeError, ValueError):
        pass

    # Fall back to type default
    sensor_type = str(row.get("sensor_type", "")).strip()
    return SENSOR_TYPE_DEFAULTS.get(sensor_type, SENSOR_TYPE_DEFAULT_FALLBACK)


# ---------------------------------------------------------------------------
# SENSOR → CHAINAGE INTERVAL
# ---------------------------------------------------------------------------

def sensor_to_chainage_interval(sensor_point_utm, detection_range_m,
                                 corridor_utm):
    """
    Project a sensor's circular buffer onto the corridor centerline and
    return the covered interval as (start_m, end_m) chainage values.

    HOW IT WORKS:
        1. Build a circular buffer of radius = detection_range_m around
           the sensor point (in UTM meters).
        2. Intersect that circle with the corridor LineString.
        3. If the intersection is non-empty, find the chainage (distance
           along the corridor) of the closest and farthest intersection
           points. Those become the interval endpoints.

    Args:
        sensor_point_utm   : Shapely Point in UTM
        detection_range_m  : float, meters
        corridor_utm       : Shapely LineString in UTM

    Returns:
        (start_m, end_m) tuple, or None if sensor doesn't touch corridor
    """
    # Step 1: Build circular buffer
    buffer = sensor_point_utm.buffer(detection_range_m)

    # Step 2: Intersect buffer with corridor
    intersection = corridor_utm.intersection(buffer)

    if intersection.is_empty:
        return None   # Sensor is too far from corridor to cover any of it

    # Step 3: Find the chainage extent of the intersection
    # We sample points along the intersection and measure their distance
    # along the corridor from the start (project() returns chainage in meters)

    # Handle different intersection geometry types
    from shapely.geometry import MultiLineString, GeometryCollection, MultiPoint

    sample_points = []

    geom_type = intersection.geom_type
    if geom_type in ("LineString",):
        # Sample the start and end of each line segment
        coords = list(intersection.coords)
        sample_points.extend([Point(c) for c in coords])
    elif geom_type in ("MultiLineString",):
        for geom in intersection.geoms:
            coords = list(geom.coords)
            sample_points.extend([Point(c) for c in coords])
    elif geom_type == "Point":
        sample_points.append(intersection)
    elif geom_type == "MultiPoint":
        sample_points.extend(list(intersection.geoms))
    elif geom_type == "GeometryCollection":
        for geom in intersection.geoms:
            if not geom.is_empty:
                sample_points.append(geom.centroid)
    else:
        # Fallback: use the centroid
        sample_points.append(intersection.centroid)

    if not sample_points:
        return None

    # project() returns the distance along the corridor to the nearest point
    chainages = [corridor_utm.project(pt) for pt in sample_points]

    start_m = max(0.0, min(chainages) - detection_range_m * 0.1)
    end_m   = min(corridor_utm.length, max(chainages) + detection_range_m * 0.1)

    # Ensure valid interval
    if end_m <= start_m:
        return None

    return (start_m, end_m)


# ---------------------------------------------------------------------------
# INTERVAL UNION
# ---------------------------------------------------------------------------

def union_intervals(intervals):
    """
    Merge a list of (start, end) intervals into a non-overlapping sorted union.

    Example:
        [(0, 100), (80, 200), (300, 400)] → [(0, 200), (300, 400)]

    This is standard interval merging: sort by start, then sweep.
    """
    if not intervals:
        return []

    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]

    for current_start, current_end in sorted_ivs[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            # Overlapping or adjacent — extend the last interval
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap — start a new interval
            merged.append((current_start, current_end))

    return merged


# ---------------------------------------------------------------------------
# MAIN COVERAGE COMPUTATION
# ---------------------------------------------------------------------------

def compute_coverage(sensors_df, corridor_utm, utm_crs):
    """
    Compute coverage intervals for all sensors on the corridor.

    Args:
        sensors_df   : pandas DataFrame with columns:
                       sensor_id, sensor_type, lat, lon,
                       detection_range_m (optional), aadt (optional)
        corridor_utm : Shapely LineString in UTM
        utm_crs      : EPSG string of the UTM projection

    Returns dict with:
        sensor_details   : list of dicts, one per sensor, containing:
                           - all input fields
                           - effective_range_m
                           - point_utm (Shapely Point)
                           - interval (start_m, end_m) or None
                           - traffic_weight
                           - traffic_class
        covered_intervals: sorted merged list of (start_m, end_m)
        coverage_pct     : percentage of corridor length that is covered
        corridor_length_m: total corridor length
    """
    sensor_details    = []
    raw_intervals     = []
    corridor_length_m = corridor_utm.length

    for _, row in sensors_df.iterrows():
        row_dict = row.to_dict()

        # Resolve detection range
        effective_range = resolve_detection_range(row_dict)

        # Project sensor to UTM
        sensor_point_utm = project_sensor_to_utm(
            lat=float(row_dict["lat"]),
            lon=float(row_dict["lon"]),
            utm_crs=utm_crs
        )

        # Compute chainage interval
        interval = sensor_to_chainage_interval(
            sensor_point_utm, effective_range, corridor_utm
        )

        # Traffic weighting
        aadt            = row_dict.get("aadt", None)
        traffic_weight  = get_traffic_weight(aadt)
        from sample_data import get_traffic_class
        traffic_class   = get_traffic_class(aadt)

        sensor_details.append({
            **row_dict,
            "effective_range_m": effective_range,
            "point_utm":         sensor_point_utm,
            "interval":          interval,
            "traffic_weight":    traffic_weight,
            "traffic_class":     traffic_class,
        })

        if interval is not None:
            raw_intervals.append(interval)

    # Merge overlapping intervals into a clean union
    covered_intervals = union_intervals(raw_intervals)

    # Compute coverage percentage
    covered_length = sum(end - start for start, end in covered_intervals)
    coverage_pct   = (covered_length / corridor_length_m * 100
                      if corridor_length_m > 0 else 0.0)

    return {
        "sensor_details":    sensor_details,
        "covered_intervals": covered_intervals,
        "coverage_pct":      round(coverage_pct, 1),
        "corridor_length_m": round(corridor_length_m, 1),
    }
