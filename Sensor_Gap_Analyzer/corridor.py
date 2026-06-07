"""
corridor.py
===========
Fetches road centerline geometry for the user-defined corridor.

PRIMARY PATH  : osmnx queries the OpenStreetMap Overpass API and returns the
                nearest road network path between two points. This gives us
                the true road geometry — curves, intersections, etc.

FALLBACK PATH : If the OSM fetch fails (timeout, no WiFi, firewall), we
                interpolate a straight-line corridor from the user's waypoints.
                The fallback is clearly labeled in the UI so the audience
                understands what they're seeing.

OUTPUT in both cases: a Shapely LineString in WGS84 (EPSG:4326), plus
                      the total corridor length in meters.

PROJECTION STRATEGY:
    All geometric operations (buffering, chainage math) must happen in a
    projected CRS — not latitude/longitude. We auto-detect the UTM zone
    from the corridor's centroid and project there. UTM gives meter-accurate
    distances without needing a global projection.
"""

import warnings
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj

# Suppress osmnx's verbose logging during demo
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# UTM PROJECTION HELPERS
# ---------------------------------------------------------------------------

def get_utm_crs(lat, lon):
    """
    Return the EPSG code for the UTM zone containing (lat, lon).
    UTM zones are 6-degree longitude bands; zone number = floor((lon+180)/6)+1.
    Northern hemisphere uses EPSG:326XX; southern uses EPSG:327XX.
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone_number   # Northern hemisphere
    else:
        epsg = 32700 + zone_number   # Southern hemisphere
    return f"EPSG:{epsg}"


def project_linestring(linestring_wgs84, utm_crs):
    """
    Re-project a Shapely LineString from WGS84 to the given UTM CRS.
    Returns the projected LineString (coordinates now in meters).
    """
    wgs84    = pyproj.CRS("EPSG:4326")
    utm      = pyproj.CRS(utm_crs)
    projector = pyproj.Transformer.from_crs(wgs84, utm,
                                             always_xy=True).transform
    return transform(projector, linestring_wgs84)


def unproject_linestring(linestring_utm, utm_crs):
    """
    Re-project a Shapely LineString from UTM back to WGS84.
    Used to convert computed gap/placement geometries back to lat/lon
    for Folium (which expects WGS84).
    """
    utm   = pyproj.CRS(utm_crs)
    wgs84 = pyproj.CRS("EPSG:4326")
    projector = pyproj.Transformer.from_crs(utm, wgs84,
                                             always_xy=True).transform
    return transform(projector, linestring_utm)


def unproject_point(x_utm, y_utm, utm_crs):
    """Convert a single UTM (x, y) point back to (lat, lon) in WGS84."""
    utm   = pyproj.CRS(utm_crs)
    wgs84 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)
    lon, lat = transformer.transform(x_utm, y_utm)
    return lat, lon


# ---------------------------------------------------------------------------
# STRAIGHT-LINE FALLBACK CORRIDOR
# ---------------------------------------------------------------------------

def build_fallback_corridor(waypoints):
    """
    Build a straight-line corridor from a list of (lat, lon) waypoints.
    Used when OSM fetch is unavailable.

    Args:
        waypoints: list of (lat, lon) tuples

    Returns:
        (linestring_wgs84, utm_crs, linestring_utm, length_m, "fallback")
    """
    # Shapely LineString expects (lon, lat) — note the swap from (lat, lon)
    coords_lonlat = [(lon, lat) for lat, lon in waypoints]
    line_wgs84    = LineString(coords_lonlat)

    # Auto-detect UTM zone from corridor centroid
    centroid = line_wgs84.centroid
    utm_crs  = get_utm_crs(centroid.y, centroid.x)

    line_utm   = project_linestring(line_wgs84, utm_crs)
    length_m   = line_utm.length

    return line_wgs84, utm_crs, line_utm, length_m, "fallback"


# ---------------------------------------------------------------------------
# OSM CORRIDOR FETCH (PRIMARY PATH)
# ---------------------------------------------------------------------------

def fetch_osm_corridor(start_latlon, end_latlon, timeout=10):
    """
    Query OpenStreetMap for the road network between two points and extract
    the shortest path as a LineString.

    Args:
        start_latlon : (lat, lon) tuple for corridor start
        end_latlon   : (lat, lon) tuple for corridor end
        timeout      : seconds before giving up and returning None

    Returns:
        (linestring_wgs84, utm_crs, linestring_utm, length_m, "osm")
        or None if the fetch fails for any reason.

    HOW osmnx WORKS HERE:
        1. graph_from_point() downloads the drivable road network within a
           bounding box that contains both points.
        2. nearest_nodes() snaps our start/end coords to the closest graph nodes.
        3. shortest_path() returns the list of node IDs on the shortest route.
        4. We extract the (lon, lat) coordinates of each node to form a LineString.
    """
    try:
        import osmnx as ox
        import networkx as nx

        # Configure osmnx: suppress logs, set HTTP timeout
        ox.settings.log_console = False
        ox.settings.timeout     = timeout

        start_lat, start_lon = start_latlon
        end_lat,   end_lon   = end_latlon

        # Download road network for a bounding box covering both endpoints
        # 'drive' network type = roads accessible to vehicles
        north = max(start_lat, end_lat) + 0.005
        south = min(start_lat, end_lat) - 0.005
        east  = max(start_lon, end_lon) + 0.005
        west  = min(start_lon, end_lon) - 0.005

        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type="drive",
            simplify=True
        )

        # Snap start/end coordinates to nearest graph nodes
        orig_node = ox.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.nearest_nodes(G, X=end_lon,   Y=end_lat)

        # Find shortest path (by edge length) between the two nodes
        path_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")

        # Extract (lon, lat) coordinate sequence from the path nodes
        coords_lonlat = [
            (G.nodes[n]["x"], G.nodes[n]["y"]) for n in path_nodes
        ]

        if len(coords_lonlat) < 2:
            return None   # Degenerate path — fall back

        line_wgs84 = LineString(coords_lonlat)
        centroid   = line_wgs84.centroid
        utm_crs    = get_utm_crs(centroid.y, centroid.x)
        line_utm   = project_linestring(line_wgs84, utm_crs)
        length_m   = line_utm.length

        return line_wgs84, utm_crs, line_utm, length_m, "osm"

    except Exception:
        # Any failure — network error, timeout, no path found — returns None
        # The caller will use the fallback corridor instead
        return None


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT FOR app.py
# ---------------------------------------------------------------------------

def get_corridor(waypoints, start_address=None, end_address=None,
                 use_osm=True, timeout=10):
    """
    Master function called by app.py. Tries OSM first; falls back gracefully.

    Args:
        waypoints     : list of (lat, lon) tuples (minimum 2)
        start_address : human-readable label for UI display only
        end_address   : human-readable label for UI display only
        use_osm       : if False, skips OSM and goes straight to fallback
                        (useful for testing or when user disables it)
        timeout       : OSM fetch timeout in seconds

    Returns dict with keys:
        line_wgs84   : Shapely LineString in WGS84 (for Folium)
        utm_crs      : EPSG string of the projected CRS
        line_utm     : Shapely LineString in UTM meters (for geometry math)
        length_m     : total corridor length in meters
        source       : "osm" or "fallback"
        warning      : None or a user-facing warning string
    """
    result = None

    if use_osm and len(waypoints) >= 2:
        result = fetch_osm_corridor(
            start_latlon=waypoints[0],
            end_latlon=waypoints[-1],
            timeout=timeout
        )

    if result is None:
        # OSM failed or was disabled — use straight-line fallback
        line_wgs84, utm_crs, line_utm, length_m, source = \
            build_fallback_corridor(waypoints)
        warning = (
            "⚠️ OSM fetch unavailable — using straight-line corridor. "
            "Road geometry may not match actual road alignment. "
            "Coverage and gap analysis remain valid relative to this centerline."
        )
    else:
        line_wgs84, utm_crs, line_utm, length_m, source = result
        warning = None

    return {
        "line_wgs84": line_wgs84,
        "utm_crs":    utm_crs,
        "line_utm":   line_utm,
        "length_m":   length_m,
        "source":     source,
        "warning":    warning,
    }
