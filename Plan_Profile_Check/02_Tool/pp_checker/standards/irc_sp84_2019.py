"""
irc_sp84_2019.py
================
IRC:SP:84-2019 — Manual for Four-Laning of Highways (PPP/EPC Projects)
All design parameters as Python dictionaries for use in the
IRC Geometric Design Compliance Checker rule engine.

Source   : IRC:SP:84-2019 (Sections 1–14 + Supplementary)
Scope    : Four-lane divided highway projects in India (NH / SH level)
Companion: irc_73_2023.py (vertical curve K-values cross-referenced here)
Resolved : IRC:92-2017 interchange geometry (13 supplementary tables)
           IRC:73-2023 vertical curve K-values (VC-01 to VC-05)
           IRC:35 parking geometry (Table 12-10)

Rule Types:
    HARD FAIL  = non-compliant, immediate design rejection
    ADVISORY   = warning, requires designer justification
    INFO       = reference only, no automated pass/fail

Usage:
    from data.irc_sp84_2019 import (
        DESIGN_SPEED, LANE_WIDTH, MEDIAN_WIDTH,
        SHOULDER_PLAIN_ROLLING, MIN_RADIUS, SSD_TABLE,
        GRADIENT, K_SUMMIT, K_VALLEY, ...
    )
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GEOMETRIC DESIGN AND GENERAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Clause 2.2.1, Table 2.1 — Design Speed (km/h)
# Key: (terrain_group, speed_type)
# Rule: < minimum → HARD FAIL; < ruling (= minimum here) → ADVISORY
DESIGN_SPEED = {
    ("plain_rolling",    "ruling"):  100,
    ("plain_rolling",    "minimum"):  80,   # ADVISORY — must be declared in Concession Agreement
    ("mountain_steep",   "ruling"):   60,
    ("mountain_steep",   "minimum"):  40,   # ADVISORY — must be declared in Concession Agreement
}
# Service road: minimum 40 km/h (Clause 2.12.2)
SERVICE_ROAD_DESIGN_SPEED_MIN = 40  # km/h  HARD FAIL

# Clause 2.4 — Lane Width (m)
LANE_WIDTH = 3.5          # m — all locations, HARD FAIL if < 3.5 m

# Clause 2.5.1, Table 2.2 — Minimum Median Width (m)
# Key: (location_type, terrain_group)
# Includes 0.5 m kerb shyness on each side
MEDIAN_WIDTH_MIN = {
    ("open_country",             "plain_rolling"):    5.0,   # HARD FAIL
    ("open_country",             "mountain_steep"):   7.0,   # HARD FAIL
    ("builtup",                  "all"):              2.5,   # HARD FAIL
    ("grade_sep_approach",       "plain_rolling"):    5.0,   # HARD FAIL
    ("grade_sep_approach",       "steep"):            2.5,   # HARD FAIL
}
MEDIAN_CRASH_BARRIER_EXEMPTION_WIDTH = 9.0   # m — no barrier needed if median > 9 m
KERB_SHYNESS_MIN = 0.5                        # m each side (ADVISORY below this)
MEDIAN_TRANSITION_TAPER = "1 in 50"          # ADVISORY (Clause 2.5.4)
DEPRESSED_MEDIAN_PAVED_WIDTH_MIN = 0.6       # m each side of carriageway (HARD FAIL)

# Clause 2.6.1, Table 2.3 — Shoulder Widths: Plain and Rolling Terrain (m)
# Value: {"paved": x, "earthen": y, "total": z}
SHOULDER_PLAIN_ROLLING = {
    "open_country":              {"paved": 2.5, "earthen": 1.5, "total": 4.0},
    "builtup":                   {"paved": 2.5, "earthen": 0.0, "total": 2.5},
    "grade_sep_approach":        {"paved": 2.5, "earthen": 0.0, "total": 2.5},
    "bridge_approach":           {"paved": 2.5, "earthen": 1.5, "total": 4.0},
}
# All above: HARD FAIL if below prescribed values

# Clause 2.6.1, Table 2.4 — Shoulder Widths: Mountainous and Steep Terrain (m)
SHOULDER_MOUNTAIN_STEEP = {
    "open_country_hill_side":    {"paved": 1.5, "earthen": 0.0, "total": 1.5},
    "open_country_valley_side":  {"paved": 1.5, "earthen": 1.0, "total": 2.5},
    "builtup_or_structure":      {"raised": 0.25, "paved": 1.5, "total": 1.75},
}
# Kerb with channel required where embankment > 6.0 m (ADVISORY, Clause 2.6.2)
EMBANKMENT_KERB_TRIGGER_M = 6.0

# Clause 2.7.2, Table 2.5 — Extra Carriageway Width on Curves (per carriageway, m)
# Key: radius range (lower, upper); upper=None means > 300 m → no extra width
EXTRA_WIDTH_ON_CURVES = {
    (75,  100): 0.9,   # HARD FAIL if not provided
    (101, 300): 0.6,   # HARD FAIL if not provided
    # radius > 300: no extra width required
}

# Clause 2.8.1 / 2.8.3 — Crossfall (%)
CROSSFALL = {
    "bituminous":        2.5,   # HARD FAIL
    "cement_concrete":   2.0,   # HARD FAIL
    "earthen_shoulder":  3.0,   # HARD FAIL — minimum; must be ≥ 0.5% steeper than paved surface
    # Superelevated sections: outer earthen shoulder at reverse crossfall = 0.5%
    "superelevated_outer_earthen_shoulder": 0.5,
}

# Clause 2.9.3 — Maximum Superelevation (e_max)
# Key: condition string
E_MAX = {
    "below_desirable_min_radius": 0.07,   # 7% — HARD FAIL if exceeded
    "above_desirable_min_radius": 0.05,   # 5% — HARD FAIL if exceeded
    "urban_section":              0.05,   # 5% always in urban — HARD FAIL
    "major_junction":             0.05,   # 5% — HARD FAIL
}
# Note: 5% applies regardless of radius at urban sections and major junctions

# Clause 2.9.4, Table 2.6 — Horizontal Curve Radii (m)
# Key: (terrain_group, radius_type)
MIN_RADIUS = {
    ("plain_rolling",  "desirable"): 400,   # HARD FAIL if below without deviation flag
    ("plain_rolling",  "absolute"):  250,   # HARD FAIL (floor) — deviation required below desirable
    ("mountain_steep", "desirable"): 150,   # HARD FAIL
    ("mountain_steep", "absolute"):   75,   # HARD FAIL (floor)
}
# Rule engine:
#   R < absolute → HARD FAIL
#   absolute ≤ R < desirable AND no deviation flag → HARD FAIL
#   absolute ≤ R < desirable AND deviation declared → ADVISORY

# Clause 2.9.5, Table 2.7 — Sight Distance (SSD and Desirable Minimum, m)
# Key: design_speed (km/h)
# Value: {"ssd": m, "desirable": m}
# For new construction: HARD FAIL at desirable minimum; SSD = existing roads only
SSD_TABLE = {
    100: {"ssd": 180, "desirable": 360},
     80: {"ssd": 130, "desirable": 260},
     60: {"ssd":  90, "desirable": 180},
     40: {"ssd":  45, "desirable":  90},
}
# Rule: available < ssd → HARD FAIL
#       ssd ≤ available < desirable → ADVISORY (new construction: treat as HARD FAIL)

# Clause 2.9.6.2, Table 2.8 — Gradients (%)
# Key: terrain_group
# Value: {"ruling": %, "limiting": %}
GRADIENT = {
    "plain_rolling": {"ruling": 2.5, "limiting": 3.3},
    "mountainous":   {"ruling": 5.0, "limiting": 6.0},
    "steep":         {"ruling": 6.0, "limiting": 7.0},
}
# Rule: > limiting → HARD FAIL; > ruling and ≤ limiting → ADVISORY

# Clause 2.9.6.1, Table 2-12 — Vertical Alignment General
MIN_DISTANCE_BETWEEN_GRADE_CHANGES_M = 150   # HARD FAIL if < 150 m between VPI stations

# Clause 2.10.2 / 2.11.2, Table 2-13 — Clearances at Underpasses (m)
UNDERPASS_CLEARANCE = {
    # (width_m, vertical_clearance_m) — HARD FAIL
    20: {"vertical": 5.5},
    12: {"vertical": 4.0},
     7: {"vertical": 4.0},
}
OVERPASS_VERTICAL_CLEARANCE_MIN = 5.5   # m — at all points over project highway, HARD FAIL

# Clause 2.12.2, Table 2-14 — Service Road Parameters
SERVICE_ROAD = {
    "carriageway_width_open_country_m": 7.0,    # HARD FAIL
    "earthen_shoulder_each_side_m":     1.5,    # HARD FAIL
    "design_speed_min_kmph":           40,      # HARD FAIL
    # Bridge trigger:
    "small_bridge_threshold_m":         60,     # < 60 m → 2-lane bridge for service road
    "merge_before_large_bridge_m":      50,     # service road merges 50 m before bridge ≥ 60 m
    # Parking bay (INFO):
    "parking_bay_length_m":             20,
    "parking_bay_width_m":               3,
}

# Clause 2.12.3 / 2.12.4, Table 2-15 — Acceleration and Deceleration Lanes
# (Source: IRC:92-2017 §6.5, Table 6.4)
ACC_DEC_LANES = {
    "acc_lane_desirable_m": 250,   # ADVISORY if < 250 m
    "acc_lane_minimum_m":   180,   # HARD FAIL if < 180 m
    "dec_lane_desirable_m": 120,   # ADVISORY if < 120 m
    "dec_lane_minimum_m":    90,   # HARD FAIL if < 90 m
    "nose_offset_m":          2.0, # HARD FAIL if flush with through lane
    # Preferred forms:
    "acc_taper_direct":     "1:25 to 1:30 (L≈200 m taper zone)",
    "dec_taper_direct":     "1:12 to 1:15 (L≈120 m taper zone)",
    # Sight distance triggers:
    "exit_taper_travel_time_s":    10,  # 10 s travel time at highway speed, HARD FAIL
    "aux_lane_travel_time_s":       7,  # 7 s travel time at highway speed, HARD FAIL
}
# At 100 km/h: exit taper sight dist = 278 m; aux lane sight dist = 194 m
# At  80 km/h: exit taper sight dist = 222 m; aux lane sight dist = 156 m

# Clause 2.14, Table 2-16 — Median Openings
MEDIAN_OPENINGS = {
    "min_spacing_open_country_m":  2000,    # HARD FAIL
    "min_spacing_builtup_m":        500,    # HARD FAIL
    "min_opening_length_m":          18,    # HARD FAIL if < 18 m
    "max_opening_length_m":          20,    # ADVISORY if > 20 m (unless no storage lane)
    "shelter_lane_width_m":           3.5,  # HARD FAIL
    "visibility_clearance_from_tip_m": 120, # HARD FAIL (free of plantations and objects)
}

# Clause 2.18 / 2.19, Table 2-17 — 4-Laning Warrants (INFO)
LANING_WARRANTS_PCU_PER_DAY = {
    "4_lane_plain_rolling":     {"existing": 40000, "future": 60000},
    "4_lane_mountain_steep":    {"existing": 20000, "future": 30000},
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INTERSECTIONS AND GRADE SEPARATORS
# ─────────────────────────────────────────────────────────────────────────────

# Clause 3.2.2 — Intersection Design Speed
INTERSECTION_DESIGN_SPEED_FACTOR = 0.60   # 60% of approach speed, HARD FAIL

# Clause 3.2.2, Table 3.1 — Carriageway Widening Taper Rates at Intersections
# Key: (approach_speed_range_label)
# Value: {"desirable": "1:X", "absolute_min": "1:Y"}
INTERSECTION_TAPER = {
    "lt_50":    {"desirable": "1:35", "absolute_min": "1:20"},   # speed < 50 km/h
    "50_65":    {"desirable": "1:40", "absolute_min": "1:25"},   # 50–65 km/h
    "66_80":    {"desirable": "1:45", "absolute_min": "1:30"},   # 66–80 km/h
    "gt_80":    {"desirable": "1:50", "absolute_min": "1:40"},   # > 80 km/h
}
# Desirable: ADVISORY; Absolute minimum: HARD FAIL

# Clause 3.2.2 / 3.2.6 — Sight Distance at Intersections
# Intersection Sight Distance = 2 × SSD (from SSD_TABLE)
# Example at 100 km/h: 2 × 180 = 360 m (HARD FAIL)
INTERSECTION_SD_FACTOR = 2.0   # multiply SSD by this factor

# Clause 3.2.1 / 3.2.6 / 3.4.3 — At-Grade Intersection Geometric Standards
AT_GRADE_INTERSECTION = {
    "min_angle_hard_fail_deg":   70,   # HARD FAIL if < 70°
    "ideal_angle_deg":           90,   # ADVISORY if 70°–90°
    "min_level_approach_m":      30,   # minimum 30 m of side road at same level, HARD FAIL
    "right_turn_storage_width_m": 3.0, # ADVISORY
    # Ramp design speed (IRC:92-2017 §6.2.1 cross-reference):
    "ramp_speed_80kmh_highway_min_kmh":   40,
    "ramp_speed_100kmh_highway_min_kmh":  50,
    "loop_ramp_speed_min_kmh":            30,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EMBANKMENT AND CUT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

EMBANKMENT = {
    "high_embankment_trigger_m":        6.0,   # height ≥ 6.0 m → IRC:75 required, ADVISORY
    "max_side_slope_without_structure": "2H:1V",  # HARD FAIL if steeper
    "borehole_spacing_max_m":           100,   # INFO
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PAVEMENT DESIGN
# ─────────────────────────────────────────────────────────────────────────────

PAVEMENT = {
    "min_design_period_rigid_yr":  30,      # HARD FAIL; stage construction NOT permitted
    "max_iri_operation_mm_per_km": 2000,    # ADVISORY
    "max_iri_post_strengthening":  1800,    # ADVISORY
    "cracking_rutting_new":        "Nil",   # ADVISORY
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HIGHWAY DRAINAGE
# ─────────────────────────────────────────────────────────────────────────────

DRAINAGE = {
    "min_longitudinal_slope_lined_drains_pct": 0.2,   # HARD FAIL
    "max_side_slope_open_channel":             "2H:1V",  # HARD FAIL
    "min_subsurface_drain_dia_mm":             150,    # HARD FAIL
    "min_depth_below_subgrade_m":              0.5,    # HARD FAIL
    "min_filter_thickness_above_pipe_mm":      300,    # HARD FAIL
    # Culverts (Clause 7.4):
    "min_pipe_culvert_dia_new_mm":             1200,   # HARD FAIL; existing < 900 mm → replace
    "existing_culvert_retain_min_dia_mm":      900,    # existing ≥ 900 mm satisfactory → may extend
    "min_earth_cushion_over_pipe_mm":          600,    # HARD FAIL (excluding road crust)
    # Water spouts (Clause 6.8.2.3):
    "water_spout_rate_level_portions_sqm":     12,     # 1 per 12 m², ADVISORY
    "water_spout_rate_gradient_portions_sqm":  15,     # 1 per 15 m², ADVISORY
    # Embankment protection:
    "slope_protection_medium_embankment_m":    (3, 6), # height 3–6 m, ADVISORY
    "special_protection_high_embankment_m":    6,      # height > 6 m + bridge approaches, ADVISORY
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DESIGN OF STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURES = {
    # Embankment / RE wall limit (Clause 7.1):
    "max_height_without_structure_m":      7.5,    # HARD FAIL if height > 7.5 m without structure

    # Median on structures:
    "median_width_on_structure":           "same as approach section",  # HARD FAIL
    "median_transition_rate_on_structure": "1 in 50",                   # HARD FAIL

    # Crash barrier on structures (Clause 7.1):
    "crash_barrier_clearance_from_edge_m": 0.5,    # HARD FAIL

    # Culvert widths (Clause 7.3):
    "culvert_width":                       "equal to roadway width of approaches",  # HARD FAIL
    "railing_outermost_edge":              "in line with outermost edge of shoulder",  # HARD FAIL

    # Bridge widths (Clause 7.3):
    "bridge_full_width":                   "same as approaches — no narrowing",   # HARD FAIL
    "bridge_shyness_width_m":              0.5,    # HARD FAIL (each side, additional)
    # Reference bridge widths (INFO):
    "bridge_4lane_roadway_m":              13.5,
    "bridge_4lane_carriageway_m":          10.5,
    "bridge_4lane_footpath_m":              1.5,

    # Crash barrier at bridge approaches (Clause 7.3):
    "crash_barrier_transition_rate":       "1 in 20",   # HARD FAIL

    # ROB / Road-rail structures (Clause 7.18):
    "rob_max_skew_angle_deg":              45,           # HARD FAIL
    "rob_service_road_continuity":         "joined through one ROB viaduct",  # HARD FAIL

    # Crash barriers on structures (Clause 7.17):
    "rcc_crash_barrier_all_new_bridges":   True,         # HARD FAIL

    # Box girder (Clause 7.5):
    "min_box_girder_internal_height_m":    1.5,          # HARD FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — TRAFFIC CONTROL DEVICES AND ROAD SAFETY FURNITURE
# ─────────────────────────────────────────────────────────────────────────────

# Clause 9.2.3, Table 9-01 — Sign Sheeting Requirements
SIGN_SHEETING = {
    "standard_ground_mounted":   "Class C Prismatic (Aluminum/ACM substrate)",  # HARD FAIL
    "overhead_gantry":           "Type XI Micro-Prismatic",                      # HARD FAIL
    "delineator_post":           "Type IV Micro-Prismatic",                      # ADVISORY
}

# Clause 9.2.4, Table 9-02 — Sign Support Structures
SIGN_SUPPORT = {
    "shoulder_mounted_material":        "GI pipes",            # ADVISORY
    "overhead_structure_requirement":   "structurally sound gantry or cantilever",  # HARD FAIL
    "advertisement_on_sign":            False,                  # HARD FAIL (no ads)
    "min_gap_between_signs_same_post_m": 0.6,                  # > 0.6 m, HARD FAIL
}
# Note: Gantry vertical and lateral clearance PENDING per IRC:67

# Clause 9.2.7 / 9.4.4, Table 9-03 — Curve Warning Signs and Chevron Placement
CURVE_WARNING = {
    "warning_sign_trigger_radius_m":     450,    # required for ALL curves R < 450 m, HARD FAIL
    "dual_sided_speed_diff_trigger_kmh":  15,    # if (approach − safe speed) > 15 → both sides, HARD FAIL
    "chevron_spacing_r_451_to_1200_m":    75,    # m, HARD FAIL
    "delineator_spacing_r_1201_to_2000_m": 40,   # m (forgiving type), HARD FAIL
    # Safe negotiating speed formula:
    "safe_speed_formula":               "V_per = sqrt(127 * R * (e + f))",
}
# Engine logic:
#   R < 450 m  → chevrons (IRC:67 spacing) + warning + speed plate
#   R 451–1200 → chevrons at 75 m
#   R 1201–2000 → forgiving delineators at 40 m
#   R > 2000   → no chevron/delineator requirement

# Clause 9.3.2, Table 9-06 — Road Markings Edge Line
ROAD_MARKINGS = {
    "median_edge_line_min_offset_from_kerb_mm": 350,   # HARD FAIL if < 350 mm
    "shorter_lane_line_interval_trigger":        "curves R < 450 m at 100 km/h",
}

# Clause 9.5, Table 9.1, Table 9-07 — Road Studs (RRPM — ASTM D-4280)
RRPM = {
    # Horizontal Curves
    "hcurve_r_lte_450m":     {"coverage": "curve+transition+20m each side", "spacing_m":  9, "color": {"shoulder": "yellow", "median": "amber"}},
    "hcurve_r_451_750m":     {"coverage": "curve+transition+20m each side", "spacing_m": 18, "color": {"shoulder": "yellow", "median": "amber"}},
    "hcurve_r_751_2000m":    {"coverage": "curve+transition+20m each side", "spacing_m": 27, "color": {"shoulder": "yellow", "median": "amber"}},
    # Vertical Grades ≥ 2.5%
    "vertical_grade_gte_2_5pct": {"coverage": "full grade+VCs+180m either side", "spacing_m": 18, "color": {"shoulder": "yellow", "median": "amber"}},
    # No-Overtaking Zones
    "no_overtaking_zone":    {"spacing_m": 18, "color": {"shoulder": "yellow", "median": "amber", "center": "red"}},
    # Structures (major/minor bridges, ROBs)
    "structure_portion":     {"spacing_m": 9,  "color": {"shoulder": "yellow", "median": "amber"}},
    "structure_approach_180m": {"spacing_m": 18, "color": {"shoulder": "yellow", "median": "amber"}},
    # Built-up sections
    "builtup_sections":      {"spacing_m": 18, "color": {"shoulder": "yellow", "median": "amber"}},
    # Slip roads / ramps
    "slip_road_edge":        {"spacing_m": 9,  "color": "red"},
    "slip_road_chevron":     {"spacing_m": 6,  "color": "red"},
    "slip_road_continuity":  {"spacing_m": 8,  "color": "green"},
    # Junctions / Median Openings
    "junction_storage_lane": {"spacing_m": 18, "color": {"shoulder": "red", "median": "amber"}},
    "junction_chevron":      {"spacing_m": 6,  "color": "red"},
    "junction_acc_lane":     {"rows": 3, "row_spacing_m": 1, "color": "green"},
    # Pedestrian crossings
    "zebra_crossing":        {"spacing_m": 0.5, "color": "amber"},
    # All RRPM are HARD FAIL
}

# Clause 9.6, Table 9-08 — Crash Attenuators
CRASH_ATTENUATOR = {
    "mandatory_locations":           "all hazardous locations and gorge areas",
    "testing_standard":              "NCHRP Report 350",
    "speed_threshold_diverge_kmh":   70,   # required if speed > 70 km/h at diverge, HARD FAIL
    # Reserve space (N=perpendicular setback, F=flush clearance, L=lateral clearance):
    "reserve_space": {
        50:  {"restricted": {"N": 2.0, "F": 0.5,  "L": 2.5}, "unrestricted": {"N": 2.5, "F":  3.5, "L":  3.5}, "preferred": {"N": 3.5, "L":  4, "F": 1.0}},
        80:  {"restricted": {"N": 2.0, "F": 5.0,  "L": 3.5}, "unrestricted": {"N": 2.5, "F":  7.5, "L": 13.5}, "preferred": {"N": 3.5, "L": 17, "F": 1.0}},
        100: {"restricted": {"N": 2.0, "F": 8.5,  "L": 3.5}, "unrestricted": {"N": 2.5, "F": 13.5, "L": 17.0}, "preferred": {"N": 3.5, "L": 21, "F": 1.5}},
        120: {"restricted": {"N": 2.0, "F": 8.5,  "L": 3.5}, "unrestricted": {"N": 2.5, "F": 17.0, "L": 17.0}, "preferred": {"N": 3.5, "L": 23, "F": 1.5}},
    },
    # All above: HARD FAIL
}

# Clause 9.7.1–9.7.4 — Roadside Safety Barriers
SAFETY_BARRIERS = {
    # Type selection (INFO):
    "types": ["W-beam steel", "New Jersey concrete (rigid)", "Wire rope"],

    # Warrants (HARD FAIL):
    "concrete_mandatory_locations": ["all bridges", "ROBs", "restricted clearance zones"],
    "wire_rope_bridge_prohibition": True,   # wire rope NOT permitted over major/minor bridges
    "builtup_section_mandatory":    True,   # crash barriers mandatory in built-up sections
    "embankment_adjacent_hazards":  True,   # barrier at embankments with hazard risk

    # W-beam (Clause 9.7.2):
    "wbeam": {
        "rail_thickness_mm":           3,
        "post_section_mm":             "75×150×5 channel",
        "post_spacing_standard_m":     2.0,
        "splice_overlap_mm":           318,
        "splice_bolts":                "8 × 16mm Ø button-head",
        "galvanization":               "hot dip — all components",
        "max_face_to_kerb_mm":         100,    # < 100 mm from kerb face, HARD FAIL
        "end_treatment_approach":      "MELT (Modified Eccentric Loader Terminal)",
        "end_treatment_departure":     "Trailing Terminal (TT)",
        "transition_to_concrete_min_m": 7.5,
        "max_ground_slope_in_front":   "10:1 (H:V)",
    },

    # Thrie beam (Clause 9.7.2):
    "thrie_beam": {
        "post_section_mm":   "75×150×5 channel",
        "galvanization":     "hot dip",
    },

    # Concrete New Jersey type (Clause 9.7.3):
    "concrete_nj": {
        "concrete_grade_min":             "M30",
        "foundation_base_thickness_mm":   25,    # standard; 125 mm if overlay > 75 mm anticipated
        "end_taper_length_m":             (8, 9),
        "flare_rates":                    {100: "17:1", 80: "14:1", 60: "11:1", 40: "8:1"},
        "max_ground_slope_in_front":      "10:1",
        "max_precast_segment_m":           6,    # ADVISORY
    },

    # Wire rope (Clause 9.7.4):
    "wire_rope": {
        "containment_level":             "EN 1317-2 Level H2 minimum",
        "transition_sequence":           "Wire Rope → W-beam → Concrete (mandatory)",
        "extension_approach_m":          30,    # ≥ 30 m before hazard at full height, HARD FAIL
        "extension_departure_m":          7.5,  # ≥ 7.5 m beyond hazard, HARD FAIL
        "max_ground_slope_in_front":     "10:1",
        "no_kerb_adjacent":              True,  # HARD FAIL
    },

    # Median barriers (Clause 9.7.5):
    "median": {
        "wide_depressed_gte_7m_type":     "W-beam or wire rope at BOTH edges",  # HARD FAIL
        "narrow_lte_2m_type":             "New Jersey concrete",                  # HARD FAIL
        "narrow_lte_2m_antiglare":        True,                                   # HARD FAIL
        "concrete_end_taper_m":           (8, 9),
        "wire_rope_same_level_position":  "centre of median",
        "wire_rope_diff_level_position":  "both sides of median edge",
        "wire_rope_split_median":         "on carriageway of higher side",
    },
}

# Clause 9.8 — Pedestrian Facilities
PEDESTRIAN = {
    "footpath_min_width_m":               1.5,    # HARD FAIL
    "guardrail_height_m":                 1.2,    # HARD FAIL
    "guardrail_setback_from_edge_mm":     150,    # ≥ 150 mm, HARD FAIL
    "zebra_crossing_min_spacing_m":       150,    # HARD FAIL
    "zebra_crossing_width_m":             (2.0, 4.0),  # range, HARD FAIL
    "median_kerb_ht_at_refuge_mm":        150,    # ≤ 150 mm at pedestrian refuge, HARD FAIL
}

# Clause 9.9, Table 9-16 — Work Zone Advance Warning Sign Distances (m)
# Key: speed_band_label
# Value: {a, b, c, d} — positions from work zone taper
WORK_ZONE_WARNING = {
    "lte_50":    {"a": 60,  "b": 60,  "c": 60,  "d": 40},   # speed ≤ 50 km/h
    "51_65":     {"a": 90,  "b": 90,  "c": 90,  "d": 45},
    "66_80":     {"a": 90,  "b": 90,  "c": 90,  "d": 45},
    "81_100":    {"a": 110, "b": 120, "c": 120, "d": 60},
    "81_105":    {"a": 160, "b": 160, "c": 180, "d": 75},
    # All HARD FAIL
}
WORK_ZONE_LANE_WIDTH = {
    "through_lane_min_m":    3.25,   # HARD FAIL
    "reduced_lane_min_m":    1.50,   # HARD FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — TOLL PLAZAS
# ─────────────────────────────────────────────────────────────────────────────

TOLL_PLAZA = {
    # Lane geometry (Clause 10.4.2):
    "standard_lane_width_m":         3.2,    # HARD FAIL
    "odc_lane_width_m":              4.5,    # HARD FAIL (over-dimensional/slow traffic)

    # Traffic island geometry (Clause 10.4.3):
    "island_length_standard_m":      25,     # HARD FAIL (lanes without WIM)
    "island_length_wim_m":           35,     # HARD FAIL (with MS-WIM; 22.5 m approach side)
    "island_width_min_m":             1.9,   # HARD FAIL
    "island_barrier_type":            "RCC with chevron markings",  # HARD FAIL

    # Toll booth (Clause 10.4.4):
    "cctv_per_booth":                 1,     # HARD FAIL
    "canopy_height_m":                6.5,   # HARD FAIL (from Fig. 10.3)

    # Underground tunnel (Clause 10.4.5):
    "tunnel_min_width_m":             3.0,   # HARD FAIL
    "tunnel_min_height_m":            2.5,   # HARD FAIL

    # Transition taper (Clause 10.4.6):
    "transition_taper_min":          "1:10",  # HARD FAIL if steeper than 1:10
    "transition_taper_preferred":    "1:20",  # ADVISORY if between 1:10 and 1:20
    "canopy_coverage":                "all toll lanes and booths",  # HARD FAIL

    # Pavement (Clause 10.7):
    "plaza_pavement_type":            "Rigid (CC) per IRC:58",  # HARD FAIL

    # Signage (Clause 10.8):
    "advance_sign_1km_m":             1000,   # HARD FAIL (first toll gate info sign)
    "advance_sign_500m_m":             500,   # HARD FAIL (repeater sign)
    "advance_sign_1km_dims_mm":        {"width": 2864, "height": 1150},
    "advance_sign_500m_dims_mm":       {"width": 3262, "height": 1150},
    "sign_placement_max_height_m":     3,     # HARD FAIL
    "sign_angle_deg":                  45,    # angled to direction of travel

    # Lane capacity and queue (Clause 10.6.2):
    "etc_lane_capacity_vph":          1200,   # INFO (design reference)
    "reversible_lanes_min":              2,   # HARD FAIL (not less than 2 middle lanes)
    "max_queue_wait_time_min":           3,   # ADVISORY trigger

    # ETC system (Clause 10.5.4):
    "etc_uptime_pct_per_month":       98,     # HARD FAIL
    "etc_max_scheduled_downtime_hr_per_month": 4,  # HARD FAIL
    "broadband_min_speed_mbps":        2,     # HARD FAIL

    # Weigh-in-motion (Clause 10.6.1):
    "wim_at_each_toll_lane":          True,   # HARD FAIL
    "static_weigh_bridge_per_dir":    True,   # HARD FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — LANDSCAPING AND TREE PLANTATION
# ─────────────────────────────────────────────────────────────────────────────

LANDSCAPING = {
    "second_row_setback_beyond_first_m":   6,      # HARD FAIL
    "plantation_embankment_slopes":        False,   # HARD FAIL (no plantation on embankment slopes)
    "urban_tree_setback_from_kerb_m":      2,      # HARD FAIL
    # Sight distance on curves (Clause 11.2.2):
    "ssd_curve_plain_normal_m":           180,     # HARD FAIL (100 km/h)
    "ssd_curve_plain_restricted_m":       130,     # ADVISORY (80 km/h; requires formal declaration)
    # Vertical clearance (Clause 11.2.3):
    "min_vertical_clearance_m":           5,       # HARD FAIL (absolute minimum)
    "trim_height_rural_m":                6,       # HARD FAIL
    "trim_height_urban_m":                6.5,     # HARD FAIL
    # Median plantation (Clause 11.2.4):
    "median_plantation_min_width_m":      2.5,     # HARD FAIL (shrubs permitted only > 2.5 m)
    # Avenue tree spacing (Clause 11.2.5):
    "avenue_tree_spacing_m":              (10, 15), # ADVISORY
    # Species criteria (Clause 11.2.6):
    "first_branch_height_min_m":          2.5,     # HARD FAIL
    "first_branch_height_max_m":          3.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — PROJECT FACILITIES
# ─────────────────────────────────────────────────────────────────────────────

# Clause 12.2, Table 12-01 — Road Boundary Wall
BOUNDARY_WALL = {
    "material":         "RCC",    # HARD FAIL
    "height_mm":         900,     # HARD FAIL
    "top_width_mm":      200,
    "base_width_mm":     700,     # HARD FAIL
    "toe_size_mm":       "300×100",
}

# Clause 12.5.2, Table 12-03 — Street Lighting
STREET_LIGHTING = {
    "min_illumination_lux":  40,    # HARD FAIL (average maintained)
    "dg_backup_required":    True,  # HARD FAIL
    "design_standard":       "IRC:32",
    # Mandatory lighting locations (all HARD FAIL if absent):
    "mandatory_locations": [
        "toll plaza and approach roads",
        "rest areas",
        "truck lay-byes and 50 m on each side",
        "bus bays",
        "grade-separated structures (flyovers, VUPs, PUPs)",
        "service roads",
    ],
}

# Clause 12.6 — Truck Lay-bye (Table 12-05, Fig. 12.2)
TRUCK_LAYBE = {
    "taper_length_m":    70,     # each end, HARD FAIL
    "straight_length_m": 100,    # parking bay, HARD FAIL
    "illumination_lux":  40,     # lay-bye + 50 m each side, HARD FAIL
    "taper_rate":        "1:20", # HARD FAIL
}

# Clause 12.7 — Bus Bay (Table 12-06)
BUS_BAY = {
    "min_bay_length_per_bus_m":    15,      # HARD FAIL (multiply by simultaneous buses)
    "prohibited_embankment_m":      3,      # NOT at embankment > 3 m height, HARD FAIL
    "prohibited_from_intersection_tp_m": 30,  # HARD FAIL (< 30 m is HARD FAIL)
    "advisory_from_intersection_tp_m": 60,    # 30–60 m ADVISORY
    "pavement_type":               "CC (cement concrete) blocks, not raised",  # HARD FAIL
    "opposite_side_staggering":    True,    # HARD FAIL
    "illumination_lux":            40,      # HARD FAIL
}

# Clause 12.9 — Rest Areas (Table 12-07)
REST_AREA = {
    "min_car_parking_spaces":  50,    # HARD FAIL
    "mandatory_facilities": [
        "toilets", "STD/ISD telephones", "cafeteria/restaurant",
        "car/bus/truck parking", "dormitory", "rest rooms",
        "shops for travel needs", "fuel stations", "first aid",
    ],  # Each mandatory, HARD FAIL
    "illumination_lux":        40,    # HARD FAIL
}

# Clause 12.10 — Highway Patrol
HIGHWAY_PATROL = {
    "max_patrol_stretch_km":  50,    # HARD FAIL
    "real_time_communication": True, # HARD FAIL
}

# Clause 12.15 — O&M Centre
OM_CENTRE = {
    "parking_cross_slope_pct": (1.0, 2.5),   # HARD FAIL outside this range
    "demarcation_standard":    "IRC:35",
    "illumination_lux":        40,
}

# Table 12-10 (IRC:35 standard) — Parking Bay Dimensions
PARKING_GEOMETRY = {
    "car_bay_m":           {"width": 2.5, "length": 5.0},       # HARD FAIL
    "bus_bay_m":           {"width": 3.5, "length": 12.0},      # HARD FAIL
    "truck_bay_m":         {"width": 3.5, "length": 15.0},      # HARD FAIL
    "bay_marking_width_mm": 100,                                  # ADVISORY
    # Circulation aisle widths:
    "aisle_90deg_twoway_m":  6.0,   # HARD FAIL
    "aisle_90deg_oneway_m":  4.5,   # HARD FAIL
    "aisle_60deg_oneway_m":  4.5,   # HARD FAIL
    "aisle_60deg_twoway_m":  6.0,   # HARD FAIL
    "aisle_45deg_oneway_m":  3.5,   # HARD FAIL
    "aisle_45deg_twoway_m":  6.0,   # HARD FAIL
    # Counts:
    "rest_area_min_car_spaces": 50, # HARD FAIL
    # Drainage:
    "parking_cross_slope_pct": {"min": 1.0, "max": 2.5},        # HARD FAIL outside range
    # Covered parking:
    "covered_parking_min_headroom_m": 2.1,                       # HARD FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — SPECIAL REQUIREMENTS FOR HILL ROADS
# ─────────────────────────────────────────────────────────────────────────────

HILL_ROAD = {
    "hairpin_bend_standard":          "IRC:52",   # HARD FAIL
    "hairpin_bend_full_surfacing":    True,        # HARD FAIL
    "grade_compensation_trigger_pct": 4.0,         # apply for gradient > 4%, HARD FAIL
    "slope_stability_methods": [
        "Breast Wall", "Concrete Cladding", "Soil Nailing",
        "Rock Bolting", "Rock Fall Netting", "Hydro-seeding", "Geo-fabric",
    ],
    "blasting":                       "controlled blasting only",  # HARD FAIL
    "debris_valley_dumping":          False,   # HARD FAIL (no dumping into valley/waterways)
    # Drainage:
    "catchwater_drains_cut_sections": True,    # HARD FAIL
    # Retaining/breast walls:
    "retaining_wall_standard":        "IRC:SP:48",  # HARD FAIL
    "breast_wall_side_drain":         True,          # HARD FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — TUNNELS
# ─────────────────────────────────────────────────────────────────────────────

# Clause 14.2 — Tunnel Cross-Section (Table 14-01)
TUNNEL_CROSS_SECTION = {
    "geometric_standard":             "same as main carriageway — no relaxation",  # HARD FAIL
    "min_vc_over_carriageway_and_shoulder_m": 5.5,  # HARD FAIL
    "min_vc_over_footway_m":          3.0,    # HARD FAIL
    "total_internal_width_3lane_m":  17.0,    # HARD FAIL
    "carriageway_width_3lane_m":     10.5,    # 3 × 3.5 m, HARD FAIL
    "left_paved_shoulder_m":          3.0,    # HARD FAIL
    "right_edge_strip_m":             0.5,    # HARD FAIL
    "escape_footway_width_m":         0.95,   # HARD FAIL (left side)
    "walkway_right_side_m":           0.75,   # HARD FAIL
    "utility_duct":                   "both sides",  # HARD FAIL
    "pavement_type":                  "CC (cement concrete high performance)",  # HARD FAIL
    # Drainage camber:
    "bidirectional_camber":           "outward from centre (crown to both edges)",
    "unidirectional_camber":          "from high point to one side",
    "carriageway_cross_slope_pct":    2.5,    # HARD FAIL
    "waterproofing_min_thickness_mm": 0.8,    # HARD FAIL
}

# Clause 14.2.6 — Tunnel Emergency Lay-bye (Table 14-03)
TUNNEL_LAYBE = {
    "trigger_length_m":     500,   # required in tunnels > 500 m, HARD FAIL
    "laybe_length_m":       100,   # HARD FAIL
    "laybe_width_m":          3.0, # HARD FAIL (beyond left lane, additional)
    "spacing_max_m":        750,   # HARD FAIL (c/c spacing)
}

# Clause 14.2.8 — Tunnel Cross Passage (Table 14-04)
TUNNEL_CROSS_PASSAGE = {
    "trigger_length_m":        500,   # required in twin tunnels > 500 m, HARD FAIL
    "max_spacing_m":           300,   # c/c maximum, HARD FAIL
    "angle_to_tunnel_axis_deg": 30,   # HARD FAIL
}

# Clause 14.2.9 / 14.2.10 — Tunnel Alignment (Table 14-05)
TUNNEL_ALIGNMENT = {
    "max_gradient_long_m":    2.5,    # % — tunnels > 500 m, HARD FAIL
    "max_gradient_short_m":   6.0,    # % — tunnels ≤ 500 m; enhanced ventilation mandatory
    "max_straight_inside_m":  1500,   # HARD FAIL (monotony / speed increase risk)
}

# Clause 14.7 — Tunnel Ventilation (Table 14-06)
TUNNEL_VENTILATION = {
    "natural_only_max_m":         250,   # natural ventilation acceptable ≤ 250 m (ADVISORY)
    "mechanical_mandatory_gt_m":  500,   # HARD FAIL for tunnels > 500 m
    "design_standard":            "IRC:SP:91",
}

# Clause 14.8 — Tunnel Lighting (Table 14-07)
TUNNEL_LIGHTING = {
    "design_standard":            "MoRTH Expressway Guidelines Ch. 13.5",
    "cctv_spacing_m":             200,    # with zoom function, HARD FAIL
    "emergency_lighting_min_min": 40,     # 40 minutes capacity, HARD FAIL
    "emergency_lighting_design_allowance_pct": 20,
    # Interior / threshold zone luminance: PENDING from MoRTH Ch. 13.5
}

# Clause 14.11 — Tunnel Emergency Facilities (Table 14-09)
# Classification: AA (highest), A, B, C, D (< 200 m)
TUNNEL_EMERGENCY = {
    "emergency_telephone":     {"AA": True, "A": True, "B": True, "C": True, "D": False},
    "fire_detector":           {"AA": True, "A": True, "B": True, "C": False, "D": False},
    "fire_extinguisher":       {"AA": True, "A": True, "B": True, "C": True,  "D": False},
    "fire_plug":               {"AA": True, "A": True, "B": True, "C": True,  "D": False},
    "cctv_spacing_m":          200,   # all classes, HARD FAIL
    "emergency_power":         {"AA": True, "A": True, "B": True},   # ≥ 200 m, HARD FAIL
    "evacuation_tunnel_vc_m":  4.5,   # HARD FAIL (escape tunnel vertical clearance)
    "water_sprinkler_trigger": "Class AA ≥ 3000 m",
    # All HARD FAIL for stated classes
}


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — VERTICAL CURVE K-VALUES (from IRC:73-2023)
# Resolves PENDING 2-P2 in IRC:SP:84-2019
# Source: IRC:73-2023 Tables 7.3 / 7.5 — applicable to SP:84 per §2.9.6
# ─────────────────────────────────────────────────────────────────────────────

# Table VC-01 — Summit (Crest) Curve K-values (L > S case)
# K = S² / 4.4 ;  L = K × N (N in %)  → HARD FAIL if K_provided < K_required
# Primary SP:84 check speeds: 100 km/h (ruling) and 80 km/h (minimum)
K_SUMMIT_SSD = {
    40:  4.60,
    50:  8.18,
    60:  14.55,
    65:  18.41,
    80:  38.41,   # PRIMARY CHECK — minimum design speed
    100: 73.64,   # PRIMARY CHECK — ruling design speed
    120: 56.82,
}

# Table VC-01 — Summit Curve K-values based on ISD (governing for new construction)
# K = ISD² / 9.6  (ISD = 2 × SSD per IRC:SP:84 Table 2.7)
# IRC:SP:84 Clause 2.9.7.1: New construction checked against ISD; SSD only where stated
K_SUMMIT_ISD = {
    40:   8.44,   # ISD=90m  (2×45):  90²/9.6=844
    50:  15.00,   # ISD=120m (2×60): 120²/9.6=1500
    60:  33.75,   # ISD=180m (2×90): 180²/9.6=3375
    65:  33.75,   # ISD=180m (2×90): 180²/9.6=3375
    80:  70.42,   # ISD=260m (2×130): 260²/9.6=7042
    100: 135.00,  # ISD=360m (2×180): 360²/9.6=13500
    120: 260.42,  # ISD=500m (2×250): 500²/9.6=26042
}

# Table VC-02 — Valley (Sag) Curve K-values (L > S case, headlight SSD)
# K = S² / (1.50 + 0.035×S)  →  HARD FAIL if K_provided < K_required
K_VALLEY = {
    40:  5.29,
    50:  7.89,
    60:  11.48,
    65:  13.48,
    80:  22.18,   # PRIMARY CHECK
    100: 35.36,   # PRIMARY CHECK
    120: 56.82,   # PRIMARY CHECK
}

# Table VC-03 — Minimum Vertical Curve Length (independent of K-value)
# Key: design_speed (km/h)
# Value: {"max_N_no_vc_pct": %, "min_L_m": m}
MIN_VC_LENGTH = {
    35:  {"max_N_no_vc_pct": 1.5, "min_L_m": 15},
    40:  {"max_N_no_vc_pct": 1.2, "min_L_m": 20},
    50:  {"max_N_no_vc_pct": 1.0, "min_L_m": 30},
    60:  {"max_N_no_vc_pct": 0.8, "min_L_m": 40},
    80:  {"max_N_no_vc_pct": 0.6, "min_L_m": 50},   # PRIMARY CHECK
    100: {"max_N_no_vc_pct": 0.5, "min_L_m": 60},   # PRIMARY CHECK
    120: {"max_N_no_vc_pct": 0.5, "min_L_m": 100},   # PRIMARY CHECK
}
# Engine logic:
#   Step 1: N ≤ max_N_no_vc → PASS (no VC required)
#   Step 2: N > threshold AND no VC → HARD FAIL
#   Step 3: L_required = max(K × N, min_L_m); if L_provided < L_required → HARD FAIL

# Table VC-04 — L < S Case Formulas
VC_LSS_FORMULAS = {
    "summit_ssd":    "L = 2*S - 4.4/N",
    "summit_isd":    "L = 2*S - 9.6/N",
    "valley_hl_ssd": "L = 2*S - (1.50 + 0.035*S)/N",
}
# Quick reference for primary SP:84 speeds:
VC_LSS_QUICK_REF = {
    80:  {"summit_ssd": "260 - 4.4/N", "valley_hl": "260 - 6.1/N"},
    100: {"summit_ssd": "360 - 4.4/N", "valley_hl": "360 - 7.8/N"},
}

# Table VC-05 — Complete Engine Logic (pseudo-code)
VC_ENGINE_LOGIC = """
Inputs: V (km/h), G1 (%), G2 (%), curve_type ("summit"|"valley"), L_provided (m)
Step 1 — Is VC required?
    N = abs(G1 - G2)
    N_min = MIN_VC_LENGTH[V]["max_N_no_vc_pct"]
    if N <= N_min: return PASS

Step 2 — Compute required length:
    SSD = SSD_TABLE[V]["ssd"]
    if curve_type == "summit":
        K_req = K_SUMMIT_SSD[V]
        L_req_sight = K_req * N
        if L_req_sight < SSD:  L_req_sight = 2*SSD - 4.4/N   # L<S formula
    else:  # valley
        K_req = K_VALLEY[V]
        L_req_sight = K_req * N
        if L_req_sight < SSD:  L_req_sight = 2*SSD - (1.50 + 0.035*SSD)/N
    L_req_min = MIN_VC_LENGTH[V]["min_L_m"]
    L_required = max(L_req_sight, L_req_min)

Step 3 — Compare:
    if L_provided < L_required: return HARD FAIL
    return PASS
"""


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — IRC:92-2017 INTERCHANGE DESIGN PARAMETERS
# Resolves PENDING IRC-XREF-3.4 and PENDING 2-P1
# Source: IRC:92-2017 (Guidelines for Design of Interchanges in Urban Areas)
# Applicable to IRC:SP:84 projects with interchange components
# Primary case: highway design speed = 100 km/h
# ─────────────────────────────────────────────────────────────────────────────

# Table IRC92-01 — Ramp Design Speed (km/h)
# Key: (mainline_speed, ramp_type)
RAMP_DESIGN_SPEED = {
    (80,  "standard"): {"minimum": 40, "desirable": 50},
    (100, "standard"): {"minimum": 50, "desirable": 65},  # PRIMARY case
    (100, "loop"):     {"minimum": 30, "desirable": 40},
    (80,  "loop"):     {"minimum": 30, "desirable": 40},
}
# HARD FAIL if ramp speed < minimum

# Table IRC92-02 — Ramp Horizontal Curve Radii and SSD (m)
# Key: mainline_design_speed_kmh
RAMP_GEOMETRY = {
    80: {
        "min_radius_m":    60,    # HARD FAIL; computed at e_max = 7%
        "desirable_radius_m": 90,
        "ssd_min_m":       45,    # HARD FAIL
        "ssd_desirable_m": 60,
    },
    100: {                        # PRIMARY case for SP:84 projects
        "min_radius_m":    90,    # HARD FAIL
        "desirable_radius_m": 155,
        "ssd_min_m":       60,    # HARD FAIL
        "ssd_desirable_m": 90,
    },
    "loop": {
        "min_radius_m":    30,    # HARD FAIL
        "desirable_radius_m": 60,
        "ssd_min_m":       25,    # HARD FAIL
        "ssd_desirable_m": 45,
    },
}
RAMP_COMPOUND_CURVE_MIN_RATIO = 0.5   # R_smaller ≥ 0.5 × R_larger, HARD FAIL
RAMP_SSD_EYE_HEIGHT_M = 1.2
RAMP_SSD_OBJECT_HEIGHT_M = 0.15

# Table IRC92-03 — Ramp Grade and Vertical Curves
RAMP_GRADE = {
    "desirable_max_pct":  4.0,   # ADVISORY
    "absolute_max_pct":   6.0,   # HARD FAIL
}
# Ramp vertical curve K-values (L = K × A, A = algebraic grade diff %)
RAMP_K_SUMMIT = {30: 2.0, 40: 4.6, 50: 8.2, 65: 18.4, 80: 32.6, 100: 73.6}
RAMP_K_VALLEY = {30: 3.5, 40: 6.6, 50: 10.0, 65: 17.4, 80: 25.3, 100: 41.5}
RAMP_MIN_VC_LENGTH = {30: 15, 40: 20, 50: 30, 65: 40, 80: 50, 100: 60}  # m, HARD FAIL

# Table IRC92-04 — Ramp Superelevation
RAMP_SE = {
    "desirable_max_pct": 6.0,   # ADVISORY
    "absolute_max_pct":  7.0,   # HARD FAIL
}

# Table IRC92-05 — Ramp Cross-Section and Width
RAMP_CROSS_SECTION = {
    "median_two_way_divided_min_m":   1.2,   # HARD FAIL
    "shoulder_total_min_m":           2.0,   # HARD FAIL
    "shoulder_paved_min_m":           1.0,   # HARD FAIL
}

# Table IRC92-06 — Clearances at Interchanges
INTERCHANGE_CLEARANCE = {
    "underpass_vertical_min_m":       5.5,   # HARD FAIL (urban areas)
    "lateral_clearance":              "equal to approach shoulder width",  # HARD FAIL
    "gore_area_obstructions":         False,  # HARD FAIL (core area must be hazard-free)
}

# Table IRC92-07 — Interchange Spacing
INTERCHANGE_SPACING = {
    "urban_min_m":             1600,    # HARD FAIL if < 1.6 km without collector-distributor
    "rural_min_m":             4800,    # ADVISORY if < 4.8 km; HARD FAIL if < 1.6 km
}

# Table IRC92-08 — Weaving Section
WEAVING_SECTION = {
    "max_defined_length_m":        450,   # HARD FAIL if exceeds 450 m
    "influence_area_upstream_m":   450,   # INFO (for capacity analysis)
    "influence_area_downstream_m": 450,   # INFO
}

# Table IRC92-09 — Ramp Metering
RAMP_METERING = {
    "sov_single_meter_max_vph":   720,   # INFO
    "hov_warrant_pct":              9,   # ADVISORY (HOV lane if HOV ≥ 9% of peak volume)
}

# Table IRC92-10 — Interchange Type and Land Requirements
INTERCHANGE_LAND = {
    "trumpet_m2":        44000,    # INFO
    "diamond_m2":        28000,    # INFO
    "full_cloverleaf_m2": 73000,   # INFO
    "bridged_rotary_m2": 180000,   # INFO
    "warrant_pcu_per_hr": 10000,   # ADVISORY — interchange warranted above this
}

# Table IRC92-11 — Lane Balance at Interchanges
LANE_BALANCE = {
    # Merge: downstream ≥ sum_of_merging − 1
    "merge_min_downstream":    "sum_of_merging_lanes - 1",   # HARD FAIL
    # Diverge: diverging lanes = (through beyond exit) + (exit lanes) − 1
    "diverge_formula":         "(through_beyond_exit) + (exit_lanes) - 1",
    "max_lane_reduction_per_point": 1,   # HARD FAIL (not more than 1 lane at a time)
    "basic_lane_continuity":   True,     # HARD FAIL (maintain basic lanes through interchanges)
}

# Table IRC92-12 — NMT at Interchanges
NMT_INTERCHANGE = {
    "nmt_vertical_sep_min_mm":          100,  # HARD FAIL (minimum raised curb)
    "cycle_track_width_min_m":          1.5,  # HARD FAIL
    "cycle_track_width_desirable_m":    2.5,  # ADVISORY
    "buffer_zone_m":                    (0.5, 1.0),   # ADVISORY
    "max_speed_at_nmt_crossing_kmh":    50,   # HARD FAIL
    "expressway_nmt_grade_sep":         True, # HARD FAIL (at-grade crossing prohibited)
}

# Table IRC92-13 — Illumination at Interchanges
INTERCHANGE_ILLUMINATION = {
    "full_illumination_required":  True,     # HARD FAIL
    "design_standard":             "IRC:SP:90",
    "noise_barrier_trigger":       "adjacent sensitive land uses",  # ADVISORY
    "noise_barrier_ht_visibility_impact_m": 1.5,  # affects motorist visibility, ADVISORY
}


# ─────────────────────────────────────────────────────────────────────────────
# PENDING PARAMETERS (not yet resolved — cross-referenced to other standards)
# ─────────────────────────────────────────────────────────────────────────────

PENDING = {
    "9-P1": {"desc": "Sign gantry vertical clearance", "source": "IRC:67", "clause": "9.2.4"},
    "9-P2": {"desc": "Sign gantry lateral clearance (post offset)", "source": "IRC:67", "clause": "9.2.4"},
    "9-P3": {"desc": "Sign advance placement distances by design speed", "source": "IRC:67", "clause": "9.2.9"},
    "9-P4": {"desc": "Chevron sign spacing for R ≤ 450 m", "source": "IRC:67", "clause": "9.2.7"},
    "12-P1": {"desc": "Street light pole spacing", "source": "IRC:32", "clause": "12.5.2"},
    "12-P2": {"desc": "Luminaire mounting height", "source": "IRC:32", "clause": "12.5.2"},
    "12-P3": {"desc": "Bracket arm / overhang length", "source": "IRC:32", "clause": "12.5.2"},
    "14-P1": {"desc": "Tunnel interior zone luminance (Lux/cd/m²)", "source": "MoRTH Ch.13.5", "clause": "14.8"},
    "14-P2": {"desc": "Tunnel threshold/transition zone luminance curve", "source": "MoRTH Ch.13.5", "clause": "14.8"},
    "14-P3": {"desc": "Tunnel ventilation CO limit, air velocity, visibility threshold", "source": "IRC:SP:91", "clause": "14.7.3"},
}


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

PARAMETER_COUNT = {
    "section_2":    53,
    "section_3":     9,
    "section_4":     3,
    "section_5":     4,
    "section_6":     9,
    "section_7":    19,
    "section_8":     2,
    "section_9":    88,
    "section_10":   31,
    "section_11":   17,
    "section_12":   29,
    "section_13":   19,
    "section_14":   52,
    "supp_vc":      10,
    "supp_parking":  9,
    "supp_irc92":   78,
    "total":        432,
    "hard_fail":    300,
    "advisory":      80,
    "info":          52,
    "pending":        8,
}
