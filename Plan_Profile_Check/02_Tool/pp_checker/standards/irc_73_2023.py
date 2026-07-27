"""
irc_73_2023.py
==============
IRC:73-2023 — Geometric Design Standards for Rural Highways
All design parameters as Python dictionaries for use in the
IRC Geometric Design Compliance Checker rule engine.

Source  : IRC:73-2023 (Sections 1-10)
Scope   : Non-urban roads — NH, SH, MDR, ODR, VR
Excludes: Urban roads, intersections (Clause 1.3)
Standard: Rule-based engine (NOT ML/AI)

Key     : HARD FAIL  = non-compliant, must be redesigned
          ADVISORY   = flag for designer review / justification
          PASS       = compliant

Usage:
    from data.irc_73_2023 import (
        DESIGN_SPEED, SSD_TABLE, MIN_RADIUS_NH_SH,
        GRADIENT_HIGHWAYS, K_SUMMIT, K_VALLEY, ...
    )
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CONTROL FACTORS
# ─────────────────────────────────────────────────────────────────────────────

# Clause 3.1 — Terrain Classification
# Key: terrain type (str, lowercase)
# Value: cross-slope range description
TERRAIN_CLASSIFICATION = {
    "plain":       {"cross_slope_pct": (0,   10),  "ratio": "1 in 10 or flatter"},
    "rolling":     {"cross_slope_pct": (10,  25),  "ratio": "1 in 10 to 1 in 4"},
    "mountainous": {"cross_slope_pct": (25,  60),  "ratio": "1 in 4 to 1 in 1.67"},
    "steep":       {"cross_slope_pct": (60, 100),  "ratio": "steeper than 1 in 1.67"},
}
# Rule: short isolated stretches < 1 km should not drive terrain classification.

# Clause 3.2 — Design Speeds (km/h)
# Key: (road_category, terrain)  — all lowercase
# Value: {"ruling": int, "minimum": int}
DESIGN_SPEED = {
    # 4/6/8-Lane NH/SH (also used for IRC:SP:84 4-lane)
    ("4_lane_nh_sh", "plain"):       {"ruling": 100, "minimum": 80},
    ("4_lane_nh_sh", "rolling"):     {"ruling": 100, "minimum": 80},
    ("4_lane_nh_sh", "mountainous"): {"ruling":  60, "minimum": 40},
    ("4_lane_nh_sh", "steep"):       {"ruling":  60, "minimum": 40},

    # 2-Lane NH/SH
    ("2_lane_nh_sh", "plain"):       {"ruling": 100, "minimum": 80},
    ("2_lane_nh_sh", "rolling"):     {"ruling": 100, "minimum": 80},
    ("2_lane_nh_sh", "mountainous"): {"ruling":  50, "minimum": 40},
    ("2_lane_nh_sh", "steep"):       {"ruling":  40, "minimum": 30},

    # MDR
    ("mdr", "plain"):                {"ruling":  80, "minimum": 65},
    ("mdr", "rolling"):              {"ruling":  65, "minimum": 50},
    ("mdr", "mountainous"):          {"ruling":  40, "minimum": 30},
    ("mdr", "steep"):                {"ruling":  30, "minimum": 20},

    # ODR / VR
    ("odr_vr", "plain"):             {"ruling":  50, "minimum": 40},
    ("odr_vr", "rolling"):           {"ruling":  50, "minimum": 40},
    ("odr_vr", "mountainous"):       {"ruling":  30, "minimum": 20},
    ("odr_vr", "steep"):             {"ruling":  30, "minimum": 20},

    # Special
    ("expressway", "any"):           {"ruling": 120, "minimum": 120},
    ("service_road", "any"):         {"ruling":  40, "minimum":  40},
}
# Rule engine logic:
#   speed < minimum  → HARD FAIL
#   minimum <= speed < ruling  → ADVISORY ("below ruling, justify on record")
#   speed >= ruling  → PASS

# Clause 3.3.2 — Design Vehicle Dimensions (m)
DESIGN_VEHICLE = {
    "standard": {
        "max_width":          2.6,
        "max_height":         4.0,
        "max_length_truck":   12.0,    # N3 single truck
        "max_length_bus":     15.0,    # M3 single bus
        "max_length_trailer": 18.75,   # truck-trailer / semi-trailer
    },
    "construction_equipment": {
        "max_width":  3.0,
        "max_height": 4.75,
    },
}

# Clause 3.3.2 — Minimum Turning Diameter (swept path, m)
MIN_TURNING_DIAMETER = {
    "psv_lte_8.3m":          20,
    "psv_8_to_11m":          22,
    "psv_gt_11m":            24,
    "commercial_range":      (12, 21),
    "private_car_range":     (8,  14),
    "general_design_min":    26,
    "mountain_semi_trailer": 21,    # acceptable minimum
}

# Clause 3.3.3 — PIEV / Reaction Time (seconds)
REACTION_TIME = {
    "ssd": 2.5,   # Stopping Sight Distance
    "osd": 2.0,   # Overtaking Sight Distance
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CROSS-SECTIONAL ELEMENTS
# ─────────────────────────────────────────────────────────────────────────────

# Clause 4.1, Table 4.1 — Minimum ROW for Highways & Expressways (m)
# Value: {"normal": x} or {"normal": x, "exceptional": y}
MIN_ROW_HIGHWAY = {
    "2_lane":                          {"normal": 30},
    "4_lane":                          {"normal": 45},
    "6_lane":                          {"normal": 60},
    "8_lane":                          {"normal": 120},
    "expressway":                      {"normal": 90, "maximum": 120},
    "2_lane_bypass":                   {"normal": 45, "maximum": 60},
    "2_lane_mountain_open":            {"normal": 24, "exceptional": 18},
    "2_lane_mountain_builtup":         {"normal": 20, "exceptional": 18},
}

# Clause 4.1, Table 4.2 — Minimum ROW for MDR / ODR / VR (m)
# Key: (road_class, terrain_group, area_type)
# terrain_group: "plain_rolling" or "mountain_steep"
# area_type: "open" or "builtup"
# Value: {"normal": x, "exceptional": y}
MIN_ROW_MDR = {
    ("mdr",    "plain_rolling", "open"):    {"normal": 25, "range": (25, 30)},
    ("mdr",    "plain_rolling", "builtup"): {"normal": 20, "range": (15, 25)},
    ("mdr",    "mountain_steep","open"):    {"normal": 18, "exceptional": 15},
    ("mdr",    "mountain_steep","builtup"): {"normal": 15, "exceptional": 12},

    ("odr",    "plain_rolling", "open"):    {"normal": 15, "range": (15, 25)},
    ("odr",    "plain_rolling", "builtup"): {"normal": 15, "range": (15, 20)},
    ("odr",    "mountain_steep","open"):    {"normal": 15, "exceptional": 12},
    ("odr",    "mountain_steep","builtup"): {"normal": 12, "exceptional":  9},

    ("vr",     "plain_rolling", "open"):    {"normal": 12, "range": (12, 18)},
    ("vr",     "plain_rolling", "builtup"): {"normal": 10, "range": (10, 15)},
    ("vr",     "mountain_steep","open"):    {"normal":  9, "exceptional":  9},
    ("vr",     "mountain_steep","builtup"): {"normal":  9, "exceptional":  9},
}
# Rule: below normal → ADVISORY; below exceptional → HARD FAIL

# Clause 4.2, Table 4.3 — Lane / Carriageway Width (m)
LANE_WIDTH = {
    "single_lane":          3.75,
    "multilane_highway":    3.50,   # per lane — NH, SH, MDR, ODR, VR
    "multilane_expressway": 3.75,   # per lane
    "hilly_2lane_min":      7.00,   # minimum 2-lane carriageway in hilly areas
    "intermediate_2lane":   5.50,   # permitted on non-trunk routes (advisory)
}
# Width transition tapers:
WIDTH_TAPER = {
    "narrow_to_wide": "1 in 15",
    "wide_to_narrow": "1 in 20",
    "4lane_to_6_8lane": "1 in 50",
}

# Clause 4.4, Table 4.4 — Shoulder Width, Plain & Rolling, 2-Lane (m)
# Key: section_type
# Value: {"paved": x, "earthen": y, "total": z}
SHOULDER_2LANE_PLAIN_ROLLING = {
    "open_country":                  {"paved": 1.5, "earthen": 1.0, "total": 2.5},
    "builtup_2lane":                 {"paved": 2.5, "earthen": 0.0, "total": 2.5},
    "grade_separated_approach":      {"paved": 1.5, "earthen": 0.0, "total": 1.5},
    "bridge_approach":               {"paved": 1.5, "earthen": 1.0, "total": 2.5},
}

# Clause 4.4, Table 4.5 — Shoulder Width, Plain & Rolling, 4/6/8-Lane (m)
SHOULDER_MULTILANE_PLAIN_ROLLING = {
    "open_country":                  {"paved": 2.0, "earthen": 1.5, "total": 3.5},
    "builtup":                       {"paved": 2.0, "earthen": 0.0, "total": 2.0},
    "grade_separated_approach":      {"paved": 2.0, "earthen": 0.0, "total": 2.0},
    "bridge_approach":               {"paved": 2.0, "earthen": 1.5, "total": 3.5},
}

# Clause 4.4, Table 4.6 — Shoulder Width, Mountainous & Steep (m)
# Key: (section_type, side)
SHOULDER_MOUNTAIN_STEEP = {
    ("open_country", "hill"):    {"paved": 1.5,  "earthen": 0.0, "total": 1.5},
    ("open_country", "valley"):  {"paved": 1.5,  "earthen": 1.0, "total": 2.5},
    ("builtup_or_structure", "hill"):   {"paved": 1.75, "earthen": 0.0, "total": 1.75},
    ("builtup_or_structure", "valley"): {"paved": 1.75, "earthen": 0.0, "total": 1.75},
}
# Note: MDR/ODR/VR mountain/steep: min shoulder = 1.5 m each side (0.5 paved + 1.0 earthen)
SHOULDER_MDR_ODR_VR = {
    "plain_rolling": {"paved": 0.5, "earthen": 1.5, "total": 2.0},
    "mountain_steep": {"paved": 0.5, "earthen": 1.0, "total": 1.5},
    "min_roadway_width": 7.0,   # carriageway + both shoulders minimum
}

# Clause 4.7 — Camber / Cross Fall (%)
CAMBER = {
    "bituminous": 2.5,
    "cement_concrete": 2.0,
    "earthen_shoulder_min_steeper_than_pavement": 0.5,
    "earthen_shoulder_desirable_steeper": 1.0,
    "expressway_earthen_shoulder_min_steeper": 1.0,
    "max_rollover_angle_superelevated": 7.5,
    "reverse_crossfall_outer_shoulder": 0.5,
}

# Clause 4.3 — Median Width (m)
MEDIAN_WIDTH = {
    "4_lane_plain_rolling_min":     7.0,
    "4_lane_mountain_steep_min":    2.5,   # flush with collapsible crash barrier
    "6_8_lane_min":                 7.0,
    "6_8_lane_desirable":          12.0,
    "expressway_desirable":        15.0,
}
# Rule: median width check = ADVISORY


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SIGHT DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

# Clause 5.2, Table 5.1 — Measurement Criteria
SIGHT_DISTANCE_CRITERIA = {
    "driver_eye_height_m":   1.2,
    "ssd_object_height_m":   0.15,
    "isd_object_height_m":   1.2,
    "osd_object_height_m":   1.2,
    # For valley curve headlight check:
    "headlight_height_m":    0.75,
    "headlight_beam_angle_deg": 1.0,
    "valley_object_height_m": 0.0,
}

# Clause 5.3, Table 5.2 — SSD Design Values
# Key: design_speed (km/h)
# Value: {"friction": f, "ssd": m, "isd": m}
# Formula: SSD = 0.278 * V * t + V^2/(254*f)   t=2.5s
SSD_TABLE = {
    20:  {"friction": 0.40, "ssd":  20, "isd":  40},
    25:  {"friction": 0.40, "ssd":  25, "isd":  50},
    30:  {"friction": 0.40, "ssd":  30, "isd":  60},
    40:  {"friction": 0.38, "ssd":  45, "isd":  90},
    50:  {"friction": 0.37, "ssd":  60, "isd": 120},
    60:  {"friction": 0.36, "ssd":  80, "isd": 160},
    65:  {"friction": 0.36, "ssd":  90, "isd": 180},
    80:  {"friction": 0.35, "ssd": 130, "isd": 260},
    100: {"friction": 0.35, "ssd": 180, "isd": 360},
    120: {"friction": 0.35, "ssd": 250, "isd": 500},
}
# Rule: available < ssd → HARD FAIL
#       ssd <= available < isd → ADVISORY

# Clause 5.4.2, Table 5.3 — Overtaking Sight Distance (m)
OSD_TABLE = {
    40:  165,
    50:  235,
    60:  300,
    65:  340,
    80:  470,
    100: 640,
}
# Application notes:
#   2-lane undivided: provide OSD wherever possible; else ISD throughout
#   4/6/8-lane divided: OSD not required; ISD throughout
#   Expressways: 500 m minimum SSD throughout (Clause 5.6.4)
#   Grade-change locations (VUP, ROB, bridges): use ISD (Clause 5.6.5)

# Clause 5.7.4, Table 5.4 — Intersection Time Gap (s)
INTERSECTION_TIME_GAP = {
    "passenger_car": 8.0,
    "heavy_vehicle": 10.0,
    "additional_per_lane_car":   0.5,
    "additional_per_lane_truck": 0.7,
}

# Clause 5.7.4.1, Table 5.5 — Departure Sight Triangle, Stop-Controlled (m)
# Leg 'b' along Major Road for Passenger Car
# Key: major_road_speed (km/h)
# Value: {"2_lane": m, "4_lane": m}
DEPARTURE_SIGHT_TRIANGLE_STOP = {
    20:  {"2_lane":  45, "4_lane":  55},
    25:  {"2_lane":  55, "4_lane":  70},
    30:  {"2_lane":  70, "4_lane":  85},
    40:  {"2_lane":  90, "4_lane": 110},
    50:  {"2_lane": 110, "4_lane": 140},
    60:  {"2_lane": 135, "4_lane": 170},
    65:  {"2_lane": 145, "4_lane": 180},
    80:  {"2_lane": 180, "4_lane": 220},
    100: {"2_lane": 220, "4_lane": 280},
}

# Clause 5.7.4.2, Table 5.6 — Yield-Controlled Sight Triangle, 2-Lane Major Road (m)
# Key: (minor_road_speed, major_road_speed)
# Value: leg 'b' in metres; None = not applicable
YIELD_SIGHT_TRIANGLE_2LANE = {
    (20, 40): 120, (20, 50): 150, (20, 60): 180, (20, 65): 195, (20, 80): 240, (20, 100): 300,
    (25, 40): 105, (25, 50): 135, (25, 60): 160, (25, 65): 175, (25, 80): 215, (25, 100): 270,
    (30, 40): 100, (30, 50): 120, (30, 60): 145, (30, 65): 160, (30, 80): 195, (30, 100): 245,
    (40, 40):  95, (40, 50): 115, (40, 60): 140, (40, 65): 150, (40, 80): 185, (40, 100): 235,
    (50, 40): None,(50, 50): 110, (50, 60): 135, (50, 65): 145, (50, 80): 180, (50, 100): 225,
    (60, 40): None,(60, 50): None,(60, 60): 140, (60, 65): 150, (60, 80): 185, (60, 100): 230,
}

# Clause 5.7.4.2, Table 5.7 — Yield-Controlled Sight Triangle, 4-Lane Major Road (m)
YIELD_SIGHT_TRIANGLE_4LANE = {
    (20, 40): 125, (20, 50): 155, (20, 60): 190, (20, 65): 205, (20, 80): 250, (20, 100): 315,
    (25, 40): 110, (25, 50): 140, (25, 60): 165, (25, 65): 180, (25, 80): 220, (25, 100): 280,
    (30, 40): 110, (30, 50): 140, (30, 60): 165, (30, 65): 180, (30, 80): 220, (30, 100): 280,
    (40, 40): 110, (40, 50): 140, (40, 60): 165, (40, 65): 180, (40, 80): 220, (40, 100): 280,
    (50, 40): None,(50, 50): 140, (50, 60): 165, (50, 65): 180, (50, 80): 220, (50, 100): 280,
    (60, 40): None,(60, 50): None,(60, 60): 165, (60, 65): 180, (60, 80): 220, (60, 100): 280,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HORIZONTAL ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

# Clause 6.3.2 — Lateral Friction Coefficient
LATERAL_FRICTION = 0.15   # fixed constant for all speeds

# Clause 6.3.1 — Equilibrium Equation
# e + f = V^2 / (127 * R)   →   R_min = V^2 / (127 * (e_max + f))
# e = V^2 / (225 * R)       →   superelevation formula

# Clause 6.4.3, Table 6.3 — Maximum Superelevation (e_max)
E_MAX = {
    "plain_rolling":         0.07,   # 7%
    "snow_bound":            0.07,   # 7%
    "hilly_not_snow_bound":  0.10,   # 10%
    # For 2/4/6/8-lane highways:
    "below_desirable_min_radius": 0.07,
    "above_desirable_min_radius": 0.05,
    "urban_or_major_junction":    0.05,
}

# Clause 6.3.3, Table 6.1 — Horizontal Curve Radii for EW / NH / SH (m)
MIN_RADIUS_NH_SH = {
    ("expressway",   "plain_rolling"):    {"desirable": 1000, "absolute": 650},
    ("nh_sh",        "plain_rolling"):    {"desirable":  400, "absolute": 250},
    ("nh_sh",        "mountain_steep"):   {"desirable":  150, "absolute":  75},
}
# Rule: R < absolute → HARD FAIL
#       absolute <= R < desirable → ADVISORY (confirm e capped at 7%)
#       R >= desirable → PASS (e capped at 5%)

# Clause 6.3.3, Table 6.2 — Horizontal Curve Radii for MDR / ODR / VR (m)
# Key: (road_class, terrain, snow_bound)
# Value: {"ruling": x, "absolute": y}
MIN_RADIUS_MDR = {
    ("mdr", "plain",        False): {"ruling": 230, "absolute": 155},
    ("mdr", "rolling",      False): {"ruling": 155, "absolute":  90},
    ("mdr", "mountainous",  False): {"ruling":  50, "absolute":  30},
    ("mdr", "mountainous",  True):  {"ruling":  60, "absolute":  33},
    ("mdr", "steep",        False): {"ruling":  30, "absolute":  14},
    ("mdr", "steep",        True):  {"ruling":  33, "absolute":  15},

    ("odr", "plain",        False): {"ruling":  90, "absolute":  60},
    ("odr", "rolling",      False): {"ruling":  90, "absolute":  60},
    ("odr", "mountainous",  False): {"ruling":  30, "absolute":  13},
    ("odr", "mountainous",  True):  {"ruling":  33, "absolute":  15},
    ("odr", "steep",        False): {"ruling":  30, "absolute":  13},
    ("odr", "steep",        True):  {"ruling":  33, "absolute":  15},
    # VR same as ODR
    ("vr",  "plain",        False): {"ruling":  90, "absolute":  60},
    ("vr",  "rolling",      False): {"ruling":  90, "absolute":  60},
    ("vr",  "mountainous",  False): {"ruling":  30, "absolute":  13},
    ("vr",  "mountainous",  True):  {"ruling":  33, "absolute":  15},
    ("vr",  "steep",        False): {"ruling":  30, "absolute":  13},
    ("vr",  "steep",        True):  {"ruling":  33, "absolute":  15},
}

# Clause 6.4.4, Table 6.4 — Radii Beyond Which SE / Transition Not Required (m)
# Key: (design_speed, camber_pct)
# Value: threshold radius (m); if R > this value, normal camber OK, no SE needed
SE_NOT_REQUIRED_RADIUS = {
    (20,  4.0): 50,  (20,  3.0): 60,  (20,  2.5): 70,  (20,  2.0): 90,  (20,  1.7): 100,
    (25,  4.0): 70,  (25,  3.0): 90,  (25,  2.5): 110, (25,  2.0): 140, (25,  1.7): 160,
    (30,  4.0): 100, (30,  3.0): 130, (30,  2.5): 160, (30,  2.0): 200, (30,  1.7): 240,
    (35,  4.0): 140, (35,  3.0): 180, (35,  2.5): 220, (35,  2.0): 270, (35,  1.7): 320,
    (40,  4.0): 180, (40,  3.0): 240, (40,  2.5): 280, (40,  2.0): 360, (40,  1.7): 420,
    (50,  4.0): 280, (50,  3.0): 370, (50,  2.5): 450, (50,  2.0): 560, (50,  1.7): 650,
    (65,  4.0): 470, (65,  3.0): 630, (65,  2.5): 750, (65,  2.0): 940, (65,  1.7): 1100,
    (80,  4.0): 710, (80,  3.0): 950, (80,  2.5): 1140,(80,  2.0): 1420,(80,  1.7): 1670,
    (100, 4.0): 1110,(100, 3.0): 1480,(100, 2.5): 1780,(100, 2.0): 2220,(100, 1.7): 2610,
}

# Clause 6.4.5, Table 6.5 — Rate of Change of Superelevation
SE_RATE_OF_CHANGE = {
    "plain_rolling":    150,   # 1 in 150
    "mountain_steep":    60,   # 1 in 60
}
# Rule: if actual rate > 1:N → ADVISORY

# Clause 6.5.2 — Extra Widening Formula
# W_e = n*l^2/(2*R) + V/(9.5*sqrt(R))
# n = number of lanes, l = 6 m (design wheelbase), V = speed, R = radius
EXTRA_WIDENING_WHEELBASE = 6.0   # metres (standard design vehicle)

# Clause 6.5.2, Table 6.6 — Extra Widening for 2/4/6/8-Lane Highways (m)
# Key: radius range string
EXTRA_WIDENING_HIGHWAYS = {
    (75,  100): 0.9,   # radius 75-100 m
    (101, 300): 0.6,   # radius 101-300 m
    # radius > 300: no extra widening required
}

# Clause 6.5.2, Table 6.7 — Extra Widening for MDR/ODR/VR (m)
# Key: (lane_type, radius_upper_bound)
EXTRA_WIDENING_MDR = {
    ("two_lane",    20):  1.5,
    ("two_lane",    40):  1.5,
    ("two_lane",    60):  1.2,
    ("two_lane",   100):  0.9,
    ("two_lane",   300):  0.6,
    ("single_lane", 20):  0.9,
    ("single_lane", 40):  0.6,
    ("single_lane", 60):  0.6,
    # above 100 m single lane: nil
}

# Clause 6.7 — Transition Curve (Clothoid / Spiral)
# Rate of change of centrifugal acceleration: C = 80/(75+V)
# Subject to: 0.5 <= C <= 0.8
# Transition length criteria (use LARGER):
#   (i)  Ls = 0.0215 * V^3 / (C * R)
#   (ii) Ls = (e * N / 2) * (W + We)   [rotation about centreline]
#       or Ls = e * N * (W + We)         [rotation about inner edge]
TRANSITION_SE_RATE = {
    "plain_rolling":   150,   # N value: 1 in 150
    "mountain_steep":   60,   # N value: 1 in 60
}

# Clause 6.7.5 — Same as Table 6.4 — if R > threshold, no transition required

# Clause 6.8 — Compound Curves
COMPOUND_CURVE = {
    "max_radius_ratio":          1.5,    # R_flatter <= 1.5 * R_sharper (HARD FAIL)
    "min_transition_if_diff_gt_50pct": 0,  # transition required (HARD FAIL if missing)
    "min_transition_length":    30,      # minimum 30 m if radius diff < 50%
}

# Clause 6.10 — Hairpin Bend Parameters
HAIRPIN_BEND = {
    "min_design_speed_kmph":     20,
    "min_inner_radius_m":        14,
    "min_transition_length_m":   15,
    "max_gradient_pct":           2.5,
    "min_gradient_pct":           0.5,
    "superelevation_pct":        10,
    "min_straight_between_bends_m": 60,
    "min_roadway_width": {
        "nh_sh_double_lane": 11.5,
        "nh_sh_single_lane":  9.0,
        "mdr_odr":            7.5,
        "vr":                 6.5,
    },
}
# All above = HARD FAIL thresholds

# Clause 6.1 — General Alignment Principles (quantitative rules)
ALIGNMENT_GENERAL = {
    "max_straight_tangent_m":   3000,   # ADVISORY if exceeded
    "min_curve_length_5deg_m":   150,   # for 5° deflection; +30m per 1° below 5°
    "broken_back_min_time_s":     10,   # seconds travel time between curves (same dir)
    "compound_max_radius_ratio":  1.5,  # R_flatter / R_sharper (HARD FAIL)
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VERTICAL ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

# Clause 7.2, Table 7.1 — Gradients for 2/4/6/8-Lane Highways (%)
GRADIENT_HIGHWAYS = {
    "plain_rolling":    {"ruling": 2.5, "limiting": 3.3},
    "mountainous":      {"ruling": 5.0, "limiting": 6.0},
    "steep":            {"ruling": 6.0, "limiting": 7.0},
    "expressway":       {"ruling": 2.0, "limiting": 3.0},
}
# Rule: > limiting → HARD FAIL
#       > ruling and <= limiting → ADVISORY
#       <= ruling → PASS

# Clause 7.2, Table 7.2 — Gradients for MDR / ODR / VR (%)
GRADIENT_MDR = {
    "plain_rolling":              {"ruling": 3.3, "limiting": 5.0, "exceptional": 6.0},
    "mountainous_gt_3000m_steep": {"ruling": 5.0, "limiting": 7.0, "exceptional": 10.0},
    "steep_lte_3000m":            {"ruling": 6.0, "limiting": 8.0, "exceptional": 10.0},
}
# Rule: > exceptional → HARD FAIL
#       > limiting and <= exceptional → ADVISORY (max 100 m continuous)
#       > ruling and <= limiting → ADVISORY
#       <= ruling → PASS
EXCEPTIONAL_GRADIENT_MAX_LENGTH_M = 100   # max continuous length at exceptional grade

# Clause 7.5.2 — Summit Curve Length Formulas
# L > S:  L = N*S^2/4.4   (SSD);   L = N*S^2/9.6  (ISD/OSD)
# L < S:  L = 2S - 4.4/N  (SSD);   L = 2S - 9.6/N (ISD/OSD)
SUMMIT_CURVE_CONSTANTS = {
    "ssd": {"l_gt_s": 4.4, "l_lt_s": 4.4},
    "isd": {"l_gt_s": 9.6, "l_lt_s": 9.6},
    "osd": {"l_gt_s": 9.6, "l_lt_s": 9.6},
}

# Clause 7.6.2 — Valley Curve Length Formulas (headlight SSD)
# L > S:  L = N*S^2 / (1.50 + 0.035*S)
# L < S:  L = 2S - (1.50 + 0.035*S) / N
VALLEY_CURVE_HEADLIGHT = {
    "h1_m":          0.75,   # headlight height
    "beam_angle_deg": 1.0,
    "a_const":        1.50,
    "b_const":        0.035,
}
# Formula: denominator = 1.50 + 0.035*S

# Clause 7.5.3, Table 7.5 — Minimum Length of Vertical Curve (m)
# Key: design_speed (km/h)
# Value: {"max_N_no_vc": %, "min_L": m, "expressway_min_L": m}
MIN_VC_LENGTH = {
    20:  {"max_N_no_vc": 1.5, "min_L": 15},
    25:  {"max_N_no_vc": 1.5, "min_L": 15},
    30:  {"max_N_no_vc": 1.5, "min_L": 15},
    35:  {"max_N_no_vc": 1.5, "min_L": 15},
    40:  {"max_N_no_vc": 1.2, "min_L": 20},
    50:  {"max_N_no_vc": 1.0, "min_L": 30},
    60:  {"max_N_no_vc": 0.8, "min_L": 40},
    80:  {"max_N_no_vc": 0.6, "min_L": 50,  "expressway_min_L":  70},
    100: {"max_N_no_vc": 0.5, "min_L": 60,  "expressway_min_L":  85},
    120: {"max_N_no_vc": 0.5, "min_L": 100, "expressway_min_L": 100},
}
# Rule: N <= max_N_no_vc → no VC required → PASS
#       N > max_N_no_vc and no VC → HARD FAIL
#       VC provided but L < min_L → HARD FAIL
#       Required L = max(sight_distance_formula_L, min_L)

# Clause 7.5.2, Table 7.3 — K-values (L > S case), coefficient only
# Multiply by N% to get L. K = L/N.
# Derived: K_summit_ssd = S^2/4.4 ; K_valley = S^2/(1.50+0.035*S)
K_SUMMIT_SSD = {
    20:  0.91,
    25:  1.42,
    30:  2.05,
    40:  4.60,
    50:  8.18,
    60:  14.55,
    65:  18.41,
    80:  38.41,
    100: 73.64,
    120: 142.05,
}
# Values represent K in m per 1% of N (i.e., L = K * N_pct)

# K_SUMMIT_ISD — Summit curve K-values based on ISD (governing for IRC:73 roads)
# K = ISD² / 9.6  (ISD from SSD_TABLE isd column)
# IRC:73-2023 Clause 7.5.2: ISD governs for new construction on 2-lane roads
K_SUMMIT_ISD = {
    20:   1.67,   # ISD=40m:   40²/9.6=167
    25:   2.60,   # ISD=50m:   50²/9.6=260
    30:   3.75,   # ISD=60m:   60²/9.6=375
    40:   8.44,   # ISD=90m:   90²/9.6=844
    50:  15.00,   # ISD=120m: 120²/9.6=1500
    60:  26.67,   # ISD=160m: 160²/9.6=2667
    65:  33.75,   # ISD=180m: 180²/9.6=3375
    80:  70.42,   # ISD=260m: 260²/9.6=7042
    100: 135.00,  # ISD=360m: 360²/9.6=13500
    120: 260.42,  # ISD=500m: 500²/9.6=26042
}

K_VALLEY = {
    20:  1.80,
    25:  2.60,
    30:  3.50,
    40:  6.60,
    50:  10.0,
    60:  14.90,
    65:  17.40,
    80:  27.90,
    100: 41.50,
    120: 61.0,
}

# Clause 7.3 — Practical Vertical Alignment Rules
VERTICAL_PRACTICAL = {
    "min_distance_between_grade_changes_m":     150,  # ADVISORY if < 150m
    "min_distance_mountain_m":                   75,  # mountainous terrain
    "min_distance_overlay_m":                    90,  # overlay sections
}

# Grade Compensation at Horizontal Curves (Clause 8.5)
# GC (%) = min[(30+R)/R, 75/R]
# Apply only when gradient > 4%; compensated gradient = original - GC
GRADE_COMPENSATION = {
    "trigger_gradient_pct": 4.0,
    "formula": "min((30+R)/R, 75/R)",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — ALIGNMENT COORDINATION (Quantitative Rules)
# ─────────────────────────────────────────────────────────────────────────────

COORDINATION = {
    "max_tangent_length_factor":   20,   # max tangent (m) = 20 * V (km/h)  ADVISORY
    "min_tangent_length_factor":    4,   # min tangent (m) = 4 * V           ADVISORY
    "min_hcurve_length_factor":     3,   # HARD FAIL if < 3*V
    "preferred_hcurve_length_factor": 6, # ADVISORY if 3V <= L < 6V
    "min_cross_slope_drainage_pct": 0.5, # HARD FAIL if flat spot < 0.5%
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — LATERAL AND VERTICAL CLEARANCES
# ─────────────────────────────────────────────────────────────────────────────

# Clause 9.2.1, Table 9.1 — Underpass / Overpass Horizontal Span (m)
MIN_HORIZONTAL_SPAN = {
    "vup":   {"standard": 20.0, "two_lane_min": 12.0},
    "lvup":  {"standard": 12.0, "with_footpath": 10.5},
    "svup":  {"standard":  7.0},
    "pup":   {"standard":  7.0},
    "cup":   {"standard":  7.0},
    "vop":   {"standard": "full_roadway_width"},
}
LATERAL_CLEARANCE_UNDERPASS = {
    "mdr_odr_normal_m":         2.0,
    "mdr_odr_exceptional_m":    1.5,
    "pier_desirable_m":         2.0,
    "pier_min_m":               1.5,
    "pier_with_kerbed_median_desirable_m": 1.5,
    "pier_with_kerbed_median_exceptional_m": 1.0,
}

# Clause 9.3.1, Table 9.2 — Minimum Vertical Clearance (m)
MIN_VERTICAL_CLEARANCE = {
    "vup":  5.5,
    "lvup": 4.0,
    "svup": 4.0,
    "pup":  3.0,
    "cup":  3.0,
    "vop":  5.5,
}

# Clause 9.3.2, Table 9.3 — AC Overhead Line Vertical Clearance (m)
# Key: (voltage_kv_str, orientation)
# orientation: "across_street" | "along_street" | "elsewhere" | "across_highway"
AC_LINE_CLEARANCE = {
    ("0.65",   "across_street"): 5.80, ("0.65",   "along_street"): 5.50, ("0.65",   "elsewhere"): 4.60,
    ("11",     "across_street"): 6.50, ("11",     "along_street"): 5.80, ("11",     "elsewhere"): 4.60,
    ("22",     "across_street"): 6.50, ("22",     "along_street"): 5.80, ("22",     "elsewhere"): 5.20,
    ("33",     "across_street"): 6.50, ("33",     "along_street"): 5.80, ("33",     "elsewhere"): 5.20,  ("33",  "across_highway"): 11.60,
    ("66",     "across_street"): 6.50, ("66",     "along_street"): 6.10, ("66",     "elsewhere"): 5.50,  ("66",  "across_highway"): 11.60,
    ("110",    "across_street"): 6.50, ("110",    "along_street"): 6.10, ("110",    "elsewhere"): 6.10,  ("110", "across_highway"): 11.60,
    ("132",    "across_street"): 6.50, ("132",    "along_street"): 6.10, ("132",    "elsewhere"): 6.10,  ("132", "across_highway"): 11.60,
    ("220",    "across_street"): 7.02, ("220",    "along_street"): 7.02, ("220",    "elsewhere"): 7.02,  ("220", "across_highway"): 12.52,
    ("400",    "across_street"): 8.84, ("400",    "along_street"): 8.84, ("400",    "elsewhere"): 8.84,  ("400", "across_highway"): 14.00,
    ("765",    "across_street"): 18.00,("765",    "along_street"): 18.00,("765",    "elsewhere"): 18.00, ("765", "across_highway"): 18.80,
    ("1200",   "across_street"): 24.00,("1200",   "along_street"): 24.00,("1200",   "elsewhere"): 24.00, ("1200","across_highway"): 30.00,
}

# Clause 9.3.2, Table 9.4 — HVDC Line Clearance (m)
HVDC_LINE_CLEARANCE = {
    100: {"ground": 6.50},
    200: {"ground": 7.30},
    300: {"ground": 8.50},
    400: {"ground": 9.40},
    500: {"ground": 10.60, "across_nh_sh": 17.25},
    800: {"ground": 13.90, "across_nh_sh": 22.75},
}

# Power pole lateral clearance (Clause 9.2.2)
POWER_POLE_LATERAL = {
    "min_from_road_edge_non_mountain_m": 10.0,
    "min_from_avenue_trees_m":            5.0,
    "street_lighting_with_kerb_desirable_m":  0.6,
    "street_lighting_with_kerb_min_m":        0.3,
    "street_lighting_no_kerb_from_edge_m":    1.5,
    "street_lighting_no_kerb_from_centre_m":  5.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ACCESS CONTROL
# ─────────────────────────────────────────────────────────────────────────────

# Clause 10.6.1, Table 10.1 — Min Distance Intersection to Fuel Station (m)
FUEL_STATION_FROM_INTERSECTION = {
    ("plain_rolling", "nh_sh_mdr"):   300,
    ("plain_rolling", "other"):        300,
    ("hilly_mountainous", "nh_sh_mdr"): 100,
    ("hilly_mountainous", "other"):    100,
}

# Clause 10.6.1, Table 10.2 — Min Distance Between Two Fuel Stations on NH (m)
FUEL_STATION_SPACING = {
    ("plain_rolling", "undivided"):   300,
    ("plain_rolling", "divided"):    1000,
    ("hilly_mountainous", "undivided"): 300,
    ("hilly_mountainous", "divided"):   300,
}

# Clause 10.6.1 — Fuel Station Special Setbacks (m)
FUEL_STATION_SETBACKS = {
    "from_toll_plaza_or_check_barrier_m": 1000,
    "from_rob_approach_m":                 200,
    "from_grade_separator_ramp_m":         300,
    "no_median_gap_plain_rolling_m":       300,   # each side
    "no_median_gap_hilly_mountain_m":      100,   # each side
}

# Clause 10.6.1, Table 10.3 — Fuel Station Decel/Accel Lane (m)
FUEL_STATION_LANES = {
    "decel_min_length_m":  70,
    "accel_min_length_m": 100,
    "lane_width_m":          5.5,
    "shoulder_width_m":      2.25,
}

# Clause 10.5.3 — 4-Lane Highway Access Spacing
ACCESS_4LANE = {
    "village_road_min_from_intersection_m": 3000,   # 3 km
}

# Grade Separation Warrant (Clause 10.5.6)
GRADE_SEPARATION_WARRANT = {
    "total_intersection_pcu_per_hour": 10000,   # ADVISORY trigger
}


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE — COMPLETE CHECK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

RULE_REGISTRY = {
    # Horizontal Alignment
    "H-01": {"param": "Min Horizontal Curve Radius",    "clause": "6.3.3", "type": "HARD FAIL"},
    "H-02": {"param": "Desirable Horizontal Radius",    "clause": "6.3.3", "type": "ADVISORY"},
    "H-03": {"param": "Superelevation friction safety", "clause": "6.4.2", "type": "HARD FAIL"},
    "H-04": {"param": "Superelevation max (e_max)",     "clause": "6.4.3", "type": "HARD FAIL"},
    "H-05": {"param": "SE not required threshold",      "clause": "6.4.4", "type": "ADVISORY"},
    "H-06": {"param": "Rate of SE change",              "clause": "6.4.5", "type": "ADVISORY"},
    "H-07": {"param": "Extra widening",                 "clause": "6.5.2", "type": "ADVISORY"},
    "H-08": {"param": "Setback distance",               "clause": "6.6",   "type": "HARD FAIL"},
    "H-09": {"param": "Transition curve length",        "clause": "6.7.4", "type": "HARD FAIL"},
    "H-10": {"param": "No transition at reverse curve", "clause": "6.9",   "type": "HARD FAIL"},
    "H-11": {"param": "Compound curve radius ratio",    "clause": "6.1/6.8","type": "HARD FAIL"},
    "H-12": {"param": "Compound curve transition",      "clause": "6.8",   "type": "HARD FAIL"},
    "H-13": {"param": "Hairpin bend inner radius",      "clause": "6.10",  "type": "HARD FAIL"},
    "H-14": {"param": "Hairpin bend width",             "clause": "6.10",  "type": "HARD FAIL"},
    "H-15": {"param": "Hairpin bend gradient",          "clause": "6.10",  "type": "HARD FAIL"},
    "H-16": {"param": "Hairpin bend superelevation",    "clause": "6.10",  "type": "HARD FAIL"},
    "H-17": {"param": "Straight tangent length",        "clause": "6.1",   "type": "ADVISORY"},
    "H-18": {"param": "Curve length (kink appearance)", "clause": "6.1",   "type": "ADVISORY"},
    # Vertical Alignment
    "V-01": {"param": "Max ruling gradient",            "clause": "7.2",   "type": "ADVISORY"},
    "V-02": {"param": "Max limiting gradient",          "clause": "7.2",   "type": "HARD FAIL"},
    "V-03": {"param": "Max exceptional gradient",       "clause": "7.2",   "type": "HARD FAIL"},
    "V-04": {"param": "Exceptional grade length",       "clause": "7.2",   "type": "ADVISORY"},
    "V-05": {"param": "Vertical curve required",        "clause": "7.5.3", "type": "HARD FAIL"},
    "V-06": {"param": "Min VC length",                  "clause": "7.5.3", "type": "HARD FAIL"},
    "V-07": {"param": "Summit curve SSD check",         "clause": "7.5.2", "type": "HARD FAIL"},
    "V-08": {"param": "Valley curve headlight SSD",     "clause": "7.6.2", "type": "HARD FAIL"},
    "V-09": {"param": "Grade change spacing",           "clause": "7.3",   "type": "ADVISORY"},
    "V-10": {"param": "Elevation gain per 2 km",        "clause": "7.2",   "type": "ADVISORY"},
    # Alignment Coordination
    "C-01": {"param": "Grade compensation",             "clause": "8.5",   "type": "HARD FAIL"},
    "C-02": {"param": "Horizontal curve min length",    "clause": "8.4",   "type": "HARD FAIL"},
    "C-03": {"param": "Tangent length limits",          "clause": "8.3",   "type": "ADVISORY"},
    "C-04": {"param": "Drainage cross-slope",           "clause": "8.4",   "type": "HARD FAIL"},
    "C-05": {"param": "Broken-back curve detection",    "clause": "8.4",   "type": "ADVISORY"},
    "C-06": {"param": "Phase shift H/V vertices",       "clause": "8.4",   "type": "ADVISORY"},
    # Clearances
    "CL-01": {"param": "Underpass vertical clearance",  "clause": "9.3.1", "type": "HARD FAIL"},
    "CL-02": {"param": "Underpass lateral span",        "clause": "9.2.1", "type": "HARD FAIL"},
    "CL-03": {"param": "AC power line clearance",       "clause": "9.3.2", "type": "HARD FAIL"},
    "CL-04": {"param": "HVDC power line clearance",     "clause": "9.3.2", "type": "HARD FAIL"},
    "CL-05": {"param": "Power pole lateral clearance",  "clause": "9.2.2", "type": "HARD FAIL"},
    # Access Control
    "AC-01": {"param": "Fuel station from intersection","clause": "10.6.1","type": "HARD FAIL"},
    "AC-02": {"param": "Fuel station spacing",          "clause": "10.6.1","type": "HARD FAIL"},
    "AC-03": {"param": "Fuel station special setbacks", "clause": "10.6.1","type": "HARD FAIL"},
    "AC-04": {"param": "Fuel station lane lengths",     "clause": "10.6.1","type": "HARD FAIL"},
    "AC-05": {"param": "Median gap near fuel station",  "clause": "10.6.1","type": "HARD FAIL"},
    "AC-06": {"param": "Grade separation warrant",      "clause": "10.5.6","type": "ADVISORY"},
    "AC-07": {"param": "4-lane access point spacing",   "clause": "10.5.3","type": "HARD FAIL"},
}

# Total checks: 46
# H-01 to H-18 : Horizontal alignment (18)
# V-01 to V-10 : Vertical alignment (10)
# C-01 to C-06 : Alignment coordination (6)
# CL-01 to CL-05: Clearances (5)
# AC-01 to AC-07: Access control (7)


# ─────────────────────────────────────────────────────────────────────────────
# KEY FORMULAS — QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

FORMULAS = {
    "superelevation":   "e = V**2 / (225 * R)",
    "equilibrium":      "e + f = V**2 / (127 * R)   [f = 0.15]",
    "r_min":            "R_min = V**2 / (127 * (e_max + 0.15))",
    "extra_widening":   "We = n*l**2/(2*R) + V/(9.5*sqrt(R))",
    "transition_i":     "Ls = 0.0215 * V**3 / (C * R)  ;  C = 80/(75+V), capped 0.5-0.8",
    "transition_ii_cl": "Ls = (e * N / 2) * (W + We)   [rotation about centreline]",
    "transition_ii_ie": "Ls = e * N * (W + We)          [rotation about inner edge]",
    "summit_ssd_lgs":   "L = N * S**2 / 4.4",
    "summit_ssd_lls":   "L = 2*S - 4.4/N",
    "summit_isd_lgs":   "L = N * S**2 / 9.6",
    "summit_isd_lls":   "L = 2*S - 9.6/N",
    "valley_lgs":       "L = N * S**2 / (1.50 + 0.035*S)",
    "valley_lls":       "L = 2*S - (1.50 + 0.035*S)/N",
    "grade_compensation":"GC = min((30+R)/R, 75/R)   [apply only if grade > 4%]",
    "setback_lgs":      "m = R - (R-n)*cos(S/(2*(R-n)))",
    "setback_lls":      "m = R - (R-n)*cos(Lc/(2*(R-n))) + (S-Lc)/2*sin(Lc/(2*(R-n)))",
    "ssd_formula":      "SSD = 0.278*V*t + V**2/(254*f)   [t=2.5s]",
    "ssd_grade":        "d2 = V**2 / (254*(f +/- 0.01*G))",
    "k_value":          "K = L / N   (L in m, N as decimal or %)",
}
