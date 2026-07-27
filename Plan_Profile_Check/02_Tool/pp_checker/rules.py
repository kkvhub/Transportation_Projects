"""
rules.py — IRC rule engine parameterized by road class.

road_class: "2_lane" -> IRC:73-2023 | "4_lane" -> IRC:SP:84-2019
terrain:    "plain_rolling" | "mountainous" | "steep"
"""
from __future__ import annotations
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, "standards", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_standard(road_class):
    if road_class == "4_lane":
        return _load("irc_sp84_2019"), "IRC:SP:84-2019"
    return _load("irc_73_2023"), "IRC:73-2023"


class Verdicts:
    PASS, ADVISORY, FAIL, INFO = "PASS", "ADVISORY", "HARD FAIL", "INFO"


def check(model, road_class="2_lane", terrain="mountainous", carriageway_w=7.0):
    std, std_name = get_standard(road_class)
    if road_class == "4_lane":
        std_name += " + IRC:73-2023 alignment provisions"
    R_ = []

    def add(element, chk, provided, limit, verdict, note=""):
        R_.append({"element": element, "check": chk, "provided": str(provided),
                   "limit": str(limit), "verdict": verdict, "note": note})

    mt = terrain in ("mountainous", "steep")
    # ---- speed / radius / gradient limits per standard --------------------
    if road_class == "4_lane":
        tkey = "mountain_steep" if mt else "plain_rolling"
        ds = {"ruling": std.DESIGN_SPEED[(tkey, "ruling")],
              "minimum": std.DESIGN_SPEED[(tkey, "minimum")]}
        rad = std.MIN_RADIUS[(tkey, "desirable")], std.MIN_RADIUS[(tkey, "absolute")]
        glim = std.GRADIENT["mountainous" if mt else "plain_rolling"]
        k_sum_ssd, k_sum_isd = std.K_SUMMIT_SSD, std.K_SUMMIT_ISD
        k_val, min_vc = std.K_VALLEY, std.MIN_VC_LENGTH
        e_low, e_high = std.E_MAX["below_desirable_min_radius"], std.E_MAX["above_desirable_min_radius"]
        vc_len_key = "min_L_m"
    else:
        tkey = "mountainous" if mt else "plain"
        ds = std.DESIGN_SPEED[("2_lane_nh_sh", tkey)]
        mr = std.MIN_RADIUS_NH_SH[("nh_sh", "mountain_steep" if mt else "plain_rolling")]
        rad = mr["desirable"], mr["absolute"]
        glim = std.GRADIENT_HIGHWAYS["mountainous" if mt else "plain_rolling"]
        k_sum_ssd, k_sum_isd = std.K_SUMMIT_SSD, std.K_SUMMIT_ISD
        k_val, min_vc = std.K_VALLEY, std.MIN_VC_LENGTH
        e_low, e_high = std.E_MAX["below_desirable_min_radius"], std.E_MAX["above_desirable_min_radius"]
        vc_len_key = "min_L"
    des_r, abs_r = rad

    # ---- horizontal curves -------------------------------------------------
    for i, c in enumerate(model.get("curves", []), 1):
        ch = c.get("hip_ch")
        name = f"HC-{i} (HIP {_fmt_ch(ch)})"
        V, Rr, Ls, Lc, e = (c.get(k) for k in ("V", "R", "Ls", "Lc", "e"))
        if c.get("math_ok") is False:
            add(name, "OCR consistency", "Lc/Ts recompute mismatch", "-",
                Verdicts.ADVISORY, "verify extraction (auto-flag)")
        if V:
            if V < ds["minimum"]:
                add(name, "design speed", V, f">={ds['minimum']}", Verdicts.FAIL)
            elif V < ds["ruling"]:
                add(name, "design speed", V, f"ruling {ds['ruling']}",
                    Verdicts.ADVISORY, "below ruling — justify on record")
            else:
                add(name, "design speed", V, f"ruling {ds['ruling']}", Verdicts.PASS)
        if Rr:
            if Rr < abs_r:
                add(name, "min radius", Rr, f"abs {abs_r}", Verdicts.FAIL)
            elif Rr < des_r:
                add(name, "min radius", Rr, f"des {des_r}", Verdicts.ADVISORY,
                    "below desirable — deviation must be on record")
            else:
                add(name, "min radius", Rr, f"des {des_r}", Verdicts.PASS)
        if e is not None and Rr:
            cap = e_low if Rr < des_r else e_high
            add(name, "superelevation cap", f"{e}%", f"<={cap*100:.0f}%",
                Verdicts.PASS if e <= cap * 100 + 1e-9 else Verdicts.FAIL)
            if V:
                e_calc = min(V * V / (225.0 * Rr), cap) * 100
                add(name, "e vs V^2/225R", f"{e}%", f"calc {e_calc:.1f}%",
                    Verdicts.PASS if abs(e_calc - e) < 0.15 else Verdicts.ADVISORY)
        if V and Rr and Ls:
            C = max(0.5, min(0.8, 80.0 / (75.0 + V)))
            ls1 = 0.0215 * V ** 3 / (C * Rr)
            we = _extra_widening(Rr)
            N = 60 if mt else 150
            ls2 = (e or 0) / 100.0 * N * (carriageway_w + we) / 2.0
            req = max(ls1, ls2)
            add(name, "spiral length Ls", f"{Ls:.0f} m", f">={req:.1f} m",
                Verdicts.PASS if Ls >= req - 0.05 else Verdicts.FAIL,
                f"comfort {ls1:.1f} / rotation {ls2:.1f}")
            if we > 0:
                add(name, "extra widening", "-", f"{we} m req.", Verdicts.INFO,
                    "verify on cross-sections")
        if Rr and Rr < 450:
            add(name, "signage", f"R={Rr:.0f}", "R<450", Verdicts.INFO,
                "curve warning sign + chevrons required")

    # IRC:73 Clause 6.1: broken-back horizontal curves in the same
    # direction require at least 10 seconds of travel on the tangent.
    if road_class in ("2_lane", "4_lane"):
        h_pairs = sorted(enumerate(model.get("curves", []), 1),
                         key=lambda x: x[1].get("hip_ch") or float("inf"))
        for (i1, c1), (i2, c2) in zip(h_pairs, h_pairs[1:]):
            element = f"HC-{i1} to HC-{i2}"
            d1, d2 = c1.get("direction_sign"), c2.get("direction_sign")
            if d1 is None or d2 is None:
                add(element, "10 s same-direction tangent", "not checked", "direction required",
                    Verdicts.INFO, "profile signed-radius match unavailable; manual review")
                continue
            if d1 != d2:
                add(element, "10 s same-direction tangent", "opposite direction", "not applicable",
                    Verdicts.INFO, "reverse curves are checked under separate transition requirements")
                continue
            end1, start2 = c1.get("curve_end_ch"), c2.get("curve_start_ch")
            speed = max(v for v in (c1.get("V"), c2.get("V"), ds["ruling"]) if v is not None)
            required = speed * 10.0 / 3.6
            if end1 is None or start2 is None:
                add(element, "10 s same-direction tangent", "not checked", f">={required:.1f} m",
                    Verdicts.INFO, "HIP or Ts missing; curve limits unavailable")
                continue
            gap = start2 - end1
            add(element, "10 s same-direction tangent", f"{gap:.1f} m", f">={required:.1f} m",
                Verdicts.PASS if gap >= required else Verdicts.ADVISORY,
                f"same profile radius sign; {speed:g} km/h design basis")

    # ---- vertical: grades + curves from band annotations -------------------
    v_speed = _governing_speed(model, ds)
    for si, ann in enumerate(model.get("vertical_annotations", [])):
        grades = _pair_grades(ann)
        for g in grades:
            gname = f"G@{_fmt_ch(g['ch_mid'])} (sheet {si})"
            gv = abs(g["G"])
            if gv > glim["limiting"]:
                add(gname, "gradient", f"{g['G']}%", f"lim {glim['limiting']}%", Verdicts.FAIL)
            elif gv > glim["ruling"]:
                add(gname, "gradient", f"{g['G']}%", f"rul {glim['ruling']}%", Verdicts.ADVISORY)
            else:
                add(gname, "gradient", f"{g['G']}%", f"rul {glim['ruling']}%", Verdicts.PASS)
        for vc in ([] if model.get("vertical_curves") else _pair_vcs(ann)):
            vname = f"VC@{_fmt_ch(vc['ch_mid'])} (sheet {si})"
            K, L = abs(vc["K"]), vc["L"]
            minL = min_vc.get(int(v_speed), {}).get(vc_len_key, 40)
            add(vname, "min VC length", f"{L:.0f} m", f">={minL} m",
                Verdicts.PASS if L >= minL else Verdicts.FAIL, f"N={L/K:.2f}%")
            kv = k_val.get(int(v_speed))
            ks = k_sum_ssd.get(int(v_speed))
            ki = k_sum_isd.get(int(v_speed))
            if vc.get("type") == "valley" and kv:
                add(vname, "K valley (headlight)", f"{K:.2f}", f">={kv}",
                    Verdicts.PASS if K >= kv else Verdicts.FAIL)
            elif vc.get("type") == "summit":
                if ks:
                    add(vname, "K summit (SSD)", f"{K:.2f}", f">={ks}",
                        Verdicts.PASS if K >= ks else Verdicts.FAIL)
                if ki:
                    add(vname, "K summit (ISD)", f"{K:.2f}", f">={ki}",
                        Verdicts.PASS if K >= ki else Verdicts.FAIL,
                        "ISD basis for new construction")
            else:
                if kv:
                    add(vname, "K (type unknown — checked as valley)", f"{K:.2f}",
                        f">={kv}", Verdicts.PASS if K >= kv else Verdicts.ADVISORY,
                        "curve type not auto-detected")
    for vc in model.get("vertical_curves", []):
        page_note = "continued sheets " + ",".join(str(s) for s in vc.get("sheets", [])) \
            if vc.get("continued") else "sheet " + ",".join(str(s) for s in vc.get("sheets", []))
        _check_vc(add, vc, f"{vc.get('id', 'VC')}@{_fmt_ch(vc.get('ch_mid'))} ({page_note})",
                  v_speed, min_vc, vc_len_key, k_val, k_sum_ssd, k_sum_isd)
    # IRC:73 Clause 7.3: spacing of grade changes, plus the separate
    # 10-second tangent check for same-direction broken-back vertical curves.
    if road_class in ("2_lane", "4_lane"):
        vcs = sorted((v for v in model.get("vertical_curves", [])
                      if v.get("pvi_chainage") is not None),
                     key=lambda v: v["pvi_chainage"])
        practical = getattr(std, "VERTICAL_PRACTICAL", {})
        grade_min = practical.get("min_distance_mountain_m", 75) if mt \
            else practical.get("min_distance_between_grade_changes_m", 150)
        tangent_min = v_speed * 10.0 / 3.6
        for v1, v2 in zip(vcs, vcs[1:]):
            element = f"{v1.get('id', 'VC')} to {v2.get('id', 'VC')}"
            spacing = v2["pvi_chainage"] - v1["pvi_chainage"]
            add(element, "grade-change spacing", f"{spacing:.1f} m", f">={grade_min:.0f} m",
                Verdicts.PASS if spacing >= grade_min else Verdicts.ADVISORY,
                "approximate PVI centre-to-centre spacing from profile schematic")
            same_type = v1.get("type") in ("summit", "valley") and v1.get("type") == v2.get("type")
            if not same_type:
                continue
            pvt, pvc = v1.get("pvt_chainage"), v2.get("pvc_chainage")
            if pvt is None or pvc is None:
                add(element, "10 s vertical broken-back tangent", "not checked",
                    f">={tangent_min:.1f} m", Verdicts.INFO,
                    "same-direction curves but PVI or L is missing")
                continue
            gap = pvc - pvt
            add(element, "10 s vertical broken-back tangent", f"{gap:.1f} m",
                f">={tangent_min:.1f} m",
                Verdicts.PASS if gap >= tangent_min else Verdicts.ADVISORY,
                f"same {v1['type']} direction; {v_speed:g} km/h design basis")
    return {"standard": std_name, "road_class": road_class, "terrain": terrain,
            "results": R_, "summary": _summary(R_)}


def _check_vc(add, vc, vname, v_speed, min_vc, vc_len_key, k_val, k_sum_ssd, k_sum_isd):
    raw_k, L = vc.get("K"), vc.get("L")
    K = abs(raw_k) if raw_k is not None else None
    if L:
        minL = min_vc.get(int(v_speed), {}).get(vc_len_key, 40)
        add(vname, "min VC length", f"{L:.0f} m", f">={minL} m",
            Verdicts.PASS if L >= minL else Verdicts.FAIL,
            f"N={L/K:.2f}%" if K else "K missing/blank on drawing")
    else:
        add(vname, "min VC length", "not read", "-", Verdicts.INFO,
            "vertical curve length missing/blank on drawing")
    if K is None:
        add(vname, "K value", "not read", "-", Verdicts.INFO,
            "K missing/blank on drawing; manual review required")
        return
    kv = k_val.get(int(v_speed))
    ks = k_sum_ssd.get(int(v_speed))
    ki = k_sum_isd.get(int(v_speed))
    if vc.get("type") == "valley" and kv:
        add(vname, "K valley (headlight)", f"{K:.2f}", f">={kv}",
            Verdicts.PASS if K >= kv else Verdicts.FAIL)
    elif vc.get("type") == "summit":
        if ks:
            add(vname, "K summit (SSD)", f"{K:.2f}", f">={ks}",
                Verdicts.PASS if K >= ks else Verdicts.FAIL)
        if ki:
            add(vname, "K summit (ISD)", f"{K:.2f}", f">={ki}",
                Verdicts.PASS if K >= ki else Verdicts.FAIL,
                "ISD basis for new construction")
    elif kv:
        add(vname, "K (type unknown - checked as valley)", f"{K:.2f}",
            f">={kv}", Verdicts.PASS if K >= kv else Verdicts.ADVISORY,
            "curve type not auto-detected")


def _extra_widening(R):
    if 75 <= R <= 100:
        return 0.9
    if 100 < R <= 300:
        return 0.6
    return 0.0


def _governing_speed(model, ds):
    vs = [c.get("V") for c in model.get("curves", []) if c.get("V")]
    return max(vs) if vs else ds["ruling"]


def _pair_grades(ann):
    """G= annotations paired with following L= (grade length)."""
    out = []
    for i, a in enumerate(ann):
        if a["key"] == "G":
            L = None
            for b in ann[i + 1:i + 3]:
                if b["key"] == "L":
                    L = b["value"]
                    break
            out.append({"G": a["value"], "L": L, "ch_mid": a["ch_mid"]})
    return out


def _pair_vcs(ann):
    """K= annotations paired with adjacent L= -> vertical curves."""
    out = []
    for i, a in enumerate(ann):
        if a["key"] == "K":
            L = None
            for b in ann[max(0, i - 2):i + 3]:
                if b["key"] == "L" and abs(b["ch_mid"] - a["ch_mid"]) < 60:
                    L = b["value"]
                    break
            if L:
                typ = "summit" if a["value"] < 0 else None
                out.append({"K": a["value"], "L": L, "ch_mid": a["ch_mid"],
                            "type": typ or "valley" if a["value"] > 0 else "summit"})
    return out


def _fmt_ch(ch):
    if ch is None:
        return "?"
    return f"{int(ch // 1000)}+{ch % 1000:07.3f}".rstrip("0").rstrip(".")


def _summary(results):
    n = {"PASS": 0, "ADVISORY": 0, "HARD FAIL": 0, "INFO": 0}
    for r in results:
        n[r["verdict"]] += 1
    return n
