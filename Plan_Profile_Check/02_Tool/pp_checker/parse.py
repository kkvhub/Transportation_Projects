"""
parse.py — assemble structured design data from a P&P PDF.

Pipeline: open doc -> classify sheets -> HIP tables (OCR) -> profile bands
(vector + OCR) -> structures -> validated model dict.
"""
from __future__ import annotations
import math
import re
import fitz

from . import extract, ocr


def open_document(path):
    doc = fitz.open(path)
    pages = [extract.Page(doc, i) for i in range(len(doc))]
    plan, profile = [], []
    for pg in pages:
        # A standard P&P sheet can contain plan geometry above the profile.
        # Every page is therefore eligible for plan-table extraction, while
        # pages carrying a profile band are additionally processed as profile.
        plan.append(pg)
        if extract.is_profile_sheet(pg):
            profile.append(pg)
    return doc, plan, profile


# ---------------------------------------------------------------- curves

def horizontal_curves(plan_pages, llm_reader=None, km_range=None):
    """OCR every HIP table found on plan sheets -> list of curve dicts."""
    curves = []
    for pg in plan_pages:
        for grid in extract.find_table_grids(pg):
            rows = ocr.read_grid(pg, grid)
            flat = " ".join(c for r in rows for c in r)
            if "HIP" not in flat:
                continue
            d = ocr.parse_hip_rows(rows, pg=pg, grid=grid, km_range=km_range)
            d["ocr_rows"] = rows
            d["page"] = pg.index
            d.update(_verify_curve_math(d))
            if llm_reader and not d.get("math_ok", False):
                d = llm_reader.reread_hip(pg, grid, d)
                d.update(_verify_curve_math(d))
            if d.get("R"):
                curves.append(d)
    curves.sort(key=lambda c: c.get("hip_ch") or 0)
    return curves


def _verify_curve_math(d):
    """Recompute Lc and Ts from delta/R/Ls; flags OCR misreads."""
    out = {"math_ok": None, "lc_calc": None, "ts_calc": None}
    if not all(d.get(k) for k in ("R", "Ls", "delta_deg")):
        return out
    R, Ls, dl = d["R"], d["Ls"], math.radians(d["delta_deg"])
    th = Ls / (2 * R)
    lc = R * (dl - 2 * th)
    dr = Ls ** 2 / (24 * R)
    ts = (R + dr) * math.tan(dl / 2) + Ls / 2
    out["lc_calc"], out["ts_calc"] = round(lc, 2), round(ts, 2)
    ok = True
    if d.get("Lc") is not None:
        ok &= abs(lc - d["Lc"]) < 0.5
    if d.get("Ts") is not None:
        ok &= abs(ts - d["Ts"]) < 0.5
    # If OCR supplied a DMS delta that disagrees but the table's own
    # Lc/R/Ls relation is coherent, keep the table fields as usable and
    # mark only the angular text as suspect.
    if not ok and d.get("Lc") is not None and abs(lc - d["Lc"]) < 0.5:
        if d.get("delta_ocr_mismatch") is not None:
            ok = True
            d["delta_confidence"] = "derived"
    out["math_ok"] = ok
    return out


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def profile_bands(profile_pages):
    """Extract vertical/horizontal/superelevation band content per sheet."""
    sheets = []
    for pg in profile_pages:
        axis = extract.ChainageAxis(pg)
        seps = extract.find_band_rows(pg, axis)
        # band rows are the separators directly above the chainage band
        below = [y for y in seps if y <= axis.band_y + 6]
        rows = below[-5:] if len(below) >= 5 else below
        if len(rows) < 5:
            # full-width separators weren't found (some exports only draw
            # row dividers across the narrow label column) — fall back to
            # the tick-mark ladder, whose last 5 boundaries are always
            # vertical/horizontal/superelevation/chainage
            ladder, _lx0, _lx1 = extract.find_band_row_ladder(pg)
            if len(ladder) >= 5:
                rows = ladder[-5:]
        bands = {}
        if getattr(axis, "axis_source", "") == "combined-sheet profile grid":
            y = axis.profile_vertical_bottom
            bands = {
                "vertical": (y - 27, y),
                "horizontal": (y, y + 8),
                "superelevation": (y + 8, y + 18),
                "chainage": (y + 18, y + 34),
            }
        names = ["vertical", "horizontal", "superelevation", "chainage"]
        if not bands:
            for k in range(len(rows) - 1):
                if k < len(names):
                    bands[names[k]] = (rows[k], rows[k + 1])
        info = {"page": pg.index, "axis": axis, "bands": bands, "ch_min": axis.ch_min, "ch_max": axis.ch_max}
        if "superelevation" in bands:
            y0, y1 = bands["superelevation"]
            ramps = extract.superelevation_ramps(pg, axis, y0, y1)
            info["se_ramps"] = ramps
            info["se_plateaus"] = extract.plateaus_from_ramps(ramps)
            info["se_labels"] = _band_values(pg, axis, y0, y1, r"e=(%s)" % r"[\d.]+")
        if "horizontal" in bands:
            y0, y1 = bands["horizontal"]
            info["h_labels"] = ocr.band_words(pg, axis, y0, y1)
        if "vertical" in bands:
            y0, y1 = bands["vertical"]
            info["v_labels"] = ocr.band_words(pg, axis, y0, y1)
            # Standard OCR often reads only the largest labels on a sheet.
            # Always supplement it with the high-zoom line pass so curves in
            # the middle of a dense schematic are not silently omitted.
            info["v_labels"].extend(ocr.band_line_words(pg, axis, y0, y1))
            k_pos = [(c0 + c1) / 2 for t, c0, c1 in info["v_labels"]
                     if re.match(r"K\s*=", t.strip(), re.I)]
            l_pos = [(c0 + c1) / 2 for t, c0, c1 in info["v_labels"]
                     if re.match(r"(?:L|1)\s*=", t.strip(), re.I)]
            focused_positions = []
            # Recheck around either side of an incomplete K/L pair. A first
            # pass may read K but miss L, or vice versa, on very small text.
            for ch in k_pos + l_pos:
                if any(abs(done - ch) < 5 for done in focused_positions):
                    continue
                focused_positions.append(ch)
                focused = ocr.band_column_words(pg, axis, y0, y1, ch)
                info["v_labels"].extend(
                    v for v in focused
                    if re.match(r"^(?:K|L|1)\s*(?:=|$)", v[0].strip(), re.I)
                )
        if "v_labels" not in info:
            info["v_labels"] = _text_annotations(pg, axis)
        if "h_labels" not in info:
            info["h_labels"] = _text_annotations(pg, axis, keys=("R", "Ls", "Lc", "L"))
        info["profile_events"] = _profile_events(pg)
        info["h_text_annotations"] = _profile_curve_annotations(pg, axis)
        info["levels"] = extract.level_rows(pg, axis)
        culv, vert = extract.structure_symbols(pg, axis)
        info["culvert_symbols"], info["bridge_verticals"] = culv, vert
        sheets.append(info)
    return sheets


def _band_values(pg, axis, y0, y1, pattern):
    out = []
    for txt, c0, c1 in ocr.band_words(pg, axis, y0, y1):
        m = re.search(pattern, txt)
        if m:
            out.append((float(m.group(1)), (c0 + c1) / 2))
    return out


def _text_annotations(pg, axis, keys=("R", "L", "K", "G", "e")):
    """Read visible text-layer annotations when no OCR band was found."""
    out = []
    key_re = "|".join(re.escape(k) for k in keys)
    pat = re.compile(rf"^(?:{key_re})=([-+]?\d+(?:\.\d+)?)(?:m|%)?$")
    for r, txt in pg.words:
        if pat.match(txt.strip()):
            out.append((txt.strip(), axis.ch((r.x0 + r.x1) / 2), axis.ch((r.x0 + r.x1) / 2)))
    return out


def _profile_events(pg):
    """Text-layer horizontal alignment events: TS/SC/CS/ST plus chainage."""
    events = []
    words = [(r, t.strip()) for r, t in pg.words]
    for i, (r, txt) in enumerate(words[:-1]):
        key = txt.upper().rstrip(":")
        if key not in {"TS", "SC", "CS", "ST"}:
            continue
        nxt = words[i + 1][1].strip()
        m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", nxt)
        if not m:
            continue
        ch = float(m.group())
        if ch < 0:
            continue
        events.append({"key": key, "chainage": ch, "x": (r.x0 + r.x1) / 2, "y": r.y0})
    events.sort(key=lambda e: (e["chainage"], e["y"], e["x"]))
    ded = []
    for e in events:
        if ded and e["key"] == ded[-1]["key"] and abs(e["chainage"] - ded[-1]["chainage"]) < 0.05:
            continue
        ded.append(e)
    return ded


def _profile_curve_annotations(pg, axis):
    """Text-layer R/Ls/Lc labels from profile schematic bands."""
    out = []
    pat = re.compile(r"^(R|Ls|Lc|Le|L)[=\-:]?([-+]?\d+(?:\.\d+)?)(?:m)?$", re.I)
    for r, txt in pg.words:
        t = txt.strip().replace(" ", "")
        m = pat.match(t)
        if not m:
            continue
        key = m.group(1)
        if key.lower() == "le":
            key = "Lc"
        try:
            value = abs(float(m.group(2)))
        except ValueError:
            continue
        out.append({"key": key, "value": value, "ch_mid": axis.ch((r.x0 + r.x1) / 2), "raw": txt})
    return out


def profile_horizontal_curves(sheets):
    """Derive horizontal curves from profile schematic event labels.

    This covers P&P drawings without HIP tables, where the alignment is
    encoded by TS/SC/CS/ST chainages and R/Ls/Lc text in the profile band.
    """
    curves = []
    for sh in sheets:
        events = sorted(sh.get("profile_events", []), key=lambda e: e["chainage"])
        anns = sh.get("h_text_annotations", [])
        used_cs = set()
        for i, sc in enumerate(events):
            if sc["key"] != "SC":
                continue
            cs_idx = None
            for j in range(i + 1, len(events)):
                if events[j]["key"] == "CS" and events[j]["chainage"] > sc["chainage"] and j not in used_cs:
                    cs_idx = j
                    break
            if cs_idx is None:
                continue
            cs = events[cs_idx]
            used_cs.add(cs_idx)
            ts = None
            for j in range(i - 1, -1, -1):
                if events[j]["key"] == "TS" and 0 <= sc["chainage"] - events[j]["chainage"] <= 250:
                    ts = events[j]
                    break
            st = None
            for j in range(cs_idx + 1, len(events)):
                if events[j]["key"] == "ST" and 0 <= events[j]["chainage"] - cs["chainage"] <= 250:
                    st = events[j]
                    break
            start_ch = ts["chainage"] if ts else sc["chainage"]
            end_ch = st["chainage"] if st else cs["chainage"]
            mid = (sc["chainage"] + cs["chainage"]) / 2.0
            c = {
                "source": "profile_schematic",
                "page": sh.get("page"),
                "hip_ch": round(mid, 3),
                "TS": ts["chainage"] if ts else None,
                "SC": sc["chainage"],
                "CS": cs["chainage"],
                "ST": st["chainage"] if st else None,
                "Ls": round(sc["chainage"] - ts["chainage"], 3) if ts else None,
                "Lc": round(cs["chainage"] - sc["chainage"], 3),
                "math_ok": None,
                "extraction_confidence": "medium",
            }
            if st and c["Ls"] is not None:
                out_ls = st["chainage"] - cs["chainage"]
                if abs(out_ls - c["Ls"]) <= 2:
                    c["Ls"] = round((c["Ls"] + out_ls) / 2.0, 3)
                else:
                    c["Ls_out"] = round(out_ls, 3)
            for key, target in (("R", "R"), ("Lc", "Lc_label"), ("Ls", "Ls_label")):
                vals = [a for a in anns if a["key"].lower() == key.lower()
                        and start_ch - 80 <= a["ch_mid"] <= end_ch + 80]
                if vals:
                    near = min(vals, key=lambda a: abs(a["ch_mid"] - mid))
                    c[target] = near["value"]
            if c.get("R") is not None:
                c["R"] = abs(c["R"])
            if c.get("Ls_label") and c.get("Ls") is None:
                c["Ls"] = c["Ls_label"]
            if c.get("Lc_label") and abs(c["Lc_label"] - c["Lc"]) > 2:
                c["Lc_text"] = c["Lc_label"]
            curves.append(c)
    curves.sort(key=lambda c: (c.get("hip_ch") or 0, c.get("page") or 0))
    ded = []
    for c in curves:
        if ded and abs((c.get("SC") or 0) - (ded[-1].get("SC") or 0)) < 0.5 \
                and abs((c.get("CS") or 0) - (ded[-1].get("CS") or 0)) < 0.5:
            continue
        ded.append(c)
    return ded


def enrich_alignment_geometry(curves, sheets, vertical_curves):
    """Add spacing geometry without changing the primary curve extraction."""
    signed_radii = []
    for sh in sheets:
        # On tightly stacked bands, OCR may place a horizontal R label in the
        # adjacent vertical crop. R is not a vertical-profile parameter, so
        # both raw label sets are safe sources for direction matching.
        radius_labels = sh.get("h_labels", []) + sh.get("v_labels", [])
        for raw, c0, c1 in radius_labels:
            cleaned = raw.strip().replace("R--", "R=-")
            m = re.match(r"^R\s*=\s*([-+]?\d+(?:\.\d+)?)", cleaned, re.I)
            if not m:
                continue
            signed_r = float(m.group(1))
            if abs(signed_r) < 20:
                continue
            item = {"signed_R": signed_r, "R": abs(signed_r),
                    "ch_mid": (c0 + c1) / 2.0, "page": sh.get("page"), "raw": raw}
            if any(abs(x["ch_mid"] - item["ch_mid"]) < 3
                   and abs(x["signed_R"] - signed_r) < 1 for x in signed_radii):
                continue
            signed_radii.append(item)

    used = set()
    for c in sorted(curves, key=lambda x: x.get("hip_ch") or float("inf")):
        hip, radius, tangent = c.get("hip_ch"), c.get("R"), c.get("Ts")
        if hip is not None and tangent is not None:
            c["curve_start_ch"] = round(hip - abs(tangent), 3)
            c["curve_end_ch"] = round(hip + abs(tangent), 3)
            c["curve_limit_source"] = "HIP +/- Ts"
        if hip is None or radius is None:
            continue
        candidates = []
        for j, p in enumerate(signed_radii):
            if j in used:
                continue
            r_tol = max(5.0, abs(radius) * 0.05)
            ch_tol = max(250.0, abs(tangent or 0) + 120.0)
            r_diff, ch_diff = abs(p["R"] - abs(radius)), abs(p["ch_mid"] - hip)
            if r_diff <= r_tol and ch_diff <= ch_tol:
                candidates.append((r_diff / r_tol + ch_diff / ch_tol, j, p, r_diff, ch_diff))
        if not candidates:
            c["direction_confidence"] = "unmatched"
            continue
        _, j, p, r_diff, ch_diff = min(candidates, key=lambda x: x[0])
        used.add(j)
        c["direction_sign"] = -1 if p["signed_R"] < 0 else 1
        c["direction"] = "negative R" if p["signed_R"] < 0 else "positive R"
        c["direction_source"] = "matched signed R in profile horizontal schematic"
        c["direction_profile_ch"] = round(p["ch_mid"], 3)
        c["direction_profile_R"] = p["signed_R"]
        c["direction_confidence"] = (
            "high" if r_diff <= max(2.0, abs(radius) * 0.02) and ch_diff <= 150
            else "medium"
        )

    for vc in vertical_curves:
        pvi, length = vc.get("ch_mid"), vc.get("L")
        vc["pvi_chainage"] = pvi
        vc["pvi_source"] = "schematic K/L centre (approximate PVI)"
        if pvi is not None and length is not None:
            vc["pvc_chainage"] = round(pvi - length / 2.0, 3)
            vc["pvt_chainage"] = round(pvi + length / 2.0, 3)
            vc["curve_limits_source"] = "PVI +/- L/2 (symmetric parabola)"


def band_annotations(words):
    """Group OCR'd band words into R=/L=/K=/G= annotations."""
    ann = []
    for txt, c0, c1 in words:
        cleaned = txt.strip()
        if re.match(r"^1\s*=", cleaned):
            cleaned = re.sub(r"^1\s*=", "L=", cleaned)
        ch_mid = (c0 + c1) / 2
        if cleaned.upper() in {"K", "L"}:
            ann.append({"key": cleaned.upper(), "value": None,
                        "ch_mid": ch_mid, "confidence": "blank", "raw": cleaned})
            continue
        # Common OCR slips in vertical schematic bands:
        #   K=-39.892 may lose the K and become =-39.892m
        #   L=200.000 may lose the L and become 200.
        m_lost_k = re.match(r"^=([-+]?\d+(?:\.\d+)?)(?:m)?$", cleaned)
        if m_lost_k:
            try:
                val = float(m_lost_k.group(1))
                key = "L" if val > 0 and cleaned.endswith(".000") and 20 <= val <= 1000 else "K"
                if key == "K" and not (10 <= abs(val) <= 200):
                    continue
                ann.append({"key": key, "value": val, "ch_mid": ch_mid,
                            "confidence": "low", "raw": cleaned})
            except ValueError:
                pass
            continue
        m_lost_l = re.match(r"^(\d{2,4})(?:\.)?$", cleaned)
        if m_lost_l:
            try:
                val = float(m_lost_l.group(1))
                if 20 <= val <= 1000:
                    ann.append({"key": "L", "value": val, "ch_mid": ch_mid,
                                "confidence": "low", "raw": cleaned})
            except ValueError:
                pass
            continue
        for key in ("R", "L", "K", "G", "e"):
            m = re.match(rf"{key}=([-+]?[\d.]+)", cleaned)
            if m:
                try:
                    val = float(m.group(1).rstrip("."))
                    ann_key = key
                    if key == "R" and val < 0 and 10 <= abs(val) <= 250:
                        ann_key = "K"
                    ann.append({"key": ann_key, "value": val,
                                "ch_mid": ch_mid, "raw": cleaned})
                except ValueError:
                    pass
        m_lost_g = re.match(r"^G([-+]?\d+(?:\.\d+)?)%?$", cleaned)
        if m_lost_g:
            try:
                ann.append({"key": "G", "value": float(m_lost_g.group(1)),
                            "ch_mid": ch_mid, "confidence": "low", "raw": cleaned})
            except ValueError:
                pass
    return ann


def _set_vc_type_from_gradients(vc):
    """Prefer profile grade direction over K sign when both grades are known."""
    g_in, g_out = vc.get("G_in"), vc.get("G_out")
    if g_in is None or g_out is None:
        return vc
    if abs(g_out - g_in) <= 0.01:
        vc["type"] = "unknown"
        vc["type_source"] = "incoming/outgoing profile gradients are effectively equal"
    elif g_out < g_in:
        vc["type"] = "summit"
        vc["type_source"] = "incoming/outgoing profile gradients"
    else:
        vc["type"] = "valley"
        vc["type_source"] = "incoming/outgoing profile gradients"
    return vc


def _infer_vc_grades(vc):
    """Fill or correct side gradients from L/K when possible."""
    L, K = vc.get("L"), vc.get("K")
    g_in, g_out = vc.get("G_in"), vc.get("G_out")
    _set_vc_type_from_gradients(vc)
    grade_diff = (abs(g_out - g_in)
                  if g_in is not None and g_out is not None else None)
    if K is not None and not L and grade_diff and grade_diff > 0.001:
        vc["L"] = L = round(abs(K) * grade_diff, 3)
        vc["L_source"] = "derived from K and gradients after OCR recheck"
    elif L and K is None and grade_diff and grade_diff > 0.001:
        signed_diff = g_out - g_in
        vc["K"] = K = round(L / signed_diff, 3)
        vc["K_source"] = "derived from L and gradients after OCR recheck"
    if not L or not K:
        return vc
    diff = abs(L / K)
    if g_in is None and g_out is None:
        return vc
    typ = vc.get("type")
    if typ not in {"summit", "valley"}:
        if vc.get("G_in") is not None and vc.get("G_out") is not None:
            vc["algebraic_difference_pct"] = round(vc["G_out"] - vc["G_in"], 3)
            vc["algebraic_difference_abs_pct"] = round(abs(vc["G_out"] - vc["G_in"]), 3)
            _set_vc_type_from_gradients(vc)
        return vc
    if g_out is None and g_in is not None:
        g_out = g_in + diff if typ == "valley" else g_in - diff
        vc["G_out"] = vc["grade_out_pct"] = round(g_out, 3)
        vc["grade_out_source"] = "inferred from L/K"
    elif g_in is None and g_out is not None:
        g_in = g_out - diff if typ == "valley" else g_out + diff
        vc["G_in"] = vc["grade_in_pct"] = round(g_in, 3)
        vc["grade_in_source"] = "inferred from L/K"
    elif g_in is not None and g_out is not None:
        observed = abs(g_out - g_in)
        if abs(observed - diff) > 0.5:
            if typ == "valley":
                expected_out = g_in + diff
                expected_in = g_out - diff
            else:
                expected_out = g_in - diff
                expected_in = g_out + diff
            # The incoming grade is usually the stronger anchor for continued
            # curves; the outgoing label can be borrowed accidentally from the
            # next schematic curve when pages overlap.
            if abs(expected_out) <= 12:
                vc["G_out"] = vc["grade_out_pct"] = round(expected_out, 3)
                vc["grade_out_source"] = "corrected from L/K consistency"
            elif abs(expected_in) <= 12:
                vc["G_in"] = vc["grade_in_pct"] = round(expected_in, 3)
                vc["grade_in_source"] = "corrected from L/K consistency"
    if vc.get("G_in") is not None and vc.get("G_out") is not None:
        vc["algebraic_difference_pct"] = round(vc["G_out"] - vc["G_in"], 3)
        vc["algebraic_difference_abs_pct"] = round(abs(vc["G_out"] - vc["G_in"]), 3)
        _set_vc_type_from_gradients(vc)
    return vc


def vertical_curves_from_annotations(vertical_annotations):
    """Pair K/L annotations into vertical curves and merge page continuations."""
    curves = []
    for sheet_i, ann in enumerate(vertical_annotations):
        ann = sorted(ann, key=lambda a: a["ch_mid"])
        for i, a in enumerate(ann):
            if a["key"] != "K":
                continue
            nearby_l = [
                b for b in ann
                if b["key"] == "L" and abs(b["ch_mid"] - a["ch_mid"]) < 35
            ]
            # A positive '=number' beside G but without a co-located L is
            # usually a gradient-segment length whose L was lost by OCR.
            if a.get("confidence") == "low" and a.get("value") is not None and a["value"] > 0:
                near_g = any(g["key"] == "G" and abs(g["ch_mid"] - a["ch_mid"]) < 40 for g in ann)
                duplicates_l = any(b.get("value") is not None
                                   and abs(b["value"] - a["value"]) < 0.2 for b in nearby_l)
                if near_g and (not nearby_l or duplicates_l):
                    continue
            L = min(nearby_l, key=lambda b: abs(b["ch_mid"] - a["ch_mid"]))["value"] if nearby_l else None
            grades_before = [g for g in ann if g["key"] == "G" and g["ch_mid"] <= a["ch_mid"]]
            grades_after = [g for g in ann if g["key"] == "G" and g["ch_mid"] > a["ch_mid"]]
            prev_ks = [k for k in ann[:i] if k["key"] == "K"]
            next_ks = [k for k in ann[i + 1:] if k["key"] == "K"]
            if prev_ks:
                prev_ch = prev_ks[-1]["ch_mid"]
                grades_before = [g for g in grades_before if g["ch_mid"] > prev_ch]
            if next_ks:
                next_ch = next_ks[0]["ch_mid"]
                grades_after = [g for g in grades_after if g["ch_mid"] < next_ch]
            g_in = grades_before[-1]["value"] if grades_before else None
            g_out = grades_after[0]["value"] if grades_after else None
            numeric_l = [b for b in nearby_l if b.get("value") is not None]
            if a.get("value") is not None and g_in is not None and g_out is not None and numeric_l:
                expected_l = abs(a["value"] * (g_out - g_in))
                L = min(numeric_l, key=lambda b: abs(b["value"] - expected_l))["value"]
            typ = "summit" if a["value"] is not None and a["value"] < 0 else "unknown"
            type_source = (
                "negative K sign before gradient reconciliation"
                if typ == "summit" else
                "not established before gradient reconciliation"
            )
            curves.append({
                "sheet": sheet_i,
                "sheets": [sheet_i],
                "K": a["value"],
                "L": L,
                "G_in": g_in,
                "G_out": g_out,
                "grade_in_pct": g_in,
                "grade_out_pct": g_out,
                "algebraic_difference_pct": (g_out - g_in) if g_in is not None and g_out is not None else None,
                "ch_mid": a["ch_mid"],
                "type": typ,
                "type_source": type_source,
                "continued": False,
            })
            _infer_vc_grades(curves[-1])
    merged = []
    for vc in sorted(curves, key=lambda c: (round(c["ch_mid"], 1), c["sheet"])):
        match = None
        for prev in reversed(merged):
            if prev.get("K") is not None and vc.get("K") is not None:
                same_k = abs(abs(prev["K"]) - abs(vc["K"])) <= 0.1
            else:
                same_k = (prev.get("K") is None and vc.get("K") is None
                          and prev.get("L") == vc.get("L"))
            nearby_sheet = vc["sheet"] - prev["sheets"][-1] <= 1
            nearby_chainage = abs(vc["ch_mid"] - prev["ch_mid"]) <= 350
            if same_k and nearby_sheet and nearby_chainage:
                match = prev
                break
        if match:
            match["sheets"].extend(s for s in vc["sheets"] if s not in match["sheets"])
            match["continued"] = True
            if match.get("L") is None or (vc.get("L") and vc["L"] > match["L"]):
                match["L"] = vc.get("L")
            for key in ("G_in", "G_out", "grade_in_pct", "grade_out_pct", "algebraic_difference_pct"):
                if match.get(key) is None and vc.get(key) is not None:
                    match[key] = vc[key]
            match["ch_mid"] = min(match["ch_mid"], vc["ch_mid"])
            _infer_vc_grades(match)
        else:
            merged.append(vc)
    for i, vc in enumerate(merged, 1):
        _infer_vc_grades(vc)
        vc["id"] = f"VC-{i:02d}"
    return merged


def vertical_gradient_segments_from_annotations(vertical_annotations):
    """Classify G+L schematic elements that are not associated with a K."""
    segments = []
    for sheet_i, ann in enumerate(vertical_annotations):
        ann = sorted(ann, key=lambda a: a["ch_mid"])
        ks = [a for a in ann if a["key"] == "K"]
        grades = [a for a in ann if a["key"] == "G"]
        for length in (a for a in ann if a["key"] == "L"):
            if length.get("value") is None:
                continue
            if any(abs(k["ch_mid"] - length["ch_mid"]) < 90 for k in ks):
                continue
            nearby_g = [g for g in grades if abs(g["ch_mid"] - length["ch_mid"]) < 180]
            grade = min(nearby_g, key=lambda g: abs(g["ch_mid"] - length["ch_mid"])) if nearby_g else None
            if not grade:
                continue
            if any(s["sheet"] == sheet_i and abs(s["ch_mid"] - length["ch_mid"]) < 8
                   for s in segments):
                continue
            segments.append({
                "sheet": sheet_i,
                "ch_mid": length["ch_mid"],
                "G": grade["value"],
                "L": length["value"],
                "type": "constant gradient",
            })
    for i, segment in enumerate(segments, 1):
        segment["id"] = f"VG-{i:02d}"
    return segments




STRUCTURE_TYPE_RE = (
    r"BOX\s+CULVERT|SLAB\s+CULVERT|PIPE\s+CULVERT|CULVERT|"
    r"VOP|LVUP|SVUP|VUP|VIADUCT|MINOR\s+BRIDGE|MAJOR\s+BRIDGE|BRIDGE|ROB|FLYOVER"
)


def _norm_structure_type(raw):
    t = re.sub(r"\s+", " ", raw.strip()).upper()
    mapping = {
        "BOX CULVERT": "Box Culvert",
        "SLAB CULVERT": "Slab Culvert",
        "PIPE CULVERT": "Pipe Culvert",
        "CULVERT": "Culvert",
        "VOP": "VOP",
        "LVUP": "LVUP",
        "SVUP": "SVUP",
        "VUP": "VUP",
        "VIADUCT": "Viaduct",
        "MINOR BRIDGE": "Minor Bridge",
        "MAJOR BRIDGE": "Major Bridge",
        "BRIDGE": "Bridge",
        "ROB": "ROB",
        "FLYOVER": "Flyover",
    }
    return mapping.get(t, t.title())


def _parse_ch(km, metres, km_range=None):
    km_i, m = int(km), float(metres)
    vals = [km_i * 1000 + m]
    for n in (1, 2, 3):
        s = str(km_i)
        if len(s) >= n:
            vals.append(int(s[-n:]) * 1000 + m)
            vals.append(int(s[:n]) * 1000 + m)
    if km_range:
        lo, hi = km_range
        low, high = lo * 1000 - 50, (hi + 1) * 1000 + 50
        in_range = [v for v in vals if low <= v <= high]
        if in_range:
            return min(in_range, key=lambda v: abs(v - vals[0]))
    return vals[0]


def profile_text_structures(pg, sh, km_range=None):
    """Profile callouts are the reference list for structure consistency."""
    text = " ".join(t for _, t in pg.words)
    pat = re.compile(
        rf"(?P<type>{STRUCTURE_TYPE_RE})\s+(?:CUM\s+\w+\s+)?AT\s+CH\.?\s*"
        r"(?P<km>\d{1,5})\+(?P<m>\d{1,3}(?:\.\d+)?)",
        re.I,
    )
    out = []
    for m in pat.finditer(text):
        ch = _parse_ch(m.group("km"), m.group("m"), km_range)
        typ = _norm_structure_type(m.group("type"))
        d = {
            "chainage": ch,
            "proposed_type": typ,
            "profile_present": True,
            "source": "profile_text",
            "page": pg.index,
            "raw": m.group(0),
        }
        _mark_plan_presence(d, sh)
        out.append(d)
    return out


def _mark_plan_presence(d, sh):
    ch = d.get("chainage")
    if ch is None:
        d["plan_present"] = None
        d["consistency_status"] = "Plan not verified"
        return d
    syms = sh.get("culvert_symbols", [])
    near = min(syms, key=lambda s: abs(s - ch), default=None)
    if near is not None and abs(near - ch) <= 10:
        d["symbol_ch"] = round(near, 1)
        d["plan_present"] = True
        d["consistency_status"] = "Present in both"
        return d
    verts = sorted(sh.get("bridge_verticals", []))
    for v0, v1 in zip(verts, verts[1:]):
        if v0 - 50 <= ch <= v1 + 50 and 5 < v1 - v0 < 250:
            d["drawn_span_m"] = round(v1 - v0, 1)
            d["symbol_ch"] = round((v0 + v1) / 2, 1)
            d["plan_present"] = True
            d["consistency_status"] = "Present in both"
            return d
    d["plan_present"] = False
    d["consistency_status"] = "Profile only - plan mark not found"
    return d


def structures(profile_pages, sheets, km_range=None):
    """OCR structure leader tables and match them to drawn symbols."""
    out = []
    for pg, sh in zip(profile_pages, sheets):
        out.extend(profile_text_structures(pg, sh, km_range=km_range))
        for grid in extract.find_table_grids(pg, min_rows=4, col_range=(30, 90)):
            rows = ocr.read_grid_text(pg, grid)
            d = ocr.parse_structure_rows(rows, km_range=km_range)
            if not d:
                continue
            d["page"] = pg.index
            d["profile_present"] = True
            ch = d.get("chainage")
            if ch:
                _mark_plan_presence(d, sh)
            out.append(d)
    # drop duplicates (double-stroked grids)
    ded = []
    for d in sorted(out, key=lambda d: d.get("chainage") or 0):
        if ded and d.get("chainage") and ded[-1].get("chainage") and abs(d["chainage"] - ded[-1]["chainage"]) < 2:
            # Prefer the richer record, but keep profile reference semantics.
            if d.get("source") == "profile_text" and ded[-1].get("source") != "profile_text":
                ded[-1].update({k: v for k, v in d.items() if v is not None})
            elif ded[-1].get("source") == "profile_text":
                ded[-1].update({k: v for k, v in d.items() if v is not None and k not in ("source", "raw")})
            continue
        ded.append(d)
    return ded


def assemble(path, llm_reader=None):
    """Full parse -> model dict ready for validation + rules."""
    doc, plan, profile = open_document(path)
    sheets = profile_bands(profile)
    km_range = None
    if sheets:
        chs = [s for sh in sheets for s in
               (sh.get("ch_min"), sh.get("ch_max")) if s is not None]
        if chs:
            km_range = (int(min(chs) // 1000), int(max(chs) // 1000))
    curves = horizontal_curves(plan, llm_reader, km_range=km_range)
    profile_curves = profile_horizontal_curves(sheets)
    if not curves and profile_curves:
        curves = profile_curves
    elif profile_curves:
        for c in profile_curves:
            c["used_for_rules"] = False
    model = {
        "file": str(path),
        "n_pages": len(doc),
        "plan_pages": [p.index for p in plan],
        "profile_pages": [p.index for p in profile],
        "curves": curves,
        "profile_curves": profile_curves,
        "sheets": sheets,
        "structures": structures(profile, sheets, km_range=km_range),
    }
    # vertical elements from band annotations
    ves = []
    for sh in model["sheets"]:
        ann = band_annotations(sh.get("v_labels", []))
        ann.sort(key=lambda a: a["ch_mid"])
        ves.append(ann)
    model["vertical_annotations"] = ves
    model["vertical_curves"] = vertical_curves_from_annotations(ves)
    model["vertical_gradient_segments"] = vertical_gradient_segments_from_annotations(ves)
    enrich_alignment_geometry(model["curves"], model["sheets"], model["vertical_curves"])
    for sh in model["sheets"]:
        sh.pop("axis", None)  # not JSON-serializable
    return model
