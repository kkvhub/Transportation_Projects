"""
ocr.py — OCR of stroke-drawn (CAD SHX) text via cell segmentation + Tesseract.

Strategy proven on HP-1: render each bordered table cell at high zoom,
binarize, OCR as a single line with a whitelist. Band labels are OCR'd
from full band strips using image_to_data word boxes.
"""
from __future__ import annotations
import re
import fitz
from PIL import Image

try:
    import pytesseract
    HAVE_TESS = True
except ImportError:
    HAVE_TESS = False

TESS_STATUS = "ok"
if HAVE_TESS:
    import os as _os
    import shutil as _shutil
    if _shutil.which("tesseract") is None:
        _candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            _os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            _os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
            "/usr/bin/tesseract", "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
        for _p in _candidates:
            if _os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break
        else:
            HAVE_TESS = False
            TESS_STATUS = (
                "Tesseract OCR engine not found. Install it from "
                "https://github.com/UB-Mannheim/tesseract/wiki (Windows) or "
                "'sudo apt install tesseract-ocr' (Linux), then restart the app. "
                "If installed to a custom folder, add it to PATH."
            )
else:
    TESS_STATUS = "Python package 'pytesseract' missing — run: pip install pytesseract"

CELL_WHITELIST = "0123456789.+-%:ENVRTLcseKmphHIPCO"
BAND_WHITELIST = "0123456789.+-%=eGKLRTsm"


def _img(pg, rect, zoom=12, thresh=140):
    pix = pg.pixmap(clip=rect, zoom=zoom)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    return img.point(lambda v: 0 if v < thresh else 255)


def ocr_cell(pg, rect, whitelist=CELL_WHITELIST):
    if not HAVE_TESS:
        return ""
    cfg = "--psm 7 -c tessedit_char_whitelist=" + whitelist
    return pytesseract.image_to_string(_img(pg, rect), config=cfg).strip()


def read_grid(pg, grid):
    """OCR every row of a detected table grid -> list of [label, value] rows."""
    rows = []
    hl, cols = grid["rows"], grid["cols"]
    x0, x1 = grid["x0"], grid["x1"]
    split = cols[1] if len(cols) >= 3 else None
    for i in range(len(hl) - 1):
        y0, y1 = hl[i] + 0.4, hl[i + 1] - 0.4
        if y1 - y0 < 2.5:
            continue
        if split and i >= 2:
            rows.append([ocr_cell(pg, fitz.Rect(x0 + 0.6, y0, split - 0.6, y1)),
                         ocr_cell(pg, fitz.Rect(split + 0.6, y0, x1 - 0.6, y1))])
        else:
            rows.append([ocr_cell(pg, fitz.Rect(x0 + 0.6, y0, x1 - 0.6, y1))])
    return rows


def band_words(pg, axis, y0, y1, zoom=8):
    """OCR a horizontal band strip; returns [(text, ch_start, ch_end)]."""
    if not HAVE_TESS:
        return []
    x_left, x_right = axis.x(axis.ch_min) - 2, axis.x(axis.ch_max) + 2
    rect = fitz.Rect(x_left, y0, x_right, y1)
    img = _img(pg, rect, zoom=zoom)
    cfg = "--psm 11 -c tessedit_char_whitelist=" + BAND_WHITELIST
    data = pytesseract.image_to_data(img, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    out = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt or int(data.get("conf", ["-1"] * len(data["text"]))[i]) < 30:
            continue
        px = data["left"][i] / zoom + x_left
        px2 = (data["left"][i] + data["width"][i]) / zoom + x_left
        out.append((txt, axis.ch(px), axis.ch(px2)))
    return out


def band_line_words(pg, axis, y0, y1, zoom=20, thresh=170, pad_up=1, pad_down=0):
    """High-zoom line OCR fallback for tiny vertical schematic labels."""
    if not HAVE_TESS:
        return []
    x_left, x_right = axis.x(axis.ch_min) - 2, axis.x(axis.ch_max) + 2
    rect = fitz.Rect(x_left, max(0, y0 - pad_up), x_right, y1 + pad_down)
    img = _img(pg, rect, zoom=zoom, thresh=thresh)
    cfg = "--psm 6 -c tessedit_char_whitelist=" + BAND_WHITELIST
    data = pytesseract.image_to_data(img, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    out = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        px = data["left"][i] / zoom + x_left
        px2 = (data["left"][i] + data["width"][i]) / zoom + x_left
        out.append((txt, axis.ch(px), axis.ch(px2)))
    return out


def band_column_words(pg, axis, y0, y1, center_ch, half_width_ch=45, zoom=30, thresh=170):
    """Focused OCR around one vertical-schematic element."""
    if not HAVE_TESS:
        return []
    x0, x1 = axis.x(center_ch - half_width_ch), axis.x(center_ch + half_width_ch)
    rect = fitz.Rect(max(0, x0), max(0, y0 - 3), min(pg.rect.width, x1), y1 + 1)
    img = _img(pg, rect, zoom=zoom, thresh=thresh)
    cfg = "--psm 6 -c tessedit_char_whitelist=" + BAND_WHITELIST
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        px = data["left"][i] / zoom + rect.x0
        px2 = (data["left"][i] + data["width"][i]) / zoom + rect.x0
        out.append((txt, axis.ch(px), axis.ch(px2)))
    return out


CHAIN_RE_OCR = re.compile(r"(\d{1,3})\+(\d{3}(?:\.\d+)?)")


def find_chainage_band_ocr(pg, zoom=8, min_hits=8, region_frac=(0.45, 1.0)):
    """Locate the chainage band by OCR when the PDF has no embedded text for
    it (fully stroke-drawn CAD numerals — seen on some plan-and-profile
    exports where labels like 'KM STONE' are real text but chainage digits
    are drawn as vector strokes). Chainage labels in this band are often
    rotated 90 degrees to fit narrow column spacing, so each candidate crop
    is tried both flat and rotated both ways, keeping whichever orientation
    yields the most valid NNN+NNN matches.

    Returns (hits, y0, y1) where hits = [(value_m, x_center), ...], or None.
    """
    if not HAVE_TESS:
        return None
    w, h = pg.rect.width, pg.rect.height
    from . import extract as _extract
    ladder, lx0, lx1 = _extract.find_band_row_ladder(pg)
    candidates = []
    if len(ladder) >= 2:
        candidates.append((ladder[-2], ladder[-1], lx0, lx1, zoom))
    # last-resort broad scan (no row ladder found at all — most likely a
    # plan sheet with no chainage band to find). Cheap low-res pass: this
    # runs on every non-profile plan sheet during classification, so a
    # high zoom here is wasted time chasing a band that isn't there.
    candidates.append((h * region_frac[0], h * region_frac[1], 0, w, 3))

    best = None
    for y0, y1, x0b, x1b, zm in candidates:
        rect = fitz.Rect(x0b, y0, x1b, y1)
        img = _img(pg, rect, zoom=zm, thresh=150)
        cfg = "--psm 6 -c tessedit_char_whitelist=0123456789+."
        for rot in (-90, 90, 0):
            timg = img.rotate(rot, expand=True) if rot else img
            data = pytesseract.image_to_data(timg, config=cfg, output_type=pytesseract.Output.DICT)
            pts = []
            for i, txt in enumerate(data["text"]):
                m = CHAIN_RE_OCR.fullmatch(txt.strip())
                if not m:
                    continue
                val = int(m.group(1)) * 1000 + float(m.group(2))
                if rot == -90:
                    px = data["top"][i] + data["height"][i] / 2.0
                elif rot == 90:
                    px = timg.height - (data["top"][i] + data["height"][i] / 2.0)
                else:
                    px = data["left"][i] + data["width"][i] / 2.0
                pts.append((val, px / zm + rect.x0))
            if len(pts) >= min_hits and (best is None or len(pts) > len(best[0])):
                best = (pts, y0, y1)
        if best is not None:
            break  # a ladder-based band that OCR'd cleanly wins outright
    if best is None:
        return None
    hits, y0, y1 = best
    return hits, y0, y1


NUM = r"[-+]?\d+(?:\.\d+)?"


def _chainage_value(km_text, metre_text, km_range=None):
    km, metres = int(km_text), float(metre_text)
    candidates = [km * 1000 + metres]
    if km_range:
        lo, hi = km_range
        # Some OCR reads 0+217 as 04+217 or 004+217 on local-chainage
        # sheets. Try suffix km digits and also plain metres for 0-1 km jobs.
        for n in (1, 2, 3):
            if len(km_text) >= n:
                candidates.append(int(km_text[-n:]) * 1000 + metres)
                candidates.append(int(km_text[:n]) * 1000 + metres)
        if lo <= 0 <= hi:
            candidates.append(metres)
        low_ch, high_ch = lo * 1000 - 50, (hi + 1) * 1000 + 50
        in_range = [c for c in candidates if low_ch <= c <= high_ch]
        if in_range:
            return min(in_range, key=lambda c: abs(c - metres) if hi <= 1 else 0)
    return candidates[0]


def _clean_curve_values(d):
    """Apply conservative road-geometry sanity cleanup to OCR'd HIP cells."""
    notes = []
    for key in ("V", "R", "Ts", "Lc", "Ls", "e"):
        if d.get(key) is not None:
            d[key] = abs(float(d[key]))
    if d.get("Lc") is not None and any("Le', '-" in str(r) or "Lc', '-" in str(r)
                                       for r in d.get("ocr_rows", [])):
        notes.append("negative curve length OCR sign corrected")
    if d.get("e") is not None:
        original = d["e"]
        while d["e"] > 10:
            d["e"] = d["e"] / 10.0
        if d["e"] != original:
            notes.append(f"superelevation OCR scaled from {original:g}% to {d['e']:g}%")
    if d.get("R") is not None and d["R"] < 20:
        d["extraction_confidence"] = "low"
        notes.append("radius below plausible extraction floor")
    if notes:
        d["extraction_notes"] = notes
    return d


def parse_hip_rows(rows, pg=None, grid=None, km_range=None):
    """Turn OCR'd HIP-table rows into a dict; positional fallback for lost labels."""
    d = {}
    d["ocr_rows"] = rows
    for r in rows:
        line = "".join(c for c in r if c).replace(" ", "")
        m = re.search(r"HIPCH:?(\d{1,5})\+(\d{1,3}(?:\.\d+)?)", line)
        if m:
            d["hip_ch"] = _chainage_value(m.group(1), m.group(2), km_range)
    ORDER = ["E", "N", "V", "delta", "R", "Ts", "Lc", "Ls", "e"]
    LBL = {"E": "E", "N": "N", "V": "V", "R": "R", "Ts": "Ts", "TS": "Ts",
           "Lc": "Lc", "Le": "Lc", "LC": "Lc", "Ls": "Ls", "LS": "Ls", "e": "e"}
    data_rows = [r for r in rows if len(r) == 2]
    for i, r in enumerate(data_rows):
        label, val = r[0].strip(), r[1].strip().replace(" ", "")
        key = LBL.get(label) or (ORDER[i] if i < len(ORDER) else None)
        if key is None:
            continue
        if key == "delta":
            d["delta_raw"] = val
            continue
        v = re.search(NUM, val)
        if v:
            d[key] = float(v.group())
    _clean_curve_values(d)
    if all(d.get(k) is not None for k in ("R", "Lc", "Ls")):
        import math as _m
        d["delta_deg"] = round(_m.degrees(d["Lc"] / d["R"] + d["Ls"] / d["R"]), 4)
        d["delta_source"] = "derived from Lc,R,Ls"
        ocr_delta = parse_dms(d.get("delta_raw", ""))
        if ocr_delta and abs(ocr_delta - d["delta_deg"]) > 0.05:
            d["delta_ocr_mismatch"] = ocr_delta
    elif d.get("delta_raw"):
        d["delta_deg"] = parse_dms(d["delta_raw"])
        d["delta_source"] = "ocr"
    if pg is not None and grid is not None and len(grid["rows"]) > 2:
        import fitz as _f
        from collections import Counter
        hl = grid["rows"]
        cell = _f.Rect(grid["x0"] + 0.6, hl[1] + 0.4, grid["x1"] - 0.6, hl[2] - 0.4)
        votes = []
        for zoom, th in ((12, 120), (16, 120), (16, 150), (16, 180), (16, None)):
            if not HAVE_TESS:
                break
            import pytesseract as _t
            pix = pg.pixmap(clip=cell, zoom=zoom)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
            if th:
                img = img.point(lambda v, t=th: 0 if v < t else 255)
            txt = _t.image_to_string(
                img, config="--psm 7 -c tessedit_char_whitelist=0123456789.+:HIPC")
            m = re.search(r"(\d{2,5})\+(\d{1,3}(?:\.\d+)?)$", txt.strip().replace(" ", ""))
            if not m:
                continue
            digits, metres = m.group(1), float(m.group(2))
            cands = {int(digits)} | {int(digits[-k:]) for k in (2, 3) if len(digits) >= k}
            cands |= {int(digits[:k]) for k in (2, 3) if len(digits) >= k}
            if km_range:
                lo, hi = km_range
                if hi <= 1:
                    cands = {c for c in cands if lo <= c <= hi}
                else:
                    cands = {c for c in cands if lo - 3 <= c <= hi + 3}
            for c in cands:
                votes.append(c * 1000 + metres)
            if km_range and km_range[0] <= 0 <= km_range[1]:
                votes.append(metres)
        if votes:
            best, n = Counter(votes).most_common(1)[0]
            majority = n > len(votes) / 2
            prior = d.get("hip_ch")
            if prior is None or majority:
                d["hip_ch"] = best
                d["ch_votes"] = f"{n}/{len(votes)}"
                if not majority:
                    d["ch_confidence"] = "low"
            else:
                # minority OCR guess — don't let it clobber a value already
                # parsed from the "HIPCH:" text line; just flag it as low
                # confidence for review instead.
                d["ch_votes"] = f"{n}/{len(votes)} (kept HIPCH: line reading)"
                d["ch_confidence"] = "low"
    return d


def parse_dms(raw):
    """'603245.60' -> 60.5460 (dd mm ss.ss, symbols stripped by OCR)."""
    if not raw:
        return None
    raw = raw.replace(" ", "")
    m = re.fullmatch(r"(\d{1,3})(\d{2})(\d{2}\.\d+)", raw)
    if not m:
        m = re.fullmatch(r"(\d{1,3})[^\d](\d{1,2})[^\d](\d{1,2}\.?\d*)", raw)
    if not m:
        return None
    dd, mm, ss = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if dd > 180 or mm >= 60 or ss >= 60:
        return None
    return dd + mm / 60 + ss / 3600


def read_grid_text(pg, grid, extra_above=8.0):
    """Generic (letters allowed) OCR of a grid; includes the row just above
    the detected stack — structure tables carry 'Chainage at Km' there."""
    if not HAVE_TESS:
        return []
    rows = []
    hl, cols = grid["rows"], grid["cols"]
    x0, x1 = grid["x0"], grid["x1"]
    split = cols[1] if len(cols) >= 3 else None

    def _cell(rect):
        return pytesseract.image_to_string(_img(pg, rect), config="--psm 7").strip()

    if extra_above:
        top = fitz.Rect(x0 + 0.5, hl[0] - extra_above, x1 - 0.5, hl[0] - 0.3)
        generic = _cell(top)
        digits = pytesseract.image_to_string(
            _img(pg, top), config="--psm 7 -c tessedit_char_whitelist=0123456789+.:CHKm").strip()
        rows.append([generic, digits])
    for i in range(len(hl) - 1):
        y0, y1 = hl[i] + 0.3, hl[i + 1] - 0.3
        if y1 - y0 < 2.5:
            continue
        if split:
            rows.append([_cell(fitz.Rect(x0 + 0.5, y0, split - 0.5, y1)),
                         _cell(fitz.Rect(split + 0.5, y0, x1 - 0.5, y1))])
        else:
            rows.append([_cell(fitz.Rect(x0 + 0.5, y0, x1 - 0.5, y1))])
    return rows


def parse_structure_rows(rows, km_range=None):
    """Parse a structure leader-table into a dict, or None if not one."""
    flat = " | ".join(" ".join(c for c in r if c) for r in rows)
    if "Str" not in flat and "Culvert" not in flat and "Bridge" not in flat:
        return None
    d = {"raw": flat}
    def _km_ok(km):
        if not km_range:
            return True
        lo, hi = km_range
        return lo - 3 <= km <= hi + 3

    found = False
    for m in re.finditer(r"(\d{2,5})\+(\d{2,3}(?:\.\d+)?)", flat.replace(" ", "")):
        digits, metres = m.group(1), float(m.group(2))
        cands = {int(digits)}
        for k in (2, 3):
            if len(digits) >= k:
                cands.add(int(digits[-k:]))
                cands.add(int(digits[:k]))
        ok = sorted({c for c in cands if _km_ok(c)})
        if len(ok) >= 1:
            d["chainage"] = ok[0] * 1000 + metres
            if int(digits) != ok[0]:
                d["chainage_confidence"] = "low"
            found = True
            break
    if not found:
        for m in re.finditer(r"(\d{3})\d?(\d{3})(?:\.\d+)?", flat.replace(" ", "")):
            if _km_ok(int(m.group(1))):
                d["chainage"] = int(m.group(1)) * 1000 + float(m.group(2))
                d["chainage_confidence"] = "low"
                break
    m = re.search(r"(\d{1,4})/(\d{1,2})", flat)
    if m:
        d["str_no"] = f"{m.group(1)[-3:]}/{m.group(2)}"
    for r in rows:
        line = " ".join(c for c in r if c)
        low = line.lower()
        val = r[1].strip() if len(r) == 2 else ""
        if "existing" in low:
            d["existing"] = val or line.split("Size", 1)[-1].strip(" |:-")
        elif "proposed type" in low:
            d["proposed_type"] = val or line.split("Type", 1)[-1].strip(" |:-")
        elif "proposed size" in low:
            d["proposed_size"] = val or line.split("Size", 1)[-1].strip(" |:-")
        elif "proposal" in low or "praposal" in low:
            d["proposal"] = (val or line.split("l", 1)[-1]).strip(" |:-,")
    m = re.search(r"x\s*([\d.]+)\s*m", d.get("proposed_size", ""), re.I)
    if m:
        d["span_m"] = float(m.group(1))
    d["is_bridge"] = "bridge" in flat.lower()
    return d if ("chainage" in d or "str_no" in d) else None
