"""Targeted engineering extraction for CAD drawings made from vector strokes.

The searchable-text adapter cannot read these sheets because their visible
letters are drawing paths rather than PDF text.  This module uses geometry to
locate curve-table cells, deskews and OCRs only those tables, and uses OCR
anchors to crop profile structure schedules.  Every accepted record carries
confidence and raw evidence; uncertain fields remain absent or flagged.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import cv2
import fitz
import numpy as np


REPLACEMENTS = str.maketrans({
    "—": "-", "–": "-", "−": "-", ",": ".", "░": "°",
})


def _flat(text: str) -> str:
    return " ".join(text.translate(REPLACEMENTS).replace("�", "°").split())


def _page_image(page: fitz.Page, scale: int = 4) -> np.ndarray:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _curve_cell_groups(image: np.ndarray, scale: int = 4) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        short, long = sorted((width / scale, height / scale))
        extent = abs(cv2.contourArea(contour)) / max(width * height, 1)
        if 2.2 <= short <= 7 and 5 <= long <= 65 and extent >= 0.45:
            theta = (angle if width >= height else angle + 90) % 180
            cells.append({
                "contour": contour, "cx": cx, "cy": cy,
                "long": long * scale, "angle": theta,
            })
    parent = list(range(len(cells)))
    for first_i, first in enumerate(cells):
        for second_i in range(first_i + 1, len(cells)):
            second = cells[second_i]
            angle_difference = abs(first["angle"] - second["angle"])
            angle_difference = min(angle_difference, 180 - angle_difference)
            distance = math.hypot(first["cx"] - second["cx"], first["cy"] - second["cy"])
            threshold = min(110, max(first["long"], second["long"]) * 0.8 + 20)
            if angle_difference <= 5 and distance <= threshold:
                root_a, root_b = _find(parent, first_i), _find(parent, second_i)
                if root_a != root_b:
                    parent[root_b] = root_a
    grouped = defaultdict(list)
    for index, cell in enumerate(cells):
        grouped[_find(parent, index)].append(cell)
    candidates = []
    for group in grouped.values():
        if len(group) < 10:
            continue
        points = np.vstack([cell["contour"] for cell in group])
        x, y, width, height = cv2.boundingRect(points)
        pdf_width, pdf_height = width / scale, height / scale
        if not (20 <= pdf_width <= 140 and 25 <= pdf_height <= 110
                and x / scale > 80 and y / scale < 370):
            continue
        candidates.append({
            "cells": group, "x": x, "y": y, "width": width, "height": height,
            "bbox_pdf": [round(x / scale, 2), round(y / scale, 2),
                         round((x + width) / scale, 2), round((y + height) / scale, 2)],
        })
    return candidates


def _remove_table_lines(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    )
    return cv2.bitwise_not(cv2.bitwise_and(
        binary, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical))
    ))


def _ocr_curve_candidate(image: np.ndarray, candidate: dict, checker) -> list[str]:
    x, y, width, height = (candidate[key] for key in ("x", "y", "width", "height"))
    pad = 20
    crop = image[
        max(0, y - pad):min(image.shape[0], y + height + pad),
        max(0, x - pad):min(image.shape[1], x + width + pad),
    ]
    theta = float(np.median([cell["angle"] for cell in candidate["cells"]]))
    if theta > 90:
        theta -= 180
    matrix = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), theta, 1.0)
    crop = cv2.warpAffine(
        crop, matrix, (crop.shape[1], crop.shape[0]), borderValue=(255, 255, 255)
    )
    crop = np.rot90(crop, 3).copy()
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cleaned = _remove_table_lines(crop)
    texts = []
    for version in (crop, cleaned):
        for psm in (6, 11, 12):
            text = checker.ocr.pytesseract.image_to_string(version, config=f"--psm {psm}")
            if text.strip():
                texts.append(text)
    return texts


def _normalise_chainage(raw_km: str, raw_metres: str, sheet_range: list | None) -> float | None:
    metres = float(raw_metres)
    if not 0 <= metres < 1000:
        return None
    km = int(raw_km)
    if sheet_range:
        expected = int(float(sheet_range[0]) // 1000)
        allowed = {expected, int(float(sheet_range[1]) // 1000)}
        if km not in allowed:
            km = expected
    return km * 1000 + metres


def _chainage_from_text(text: str, sheet_range: list | None) -> float | None:
    matches = re.findall(r"(\d{1,3})\s*[+\-]\s*(\d{3})", text)
    if matches:
        return _normalise_chainage(matches[0][0], matches[0][1], sheet_range)
    expected = int(float(sheet_range[0]) // 1000) if sheet_range else None
    if expected is not None:
        near_ch = re.search(r"Ch[^\d]{0,8}(\d{5,7})", text, re.I)
        if near_ch:
            digits = near_ch.group(1)
            return expected * 1000 + float(digits[-3:])
    return None


def _dms(text: str) -> float | None:
    patterns = (
        # Rotated stroke text often shifts the degree symbol one cell right:
        # "D1 8 26° 56.4" represents D = 8° 26' 56.4".
        r"\bD{1,2}\s*1?\s*(\d{1,2})\s+(\d{1,2})\s*[°º]\s*(\d{1,2}(?:\.\d+)?)",
        # A split leading degree digit such as "3 4° 26 44.3" is 34°.
        r"(?<![+\d])(\d)\s+(\d)\s*[°º]\s*(\d{1,2})\D{1,3}(\d{1,2}(?:\.\d+)?)",
        r"(?<![+\d])(\d{1,3})\s*[°º]\s*(\d{1,2})\s*['’°]?\s*(\d{1,2}(?:\.\d+)?)",
        r"(?:\bD\b|\[D\]|\|D\|)[^\d]{0,8}(\d{1,3})\D{1,4}(\d{1,2})\D{1,4}(\d{1,2}(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            values = match.groups()
            if len(values) == 4:
                degrees, minutes, seconds = float(values[0] + values[1]), float(values[2]), float(values[3])
            else:
                degrees, minutes, seconds = map(float, values)
            if degrees <= 180 and minutes < 60 and seconds < 60:
                return degrees + minutes / 60 + seconds / 3600
    return None


def _labelled_value(text: str, label: str, unit: str = "m") -> float | None:
    match = re.search(
        rf"(?<![A-Za-z])(?:{label}|\[{label}\]|\|{label}\|)(?![A-Za-z])"
        rf"[^\d+\-]{{0,12}}"
        rf"([-+]?\d+(?:\.\d+)?)\s*{unit}(?![A-Za-z])",
        text,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _metre_values(text: str) -> list[float]:
    """Collect metre cells without confusing km/h or a following heading."""
    values = []
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*m(?![A-Za-z])", text, re.I):
        prefix = text[max(0, match.start() - 2):match.start()]
        suffix = text[match.end():match.end() + 7]
        if re.search(r"[Kk]\s*$", prefix) or re.match(r"\s*/\s*h(?:r)?", suffix, re.I):
            continue
        values.append(float(match.group(1)))
    return values


def _parse_curve(text: str, page: int, sheet_range: list | None, bbox: list) -> dict | None:
    clean = text.translate(REPLACEMENTS).replace("�", "°")
    number = re.search(r"CURV\w*\s*[_-]?\s*NO\.?\s*(\d{1,3})", clean, re.I)
    if not number:
        return None
    chainage = _chainage_from_text(clean, sheet_range)
    if chainage is None:
        return None
    delta = _dms(clean)
    speed_match = re.search(r"(\d{2,3})\s*Km\s*/?\s*h", clean, re.I)
    speed = float(speed_match.group(1)) if speed_match and 20 <= float(speed_match.group(1)) <= 160 else None
    normal_camber = bool(re.search(r"\bNC\b", clean, re.I))
    percent_values = [float(value) for value in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*%", clean)]
    e = 2.5 if normal_camber else (percent_values[-1] if percent_values else None)

    # OCR commonly turns boxed CAD labels into LR, PR, |R|, lL c or lLs.
    # Keep matching constrained by the trailing metre unit so ordinary prose
    # cannot silently become an engineering value.
    radius = _labelled_value(clean, r"(?:L|P)?\s*R[Yy]?")
    lc = _labelled_value(clean, r"[lLI|]\s*[cC]")
    ls_label = _labelled_value(clean, r"[lLI|]\s*[sS]")
    es = _labelled_value(clean, r"[eE]\s*[sS]")
    metre_values = _metre_values(clean)
    if es is None and len(metre_values) >= 3 and 0 <= abs(metre_values[-1]) <= 100:
        es = abs(metre_values[-1])
    core = list(metre_values)
    if es is not None and core and abs(abs(core[-1]) - es) < 0.05:
        core.pop()

    if radius is None and len(core) >= 3:
        radius = core[0]
    if lc is None:
        if radius is not None and core and abs(core[0] - radius) < 0.05 and len(core) >= 2:
            lc = abs(core[1])
        elif core:
            lc = abs(core[0])
    transition_values = []
    if ls_label is not None:
        transition_values.append(abs(ls_label))
    if lc is not None:
        start = 0
        for index, value in enumerate(core):
            if abs(abs(value) - lc) < 0.05:
                start = index + 1
                break
        transition_values.extend(
            abs(value) for value in core[start:] if 0 <= abs(value) <= 300
        )
    # Preserve order but remove OCR repeats.
    transitions = []
    for value in transition_values:
        if not transitions or abs(transitions[-1] - value) > 0.05:
            transitions.append(value)
    transitions = transitions[:2]
    ls_effective = sum(transitions) / len(transitions) if transitions else 0.0

    warnings = []
    e_raw_invalid = None
    if e is not None and abs(e) > 12:
        e_raw_invalid = e
        e = None
        warnings.append(
            f"implausible OCR superelevation e={e_raw_invalid:g}%; value withheld for manual review"
        )
    if radius is not None:
        signed_radius = radius
        radius = abs(radius)
    else:
        signed_radius = None
    if delta and lc is not None:
        derived_radius = (lc + ls_effective) / math.radians(delta)
        if (radius is None or radius < 20) and 40 <= derived_radius <= 50000:
            radius = derived_radius
            warnings.append("R derived from Lc, transition lengths and delta after OCR omission")
        elif (radius is not None
              and abs(radius - derived_radius) / max(radius, derived_radius) > 0.08):
            # Do not silently replace a legible table value.  A damaged delta,
            # Lc or transition cell can also cause this mismatch.
            warnings.append(
                f"curve geometry mismatch: OCR R={radius:g}, geometry-derived R={derived_radius:.2f}"
            )
    math_ok = None
    lc_calc = None
    if radius and delta and lc is not None:
        lc_calc = radius * math.radians(delta) - ls_effective
        math_ok = abs(lc_calc - lc) <= max(1.0, lc * 0.02)
    ls_rule = min(transitions) if transitions else None
    curve = {
        "source": "targeted stroke curve-table OCR",
        "curve_no": int(number.group(1)), "page": page, "hip_ch": chainage,
        "R": round(radius, 3) if radius else None,
        "Lc": round(lc, 3) if lc is not None else None,
        "Ls": round(ls_rule, 3) if ls_rule is not None else None,
        "Ls_in": round(transitions[0], 3) if transitions else None,
        "Ls_out": round(transitions[1], 3) if len(transitions) > 1 else None,
        "delta_deg": delta, "Es": es, "e": e, "V": speed,
        "math_ok": math_ok, "lc_calc": round(lc_calc, 2) if lc_calc is not None else None,
        "raw": _flat(clean), "ocr_bbox": bbox, "extraction_warnings": warnings,
    }
    if normal_camber:
        curve["e_raw"] = "NC"
        curve["e_source"] = "NC (Normal Camber) interpreted as 2.5%"
    elif e_raw_invalid is not None:
        curve["e_raw"] = f"{e_raw_invalid:g}%"
        curve["e_source"] = "withheld: implausible targeted OCR superelevation"
    if signed_radius is not None:
        curve["direction_sign"] = -1 if signed_radius < 0 else 1
        curve["direction"] = "negative R" if signed_radius < 0 else "positive R"
        curve["direction_source"] = "signed R in targeted curve-table OCR"
        curve["direction_confidence"] = "medium"
    plausible_radius = radius is None or 40 <= radius <= 50000
    if radius is not None and not plausible_radius:
        warnings.append(f"implausible OCR radius R={radius:g}; manual review required")
    completeness = sum(curve.get(key) is not None for key in ("R", "Lc", "delta_deg", "V", "e"))
    curve["extraction_confidence"] = (
        "high" if completeness >= 5 and math_ok is True and not warnings and plausible_radius
        else "medium" if completeness >= 3 and plausible_radius else "low"
    )
    if radius and delta and ls_rule is not None:
        shift = ls_effective**2 / (24 * radius)
        tangent = (radius + shift) * math.tan(math.radians(delta) / 2) + ls_effective / 2
        curve["Ts"] = round(tangent, 3)
        curve["curve_start_ch"] = round(chainage - tangent, 3)
        curve["curve_end_ch"] = round(chainage + tangent, 3)
        curve["curve_limit_source"] = "curve-table HIP +/- calculated Ts"
    return curve


def _curve_score(curve: dict) -> tuple:
    return (
        curve.get("R") is None or 40 <= curve["R"] <= 50000,
        curve.get("math_ok") is True,
        sum(curve.get(key) is not None for key in ("R", "Lc", "Ls", "delta_deg", "V", "e", "Es")),
        -len(curve.get("extraction_warnings", [])),
    )


def _merge_curve_records(records: list[dict]) -> dict:
    """Combine complementary OCR passes and retain every material conflict."""
    ordered = sorted(records, key=_curve_score, reverse=True)
    merged = dict(ordered[0])
    fields = ("R", "Lc", "Ls", "Ls_in", "Ls_out", "delta_deg", "Es", "e", "V")
    conflicts = {}
    for field in fields:
        observed = []
        support = []
        for record in ordered:
            value = record.get(field)
            if value is None:
                continue
            matching = next((index for index, old in enumerate(observed)
                             if abs(float(value) - float(old))
                             <= max(0.05, abs(float(old)) * 0.005)), None)
            if matching is None:
                observed.append(value)
                support.append(1)
            else:
                support[matching] += 1
        # Complementary OCR is accepted only with independent agreement.
        # A single pass cannot populate a field the highest-ranked record missed.
        if merged.get(field) is None and observed:
            best = max(range(len(observed)), key=lambda index: support[index])
            if support[best] >= 2:
                merged[field] = observed[best]
        if len(observed) > 1:
            conflicts[field] = observed
    if conflicts:
        merged["ocr_field_conflicts"] = conflicts
        merged.setdefault("extraction_warnings", []).append(
            "multiple targeted OCR passes disagree; retained highest-ranked values"
        )
    merged["raw_evidence"] = [record["raw"] for record in ordered]

    radius, lc, delta = merged.get("R"), merged.get("Lc"), merged.get("delta_deg")
    transitions = [value for value in (merged.get("Ls_in"), merged.get("Ls_out"))
                   if value is not None]
    ls_effective = sum(transitions) / len(transitions) if transitions else 0.0
    if radius and lc is not None and delta:
        calculated = radius * math.radians(delta) - ls_effective
        merged["lc_calc"] = round(calculated, 2)
        merged["math_ok"] = abs(calculated - lc) <= max(1.0, lc * 0.02)
    completeness = sum(merged.get(key) is not None for key in ("R", "Lc", "delta_deg", "V", "e"))
    plausible_radius = radius is None or 40 <= radius <= 50000
    merged["extraction_confidence"] = (
        "high" if completeness >= 5 and merged.get("math_ok") is True
        and not merged.get("extraction_warnings") and not conflicts and plausible_radius
        else "medium" if completeness >= 3 and plausible_radius else "low"
    )
    return merged


def _line_groups(data: dict) -> list[tuple[str, list[int]]]:
    grouped = defaultdict(list)
    for index, text in enumerate(data["text"]):
        if text.strip():
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            grouped[key].append(index)
    return [(" ".join(data["text"][i].strip() for i in indices), indices) for indices in grouped.values()]


def _structure_anchors(image: np.ndarray, checker, layout: str) -> list[tuple[float, float]]:
    data = checker.ocr.pytesseract.image_to_data(
        image, config="--psm 11", output_type=checker.ocr.pytesseract.Output.DICT
    )
    height = image.shape[0] / 4
    anchors = []
    for text, indices in _line_groups(data):
        if not re.search(r"(?:STR.*TYPE|EXIST.*STR)", text, re.I):
            continue
        x0 = min(data["left"][i] for i in indices) / 4
        y0 = min(data["top"][i] for i in indices) / 4
        if layout != "profile_only" and not height * 0.32 <= y0 <= height * 0.48:
            continue
        if re.search(r"EXIST.*STR", text, re.I):
            y0 -= 4
        if any(abs(x0 - x) < 9 and abs(y0 - y) < 9 for x, y in anchors):
            continue
        anchors.append((x0, y0))
    return anchors


def _structure_type(text: str) -> str | None:
    patterns = (
        (r"Minor\s+Bridge\s+(?:cum|Cum)\s+(?:Animal\s+Underpass|AUP)", "Minor Bridge cum Animal Underpass"),
        (r"Minor\s+Bridge", "Minor Bridge"),
        (r"Grade\s+Separator", "Grade Separator"),
        (r"Animal\s+Underp\w+", "Animal Underpass"),
        (r"Box\s+Culvert", "Box Culvert"),
        (r"Arch\s+Culvert", "Arch Culvert"),
        (r"(?:Slab|Stab)\s+Culvert", "Slab Culvert"),
        (r"Pipe\s+Culvert", "Pipe Culvert"),
        (r"Culvert", "Culvert"),
    )
    for pattern, value in patterns:
        if re.search(pattern, text, re.I):
            return value
    return None


def _parse_structure(text: str, page: int, sheet_range: list | None,
                     paired_plan_page: int | None, bbox: list) -> dict | None:
    clean = text.translate(REPLACEMENTS)
    proposed_line = re.search(r"PROP\.?\s*CH\.?([^\n]*)", clean, re.I)
    chainage = _chainage_from_text(proposed_line.group(1), sheet_range) if proposed_line else None
    if chainage is None and proposed_line and sheet_range:
        digits = re.search(r"(\d{5,7})", proposed_line.group(1))
        if digits:
            chainage = int(float(sheet_range[0]) // 1000) * 1000 + int(digits.group(1)[-3:])
    proposed_type = _structure_type(clean.split("EXIST.STR", 1)[0]) or _structure_type(clean)
    if chainage is None or proposed_type is None:
        return None
    span_line = re.search(r"PROP\.?\s*SPAN([^\n]*)", clean, re.I)
    proposal_line = re.search(r"IMPR\.?\s*PROP\w*([^\n]*)", clean, re.I)
    existing_line = re.search(r"EXIST\.?\s*STR([^\n]*)", clean, re.I)
    str_number = re.search(r"\b(\d{1,3}/\d{1,4})\b", clean)
    return {
        "chainage": chainage, "proposed_type": proposed_type,
        "proposed_size": span_line.group(1).strip(" |:-") if span_line else None,
        "proposed_span": span_line.group(1).strip(" |:-") if span_line else None,
        "proposal": proposal_line.group(1).strip(" |:-") if proposal_line else None,
        "improvement_proposal": proposal_line.group(1).strip(" |:-") if proposal_line else None,
        "existing": _structure_type(existing_line.group(1)) if existing_line else None,
        "str_no": str_number.group(1) if str_number else None,
        "page": page, "profile_present": True, "plan_present": None,
        "paired_plan_page": paired_plan_page,
        "consistency_status": "Profile schedule extracted; plan presence requires spatial reconciliation",
        "source": "targeted stroke profile-schedule OCR",
        "extraction_confidence": "medium (targeted stroke OCR)",
        "ocr_bbox": bbox, "raw": _flat(clean),
    }


def extract(doc: fitz.Document, checker, layout_pages: list[dict]) -> dict:
    """Return stroke-font curves, structures and auditable QA metadata."""
    layout_by_page = {item["page"]: item for item in layout_pages}
    curves, structures = [], []
    qa = {"curve_candidates": 0, "curve_rejected": [], "structure_candidates": 0,
          "structure_rejected": []}
    for page_number, page in enumerate(doc):
        layout = layout_by_page.get(page_number, {})
        sheet_range = layout.get("sheet_range")
        image = _page_image(page)
        if layout.get("layout") != "profile_only":
            for candidate in _curve_cell_groups(image):
                qa["curve_candidates"] += 1
                records = [
                    _parse_curve(text, page_number, sheet_range, candidate["bbox_pdf"])
                    for text in _ocr_curve_candidate(image, candidate, checker)
                ]
                records = [record for record in records if record]
                if not records:
                    qa["curve_rejected"].append({
                        "page": page_number, "bbox": candidate["bbox_pdf"],
                        "reason": "no validated CURVE NO. and chainage",
                    })
                    continue
                curves.append(_merge_curve_records(records))

        if layout.get("layout") in {"profile_capable", "profile_only"}:
            paired_plan_page = page_number
            if layout.get("layout") == "profile_only" and page_number > 0:
                paired_plan_page = page_number - 1
            for x0, y0 in _structure_anchors(image, checker, layout.get("layout")):
                qa["structure_candidates"] += 1
                scale = 4
                x_start, y_start = max(0, int((x0 - 5) * scale)), max(0, int((y0 - 5) * scale))
                crop = image[
                    y_start:min(image.shape[0], int((y0 + 34) * scale)),
                    x_start:min(image.shape[1], int((x0 + 110) * scale)),
                ]
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                texts = [
                    checker.ocr.pytesseract.image_to_string(crop, config=f"--psm {psm}")
                    for psm in (6, 11, 12)
                ]
                parsed = [
                    _parse_structure(text, page_number, sheet_range, paired_plan_page,
                                     [round(x0, 2), round(y0, 2), round(x0 + 110, 2), round(y0 + 34, 2)])
                    for text in texts
                ]
                parsed = [record for record in parsed if record]
                if parsed:
                    structures.append(max(parsed, key=lambda record: sum(
                        bool(record.get(key)) for key in ("proposed_type", "proposed_size", "proposal", "existing")
                    )))
                else:
                    qa["structure_rejected"].append({
                        "page": page_number, "bbox": [x0, y0, x0 + 110, y0 + 34],
                        "reason": "proposed type or proposed chainage not validated",
                    })

    curve_by_number = {}
    for curve in curves:
        number = curve["curve_no"]
        if number not in curve_by_number or _curve_score(curve) > _curve_score(curve_by_number[number]):
            curve_by_number[number] = curve
    curves = sorted(curve_by_number.values(), key=lambda curve: curve["hip_ch"])
    deduplicated_structures = []
    for structure in sorted(structures, key=lambda record: record["chainage"]):
        if any(abs(previous["chainage"] - structure["chainage"]) < 2 for previous in deduplicated_structures):
            continue
        deduplicated_structures.append(structure)
    qa["curve_accepted"] = len(curves)
    qa["structure_accepted"] = len(deduplicated_structures)
    qa["curve_confidence"] = dict(
        sorted({level: sum(curve.get("extraction_confidence") == level for curve in curves)
                for level in ("high", "medium", "low")}.items())
    )
    qa["curve_field_conflicts"] = sum(bool(curve.get("ocr_field_conflicts")) for curve in curves)
    return {"curves": curves, "structures": deduplicated_structures, "qa": qa}
