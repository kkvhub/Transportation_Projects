"""Compatibility extraction for searchable combined plan/profile sheets.

This adapter only supplements empty or incomplete records produced by the
preserved 02_Tool parser.  It reads explicit engineering labels from the PDF
text layer and returns the same model shapes expected by the original IRC
rules engine.  It does not infer compliance from ML detections.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image

from stroke_drawing_adapter import extract as extract_stroke_engineering


CURVE_HEADING_RE = re.compile(r"\bCURVE\s*NO\.?\s*(\d+)\b", re.IGNORECASE)
CHAINAGE_RE = re.compile(r"\b(?:Ch\.?\s*)?(\d{1,3})\+(\d{1,3}(?:\.\d+)?)\b", re.IGNORECASE)
PROFILE_LABEL_RE = re.compile(
    r"^(?P<key>Lv|L|K|G)\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)(?:m|%)?$",
    re.IGNORECASE,
)


def _normalise_normal_camber(curve: dict) -> bool:
    """Interpret an explicit curve-table NC cell as 2.5% normal camber."""
    if curve.get("e") is not None:
        return False
    evidence = str(curve.get("raw") or "")
    if curve.get("ocr_rows"):
        evidence += " " + " ".join(
            str(cell) for row in curve["ocr_rows"] for cell in row
        )
    if not re.search(r"\bNC\b", evidence, re.IGNORECASE):
        return False
    curve["e"] = 2.5
    curve["e_raw"] = "NC"
    curve["e_source"] = "NC (Normal Camber) interpreted as 2.5%"
    return True


def _field(text: str, key: str) -> float | None:
    match = re.search(
        rf"\b{re.escape(key)}\s*=?\s*([-+]?\d+(?:\.\d+)?)\s*m?\b",
        text,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _dms_degrees(text: str) -> float | None:
    match = re.search(
        r"\bD\s*(\d+(?:\.\d+)?)\s*[°º]\s*(\d+(?:\.\d+)?)\s*['’]\s*"
        r"(\d+(?:\.\d+)?)\s*[\"”]",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    degrees, minutes, seconds = map(float, match.groups())
    return degrees + minutes / 60.0 + seconds / 3600.0


def _horizontal_curves(doc: fitz.Document, checker) -> list[dict]:
    curves = []
    for page_number, page in enumerate(doc):
        for block in page.get_text("blocks"):
            text = " ".join(block[4].split())
            heading = CURVE_HEADING_RE.search(text)
            if not heading:
                continue
            chainage = CHAINAGE_RE.search(text)
            signed_radius = _field(text, "R")
            radius = abs(signed_radius) if signed_radius is not None else None
            length_spiral = _field(text, "Ls")
            length_circular = _field(text, "Lc")
            delta = _dms_degrees(text)
            speed_match = re.search(r"\bV\s*=?\s*(\d+(?:\.\d+)?)\s*kmph\b", text, re.IGNORECASE)
            percent = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", text)
            normal_camber = bool(re.search(r"\bNC\b", text, re.IGNORECASE))
            hip = None
            if chainage:
                hip = int(chainage.group(1)) * 1000 + float(chainage.group(2))
            curve = {
                "source": "searchable PDF CURVE NO. table",
                "curve_no": int(heading.group(1)),
                "page": page_number,
                "hip_ch": hip,
                "R": radius,
                "Ls": length_spiral,
                "Lc": length_circular,
                "delta_deg": delta,
                "Es": _field(text, "Es"),
                "e": float(percent.group(1)) if percent else (2.5 if normal_camber else None),
                "V": float(speed_match.group(1)) if speed_match else None,
                "raw": text,
                "extraction_confidence": "high (embedded text table)",
            }
            if normal_camber and not percent:
                curve["e_raw"] = "NC"
                curve["e_source"] = "NC (Normal Camber) interpreted as 2.5%"
            if signed_radius is not None:
                curve["direction_sign"] = -1 if signed_radius < 0 else 1
                curve["direction"] = "negative R" if signed_radius < 0 else "positive R"
                curve["direction_source"] = "signed radius in CURVE NO. table"
                curve["direction_confidence"] = "high"
            curve.update(checker.parse._verify_curve_math(curve))
            if radius and delta is not None and length_spiral is not None:
                theta = length_spiral / (2.0 * radius)
                shift = length_spiral**2 / (24.0 * radius)
                tangent = (radius + shift) * math.tan(math.radians(delta) / 2.0) + length_spiral / 2.0
                curve["Ts"] = round(tangent, 3)
                if hip is not None:
                    curve["curve_start_ch"] = round(hip - tangent, 3)
                    curve["curve_end_ch"] = round(hip + tangent, 3)
                    curve["curve_limit_source"] = "CURVE NO. table HIP +/- calculated Ts"
            if radius:
                curves.append(curve)
    curves.sort(key=lambda item: item.get("hip_ch") or 0)
    return curves


def _axis_for_page(checker, doc: fitz.Document, page_number: int):
    page = checker.extract.Page(doc, page_number)
    return page, checker.extract.ChainageAxis(page)


class _GridAxis:
    """Minimal x/chainage axis compatible with the preserved OCR helpers."""

    def __init__(self, x0: float, x1: float, ch_min: float, ch_max: float):
        self.ch_min, self.ch_max = ch_min, ch_max
        self._scale = (ch_max - ch_min) / (x1 - x0)
        self._offset = ch_min - x0 * self._scale
        self.axis_source = "profile grid plus OCR sheet range"

    def ch(self, x: float) -> float:
        return x * self._scale + self._offset

    def x(self, chainage: float) -> float:
        return (chainage - self._offset) / self._scale


def _profile_grid(page) -> tuple[list[float], float, float] | None:
    """Find the repeated full-width profile/data-table rows."""
    groups = defaultdict(list)
    width, height = page.rect.width, page.rect.height
    for start, end, _ in page.lines():
        if abs(start.y - end.y) > 0.5:
            continue
        x0, x1 = sorted((start.x, end.x))
        y = (start.y + end.y) / 2.0
        if x1 - x0 < width * 0.65 or not height * 0.35 <= y <= height * 0.90:
            continue
        groups[(round(x0, 1), round(x1, 1))].append(round(y, 1))
    if not groups:
        return None
    (x0, x1), raw_rows = max(groups.items(), key=lambda item: len(set(item[1])))
    rows = sorted(set(raw_rows))
    if len(rows) < 8:
        return None
    return rows, x0, x1


def _pixmap_image(pixmap: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pixmap.alpha else "RGB"
    return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples).convert("RGB")


def _ocr_sheet_range(page, checker) -> tuple[float, float, str] | None:
    """Read the explicit '(Ch. N+000 To Ch.M+000)' sheet-range title."""
    if not checker.ocr.HAVE_TESS:
        return None
    width, height = page.rect.width, page.rect.height
    clips = (
        fitz.Rect(0, height * 0.84, width, height),
        fitz.Rect(0, 0, width, height),
    )
    pattern = re.compile(
        r"(?:Ch\.?\s*)?(\d{1,3})\s*\+\s*(\d{3})\s*(?:To|TO|-|–)\s*"
        r"(?:Ch\.?\s*)?(\d{1,3})\s*\+\s*(\d{3})",
        re.IGNORECASE,
    )
    for clip in clips:
        pixmap = page.pixmap(clip=clip, zoom=4)
        text = checker.ocr.pytesseract.image_to_string(_pixmap_image(pixmap), config="--psm 11")
        match = pattern.search(" ".join(text.split()))
        if match:
            start = int(match.group(1)) * 1000 + int(match.group(2))
            end = int(match.group(3)) * 1000 + int(match.group(4))
            if 0 < end - start <= 5000:
                return float(start), float(end), " ".join(match.group(0).split())
    return None


def _composite_band_annotations(words: list[tuple], axis_source: str) -> list[dict]:
    """Parse compact OCR such as 'K=135.22L=311.00' conservatively."""
    candidates = []
    for raw, ch0, ch1 in words:
        cleaned = re.sub(r"LG\s*=", "L=", raw.strip(), flags=re.IGNORECASE)
        matches = list(re.finditer(r"(?P<key>K|G|L)\s*=?\s*(?P<value>[-+]?\d+(?:\.\d+)?)", cleaned, re.I))
        k_positions = [match.start() for match in matches if match.group("key").upper() == "K"]
        for match in matches:
            key = match.group("key").upper()
            value = float(match.group("value"))
            if key == "G" and abs(value) > 20:
                continue
            if key == "K" and not 5 <= abs(value) <= 5000:
                continue
            if key == "L" and not 10 <= value <= 1000:
                continue
            fraction = (match.start() + match.end()) / 2.0 / max(len(cleaned), 1)
            ch_mid = ch0 + (ch1 - ch0) * fraction
            # Curve length printed after K belongs to that same curve centre.
            prior_k = [position for position in k_positions if position < match.start()]
            if key == "L" and prior_k:
                k_match = max(
                    (item for item in matches if item.group("key").upper() == "K" and item.start() < match.start()),
                    key=lambda item: item.start(),
                )
                k_fraction = (k_match.start() + k_match.end()) / 2.0 / max(len(cleaned), 1)
                ch_mid = ch0 + (ch1 - ch0) * k_fraction
            candidates.append({
                "key": key, "value": value, "ch_mid": ch_mid, "raw": raw,
                "source": axis_source, "confidence": "medium (targeted stroke OCR)",
            })
    # The same schematic is commonly repeated for LHS and RHS carriageways.
    # Vote co-located readings and retain disagreements as explicit QA data.
    groups = []
    for item in sorted(candidates, key=lambda value: value["ch_mid"]):
        group = next(
            (group for group in groups
             if group[0]["key"] == item["key"]
             and abs(group[0]["ch_mid"] - item["ch_mid"]) < 8),
            None,
        )
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    deduplicated = []
    for group in groups:
        votes = defaultdict(list)
        for item in group:
            votes[round(item["value"], 3)].append(item)
        chosen_value, chosen_items = max(votes.items(), key=lambda pair: len(pair[1]))
        chosen = dict(chosen_items[0])
        chosen["ocr_votes"] = len(chosen_items)
        if len(votes) > 1:
            chosen["conflicting_values"] = sorted(votes)
            chosen["confidence"] = "low (conflicting repeated OCR; majority selected)"
        chosen["value"] = chosen_value
        deduplicated.append(chosen)
    return sorted(deduplicated, key=lambda value: value["ch_mid"])


def _vertical_interval_score(words: list[tuple]) -> int:
    text = " ".join(raw for raw, _ch0, _ch1 in words)
    annotations = _composite_band_annotations(words, "profile grid scoring")
    gradient_or_curve = sum(item["key"] in {"G", "K"} for item in annotations)
    lengths = sum(item["key"] == "L" for item in annotations)
    horizontal = len(re.findall(r"\bR\s*=", text, re.I))
    # Vertical schematic rows usually carry G/K labels, sometimes with L
    # packed into the same OCR token. Horizontal rows are dominated by R/L.
    if not annotations:
        return -horizontal * 5
    return gradient_or_curve * 4 + min(lengths, gradient_or_curve + 1) - horizontal * 5


def _vertical_schematic_words(page, checker, axis, rows: list[float]) -> tuple[list[tuple], list[tuple[float, float]]]:
    intervals = []
    for y0, y1 in zip(rows[:-1], rows[1:]):
        words = []
        words.extend(checker.ocr.band_words(page, axis, y0, y1))
        words.extend(checker.ocr.band_line_words(page, axis, y0, y1))
        intervals.append({"range": (y0, y1), "words": words, "score": _vertical_interval_score(words)})

    best_index = None
    best_score = 0
    for index in range(len(intervals) - 1):
        score = intervals[index]["score"] + intervals[index + 1]["score"]
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is not None and best_score > 0:
        chosen = intervals[best_index:best_index + 2]
    else:
        # Conservative fallback for older layouts where OCR did not read labels.
        chosen = intervals[-6:-4]
    words = []
    ranges = []
    for item in chosen:
        words.extend(item["words"])
        ranges.append(item["range"])
    return words, ranges


def _stroke_profile_page(page, checker, page_number: int):
    grid = _profile_grid(page)
    sheet_range = _ocr_sheet_range(page, checker)
    if not grid:
        return None, {
            "page": page_number, "layout": "plan_only",
            "sheet_range": list(sheet_range[:2]) if sheet_range else None,
            "reason": "no repeated profile grid",
        }
    if not sheet_range:
        return None, {
            "page": page_number, "layout": "profile_unresolved", "sheet_range": None,
            "reason": "profile grid found but sheet chainage range was not read",
        }
    rows, x0, x1 = grid
    ch_min, ch_max, raw_range = sheet_range
    axis = _GridAxis(x0, x1, ch_min, ch_max)
    words, vertical_ranges = _vertical_schematic_words(page, checker, axis, rows)
    annotations = _composite_band_annotations(words, "profile grid plus targeted stroke OCR")
    sheet = {
        "page": page_number, "ch_min": ch_min, "ch_max": ch_max,
        "bands": {"vertical_lhs": vertical_ranges[0], "vertical_rhs": vertical_ranges[1]},
        "axis_source": axis.axis_source, "text_adapter_labels": len(annotations),
        "layout": "profile_capable", "sheet_range_raw": raw_range,
        "annotation_conflicts": sum(1 for item in annotations if item.get("conflicting_values")),
    }
    return (annotations, sheet), {
        "page": page_number, "layout": "profile_capable",
        "sheet_range": [ch_min, ch_max], "profile_grid_rows": len(rows),
    }


def _profile_annotations(doc: fitz.Document, checker) -> tuple[list[list[dict]], list[dict], list[dict]]:
    all_annotations = []
    sheet_meta = []
    layout_pages = []
    for page_number in range(len(doc)):
        page = checker.extract.Page(doc, page_number)
        if not page.words:
            stroke_result, layout = _stroke_profile_page(page, checker, page_number)
            layout_pages.append(layout)
            if stroke_result:
                annotations, sheet = stroke_result
                all_annotations.append(annotations)
                sheet_meta.append(sheet)
            continue
        try:
            axis = checker.extract.ChainageAxis(page)
        except ValueError:
            layout_pages.append({
                "page": page_number, "layout": "plan_only",
                "reason": "no valid profile chainage axis",
            })
            continue
        annotations = []
        seen = set()
        for rect, raw in page.words:
            # Combined P&P profile schematics occupy the lower drawing field.
            if not (page.rect.height * 0.68 <= rect.y0 <= page.rect.height * 0.88):
                continue
            match = PROFILE_LABEL_RE.fullmatch(raw.strip())
            if not match:
                continue
            source_key = match.group("key")
            key = "L" if source_key.lower() == "lv" else source_key.upper()
            value = float(match.group("value"))
            x_mid = (rect.x0 + rect.x1) / 2.0
            ch_mid = axis.ch(x_mid)
            # BEHU repeats the vertical schematic for both carriageways.
            identity = (key, round(value, 3), round(ch_mid, 1))
            if identity in seen:
                continue
            seen.add(identity)
            annotations.append({
                "key": key,
                "value": value,
                "ch_mid": ch_mid,
                "raw": raw.strip(),
                "source": "searchable PDF profile schematic",
                "source_key": source_key,
                "confidence": "high",
            })
        annotations.sort(key=lambda item: item["ch_mid"])
        all_annotations.append(annotations)
        sheet_meta.append({
            "page": page_number,
            "ch_min": axis.ch_min,
            "ch_max": axis.ch_max,
            "text_adapter_labels": len(annotations),
        })
        layout_pages.append({
            "page": page_number, "layout": "profile_capable",
            "sheet_range": [axis.ch_min, axis.ch_max],
        })

    # Consecutive pages with the same range are a separate plan/profile pair.
    by_page = {item["page"]: item for item in layout_pages}
    for page_number in range(1, len(doc)):
        current, previous = by_page.get(page_number), by_page.get(page_number - 1)
        if not current or not previous:
            continue
        if (current.get("layout") == "profile_capable"
                and previous.get("layout") == "plan_only"
                and current.get("sheet_range")
                and current.get("sheet_range") == (previous.get("sheet_range") or [None, None])):
            current["layout"] = "profile_only"
            previous["layout"] = "plan_only_paired"
    return all_annotations, sheet_meta, layout_pages


def _nearby_text(page: fitz.Page, header: tuple) -> str:
    x0, y0, x1, _y1 = header[:4]
    clip = fitz.Rect(x0 - 5, y0 - 8, min(page.rect.width, x1 + 210), y0 + 92)
    return " ".join(page.get_textbox(clip).split())


def _structures(doc: fitz.Document) -> list[dict]:
    found = []
    for page_number, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for block in blocks:
            if "PROP.STR.TYPE" not in block[4].upper():
                continue
            text = _nearby_text(page, block)
            # Select the value cell immediately right of this schedule, not a
            # nearby curve-table chainage that happens to enter the clip.
            chainage_candidates = []
            for word in page.get_text("words"):
                wx0, wy0, _wx1, _wy1, value, *_ = word
                match = re.fullmatch(r"(\d{1,3})\+(\d{1,3}(?:\.\d+)?)", value)
                if not match:
                    continue
                if block[2] - 5 <= wx0 <= block[2] + 105 and block[1] + 18 <= wy0 <= block[1] + 75:
                    chainage_candidates.append((abs(wx0 - block[2]), match))
            chainage_match = min(chainage_candidates, default=(None, None), key=lambda item: item[0])[1]
            type_match = re.search(
                r"\b(Box|Slab|Pipe)\s+Culvert(?:\s*\(UNDER\s+COS\))?|\bCulvert\b",
                text,
                re.IGNORECASE,
            )
            if not chainage_match or not type_match:
                continue
            chainage = int(chainage_match.group(1)) * 1000 + float(chainage_match.group(2))
            nearby_values = [
                candidate for candidate in blocks
                if block[2] - 5 <= candidate[0] <= block[2] + 140
                and block[1] + 35 <= candidate[1] <= block[1] + 100
            ]
            span_pattern = re.compile(
                r"\b\d+\s*x\s*\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?(?:\s*\([^)]*\))?",
                re.IGNORECASE,
            )
            span_candidates = [
                (abs(candidate[0] - block[2]), span_pattern.search(" ".join(candidate[4].split())))
                for candidate in nearby_values
                if span_pattern.search(" ".join(candidate[4].split()))
            ]
            proposal_pattern = re.compile(r"\b(New Construction|Reconstruction|Widening)\b", re.IGNORECASE)
            proposal_candidates = [
                (abs(candidate[0] - block[2]), proposal_pattern.search(" ".join(candidate[4].split())))
                for candidate in nearby_values
                if proposal_pattern.search(" ".join(candidate[4].split()))
            ]
            span_match = min(span_candidates, default=(None, None), key=lambda item: item[0])[1]
            proposal_match = min(proposal_candidates, default=(None, None), key=lambda item: item[0])[1]
            y_mid = (block[1] + block[3]) / 2.0
            found.append({
                "chainage": chainage,
                "proposed_type": "Box Culvert" if "box" in type_match.group(0).lower() else type_match.group(0).title(),
                "proposed_span": span_match.group(0) if span_match else None,
                "improvement_proposal": proposal_match.group(1).title() if proposal_match else None,
                "page": page_number,
                "pages": [page_number],
                "plan_present": y_mid < page.rect.height * 0.30,
                "profile_present": page.rect.height * 0.30 <= y_mid < page.rect.height * 0.65,
                "source": "searchable PDF structure schedule",
                "raw": text,
                "extraction_confidence": "high (embedded text schedule)",
            })
    merged = []
    for record in sorted(found, key=lambda item: item["chainage"]):
        existing = next(
            (item for item in merged if abs(item["chainage"] - record["chainage"]) < 1.0),
            None,
        )
        if existing is None:
            merged.append(record)
            continue
        existing["plan_present"] = existing["plan_present"] or record["plan_present"]
        existing["profile_present"] = existing["profile_present"] or record["profile_present"]
        if record["page"] not in existing["pages"]:
            existing["pages"].append(record["page"])
        for key in ("proposed_span", "improvement_proposal"):
            if not existing.get(key) and record.get(key):
                existing[key] = record[key]
    for record in merged:
        if record["plan_present"] and record["profile_present"]:
            record["consistency_status"] = "Present in both"
        elif record["profile_present"]:
            record["consistency_status"] = "Profile only - plan mark not found"
        else:
            record["consistency_status"] = "Plan only - profile callout not found"
    return merged


def enrich(pdf: Path, model: dict, checker) -> tuple[dict, dict]:
    """Supplement the original model using format-aware, auditable extraction."""
    doc = fitz.open(str(pdf))
    applied = []
    stroke_result = {"curves": [], "structures": [], "qa": {}}

    curves = _horizontal_curves(doc, checker)
    if curves and not model.get("curves"):
        model["curves"] = curves
        applied.append(f"horizontal curves: {len(curves)}")
    annotations, sheet_meta, layout_pages = _profile_annotations(doc, checker)
    if any(annotations) and not model.get("vertical_curves"):
        model["profile_pages"] = [sheet["page"] for sheet in sheet_meta]
        model["sheets"] = sheet_meta
        model["vertical_annotations"] = annotations
        model["vertical_curves"] = checker.parse.vertical_curves_from_annotations(annotations)
        for curve in model["vertical_curves"]:
            grade_in, grade_out = curve.get("G_in"), curve.get("G_out")
            if grade_in is not None and grade_out is not None and abs(grade_out - grade_in) > 0.01:
                curve["type"] = "summit" if grade_out < grade_in else "valley"
                curve["type_source"] = "incoming/outgoing profile gradients"
            else:
                curve["type"] = "unknown"
                curve["type_source"] = "stroke OCR did not establish both gradients"
        model["vertical_gradient_segments"] = checker.parse.vertical_gradient_segments_from_annotations(annotations)
        applied.append(f"vertical curves: {len(model['vertical_curves'])}")

    structures = _structures(doc)
    if structures and not model.get("structures"):
        model["structures"] = structures
        applied.append(f"structures: {len(structures)}")

    # Some CAD exports draw every character as vector strokes and therefore
    # expose no searchable PDF text.  Route those pages through geometry-led,
    # targeted OCR only when an engineering category is still absent.  The
    # original searchable-text path remains preferred when it has succeeded.
    stroke_pages = [
        page_number for page_number, page in enumerate(doc)
        if not page.get_text("text").strip()
    ]
    if (checker.ocr.HAVE_TESS and stroke_pages
            and (not model.get("curves") or not model.get("structures"))):
        stroke_result = extract_stroke_engineering(doc, checker, layout_pages)
        if stroke_result["curves"] and not model.get("curves"):
            model["curves"] = stroke_result["curves"]
            applied.append(f"stroke horizontal curves: {len(stroke_result['curves'])}")
        if stroke_result["structures"] and not model.get("structures"):
            model["structures"] = stroke_result["structures"]
            applied.append(f"stroke structures: {len(stroke_result['structures'])}")

    # Apply the engineering convention after every extraction route so NC is
    # consistently evaluated as 2.5%, including targeted stroke-table OCR.
    for curve in model.get("curves", []):
        _normalise_normal_camber(curve)
    normal_camber_count = sum(
        1 for curve in model.get("curves", []) if curve.get("e_raw") == "NC"
    )
    if normal_camber_count:
        applied.append(f"normal camber NC -> 2.5%: {normal_camber_count}")

    if model.get("vertical_curves"):
        checker.parse.enrich_alignment_geometry(
            model.get("curves", []), model.get("sheets", []), model["vertical_curves"]
        )

    doc.close()
    return model, {
        "used": bool(applied),
        "applied": applied,
        "method": "format-aware embedded-text and targeted stroke OCR extraction",
        "profile_pages": sheet_meta,
        "layout_pages": layout_pages,
        "stroke_pages": stroke_pages,
        "stroke_adapter": stroke_result.get("qa", {}),
    }
