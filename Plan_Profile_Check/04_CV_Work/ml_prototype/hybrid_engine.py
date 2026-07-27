"""Hybrid ML + anchored text/OCR evidence extraction for P&P drawings.

The detector supplies a page box. This module searches a class-specific region
around that box for embedded PDF text and, when available, optional Tesseract
OCR. It extracts conservative, rule-ready fields and associates plan/profile
culvert candidates by chainage. It does not issue compliance verdicts.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import os
import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import fitz

from infer_report import CLASSES, COLORS, DEFAULT_MODEL, ROOT, render, run as run_detector


FIELD_PATTERNS = {
    "gradient_percent": re.compile(r"\bG\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*%", re.I),
    "k_value": re.compile(r"\bK\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*(?:m)?\b", re.I),
    "curve_length_m": re.compile(r"\bL(?:V)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*m\b", re.I),
    "curve_table_id": re.compile(r"\b(?:CURVE|HIP)\s*(?:NO\.?)?\s*[:=]?\s*([A-Z]?\d+)\b", re.I),
    "structure_number": re.compile(r"\bSTRUCTURE[ \t]+NO\.?[ \t]*[:=]?[ \t]*([\w./-]+)", re.I),
}
CHAINAGE_RE = re.compile(
    r"(?:\bCH(?:AINAGE)?\.?\s*(?:AT\s*)?[:=]?\s*)?(\d{1,3})\s*\+\s*(\d{1,3}(?:\.\d+)?)\b",
    re.I,
)
DIMENSION_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?){1,2})\s*(?:m|M)?\b"
)
STRUCTURE_CHAINAGE_RE = re.compile(
    r"(?:BOX|PIPE|SLAB|EXISTING|PROPOSED|RCC)[^\n]{0,45}?CULVERT[^\n]{0,45}?"
    r"(?:CH(?:AINAGE)?\.?[ \t]*(?:AT[ \t]*)?[:=]?[ \t]*)?(\d{1,3})[ \t]*\+[ \t]*(\d{1,3}(?:\.\d+)?)",
    re.I,
)


def clip_box(box: list[float], width: int, height: int) -> list[int]:
    return [max(0, int(round(box[0]))), max(0, int(round(box[1]))),
            min(width, int(round(box[2]))), min(height, int(round(box[3])))]


def evidence_region(class_name: str, box: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    padding = {
        "culvert_plan": (330, 340),
        "culvert_profile": (260, 390),
        "gradient_segment": (80, 100),
        "vertical_curve_summit": (110, 130),
        "vertical_curve_valley": (110, 130),
        "curve_table": (70, 70),
    }[class_name]
    px, py = padding
    return clip_box([x0 - px, y0 - py, x1 + px, y1 + py], width, height)


def page_words(page: fitz.Page, dpi: int) -> list[dict]:
    """Return embedded PDF words in rendered-pixel coordinates."""
    scale = dpi / 72.0
    words = []
    for item in page.get_text("words"):
        x0, y0, x1, y1, text = item[:5]
        words.append({
            "text": str(text),
            "box": [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
            "block": int(item[5]), "line": int(item[6]), "word": int(item[7]),
        })
    return words


def words_in_region(words: list[dict], region: list[int]) -> list[dict]:
    x0, y0, x1, y1 = region
    return [word for word in words if
            x0 <= (word["box"][0] + word["box"][2]) / 2 <= x1 and
            y0 <= (word["box"][1] + word["box"][3]) / 2 <= y1]


def format_words(words: list[dict]) -> str:
    lines: dict[tuple[int, int], list[dict]] = {}
    for word in words:
        lines.setdefault((word["block"], word["line"]), []).append(word)
    ordered = sorted(lines.values(), key=lambda line: (
        min(w["box"][1] for w in line), min(w["box"][0] for w in line)
    ))
    return "\n".join(
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["word"]))
        for line in ordered
    )


def tesseract_status() -> tuple[bool, str]:
    if not importlib.util.find_spec("pytesseract"):
        return False, "pytesseract Python package is not installed"
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).is_file():
        return True, f"available at {configured}"
    if not shutil.which("tesseract"):
        return False, "Tesseract executable is not installed or not on PATH"
    return True, "available"


def ocr_crop(image, region: list[int]) -> str:
    import pytesseract

    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).is_file():
        pytesseract.pytesseract.tesseract_cmd = configured

    x0, y0, x1, y1 = region
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(gray, config="--psm 11").strip()


def chainage_value(km: str, metres: str) -> float | None:
    value = int(km) * 1000 + float(metres)
    return round(value, 3) if 0 <= float(metres) < 1000 else None


def extract_fields(text: str) -> dict:
    normalized = text.replace("—", "-").replace("−", "-").replace("Ø", "DIA ")
    chainages = []
    for match in CHAINAGE_RE.finditer(normalized):
        value = chainage_value(match.group(1), match.group(2))
        if value is not None and value not in chainages:
            chainages.append(value)
    dimensions = []
    for match in DIMENSION_RE.finditer(normalized):
        dims = [float(value) for value in re.split(r"\s*[xX×]\s*", match.group(1))]
        if dims not in dimensions:
            dimensions.append(dims)
    explicit_structure_chainages = []
    for match in STRUCTURE_CHAINAGE_RE.finditer(normalized):
        value = chainage_value(match.group(1), match.group(2))
        if value is not None and value not in explicit_structure_chainages:
            explicit_structure_chainages.append(value)
    fields: dict = {"chainages_m": chainages, "dimensions": dimensions}
    if explicit_structure_chainages:
        fields["explicit_structure_chainages_m"] = explicit_structure_chainages
    for key, pattern in FIELD_PATTERNS.items():
        hits = []
        for match in pattern.finditer(normalized):
            value = match.group(1)
            if key == "structure_number" and not re.search(r"\d", value):
                continue
            parsed = value if key in {"structure_number", "curve_table_id"} else float(value)
            if parsed not in hits:
                hits.append(parsed)
        if hits:
            fields[key] = hits
    types = []
    upper = normalized.upper()
    for label in ("BOX CULVERT", "PIPE CULVERT", "RCC BOX", "RCC DECK SLAB"):
        if label in upper:
            types.append(label.lower().replace(" ", "_"))
    if types:
        fields["structure_types"] = types
    return fields


def useful_field_count(fields: dict) -> int:
    return sum(len(value) if isinstance(value, list) else 1
               for key, value in fields.items() if value and key != "dimensions") + len(fields.get("dimensions", []))


def relevant_fields(class_name: str, fields: dict) -> dict:
    if class_name.startswith("culvert"):
        relevant = {}
        for key in ("explicit_structure_chainages_m", "structure_types", "structure_number"):
            if fields.get(key):
                relevant[key] = fields[key]
        has_structure_cue = bool(relevant)
        if has_structure_cue and fields.get("dimensions"):
            relevant["dimensions"] = fields["dimensions"]
        return relevant
    if class_name == "gradient_segment":
        relevant = {}
        if fields.get("gradient_percent"):
            relevant["gradient_percent"] = fields["gradient_percent"]
            if fields.get("curve_length_m"):
                relevant["segment_length_candidates_m"] = fields["curve_length_m"]
        return relevant
    if class_name == "curve_table":
        return {
            key: fields[key]
            for key in ("curve_table_id", "chainages_m", "curve_length_m")
            if fields.get(key)
        }
    return {key: fields[key] for key in ("k_value", "curve_length_m") if fields.get(key)}


def interpretation(class_name: str, relevant: dict) -> tuple[str, str]:
    if useful_field_count(relevant) == 0:
        return "located_text_unresolved", "Feature candidate located; no rule-ready value was read nearby."
    if class_name.startswith("culvert"):
        return "located_with_evidence", "Culvert candidate located with nearby structure evidence for review."
    if class_name == "gradient_segment":
        if relevant.get("gradient_percent"):
            return "located_with_evidence", "Gradient segment located and a nearby gradient value was read."
        return "located_partial_evidence", "Gradient segment located, but its gradient percentage remains unresolved."
    if class_name == "curve_table":
        if relevant.get("curve_table_id") or relevant.get("chainages_m"):
            return "located_with_evidence", "Horizontal curve table located with nearby table text for review."
        return "located_text_unresolved", "Curve table candidate located; table text still requires review."
    if relevant.get("k_value") or relevant.get("curve_length_m"):
        return "located_with_evidence", "Vertical curve located with nearby curve parameter evidence."
    return "located_partial_evidence", "Vertical curve located; nearby text requires engineering review."


def nearest_chainage(fields: dict, box: list[float], region: list[int], words: list[dict]) -> float | None:
    """Prefer the chainage whose text occurrence is closest to the ML anchor."""
    values = fields.get("chainages_m", [])
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    candidates = []
    for word in words_in_region(words, region):
        compact = word["text"].replace(" ", "")
        match = re.search(r"(\d{1,3})\+(\d{1,3}(?:\.\d+)?)", compact)
        if not match:
            continue
        value = chainage_value(match.group(1), match.group(2))
        if value in values:
            wx, wy = (word["box"][0] + word["box"][2]) / 2, (word["box"][1] + word["box"][3]) / 2
            candidates.append(((wx - cx) ** 2 + (wy - cy) ** 2, value))
    return min(candidates)[1] if candidates else values[0]


def relate_culverts(findings: list[dict], tolerance_m: float = 3.0) -> list[dict]:
    plans = [f for f in findings if f["class"] == "culvert_plan" and f.get("primary_chainage_m") is not None]
    profiles = [f for f in findings if f["class"] == "culvert_profile" and f.get("primary_chainage_m") is not None]
    relationships = []
    used_profiles: set[str] = set()
    for plan in plans:
        eligible = [p for p in profiles if p["id"] not in used_profiles]
        if not eligible:
            continue
        profile = min(eligible, key=lambda p: abs(p["primary_chainage_m"] - plan["primary_chainage_m"]))
        delta = abs(profile["primary_chainage_m"] - plan["primary_chainage_m"])
        if delta <= tolerance_m:
            used_profiles.add(profile["id"])
            relationships.append({
                "type": "plan_profile_culvert", "plan_finding": plan["id"],
                "profile_finding": profile["id"], "plan_chainage_m": plan["primary_chainage_m"],
                "profile_chainage_m": profile["primary_chainage_m"], "difference_m": round(delta, 3),
                "status": "candidate_chainage_match",
                "interpretation": "Plan and profile ML candidates have compatible extracted chainages.",
            })
    return relationships


def draw_evidence_crop(image, region: list[int], box: list[float], finding_id: str):
    x0, y0, x1, y1 = region
    crop = image[y0:y1, x0:x1].copy()
    ax0, ay0, ax1, ay1 = [int(round(v)) for v in box]
    cv2.rectangle(crop, (ax0 - x0, ay0 - y0), (ax1 - x0, ay1 - y0), (0, 0, 230), 4)
    cv2.putText(crop, finding_id, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 230), 2, cv2.LINE_AA)
    return crop


def write_findings_csv(path: Path, findings: list[dict]) -> None:
    fields = ["id", "page", "class", "confidence", "status", "primary_chainage_m",
              "gradient_percent", "k_value", "curve_length_m", "dimensions", "extraction_method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            parsed = finding["relevant_fields"]
            writer.writerow({
                "id": finding["id"], "page": finding["page"], "class": finding["class"],
                "confidence": finding["confidence"], "status": finding["status"],
                "primary_chainage_m": finding.get("primary_chainage_m"),
                "gradient_percent": json.dumps(parsed.get("gradient_percent", [])),
                "k_value": json.dumps(parsed.get("k_value", [])),
                "curve_length_m": json.dumps(parsed.get("curve_length_m", [])),
                "dimensions": json.dumps(parsed.get("dimensions", [])),
                "extraction_method": finding["extraction_method"],
            })


def write_hybrid_html(path: Path, report: dict) -> None:
    relationship_rows = "".join(
        f"<tr><td>{r['plan_finding']}</td><td>{r['profile_finding']}</td>"
        f"<td>{r['plan_chainage_m']:.3f}</td><td>{r['profile_chainage_m']:.3f}</td>"
        f"<td>{r['difference_m']:.3f}</td><td>{html.escape(r['status'])}</td></tr>"
        for r in report["relationships"]
    ) or "<tr><td colspan='6'>No plan/profile culvert pair was associated in the processed pages.</td></tr>"
    page_sections = []
    for page in report["pages"]:
        rows = []
        for finding in page["findings"]:
            values = html.escape(json.dumps(finding["relevant_fields"], ensure_ascii=False))
            text = html.escape(finding["evidence_text"][:900]) or "<em>No readable nearby text.</em>"
            rows.append(
                f"<article><div><h3>{finding['id']} — {html.escape(finding['class'])} "
                f"({finding['confidence']:.2f})</h3><p><span class='status'>{html.escape(finding['status'])}</span> "
                f"{html.escape(finding['interpretation'])}</p><p><strong>Rule-ready fields:</strong> "
                f"<code>{values}</code></p><p><strong>Method:</strong> {html.escape(finding['extraction_method'])}</p>"
                f"<details><summary>Nearby extracted text</summary><pre>{text}</pre></details></div>"
                f"<a href='{html.escape(finding['evidence_crop'])}'><img src='{html.escape(finding['evidence_crop'])}'></a></article>"
            )
        page_sections.append(
            f"<section><h2>Page {page['page']}</h2><p>{len(page['findings'])} ML candidates</p>"
            f"<a href='{html.escape(page['ml_overlay'])}'><img class='page' src='{html.escape(page['ml_overlay'])}'></a>"
            f"{''.join(rows) if rows else '<p>No candidates above the operational thresholds.</p>'}</section>"
        )
    counts = "".join(f"<li>{html.escape(name)}: {report['class_counts'].get(name, 0)}</li>" for name in CLASSES)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Hybrid ML + OCR engineering evidence</title><style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#17212b}}main{{max-width:1360px;margin:auto;padding:28px}}
h1,h2,h3{{color:#173f5f}}.notice{{padding:15px 18px;background:#fff1cf;border-left:5px solid #c77b00}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}}.card,section,article{{background:#fff;border-radius:9px;box-shadow:0 1px 5px #0002;padding:17px}}
article{{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;margin:14px 0}}article img,.page{{max-width:100%;border:1px solid #cbd4dc}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:8px;border:1px solid #d8e0e6;text-align:left}}th{{background:#eaf0f4}}
pre{{white-space:pre-wrap;max-height:250px;overflow:auto}}code{{overflow-wrap:anywhere}}.status{{background:#deebf4;padding:3px 7px;border-radius:12px}}
@media(max-width:850px){{article{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Hybrid ML + text/OCR engineering evidence</h1>
<p class="notice"><strong>Review-stage output.</strong> ML detections and extracted values are evidence candidates, not verified facts or compliance verdicts. A rule may run only after required fields and project design context are validated.</p>
<div class="cards"><div class="card"><strong>PDF</strong><br>{html.escape(report['pdf'])}</div><div class="card"><strong>Pages</strong><br>{len(report['pages'])}</div>
<div class="card"><strong>ML candidates</strong><br>{len(report['findings'])}</div><div class="card"><strong>With readable fields</strong><br>{report['findings_with_fields']}</div>
<div class="card"><strong>Text/OCR mode</strong><br>{html.escape(report['ocr_mode'])}</div></div>
<h2>Class totals</h2><ul>{counts}</ul><h2>Plan/profile culvert associations</h2>
<table><thead><tr><th>Plan ID</th><th>Profile ID</th><th>Plan chainage (m)</th><th>Profile chainage (m)</th><th>Difference (m)</th><th>Status</th></tr></thead><tbody>{relationship_rows}</tbody></table>
<h2>What is genuinely hybrid here</h2><ol><li>The ML model supplies the feature class and precise page location.</li><li>The location limits text/OCR extraction to relevant nearby evidence.</li><li>Parsed values remain linked to the visual evidence crop.</li><li>Plan and profile culvert candidates are associated by extracted chainage.</li><li>Unresolved or low-confidence evidence stays in human review instead of becoming a false compliance result.</li></ol>
{''.join(page_sections)}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def write_compact_hybrid_html(path: Path, report: dict) -> None:
    """Write the table-first report format used by the original checker."""
    def badge(label: str, color: str) -> str:
        return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'

    def values(items) -> str:
        if not items:
            return "-"
        return ", ".join(f"{item:g}" if isinstance(item, (int, float)) else str(item) for item in items)

    def chainage(value) -> str:
        if value is None:
            return "-"
        km = int(value // 1000)
        metres = value - km * 1000
        return f"{km}+{metres:06.3f}".rstrip("0").rstrip(".")

    def size_text(dimensions) -> str:
        if not dimensions:
            return "-"
        return "; ".join(" x ".join(f"{part:g}" for part in dims) + " m" for dims in dimensions)

    def status_cell(finding: dict) -> str:
        if finding["status"] == "located_with_evidence":
            return (f'<span class="cell-main">{badge("EXTRACTED", "#1a7f37")}</span>'
                    '<span class="cell-sub">linked to evidence crop</span>')
        if finding["status"] == "located_partial_evidence":
            return (f'<span class="cell-main">{badge("REVIEW", "#b58900")}</span>'
                    '<span class="cell-sub">partial values only</span>')
        return (f'<span class="cell-main">{badge("REVIEW", "#2471a3")}</span>'
                '<span class="cell-sub">value not resolved</span>')

    def confidence_cell(finding: dict) -> str:
        score = finding["confidence"]
        color, label = ("#1a7f37", "HIGH") if score >= 0.65 else ("#b58900", "CHECK")
        passes = "+".join(finding["inference_sources"])
        return (f'<span class="cell-main">{badge(label, color)}</span>'
                f'<span class="cell-sub">{score:.2f} &middot; {html.escape(passes)}</span>')

    findings = report["findings"]
    verticals = [f for f in findings if f["class"].startswith("vertical_curve_")]
    gradients = [f for f in findings if f["class"] == "gradient_segment"]
    structures = [f for f in findings if f["class"].startswith("culvert_")]
    unresolved = len(findings) - report["findings_with_fields"]

    vertical_rows = "".join(
        f"<tr><td>{f['id']}</td><td>{f['page']}</td>"
        f"<td>{html.escape(f['class'].removeprefix('vertical_curve_'))}</td>"
        f"<td>{values(f['relevant_fields'].get('k_value'))}</td>"
        f"<td>{values(f['relevant_fields'].get('curve_length_m'))}</td>"
        f"<td>{confidence_cell(f)}</td><td>{status_cell(f)}</td>"
        f"<td><a href='{html.escape(f['evidence_crop'])}'>view crop</a></td></tr>"
        for f in verticals
    ) or "<tr><td colspan='8'>No vertical-curve candidates detected.</td></tr>"

    gradient_rows = "".join(
        f"<tr><td>{f['id']}</td><td>{f['page']}</td>"
        f"<td>{values(f['relevant_fields'].get('gradient_percent'))}</td>"
        f"<td>{values(f['relevant_fields'].get('segment_length_candidates_m'))}</td>"
        f"<td>{confidence_cell(f)}</td><td>{status_cell(f)}</td>"
        f"<td><a href='{html.escape(f['evidence_crop'])}'>view crop</a></td></tr>"
        for f in gradients
    ) or "<tr><td colspan='7'>No gradient-segment candidates detected.</td></tr>"

    structure_rows = "".join(
        f"<tr><td>{f['id']}</td><td>{f['page']}</td>"
        f"<td>{'Plan' if f['class'] == 'culvert_plan' else 'Profile'}</td>"
        f"<td>{chainage(f.get('primary_chainage_m'))}</td>"
        f"<td>{html.escape(', '.join(f['relevant_fields'].get('structure_types', [])) or '-')}</td>"
        f"<td>{html.escape(size_text(f['relevant_fields'].get('dimensions')))}</td>"
        f"<td>{confidence_cell(f)}</td><td>{status_cell(f)}</td>"
        f"<td><a href='{html.escape(f['evidence_crop'])}'>view crop</a></td></tr>"
        for f in structures
    ) or "<tr><td colspan='9'>No culvert candidates detected.</td></tr>"

    review_rows = "".join(
        f"<tr><td>{f['id']} (page {f['page']})</td>"
        f"<td><span class='cell-main'>{html.escape(f['class'])}</span>"
        f"<span class='cell-sub'>box {', '.join(str(int(v)) for v in f['box_page'])}</span></td>"
        f"<td>{confidence_cell(f)}</td><td>{status_cell(f)}</td>"
        f"<td>{html.escape(f['interpretation'])}</td></tr>"
        for f in findings
    ) or "<tr><td colspan='5'>No findings.</td></tr>"

    relationship_rows = "".join(
        f"<tr><td>{r['plan_finding']}</td><td>{r['profile_finding']}</td>"
        f"<td>{chainage(r['plan_chainage_m'])}</td><td>{chainage(r['profile_chainage_m'])}</td>"
        f"<td>{r['difference_m']:.3f}</td><td>{badge('MATCH', '#1a7f37')}</td></tr>"
        for r in report["relationships"]
    ) or ("<tr><td colspan='6'><span class='cell-main'>No confirmed pair in processed pages</span>"
          "<span class='cell-sub'>An unresolved ML candidate is not evidence that a feature is absent.</span></td></tr>")

    page_rows = "".join(
        f"<tr><td>{page['page']}</td><td>{len(page['findings'])}</td>"
        f"<td><a href='{html.escape(page['ml_overlay'])}'>open marked drawing</a></td></tr>"
        for page in report["pages"]
    )
    model_name = Path(report["model"]).name
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Plan &amp; Profile Hybrid Review Report</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;max-width:1400px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
th,td{{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} a{{color:#145a86}}
.badge{{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px}}
.cell-main{{display:block;margin-bottom:2px}}.cell-sub{{display:block;color:#666;font-size:11px}}
.compact td{{min-width:105px}}.compact td:first-child{{min-width:150px;font-weight:600}}
.sum{{font-size:15px;margin:12px 0}}.meta{{color:#666;font-size:12px}}
.notice{{font-size:12px;background:#fff8e5;border:1px solid #e5cf8d;padding:8px 10px;margin:14px 0}}
</style></head><body>
<h1>Plan &amp; Profile Hybrid Review Report</h1>
<div class="meta">File: {html.escape(report['pdf'])} &middot; Model: {html.escape(model_name)} &middot;
Pages: {', '.join(str(p['page']) for p in report['pages'])} &middot; ML mode: {html.escape(report['mode'])} &middot;
Text/OCR: {html.escape(report['ocr_mode'])} &middot; Generated {date.today().isoformat()}</div>
<div class="sum">{len(findings)} ML CANDIDATES &middot; {report['findings_with_fields']} EXTRACTED &middot;
{unresolved} MANUAL REVIEW &middot; 0 COMPLIANCE VERDICTS</div>
<div class="notice"><strong>Report status:</strong> This uses the compact format of the earlier compliance report, but the current hybrid stage reports extracted evidence only. PASS/HARD FAIL is withheld until the applicable standard, road class, terrain, design speed and required values are verified.</div>

<h2>Extracted vertical curves</h2>
<table><tr><th>#</th><th>Page</th><th>Type</th><th>K</th><th>L / Lv (m)</th><th>ML confidence</th><th>Extraction status</th><th>Evidence</th></tr>{vertical_rows}</table>
<h2>Extracted gradient segments</h2>
<table><tr><th>#</th><th>Page</th><th>Gradient (%)</th><th>Length candidate (m)</th><th>ML confidence</th><th>Extraction status</th><th>Evidence</th></tr>{gradient_rows}</table>
<h2>Structures (culverts)</h2>
<table><tr><th>#</th><th>Page</th><th>Drawing view</th><th>Chainage</th><th>Type</th><th>Size candidate</th><th>ML confidence</th><th>Extraction status</th><th>Evidence</th></tr>{structure_rows}</table>
<h2>Plan &harr; profile structure cross-check</h2>
<table><tr><th>Plan finding</th><th>Profile finding</th><th>Plan chainage</th><th>Profile chainage</th><th>Difference (m)</th><th>Status</th></tr>{relationship_rows}</table>
<h2>Hybrid extraction review</h2>
<table class="compact"><tr><th>Element</th><th>ML location</th><th>Detection confidence</th><th>Text/OCR extraction</th><th>Notes</th></tr>{review_rows}</table>
<h2>Marked drawing pages</h2>
<table><tr><th>Page</th><th>ML candidates</th><th>Drawing overlay</th></tr>{page_rows}</table>
<p class="meta">Hybrid ML detection with feature-anchored PDF text/OCR extraction. “EXTRACTED” means a value was associated with a visual evidence crop; it does not mean the value passed an IRC compliance check. Detailed text, coordinates and raw fields remain in hybrid_findings.json.</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def build_hybrid(args: argparse.Namespace) -> dict:
    detector_report = run_detector(args)
    output = args.output.resolve()
    pdf = args.pdf.resolve()
    tesseract_available, tesseract_message = tesseract_status()
    if args.ocr == "required" and not tesseract_available:
        raise RuntimeError(tesseract_message)
    use_ocr = args.ocr in {"auto", "required"} and tesseract_available
    doc = fitz.open(pdf)
    (output / "evidence_crops").mkdir(exist_ok=True)
    findings = []
    page_reports = []
    finding_number = 0
    for detected_page in detector_report["pages"]:
        page_number = detected_page["page"]
        page = doc[page_number - 1]
        image = render(page, args.dpi)
        words = page_words(page, args.dpi)
        page_findings = []
        for detection in detected_page["detections"]:
            finding_number += 1
            finding_id = f"F{finding_number:04d}"
            region = evidence_region(detection["class"], detection["box_page"],
                                     image.shape[1], image.shape[0])
            selected_words = words_in_region(words, region)
            vector_text = format_words(selected_words)
            ocr_text = ""
            if use_ocr and (len(vector_text) < args.ocr_min_chars or args.ocr == "required"):
                ocr_text = ocr_crop(image, region)
            combined = vector_text
            if ocr_text and ocr_text not in combined:
                combined = (combined + "\n--- OCR fallback ---\n" + ocr_text).strip()
            fields = extract_fields(combined)
            relevant = relevant_fields(detection["class"], fields)
            status, explanation = interpretation(detection["class"], relevant)
            crop_name = f"evidence_crops/{finding_id}_p{page_number:04d}.jpg"
            crop = draw_evidence_crop(image, region, detection["box_page"], finding_id)
            if not cv2.imwrite(str(output / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"failed to write {output / crop_name}")
            finding = {
                "id": finding_id, "page": page_number, "class": detection["class"],
                "confidence": detection["confidence"], "box_page": detection["box_page"],
                "box_model_anchor": detection.get("box_model_anchor"),
                "inference_sources": detection["inference_sources"], "status": status,
                "interpretation": explanation, "evidence_region": region,
                "evidence_crop": crop_name, "extraction_method": (
                    "embedded_pdf_text+optional_tesseract" if ocr_text else "embedded_pdf_text"
                ), "evidence_text": combined, "extracted_fields": fields,
                "relevant_fields": relevant,
            }
            if detection["class"].startswith("culvert"):
                explicit = relevant.get("explicit_structure_chainages_m", [])
                finding["primary_chainage_m"] = explicit[0] if explicit else None
            else:
                finding["primary_chainage_m"] = None
            findings.append(finding)
            page_findings.append(finding)
        page_reports.append({
            "page": page_number, "ml_overlay": detected_page["overlay"],
            "page_width": detected_page["page_width"], "page_height": detected_page["page_height"],
            "findings": page_findings,
        })
    doc.close()
    report = {
        "schema_version": 1, "purpose": "review-stage hybrid engineering evidence",
        "pdf": str(pdf), "model": detector_report["model"], "mode": detector_report["mode"],
        "ocr_mode": args.ocr, "tesseract_available": tesseract_available,
        "tesseract_status": tesseract_message, "thresholds": detector_report["thresholds"],
        "pages": page_reports, "findings": findings,
        "findings_with_fields": sum(useful_field_count(f["relevant_fields"]) > 0 for f in findings),
        "class_counts": dict(Counter(f["class"] for f in findings)),
        "relationships": relate_culverts(findings),
        "compliance_verdicts_emitted": 0,
        "limitations": [
            "ML candidates are not verified ground truth.",
            "Nearby text association can include unrelated labels on dense sheets.",
            "Compliance requires verified values, road context, and an explicitly selected standard.",
            "Scanned PDFs require the optional Tesseract fallback or another OCR service.",
        ],
    }
    (output / "hybrid_findings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_findings_csv(output / "hybrid_findings.csv", findings)
    write_compact_hybrid_html(output / "hybrid_report.html", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("pdf", type=Path)
    result.add_argument("--pages", default="1-3", help="1-based list/range or all")
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--output", type=Path, default=ROOT / "outputs" / "hybrid_prototype")
    result.add_argument("--mode", choices=("color", "grayscale", "dual"), default="dual")
    result.add_argument("--ocr", choices=("auto", "off", "required"), default="auto")
    result.add_argument("--ocr-min-chars", type=int, default=12)
    result.add_argument("--dpi", type=int, default=200)
    result.add_argument("--imgsz", type=int, default=1280)
    result.add_argument("--tile-size", type=int, default=1280)
    result.add_argument("--tile-overlap", type=int, default=192)
    result.add_argument("--nms-iou", type=float, default=0.45)
    result.add_argument("--device", default="cpu")
    result.add_argument("--allow-sealed-test", action="store_true", help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    report = build_hybrid(args)
    print(json.dumps({
        "output": str(args.output.resolve()), "pages": len(report["pages"]),
        "findings": len(report["findings"]), "with_fields": report["findings_with_fields"],
        "relationships": len(report["relationships"]),
        "tesseract": report["tesseract_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
