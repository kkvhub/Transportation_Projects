"""OCR/rules-first IRC checker with ML corroboration.

The original checker remains authoritative for engineering extraction and IRC
verdicts. ML adds visual candidates and evidence; it never replaces the model
or silently downgrades the command to detector-only output.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from hybrid_engine import DEFAULT_MODEL, ROOT, build_hybrid
from text_layer_adapter import enrich as enrich_from_text_layer

WORKSPACE = ROOT.parent
TOOL_ROOT = WORKSPACE / "02_Tool"
REGRESSION_CONFIG = ROOT / "engineering_regression.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_checker():
    # Import 02_Tool read-only: never create __pycache__ inside the preserved tool.
    sys.dont_write_bytecode = True
    if str(TOOL_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOL_ROOT))
    import pp_checker
    # Support a D:-drive Tesseract installation without changing 02_Tool or
    # relying on PATH propagation through nested PowerShell/Python processes.
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).is_file():
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = configured
        pp_checker.ocr.pytesseract = pytesseract
        pp_checker.ocr.HAVE_TESS = True
        pp_checker.ocr.TESS_STATUS = "ok"
    return pp_checker


def fixtures() -> list[dict]:
    return json.loads(REGRESSION_CONFIG.read_text(encoding="utf-8"))["fixtures"]


def matching_fixture(pdf: Path) -> dict | None:
    digest = sha256(pdf)
    return next((item for item in fixtures() if item["pdf_sha256"].lower() == digest.lower()), None)


def validate_engineering(engineering: dict, expected: dict | None = None) -> None:
    model, rules = engineering["model"], engineering["rules"]
    required = {"curves", "vertical_curves", "structures"}
    if required - set(model):
        raise ValueError(f"engineering model missing: {sorted(required - set(model))}")
    if not rules.get("results") or not rules.get("summary"):
        raise ValueError("IRC rule engine produced no results; refusing detector-only report")
    if expected:
        actual = {
            "horizontal_curves": len(model["curves"]),
            "vertical_curves": len(model["vertical_curves"]),
            "structures": len(model["structures"]),
            "rule_results": len(rules["results"]),
            "summary": rules["summary"],
        }
        for key in ("horizontal_curves", "vertical_curves", "structures", "rule_results", "summary"):
            if actual[key] != expected[key]:
                raise ValueError(f"engineering regression failed for {key}: {actual[key]} != {expected[key]}")


def load_engineering(pdf: Path, source: str, road_class: str, terrain: str,
                     allow_no_tesseract: bool) -> tuple[dict, dict]:
    checker = original_checker()
    fixture = matching_fixture(pdf)
    have_tesseract = bool(checker.ocr.HAVE_TESS)
    if source == "fixture" or (source == "auto" and fixture and not have_tesseract):
        if not fixture:
            raise ValueError("no verified fixture matches this exact PDF SHA-256")
        fixture_path = WORKSPACE / fixture["engineering_json"]
        if sha256(fixture_path).lower() != fixture["engineering_json_sha256"].lower():
            raise ValueError("verified engineering fixture SHA-256 changed")
        engineering = json.loads(fixture_path.read_text(encoding="utf-8"))
        if (engineering["rules"].get("road_class") != road_class or
                engineering["rules"].get("terrain") != terrain):
            raise ValueError(
                "verified fixture context does not match requested road class/terrain; "
                "live engineering extraction is required"
            )
        validate_engineering(engineering, fixture["expected"])
        return engineering, {
            "mode": "verified_fixture", "fixture_id": fixture["id"],
            "fixture_path": str(fixture_path), "pdf_sha256": fixture["pdf_sha256"],
            "tesseract_available": have_tesseract,
        }
    if not have_tesseract and not allow_no_tesseract:
        raise RuntimeError(
            "Live engineering extraction requires Tesseract for stroke-font drawings. "
            "Install/configure Tesseract on D:, or use an exact verified regression fixture."
        )
    model, rule_result, xcheck = checker.run(
        str(pdf), road_class=road_class, terrain=terrain
    )
    model, adapter = enrich_from_text_layer(pdf, model, checker)
    if adapter["used"]:
        # The compatibility records use the original model schema and are
        # deliberately evaluated by the unchanged original IRC rule engine.
        rule_result = checker.rules.check(model, road_class=road_class, terrain=terrain)
        xcheck = checker.validate.cross_checks(model)
    engineering = {"model": model, "rules": rule_result, "xcheck": xcheck}
    validate_engineering(engineering, fixture["expected"] if fixture else None)
    return engineering, {
        "mode": "live_original_checker", "fixture_id": fixture["id"] if fixture else None,
        "pdf_sha256": sha256(pdf), "tesseract_available": have_tesseract,
        "text_layer_adapter": adapter,
    }


def ml_args(args: argparse.Namespace, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pdf=args.pdf, pages=args.ml_pages, model=args.model, output=output,
        mode=args.ml_mode, ocr=args.ml_ocr, ocr_min_chars=12, dpi=args.ml_dpi,
        imgsz=1280, tile_size=1280, tile_overlap=192, nms_iou=0.45,
        device=args.device, allow_sealed_test=False,
    )


def reconciliation(engineering: dict, ml_report: dict) -> dict:
    model = engineering["model"]
    page_counts: dict[int, Counter] = defaultdict(Counter)
    page_conf: dict[tuple[int, str], list[float]] = defaultdict(list)
    for finding in ml_report["findings"]:
        page_counts[finding["page"]][finding["class"]] += 1
        page_conf[(finding["page"], finding["class"])].append(finding["confidence"])
    pages = []
    for page in ml_report["pages"]:
        number = page["page"]
        counts = dict(page_counts[number])
        pages.append({
            "page": number, "counts": counts,
            "mean_confidence": {
                name: round(sum(scores) / len(scores), 4)
                for (p, name), scores in page_conf.items() if p == number
            },
            "overlay": f"ml_diagnostic/{page['ml_overlay']}",
        })
    vertical_ml = sum(
        count for page in page_counts.values() for name, count in page.items()
        if name.startswith("vertical_curve_")
    )
    curve_table_ml = sum(page.get("curve_table", 0) for page in page_counts.values())
    plan_ml = sum(page.get("culvert_plan", 0) for page in page_counts.values())
    profile_ml = sum(page.get("culvert_profile", 0) for page in page_counts.values())
    comparisons = [
        {
            "element": "Horizontal curves", "engineering_records": len(model["curves"]),
            "ml_candidates": {"curve_table": curve_table_ml},
            "status": "VISUAL SUPPORT" if curve_table_ml else ("OCR/RULES PRIMARY" if model["curves"] else "REVIEW"),
            "note": (
                "Curve-table ML candidates visually support the original table extraction; IRC checks still use verified rule-engine values."
                if model["curves"] else
                "No horizontal curve engineering records were extracted; horizontal compliance coverage is incomplete."
            ),
        },
        {
            "element": "Vertical curves", "engineering_records": len(model["vertical_curves"]),
            "ml_candidates": vertical_ml,
            "status": "VISUAL SUPPORT" if vertical_ml else "REVIEW",
            "note": "ML candidates corroborate locations/types but do not replace K, L, gradient, chainage, or IRC checks.",
        },
        {
            "element": "Structures", "engineering_records": len(model["structures"]),
            "ml_candidates": {"plan": plan_ml, "profile": profile_ml},
            "status": "REVIEW",
            "note": "Candidate counts are independent visual observations, not final structure counts; reconciliation remains reviewable.",
        },
    ]
    return {"pages": pages, "comparisons": comparisons}


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'


def ml_section(ml_report: dict, reconciled: dict, provenance: dict) -> str:
    comparison_rows = []
    colors = {"OCR/RULES PRIMARY": "#2471a3", "VISUAL SUPPORT": "#1a7f37", "REVIEW": "#b58900"}
    for item in reconciled["comparisons"]:
        ml_value = "not applicable" if item["ml_candidates"] is None else json.dumps(item["ml_candidates"])
        comparison_rows.append(
            f"<tr><td>{html.escape(item['element'])}</td><td>{item['engineering_records']}</td>"
            f"<td>{html.escape(ml_value)}</td><td>{badge(item['status'], colors[item['status']])}</td>"
            f"<td>{html.escape(item['note'])}</td></tr>"
        )
    page_rows = []
    for page in reconciled["pages"]:
        counts = page["counts"]
        page_rows.append(
            f"<tr><td>{page['page']}</td><td>{counts.get('vertical_curve_summit', 0)}</td>"
            f"<td>{counts.get('vertical_curve_valley', 0)}</td><td>{counts.get('gradient_segment', 0)}</td>"
            f"<td>{counts.get('culvert_plan', 0)}</td><td>{counts.get('culvert_profile', 0)}</td>"
            f"<td>{counts.get('curve_table', 0)}</td>"
            f"<td><a href='{html.escape(page['overlay'])}'>view marked drawing</a></td></tr>"
        )
    layout_section = ""
    adapter = provenance.get("text_layer_adapter") or {}
    layout_pages = adapter.get("layout_pages") or []
    if layout_pages:
        profile_meta = {
            item["page"]: item for item in adapter.get("profile_pages", [])
        }
        layout_rows = []
        labels = {
            "profile_capable": "Combined plan/profile",
            "plan_only_paired": "Plan-only (paired)",
            "profile_only": "Profile-only (paired)",
            "plan_only": "Plan-only",
            "profile_unresolved": "Profile grid - range unresolved",
        }
        for item in layout_pages:
            chainage_range = item.get("sheet_range")
            range_text = (
                f"{chainage_range[0] / 1000:g}+000 to {chainage_range[1] / 1000:g}+000"
                if chainage_range else "not resolved"
            )
            meta = profile_meta.get(item["page"], {})
            layout_rows.append(
                f"<tr><td>{item['page'] + 1}</td>"
                f"<td>{html.escape(labels.get(item.get('layout'), item.get('layout', 'unknown')))}</td>"
                f"<td>{html.escape(range_text)}</td>"
                f"<td>{meta.get('text_adapter_labels', 0)}</td>"
                f"<td>{meta.get('annotation_conflicts', 0)}</td></tr>"
            )
        layout_section = f"""
<h2>Drawing layout and chainage continuity</h2>
<p class="meta">Page type is determined from profile-grid geometry. Separate plan/profile sheets are paired by their printed chainage range, not merely by page adjacency. OCR conflicts are retained in the result JSON for review.</p>
<table><tr><th>PDF page</th><th>Detected layout</th><th>Printed range</th><th>Profile labels</th><th>OCR conflicts</th></tr>{''.join(layout_rows)}</table>
"""
    extraction_qa = ""
    stroke_qa = adapter.get("stroke_adapter") or {}
    if stroke_qa:
        confidence = stroke_qa.get("curve_confidence") or {}
        extraction_qa = f"""
<h2>Format-aware extraction QA</h2>
<p class="meta">Stroke-font CAD pages were routed through geometry-led targeted OCR. Accepted records are kept separate from rejected geometric candidates; OCR disagreements remain in the result JSON.</p>
<table><tr><th>Item</th><th>Accepted</th><th>Candidates rejected</th><th>Confidence / conflicts</th></tr>
<tr><td>Horizontal curve tables</td><td>{stroke_qa.get('curve_accepted', 0)}</td><td>{len(stroke_qa.get('curve_rejected', []))}</td><td>high {confidence.get('high', 0)}, medium {confidence.get('medium', 0)}, low {confidence.get('low', 0)}; field conflicts {stroke_qa.get('curve_field_conflicts', 0)}</td></tr>
<tr><td>Profile structure schedules</td><td>{stroke_qa.get('structure_accepted', 0)}</td><td>{len(stroke_qa.get('structure_rejected', []))}</td><td>Plan presence remains REVIEW until spatially reconciled</td></tr></table>
"""
    missing_coverage = []
    for item in reconciled["comparisons"]:
        if item["engineering_records"] == 0:
            if item["element"] == "Structures" and not any(
                value for value in (item.get("ml_candidates") or {}).values()
            ):
                continue
            missing_coverage.append(item["element"])
    coverage_warning = ""
    if missing_coverage:
        coverage_warning = (
            '<p style="border:2px solid #b58900;padding:10px;background:#fff8dc">'
            '<strong>PARTIAL ENGINEERING COVERAGE:</strong> No rule-ready records were extracted for '
            + html.escape(", ".join(missing_coverage))
            + ". PASS totals apply only to the engineering records listed in this report.</p>"
        )
    source = "verified SM_1 engineering fixture" if provenance["mode"] == "verified_fixture" else "live original checker"
    return f"""
{layout_section}
{extraction_qa}
{coverage_warning}
<h2>Hybrid ML corroboration (engineering results preserved)</h2>
<p class="meta">Engineering source: {html.escape(source)}. ML is an independent visual support layer; it does not change the IRC verdicts above.</p>
<table><tr><th>Engineering scope</th><th>Engineering records</th><th>ML candidates</th><th>Hybrid status</th><th>Interpretation</th></tr>{''.join(comparison_rows)}</table>
<h2>ML visual candidates by drawing page</h2>
<table><tr><th>Page</th><th>Summit</th><th>Valley</th><th>Gradient</th><th>Plan culvert</th><th>Profile culvert</th><th>Curve table</th><th>Overlay</th></tr>{''.join(page_rows)}</table>
<p class="meta"><a href="ml_diagnostic/hybrid_report.html">Open detailed ML evidence appendix</a> (diagnostic only; not a separate compliance result).</p>
"""


def inject_section(base_html: str, section: str) -> str:
    marker = '<p class="meta">Deterministic extraction with OCR of stroke text;'
    if marker not in base_html:
        raise ValueError("original report template marker not found")
    return base_html.replace(marker, section + marker, 1)


def validate_final_html(document: str, engineering: dict, expected: dict | None) -> None:
    required = (expected or {}).get("required_report_sections", [
        "Extracted horizontal curves", "Horizontal curve compliance",
        "Vertical curve / profile compliance", "Structures",
    ])
    missing = [heading for heading in required if heading not in document]
    if missing:
        raise ValueError(f"final checker report missing required sections: {missing}")
    summary = engineering["rules"]["summary"]
    for verdict in ("PASS", "ADVISORY", "HARD FAIL", "INFO"):
        if f"{summary[verdict]} {verdict}" not in document:
            raise ValueError(f"final report lost {verdict} summary")
    if "Hybrid ML corroboration" not in document:
        raise ValueError("final report missing ML corroboration")


def run(args: argparse.Namespace) -> dict:
    pdf, output = args.pdf.resolve(), args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError(f"output must remain under {ROOT}")
    if "shimla" in pdf.name.lower():
        raise ValueError("Shimla remains sealed and is refused by the hybrid checker")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    engineering, provenance = load_engineering(
        pdf, args.engineering_source, args.road_class, args.terrain, args.allow_no_tesseract
    )
    output.mkdir(parents=True, exist_ok=True)
    ml_output = output / "ml_diagnostic"
    ml_report = build_hybrid(ml_args(args, ml_output))
    reconciled = reconciliation(engineering, ml_report)
    checker = original_checker()
    base_html = checker.report.to_html(
        engineering["model"], engineering["rules"], engineering["xcheck"]
    )
    final_html = inject_section(base_html, ml_section(ml_report, reconciled, provenance))
    fixture = matching_fixture(pdf)
    expected = fixture["expected"] if fixture else None
    validate_final_html(final_html, engineering, expected)
    (output / "hybrid_compliance_report.html").write_text(final_html, encoding="utf-8")
    (output / "hybrid_report.html").write_text(final_html, encoding="utf-8")
    combined = {
        "schema_version": 1, "purpose": "IRC compliance checker with ML corroboration",
        "engineering": engineering, "engineering_provenance": provenance,
        "ml": ml_report, "reconciliation": reconciled,
        "engineering_verdicts_preserved": True,
    }
    (output / "hybrid_compliance_result.json").write_text(
        json.dumps(combined, indent=2, default=str), encoding="utf-8"
    )
    return combined


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("pdf", type=Path)
    result.add_argument("--output", type=Path, default=ROOT / "outputs" / "engineering_hybrid")
    result.add_argument("--road-class", choices=("2_lane", "4_lane"), default="2_lane")
    result.add_argument("--terrain", choices=("mountainous", "plain_rolling", "steep"), default="mountainous")
    result.add_argument("--engineering-source", choices=("auto", "live", "fixture"), default="auto")
    result.add_argument("--allow-no-tesseract", action="store_true")
    result.add_argument("--ml-pages", default="all")
    result.add_argument("--ml-mode", choices=("color", "grayscale", "dual"), default="dual")
    result.add_argument("--ml-ocr", choices=("auto", "off", "required"), default="auto")
    result.add_argument("--ml-dpi", type=int, default=200)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--device", default="cpu")
    return result


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "engineering_source": result["engineering_provenance"]["mode"],
        "summary": result["engineering"]["rules"]["summary"],
        "horizontal_curves": len(result["engineering"]["model"]["curves"]),
        "vertical_curves": len(result["engineering"]["model"]["vertical_curves"]),
        "structures": len(result["engineering"]["model"]["structures"]),
        "ml_findings": len(result["ml"]["findings"]),
    }, indent=2))


if __name__ == "__main__":
    main()
