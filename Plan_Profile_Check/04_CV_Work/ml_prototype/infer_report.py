"""Page-level inference and review report for the isolated P&P detector.

This module intentionally has no imports from, and makes no writes to, 02_Tool.
It turns tiled YOLO detections back into drawing-page coordinates, removes
overlap duplicates, and produces JSON, CSV, page overlays, and an HTML report.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "config"))
os.environ.setdefault("TEMP", str(ROOT / "tmp"))
os.environ.setdefault("TMP", str(ROOT / "tmp"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))

import cv2
import fitz
import numpy as np
from ultralytics import YOLO


CLASSES = [
    "vertical_curve_summit",
    "vertical_curve_valley",
    "gradient_segment",
    "culvert_plan",
    "culvert_profile",
    "curve_table",
]
COLORS = {
    "vertical_curve_summit": (42, 42, 220),
    "vertical_curve_valley": (220, 90, 30),
    "gradient_segment": (20, 170, 230),
    "culvert_plan": (190, 45, 190),
    "culvert_profile": (35, 165, 55),
    "curve_table": (0, 120, 255),
}
DEFAULT_THRESHOLDS = {
    "vertical_curve_summit": 0.35,
    "vertical_curve_valley": 0.35,
    "gradient_segment": 0.35,
    "culvert_plan": 0.15,
    "culvert_profile": 0.35,
    "curve_table": 0.25,
}
DEFAULT_MODEL = (
    ROOT / "plan_profile_curve_table_results" / "runs" /
    "retrain_v2_yolo11n_1280" / "weights" / "best.pt"
)
PROFILE_CURVE_CLASSES = {"vertical_curve_summit", "vertical_curve_valley"}


def page_numbers(value: str, count: int) -> list[int]:
    if value.lower() == "all":
        return list(range(count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            selected.update(range(a - 1, b))
        else:
            selected.add(int(part) - 1)
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"pages must be inside 1..{count}")
    return sorted(selected)


def windows(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def render(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def suppress_duplicates(items: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(items, key=lambda d: d["confidence"], reverse=True):
        duplicate = next(
            (existing for existing in kept
             if existing["class"] == candidate["class"]
             and iou(existing["box_page"], candidate["box_page"]) >= threshold),
            None,
        )
        if duplicate is None:
            candidate["inference_sources"] = [candidate.pop("inference_source")]
            kept.append(candidate)
        else:
            source = candidate["inference_source"]
            if source not in duplicate["inference_sources"]:
                duplicate["inference_sources"].append(source)
            duplicate["duplicate_candidates_merged"] += 1
    return kept


def expand_vertical_curve_boxes(items: list[dict], page_width: int, page_height: int) -> list[dict]:
    """Turn a small YOLO anchor into a review-sized vertical curve evidence box."""
    expanded: list[dict] = []
    for item in items:
        if item["class"] not in PROFILE_CURVE_CLASSES:
            expanded.append(item)
            continue
        x0, y0, x1, y1 = item["box_page"]
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        target_width = min(max(width * 2.2, page_width * 0.12), page_width * 0.24)
        target_height = min(max(height * 2.6, page_height * 0.045), page_height * 0.12)
        item["box_model_anchor"] = [round(v, 2) for v in item["box_page"]]
        item["box_page"] = [
            round(max(0.0, cx - target_width / 2), 2),
            round(max(0.0, cy - target_height / 2), 2),
            round(min(float(page_width), cx + target_width / 2), 2),
            round(min(float(page_height), cy + target_height / 2), 2),
        ]
        item["box_expanded"] = True
        item["box_expansion_reason"] = "vertical curve anchor expanded for engineering evidence crop"
        expanded.append(item)
    return expanded


def merge_vertical_curve_fragments(items: list[dict]) -> list[dict]:
    """Merge overlapping summit/valley fragments after expansion, preserving review notes."""
    kept: list[dict] = []
    for candidate in sorted(items, key=lambda d: d["confidence"], reverse=True):
        if candidate["class"] not in PROFILE_CURVE_CLASSES:
            kept.append(candidate)
            continue
        duplicate = next(
            (existing for existing in kept
             if existing["class"] in PROFILE_CURVE_CLASSES
             and iou(existing["box_page"], candidate["box_page"]) >= 0.28),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        duplicate["duplicate_candidates_merged"] += 1
        for source in candidate.get("inference_sources", []):
            if source not in duplicate["inference_sources"]:
                duplicate["inference_sources"].append(source)
        alternatives = duplicate.setdefault("alternative_class_candidates", [])
        alternatives.append({
            "class": candidate["class"],
            "confidence": candidate["confidence"],
            "box_model_anchor": candidate.get("box_model_anchor", candidate["box_page"]),
        })
        if candidate["class"] != duplicate["class"]:
            duplicate["class_review_note"] = (
                "Overlapping summit/valley anchors were merged; curve type needs review."
            )
    return kept


def predict_tiles(model: YOLO, page_image: np.ndarray, mode: str, tile_size: int,
                  overlap: int, imgsz: int, device: str, min_conf: float) -> list[dict]:
    height, width = page_image.shape[:2]
    passes = ["color", "grayscale"] if mode == "dual" else [mode]
    detections: list[dict] = []
    for pass_name in passes:
        source = page_image if pass_name == "color" else cv2.cvtColor(
            cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
        for y in windows(height, tile_size, overlap):
            for x in windows(width, tile_size, overlap):
                tile = source[y:y + tile_size, x:x + tile_size]
                result = model.predict(
                    tile, imgsz=imgsz, conf=min_conf, device=device, verbose=False
                )[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(result.names[class_id])
                    score = float(box.conf[0].item())
                    local = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    detections.append({
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": round(score, 5),
                        "box_page": [round(local[0] + x, 2), round(local[1] + y, 2),
                                     round(local[2] + x, 2), round(local[3] + y, 2)],
                        "tile_origin": [x, y],
                        "inference_source": pass_name,
                        "duplicate_candidates_merged": 0,
                    })
    return detections


def draw_overlay(image: np.ndarray, detections: list[dict], page: int) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 64), (245, 245, 245), -1)
    title = f"ML candidate locations - page {page} - {len(detections)} candidates"
    cv2.putText(overlay, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (35, 35, 35), 2, cv2.LINE_AA)
    for detection in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in detection["box_page"]]
        name, score = detection["class"], detection["confidence"]
        color = COLORS.get(name, (0, 0, 220))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 4)
        if detection.get("box_model_anchor"):
            ax0, ay0, ax1, ay1 = [int(round(v)) for v in detection["box_model_anchor"]]
            cv2.rectangle(overlay, (ax0, ay0), (ax1, ay1), color, 1)
        label = f"{name.replace('_', ' ')} {score:.2f}"
        scale = 0.58
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        label_y = max(th + 7, y0)
        cv2.rectangle(overlay, (x0, label_y - th - 7), (x0 + tw + 8, label_y + 3), color, -1)
        cv2.putText(overlay, label, (x0 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def write_csv(path: Path, report: dict) -> None:
    fields = ["page", "class", "confidence", "review_status", "x0", "y0", "x1", "y1",
              "anchor_x0", "anchor_y0", "anchor_x1", "anchor_y1",
              "inference_sources", "duplicate_candidates_merged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in report["pages"]:
            for detection in page["detections"]:
                x0, y0, x1, y1 = detection["box_page"]
                anchor = detection.get("box_model_anchor", detection["box_page"])
                writer.writerow({
                    "page": page["page"], "class": detection["class"],
                    "confidence": detection["confidence"],
                    "review_status": detection["review_status"],
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "anchor_x0": anchor[0], "anchor_y0": anchor[1],
                    "anchor_x1": anchor[2], "anchor_y1": anchor[3],
                    "inference_sources": "+".join(detection["inference_sources"]),
                    "duplicate_candidates_merged": detection["duplicate_candidates_merged"],
                })


def write_html(path: Path, report: dict) -> None:
    rows = []
    sections = []
    for page in report["pages"]:
        for name, count in page["class_counts"].items():
            rows.append(f"<tr><td>{page['page']}</td><td>{html.escape(name)}</td><td>{count}</td></tr>")
        det_rows = "".join(
            "<tr>"
            f"<td>{html.escape(d['class'])}</td><td>{d['confidence']:.3f}</td>"
            f"<td>{html.escape(d['review_status'])}</td>"
            f"<td>{html.escape('+'.join(d['inference_sources']))}</td>"
            f"<td>{', '.join(str(int(v)) for v in d['box_page'])}"
            f"{'<br><span class=\"muted\">anchor: ' + ', '.join(str(int(v)) for v in d['box_model_anchor']) + '</span>' if d.get('box_model_anchor') else ''}"
            f"{'<br><span class=\"muted\">' + html.escape(d.get('class_review_note', '')) + '</span>' if d.get('class_review_note') else ''}"
            "</td></tr>"
            for d in page["detections"]
        ) or "<tr><td colspan='5'>No candidates above the configured class thresholds.</td></tr>"
        sections.append(
            f"<section><h2>Page {page['page']}</h2>"
            f"<a href='{html.escape(page['overlay'])}'><img src='{html.escape(page['overlay'])}'></a>"
            "<table><thead><tr><th>Class</th><th>Confidence</th><th>Status</th>"
            f"<th>Pass</th><th>Page box x0,y0,x1,y1</th></tr></thead><tbody>{det_rows}</tbody></table></section>"
        )
    class_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{report['class_counts'].get(name, 0)}</td>"
        f"<td>{report['thresholds'][name]:.2f}</td></tr>" for name in CLASSES
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>P&amp;P ML detector review</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18212b}}
main{{max-width:1320px;margin:auto;padding:28px}} h1,h2{{color:#173b57}}
.notice{{padding:14px 18px;border-left:5px solid #c77d00;background:#fff3d6}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}}
.card,section{{background:white;padding:18px;border-radius:10px;box-shadow:0 1px 5px #0002}}
img{{max-width:100%;height:auto;border:1px solid #ccd5dd}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{padding:8px;border-bottom:1px solid #dce2e7;text-align:left}}
th{{background:#edf2f6}} code{{background:#eef1f3;padding:2px 5px}} .muted{{color:#56636f}}
</style></head><body><main>
<h1>Plan/Profile ML detection prototype</h1>
<p class="notice"><strong>Candidate-finding output, not a compliance verdict.</strong> Every box must remain reviewable. The detector locates drawing elements; OCR/vector extraction is still needed to read chainages, dimensions, levels, and design values.</p>
<div class="cards"><div class="card"><strong>Source</strong><br>{html.escape(report['pdf'])}</div>
<div class="card"><strong>Pages processed</strong><br>{len(report['pages'])}</div>
<div class="card"><strong>Candidates</strong><br>{report['detection_count']}</div>
<div class="card"><strong>Inference mode</strong><br>{html.escape(report['mode'])}</div></div>
<h2>What the approaches produce</h2>
<table><thead><tr><th>Approach</th><th>Best contribution</th><th>Current limitation</th></tr></thead><tbody>
<tr><td>Existing OCR/vector workflow</td><td>Reads text, chainages, levels, tables and numerical design parameters.</td><td>Layout/source-specific heuristics; weak on visual symbols and raster drawings.</td></tr>
<tr><td>Current ML detector</td><td>Locates and classifies summit/valley curves, gradients and plan/profile culverts with page coordinates.</td><td>Does not read the values inside or beside the detected feature; colored plan-culvert recall is still developing.</td></tr>
<tr><td>Expected hybrid product</td><td>ML finds the feature and supplies a focused crop; OCR/vector parsing reads its values; rules check compliance; the report links every finding back to the drawing.</td><td>Requires confidence calibration, feature-to-text association and human-review handling.</td></tr>
</tbody></table>
<h2>Candidate totals and operational thresholds</h2>
<table><thead><tr><th>Class</th><th>Count</th><th>Threshold</th></tr></thead><tbody>{class_rows}</tbody></table>
<p class="muted">“candidate-high” means confidence ≥ 0.65. It is still a model prediction, not verified truth. Lower accepted candidates are explicitly marked “candidate-review”.</p>
{''.join(sections)}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    pdf = args.pdf.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError(f"output must remain under {ROOT}")
    if "shimla" in pdf.name.lower() and not args.allow_sealed_test:
        raise ValueError("Shimla is sealed. This prototype refuses to process it.")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    model = YOLO(str(model_path))
    model_names = [str(model.names[i]) for i in sorted(model.names)]
    if model_names != CLASSES:
        raise ValueError(f"unexpected model taxonomy: {model_names}")

    doc = fitz.open(pdf)
    selected = page_numbers(args.pages, len(doc))
    report = {
        "schema_version": 1, "purpose": "human-review candidate finding",
        "pdf": str(pdf), "model": str(model_path), "mode": args.mode,
        "dpi": args.dpi, "imgsz": args.imgsz, "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap, "nms_iou": args.nms_iou,
        "thresholds": DEFAULT_THRESHOLDS, "pages": [],
    }
    min_conf = min(DEFAULT_THRESHOLDS.values())
    for page_index in selected:
        page_image = render(doc[page_index], args.dpi)
        raw = predict_tiles(model, page_image, args.mode, args.tile_size,
                            args.tile_overlap, args.imgsz, args.device, min_conf)
        filtered = [d for d in raw if d["class"] in DEFAULT_THRESHOLDS and
                    d["confidence"] >= DEFAULT_THRESHOLDS[d["class"]]]
        detections = suppress_duplicates(filtered, args.nms_iou)
        detections = expand_vertical_curve_boxes(detections, page_image.shape[1], page_image.shape[0])
        detections = merge_vertical_curve_fragments(detections)
        for detection in detections:
            detection["review_status"] = (
                "candidate-high" if detection["confidence"] >= 0.65 else "candidate-review"
            )
        detections.sort(key=lambda d: (d["class_id"], -d["confidence"]))
        overlay_name = f"overlays/page_{page_index + 1:04d}_ml_overlay.jpg"
        overlay = draw_overlay(page_image, detections, page_index + 1)
        if not cv2.imwrite(str(output / overlay_name), overlay,
                           [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise OSError(f"failed to write {output / overlay_name}")
        report["pages"].append({
            "page": page_index + 1, "page_width": page_image.shape[1],
            "page_height": page_image.shape[0], "raw_candidates": len(raw),
            "overlay": overlay_name.replace("\\", "/"),
            "class_counts": dict(Counter(d["class"] for d in detections)),
            "detections": detections,
        })
    doc.close()
    report["detection_count"] = sum(len(p["detections"]) for p in report["pages"])
    report["class_counts"] = dict(Counter(
        d["class"] for p in report["pages"] for d in p["detections"]
    ))
    (output / "ml_detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output / "ml_detections.csv", report)
    write_html(output / "review_report.html", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="1-3", help="1-based list/range or all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ml_prototype")
    parser.add_argument("--mode", choices=("color", "grayscale", "dual"), default="dual")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-sealed-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"output": str(args.output.resolve()),
                      "pages": len(report["pages"]),
                      "detections": report["detection_count"],
                      "class_counts": report["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
