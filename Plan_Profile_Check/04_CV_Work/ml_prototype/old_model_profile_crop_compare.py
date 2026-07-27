"""Compare the old 6-class detector on full pages vs vertical-schematic row crops.

This is a diagnostic wrapper only. It does not modify the older inference
scripts or the hybrid checker. It answers one question:

    Does routing the old detector through deterministic profile-row crops
    improve vertical-curve/gradient detections?
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
PROFILE_CLASSES = {"vertical_curve_summit", "vertical_curve_valley", "gradient_segment"}
COLORS = {
    "vertical_curve_summit": (42, 42, 220),
    "vertical_curve_valley": (220, 90, 30),
    "gradient_segment": (20, 170, 230),
    "culvert_plan": (190, 45, 190),
    "culvert_profile": (35, 165, 55),
    "curve_table": (0, 120, 255),
}
THRESHOLDS = {
    "vertical_curve_summit": 0.25,
    "vertical_curve_valley": 0.25,
    "gradient_segment": 0.25,
    "culvert_plan": 0.15,
    "culvert_profile": 0.30,
    "curve_table": 0.25,
}
DEFAULT_MODEL = (
    ROOT / "plan_profile_curve_table_results" / "runs" /
    "retrain_v2_yolo11n_1280" / "weights" / "best.pt"
)


@dataclass
class RowCrop:
    name: str
    side: str
    box_page: list[int]
    source: str


def page_numbers(value: str, count: int) -> list[int]:
    if value.lower() == "all":
        return list(range(count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(v) for v in part.split("-", 1))
            selected.update(range(start - 1, end))
        else:
            selected.add(int(part) - 1)
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"pages must be inside 1..{count}")
    return sorted(selected)


def render(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def windows(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    step = size - overlap
    starts = list(range(0, length - size + 1, step))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def suppress_duplicates(items: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(items, key=lambda item: item["confidence"], reverse=True):
        duplicate = next(
            (
                existing
                for existing in kept
                if existing["class"] == candidate["class"]
                and iou(existing["box_page"], candidate["box_page"]) >= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate["duplicate_candidates_merged"] += 1
    return kept


def text_row_crops(page: fitz.Page, image: np.ndarray, pad_px: int) -> list[RowCrop]:
    """Find rows using embedded text such as VERTICAL SCHEMATIC (LEFT)."""
    scale_x = image.shape[1] / float(page.rect.width)
    scale_y = image.shape[0] / float(page.rect.height)
    words = page.get_text("words")
    if not words:
        return []

    lines: dict[tuple[int, int], list[tuple]] = {}
    for word in words:
        block, line = int(word[5]), int(word[6])
        lines.setdefault((block, line), []).append(word)

    crops: list[RowCrop] = []
    for line_words in lines.values():
        line_words = sorted(line_words, key=lambda w: w[0])
        text = " ".join(str(w[4]) for w in line_words).upper()
        if "VERTICAL" not in text or "SCHEMATIC" not in text:
            continue
        side = "left" if "LEFT" in text else ("right" if "RIGHT" in text else "unknown")
        x0 = int(max(0, min(w[0] for w in line_words) * scale_x))
        y0 = int(max(0, min(w[1] for w in line_words) * scale_y))
        y1 = int(min(image.shape[0], max(w[3] for w in line_words) * scale_y))
        row_top, row_bottom = nearest_row_bounds(image, (y0 + y1) // 2, pad_px)
        crops.append(RowCrop(
            name=f"vertical_schematic_{side}",
            side=side,
            box_page=[0, row_top, image.shape[1], row_bottom],
            source=f"embedded_text:{x0},{y0},{y1}",
        ))
    return dedupe_crops(crops)


def horizontal_line_positions(image: np.ndarray) -> list[int]:
    """Find strong horizontal table/grid lines in the lower drawing region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_start = int(image.shape[0] * 0.25)
    roi = gray[lower_start:, :]
    dark = roi < 120
    # Prefer long table/grid separators over text strokes.
    counts = dark[:, : int(image.shape[1] * 0.95)].sum(axis=1)
    threshold = max(80, int(image.shape[1] * 0.18))
    candidates = np.where(counts >= threshold)[0] + lower_start
    if len(candidates) == 0:
        return []
    grouped: list[int] = []
    current = [int(candidates[0])]
    for y in candidates[1:]:
        y = int(y)
        if y - current[-1] <= 3:
            current.append(y)
        else:
            grouped.append(int(round(sum(current) / len(current))))
            current = [y]
    grouped.append(int(round(sum(current) / len(current))))
    return grouped


def nearest_row_bounds(image: np.ndarray, y_center: int, pad_px: int) -> tuple[int, int]:
    lines = horizontal_line_positions(image)
    above = [y for y in lines if y < y_center - 3]
    below = [y for y in lines if y > y_center + 3]
    top = above[-1] if above else max(0, y_center - int(image.shape[0] * 0.025))
    bottom = below[0] if below else min(image.shape[0], y_center + int(image.shape[0] * 0.025))
    return max(0, top - pad_px), min(image.shape[0], bottom + pad_px)


def fallback_row_crops(image: np.ndarray, pad_px: int) -> list[RowCrop]:
    """Fallback when embedded row-label text is not available."""
    lines = horizontal_line_positions(image)
    usable = [y for y in lines if image.shape[0] * 0.28 <= y <= image.shape[0] * 0.88]
    if len(usable) >= 2 and usable[-1] - usable[0] > image.shape[0] * 0.18:
        top = max(0, usable[0] - pad_px * 3)
        bottom = min(image.shape[0], usable[-1] + pad_px * 3)
        return [RowCrop("profile_chart_fallback", "unknown", [0, top, image.shape[1], bottom], "horizontal_line_fallback")]

    top = int(image.shape[0] * 0.30)
    bottom = int(image.shape[0] * 0.84)
    return [RowCrop("profile_chart_fallback", "unknown", [0, top, image.shape[1], bottom], "fraction_fallback")]


def dedupe_crops(crops: list[RowCrop]) -> list[RowCrop]:
    kept: list[RowCrop] = []
    for crop in sorted(crops, key=lambda item: (item.box_page[1], item.box_page[3])):
        duplicate = next((item for item in kept if iou(item.box_page, crop.box_page) >= 0.85), None)
        if duplicate is None:
            kept.append(crop)
    return kept


def find_profile_rows(page: fitz.Page, image: np.ndarray, pad_px: int) -> list[RowCrop]:
    crops = text_row_crops(page, image, pad_px)
    if crops:
        return crops
    return fallback_row_crops(image, pad_px)


def predict_tiles(model: YOLO, image: np.ndarray, mode: str, tile_size: int, overlap: int,
                  imgsz: int, device: str, min_conf: float) -> list[dict]:
    height, width = image.shape[:2]
    passes = ["color", "grayscale"] if mode == "dual" else [mode]
    detections: list[dict] = []
    for pass_name in passes:
        source = image if pass_name == "color" else cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
        for y in windows(height, tile_size, overlap):
            for x in windows(width, tile_size, overlap):
                tile = source[y:y + tile_size, x:x + tile_size]
                result = model.predict(tile, imgsz=imgsz, conf=min_conf, device=device, verbose=False)[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(result.names[class_id])
                    score = float(box.conf[0].item())
                    if score < THRESHOLDS.get(class_name, 1.0):
                        continue
                    local = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    detections.append({
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": round(score, 5),
                        "box_page": [round(local[0] + x, 2), round(local[1] + y, 2),
                                     round(local[2] + x, 2), round(local[3] + y, 2)],
                        "inference_source": pass_name,
                        "duplicate_candidates_merged": 0,
                    })
    return detections


def remap_crop_detections(detections: list[dict], crop: RowCrop) -> list[dict]:
    x_offset, y_offset = crop.box_page[0], crop.box_page[1]
    remapped = []
    for detection in detections:
        item = dict(detection)
        x0, y0, x1, y1 = item["box_page"]
        item["box_page"] = [
            round(x0 + x_offset, 2), round(y0 + y_offset, 2),
            round(x1 + x_offset, 2), round(y1 + y_offset, 2),
        ]
        item["crop_name"] = crop.name
        item["crop_side"] = crop.side
        item["crop_source"] = crop.source
        item["mode"] = "crop"
        remapped.append(item)
    return remapped


def draw_overlay(image: np.ndarray, full_detections: list[dict], crop_detections: list[dict],
                 crops: list[RowCrop], page_number: int) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 74), (245, 245, 245), -1)
    title = (
        f"Old 6-class model comparison - page {page_number} - "
        f"full {len(full_detections)} / crop {len(crop_detections)}"
    )
    cv2.putText(overlay, title, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)

    for crop in crops:
        x0, y0, x1, y1 = crop.box_page
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 80, 80), 2)
        cv2.putText(overlay, crop.name, (x0 + 10, max(95, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2, cv2.LINE_AA)

    def draw_detection(item: dict, thickness: int, suffix: str) -> None:
        x0, y0, x1, y1 = [int(round(v)) for v in item["box_page"]]
        name = item["class"]
        color = COLORS.get(name, (0, 0, 220))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
        label = f"{suffix} {name.replace('_', ' ')} {item['confidence']:.2f}"
        scale = 0.48
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        label_y = max(th + 8, y0)
        cv2.rectangle(overlay, (x0, label_y - th - 8), (x0 + tw + 8, label_y + 3), color, -1)
        cv2.putText(overlay, label, (x0 + 4, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    for item in full_detections:
        draw_detection(item, 2, "FULL")
    for item in crop_detections:
        draw_detection(item, 4, "CROP")
    return overlay


def write_csv(path: Path, pages: list[dict]) -> None:
    fields = [
        "page", "mode", "class", "confidence", "x0", "y0", "x1", "y1",
        "crop_name", "crop_side", "crop_source", "inference_source",
        "duplicate_candidates_merged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in pages:
            for mode in ("full_detections", "crop_detections"):
                for detection in page[mode]:
                    x0, y0, x1, y1 = detection["box_page"]
                    writer.writerow({
                        "page": page["page"],
                        "mode": detection["mode"],
                        "class": detection["class"],
                        "confidence": detection["confidence"],
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "crop_name": detection.get("crop_name", ""),
                        "crop_side": detection.get("crop_side", ""),
                        "crop_source": detection.get("crop_source", ""),
                        "inference_source": detection["inference_source"],
                        "duplicate_candidates_merged": detection["duplicate_candidates_merged"],
                    })


def write_html(path: Path, report: dict) -> None:
    sections = []
    for page in report["pages"]:
        rows = []
        for mode, detections in (("full", page["full_detections"]), ("crop", page["crop_detections"])):
            for detection in detections:
                rows.append(
                    "<tr>"
                    f"<td>{mode}</td><td>{html.escape(detection['class'])}</td>"
                    f"<td>{detection['confidence']:.3f}</td>"
                    f"<td>{html.escape(detection.get('crop_name', ''))}</td>"
                    f"<td>{', '.join(str(int(v)) for v in detection['box_page'])}</td>"
                    "</tr>"
                )
        table = "".join(rows) or "<tr><td colspan='5'>No detections above thresholds.</td></tr>"
        crop_rows = "".join(
            "<tr>"
            f"<td>{html.escape(crop['name'])}</td><td>{html.escape(crop['side'])}</td>"
            f"<td>{html.escape(crop['source'])}</td><td>{crop['box_page']}</td>"
            "</tr>"
            for crop in page["crops"]
        )
        sections.append(
            f"<section><h2>Page {page['page']}</h2>"
            f"<a href='{html.escape(page['overlay'])}'><img src='{html.escape(page['overlay'])}'></a>"
            "<h3>Detected row crops</h3>"
            f"<table><thead><tr><th>Name</th><th>Side</th><th>Source</th><th>Box</th></tr></thead><tbody>{crop_rows}</tbody></table>"
            "<h3>Detections</h3>"
            f"<table><thead><tr><th>Mode</th><th>Class</th><th>Confidence</th><th>Crop</th><th>Box</th></tr></thead><tbody>{table}</tbody></table>"
            "</section>"
        )
    class_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{report['full_class_counts'].get(name, 0)}</td>"
        f"<td>{report['crop_class_counts'].get(name, 0)}</td><td>{THRESHOLDS[name]:.2f}</td></tr>"
        for name in CLASSES
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Old model crop comparison</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f5f7f9;color:#17212b}}
main{{max-width:1320px;margin:auto;padding:28px}} h1,h2,h3{{color:#173b57}}
.notice{{padding:14px 18px;border-left:5px solid #b56b28;background:white}}
section{{background:white;padding:18px;margin:18px 0;border-radius:8px;box-shadow:0 1px 5px #0002}}
img{{max-width:100%;height:auto;border:1px solid #ccd5dd}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{padding:8px;border-bottom:1px solid #dce2e7;text-align:left}}
th{{background:#edf2f6}}
</style></head><body><main>
<h1>Old 6-class model: full page vs profile-row crop</h1>
<p class="notice">This is a diagnostic comparison. It does not change the production hybrid runner or retrain any model.</p>
<table><thead><tr><th>Class</th><th>Full-page count</th><th>Crop-routed count</th><th>Threshold</th></tr></thead><tbody>{class_rows}</tbody></table>
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
        raise ValueError("Shimla is sealed. This diagnostic refuses to process it.")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    (output / "row_crops").mkdir(exist_ok=True)

    model = YOLO(str(model_path))
    model_names = [str(model.names[i]) for i in sorted(model.names)]
    if model_names != CLASSES:
        raise ValueError(f"unexpected model taxonomy: {model_names}")

    doc = fitz.open(pdf)
    selected = page_numbers(args.pages, len(doc))
    pages: list[dict] = []
    min_conf = min(THRESHOLDS.values())
    for page_index in selected:
        page_number = page_index + 1
        page = doc[page_index]
        image = render(page, args.dpi)
        crops = find_profile_rows(page, image, args.row_pad_px)

        full = predict_tiles(model, image, args.mode, args.tile_size, args.tile_overlap,
                             args.imgsz, args.device, min_conf)
        full = [item | {"mode": "full"} for item in full]
        full = suppress_duplicates(full, args.nms_iou)

        crop_detections: list[dict] = []
        crop_records = []
        for crop_index, crop in enumerate(crops, 1):
            x0, y0, x1, y1 = crop.box_page
            crop_image = image[y0:y1, x0:x1]
            crop_name = f"page_{page_number:04d}_{crop_index:02d}_{crop.name}.jpg"
            cv2.imwrite(str(output / "row_crops" / crop_name), crop_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            raw_crop = predict_tiles(model, crop_image, args.mode, args.tile_size, args.tile_overlap,
                                     args.imgsz, args.device, min_conf)
            crop_detections.extend(remap_crop_detections(raw_crop, crop))
            crop_records.append({
                "name": crop.name,
                "side": crop.side,
                "source": crop.source,
                "box_page": crop.box_page,
                "image": f"row_crops/{crop_name}",
            })
        crop_detections = suppress_duplicates(crop_detections, args.nms_iou)

        overlay_name = f"overlays/page_{page_number:04d}_old_model_crop_compare.jpg"
        overlay = draw_overlay(image, full, crop_detections, crops, page_number)
        cv2.imwrite(str(output / overlay_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 91])

        pages.append({
            "page": page_number,
            "page_width": image.shape[1],
            "page_height": image.shape[0],
            "crops": crop_records,
            "overlay": overlay_name.replace("\\", "/"),
            "full_detections": full,
            "crop_detections": crop_detections,
            "full_class_counts": dict(Counter(item["class"] for item in full)),
            "crop_class_counts": dict(Counter(item["class"] for item in crop_detections)),
        })
    doc.close()

    report = {
        "schema_version": 1,
        "pdf": str(pdf),
        "model": str(model_path),
        "mode": args.mode,
        "dpi": args.dpi,
        "pages": pages,
        "thresholds": THRESHOLDS,
        "full_detection_count": sum(len(page["full_detections"]) for page in pages),
        "crop_detection_count": sum(len(page["crop_detections"]) for page in pages),
        "full_profile_detection_count": sum(
            item["class"] in PROFILE_CLASSES for page in pages for item in page["full_detections"]
        ),
        "crop_profile_detection_count": sum(
            item["class"] in PROFILE_CLASSES for page in pages for item in page["crop_detections"]
        ),
        "full_class_counts": dict(Counter(item["class"] for page in pages for item in page["full_detections"])),
        "crop_class_counts": dict(Counter(item["class"] for page in pages for item in page["crop_detections"])),
    }
    (output / "old_model_crop_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output / "old_model_crop_compare.csv", pages)
    write_html(output / "old_model_crop_compare.html", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="1-3", help="1-based list/range or all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "old_model_crop_compare")
    parser.add_argument("--mode", choices=("color", "grayscale", "dual"), default="color")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--row-pad-px", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-sealed-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "pages": len(report["pages"]),
        "full_detections": report["full_detection_count"],
        "crop_detections": report["crop_detection_count"],
        "full_profile_detections": report["full_profile_detection_count"],
        "crop_profile_detections": report["crop_profile_detection_count"],
        "full_class_counts": report["full_class_counts"],
        "crop_class_counts": report["crop_class_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
