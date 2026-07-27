"""Run the clean 3-class profile-band detector on full drawing pages."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
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

CLASSES = ["profile_band", "vertical_curve_element", "gradient_segment"]
COLORS = {
    "profile_band": (225, 75, 20),
    "vertical_curve_element": (210, 210, 30),
    "gradient_segment": (35, 90, 210),
}
THRESHOLDS = {
    "profile_band": 0.45,
    "vertical_curve_element": 0.35,
    "gradient_segment": 0.35,
}
DEFAULT_MODEL = ROOT / "profile_band_results" / "runs" / "profile_band_yolo11n_1280" / "weights" / "best.pt"


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


def windows(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    step = size - overlap
    starts = list(range(0, length - size + 1, step))
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
            (
                existing
                for existing in kept
                if existing["class"] == candidate["class"]
                and iou(existing["box_page"], candidate["box_page"]) >= threshold
            ),
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


def predict_tiles(model: YOLO, image: np.ndarray, mode: str, tile_size: int, overlap: int,
                  imgsz: int, device: str) -> list[dict]:
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
                result = model.predict(
                    tile, imgsz=imgsz, conf=min(THRESHOLDS.values()), device=device, verbose=False
                )[0]
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
                        "tile_origin": [x, y],
                        "inference_source": pass_name,
                        "duplicate_candidates_merged": 0,
                    })
    return detections


def draw_overlay(image: np.ndarray, detections: list[dict], page: int) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 64), (245, 245, 245), -1)
    cv2.putText(
        overlay,
        f"Profile-band model - page {page} - {len(detections)} candidates",
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    for detection in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in detection["box_page"]]
        name, score = detection["class"], detection["confidence"]
        color = COLORS.get(name, (0, 0, 220))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 4)
        label = f"{name.replace('_', ' ')} {score:.2f}"
        scale = 0.58
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        label_y = max(th + 7, y0)
        cv2.rectangle(overlay, (x0, label_y - th - 7), (x0 + tw + 8, label_y + 3), color, -1)
        cv2.putText(overlay, label, (x0 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def profile_crop(image: np.ndarray, top_fraction: float, bottom_fraction: float) -> tuple[np.ndarray, int]:
    height = image.shape[0]
    y0 = int(round(height * top_fraction))
    y1 = int(round(height * bottom_fraction))
    if not 0 <= y0 < y1 <= height:
        raise ValueError("crop fractions must satisfy 0 <= top < bottom <= 1")
    return image[y0:y1, :], y0


def write_csv(path: Path, report: dict) -> None:
    fields = ["page", "class", "confidence", "x0", "y0", "x1", "y1",
              "inference_sources", "duplicate_candidates_merged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in report["pages"]:
            for detection in page["detections"]:
                x0, y0, x1, y1 = detection["box_page"]
                writer.writerow({
                    "page": page["page"],
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "inference_sources": "+".join(detection["inference_sources"]),
                    "duplicate_candidates_merged": detection["duplicate_candidates_merged"],
                })


def write_html(path: Path, report: dict) -> None:
    class_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{report['class_counts'].get(name, 0)}</td>"
        f"<td>{THRESHOLDS[name]:.2f}</td></tr>"
        for name in CLASSES
    )
    sections = []
    for page in report["pages"]:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(d['class'])}</td><td>{d['confidence']:.3f}</td>"
            f"<td>{html.escape('+'.join(d['inference_sources']))}</td>"
            f"<td>{', '.join(str(int(v)) for v in d['box_page'])}</td>"
            "</tr>"
            for d in page["detections"]
        ) or "<tr><td colspan='4'>No candidates above thresholds.</td></tr>"
        sections.append(
            f"<section><h2>Page {page['page']}</h2>"
            f"<a href='{html.escape(page['overlay'])}'><img src='{html.escape(page['overlay'])}'></a>"
            "<table><thead><tr><th>Class</th><th>Confidence</th><th>Pass</th><th>Box</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Profile-band detector review</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f5f7f9;color:#17212b}}
main{{max-width:1320px;margin:auto;padding:28px}} h1,h2{{color:#173b57}}
.notice{{padding:14px 18px;border-left:5px solid #2878b5;background:white}}
section{{background:white;padding:18px;margin:18px 0;border-radius:8px;box-shadow:0 1px 5px #0002}}
img{{max-width:100%;height:auto;border:1px solid #ccd5dd}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{padding:8px;border-bottom:1px solid #dce2e7;text-align:left}}
th{{background:#edf2f6}}
</style></head><body><main>
<h1>Profile-band detector review</h1>
<p class="notice">This report checks localization quality only. Summit/valley classification still belongs to later geometry/OCR/rule logic.</p>
<table><thead><tr><th>Class</th><th>Count</th><th>Threshold</th></tr></thead><tbody>{class_rows}</tbody></table>
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
        raise ValueError("Shimla is sealed. This profile-band diagnostic refuses to process it.")
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
        "schema_version": 1,
        "pdf": str(pdf),
        "model": str(model_path),
        "mode": args.mode,
        "dpi": args.dpi,
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "crop_top_fraction": args.crop_top_fraction,
        "crop_bottom_fraction": args.crop_bottom_fraction,
        "thresholds": THRESHOLDS,
        "pages": [],
    }
    for page_index in selected:
        image = render(doc[page_index], args.dpi)
        crop, crop_y0 = profile_crop(image, args.crop_top_fraction, args.crop_bottom_fraction)
        raw = predict_tiles(model, crop, args.mode, args.tile_size, args.tile_overlap, args.imgsz, args.device)
        for detection in raw:
            detection["box_page"][1] = round(detection["box_page"][1] + crop_y0, 2)
            detection["box_page"][3] = round(detection["box_page"][3] + crop_y0, 2)
            detection["crop_y_offset"] = crop_y0
        detections = suppress_duplicates(raw, args.nms_iou)
        detections.sort(key=lambda d: (d["class_id"], -d["confidence"]))
        overlay_name = f"overlays/page_{page_index + 1:04d}_profile_band_overlay.jpg"
        if not cv2.imwrite(str(output / overlay_name), draw_overlay(image, detections, page_index + 1),
                           [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise OSError(f"failed to write {output / overlay_name}")
        report["pages"].append({
            "page": page_index + 1,
            "page_width": image.shape[1],
            "page_height": image.shape[0],
            "raw_candidates": len(raw),
            "overlay": overlay_name.replace("\\", "/"),
            "class_counts": dict(Counter(d["class"] for d in detections)),
            "detections": detections,
        })
    doc.close()
    report["detection_count"] = sum(len(page["detections"]) for page in report["pages"])
    report["class_counts"] = dict(Counter(d["class"] for page in report["pages"] for d in page["detections"]))
    (output / "profile_band_detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output / "profile_band_detections.csv", report)
    write_html(output / "profile_band_review.html", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="1-3", help="1-based list/range or all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "profile_band_model_test")
    parser.add_argument("--mode", choices=("color", "grayscale", "dual"), default="color")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument("--crop-top-fraction", type=float, default=0.32)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.94)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-sealed-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "pages": len(report["pages"]),
        "detections": report["detection_count"],
        "class_counts": report["class_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
