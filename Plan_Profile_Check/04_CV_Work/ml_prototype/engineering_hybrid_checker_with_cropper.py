"""Final IRC hybrid checker variant with profile-crop-routed ML detections.

This keeps the original engineering/rules path unchanged. The only difference
from engineering_hybrid_checker.py is the ML corroboration layer:

- full-page old 6-class detector is still used for curve_table and culverts;
- vertical_curve_summit, vertical_curve_valley, and gradient_segment candidates
  are taken from deterministic profile-chart crops.

Use this beside the original checker to compare the same PDF with and without
the cropper.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import fitz

import engineering_hybrid_checker as base
import hybrid_engine
from old_model_profile_crop_compare import (
    CLASSES,
    DEFAULT_MODEL,
    PROFILE_CLASSES,
    THRESHOLDS,
    draw_overlay,
    find_profile_rows,
    page_numbers,
    predict_tiles,
    render,
    remap_crop_detections,
    suppress_duplicates,
)
from ultralytics import YOLO


def cropped_detector_report(args) -> dict:
    """Emit the detector schema expected by hybrid_engine.build_hybrid()."""
    pdf = args.pdf.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if hybrid_engine.ROOT not in output.parents:
        raise ValueError(f"output must remain under {hybrid_engine.ROOT}")
    if "shimla" in pdf.name.lower() and not args.allow_sealed_test:
        raise ValueError("Shimla is sealed. This cropped checker refuses to process it.")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    (output / "profile_row_crops").mkdir(exist_ok=True)

    model = YOLO(str(model_path))
    model_names = [str(model.names[i]) for i in sorted(model.names)]
    if model_names != CLASSES:
        raise ValueError(f"unexpected model taxonomy: {model_names}")

    doc = fitz.open(pdf)
    selected = page_numbers(args.pages, len(doc))
    min_conf = min(THRESHOLDS.values())
    page_reports = []

    for page_index in selected:
        page_number = page_index + 1
        page = doc[page_index]
        image = render(page, args.dpi)
        crops = find_profile_rows(page, image, pad_px=12)

        full_raw = predict_tiles(
            model, image, args.mode, args.tile_size, args.tile_overlap,
            args.imgsz, args.device, min_conf
        )
        full_raw = [item | {"mode": "full"} for item in full_raw]
        full = suppress_duplicates(full_raw, args.nms_iou)
        non_profile = [item for item in full if item["class"] not in PROFILE_CLASSES]

        crop_profile = []
        crop_records = []
        for crop_index, crop in enumerate(crops, 1):
            x0, y0, x1, y1 = crop.box_page
            crop_image = image[y0:y1, x0:x1]
            crop_name = f"profile_row_crops/page_{page_number:04d}_{crop_index:02d}_{crop.name}.jpg"
            cv2.imwrite(str(output / crop_name), crop_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            crop_raw = predict_tiles(
                model, crop_image, args.mode, args.tile_size, args.tile_overlap,
                args.imgsz, args.device, min_conf
            )
            crop_profile.extend(
                item for item in remap_crop_detections(crop_raw, crop)
                if item["class"] in PROFILE_CLASSES
            )
            crop_records.append({
                "name": crop.name,
                "side": crop.side,
                "source": crop.source,
                "box_page": crop.box_page,
                "image": crop_name,
            })

        crop_profile = suppress_duplicates(crop_profile, args.nms_iou)
        detections = sorted(non_profile + crop_profile, key=lambda item: (item["class_id"], -item["confidence"]))

        overlay_name = f"overlays/page_{page_number:04d}_cropped_hybrid_overlay.jpg"
        overlay = draw_overlay(image, non_profile, crop_profile, crops, page_number)
        cv2.imwrite(str(output / overlay_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 91])

        for detection in detections:
            detection["inference_sources"] = [detection.get("inference_source", detection.get("mode", "unknown"))]

        page_reports.append({
            "page": page_number,
            "page_width": image.shape[1],
            "page_height": image.shape[0],
            "raw_candidates": len(full_raw),
            "overlay": overlay_name.replace("\\", "/"),
            "ml_overlay": overlay_name.replace("\\", "/"),
            "cropper": {
                "enabled": True,
                "strategy": "full-page non-profile + profile-crop profile classes",
                "crops": crop_records,
            },
            "class_counts": dict(Counter(item["class"] for item in detections)),
            "detections": detections,
        })

    doc.close()
    report = {
        "schema_version": 1,
        "purpose": "old 6-class detector with profile-crop routing",
        "pdf": str(pdf),
        "model": str(model_path),
        "mode": f"{args.mode}+profile_cropper",
        "dpi": args.dpi,
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "nms_iou": args.nms_iou,
        "thresholds": THRESHOLDS,
        "pages": page_reports,
        "detection_count": sum(len(page["detections"]) for page in page_reports),
        "class_counts": dict(Counter(item["class"] for page in page_reports for item in page["detections"])),
        "cropper_enabled": True,
    }
    (output / "cropped_detector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args):
    hybrid_engine.run_detector = cropped_detector_report
    result = base.run(args)
    output = args.output.resolve()
    diagnostic = output / "ml_diagnostic"
    source = diagnostic / "cropped_detector_report.json"
    if source.is_file():
        shutil.copy2(source, output / "cropped_detector_report.json")
    return result


def parser():
    result = base.parser()
    result.description = __doc__
    result.set_defaults(model=DEFAULT_MODEL)
    return result


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "variant": "with_profile_cropper",
        "engineering_source": result["engineering_provenance"]["mode"],
        "summary": result["engineering"]["rules"]["summary"],
        "horizontal_curves": len(result["engineering"]["model"]["curves"]),
        "vertical_curves": len(result["engineering"]["model"]["vertical_curves"]),
        "structures": len(result["engineering"]["model"]["structures"]),
        "ml_findings": len(result["ml"]["findings"]),
        "ml_class_counts": result["ml"]["class_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
