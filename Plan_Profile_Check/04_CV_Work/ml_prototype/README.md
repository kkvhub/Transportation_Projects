# Page-level ML prototype

This prototype demonstrates the current V2 detector as a drawing-review aid.
All new code and outputs stay isolated under `04_CV_Work`. The engineering
hybrid entry point imports `02_Tool` read-only so the verified IRC rules remain
authoritative; it never writes to or modifies that preserved tool.

It produces:

- full-page detection overlays;
- JSON and CSV detections with drawing-page coordinates;
- overlap de-duplication across 1280 px tiles;
- optional color, grayscale, or dual-pass inference;
- an HTML review report explaining the intended ML + OCR hybrid workflow.

The default dual-pass mode tests the original drawing and a grayscale copy.
`culvert_plan` uses a provisional 0.15 threshold; the other four classes use
0.35. Lower-confidence accepted candidates are clearly marked for review.
These settings are experimental and must not be treated as compliance rules.

Example using the current held-out validation project:

```powershell
& 04_CV_Work\.venv\Scripts\python.exe `
  04_CV_Work\ml_prototype\infer_report.py `
  "00_Raw_data\Sunder Nagar Bypass P&P.pdf" `
  --pages 1-3 `
  --mode dual `
  --output "04_CV_Work\outputs\sunder_nagar_ml_demo"
```

The Shimla test project is refused by default and remains sealed.

## IRC engineering hybrid checker

The primary hybrid entry point is OCR/rules-first. It runs the original
engineering extraction and IRC rule engine, preserves horizontal curves,
vertical curves, structures and PASS/ADVISORY/HARD FAIL/INFO verdicts, and then
adds ML visual corroboration. It refuses to silently produce detector-only
output.

Run the hybrid workflow on any PDF that you are authorized to process:

```powershell
& 04_CV_Work\.venv\Scripts\python.exe `
  04_CV_Work\ml_prototype\engineering_hybrid_checker.py `
  "D:\path\to\your\external_drawing.pdf" `
  --road-class 2_lane `
  --terrain mountainous `
  --ml-pages all `
  --ml-mode dual `
  --ml-ocr auto `
  --output "04_CV_Work\outputs\external_drawing_test"
```

Open `hybrid_report.html` from the selected output folder. The same folder also
contains JSON and CSV results, full-page ML overlays, and one evidence crop per
detection.

The main HTML uses the original checker’s compact, table-first report style:
summary counts, extracted vertical curves, gradients, structures, cross-checks,
review status, and links to evidence. Large debug images and raw nearby text are
kept out of the main report; they remain available through the evidence links
and `hybrid_findings.json`.

For normal Windows use, the shorter wrapper command is:

```powershell
powershell -ExecutionPolicy Bypass -File 04_CV_Work\run_hybrid_prototype.ps1 `
  -Pdf "D:\Drawings\My_New_Plan_Profile.pdf" `
  -Pages "all" `
  -Mode dual `
  -Ocr auto `
  -RoadClass 2_lane `
  -Terrain mountainous `
  -OutputName "my_new_drawing_test"
```

The completed report will be at
`04_CV_Work\outputs\my_new_drawing_test\hybrid_compliance_report.html`.

The older `hybrid_engine.py` is retained only as an ML/text diagnostic. Its
report is not an IRC checker and must not be used as the final product.

The input PDF can be outside this repository, but the output is deliberately
required to stay under `04_CV_Work`. External test drawings are not copied into
the training dataset and do not affect later retraining.

`--ocr auto` uses embedded PDF text first and calls Tesseract only when a region
has little embedded text. On a machine that needs raster/scanned-PDF support,
install Tesseract OCR and `pytesseract` into the CV virtual environment. Use
`--ocr off` for vector PDFs when Tesseract is unavailable. Use `--pages 2,4-7`
to test only selected sheets.

Operational modes:

- `--mode dual`: best diagnostic coverage; runs color and grayscale passes.
- `--mode color`: faster and preserves colored drawing cues.
- `--mode grayscale`: useful for checking color-domain sensitivity.

The V2 checkpoint is only the default. Future retraining can replace it without
changing the report pipeline by passing `--model D:\path\to\new_best.pt`.

## Drawing-format routing

The engineering path now selects an extraction route from the PDF itself:

- Searchable/vector-text sheets use embedded labels and the original checker.
- Raster or stroke-font CAD sheets use page geometry to locate profile grids,
  curve-table cells, and profile structure schedules before targeted OCR.
- Combined plan/profile sheets and separate plan-only/profile-only sheets are
  supported. Separate sheets are paired by their printed chainage range.
- `NC` in the curve-table superelevation cell is reported and checked as 2.5%
  normal camber.
- Multiple OCR passes are reconciled field by field. Differing values are kept
  as conflicts in `hybrid_compliance_result.json`; they are not silently
  overwritten.
- The report exposes accepted/rejected candidates and high/medium/low curve
  extraction counts. Low-confidence or incomplete rows require drawing review.

This is format-aware rather than format-assuming, but it is not a promise that
every unseen consultant template will be read perfectly. A new layout should
first be run as an external test. Review the layout table, extraction QA,
horizontal/vertical/structure tables, and advisories before relying on its
engineering totals. External test drawings are not added to training data.

## Intended product architecture

1. ML finds and classifies a drawing feature.
2. Its page box creates a focused region for OCR/vector extraction.
3. Extracted chainage, geometry, dimensions, and levels are associated with
   the detected feature.
4. The deterministic rule engine evaluates those values.
5. The final report shows the finding, extracted evidence, confidence, and a
   link back to the marked drawing region.

The current prototype implements the full OCR/rules-first report path and ML
corroboration. On supported searchable and stroke-font formats it also performs
targeted extraction for steps 2–4. Spatial plan/profile reconciliation for every
structure style remains a review item and a future development area.
