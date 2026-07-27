from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
CV_ROOT = REPO_ROOT / "04_CV_Work"
RUNNER = CV_ROOT / "ml_prototype" / "engineering_hybrid_checker_with_cropper.py"
UPLOAD_DIR = CV_ROOT / "streamlit_uploads"
OUTPUT_ROOT = CV_ROOT / "outputs" / "streamlit_cropper_runs"
DEFAULT_MODEL = CV_ROOT / "plan_profile_curve_table_results" / "runs" / "retrain_v2_yolo11n_1280" / "weights" / "best.pt"


st.set_page_config(page_title="P&P Design Checker", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #07110d;
          --muted: #5e6a64;
          --line: #dbe5df;
          --accent: #b8ff3d;
          --soft: #f4f8f5;
        }
        .stApp { background: linear-gradient(180deg, #f7faf8 0%, #eef5f0 100%); color: var(--ink); }
        .block-container { padding-top: 2rem; max-width: 1180px; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0 !important; }
        .hero {
          background: #07110d; color: white; border-radius: 8px; padding: 28px 30px;
          border: 1px solid #13221a; box-shadow: 0 18px 45px rgba(7,17,13,.18);
        }
        .hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.1rem; }
        .hero p { color: #dce7df; margin: 0; max-width: 760px; }
        .pill {
          display: inline-block; background: var(--accent); color: #07110d; border-radius: 999px;
          padding: 4px 10px; font-weight: 700; font-size: .78rem; margin-bottom: 14px;
        }
        .metric-card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
        .metric-card strong { display: block; color: var(--muted); font-size: .78rem; text-transform: uppercase; }
        .metric-card span { color: var(--ink); font-size: 1.6rem; font-weight: 750; }
        .stButton > button, .stDownloadButton > button {
          border-radius: 6px; border: 1px solid #07110d; background: #07110d; color: white; font-weight: 700;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          border-color: #07110d; background: var(--accent); color: #07110d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def public_report_html(source_html: str) -> str:
    replacement = """
<h2>ML corroboration summary</h2>
<p class="meta">ML diagnostics were generated for audit, but detailed ML corroboration tables and page-by-page visual candidate tables are hidden in this public Streamlit report.</p>
"""
    pattern = re.compile(
        r"\s*<h2>Hybrid ML corroboration \(engineering results preserved\)</h2>.*?"
        r"<p class=\"meta\"><a href=\"ml_diagnostic/hybrid_report\.html\">Open detailed ML evidence appendix</a>.*?</p>",
        re.S,
    )
    cleaned = pattern.sub(replacement, source_html)
    cleaned = re.sub(r"\s*<h2>ML visual candidates by drawing page</h2>.*?</table>", "", cleaned, flags=re.S)
    return cleaned


def zip_folder(folder: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder.parent))
    return zip_path


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "uploaded_drawing"


def run_checker(pdf_path: Path, output_name: str, pages: str, mode: str, ocr: str,
                road_class: str, terrain: str, ml_dpi: int, model_path: Path) -> tuple[Path, dict, str]:
    output_dir = OUTPUT_ROOT / output_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(CV_ROOT / "ml_prototype"),
        str(REPO_ROOT / "02_Tool"),
        env.get("PYTHONPATH", ""),
    ])
    command = [
        sys.executable,
        str(RUNNER),
        str(pdf_path),
        "--ml-pages", pages,
        "--ml-mode", mode,
        "--ml-ocr", ocr,
        "--road-class", road_class,
        "--terrain", terrain,
        "--engineering-source", "auto",
        "--ml-dpi", str(ml_dpi),
        "--model", str(model_path),
        "--output", str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log_text = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise RuntimeError(log_text or f"Checker failed with exit code {completed.returncode}")

    result_path = output_dir / "hybrid_compliance_result.json"
    report_path = output_dir / "hybrid_compliance_report.html"
    if not result_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Checker finished, but expected report files were not created.")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    public_path = output_dir / "public_report.html"
    public_path.write_text(public_report_html(report_path.read_text(encoding="utf-8")), encoding="utf-8")
    zip_folder(output_dir, output_dir.with_suffix(".zip"))
    return output_dir, result, log_text


inject_theme()
st.markdown(
    """
    <div class="hero">
      <span class="pill">Plan & Profile - cropped ML route</span>
      <h1>Road Design Compliance Checker</h1>
      <p>Upload a Plan & Profile PDF, run the cropped-variant hybrid checker, and download a public report with detailed ML diagnostic tables hidden.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

missing = [path for path in (RUNNER, DEFAULT_MODEL, REPO_ROOT / "02_Tool" / "pp_checker") if not path.exists()]
if missing:
    st.error("Deployment is missing required checker files:\n\n" + "\n".join(str(path) for path in missing))
    st.stop()

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    uploaded = st.file_uploader("Upload Plan & Profile PDF", type=["pdf"])
    pages = st.text_input("Pages", value="all", help='Use "all" or ranges like 1-3,5.')
    mode = st.selectbox("ML mode", ["color", "dual", "grayscale"], index=0)
    ocr = st.selectbox("OCR mode", ["auto", "off", "required"], index=0)
with right:
    road_class = st.selectbox("Road class", ["2_lane", "4_lane"], index=0)
    terrain = st.selectbox("Terrain", ["mountainous", "plain_rolling", "steep"], index=0)
    ml_dpi = st.slider("ML DPI", min_value=100, max_value=600, value=220, step=20)
    model_path = Path(st.text_input("Model checkpoint", value=str(DEFAULT_MODEL)))

run_button = st.button("Run Cropped Checker", type="primary", disabled=uploaded is None)

if run_button and uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = UPLOAD_DIR / f"{timestamp}_{safe_stem(uploaded.name)}.pdf"
    pdf_path.write_bytes(uploaded.getbuffer())
    output_name = f"streamlit_{timestamp}_{safe_stem(uploaded.name)}"

    with st.status("Running cropped hybrid checker...", expanded=True) as status:
        st.write("Saving uploaded PDF")
        st.write("Running deterministic extraction, IRC rules, and cropped ML diagnostics")
        try:
            output_dir, result, log_text = run_checker(
                pdf_path=pdf_path,
                output_name=output_name,
                pages=pages,
                mode=mode,
                ocr=ocr,
                road_class=road_class,
                terrain=terrain,
                ml_dpi=ml_dpi,
                model_path=model_path,
            )
        except Exception as exc:
            status.update(label="Checker failed", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="Checker complete", state="complete")

    summary = result["engineering"]["rules"]["summary"]
    columns = st.columns(4)
    for col, label in zip(columns, ("PASS", "ADVISORY", "HARD FAIL", "INFO")):
        col.markdown(
            f'<div class="metric-card"><strong>{html.escape(label)}</strong><span>{summary.get(label, 0)}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Public Report")
    public_report = output_dir / "public_report.html"
    zip_path = output_dir.with_suffix(".zip")
    st.components.v1.html(public_report.read_text(encoding="utf-8"), height=900, scrolling=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Public HTML Report",
        data=public_report.read_bytes(),
        file_name=f"{safe_stem(uploaded.name)}_public_report.html",
        mime="text/html",
    )
    d2.download_button(
        "Download Full Output ZIP",
        data=zip_path.read_bytes(),
        file_name=f"{safe_stem(uploaded.name)}_checker_output.zip",
        mime="application/zip",
    )

    with st.expander("Run log"):
        st.code(log_text[-6000:] if log_text else "No log text captured.")
