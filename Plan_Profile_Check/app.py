from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
DEFAULT_OCR_MODE = "auto"
DEFAULT_ML_DPI = 400
CACHE_ROOT = Path(tempfile.gettempdir()) / "pp_checker_cache"
REQUIRED_PYTHON_MODULES = {
    "cv2": "opencv-python-headless",
    "fitz": "PyMuPDF",
    "numpy": "numpy",
    "PIL": "pillow",
    "torch": "torch",
    "ultralytics": "ultralytics",
}


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
          --warning: #f59e0b;
          --danger: #dc2626;
        }
        .stApp { background: #eef5f0; color: var(--ink); }
        .block-container { padding-top: 1.4rem; max-width: 1240px; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0 !important; }
        div[data-testid="stAppViewContainer"] label,
        div[data-testid="stAppViewContainer"] label p,
        div[data-testid="stAppViewContainer"] .stTextInput label,
        div[data-testid="stAppViewContainer"] .stSelectbox label,
        div[data-testid="stAppViewContainer"] .stFileUploader label {
          color: var(--ink) !important;
          font-weight: 750 !important;
        }
        div[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
        div[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
        div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p {
          color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] {
          background: #07110d;
          border-right: 1px solid #13221a;
        }
        section[data-testid="stSidebar"] * { color: #eaf3ed; }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stSlider label {
          color: #eaf3ed !important;
        }
        .hero {
          background:
            radial-gradient(circle at 85% 15%, rgba(184,255,61,.25), transparent 28%),
            linear-gradient(135deg, #07110d 0%, #10251a 72%, #163822 100%);
          color: white; border-radius: 8px; padding: 34px 34px;
          border: 1px solid #13221a; box-shadow: 0 18px 45px rgba(7,17,13,.20);
        }
        .hero h1 { color: white; margin: 0 0 10px 0; font-size: 2.35rem; line-height: 1.08; }
        .hero p { color: #dce7df; margin: 0; max-width: 820px; font-size: 1.03rem; }
        .pill {
          display: inline-block; background: var(--accent); color: #07110d; border-radius: 999px;
          padding: 4px 10px; font-weight: 700; font-size: .78rem; margin-bottom: 14px;
        }
        .step-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 14px 0 12px; }
        .step-card {
          background: #ffffff; border: 1px solid var(--line); border-radius: 8px; padding: 16px;
          min-height: 112px;
        }
        .step-card strong { display:block; color:#07110d; margin-bottom: 6px; }
        .step-card p { margin: 0; color: var(--muted); font-size: .92rem; }
        .sidebar-title { font-size: 1.25rem; font-weight: 800; margin: .5rem 0 .15rem; color: white; }
        .sidebar-muted { color: #b8c8bf; font-size: .9rem; line-height: 1.4; }
        .side-box {
          border: 1px solid rgba(255,255,255,.12); border-radius: 8px; padding: 12px 13px; margin: 12px 0;
          background: rgba(255,255,255,.045);
        }
        .side-box strong { color: var(--accent); }
        .side-box ul { padding-left: 18px; margin: 8px 0 0; }
        .side-box li { margin-bottom: 5px; color: #e6eee9; }
        .profile-links a {
          display: inline-block; margin: 6px 8px 0 0; padding: 7px 10px; border-radius: 6px;
          background: var(--accent); color: #07110d !important; text-decoration: none; font-weight: 800;
        }
        .metric-card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
        .metric-card strong { display: block; color: var(--muted); font-size: .78rem; text-transform: uppercase; }
        .metric-card span { color: var(--ink); font-size: 1.6rem; font-weight: 750; }
        .report-frame {
          border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: white;
          box-shadow: 0 10px 28px rgba(7,17,13,.07);
        }
        .footer {
          margin-top: 30px; padding: 18px 20px; border-top: 1px solid var(--line);
          color: var(--muted); font-size: .92rem; text-align: center;
        }
        .footer a { color: #07110d; font-weight: 800; text-decoration: none; }
        .stButton > button {
          border-radius: 6px; border: 1px solid var(--accent); background: var(--accent); color: #07110d;
          font-weight: 900; padding: .55rem 1rem; box-shadow: 0 0 0 1px rgba(184,255,61,.35), 0 10px 24px rgba(184,255,61,.28);
        }
        .stDownloadButton > button {
          border-radius: 6px; border: 1px solid #07110d; background: #07110d; color: white; font-weight: 800;
          padding: .55rem 1rem;
        }
        .stButton > button:hover {
          border-color: #07110d; background: #d7ff75; color: #07110d;
          box-shadow: 0 0 0 1px rgba(184,255,61,.65), 0 14px 30px rgba(184,255,61,.36);
        }
        .stDownloadButton > button:hover {
          border-color: #07110d; background: var(--accent); color: #07110d;
        }
        .stButton > button:disabled {
          background: #dbe5df !important;
          border-color: #aab8af !important;
          color: #5e6a64 !important;
          opacity: 1 !important;
        }
        div[data-testid="stFileUploader"] {
          background: #f9fbfa; border: 1px dashed #aab8af; border-radius: 8px; padding: 12px;
        }
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {
          color: var(--ink) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: #ffffff; border-color: var(--line); box-shadow: 0 10px 28px rgba(7,17,13,.07);
        }
        @media (max-width: 900px) {
          .step-grid { grid-template-columns: 1fr; }
          .hero h1 { font-size: 1.8rem; }
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


def missing_python_packages() -> list[str]:
    return [
        package
        for module, package in REQUIRED_PYTHON_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]


def run_checker(pdf_path: Path, output_name: str, pages: str, mode: str, ocr: str,
                road_class: str, terrain: str, ml_dpi: int, model_path: Path) -> tuple[Path, dict, str]:
    output_dir = OUTPUT_ROOT / output_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cache_dir in (
        CACHE_ROOT / "Ultralytics",
        CACHE_ROOT / "matplotlib",
        CACHE_ROOT / "torch",
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "Ultralytics"))
    env.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
    env.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
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
        "--allow-no-tesseract",
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

with st.sidebar:
    st.markdown('<div class="sidebar-title">P&P Checker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-muted">Hybrid design checker for road Plan & Profile drawings.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="side-box">
          <strong>What this tool does</strong>
          <ul>
            <li>Reads Plan & Profile PDF drawings.</li>
            <li>Extracts design elements using OCR/vector parsing.</li>
            <li>Runs IRC rule checks on extracted values.</li>
            <li>Uses ML detections as visual support.</li>
          </ul>
        </div>
        <div class="side-box">
          <strong>Current limitations</strong>
          <ul>
            <li>ML detections are advisory, not ground truth.</li>
            <li>Unclear OCR fields may become unknown/review items.</li>
            <li>New consultant drawing formats need manual QA.</li>
            <li>Large PDFs can take several minutes.</li>
          </ul>
        </div>
        <div class="side-box">
          <strong>Output</strong>
          <ul>
            <li>Public HTML report.</li>
            <li>Full diagnostic ZIP.</li>
            <li>Hidden ML detail tables in public view.</li>
          </ul>
        </div>
        <div class="side-box">
          <strong>Created by</strong><br>
          <span class="sidebar-muted">Kaushlendra Kumar Verma</span>
          <div class="profile-links">
            <a href="https://kkvhub.github.io/" target="_blank">Portfolio</a>
            <a href="https://www.linkedin.com/in/kaushlendra-kumar-verma/" target="_blank">LinkedIn</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <span class="pill">Plan & Profile design review</span>
      <h1>Road Design Compliance Checker</h1>
      <p>Upload a road Plan & Profile PDF and generate an IRC-oriented design-check report. The app combines deterministic extraction, rule checks, and ML visual corroboration.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="step-grid">
      <div class="step-card"><strong>1. Upload Drawing</strong><p>Use a Plan & Profile PDF. You can run all pages or a selected page range for faster review.</p></div>
      <div class="step-card"><strong>2. Run Checker</strong><p>The engine extracts curves, profile items, structures, and design values, then applies IRC checks.</p></div>
      <div class="step-card"><strong>3. Review Output</strong><p>Download a clean public report or the full ZIP with detailed ML diagnostics and evidence files.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = [path for path in (RUNNER, DEFAULT_MODEL, REPO_ROOT / "02_Tool" / "pp_checker") if not path.exists()]
if missing:
    st.error("Deployment is missing required checker files:\n\n" + "\n".join(str(path) for path in missing))
    st.stop()

missing_packages = missing_python_packages()
if missing_packages:
    st.error(
        "Deployment is missing required Python packages:\n\n"
        + "\n".join(f"- {package}" for package in missing_packages)
        + "\n\nCheck `requirements.txt`, then reboot/redeploy the Streamlit app."
    )
    st.stop()

with st.container(border=True):
    st.subheader("Run Configuration")
    upload_col, settings_col = st.columns(2, gap="large")
    with upload_col:
        uploaded = st.file_uploader("Upload Plan & Profile PDF", type=["pdf"])
        pages = st.text_input("Pages", value="all", help='Use "all" or ranges like 1-3,5.')
    with settings_col:
        road_class = st.selectbox("Road class", ["2_lane", "4_lane"], index=0)
        terrain = st.selectbox("Terrain", ["mountainous", "plain_rolling", "steep"], index=0)
        mode = st.selectbox("ML mode", ["color", "dual", "grayscale"], index=0)

    run_button = st.button("Run Checker", type="primary", disabled=uploaded is None, use_container_width=True)

if run_button and uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = UPLOAD_DIR / f"{timestamp}_{safe_stem(uploaded.name)}.pdf"
    pdf_path.write_bytes(uploaded.getbuffer())
    output_name = f"streamlit_{timestamp}_{safe_stem(uploaded.name)}"

    with st.status("Running design checker...", expanded=True) as status:
        st.write("Saving uploaded PDF")
        st.write("Running deterministic extraction, IRC rules, and ML diagnostics")
        try:
            output_dir, result, log_text = run_checker(
                pdf_path=pdf_path,
                output_name=output_name,
                pages=pages,
                mode=mode,
                ocr=DEFAULT_OCR_MODE,
                road_class=road_class,
                terrain=terrain,
                ml_dpi=DEFAULT_ML_DPI,
                model_path=DEFAULT_MODEL,
            )
        except Exception as exc:
            status.update(label="Checker failed", state="error", expanded=True)
            error_text = str(exc).strip() or repr(exc)
            st.error("Checker failed. The technical log below shows the exact cause.")
            st.code(error_text[-12000:], language="text")
            st.stop()
        status.update(label="Checker complete", state="complete")

    summary = result["engineering"]["rules"]["summary"]
    columns = st.columns(4)
    for col, label in zip(columns, ("PASS", "ADVISORY", "HARD FAIL", "INFO")):
        col.markdown(
            f'<div class="metric-card"><strong>{html.escape(label)}</strong><span>{summary.get(label, 0)}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Result Summary")
    meta1, meta2, meta3 = st.columns(3)
    meta1.info(f"Horizontal curves: {len(result['engineering']['model'].get('curves', []))}")
    meta2.info(f"Vertical curves: {len(result['engineering']['model'].get('vertical_curves', []))}")
    meta3.info(f"Structures: {len(result['engineering']['model'].get('structures', []))}")

    st.subheader("Public Report")
    public_report = output_dir / "public_report.html"
    zip_path = output_dir.with_suffix(".zip")
    st.markdown('<div class="report-frame">', unsafe_allow_html=True)
    st.components.v1.html(public_report.read_text(encoding="utf-8"), height=900, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)

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

st.markdown(
    """
    <div class="footer">
      Built by <strong>Kaushlendra Kumar Verma</strong> -
      <a href="https://kkvhub.github.io/" target="_blank">Portfolio</a> -
      <a href="https://www.linkedin.com/in/kaushlendra-kumar-verma/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
