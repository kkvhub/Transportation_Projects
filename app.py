"""
app.py  --  Tempe Crash Chatbot UI (public deployment build)
============================================================
Streamlit web app: type a question, get a data-backed answer plus an interactive
map of the matching crashes. Runs publicly on Streamlit Community Cloud with NO
API key (the router falls back to the offline rule parser).

PIPELINE
    question -> nl_router.answer(...) -> result {answer, rows, meta}
             -> show answer + optional breakdown/ranking
             -> map_render.render_result(...) -> interactive map embedded in page

LAYOUT
  * Left sidebar: intro, a "Data dictionary" button, an always-visible list of
    sample questions (click to run), and the author's links pinned at the bottom.
  * Main area: title, a "Data dictionary" button, the compact landing snapshot,
    then the chat (answers + maps).
  * A Data Dictionary dialog (popup) lists a curated set of dataset attributes and
    the values each can take, so visitors know what's explorable.
Beige theme via injected CSS + .streamlit/config.toml.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nl_router as nlr      # noqa: E402
import map_render as mr      # noqa: E402

GITHUB_URL = "https://github.com/kkvhub"
PORTFOLIO_URL = "https://kkvhub.github.io/"
STORYMAP_URL = "https://arcg.is/0HX09S1"

DEFAULT_VIEW = "buffer"
DEFAULT_RADIUS = 0.25

st.set_page_config(page_title="Tempe Crash Chatbot", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

# --- Beige theme -----------------------------------------------------------
st.markdown("""
<style>
/* Force colours explicitly so the look is identical whether or not Streamlit
   Cloud picks up .streamlit/config.toml (it doesn't when the app is in a
   subfolder, which previously left invisible text on the default dark theme). */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f4ead5 !important; color: #423a2f !important; }
[data-testid="stHeader"] { background: transparent; }

/* Sidebar: beige bg, dark text on everything, brown headings, orange links */
[data-testid="stSidebar"] { background-color: #efe2c6 !important; }
[data-testid="stSidebar"] * { color: #423a2f !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7a4b1e !important; }
[data-testid="stSidebar"] a { color: #9c5a1c !important; }

/* Headings + general text */
h1, h2, h3 { color: #7a4b1e !important; font-family: Georgia, 'Times New Roman', serif; }
[data-testid="stCaptionContainer"], small { color: #6b5c44 !important; }

/* Main-body markdown labels (e.g. "School ranking") + chat message text were
   white on the cloud's dark theme -> force dark so they're readable on beige. */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * { color: #423a2f !important; }
.stChatMessage, .stChatMessage p, .stChatMessage span, .stChatMessage div { color: #423a2f !important; }

/* Chat message bubbles */
.stChatMessage { background-color: #fbf4e4 !important; border: 1px solid #e6d6b8;
                 border-radius: 12px; }

/* Bottom chat-input bar (was a black strip with invisible text on cloud) */
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background-color: #f4ead5 !important; }
div[data-testid="stChatInput"] { background-color: #fbf4e4 !important;
    border: 1px solid #d9c39a; border-radius: 10px; }
div[data-testid="stChatInput"] textarea {
    background-color: #fbf4e4 !important; color: #423a2f !important;
    caret-color: #9c5a1c !important; }   /* visible blinking cursor */
div[data-testid="stChatInput"] textarea::placeholder { color: #8a7a60 !important; }

/* Buttons */
.stButton>button { background-color:#fbf4e4 !important; color:#5a4326 !important;
    border:1px solid #d9c39a; border-radius:8px; text-align:left; font-size:13px; }
.stButton>button:hover { background-color:#e3d0a8 !important; color:#3d2c14 !important; }

a { color: #9c5a1c; }
.side-refs { font-size:14px; }
.side-refs a { display:block; margin:4px 0; text-decoration:none; font-weight:600; }

/* Custom (CSS-controlled) ranking table + bar charts so they're readable on any
   Streamlit theme — replaces st.dataframe / st.bar_chart which follow the theme. */
.cb-table { width:100%; border-collapse:collapse; font-size:14px; color:#423a2f;
            background:#fbf4e4; margin:4px 0 10px; }
.cb-table th { background:#e3d0a8; color:#5a4326; text-align:left; padding:6px 10px;
               border:1px solid #d9c39a; }
.cb-table td { padding:6px 10px; border:1px solid #e6d6b8; }
.cb-bars { display:flex; align-items:flex-end; gap:6px; height:230px;
           padding:8px 4px 0; margin:4px 0 10px; overflow-x:auto;
           border-bottom:2px solid #d9c39a; }
.cb-col { display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
          flex:1 1 0; min-width:26px; height:100%; }
.cb-num { font-size:11px; color:#5a4326; margin-bottom:3px; white-space:nowrap; }
.cb-bar { width:70%; background:#c8843c; border-radius:3px 3px 0 0; }
.cb-cap { font-size:11px; color:#423a2f; margin-top:4px; text-align:center;
          white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Curated data dictionary (~15 most useful attributes). Shown in a popup so
# visitors understand what categories of data they can ask about.
# ---------------------------------------------------------------------------
DATA_DICTIONARY = [
    ("Year", "Calendar year of the crash", "2012 – 2025"),
    ("crash_hour", "Hour of day", "0 – 23 (e.g. 17 = 5 PM)"),
    ("crash_dow", "Day of week", "Monday … Sunday"),
    ("crash_month", "Month", "January … December"),
    ("KSI", "Killed or Seriously Injured", "1 = serious/fatal, 0 = not"),
    ("Injuryseverity", "Reported injury severity",
     "Fatal · Incapacitating Injury · Suspected Serious Injury · Possible Injury · No Injury"),
    ("Weather", "Weather at the time",
     "Clear · Cloudy · Rain · Fog Smog Smoke · Snow"),
    ("SurfaceCondition", "Road surface", "Dry · Wet · Ice Frost · Snow"),
    ("Lightcondition", "Lighting", "Daylight · Dark Lighted · Dark Not Lighted · Dusk · Dawn"),
    ("Collisionmanner", "How vehicles collided",
     "Rear End · Left Turn · Head On · Sideswipe · Single Vehicle · Angle"),
    ("AlcoholUse_Drv1", "Alcohol involvement (driver 1)", "Alcohol · No Apparent Influence"),
    ("DrugUse_Drv1", "Drug involvement (driver 1)", "Drugs · No Apparent Influence"),
    ("Unittype_One", "Road-user type", "Driver · Pedestrian · Pedalcyclist"),
    ("Gender_Drv1", "Driver 1 gender", "Male · Female · Unknown"),
    ("nearest_school / dist_school_mi", "Closest school & distance",
     "142 Tempe schools · distance in miles (e.g. within 0.25 mi)"),
    ("StreetName", "Street where the crash occurred",
     "548 streets — e.g. Rural Rd, Baseline Rd, University Dr"),
]


@st.dialog("📖 Data dictionary — what you can explore", width="large")
def show_data_dictionary():
    st.markdown("The chatbot can filter and aggregate the Tempe crash dataset by "
                "these attributes. **Combine them freely** in a question — e.g. "
                "*\"alcohol crashes at night in 2023\"* or *\"peak time near "
                "Holdeman Elementary\"*.")
    df = pd.DataFrame(DATA_DICTIONARY,
                      columns=["Attribute", "What it means", "Example values you can ask about"])
    st.dataframe(df, use_container_width=True, hide_index=True, height=560)
    st.caption("Scope: Tempe, Arizona only, 2012–2025. Anything outside this "
               "(other cities, future dates, predictions) returns \"No data available\".")


SAMPLE_QUESTIONS = [
    "Which year had the most crashes?",
    "What time of day is most prevalent for accidents?",
    "Which day of the week has the most crashes?",
    "Which school has the most crashes within 0.25 mi?",
    "How many crashes within 0.25 mi of Holdeman Elementary with KSI = 1?",
    "What time of day do most crashes happen near Holdeman Elementary?",
    "How many alcohol related crashes at night in 2023?",
    "How many crashes in the rain?",
    "Break down crashes by weather",
    "How many crashes on Rural Rd?",
]


def _set_q(q):
    st.session_state["pending_q"] = q


# ---------------------------------------------------------------------------
# SIDEBAR: intro, data-dictionary button, sample questions, references (bottom)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🚗 Tempe Crash Chatbot")
    st.caption("Ask about traffic crashes in Tempe, AZ (2012–2025). "
               "Free & runs on public data.")

    if st.button("📖  Data dictionary", key="dd_side", use_container_width=True):
        show_data_dictionary()

    st.markdown("**Sample questions** — click to run:")
    for q in SAMPLE_QUESTIONS:
        st.button(q, key=f"side_{q}", on_click=_set_q, args=(q,), use_container_width=True)

    st.divider()
    st.markdown(
        f'<div class="side-refs">'
        f'<b>Kaushlendra Kumar Verma</b>'
        f'<a href="{PORTFOLIO_URL}" target="_blank">🌐 My portfolio</a>'
        f'<a href="{GITHUB_URL}" target="_blank">🐙 My GitHub</a>'
        f'<a href="{STORYMAP_URL}" target="_blank">🗺️ Road Safety Analysis: Tempe, AZ — StoryMap</a>'
        f'</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("🚗 Tempe Crash Data Chatbot")
st.caption("Answers come straight from the data. Not sure what to ask? "
           "Open the data dictionary or pick a sample question on the left.")

if st.button("📖  Data dictionary", key="dd_main"):
    show_data_dictionary()

if "history" not in st.session_state:
    st.session_state["history"] = []


def _html_table(records):
    """Render a list of dicts as a beige HTML table (theme-proof)."""
    if not records:
        return ""
    cols = list(records[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in records:
        body += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>"
    return f'<table class="cb-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _html_bars(d):
    """Render a {label: value} dict as VERTICAL (column) CSS bars (theme-proof)."""
    if not d:
        return ""
    mx = max(d.values()) or 1
    cols = ""
    for k, v in d.items():
        h = max(2, int(v / mx * 100))          # bar height as % of plot area
        cols += (f'<div class="cb-col">'
                 f'<span class="cb-num">{int(v):,}</span>'
                 f'<div class="cb-bar" style="height:{h}%"></div>'
                 f'<span class="cb-cap">{k}</span></div>')
    return f'<div class="cb-bars">{cols}</div>'


def run_question(q):
    """Run one question end-to-end and append the turn to history."""
    with st.spinner("Analysing crashes…"):
        result = nlr.answer(q, prefer_llm=True, default_view=DEFAULT_VIEW,
                            default_radius=DEFAULT_RADIUS)
        map_html = None
        if result.get("map"):
            spec = result.get("meta", {}).get("spec") or result.get("intent", {})
            tmp = Path(tempfile.gettempdir()) / "tempe_chat_map.html"
            mr.render_result(result, tmp, title=q,
                             school=spec.get("school"), radius_mi=spec.get("radius_mi"))
            map_html = tmp.read_text(encoding="utf-8")
    st.session_state["history"].append({"q": q, "result": result, "map_html": map_html})


pending = st.session_state.pop("pending_q", None)
typed = st.chat_input("Ask a question about Tempe crashes...")
question = typed or pending
if question:
    run_question(question)

# Landing snapshot (compact, fits without scrolling) when no question asked yet.
if not st.session_state["history"]:
    st.subheader("All Tempe crashes, 2012–2025")
    st.caption("Snapshot (red = serious/fatal, blue = other). "
               "Ask a question for an interactive, zoomable map of the results.")
    overview_png = ROOT / "data" / "overview_all_crashes.png"
    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c2:
        if overview_png.exists():
            st.image(str(overview_png), use_container_width=True)
        else:
            st.info("Overview image not found. Run: python src/make_overview_snapshot.py")

# Conversation. (Maps can be heavy, so only the LATEST turn shows its interactive
# map; older turns keep the answer + charts. This keeps the page responsive.)
hist = st.session_state["history"]
for i, turn in enumerate(hist):
    is_last = (i == len(hist) - 1)
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        res = turn["result"]
        (st.success if res["ok"] else st.warning)(res["answer"])
        meta = res.get("meta", {})
        if meta.get("ranking"):
            st.markdown("**School ranking**")
            st.markdown(_html_table(meta["ranking"]), unsafe_allow_html=True)
        for key in ("by_year", "by_hour", "by_dow", "by_month", "breakdown"):
            if meta.get(key):
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                st.markdown(_html_bars(meta[key]), unsafe_allow_html=True)
                break
        if turn["map_html"] and is_last:
            st.components.v1.html(turn["map_html"], height=460, scrolling=False)
        elif turn["map_html"]:
            st.caption("🗺️ Map shown only for the most recent question (for speed) — "
                       "re-ask to view it again.")
