> **This is the deploy-ready copy.** Push the contents of this folder to GitHub and point Streamlit Community Cloud at `app/app.py`. Runs free with no API key. See `docs/deploy_streamlit.md`.

# Tempe Crash Data Chatbot

An LLM-powered assistant that answers natural-language questions about Tempe crash data (2012–2025) and plots results on an interactive map. Add-on to the Tempe Road Safety Network project.

## Folder structure

```
tempe_chatbot/
├── README.md              ← this file
├── data/                  ← datasets the bot queries
│   ├── school_crashes_tempe.csv      (primary table, 56,250 crashes + nearest school)
│   ├── crash_data_tempe.csv          (raw attributes: streets, severity, demographics)
│   ├── feature_table_tempe.csv       (engineered features)
│   ├── school_crash_summary_tempe.csv(per-school aggregates)
│   ├── schools_tempe_clean.json      (188 verified school points — GeoJSON)
│   └── tempe_schools_for_verification.csv (working verification sheet)
├── schema/
│   └── data_schema.json   ← Phase 0 data dictionary (LLM's map of queryable fields)
├── src/                   ← query engine (Phase 1, to build)
├── app/                   ← chat + map UI (Phase 4, to build)
└── docs/
    └── chatbot_roadmap.md ← full brainstorm + phased roadmap
```

## Status
- [x] Roadmap & architecture defined (`docs/chatbot_roadmap.md`)
- [x] School names verified; `schools_tempe_clean.json` rebuilt (188 schools, 0 blank names)
- [x] **Phase 0** — data schema/dictionary (`schema/data_schema.json`)
- [x] **Phase 1** — query engine + data pipeline (9/9 tests pass; "Unknown" school join fixed 11,704 → 0)
- [x] Gemini Flash billing-safe setup guide (`docs/gemini_setup_guide.md`) + test client (`src/llm_client.py`)
- [x] **Phase 2** — many-to-many buffer table + buffer (overlap) analysis + Folium map renderer (7/7 tests pass)
- [x] **Phase 3** — NL router: Gemini Flash intent parser + rule-based fallback + dispatcher (22/22 tests pass; buffer view default for schools)
- [x] **Phase 4** — Streamlit chat + map UI (`app/app.py`); verified end-to-end headless, no exceptions
- [x] **Phase 5** — hardening: co-located school dedup (188→142), input guards, fixed street-parse bug, 33-question battery (71 checks total pass)

## Composable queries (Phase 5 rework)
The engine now combines a **filter** (school + radius + attributes) with an **aggregation** (count / peak time / ranking / breakdown), so questions like these work:
- "crashes within 0.25 mi of Holdeman Elementary **with KSI = 1**" → 19
- "**what time of day** do most crashes happen near Holdeman Elementary" → 17:00 (74 of 697)

Core file: `src/analysis.py` (the `spec → subset → metric` engine). `src/nl_router.py` parses a question into a spec (Gemini or offline rules). The landing image (`src/make_overview_snapshot.py`) now draws crashes over a road-network basemap.

## Run the chatbot
```
pip install -r requirements.txt
streamlit run app/app.py
```
Works offline with the rule parser; set `GEMINI_API_KEY` to enable Gemini Flash. Full guide: `docs/how_to_run_chatbot.md`.

## Phase 1 files (in `src/`)
- `build_dataset.py` — merges the 3 source CSVs + re-computes nearest school (no cap) → `data/crashes_enriched.csv` (56,250 rows, 36 cols).
- `query_engine.py` — the query functions (busiest_year, peak_hour, peak_day_of_week, peak_month, school_ranking, crashes_near_school, filter_crashes, count_by). Each returns text + mappable rows; unknown fields return "No data available".
- `test_query_engine.py` — 9 sanity checks (run: `python src/test_query_engine.py`).
- `llm_client.py` — minimal billing-safe Gemini wrapper (key from env, model pinned to free-tier flash).

To rebuild the data from scratch: `python src/build_dataset.py`

## Phase 2 files (in `src/`)
- `build_buffer_table.py` — many-to-many crash↔school pairs within 0.5 mi → `data/crash_school_pairs.csv` (152,592 pairs, avg 3.5 schools/crash).
- `buffer_analysis.py` — school questions in the **buffer (overlap-allowed)** view: `school_ranking_buffer`, `crashes_in_buffer`. Complements the nearest-only view in `query_engine.py`.
- `map_render.py` — `render_result(...)` turns any query's rows into an interactive Folium/Leaflet HTML map (clustered crash points coloured by severity, optional school marker + buffer circle). Free, no API key.
- `test_phase2.py` — 7 checks (overlap exists, counts consistent, map HTML valid).

**Two school-count views:** *nearest-only* (each crash → 1 school, no double counting; `query_engine.school_ranking`) vs *buffer* (a crash counts for every school within range; `buffer_analysis.school_ranking_buffer`). At 0.25 mi, ~55% of in-buffer crashes fall near more than one school.

## Phase 3 files (in `src/`)
- `nl_router.py` — the natural-language layer. `answer(question)` parses a free-text question into a structured intent and dispatches it to the query functions. **Gemini Flash** is the primary parser (free tier); a deterministic **rule-based parser** is the fallback when no `GEMINI_API_KEY` is set or the call fails, so the bot always works offline. The LLM only picks an action + params — never the numbers. Unmapped questions → "No data available". **School questions default to the buffer view**; add "nearest" to switch.
- `test_phase3.py` — 22 checks (action mapping, buffer-default, filter extraction, out-of-scope, end-to-end).

Try it: `python src/nl_router.py` (runs on the rule engine; set `GEMINI_API_KEY` to use Gemini).

## Phase 5 — hardening
- `src/dedupe_schools.py` — collapses co-located school entries to one canonical campus per coordinate (188 → 142), keeping the rest as `aliases`. Backs up the original and writes `data/school_dedup_report.csv` for review. Re-run `build_dataset.py` + `build_buffer_table.py` afterward.
- Input guards in `nl_router.answer` (empty/junk questions → "No data available").
- Fixed a street-parsing bug (the "on" in "head-on" was wrongly read as a street name).
- `src/test_battery.py` — 33 representative questions end-to-end.

### Run all tests (71 checks)
```
cd src
python test_query_engine.py && python test_phase2.py && python test_phase3.py && python test_battery.py
```

### If you re-verify / edit the school dedup
1. Edit canonical names in `data/school_dedup_report.csv` (review only) or re-run `python src/dedupe_schools.py`.
2. Rebuild: `python src/build_dataset.py` then `python src/build_buffer_table.py`.

## Architecture (one line)
User question → LLM translates to a structured query (constrained to `schema/data_schema.json`) → pandas runs it on the CSVs → text answer + matching lat/long plotted on map. The LLM never invents numbers; unmapped questions return "No data available."

## LLM provider
Local **Ollama** for offline/private use; swap to **Groq/Gemini** free tier when deploying to a public host (Ollama can't run on Streamlit Cloud).
