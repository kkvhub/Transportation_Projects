# Tempe Crash Data Chatbot — Brainstorm & Roadmap

*Add-on project to the Tempe Road Safety Network. Last updated: June 20, 2026.*

---

## 1. The idea in one line

A small, project-specific assistant that answers natural-language questions about the Tempe crash dataset (e.g. *"which year had the most crashes?"*, *"which school has the most crashes within 0.25 mi?"*, *"what time of day is most dangerous?"*), returns a clear text answer, and — when the question is about a place — also plots the relevant crash points on an interactive map. If a question falls outside the data, it replies plainly that no data is available.

This is feasible, and the bulk of it can be built for free.

---

## 2. What the bot can actually answer (grounded in your data)

Your dataset is strong for this. The key files it would sit on top of:

- `data/raw/crash_data_tempe.csv` — 56,563 crash records, 2012–2025, 33 attributes each, with `Latitude`/`Longitude`.
- `data/processed/feature_table_tempe.csv` — engineered features (`crash_hour`, `crash_dow`, `crash_month`, `is_weekend`, `is_dark`, `KSI`, `dist_school_ft`, `dist_junction_ft`, `dist_busstop_ft`, `speed_limit_proxy`, `road_func_class`).
- `outputs/school_crashes_tempe.csv` — every crash tagged with `nearest_school`, `dist_school_mi`, `school_zone`.
- `outputs/school_crash_summary_tempe.csv` — per-school crash and KSI counts.
- `outputs/high_risk_crashes_tempe.csv`, `outputs/model_results_tempe.csv`, `outputs/population_density_tempe.csv`.

Because every crash has coordinates, **almost any answer can also be drawn on a map** by filtering the relevant rows and plotting their lat/long.

The questions naturally fall into a few families:

| Question family | Example | Columns used | Map output? |
|---|---|---|---|
| Time trends | "Which year had the most crashes?" / "Most dangerous month?" | `Year`, `crash_month`, `crash_hour`, `crash_dow` | Optional heat over time |
| Time-of-day / day-of-week | "What time is most prevalent for accidents?" | `crash_hour`, `is_weekend` | Filtered points |
| School proximity | "Which school has the most crashes within 0.25 mi?" | `nearest_school`, `dist_school_mi` | School + buffer + points |
| Severity / KSI | "Where are the worst (KSI) crashes?" | `KSI`, `Injuryseverity`, `Totalfatalities` | High-risk points |
| Conditions | "How many crashes happened in the rain / in the dark?" | `Weather`, `SurfaceCondition`, `is_dark`, `Lightcondition` | Filtered points |
| Behavior | "How many alcohol-related crashes?" | `AlcoholUse_Drv1/2`, `DrugUse_Drv1/2`, `Violation1_*` | Filtered points |
| Location / street | "Crashes on University Dr?" | `StreetName`, `CrossStreet`, lat/long | Street points |
| Cross-filters | "Night-time KSI crashes near schools in 2023" | combination of above | Filtered points |

Anything not represented in these columns — pedestrian-vs-cyclist breakdowns, exact dollar cost, weather forecasts, data for Phoenix/Mesa, anything after 2025 — should trigger the **"no data available for that"** response.

---

## 3. How it works (LLM-powered architecture)

You chose an **LLM-powered** natural-language engine. The cleanest and safest pattern here is **NL → structured query → local data → answer + map**. The LLM never sees the full dataset and never invents numbers; it only translates the question into a query plan that your own code runs against the CSVs.

```
┌──────────────┐    question      ┌─────────────────────┐
│   User UI    │ ───────────────▶ │   LLM (parser)      │
│ (chat + map) │                  │ free-form text  →   │
└──────────────┘                  │ structured intent   │
       ▲                          │ {metric, filters,   │
       │   answer + map points    │  groupby, geo?}     │
       │                          └──────────┬──────────┘
       │                                     │ JSON plan
       │                          ┌──────────▼──────────┐
       │                          │  Query engine       │
       │                          │  (pandas / DuckDB)  │
       │                          │  runs on your CSVs  │
       │                          └──────────┬──────────┘
       │   text + filtered rows              │
       └─────────────────────────────────────┘
                                             │ lat/long rows
                                  ┌──────────▼──────────┐
                                  │ Map renderer        │
                                  │ Leaflet / Folium    │
                                  └─────────────────────┘
```

Why this "LLM-as-translator, not LLM-as-answerer" design matters:

- **Accuracy** — counts and rankings come from pandas, not from the model's memory, so numbers are always correct.
- **No-data handling** — if the LLM maps a question to a metric/column that doesn't exist, the query layer returns empty and the bot says *"no data available."*
- **Cheap & fast** — the LLM only outputs a small JSON plan (tens of tokens), so even tight free tiers go a long way.
- **Safe** — restrict the LLM to a fixed schema of allowed fields/operations; reject anything else (prevents prompt-injection style misuse and hallucinated filters).

A small **rule-based fast-path** can sit in front for the handful of canned questions (year with most crashes, top school, peak hour) so common queries don't even need an API call. (This is effectively a light hybrid, but the LLM remains the general engine as you chose.)

---

## 4. Can it be done for free? — Yes, with realistic limits

**Everything except the LLM brain is free and open-source.** The only component with a potential cost is the language model, and even that has solid free options.

**Maps (free):**
- *Leaflet.js + OpenStreetMap tiles* — free, no API key, ideal for a web/HTML bot.
- *Folium* (Python wrapper over Leaflet) — free, great if you build in Python/Streamlit.
- *Marker clustering / heatmaps* via free plugins.
- (Google Maps and Mapbox also have free tiers but require keys and have caps — not needed here.)

**Data layer (free):** pandas or DuckDB over your existing CSVs. No database server needed.

**UI (free):** Streamlit, Gradio, or a plain HTML page. All free; Streamlit/Gradio can be hosted free on Hugging Face Spaces or Streamlit Community Cloud.

**The LLM (free tiers verified June 2026):**

| Option | Cost | Free limits (approx.) | Trade-off |
|---|---|---|---|
| **Ollama (local)** — Llama 3.1 8B / Mistral on your PC | 100% free, offline | Unlimited, no key, data never leaves your machine | Needs a decent CPU/GPU + ~5–8 GB RAM; setup effort |
| **Groq API** — Llama 3.1 8B / 3.3 70B | Free tier, no card | ~30 req/min, up to ~14,400 req/day (8B model) | Org-level limits; cloud (data leaves machine) |
| **Google Gemini Flash / Flash-Lite** | Free tier, no card | ~15 RPM (Flash) / ~30 RPM (Flash-Lite), ~1,500 req/day | Enabling billing removes the free tier |

For a single-user analytical chatbot, **any** of these is comfortably within free limits — you'll make at most a few requests per question. **Recommendation: prototype on Groq (fast, generous free tier, zero setup) and offer Ollama as the fully-offline/private fallback.**

Bottom line: a fully working, free version is realistic. The only "cost" is your build time and, for the cloud LLMs, sending the (short, non-sensitive) question text to a third party.

---

## 5. What other data might be required

You can ship a strong v1 with what you already have. These additions would *widen* what the bot can answer:

**Nice-to-have, already partially present:**
- **Pedestrian / cyclist / motorcycle flags** — `Unittype_One/Two` already hints at this; explicitly parsing it would unlock "pedestrian crashes near schools" questions.
- **Intersection names as a clean field** — you have `StreetName` + `CrossStreet`; a normalized intersection ID would make "worst intersection" queries exact.
- **AADT / traffic volume join** — `data/raw/adot_aadt.json` exists; joining it enables *crash-rate-per-traffic* answers (fairer than raw counts).

**Would require new sourcing:**
- **Speed limits (actual, not proxy)** — you currently use a `speed_limit_proxy` from road class. Real posted limits would improve speed-related answers.
- **School enrollment / start-end times** — to contextualize school-zone crashes (per-student rate, drop-off windows).
- **Pedestrian/bike counts** — exposure denominators for vulnerable-user risk.
- **Weather time-series** — you have per-crash weather already; external data only needed for forecasting, which is out of scope.
- **Post-2025 / live crash feed** — only if you want the bot to stay current; otherwise the bot is explicitly a 2012–2025 historical tool.

**Metadata the bot itself needs (small but important):**
- A **data dictionary / schema file** listing every queryable column, its type, and allowed values (e.g. the exact strings in `Weather`). The LLM uses this to map questions to fields and to know what's *out of scope*. This is the single most valuable artifact to create early.

---

## 6. Limitations to set expectations

**Data limitations**
- **Scope is Tempe, 2012–2025 only.** Anything outside that = "no data available."
- **Reporting bias** — only police-reported crashes; minor incidents are undercounted.
- **"Unknown" school** is the single largest bucket in the school summary (11,704 crashes), so school-proximity rankings exclude a big chunk of crashes — the bot must caveat this.
- **Counts ≠ risk.** Raw crash counts favor high-traffic roads; without an exposure denominator (AADT), "most dangerous" answers are about *frequency*, not *rate*. The bot should say which it's reporting.
- **Geocoding precision** — answers within a 0.25 mi radius depend on the accuracy of crash coordinates and the school point locations.

**Chatbot / LLM limitations**
- **Ambiguous phrasing** — "dangerous" could mean most crashes, most KSI, or highest rate; the bot should ask or state its assumption.
- **LLM mis-parsing** — occasionally the model maps a question to the wrong column; a confirmation echo ("Showing KSI crashes near schools in 2023 — correct?") mitigates this.
- **Free-tier rate caps** — fine for one user; a public demo with many users could hit Groq/Gemini daily limits. Local Ollama removes this.
- **Not predictive in chat** — your ML model predicts KSI, but wiring live prediction into the chatbot ("how risky is X intersection at night?") is a stretch goal, not v1.
- **No causal claims** — the bot reports correlations/patterns, not causes; it should avoid "X causes Y" phrasing.

**Privacy/ethics**
- Crash records can include driver age/gender/impairment. Keep the bot reporting **aggregates**, not individual-identifiable records, especially if hosted publicly.

---

## 7. Roadmap (phased)

### Phase 0 — Foundations (½–1 day)
- Write the **data dictionary / schema JSON**: queryable columns, types, allowed values, human-readable synonyms, and an explicit "out-of-scope" list.
- Decide hosting target (local Streamlit vs. web HTML) and LLM provider (Groq to start).
- Pin a clean, deduplicated query dataset (likely `feature_table_tempe.csv` joined with `school_crashes_tempe.csv` for names + coordinates).

### Phase 1 — Query engine, no LLM yet (1–2 days)
- Build a Python function library over the CSVs: `top_year()`, `peak_hour()`, `school_ranking(radius)`, `filter_crashes(**criteria)`, each returning *both* a text summary and the matching lat/long rows.
- Implement the **"no data available"** guard at this layer.
- Unit-test against your existing outputs (e.g. school summary numbers must match `school_crash_summary_tempe.csv`).

### Phase 2 — Map rendering (1 day)
- Add Leaflet/Folium rendering that takes any filtered row set → clustered markers + optional school buffer circle + heatmap toggle.
- Verify a known case visually (e.g. crashes within 0.25 mi of one named school).

### Phase 3 — LLM parsing layer (1–2 days)
- Define the **structured intent JSON schema** (metric, filters, groupby, radius, needs_map).
- Prompt the LLM to translate question → intent, constrained to the schema; reject/—> "no data" on anything unmapped.
- Add a confirmation echo of the interpreted query.
- Wire fast-path rules for the top canned questions.

### Phase 4 — Chat UI integration (1 day)
- Streamlit/Gradio (or HTML) layout: chat pane + live map pane side by side.
- Show answer text, the map, and a "based on N crashes" footnote with caveats.

### Phase 5 — Hardening & verification (1 day)
- Test suite of ~30 representative questions (incl. out-of-scope and ambiguous ones).
- Add rate-limit handling / fallback to local Ollama.
- Write a short user guide + list of supported question types.

### Stretch goals (optional, later)
- Join AADT for crash-*rate* answers.
- Live KSI risk prediction from your existing model.
- Resolve the "Unknown school" bucket.
- Public deployment on Hugging Face Spaces / Streamlit Cloud.

**Realistic effort for a polished v1: ~6–9 working days.** A rough demo answering your three headline questions with a map could be stood up in 1–2 days.

---

## 8. Recommended starting stack (all free)

- **Data:** pandas / DuckDB over existing CSVs
- **LLM:** Groq (Llama 3.1 8B) free tier → Ollama for offline/private mode
- **Map:** Folium/Leaflet + OpenStreetMap
- **UI:** Streamlit
- **Hosting (optional):** Streamlit Community Cloud or Hugging Face Spaces

---

*Sources for free-tier figures: Groq and Gemini free-tier documentation/aggregators, accessed June 2026 (limits change — re-verify before launch).*
