# How to Run the Tempe Crash Chatbot

A 5-minute guide to running the chatbot on your own machine. Everything here is free.

## 1. One-time setup

Open a terminal in the `tempe_chatbot` folder and install the dependencies:

```bash
pip install -r requirements.txt
```

(Optional — only if you want Gemini Flash instead of the offline rule parser:)

```bash
setx GEMINI_API_KEY "AIza...your_key..."
```

Then **reopen the terminal** so it picks up the key. See `docs/gemini_setup_guide.md` for how to get a free, billing-safe key. Without a key the chatbot still works fully — it falls back to the built-in rule parser.

## 2. Build the data (only needed once, or after changing the school file)

```bash
python src/build_dataset.py        # creates data/crashes_enriched.csv
python src/build_buffer_table.py   # creates data/crash_school_pairs.csv
```

```bash
python src/make_overview_snapshot.py   # creates data/overview_all_crashes.png (landing image)
```

All three are already built and shipped in `data/`, so you can usually skip this.

## 3. Start the app

```bash
streamlit run app/app.py
```

Streamlit prints a local URL (usually http://localhost:8501). Open it in your browser.

## 4. Using it

Type a question in the chat box, or click one of the example buttons. The bot shows:

- the **text answer** (or "No data available" if the question is out of scope),
- a small **"How I read this"** panel showing how it parsed your question,
- a **breakdown** chart or ranking table when relevant, and
- an **interactive map** of the matching crashes (red = serious/fatal, blue = other), with the school buffer circle drawn for school questions.

### Sidebar controls

- **School counting view** — `buffer` (default; a crash near several schools counts for each) or `nearest` (each crash counts only for its closest school). You can also just type "nearest" in a question to override per-question.
- **Default school radius** — used when your question doesn't name a distance. Typing "within 0.5 mi" in a question overrides it.
- **Use Gemini when available** — toggle between Gemini parsing and the offline rule parser.

## 5. Example questions

- Which year had the most crashes?
- What time of day is most prevalent for accidents?
- Which school has the most crashes within 0.25 mi?
- How many alcohol related crashes at night in 2023?
- Break down crashes by weather
- How many crashes on Rural Rd?

Out-of-scope (returns "No data available"): other cities, dates after 2025, predictions, crash cost, vehicle make/model.

## Notes & limitations

- Data is **Tempe only, 2012–2025**, police-reported crashes.
- Counts reflect crash **frequency**, not risk per traffic volume.
- The map embed uses Streamlit's HTML component; if a future Streamlit version removes it, pin the version in `requirements.txt` or switch to `streamlit-folium`.
- Tests: `python src/test_query_engine.py`, `src/test_phase2.py`, `src/test_phase3.py` (38 checks total).
