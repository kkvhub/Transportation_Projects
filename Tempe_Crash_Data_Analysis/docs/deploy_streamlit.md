# Publishing the Chatbot (free, no Gemini key)

Yes — this app runs publicly on **Streamlit Community Cloud** straight from **GitHub**, with **no API key**. The natural-language router falls back to the built-in offline rule parser, so visitors to your portfolio can query it for free. (Adding a Gemini key later is optional and only improves handling of unusually phrased questions.)

## What gets published

The live app only needs the *processed* files (~25 MB total), which are committed:

- `data/crashes_enriched.csv`, `data/crash_school_pairs.csv`
- `data/schools_tempe_clean.json`, `data/overview_all_crashes.png`
- everything in `src/` and `app/`, plus `requirements.txt`

The large raw inputs (`crash_data_tempe.csv`, `feature_table_tempe.csv`, `school_crashes_tempe.csv`, `adot_roads.json`) and the build tools (matplotlib, google-genai) are **build-time only** and are excluded via `.gitignore` / `requirements-dev.txt`. The deployed app never loads them.

## Step 1 — Put the project on GitHub

This **Git_deploy** folder IS the repository root (`app/app.py` and `requirements.txt` sit at the top level, where Streamlit expects them). Push its *contents* to the repo.

```bash
cd Git_deploy
git init
git add .
git commit -m "Tempe crash chatbot"
git branch -M main
git remote add origin https://github.com/<you>/tempe-crash-chatbot.git
git push -u origin main
```

The `.gitignore` already keeps caches, backups, and the big raw inputs out, so the repo stays ~27 MB.

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub (free).
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<you>/tempe-crash-chatbot`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. (Optional) Advanced settings → Python version 3.11.
5. Click **Deploy**. First build takes a few minutes; you'll get a public URL like `https://tempe-crash-chatbot.streamlit.app`.

No secrets to configure — leave `GEMINI_API_KEY` unset and the app uses the rule parser automatically.

## Step 3 — Link it from your portfolio

Use the Streamlit URL as a button/link, e.g.:

```html
<a href="https://tempe-crash-chatbot.streamlit.app" target="_blank">
  Launch the Tempe Crash Chatbot
</a>
```

(You can also embed it in an `<iframe>`, but a "Launch" button is usually smoother.)

## Good to know (free tier)

- **Cold start:** the app sleeps after inactivity; the first visitor waits ~30 s while it wakes. After that it's instant.
- **Resources:** ~1 GB RAM per app. This app loads ~25 MB of data and renders static/lightweight maps, so it's well within limits.
- **No data risk:** all data here is public crash data; never commit an API key. (`.gitignore` and unset secrets handle this.)
- **Question coverage without Gemini:** the rule parser handles the documented patterns (year/time/day/month, school + radius, KSI/weather/alcohol/street filters, breakdowns, and combinations). Unusual phrasings may return "No data available" — the example buttons in the sidebar guide visitors to supported questions.

## Optional: enable Gemini on the cloud later

If you ever want LLM parsing in production: in Streamlit Cloud, open the app's **Settings → Secrets** and add

```
GEMINI_API_KEY = "AIza...your_free_key..."
```

and add `google-genai` to `requirements.txt`. The app picks it up automatically. (Free-tier, billing-safe key steps are in `gemini_setup_guide.md`.)

## Regenerating data (only if you change inputs)

On your own machine (not needed for deployment):

```bash
pip install -r requirements.txt -r requirements-dev.txt
python src/dedupe_schools.py          # if school file changed
python src/build_dataset.py
python src/build_buffer_table.py
python src/make_overview_snapshot.py
```
