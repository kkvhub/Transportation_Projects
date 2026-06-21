# Gemini Flash Setup — Billing-Safe Guide

How to wire Gemini Flash into the chatbot so it **can never generate a charge**. Written June 2026; re-check Google's pages before launch as limits change.

---

## The one thing you must understand first

Your **Google AI Pro subscription** (Gemini app / Gemini Advanced) and the **Gemini API** are two completely separate products:

- **Google AI Pro** = the consumer chat app subscription. It does **not** give the chatbot any API access or quota.
- **Gemini API** (what our code calls) = a developer service with its **own free tier**, obtained from **Google AI Studio**. This is what we use.

So we ignore the Pro subscription for this project and use the **free Gemini API tier**, which requires only a Google account and **no credit card**.

---

## The golden rule for never being billed

> **Get the key from Google AI Studio and never attach a billing account to the project.**

The free tier is the *default* state of a fresh API key. Billing only ever starts if **you** explicitly enable it. Specifically:

- A key created in **AI Studio with no Cloud billing account linked** is free-tier only. If you exceed a limit, requests get **rejected with a 429 error** — they are **not** silently billed.
- The moment you (or anyone) **enable billing** on that Google Cloud project, the free tier **disappears for that project** and *every* call becomes billable from the first token. **Never do this** for the chatbot's project.

That's the whole safety model: no billing account linked = hard free ceiling, not a meter.

---

## Step-by-step: get a free, billing-safe key

1. Go to **https://aistudio.google.com** and sign in with a normal Google account.
2. In the left sidebar click **"Get API key"** → **"Create API key"**.
3. When asked which project to use, let it create a **new project** (e.g. `tempe-chatbot`). **Do not** pick a project that already has billing enabled.
4. Copy the key (it starts with `AIza…`). Treat it like a password.
5. **Do NOT** visit Google Cloud Billing and link a card to this project. Leave billing off. That's it — the key is now free-tier-locked.

(Optional belt-and-suspenders: in the Google Cloud Console for this project, confirm **Billing → "This project has no billing account"**. If it says that, you cannot be charged.)

---

## Free-tier limits you're working within (2026)

| Model | Requests/min | Requests/day | Notes |
|---|---|---|---|
| **gemini-2.0-flash** | ~15 RPM | ~1,500 RPD | Generous tokens (~1M TPM); good default |
| **gemini-2.5-flash** | ~10 RPM | ~1,500 RPD | Newer, slightly tighter RPM |
| **gemini-2.5-flash-lite** | ~30 RPM | ~1,500 RPD | Fastest, best for bursts |

For a single-user analytical chatbot you'll make ~1 request per question, so these are far more than enough. If you ever hit a limit you get a `429` (try again later) — **never a charge**.

**Privacy note:** on the free tier, Google may use prompt content to improve its products. We only ever send the user's *question text* (never the crash dataset), but avoid typing anything sensitive into the chatbot.

---

## How the code uses the key (no secrets in source)

Store the key in an **environment variable**, never hard-coded in a `.py` file that could end up shared or in git.

**Windows (PowerShell), set once for your user:**
```powershell
setx GEMINI_API_KEY "AIza...your_key_here..."
```
(Re-open the terminal afterwards so it picks up the variable.)

Then install the official SDK:
```powershell
pip install google-genai
```

The chatbot reads the key from the environment, so the key never appears in any file we save:
```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # raises if not set
resp = client.models.generate_content(
    model="gemini-2.0-flash",          # a FREE-TIER model — do not switch to a Pro model
    contents="Say hello in 3 words.",
)
print(resp.text)
```

A minimal, commented test client lives at `src/llm_client.py` — run it after setting the key to confirm everything works end-to-end before we build the full NL→query layer in Phase 3.

---

## Staying safe — quick checklist

- [ ] Key created in **AI Studio**, not by enabling billing in Cloud.
- [ ] Project has **no billing account** linked.
- [ ] Code only ever calls a **`flash` / `flash-lite`** model (never `pro`, which isn't free).
- [ ] Key is in the **`GEMINI_API_KEY`** environment variable, **not** written into any script.
- [ ] We send only the **question text** to the API, never the dataset.

---

*Sources (verify before launch):*
- *[Free Gemini API key, no billing required (MakeUseOf)](https://www.makeuseof.com/you-can-get-free-gemini-api-key-right-now-no-billing-required/)*
- *[Gemini API free tier: limits & billing rules (AI Free API)](https://www.aifreeapi.com/en/posts/google-gemini-api-free-tier)*
- *[Gemini free tier guide (PE Collective)](https://pecollective.com/tools/gemini-free-tier-guide/)*
