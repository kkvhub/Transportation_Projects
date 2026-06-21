"""
llm_client.py
=============
PHASE 1 — STEP 4: A tiny, billing-safe Gemini client used to (a) confirm the API
key works and (b) act as the single place the rest of the app talks to Gemini.

WHY THIS FILE EXISTS
--------------------
We keep ALL Gemini calls behind one small function (`ask_gemini`). That means:
  * the API key is read from an environment variable in exactly one place
    (never hard-coded in source that could be shared or committed), and
  * the model name is pinned to a FREE-TIER 'flash' model in one place, so we
    can never accidentally call a billable 'pro' model from scattered code.

See docs/gemini_setup_guide.md for how to get a free key without enabling billing.

SETUP (once):
    pip install google-genai
    setx GEMINI_API_KEY "AIza...your_key..."     # Windows; reopen the terminal

TEST:
    python src/llm_client.py
"""

import os

# Pinned to a free-tier model on purpose. Do NOT change this to a 'pro' model:
# pro models are not on the free tier and would start incurring charges.
FREE_TIER_MODEL = "gemini-2.0-flash"


def get_client():
    """Create a Gemini client using the key from the environment.

    Reading the key from GEMINI_API_KEY (not from a file) keeps the secret out
    of our source code. Raises a clear error if the key isn't set so the failure
    is obvious during setup.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. See docs/gemini_setup_guide.md "
            "(set it with: setx GEMINI_API_KEY \"AIza...\" then reopen the terminal)."
        )
    # Imported here (not at top) so the rest of the project can import this module
    # even before the google-genai package is installed.
    from google import genai
    return genai.Client(api_key=api_key)


def ask_gemini(prompt, model=FREE_TIER_MODEL):
    """Send a single prompt to Gemini and return the plain-text reply.

    This is deliberately minimal. In Phase 3 we'll extend it to request a
    STRUCTURED (JSON) reply that maps the user's question to a query-engine call,
    but the key-handling and model-pinning logic stays exactly the same.
    """
    client = get_client()
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


if __name__ == "__main__":
    # A one-line smoke test: proves the key is valid and the free-tier model
    # responds. If this prints a short greeting, your setup is correct.
    try:
        print("Gemini says:", ask_gemini("Reply with exactly: setup works"))
    except Exception as e:
        print("Setup not complete yet ->", e)
