"""
nl_router.py
============
PHASE 3 (reworked in Phase 5): the natural-language layer.

WHAT THIS DOES
--------------
Turns a user's sentence into a composable "spec" (see analysis.py) and runs it:

    question -> PARSE into spec {metric, school, radius, filters, ...}
             -> analysis.analyze(spec) -> standard result dict

The spec model lets us answer COMPOSITIONAL questions the old one-action design
couldn't, e.g. "what time of day do most crashes happen within 0.25 mi of
Holdeman Elementary" (metric=peak_hour + school filter) or "crashes near Holdeman
with KSI=1" (metric=count + school + KSI filter).

GEMINI + RULE FALLBACK
----------------------
parse_with_gemini() asks Gemini Flash to emit the spec JSON; parse_spec() is a
deterministic keyword parser used when there's no API key or Gemini fails, so the
bot always works offline. The LLM only fills in the spec -- numbers still come
from pandas in analysis.py, so answers can't be hallucinated.

Run a demo:  python src/nl_router.py
"""

import json
import re

import analysis as an

# Cities other than Tempe / unsupported asks -> out of scope.
OTHER_CITIES = ["phoenix", "mesa", "chandler", "scottsdale", "gilbert",
                "glendale", "peoria", "avondale"]
OOS_WORDS = ["predict", "forecast", "tomorrow", "next year", "cost", "dollar",
             "make and model", "license plate", "vehicle model"]

# Phrase -> (field, value) equality filters for the rule parser.
KEYWORD_FILTERS = {
    r"\brain|raining|rainy|wet weather\b": ("Weather", "Rain"),
    r"\bsnow\b": ("Weather", "Snow"),
    r"\bfog|foggy\b": ("Weather", "Fog Smog Smoke"),
    r"\balcohol|drunk|dui|impaired\b": ("AlcoholUse_Drv1", "Alcohol"),
    r"\bdrug|drugs\b": ("DrugUse_Drv1", "Drugs"),
    r"\bpedestrian|on foot\b": ("Unittype_One", "Pedestrian"),
    r"\bcyclist|bicycle|bike|pedalcyclist\b": ("Unittype_One", "Pedalcyclist"),
    r"\brear[- ]?end\b": ("Collisionmanner", "Rear End"),
    r"\bhead[- ]?on\b": ("Collisionmanner", "Head On"),
    r"\bfatal|deadly|death\b": ("Injuryseverity", "Fatal"),
}

# "break down by X" field synonyms.
COUNT_BY_FIELDS = {
    "weather": "Weather", "lighting": "Lightcondition", "light": "Lightcondition",
    "collision": "Collisionmanner", "manner": "Collisionmanner",
    "severity": "Injuryseverity", "junction": "JunctionRelation",
    "year": "Year", "month": "crash_month", "hour": "crash_hour",
    "day": "crash_dow", "gender": "Gender_Drv1", "surface": "SurfaceCondition",
}

# Instruction for Gemini: emit the same spec the rule parser produces.
SYSTEM_PROMPT = """You convert a question about Tempe, Arizona crash data (2012-2025)
into ONE JSON "spec" describing the analysis to run. Output JSON ONLY.

Spec fields:
{
 "metric": one of [count, peak_hour, peak_day, peak_month, busiest_year,
                   school_ranking, breakdown],
 "by_field": <field> (only when metric=breakdown),
 "school": <school name mentioned, else null>,
 "radius_mi": <number, default 0.25>,
 "view": "buffer" or "nearest" (default "buffer"),
 "filters": { field: value }  // equality filters
            allowed fields: Year, crash_hour, is_dark, is_weekend, KSI, Weather,
            SurfaceCondition, Collisionmanner, JunctionRelation, Unittype_One,
            AlcoholUse_Drv1, DrugUse_Drv1, Gender_Drv1, Injuryseverity,
 "street": <street name substring or null>,
 "rank_metric": "crashes" or "ksi"  // for school_ranking ordering
 "in_scope": true/false
}

Rules:
- Combine freely: e.g. "peak time near School X with KSI=1" ->
  {"metric":"peak_hour","school":"School X","filters":{"KSI":1},...}.
- "KSI", "serious", "killed or seriously injured" -> filters {"KSI":1}
  (but for "which school has most KSI" use metric school_ranking, rank_metric "ksi").
- Data is TEMPE ONLY, 2012-2025. Other cities, future dates, predictions, cost,
  vehicle make/model => {"in_scope": false}.
Return ONLY the JSON."""


# ---------------------------------------------------------------------------
def _extract_radius(q, default=0.25):
    """Pull a radius in miles from text like '0.25 mi', 'quarter mile', 'half mile'."""
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(mi|mile|miles)", q)
    if m:
        return float(m.group(1))
    if "quarter mile" in q:
        return 0.25
    if "half mile" in q or "half a mile" in q:
        return 0.5
    return default


def _blank_spec():
    return {"metric": None, "by_field": None, "school": None, "radius_mi": 0.25,
            "view": "buffer", "filters": {}, "street": None,
            "rank_metric": "crashes", "in_scope": True}


def parse_spec(question):
    """Deterministic keyword parser -> spec dict (same shape Gemini emits)."""
    q = (question or "").lower().strip()
    spec = _blank_spec()
    if len(q) < 3:
        spec["in_scope"] = False
        return spec
    if any(c in q for c in OTHER_CITIES) or any(w in q for w in OOS_WORDS):
        spec["in_scope"] = False
        return spec

    spec["view"] = "nearest" if ("nearest" in q or "assigned" in q) else "buffer"
    spec["radius_mi"] = _extract_radius(q)

    # School (matched against the real school list, so "Holdeman" resolves).
    school = an.find_school(q)
    if school:
        spec["school"] = school

    # KSI mention (NOT 'fatal' -- that's handled as an Injuryseverity filter).
    ksi_zero = bool(re.search(r"ksi\s*=?\s*0", q)) or "non-ksi" in q
    ksi_pos = ("ksi" in q or "serious" in q or "killed or seriously" in q) and not ksi_zero

    # Attribute filters.
    for pattern, (field, value) in KEYWORD_FILTERS.items():
        if re.search(pattern, q):
            spec["filters"][field] = value
    ym = re.search(r"\b(20[0-2][0-9])\b", q)
    if ym:
        spec["filters"]["Year"] = int(ym.group(1))
    if "at night" in q or "after dark" in q or "in the dark" in q or "night" in q:
        spec["filters"]["is_dark"] = 1
    if "weekend" in q:
        spec["filters"]["is_weekend"] = 1
    sm = re.search(r"(?<![-\w])on\s+([a-z0-9 ]+?\s+(?:rd|road|dr|drive|st|street|ave|"
                   r"avenue|blvd|way|pkwy))\b", q)
    if sm:
        spec["street"] = sm.group(1).strip()

    # Metric.
    metric = None
    if any(w in q for w in ["time of day", "what time", "which hour",
                            "what hour", "peak hour", "busiest hour"]):
        metric = "peak_hour"
    elif "day of the week" in q or "which day" in q or "what day" in q:
        metric = "peak_day"
    elif "month" in q and any(w in q for w in ["most", "which", "worst", "peak"]):
        metric = "peak_month"
    elif "year" in q and any(w in q for w in ["most", "which", "worst", "highest", "peak"]):
        metric = "busiest_year"
    else:
        bd = re.search(r"(?:break ?down by|grouped by|broken down by|by)\s+([a-z ]+)", q)
        is_bd = ("breakdown" in q or "break down" in q or "grouped by" in q
                 or re.search(r"\bby (weather|lighting|severity|collision|junction|"
                              r"gender|surface|day|hour|month|year)\b", q))
        if is_bd and bd:
            for word, field in COUNT_BY_FIELDS.items():
                if word in bd.group(1):
                    metric, spec["by_field"] = "breakdown", field
                    break
        if metric is None and spec["school"] is None and "school" in q and \
                any(w in q for w in ["most", "top", "highest", "rank", "ranking", "worst"]):
            metric = "school_ranking"
            spec["rank_metric"] = "ksi" if ksi_pos else "crashes"

    if metric is None:
        # Fall back to a plain count whenever we have something to filter on.
        if spec["school"] or spec["filters"] or spec["street"] or ksi_pos:
            metric = "count"

    # KSI as a FILTER for everything except ranking (where it sets ordering).
    if metric != "school_ranking":
        if ksi_zero:
            spec["filters"]["KSI"] = 0
        elif ksi_pos:
            spec["filters"]["KSI"] = 1

    spec["metric"] = metric
    if metric is None:
        spec["in_scope"] = False
    return spec


def parse_with_gemini(question):
    """Ask Gemini to return the spec JSON. Returns a dict, or None on failure."""
    import llm_client
    raw = llm_client.ask_gemini(SYSTEM_PROMPT + f'\n\nQuestion: "{question}"\nJSON:')
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        spec = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    # Fill any missing keys with defaults so analyze() is robust.
    base = _blank_spec()
    base.update({k: spec.get(k, base[k]) for k in base})
    base["filters"] = spec.get("filters") or {}
    return base


def answer(question, prefer_llm=True, default_view="buffer", default_radius=None):
    """Main entry point: question text -> result dict (+ parsed spec & engine).

    UI defaults (view / radius) are applied only when the user didn't state them
    explicitly in the question.
    """
    import os
    if not question or len(question.strip()) < 3:
        r = an_no_data("Please ask a question about Tempe crashes.")
        r["intent"] = _blank_spec()
        r["engine"] = "guard"
        return r

    spec, engine = None, "rules"
    if prefer_llm and os.environ.get("GEMINI_API_KEY"):
        try:
            spec = parse_with_gemini(question)
            engine = "gemini"
        except Exception:
            spec = None
    if spec is None:
        spec = parse_spec(question)
        engine = "rules"

    ql = question.lower()
    if "nearest" not in ql and "assigned" not in ql:
        spec["view"] = default_view
    if (default_radius is not None
            and not re.search(r"\b\d*\.?\d+\s*(mi|mile)", ql)):
        spec["radius_mi"] = float(default_radius)

    result = an.analyze(spec)
    result["intent"] = spec
    result["engine"] = engine
    return result


def an_no_data(msg):
    """Tiny shim so the empty-question guard doesn't need to import query_engine."""
    return {"ok": False, "answer": msg, "rows": __import__("pandas").DataFrame(),
            "map": False, "meta": {}}


def _demo():
    qs = [
        "Which year had the most crashes?",
        "What time of day is most prevalent for accidents?",
        "number of crashes in 0.25mi of Holdeman Elementary School with KSI = 1",
        "at what time of day most crashes happen in 0.25mi of Holdeman Elementary School",
        "Which school has the most serious crashes within 0.25 mi?",
        "How many alcohol related crashes at night in 2023?",
        "Break down crashes by weather",
        "How many crashes in Phoenix?",
    ]
    for q in qs:
        r = answer(q, prefer_llm=False)
        tag = "OK " if r["ok"] else "NO "
        print(f"{tag}[{r['intent'].get('metric')}] {q}\n     -> {r['answer']}")


if __name__ == "__main__":
    _demo()
