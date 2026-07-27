"""
report.py — HTML report generation.
"""
from __future__ import annotations
import datetime
import html
import json

BADGE = {"PASS": "#1a7f37", "ADVISORY": "#b58900", "HARD FAIL": "#c0392b",
         "INFO": "#2471a3", "OK": "#1a7f37", "REVIEW": "#c0392b"}

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;max-width:1400px}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f4f4f4}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px}
.cell-main{display:block;margin-bottom:2px}.cell-sub{display:block;color:#666;font-size:11px}
.compact td{min-width:105px}.compact td:first-child{min-width:190px;font-weight:600}
.sum{font-size:15px;margin:12px 0}
.meta{color:#666;font-size:12px}
"""


def _badge(v):
    return f'<span class="badge" style="background:{BADGE.get(v, "#666")}">{html.escape(v)}</span>'


def _fmt_ch(v):
    if v is None:
        return "-"
    return "%d+%03.0f" % (int(v // 1000), v % 1000)


def _condensed_compliance(results):
    """Pivot rule results so each element appears once."""
    if not results:
        return "<p class=\"meta\">No checks available.</p>"
    preferred = [
        "design speed", "min radius", "superelevation cap", "e vs V^2/225R",
        "spiral length Ls", "extra widening", "signage", "gradient",
        "min VC length", "K valley (headlight)", "K summit (SSD)",
        "K summit (ISD)", "K (type unknown - checked as valley)",
        "10 s same-direction tangent", "grade-change spacing",
        "10 s vertical broken-back tangent",
        "OCR consistency",
    ]
    checks = []
    for chk in preferred:
        if any(r["check"] == chk for r in results):
            checks.append(chk)
    for r in results:
        if r["check"] not in checks:
            checks.append(r["check"])
    grouped = {}
    order = []
    for r in results:
        element = r["element"]
        if element not in grouped:
            grouped[element] = {}
            order.append(element)
        grouped[element][r["check"]] = r
    head = "<tr><th>Element</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in checks) + "<th>Notes</th></tr>"
    rows = []
    for element in order:
        notes = []
        cells = [f"<td>{html.escape(element)}</td>"]
        for chk in checks:
            r = grouped[element].get(chk)
            if not r:
                cells.append("<td>-</td>")
                continue
            if r.get("note"):
                notes.append(f"{chk}: {r['note']}")
            detail = f"{r['provided']} / {r['limit']}" if r.get("limit") not in ("", "-") else r.get("provided", "")
            cells.append(
                "<td><span class=\"cell-main\">%s</span><span class=\"cell-sub\">%s</span></td>" %
                (_badge(r["verdict"]), html.escape(str(detail)))
            )
        cells.append("<td>%s</td>" % html.escape("; ".join(notes) if notes else "-"))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table class=\"compact\">" + head + "".join(rows) + "</table>"


def _presence_value(s, key, default=None):
    v = s.get(key)
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, str) and v.strip():
        return v
    return default


def _structure_presence(s):
    profile = _presence_value(s, "profile_present", "Yes")
    plan = _presence_value(s, "plan_present")
    if plan is None:
        plan = "Not checked"
    if s.get("consistency_status"):
        return plan, profile, str(s["consistency_status"])
    if plan == "Yes" and profile == "Yes":
        status = "Present in both"
    elif plan == "No" or profile == "No":
        status = "Mismatch"
    else:
        status = "Plan not verified"
    return plan, profile, status


def _is_horizontal_result(r):
    return str(r.get("element", "")).startswith("HC-")


def _is_vertical_result(r):
    el = str(r.get("element", ""))
    return el.startswith("VC") or el.startswith("G@")


def _vertical_curve_rows(model):
    rows = []
    for vc in model.get("vertical_curves", []):
        sheets = ",".join(str(s) for s in vc.get("sheets", []))
        cells = [
            vc.get("id", "-"),
            _fmt_ch(vc.get("ch_mid")),
            vc.get("type", "-"),
            vc.get("G_in", "-"),
            vc.get("G_out", "-"),
            vc.get("algebraic_difference_pct", "-"),
            vc.get("K") if vc.get("K") is not None else "-",
            vc.get("L") if vc.get("L") is not None else "-",
            _fmt_ch(vc.get("pvi_chainage")),
            _fmt_ch(vc.get("pvc_chainage")),
            _fmt_ch(vc.get("pvt_chainage")),
            "Yes" if vc.get("continued") else "No",
            sheets or "-",
        ]
        rows.append("<tr>" + "".join("<td>%s</td>" % html.escape(str(c)) for c in cells) + "</tr>")
    return "".join(rows)


def _vertical_gradient_rows(model):
    rows = []
    for segment in model.get("vertical_gradient_segments", []):
        cells = [segment.get("id", "-"), _fmt_ch(segment.get("ch_mid")),
                 segment.get("G", "-"), segment.get("L", "-"),
                 segment.get("sheet", "-")]
        rows.append("<tr>" + "".join("<td>%s</td>" % html.escape(str(c)) for c in cells) + "</tr>")
    return "".join(rows)


def to_html(model, rule_result, xcheck):
    n = rule_result["summary"]
    h_results = [r for r in rule_result["results"] if _is_horizontal_result(r)]
    v_results = [r for r in rule_result["results"] if _is_vertical_result(r)]
    other_results = [r for r in rule_result["results"] if r not in h_results and r not in v_results]
    h_compliance_table = _condensed_compliance(h_results)
    v_compliance_table = _condensed_compliance(v_results)
    other_compliance_table = _condensed_compliance(other_results) if other_results else ""
    rows_c = "".join(
        f"<tr><td>HC-{i}</td><td>{c.get('hip_ch','?')}</td><td>{c.get('V','')}</td>"
        f"<td>{c.get('R','')}</td><td>{c.get('delta_deg','') and round(c['delta_deg'],4)}</td>"
        f"<td>{c.get('Ts','')}</td><td>{c.get('Lc','')}</td><td>{c.get('Ls','')}</td>"
        f"<td>{c.get('e','')}</td><td>{html.escape(str(c.get('direction') or '-'))}</td>"
        f"<td>{html.escape(str(c.get('direction_confidence') or '-'))}</td>"
        f"<td>{_fmt_ch(c.get('curve_start_ch'))}</td><td>{_fmt_ch(c.get('curve_end_ch'))}</td>"
        f"<td>{'yes' if c.get('math_ok') else 'CHECK'}</td></tr>"
        for i, c in enumerate(model["curves"], 1))
    def _srow(s):
        plan_present, profile_present, consistency = _structure_presence(s)
        cells = [str(s.get("str_no", "") or "-"),
                 _fmt_ch(s.get("chainage")),
                 plan_present,
                 profile_present,
                 consistency,
                 str(s.get("existing") or "-"),
                 str(s.get("proposed_type") or ("Bridge" if s.get("is_bridge") else "-")),
                 str(s.get("proposed_size") or "-"),
                 str(s.get("proposal") or "-"),
                 _fmt_ch(s.get("symbol_ch")),
                 str(s.get("drawn_span_m") or "-")]
        return "<tr>" + "".join("<td>%s</td>" % html.escape(c) for c in cells) + "</tr>"

    rows_s = "".join(_srow(s) for s in model.get("structures", []))
    structures_html = ""
    if model.get("structures"):
        structures_html = ("<h2>Structures (culverts / bridges)</h2>"
                           "<table><tr><th>Str. No</th><th>Chainage</th>"
                           "<th>Plan present</th><th>Profile present</th><th>Consistency status</th>"
                           "<th>Existing</th>"
                           "<th>Proposed type</th><th>Proposed size</th><th>Proposal</th>"
                           "<th>Symbol drawn at</th><th>Drawn span (m)</th></tr>"
                           + rows_s + "</table>")
    vertical_html = ""
    if model.get("vertical_curves"):
        vertical_html = (
            "<h2>Extracted vertical curves</h2>"
            "<table><tr><th>#</th><th>Approx. CH</th><th>Type</th>"
            "<th>In gradient (%)</th><th>Out gradient (%)</th><th>Algebraic diff (%)</th>"
            "<th>K</th><th>L (m)</th><th>Approx. PVI</th><th>PVC</th><th>PVT</th>"
            "<th>Continued on next sheet</th><th>Sheet index</th></tr>"
            + _vertical_curve_rows(model) + "</table>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>P&P Compliance Report</title><style>{CSS}</style></head><body>
<h1>Plan &amp; Profile Compliance Report</h1>
<div class="meta">File: {html.escape(model['file'])} &middot;
Standard: {rule_result['standard']} ({rule_result['road_class']},
{rule_result['terrain']}) &middot; Generated {datetime.date.today().isoformat()}</div>
<div class="sum">{n['PASS']} PASS &middot; {n['ADVISORY']} ADVISORY &middot;
{n['HARD FAIL']} HARD FAIL &middot; {n['INFO']} INFO</div>
<h2>Extracted horizontal curves</h2>
<table><tr><th>#</th><th>HIP CH (m)</th><th>V</th><th>R</th><th>&Delta;&deg;</th>
<th>Ts</th><th>Lc</th><th>Ls</th><th>e%</th><th>Direction sign</th>
<th>Direction confidence</th><th>Curve start</th><th>Curve end</th><th>math ok</th></tr>{rows_c}</table>
{vertical_html}
<h2>Horizontal curve compliance</h2>
{h_compliance_table}
<h2>Vertical curve / profile compliance</h2>
{v_compliance_table}
{('<h2>Other compliance checks</h2>' + other_compliance_table) if other_compliance_table else ''}
{structures_html}
<p class="meta">Deterministic extraction with OCR of stroke text; values flagged
"CHECK" failed internal-math verification and need manual review.
Generated by pp_checker.</p></body></html>"""


def to_json(model, rule_result, xcheck):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "ocr_rows"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float):
            return round(o, 4)
        return o
    return json.dumps({"model": clean(model), "rules": rule_result,
                       "cross_checks": xcheck}, indent=1, default=str)
