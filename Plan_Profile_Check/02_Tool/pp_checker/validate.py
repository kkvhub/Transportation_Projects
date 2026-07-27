"""
validate.py — plan<->profile cross-checks (the checks that caught the stale
HIP chainages on HP-1). Run after parse.assemble().
"""
from __future__ import annotations
import math


def cross_checks(model, tol_plateau=4.0, tol_tangent=2.0):
    issues, confirms = [], []
    curves = [c for c in model.get("curves", []) if c.get("hip_ch") and c.get("Ts")]

    # 1. tangent between consecutive curves: coordinates vs chainage arithmetic
    for a, b in zip(curves, curves[1:]):
        if not all(a.get(k) for k in ("E", "N")) or not all(b.get(k) for k in ("E", "N")):
            continue
        dist = math.hypot(b["E"] - a["E"], b["N"] - a["N"])
        t_coord = dist - a["Ts"] - b["Ts"]
        st_a = a["hip_ch"] - a["Ts"] + 2 * (a.get("Ls") or 0) + (a.get("Lc") or 0)
        ts_b = b["hip_ch"] - b["Ts"]
        t_chain = ts_b - st_a
        d = t_chain - t_coord
        msg = (f"tangent {_ch(a['hip_ch'])}->{_ch(b['hip_ch'])}: "
               f"coords {t_coord:.2f} m vs chainage {t_chain:.2f} m (d={d:+.2f} m)")
        (issues if abs(d) > tol_tangent else confirms).append(
            {"check": "HIP chainage closure", "detail": msg,
             "severity": "ADVISORY" if abs(d) > tol_tangent else "OK"})

    # 2. superelevation plateau position/length vs curve SC-CS
    plateaus = []
    for sh in model.get("sheets", []):
        plateaus += sh.get("se_plateaus", [])
    for c in curves:
        if not all(c.get(k) for k in ("Ls", "Lc")):
            continue
        sc = c["hip_ch"] - c["Ts"] + c["Ls"]
        cs = sc + c["Lc"]
        best = None
        for p0, p1 in plateaus:
            if best is None or abs(p0 - sc) < abs(best[0] - sc):
                best = (p0, p1)
        if not best:
            continue
        d0, dl = best[0] - sc, (best[1] - best[0]) - c["Lc"]
        ok_len = abs(dl) < tol_plateau
        ok_pos = abs(d0) < tol_plateau
        msg = (f"curve {_ch(c['hip_ch'])}: e-plateau {best[0]:.0f}-{best[1]:.0f} "
               f"vs SC-CS {sc:.0f}-{cs:.0f} (offset {d0:+.1f} m, len diff {dl:+.1f} m)")
        sev = "OK" if (ok_len and ok_pos) else ("ADVISORY" if ok_len else "REVIEW")
        (confirms if sev == "OK" else issues).append(
            {"check": "SE plateau vs plan curve", "detail": msg, "severity": sev,
             "note": "" if sev == "OK" else
             "plateau length matches Lc but position offset — printed HIP CH likely stale"
             if ok_len else "plateau length disagrees with Lc — check extraction"})

    # 3. internal curve math already computed in parse (surface here)
    for c in model.get("curves", []):
        if c.get("math_ok") is True:
            confirms.append({"check": "curve internal math",
                             "detail": f"{_ch(c.get('hip_ch'))}: Lc/Ts consistent "
                                       f"with delta/R/Ls", "severity": "OK"})
        elif c.get("math_ok") is False:
            issues.append({"check": "curve internal math",
                           "detail": f"{_ch(c.get('hip_ch'))}: recomputed Lc "
                                     f"{c.get('lc_calc')} / Ts {c.get('ts_calc')} "
                                     f"disagree with table", "severity": "REVIEW"})
    # 4. structures: leader-table chainage vs drawn symbol / bridge span
    for s in model.get("structures", []):
        ch, sym = s.get("chainage"), s.get("symbol_ch")
        label = s.get("str_no") or "structure"
        if ch and sym is not None:
            d = sym - ch
            msg = (f"{label} at {_ch(ch)}: drawn symbol at {_ch(sym)} "
                   f"(offset {d:+.1f} m)")
            (confirms if abs(d) <= 6 else issues).append(
                {"check": "structure position", "detail": msg,
                 "severity": "OK" if abs(d) <= 6 else "REVIEW"})
        elif ch:
            confirms.append({"check": "structure position",
                             "detail": f"{label} at {_ch(ch)}: no separate symbol "
                                       f"detected (existing/minor structure)",
                             "severity": "OK"})
        if s.get("span_m") and s.get("drawn_span_m"):
            d = s["drawn_span_m"] - s["span_m"]
            msg = (f"{label}: drawn span {s['drawn_span_m']} m vs proposed size "
                   f"{s['span_m']} m (diff {d:+.1f} m)")
            (confirms if abs(d) <= 2 else issues).append(
                {"check": "bridge span vs size", "detail": msg,
                 "severity": "OK" if abs(d) <= 2 else "REVIEW"})
    return {"issues": issues, "confirms": confirms}


def _ch(ch):
    if ch is None:
        return "?"
    return f"{int(ch // 1000)}+{ch % 1000:.3f}"
