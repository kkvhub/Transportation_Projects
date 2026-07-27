"""
extract.py — deterministic (non-OCR) extraction from vector P&P PDFs.

Handles: page classification, rotated-word extraction, chainage mapping,
superelevation ramp/tick geometry, structure symbols, level rows.
All coordinates are in "display" space (page rotated to landscape).
"""
from __future__ import annotations
import math
import re
import numpy as np
import fitz


class Page:
    """A P&P sheet rotated to landscape with cached words/drawings."""

    def __init__(self, doc: fitz.Document, index: int):
        self.doc, self.index = doc, index
        self.page = doc[index]
        # rotate portrait pages so drawing reads horizontally
        if self.page.rect.width < self.page.rect.height:
            self.page.set_rotation((self.page.rotation + 90) % 360)
        self.M = self.page.rotation_matrix
        self.rect = fitz.Rect(0, 0, self.page.rect.height, self.page.rect.width) \
            if self.page.rotation in (90, 270) else fitz.Rect(self.page.rect)
        self._words = None
        self._drawings = None

    @property
    def words(self):
        if self._words is None:
            out = []
            for w in self.page.get_text("words"):
                r = fitz.Rect(w[:4]) * self.M
                r.normalize()
                out.append((r, w[4]))
            self._words = out
        return self._words

    @property
    def drawings(self):
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings

    def lines(self):
        """All straight segments in display coords."""
        for path in self.drawings:
            for it in path["items"]:
                if it[0] == "l":
                    a, b = it[1] * self.M, it[2] * self.M
                    yield a, b, path

    def pixmap(self, clip=None, zoom=8):
        return self.page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)


# ---------------------------------------------------------------- chainage

CH_RE = re.compile(r"^(\d{1,3})\+(\d{1,3}(?:\.\d+)?)$")
CHAINAGE_EQ_RE = re.compile(r"^Chainage=([-+]?\d+(?:\.\d+)?)m$", re.I)


def chainage_labels(pg: Page, fallback_ocr=True):
    """(value_m, x_center, y0) for every km+m label in the text layer.

    Some P&P exports draw chainage digits as vector strokes with no
    embedded text at all (unlike HP-1, where they're real text). When the
    text layer yields too few labels to be a profile sheet, fall back to
    OCR-scanning the sheet for a dense NNN+NNN chainage band.
    """
    out = []
    for r, t in pg.words:
        m = CH_RE.match(t)
        if m:
            metres = float(m.group(2))
            if 0 <= metres < 1000:
                out.append((int(m.group(1)) * 1000 + metres,
                            (r.x0 + r.x1) / 2.0, r.y0))
            continue
        m = CHAINAGE_EQ_RE.match(t)
        if m:
            out.append((float(m.group(1)),
                        (r.x0 + r.x1) / 2.0, r.y0))
    if len(out) < 8:
        row = plain_meter_chainage_row(pg)
        if len(row) > len(out):
            out = row
    # Several structure callouts can contain valid chainages but are spread
    # over the chart. They must not suppress OCR of the dense bottom axis.
    dense_count = 0
    for _, _, y in out:
        dense_count = max(dense_count, sum(1 for _, _, yy in out if abs(yy - y) < 15))
    if dense_count < 5 and fallback_ocr:
        from . import ocr as _ocr
        found = _ocr.find_chainage_band_ocr(pg)
        if found:
            hits, y0, y1 = found
            ymid = (y0 + y1) / 2.0
            out = [(v, x, ymid) for v, x in hits]
    return out


def plain_meter_chainage_row(pg: Page):
    """Dense profile chainage row labelled as 0, 25, 50 ... 1+000.

    Some sheets, especially compact profile sheets, use plain metre labels
    along the bottom band instead of km+m labels. This detects the densest
    numeric row with mostly increasing values and returns it in the same
    format as chainage_labels().
    """
    hits = []
    for r, t in pg.words:
        txt = t.strip()
        val = None
        m = CH_RE.match(txt)
        if m:
            val = int(m.group(1)) * 1000 + float(m.group(2))
        elif re.fullmatch(r"\d{1,4}", txt):
            val = float(txt)
        if val is None or not (0 <= val <= 200000):
            continue
        # Avoid plan callouts high in the drawing; dense profile rows sit
        # in the lower annotation/table half on the sample sheets.
        if r.y0 < pg.rect.height * 0.45:
            continue
        hits.append((val, (r.x0 + r.x1) / 2.0, r.y0))
    if len(hits) < 8:
        return []
    clusters = []
    for hit in sorted(hits, key=lambda h: h[2]):
        for cl in clusters:
            if abs(hit[2] - cl[0][2]) <= 8:
                cl.append(hit)
                break
        else:
            clusters.append([hit])
    best = []
    for cl in clusters:
        cl = sorted(cl, key=lambda h: h[1])
        if len(cl) < 8:
            continue
        vals = [h[0] for h in cl]
        inc = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        span = vals[-1] - vals[0]
        if inc >= len(vals) * 0.75 and span >= 100 and len(cl) > len(best):
            best = cl
    return best


class ChainageAxis:
    """Linear x->chainage map built from the profile chainage band."""

    def __init__(self, pg: Page):
        labels = chainage_labels(pg)
        if len(labels) < 5:
            raise ValueError("not enough chainage labels — not a profile sheet?")
        ladder, table_x0, table_x1 = find_band_row_ladder(pg)
        # Combined plan/profile sheets can have many valid chainages along
        # the curved plan alignment and a stroke-drawn profile axis. Locate
        # the wide lower profile-table divider and map its data span to the
        # dominant one-kilometre range.
        km_prefixes = [int(v // 1000) for v, _, _ in labels if v >= 1000]
        wide_lower = []
        for a, b, _ in pg.lines():
            if abs(a.y - b.y) >= 0.4:
                continue
            x0, x1 = min(a.x, b.x), max(a.x, b.x)
            y = (a.y + b.y) / 2.0
            if (x1 - x0 > pg.rect.width * 0.5 and pg.rect.height * 0.55 < y < pg.rect.height * 0.8
                    and x0 > pg.rect.width * 0.1 and x1 < pg.rect.width * 0.9):
                wide_lower.append((y, x0, x1))
        if km_prefixes and wide_lower:
            km = max(set(km_prefixes), key=km_prefixes.count)
            if km_prefixes.count(km) >= 8:
                divider_y, data_x0, data_x1 = min(wide_lower, key=lambda p: p[0])
                scale = 1000.0 / (data_x1 - data_x0)
                self.coef = np.array([scale, km * 1000.0 - data_x0 * scale])
                self.resid = 0.0
                self.band_y = float(divider_y + 30.0)
                self.ch_min, self.ch_max = float(km * 1000), float((km + 1) * 1000)
                self.axis_source = "combined-sheet profile grid"
                self.profile_vertical_bottom = float(divider_y)
                return
        # Some CAD exports draw the dense rotated chainage row as vector
        # strokes. If OCR then sees only scattered structure chainages, a
        # strong table ladder and repeated km prefix recover the sheet axis.
        y_span = max(y for _, _, y in labels) - min(y for _, _, y in labels)
        if (len(ladder) >= 8 and table_x0 is not None and table_x1 is not None
                and y_span > 30 and km_prefixes):
            km = max(set(km_prefixes), key=km_prefixes.count)
            if km_prefixes.count(km) >= 3:
                scale = 1000.0 / (table_x1 - table_x0)
                self.coef = np.array([scale, km * 1000.0 - table_x0 * scale])
                self.resid = 0.0
                self.band_y = float((ladder[-2] + ladder[-1]) / 2.0)
                self.ch_min, self.ch_max = float(km * 1000), float((km + 1) * 1000)
                self.axis_source = "profile grid and repeated km prefix"
                return
        # choose the densest y-cluster of labels = the chainage band
        ys = sorted(l[2] for l in labels)
        best, cur = [], []
        for y in ys:
            if cur and y - cur[-1] > 15:
                if len(cur) > len(best):
                    best = cur
                cur = []
            cur.append(y)
        if len(cur) > len(best):
            best = cur
        band_y = float(np.median(best))
        pts = [(v, x) for v, x, y in labels if abs(y - band_y) < 15]
        xs = np.array([p[1] for p in pts])
        vs = np.array([float(p[0]) for p in pts])
        coef = np.polyfit(xs, vs, 1)
        # drop outliers (occasional OCR misreads on stroke-font sheets) and
        # refit once — a handful of garbled labels shouldn't skew the axis
        resid = np.abs(vs - np.polyval(coef, xs))
        keep = resid < max(10.0, 3 * np.median(resid) if np.median(resid) > 0 else 10.0)
        if keep.sum() >= 5 and keep.sum() < len(xs):
            xs, vs = xs[keep], vs[keep]
            coef = np.polyfit(xs, vs, 1)
        self.coef = coef
        self.resid = float(np.abs(vs - np.polyval(self.coef, xs)).max())
        self.band_y = float(band_y)
        self.ch_min, self.ch_max = float(vs.min()), float(vs.max())
        span = self.ch_max - self.ch_min
        km_base = math.floor(self.ch_min / 1000.0) * 1000.0
        if 500 <= span < 990 and 0 <= self.ch_min - km_base <= 60:
            self.ch_min = km_base
            self.ch_max = km_base + 1000.0

    def ch(self, x: float) -> float:
        return float(np.polyval(self.coef, x))

    def x(self, ch: float) -> float:
        return (ch - self.coef[1]) / self.coef[0]


def is_profile_sheet(pg: Page) -> bool:
    """Profile sheets carry a dense chainage band plus level columns.

    Text-layer sheets carry 20+ (checked against the stricter bound);
    OCR-fallback sheets (see chainage_labels) already require >= 8 hits
    internally, which is well above the handful of stray NNN+NNN-shaped
    callouts (HIP/structure chainages) a plan sheet might otherwise have.
    """
    labels = chainage_labels(pg, fallback_ocr=False)
    if len(labels) >= 8:
        return True
    words = [t.lower() for _, t in pg.words]
    chainage_hits = sum(1 for t in words if t.startswith("chainage="))
    level_hits = sum(1 for t in words if t.startswith("level="))
    if chainage_hits >= 2 and level_hits >= 2:
        return True
    # Only pay the OCR cost if vector/text hints suggest a profile band.
    ladder, _, _ = find_band_row_ladder(pg)
    # A long regular row ladder is itself a strong profile-table signature.
    if len(ladder) >= 8 and len(labels) >= 3:
        return True
    if len(ladder) >= 5:
        return len(chainage_labels(pg, fallback_ocr=True)) >= 8
    return False


def find_band_row_ladder(pg: Page, min_rows=5):
    """Locate the profile-band row boundaries via the repeated tick marks in
    the row-label column (left edge of the 'Proposed Road Level / Existing
    Ground / .../ Proposed Chainage' stack), plus the data-table's x-extent.

    Unlike find_band_rows(), this doesn't require full-width separator
    lines — some P&P exports only draw the row dividers across the narrow
    label column, not the full data width. Returns
    (row_ys, x0, x1) where row_ys is the sorted list of row boundaries
    (bottom-most gap is always the chainage row) and (x0, x1) bounds the
    data columns (excludes the label column and anything to its left, and
    the legend/title-block area to the right) — or ([], None, None).
    """
    from collections import defaultdict
    segs = defaultdict(list)
    for a, b, _ in pg.lines():
        if abs(a.y - b.y) < 0.4 and 20 < abs(a.x - b.x) < 120:
            key = (round(min(a.x, b.x), 0), round(max(a.x, b.x), 0))
            segs[key].append(round((a.y + b.y) / 2.0, 1))
    best, best_key = [], None
    for key, ys in segs.items():
        ys = sorted(set(ys))
        if len(ys) < min_rows:
            continue
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        med = sorted(diffs)[len(diffs) // 2]
        if med <= 0:
            continue
        if all(abs(d - med) < med * 0.3 for d in diffs) and len(ys) > len(best):
            best, best_key = ys, key
    if not best:
        return [], None, None
    label_x1 = best_key[1]
    # the data table's right edge = the longest horizontal line that starts
    # at the label column's right edge (a row divider spanning the chart)
    x1_candidates = [max(a.x, b.x) for a, b, _ in pg.lines()
                      if abs(a.y - b.y) < 0.4 and abs(min(a.x, b.x) - label_x1) < 3]
    table_x1 = max(x1_candidates) if x1_candidates else pg.rect.width
    return best, label_x1, table_x1


# ---------------------------------------------------------- band geometry

def find_band_rows(pg: Page, axis: ChainageAxis):
    """Locate the annotation band rows above the chainage band.

    Returns dict name -> (y0, y1) using the full-width horizontal separator
    lines that bound the data table at the sheet foot.
    """
    x_left, x_right = axis.x(axis.ch_min), axis.x(axis.ch_max)
    width = x_right - x_left
    seps = set()
    for a, b, _ in pg.lines():
        if abs(a.y - b.y) < 0.3 and abs(a.x - b.x) > 0.9 * width and a.y < axis.band_y + 20:
            seps.add(round((a.y + b.y) / 2.0, 1))
    seps = sorted(seps)
    ded = []
    for y in seps:
        if not ded or y - ded[-1] > 2.5:
            ded.append(y)
    return ded  # caller pairs rows with the labels column via OCR


def superelevation_ramps(pg: Page, axis: ChainageAxis, y0: float, y1: float):
    """Sloped segments inside the superelevation band -> e transitions."""
    ramps = []
    for a, b, _ in pg.lines():
        if y0 <= a.y <= y1 and y0 <= b.y <= y1:
            if a.x > b.x:
                a, b = b, a
            dx, dy = b.x - a.x, b.y - a.y
            if dx > 2 and abs(dy) > 0.5:
                ramps.append((axis.ch(a.x), axis.ch(b.x)))
    ramps.sort()
    ded = []
    for c0, c1 in ramps:
        if ded and abs(c0 - ded[-1][0]) < 1.5 and abs(c1 - ded[-1][1]) < 1.5:
            continue
        ded.append((c0, c1))
    return ded


def plateaus_from_ramps(ramps, min_gap=8.0):
    """Constant-e plateaus = gaps between a down-ramp end and next up-ramp start."""
    edges = sorted(set(round(c, 1) for r in ramps for c in r))
    plateaus = []
    for i in range(len(edges) - 1):
        gap = edges[i + 1] - edges[i]
        covered = any(c0 <= edges[i] + 0.5 and edges[i + 1] - 0.5 <= c1 for c0, c1 in ramps)
        if gap >= min_gap and not covered:
            plateaus.append((edges[i], edges[i + 1]))
    return plateaus


# ---------------------------------------------------------- structures

def structure_symbols(pg: Page, axis: ChainageAxis, y0=250, y1=470):
    """Filled culvert boxes + long gray verticals (bridges) in the plot area."""
    culverts, verticals = [], []
    xl, xr = axis.x(axis.ch_min), axis.x(axis.ch_max)
    for path in pg.drawings:
        if path["fill"] is not None:
            r = fitz.Rect(path["rect"]) * pg.M
            r.normalize()
            if 4 < r.width < 12 and 4 < r.height < 12 and y0 < r.y0 < y1 and xl < r.x0 < xr:
                culverts.append(axis.ch((r.x0 + r.x1) / 2.0))
    for a, b, path in pg.lines():
        col = path.get("color") or (0, 0, 0)
        gray = len(set(round(c, 2) for c in col)) == 1 and 0.2 < col[0] < 0.9
        if gray and abs(a.x - b.x) < 0.5 and abs(a.y - b.y) > 25 \
                and y0 < min(a.y, b.y) and max(a.y, b.y) < y1 and xl + 2 < a.x < xr - 2:
            verticals.append(axis.ch(a.x))
    culverts = _dedupe(sorted(culverts), 3.0)
    verticals = _dedupe(sorted(verticals), 2.0)
    return culverts, verticals


def _dedupe(vals, tol):
    out = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


# ---------------------------------------------------------- level rows

LEVEL_RE = re.compile(r"^\d{3,4}\.\d{2,3}$")


def level_rows(pg: Page, axis: ChainageAxis):
    """Group elevation words into horizontal rows (RL / EGL / L / R)."""
    pts = []
    for r, t in pg.words:
        if LEVEL_RE.match(t):
            pts.append(((r.y0 + r.y1) / 2.0, axis.ch((r.x0 + r.x1) / 2.0), float(t)))
    if not pts:
        return []
    pts.sort()
    rows, cur, last_y = [], [], None
    for y, ch, v in pts:
        if last_y is not None and y - last_y > 6:
            rows.append(cur)
            cur = []
        cur.append((ch, v))
        last_y = y
    rows.append(cur)
    return [sorted(r) for r in rows if len(r) >= 10]


# ---------------------------------------------------------- table grids

def find_table_grids(pg: Page, min_rows=8, col_range=(30, 70)):
    """Detect small bordered tables (HIP tables, structure leader boxes).

    Returns list of (hlines, vlines) grids ordered left-to-right.
    """
    hs, vs = [], []
    for a, b, _ in pg.lines():
        if abs(a.y - b.y) < 0.4 and 8 < abs(a.x - b.x) < 80:
            hs.append((min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2.0))
        if abs(a.x - b.x) < 0.4 and 3 < abs(a.y - b.y) < 90:
            vs.append(((a.x + b.x) / 2.0, min(a.y, b.y), max(a.y, b.y)))
    grids = []
    used = [False] * len(hs)
    hs.sort(key=lambda h: (round(h[0], 0), h[2]))
    for i, h in enumerate(hs):
        if used[i]:
            continue
        stack = [h]
        for j in range(i + 1, len(hs)):
            g = hs[j]
            if used[j]:
                continue
            if abs(g[0] - h[0]) < 3 and abs(g[1] - h[1]) < 3 and 3 < g[2] - stack[-1][2] < 10:
                stack.append(g)
                used[j] = True
        if len(stack) >= min_rows and col_range[0] < h[1] - h[0] < col_range[1]:
            x0, x1 = h[0], h[1]
            y0, y1 = stack[0][2], stack[-1][2]
            # column separators: vertical segments inside the table box,
            # clustered by x (they may be drawn per-row, not full height)
            hits = {}
            for x, vy0, vy1 in vs:
                if x0 - 2 < x < x1 + 2 and vy0 > y0 - 3 and vy1 < y1 + 3:
                    key = round(x, 0)
                    hits[key] = hits.get(key, 0) + (vy1 - vy0)
            cols = sorted(k for k, tot in hits.items()
                          if tot >= 15 or abs(k - x0) < 2 or abs(k - x1) < 2)
            cols = _dedupe(cols, 2.0)
            if not cols or abs(cols[0] - x0) > 2:
                cols.insert(0, x0)
            if abs(cols[-1] - x1) > 2:
                cols.append(x1)
            grids.append({"x0": x0, "x1": x1,
                          "rows": [s[2] for s in stack], "cols": cols})
    # drop duplicate grids from double-stroked borders
    ded = []
    for g in sorted(grids, key=lambda g: (g["x0"], -len(g["rows"]))):
        if ded and abs(g["x0"] - ded[-1]["x0"]) < 4 and abs(g["rows"][0] - ded[-1]["rows"][0]) < 4:
            continue
        ded.append(g)
    ded.sort(key=lambda g: g["x0"])
    return ded
