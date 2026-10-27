"""Pixels -> inches on the target, from something the camera can see.

Choose ONE reference (whichever the team settles on). Everything comes from YOLO
detections, so nothing about the scope or a particular printed sheet is assumed:

  grid lines         --grid-in 1
        printed grid squares of known size, found with OpenCV (no model needed)
  landmark spacing   --landmark-spacing-in 3
        real distance between neighbouring repeated marks found by YOLO: the
        Target-Center dots by default, or e.g. a "Grid-Intersection" class
        (--scale-class) if the team trains one for the grid marks
  block              --block-class Scale-Block --block-size-in 2 [--block-model m.pt]
        a block of known width placed on/next to the target, found by YOLO
  manual             --px-per-inch 38
        a number you measured yourself

(The grid method looks only for long, thin, straight lines and checks that their
gaps agree on one square size; an earlier "whatever repeats in the picture"
approach was rejected because a busy target repeats at several spacings.)

Inches become MOA using the distance (default 25 yd): 1 MOA = 1.0472 in x yards/100.
All scales are measured in the POINT-OF-AIM frame, where the displacement is measured.
"""
import cv2
import numpy as np
import landmarks


def from_landmarks(points_px, spacing_in, name="landmark"):
    """Median neighbour spacing of repeated YOLO marks (px) / their real spacing (in)."""
    sp = landmarks.spacing(points_px)
    if not sp:
        return None, f"need at least 2 '{name}' detections in the point-of-aim frame"
    return sp / spacing_in, f"{name} spacing ({len(points_px)} marks, {sp:.1f} px = {spacing_in:g} in)"


def from_block(detector, image, label, size_in, conf, circle=None, aspect=1.0):
    """Average of a reference block's box width and height (px) / its real size (in).

    Rejects a block that is partly hidden (box shape doesn't match the block's
    known width/height ratio) or partly outside the eyepiece view, because then
    the box is too small and the scale would be wrong."""
    found = [d for d in detector.detect_all(image, label) if d.confidence >= conf and d.box]
    if not found:
        return None, f"no '{label}' found in the point-of-aim frame"
    x1, y1, x2, y2 = found[0].box
    w, h = x2 - x1, y2 - y1
    if abs((w / max(h, 1e-6)) / aspect - 1) > 0.08:
        return None, f"'{label}' looks partly hidden (box {w:.0f} x {h:.0f} px)"
    if circle is not None:
        cx, cy, r = circle
        if max(np.hypot(px - cx, py - cy) for px in (x1, x2) for py in (y1, y2)) > 0.95 * r:
            return None, f"'{label}' is partly outside the eyepiece view"
    px = (w + h) / 2.0
    return px / size_in, f"reference block ({px:.1f} px = {size_in:g} in, conf {found[0].confidence:.2f})"


def from_grid(image, square_in, circle=None, fixed_mask=None, strip=96, min_gaps=8, debug=None):
    """Pixels per inch from printed grid lines, with OpenCV only (no model).

    1. thin features only, dark-on-light OR light-on-dark (top-hat / black-hat with a
       small kernel: filled areas give nothing)
    2. the grid's tilt is measured (dominant edge direction) and the picture is turned
       to straighten it, so a target hung at an angle still works
    3. a long horizontal / vertical opening keeps only long straight runs, so rings,
       numbers, holes and 45-degree diamonds drop out
    4. the view is cut into overlapping strips; in each strip the typical value of every
       row (or column) forms a profile in which each grid line is a peak, located to
       sub-pixel
    5. gaps between neighbouring lines: each gap is a whole number of squares (a line
       hidden by the reticle or a hole makes a 2-square gap), so the square size is
       fitted to all gaps at once. It refuses if the gaps don't agree on one size."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark_lines = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k9).astype(np.float32)
    light_lines = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k9).astype(np.float32)
    h, w = g.shape
    ok = np.full((h, w), 255, np.uint8)
    if circle is not None:
        ok = np.zeros((h, w), np.uint8)
        cv2.circle(ok, (int(circle[0]), int(circle[1])), int(circle[2] * 0.92), 255, -1)
    if fixed_mask is not None:
        ok[cv2.dilate(fixed_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    centre = (circle[0], circle[1]) if circle is not None else (w / 2, h / 2)
    # dark lines on light paper, or light lines on a dark target: measure both, keep the one
    # whose line gaps agree best on a single square size
    results = []
    for polarity, r in (("dark", dark_lines), ("light", light_lines)):
        r[ok == 0] = 0
        results.append(_grid_spacing(r, ok, centre, polarity, strip, min_gaps, debug))
    good = [x for x in results if x[0] is not None]
    if not good:
        return None, max(results, key=lambda x: x[3])[1]
    sq, how, dbg, n = max(good, key=lambda x: x[3])
    if debug is not None:
        debug.clear()
        debug.update(dbg)
    return sq / square_in, how


def _grid_spacing(resp, ok, centre, polarity, strip, min_gaps, want_debug):
    """Square size (px) from one polarity of line image: (size or None, text, debug, n gaps fitted)."""
    h, w = resp.shape
    tilt = grid_tilt(resp, ok, centre)
    if abs(tilt) > 0.3:                         # straighten the grid
        R = cv2.getRotationMatrix2D(centre, tilt, 1.0)
        resp = cv2.warpAffine(resp, R, (w, h), flags=cv2.INTER_LINEAR)
        ok = cv2.warpAffine(ok, R, (w, h), flags=cv2.INTER_NEAREST)
        ok = cv2.erode(ok, np.ones((5, 5), np.uint8))
    dbg = dict(tilt=tilt, polarity=polarity, resp=resp.copy(), ok=ok.copy(), peaks=[], strip=strip,
               centre=centre) if want_debug is not None else {}
    gaps = []
    for axis in (0, 1):                       # 0: horizontal lines (peaks along y), 1: vertical
        k = (31, 1) if axis == 0 else (1, 31)
        L = cv2.morphologyEx(resp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        img, valid = (L, ok) if axis == 0 else (L.T, ok.T)
        for s0 in range(0, img.shape[1] - strip // 2, strip // 2):
            seg, seg_ok = img[:, s0:s0 + strip], valid[:, s0:s0 + strip] > 0
            rows = np.where(seg_ok.sum(axis=1) > strip * 0.6)[0]
            if len(rows) < 40:
                continue
            prof = np.zeros(seg.shape[0], np.float32)
            for r in rows:
                prof[r] = np.median(seg[r][seg_ok[r]])
            peaks = _peaks(prof, max(3.0, 4 * float(np.median(prof[rows]))))
            if dbg:
                dbg["peaks"].append((axis, s0, peaks))
            for p1, p2 in zip(peaks[:-1], peaks[1:]):   # only gaps with valid rows all the way
                if np.all(np.isin(np.arange(int(p1), int(p2) + 1), rows)):
                    gaps.append(p2 - p1)
    gaps = np.array([x for x in gaps if x > 5])
    if len(gaps) < min_gaps:
        return None, f"too few grid lines found ({len(gaps)} gaps)", dbg, 0
    best = None                                # the smallest spacing most gaps are whole multiples of
    for cand in np.unique(np.round(gaps)):
        if cand < 6:
            continue
        m = np.maximum(np.round(gaps / cand), 1)
        fit = np.abs(gaps / m - cand) < 0.06 * cand
        if fit.mean() < 0.6 or (m[fit] == 1).mean() < 0.4:
            continue
        if best is None or cand < best * 0.9:
            best = float(cand)
    if best is None:
        return None, f"grid lines found but their spacing is inconsistent ({len(gaps)} gaps)", dbg, 0
    sq = best                                  # centre the estimate: re-fit around it until it settles
    for _ in range(10):
        m = np.maximum(np.round(gaps / sq), 1)
        fit = np.abs(gaps / m - sq) < 0.04 * sq
        sq = float(np.median(gaps[fit] / m[fit]))
    n, share = int(fit.sum()), float(fit.mean())
    if dbg:
        dbg["square_px"] = sq
    return sq, (f"grid lines ({n} gaps, {sq:.2f} px = 1 square, {share:.0%} consistent, {polarity} lines, "
                f"grid turned {tilt:+.1f} deg to straighten)"), dbg, n


def grid_tilt(resp, ok, centre, max_deg=20.0, with_score=False):
    """Angle (degrees) to turn the picture so the grid lines run along rows/columns.
    Tries angles and keeps the one where the lines line up best: the row and column
    sums of the line image are then most "peaky" (sharp lines, empty squares between).
    Searches +-20 degrees: enough for a target hung roughly level, and it keeps
    45-degree shapes on the target (diamonds) from being mistaken for the grid."""
    s = 0.5                                             # work at half size: fast and enough
    small = cv2.resize(resp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    oks = cv2.resize(ok, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    c = (centre[0] * s, centre[1] * s)
    hh, ww = small.shape

    def score(deg):
        R = cv2.getRotationMatrix2D(c, deg, 1.0)
        r = cv2.warpAffine(small, R, (ww, hh))
        m = cv2.warpAffine(oks, R, (ww, hh), flags=cv2.INTER_NEAREST) > 0
        total = 0.0
        for ax in (0, 1):
            n = m.sum(axis=ax)
            keep = n > 0.5 * n.max()
            prof = (r.sum(axis=ax)[keep] / n[keep]).astype(np.float64)
            fine_detail = prof - cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 6).ravel()
            total += float(fine_detail.var())           # sharp lines only, not slow changes
        return total

    coarse = np.arange(-max_deg, max_deg + 0.5, 1.0)
    best = coarse[int(np.argmax([score(d) for d in coarse]))]
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    scores = [score(d) for d in fine]
    k = int(np.argmax(scores))
    tilt = float(fine[k])
    if not with_score:
        return tilt
    flat = np.median([score(d) for d in (tilt + 22.5, tilt - 22.5)])  # same picture, lines not lined up
    return tilt, scores[k] / max(flat, 1e-9)


def _peaks(profile, level):
    """Sub-pixel positions of local maxima above `level` (parabola through the top 3 values)."""
    out = []
    for i in range(1, len(profile) - 1):
        v = profile[i]
        if v >= level and v >= profile[i - 1] and v > profile[i + 1] and v == profile[max(0, i - 3):i + 4].max():
            a, b, c = profile[i - 1], v, profile[i + 1]
            d = a - 2 * b + c
            out.append(i + (0.5 * (a - c) / d if d < 0 else 0.0))
    return np.array(out)


def grid_region(image, circle=None, fixed_mask=None, square_px=None):
    """Mask of the gridded target area (where long grid lines are), so things off the
    target, such as the hand holding it, can be ignored. None if no grid is found."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    h, w = g.shape
    ok = np.full((h, w), 255, np.uint8)
    if circle is not None:
        ok = np.zeros((h, w), np.uint8)
        cv2.circle(ok, (int(circle[0]), int(circle[1])), int(circle[2] * 0.95), 255, -1)
    best = None
    for r in (cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k9), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k9)):
        r = r.astype(np.float32)
        r[ok == 0] = 0
        if fixed_mask is not None:
            r[fixed_mask > 0] = 0
        lines = np.zeros((h, w), np.uint8)
        for k in ((21, 1), (1, 21)):
            L = cv2.morphologyEx(r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
            lines |= (L > max(6.0, float(np.percentile(L[ok > 0], 90)))).astype(np.uint8)
        if best is None or lines.sum() > best.sum():
            best = lines
    sq = int(square_px or 30)
    region = cv2.morphologyEx(best * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (sq * 2, sq * 2)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(region)
    if n < 2:
        return None
    big = ((lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(big)
    cv2.drawContours(filled, contours, -1, 255, -1)          # no holes inside the target
    return cv2.dilate(filled, np.ones((5, 5), np.uint8))
