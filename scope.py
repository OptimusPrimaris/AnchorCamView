"""Image helpers specific to a camera looking through a rifle scope.

The camera is fixed to the scope, so the reticle and the dark eyepiece ring sit
at the SAME pixels in every frame while the target moves behind them (recoil).
Anything fixed in the picture must be kept out of the target alignment, or the
alignment "sees" no movement and the measured offset is wrong.
"""
import cv2
import numpy as np


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def eyepiece_circle(image, shrink=0.93):
    """Find the bright round field of view. Returns (cx, cy, r) or None."""
    g = gray(image)
    s = 400.0 / max(g.shape)
    small = cv2.GaussianBlur(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 3)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    (cx, cy), r = cv2.minEnclosingCircle(biggest)
    roundness = area / (np.pi * r * r + 1e-9)
    if area < 0.05 * small.size or roundness < 0.6:
        return None                     # no clear round field of view (e.g. camera not on the scope)
    return (cx / s, cy / s, r / s * shrink)


def circle_mask(shape, circle):
    mask = np.zeros(shape[:2], np.uint8)
    if circle is None:
        mask[:] = 255
    else:
        cv2.circle(mask, (int(round(circle[0])), int(round(circle[1]))), int(round(circle[2])), 255, -1)
    return mask


def reticle_from_calibration(image, circle=None, grow=7):
    """Reticle mask + centre from a CALIBRATION frame: the scope pointed at a plain,
    bright surface (white wall, sky, blank paper). Everything clearly darker than
    the background inside the eyepiece is the reticle. Do this once per camera
    mounting; the reticle stays at the same pixels in every later frame."""
    circle = circle or eyepiece_circle(image)
    g = cv2.GaussianBlur(gray(image), (0, 0), 1.0).astype(np.float32)
    inside = circle_mask(g.shape, circle) > 0
    # Remove slow brightness falloff toward the edge, then threshold what is darker.
    bg = cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), 51).astype(np.float32)
    rel = g / np.maximum(bg, 1)
    dark = ((rel < 0.7) & inside).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return mask, reticle_center(dark, circle), circle


def reticle_center(rmask, circle=None):
    """Crosshair centre from the reticle mask: the strongest vertical and
    horizontal lines. A fallback when no crosshair model / fixed point is given."""
    m = rmask.astype(np.float32) / 255
    if circle is not None:
        m *= circle_mask(m.shape, circle) / 255
    cols, rows = m.sum(axis=0), m.sum(axis=1)
    if cols.max() < 20 or rows.max() < 20:
        return None
    if circle is not None:
        # A target line that recoil slid along itself also looks "fixed"; the real
        # reticle crosses near the middle of the eyepiece, so favour lines there.
        cx, cy, r = circle
        cols = cols * np.exp(-((np.arange(cols.size) - cx) / (0.5 * r)) ** 2)
        rows = rows * np.exp(-((np.arange(rows.size) - cy) / (0.5 * r)) ** 2)

    def peak(profile):
        i = int(np.argmax(profile))
        lo, hi = max(0, i - 6), min(profile.size, i + 7)
        w = profile[lo:hi]
        return float((np.arange(lo, hi) * w).sum() / max(w.sum(), 1e-9))
    return (peak(cols), peak(rows))


def new_hole_by_difference(before, after, H_after_to_before, valid_after, valid_before, fixed_mask,
                           min_area=12, max_area=None):
    """Find the new bullet hole: line the BEFORE frame up onto the AFTER frame,
    compare, and keep the strongest compact, roughly round spot that changed.
    A hole on white paper gets darker; a hole in a black bullseye gets lighter,
    so changes in both directions count.

    Returns (best, candidates, heat) in AFTER-frame pixels. `fixed_mask` covers
    things fixed in the camera (reticle); it is excluded at its spot in the after
    frame and at its moved spot in the warped before frame."""
    h, w = after.shape[:2]
    H_ba = np.linalg.inv(H_after_to_before)
    # The before frame's reticle hid some paper (usually right around the point of
    # aim, where shots land). Fill it in from the surrounding paper so those spots
    # can still be compared, instead of being skipped.
    before_clean = cv2.inpaint(before, fixed_mask, 5, cv2.INPAINT_TELEA)
    warped = cv2.warpPerspective(before_clean, H_ba, (w, h), flags=cv2.INTER_LINEAR)
    vb = cv2.warpPerspective(valid_before, H_ba, (w, h), flags=cv2.INTER_NEAREST)
    area_ok = cv2.erode(((valid_after > 0) & (vb > 0)).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)) > 0
    hidden = fixed_mask > 0                       # behind the reticle in the AFTER frame: can't be seen
    # Where the before frame's reticle was, the paper was filled in by guesswork; an
    # old hole hidden there would look "new". Spots found there count for less.
    guessed = cv2.warpPerspective(fixed_mask, H_ba, (w, h), flags=cv2.INTER_NEAREST) > 0
    valid = area_ok & ~hidden
    if valid.sum() < 500:
        return None, [], None
    g1 = cv2.GaussianBlur(gray(warped).astype(np.float32), (0, 0), 1.2)
    g2 = cv2.GaussianBlur(gray(after).astype(np.float32), (0, 0), 1.2)
    # Match brightness/contrast of the two frames inside the usable area.
    a, b = np.polyfit(g2[valid], g1[valid], 1)
    diff = np.zeros_like(g1)
    diff[valid] = np.abs(g1 - (a * g2 + b))[valid]    # how much each spot changed
    v = diff[valid]
    med, mad = float(np.median(v)), float(np.median(np.abs(v - np.median(v)))) + 1e-6
    thresh = max(20.0, med + 6 * 1.4826 * mad)
    bw = ((diff > thresh) & valid).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # A hole partly behind a thin reticle line is split in two; rejoin across the line.
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    bw = np.where(hidden & area_ok, closed, bw).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    max_area = max_area or 0.01 * valid.sum()
    cands = []
    for i in range(1, n):
        x, y, bw_, bh_, area = (int(v) for v in stats[i])
        if not (min_area <= area <= max_area):
            continue
        fill = area / float(bw_ * bh_)
        aspect = max(bw_, bh_) / max(1, min(bw_, bh_))
        if fill < 0.4 or aspect > 2.5:
            continue                                   # thin streak = alignment/reticle residue, not a hole
        ys, xs = np.nonzero(labels == i)
        wts = diff[ys, xs] + 1e-3
        cx, cy = float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
        in_guess = float(guessed[ys, xs].mean())
        cands.append({"x": cx, "y": cy, "area_px": area, "box": (x, y, x + bw_, y + bh_),
                      "strength": float(wts.sum() * fill * (1 - 0.95 * in_guess)), "peak": float(wts.max()),
                      "roundness": round(fill, 2), "under_old_reticle": round(in_guess, 2)})
    cands.sort(key=lambda c: -c["strength"])
    if cands:
        cands[0] = refine_hole_centre(after, cands[0], fixed_mask)
    heat = np.clip(diff / max(thresh * 3, 1) * 255, 0, 255).astype(np.uint8)
    return (cands[0] if cands else None), cands, heat


def refine_hole_centre(after, cand, fixed_mask=None, pad=6, min_contrast=40):
    """Re-centre a hole on its DARK CORE in the after frame.

    The change map is lopsided when the hole's torn edge covers a printed line, so
    its centre can be pulled several pixels off. The hole's core (the dark opening)
    isn't. Falls back to the change-map centre when the hole isn't clearly darker
    than its surroundings (e.g. inside a black bullseye)."""
    x1, y1, x2, y2 = (int(v) for v in cand["box"])
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(after.shape[1], x2 + pad + 1), min(after.shape[0], y2 + pad + 1)
    g = cv2.GaussianBlur(gray(after), (0, 0), 1.0)[Y1:Y2, X1:X2].astype(np.float32)
    rm = (fixed_mask[Y1:Y2, X1:X2] > 0) if fixed_mask is not None else np.zeros(g.shape, bool)
    vals = g[~rm] if (~rm).any() else g.ravel()
    lo, mid = float(np.percentile(vals, 2)), float(np.median(vals))
    if mid - lo < min_contrast:
        return cand
    m = ((g < lo + 0.5 * (mid - lo)) & ~rm).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return cand
    j = 1 + int(np.argmin([np.hypot(cen[i][0] + X1 - cand["x"], cen[i][1] + Y1 - cand["y"]) for i in range(1, n)]))
    if st[j, 4] < 6:
        return cand
    out = dict(cand)
    out["x"], out["y"] = float(cen[j][0] + X1), float(cen[j][1] + Y1)
    out["centred_on"] = "dark core"
    return out


import settings as _S
NEW_DARKENING = _S.NEW_IMPACT_CHANGE   # grey levels: how much a new impact changes the target (model path)


def new_hole_by_model(detector, label, before, after, H_after_to_before, conf, match_px=15, fixed_mask=None):
    """With a trained impact model: impacts in AFTER that weren't there BEFORE.
    Returns (best, candidates).

    An after-frame impact is new if no before-frame impact maps onto it AND the paper
    there actually changed (the before frame, aligned onto the after frame, looks
    different inside the box). The second test matters because an old hole can be
    missed in the before frame (under the reticle, or outside the eyepiece view).
    Detections at the same pixel in both frames are fixed to the camera (reticle,
    lens dust) and ignored; ones on the reticle are ranked last."""
    after_holes = [d for d in detector.detect_all(after, label) if d.confidence >= conf]
    before_holes = [d for d in detector.detect_all(before, label) if d.confidence >= conf]
    H_ba = np.linalg.inv(H_after_to_before)
    old = [cv2.perspectiveTransform(np.float32([[d.pixel_position]]), H_ba)[0, 0] for d in before_holes]
    fixed = [np.array(d.pixel_position) for d in before_holes]
    h, w = after.shape[:2]
    gb = cv2.warpPerspective(gray(before), H_ba, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    seen = cv2.warpPerspective(np.full(before.shape[:2], 255, np.uint8), H_ba, (w, h), flags=cv2.INTER_NEAREST)
    if fixed_mask is not None:                        # paper under the reticle in the before frame wasn't seen
        seen[cv2.warpPerspective(fixed_mask, H_ba, (w, h), flags=cv2.INTER_NEAREST) > 0] = 0
    ga = gray(after)
    cands = []
    for d in after_holes:
        p = np.array(d.pixel_position)
        nearest = min((float(np.hypot(*(p - o))) for o in old), default=1e9)
        if any(np.hypot(*(p - f)) <= 3 for f in fixed):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        vis = seen[y1:y2, x1:x2] > 0
        # change inside the box, minus the change around it (lighting changes). Either
        # direction counts: a hole darkens white paper but can lighten a dark target.
        dark = gb[y1:y2, x1:x2].astype(float) - ga[y1:y2, x1:x2]
        m = 6
        X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
        ring = gb[Y1:Y2, X1:X2].astype(float) - ga[Y1:Y2, X1:X2]
        ring_seen = seen[Y1:Y2, X1:X2] > 0
        ring_seen[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
        base = float(np.median(ring[ring_seen])) if ring_seen.any() else 0.0
        change = abs(float(np.mean(dark[vis])) - base) if vis.mean() > 0.2 else None
        if nearest <= match_px and not (change or 0) > 2 * NEW_DARKENING:
            continue                      # an old impact (unless the paper there clearly changed)
        on_reticle = bool(fixed_mask is not None and fixed_mask[int(round(p[1])), int(round(p[0]))])
        cands.append({"x": float(p[0]), "y": float(p[1]), "box": d.box, "confidence": d.confidence,
                      "change": change, "seen_before": change is not None,
                      "nearest_old_hole_px": nearest if old else None, "on_reticle": on_reticle})
    # clearly darkened first; then spots hidden in the before frame; then unchanged ones
    rank = lambda c: (c["on_reticle"], 0 if (c["change"] or 0) > NEW_DARKENING else 1 if c["change"] is None else 2,
                      -(c["change"] or 0) * c["confidence"])
    cands.sort(key=rank)
    return (cands[0] if cands else None), cands


def rough_shift(before, after, circle=None, fixed_mask=None, size=320):
    """Approximate paper movement BEFORE -> AFTER (dx, dy in full-res px) by phase
    correlation on a square inside the eyepiece, with the reticle filled in so it
    can't pin the answer at zero. Uses the whole picture, so a repeating pattern of
    dots can't fool it the way pairing individual dots can."""
    h, w = before.shape[:2]
    cx, cy, r = circle if circle is not None else (w / 2, h / 2, min(w, h) / 2)
    half = r / np.sqrt(2) * 0.95
    x1, y1, x2, y2 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))
    k = (x2 - x1) / float(size)

    def prep(img):
        g = cv2.resize(gray(img)[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
        if fixed_mask is not None:
            m = cv2.resize(fixed_mask[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST)
            g = cv2.inpaint(g, cv2.dilate(m, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
        return g.astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(prep(before), prep(after), cv2.createHanningWindow((size, size), cv2.CV_32F))
    return (dx * k, dy * k), float(resp)


def refine_alignment(before, after, H_ab, fixed_mask=None, valid_a=None, valid_b=None, scale=0.5, iters=100):
    """Fine-tune a rough AFTER -> BEFORE transform (e.g. from target-center dots) to
    sub-pixel accuracy with ECC: it adjusts the transform until the paper's
    brightness pattern lines up, using all paper pixels rather than a few points.
    Needs a decent starting guess; returns (refined H, correlation) or (None, reason)."""
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])

    def prep(img):
        g = cv2.resize(gray(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if fixed_mask is not None:                      # remove the reticle from both frames
            m = cv2.resize(fixed_mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            g = cv2.inpaint(g, m, 3, cv2.INPAINT_TELEA)
        return g.astype(np.float32)
    tb, ta = prep(before), prep(after)
    mask = None
    if valid_a is not None:
        mask = cv2.resize(cv2.erode(valid_a, np.ones((15, 15), np.uint8)), (ta.shape[1], ta.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    # ECC warp maps template (before) coordinates to input (after) coordinates.
    W = (S @ np.linalg.inv(H_ab) @ Si).astype(np.float32)
    W /= W[2, 2]
    try:
        cc, W = cv2.findTransformECC(tb, ta, W, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6), mask, 5)
    except cv2.error as e:
        return None, f"did not converge ({str(e).splitlines()[-1][:60]})"
    H = Si @ np.linalg.inv(W.astype(np.float64)) @ S
    return H / H[2, 2], float(cc)


def reticle_band_mask(shape, centre, half=8):
    """Reticle mask without a calibration frame: a horizontal and a vertical band
    through the crosshair centre (from YOLO). Assumes the reticle lines run along
    the image rows/columns, i.e. the camera is mounted level with the scope."""
    m = np.zeros(shape[:2], np.uint8)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    m[max(0, cy - half):cy + half + 1, :] = 255
    m[:, max(0, cx - half):cx + half + 1] = 255
    return m


def reticle_line_mask(image, centre, search_px=12, margin_px=2, info=None, circle=None, dot_px=2):
    """Reticle mask that hugs the actual reticle lines instead of a wide band.

    The reticle is two straight lines through the crosshair. For each, the code looks
    within +-search_px of the crosshair for the row (or column) that is darkest along
    its whole length, measures how thick the line is there (including its dots/hash
    marks), and masks only that strip, plus margin_px either side. Things right next
    to the line (an impact mark) stay visible. Assumes the lines run along the image
    rows/columns. Returns None if a line can't be measured (use reticle_band_mask)."""
    g = cv2.GaussianBlur(gray(image).astype(np.float32), (0, 0), 0.8)
    dark = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    h, w = g.shape
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    circle = circle if circle is not None else eyepiece_circle(image)
    if circle is not None:                         # only judge the line where the view is visible
        inside = circle_mask(image.shape, (circle[0], circle[1], circle[2] * 0.85)) > 0
        dark = np.where(inside, dark, np.nan)
    mask = np.zeros((h, w), np.uint8)
    found = {}
    for axis, c in ((0, cy), (1, cx)):          # 0: horizontal line (rows), 1: vertical line (columns)
        lo, hi = max(0, c - search_px), min(h if axis == 0 else w, c + search_px + 1)
        band = dark[lo:hi, :] if axis == 0 else dark[:, lo:hi].T
        # use only the half of the line away from the crossing on each side? no: whole length,
        # but a high percentile so dots and faint stretches both count
        prof = np.nan_to_num(np.nanpercentile(band, 75, axis=1))
        if prof.max() < 4:
            return None
        k = int(np.argmax(prof))
        half_max = prof[k] / 2
        a_, b_ = k, k
        while a_ > 0 and prof[a_ - 1] > half_max:
            a_ -= 1
        while b_ < len(prof) - 1 and prof[b_ + 1] > half_max:
            b_ += 1
        a2, b2 = a_ - dot_px, b_ + dot_px                 # room for the dots/hash marks on the line
        s0, s1 = lo + a2 - margin_px, lo + b2 + margin_px
        found["horizontal" if axis == 0 else "vertical"] = (lo + (a_ + b_) / 2, b_ - a_ + 1)
        if axis == 0:
            mask[max(0, s0):s1 + 1, :] = 255
        else:
            mask[:, max(0, s0):s1 + 1] = 255
    if info is not None:
        info.update(found)
    return mask
