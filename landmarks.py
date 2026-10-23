"""Relate two frames using printed landmarks found by YOLO (e.g. Target-Center dots).

The dots are printed on the paper, so they move with it during recoil. Pairing
the same dots in the BEFORE and AFTER frames gives the transform between the
frames without relying on paper texture or a known sheet layout:

  dots matched   transform fitted          corrects for
  1              translation               sliding only (the "target-center origin" method)
  2              similarity                sliding + rotation + scale
  3              affine                    + shear
  4 or more      homography (RANSAC)       + perspective (same model as feature alignment)

Pairing: every (before dot, after dot) pair proposes a shift; the shift that
lines up the most dots wins (or a predicted transform is used when available).
"""
import cv2
import numpy as np


def centres(detector, image, label, conf):
    """Centres of all detections of `label` above `conf`, most confident first."""
    return [tuple(map(float, d.pixel_position)) for d in detector.detect_all(image, label) if d.confidence >= conf]


def spacing(pts):
    """Typical distance between neighbouring dots (median nearest-neighbour), or None."""
    if len(pts) < 2:
        return None
    P = np.float32(pts)
    d = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def match(before_pts, after_pts, predict=None, tol=None):
    """Pairs (i_before, j_after) of the same physical dot.

    predict: optional 3x3 transform BEFORE -> AFTER (from feature alignment or the
    rough shift). Dots often form a repeating pattern, so without a prediction a
    shift of one whole pattern step can look as good as the true one; then the
    shift agreeing with the most dots wins and ties go to the smaller shift."""
    if not before_pts or not after_pts:
        return []
    B, A = np.float32(before_pts), np.float32(after_pts)
    if tol is None:
        sp = spacing(before_pts) or spacing(after_pts)
        tol = max(15.0, 0.35 * sp) if sp else 25.0
    if predict is not None:
        guesses = [cv2.perspectiveTransform(B.reshape(-1, 1, 2), predict).reshape(-1, 2)]
    else:
        shifts = sorted({(float(A[j][0] - B[i][0]), float(A[j][1] - B[i][1]))
                         for i in range(len(B)) for j in range(len(A))}, key=lambda t: np.hypot(*t))
        guesses = [B + np.float32(t) for t in shifts]
    best = []
    for G in guesses:
        pairs = _pair(G, A, tol)
        if len(pairs) > len(best):
            best = pairs
    # Second pass: fit the loose pairs, then re-pair strictly against that fit, so a
    # dot paired with its neighbour (allowed by the loose tolerance) is corrected.
    if len(best) >= 2:
        M, _ = cv2.estimateAffinePartial2D(B[[i for i, _ in best]], A[[j for _, j in best]], method=cv2.LMEDS)
        if M is not None:
            G = cv2.transform(B.reshape(-1, 1, 2), M).reshape(-1, 2)
            strict = _pair(G, A, max(8.0, 0.1 * tol / 0.35))
            if len(strict) >= min(2, len(best)):
                best = strict
    return best


def candidates(before_pts, after_pts, top=4, tol=None):
    """Without a motion estimate: the few best pairings (most dots agreeing), so a
    later image-based check can choose between "true" and "one pattern step off"."""
    if not before_pts or not after_pts:
        return []
    B, A = np.float32(before_pts), np.float32(after_pts)
    if tol is None:
        sp = spacing(before_pts) or spacing(after_pts)
        tol = max(15.0, 0.35 * sp) if sp else 25.0
    seen, out = set(), []
    for i in range(len(B)):
        for j in range(len(A)):
            pairs = _pair(B + (A[j] - B[i]), A, tol)
            key = tuple(pairs)
            if pairs and key not in seen:
                seen.add(key)
                out.append(pairs)
    out.sort(key=len, reverse=True)
    best = len(out[0]) if out else 0
    return [p for p in out if len(p) >= best - 1][:top]


def _pair(G, A, tol):
    """Nearest after-dot for each predicted before-dot, one-to-one, within tol."""
    pairs, used = [], set()
    for i, g in sorted(enumerate(G), key=lambda t: float(np.hypot(*(A - t[1]).T).min())):
        d = np.hypot(*(A - g).T)
        for j in np.argsort(d):
            if d[j] > tol:
                break
            if int(j) not in used:
                pairs.append((i, int(j)))
                used.add(int(j))
                break
    return sorted(pairs)


def inside(pts, circle, frac=0.93):
    """Drop dots near/over the eyepiece edge (partly cut off -> centre is biased)."""
    if circle is None:
        return list(pts)
    cx, cy, r = circle
    return [p for p in pts if np.hypot(p[0] - cx, p[1] - cy) < frac * r]


def _spread(pts):
    """How 2-D the dot layout is: area of their hull relative to their size (0 = all in a line)."""
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(np.float32(pts).reshape(-1, 1, 2))
    span = float(np.ptp(np.float32(pts), axis=0).max()) + 1e-9
    return float(cv2.contourArea(hull)) / (span * span)


def fit(before_pts, after_pts, pairs, min_spread=0.15, homography_min=5):
    """Transform AFTER -> BEFORE (3x3) from matched dots, and how well it fits.

    Uses the richest model the dots support, but no more: 1 dot = translation;
    2 dots, or dots in a line = similarity; 3+ = affine (robust fit, odd dots
    dropped); a full homography only with >= `homography_min` well-spread dots
    AND only if it fits them better than the affine did."""
    n = len(pairs)
    if n == 0:
        return {"valid": False, "reason": "No target-center dot seen in both frames"}
    src = np.float32([after_pts[j] for _, j in pairs])      # after
    dst = np.float32([before_pts[i] for i, _ in pairs])     # before

    def residuals(H, keep):
        proj = cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.hypot(*(proj - dst[keep]).T)

    def similarity(keep):
        M, _ = cv2.estimateAffinePartial2D(src[keep], dst[keep], method=cv2.LMEDS)
        return None if M is None else np.vstack([M, [0, 0, 1]])

    keep = np.ones(n, bool)
    if n == 1:
        t = dst[0] - src[0]
        H, model = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], float), "translation (1 dot)"
    elif n == 2 or _spread(dst) < min_spread:
        H, model = similarity(keep), f"similarity ({n} dots{', in a line' if n > 2 else ''})"
    else:
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return {"valid": False, "reason": "Could not fit the dots"}
        H = np.vstack([M, [0, 0, 1]])
        r = residuals(H, np.ones(n, bool))
        keep = r <= max(6.0, 3 * float(np.median(r)))           # drop dots that clearly disagree
        if keep.sum() < 3 or _spread(dst[keep]) < min_spread:
            keep = np.ones(n, bool)
            H, model = similarity(keep), f"similarity ({n} dots, too few spread out for more)"
        else:
            M, _ = cv2.estimateAffine2D(src[keep], dst[keep], method=cv2.LMEDS)
            H, model = np.vstack([M, [0, 0, 1]]), f"affine ({int(keep.sum())} of {n} dots)"
            if keep.sum() >= homography_min:
                Hh, _ = cv2.findHomography(src[keep], dst[keep], cv2.LMEDS)
                if Hh is not None and np.median(residuals(Hh, keep)) < np.median(residuals(H, keep)):
                    H, model = Hh, f"homography ({int(keep.sum())} of {n} dots)"
    if H is None:
        return {"valid": False, "reason": "Could not fit the dots"}
    return {"valid": True, "matrix": H, "model": model, "dots": n, "spread": round(_spread(dst), 3),
            "residual_px": float(np.median(residuals(H, keep))) if n > 1 else None}
