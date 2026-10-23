"""Align common stationary PLANAR texture between two images.

`estimate_homography` maps pixels of `image_src` into `image_dst` from matched
features, and refuses to return a transform when quality checks fail (too few
matches, poor agreement, high error, or matches crowded into a small area).
"""
import cv2
import numpy as np


def _feature_mask(shape, exclude_boxes, pad=12, base=None):
    """Where features may be found: `base` mask (255 = usable) minus detected boxes."""
    if not exclude_boxes and base is None:
        return None
    mask = np.full(shape[:2], 255, np.uint8) if base is None else base.copy()
    h, w = shape[:2]
    for x1, y1, x2, y2 in exclude_boxes or []:
        cv2.rectangle(mask,
                      (max(0, int(x1) - pad), max(0, int(y1) - pad)),
                      (min(w - 1, int(x2) + pad), min(h - 1, int(y2) + pad)),
                      0, -1)
    return mask


def _features(gray, method, mask):
    if method == "sift" and hasattr(cv2, "SIFT_create"):
        finder, norm = cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    else:
        finder, norm = cv2.ORB_create(nfeatures=2200), cv2.NORM_HAMMING
    points, descriptors = finder.detectAndCompute(gray, mask)
    return points, descriptors, norm


def estimate_homography(image_src, image_dst, method="orb",
                        exclude_src=None, exclude_dst=None, mask_src=None, mask_dst=None,
                        min_matches=12, min_inliers=10, min_inlier_ratio=0.5,
                        max_median_error=2.0, min_coverage=0.10):
    """Return a dict with `valid`, `matrix` (src -> dst) and quality figures."""
    gray_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY) if image_src.ndim == 3 else image_src
    gray_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2GRAY) if image_dst.ndim == 3 else image_dst
    points_src, desc_src, norm = _features(gray_src, method,
                                           _feature_mask(gray_src.shape, exclude_src, base=mask_src))
    points_dst, desc_dst, _ = _features(gray_dst, method,
                                        _feature_mask(gray_dst.shape, exclude_dst, base=mask_dst))
    if desc_src is None or desc_dst is None or len(points_src) < 2 or len(points_dst) < 2:
        return {"valid": False, "reason": "Insufficient scene texture"}
    pairs = cv2.BFMatcher(norm).knnMatch(desc_src, desc_dst, k=2)
    matches = [pair[0] for pair in pairs
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < min_matches:
        return {"valid": False, "reason": "Too few reliable feature matches",
                "matches": len(matches)}
    src = np.float32([points_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([points_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return {"valid": False, "reason": "No usable alignment transform"}
    keep = mask.ravel().astype(bool)
    inliers = int(keep.sum())
    if inliers < min_inliers or inliers / len(matches) < min_inlier_ratio:
        return {"valid": False, "reason": "Insufficient agreement between matches",
                "matches": len(matches), "inliers": inliers}
    projected = cv2.perspectiveTransform(src[keep], matrix)
    errors = np.linalg.norm(projected - dst[keep], axis=2)
    median_error = float(np.median(errors))
    # Reject matches crowded into one small portion of the image.
    usable = gray_src.size if mask_src is None else max(1, int(np.count_nonzero(mask_src)))
    coverage = cv2.contourArea(cv2.convexHull(src[keep])) / usable
    if median_error > max_median_error or coverage < min_coverage:
        return {"valid": False, "reason": "Alignment error or feature coverage failed",
                "median_reprojection_error_pixels": median_error,
                "feature_coverage_fraction": float(coverage)}
    return {
        "valid": True,
        "matrix": matrix,
        "method": method,
        "inliers": inliers,
        "matches": len(matches),
        "median_reprojection_error_pixels": median_error,
        "feature_coverage_fraction": float(coverage),
    }
