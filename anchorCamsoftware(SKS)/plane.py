"""The absolute coordinate plane both images are measured in.

The plane is defined by a reference image of the flat surface (a template such
as your reticle printout, a floor mat, a calibration board) or, if none is
given, by image 1 itself. Each image gets its own homography image -> plane, so
both images' detections land in ONE shared coordinate system.

Plane coordinates: start as template pixels, then optionally
  * shifted so `origin_px` becomes (0, 0)  (e.g. the reticle centre),
  * flipped so +y points up (`y_up`),
  * scaled by `units_per_px` (e.g. 0.5 mm per template pixel) for real units.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from alignment import estimate_homography


@dataclass
class PlaneFrame:
    image: np.ndarray                 # the reference image that defines the plane
    source: str                       # "template:<path>" or "image1"
    origin_px: tuple = (0.0, 0.0)
    y_up: bool = False
    units_per_px: float = 1.0
    units: str = "plane px"

    def to_units(self, plane_px):
        """Template-pixel coordinates -> absolute plane coordinates."""
        x = (plane_px[0] - self.origin_px[0]) * self.units_per_px
        y = (plane_px[1] - self.origin_px[1]) * self.units_per_px
        return (float(x), float(-y if self.y_up else y))

    def describe(self):
        h, w = self.image.shape[:2]
        return {"source": self.source, "reference_size_px": [w, h],
                "origin_px": list(self.origin_px), "y_axis": "up" if self.y_up else "down",
                "units": self.units, "units_per_reference_px": self.units_per_px}


def _shrink(image, max_side):
    """Downscale big photos for feature matching; returns (small_image, scale)."""
    s = min(1.0, max_side / max(image.shape[:2]))
    if s >= 1.0:
        return image, 1.0
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def register_images(src, dst, exclude_src=None, method="sift", min_coverage=0.10, max_side=1600,
                    mask_src=None, mask_dst=None, exclude_dst=None, min_inlier_ratio=0.5):
    """Homography src pixels -> dst pixels at FULL resolution.

    Matching runs on copies no larger than `max_side` (phone photos are 12+ MP),
    then the transform is rescaled: H_full = S_dst^-1 @ H_small @ S_src.
    """
    small_src, s_src = _shrink(src, max_side)
    small_dst, s_dst = _shrink(dst, max_side)
    boxes = [tuple(v * s_src for v in b) for b in exclude_src] if exclude_src else None
    boxes_dst = [tuple(v * s_dst for v in b) for b in exclude_dst] if exclude_dst else None
    m_src = None if mask_src is None else cv2.resize(mask_src, (small_src.shape[1], small_src.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
    m_dst = None if mask_dst is None else cv2.resize(mask_dst, (small_dst.shape[1], small_dst.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
    fit = estimate_homography(small_src, small_dst, method=method, exclude_src=boxes, exclude_dst=boxes_dst,
                              mask_src=m_src, mask_dst=m_dst, min_coverage=min_coverage,
                              min_inlier_ratio=min_inlier_ratio)
    if fit["valid"]:
        S_src = np.diag([s_src, s_src, 1.0])
        S_dst_inv = np.diag([1.0 / s_dst, 1.0 / s_dst, 1.0])
        H = S_dst_inv @ fit["matrix"] @ S_src
        fit["matrix"] = H / H[2, 2]
        if s_src < 1.0 or s_dst < 1.0:
            fit["matching_scale"] = [s_src, s_dst]
    return fit


def register_to_plane(image, plane, exclude_boxes=None, method="sift", min_coverage=0.10):
    """Homography that takes `image` pixels to plane (reference) pixels."""
    if plane.source == "image1" and image is plane.image:
        return {"valid": True, "matrix": np.eye(3), "method": "identity (image 1 defines the plane)"}
    return register_images(image, plane.image, exclude_src=exclude_boxes,
                           method=method, min_coverage=min_coverage)


def map_point(matrix, point):
    return tuple(float(v) for v in cv2.perspectiveTransform(np.float32([[point]]), matrix)[0, 0])


def anchor_point(detection, anchor):
    """Which pixel of a detection represents its position ON the plane.

    'center' suits flat objects lying on the plane. For objects standing on a
    floor/table, the box centre floats above the plane, so 'bottom' (bottom-centre
    of the box, where the object touches the surface) is the better choice.
    """
    if anchor == "bottom" and detection.box is not None:
        x1, _, x2, y2 = detection.box
        return ((x1 + x2) / 2.0, float(y2))
    return tuple(float(v) for v in detection.pixel_position)


def vector(p_from, p_to):
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return {"dx": dx, "dy": dy, "distance": float(np.hypot(dx, dy)), "angle_deg": angle}
