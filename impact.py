"""Point of aim vs point of impact from two frames taken through a scope, in MOA.

Everything flows from YOLO detections (no scope specs, no particular printed sheet):

  POA frame    Crosshair-Center      -> point of aim
               Target-Center dots    -> landmarks on the paper
  AFTER frame  Target-Center dots    -> the same landmarks, moved by recoil
               Impact                -> the impact that wasn't there before
               (until an Impact model exists: frame comparison stands in)

`measure()` steps:
  1. aim point   Crosshair-Center (or a calibration frame / fixed pixel if you have one)
  2. reticle     mask = bands through the crosshair along the reticle lines (or the
                 calibration frame's exact mask), so the reticle can't confuse alignment
  3. relate      transform AFTER -> POA frame from the paired Target-Center dots, refined
                 to sub-pixel by ECC (dense brightness alignment). Paper-texture
                 alignment runs too, as an independent cross-check (and fallback).
  4. scale       pixels -> inches from a visible reference (see scale.py):
                 printed grid lines (OpenCV), Target-Center spacing, a reference
                 block, or a measured number
  5. new impact  Impact detections not present before (or frame comparison)
  6. result      displacement in MOA (inches at 25 yd) and the CORRECTION to dial,
                 measured along the camera/reticle axes (the axes the turrets move)

  python impact.py --before b.jpg --after a.jpg --aim-model models/crosshair_target_v1.pt \
      --landmark-spacing-in 3
  python impact.py --video clip.mp4 --before-frame 120 --after-frame 300 --aim-model ... --block-size-in 2 ...
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from common import read_image, label
import detectors
from detectors import load_detector
from plane import register_images, map_point
import landmarks
import arduino_link
import scale as scale_ref
import settings as S
import scope

IN_PER_MOA_100YD = 1.0472
MOA_PER_MIL = 3.4377


def parse_args(argv=None):
    bad = S.check()
    if bad:
        raise ValueError("settings.py: " + "; ".join(bad))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--before", help="Before-shot (point-of-aim) image")
    src.add_argument("--after", help="After-shot image")
    src.add_argument("--video", help="Video file; use with --before-frame and --after-frame")
    src.add_argument("--before-frame", type=int, help="Frame number (1 = first) of the before frame")
    src.add_argument("--after-frame", type=int, help="Frame number of the after frame")
    det = p.add_argument_group("YOLO models / classes")
    det.add_argument("--aim-model", default=S.MODEL, help="Weights with the crosshair class (e.g. models/crosshair_target_v1.pt)")
    det.add_argument("--aim-class", default=S.AIM_CLASS)
    det.add_argument("--target-model", help="Weights with the target-center class (default: --aim-model)")
    det.add_argument("--target-class", default=S.TARGET_CLASS)
    det.add_argument("--hole-model", default=S.MODEL if S.USE_IMPACT_MODEL else None, help="Weights with the impact class (when the team has one)")
    det.add_argument("--hole-class", default=S.IMPACT_CLASS)
    det.add_argument("--conf", type=float, default=S.MIN_CONFIDENCE, help="Minimum YOLO confidence")
    det.add_argument("--yolo-zoom", type=float, default=S.YOLO_ZOOM,
                     help="Crop around the eyepiece and enlarge this many times before YOLO looks "
                          "(helps small dots/impacts; 2.5 worked on the first real video)")
    sc = p.add_argument_group("scale (choose one; see scale.py)")
    sc.add_argument("--grid-in", type=float, default=S.GRID_SQUARE_IN, help="Printed grid square size (in), found with OpenCV, e.g. 1")
    sc.add_argument("--landmark-spacing-in", type=float, help="Real distance between neighbouring landmarks (in)")
    sc.add_argument("--scale-class", help="Class used for --landmark-spacing-in (default: --target-class)")
    sc.add_argument("--block-size-in", type=float, help="Real width of the reference block (in)")
    sc.add_argument("--block-class", default="Scale-Block")
    sc.add_argument("--block-model", help="Weights with the block class (default: --aim-model)")
    sc.add_argument("--block-aspect", type=float, default=1.0, help="Block width/height (1 = square)")
    sc.add_argument("--px-per-inch", type=float, help="Pixels per inch on the target, if measured yourself")
    sc.add_argument("--px-per-moa", type=float, help="Pixels per MOA, if known")
    sc.add_argument("--distance-yd", type=float, default=S.DISTANCE_YD, help="Target distance in yards (default 25)")
    sc.add_argument("--click-moa", type=float, default=S.CLICK_MOA, help="Turret click value in MOA (default 1/4)")
    rel = p.add_argument_group("relating the frames")
    rel.add_argument("--align", choices=["dots", "auto", "features"], default=S.ALIGN,
                     help="dots: Target-Center dots first, texture as cross-check (default); "
                          "auto: texture first, dots as cross-check; features: texture only")
    rel.add_argument("--agree-moa", type=float, default=S.AGREE_MOA, help="Allowed disagreement between the two methods")
    rel.add_argument("--reticle-mask", choices=["lines", "band"], default=S.RETICLE_MASK,
                     help="lines: hug the measured reticle lines; band: fixed-width strips")
    rel.add_argument("--reticle-margin-px", type=int, default=S.RETICLE_MARGIN_PX)
    rel.add_argument("--reticle-dot-px", type=int, default=S.RETICLE_DOT_PX)
    rel.add_argument("--reticle-band-px", type=int, default=S.RETICLE_BAND_PX,
                     help="Half-width of the bands masked along the reticle lines (px)")
    adv = p.add_argument_group("optional extras")
    adv.add_argument("--calibration", help="Frame of the scope aimed at a plain bright surface (not required)")
    adv.add_argument("--aim-xy", type=float, nargs=2, metavar=("X", "Y"), help="Fixed crosshair pixel (no model)")
    adv.add_argument("--impact-xy", type=float, nargs=2, metavar=("X", "Y"),
                     help="Impact pixel in the AFTER frame, marked by hand (testing without an Impact model)")
    adv.add_argument("--min-hole-px", type=int, default=S.MIN_HOLE_PX, help="Smallest hole area, frame comparison only")
    adv.add_argument("--features", choices=["sift", "orb"], default="sift")
    p.add_argument("--output", default="output/impact")
    return p.parse_args(argv)


def read_video_frame(path, number):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, number - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Frame {number} not readable in {path}")
    return frame


def in_per_moa(yards):
    return IN_PER_MOA_100YD * yards / 100


def express(right_px, up_px, scale, args):
    """A displacement in POA-frame pixels (right/up positive) -> every unit we report,
    plus the correction to dial. Used by measure() and by shot_tracker (median of
    several settled frames)."""
    out = {"displacement_px": [float(right_px), float(up_px)], "distance_yd": args.distance_yd}
    if scale.get("px_per_moa"):
        mr, mu = right_px / scale["px_per_moa"], up_px / scale["px_per_moa"]
    elif scale.get("px_per_in"):
        k = scale["px_per_in"] * in_per_moa(args.distance_yd)       # px per MOA at this distance
        mr, mu = right_px / k, up_px / k
    else:
        out.update(units="px", right=float(right_px), up=float(up_px), distance=float(np.hypot(right_px, up_px)))
        return out
    ir, iu = mr * in_per_moa(args.distance_yd), mu * in_per_moa(args.distance_yd)
    elev, wind = -mu, -mr                     # correction = opposite of where the shot went
    out.update(units="MOA", right=float(mr), up=float(mu), distance=float(np.hypot(mr, mu)),
               moa_right=float(mr), moa_up=float(mu),
               mil_right=float(mr / MOA_PER_MIL), mil_up=float(mu / MOA_PER_MIL),
               in_right=float(ir), in_up=float(iu),
               correction={"elevation_moa": float(elev), "windage_moa": float(wind),
                           "elevation_clicks": int(round(elev / args.click_moa)),
                           "windage_clicks": int(round(wind / args.click_moa)),
                           "click_moa": args.click_moa})
    ws, es = arduino_link.motor_steps(wind, elev)                 # what the motors will be told
    out["correction"].update(windage_steps=ws, elevation_steps=es)
    return out


def px_per_moa(scale, args):
    if scale.get("px_per_moa"):
        return scale["px_per_moa"]
    if scale.get("px_per_in"):
        return scale["px_per_in"] * in_per_moa(args.distance_yd)
    return None


# --------------------------------------------------------------------------- the steps

class Unmeasurable(Exception):
    """A step could not be done; the message says why."""


def _aim_point(before, after, args, calib_centre, res):
    """Step 1: the crosshair centre in POA-frame pixels. Returns (point, box or None)."""
    if args.aim_model:
        model = load_detector(args.aim_model)
        found = [d for d in model.detect_all(before, args.aim_class) if d.confidence >= args.conf]
        if not found:
            raise Unmeasurable(f"No '{args.aim_class}' found in the point-of-aim frame.")
        d = found[0]
        res["aim_method"] = f"YOLO {args.aim_class} ({d.confidence:.2f})"
        later = [x for x in model.detect_all(after, args.aim_class) if x.confidence >= args.conf]
        if later:
            res["_aim_after"] = tuple(map(float, later[0].pixel_position))
            res["_after_img"] = after
            drift = float(np.hypot(*(np.array(later[0].pixel_position) - np.array(d.pixel_position))))
            res["crosshair_shift_between_frames_px"] = drift
            if drift > 4:
                res["notes"].append(f"Crosshair moved {drift:.1f} px in the picture between frames: "
                                    "the camera may have shifted on the scope mount.")
        return tuple(map(float, d.pixel_position)), d.box
    if args.aim_xy:
        res["aim_method"] = "fixed pixel"
        return tuple(args.aim_xy), None
    if calib_centre is not None:
        res["aim_method"] = "calibration frame"
        return tuple(calib_centre), None
    raise Unmeasurable(f"No aim point: give --aim-model (with the '{args.aim_class}' class).")


def _reticle_mask(before, args, calib_img, poa, res):
    """Step 2: pixels that are reticle (fixed in the picture), to keep out of alignment."""
    if calib_img is not None:
        if calib_img.shape[:2] != before.shape[:2]:
            raise Unmeasurable("Calibration frame size differs from the shot frames.")
        res["reticle_source"] = "calibration frame"
        return scope.reticle_from_calibration(calib_img)[0]
    def one(img, centre):
        if args.reticle_mask == "lines":
            info = {}
            m = scope.reticle_line_mask(img, centre, search_px=max(12, args.reticle_band_px),
                                        margin_px=args.reticle_margin_px, dot_px=args.reticle_dot_px, info=info)
            if m is not None:
                return m, "the measured reticle lines (" + ", ".join(
                    f"{k} {v[1]} px thick" for k, v in info.items()) + f", +{args.reticle_margin_px} px margin)"
        return (scope.reticle_band_mask(img.shape, centre, args.reticle_band_px),
                f"bands through the crosshair (+-{args.reticle_band_px} px)")
    mask, res["reticle_source"] = one(before, poa)
    if res.get("_aim_after") is not None:      # the reticle may sit elsewhere in the after frame
        mask = mask | one(res["_after_img"], res["_aim_after"])[0]
    return mask


MAX_TURN_DEG = S.MAX_TURN_DEG          # recoil can't turn the target more than this between frames
MAX_SCALE_CHANGE = S.MAX_SCALE_CHANGE  # ... or change its size in the picture by more than this


def implausible(H, centre):
    """Why a frame-to-frame transform can't be real (None if it can). A target with a
    repeating, symmetric pattern can be matched turned by 90 degrees or one pattern
    step off; recoil can't do that, so such answers are rejected."""
    c = np.float32([[centre, (centre[0] + 50, centre[1]), (centre[0], centre[1] + 50)]])
    p = cv2.perspectiveTransform(c, np.asarray(H, float))[0]
    vx, vy = p[1] - p[0], p[2] - p[0]
    turn = float(np.degrees(np.arctan2(vx[1], vx[0])))
    size = float((np.hypot(*vx) + np.hypot(*vy)) / 100.0)
    if abs(turn) > MAX_TURN_DEG:
        return f"turned {turn:.0f} deg between frames (a symmetric pattern matched the wrong way round)"
    if abs(size - 1) > MAX_SCALE_CHANGE:
        return f"size changed {100 * (size - 1):+.0f}% between frames"
    return None


def _features(before, after, args, paper_b, paper_a, boxes, centre=None):
    """Homography AFTER -> POA frame from the paper's texture."""
    feat = register_images(after, before, mask_src=paper_a, mask_dst=paper_b, exclude_src=boxes,
                           exclude_dst=boxes, method=args.features, min_coverage=0.05, min_inlier_ratio=0.3)
    if feat.get("valid") and centre is not None:
        why = implausible(feat["matrix"], centre)
        if why:
            feat = {"valid": False, "reason": f"texture match {why}", "_rejected_matrix": feat["matrix"]}
    return feat


def _landmarks(before, after, args, circle_b, circle_a, rmask, H_feat, valid_a, valid_b, res):
    """Transform AFTER -> POA frame from YOLO target-center dots, ECC-refined."""
    model_path = args.target_model or args.aim_model
    if not model_path:
        return {"valid": False, "reason": "no target-center model (--target-model / --aim-model)"}
    model = load_detector(model_path)
    try:
        B = landmarks.inside(landmarks.centres(model, before, args.target_class, args.conf), circle_b)
        A = landmarks.inside(landmarks.centres(model, after, args.target_class, args.conf), circle_a)
    except ValueError as e:                   # class not in this model
        return {"valid": False, "reason": str(e)}
    res["_dots_before"], res["_dots_after"] = B, A
    if H_feat is not None:
        predict, how = np.linalg.inv(H_feat), "texture alignment"
    else:
        (dx, dy), resp = scope.rough_shift(before, after, circle_b, rmask)
        predict, how = (np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float), "rough shift") if resp >= 0.1 else (None, "voting")
    # Without a motion estimate a repeating pattern can pair "one dot spacing off" as well as
    # the truth: then refine each likely pairing and keep the one whose images agree best.
    options = [landmarks.match(B, A, predict=predict)] if predict is not None else landmarks.candidates(B, A)
    best = None
    for pairs in options:
        fit = landmarks.fit(B, A, pairs)
        if not fit["valid"] or implausible(fit["matrix"], res["poa_before_px"]):
            continue
        H, cc = scope.refine_alignment(before, after, fit["matrix"], rmask, valid_a, valid_b)
        if H is not None and implausible(H, res["poa_before_px"]):
            H, cc = None, "refinement drifted"
        fit["matrix"], fit["refined"] = (H, f"ECC, correlation {cc:.3f}") if H is not None else (fit["matrix"], f"no ({cc})")
        fit["_cc"], fit["_pairs"] = (cc if H is not None else -1.0), pairs
        if best is None or fit["_cc"] > best["_cc"]:
            best = fit
    if best is None:
        return {"valid": False, "reason": f"no '{args.target_class}' seen in both frames",
                "dots_before": len(B), "dots_after": len(A)}
    best["paired_by"] = how + (f" (best of {len(options)} pairings by image agreement)" if predict is None else "")
    best["dots_before"], best["dots_after"] = len(B), len(A)
    res["_dot_pairs"] = [(B[i], A[j]) for i, j in best.pop("_pairs")]
    best.pop("_cc")
    return best


def _relate(before, after, args, circle_b, circle_a, rmask, paper_b, paper_a, valid_b, valid_a, boxes, res):
    """Step 3: transform AFTER -> POA frame; the other method is kept as a cross-check."""
    feat = _features(before, after, args, paper_b, paper_a, boxes, res["poa_before_px"])
    land = (_landmarks(before, after, args, circle_b, circle_a, rmask, feat["matrix"] if feat["valid"] else None,
                       valid_a, valid_b, res) if args.align != "features" else {"valid": False, "reason": "not used"})
    order = ["landmarks", "features"] if args.align == "dots" else ["features", "landmarks"]
    if args.align == "features":
        order = ["features"]
    methods = {"features": feat, "landmarks": land}
    res["_H_features"] = feat.get("matrix") if feat.get("valid") else feat.get("_rejected_matrix")
    res["_H_landmarks"] = land.get("matrix") if land.get("valid") else None
    chosen = next((m for m in order if methods[m].get("valid")), None)
    if chosen is None:                                   # last resort: rough shift + ECC
        (dx, dy), resp = scope.rough_shift(before, after, circle_b, rmask)
        if resp >= 0.1:
            H, cc = scope.refine_alignment(before, after, np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], float),
                                           rmask, valid_a, valid_b)
            if H is not None:
                res["notes"].append("Frames related by the rough-shift + ECC fallback (no dots, no texture "
                                    "match); check the picture.")
                methods["rough"] = {"valid": True, "matrix": H, "model": "rough shift", "refined": f"ECC {cc:.3f}"}
                chosen = "rough"
    res["alignment"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("matrix", "_rejected_matrix")}
                        for k, v in methods.items()}
    if chosen is None:
        raise Unmeasurable(f"Could not relate the frames: dots ({land.get('reason')}), "
                           f"texture ({feat.get('reason')}).")
    if args.align == "dots" and chosen != "landmarks":
        res["notes"].append(f"Target-Center dots unusable ({land.get('reason')}); used {chosen} instead.")
    res["alignment"]["used"] = chosen
    others = [methods[m]["matrix"] for m in ("landmarks", "features") if m != chosen and methods[m].get("valid")]
    res["after_to_before"] = np.asarray(methods[chosen]["matrix"]).tolist()
    return methods[chosen]["matrix"], (others[0] if others else None)


def _scale(before, args, circle_b, res, after=None, circle_a=None, rmask=None):
    """Step 4: pixels -> inches (or MOA) from a visible reference."""
    if args.px_per_moa:
        return {"px_per_moa": args.px_per_moa, "source": "given (--px-per-moa)"}
    if args.grid_in:
        v, how = scale_ref.from_grid(before, args.grid_in, circle_b, rmask)
        if v is None and after is not None:
            v2, how2 = scale_ref.from_grid(after, args.grid_in, circle_a, rmask)
            v, how = (v2, how2 + ", after frame") if v2 is not None else (None, how)
    elif args.landmark_spacing_in:
        cls = args.scale_class or args.target_class
        if cls == args.target_class and res.get("_dots_before") is not None:
            pts = res["_dots_before"]
        else:
            model = load_detector(args.target_model or args.aim_model)
            pts = landmarks.inside(landmarks.centres(model, before, cls, args.conf), circle_b)
        v, how = scale_ref.from_landmarks(pts, args.landmark_spacing_in, cls)
    elif args.block_size_in:
        model = load_detector(args.block_model or args.target_model or args.aim_model)
        v, how = scale_ref.from_block(model, before, args.block_class, args.block_size_in, args.conf,
                                      circle_b, args.block_aspect)
        if v is None and after is not None:          # same magnification, so the after frame works too
            v2, how2 = scale_ref.from_block(model, after, args.block_class, args.block_size_in, args.conf,
                                            circle_a, args.block_aspect)
            v, how = (v2, how2 + ", after frame") if v2 is not None else (None, how)
    elif args.px_per_inch:
        v, how = args.px_per_inch, "given (--px-per-inch)"
    else:
        res["notes"].append("No scale reference given: results are in pixels "
                            "(--landmark-spacing-in, --block-size-in or --px-per-inch).")
        return {}
    if v is None:
        raise Unmeasurable(f"Scale reference failed: {how}.")
    return {"px_per_in": v, "source": how}


def _new_impact(before, after, H_ab, args, valid_a, valid_b, rmask, res):
    """Step 5: the new impact in AFTER-frame pixels. With an Impact model, frame
    comparison still runs as a cross-check, and takes over if the model's pick
    didn't visibly change the paper (missed or mistaken detection)."""
    if getattr(args, "impact_xy", None):         # marked by hand (testing before the model finds impacts)
        x, y = map(float, args.impact_xy)
        res.update(hole_method="marked by hand", hole_candidates=[], _heat=None, poi_after_px=[x, y])
        return x, y
    diff_best, diff_cands, res["_heat"] = scope.new_hole_by_difference(before, after, H_ab, valid_a, valid_b, rmask,
                                                                       min_area=args.min_hole_px)
    best, cands, res["hole_method"] = diff_best, diff_cands, "frame comparison (stand-in until an Impact model exists)"
    if args.hole_model:
        try:
            m_best, m_cands = scope.new_hole_by_model(load_detector(args.hole_model), args.hole_class,
                                                      before, after, H_ab, args.conf, fixed_mask=rmask)
        except ValueError as e:                    # the model has no such class
            res["notes"].append(f"Impact model not usable ({e}); used frame comparison.")
            m_best, m_cands = None, []
        clear = m_best is not None and (m_best.get("change") or 0) > scope.NEW_DARKENING
        if clear or (m_best is not None and diff_best is None):
            best, cands, res["hole_method"] = m_best, m_cands, f"YOLO {args.hole_class}"
            if not clear:
                res["notes"].append("The Impact detection barely changed the paper (or was hidden before); "
                                    "check the picture.")
            if diff_best is not None:
                gap = float(np.hypot(diff_best["x"] - best["x"], diff_best["y"] - best["y"]))
                res["hole_cross_check_px"] = gap
                if gap > 6:
                    res["notes"].append(f"Impact model and frame comparison disagree by {gap:.0f} px on the "
                                        "impact; check the picture.")
        else:
            res["notes"].append(f"No clearly new '{args.hole_class}' detection; used frame comparison instead.")
    if getattr(args, "grid_in", None) and cands:     # ignore changes off the gridded target (e.g. a hand)
        sq = res.get("_square_px")
        region = scale_ref.grid_region(after, scope.eyepiece_circle(after), rmask, sq)
        if region is not None:
            on = [c for c in cands if region[min(region.shape[0] - 1, int(round(c["y"]))),
                                             min(region.shape[1] - 1, int(round(c["x"])))]]
            if len(on) < len(cands):
                res["notes"].append(f"{len(cands) - len(on)} change(s) off the target ignored (e.g. a hand).")
            cands = on
            best = cands[0] if cands else None
    res["hole_candidates"] = cands[:5]
    if best is None:
        raise Unmeasurable("No new impact found.")
    if res["hole_method"].startswith("frame comparison") and res.get("_heat") is not None:
        # A real new impact is one small change. If much of the target "changed" as strongly,
        # the frames differ in blur/zoom/lighting and frame comparison can't be trusted.
        heat = res["_heat"]
        area = res.get("_target_area_after")
        area = (area > 0) if area is not None else (valid_a > 0)
        busy = float(((heat > 0.5 * best.get("peak", heat.max())) & area).sum()) / max(1, area.sum())
        res["frame_comparison_busy_pct"] = 100 * busy
        if busy > S.MAX_CHANGED_PCT / 100:
            raise Unmeasurable(f"Frame comparison unreliable here: {100 * busy:.1f}% of the target changed as much "
                               "as the best candidate (the frames differ in blur, zoom or lighting). "
                               "An Impact detection is needed.")
    if len(cands) > 1:
        score = lambda c: c.get("strength", c.get("change") or 0)
        if score(cands[1]) / max(score(best), 1e-9) > 0.5:
            res["notes"].append(f"{len(cands)} possible new impacts; the best was chosen but check the picture.")
    if best.get("under_old_reticle", 0) > 0.5:
        res["notes"].append("The chosen impact is where the reticle covered the paper in the point-of-aim "
                            "frame, so an old hole there can't be ruled out; check the picture.")
    if best.get("on_reticle"):
        res["notes"].append("The chosen impact is on the reticle; check the picture.")
    res["poi_after_px"] = [best["x"], best["y"]]
    return best["x"], best["y"]


def _result(poa, poi, H_check, hole_after, scale, args, res):
    """Step 6: displacement + correction, the cross-check, and target-center coordinates."""
    res.update(express(poi[0] - poa[0], -(poi[1] - poa[1]), scale, args))      # image y points down
    res["scale"] = dict(scale)
    k = px_per_moa(scale, args)
    if k:
        res["scale"]["px_per_moa"] = k
    if H_check is not None:                                            # the other method's answer
        poi2 = map_point(H_check, hole_after)
        d_px = float(np.hypot(poi2[0] - poi[0], poi2[1] - poi[1]))
        res["cross_check"] = {"other_method_poi_px": list(poi2), "difference_px": d_px,
                              "difference_moa": d_px / k if k else None}
        if d_px > (args.agree_moa * k if k else 6.0):
            amount = f"{d_px / k:.2f} MOA" if k else f"{d_px:.1f} px"
            res["notes"].append(f"Dot and texture alignment disagree by {amount}; check the picture.")
    dots = res.get("_dots_before") or []
    if dots and k:                                                     # coordinates with a dot as origin
        o = min(dots, key=lambda p: np.hypot(p[0] - poa[0], p[1] - poa[1]))
        res["target_center_origin_px"] = list(o)
        res["aim_vs_target_center_moa"] = [(poa[0] - o[0]) / k, -(poa[1] - o[1]) / k]
        res["impact_vs_target_center_moa"] = [(poi[0] - o[0]) / k, -(poi[1] - o[1]) / k]


def _on_target(img, circle, rmask, args, valid, res, which):
    """Restrict a frame's usable area to the gridded target (plus a margin), so the
    frames are lined up on the paper itself even if the background stays still
    (e.g. a hand-held or swinging target). Unchanged if no clear target area is found."""
    v, _ = scale_ref.from_grid(img, args.grid_in, circle, rmask)
    sq = v * args.grid_in if v else None
    region = scale_ref.grid_region(img, circle, rmask, sq)
    if region is None or (region > 0).sum() < 0.15 * max(1, (valid > 0).sum()):
        return valid
    region = cv2.dilate(region, np.ones((int(sq or 30), int(sq or 30)), np.uint8))
    res[f"_target_area_{which}"] = region
    return cv2.bitwise_and(valid, region)


def measure(before, after, args, plane_img=None, calib_img=None):
    """Steps 1-6. Returns a result dict; res["valid"] says whether it worked and
    res["notes"] explains anything doubtful. Keys starting with "_" are for render().
    (plane_img is accepted for compatibility and ignored.)"""
    res = {"valid": False, "notes": []}
    detectors.ZOOM = getattr(args, "yolo_zoom", 1.0)
    circle_b, circle_a = scope.eyepiece_circle(before), scope.eyepiece_circle(after)
    if circle_b is None:
        res["notes"].append("No round eyepiece view found; using the whole frame.")
    res["eyepiece_before"] = circle_b
    valid_b, valid_a = scope.circle_mask(before.shape, circle_b), scope.circle_mask(after.shape, circle_a)
    try:
        calib_centre = scope.reticle_from_calibration(calib_img)[1] if calib_img is not None else None
        poa, aim_box = _aim_point(before, after, args, calib_centre, res)
        res["poa_before_px"] = [float(poa[0]), float(poa[1])]
        rmask = _reticle_mask(before, args, calib_img, poa, res)
        res["_reticle_mask"] = rmask
        boxes = [aim_box] if aim_box else None
        if args.grid_in:              # line up the TARGET only, not a background that may move differently
            valid_b, valid_a = _on_target(before, circle_b, rmask, args, valid_b, res, "before"), \
                               _on_target(after, circle_a, rmask, args, valid_a, res, "after")
        paper_a = cv2.bitwise_and(valid_a, cv2.bitwise_not(rmask))
        paper_b = cv2.bitwise_and(valid_b, cv2.bitwise_not(rmask))
        H_ab, H_check = _relate(before, after, args, circle_b, circle_a, rmask, paper_b, paper_a, valid_b, valid_a,
                                boxes, res)
        res["recoil_shift_px"] = float(np.hypot(*(np.array(map_point(H_ab, poa)) - np.array(poa))))
        scale = _scale(before, args, circle_b, res, after, circle_a, rmask)
        if scale.get("px_per_in") and getattr(args, "grid_in", None):
            res["_square_px"] = scale["px_per_in"] * args.grid_in
        hole_after = _new_impact(before, after, H_ab, args, valid_a, valid_b, rmask, res)
        poi = map_point(H_ab, hole_after)
        res["poi_before_px"] = list(poi)
        _result(poa, poi, H_check, hole_after, scale, args, res)
        res["valid"] = True
    except Unmeasurable as why:
        res["notes"].append(str(why))
    return res


# --------------------------------------------------------------------------- reporting

def describe(res):
    if not res["valid"]:
        return "NO MEASUREMENT: " + "; ".join(res["notes"])
    u = res["units"]
    h = f"{abs(res['right']):.2f} {u} {'RIGHT' if res['right'] >= 0 else 'LEFT'}"
    v = f"{abs(res['up']):.2f} {u} {'HIGH' if res['up'] >= 0 else 'LOW'}"
    s = f"Impact is {h} and {v} of the point of aim ({res['distance']:.2f} {u} total)"
    if "correction" in res:
        c = res["correction"]
        s += (f"\n  = {abs(res['in_right']):.2f} in {'right' if res['in_right'] >= 0 else 'left'}, "
              f"{abs(res['in_up']):.2f} in {'high' if res['in_up'] >= 0 else 'low'} at {res['distance_yd']:.0f} yd"
              f"\n  CORRECTION: {abs(c['elevation_moa']):.2f} MOA {'UP' if c['elevation_moa'] >= 0 else 'DOWN'} "
              f"({abs(c['elevation_clicks'])} clicks), {abs(c['windage_moa']):.2f} MOA "
              f"{'RIGHT' if c['windage_moa'] >= 0 else 'LEFT'} ({abs(c['windage_clicks'])} clicks) "
              f"[{c['click_moa']:g} MOA/click]")
        if "windage_steps" in c:
            s += f"\n  MOTORS: MOVE {c['windage_steps']} {c['elevation_steps']}   (windage, elevation steps)"
    return s


def render(before, after, res):
    """2x2 picture: before (aim), after (hole), what-changed map, result on the before frame."""
    W, Hh = 640, 360

    def fit(img):
        return cv2.resize(img, (W, Hh), interpolation=cv2.INTER_AREA)
    sx, sy = W / before.shape[1], Hh / before.shape[0]
    S = lambda p: (int(round(p[0] * sx)), int(round(p[1] * sy)))

    b = fit(before)
    tint = fit(cv2.cvtColor(res.get("_reticle_mask", np.zeros(before.shape[:2], np.uint8)), cv2.COLOR_GRAY2BGR))
    b = cv2.addWeighted(b, 1.0, (tint > 0).astype(np.uint8) * np.uint8([0, 0, 120]), 0.6, 0)
    if res.get("eyepiece_before"):
        cx, cy, r = res["eyepiece_before"]
        cv2.circle(b, S((cx, cy)), int(r * sx), (0, 200, 255), 1, cv2.LINE_AA)
    if "poa_before_px" in res:
        cv2.drawMarker(b, S(res["poa_before_px"]), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    paired = {tuple(p) for p, _ in res.get("_dot_pairs", [])}
    for p in res.get("_dots_before", []):
        cv2.circle(b, S(p), 6, (255, 200, 0) if tuple(p) in paired else (120, 120, 120), 2, cv2.LINE_AA)
    label(b, "BEFORE: aim (green), target dots (blue), reticle (red)", (8, 22))

    a = fit(after)
    for i, c in enumerate(res.get("hole_candidates", [])):
        if c.get("box"):
            x1, y1, x2, y2 = c["box"]
            cv2.rectangle(a, S((x1 - 3, y1 - 3)), S((x2 + 3, y2 + 3)), (0, 0, 255) if i == 0 else (0, 165, 255), 2)
    if "poi_after_px" in res:
        cv2.circle(a, S(res["poi_after_px"]), 10, (0, 0, 255), 2, cv2.LINE_AA)
    for _, p in res.get("_dot_pairs", []):
        cv2.circle(a, S(p), 6, (255, 200, 0), 2, cv2.LINE_AA)
    label(a, f"AFTER: new hole (red) by {res.get('hole_method', '?')}", (8, 22))

    heat = res.get("_heat")
    d = fit(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)) if heat is not None else np.full((Hh, W, 3), 30, np.uint8)
    label(d, "What changed between the aligned frames (bright = changed)", (8, 22))

    r_img = fit(before)
    if res["valid"]:
        pa, pi = S(res["poa_before_px"]), S(res["poi_before_px"])
        cv2.drawMarker(r_img, pa, (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(r_img, pi, 9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(r_img, pa, pi, (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.12)
    used = res.get("alignment", {}).get("used")
    label(r_img, f"RESULT: aim (green) -> impact (red); frames related by {used or '?'}", (8, 22))

    grid = np.vstack([np.hstack([b, a]), np.hstack([d, r_img])])
    lines = describe(res).split("\n") + [f"note: {n}" for n in res["notes"]]
    panel = np.full((26 * len(lines) + 14, grid.shape[1], 3), 30, np.uint8)
    for i, t in enumerate(lines):
        col = (120, 255, 120) if (res["valid"] and i == 0) else (220, 220, 220)
        cv2.putText(panel, t, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return np.vstack([grid, panel])


def save(out, before, after, res, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "impact_result.png"), render(before, after, res))
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    if extra:
        clean.update(extra)
    (out / "impact_summary.json").write_text(json.dumps(clean, indent=2, default=float), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if args.video:
        if not (args.before_frame and args.after_frame):
            raise ValueError("With --video, give --before-frame and --after-frame")
        before, after = read_video_frame(args.video, args.before_frame), read_video_frame(args.video, args.after_frame)
    elif args.before and args.after:
        before, after = read_image(args.before), read_image(args.after)
    else:
        raise ValueError("Give --before and --after, or --video with --before-frame and --after-frame")
    calib_img = read_image(args.calibration) if args.calibration else None
    res = measure(before, after, args, None, calib_img)
    print(describe(res))
    for n in res["notes"]:
        print("  note:", n)
    save(args.output, before, after, res)
    print(f"Saved {Path(args.output) / 'impact_result.png'} and {Path(args.output) / 'impact_summary.json'}")
    return res


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
