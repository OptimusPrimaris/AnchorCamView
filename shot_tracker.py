"""Watch a scope-camera feed, detect each shot, and measure aim vs impact.

Frame by frame (works the same on a video file or a live camera):

 1. MOTION   How far the target moved since the previous frame, from phase
             correlation (one overall shift; fast) on the paper only: the
             reticle is filled in and the eyepiece ring cropped away, so they
             can't pin the answer at "no movement".
 2. BLUR     A sharpness score (variance of the Laplacian = how much fine
             detail the frame has). Recoil smears the target -> sharpness drops.
 3. BUFFER   The last few seconds of frames are kept (ring buffer).

States:  AIMING --spike--> RECOIL --quiet & sharp for a while--> SETTLED -> measure -> AIMING

 * Shot detected: motion jumps far above the aiming wobble, or sharpness collapses.
 * Point-of-aim frame: picked RETROACTIVELY from the buffer = the last frame
   before the spike (the aim at trigger break), stepping back a frame if
   that one already shows blur.
 * After frame(s): once motion stays small and frames are sharp again. The
   impact is measured against the sharpest few settled frames and must agree
   ("confirmed"); the median is reported.
 * The measurement itself is impact.py: homography between the POA frame and
   the settled frame on the target paper, new hole (frame difference or your
   impact model), offset in inches / MOA.

  python shot_tracker.py --video range.mp4 --aim-model models/crosshair_target_v1.pt --landmark-spacing-in 3
  python shot_tracker.py --camera 0 --show ...same options...                          # live camera

Simulated clips with known answers: tests/test_tracker.py
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from common import read_image
import arduino_link
import camera
import impact
import settings as S
from detectors import load_detector
import scope


# --------------------------------------------------------------------------- per-frame analysis

class FrameAnalyzer:
    """Motion and sharpness of each frame, measured on the target paper only."""

    def __init__(self, shape, calib_mask=None, circle=None, size=256):
        h, w = shape[:2]
        if circle is None:
            circle = (w / 2, h / 2, min(w, h) / 2)
        cx, cy, r = circle
        half = r / np.sqrt(2) * 0.95                       # square inside the eyepiece circle
        self.box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half)))
        x1, y1, x2, y2 = self.box
        self.size = size
        self.scale = (x2 - x1) / float(size)               # full-res px per analysis px
        m = np.zeros((h, w), np.uint8) if calib_mask is None else calib_mask
        self.mask = cv2.resize(m[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST)
        self.mask = cv2.dilate(self.mask, np.ones((3, 3), np.uint8))
        self.window = cv2.createHanningWindow((size, size), cv2.CV_32F)
        self.prev = None

    def prepare(self, frame):
        x1, y1, x2, y2 = self.box
        g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.mask.any():
            g = cv2.inpaint(g, self.mask, 3, cv2.INPAINT_TELEA)    # hide the reticle
        return g

    def update(self, frame):
        g = self.prepare(frame)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        if self.prev is None:
            shift, resp = (0.0, 0.0), 1.0
        else:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev.astype(np.float32), g.astype(np.float32), self.window)
            shift = (dx * self.scale, dy * self.scale)
        self.prev = g
        return {"shift": shift, "motion": float(np.hypot(*shift)), "response": float(resp), "sharpness": sharp}

    def flow_mesh(self, frame_a, frame_b, step=16):
        """Dense optical flow between two frames on a grid: the 'mesh' of motion vectors."""
        a, b = self.prepare(frame_a), self.prepare(frame_b)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, 0)
        x1, y1, _, _ = self.box
        pts = []
        for yy in range(step // 2, self.size, step):
            for xx in range(step // 2, self.size, step):
                fx, fy = flow[yy, xx]
                pts.append((x1 + xx * self.scale, y1 + yy * self.scale, fx * self.scale, fy * self.scale))
        return pts


# --------------------------------------------------------------------------- the state machine

class ShotTracker:
    def __init__(self, fps, analyzer, buffer_seconds=1.0, settle_frames=None, min_aim_frames=10,
                 spike_px=6.0, settle_px=2.5, timeout_s=6.0):
        self.fps = fps
        self.an = analyzer
        self.history = []                                   # stats of every frame (small; kept all session)
        self.state = "AIMING"
        self.aim_start = 0
        self.settle_frames = settle_frames or max(5, int(round(0.25 * fps)))
        # Full frames are only needed briefly: the point-of-aim frame is copied out the moment
        # the shot is detected, and the settled frames are the last `settle_frames` frames.
        self.buffer = deque(maxlen=max(int(buffer_seconds * fps), self.settle_frames + 6))
        self.min_aim_frames = min_aim_frames
        self.spike_px, self.settle_px = spike_px, settle_px
        self.timeout = int(timeout_s * fps)
        self.shot = None
        self.events = []

    def _baseline(self):
        aim = [h for h in self.history[self.aim_start:] if h["state"] == "AIMING"][-int(2 * self.fps):]
        if len(aim) < self.min_aim_frames:
            return None
        m = np.array([h["motion"] for h in aim[1:]] or [0.0])
        s = np.array([h["sharpness"] for h in aim])
        med = float(np.median(m))
        mad = float(np.median(np.abs(m - med))) * 1.4826 + 1e-6
        return {"motion_med": med, "motion_mad": mad, "sharp_med": float(np.median(s))}

    def update(self, index, frame):
        st = self.an.update(frame)
        st.update(index=index, time_s=index / self.fps)
        self.buffer.append((index, frame))
        event = None

        if self.state == "AIMING":
            base = self._baseline()
            st["state"] = "AIMING"
            if base is not None:
                limit = max(self.spike_px, base["motion_med"] + 10 * base["motion_mad"])
                blurred = st["sharpness"] < 0.55 * base["sharp_med"]
                if st["motion"] > limit or blurred or st["response"] < 0.05:
                    st["state"] = "RECOIL"
                    self.state = "RECOIL"
                    self.shot = {"trigger_frame": index, "baseline": base,
                                 "trigger_reason": "motion" if st["motion"] > limit else "blur"}
                    self.shot["poa_frame"] = self._pick_poa(index, base)
                    # keep the frames this shot needs now, before they leave the short buffer
                    self.shot["poa_image"] = self.frame(self.shot["poa_frame"])
                    self.shot["flow_pair"] = (self.frame(index - 1), frame)
        elif self.state == "RECOIL":
            st["state"] = "RECOIL"
            base = self.shot["baseline"]
            quiet = st["motion"] < max(self.settle_px, 3 * base["motion_med"]) and st["response"] > 0.1
            sharp = st["sharpness"] > 0.8 * base["sharp_med"]
            self.shot["quiet_run"] = self.shot.get("quiet_run", 0) + 1 if (quiet and sharp) else 0
            if self.shot["quiet_run"] >= self.settle_frames:
                st["state"] = "SETTLED"
                run = self.history[-(self.settle_frames - 1):] + [st]
                best = sorted(run, key=lambda h: -h["sharpness"])[:3]
                self.shot["after_frames"] = sorted(h["index"] for h in best)
                self.shot["settled_frame"] = index
                event = self._finish()
            elif index - self.shot["trigger_frame"] > self.timeout:
                st["state"] = "TIMEOUT"
                self.shot["error"] = "View never settled after the shot"
                event = self._finish()
        self.history.append(st)
        return event

    def _pick_poa(self, trigger_index, base):
        """Last frame before the spike; step back if it already shows blur/motion."""
        prior = [h for h in self.history if h["index"] < trigger_index][-3:]
        for h in reversed(prior):
            if h["sharpness"] >= 0.85 * base["sharp_med"] and h["motion"] < max(self.spike_px, base["motion_med"] + 6 * base["motion_mad"]):
                return h["index"]
        return trigger_index - 1

    def frame(self, index):
        for i, f in self.buffer:
            if i == index:
                return f
        return None

    def _finish(self):
        shot = self.shot
        shot["after_images"] = [self.frame(i) for i in shot.get("after_frames", [])]
        self.events.append(shot)
        self.state, self.shot = "AIMING", None
        self.aim_start = len(self.history) + 1
        return shot


# --------------------------------------------------------------------------- measuring a detected shot

def measure_shot(shot, args, calib_img, plane_img=None):
    """Run impact.measure between the POA frame and each settled frame; confirm agreement."""
    if shot.get("error") or shot["poa_image"] is None:
        return {"valid": False, "notes": [shot.get("error", "Point-of-aim frame was not captured")]}
    results = [impact.measure(shot["poa_image"], img, args, plane_img, calib_img)
               for img in shot["after_images"] if img is not None]
    good = [r for r in results if r["valid"]]
    if not good:
        notes = results[0]["notes"] if results else ["No settled frames"]
        return {"valid": False, "notes": notes, "_detail": results[0] if results else None}
    px = np.array([r["displacement_px"] for r in good])          # right/up in measuring-plane pixels
    mid = int(np.argsort(px.sum(axis=1))[len(good) // 2])
    res = dict(good[mid])
    res.update(impact.express(float(np.median(px[:, 0])), float(np.median(px[:, 1])), res.get("scale", {}), args))
    res["confirm_frames"] = len(good)
    spread_px = float(max(np.ptp(px[:, 0]), np.ptp(px[:, 1]))) if len(good) > 1 else 0.0
    k = res.get("scale", {}).get("px_per_moa")
    res["confirm_spread"] = spread_px / k if k else spread_px
    tol = 0.25 if k else 3.0                                          # MOA, or px without a scale
    if len(good) < 2:
        res["notes"] = res["notes"] + ["Impact seen in only one settled frame (not confirmed)."]
    elif res["confirm_spread"] > tol:
        res["notes"] = res["notes"] + [f"Settled frames disagree by {res['confirm_spread']:.2f} {res['units']}; check the picture."]
    return res


# --------------------------------------------------------------------------- pictures

def render_timeline(history, events, fps, width=1280, height=360):
    """Motion (px/frame) and sharpness over time, with POA / settled frames marked."""
    img = np.full((height, width, 3), 28, np.uint8)
    if not history:
        return img
    n = len(history)
    pad_l, pad_r, top, mid, bot = 60, 20, 30, height // 2 + 5, height - 30
    X = lambda i: int(pad_l + (width - pad_l - pad_r) * i / max(1, n - 1))
    motion = np.array([h["motion"] for h in history])
    sharp = np.array([h["sharpness"] for h in history])
    colors = {"AIMING": (60, 60, 60), "RECOIL": (40, 40, 120), "SETTLED": (40, 110, 40), "TIMEOUT": (0, 0, 160)}
    for i, h in enumerate(history):
        cv2.line(img, (X(i), top), (X(i), bot), colors.get(h["state"], (60, 60, 60)), max(1, (width - 80) // n))
    mmax = max(10.0, float(np.percentile(motion, 99)))
    smax = max(1.0, float(sharp.max()))
    pm = [(X(i), int(mid - 8 - (mid - 8 - top) * min(v, mmax) / mmax)) for i, v in enumerate(motion)]
    ps = [(X(i), int(bot - (bot - mid - 8) * v / smax)) for i, v in enumerate(sharp)]
    cv2.polylines(img, [np.int32(pm)], False, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.int32(ps)], False, (255, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(img, f"motion (px/frame, max {mmax:.0f})", (pad_l + 5, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "sharpness", (pad_l + 5, mid + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80), 1, cv2.LINE_AA)
    for k, ev in enumerate(events):
        for idx, col, name in ((ev.get("poa_frame"), (0, 255, 0), f"POA {k + 1}"),
                               (ev.get("settled_frame"), (0, 0, 255), "settled")):
            if idx is None:
                continue
            j = next((i for i, h in enumerate(history) if h["index"] == idx), None)
            if j is not None:
                cv2.line(img, (X(j), top), (X(j), bot), col, 2)
                cv2.putText(img, name, (X(j) + 3, bot + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    t_end = history[-1]["index"] / fps
    cv2.putText(img, f"0 s", (pad_l, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"{t_end:.1f} s", (width - pad_r - 50, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "background: grey = aiming, red = recoil, green = settled", (width - 470, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def render_flow(frame, mesh, title):
    img = frame.copy()
    mags = [np.hypot(fx, fy) for _, _, fx, fy in mesh]
    top = max(1.0, float(np.percentile(mags, 95))) if mags else 1.0
    k = 40.0 / top                                          # longest arrows ~40 px
    for x, y, fx, fy in mesh:
        cv2.arrowedLine(img, (int(x), int(y)), (int(x + fx * k), int(y + fy * k)), (0, 255, 255), 1,
                        cv2.LINE_AA, tipLength=0.3)
    cv2.rectangle(img, (5, 5), (5 + 11 * len(title), 30), (25, 25, 25), -1)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- running

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="Recorded clip from the scope camera")
    p.add_argument("--camera", default=None, help="Live camera: 0, 1 ... (USB) or picamera (default: settings.CAMERA)")
    p.add_argument("--buffer-seconds", type=float, default=S.BUFFER_SECONDS,
                   help="Full frames kept in memory (the point-of-aim frame is saved at the shot)")
    p.add_argument("--spike-px", type=float, default=S.SPIKE_PX, help="Minimum per-frame jump that counts as a shot")
    p.add_argument("--settle-px", type=float, default=S.SETTLE_PX, help="Per-frame motion below this counts as settled")
    p.add_argument("--settle-seconds", type=float, default=S.SETTLE_SECONDS, help="How long it must stay settled")
    p.add_argument("--show", action="store_true", help="Show the feed with the tracker state (live use)")
    p.add_argument("--output", default="output/shots")
    p.add_argument("--serial-port", default=S.SERIAL_PORT, help="Send each correction to an Arduino on this port (COM3, /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=S.BAUD)
    # measurement options are impact.py's
    known, rest = p.parse_known_args(argv)
    m = impact.parse_args(rest)
    for k, v in vars(m).items():
        if not hasattr(known, k):
            setattr(known, k, v)
    return known


def frames_from(args):
    """The video file, or the live camera (USB index or "picamera"; see camera.py)."""
    return camera.open_source(args.video if args.video else args.camera)


# --------------------------------------------------------------------------- operator interface

# Operator interface states
STANDBY = "STANDBY"
READY_PROMPT = "READY_PROMPT"
DETECTING = "DETECTING"
REVIEW = "REVIEW"


def operator_prompt(frame, message):
    """Display a Yes/No prompt (blocking test function)."""

    display = frame.copy()

    cv2.rectangle(display, (10, 45), (900, 115), (30, 30, 30), -1)

    cv2.putText(display, message, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Operator Interface", display)

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("y"):
            return True
        if key == ord("n"):
            return False
        if key == ord("q"):
            return None


def draw_operator_status(frame, mode):
    """Display operator status over the live camera image."""

    display = frame.copy()

    messages = {
        READY_PROMPT: "READY?  Y = YES   N = NO",
        DETECTING: "DETECTING...  N = STOP",
        REVIEW: "SHOT DETECTED - MONITORING PAUSED",
    }

    message = messages.get(mode, mode)

    # Draw status banner
    cv2.rectangle(display, (5, 5), (display.shape[1] - 5, 60), (30, 30, 30), -1)
    cv2.putText(display, message, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Additional information during review
    if mode == REVIEW:

        cv2.rectangle(display, (10, 80), (500, 245), (30, 30, 30), -1)

        lines = [
            "EVENT NOTIFICATION",
            "Monitoring paused",
            "Awaiting operator response",
            "Y = PROCESS SHOT  N = DISCARD"
        ]

        for i, line in enumerate(lines):
            cv2.putText(display, line, (25, 115 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return display


def process_event_notification(event):
    """Convert a detected event into a message for the operator.

    The full tracker event is kept under "event" so it can be processed
    (measured, saved, sent to the Arduino) once the operator presses Y.
    """
    return {
        "event_detected": True,
        "frame": event.get("trigger_frame"),
        "reason": event.get("trigger_reason", "unknown"),
        "event": event,
    }


def run_stream(frames, fps, calib_img, args, plane_img=None, out=None, quiet=False, link=None):
    first_index, first = next(frames)
    calib_mask, circle = None, scope.eyepiece_circle(first)
    if calib_img is not None:
        calib_mask, _, circle_c = scope.reticle_from_calibration(calib_img)
        circle = circle_c or circle
    else:                                       # bands through the YOLO crosshair in the first frame
        import detectors
        detectors.ZOOM = getattr(args, "yolo_zoom", 1.0)
        centre = None
        if args.aim_model:
            found = [d for d in load_detector(args.aim_model).detect_all(first, args.aim_class)
                     if d.confidence >= args.conf]
            centre = found[0].pixel_position if found else None
        centre = centre or args.aim_xy
        if centre is None:
            raise ValueError(f"No '{args.aim_class}' in the first frame (or no --aim-model given).")
        calib_mask = scope.reticle_band_mask(first.shape, centre, args.reticle_band_px)
    analyzer = FrameAnalyzer(first.shape, calib_mask, circle)
    tracker = ShotTracker(fps, analyzer, buffer_seconds=args.buffer_seconds, spike_px=args.spike_px,
                          settle_px=args.settle_px, settle_frames=max(3, int(round(args.settle_seconds * fps))))
    results = []

    def handle(ev):
        """Measure, save, and send one detected shot (same process as the original script)."""
        res = measure_shot(ev, args, calib_img, plane_img)
        res["poa_frame"], res["trigger_frame"] = ev.get("poa_frame"), ev.get("trigger_frame")
        res["after_frames"], res["trigger_reason"] = ev.get("after_frames"), ev.get("trigger_reason")
        k = len(results) + 1
        if out is not None:
            d = out / f"shot_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            if ev.get("poa_image") is not None:
                cv2.imwrite(str(d / "poa_frame.jpg"), ev["poa_image"])
                if ev["after_images"]:
                    cv2.imwrite(str(d / "after_frame.jpg"), ev["after_images"][0])
                detail = res if res.get("valid") else res.get("_detail")
                if detail and ev["after_images"]:
                    cv2.imwrite(str(d / "impact_result.png"), impact.render(ev["poa_image"], ev["after_images"][0], detail))
                trig = ev["trigger_frame"]
                pa, pb = ev.get("flow_pair", (None, None))
                if pa is not None and pb is not None:
                    cv2.imwrite(str(d / "recoil_flow.png"),
                                render_flow(pb, analyzer.flow_mesh(pa, pb), f"motion vectors at the shot (frame {trig})"))

        # Motor guard: everything above is always processed and recorded,
        # but the correction is only sent to the Arduino (arduino_link.py)
        # when S.MOTORS_ENABLED is True in settings.py.
        sent = None
        if S.MOTORS_ENABLED:
            sent = arduino_link.send_result(link, res)
        else:
            print("Motor output disabled in settings.py (correction not sent)")
        if sent:
            res["arduino"] = sent
        results.append(res)

        if not quiet:
            print(f"\nSHOT {k}: trigger at frame {ev['trigger_frame']} ({ev['trigger_reason']}), "
                  f"point-of-aim frame {ev['poa_frame']}, settled frames {ev.get('after_frames')}")
            print("  " + impact.describe(res).replace("\n", "\n  "))
            if res.get("valid"):
                print(f"  confirmed in {res['confirm_frames']} settled frame(s), spread {res['confirm_spread']:.3f} {res['units']}")
            for n in res.get("notes", []):
                print("  note:", n)
            if sent:
                print("  " + sent)

    # Start in operator-ready mode
    operator_mode = READY_PROMPT
    last_action_time = 0.0
    pending_event = None

    print("Track interface started.")
    print("Y = Yes | N = No | Q = Quit")

    try:
        for idx, f in frames:

            # Run the shot tracker only while monitoring is enabled.
            if operator_mode == DETECTING:

                ev = tracker.update(idx, f)

                if ev is not None:
                    pending_event = process_event_notification(ev)
                    operator_mode = REVIEW

                    print("\nShot detected.")
                    print(f"Frame: {pending_event['frame']}")
                    print(f"Reason: {pending_event['reason']}")
                    print("Monitoring paused. Y = process shot, N = discard.")

            # Keep displaying the camera regardless of operator state.
            if args.show:

                view = draw_operator_status(f, operator_mode)

                # Display event details during REVIEW.
                if operator_mode == REVIEW and pending_event is not None:

                    cv2.putText(view, f"Frame: {pending_event['frame']}", (25, 275),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(view, f"Reason: {pending_event['reason']}", (25, 310),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                cv2.imshow("shot tracker (Q quits)", view)

                key = cv2.waitKey(30) & 0xFF

                # Keyboard debounce
                now = time.monotonic()

                if key in (ord("y"), ord("n")):
                    if now - last_action_time < 0.75:
                        key = 255
                    else:
                        last_action_time = now

                if key == ord("q"):
                    break

                if operator_mode == READY_PROMPT:

                    if key == ord("y"):

                        # Reset detector so previous frames do not
                        # become part of the next monitoring session.
                        analyzer = FrameAnalyzer(first.shape, calib_mask, circle)

                        tracker = ShotTracker(
                            fps,
                            analyzer,
                            buffer_seconds=args.buffer_seconds,
                            spike_px=args.spike_px,
                            settle_px=args.settle_px,
                            settle_frames=max(3, int(round(args.settle_seconds * fps)))
                        )

                        pending_event = None
                        operator_mode = DETECTING

                        print("Monitoring enabled.")

                    elif key == ord("n"):
                        print("Monitoring remains disabled.")

                elif operator_mode == DETECTING:

                    if key == ord("n"):

                        operator_mode = READY_PROMPT
                        pending_event = None
                        print("Monitoring disabled.")

                elif operator_mode == REVIEW:

                    if key == ord("y"):

                        print("Processing shot...")
                        handle(pending_event["event"])
                        pending_event = None
                        operator_mode = READY_PROMPT

                    elif key == ord("n"):

                        print("Event discarded (not processed, nothing sent).")
                        pending_event = None
                        operator_mode = READY_PROMPT

    except KeyboardInterrupt:                  # Ctrl+C: stop cleanly and still save everything
        if not quiet:
            print("\nStopped (Ctrl+C). Saving what was recorded...")

    finally:
        if hasattr(frames, "close"):
            frames.close()                     # releases the camera

    if args.show:
        cv2.destroyAllWindows()

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "timeline.png"), render_timeline(tracker.history, tracker.events, fps))
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        hist = [{k: v for k, v in h.items()} for h in tracker.history]
        (out / "shots_summary.json").write_text(json.dumps({"fps": fps, "shots": clean, "frames": hist},
                                                           indent=2, default=float), encoding="utf-8")
    return results, tracker


def main(argv=None):
    args = parse_args(argv)
    if not args.video and args.camera is None:
        args.camera = S.CAMERA
    if not (args.video or args.camera is not None):
        raise ValueError("Give --video clip.mp4 or --camera 0  (simulated clips: tests/test_tracker.py)")
    calib_img = read_image(args.calibration) if args.calibration else None
    if not (args.aim_model or args.aim_xy or calib_img is not None):
        raise ValueError("Give --aim-model (with the Crosshair-Center class).")
    plane_img = None
    link = arduino_link.ArduinoLink(args.serial_port, args.baud) if args.serial_port else None
    stream, fps = frames_from(args)
    try:
        results, _ = run_stream(stream, fps, calib_img, args, plane_img, Path(args.output), link=link)
    finally:
        if link:
            link.close()
    print(f"\n{len(results)} shot(s). Saved timeline.png, shots_summary.json and shot_XX folders in {args.output}")
    return results


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
