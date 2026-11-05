"""Scope touchscreen app for the Waveshare 1.28" round touch LCD (240x240).

Replaces the VNC / OpenCV-window workflow of run.py with on-screen buttons:

    MENU -> CHECK      camera, model, grid scale, Arduino (results shown on the LCD)
         -> LIVE VIEW  live eyepiece view, tap BEFORE (aim) then AFTER (measure)
                       if no impact is found you tap it by hand (tap, ZOOM, tap, OK)
                       result screen: MOVE sends it to the turrets, SKIP ignores it
         -> TRACK      automatic shot detection (shot_tracker.py): START, fire, then
                       PROCESS the detected shot (measure, save, send) or DISCARD it
         -> MOTORS     jog the turrets with arrows, STEP cycles the size in MOA

Run:  python3 touchscreen_app.py      (or: python3 run.py touch, after pointing it at this file)
No VNC and no cv2.imshow windows are used, so it works headless.
"""
import sys
import time
import io
import json
import contextlib
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------
# WAVESHARE DRIVERS
# --------------------------------------------------
WAVESHARE = Path.home() / "1.28inch_Touch_LCD_RPI/python"
sys.path.insert(0, str(WAVESHARE))
from lib import LCD_1inch28, Touch_1inch28

# --------------------------------------------------
# YOUR SCOPE SYSTEM (same modules run.py uses)
# --------------------------------------------------
import settings as S
import camera
import scope
import impact
import arduino_link
import shot_tracker as ST
from common import read_image
from detectors import load_detector
from run import check, measure_pair, load_model, open_link, OUT   # importing run.py does NOT start VNC

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
W = H = 240
ANNOTATE_EVERY = 10          # run the model on every Nth live frame to draw the crosshair (0 = never)
TRACK_DRAW_EVERY = 3         # in TRACK, analyse every frame but redraw the LCD only every Nth one
TAP_COOLDOWN = 0.25          # seconds between accepted taps
MOA_STEPS = [0.25, 0.5, 1, 2, 5]

BG = "#0B1422"
BLUE = "#244765"
DARK = "#16263A"
WHITE = "#FFFFFF"
GREEN = "#50D49A"
RED = "#FF6868"
YELLOW = "#FFD866"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(FONT_PATH, 14)
small_font = ImageFont.truetype(FONT_PATH, 11)

lcd = LCD_1inch28.LCD_1inch28()
touch = Touch_1inch28.Touch_1inch28()

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------
page = "menu"
btns = {}                    # buttons of the current page: key -> (label, (x1, y1, x2, y2))
model = None
link = None                  # Arduino link or None
live = {}                    # live-view state
mark = None                  # Marker instance while marking an impact
result = {}                  # result-page state
track = {}                   # shot-tracker state
motor = {"step_idx": 2, "msg": ""}


# --------------------------------------------------
# DRAWING HELPERS
# --------------------------------------------------
def blank():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def ring(d):
    d.ellipse((2, 2, 237, 237), outline="#48617D", width=2)


def draw_buttons(d, buttons, fills=None):
    fills = fills or {}
    for key, (label, (x1, y1, x2, y2)) in buttons.items():
        d.rounded_rectangle((x1, y1, x2, y2), radius=9, fill=fills.get(key, BLUE))
        d.text(((x1 + x2) / 2, (y1 + y2) / 2), label, font=font if (x2 - x1) > 70 else small_font,
               fill=WHITE, anchor="mm")


def hit(buttons, x, y):
    for key, (_, (x1, y1, x2, y2)) in buttons.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None


def show_screen(title, lines=(), buttons=None, fills=None, title_color=WHITE):
    """Generic text screen. lines = [(text, color), ...]"""
    global btns
    btns = buttons or {}
    img, d = blank()
    ring(d)
    d.text((120, 28), title, font=font, fill=title_color, anchor="mm")
    y = 50
    for text, col in lines:
        d.text((34, y), text, font=small_font, fill=col)
        y += 14
    draw_buttons(d, btns, fills)
    lcd.ShowImage(img)


def wrap_lines(text, width=30, color=WHITE):
    out = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw.strip():
            continue
        if set(raw.strip()) == {"="}:
            continue
        parts = textwrap.wrap(raw.strip(), width) or [""]
        up = raw.upper()
        col = color
        if "PROBLEM" in up or "NOT " in up or "COULD NOT" in up or "ERROR" in up:
            col = RED
        elif any(k in up for k in ("LOADED", "FOUND (", "CONNECTED", "ALL SET", "PX PER")):
            col = GREEN
        out += [(p, col) for p in parts]
    return out


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def warp(img, cx, cy, zoom, interp=cv2.INTER_LINEAR):
    """240x240 view of img centred on (cx, cy); zoom = screen pixels per image pixel."""
    M = np.float32([[zoom, 0, W / 2 - cx * zoom], [0, zoom, H / 2 - cy * zoom]])
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderValue=(40, 40, 40))


def eyepiece(frame):
    h, w = frame.shape[:2]
    c = scope.eyepiece_circle(frame)
    return c if c is not None else (w / 2, h / 2, min(w, h) / 2)


# --------------------------------------------------
# MENU
# --------------------------------------------------
def go_menu():
    global page
    page = "menu"
    show_screen("OPTIC ZEROING", buttons={
        "check": ("CHECK", (55, 48, 185, 82)),
        "live": ("LIVE VIEW", (40, 90, 200, 124)),
        "track": ("TRACK", (40, 132, 200, 166)),
        "motors": ("MOTORS", (55, 174, 185, 208)),
    })


# --------------------------------------------------
# CHECK
# --------------------------------------------------
def go_check():
    global page
    page = "check"
    show_screen("CHECKING...", [("camera, model, grid, Arduino", YELLOW)])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = check(None, model)
    except Exception as e:
        buf.write(f"\n  ERROR: {e}")
        ok = False
    text = buf.getvalue()
    print(text)
    lines = wrap_lines(text)[:11]
    show_screen("CHECK OK" if ok else "CHECK: ATTENTION", lines,
                {"back": ("BACK", (75, 198, 165, 230))},
                title_color=GREEN if ok else RED)


# --------------------------------------------------
# LIVE VIEW
# --------------------------------------------------
def go_live():
    global page, live
    page = "live"
    show_screen("STARTING...", [("opening camera", YELLOW)])
    try:
        frames, fps = camera.open_source(S.CAMERA)
    except Exception as e:
        print("Camera error:", e)
        page = "error"
        show_screen("CAMERA ERROR", wrap_lines(str(e), 28)[:8], {"back": ("BACK", (75, 198, 165, 230))},
                    title_color=RED)
        return
    live = dict(frames=frames, n=0, frame=None, circle=None, aim=None, before=None, after=None)


def stop_live():
    global live
    if live.get("frames") is not None:
        try:
            live["frames"].close()
        except Exception:
            pass
    live = {}


def live_buttons():
    return {
        "back": ("BACK", (80, 16, 160, 40)),
        "before": ("BEFORE", (62, 192, 118, 224)),
        "after": ("AFTER", (122, 192, 178, 224)),
    }


def live_tick():
    global page
    L = live
    try:
        _, frame = next(L["frames"])
    except StopIteration:
        stop_live()
        go_menu()
        return
    L["n"] += 1
    L["frame"] = frame
    if L["circle"] is None and L["n"] % 15 == 1:
        L["circle"] = eyepiece(frame)
    cx, cy, r = L["circle"] or eyepiece(frame)
    zoom = W / (2 * r)

    if model is None:
        L["aim_note"] = ("model not loaded (see console)", RED)
    elif ANNOTATE_EVERY and L["n"] % ANNOTATE_EVERY == 0:
        try:
            allf = model.detect_all(frame, S.AIM_CLASS)
            dets = [d for d in allf if d.confidence >= S.MIN_CONFIDENCE]
            L["aim"] = dets[0].pixel_position if dets else None
            best = max((d.confidence for d in allf), default=0.0)
            L["aim_note"] = ((f"crosshair {dets[0].confidence:.2f}", GREEN) if dets else
                             (f"no crosshair (best {best:.2f} < {S.MIN_CONFIDENCE:g})", RED))
        except Exception as e:
            L["aim"] = None
            L["aim_note"] = ("detect error (see console)", RED)
            print("Live detect error:", e)

    img = to_pil(warp(frame, cx, cy, zoom))
    d = ImageDraw.Draw(img)
    if L["aim"] is not None:
        u = (L["aim"][0] - cx) * zoom + W / 2
        v = (L["aim"][1] - cy) * zoom + H / 2
        d.line((u - 12, v, u + 12, v), fill=GREEN, width=2)
        d.line((u, v - 12, u, v + 12), fill=GREEN, width=2)

    status = "BEFORE taken: now fire, tap AFTER" if L["before"] is not None else "aim, then tap BEFORE"
    d.rectangle((30, 168, 210, 184), fill=BG)
    d.text((120, 176), status, font=small_font, fill=GREEN if L["before"] is not None else YELLOW, anchor="mm")
    note = L.get("aim_note")
    if note:
        d.rectangle((50, 44, 190, 58), fill=BG)
        d.text((120, 51), note[0], font=small_font, fill=note[1], anchor="mm")
    b = live_buttons()
    draw_buttons(d, b, {"before": "#2E7D5B"} if L["before"] is not None else None)
    globals()["btns"] = b
    lcd.ShowImage(img)


def live_tap(key):
    L = live
    if key == "back":
        stop_live()
        go_menu()
    elif key == "before" and L.get("frame") is not None:
        L["before"] = L["frame"].copy()
        L["after"] = None
        OUT.joinpath("live").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / "live" / "before.png"), L["before"])
        print("BEFORE taken")
    elif key == "after":
        if L.get("before") is None:
            return
        L["after"] = L["frame"].copy()
        cv2.imwrite(str(OUT / "live" / "after.png"), L["after"])
        do_measure()


# --------------------------------------------------
# MEASURE / MARK / RESULT
# --------------------------------------------------
def do_measure(xy=None):
    show_screen("MEASURING...", [("please wait", YELLOW)])
    try:
        res = measure_pair(live["before"], live["after"], xy)
    except Exception as e:
        print("Measure error:", e)
        res = {"valid": False, "notes": [f"error: {e}"]}
    if not res.get("valid") and xy is None:
        print(impact.describe(res))
        go_mark()
    else:
        go_result(res)


class Marker:
    LEVELS = [1, 2, 4, 8]

    def __init__(self, img):
        cx, cy, r = eyepiece(img)
        self.img = img
        self.home = (cx, cy, W / (2 * r))
        self.cx, self.cy = cx, cy
        self.level = 0
        self.m = None

    @property
    def zoom(self):
        return self.home[2] * self.LEVELS[self.level]

    def tap(self, u, v):
        z = self.zoom
        self.m = (self.cx + (u - W / 2) / z, self.cy + (v - H / 2) / z)

    def cycle(self):
        if self.m is not None:
            self.cx, self.cy = self.m
        self.level = (self.level + 1) % len(self.LEVELS)
        if self.level == 0:
            self.cx, self.cy = self.home[:2]

    def render(self):
        z = self.zoom
        interp = cv2.INTER_NEAREST if self.level >= 2 else cv2.INTER_LINEAR   # see single pixels when zoomed
        img = to_pil(warp(self.img, self.cx, self.cy, z, interp))
        d = ImageDraw.Draw(img)
        if self.m is not None:
            u = (self.m[0] - self.cx) * z + W / 2
            v = (self.m[1] - self.cy) * z + H / 2
            for a, b in (((u - 18, v), (u - 5, v)), ((u + 5, v), (u + 18, v)),
                         ((u, v - 18), (u, v - 5)), ((u, v + 5), (u, v + 18))):
                d.line((*a, *b), fill=RED, width=2)
        d.rectangle((40, 44, 200, 60), fill=BG)
        d.text((120, 52), f"TAP IMPACT  zoom {self.LEVELS[self.level]}x", font=small_font, fill=YELLOW, anchor="mm")
        return img


def mark_buttons():
    return {
        "cancel": ("CANCEL", (80, 16, 160, 40)),
        "zoom": ("ZOOM", (62, 192, 118, 224)),
        "ok": ("OK", (122, 192, 178, 224)),
    }


def go_mark():
    global page, mark
    page = "mark"
    mark = Marker(live["after"])
    draw_mark()


def draw_mark():
    global btns
    img = mark.render()
    d = ImageDraw.Draw(img)
    btns = mark_buttons()
    draw_buttons(d, btns, {"ok": "#2E7D5B"} if mark.m is not None else {"ok": DARK})
    lcd.ShowImage(img)


def mark_tap(x, y):
    key = hit(btns, x, y)
    if key == "cancel":
        go_live_resume()
    elif key == "zoom":
        mark.cycle()
        draw_mark()
    elif key == "ok":
        if mark.m is not None:
            do_measure(mark.m)
    else:
        mark.tap(x, y)
        draw_mark()


def go_result(res, back="live", image=None, sent=False):
    """back = page to return to ("live" or "track"); sent = correction already went to the Arduino."""
    global page, result
    page = "result"
    result = dict(res=res, view="text", back=back, image=image, sent=sent)
    draw_result()


def draw_result():
    global btns
    res = result["res"]
    can_move = bool(res.get("valid")) and link is not None and not result.get("sent")
    btns = {"view": ("VIEW", (80, 16, 160, 40))}
    if can_move:
        btns["move"] = ("MOVE", (62, 192, 118, 224))
        btns["skip"] = ("SKIP", (122, 192, 178, 224))
    else:
        btns["skip"] = ("DONE", (80, 192, 160, 224))

    img, d = blank()
    ring(d)
    if result["view"] == "image":
        p = result.get("image") or OUT / "live" / "result" / "impact_result.png"
        try:
            pic = Image.open(p).convert("RGB")
            pic.thumbnail((W, H))
            img.paste(pic, ((W - pic.width) // 2, (H - pic.height) // 2))
            d = ImageDraw.Draw(img)
        except Exception:
            d.text((120, 120), "no result image", font=small_font, fill=RED, anchor="mm")
    else:
        text = impact.describe(res)
        for n in res.get("notes", []):
            text += "\nnote: " + str(n)
        lines = wrap_lines(text)[:10]
        d.text((120, 56), "RESULT" if res.get("valid") else "NO VALID RESULT", font=font,
               fill=GREEN if res.get("valid") else RED, anchor="mm")
        y = 70
        for t, c in lines:
            d.text((34, y), t, font=small_font, fill=WHITE)
            y += 13
    draw_buttons(d, btns, {"move": "#2E7D5B"})
    lcd.ShowImage(img)


def result_tap(key):
    if key == "view":
        result["view"] = "image" if result["view"] == "text" else "text"
        draw_result()
    elif key == "move":
        show_screen("MOVING...", [("turrets", YELLOW)])
        try:
            msg = arduino_link.send_result(link, result["res"])
        except Exception as e:
            msg = f"error: {e}"
        print("  " + str(msg))
        show_screen("MOVE DONE", wrap_lines(str(msg), 28)[:6])
        time.sleep(1.5)
        leave_result()
    elif key == "skip":
        leave_result()


def leave_result():
    """Leave the result page: back to the TRACK screen or the live view, whichever it came from."""
    global page
    if result.get("back") == "track" and track:
        track["mode"] = "READY"
        track["dirty"] = True
        page = "track"
    else:
        page = "live"


def go_live_resume():
    """Back to the live view with the camera still open (BEFORE is kept)."""
    global page
    page = "live"


# --------------------------------------------------
# TRACK (automatic shot detection, from shot_tracker.py)
# --------------------------------------------------
def track_calibration(frames, args, calib_img, tries=30):
    """Reticle mask + eyepiece circle, like shot_tracker.run_stream(), but patient: skips the first
    frames (exposure settling, as check() does) and keeps looking for the crosshair for a few seconds.
    Returns (first_frame, calib_mask, circle)."""
    for _ in range(10):
        _, first = next(frames)
    calib_mask, circle = None, scope.eyepiece_circle(first)
    if calib_img is not None:
        calib_mask, _, circle_c = scope.reticle_from_calibration(calib_img)
        return first, calib_mask, (circle_c or circle)
    import detectors
    detectors.ZOOM = getattr(args, "yolo_zoom", S.YOLO_ZOOM)      # never fall back to 1.0: it would break LIVE too
    centre, best, det = None, 0.0, None
    zoom_used = detectors.ZOOM
    if args.aim_model:
        det = model if (model is not None and args.aim_model == S.MODEL) else load_detector(args.aim_model)
        for _ in range(tries):
            allf = det.detect_all(first, args.aim_class)
            if allf:
                best = max(best, max(d.confidence for d in allf))
            found = [d for d in allf if d.confidence >= args.conf]
            if found:
                centre = found[0].pixel_position
                break
            _, first = next(frames)
    if centre is None:
        centre = args.aim_xy
    if centre is None:
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "track_first_frame.png"), first)
        sweep = []                                   # does a different model zoom find it?
        if det is not None:
            for z in (1.0, 1.5, 2.0, 3.0):
                detectors.ZOOM = z
                try:
                    allf = det.detect_all(first, args.aim_class)
                    sweep.append(f"{z:g}:{max((d.confidence for d in allf), default=0.0):.2f}")
                except Exception as e:
                    sweep.append(f"{z:g}:err")
                    print("  zoom sweep error:", e)
            detectors.ZOOM = zoom_used
        print(f"Crosshair not found. class={args.aim_class!r} model={args.aim_model!r} conf>={args.conf:g} "
              f"zoom in use={zoom_used:g} (settings YOLO_ZOOM={S.YOLO_ZOOM:g}) frame={first.shape[1]}x{first.shape[0]} "
              f"best={best:.2f} zoom sweep (zoom:best conf) = {' '.join(sweep)} "
              f"frame saved to {OUT / 'track_first_frame.png'}")
        raise ValueError(f"No crosshair. best {best:.2f} (need {args.conf:g}). zoom {zoom_used:g}, "
                         f"settings {S.YOLO_ZOOM:g}. sweep {' '.join(sweep)}")
    return first, scope.reticle_band_mask(first.shape, centre, args.reticle_band_px), circle


def track_new_tracker():
    """Fresh detector, so earlier frames never leak into the next monitoring session."""
    T = track
    a = T["args"]
    an = ST.FrameAnalyzer(T["first"].shape, T["calib_mask"], T["circle"])
    T["tracker"] = ST.ShotTracker(T["fps"], an, buffer_seconds=a.buffer_seconds, spike_px=a.spike_px,
                                  settle_px=a.settle_px,
                                  settle_frames=max(3, int(round(a.settle_seconds * T["fps"]))))
    T["analyzer"] = an


def go_track():
    global page, track
    page = "track"
    show_screen("STARTING...", [("opening camera", YELLOW)])
    frames = None
    try:
        args = ST.parse_args([])
        if not args.video and args.camera is None:
            args.camera = S.CAMERA
        calib_img = read_image(args.calibration) if args.calibration else None
        if not (args.aim_model or args.aim_xy or calib_img is not None):
            raise ValueError("No aim model or calibration in settings.py")
        frames, fps = ST.frames_from(args)
        show_screen("STARTING...", [("looking for crosshair", YELLOW), ("aim at the target", YELLOW)])
        first, calib_mask, circle = track_calibration(frames, args, calib_img)
    except Exception as e:
        print("Track error:", e)
        if frames is not None:
            try:
                frames.close()
            except Exception:
                pass
        page = "error"
        show_screen("TRACK ERROR", wrap_lines(str(e), 28)[:8], {"back": ("BACK", (75, 198, 165, 230))},
                    title_color=RED)
        return
    h, w = first.shape[:2]
    track = dict(args=args, calib_img=calib_img, frames=frames, fps=fps, first=first, calib_mask=calib_mask,
                 circle=circle, view=circle or (w / 2, h / 2, min(w, h) / 2), mode="READY", pending=None,
                 results=[], out=Path(args.output), n=0, frame=first, dirty=True)
    track_new_tracker()


def stop_track():
    """Release the camera and save timeline.png + shots_summary.json (as shot_tracker does at the end)."""
    global track
    T = track
    if not T:
        return
    try:
        T["frames"].close()
    except Exception:
        pass
    try:
        out = T["out"]
        out.mkdir(parents=True, exist_ok=True)
        tr = T["tracker"]
        cv2.imwrite(str(out / "timeline.png"), ST.render_timeline(tr.history, tr.events, T["fps"]))
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in T["results"]]
        (out / "shots_summary.json").write_text(
            json.dumps({"fps": T["fps"], "shots": clean, "frames": list(tr.history)}, indent=2, default=float),
            encoding="utf-8")
    except Exception as e:
        print("Could not save track summary:", e)
    track = {}


def track_buttons():
    mode = track.get("mode")
    if mode == "DETECTING":
        return {"stop": ("STOP", (80, 192, 160, 224))}
    if mode == "REVIEW":
        return {"process": ("PROCESS", (62, 192, 118, 224)), "discard": ("DISCARD", (122, 192, 178, 224))}
    return {"back": ("BACK", (80, 16, 160, 40)), "start": ("START", (80, 192, 160, 224))}


def track_render():
    global btns
    T = track
    cx, cy, r = T["view"]
    img = to_pil(warp(T["frame"], cx, cy, W / (2 * r)))
    d = ImageDraw.Draw(img)
    mode = T["mode"]
    if mode == "DETECTING":
        text, col = f"DETECTING  shots: {len(T['results'])}", GREEN
    elif mode == "REVIEW":
        text, col = f"SHOT! ({T['pending'].get('trigger_reason', '?')}) paused", RED
    else:
        text, col = "ready: tap START", YELLOW
    d.rectangle((30, 168, 210, 184), fill=BG)
    d.text((120, 176), text, font=small_font, fill=col, anchor="mm")
    btns = track_buttons()
    draw_buttons(d, btns, {"start": "#2E7D5B", "stop": "#8A3A3A", "process": "#2E7D5B"})
    lcd.ShowImage(img)


def track_tick():
    T = track
    try:
        idx, f = next(T["frames"])
    except StopIteration:
        stop_track()
        go_menu()
        return
    T["n"] += 1
    T["frame"] = f
    if T["mode"] == "DETECTING":
        ev = T["tracker"].update(idx, f)
        if ev is not None:
            T["pending"] = ev
            T["mode"] = "REVIEW"
            T["dirty"] = True
            print(f"\nShot detected. Frame: {ev.get('trigger_frame')}  Reason: {ev.get('trigger_reason')}")
    if T["dirty"] or T["n"] % TRACK_DRAW_EVERY == 0:
        T["dirty"] = False
        track_render()


def track_process(ev):
    """Measure, save and (if enabled) send one detected shot: same steps as shot_tracker's handle()."""
    T = track
    args = T["args"]
    show_screen("MEASURING...", [("please wait", YELLOW)])
    res = ST.measure_shot(ev, args, T["calib_img"], None)
    res["poa_frame"], res["trigger_frame"] = ev.get("poa_frame"), ev.get("trigger_frame")
    res["after_frames"], res["trigger_reason"] = ev.get("after_frames"), ev.get("trigger_reason")
    k = len(T["results"]) + 1
    d = T["out"] / f"shot_{k:02d}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if ev.get("poa_image") is not None:
            cv2.imwrite(str(d / "poa_frame.jpg"), ev["poa_image"])
            if ev["after_images"]:
                cv2.imwrite(str(d / "after_frame.jpg"), ev["after_images"][0])
            detail = res if res.get("valid") else res.get("_detail")
            if detail and ev["after_images"]:
                cv2.imwrite(str(d / "impact_result.png"), impact.render(ev["poa_image"], ev["after_images"][0], detail))
            pa, pb = ev.get("flow_pair", (None, None))
            if pa is not None and pb is not None:
                cv2.imwrite(str(d / "recoil_flow.png"),
                            ST.render_flow(pb, T["analyzer"].flow_mesh(pa, pb),
                                           f"motion vectors at the shot (frame {ev['trigger_frame']})"))
    except Exception as e:
        print("Could not save shot files:", e)

    sent = None
    if S.MOTORS_ENABLED and link is not None:
        sent = arduino_link.send_result(link, res)
    elif not S.MOTORS_ENABLED:
        print("Motor output disabled in settings.py (correction not sent)")
    else:
        print("No Arduino link (correction not sent)")
    if sent:
        res["arduino"] = sent
    T["results"].append(res)

    print(f"\nSHOT {k}: trigger at frame {ev['trigger_frame']} ({ev['trigger_reason']}), "
          f"point-of-aim frame {ev['poa_frame']}, settled frames {ev.get('after_frames')}")
    print("  " + impact.describe(res).replace("\n", "\n  "))
    for n in res.get("notes", []):
        print("  note:", n)
    if sent:
        print("  " + str(sent))
    go_result(res, back="track", image=d / "impact_result.png", sent=bool(sent))


def track_tap(key):
    T = track
    if key == "back":
        stop_track()
        go_menu()
    elif key == "start":
        track_new_tracker()
        T["pending"] = None
        T["mode"] = "DETECTING"
        T["dirty"] = True
        print("Monitoring enabled.")
    elif key == "stop":
        T["mode"] = "READY"
        T["dirty"] = True
        print("Monitoring disabled.")
    elif key == "process":
        ev, T["pending"] = T["pending"], None
        T["mode"] = "READY"
        T["dirty"] = True
        if ev is not None:
            track_process(ev)
    elif key == "discard":
        T["pending"] = None
        T["mode"] = "READY"
        T["dirty"] = True
        print("Event discarded (not processed, nothing sent).")


# --------------------------------------------------
# MOTORS
# --------------------------------------------------
def go_motors():
    global page
    page = "motors"
    draw_motors()


def draw_motors():
    step = MOA_STEPS[motor["step_idx"]]
    b = {
        "up": ("UP", (95, 50, 145, 95)),
        "left": ("L", (35, 100, 90, 140)),
        "step": (f"{step:g}", (95, 100, 145, 140)),
        "right": ("R", (150, 100, 205, 140)),
        "down": ("DN", (95, 145, 145, 190)),
        "back": ("BACK", (75, 196, 165, 226)),
    }
    title = "MOTORS" if link is not None else "NO ARDUINO"
    lines = [(t, GREEN) for t, _ in wrap_lines(motor["msg"], 30)[:1]] if motor["msg"] else []
    show_screen(title + f"  step {step:g} MOA", lines, b, {"step": "#2E7D5B"},
                title_color=WHITE if link is not None else RED)


def motors_tap(key):
    if key == "back":
        go_menu()
        return
    if key == "step":
        motor["step_idx"] = (motor["step_idx"] + 1) % len(MOA_STEPS)
        draw_motors()
        return
    if link is None:
        return
    step = MOA_STEPS[motor["step_idx"]]
    dx, dy = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[key]
    w, e = arduino_link.motor_steps(dx * step, dy * step)
    print(f"MOVE {w} {e}")
    try:
        r = link.move(w, e)
        motor["msg"] = str(r[1])
    except Exception as ex:
        motor["msg"] = f"error: {ex}"
    print("  " + motor["msg"])
    draw_motors()


# --------------------------------------------------
# TOUCH DISPATCH
# --------------------------------------------------
def handle_touch(x, y):
    if page == "menu":
        key = hit(btns, x, y)
        if key == "check":
            go_check()
        elif key == "live":
            go_live()
        elif key == "track":
            go_track()
        elif key == "motors":
            go_motors()
    elif page == "track":
        key = hit(track_buttons(), x, y)
        if key:
            track_tap(key)
    elif page == "live":
        key = hit(live_buttons(), x, y)
        if key:
            live_tap(key)
    elif page == "mark":
        mark_tap(x, y)
    elif page == "result":
        key = hit(btns, x, y)
        if key:
            result_tap(key)
    elif page == "motors":
        key = hit(btns, x, y)
        if key:
            motors_tap(key)
    else:                                   # check, error screens: only BACK
        if hit(btns, x, y) == "back":
            go_menu()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    global model, link
    try:
        lcd.Init()
        touch.init()
        touch.Set_Mode(1)

        show_screen("STARTING", [("loading model...", YELLOW)])
        try:
            model = load_model()
        except Exception as e:
            print("Model load problem:", e)
            model = None
        link = open_link(quiet=True)
        go_menu()

        print("Touchscreen app running. Ctrl+C to exit.")
        was_pressed = False
        last_tap = 0.0

        while True:
            if page == "live" and live:
                try:
                    live_tick()
                except Exception as e:
                    print("Live error:", e)
                    stop_live()
                    show_screen("LIVE ERROR", wrap_lines(str(e), 28)[:8],
                                {"back": ("BACK", (75, 198, 165, 230))}, title_color=RED)
                    globals()["page"] = "error"
            elif page == "track" and track:
                try:
                    track_tick()
                except Exception as e:
                    print("Track error:", e)
                    stop_track()
                    show_screen("TRACK ERROR", wrap_lines(str(e), 28)[:8],
                                {"back": ("BACK", (75, 198, 165, 230))}, title_color=RED)
                    globals()["page"] = "error"

            pressed = touch.GPIO.input(touch.TP_INT) == 0
            if pressed and not was_pressed:
                touch.get_point()
                x, y = touch.X_point, touch.Y_point
                now = time.monotonic()
                if 0 <= x < W and 0 <= y < H and now - last_tap > TAP_COOLDOWN:
                    last_tap = now
                    print(f"Touch: X={x}, Y={y}  page={page}")
                    try:
                        handle_touch(x, y)
                    except Exception as e:
                        print("Handler error:", e)
                        stop_live()
                        stop_track()
                        show_screen("ERROR", wrap_lines(str(e), 28)[:8],
                                    {"back": ("BACK", (75, 198, 165, 230))}, title_color=RED)
                        globals()["page"] = "error"
            was_pressed = pressed
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nClosing.")
    finally:
        stop_live()
        stop_track()
        if link:
            try:
                link.close()
            except Exception:
                pass
        try:
            lcd.module_exit()
        except Exception as e:
            print(f"LCD cleanup error: {e}")
        print("Program closed.")


if __name__ == "__main__":
    main()
