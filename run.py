"""Scope camera test program: one entry point for testing on the Raspberry Pi.

    python3 run.py              menu
    python3 run.py check        camera, model, grid scale and Arduino: are they all working?
    python3 run.py live         live view; b = BEFORE, a = AFTER -> measure (+ click the impact by hand)
    python3 run.py track        detect shots automatically and measure them (the full system)
    python3 run.py motors       move the turrets by hand (MOA or steps)

    python run.py live --video videos/camera_videoall.mp4     try it on a recording (e.g. on a PC)

All values (camera, model, distance, motors, serial port ...) come from settings.py.
Run inside the .venv made by setup_pi.py:  .venv/bin/python run.py
"""
import sys
import time
from pathlib import Path
import cv2
from networkx import display
import numpy as np
from tomlkit import key
import os
import socket
import subprocess
#from win32timezone import now
import settings as S

OUT = Path("output")
GREEN, BLUE, RED, YELLOW, CYAN = (0, 220, 0), (255, 150, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)


def banner(text):
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def load_model():
    import detectors
    detectors.ZOOM = S.YOLO_ZOOM
    return detectors.load_detector(S.MODEL)


def open_link(quiet=False):
    """The Arduino link, or None (with the reason printed) if it can't be opened."""
    if not S.SERIAL_PORT:
        if not quiet:
            print("  Arduino: SERIAL_PORT is not set in settings.py (e.g. \"/dev/ttyACM0\"), so nothing will move.")
        return None
    try:
        import arduino_link
        return arduino_link.ArduinoLink(S.SERIAL_PORT, S.BAUD)
    except Exception as e:                                   # port missing, permission, cable
        print(f"  Arduino: could not open {S.SERIAL_PORT}: {e}")
        return None


# --------------------------------------------------------------------------- check
def check(camera_source=None, model=None):
    banner("CHECK: camera, model, grid scale, Arduino")
    import camera, scale, scope, impact, arduino_link
    print(f"  distance {S.DISTANCE_YD:g} yd | grid square {S.GRID_SQUARE_IN} in | click {S.CLICK_MOA} MOA | "
          f"{arduino_link.steps_per_moa():.1f} motor steps per MOA")
    ok = True
    try:
        model = model or load_model()
        print(f"  model: {S.MODEL} loaded")
    except Exception as e:
        print(f"  MODEL PROBLEM: {e}")
        return False
    try:
        frames, fps = camera.open_source(camera_source if camera_source is not None else S.CAMERA)
        for _ in range(10):                                  # skip the first frames (exposure settling)
            i, frame = next(frames)
        frames.close()
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "check_frame.png"), frame)
        src = camera_source if camera_source is not None else S.CAMERA
        print(f"  camera {src!r}: {frame.shape[1]} x {frame.shape[0]} at {fps:g} fps "
              f"(saved {OUT / 'check_frame.png'})")
    except Exception as e:
        print(f"  CAMERA PROBLEM: {e}\n    -> check CAMERA in settings.py (0 = USB, \"picamera\" = camera module)")
        return False
    t = time.time()
    aim = [d for d in model.detect_all(frame, S.AIM_CLASS) if d.confidence >= S.MIN_CONFIDENCE]
    ms = 1000 * (time.time() - t)
    try:
        dots = [d for d in model.detect_all(frame, S.TARGET_CLASS) if d.confidence >= S.MIN_CONFIDENCE]
    except ValueError:
        dots = []
    print(f"  model look: {ms:.0f} ms | crosshair: " +
          (f"found ({aim[0].confidence:.2f}) at ({aim[0].pixel_position[0]:.0f}, {aim[0].pixel_position[1]:.0f})"
           if aim else "NOT found (aim at the target)") + f" | target dots: {len(dots)}")
    ok &= bool(aim)
    if S.GRID_SQUARE_IN:
        v, how = scale.from_grid(frame, S.GRID_SQUARE_IN, scope.eyepiece_circle(frame))
        k = v * impact.in_per_moa(S.DISTANCE_YD) if v else None
        print("  grid scale: " + (f"{v:.1f} px per inch -> {k:.1f} px per MOA, 1 click = {k * S.CLICK_MOA:.1f} px"
                                  if v else f"NOT found ({how})"))
        ok &= bool(v)
    link = open_link()
    if link:
        print(f"  Arduino: connected on {S.SERIAL_PORT}")
        link.close()
    print("\n  " + ("All set." if ok else "Something above needs attention (see the messages)."))
    return ok


# --------------------------------------------------------------------------- live
def _annotate(frame, model, status):
    v = frame.copy()
    for label, col in ((S.AIM_CLASS, GREEN), (S.TARGET_CLASS, BLUE), (S.IMPACT_CLASS, RED)):
        try:
            dets = [d for d in model.detect_all(frame, label) if d.confidence >= S.MIN_CONFIDENCE]
        except ValueError:
            continue
        for d in (dets[:1] if label == S.AIM_CLASS else dets):
            x, y = map(int, d.pixel_position)
            if label == S.AIM_CLASS:
                cv2.drawMarker(v, (x, y), col, cv2.MARKER_CROSS, 30, 2)
            elif label == S.TARGET_CLASS:
                cv2.circle(v, (x, y), 9, col, 2)
            else:
                cv2.rectangle(v, tuple(map(int, d.box[:2])), tuple(map(int, d.box[2:])), col, 2)
    for i, t in enumerate(status):
        cv2.putText(v, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(v, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1, cv2.LINE_AA)
    return v


class ImpactMarker:
    """The "mark the impact" window: click to place a crosshair, click again to move it,
    zoom with the mouse wheel (or + / -) around the cursor, Enter to confirm.

        left click        place / move the mark
        mouse wheel, + -  zoom in / out around the cursor (up to 12x)
        right-drag, WASD  move the view when zoomed in
        r                 reset the view
        Enter / Space     confirm the mark
        Esc               cancel
    """
    SIZE = 720                                        # window is SIZE x SIZE pixels

    def __init__(self, img):
        import scope
        self.img = img
        h, w = img.shape[:2]
        c = scope.eyepiece_circle(img)
        cx, cy, r = c if c is not None else (w / 2, h / 2, min(w, h) / 2)
        self.home = (cx, cy, self.SIZE / (2 * r))     # whole eyepiece fills the window
        self.cx, self.cy, self.zoom = self.home
        self.mark = None                              # full-frame pixel (x, y)
        self.mouse = (self.SIZE / 2, self.SIZE / 2)
        self._drag = None

    # window pixel <-> full-frame pixel
    def to_img(self, u, v):
        return self.cx + (u - self.SIZE / 2) / self.zoom, self.cy + (v - self.SIZE / 2) / self.zoom

    def to_win(self, x, y):
        return (x - self.cx) * self.zoom + self.SIZE / 2, (y - self.cy) * self.zoom + self.SIZE / 2

    def zoom_at(self, u, v, factor):
        x, y = self.to_img(u, v)                      # keep the point under the cursor fixed
        self.zoom = float(np.clip(self.zoom * factor, self.home[2] * 0.8, self.home[2] * 12))
        self.cx, self.cy = x - (u - self.SIZE / 2) / self.zoom, y - (v - self.SIZE / 2) / self.zoom

    def on_mouse(self, event, u, v, flags, *_):
        self.mouse = (u, v)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mark = self.to_img(u, v)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(u, v, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drag = (u, v, self.cx, self.cy)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag and flags & cv2.EVENT_FLAG_RBUTTON:
            u0, v0, cx0, cy0 = self._drag
            self.cx, self.cy = cx0 - (u - u0) / self.zoom, cy0 - (v - v0) / self.zoom
        elif event == cv2.EVENT_RBUTTONUP:
            self._drag = None

    def key(self, k):
        """Returns "confirm", "cancel" or None."""
        step = 60 / self.zoom
        if k in (13, 10, 32):
            return "confirm" if self.mark is not None else None
        if k == 27:
            return "cancel"
        if k in (ord("+"), ord("=")):
            self.zoom_at(*self.mouse, 1.25)
        elif k in (ord("-"), ord("_")):
            self.zoom_at(*self.mouse, 0.8)
        elif k == ord("r"):
            self.cx, self.cy, self.zoom = self.home
        elif k in (ord("w"), ord("a"), ord("s"), ord("d")):
            dx, dy = {"w": (0, -step), "s": (0, step), "a": (-step, 0), "d": (step, 0)}[chr(k)]
            self.cx, self.cy = self.cx + dx, self.cy + dy
        return None

    def render(self):
        z = self.zoom
        M = np.float32([[z, 0, self.SIZE / 2 - self.cx * z], [0, z, self.SIZE / 2 - self.cy * z]])
        interp = cv2.INTER_NEAREST if z > 4 else cv2.INTER_LINEAR    # see single pixels when zoomed in
        view = cv2.warpAffine(self.img, M, (self.SIZE, self.SIZE), flags=interp, borderValue=(40, 40, 40))
        if self.mark is not None:
            u, v = map(int, map(round, self.to_win(*self.mark)))
            for a_, b_ in (((u - 22, v), (u - 6, v)), ((u + 6, v), (u + 22, v)),
                           ((u, v - 22), (u, v - 6)), ((u, v + 6), (u, v + 22))):
                cv2.line(view, a_, b_, (0, 0, 0), 3)
                cv2.line(view, a_, b_, (0, 0, 255), 1)
            cv2.circle(view, (u, v), 2, (0, 0, 255), -1)
        lines = ["Click the centre of the new impact (click again to move it)",
                 "wheel or +/- zoom | right-drag or WASD move | r reset",
                 ("ENTER = confirm" if self.mark is not None else "then press ENTER") + " | Esc = cancel",
                 f"zoom {z / self.home[2]:.1f}x" + (f" | mark at ({self.mark[0]:.1f}, {self.mark[1]:.1f})"
                                                      if self.mark is not None else "")]
        top = view[0:8 + 22 * len(lines)]
        top[:] = (top * 0.35).astype(np.uint8)        # darkened strip so the text is readable
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        CYAN if i < 3 else YELLOW, 1, cv2.LINE_AA)
        return view


def _click_impact(after):
    """Let the user mark the impact in the AFTER frame. Returns the full-frame pixel or None."""
    m = ImpactMarker(after)
    name = "mark the impact"
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, m.on_mouse)
    result = None
    while True:
        cv2.imshow(name, m.render())
        k = cv2.waitKey(20)
        if k != -1:
            action = m.key(k & 0xFF)
            if action:
                result = action
                break
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:      # window closed with the X
            result = "cancel"
            break
    cv2.destroyWindow(name)
    return m.mark if result == "confirm" else None


def measure_pair(before, after, impact_xy=None, out=OUT / "live"):
    """Measure BEFORE/AFTER (optionally with the impact marked by hand); save + return the result."""
    import impact
    args = impact.parse_args(["--output", str(out / "result")])
    args.impact_xy = impact_xy
    res = impact.measure(before, after, args)
    impact.save(out / "result", before, after, res)
    return res


def live(camera_source=None, model=None, every=5):
    banner("LIVE: b = BEFORE (aim)   a = AFTER (measure)   m = mark impact by hand   s = save   q = quit")
    import camera, impact, arduino_link
    model = model or load_model()
    out = OUT / "live"
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = camera.open_source(camera_source if camera_source is not None else S.CAMERA)
    link = open_link()
    is_file = isinstance(camera_source, str) and Path(camera_source).is_file()
    delay = max(1, int(1000 / max(fps, 1))) if is_file else 1     # play a recording at its own speed
    before = after = None
    after_processed = False
    shown, n, t0, ms = None, 0, time.time(), 0.0
    last_key_time= 0.0
    try:
        for idx, frame in frames:
            n += 1
            if n % every == 1 or shown is None:
                t = time.time()
                status = [f"{n / max(1e-6, time.time() - t0):.1f} fps | model {ms:.0f} ms",
                          "BEFORE taken" if before is not None else "aim, then press b",
                          "b before | a after | m mark impact | s save | q quit"]
                shown = _annotate(frame, model, status) 
                ms = 1000 * (time.time() - t)
                display = cv2.resize(shown, (640, 360))
                cv2.imshow("live (q quits)", display)
                key = cv2.waitKey(30) & 0xFF

# Prevent repeated key presses within 0.5 seconds
            if key in (ord("b"), ord("a"), ord("m"), ord("s")):
                now = time.monotonic()

                if now - last_key_time < 0.5:
                    key = 255
                else:
                    last_key_time = now
                    #------------
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(str(out / f"frame_{idx:06d}.png"), frame)
                print("  saved", out / f"frame_{idx:06d}.png")
            if key == ord("b"):
                before = frame.copy()
                after = None
                after_processed = False

                cv2.imwrite(str(out / "before.png"), before)
                print("\n  BEFORE taken")
            if key in (ord("a"), ord("m")) and before is not None and not after_processed:
                after_processed = True
                if key == ord("a") or after is None:
                    after = frame.copy()
                    cv2.imwrite(str(out / "after.png"), after)
                res = measure_pair(before, after) if key == ord("a") else {"valid": False, "notes": []}
                if not res["valid"]:                          # no impact found: offer to mark it by hand
                    if key == ord("a"):
                        print("  " + impact.describe(res))
                        print("  -> click the impact in the window that opens (Esc to skip)")
                    xy = _click_impact(after)
                    if xy is None:
                        continue
                    res = measure_pair(before, after, xy)
                print("\n" + impact.describe(res))
                for note in res["notes"]:
                    print("  note:", note)
                view = cv2.imread(str(out / "result" / "impact_result.png"))
                move = res.get("valid") and link is not None
                msg = "y = MOVE THE TURRETS   any other key = skip" if move else "any key = continue"
                cv2.putText(view, msg, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
                cv2.imshow("result", view)
                k2 = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("result")
                if move and k2 == ord("y"):
                    print("  " + arduino_link.send_result(link, res))
    except KeyboardInterrupt:
        print("\n  Stopped (Ctrl+C).")
    finally:
        frames.close()
        if link:
            link.close()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- motors
def motors():
    banner("MOTORS: move the turrets by hand")
    import arduino_link
    print(f"  {arduino_link.steps_per_moa():.2f} steps per MOA "
          f"(+ = RIGHT / UP; windage moves first). Type:\n"
          "    1.5 -2        -> 1.5 MOA right, 2 MOA down\n"
          "    s 100 -50     -> exactly 100 windage steps, -50 elevation steps\n"
          "    q             -> quit")
    link = open_link()
    if link is None:
        return
    try:
        while True:
            line = input("  move> ").strip().lower()
            if line in ("q", "quit", "exit"):
                break
            try:
                parts = line.split()
                if parts and parts[0] == "s":
                    w, e = int(parts[1]), int(parts[2])
                else:
                    w, e = arduino_link.motor_steps(float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                print("  ? type two numbers, e.g. 1.5 -2  (or s 100 -50)")
                continue
            print(f"  MOVE {w} {e} ...")
            print("  " + link.move(w, e)[1])
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        link.close()


# --------------------------------------------------------------------------- menu
def main(argv):
    # --video FILE (or --camera N / picamera) replaces settings.CAMERA: e.g. try it on a PC with a recording
    source = None
    for flag in ("--video", "--camera"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} needs a value")
            source = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    track_args = ["--show"] + (["--video", source] if source and not str(source).isdigit()
                               and source.lower() not in ("picamera", "pi", "picam") else
                               ["--camera", source] if source else [])
    modes = {"check": lambda: check(source), "live": lambda: live(source), "motors": motors,
             "track": lambda: __import__("shot_tracker").main(track_args)}
    if argv and argv[0] in modes:
        return modes[argv[0]]()
    if argv:
        print(__doc__)
        return
    while True:
        print("\nScope camera test" + (f"   (source: {source})" if source else "") +
              "\n  1  check   camera, model, grid scale, Arduino\n"
              "  2  live    live view, BEFORE/AFTER measure (impact by hand if needed)\n"
              "  3  track   detect shots automatically (full system)\n"
              "  4  motors  move the turrets by hand\n  q  quit")
        choice = input("choose: ").strip().lower()
        if choice in ("q", "quit", ""):
            break
        name = {"1": "check", "2": "live", "3": "track", "4": "motors"}.get(choice, choice)
        if name in modes:
            try:
                modes[name]()
            except KeyboardInterrupt:
                print("\n  (stopped)")
        else:
            print("  ? choose 1-4 or q")


if __name__ == "__main__":


    # Launch the standalone Waveshare touchscreen interface
    if len(sys.argv) > 1 and sys.argv[1].lower() == "touch":
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "touchscreen_app.py")
            ],
            check=False
        )
        sys.exit(0)

    # Check whether the VNC desktop is already running
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        vnc_running = sock.connect_ex(("127.0.0.1", 5901)) == 0

    if not vnc_running:
        print("Starting VNC virtual desktop...")

        subprocess.run([
            "vncserver-virtual",
            ":1",
            "-geometry", "1280x800",
            "-depth", "24"
        ], check=True)

        # Allow the server to initialize
        time.sleep(2)

    else:
        print("VNC virtual desktop already running.")

    # Configure the graphical display
    os.environ["DISPLAY"] = ":1"
    os.environ.pop("QT_QPA_PLATFORM", None)

    print("Display configured: :1")

    # Start the program AFTER configuring VNC
    main(sys.argv[1:])