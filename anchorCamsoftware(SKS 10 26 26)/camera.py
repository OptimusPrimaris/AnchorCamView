"""Frames from a video file, a USB camera, or the Raspberry Pi camera module.

    open_source("videos/clip.mp4")        a recorded video
    open_source(0)                        USB camera 0 (OpenCV)
    open_source("picamera")               Raspberry Pi camera module (Picamera2)

Returns (frames, fps): frames yields (frame_number, BGR image), numbered from 1.
Size and frame rate for live cameras come from settings.py (CAMERA_*).
"""
import time
import cv2
import settings as S


def open_source(source, width=None, height=None, fps=None):
    width, height, fps = width or S.CAMERA_WIDTH, height or S.CAMERA_HEIGHT, fps or S.CAMERA_FPS
    if isinstance(source, str) and source.lower() in ("picamera", "picam", "pi"):
        return _picamera(width, height, fps)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'camera' if isinstance(source, int) else 'video'} {source!r}")
    if isinstance(source, int):                         # live USB camera: ask for size and rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    rate = cap.get(cv2.CAP_PROP_FPS) or fps

    def gen():
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            yield i, f
        cap.release()
    return gen(), rate


def _picamera(width, height, fps):
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError("Picamera2 not found. On the Pi run: sudo apt install -y python3-picamera2 "
                           "(setup_pi.py does this), and use the .venv it creates.") from e
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (int(width), int(height)), "format": "RGB888"},
                                         controls={"FrameRate": float(fps)})
    cam.configure(cfg)
    cam.start()                                    # exposure and white balance stay automatic
    _set_focus(cam)                                # manual focus from settings.CAMERA_FOCUS
    time.sleep(1.0)                                # let the lens move and exposure adjust

    def gen():
        i = 0
        try:
            while True:
                frame = cam.capture_array()        # "RGB888" arrives in BGR order, as OpenCV expects
                i += 1
                yield i, frame
        finally:
            print("Closing Raspberry Pi camera...")
            try:
                cam.stop()
            finally:
                cam.close()
            print("Camera released.")
    return gen(), float(fps)


def _set_focus(cam):
    """Focus for the Camera Module 3 (motorised lens), from settings.CAMERA_FOCUS:
    a number = manual lens position (e.g. 3.2), held fixed for the whole run.
    Cameras without autofocus (Module 2, HQ camera) are left alone; focus those
    by hand on the lens.""" 
    if "AfMode" not in getattr(cam, "camera_controls", {}):
        print("  camera: no autofocus on this camera (focus it by hand)")
        return
    from libcamera import controls
    setting = S.CAMERA_FOCUS
    if isinstance(setting, (int, float)):
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(setting)})
        print(f"  camera: focus fixed at lens position {float(setting):g}")
    elif setting == "auto":
        cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        print("  camera: continuous autofocus (frames may change focus between BEFORE and AFTER)")
    else:                                          # "auto-once": focus now, then hold that position
        ok = cam.autofocus_cycle()
        pos = cam.capture_metadata().get("LensPosition")
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                          **({"LensPosition": pos} if pos is not None else {})})
        print(f"  camera: autofocus {'done' if ok else 'did not converge'}, locked at lens position "
              f"{pos if pos is not None else '?'}")
