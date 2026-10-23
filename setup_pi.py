"""One-shot setup on a Raspberry Pi (Pi 4 or 5, 64-bit Raspberry Pi OS Bookworm).

Run in a terminal on the Pi, inside the unzipped camera_prototype folder:
    python3 setup_pi.py

What it does (safe to re-run):
  1. checks the Pi is running a 64-bit OS and a usable Python
  2. installs system pieces with apt (asks for your password): the Pi camera library
     (Picamera2), venv support; and lets you use the Arduino's USB port (dialout group)
  3. creates .venv (able to see Picamera2) and installs OpenCV, NumPy, pyserial, YOLO
  4. converts the YOLO model to NCNN, a format that runs several times faster on the
     Pi's processor, and points settings.py at it
  5. times the model on one frame and tries the camera

Options:  --no-apt (skip step 2)   --no-export (keep the .pt model)   --no-camera
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
VENV_PY = VENV / "bin" / "python"


def step(text):
    print(f"\n=== {text}", flush=True)


def run(cmd, what, check=True):
    print("  >", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=HERE)
    if check and r.returncode != 0:
        sys.exit(f"\nFAILED: {what} (exit code {r.returncode}). Copy the error text above and ask for help.")
    return r.returncode == 0


def in_venv(code, what, check=True):
    return run([VENV_PY, "-c", code], what, check)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-apt", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-camera", action="store_true")
    a = p.parse_args()

    step("1. Checking the Pi")
    machine = platform.machine()
    print(f"  machine {machine}, Python {platform.python_version()}")
    if machine not in ("aarch64", "arm64"):
        sys.exit("  This needs a 64-bit OS (aarch64). Install 64-bit Raspberry Pi OS (Bookworm) and try again.")
    if sys.version_info < (3, 10):
        sys.exit("  Python 3.10 or newer is needed (Bookworm has 3.11).")

    if not a.no_apt:
        step("2. System packages (apt) - you may be asked for your password")
        run(["sudo", "apt", "update"], "apt update")
        run(["sudo", "apt", "install", "-y", "python3-venv", "python3-picamera2", "libopenblas-dev"],
            "apt install")
        user = os.environ.get("USER") or os.environ.get("LOGNAME")
        if user:
            run(["sudo", "usermod", "-aG", "dialout", user], "serial port access", check=False)
            print("  (USB serial access for the Arduino takes effect after you log out and in, or reboot)")

    step("3. Python packages in .venv")
    if not VENV_PY.exists():
        # --system-site-packages lets the venv see Picamera2, which only comes from apt
        run([sys.executable, "-m", "venv", "--system-site-packages", VENV], "create .venv")
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip")
    run([VENV_PY, "-m", "pip", "install", "-r", "requirements.txt"], "base requirements")
    run([VENV_PY, "-m", "pip", "install", "-r", "requirements-yolo.txt"], "YOLO (this one takes a while)")
    in_venv("import cv2, numpy, serial, ultralytics; print('  imports OK, OpenCV', cv2.__version__)", "imports")

    sys.path.insert(0, str(HERE))
    import settings as S
    model = S.MODEL
    if not a.no_export and str(model).endswith(".pt"):
        step("4. Converting the model to NCNN (fast on the Pi)")
        pt = HERE / model
        if not pt.exists():
            sys.exit(f"  Model not found: {pt}")
        out_dir = pt.with_name(pt.stem + "_ncnn_model")
        if not out_dir.exists():
            in_venv(f"from ultralytics import YOLO; YOLO(r'{pt}').export(format='ncnn', imgsz=640)", "NCNN export")
        if out_dir.exists():
            rel = out_dir.relative_to(HERE).as_posix()
            text = (HERE / "settings.py").read_text(encoding="utf-8")
            shutil.copy(HERE / "settings.py", HERE / "settings_before_pi_setup.py")
            text = re.sub(r'^MODEL = .*$', f'MODEL = "{rel}"   # NCNN version of {pt.name} (made by setup_pi.py)',
                          text, count=1, flags=re.M)
            (HERE / "settings.py").write_text(text, encoding="utf-8")
            model = rel
            print(f"  settings.py now uses {rel}  (old settings saved as settings_before_pi_setup.py)")
        else:
            print("  Export did not produce a folder; keeping the .pt model (slower but works).")

    step("5. Checks")
    in_venv("import time, numpy as np, detectors, settings as S\n"
            "detectors.ZOOM = S.YOLO_ZOOM\n"
            "m = detectors.load_detector(S.MODEL)\n"
            "img = np.full((S.CAMERA_HEIGHT, S.CAMERA_WIDTH, 3), 128, np.uint8)\n"
            "m.detect_all(img, S.AIM_CLASS); img = img.copy()\n"
            "t = time.time()\n"
            "for _ in range(3):\n"
            "    img = img.copy(); m.detect_all(img, S.AIM_CLASS)\n"
            "print(f'  model: {1000 * (time.time() - t) / 3:.0f} ms per look ({S.MODEL})')",
            "model timing")
    if not a.no_camera:
        ok = in_venv("import camera, settings as S\n"
                     "frames, fps = camera.open_source(S.CAMERA)\n"
                     "i, f = next(frames)\n"
                     "print(f'  camera {S.CAMERA!r}: frame {f.shape[1]} x {f.shape[0]} at {fps:g} fps')",
                     "camera", check=False)
        if not ok:
            print("  Camera not opened. USB camera: plug it in (CAMERA = 0 in settings.py). Pi camera "
                  "module: set CAMERA = \"picamera\" in settings.py, then re-run with --no-apt --no-export.")
    if (HERE / "tests" / "test_impact.py").exists():
        run([VENV_PY, "tests/test_impact.py"], "simulated shot test")
    print("\nDone. Next (see PI_README.md):\n"
          "  .venv/bin/python run.py          # menu: check / live / track / motors")


if __name__ == "__main__":
    main()
