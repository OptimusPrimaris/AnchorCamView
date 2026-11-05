# Raspberry Pi test package

One program, `run.py`, does everything on test day. Everything adjustable is in `settings.py`.
Designed for a **Raspberry Pi 4 or 5 with 64-bit Raspberry Pi OS (Bookworm)**, a screen (desktop
or VNC), a USB camera or the Pi camera module, and the Feather + Motor FeatherWing.

## Set up (once)

```bash
cd ~
unzip pi_package.zip -d scope_test
cd scope_test
python3 setup_pi.py          # installs everything, converts the model to NCNN (fast), checks camera + model
```

Then edit `settings.py`:

| Setting | Set to |
| --- | --- |
| `CAMERA` | `0` (USB camera) or `"picamera"` (camera module) |
| `SERIAL_PORT` | `"/dev/ttyACM0"` (check with `ls /dev/ttyACM*` with the Feather plugged in) |
| `CAMERA_FPS` | as high as the camera allows |

Log out and back in once after setup (permission for the Arduino's USB port).

## Run

```bash
.venv/bin/python run.py            # menu
```

| Choice | What it does | Keys |
| --- | --- | --- |
| **1 check** | Camera frame, model speed, crosshair, grid scale, Arduino connection | - |
| **2 live** | Live view of what the model sees. Take BEFORE, then AFTER: it measures. If the model can't find the impact, **mark it** in the window that opens: zoom with the mouse wheel (or + / -), click the centre, click again to adjust, **Enter** to confirm. Then **y** moves the turrets | **b** before, **a** after, **m** mark impact by hand, **s** save frame, **q** quit |
| **3 track** | Automatic: detects each shot, measures it, moves the turrets (full system). It does **not** ask before moving: set `SERIAL_PORT = None` to watch it measure without moving anything | **q** or Ctrl+C stops (results still saved) |
| **4 motors** | Move the turrets by hand: `1.5 -2` = 1.5 MOA right, 2 MOA down; `s 100 -50` = exact steps | **q** quits |

You can also skip the menu: `.venv/bin/python run.py live` (or `check`, `track`, `motors`).
Add `--video some_clip.mp4` to use a recording instead of the camera (also works on a PC).

## Test-day order

1. **check**: everything reports OK with the scope aimed at the target.
2. **motors**: `1 1` should turn windage RIGHT and elevation UP (else set `WINDAGE_DIRECTION` /
   `ELEVATION_DIRECTION` to -1). `s 1000 0` = exactly one turret turn.
3. **live**: aim, **b**, mark the target, **a**, click the mark if needed, check the result picture,
   **y** to move the turrets, and confirm the crosshair moved onto the mark.
4. **track**: bump/fire, check each shot's result in `output/shots/`.

Results and pictures are saved in `output/`. Save frames with **s** in live view; after-frames with
marks are exactly what the team needs to train the Impact class.

## If something is off

* Model slower than ~300 ms per look: lower `YOLO_ZOOM` or the camera resolution.
* Camera won't open: check `CAMERA`; for the module, `rpicam-hello` should show a picture.
* Arduino not connecting: `SERIAL_PORT`, the cable, log out/in after setup, sketch uploaded.
* How the code works: `docs/HOW_IT_WORKS.md`.
