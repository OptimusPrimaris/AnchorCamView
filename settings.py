"""All the settings someone might want to change, in one place.

Edit the values below and save; every script (impact.py, shot_tracker.py,
tools/walkthrough.py) uses them as its defaults. A command-line option still
wins for a one-off run, e.g.  --reticle-margin-px 1  or  --distance-yd 50.

    python settings.py        # prints the current settings and checks them
"""

# --------------------------------------------------------------------------- range and target
DISTANCE_YD = 25.0            # distance to the target, yards
GRID_SQUARE_IN = 1.0          # size of one printed grid square, inches (scale comes from this);
                              # set to None to use another scale reference (see scale.py)
CLICK_MOA = 0.25              # one turret click, MOA (corrections are rounded to whole clicks)

# --------------------------------------------------------------------------- YOLO model
MODEL = "models/crosshair_impact_target_v1.pt"   # weights with the classes below
AIM_CLASS = "Crosshair-Center"
TARGET_CLASS = "Target-Center"
IMPACT_CLASS = "Impact"
USE_IMPACT_MODEL = True       # False = find the new impact by comparing frames only
MIN_CONFIDENCE = 0.25         # detections below this are ignored (0..1)
YOLO_ZOOM = 2.5               # crop around the eyepiece and enlarge before YOLO looks;
                              # 1 = whole frame. 2.5 found the target dots in the first real video

# --------------------------------------------------------------------------- reticle mask
# The reticle is fixed to the camera, not the paper, so it is hidden from the alignment
# and from the frame comparison. Anything under the mask is invisible to those steps.
RETICLE_MASK = "lines"        # "lines" = hug the measured reticle lines; "band" = fixed-width strips
RETICLE_MARGIN_PX = 2         # "lines": extra pixels either side of each line
RETICLE_DOT_PX = 2            # "lines": extra room for the dots / hash marks on the lines
RETICLE_BAND_PX = 8           # "band": half-width of each strip (also how far to search for the lines)

# --------------------------------------------------------------------------- lining the frames up
ALIGN = "dots"                # "dots" (target dots first, texture cross-check), "auto" (texture
                              # first), or "features" (texture only)
AGREE_MOA = 0.5               # the two methods disagreeing by more than this adds a note
MAX_TURN_DEG = 20.0           # reject a match that turns the target more than this between frames
MAX_SCALE_CHANGE = 0.20       # ... or changes its size by more than this (0.20 = 20%)

# --------------------------------------------------------------------------- new impact
MIN_HOLE_PX = 12              # frame comparison: smallest changed area counted as an impact (pixels)
MAX_CHANGED_PCT = 5.0         # frame comparison: give up if more than this % of the target changed as
                              # strongly as the best candidate (frames too different to compare)
NEW_IMPACT_CHANGE = 15        # Impact model: how much (grey levels) the target must change under a
                              # detection for it to count as new

# --------------------------------------------------------------------------- shot detection (shot_tracker.py)
BUFFER_SECONDS = 1.0          # full frames kept in memory
SPIKE_PX = 6.0                # a jump of at least this many pixels between frames = a shot
SETTLE_PX = 2.5               # motion below this counts as settled
SETTLE_SECONDS = 0.25         # ... for at least this long

# --------------------------------------------------------------------------- live camera (shot_tracker.py, tools/live_view.py)
CAMERA = 0                    # 0, 1 ... = USB camera; "picamera" = Raspberry Pi camera module
CAMERA_WIDTH = 1280           # pixels; the first real video was 1280 x 720
CAMERA_HEIGHT = 720
CAMERA_FPS = 30               # higher = point-of-aim frame closer to the trigger (the test video was ~14)
CAMERA_LOCK_EXPOSURE = True   # Pi camera: fix exposure/white balance after start-up (steadier frames)

# --------------------------------------------------------------------------- Arduino and motors
# The board runs arduino/OpticMotorController/OpticMotorController.ino and takes
#   MOVE windageSteps elevationSteps
# The correction in MOA is turned into motor steps with the numbers below.
SERIAL_PORT = None            # e.g. "COM3" (Windows) or "/dev/ttyACM0" (Raspberry Pi); None = don't send
BAUD = 115200                 # must match Serial.begin() in the sketch

MOTOR_STEPS_PER_REV = 200     # full steps per motor revolution (sketch: STEPS_PER_REVOLUTION, NEMA 8)
MOTOR_SPEED_RPM = 20          # sketch: MOTOR_SPEED_RPM (used to know how long a move takes)
MOTOR_HOLD_MS = 1000          # sketch: HOLD_TIME_MS, motors held powered after a move before release
GEAR_RATIO = 5.0              # MOTOR turns per one TURRET turn: 5:1 gearbox (x any belt/pulley ratio)
MOA_PER_TURRET_REV = 15.0     # MOA in one full turret turn: 60 clicks counted x 0.25 MOA = 15 (confirmed)
WINDAGE_DIRECTION = 1         # +1, or -1 if a positive step count turns the windage turret LEFT
ELEVATION_DIRECTION = 1       # +1, or -1 if a positive step count turns the elevation turret DOWN
MAX_STEPS_PER_MOVE = 5000     # safety: refuse any single move bigger than this (per motor)


def check():
    """Catch typos and impossible values early, with a plain message."""
    problems = []
    for name in ("DISTANCE_YD", "CLICK_MOA", "YOLO_ZOOM", "BUFFER_SECONDS", "SPIKE_PX", "SETTLE_PX",
                 "SETTLE_SECONDS", "MAX_TURN_DEG", "MAX_SCALE_CHANGE", "AGREE_MOA", "MAX_CHANGED_PCT",
                 "CAMERA_WIDTH", "CAMERA_HEIGHT", "CAMERA_FPS", "MOTOR_STEPS_PER_REV", "MOTOR_SPEED_RPM",
                 "GEAR_RATIO", "MOA_PER_TURRET_REV", "MAX_STEPS_PER_MOVE"):
        v = globals()[name]
        if not isinstance(v, (int, float)) or v <= 0:
            problems.append(f"{name} must be a positive number (is {v!r})")
    if GRID_SQUARE_IN is not None and (not isinstance(GRID_SQUARE_IN, (int, float)) or GRID_SQUARE_IN <= 0):
        problems.append(f"GRID_SQUARE_IN must be a positive number or None (is {GRID_SQUARE_IN!r})")
    if not 0 <= MIN_CONFIDENCE <= 1:
        problems.append(f"MIN_CONFIDENCE must be between 0 and 1 (is {MIN_CONFIDENCE!r})")
    for name in ("WINDAGE_DIRECTION", "ELEVATION_DIRECTION"):
        if globals()[name] not in (1, -1):
            problems.append(f"{name} must be 1 or -1 (is {globals()[name]!r})")
    if RETICLE_MASK not in ("lines", "band"):
        problems.append(f'RETICLE_MASK must be "lines" or "band" (is {RETICLE_MASK!r})')
    if ALIGN not in ("dots", "auto", "features"):
        problems.append(f'ALIGN must be "dots", "auto" or "features" (is {ALIGN!r})')
    for name in ("RETICLE_MARGIN_PX", "RETICLE_DOT_PX", "RETICLE_BAND_PX", "MIN_HOLE_PX"):
        if not isinstance(globals()[name], int) or globals()[name] < 0:
            problems.append(f"{name} must be a whole number, 0 or more (is {globals()[name]!r})")
    return problems


if __name__ == "__main__":
    for k, v in globals().copy().items():
        if k.isupper():
            print(f"  {k:20s} = {v!r}")
    bad = check()
    print("\n" + ("\n".join("PROBLEM: " + b for b in bad) if bad else "All settings look valid."))
