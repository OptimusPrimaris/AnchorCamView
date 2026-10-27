# How the code works

This guide explains the scope camera code from the top down: what each file is for,
the path one measurement takes through the code, and where to go to change something.
Function names are written `file.py → function()` so you can jump straight to them.

---

## 1. The big picture

```
 camera / video ──► shot_tracker.py ──► impact.py ──► MOA + clicks ──► arduino_link.py ──► Arduino
                    (WHEN was the shot?   (WHERE did it land
                     pick 2 frames)        relative to the aim?)
```

There are only two "brains":

| Brain | Question it answers | File |
| --- | --- | --- |
| **Shot tracker** | Which frame shows the aim at the moment of the shot, and which frames show the result after it settles? | `shot_tracker.py` |
| **Measurement** | Given a BEFORE frame and an AFTER frame, how far (in MOA) is the impact from the point of aim, and what correction is that in clicks? | `impact.py` |

Everything else is a helper used by one of those two.

---

## 2. The files, grouped by job

### The ones you run
| File | What it does |
| --- | --- |
| `shot_tracker.py` | **Main program.** Reads a video or live camera, detects shots, measures each one, saves pictures + JSON, optionally sends clicks to the Arduino. |
| `impact.py` | Measures one BEFORE/AFTER pair (from two images or two frame numbers of a video). Also the library the tracker calls. |
| `tools/walkthrough.py` | Runs `impact.py` and saves **one picture per step** with INTENDED vs GOT captions. Best way to see what is happening. |
| `tools/live_view.py` | Live camera window showing what the model sees; **b**/**a** keys take BEFORE/AFTER and measure. |
| `tools/inspect_detections.py` | Lists every YOLO detection in a frame (any confidence), with pictures. |
| `settings.py` | **All adjustable values** in one place. `python settings.py` prints and checks them. |

### Helpers for the measurement (`impact.py` uses these)
| File | Job |
| --- | --- |
| `detectors.py` | Loads the YOLO model and answers "where are the boxes of class X in this image?" (with the `YOLO_ZOOM` crop-and-enlarge). |
| `scope.py` | Everything specific to looking through a scope: find the round eyepiece view, mask the reticle, find the new hole (frame comparison or Impact model), fine alignment (ECC), rough shift. |
| `landmarks.py` | Pair the red target dots between the two frames and fit the transform from them. |
| `plane.py` + `alignment.py` | Feature matching (SIFT) on the paper texture: the second, independent way to line the frames up. |
| `scale.py` | Pixels → inches: the OpenCV grid-line method (used), plus dot spacing and reference block (alternatives). Also finds the target area. |

### Helpers for the tracker / output
| File | Job |
| --- | --- |
| `camera.py` | Gives frames from a video file, USB camera or Pi camera module, one at a time. |
| `arduino_link.py` | Turns the correction (MOA) into motor steps (`motor_steps()`), sends `MOVE windage elevation` and waits for `DONE`. |
| `arduino/OpticMotorController/OpticMotorController.ino` | The Arduino side (your team's sketch): moves the windage motor, then the elevation motor. |

### Small shared pieces
| File | Job |
| --- | --- |
| `data.py` | The `Detection` record (label, confidence, centre, box). |
| `common.py` | Reading images, drawing labels. |

### Tests (no footage needed)
`tests/scope_sim.py` draws simulated scope views with known answers; `tests/sim_detectors.py`
pretends to be the YOLO model on those; `tests/test_impact.py`, `test_tracker.py`,
`test_arduino.py` score the code against the known answers.

### Not part of the pipeline (kept for reference)
`legacy/` (the original prototype), `tools/two_view.py`, `tools/batch_view.py`,
`tools/objects.py`, `tools/demo_scenes.py`, and in `scope.py` the calibration-frame
functions (`reticle_from_calibration`, `reticle_center`), which only run if you pass
`--calibration`. `detectors.py → MockDetector` is a colour-blob stand-in used by old tools.
These can be deleted later without affecting the measurement.

---

## 3. One measurement, step by step (`impact.py → measure()`)

`measure(before, after, args)` runs these steps in order. Each step either adds results
to a dictionary called `res` or stops with a reason (`Unmeasurable`). The step numbers
match the `tools/walkthrough.py` pictures.

```
measure()
 ├─ eyepiece circle            scope.eyepiece_circle()          round view inside the black ring
 ├─ STEP 2 aim point           _aim_point()                     YOLO Crosshair-Center (best box)
 ├─ STEP 3 reticle mask        _reticle_mask()                  scope.reticle_line_mask() (or band)
 ├─ target area                _on_target()                     scale.from_grid() + scale.grid_region()
 ├─ STEP 4–5 line frames up    _relate()
 │     ├─ texture              _features() → plane.register_images()   SIFT homography
 │     │                       implausible()  rejects >20° turns / >20% size changes
 │     ├─ dots                 _landmarks()
 │     │     ├─ landmarks.centres()      YOLO Target-Center boxes in each frame
 │     │     ├─ landmarks.match()        which dot in AFTER is which dot in BEFORE
 │     │     ├─ landmarks.fit()          transform from the pairs (1 dot: shift … 5+: homography)
 │     │     └─ scope.refine_alignment() ECC: sub-pixel brightness alignment
 │     └─ choose: dots first, texture as cross-check (settings.ALIGN)
 ├─ STEP 6 scale               _scale() → scale.from_grid()     px per inch from the 1-in grid
 ├─ STEP 8 new impact          _new_impact()
 │     ├─ scope.new_hole_by_model()       YOLO Impact boxes that weren't there before
 │     ├─ scope.new_hole_by_difference()  frame comparison (fallback / cross-check)
 │     └─ STEP 7: ignore anything off the target area; refuse if the frames are too different
 └─ STEP 9 result              _result() → express()            MOA, inches, correction, clicks
```

**The geometry in one paragraph.** The camera sees the paper move during recoil. Step 5
finds the transform `H` that maps any AFTER-frame pixel to where that same spot of paper
was in the BEFORE frame. The impact found in the AFTER frame is mapped through `H` into the
BEFORE frame. There, the displacement is simply *impact − crosshair*, in pixels. Pixels
become inches with the grid scale, inches become MOA with the distance
(1 MOA = 1.0472 in × yards / 100), and the correction is the opposite direction, rounded to
whole clicks (`CLICK_MOA`). For the motors, the MOA (not the rounded clicks) becomes steps:
steps = MOA x `MOTOR_STEPS_PER_REV` x `GEAR_RATIO` / `MOA_PER_TURRET_REV` (`arduino_link.motor_steps()`).

**Why so many checks.** Every risky step has an independent second opinion, and the code
prefers refusing ("NO MEASUREMENT: …") over guessing:
* texture vs dots for lining up (and rejecting physically impossible matches);
* Impact model vs frame comparison for the hole;
* in the tracker, 3 settled frames measured separately must agree.

### What comes out (`res`)
`res` is a plain dictionary. The important keys:

| Key | Meaning |
| --- | --- |
| `valid` | True if a measurement was made |
| `notes` | List of warnings / the reason it stopped |
| `moa_right`, `moa_up` | Impact relative to aim, MOA (right/up positive) |
| `in_right`, `in_up` | Same in inches at the target distance |
| `correction` | `elevation_moa`, `windage_moa`, `elevation_clicks`, `windage_clicks` (+ = UP / RIGHT), `windage_steps`, `elevation_steps` (what the motors are told) |
| `poa_before_px`, `poi_before_px` | Aim and impact, in BEFORE-frame pixels |
| `alignment`, `scale`, `hole_method` | How each step was done (for checking) |

Keys starting with `_` (e.g. `_reticle_mask`, `_heat`) are pictures/arrays for drawing only.

---

## 4. The shot tracker (`shot_tracker.py`)

```
run_stream()
 ├─ FrameAnalyzer.update()   every frame: how far the paper moved (phase correlation on a
 │                           small grey crop) + how sharp the frame is
 ├─ ShotTracker.update()     state machine:
 │      AIMING ──(sudden jump or blur)──► RECOIL ──(steady & sharp for SETTLE_SECONDS)──► SETTLED
 │        │ at the jump: _pick_poa() takes the last steady frame BEFORE it = point of aim,
 │        │              and copies it out of the short frame buffer immediately
 │        └ at SETTLED:  the 3 sharpest settled frames = AFTER frames
 ├─ measure_shot()           impact.measure(POA, each AFTER frame) → median, agreement check
 ├─ arduino_link.send_result()   clicks to the Arduino (if SERIAL_PORT is set)
 └─ saves shot_XX/ pictures, timeline.png, shots_summary.json
```

Memory: only `BUFFER_SECONDS` (1 s) of full frames is kept; the motion/sharpness numbers
for every frame are kept for the timeline. Ctrl+C or Q stops cleanly and still saves.

---

## 5. "I want to change …" → where

| I want to … | Change |
| --- | --- |
| Distance, grid square size, click value | `settings.py`: `DISTANCE_YD`, `GRID_SQUARE_IN`, `CLICK_MOA` |
| Use a new model | `settings.py`: `MODEL` (and class names if they changed) |
| Make small dots/impacts easier to detect | `settings.py`: `YOLO_ZOOM` |
| Hide less/more around the reticle | `settings.py`: `RETICLE_MASK`, `RETICLE_MARGIN_PX`, `RETICLE_DOT_PX` |
| Shot detection too sensitive / not sensitive enough | `settings.py`: `SPIKE_PX`, `SETTLE_PX`, `SETTLE_SECONDS` (look at `timeline.png`) |
| Camera, resolution, frame rate | `settings.py`: `CAMERA`, `CAMERA_WIDTH/HEIGHT/FPS` |
| Gearbox, turret, motor direction | `settings.py`: `GEAR_RATIO`, `MOA_PER_TURRET_REV`, `WINDAGE_DIRECTION`, `ELEVATION_DIRECTION` |
| What is sent to the Arduino | `arduino_link.py → send_result()` / `move()` and the sketch |
| How the result picture looks | `impact.py → render()`; step pictures: `tools/walkthrough.py` |
| How MOA/clicks are computed | `impact.py → express()` |

---

## 6. How to check that it works

```powershell
.\.venv\Scripts\python.exe tests\test_impact.py --trials 60     # accuracy on simulated shots
.\.venv\Scripts\python.exe tests\test_tracker.py --trials 12    # shot detection on simulated videos
.\.venv\Scripts\python.exe tests\test_arduino.py                # the serial messages
.\.venv\Scripts\python.exe tools\walkthrough.py --video videos\camera_videoall.mp4 --before-frame 801 --after-frame 831
```

Current simulated accuracy: median 0.054 MOA, every shot within 1 click. On the real video,
steps 1–7 work; step 8 (impact) needs the model to be trained on this camera's footage.

---

## 7. Honest notes on the structure

* `impact.py` is the busiest file (~600 lines). Its steps are separate functions in the order
  they run (`_aim_point`, `_reticle_mask`, `_relate`, `_scale`, `_new_impact`, `_result`), so
  read `measure()` first: it is a short table of contents.
* `scope.py` mixes several scope-specific jobs; they are independent functions, so it is a
  toolbox rather than a chain.
* The "not part of the pipeline" files in section 2 are the main source of clutter. Removing
  them, plus a slim Pi entry point, is the planned streamlined version.
