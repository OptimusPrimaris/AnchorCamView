"""Turn the correction into motor steps and send it to the Arduino.

The board runs arduino/OpticMotorController/OpticMotorController.ino, which expects
one line per move:

    MOVE <windageSteps> <elevationSteps>\\n        e.g.  "MOVE -12 8"

and answers
    Windage command: -12
    Elevation command: 8
    ...moves windage first, then elevation...
    HOLDING        (both motors held powered for HOLD_TIME_MS so the turrets settle)
    DONE           (then the motors are released)

MOA -> steps (settings.py, section "Arduino and motors"):
    steps per MOA = MOTOR_STEPS_PER_REV x GEAR_RATIO / MOA_PER_TURRET_REV
    windage steps   = WINDAGE_DIRECTION   x round(windage MOA   x steps per MOA)   (+ = RIGHT)
    elevation steps = ELEVATION_DIRECTION x round(elevation MOA x steps per MOA)   (+ = UP)
The MOA values are used directly (not the rounded clicks), since the turrets' detents
are disabled and the motor can stop between clicks.

  python arduino_link.py --steps -12 8          # send a move by hand (windage, elevation)
  python arduino_link.py --moa 1.5 -2.0         # convert MOA to steps and send
  python arduino_link.py --moa 1.5 -2.0 --dry   # only show the steps, send nothing
"""
import argparse
import time
import settings as S


def steps_per_moa():
    return S.MOTOR_STEPS_PER_REV * S.GEAR_RATIO / S.MOA_PER_TURRET_REV


def motor_steps(windage_moa, elevation_moa):
    """Correction in MOA (+ = RIGHT / UP) -> (windage_steps, elevation_steps) for the sketch."""
    k = steps_per_moa()
    return (S.WINDAGE_DIRECTION * int(round(windage_moa * k)),
            S.ELEVATION_DIRECTION * int(round(elevation_moa * k)))


class ArduinoLink:
    def __init__(self, port, baud=None, connection=None, quiet=False):
        """connection: an already-open serial-like object (used by tests)."""
        self.quiet = quiet

        if connection is None:
            import serial
            connection = serial.Serial(
            port,
            baud or S.BAUD,
            timeout=0.2
        )
            self.ser = connection
            self._wait_ready(6.0)
        else:
            self.ser = connection

    def _wait_ready(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            line = self.ser.readline().decode("ascii", "replace").strip()
            if line == "READY" or line.startswith("ERROR"):
                if not self.quiet:
                    print("  Arduino:", line)
                return line == "READY"
        return False                                        # no reset on open: fine, already running

    def move(self, windage_steps, elevation_steps):
        """Send one MOVE and wait for DONE. Returns (ok, message).

        The move is sent once only: re-sending could make the turret move twice."""
        w, e = int(windage_steps), int(elevation_steps)
        if max(abs(w), abs(e)) > S.MAX_STEPS_PER_MOVE:
            return False, f"refused: {w}, {e} steps is more than MAX_STEPS_PER_MOVE ({S.MAX_STEPS_PER_MOVE})"
        self.ser.write(f"MOVE {w} {e}\n".encode("ascii"))
        self.ser.flush()
        step_rate = S.MOTOR_STEPS_PER_REV * S.MOTOR_SPEED_RPM / 60.0       # steps per second
        timeout = (3.0 + (abs(w) + abs(e)) / max(step_rate, 1e-6)        # moves happen one after another
                   + S.MOTOR_HOLD_MS / 1000.0)                             # then the hold before DONE
        got_w = got_e = None
        end = time.time() + timeout
        while time.time() < end:
            line = self.ser.readline().decode("ascii", "replace").strip()
            if not line:
                if not hasattr(self.ser, "in_waiting"):
                    break
                continue
            if line.startswith("Windage command:"):
                got_w = int(line.split(":")[1])
            elif line.startswith("Elevation command:"):
                got_e = int(line.split(":")[1])
            elif line.startswith("ERROR"):
                return False, f"Arduino: {line}"
            elif line == "DONE":
                if (got_w, got_e) != (w, e):
                    return False, f"Arduino moved {got_w}, {got_e} but {w}, {e} was sent"
                return True, f"moved windage {w:+d}, elevation {e:+d} steps (DONE)"
        if got_w is None:
            return False, "no answer from the Arduino (check the cable, SERIAL_PORT and that the sketch is running)"
        return False, f"Arduino started the move ({got_w}, {got_e}) but no DONE within {timeout:.0f} s"

    def close(self):
        self.ser.close()


def send_result(link, res):
    """Send a measured shot (from impact / shot_tracker). Returns a short status line."""
    if link is None:
        return None
    c = res.get("correction") if res.get("valid") else None
    if not c:
        return "Arduino: nothing sent (no valid measurement)"
    w, e = motor_steps(c["windage_moa"], c["elevation_moa"])
    ok, msg = link.move(w, e)
    return f"Arduino: MOVE {w} {e} -> {msg}" + ("" if ok else "  <-- CHECK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=S.SERIAL_PORT)
    p.add_argument("--baud", type=int, default=S.BAUD)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--steps", type=int, nargs=2, metavar=("WINDAGE", "ELEVATION"))
    g.add_argument("--moa", type=float, nargs=2, metavar=("WINDAGE", "ELEVATION"),
                   help="correction in MOA, + = RIGHT / UP")
    p.add_argument("--dry", action="store_true", help="only print the steps")
    a = p.parse_args()
    w, e = a.steps if a.steps else motor_steps(*a.moa)
    print(f"{steps_per_moa():.2f} steps per MOA -> MOVE {w} {e}")
    if a.dry:
        raise SystemExit
    if not a.port:
        raise SystemExit("Give --port (e.g. COM3 or /dev/ttyACM0) or set SERIAL_PORT in settings.py")
    link = ArduinoLink(a.port, a.baud)
    print(link.move(w, e)[1])
    link.close()
