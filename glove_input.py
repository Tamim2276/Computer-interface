"""Glove input for AirPen: an ESP32 glove with a BNO055 motion sensor and a flex sensor.

Turning the hand moves the cursor and curling the flex finger puts the pen
down (both worked out in glove_reader.py). The glove has no spare sensors for
gestures, so its commands come from the keyboard. Everything glove-specific,
including its K direction calibration, lives here and in glove_reader.py.
"""

import os
from pathlib import Path
import time

from serial.tools import list_ports

import glove_reader
from input_common import InputFrame

# "auto" finds the glove's USB port by itself, so it works on any computer
# whatever COM number Windows gives it. Set GLOVE_SERIAL_PORT=COM5 (say) to force one.
SERIAL_PORT = os.environ.get("GLOVE_SERIAL_PORT", "auto").strip() or "auto"
# USB-to-serial chips used on ESP32 boards.
GLOVE_CHIPS = ("CP210", "CH910", "CH34", "USB-SERIAL", "USB SERIAL", "UART")
RECONNECT_SECONDS = 5.0
# Sensitivity: how far the cursor moves per degree the hand turns. + and -
# change it, and the value is kept in this file for the next run.
SENSITIVITY_FILE = Path(__file__).with_name("glove_sensitivity.txt")

# K calibration: (name, seconds, instruction). The gyro is recorded during
# every step except "return", where the hand comes back to the middle.
CALIBRATION_STEPS = [
    ("still", 1.5, "Hold your hand still"),
    ("right", 2.5, "Turn slowly to the RIGHT, then hold"),
    ("return", 2.0, "Come back to the middle"),
    ("down", 2.5, "Turn slowly DOWN, then hold"),
]

HELP_LINES = [
    "GLOVE MODE",
    "Turn your hand: move",
    "Curl the flex finger: draw",
    "Straighten it: stop drawing",
    "Enter: submit word",
    "Space / Backspace: edit text",
    "K: calibrate directions",
    "R: centre cursor",
    "+ / -: sensitivity (remembered)",
    "[ / ]: pen thinner / thicker",
    "X: clear all submitted text",
    "V: switch to CAMERA",
]


def find_glove_port():
    """Return the COM port the glove is plugged into, or None if there is none."""
    if SERIAL_PORT.lower() != "auto":
        return SERIAL_PORT
    ports = list(list_ports.comports())
    for port in ports:
        described = f"{port.description} {port.manufacturer or ''}".upper()
        if any(chip in described for chip in GLOVE_CHIPS):
            return port.device
    return ports[0].device if len(ports) == 1 else None


class GloveInput:
    """AirPen input mode driven by the glove. See the module docstring."""

    name = "GLOVE"
    help_lines = HELP_LINES
    # Straightening the finger moves the hand before the flex sensor switches
    # the pen up, which drew a hook at the end of every stroke.
    end_trim_seconds = 0.25
    # Finger twitches and sensor flickers give brief false pen-downs.
    blip_seconds = 0.25

    def __init__(self):
        self._last_attempt = 0.0
        self._port = None
        self._message = None
        self._step = None  # Index into CALIBRATION_STEPS while calibrating.
        self._step_ends = 0.0
        self._recordings = {}

    def start(self, width, height):
        glove_reader.configure_canvas(width, height)
        if not os.environ.get("GLOVE_SCALE") and SENSITIVITY_FILE.exists():
            try:
                glove_reader.set_scale(float(SENSITIVITY_FILE.read_text(encoding="utf-8")))
            except ValueError:
                pass  # A damaged file: keep the default.
        self._last_attempt = time.monotonic()
        self._connect()

    def _connect(self):
        self._port = find_glove_port()
        if self._port is None:
            print("Glove mode: no glove USB port found yet - plug the glove in")
            return
        glove_reader.start(port=self._port)
        print(f"Glove mode: reading {self._port}")

    def stop(self):
        if self._step is not None:
            glove_reader.take_capture()
            self._step = None
        glove_reader.stop()

    def handle_key(self, key):
        if key in (ord("k"), ord("K")):
            return self._start_calibration()
        if key in (ord("r"), ord("R")):
            glove_reader.recenter()
            return "Cursor centred"
        if key in (ord("+"), ord("="), ord("-")):
            factor = 1.15 if key != ord("-") else 1 / 1.15
            sensitivity = glove_reader.set_scale(glove_reader.SCALE * factor)
            SENSITIVITY_FILE.write_text(f"{sensitivity:.1f}\n", encoding="utf-8")
            return f"Glove sensitivity {sensitivity:.0f} (lower = calmer, needs more hand movement)"
        return None

    def read(self, width, height):
        now = time.monotonic()
        if not glove_reader.is_running() and now - self._last_attempt > RECONNECT_SECONDS:
            self._last_attempt = now
            self._connect()  # The cable was out, or the port changed; look again.

        message, self._message = self._message, None
        banner = self._update_calibration(now)
        x, y, pen_down, _, ready = glove_reader.get_state()
        if not ready:
            where = f"on {self._port}" if self._port else "(no glove USB port found)"
            return InputFrame(
                status=f"Glove not connected {where} - plug in its USB cable and close Thonny "
                       "(retrying every few seconds)",
                banner=banner, message=message)

        flex, _ = glove_reader.get_flex()
        heading, pitch, roll = glove_reader.get_angles()
        status = (f"flex {flex:4d} | pen {'DOWN' if pen_down else 'up'} | sensitivity {glove_reader.SCALE:.0f} | "
                  f"heading {heading:5.1f} pitch {pitch:5.1f} roll {roll:5.1f}")
        return InputFrame(cursor=(x, y), pen_down=pen_down and self._step is None,
                          status=status, banner=banner, message=message)

    # ------------------------------------------------------ K calibration
    def _start_calibration(self):
        if self._step is not None:
            return None
        if not glove_reader.get_state()[4]:
            return "Connect the glove before calibrating"
        self._recordings.clear()
        self._step = 0
        self._step_ends = time.monotonic() + CALIBRATION_STEPS[0][1]
        glove_reader.start_capture()
        print("Calibrating glove directions")
        return None

    def _update_calibration(self, now):
        """Advance the calibration steps; return the instruction to show, if any."""
        if self._step is None:
            return None
        if now >= self._step_ends:
            name = CALIBRATION_STEPS[self._step][0]
            samples = glove_reader.take_capture()
            if name != "return":
                self._recordings[name] = samples
            self._step += 1
            if self._step == len(CALIBRATION_STEPS):
                self._step = None
                self._finish_calibration()
                return None
            next_name, seconds, _ = CALIBRATION_STEPS[self._step]
            self._step_ends = now + seconds
            if next_name != "return":
                glove_reader.start_capture()
        _, _, instruction = CALIBRATION_STEPS[self._step]
        remaining = max(0.0, self._step_ends - now)
        return f"CALIBRATE {self._step + 1}/{len(CALIBRATION_STEPS)}: {instruction}  ({remaining:.1f}s)"

    def _finish_calibration(self):
        try:
            angle = glove_reader.calibrate_axes(
                self._recordings["still"], self._recordings["right"], self._recordings["down"])
            self._message = "Directions calibrated"
            print(f"Glove directions calibrated ({angle:.0f} degrees apart); saved to glove_axes.json")
        except ValueError as error:
            self._message = str(error)
            print(f"Calibration failed: {error}")
        glove_reader.recenter()
