"""Pieces shared by AirPen's input modes (camera_input.py and glove_input.py)."""

from dataclasses import dataclass
import math


@dataclass
class InputFrame:
    """What an input mode reports to AirPen for one screen update."""

    cursor: tuple | None = None       # Canvas pixel (x, y), or None when unknown.
    pen_down: bool = False            # Already debounced by the input mode.
    command: str | None = None        # One-shot: "submit", "space", "backspace" or "clear".
    status: str = ""                  # One line of live readings for the bottom of the canvas.
    banner: str | None = None         # A big instruction, such as a calibration step.
    message: str | None = None        # A one-off notice, shown briefly.
    gesture: str | None = None        # Command whose hand shape is being held.
    gesture_progress: float = 0.0     # 0-1 of the hold needed to trigger that command.
    preview: object = None            # Camera picture for the camera window, if any.


class OneEuroFilter:
    """Smooth a pointer: strongly when it is nearly still, lightly when it moves.

    From Casiez et al., "1 Euro Filter" (CHI 2012). The cutoff frequency rises
    with speed, so a resting hand stops jittering while fast strokes do not lag.
    """

    def __init__(self, min_cutoff, beta, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    def reset(self):
        self.value = None
        self.speed = 0.0
        self.time = None

    @staticmethod
    def _alpha(elapsed, cutoff):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * elapsed))

    def __call__(self, value, now):
        if self.value is None:
            self.value, self.time = float(value), now
            return self.value
        elapsed = now - self.time
        if elapsed <= 0:
            return self.value
        self.time = now
        speed = (value - self.value) / elapsed
        self.speed += self._alpha(elapsed, self.derivative_cutoff) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * abs(self.speed)
        self.value += self._alpha(elapsed, cutoff) * (value - self.value)
        return self.value
