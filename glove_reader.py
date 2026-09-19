"""Read AirPen glove data from an ESP32 over USB serial.

The ESP32 sends comma-separated values at 115200 baud:
heading,pitch,roll,flex_index,flex_middle,pen_down,gesture[,gyro_x,gyro_y,gyro_z]
The three gyro rates (degrees per second) come from the MicroPython firmware;
older firmware sends only the first seven values.
"""

import json
import math
import os
from pathlib import Path
import threading
import time

import serial


SERIAL_PORT = os.environ.get("GLOVE_SERIAL_PORT", "COM3")
BAUD_RATE = 115200
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
# Pixels per degree. At 30 the canvas is about 53 degrees of wrist turn wide
# and 30 degrees tall; at 16 it took 100 degrees, which was tiring to cross.
SCALE = float(os.environ.get("GLOVE_SCALE", "30.0"))
# Degrees per sample to ignore. Recorded drawing showed any dead zone throws
# away the slow movements that form letters (0.12 discarded 30-60% of them),
# and the BNO055's fused angles are steady enough at rest to need none.
DEAD_ZONE = float(os.environ.get("GLOVE_DEAD_ZONE", "0.0"))
SMOOTHING_ALPHA = float(os.environ.get("GLOVE_SMOOTHING", "0.55"))
MAX_ANGLE_STEP = float(os.environ.get("GLOVE_MAX_ANGLE_STEP", "4.0"))
# Which way each axis moves depends on how the sensor sits on the glove.
# Set either to -1 when that axis moves the wrong way.
HEADING_DIRECTION = float(os.environ.get("GLOVE_HEADING_DIRECTION", "1"))
PITCH_DIRECTION = float(os.environ.get("GLOVE_PITCH_DIRECTION", "1"))
# GLOVE_LOG=1 records every sample to glove_logs/ for debugging; any other
# non-empty value is used as the log file path.
LOG_SETTING = os.environ.get("GLOVE_LOG", "").strip()
LOG_DIRECTORY = Path(__file__).resolve().with_name("glove_logs")

# Steering. The fused angles become unstable when the sensor points close to
# straight up or down (gimbal lock), which is how it sits on this glove, so the
# cursor follows the gyro's rotation speed instead. glove_axes.json holds, for
# this particular mounting, which mix of the sensor's x/y/z rotation means
# "left/right" and which means "up/down". Without that file, or with
# GLOVE_POINTER=angles, the reader falls back to steering by angles.
AXES_FILE = Path(__file__).resolve().with_name("glove_axes.json")
POINTER = os.environ.get("GLOVE_POINTER", "gyro").strip().lower()
GYRO_DEADBAND = float(os.environ.get("GLOVE_GYRO_DEADBAND", "1.5"))  # deg/s of sensor noise
# While the pen is up, fast turns carry the cursor further (up to 1 + MOVE_BOOST
# times, reached at MOVE_FAST_SPEED deg/s), so crossing the canvas takes a
# quick flick. While drawing, the gain stays constant so letters keep shape.
MOVE_BOOST = float(os.environ.get("GLOVE_MOVE_BOOST", "2.0"))
MOVE_FAST_SPEED = 90.0
MAX_GYRO_RATE = 500.0  # deg/s; anything faster is a bad sample.
MAX_GYRO_STEP = 0.05   # seconds; longer gaps are not integrated.


def _load_axes():
    if POINTER != "gyro" or not AXES_FILE.exists():
        return None
    axes = json.loads(AXES_FILE.read_text(encoding="utf-8"))
    return tuple(axes["x"]), tuple(axes["y"])


GYRO_AXES = _load_axes()

# Pen. The PC decides pen up/down from the raw finger reading, so it can be
# tuned here without uploading new firmware (the firmware's own pen flag is
# ignored). Curling makes the reading fall. The pen goes down when the
# filtered reading drops below PEN_DOWN_BELOW, and lifts as soon as it rises
# PEN_RELEASE_RISE above the lowest point of the curl (a small unbend), or
# passes PEN_UP_ABOVE. A sensor that jumps straight between ~0 and ~250 gives
# no in-between readings, so on it only the jump itself lifts the pen.
PEN_DOWN_BELOW = float(os.environ.get("GLOVE_PEN_DOWN_BELOW", "45"))
PEN_UP_ABOVE = float(os.environ.get("GLOVE_PEN_UP_ABOVE", "70"))
PEN_RELEASE_RISE = float(os.environ.get("GLOVE_PEN_RELEASE_RISE", "25"))
PEN_MEDIAN_SAMPLES = 5   # 0.1 s median: ignores one- and two-sample spikes.
# No new sample for this long (the motion sensor dropping out, or USB) lifts
# the pen, so a line is never drawn straight across the gap.
PEN_STALE_SECONDS = 0.25


class _PenDetector:
    def __init__(self):
        self.recent = []
        self.down = False
        self.lowest = 0.0

    def update(self, reading):
        self.recent.append(reading)
        if len(self.recent) > PEN_MEDIAN_SAMPLES:
            self.recent.pop(0)
        value = sorted(self.recent)[len(self.recent) // 2]
        if self.down:
            self.lowest = min(self.lowest, value)
            if value > PEN_UP_ABOVE or value > self.lowest + PEN_RELEASE_RISE:
                self.down = False
        elif value < PEN_DOWN_BELOW:
            self.down = True
            self.lowest = value
        return self.down


_pen = _PenDetector()

_state = {
    "canvas_x": CANVAS_WIDTH // 2,
    "canvas_y": CANVAS_HEIGHT // 2,
    "pen_down": False,
    "gesture": False,
    "ready": False,
    "flex_i": 0,
    "flex_m": 0,
    "heading": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "last_sample": 0.0,
}
_lock = threading.Lock()
_position_x = float(CANVAS_WIDTH // 2)
_position_y = float(CANVAS_HEIGHT // 2)
_previous_heading = None
_previous_pitch = None
_filtered_heading_change = 0.0
_filtered_pitch_change = 0.0
_last_gyro_time = None
_capture = None  # (time, gyro) samples while a direction calibration records
_running = False
_thread = None


def configure_canvas(width, height):
    """Match the reader coordinate limits to the OpenCV canvas."""
    global CANVAS_WIDTH, CANVAS_HEIGHT
    CANVAS_WIDTH = int(width)
    CANVAS_HEIGHT = int(height)
    recenter()


def _parse(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) not in (7, 10):
        return None
    try:
        gyro = tuple(float(part) for part in parts[7:10]) if len(parts) == 10 else (0.0, 0.0, 0.0)
        return {
            "heading": float(parts[0]),
            "pitch": float(parts[1]),
            "roll": float(parts[2]),
            "flex_i": int(parts[3]),
            "flex_m": int(parts[4]),
            "pen_down": parts[5] == "1",
            "gesture": parts[6] == "1",
            "gyro": gyro,
        }
    except ValueError:
        return None


def _integrate(heading, pitch):
    global _position_x, _position_y, _previous_heading, _previous_pitch
    global _filtered_heading_change, _filtered_pitch_change
    if _previous_heading is None:
        _previous_heading = heading
        _previous_pitch = pitch
        return int(_position_x), int(_position_y)

    # Both Euler axes can cross the 0/360 boundary. Treat them as circular so
    # crossing that boundary cannot throw the cursor to the top or bottom.
    heading_change = (heading - _previous_heading + 180.0) % 360.0 - 180.0
    pitch_change = (pitch - _previous_pitch + 180.0) % 360.0 - 180.0

    # A bad IMU sample should never create a large cursor jump.
    heading_change = max(-MAX_ANGLE_STEP, min(MAX_ANGLE_STEP, heading_change))
    pitch_change = max(-MAX_ANGLE_STEP, min(MAX_ANGLE_STEP, pitch_change))

    if abs(heading_change) < DEAD_ZONE:
        heading_change = 0.0
    else:
        heading_change -= DEAD_ZONE if heading_change > 0 else -DEAD_ZONE
    if abs(pitch_change) < DEAD_ZONE:
        pitch_change = 0.0
    else:
        pitch_change -= DEAD_ZONE if pitch_change > 0 else -DEAD_ZONE

    # Filter only active movement. Resetting immediately at rest prevents a
    # filtered value from making the cursor drift after the hand stops.
    if heading_change == 0.0:
        _filtered_heading_change = 0.0
    else:
        _filtered_heading_change = (
            SMOOTHING_ALPHA * heading_change
            + (1.0 - SMOOTHING_ALPHA) * _filtered_heading_change
        )
    if pitch_change == 0.0:
        _filtered_pitch_change = 0.0
    else:
        _filtered_pitch_change = (
            SMOOTHING_ALPHA * pitch_change
            + (1.0 - SMOOTHING_ALPHA) * _filtered_pitch_change
        )

    _position_x += HEADING_DIRECTION * _filtered_heading_change * SCALE
    _position_y += PITCH_DIRECTION * _filtered_pitch_change * SCALE
    _position_x = max(0.0, min(float(CANVAS_WIDTH - 1), _position_x))
    _position_y = max(0.0, min(float(CANVAS_HEIGHT - 1), _position_y))
    _previous_heading = heading
    _previous_pitch = pitch
    return int(_position_x), int(_position_y)


CTRL_D = b"\x04"  # MicroPython soft-reboot key.


def _restart_board(connection):
    """Reboot the ESP32 so its main.py firmware is running and sending data.

    The USB adapter's RTS line drives the board's reset pin, so a short pulse
    restarts it. This also recovers a board left idle at the MicroPython
    prompt, for example after Ctrl+C in Thonny. Ctrl+D afterwards covers
    adapters that do not wire RTS to reset.
    """
    try:
        connection.dtr = False
        connection.rts = True
        time.sleep(0.1)
        connection.rts = False
    except (OSError, serial.SerialException):
        pass  # Not every adapter exposes these lines; the Ctrl+D below may still work.
    time.sleep(1.0)
    try:
        connection.write(CTRL_D)  # Soft reboot if the board sits at the prompt.
    except (OSError, serial.SerialException):
        pass
    time.sleep(2.0)  # Give MicroPython and the BNO055 time to start.


def _open_log():
    """Open the per-sample debug log when GLOVE_LOG is set, else return None."""
    if LOG_SETTING in ("", "0"):
        return None
    if LOG_SETTING == "1":
        LOG_DIRECTORY.mkdir(exist_ok=True)
        path = LOG_DIRECTORY / time.strftime("glove_%Y%m%d_%H%M%S.csv")
    else:
        path = Path(LOG_SETTING)
    log_file = open(path, "w", buffering=1, encoding="utf-8")
    log_file.write("time_s,heading,pitch,roll,flex_i,flex_m,pen_down,gesture,canvas_x,canvas_y,gyro_x,gyro_y,gyro_z\n")
    print(f"[glove] logging every sample to {path.resolve()}")
    return log_file


def _dead_band(rate):
    """Shrink a rotation speed towards zero by the noise level, keeping it smooth."""
    if abs(rate) <= GYRO_DEADBAND:
        return 0.0
    return rate - GYRO_DEADBAND if rate > 0 else rate + GYRO_DEADBAND


def _integrate_gyro(gyro, now):
    """Move the cursor by how fast the finger rotates, like an air mouse."""
    global _position_x, _position_y, _last_gyro_time
    if _last_gyro_time is None:
        _last_gyro_time = now
        return int(_position_x), int(_position_y)
    elapsed = now - _last_gyro_time
    _last_gyro_time = now
    if elapsed <= 0 or elapsed > MAX_GYRO_STEP:
        return int(_position_x), int(_position_y)

    rates = [max(-MAX_GYRO_RATE, min(MAX_GYRO_RATE, value)) for value in gyro]
    x_axis, y_axis = GYRO_AXES
    across = sum(weight * rate for weight, rate in zip(x_axis, rates))
    down = sum(weight * rate for weight, rate in zip(y_axis, rates))
    across, down = _dead_band(across), _dead_band(down)
    gain = SCALE
    if not _pen.down:
        speed = (across * across + down * down) ** 0.5
        gain *= 1.0 + MOVE_BOOST * min(1.0, speed / MOVE_FAST_SPEED)
    _position_x += HEADING_DIRECTION * across * elapsed * gain
    _position_y += PITCH_DIRECTION * down * elapsed * gain
    _position_x = max(0.0, min(float(CANVAS_WIDTH - 1), _position_x))
    _position_y = max(0.0, min(float(CANVAS_HEIGHT - 1), _position_y))
    return int(_position_x), int(_position_y)


def _reader(port, baud):
    global _running
    print(f"[glove] opening {port} @ {baud}...")
    try:
        connection = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as error:
        print(f"[glove] cannot open {port}: {error}")
        print("[glove] close Arduino Serial Monitor, check the port, and restart AirPen")
        _running = False
        return

    _restart_board(connection)
    connection.reset_input_buffer()
    print("[glove] connected; waiting for BNO055 calibration")
    if GYRO_AXES is not None:
        print(f"[glove] steering with the gyro, using {AXES_FILE.name}")
    else:
        print("[glove] steering with angles (no glove_axes.json, or GLOVE_POINTER=angles)")
    log_file = _open_log()

    while _running:
        try:
            raw = connection.readline().decode("utf-8", errors="ignore")
            if raw.startswith("#"):
                print("[glove]", raw.strip())
                continue
            data = _parse(raw)
            if data is None:
                continue
            now = time.monotonic()
            pen_down = _pen.update(data["flex_i"])
            if GYRO_AXES is not None and len(raw.split(",")) == 10:
                canvas_x, canvas_y = _integrate_gyro(data["gyro"], now)
            else:
                canvas_x, canvas_y = _integrate(data["heading"], data["pitch"])
            with _lock:
                _state.update(
                    canvas_x=canvas_x,
                    canvas_y=canvas_y,
                    pen_down=pen_down,
                    gesture=data["gesture"],
                    ready=True,
                    flex_i=data["flex_i"],
                    flex_m=data["flex_m"],
                    heading=data["heading"],
                    pitch=data["pitch"],
                    roll=data["roll"],
                    last_sample=now,
                )
            if _capture is not None:
                with _lock:
                    if _capture is not None:
                        _capture.append((now, data["gyro"]))
            if log_file is not None:
                log_file.write(
                    f"{now:.3f},{data['heading']:.2f},{data['pitch']:.2f},{data['roll']:.2f},"
                    f"{data['flex_i']},{data['flex_m']},{int(pen_down)},"
                    f"{int(data['gesture'])},{canvas_x},{canvas_y},"
                    f"{data['gyro'][0]:.2f},{data['gyro'][1]:.2f},{data['gyro'][2]:.2f}\n"
                )
        except (OSError, serial.SerialException) as error:
            print(f"[glove] serial connection lost: {error}")
            break

    connection.close()
    if log_file is not None:
        log_file.close()
    with _lock:
        _state["ready"] = False
        _state["pen_down"] = False
        _state["gesture"] = False
    _running = False


def start(port=None, baud=BAUD_RATE):
    """Start the background reader. Call once before the UI loop."""
    global _running, _thread
    if _running:
        return
    _running = True
    _thread = threading.Thread(
        target=_reader,
        args=(port or SERIAL_PORT, baud),
        daemon=True,
        name="airpen-glove-reader",
    )
    _thread.start()


def stop():
    global _running
    _running = False


def get_state():
    """Return (x, y, pen_down, recognition_gesture, ready)."""
    with _lock:
        age = time.monotonic() - _state["last_sample"]
        fresh = age < 1.5
        pen_fresh = age < PEN_STALE_SECONDS
        return (
            _state["canvas_x"],
            _state["canvas_y"],
            _state["pen_down"] if pen_fresh else False,
            _state["gesture"] if pen_fresh else False,
            _state["ready"] and fresh,
        )


def get_flex():
    with _lock:
        return _state["flex_i"], _state["flex_m"]


def set_scale(pixels_per_degree):
    """Change the writing size live (pixels of cursor travel per degree turned)."""
    global SCALE
    SCALE = max(5.0, min(120.0, float(pixels_per_degree)))
    return SCALE


def get_angles():
    """Return the latest (heading, pitch, roll) in degrees, for on-screen debugging."""
    with _lock:
        return _state["heading"], _state["pitch"], _state["roll"]


def start_capture():
    """Begin recording gyro samples for a direction calibration."""
    global _capture
    with _lock:
        _capture = []


def take_capture():
    """Stop recording and return the (time, gyro) samples collected."""
    global _capture
    with _lock:
        samples, _capture = _capture or [], None
    return samples


def _net_turn(samples, bias):
    """Total rotation, in degrees about each sensor axis, over a recording."""
    total = [0.0, 0.0, 0.0]
    for (previous_time, _), (sample_time, gyro) in zip(samples, samples[1:]):
        elapsed = min(MAX_GYRO_STEP, sample_time - previous_time)
        for axis in range(3):
            total[axis] += (gyro[axis] - bias[axis]) * elapsed
    return total


def calibrate_axes(still, right, down):
    """Learn which rotation means right and which means down, and start using it.

    The sensor sits at a different angle each time the glove is put on, so
    the mix of its x/y/z rotation that makes a "right" or "down" movement
    changes too. Given recordings of holding still, turning right, and
    turning down, this saves the new mix to glove_axes.json and applies it
    at once. Raises ValueError with advice when a recording is unusable.
    """
    global GYRO_AXES
    if not still:
        raise ValueError("No glove data arrived - is the glove connected?")
    bias = [sum(gyro[axis] for _, gyro in still) / len(still) for axis in range(3)]
    directions = []
    for name, samples in (("RIGHT", right), ("DOWN", down)):
        turn = _net_turn(samples, bias)
        size = sum(value * value for value in turn) ** 0.5
        if size < 10.0:
            raise ValueError(f"Turn further {name} (only {size:.0f} degrees) - try again")
        directions.append([value / size for value in turn])
    right_axis, down_axis = directions
    overlap = sum(a * b for a, b in zip(right_axis, down_axis))
    if abs(overlap) > 0.7:
        raise ValueError("RIGHT and DOWN turns were too alike - try again")

    # Separate the two exactly, so a pure right turn moves the cursor only
    # sideways even though the measured directions are not at 90 degrees.
    scale = 1.0 - overlap * overlap
    x_axis = [(r - overlap * d) / scale for r, d in zip(right_axis, down_axis)]
    y_axis = [(d - overlap * r) / scale for r, d in zip(right_axis, down_axis)]
    AXES_FILE.write_text(json.dumps({
        "note": "Gyro mix for left/right (x) and up/down (y), measured "
                + time.strftime("%Y-%m-%d %H:%M")
                + " with the K calibration in AirPen. Re-measure whenever the sensor moves.",
        "x": [round(value, 4) for value in x_axis],
        "y": [round(value, 4) for value in y_axis],
    }, indent=2) + "\n", encoding="utf-8")
    GYRO_AXES = (tuple(x_axis), tuple(y_axis))
    return math.degrees(math.acos(min(1.0, abs(overlap))))


def recenter():
    """Place the cursor in the canvas centre without drawing a connecting line."""
    global _position_x, _position_y, _previous_heading, _previous_pitch
    global _filtered_heading_change, _filtered_pitch_change, _last_gyro_time
    _position_x = float(CANVAS_WIDTH // 2)
    _position_y = float(CANVAS_HEIGHT // 2)
    _previous_heading = None
    _previous_pitch = None
    _last_gyro_time = None
    _filtered_heading_change = 0.0
    _filtered_pitch_change = 0.0
    with _lock:
        _state["canvas_x"] = int(_position_x)
        _state["canvas_y"] = int(_position_y)


if __name__ == "__main__":
    import sys

    test_port = sys.argv[1] if len(sys.argv) > 1 else SERIAL_PORT
    start(port=test_port)
    try:
        while True:
            time.sleep(0.1)
            x, y, pen, gesture, ready = get_state()
            flex_i, flex_m = get_flex()
            print(
                f"x={x:4d} y={y:4d} pen={'DOWN' if pen else 'up':4s} "
                f"gesture={int(gesture)} ready={int(ready)} "
                f"flex={flex_i}/{flex_m}"
            )
    except KeyboardInterrupt:
        stop()
