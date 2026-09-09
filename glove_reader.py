"""Read AirPen glove data from an ESP32 over USB serial.

The ESP32 sends seven comma-separated values at 115200 baud:
heading,pitch,roll,flex_index,flex_middle,pen_down,gesture
"""

import os
import threading
import time

import serial


SERIAL_PORT = os.environ.get("GLOVE_SERIAL_PORT", "COM3")
BAUD_RATE = 115200
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
# A little more gain means less wrist travel. The smaller dead zone avoids the
# sticky feeling caused by ignoring normal slow writing movements.
SCALE = float(os.environ.get("GLOVE_SCALE", "16.0"))
DEAD_ZONE = float(os.environ.get("GLOVE_DEAD_ZONE", "0.12"))
SMOOTHING_ALPHA = float(os.environ.get("GLOVE_SMOOTHING", "0.68"))
MAX_ANGLE_STEP = float(os.environ.get("GLOVE_MAX_ANGLE_STEP", "4.0"))

_state = {
    "canvas_x": CANVAS_WIDTH // 2,
    "canvas_y": CANVAS_HEIGHT // 2,
    "pen_down": False,
    "gesture": False,
    "ready": False,
    "flex_i": 0,
    "flex_m": 0,
    "last_sample": 0.0,
}
_lock = threading.Lock()
_position_x = float(CANVAS_WIDTH // 2)
_position_y = float(CANVAS_HEIGHT // 2)
_previous_heading = None
_previous_pitch = None
_filtered_heading_change = 0.0
_filtered_pitch_change = 0.0
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
    if len(parts) != 7:
        return None
    try:
        return {
            "heading": float(parts[0]),
            "pitch": float(parts[1]),
            "roll": float(parts[2]),
            "flex_i": int(parts[3]),
            "flex_m": int(parts[4]),
            "pen_down": parts[5] == "1",
            "gesture": parts[6] == "1",
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

    _position_x += _filtered_heading_change * SCALE
    _position_y -= _filtered_pitch_change * SCALE
    _position_x = max(0.0, min(float(CANVAS_WIDTH - 1), _position_x))
    _position_y = max(0.0, min(float(CANVAS_HEIGHT - 1), _position_y))
    _previous_heading = heading
    _previous_pitch = pitch
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

    time.sleep(2.0)  # Opening the port resets most ESP32 boards.
    connection.reset_input_buffer()
    print("[glove] connected; waiting for BNO055 calibration")

    while _running:
        try:
            raw = connection.readline().decode("utf-8", errors="ignore")
            if raw.startswith("#"):
                print("[glove]", raw.strip())
                continue
            data = _parse(raw)
            if data is None:
                continue
            canvas_x, canvas_y = _integrate(data["heading"], data["pitch"])
            with _lock:
                _state.update(
                    canvas_x=canvas_x,
                    canvas_y=canvas_y,
                    pen_down=data["pen_down"],
                    gesture=data["gesture"],
                    ready=True,
                    flex_i=data["flex_i"],
                    flex_m=data["flex_m"],
                    last_sample=time.monotonic(),
                )
        except (OSError, serial.SerialException) as error:
            print(f"[glove] serial connection lost: {error}")
            break

    connection.close()
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
        fresh = time.monotonic() - _state["last_sample"] < 1.5
        return (
            _state["canvas_x"],
            _state["canvas_y"],
            _state["pen_down"] if fresh else False,
            _state["gesture"] if fresh else False,
            _state["ready"] and fresh,
        )


def get_flex():
    with _lock:
        return _state["flex_i"], _state["flex_m"]


def recenter():
    """Place the cursor in the canvas centre without drawing a connecting line."""
    global _position_x, _position_y, _previous_heading, _previous_pitch
    global _filtered_heading_change, _filtered_pitch_change
    _position_x = float(CANVAS_WIDTH // 2)
    _position_y = float(CANVAS_HEIGHT // 2)
    _previous_heading = None
    _previous_pitch = None
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
