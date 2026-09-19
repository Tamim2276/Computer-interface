"""Turn a glove recording into a debugging report.

Run AirPen with GLOVE_LOG=1 (run_glove.bat does this), write something, quit,
then run:

    airwrite_env\\Scripts\\python.exe glove_debug.py            (newest recording)
    airwrite_env\\Scripts\\python.exe glove_debug.py <file.csv> (a specific one)

It prints a summary of the pen and movement, and saves a picture next to the
recording (same name, .png) with the drawing on top and, underneath, a
timeline of the finger sensor, the pen state, and how fast the hand turned.
"""

from pathlib import Path
import csv
import sys

import cv2
import numpy as np

LOG_DIRECTORY = Path(__file__).resolve().with_name("glove_logs")
SAMPLE_SECONDS = 0.02    # The firmware sends 50 samples a second.
BLIP_SECONDS = 0.25      # Matches MIN_STROKE_SECONDS in airwrite.py.
STUCK_SECONDS = 10.0     # A pen held down this long is probably stuck.
WIDTH = 1600
TIMELINE_HEIGHT = 420


def load(path):
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no samples")
    columns = {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}
    columns["time_s"] -= columns["time_s"][0]
    return columns


def strokes(pen):
    """Return (start, end) sample indexes of every pen-down period."""
    edges = np.flatnonzero(np.diff(np.r_[0, pen.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def summary(data):
    pen = data["pen_down"] > 0.5
    periods = strokes(pen)
    durations = [(end - start) * SAMPLE_SECONDS for start, end in periods]
    flex = data["flex_i"]
    lines = [
        f"length: {data['time_s'][-1]:.0f} s, {len(flex)} samples",
        f"pen-down periods: {len(periods)} "
        f"({sum(d < BLIP_SECONDS for d in durations)} blips under {BLIP_SECONDS} s, "
        f"{sum(d > STUCK_SECONDS for d in durations)} longer than {STUCK_SECONDS:.0f} s)",
        f"pen down {pen.mean():.0%} of the time",
        f"finger sensor while pen up: median {np.median(flex[~pen]) if (~pen).any() else float('nan'):.0f}, "
        f"while pen down: median {np.median(flex[pen]) if pen.any() else float('nan'):.0f}",
        f"finger sensor readings between 45 and 70 (neither clearly up nor down): {np.mean((flex > 45) & (flex < 70)):.0%}",
    ]
    if "gyro_x" in data:
        rate = np.linalg.norm(np.c_[data["gyro_x"], data["gyro_y"], data["gyro_z"]], axis=1)
        lines.append(f"hand turning speed: median {np.median(rate):.0f} deg/s, still (<3 deg/s) {np.mean(rate < 3):.0%} of the time")
    return lines


def draw_path(data):
    image = np.zeros((900, WIDTH, 3), np.uint8)
    pen = data["pen_down"] > 0.5
    x = data["canvas_x"].astype(int)
    y = data["canvas_y"].astype(int)
    for index in range(1, len(x)):
        start, end = (x[index - 1], y[index - 1]), (x[index], y[index])
        if pen[index] and pen[index - 1]:
            cv2.line(image, start, end, (255, 255, 255), 8, cv2.LINE_AA)
        else:
            cv2.line(image, start, end, (70, 70, 70), 1, cv2.LINE_AA)  # Pen-up travel.
    cv2.putText(image, "drawing (white = pen down, grey = pen-up movement)", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    return image


def draw_timeline(data):
    image = np.full((TIMELINE_HEIGHT, WIDTH, 3), 18, np.uint8)
    time_s = data["time_s"]
    total = max(time_s[-1], 1e-6)
    to_x = lambda seconds: (seconds / total * (WIDTH - 1)).astype(int)
    pen = data["pen_down"] > 0.5
    for start, end in strokes(pen):
        cv2.rectangle(image, (int(to_x(time_s[start:start + 1])[0]), 0),
                      (int(to_x(time_s[end - 1:end])[0]), TIMELINE_HEIGHT - 1), (40, 70, 40), -1)

    top = 40
    flex_height = 220
    flex_to_y = lambda value: (top + flex_height - np.clip(value, 0, 400) / 400 * flex_height).astype(int)
    for level, label in ((45, "down below 45"), (70, "up above 70")):
        line_y = int(flex_to_y(np.array([level]))[0])
        cv2.line(image, (0, line_y), (WIDTH, line_y), (0, 120, 200), 1)
        cv2.putText(image, label, (WIDTH - 170, line_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 255), 1, cv2.LINE_AA)
    points = np.c_[to_x(time_s), flex_to_y(data["flex_i"])].astype(np.int32)
    cv2.polylines(image, [points], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "finger sensor (green background = pen down)", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)

    if "gyro_x" in data:
        rate = np.linalg.norm(np.c_[data["gyro_x"], data["gyro_y"], data["gyro_z"]], axis=1)
        base = TIMELINE_HEIGHT - 20
        rate_to_y = lambda value: (base - np.clip(value, 0, 120) / 120 * 110).astype(int)
        points = np.c_[to_x(time_s), rate_to_y(rate)].astype(np.int32)
        cv2.polylines(image, [points], False, (255, 180, 80), 1, cv2.LINE_AA)
        cv2.putText(image, "hand turning speed (0-120 deg/s)", (16, base - 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 80), 1, cv2.LINE_AA)
    for second in range(0, int(total) + 1, 10):
        tick = int(second / total * (WIDTH - 1))
        cv2.putText(image, f"{second}s", (tick + 2, TIMELINE_HEIGHT - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return image


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        recordings = sorted(LOG_DIRECTORY.glob("*.csv"), key=lambda item: item.stat().st_mtime)
        if not recordings:
            raise SystemExit(f"No recordings in {LOG_DIRECTORY}; run AirPen with GLOVE_LOG=1 first.")
        path = recordings[-1]
    data = load(path)
    print(path)
    for line in summary(data):
        print("  " + line)
    report = np.vstack([draw_path(data), draw_timeline(data)])
    output = path.with_suffix(".png")
    cv2.imwrite(str(output), report)
    print(f"report picture: {output}")


if __name__ == "__main__":
    main()
