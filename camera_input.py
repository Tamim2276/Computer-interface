"""Camera input for AirPen: write in the air in front of a phone camera or webcam.

MediaPipe finds the hand in each camera frame. Pinching the thumb and index
together puts the pen down, and opening them lifts it. Hand shapes held for
half a second give commands. Everything that needs tuning for the camera is
in the settings below; the glove has its own in glove_input.py.
"""

import math
import os
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from input_common import InputFrame, OneEuroFilter

# ------------------------------------------------------------------ settings
# AIRPEN_LOG=1 records every camera frame (hand found, pinch, pen, cursor) to
# camera_logs/, for working out why a stroke broke or a gesture misfired.
LOG_SETTING = os.environ.get("AIRPEN_LOG", "").strip()
LOG_DIRECTORY = Path(__file__).resolve().with_name("camera_logs")
# The phone's address changes with the Wi-Fi network, so it lives in one small
# file, camera_address.txt, holding just the IP (e.g. 192.168.0.101). Press I
# in camera mode to type a new one; it reconnects and saves it there. A full
# URL or a webcam number (0) also works. AIRPEN_CAMERA, if set, overrides it.
ADDRESS_FILE = Path(__file__).with_name("camera_address.txt")
DEFAULT_ADDRESS = "192.168.0.101"
CAMERA_PORT = 8080     # The IP Webcam app's port and video path.
CAMERA_PATH = "/video"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
RECONNECT_SECONDS = 3.0
STALE_FRAME_SECONDS = 0.5  # Older than this, the camera has stalled.

# The middle of the camera view covers the whole canvas, so the hand never has
# to reach the edge of the picture, where tracking becomes unreliable. + and -
# change AMPLIFY live (how far the pen moves for a given hand movement).
AMPLIFY = float(os.environ.get("AIRPEN_CAMERA_REACH", "1.35"))
INPUT_CENTER_X = 0.50
INPUT_CENTER_Y = 0.45

# Pinch distance is measured as a share of the palm's length, so it works
# whether the hand is near to or far from the camera.
PINCH_CLOSE = 0.33   # Thumb and index tips closer than this: pen down.
PINCH_OPEN = 0.48    # Further apart than this: pen up. The gap stops flicker.
# The pen lifts only once the pinch has been open for this many frames in a
# row: a single noisy frame mid-stroke otherwise split strokes in two.
PINCH_RELEASE_FRAMES = 3
# A finger counts as extended when its tip is this much further from the
# wrist than its middle joint, which holds however the hand is turned.
EXTENDED_RATIO = 1.12
# If tracking loses the hand for a moment mid-stroke, keep the pen as it was.
LOST_HAND_GRACE_SECONDS = 0.3

GESTURE_HOLD_SECONDS = 0.5
POSE_DROPOUT_SECONDS = 0.15  # A held shape may flicker away this long.
GESTURE_COMMANDS = {
    "peace": "submit",
    "thumbs_up": "space",
    "three": "backspace",
    "palm": "clear",
}
GESTURE_LABELS = {
    "submit": "Submit word",
    "space": "Space",
    "backspace": "Backspace",
    "clear": "Clear canvas",
}

# Cursor smoothing (see OneEuroFilter): heavy when the hand is nearly still,
# light while it moves, so lines are steady without trailing behind the hand.
SMOOTH_MIN_CUTOFF = 1.0
SMOOTH_BETA = 0.007

HELP_LINES = [
    "CAMERA MODE",
    "Pinch thumb + index: draw",
    "Open them: move without drawing",
    "Hold a hand shape for 1/2 second:",
    "  Peace sign: submit word",
    "  Thumbs up: space",
    "  Three fingers: backspace",
    "  Open palm: clear canvas",
    "+ / -: reach",
    "[ / ]: pen thinner / thicker",
    "X: clear all submitted text",
    "I: type the phone's camera address",
    "V: switch to GLOVE",
]

FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def _distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def hand_shape(points):
    """Describe a hand from its 21 landmark points in pixels.

    Returns (pinch, pose): pinch is the thumb-to-index gap as a share of palm
    length, and pose is "peace", "three", "palm", "thumbs_up" or "other".
    """
    wrist = points[0]
    # Tilting the hand shortens the palm's length on camera and turning it
    # narrows its width, but rarely both; the larger stays steady either way.
    palm = max(1e-6, _distance(wrist, points[9]), 1.25 * _distance(points[5], points[17]))
    pinch = _distance(points[4], points[8]) / palm
    extended = [
        _distance(wrist, points[tip]) > EXTENDED_RATIO * _distance(wrist, points[pip])
        for pip, tip in FINGER_JOINTS.values()
    ]
    if extended == [True, True, False, False]:
        return pinch, "peace"
    if extended == [True, True, True, False]:
        return pinch, "three"
    if all(extended):
        return pinch, "palm"
    if not any(extended) and _thumb_is_up(points, palm):
        return pinch, "thumbs_up"
    return pinch, "other"


def _thumb_is_up(points, palm):
    """A thumbs-up: the thumb tip stands clearly above the rest of the fist."""
    thumb_tip = points[4]
    highest_other = min(points[index][1] for index in (5, 6, 8, 9, 12, 13, 16, 17, 20))
    return (thumb_tip[1] < highest_other - 0.35 * palm
            and _distance(thumb_tip, points[5]) > 0.6 * palm)


def load_address():
    """The camera address to use: AIRPEN_CAMERA, else camera_address.txt, else the default."""
    if os.environ.get("AIRPEN_CAMERA", "").strip():
        return os.environ["AIRPEN_CAMERA"].strip()
    if ADDRESS_FILE.exists():
        saved = ADDRESS_FILE.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    return DEFAULT_ADDRESS


def camera_url(address):
    """Turn "192.168.0.101" into the phone's stream URL; leave URLs and webcam numbers alone."""
    address = address.strip()
    if address.isdigit() or "://" in address:
        return address
    host = address if ":" in address else f"{address}:{CAMERA_PORT}"
    return f"http://{host}{CAMERA_PATH}"


def _open_source(source):
    return cv2.VideoCapture(int(source) if str(source).isdigit() else source)


class _CameraWorker:
    """Reads the camera and runs hand tracking away from the screen loop.

    One thread only reads frames and keeps the newest; another tracks the
    hand in whichever frame is newest. A network stream therefore never
    builds up a backlog, and the pen follows the hand as it is now.
    """

    def __init__(self, source):
        self.source = source
        self.running = True
        self.status = f"Connecting to camera {source} ..."
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._result = None  # (time, frame, landmarks or None)
        self._thread = threading.Thread(target=self._track, daemon=True, name="airpen-camera")
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._result

    def stop(self):
        self.running = False
        self._thread.join(timeout=3.0)

    def _read_frames(self, capture):
        while self.running:
            received, frame = capture.read()
            if not received:
                return
            with self._lock:
                self._frame, self._frame_time = frame, time.monotonic()

    def _track(self):
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        with HandLandmarker.create_from_options(options) as detector:
            last_timestamp = 0
            while self.running:
                self.status = f"Connecting to camera {self.source} ..."
                capture = _open_source(self.source)
                if not capture.isOpened():
                    capture.release()
                    self.status = (f"Cannot reach camera {self.source} - is the phone app running on the "
                                   "same Wi-Fi? Press I to type the phone's address")
                    self._wait(RECONNECT_SECONDS)
                    continue
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                reader = threading.Thread(target=self._read_frames, args=(capture,), daemon=True)
                reader.start()
                self.status = ""
                handled = 0.0
                while self.running and reader.is_alive():
                    with self._lock:
                        frame, frame_time = self._frame, self._frame_time
                    if frame is None or frame_time == handled:
                        time.sleep(0.002)
                        continue
                    handled = frame_time
                    frame = cv2.flip(frame, 1)  # Mirror, so moving right moves right.
                    last_timestamp = max(last_timestamp + 1, int(frame_time * 1000))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    found = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_timestamp)
                    landmarks = found.hand_landmarks[0] if found.hand_landmarks else None
                    with self._lock:
                        self._result = (frame_time, frame, landmarks)
                reader.join(timeout=2.0)
                capture.release()
                if self.running:
                    self.status = f"Camera stream stopped - reconnecting to {self.source} ..."
                    self._wait(1.0)

    def _wait(self, seconds):
        deadline = time.monotonic() + seconds
        while self.running and time.monotonic() < deadline:
            time.sleep(0.05)


class CameraInput:
    """AirPen input mode driven by hand tracking. See the module docstring."""

    name = "CAMERA"
    help_lines = HELP_LINES
    # The pen lifts PINCH_RELEASE_FRAMES after the fingers open, and opening
    # them nudges the pen point, so that last moment is trimmed off.
    end_trim_seconds = 0.12
    blip_seconds = 0.10      # Pinch detection is steady, so short dots are kept.

    def __init__(self):
        self._worker = None
        self._typing = None  # The address being typed after pressing I, or None.
        self._source = ""
        self._filter_x = OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA)
        self._filter_y = OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA)
        self._reset()

    def _reset(self):
        self._pen_down = False
        self._cursor = None
        self._last_frame_time = None
        self._last_hand_time = 0.0
        self._pose = None
        self._pose_since = 0.0
        self._pose_seen = 0.0
        self._pose_fired = False
        self._pinch = None
        self._pose_name = "none"
        self._open_frames = 0
        self._frame_times = []
        self._filter_x.reset()
        self._filter_y.reset()

    def start(self, width, height):
        self._reset()
        self._typing = None
        self._source = camera_url(load_address())
        self._worker = _CameraWorker(self._source)
        self._log = self._open_log()
        print(f"Camera mode: reading {self._source}")

    @property
    def captures_keys(self):
        """True while typing an address, so every key comes here first."""
        return self._typing is not None

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if getattr(self, "_log", None) is not None:
            self._log.close()
            self._log = None
        self._reset()

    def handle_key(self, key):
        global AMPLIFY
        if self._typing is not None:
            return self._type_address(key)
        if key in (ord("i"), ord("I")):
            self._typing = ""
            return None
        if key in (ord("+"), ord("=")):
            AMPLIFY = min(3.0, AMPLIFY * 1.1)
        elif key == ord("-"):
            AMPLIFY = max(0.5, AMPLIFY / 1.1)
        else:
            return None
        self._filter_x.reset()
        self._filter_y.reset()
        return f"Camera reach {AMPLIFY:.2f} (bigger = less hand movement)"

    def _type_address(self, key):
        """Collect an address typed on the keyboard; Enter connects, Esc cancels."""
        if key == 27:
            self._typing = None
            return "Camera address unchanged"
        if key == 8:
            self._typing = self._typing[:-1]
        elif key == 13:
            address, self._typing = self._typing.strip(), None
            if not address:
                return "Camera address unchanged"
            ADDRESS_FILE.write_text(address + "\n", encoding="utf-8")
            self.stop()
            self.start(0, 0)
            return f"Connecting to {self._source} (saved in {ADDRESS_FILE.name})"
        elif chr(key) in "0123456789.:" and len(self._typing) < 40:
            self._typing += chr(key)
        return None

    def read(self, width, height):
        if self._typing is not None:
            return InputFrame(banner=f"Phone camera address: {self._typing}_   (Enter: connect, Esc: cancel)",
                              status=f"Now reading {self._source}. Type the IP the phone app shows, "
                                     "for example 192.168.0.101")
        latest = self._worker.latest() if self._worker else None
        if latest is None:
            return InputFrame(status=self._worker.status if self._worker else "Camera stopped")
        frame_time, frame, landmarks = latest
        now = time.monotonic()
        preview = frame.copy()
        self._draw_reach_box(preview)

        if now - frame_time > STALE_FRAME_SECONDS:
            self._lose_hand(now)
            return InputFrame(status=self._worker.status or "Camera paused - waiting for frames",
                              preview=preview)

        new_frame = frame_time != self._last_frame_time
        if new_frame:
            self._last_frame_time = frame_time
            self._frame_times = [t for t in self._frame_times if frame_time - t < 1.0] + [frame_time]
            if landmarks is None:
                self._lose_hand(frame_time)
            else:
                self._update_hand(landmarks, frame, frame_time, width, height)
            self._write_log(frame_time, landmarks is not None)

        if landmarks is not None:
            self._draw_hand(preview, landmarks)
        command, label, progress = self._command(now)
        fps = len(self._frame_times)
        if landmarks is None:
            status = f"camera {fps} fps | show your hand to the camera"
        else:
            status = (f"camera {fps} fps | pinch {self._pinch:.2f} (draw below {PINCH_CLOSE:.2f}) | "
                      f"pen {'DOWN' if self._pen_down else 'up'} | reach {AMPLIFY:.2f}")
        return InputFrame(cursor=self._cursor, pen_down=self._pen_down and self._cursor is not None,
                          command=command, status=status, gesture=label,
                          gesture_progress=progress, preview=preview)

    # ------------------------------------------------------------ internals
    def _open_log(self):
        if LOG_SETTING in ("", "0"):
            return None
        LOG_DIRECTORY.mkdir(exist_ok=True)
        path = LOG_DIRECTORY / time.strftime("camera_%Y%m%d_%H%M%S.csv")
        log = open(path, "w", buffering=1, encoding="utf-8")
        log.write("time_s,hand,pinch,pose,pen_down,cursor_x,cursor_y\n")
        print(f"Camera mode: logging every frame to {path}")
        return log

    def _write_log(self, frame_time, hand_found):
        if self._log is None:
            return
        x, y = self._cursor if self._cursor is not None else ("", "")
        pinch = f"{self._pinch:.3f}" if self._pinch is not None else ""
        self._log.write(f"{frame_time:.3f},{int(hand_found)},{pinch},{self._pose_name if hand_found else ''},"
                        f"{int(self._pen_down)},{x},{y}\n")

    def _update_hand(self, landmarks, frame, frame_time, width, height):
        frame_height, frame_width = frame.shape[:2]
        points = [(mark.x * frame_width, mark.y * frame_height) for mark in landmarks]
        self._pinch, pose = hand_shape(points)
        self._pose_name = pose
        if not self._pen_down:
            self._pen_down = self._pinch < PINCH_CLOSE
            self._open_frames = 0
        elif self._pinch > PINCH_OPEN:
            self._open_frames += 1
            self._pen_down = self._open_frames < PINCH_RELEASE_FRAMES
        else:
            self._open_frames = 0
        self._last_hand_time = frame_time
        self._track_pose(pose if not self._pen_down else "other", frame_time)

        # The pen point is halfway between the thumb and index tips: it stays
        # put as they close, so starting a stroke does not make a hook.
        middle_x = (landmarks[4].x + landmarks[8].x) / 2
        middle_y = (landmarks[4].y + landmarks[8].y) / 2
        canvas_x = (0.5 + (middle_x - INPUT_CENTER_X) * AMPLIFY) * (width - 1)
        canvas_y = (0.5 + (middle_y - INPUT_CENTER_Y) * AMPLIFY) * (height - 1)
        canvas_x = self._filter_x(min(max(canvas_x, 0.0), width - 1.0), frame_time)
        canvas_y = self._filter_y(min(max(canvas_y, 0.0), height - 1.0), frame_time)
        self._cursor = (int(round(canvas_x)), int(round(canvas_y)))

    def _lose_hand(self, when):
        if self._pen_down and when - self._last_hand_time < LOST_HAND_GRACE_SECONDS:
            return  # A tracking blink: keep drawing from where the hand was.
        self._pen_down = False
        self._cursor = None
        self._pinch = None
        self._pose = None
        self._filter_x.reset()
        self._filter_y.reset()

    def _track_pose(self, pose, when):
        held = pose if pose in GESTURE_COMMANDS else None
        if held is not None and held == self._pose:
            self._pose_seen = when
        elif held is not None or when - self._pose_seen > POSE_DROPOUT_SECONDS:
            # A different shape, or the held one has really gone.
            self._pose, self._pose_since, self._pose_seen, self._pose_fired = held, when, when, False

    def _command(self, now):
        """Return (command to run now or None, label being held, hold progress)."""
        if self._pose is None or self._pose_fired:
            return None, None, 0.0
        command = GESTURE_COMMANDS[self._pose]
        progress = min(1.0, (now - self._pose_since) / GESTURE_HOLD_SECONDS)
        if progress >= 1.0:
            self._pose_fired = True
            return command, None, 0.0
        return None, GESTURE_LABELS[command], progress

    def _draw_reach_box(self, image):
        """Outline the part of the camera view that covers the canvas."""
        height, width = image.shape[:2]
        half = 0.5 / AMPLIFY
        top_left = (int((INPUT_CENTER_X - half) * width), int((INPUT_CENTER_Y - half) * height))
        bottom_right = (int((INPUT_CENTER_X + half) * width), int((INPUT_CENTER_Y + half) * height))
        cv2.rectangle(image, top_left, bottom_right, (80, 160, 80), 2, cv2.LINE_AA)

    def _draw_hand(self, image, landmarks):
        height, width = image.shape[:2]
        points = [(int(mark.x * width), int(mark.y * height)) for mark in landmarks]
        for first, second in HAND_CONNECTIONS:
            cv2.line(image, points[first], points[second], (100, 200, 100), 2, cv2.LINE_AA)
        colour = (0, 80, 255) if self._pen_down else (0, 220, 255)
        middle = ((points[4][0] + points[8][0]) // 2, (points[4][1] + points[8][1]) // 2)
        cv2.line(image, points[4], points[8], colour, 3, cv2.LINE_AA)
        cv2.circle(image, middle, 9, colour, -1 if self._pen_down else 2, cv2.LINE_AA)
