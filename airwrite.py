from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import warnings

# Suppress known non-actionable third-party startup messages before MediaPipe loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated.*",
    category=FutureWarning,
)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Set OCR_BACKEND=huggingface on the Pi to call the deployed cloud endpoint.
# Local is retained for desktop development and offline testing.
OCR_BACKEND = os.environ.get("OCR_BACKEND", "local").strip().lower()
if OCR_BACKEND == "local":
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers.utils import logging as transformers_logging
elif OCR_BACKEND == "huggingface":
    from ocr_client import OCRServiceError, recognize_image
else:
    raise ValueError("OCR_BACKEND must be 'local' or 'huggingface'.")

# 0 = laptop webcam. Replace with an IP-camera URL if required.
CAMERA_SOURCE = "http://192.168.0.100:8080/video"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Drawing feel
# A comfortable amount of amplification reduces arm movement while retaining
# enough edge room for normal writing.
AMPLIFY = 1.35
STROKE_WIDTH = 20
PEN_VOTE_FRAMES = 3
GESTURE_HOLD = 18  # About 0.6 seconds at 30 FPS.
SPACE_HOLD_FRAMES = 9
MIN_INK_PIXELS = 2_500
# A wide high-resolution surface gives a full sentence room without making
# fullscreen drawing depend on the incoming camera resolution.
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
MAX_UNDO_STEPS = 30
RESULT_DISPLAY_SECONDS = 6.0
# Light adaptive smoothing: stable when holding still, near-direct while moving.
CURSOR_DRAW_ALPHA = 0.84
CURSOR_HOVER_ALPHA = 0.90
MAX_CURSOR_SPEED = 50_000  # Effectively no motion clamp during normal use.
MAX_STROKE_SEGMENT = 320

CAMERA_WINDOW = "AirPen Camera"
CANVAS_WINDOW = "AirPen Canvas"
PREVIEW_WINDOW = "OCR sees this"
TROCR_MODEL = "microsoft/trocr-base-handwritten"

WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 220, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (45, 45, 45)
RED = (0, 80, 255)
ORANGE = (0, 165, 255)

if OCR_BACKEND == "local":
    # Leave CPU capacity for the camera/UI while local TrOCR is recognizing.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    TROCR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local TrOCR model on {TROCR_DEVICE}...")
    transformers_logging.set_verbosity_error()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(TROCR_DEVICE)
    trocr_model.eval()
    print("Local TrOCR ready")
else:
    print("Using Hugging Face OCR endpoint; no TrOCR model is loaded locally.")

# Application state
state = "ready"
input_mode = "word"
recognized_text = ""
submitted = []
last_result = ""
result_visible_until = 0.0
transcript_visible_until = 0.0
latest_landmarks = None
canvas = None
stroke_history = []
pen_votes = deque(maxlen=PEN_VOTE_FRAMES)
pen_down = False
prev_point = None
prev_midpoint = None
last_command = "none"
gesture_frames = 0
command_latched = False
start_frames = 0
windows_positioned = False
recognition_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trocr")
recognition_future = None
recognition_mode = "manual"
last_canvas_metrics_at = 0.0
canvas_metrics_dirty = True
cached_ink_box = None
cached_ink_pixels = 0
filtered_cursor = None
last_cursor_at = None


def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None


options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
    result_callback=on_result,
)
detector = HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def is_pinching(landmarks):
    """Return True when thumb and index fingertips touch (move mode)."""
    thumb, index = landmarks[4], landmarks[8]
    return ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5 < 0.08


def extended_fingers(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]


def is_writing_pose(landmarks):
    """Index-only writes; pinching lifts the virtual pen to reposition."""
    index, middle, ring, pinky = extended_fingers(landmarks)
    return index and not middle and not ring and not pinky and not is_pinching(landmarks)


def detect_command(landmarks):
    """Recognize deliberate commands; pinch is reserved for repositioning."""
    extended = extended_fingers(landmarks)
    thumb, index = landmarks[4], landmarks[8]
    thumb_index_distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "peace"
    # Do not use thumb direction: it differs for left and right hands.
    if all(extended):
        return "palm"
    # Keep a regular fist distinct from a writing pinch. A clear thumbs-up
    # becomes the space command; a closed fist does nothing.
    if not any(extended) and thumb_index_distance > 0.12:
        # A space gesture needs the thumb clearly above the wrist.
        if thumb.y < landmarks[0].y - 0.06:
            return "space"
        return "fist"
    return "none"


def is_start_gesture(landmarks):
    """An index finger held up starts a new writing session."""
    index, middle, ring, pinky = extended_fingers(landmarks)
    return index and not middle and not ring and not pinky


def amplified_point(landmarks, width, height):
    """Map the index tip to canvas coordinates, amplifying about the frame centre."""
    x = landmarks[8].x * (width - 1)
    y = landmarks[8].y * (height - 1)
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    x = center_x + (x - center_x) * AMPLIFY
    y = center_y + (y - center_y) * AMPLIFY
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def stabilize_cursor(raw_point, drawing):
    """Reject one-frame landmark spikes while staying responsive to hand motion."""
    global filtered_cursor, last_cursor_at
    now = time.monotonic()
    if filtered_cursor is None or last_cursor_at is None:
        filtered_cursor = (float(raw_point[0]), float(raw_point[1]))
        last_cursor_at = now
        return raw_point

    elapsed = min(0.10, max(1 / 120, now - last_cursor_at))
    last_cursor_at = now
    dx = raw_point[0] - filtered_cursor[0]
    dy = raw_point[1] - filtered_cursor[1]
    distance = float(np.hypot(dx, dy))
    max_distance = max(35.0, MAX_CURSOR_SPEED * elapsed)
    if distance > max_distance:
        # A real hand cannot move this far in one camera frame; clamp the spike.
        dx *= max_distance / distance
        dy *= max_distance / distance

    base_alpha = CURSOR_DRAW_ALPHA if drawing else CURSOR_HOVER_ALPHA
    # Faster movement gets less filtering, so the cursor stays fluid rather
    # than feeling like it is dragging behind the hand.
    alpha = min(0.98, base_alpha + min(distance, 180.0) / 180.0 * 0.14)
    filtered_cursor = (
        filtered_cursor[0] + alpha * dx,
        filtered_cursor[1] + alpha * dy,
    )
    return int(round(filtered_cursor[0])), int(round(filtered_cursor[1]))


def draw_hand(frame, landmarks, width, height):
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for first, second in CONNECTIONS:
        cv2.line(frame, points[first], points[second], (100, 200, 100), 1, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, GREEN, -1, cv2.LINE_AA)


def ink_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    return cv2.boundingRect(coords) if coords is not None else None


def ink_count(image):
    return cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def mark_canvas_changed():
    global canvas_metrics_dirty
    canvas_metrics_dirty = True


def canvas_metrics(force=False):
    """Refresh bounding-box/fill feedback at most ten times per second."""
    global cached_ink_box, cached_ink_pixels, canvas_metrics_dirty, last_canvas_metrics_at
    now = time.monotonic()
    if force or (canvas_metrics_dirty and now - last_canvas_metrics_at >= 0.1):
        cached_ink_box = ink_bbox(canvas)
        cached_ink_pixels = ink_count(canvas)
        canvas_metrics_dirty = False
        last_canvas_metrics_at = now
    return cached_ink_box, cached_ink_pixels


def preprocess_canvas(image):
    box = ink_bbox(image)
    if box is None or ink_count(image) < 100:
        return None

    x, y, width, height = box
    cropped = image[y:y + height, x:x + width]
    inverted = cv2.bitwise_not(cropped)
    target_height = 100
    target_width = max(1, int(width * target_height / height))
    resized = cv2.resize(inverted, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        resized, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def recognize_canvas_part(image):
    """Return TrOCR text and a displayable preview for one word region."""
    image = preprocess_canvas(image)
    if image is None:
        return "?", None

    preview_width = max(1, int(image.shape[1] * 150 / image.shape[0]))
    preview = cv2.resize(image, (preview_width, 150), interpolation=cv2.INTER_AREA)
    try:
        if OCR_BACKEND == "huggingface":
            text = recognize_image(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(TROCR_DEVICE)
            with torch.inference_mode():
                generated_ids = trocr_model.generate(pixel_values, max_new_tokens=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as error:
        print(f"OCR error: {error}")
        return "?", preview
    # Normalize independent word reads so the final transcript never mixes
    # model-predicted upper- and lowercase letters.
    cleaned_text = "".join(char for char in text if char.isprintable()).strip()
    return cleaned_text.upper() or "?", preview


def classify_word(canvas_snapshot):
    """Run the slow OCR model in the worker thread, never in the UI loop."""
    return recognize_canvas_part(canvas_snapshot)


def start_recognition(mode="manual"):
    global recognition_future, recognition_mode, state
    if recognition_future is None:
        recognition_mode = mode
        recognition_future = recognition_executor.submit(classify_word, canvas.copy())
        state = "recognizing"


def finish_recognition_if_ready():
    """Publish an OCR result without interrupting camera or canvas rendering."""
    global last_result, recognition_future, recognized_text
    global result_visible_until, state, pen_down
    if recognition_future is None or not recognition_future.done():
        return

    try:
        word, preview = recognition_future.result()
    except Exception as error:
        print(f"TrOCR worker error: {error}")
        word, preview = "?", None
    finally:
        recognition_future = None

    if preview is not None:
        cv2.imshow(PREVIEW_WINDOW, preview)
    if recognition_mode == "word":
        if word != "?":
            recognized_text += word
            submitted.append(word)
            # Keep each word in the hidden transcript. Peace on an empty pad
            # is the deliberate action that reveals the complete sentence.
            result_visible_until = 0.0
            print(f"Word saved: {word}")
            clear_canvas()
        else:
            last_result = "Could not read that word"
            result_visible_until = time.monotonic() + 2.5
            print("Word read was unclear; keeping the canvas so you can continue writing")
        pen_votes.clear()
        pen_down = False
        state = "writing"
        return

    if word != "?":
        recognized_text += word
        submitted.append(word)
        last_result = word
        result_visible_until = time.monotonic() + RESULT_DISPLAY_SECONDS
        print(f"Result: {word}")
    else:
        last_result = "Could not read that"
        result_visible_until = time.monotonic() + RESULT_DISPLAY_SECONDS
        print("Could not read the canvas; write bigger and wait for the green fill bar.")

    clear_canvas()
    pen_votes.clear()
    pen_down = False
    state = "writing"


def show_final_transcript():
    global last_result, result_visible_until, transcript_visible_until
    if recognized_text.strip():
        last_result = recognized_text.strip()
        result_visible_until = time.monotonic() + RESULT_DISPLAY_SECONDS
        transcript_visible_until = result_visible_until
        print(f"Final text: {last_result}")


def toggle_input_mode():
    """Switch modes only on a blank pad, so no handwriting is discarded."""
    global input_mode, last_result, result_visible_until
    if ink_count(canvas) >= 100:
        last_result = "Submit or clear canvas first"
        result_visible_until = time.monotonic() + 2.5
        return
    input_mode = "sentence" if input_mode == "word" else "word"
    last_result = f"{input_mode.title()} Mode"
    result_visible_until = time.monotonic() + 2.5
    print(f"Switched to {input_mode} mode")


def draw_canvas_overlay(image, cursor):
    """Draw UI-only feedback without adding it to the image sent to OCR."""
    height, width = image.shape[:2]
    for y in (height // 4, height // 2, 3 * height // 4):
        cv2.line(image, (0, y), (width, y), DARK_GRAY, 1, cv2.LINE_AA)

    box, ink_pixels = canvas_metrics()
    if box:
        x, y, box_width, box_height = box
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (40, 110, 40), 1, cv2.LINE_AA)

    progress = min(1.0, ink_pixels / MIN_INK_PIXELS)
    bar_x, bar_y = 20, height - 30
    bar_width, bar_height = width - 40, 14
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), GRAY, 1)
    color = GREEN if progress >= 1 else YELLOW
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), color, -1)
    label = "READY TO READ" if progress >= 1 else f"Add more ink: {int(progress * 100)}%"
    cv2.putText(image, label, (bar_x, bar_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if cursor is not None:
        cv2.circle(image, cursor, 7, RED if pen_down else WHITE, -1 if pen_down else 1, cv2.LINE_AA)

    if state == "ready":
        hint = "Index up: start writing"
    elif state == "recognizing":
        hint = "TrOCR is reading — your camera and canvas remain live"
    elif input_mode == "sentence":
        hint = "Sentence Mode: write one line with gaps | Peace: read full line | Fist: word mode"
    else:
        hint = "Word Mode: index writes | Pinch moves | Peace: read word | Thumbs up: space | Fist: sentence mode"
    cv2.putText(image, hint, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1, cv2.LINE_AA)
    if recognized_text and time.monotonic() < transcript_visible_until:
        preview_text = recognized_text[-45:]
        if preview_text.endswith(" "):
            preview_text += "[space]"
        cv2.putText(image, f"Text: {preview_text}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1, cv2.LINE_AA)
    if time.monotonic() < result_visible_until:
        banner = f"READ: {last_result}"
        (text_width, text_height), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
        x = max(20, (width - text_width) // 2)
        cv2.rectangle(image, (x - 18, 72), (x + text_width + 18, 72 + text_height + 28), (18, 70, 18), -1)
        cv2.putText(image, banner, (x, 72 + text_height + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.05, GREEN, 2, cv2.LINE_AA)


def draw_camera_hud(frame, raw_pen_down):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 54), (20, 20, 20), -1)
    mode = "READY — raise index finger" if state == "ready" else "WRITING"
    cv2.putText(frame, mode, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if state == "writing" else YELLOW, 1, cv2.LINE_AA)
    status = "WRITING - INDEX UP" if raw_pen_down else "MOVE - PINCH TO LIFT"
    cv2.putText(frame, f"{status}  ({len(pen_votes)}/{PEN_VOTE_FRAMES} vote samples)", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED if pen_down else GRAY, 1, cv2.LINE_AA)


def reset_stroke():
    global prev_point, prev_midpoint
    prev_point = None
    prev_midpoint = None


def begin_stroke():
    """Store a reversible canvas snapshot once per stroke for U/undo."""
    stroke_history.append(canvas.copy())
    if len(stroke_history) > MAX_UNDO_STEPS:
        stroke_history.pop(0)
    reset_stroke()


def draw_smooth_stroke(point):
    """Use midpoint interpolation plus anti-aliased lines for fluid curves."""
    global prev_point, prev_midpoint
    if prev_point is None:
        prev_point = point
        prev_midpoint = point
        cv2.circle(canvas, point, STROKE_WIDTH // 2, WHITE, -1, cv2.LINE_AA)
        mark_canvas_changed()
        return

    if np.hypot(point[0] - prev_point[0], point[1] - prev_point[1]) > MAX_STROKE_SEGMENT:
        # Never draw a long accidental line when MediaPipe briefly loses the fingertip.
        prev_point = point
        prev_midpoint = point
        cv2.circle(canvas, point, STROKE_WIDTH // 2, WHITE, -1, cv2.LINE_AA)
        mark_canvas_changed()
        return

    midpoint = ((prev_point[0] + point[0]) // 2, (prev_point[1] + point[1]) // 2)
    cv2.line(canvas, prev_midpoint, midpoint, WHITE, STROKE_WIDTH, cv2.LINE_AA)
    prev_point = point
    prev_midpoint = midpoint
    mark_canvas_changed()


def clear_canvas():
    global canvas
    canvas = np.zeros_like(canvas)
    stroke_history.clear()
    reset_stroke()
    mark_canvas_changed()


cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    detector.close()
    raise RuntimeError(f"Could not open camera source: {CAMERA_SOURCE!r}")

frame_timestamp = 0
print("AirPen started. Raise only your index finger for about 0.6 seconds to begin writing.")

try:
    while True:
        received, frame = cap.read()
        if not received:
            print("Camera frame could not be read; stopping.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        if not windows_positioned:
            cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow(CANVAS_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(CAMERA_WINDOW, 360, 270)
            cv2.resizeWindow(CANVAS_WINDOW, 1120, 630)
            cv2.moveWindow(CAMERA_WINDOW, 20, 20)
            cv2.moveWindow(CANVAS_WINDOW, 400, 20)
            windows_positioned = True

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp = max(frame_timestamp + 1, time.monotonic_ns() // 1_000_000)
        detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), frame_timestamp)

        cursor = None
        raw_pen_down = False
        landmarks = latest_landmarks
        if landmarks:
            draw_hand(frame, landmarks, width, height)
            command = detect_command(landmarks)
            previous_pen_down = pen_down
            moving = is_pinching(landmarks)
            raw_pen_down = is_writing_pose(landmarks) and command == "none"
            if moving or command != "none":
                # Pinch and commands must lift immediately; only starting a
                # new index-only stroke uses the short stability vote.
                pen_votes.clear()
                pen_down = False
            else:
                pen_votes.append(raw_pen_down)
                pen_down = len(pen_votes) == PEN_VOTE_FRAMES and sum(pen_votes) >= (PEN_VOTE_FRAMES // 2 + 1)
            raw_cursor = amplified_point(landmarks, canvas.shape[1], canvas.shape[0])
            cursor = stabilize_cursor(raw_cursor, pen_down)

            if command == "none":
                last_command, gesture_frames, command_latched = "none", 0, False
            elif command == last_command:
                gesture_frames += 1
            else:
                last_command, gesture_frames, command_latched = command, 1, False
            required_hold = SPACE_HOLD_FRAMES if command == "space" else GESTURE_HOLD
            command_triggered = gesture_frames >= required_hold and not command_latched
            if command_triggered:
                command_latched = True

            if state == "ready":
                start_frames = start_frames + 1 if is_start_gesture(landmarks) else 0
                if start_frames >= GESTURE_HOLD:
                    state = "writing"
                    start_frames = 0
                    pen_votes.clear()
                    pen_down = False
                    print("Writing mode enabled")
            elif state == "writing":
                if command_triggered and command == "fist":
                    toggle_input_mode()
                elif command_triggered and command == "peace":
                    if ink_count(canvas) >= 100:
                        start_recognition("word" if input_mode == "word" else "manual")
                    else:
                        show_final_transcript()
                elif command_triggered and command == "palm":
                    clear_canvas()
                    print("Canvas cleared")
                elif command_triggered and command == "space":
                    if input_mode == "word" and ink_count(canvas) >= 100:
                        last_result = "Use peace to read this word"
                        result_visible_until = time.monotonic() + 2.5
                    elif input_mode == "word" and recognized_text and not recognized_text.endswith(" "):
                        recognized_text += " "
                        last_result = "SPACE ADDED"
                        result_visible_until = time.monotonic() + 3.0
                        transcript_visible_until = result_visible_until
                        print("Space added")
                elif pen_down:
                    if not previous_pen_down:
                        begin_stroke()
                    draw_smooth_stroke(cursor)
                elif previous_pen_down:
                    reset_stroke()
        else:
            pen_votes.clear()
            if pen_down:
                reset_stroke()
            pen_down = False
            filtered_cursor = None
            last_cursor_at = None

        finish_recognition_if_ready()

        camera_display = frame.copy()
        draw_camera_hud(camera_display, raw_pen_down)
        canvas_display = canvas.copy()
        draw_canvas_overlay(canvas_display, cursor)
        cv2.imshow(CAMERA_WINDOW, camera_display)
        cv2.imshow(CANVAS_WINDOW, canvas_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        if key in (ord("u"), ord("U")) and stroke_history:
            canvas = stroke_history.pop()
            reset_stroke()
            mark_canvas_changed()
        elif key in (ord("c"), ord("C")):
            clear_canvas()
            recognized_text = ""
        elif key == ord(" "):
            if recognized_text and not recognized_text.endswith(" "):
                recognized_text += " "
        elif key in (ord("f"), ord("F")):
            current = cv2.getWindowProperty(CANVAS_WINDOW, cv2.WND_PROP_FULLSCREEN)
            target = cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(CANVAS_WINDOW, cv2.WND_PROP_FULLSCREEN, target)
        elif key in (ord("m"), ord("M")):
            toggle_input_mode()
finally:
    detector.close()
    cap.release()
    recognition_executor.shutdown(wait=False, cancel_futures=True)
    cv2.destroyAllWindows()
