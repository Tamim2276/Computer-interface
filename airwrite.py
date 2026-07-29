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
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import easyocr

#  CAMERA: 0 = laptop, "http://x.x.x.x:8080/video" = phone
CAMERA_SOURCE = "http://192.168.0.108:8080/video"
MODEL_PATH    = "hand_landmarker.task"

#  EasyOCR Setup
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)  
print("EasyOCR ready")

#Colors
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 120)
YELLOW = (0, 220, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (45, 45, 45)
RED = (0, 80, 255)
ORANGE = (0, 165, 255)

# State
state     = "writing"  
last_word = ""
submitted = []

#Gesture hold
last_command   = "none"
gesture_frames = 0
GESTURE_HOLD   = 15    

#Drawing
latest_landmarks = None
canvas   = None
prev_x, prev_y = None, None

#  MEDIAPIPE SETUP

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

#  GESTURE LOGIC
def is_pen_down(lm):
    """ Simply checks the distance between thumb tip [4] and index tip [8] """
    tx, ty = lm[4].x, lm[4].y
    ix, iy = lm[8].x, lm[8].y
    # 0.08 is a firm pinch. The moment they separate slightly, it returns False.
    return ((tx-ix)**2 + (ty-iy)**2)**0.5 < 0.08

def detect_command(lm):
    """ Detects system commands (Peace/Palm) independent of drawing """
    tips      = [8, 12, 16, 20]
    pips      = [6, 10, 14, 18]
    extended  = [lm[tips[i]].y < lm[pips[i]].y for i in range(4)]
    thumb_out = lm[4].x < lm[3].x
    
    # Peace sign (Index and Middle open) -> Submit
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "peace"
        
    # High Five / Palm (All open) -> Clear canvas
    if all(extended) and thumb_out:
        return "palm"
        
    return "none"

def draw_hand(frame, lm, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (100, 200, 100), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, GREEN, -1)

#  IMAGE PREPROCESSOR

def preprocess_canvas(canvas):
    gray_check = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_check) < 100:
        return None

    coords = cv2.findNonZero(gray_check)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)

    cropped = canvas[y:y+bh, x:x+bw]
    inverted = cv2.bitwise_not(cropped)

    target_h = 100
    scale = target_h / bh
    target_w = int(bw * scale)
    resized = cv2.resize(inverted, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    padding = 30
    padded = cv2.copyMakeBorder(
        resized, 
        padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, 
        value=(255, 255, 255)
    )
    return padded

#  EASYOCR WORD RECOGNIZER

def classify_word():
    img_np = preprocess_canvas(canvas)
    if img_np is None:
        print("Canvas empty — write something first")
        return "?"

    h_preview, w_preview = img_np.shape[:2]
    preview_scale = 150.0 / h_preview
    preview_w = int(w_preview * preview_scale)
    preview = cv2.resize(img_np, (preview_w, 150), interpolation=cv2.INTER_AREA)
    cv2.imshow("EasyOCR sees this", preview)

    try:
        results = reader.readtext(img_np, detail=0)
        if not results:
            return "?"

        word = ' '.join(results).upper().strip()
        word = ''.join(c for c in word if c.isalpha() or c == ' ').strip()

        if word:
            return word
        return "?"
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return "?"

#  HUD DISPLAY

def draw_hud(frame, pen_down):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    state_color = GREEN if state == "writing" else ORANGE
    cv2.putText(frame, f"Mode: {state.upper()}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
    pen_color = RED if pen_down else GRAY
    pen_text = "PEN DOWN (Drawing...)" if pen_down else "PEN UP (Hovering)"
    cv2.putText(frame, pen_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pen_color, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h-115), (w, h), (20, 20, 20), -1)
    hints = {
        "writing":     "PINCH = Draw  |  RELEASE PINCH = Lift  |  PEACE = Read  |  PALM = Clear",
        "recognizing": "EasyOCR reading your handwriting — please wait...",
    }
    cv2.putText(frame, hints.get(state, ""),
                (10, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

    display = last_word if last_word else "___"
    cv2.putText(frame, f"Word:  {display}",
                (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, GREEN, 2)

    if submitted:
        cv2.putText(frame, f"History: {' | '.join(submitted[-4:])}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)

#  MAIN LOOP

cap = cv2.VideoCapture(CAMERA_SOURCE)
frame_ts = 0

print("═══════════════════════════════════════════════")
print("  AirPen — Natural Pinch & Write")
print("═══════════════════════════════════════════════")
print("  🤏  PINCH FINGERS  → Pen down, draw")
print("  👋  RELAX FINGERS  → Pen up, move freely")
print("  ✌️  PEACE SIGN     → Read word")
print("  🖐️  FLAT PALM      → Clear canvas")
print("  ESC = quit")
print("═══════════════════════════════════════════════")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 1
    detector.detect_async(mp_image, frame_ts)

    pen_down = False
    command  = "none"
    lm       = latest_landmarks

    if lm:
        draw_hand(frame, lm, w, h)
        
        # Track the index finger tip to draw
        tip_x = int(lm[8].x * w)
        tip_y = int(lm[8].y * h)

        # 1. Determine natural drawing state (Instant response)
        pen_down = is_pen_down(lm)
        
        # 2. Determine command state (Requires a brief hold to prevent accidents)
        command = detect_command(lm)
        if command == last_command and command != "none":
            gesture_frames += 1
        else:
            last_command   = command
            gesture_frames = 0
            
        triggered = (gesture_frames == GESTURE_HOLD)

        # STATE MACHINE
        if state == "writing":
            
            if triggered and command == "peace":
                state = "recognizing"
                print("Recognizing — hold still...")
                
            elif triggered and command == "palm":
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = None, None
                print("Cleared")
                
            else:
                # Drawing happens entirely naturally based on the pinch distance!
                if pen_down:
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (tip_x, tip_y), WHITE, 20)
                    prev_x, prev_y = tip_x, tip_y
                else:
                    # The absolute instant you stop pinching, the line breaks.
                    prev_x, prev_y = None, None

        # Draw the cursor: Red dot when writing, hollow white circle when hovering
        if pen_down:
            cv2.circle(frame, (tip_x, tip_y), 10, RED, -1)
        else:
            cv2.circle(frame, (tip_x, tip_y), 10, WHITE, 2)

    # RECOGNITION
    if state == "recognizing":
        combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
        draw_hud(combined, pen_down)
        cv2.imshow("AirPen", combined)
        cv2.waitKey(1)

        word = classify_word()
        if word != "?":
            last_word = word
            submitted.append(word)
            print(f"Result: {word}")
        else:
            last_word = "unclear — write bigger"

        # Auto-reset back to writing mode
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        state  = "writing"
        prev_x, prev_y = None, None
        continue

    # Standard loop rendering
    combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
    draw_hud(combined, pen_down)
    cv2.imshow("AirPen", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break
    if key == ord('c'):
        canvas    = np.zeros((h, w, 3), dtype=np.uint8)
        last_word = ""
        prev_x, prev_y = None, None

detector.close()
cap.release()
cv2.destroyAllWindows()
