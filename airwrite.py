import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ══════════════════════════════════════════
#  CAMERA: 0 = laptop, "http://x.x.x.x:8080/video" = phone
# ══════════════════════════════════════════
CAMERA_SOURCE = "http://192.168.0.103:8080/video"
MODEL_PATH    = "hand_landmarker.task"

# ════════════════════════════════
#  TrOCR — Microsoft handwriting model
#  Downloads ~1.5GB on first run
#  then works completely offline
# ════════════════════════════════
print("Loading TrOCR handwriting model...")
print("First run downloads ~1.5GB — please wait...")
processor = TrOCRProcessor.from_pretrained(
    'microsoft/trocr-base-handwritten',
    local_files_only=False
)
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-base-handwritten'
)
trocr_model.eval()
print("TrOCR ready")

# ── Colors ──
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 120)
YELLOW = (0, 220, 255)
GRAY   = (180, 180, 180)
RED    = (0, 80, 255)
ORANGE = (0, 165, 255)

# ── State ──
state     = "idle"
last_word = ""
submitted = []

# ── Gesture hold ──
last_gesture   = ""
gesture_frames = 0
GESTURE_HOLD   = 20

# ── Drawing ──
latest_landmarks = None
canvas   = None
prev_x, prev_y = None, None

# ════════════════════════════════
#  MEDIAPIPE SETUP
# ════════════════════════════════
def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
    result_callback=on_result
)
detector = HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ════════════════════════════════
#  GESTURE DETECTION
# ════════════════════════════════
def detect_gesture(lm):
    tips      = [8, 12, 16, 20]
    pips      = [6, 10, 14, 18]
    extended  = [lm[tips[i]].y < lm[pips[i]].y for i in range(4)]
    thumb_out = lm[4].x < lm[3].x
    if extended[0] and not extended[1] and not extended[2] and not extended[3]:
        return "index_up"
    if not any(extended):
        return "fist"
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "peace"
    if all(extended) and thumb_out:
        return "palm"
    return "none"

def is_pen_down(lm):
    tx, ty = lm[4].x, lm[4].y
    ix, iy = lm[8].x, lm[8].y
    return ((tx-ix)**2 + (ty-iy)**2)**0.5 < 0.12

def draw_hand(frame, lm, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (100, 200, 100), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, GREEN, -1)

# ════════════════════════════════
#  TrOCR WORD CLASSIFIER
#  Reads the whole word at once
# ════════════════════════════════
def preprocess_for_trocr(canvas):
    """
    Crop to written area, invert to white bg,
    resize to TrOCR expected format
    """
    gray_check = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_check) < 100:
        return None

    # Find written area
    coords = cv2.findNonZero(gray_check)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)

    # Add padding
    pad = 30
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(canvas.shape[1], x + bw + pad)
    y2  = min(canvas.shape[0], y + bh + pad)
    cropped = canvas[y1:y2, x1:x2]

    # Invert — white background, black text
    inverted = cv2.bitwise_not(cropped)

    # Scale up to minimum height TrOCR works well with
    h_crop, w_crop = inverted.shape[:2]
    target_h = 64
    if h_crop < target_h:
        scale    = target_h / h_crop
        new_w    = int(w_crop * scale)
        inverted = cv2.resize(inverted, (new_w, target_h),
                              interpolation=cv2.INTER_LINEAR)

    # Convert to PIL RGB (TrOCR requires RGB)
    pil_img = Image.fromarray(
        cv2.cvtColor(inverted, cv2.COLOR_BGR2RGB)
    )
    return pil_img


def classify_word():
    pil_img = preprocess_for_trocr(canvas)
    if pil_img is None:
        print("Canvas empty — write something first")
        return "?"

    # Show preview
    preview_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    preview    = cv2.resize(preview_np, (400, 100),
                            interpolation=cv2.INTER_AREA)
    cv2.imshow("TrOCR sees this", preview)

    # Run TrOCR inference
    try:
        pixel_values = processor(
            images=pil_img,
            return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            generated = trocr_model.generate(
                pixel_values,
                max_new_tokens=30
            )

        word = processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0]

        word = word.upper().strip()
        word = ''.join(c for c in word
                       if c.isalpha() or c == ' ').strip()

        if word:
            print(f"TrOCR recognized: {word}")
            return word
        else:
            print("TrOCR returned empty — write larger")
            return "?"

    except Exception as e:
        print(f"TrOCR error: {e}")
        return "?"

# ════════════════════════════════
#  HUD DISPLAY
# ════════════════════════════════
def draw_hud(frame, gesture, pen_down):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    state_colors = {
        "idle":        GRAY,
        "drawing":     GREEN,
        "recognizing": ORANGE
    }
    cv2.putText(frame, f"State: {state.upper()}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                state_colors.get(state, GRAY), 2)
    pen_color = RED if pen_down else GRAY
    cv2.putText(frame,
                "PEN DOWN — writing" if pen_down else "PEN UP — move freely",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pen_color, 1)
    cv2.putText(frame, f"Gesture: {gesture}",
                (w-230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    if state == "drawing":
        cv2.putText(frame, "PEACE = recognize word",
                    (w-265, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h-115), (w, h), (20, 20, 20), -1)
    hints = {
        "idle":        "Raise INDEX finger to start writing",
        "drawing":     "THUMB+INDEX=write  |  LIFT=move  |  PEACE=recognize  |  PALM=cancel",
        "recognizing": "TrOCR reading your handwriting — please wait...",
    }
    cv2.putText(frame, hints.get(state, ""),
                (10, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)

    display = last_word if last_word else "___"
    cv2.putText(frame, f"Word:  {display}",
                (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, GREEN, 2)

    if submitted:
        cv2.putText(frame, f"History: {' | '.join(submitted[-4:])}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)

# ════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════
cap = cv2.VideoCapture(CAMERA_SOURCE)
frame_ts = 0

print("═══════════════════════════════════════════════")
print("  AirPen — Whole Word Recognition via TrOCR")
print("═══════════════════════════════════════════════")
print("  1. INDEX up     → enter writing mode")
print("  2. THUMB+INDEX  → pen down, write word")
print("  3. LIFT THUMB   → pen up, reposition")
print("  4. PEACE sign ✌ → TrOCR reads whole word")
print("  5. PALM         → clear, start over")
print("  C key = clear  |  ESC = quit")
print("  TIP: Write BIG, CLEAR, connected letters")
print("  NOTE: Recognition takes 1-3 sec on CPU")
print("═══════════════════════════════════════════════")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera — check CAMERA_SOURCE")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        print(f"Canvas ready: {w}x{h}")

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 1
    detector.detect_async(mp_image, frame_ts)

    gesture  = "none"
    pen_down = False
    lm       = latest_landmarks

    if lm:
        draw_hand(frame, lm, w, h)
        tip_x = int(lm[8].x * w)
        tip_y = int(lm[8].y * h)

        gesture  = detect_gesture(lm)
        pen_down = is_pen_down(lm)

        if gesture == last_gesture:
            gesture_frames += 1
        else:
            last_gesture   = gesture
            gesture_frames = 0
        triggered = (gesture_frames == GESTURE_HOLD)

        # ── STATE MACHINE ──
        if state == "idle":
            if triggered and gesture == "index_up":
                state  = "drawing"
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = None, None
                print("Writing mode — write full word then peace sign")

        elif state == "drawing":
            if gesture == "index_up":
                if pen_down:
                    if prev_x is not None:
                        cv2.line(canvas,
                                 (prev_x, prev_y),
                                 (tip_x, tip_y),
                                 WHITE, 20)
                    prev_x, prev_y = tip_x, tip_y
                else:
                    prev_x, prev_y = None, None
            else:
                prev_x, prev_y = None, None

            # Peace = recognize whole word
            if triggered and gesture == "peace":
                state = "recognizing"
                print("Recognizing — hold still...")

            # Palm = clear
            if triggered and gesture == "palm":
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                state  = "idle"
                prev_x, prev_y = None, None
                print("Cleared")

        dot_color = RED if pen_down else GREEN
        cv2.circle(frame, (tip_x, tip_y), 10, dot_color, -1)
        cv2.circle(frame, (tip_x, tip_y), 10, WHITE, 1)

    # ── RECOGNITION ──
    if state == "recognizing":
        combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
        draw_hud(combined, gesture, pen_down)
        cv2.imshow("AirPen - TrOCR Word Recognition", combined)
        cv2.waitKey(1)

        word = classify_word()
        if word != "?":
            last_word = word
            submitted.append(word)
            print(f"Result: {word}")
        else:
            last_word = "unclear — write bigger"

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        state  = "idle"
        prev_x, prev_y = None, None
        continue

    combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
    draw_hud(combined, gesture, pen_down)
    cv2.imshow("AirPen - TrOCR Word Recognition", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('c'):
        canvas    = np.zeros((h, w, 3), dtype=np.uint8)
        state     = "idle"
        last_word = ""
        prev_x, prev_y = None, None
        print("Cleared")

detector.close()
cap.release()
cv2.destroyAllWindows()
print("AirPen closed.")