import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import easyocr

# ══════════════════════════════════════════
#  CAMERA: 0 = laptop, "http://x.x.x.x:8080/video" = phone
# ══════════════════════════════════════════
CAMERA_SOURCE =  "http://192.168.0.101:8080/video"  # Changed to 0 for quick testing, update to your phone IP if needed
MODEL_PATH    = "hand_landmarker.task"

# ════════════════════════════════
#  EASYOCR
# ════════════════════════════════
print("Loading EasyOCR model — please wait...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready")

# ── Colors ──
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 120)
YELLOW = (0, 220, 255)
GRAY   = (180, 180, 180)
RED    = (0, 80, 255)
ORANGE = (0, 165, 255)

# ── State ──
last_word = ""
submitted = []

# ── Gesture hold ──
last_gesture   = ""
gesture_frames = 0
GESTURE_HOLD   = 15  # Lowered from 20 for faster response

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
#  GESTURE DETECTION (Simplified & Improved)
# ════════════════════════════════
def get_pinch_midpoint(lm, w, h):
    # Returns the midpoint between thumb and index for smoother drawing
    mx = int((lm[4].x + lm[8].x) / 2 * w)
    my = int((lm[4].y + lm[8].y) / 2 * h)
    return mx, my

def detect_gesture(lm):
    # 1. Check if pinched (distance between thumb and index tip)
    pinch_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5
    if pinch_dist < 0.07:  # Tighter threshold for intentional writing
        return "write"
    
    # 2. Check extended fingers for commands
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    ext = [lm[tips[i]].y < lm[pips[i]].y for i in range(4)]
    
    if ext[0] and not ext[1] and not ext[2] and not ext[3]:
        return "hover"  # Only index up -> move without drawing
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        return "peace"  # Index + Middle -> submit/recognize
    if all(ext):
        return "palm"   # All fingers up -> clear canvas
    
    return "none"

def draw_hand(frame, lm, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (100, 200, 100), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, GREEN, -1)

# ════════════════════════════════
#  EASYOCR WITH IMPROVED PRE-PROCESSING
# ════════════════════════════════
def classify_word():
    gray_check = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_check) < 100:
        return "?"

    # 1. Invert (Black text on White background)
    inverted = cv2.bitwise_not(canvas)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    
    # 2. Find bounding box of drawing
    coords = cv2.findNonZero(gray_check)
    if coords is None:
        return "?"
    x, y, bw, bh = cv2.boundingRect(coords)
    
    # 3. Add GENEROUS padding (EasyOCR needs whitespace to recognize edges)
    pad = 50
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(canvas.shape[1], x + bw + pad)
    y2 = min(canvas.shape[0], y + bh + pad)
    cropped = inverted[y1:y2, x1:x2]

    # 4. Scale up
    scale = max(1, 400 // max(cropped.shape[0], cropped.shape[1]))
    scaled = cv2.resize(cropped, (cropped.shape[1]*scale, cropped.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

    # 5. Morphological Smoothing (Thickens and connects broken lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed = cv2.erode(scaled, kernel, iterations=1)
    processed = cv2.GaussianBlur(processed, (5, 5), 0)

    # Show preview of enhanced image
    preview = cv2.resize(processed, (300, 150), interpolation=cv2.INTER_AREA)
    cv2.imshow("EasyOCR sees this", preview)

    # 6. Run EasyOCR
    # Removed paragraph=True, added allowlist to stop it from guessing numbers/symbols
    result = reader.readtext(
        processed, 
        detail=0, 
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    )
    
    print(f"Raw OCR Output: {result}") # This will help debug if it misreads

    if result:
        # Join the list into a single string and uppercase it
        word = ' '.join(result).upper().strip()
        if word:
            return word
            
    return "?"

# ════════════════════════════════
#  HUD DISPLAY
# ════════════════════════════════
def draw_hud(frame, gesture):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    
    status_text = "WRITING" if gesture == "write" else ("HOVERING" if gesture == "hover" else "WAITING")
    status_color = RED if gesture == "write" else (GREEN if gesture == "hover" else GRAY)
    
    cv2.putText(frame, f"Mode: {status_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Gesture: {gesture}", (w-200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h-115), (w, h), (20, 20, 20), -1)
    hints = "PINCH to write | UN-PINCH to hover | PEACE to read | PALM to clear"
    cv2.putText(frame, hints, (10, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

    display = last_word if last_word else "___"
    cv2.putText(frame, f"Word:  {display}", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, GREEN, 2)

    if submitted:
        cv2.putText(frame, f"History: {' | '.join(submitted[-4:])}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

# ════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════
cap = cv2.VideoCapture(CAMERA_SOURCE)
frame_ts = 0

print("\n═══════════════════════════════════════════")
print("  AirPen — Improved Edition")
print("═══════════════════════════════════════════")
print("  1. PINCH fingers   → Pen down, draw")
print("  2. UN-PINCH (Index)→ Pen up, hover/move")
print("  3. PEACE sign ✌    → Recognize word")
print("  4. PALM            → Clear canvas")
print("═══════════════════════════════════════════\n")

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

    gesture = "none"
    lm      = latest_landmarks

    if lm:
        draw_hand(frame, lm, w, h)
        gesture = detect_gesture(lm)
        
        # Get midpoint of pinch for smoother drawing cursor
        cursor_x, cursor_y = get_pinch_midpoint(lm, w, h)

        # Trigger logic for commands
        if gesture == last_gesture and gesture in ["peace", "palm"]:
            gesture_frames += 1
        else:
            last_gesture   = gesture
            gesture_frames = 0
            
        triggered = (gesture_frames == GESTURE_HOLD)

        # ── DRAWING LOGIC ──
        if gesture == "write":
            if prev_x is not None:
                # Anti-aliased, thicker lines for better OCR reading
                cv2.line(canvas, (prev_x, prev_y), (cursor_x, cursor_y), WHITE, 25, cv2.LINE_AA)
            prev_x, prev_y = cursor_x, cursor_y
        else:
            prev_x, prev_y = None, None

        # ── COMMAND LOGIC ──
        if triggered and gesture == "peace":
            # Show processing state briefly
            cv2.putText(frame, "READING...", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ORANGE, 3)
            cv2.imshow("AirPen", frame)
            cv2.waitKey(1)
            
            word = classify_word()
            if word != "?":
                last_word = word
                submitted.append(word)
                print(f"Recognized: {word}")
            else:
                last_word = "unclear — write bigger"
                
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            gesture_frames = 0

        if triggered and gesture == "palm":
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            last_word = ""
            gesture_frames = 0
            print("Canvas Cleared")

        # Draw Cursor
        dot_color = RED if gesture == "write" else GREEN
        cv2.circle(frame, (cursor_x, cursor_y), 8, dot_color, -1)
        cv2.circle(frame, (cursor_x, cursor_y), 8, WHITE, 2)

    # Combine canvas and camera
    combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
    draw_hud(combined, gesture)
    cv2.imshow("AirPen", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

detector.close()
cap.release()
cv2.destroyAllWindows()