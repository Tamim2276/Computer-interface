"""AirPen: write words in the air and turn them into text.

Two input modes share one canvas and one piece of text:

    camera - hand tracking from a phone or webcam      (camera_input.py)
    glove  - the ESP32 motion-sensor glove             (glove_input.py)

Press V to switch between them. Each mode keeps its own settings and
calibration in its own file; this file holds what they share: the canvas and
its strokes, handwriting recognition, the text built up word by word, and the
on-screen help.

Start in a mode with INPUT_BACKEND=camera or INPUT_BACKEND=glove (camera is
the default). OCR_BACKEND=local runs TrOCR here; OCR_BACKEND=huggingface sends
the handwriting to the deployed endpoint instead.
"""

from concurrent.futures import ThreadPoolExecutor
import os
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
import numpy as np

# A wide high-resolution surface gives a full word or short sentence room,
# independent of the camera's resolution.
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
STROKE_WIDTH = 20          # Starting pen thickness; [ and ] change it live.
MIN_STROKE_WIDTH = 4
MAX_STROKE_WIDTH = 60
MIN_INK_PIXELS = 2_500     # The fill bar turns green here: enough ink to read.
MIN_READABLE_PIXELS = 100  # Less ink than this is treated as an empty canvas.
MAX_UNDO_STEPS = 30
# A jump this long between two samples is a tracking glitch, never a line.
MAX_STROKE_SEGMENT = 220
# A stroke this short (under the mode's blip_seconds and this length) is a pen
# blip, not writing, and is erased again when the pen lifts.
MIN_STROKE_PIXELS = 30
# The end of each stroke is trimmed by the mode's end_trim_seconds, but never
# by more than this share of the stroke, so short strokes survive.
END_TRIM_MAX_SHARE = 0.30
MESSAGE_SECONDS = 2.5

CANVAS_WINDOW = "AirPen Canvas"
CAMERA_WINDOW = "AirPen Camera"
OCR_WINDOW = "OCR sees this"
TROCR_MODEL = "microsoft/trocr-base-handwritten"

WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 220, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (45, 45, 45)
PANEL = (32, 32, 32)
RED = (0, 80, 255)
MODE_COLOURS = {"CAMERA": (255, 170, 60), "GLOVE": (60, 200, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX

KEY_HELP = ("Enter: submit | Space | Backspace | [ ]: pen thickness | C: clear canvas | X: clear text | "
            "U: undo | H: help | Esc: quit")


# --------------------------------------------------------- recognition
def load_recognizer():
    """Return a function that reads a BGR handwriting image and returns its text."""
    backend = os.environ.get("OCR_BACKEND", "local").strip().lower()
    if backend == "huggingface":
        from ocr_client import recognize_image
        print("Using the Hugging Face OCR endpoint; no TrOCR model is loaded locally.")
        return recognize_image
    if backend != "local":
        raise ValueError("OCR_BACKEND must be 'local' or 'huggingface'.")

    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers.utils import logging as transformers_logging

    # Leave CPU capacity for tracking and drawing while TrOCR is reading.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local TrOCR model on {device}...")
    transformers_logging.set_verbosity_error()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
    model.eval()
    print("Local TrOCR ready")

    def recognize(image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated = model.generate(pixel_values, max_new_tokens=64)
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    return recognize


def ink_box(image):
    coords = cv2.findNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return cv2.boundingRect(coords) if coords is not None else None


def ink_count(image):
    return cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def prepare_for_ocr(canvas):
    """Crop the ink, make it dark-on-white like paper, and pad it for TrOCR."""
    box = ink_box(canvas)
    if box is None or ink_count(canvas) < MIN_READABLE_PIXELS:
        return None
    x, y, width, height = box
    inverted = cv2.bitwise_not(canvas[y:y + height, x:x + width])
    target_height = 100
    target_width = max(1, int(width * target_height / height))
    resized = cv2.resize(inverted, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(resized, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=WHITE)


def read_canvas(recognize, canvas):
    """Return (WORD or "?", the image the recognizer saw). Runs off the UI thread."""
    image = prepare_for_ocr(canvas)
    if image is None:
        return "?", None
    try:
        text = recognize(image)
    except Exception as error:  # A failed read must never stop the app.
        print(f"OCR error: {error}")
        return "?", image
    # Upper case keeps words from separate reads consistent with each other.
    cleaned = "".join(char for char in str(text) if char.isprintable()).strip()
    return cleaned.upper() or "?", image


# -------------------------------------------------------------- strokes
def _midpoint(first, second):
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


class StrokeRenderer:
    """Draw a stroke as a smooth curve through its samples.

    Each new sample adds a curve from the middle of the previous segment to
    the middle of this one, bent towards the sample in between. Corners from
    the camera's 30 samples a second become smooth arcs, at a cost of half a
    segment of delay.
    """

    def __init__(self, canvas, width=STROKE_WIDTH):
        self.canvas = canvas
        self.width = width
        self.recent = []
        self.length = 0.0

    def add(self, point):
        point = (float(point[0]), float(point[1]))
        if self.recent and np.hypot(point[0] - self.recent[-1][0],
                                    point[1] - self.recent[-1][1]) > MAX_STROKE_SEGMENT:
            self.finish()  # Tracking jumped: start afresh instead of drawing a long line.
            self.recent = []
        if self.recent:
            self.length += float(np.hypot(point[0] - self.recent[-1][0], point[1] - self.recent[-1][1]))
        self.recent = (self.recent + [point])[-3:]
        if len(self.recent) == 1:
            self._line(point, point)
        elif len(self.recent) == 2:
            self._line(self.recent[0], _midpoint(*self.recent))
        else:
            first, control, last = self.recent
            self._curve(_midpoint(first, control), control, _midpoint(control, last))

    def finish(self):
        """Draw the last half segment up to the final sample."""
        if len(self.recent) >= 2:
            self._line(_midpoint(self.recent[-2], self.recent[-1]), self.recent[-1])

    def _curve(self, start, control, end):
        span = np.hypot(end[0] - start[0], end[1] - start[1]) + np.hypot(control[0] - start[0],
                                                                          control[1] - start[1])
        steps = max(2, int(span / 6))
        previous = start
        for step in range(1, steps + 1):
            t = step / steps
            point = ((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0],
                     (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1])
            self._line(previous, point)
            previous = point

    def _line(self, start, end):
        # OpenCV draws thick lines with round ends, so segments join smoothly.
        cv2.line(self.canvas, (int(round(start[0])), int(round(start[1]))),
                 (int(round(end[0])), int(round(end[1]))), WHITE, self.width, cv2.LINE_AA)


# ------------------------------------------------------------------ app
class AirPen:
    def __init__(self, recognize, first_mode="camera"):
        from camera_input import CameraInput
        from glove_input import GloveInput

        self.recognize = recognize
        self.inputs = {"camera": CameraInput(), "glove": GloveInput()}
        self.mode = first_mode if first_mode in self.inputs else "camera"
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8)
        self.history = []    # Canvas snapshots before each stroke, for undo.
        self.text = ""
        self.stroke = None   # StrokeRenderer while the pen is down.
        self.stroke_points = []
        self.stroke_started = 0.0
        self.frame = None    # The latest InputFrame.
        self.message = ""
        self.message_until = 0.0
        self.show_help = True
        self.stroke_width = STROKE_WIDTH
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        self.reading = None  # Future of the recognition in progress.
        self.space_after_reading = False  # Space asked for while a word is read.
        self._ink_cache = (None, 0)
        self._ink_dirty = True

    @property
    def active(self):
        return self.inputs[self.mode]

    # ---------------------------------------------------------- text
    def say(self, message, seconds=MESSAGE_SECONDS):
        self.message, self.message_until = message, time.monotonic() + seconds
        print(message)

    def has_ink(self):
        return ink_count(self.canvas) >= MIN_READABLE_PIXELS

    def submit(self):
        """Read the canvas and add the word to the text."""
        if self.reading is not None:
            return
        if not self.has_ink():
            self.say("Write a word first, then submit it")
            return
        self.end_stroke()
        self.reading = self.executor.submit(read_canvas, self.recognize, self.canvas.copy())
        self.say("Reading...", seconds=30)

    def add_space(self):
        """Add a space, finishing off whatever is on the canvas first.

        Moving the hand between gestures can pinch for a moment and leave a
        stray mark, and a word may still be being read, so a space asked for
        then is not refused: it follows the word once it has been read.
        """
        if self.reading is not None:
            self.space_after_reading = True
            self.say("Space will follow this word")
            return
        ink = ink_count(self.canvas)
        if ink >= MIN_INK_PIXELS:
            self.space_after_reading = True
            self.submit()  # A whole word: read it, then add the space.
            return
        if ink >= MIN_READABLE_PIXELS:
            self.clear_canvas()  # Only stray marks; drop them.
        if not self.text:
            self.say("Write and submit a word first")
        elif self.text.endswith(" "):
            self.say("There is already a space")
        else:
            self.text += " "
            self.say("Space added")

    def backspace(self):
        """Undo the last stroke while writing; otherwise remove the last space or word."""
        if self.has_ink():
            self.undo_stroke()
            self.say("Stroke removed")
        elif self.text.endswith(" "):
            self.text = self.text[:-1]
            self.say("Space removed")
        elif self.text:
            cut = self.text.rfind(" ") + 1
            removed, self.text = self.text[cut:], self.text[:cut]
            self.say(f"Removed {removed}")

    def finish_reading(self):
        if self.reading is None or not self.reading.done():
            return
        try:
            word, seen = self.reading.result()
        except Exception as error:
            print(f"OCR worker error: {error}")
            word, seen = "?", None
        self.reading = None
        if seen is not None:
            cv2.imshow(OCR_WINDOW, cv2.resize(seen, (max(1, int(seen.shape[1] * 150 / seen.shape[0])), 150)))
        space, self.space_after_reading = self.space_after_reading, False
        if word == "?":
            self.say("Could not read that - write bigger or clearer, then submit again")
            return
        self.text += word + (" " if space else "")
        self.clear_canvas()
        self.say(f"READ: {word}", seconds=3.0)
        print(f"Text so far: {self.text}")

    # -------------------------------------------------------- canvas
    def clear_canvas(self):
        self.end_stroke()
        self.canvas[:] = 0
        self.history.clear()
        self._ink_dirty = True

    def undo_stroke(self):
        self.end_stroke()
        if self.history:
            self.canvas = self.history.pop()
            self._ink_dirty = True

    def begin_stroke(self):
        self.history.append(self.canvas.copy())
        if len(self.history) > MAX_UNDO_STEPS:
            self.history.pop(0)
        self.stroke = StrokeRenderer(self.canvas, self.stroke_width)
        self.stroke_points = []
        self.stroke_started = time.monotonic()

    def add_point(self, point):
        self.stroke_points.append((time.monotonic(), point))
        self.stroke.add(point)
        self._ink_dirty = True

    def end_stroke(self):
        """Finish the stroke: erase it if it was a blip, otherwise trim its end."""
        if self.stroke is None:
            return
        now = time.monotonic()
        duration = now - self.stroke_started
        if duration < self.active.blip_seconds and self.stroke.length < MIN_STROKE_PIXELS:
            self.canvas = self.history.pop()
        else:
            trim = min(self.active.end_trim_seconds, END_TRIM_MAX_SHARE * duration)
            if trim > 0:
                # Redraw without the final moment, where lifting the pen
                # moved the hand and would leave a hook.
                self.canvas[:] = self.history[-1]
                redraw = StrokeRenderer(self.canvas, self.stroke.width)
                for moment, point in self.stroke_points:
                    if moment <= now - trim:
                        redraw.add(point)
                redraw.finish()
            else:
                self.stroke.finish()
        self.stroke = None
        self.stroke_points = []
        self._ink_dirty = True

    def ink_metrics(self):
        if self._ink_dirty:
            self._ink_cache = (ink_box(self.canvas), ink_count(self.canvas))
            self._ink_dirty = False
        return self._ink_cache

    # -------------------------------------------------------- input
    def apply(self, frame):
        self.frame = frame
        if frame.message:
            self.say(frame.message)
        if frame.command == "submit":
            self.submit()
        elif frame.command == "space":
            self.add_space()
        elif frame.command == "backspace":
            self.backspace()
        elif frame.command == "clear":
            self.clear_canvas()
            self.say("Canvas cleared")

        drawing = frame.pen_down and frame.cursor is not None and self.reading is None
        if drawing:
            if self.stroke is None:
                self.begin_stroke()
            self.add_point(frame.cursor)
        elif self.stroke is not None:
            self.end_stroke()

    def switch_mode(self):
        self.end_stroke()
        self.active.stop()
        if self.mode == "camera":
            try:
                cv2.destroyWindow(CAMERA_WINDOW)
            except cv2.error:
                pass
        self.mode = "glove" if self.mode == "camera" else "camera"
        self.active.start(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.say(f"{self.active.name} MODE - see the help box for how to use it")

    def handle_key(self, key):
        """Act on a key press. Returns False when AirPen should quit."""
        if key == 255:
            return True
        if getattr(self.active, "captures_keys", False):
            # The mode is taking typed input (such as a camera address).
            reply = self.active.handle_key(key)
            if reply:
                self.say(reply)
            return True
        if key == 27:  # Esc
            return False
        if key in (ord("v"), ord("V")):
            self.switch_mode()
        elif key == 13:  # Enter
            self.submit()
        elif key == ord(" "):
            self.add_space()
        elif key == 8:  # Backspace
            self.backspace()
        elif key in (ord("c"), ord("C")):
            self.clear_canvas()
            self.say("Canvas cleared")
        elif key in (ord("x"), ord("X")):
            self.text = ""
            self.say("Text cleared")
        elif key in (ord("u"), ord("U")):
            self.undo_stroke()
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        elif key in (ord("["), ord("]")):
            step = 1.25 if key == ord("]") else 1 / 1.25
            self.stroke_width = int(round(min(MAX_STROKE_WIDTH, max(MIN_STROKE_WIDTH, self.stroke_width * step))))
            self.say(f"Pen thickness {self.stroke_width}")
        elif key in (ord("f"), ord("F")):
            current = cv2.getWindowProperty(CANVAS_WINDOW, cv2.WND_PROP_FULLSCREEN)
            target = cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(CANVAS_WINDOW, cv2.WND_PROP_FULLSCREEN, target)
        else:
            reply = self.active.handle_key(key)
            if reply:
                self.say(reply)
        return True

    # ------------------------------------------------------- drawing
    def render(self):
        image = self.canvas.copy()
        height, width = image.shape[:2]
        frame = self.frame
        for y in (height // 4, height // 2, 3 * height // 4):
            cv2.line(image, (0, y), (width, y), DARK_GRAY, 1, cv2.LINE_AA)

        box, pixels = self.ink_metrics()
        if box:
            x, y, box_width, box_height = box
            cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (40, 110, 40), 1, cv2.LINE_AA)
        progress = min(1.0, pixels / MIN_INK_PIXELS)
        colour = GREEN if progress >= 1 else YELLOW
        cv2.rectangle(image, (20, height - 30), (width - 20, height - 16), GRAY, 1)
        cv2.rectangle(image, (20, height - 30), (20 + int((width - 40) * progress), height - 16), colour, -1)
        cv2.putText(image, "READY TO SUBMIT" if progress >= 1 else f"Ink: {int(progress * 100)}%",
                    (20, height - 37), FONT, 0.5, colour, 1, cv2.LINE_AA)

        if frame is not None and frame.status:
            cv2.putText(image, frame.status, (20, height - 62), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        cv2.putText(image, KEY_HELP, (20, height - 88), FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        self._draw_mode_badge(image)
        self._draw_text_line(image)
        if self.show_help:
            self._draw_help(image)
        if frame is not None and frame.gesture:
            self._draw_gesture_hold(image, frame.gesture, frame.gesture_progress)
        banner = frame.banner if frame is not None and frame.banner else None
        if banner:
            self._draw_banner(image, banner, (20, 60, 90), YELLOW)
        elif time.monotonic() < self.message_until:
            self._draw_banner(image, self.message, (18, 70, 18), GREEN)

        if frame is not None and frame.cursor is not None:
            pen = frame.pen_down and self.reading is None
            cv2.circle(image, frame.cursor, 9, RED if pen else WHITE, -1 if pen else 2, cv2.LINE_AA)
        return image

    def _draw_mode_badge(self, image):
        name = self.active.name
        other = "GLOVE" if name == "CAMERA" else "CAMERA"
        cv2.rectangle(image, (16, 14), (300, 58), MODE_COLOURS[name], -1)
        cv2.putText(image, f"{name} MODE", (26, 46), FONT, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"V: switch to {other}", (310, 44), FONT, 0.6, MODE_COLOURS[name], 1, cv2.LINE_AA)

    def _draw_text_line(self, image):
        shown = self.text[-40:] if self.text else ""
        # A bar, not an underscore, so a space before it stays visible: "AIR |".
        caret = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
        line = f"Text: {shown}{caret}" if shown else "Text: (write a word, then submit it)"
        cv2.putText(image, line, (20, 104), FONT, 1.1, WHITE if shown else GRAY, 2, cv2.LINE_AA)

    def _draw_help(self, image):
        lines = self.active.help_lines
        width = 420
        left = image.shape[1] - width - 16
        bottom = 20 + 30 * len(lines) + 12
        colour = MODE_COLOURS[self.active.name]
        cv2.rectangle(image, (left, 16), (left + width, bottom), PANEL, -1)
        cv2.rectangle(image, (left, 16), (left + width, bottom), colour, 1)
        for row, line in enumerate(lines):
            title = row == 0
            cv2.putText(image, line, (left + 14, 44 + 30 * row), FONT, 0.7 if title else 0.58,
                        colour if title else WHITE, 2 if title else 1, cv2.LINE_AA)
        cv2.putText(image, "H: hide help", (left + 14, bottom + 22), FONT, 0.5, GRAY, 1, cv2.LINE_AA)

    def _draw_gesture_hold(self, image, label, progress):
        left, top, bar_width = 20, 124, 420
        cv2.putText(image, f"Hold for: {label}", (left, top + 20), FONT, 0.75, YELLOW, 2, cv2.LINE_AA)
        cv2.rectangle(image, (left, top + 30), (left + bar_width, top + 46), GRAY, 1)
        cv2.rectangle(image, (left, top + 30), (left + int(bar_width * progress), top + 46), YELLOW, -1)

    def _draw_banner(self, image, text, background, colour):
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 1.05, 2)
        x = max(20, (image.shape[1] - text_width) // 2)
        top = image.shape[0] // 2 - 60  # Below the help box, which fills the top right.
        cv2.rectangle(image, (x - 18, top), (x + text_width + 18, top + text_height + 28), background, -1)
        cv2.putText(image, text, (x, top + text_height + 8), FONT, 1.05, colour, 2, cv2.LINE_AA)

    # ----------------------------------------------------------- run
    def run(self):
        cv2.namedWindow(CANVAS_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(CANVAS_WINDOW, 1120, 630)
        cv2.moveWindow(CANVAS_WINDOW, 400, 20)
        camera_window_open = False
        self.active.start(CANVAS_WIDTH, CANVAS_HEIGHT)
        print(f"AirPen started in {self.active.name} mode. Press V to switch, Esc to quit.")
        try:
            while True:
                frame = self.active.read(CANVAS_WIDTH, CANVAS_HEIGHT)
                self.apply(frame)
                self.finish_reading()
                if frame.preview is not None:
                    if not camera_window_open:
                        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(CAMERA_WINDOW, 360, 270)
                        cv2.moveWindow(CAMERA_WINDOW, 20, 20)
                        camera_window_open = True
                    cv2.imshow(CAMERA_WINDOW, frame.preview)
                elif self.mode != "camera":
                    camera_window_open = False
                cv2.imshow(CANVAS_WINDOW, self.render())
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.active.stop()
            self.executor.shutdown(wait=False, cancel_futures=True)
            cv2.destroyAllWindows()
            if self.text:
                print(f"Final text: {self.text}")


def main():
    first_mode = os.environ.get("INPUT_BACKEND", "camera").strip().lower()
    if first_mode not in ("camera", "glove"):
        raise ValueError("INPUT_BACKEND must be 'camera' or 'glove'.")
    AirPen(load_recognizer(), first_mode).run()


if __name__ == "__main__":
    main()
