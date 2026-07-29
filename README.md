# AirPen

AirPen is a real-time air-writing application. It tracks one hand with MediaPipe, turns an index-finger pose into strokes, and reads completed writing with TrOCR.

## Features

- Two automatic windows: a small hand-tracking camera view at top-left and a wide 1600x900 virtual writing canvas on the right. The high-resolution canvas stays accurate when fullscreen and has room for a sentence.
- Comfortable 1.35x fingertip motion (`AMPLIFY` in `airwrite.py`), adaptive cursor smoothing, anti-aliased strokes, and midpoint interpolation for smoother curves.
- Short pen-state stability check to prevent flickering without delaying pinch-to-move.
- Canvas-only writing guides, an amplified cursor dot, a live bounding-box preview, a fill bar that turns green when enough ink is present for OCR, and a prominent recognition-result banner.
- Background TrOCR recognition, so the camera and UI continue updating while the model reads the canvas.
- Two writing modes: sentence recognition for a full line, or deliberate peace-sign recognition for one word.
- Index-only writing, pinch-to-reposition, peace-sign recognition, thumbs-up spacing, palm clear, and keyboard undo.

## Setup

1. Keep `hand_landmarker.task` beside `airwrite.py`.
2. In `airwrite.py`, set `CAMERA_SOURCE` to `0` for the laptop webcam, or set it to the URL of an IP camera stream.
3. Install the dependencies into a virtual environment.

```bash
python -m venv airwrite_env
```

In Git Bash, activate it with (there is no leading `p`):

```bash
source airwrite_env/Scripts/activate
python -m pip install -r requirements.txt
python airwrite.py
```

On PowerShell, use:

```powershell
.\airwrite_env\Scripts\Activate.ps1
python airwrite.py
```

## Controls

| Input | Action |
| --- | --- |
| Index finger up, held about 0.6 seconds | Start writing mode |
| Index finger only | Draw a stroke |
| Pinch thumb and index | Lift the pen and reposition without drawing |
| Peace sign, held briefly | Read the canvas with TrOCR |
| Thumbs up, held briefly | In Word Mode, add a space after a recognized word |
| Closed fist, held briefly on a blank canvas | Toggle Word Mode / Sentence Mode |
| Open palm, held briefly | Clear the canvas |
| `U` | Undo the last stroke |
| `C` | Clear canvas and recognized text |
| `Space` | Add a space after recognized text |
| `F` | Toggle fullscreen canvas |
| `Esc` | Quit |

In **Word Mode**, write a word of any length, pinch to lift/reposition, then use peace to read it. AirPen clears the canvas and appends the result to the transcript. After the word is recognized, a thumbs-up adds its space. In **Sentence Mode**, write a complete line with physical gaps between words, then use peace once to read it. Use peace with an empty canvas to show the accumulated text. The first launch downloads Microsoft’s `trocr-base-handwritten` model (about 1.33 GB) and then caches it locally; later launches use the cached model.

## Project files

| File | Purpose |
| --- | --- |
| `airwrite.py` | Live camera, gesture, canvas, and TrOCR application |
| `hand_landmarker.task` | MediaPipe hand-landmarker model |
| `train_cnn.py` | Optional EMNIST CNN training script |
| `requirements.txt` | Python dependencies |
