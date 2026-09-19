# AirPen

AirPen is a real-time air-writing application. You write words in the air, one at a time; TrOCR reads each one and adds it to a line of text. There are two ways to write, and `V` switches between them while AirPen runs:

- **Camera mode** tracks your hand with MediaPipe through a phone (IP camera) or webcam. Pinch thumb and index together to draw.
- **Glove mode** uses the ESP32 glove: turning your hand moves the pen and curling the flex-sensor finger draws. See `GLOVE_SETUP.md`.

## Features

- One 1600x900 canvas and one line of text, shared by both modes. A corner badge shows the current mode and a help box shows how to use it.
- Word-by-word writing: submit a word, and AirPen reads it, adds it to the text and clears the canvas. Space and backspace edit the text.
- Smooth ink: strokes are drawn as curves through the samples, the camera cursor is steadied with a 1-Euro filter, pen blips are erased and the hook left when lifting the pen is trimmed.
- Recognition runs in the background, so tracking and drawing never pause.
- Camera and glove keep their own settings and calibration in their own files.

## Setup

1. Keep `hand_landmarker.task` beside `airwrite.py`.
2. Set the camera: the phone's IP address lives in `camera_address.txt` (just the IP, such as `192.168.0.101`; a full URL or `0` for a webcam also works). When the address changes with the Wi-Fi network, press `I` in camera mode, type the new IP and press Enter: AirPen reconnects and saves it.
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

On Windows, double-click `run_camera.bat` or `run_glove.bat` to start in that mode.

On PowerShell, use:

```powershell
.\airwrite_env\Scripts\Activate.ps1
python airwrite.py
```

## Raspberry Pi with Hugging Face OCR

For the Pi, install the lightweight cloud-OCR runtime instead of the local TrOCR stack:

```bash
python3 -m venv airwrite_env
source airwrite_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt
```

Deploy the contents of `hf_endpoint/` as a Hugging Face Inference Endpoint custom container, with your TrOCR model repository selected as the endpoint model. Then configure the Pi without placing the token in source code:

```bash
export OCR_BACKEND=huggingface
export HF_ENDPOINT_URL="https://YOUR_ENDPOINT_URL"
export HF_TOKEN="hf_your_token"
python airwrite.py
```

The application sends the cropped handwriting image to `POST /ocr`; the endpoint returns `{"text": "AIR PEN"}`. Local development remains available with `OCR_BACKEND=local`.

## Controls

Both modes:

| Key | Action |
| --- | --- |
| `V` | Switch between camera and glove mode |
| `Enter` | Submit: read the word on the canvas and add it to the text |
| `Space` | Add a space after the last word (submitting a word still on the canvas first) |
| `Backspace` | Remove the last stroke while writing; otherwise the last space or word |
| `[` / `]` | Thinner / thicker pen (for new strokes) |
| `C` / `X` | Clear the canvas / clear the text |
| `U` | Undo the last stroke |
| `H` | Hide or show the help box |
| `F` | Toggle fullscreen |
| `Esc` | Quit |

Camera mode (hold each hand shape for half a second; a bar shows the progress):

| Gesture | Action |
| --- | --- |
| Pinch thumb and index | Draw (open them to move without drawing) |
| Peace sign | Submit word |
| Thumbs up | Space (a word still on the canvas is submitted first, then the space follows) |
| Three fingers up | Backspace |
| Open palm | Clear the canvas |
| `+` / `-` | Reach: how far the pen moves for a hand movement (0.5-3) |

Glove mode: curl the flex finger to draw, and use the keys above for text. `K` calibrates the movement directions, `R` centres the cursor, and `+` / `-` change the sensitivity (remembered in `glove_sensitivity.txt`).

The first launch downloads Microsoft's `trocr-base-handwritten` model (about 1.33 GB) and then caches it locally.

## Project files

| File | Purpose |
| --- | --- |
| `airwrite.py` | Shared app: canvas, strokes, recognition, text, help, mode switching |
| `camera_input.py` | Camera mode: hand tracking, pinch, gestures, smoothing and its settings |
| `glove_input.py` | Glove mode: pen, keys and the `K` direction calibration |
| `glove_reader.py` | Reads the glove over USB and turns its sensors into cursor and pen |
| `input_common.py` | What both modes report to the app, and the smoothing filter |
| `glove_debug.py` | Turns a glove recording into a debugging report |
| `glove_firmware/micropython/` | ESP32 glove firmware (MicroPython, uploaded with Thonny) |
| `hand_landmarker.task` | MediaPipe hand-landmarker model |
| `requirements.txt` | Python dependencies |
