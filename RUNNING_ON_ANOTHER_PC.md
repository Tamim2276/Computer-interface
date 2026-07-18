# Running Airwrite on Another PC

Use this guide after cloning the repository onto a different Windows computer.

## 1. Install prerequisites

Install these first:

- Python 3.10 or 3.11
- Git
- A working webcam, or a phone camera stream if you want to use the phone as the camera source

## 2. Clone the repository

```bash
git clone https://github.com/Tamim2276/Computer-interface.git
cd Computer-interface
```

If you already cloned it, just open the project folder.

## 3. Create a fresh virtual environment

Do not copy the existing `airwrite_env` folder to another PC. Create a new environment instead:

```bash
python -m venv .venv
.venv\Scripts\activate
```

If `python` does not work on your machine, use `py` instead:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run may also download model files used by EasyOCR.

## 5. Check the required project files

Make sure these files are still in the project root:

- `airwrite.py`
- `hand_landmarker.task`
- `requirements.txt`

If `hand_landmarker.task` is missing, the script will fail when it tries to load Mediapipe.

## 6. Set the camera source

Open `airwrite.py` and check this line near the top:

```python
CAMERA_SOURCE = "http://192.168.0.108:8080/video"
```

Use one of these values:

- `0` for the built-in laptop webcam
- `1` or another camera index if needed
- A phone stream URL like `http://192.168.x.x:8080/video` if you are using an IP camera app

## 7. Run the app

```bash
python airwrite.py
```

## 8. How to use it

The script uses hand gestures to write and recognize words:

- Raise the index finger to enter writing mode
- Pinch thumb and index finger to draw
- Release the pinch to move without drawing
- Show the peace sign to recognize the written word
- Show a palm to clear the canvas
- Press `C` to clear manually
- Press `Esc` to quit

## Troubleshooting

If the app does not start, check these common issues:

- `ModuleNotFoundError`: rerun `pip install -r requirements.txt` inside the virtual environment
- Camera not opening: change `CAMERA_SOURCE` to `0` for the laptop webcam, or verify the phone stream URL
- Mediapipe task error: confirm `hand_landmarker.task` is in the same folder as `airwrite.py`
- Slow first run: EasyOCR can take longer the first time because it may download model data

## Recommended sharing checklist

Before giving the project to someone else, make sure you share:

- The full repository folder
- The `requirements.txt` file
- The `hand_landmarker.task` file
- Any notes about the correct `CAMERA_SOURCE` value
