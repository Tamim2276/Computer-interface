# AirPen: Natural Hand-Tracking Word Recognition

AirPen is a real-time computer vision application that turns your hand into a digital pen. Using your webcam, the software tracks your finger movements to draw on the air, allowing you to write words and have them automatically recognized using optical character recognition (OCR).

## How It Works

The project combines advanced computer vision, gesture-based state management, and machine learning to create a seamless drawing experience.

### 1. The Engine (MediaPipe)
We use the **MediaPipe Hand Landmarker** to track 21 distinct points on your hand. By monitoring the distance between the tip of your thumb and your index finger, the system determines exactly when your "pen" is touching the virtual canvas.

### 2. Gesture Controls (Natural Interaction)
We replaced clunky menu buttons with intuitive, natural hand gestures:
* **Pinch (Thumb + Index):** Pen Down — draw as you move your index finger.
* **Release:** Pen Up — hover and move your hand to start a new letter without drawing.
* **Peace Sign (✌️):** Triggers the recognition engine to read what you have written.
* **Fist (✊):** Clears the canvas entirely.

### 3. Smart Processing (Pre-processing & OCR)
To ensure high accuracy, the project uses a custom image-processing pipeline before text recognition:
* **Normalization:** The application crops only the area where you have written, removing excess black space.
* **Enhancement:** It inverts the colors (creating clean black text on a white background) and resizes the image while maintaining the original aspect ratio.
* **Recognition:** The cleaned image is processed by **EasyOCR**, which converts your air-written strokes into digital text.

## Features
* **Zero-Setup Interaction:** Start writing the moment the app launches.
* **Smart Pen-Lift:** No accidental "tails" or streaks; the pen lifts the instant you stop pinching.
* **Real-time HUD:** A clean dashboard shows your current mode, recognized word history, and clear instructions.
* **Dynamic Preview:** See exactly what the AI sees in a dedicated preview window during recognition.

## Tech Stack
* **Python**
* **OpenCV:** For real-time video streaming and image manipulation.
* **MediaPipe:** For high-speed, accurate hand-joint tracking.
* **EasyOCR:** For state-of-the-art text recognition.
* **NumPy:** For canvas and matrix calculations.

---
*Created as a lightweight, efficient computer vision project for real-time gesture-based interaction.*