# AirPen Glove — Laptop Setup

This is the usable version of the code and steps extracted from
`glove_laptop_guide.pdf`. The PDF's code listing has broken character spacing;
use the source files in this repository instead of copying code from the PDF.

## Files

- `glove_firmware/glove_firmware.ino` — complete ESP32 firmware
- `glove_firmware/bno055_test/bno055_test.ino` — BNO055-only test
- `glove_firmware/flex_test/flex_test.ino` — flex-sensor-only test
- `glove_reader.py` — background USB serial reader
- `airwrite.py` — supports both `camera` and `glove` input backends

## Wiring

Disconnect USB power while changing wires.

| Component | Component pin | ESP32 connection |
|---|---|---|
| BNO055 | VIN | 3.3V |
| BNO055 | GND | GND |
| BNO055 | SDA | GPIO 21 |
| BNO055 | SCL | GPIO 22 |
| Index flex sensor | one end | 3.3V |
| Index flex sensor | other end | GPIO 34 and one end of 10 kΩ resistor |
| Index resistor | other end | GND |
| Middle flex sensor | one end | 3.3V |
| Middle flex sensor | other end | GPIO 35 and one end of 10 kΩ resistor |
| Middle resistor | other end | GND |
| ESP32 | USB | Laptop USB port |

Use the voltage printed on the actual BNO055 breakout board. The guide specifies
3.3V, which is the safe choice for this wiring.

## Arduino IDE

1. Install Arduino IDE.
2. Add this Boards Manager URL:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Install the `esp32` board package.
4. Select **ESP32 Dev Module**.
5. Install **Adafruit BNO055** and **Adafruit Unified Sensor** in Library Manager.
6. Connect the ESP32 and select its COM port.

If no COM port appears, install the CP210x or CH340 driver used by your ESP32
board and reconnect it.

## Test and calibrate in this order

1. Upload `bno055_test.ino`. At 115200 baud, confirm `BNO055 OK` and changing
   heading/pitch/roll values.
2. Upload `flex_test.ino`. Record each sensor value with the finger straight and
   bent comfortably.
3. In `glove_firmware.ino`, set each threshold halfway between its straight and
   bent readings.
4. If the reading becomes higher when bent, set `FLEX_BENT_IS_HIGH = true`.
   If it becomes lower when bent, leave it `false`.
5. Upload `glove_firmware.ino` and confirm seven CSV values appear at 115200
   baud, for example:

   ```text
   243.18,11.90,-2.35,1874,1902,0,0
   ```

6. Close Arduino Serial Monitor before starting Python. Only one program can
   open a COM port at a time.

## Laptop installation

Use Python 3.10 or 3.11:

```bash
python -m venv airwrite_env
source airwrite_env/Scripts/activate
python -m pip install -r requirements.txt
```

List ports on Windows:

```bash
python -m serial.tools.list_ports
```

Test the reader after closing Arduino Serial Monitor:

```bash
python glove_reader.py COM3
```

Press `Ctrl+C` after confirming that the coordinates and flex readings change.

## Run in glove mode

In Git Bash, replace `COM3` with the detected port:

```bash
source airwrite_env/Scripts/activate
export INPUT_BACKEND=glove
export GLOVE_SERIAL_PORT=COM3
export OCR_BACKEND=local
python airwrite.py
```

Controls:

- Move the glove: move the cursor
- Bend index finger: draw
- Straighten index finger: reposition without drawing
- Bend and hold middle finger: recognize current word
- `C`: clear canvas
- `U`: undo last stroke
- `R`: recenter the glove cursor
- `M`: switch word/sentence mode on an empty canvas
- `Space`: add a space to accumulated text
- `Esc`: quit

## Tuning

Set these before launching AirPen:

```bash
export GLOVE_SCALE=16.0
export GLOVE_DEAD_ZONE=0.12
export GLOVE_SMOOTHING=0.68
```

- Letters too small or movement needs too much wrist travel: raise scale to 20 or 24.
- Cursor reaches an edge too fast: lower scale to 6 or 8.
- Cursor drifts: raise dead zone gradually to 0.18, 0.25, or 0.35.
- Small movements are ignored: lower dead zone to 0.08.
- Cursor is still noisy: lower smoothing to 0.55.
- Cursor feels delayed: raise smoothing to 0.78.
- Mirrored movement: change the sign of `heading_change` in `glove_reader.py`.
- Upside-down movement: change the sign of `pitch_change` in `glove_reader.py`.
