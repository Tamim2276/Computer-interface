# AirPen Glove — Laptop Setup

This is the usable version of the code and steps extracted from
`glove_laptop_guide.pdf`. The PDF's code listing has broken character spacing;
use the source files in this repository instead of copying code from the PDF.

## Important: which firmware is active

The normal AirPen glove uses the MicroPython firmware in
`glove_firmware/micropython/main.py`. It is copied to the ESP32 as
`main.py`, starts automatically when the board receives power, and sends ten
CSV values at 115200 baud. The Arduino sketches in this repository are
diagnostic/legacy sketches; do not upload `glove_firmware.ino` for the normal
AirPen presentation unless you intentionally want the Arduino data format.

If the ESP32 already runs the glove, the friend’s laptop does not need
Thonny. The upload steps are included below for recovering or replacing the
firmware.

## Files

- `glove_firmware/glove_firmware.ino` — complete ESP32 firmware
- `glove_firmware/bno055_test/bno055_test.ino` — BNO055-only test
- `glove_firmware/flex_test/flex_test.ino` — flex-sensor-only test
- `glove_reader.py` — background USB serial reader
- `airwrite.py` — supports both `camera` and `glove` input backends

## Wiring

Disconnect USB power while changing wires.

| Component                                                      | Component pin | ESP32 connection                      |
| -------------------------------------------------------------- | ------------- | ------------------------------------- |
| BNO055                                                         | VIN           | 3.3V                                  |
| BNO055                                                         | GND           | GND                                   |
| BNO055                                                         | SDA           | GPIO 19                               |
| BNO055                                                         | SCL           | GPIO 21                               |
| Active pen flex sensor (currently sewn into the middle finger) | one end       | 3.3V                                  |
| Active pen flex sensor                                         | other end     | GPIO 35 and one end of 10 kΩ resistor |
| Active pen resistor                                            | other end     | GND                                   |
| Unused second flex sensor                                      | one end       | 3.3V, only if physically installed    |
| Unused second flex sensor                                      | other end     | GPIO 34 and one end of 10 kΩ resistor |
| Unused second resistor                                         | other end     | GND                                   |
| ESP32                                                          | USB           | Laptop USB port                       |

Use the voltage printed on the actual BNO055 breakout board. The guide specifies
3.3V, which is the safe choice for this wiring.

## Recovering the MicroPython firmware (only if needed)

Do this before the laptop presentation only if the board does not already
send data. Install Thonny from <https://thonny.org/> and use a data-capable
USB cable. In Thonny:

1. Open **Tools > Options > Interpreter**.
2. Select **MicroPython (ESP32)** and the ESP32’s COM port.
3. Open `glove_firmware/micropython/main.py` from this project.
4. Choose **File > Save as**, select **MicroPython device**, and save it as
   exactly `main.py` in the device root.
5. Press the Thonny stop/restart button, wait for `# BNO055 ready`, and check
   that ten comma-separated values appear in the Shell about 50 times a
   second.
6. Close Thonny completely before starting AirPen. Thonny and AirPen cannot
   open the same COM port at the same time.

## Arduino IDE diagnostics (optional)

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
   This project's sensor was detected at I2C address `0x29`; the complete
   Arduino firmware is configured to use that address.
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

Or double-click `run_glove.bat`, which sets these and starts AirPen.

Controls:

- Rotate the finger or hand: move the cursor (like an air mouse)
- Curl the flex-sensor finger: draw
- Straighten it: reposition without drawing
- `Enter`: recognize the current word
- `+` / `-`: more / less sensitive: how far the cursor moves for a hand turn
  (shown as "sensitivity" at the bottom of the canvas and remembered in
  `glove_sensitivity.txt`)
- `K`: calibrate the movement directions (after putting the glove on, or
  whenever moving in one direction comes out diagonal)
- `C`: clear canvas
- `U`: undo last stroke
- `R`: recenter the glove cursor
- `M`: switch word/sentence mode on an empty canvas
- `Space`: add a space to accumulated text
- `Esc`: quit

## Tuning

Set these before launching AirPen:

```bash
export GLOVE_SCALE=30.0
export GLOVE_DEAD_ZONE=0.0
export GLOVE_SMOOTHING=0.55
```

- Letters too small or movement needs too much wrist travel: raise scale to 36 or 40.
- Cursor reaches an edge too fast: lower scale to 20 or 24.
- Cursor drifts while the hand is still: raise dead zone slightly, to 0.02 or 0.04.
  Larger values discard the slow movements that form letters.
- Pen goes down too easily or not at all: the PC decides the pen from the
  raw finger reading. It goes down below `GLOVE_PEN_DOWN_BELOW` (45) and lifts
  after rising `GLOVE_PEN_RELEASE_RISE` (25) above the curl, or above
  `GLOVE_PEN_UP_ABOVE` (70). After fitting a new flex sensor, measure its
  straight and curled readings with `glove_firmware/micropython/flex_pins_test.py`
  and set `GLOVE_PEN_DOWN_BELOW` between them.

## Debugging

`run_glove.bat` records every glove sample to `glove_logs/` (`GLOVE_LOG=1`).
After quitting AirPen, run:

```bash
airwrite_env/Scripts/python.exe glove_debug.py
```

It prints a summary of the newest recording (pen blips, stuck pen, finger
readings, hand speed) and saves a picture next to it: the drawing, plus a
timeline of the finger sensor, pen state and hand speed. Gaps in the timeline
mean the motion sensor dropped out; check its four wires.

- Cursor is still noisy: lower smoothing to 0.55.
- Cursor feels delayed: raise smoothing to 0.78.
- Mirrored movement: `export GLOVE_HEADING_DIRECTION=-1`.
- Upside-down movement: `export GLOVE_PITCH_DIRECTION=-1`.
