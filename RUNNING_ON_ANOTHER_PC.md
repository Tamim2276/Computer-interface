# AirPen on another laptop: step-by-step setup and presentation guide

This guide takes you from an empty Windows laptop to a working AirPen demo.
Follow the steps in order and do not skip any. Every command is written out
in full: type it exactly as shown and press **Enter** after each one.

> **Do the setup the day before the presentation.** It downloads about 1 GB
> of software and takes 30-60 minutes. On the day itself, only Part 4 is left.

### Contents

- Part 0 - What the two modes do (for your presentation)
- Part 1 - On YOUR computer, before you go
- Part 2 - On your FRIEND'S laptop: one-time setup
- Part 3 - Test everything
- Part 4 - On the presentation day
- Part 5 - Controls cheat sheet
- Part 6 - If something goes wrong

---

## Part 0 - What the two modes do (for your presentation)

AirPen lets you write words in the air. Each word is read by a handwriting
recognition model (Microsoft TrOCR) and added to a line of text at the top of
the screen. There are two ways to write, and the **V** key switches between
them while AirPen runs.

### Camera mode

- **Hand tracking:** a phone camera (or the laptop's webcam) films your hand,
  and MediaPipe finds 21 points on it, 30 times a second.
- **Pinch to write:** touch your thumb and index finger together to draw;
  open them to move without drawing, like lifting a pen.
- **Hand-shape commands:** hold a shape for half a second (a yellow bar fills up):
  peace sign = submit the word, thumbs up = space, three fingers = backspace,
  open palm = clear the canvas.
- **Smooth writing:** jitter is filtered out, strokes are drawn as curves, and
  a one-frame tracking glitch cannot break a stroke.
- **Adjustable:** `+` / `-` change the reach (how far the pen moves for a hand
  movement), `[` / `]` change the pen thickness.

### Glove mode

- **Motion glove:** an ESP32 microcontroller with a BNO055 motion sensor and a
  flex (bend) sensor, connected to the laptop by USB.
- **Air-mouse movement:** turning your hand moves the cursor, using the
  sensor's gyroscope, like a presentation air mouse.
- **Bend to write:** curling the finger with the flex sensor draws;
  straightening it lifts the pen.
- **Calibration:** `K` teaches AirPen which way is right and down for how the
  glove sits on your hand today (takes 8 seconds).
- **Adjustable and robust:** `+` / `-` change the sensitivity (remembered for
  next time), accidental twitches are erased automatically, and the glove
  reconnects by itself if its cable is pulled out.

### Shared by both modes

- One canvas and one line of text, word by word: write, submit, and the word is
  added to the text.
- Space, backspace and clear work in both modes (gestures or keyboard).
- A badge in the top-left corner shows the current mode, and a help box on
  the right lists every control for that mode.

---

## Part 1 - On YOUR computer, before you go

### Step 1.1 - Upload the latest code to GitHub

The friend's laptop downloads the project from GitHub, so GitHub must have
your newest version.

1. Press the **Windows key**, type `cmd`, and press **Enter**. A black
   window called Command Prompt opens.
2. Type these commands one at a time:

   ```bat
   cd %USERPROFILE%\Desktop\ci
   git add -A
   git commit -m "Ready for presentation"
   git push
   ```

3. The last command should end with a line like `main -> main`. If
   `git commit` says `nothing to commit`, everything was already uploaded; that
   is fine.

### Step 1.2 - Copy three things to a USB stick

These save a lot of time and make you independent of the venue's internet.

1. **The handwriting model (about 1.3 GB).** On the computer where AirPen
   already works, find where it is stored by typing this in the project folder:

   ```bat
   airwrite_env\Scripts\python.exe -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)"
   ```

   Open that folder (Windows key + R, paste the path, Enter) and copy the
   folder **`models--microsoft--trocr-base-handwritten`** to the USB stick.
   You can skip this and let the friend's laptop download it in Step 2.6,
   but that needs a good internet connection on the day.
2. **The glove's USB driver.** Press **Windows key + R**, type
   `%USERPROFILE%\Desktop` and press **Enter**. Copy the folder
   **`cp210x_driver`** to the USB stick.
3. **A backup of the project** (in case GitHub or the internet fails). In the
   same Desktop window, copy the whole **`ci`** folder to the USB stick, then on
   the USB stick **delete the `airwrite_env` folder inside it** (it is large
   and does not work on another computer anyway).

### Step 1.3 - Pack the hardware

- [ ] The glove (ESP32 board, motion sensor and flex sensor, all wires fixed with tape)
- [ ] **The USB cable that transfers data** (the one that worked on your PC;
      charge-only cables light the board up but the laptop cannot see it)
- [ ] Tape, to hold the cable to your wrist
- [ ] Your phone with the IP camera app, and its charger
- [ ] The USB stick from Step 1.2
- [ ] The laptop's charger (handwriting recognition works the processor hard)

The glove's program (`main.py`) is already saved on the ESP32 and starts by
itself whenever the board gets power. **You do not need Thonny or Arduino on
the friend's laptop.**

---

## Part 2 - On your FRIEND'S laptop: one-time setup

### Step 2.1 - Install Python 3.11

AirPen is tested with Python **3.11.9**. Newer versions (3.12, 3.13) will not
work with these packages, so install exactly this one, even if the laptop
already has another Python.

1. Open this page in a browser: <https://www.python.org/downloads/release/python-3119/>
2. Scroll to **Files** and click **Windows installer (64-bit)**.
3. Run the downloaded file `python-3.11.9-amd64.exe`.
4. On the first screen, **tick the box "Add python.exe to PATH"** at the bottom.
5. Click **Install Now**. Wait until it says "Setup was successful", then click **Close**.
6. Check it worked. Open Command Prompt (Windows key, type `cmd`, Enter) and type:

   ```bat
   py -3.11 --version
   ```

   It must print **`Python 3.11.9`**.

### Step 2.2 - Install Git

(Skip this step if you will copy the project from the USB stick instead.)

1. Open <https://git-scm.com/download/win> and download **64-bit Git for Windows Setup**.
2. Run it and click **Next** on every screen, then **Install**, then **Finish**.
3. **Close Command Prompt and open a new one** (so it notices Git), then type:

   ```bat
   git --version
   ```

   It should print something like `git version 2.x`.

### Step 2.3 - Get the project

**Option A - from GitHub (needs Git):** in Command Prompt type:

```bat
cd %USERPROFILE%\Desktop
git clone https://github.com/Tamim2276/Computer-interface.git
cd Computer-interface
```

**Option B - from the USB stick:** copy the `ci` folder from the USB stick to
the laptop's Desktop. Then in Command Prompt type:

```bat
cd %USERPROFILE%\Desktop\ci
```

If you cloned from GitHub, the folder is normally named `Computer-interface`,
so use this command instead:

```bat
cd %USERPROFILE%\Desktop\Computer-interface
```

**Check:** type `dir` and press Enter. The list must include `airwrite.py`,
`run_camera.bat`, `run_glove.bat`, `requirements.txt` and
`hand_landmarker.task`. If it does not, you are in the wrong folder.

> **From now on, all commands are typed in this folder.** If you close Command
> Prompt, open it again and repeat the `cd` line from this step first.
> (Shortcut: open the project folder in File Explorer, click the address bar,
> type `cmd` and press Enter; Command Prompt opens already in that folder.)

### Step 2.4 - Create AirPen's own Python environment

This makes a private folder of packages just for AirPen. The name
**`airwrite_env`** must be exactly this, because the start-up files look for it.

```bat
py -3.11 -m venv airwrite_env
```

It finishes silently after a few seconds. A new folder `airwrite_env` appears
in the project folder.

### Step 2.5 - Install the packages

First update the installer, then install everything AirPen needs:

```bat
airwrite_env\Scripts\python.exe -m pip install --upgrade pip
airwrite_env\Scripts\python.exe -m pip install -r requirements.txt
```

The second command downloads about 1 GB and takes **5-20 minutes**. Lots of
text scrolls past; that is normal. It is finished when you see a line starting
with **`Successfully installed`** and the prompt comes back.

If it stops with a red error about the network or a timeout, just type the
same command again; it continues where it stopped.

**Check** that everything is there:

```bat
airwrite_env\Scripts\python.exe -c "import cv2, mediapipe, numpy, PIL, torch, transformers, serial, requests; print('All packages OK')"
```

It must print **`All packages OK`**.

Check the required MediaPipe model file:

```bat
if exist hand_landmarker.task (echo Hand model found) else (echo ERROR: hand_landmarker.task is missing)
```

Do not continue if it is missing. Copy `hand_landmarker.task` from the project
backup or download the exact file supplied with this project into the folder
beside `airwrite.py`.

### Step 2.6 - Put the handwriting model in place

AirPen needs the TrOCR model (1.24 GB). Choose one option.

**Option A - copy it from the USB stick (fast, no internet needed):**

1. In Command Prompt, create the folder where the model belongs, and open it:

   ```bat
   mkdir %USERPROFILE%\.cache\huggingface\hub
   explorer %USERPROFILE%\.cache\huggingface\hub
   ```

   (If `mkdir` says the folder already exists, that is fine.)

2. A File Explorer window opens. Copy the folder
   **`models--microsoft--trocr-base-handwritten`** from the USB stick into it.
   The result must be:
   `C:\Users\<name>\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten`

**Option B - download it (needs good internet, about 10 minutes):** skip
straight to the check below; it downloads the model if it is missing.

**Check** (this also does the download for Option B):

```bat
airwrite_env\Scripts\python.exe -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; m='microsoft/trocr-base-handwritten'; TrOCRProcessor.from_pretrained(m); VisionEncoderDecoderModel.from_pretrained(m); print('Model ready')"
```

It must end with **`Model ready`**. A warning about "newly initialized"
weights may appear before it; that is normal and harmless.

### Step 2.7 - Install the glove's USB driver

Windows needs a driver to talk to the ESP32's USB chip. Many laptops do not
have it.

1. Plug the glove's ESP32 into the laptop with **the data cable**. Its red
   light should come on.
2. Press **Windows key + X** and click **Device Manager**.
3. Look for a section called **Ports (COM & LPT)**:
   - If it shows **`Silicon Labs CP210x USB to UART Bridge (COMx)`**, the
     driver is already installed. Go to Step 2.8.
   - If instead **Other devices** shows **`CP2102 USB to UART Bridge Controller`**
     with a yellow triangle, install the driver:
     1. Right-click it and click **Update driver**.
     2. Click **Browse my computer for drivers**.
     3. Click **Browse...** and choose the **`cp210x_driver`** folder on the USB stick.
     4. Make sure **Include subfolders** is ticked, click **Next**, and allow
        the installation if Windows asks.
     5. It now appears under **Ports (COM & LPT)**.
   - If nothing new appears at all, the cable is charge-only; try another cable.
   - (No USB stick? Search the web for **"Silicon Labs CP210x Universal
     Windows Driver"**, download the ZIP, extract it, and point step 3 at that folder.)

You do **not** need to note the COM number: AirPen finds the glove's port by
itself.

The project's normal glove firmware is already stored on the ESP32 as
MicroPython `main.py`; Arduino and Thonny are not required on presentation
day. If the board does not send data, follow the recovery instructions in
`GLOVE_SETUP.md` before testing AirPen.

### Step 2.8 - Connect the phone camera

The laptop and the phone must be on **the same Wi-Fi network**.

1. On the phone, open the IP camera app (for example **IP Webcam**) and start
   the video server.
2. The app shows an address such as `http://192.168.0.101:8080`. The IP is the
   middle part: **`192.168.0.101`**.
3. You will type that IP into AirPen in Part 3 (press `I`). AirPen remembers
   it in the file `camera_address.txt`.

**Quick test:** open a browser on the laptop and go to
`http://<the IP>:8080` (for example `http://192.168.0.101:8080`). If the camera
app's web page appears, the laptop can reach the phone.

> **Venue Wi-Fi often blocks devices from talking to each other.** The
> reliable fix: turn on the **phone's hotspot** and connect the laptop to it.
> Then read the phone's new IP in the camera app. (The model must already be
> installed, which Step 2.6 made sure of.)
>
> **No phone at all?** The laptop's own webcam works too: in AirPen press `I`,
> type `0` and press Enter.

---

## Part 3 - Test everything

Do this the day before, on the friend's laptop.

### Step 3.1 - Start AirPen in camera mode

1. Open the project folder in File Explorer and **double-click
   `run_camera.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then
   **Run anyway**.
3. A black window shows progress. After 10-30 seconds (loading the handwriting
   model) the **AirPen Canvas** window opens. If you cannot see it, click the
   Python icon on the taskbar; it may be behind other windows.
4. If a **Windows Firewall** window pops up, click **Allow**.
5. Click on the canvas window once, so it receives your key presses.

To test from Command Prompt instead of double-clicking, use:

```bat
cd %USERPROFILE%\Desktop\ci
run_camera.bat
```

For a GitHub clone, replace `ci` with `Computer-interface`. A successful
start prints `AirPen started in CAMERA mode` in the black window.

### Step 3.2 - Set the camera address

1. Press **`I`**.
2. Type the phone's IP (for example `192.168.0.101`). **Backspace** corrects mistakes.
3. Press **Enter**. A small **AirPen Camera** window appears showing the video.
   Hold your hand up: green lines appear on it.

The bottom line of the canvas shows `camera 25-30 fps | pinch ...` when it
works, or `Cannot reach camera ...` when it does not (see Part 6).

### Step 3.3 - Test camera writing

1. Hold your hand inside the green box in the camera window.
2. Pinch thumb and index together and write a short word, for example **HI**.
   Open your fingers between strokes.
3. Hold a **peace sign** until the yellow bar is full: the word appears at the top.
4. Hold a **thumbs up**: a space is added (`HI |`).
5. Write a second word and submit it.
6. Try **three fingers** (backspace) and **open palm** (clear the canvas).
7. Try `[` and `]` for pen thickness, and `+` / `-` for reach.

### Step 3.4 - Test the glove

1. Plug the glove in (data cable), and tape the cable to your wrist.
2. On the canvas window press **`V`**. The badge changes to **GLOVE MODE**.
3. Within a few seconds the bottom line shows readings like
   `flex 250 | pen up | sensitivity 30 ...`. That means the glove is connected.
4. Put the glove on, hold your hand in your writing position, and press
   **`K`**. Follow the four yellow instructions (still, turn right, back to the
   middle, turn down). It ends with **"Directions calibrated"**.
5. Press **`R`** to centre the cursor. Turn your hand: the dot follows.
6. Curl the flex finger to draw, straighten it to stop. Press **Enter** to
   submit the word.
7. If the cursor is too fast, press **`-`** a few times (it is remembered).
8. Press **`V`** again to go back to camera mode, and **Esc** to quit.

If all of this works, the laptop is ready.

You can also start directly in glove mode by double-clicking `run_glove.bat`,
or by running this from the project folder:

```bat
cd %USERPROFILE%\Desktop\ci
run_glove.bat
```

This still uses the same local TrOCR model. The phone camera is not required
when starting directly in glove mode, but the glove must be connected.

---

## Part 4 - On the presentation day

1. Plug in the laptop's charger.
2. Connect the laptop and the phone to the same network (the phone's hotspot is safest).
3. Start the phone's camera app.
4. Plug in the glove (if you will show it) and tape the cable to your wrist.
5. Double-click **`run_camera.bat`** (or `run_glove.bat` to start with the glove).
6. If the phone's IP changed: press **`I`**, type the new IP, press **Enter**.
7. Glove: put it on, press **`V`**, then **`K`** to calibrate, then **`R`**.
8. Press **`F`** for full screen. Press **`H`** if you want to hide the help box.
9. Press **`X`** to clear the practice text before you start.

---

## Part 5 - Controls cheat sheet

### Keys (both modes)

| Key         | Action                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| `V`         | Switch between camera and glove mode                                   |
| `Enter`     | Submit: read the word and add it to the text                           |
| `Space`     | Add a space (submits a word still on the canvas first)                 |
| `Backspace` | Remove the last stroke while writing; otherwise the last word or space |
| `[` / `]`   | Thinner / thicker pen                                                  |
| `C`         | Clear the canvas                                                       |
| `X`         | Clear all submitted text                                               |
| `U`         | Undo the last stroke                                                   |
| `H`         | Hide / show the help box                                               |
| `F`         | Full screen on / off                                                   |
| `Esc`       | Quit                                                                   |

### Camera mode controls

| Gesture or key       | Action                          |
| -------------------- | ------------------------------- |
| Pinch thumb + index  | Draw                            |
| Open fingers         | Move without drawing            |
| Peace sign (hold)    | Submit word                     |
| Thumbs up (hold)     | Space                           |
| Three fingers (hold) | Backspace                       |
| Open palm (hold)     | Clear canvas                    |
| `+` / `-`            | Reach                           |
| `I`                  | Type the phone's camera address |

### Glove mode controls

| Action or key        | Result                   |
| -------------------- | ------------------------ |
| Turn your hand       | Move the cursor          |
| Curl the flex finger | Draw                     |
| Straighten it        | Stop drawing             |
| `K`                  | Calibrate directions     |
| `R`                  | Centre the cursor        |
| `+` / `-`            | Sensitivity (remembered) |

---

## Part 6 - If something goes wrong

| Problem                                                          | What to do                                                                                                                                                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `'py' is not recognized`                                         | Python is not installed properly. Run the installer again (Step 2.1), choose **Modify** or **Repair**, and make sure **py launcher** is ticked. Open a **new** Command Prompt afterwards.               |
| `py -3.11 --version` shows another version or an error           | Python 3.11 is missing. Install it (Step 2.1); other Python versions can stay.                                                                                                                          |
| `pip install` stops with a network or timeout error              | Type the same command again; it continues. Use a better connection if it keeps failing.                                                                                                                 |
| `No module named ...` when starting                              | The packages are not installed in `airwrite_env`. Repeat Step 2.5 in the project folder.                                                                                                                |
| Double-clicking the `.bat` flashes and closes, or shows an error | Open Command Prompt in the project folder (Step 2.3) and type `run_camera.bat`; the error message stays visible.                                                                                        |
| "Windows protected your PC"                                      | Click **More info**, then **Run anyway**.                                                                                                                                                               |
| The canvas window does not appear                                | It is probably behind other windows: click the Python icon on the taskbar, or press Alt+Tab.                                                                                                            |
| `Cannot reach camera ...`                                        | Is the camera app running? Are both devices on the same network? Try `http://<IP>:8080` in the laptop's browser (Step 2.8). Press `I` and type the correct IP. On venue Wi-Fi, use the phone's hotspot. |
| The camera window shows video but no green hand lines            | Move your hand inside the green box and closer to the camera, with good light, palm facing the camera.                                                                                                  |
| Strokes appear where you did not want them                       | Keep the thumb and index clearly apart when not writing. Press `U` to undo.                                                                                                                             |
| `Glove not connected (no glove USB port found)`                  | Use the data cable. Check Device Manager (Step 2.7): the board must be under Ports (COM & LPT). Close Thonny or Arduino if they are open. AirPen retries every few seconds by itself.                   |
| The glove cursor moves diagonally or the wrong way               | Press `K` and calibrate again, with your hand in the writing position.                                                                                                                                  |
| The glove cursor is too fast or too slow                         | `-` or `+` in glove mode.                                                                                                                                                                               |
| The pen stays down or draws by itself in glove mode              | Straighten the finger fully. If the flex sensor wires are loose, press them in and tape them.                                                                                                           |
| `Could not read that`                                            | Write bigger and clearer, with thicker lines (`]`), then submit again.                                                                                                                                  |
| Recognition takes long the first time                            | Normal: the first reading warms up the model. Later ones are faster.                                                                                                                                    |

**Logs for troubleshooting:** every session records the glove's data in
`glove_logs` and the camera's in `camera_logs`, inside the project folder.
You can delete these folders at any time.
