"""AirPen glove firmware (MicroPython).

Sends one CSV line about 50 times a second over the USB cable:

    heading,pitch,roll,flex_index,flex_middle,pen_down,gesture,gyro_x,gyro_y,gyro_z

The gyro values are rotation speeds in degrees per second around the
sensor's own axes. Unlike the angles, they behave the same however the
sensor is mounted, so the PC can steer the cursor with them.

glove_reader.py on the PC reads these lines. Lines starting with "#" are
status messages and are ignored by the reader.

Saved on the board as main.py, this starts by itself whenever the ESP32 is
powered on, with no PC needed to launch it.
"""

from machine import ADC, I2C, Pin
import time

# ---------------------------------------------------------------- settings
SCL_PIN = 21
SDA_PIN = 19
BNO_ADDRESS = 0x29
I2C_FREQUENCY = 50_000  # Slower is more tolerant of long glove wires.

FLEX_INDEX_PIN = 35  # The pen sensor: the one working flex sensor, sewn into the middle finger.
FLEX_MIDDLE_PIN = 34  # Unused while USE_MIDDLE_FINGER is False.
# The index-finger sensor broke, so the glove runs on one sensor: the middle
# finger (on FLEX_INDEX_PIN) works the pen. With this False the second
# channel is not read (an empty pin picks up noise) and "recognize" comes
# from the Enter key on the PC instead.
USE_MIDDLE_FINGER = False
FLEX_SAMPLES = 4

# Both sensors read LOWER when the finger curls. Measured with the finger
# held straight and curled for 10 s each, this sensor now snaps to about
# 0-15 when curled, while straight has read anywhere from about 80 to 400 as
# it sags and jumps over time. So a bend is judged by a fixed low line, which
# the wandering straight level never reaches, rather than relative to it.
BENT_BELOW = 45          # Curled: the filtered reading falls below this.
STRAIGHT_ABOVE = 70      # Straight again: it climbs back above this.
MEDIAN_WINDOW = 5        # Median of 0.1 s: drops flickers and 2-sample spikes.

SEND_INTERVAL_MS = 20  # 50 lines per second.

# --------------------------------------------------------------- registers
REG_CHIP_ID = 0x00
REG_GYRO = 0x14
REG_EULER = 0x1A
REG_OPR_MODE = 0x3D
REG_SYS_TRIGGER = 0x3F
CHIP_ID_BNO055 = 0xA0
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C

i2c = I2C(0, scl=Pin(SCL_PIN), sda=Pin(SDA_PIN), freq=I2C_FREQUENCY)


def recover_bus():
    """Free an I2C bus jammed by a message cut short mid-byte.

    A glove wire that wobbles can interrupt a transfer while the sensor is
    holding the data line low. The sensor then waits forever for clock pulses
    that never come, and every later read fails. Sending nine manual clock
    pulses lets it finish, and a STOP returns the bus to idle.
    """
    global i2c
    try:
        i2c.deinit()
    except Exception:
        pass

    sda = Pin(SDA_PIN, Pin.OPEN_DRAIN, Pin.PULL_UP)
    scl = Pin(SCL_PIN, Pin.OPEN_DRAIN, Pin.PULL_UP)
    sda.value(1)
    for _ in range(9):
        scl.value(0)
        time.sleep_us(10)
        scl.value(1)
        time.sleep_us(10)
    sda.value(0)  # STOP: data rises while the clock is high.
    time.sleep_us(10)
    scl.value(1)
    time.sleep_us(10)
    sda.value(1)
    time.sleep_ms(20)

    i2c = I2C(0, scl=Pin(SCL_PIN), sda=Pin(SDA_PIN), freq=I2C_FREQUENCY)
    time.sleep_ms(50)


def read_reg(register, length, retries=3):
    for _ in range(retries):
        try:
            return i2c.readfrom_mem(BNO_ADDRESS, register, length)
        except OSError:
            time.sleep_ms(5)
    return None


def write_reg(register, value):
    try:
        i2c.writeto_mem(BNO_ADDRESS, register, bytes([value]))
        return True
    except OSError:
        return False


def start_sensor():
    """Reset the BNO055 and put it into fused-orientation mode."""
    chip = read_reg(REG_CHIP_ID, 1)
    if chip is None or chip[0] != CHIP_ID_BNO055:
        return False
    write_reg(REG_SYS_TRIGGER, 0x20)
    time.sleep_ms(700)
    write_reg(REG_OPR_MODE, MODE_CONFIG)
    time.sleep_ms(50)
    write_reg(REG_OPR_MODE, MODE_NDOF)
    time.sleep_ms(100)
    return True


def read_angles():
    """Return (heading, roll, pitch) in degrees, or None if the read failed."""
    data = read_reg(REG_EULER, 6)
    if data is None:
        return None
    angles = []
    for index in range(0, 6, 2):
        raw = data[index] | (data[index + 1] << 8)
        if raw > 32767:
            raw -= 65536
        angles.append(raw / 16.0)
    return angles


def read_gyro():
    """Return (x, y, z) rotation speed in degrees per second, or None."""
    data = read_reg(REG_GYRO, 6)
    if data is None:
        return None
    rates = []
    for index in range(0, 6, 2):
        raw = data[index] | (data[index + 1] << 8)
        if raw > 32767:
            raw -= 65536
        rates.append(raw / 16.0)
    return rates


def make_adc(pin_number):
    adc = ADC(Pin(pin_number))
    adc.atten(ADC.ATTN_11DB)
    return adc


def read_flex(adc):
    total = 0
    for _ in range(FLEX_SAMPLES):
        total += adc.read_u16()
    return (total // FLEX_SAMPLES) >> 4


class BendDetector:
    """Decide bent or straight from a filtered flex reading.

    Two lines instead of one (BENT_BELOW and STRAIGHT_ABOVE) keep the state
    from flickering when a reading hovers near either of them.
    """

    def __init__(self):
        self.window = []
        self.bent = False

    def update(self, reading):
        self.window.append(reading)
        if len(self.window) > MEDIAN_WINDOW:
            self.window.pop(0)
        value = sorted(self.window)[len(self.window) // 2]
        if self.bent:
            self.bent = value < STRAIGHT_ABOVE
        else:
            self.bent = value < BENT_BELOW
        return self.bent


index_adc = make_adc(FLEX_INDEX_PIN)
middle_adc = make_adc(FLEX_MIDDLE_PIN) if USE_MIDDLE_FINGER else None

print("# AirPen glove starting")
while not start_sensor():
    print("# BNO055 not answering - check the wiring")
    recover_bus()
    time.sleep_ms(1000)
print("# BNO055 ready - sending heading,pitch,roll,flex_i,flex_m,pen,gesture,gx,gy,gz")

# Assume the fingers start straight, and measure that as the first level.
index_detector = BendDetector()
middle_detector = BendDetector() if USE_MIDDLE_FINGER else None
print("# pen finger reads {} now (curled is below {})".format(read_flex(index_adc), BENT_BELOW))

pen_down = False
gesture = False
dropouts = 0
reported_dropout = False
next_send = time.ticks_ms()

while True:
    now = time.ticks_ms()
    if time.ticks_diff(next_send, now) > 0:
        continue
    next_send = time.ticks_add(now, SEND_INTERVAL_MS)

    angles = read_angles()
    if angles is None:
        # Unjam the bus, then set the sensor up again. Report only the first
        # of a run of failures, so one loose wire cannot flood the PC.
        recover_bus()
        recovered = start_sensor()
        dropouts += 1
        if not reported_dropout:
            print("# sensor dropped out ({}) - recovering".format(dropouts))
            reported_dropout = True
        if recovered:
            print("# sensor back after {} dropouts".format(dropouts))
            reported_dropout = False
        continue
    reported_dropout = False

    heading, roll, pitch = angles
    gyro = read_gyro() or (0.0, 0.0, 0.0)
    flex_index = read_flex(index_adc)
    flex_middle = read_flex(middle_adc) if USE_MIDDLE_FINGER else 0
    pen_down = index_detector.update(flex_index)
    gesture = USE_MIDDLE_FINGER and middle_detector.update(flex_middle)

    # The PC reader expects heading, then pitch, then roll.
    print("{:.2f},{:.2f},{:.2f},{},{},{},{},{:.2f},{:.2f},{:.2f}".format(
        heading, pitch, roll, flex_index, flex_middle,
        1 if pen_down else 0, 1 if gesture else 0, gyro[0], gyro[1], gyro[2]))
