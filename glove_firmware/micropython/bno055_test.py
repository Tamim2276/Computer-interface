"""AirPen glove - motion sensor test (MicroPython, run in Thonny).

Prints heading / roll / pitch plus the calibration status four times a second.
Tilt and turn the sensor: the numbers must change smoothly.
If a wire wobbles and the sensor restarts, this sets it up again by itself.
"""

from machine import I2C, Pin
import time

# Wiring that is known to work on this glove.
SCL_PIN = 21
SDA_PIN = 19
BNO_ADDRESS = 0x29
I2C_FREQUENCY = 100_000  # Drop to 50_000 if readings ever become unreliable.

REG_CHIP_ID = 0x00
REG_EULER = 0x1A
REG_CALIB_STATUS = 0x35
REG_OPR_MODE = 0x3D
REG_SYS_TRIGGER = 0x3F
CHIP_ID_BNO055 = 0xA0
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C  # Fused absolute orientation, the mode AirPen needs.

i2c = I2C(0, scl=Pin(SCL_PIN), sda=Pin(SDA_PIN), freq=I2C_FREQUENCY)


def read_reg(register, length, retries=3):
    """Read registers, returning None when the sensor does not answer."""
    for _ in range(retries):
        try:
            return i2c.readfrom_mem(BNO_ADDRESS, register, length)
        except OSError:
            time.sleep_ms(20)
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
    write_reg(REG_SYS_TRIGGER, 0x20)  # Soft reset.
    time.sleep_ms(700)
    write_reg(REG_OPR_MODE, MODE_CONFIG)
    time.sleep_ms(50)
    write_reg(REG_OPR_MODE, MODE_NDOF)
    time.sleep_ms(100)
    return True


def sensor_is_running():
    """A sensor that lost power drops out of NDOF mode and reports zeros."""
    mode = read_reg(REG_OPR_MODE, 1)
    return mode is not None and mode[0] == MODE_NDOF


def read_angles():
    """Return (heading, roll, pitch) in degrees, or None if the read failed."""
    data = read_reg(REG_EULER, 6)
    if data is None:
        return None
    angles = []
    for index in range(0, 6, 2):
        raw = data[index] | (data[index + 1] << 8)
        if raw > 32767:  # Two's complement: the sensor sends signed values.
            raw -= 65536
        angles.append(raw / 16.0)
    return angles


def read_calibration():
    """Return (system, gyro, accel, mag), each 0 (uncalibrated) to 3 (best)."""
    data = read_reg(REG_CALIB_STATUS, 1)
    status = data[0] if data else 0
    return (status >> 6) & 3, (status >> 4) & 3, (status >> 2) & 3, status & 3


if not start_sensor():
    raise RuntimeError("BNO055 did not answer - check the wiring")
print("Chip ID 0xA0 - BNO055 found")
print("\n heading     roll    pitch  | sys gyr acc mag")
print("-" * 48)

dropouts = 0
while True:
    angles = read_angles()
    if angles is None or not sensor_is_running():
        # A loose wire or a power dip restarts the sensor; set it up again.
        dropouts += 1
        print("sensor dropped out ({}) - restarting it".format(dropouts))
        if not start_sensor():
            time.sleep_ms(500)
        continue

    heading, roll, pitch = angles
    system, gyro, accel, mag = read_calibration()
    print("{:8.2f} {:8.2f} {:8.2f}  |  {}   {}   {}   {}".format(
        heading, roll, pitch, system, gyro, accel, mag))
    time.sleep_ms(250)
