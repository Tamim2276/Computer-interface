"""AirPen glove - middle finger flex sensor test on GPIO 35 (MicroPython, Thonny).

Prints the GPIO 35 reading five times a second with a simple bar, and says
what the number means. Bend and straighten the middle finger and watch it move.
"""

from machine import ADC, Pin
import time

FLEX_PIN = 35
SAMPLES = 8
DISCONNECTED_BELOW = 40   # A sensor that is not wired in reads close to zero.
SHORTED_ABOVE = 4000      # The pin is touching 3V3 directly, skipping the sensor.

adc = ADC(Pin(FLEX_PIN))
adc.atten(ADC.ATTN_11DB)  # Full 0-3.3 V range.


def read_flex():
    """Return a steady 0-4095 reading."""
    total = 0
    for _ in range(SAMPLES):
        total += adc.read_u16()
    return (total // SAMPLES) >> 4


def describe(value):
    if value < DISCONNECTED_BELOW:
        return "NOT CONNECTED - check the sensor legs and the GPIO 35 wire"
    if value > SHORTED_ABOVE:
        return "TOUCHING 3V3 directly - is the resistor to GND missing?"
    return "OK - bend the finger and this number should drop"


print("GPIO 35 flex test - press Ctrl+C to stop\n")
lowest, highest = 4095, 0
while True:
    value = read_flex()
    lowest = min(lowest, value)
    highest = max(highest, value)
    bar = "#" * (value // 100)
    print("{:5d}  (min {:4d}, max {:4d})  {:<20}  {}".format(
        value, lowest, highest, bar[:20], describe(value)))
    time.sleep_ms(200)
