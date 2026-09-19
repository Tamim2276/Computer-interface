"""AirPen glove - read GPIO 32, 34 and 35 side by side (MicroPython, Thonny).

Whichever pin the flex sensor is wired to shows a live number; empty pins
read near 0. Hold the finger straight, then bend it, and watch that column.
"""

from machine import ADC, Pin
import time

PINS = (32, 34, 35)
SAMPLES = 8

adcs = []
for pin_number in PINS:
    adc = ADC(Pin(pin_number))
    adc.atten(ADC.ATTN_11DB)
    adcs.append(adc)


def read_flex(adc):
    total = 0
    for _ in range(SAMPLES):
        total += adc.read_u16()
    return (total // SAMPLES) >> 4


print("Flex pin test - press Ctrl+C to stop\n")
print("   ".join("GPIO{:>3}".format(pin) for pin in PINS))
print("-" * 30)
while True:
    print("   ".join("{:>7d}".format(read_flex(adc)) for adc in adcs))
    time.sleep_ms(200)
