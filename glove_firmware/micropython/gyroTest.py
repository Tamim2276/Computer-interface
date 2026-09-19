from machine import ADC, Pin
from time import sleep

index_flex = ADC(Pin(34))
middle_flex = ADC(Pin(35))

index_flex.atten(ADC.ATTN_11DB)
middle_flex.atten(ADC.ATTN_11DB)

index_flex.width(ADC.WIDTH_12BIT)
middle_flex.width(ADC.WIDTH_12BIT)


def average(adc, samples=10):
    total = 0

    for _ in range(samples):
        total += adc.read()
        sleep(0.01)

    return total // samples


while True:
    index_value = average(index_flex)
    middle_value = average(middle_flex)

    print(
        "Index:", index_value,
        "| Middle:", middle_value
    )

    sleep(0.15)