// Step 2 test: is a real BNO055 answering on the I2C bus?
// Instead of a noisy address scan, this asks the sensor for its chip ID.
// A genuine BNO055 always replies 0xA0, which random noise cannot fake.

#include <Wire.h>

const int SDA_PIN = 19;
const int SCL_PIN = 21;
const byte BNO055_CHIP_ID = 0xA0;

// Returns the byte in register 0 (chip ID), or -1 if nothing answered.
int readChipId(byte address) {
  Wire.beginTransmission(address);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) return -1;
  if (Wire.requestFrom(address, (uint8_t)1) != 1) return -1;
  return Wire.read();
}

void setup() {
  Serial.begin(115200);
  delay(1000);  // The BNO055 needs about 650 ms after power-on.
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Serial.println("BNO055 chip ID test - repeats every 2 seconds");
}

void loop() {
  bool found = false;
  for (byte address : {0x28, 0x29}) {
    int id = readChipId(address);
    if (id == BNO055_CHIP_ID) {
      Serial.printf("0x%02X: chip ID 0xA0  -> BNO055 FOUND!\n", address);
      found = true;
    } else if (id >= 0) {
      Serial.printf("0x%02X: answered, but chip ID is 0x%02X (expected 0xA0)\n", address, id);
    } else {
      Serial.printf("0x%02X: no answer\n", address);
    }
  }
  if (!found) Serial.println("BNO055 not found");
  Serial.println();
  delay(2000);
}
