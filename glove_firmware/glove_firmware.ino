#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

#define FLEX_INDEX_PIN 34
#define FLEX_MIDDLE_PIN 35
#define SDA_PIN 21
#define SCL_PIN 22

Adafruit_BNO055 bno(55, 0x28);

// Calibrate these using the Serial Monitor. With the guide's wiring
// (flex sensor to 3.3V and 10k resistor to GND), bending usually lowers ADC.
const int FLEX_INDEX_THRESHOLD = 2700;
const int FLEX_MIDDLE_THRESHOLD = 2700;
const bool FLEX_BENT_IS_HIGH = false;

unsigned long lastSend = 0;
const unsigned long RATE_MS = 10;

bool isBent(int value, int threshold) {
  return FLEX_BENT_IS_HIGH ? value > threshold : value < threshold;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("# AirPen Glove starting");

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!bno.begin()) {
    Serial.println("# ERROR: BNO055 not found");
    Serial.println("# Check 3.3V, GND, SDA GPIO21, and SCL GPIO22");
    while (true) delay(1000);
  }

  delay(500);
  bno.setMode(OPERATION_MODE_NDOF);
  delay(1000);
  Serial.println("# BNO055 OK; move the glove in a figure-8 to calibrate");

  uint8_t systemCal = 0, gyroCal = 0, accelCal = 0, magCal = 0;
  int attempts = 0;
  do {
    bno.getCalibration(&systemCal, &gyroCal, &accelCal, &magCal);
    Serial.printf("# CAL sys=%d gyro=%d accel=%d mag=%d\n",
                  systemCal, gyroCal, accelCal, magCal);
    delay(500);
    attempts++;
  } while (systemCal < 2 && attempts < 60);

  Serial.println("# READY");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend < RATE_MS) return;
  lastSend = now;

  imu::Vector<3> orientation =
      bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  int flexIndex = analogRead(FLEX_INDEX_PIN);
  int flexMiddle = analogRead(FLEX_MIDDLE_PIN);
  bool penDown = isBent(flexIndex, FLEX_INDEX_THRESHOLD);
  bool recognizeGesture = isBent(flexMiddle, FLEX_MIDDLE_THRESHOLD);

  Serial.print(orientation.x(), 2); Serial.print(',');
  Serial.print(orientation.y(), 2); Serial.print(',');
  Serial.print(orientation.z(), 2); Serial.print(',');
  Serial.print(flexIndex); Serial.print(',');
  Serial.print(flexMiddle); Serial.print(',');
  Serial.print(penDown ? 1 : 0); Serial.print(',');
  Serial.println(recognizeGesture ? 1 : 0);
}

