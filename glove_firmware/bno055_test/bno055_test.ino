#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

Adafruit_BNO055 bno(55, 0x28);

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Serial.println(bno.begin() ? "BNO055 OK" : "BNO055 NOT FOUND");
}

void loop() {
  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  Serial.printf("H %.1f P %.1f R %.1f\n", e.x(), e.y(), e.z());
  delay(200);
}

