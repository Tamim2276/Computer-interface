// Wire test: shows whether GPIO 19 and GPIO 21 read HIGH or LOW.
// The pins are inputs only, so touching a wire from them to 3V3 or GND is safe.

const int SDA_PIN = 19;
const int SCL_PIN = 21;

void setup() {
  Serial.begin(115200);
  pinMode(SDA_PIN, INPUT_PULLDOWN);
  pinMode(SCL_PIN, INPUT_PULLDOWN);
  delay(1000);
  Serial.println("Pin test - touch a wire to 3V3 and watch it change to HIGH");
}

void loop() {
  Serial.printf("GPIO19 (SDA): %s    GPIO21 (SCL): %s\n",
                digitalRead(SDA_PIN) ? "HIGH" : "low ",
                digitalRead(SCL_PIN) ? "HIGH" : "low ");
  delay(500);
}
