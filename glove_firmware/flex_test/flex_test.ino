void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.printf("index %d middle %d\n", analogRead(34), analogRead(35));
  delay(120);
}

