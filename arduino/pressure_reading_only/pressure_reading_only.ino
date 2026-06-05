#include <Wire.h>
#include <cmath>
#include <numeric>

#define TCA_ADDR 0x70
#define SENSOR_ADDR 0x28

// --- Global Constants ---
const int AVG_WINDOW = 5;
float pressureBuffer[2][AVG_WINDOW] = {0};
int bufferIndex = 0;
float zeroOffset[2] = {0};
const char *fanType = "9_blade";

// --- Test Routine Constants ---
const int MEASUREMENT_DURATION_MS = 2000;
const int SAMPLES_PER_RECORDING = 100;
const int SAMPLE_INTERVAL_MS = MEASUREMENT_DURATION_MS / SAMPLES_PER_RECORDING; // 20ms

// --- Test State Management ---
// ADJUST_WAIT: 2s idle while suction_throttle adjusts position
// RECORDING:   2s active sampling
enum TestState { IDLE, DELAY_10S, ADJUST_WAIT, RECORDING, DONE };
TestState currentTestState = IDLE;
unsigned long testTimer = 0;
unsigned long lastSampleTime = 0;
int currentPoint = 0;
int sampleCount = 0;

// --- Data Storage ---
float dp0Samples[SAMPLES_PER_RECORDING];
float dp1Samples[SAMPLES_PER_RECORDING];
float temperatureSamples[SAMPLES_PER_RECORDING];

// --- Data Structure for Results ---
struct Stats {
  float mean;
  float stdDev;
};

// --- Utility Functions ---

void selectTCAChannel(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

bool readSensorRaw(float &pressurePa, float &temperatureC) {
  Wire.beginTransmission(SENSOR_ADDR);
  if (Wire.endTransmission() != 0) return false;

  Wire.requestFrom(SENSOR_ADDR, 4);
  if (Wire.available() != 4) return false;

  uint8_t data[4];
  for (int i = 0; i < 4; i++) data[i] = Wire.read();

  uint16_t rawPressure = ((data[0] & 0x3F) << 8) | data[1];
  uint16_t rawTemp = ((data[2] << 8) | data[3]) >> 5;

  float pressureInH2O = ((float)(rawPressure - 8192) / (6554) * 0.5);
  pressurePa = pressureInH2O * 249.08891;
  temperatureC = ((float)rawTemp * 200.0 / 2047.0) - 50.0;

  return true;
}

void zeroCalibrate() {
  Serial.println("Calibrating... hold steady.");
  const int samples = 20;
  for (int s = 0; s < 2; s++) {
    float sum = 0;
    for (int i = 0; i < samples; i++) {
      selectTCAChannel(s);
      float p, t;
      if (readSensorRaw(p, t)) sum += p;
      delay(20);
    }
    zeroOffset[s] = sum / samples;
    Serial.print("Sensor "); Serial.print(s);
    Serial.print(" zero offset: "); Serial.print(zeroOffset[s], 2);
    Serial.println(" Pa");
  }
  Serial.println("Calibration complete.\n");
}

void addToMovingAverage(uint8_t sensor, float value) {
  pressureBuffer[sensor][bufferIndex] = value;
}

float getAverage(uint8_t sensor) {
  float sum = 0;
  for (int i = 0; i < AVG_WINDOW; i++) sum += pressureBuffer[sensor][i];
  return sum / AVG_WINDOW;
}

// --- Statistical Functions ---

Stats calculateStats(float data[], int count) {
  Stats s;
  if (count == 0) { s.mean = 0.0; s.stdDev = 0.0; return s; }

  float sum = 0;
  for (int i = 0; i < count; i++) sum += data[i];
  s.mean = sum / count;

  float varianceSum = 0;
  for (int i = 0; i < count; i++) varianceSum += pow(data[i] - s.mean, 2);
  s.stdDev = (count > 1) ? sqrt(varianceSum / count) : 0.0;

  return s;
}

// --- Test Routine Functions ---

void performMeasurementAndStore() {
  selectTCAChannel(0);
  float p0_raw, t0;
  if (readSensorRaw(p0_raw, t0)) {
    float p0_corrected = p0_raw - zeroOffset[0];
    dp0Samples[sampleCount] = p0_corrected;
    temperatureSamples[sampleCount] = t0;

    selectTCAChannel(1);
    float p1_raw, t1;
    if (readSensorRaw(p1_raw, t1)) {
      float p1_corrected = p1_raw - zeroOffset[1];
      dp1Samples[sampleCount] = p1_corrected;
      sampleCount++;
    } else {
      Serial.println("Sensor 1 read error during recording.");
    }
  } else {
    Serial.println("Sensor 0 read error during recording.");
  }
}

void calculateAndReportStats() {
  Serial.println("\n--- TEST POINT COMPLETE ---");
  Serial.print("Point: "); Serial.println(currentPoint);
  Serial.print("Samples Collected: "); Serial.println(sampleCount);
  Serial.println("Variable,Mean,StdDev");

  Stats dp0Stats = calculateStats(dp0Samples, sampleCount);
  Serial.print("dp_sensor0,"); Serial.print(dp0Stats.mean, 2);
  Serial.print(","); Serial.println(dp0Stats.stdDev, 2);

  Stats dp1Stats = calculateStats(dp1Samples, sampleCount);
  Serial.print("dp_sensor1,"); Serial.print(dp1Stats.mean, 2);
  Serial.print(","); Serial.println(dp1Stats.stdDev, 2);

  Stats temperatureStats = calculateStats(temperatureSamples, sampleCount);
  Serial.print("temp,"); Serial.print(temperatureStats.mean, 2);
  Serial.print(","); Serial.println(temperatureStats.stdDev, 2);

  Serial.println("---------------------------\n");

  sampleCount = 0;
  currentPoint++;
}

// --- Idle Mode ---

void handleIdleMode() {
  unsigned long currentMillis = millis();
  static unsigned long lastIdlePrint = 0;
  const unsigned long idleInterval = 1000;

  if (currentMillis - lastIdlePrint >= idleInterval) {
    lastIdlePrint = currentMillis;
    for (int sensorID = 0; sensorID < 2; sensorID++) {
      selectTCAChannel(sensorID);
      float pressureRaw, temp;
      if (readSensorRaw(pressureRaw, temp)) {
        float pressureCorrected = pressureRaw - zeroOffset[sensorID];
        addToMovingAverage(sensorID, pressureCorrected);
        float avgPressure = getAverage(sensorID);
        if (sensorID == 0) {
          Serial.print("Sensor 0: dp:"); Serial.print(avgPressure, 2);
          Serial.print(" ; (Temp: "); Serial.print(temp, 2); Serial.println(" °C)");
        } else {
          Serial.print("Sensor 1: dp:"); Serial.println(avgPressure, 2);
        }
      } else {
        Serial.print("Sensor "); Serial.print(sensorID); Serial.println(": read error");
      }
    }
    Serial.println();
    bufferIndex = (bufferIndex + 1) % AVG_WINDOW;
  }
}

// --- Setup and Loop ---

void setup() {
  Wire.begin();
  Serial.begin(115200);
  delay(100);
  zeroCalibrate();
  Serial.println("Ready. Send 'run_test' to begin.");
}

void loop() {
  // Serial input
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.equalsIgnoreCase("run_test")) {
      if (currentTestState == IDLE) {
        currentTestState = DELAY_10S;
        testTimer = millis();
        currentPoint = 0;
        Serial.println("\n--- RUNNING TEST ---");
        Serial.println("Command received. Calibrating and waiting 10s...");
        zeroCalibrate();
      } else {
        Serial.println("Test already running.");
      }
    }
    if (command.equalsIgnoreCase("stop")) {
      currentTestState = DONE;
    }
  }

  unsigned long currentMillis = millis();

  switch (currentTestState) {
    case IDLE:
      handleIdleMode();
      break;

    case DELAY_10S:
      // 10s startup delay, then begin first adjust-wait
      if (currentMillis - testTimer >= 10000) {
        Serial.println("Startup delay complete. Entering first adjust window...");
        currentTestState = ADJUST_WAIT;
        testTimer = currentMillis;
      }
      break;

    case ADJUST_WAIT:
      // 2s window: suction_throttle is adjusting position, we do nothing
      if (currentMillis - testTimer >= 2000) {
        Serial.print("Starting Recording Point ");
        Serial.println(currentPoint + 1);
        currentTestState = RECORDING;
        testTimer = currentMillis;
        sampleCount = 0;
        lastSampleTime = currentMillis;
      }
      break;

    case RECORDING:
      // Non-blocking sampling every 20ms
      if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        performMeasurementAndStore();
        lastSampleTime = currentMillis;
      }
      // After 2s, report stats and go back to adjust-wait for next point
      if (currentMillis - testTimer >= 2000) {
        calculateAndReportStats();
        currentTestState = ADJUST_WAIT;
        testTimer = currentMillis;
      }
      break;

    case DONE:
      currentTestState = IDLE;
      Serial.println("Test finished. Returning to IDLE.");
      break;
  }
}
