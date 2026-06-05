// PID Motor Controller with RPM Feedback & INA226 Power Monitoring
// Synchronized timing: 2s ADJUST_WAIT then 2s RECORDING per point

#include <INA226_WE.h>
#include <Wire.h>

// --- INA226 Settings ---
#define INA226_ADDR 0x41
INA226_WE ina226(INA226_ADDR);

// --- Welford's Algorithm Class ---
class Stats {
  public:
    void reset() {
      count = 0; mean = 0; M2 = 0;
      minVal = 999999; maxVal = -999999;
    }

    void update(double newValue) {
      count++;
      double delta = newValue - mean;
      mean += delta / count;
      double delta2 = newValue - mean;
      M2 += delta * delta2;
      if (newValue < minVal) minVal = newValue;
      if (newValue > maxVal) maxVal = newValue;
    }

    double getMean()   { return mean; }
    double getStdDev() { return (count < 2) ? 0.0 : sqrt(M2 / (count - 1)); }
    double getMin()    { return minVal; }
    double getMax()    { return maxVal; }
    long   getCount()  { return count; }

  private:
    long count = 0;
    double mean = 0.0, M2 = 0.0;
    double minVal = 999999, maxVal = -999999;
};

Stats rpmStats;
Stats powerStats;

// --- PID Constants ---
double Kp = 0;
double Ki = 0.08;
double Kd = 0;

// --- Pin Definitions ---
const int MOTOR_PIN    = 3;
const int FEEDBACK_PIN = 2;

// --- Variables ---
volatile unsigned long pulseCount = 0;
unsigned long lastTime   = 0;
unsigned long sampleTime = 200;

double targetRPM  = 0;
double currentRPM = 0;
double pwmOutput  = 0;
float  smoothedRPM = 0;
float  power_mW   = 0;
float  alpha      = 0.3;

double targetPWM = 0;
double prevPWM   = 0;
const uint8_t  rampStep  = 1;
const uint16_t stepDelay = 2;

// --- PID State ---
double error = 0, lastError = 0, integral = 0, derivative = 0;

// --- Test Sequence ---
const int TEST_RPM = 3000;

// ADJUST_WAIT: 2s idle while suction_throttle adjusts position
// RECORDING:   2s active sampling
enum TestState { IDLE, DELAY_10S, ADJUST_WAIT, RECORDING, DONE };
TestState currentTestState = IDLE;
unsigned long testStateStartTime = 0;
int currentPoint = 0;

// --- Functions ---

void adjustPWM() {
  if (targetPWM > prevPWM) {
    for (int d = prevPWM; d <= targetPWM; d += rampStep) {
      analogWrite(MOTOR_PIN, d);
      delay(stepDelay);
    }
  } else if (targetPWM < prevPWM) {
    for (int d = prevPWM; d >= targetPWM; d -= rampStep) {
      analogWrite(MOTOR_PIN, d);
      delay(stepDelay);
    }
  }
  prevPWM = targetPWM;
}

void set_pwm(unsigned long timeChange) {
  error      = targetRPM - smoothedRPM;
  integral  += error * (timeChange / 1000.0);
  derivative = (error - lastError) / (timeChange / 1000.0);

  if (targetRPM == 0) {
    pwmOutput = 0;
    integral  = 0;
  } else {
    pwmOutput = (Kp * error) + (Ki * integral) + (Kd * derivative);
    pwmOutput = constrain(pwmOutput, 30, 255);
  }

  targetPWM = (int)pwmOutput;
  adjustPWM();
  prevPWM = targetPWM;
}

void idle_mode(unsigned long now, unsigned long timeChange) {
  if (timeChange >= sampleTime) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    currentRPM  = (pulses * 60000.0) / (timeChange * 2);
    smoothedRPM = alpha * currentRPM + (1 - alpha) * smoothedRPM;

    ina226.readAndClearFlags();
    power_mW = ina226.getBusPower();

    set_pwm(timeChange);
    lastError = error;
    lastTime  = now;

    static unsigned long lastPrint = 0;
    if (now - lastPrint >= 500) {
      Serial.print("Tgt:"); Serial.print(targetRPM, 0);
      Serial.print(" | RPM:"); Serial.print(smoothedRPM, 0);
      Serial.print(" | PWM:"); Serial.print((int)pwmOutput);
      Serial.print(" | Pwr:"); Serial.println(power_mW, 1);
      lastPrint = now;
    }
  }
}

void printStats() {
  Serial.print("DATA_PKT,");
  Serial.print(currentPoint);        Serial.print(",");
  Serial.print(rpmStats.getMean());  Serial.print(",");
  Serial.print(rpmStats.getStdDev()); Serial.print(",");
  Serial.print(powerStats.getMean()); Serial.print(",");
  Serial.println(powerStats.getStdDev());
}

void run_test(unsigned long now, unsigned long timeChange) {

  // Continuous measurement update
  bool newSampleReady = false;
  if (timeChange >= sampleTime) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    currentRPM  = (pulses * 60000.0) / (timeChange * 2);
    smoothedRPM = alpha * currentRPM + (1 - alpha) * smoothedRPM;
    power_mW    = ina226.getBusPower();

    set_pwm(timeChange);
    lastTime       = now;
    newSampleReady = true;
  }

  switch (currentTestState) {

    case DELAY_10S:
      // Ramp up RPM after 3s, transition to first adjust-wait after 10s
      if (now - testStateStartTime < 3000) targetRPM = 0;
      else targetRPM = TEST_RPM;

      if (now - testStateStartTime >= 10000) {
        Serial.println("Startup complete. Entering first adjust window...");
        currentTestState    = ADJUST_WAIT;
        testStateStartTime  = now;
        currentPoint        = 0;
      }
      break;

    case ADJUST_WAIT:
      // 2s idle: suction_throttle is moving, we just hold RPM
      if (now - testStateStartTime >= 2000) {
        currentTestState   = RECORDING;
        testStateStartTime = now;
        rpmStats.reset();
        powerStats.reset();
      }
      break;

    case RECORDING:
      if (newSampleReady) {
        rpmStats.update(smoothedRPM);
        powerStats.update(power_mW);
      }
      // After 2s, print stats and return to adjust-wait
      if (now - testStateStartTime >= 2000) {
        printStats();
        currentPoint++;
        currentTestState   = ADJUST_WAIT;
        testStateStartTime = now;
      }
      break;

    case DONE:
      targetRPM = 0;
      analogWrite(MOTOR_PIN, 0);
      currentTestState = IDLE;
      Serial.println("Test Done. Idle.");
      break;

    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_PIN,    OUTPUT);
  pinMode(FEEDBACK_PIN, INPUT);

  Wire.begin();
  if (!ina226.init()) {
    Serial.println("Failed to init INA226!");
    while (1) {}
  }

  ina226.waitUntilConversionCompleted();
  ina226.setResistorRange(0.08, 1.0);
  ina226.setAverage(INA226_AVERAGE_128);

  TCCR2B = (TCCR2B & 0b11111000) | 0x01;
  attachInterrupt(digitalPinToInterrupt(FEEDBACK_PIN), countPulse, RISING);

  lastTime = millis();
  Serial.println("System Ready. Send 'run_test' or RPM value.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("run_test")) {
      if (currentTestState == IDLE) {
        Serial.println("Test Initiated.");
        currentTestState   = DELAY_10S;
        testStateStartTime = millis();
        integral           = 0;
      }
    } else if (input.equalsIgnoreCase("stop")) {
      currentTestState = DONE;
    } else {
      targetRPM = input.toFloat();
      targetRPM = constrain(targetRPM, 0, 3700);
      integral  = 0;
      Serial.print("Target RPM: "); Serial.println(targetRPM);
    }
  }

  unsigned long now        = millis();
  unsigned long timeChange = now - lastTime;

  if (currentTestState == IDLE) {
    idle_mode(now, timeChange);
  } else {
    run_test(now, timeChange);
  }
}

void countPulse() {
  pulseCount++;
}
