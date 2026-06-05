#include <Encoder.h>

// --- Pins ---
const uint8_t pwmPin       = 3;
const int     pulsePin     = 5;
const int     directionPin = 6;
const int     pinA         = 2;
const int     pinB         = 8;

// --- Constants ---
const unsigned long stepsPerRev = 20000; // 200*100
const unsigned long rpm_stepper = 200;
const int stepDelay = (60 * 1000 * 1000) / (2 * stepsPerRev * rpm_stepper);

const int encoderStatesPerRev = 96;
const int nbTestPointsFan      = 0;
const int nbTestPointsThrottle = 35;

const uint8_t  rampStep  = 1;
const uint16_t rampDelay = 392; // us

// Timing (must match pressure and RPM boards)
const unsigned long ADJUST_DURATION_MS = 2000; // move + stabilise
const unsigned long HOLD_DURATION_MS   = 2000; // recording window on other boards
const unsigned long STARTUP_DELAY_MS   = 10000;
const unsigned long CALIB_DELAY_MS     = 3000;

// --- Variables ---
double  currentAngle = 0;
int     targetPWM    = 0;
int     prevPWM      = 0;
int     currentPoint = -1;

double testAngles[nbTestPointsThrottle * 2];
int    totalPoints; // fan + throttle

// State machine
enum TestState { IDLE, DELAY_10S, ADJUSTING, HOLDING, DONE };
TestState currentTestState = IDLE;
unsigned long stateTimer   = 0;

// Track which step of the sequence we're on
int sequenceIndex = 0; // indexes into the full point list

Encoder myEnc(pinA, pinB);

// --- Stepper Logic ---
void rotateToAngle(double targetAngle) {
  targetAngle   = constrain(targetAngle, 0, 360);
  double deltaAngle = targetAngle - currentAngle;
  long   steps   = (deltaAngle * stepsPerRev) / 360.0;

  if (abs(steps) < 1) return;

  digitalWrite(directionPin, (steps > 0) ? LOW : HIGH);
  steps = abs(steps);

  for (long i = 0; i < steps; i++) {
    digitalWrite(pulsePin, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(pulsePin, LOW);
    delayMicroseconds(stepDelay);
  }
  currentAngle = targetAngle;
}

void verifyAndCorrect() {
  long expectedCount = (currentAngle / 360.0) * encoderStatesPerRev;
  long actualCount   = abs(myEnc.read()) % encoderStatesPerRev;

  if (actualCount - 1 > expectedCount || actualCount + 1 < expectedCount) {
    double errorDegrees = ((actualCount - expectedCount) / (double)encoderStatesPerRev) * 360.0;
    rotateToAngle(currentAngle + errorDegrees);
    currentAngle += errorDegrees;
  }
}

void adjustPWM() {
  if (targetPWM == prevPWM) return;

  if (targetPWM > prevPWM) {
    for (int d = prevPWM; d <= targetPWM; d += rampStep) {
      analogWrite(pwmPin, d);
      delayMicroseconds(rampDelay);
    }
  } else {
    for (int d = prevPWM; d >= targetPWM; d -= rampStep) {
      analogWrite(pwmPin, d);
      delayMicroseconds(rampDelay);
    }
  }
  prevPWM = targetPWM;
}

// --- Apply the setpoint for a given sequence index ---
void applySetpoint(int idx) {
  if (idx < nbTestPointsFan) {
    // Fan sweep: throttle closed, PWM varies
    rotateToAngle(0);
    targetPWM = map(idx, 0, nbTestPointsFan, 255, 185);
    adjustPWM();
  } else {
    // Throttle sweep: fan off, angle varies
    int tIdx  = idx - nbTestPointsFan;
    targetPWM = 0;
    adjustPWM();
    rotateToAngle(testAngles[tIdx]);
    verifyAndCorrect();
  }
}

// --- Report current location (called at start of HOLDING so PC knows position is locked) ---
void reportLocation(int idx) {
  Serial.print("Point: ");
  Serial.print(currentPoint);
  Serial.print(", Location: ");
  if (idx < nbTestPointsFan) {
    Serial.println(-1 * targetPWM); // negative = fan mode, consistent with original
  } else {
    int tIdx = idx - nbTestPointsFan;
    Serial.println(testAngles[tIdx]);
  }
}

// --- Setup ---
void setup() {
  pinMode(pwmPin,       OUTPUT);
  pinMode(pulsePin,     OUTPUT);
  pinMode(directionPin, OUTPUT);

  Serial.begin(115200);

  TCCR2A = _BV(COM2B1) | _BV(WGM21) | _BV(WGM20);
  TCCR2B = _BV(CS20);

  totalPoints = nbTestPointsFan + nbTestPointsThrottle * 2;

  // Build throttle angle array (up then down)
  for (int i = 0; i < nbTestPointsThrottle * 2; i++) {
    if (i >= nbTestPointsThrottle) {
      testAngles[i] = 90 - (90.0 / nbTestPointsThrottle) * (i - nbTestPointsThrottle);
    } else {
      testAngles[i] = (90.0 / nbTestPointsThrottle) * (i);
    }
  }
  myEnc.write(0);
  Serial.println("System Ready. Enter % (-100 to 100) or send 'run_test':");
}

// --- Loop ---
void loop() {

  // Handle serial input
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "run_test") {
      if (currentTestState == IDLE) {
        Serial.println("Test initiated. Calibrating...");
        // Initial conditions: ensure throttle closed, fan at full
        rotateToAngle(0);
        // targetPWM = 255;
        // adjustPWM();
        currentPoint   = 0;
        sequenceIndex  = 0;
        currentTestState = DELAY_10S;
        stateTimer     = millis();
      }
    } else if (input == "stop") {
      currentTestState = DONE;
      targetPWM = 0;
      adjustPWM();
      rotateToAngle(0);
    } else {
      // Manual control when IDLE
      if (currentTestState == IDLE) {
        float val = input.toFloat();
        val = constrain(val, -100, 100);
        if (val < 0) {
          targetPWM = map(abs(val), 100, 0, 255, 185);
          adjustPWM();
          rotateToAngle(0);
        } else {
          targetPWM = 0;
          adjustPWM();
          double reqAngle = (val / 100.0) * 90.0;
          rotateToAngle(reqAngle);
        }
        Serial.print("Target PWM: "); Serial.print(targetPWM);
        Serial.print(" | Target Angle: "); Serial.println(currentAngle);
      }
    }
  }

  unsigned long now = millis();

  switch (currentTestState) {

    case IDLE:
      verifyAndCorrect();
      delay(10);
      break;

    case DELAY_10S:
      // Wait for startup (fan spin-up etc.)
      if (now - stateTimer >= STARTUP_DELAY_MS) {
        Serial.println("Startup delay complete. Entering first adjust window...");
        // Apply first setpoint immediately
        applySetpoint(sequenceIndex);
        currentTestState = ADJUSTING;
        stateTimer = now;
      }
      break;

    case ADJUSTING:
      // 2s window: move to position and stabilise.
      // (rotateToAngle/adjustPWM already called when entering this state)
      if (now - stateTimer >= ADJUST_DURATION_MS) {
        // Position locked - report it so PC / other boards know recording can begin
        reportLocation(sequenceIndex);
        currentTestState = HOLDING;
        stateTimer = now;
      }
      break;

    case HOLDING:
      // 2s window: hold position while pressure/RPM boards record.
      // Nothing to do except wait.
      if (now - stateTimer >= HOLD_DURATION_MS) {
        currentPoint++;
        sequenceIndex++;

        if (sequenceIndex >= totalPoints) {
          // All points done
          rotateToAngle(0);
          targetPWM = 0;
          adjustPWM();
          currentTestState = DONE;
        } else {
          // Apply next setpoint and start next adjust window
          applySetpoint(sequenceIndex);
          currentTestState = ADJUSTING;
          stateTimer = now;
        }
      }
      break;

    case DONE:
      Serial.println("Test Complete.");
      currentTestState = IDLE;
      break;
  }
}
