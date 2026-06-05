  #include <Encoder.h>

  // Pins
const uint8_t pwmPin = 3;       // Fan PWM (Timer2)
const int pulsePin = 5;         // Stepper Pulse
const int directionPin = 6;     // Stepper Direction
const int pinA = 2;             // Encoder A (Interrupt)
const int pinB = 8;             // Encoder B 

  // Constants
  const unsigned long stepsPerRev = 20000; 
  const int encoderStatesPerRev = 96; // 24 pulses * 4 quadrature states

  // Variables
  double currentAngle = 0; // Where the motor thinks it is
  Encoder myEnc(pinA, pinB);
  const int nbTestPoints = 50;
  double testAngles[nbTestPoints];


  void setup() {
    pinMode(pulsePin, OUTPUT);
    pinMode(directionPin, OUTPUT);
    Serial.begin(115200);
    delay(500);

    // Initialize test points
    for (int i = 0; i < nbTestPoints; i++) {
      testAngles[i] = (90.0 / nbTestPoints) * (i + 1);
    }
    
    // Initial sync: set encoder to 0
    myEnc.write(0);
    currentAngle = 0;
    
    Serial.println("System Online. Enter angle (-360-360):");
  }

  void rotateToAngle(double targetAngle) {
    if (targetAngle < -360 || targetAngle > 360) return;

    double deltaAngle = targetAngle - currentAngle;
    long steps = (deltaAngle * stepsPerRev) / 360.0;

    if (steps == 0) return;

    // Set Direction
    digitalWrite(directionPin, (steps > 0) ? LOW : HIGH);
    steps = abs(steps);

    // Move Motor
    for (long i = 0; i < steps; i++) {
      digitalWrite(pulsePin, HIGH);
      delayMicroseconds(5);
      digitalWrite(pulsePin, LOW);
      delayMicroseconds(1000);
    }

    currentAngle = targetAngle;
  }

  // THIS IS THE FEEDBACK LOGIC
  void verifyAndCorrect() {
    // 1. Calculate what the encoder SHOULD read based on currentAngle
    long expectedEncoderCount = (currentAngle / 360.0) * encoderStatesPerRev;
    
    // 2. Read what the encoder ACTUALLY reads
    long actualEncoderCount = myEnc.read()%encoderStatesPerRev;

    Serial.println(expectedEncoderCount);
    Serial.println(actualEncoderCount);

    // 3. If there is a discrepancy, the motor skipped/was forced
    if (actualEncoderCount-1 > expectedEncoderCount || actualEncoderCount+1 < expectedEncoderCount) {
      Serial.print("Discrepancy Detected! Correcting...");
      
      // Calculate error in degrees
      double errorDegrees = ((actualEncoderCount - expectedEncoderCount) / (double)encoderStatesPerRev) * 360.0;
      
      // Nudge the motor to match the encoder's truth
      // rotateToAngle(currentAngle + errorDegrees);

      // Set motor angle to new value
      currentAngle = currentAngle + errorDegrees;
      Serial.print("Changed by: ");
      Serial.println(errorDegrees);

      Serial.print("current angle: ");
      Serial.println(currentAngle);
      
      Serial.println(" Fixed.");
    }
  }

  void test() {
    rotateToAngle(0);
    verifyAndCorrect();
    myEnc.write(0); // Reset encoder to 0
    delay(2000);
    for (int i = 0; i < nbTestPoints; i++) {
      rotateToAngle(testAngles[i]);
      verifyAndCorrect();
      delay(500);
    }
  }

  void loop() {
    // 1. Check for Serial Input to move the motor
    if (Serial.available()) {
      String input = Serial.readStringUntil('\n');

      if (input=="test"){
        test();
        rotateToAngle(0);
      }
      else{
        double requestedAngle = input.toFloat();
        
        // 0.018deg increments
        requestedAngle = round(requestedAngle / 0.018) * 0.018;
        
        rotateToAngle(requestedAngle);
        Serial.print("Moved to: ");
        Serial.println(currentAngle);
      }

      // 2. CONSTANT FEEDBACK CHECK
      verifyAndCorrect();
      }
      
    
    delay(100);
  }