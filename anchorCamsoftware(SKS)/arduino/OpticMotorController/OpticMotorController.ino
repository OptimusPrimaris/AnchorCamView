#include <Wire.h>                 // Allows I2C communication with the FeatherWing.
#include <Adafruit_MotorShield.h> // Provides Adafruit's motor-control instructions.


// Create the motor-driver controller.
// 0x60 is the FeatherWing's default I2C address supposedly.
Adafruit_MotorShield motorWing = Adafruit_MotorShield(0x60);


// Create references for our two physical motors.
// The motors will be assigned to their FeatherWing ports in setup().
Adafruit_StepperMotor *windageMotor;
Adafruit_StepperMotor *elevationMotor;


// Motor settings.
// The NEMA 8 motor requires 200 full steps to complete one revolution.
const int STEPS_PER_REVOLUTION = 200;

// Set the initial motor speed to 20 revolutions per minute.
const int MOTOR_SPEED_RPM = 20;

// After a move, keep both motors powered (holding their position) for this long
// before releasing them, so the turrets settle and stay put.
// 1000 milliseconds = one second. Raise it if the turrets still creep.
const unsigned long HOLD_TIME_MS = 1000;

// ---------- END OF MOTOR SETTINGS ----------


void setup()
{
  // Start USB serial communication with the computer.
  // I have no idea what baud rate is but apparently I have to set it up to be 115200.
  Serial.begin(115200);

  // Waits no longer than 100 milliseconds for the rest of a serial message.
  Serial.setTimeout(100);

  // Pause for one second to allow the controller to start.
  delay(1000);

  Serial.println("Starting two-axis optic controller...");

  // ---------- END OF SERIAL STARTUP ----------


  // Attempt to start communication with the Motor FeatherWing.
  if (!motorWing.begin())
  {
    Serial.println("ERROR: Motor FeatherWing not detected.");

    // Stop here permanently if the Motor FeatherWing is not detected.
    while (true)
    {
      delay(100);
    }
  }

  // ---------- END OF FEATHERWING CONNECTION CHECK ----------


  // Assign stepper port 1 to the windage motor.
  // Stepper port 1 uses the M1 and M2 terminal groups.
  windageMotor =
      motorWing.getStepper(STEPS_PER_REVOLUTION, 1);

  // Assign stepper port 2 to the elevation motor.
  // Stepper port 2 uses the M3 and M4 terminal groups.
  elevationMotor =
      motorWing.getStepper(STEPS_PER_REVOLUTION, 2);

  // ---------- END OF MOTOR PORT ASSIGNMENT ----------


  // Set the speed of both motors.
  windageMotor->setSpeed(MOTOR_SPEED_RPM);
  elevationMotor->setSpeed(MOTOR_SPEED_RPM);

  // ---------- END OF MOTOR SPEED SETUP ----------


  // Tell the computer that the controller is ready.
  Serial.println("READY");
  Serial.println("Enter: MOVE windageSteps elevationSteps");
  Serial.println("Example: MOVE -12 8");

  // ---------- END OF SETUP ----------
}


void loop()
{
  // Check whether the computer has sent any characters.
  if (Serial.available() > 0)
  {
    // Read characters until the program reaches a newline.
    String command = Serial.readStringUntil('\n');

    // Remove unwanted spaces and line-ending characters.
    command.trim();

    // Only process the message if it is not empty.
    if (command.length() > 0)
    {
      processCommand(command);
    }
  }

  // The program returns to the beginning of loop() and checks again.
}

// ---------- END OF MAIN LOOP ----------


void processCommand(const String &command)
{
  // Create containers for the two requested step amounts.
  long windageSteps;
  long elevationSteps;


  // Look for a command matching:
  //
  // MOVE windageSteps elevationSteps
  //
  // Example:
  //
  // MOVE -12 8
  //
  // The first number is stored in windageSteps.
  // The second number is stored in elevationSteps.
  int valuesRead = sscanf(
    command.c_str(),
    "MOVE %ld %ld",
    &windageSteps,
    &elevationSteps
  );

  // ---------- END OF COMMAND READING ----------


  // Confirm that the program successfully found both numbers.
  if (valuesRead != 2)
  {
    Serial.println(
      "ERROR: Use MOVE windageSteps elevationSteps"
    );

    // Exit this function without moving the motors.
    return;
  }

  // ---------- END OF COMMAND VALIDATION ----------


  // Report the interpreted command to the computer.
  Serial.print("Windage command: ");
  Serial.println(windageSteps);

  Serial.print("Elevation command: ");
  Serial.println(elevationSteps);

  // ---------- END OF COMMAND REPORT ----------


  // Move the windage motor first.
  moveMotor(windageMotor, windageSteps);

  // Move the elevation motor second.
  moveMotor(elevationMotor, elevationSteps);

  // ---------- END OF MOTOR MOVEMENT ----------


  // Hold: the coils are still powered after step(), so both motors keep holding
  // their position. Wait HOLD_TIME_MS so the turrets settle before letting go.
  Serial.println("HOLDING");
  delay(HOLD_TIME_MS);

  // ---------- END OF HOLD ----------


  // Remove power from both motor coils after the hold.
  // This reduces motor heating; the turret's own friction keeps it in place afterwards.
  windageMotor->release();
  elevationMotor->release();

  // Tell the computer that the complete movement is finished.
  Serial.println("DONE");

  // ---------- END OF COMMAND PROCESSING ----------
}


void moveMotor(Adafruit_StepperMotor *motor, long steps)
{
  // Do nothing if the requested movement is zero steps.
  if (steps == 0)
  {
    return;
  }

  // ---------- END OF ZERO-MOVEMENT CHECK ----------


  // Create a container for the selected direction.
  uint8_t direction;


  // Uses the sign of the number to select motor direction.
  if (steps > 0)
  {
    direction = FORWARD;
  }
  else
  {
    direction = BACKWARD;
  }

  // ---------- END OF DIRECTION SELECTION ----------


  // Convert a negative step amount into a positive quantity.
  //
  // Example:
  // -12 chooses BACKWARD above and becomes a step count of 12.
  unsigned long stepCount = labs(steps);

  // Move the selected motor.
  //
  // stepCount = how many steps to move
  // direction = FORWARD or BACKWARD
  // DOUBLE = energize both coils for more torque
  motor->step(stepCount, direction, DOUBLE);

  // ---------- END OF INDIVIDUAL MOTOR MOVEMENT ----------
}