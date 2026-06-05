# Info for getting started with the rig

## Connections
The arduinos are currently operating as:
1. Nano 1 - RPM PID controller, efficiency monitor [micro-usb->usb-A (to PC)]
2. ESP32 - Pressure readings [micro-usb connection, need a new cable, ideally long]
3. Nano 2 - Throttle adjustment [usb-c connection, need a new cable, ideally long]
[Both cables for 2 and 3 were mine so I have taken them back]#
The microphone has a usb-c->usb-c cable which is pretty long. 

## Things to watch out for
The Hall effect sensor is connected to the arduino via 3 cables. They are leftover from the previous year's design and I never got around to replacing them. They don't have a very good connection so sometimes needed to be taped/held in place.
The magnets aren't always picked up by the sensor (so you see a reading of half the RPM expected). Sometimes it works just if you stop it, push the fan a bit more in and hope.
The bit you screw the fan onto (which is semi-permanently on the motor) is only glued on. It might come un-done. 

## Useful bits
The fan uses a 2.5mm hex bolt for connection to the motor. The current bolt is screwed on to the motor without a fan attached.
The pre-existing fans are in 4 boxes labeled in groups of 20 (or so). The additional 3 inverse design ones are left separate. 