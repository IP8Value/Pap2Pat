# DESCRIPTION

## FIELD OF THE INVENTION

- relate to wheelchairs with inertial sensors

The present invention relates to powered and manual wheelchair systems equipped with inertial measurement units for the real-time detection of destabilizing motion events and the automated activation of neuromuscular stimulation to enhance trunk and pelvic stability in individuals with spinal cord injuries. Specifically, the invention encompasses a wheelchair-integrated control system that utilizes tri-axial accelerometers and gyroscopes to monitor linear and angular motion parameters of the wheelchair frame, identifies potentially hazardous dynamic events such as collisions and sharp turns based on predefined threshold criteria, and triggers precisely timed neuromuscular stimulation to activate paralyzed musculature responsible for postural control. This system is designed to operate autonomously without requiring user intervention, thereby providing immediate biomechanical support during sudden destabilizations that would otherwise result in loss of seated posture, falls, or injury. The integration of inertial sensing with neuromuscular actuation represents a novel approach to improving mobility safety and functional independence for wheelchair users with impaired trunk control.

## BACKGROUND

- describe problem of falls for wheelchair users
- describe need for trunk stability and efficient propulsion

Individuals living with spinal cord injuries frequently rely on wheelchairs for daily mobility, yet they remain at substantially elevated risk for falls and tipping incidents during routine activities such as propulsion, turning, or navigating uneven terrain. These events are not merely inconvenient—they often result in severe secondary injuries including fractures, soft tissue trauma, and prolonged hospitalization, with associated medical costs ranging from twenty-five to seventy-five thousand dollars per incident. The underlying cause of these accidents is the loss of voluntary control over trunk and hip musculature, which compromises the user’s ability to maintain upright posture in response to dynamic perturbations. Conventional strategies to mitigate this risk, such as seat belts, rigid back supports, or customized cushioning systems, impose significant functional limitations by restricting voluntary movement, interfering with activities of daily living, and contributing to pressure ulcers, skin breakdown, and psychological distress. Many users avoid these devices due to perceived stigma, discomfort, or reduced autonomy. Moreover, power wheelchairs with tilting mechanisms, while offering greater postural adjustment, do not restore active muscular control and often fail to respond in real time to unanticipated disturbances. There exists a critical unmet need for an intelligent, non-restrictive, and physiologically grounded solution that actively restores postural stability at the moment of destabilization, mimicking the natural neuromuscular reflexes lost due to injury.

## SUMMARY

- introduce wheelchair system with neural stimulation
- describe sensor measuring motion parameter
- describe neural stimulating electrodes
- describe controller receiving signals and activating muscles
- describe controller mounted on wheelchair or user
- describe linear motion parameter
- describe angular motion parameter
- describe electrodes attached to muscles
- describe inertial measurement unit
- describe gyroscope and accelerometer
- describe radio frequency transmitter
- describe controller with radio frequency receiver and microprocessor
- describe seat belt controlled by motor
- describe brake controlled by actuator
- describe distress indicator
- introduce method of providing neural stimulation
- measure motion parameter
- generate signal indicative of motion parameter
- evaluate signal
- activate muscle in response to signal
- select muscles based on motion parameter
- describe method of providing neural stimulation during collision
- monitor linear acceleration
- calculate moving root mean square
- compare against threshold values
- apply neuromuscular stimulation
- describe method of applying restraint
- describe system for providing assistance to user

The invention discloses a wheelchair system incorporating an inertial measurement unit that continuously monitors linear and angular motion parameters of the wheelchair frame, a controller configured to evaluate these parameters in real time, and a neuromuscular stimulation subsystem that activates specific muscle groups in response to detected destabilizing events. The inertial measurement unit comprises a tri-axial accelerometer and a tri-axial gyroscope, each sampling data at a rate sufficient to capture transient dynamic events with millisecond precision. These sensors are rigidly mounted to the wheelchair frame to ensure accurate representation of the system’s motion, and their output signals are transmitted wirelessly via a radio frequency transmitter to a controller unit mounted either on the wheelchair frame or on the user’s torso. The controller contains a radio frequency receiver, a microprocessor, and embedded software algorithms that compute a moving root mean square of the measured acceleration and angular velocity to detect abrupt changes indicative of collisions or sharp turns. Upon exceeding predefined, subject-specific threshold values, the controller generates a stimulation trigger signal that activates implanted or surface electrodes attached to key postural muscles, including the erector spinae, gluteus maximus, quadratus lumborum, hamstrings, and adductors, to generate torque that resists forward flexion or lateral displacement. In addition to neuromuscular stimulation, the system may optionally activate a motorized seat belt to apply controlled restraint or an actuator-driven brake to decelerate the wheelchair during high-risk events. A distress indicator, such as an audible alarm or visual signal, may be triggered to alert caregivers in the event of prolonged instability or system malfunction. The method of providing neural stimulation involves continuously measuring the motion parameter of the wheelchair, generating an electrical signal representative of that parameter, evaluating the signal against a dynamic threshold derived from baseline motion patterns, and activating at least one muscle group in direct response to the evaluation outcome. The selection of muscles is determined by the nature of the detected event—trunk and hip extensors are activated during forward collisions to counteract flexion, while contralateral quadratus lumborum and hip extensors are stimulated during turns to resist lateral lean. During collision events, the system monitors linear acceleration along the anterior-posterior axis, calculates a moving root mean square over a sliding time window, and compares it to a threshold calibrated to the individual’s anthropometry and wheelchair dynamics. Neuromuscular stimulation is applied within 150 milliseconds of threshold exceedance to coincide with the onset of postural instability. Similarly, during turning events, angular velocity along the superior-inferior axis is evaluated, and stimulation is applied to muscles that stabilize the pelvis and lumbar spine against centrifugal forces. The system further includes a method of applying restraint by evaluating the same motion parameters and activating a motorized seat belt when instability exceeds a secondary, higher threshold. The overall system is designed to provide autonomous, real-time assistance to the user, enhancing seated stability, reducing fall risk, and promoting functional independence without external physical constraints.

## DETAILED DESCRIPTION

- describe wheelchair system 10
- introduce motion sensor 14
- detail sensor 14 measurements
- describe controller 18
- introduce microprocessor 24
- explain software evaluation of signals
- describe neural stimulating electrodes 26
- detail electrode attachment methods
- explain system 10 goal
- describe collision event response
- introduce linear motion parameters
- detail muscle activation for collisions
- describe sharp turn event response
- introduce angular motion parameters
- detail muscle activation for sharp turns
- introduce method of providing neural stimulation
- measure motion parameter of wheelchair
- generate signal indicative of motion parameter
- evaluate signal
- activate at least one muscle in response to signal
- describe collision event evaluation
- describe turning event evaluation
- introduce FIG. 4
- describe collision detection algorithm
- introduce FIG. 5
- describe method of providing user restraint
- measure motion parameter of wheelchair
- generate signal indicative of motion parameter
- evaluate signal
- activate at least one user restraint in response to signal
- introduce system 66
- describe sensor 74 measurements
- introduce method of providing assistance to user 70
- describe neural stimulation application and removal
- describe algorithm for detecting push completion and recovery

The wheelchair system 10 comprises a standard manual or powered wheelchair frame integrated with a motion sensor 14, a controller 18, and a set of neural stimulating electrodes 26. The motion sensor 14 is a compact inertial measurement unit mounted centrally on the rear crossbar of the wheelchair, containing a tri-axial accelerometer and a tri-axial gyroscope that sample linear acceleration and angular velocity at 100 Hz. These measurements are transmitted wirelessly via a radio frequency transmitter to the controller 18, which may be affixed to the wheelchair frame or worn on the user’s torso. The controller 18 houses a microprocessor 24 that executes real-time signal processing algorithms to evaluate the incoming motion data. The software evaluates the magnitude and rate of change of linear motion parameters, particularly anterior-posterior acceleration, and angular motion parameters, particularly superior-inferior angular velocity, to distinguish between normal propulsion, sharp turns, and collision events. Neural stimulating electrodes 26 are either surgically implanted intramuscularly or placed transcutaneously over the motor points of the erector spinae, gluteus maximus, quadratus lumborum, hamstrings, and posterior adductors, and are connected to an implanted or external pulse generator capable of delivering charge-balanced, current-controlled stimulation pulses with programmable amplitude, duration, and frequency. The primary goal of system 10 is to restore postural stability during destabilizing events by activating the paralyzed musculature responsible for trunk extension and pelvic control, thereby preventing forward flexion or lateral lean that leads to falls. During a collision event, the system detects a sharp spike in anterior-posterior linear acceleration exceeding a subject-specific threshold derived from baseline calibration trials, and immediately activates the lumbar erector spinae and hip extensors to generate an opposing extension torque that resists trunk flexion. For sharp turns, the system identifies a rapid change in superior-inferior angular velocity, and activates the quadratus lumborum on the inside of the turn and the hip extensor on the outside to stabilize the pelvis against centrifugal forces. The method of providing neural stimulation involves continuously measuring the motion parameter of the wheelchair, generating a digitized signal representative of that parameter, evaluating the signal using a moving root mean square algorithm over a 50-millisecond window, and comparing it to a threshold value calibrated to the individual’s body mass, center of gravity, and wheelchair dynamics. If the evaluated signal exceeds the threshold, the controller activates the corresponding muscle group within 100 to 150 milliseconds. Collision event evaluation is performed by analyzing the peak and rate of change of linear acceleration, while turning event evaluation relies on the magnitude and sign of angular velocity to distinguish left from right turns. FIG. 4 illustrates the collision detection algorithm, which triggers stimulation only when the moving root mean square of anterior-posterior acceleration exceeds a threshold defined as the mean peak acceleration minus two standard deviations from calibration trials. FIG. 5 describes the method of providing user restraint, wherein if the motion parameter exceeds a second, higher threshold indicating imminent tipping, the controller activates a motorized seat belt to apply gentle, controlled tension across the torso. System 66 represents an alternative embodiment wherein sensor 74, mounted on the user’s sternum, measures trunk angle and angular velocity directly, enabling more precise detection of postural deviation. The method of providing assistance to user 70 involves the coordinated application of stimulation and restraint based on the severity and type of motion event detected. Neural stimulation is applied only during the destabilizing event and automatically ceases once the motion parameter returns to baseline for a sustained period, ensuring that stimulation does not interfere with voluntary movement during stable conditions. An algorithm for detecting push completion and recovery monitors the cessation of propulsion-related acceleration patterns and confirms return to upright posture before deactivating stimulation, preventing premature termination or prolonged activation. The system is designed to operate silently and unobtrusively, enhancing safety without compromising user autonomy or dignity.