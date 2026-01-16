# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a system and method for maintaining trunk stability in manual wheelchair users with spinal cord injuries (SCI). More specifically, the invention involves a threshold-based system that utilizes inertial measurements to detect destabilizing events such as collisions and sharp turns, and provides an appropriate stimulated response of the hip and trunk muscles through implanted pulse generators (IPGs) or surface stimulation to enhance recovery of a stable seated posture.

## BACKGROUND

Spinal cord injuries (SCI) affect approximately 285,000 people in the United States, with about two-thirds of these individuals relying on scooters or wheelchairs for daily mobility. Among the top functions that individuals with SCI desire to improve or restore is trunk stability. Loss of trunk control leads to increased postural sway during sitting, making it difficult to maintain an erect posture. Minor disturbances can cause a loss of balance, leading to falls and injuries. Tips and falls are the leading cause of injury for wheelchair users, accounting for two-thirds of the more than 100,000 wheelchair-related injuries annually that require emergency department treatment. These injuries can result in fractures, lacerations, contusions, and abrasions, and even death. Rehabilitation for such injuries often requires extended hospital stays, leading to reduced strength and increased risk of blood clots. The cost of treatment for wheelchair-related falls typically ranges from $25,000 to $75,000.

Current methods to maintain trunk stability in manual wheelchair users with SCI include seat belts, seat cushioning systems, increased seat dump, and support structures. These methods have several disadvantages, including reduced work volumes, impaired ability to reach and manipulate objects, and increased risk of pressure ulcers, skin tears, and lowered self-esteem. Non-compliance with these strategies is high due to the restrictions they impose on voluntary motion. Power wheelchairs offer more options for specialized seating systems and can provide power tilting and reclining, but they also restrict voluntary motion and are often seen as an indication of greater disability.

Functional neuromuscular stimulation (FNS) has been used to restore and maintain trunk stability by significantly increasing the trunk extension moment and multidirectional trunk stiffness. When the trunk flexor and extensor muscles are activated with neural stimulation, individuals show physical and psychological improvements in strength and stability, particularly in response to unexpected, destabilizing forces. Continuous activation of the otherwise paralyzed hip and trunk muscles with FNS can maintain trunk stability during anteriorly-directed forces and restore upright sitting from forward flexed or side-leaning positions. However, automatically modulating stimulation at the onset of destabilizing events during manual wheelchair propulsion to minimize the effects of unanticipated perturbations or restore seated posture with neural stimulation has not been extensively studied in individuals with SCI under real-world conditions.

To employ neural stimulation in response to potentially destabilizing events, a method of accurately predicting or detecting such situations must be identified. Inertial measurement units (IMUs) have been utilized to monitor trunk angle in static environments, determine phases of the manual wheelchair stroke cycle, and classify physical activities of daily living. Prior research has combined IMUs and machine learning to differentiate destabilizing conditions. The present invention utilizes a simple threshold-based method to detect and classify destabilizing events based on the inertial signature of the wheelchair to automatically trigger the appropriate muscle stimulation and determine its impact on seated posture.

The static and dynamic tipping stability of the wheelchair-user system has been examined in modeling studies. These studies inform the design of wheelchairs for safe operation and define their inherent mechanical limits of stability, but do not react to potentially destabilizing events or assist wheelchair users in maintaining and/or regaining stable seated postures when they occur. The present invention addresses this gap by providing a real-time system that detects destabilizing events and applies neural stimulation to enhance trunk stability.

## SUMMARY

The present invention provides a threshold-based system for maintaining trunk stability in manual wheelchair users with spinal cord injuries (SCI). The system includes an inertial measurement unit (IMU) attached to the wheelchair to monitor the wheelchair's inertial signature, a processor to analyze the inertial data and detect destabilizing events such as collisions and sharp turns, and a stimulation device to deliver functional neuromuscular stimulation (FNS) to the hip and trunk muscles when a destabilizing event is detected.

The system is designed to detect destabilizing events in real-time and apply FNS to the appropriate muscle groups to maintain or restore a stable seated posture. The IMU monitors the anterior-posterior (AP) acceleration to detect collisions and the superior-inferior (SI) angular velocity to detect turns. Simple threshold-based algorithms are used to determine when a destabilizing event occurs. For collisions, the system delivers maximal stimulation to the trunk (lumbar erector spinae) and hip (gluteus maximus, posterior adductor, and/or hamstrings) extensor muscles to resist forward flexion and assist return to an upright sitting position. For turns, the system activates the quadratus lumborum on the inside of the turn and the hip extensor on the outside of the turn to resist lateral displacement of the trunk and pelvis.

The system has been tested in a laboratory setting with four participants with SCI. The classifier accurately detected 93% of the trials for collisions and 93% for right turns, with an average detection delay of 88 ms for collisions and 342 ms for turns. The application of FNS significantly decreased the maximum AP trunk angle during collisions for two of the three subjects tested, and improved the return time to an erect posture. However, the delay in detection for turns negatively affected the ability of the system to return the trunk to an erect position from a lateral bend. Future improvements will focus on reducing the detection delay and exploring other sensor modalities to predict destabilizing events before they occur.

The invention offers a promising solution to reduce the risk of injurious falls and improve the safety and independence of manual wheelchair users with SCI. By providing real-time, automated assistance in maintaining trunk stability, the system enhances the quality of life for individuals with SCI and reduces the burden of care associated with wheelchair-related injuries.

## DETAILED DESCRIPTION

### System Overview

The present invention is a threshold-based system for maintaining trunk stability in manual wheelchair users with spinal cord injuries (SCI). The system comprises an inertial measurement unit (IMU) attached to the wheelchair, a processor to analyze the inertial data, and a stimulation device to deliver functional neuromuscular stimulation (FNS) to the hip and trunk muscles. The system is designed to detect destabilizing events such as collisions and sharp turns in real-time and apply FNS to the appropriate muscle groups to maintain or restore a stable seated posture.

### Components of the System

#### Inertial Measurement Unit (IMU)

The IMU is a critical component of the system, responsible for monitoring the inertial signature of the wheelchair. The IMU contains a tri-axial accelerometer and a tri-axial gyroscope to measure the linear acceleration and angular velocity of the wheelchair, respectively. The IMU is placed on the center of the rear crossbar of the wheelchair to ensure accurate and reliable measurements. The IMU samples the tri-axial acceleration and angular velocity at a frequency of 100 Hz.

#### Processor

The processor is responsible for analyzing the inertial data from the IMU and detecting destabilizing events. The processor runs threshold-based algorithms to determine when a destabilizing event occurs. For collisions, the processor monitors the anterior-posterior (AP) acceleration and compares it to a subject-specific threshold. For turns, the processor monitors the superior-inferior (SI) angular velocity and compares it to a subject-specific threshold. The thresholds are determined based on preliminary calibration trials and are specific to each participant.

#### Stimulation Device

The stimulation device delivers functional neuromuscular stimulation (FNS) to the hip and trunk muscles. The stimulation device can be either an implanted pulse generator (IPG) or a surface stimulator, depending on the participant's preference and medical condition. The IPG delivers asymmetrical charge-balanced current-controlled stimulus waveforms with pulse amplitudes (0 to 20 mA) selectable for each channel and variable pulse durations (0 to 250 μsec) and frequencies (0 to 20 Hz) set on a pulse-by-pulse basis. Surface stimulators can also be used to deliver FNS to the appropriate muscle groups.

### Detection Algorithms

#### Collision Detection Algorithm

The collision detection algorithm monitors the AP acceleration of the wheelchair. When the AP acceleration exceeds the subject-specific threshold, the algorithm triggers the stimulation device to deliver maximal stimulation to the trunk (lumbar erector spinae) and hip (gluteus maximus, posterior adductor, and/or hamstrings) extensor muscles. The threshold for collisions is calculated as the mean peak of the absolute value of AP acceleration minus two standard deviations from the participant's own 20 calibration trials.

#### Turn Detection Algorithm

The turn detection algorithm monitors the SI angular velocity of the wheelchair. When the SI angular velocity exceeds the subject-specific threshold, the algorithm triggers the stimulation device to activate the quadratus lumborum on the inside of the turn and the hip extensor on the outside of the turn. The threshold for turns is calculated as the mean peak of the absolute value of the SI angular velocity minus two standard deviations from the participant's own 20 calibration trials. Right and left turn thresholds are equal in magnitude but opposite in sign, with a positive threshold detecting a left turn and a negative threshold detecting a right turn.

### Experimental Setup

To design an appropriate and safe experimental setup, mathematical models of the wheelchair-user system were utilized to simulate collision and turning events. The critical velocity for tipping of the wheelchair-user system was calculated using the equations derived by Li et al. and Cooper et al. The critical velocity for tipping due to a collision was expected to occur at 1.6 m/s, and a slightly lower velocity of 1.5 m/s was chosen for experimental sessions to ensure subject safety. The critical velocity for rollover during a destabilizing turn was calculated based on the initial velocity and radius of the turn.

A 2-meter ramp with a 5-degree incline was constructed for the experimental setup. A guidance track was mounted on top of the ramp, ending with a 90-degree turn of 25-inch (63.5 cm) radius. Roller bearings were installed beneath the frame of a standard wheelchair to guide descent and turning and achieve a consistent velocity at impact or time of turn. A barrier was erected 2 meters after the turn to suddenly stop the wheelchair and rider, simulating a collision.

### Participant Testing

Four participants with low thoracic or high cervical spinal cord injuries participated in the study. Each participant had received a surgically implanted neuromuscular stimulator to activate the knee, hip, and trunk extensor muscles with implanted intramuscular or nerve cuff electrodes. One participant utilized surface stimulation due to technical difficulties with the implanted system.

Each participant went through 20 calibration trials to determine threshold values of AP acceleration and SI angular velocity indicative of collisions and turns. The real-time decision algorithms for collisions and turns were created in Simulink, which received the IMU signals from the wheelchair, compared them to the subject-specific thresholds, and delivered stimulation to the muscle groups at the levels customized for each subject when the threshold was exceeded.

### Results

The classifier accurately detected 93% of the trials for collisions and 93% for right turns, with an average detection delay of 88 ms for collisions and 342 ms for turns. The application of FNS significantly decreased the maximum AP trunk angle during collisions for two of the three subjects tested, and improved the return time to an erect posture. However, the delay in detection for turns negatively affected the ability of the system to return the trunk to an erect position from a lateral bend.

### Conclusion

The present invention provides a threshold-based system for maintaining trunk stability in manual wheelchair users with spinal cord injuries (SCI). The system utilizes an IMU to monitor the inertial signature of the wheelchair, a processor to analyze the inertial data and detect destabilizing events, and a stimulation device to deliver FNS to the appropriate muscle groups. The system has been tested in a laboratory setting and has shown promise in reducing the risk of injurious falls and improving the safety and independence of manual wheelchair users with SCI. Future improvements will focus on reducing the detection delay and exploring other sensor modalities to predict destabilizing events before they occur.