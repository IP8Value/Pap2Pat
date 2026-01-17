# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to methods and systems for estimating the relative position of fingertips, particularly in the context of dexterous manipulation. More specifically, the invention provides a method for quantifying the contribution of tactile and non-tactile inputs to the estimation of fingertip distance, which is crucial for tasks requiring precise hand movements and object manipulation.

## BACKGROUND

The central nervous system (CNS) plays a vital role in integrating sensory feedback and motor commands to control hand movements and object manipulation. Successful manipulation often involves the accurate estimation of the relative positions of the fingertips, which is essential for tasks such as grasping, holding, and manipulating objects. However, the CNS lacks dedicated sensory receptors for encoding the distance between limb endpoints, such as the fingertips. Therefore, the estimation of fingertip distance relies on the integration of various sensory and sensorimotor inputs, including tactile cues, proprioceptive inputs, and efference copy.

Tactile cues arise from mechanical deformation of the finger pads and provide information about the magnitude and direction of forces acting on the fingers. Proprioceptive inputs are triggered by changes in the length and tension of the muscle-tendon complex and joint angles, contributing to the perception of limb position. Efference copy, a copy of the motor command responsible for finger placement and contact forces, also plays a significant role in estimating the relative position of the fingertips.

Previous studies have explored the role of these inputs in the estimation of fingertip distance, but the relative contribution of tactile and non-tactile inputs remains unclear. Understanding this relationship is crucial for developing advanced haptic devices and robotic systems that can mimic human dexterity and improve the accuracy of finger position estimation in various applications, including virtual reality, teleoperation, and prosthetics.

## SUMMARY

The present invention addresses the need for a method and system to quantify the relative contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The invention provides a method for assessing the role of tactile and non-tactile inputs in the perception of digit relative position, which is essential for dexterous manipulation tasks.

The method includes the following steps:
1. **Digit Placement**: Placing the thumb and index finger in a collinear position on a sensorized handle or using wearable haptic devices.
2. **Force Application**: Applying or exerting normal and tangential forces on the thumb and index finger in various combinations, including both digits exerting forces in the same direction, opposite directions, and no forces.
3. **Sensing and Memorization**: Sensing and memorizing the perceived vertical distance between the thumb and index finger while the forces are applied.
4. **Relaxation**: Relaxing the hand for a short period.
5. **Matching**: Reproducing the previously sensed vertical distance between the thumb and index finger.
6. **Holding**: Maintaining the matched vertical distance for a specified period.

The method can be performed in three different experimental setups:
- **Experiment 1**: Both tactile and non-tactile inputs are available.
- **Experiment 2**: Only tactile inputs are available.
- **Experiment 3**: Only non-tactile inputs are available.

The invention also includes a system for implementing the method, comprising:
- **Sensorized Handle**: A custom-made handle equipped with force sensors to measure normal and tangential forces and the three-dimensional coordinates of the center of pressure (CoP) of the thumb and index finger.
- **Wearable Haptic Devices**: Devices that deliver precise and repeatable tactile stimulations on the thumb and index finger pads.
- **Haptic Devices**: Devices that apply forces to the intermediate phalanx of the index finger and the proximal phalanx of the thumb, allowing for the measurement of the three-dimensional position of the endpoints.

The invention further provides a method for analyzing the data collected from the experiments to quantify the contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The analysis includes:
- **Matching Errors**: Calculating the difference between the sensed and matched vertical distances.
- **Statistical Analysis**: Performing nonparametric tests to determine the significance of the matching errors.
- **Linear Combination Analysis**: Assuming that fingertip distance estimation relies on a linear combination of tactile and non-tactile inputs and calculating the relative contributions of each input.

The invention has numerous applications, including the development of advanced haptic devices, robotic systems, and prosthetics that can accurately estimate and control the relative position of the fingertips, thereby enhancing dexterity and precision in various tasks.

## DETAILED DESCRIPTION

### Introduction

The present invention is directed to methods and systems for estimating the relative position of fingertips, particularly in the context of dexterous manipulation. The invention provides a method for quantifying the contribution of tactile and non-tactile inputs to the estimation of fingertip distance, which is crucial for tasks requiring precise hand movements and object manipulation.

### Technical Field

The technical field of the invention encompasses biomechanics, sensorimotor integration, and haptic technology. The invention is particularly relevant to the development of advanced haptic devices, robotic systems, and prosthetics that can mimic human dexterity and improve the accuracy of finger position estimation.

### Background

The central nervous system (CNS) integrates sensory feedback and motor commands to control hand movements and object manipulation. Successful manipulation often involves the accurate estimation of the relative positions of the fingertips, which is essential for tasks such as grasping, holding, and manipulating objects. However, the CNS lacks dedicated sensory receptors for encoding the distance between limb endpoints, such as the fingertips. Therefore, the estimation of fingertip distance relies on the integration of various sensory and sensorimotor inputs, including tactile cues, proprioceptive inputs, and efference copy.

Tactile cues arise from mechanical deformation of the finger pads and provide information about the magnitude and direction of forces acting on the fingers. Proprioceptive inputs are triggered by changes in the length and tension of the muscle-tendon complex and joint angles, contributing to the perception of limb position. Efference copy, a copy of the motor command responsible for finger placement and contact forces, also plays a significant role in estimating the relative position of the fingertips.

Previous studies have explored the role of these inputs in the estimation of fingertip distance, but the relative contribution of tactile and non-tactile inputs remains unclear. Understanding this relationship is crucial for developing advanced haptic devices and robotic systems that can mimic human dexterity and improve the accuracy of finger position estimation in various applications, including virtual reality, teleoperation, and prosthetics.

### Summary of the Invention

The present invention addresses the need for a method and system to quantify the relative contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The invention provides a method for assessing the role of tactile and non-tactile inputs in the perception of digit relative position, which is essential for dexterous manipulation tasks.

The method includes the following steps:
1. **Digit Placement**: Placing the thumb and index finger in a collinear position on a sensorized handle or using wearable haptic devices.
2. **Force Application**: Applying or exerting normal and tangential forces on the thumb and index finger in various combinations, including both digits exerting forces in the same direction, opposite directions, and no forces.
3. **Sensing and Memorization**: Sensing and memorizing the perceived vertical distance between the thumb and index finger while the forces are applied.
4. **Relaxation**: Relaxing the hand for a short period.
5. **Matching**: Reproducing the previously sensed vertical distance between the thumb and index finger.
6. **Holding**: Maintaining the matched vertical distance for a specified period.

The method can be performed in three different experimental setups:
- **Experiment 1**: Both tactile and non-tactile inputs are available.
- **Experiment 2**: Only tactile inputs are available.
- **Experiment 3**: Only non-tactile inputs are available.

The invention also includes a system for implementing the method, comprising:
- **Sensorized Handle**: A custom-made handle equipped with force sensors to measure normal and tangential forces and the three-dimensional coordinates of the center of pressure (CoP) of the thumb and index finger.
- **Wearable Haptic Devices**: Devices that deliver precise and repeatable tactile stimulations on the thumb and index finger pads.
- **Haptic Devices**: Devices that apply forces to the intermediate phalanx of the index finger and the proximal phalanx of the thumb, allowing for the measurement of the three-dimensional position of the endpoints.

The invention further provides a method for analyzing the data collected from the experiments to quantify the contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The analysis includes:
- **Matching Errors**: Calculating the difference between the sensed and matched vertical distances.
- **Statistical Analysis**: Performing nonparametric tests to determine the significance of the matching errors.
- **Linear Combination Analysis**: Assuming that fingertip distance estimation relies on a linear combination of tactile and non-tactile inputs and calculating the relative contributions of each input.

### Detailed Description

#### Method for Estimating Fingertip Distance

The method of the present invention involves a series of steps to quantify the contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The method can be performed in three different experimental setups to isolate the effects of tactile and non-tactile inputs.

##### Step 1: Digit Placement

In this step, the thumb and index finger are placed in a collinear position. For Experiment 1, the digits are placed on a sensorized handle. For Experiment 2, the digits are attached to wearable haptic devices. For Experiment 3, the digits are attached to haptic devices that apply forces to the intermediate phalanx of the index finger and the proximal phalanx of the thumb.

##### Step 2: Force Application

Normal and tangential forces are applied or exerted on the thumb and index finger in various combinations. The force combinations include:
- Both digits exerting tangential forces in the same direction (TUP-IUP, TDN-IDN).
- Both digits exerting tangential forces in opposite directions (TUP-IDN, TDN-IUP).
- No forces (Null).
- Only normal forces (Fn only).

The forces are applied or exerted within specified ranges to ensure consistency across trials.

##### Step 3: Sensing and Memorization

During this step, the subject senses and memorizes the perceived vertical distance between the thumb and index finger while the forces are applied. The subject is instructed to focus on the sensation of the forces and the perceived distance between the digits.

##### Step 4: Relaxation

After the sensing and memorization phase, the subject relaxes the hand for a short period. This relaxation phase helps to reset the sensory and motor systems before the matching phase.

##### Step 5: Matching

The subject is then asked to reproduce the previously sensed vertical distance between the thumb and index finger. The subject is given a specified time to match the distance, and the matched distance is recorded.

##### Step 6: Holding

Once the subject reports having reached the matched distance, the subject is instructed to hold the digit configuration for a specified period. This holding phase ensures that the matched distance is stable and accurate.

#### System for Implementing the Method

The invention also includes a system for implementing the method, comprising the following components:

##### Sensorized Handle

A custom-made handle equipped with force sensors to measure normal and tangential forces and the three-dimensional coordinates of the center of pressure (CoP) of the thumb and index finger. The handle is secured to a table to allow subjects to apply forces without rotating or lifting the object.

##### Wearable Haptic Devices

Devices that deliver precise and repeatable tactile stimulations on the thumb and index finger pads. The haptic devices generate normal and shear (tangential) forces by providing both compression and skin stretch on the fingertips.

##### Haptic Devices

Devices that apply forces to the intermediate phalanx of the index finger and the proximal phalanx of the thumb. The haptic devices are equipped with internal encoders to measure the three-dimensional position of the endpoints attached to each digit.

#### Data Analysis

The data collected from the experiments are analyzed to quantify the contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The analysis includes the following steps:

##### Matching Errors

The difference between the sensed and matched vertical distances (dy) is calculated for each trial. The matching errors are defined as the dy measured during the holding phase minus the dy observed during the sensing phase.

##### Statistical Analysis

Nonparametric tests, such as the Wilcoxon signed-rank test, are performed to determine the significance of the matching errors. The tests are used to compare the matching errors across different force conditions and experimental setups.

##### Linear Combination Analysis

Assuming that fingertip distance estimation relies on a linear combination of tactile and non-tactile inputs, the relative contributions of each input are calculated. The analysis involves comparing the matching errors obtained when both tactile and non-tactile inputs are available (Experiment 1) with the sum of the matching errors obtained when only tactile inputs (Experiment 2) or only non-tactile inputs (Experiment 3) are available.

### Conclusion

The present invention provides a method and system for quantifying the relative contribution of tactile and non-tactile inputs to the estimation of fingertip distance. The method involves a series of steps to assess the role of tactile and non-tactile inputs in the perception of digit relative position, which is essential for dexterous manipulation tasks. The invention has numerous applications, including the development of advanced haptic devices, robotic systems, and prosthetics that can accurately estimate and control the relative position of the fingertips, thereby enhancing dexterity and precision in various tasks.