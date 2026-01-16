# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of upper limb prosthetic design, specifically to a novel multi-modal prosthetic fingertip sensor that integrates an infrared (IR) proximity sensor and a barometric pressure sensor. The sensor is designed to provide rich sensory information, including proximity, contact, force, spatial location, and angular orientation of applied forces, which is essential for effective grasping and manipulation in prosthetic hands.

## BACKGROUND

The field of upper limb prosthetic design has made significant strides in recent years, with multi-functional prosthetic hands capable of producing closing speeds of less than 1 second and forces as high as 100 Newtons. Advanced feedforward control algorithms have further enhanced the intuitiveness of these devices, making them more user-friendly. However, a critical deficiency remains: the lack of feedback systems to provide sensory restoration for users. Current prosthetic hands are essentially numb, meaning users are not directly aware of the tactile interactions between the hand and the environment. Somatosensory information is crucial for effective grasping and manipulation, and a stable and precise neural interface is necessary to provide physiologically appropriate sensory feedback.

Several technologies have been explored to address this issue, including the use of the peripheral nervous system as a portal to afferent pathways. Techniques such as the transversal multichannel intrafascicular electrode and the Utah Slant Electrode Array have shown promise in human trials, demonstrating the ability to elicit sensory percepts and control virtual robotic hands. The flat interface nerve electrode (FINE) has also been successful in providing subjects with the ability to discriminate and match intensity of sensation and recognize changes in magnitude of stimulation. However, these systems often rely on simple fingertip sensors, such as force-sensitive resistors, which have limitations in detecting off-centered and non-normal loads.

There is a need for a more advanced and comprehensive sensor that can provide rich, multi-modal sensory information to enhance the functionality of upper limb prosthetics. Such a sensor should be capable of detecting proximity, contact, force, spatial location, and angular orientation of applied forces, ensuring reliable and repeatable sensory feedback.

## SUMMARY

The present invention addresses the aforementioned needs by providing a novel multi-modal prosthetic fingertip sensor. The sensor integrates an infrared (IR) proximity sensor and a barometric pressure sensor within a durable elastomer overmold. The IR sensor detects proximity and contact, while the barometric sensor measures linear force readings from 0 to 50 Newtons. The sensor is designed to classify five spatial locations and three angles of incidence, ensuring that sensory restoration can take place reliably and repeatedly even when external forces are not centered on the fingertip surface.

The multi-modal fingertip sensor offers several advantages:
1. **Proximity Sensing**: The IR sensor detects the presence of objects before contact, enabling pre-grasp alignment and reactive grasping.
2. **Contact Detection**: The IR sensor can detect contact forces close to 0 Newtons, providing a reliable zero-force contact signal.
3. **Force Measurement**: The barometric sensor measures linear force readings from 0 to 50 Newtons, ensuring accurate force feedback.
4. **Spatial and Angular Localization**: The sensor can classify the spatial location and angular orientation of applied forces, enhancing the precision of sensory feedback.
5. **Robust Design**: The sensor is overmolded with a liquid silicon polymer, providing a durable and mechanically robust contact surface.

The sensor is designed to communicate with a prosthetic hand controller via standard I²C communication, ensuring stable and reliable data transmission. The multi-modal sensory information can be fused using Gaussian process (GP) regression and machine learning techniques, such as support vector machines (SVM) and convolutional neural networks (CNN), to derive additional information not available from each sensor independently.

## DETAILED DESCRIPTION

### Sensor Design

The multi-modal prosthetic fingertip sensor (PCF sensor) is a combination of two integrated circuit (IC) sensors: a microelectromechanical system (MEMS)-based barometric pressure sensor (MS5637-02BA03) and an IR proximity sensor (VCNL4010). These sensors are arranged on a custom printed circuit board (PCB) along the mid-line of the fingertip. The PCB is designed to fit within a cavity in the prosthetic finger, which is prototyped using standard three-dimensional (3D) printing techniques. A liquid silicon polymer (Dragon Skin 10) is poured into a mold with the fingertip sensor to create a robust contact surface. The elastomer is chosen for its low viscosity when pouring into molds and mechanical robustness post-curing.

An additional PCB is designed to multiplex the sensor's I²C signals for access by a host computer. The Arduino microcontroller firmware performs the proximity calculation for the IR sensor and the calibration and temperature compensation for the barometric sensor. The firmware then sends the calibrated proximity and pressure data to a laptop computer through a serial USB interface. A custom LabVIEW program is used to visualize the real-time signals and store the data for offline processing and analysis.

### Multi-Modal Signals

The PCF sensor provides multiple sensing modalities, including proximity, contact, and force. When a small piece of cotton is dropped onto the sensor, the contact detection is visible as a small peak in the IR sensor signal. As the cotton is gently pressed against the sensor, the change in force is picked up by the barometric sensor in a nearly linear manner. The barometric sensor provides a proportional measurement of the pressure within the fingertip sensor, which is stable across all loads (tested up to 50 Newtons).

### Experimental Characterization

To characterize the performance of the PCF sensor, multiple fingertip sensors were fabricated and tested using an Instron material testing machine. The machine applied calibrated loads at various spatial positions and angles of incidence on the fingertip. The loads were applied using a probe with a flat circular tip (15 mm diameter) and monitored using a 250 Newton load cell. The MTS machine applied prescribed loads ranging from 1 to 50 Newtons at a rate of 1 mm/s with a sampling rate of 16 Hz. Custom 3D-printed "pillows" were used to locate the fingertip sensor in the prescribed spatial and angular orientations with respect to the probe.

The spatial dataset measured contact events at the center, 2.5 mm distally, 2.5 mm proximally, 2.5 mm medially, and 2.5 mm laterally. The angular orientation dataset measured contact events at angles of 0, 20, and -20 degrees. These conditions were chosen to span the entire range of detectable volume of the fingertip sensor. In each condition, a sequence of 10 contact events at each maximum load took place, separated by a 1-second delay. The maximum loads tested were 1, 5, 30, and 50 Newtons.

### Data Analysis

The calibration of multi-modal analog data to actual force is non-trivial due to the varying nature of the signals based on the position and orientation of contact. Gaussian process (GP) regression is used to map the raw barometer and IR readings to true force. The GP approach is a non-parametric method that finds a distribution over possible functions consistent with the observed data. The combined signals from the fingertip vary based on the position and orientation of contact, making it challenging to estimate a single function with a fixed number of parameters. The GP method is computationally affordable on small datasets and has a well-tuned smoothing property.

The problem of localizing external loads on the finger is framed into two separate supervised learning problems: (1) classification of the spatial location of load and (2) classification of the angle of incidence of the force. Support vector machine (SVM) and convolutional neural network (CNN) classifiers are trained for each of these tasks. The SVM classifier uses features extracted from the barometer and IR sensor signals, including the ratio of IR and barometer values, maximum and minimum force values, and the use of a polynomial kernel with a penalty factor of C = 1. The CNN classifier uses raw data and consists of two 2D convolution layers followed by a flattened layer and a dense output layer with softmax activation.

### Results

The PCF sensors were characterized and shown to be capable of detecting five spatial locations and three angles of incidence. The IR sensor is sensitive to contact forces close to 0 Newtons, while the barometric sensor provides linear force readings from 0 to 50 Newtons. The responses of the barometer and IR proximity sensor to applied force at any spatial location on the finger are distinctively different. The barometer shows a linear behavior to applied force after its minimum range has been crossed, whereas the IR sensor shows a non-linear behavior while being sensitive in a range below that of the barometer. The signals are repeatable over a fixed location on the finger over multiple days, but vary in an irregular manner across different positions on the finger.

### Sensor Fusion

The raw signal data are preprocessed through a low-pass filter to remove unwanted noise. Individual loading and unloading curves are segmented out by locating the peaks from each contact. The data are then concatenated and normalized. The Gaussian kernel used is a radial basis function (RBF) kernel implemented in the scikit-learn library. The kernel parameters are experimentally calculated to minimize error in the 3D plot. The root mean square error (RMSE) and R² score are used to determine the accuracy of the fit.

### Force Localization

The interaction between the elastomer mold and the sensors is difficult to model due to the non-linear nature of the geometry and loading conditions. To localize impact on the sensor, the problem is broken down into two smaller subproblems: identifying the angular direction of probing and the spatial location of impact. SVM and CNN classifiers are trained for each subproblem. The SVM classifier uses features extracted from the barometer and IR sensor signals, including the ratio of IR and barometer values, maximum and minimum force values, and the use of a polynomial kernel with a penalty factor of C = 1. The CNN classifier uses raw data and consists of two 2D convolution layers followed by a flattened layer and a dense output layer with softmax activation.

### Discussion

The multi-modal prosthetic fingertip sensor has a wide range of potential applications in prosthetic and robotic grasping. The sensor's ability to estimate proximity, contact, force, location, and direction of impact provides supplemental information compared to standard tactile sensors. The proximity sensing can be useful in grasp planning and other shared control methods of prosthetic hands. The utility of the proximity data for sensory restoration is not yet known, but novel mappings between the proximity signal and other tactile percepts are possible using the technology presented here.

The GP method enabled the fusion of the barometer and IR sensor values to form a calibrated force signal. For the classification task, SVM outperformed the CNN approach, likely due to overfitting. Although the numerical values are a good fit, the proposed methods might not generalize over different probing shapes and materials. The IR proximity sensor has a strong dependence on the surface properties of an object, which can affect calibration. However, the sensor's multiple sensing modalities help mitigate these challenges. The linear behavior of the barometer can help calibrate the sensor against objects with various surface properties, while the non-linear response of the IR sensor can identify those surface properties.

The extended spatial capabilities of the sensor provide relevant force feedback to amputees even when an object is not centered against each digit. This ensures a reliable source of force feedback during complex activities of daily life. The combination of the reflectance of IR light and the change in pressure due to the compression of an elastomer creates a richer picture of the interactions between the outside world and the fingertips of a prosthetic hand.

### Future Work

Future developments will focus on integrating these sensors with neural interfaces to provide rich sensory information to upper limb amputees. The calibrated force signal will provide a reliable tactile signal, while the proximity and contact signals allow for investigations of new sensory paradigms. The proximity signal can be mapped to non-physiological percepts, while the contact signal can be utilized in a DESC-based manner. Real-time sensor fusion classification will be implemented to provide spatial and angular information relevant to certain neural interfaces and shared control paradigms of the prosthetic limb.

## CONCLUSION

In summary, the present invention describes a novel multi-modal prosthetic fingertip sensor that integrates an IR proximity sensor and a barometric pressure sensor. The sensor is designed to provide rich sensory information, including proximity, contact, force, spatial location, and angular orientation of applied forces. The sensor is overmolded with a liquid silicon polymer to create a durable and mechanically robust contact surface. The multi-modal sensory information can be fused using Gaussian process regression and machine learning techniques to derive additional information not available from each sensor independently. The sensor's extended spatial capabilities ensure reliable force feedback during complex activities of daily life, making it a valuable tool in the field of upper limb prosthetic design.