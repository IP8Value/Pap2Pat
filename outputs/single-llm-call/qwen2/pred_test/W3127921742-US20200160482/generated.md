# DESCRIPTION

## FIELD OF THE DISCLOSURE

The present disclosure relates to imaging and vision systems, and more specifically, to methods and systems for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology addresses the challenges of energy efficiency and image fidelity in near-sensor processing by providing dynamic thermal management techniques that adapt to the situational needs of vision and imaging applications.

## BACKGROUND

Imaging and vision systems play a crucial role in modern computing, enabling a wide range of applications such as object detection, augmented reality, and autonomous navigation. These systems typically consist of an image sensor and a processing unit, which are often separated by long interconnects. The high data rates required to transfer pixel data from the sensor to the processing unit, often exceeding 2 Gbps for high-resolution video, create significant bottlenecks in terms of energy efficiency and processing performance. As a result, current vision systems often consume multiple watts of power, which is prohibitive for many applications, especially those requiring continuous operation.

To address these challenges, recent advancements in 3D stacked integrated circuit architectures have enabled the integration of processing units near the image sensor. This approach, known as near-sensor processing, significantly reduces the energy required for data transfer and improves overall system efficiency. However, near-sensor processing introduces new challenges, particularly related to thermal management. The close proximity of the processing unit to the sensor can cause the sensor to heat up, leading to increased thermal noise and degraded image quality. This is particularly problematic in low-light conditions, where the sensor is already more susceptible to noise.

Existing dynamic thermal management (DTM) techniques, which are primarily designed for general-purpose processors, do not adequately address the specific needs of imaging systems. These techniques often focus on keeping the processor temperature below a thermal design power (TDP) threshold, without considering the impact of transient temperature changes on image fidelity. Therefore, there is a need for novel thermal management strategies that can effectively regulate sensor temperature while maintaining high image quality and system efficiency.

## SUMMARY

The present disclosure provides a comprehensive solution for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology includes a runtime controller, referred to as Stagioni, which orchestrates temperature management policies to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application.

In one aspect, the disclosure provides a method for managing thermal noise in a 3D stacked image sensor. The method includes the steps of:
1. Monitoring the temperature of the image sensor.
2. Determining a situational temperature threshold based on the current environmental conditions and the fidelity requirements of the application.
3. Activating a stop-capture-go policy to temporarily halt near-sensor processing when the sensor temperature exceeds the situational threshold, thereby allowing the sensor to cool down and maintain high image fidelity.
4. Activating a seasonal migration policy to periodically shift processing to a thermally isolated far-sensor processing unit when the sensor temperature exceeds a predefined high temperature threshold, thereby allowing the sensor to cool down while maintaining continuous processing.

In another aspect, the disclosure provides a system for managing thermal noise in a 3D stacked image sensor. The system includes:
1. An image sensor configured to capture image data.
2. A near-sensor processing unit configured to perform image processing tasks.
3. A far-sensor processing unit configured to perform image processing tasks when the near-sensor processing unit is deactivated.
4. A runtime controller configured to monitor the temperature of the image sensor, determine situational temperature thresholds, and activate stop-capture-go and seasonal migration policies based on the temperature and application requirements.

The disclosed technology offers several advantages over existing thermal management techniques. By dynamically adapting to the situational needs of the application, the disclosed methods and systems can maintain high image fidelity while minimizing system power consumption. Additionally, the use of stop-capture-go and seasonal migration policies ensures that the sensor temperature is regulated effectively, preventing thermal noise from degrading image quality.

## DETAILED DESCRIPTION

The present disclosure provides a detailed description of methods and systems for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology addresses the challenges of energy efficiency and image fidelity by providing dynamic thermal management techniques that adapt to the situational needs of vision and imaging applications.

### Field of the Disclosure

The field of the disclosure pertains to imaging and vision systems, particularly those utilizing 3D stacked image sensors for near-sensor processing. The disclosed technology is applicable to a wide range of applications, including but not limited to, object detection, augmented reality, and autonomous navigation.

### Background

Imaging and vision systems are essential components in modern computing, enabling a variety of applications such as object detection, augmented reality, and autonomous navigation. These systems typically consist of an image sensor and a processing unit, which are often separated by long interconnects. The high data rates required to transfer pixel data from the sensor to the processing unit, often exceeding 2 Gbps for high-resolution video, create significant bottlenecks in terms of energy efficiency and processing performance. As a result, current vision systems often consume multiple watts of power, which is prohibitive for many applications, especially those requiring continuous operation.

To address these challenges, recent advancements in 3D stacked integrated circuit architectures have enabled the integration of processing units near the image sensor. This approach, known as near-sensor processing, significantly reduces the energy required for data transfer and improves overall system efficiency. However, near-sensor processing introduces new challenges, particularly related to thermal management. The close proximity of the processing unit to the sensor can cause the sensor to heat up, leading to increased thermal noise and degraded image quality. This is particularly problematic in low-light conditions, where the sensor is already more susceptible to noise.

Existing dynamic thermal management (DTM) techniques, which are primarily designed for general-purpose processors, do not adequately address the specific needs of imaging systems. These techniques often focus on keeping the processor temperature below a thermal design power (TDP) threshold, without considering the impact of transient temperature changes on image fidelity. Therefore, there is a need for novel thermal management strategies that can effectively regulate sensor temperature while maintaining high image quality and system efficiency.

### Summary

The present disclosure provides a comprehensive solution for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology includes a runtime controller, referred to as Stagioni, which orchestrates temperature management policies to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application.

In one aspect, the disclosure provides a method for managing thermal noise in a 3D stacked image sensor. The method includes the steps of:
1. Monitoring the temperature of the image sensor.
2. Determining a situational temperature threshold based on the current environmental conditions and the fidelity requirements of the application.
3. Activating a stop-capture-go policy to temporarily halt near-sensor processing when the sensor temperature exceeds the situational threshold, thereby allowing the sensor to cool down and maintain high image fidelity.
4. Activating a seasonal migration policy to periodically shift processing to a thermally isolated far-sensor processing unit when the sensor temperature exceeds a predefined high temperature threshold, thereby allowing the sensor to cool down while maintaining continuous processing.

In another aspect, the disclosure provides a system for managing thermal noise in a 3D stacked image sensor. The system includes:
1. An image sensor configured to capture image data.
2. A near-sensor processing unit configured to perform image processing tasks.
3. A far-sensor processing unit configured to perform image processing tasks when the near-sensor processing unit is deactivated.
4. A runtime controller configured to monitor the temperature of the image sensor, determine situational temperature thresholds, and activate stop-capture-go and seasonal migration policies based on the temperature and application requirements.

The disclosed technology offers several advantages over existing thermal management techniques. By dynamically adapting to the situational needs of the application, the disclosed methods and systems can maintain high image fidelity while minimizing system power consumption. Additionally, the use of stop-capture-go and seasonal migration policies ensures that the sensor temperature is regulated effectively, preventing thermal noise from degrading image quality.

### Detailed Description

#### Field of the Disclosure

The field of the disclosure pertains to imaging and vision systems, particularly those utilizing 3D stacked image sensors for near-sensor processing. The disclosed technology is applicable to a wide range of applications, including but not limited to, object detection, augmented reality, and autonomous navigation.

#### Background

Imaging and vision systems are essential components in modern computing, enabling a variety of applications such as object detection, augmented reality, and autonomous navigation. These systems typically consist of an image sensor and a processing unit, which are often separated by long interconnects. The high data rates required to transfer pixel data from the sensor to the processing unit, often exceeding 2 Gbps for high-resolution video, create significant bottlenecks in terms of energy efficiency and processing performance. As a result, current vision systems often consume multiple watts of power, which is prohibitive for many applications, especially those requiring continuous operation.

To address these challenges, recent advancements in 3D stacked integrated circuit architectures have enabled the integration of processing units near the image sensor. This approach, known as near-sensor processing, significantly reduces the energy required for data transfer and improves overall system efficiency. However, near-sensor processing introduces new challenges, particularly related to thermal management. The close proximity of the processing unit to the sensor can cause the sensor to heat up, leading to increased thermal noise and degraded image quality. This is particularly problematic in low-light conditions, where the sensor is already more susceptible to noise.

Existing dynamic thermal management (DTM) techniques, which are primarily designed for general-purpose processors, do not adequately address the specific needs of imaging systems. These techniques often focus on keeping the processor temperature below a thermal design power (TDP) threshold, without considering the impact of transient temperature changes on image fidelity. Therefore, there is a need for novel thermal management strategies that can effectively regulate sensor temperature while maintaining high image quality and system efficiency.

#### Summary

The present disclosure provides a comprehensive solution for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology includes a runtime controller, referred to as Stagioni, which orchestrates temperature management policies to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application.

In one aspect, the disclosure provides a method for managing thermal noise in a 3D stacked image sensor. The method includes the steps of:
1. Monitoring the temperature of the image sensor.
2. Determining a situational temperature threshold based on the current environmental conditions and the fidelity requirements of the application.
3. Activating a stop-capture-go policy to temporarily halt near-sensor processing when the sensor temperature exceeds the situational threshold, thereby allowing the sensor to cool down and maintain high image fidelity.
4. Activating a seasonal migration policy to periodically shift processing to a thermally isolated far-sensor processing unit when the sensor temperature exceeds a predefined high temperature threshold, thereby allowing the sensor to cool down while maintaining continuous processing.

In another aspect, the disclosure provides a system for managing thermal noise in a 3D stacked image sensor. The system includes:
1. An image sensor configured to capture image data.
2. A near-sensor processing unit configured to perform image processing tasks.
3. A far-sensor processing unit configured to perform image processing tasks when the near-sensor processing unit is deactivated.
4. A runtime controller configured to monitor the temperature of the image sensor, determine situational temperature thresholds, and activate stop-capture-go and seasonal migration policies based on the temperature and application requirements.

The disclosed technology offers several advantages over existing thermal management techniques. By dynamically adapting to the situational needs of the application, the disclosed methods and systems can maintain high image fidelity while minimizing system power consumption. Additionally, the use of stop-capture-go and seasonal migration policies ensures that the sensor temperature is regulated effectively, preventing thermal noise from degrading image quality.

#### Detailed Description

##### Introduction

Imaging and vision systems are critical for modern computing, enabling applications such as object detection, augmented reality, and autonomous navigation. These systems typically consist of an image sensor and a processing unit, which are often separated by long interconnects. The high data rates required to transfer pixel data from the sensor to the processing unit, often exceeding 2 Gbps for high-resolution video, create significant bottlenecks in terms of energy efficiency and processing performance. As a result, current vision systems often consume multiple watts of power, which is prohibitive for many applications, especially those requiring continuous operation.

To address these challenges, recent advancements in 3D stacked integrated circuit architectures have enabled the integration of processing units near the image sensor. This approach, known as near-sensor processing, significantly reduces the energy required for data transfer and improves overall system efficiency. However, near-sensor processing introduces new challenges, particularly related to thermal management. The close proximity of the processing unit to the sensor can cause the sensor to heat up, leading to increased thermal noise and degraded image quality. This is particularly problematic in low-light conditions, where the sensor is already more susceptible to noise.

Existing dynamic thermal management (DTM) techniques, which are primarily designed for general-purpose processors, do not adequately address the specific needs of imaging systems. These techniques often focus on keeping the processor temperature below a thermal design power (TDP) threshold, without considering the impact of transient temperature changes on image fidelity. Therefore, there is a need for novel thermal management strategies that can effectively regulate sensor temperature while maintaining high image quality and system efficiency.

##### Problem Statement

The primary challenge in near-sensor processing is the management of thermal noise. The close proximity of the processing unit to the sensor can cause the sensor to heat up, leading to increased thermal noise and degraded image quality. This is particularly problematic in low-light conditions, where the sensor is already more susceptible to noise. Existing DTM techniques, which are primarily designed for general-purpose processors, do not adequately address the specific needs of imaging systems. These techniques often focus on keeping the processor temperature below a TDP threshold, without considering the impact of transient temperature changes on image fidelity.

##### Solution Overview

The present disclosure provides a comprehensive solution for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology includes a runtime controller, referred to as Stagioni, which orchestrates temperature management policies to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application.

The key components of the disclosed technology are:
1. **Image Sensor**: Captures image data.
2. **Near-Sensor Processing Unit**: Performs image processing tasks near the sensor.
3. **Far-Sensor Processing Unit**: Performs image processing tasks when the near-sensor processing unit is deactivated.
4. **Runtime Controller (Stagioni)**: Monitors the temperature of the image sensor, determines situational temperature thresholds, and activates stop-capture-go and seasonal migration policies based on the temperature and application requirements.

##### Method for Managing Thermal Noise

The method for managing thermal noise in a 3D stacked image sensor includes the following steps:

1. **Monitoring the Temperature of the Image Sensor**:
   - The runtime controller continuously monitors the temperature of the image sensor using an on-chip temperature sensor.
   - The temperature data is used to determine the current operating conditions of the sensor.

2. **Determining a Situational Temperature Threshold**:
   - The runtime controller determines a situational temperature threshold based on the current environmental conditions (e.g., ambient temperature, lighting conditions) and the fidelity requirements of the application.
   - The situational temperature threshold is adjusted dynamically to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application.

3. **Activating a Stop-Capture-Go Policy**:
   - When the sensor temperature exceeds the situational temperature threshold, the runtime controller activates a stop-capture-go policy.
   - The stop-capture-go policy temporarily halts near-sensor processing to allow the sensor to cool down and maintain high image fidelity.
   - The duration of the stop period is determined based on the temperature difference and the required cooling time.

4. **Activating a Seasonal Migration Policy**:
   - When the sensor temperature exceeds a predefined high temperature threshold, the runtime controller activates a seasonal migration policy.
   - The seasonal migration policy periodically shifts processing to a thermally isolated far-sensor processing unit to allow the sensor to cool down while maintaining continuous processing.
   - The frequency and duration of the migration periods are determined based on the temperature difference and the required cooling time.

##### System for Managing Thermal Noise

The system for managing thermal noise in a 3D stacked image sensor includes the following components:

1. **Image Sensor**:
   - Captures image data.
   - Includes an on-chip temperature sensor for monitoring the temperature of the sensor.

2. **Near-Sensor Processing Unit**:
   - Performs image processing tasks near the sensor.
   - Includes a clock gating mechanism to temporarily halt processing when the stop-capture-go policy is activated.

3. **Far-Sensor Processing Unit**:
   - Performs image processing tasks when the near-sensor processing unit is deactivated.
   - Includes a state transfer mechanism to synchronize with the near-sensor processing unit during seasonal migration.

4. **Runtime Controller (Stagioni)**:
   - Monitors the temperature of the image sensor using the on-chip temperature sensor.
   - Determines situational temperature thresholds based on the current environmental conditions and the fidelity requirements of the application.
   - Activates stop-capture-go and seasonal migration policies based on the temperature and application requirements.
   - Manages the state transfer and synchronization between the near-sensor and far-sensor processing units during seasonal migration.

##### Advantages

The disclosed technology offers several advantages over existing thermal management techniques:

1. **High Image Fidelity**:
   - By dynamically adapting to the situational needs of the application, the disclosed methods and systems can maintain high image fidelity while minimizing system power consumption.
   - The use of stop-capture-go and seasonal migration policies ensures that the sensor temperature is regulated effectively, preventing thermal noise from degrading image quality.

2. **Energy Efficiency**:
   - The disclosed methods and systems minimize system power consumption by dynamically adjusting the temperature thresholds and activating stop-capture-go and seasonal migration policies based on the application requirements.
   - The use of near-sensor processing reduces the energy required for data transfer, leading to significant energy savings.

3. **Adaptability**:
   - The runtime controller (Stagioni) is designed to adapt to the situational needs of the application, ensuring that the sensor operates within a temperature range that meets the fidelity requirements of the application.
   - The system can handle a wide range of environmental conditions and application requirements, making it suitable for a variety of imaging and vision applications.

4. **Performance**:
   - The use of stop-capture-go and seasonal migration policies ensures that the sensor temperature is regulated effectively, preventing thermal noise from degrading image quality.
   - The system maintains continuous processing by periodically shifting processing to a thermally isolated far-sensor processing unit, ensuring that the application performance is not compromised.

##### Conclusion

The present disclosure provides a comprehensive solution for managing thermal noise in 3D stacked image sensors to enable efficient and high-fidelity near-sensor processing. The disclosed technology includes a runtime controller, referred to as Stagioni, which orchestrates temperature management policies to ensure that the sensor operates within a temperature range that meets the fidelity requirements of the application. The disclosed methods and systems offer several advantages over existing thermal management techniques, including high image fidelity, energy efficiency, adaptability, and performance. The disclosed technology is applicable to a wide range of imaging and vision applications, making it a valuable tool for enabling the next generation of imaging and vision systems.