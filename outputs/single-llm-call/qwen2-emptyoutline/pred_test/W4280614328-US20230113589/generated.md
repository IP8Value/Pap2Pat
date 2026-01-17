# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a low-cost, portable, and smart Internet-of-Things (IoT) weather station for measuring Mean Radiant Temperature (MRT) and monitoring pedestrian behavior in outdoor urban environments. More specifically, the invention pertains to a compact and integrated system that combines meteorological sensors with a vision system to provide real-time data on thermal conditions and space use, enabling active shade management and enhancing urban resilience to extreme heat.

## BACKGROUND

Global temperatures have been rising, with the past decade marking the warmest 10-year period on record. Extreme heat events are becoming more frequent and prolonged, posing significant health risks and limiting outdoor activities. Traditional methods of reporting urban heat, such as air temperature, are insufficient for quantifying personal heat exposure. A more comprehensive metric, Mean Radiant Temperature (MRT), is essential for understanding the heat load on the human body. MRT accounts for both shortwave and longwave radiation from the environment, providing a more accurate representation of thermal comfort.

High-precision MRT measurement systems, such as the biometeorological instrument platform MaRTy, are expensive and cumbersome. Lower-cost alternatives, such as gray globe thermometers, have been developed but lack the connectivity and data processing capabilities needed for real-time monitoring and analysis. Additionally, these systems do not integrate pedestrian behavior data, which is crucial for effective urban planning and heat mitigation strategies.

There is a need for a low-cost, compact, and intelligent system that can measure MRT and monitor pedestrian behavior in real-time. Such a system would enable cities to implement targeted cooling infrastructure and improve public health and well-being in outdoor urban environments.

## SUMMARY

The present invention addresses the aforementioned needs by providing a low-cost, portable, and smart IoT weather station, referred to as MaRTiny, for measuring Mean Radiant Temperature (MRT) and monitoring pedestrian behavior in outdoor urban environments. The MaRTiny system comprises a weather station and a vision system, both integrated into a compact and low-power design.

The weather station includes sensors for measuring air temperature, relative humidity, globe temperature, and wind speed. The globe temperature is converted to MRT using an empirical model, and a machine learning algorithm further refines the MRT estimation to account for sensor inaccuracies and environmental variations. The vision system uses a camera and advanced machine learning algorithms to detect and count pedestrians, as well as to identify whether they are in the shade or sun.

Key features of the MaRTiny system include:
- **Low-Cost and Compact Design**: Built using off-the-shelf components and microcontrollers, the system is affordable and easy to deploy.
- **Real-Time Data Transmission**: Data is transmitted to a cloud database via Wi-Fi, enabling remote monitoring and analysis.
- **Machine Learning for MRT Estimation**: A supervised learning model, specifically a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel, is used to improve the accuracy of MRT measurements.
- **Pedestrian and Shade Detection**: The vision system uses deep learning models, including YOLOv3 for object detection and BDRAR for shadow detection, to monitor pedestrian behavior and shade utilization.
- **Privacy Preservation**: Only quantitative metrics are stored, and images are discarded after analysis to protect individual privacy.

The MaRTiny system is designed to support active shade management and enhance urban resilience to extreme heat by providing real-time data on thermal conditions and space use in outdoor urban environments.

## DETAILED DESCRIPTION

### Mean Radiant Temperature (MRT) Sensing

Mean Radiant Temperature (MRT) is a critical metric for assessing human thermal comfort in outdoor environments. MRT quantifies the total shortwave and longwave radiation the human body is exposed to, including radiation from the sun and surrounding surfaces. High-precision MRT measurement systems, such as the 6-directional method using net radiometers, are expensive and complex. Lower-cost alternatives, such as gray globe thermometers, offer a more practical solution but can suffer from inaccuracies due to factors like sensor lag and environmental variations.

The MaRTiny system uses a gray globe thermometer to measure globe temperature, which is then converted to MRT using an empirical model. The empirical model accounts for the relationship between globe temperature, air temperature, and other meteorological parameters. To further improve the accuracy of MRT measurements, a machine learning algorithm is employed. Specifically, a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel is trained on a dataset of paired MaRTiny and high-precision MRT measurements. The SVM model learns to correct for sensor inaccuracies and environmental variations, providing a robust and accurate MRT estimation.

### System Overview

The MaRTiny system is a compact, low-cost, and smart IoT weather station designed for measuring MRT and monitoring pedestrian behavior in outdoor urban environments. The system consists of two main components: the weather station and the vision system.

#### Weather Station

The weather station includes the following sensors:
- **Air Temperature Sensor**: Measures the ambient air temperature using a downward-facing white cup to shield the sensor from direct sunlight.
- **Globe Temperature Sensor**: Measures the temperature of a gray ping-pong ball attached to a probe, which approximates the albedo of human skin and clothing.
- **Relative Humidity Sensor**: Measures the relative humidity of the air.
- **Wind Speed Sensor (Anemometer)**: Measures the wind speed at the location.
- **UV Sensor**: Measures the ultraviolet (UV) intensity, which is used to train the machine learning model for MRT estimation.

The weather station is powered by a DC adapter of 5V/4A, which is shared between the weather station and the vision system. The data from the sensors are collected and averaged every minute by an Arduino Uno microcontroller. The data are then transmitted to a NodeMCU microcontroller, which communicates with a cloud database via Wi-Fi using the MQTT protocol.

#### Vision System

The vision system is designed to detect and count pedestrians, as well as to identify whether they are in the shade or sun. The system uses a compact MIPI (Mobile Industry Processor Interface) camera to capture video streams, which are processed by an NVIDIA Jetson Nano edge device. The Jetson Nano runs state-of-the-art deep learning models for object detection and shadow detection.

- **Object Detection**: The YOLOv3 (You Only Look Once version 3) model is used for object detection. YOLOv3 is trained on the Microsoft COCO dataset and is capable of detecting 80 different classes of objects, including pedestrians. The model is optimized for the Jetson Nano using Nvidia's TensorRT engine to achieve a frame rate of 4 fps.
- **Shadow Detection**: The BDRAR (Bi-directional Feature Pyramid with Recurrent Attention Residual Modules) model is used for shadow detection. BDRAR takes a single image as input and outputs a binary shadow map, which indicates the presence of shade per pixel. The model leverages a convolutional neural network (CNN) to extract feature maps at different spatial resolutions and employs recurrent attention residual modules to fully exploit global and local context.

The vision system calculates the Intersection over Union (IOU) of the bounding box of detected objects with the shadow map to determine whether a pedestrian is in the shade or sun. A pedestrian is considered to be in the shade if 40% of the bounding box region is inside the shade map. The system reports pedestrian counts under sun and shade, along with other relevant counts (e.g., umbrellas, pets, and bicycles) to the cloud database. The captured images are discarded after analysis to preserve privacy.

### Machine Learning Algorithm Development

#### Machine Learning for Accurate MRT Estimation

The MaRTiny system uses a machine learning algorithm to improve the accuracy of MRT measurements. The algorithm is formulated as a supervised learning problem, where labeled ground-truth MRT values are provided in correspondence with the less robust meteorological sensor data. The machine learning model is trained on a dataset of paired MaRTiny and high-precision MRT measurements.

Two machine learning models were explored: a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel and a traditional Artificial Neural Network (ANN). The SVM with RBF kernel achieved the highest accuracy on the evaluation dataset, with a Root Mean Square Error (RMSE) of less than 4 °C. The SVM model is computationally lightweight and can be easily deployed on the Jetson Nano for performing inference.

#### People and Shade Detection

##### Shadow Detection

The BDRAR model is used for shadow detection. The model takes a single image as input and outputs a binary shadow map. The BDRAR network leverages a CNN to extract feature maps at different spatial resolutions and employs recurrent attention residual modules to fully exploit global and local context. The model captures shadow details in local regions and understands the overall shadow region of the image.

##### Object Detection

The YOLOv3 model is used for object detection. YOLOv3 is trained on the Microsoft COCO dataset and is capable of detecting 80 different classes of objects, including pedestrians. The model is optimized for the Jetson Nano using Nvidia's TensorRT engine to achieve a frame rate of 4 fps.

##### Pedestrian in Shade Detection

An algorithm is used to identify pedestrians in the shade without determining their exact position in 3D space. The algorithm calculates the Intersection over Union (IOU) of the bounding box of detected objects with the shadow map. A pedestrian is considered to be in the shade if 40% of the bounding box region is inside the shade map. The algorithm also considers the bottom half of the bounding box as the Region of Interest (ROI) to account for the cooling effect of shade. The system reports pedestrian counts under sun and shade, along with other relevant counts, to the cloud database. The captured images are discarded after analysis to preserve privacy.

### System Evaluation

#### Data Collection

The MaRTiny system was evaluated in two sun-exposed outdoor locations in Tempe, Arizona, United States. The system was paired with the MaRTy human-biometeorological platform for simultaneous data logging. Ground truth MRT values were calculated using the 6-directional method with net radiometers. The MaRTiny system logged data every minute, while the MaRTy system logged data every 2 seconds. One-minute averages were calculated for comparison.

An image dataset was also collected for evaluating object and shade detection. Images from the MIPI camera were stored at random intervals along with the bounding boxes of the interested objects. Ground truth bounding boxes were drawn manually using tools such as AlexyAB and Tosmonav. Precision and Recall for each object were calculated, and the mean Average Precision (mAP) was used to evaluate the performance of the object detection model.

#### MRT Estimation

The performance of the MaRTiny system in estimating MRT values was evaluated using the empirical model and the machine learning algorithm. The empirical model was used to convert globe temperature to MRT, and the machine learning algorithm was used to refine the MRT estimation. The machine learning model, specifically the SVM with RBF kernel, achieved the highest accuracy on the evaluation dataset, with an RMSE of less than 4 °C.

#### Shade and Object Detection

The performance of the vision system in detecting pedestrians and shade was evaluated using the YOLOv3 and BDRAR models. The YOLOv3 model achieved an mAP of around 55%, with an Average Precision of more than 85% for the class of Pedestrian. The BDRAR model achieved a precision of 90% for shadow detection. The pedestrian in shade detection algorithm achieved an accuracy of around 80% on a custom dataset of 50 images.

### Discussion

The MaRTiny system is a novel low-cost device that combines meteorological sensing with computer vision to estimate MRT and monitor pedestrian behavior in outdoor urban environments. The system is designed to be used by non-experts, such as city staff and citizen scientists, to support active shade management and enhance urban resilience to extreme heat.

While the system shows promising results, there are several limitations and areas for future improvement:
- **Calibration**: The MaRTiny system should be fully calibrated against NIST-certified sensors before deployment to ensure accuracy.
- **Environmental Variations**: The system may be affected by environmental variations, such as shading from nearby objects, which can introduce errors in MRT measurements.
- **Vision System Accuracy**: The vision system may exhibit minor inaccuracies in shadow map estimation and object detection, particularly in crowded scenes or when objects occlude each other.

Future research could focus on improving the calibration of the system, refining the machine learning models, and enhancing the vision system to handle more complex scenarios. The MaRTiny system represents a significant step forward in low-cost, portable, and intelligent sensing for urban climate and thermal comfort studies.