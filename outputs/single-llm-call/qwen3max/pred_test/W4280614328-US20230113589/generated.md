# DESCRIPTION

## TECHNICAL FIELD

The present invention relates generally to biometeorological sensing devices, and more particularly to an integrated Internet-of-Things (IoT) weather station capable of measuring hyperlocal meteorological conditions, estimating Mean Radiant Temperature (MRT), and analyzing human behavior in outdoor urban environments through computer vision. The invention provides a compact, low-cost, portable platform that combines environmental sensing with artificial intelligence-driven image analysis to assess thermal exposure and space utilization by pedestrians in real time. This system is especially suited for deployment in hot, arid climates where accurate assessment of radiant heat load and shade availability is critical for public health, urban planning, and climate resilience strategies.

## BACKGROUND

Extreme heat events and prolonged heat waves have become increasingly frequent and intense due to global climate change, posing significant risks to human health, productivity, and social activity in outdoor urban settings. The year 2020 marked the warmest decade on record, with global average temperatures rising approximately 1.3 °C above preindustrial levels. These trends are expected to continue, necessitating adaptive measures to enhance urban resilience and protect vulnerable populations. Traditional metrics such as air temperature, while widely reported, fail to capture the full spectrum of thermal stress experienced by individuals outdoors because they do not account for radiative heat exchange between the human body and its surroundings.

A more physiologically relevant metric is the Mean Radiant Temperature (MRT), which quantifies the net effect of shortwave solar radiation and longwave thermal radiation from surrounding surfaces on the human body. MRT can differ substantially from air temperature—by as much as 30 °C under direct sunlight—and is recognized as the dominant factor influencing outdoor thermal comfort in hot, dry climates such as Phoenix, Arizona. Accurate MRT measurements are essential for evaluating heat mitigation strategies, designing thermally comfortable urban spaces, and predicting heat-related health outcomes.

Despite its importance, precise MRT measurement has historically required expensive, bulky instrumentation such as the six-directional radiometer array used in research-grade platforms like MaRTy. These systems employ three orthogonal net radiometers to measure directional shortwave and longwave fluxes, which are then converted to MRT using the Stefan-Boltzmann law. However, the high cost (often exceeding $15,000 per unit) and complexity of such setups limit their scalability for widespread urban monitoring.

Lower-cost alternatives, such as black or gray globe thermometers, offer a practical compromise but suffer from several limitations. Globe-based MRT estimates are sensitive to variations in globe material, color, size, wind speed, and local shading conditions. Empirical models exist to convert globe temperature to MRT, but these often assume idealized environmental conditions and may introduce significant errors—reported root mean square errors (RMSE) range from 7 °C to over 20 °C depending on context. Furthermore, existing low-cost sensors lack integration with real-time data transmission, cloud storage, or behavioral analytics, rendering them insufficient for dynamic urban heat management.

Concurrently, understanding how people use public spaces—particularly their preference for shaded versus sun-exposed areas—is crucial for effective urban design and heat adaptation. Yet, current methods rely on manual observation or disconnected sensor networks, which are labor-intensive, non-scalable, and incapable of correlating microclimate data with human behavior in real time. There remains a critical need for an integrated, intelligent, and affordable biometeorological sensing platform that simultaneously captures environmental conditions and pedestrian activity with privacy-preserving analytics.

## SUMMARY

The present invention discloses a novel IoT-enabled biometeorological sensing device, herein referred to as MaRTiny, designed to address the limitations of existing systems by providing accurate, hyperlocal MRT estimation coupled with real-time pedestrian and shade detection. MaRTiny is a compact, low-cost, self-contained weather station that integrates off-the-shelf meteorological sensors—including air temperature, relative humidity, wind speed, UV intensity, and a custom gray globe thermometer—with a vision system powered by edge-computing artificial intelligence.

The device leverages a machine learning model trained on paired high-fidelity and low-cost sensor data to correct systematic errors in globe-derived MRT estimates, achieving a root mean square error (RMSE) of approximately 4 °C compared to reference-grade measurements. Concurrently, an onboard NVIDIA Jetson Nano processes video streams from a MIPI camera using deep learning algorithms: YOLOv3 for object and pedestrian detection, and BDRAR (Bi-directional Feature Pyramid with Recurrent Attention Residual Modules) for pixel-level shade mapping. A novel algorithm computes the intersection over union (IOU) between pedestrian bounding boxes and the shade map, specifically analyzing the lower half of each bounding box to determine whether a person is experiencing the cooling benefits of shade.

MaRTiny transmits anonymized data—including MRT, meteorological parameters, pedestrian counts in sun and shade, and other relevant object detections—to a secure cloud database via Wi-Fi using the MQTT protocol. All raw images are discarded after processing to preserve individual privacy. The entire system operates on less than 20 watts of power and can be constructed for under $200, making it suitable for large-scale deployment by municipalities, researchers, and citizen scientists. By unifying environmental sensing and behavioral analytics in a single, intelligent platform, MaRTiny enables data-driven decisions for urban heat mitigation, shade infrastructure investment, and public health protection.

## DETAILED DESCRIPTION

### Mean Radiant Temperature (MRT) Sensing

Mean Radiant Temperature (MRT) represents the uniform temperature of an imaginary blackbody enclosure that would result in the same net radiant heat exchange as the actual non-uniform environment. It is a key parameter in human biometeorology and is defined by the Stefan-Boltzmann law as:

\[
MRT = \left( \frac{\sum_{i=1}^{6} W_i (a_k K_i + a_l L_i)}{\sigma} \right)^{1/4}
\]

where \(K_i\) and \(L_i\) are directional shortwave and longwave radiation fluxes, \(a_k\) and \(a_l\) are absorption coefficients, \(W_i\) are angular weighting factors (0.06 for vertical, 0.22 for horizontal directions), and \(\sigma\) is the Stefan-Boltzmann constant. While this six-directional method is highly accurate, it requires three net radiometers costing thousands of dollars each, limiting practical deployment.

As a cost-effective alternative, globe thermometers—typically hollow spheres painted matte black or gray—are used to estimate MRT based on equilibrium temperature under combined radiative and convective heat transfer. Thorsson et al. demonstrated that a 38-mm acrylic gray globe approximates the average albedo of human skin and clothing, enabling reasonable MRT estimates. However, globe performance is affected by wind speed, solar elevation, material emissivity, and local obstructions. Empirical corrections, such as those proposed by Vanos et al., relate globe temperature (\(T_g\)) to MRT using air temperature (\(T_a\)), wind speed (\(v\)), and globe properties, yet residual errors persist due to sensor inaccuracies and environmental heterogeneity.

Conventional urban microclimate models rely on detailed 3D city geometry and radiative transfer simulations, which are computationally intensive and require extensive input data. Meanwhile, pedestrian counting techniques—ranging from infrared beams to computer vision—have advanced significantly, with deep convolutional neural networks (CNNs) now enabling robust crowd estimation via individual detection or density regression. However, none of these approaches integrate real-time MRT sensing with behavioral analytics in a unified, field-deployable system.

The present invention introduces the MaRTiny system, which comprises a biometeorological sensing device (BSD) integrating a weather station, a vision system, and a machine learning module. As illustrated in FIG. 1, the BSD includes a gray globe thermometer, shielded air temperature probe, humidity sensor, anemometer, and UV sensor, all interfaced with an Arduino Uno microcontroller. Meteorological data are averaged over one-minute intervals and transmitted via UART to a NodeMCU ESP8266 module, which securely uploads the data to an AWS DynamoDB cloud database using MQTT and stored PEM authentication files.

The vision system employs an NVIDIA Jetson Nano edge AI computer connected to a MIPI camera. Video frames are processed through a GStreamer pipeline, with inference performed locally using TensorRT-optimized deep learning models. YOLOv3 detects pedestrians and other objects, while BDRAR generates a binary shade map. A custom algorithm computes the IOU between the lower 50% of each pedestrian’s bounding box and the shade map; if IOU ≥ 0.8 (equivalent to 40% of the full box), the person is classified as shaded. Only aggregated counts—not images—are transmitted to the cloud, ensuring privacy compliance.

FIG. 2 details the sensor configuration and data flow. Power is supplied via a 5V/4A DC adapter, with voltage stepped up to 9V for the anemometer. The system is designed for modularity, allowing additional sensors without significant redesign. Machine learning models for MRT correction run on the Jetson Nano, enabling real-time error compensation based on inputs including globe temperature, air temperature, humidity, and UV index.

## EXAMPLES

### System Overview

The MaRTiny system functions as an autonomous, solar-compatible IoT node for biometeorological monitoring. Its core components include a custom gray globe thermometer made from a 38-mm acrylic ping-pong ball housing a DS18B20 temperature probe, which emulates human radiative absorption. Air temperature is measured by a downward-facing probe within a white reflective cup to minimize solar heating. Relative humidity is captured via a DHT22 sensor, wind speed by a reed-switch anemometer, and UV intensity by a GUVA-S12SD photodiode. These sensors interface with an Arduino Uno, which samples data at 80 Hz and outputs one-minute averages to a NodeMCU microcontroller.

The vision subsystem centers on an NVIDIA Jetson Nano configured in 10W mode, connected to a Raspberry Pi-compatible MIPI camera. The Jetson runs a GStreamer pipeline that feeds frames into two deep learning models: YOLOv3-Darknet for object detection and BDRAR for shade segmentation. Detected pedestrians are analyzed for shade exposure using the ROI-based IOU method. All visual data are processed on-device; only numerical summaries (e.g., “3 pedestrians in shade, 5 in sun”) are transmitted.

Communication occurs via Wi-Fi using the MQTT protocol to an AWS DynamoDB instance. Security is ensured through TLS encryption and client certificates stored in the NodeMCU’s flash memory. The entire system draws less than 20W and costs under $200, enabling scalable deployment across urban parks, transit stops, and playgrounds.

### Machine Learning Algorithm Development

To enhance MRT accuracy, a supervised learning approach was adopted. A labeled dataset was created by co-locating MaRTiny with the high-fidelity MaRTy platform over multiple days in Tempe, Arizona, yielding 12,000 training and 3,000 testing samples. Input features included globe temperature, air temperature, relative humidity, and UV index. Both Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel and a shallow Artificial Neural Network (ANN) with ReLU activation were trained using scikit-learn.

Five-fold cross-validation optimized hyperparameters. The SVM-RBF achieved RMSE of 4.1 °C on the evaluation set, outperforming linear and polynomial kernels and matching ANN performance while requiring fewer computational resources. Given the Jetson Nano’s constraints, the SVM model was selected for deployment due to its efficiency and robustness to sensor noise and environmental variability.

### System Evaluation

Evaluation involved paired MaRTy–MaRTiny deployments over three days. Initial globe-based MRT estimates showed RMSE of 10.0 °C due to morning shading from a palm tree—a common real-world challenge. After applying the SVM correction model, RMSE dropped to 4.0 °C, demonstrating the model’s ability to compensate for spatial and instrumental discrepancies.

For vision tasks, a custom dataset of 30 annotated images was used. YOLOv3 achieved 55% mAP at IOU=0.5 and 85% average precision for pedestrians. BDRAR yielded 90% pixel-wise precision in shade detection. The pedestrian-in-shade algorithm achieved 80% accuracy on a 50-image test set, validated via confusion matrix analysis. Errors arose primarily from cast shadows and occlusions, highlighting opportunities for 3D-aware refinement.

### Discussion

MaRTiny represents a significant advancement in urban climate sensing by merging affordable meteorology with privacy-conscious computer vision. While globe thermometers inherently suffer from response lag and environmental sensitivity, the integrated machine learning model mitigates these issues, enabling scientific-grade MRT estimation at a fraction of traditional costs. The system’s edge-processing architecture ensures real-time operation without continuous cloud dependency.

Future work includes NIST-traceable calibration, multi-season validation, and integration of 3D scene reconstruction to improve shade-person alignment. Deployment plans in Tempe, Arizona, aim to inform municipal cooling infrastructure investments. Critically, MaRTiny preserves privacy by design—no identifiable imagery is stored or transmitted—making it suitable for public spaces. The platform exemplifies Urban Climate Informatics, bridging environmental science, AI, and civic technology to build heat-resilient cities.