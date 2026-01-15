- **Introduction**: The paper introduces MaRTiny, a low-cost device combining meteorological sensing and computer vision to estimate Mean Radiant Temperature (MRT) and monitor public space usage. It aims to provide non-experts with an affordable tool for urban climate studies, addressing gaps in existing sensor accuracy and precision.

- **System Design**: MaRTiny integrates off-the-shelf sensors, edge devices, and machine learning algorithms. The hardware setup includes meteorological sensors and a camera, while the software processes data to estimate MRT and detect pedestrians and shading.

- **MRT Estimation**: Initial tests show an RMSE of 10 °C between MaRTiny and reference measurements, which is common for outdoor MRT measurements. Machine learning models, specifically SVMs, are used to improve accuracy by correcting sensor errors.

- **Vision System**: The vision system uses BDRAR for shadow detection and YOLOv3 for pedestrian recognition. While effective in detecting pedestrians, it struggles with occlusions and 3D positioning, leading to occasional misclassifications of shading.

- **Evaluation**: MaRTiny's MRT estimation achieved an RMSE < 4 °C after excluding morning hours affected by shading. The vision system had a precision of 90% for shadow detection and 85% for pedestrian detection, with an overall accuracy of 80% for detecting shaded pedestrians.

- **Discussion**: Despite limitations, MaRTiny demonstrates the potential of low-cost sensors and machine learning in urban climate monitoring. Future work will focus on calibrating the system and collecting more data to refine the models. The device can support heat mitigation efforts by informing decisions on cooling infrastructure.

- **Conclusion**: MaRTiny is a promising tool for urban climate informatics, enabling detailed analyses of how weather impacts public space usage. Its low-cost and portable design make it suitable for widespread deployment, potentially aiding municipal decision-making in heat mitigation strategies.