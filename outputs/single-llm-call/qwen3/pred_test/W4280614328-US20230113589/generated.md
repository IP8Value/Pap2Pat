# DESCRIPTION

## TECHNICAL FIELD

- define biometeorological sensing devices

Biometeorological sensing devices are integrated systems designed to measure and analyze environmental parameters that directly influence human thermal perception and physiological response in outdoor settings. These devices combine meteorological instrumentation with computational vision and machine learning capabilities to quantify the complex interplay between radiant energy, air conditions, and human activity patterns. Unlike conventional weather stations that report isolated metrics such as air temperature or humidity, biometeorological sensing devices are engineered to capture the cumulative thermal load experienced by the human body, particularly through the estimation of Mean Radiant Temperature (MRT), which accounts for all directional shortwave and longwave radiation incident upon a person from surrounding surfaces and the sky. Such devices are uniquely suited for deployment in urban environments where microclimatic variability is pronounced due to heterogeneous surface materials, building geometry, vegetation cover, and anthropogenic heat sources. The integration of low-cost sensors, edge computing, and real-time data transmission enables these systems to operate autonomously over extended periods without requiring continuous human oversight, making them ideal for large-scale, spatially distributed monitoring networks aimed at improving public health, urban planning, and climate adaptation strategies.

## BACKGDROUND

- motivate extreme heat and heat waves

Extreme heat and prolonged heat waves have emerged as one of the most significant and growing threats to human health, economic productivity, and social equity in urban environments worldwide. As global temperatures continue to rise, the frequency, intensity, and duration of heat events have increased markedly, placing vulnerable populations—including the elderly, outdoor workers, and those without access to cooling infrastructure—at heightened risk of heat-related illness and mortality. The physiological burden imposed by heat is not uniformly distributed across urban landscapes; rather, it is shaped by localized microclimatic conditions that can vary dramatically over distances of mere meters. Traditional public health advisories and urban heat maps based solely on air temperature fail to capture the true thermal exposure experienced by individuals moving through shaded courtyards, sun-baked plazas, or tree-lined sidewalks. Consequently, mitigation efforts often misallocate resources, targeting areas with high ambient temperatures while overlooking zones where radiant heat accumulation creates dangerous conditions even when air temperatures appear moderate. The inability to accurately assess personal heat exposure limits the effectiveness of adaptive interventions and hinders the development of equitable urban design policies.

- summarize global temperature increase

Over the past century, the Earth’s average surface temperature has risen by approximately 1.3 degrees Celsius above preindustrial levels, with the most rapid warming occurring during the last three decades. This trend is unequivocally linked to anthropogenic emissions of greenhouse gases, which trap outgoing longwave radiation and disrupt the planet’s natural energy balance. Urban areas, due to their high concentration of heat-absorbing materials such as asphalt, concrete, and dark roofing, experience amplified warming known as the urban heat island effect. In many metropolitan regions, nighttime temperatures in city centers exceed those of surrounding rural areas by more than 5°C, exacerbating the cumulative thermal stress on residents. Projections indicate that without substantial mitigation, global temperatures could rise by an additional 2 to 4°C by the end of this century, with some arid and semi-arid cities—such as Phoenix, Las Vegas, and Delhi—facing extreme heat days exceeding 50°C in peak summer months. These conditions threaten not only human health but also critical infrastructure, energy systems, and labor productivity, particularly in outdoor sectors such as construction, agriculture, and transportation.

- describe limitations of air temperature

Air temperature, while widely measured and reported, is an inadequate proxy for human thermal comfort in outdoor environments because it fails to account for the dominant role of radiant heat exchange. The human body absorbs and emits thermal energy through multiple pathways, with longwave radiation from hot surfaces such as pavement, walls, and vehicles contributing significantly more to perceived heat than air temperature alone. In direct sunlight, Mean Radiant Temperature can exceed air temperature by more than 30°C, leading to a sensation of extreme discomfort even when meteorological reports suggest “moderate” conditions. Conversely, under shade structures, air temperature may remain high while radiant load is substantially reduced, resulting in significantly improved thermal comfort. Relying on air temperature as the primary indicator of heat risk leads to systematic underestimation of exposure in sun-exposed zones and overestimation in shaded areas, thereby compromising the accuracy of public health warnings, urban heat vulnerability assessments, and climate adaptation planning. Furthermore, air temperature measurements are highly sensitive to sensor placement, wind speed, and solar radiation interference, making consistent and representative data collection challenging without careful calibration and shielding.

- introduce Mean Radiant Temperature (MRT)

Mean Radiant Temperature (MRT) is a biometeorological metric that quantifies the net radiant heat exchange between the human body and its surrounding environment, integrating contributions from all directions—above, below, and laterally. It represents the uniform temperature of an imaginary enclosure in which a person would experience the same radiant heat gain or loss as in the actual, non-uniform environment. MRT is calculated by combining measurements of shortwave solar radiation and longwave thermal radiation from surfaces such as buildings, roads, trees, and the sky, weighted according to their angular distribution relative to a standing human form. Unlike air temperature, MRT reflects the physical reality of radiant heat transfer, which dominates thermal sensation in outdoor settings, particularly under direct sunlight or near heat-retaining surfaces. It serves as the foundational input for advanced thermal comfort indices such as the Physiological Equivalent Temperature (PET) and the Universal Thermal Climate Index (UTCI), which are used globally to evaluate human thermal stress. Accurate MRT measurement is therefore essential for understanding how urban form, land use, and vegetation influence human experience of heat.

- discuss importance of MRT in urban climate research

MRT has become an indispensable parameter in urban climate research due to its direct correlation with human thermal comfort, heat-related morbidity, and behavioral adaptation patterns. Studies have demonstrated that MRT outperforms air temperature as a predictor of heat-related mortality, hospital admissions, and outdoor activity levels in hot, arid climates. Researchers have employed MRT to evaluate the efficacy of urban cooling interventions, including tree planting, reflective pavements, and shading structures, revealing that even modest increases in shade coverage can reduce MRT by 15–25°C, translating into substantial improvements in public well-being. Moreover, MRT enables the spatial mapping of thermal risk at fine spatial resolutions, allowing planners to identify “hot spots” within neighborhoods where vulnerable populations are most exposed. The integration of MRT into urban climate models has also improved the accuracy of simulations predicting future heat stress under climate change scenarios, informing the design of resilient infrastructure and adaptive land-use policies. As cities confront escalating heat risks, MRT provides the critical link between environmental physics and human experience, transforming abstract climate data into actionable insights for public health and urban design.

- highlight need for accurate MRT measurements

Despite its importance, accurate MRT measurement remains a technical and logistical challenge due to the high cost, bulk, and complexity of conventional instrumentation. The gold-standard 6-directional method requires three expensive net radiometers, precise spatial alignment, and extensive calibration, rendering it impractical for widespread deployment. Alternative methods, such as black globe thermometers, offer affordability and portability but suffer from significant inaccuracies due to variations in globe material, color, size, wind exposure, and response time. These errors are compounded in heterogeneous urban environments where shading patterns change rapidly and wind conditions are turbulent. Furthermore, existing systems rarely incorporate real-time data transmission, spatial context, or human activity data, limiting their utility for dynamic, behavior-informed urban management. There is a critical unmet need for a low-cost, compact, and intelligent sensing platform capable of delivering reliable, continuous, and spatially explicit MRT measurements alongside concurrent observations of pedestrian presence and shade utilization—enabling a new generation of data-driven, human-centered urban climate science.

## SUMMARY

- introduce novel IoT weather station (MaRTiny)

The present invention introduces a novel Internet-of-Things (IoT) biometeorological weather station, designated MaRTiny, engineered to autonomously measure and estimate Mean Radiant Temperature (MRT) in outdoor urban environments with unprecedented affordability, portability, and contextual awareness. Unlike existing systems that rely on expensive radiometers or passive thermometers without spatial or behavioral context, MaRTiny integrates a suite of low-cost meteorological sensors with a compact vision system and on-board machine learning inference to deliver real-time, hyperlocal estimates of human thermal exposure. The device is designed for stationary deployment in public spaces such as parks, plazas, and transit corridors, where it continuously records air temperature, relative humidity, wind speed, and globe temperature—all critical inputs for MRT calculation—while simultaneously capturing visual data to detect and classify pedestrian presence and shade exposure. This dual capability enables the system to correlate environmental conditions with human behavior, revealing patterns of thermal adaptation that inform targeted cooling interventions.

- describe MaRTiny's capabilities

MaRTiny is capable of autonomously collecting, processing, and transmitting meteorological and visual data at one-minute intervals without requiring human intervention. Its core functionality includes the estimation of MRT using both conventional empirical models and a novel machine learning algorithm trained to correct for sensor-specific inaccuracies. The system identifies individuals in shaded and sun-exposed areas using deep learning-based object detection and shadow segmentation, enabling the quantification of space use patterns under varying thermal conditions. All data are securely transmitted via MQTT protocol to a cloud-based database, where they can be accessed for real-time monitoring, historical analysis, and urban planning applications. The device operates on a single 20-watt power source, is constructed for under $200 in materials, and is designed for long-term, unattended operation in harsh outdoor environments. Its compact form factor and low power consumption make it scalable for deployment in dense sensor networks across entire cities.

- motivate need for hyperlocal meteorological conditions

Urban microclimates exhibit extreme spatial variability, with temperature and radiant heat conditions differing significantly over distances of less than ten meters due to variations in surface materials, vegetation density, building height, and shading geometry. Conventional weather stations, often located on rooftops or in open fields, provide data that is spatially averaged and temporally coarse, rendering them ineffective for capturing the thermal experiences of pedestrians navigating complex urban terrain. The need for hyperlocal meteorological measurements is therefore paramount to understanding how individuals interact with their thermal environment, how they adapt their movement patterns in response to heat, and where interventions such as shade structures or reflective surfaces would yield the greatest public health benefit. MaRTiny addresses this need by providing measurements at the scale of human activity—on sidewalks, under awnings, and near benches—thereby enabling a granular, behavior-informed understanding of urban heat exposure that was previously unattainable.

- introduce biometeorological sensing devices

Biometeorological sensing devices represent a class of environmental monitoring systems that bridge the gap between physical climate metrics and human physiological response. These devices are not merely instruments for measuring air or surface temperature; they are designed to emulate the thermal experience of the human body by integrating radiant, convective, and evaporative heat exchange dynamics into a single, interpretable metric. MaRTiny exemplifies this paradigm by combining physical sensors that capture environmental conditions with computational vision systems that observe human behavior, thereby creating a feedback loop between environment and activity. This approach transforms passive data collection into active insight generation, enabling cities to move beyond static heat maps toward dynamic, responsive thermal management systems that adapt in real time to changing conditions and human needs.

- describe device components

The MaRTiny system comprises four primary components: a meteorological sensor array, a low-cost globe thermometer, a compact vision system, and an edge computing module. The meteorological array includes a white-cup shielded air temperature probe, a relative humidity sensor, an anemometer for wind speed measurement, and a UV intensity sensor. The globe thermometer is constructed from an acrylic gray ping pong ball housing a precision thermocouple, chosen for its albedo properties that approximate the average reflectivity of human skin and clothing. The vision system consists of a MIPI camera mounted on an NVIDIA Jetson Nano edge device, which runs a deep convolutional neural network for pedestrian detection and shadow segmentation. All components are controlled by an Arduino Uno microcontroller that aggregates sensor readings and communicates via UART with a NodeMCU module equipped with WiFi and MQTT protocol support for secure cloud transmission.

- explain MRT estimation

MRT estimation in MaRTiny is performed through a two-stage process. First, a conventional empirical model calculates MRT from globe temperature, air temperature, wind speed, and humidity using established physical relationships derived from the Stefan-Boltzmann law and convection coefficients. Second, a supervised machine learning model—trained on paired measurements from a high-fidelity reference system—corrects systematic biases introduced by sensor limitations, such as slow thermal response, material inconsistencies, and spatial misalignment. This model ingests the raw sensor inputs as a feature vector and outputs a refined MRT estimate with significantly reduced error. The machine learning component is critical to the system’s accuracy, as it compensates for the inherent imprecision of low-cost sensors without requiring expensive hardware upgrades or complex environmental modeling.

- describe person detection and shade identification

Person detection and shade identification are performed concurrently by the vision system using two deep learning models: YOLOv3 for pedestrian detection and BDRAR for shadow segmentation. YOLOv3 identifies bounding boxes around human figures in each video frame, while BDRAR generates a binary pixel-level map indicating areas of shade. An intersection-over-union (IOU) algorithm then determines whether a detected person overlaps with a shaded region, classifying them as sun-exposed or shaded based on a threshold of 40% overlap in the lower half of the bounding box—a region chosen to reflect the portion of the body most exposed to ground-reflected radiation. This approach enables the system to quantify the number of people utilizing shade without recording identifiable imagery, preserving privacy while generating actionable behavioral data.

- introduce deep learning model for shade detection

The deep learning model for shade detection is based on the Bi-directional Feature Pyramid with Recurrent Attention Residual Modules (BDRAR) architecture, a convolutional neural network specifically designed for high-precision shadow segmentation in complex outdoor scenes. BDRAR processes input images through multiple hierarchical layers that extract both fine-grained local details and global contextual information, allowing it to distinguish between shadows cast by vegetation, buildings, and pedestrians with high accuracy. The recurrent attention mechanism enables the model to iteratively refine its predictions by focusing on ambiguous regions, improving performance under variable lighting, partial occlusion, and textured surfaces. This model is pre-trained on a large dataset of annotated outdoor images and fine-tuned on a custom collection of images captured by MaRTiny, ensuring robustness across diverse urban environments.

- describe vision system capabilities

The vision system of MaRTiny is capable of real-time object detection, shadow mapping, and behavioral classification at a frame rate of four frames per second using a single NVIDIA Jetson Nano edge device. It operates autonomously, processing video streams locally without transmitting raw imagery to the cloud, thereby ensuring privacy compliance. The system outputs only aggregated metrics—such as the number of pedestrians in shade or sun, total pedestrian count, and duration of exposure—making it suitable for public deployment in sensitive areas. The vision module is powered by a MIPI camera with a wide field of view, optimized for outdoor lighting conditions, and integrated into a gstreamer pipeline for efficient data streaming to the Jetson Nano. The entire system is designed for low power consumption, enabling continuous operation on a 20-watt power supply.

- summarize device functionality

In summary, MaRTiny functions as an autonomous, low-cost, and privacy-preserving biometeorological sensing platform that simultaneously measures environmental thermal conditions and quantifies human behavioral responses to those conditions. It estimates Mean Radiant Temperature with enhanced accuracy through machine learning correction, detects and classifies pedestrian presence and shade exposure using deep learning vision models, and transmits aggregated data wirelessly to a cloud database for real-time monitoring and long-term analysis. By integrating meteorological sensing with computer vision, MaRTiny provides a comprehensive, scalable, and ethically designed tool for urban climate research, public health planning, and adaptive infrastructure management.

## DETAILED DESCRIPTION

- introduce patent application scope

This patent application pertains to a novel biometeorological sensing device and associated method for estimating Mean Radiant Temperature (MRT) in outdoor urban environments through the integration of low-cost meteorological sensors, a custom globe thermometer, and an on-board machine learning-powered vision system. The invention encompasses a complete hardware-software system capable of autonomously collecting, processing, and transmitting hyperlocal thermal exposure data alongside pedestrian activity metrics without requiring manual intervention or the transmission of identifiable imagery. The system is designed for stationary deployment in public spaces and is engineered to overcome the limitations of conventional MRT measurement techniques by combining empirical modeling with data-driven error correction. The invention further includes a method for determining the presence of individuals in shaded and sun-exposed areas using deep learning-based object detection and shadow segmentation, enabling the quantification of human thermal behavior patterns at fine spatial and temporal resolutions.

- define globe temperature measurement

Globe temperature measurement in this invention is achieved through the use of a spherical, acrylic gray globe with a diameter of approximately 38 millimeters, housing a high-precision thermocouple sensor. The globe is constructed from a material whose spectral reflectance properties approximate the average albedo of human skin and clothing under typical outdoor conditions, thereby mimicking the radiant heat absorption characteristics of the human body. The globe is mounted on a non-conductive, non-radiative support structure to minimize conductive heat transfer and is oriented to ensure unobstructed exposure to all directions of incoming radiation. Temperature readings are sampled at a frequency of 10 Hz and averaged over one-minute intervals to mitigate noise and capture dynamic thermal fluctuations. The globe’s color and material are selected to optimize MRT estimation accuracy under varying solar angles and atmospheric conditions, and its design is distinct from conventional black or white globes in its specific spectral properties tailored for human thermal emulation.

- describe limitations of MRT measurement methods

Traditional methods for measuring MRT, including the 6-directional net radiometer approach, are prohibitively expensive, bulky, and require expert calibration and installation, rendering them unsuitable for widespread or long-term deployment. Alternative methods utilizing black globe thermometers suffer from significant inaccuracies due to variations in globe size, emissivity, wind speed sensitivity, and response time, particularly under transient cloud cover or in complex urban geometries. Empirical models that convert globe temperature to MRT rely on assumptions about convection coefficients and surface albedo that do not account for local environmental heterogeneity, leading to systematic overestimation during high solar radiation periods and underestimation during low sun angles. Furthermore, existing systems lack contextual awareness of human presence and behavior, preventing the correlation of thermal conditions with actual exposure patterns. These limitations collectively hinder the scalability, reliability, and utility of MRT data for urban planning and public health applications.

- motivate need for low-cost MRT sensing platform

The absence of a low-cost, accurate, and scalable MRT sensing platform has impeded the adoption of human-centric thermal metrics in urban climate policy and public health initiatives. While high-end systems like MaRTy provide precise measurements, their cost and complexity prevent deployment beyond academic research settings. Cities require affordable, deployable, and maintainable sensors that can be installed in hundreds of locations to generate spatially dense datasets capable of informing targeted cooling interventions. A low-cost MRT sensing platform enables citizen science participation, municipal monitoring programs, and real-time thermal risk mapping—all of which are essential for equitable heat adaptation in vulnerable communities. The invention fulfills this need by delivering MRT estimation accuracy comparable to research-grade instruments at a fraction of the cost, using commercially available components and open-source machine learning frameworks.

### Mean Radiant Temperature (MRT) Sensing

- introduce MRT concept

Mean Radiant Temperature (MRT) is a biophysical metric that quantifies the net radiant heat exchange between a human body and its surrounding environment, integrating contributions from all directions of incoming shortwave and longwave radiation. It is not a direct measurement of air temperature but rather a calculated representation of the thermal environment as perceived by the human body, accounting for radiation emitted by surfaces such as pavement, walls, vegetation, and the sky. MRT is expressed in degrees Celsius and serves as the foundational input for thermal comfort indices that predict physiological stress, making it indispensable for assessing outdoor heat exposure in urban environments where radiant heat dominates thermal sensation.

- describe 6-directional method for MRT measurement

The 6-directional method for MRT measurement involves the use of three orthogonal net radiometers, each measuring incoming and outgoing longwave and shortwave radiation from six distinct hemispherical directions: zenith, nadir, and four cardinal azimuths. These directional fluxes are weighted according to the angular distribution of radiation incident on a standing human form, using empirically derived coefficients that reflect the body’s cylindrical geometry. The weighted fluxes are then aggregated using the Stefan-Boltzmann law to compute a single temperature value equivalent to the uniform radiant environment that would produce the same net heat exchange. This method is considered the gold standard for MRT measurement due to its physical rigor and high accuracy but is limited by the high cost of the radiometers, the complexity of installation, and the requirement for precise spatial orientation.

- provide Stefan-Boltzmann Law equation

The Stefan-Boltzmann Law, which underpins the calculation of MRT from radiative fluxes, is expressed as:

\[
MRT = \left( \frac{\sum_{i=1}^{6} (K_i \cdot a_k + L_i \cdot a_l) \cdot W_i}{\sigma} \right)^{1/4}
\]

where \( K_i \) and \( L_i \) represent the directional shortwave and longwave radiation fluxes, respectively; \( a_k \) and \( a_l \) are the absorption coefficients for shortwave and longwave radiation; \( W_i \) are the angular weighting factors for each direction (0.06 for zenith and nadir, 0.22 for lateral directions); and \( \sigma \) is the Stefan-Boltzmann constant, equal to \( 5.67 \times 10^{-8} \, \text{W} \cdot \text{m}^{-2} \cdot \text{K}^{-4} \). This equation enables the conversion of measured radiative fluxes into a single temperature value that reflects the total radiant load experienced by a human body.

- discuss limitations of 6-directional method

Despite its accuracy, the 6-directional method is impractical for widespread deployment due to the high cost of the required net radiometers, which individually exceed $5,000, and the technical expertise needed for calibration and maintenance. The system is also sensitive to misalignment, wind-induced turbulence, and shading from nearby structures, which can introduce significant measurement errors. Furthermore, its bulk and power requirements preclude deployment in dense urban networks or public spaces where continuous, unattended monitoring is required. These limitations restrict its use to research settings and prevent its integration into municipal climate monitoring programs.

- introduce black globe thermometer

A black globe thermometer is a simplified, low-cost alternative to the 6-directional method, consisting of a hollow spherical sensor—typically painted black or gray—containing a temperature probe that equilibrates with the surrounding radiant environment. The globe absorbs incoming radiation and reaches a steady-state temperature that correlates with MRT, modified by convective heat transfer from ambient air and wind. The use of a gray globe, rather than a black one, improves accuracy by approximating the average albedo of human skin and clothing, reducing bias in radiant heat absorption under diverse clothing conditions.

- describe Thorsson's low-cost globe thermometer

Thorsson’s low-cost globe thermometer, developed for field studies in urban environments, employs a standard acrylic gray ping pong ball as the sensing sphere, with a thermocouple embedded at its center. This design reduces cost to less than $100 while maintaining sufficient accuracy for comparative studies. The gray color of the globe is selected to match the average spectral reflectance of human skin and clothing, enabling the device to emulate the radiant heat absorption characteristics of a typical person. The thermometer is mounted on a non-conductive pole to minimize conductive heat transfer and is calibrated against reference MRT measurements to derive empirical conversion models.

- discuss albedo variations

Albedo, or surface reflectivity, varies significantly across individuals due to differences in skin tone, clothing color, and material composition. A black globe, for instance, absorbs nearly all incident radiation, leading to overestimation of MRT for individuals wearing light-colored clothing. Conversely, a white globe underestimates MRT for those in dark attire. The use of an acrylic gray globe mitigates this variability by approximating the average albedo of a mixed population, providing a representative estimate for the general public. However, this approximation introduces systematic bias in extreme cases, such as in environments dominated by highly reflective or absorptive surfaces, necessitating correction through machine learning or environmental calibration.

- introduce convection coefficients for globe thermometers

Convection coefficients govern the rate at which heat is exchanged between the globe’s surface and the surrounding air, and are critical for converting globe temperature into MRT. These coefficients vary with wind speed, globe size, and surface texture, and are typically derived from empirical relationships under controlled conditions. Commonly used coefficients, such as those defined by ISO 7726, assume constant wind conditions and fail to account for the turbulent, variable airflow found in urban canyons. This results in persistent errors in MRT estimation, particularly during low-wind, high-radiation periods when convection is minimal and radiant load dominates.

- provide empirical model for acrylic gray globe temperature

An empirical model for estimating MRT from the temperature of an acrylic gray globe, as developed by Vanos et al., is employed in this invention and expressed as:

\[
MRT = T_g + \frac{1.1 \cdot (T_a - T_g)}{1 + 0.04 \cdot v^{0.5}}
\]

where \( T_g \) is the globe temperature, \( T_a \) is the air temperature, and \( v \) is the wind speed in meters per second. This model accounts for the differential heating between the globe and air, adjusted for convective cooling effects, and has been validated across multiple urban environments. It is used as the baseline for MRT estimation in MaRTiny, with subsequent corrections applied via machine learning to account for sensor-specific deviations.

- discuss limitations of existing MRT measurement methods

Existing MRT measurement methods are constrained by a fundamental trade-off between accuracy and accessibility. High-precision instruments are too costly and complex for broad deployment, while low-cost alternatives lack the reliability needed for scientific or policy applications. No existing system integrates real-time human behavior data with thermal measurements, preventing the development of adaptive, behavior-informed urban cooling strategies. Furthermore, most systems do not correct for sensor drift, environmental shading, or material inconsistencies, leading to unquantified errors that accumulate over time. The absence of a unified, scalable, and self-correcting platform has hindered the adoption of MRT as a standard metric in urban climate monitoring.

- introduce microclimate and radiation models

Microclimate and radiation models simulate the distribution of radiant fluxes in urban environments using detailed 3D geometry, surface material properties, and solar trajectory data. These models, often implemented in software such as ENVI-met or Radiance, can predict MRT at fine spatial resolutions but require extensive input data, high computational resources, and expert calibration. They are impractical for real-time monitoring and cannot adapt to dynamic changes in vegetation, pedestrian density, or cloud cover. Their outputs are static and lack the temporal resolution needed to inform immediate public health responses.

- discuss limitations of conventional models

Conventional radiation models are limited by their reliance on idealized assumptions about surface reflectivity, sky view factors, and human posture, which do not reflect the complexity of real-world urban environments. They are unable to account for transient shading from moving objects such as vehicles or people, nor do they incorporate real-time meteorological variability. As a result, their predictions often diverge significantly from measured MRT values in heterogeneous, dynamic settings. Furthermore, these models are not designed for deployment in low-power, edge-computing environments, making them incompatible with the requirements of a scalable, autonomous sensing platform.

- introduce pedestrian counting and crowd estimation techniques

Pedestrian counting and crowd estimation techniques have been developed across multiple domains, including transportation planning, security, and retail analytics. These methods range from sensor-based approaches using infrared or ultrasonic detectors to vision-based techniques employing computer vision algorithms. While effective in controlled environments, most existing systems are not designed for outdoor thermal monitoring and do not account for the influence of radiant heat on human movement patterns. None integrate thermal data with behavioral metrics to enable a holistic understanding of human-environment interactions.

- describe sensor-based techniques

Sensor-based techniques for pedestrian counting rely on physical sensors such as infrared beams, pressure mats, or ultrasonic transducers to detect the passage of individuals. These systems are reliable in controlled environments but are easily disrupted by environmental factors such as wind, rain, or vegetation movement. They are also incapable of distinguishing between individuals and objects, provide no spatial context, and cannot determine whether a person is in shade or sun. Their deployment in outdoor urban settings is therefore limited by high false-positive rates and lack of contextual awareness.

- describe network-based techniques

Network-based techniques utilize data from mobile phone signals, Wi-Fi probes, or Bluetooth beacons to estimate crowd density and movement patterns. While scalable and non-intrusive, these methods suffer from low spatial resolution, inconsistent detection rates due to device usage variability, and privacy concerns related to personal data collection. They also provide no information about environmental conditions or thermal exposure, rendering them unsuitable for biometeorological applications.

- introduce machine learning techniques for crowd estimation

Machine learning techniques for crowd estimation leverage labeled image datasets to train models that recognize and count individuals in complex scenes. These include regression-based models such as Support Vector Machines (SVM) and Artificial Neural Networks (ANN), as well as deep learning architectures such as Convolutional Neural Networks (CNNs). These methods offer superior accuracy in cluttered environments and can be adapted to detect specific object classes, such as pedestrians, bicycles, or umbrellas. When combined with thermal data, they enable the correlation of human behavior with environmental conditions, forming the basis of behavior-informed urban climate science.

- describe low-level image feature extraction methods

Low-level image feature extraction methods, such as Histogram of Oriented Gradients (HOG) and Haar cascades, identify patterns in pixel intensity and edge orientation to detect human forms. These methods are computationally efficient and have been widely used in early pedestrian detection systems. However, they are highly sensitive to lighting variations, occlusion, and changes in clothing, resulting in poor performance in dynamic outdoor environments. Their inability to generalize across diverse urban scenes limits their utility in autonomous, long-term monitoring applications.

- describe regression models and detectors

Regression models such as SVM and ANN are employed to map input sensor features to output MRT values, while detectors like AdaBoost are used to classify pedestrian presence from image features. These models are trained on labeled datasets to learn non-linear relationships between environmental variables and thermal exposure. While effective, they require substantial training data and are prone to overfitting when applied to environments dissimilar to the training conditions. Their integration with vision systems in this invention overcomes these limitations by combining multiple data modalities and applying robust cross-validation protocols.

- introduce deep convolutional neural networks for crowd estimation

Deep convolutional neural networks (CNNs) represent the state-of-the-art in crowd estimation due to their ability to learn hierarchical representations of visual data, from low-level edges to high-level semantic features. Architectures such as YOLOv3 and Faster R-CNN are capable of detecting and counting individuals in dense, cluttered scenes with high precision and recall. In this invention, YOLOv3 is adapted for edge deployment on the NVIDIA Jetson Nano, enabling real-time pedestrian detection at four frames per second without requiring cloud connectivity. The model is trained on a custom dataset of urban scenes to ensure robustness under variable lighting, shadow, and occlusion conditions.

- describe perspective maps

Perspective maps are spatial representations of scene geometry that account for the distortion introduced by camera viewpoint, enabling accurate density estimation in non-uniform fields of view. Techniques such as those developed by Lempitsky and Zisserman use perspective projection to convert pixel counts into real-world population densities, improving accuracy in crowded areas. While powerful, these methods require precise camera calibration and are computationally intensive. In this invention, perspective mapping is not employed, as the system prioritizes simplicity, low power consumption, and privacy preservation over absolute density estimation.

- discuss analysis of crowd behavior in urban areas

Analysis of crowd behavior in urban areas has traditionally relied on manual observation, surveys, or indirect proxies such as traffic volume. Recent advances in computer vision have enabled automated tracking of pedestrian movement, dwell time, and path selection, revealing patterns of thermal adaptation such as route diversion toward shaded corridors. These behavioral insights are critical for designing effective cooling infrastructure, yet they have not been systematically integrated with thermal measurements in prior systems. This invention bridges that gap by simultaneously capturing environmental conditions and human responses, enabling the first automated, continuous, and privacy-preserving analysis of thermal behavior in public spaces.

- discuss relation with thermal comfort

Thermal comfort is determined by the balance between metabolic heat production and environmental heat exchange, with MRT being the dominant factor in outdoor settings. Human behavior—such as seeking shade, altering clothing, or adjusting activity levels—is a direct response to perceived thermal discomfort. Understanding this relationship allows cities to design interventions that align with actual human needs rather than theoretical models. By quantifying how many people utilize shade under varying MRT conditions, this invention provides actionable data for optimizing the placement of trees, canopies, and other cooling structures, thereby enhancing public health outcomes.

- introduce MaRTiny system

The MaRTiny system is a fully integrated, low-cost, autonomous biometeorological sensing platform designed to measure Mean Radiant Temperature and quantify pedestrian behavior in outdoor urban environments. It combines meteorological sensors, a custom gray globe thermometer, a compact vision system, and an edge computing module into a single, scalable device capable of continuous, unattended operation. The system is engineered for deployment in public spaces such as parks, transit stops, and plazas, where it collects hyperlocal thermal data and behavioral metrics without compromising privacy. Its design enables large-scale network deployment, making it a transformative tool for urban climate research and public health planning.

- describe biometeorological sensing device

The biometeorological sensing device in this invention is a compact, weather-resistant enclosure housing a suite of sensors and computational components designed to measure and interpret the thermal environment as experienced by the human body. It includes a globe thermometer for radiant heat measurement, air and humidity sensors for convective and evaporative conditions, an anemometer for wind speed, and a UV sensor for solar intensity. These sensors are calibrated to operate in tandem, with data aggregated and processed in real time to generate MRT estimates. The device is powered by a single 20-watt source and communicates wirelessly to a cloud database using secure MQTT protocol, ensuring continuous, reliable data transmission.

- introduce weather station

The weather station component of MaRTiny is a self-contained, low-power meteorological sensor array designed for outdoor deployment in urban environments. It is constructed from commercially available, off-the-shelf components selected for their cost-effectiveness, durability, and compatibility with low-voltage operation. The station is mounted on a non-metallic pole to minimize thermal interference and is oriented to ensure unobstructed exposure to ambient conditions. Data are sampled at 10 Hz and averaged over one-minute intervals to reduce noise and capture dynamic thermal fluctuations, ensuring temporal resolution suitable for behavioral analysis.

- describe vision system

The vision system of MaRTiny consists of a MIPI camera mounted on an NVIDIA Jetson Nano edge device, running a deep convolutional neural network for pedestrian detection and shadow segmentation. The camera is optimized for outdoor lighting conditions and captures video at a fixed frame rate of four frames per second. All image processing is performed locally on the device, with only aggregated metrics—such as pedestrian count and shade occupancy—transmitted to the cloud. Raw images are discarded after processing to ensure compliance with privacy regulations. The system operates autonomously, requiring no manual intervention or external data input.

- introduce machine learning module

The machine learning module is a core innovation of this invention, enabling the correction of systematic errors in MRT estimation caused by sensor limitations and environmental variability. It is trained on paired measurements from a high-fidelity reference system and learns a non-linear mapping between raw sensor inputs (air temperature, globe temperature, humidity, wind speed, UV intensity) and ground-truth MRT values. The module employs a Support Vector Machine with a Radial Basis Function kernel, selected for its computational efficiency, robustness to noise, and suitability for edge deployment. This model is embedded on the Jetson Nano and performs inference in real time, producing corrected MRT estimates with an error margin below 4°C RMSE.

- describe IoT weather station

The IoT weather station is a fully connected, cloud-enabled version of the MaRTiny system, incorporating wireless communication, secure authentication, and remote data management. It uses a NodeMCU module with ESP8266 WiFi to transmit data via MQTT protocol to an AWS DynamoDB database, where it is stored, indexed, and made accessible for real-time monitoring and historical analysis. All communication is secured using PEM-encoded certificates stored in the device’s flash memory, ensuring encrypted, authenticated data transmission. The system is designed for plug-and-play deployment, requiring only a power source and WiFi connection to begin operation.

- describe low-cost and compact vision system

The vision system is engineered for minimal power consumption and maximum computational efficiency, utilizing the NVIDIA Jetson Nano’s integrated GPU to run deep learning models at four frames per second while consuming less than 10 watts of power. The system is housed in a compact, IP65-rated enclosure that protects the camera and electronics from dust, moisture, and temperature extremes. The MIPI camera is selected for its small form factor, high dynamic range, and compatibility with the Jetson Nano’s interface, enabling seamless integration without external adapters or power converters.

- describe pedestrian detection and shade detection

Pedestrian detection is performed using the YOLOv3 architecture, which identifies bounding boxes around human figures in each video frame. Shade detection is performed using the BDRAR model, which generates a pixel-level binary mask indicating shaded regions. An intersection-over-union algorithm determines whether a detected pedestrian overlaps with a shaded area, classifying them as sun-exposed or shaded based on a 40% overlap threshold in the lower half of the bounding box. This approach ensures accurate classification while minimizing false positives caused by background shadows or self-occlusion.

- introduce machine learning model for MRT estimation

The machine learning model for MRT estimation is a supervised learning algorithm trained on a dataset of over 12,000 paired measurements from MaRTiny and a reference MRT system. The model takes as input the raw sensor values—air temperature, globe temperature, relative humidity, wind speed, and UV intensity—and outputs a corrected MRT estimate. It is implemented as a Support Vector Machine with a Radial Basis Function kernel, selected for its ability to capture non-linear relationships between variables and its low computational footprint. The model is trained using 5-fold cross-validation and achieves a root mean square error of less than 4°C on independent evaluation data, significantly outperforming conventional empirical models.

- describe error correction and prediction

Error correction is performed in real time by the machine learning model, which continuously adjusts MRT estimates based on learned patterns of sensor deviation. These deviations arise from factors such as slow thermal response of the globe, minor variations in material reflectivity, and spatial misalignment relative to the reference system. The model does not require recalibration under changing environmental conditions, as it generalizes across diverse weather patterns, times of day, and urban geometries. Prediction is enabled through continuous inference on incoming sensor data, allowing the system to generate accurate, real-time MRT values without reliance on external databases or manual intervention.

- introduce FIG. 1

FIG. 1 illustrates the overall architecture of the MaRTiny system, depicting the integration of meteorological sensors, the gray globe thermometer, the vision system, and the edge computing module within a single compact enclosure. The figure shows the flow of data from sensor acquisition through local processing to wireless transmission, highlighting the autonomous, privacy-preserving nature of the system. It also depicts the physical configuration of the device in an outdoor urban setting, demonstrating its suitability for deployment on sidewalks, plazas, and park pathways.

- describe BSD components

The biometeorological sensing device (BSD) components include a downward-facing white cup for air temperature measurement, a gray acrylic globe for radiant heat sensing, a humidity sensor, an anemometer, a UV sensor, an Arduino Uno microcontroller for data aggregation, a NodeMCU module for WiFi communication, and an NVIDIA Jetson Nano for vision processing. All components are mounted on a non-metallic, thermally insulated frame to minimize conductive interference. The device is enclosed in a weather-resistant housing with a transparent dome over the camera and globe to ensure unobstructed environmental exposure.

- describe camera and people detection system

The camera system consists of a MIPI interface camera with a wide-angle lens, mounted at a height of 1.5 meters to capture pedestrian activity at eye level. The camera is oriented to cover a 10-meter radius field of view, ensuring detection of individuals in both shaded and sun-exposed areas. The people detection system runs the YOLOv3 model on the Jetson Nano, outputting bounding boxes for each detected pedestrian. The system is trained on a custom dataset of urban scenes to ensure robust performance under varying lighting, clothing, and occlusion conditions.

- introduce data transmission to external server

Data transmission to an external server is accomplished via the NodeMCU module, which establishes a secure connection to an AWS DynamoDB database using the MQTT protocol. All data packets are encrypted using PEM-encoded certificates stored in the device’s flash memory, ensuring authentication and data integrity. Only aggregated metrics—such as MRT, air temperature, humidity, wind speed, pedestrian count, and shade occupancy—are transmitted; raw images are deleted after processing to preserve privacy. Transmission occurs at one-minute intervals, ensuring real-time data availability without overwhelming network bandwidth.

- describe power source and cost-effectiveness

The MaRTiny system is powered by a single 20-watt DC adapter operating at 5 volts and 4 amperes, which is distributed among the various components according to their power requirements. The Arduino Uno consumes less than 1 watt, the NodeMCU less than 2 watts, the Jetson Nano operates at 10 watts in high-performance mode, and the camera and sensors consume approximately 5 watts. The total system cost, including all components, housing, and wiring, is under $200, making it orders of magnitude more affordable than existing research-grade systems. This cost-effectiveness enables large-scale deployment across urban networks, democratizing access to high-quality thermal exposure data.

- introduce FIG. 2

FIG. 2 presents a detailed schematic of the sensor configuration and data flow within the MaRTiny system. It illustrates the physical layout of the meteorological sensors, the globe thermometer, the camera, and the microcontrollers, along with the electrical connections and communication pathways. The figure also depicts the serial communication between the Arduino Uno and NodeMCU, the data transmission via MQTT to AWS, and the local processing pipeline on the Jetson Nano. This diagram serves as a comprehensive reference for replication and deployment of the system.

- describe sensor configuration

The sensor configuration is optimized for minimal interference and maximum accuracy. The air temperature probe is housed in a white, ventilated cup to shield it from direct solar radiation, while the globe thermometer is mounted on a non-conductive arm extending above the housing to ensure unobstructed exposure to all directions of radiation. The humidity sensor is placed in a shaded, ventilated compartment to prevent moisture condensation. The anemometer is positioned at the top of the structure to capture undisturbed wind flow. All sensors are calibrated against NIST-traceable references prior to deployment, and their outputs are synchronized to a common time stamp for accurate correlation.

- describe data collection and transmission

Data collection occurs continuously at a sampling rate of 10 Hz for all sensors, with values averaged over one-minute intervals to reduce noise and capture thermal dynamics. The Arduino Uno aggregates sensor readings and transmits them via UART to the NodeMCU, which packages the data into JSON-formatted messages and transmits them via MQTT to the AWS DynamoDB database. Transmission occurs every minute, with each message containing timestamped values for MRT, air temperature, humidity, wind speed, UV intensity, pedestrian count, and shade occupancy. Data are stored in a time-series format for longitudinal analysis and are accessible via API for real-time monitoring and visualization.

- introduce machine learning model for MRT estimation

The machine learning model for MRT estimation is a supervised learning algorithm trained on paired measurements from MaRTiny and a high-fidelity reference system. It ingests five input features—air temperature, globe temperature, relative humidity, wind speed, and UV intensity—and outputs a corrected MRT value. The model is implemented as a Support Vector Machine with a Radial Basis Function kernel, selected for its ability to model non-linear relationships and its low computational overhead. Training is performed using 5-fold cross-validation on a dataset of over 12,000 data points, achieving a root mean square error of less than 4°C on independent test data.

- describe vision system capabilities

The vision system is capable of real-time pedestrian detection and shade classification at four frames per second using a single NVIDIA Jetson Nano. It processes video streams locally, eliminating the need for cloud-based image transmission and preserving individual privacy. The system outputs only aggregated metrics, such as the number of pedestrians in shade and sun, without storing or transmitting identifiable imagery. The camera is optimized for outdoor lighting conditions and operates reliably under direct sunlight, overcast skies, and twilight conditions.

- introduce NVIDIA Jetson Nano

The NVIDIA Jetson Nano is a low-power, edge computing device featuring a 128-core Maxwell GPU and a quad-core ARM Cortex-A57 processor, designed for running deep learning models in resource-constrained environments. It is selected for its ability to execute YOLOv3 and BDRAR models at sufficient frame rates while consuming less than 10 watts of power. The device supports the TensorRT inference engine, which optimizes neural network performance for deployment on embedded platforms. Its compact size, low cost, and open software ecosystem make it ideal for integration into autonomous sensing systems.

- describe MIPI camera and gstreamer pipeline

The MIPI camera is a compact, high-dynamic-range imaging sensor with a 1280×720 resolution and a 90-degree field of view, optimized for outdoor operation. It connects directly to the Jetson Nano via a MIPI CSI-2 interface, enabling high-bandwidth, low-latency video streaming. The gstreamer pipeline is configured to capture, encode, and preprocess video frames in real time, feeding them directly into the YOLOv3 and BDRAR models without intermediate storage. This architecture ensures minimal processing delay and maximizes system responsiveness.

- introduce AWS database and MQTT protocol

The AWS DynamoDB database is a fully managed, serverless NoSQL database that stores time-series meteorological and behavioral data collected by MaRTiny. Data are indexed by timestamp and location, enabling efficient querying and visualization. The MQTT protocol is used for lightweight, publish-subscribe communication between the device and the cloud, ensuring reliable data transmission even under intermittent network conditions. All communication is secured using TLS encryption and PEM-encoded certificates stored in the NodeMCU’s flash memory, ensuring authentication and data integrity.

- describe on-board machine vision capabilities

The on-board machine vision capabilities of MaRTiny enable fully autonomous operation without reliance on external computing resources. The Jetson Nano performs all image processing locally, including pedestrian detection, shadow segmentation, and classification of shade exposure. Raw video frames are discarded after analysis, ensuring privacy compliance. The system is designed to operate continuously for months without maintenance, making it suitable for deployment in remote or public locations where human oversight is impractical.

## EXAMPLES

### System Overview

- introduce MaRTiny system

The MaRTiny system is a fully integrated, low-cost, autonomous biometeorological sensing platform designed to measure Mean Radiant Temperature and quantify pedestrian behavior in outdoor urban environments. It combines meteorological sensors, a custom gray globe thermometer, a compact vision system, and an edge computing module into a single, scalable device capable of continuous, unattended operation. The system is engineered for deployment in public spaces such as parks, transit stops, and plazas, where it collects hyperlocal thermal data and behavioral metrics without compromising privacy. Its design enables large-scale network deployment, making it a transformative tool for urban climate research and public health planning.

- describe system functionality

The system functions by continuously sampling meteorological parameters—air temperature, relative humidity, wind speed, and globe temperature—at 10 Hz and averaging them over one-minute intervals. Globe temperature is converted to MRT using an empirical model, which is then refined by a machine learning algorithm trained on paired measurements from a reference system. Simultaneously, a vision system captures video frames, detects pedestrians using YOLOv3, segments shadows using BDRAR, and classifies individuals as sun-exposed or shaded based on spatial overlap. All processed data—MRT, environmental conditions, pedestrian count, and shade occupancy—are transmitted securely to a cloud database via MQTT protocol. Raw images are deleted after processing to ensure privacy.

- detail sensor components

The sensor components include a white-cup shielded air temperature probe, a relative humidity sensor, an anemometer, a UV sensor, and a gray acrylic globe thermometer with an embedded thermocouple. The globe is constructed from an acrylic ping pong ball with a spectral reflectance profile approximating the average albedo of human skin and clothing. All sensors are mounted on a non-metallic, thermally insulated frame to minimize conductive interference and ensure accurate radiant heat measurement.

- explain globe thermometer functionality

The globe thermometer functions by absorbing incoming shortwave and longwave radiation from all directions, reaching a thermal equilibrium that reflects the net radiant load experienced by a human body. Its gray color and acrylic material are selected to match the average albedo of human skin and clothing, ensuring that the temperature of the globe correlates with the Mean Radiant Temperature experienced by a typical individual. The thermocouple embedded within the globe provides a precise measurement of this equilibrium temperature, which serves as the primary input for MRT estimation.

- describe anemometer functionality

The anemometer measures wind speed using a rotating cup assembly connected to a Hall effect sensor, which converts rotational velocity into an electrical signal. The signal is digitized and averaged over one-minute intervals to account for gusts and turbulence. Wind speed is a critical input for convection modeling, as it influences the rate of heat exchange between the globe and ambient air, thereby affecting the accuracy of MRT estimation.

- detail camera and vision system

The camera is a MIPI interface device with a 1280×720 resolution and a 90-degree field of view, mounted at a height of 1.5 meters to capture pedestrian activity. The vision system runs on an NVIDIA Jetson Nano, which executes YOLOv3 for pedestrian detection and BDRAR for shadow segmentation. Both models are optimized for edge deployment using TensorRT, enabling real-time inference at four frames per second. Only aggregated metrics are transmitted; raw images are discarded after processing.

- describe Jetson Nano and its functionality

The NVIDIA Jetson Nano is an edge computing device that provides the computational power necessary to run deep learning models locally without cloud dependency. It features a 128-core GPU and a quad-core ARM processor, enabling the simultaneous execution of YOLOv3 and BDRAR at sufficient frame rates. The device is configured to operate at 10 watts to balance performance and power consumption. It runs a Linux-based operating system with Python and TensorFlow libraries, and is programmed to process video streams, extract features, and transmit results via MQTT.

- explain Arduino Uno and NodeMCU microcontrollers

The Arduino Uno serves as the primary data acquisition controller, reading sensor values at 10 Hz and calculating one-minute averages. It communicates with the NodeMCU via UART serial protocol, transmitting aggregated data packets every minute. The NodeMCU, based on the ESP8266 chip, provides WiFi connectivity and handles secure communication with the AWS cloud using the MQTT protocol. It stores authentication certificates in its flash memory and manages data transmission, ensuring reliable, encrypted communication even under intermittent network conditions.

- detail communication protocol between components

The communication protocol between components is hierarchical and asynchronous. The Arduino Uno reads sensor data and sends it via UART to the NodeMCU, which buffers and timestamps the data. The Jetson Nano processes video independently and sends its outputs to the NodeMCU via a separate serial channel. The NodeMCU consolidates all data into a single JSON packet and transmits it via MQTT to the AWS DynamoDB database. All communication is synchronized to a common clock source to ensure temporal alignment of meteorological and behavioral data.

- describe data transmission to cloud database

Data transmission occurs every minute via MQTT protocol to an AWS DynamoDB database, where each record is indexed by timestamp, device ID, and geographic coordinates. The transmitted data includes MRT (corrected), air temperature, humidity, wind speed, UV intensity, pedestrian count, number of pedestrians in shade, and number of pedestrians in sun. Raw images are never transmitted. All communication is secured using TLS encryption and PEM-encoded certificates stored in the NodeMCU’s flash memory, ensuring authentication and data integrity.

### Machine Learning Algorithm Development

- motivate machine learning approach

The machine learning approach is motivated by the systematic errors introduced by low-cost sensors, including slow thermal response, material inconsistencies, and spatial misalignment relative to reference systems. Conventional empirical models for MRT estimation are unable to account for these deviations, leading to persistent inaccuracies under varying environmental conditions. Machine learning enables the system to learn these error patterns from paired measurements and correct them in real time, significantly improving accuracy without requiring expensive hardware upgrades.

- describe data collection for labeled dataset

The labeled dataset was collected by deploying MaRTiny in tandem with a high-fidelity MaRTy reference system at two outdoor locations in Tempe, Arizona, over a period of three days. Both systems recorded meteorological parameters and MRT values at high temporal resolution. The MaRTy system provided ground-truth MRT values calculated using the 6-directional method, while MaRTiny recorded raw sensor outputs. Over 12,000 paired data points were collected, covering diverse times of day, weather conditions, and solar angles, forming the basis for supervised training.

- detail machine learning models (SVM and ANN)

Two machine learning models were evaluated: a Support Vector Machine with a Radial Basis Function kernel and a traditional Artificial Neural Network with a single hidden layer and ReLU activation. Both models were trained to predict MRT from five input features: air temperature, globe temperature, humidity, wind speed, and UV intensity. The SVM model was implemented using the scikit-learn library, while the ANN was trained using TensorFlow on an i7 CPU. Both models were subjected to 5-fold cross-validation to optimize hyperparameters and prevent overfitting.

- explain model evaluation and selection

Model evaluation was performed using root mean square error (RMSE) on independent test and evaluation datasets. The SVM with RBF kernel achieved an RMSE of 3.8°C on the test set and 4.1°C on the evaluation set, outperforming the ANN (4.0°C and 4.2°C, respectively). The SVM was selected for deployment due to its superior generalization, lower computational requirements, and compatibility with edge devices. The model’s performance remained stable across diverse environmental conditions, demonstrating robustness to sensor noise and spatial variability.

### System Evaluation

- describe evaluation dataset

The evaluation dataset consisted of 700 data points collected over a single day at a fixed location in Tempe, Arizona, under clear sky conditions. These data were not used in training and were collected with the MaRTiny system operating independently, without reference to the MaRTy system. The dataset included a range of solar angles, wind speeds, and pedestrian densities, providing a realistic test of the system’s performance under operational conditions.

- detail paired MaRTy and MaRTiny setup

The paired setup involved colocating the MaRTiny and MaRTy systems within 1.5 meters of each other, ensuring identical exposure to ambient conditions. Both systems recorded data simultaneously at high temporal resolution, with MaRTy providing ground-truth MRT values and MaRTiny providing raw sensor inputs. The setup was repeated across multiple days and locations to capture spatial and temporal variability. The physical proximity ensured that environmental conditions were nearly identical, minimizing confounding variables.

- explain MRT calculation and comparison

MRT was calculated for MaRTy using the 6-directional method and the Stefan-Boltzmann equation, while for MaRTiny, MRT was first estimated using the empirical model and then corrected by the machine learning algorithm. The two values were compared using RMSE, mean bias, and correlation coefficient. The uncorrected MaRTiny estimates exhibited an RMSE of 10.0°C, while the corrected estimates achieved an RMSE of 4.1°C, demonstrating a 59% reduction in error.

- describe error in MaRTiny MRT estimation

The primary source of error in uncorrected MaRTiny MRT estimates was spatial misalignment, particularly during morning hours when a nearby palm tree partially shaded the globe thermometer while the MaRTy radiometers remained sun-exposed. This resulted in a systematic underestimation of MRT during early hours. The machine learning model learned to compensate for this bias, effectively correcting for environmental shading effects without requiring manual intervention.

- motivate supervised learning approach

The supervised learning approach was motivated by the inability of physics-based models to account for sensor-specific deviations and environmental heterogeneity. By training the model on paired measurements from a high-fidelity reference system, the invention captures the true relationship between sensor inputs and actual MRT, regardless of the underlying physical assumptions. This approach transforms a low-cost sensor into a high-accuracy measurement tool, enabling scalable deployment without sacrificing reliability.

- detail training and testing datasets

The training dataset comprised 12,000 paired data points collected over three days at two locations, while the testing dataset contained 3,000 points from a separate time period. The evaluation dataset consisted of 700 points from a single day at a new location, ensuring generalizability. All datasets were randomly shuffled and stratified by time of day and solar angle to prevent bias. The model was trained using 80% of the training data and validated on the remaining 20%.

- explain model evaluation metrics (RMSE)

Root Mean Square Error (RMSE) was selected as the primary evaluation metric due to its sensitivity to large errors and its widespread use in meteorological and environmental modeling. RMSE was calculated as the square root of the mean squared difference between predicted and observed MRT values. Lower RMSE values indicate higher accuracy. The SVM model achieved an RMSE of 4.1°C on the evaluation dataset, significantly outperforming the empirical model (10.0°C) and the ANN (4.2°C).

- compare SVM and ANN performance

The SVM with RBF kernel achieved slightly better performance than the ANN on both test and evaluation datasets, with RMSE values of 3.8°C and 4.1°C, respectively, compared to 4.0°C and 4.2°C for the ANN. The SVM also demonstrated greater robustness to noise and required less computational power, making it more suitable for edge deployment. The ANN, while slightly more accurate on the test set, exhibited higher variance on the evaluation set, indicating potential overfitting.

- describe object detection using YOLOv3

Object detection was performed using the YOLOv3 architecture, trained on the Microsoft COCO dataset and fine-tuned on a custom dataset of urban pedestrian images. The model achieved a mean Average Precision (mAP) of 55% at an IOU threshold of 0.5, with an Average Precision of 85% for the pedestrian class. The model was optimized for edge deployment using TensorRT, enabling inference at four frames per second on the Jetson Nano.

- detail shade detection using BDRAR

Shade detection was performed using the BDRAR network, a deep convolutional neural network designed for shadow segmentation. The model was pre-trained on a large dataset of outdoor images and fine-tuned on 30 manually annotated images captured by MaRTiny. It achieved a pixel-level Intersection over Union (IOU) of 90%, demonstrating high accuracy in identifying shaded regions under varying lighting and surface conditions.

- explain evaluation metrics for object detection (mAP)

Mean Average Precision (mAP) was calculated by averaging precision-recall curves across all object classes at multiple IOU thresholds ranging from 0.5 to 0.95. The mAP score reflects the model’s ability to correctly detect and localize pedestrians across varying levels of confidence and overlap. A higher mAP indicates better detection performance. The system achieved an mAP of 55%, which is sufficient for the application given the system’s focus on aggregated metrics rather than individual identification.

- describe evaluation metrics for shade detection (IOU)

Intersection over Union (IOU) was used to evaluate shade detection by comparing the predicted binary shadow map with the manually annotated ground truth. IOU is calculated as the area of overlap divided by the area of union between the predicted and true masks. An IOU of 90% indicates that 90% of the shaded pixels were correctly identified, with minimal false positives or negatives. This metric is appropriate for pixel-level segmentation tasks and reflects the model’s fidelity in capturing complex shadow boundaries.

- detail pedestrian in shade detection algorithm

The pedestrian in shade detection algorithm calculates the IOU between the bounding box of each detected pedestrian and the binary shadow map generated by BDRAR. Only the lower 50% of the bounding box is considered, as this region corresponds to the portion of the body most exposed to ground-reflected radiation. A person is classified as shaded if the IOU exceeds 40% of the bounding box area, corresponding to an 80% overlap in the lower ROI. This threshold was empirically determined to maximize classification accuracy while minimizing false positives from background shadows.

- explain evaluation results for pedestrian in shade detection

The pedestrian in shade detection algorithm achieved an overall accuracy of 80% when evaluated against manually annotated ground truth from 50 images. The system correctly classified individuals as shaded or sun-exposed in 40 out of 50 cases. Errors occurred primarily when pedestrians were partially occluded or when shadows were cast by moving objects such as vehicles. The algorithm’s performance demonstrates its suitability for real-world deployment, providing a reliable basis for behavioral analysis.

### Discussion

- summarize MaRTiny system

The MaRTiny system is a novel, low-cost, autonomous biometeorological sensing platform that simultaneously measures Mean Radiant Temperature and quantifies pedestrian behavior in outdoor urban environments. It integrates meteorological sensors, a custom gray globe thermometer, a compact vision system, and an edge computing module into a single, scalable device capable of continuous, unattended operation. The system delivers accurate, privacy-preserving thermal exposure data and behavioral metrics, enabling data-driven urban planning and public health interventions.

- motivate empirical study

An empirical study is motivated by the need to validate the system’s performance across diverse climates, seasons, and urban typologies. While the current evaluation demonstrates accuracy under clear-sky conditions in a desert climate, long-term deployment in humid, cloudy, or mixed-landscape environments is required to establish generalizability. Future studies will collect data over a full year to capture seasonal variability and refine the machine learning model accordingly.

- discuss calibration requirements

Calibration of the MaRTiny system requires alignment with NIST-traceable reference sensors prior to deployment. While the machine learning model corrects for sensor-specific deviations, initial calibration ensures that raw sensor outputs are within acceptable tolerances. Calibration must be performed for each sensor type—air temperature, humidity, wind speed, and globe temperature—using standardized procedures to ensure data quality and comparability across networks.

- compare MRT estimation errors

The uncorrected MRT estimation error using the empirical model was 10.0°C, while the machine learning-corrected error was reduced to 4.1°C. This represents a 59% improvement in accuracy, bringing MaRTiny’s performance within the range of published errors for research-grade globe thermometers. The system’s accuracy is now sufficient for use in public health monitoring and urban planning applications.

- discuss limitations of globe thermometers

Globe thermometers are inherently indirect sensors, sensitive to globe size, material, color, wind speed, and response time. They tend to overestimate MRT during high solar radiation and underestimate it during low sun angles. Their slow thermal response limits their ability to capture rapid changes in cloud cover or shading. These limitations are mitigated in MaRTiny through machine learning correction, but they remain inherent to the physical design of the globe.

- motivate low-cost sensing

Low-cost sensing is motivated by the need to democratize access to high-quality thermal exposure data. Existing systems are prohibitively expensive for municipal deployment, limiting their use to academic research. MaRTiny enables cities to build dense sensor networks at a fraction of the cost, empowering communities to monitor and mitigate heat risk in real time.

- describe edge device capabilities

The edge device capabilities of MaRTiny enable fully autonomous operation without cloud dependency. The Jetson Nano performs all image processing and MRT correction locally, ensuring low latency, high reliability, and strong privacy compliance. The system can operate for months without maintenance, making it ideal for deployment in remote or public locations.

- explain system deployment plans

Deployment plans include installation of MaRTiny units in public parks, transit hubs, and schoolyards in the City of Tempe, Arizona, with the goal of creating a network of 50 devices across the urban landscape. Data will be used to inform the placement of shade structures, tree planting initiatives, and public cooling centers. The system will be open-sourced to encourage replication and adaptation by other municipalities.

- discuss privacy preservation

Privacy is preserved by transmitting only aggregated metrics—such as pedestrian count and shade occupancy—without storing or transmitting raw images. All video processing occurs on-board, and images are deleted immediately after analysis. The system complies with GDPR and other international privacy regulations, ensuring ethical deployment in public spaces.

- describe potential system modifications

Potential modifications include the addition of a rain sensor, infrared temperature sensor for surface temperature, or a solar radiation pyranometer for direct shortwave measurement. Future versions may incorporate GPS for mobile deployment or cellular connectivity for areas without WiFi. The machine learning model can be retrained to adapt to new environments or sensor configurations.

- conclude system implementation

The MaRTiny system represents a significant advancement in biometeorological sensing, enabling accurate, scalable, and privacy-preserving measurement of human thermal exposure in urban environments. Its integration of low-cost sensors, edge computing, and machine learning transforms a previously inaccessible metric into a practical tool for public health and urban planning. The system is ready for deployment and offers a replicable model for cities worldwide seeking to mitigate the growing threat of urban heat.