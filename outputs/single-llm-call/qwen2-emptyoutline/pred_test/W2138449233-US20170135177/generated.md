# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to smart lighting systems and, more particularly, to methods and systems for estimating occupancy distribution in an indoor space using color-controllable LEDs and sparsely distributed color sensors. The invention provides a novel approach to occupancy sensing that is both privacy-preserving and energy-efficient, suitable for various indoor environments such as offices, homes, and industrial settings.

## BACKGROUND

Modern lighting systems are increasingly moving towards energy efficiency and smart functionalities. Traditional lighting solutions, such as incandescent and fluorescent bulbs, are being replaced by more advanced technologies like Light-Emitting Diodes (LEDs). These LEDs are not only more energy-efficient but also offer the capability to be controlled and modulated in real-time. This has led to the development of smart lighting systems that can adjust the lighting based on the occupancy and activity in a space.

Smart lighting systems typically use sensors to detect the presence and movement of occupants. These sensors can be broadly categorized into imaging sensors, such as cameras and depth sensors, and non-imaging sensors, such as passive infrared (PIR) sensors, ultrasonic sensors, and color sensors. While imaging sensors provide detailed and high-resolution information, they raise significant privacy concerns. Non-imaging sensors, on the other hand, offer a balance between functionality and privacy but are limited in the amount of information they can provide.

The emergence of modern LEDs, which can be rapidly modulated and controlled over multiple color channels, opens up new possibilities for occupancy sensing. By modulating the light and measuring the changes in the sensor output, it is possible to construct a light transport matrix that captures the spatial distribution of light in the room. This matrix can then be used to estimate the occupancy distribution, allowing the lighting system to adjust the lighting condition accordingly.

However, the problem of estimating the occupancy distribution from a limited number of non-imaging color sensors is highly ill-posed and challenging. The present invention addresses this challenge by proposing a novel method that combines perturbation-modulated lighting with advanced signal processing techniques to accurately estimate the occupancy distribution while preserving privacy.

## SUMMARY

The present invention provides a method and system for estimating occupancy distribution in an indoor space using color-controllable LEDs and sparsely distributed color sensors. The method involves modulating imperceptible perturbations onto the light and measuring the changes in the sensor output to recover a light transport matrix. Two approaches, based on the light blockage model and the light reflection model, are proposed to estimate the occupancy distribution using the light transport matrix.

In one aspect, the invention provides a method for estimating occupancy distribution in an indoor space, comprising the steps of:
1. Modulating imperceptible perturbations onto the light emitted by color-controllable LEDs.
2. Measuring the changes in the sensor output from sparsely distributed color sensors.
3. Constructing a light transport matrix based on the measured changes.
4. Estimating the occupancy distribution using the light transport matrix, wherein the estimation is based on either a light blockage model or a light reflection model.

In another aspect, the invention provides a system for estimating occupancy distribution in an indoor space, comprising:
1. A plurality of color-controllable LEDs configured to emit light with modulated perturbations.
2. A plurality of sparsely distributed color sensors configured to measure the changes in the sensor output.
3. A processor configured to construct a light transport matrix based on the measured changes and estimate the occupancy distribution using the light transport matrix, wherein the estimation is based on either a light blockage model or a light reflection model.

The invention further provides a method for perturbation ordering to maximize human comfort during the sensing stage, involving the use of a genetic algorithm to find the optimal sequence of perturbation patterns.

The invention offers several advantages, including:
- Privacy preservation: The use of non-imaging color sensors ensures that no detailed images or personal information is captured.
- Energy efficiency: The lighting system can adjust the lighting condition based on the estimated occupancy distribution, reducing energy consumption.
- Flexibility: The system can be adapted to various indoor environments, including offices, homes, and industrial settings.
- Real-time performance: The use of fast LEDs and rapid-response color sensors allows for real-time occupancy sensing and lighting adjustment.

## DETAILED DESCRIPTION

### Rank Minimization

The problem of estimating the occupancy distribution from a limited number of non-imaging color sensors is inherently ill-posed. The light transport matrix, which captures the spatial distribution of light in the room, is a key component in this estimation process. To address the ill-posed nature of the problem, the present invention employs rank minimization techniques.

Rank minimization is a mathematical approach that seeks to find the lowest-rank matrix that fits the given data. In the context of the present invention, the light transport matrix \( A \) is a low-rank matrix because the number of fixtures and sensors is relatively small compared to the dimensionality of the space. By minimizing the rank of \( A \), the invention can effectively capture the essential spatial information while filtering out noise and irrelevant details.

The rank minimization problem can be formulated as:
\[
\min_{A} \text{rank}(A) \quad \text{subject to} \quad Y = AX
\]
where \( Y \) is the matrix of measured changes in the sensor output, and \( X \) is the matrix of perturbation patterns applied to the LEDs. This problem is NP-hard, but it can be relaxed to a convex optimization problem using the nuclear norm:
\[
\min_{A} \|A\|_* \quad \text{subject to} \quad Y = AX
\]
where \( \|A\|_* \) denotes the nuclear norm of \( A \), which is the sum of the singular values of \( A \).

### Perturbation-Modulated Lighting

The core of the present invention is the use of perturbation-modulated lighting to estimate the occupancy distribution. The method involves modulating imperceptible perturbations onto the light emitted by color-controllable LEDs and measuring the changes in the sensor output from sparsely distributed color sensors.

The perturbation patterns are designed to be rich in variation to capture sufficient information from the scene while being small enough not to bother the occupants. The magnitude of the perturbation is carefully chosen to balance the need for accurate measurements with the need for human comfort.

The perturbation patterns are applied in a specific order to maximize human comfort. This is achieved using a genetic algorithm to find the optimal sequence of perturbation patterns. The genetic algorithm optimizes the sequence by minimizing the total change in the lighting condition between consecutive perturbations, which helps to reduce the perceptibility of the perturbations.

### Analysis of the Light Transport Matrix

The light transport matrix \( A \) is a fundamental component of the present invention. It captures the spatial distribution of light in the room and is used to estimate the occupancy distribution. The matrix \( A \) is constructed by applying perturbation patterns to the LEDs and measuring the changes in the sensor output.

The light transport matrix \( A \) can be expressed as:
\[
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{bmatrix}
\]
where \( a_{ij} \) represents the response of the \( j \)-th sensor to the \( i \)-th perturbation pattern applied to the LEDs.

The light transport matrix \( A \) is independent of the ambient light and is only dependent on the light transport of the scene, such as diffuse reflection, specular reflection, interreflection, and refraction. By analyzing the matrix \( A \), it is possible to extract spatial information about the scene and estimate the occupancy distribution.

### Volume Rendering

The present invention proposes two approaches for estimating the occupancy distribution using the light transport matrix: the light blockage model and the light reflection model.

#### Light Blockage Model

The light blockage model is based on the assumption that the color sensors are installed on the walls of the room. The method involves constructing a difference matrix \( E \) by subtracting the light transport matrix of an empty room \( A_0 \) from the current light transport matrix \( A \):
\[
E = A_0 - A
\]
Each entry of \( E \) corresponds to one fixture channel and one sensor channel. A large positive value in \( E \) indicates that the total flux is significantly attenuated, suggesting that the corresponding direct path from the fixture to the sensor is likely blocked by an occupant.

The difference matrix \( E \) is aggregated to an \( N_S \times N_L \) matrix \( \hat{E} \), where \( N_S \) is the number of sensors and \( N_L \) is the number of fixtures. The aggregation is performed by summing the entries of \( E \) over the color channels, weighted by the sensitivity of the sensors on different color channels.

The 3D reconstruction algorithm estimates the confidence that a point in the 3D space is occupied by considering the point-to-line distance from the point to the direct paths from the fixtures to the sensors. The confidence is computed using a Gaussian kernel:
\[
C(P) = \frac{\sum_{i=1}^{N_S} \sum_{j=1}^{N_L} \hat{E}_{ij} \cdot G(d_{ij}(P), \sigma)}{\sum_{i=1}^{N_S} \sum_{j=1}^{N_L} G(d_{ij}(P), \sigma)}
\]
where \( G(\cdot, \cdot) \) is the Gaussian kernel, \( d_{ij}(P) \) is the point-to-line distance from point \( P \) to the direct path from fixture \( j \) to sensor \( i \), and \( \sigma \) is a parameter that controls the smoothness of the occupancy distribution.

#### Light Reflection Model

The light reflection model is based on the assumption that the color sensors are installed on the ceiling of the room. The method involves modeling the light transport for ceiling-mounted sensors using geometrical optics and photometry analysis.

The light transport matrix \( A \) is again used to construct a difference matrix \( E \). The difference matrix \( E \) is aggregated to an \( N_S \times N_L \) matrix \( \hat{E} \), similar to the light blockage model. However, in this case, the direct paths from the fixtures to the sensors are replaced by reflection paths from the floor.

The 2D confidence map is computed by summing the reflection kernels of all fixture-sensor pairs, weighted by the entries of \( \hat{E} \):
\[
C(x, y) = \sum_{i=1}^{N_S} \sum_{j=1}^{N_L} \hat{E}_{ij} \cdot R_{ij}(x, y)
\]
where \( R_{ij}(x, y) \) is the reflection kernel for fixture \( j \) and sensor \( i \), which is pre-computed based on the geometry of the room and the properties of the floor surface.

The 2D confidence map provides a projection of the occupancy distribution onto the floor plane, indicating the regions of the room that are likely occupied.

### Conclusion

The present invention provides a novel method and system for estimating occupancy distribution in an indoor space using color-controllable LEDs and sparsely distributed color sensors. The method combines perturbation-modulated lighting with advanced signal processing techniques to accurately estimate the occupancy distribution while preserving privacy. The invention offers several advantages, including energy efficiency, flexibility, and real-time performance, making it suitable for various indoor environments.