# DESCRIPTION

## PRIORITY APPLICATIONS

This application claims the benefit of U.S. Provisional Application No. 63/XXX/XXX, filed on [Date], which is hereby incorporated by reference in its entirety.

## BACKGROUND

### Field of the Invention

The present invention relates generally to the field of virtual production (VP) and, more specifically, to methods and systems for improving color rendition in RGB LED-based virtual production stages.

### Description of Related Art

Virtual Production (VP) has revolutionized the film and television industry by enabling filmmakers to create realistic and immersive environments using LED stages or "volumes." These stages surround actors with high-resolution imagery, reducing the need for post-production compositing and enhancing the realism of the scenes. One critical aspect of VP is lighting reproduction, where the LED panels are used to simulate the illumination of a given scene. However, RGB LED panels, while effective for displaying a wide gamut of colors, suffer from color rendition challenges, particularly when simulating broad-spectrum lighting. These challenges manifest as color shifts in skin tones, orange materials, cyan materials, and yellow materials, among others.

### Prior Art

Prior approaches to calibrating RGB LED panels for virtual production have focused primarily on ensuring that the in-camera background colors match the intended content. These methods involve capturing and processing calibration images to align the color primaries of the LED panels with those of the camera. However, these techniques do not address the color rendition issues that arise when the panels are used as light sources. As a result, the colors of actors and objects in the scene can appear overly saturated or shifted, leading to a loss of realism.

### Problem Statement

There is a need for a method that improves the color rendition properties of RGB LED panels when used for lighting reproduction in virtual production stages. Specifically, the method should optimize the color rendition for both the in-camera background and the foreground content (e.g., actors, set, costumes) while maintaining the accuracy of the in-camera background colors.

## SUMMARY

The present invention provides a novel technique for the color calibration of RGB LED stages in virtual production environments. The primary goal of the invention is to optimize the color rendition properties of the LED volume acting as a light source, while still maintaining accurate in-camera background colors. This is achieved through the use of a set of linear transformations represented as 3 × 3 color correction matrices.

### Key Features

1. **Pre-Correction Matrix (M)**: A 3 × 3 pre-correction matrix is applied to the displayed content to map the target scene's pixel colors to the LED panel colors, ensuring that they look the same to the motion picture camera.
2. **Post-Correction Matrix (Q)**: A 3 × 3 post-correction matrix is applied to the final recorded image to make a photographed color chart lit by the VP stage look as close as possible to how it appeared in the real scene.
3. **In-Camera-Frustum Pre-Correction Matrix (N)**: A different 3 × 3 pre-correction matrix is applied to the in-camera-frustum content to maintain the appearance of the in-camera background pixels while optimizing color rendition for the actors and other foreground content.

### Advantages

- **Improved Color Rendition**: The invention significantly reduces color shifts and over-saturation, leading to more realistic and accurate color representation in the final footage.
- **Maintained Background Accuracy**: The in-camera background colors remain accurate, ensuring that the virtual environment blends seamlessly with the actors and objects in the scene.
- **Simplified Workflow**: The method requires only four calibration images and basic shaders, making it straightforward to implement in existing virtual production systems.

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS

### Example Embodiments

#### Overview of the Method

The invention involves a series of steps to calibrate the RGB LED panels for optimal color rendition in virtual production stages. The method can be summarized as follows:

1. **Solve for Pre-Correction Matrix (M)**: Compute a 3 × 3 pre-correction matrix that maps the target scene's pixel colors to the LED panel colors, ensuring a metameric illuminant match when a scene's HDRI map is displayed.
2. **Solve for Post-Correction Matrix (Q)**: Compute a 3 × 3 post-correction matrix that makes a color chart lit by the VP stage look as close as possible to how it appeared in the real-world scene.
3. **Solve for In-Camera-Frustum Pre-Correction Matrix (N)**: Compute a different 3 × 3 pre-correction matrix for the in-camera-frustum content to maintain the appearance of the in-camera background pixels while optimizing color rendition for the actors and other foreground content.

#### Assumptions and Prerequisites

- **Panel and Camera Linearity**: The LED volume is assumed to be calibrated to act as a linear display, and the camera used in the imaging workflow is assumed to have a linear response.
- **Radiometric Alignment of Different Panel Types**: The relative brightness levels of different panel types comprising the LED volume are calibrated to produce a uniform sphere of light of even intensity and color balance from all directions.
- **HDRI Map Acquisition and Display**: The lighting environment to be displayed on the VP stage is captured using HDR panoramic photography techniques, and the VP stage is capable of representing the full dynamic range of the HDRI map without clipping any light sources.

#### Calibration Images and Equations

The method requires four calibration images, all photographed using the target camera to be used for filming in the LED volume.

1. **Primary-Based Calibration**: Capture images of the LED panels displaying pure red, pure green, and pure blue patches. These images are used to compute the pre-correction matrix \( M \).
2. **Color Rendition Calibration**: Capture images of a color chart illuminated by each spectral channel of the LED volume individually. These images are used to compute the post-correction matrix \( Q \).

**Solving for \( M \)**:
\[ [SL] = \begin{bmatrix} S_{R1} & S_{R2} & S_{R3} \\ S_{G1} & S_{G2} & S_{G3} \\ S_{B1} & S_{B2} & S_{B3} \end{bmatrix} \]
\[ M = [SL]^{-1} \]

**Solving for \( Q \)**:
\[ [SRL]_j = \begin{bmatrix} S_{R1}R_j & S_{R2}R_j & S_{R3}R_j \\ S_{G1}R_j & S_{G2}R_j & S_{G3}R_j \\ S_{B1}R_j & S_{B2}R_j & S_{B3}R_j \end{bmatrix} \]
\[ \text{Predicted pixel values} = \frac{1}{\beta} [SRL]_j M w_{avg} \]
\[ \text{Minimize} \sum_{j=1}^{n} \left( Q \left( \frac{1}{\beta} [SRL]_j M w_{avg} \right) - p_j \right)^2 \]

**Solving for \( N \)**:
\[ N = M Q^{-1} \]

#### Black Level Subtraction

A current limitation of LED panels is their non-zero albedo, which causes them to reflect some incident illumination. To compensate for this, the method includes a black level subtraction step. For each lighting environment, the in-camera-frustum pixel values are adjusted with an RGB offset using a "black level" measurement.

#### Experimental Results

The method was tested by reproducing several real-world lighting environments in an RGB LED-based virtual production stage. The results demonstrated significant improvements in color rendition for the lit color charts and the appearance of actors and objects in the scene. The in-camera background colors remained accurate, and the overall quality of the reproduced illumination was enhanced.

#### Theoretical Results

Theoretical results were also generated to simulate the appearance of color charts using the derived matrices. The results showed that the method effectively reduced color errors and improved the overall color rendition properties of the LED panels.

### Conclusion

The present invention provides a robust and efficient method for improving color rendition in RGB LED-based virtual production stages. By using a set of linear transformations represented as 3 × 3 color correction matrices, the method optimizes the color rendition properties of the LED volume acting as a light source while maintaining the accuracy of the in-camera background colors. This innovation addresses a critical challenge in virtual production and enhances the realism and quality of the final footage.