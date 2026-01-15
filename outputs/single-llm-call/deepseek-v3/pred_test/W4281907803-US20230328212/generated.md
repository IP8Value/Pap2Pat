Here is the drafted patent application following the provided outline:

# DESCRIPTION

## PRIORITY APPLICATIONS

The present application claims priority to U.S. Provisional Patent Application No. [XXXXXX], filed [DATE], entitled "Color Rendition Optimization System for LED Volumes," the entire contents of which are hereby incorporated by reference.

## BACKGROUND

LED volumes have become increasingly prevalent in virtual production stages for film and television production. These systems utilize arrays of light-emitting diode (LED) panels to create immersive environments that serve both as dynamic backgrounds and lighting sources. The technology enables cinematographers to capture in-camera visual effects while providing actors with realistic environmental lighting and visual references.

Current LED panel systems face significant limitations in color rendition accuracy. While RGB LED panels can display a wide gamut of colors, their spectral emission characteristics differ substantially from natural light sources. The discrete spectral peaks of red, green, and blue LEDs create gaps in wavelength coverage that lead to inaccurate color reproduction, particularly for materials with specific spectral reflectance properties. This manifests as color shifts in skin tones (toward pink/red), orange materials (toward red), cyan materials (toward blue), and yellow materials (appearing darker).

Traditional color calibration workflows for LED volumes focus primarily on ensuring accurate color reproduction for in-camera background elements. These methods typically involve displaying primary color patches on LED panels and capturing them with the principal photography camera to generate a 3×3 color correction matrix. While effective for display calibration, these approaches fail to address the color rendition challenges that arise when using LED panels as lighting sources for subjects within the volume.

## SUMMARY

The present invention provides a color correction method for LED volumes that simultaneously optimizes color rendition for both lighting reproduction and in-camera background display. The method employs multiple 3×3 color correction matrices applied at different stages of the imaging pipeline to achieve these dual objectives.

The system first generates a pre-correction matrix (M) through primary-based calibration, capturing how each LED primary color appears to the motion picture camera. This matrix ensures metameric illuminant matching when displaying content on out-of-camera-frustum LED panels. A post-correction matrix (Q) is then computed to optimize color rendition for materials illuminated by the LED volume. This matrix is derived from calibration images showing how each spectral channel of the LED system illuminates a reference color chart.

For in-camera-frustum content, the system generates a second pre-correction matrix (N) that combines the effects of matrices M and Q (specifically, N = MQ⁻¹). This maintains accurate color reproduction for background elements displayed on LED panels within the camera's view while preserving the color rendition benefits achieved through matrix Q.

The method further includes techniques for adjusting pixel values through RGB offset compensation to account for light reflected by LED panels (black level measurement). The system solves for the first 3×3 matrix (M) by displaying pure color patches on in-camera-frustum LED panels and analyzing their appearance to the camera. A color calibration image is generated for each spectral channel of the LED volume illuminating the reference color chart, from which average pixel values are sampled.

The system solves for the second 3×3 matrix (Q) by analyzing the relationship between these calibration images and the desired color rendition properties. Calibration imagery is generated under controlled conditions to establish the lighting reproduction characteristics of the LED volume. The system embodiment includes computer-readable instructions for implementing the color correction workflow and combines features from various implementations to optimize performance.

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS

Virtual production stages utilizing LED volumes have revolutionized media production by enabling in-camera visual effects and realistic environmental lighting. These systems typically consist of arrays of RGB LED panels arranged to surround the filming area, creating immersive environments that serve both as dynamic backgrounds and lighting sources. However, the color rendition characteristics of these systems present significant challenges when used for lighting reproduction.

Prior color calibration workflows for LED volumes focused primarily on ensuring accurate color reproduction for displayed background imagery. These approaches fail to address the spectral limitations of RGB LED panels when used as lighting sources, leading to inaccurate color reproduction for illuminated subjects and set pieces. The present invention introduces a color rendition optimization system that overcomes these limitations through a multi-matrix correction approach.

The system corrects color rendition by employing three distinct 3×3 color correction matrices applied at different stages of the imaging pipeline. This approach maintains the benefits of traditional display calibration while significantly improving lighting reproduction accuracy. The system's advantages include simultaneous optimization of both lighting and display color characteristics, minimal additional calibration requirements, and seamless integration with existing virtual production workflows.

Calibration images are generated by displaying specific patterns on the LED panels and capturing them with the principal-photography camera. These include images of pure primary colors displayed on in-camera-frustum panels for matrix M calculation, and images of a color chart illuminated by each spectral channel of the LED volume for matrix Q determination. The system solves for color correction matrices by analyzing the relationships between displayed content, captured imagery, and desired color characteristics.

The correction matrices are applied to different components of the LED volume based on their functional role. Out-of-camera-frustum panels receive content corrected with matrix M to optimize lighting reproduction, while in-camera-frustum panels use matrix N (MQ⁻¹) to maintain accurate background display. The post-correction matrix Q is applied to the final recorded imagery to achieve optimized color rendition.

An example implementation utilizes an RGB LED volume virtual production stage comprising multiple LED panels arranged to provide comprehensive environmental lighting. The system distinguishes between out-of-camera-frustum pixels (those contributing primarily to subject lighting) and in-camera-frustum pixels (those directly visible to the camera as background elements). Camera movement is tracked in real-time to dynamically determine pixel classification.

The color rendition optimization system operates within the virtual production stage, receiving data from both LED panels and the principal photography camera. The system transmits correction data to LED panel controllers and production servers, enabling real-time color optimization. Key assumptions include panel and camera linearity, radiometric alignment between different panel types, and proper acquisition/display of high dynamic range image maps.

Three correction matrices are generated based on calibration imagery. The in-camera-frustum LED panel color calibration image establishes the display characteristics for matrix M. Out-of-camera-frustum color rendition calibration images characterize the lighting reproduction properties for matrix Q determination. These out-of-camera-frustum images can be generated through various methods depending on stage configuration.

The first pre-correction matrix (M) enables metameric illuminant matching by transforming displayed content to appear correct to the camera. Average pixel values are extracted from the primary color calibration image, and the first pre-correction matrix is solved to map these values to expected primary color responses. The post-correction matrix (Q) optimizes color rendition by compensating for spectral limitations in LED lighting.

To determine Q, the system simulates color chart appearance under LED illumination and compares it to target appearance under reference lighting. The relationship between calibration images and full illumination is established through geometric considerations, including defining a scale factor for setup geometry. A cube map environment is constructed to model illumination distribution, with diffuse convolution computed for frontal lighting direction.

Calibration images are scaled by the determined scale factor to predict color chart appearance under uniform illumination. The system focuses on the diffuse integral of illumination, defining this quantity for the high-dynamic range image map. The white balance reference (w_avg) is equated to the pixel value of the white square of the color chart under reference illumination.

FIG. 1 illustrates an RGB LED volume virtual production stage, showing panel arrangement and camera positioning. FIG. 2 presents an environmental diagram of the color rendition optimization system, depicting data flow between components. FIG. 3 provides additional detail on system operation, including matrix computation and application processes. FIG. 4 shows the method flow, from calibration through final color correction.

The system combines features from different implementations, including storing digital media on camera storage and various embodiments of the color correction workflow. Key components include a color chart for reference, calibration data derived from captured images, and the three correction matrices (M, Q, and N). The pre-correction matrix M is applied to out-of-frustum content, while N (MQ⁻¹) corrects in-frustum display.

Color chart appearance is estimated through linear combination of calibration images, scaled appropriately for illumination conditions. The post-correction matrix Q is solved to minimize color difference between predicted and target chart appearance. When applied to final imagery, Q optimizes color rendition for illuminated subjects. The second pre-correction matrix N is derived as MQ⁻¹ and applied to in-frustum content to maintain background color accuracy.

Color rendition optimization addresses inherent limitations of RGB LED panels, particularly their spectral gaps. An RGB offset compensates for panel albedo effects, with black level computed from additional calibration measurements. Alternative arrangements may incorporate different panel configurations or calibration methodologies.

The system architecture includes a pre-correction matrix manager, post-correction matrix manager, and calibration manager that coordinate the color optimization process. A joint optimization approach considers both lighting and display requirements, with constraints on white point maintenance. The system accommodates various virtual production stage types and server architectures, utilizing physical processors and memory for efficient computation.

The method flow diagram illustrates generation of the first pre-correction matrix, post-correction matrix, and second pre-correction matrix in sequence. Color rendition is optimized through coordinated application of these matrices at appropriate pipeline stages. The system corrects color rendition while preserving in-camera background accuracy, overcoming limitations of existing single-matrix approaches.

The importance of accurate color rendition in virtual production cannot be overstated, as it affects both visual quality and artistic intent. Applications extend beyond film production to any scenario requiring faithful color reproduction under LED illumination. Future developments may incorporate additional spectral channels or advanced correction techniques while maintaining the core principles of this invention.

### EXAMPLE EMBODIMENTS

Example 1 describes a computer-implemented method for color correction in LED volumes. The method specifies generating pre-correction matrices through primary color calibration, creating a post-correction matrix via color rendition analysis, and utilizing these matrices at appropriate pipeline stages to optimize both lighting and display color characteristics.

Example 2 details solving for the first pre-correction matrix (M). The process involves displaying primary color patches on LED panels, capturing these with the principal photography camera, and computing a transformation matrix that makes displayed primaries appear correct to the camera.

Example 3 explains solving for the post-correction matrix (Q). This requires capturing images of a color chart illuminated by each LED spectral channel, analyzing these against target appearance, and computing a matrix that minimizes color difference while maintaining neutral color balance.

Example 4 covers generating the second pre-correction matrix (N) as the product of M and the inverse of Q (N = MQ⁻¹). This matrix compensates for the effects of Q on in-camera background elements while preserving lighting optimization benefits.

Example 5 describes generating the in-camera-frustum LED panel color calibration image. This involves displaying primary color patches on panels within the camera's view and capturing these with the principal photography camera under controlled conditions.

Example 6 elaborates on solving for the post-correction matrix Q through least-squares optimization. The method minimizes color difference across all color chart squares while maintaining constraints on neutral color reproduction.

Example 7 explains generating calibration imagery for out-of-camera-frustum analysis. This involves illuminating a color chart with each LED spectral channel separately and capturing these under controlled geometric conditions.

Example 8 details adjusting pixel values through RGB offset compensation. The method measures and compensates for light reflected by LED panels (black level) to maintain accurate background color reproduction.

System embodiments incorporate the described methods into virtual production workflows, with components for matrix computation, application, and real-time adjustment. Non-transitory computer-readable medium embodiments store instructions for implementing the color correction workflow on appropriate hardware platforms. The specification defines terms used throughout, including precise meanings for "in-camera-frustum," "out-of-camera-frustum," and specific matrix operations.