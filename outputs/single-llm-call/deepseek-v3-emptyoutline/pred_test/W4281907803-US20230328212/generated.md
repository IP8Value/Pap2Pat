Here is the complete patent application following the provided outline:

# DESCRIPTION  

## PRIORITY APPLICATIONS  

This application claims priority to U.S. Provisional Patent Application No. [APPLICATION NUMBER], filed on [DATE], entitled "SYSTEM AND METHOD FOR COLOR CALIBRATION OF RGB LED VIRTUAL PRODUCTION STAGES," the entire contents of which are incorporated herein by reference.  

## BACKGROUND  

Virtual production (VP) techniques employing LED stages or "volumes" have revolutionized film and television production by enabling actors to be photographed against high-resolution background imagery displayed on LED panels. This approach eliminates much of the post-production compositing work required with traditional greenscreen techniques. A critical aspect of filming within an LED volume involves lighting reproduction, where individual light sources surrounding an actor are driven to reproduce the illumination of a target scene. When displaying a high dynamic range image-based lighting (HDR IBL) environment, such systems can generally match a subject's appearance to how they would appear in a real-world setting.  

However, significant color rendition challenges persist when using RGB LED panels for lighting reproduction. The emission spectra of RGB LEDs differ markedly from most real-world illuminants, resulting in perceptible color shifts. Specifically, lighter skin tones tend to shift toward pink, darker skin tones toward red, orange materials toward red, cyan materials toward blue, and yellow materials darken. These errors stem from the "peaky" emission spectra of RGB LEDs, which create distinct gaps between the spectral channels. While prior solutions have incorporated additional spectral channels beyond RGB to fill these gaps in specialized lighting systems, conventional LED panels remain optimized for display gamut rather than spectral accuracy.  

Existing calibration workflows for RGB LED VP stages focus primarily on ensuring accurate color reproduction for in-camera background imagery. These processes typically involve displaying pure red, green, and blue patches on the panels and computing a 3×3 color correction matrix to align the camera-recorded primaries with the content's color primaries. While effective for display calibration, this approach fails to address color rendition errors when the panels function as light sources illuminating actors and physical set elements.  

## SUMMARY  

The present invention discloses a novel system and method for color calibration of RGB LED virtual production stages that simultaneously optimizes both lighting reproduction accuracy and in-camera background color fidelity. The technique employs multiple linear transformations represented as 3×3 color correction matrices to address different aspects of the color reproduction challenge.  

A first pre-correction matrix (M) is computed to ensure metameric matching between displayed content and camera-captured imagery when the LED panels function as displays. A second post-correction matrix (Q) is derived to optimize color rendition for materials illuminated by the LED volume, compensating for spectral deficiencies in the RGB LED emission. A third pre-correction matrix (N), computed as the product of M and the inverse of Q, is applied to in-camera frustum content to maintain accurate background colors while Q corrects the foreground lighting.  

The method requires only four calibration images captured with the principal photography camera: (1) an image of the LED panels displaying pure red, green, and blue patches for computing M; and (2-4) images of a color chart illuminated separately by each spectral channel of the LED volume for computing Q. No spectral measurements of the camera, materials, or LED panels are required.  

An additional black level subtraction process compensates for non-zero panel albedo, where LED panels reflect incident illumination from other panels. This is implemented by measuring the camera-observed color when out-of-frustum content is active while in-camera background panels are disabled, then applying a proportional offset to the rendered content.  

The technique provides significant improvements in color rendition accuracy compared to conventional primary-based calibration methods, particularly for skin tones and materials with specific spectral reflectance characteristics (e.g., orange, yellow, and cyan). Simultaneously, it maintains accurate color reproduction for in-camera background imagery. The method is computationally efficient, requiring only matrix operations that can be implemented in real-time within virtual production pipelines.  

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS  

### EXAMPLE EMBODIMENTS  

The following detailed description illustrates specific implementations of the invention but should not be construed as limiting its scope. Alternative embodiments incorporating the novel aspects described herein will be apparent to those skilled in the art.  

**System Architecture**  

The color calibration system operates within a virtual production environment comprising:  

1. An LED volume consisting of multiple RGB LED panels arranged to provide omnidirectional illumination and display capabilities. The panels may include wall, ceiling, and floor configurations with potentially different specifications, though radiometric alignment between panel types is assumed.  

2. A motion picture camera with known spectral sensitivity characteristics and linear response, typically a digital cinema camera configured for raw capture.  

3. A real-time rendering system (e.g., Unreal Engine) capable of applying distinct color correction matrices to in-frustum and out-of-frustum content separately, with camera tracking to adjust the frustum dynamically.  

4. A calibration processing unit that computes the required color correction matrices from captured calibration imagery.  

**Calibration Methodology**  

The calibration process comprises three primary phases:  

1. **Primary-Based Calibration for Matrix M**:  
   The system displays pure red, green, and blue patches on the in-camera frustum LED panels while keeping out-of-frustum panels disabled. The camera records these patches in a single image, from which average pixel values are extracted for each primary. These values form a 3×3 matrix [SL] representing the camera's response to each LED spectral channel.  

   Matrix M is computed as the inverse of [SL], ensuring that pure primary values displayed on the panels ([1,0,0], [0,1,0], [0,0,1]) are recorded as identical values by the camera. This matrix will be applied to out-of-frustum content during normal operation to maintain metameric illuminant matching.  

2. **Color Rendition Calibration for Matrix Q**:  
   The system illuminates a color chart separately with each spectral channel of the LED volume. For each channel (R, G, B), a 1m×1m square of LED wall is activated while the chart is positioned 1m away, oriented at 45° to the surface normal. The camera captures three images (one per channel) of the illuminated chart.  

   From these images, a set of 3×3 matrices [SRL]j is constructed for each color chart square j, representing how that material appears when illuminated by each spectral channel. These matrices incorporate the full spectral interaction between camera sensitivities, LED emissions, and material reflectances.  

   For a target lighting environment with known diffuse integral wavg (derived from a white square value in the real scene), the system predicts the appearance of each chart square under LED illumination as (1/β)[SRL]jMwavg, where β≈0.311 accounts for the solid angle difference between the calibration setup and full spherical illumination.  

   Matrix Q is solved to minimize the squared error between these predicted values and the target color chart appearance from the real scene. This can be done either by exactly matching three chart squares or through least-squares optimization using all available squares.  

3. **In-Camera Frustum Calibration for Matrix N**:  
   Matrix N is computed as MQ⁻¹ and applied to in-camera frustum content. This preemptively inverts the effect of Q on background imagery, allowing Q to correct foreground lighting while maintaining accurate background colors.  

**Black Level Compensation**  

To address light reflected from in-camera background panels:  

1. With out-of-frustum content active (applying M) and in-camera background disabled, the system records an image to measure the camera-observed color bcamera from panel reflections.  

2. From the primary calibration images, the system computes wcamera representing the camera response to a [1,1,1] panel display.  

3. An RGB offset of bcamera/wcamera is subtracted from rendered content before display to compensate for the reflected light.  

**Operational Workflow**  

During virtual production:  

1. Out-of-frustum content (lighting the actors) is pre-corrected with M.  
2. In-camera frustum content (background imagery) is pre-corrected with N.  
3. The black level offset is applied to all content.  
4. Captured footage is post-corrected with Q to optimize color rendition.  

**Theoretical Basis**  

The method leverages several physical principles:  

1. The superposition principle allows simulating any lighting environment as a linear combination of the three spectral channel responses.  

2. Diffuse materials integrate illumination over the full hemisphere, enabling characterization using the diffuse integral wavg.  

3. Metamerism permits matching perceived colors despite spectral differences, though with limitations for non-neutral materials.  

4. The near-Lambertian reflectance of color charts simplifies the relationship between illumination and observed color.  

**Performance Characteristics**  

Experimental results demonstrate:  

1. Average color errors below 4% relative to white square intensity for both lit and displayed color charts across diverse lighting environments.  

2. Significant improvement in skin tone reproduction, reducing pink/red shifts characteristic of RGB LED illumination.  

3. Better matching of orange, yellow, and cyan materials compared to primary-only calibration.  

4. Maintenance of accurate in-camera background colors despite aggressive post-correction for lighting.  

**Alternative Implementations**  

Variations include:  

1. Joint optimization of M and Q to minimize a combined error metric.  
2. Incorporation of additional spectral channels beyond RGB in LED panels.  
3. Machine learning approaches to predict optimal corrections from limited calibration data.  
4. Extension to mixed practical and virtual lighting scenarios.  

The invention provides a practical solution to color rendition challenges in RGB LED virtual production while requiring minimal additional equipment or complex measurements beyond standard calibration procedures. Its matrix-based approach ensures computational efficiency suitable for real-time operation in production environments.