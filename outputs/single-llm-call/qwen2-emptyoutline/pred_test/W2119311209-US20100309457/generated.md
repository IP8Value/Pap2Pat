# DESCRIPTION

## BACKGROUND OF THE INVENTION

The invention pertains to the field of optical imaging, particularly to the development of a wavefront image sensor chip (WIS) that is capable of simultaneously measuring both the intensity and the phase front variations of an incident light field. Conventional image sensor chips, such as those used in digital cameras and microscopes, are primarily designed to capture the intensity variations of light waves. However, the phase front of a light wave carries additional information that is crucial for various applications, especially in biomedical imaging. Traditional phase microscopy techniques, such as differential interference contrast (DIC) microscopy, phase contrast microscopy, and Hoffman modulation contrast microscopy, have limitations in terms of cost, complexity, and the mixing of phase and intensity information. These limitations introduce ambiguities in the rendered images and prevent straightforward quantitative phase analysis. Furthermore, these techniques often require specialized optical components and complex setups, which can be cumbersome and expensive.

The need for a simpler, more cost-effective, and more versatile solution for phase imaging has led to the development of the WIS. This sensor chip integrates a grid of apertures directly on a standard image sensor chip, enabling it to measure both the intensity and the phase front variations of the incident light wave. The WIS operates in a high Fresnel number regime, allowing for tight confinement of the projection spots and high sensitivity to phase gradients. This innovation has the potential to revolutionize phase microscopy by providing a compact, integrated solution that can be easily incorporated into existing microscope systems.

## BRIEF SUMMARY OF THE INVENTION

The present invention discloses a wavefront image sensor chip (WIS) that is capable of simultaneously measuring both the intensity and the phase front variations of an incident light field. The WIS comprises a 2D array of circular apertures defined on top of a metal-coated image sensor chip, such as a charge-coupled device (CCD) or complementary metal-oxide-semiconductor (CMOS) chip. A transparent spacer separates the apertures from the sensor pixels. When a plane light wave is incident upon the aperture array, the transmission through each aperture forms a projection spot on the sensor pixels underneath. The center of each projection spot shifts according to the phase gradient of the light wave over its corresponding aperture. By evaluating the shift and the total intensity of each projection spot, the WIS can retrieve both the intensity and the phase information of the unknown light wave.

The WIS operates in a high Fresnel number regime, which allows for tight confinement of the projection spots and high sensitivity to phase gradients. The sensor chip is designed to be fabricated at the foundry level, making it cost-effective and suitable for mass production. The WIS can be integrated into standard microscope systems, transforming them into wavefront microscopes (WMs) that provide both bright-field and quantitative normalized phase gradient images. These images are improvements over standard DIC images in that they are quantitative, immune to birefringence-generated artifacts, and clearly separate the intensity and phase information of a light wave.

## DETAILED DESCRIPTION OF THE INVENTION

### Principle

The wavefront image sensor chip (WIS) consists of a 2D array of circular apertures defined on top of a metal-coated image sensor chip. A transparent spacer separates the apertures from the sensor pixels. The coordinate systems used in this description are illustrated in the figures. When a plane light wave is incident upon the aperture array, the transmission through each aperture forms a projection spot on the sensor pixels underneath. The center of each projection spot shifts according to the phase gradient of the light wave over its corresponding aperture. Mathematically, this shift in the \( s \) direction can be expressed as:

\[
\Delta s = \frac{H}{\lambda} \left( \frac{\partial \phi(x, y)}{\partial x} \right)
\]

where \( H \) is the distance from the aperture to the image sensor chip, \( \lambda \) is the wavelength of the light wave, and \( \frac{\partial \phi(x, y)}{\partial x} \) is the phase gradient of the light wave in the \( x \) direction over the aperture. Similar expressions can be derived for the \( y \) direction. The normalized phase gradient \( \theta_x \) (and \( \theta_y \)) measures the directionality of the incoming light wave and is a wavelength-independent measure of the angle at which the light impinges upon the aperture.

Each projection spot also provides a measurement of the local intensity of the light wave over its corresponding aperture by summing the total image sensor signal associated with the projection spot. The WIS retrieves the intensity and phase information of the unknown light wave by evaluating two independent aspects of each projection spot. A grid of \( N \times N \) pixels is assigned underneath each aperture to measure the transmission and shift of the projection spot. If an image sensor chip with \( M \times N \) pixels is used, the WIS can create a light wave image of \( M \times N \) pixels.

### Self-focusing Effect of the WIS Apertures in the High Fresnel Number Regime

The WIS operates in a high Fresnel number regime, which allows for tight confinement of the projection spots. Finite-difference time-domain (FDTD) simulations and experimental measurements were conducted to determine the distribution of light transmitted through a WIS aperture. The aperture diameter was set at 6 µm, and the refractive index of the spacer material was set at 1.6. The simulations and experiments showed that the light projection shrinks to a tightly confined spot before expanding linearly. The spot's width (full width at half maximum - FWHM) reached a minimum of 3.8 µm at an axial displacement of \( H = 18 \) µm, which is 37% smaller than the aperture diameter. This spot size confinement is robust over a range of axial displacements, ensuring high sensitivity and accuracy in phase gradient measurements.

### Fabrication

The high-density WIS prototype was fabricated using a commercially available CMOS image sensor chip (MT9P031I12STM from Aptina Imaging) as the substrate. The sensor has 1944 × 2592 pixels of size 2.2 µm. The glass window was removed to gain access to the surface of the sensor. The surface was planarized with a 10 µm thick layer of SU8 resin, which served to nullify the optical properties of the lens on top of each sensor pixel and act as a spacer. A 150 nm thick layer of aluminum was coated on the SU8 layer to mask the sensor from light. Photolithography was used to create a 2D aperture array (280 × 350 apertures, 6 µm aperture diameter, and 11 µm aperture-to-aperture spacing) on the aluminum film. A dedicated grid of 5 × 5 sensor pixels was assigned underneath each aperture to detect the associated projection spot.

The total signal accumulation time was 1.0 second, and the typical light intensity on the sensor was 9.2 µW/cm². The center of each projection spot was determined with a precision of 1.8 nm, translating to a local normalized phase gradient sensitivity of 0.1 mrad. The WIS prototype can measure the local normalized phase gradient linearly over a range of ±15 mrad, which is adequate for microscopy applications.

### Cyclic Algorithm for Estimating the Center of Each Projection Spot

The centroid method is a straightforward algorithm for determining the center of each projection spot, but it is unstable due to the significant weights assigned to noise-corrupted data from dark pixels. The Fourier-demodulation algorithm, developed by Ribak's group, is more robust for dealing with light spots arranged in an approximately regular grid. A modified version, termed the cyclic algorithm, was developed for the WIS. This algorithm uses cyclic and uni-norm complex weights to determine the lateral shift of the projection spot with excellent sub-pixel accuracy. The cyclic algorithm is particularly suited for the 2D image sensor pixels and can be calibrated to correct for any biases introduced by the discrete data.

### Calibration Experiment for the Normalized Phase Gradient Response of the WIS

Calibration experiments were conducted to establish the distance from the WIS apertures to the photosensitive areas of the sensor pixels. The slopes of the calibration curves were used to estimate this distance, which was found to be 27.2 µm and 28.0 µm in the \( x \) and \( y \) directions, respectively. The sensitivity of the normalized phase gradient measurement was better than 0.1 mrad under typical working conditions. The normalized intensity gradient can also induce a shift to each aperture projection spot, but this can be minimized by reducing the aperture size or increasing the distance \( H \).

### Wavefront Microscopy Setup

By employing the WIS chip in place of the conventional camera in a standard bright-field microscope, the microscope can be transformed into a wavefront microscope (WM) capable of simultaneously acquiring bright-field and quantitative normalized phase gradient images. The WIS prototype was attached to an Olympus BX 51 microscope via its camera port, and the microscope was outfitted with a standard halogen light source. The microscope was also equipped with DIC prisms and polarizers for comparison. The WIS prototype performed well with various objective lenses, achieving resolutions close to the specified microscopy resolution.

### Results

#### Polystyrene Microspheres

The WIS was used to image polystyrene microspheres. The intensity image of the WM was consistent with the bright-field image, and the normalized phase gradient images in the \( x \) and \( y \) directions provided orthogonal phase information. The normalized intensity gradient component was removed from the phase gradient images to ensure accurate phase measurements.

#### Unstained Starfish Embryo in the Late Gastrula Stage

The WIS was used to image an unstained starfish embryo in the late gastrula stage. The intensity image of the WM was consistent with the bright-field image, and the normalized phase gradient images in the \( x \) and \( y \) directions provided clear phase information. The WM images were free from birefringence artifacts, which are common in DIC images of birefringent samples.

#### Stained Starfish Embryo in the Early Gastrula Stage

The WIS was used to image a stained starfish embryo in the early gastrula stage. The intensity image of the WM was consistent with the bright-field image, and the normalized phase gradient images in the \( x \) and \( y \) directions provided clear phase information. The WM images were free from the ambiguities associated with DIC images of stained samples.

#### Strongly Birefringent Potato Starch Granules

The WIS was used to image potato starch granules, which are known to be strongly birefringent. The intensity image of the WM was consistent with the bright-field image, and the normalized phase gradient images in the \( x \) and \( y \) directions provided clear phase information. The WM images were free from birefringence artifacts, which are common in DIC images of birefringent samples.

### Discussion

The WIS offers several advantages over traditional phase microscopy techniques. It is a low-cost, compact, and integrated solution that can be easily incorporated into existing microscope systems. The WIS can provide quantitative, artifact-free phase images, which are valuable for various applications in biomedical imaging, such as quantifying the optical thickness of cells or organelles and determining chemical concentrations in microfluidic systems. The WIS can also be adapted for color imaging by starting with a substrate sensor chip that has stratified color sensing sites or by developing more sophisticated projection spot localization algorithms for conventional color sensors with Bayer color pixel arrangements.

### Conclusion

The wavefront image sensor chip (WIS) is a novel and innovative solution for phase imaging. It can transform a standard microscope into a wavefront microscope (WM) that provides both bright-field and quantitative normalized phase gradient images. The WIS is easy to use, cost-effective, and does not require complex bulk optical arrangements. The required fabrication processes are compatible with semiconductor foundry procedures, making it possible to produce WIS chips in large quantities. The WIS has the potential to significantly impact various fields, including biomedical imaging, adaptive optics, machine recognition, texture assessment, and object ranging.