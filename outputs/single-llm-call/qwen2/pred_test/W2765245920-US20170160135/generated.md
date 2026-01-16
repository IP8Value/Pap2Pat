# DESCRIPTION

## FIELD

The present invention relates to the field of spectral imaging, particularly to a method and apparatus for snapshot spectral imaging (SSI) using a dispersive diffuser and compressed sensing (CS) techniques. The invention aims to provide a compact and cost-effective solution for acquiring high-quality spectral images of dynamic objects in various applications, including but not limited to remote sensing, astronomy, biology, environmental studies, agriculture, food and drug inspection, automotive and vehicle sensors, medical diagnostics, photographic and video cameras, smartphones, wearable devices, and augmented reality.

## BACKGROUND

Spectral imaging (SI) involves the acquisition of a three-dimensional (3D) spectral cube of spatial and spectral data of a source object at a limited number of wavelengths within a given wavelength range. SI has found extensive applications in numerous fields, including biology, medicine, food inspection, archaeology, art conservation, astronomy, and remote sensing. Traditional SI methods, such as those using mosaic spectral filter arrays on the image sensor, suffer from substantial light gathering losses. Staring or pushbroom SI systems, which rely on removable sets of narrow bandpass filters or time-sequential dynamic spectral filters, are slow and unsuitable for dynamic, fast-changing objects.

Modern trends in digital imaging often combine optics with digital processing and compressed sensing (CS) to achieve various purposes and applications. CS-based algorithms have been successfully applied in fields such as astronomy, biology, medicine, radar, and seismology. Snapshot spectral imaging (SSI) refers to the instantaneous acquisition of the spectral cube, making it suitable for fast-changing objects. Several known SSI devices, such as the Coded Aperture Snapshot Spectral Imager (CASSI), use intermediate image plane optics and coded apertures, which increase the total track length, weight, and production costs of such devices.

To address these limitations, the present invention proposes a novel approach that converts a regular digital camera into an SSI camera for arbitrary objects. This is achieved by incorporating a diffusing and dispersing "phase-only" static optical element at the entrance pupil and using tailored CS methods for digital processing of the diffused and dispersed (DD) image recorded on the image sensor. The diffuser is designed to mix the spectral cube data spectrally and spatially, enabling convergence in its reconstruction by CS-based algorithms.

## SUMMARY

The present invention provides a method and apparatus for snapshot spectral imaging (SSI) using a dispersive diffuser and compressed sensing (CS) techniques. The apparatus includes a digital camera comprising an imaging lens, a monochromatic image sensor, and a bandpass spectral filter. A diffuser is positioned at the entrance pupil of the imaging lens, functioning as a random dispersing element that provides a diffused and dispersed (DD) image at the monochromatic image sensor. A digital processor processes the DD image to reconstruct a plurality of monochromatic images (i.e., the spectral cube) of the source object through iterative CS-based algorithms.

The diffuser is designed with a phase function that combines dispersive and diffusing properties, enabling spectral and spatial multiplexing. The phase function is a randomly permuted, nonlinear saw-tooth phase, which ensures a highly randomized system response required for effective CS reconstruction. The method involves capturing a DD image of the source object and reconstructing the spectral cube using CS-based algorithms, such as split Bregman iterations (SBI).

The invention also includes a method for calibrating the SSI camera by measuring the point-spread function (PSF) of the system and using the measured PSF to construct a sensing matrix. The reconstructed spectral cube is evaluated for quality by comparing it to a reference spectral cube obtained using traditional methods.

## DETAILED DESCRIPTION

### Diffuser Design Formulation

The diffuser is a critical component of the SSI system, designed to mix the spectral cube data spectrally and spatially. The diffuser is positioned at the entrance pupil of the imaging lens and functions as a random dispersing element. The phase function of the diffuser is a randomly permuted, nonlinear saw-tooth phase, which ensures a highly randomized system response required for effective CS reconstruction.

The phase function of the diffuser can be mathematically described as follows:
\[ \phi(x) = \sum_{k=1}^{N} a_k \sin\left(\frac{2\pi k x}{\lambda}\right) \]
where \( \phi(x) \) is the phase function, \( a_k \) are the coefficients, \( x \) is the spatial coordinate, and \( \lambda \) is the wavelength. The coefficients \( a_k \) are randomly permuted to ensure randomness in the phase function.

The diffuser provides wavelength-dependent light diffusion and serves as an inherent disperser, enabling spectral multiplexing along with spatial multiplexing. The diffuser modifies the system pupil function of the entire SSI optical system, leading to a coherent point-spread function (PSF) that can be calculated as an inverse Fourier transform of the modified pupil function.

### Finally

The invention also includes a method for calibrating the SSI camera to ensure accurate reconstruction of the spectral cube. Calibration involves measuring the point-spread function (PSF) of the system using a known reference object. The PSF measurements are used to construct a sensing matrix, which is essential for the CS-based reconstruction algorithms.

The calibration process involves the following steps:
1. **Reference Object Preparation**: Prepare a reference object with known spectral characteristics, such as a single thin white vertical column displayed on a screen.
2. **PSF Measurement**: Image the reference object at several spatial positions in each spectral band using a set of narrow-bandpass filters.
3. **Sensing Matrix Construction**: Use the PSF measurements to construct the sensing matrix, which describes the relationship between the DD image and the spectral cube.
4. **Reconstruction Algorithm Optimization**: Optimize the parameters of the CS-based reconstruction algorithm, such as the Lagrange weight coefficients, to ensure accurate and efficient reconstruction.

### Exemplary RIP Diffuser Implementation

An exemplary implementation of the RIP (Restricted Isometry Property) diffuser is described below. The diffuser is designed to meet the RIP condition, which is crucial for the successful reconstruction of the spectral cube using CS-based algorithms. The RIP condition ensures that any sub-matrix of the sensing matrix formed by less than K columns satisfies the inequality:
\[ (1 - \delta_K) \| \mathbf{d} \|_2^2 \leq \| \mathbf{Qd} \|_2^2 \leq (1 + \delta_K) \| \mathbf{d} \|_2^2 \]
for any K-sparse vector \( \mathbf{d} \), where \( \delta_K \) is a small positive constant.

The diffuser is fabricated with a 3.2 mm clear aperture and includes 400 stripes, each 8 µm wide. The phase function of the diffuser is a randomly permuted, nonlinear saw-tooth phase, ensuring a highly randomized system response. The diffuser is positioned at the entrance pupil of the imaging lens, and the PSF measurements are used to construct the sensing matrix.

### Apparatus Embodiments

The apparatus for snapshot spectral imaging (SSI) includes the following components:
1. **Digital Camera**: A regular digital camera comprising an imaging lens, a monochromatic image sensor, and a bandpass spectral filter.
2. **Diffuser**: A phase-only static optical element positioned at the entrance pupil of the imaging lens. The diffuser is designed to combine dispersive and diffusing properties, enabling spectral and spatial multiplexing.
3. **Digital Processor**: A digital processor that processes the diffused and dispersed (DD) image recorded on the image sensor to reconstruct the spectral cube using CS-based algorithms.

The digital processor implements the following steps:
1. **Image Capture**: Capture a DD image of the source object using the digital camera.
2. **Reconstruction**: Reconstruct the spectral cube from the DD image using CS-based algorithms, such as split Bregman iterations (SBI).
3. **Post-Processing**: Convert the reconstructed spectral cube to RGB color coordinates for visualization and further analysis.

### Computer Simulations

Computer simulations were performed to validate the effectiveness of the proposed SSI system. The simulations used a spectral cube obtained with a VTT spectral imager prototype, which is integrated with a piezo-driven Fabri-Perot Interferometer (FPI) for time-sequential scanning of the spectral domain. The spectral cube was acquired from human skin tissue with a suspected melanoma tumor.

The simulations involved the following steps:
1. **Preprocessing**: The original spectral cube was spatially cropped and padded to reduce noise and prepare it for the computational scheme.
2. **Sensing Matrix Generation**: The sensing matrix was generated by performing theoretical PSF calculations, assuming a nominal phase profile diffuser.
3. **Sensor Image Calculation**: The sensor image was calculated using the sensing matrix and a matrix of randomly distributed plus-and-minus-ones to reduce coherency.
4. **Reconstruction**: The spectral cube was reconstructed using the 2D framelet reconstruction method.

The results of the simulations demonstrated the feasibility of the proposed SSI system, showing that it can effectively reconstruct the spectral cube with high accuracy and efficiency.

In conclusion, the present invention provides a compact and cost-effective solution for snapshot spectral imaging (SSI) using a dispersive diffuser and compressed sensing (CS) techniques. The invention offers significant advantages in applications where weight, volume, and price are critical, making it suitable for a wide range of fields, including remote sensing, astronomy, biology, environmental studies, agriculture, food and drug inspection, automotive and vehicle sensors, medical diagnostics, photographic and video cameras, smartphones, wearable devices, and augmented reality.