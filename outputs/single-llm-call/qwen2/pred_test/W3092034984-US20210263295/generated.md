# DESCRIPTION

## BACKGROUND

### 1. Field

The present invention relates to the field of optical imaging systems, particularly to methods and apparatuses for rapid axial re-focusing in microscopy, computer vision, and laser machining. More specifically, the invention provides a novel approach to remote focusing that leverages lateral scanning technologies to achieve rapid and precise axial re-focusing without introducing spherical aberrations.

### 2. Discussion of Related Art

The ability to rapidly change the focal plane of an optical imaging system is crucial in various applications, including microscopy, computer vision, and laser machining. Traditional methods for re-focusing involve mechanically moving the microscope objective or the sample under study. However, these methods are often rate-limiting due to the high mass of the objects being moved, which is particularly problematic in fields requiring high-speed volumetric imaging, such as neuroscience.

Remote focusing techniques have emerged as a solution to this problem, where the position of the optical focus is adjusted without moving the primary objective or the sample. These techniques typically involve altering the wavefront in the optical train. For example, wavefront modifications can be introduced in Fourier space using tunable acoustic gradient index of refraction (TAG) lenses, electrically tunable lenses (ETLs), or deformable mirrors (DMs). While TAG lenses can achieve axial scan rates of hundreds of kHz, they and ETLs approximate only a quadratic phase function for defocusing, which does not account for higher-order aberrations necessary to maintain a diffraction-limited focus. Deformable mirrors, though capable of producing more complex wavefronts, face trade-offs between speed and actuator stroke, limiting their axial focusing range.

Another approach involves introducing wavefront alterations in a region of the optical train that is conjugate to the specimen. By carefully matching the pupils of an imaging objective and a remote-focusing objective, aberration-free remote focusing can be achieved by moving a small mirror at the focus of the remote objective. However, this method is limited by the actuator technology and typically does not exceed 1 kHz.

Recent advancements in axial re-focusing on the nanosecond timescale have been reported using multiplexing laser pulses or reverberation loops. However, these techniques are limited by technological and photophysical constraints, such as pulse repetition rate and fluorescence lifetime, and are best suited for focal planes spaced by one scattering mean-free path. Additionally, they introduce only a quadratic phase function for re-focusing, limiting them to low-resolution imaging.

Given these limitations, there remains a need for a high-resolution axial scanning technology capable of reaching multi-kHz rates while avoiding spherical aberrations. The present invention addresses this need by providing a method and apparatus that transform lateral-scan motion into axial re-focusing, leveraging high-speed scanning technologies while maintaining aberration-free performance.

## SUMMARY

The present invention provides a novel method and apparatus for rapid axial re-focusing in optical imaging systems. The invention utilizes lateral scanning technologies to transform lateral-scan motion into axial re-focusing, thereby achieving high-speed and high-resolution axial scanning without introducing spherical aberrations.

In one embodiment, the invention includes a remote-focusing arm and an illumination arm, each containing an objective lens. The remote-focusing arm includes a galvanometric scanning mirror (GSM) and a step mirror or a tilted planar mirror. The GSM is used to scan a laser spot laterally over the mirror, which introduces defocus in the wavefront. The defocused wavefront is then relayed to the illumination arm, where it forms an axially re-focused spot in the sample plane. The pupils of the two objectives are matched to ensure aberration-free remote focusing.

In another embodiment, the invention provides a method for rapid axial re-focusing using a resonant galvanometric mirror. The resonant mirror is driven at high frequencies (e.g., 12 kHz) to achieve continuous axial scanning. The method can be applied to various imaging modalities, including axially swept light-sheet microscopy (ASLM) and two-photon raster-scanning microscopy.

The invention offers several advantages over existing technologies, including:
- High-speed axial scanning capabilities (up to 12 kHz).
- Aberration-free remote focusing over a wide axial range.
- Compatibility with high-speed scanning technologies, such as resonant galvanometric mirrors.
- Flexibility in choosing the scanning waveform and performing discrete scanning steps.
- Improved spatial resolution and imaging quality compared to traditional methods.

## DETAILED DESCRIPTION

### Experimental Setup

The invention is implemented in an optical imaging system comprising a remote-focusing arm and an illumination arm. The remote-focusing arm includes a galvanometric scanning mirror (GSM) and an air objective lens (OBJ1), while the illumination arm includes a pupil-matched water immersion objective (OBJ2). The two arms are aligned such that the GSM is conjugate to the back focal plane of both objectives.

#### Discrete Axial Re-Focusing

In the discrete axial re-focusing embodiment, a step mirror is placed in the focal plane of OBJ1. The GSM scans the laser spot laterally over the steps of the mirror, introducing defocus in the wavefront. The defocused wavefront is relayed to OBJ2, forming an axially re-focused spot in the sample plane. The step mirror allows for discrete axial steps, with the step height determining the axial displacement. The pupils of OBJ1 and OBJ2 are matched to ensure aberration-free remote focusing.

#### Continuous Axial Re-Focusing

In the continuous axial re-focusing embodiment, the step mirror is replaced by a slightly tilted planar mirror. The incoming laser focus is tilted such that it is incident in a direction normal to the mirror surface. The GSM scans the laser spot laterally over the tilted mirror, introducing a continuous range of defocus in the wavefront. The defocused wavefront is relayed to OBJ2, forming a continuously axially re-focused spot in the sample plane. The achievable axial scan range scales with the field of view of OBJ1 and the tangent of the mirror tilt angle. The angular aperture of OBJ1 is larger than that of OBJ2 to maintain the numerical aperture of OBJ2.

#### Resonant Axial Scanning

To achieve high-speed axial scanning, a resonant galvanometric mirror is used in place of the GSM. The resonant mirror is driven at high frequencies (e.g., 12 kHz) to generate a sinusoidal scan motion. This motion is used to scan the laser spot over the step mirror or the tilted planar mirror, resulting in rapid axial re-focusing. The resonant scanning technology allows for continuous axial scanning with a linear focusing response to an increment in the mirror angle.

#### Application to Axially Swept Light-Sheet Microscopy (ASLM)

The invention is applied to ASLM to accelerate the frame rate of volumetric imaging. A cylindrical lens shapes the input beam into a light sheet, which is imaged into the image space of OBJ1. The tilted planar mirror is used to scan the light sheet over a specified axial range. The microscope achieves high spatial resolution and can image RPE cells labeled with vimentin-GFP at both 50 ms and 5 ms exposure times per frame. The improved frame rate allows for the tracking of genetically encoded multimeric nanoparticles (GEMs) as they rapidly diffuse through the cellular cytoplasm.

#### Application to Two-Photon Raster-Scanning Microscopy

The invention is also applied to two-photon raster-scanning microscopy to perform high-resolution volumetric imaging. A new remote-focusing arm is optimized for near-infrared transmission and matched to an existing two-photon raster-scanning microscope. The remote-focusing system uses a water dipping lens with a numerical aperture of 1.05 to maximize spatial resolution. The tilted planar mirror is used to achieve an axial scan range of 55 µm. The resonant galvanometric mirror is driven at 12 kHz to generate rapid axial re-focusing. The system is used to image the beating heart of a zebrafish embryo at a volume rate of 156 Hz, demonstrating the potential for intravital imaging.

### Conclusion

The present invention provides a novel method and apparatus for rapid axial re-focusing in optical imaging systems. By leveraging lateral scanning technologies, the invention achieves high-speed and high-resolution axial scanning without introducing spherical aberrations. The invention is applicable to various imaging modalities, including ASLM and two-photon raster-scanning microscopy, and offers significant improvements in imaging speed and quality.