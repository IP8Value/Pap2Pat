# DESCRIPTION

## BACKGROUND

The field of optics and photonics has seen significant advancements in recent years, particularly in the manipulation of electromagnetic (EM) radiation at subwavelength scales. Conventional optical elements such as lenses, polarizers, beamsplitters, and mirrors are typically engineered at a scale much larger than the wavelength of light, which limits their ability to achieve certain functionalities. However, the development of metamaterials and metasurfaces has opened new avenues for controlling the phase, amplitude, and polarization of EM radiation with unprecedented precision.

Metamaterials, which are artificial materials engineered to have properties not found in nature, have been used to achieve sub-diffraction lensing and other exotic optical effects. Metasurfaces, which are two-dimensional arrays of subwavelength structures, have gained substantial interest due to their ability to perform complex optical functions within extremely thin layers. Despite these advancements, initial metasurface designs using plasmonic metals suffered from high optical losses and narrow bandwidths, limiting their practical applications.

This invention addresses these limitations by utilizing an inverse electromagnetic design method to create high-efficiency, broadband, dielectric-based thin electromagnetic metadevices. These metadevices can perform various optical functionalities such as polarization splitting, bending, and focusing of EM radiation, making them suitable for a wide range of applications in telecommunications, imaging, and sensing.

## SUMMARY

The present invention relates to a method and system for designing and fabricating high-efficiency, broadband, dielectric-based thin electromagnetic metadevices using an inverse electromagnetic design approach. The metadevices are capable of performing various optical functionalities such as polarization splitting, bending, and focusing of electromagnetic radiation. The invention includes the following key aspects:

1. **Inverse Electromagnetic Design Method**: The method involves defining the desired optical functionality in terms of input and output field distributions and solving the electromagnetic wave equation as an optimization problem. This approach allows for the exploration of the entire design space, enabling the creation of metadevices with enhanced functionalities.

2. **Dielectric Materials**: The metadevices are fabricated using low-loss dielectric materials, such as high impact polystyrene (HIPS), which have a low attenuation in the millimeter-wave to microwave region. This choice of material ensures high efficiency and broad bandwidth operation.

3. **Additive Manufacturing**: The metadevices are fabricated using additive manufacturing techniques, such as 3D printing, which allows for the creation of complex geometrical patterns with high precision and scalability. This method enables the fabrication of metadevices with subwavelength feature sizes, suitable for operation across a wide range of wavelengths from visible to millimeter-wave frequencies.

4. **Broadband Operation**: The metadevices are designed to operate over a broad bandwidth, typically greater than 25% of the central wavelength. This broadband capability is achieved by optimizing the dielectric structures to avoid resonant modes, which typically result in narrowband operation.

5. **Versatile Applications**: The metadevices can be used in various applications, including polarization beam splitters, bending devices, and metalenses. These devices can be integrated into complex electromagnetic systems for telecommunications, imaging, and sensing applications.

## DETAILED DESCRIPTION

### Inverse Electromagnetic Design Method

The inverse electromagnetic design method is a powerful tool for creating metadevices with specific optical functionalities. The method starts by defining the desired input and output field distributions at the boundaries of the design space. The electromagnetic wave equation is then treated as an optimization problem, where the goal is to find the refractive index distribution that satisfies the desired boundary conditions. The optimization problem is solved using an iterative algorithm, such as the objective-first algorithm, which alternates between optimizing the electric field and the permittivity distribution.

For example, to design a polarization beam splitter, the input is a normally incident plane wave with parallel and perpendicular polarizations, and the output is two different diffraction orders. The algorithm generates a binary refractive index distribution of dielectric and air that achieves the desired polarization splitting. Similarly, for a bending device, the input is a normally incident plane wave, and the output is a plane wave with a different diffraction order. For a metalens, the input is a normally incident plane wave, and the output is a cylindrical wave converging to a chosen focal point.

### Dielectric Materials

The metadevices are fabricated using low-loss dielectric materials, such as high impact polystyrene (HIPS), which have a low attenuation in the millimeter-wave to microwave region. The dielectric constant of HIPS is approximately 2.3, and its loss tangent is less than 0.003 over the 26–38 GHz band. This choice of material ensures high efficiency and broad bandwidth operation. The dielectric constant is used as a constraint in the inverse design algorithm to create binary devices made of air (ε = 1) and HIPS (ε = 2.3).

### Additive Manufacturing

The metadevices are fabricated using additive manufacturing techniques, such as 3D printing, which allows for the creation of complex geometrical patterns with high precision and scalability. The 3D printing process is a bottom-up approach that enables the fabrication of devices with a large aspect ratio and subwavelength feature sizes. The devices are printed using a consumer-grade 3D printer based on fused deposition modeling, which has a resolution ranging from 100 nm to 1 mm. This method is scalable and can be used to fabricate electromagnetic devices for applications from the visible to the millimeter-wave and microwave regimes.

### Broadband Operation

The metadevices are designed to operate over a broad bandwidth, typically greater than 25% of the central wavelength. This broadband capability is achieved by optimizing the dielectric structures to avoid resonant modes, which typically result in narrowband operation. For example, the polarization beam splitter and bending devices are designed to operate over a bandwidth of 27 to 38 GHz, corresponding to a relative bandwidth of 33%. The metalenses are designed to focus EM radiation over a bandwidth of 28 to 40 GHz, corresponding to a relative bandwidth of 33%.

### Versatile Applications

The metadevices can be used in various applications, including polarization beam splitters, bending devices, and metalenses. The polarization beam splitters can deflect normally incident plane waves of parallel and perpendicular polarizations into different diffraction orders. The bending devices can convert a normally incident plane wave into a plane wave with a different diffraction order. The metalenses can focus a normally incident plane wave into a cylindrical wave converging to a chosen focal point. These devices can be integrated into complex electromagnetic systems for telecommunications, imaging, and sensing applications.

### Experimental Verification

To verify the performance of the metadevices, a series of experiments were conducted using a vector network analyzer (VNA) and high-gain horn antennas. The input signal was transmitted through a high-gain horn antenna placed far away from the sample to produce a plane wave perpendicularly incident on the input surface. The device was surrounded by radar-absorbing material to prevent reflections from the surroundings. For the meta-gratings, the transmitted power was measured in the far-field using a low-gain horn antenna as a function of the angle and frequency. For the metalenses, the output power was mapped along the axial plane using a probe antenna attached to an X-Y stage.

The experimental results showed that the metadevices performed as expected. The polarization beam splitter deflected parallel and perpendicular polarizations into different diffraction orders with high efficiency and over a broad bandwidth. The bending devices converted a normally incident plane wave into a plane wave with a different diffraction order, and the metalenses focused EM radiation into a cylindrical wave converging to a chosen focal point. The experimental results closely matched the simulated results, demonstrating the effectiveness of the inverse electromagnetic design method and the fabrication process.

### Conclusion

The present invention provides a platform for designing and fabricating high-efficiency, broadband, dielectric-based thin electromagnetic metadevices using an inverse electromagnetic design method and additive manufacturing. The metadevices can perform various optical functionalities such as polarization splitting, bending, and focusing of EM radiation, making them suitable for a wide range of applications in telecommunications, imaging, and sensing. The invention addresses the limitations of conventional metasurfaces and opens new possibilities for the development of compact, low-cost, and high-performance electromagnetic devices.