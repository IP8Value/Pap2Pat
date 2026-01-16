Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

The field of electromagnetic (EM) radiation manipulation has traditionally relied on bulk optical elements such as lenses, polarizers, beamsplitters, and mirrors, which operate at scales much larger than the wavelength of light. Over the past two decades, significant advancements have been made in understanding light-matter interactions at subwavelength scales, leading to the development of metamaterials and nanostructured materials capable of unconventional EM wave manipulation. These include photonic crystals and negative-index materials, which have enabled sub-diffraction lensing. More recently, metasurfaces—ultra-thin layers composed of subwavelength structures—have emerged as promising platforms for achieving optical functionalities like lensing, holography, and beam shaping.  

Despite their potential, conventional metasurfaces suffer from limitations such as high optical losses in plasmonic metals and narrow operational bandwidths in dielectric-based designs. Traditional metasurface design approaches rely on predefined geometric shapes (e.g., triangles, rectangles, or V-antennas) with limited degrees of freedom, making it challenging to optimize both efficiency and bandwidth while maintaining full polarization control. There exists a need for a more versatile and efficient design methodology that overcomes these limitations, enabling high-performance, broadband, and polarization-controlled metadevices.  

## SUMMARY  

The present invention discloses a novel approach for designing and fabricating high-efficiency, broadband, and ultra-thin electromagnetic metadevices using an inverse electromagnetic design method combined with additive manufacturing. The invention provides a systematic framework for creating metadevices that manipulate EM radiation with unprecedented control over phase, amplitude, and polarization.  

Key aspects of the invention include:  
1. **Inverse Electromagnetic Design**: The invention employs an optimization-based algorithm that treats Maxwell's equations as an inverse problem, enabling the discovery of complex dielectric structures that achieve desired EM functionalities without being constrained by predefined geometric shapes.  
2. **Broadband and High-Efficiency Operation**: Unlike conventional resonant metasurfaces, the disclosed metadevices operate over a wide wavelength range (Δλ/λ > 25%) with efficiencies exceeding 60%, achieved through non-resonant dielectric interactions.  
3. **Polarization Control**: The invention enables the design of polarization-sensitive devices, such as polarization beam splitters, as well as polarization-independent devices, such as bending gratings and lenses.  
4. **Additive Manufacturing**: The metadevices are fabricated using scalable 3D-printing techniques, allowing for rapid prototyping and cost-effective production of complex dielectric structures with subwavelength precision.  

Demonstrated applications include meta-gratings for polarization splitting and bending, as well as flat metalenses for focusing EM radiation. These devices exhibit superior performance compared to conventional designs, with reduced power loss and enhanced bandwidth.  

## DETAILED DESCRIPTION  

### **Inverse Electromagnetic Design Methodology**  

The invention utilizes an inverse-design algorithm to optimize the spatial distribution of dielectric material within a predefined design space. The electromagnetic wave equation is treated as an optimization problem:  

\[
\min_{\varepsilon,E} \nabla \times \nabla \times E - \omega^2 \varepsilon E
\]  

where \( \varepsilon \) is the permittivity distribution and \( E \) is the electric field. The optimization is decomposed into two subproblems:  
1. **Field Optimization**: The electric field \( E \) is computed while fixing the permittivity distribution \( \varepsilon \).  
2. **Permittivity Optimization**: The dielectric structure \( \varepsilon \) is updated while fixing the electric field \( E \).  

This iterative process converges toward a dielectric structure that satisfies predefined input-output field conditions, enabling the design of devices with tailored EM responses.  

### **Design of Meta-Gratings**  

The invention includes meta-gratings that perform polarization splitting and bending of EM radiation. Key design parameters include:  
- **Periodicity (L)**: Determined by the desired diffraction angle \( \theta \) via the grating equation \( L \sin \theta = m \lambda \).  
- **Dielectric Thickness (t)**: Ensures a \( 2\pi \) phase shift between dielectric (\( \varepsilon = 2.3 \)) and air (\( \varepsilon = 1 \)), requiring \( t \geq 2\lambda \).  

#### **Polarization Beam Splitters**  
Two types of polarization splitters are disclosed:  
1. **Dual-Deflection Splitter**: Splits parallel and perpendicular polarizations into opposite diffraction orders (e.g., \( m = +1 \) and \( m = -1 \)).  
2. **Single-Deflection Splitter**: Bends both polarizations to the same diffraction order.  

Experimental validation shows deflection angles of \( \pm 30^\circ \) and \( \pm 15^\circ \), with measured efficiencies of 76% (parallel) and 54% (perpendicular) at 33 GHz. The devices exhibit broadband operation (27–38 GHz) and high rejection ratios (>5 dB).  

#### **Comparison with Conventional Gratings**  
Inverse-designed gratings outperform traditional blazed gratings by reducing unwanted diffraction orders by a factor of 2.8 (parallel) and 2.0 (perpendicular), while maintaining high rejection ratios (>10 dB).  

### **Design of Metalenses**  

The invention further discloses flat metalenses that focus EM radiation with subwavelength thickness. Key features include:  
- **Focal Length Customization**: Lenses are designed for focal lengths of \( 2\lambda \) and \( 15\lambda \).  
- **Broadband Focusing**: Operates from 28 GHz to 40 GHz with practical numerical apertures (NA) of 0.8 and 0.36.  

Experimental results confirm focusing at 1.5 cm (\( 2\lambda \)) and 12 cm (\( 15\lambda \)), with full-width-at-half-maximum (FWHM) beam widths of 0.5 cm and 1.1 cm, respectively.  

### **Fabrication via Additive Manufacturing**  

The metadevices are fabricated using fused deposition modeling (FDM) 3D-printing with high-impact polystyrene (HIPS), a low-loss dielectric (\( \tan \delta < 0.003 \)) with a refractive index of \( n \approx 1.52 \). The 3D-printing process enables the realization of complex dielectric structures with high fidelity, scalability, and cost efficiency.  

### **Advantages Over Prior Art**  

1. **Enhanced Performance**: The inverse-designed metadevices achieve higher efficiency (>60%) and broader bandwidth (Δλ/λ > 25%) compared to resonant metasurfaces.  
2. **Polarization Versatility**: The methodology supports both polarization-sensitive and polarization-independent designs.  
3. **Scalability**: The approach is applicable across the EM spectrum, from visible to microwave frequencies, provided suitable dielectric materials are available.  

### **Experimental Validation**  

The performance of the disclosed metadevices is validated through:  
- **Far-Field Measurements**: Angular transmission profiles confirm predicted deflection and polarization splitting.  
- **Spatial Field Mapping**: Intensity distributions verify focusing behavior in metalenses.  
- **Broadband Characterization**: Devices operate consistently across a wide frequency range (26–44 GHz).  

### **Conclusion**  

The invention provides a transformative platform for designing and fabricating high-performance electromagnetic metadevices. By combining inverse electromagnetic design with additive manufacturing, the disclosed methodology enables rapid prototyping of compact, low-loss, and broadband optical components with applications in imaging, sensing, and communication systems.  

The foregoing description is illustrative and not limiting. Modifications and variations will be apparent to those skilled in the art without departing from the scope of the invention as defined by the appended claims.