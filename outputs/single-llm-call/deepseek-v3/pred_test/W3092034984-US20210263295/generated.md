Here is the patent application following your outline and research paper:

---

# DESCRIPTION  

## BACKGROUND  

### 1. Field  

The present invention relates to the field of optical microscopy, specifically to systems and methods for rapid axial scanning in high-resolution imaging applications. The invention enables aberration-free remote focusing by converting lateral-scan motion into axial displacement, thereby overcoming limitations of conventional mechanical or electro-optical focusing techniques.  

### 2. Discussion of Related Art  

Traditional optical microscopes rely on mechanical movement of objectives or samples to achieve axial scanning, which introduces significant limitations due to the inertia of moving components. Rapid axial scanning using movable mirrors or electrically tunable lenses (ETLs) has been explored, but these methods suffer from inherent trade-offs between speed, stroke, and optical performance. ETLs and TAG lenses, for example, approximate only quadratic phase functions, leading to spherical aberrations that degrade resolution. Deformable mirrors (DMs) can correct higher-order aberrations but are constrained by limited actuator stroke at high speeds.  

Remote focusing techniques have emerged as an alternative, wherein the focal plane is adjusted without moving the primary objective or sample. Current implementations involve moving a small mirror at the focus of a remote objective, but these systems are limited by actuator bandwidth, typically achieving scan rates below 1 kHz. Recent advances, such as reverberation microscopy, enable nanosecond-scale refocusing but are restricted to a small number of focal planes and suffer from uneven illumination and spherical aberrations.  

Existing remote focusing technologies fail to simultaneously achieve high resolution, large scan range, and multi-kHz scan rates. There remains an unmet need for an axial scanning system that combines the speed of lateral-scan technologies (e.g., galvanometric mirrors) with the aberration-free performance of remote focusing.  

## SUMMARY  

The present invention provides an optical imaging system that transforms lateral-scan motion into axial displacement, enabling high-speed, aberration-free remote focusing. The system comprises a light source, a scanning mechanism (e.g., galvanometric mirror), a remote-focusing objective, and a reflective element (e.g., step mirror or tilted planar mirror) positioned at the focal plane of the remote objective.  

Key innovations include:  
1. **Lateral-Scan Motion Conversion**: The system scans a laser spot laterally over a stationary mirror whose distance from the objective varies in the scan direction, introducing defocus for remote refocusing.  
2. **Aberration-Free Remote Focusing**: By compensating for lateral-scan components on the return path, the system achieves pure axial motion while maintaining diffraction-limited performance.  
3. **Discrete and Continuous Scanning**: A step mirror enables discrete axial steps of arbitrary size, while a tilted planar mirror permits continuous axial scanning over a defined range.  
4. **Applications**: The technology is adaptable to axially swept light-sheet microscopy (ASLM) and raster-scanning microscopes, enabling volumetric imaging at rates exceeding 10 kHz.  

Embodiments include step mirrors with non-tilted or tilted surfaces, as well as diffraction gratings for extended functionality. The system can be extended to two scan dimensions for increased flexibility.  

## DETAILED DESCRIPTION  

### System Configurations  

The imaging system comprises two primary configurations:  
1. **Step Mirror Configuration**: A mirror with discrete steps of varying heights is positioned at the focal plane of the remote objective. Lateral scanning over the steps generates axially displaced foci in the sample plane.  
2. **Tilted Mirror Configuration**: A planar mirror is tilted with respect to the optical axis, enabling continuous axial scanning as the beam is swept laterally.  

### Components and Operation  

1. **Light Source and Lenses**: A collimated laser beam is delivered to the system via a polarizing beam splitter and expanded to fill the aperture of the remote objective.  
2. **Remote Focusing Mechanism**: A galvanometric scanning mirror (GSM) directs the beam onto the remote objective, which focuses it onto the reflective element. The returning beam is descanned by the GSM, eliminating lateral motion.  
3. **Objectives and Detection**: Pupil-matched objectives ensure aberration-free focusing. A water-immersion objective in the illumination arm forms the final focus in the sample, while a detection objective and camera capture the image.  
4. **Mirror Arrangements**:  
   - **Step Mirror**: Each step corresponds to a discrete axial focus position. The height difference between steps determines the refocusing range.  
   - **Tilted Mirror**: The tilt angle and scan range dictate the continuous axial displacement. The angular aperture of the remote objective must exceed that of the illumination objective to maintain numerical aperture.  

### Performance and Results  

1. **Discrete Axial Scanning**: A three-step mirror (step height ~6 µm) produced axially displaced foci with diffraction-limited resolution (FWHM: 364 nm laterally, 371 nm axially).  
2. **Continuous Scanning**: A 7.5° tilted mirror achieved a 43 µm scan range with minimal aberrations. Resonant scanning at 12 kHz demonstrated uniform intensity distribution.  
3. **Applications**:  
   - **ASLM**: Achieved 20-fold faster volumetric imaging (3.5 Hz) with isotropic ~400 nm resolution.  
   - **Two-Photon Microscopy**: Imaged zebrafish cardiac dynamics at 156 Hz, resolving subcellular features.  

### Experimental Setup  

1. **Laser Light Generation**: A 488 nm CW laser was spatially filtered and expanded for ASLM; a Ti:sapphire laser (900 nm) was used for two-photon imaging.  
2. **Remote Focusing Arm**: Included a GSM, air objective, and reflective element (step or tilted mirror).  
3. **Illumination Arm**: Comprised a water-immersion objective and relay lenses for pupil matching.  
4. **Detection**: Fluorescence was captured via sCMOS cameras (ASLM) or PMTs (two-photon microscopy).  

### Fabrication of Micro Step Mirror  

SU-8 photoresist was spin-coated onto a silicon wafer and patterned via UV lithography to create steps of 5–7 µm height. A platinum coating ensured high reflectivity.  

### Applications  

The system enables:  
- High-speed volumetric imaging of dynamic processes (e.g., neuronal activity, blood flow).  
- Intravital microscopy with submicron resolution.  
- Multiphoton raster scanning for deep-tissue imaging.  

--- 

This application provides a comprehensive description of the invention, including its components, operation, and advantages over prior art. Let me know if you'd like any modifications or additional details.