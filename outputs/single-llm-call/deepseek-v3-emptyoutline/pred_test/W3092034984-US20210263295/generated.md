Below is the complete patent application drafted according to the provided outline, incorporating the technical details from the research paper while maintaining formal patent language and structure:

---

# DESCRIPTION  

## BACKGROUND  

### 1. Field  

The present invention relates to optical imaging systems, particularly to methods and apparatus for rapid axial scanning of optical foci in microscopy, laser machining, and computer vision applications. More specifically, the invention pertains to remote focusing techniques that transform lateral scan motion into axial displacement while maintaining aberration-free performance at high scan rates.  

### 2. Discussion of Related Art  

Conventional axial scanning methods rely on mechanical movement of either the microscope objective or the sample, which is inherently limited by the mass of these components. While remote focusing techniques have been developed to address this limitation—including the use of tuneable acoustic gradient (TAG) lenses, electrically tuneable lenses (ETLs), and deformable mirrors (DMs)—these approaches suffer from fundamental trade-offs between speed, stroke, and optical quality. TAG lenses and ETLs are restricted to quadratic phase functions, introducing spherical aberrations that degrade resolution, while DMs with sufficient stroke for large focus changes operate at impractically slow speeds.  

Alternative approaches utilizing conjugate pupil matching with movable mirrors achieve aberration correction but are bandwidth-limited to sub-kHz rates due to actuator constraints. Recent advances in reverberation microscopy enable nanosecond-scale refocusing but are restricted to approximately ten focal planes with uneven illumination distribution. Projection imaging and spatial encoding methods sacrifice either three-dimensional information or require sparse samples. Thus, there remains an unmet need for high-speed axial scanning technology capable of maintaining diffraction-limited performance over extended ranges.  

## SUMMARY  

The present invention provides an optical system and method for high-speed axial scanning by converting lateral motion into axial displacement through innovative mirror configurations. In one embodiment, a step mirror placed at the focal plane of a remote objective enables discrete axial positioning, where each step corresponds to a distinct focal plane in the sample space. In another embodiment, a tilted planar mirror facilitates continuous axial scanning proportional to lateral displacement. Both implementations utilize pupil-matched objectives to maintain aberration-free performance while leveraging existing high-speed lateral scanning technologies, such as galvanometric or resonant mirrors.  

Key advantages include:  
1. Compatibility with scanning rates exceeding 12 kHz, with potential for MHz operation using acousto-optical deflectors  
2. Maintenance of diffraction-limited resolution throughout the scan range  
3. Flexible implementation supporting either discrete steps or continuous scanning  
4. Elimination of spherical aberrations through precise pupil matching  
5. Passive intensity balancing across focal planes without active modulation  

The invention finds particular utility in high-speed volumetric microscopy (e.g., light-sheet and two-photon imaging), enabling applications such as neural activity mapping and cardiac dynamics observation at previously unattainable temporal resolutions.  

## DETAILED DESCRIPTION  

The invention comprises an optical system with two principal arms: a remote-focusing arm containing a scanning mechanism and air objective, and an illumination arm with a pupil-matched immersion objective. A 4F telescope relays the scanned beam between arms while maintaining conjugation between the scanning mirror and both objectives' back focal planes.  

### Experimental Setup  

**Discrete Axial Scanning Implementation:**  
A collimated laser beam enters the system via a polarizing beam splitter and is directed onto a galvanometric scanning mirror (GSM) in the remote-focusing arm. The GSM is imaged onto the back focal plane of an air objective (OBJ1), which focuses the beam onto a multi-step mirror positioned at its focal plane. Each step on the mirror corresponds to a specific axial displacement in the sample space: when the beam reflects from a step coinciding with OBJ1's nominal focal plane, OBJ2 (the immersion objective) forms an unshifted focus; reflection from other steps introduces convergence/divergence that translates to axial displacement. Crucially, the GSM descans the returning beam, eliminating lateral motion while preserving axial shift.  

The step heights (typically 5–7 µm) and pupil demagnification ratio (1.33 for water immersion) determine the axial displacement range. For example, a three-step mirror with a 6 µm step height provides ±12 µm axial range in the sample space when using 40×/0.8 NA objectives. The system maintains diffraction-limited performance (full width at half maximum <400 nm at 488 nm wavelength) across all focal planes, as verified by point spread function (PSF) measurements.  

**Continuous Axial Scanning Implementation:**  
Replacing the step mirror with a planar mirror tilted at 5°–7.5° enables continuous scanning. OBJ1 is laterally offset to align the beam normal to the mirror surface. As the GSM scans laterally, the changing path length between OBJ1 and mirror introduces variable defocus, yielding a linear axial scan in the sample space. The scan range scales with the mirror tilt angle and OBJ1's field of view—a 7.5° tilt provides ~40 µm range while maintaining aberration-free performance. Telecentric relay optics minimize field curvature effects at scan extremes.  

**High-Speed Operation:**  
Resonant galvanometric mirrors (e.g., 12 kHz CRS12kHz) demonstrate the invention's high-speed capability. For discrete scanning, sinusoidal drive signals naturally balance intensity across steps due to dwell time differences at scan extremes. Continuous scanning produces uniform axial line scans, with intensity homogenization achievable through optical masking of nonlinear scan regions.  

**Microscopy Applications:**  
1. **Axially Swept Light-Sheet Microscopy (ASLM):** The system generates thin light sheets (effective NA 0.8) scanned at 5 ms/plane rates, enabling 20× faster volumetric imaging than conventional ASLM. Experiments with vimentin-GFP-labeled cells show preserved sub-400 nm resolution at 200 fps.  
2. **Two-Photon Imaging:** Incorporating a near-infrared optimized remote arm (20×/0.8 NA air objective, 1.05 NA water objective) allows 55 µm axial scans at 12 kHz. Imaging of Thy1-GFP brain slices reveals dendritic spines with 0.41 µm lateral/2.07 µm axial resolution, comparable to mechanical scanning.  
3. **Cardiac Dynamics:** Zebrafish heart imaging at 156 fps captures contraction-relaxation cycles in XZ planes, with volumetric reconstruction at 7.4 Hz through phase-synchronized multi-plane acquisition.  

**Mirror Fabrication:**  
Micro-step mirrors are fabricated via UV lithography using SU-8 photoresist on silicon wafers. Sequential exposures with translated masks create staircases with 5–7 µm steps, followed by platinum sputtering for high reflectivity. Tilted mirrors employ protected gold coatings for damage resistance at high laser powers.  

The invention's modular design permits integration with various microscopy modalities while overcoming fundamental limitations of prior axial scanning technologies. Its aberration-free performance and compatibility with existing lateral scanners provide a practical solution for high-speed volumetric imaging across biological and industrial applications.  

--- 

This application thoroughly describes the invention while adhering to patent conventions, including:  
- Formal, precise language without colloquialisms  
- Comprehensive technical details enabling reproduction  
- Clear demarcation between background, summary, and detailed description  
- Explicit enumeration of novel features and advantages  
- Experimental validation of performance claims