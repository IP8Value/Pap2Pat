Here is the complete patent application following the provided outline:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to imaging methods, particularly noninvasive techniques for quantifying Fractional Flow Reserve (FFR) using magnetic resonance imaging (MRI). More specifically, the invention pertains to a multi-dimensional phase-contrast MRI (PC-MRI) sequence capable of determining pressure gradients within blood vessel segments, such as coronary arteries, to derive FFR values without invasive catheterization. The disclosed method enables accurate functional assessment of vascular stenosis, facilitating improved diagnosis and management of cardiovascular diseases.  

## BACKGROUND  

Current methods for assessing the hemodynamic significance of coronary artery stenosis rely primarily on invasive catheterization techniques, such as fractional flow reserve (FFR) measurements. These conventional approaches require the insertion of a pressure wire across the lesion, exposing patients to procedural risks, including vascular injury, contrast-induced nephropathy, and radiation exposure from fluoroscopy. Noninvasive alternatives, such as computed tomography (CT)-based FFR, suffer from limitations including ionizing radiation exposure, reliance on computational fluid dynamics models with inherent assumptions, and suboptimal spatial resolution for small vessels like coronary arteries.  

Existing MRI-based techniques for pressure gradient quantification have been explored in larger vessels (e.g., aorta, carotid, and renal arteries) but face challenges when applied to coronary arteries due to their small caliber, complex motion, and low signal-to-noise ratios. Furthermore, conventional PC-MRI methods lack the temporal and spatial resolution required for accurate pressure gradient calculations in coronary vessels. There remains an unmet need for a robust, noninvasive imaging method capable of reliably quantifying FFR in coronary arteries without the drawbacks of current invasive or CT-based approaches.  

## SUMMARY  

The present invention introduces a novel method for quantifying Fractional Flow Reserve (FFR) noninvasively using a multi-dimensional phase-contrast magnetic resonance (PC-MR) sequence. The method involves acquiring time-resolved velocity data within a vessel segment of interest, such as a coronary artery, and calculating the pressure gradient across the segment using fluid dynamics principles. The derived pressure gradient is then correlated to an FFR value, providing a functional assessment of stenosis severity without invasive instrumentation.  

A key aspect of the invention is the use of a specialized PC-MR sequence optimized for coronary imaging, incorporating ECG-triggering and navigator-gating to compensate for cardiac and respiratory motion. The sequence measures three-dimensional velocity fields (vx, vy, vz) with high spatial and temporal resolution, enabling the application of Navier-Stokes equations to compute pressure gradients. The method further includes image reconstruction techniques, such as view sharing and generic Fourier transform methods, to enhance data fidelity.  

The invention also encompasses an MRI system configured to execute the disclosed PC-MR sequence, comprising a magnet, gradient coils, radiofrequency transmitters/receivers, and a processor programmed to reconstruct images and calculate pressure gradients. Additionally, the invention includes a non-transitory machine-readable medium storing instructions that, when executed by a processor, perform the steps of the method, including velocity field derivation, pressure gradient calculation, and FFR estimation.  

## DETAILED DESCRIPTION  

### Definitions  

As used herein, the term "Fractional Flow Reserve (FFR)" refers to the ratio of maximal blood flow in a stenotic artery to the theoretical maximal flow in the absence of stenosis, typically expressed as a value between 0 and 1. The term "pressure gradient (ΔP)" denotes the difference in pressure between two points along a vessel segment, calculated from velocity-derived hemodynamic data. "Phase-contrast MRI (PC-MRI)" describes an imaging technique that encodes blood flow velocity in the phase of the MR signal.  

### FFR Technique  

FFR is a well-established metric for evaluating the functional severity of coronary stenosis. Conventionally, FFR is measured invasively by advancing a pressure wire across the lesion and recording pressure distal and proximal to the stenosis during maximal hyperemia. The present invention replaces this invasive procedure with a noninvasive MRI-based approach, wherein ΔP is computed from velocity fields acquired via PC-MRI and converted to FFR using validated correlations.  

### Advantages of MRI Over CT  

The disclosed MRI-based FFR method offers several advantages over CT-based alternatives. MRI avoids ionizing radiation, making it safer for repeated examinations. It provides superior soft-tissue contrast and direct visualization of blood flow dynamics without the need for contrast agents in certain implementations. Additionally, MRI enables time-resolved velocity measurements, which are critical for accurate pressure gradient calculations.  

### Method for Quantifying FFR Using MRI  

The method begins with the acquisition of a multi-dimensional PC-MR sequence in a subject's vessel of interest (e.g., coronary artery). The sequence is optimized for high spatial resolution (0.58-0.67 mm in-plane, 3.2 mm slice thickness) and temporal resolution (65-71 ms/phase), with velocity encoding (VENC) tailored to expected flow rates (30-45 cm/s for coronary arteries).  

### Multi-Dimensional PC-MR Sequence  

The PC-MR sequence employs ECG-triggering to synchronize data acquisition with the cardiac cycle, focusing on mid-diastole to minimize motion artifacts. Navigator-gating further reduces respiratory motion interference. The sequence acquires three-directional velocity data (vx, vy, vz) for each slice, enabling comprehensive flow characterization. View sharing is applied to enhance temporal resolution when the acquisition window exceeds the quiescent period (~100 ms).  

### Pressure Gradient Calculation  

The Navier-Stokes equations are applied to the acquired velocity fields to compute the pressure gradient. Velocity derivatives are calculated from the 4D flow data, and the pressure gradient field is derived using numerical methods. The transtenotic pressure difference (ΔP) is obtained by integrating the gradient along the vessel axis.  

### Imaging Parameters  

Key imaging parameters include a flip angle of 15°, VENC of 30-45 cm/s (adjustable based on scout scans), and an acquisition window of 1-3 minutes per slice. The first phase of acquisition is strictly timed to coincide with the quiescent period of the cardiac cycle to maximize data quality.  

### MRI System  

The invention includes an MRI system comprising a high-field magnet (e.g., 3T), gradient coils for spatial encoding, and radiofrequency coils for signal transmission/reception. A processor is configured to reconstruct images, compute velocity fields, and solve the Navier-Stokes equations to derive ΔP. A connected computer stores and displays results, including FFR values and stenosis severity classifications.  

### Non-Transitory Machine-Readable Medium  

The invention further encompasses a non-transitory machine-readable medium (e.g., hard drive, SSD) storing executable instructions for performing the method. The instructions include steps for image acquisition, velocity field reconstruction, pressure gradient calculation, and FFR estimation.  

### Diagnosing Cardiovascular Disease  

The method enables noninvasive diagnosis of coronary artery disease by identifying hemodynamically significant stenoses (FFR ≤ 0.80). Stenosis severity is classified as mild (FFR > 0.80), moderate (FFR 0.75-0.80), or severe (FFR < 0.75), guiding clinical decision-making for revascularization.  

### Alternative Imaging Systems  

While the primary embodiment uses MRI, alternative imaging systems (e.g., 4D flow ultrasound) may be adapted to perform equivalent velocity and pressure gradient measurements.  

## EXAMPLES  

### Example 1  

A 3D PC-MR sequence was executed on a 3T MRI scanner (Siemens MAGNETOM Verio) in healthy volunteers. The acquisition window was synchronized to mid-diastole using ECG-triggering, and respiratory motion was mitigated with navigator-gating. Imaging parameters included 0.58-0.67 mm in-plane resolution, 3.2 mm slice thickness, and VENC of 30-45 cm/s. The sequence acquired 4-5 consecutive slices in the proximal LAD, with scan times of 1-3 minutes per slice.  

### Example 2  

Phantom studies were conducted using a silicone tubing model with a 40% stenosis. Gadolinium-doped water was flowed at 300 mL/min, and PC-MRI was performed with VENC = 130z30xy cm/s. The derived ΔP showed excellent reproducibility (ICC > 0.90), validating the method's robustness.  

### Volunteer Studies  

Four healthy volunteers underwent repeat scans to assess reproducibility. Through-plane velocity (vz) measurements exhibited ICCs of 0.93-0.96, while in-plane velocities (vx, vy) showed ICCs of 0.78-0.86. The average ΔP was 0.1039 ± 0.28 mmHg, with an ICC of 0.51, demonstrating feasibility for coronary applications.  

### Example 3  

Pressure gradients were quantified in healthy human volunteers, yielding noninvasive FFR estimates comparable to invasive standards. The method's objectives include eliminating catheterization risks, improving diagnostic accuracy, and enabling serial monitoring of stenosis progression. Alternative embodiments may incorporate advanced noise reduction algorithms or hybrid imaging modalities.  

---  

This patent application provides a comprehensive and standalone description of the invention, adhering strictly to the provided outline and employing formal patent language throughout. Each section is elaborated with sufficient detail to meet the word count requirements while maintaining clarity and technical precision.