Here is the patent application following the provided outline:

---

# DESCRIPTION  

## FIELD  

The present invention relates generally to the field of magnetic resonance imaging (MRI), and more specifically to methods and systems for improving cardiac gating in magnetic resonance coronary angiography (MRCA). The invention provides a novel technique called the "Septal Scout" for accurately determining the timing of the diastasis period in cardiac cycles, thereby enabling sharper and more diagnostically useful MRCA images compared to conventional cine cardiovascular magnetic resonance (CMR) gating methods.  

## BACKGROUND  

Magnetic resonance coronary angiography (MRCA) is a non-invasive diagnostic tool for assessing coronary artery disease (CAD). Unlike conventional x-ray angiography, MRCA offers three-dimensional visualization of coronary vessel lumens without exposing patients to ionizing radiation or the risks associated with catheterization. However, MRCA faces significant challenges due to cardiac motion artifacts, which degrade image quality. High spatial resolution is required for accurate diagnosis of CAD, but long acquisition times spanning multiple heartbeats introduce motion-related blurring.  

Prospective cardiac gating is the most effective technique for reducing cardiac motion artifacts in MRCA. Conventional cardiac gating relies on synchronizing the imaging window with the diastasis period—a phase of relative cardiac quiescence during diastole. Prior studies have measured coronary artery velocities during diastasis to be approximately 10 mm/s, with estimated imaging window durations ranging from 65 to 330 ms depending on the coronary artery segment.  

Current cardiac gating methods primarily utilize two components: (1) electrocardiogram (ECG) monitoring to detect ventricular systole onset, and (2) a pre-scan cine CMR video to determine the timing of diastasis relative to the ECG. However, conventional cine CMR has insufficient temporal resolution to accurately resolve transitional cardiac motion preceding and following diastasis. For example, a typical cine acquisition with a temporal resolution of 64 ms may fail to capture rapid coronary artery motion (e.g., 12 cm/s) during transitional phases, leading to gating errors. Additionally, cine acquisitions are susceptible to temporal data-mixing artifacts if the subject's heart rate varies during the scan.  

There exists a need for an improved cardiac gating method that provides higher temporal resolution, adapts to heart rate variability, and accurately identifies the diastasis period for optimal MRCA image quality.  

## SUMMARY  

The present invention addresses the limitations of conventional cardiac gating methods by introducing the "Septal Scout," a high-temporal-resolution motion monitoring technique for the ventricular septum. The Septal Scout operates by acquiring one-dimensional (1D) projections of the basal interventricular septum (IVS) at very short repetition times (TR), typically 10 ms or less, enabling precise tracking of septal motion throughout the cardiac cycle.  

Key aspects of the invention include:  
1. **High-Temporal-Resolution Acquisition**: The Septal Scout forgoes one dimension of spatial encoding to achieve rapid sampling of septal motion, providing temporal resolution superior to conventional cine CMR.  
2. **Motion Tracking via Optical Flow**: Septal displacement and velocity are derived using gradient optical flow analysis, which identifies inflection points in the velocity profile corresponding to the start and end of diastasis.  
3. **Adaptation to Heart Rate Variability**: The Septal Scout acquires data over multiple heartbeats, allowing determination of a robust diastasis window that accounts for heart rate fluctuations during breath-holds.  
4. **Improved MRCA Image Quality**: By providing more accurate gating windows, the Septal Scout reduces motion blurring in MRCA images, as demonstrated by quantitative and qualitative assessments of vessel sharpness.  

The invention further encompasses variations of the Septal Scout, including implementations with two-dimensional (2D) radiofrequency (RF) excitation to isolate the ventricular septum more precisely. Additionally, the method may be extended to function as a realtime "Septal Navigator," integrating with ECG and respiratory navigators for comprehensive motion-compensated imaging.  

## DETAILED DESCRIPTION  

The Septal Scout technique is implemented as follows:  

### **Septal Scout Acquisition**  
1. **Prescription**: From a 4-chamber long-axis cine CMR view, a scout plane (5–10 mm thick) is prescribed along the ventricular septum.  
2. **Imaging Parameters**: A modified steady-state free precession (SSFP) sequence is used with phase encoding disabled, acquiring 1D projections at a TR of 10 ms or shorter. Typical parameters include a flip angle of 55°, field of view (FOV) of 31 cm, and spatial resolution along the scout of 0.8 mm.  
3. **Data Collection**: Successive projections are appended along a time axis, analogous to M-mode ultrasound, forming a spatiotemporal intensity plot of septal motion.  

### **Motion Analysis**  
1. **Region of Interest (ROI) Selection**: A rigid ROI (±2.5 mm) is placed over the basal septum to average signals along the depth dimension, improving signal-to-noise ratio (SNR).  
2. **Optical Flow Processing**: The gradient optical flow method is applied to the intensity plot to derive pseudo-displacement and pseudo-velocity functions. The inflection points between the E-wave (early diastolic filling) and A-wave (atrial contraction) define the diastasis window.  

### **Integration with MRCA**  
1. **Gating Window Determination**: The Septal Scout is performed at the beginning and end of a breath-hold to account for heart rate variability. The intersection of diastasis periods across heartbeats yields a robust gating window (Wsep).  
2. **MRCA Acquisition**: A 3D SSFP sequence is used with parameters optimized for coronary imaging (e.g., TR/TE = 3.9/1.9 ms, resolution = 1.5 × 1.5 × 2.0 mm). The Septal Scout-derived window ensures data acquisition during true diastasis.  

### **Alternative Implementations**  
1. **2D RF Excitation**: A spiral RF pulse may be used to excite a cylindrical volume along the septum, reducing interference from static tissues while maintaining TR ≤ 10 ms.  
2. **Septal Navigator**: The Septal Scout may be adapted for realtime gating by interleaving with imaging, though care must be taken to avoid signal saturation in coronary arteries.  

### **Advantages Over Prior Art**  
- **Superior Temporal Resolution**: The Septal Scout’s 10-ms TR resolves rapid septal motion undetectable by cine CMR.  
- **Robustness to Heart Rate Variability**: Per-heartbeat diastasis tracking accommodates physiological HRV.  
- **Improved Diagnostic Quality**: Studies demonstrate significantly sharper vessel boundaries in Septal Scout-gated MRCA compared to cine-CMR (p < 0.05).  

### **Applications Beyond MRCA**  
The Septal Scout principle may be extended to:  
- **Systolic Triggering**: Replacing ECG R-peak detection for more reliable systolic gating.  
- **Pathological Conditions**: Future studies will assess performance in septal wall defects (e.g., Septal Bounce in constrictive cardiomyopathy).  

In summary, the Septal Scout represents a significant advancement in cardiac MRI gating, enabling sharper, more reliable coronary angiography without the limitations of conventional techniques.  

--- 

This application provides a comprehensive, standalone disclosure of the invention in formal patent language, adhering to the specified outline while ensuring technical completeness and legal robustness. Let me know if you'd like any modifications or additional details.