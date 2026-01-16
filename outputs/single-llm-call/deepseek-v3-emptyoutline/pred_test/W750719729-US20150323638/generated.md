Here is the complete patent application following the provided outline and incorporating the research paper's content:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of noninvasive medical diagnostics, specifically to methods and systems for assessing coronary artery stenosis using magnetic resonance imaging (MRI). More particularly, the invention pertains to a reproducible technique for measuring pressure gradients (ΔP) and flow velocities in coronary arteries through phase-contrast MRI (PC-MRI), enabling the noninvasive determination of Fractional Flow Reserve (FFR). The disclosed method eliminates the need for invasive catheterization by providing accurate, high-resolution hemodynamic assessments of coronary lesions.  

## BACKGROUND  

Coronary artery disease (CAD) remains a leading cause of morbidity and mortality worldwide. The functional assessment of coronary stenosis has traditionally relied on invasive techniques such as FFR, which measures the pressure drop across a lesion using a pressure wire during cardiac catheterization. While FFR is clinically validated, its invasiveness, cost, and procedural risks limit widespread adoption. Noninvasive alternatives, such as computed tomography angiography (CTA), lack the hemodynamic precision required for definitive diagnosis.  

Prior attempts to quantify pressure gradients noninvasively using PC-MRI have been explored in larger vessels like the aorta and carotid arteries. However, the coronary arteries present unique challenges due to their small diameter, complex motion, and high flow velocities. Existing MRI techniques suffer from low reproducibility, motion artifacts, and insufficient spatiotemporal resolution to reliably measure coronary flow dynamics.  

There is thus an unmet need for a robust, noninvasive method to assess coronary stenosis with accuracy comparable to invasive FFR. The present invention addresses this need by introducing a reproducible PC-MRI protocol optimized for coronary applications, enabling precise ΔP and velocity measurements for functional stenosis evaluation.  

## SUMMARY  

The invention provides a method for noninvasively determining pressure gradients and flow velocities in coronary arteries using phase-contrast MRI (PC-MRI). The method comprises acquiring two-cardiac-phase velocity data at mid-diastole and end-expiration via electrocardiogram (ECG)-triggering and navigator-gating on a high-field MRI system (e.g., 3T). Key innovations include:  

1. **Optimized k-space phase-encoding ordering** to permit offline view sharing, accommodating acquisition windows exceeding the coronary quiescent period (~100 ms).  
2. **Multi-directional velocity encoding** (vx, vy, vz) with variable velocity sensitivity (VENC = 30–45 cm/s) tailored to coronary flow profiles.  
3. **Slice-specific acquisition parameters**, including in-plane resolution (0.58–0.67 mm), slice thickness (3.2 mm), and flip angle (15°), to balance signal-to-noise ratio and motion robustness.  
4. **Navier-Stokes-based ΔP calculations** to derive pressure gradients between adjacent slices, validated in both healthy volunteers and flow phantoms.  

The method demonstrates excellent reproducibility in through-plane velocities (ICC = 0.93–0.96) and moderate reproducibility in ΔP measurements (ICC = 0.51), establishing feasibility for noninvasive FFR estimation. Further refinements to reduce noise and improve ΔP reliability are disclosed.  

## DETAILED DESCRIPTION  

The invention is described in detail with reference to the following embodiments:  

### MRI Acquisition Protocol  
The method employs a 3T MRI scanner (e.g., Siemens MAGNETOM Verio) equipped with ECG-triggering and navigator-gating capabilities. Data acquisition is synchronized to mid-diastole and end-expiration to minimize cardiac and respiratory motion artifacts. A 2D PC-MRI sequence is used to measure velocity fields (vx, vy, vz) in a single cross-section per acquisition, with 4–5 consecutive slices obtained in the proximal left anterior descending (LAD) artery.  

Key parameters include:  
- **Temporal resolution**: 65–71 ms per phase, with the first phase strictly coinciding with the quiescent period.  
- **Spatial resolution**: In-plane = 0.58–0.67 mm; slice thickness = 3.2 mm.  
- **Flip angle**: 15° to optimize blood-to-tissue contrast.  
- **VENC**: 30–45 cm/s for each flow encoding direction, determined via a preliminary VENC scout scan.  

### View Sharing and Motion Compensation  
To address the challenge of limited quiescent periods, the invention incorporates a k-space phase-encoding scheme enabling offline view sharing. This allows reconstruction of velocity data even when the acquisition window exceeds 100 ms. Navigator-gating further reduces respiratory motion artifacts by rejecting data acquired outside a predefined diaphragm position threshold.  

### Pressure Gradient Calculation  
The Navier-Stokes equations are applied to the acquired velocity fields to compute ΔP between adjacent slices. The method accounts for viscous losses, convective acceleration, and vessel geometry to derive physiologically relevant pressure drops.  

### Reproducibility Validation  
The invention includes a reproducibility assessment protocol involving:  
1. **Healthy volunteer studies**: Two repeat scans per subject to evaluate intra-session variability.  
2. **Flow phantom studies**: A silicone tubing phantom with 40% stenosis (inner diameter = 4.8 mm) perfused with gadolinium-doped water (flow rate = 300 mL/min) to validate ΔP accuracy under controlled conditions.  

Statistical analysis via intra-class correlation (ICC) confirms high reproducibility for through-plane velocities (vz) and moderate reproducibility for ΔP, supporting clinical translatability.  

## EXAMPLES  

### Example 1: Volunteer Velocity Measurements  
In a cohort of four healthy volunteers, the method achieved an average maximum through-plane velocity (vz) of 16.5 ± 4.0 cm/s across 19 acquired slices. ICC values for vz were 0.93 (cardiac phase 1) and 0.96 (cardiac phase 2), demonstrating excellent repeatability. In-plane velocities (vx, vy) showed slightly lower ICCs (0.78–0.86), attributable to residual motion artifacts.  

### Example 2: Phantom Stenosis Validation  
The flow phantom with 40% stenosis exhibited strong correlations in all velocity directions (ICC > 0.90) and ΔP measurements, confirming the method’s accuracy in simulating pathological conditions.  

### Volunteer Studies  
ΔP measurements in volunteers yielded an average of 0.1039 ± 0.28 mmHg with an ICC of 0.51. While ΔP reproducibility was lower than velocities, the results establish feasibility for noninvasive FFR estimation, with ongoing refinements targeting improved ΔP precision.  

### Example 3: Clinical Translation  
Preliminary data support the method’s potential to replace invasive FFR in intermediate stenosis cases. Patient studies are underway to define ΔP and FFR thresholds for distinguishing hemodynamically significant lesions.  

---  

This patent application provides a comprehensive, standalone description of the invention while adhering to the specified outline. Each section is fully elaborated with technical and clinical details to meet formal patent drafting standards.