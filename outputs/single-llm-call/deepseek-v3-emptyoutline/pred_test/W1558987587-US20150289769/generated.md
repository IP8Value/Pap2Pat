Below is the complete patent application drafted according to the provided outline, incorporating the invention described in the research paper while adhering to formal patent language and structure.  

---  

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of cardiovascular diagnostics, specifically to non-invasive methods for assessing diastolic function of the left ventricle (LV) using cardiac magnetic resonance (CMR) imaging. More particularly, the invention introduces a novel strain relaxation index (SRI) derived from myocardial deformation analysis during LV relaxation, providing a quantitative measure of diastolic dysfunction. The SRI integrates both early myocardial relaxation dynamics and tissue compliance, offering a comprehensive diagnostic tool that correlates with established echocardiographic parameters and mass-to-volume ratio (MVR) measurements. This invention is applicable in clinical cardiology for early detection, risk stratification, and monitoring of diastolic heart failure and related conditions.  

## BACKGROUND OF THE INVENTION  

Diastolic dysfunction, characterized by impaired relaxation and filling of the left ventricle, is a significant contributor to heart failure with preserved ejection fraction (HFpEF). Current diagnostic methods rely heavily on echocardiography, particularly tissue Doppler imaging (TDI) and pulse-wave Doppler measurements, which assess parameters such as early diastolic tissue velocity (e' wave) and the ratio of early mitral inflow velocity to e' (E/e'). While these techniques are widely used, they are subject to limitations including operator dependency, acoustic window constraints, and variability in measurements.  

Cardiac magnetic resonance (CMR) imaging provides superior spatial resolution and tissue characterization but has been underutilized in diastolic function assessment due to the lack of standardized quantitative indices. Existing CMR-based approaches, such as LV mass-to-volume ratio (MVR), offer indirect insights into diastolic dysfunction but fail to capture the dynamic aspects of myocardial relaxation and compliance. There remains an unmet need for a robust, CMR-derived biomarker that integrates both temporal and mechanical properties of diastolic function.  

The present invention addresses these limitations by introducing the strain relaxation index (SRI), a novel metric derived from tagged CMR analysis. The SRI quantifies the interplay between post-systolic strain timing and early diastolic strain rate, normalized by the total relaxation period. This index correlates strongly with echocardiographic diastolic parameters and MVR, providing a standalone or complementary diagnostic tool for comprehensive diastolic assessment.  

## SUMMARY OF THE INVENTION  

The invention provides a method for assessing diastolic function in a subject by calculating a strain relaxation index (SRI) from tagged cardiac magnetic resonance (CMR) imaging data. The SRI is computed as the difference between post-systolic and systolic peak strain times, divided by the early diastolic strain rate peak, and normalized by the total relaxation time (RR interval minus systolic interval). This index reflects both myocardial relaxation kinetics and tissue compliance, offering a direct measure of diastolic dysfunction.  

Key advantages of the invention include:  
- **Objective quantification**: Unlike echocardiography, SRI is derived from standardized CMR deformation analysis, reducing inter-observer variability.  
- **Comprehensive assessment**: SRI integrates multiple diastolic parameters (relaxation delay and compliance) into a single index.  
- **Strong correlation with gold standards**: Validation studies demonstrate significant correlations between SRI and echocardiographic markers (e', E/e') as well as CMR-derived MVR.  
- **Clinical utility**: The index stratifies diastolic dysfunction severity, aiding in early diagnosis and therapeutic monitoring.  

The method further includes steps for acquiring tagged CMR images, processing myocardial strain data using harmonic phase analysis, and deriving the SRI for diagnostic interpretation. Applications extend to risk prediction in hypertensive, diabetic, and aging populations prone to diastolic impairment.  

## DETAILED DESCRIPTION  

The invention is implemented through the following steps:  

1. **Image Acquisition**:  
   - A subject undergoes tagged CMR imaging, wherein myocardial tissue is magnetically labeled to enable deformation tracking.  
   - Mid-ventricular mid-wall circumferential strains and strain rates are acquired during the cardiac cycle.  

2. **Strain Analysis**:  
   - Harmonic phase (HARP) analysis processes tagged images to generate strain curves, identifying systolic peak strain (ε_sys) and post-systolic peak strain (ε_ps).  
   - Early diastolic strain rate (SR_ed) is determined as the maximum derivative of strain during early diastole.  

3. **Calculation of SRI**:  
   - The relaxation time delay (ΔT) is computed as the interval between ε_ps and ε_sys.  
   - SRI is derived using the formula:  
     \[
     \text{SRI} = \frac{\Delta T}{\text{SR}_{ed}} \times \frac{1}{T_{\text{relax}}}
     \]  
     where \( T_{\text{relax}} \) is the total relaxation time (RR interval − systolic interval).  

4. **Clinical Correlation**:  
   - SRI values are compared to reference ranges or correlated with echocardiographic parameters (e.g., E/e' ≥ 10 indicates elevated filling pressures).  
   - Higher SRI values denote worse diastolic function, with thresholds validated against diastolic dysfunction grades (0–3).  

The invention further encompasses software algorithms for automated SRI computation and integration with CMR post-processing platforms.  

### EXAMPLE  

In a cohort of 125 subjects (age 61 ± 8 years, 41% male), SRI demonstrated the following clinical correlations:  
- Positive association with MVR (r = 0.42, p < 0.001), reflecting increased LV stiffness.  
- Positive correlation with E/e' (r = 0.31, p < 0.001), aligning with echocardiographic filling pressure estimates.  
- Negative correlations with septal e' (r = −0.28, p < 0.001) and lateral e' (r = −0.22, p = 0.01), consistent with impaired relaxation.  
- Graded increase across diastolic dysfunction severity (p = 0.03), confirming discriminative capacity.  

These results validate SRI as a robust CMR biomarker for diastolic dysfunction, with potential to augment or replace conventional echocardiographic assessments.  

---  

This patent application provides a complete, standalone description of the invention, adhering to formal patent drafting conventions and ensuring comprehensive coverage of the novel strain relaxation index (SRI) and its applications.