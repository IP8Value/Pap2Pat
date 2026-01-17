# DESCRIPTION

## FIELD OF THE INVENTION

The present invention pertains to the field of medical imaging and, more specifically, to a novel method and system for assessing diastolic function using cardiovascular magnetic resonance (CMR) imaging. The invention introduces a strain relaxation index (SRI) that quantifies myocardial relaxation and tissue compliance, providing a comprehensive and accurate assessment of diastolic function.

## BACKGROUND OF THE INVENTION

Diastolic function, which refers to the ability of the heart to relax and fill with blood, is a critical aspect of cardiac health. Impaired diastolic function can lead to various cardiovascular conditions, including heart failure with preserved ejection fraction (HFpEF). Traditional methods for assessing diastolic function, such as echocardiography (echo), have limitations in terms of accuracy and reproducibility. Echo parameters, such as tissue Doppler imaging (TDI) and pulse-wave Doppler, provide valuable information but may not fully capture the complex dynamics of myocardial relaxation and compliance.

Cardiovascular magnetic resonance (CMR) imaging has emerged as a powerful tool for evaluating cardiac function due to its high spatial and temporal resolution. CMR techniques, such as harmonic phase (HARP) analysis, can measure myocardial deformation and strain, offering detailed insights into myocardial mechanics. However, there is a need for a robust and standardized method to integrate these measurements into a comprehensive assessment of diastolic function.

The present invention addresses this need by introducing a novel strain relaxation index (SRI) derived from CMR data. SRI combines measures of myocardial relaxation and tissue compliance, providing a more accurate and reliable assessment of diastolic function compared to traditional echo parameters. This invention has significant implications for the diagnosis and management of cardiovascular diseases, particularly those involving diastolic dysfunction.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for assessing diastolic function using a novel strain relaxation index (SRI) derived from cardiovascular magnetic resonance (CMR) imaging. The SRI is calculated based on myocardial deformation and strain rate measurements obtained from HARP analysis of CMR images. Specifically, SRI is defined as the difference between post-systolic and systolic times of the strain peaks, divided by the early diastolic strain rate peak, and normalized by the total relaxation time.

The invention includes the following steps:
1. Acquiring CMR images of the left ventricle (LV) using a tagging technique.
2. Performing HARP analysis to compute mid-ventricular mid-wall circumferential strains and strain rates.
3. Calculating the SRI using the formula:
   \[
   \text{SRI} = \frac{\text{(Post-systolic time of strain peak - Systolic time of strain peak)}}{\text{Early diastolic strain rate peak}} \times \frac{1}{\text{Total relaxation time}}
   \]
   where the total relaxation time is the difference between the RR interval and the systolic interval.
4. Relating the SRI to other CMR and echo parameters, such as mass-to-volume ratio (MVR) and tissue Doppler velocities, to assess diastolic function.

The invention further provides a system for implementing the method, comprising:
- A CMR imaging device for acquiring tagged CMR images.
- A processing unit for performing HARP analysis and calculating the SRI.
- A display unit for visualizing the results and comparing them with standard diastolic parameters.

The SRI offers several advantages over existing methods:
- **Comprehensive Assessment**: SRI accounts for both myocardial relaxation and tissue compliance, providing a more holistic view of diastolic function.
- **High Accuracy**: CMR imaging and HARP analysis offer high spatial and temporal resolution, ensuring precise measurements.
- **Standardization**: The SRI can be easily standardized and integrated into clinical workflows, facilitating consistent and reliable assessments.

## DETAILED DESCRIPTION

### Acquisition of CMR Images

The first step in the method involves acquiring CMR images of the left ventricle (LV) using a tagging technique. Tagging involves applying a grid or pattern of lines to the myocardium, which allows for the tracking of myocardial motion and deformation. The CMR imaging is performed using a high-field MRI scanner, typically 1.5T or 3T, equipped with a cardiac-specific coil. The imaging protocol includes the acquisition of cine images and tagged images during a single breath-hold.

### HARP Analysis

Once the CMR images are acquired, they are processed using harmonic phase (HARP) analysis. HARP analysis is a computational technique that extracts myocardial strain and strain rate information from tagged CMR images. The HARP algorithm identifies the displacement of the tag lines over time and calculates the corresponding strain and strain rate values. This process is performed for the mid-ventricular mid-wall region, focusing on circumferential strains and strain rates.

### Calculation of SRI

The SRI is calculated using the following steps:
1. **Determine Strain Peaks**: Identify the systolic and post-systolic times of the strain peaks from the HARP analysis. The systolic time corresponds to the end of systole, while the post-systolic time is the time at which the strain returns to baseline after systole.
2. **Calculate Early Diastolic Strain Rate Peak**: Determine the peak value of the early diastolic strain rate from the HARP analysis. This value reflects the rate of myocardial relaxation during the early diastolic phase.
3. **Compute Total Relaxation Time**: Calculate the total relaxation time as the difference between the RR interval (the time between two consecutive R waves in the electrocardiogram) and the systolic interval (the duration of systole).
4. **Calculate SRI**: Use the formula:
   \[
   \text{SRI} = \frac{\text{(Post-systolic time of strain peak - Systolic time of strain peak)}}{\text{Early diastolic strain rate peak}} \times \frac{1}{\text{Total relaxation time}}
   \]

### Relation to Other Parameters

The SRI is then related to other CMR and echo parameters to assess diastolic function. These parameters include:
- **Mass-to-Volume Ratio (MVR)**: MVR is calculated as the ratio of LV mass to end-diastolic volume, both of which are determined from CMR images using the Simpson method. An increased MVR is indicative of diastolic dysfunction.
- **Tissue Doppler Velocities**: Tissue Doppler imaging (TDI) is used to measure the lateral and septal diastolic tissue velocities (e' waves) from echo. These velocities reflect the speed of myocardial relaxation.
- **E/e' Ratio**: The E/e' ratio is calculated by dividing the peak early diastolic filling velocity (E wave) by the average of the septal and lateral e' waves. A higher E/e' ratio suggests impaired diastolic function.

### Statistical Analysis

Statistical analysis is performed to evaluate the relationship between SRI and other parameters. Pearson's correlation coefficient is used to assess the linear relationship between SRI and MVR, as well as between SRI and echo parameters. Analysis of variance (ANOVA) is employed to test for differences in SRI across different levels of diastolic function, as rated by the number of matched criteria from echo parameters.

### Example

To illustrate the application of the SRI, consider a study involving 125 participants from the Multi-Ethnic Study of Atherosclerosis (MESA). The participants underwent both echo and tagged CMR on the same day at the Johns Hopkins Hospital. The mean values for MVR, septal e', lateral e', E/e', and SRI were 0.97±0.17 g/mL, 9.1±2.3 cm/s, 10.6±3.1 cm/s, 8.0±2.5, and 2.94±1.4 ms, respectively. The SRI was found to correlate positively with MVR (r=0.42, p < 0.001) and E/e' (r=0.31, p < 0.001), but negatively with e' values (septal r = -0.28, p < 0.001; lateral r = -0.22, p = 0.01). Additionally, SRI showed an increasing trend across diastolic function ratings (p = 0.03).

These results demonstrate the utility of SRI in assessing diastolic function and its strong correlation with established markers of diastolic dysfunction. The SRI provides a novel and comprehensive approach to evaluating diastolic function, offering improved accuracy and reliability compared to traditional methods.

### Conclusion

The present invention introduces a novel strain relaxation index (SRI) for assessing diastolic function using cardiovascular magnetic resonance (CMR) imaging. The SRI combines measures of myocardial relaxation and tissue compliance, providing a more accurate and comprehensive assessment of diastolic function. The method and system described herein offer significant advantages over existing techniques, making them valuable tools for the diagnosis and management of cardiovascular diseases.