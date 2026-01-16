# DESCRIPTION

## FIELD

The field of the present invention pertains to medical devices and methods for analyzing and treating atrial fibrillation (AF). Specifically, the invention relates to a novel method and system for quantifying regional wavefront discontinuities, or wave breaks, in atrial electrograms (IEGMs) to guide catheter ablation therapy.

## BACKGROUND

Atrial fibrillation (AF) is the most common cardiac arrhythmia and a primary cause of stroke. It is characterized by heterogeneous spatiotemporal wavefront propagation, leading to complex signal formation and fragmentation. Traditional methods for guiding catheter ablation therapy, such as the use of relative timing of pulmonary vein potentials, duration of bipolar electrograms, complex fractionated electrograms (CFE), and dominant frequency (DF), have been insufficient in fully understanding the underlying mechanisms of AF perpetuation. Consequently, outcomes of ablation therapy remain suboptimal, particularly in persistent AF cases.

There is a need for a more sophisticated and accurate method to analyze the dynamic wavefront characteristics of AF. The present invention addresses this need by introducing a novel approach that quantifies regional wavefront discontinuities, or wave breaks, using short-term changes in regional dominant frequency (RDF). This method provides a more detailed and dynamic understanding of wavefront propagation, which can be used to identify critical areas for ablation therapy.

## SUMMARY

The present invention provides a method and system for analyzing intracardiac electrograms (IEGMs) to identify and quantify regional wavefront discontinuities, or wave breaks, in atrial fibrillation (AF). The method involves the following steps:

1. **Preprocessing of IEGMs**: The IEGMs recorded from multiple electrode pairs of a high-density mapping catheter are preprocessed to generate smooth trains of pulses from the electrograms.
2. **Calculation of Instantaneous Electrode Pair Dominant Frequency (iEDF)**: The power spectrum of each preprocessed IEGM is calculated using short-time Fourier transform (STFT) with a time window of 1 second and 95% overlap. The instantaneous electrode pair dominant frequency (iEDF) is determined from the power spectrum.
3. **Calculation of Regional Dominant Frequency (RDF)**: The preprocessed signals from all electrode pairs are averaged, smoothed, and the mean amplitude is removed. The power spectrum of the resulting signal is used to estimate the regional dominant frequency (RDF).
4. **Wave Break Identification**: Wave breaks are identified as drops in the instantaneous RDF (iRDF) more than 3 Hz below the RDF, lasting longer than 100 milliseconds. The wave break rate (WBR) is calculated as the number of wave breaks per second.
5. **Mapping and Visualization**: The RDF and WBR values are mapped onto the atrial geometry and visualized using color-coding to identify regions with high RDF and low WBR (↑RDF,↓WBR), which are potential targets for ablation therapy.

The invention also includes a computer-implemented method for modeling the spiral rotor and analyzing wave breaks, as well as a clinical example demonstrating the identification of a rotor during wave break.

## DETAILED DESCRIPTION

### 1A. Regional Dominant Frequency and Wave Break Rate

The present invention introduces a novel method for quantifying regional wavefront discontinuities, or wave breaks, in atrial fibrillation (AF) using intracardiac electrograms (IEGMs). The method involves the following steps:

1. **Preprocessing of IEGMs**: The IEGMs recorded from multiple electrode pairs of a high-density mapping catheter are preprocessed to generate smooth trains of pulses from the electrograms. This preprocessing step replaces the complex morphologies of the IEGMs with smooth Gaussian shape pulses, making it easier to identify wavefront characteristics.
2. **Calculation of Instantaneous Electrode Pair Dominant Frequency (iEDF)**: The power spectrum of each preprocessed IEGM is calculated using short-time Fourier transform (STFT) with a time window of 1 second and 95% overlap. The Hanning window is applied to each segment, and the power spectrum is estimated using the fast Fourier transform (FFT). The instantaneous electrode pair dominant frequency (iEDF) is determined as the frequency corresponding to the maximum power in the power spectrum.
3. **Calculation of Regional Dominant Frequency (RDF)**: The preprocessed signals from all electrode pairs are averaged, smoothed using a two-sided exponential finite impulse response (FIR) filter, and the mean amplitude is removed. The power spectrum of the resulting signal is used to estimate the regional dominant frequency (RDF). The upper quartile of the power spectrum is reported as the RDF.
4. **Wave Break Identification**: Wave breaks are identified as drops in the instantaneous RDF (iRDF) more than 3 Hz below the RDF, lasting longer than 100 milliseconds. The wave break rate (WBR) is calculated as the number of wave breaks per second.
5. **Mapping and Visualization**: The RDF and WBR values are mapped onto the atrial geometry and visualized using color-coding to identify regions with high RDF and low WBR (↑RDF,↓WBR), which are potential targets for ablation therapy.

### 1B. Example of RDF-Based Wave Break Identification

An example of RDF-based wave break identification is illustrated using IEGMs recorded from the roof of the left atrium of a patient with persistent AF. The preprocessed signals from all electrode pairs show clear wavefronts when a single wavefront is present. However, during wave breaks, the delays between the activation times (ATs) of the electrodes increase, resulting in several small peaks in the averaged signal. This high-frequency component is attenuated by the lowpass filter, leading to a drop in the iRDF. In this example, three wave breaks were identified, and the WBR was estimated as 0.1 WB/s.

### 1C. Minimum Required Segment Duration for Accurate RDF Estimation

To ensure accurate and robust estimation of the RDF, the minimum required segment duration was determined. The data obtained using a 30-second segment was considered the gold standard. By comparing the RDF obtained from segments of varying durations, it was found that segments longer than 4 seconds provide an accurate estimate of the RDF, with a Pearson correlation of 90% with the 30-second gold standard.

### 1D. Minimum Required Segment Duration for Accurate WBR Estimation

Similarly, the minimum required segment duration for reliable estimation of the WBR was determined. Segments longer than 50 seconds were used as the gold standard. By comparing the WBR obtained from segments of varying durations, it was found that segments longer than 25 seconds provide a reliable estimate of the WBR, with a Pearson correlation of 90% with the 50-second gold standard.

### 1E. Statistics

Statistical analysis was performed to compare the RDF and WBR values between patients with paroxysmal and persistent AF. The Anderson-Darling test was used to check for normality, and non-parametric data was compared using the Mann-Whitney test. Spearman's rank correlation coefficient was used to study the correlation between WBR and RDF. The mean and standard deviation of the variables were reported using the mean ± std notation.

### 1F. Implementation

The method was implemented using a high-density mapping catheter, such as the Reflexion™ Spiral catheter, which has 20 electrodes and a diameter of 25 mm. The bipolar pair electrodes spacing for the Reflexion™ Spiral catheter is 1 mm. The IEGMs were recorded at a sampling frequency of 2034.5 Hz and exported to a MATLAB environment for signal processing. The RDF and WBR values were calculated and mapped onto the atrial geometry using an electroanatomic mapping (EAM) system, such as the EnSite™ Velocity™ system.

### 1G. Results

The method was tested on 15 patients, including 5 paroxysmal and 10 persistent AF patients. The mean RDF of the segments was 5.5 ± 0.82 Hz, and the WBR was 0.16 ± 0.13 WB/s. The RDF and WBR values were significantly different between paroxysmal and persistent patients (p < 0.001). The WBR and RDF were weakly correlated (0.3; p < 0.001). Sites with high RDF and low WBR (↑RDF,↓WBR) were identified in 14/15 patients, with a mean of 2.6 ± 1.2 sites per patient. Ablation at these sites was associated with termination of AF in 8/9 patients.

### 1H. Discussion

The present invention provides a novel method for quantifying regional wavefront discontinuities, or wave breaks, in atrial fibrillation (AF). The method uses short-term changes in regional dominant frequency (RDF) to identify and quantify wave breaks, which can be used to guide catheter ablation therapy. The method is computationally efficient and provides a more detailed and dynamic understanding of wavefront propagation compared to traditional methods. The identification of sites with high RDF and low WBR (↑RDF,↓WBR) is associated with termination of AF, suggesting that these sites may be critical areas for ablation therapy.

## 2. Computer Modelling of Spiral Rotor and Associated Wave Break Analysis

The invention also includes a computer-implemented method for modeling the spiral rotor and analyzing wave breaks. The method involves simulating the spiral rotor using a computational model and analyzing the resulting wavefronts to identify wave breaks. The model can be used to predict the behavior of wavefronts in different regions of the atrium and to optimize the placement of ablation lesions.

## 3. Clinical Example of an Identified Rotor During Wave Break

A clinical example is provided to demonstrate the identification of a rotor during wave break. In a patient with persistent AF, a high-density mapping catheter was used to record IEGMs from the left atrium. The RDF and WBR values were calculated and mapped onto the atrial geometry. A site with high RDF and low WBR (↑RDF,↓WBR) was identified near the right inferior pulmonary vein (RIPV). Ablation at this site resulted in the termination of AF to sinus rhythm, confirming the presence of a rotor at this location.

## EQUIVALENTS

While specific embodiments of the invention have been described, it should be understood that the invention is not limited to the specific forms set forth herein. Various modifications, substitutions, and variations can be made without departing from the spirit and scope of the invention. For example, the method can be applied to other types of high-density mapping catheters and can be integrated with other electroanatomic mapping systems. The invention is intended to cover all such modifications, substitutions, and variations as fall within the scope of the appended claims.