Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FIELD  

The present invention relates generally to the field of cardiac electrophysiology and, more particularly, to systems and methods for analyzing intracardiac electrograms to identify and characterize wavefront propagation patterns during atrial fibrillation. Specifically, the invention provides novel techniques for determining regional dominant frequency (RDF) and wave break rate (WBR) from intracardiac electrogram signals, enabling improved identification of arrhythmia sources and guidance for ablation therapy.  

## BACKGROUND  

Atrial fibrillation (AF) remains the most common sustained cardiac arrhythmia and a significant cause of stroke and heart failure. Current treatment strategies often involve catheter ablation procedures that aim to electrically isolate pulmonary veins or target areas exhibiting complex fractionated electrograms. However, these approaches yield suboptimal outcomes, particularly in persistent AF cases, due to an incomplete understanding of the underlying mechanisms that sustain AF.  

Existing methods for analyzing intracardiac electrograms (IEGMs) during AF rely on parameters such as dominant frequency (DF), complex fractionated electrogram mean intervals (CFEmean), and activation timing measurements. While these metrics provide some insight into local electrical activity, they fail to adequately characterize dynamic wavefront propagation patterns and discontinuities that may indicate critical sites for AF perpetuation.  

There exists an unmet need for improved signal processing techniques that can more accurately identify and quantify transient changes in wavefront propagation during AF. Such methods would enable better differentiation between active arrhythmia sources and passive collision sites, thereby improving the efficacy of ablation procedures.  

## SUMMARY  

The present invention provides novel systems and methods for analyzing intracardiac electrograms to characterize wavefront propagation during atrial fibrillation. The disclosed techniques involve processing IEGM signals to determine both regional dominant frequency (RDF) and wave break rate (WBR), which together provide valuable information about wavefront dynamics and potential arrhythmia sources.  

Key aspects of the invention include:  

1. A signal processing method that transforms raw IEGM signals into smoothed representations suitable for frequency analysis while preserving transient wavefront characteristics.  

2. Calculation of instantaneous electrode dominant frequency (iEDF) for individual bipolar electrode pairs using short-time Fourier transform analysis with optimized time windows.  

3. Determination of regional dominant frequency (RDF) through averaging and filtering of processed signals from multiple adjacent electrodes, enabling identification of wavefront propagation patterns across a cardiac region.  

4. Identification of wave breaks as transient decreases in RDF below defined thresholds, with quantification of wave break rate (WBR) as a measure of wavefront discontinuity.  

5. Establishment of minimum recording durations (4 seconds for RDF, 25 seconds for WBR) required for reliable parameter estimation.  

6. Correlation of RDF and WBR patterns with clinical outcomes, particularly the identification of sites exhibiting both high RDF and low WBR (↑RDF,↓WBR) that may represent critical locations for AF termination.  

The invention further provides systems implementing these methods, including specialized catheter configurations and processing algorithms optimized for real-time analysis during electrophysiology procedures. Clinical applications include improved guidance for ablation therapy in both paroxysmal and persistent AF cases.  

## DETAILED DESCRIPTION  

### 1A. Regional Dominant Frequency and Wave Break Rate  

The present invention provides a novel approach for analyzing intracardiac electrograms (IEGMs) to characterize wavefront propagation during atrial fibrillation. The method involves processing signals from a multi-electrode catheter to determine both regional dominant frequency (RDF) and wave break rate (WBR), which together provide valuable information about wavefront dynamics.  

The processing begins with preprocessing individual bipolar electrogram signals to generate smooth pulse trains representing local electrical activity. This preprocessing step replaces complex electrogram morphologies with standardized waveforms while preserving timing information. For each electrode pair, the instantaneous electrode dominant frequency (iEDF) is calculated using short-time Fourier transform (STFT) analysis with a 1-second time window and 95% overlap, providing both time and frequency resolution suitable for tracking transient changes.  

To determine RDF, the preprocessed signals from all catheter electrodes are averaged, filtered using a specialized two-sided exponential finite impulse response (FIR) filter, and analyzed to extract the upper quartile of the instantaneous frequency distribution. This regional measure reflects coordinated electrical activity across the catheter's coverage area, typically spanning approximately 2.5 cm in diameter.  

Wave breaks are identified as transient decreases in RDF exceeding 3 Hz below the baseline or dropping below 0.5 Hz, lasting longer than 100 milliseconds. The wave break rate (WBR) is then calculated as the number of such events per second, providing a quantitative measure of wavefront discontinuity in the recorded region.  

### 1B. Example of RDF-Based Wave Break Identification  

An exemplary application of the RDF-based wave break identification method is illustrated through analysis of electrograms recorded from the left atrial roof during persistent AF. When a clear, organized wavefront propagates past the catheter, the preprocessed signals from all electrodes show nearly simultaneous peaks, resulting in a strong averaged signal with well-defined frequency components.  

During wave break events, activation times become dispersed across the electrode array, causing the averaged signal to develop multiple smaller peaks over an extended time interval. This temporal dispersion manifests as high-frequency components in the unfiltered signal, which are subsequently attenuated by the lowpass filtering stage. The resulting drop in RDF provides a clear marker of wave break occurrence.  

In one representative example, analysis of a 30-second recording identified three distinct wave break events, corresponding to a WBR of 0.1 breaks per second. This quantitative measure allows comparison of wavefront stability across different cardiac regions and between patients.  

### 1C. Minimum Required Segment Duration for Accurate RDF Estimation  

The invention establishes minimum recording durations required for reliable estimation of both RDF and WBR. For RDF determination, comparative analysis using segments of varying lengths demonstrated that 4 seconds of recording provides an accurate estimate (90% correlation) compared to the gold standard 30-second reference.  

This finding enables efficient procedural workflow by identifying the shortest recording duration that still yields clinically useful RDF measurements. The 4-second threshold represents an optimal balance between time resolution (for detecting transient changes) and frequency resolution (for accurate dominant frequency determination).  

### 1D. Minimum Required Segment Duration for Accurate WBR Estimation  

Similarly, systematic evaluation established that 25 seconds represents the minimum recording duration required for reliable WBR estimation (correlation >85% with 50-second reference). This longer duration reflects the need to capture sufficient wave break events for statistically meaningful rate calculation.  

The differential requirements for RDF (4 seconds) versus WBR (25 seconds) estimation enable a two-stage analysis approach during procedures: initial brief recordings to identify regions of interest based on RDF, followed by extended recordings at selected sites for comprehensive WBR characterization.  

### 1E. Statistics  

Clinical validation studies incorporating data from 15 patients (5 paroxysmal, 10 persistent AF) demonstrated significant differences in both RDF and WBR between AF types. Paroxysmal AF showed higher mean RDF (5.99±0.8 Hz vs 5.32±0.75 Hz) and WBR (0.24±0.14 vs 0.14±0.11 breaks/sec) compared to persistent AF (p<0.001).  

Spatial analysis revealed heterogeneous distribution of these parameters across the left atrium, with weak overall correlation between RDF and WBR (r=0.3). This suggests that the two measures provide complementary information about wavefront characteristics.  

### 1F. Implementation  

The invention may be implemented through specialized catheter systems incorporating high-density electrode arrays (e.g., 20-electrode spiral or circular configurations) coupled to signal processing hardware and software. Real-time implementation involves:  

1. Signal acquisition and preprocessing modules to transform raw electrograms into standardized pulse trains.  

2. Parallel computation of iEDF for individual electrode pairs using optimized STFT parameters.  

3. Regional signal averaging and filtering using the disclosed two-sided exponential FIR filter.  

4. RDF calculation through spectral analysis of the filtered regional signal.  

5. Wave break detection and WBR calculation algorithms operating on the RDF time series.  

6. Graphical display integration with electroanatomic mapping systems, presenting RDF and WBR data as color-coded overlays on 3D atrial geometries.  

### 1G. Results  

Clinical application in 15 patients demonstrated that ablation at sites exhibiting both high RDF and low WBR (↑RDF,↓WBR) frequently resulted in AF termination (8/9 cases where such sites were ablated). These sites were identified in 14/15 patients, averaging 2.6±1.2 per patient, with distribution varying between paroxysmal (predominantly pulmonary vein locations) and persistent AF (more widespread distribution).  

Notably, patients where ablation terminated AF to sinus rhythm showed no recurrences during follow-up (mean 24.5 months), while those requiring cardioversion had higher recurrence rates (4/6 cases). This suggests that ↑RDF,↓WBR sites may identify critical locations for AF maintenance.  

### 1H. Discussion  

The RDF/WBR analysis method provides several advantages over existing techniques:  

1. It avoids reliance on precise activation timing determination, which is problematic in fractionated electrograms.  

2. The regional analysis approach is more robust than single-electrode measurements for characterizing wavefront propagation.  

3. The combination of RDF and WBR helps differentiate active sources (high RDF, low WBR) from passive collision sites (variable RDF, high WBR).  

4. The method works effectively with sequential point-by-point mapping, avoiding limitations of simultaneous panoramic mapping systems.  

These features make the invention particularly suitable for analyzing complex substrates in persistent AF, where traditional approaches have shown limited efficacy.  

## 2. Computer Modelling of Spiral Rotor and Associated Wave Break Analysis  

Computer simulations of spiral wave reentry provide theoretical validation for the RDF/WBR analysis approach. In these models, the rotor core exhibits high, stable frequency (analogous to high RDF) with minimal wave break activity (low WBR), while surrounding areas show more variable frequencies and frequent wave breaks due to wavefront fragmentation.  

This pattern matches clinical observations of ↑RDF,↓WBR sites potentially representing rotor cores, with the surrounding atrium showing higher WBR values. The modeling further confirms that RDF analysis can identify rotor activity without requiring complete panoramic activation mapping.  

## 3. Clinical Example of an Identified Rotor During Wave Break  

A representative case illustrates the clinical application of the method. In a persistent AF patient, analysis revealed a distinct ↑RDF,↓WBR site posterior to the right inferior pulmonary vein. Ablation lesions crossing this region resulted in immediate AF termination to sinus rhythm.  

Electrogram analysis at this site showed organized, high-frequency activity (RDF=6.2 Hz) with infrequent wave breaks (WBR=0.08/s), surrounded by areas exhibiting more fractionated signals and higher WBR (0.3-0.4/s). This pattern suggests the ablated site may have represented a stable rotor core maintaining the arrhythmia.  

## EQUIVALENTS  

While the invention has been described with reference to specific embodiments, various modifications and equivalent implementations will be apparent to those skilled in the art. These include:  

1. Alternative electrode configurations and catheter designs for signal acquisition.  

2. Variations in signal processing parameters (e.g., filter characteristics, time window durations).  

3. Different threshold values for wave break identification.  

4. Integration with various electroanatomic mapping systems.  

5. Application to other cardiac arrhythmias beyond atrial fibrillation.  

All such modifications that incorporate the fundamental principles of RDF and WBR analysis are considered within the scope of the present invention.