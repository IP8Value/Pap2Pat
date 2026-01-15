# DESCRIPTION

## FIELD

- define field of invention

The invention resides in the field of cardiac electrophysiology and medical signal processing, specifically in the detection, quantification, and graphical visualization of spatiotemporal wavefront dynamics during atrial fibrillation using intracardiac electrograms. The invention provides a computational methodology for identifying regions of abnormal electrical activity associated with the perpetuation of atrial fibrillation by analyzing transient changes in regional dominant frequency and wave break rate derived from high-density catheter recordings. This approach enables clinicians to distinguish between stable sources of arrhythmia and passive conduction disturbances, thereby improving the precision of catheter ablation therapy. The system integrates advanced time-frequency analysis, signal fusion techniques, and graphical mapping to transform raw electrogram data into actionable clinical insights, representing a significant advancement over conventional methods reliant on static metrics such as complex fractionated electrograms or average dominant frequency. The invention is particularly suited for use in electrophysiology laboratories during interventional procedures for persistent and paroxysmal atrial fibrillation, where real-time, regionally resolved characterization of wavefront heterogeneity is critical for therapeutic decision-making.

## BACKGROUND

- introduce atrial fibrillation

Atrial fibrillation is the most prevalent sustained cardiac arrhythmia in clinical practice, characterized by disorganized, rapid, and irregular electrical activation of the atria that disrupts normal contractile function and predisposes patients to thromboembolic events, including stroke. The underlying electrophysiological substrate involves heterogeneous propagation of multiple wavefronts that fragment, collide, and rotate within the atrial tissue, resulting in complex and non-repetitive electrogram patterns. These dynamics are not uniformly distributed across the atria, and certain regions may serve as persistent drivers of the arrhythmia, while others reflect passive conduction of wavefronts originating elsewhere. Despite advances in mapping technologies and ablation techniques, the precise identification of these critical driver sites remains elusive, leading to suboptimal long-term outcomes, particularly in patients with persistent forms of the disease.

- describe catheter ablation therapy

Catheter ablation is a widely employed interventional procedure for the treatment of atrial fibrillation, wherein radiofrequency or cryoenergy is delivered through a catheter positioned within the heart to create localized lesions that interrupt abnormal electrical pathways. The most common approach involves pulmonary vein isolation, where circumferential lesions are created around the ostia of the pulmonary veins to electrically disconnect them from the left atrium. In patients with persistent atrial fibrillation, additional substrate-based ablation is often required, targeting regions of complex or fractionated electrograms believed to harbor sustaining mechanisms. However, current strategies for identifying these regions rely on subjective interpretation of electrogram morphology, duration, or frequency content, which lack specificity and reproducibility.

- discuss limitations of conventional methods

Conventional methods for guiding ablation, such as the identification of complex fractionated electrograms based on mean cycle length thresholds, or the measurement of dominant frequency at individual electrode sites, fail to capture the dynamic, spatially distributed nature of wavefront propagation during atrial fibrillation. These approaches are confounded by noise, poor electrode-tissue contact, variable signal amplitude, and the inherent temporal instability of atrial activation patterns. Furthermore, averaging techniques over prolonged time windows obscure transient events such as wave breaks, which are mechanistically linked to rotor formation and wavefront collision. As a result, ablation targets selected using these methods often fail to eliminate the arrhythmia, and recurrence rates remain high, particularly in persistent atrial fibrillation. There is a critical need for a more objective, quantitative, and physiologically grounded method to identify regions of active wavefront disruption that are likely to sustain the arrhythmia.

- highlight need for better understanding

There is an urgent clinical imperative to develop a method that can reliably distinguish between regions that actively perpetuate atrial fibrillation and those that merely reflect passive conduction of wavefronts. Such a method must be capable of detecting subtle, transient changes in wavefront organization at high temporal resolution, while remaining robust to signal artifacts and electrode variability. It must also provide spatially resolved, visually interpretable outputs that can be integrated into existing electroanatomic mapping platforms to guide real-time ablation decisions. The invention fulfills this need by introducing a novel analytical framework that quantifies regional wavefront heterogeneity through the combined assessment of instantaneous regional dominant frequency and wave break rate, offering a mechanistic insight into the dynamics of atrial fibrillation that was previously unattainable with existing technologies.

## SUMMARY

- introduce method for detecting abnormality

A method is disclosed for detecting regions of abnormal electrical activity during atrial fibrillation by analyzing intracardiac electrograms to identify patterns of wavefront discontinuity that correlate with the persistence of the arrhythmia. The method involves the extraction of time-varying features from high-density electrogram recordings, followed by a multi-stage signal processing pipeline that isolates regional dynamics distinct from local electrode-level activity. By focusing on the collective behavior of multiple electrodes within a defined anatomical region, the method overcomes the limitations of single-point measurements and provides a more accurate representation of underlying wavefront organization.

- extract features from intracardiac electrograms

Intracardiac electrograms are acquired from a high-density catheter positioned within the atria, with each electrode pair generating a bipolar signal that reflects local electrical activity. These signals are preprocessed to remove baseline drift, normalize amplitude, and transform complex morphologies into a train of smooth, pulse-like waveforms that represent the timing and integrity of wavefront passage. The preprocessing step eliminates noise and non-physiological artifacts while preserving the temporal relationships between consecutive activations across the catheter.

- use time-frequency analysis to detect heterogeneity

Time-frequency analysis is applied to the processed signals to quantify the dominant frequency of electrical activation at each moment in time. A short-time Fourier transform is performed on overlapping segments of the signal, enabling the estimation of instantaneous dominant frequency with high temporal resolution. The resulting time-varying frequency profile is then aggregated across all electrodes within a region to derive a regional dominant frequency that reflects the collective behavior of the wavefront passing through that anatomical segment.

- combine features to extract regional feature

The regional dominant frequency is further analyzed to detect transient drops in frequency that signify wavefront fragmentation or discontinuity. These drops, referred to as wave breaks, are identified when the instantaneous regional dominant frequency decreases by more than three hertz below the upper quartile of the regional frequency distribution and persists for longer than one hundred milliseconds. The number of such events per second is calculated as the wave break rate, which serves as a quantitative metric of wavefront disorganization.

- exclude irrelevant signals

Signals that lack sufficient amplitude, exhibit poor signal-to-noise ratio, or originate from regions with incomplete catheter coverage are excluded from analysis to ensure that only physiologically meaningful data contributes to the final output. This filtering step enhances the reliability of the regional features and minimizes the influence of non-representative or artifact-laden recordings.

- output results graphically

The calculated regional dominant frequency and wave break rate are mapped onto a three-dimensional electroanatomic geometry of the atria using a color-coded scale, where higher frequencies are represented in red and lower wave break rates in blue, enabling intuitive visualization of spatial heterogeneity. This graphical representation allows clinicians to identify regions of high frequency and low wave break rate, which are associated with the presence of stable arrhythmia drivers.

- identify source of cardiac atrial fibrillation

Regions exhibiting high regional dominant frequency and low wave break rate are identified as potential sources of atrial fibrillation, as these characteristics are consistent with the presence of rotating wavefronts or focal drivers that maintain the arrhythmia without extensive fragmentation. These regions are prioritized for targeted ablation.

- detect change in wavefront dynamics

The method detects dynamic changes in wavefront propagation by continuously monitoring the instantaneous regional dominant frequency and identifying transient deviations that indicate the onset or resolution of wave breaks. This capability enables real-time tracking of wavefront evolution during the procedure.

- determine wave break rate

The wave break rate is determined as the number of identified wave breaks per second within a defined time window, providing a quantitative measure of the degree of wavefront disorganization at each anatomical site.

- display colour-coded map

The final output is a color-coded map overlaid on the electroanatomic geometry, where each point is assigned a color based on its regional dominant frequency and wave break rate, allowing the operator to visually identify and target regions most likely to sustain atrial fibrillation.

## DETAILED DESCRIPTION

- describe methods for analyzing IEGMs to determine wavefront characteristics

The method for analyzing intracardiac electrograms to determine wavefront characteristics involves a multi-stage computational pipeline that transforms raw bipolar electrogram signals into regionally aggregated metrics of wavefront organization. Each electrogram is first preprocessed to remove baseline drift and normalize amplitude, followed by morphological transformation into a train of smooth, Gaussian-shaped pulses that represent the timing of wavefront passage. These pulses are then averaged across all electrodes of the catheter to generate a single regional signal that reflects the collective activation pattern of the underlying tissue. This regional signal is subjected to a two-sided exponential finite impulse response filter to smooth transient fluctuations and enhance the detection of discontinuities in wavefront propagation. The filtered signal is then analyzed using short-time Fourier transform to compute its instantaneous frequency content, from which the regional dominant frequency and its temporal variations are derived. Wave breaks are identified as sustained drops in this frequency, and the rate at which these events occur is calculated as the wave break rate. This approach enables the detection of regions where wavefronts are organized yet rapidly rotating, as opposed to those where wavefronts are fragmented and chaotic.

- introduce block diagram of FIG. 1A

The method is implemented using a block diagram, as illustrated in Figure 1A, which outlines the sequential stages of signal acquisition, preprocessing, regional aggregation, filtering, time-frequency analysis, wave break detection, and graphical mapping. The input consists of high-density intracardiac electrograms recorded from a catheter with multiple bipolar electrode pairs. These signals are processed individually to extract pulse-like waveforms, which are then averaged to form a regional signal. This signal is filtered using a two-sided exponential kernel, followed by short-time Fourier transform to compute the instantaneous regional dominant frequency. The upper quartile of this frequency distribution is determined as the reference value, and deviations below this threshold are classified as wave breaks. The number of such deviations per second is computed as the wave break rate, and both metrics are mapped onto a three-dimensional atrial geometry using a color scale.

- describe feature extraction and fusion

Feature extraction involves the transformation of each bipolar electrogram into a pulse train that represents the timing of wavefront arrival at each electrode. These pulses are then fused by averaging across all electrodes within a defined anatomical region to produce a single representative signal. This fusion process eliminates electrode-specific noise and amplifies the common temporal pattern of wavefront propagation, thereby enhancing the sensitivity to regional changes in activation dynamics. The resulting signal is more robust to variations in electrode-tissue contact and signal amplitude than individual electrode measurements.

- describe time-frequency and/or time-scale analysis of regional features

Time-frequency analysis is performed on the fused regional signal using a short-time Fourier transform with a one-second sliding window and 95% overlap. The Hanning window is applied to each segment to minimize spectral leakage, and the power spectrum is computed using the fast Fourier transform. The frequency corresponding to the maximum power at each time point is identified as the instantaneous regional dominant frequency. This method provides high temporal resolution while maintaining sufficient frequency precision to detect transient changes in activation rate. The upper quartile of the instantaneous regional dominant frequency distribution is used as a reference threshold to identify significant drops indicative of wave breaks.

- describe embodiment of study with twenty patients

The method was validated in a cohort of twenty consecutive patients undergoing catheter ablation for atrial fibrillation. Electrograms were recorded from the left atrium using a high-density circular mapping catheter with twenty electrodes and a diameter of twenty-five millimeters. Data was collected over thirty-second intervals at multiple anatomical locations, with additional recordings of up to one minute obtained in two patients to assess signal stability. All recordings were performed during sustained atrial fibrillation, and the data was processed offline using a custom MATLAB-based algorithm.

- describe data collection and processing

Data was acquired at a sampling rate of two thousand thirty-four point five hertz and exported in digital format for offline analysis. Each bipolar electrogram was preprocessed to remove baseline drift, normalize amplitude, and convert complex morphologies into smooth pulse trains. These pulses were averaged across electrodes to form a regional signal, which was then filtered using a two-sided exponential finite impulse response filter with a length of one hundred eighty samples. The filtered signal was subjected to short-time Fourier transform to compute the instantaneous regional dominant frequency, and wave breaks were identified as sustained drops exceeding three hertz below the upper quartile of the frequency distribution. The wave break rate was calculated as the number of such events per second.

- describe correlation of MATLAB generated maps with procedural outcomes

The color-coded maps generated by the algorithm were overlaid onto the electroanatomic mapping system used during the procedure. Sites where atrial fibrillation terminated following ablation were marked by the operator and correlated with the algorithm’s output. A strong association was observed between termination sites and regions exhibiting high regional dominant frequency and low wave break rate. These regions were identified in fourteen of fifteen patients and were located predominantly within the pulmonary vein antra and posterior left atrial wall. The correlation between algorithm-derived metrics and procedural outcomes was statistically significant, supporting the clinical utility of the method.

### 1A. Regional Dominant Frequency and Wave Break Rate

- describe calculation of electrode dominant frequency (EDF)

The electrode dominant frequency is calculated by applying a short-time Fourier transform to each bipolar electrogram segment, with a one-second window and 95% overlap. The frequency at which the power spectrum reaches its maximum value at each time point is identified as the instantaneous electrode dominant frequency.

- describe preprocessing of IEGMs

Each intracardiac electrogram is preprocessed by removing the mean amplitude, applying a bandpass filter to eliminate low-frequency drift and high-frequency noise, and transforming the complex waveform into a train of smooth, Gaussian-shaped pulses using a threshold-based detection algorithm.

- describe calculation of regional dominant frequency (RDF)

The regional dominant frequency is calculated by averaging the preprocessed signals from all electrodes within a defined region, applying a two-sided exponential finite impulse response filter to smooth the resulting signal, and computing the power spectrum using short-time Fourier transform. The upper quartile of the instantaneous frequency distribution is selected as the regional dominant frequency.

- describe preprocessing steps for RDF calculation

The preprocessing steps for regional dominant frequency calculation include baseline removal, amplitude normalization, pulse transformation, and spatial averaging across electrodes. The resulting signal is then filtered using a two-sided exponential kernel to enhance the detection of wavefront discontinuities.

- describe calculation of instantaneous RDF (iRDF)

The instantaneous regional dominant frequency is computed at each time point by identifying the frequency with maximum power in the short-time Fourier transform of the filtered regional signal.

- describe calculation of upper quartile of iRDF

The upper quartile of the instantaneous regional dominant frequency distribution is determined by sorting all instantaneous values over the recording duration and selecting the value at the seventy-fifth percentile.

- describe selection of time window T

The time window T for the short-time Fourier transform is selected as one second to balance temporal resolution and frequency precision, enabling the detection of transient wave breaks while maintaining sufficient spectral resolution.

- define wave break (WB) and wave break rate (WBR)

A wave break is defined as a sustained drop in the instantaneous regional dominant frequency of more than three hertz below the upper quartile of the distribution, lasting longer than one hundred milliseconds. The wave break rate is the number of such events occurring per second.

- describe calculation of WBR

The wave break rate is calculated by counting the number of identified wave breaks within a defined time window and dividing by the duration of the window in seconds.

- describe use of WBR as a feature to characterize wavefront propagation

The wave break rate serves as a quantitative indicator of wavefront disorganization, with lower values indicating more organized, stable propagation consistent with the presence of a rotational driver, and higher values indicating chaotic, fragmented conduction.

### 1B. Example of RDF-Based Wave Break Identification

- describe example of RDF-based wave break identification

An example is provided from a patient with persistent atrial fibrillation, where electrograms recorded from the left atrial roof exhibited consistent wavefront activation during certain intervals, resulting in high-amplitude peaks in the averaged signal. During other intervals, delays in activation timing across electrodes caused the peaks to become dispersed, resulting in a high-frequency component in the averaged signal. This component was attenuated by the exponential filter, leading to a distinct drop in the instantaneous regional dominant frequency, which was identified as a wave break.

- illustrate example with plots of IEGMs and processing outputs

Plots of the raw electrograms, preprocessed pulses, averaged signal, filtered signal, and instantaneous regional dominant frequency are presented, demonstrating the transition from organized to fragmented activation and the corresponding drop in frequency. Three wave breaks are clearly identifiable in the time series, with a corresponding wave break rate of 0.1 per second.

### 1C. Minimum Required Segment Duration for Accurate RDF Estimation

- describe aim to find minimum segment duration for accurate RDF estimation

The aim was to determine the minimum duration of electrogram recording required to achieve a stable and reproducible estimate of the regional dominant frequency.

- describe results of analysis and conclusion

Analysis of two hundred one segments revealed that a recording duration of four seconds yielded a Pearson correlation coefficient of greater than 0.90 with the reference thirty-second measurement, indicating that four seconds is sufficient for accurate regional dominant frequency estimation.

### 1D. Minimum Required Segment Duration for Accurate WBR Estimation

- describe results of analysis and conclusion

Analysis of thirty-seven segments demonstrated that a recording duration of twenty-five seconds was required to achieve a stable estimate of the wave break rate, with a correlation coefficient exceeding 0.85 with the fifty-second reference. Shorter durations resulted in significant variability due to the infrequent nature of wave break events.

### 1E. Statistics

- describe statistical methods used

Statistical analysis was performed using the Anderson-Darling test to assess normality. Non-parametric comparisons between groups were conducted using the Mann-Whitney U test. Spearman’s rank correlation coefficient was used to evaluate the relationship between regional dominant frequency and wave break rate. Results are reported as mean ± standard deviation, with statistical significance defined as p < 0.05.

### 1F. Implementation

- describe implementation of embodiments in software

The method is implemented as a software application running on a general-purpose computer system, with executable instructions written in MATLAB and compiled into a standalone application for clinical use.

- describe data processing system

The data processing system receives digitized intracardiac electrograms via a high-speed interface, performs preprocessing, regional aggregation, filtering, and time-frequency analysis, and outputs color-coded maps to a display device.

- describe user interface

The user interface provides a graphical display of the atrial geometry with overlaid color maps of regional dominant frequency and wave break rate, along with controls for adjusting color scales, filtering parameters, and time windows.

- describe input device

Input devices include a keyboard, mouse, and foot pedal for operator interaction, as well as an interface for importing electrogram data from electrophysiology recording systems.

- describe central processing unit (CPU)

The central processing unit executes the computational algorithms, performs signal processing, and manages data flow between memory, storage, and display components.

- describe memory

Memory includes random-access memory for real-time signal buffering and storage memory for archiving electrogram data, processing parameters, and generated maps.

- describe display device

The display device is a high-resolution monitor capable of rendering three-dimensional atrial geometries with color-coded overlays, enabling real-time visualization of regional electrophysiological properties.

- describe interface device

The interface device connects the system to the electrophysiology recording system, enabling synchronized data acquisition and real-time map integration.

- describe network connections

Network connections allow for secure data transfer to hospital information systems and remote diagnostic support.

- describe database system

A database system stores patient records, electrogram recordings, processing parameters, and procedural outcomes for longitudinal analysis and quality assurance.

- describe computer executable programmed instructions

Computer executable programmed instructions are stored in non-transitory memory and configured to perform the steps of preprocessing, regional aggregation, filtering, time-frequency analysis, wave break detection, and graphical mapping.

- describe graphical user interface (GUI)

The graphical user interface provides an intuitive, menu-driven environment for clinicians to initiate analysis, adjust parameters, view results, and export maps for procedural documentation.

### 1G. Results

- exclude patients due to poor data quality

Five patients were excluded due to incomplete left atrial coverage, defined as endocardial surface coverage of less than sixty percent, which compromised the reliability of regional feature estimation.

- describe patient demographics

The remaining fifteen patients included five with paroxysmal atrial fibrillation and ten with persistent atrial fibrillation. The mean age was sixty-one point three years, with a mean left atrial diameter of forty-seven millimeters. Thirteen patients were male, and two were taking amiodarone.

- summarize procedural duration

The mean procedural duration was four hours and thirty-nine minutes, with no significant difference between paroxysmal and persistent patients.

- report recording locations and duration

An average of twenty-four point four recording locations were sampled per patient, with a mean recording duration of twenty-nine point nine seconds per location.

- administer Ibutilide to patients

Ibutilide was administered to eight of ten persistent patients prior to data collection to facilitate arrhythmia persistence, but not to paroxysmal patients.

- select segments for RDF and WBR estimation

Segments longer than twenty-five seconds were selected for analysis, with the first twenty-five seconds used for calculation of regional dominant frequency and wave break rate.

- calculate RDF and WBR for all patients

Regional dominant frequency ranged from two point eight six to seven point six six hertz, with a mean of five point five hertz. Wave break rate ranged from zero to zero point six three per second, with a mean of zero point one six per second.

- compare RDF and WBR for paroxysmal and persistent patients

Paroxysmal patients exhibited significantly higher regional dominant frequency and wave break rate compared to persistent patients, with p-values less than zero point zero zero one.

- show scatter plot and histograms of WBR and RDF

Scatter plots and histograms demonstrate the distribution of regional dominant frequency and wave break rate across all patients, revealing weak but statistically significant correlation between the two metrics.

- describe ablation results and termination sites

Ablation terminated atrial fibrillation in nine patients, with eight of these having at least one region of high regional dominant frequency and low wave break rate. These regions were ablated in all cases, and recurrence was absent in the six patients who achieved sinus rhythm.

### 1H. Discussion

- introduce novel metric for AF investigation

The invention introduces a novel metric for the investigation of atrial fibrillation based on the combined analysis of regional dominant frequency and wave break rate, which together provide a quantitative measure of wavefront organization and stability.

- describe regional dominant frequency

Regional dominant frequency reflects the average rate of electrical activation within a defined anatomical region, and is derived from the aggregated activity of multiple electrodes, thereby reducing noise and enhancing signal reliability.

- discuss advantages over traditional methods

Unlike traditional methods that rely on single-electrode measurements or static thresholds, the invention provides a dynamic, regionally resolved assessment of wavefront behavior that is less susceptible to artifacts and more reflective of underlying physiological mechanisms.

- explain wave break rate quantification

Wave break rate quantifies the frequency of wavefront fragmentation events, which are mechanistically linked to the presence of rotational drivers and conduction block, and serve as a marker of arrhythmia stability.

- associate termination sites with wavefront dynamics

Sites where atrial fibrillation terminated following ablation were strongly associated with regions exhibiting high regional dominant frequency and low wave break rate, suggesting that these regions represent stable sources of arrhythmia perpetuation.

- discuss spatial distribution of WBR and RDF

The spatial distribution of wave break rate and regional dominant frequency revealed a non-uniform pattern across the left atrium, with higher values observed in the posterior wall and pulmonary vein antra, consistent with known locations of arrhythmia drivers.

- propose WBR as a feature for ablation target

The wave break rate is proposed as a new feature for guiding ablation, as low values indicate organized, stable wavefront propagation consistent with a focal or rotational driver, making these regions ideal targets for ablation.

- discuss data collection and regional features

Data is collected sequentially using a high-density catheter, enabling high-resolution mapping of regional features without requiring simultaneous multi-site acquisition, which enhances practicality and reduces procedural complexity.

- describe critical sites and evaluation methods

Critical sites are identified based on the combination of high regional dominant frequency and low wave break rate, and are evaluated by correlation with procedural outcomes and follow-up recurrence rates.

- discuss limitations and future directions

Limitations include the reliance on left atrial recordings and the potential influence of antiarrhythmic drugs. Future directions include validation in larger cohorts, integration with real-time mapping systems, and extension to right atrial and ventricular applications.

## 2. Computer Modelling of Spiral Rotor and Associated Wave Break Analysis

- simulate cardiac cells using computer model

A computer model of cardiac tissue is implemented using a modified FitzHugh-Nagumo model to simulate the propagation of electrical wavefronts through a two-dimensional sheet of excitable tissue.

- generate spiral rotor using modified FitzHugh-Nagumo model

A spiral rotor is generated by introducing a localized perturbation that initiates a rotating wavefront, which persists due to the re-entry mechanism inherent in the model.

- calculate unipolar and bipolar electrograms

Unipolar and bipolar electrograms are calculated at multiple points on the tissue surface, mimicking the spatial configuration of a clinical mapping catheter.

- analyze iRDF-drop or WB using simulated electrograms

The simulated electrograms are processed using the same algorithm as applied to clinical data, and the resulting instantaneous regional dominant frequency and wave break rate are analyzed. The model demonstrates that wave breaks consistently occur at the core of the spiral rotor, and that the wave break rate is inversely correlated with the stability of the rotor, validating the clinical observations.

## 3. Clinical Example of an Identified Rotor During Wave Break

- illustrate rotational activity observed during wave break using propagation map

A clinical example is presented in which a propagation map derived from sequential mapping reveals a rotational pattern of activation centered on a region with high regional dominant frequency and low wave break rate. During the period of rotational activity, a wave break is observed at the core of the rotation, confirming the association between rotor dynamics and wave break events as predicted by the model.

## EQUIVALENTS

- describe scope of invention

The scope of the invention encompasses all methods, systems, and computer-readable media that implement the steps of preprocessing intracardiac electrograms, aggregating signals across multiple electrodes to derive regional features, performing time-frequency analysis to detect transient changes in dominant frequency, identifying wave breaks based on predefined thresholds, calculating wave break rate, and displaying the results as a color-coded map on an electroanatomic geometry. The invention includes all variations in signal processing parameters, filtering kernels, time windows, and graphical representations that achieve the same functional outcome of identifying regions of stable wavefront propagation during atrial fibrillation. The invention further includes all implementations in hardware, software, firmware, or any combination thereof, whether deployed in standalone systems, integrated into electrophysiology mapping platforms, or accessed remotely via networked interfaces.