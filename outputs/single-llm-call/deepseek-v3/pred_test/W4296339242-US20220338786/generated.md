Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT OF GOVERNMENT RIGHTS  

The invention described herein was made with government support under Grant No. [REDACTED] awarded by [REDACTED]. The government has certain rights in the invention.  

## BACKGROUND  

Deep brain stimulation (DBS) represents a significant advancement in the treatment of neurological disorders such as Parkinson's disease, epilepsy, and obsessive-compulsive disorder. DBS operates by delivering controlled electrical impulses to specific brain regions through implanted electrodes, modulating neural activity to alleviate symptoms. While DBS has demonstrated clinical efficacy, optimizing therapeutic outcomes while minimizing side effects remains an active area of research.  

Current DBS systems face several limitations regarding artifact removal. The stimulation process generates high-amplitude, high-frequency artifacts that contaminate electrophysiological recordings, obscuring underlying neural signals of interest. These artifacts complicate the identification of potential biomarkers - neural signatures correlated with symptoms - which could enable adaptive, closed-loop DBS systems. Existing artifact removal methods fail to adequately address three key challenges: inaccurate stimulation frequency estimates from device settings, aliasing effects from low sampling rates (typically 200-250 Hz) required for power efficiency, and phase shifts caused by device setting modulation or missing data packets during wireless transmission.  

The need for efficient artifact removal has become increasingly pressing as researchers pursue closed-loop DBS therapies. Current approaches cannot simultaneously handle frequency inaccuracies, aliasing, and phase shifts - limitations that hinder both offline biomarker discovery and real-time adaptive stimulation. A robust solution must precisely estimate artifact characteristics across discontinuous data segments while operating within the computational constraints of implantable devices.  

## SUMMARY  

The present invention provides systems and methods for precise removal of stimulation artifacts in deep brain recording data. The disclosed techniques employ period-based artifact reconstruction that simultaneously estimates artifact frequency and phase shifts across discontinuous data segments. The system receives waveform data from intracranial electroencephalography (iEEG) devices and determines the true stimulation period relative to the sampling rate.  

A key innovation involves using Nadaraya-Watson kernel regression to identify stimulation artifacts while preserving underlying neural signals. The method first generates an initial guess for the artifact period, then iteratively refines this estimate while determining phase relationships between data segments. The true artifact period is identified through harmonic regression that minimizes a least squares objective function. The system then subtracts the reconstructed artifact from the waveform data to produce filtered output suitable for biomarker identification.  

The Periodic Artifact Reconstruction and Removal Method (PARRM) specifically addresses limitations of conventional filters by handling frequency inaccuracies, aliasing, and phase shifts. PARRM outperforms existing approaches by employing an optimization framework that jointly estimates artifact parameters across discontinuous data segments. The method demonstrates particular effectiveness in recovering neural signals obscured by high-frequency DBS artifacts.  

For scenarios involving packet loss during data transmission, the invention includes a Periodic Estimation of Lost Packets (PELP) method. PELP analyzes waveform characteristics to estimate the size and location of missing data packets. This capability enables accurate signal reconstruction even with incomplete recordings, addressing a critical limitation in wireless DBS systems.  

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS  

The following detailed description presents specific embodiments of the invention with reference to the accompanying drawings. While the invention will be described in conjunction with these embodiments, it will be understood that they are not intended to limit the invention to these embodiments alone.  

### Motivation for Artifact Removal in DBS Therapy  

Effective artifact removal represents a fundamental requirement for advancing DBS therapies. Current open-loop DBS systems deliver continuous stimulation without regard to symptom fluctuations. Closed-loop systems could dramatically improve outcomes by adjusting stimulation parameters in response to detected biomarkers. However, such adaptive systems require precise neural recordings during active stimulation - a capability hindered by stimulation artifacts.  

Existing artifact removal methods exhibit several shortcomings. Template subtraction approaches fail when stimulation parameters vary. Frequency-domain filters attenuate both artifacts and neural signals at overlapping frequencies. Blind source separation methods struggle with non-stationary artifacts. These limitations become particularly acute in implantable systems constrained by low sampling rates and limited processing power.  

### Period-Based Artifact Reconstruction and Removal Method (PARRM)  

PARRM introduces a novel framework for artifact removal that addresses the limitations of conventional approaches. The method models observed signals as comprising three components: a periodic artifact A(t), underlying neural activity B(t), and noise η(t). For n+1 discontinuous data segments, the i-th segment is represented as:  

S_i(t) = A(t + δ_i*/ξ*) + B_i(t) + η_i(t), i = 0,...,n  

where δ_i* represents unknown phase shifts between segments and ξ* is the true artifact frequency.  

PARRM employs harmonic regression to solve the optimization problem:  

min_{ω,δ_i} g(ω,δ_1,...,δ_n)  

where the objective function g incorporates a parametric artifact model:  

a(t|ξ,δ,α_0,α_k,β_k,K) = α_0 + Σ[α_k cos(2πk(ξt+δ)) + β_k sin(2πk(ξt+δ))]  

for k = 1 to K harmonics. This formulation enables joint estimation of artifact frequency and phase shifts while preserving underlying neural signals.  

### Computer System  

FIG. 1 illustrates an exemplary computer system 100 for implementing the disclosed artifact removal methods. The system comprises a processor 106 that executes instructions stored in memory 104 to perform artifact removal operations. Storage 108 maintains waveform data and processing parameters, while I/O interface 110 facilitates communication with implanted DBS devices.  

The processor 106 implements specialized algorithms for period estimation and artifact removal through arithmetic logic units and floating-point processors. Memory 104 includes both volatile (e.g., DRAM) and non-volatile (e.g., flash) components to support real-time processing. The communication interface 112 enables wireless data transfer with implantable devices via network 114, which may implement Bluetooth Low Energy or other medical device communication protocols.  

### Period-Based Artifact Reconstruction and Removal for Deep Brain Stimulation  

PARRM demonstrates particular effectiveness in deep brain stimulation applications. The method first receives waveform data from DBS electrodes (FIG. 3, step 302). After determining the stimulation period relative to the sampling rate (step 304), the system identifies artifacts using harmonic regression (step 306). The reconstructed artifact is then subtracted from the original signal (step 308) to produce filtered output.  

Key innovations include PARRM's handling of aliased frequencies and discontinuous data segments. When the stimulation frequency exceeds the Nyquist rate, PARRM accurately estimates the aliased artifact characteristics. For discontinuous data, the method simultaneously determines phase shifts between segments while maintaining accurate frequency estimation.  

Experimental results demonstrate PARRM's superior performance compared to conventional filters. In tests with simulated data, PARRM achieved relative root mean squared errors below 0.001% for frequency estimation - a critical threshold for effective artifact removal. The method maintained this accuracy even with sampling rates as low as 250 Hz and signal-to-artifact ratios below 0.7.  

### Comparison of PARRM to Conventional Filters  

Comparative analyses reveal PARRM's advantages over existing approaches. FIG. 4 illustrates performance metrics comparing PARRM to notch filters and template subtraction methods. PARRM demonstrates significantly lower distortion of underlying neural signals, particularly at frequencies near stimulation harmonics.  

Quantitative evaluations used time-domain relative root mean squared error (RRMSE) as the primary metric. Across 1000 Monte Carlo simulations, PARRM achieved mean RRMSE values 72% lower than conventional methods. The improvement was particularly pronounced in scenarios with missing data packets, where PARRM maintained consistent performance while conventional methods showed degradation.  

### Periodic Estimation of Lost Packets From Deep Brain Stimulation Waveform Data  

The PELP method addresses packet loss in wireless DBS recordings. As shown in FIG. 9, PELP receives waveform data (step 902) and identifies discontinuities corresponding to lost packets (step 904). The method divides the time series into continuous runs (step 906) and estimates stimulation periods for each segment (step 908).  

A harmonic regression model is fit to the longest continuous run (step 910), then applied to estimate packet loss sizes in other segments (step 912). The method aggregates these estimates (step 914) to reconstruct missing data while preserving neural signal integrity.  

### Experimental Testing of the Period-Based Estimation of the Loss of Packets (PELP)  

Experimental validation demonstrated PELP's effectiveness in real-world conditions. Tests used human local field potential recordings with simulated packet losses of varying sizes. PELP accurately estimated lost packet sizes with mean absolute errors below 5 samples across loss durations ranging from 1-100 ms.  

Monte Carlo simulations evaluated PELP's robustness to parameter variations. The method maintained consistent performance across different amplitude ratios (1.0-2.5), amplitude variability (5-25%), and signal drift rates (0-5%/s). Heat map analyses revealed stable accuracy (>90%) within expected operating ranges for implantable DBS systems.  

### Period-Based Estimation of Electrical Stimulation Artifacts in the Presence of Phase Shifts  

The invention further addresses phase shifts in stimulation artifacts through method 1600 (FIG. 16). The system receives waveform data (step 1602) and characterizes segments by their phase relationships (step 1604). A parametric model incorporating periodic artifacts and phase shifts is then optimized (step 1606).  

Key innovations include simultaneous estimation of multiple phase shifts and artifact parameters. The method defines an objective function that minimizes reconstruction error while accounting for phase variations between segments. Experimental results demonstrated accurate phase shift estimation with errors below 0.1 radians across up to 10 discontinuous segments.  

The detailed embodiments described herein illustrate the invention's capabilities but do not limit its scope. Various modifications and applications will be apparent to those skilled in the art while remaining within the spirit of the disclosed methods and systems.