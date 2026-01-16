Below is the drafted patent application following the provided outline and research paper content. The language has been formalized for patent purposes while maintaining technical accuracy.

---

# DESCRIPTION  

## STATEMENT OF GOVERNMENT RIGHTS  
The invention described herein was made with government support under [Grant/Award Number] awarded by [Agency Name]. The government has certain rights in the invention.  

## BACKGROUND  
Deep brain stimulation (DBS) is a therapeutic intervention involving the delivery of electrical impulses via implanted electrodes to modulate neural activity in targeted brain regions. While DBS has proven effective in treating neurological disorders such as Parkinson’s disease, epilepsy, and obsessive-compulsive disorder (OCD), the optimization of therapeutic efficacy remains an area of active research. A critical challenge in advancing DBS therapies is the accurate extraction of neural signals from recordings contaminated by high-amplitude stimulation artifacts. These artifacts obscure underlying biomarkers—neural signatures correlated with disease symptoms—which are essential for developing adaptive, closed-loop DBS systems capable of real-time parameter adjustment.  

Existing artifact removal methods, including discrete Fourier transform (DFT)-based techniques and harmonic regression, suffer from limitations in accuracy, particularly when confronted with unknown phase shifts, aliasing due to low sampling rates, and missing data segments. Conventional approaches fail to simultaneously address these challenges, necessitating a robust solution capable of operating under real-world constraints imposed by DBS devices, such as power efficiency and wireless data transmission limitations.  

## SUMMARY  
The present invention discloses a novel method for period-based artifact reconstruction and removal (PARRM) in DBS applications. The method comprises an iterative algorithm that jointly estimates the stimulation artifact’s fundamental frequency and phase shifts across discontinuous data segments while employing harmonic regression to model and subtract the artifact. Key innovations include:  

1. **Frequency and Phase Estimation**: A least-squares optimization framework that refines initial estimates of the artifact frequency and phase shifts, achieving sub-0.001% relative error even in the presence of aliasing and missing data.  
2. **Harmonic Artifact Modeling**: Parametric representation of the artifact using a finite Fourier series, enabling computationally efficient reconstruction.  
3. **Initialization via Energy Maximization**: A preprocessing step that aligns discontinuous segments by maximizing the signal’s energy at candidate frequencies, ensuring convergence of the iterative solver.  

The method is computationally lightweight, requiring only one tunable parameter (the number of harmonics, *K*), and is suitable for real-time implementation in embedded systems. Experimental validation demonstrates successful artifact removal in simulated and human local field potential (LFP) recordings, even under worst-case conditions of low sampling rates (e.g., 250 Hz) and unknown phase shifts.  

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS  

### Computer System  
The invention may be implemented on a computer system comprising a processor, memory, and input/output interfaces for receiving DBS waveform data. The system executes the following steps:  
1. **Data Segmentation**: Divides the input signal into contiguous segments, accounting for gaps caused by missing data or device modulation.  
2. **Initialization**: Applies Algorithm 2 to estimate initial frequency and phase shifts by maximizing the energy of the observed signal.  
3. **Artifact Removal**: Executes Algorithm 1 to iteratively refine frequency and phase estimates while fitting the artifact model via harmonic regression.  
4. **Signal Recovery**: Subtracts the reconstructed artifact from the observed signal to isolate the underlying neural activity.  

### Period-Based Artifact Reconstruction and Removal for Deep Brain Stimulation  
The PARRM method operates on the signal model:  
\[ S_i(t) = A\left(t + \frac{\delta_i^*}{\xi^*}\right) + B_i(t) + \eta_i(t), \]  
where *A* is the periodic artifact with unknown frequency *ξ** and phase shifts *δ_i**, *B_i* is the neural signal, and *η_i* is noise. The artifact is modeled as:  
\[ a(t | \xi, \delta, \alpha_k, \beta_k) = \alpha_0 + \sum_{k=1}^K \left[\alpha_k \cos(2\pi k (\xi t + \delta)) + \beta_k \sin(2\pi k (\xi t + \delta))\right]. \]  
The optimization minimizes the loss function:  
\[ \mathcal{L}(\xi, \delta_i, \theta) = \sum_{i=0}^n \sum_{t \in T_i} \left(S_i(t) - a(t | \xi, \delta_i, \theta)\right)^2, \]  
where *θ* = (*α_0*, *α_k*, *β_k*).  

### Comparison of PARRM to Conventional Filters  
Unlike DFT-based methods constrained to grid frequencies, PARRM achieves machine-precision frequency estimation (e.g., 3.77 × 10<sup>−14</sup>% error in Example 1). Harmonic regression alone fails with minor frequency errors (>10% RMSE at 0.001% error), whereas PARRM maintains accuracy even with aliasing (Example 3: 5.55% RMSE at 250 Hz sampling).  

### Periodic Estimation of Lost Packets From Deep Brain Stimulation Waveform Data  
For discontinuous data with *n* gaps, PARRM estimates phase shifts *δ_i* up to integer multiples via energy maximization (Algorithm 2) and refines them via least squares (Algorithm 1). In Example 4, this recovered gaps in human LFP data with a frequency estimate of 150.6093 Hz (device setting: 150.6 Hz).  

### Experimental Testing of the Period-Based Estimation of the Loss of Packets (PELP)  
Testing included:  
1. **Simulated Artifacts**: Example 1 achieved 1.79 × 10<sup>−10</sup% RMSE; Example 2 removed a chirp artifact with 5.55% RMSE.  
2. **Aliased Data**: Example 3 reconstructed artifacts at 99.3883 Hz (aliased from 150.6117 Hz) with 11.05% RMSE.  
3. **Human LFP**: Example 4 confirmed removal of 150.6 Hz artifacts despite unknown gap lengths.  

### Period-Based Estimation of Electrical Stimulation Artifacts in the Presence of Phase Shifts  
Phase shifts are resolved by maximizing the aligned energy:  
\[ E(\omega, \delta_i) = \left| \int S_i(t) e^{-2\pi i (\omega t + \delta_i)} dt \right|^2. \]  
Newton’s ascent with backtracking line search (Algorithm 2) ensures robust initialization for Algorithm 1’s iterative solver.  

---  

The application is drafted to comply with patent office requirements, emphasizing novelty, utility, and enablement. Let me know if further refinements are needed.