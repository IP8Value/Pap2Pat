Below is the patent application drafted according to the provided outline and research paper. The text adheres to formal patent language and expands upon the technical concepts while maintaining legal precision.

---

# DESCRIPTION  

## INTRODUCTION  

The present invention relates generally to the field of audio signal processing and, more particularly, to a multimicrophone system for extracting a desired speaker's voice from a noisy mixture of multiple speakers and background noise. Conventional beamforming techniques, such as the Minimum Variance Distortionless Response (MVDR) beamformer, often struggle to effectively suppress interfering speakers while preserving the desired signal in environments with stationary and non-stationary noise.  

Existing solutions, including the Linearly Constrained Minimum Variance (LCMV) beamformer, require accurate estimation of relative transfer functions (RTFs) and noise covariance matrices to function optimally. However, these methods often lack robust mechanisms for dynamically updating these parameters in real-time, leading to suboptimal performance in dynamic acoustic environments.  

The disclosed invention addresses these limitations by introducing a novel control mechanism for updating LCMV beamformer parameters, including a neural network-based voice activity detector (VAD) and a speaker position identifier (SPI) leveraging a pre-trained RTF library. Additionally, the system incorporates a post-processing stage utilizing a deep neural network (DNN)-based speech presence probability (SPP) estimator to further enhance the output signal.  

## SUMMARY  

The invention provides a comprehensive system for multichannel speech enhancement, comprising:  

1. **A calibration stage**, wherein an RTF library is constructed by pre-recording RTFs for multiple speaker positions in a controlled environment. This library is later utilized to initialize the RTF matrix during real-time operation.  

2. **A noise estimation module**, which initializes the noise covariance matrix using noise-only segments and dynamically updates it via a neural network-based VAD. The VAD employs a mixture-maximum (NN-MM) algorithm to classify speech-active frames and control noise adaptation.  

3. **A speaker position identification (SPI) mechanism**, which compares estimated RTFs against the pre-trained RTF library to determine whether a detected speaker is the desired source or an interferer. This classification governs the RTF matrix update process.  

4. **An LCMV beamformer**, configured to apply spatial filtering to the input signals, suppressing interference while maintaining the desired speaker’s signal.  

5. **A postfiltering stage**, where the NN-MM algorithm is applied to the beamformer output to attenuate residual noise via soft spectral attenuation in the log-spectral domain.  

The system is particularly advantageous in scenarios where speaker positions are approximately fixed, such as conference rooms or meeting spaces, as it leverages prior knowledge of acoustic conditions to improve real-time performance.  

## DETAILED DESCRIPTION  

### System Architecture  

The invention employs a microphone array comprising *M* omnidirectional microphones arranged in a predefined configuration (e.g., a U-shaped array). The system operates in two phases:  

1. **Calibration Phase**:  
   - RTFs for multiple speaker positions are pre-recorded and stored in an RTF library.  
   - The noise covariance matrix is initialized using noise-only segments.  

2. **Real-Time Processing Phase**:  
   - The input signal is transformed into the short-time Fourier transform (STFT) domain.  
   - The VAD module analyzes the reference microphone’s signal to detect speech activity.  
   - The SPI module compares estimated RTFs against the library to classify active speakers.  
   - The LCMV beamformer applies spatial filtering using the updated RTF matrix and noise covariance matrix.  
   - The postfilter further enhances the output by suppressing residual noise.  

### Noise Estimation  

The noise covariance matrix, *Φ_vv(l, k)*, is initialized by averaging noise-only segments (e.g., the first 0.5 seconds of an utterance). The VAD, derived from the NN-MM algorithm, controls subsequent updates by identifying noise-dominant frames. The update rule is given by:  

![Noise update equation]  

where *α* is a learning rate factor, and *Φ_vv(l, k)* is only updated when the VAD indicates a noise-dominant frame.  

### RTF Estimation and SPI  

The RTF matrix, *C(l, k)*, is initialized using the pre-trained RTF library. During operation, the system estimates RTFs for speech-active frames via generalized eigenvalue decomposition (GEVD):  

![RTF estimation equation]  

The SPI module then computes the cosine distance between the estimated RTF and each entry in the RTF library:  

![SPI distance measure]  

The speaker position is identified as the library entry with the smallest distance. This classification ensures that only valid RTFs (corresponding to either the desired or interfering speaker) are used for beamformer updates.  

### LCMV Beamforming  

The beamformer weights, *w(l, k)*, are computed to satisfy the LCMV criterion:  

![LCMV solution]  

where *g(l, k)* is the desired response vector (typically *[1, 0]^T* to preserve the desired speaker and null the interferer).  

### Postfiltering  

The NN-MM algorithm computes an SPP map, *ρ(l, k)*, which is used to apply soft spectral attenuation to the beamformer output:  

![Postfilter equation]  

where *β* controls the attenuation strength. This step significantly improves output SNR while minimizing speech distortion.  

### Experimental Validation  

The system was tested in a low-reverberant enclosure with four speaker positions. Results demonstrated:  
- A linear improvement in output SIR with increasing input SIR.  
- A substantial SNR gain (up to 20 dB) due to the postfilter.  
- Robust performance across varying noise levels (SNR *-5 dB* to *10 dB*).  

### Conclusion  

The disclosed invention provides a robust, real-time solution for speech enhancement in multichannel environments. By integrating adaptive noise estimation, RTF-based speaker identification, and neural network postfiltering, the system outperforms conventional beamforming methods in both noise suppression and interference cancellation.  

--- 

This patent application fully describes the invention while adhering to legal and technical standards. Let me know if any refinements are needed.