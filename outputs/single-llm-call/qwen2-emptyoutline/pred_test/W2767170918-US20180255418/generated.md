# DESCRIPTION

## INTRODUCTION

The present invention relates to a system and method for enhancing speech signals in a noisy environment, particularly in scenarios involving multiple concurrent speakers. The invention addresses the challenge of extracting a desired speaker's voice from a mixture of voices and background noise using a multi-microphone array. Traditional beamforming techniques, such as the Minimum Variance Distortionless Response (MVDR) and Linearly Constrained Minimum Variance (LCMV) beamformers, have limitations in effectively mitigating interfering speakers and background noise. This invention introduces an advanced multichannel speech enhancement system that combines the LCMV beamformer with a post-processing stage to achieve superior speech clarity and noise reduction.

## SUMMARY

The invention provides a method and system for enhancing speech signals in environments with multiple speakers and background noise. The system includes a multi-microphone array, a Linearly Constrained Minimum Variance (LCMV) beamformer, a Voice Activity Detector (VAD), a Speaker Position Identifier (SPI), and a post-filter based on a Neural Network Mixture-Maximum (NN-MM) algorithm. The method involves the following steps:

1. **Noise Estimation**: Initialize the noise covariance matrix using the initial frames of the utterance, assumed to be noise-only.
2. **VAD Calculation**: Apply the NN-MM algorithm to the reference microphone to generate a Speech Presence Probability (SPP) map and calculate the VAD.
3. **RTF Estimation**: Use a pre-trained Room Transfer Function (RTF) library to estimate the RTFs of the desired and interfering speakers.
4. **Speaker Position Identification**: Classify speech-active frames to determine whether they belong to the desired or interfering speaker.
5. **LCMV Beamforming**: Apply the LCMV beamformer to the noisy input signals using the estimated RTFs and noise covariance matrix.
6. **Post-Filtering**: Enhance the output of the LCMV beamformer using the NN-MM algorithm to reduce residual noise.

The invention is particularly useful in scenarios where the positions of the speakers are approximately fixed, such as in conference rooms or meeting spaces. The system can be calibrated once and then used to enhance speech signals in real-time, providing clear and intelligible speech even in challenging acoustic environments.

## DETAILED DESCRIPTION

### 1. Introduction

Enhancing speech signals in noisy environments is a critical task in various applications, including teleconferencing, hearing aids, and speech recognition systems. The presence of multiple speakers and background noise can severely degrade the quality and intelligibility of the desired speech. Traditional beamforming techniques, such as the Minimum Variance Distortionless Response (MVDR) and Linearly Constrained Minimum Variance (LCMV) beamformers, have been widely used for speech enhancement. However, these methods often struggle to effectively mitigate interfering speakers and background noise, especially in dynamic and reverberant environments.

The present invention addresses these challenges by introducing an advanced multichannel speech enhancement system. The system combines the LCMV beamformer with a post-processing stage to achieve superior speech clarity and noise reduction. The invention is particularly suited for scenarios where the positions of the speakers are approximately fixed, such as in conference rooms or meeting spaces.

### 2. System Overview

The invention comprises a multi-microphone array, a Linearly Constrained Minimum Variance (LCMV) beamformer, a Voice Activity Detector (VAD), a Speaker Position Identifier (SPI), and a post-filter based on a Neural Network Mixture-Maximum (NN-MM) algorithm. The system operates in the following stages:

#### 2.1. Noise Estimation

The first step in the process is to estimate the noise covariance matrix. This is done by averaging the first frames of the utterance, which are assumed to be noise-only. The noise covariance matrix, denoted as \(\Phi_{vv}\), is initialized using the following equation:

\[
\Phi_{vv}(l, k) = \frac{1}{L} \sum_{l=l_{\text{start}}^{v}}^{l_{\text{stop}}^{v}} z_m(n) z_m^H(n)
\]

where \(L\) is the number of noise-only frames, \(z_m(n)\) is the signal received by the \(m\)-th microphone, and \(z_m^H(n)\) is the conjugate transpose of \(z_m(n)\).

#### 2.2. VAD Calculation

The next step is to calculate the Voice Activity Detector (VAD) using the Neural Network Mixture-Maximum (NN-MM) algorithm. The NN-MM algorithm generates a Speech Presence Probability (SPP) map, which is used to identify speech-active frames. The VAD is calculated based on the SPP map and is used to control the noise estimation update. The VAD decision for each frame is given by:

\[
\text{VAD}(l) = \begin{cases} 
1 & \text{if } \sum_{k=1}^{N_{\text{DFT}}} \rho(l, k) > T_r \\
0 & \text{otherwise}
\end{cases}
\]

where \(\rho(l, k)\) is the SPP map, \(N_{\text{DFT}}\) is the STFT frame length, and \(T_r\) is the threshold value. In our implementation, \(T_r\) is set to \(N_{\text{DFT}} / 4\).

#### 2.3. RTF Estimation

The Room Transfer Function (RTF) matrix is a crucial component in the LCMV beamformer design. The RTF matrix, denoted as \(C(l, k)\), is estimated using a pre-trained RTF library. The RTF library consists of specific RTFs for each possible speaker position. In the calibration stage, the RTF library is constructed using different speakers and utterances than those used in the test stage. The RTF matrix is initialized with the components from the library.

During the test phase, the RTF matrix is updated using the following steps:

1. **Identify Speech-Active Frames**: Use the VAD to identify frames where speech is active.
2. **Estimate RTF**: For each speech-active frame, estimate the RTF using the generalized eigenvalue decomposition (GEVD) of the correlation matrix \(\Phi_{zz}(l, k)\) and the noise covariance matrix \(\Phi_{vv}(l, k)\):

\[
\Phi_{zz}(l, k) f(k) = \lambda(k) \Phi_{vv}(l, k) f(k)
\]

where \(\lambda(k)\) is the generalized eigenvalue and \(f(k)\) is the corresponding generalized eigenvector. The RTF is then normalized to obtain a proper estimate:

\[
\hat{c}(l, k) = \frac{f(k)}{\| f(k) \|}
\]

#### 2.4. Speaker Position Identification

The Speaker Position Identifier (SPI) is used to classify speech-active frames as belonging to the desired or interfering speaker. The SPI is based on the cosine distance between the estimated RTF and the components of the RTF library. The cosine distance is calculated as follows:

\[
D_i(l, k) = \cos^{-1} \left( \frac{\hat{c}(l, k) \cdot c_s(k)}{\| \hat{c}(l, k) \| \| c_s(k) \|} \right)
\]

where \(c_s(k)\) is the \(s\)-th component of the RTF library. The position of the active speaker is determined by:

\[
I(l) = \arg \min_i D_i(l)
\]

If the minimum distance is below a certain threshold, the frame is classified as belonging to the desired speaker; otherwise, it is classified as belonging to the interfering speaker.

#### 2.5. LCMV Beamforming

The LCMV beamformer is applied to the noisy input signals using the estimated RTF matrix and noise covariance matrix. The LCMV beamformer is designed to minimize the noise power at the output while maintaining the desired signal. The beamformer weights, denoted as \(w(l, k)\), are calculated using the following equation:

\[
w(l, k) = \Phi_{vv}^{-1}(l, k) C(l, k) \left( C^H(l, k) \Phi_{vv}^{-1}(l, k) C(l, k) \right)^{-1} g(l, k)
\]

where \(g(l, k)\) is the desired response vector, set to \([1, 0]^T\).

#### 2.6. Post-Filtering

The output of the LCMV beamformer, denoted as \(\hat{s}_d(l, k)\), is a single-channel signal contaminated by residual noise. To further enhance the speech signal, a post-filter based on the NN-MM algorithm is applied. The NN-MM algorithm constructs a time-frequency SPP map, \(\rho(l, k)\), and applies a soft spectral attenuation to the BF output:

\[
\hat{s}_d'(l, k) = \hat{s}_d(l, k) \exp \left( -\beta \rho(l, k) \right)
\]

where \(\beta\) is the soft attenuation level. The post-filter significantly reduces the residual noise, resulting in a clearer and more intelligible speech signal.

### 3. Control Mechanisms

The invention includes several control mechanisms to ensure the robustness and effectiveness of the speech enhancement process.

#### 3.1. SPP-Based VAD

The SPP-based VAD is a crucial component in the noise estimation and RTF estimation processes. The VAD is calculated using the SPP map generated by the NN-MM algorithm. The VAD decision for each frame is given by:

\[
\text{VAD}(l) = \begin{cases} 
1 & \text{if } \sum_{k=1}^{N_{\text{DFT}}} \rho(l, k) > T_r \\
0 & \text{otherwise}
\end{cases}
\]

where \(\rho(l, k)\) is the SPP map, \(N_{\text{DFT}}\) is the STFT frame length, and \(T_r\) is the threshold value. The VAD accurately tracks the speech-active frames and can be easily verified that it is on when both the desired and interfering speakers are active.

#### 3.2. Speaker Position Identification Based on Pre-Trained RTFs

The Speaker Position Identifier (SPI) is based on a pre-trained RTF library. The RTF library consists of specific RTFs for each possible speaker position. During the test phase, the RTF matrix is updated using the cosine distance between the estimated RTF and the components of the RTF library. The cosine distance is calculated as follows:

\[
D_i(l, k) = \cos^{-1} \left( \frac{\hat{c}(l, k) \cdot c_s(k)}{\| \hat{c}(l, k) \| \| c_s(k) \|} \right)
\]

where \(c_s(k)\) is the \(s\)-th component of the RTF library. The position of the active speaker is determined by:

\[
I(l) = \arg \min_i D_i(l)
\]

If the minimum distance is below a certain threshold, the frame is classified as belonging to the desired speaker; otherwise, it is classified as belonging to the interfering speaker.

### 4. Experimental Results

The performance of the proposed system was evaluated using real-life recordings in a low-reverberant enclosure. The experimental setup included a microphone array consisting of seven omnidirectional microphones arranged in a U-shape. The desired speaker was located at position #1, and the interfering speakers were located at positions #2, #3, and #4. The background noise was recorded separately and added to the synthesized scenarios.

The algorithm's performance was evaluated by measuring the Signal-to-Noise Ratio (SNR) and Signal-to-Interference Ratio (SIR) at the output of the algorithm as a function of the input SNR and SIR. The results showed that the proposed system significantly improved the SNR and SIR, even in challenging acoustic conditions. The post-filter based on the NN-MM algorithm further enhanced the speech quality by reducing residual noise.

### 5. Conclusion

The present invention provides a robust and effective system for enhancing speech signals in environments with multiple speakers and background noise. The system combines the LCMV beamformer with a post-processing stage to achieve superior speech clarity and noise reduction. The invention is particularly useful in scenarios where the positions of the speakers are approximately fixed, such as in conference rooms or meeting spaces. The system can be calibrated once and then used to enhance speech signals in real-time, providing clear and intelligible speech even in challenging acoustic environments.