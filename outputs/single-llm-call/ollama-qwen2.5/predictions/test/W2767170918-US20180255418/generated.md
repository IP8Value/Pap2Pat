# Patent Application: System and Method for Speaker Position Identification and Noise Reduction

## Background

The invention relates to systems and methods for speaker position identification and noise reduction in low-reverberant environments. Specifically, the invention addresses the challenge of accurately identifying active speakers and reducing background noise to enhance speech clarity and quality.

### Field of Invention

The field of this invention is audio processing, particularly focusing on multi-speaker environments where accurate speaker identification and noise reduction are critical for effective communication systems.

### Description of Related Art

In multi-speaker environments, such as conference rooms or vehicles, accurately identifying the active speaker and reducing background noise is crucial. Traditional methods often struggle with these tasks, leading to degraded speech quality and user experience. There is a need for advanced techniques that can effectively handle these challenges in real-time.

## Summary of the Invention

The invention provides a system and method for speaker position identification and noise reduction. The system utilizes an SPP-based VAD (Speech Presence Probability-based Voice Activity Detector) to control the noise covariance matrix update and an SPI (Speaker Position Identification) method based on an RTF (Room Transfer Function) library to control the RTFs-matrix update. The updated LCMV-BF (Linearly Constrained Minimum Variance Beamformer) is then applied to enhance speech, followed by a postfilter using the NN-MM algorithm to further reduce residual noise.

### Objectives

- Accurately identify active speakers in multi-speaker environments.
- Reduce background noise and interference to improve speech clarity.
- Operate effectively in low-reverberant enclosures.

## Detailed Description of the Invention

### System Overview

The system comprises a microphone array, an SPP-based VAD, an SPI method, an LCMV-BF, and a postfilter. The microphone array captures audio signals from multiple positions. The SPP-based VAD determines noise-only frames, enabling recursive noise covariance matrix updates. The SPI method uses a pre-trained RTF library to identify active speakers. The LCMV-BF enhances speech by suppressing interference, and the postfilter further reduces residual noise.

### Microphone Array

A microphone array consisting of seven omnidirectional microphones arranged in a U-shape is used. This configuration ensures comprehensive coverage of the environment, capturing audio signals from multiple positions effectively.

### SPP-based VAD

The noisy signal from the reference microphone is input to the NN-MM algorithm, which calculates the SPP. The probabilities are aggregated across frequencies to yield a VAD decision per frame. If the current frame is noise-dominant, the noise estimation can be recursively updated. Otherwise, no noise adaptation is applied.

### Speaker Position Identification

An RTF library, consisting of specific RTFs for each position, is measured during a calibration stage. During the test phase, speech-active frames are identified using the VAD. An RTF is estimated and projected to each component of the RTF library by calculating the cosine distance. The position of the active speaker is determined based on the maximum cosine distance.

### LCMV-BF

The updated noise covariance matrix and RTFs-matrix are used to design the LCMV-BF. The BF enhances speech by suppressing interference, improving the SNR (Signal-to-Noise Ratio) and SIR (Signal-to-Interference Ratio).

### Postfilter

The output of the BF is a single-channel signal contaminated by residual noise. The NN-MM algorithm applies a soft spectral attenuation to further enhance the noisy signal. This postfilter significantly improves speech quality by reducing residual background noise.

## Experimental Results

The performance of the system was evaluated using real-life recordings in a low-reverberant enclosure. Experiments were conducted with four positions, and the desired speaker was located at position #1. The fifth microphone served as the reference microphone. The system demonstrated significant improvements in SNRout and SIRout across various input conditions.

### Experiment Setup

The algorithm performance was evaluated using a recording campaign carried out in a low-reverberant enclosure with four positions. A U-shaped microphone array consisting of seven omnidirectional microphones was used. Separate recordings of the desired speaker, interfering speakers, and background noise were synthesized to create real-life scenarios.

### Results

Results showed that the SIRout linearly grows with SIRin, and the NN-MM postfilter significantly improves SNRout. The system effectively suppressed interference by approximately 20dB and reduced residual background noise, enhancing overall speech quality.

## Conclusion

The invention provides a robust system for speaker position identification and noise reduction in low-reverberant environments. By combining an SPP-based VAD, SPI method, LCMV-BF, and postfilter, the system achieves significant improvements in speech clarity and quality, making it suitable for various applications such as conference rooms, vehicles, and other multi-speaker environments.

## Claims

1. A system for speaker position identification and noise reduction comprising:
   - a microphone array for capturing audio signals from multiple positions;
   - an SPP-based VAD for determining noise-only frames;
   - an SPI method based on an RTF library for identifying active speakers;
   - an LCMV-BF for enhancing speech by suppressing interference; and
   - a postfilter using the NN-MM algorithm for reducing residual noise.

2. The system of claim 1, wherein the microphone array comprises seven omnidirectional microphones arranged in a U-shape.

3. The system of claim 1, wherein the SPP-based VAD aggregates probabilities across frequencies to yield a VAD decision per frame.

4. The system of claim 1, wherein the RTF library is measured during a calibration stage and consists of specific RTFs for each position.

5. The system of claim 1, wherein the LCMV-BF uses the updated noise covariance matrix and RTFs-matrix to enhance speech by suppressing interference.

6. The system of claim 1, wherein the postfilter applies a soft spectral attenuation to further reduce residual background noise.

7. A method for speaker position identification and noise reduction comprising:
   - capturing audio signals from multiple positions using a microphone array;
   - determining noise-only frames using an SPP-based VAD;
   - identifying active speakers using an SPI method based on an RTF library;
   - enhancing speech by suppressing interference using an LCMV-BF; and
   - reducing residual noise using a postfilter.

8. The method of claim 7, wherein the microphone array comprises seven omnidirectional microphones arranged in a U-shape.

9. The method of claim 7, wherein the SPP-based VAD aggregates probabilities across frequencies to yield a VAD decision per frame.

10. The method of claim 7, wherein the RTF library is measured during a calibration stage and consists of specific RTFs for each position.

11. The method of claim 7, wherein the LCMV-BF uses the updated noise covariance matrix and RTFs-matrix to enhance speech by suppressing interference.

12. The method of claim 7, wherein the postfilter applies a soft spectral attenuation to further reduce residual background noise.

## Drawings

- Figure 1: System overview.
- Figure 2: Microphone array configuration.
- Figure 3: SPP-based VAD and SPI decisions.
- Figure 4: SNRout and SIRout results.

## Abstract

The invention provides a system and method for speaker position identification and noise reduction in low-reverberant environments. The system utilizes an SPP-based VAD, an SPI method based on an RTF library, an LCMV-BF, and a postfilter to enhance speech quality by accurately identifying active speakers and reducing background noise.

## Keywords

- Speaker Position Identification (SPI)
- Linearly Constrained Minimum Variance Beamformer (LCMV-BF)
- Speech Presence Probability-based Voice Activity Detector (SPP-based VAD)
- Room Transfer Function (RTF)
- Deep Neural Network Mixture of Gaussians (NN-MM)