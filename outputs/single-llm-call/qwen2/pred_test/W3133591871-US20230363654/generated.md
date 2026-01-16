# DESCRIPTION

## STATEMENT REGARDING RESEARCH & DEVELOPMENT

This invention was developed through extensive research and development efforts aimed at creating a noncontact, smart speaker-based system for monitoring cardiac rhythm. The system utilizes active sonar technology to detect individual heartbeats and extract heart rate and R–R intervals. The development process involved interdisciplinary collaboration between experts in signal processing, acoustics, and cardiology to ensure the system's accuracy and reliability in both healthy individuals and patients with cardiac conditions.

## TECHNICAL FIELD

The present invention relates to a noncontact system for monitoring cardiac rhythm using smart speakers. More specifically, the invention pertains to a method and apparatus for extracting heart rate and R–R intervals by transforming a smart speaker into a short-range active sonar system. The system is designed to detect individual heartbeats in both regular and irregular rhythms, making it particularly useful for diagnosing cardiac arrhythmias and studying heart rate variability (HRV).

## BACKGROUND

Clinical heart rhythm assessment is critical for diagnosing cardiac arrhythmias and studying HRV. Traditional methods rely on electrocardiography (ECG), which requires physical contact with the skin. While ECG is effective, it is not suitable for all scenarios, especially in cases where contact-based devices are impractical or undesirable. For example, contactless monitoring is advantageous for infectious and contagious patients, home isolation and quarantine settings, and patients with skin allergies. Additionally, contactless monitoring can facilitate telemedicine, enabling remote patient monitoring and reducing the need for in-person visits.

Recent advancements in smart speaker technology have opened new possibilities for noncontact health monitoring. Smart speakers, equipped with multiple microphones and speakers, can emit inaudible acoustic signals and analyze their reflections to detect subtle movements. This capability has been explored for monitoring breathing signals, but extending it to cardiac rhythm monitoring presents significant challenges. Heartbeats result in minute chest wall movements, which are orders of magnitude smaller than the wavelengths of the emitted acoustic signals. Moreover, the presence of larger breathing motions and ambient noise complicates the extraction of heart signals.

Existing noncontact monitoring technologies, such as Doppler radar and optical vibrocardiography, have limitations in terms of hardware requirements and privacy concerns. Doppler radar systems require custom hardware and have limited availability, while optical methods involve camera usage, raising privacy issues. In contrast, active sonar using smart speakers leverages existing hardware and operates using inaudible acoustic signals, making it a promising solution for widespread adoption.

## DETAILED DESCRIPTION

### IMPLEMENTED EXAMPLES

The present invention provides a noncontact system for monitoring cardiac rhythm using smart speakers. The system transforms a smart speaker into a short-range active sonar device capable of detecting individual heartbeats and extracting heart rate and R–R intervals. The key components and processes of the system are described below.

#### System Overview

The system consists of a smart speaker equipped with multiple microphones and a speaker. The smart speaker emits inaudible acoustic signals in the 18–22 kHz range and receives reflections from the human body. The received signals are processed to extract the subtle chest wall movements caused by heartbeats, separating them from larger breathing motions and ambient noise.

#### Signal Generation and Reception

The smart speaker generates frequency modulated continuous wave (FMCW) signals with a linear frequency sweep from 18 to 22 kHz. These signals are emitted in a loop, and the reflections are captured by the microphone array. Each microphone receives a signal that is a superposition of reflections from various body parts and environmental objects.

#### Preprocessing and Echo Suppression

The received signals are preprocessed to filter out audible frequencies and remove background noise. The impulse response of the acoustic channel is then extracted using discrete Fourier transforms (DFTs). To eliminate reflections from distant locations, echo suppression is applied by zeroing out the impulse responses beyond a certain distance threshold (e.g., 1 meter). This step helps to focus on the subtle chest wall movements caused by heartbeats.

#### Adaptive Maximum-SINR Beamformer

An adaptive maximum signal-to-interference and noise ratio (SINR) beamformer is employed to separate heart signals from breathing motions and noise. The beamformer combines signals from different microphones and frequencies to maximize the heart signal while minimizing interference. The optimization process involves computing complex weights that align heart signals across microphones and frequencies. Regularization techniques are used to prevent the beamformer from amplifying high-frequency, impulse-like signals caused by abrupt breaths or environmental interference.

#### Heartbeat Segmentation

After beamforming, the resulting heart rhythm signal is segmented into individual heartbeats. The challenge lies in dealing with residual interference from breathing motions, which can modulate the heart signal. The segmentation algorithm identifies segmenting points and the corresponding rotations of each segment to account for the modulation. By comparing adjacent segments and adjusting for temporal scaling and rotation, the algorithm accurately identifies heartbeats and computes the heart rate and R–R intervals.

#### Testing and Validation

The system was tested with both healthy participants and hospitalized cardiac patients to validate its performance. For healthy participants, the system was placed at various distances and angles from the chest, and measurements were taken under different conditions, including background music and post-exercise. The results showed high accuracy in heart rate and R–R interval measurements, with median absolute errors of 1 BPM and 28 ms, respectively.

For cardiac patients, the system was tested in a hospital setting, where patients were categorized into regular and irregular rhythm groups. The system demonstrated robust performance in both groups, with median absolute errors of 2 BPM and 30 ms for heart rate and R–R intervals, respectively. The system was particularly effective in detecting irregular heartbeats, such as those associated with atrial fibrillation, with a mean absolute R–R interval error of 35 ms.

#### Limitations and Future Work

While the system performs well in most scenarios, it has limitations. The accuracy can be affected by factors such as extreme obesity, which dampens the chest wall movements, and the presence of multiple layers of clothing. Future improvements could include enhancing the hardware to support higher sampling rates and better frequency response, as well as developing algorithms for continuous monitoring of multiple participants.

### Conclusion

The present invention provides a noncontact, smart speaker-based system for monitoring cardiac rhythm. By leveraging active sonar technology, the system can detect individual heartbeats and extract heart rate and R–R intervals with high accuracy. The system's ability to operate using existing smart speaker hardware and inaudible acoustic signals makes it a promising tool for widespread adoption in various healthcare settings, including telemedicine and home monitoring. Further research and development will continue to enhance the system's performance and expand its applications.