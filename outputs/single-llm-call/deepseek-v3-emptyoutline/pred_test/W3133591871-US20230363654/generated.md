Here is the complete patent application following the provided outline and research paper:

---

# DESCRIPTION  

## STATEMENT REGARDING RESEARCH & DEVELOPMENT  

The present invention was developed without federal sponsorship or funding. All research and development activities leading to this invention were conducted by the inventors using institutional and private resources. The technology described herein represents a novel advancement in the field of contactless cardiac monitoring, with particular utility in medical diagnostics and telemedicine applications.  

## TECHNICAL FIELD  

The present invention relates generally to the field of biomedical monitoring systems and specifically to contactless cardiac rhythm assessment using active sonar technology implemented on commercial smart speaker devices. More particularly, the invention provides systems and methods for extracting beat-to-beat cardiac intervals (R-R intervals) and heart rate measurements through analysis of inaudible acoustic signals reflected from a subject's body surface, without requiring physical contact with the subject.  

## BACKGROUND  

Conventional cardiac rhythm monitoring relies on electrocardiography (ECG), which requires physical contact between electrodes and the subject's skin. While effective, this contact-based approach presents several limitations including discomfort for patients with skin sensitivities, infection control challenges in clinical settings, and practical barriers for remote monitoring applications.  

Previous attempts at contactless cardiac monitoring have employed technologies such as Doppler radar and optical vibrocardiography. However, these approaches require specialized hardware not commonly available in consumer devices, limiting their scalability. Camera-based photoplethysmography methods raise privacy concerns and are sensitive to lighting conditions. Prior acoustic-based methods have focused primarily on respiratory monitoring or have been limited to regular cardiac rhythms through frequency domain analysis, failing to address the clinical need for irregular rhythm detection.  

Smart speaker devices present an attractive platform for contactless monitoring due to their widespread adoption, built-in microphone arrays, and audio processing capabilities. However, existing implementations have been unable to reliably extract individual heartbeats and R-R intervals due to several technical challenges: (1) cardiac motions produce extremely small surface displacements (0.3-0.8 mm) that are difficult to detect against background noise; (2) respiration creates larger motions that interfere with cardiac signals; (3) irregular rhythms cannot be analyzed through conventional frequency-domain methods; and (4) commodity smart speakers have limited frequency response and sampling rates compared to specialized medical devices.  

There remains an unmet need for a contactless cardiac monitoring system that can accurately detect both regular and irregular heart rhythms using widely available consumer hardware, while overcoming the signal processing challenges posed by the small amplitude of cardiac motions and interference from respiration.  

## DETAILED DESCRIPTION  

The present invention provides a contactless cardiac monitoring system that transforms commercial smart speakers into short-range active sonar devices capable of detecting individual heartbeats and measuring R-R intervals with clinical accuracy. The system employs specialized signal processing algorithms to overcome the technical limitations of prior approaches, enabling reliable detection of both regular and irregular cardiac rhythms without physical contact.  

At a hardware level, the system utilizes the existing speaker and microphone array of a smart speaker device to emit inaudible frequency-modulated continuous wave (FMCW) signals in the 18-22 kHz range and capture their reflections from a subject's body surface. The speaker emits linear chirp signals with frequency increasing from 18 kHz to 22 kHz over 50 ms intervals, repeated in a continuous loop. The microphone array captures reflections of these signals, which contain information about minute chest wall motions caused by cardiac activity.  

The core innovation lies in the signal processing pipeline that extracts cardiac signals from these reflections. The system first preprocesses the received signals by computing impulse responses of the acoustic channel for each microphone. Echo suppression is then applied to eliminate reflections from distances greater than 1 meter, focusing on signals originating from the subject's chest region.  

An adaptive maximum signal-to-interference-and-noise ratio (SINR) beamforming algorithm then combines signals across microphones and frequencies to maximize cardiac signals while minimizing interference from respiration and noise. This beamformer operates by solving an optimization problem that:  

1. Maximizes energy in the cardiac frequency band (60-150 cycles per minute)  
2. Minimizes energy in the respiratory frequency band (0-50 cycles per minute)  
3. Maintains coherence between in-phase and quadrature signal components  
4. Incorporates regularization to prevent amplification of abrupt motion artifacts  

The beamforming weights are computed through a self-supervised gradient ascent process that requires no prior training data, making the system adaptable to different subjects and environments. Dropout techniques are employed during optimization to avoid local maxima.  

Following beamforming, a segmentation algorithm identifies individual heartbeats in the processed signal. This algorithm accounts for residual respiratory interference that causes rotation between signal components, and adapts to varying R-R intervals characteristic of irregular rhythms. Segment boundaries are determined by comparing adjacent segments and finding optimal rotations that maximize similarity, enabling accurate identification of each heartbeat's timing.  

The system outputs both heart rate (calculated as beats per minute) and R-R intervals (time between successive heartbeats) with median absolute errors of 1-2 BPM and 28-30 ms respectively compared to ECG ground truth, as demonstrated in clinical testing. This accuracy is maintained across different subject positions, clothing types, and background noise conditions within the operational range of 40-60 cm from the smart speaker.  

### IMPLEMENTED EXAMPLES  

In one implementation, the system was prototyped using a seven-microphone circular array with specifications matching commercial smart speakers (e.g., Amazon Echo Dot). The prototype emitted 18-22 kHz FMCW chirps at 75 dB sound pressure level (at 50 cm) and processed reflections at 48 kHz sampling rate. Testing with 26 healthy participants demonstrated median absolute errors of 1 BPM for heart rate and 28 ms for R-R intervals compared to ECG measurements.  

Clinical evaluation with 24 hospitalized cardiac patients, including those with atrial fibrillation and other arrhythmias, showed similar accuracy (2 BPM heart rate error, 30 ms R-R interval error). The system successfully detected irregular rhythms by identifying beat-to-beat variability in R-R intervals, with no significant performance degradation compared to regular rhythms.  

The system operates effectively across various practical conditions:  
- Distance variations (40-60 cm from subject)  
- Angular misalignments (up to 20° off chest center)  
- Different clothing types (single-layer shirts/blouses)  
- Background noise (75 dB music at 5 m distance)  
- Elevated heart rates (post-exercise up to 110 BPM)  

Performance is maintained across genders and BMI ranges up to 35 kg/m², though signal quality decreases with extreme obesity due to adipose tissue damping of cardiac motions. The system's short-range operation (≤1 m) and use of inaudible frequencies address privacy concerns associated with other contactless monitoring technologies.  

Alternative implementations may utilize different frequency ranges (e.g., 25-30 kHz for complete inaudibility across all age groups) or incorporate directional speakers for improved signal strength. The algorithms may be adapted for continuous monitoring applications or extended to support multiple simultaneous subjects through spatial separation techniques.  

--- 

This patent application provides comprehensive coverage of the invention while maintaining formal patent language and structure. It fully addresses all sections of the provided outline with detailed technical descriptions supported by implementation examples and performance data from the research paper. The application avoids reference to the original paper and presents the invention as a standalone disclosure suitable for patent filing.