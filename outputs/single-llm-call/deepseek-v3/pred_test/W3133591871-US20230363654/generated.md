Here is the complete patent application following the provided outline:

# DESCRIPTION

## STATEMENT REGARDING RESEARCH & DEVELOPMENT  

The subject matter of this patent application was made with government support under Grant No. XXXXXXX awarded by the National Institutes of Health. The government has certain rights in the invention.

## TECHNICAL FIELD  

The present invention relates generally to the field of contactless physiological monitoring, and more particularly to systems and methods for detecting cardiac rhythm using smart speaker technology. Specifically, the invention enables the extraction of heart rate and R-R intervals through active sonar techniques implemented on commercially available smart speaker platforms.

## BACKGROUND  

Accurate assessment of heart rhythm through measurement of beat-to-beat intervals (R-R intervals) is essential for diagnosing cardiac arrhythmias and analyzing heart rate variability. Conventional electrocardiography (ECG) requires physical skin contact, creating limitations for infectious patients, those with skin allergies, and remote monitoring scenarios. While frequency domain analysis can estimate average heart rate for regular rhythms, it fails for irregular rhythms common in conditions like atrial fibrillation.  

Existing contactless monitoring approaches using Doppler radar or optical vibrocardiography require specialized hardware not widely available. Camera-based photoplethysmography raises privacy concerns. Passive acoustic methods lack sufficient temporal resolution for irregular rhythm detection. There exists an unmet need for a scalable, privacy-preserving contactless cardiac monitoring solution that can accurately detect both regular and irregular heart rhythms using widely available consumer hardware.

## DETAILED DESCRIPTION  

The present invention transforms commercial smart speakers into short-range active sonar systems capable of detecting subtle chest wall motions caused by cardiac activity. This approach leverages the microphone arrays and speakers already present in smart speaker devices to enable contactless cardiac rhythm monitoring without requiring additional specialized hardware.  

Smart speaker technology has advanced significantly, with devices now incorporating multiple microphones (6-7 microphone arrays in commercial products) and sophisticated acoustic processing capabilities. The invention utilizes these existing hardware components to implement an active sonar system that operates in the 18-22 kHz range - frequencies that are generally inaudible to humans while providing sufficient resolution for cardiac motion detection.  

The contactless nature of this technology provides numerous advantages over conventional ECG monitoring. It eliminates the need for skin contact, making it suitable for infectious patients where device cleaning is burdensome, patients with skin allergies, and home monitoring scenarios. The self-administered capability enables remote cardiac monitoring, connecting rural patients to specialists and facilitating large-scale screening for conditions like atrial fibrillation.  

The invention employs a frequency-modulated continuous wave (FMCW) active sonar approach where the smart speaker emits inaudible acoustic signals that reflect off the subject's chest. These reflections contain information about both respiratory and cardiac motions, with heartbeats causing subtle 0.3-0.8 mm displacements on the chest surface. The system's advanced signal processing algorithms separate these minute cardiac signals from the much larger respiratory motions and environmental noise.  

Key innovations of the invention include:  
1) An adaptive maximum signal-to-interference-and-noise ratio (SINR) beamforming algorithm that optimally combines signals across microphones and frequencies to enhance cardiac signals while suppressing respiratory interference  
2) A self-supervised learning approach for beamformer weight calculation that doesn't require pre-training or ground truth data  
3) A robust heartbeat segmentation algorithm that accounts for residual respiratory motion artifacts  
4) Precise R-R interval extraction capable of detecting both regular and irregular rhythms  

The system architecture comprises several key components:  
- A signal generator producing FMCW chirps between 18-22 kHz  
- A speaker array for transmitting interrogation signals  
- A microphone array for receiving reflected signals  
- Digital signal processing modules for echo suppression, beamforming, and segmentation  
- Algorithms for calculating heart rate, R-R intervals, and detecting arrhythmias  

FIG. 1 illustrates the system arrangement, showing the smart speaker positioned approximately 50 cm from the subject's chest at nipple level. The speaker emits inaudible FMCW signals that reflect off the chest wall, with the microphone array capturing these reflections. The system achieves sub-millimeter displacement resolution through advanced phase analysis of the received signals across multiple microphones.  

The signal processing pipeline begins with preprocessing to extract the acoustic channel's impulse response. Echo suppression eliminates reflections from distances greater than 1 meter to focus on cardiac motions. The adaptive beamformer then combines signals across microphones using complex weights calculated through gradient ascent optimization of an objective function that maximizes cardiac signal energy while minimizing respiratory interference.  

Following beamforming, the system segments the cardiac rhythm signal into individual heartbeats. This segmentation accounts for residual respiratory motion that causes rotation between in-phase and quadrature signal components. The algorithm identifies segment boundaries and applies appropriate rotational transformations to maintain signal coherence. From these segments, the system calculates precise timing for each heartbeat, enabling computation of heart rate and R-R intervals.  

The invention addresses several technical challenges in contactless cardiac monitoring:  
- The extremely small (sub-millimeter) displacements caused by heartbeats  
- Strong interference from respiratory motions that are 10x larger in amplitude  
- Limited bandwidth (4 kHz) and sampling rate (48 kHz) of commodity smart speakers  
- Non-ideal frequency response in the 18-22 kHz range  
- Need for high temporal resolution to detect irregular rhythms  

Privacy protections are inherent in the system design. The short operational range (≤1 m) requires deliberate user participation. The 18-22 kHz signals contain minimal audible information. Commercial smart speaker platforms don't provide third-party developers access to raw microphone data, preventing unauthorized audio capture.  

The system demonstrates robust performance across diverse test scenarios:  
- Accurate heart rate measurement with median absolute error of 1 BPM in healthy subjects  
- Precise R-R interval detection with median absolute error of 28 ms  
- Effective operation through normal clothing (though multiple layers may attenuate signals)  
- Tolerance to minor speaker misalignment (up to 20° off-axis)  
- Resilience to moderate background noise  

Clinical testing with cardiac patients showed comparable accuracy to healthy subjects, with median R-R interval errors of 30 ms. The system successfully detected atrial fibrillation and other arrhythmias by identifying characteristic irregular R-R interval patterns. Performance was maintained across different patient demographics, though extreme obesity (BMI >35) reduced signal quality due to adipose tissue damping.  

The invention's applications extend beyond clinical settings to include:  
- Home health monitoring for elderly or high-risk individuals  
- Fitness tracking during exercise  
- Stress and anxiety monitoring through heart rate variability analysis  
- Integration with smart home systems for ambient health sensing  
- Automotive health monitoring for driver safety systems  

Implementation examples demonstrate the system's practical utility. In a clinical study with 26 healthy participants and 24 cardiac patients, the technology achieved:  
- Intraclass correlation coefficients >0.9 for R-R interval measurements  
- 90th percentile errors <4 BPM for heart rate  
- Successful detection of atrial fibrillation and other arrhythmias  

The system's performance generalizes across different smart speaker hardware configurations and subject populations. While optimized for the 18-22 kHz range, the algorithms can adapt to other frequency bands supported by different speaker models. The beamforming and segmentation approaches remain effective across varying room acoustics and environmental conditions.  

### IMPLEMENTED EXAMPLES  

A clinical validation study was conducted with 26 healthy participants (median age 31 years) and 24 hospitalized cardiac patients (mean age 63-68 years). Healthy participants had no history of cardiac conditions, while cardiac patients included those with regular rhythms (sinus rhythm, paced rhythms) and irregular rhythms (atrial fibrillation). Patients with BMI >35 were excluded from the main study but evaluated separately.  

Ground truth data was collected using Polar H10 ECG sensors for most participants, with fingertip PPG sensors (CorSense) used when ECG was impractical. Testing occurred in controlled environments - a private room for healthy participants and hospital rooms for cardiac patients. The smart speaker system was positioned 40-60 cm from participants at chest level.  

Healthy participants completed seven 60-second measurement sessions under varying conditions:  
1-3) Different distances (40, 50, 60 cm)  
4) Speaker pointed 10 cm above chest  
5) 20° off-axis alignment  
6) With background jazz music (75 dB)  
7) After exercise to elevate heart rate  

Cardiac patients underwent five 60-second sessions with the speaker 50-60 cm away in quiet conditions. Data analysis compared smart speaker outputs to ground truth sensors for heart rate and R-R intervals. Statistical measures included intraclass correlation coefficients (ICC), concordance correlation coefficients (CCC), and median absolute errors.  

Results demonstrated:  
- Healthy participants:  
  - Heart rate ICC/CCC of 0.983  
  - R-R interval ICC/CCC of 0.929/0.927  
  - Median absolute error: 1 BPM (heart rate), 28 ms (R-R intervals)  
  - 90th percentile errors: <4 BPM, <75 ms  

- Cardiac patients:  
  - Heart rate median absolute error: 2 BPM  
  - R-R interval median absolute error: 30 ms  
  - ICC/CCC of 0.901/0.898 for R-R intervals  
  - Successful detection of atrial fibrillation patterns  

Performance remained stable across different testing conditions, with errors increasing slightly at greater distances or with background noise. The system successfully identified characteristic irregular R-R intervals in atrial fibrillation patients, demonstrating clinical utility for arrhythmia detection. Extreme obesity (BMI >35) reduced signal quality, with successful measurements obtained in only 1 of 5 such patients.  

These results validate the invention's ability to provide accurate, contactless cardiac rhythm monitoring using commercially available smart speaker hardware. The technology's performance meets clinical needs for both regular and irregular rhythm assessment, enabling new applications in telemedicine, home health monitoring, and preventive cardiology.