Here is the patent application following the provided outline:

# DESCRIPTION  

## FIELD  

The present invention relates to magnetic resonance imaging (MRI) of the heart, specifically to methods and systems for improving cardiac gating in MRI through high-temporal-resolution monitoring of ventricular septal motion. The invention enables more accurate determination of cardiac rest periods, particularly the diastasis phase, which is critical for reducing motion artifacts in cardiac MRI applications such as coronary angiography.  

## BACKGROUND  

Cardiac MRI faces significant challenges due to the heart's continuous motion, particularly when imaging small structures like coronary arteries. Prospective cardiac gating, which synchronizes image acquisition with periods of minimal cardiac motion, remains the most effective approach for reducing motion artifacts. However, current gating techniques relying on electrocardiogram (ECG) signals and conventional cine MRI have inherent limitations in temporal resolution and adaptability to heart rate variations.  

Prospective cardiac gating typically uses ECG signals to identify the R-wave peak as a marker of ventricular systole onset, combined with pre-scan cine MRI to estimate the timing of diastasis relative to the ECG signal. The cine MRI approach suffers from limited temporal resolution (typically 30-60 ms per frame) due to the need for spatial encoding across multiple heartbeats. This resolution is insufficient to accurately capture the transitional motion of coronary arteries approaching diastasis, where velocities can reach 12 cm/s, potentially causing displacement of 1.5 mm in just 50 ms.  

Furthermore, conventional cine MRI is susceptible to temporal data mixing when heart rate varies during the scan, as the acquisition spans multiple heartbeats. These limitations in temporal resolution and heart rate adaptability lead to suboptimal gating window determination, resulting in motion blurring in the final images. There exists a need for a cardiac gating method that provides higher temporal resolution and better adapts to heart rate variations during scanning.  

## SUMMARY  

The present invention provides a method for determining cardiac diastasis timing using high-temporal-resolution monitoring of ventricular septal motion, referred to as the Septal Scout technique. The method involves acquiring one-dimensional projection images of the ventricular septum at very short repetition times (as low as 3 ms), enabling temporal resolution approximately an order of magnitude better than conventional cine MRI.  

The Septal Scout technique includes acquiring MRI projection images along the ventricular septum, processing these images to generate displacement and velocity graphs of septal motion, and identifying the diastasis period based on characteristic features in the velocity graph. The start and end times of diastasis are determined by analyzing inflection points between the E-wave (early diastolic filling) and A-wave (atrial contraction) components of the velocity graph.  

The invention further encompasses variations in defining the diastasis period, including alternative processing methods for the velocity graph and different approaches to handling heart rate variability. The method can be implemented as a standalone pre-scan for gating window calibration or integrated into real-time imaging sequences as a motion monitoring navigator.  

## DETAILED DESCRIPTION  

The following detailed description provides a comprehensive explanation of the invention, including definitions of terms, technical implementations, and various embodiments.  

The term "Septal Scout" refers to the high-temporal-resolution MRI technique for monitoring ventricular septal motion. "Scout Plane" denotes the imaging plane prescribed along the ventricular septum for Septal Scout acquisitions. The "4-chamber long-axis plane" is the standard cardiac MRI view used to prescribe the Scout Plane.  

In the medical imaging context, cardiac MRI requires synchronization with the cardiac cycle to minimize motion artifacts. Cardiac gating refers to the process of synchronizing image acquisition with specific phases of the cardiac cycle, typically targeting the diastasis period when cardiac motion is minimal.  

The RR interval represents the time between consecutive R-wave peaks in the ECG, corresponding to one complete cardiac cycle. Trigger delay is the time between the R-wave peak and the start of image acquisition. Gating parameters include the trigger delay and acquisition window duration, while gating error refers to deviations between the actual and optimal gating windows.  

The Septal Scout technique begins with prescribing the Scout Plane along the ventricular septum using the 4-chamber long-axis view as reference. The Septal Scout acquires one-dimensional projection images (line acquisitions) of the septum at high temporal resolution, typically with repetition times of 3-10 ms. These line acquisitions are compiled over time to create a spatiotemporal representation of septal motion.  

Displacement measurements are derived from the Septal Scout data by tracking intensity variations over time within a region of interest positioned at the basal septum. The displacement graph is processed to generate a velocity graph through temporal differentiation. The velocity graph exhibits characteristic E-wave and A-wave peaks corresponding to early diastolic filling and atrial contraction, respectively.  

Diastasis period determination involves identifying the plateau between E-wave and A-wave peaks in the velocity graph. The start and end times of diastasis are defined by inflection points where the velocity graph transitions between acceleration and deceleration phases. Alternative diastasis period determinations may use threshold-based approaches or pattern recognition algorithms.  

The invention includes embodiments using two-dimensional excitation schemes to improve septal signal isolation. A 2D excitation pulse can selectively excite tissue along the ventricular septum while suppressing signal from surrounding tissues. Phase images derived from Septal Scout acquisitions provide additional motion information through phase-contrast techniques.  

End-systole period determination is achieved by analyzing the systolic components of the velocity graph, enabling comprehensive cardiac phase monitoring. The invention supports free-breathing MRI through integration with respiratory navigators that monitor diaphragm position.  

Real-time Septal Scout acquisition enables continuous cardiac motion monitoring during imaging. The MRI-based cardiac gating system (MRI-CGS) incorporates calibration scans to establish baseline motion parameters and performs periodic calibration checks to adapt to physiological changes.  

Ventricular systole detection forms part of the comprehensive cardiac gating functionality, providing alternatives to ECG-based R-wave detection. The system includes heart rate variability (HRV) tracking to adjust gating parameters dynamically, with HRV checks ensuring reliable operation during variable heart rates.  

Alternative ventricular systole detection methods analyze systolic features in the Septal Scout data. The invention encompasses Septal Scout prescription techniques at different anatomical depths and phase-intensity analysis at the ascending aorta for complementary motion information.  

Comparative analysis demonstrates superior performance of the Septal Scout technique versus conventional MRI methods in imaging coronary artery stenosis. The invention distinguishes itself from MRI navigator techniques by focusing on septal motion as a global cardiac motion surrogate rather than local tissue tracking.  

The Septal Scout technique provides significant advantages over existing cardiac gating methods, including:  
1) Superior temporal resolution for accurate motion tracking  
2) Robustness to heart rate variability through per-heartbeat analysis  
3) Elimination of ECG signal dependency in certain embodiments  
4) Improved image quality in coronary artery visualization  

While particular embodiments of the invention have been described, it will be understood that various modifications may be made without departing from the scope of the invention as defined by the appended claims.