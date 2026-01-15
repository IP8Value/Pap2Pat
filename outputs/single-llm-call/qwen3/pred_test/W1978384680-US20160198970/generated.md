# DESCRIPTION

## FIELD

- relate to MRI of the heart

The present invention relates to magnetic resonance imaging (MRI) of the heart, specifically to methods and systems for improving the accuracy of cardiac gating during coronary magnetic resonance angiography (MRA) and other cardiac imaging procedures. Cardiac motion remains a primary source of image degradation in high-resolution MRI of the coronary arteries, particularly when imaging is performed over multiple cardiac cycles. Conventional approaches to mitigate this motion rely on electrocardiogram (ECG) triggering combined with retrospective or prospective gating based on cine MRI acquisitions of the heart’s four-chamber view. However, these methods are limited by insufficient temporal resolution, susceptibility to heart rate variability, and the inability to capture rapid transitions between systolic and diastolic phases with sufficient fidelity. The invention introduces a novel technique for determining the precise timing of diastasis—the period of minimal myocardial motion—by directly monitoring the one-dimensional motion of the ventricular septum using a high-temporal-resolution projection sequence known as the Septal Scout. This method enables more accurate synchronization of image acquisition with cardiac quiescence, resulting in significantly improved spatial clarity of coronary vasculature without increasing scan time or requiring contrast agents.

## BACKGROUND

- introduce cardiac MRI limitations
- explain prospective cardiac gating
- describe calibration of gating parameters
- motivate septal motion-based cardiac gating

Magnetic resonance coronary angiography (MRCA) offers a non-invasive, radiation-free alternative to conventional x-ray angiography for the evaluation of coronary artery disease. Despite its advantages, MRCA is critically constrained by the necessity to acquire high-resolution three-dimensional data over multiple cardiac cycles, during which cardiac motion introduces blurring and misregistration artifacts. To address this, prospective cardiac gating is employed to restrict data acquisition to a narrow temporal window within each cardiac cycle, ideally coinciding with diastasis, when coronary artery motion is minimized. This window is typically determined by correlating the R-wave of the ECG with the timing of diastasis derived from a prior cine MRI acquisition of the heart in the four-chamber long-axis orientation. However, the temporal resolution of conventional cine sequences is inherently limited by the number of k-space lines acquired per heartbeat and the repetition time of the pulse sequence, often resulting in frame rates insufficient to resolve the rapid deceleration and acceleration phases that define the boundaries of diastasis. Furthermore, heart rate variability during a breath-hold scan can cause temporal misalignment of data bins, leading to inconsistent gating and degraded image quality. Existing methods for calibrating gating parameters rely on post-processing of cine images using frame-to-frame correlation analysis to identify inflection points between the E and A waves of diastolic filling, but these approaches are vulnerable to motion artifacts, partial volume effects, and temporal averaging that obscure the true onset and termination of diastasis. The present invention addresses these limitations by introducing a method that directly measures the longitudinal motion of the ventricular septum with a temporal resolution of 10 milliseconds or less, thereby capturing the precise kinematics of cardiac quiescence with fidelity unmatched by conventional cine techniques. The ventricular septum has been shown to serve as a reliable surrogate for global ventricular motion, and its motion profile during diastasis closely mirrors that of the coronary arteries, making it an ideal target for real-time gating calibration.

## SUMMARY

- introduce method for determining diastasis timing
- describe MRI image acquisition
- explain image processing for velocity graph
- determine start and end times of diastasis
- outline variations of diastasis period definition

The invention provides a method for determining the timing of diastasis in the cardiac cycle by acquiring a one-dimensional projection of the ventricular septum using a modified steady-state free precession (SSFP) sequence with phase encoding disabled, thereby enabling a repetition time as short as 3 to 10 milliseconds. This high-temporal-resolution acquisition generates a time-series intensity profile representing the motion of the septal tissue along the long axis of the heart. The intensity profile is processed using gradient optical flow analysis to compute a pseudo-displacement function over time, from which a pseudo-velocity function is derived by taking the temporal derivative. The velocity graph exhibits characteristic E and A waves corresponding to early and late diastolic filling, respectively, with a plateau region between them representing diastasis. The start and end times of diastasis are identified as the inflection points—the points of zero second derivative—between the E and A wave peaks, ensuring robust detection even in the presence of minor motion noise. Alternative definitions of the diastasis period may be derived by applying threshold-based criteria to the velocity profile, such as identifying the interval during which velocity remains below a predefined fraction of peak systolic velocity, or by using cross-correlation analysis across successive cardiac cycles to determine the most consistent quiescent interval. The method further permits the determination of a multi-heartbeat diastasis window by aggregating diastasis intervals from multiple consecutive heartbeats, thereby accommodating natural heart rate variability and producing a gating window that is both temporally precise and physiologically adaptive.

## DETAILED DESCRIPTION

- introduce patent application structure
- define terms used in patent application
- describe medical imaging context
- introduce cardiac gating concept
- define RR interval and trigger delay
- define gating parameters and gating error
- introduce Septal Scout technique
- describe Septal Scout image acquisition
- define Scout Plane and 4-chamber long-axis plane
- describe Septal Scout line acquisitions
- introduce displacement measurements
- describe displacement graph processing
- introduce velocity graph processing
- describe velocity graph features
- introduce diastasis period determination
- describe alternative diastasis period determination
- introduce 2D excitation schemes
- describe 2D excitation pulse
- introduce phase images in Septal Scouts
- describe phase image features
- introduce end-systole period determination
- describe end-systole period features
- introduce free-breathing MRI with respiratory navigators
- describe respiratory navigator functionality
- introduce real-time Septal Scout acquisition
- describe real-time cardiac-gated imaging
- introduce MRI-based cardiac gating system (MRI-CGS)
- describe calibration scan functionality
- describe calibration check functionality
- introduce ventricular systole detection
- describe cardiac gating functionality
- introduce heart rate variability tracking
- describe HRV tracking functionality
- introduce HRV check functionality
- describe MRI-CGS system flowchart
- introduce alternative ventricular systole detection
- describe Septal Scout prescription
- introduce displacement graphs at different depths
- describe phase-intensity graph at ascending aorta
- introduce systole detection comparison
- describe imaging of coronary artery stenosis
- introduce x-ray angiography image
- describe MRA image using Septal Scout method
- describe MRA image using conventional MRI technique
- introduce comparison of MRA images
- distinguish Septal Scout technique from MRI navigator techniques
- conclude patent application

The present patent application describes a comprehensive system and method for cardiac-gated magnetic resonance imaging that utilizes the Septal Scout technique to determine the optimal timing for data acquisition during the cardiac cycle. For the purposes of this disclosure, the term “Septal Scout” refers to a one-dimensional projection imaging sequence that acquires signal intensity along a prescribed plane aligned with the ventricular septum, with phase encoding disabled to maximize temporal resolution. The “Scout Plane” is defined as the thin, slab-like imaging volume oriented perpendicular to the long axis of the heart and positioned to intersect the basal portion of the interventricular septum throughout the cardiac cycle. This plane is prescribed using a prior four-chamber long-axis cine MRI acquisition, which serves as an anatomical reference for proper alignment. The Septal Scout sequence is executed with a repetition time (TR) of no greater than 10 milliseconds, producing a continuous series of one-dimensional intensity profiles sampled at intervals corresponding to the TR. Each profile represents the average signal intensity along the depth dimension of the Scout Plane, captured over a duration sufficient to encompass at least one full cardiac cycle. These intensity profiles are concatenated along the time axis to form a motion trace, analogous to M-mode echocardiography, which is then processed to derive a pseudo-displacement function using gradient optical flow analysis. The temporal derivative of this function yields a pseudo-velocity graph, characterized by distinct E and A waves corresponding to rapid filling and atrial contraction phases, respectively, with a central plateau region representing diastasis. The start and end of diastasis are determined by identifying the inflection points of this velocity curve, where the second derivative equals zero, thereby defining a gating window that is both physiologically accurate and computationally robust. Alternative methods for defining diastasis include thresholding the velocity profile at a percentage of peak systolic velocity or using cross-correlation to identify the most temporally consistent quiescent interval across multiple heartbeats. The Septal Scout may be implemented using a two-dimensional radiofrequency excitation pulse to selectively excite tissue along the septal wall within the four-chamber plane, thereby reducing contamination from static background signals and improving signal-to-noise ratio, though such an implementation requires careful optimization of pulse duration to maintain temporal resolution. Phase-encoded images may also be acquired simultaneously to provide additional information regarding tissue motion directionality, particularly useful for distinguishing septal motion from adjacent structures such as the ascending aorta. The velocity profile derived from the Septal Scout may also be used to detect the onset of ventricular systole by identifying the sharp rise in velocity immediately following the T-wave of the ECG, thereby enabling ECG-independent gating. The invention further encompasses a magnetic resonance-based cardiac gating system (MRI-CGS) that integrates the Septal Scout as a pre-scan calibration tool, followed by a calibration check to verify consistency of the diastasis window across successive heartbeats. The system tracks heart rate variability in real time and performs an HRV check to ensure the gating window remains valid throughout the imaging session. In free-breathing applications, the Septal Scout may be combined with respiratory navigators to simultaneously correct for respiratory motion, enabling uninterrupted imaging without breath-holds. The MRI-CGS operates according to a defined flowchart that initiates with a localization scan, followed by Septal Scout acquisition, diastasis window determination, gating parameter validation, and finally, high-resolution coronary MRA acquisition triggered by the validated window. The Septal Scout technique is distinguished from conventional MRI navigators, which monitor diaphragmatic motion for respiratory gating, by its direct measurement of cardiac motion with sub-10-millisecond resolution, enabling precise cardiac-phase-specific triggering. When applied to coronary MRA, the method produces images with significantly sharper vessel boundaries compared to those obtained using conventional cine-calibrated gating, as demonstrated by reduced full width at half maximum (FWHM) measurements and higher subjective image quality scores. The method does not require contrast agents and is compatible with standard MRI hardware, making it immediately translatable to clinical practice. Unlike prior techniques that rely on fixed or averaged gating windows, the Septal Scout adapts dynamically to physiological variation, ensuring consistent image quality even in patients with arrhythmias or elevated heart rates. The invention concludes with the recognition that this method fundamentally redefines the paradigm of cardiac gating by replacing indirect, low-resolution surrogate measures with direct, high-fidelity monitoring of myocardial motion, thereby enabling the routine acquisition of diagnostic-quality coronary MRA without the need for invasive procedures or ionizing radiation.