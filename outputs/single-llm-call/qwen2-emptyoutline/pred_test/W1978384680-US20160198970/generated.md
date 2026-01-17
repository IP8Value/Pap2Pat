# DESCRIPTION

## FIELD

The present invention relates to the field of magnetic resonance imaging (MRI) and, more specifically, to a method and system for determining cardiac gating windows for magnetic resonance coronary angiography (MRCA). The invention aims to improve the accuracy and reliability of cardiac gating by utilizing a high-temporal-resolution motion monitoring sequence, referred to as the Septal Scout, to track the motion of the ventricular septum.

## BACKGROUND

Magnetic resonance coronary angiography (MRCA) is a non-invasive diagnostic tool for assessing coronary artery disease (CAD). Compared to traditional x-ray angiography, MRCA offers several advantages, including three-dimensional visualization of coronary vessel lumens and the absence of ionizing radiation and the risks associated with catheterization. However, MRCA faces significant challenges due to the long acquisition times required to achieve high spatial resolution, which can lead to motion artifacts caused by cardiac and respiratory movements.

Prospective cardiac gating is a widely used technique to mitigate these motion artifacts by synchronizing the imaging window with the diastasis period, a phase of minimal cardiac motion. Traditionally, cardiac gating is facilitated by the electrocardiogram (ECG) and a cine cardiovascular magnetic resonance (CMR) video to determine the timing of diastasis. However, the temporal resolution of conventional cine acquisitions is often insufficient to accurately capture the rapid changes in cardiac motion during the diastasis period, leading to suboptimal gating windows and blurred images.

To address these limitations, the present invention introduces a novel CMR technique called the Septal Scout. The Septal Scout measures the motion of the ventricular septum with high temporal resolution, providing a more accurate and robust method for determining the optimal cardiac gating window. This technique is designed to produce sharper MRCA images by ensuring that data acquisition occurs during periods of minimal cardiac motion.

## SUMMARY

The present invention provides a method and system for determining cardiac gating windows for magnetic resonance coronary angiography (MRCA) using a high-temporal-resolution motion monitoring sequence, referred to as the Septal Scout. The Septal Scout acquires one-dimensional (1D) projections of the ventricular septum at a high frame rate, allowing for precise tracking of septal motion and identification of the diastasis period.

The method includes the following steps:
1. **Localization**: Using a real-time MR-Echo sequence, the short-axis 2-chamber view and the long-axis 2-chamber view of the left ventricle are localized.
2. **Cine CMR Gating Window Calibration**: A 4-chamber slice is prescribed for a cine acquisition to determine the diastasis imaging window using frame-to-frame correlation coefficients.
3. **Septal Scout Gating Window Calibration**: A slice is prescribed along the ventricular septum for fast, projection imaging. The timing of diastasis is determined by analyzing the intensity plot and calculating the pseudo-velocity function.
4. **High-Resolution MRA Acquisition**: A 3D SSFP sequence is performed to acquire high-resolution MRCA images using the gating windows determined by the Septal Scout.

The system includes:
- **Imaging Device**: A magnetic resonance imaging (MRI) scanner capable of performing real-time sequences, cine acquisitions, and 3D SSFP imaging.
- **Processing Unit**: A computer system equipped with software for processing the cine and Septal Scout data to determine the optimal gating windows.
- **User Interface**: A graphical user interface for visualizing the acquired data and adjusting the imaging parameters.

The invention also includes a method for comparing the performance of the Septal Scout with tissue Doppler echocardiography (TDE) to validate the accuracy of the determined gating windows. The method involves acquiring TDE data of the interventricular septum (IVS) and comparing the estimated diastasis windows with those determined by the Septal Scout.

The primary advantage of the Septal Scout is its ability to provide high-temporal-resolution motion data, enabling more accurate and robust determination of the diastasis period. This leads to improved image quality in MRCA, with sharper and more defined coronary artery segments.

## DETAILED DESCRIPTION

### Introduction

Magnetic resonance coronary angiography (MRCA) is a promising non-invasive technique for diagnosing coronary artery disease (CAD). However, the long acquisition times required for high spatial resolution make MRCA susceptible to motion artifacts caused by cardiac and respiratory movements. Prospective cardiac gating is a critical tool for reducing these artifacts by synchronizing the imaging window with the diastasis period, a phase of minimal cardiac motion. Traditional methods for determining the diastasis period, such as cine cardiovascular magnetic resonance (CMR), often suffer from insufficient temporal resolution, leading to suboptimal gating windows and blurred images.

The present invention addresses these limitations by introducing the Septal Scout, a high-temporal-resolution motion monitoring sequence that tracks the motion of the ventricular septum. The Septal Scout provides a more accurate and robust method for determining the optimal cardiac gating window, resulting in sharper and more reliable MRCA images.

### Method for Determining Cardiac Gating Windows

#### Step 1: Localization

The first step in the method is to localize the short-axis 2-chamber view and the long-axis 2-chamber view of the left ventricle using a real-time MR-Echo sequence. This step ensures that the subsequent imaging planes are correctly aligned with the heart's anatomy.

#### Step 2: Cine CMR Gating Window Calibration

A 4-chamber slice is prescribed for a cine acquisition to determine the diastasis imaging window. The cine acquisition is performed using a breath-held SSFP sequence with the following parameters:
- Field of View (FOV): 35 cm
- In-plane resolution: 1.4 × 1.8 mm
- Slice thickness: 5 mm
- Repetition Time (TR): 3.9 ms
- Echo Time (TE): 1.7 ms
- Flip angle: 45°
- Number of frames: 30
- Lines per segment (LPS): 16

The cine MR images are cropped to reduce the amount of stationary background tissue and exported to a processing unit. Frame-to-frame correlation coefficients (CC) are calculated, and the CC function is spline-interpolated. The inflection points (second-derivative nulls) between the E and A wave peaks are used to define the start and end times of the diastasis window (Wcine).

#### Step 3: Septal Scout Gating Window Calibration

A slice is prescribed along the ventricular septum for fast, projection imaging. The Septal Scout is a modified SSFP sequence with phase encodes turned off, acquiring a 1D projection of the prescribed septal Scout Plane (5 to 10 mm thick) every TR, which may be as short as 3 ms. In practice, a 10-ms TR is chosen to balance data frame rate and processing speed. The imaging parameters are as follows:
- Flip angle: 55°
- FOV: 31 cm
- Spatial resolution along the scout: 0.8 mm

Successive Septal Scout projections are appended along a time axis, similar to M-mode ultrasound. An ROI spanning a depth of ±2.5 mm is selected to coincide with the location of the basal septum. The signals within the ROI are averaged to improve the signal-to-noise ratio (SNR). The intensity plot over time is analyzed using the gradient optical flow method to calculate pixel intensity variations and determine object motion. The pseudo-velocity function is obtained by taking the temporal derivative of the pseudo-displacement plot, revealing the characteristic E and A waves that border the diastasis period. The inflection points between the E and A waves are used to define the start and end times of the diastasis period (Wsep).

The Septal Scout acquisition is triggered from the R-peak of the ECG and lasts 5 seconds. Two acquisitions are performed at the beginning and end of a practice 20-second breath hold to account for heart rate variability (HRV) during the breath hold. The intersection of diastases across all heartbeats observed produces the multi-heartbeat diastasis window (Wsep) as determined by the Septal Scout.

#### Step 4: High-Resolution MRA Acquisition

A 3D SSFP sequence is performed to acquire high-resolution MRCA images using the gating windows determined by the Septal Scout. The acquisition parameters are as follows:
- 3D fat-suppressed SSFP
- TR: 3.9 ms
- TE: 1.9 ms
- Flip angle: 55°
- FOV: 35 × 35 × 4 cm
- Resolution: 1.5 × 1.5 × 2.0 mm
- Slice oversampling: none
- Cartesian trajectory
- α/2 pre-pulse
- 20 dummy cycles to obtain steady state

The number of TRs per heartbeat and the total scan time vary with the gating window used. The 3D dataset is reconstructed to produce high-resolution images of the coronary arteries.

### System for Determining Cardiac Gating Windows

The system for determining cardiac gating windows includes the following components:

#### Imaging Device

A magnetic resonance imaging (MRI) scanner capable of performing real-time sequences, cine acquisitions, and 3D SSFP imaging. The scanner should have a high-field strength (e.g., 1.5 T) and a multi-channel cardiac phased-array coil to ensure high image quality and spatial resolution.

#### Processing Unit

A computer system equipped with software for processing the cine and Septal Scout data. The software should include algorithms for calculating frame-to-frame correlation coefficients, spline interpolation, and gradient optical flow analysis. The processing unit should be capable of real-time data processing to provide immediate feedback to the operator.

#### User Interface

A graphical user interface (GUI) for visualizing the acquired data and adjusting the imaging parameters. The GUI should allow the operator to select the ROI for the Septal Scout, review the intensity and velocity plots, and adjust the gating windows as needed.

### Validation and Performance Evaluation

#### Comparison with Tissue Doppler Echocardiography (TDE)

To validate the accuracy of the Septal Scout, a subcohort of subjects undergoes tissue Doppler echocardiography (TDE) of the interventricular septum (IVS). The TDE data is acquired using a Philips iE33 system with a phased array transducer operating at 3.5 MHz. The IVS is imaged in an apical 4-chamber view at 150 fps, with velocity resolved at ±15 cm/s near the basal septum. Diastasis window estimates are determined from the IVS velocity plots using a custom MATLAB® script. The inflection points between the E and A waves are used to define the start and end times of the diastasis period.

The start and end times of the diastasis windows determined by the Septal Scout and TDE are compared using Pearson’s correlation coefficient and Bland-Altman analysis. High agreement is observed, with correlation coefficients greater than 0.98 (p < 0.001).

#### Image Quality Assessment

The image quality of the MRCA images acquired using the cine-calibrated and Septal-Scout-calibrated gating windows is assessed both qualitatively and quantitatively. Subjective assessment is performed by two experienced observers using a 5-point Likert scale, with scores ranging from 0 (not visible) to 4 (sharply defined edges). The scores are averaged within each vessel size category.

Quantitative assessment includes measuring the signal-to-noise ratio (SNR) and vessel sharpness. The SNR is determined by taking the mean signal intensity measured in a region-of-interest (ROI) in the aortic root blood pool and dividing it by the standard deviation of a noise ROI measured outside the patient. Vessel sharpness is determined by full width half maximum (FWHM) measurements, which are made using a custom MATLAB® script. For each coronary artery segment, 3 cross-sectional views at 1 mm increments along the vessel are selected, and the FWHMs are calculated and averaged across the radial profiles.

Statistical analysis is performed using the Wilcoxon signed-ranks test to compare the image quality metrics between the cine-CMR and Septal Scout methods. The results show that the Septal Scout method leads to significantly sharper images, with smaller FWHM values for both large and small vessel segments.

### Conclusion

The present invention, the Septal Scout, provides a novel and effective method for determining cardiac gating windows for magnetic resonance coronary angiography (MRCA). By utilizing high-temporal-resolution motion monitoring of the ventricular septum, the Septal Scout offers a more accurate and robust alternative to traditional cine-CMR techniques. The improved accuracy of the gating windows leads to sharper and more reliable MRCA images, enhancing the diagnostic value of the technique. Future work will focus on exploring the use of the Septal Scout in the presence of cardiac pathology and integrating it into a real-time cardiac motion monitoring system.