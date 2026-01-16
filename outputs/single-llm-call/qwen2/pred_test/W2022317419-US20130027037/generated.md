# DESCRIPTION

## BACKGROUND

Cardiovascular magnetic resonance (CMR) imaging has become an essential tool for assessing myocardial wall motion, particularly in the context of stress testing. Traditional CMR techniques often require patients to remain still and follow specific breathing patterns, which can be challenging, especially after intense physical activity such as treadmill exercise. Post-exercise imaging is critical because stress-induced wall motion abnormalities rapidly diminish after the cessation of exercise. Therefore, real-time imaging is often the only feasible option for capturing these transient changes.

However, real-time imaging comes with its own set of challenges. The rapid heart rate and exaggerated breathing movements that occur post-exercise can significantly degrade image quality. These factors often lead to reduced signal-to-noise ratio (SNR), lower temporal and spatial resolution, and increased artifacts, such as ghosting caused by chest wall motion. To address these issues, various advanced reconstruction techniques have been developed, including the SPIRIT (Self-Consistent Parallel Imaging Technique) method.

SPIRIT is a powerful reconstruction algorithm that leverages parallel imaging and compressed sensing to improve image quality. However, it does not fully account for the dynamic nature of coil sensitivities over time, which can vary due to patient movement and physiological changes. To enhance the performance of SPIRIT, we propose an extension that incorporates temporal sensitivity estimation (TSPIRIT) and spatial regularization. This novel approach aims to further improve the SNR and reduce artifacts in real-time exercise stress cine imaging.

## BRIEF SUMMARY

The present invention relates to a method and system for improving the image quality and signal-to-noise ratio (SNR) of real-time exercise stress cardiac cine imaging. Specifically, the invention extends the SPIRIT reconstruction technique by incorporating temporal sensitivity estimation (TSPIRIT) and spatial regularization. The method involves acquiring temporally interleaved k-space data, estimating full k-space using GRAPPA, applying Karhunen-Loeve transform filtering to improve sensitivity estimation, performing SPIRIT calibration for each frame, and using a non-linear solver with spatial regularization to generate final images. The invention has been tested on human subjects and has demonstrated a significant increase in SNR without increasing ghosting artifacts.

## DETAILED DESCRIPTION OF THE DRAWINGS AND PRESENTLY PREFERRED EMBODIMENTS

### Introduction

The invention provides a method and system for enhancing the quality of real-time exercise stress cardiac cine imaging. The primary objective is to improve the SNR and reduce artifacts, particularly ghosting caused by chest wall motion. The method builds upon the existing SPIRIT reconstruction technique by introducing temporal sensitivity estimation and spatial regularization. This combination of techniques addresses the limitations of traditional methods and provides a robust solution for post-exercise imaging.

### Method Overview

The method involves several key steps to achieve the desired improvements in image quality. First, temporally interleaved k-space data is acquired using a bSSFP sequence on a 1.5T MRI scanner. The data is then processed through a series of computational steps, including GRAPPA reconstruction, Karhunen-Loeve transform filtering, SPIRIT calibration, and non-linear solving with spatial regularization. The final images are generated after the solver converges, and their quality is evaluated against conventional TGRAPPA reconstructions.

### Data Acquisition

The data acquisition process begins with the preparation of the patient. Ten healthy volunteers (6 males, age range 23.1 to 41.1 years) with normal left ventricular thickness participated in the study. Each volunteer gave written consent and underwent a free-breathing real-time exercise stress cine examination using an MR-compatible treadmill system. The imaging was performed on a 1.5T scanner (Avanto, Siemens) equipped with a 32-channel coil (Rapid MRI).

Three slices (one short-axis and two long-axis views) were acquired for each subject. The imaging parameters included a bSSFP sequence with a repetition time (TR) of 1.09 ms, echo time (TE) of 0.9 ms, image matrix size of 160 × 80, flip angle of 58°, resolution of 2.44 × 2.44 mm², bandwidth of 1420 Hz, and an acceleration rate of 4 with time-interleaved sampling of k-space.

### Initial Processing

Once the k-space data is acquired, the initial processing step involves averaging all temporally interleaved k-space frames to generate auto-calibration signal (ACS) lines. These ACS lines are crucial for the subsequent GRAPPA reconstruction, which estimates the full k-space for every frame. GRAPPA is a widely used parallel imaging technique that reconstructs missing k-space data based on the known data from the ACS lines.

### Sensitivity Estimation

After the full k-space is estimated using GRAPPA, the next step is to apply Karhunen-Loeve transform filtering to improve the sensitivity estimation. The Karhunen-Loeve transform is a mathematical technique that decomposes the data into orthogonal components, effectively reducing noise and enhancing the quality of the sensitivity maps. This step is critical for ensuring accurate and reliable sensitivity estimation, which is essential for the subsequent SPIRIT calibration.

### SPIRIT Calibration

SPIRIT calibration is performed for each frame using the under-sampled k-space data and the estimated sensitivity maps. SPIRIT is a self-consistent parallel imaging technique that iteratively refines the sensitivity maps and the reconstructed images. The calibration process involves solving a series of linear equations to estimate the sensitivity kernels, which are then used to reconstruct the images.

### Non-Linear Solving with Spatial Regularization

The under-sampled k-space data and the estimated sensitivity kernels serve as inputs to a non-linear solver. The solver first performs an LSQR (Least Squares QR) matrix inversion to obtain an initial estimate of the image. This is followed by a non-linear conjugate gradient solver, which incorporates spatial regularization to further refine the image. Spatial regularization helps to smooth the image and reduce noise, leading to improved SNR and reduced artifacts.

### Image Generation and Evaluation

Final images are generated after the non-linear solver converges. These images are then compared to those obtained using conventional TGRAPPA reconstruction of the same raw data. The SNR is estimated using a standard method, and ghosting artifacts are quantified by computing the peak spatial cross-correlation ratio along the phase-encoding direction. The results demonstrate a significant increase in SNR (mean relative gain of 38.2 ± 17.8%, P < 1e-5) without raising ghosting artifacts (P = 0.960).

### Conclusion

The proposed TSPIRIT reconstruction scheme represents a significant advancement in real-time exercise stress cardiac cine imaging. By incorporating temporal sensitivity estimation and spatial regularization, the method achieves a substantial improvement in SNR without compromising image quality or introducing additional artifacts. This innovation has the potential to enhance the diagnostic capabilities of CMR in the context of stress testing, providing clinicians with more accurate and reliable information about myocardial wall motion.

### Drawings

#### Figure 1: Example of Improved Image Quality

Figure 1 illustrates the enhanced image quality achieved with the TSPIRIT technique compared to TGRAPPA. The figure shows a side-by-side comparison of the images, highlighting the superior clarity and reduced noise in the TSPIRIT-reconstructed images.

#### Figure 2: SNR and Artifact Scores

Figure 2 presents the measured SNR and artifact scores for both TSPIRIT and TGRAPPA reconstructions. The bar graph clearly demonstrates the significant increase in SNR (mean relative gain of 38.2 ± 17.8%, P < 1e-5) and the lack of increase in ghosting artifacts (P = 0.960) with the TSPIRIT method.

### Preferred Embodiments

While the invention has been described in the context of a 1.5T MRI scanner and a 32-channel coil, it is not limited to these specific configurations. The method can be adapted for use with different MRI systems and coils, as well as for other types of real-time imaging applications. The key steps of the method—temporal sensitivity estimation, spatial regularization, and non-linear solving—can be implemented using various algorithms and computational tools, depending on the specific requirements of the imaging scenario.

### Applications

The TSPIRIT reconstruction technique has broad applications in cardiovascular imaging, particularly in the assessment of myocardial wall motion during stress testing. It can be used in clinical settings to improve the accuracy and reliability of CMR exams, leading to better patient outcomes. Additionally, the method can be applied to research studies to advance our understanding of cardiac function and disease.

### Conclusion

In summary, the invention provides a robust and effective method for improving the image quality and SNR of real-time exercise stress cardiac cine imaging. By extending the SPIRIT technique with temporal sensitivity estimation and spatial regularization, the method addresses the challenges associated with post-exercise imaging and offers a significant advantage over conventional techniques. The results of the in vivo testing demonstrate the effectiveness of the TSPIRIT method, making it a valuable tool for both clinical and research applications.