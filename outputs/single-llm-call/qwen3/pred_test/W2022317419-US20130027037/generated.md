# DESCRIPTION

## BACKGROUND

- introduce parallel imaging  
Parallel imaging is a widely adopted technique in magnetic resonance imaging that enables accelerated data acquisition by undersampling k-space and leveraging spatial information from multiple receiver coils to reconstruct full-field-of-view images. By reducing the number of phase-encoding steps, parallel imaging significantly shortens scan time, which is particularly advantageous in dynamic imaging applications such as real-time cardiac cine, where temporal fidelity is critical. This approach has become indispensable in clinical settings where patient motion, respiratory variability, or physiological stress limit the feasibility of conventional full-sampling protocols. The method exploits the distinct spatial sensitivity profiles of each coil element to recover missing k-space data, thereby preserving image quality despite reduced acquisition time.

- describe linear approaches  
Linear parallel imaging methods, such as GRAPPA and SENSE, have been the cornerstone of accelerated MRI for over two decades. These techniques rely on pre-calibrated coil sensitivity maps and linear interpolation or inverse transformation to reconstruct undersampled data. In GRAPPA, for instance, kernel weights are learned from fully sampled central k-space lines—referred to as autocalibration signals—and applied to predict missing k-space points through convolution. SENSE, by contrast, formulates the reconstruction as a linear system where coil sensitivities are used to solve for the underlying image distribution in image space. Both methods assume spatial and temporal stationarity of coil sensitivities across the entire acquisition window, treating each frame independently or with fixed sensitivity estimates derived from averaged or low-resolution reference scans.

- limitations of linear approaches  
Despite their widespread adoption, linear parallel imaging techniques suffer from critical limitations in dynamic imaging scenarios, particularly during real-time cardiac cine under physiological stress. The assumption of static coil sensitivity profiles breaks down when rapid changes in anatomy, respiration, or heart rate induce spatial deformation and coil coupling variations. This results in residual aliasing, blurring, and elevated noise levels, especially at high acceleration factors. Furthermore, linear methods are inherently sensitive to calibration errors; any mismatch between the reference data used to compute kernels and the actual dynamic state of the patient leads to systematic artifacts. In post-exercise cardiac imaging, where breathing patterns are irregular and heart rates are elevated, these limitations manifest as ghosting artifacts and reduced signal-to-noise ratio, compromising diagnostic confidence. The inability of linear approaches to adapt coil sensitivity estimates on a frame-by-frame basis renders them suboptimal for capturing transient physiological events with high fidelity.

## BRIEF SUMMARY

- introduce preferred embodiments  
Preferred embodiments of the present invention introduce a novel reconstruction framework that dynamically estimates temporal variations in coil sensitivity during parallel imaging, thereby overcoming the fundamental limitations of static, linear methods. This framework, termed Temporal Sensitivity Estimation in Parallel Imaging Reconstruction (TSPIRIT), integrates iterative self-consistent calibration with spatial regularization to reconstruct high-fidelity images from highly undersampled k-space data acquired during real-time physiological stress. The invention enables substantial improvements in image quality without increasing scan time or introducing additional artifacts, making it uniquely suited for clinical applications such as exercise stress cardiac cine.

- motivate temporal sensitivity  
Temporal sensitivity estimation is motivated by the observation that coil sensitivity profiles are not invariant over time, particularly during dynamic physiological states such as post-exercise cardiac imaging. As the heart rate increases and respiratory motion becomes more pronounced, the relative positioning of anatomical structures with respect to the coil array changes, altering the spatial encoding characteristics of each coil element. By explicitly modeling and estimating these temporal variations, the reconstruction process becomes adaptive rather than fixed, leading to more accurate recovery of the underlying signal and suppression of noise and artifacts.

- describe first aspect  
The first aspect of the invention is a method for reconstructing magnetic resonance images by iteratively estimating time-varying coil sensitivity maps during the reconstruction process, without reliance on pre-acquired calibration data. This is achieved by incorporating a self-consistency constraint that enforces agreement between the measured undersampled k-space data and the forward model of image formation using the current estimate of coil sensitivities and image content.

- describe second aspect  
The second aspect is the integration of spatial regularization into the non-linear reconstruction solver to further suppress noise and preserve anatomical boundaries. This regularization term penalizes image gradients that are inconsistent with known tissue structures, promoting smoothness in homogeneous regions while maintaining sharpness at tissue interfaces, thereby enhancing diagnostic utility in regions of myocardial wall motion.

- describe third aspect  
The third aspect is a fully automated reconstruction pipeline that requires no manual intervention, user-defined parameters, or separate calibration scans. The entire process—from k-space acquisition to final image output—is executed in real time using a unified algorithmic framework that simultaneously solves for image content and temporal coil sensitivity maps, enabling seamless integration into clinical workflows.

- disclaim limitations  
It is expressly understood that the invention is not limited to any specific magnetic field strength, coil configuration, or pulse sequence. Nor is it confined to cardiac imaging; the principles disclosed herein are applicable to any dynamic imaging modality where temporal variation in coil sensitivity affects image fidelity. The invention does not require external hardware modifications, nor does it depend on ECG gating, breath-holding, or respiratory triggering, and is thus distinct from conventional approaches that rely on physiological synchronization.

## DETAILED DESCRIPTION OF THE DRAWINGS AND PRESENTLY PREFERRED EMBODIMENTS

- introduce linear parallel imaging algorithms  
Linear parallel imaging algorithms, such as GRAPPA and SENSE, operate under the assumption that coil sensitivity profiles remain constant throughout the duration of the imaging sequence. These methods typically compute a set of interpolation kernels or transformation matrices from a small set of fully sampled central k-space lines, which are then applied uniformly across all temporal frames to reconstruct undersampled data. While computationally efficient, this approach fails to account for physiological motion-induced changes in coil sensitivity, leading to reconstruction errors in dynamic settings.

- limitations of linear parallel imaging algorithms  
The primary limitation of linear parallel imaging algorithms lies in their inability to adapt to temporal variations in coil sensitivity. During real-time cardiac cine imaging following exercise, rapid changes in heart position, diaphragmatic motion, and respiratory drift cause the relative spatial relationship between the anatomy and the coil elements to evolve over time. Linear methods, which rely on static sensitivity maps derived from averaged or low-resolution references, cannot capture these dynamics, resulting in residual aliasing, signal loss, and increased noise. These artifacts are particularly detrimental in post-exercise imaging, where diagnostic accuracy depends on the clear visualization of transient wall motion abnormalities.

- introduce self-consistent parallel imaging with temporal sensitivity estimation (TSPIRIT)  
The present invention introduces TSPIRIT, a non-linear reconstruction framework that simultaneously estimates the time-varying coil sensitivity maps and the underlying image content through an iterative self-consistent procedure. Unlike conventional methods, TSPIRIT does not require pre-calibration or fixed sensitivity maps. Instead, it treats both the image and the coil sensitivities as unknowns to be solved for in each temporal frame, using the acquired undersampled k-space data as the sole constraint.

- describe TSPIRIT reconstruction  
TSPIRIT reconstruction begins with an initial linear reconstruction to generate a coarse estimate of the image and coil sensitivities. Subsequently, an iterative optimization process is employed, wherein the coil sensitivity maps are refined in each frame based on the current image estimate, and vice versa. This mutual refinement continues until convergence, ensuring that the reconstructed image is consistent with both the acquired k-space data and the spatially and temporally varying coil responses.

- describe application of TSPIRIT to cardiac real-time cine imaging  
TSPIRIT has been specifically optimized for real-time cardiac cine imaging during exercise stress, where high temporal resolution and robustness to motion are paramount. The method was applied to free-breathing, non-gated acquisitions at an acceleration factor of four, demonstrating significant improvements in image quality compared to conventional TGRAPPA. The ability to adapt coil sensitivities on a frame-by-frame basis allowed for the preservation of myocardial detail even during periods of rapid heart rate and irregular breathing.

- describe fully automated reconstruction  
The TSPIRIT framework operates as a fully automated pipeline, requiring no user input, manual calibration, or parameter tuning. All steps—from k-space acquisition to final image output—are performed within a single computational workflow, making it suitable for deployment in clinical environments where operator expertise and scan time are limited.

- describe various clinical uses of TSPIRIT  
Beyond cardiac stress imaging, TSPIRIT is applicable to any dynamic MRI application involving rapid physiological changes, including fetal imaging, abdominal imaging during respiration, functional brain imaging with motion, and real-time interventional procedures. Its ability to enhance SNR without increasing artifacts makes it particularly valuable in low-signal environments such as high-field systems with high acceleration or in patients with implanted devices that limit signal reception.

- show flow chart of method for parallel imaging with temporal sensitivity in magnetic resonance reconstruction  
A flow chart is provided illustrating the sequence of computational steps in the TSPIRIT method. The flow begins with the acquisition of undersampled k-space data, followed by initial linear reconstruction, temporal filtering, coil sensitivity estimation, non-linear reconstruction, and final image combination. Each step is interconnected in a closed-loop iterative structure, ensuring mutual refinement of image and sensitivity estimates.

- describe acts in flow chart  
The acts in the flow chart include: acquiring magnetic resonance data using a multi-channel coil array; performing an initial linear reconstruction to generate preliminary image and sensitivity estimates; applying temporal filtering to enhance signal consistency across frames; estimating coil sensitivity information through iterative self-consistency; performing non-linear reconstruction using a least-squares solver with spatial regularization; combining reconstructed frames via sum-of-squares coil combination; and generating final output data for display or storage.

- describe flexibility of acts in flow chart  
The acts described in the flow chart are not bound to a fixed order or sequence. Alternative implementations may reorder or combine certain steps, such as performing temporal filtering after non-linear reconstruction, or integrating spatial regularization earlier in the iterative loop. The core principle of mutual refinement between image and sensitivity estimates remains invariant across all configurations.

- show example implementation of method of parallel imaging  
An example implementation of TSPIRIT was executed on a 1.5T MRI system equipped with a 32-channel cardiac coil. Undersampled k-space data were acquired using a balanced steady-state free precession sequence with a time-interleaved sampling pattern. Reconstruction was performed on a workstation with a multi-core processor, utilizing optimized C++ and MATLAB code for numerical efficiency.

- describe data flow for temporal sensitivity framework  
Data flow begins with raw k-space measurements from each coil element, which are partitioned into temporal frames. Each frame undergoes initial linear reconstruction, followed by temporal filtering to extract coherent signal components. Coil sensitivity maps are then estimated by enforcing self-consistency between the forward model and measured data. These maps are used in the non-linear reconstruction stage to refine the image estimate, which in turn informs the next iteration of sensitivity estimation.

- acquire magnetic resonance data  
Magnetic resonance data are acquired using a multi-channel radiofrequency coil array configured to capture spatially encoded signals from the region of interest. The acquisition employs a time-interleaved undersampling pattern to reduce the number of phase-encoding steps while maintaining temporal coverage.

- describe k-space data acquisition  
K-space data are sampled at a reduced rate, with each frame containing only a subset of phase-encoding lines. The undersampling pattern is designed to preserve central k-space coverage for initial calibration and to ensure temporal consistency across frames.

- describe coil usage  
A multi-channel coil array with 32 independent receiver elements is employed, each with a distinct spatial sensitivity profile. The coil is positioned to maximize signal reception from the heart and surrounding tissues.

- describe sequence of transmissions  
The imaging sequence employs a balanced steady-state free precession (bSSFP) pulse sequence with a repetition time of 1.09 ms and echo time of 0.9 ms. Each frame is acquired with a flip angle of 58°, and the sequence is repeated at a rate sufficient to capture cardiac motion over the cardiac cycle.

- describe frames representing different times  
Each acquired k-space dataset corresponds to a discrete temporal frame, representing the state of the anatomy at a specific moment during the cardiac cycle. These frames are temporally ordered to reconstruct a cine loop.

- describe reduction factor  
The reduction factor is set to four, meaning that only one out of every four phase-encoding lines is acquired, significantly reducing scan time while maintaining diagnostic utility.

- perform initial reconstruction  
An initial linear reconstruction is performed using GRAPPA to generate preliminary image estimates and approximate coil sensitivity maps for each frame.

- describe linear reconstruction  
Linear reconstruction involves applying pre-computed interpolation kernels derived from central k-space lines to predict missing data. This step provides a starting point for the iterative process but does not account for temporal variations in sensitivity.

- filter frames  
Temporal filtering is applied to the sequence of reconstructed frames to enhance signal consistency and suppress noise that is uncorrelated across time.

- describe temporal filtering  
Temporal filtering is implemented using a Karhunen-Loeve transform to decompose the image sequence into principal components, retaining only those that represent coherent physiological motion.

- output filtered frames  
Filtered frames are passed to the next stage of reconstruction, where coil sensitivity estimation is performed using the enhanced signal content.

- estimate coil sensitivity information  
Coil sensitivity information is estimated by enforcing self-consistency: the forward model of image formation, using current estimates of image and sensitivity, must reproduce the measured k-space data within a defined tolerance.

- describe iterative self-constraint parallel imaging reconstruction calibration  
Iterative self-constraint calibration alternates between updating the image estimate and refining the coil sensitivity maps, ensuring that both are mutually consistent with the acquired data and the physical model of signal generation.

- perform non-linear reconstruction  
Non-linear reconstruction is performed using an optimization solver that minimizes a cost function composed of data fidelity and spatial regularization terms.

- describe non-linear solver  
The non-linear solver employs a combination of least-squares matrix inversion and non-linear conjugate gradient methods to efficiently navigate the high-dimensional solution space.

- describe least square matrix inversion solver  
The least-squares solver computes the optimal image estimate by minimizing the squared difference between the predicted and measured k-space data, subject to the current coil sensitivity estimates.

- describe non-linear conjugate gradient solver  
The non-linear conjugate gradient solver incorporates a spatial regularization term that penalizes high-frequency noise and enforces anatomical plausibility, guiding the solution toward physiologically realistic images.

- output reconstructed data  
Reconstructed data for each temporal frame are output as high-fidelity image volumes, with improved signal-to-noise ratio and reduced artifact levels.

- combine reconstructed data  
Reconstructed data from all coil elements are combined using a sum-of-squares method to produce a single composite image per frame.

- describe sum-of-square coil combination  
Sum-of-squares coil combination computes the magnitude of the root-sum-square of the complex images from each coil, yielding a final image with uniform signal intensity and minimal noise bias.

- generate output data  
Output data are generated as a time-resolved cine series, suitable for visual interpretation and quantitative analysis.

- describe further processing of output data  
Further processing may include motion correction, segmentation, strain analysis, or quantitative perfusion mapping, depending on clinical requirements.

- output to memory or transmission  
The final output data are stored in digital memory or transmitted to a diagnostic workstation via secure network protocols.

- describe display of output  
Output images are displayed in real time on a high-resolution monitor, enabling immediate clinical assessment during the imaging session.

- describe system for parallel imaging with temporal sensitivity in magnetic resonance reconstruction  
The system comprises a magnetic resonance imaging scanner, a multi-channel radiofrequency coil array, a digital processor, a memory unit, and a display interface, all integrated into a unified reconstruction platform.

- describe MR system  
The MR system is a 1.5T or 3T scanner equipped with a multi-channel cardiac coil and time-interleaved k-space sampling capability.

- describe memory  
Memory stores raw k-space data, intermediate reconstruction results, coil sensitivity maps, and final reconstructed images.

- describe processor  
The processor executes the TSPIRIT algorithm using optimized numerical libraries and parallel computing architectures to enable real-time reconstruction.

- describe display  
The display presents reconstructed cine loops with high temporal and spatial fidelity, enabling real-time clinical evaluation.

- store data in memory  
Raw and processed data are stored in non-volatile memory for later review, archiving, or research analysis.

- describe data stored in memory  
Data stored include undersampled k-space measurements, estimated coil sensitivity maps, reconstructed image frames, and metadata such as patient identifiers and acquisition parameters.

- implement flow as program  
The entire reconstruction flow is implemented as a software program executable on standard computing hardware.

- describe implementation languages  
Implementation is performed in C++, MATLAB, and Python, with computationally intensive components optimized using GPU acceleration.

- describe system components  
System components include the MRI scanner, coil array, data acquisition interface, reconstruction computer, memory storage, and display terminal.

- describe alternative system configurations  
Alternative configurations may include cloud-based reconstruction, embedded processors within the scanner, or integration with artificial intelligence platforms for automated diagnostic assistance.

- calculate signal-to-noise ratio (SNR)  
SNR is calculated by measuring the mean signal intensity within a region of interest in the myocardium and dividing it by the standard deviation of noise in a background region.

- describe region of interest  
The region of interest is defined as the mid-ventricular myocardial wall in the short-axis view, avoiding areas affected by motion or artifact.

- show images reconstructed using TSPIRIT and linear methods  
Images reconstructed using TSPIRIT demonstrate significantly improved myocardial border definition, reduced noise, and minimal ghosting compared to those reconstructed using TGRAPPA.

- describe advantages of TSPIRIT  
TSPIRIT offers a 38.2% average increase in SNR without increasing ghosting artifacts, enabling clearer visualization of transient wall motion abnormalities during exercise stress. It operates without requiring additional scan time, patient preparation, or manual calibration, making it uniquely suited for clinical deployment in real-time cardiac imaging.