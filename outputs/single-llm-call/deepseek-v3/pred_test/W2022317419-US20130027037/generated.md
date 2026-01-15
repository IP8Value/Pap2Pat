Here is the patent application following the provided outline and research paper content:

# DESCRIPTION  

## BACKGROUND  

Parallel imaging techniques have been developed to accelerate magnetic resonance imaging (MRI) acquisitions by exploiting spatial encoding information from multiple receiver coils. Conventional parallel imaging methods rely on linear reconstruction algorithms that estimate missing k-space data using pre-calibrated coil sensitivity profiles. These linear approaches, such as GRAPPA and SENSE, have enabled significant reductions in scan times while maintaining image quality for many clinical applications.  

However, linear parallel imaging methods exhibit notable limitations when applied to dynamic imaging scenarios, particularly in cardiac MRI during stress conditions. The primary constraint stems from the assumption of static coil sensitivity profiles throughout the acquisition. In reality, coil sensitivities vary temporally due to physiological motion, respiratory patterns, and changes in tissue properties during stress protocols. These temporal variations degrade image quality when using conventional reconstruction methods, manifesting as increased noise, residual artifacts, and reduced spatial resolution.  

The limitations become particularly pronounced in real-time exercise stress cardiac cine imaging, where both the rapid cardiac motion and exaggerated breathing patterns following exertion create dynamic changes in coil sensitivity profiles. Traditional linear reconstruction methods cannot adequately account for these temporal variations, resulting in compromised image quality precisely when diagnostic clarity is most critical for detecting stress-induced wall motion abnormalities.  

## BRIEF SUMMARY  

The present invention provides improved methods and systems for parallel imaging with temporal sensitivity estimation in magnetic resonance reconstruction. Preferred embodiments address the limitations of conventional approaches by incorporating dynamic coil sensitivity estimation and spatial regularization into the reconstruction process.  

The temporal sensitivity aspect of this invention is particularly motivated by the need for high-quality imaging during dynamic physiological states, such as post-exercise cardiac function assessment. By accounting for time-varying coil sensitivities, the disclosed techniques maintain image quality even under conditions of rapid heart rate and exaggerated breathing patterns that typically degrade conventional parallel imaging results.  

A first aspect of the invention relates to a self-consistent parallel imaging method with temporal sensitivity estimation (TSPIRIT) that performs iterative reconstruction while continuously updating coil sensitivity information throughout the acquisition. This approach overcomes the static sensitivity limitation of linear methods by incorporating temporal filtering and adaptive calibration.  

A second aspect involves a non-linear reconstruction framework that combines temporal sensitivity estimation with spatial regularization. The method employs advanced solvers including least square matrix inversion and non-linear conjugate gradient techniques to optimize image quality while suppressing noise and artifacts.  

A third aspect provides a fully automated reconstruction pipeline specifically optimized for cardiac real-time cine imaging applications. The system integrates k-space acquisition, temporal filtering, sensitivity estimation, and non-linear reconstruction into a seamless workflow suitable for clinical implementation.  

It should be noted that while the disclosed embodiments demonstrate particular advantages for cardiac stress imaging, the invention is not limited to this application. The principles of temporal sensitivity estimation and non-linear reconstruction may be beneficially applied to various other dynamic MRI scenarios where conventional parallel imaging methods prove inadequate.  

## DETAILED DESCRIPTION OF THE DRAWINGS AND PRESENTLY PREFERRED EMBODIMENTS  

Linear parallel imaging algorithms such as GRAPPA and SPIRIT operate by using pre-calibrated coil sensitivity information to reconstruct undersampled k-space data. These methods typically acquire additional auto-calibration signal (ACS) lines during a prescan or interspersed throughout the acquisition to estimate the necessary reconstruction kernels. While effective for static imaging, these linear approaches fail to account for temporal variations in coil sensitivity that occur during dynamic acquisitions.  

The limitations of linear parallel imaging algorithms become particularly evident in real-time cardiac cine imaging following exercise stress. The rapid cardiac motion combined with exaggerated breathing patterns creates complex, time-dependent changes in coil sensitivity profiles that cannot be adequately captured by static calibration methods. This results in degraded image quality precisely when diagnostic information about stress-induced wall motion abnormalities is most critical.  

The present invention introduces self-consistent parallel imaging with temporal sensitivity estimation (TSPIRIT) to address these limitations. TSPIRIT extends conventional SPIRIT reconstruction by incorporating dynamic sensitivity estimation and spatial regularization. The method maintains the computational efficiency of SPIRIT while significantly improving image quality for dynamic applications through its temporal adaptation capability.  

The TSPIRIT reconstruction process begins with acquisition of undersampled k-space data using an interleaved sampling pattern. Rather than relying on static ACS lines, the method generates initial sensitivity estimates by averaging temporally interleaved k-space frames. These initial estimates are then refined through an iterative process that incorporates temporal filtering and adaptive calibration.  

Application of TSPIRIT to cardiac real-time cine imaging demonstrates particular advantages. The method has been shown to provide an average 38.2% improvement in signal-to-noise ratio compared to conventional TGRAPPA reconstruction, without increasing ghosting artifacts. This significant SNR gain enables clearer visualization of myocardial wall motion during the critical post-exercise period when stress-induced abnormalities are most evident.  

The fully automated reconstruction pipeline integrates several key components: k-space data acquisition, temporal filtering, sensitivity estimation, non-linear reconstruction, and final image combination. This integrated approach eliminates the need for manual intervention or parameter tuning, making it practical for routine clinical use.  

Various clinical applications benefit from the TSPIRIT approach, particularly in cardiac stress testing where image quality is often compromised by physiological motion. The method's ability to maintain high spatial and temporal resolution during rapid heart rates and exaggerated breathing patterns provides clinicians with clearer visualization of stress-induced wall motion abnormalities.  

A flow chart illustrates the method for parallel imaging with temporal sensitivity in magnetic resonance reconstruction. The process begins with acquisition of undersampled k-space data from multiple receiver coils. The acquired data undergoes initial linear reconstruction to generate preliminary images, which are then temporally filtered to improve sensitivity estimation.  

The acts in the flow chart demonstrate flexibility in implementation. The temporal filtering stage may employ various approaches including Karhunen-Loeve transform filtering to optimize sensitivity estimation. Similarly, the non-linear reconstruction stage may utilize different solver configurations depending on computational resources and desired image quality tradeoffs.  

An example implementation demonstrates the method of parallel imaging with temporal sensitivity estimation. The data flow begins with acquisition of k-space data using an interleaved sampling pattern with a reduction factor of 4. Multiple receiver coils simultaneously acquire data throughout the cardiac cycle, with each frame representing a different time point in the dynamic sequence.  

The magnetic resonance data acquisition involves collection of k-space lines using a balanced steady-state free precession (bSSFP) sequence with appropriate repetition time (TR) and echo time (TE) parameters. A 32-channel cardiac coil array provides the necessary spatial encoding information, with each coil element contributing unique sensitivity information throughout the acquisition.  

The sequence of transmissions is carefully designed to maintain consistent coverage of k-space while allowing for temporal interpolation. Frames representing different times in the cardiac cycle are acquired with appropriate temporal spacing to capture both rapid cardiac motion and slower respiratory patterns. The reduction factor of 4 enables significant acceleration while maintaining sufficient data for accurate reconstruction.  

Initial reconstruction employs linear methods such as GRAPPA to generate preliminary images from the undersampled data. These initial estimates serve as input to the temporal filtering stage, where Karhunen-Loeve transform filtering is applied to separate signal components from noise and improve sensitivity estimation accuracy.  

Temporal filtering enhances the quality of sensitivity estimates by exploiting correlations between successive time frames. The filtered frames provide cleaner input for the subsequent sensitivity estimation stage, where SPIRIT calibration is performed for every frame to capture temporal variations in coil sensitivity profiles.  

Output of the filtered frames feeds into the coil sensitivity information estimation process. This stage employs iterative self-consistent parallel imaging reconstruction calibration that updates sensitivity estimates based on both spatial and temporal correlations in the data. The calibration process accounts for physiological motion patterns to maintain accuracy throughout the acquisition.  

Non-linear reconstruction follows sensitivity estimation, utilizing advanced solvers to optimize image quality. The system implements a least square matrix inversion solver as an initial step, followed by a non-linear conjugate gradient solver with spatial regularization. This combination provides robust reconstruction while suppressing noise and artifacts.  

The non-linear solver incorporates spatial regularization to further improve image quality. Regularization terms are carefully balanced to maintain diagnostic features while reducing noise amplification inherent in parallel imaging reconstructions. The solver iterates until convergence criteria are met, ensuring optimal reconstruction quality.  

Output of the reconstructed data undergoes final combination using sum-of-squares coil combination. This stage integrates information from all receiver coils to generate composite images with optimal signal-to-noise characteristics. The combined data represents the final output of the reconstruction pipeline.  

Further processing of output data may include additional filtering or reformatting for specific clinical applications. The system provides flexibility to tailor the final output to particular diagnostic needs, such as cine loop generation for cardiac functional analysis.  

Output to memory or transmission enables storage and distribution of the reconstructed images. The system supports various output formats suitable for clinical picture archiving and communication systems (PACS), as well as direct display on monitoring equipment.  

Display of output includes options for real-time visualization during acquisition as well as retrospective review. The system supports simultaneous display of multiple imaging planes and temporal sequences, facilitating comprehensive assessment of dynamic physiological processes.  

A system for parallel imaging with temporal sensitivity in magnetic resonance reconstruction includes several key components. The MR system comprises the scanner hardware, gradient systems, and RF coils necessary for data acquisition. A 32-channel cardiac coil array provides the spatial encoding information required for parallel imaging.  

Memory components store both raw k-space data and reconstructed images throughout the processing pipeline. The system maintains sufficient buffer capacity to handle the large datasets generated by dynamic acquisitions while ensuring rapid access for reconstruction algorithms.  

Processor elements implement the reconstruction pipeline, including specialized hardware for computationally intensive operations such as non-linear solving and temporal filtering. The system architecture allows for parallel processing to maintain reconstruction speeds compatible with real-time clinical applications.  

Display components present the final reconstructed images with appropriate formatting for diagnostic interpretation. The system supports high-resolution monitors capable of displaying dynamic sequences at frame rates sufficient to visualize rapid cardiac motion.  

Storage of data in memory occurs at multiple stages of the processing pipeline. Raw k-space data is preserved for potential retrospective reconstruction, while intermediate processing steps and final images are stored in organized structures for efficient retrieval.  

The data stored in memory includes both the acquired MR signals and derived information such as sensitivity maps and reconstruction parameters. This comprehensive storage enables quality control and potential reprocessing if needed.  

Implementation of the processing flow as a program allows for flexible deployment across different hardware platforms. The reconstruction algorithms may be coded in high-performance languages such as C++ with appropriate optimizations for parallel processing architectures.  

System components are designed for interoperability with existing clinical MRI systems. The invention may be implemented as a software upgrade to conventional scanners or as part of specialized cardiac imaging packages.  

Alternative system configurations accommodate varying clinical needs and resource availability. The reconstruction pipeline may be executed entirely on the scanner hardware or distributed across networked computing resources to balance processing loads.  

Calculation of signal-to-noise ratio (SNR) provides quantitative assessment of reconstruction quality. The system automatically computes SNR metrics for specified regions of interest, enabling objective comparison between different reconstruction methods and parameter settings.  

Region of interest analysis focuses on diagnostically relevant areas such as the myocardial wall in cardiac applications. The system provides tools for interactive ROI placement and automated tracking of anatomical features throughout dynamic sequences.  

Comparative images demonstrate the advantages of TSPIRIT reconstruction versus conventional linear methods. Side-by-side displays show significant improvements in SNR and artifact reduction, particularly in challenging conditions such as post-exercise cardiac imaging.  

The advantages of TSPIRIT include not only improved SNR but also maintained spatial resolution and reduced artifacts compared to conventional parallel imaging methods. These benefits are particularly evident in stress cardiac imaging where rapid physiological changes typically degrade image quality. The invention's ability to adapt to temporal variations in coil sensitivity profiles represents a significant advancement in dynamic MRI reconstruction techniques.