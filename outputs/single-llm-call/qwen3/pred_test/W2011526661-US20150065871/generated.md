## BACKGROUND

- motivate targeted blood-brain barrier opening  
The blood-brain barrier serves as a highly selective physiological interface that strictly regulates the passage of molecules from the systemic circulation into the central nervous system. While this barrier is essential for maintaining neural homeostasis and protecting the brain from toxins and pathogens, it also presents a formidable obstacle to the delivery of therapeutic agents for the treatment of neurological disorders. The vast majority of small-molecule pharmaceuticals and essentially all large-molecule biologics—including antibodies, gene vectors, and neurotrophic factors—are excluded from the brain parenchyma due to the tight junctions between cerebral endothelial cells, efflux transporters, and low pinocytic activity. As a result, conditions such as Parkinson’s disease, Huntington’s disease, Alzheimer’s disease, glioblastoma, and neuropsychiatric disorders remain largely inaccessible to systemic pharmacotherapy. Conventional approaches that bypass the barrier through direct intracerebral injection or intrathecal administration are inherently invasive, carry significant risks of hemorrhage, infection, and tissue trauma, and lack the spatial precision necessary for targeting deep subcortical nuclei. A non-invasive method capable of transiently and locally modulating the permeability of the blood-brain barrier at predefined anatomical targets would represent a transformative advancement, enabling precise drug delivery without compromising the integrity of surrounding neural structures or requiring surgical intervention.

- limitations of current techniques  
Existing strategies for overcoming the blood-brain barrier are fundamentally constrained by their invasiveness, lack of specificity, or dependence on drug-specific biochemical properties. Chemical modulation of the barrier using osmotic agents such as mannitol results in diffuse, non-focal disruption of the endothelial tight junctions across broad regions of the brain, increasing the risk of edema, neuroinflammation, and unintended exposure to neurotoxic substances. Conjugation of therapeutics to endogenous transport systems, such as receptor-mediated transcytosis pathways, is limited by the availability and specificity of endogenous carriers, the low efficiency of transport, and the potential for off-target binding. Intracranial catheter-based delivery, while spatially targeted, necessitates repeated surgical procedures, induces chronic gliosis, and cannot be easily re-targeted without further intervention. Furthermore, none of these methods allow for dynamic, real-time control over the extent or duration of barrier opening, nor do they enable precise targeting of deep structures such as the caudate nucleus or putamen without affecting overlying cortical regions. These limitations collectively hinder both clinical translation and experimental neuroscience applications, where reproducible, non-destructive, and region-specific modulation of the blood-brain barrier is essential.

- describe challenges of ultrasound beam aberrations  
The application of focused ultrasound for transcranial modulation of the blood-brain barrier is critically impeded by the acoustic inhomogeneity of the human and non-human primate skull. The skull’s dense, layered structure—comprising the inner and outer cortical tables separated by a porous trabecular layer—induces substantial phase aberrations, amplitude attenuation, and beam distortion due to the large mismatch in acoustic impedance and sound velocity between bone and soft tissue. At frequencies commonly employed for therapeutic ultrasound, the wavelength approaches the scale of microstructural heterogeneities within the skull, leading to severe wavefront distortion that defocuses the acoustic energy and shifts the focal point away from its intended location. This aberration effect is exacerbated at higher frequencies, where the spatial resolution is theoretically improved but the penetration and focusing fidelity are compromised. Even at intermediate frequencies such as 500 kHz, which offer a favorable compromise between penetration depth and focal size, the skull induces focal shifts of several millimeters and broadens the point-spread function, thereby reducing targeting accuracy and increasing the likelihood of unintended tissue exposure. Without compensation for these aberrations, the therapeutic focus may miss the intended target entirely or inadvertently affect adjacent structures, undermining the safety and efficacy of the procedure. Moreover, the variability in skull thickness, density, and geometry across individuals renders standardized transducer positioning insufficient, necessitating individualized targeting protocols and real-time feedback mechanisms to ensure consistent outcomes.

## SUMMARY

- introduce transcranial monitoring system  
A novel transcranial monitoring system is disclosed for the real-time assessment and control of blood-brain barrier opening during focused ultrasound sonication. This system enables the non-invasive, image-guided, and feedback-regulated disruption of the blood-brain barrier in deep subcortical brain regions without reliance on concurrent magnetic resonance imaging. The system integrates a single-element focused ultrasound transducer with a passive cavitation detector and a real-time signal processing unit, allowing for continuous monitoring of acoustic emissions generated by microbubble activity within the focal zone. By analyzing the spectral composition of backscattered ultrasound signals, the system distinguishes between stable and inertial cavitation behaviors, providing an immediate, quantitative measure of sonication safety and efficacy. This capability permits in situ adjustment of acoustic parameters during treatment, ensuring that the procedure remains within a predefined therapeutic window that maximizes barrier permeability while minimizing the risk of tissue damage.

- describe system components  
The system comprises a spherically focused ultrasound transducer operating at a center frequency of approximately 500 kHz, a hydrophone integrated coaxially within the transducer aperture for passive acoustic emission detection, a high-fidelity signal amplifier, a digitizing acquisition unit, and a programmable computing platform configured for real-time spectral analysis. The transducer is mounted on a stereotactic manipulator capable of precise spatial repositioning in three dimensions, with angular adjustments for orientation control. The hydrophone is calibrated for frequency response and sensitivity across the 0.6–5.2 MHz bandwidth and is connected to a pre-amplifier with 20 dB gain, followed by a high-resolution analog-to-digital converter sampling at a rate sufficient to resolve harmonic and broadband components of the cavitation signal. All components are housed within a rigid, acoustically transparent frame that maintains precise alignment between the transducer and hydrophone foci, ensuring consistent spatial correspondence of the ultrasound beam and detection volume.

- specify ultrasound transducer operation  
The ultrasound transducer is driven by a pulsed waveform generator that produces short-duration bursts at a pulse repetition frequency optimized for microbubble oscillation without inducing inertial collapse. The transducer emits acoustic energy with a focal length calibrated to penetrate the skull and converge at a predefined depth within the brain parenchyma, with the emitted pressure amplitude adjusted empirically to compensate for transcranial attenuation. The transducer operates in a continuous or intermittent burst mode, with each sonication pulse lasting between 10 and 100 milliseconds, and a duty cycle maintained below 10% to prevent thermal accumulation. The system is designed to operate without phase correction arrays, relying instead on geometric targeting and real-time feedback to achieve precise spatial control of the focal region.

- describe real-time monitoring component  
The real-time monitoring component continuously acquires and processes the acoustic emissions generated by microbubbles within the ultrasound focus. The system filters out harmonic, sub-harmonic, and ultra-harmonic frequencies associated with stable cavitation, isolating the broadband spectral component indicative of inertial cavitation. By computing the energy increase in both the harmonic and broadband bands relative to a baseline negative control, the system generates two real-time metrics: the harmonic energy increase (HEI) as an indicator of effective microbubble activation, and the broadband energy increase (BEI) as a surrogate for potentially damaging inertial activity. These metrics are displayed in real time and compared against predefined thresholds to determine sonication safety and success. The system automatically halts sonication if BEI exceeds a safety threshold or if HEI fails to reach a minimum efficacy threshold, providing an automated feedback loop for treatment control.

- mention computer program products and systems  
The system further comprises a non-transitory computer-readable storage medium encoded with software instructions that, when executed by a processor, perform the real-time spectral analysis, metric calculation, and decision logic for sonication control. The software is configured to receive digitized acoustic signals from the hydrophone, apply bandpass filtering to isolate the relevant frequency bands, compute integrated energy levels over defined spectral windows, and compare these values against empirically derived thresholds stored in a database. The software further interfaces with the stereotactic targeting module to register sonication targets in anatomical space and to log treatment parameters, acoustic emissions, and outcome metrics for each session. The system is operable as a standalone unit or integrated into a clinical workflow, and may be adapted for use with various transducer geometries and microbubble formulations.

## DETAILED DESCRIPTION

- introduce system for real-time transcranial monitoring of safe blood-brain barrier opening  
The system described herein provides a comprehensive framework for the safe, accurate, and reproducible opening of the blood-brain barrier via focused ultrasound without dependence on real-time magnetic resonance imaging. It enables the delivery of therapeutic agents to specific deep brain structures by combining precise stereotactic targeting with real-time acoustic feedback derived from passive cavitation detection. The system is designed to be portable, cost-effective, and scalable for use in clinical environments outside of specialized neurointerventional suites, thereby democratizing access to non-invasive blood-brain barrier modulation for neurodegenerative and neuropsychiatric disease treatment.

- describe ultrasound transducer and targeting component  
The ultrasound transducer is a single-element, spherically focused device with a center frequency of 500 kHz and a focal length of 120 mm, fabricated from piezoelectric ceramic and encapsulated in a water-tight housing. The transducer is rigidly mounted to a stereotactic manipulator with nine degrees of freedom, including three translational axes, two rotational axes for azimuth and elevation, and additional parameters for arm positioning, drive alignment, and transducer attachment. This configuration allows the system to align the ultrasound focal point with anatomical targets specified in a stereotactic coordinate system derived from preoperative structural imaging. The transducer is calibrated in vitro to determine the relationship between input voltage and in situ pressure, accounting for attenuation through a simulated skull medium.

- illustrate system 100 for real-time monitoring  
System 100 comprises a focused ultrasound transducer, a coaxially aligned hydrophone, a signal conditioning circuit, a digitizer, a central processing unit, and a user interface. The transducer emits ultrasound pulses into the subject’s skull, while the hydrophone simultaneously captures the backscattered acoustic emissions. These emissions are amplified, filtered, and digitized before being processed by software that decomposes the signal into harmonic and broadband components. The resulting energy metrics are plotted in real time on a graphical interface, with color-coded alerts indicating safe, borderline, or unsafe cavitation conditions. The system logs all parameters and outcomes for audit and subsequent analysis.

- describe subject positioning and anesthesia  
The subject is positioned supine within a stereotactic frame, with the head secured using non-metallic, acoustically transparent headrests to minimize acoustic reflection and distortion. General anesthesia is induced via intramuscular ketamine followed by inhaled isoflurane, ensuring immobility and suppression of physiological motion during sonication. Physiological parameters including heart rate, respiration, and core temperature are continuously monitored and maintained within physiological norms throughout the procedure.

- describe ultrasound transducer attachment and targeting  
The transducer is affixed to the stereotactic manipulator via a custom adapter that ensures mechanical stability and precise angular alignment. Targeting coordinates are derived from preoperative T1-weighted magnetic resonance images that have been stereotactically registered to the frame. Software translates these coordinates into manipulator settings, predicting the geometric focus and approach angle. The operator verifies alignment by projecting the predicted focal volume onto the anatomical image prior to sonication.

- describe negative control sonications  
Prior to microbubble administration, a series of negative control sonications are performed using identical acoustic parameters but in the absence of contrast agents. These controls establish baseline acoustic emissions and validate the stability of the system, ensuring that any subsequent spectral changes are attributable to microbubble activity rather than mechanical vibration or electrical artifact.

- describe microbubble injection and size-isolation  
Monodisperse microbubbles with a mean diameter of 4–5 micrometers are prepared in-house via differential centrifugation to eliminate oversized or undersized particles. The microbubbles are suspended in saline and administered intravenously at a dose of 0.1 mL/kg. Size isolation ensures uniform acoustic responsiveness and consistent cavitation dynamics across subjects.

- describe sonication procedure and focal maximum pressures  
Sonication is performed over a 120-second duration, using 100-millisecond pulses delivered at a 1 Hz repetition rate. The peak negative pressure at the focal point is maintained between 0.20 and 0.30 MPa, a range determined empirically to induce stable cavitation without triggering inertial collapse. The pressure amplitude is adjusted based on transcranial attenuation measurements to ensure the in situ pressure remains within the therapeutic window.

- describe post-sonication controls and BBB opening determination  
Immediately following the treatment sonication, a series of positive control sonications are performed at variable pressures (0.05–0.35 MPa) to characterize the pressure-response relationship of HEI and BEI. The extent of blood-brain barrier opening is subsequently confirmed using contrast-enhanced T1-weighted magnetic resonance imaging, with gadodiamide used to visualize extravasation into the parenchyma.

- describe scalp and brain tissue attenuation  
The total transcranial attenuation is quantified as 7.15 dB, comprising approximately 0.9 dB from the scalp (0.5 cm thickness), 1.0 dB from the skull (2.0 cm effective thickness), and 0.5 dB/cm from brain tissue over a 2 cm path. Emission amplitude is increased by a factor of 2.28 to compensate for total energy loss, ensuring sufficient in situ pressure at the target.

- describe emission amplitude adjustment  
Emission amplitude is calibrated using a hydrophone in a water bath and then corrected in vivo using an empirically derived attenuation coefficient derived from skull phantom studies. The adjustment is applied dynamically based on subject-specific skull thickness estimates from preoperative imaging.

- describe hydrophone positioning and acoustic emission monitoring  
The hydrophone is positioned through the central aperture of the transducer such that its focal plane is co-aligned with the ultrasound focus. This configuration ensures that the acoustic emissions detected originate exclusively from the therapeutic focal volume. The hydrophone is connected to a 20 dB amplifier and digitized at 10 MHz sampling rate to capture the full spectrum of cavitation activity.

- describe PC and amplifier usage  
A dedicated personal computer running customized software receives the amplified analog signal, performs real-time fast Fourier transforms, computes energy metrics, and applies safety thresholds. The amplifier ensures signal fidelity across the 0.6–5.2 MHz bandwidth, preserving harmonic and broadband components critical for cavitation classification.

- describe degassing system and acoustic coupling  
A degassing system removes entrained air from the coupling medium between the transducer and the scalp, ensuring efficient acoustic transmission. Ultrasound gel is applied to the scalp and maintained under constant pressure to eliminate air gaps and reduce reflection losses.

- describe transducer-hydrophone assembly mounting  
The transducer and hydrophone are rigidly mounted within a single housing constructed from acoustically transparent polymer, ensuring mechanical stability and precise spatial alignment. The assembly is attached to the stereotactic manipulator via a locking mechanism that prevents rotational drift during sonication.

- describe individualized targeting of ultrasound focus  
Individualized targeting is achieved by acquiring high-resolution T1-weighted structural images that are stereotactically aligned to the frame. Software converts anatomical coordinates into manipulator settings, accounting for the unique geometry of each subject’s skull and brain anatomy.

- describe T1 weighted stereotactically aligned structural images  
T1-weighted volumetric images are acquired using a 3D spoiled gradient-echo sequence with isotropic 1 mm resolution. These images are registered to the stereotactic frame using fiducial markers, enabling precise mapping of anatomical targets to manipulator coordinates.

- describe software usage for targeting in stereotactic coordinate frames  
A custom R-based software package, stereotax.R, computes the nine manipulator parameters required to position the ultrasound focus at a desired anatomical location. The software predicts the focal point and approach angle, and inverts the transformation to generate multiple possible manipulator configurations for a given target and desired incidence angle.

- describe stereotactic manipulator setting determination  
The manipulator setting is determined by optimizing the nine degrees of freedom to align the transducer’s focal point with the target, while maximizing the perpendicularity of the ultrasound beam to the skull surface to minimize aberration. The software outputs the optimal configuration, which is verified by the operator prior to sonication.

- describe focal point and axis determination  
The focal point is defined as the geometric convergence of the ultrasound beam, and the axis is defined as the vector extending from the transducer surface to the focal point. The software projects a 3D volume of expected BBB opening around this axis, based on empirical pressure-distance relationships.

- describe predicted region of BBB opening projection  
The software generates a three-dimensional ellipsoid representing the predicted region of barrier opening, centered on the focal point and elongated along the ultrasound axis. This volume is overlaid on the subject’s T1 image to confirm spatial correspondence with the intended target.

- describe software inversion for optimal approach angle determination  
The software inverts the geometric model to compute all possible manipulator configurations that direct the ultrasound beam toward a given target from a specified approach angle. The user selects the angle that minimizes transcranial path length and maximizes skull perpendicularity.

- describe BBB opening verification using contrast-enhanced MRI  
After sonication, the subject is transferred to an MRI scanner for acquisition of T1-weighted pre- and post-gadodiamide images. The difference between these images is used to quantify the spatial extent of barrier opening, with regions of increased signal intensity indicating extravasation of the contrast agent.

- describe T2 and T2 FLAIR image acquisition  
T2-weighted and T2 fluid-attenuated inversion recovery sequences are acquired to detect edema, hemorrhage, or other tissue abnormalities that may result from excessive sonication energy. These sequences are analyzed for hyperintensity or signal loss indicative of structural damage.

- describe T1 contrast agent injection and imaging  
Gadodiamide is administered intravenously at 0.15 mL/kg, and T1-weighted imaging is performed 30 minutes post-injection to capture contrast enhancement in regions where the blood-brain barrier has been transiently opened.

- describe image registration and gadodiamide concentration estimation  
Pre- and post-contrast T1 images are registered to the stereotactic reference image using affine and non-rigid transformations. The post/pre signal ratio is computed pixel-wise to estimate gadodiamide concentration, and symmetric hemispheric subtraction is applied to eliminate confounding signals from vascular structures.

- describe targeting accuracy assessment  
Targeting accuracy is quantified by comparing the center of mass of the observed BBB opening with the predicted focal point. The error is computed in the lateral (x-y) and axial (z) planes, with mean displacement and standard deviation calculated across multiple sonication sessions.

- describe image rotation and shifting into new coordinate frame  
The image data are rotated and translated such that the origin corresponds to the predicted focal point and the z-axis aligns with the ultrasound propagation direction. This transformation allows for standardized analysis of opening volume and spatial distribution independent of subject-specific orientation.

- describe voxel consideration and targeting accuracy quantification  
Voxels exhibiting a T1 signal increase greater than 10% relative to baseline are classified as “opened.” The fraction of opened voxels within a 15×15×17 mm region of interest is computed, and the center of mass of the contiguous opened region is used to determine the observed focal position.

- describe real-time monitoring using frequency content of backscattered acoustic signal  
The frequency spectrum of the backscattered signal is analyzed to identify energy shifts attributable to microbubble dynamics. Harmonic frequencies indicate stable cavitation, while broadband noise indicates inertial cavitation. The system filters out harmonics and computes the residual broadband energy as a safety metric.

- describe cavitation behavior inference and energy calculation  
The harmonic energy increase (HEI) is defined as the difference between the total energy after microbubble injection and the baseline negative control, minus the broadband component. The broadband energy increase (BEI) is computed as the energy in the 0.6–5.2 MHz band after subtraction of harmonic contributions. HEI above 15 dB and BEI below 6 dB are indicative of safe, effective sonication.