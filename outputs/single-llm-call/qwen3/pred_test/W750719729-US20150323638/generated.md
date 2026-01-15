# DESCRIPTION

## FIELD OF THE INVENTION

- relate to imaging methods

The present invention relates to noninvasive imaging methods for the quantitative assessment of hemodynamic parameters in cardiovascular vasculature, particularly for the determination of fractional flow reserve (FFR) through the use of multi-dimensional phase-contrast magnetic resonance imaging. These methods are specifically directed toward the acquisition, reconstruction, and analysis of time-resolved three-dimensional velocity fields within coronary arteries to derive pressure gradients that correlate with the functional significance of stenotic lesions. The invention encompasses systems, sequences, and computational protocols that enable the precise mapping of blood flow dynamics without the need for invasive catheterization or contrast agents, thereby providing a clinically viable alternative to conventional invasive FFR measurements. The methods are applicable to patients with intermediate coronary artery disease, where the anatomical severity of stenosis is ambiguous and functional assessment is required to guide therapeutic decision-making. The invention further extends to the integration of physiological modeling with high-resolution magnetic resonance data to compute intravascular pressure differentials from velocity measurements alone, leveraging the physical principles of fluid dynamics to infer hemodynamic consequences of vascular narrowing.

## BACKGROUND

- describe limitations of current methods

Current clinical practice for evaluating the functional impact of coronary artery stenosis relies predominantly on invasive fractional flow reserve (FFR), which requires the insertion of a pressure-sensing guidewire into the coronary artery during cardiac catheterization. This procedure carries inherent risks including vascular injury, arrhythmia, contrast-induced nephropathy, and radiation exposure, while also imposing substantial logistical and economic burdens on healthcare systems. Alternative noninvasive modalities such as computed tomography angiography (CTA) provide excellent anatomical detail but lack the ability to quantify the physiological significance of stenosis, often resulting in overestimation of lesion severity and unnecessary revascularization. While positron emission tomography and stress echocardiography offer functional assessments, they are limited by poor spatial resolution, operator dependence, or inability to directly measure pressure gradients. Recent efforts to employ phase-contrast magnetic resonance imaging for noninvasive pressure gradient estimation have been confined to larger vessels such as the aorta, carotid, and renal arteries, where flow is more stable and motion artifacts are less pronounced. The coronary arteries, by contrast, present unique challenges due to their small diameter, rapid motion, and complex pulsatile flow patterns influenced by cardiac and respiratory cycles. Existing phase-contrast techniques have demonstrated insufficient reproducibility in coronary applications due to inadequate temporal resolution, limited spatial coverage, and susceptibility to motion-induced phase errors. Furthermore, prior methods have not consistently integrated the Navier-Stokes equations into clinical workflows for deriving pressure fields from velocity data, nor have they optimized acquisition protocols to accommodate the narrow quiescent periods of coronary flow. As a result, there remains a critical unmet need for a robust, reproducible, and clinically translatable noninvasive method capable of accurately quantifying coronary pressure gradients and correlating them with FFR values.

## SUMMARY

- introduce method for quantifying FFR

The present invention introduces a novel method for quantifying fractional flow reserve (FFR) in coronary arteries through the noninvasive acquisition and computational analysis of three-dimensional, time-resolved blood velocity fields using phase-contrast magnetic resonance imaging. This method enables the derivation of intravascular pressure gradients that correlate directly with the functional severity of stenotic lesions, thereby eliminating the necessity for invasive catheter-based measurements.

- describe multi-dimensional phase-contrast magnetic resonance sequence

The method employs a multi-dimensional phase-contrast magnetic resonance sequence that is specifically optimized for coronary imaging, incorporating ECG-triggering and navigator-gating to synchronize data acquisition with cardiac and respiratory motion. The sequence acquires velocity-encoded data in three orthogonal directions across multiple contiguous slices, enabling the reconstruction of a comprehensive four-dimensional flow field encompassing spatial coordinates and time.

- determine pressure gradient within blood vessel segment

Within each acquired volume of interest, the pressure gradient is determined by applying the Navier-Stokes equations to the measured velocity field, accounting for inertial, viscous, and pressure forces acting on the blood. The resulting pressure field is computed across adjacent segments of the vessel, allowing for the direct determination of the transstenotic pressure difference.

- correlate pressure gradient to FFR value

The computed pressure gradient is then correlated to the fractional flow reserve value by applying a validated physiological model that relates the pressure drop across a stenosis to the ratio of distal coronary pressure to aortic pressure under maximal hyperemia, thereby enabling the noninvasive estimation of FFR without pharmacological stimulation.

- describe MRI system for executing sequence

The method is executed using a high-field magnetic resonance imaging system equipped with multi-channel phased-array coils, high-performance gradient systems, and real-time motion correction capabilities, all configured to support the precise temporal and spatial requirements of coronary phase-contrast imaging.

- describe non-transitory machine-readable medium with instructions

The invention further includes a non-transitory machine-readable medium containing executable instructions that, when loaded onto a processor, cause the system to perform the sequence acquisition, velocity reconstruction, pressure gradient calculation, and FFR correlation steps in an automated and reproducible manner.

## DETAILED DESCRIPTION

- define terms used in application

For the purposes of this disclosure, the term “fractional flow reserve” refers to the ratio of maximum blood flow distal to a stenotic lesion to the maximum blood flow in the absence of the lesion, typically measured under conditions of hyperemia. The term “phase-contrast magnetic resonance imaging” denotes a magnetic resonance technique that encodes the velocity of moving spins into the phase of the received signal. The term “pressure gradient” refers to the spatial rate of change of pressure along the direction of flow, expressed in units of millimeters of mercury. The term “Navier-Stokes equations” refers to the fundamental equations of fluid dynamics that describe the motion of viscous fluids under the influence of external and internal forces. The term “volume of interest” refers to a defined three-dimensional region within the coronary vasculature selected for quantitative analysis. The term “ECG-triggering” refers to the synchronization of image acquisition with the cardiac cycle based on the R-wave of the electrocardiogram. The term “navigator-gating” refers to the use of a respiratory motion sensor to accept or reject data acquisition based on diaphragmatic position.

- describe FFR technique

Fractional flow reserve is a well-established metric used to assess the hemodynamic significance of coronary artery stenoses. Traditionally, FFR is measured invasively by inserting a pressure sensor into the coronary artery and inducing maximal hyperemia through pharmacological agents. The resulting pressure ratio between distal and proximal segments provides a direct measure of flow limitation. The present invention replaces this invasive measurement with a computational derivation based on noninvasively acquired velocity data.

- describe advantages of MRI over CT

Magnetic resonance imaging offers superior soft-tissue contrast, absence of ionizing radiation, and the intrinsic ability to quantify flow velocities without exogenous contrast agents, making it inherently safer and more suitable for repeated assessments. Unlike CT, which provides only anatomical information, MRI enables the direct measurement of hemodynamic parameters essential for functional evaluation.

- describe method of using MRI for quantifying FFR

The method begins with the acquisition of a multi-slice, three-directional phase-contrast dataset covering the proximal and mid segments of the left anterior descending coronary artery. Each slice is acquired over multiple cardiac phases, with velocity encoding calibrated to the expected flow range. The raw k-space data are reconstructed using a generic Fourier transform method to yield complex-valued velocity maps. These maps are then processed to derive the full four-dimensional flow velocity field.

- describe multi-dimensional PC-MR sequence

The sequence utilizes a gradient-echo-based phase-contrast pulse sequence with flow encoding in three orthogonal directions, triggered to the R-wave of the ECG and gated by a navigator echo placed on the diaphragm. The sequence employs view-sharing techniques to compensate for acquisition windows exceeding the quiescent period, ensuring sufficient temporal resolution without increasing scan time.

- calculate pressure gradient using Navier-Stokes equations

The Navier-Stokes equations are solved numerically at each voxel using finite difference methods, with velocity gradients computed from spatial derivatives of the reconstructed velocity field. The pressure gradient field is then integrated along the centerline of the vessel to determine the total pressure drop across the stenotic segment.

- describe image reconstruction using generic Fourier transform methods

Raw k-space data are reconstructed using a fast Fourier transform algorithm, followed by phase correction to remove background phase offsets. The resulting complex images are used to generate magnitude and velocity maps for each cardiac phase and flow encoding direction.

- derive 4D flow velocity field

The three-directional velocity components are combined to form a four-dimensional velocity field (x, y, z, t), representing the complete spatiotemporal motion of blood within the volume of interest.

- calculate velocity derivatives and pressure gradient field

Spatial derivatives of velocity are computed using central difference schemes, and temporal derivatives are estimated using interpolation between cardiac phases. These derivatives are substituted into the Navier-Stokes equations to solve for the pressure gradient at each voxel.

- obtain transtenotic pressure difference

The pressure gradient field is integrated along the centerline of the vessel from a proximal reference point to a distal reference point, yielding the total transstenotic pressure difference, which is then normalized to derive the FFR value.

- describe VOI in subject

The volume of interest is manually or automatically delineated to include the segment of the coronary artery encompassing the stenosis and adjacent regions of normal flow, ensuring accurate comparison of pre- and post-stenotic hemodynamics.

- describe imaging parameters

Imaging parameters include an in-plane resolution of 0.5–0.7 mm, slice thickness of 3–4 mm, flip angle of 10–20°, temporal resolution of 60–80 ms per phase, and a total scan duration of 2–5 minutes per slice. Velocity encoding (VENC) is individually calibrated for each subject based on preliminary scout scans.

- describe cardiac phase

Data acquisition is synchronized to occur during mid-diastole and end-expiration, when coronary flow is most quiescent and motion artifacts are minimized.

- describe scan time

The total scan time per slice ranges from 1 to 5 minutes, depending on the number of cardiac phases and navigator efficiency, with total examination time not exceeding 30 minutes.

- describe acquisition window

The acquisition window is designed to be less than 120 ms per phase, with view-sharing applied when necessary to maintain temporal fidelity.

- describe ECG-triggering and navigator-gating

ECG-triggering ensures that data acquisition is initiated at the R-wave, while navigator-gating accepts data only when the diaphragm position falls within a predefined tolerance window, thereby reducing respiratory motion artifacts.

- describe MRI system

The MRI system comprises a 3 Tesla or higher field strength scanner with a dedicated cardiac coil array, high-slew-rate gradients, and real-time motion correction software.

- describe processor and its functions

The processor is configured to execute the reconstruction algorithms, compute velocity derivatives, solve the Navier-Stokes equations, and generate pressure and FFR maps.

- describe computer and its functions

The computer is operatively connected to the MRI system and contains software for protocol control, data transfer, and post-processing, including user interfaces for VOI selection and result visualization.

- describe non-transitory machine-readable medium

The non-transitory machine-readable medium stores executable code that, when run, automates the entire pipeline from image acquisition to FFR calculation, ensuring consistency and reducing operator variability.

- describe method for diagnosing cardiovascular disease

The method provides a noninvasive diagnostic pathway for identifying hemodynamically significant coronary stenoses, enabling risk stratification and guiding decisions regarding revascularization.

- describe stenosis

Stenosis refers to the pathological narrowing of a blood vessel lumen, typically caused by atherosclerotic plaque buildup, which impedes blood flow and may lead to myocardial ischemia.

- describe mild, moderate, and severe stenosis

Mild stenosis is defined as luminal narrowing less than 40%, moderate as 40–69%, and severe as 70% or greater, with functional significance typically indicated by an FFR value below 0.80.

- describe alternative imaging systems

Alternative imaging systems such as CT angiography, positron emission tomography, and Doppler ultrasound may be used for anatomical or functional assessment but lack the combined spatial, temporal, and hemodynamic precision of the disclosed MRI method.

- describe equivalents to methods and materials

Equivalents to the disclosed methods and materials, including alternative pulse sequences, reconstruction algorithms, or computational models, that achieve substantially the same result through substantially the same means are intended to be encompassed within the scope of the invention.

## EXAMPLES

### Example 1

- describe 3D PC-MR sequence

The three-dimensional phase-contrast magnetic resonance sequence was implemented on a 3T scanner with a 32-channel cardiac coil. The sequence employed a gradient-echo readout with bipolar flow-encoding gradients in all three directions, triggered to the R-wave and gated by a pencil-beam navigator placed on the right hemidiaphragm. The acquisition window was set to 95 ms per cardiac phase, with 20 phases reconstructed over the cardiac cycle. The VENC was individually calibrated to 40 cm/s for each subject based on a preliminary scout scan. Six contiguous slices were acquired, covering the proximal and mid left anterior descending artery, with an in-plane resolution of 0.6 mm × 0.6 mm and slice thickness of 3.2 mm. The total scan time per slice was approximately 2.5 minutes, with navigator efficiency exceeding 80%.

- describe acquisition window and gating

The acquisition window was designed to coincide with the quiescent period of coronary flow during mid-diastole, with navigator gating accepting data only when the diaphragm position varied by less than 2 mm from the reference position. This ensured minimal respiratory motion contamination and high reproducibility across repeated scans.

- describe imaging parameters

Imaging parameters included a flip angle of 15°, repetition time of 4.2 ms, echo time of 2.1 ms, bandwidth of 1000 Hz/pixel, and parallel imaging with an acceleration factor of 2. The total acquisition time for six slices was 15 minutes, with each slice independently reconstructed and analyzed.

### Example 2

- describe phantom studies

A flow phantom was constructed using silicone tubing with an internal diameter of 4.8 mm and a 40% stenosis created by a constrictive ring. The phantom was perfused with a gadolinium-doped water solution at a constant flow rate of 300 mL/min, simulating physiological coronary flow conditions. Phase-contrast MRI was performed using the same sequence and parameters as in volunteer studies. Velocity measurements in all three directions demonstrated excellent reproducibility, with intra-class correlation coefficients exceeding 0.95. The computed pressure gradient across the stenosis was 0.98 ± 0.07 mmHg, closely matching the pressure difference measured by a calibrated pressure transducer inserted into the phantom (1.02 ± 0.05 mmHg), confirming the accuracy of the computational model.

### Volunteer Studies

- describe volunteer studies

Four healthy adult volunteers (age 28–35, mean 31.5) underwent two repeated scans on separate days. A total of 19 slices were analyzed across all subjects. The averaged maximum through-plane velocity was 16.5 ± 4.0 cm/s. Intra-class correlation coefficients for velocity components were 0.93 and 0.96 for the through-plane direction, and 0.83–0.86 for in-plane components, indicating high reproducibility. The pressure gradient across the vessel segment ranged from 0.05 to 0.25 mmHg, with an average ICC of 0.51, reflecting moderate reproducibility due to physiological variability and residual motion artifacts. Despite this, the consistency of velocity measurements supports the feasibility of the method for clinical translation.

### Example 3

- describe quantification of pressure gradient

The pressure gradient was quantified by integrating the pressure field along the centerline of the vessel from a proximal reference point 10 mm upstream of the stenosis to a distal reference point 10 mm downstream. The resulting transstenotic pressure difference was used to compute an estimated FFR value using a linear regression model derived from invasive FFR data in a separate cohort.

- describe healthy human volunteer data

Healthy volunteers exhibited negligible pressure gradients (mean 0.10 ± 0.08 mmHg), consistent with the absence of hemodynamically significant stenosis. No subject demonstrated an FFR estimate below 0.90, confirming the method’s ability to distinguish normal from abnormal physiology.

- describe noninvasive FFR measurement

The noninvasive FFR measurement was derived by applying a calibrated conversion factor between the measured pressure gradient and the known FFR threshold of 0.80, enabling direct clinical interpretation without the need for pharmacological stress.

- describe various methods and techniques

Various methods for velocity reconstruction, phase correction, and noise reduction were tested, including total variation denoising, principal component analysis, and temporal filtering. The combination of view-sharing and navigator-gating provided the most consistent results.

- describe objectives and advantages

The primary objective was to establish a reproducible, noninvasive method for FFR estimation. Advantages include elimination of catheterization, avoidance of ionizing radiation, and the potential for longitudinal monitoring without cumulative risk.

- describe alternatives and equivalents

Alternative approaches such as computational fluid dynamics based on CT data or machine learning models trained on invasive FFR datasets are possible but lack the direct physical basis and safety profile of the disclosed method.

- describe applicability of various features

The features of ECG-triggering, navigator-gating, and multi-directional velocity encoding are essential to the method’s success and may be adapted for other vascular beds such as the pulmonary or cerebral arteries.

- describe skilled artisan recognition

A skilled artisan would recognize that modifications to the sequence parameters, reconstruction algorithms, or computational models may be made without departing from the spirit of the invention, provided the core principles of phase-contrast MRI and Navier-Stokes-based pressure derivation are preserved.

- describe embodiments and modifications

Embodiments include integration with artificial intelligence for automated VOI delineation, real-time pressure mapping during acquisition, and wireless transmission of results to clinical decision-support systems. Modifications encompass the use of ultra-high-field scanners, compressed sensing, or deep learning-based velocity prediction to further enhance speed and accuracy.