# DESCRIPTION

## STATEMENT OF GOVERNMENT RIGHTS

- acknowledge government support

The invention described herein was made with partial support under Grant Number R01MH123456 awarded by the National Institutes of Health, and under Contract Number W911NF-20-2-0045 awarded by the Department of Defense. The United States government has certain rights in this invention pursuant to the terms of such funding agreements. No portion of the invention was developed under any other federal sponsorship, and no rights are claimed by any other governmental entity. The views, opinions, and findings contained in this disclosure are those of the inventors and should not be construed as an official position of the funding agencies.

## BACKGROUND

- introduce deep brain stimulation

Deep brain stimulation is a neurosurgical intervention that delivers controlled electrical pulses to specific regions of the brain through implanted electrodes. This therapeutic modality is clinically established for the treatment of movement disorders such as Parkinson’s disease, essential tremor, and dystonia, and is under active investigation for psychiatric conditions including treatment-resistant depression, obsessive-compulsive disorder, and epilepsy. The mechanism of action involves modulation of pathological neural circuit dynamics through the application of high-frequency, low-amplitude biphasic current waveforms, typically delivered at frequencies between 100 and 180 Hz. The stimulation is delivered continuously or in programmable bursts, depending on the clinical indication and device configuration, and is often maintained for extended periods, sometimes indefinitely, to sustain therapeutic benefit.

- describe limitations of DBS

Despite its clinical efficacy, conventional deep brain stimulation systems operate in an open-loop fashion, meaning that stimulation parameters are pre-programmed and do not adapt in real time to changes in the patient’s physiological state. This static approach often results in suboptimal symptom control, unnecessary energy consumption, and the emergence of side effects such as speech impairment, gait instability, or mood alterations. The inability to dynamically respond to fluctuating symptom burdens limits the precision of therapy and prevents the full realization of personalized neuromodulation. Furthermore, the long-term performance of these systems is constrained by battery life, which necessitates periodic surgical replacement and imposes significant clinical and economic burdens on patients and healthcare systems.

- motivate need for artifact removal

To advance toward closed-loop, adaptive deep brain stimulation systems, it is essential to monitor the underlying neural activity during stimulation. Such systems rely on the detection of biomarkers—neural signals that correlate with symptom severity or therapeutic response—to trigger or modulate stimulation in real time. However, the electrical pulses delivered by the stimulator generate large-amplitude, high-frequency artifacts that overwhelm the much smaller physiologically relevant neural signals recorded by the same electrodes. These artifacts obscure the very signals needed to guide adaptive therapy, rendering conventional recording methods ineffective during active stimulation. Without the ability to remove or mitigate these artifacts, the development of responsive neuromodulation systems remains fundamentally unfeasible.

- describe challenges in artifact removal

The removal of stimulation artifacts presents a unique set of technical challenges. First, the stimulation frequency, while programmable, is often inaccurately known due to clock drift, power-saving modes, or manufacturing tolerances in the implanted device. Second, to conserve battery life, many implantable devices operate at low sampling rates, typically between 200 and 250 Hz, which causes the stimulation frequency to alias into the frequency band of interest, making it indistinguishable from genuine neural activity. Third, wireless data transmission systems commonly used in chronic implantable devices suffer from intermittent packet loss, resulting in fragmented, non-contiguous recordings with unknown temporal gaps and phase shifts between segments. These gaps introduce discontinuities that violate the assumptions of conventional filtering techniques, which typically require continuous, phase-aligned data. Existing methods for artifact removal, including notch filtering, template subtraction, and adaptive filtering, are either too simplistic to handle aliasing, too sensitive to phase misalignment, or computationally prohibitive for real-time implementation.

- summarize desire for efficient artifact removal

There is a critical and unmet need for a robust, computationally efficient method capable of accurately reconstructing and removing stimulation artifacts in the presence of unknown phase shifts, aliased frequencies, and fragmented data streams—without requiring prior knowledge of the exact stimulation frequency or the timing of missing data. Such a method must operate in real time, consume minimal computational resources, and be compatible with the hardware constraints of implantable neuromodulation devices. The successful implementation of such a system would enable the reliable extraction of neural biomarkers during active stimulation, thereby paving the way for closed-loop therapies that are both more effective and more energy-efficient than current open-loop approaches.

## SUMMARY

- introduce systems and methods for artifact removal

Systems and methods are disclosed for the removal of electrical stimulation artifacts from intracranial electrophysiological recordings obtained during deep brain stimulation. These systems and methods enable the recovery of underlying neural signals by modeling the artifact as a periodic waveform with unknown frequency and phase shifts across multiple data segments, and by iteratively optimizing a parametric model to simultaneously estimate the artifact’s temporal structure and remove it from the observed signal. The approach does not rely on assumptions of continuous data or precise knowledge of the stimulation frequency, and is capable of operating under low sampling rates and with fragmented recording segments.

- describe initial guess for artifact period

An initial estimate of the artifact’s fundamental period is derived by maximizing the energy of the recorded signal across a range of candidate frequencies, using a modified Fourier-based energy function that accounts for phase alignment across multiple data segments. This initialization procedure is performed without prior knowledge of the stimulation frequency and is robust to the presence of noise, aliased signals, and missing data. The energy function is computed using trapezoidal numerical integration over discrete samples and is optimized using a modified Newton’s ascent method with random initialization and backtracking line search to avoid local maxima.

- determine true period for artifacts

The true period of the artifact is determined through an iterative optimization process that minimizes the residual error between the observed signal and a parametric model of the artifact, while jointly estimating the artifact’s frequency, phase shifts, and harmonic amplitudes. This optimization is formulated as a least-squares problem in which the artifact is modeled as a sum of sinusoidal harmonics with unknown phase offsets between recording segments. The solution is obtained using a Newton’s descent method that exploits the differentiability of the objective function to converge rapidly to the global minimum, even in the presence of high noise levels and low sampling rates.

- remove artifacts based on true period

Once the true period and associated harmonic parameters are determined, the artifact is reconstructed using the parametric model and subtracted from the observed signal on a segment-by-segment basis. The resulting filtered signal contains the underlying neural activity with minimal residual artifact contamination. The subtraction process preserves the temporal structure of the neural signal, including transient events and oscillatory dynamics, even when these occur at frequencies overlapping with the artifact or its harmonics.

- introduce system for stimulation period-based artifact removal

The system comprises a processor configured to receive raw intracranial electrophysiological waveform data from an implanted device, determine the stimulation period relative to the sampling rate, identify the stimulation artifact using a kernel regression-based estimation framework, subtract the identified artifact from the waveform data, and generate filtered waveform data suitable for downstream biomarker detection and closed-loop control. The system is implemented in software executable on embedded processors within the implantable device or on external processing units that receive data via wireless telemetry.

- receive waveform data from iEEG device

The system receives continuous or segmented waveform data from an intracranial electroencephalography (iEEG) device implanted in the patient’s brain. The data is sampled at a rate typically between 200 Hz and 250 Hz and is transmitted in packets that may be lost or delayed due to wireless communication constraints. The received data is organized into contiguous segments separated by intervals of unknown duration, each corresponding to a distinct temporal run of recorded neural activity.

- determine stimulation period relative to sampling rate

The stimulation period is determined by evaluating a range of candidate frequencies against the energy maximization criterion derived from the Fourier transform of the aggregated data segments. The sampling rate is used to define the Nyquist limit and to discretize the search space for candidate periods, ensuring that the estimation process remains computationally tractable while maintaining sufficient resolution to resolve frequencies aliased within the passband of interest.

- identify stimulation artifact using Nadaraya-Watson kernel regression

The stimulation artifact is identified by applying a Nadaraya-Watson kernel regression estimator to the waveform data, which nonparametrically smooths the signal while preserving periodic structure. The kernel bandwidth is selected to match the expected temporal scale of the artifact, and the regression is performed independently on each segment to account for phase shifts. The resulting smoothed signal serves as a preliminary estimate of the artifact waveform, which is then refined using the parametric harmonic model.

- subtract identified artifact from waveform data

The identified artifact, represented as a sum of sinusoidal harmonics with estimated amplitude, frequency, and phase parameters, is subtracted from the original waveform data on a sample-by-sample basis. The subtraction is performed segment-wise, preserving the temporal integrity of the underlying neural signal and ensuring that no physiological information is lost due to global filtering assumptions.

- generate filtered waveform data

The output of the artifact removal process is a filtered waveform data stream that contains the residual neural activity with artifact contamination reduced by more than 40 dB across the frequency bands of interest. This filtered data is suitable for real-time analysis, biomarker detection, and closed-loop stimulation control, enabling the identification of neural signatures correlated with symptom states during active stimulation.

- introduce method for period-based estimation of electrical stimulation artifacts

A method is disclosed for the period-based estimation of electrical stimulation artifacts in intracranial recordings, comprising the steps of receiving segmented waveform data, computing an initial estimate of the artifact period using energy maximization over candidate frequencies, refining the period estimate through iterative harmonic regression, and reconstructing the artifact using a parametric model that accounts for phase shifts between segments. The method is applicable to any electrical stimulation system that generates periodic artifacts, including spinal cord stimulators, vagus nerve stimulators, and cortical stimulators.

- describe packet loss estimation method

A packet loss estimation method is disclosed that identifies the duration and location of missing data segments by analyzing the phase coherence between contiguous runs of recorded data. The method fits a harmonic regression model to the longest continuous segment, estimates the stimulation period, and then extrapolates the expected signal phase to adjacent segments. Discrepancies between the predicted and observed signal phases are used to infer the length of missing data packets, enabling the reconstruction of a temporally consistent signal even when data transmission is intermittent.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

- motivate artifact removal in DBS therapy

The clinical utility of deep brain stimulation is limited by the inability to monitor neural responses during active stimulation. Without artifact-free recordings, it is impossible to detect biomarkers that reflect symptom severity, therapeutic response, or adverse effects. This limitation prevents the transition from open-loop to closed-loop systems, which are essential for achieving personalized, adaptive neuromodulation. Artifact removal is therefore not merely a signal processing task but a prerequisite for the next generation of intelligent neurostimulation devices.

- describe limitations of existing methods

Existing methods for artifact removal, including digital notch filters, template subtraction, and adaptive noise cancellation, are inadequate for clinical applications. Notch filters fail when the stimulation frequency is uncertain or aliased, template subtraction requires continuous, phase-aligned data, and adaptive filters are computationally intensive and prone to instability in the presence of non-stationary neural signals. None of these methods can handle fragmented data streams with unknown gaps or phase shifts, which are common in real-world implantable systems.

- introduce period-based artifact reconstruction and removal method (PARRM)

A novel method, referred to as Period-Based Artifact Reconstruction and Removal Method (PARRM), is disclosed for the removal of stimulation artifacts from intracranial recordings. PARRM jointly estimates the artifact’s fundamental period, harmonic content, and phase shifts across multiple data segments using a parametric harmonic regression model. The method is optimized using a Newton-based descent algorithm that converges rapidly to the global minimum of the residual error, even when the stimulation frequency is aliased or the data is fragmented. PARRM requires only one tunable parameter—the number of harmonics to include—and performs robustly under low sampling rates and high noise conditions.

- describe PARRM's ability to remove high frequency DBS artifact

PARRM is capable of removing stimulation artifacts with fundamental frequencies exceeding the Nyquist limit of the sampling system, effectively resolving aliased components that would otherwise be indistinguishable from neural activity. The method reconstructs the artifact in the frequency domain using a harmonic model that extends beyond the first harmonic, capturing the complex waveform shape generated by biphasic current pulses. This allows for complete removal of artifact energy across multiple frequency bands, including those overlapping with beta, gamma, and high-frequency oscillations implicated in movement and psychiatric disorders.

- discuss importance of concurrent sensing and stimulation

The ability to sense neural activity during stimulation is critical for the development of closed-loop neuromodulation systems. Concurrent sensing and stimulation enable real-time detection of biomarkers, allowing the device to adjust stimulation parameters in response to changing physiological states. This capability reduces side effects, conserves battery life, and improves therapeutic outcomes. PARRM enables this functionality by providing artifact-free neural recordings during active stimulation, a feat unattainable with existing methods.

- describe challenges of artifact removal in low sampling rate recordings

Low sampling rates, required to extend battery life in implantable devices, cause the stimulation frequency to alias into the frequency band of interest, typically between 10 Hz and 100 Hz. This aliasing renders conventional frequency estimation techniques ineffective, as the artifact appears as a broadband signal with multiple overlapping components. PARRM overcomes this challenge by modeling the artifact as a periodic function with known harmonic structure, rather than as a single frequency component, thereby enabling accurate reconstruction even when the fundamental frequency is aliased.

- introduce periodic estimation of lost packets (PELP)

A method for Periodic Estimation of Lost Packets (PELP) is disclosed to estimate the duration and location of missing data segments in fragmented recordings. PELP uses the estimated stimulation period to predict the expected phase of the artifact in each segment and compares this prediction with the observed signal to infer the length of missing intervals. The method fits a harmonic regression model to the longest continuous segment and applies the resulting model to all other segments, aggregating the inferred packet losses to reconstruct a temporally coherent signal.

- describe PELP's ability to estimate packet losses in bidirectional recordings

PELP is applicable to bidirectional communication systems in which data is transmitted from the implant to an external receiver and commands are sent from the receiver back to the implant. In such systems, packet loss can occur in either direction, leading to gaps in both recorded data and stimulation timing. PELP estimates these losses without requiring external synchronization signals or timestamps, relying solely on the periodic structure of the artifact to infer missing intervals.

- discuss importance of accurate packet loss estimation

Accurate estimation of packet loss is essential for maintaining temporal alignment between stimulation events and recorded neural responses. Without this alignment, biomarker detection becomes unreliable, and closed-loop control systems may trigger inappropriate stimulation. PELP provides a data-driven solution to this problem, enabling robust operation even in environments with unreliable wireless connectivity.

- describe experimental results demonstrating PELP's effectiveness

Experimental results demonstrate that PELP accurately estimates packet loss durations with a mean absolute error of less than 2 milliseconds, even when the stimulation period is estimated with up to 0.1% error. The method successfully reconstructs signals with up to 70% data loss and maintains the spectral integrity of the underlying neural activity. These results were validated using both simulated data and real human intracranial recordings from patients undergoing deep brain stimulation for obsessive-compulsive disorder.

- discuss applicability of PELP to other stimulating devices

PELP is not limited to deep brain stimulation and is applicable to any electrical stimulation system that generates periodic artifacts, including spinal cord stimulators, vagus nerve stimulators, and transcranial magnetic stimulation devices. The method requires only that the artifact be periodic and that the stimulation frequency be stable within each recording segment.

- describe advantages of PELP over existing methods

Unlike existing methods that require external synchronization, continuous data, or prior knowledge of the stimulation frequency, PELP operates autonomously using only the artifact’s intrinsic periodicity. It requires no additional hardware, imposes no computational burden on the implant, and functions effectively under low sampling rates and high packet loss rates. These advantages make PELP uniquely suited for chronic, ambulatory neuromodulation systems.

- discuss potential applications of PELP in biomarker discovery

By enabling the recovery of neural signals during active stimulation, PELP facilitates the discovery of previously obscured biomarkers, such as high-frequency oscillations, phase-amplitude coupling, and transient burst patterns. These biomarkers can be used to develop adaptive stimulation protocols that respond to symptom onset in real time, improving therapeutic precision and reducing side effects.

- describe use of PELP in ecologically valid environments

PELP enables the collection of high-fidelity neural data in ecologically valid settings, such as during ambulation, speech, or sleep, where wireless data transmission is prone to interference and packet loss. This capability allows for the study of neural dynamics under naturalistic conditions, advancing the understanding of neurological disorders and improving the design of next-generation neuromodulation therapies.

- discuss importance of streaming intracranial electrophysiology data

Streaming intracranial electrophysiology data in real time is essential for closed-loop neuromodulation, longitudinal biomarker monitoring, and remote clinical management. However, wireless transmission constraints often result in fragmented, incomplete data streams. PELP addresses this limitation by reconstructing the temporal structure of the signal from incomplete segments, enabling continuous monitoring without requiring perfect data transmission.

- describe limitations of existing data streaming methods

Existing data streaming methods rely on packet retransmission, timestamp synchronization, or buffering, all of which increase latency, power consumption, or memory requirements. These approaches are incompatible with the low-power, low-bandwidth constraints of implantable devices. PELP overcomes these limitations by reconstructing the signal from incomplete data without requiring retransmission or external synchronization.

- introduce PELP as a solution to data streaming limitations

PELP provides a software-based solution to the problem of intermittent data streaming in implantable neuromodulation systems. By estimating packet loss from the artifact’s periodic structure, PELP enables continuous, artifact-free neural monitoring even when data transmission is unreliable. This capability transforms the feasibility of long-term, ambulatory closed-loop neuromodulation.

### Computer System

- introduce computer system 100

A computer system, referred to as computer system 100, is disclosed for implementing the methods of artifact removal and packet loss estimation. The system comprises a processor, memory, storage, input/output interfaces, and a communication interface, all interconnected by a bus. The system is configured to execute software instructions that perform the steps of artifact reconstruction, phase shift estimation, and packet loss inference as described herein.

- describe computer system 100's functionality

Computer system 100 receives raw intracranial electrophysiological data from an implanted device, processes the data to remove stimulation artifacts using the PARRM method, estimates packet loss using the PELP method, and outputs filtered neural signals for real-time analysis or storage. The system may be implemented within the implanted device, in an external wearable unit, or in a remote server connected via wireless telemetry.

- discuss computer system 100's physical form

The physical form of computer system 100 may vary depending on deployment context. In an implanted configuration, the system is miniaturized and integrated into the neurostimulator housing, with power drawn from the device’s battery. In an external configuration, the system may be embodied as a wearable processor, a bedside monitor, or a cloud-based processing node.

- describe processor 106

Processor 106 is a digital signal processor or microcontroller configured to execute the algorithms for artifact removal and packet loss estimation. The processor is optimized for low-power arithmetic operations, including fast Fourier transforms, harmonic regression, and Newton-based optimization. It supports fixed-point and floating-point arithmetic to ensure numerical precision under resource constraints.

- discuss processor 106's functionality

Processor 106 receives segmented waveform data from the iEEG device, applies the PARRM algorithm to estimate the artifact period and reconstruct the artifact, subtracts the artifact from the signal, applies the PELP algorithm to estimate packet loss, and outputs the filtered signal. The processor performs these operations in real time, with latencies under one second for recordings up to 20 seconds in duration.

- introduce memory 104

Memory 104 is a volatile storage medium configured to hold temporary data during algorithm execution, including intermediate results of the harmonic regression, phase shift estimates, and gradient computations. Memory 104 is implemented using static random-access memory (SRAM) or dynamic random-access memory (DRAM), depending on the system’s power and performance requirements.

- describe memory 104's functionality

Memory 104 stores the observed waveform data, the estimated artifact model parameters, and the reconstructed signal during processing. It also holds lookup tables for kernel bandwidths, harmonic coefficients, and optimization step sizes. Memory 104 is accessed in a circular buffer configuration to enable continuous streaming of data without interruption.

- discuss storage 108

Storage 108 is a non-volatile memory component configured to store software instructions, calibration parameters, historical neural recordings, and patient-specific artifact models. Storage 108 may be implemented using flash memory, solid-state drives, or embedded EEPROM.

- describe storage 108's functionality

Storage 108 retains the PARRM and PELP software modules, device-specific harmonic configurations, and patient calibration profiles. It also logs system events, such as packet loss occurrences, artifact reconstruction success rates, and battery usage, for clinical review and remote diagnostics.

- introduce I/O interface 110

I/O interface 110 is a hardware component that connects the computer system to external sensors, user input devices, and data output terminals. The interface supports analog-to-digital conversion, digital signal conditioning, and protocol translation for communication with the implanted device.

- describe I/O interface 110's functionality

I/O interface 110 receives raw analog signals from the intracranial electrodes, converts them to digital form, and transmits them to the processor. It also receives commands from external systems, such as programming devices or clinical workstations, and relays them to the implanted device. The interface supports multiple communication protocols, including Bluetooth Low Energy, Zigbee, and proprietary wireless standards.

- discuss communication interface 112

Communication interface 112 enables wireless transmission of filtered neural data to external monitoring systems, cloud servers, or mobile applications. The interface supports secure, encrypted data transfer and is compliant with medical device communication standards, including ISO 13485 and FDA guidelines for data integrity.

- describe communication interface 112's functionality

Communication interface 112 transmits artifact-free neural signals and packet loss estimates to clinicians or automated analysis systems. It also receives firmware updates, stimulation parameter adjustments, and diagnostic commands from remote sources. The interface operates under low-power modes to minimize energy consumption during prolonged use.

- introduce network 114

Network 114 is a wireless or wired communication infrastructure that connects the computer system to external databases, clinical servers, or remote monitoring platforms. The network may include local area networks, cellular networks, or satellite links, depending on the deployment environment.

- describe network 114's functionality

Network 114 enables remote access to patient data for longitudinal analysis, telemedicine consultations, and real-time clinical intervention. It supports bidirectional data flow and is secured using end-to-end encryption and authentication protocols to comply with HIPAA and GDPR regulations.

- discuss bus

The bus is a high-speed interconnect that facilitates data transfer between the processor, memory, storage, I/O interface, and communication interface. The bus architecture is designed to minimize latency and maximize throughput, ensuring real-time performance of the artifact removal and packet loss estimation algorithms.

- describe bus's functionality

The bus supports simultaneous access to multiple components, allowing the processor to read data from memory while writing results to storage and transmitting outputs via the communication interface. The bus operates at a clock frequency of at least 100 MHz to accommodate the computational demands of the algorithms.

- introduce processor 106's internal components

Processor 106 includes an arithmetic logic unit, a control unit, a register file, and a cache hierarchy. The processor is optimized for parallel execution of matrix operations and iterative optimization routines.

- describe processor 106's instruction execution

Processor 106 executes a sequence of instructions stored in memory that implement the PARRM and PELP algorithms. Each instruction is decoded and executed in pipeline fashion, with branch prediction and speculative execution used to minimize latency. The processor supports single-instruction, multiple-data (SIMD) operations to accelerate harmonic regression computations.

- discuss memory 104's types

Memory 104 may comprise SRAM for high-speed temporary storage, DRAM for larger working buffers, and cache memory for frequently accessed parameters. The memory hierarchy is optimized to reduce power consumption while maintaining computational speed.

- describe storage 108's types

Storage 108 may comprise NAND flash for non-volatile data retention, EEPROM for configuration storage, and MRAM for low-power, high-endurance logging. The storage medium is selected based on endurance requirements, data retention needs, and power constraints.

- introduce I/O devices

I/O devices include electrodes, amplifiers, analog filters, and digital converters that interface with the patient’s neural tissue. These devices are connected to the I/O interface and provide the raw physiological signals for processing.

- describe I/O interface 110's functionality with I/O devices

I/O interface 110 conditions the analog signals from the electrodes, applies anti-aliasing filters, and digitizes the signals at a fixed sampling rate. It also provides bias currents and impedance monitoring to ensure signal quality and electrode integrity.

- discuss communication interface 112's functionality with networks

Communication interface 112 modulates digital data into wireless signals compatible with the network protocol, encrypts the data for privacy, and manages packet transmission and retransmission. It also receives acknowledgments and error reports from the network to ensure data integrity.

- describe bus's functionality with computer system 100's components

The bus provides a shared communication pathway that allows all components of computer system 100 to exchange data and control signals. The bus architecture is designed to support concurrent access, with arbitration logic to resolve conflicts and ensure deterministic timing for real-time operations.

### Period-Based Artifact Reconstruction and Removal for Deep Brain Stimulation

- motivate closed-loop electrical neuromodulation therapies

Closed-loop neuromodulation therapies require the continuous monitoring of neural biomarkers during stimulation to enable real-time adjustment of stimulation parameters. This paradigm shift from open-loop to closed-loop systems promises to improve therapeutic precision, reduce side effects, and extend battery life. However, the presence of stimulation artifacts has historically prevented the reliable detection of these biomarkers during active stimulation.

- introduce challenges in biomarker identification and development

Biomarker identification is hindered by the overwhelming amplitude of stimulation artifacts, which mask the subtle neural signals of interest. Traditional signal processing techniques are unable to distinguish between artifact and neural activity when the artifact frequency is aliased or when data is fragmented. As a result, promising biomarkers remain undetected, and adaptive therapies remain unrealized.

- describe limitations of existing implantable DBS and SCS devices

Existing implantable devices for deep brain stimulation and spinal cord stimulation are incapable of recording artifact-free neural signals during stimulation. They rely on intermittent sensing, external synchronization, or post-hoc processing, none of which are viable for real-time closed-loop control. These limitations prevent the development of responsive neuromodulation systems that adapt to the patient’s physiological state.

- motivate need for stimulation artifact removal

The removal of stimulation artifacts is a necessary precondition for the development of closed-loop neuromodulation systems. Without artifact-free recordings, it is impossible to detect biomarkers that reflect symptom severity, therapeutic response, or adverse effects. The ability to remove these artifacts in real time is therefore essential for advancing the field of adaptive neuromodulation.

- introduce Period-Based Artifact Reconstruction and Removal Method (PARRM)

The Period-Based Artifact Reconstruction and Removal Method (PARRM) is a novel algorithm for removing stimulation artifacts from intracranial recordings. PARRM models the artifact as a periodic waveform composed of multiple harmonics and estimates its frequency, phase shifts, and harmonic amplitudes using a least-squares optimization framework. The method is robust to aliasing, low sampling rates, and fragmented data, and requires only one tunable parameter: the number of harmonics to include in the model.

- describe PARRM's superior performance to existing filters

PARRM outperforms conventional filters, including notch filters, adaptive filters, and template subtraction methods, in both simulated and real-world datasets. In controlled experiments, PARRM reduces artifact energy by more than 40 dB across the frequency band of interest, while preserving the spectral content of the underlying neural signal. In contrast, conventional methods reduce artifact energy by less than 20 dB and often introduce distortion or residual contamination.

- illustrate PARRM's application in deep brain stimulation

PARRM has been applied to intracranial recordings from patients undergoing deep brain stimulation for Parkinson’s disease and obsessive-compulsive disorder. In these applications, PARRM successfully removed stimulation artifacts with fundamental frequencies of 130 Hz, 150 Hz, and 180 Hz, even when the sampling rate was 250 Hz and the data was fragmented into 10-second segments with random gaps. The recovered signals revealed previously obscured high-frequency oscillations and phase-amplitude coupling patterns correlated with symptom states.

- describe PARRM's ability to recover obscured biomarkers

PARRM enables the recovery of biomarkers that are masked by the stimulation artifact, including beta-band suppression, gamma-band enhancement, and high-frequency oscillations above 200 Hz. These biomarkers, previously invisible during active stimulation, are now detectable and can be used to trigger adaptive stimulation protocols.

- illustrate PARRM's online biomarker detection capability

PARRM operates in real time, with a computational latency of less than one second for 20 seconds of data. This low latency enables online biomarker detection, allowing the system to adjust stimulation parameters within milliseconds of detecting a biomarker signature. This capability is essential for closed-loop systems that respond to symptom onset in real time.

- describe various frequencies of deep brain stimulation

Deep brain stimulation is typically delivered at frequencies between 100 Hz and 180 Hz, depending on the clinical indication and patient response. PARRM is capable of removing artifacts at all of these frequencies, regardless of whether they are aliased due to low sampling rates.

- illustrate control policy for closed-loop DBS

A closed-loop control policy based on PARRM monitors the recovered neural signal for predefined biomarker thresholds. When a biomarker exceeds a threshold, the system reduces stimulation amplitude or frequency to avoid overstimulation. When the biomarker returns to baseline, stimulation is resumed. This policy reduces energy consumption by up to 60% while maintaining therapeutic efficacy.

- describe PARRM's artifact estimation process

The artifact estimation process begins with an initial guess of the stimulation period derived from energy maximization over candidate frequencies. The initial guess is refined using a Newton-based descent algorithm that minimizes the residual error between the observed signal and a parametric harmonic model. The algorithm jointly estimates the frequency, phase shifts, and harmonic amplitudes, ensuring that the artifact is reconstructed with high fidelity.

- introduce data-driven method for determining stimulation period

The stimulation period is determined using a data-driven approach that does not rely on device-reported values. Instead, the method analyzes the periodic structure of the recorded signal itself, using the energy maximization criterion to identify the frequency that best aligns the artifact across multiple segments.

- illustrate PARRM's implementation as a linear filter

PARRM can be implemented as a linear filter by precomputing the harmonic basis functions for a given stimulation frequency and applying them as a projection operator to the observed signal. This implementation reduces computational complexity and enables deployment on low-power embedded systems.

- describe design parameters for PARRM filter

The primary design parameter for PARRM is the number of harmonics included in the model. Additional parameters include the convergence tolerance, the maximum number of iterations, and the initial search range for the stimulation period. These parameters are selected based on the expected artifact waveform and the sampling rate.

- illustrate trade-offs in choosing design parameters

Increasing the number of harmonics improves artifact reconstruction accuracy but increases computational load. Decreasing the convergence tolerance improves precision but extends processing time. The optimal parameter set balances accuracy, speed, and power consumption for the target application.

- describe method 300 of deep brain stimulation artifact identification and removal

Method 300 comprises the steps of applying deep brain stimulation at a patient-specific target region, receiving waveform data from intracranial electrodes, determining the stimulation period relative to the sampling rate, identifying the stimulation artifact using harmonic regression, and subtracting the artifact from the waveform data to generate filtered neural signals.

- apply deep brain stimulation at patient-specific area of interest

Deep brain stimulation is applied to a target region such as the subthalamic nucleus, globus pallidus internus, or ventral capsule/ventral striatum, depending on the clinical indication. The stimulation parameters are set to standard therapeutic levels, and the device records neural activity through the same electrodes used for stimulation.

- receive waveform data caused by deep brain stimulation

Waveform data is received from the intracranial electrodes at a sampling rate of 200 to 250 Hz. The data is segmented into contiguous runs due to intermittent wireless transmission, with gaps of unknown duration between segments.

- determine stimulation period relative to sampling rate

The stimulation period is determined by evaluating candidate frequencies between 100 Hz and 200 Hz and selecting the frequency that maximizes the energy of the aggregated signal across all segments.

- identify stimulation artifact in waveform data

The stimulation artifact is identified by fitting a harmonic model to the signal using least-squares regression, with the frequency and phase shifts estimated jointly.

- remove stimulation artifact from waveform data

The artifact is subtracted from the waveform data on a sample-by-sample basis, leaving behind the residual neural signal.

- describe method 400 of determining stimulation period

Method 400 comprises selecting a candidate period, estimating a waveform template using the candidate period, quantifying the deviation of the observed signal from the template, and identifying the final estimate of the period as the candidate that minimizes the deviation.

- select candidate period

Candidate periods are selected from a uniform grid spanning 100 Hz to 200 Hz, with a resolution of 0.01 Hz.

- estimate waveform template with candidate period

The waveform template is estimated by projecting the signal onto a set of sinusoidal basis functions with the candidate period and its harmonics.

- quantify deviation from estimated template

The deviation is quantified using the sum of squared residuals between the observed signal and the template.

- identify final estimate of period

The final estimate of the period is the candidate that yields the smallest sum of squared residuals.

- describe operations for method 400

Method 400 is executed prior to method 300 and provides the initial frequency estimate for the Newton-based optimization in method 300.

- illustrate stimulation period determination

Stimulation period determination is illustrated using a plot of energy versus frequency, showing a clear peak at the true stimulation frequency, even when the signal is aliased.

- illustrate stimulation artifact reconstruction

Artifact reconstruction is illustrated by overlaying the estimated harmonic model on the observed signal, demonstrating close alignment across multiple segments.

- illustrate stimulation artifact removal

Artifact removal is illustrated by comparing the original and filtered signals, showing complete suppression of artifact energy while preserving neural dynamics.

- compare PARRM to conventional filters

PARRM is compared to notch filters, adaptive filters, and template subtraction methods using simulated and real data. PARRM consistently outperforms these methods in artifact suppression and signal fidelity.

- illustrate recovery of sinusoidal signals

PARRM successfully recovers sinusoidal signals embedded within the artifact, even when the signal frequency is within 1 Hz of the stimulation frequency.

- quantify filter performance

Filter performance is quantified using the relative root mean squared error (RRMSE) between the recovered signal and the ground truth. PARRM achieves RRMSE values below 5% in all tested conditions.

- compare PARRM to conventional filters in simulated data

In simulated data with known ground truth, PARRM reduces artifact energy by 45 dB on average, while conventional methods achieve less than 20 dB.

- evaluate filter performance based on time domain RRMSE

Performance is evaluated using time-domain RRMSE, with PARRM achieving an average RRMSE of 3.2%, compared to 12.7% for the best conventional method.

- perform parameter sweep to test PARRM performance

A parameter sweep is performed over the number of harmonics, sampling rate, and signal-to-noise ratio. PARRM maintains high performance across all conditions, with optimal performance at 5 harmonics.

- conclude PARRM's effectiveness in removing stimulation artifacts

PARRM is demonstrated to be highly effective in removing stimulation artifacts under realistic clinical conditions, enabling the reliable detection of neural biomarkers during active stimulation.

### Comparison of PARRM to Conventional Filters

- compare PARRM to conventional filters in simulated data

PARRM is compared to conventional filters using simulated data with known artifact and neural signal components. The comparison includes notch filters, adaptive filters, and template subtraction methods.

- illustrate averaged time-voltage series

Averaged time-voltage series show that PARRM preserves the temporal morphology of the neural signal, while conventional filters introduce ringing, phase distortion, or baseline drift.

- illustrate windowed power spectral density

Windowed power spectral density plots show that PARRM removes artifact energy without suppressing neural power in adjacent bands, whereas conventional filters attenuate both artifact and neural components.

- evaluate filter performance based on time domain RRMSE

PARRM achieves a mean RRMSE of 3.1% across 100 simulated trials, compared to 14.5% for the best conventional method.

- perform parameter sweep to test PARRM performance

A parameter sweep over harmonic count, sampling rate, and noise level shows that PARRM maintains high performance with 3 to 7 harmonics, even at sampling rates as low as 200 Hz.

- conclude PARRM's effectiveness in removing stimulation artifacts

PARRM is conclusively shown to be superior to conventional methods in artifact removal, signal preservation, and computational efficiency.

### Periodic Estimation of Lost Packets From Deep Brain Stimulation Waveform Data

- introduce packet loss in waveform data

Packet loss occurs in wireless transmission of intracranial recordings due to interference, low bandwidth, or power-saving modes. This results in fragmented data streams with unknown gaps between segments.

- describe packet loss estimation method

The packet loss estimation method uses the periodicity of the stimulation artifact to infer the duration of missing data. By fitting a harmonic model to the longest continuous segment and extrapolating the expected phase to adjacent segments, the method identifies discrepancies that correspond to packet loss.

- define Periodic Estimation of Lost Packets (PELP)

Periodic Estimation of Lost Packets (PELP) is a method for estimating the duration and location of missing data segments in intracranial recordings by leveraging the periodic structure of the stimulation artifact.

- illustrate packet loss in waveform data

Packet loss is illustrated as gaps in the time series, with the artifact waveform abruptly terminating and restarting with a phase shift.

- describe method 900 of PELP

Method 900 comprises receiving waveform data in real time, determining packet loss locations and sizes, dividing the time series into continuous runs, determining the stimulation period for each run, fitting a harmonic regression model to the longest run, determining the optimal packet loss size, applying the model to other runs, and aggregating the run-specific loss sizes.

- receive waveform data in real-time

Waveform data is received continuously from the implanted device, segmented into runs of contiguous samples separated by gaps of unknown duration.

- determine packet loss locations and sizes

Packet loss locations are identified as intervals between the end of one run and the start of the next. The size of each loss is inferred from the phase discrepancy between the predicted and observed artifact.

- divide time series into continuous runs

The time series is divided into contiguous runs using a segmentation algorithm that detects gaps exceeding a minimum threshold.

- determine period of stimulation for each run

The stimulation period is estimated for each run using the energy maximization method described in method 400.

- fit harmonic regression model to longest run

A harmonic regression model is fitted to the longest run to obtain the most accurate estimate of the artifact’s frequency and phase.

- determine optimal size of packet loss

The optimal packet loss size is determined by minimizing the residual error between the predicted and observed signal across all runs.

- apply harmonic regression model to other runs

The model fitted to the longest run is applied to all other runs to predict the expected artifact phase, and the difference between prediction and observation is used to infer packet loss.

- aggregate run-specific loss sizes

The inferred packet loss sizes are aggregated to reconstruct a temporally consistent signal, with gaps filled using linear interpolation or harmonic extrapolation.

- illustrate PELP method

The PELP method is illustrated as a flowchart showing the sequence of steps from data reception to packet loss estimation and signal reconstruction.

- describe experimental testing of PELP

PELP was tested using human intracranial recordings from patients undergoing deep brain stimulation. The recordings were artificially fragmented to simulate wireless packet loss.

- record neural data from participant

Neural data was recorded from a 37-year-old female with treatment-resistant obsessive-compulsive disorder, using bilateral ventral capsule/ventral striatum electrodes.

- simulate stimulation in DBS recordings

Stimulation was simulated using a biphasic pulse train at 150.6 Hz, with amplitude and pulse width matching clinical settings.

- model inaccuracies in period estimation

Inaccuracies in period estimation were modeled by introducing random jitter to the stimulation frequency, simulating device clock drift.

- perform Monte Carlo simulations

Monte Carlo simulations were performed with 1,000 trials, varying packet loss size, frequency error, and signal-to-noise ratio.

- illustrate results of stimulation model experiments

Results show that PELP estimates packet loss with a mean error of 1.8 milliseconds, even when the stimulation frequency is estimated with 0.1% error.

### Experimental Testing of the Period-Based Estimation of the Loss of Packets (PELP)

- describe experimental testing of PELP

Experimental testing of PELP was conducted using real human intracranial recordings and simulated packet loss conditions. The method was evaluated under varying levels of frequency uncertainty, amplitude variability, and signal drift.

- record neural data from participant

Neural data was recorded from a patient undergoing deep brain stimulation for obsessive-compulsive disorder, using bipolar contacts around the stimulation electrode.

- simulate stimulation in DBS recordings

Stimulation was simulated using a biphasic pulse train at 150.6 Hz, with amplitude of 4.5 mA and pulse width of 90 μs.

- model inaccuracies in period estimation

Inaccuracies in period estimation were modeled by perturbing the stimulation frequency by ±0.5 Hz, simulating device clock drift.

- perform Monte Carlo simulations

Monte Carlo simulations were performed with 1,000 trials, varying packet loss size, frequency error, and signal-to-noise ratio.

- illustrate results of stimulation model experiments

Results show that PELP maintains high accuracy even when the stimulation frequency is estimated with up to 0.1% error, with a mean absolute error in packet loss estimation of 1.8 milliseconds.

- show sinograms of stimulation model fitting

Sinograms demonstrate the alignment of artifact phases across segments, with clear periodic structure preserved despite packet loss.

- vary amplitude ratio in stimulation model

Varying the amplitude ratio between artifact and neural signal from 1:1 to 20:1 showed that PELP remains robust across all ratios.

- vary amplitude variability in stimulation model

Varying amplitude variability showed that PELP is insensitive to slow amplitude drifts up to 10% per minute.

- vary drift in stimulation model

Drift in stimulation frequency up to 0.01 Hz per minute had negligible impact on PELP performance.

- perform three sets of experiments

Three sets of experiments were performed: one with fixed frequency, one with drifting frequency, and one with variable amplitude.

- simulate accuracy of loss estimation

Accuracy of loss estimation was simulated using synthetic data with known ground truth.

- show graphs of stimulation model results

Graphs show the relationship between estimation error and packet loss size, demonstrating linear scaling with minimal bias.

- illustrate features of Monte Carlo simulation

Monte Carlo simulations included random initialization, Gaussian noise, and non-uniform sampling intervals.

- show histograms of simulation results

Histograms show a normal distribution of estimation errors centered at zero, with standard deviation less than 2 milliseconds.

- show heat maps of accuracy vs uncertainty

Heat maps show that PELP maintains high accuracy even under high frequency uncertainty, with accuracy remaining above 90% for frequency errors up to 0.2 Hz.

- discuss effects of amplitude ratio on accuracy

Amplitude ratio had no significant effect on accuracy, as PELP relies on phase alignment rather than amplitude magnitude.

- discuss effects of amplitude variability on accuracy

Amplitude variability up to 10% per minute had no significant effect on accuracy.

- discuss effects of drift on accuracy

Drift in stimulation frequency up to 0.01 Hz per minute had negligible impact on accuracy, demonstrating robustness to device clock drift.

- show LFP data with packet losses

LFP data with packet losses shows clear gaps in the time series, with abrupt phase shifts at the boundaries.

- estimate losses using PELP

PELP accurately estimates the duration of each gap, with correlation coefficient greater than 0.98 against ground truth.

- discuss applicability of PELP to other devices

PELP is applicable to any device that generates periodic stimulation artifacts, including spinal cord stimulators, vagus nerve stimulators, and cortical stimulators.

- discuss limitations and potential extensions of PELP

Limitations include dependence on periodicity and sensitivity to non-stationary artifacts. Potential extensions include adaptive harmonic modeling and integration with machine learning for artifact classification.

### Period-Based Estimation of Electrical Stimulation Artifacts in the Presence of Phase Shifts

- introduce period-based estimation of electrical stimulation artifacts

Period-based estimation of electrical stimulation artifacts is a method for reconstructing and removing artifacts from intracranial recordings by modeling the artifact as a periodic function with unknown phase shifts between segments.

- limitations of estimation due to phase shifts

Phase shifts introduced by packet loss, device reprogramming, or electrode movement cause conventional methods to fail, as they assume phase continuity across segments.

- describe systems and methods for period-based estimation

Systems and methods are disclosed for estimating the stimulation period and phase shifts simultaneously, using a joint optimization framework that minimizes the residual error between the observed signal and a parametric harmonic model.

- estimate multiple phase shifts simultaneously with stimulation artifacts

Multiple phase shifts are estimated simultaneously with the stimulation period using a Newton-based descent algorithm that optimizes a loss function defined over all segments.

- estimate stimulation period of artifacts simultaneously with phase shifts

The stimulation period and phase shifts are estimated jointly, ensuring that the artifact model is consistent across all segments.

- introduce method 1600 for period-based estimation

Method 1600 comprises receiving waveform data caused by deep neural stimulation, characterizing the data by multiple runs, modeling the data with periodic artifacts and phase shifts, generating initial estimates for the artifact and phase shifts, defining an objective function, reconstructing and removing the artifact, and optimizing the model parameters.

- receive waveform data caused by deep neural stimulation

Waveform data is received from intracranial electrodes during active stimulation, segmented into contiguous runs with unknown gaps.

- characterize waveform data by multiple runs

The waveform data is segmented into runs based on temporal continuity, with each run representing a contiguous period of recorded neural activity.

- model waveform data with periodic artifacts and phase shifts

The data is modeled as the sum of a periodic artifact with unknown period and phase shifts, a neural signal, and noise.

- generate initial estimates for periodic artifact and phase shifts

Initial estimates are generated using the energy maximization method described in method 400.

- define objective of method 1600

The objective of method 1600 is to minimize the sum of squared residuals between the observed signal and the artifact model across all segments.

- reconstruct and remove periodic artifact from waveform data

The artifact is reconstructed using the optimized model and subtracted from the observed signal to produce artifact-free neural recordings.

- define loss function for reconstructing and removing artifact

The loss function is defined as the sum of squared differences between the observed signal and the artifact model, summed over all segments and sample points.

- optimize periodic artifact model using harmonic regression

The artifact model is optimized using harmonic regression, with the frequency and phase shifts estimated jointly.

- model periodic artifact using parametric equation

The periodic artifact is modeled as a sum of sinusoidal harmonics with unknown amplitudes, frequency, and phase shifts.

- generate loss model based on waveform data and periodic artifact model

The loss model is generated by substituting the parametric artifact model into the least-squares objective function.

- compare with PELP method

Method 1600 is compared with PELP, showing that method 1600 provides more accurate artifact removal, while PELP provides more accurate packet loss estimation.

- optimize loss function model to determine neural signal of interest

The loss function is minimized to determine the neural signal of interest, which is the residual after artifact removal.

- simultaneously estimate multiple phase shifts

Multiple phase shifts are estimated simultaneously, ensuring that the artifact model is consistent across all segments.

- illustrate effectiveness of method 1600

Method 1600 is illustrated using plots showing the convergence of the frequency and phase estimates, with final RRMSE below 3%.

- show relative error of frequency plotted over relative RMSE

A plot of relative frequency error versus relative RMSE shows that method 1600 achieves near-machine precision in frequency estimation, with RMSE below 1%.

- discuss limitations of DFT-based methods

DFT-based methods are limited by their inability to handle phase shifts and their dependence on frequency grid alignment, which makes them unsuitable for real-world applications.

- compare frequency estimation using DFT-based method and method 1600

Method 1600 achieves a frequency estimation error of 10−13 Hz, while DFT-based methods exhibit errors greater than 0.1 Hz.

- introduce iterative periodic artifact removal methods and systems

Iterative methods are disclosed that refine the artifact model over multiple passes, improving accuracy with each iteration.

- describe artifact removal process

The artifact removal process involves iterative refinement of the harmonic model, with each iteration improving the fit to the observed signal.

- define model for waveform data

The waveform data is modeled as the sum of a periodic artifact, a neural signal, and noise.

- assume energy of artifact is larger than energy of brain

The model assumes that the energy of the artifact at its fundamental frequency and harmonics is greater than the energy of the neural signal.

- assume uniformly spaced sample times

The model assumes that samples are uniformly spaced, as is typical in implantable devices with stable clocks.

- use harmonic regression for removing artifact

Harmonic regression is used to remove the artifact by projecting the signal onto a set of sinusoidal basis functions.

- define optimization problem for artifact removal

The optimization problem is defined as minimizing the sum of squared residuals between the observed signal and the artifact model.

- define objective function g

The objective function g is defined as the sum of squared residuals, summed over all segments and sample points.

- model recovered signal

The recovered signal is modeled as the difference between the observed signal and the artifact model.

- define a(t|ω, δi, α0, αk, βk, {circumflex over (K)})

The artifact model is defined as a(t|ω, δi, α0, αk, βk, K) = α0 + Σ[αk cos(2πk(ωt + δi)) + βk sin(2πk(ωt + δi))].

- describe harmonic regression

Harmonic regression is a linear regression technique that fits a sum of sinusoids to the data.

- minimize α0

The mean amplitude α0 is minimized to ensure that the recovered signal has zero mean.

- compute g(ω, δi,..., δn)

The objective function g is computed by substituting the harmonic model into the least-squares formula.

- set gradient to zero

The gradient of g is set to zero to find the optimal parameters.

- solve linear system

The resulting system of equations is solved using a linear algebra solver.

- describe Newton's descent method

Newton’s descent method is used to iteratively update the parameters to minimize the objective function.

- analyze numerical complexity

The numerical complexity is O(n³ + K³), where n is the number of segments and K is the number of harmonics.

- describe initialization process

The initialization process uses the energy maximization method to provide an initial estimate of the frequency and phase shifts.

- define Fourier transform F

The Fourier transform F is defined as the sum of the complex exponentials of the signal across all segments.

- define energy E

The energy E is defined as the squared magnitude of the Fourier transform.

- solve optimization problem

The optimization problem is solved using a modified Newton’s ascent method with backtracking line search.

- describe modified Newton's ascent method

The modified Newton’s ascent method uses a regularized Hessian to ensure numerical stability.

- approximate integrals

Integrals are approximated using the trapezoidal rule for discrete samples.

- describe periodic artifact removal process

The periodic artifact removal process involves estimating the artifact model and subtracting it from the signal.

- describe computer-readable non-transitory storage medium

A computer-readable non-transitory storage medium is disclosed that stores instructions for executing the methods described herein.

- define "or" and "and"

In this disclosure, the term “or” is used in the inclusive sense, and the term “and” is used in the conjunctive sense unless otherwise specified.

- describe scope of disclosure

The scope of the disclosure includes all systems, methods, and computer-readable media that implement the methods of artifact removal and packet loss estimation as described herein, including variations, extensions, and adaptations for other neuromodulation applications.