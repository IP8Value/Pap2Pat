# DESCRIPTION

## FIELD OF THE INVENTION

- introduce field of invention  
The present invention relates to the field of invasive hemodynamic monitoring in clinical settings, particularly to systems and methods for ensuring the fidelity of arterial pressure measurements obtained through fluid-filled catheter systems. These systems are routinely employed in intensive care units, operating rooms, and emergency departments to provide continuous, real-time assessment of arterial blood pressure and derived physiological indices such as pulse pressure variation, dP/dt max, and pulse wave velocity. The invention addresses a persistent and clinically significant limitation inherent in conventional pressure transduction systems, namely the distortion of the arterial pressure waveform due to the dynamic mechanical properties of the hydraulic connection between the arterial catheter and the pressure transducer. This distortion, often undetected by clinical staff, leads to systematic errors in the interpretation of hemodynamic status and can result in inappropriate therapeutic interventions, including misjudged fluid administration, vasopressor dosing, or inotropic support.

## SUMMARY

- introduce arterial pressure measurement  
Arterial pressure measurement via fluid-filled catheter systems remains a cornerstone of critical care monitoring, providing continuous, beat-to-beat data essential for managing hemodynamically unstable patients. The system relies on a sterile, fluid-filled tubing that transmits pressure oscillations from the arterial site to a transducer, which converts mechanical pressure into an electrical signal for display and analysis.

- describe current practice  
Current clinical practice involves the direct connection of an indwelling arterial catheter to a disposable pressure transducer via a length of non-compliant tubing, typically filled with saline and maintained under continuous flush. The transducer is calibrated at atmospheric pressure, and the resulting waveform is displayed on a monitor without further signal processing. Clinical staff rely on visual inspection of the waveform shape—particularly the presence of a sharp systolic upstroke and a distinct dicrotic notch—as indicators of signal integrity.

- describe advantages of current practice  
The primary advantages of this approach lie in its simplicity, low cost, and real-time responsiveness. It requires no complex electronics at the patient bedside, allows for immediate waveform visualization, and integrates seamlessly into existing monitoring infrastructure without requiring changes to clinical workflow.

- describe limitations of current practice  
However, the system is highly susceptible to distortion caused by the physical characteristics of the hydraulic connection, including tubing length, diameter, compliance, air bubbles, clot formation, and kinking. These factors alter the damping and natural frequency of the system, leading to underdamped or overdamped waveforms that misrepresent true arterial pressure. These distortions are frequently overlooked because they do not trigger alarms and often appear visually plausible to untrained observers.

- introduce hydraulic connection distortions  
Hydraulic connection distortions arise from the inertial, resistive, and elastic properties of the fluid-filled pathway, which act as a low-pass filter with a characteristic transfer function. When the natural frequency of the system is too low relative to the frequency content of the arterial pulse, the waveform becomes overdamped, suppressing systolic peaks and elevating diastolic values. When the natural frequency is too high, the system becomes underdamped, producing exaggerated oscillations and false peaks.

- describe transfer function of hydraulic connection  
The transfer function of the hydraulic connection is governed by the mass of the fluid column, the resistance to flow within the tubing, and the compliance of the system walls. This function determines how faithfully the incident arterial pressure waveform is reproduced at the transducer. Deviations from the ideal second-order system response result in phase shifts, amplitude attenuation, and frequency-dependent distortion.

- describe distortions caused by hydraulic connection  
These distortions manifest as systematic errors in systolic, diastolic, and mean arterial pressure values, as well as in derived indices such as pulse pressure variation and the maximum rate of pressure rise (dP/dt). In overdamped conditions, systolic pressure is underestimated, diastolic pressure is overestimated, and pulse pressure is compressed. In underdamped conditions, systolic pressure is falsely elevated, and oscillatory artifacts mimic pathological waveforms.

- describe impact on derived indices  
The impact on derived indices is clinically significant. For instance, dP/dt, used as an estimate of left ventricular contractility, may be artificially inflated by up to 24% due to underdamping. Pulse pressure variation, a key predictor of fluid responsiveness, may show false positive or negative values depending on the nature of the distortion, leading to inappropriate volume resuscitation or avoidance of necessary fluid administration.

- describe clinical staff detection of defects  
Clinical staff rarely detect these distortions reliably. Visual assessment is subjective and error-prone, and no standardized protocol exists for routine verification of hydraulic integrity. Fast flush tests, when performed, are infrequent and often misinterpreted. As a result, distorted signals persist for hours or even days without correction.

- describe proposed solutions  
Proposed solutions have included automated damping analysis based on fast flush response, frequency domain analysis of the waveform, and algorithmic correction filters. However, these approaches are either reactive, require manual initiation, or lack real-time adaptability to changing conditions.

- describe limitations of proposed solutions  
Existing solutions fail to continuously monitor the hydraulic system’s dynamic state, do not account for time-varying changes in tubing compliance or obstruction, and do not actively correct the signal in real time. They also require specialized equipment or operator intervention, limiting their adoption in routine clinical practice.

- introduce need for improvement  
There exists a critical and unmet need for a system that continuously, automatically, and non-invasively assesses the integrity of the hydraulic connection, detects distortions as they occur, and actively corrects or compensates for them without disrupting clinical workflow.

- describe aim of invention  
The aim of the invention is to provide a device and method for the continuous, real-time monitoring and dynamic correction of arterial pressure signals to ensure their physiological accuracy, regardless of changes in the hydraulic connection’s mechanical properties.

- describe method for improving quality of measured signal  
The method comprises the continuous determination of the dynamic parameters of the hydraulic connection, the frequential analysis of the incident arterial pressure signal, and the detection of the system’s operational state based on the congruence between the measured signal and the expected physiological waveform. When a distortion is detected, a mechanical actuation is applied to the tubing to excite the system and re-characterize its transfer function, enabling real-time correction or alert generation.

- describe device for improving quality of measured signal  
The device comprises a monitoring and correcting module integrated into the pressure transduction pathway, an actuator module capable of applying controlled mechanical perturbations to the tubing, and a remote processing system that performs signal analysis, derives clinical indices, and generates alerts.

- describe steps of method  
The method involves the continuous acquisition of the arterial pressure signal, the characterization of the hydraulic connection’s damping and natural frequency, the spectral analysis of the incident waveform, the detection of deviations from expected physiological patterns, and the application of corrective mechanical actions or alerts when necessary.

- determine dynamic parameters of hydraulic connection  
Dynamic parameters are determined by applying brief, controlled mechanical perturbations to the tubing and analyzing the system’s response in both time and frequency domains.

- analyse frequential content of incident signal  
The frequential content of the incident signal is analyzed using fast Fourier transform techniques to identify the energy distribution across the physiological frequency band, typically between 0.5 Hz and 15 Hz.

- detect situation of hydraulic connection  
Three distinct situations are identified: (a) optimal transmission with damping coefficient within physiological range and natural frequency above 12 Hz; (b) overdamped transmission with damping coefficient exceeding 0.7 and natural frequency below 8 Hz; and (c) underdamped transmission with damping coefficient below 0.3 and natural frequency exceeding 20 Hz.

- describe situation (a)  
In situation (a), the hydraulic connection reproduces the arterial waveform with minimal distortion, and no corrective action is required.

- describe situation (b)  
In situation (b), systolic pressure is attenuated, diastolic pressure is elevated, and pulse pressure is reduced. The system is in an overdamped state, and corrective measures include recalibration or alerting clinical staff.

- describe situation (c)  
In situation (c), the waveform exhibits excessive oscillations, false systolic peaks, and elevated dP/dt values. The system is underdamped, and mechanical stabilization or tubing replacement is indicated.

- describe mechanical action on tubing  
Mechanical action is applied via a piezoelectric or electromagnetic actuator that induces controlled micro-vibrations or pressure pulses along the tubing wall.

- describe synchronisation of mechanical action  
Mechanical action is synchronized with diastole to avoid interference with systolic pressure peaks and to ensure that the excitation does not corrupt the clinical signal.

- describe alternative mechanical actions  
Alternative mechanical actions include localized compression, torsional twisting, or acoustic stimulation using ultrasonic pulses.

- describe detection of decoupling  
Decoupling is detected by analyzing the phase lag and amplitude attenuation between the radial and aortic pressure waveforms, using pulse wave velocity estimation derived from the Moens-Korteweg equation and comparing it to population-based reference values.

- describe applanation tonometry  
Applanation tonometry is not employed in this system; instead, the invention operates on standard fluid-filled catheter systems without requiring modification to the transducer or catheter.

- describe pulse wave velocity propagation  
Pulse wave velocity is estimated from the time delay between the R-wave on the ECG and the foot of the arterial pressure waveform, corrected for the distance from the heart to the measurement site.

- describe comparison with reference value  
The estimated pulse wave velocity is compared to a reference value derived from age, height, and blood pressure norms. A significant deviation indicates decoupling or distal obstruction.

- describe alarm in case of decoupling  
An alarm is triggered if the pulse wave velocity falls below 5 m/s or exceeds 12 m/s, indicating either excessive damping or a disconnection in the circuit.

- describe device for continuous monitoring and correction  
The device enables continuous, autonomous monitoring and correction, operating in three modes: active, passive, and neutral, depending on the detected state of the hydraulic connection.

- describe monitoring and correcting module  
The monitoring and correcting module contains two processors: one dedicated to signal analysis and one dedicated to control and actuation coordination.

- describe actuator module  
The actuator module delivers precise mechanical stimuli to the tubing and is powered by a rechargeable battery, allowing for wireless operation and integration into existing monitoring chains.

- describe integration into conventional measuring chain  
The device integrates seamlessly into conventional pressure monitoring chains via standard connectors, requiring no modification to transducers, catheters, or monitors.

- describe advantages of device  
Advantages include real-time correction, zero additional workload for clinical staff, compatibility with existing equipment, and automatic logging of waveform integrity events.

- describe ease of installation  
Installation requires only the insertion of the monitoring module between the catheter and transducer, with no calibration or setup beyond standard flush procedures.

- describe transparency for hospital staff  
The device operates transparently; no changes to display, alarms, or workflow are required, and all corrections occur silently in the background.

- describe neutralisation of device  
In neutral mode, the device monitors without intervening, preserving signal fidelity while minimizing interference.

- describe first processor  
The first processor performs high-speed signal acquisition, Fourier analysis, and estimation of damping and natural frequency.

- describe second processor  
The second processor evaluates the system state, determines corrective actions, and coordinates actuator activation.

- describe redundant treatments  
Redundant signal validation is performed using both time-domain and frequency-domain methods to ensure robustness against noise and artifacts.

- describe actuator module  
The actuator module comprises a compact, sterilizable actuator, a low-power energy source, and a wireless communication interface.

- describe wired connection  
A wired connection is available for high-fidelity data transmission in environments where wireless signals are restricted.

- describe wireless connection  
Wireless communication enables remote monitoring and integration with hospital information systems.

- describe rechargeable battery  
A rechargeable lithium-polymer battery provides continuous operation for up to 72 hours.

- describe casing for monitoring module  
The casing is made of medical-grade polymer, designed for single-use or sterilizable reuse, and is compatible with standard disinfection protocols.

- describe actuator module connection  
The actuator module connects to the monitoring module via a flexible, shielded cable that minimizes electromagnetic interference.

- describe processing system  
The remote processing system receives corrected signals, calculates derived indices, displays waveforms, and logs all detected anomalies.

- describe communication with monitoring module  
Communication occurs via encrypted Bluetooth or Wi-Fi, ensuring data security and compatibility with hospital networks.

- describe applanation tonometry casing  
Applanation tonometry is not part of this invention; the system is designed exclusively for fluid-filled catheter systems and does not require skin contact or optical sensors.

## DETAILED DESCRIPTION

- introduce monitoring device for measuring arterial pressure  
The monitoring device for measuring arterial pressure comprises three integrated components: a monitoring and correcting module, an actuator module, and a remote processing system, all designed to operate in concert to ensure the fidelity of invasive arterial pressure measurements.

- describe device components: monitoring and correcting module, actuator module, remote processing system  
The monitoring and correcting module is positioned inline between the arterial catheter and the pressure transducer. The actuator module is affixed externally to the tubing adjacent to the monitoring module. The remote processing system resides on a bedside monitor or hospital server and receives processed signals for display and analysis.

- describe monitoring and correcting module functions  
The monitoring and correcting module continuously acquires the raw arterial pressure signal, performs real-time frequential analysis, estimates the damping coefficient and natural frequency of the hydraulic system, and determines whether the system is operating in an optimal, overdamped, or underdamped state.

- characterise dynamic parameters of hydraulic connection  
Dynamic parameters are characterized by applying a brief, low-amplitude mechanical impulse to the tubing during diastole and analyzing the resulting transient response using a second-order system model.

- perform frequential analysis of incident arterial pressure signal  
Frequential analysis is performed using a sliding-window fast Fourier transform with a 5-second window, sampled at 200 Hz, to capture the full spectral content of the arterial waveform.

- estimate aptitude of hydraulic connection for reproducing incident signal  
The aptitude is estimated by comparing the energy distribution in the 2–10 Hz band to a reference template derived from validated aortic pressure waveforms.

- detect measuring artefacts and incidents  
Artifacts such as air bubbles, clot formation, or tubing kinks are detected by sudden shifts in damping coefficient, loss of high-frequency energy, or phase anomalies in the waveform.

- provide information on hydraulic connection state and corrective action  
The system provides real-time feedback to the remote processing system, indicating the state of the hydraulic connection and recommending corrective actions such as flushing, repositioning, or replacement.

- describe three operating modes: active, passive, neutral  
In active mode, the system applies corrective mechanical actions to restore signal fidelity. In passive mode, it monitors and alerts but does not act. In neutral mode, it suspends all monitoring and actuation, preserving signal integrity without intervention.

- describe architecture of monitoring and correcting module  
The module contains a dual-processor architecture: a first processor handles signal acquisition and spectral analysis, while a second processor manages decision logic, actuator control, and communication.

- describe two processors: first processor for signal processing, second processor for correction  
The first processor executes algorithms for damping estimation and frequency domain analysis. The second processor evaluates the results, determines the system state, and triggers actuation or alerts.

- describe communication between processors and with actuator module and remote processing system  
Communication occurs via a high-speed internal bus and encrypted wireless link, ensuring low-latency response and data integrity.

- describe power management module  
A power management module optimizes energy consumption by activating processors only during signal acquisition windows and placing components in low-power sleep mode between measurements.

- describe energy efficiency of monitoring and correcting module  
The module consumes less than 50 mW during continuous operation, enabling extended battery life and compatibility with portable monitoring platforms.

- describe actuator module functions  
The actuator module applies controlled mechanical perturbations to the tubing to excite the hydraulic system, enabling dynamic characterization without interrupting pressure measurement.

- apply mechanical action to tubing to excite hydraulic connection  
Mechanical action is delivered via a piezoelectric element that generates micro-vibrations of 10–50 μm amplitude at frequencies between 5 and 25 Hz.

- describe pulse and sinusoidal loads for characterising hydraulic connection  
Pulse loads are used for rapid damping estimation, while sinusoidal loads are employed for precise determination of natural frequency and resonance behavior.

- synchronise mechanical action with diastole  
Mechanical excitation is triggered 100–300 ms after the R-wave on the ECG, ensuring that perturbations occur during diastole and do not interfere with systolic pressure peaks.

- describe actuator module components: actuator, power source  
The actuator is a miniaturized piezoelectric stack, and the power source is a rechargeable lithium-polymer battery housed in a sealed, sterilizable casing.

- describe remote processing system functions  
The remote processing system receives corrected pressure signals, calculates derived indices such as PPV and dP/dt, displays waveforms, logs incidents, and generates audible and visual alarms when thresholds are exceeded.

- perform quality follow-up and calibration  
The system performs automated quality follow-up by comparing current dynamic parameters to baseline values established during initial setup and recalibrating if drift exceeds 15%.

- describe methods for processing signal  
Signal processing employs a combination of time-domain and frequency-domain techniques, including fast flush analysis, harmonic excitation, and spectral coherence analysis.

- determine dynamic parameters of hydraulic connection  
Dynamic parameters are determined by fitting the system’s impulse response to a second-order differential equation and extracting damping ratio and natural frequency.

- describe percussion method using fast flushes  
The percussion method involves analyzing the transient response following a rapid flush, measuring the number of oscillations and decay time to estimate damping.

- extract response to fast flush from haemodynamic signal  
The response is isolated using template matching and temporal gating to exclude cardiac cycle artifacts.

- calculate damping factor and natural frequency  
Damping factor is calculated as the logarithmic decrement of successive peaks, and natural frequency is derived from the oscillation period.

- describe limitations of percussion method  
The percussion method is limited by its reliance on infrequent flush events and its susceptibility to noise during the flush transient.

- describe harmonic method  
The harmonic method applies a brief series of sinusoidal mechanical stimuli across a frequency sweep and measures the system’s amplitude and phase response.

- apply brief series of sinusoidal loads  
Sinusoidal loads of 1–5 seconds duration are applied at frequencies from 5 to 20 Hz in 1 Hz increments.

- describe advantages of harmonic method  
The harmonic method provides higher precision in estimating natural frequency and is less susceptible to noise than the percussion method.

- describe limitations of harmonic method  
The harmonic method requires longer acquisition time and may be confounded by non-linearities in the tubing material.

- describe combination of percussion and harmonic methods  
The combination uses percussion for rapid initial assessment and harmonic for periodic refinement, achieving both speed and accuracy.

- describe advantages of combination  
The combination provides robust, continuous monitoring with minimal interference and high diagnostic accuracy.

- describe limitations of combination  
The combination requires more computational resources and careful synchronization to avoid signal contamination.

- describe other methods for processing signal  
Other methods include wavelet decomposition, cross-correlation with reference waveforms, and machine learning classification of waveform morphology.

- describe implementation of monitoring device  
The device is implemented as a modular, plug-and-play unit that integrates into existing pressure monitoring chains without modification.

- describe casing for monitoring and correcting module  
The casing is made of transparent, biocompatible polymer with integrated fluid ports and electromagnetic shielding.

- describe connectors for interconnecting module to existing measuring device  
Connectors are standardized Luer-lock fittings compatible with all commercial arterial catheters and transducers.

- describe simplified man-machine interface  
The interface consists of a single LED indicator and a Bluetooth pairing button, minimizing user interaction.

- describe sterilisation of monitoring and correcting module  
The module is designed for autoclaving, ethylene oxide sterilization, or single-use disposal, depending on institutional policy.

- describe actuator module implementation  
The actuator module is a self-contained, wireless unit with a magnetic mount for secure attachment to the tubing.

- describe advantages of actuator module implementation  
Its wireless design eliminates cable clutter, reduces infection risk, and allows placement at any point along the tubing.

- describe remote processing system implementation  
The remote processing system is implemented as software running on existing bedside monitors or hospital servers, with a dedicated module for signal correction and alert generation.

- introduce percussion method  
The percussion method is employed as a rapid, intermittent diagnostic tool to assess damping characteristics during flush events.

- describe actuator module  
The actuator module is capable of delivering both percussion-like impulses and harmonic stimuli, enabling dual-mode characterization.

- motivate harmonic method  
The harmonic method is motivated by the need for continuous, high-resolution characterization of the hydraulic system’s transfer function.

- describe harmonic analysis  
Harmonic analysis involves applying a frequency sweep and computing the system’s frequency response function using the ratio of output to input spectra.

- derive formulas for z and f0  
The damping ratio z is derived from the bandwidth at half-power, and the natural frequency f0 is determined from the peak of the amplitude response curve.

- describe need for segmenting mechanical action  
Segmentation is required to isolate the actuator-induced response from the physiological signal and prevent signal contamination.

- motivate analysis of signal spectrum  
Spectral analysis is motivated by the fact that physiological pressure waveforms have a characteristic frequency signature that is altered by hydraulic distortion.

- describe principle of signal spectrum analysis  
The principle is that a healthy hydraulic connection preserves the high-frequency components of the arterial waveform; their attenuation indicates system degradation.

- motivate determination of frequential content  
Determining frequential content allows the system to distinguish between true physiological changes and measurement artifacts.

- describe FFT analysis  
Fast Fourier transform analysis is applied to 5-second segments of the pressure signal to compute the power spectral density.

- estimate aptitude of hydraulic connection  
Aptitude is estimated by comparing the spectral energy in the 5–12 Hz band to a reference template derived from validated aortic waveforms.

- describe first method for estimation  
The first method uses the ratio of systolic peak energy to diastolic trough energy as an index of damping.

- describe second method for estimation  
The second method computes the coherence between the pressure signal and the simultaneously recorded ECG R-wave, with low coherence indicating signal distortion.

- illustrate second method  
In a healthy system, coherence exceeds 0.85 in the 2–10 Hz band; in an overdamped system, it drops below 0.6.

- motivate monitoring of hydraulic connection  
Continuous monitoring is motivated by the high frequency of undetected distortions and their direct impact on clinical decision-making.

- describe detection of artefacts  
Artifacts are detected by deviations in spectral energy, phase lag, or waveform morphology from established physiological norms.

- describe detection of abnormal attenuation  
Abnormal attenuation is detected when the amplitude of the systolic peak falls below 80% of the predicted value based on mean arterial pressure and heart rate.

- describe correction and alert  
Correction is performed by applying mechanical stabilization; if correction fails, an alert is generated.

- describe logging of incidents  
All detected incidents, including time, type, duration, and corrective action, are logged in a secure, timestamped database.

- motivate detection of decoupling  
Detection of decoupling is motivated by the risk of false-negative fluid responsiveness predictions and misinterpretation of cardiac output.

- describe detection module  
The detection module compares the radial pressure waveform to a model of aortic pressure derived from the Moens-Korteweg equation and ECG timing.

- illustrate variation in aortic and radial pressures  
In a healthy system, the aortic and radial waveforms are similar in shape and timing; in decoupled systems, the radial waveform shows delayed and attenuated features.

- describe frequential analysis method  
Frequential analysis identifies the loss of high-frequency harmonics in the radial waveform, indicating distal signal degradation.

- illustrate spectral content of aortic and radial pressures  
The aortic waveform contains significant energy above 10 Hz; the radial waveform loses this energy when the hydraulic connection is compromised.

- describe measurement of pulse wave velocity  
Pulse wave velocity is measured as the distance from the aortic valve to the radial artery divided by the time delay between the R-wave and the foot of the radial pressure wave.

- describe Moens-Korteweg equation  
The Moens-Korteweg equation relates pulse wave velocity to arterial wall stiffness and blood density, providing a physiological reference for expected velocity.

- illustrate decrease in pulse wave velocity  
A decrease in pulse wave velocity below 5 m/s indicates excessive damping or obstruction; an increase above 12 m/s suggests disconnection.

- describe distribution of arterial pressure  
The distribution of arterial pressure along the vascular tree is modeled as a damped wave propagation system, with attenuation increasing with distance and system impedance.

- illustrate mean pulse wave velocity  
Mean pulse wave velocity in healthy adults is 6–8 m/s; deviations outside this range trigger diagnostic flags.

- describe principle of second detection method  
The second detection method uses the time delay between the R-wave and the systolic upstroke to estimate pulse wave velocity and compare it to a normative model.

- describe measurement of time delay  
Time delay is measured from the R-wave peak to the foot of the pressure waveform using a derivative-based algorithm.

- describe estimation of pulse wave velocity  
Pulse wave velocity is estimated as the distance from the heart to the catheter tip divided by the measured time delay.

- describe variant using brachial and carotid measurements  
A variant of the method uses brachial and carotid measurements to estimate central pulse wave velocity, enhancing accuracy in patients with peripheral vascular disease.

- describe transfer function for aortic pressure  
The transfer function from aortic to radial pressure is modeled as a second-order low-pass filter with known parameters based on vascular anatomy.

- describe operation of tonometric casing  
The invention does not employ tonometric casing; it is designed exclusively for fluid-filled catheter systems.

- describe monitoring of radial arterial pressure  
Radial arterial pressure is monitored continuously, with all signal processing performed on the raw waveform without external reference.

- describe warning generation  
Warnings are generated when damping exceeds 0.7, natural frequency falls below 8 Hz, or pulse wave velocity deviates by more than 20% from predicted values.

- describe input of distances  
The system allows manual input of the distance from the heart to the catheter tip, which is used to calibrate pulse wave velocity calculations.

- describe data fusion device  
A data fusion device combines information from the hydraulic characterization, pulse wave velocity, and waveform morphology to generate a composite integrity score.

- conclude invention  
This invention provides a novel, automated, and clinically practical solution to the persistent problem of hydraulic distortion in invasive arterial pressure monitoring, ensuring accurate, reliable, and actionable hemodynamic data in critical care environments.