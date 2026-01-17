# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a noninvasive super-continuum infrared spectroscopy of cytochrome c oxidase (SCISCCO) system for monitoring cerebral, tissue, or organ metabolism and hemodynamics. More specifically, the invention pertains to a portable, ruggedized, and high-sensitivity system that can be used in various medical settings, including emergency departments, operating rooms, and intensive care units, to monitor changes in oxygenated hemoglobin (HbO), deoxygenated hemoglobin (HbR), and the redox state of cytochrome c oxidase (CCO).

## BACKGROUND

Monitoring cerebral, tissue, or organ metabolism and hemodynamics is crucial for diagnosing and managing various medical conditions, such as concussions, stroke, traumatic brain injury (TBI), and hemorrhagic shock (HS). Traditional methods, such as functional magnetic resonance imaging (fMRI), provide valuable insights but are limited by their high cost, poor portability, and inability to offer continuous monitoring. Noninvasive optical techniques, such as functional near-infrared spectroscopy (fNIRS), have shown promise but suffer from low signal-to-noise ratios (SNR) and limited depth penetration.

Cytochrome c oxidase (CCO) is a key enzyme in the mitochondrial electron transport chain, and its redox state can serve as a sensitive marker of cellular metabolism. However, measuring CCO noninvasively is challenging due to its low concentration and overlapping spectra with hemoglobin. High-brightness light sources, such as supercontinuum lasers (SCL), can significantly improve the SNR and enable the simultaneous measurement of HbO, HbR, and CCO.

There is a critical need for a practical, low-cost, noninvasive method to monitor cerebral and tissue metabolism in real-time, especially in emergency and critical care settings. The present invention addresses this need by providing a high-sensitivity, portable SCISCCO system that can be used to monitor changes in HbO, HbR, and CCO simultaneously.

## SUMMARY OF THE INVENTION

The present invention provides a noninvasive super-continuum infrared spectroscopy of cytochrome c oxidase (SCISCCO) system for monitoring cerebral, tissue, or organ metabolism and hemodynamics. The system includes:

1. **Light Source**: An all-fiber integrated modulational instability-initiated supercontinuum laser (SCL) that outputs up to 11 W of time-averaged power covering the near-infrared (NIR) and short-wave infrared (SWIR) wavelength range of 670-2500 nm. The SCL provides an almost order of magnitude higher brightness compared to conventional light sources, enhancing the signal-to-noise ratio (SNR) for CCO measurements.

2. **Optical System**: A two-arm differential setup with a reference arm and a sample arm. The system includes a tunable spectrometer, a polarizer, and a beam splitter to separate the light into the reference and sample arms. Spatial apertures are used to match the beam size in both arms.

3. **Detection System**: Matched pairs of silicon photodetectors in the reference and sample arms, connected to lock-in amplifiers synchronized to a chopper. The lock-in amplifiers improve the SNR by blocking out ambient light and other spurious signals.

4. **Data Processing**: A computer running custom software with a graphical user interface (GUI) for inputting measurement parameters and displaying the metabolic and hemodynamic data in real-time. The software uses a modified Beer-Lambert law-based algorithm to process the measured spectra and calculate the concentrations of HbO, HbR, and CCO.

5. **Cart-Based Prototype**: A portable, ruggedized cart system that integrates the hardware and software into a single, transportable unit. The cart has three levels: the top level houses the SCL and laptop for processing, the middle level contains the optical system hardware, and the bottom level holds the electronics and lock-in amplifiers.

The SCISCCO system is designed to provide high-sensitivity, noninvasive measurements of cerebral and tissue metabolism and hemodynamics. It can be used in various medical settings, including emergency departments, operating rooms, and intensive care units, to monitor changes in HbO, HbR, and CCO. The system's portability and robust design make it suitable for use in remote locations and for transport within hospitals.

## DETAILED DESCRIPTION

### Light Source

The light source of the SCISCCO system is an all-fiber integrated modulational instability-initiated supercontinuum laser (SCL). The SCL operates over the NIR and SWIR wavelength range and can provide up to 11 W of time-averaged power. The SCL is based on a master oscillator power amplifier (MOPA) configuration, where a seed laser outputs ~0.5-2 ns wide pulses at an adjustable repetition rate from 100 kHz to 4 MHz. The pulses are then amplified through a multi-stage fiber amplifier, and the amplified pulses are coupled into two relatively short lengths of fiber for supercontinuum generation. The first length of fiber is a few meters of standard single-mode fiber (SMF), where modulational instability occurs, converting the nanosecond pulses into a series of shorter pulses. The second length of fiber is a nonlinear SC fiber, where the spectral broadening results from physical processes such as four-wave mixing, self-phase modulation, and the optical Raman effect.

### Optical System

The optical system of the SCISCCO system includes a tunable spectrometer, a polarizer, and a beam splitter to separate the light into a reference arm and a sample arm. The spectrometer is used to select the particular wavelength for testing, and the polarizer ensures a single polarization of light to avoid noise effects. A broadband beam splitter (50:50 ratio) is used to split the light beam, and spatial apertures are used to match the beam size in both arms. The sample arm is directed to the target tissue, and the reference arm is used to divide out any laser or environmental fluctuations.

### Detection System

The detection system consists of matched pairs of silicon photodetectors in the reference and sample arms. The detector outputs are sent to lock-in amplifiers synchronized to a chopper at 271 Hz. The lock-in amplifiers improve the SNR by blocking out ambient light and other spurious signals. The electronic output from the lock-in amplifiers is sent to a computer for data collection and processing.

### Data Processing

The data processing is handled by a computer running custom software with a graphical user interface (GUI). The software allows the user to input measurement parameters and view the metabolic and hemodynamic data in real-time. The software uses a modified Beer-Lambert law-based algorithm to process the measured spectra and calculate the concentrations of HbO, HbR, and CCO. The algorithm attributes the intensity changes across the wavelength range to the absorption changes of the three chromophores and uses a least-squares method to back-calculate the concentrations.

### Cart-Based Prototype

The SCISCCO system is integrated into a portable, ruggedized cart system that can be easily transported to different locations. The cart has three levels:
- **Top Level**: Houses the SCL and laptop for processing.
- **Middle Level**: Contains the optical system hardware.
- **Bottom Level**: Holds the electronics and lock-in amplifiers.

The cart system is designed to be stable and reliable, even when transported over rough terrain. The components are mechanically secured to ensure the system's performance remains consistent. The cart-based prototype has been tested for stability by rolling it through buildings with brick flooring and driving it in a van over local paved and dirt roads.

### Empirical Examples

#### Laboratory Measurements

The SCISCCO system was validated through laboratory measurements of CCO solutions. The optical spectra of CCO were measured, and the results were compared with published literature. The measured spectra showed good agreement, confirming the system's ability to accurately measure CCO.

#### Human Testing

The SCISCCO system was used in pilot human studies to validate its performance in vivo. The system was tested in three sets of experiments:
- **Blood Pressure Test**: The system measured changes in HbO, HbR, and CCO in the forearm during a blood pressure test. The results were consistent with expectations based on physiological principles.
- **Breath-Holding Test**: The system measured changes in HbO, HbR, and CCO in the forehead during a breath-holding test. The results were compared with a commercial fNIRS system, and the correlation was reasonable.
- **Cognitive Attention Test**: A pilot human study involving 25 participants was conducted using a cognitive attention test. The results showed that during the attention task, the level of HbO increased and the level of oxidized CCO decreased, consistent with prior studies.

#### Animal Testing

The cart-based SCISCCO prototype was used in swine animal trials to monitor changes in HbO, HbR, and CCO during hemorrhagic shock (HS) and resuscitation using partial resuscitative endovascular balloon occlusion of the aorta (pREBOA) with and without valproic acid (VPA). The data collected by the SCISCCO system were validated against other instruments, such as those used to measure arterial blood gases and mean arterial pressure. The results showed that the measured HbR and HbO concentration changes were consistent with expectations based on physiological principles, and the CCO concentration changes provided new insights into the swine metabolic reactions during the different phases of HS and treatment.

### Conclusion

The SCISCCO system provides a noninvasive, high-sensitivity tool for monitoring changes in cerebral and tissue metabolism and hemodynamics. The system's use of an all-fiber integrated SCL as the key enabling technology, combined with a portable and robust design, makes it a practical and cost-effective solution for various medical applications. The system can be used in emergency departments, operating rooms, and intensive care units to monitor changes in HbO, HbR, and CCO, providing valuable insights for diagnosing and managing medical conditions such as concussions, stroke, TBI, and HS.