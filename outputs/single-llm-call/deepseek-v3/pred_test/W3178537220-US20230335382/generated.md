Here is the patent application following your outline and research paper:

# DESCRIPTION  

## FIELD  

The present invention relates to systems and methods for measuring plasma characteristics in semiconductor manufacturing environments. More specifically, the invention concerns a radio emission spectroscopy (RES) technique for real-time monitoring of plasma parameters including power variations, pressure variations, chamber wall cleanliness, and frequency mixing phenomena in single and multiple frequency plasma chambers. The disclosed RES system provides non-invasive, contact-free measurement capabilities unaffected by viewport clouding conditions that typically impair optical emission spectroscopy (OES) techniques.  

## BACKGROUND  

Plasma systems play a critical role in semiconductor device manufacturing, particularly in etching and deposition processes where precise control of plasma parameters directly impacts process reproducibility and device yield. Traditional plasma diagnostic techniques rely on invasive probes or optical methods that suffer from significant limitations. Optical emission spectroscopy (OES), while widely adopted for plasma monitoring, becomes ineffective when viewports become clouded with process byproducts, a common occurrence in semiconductor fabrication environments.  

The RES technique addresses these limitations by employing radio frequency detection of plasma emissions through near-field magnetic loop antennas. Unlike OES, RES measurements remain unaffected by viewport transparency conditions, as demonstrated by comparative measurements through optically transparent and completely opaque viewports showing identical signal characteristics. Prior attempts at radio frequency plasma monitoring have been described in PCT/EP2018/057556, which discloses basic principles of RES detection but fails to address practical implementation challenges in semiconductor manufacturing environments.  

Mandelis et al. previously described radio frequency plasma monitoring techniques but their approach focused on fundamental physics research rather than industrial process monitoring applications. The Mandelis system lacks the sensitivity, real-time monitoring capabilities, and robustness required for semiconductor manufacturing processes. Prior art RES systems generally suffer from inadequate signal processing architectures, suboptimal antenna placement configurations, and insufficient sensitivity to detect subtle process variations critical for semiconductor fabrication.  

The disclosed RES system comprises several key components including a near-field magnetic loop antenna positioned proximate to the plasma chamber viewport, high-speed radio frequency signal acquisition hardware, and advanced spectral analysis algorithms. The system operates by detecting electromagnetic emissions from the plasma across a broad frequency spectrum, with particular sensitivity to fundamental drive frequencies and their harmonics. Signal processing occurs at data rates up to 133 kHz, enabling real-time monitoring with millisecond-scale temporal resolution.  

Plasma chambers suitable for RES monitoring include both single frequency systems, such as the Oxford Instruments PlasmaLab 100 operating at 13.56 MHz, and multiple frequency configurations like the Lam EXELAN 2300 system employing combined 2 MHz, 27 MHz and 162 MHz drive frequencies. The RES technique provides particular benefits in multiple frequency chambers where it can detect nonlinear mixing phenomena between different drive frequencies, offering insights into sheath dynamics and bulk plasma characteristics.  

The development of the present RES system was motivated by the need for robust, real-time plasma monitoring unaffected by chamber conditions that impair optical techniques. By providing sensitive, non-invasive measurement of critical plasma parameters, the invention enables improved process control, chamber condition monitoring, and early fault detection in semiconductor manufacturing applications.  

## SUMMARY  

The present invention provides a radio emission spectroscopy (RES) system for measuring plasma characteristics comprising a near-field magnetic loop antenna positioned within 1 mm of a plasma chamber viewport, a high-speed radio frequency signal acquisition module, and spectral analysis processing hardware. The system detects electromagnetic emissions from plasmas across a frequency spectrum encompassing fundamental drive frequencies and their harmonics, with particular sensitivity to variations in plasma current density.  

Operation of the RES system involves continuous acquisition of radio frequency signals induced in the loop antenna by plasma currents, followed by real-time spectral analysis using Fast Fourier Transform techniques. The system achieves data analysis rates up to 133 kHz, enabling detection of plasma parameter variations with millisecond-scale temporal resolution. Signal processing focuses on amplitude variations at fundamental drive frequencies, which correlate strongly with conduction current density in the plasma bulk.  

Key benefits of the RES system include sensitivity to power variations as small as 5W (0.4% resolution) and pressure changes below 1 mTorr (0.1% resolution). The system demonstrates particular utility in monitoring chamber wall cleanliness through detection of capacitive coupling variations caused by contaminant films. In multiple frequency plasma chambers, the RES system uniquely detects frequency mixing phenomena arising from nonlinear sheath dynamics, providing insights into both bulk plasma and sheath characteristics.  

The invention further encompasses a method for measuring plasma characteristics comprising the steps of positioning a magnetic loop antenna proximate to a plasma chamber viewport, acquiring radio frequency signals induced by plasma currents, performing spectral analysis of acquired signals, and correlating spectral features with plasma parameters. The method provides real-time monitoring of power, pressure, and chamber wall conditions with sensitivity superior to conventional optical techniques.  

A system implementation includes a custom-designed near-field antenna, high-speed digitizer, and embedded signal processing unit configured for industrial plasma tool integration. The system architecture supports both standalone operation and integration with factory automation systems through standardized communication interfaces.  

The invention further provides a computer-readable medium containing instructions for implementing the RES analysis method, including algorithms for real-time spectral processing, parameter extraction, and fault detection. The software component enables customization of analysis parameters for specific plasma processes and chamber configurations.  

## DETAILED DESCRIPTION OF THE DRAWINGS  

The RES system architecture comprises a loop antenna positioned at the plasma chamber viewport connected to high-speed signal acquisition electronics. The antenna design features a 21.6 mm diameter loop optimized for near-field magnetic detection at plasma drive frequencies from 2 MHz to 162 MHz. Antenna placement approximately halfway between chamber electrodes ensures sensitivity to bulk plasma currents rather than localized sheath phenomena.  

Plasma chamber components relevant to RES monitoring include powered electrodes, grounded surfaces, and viewport configurations. In a typical capacitively coupled system like the Oxford Instruments PlasmaLab 100, the RES antenna detects currents driven by the 13.56 MHz RF power supply through the plasma bulk. The Lam EXELAN 2300 system presents additional complexity with multiple powered electrodes operating at different frequencies, producing characteristic frequency mixing signatures detectable by the RES system.  

The custom sensor design provides several benefits including immunity to viewport clouding, minimal plasma perturbation, and linear response to plasma current density variations. The sensor's proximity to the plasma (1 mm from viewport) ensures strong signal coupling while its compact size prevents significant loading of the plasma system.  

The Oxford Instruments PlasmaLab 100 system operates with single-frequency 13.56 MHz excitation, producing RES signals dominated by the fundamental frequency and its harmonics. In contrast, the Lam EXELAN 2300 system demonstrates complex frequency mixing behavior when operating with combined 2 MHz and 162 MHz drive frequencies, generating sidebands spaced at 2 MHz intervals around the 162 MHz carrier.  

RES signal analysis involves both amplitude and frequency domain examination. Power variations manifest primarily as changes in fundamental frequency amplitude, while pressure variations affect both amplitude and harmonic content. Chamber wall contamination produces characteristic changes in signal amplitude correlated with contaminant film thickness through capacitive coupling mechanisms.  

Frequency mixing components arise particularly in multiple-powered electrode systems where nonlinear plasma sheath behavior acts as a natural diode mixer. This phenomenon produces heterodyne sidebands at frequency offsets corresponding to differences between drive frequencies, providing information about sheath dynamics and bulk plasma properties.  

The RES system finds application in monitoring various plasma characteristics including stray capacitance variations, chamber condition changes, and sheath property modifications. Compared to optical techniques, RES provides superior sensitivity to these parameters while remaining unaffected by viewport conditions that typically impair OES measurements.  

## 1. Real-Time Monitoring of Power Variations in the Process Chamber Using RES  

The RES technique demonstrates exceptional sensitivity to plasma power variations through detection of conduction current density changes in the plasma bulk. Experimental measurements using an oxygen plasma at 100 mTorr pressure show a ~10 dB variation in RES signal amplitude across a 50W to 500W power range, corresponding to a ten-fold change in signal intensity.  

The experimental setup employed an Oxford Instruments PlasmaLab 100 system with 13.56 MHz excitation, oxygen gas flow at 50 sccm, and pressure maintained at 100 mTorr. The RES loop antenna, positioned 1 mm from the plasma viewport, detected signals at the fundamental 13.56 MHz frequency while applied electrode power was varied from 50W to 500W.  

Results indicate the RES technique can detect power changes as small as 5W with less than 0.4% error, representing approximately 3.5% signal change per watt in the 50-150W range. This sensitivity arises from the proportional relationship between induced antenna voltage and plasma current density, which itself varies linearly with applied RF power under constant pressure conditions.  

The conduction current density in the plasma bulk, which the RES technique primarily monitors, depends fundamentally on electron density and collision frequency according to J_c ∝ n_e/ν_m. Measurements confirm that over the 50-500W power range at constant pressure, electron density changes dominate conduction current variations, with n_e increasing from ~7×10^15 m^-3 at 50W to ~6×10^16 m^-3 at 500W while collision frequency changes only ~50%.  

Real-time monitoring capability was demonstrated by stepping RF power during continuous plasma operation while recording RES signals at 19 kHz analysis rates. The system clearly resolved instantaneous power changes, confirming its suitability for dynamic process monitoring in semiconductor manufacturing environments.  

## 2. Real-Time Monitoring of Pressure Variations in the Process Chamber Using RES  

Pressure variations in plasma chambers produce detectable changes in RES signals through modifications to both electron density and collision frequency. Experiments using oxygen plasma at 200W power demonstrated RES sensitivity to pressure changes below 1 mTorr, with ~2.5% signal variation per mTorr in the 10-25 mTorr range.  

The experimental configuration maintained constant 200W power and 50 sccm oxygen flow while varying pressure from 10 mTorr to 250 mTorr. RES signals collected at the 13.56 MHz fundamental frequency showed characteristic variations correlated with pressure-dependent changes in plasma parameters.  

Analysis reveals that pressure changes affect RES signals through modifications to both electron density (n_e) and electron temperature (T_e). At 200W, n_e varies from ~2.75×10^15 m^-3 at 10 mTorr to ~2.5×10^16 m^-3 at 200 mTorr, while T_e decreases from 4.5 eV to 0.8 eV over the same range. These changes produce a net increase in conduction current density from 0.1 A at 10 mTorr to 0.59 A at 200 mTorr, faithfully tracked by the RES signal amplitude.  

The technique's pressure sensitivity stems from its ability to detect subtle changes in collision frequency (ν_m) which varies more significantly with pressure than with power. At constant power, increased pressure leads to higher neutral density and consequently more frequent electron-neutral collisions, modifying both current density and plasma impedance characteristics detectable by the RES system.  

Real-time pressure monitoring was demonstrated through continuous RES signal acquisition during dynamic pressure changes. The system resolved sub-mTorr variations with <0.1% error, confirming its utility for process control applications requiring precise pressure management.  

## 3. Real-Time Monitoring of Chamber Wall Cleanliness Using RES  

Chamber wall cleanliness significantly impacts plasma process reproducibility in semiconductor manufacturing. The RES technique detects wall contamination through changes in capacitive coupling between the plasma and chamber walls, providing real-time monitoring of cleaning processes.  

Experiments simulated wall contamination by applying photoresist to aluminum foil placed on the chamber wall of an Oxford Instruments PlasmaLab 100 system. Oxygen plasma at 500W and 50 mTorr was maintained while continuously monitoring RES signals at 13.56 MHz with 133 kHz analysis rates.  

Results showed clear distinction between clean and contaminated wall conditions, with RES signal amplitude increasing asymptotically toward the clean-wall value as the photoresist was removed by plasma cleaning. The approximately 1.5% wall coverage by contaminant produced measurable signal changes, demonstrating the technique's sensitivity to minor chamber wall condition variations.  

A qualitative model explains this behavior through effective capacitance changes caused by dielectric contaminant films. The total plasma-to-wall impedance Z_eff decreases with contaminant thickness t_f according to Z_eff = 1/(jω(C_w + εA/t_f)), where C_w represents the clean-wall capacitance and ε the film permittivity. As cleaning progresses and t_f→0, the impedance approaches the clean-wall value, with corresponding increases in displacement current and RES signal amplitude.  

This application of RES provides semiconductor manufacturers with a valuable tool for endpoint detection during chamber cleaning processes and real-time monitoring of wall condition between runs, addressing a critical challenge in plasma process reproducibility.  

## 4. Use of RES to Monitor Plasmas in a Multiple Frequency Chamber  

Multiple frequency plasma chambers present unique monitoring challenges due to nonlinear interactions between different drive frequencies. The RES technique detects these interactions through characteristic frequency mixing signatures, providing insights into both bulk plasma and sheath dynamics.  

Experiments using a Lam EXELAN 2300 system with combined 2 MHz and 162 MHz drive powers demonstrated clear heterodyning effects in RES spectra. Sidebands appeared at 2 MHz intervals around the 162 MHz carrier frequency, corresponding to nonlinear mixing between the two drive frequencies in the plasma sheaths.  

A simplified diode mixer model explains these observations, where the nonlinear current-voltage characteristics of plasma sheaths act similarly to semiconductor diodes, producing sum and difference frequencies when multiple RF signals are present. The small-signal analysis shows that sideband amplitude depends on electron temperature through the thermal voltage term V_t = kT_e/e, suggesting potential for electron temperature monitoring through RES sideband analysis.  

These frequency mixing phenomena provide valuable information about sheath dynamics and bulk plasma properties in multiple frequency systems. The RES technique uniquely captures these effects through non-invasive radio frequency detection, offering advantages over conventional probes that may perturb the delicate balance between different frequency components.  

## 5. Use of RES to Remotely Monitor Changes in Stray Capacitance, Chamber Conditions or Changes in the Sheath Characteristics of a Plasma  

The RES system enables remote monitoring of various plasma characteristics including stray capacitance variations, chamber condition changes, and sheath property modifications. These applications leverage the technique's sensitivity to both amplitude and frequency domain perturbations in plasma emissions.  

Power variation experiments demonstrate the system's ability to track changes in sheath characteristics through both RES amplitude variations and subtle frequency shifts. The amplitude changes correlate strongly with conduction current density in the plasma bulk, while frequency shifts may indicate modifications to sheath capacitance or plasma impedance.  

Frequency compensation characteristics of the RES system allow discrimination between true plasma parameter changes and external RF system variations. This capability proves particularly valuable for monitoring stray capacitance changes that may indicate chamber component degradation or process drift.  

Compared to optical emission spectroscopy, the RES technique provides superior sensitivity to these capacitive and impedance-related phenomena while remaining unaffected by viewport clouding. The contact-free nature of the measurement prevents plasma perturbation while enabling real-time monitoring of critical process parameters essential for semiconductor manufacturing quality control.  

The system's ability to remotely monitor these characteristics without direct electrical connection to the plasma chamber represents a significant advantage over conventional invasive probes, particularly in industrial environments where chamber access may be limited or process contamination must be minimized.