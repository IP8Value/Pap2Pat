Here is the complete patent application following the provided outline and incorporating the research paper's technical details:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to photonic devices and methods for tuning optical resonators with high precision. More specifically, the invention pertains to a resonant cavity-enhanced photoelectrochemical (PEC) etching technique that enables selective and scalable tuning of microphotonic and nanophotonic resonators. The disclosed method allows for permanent spectral alignment of optical resonators with picometer-scale precision, making it particularly suitable for applications in integrated photonics, quantum optics, and sensing technologies.  

## PRIOR ART  

Conventional methods for tuning optical resonators include thermal tuning, mechanical strain adjustment, and non-resonant photoelectrochemical etching. These approaches suffer from limited precision, lack of scalability, and inability to selectively target individual resonators within an ensemble. Thermal tuning introduces broad spectral shifts that are difficult to control at sub-nanometer scales, while mechanical methods often degrade resonator quality factors. Non-resonant PEC etching lacks spatial selectivity and requires above-bandgap optical pumping, which can cause material damage. Prior techniques also fail to address the challenge of collectively tuning multiple resonators to a common target wavelength without individual addressing.  

## SUMMARY OF THE INVENTION  

The invention provides a method for high-precision tuning of optical resonators through resonant cavity-enhanced photoelectrochemical etching. The technique utilizes below-bandgap optical pumping at a resonator's whispering gallery mode (WGM) resonance to initiate a spatially selective etching process. This approach offers several advantages: (1) sub-atomic-layer precision in resonator dimension control, enabling spectral tuning with picometer accuracy; (2) quality factor preservation or improvement during tuning; (3) inherent scalability for simultaneous tuning of multiple resonators; and (4) compatibility with both liquid and gaseous ionic environments.  

Key aspects of the invention include a photonic device configuration comprising semiconductor disk resonators coupled to optical waveguides, a resonant optical excitation system, and an ionic medium that facilitates the PEC reaction. The method involves sweeping the laser wavelength to track the shifting resonance during etching, allowing continuous control of the tuning process. Multiple resonators can be progressively aligned to a target wavelength through a cascaded tuning procedure that requires only a single laser source.  

## DETAILED DESCRIPTION  

### The Photonic Device  

The photonic device comprises gallium arsenide (GaAs) disk resonators with typical dimensions of 320 nm in thickness and 1 μm in radius, supported by aluminum gallium arsenide (AlGaAs) pedestals. These resonators support whispering gallery modes (WGMs) with mode volumes below 1 μm³ and quality factors exceeding 40,000. The device includes suspended GaAs waveguides featuring nanoscale tapers for evanescent optical coupling to the disk resonators. The entire structure is fabricated from a GaAs/AlGaAs/GaAs epitaxial wafer using electron beam lithography, inductively-coupled plasma reactive ion etching, and selective under-etching with hydrofluoric acid.  

The device architecture maintains optical confinement even when immersed in ionic liquids, enabling in situ tuning operations. The large refractive index contrast between GaAs and the surrounding medium preserves light guidance during liquid immersion. This configuration allows simultaneous optical characterization and precision etching through the same waveguide coupling system.  

### Installation Set-Up  

The tuning system comprises a tunable laser source optically coupled to the photonic device through the input waveguide. A photodetector monitors the transmission spectrum at the output waveguide. The device is immersed in an ionic liquid (typically water) or exposed to a humid gaseous environment that enables photoelectrochemical reactions.  

The laser system provides wavelength-tunable light below the GaAs bandgap (typically around 1.3 μm wavelength) with picometer-scale resolution. A control system coordinates laser wavelength sweeping with real-time spectral acquisition, enabling closed-loop control of the etching process. The ionic medium can be contained within a microfluidic chamber or provided through controlled humidity exposure in gaseous implementations.  

### Method for Tuning an Optical Resonator (FIG. 7)  

The single-resonator tuning method begins by identifying the target WGM resonance through optical transmission spectroscopy. The laser wavelength is set to resonance, initiating cavity-enhanced PEC etching through below-bandgap absorption at surface states. As etching proceeds, the resonator dimensions decrease, causing a blue shift of the resonance.  

The laser wavelength is continuously or stepwise adjusted to track the shifting resonance, maintaining resonant enhancement of the etching process. Each tuning cycle consists of a rapid wavelength sweep across the resonance for spectral characterization followed by precise wavelength adjustment. This procedure enables tuning precision better than 8 pm per cycle in resonator radius change, corresponding to less than 1/30 of an atomic monolayer.  

The process continues until the resonator reaches the target wavelength, at which point the laser is switched off and the device is dried (for liquid implementations). Typical tuning ranges span from a few picometers to several tens of nanometers, with permanent results achieved within minutes. The method preserves or improves the resonator's quality factor, with demonstrated increases from 41,000 to 70,000 in experimental implementations.  

### Method for Tuning Several Resonators at a Targeted Value (FIG. 8)  

The collective tuning method leverages the resonant selectivity of the PEC process to align multiple resonators to a common target wavelength. Beginning with an ensemble of nominally identical but spectrally dispersed resonators (due to fabrication variations), the laser is first tuned to the longest wavelength resonance (corresponding to the largest resonator).  

As the first resonator is etched and its resonance blue-shifts, the laser wavelength follows this shift until it overlaps with the next resonator's resonance. At this point, both resonators experience resonant enhancement of the etching process. The procedure continues progressively, with the laser dragging the aligned resonances toward shorter wavelengths until all resonators in the ensemble converge to the target wavelength.  

This cascaded alignment process requires only a single laser source and does not necessitate individual addressing of each resonator. Experimental demonstrations with five resonators show complete spectral overlap achieved through this method. The process can be implemented either through discrete steps (as described) or via a single continuous laser sweep from the red-detuned side of the resonances.  

### Different Embodiments for Varying the Laser Wavelength in Order to Tune Optical Resonators  

Several embodiments of the wavelength control system are disclosed:  

1. Stepwise tuning: The laser executes discrete wavelength jumps between spectral acquisition phases, enabling precise control of the etching rate per cycle. This approach provides the highest precision (better than 10 pm) at the expense of slower overall tuning speed.  

2. Continuous tracking: The laser wavelength follows the shifting resonance in real-time without discrete steps, enabling faster tuning while maintaining sub-picometer precision. This method is particularly suitable for single-resonator tuning applications requiring ultimate precision.  

3. Power-modulated tuning: The optical power is varied while maintaining resonance, providing control over the etching rate. Higher powers accelerate tuning while lower powers enable finer control near the target wavelength.  

4. Hybrid approaches: Combinations of the above methods can be employed, such as continuous tracking with periodic spectral verification or high-power coarse tuning followed by low-power fine adjustment.  

### Parameters and Experimental Data  

Experimental results demonstrate the method's capabilities:  

- Tuning precision: 7.2 pm per cycle in wavelength shift (equivalent to 8 pm radius change)  
- Etching speed: 0.5-1 nm radius change per second at 1 μW optical power  
- Quality factor improvement: From 41,000 to 70,000 after tuning  
- Tuning range: Demonstrated shifts up to 4 nm with preservation of spectral structure  
- Collective tuning: Five resonators spectrally aligned to within 10 pm  

The relationship between optical power and tuning rate is linear, with higher powers proportionally increasing the etching speed. The method shows excellent reproducibility across multiple resonators and tuning sessions.  

### Characteristics of the Tuned Optical Resonator  

The tuned resonators exhibit several advantageous characteristics:  

1. Permanent wavelength shift: The tuning results are stable after the etching process concludes and the device is dried.  

2. Improved optical quality: The quality factor typically increases due to smoothing of surface imperfections during the selective etching process.  

3. Preserved mode structure: The overall spectral characteristics of the resonances remain unchanged, with only a wavelength shift observed.  

4. Sub-atomic precision: The dimensional control achieves better than 1/30 of an atomic monolayer precision in resonator size adjustment.  

5. Scalable alignment: Multiple resonators can be matched to identical wavelengths despite initial fabrication variations.  

The invention enables unprecedented control over photonic resonator properties, addressing critical needs in integrated photonic circuits, quantum information processing, and precision sensing applications.  

--- 

This complete patent application follows the specified outline while incorporating all technical details from the research paper. Each section provides comprehensive coverage of the invention using proper patent language and structure.