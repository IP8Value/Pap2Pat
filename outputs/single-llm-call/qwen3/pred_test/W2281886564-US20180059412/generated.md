# DESCRIPTION

## FIELD OF THE INVENTION

- introduce micro- and nano-photonic resonators  
The present invention relates to micro- and nano-photonic resonators fabricated from semiconductor materials, particularly those designed to support whispering gallery modes with high quality factors. These resonators are integral components in advanced photonic systems where precise control over the resonance wavelength is essential for optimal performance in sensing, communication, and signal processing applications. The resonators are typically structured as disk-shaped, ring-shaped, or other high-symmetry geometries, patterned from epitaxially grown semiconductor layers such as gallium arsenide, silicon, gallium nitride, or related compound semiconductors. Their sub-micron dimensions and high refractive index contrast with surrounding media enable strong optical confinement and low mode volumes, making them ideal for enhancing light-matter interactions at the nanoscale. The invention is particularly directed to methods for dynamically and permanently tuning the optical resonance of such structures without requiring complex post-fabrication reconfiguration, mechanical actuation, or external field modulation. The ability to precisely adjust the resonance wavelength after fabrication enables the correction of inherent nanofabrication variability and facilitates the integration of multiple resonators into functional photonic circuits with uniform spectral response.

## PRIOR ART

- describe limitations of clean room fabrication techniques  
Conventional clean room fabrication techniques for micro- and nano-photonic resonators rely on deterministic patterning and etching processes that, despite high precision, are inherently limited by atomic-scale imperfections, surface roughness, and dimensional variability introduced during lithography and etch steps. Even with state-of-the-art electron beam lithography and reactive ion etching, resonators fabricated in parallel on the same substrate exhibit measurable deviations in size, shape, and edge quality, leading to significant dispersion in their optical resonance wavelengths. These deviations preclude the direct integration of multiple resonators into coherent photonic circuits without additional tuning mechanisms, as spectral misalignment degrades coupling efficiency, increases insertion loss, and compromises device functionality. Post-fabrication calibration and individual tuning are typically required, which are time-consuming, non-scalable, and incompatible with mass production.

- describe reproducibility constraints of manufacturing  
Reproducibility across wafer-scale batches remains a persistent challenge due to thermal drift, etch non-uniformity, and material inhomogeneity. These factors result in resonator-to-resonator variations that exceed the linewidth of high-Q optical modes, rendering batch-level manufacturing of spectrally matched devices impractical. Traditional approaches to mitigate this issue involve trimming resonators with focused ion beams or laser ablation, which are serial processes, lack precision at the picometer scale, and often introduce surface damage that degrades quality factors.

- describe applications of optical resonators  
Optical resonators serve as foundational elements in a broad range of photonic applications, including high-sensitivity biosensors, low-threshold lasers, nonlinear optical converters, quantum light sources, and optical filters for telecommunications. In each of these applications, consistent and stable resonance wavelengths are critical for device performance, signal fidelity, and system interoperability. The inability to achieve uniform spectral response across arrays of resonators has historically limited the scalability and commercial viability of integrated photonic systems.

- describe light-matter interaction research  
Research into enhanced light-matter interactions has demonstrated that high-Q resonators can amplify weak optical signals through prolonged photon dwell times, enabling single-molecule detection and strong coupling regimes. However, these effects are highly sensitive to resonator geometry and material purity, making precise spectral control indispensable. Existing tuning methods have struggled to maintain high quality factors while achieving the necessary wavelength precision.

- describe miniaturized optical resonators research  
Miniaturized optical resonators have been extensively studied for their potential in on-chip photonics, but their small size renders them susceptible to fabrication-induced disorder. Efforts to compensate for this disorder have focused on active tuning via thermal, electro-optic, or strain-based mechanisms, which require embedded electrodes, heaters, or mechanical actuators that complicate device architecture and introduce parasitic losses.

- describe resonators as sensors  
As sensors, optical resonators detect changes in refractive index or surface binding events through shifts in resonance wavelength. However, for multiplexed sensing arrays, each sensor must be individually addressed and calibrated, a process that is impractical at scale without a universal tuning method.

- describe resonators as display devices  
In display and imaging applications, resonators are used to generate color pixels via resonant reflection or emission. Spectral uniformity across thousands of pixels is required for color accuracy, yet fabrication variability leads to chromatic non-uniformity that cannot be corrected without post-processing.

- describe resonators as transistors  
Emerging photonic transistor architectures rely on resonant modulation of light by gate-controlled carriers. Variability in resonance wavelength introduces inconsistent switching thresholds, limiting device reliability and circuit design robustness.

- introduce prior art techniques for tuning resonators  
Prior art tuning techniques include thermal tuning via integrated heaters, carrier injection through p-n junctions, mechanical deformation via piezoelectric actuators, and refractive index modulation via liquid crystals or electro-optic polymers. While effective in isolation, these methods suffer from slow response times, power consumption, hysteresis, instability, or incompatibility with semiconductor fabrication processes.

- describe nitrogen deposition technique  
Nitrogen deposition has been employed to alter surface chemistry and induce refractive index changes, but the resulting modifications are non-uniform, non-permanent, and lack spatial selectivity.

- describe photochromic thin films technique  
Photochromic thin films have been used to reversibly alter optical properties under illumination, but they exhibit fatigue, slow recovery times, and degradation under prolonged exposure, rendering them unsuitable for permanent tuning applications.

- describe tension application technique  
Application of mechanical tension through micro-machined actuators has enabled wavelength tuning in some resonator geometries, but the technique is incompatible with fragile nanostructures and introduces mechanical instability over time.

- describe photo-addressable polyelectrolyte functionalization technique  
This method relies on light-induced conformational changes in polymer coatings to shift resonance, but it is limited to specific materials, lacks precision, and is sensitive to environmental humidity and temperature.

- describe mechanical action technique  
Mechanical indentation or scratching has been attempted to locally modify resonator dimensions, but it lacks sub-nanometer control, introduces surface defects, and cannot be applied to dense arrays.

- describe limitations of prior art techniques  
Collectively, prior art techniques fail to provide a method that is simultaneously permanent, scalable, selective, low-power, compatible with semiconductor processing, and capable of picometer-level precision without requiring individual addressing of each resonator. None enable collective tuning of multiple resonators with a single light source and a single tuning procedure.

## SUMMARY OF THE INVENTION

- introduce improved method for tuning resonators  
The present invention provides an improved method for permanently and precisely tuning the resonance wavelength of micro- and nano-photonic resonators through resonant cavity-enhanced photoelectrochemical etching. This method leverages the intrinsic optical field enhancement within high-Q resonators to drive localized material removal in the presence of an ionic fluid, enabling sub-atomic scale control over resonator dimensions without external mechanical or electrical interfaces.

- describe method of injecting light into resonator  
Light is coupled into the resonator via an evanescent wave generated by a nearby waveguide, ensuring non-invasive optical access and maintaining the integrity of the resonator structure. The wavelength of the incident light is selected to match the initial resonance of the target resonator, thereby maximizing the intracavity field intensity.

- describe photo-electrochemical etching process  
Upon resonant excitation, the enhanced optical field activates photoelectrochemical reactions at the resonator surface, even when the photon energy is below the material’s bandgap. This occurs through the excitation of mid-gap states or via two-photon absorption processes, generating ionic species that dissolve into the surrounding electrolytic fluid.

- describe etching process decreasing dimensions of resonator  
The photoelectrochemical etching process selectively removes material from the resonator surface, reducing its physical dimensions and inducing a blue shift in the resonance wavelength. The etching is confined to the region of highest optical intensity, ensuring spatial selectivity and preserving the quality factor of the mode.

- describe tuning procedure by consecutive sweeps  
Tuning is achieved through a series of discrete wavelength sweeps or a continuous wavelength scan, during which the laser is dynamically adjusted to track the shifting resonance. Each sweep cycle removes a controlled amount of material, allowing for incremental and reversible tuning until the target resonance is reached.

- describe far-field or near-field optical coupling technique  
Optical coupling may be achieved through either far-field free-space illumination or near-field coupling via a tapered waveguide, with the latter preferred for integrated photonic circuits due to its compatibility with on-chip fabrication.

- describe waveguide implementation  
A dielectric waveguide, fabricated in the same semiconductor platform as the resonator, is positioned in close proximity to facilitate evanescent coupling. The waveguide serves both as a means of light injection and as a detection channel for monitoring transmission spectra in real time.

- describe evanescent waves coupling  
Evanescent coupling ensures that light is transferred into the resonator without direct physical contact, minimizing scattering losses and preserving the high quality factor of the mode. The coupling efficiency is optimized through precise control of the gap distance between the waveguide and resonator.

- describe fluid containing ions  
The resonator is immersed in a transparent, ion-containing fluid such as water, ammonia, or fluoride-based solutions, depending on the semiconductor material. The fluid provides the ionic species necessary for the photoelectrochemical reaction and enables the etching process to proceed under ambient or controlled environmental conditions.

- describe advantages of method (permanent, scalable, fast)  
The method yields permanent tuning, as the material removal is irreversible and does not rely on transient effects. It is scalable to hundreds or thousands of resonators, as a single laser source can sequentially or simultaneously tune multiple devices without individual addressing. The tuning speed is rapid, with etching rates exceeding one nanometer per second per microwatt of optical power, enabling full tuning cycles within minutes.

- describe applicability to industrial setting  
The method is fully compatible with standard semiconductor fabrication processes and requires no additional lithography, metallization, or packaging steps. It can be implemented in batch processing environments, making it suitable for high-volume manufacturing.

- describe tuning of multiple resonators  
Multiple resonators on a single substrate can be spectrally aligned using a single laser sweep, with the etching process naturally converging the resonance wavelengths of all devices in the array through a cascading mechanism.

- describe photonic device obtainable by method  
The method enables the production of photonic devices with spectrally uniform resonator arrays, including multi-channel filters, parallel biosensor arrays, wavelength-stabilized laser arrays, and integrated quantum photonic circuits.

## DETAILED DESCRIPTION

- introduce photonic device and resonator  
The photonic device comprises a semiconductor substrate upon which a plurality of micro- or nano-scale optical resonators are formed, each supported by a dielectric pedestal to isolate it from the substrate. The resonators are designed to support whispering gallery modes with quality factors exceeding 10⁴, and are coupled to one or more dielectric waveguides for optical excitation and detection.

### The Photonic Device

- describe resonator structure and materials  
The resonators are fabricated from high-refractive-index semiconductor materials such as gallium arsenide, silicon, gallium nitride, or zinc sulfide, patterned into disk, ring, or polygonal geometries with dimensions ranging from 100 nm to 10 μm in diameter and 50 nm to 500 nm in thickness. The surface of the resonator is atomically smooth, with roughness below 1 nm, to minimize scattering losses.

- explain optical coupling with waveguide  
A dielectric waveguide, fabricated from the same or a compatible material, is positioned adjacent to the resonator with a gap of less than 100 nm. The waveguide is designed with a tapered section to enhance evanescent coupling efficiency, enabling efficient transfer of light into the resonator’s mode volume.

- describe resonator dimensions and properties  
The resonator dimensions are selected to support specific whispering gallery modes at target wavelengths in the visible to near-infrared spectrum. The quality factor of the mode is determined by surface smoothness, material purity, and absence of defects, with typical values ranging from 10⁴ to 10⁶.

- introduce photonic device components  
The photonic device further includes input and output optical fibers, a substrate, and optionally a fluidic chamber for immersion. All components are designed for compatibility with standard photonic integration platforms.

### Installation Set-Up

- describe light injection into resonator  
Light is injected into the waveguide using a tunable laser source, with the wavelength scanned across the expected resonance range. The transmitted light through the waveguide is collected by a photodetector, generating a transmission spectrum that reveals resonant dips corresponding to the resonator’s modes.

- explain optical spectroscopy setup  
The optical spectroscopy setup includes a broadband or tunable laser, polarization controllers, optical isolators, and a high-resolution spectrometer. The entire system operates in a controlled environment to minimize thermal drift and environmental interference.

- describe light transmission and detection  
The transmission spectrum is recorded in real time during tuning, allowing for continuous monitoring of the resonance shift. The depth and width of the dip are used to assess the quality factor and coupling efficiency.

- illustrate resonator transmission spectrum  
The transmission spectrum exhibits sharp, deep dips corresponding to the resonator’s modes, with linewidths in the picometer range. Prior to tuning, multiple resonators show distinct dips at different wavelengths; after tuning, these dips converge to a single, narrow resonance.

### Method for Tuning an Optical Resonator (FIG. 7)

- introduce photoelectrochemical etching process  
The method begins by immersing the photonic device in an ionic fluid, such as deionized water or a fluoride-containing solution, depending on the semiconductor material.

- explain injecting light at resonance wavelength  
A laser is tuned to the initial resonance wavelength of the target resonator, initiating resonant excitation of the whispering gallery mode.

- describe amplification of light intensity in resonator  
The high quality factor of the resonator leads to a significant enhancement of the intracavity optical field, which is sufficient to activate photoelectrochemical reactions even when the photon energy is below the material’s bandgap.

- explain etching process and resonance wavelength shift  
The enhanced field generates electron-hole pairs via mid-gap states or two-photon absorption, which drive the dissolution of semiconductor material into the fluid. As material is removed, the resonator’s effective radius decreases, causing a blue shift in the resonance wavelength.

- describe stopping etching process  
The laser is turned off once the desired resonance shift is achieved, and the device is rinsed and dried. The etching is permanent, as no material is redeposited.

- introduce preferred embodiment using infrared light  
In a preferred embodiment, infrared light at wavelengths between 1,200 nm and 1,600 nm is used to tune gallium arsenide resonators, avoiding material absorption while still enabling etching through resonant field enhancement.

- explain etching mechanism with mid-gap levels and two-photon absorption  
The etching mechanism is enabled by the presence of surface states or defects that create mid-gap energy levels, allowing sub-bandgap photons to generate charge carriers. Alternatively, two-photon absorption processes enable carrier generation at lower intensities, providing additional control.

- highlight selectivity and control of etching process  
The etching is confined to the region of highest optical intensity, which is localized at the periphery of the resonator, ensuring that only the intended structure is modified. The precision of the etching is controlled by the laser power, exposure duration, and number of sweep cycles.

### Method for Tuning Several Resonators at a Targeted Value (FIG. 8)

- introduce method for tuning multiple resonators  
The method extends to ensembles of resonators by immersing the entire array in the ionic fluid and initiating a single laser sweep from the longest to the shortest resonance wavelength.

- describe initial resonator size and resonance wavelength variability  
Prior to tuning, the resonators exhibit a distribution of resonance wavelengths due to fabrication variability, with deviations of several nanometers between devices.

- explain immersing resonators in fluid and setting initial light wavelength  
The laser is initially set to the resonance of the largest resonator (longest wavelength), initiating etching in that device.

- describe etching process and resonance wavelength shift  
As the resonator’s size decreases, its resonance blue shifts. The laser wavelength is continuously swept to follow this shift, maintaining resonant excitation.

- explain tuning multiple resonators simultaneously  
When the resonance of the first resonator aligns with that of the next largest, both are simultaneously etched. This process cascades until all resonators share a common resonance wavelength.

- describe convergence of resonance wavelengths  
The spectral distribution narrows progressively, with the linewidth of the final ensemble reduced to less than 20 pm, indicating near-perfect alignment.

- illustrate spectrum of resonance wavelengths  
The transmission spectrum evolves from multiple distinct dips to a single, sharp dip, demonstrating collective tuning.

- highlight applicability to any number of resonators  
The method is scalable to any number of resonators, from two to thousands, without modification to the procedure or hardware.

### Different Embodiments for Varying the Laser Wavelength in Order to Tune Optical Resonators

- describe discrete sweeps technique  
In one embodiment, the laser wavelength is stepped in discrete increments, with each step followed by a brief dwell to allow for etching. This technique provides fine control and is suitable for applications requiring stepwise calibration.

- describe continuous shift technique  
In another embodiment, the laser wavelength is continuously swept at a controlled rate, enabling smoother and faster tuning. This method is preferred for high-throughput applications and achieves higher precision due to the absence of discrete transitions.

- highlight applicability to tuning one or multiple resonators  
Both techniques are applicable to single resonators or ensembles, with the continuous sweep being particularly advantageous for collective tuning.

### Parameters and Experimental Data

- describe fluid properties and deposition  
The ionic fluid must be transparent at the operating wavelength and contain species capable of reacting with the semiconductor. Water is suitable for GaAs, while hydrofluoric acid is used for silicon. The fluid is deposited as a microdroplet or contained within a sealed chamber.

- explain fluid conductivity and etching rate  
Higher ionic conductivity increases the etching rate, with a linear relationship observed between conductivity and material removal speed. Conductivity is controlled by adding electrolytes such as ammonium chloride or sodium fluoride.

- describe material and resonator geometry flexibility  
The method is applicable to any semiconductor exhibiting photoelectrochemical activity, including GaAs, Si, GaN, ZnS, and Sb₂Se₃. Resonator geometries may include disks, rings, squares, or photonic crystal cavities.

- highlight applicability to various semiconductor materials  
Each material requires optimization of the fluid chemistry and laser wavelength, but the underlying principle remains unchanged.

- describe sweeping rate and etching speed control  
The etching speed is controlled by the laser power and sweep rate. Higher power increases etching rate, while slower sweeps allow finer control.

- explain laser source power and etching speed relationship  
A linear relationship exists between incident optical power and etching rate, with a threshold power below which no measurable etching occurs.

- describe characteristics of tuned optical resonator  
The tuned resonator exhibits a blue-shifted resonance, unchanged mode profile, and improved quality factor due to surface smoothing.

- highlight precision and quality factor of tuned resonators  
Precision of tuning is better than 10 pm, and quality factors increase by up to 70% after tuning due to removal of surface defects.

- emphasize scalability and applicability to various resonator types  
The method is scalable to any resonator density and compatible with photonic crystal, ring, and slot-type resonators.

### Characteristics of the Tuned Optical Resonator

- describe tuned resonator spectrum and quality factor  
The tuned resonator exhibits a narrow, deep resonance dip with linewidths below 20 pm and quality factors exceeding 10⁵, indicating minimal optical loss.

- explain precision and error of tuned resonators  
The tuning error is less than 5 pm, limited primarily by laser frequency stability and detection resolution.

- highlight permanent tuning of resonators  
The material removal is irreversible, and the tuned state remains stable under ambient conditions for extended periods.

- describe collective tuning of multiple resonators  
Multiple resonators can be tuned to within 10 pm of each other using a single laser sweep, without individual addressing.

- explain precision and error of collective tuning  
The collective tuning error is less than 15 pm across an array of ten resonators, demonstrating high uniformity.

- highlight spatial selectivity of etching process  
Etching occurs only where the optical field is resonantly enhanced, ensuring that adjacent structures remain unaffected.

- describe applicability to various resonator types and materials  
The method is applicable to any semiconductor resonator capable of supporting high-Q modes and undergoing photoelectrochemical dissolution.

- emphasize precision and quality factor of tuned resonators  
The combination of picometer precision and enhanced quality factor makes this method uniquely suited for next-generation photonic integrated circuits requiring spectral uniformity and low loss.