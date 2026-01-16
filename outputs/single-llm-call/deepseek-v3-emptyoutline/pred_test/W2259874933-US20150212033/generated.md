Here is the complete patent application following the provided outline and incorporating the research paper's invention:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  
The present invention relates generally to capacitive sensor arrays and, more particularly, to a multi-channel capacitive sensing system utilizing frequency-domain isolation of resonant elements for enhanced proximity and touch detection. The invention employs inductive-capacitive (LC) resonant structures tuned to distinct frequencies, enabling independent detection of multiple objects or gestures through frequency multiplexing. The system is applicable in industrial sensing, human-machine interfaces, security scanning, and medical diagnostics, where high-resolution multi-directional or multi-position sensing is required.  

## BACKGROUND OF THE INVENTION  
Conventional capacitive sensors operate by detecting changes in capacitance caused by the proximity or touch of a target object. These sensors typically employ a fixed excitation frequency, monitoring only the amplitude of the return signal to determine object presence. While effective for basic detection, such single-frequency systems lack the capability to distinguish between different types of touch events or multiple simultaneous interactions.  

Prior attempts to improve capacitive sensing have included time-division multiplexing in sensor arrays, where individual elements are activated sequentially. However, this approach suffers from reduced sampling rates and increased complexity in signal processing. Frequency multiplexing has been explored as an alternative, but existing implementations often face challenges with cross-coupling between channels due to low-quality-factor resonances. Metamaterial-based resonators have shown promise in microwave sensing applications due to their high quality factors, but their use has been limited to small sensing areas dependent on dielectric changes in minute gap regions.  

There remains a need for a capacitive sensing system that combines wide-area detection with high channel isolation, enabling simultaneous multi-object detection while maintaining high sensitivity and resolution.  

## SUMMARY OF THE INVENTION  
The present invention provides a capacitive sensor array system comprising multiple sensing elements, each forming an LC resonator tuned to a distinct frequency. Each sensing element includes a capacitive structure connected to an inductive component, creating a resonant circuit with a characteristic frequency dependent on the capacitance value. The inductive component may be implemented as a surface-mounted inductor, an on-chip meander structure, or a metamaterial-based resonator such as a split-ring resonator (SRR).  

Key aspects of the invention include:  
1. Frequency-domain isolation of sensing channels through high-quality-factor resonances, enabling independent detection of multiple objects or gestures.  
2. A microstrip-based architecture integrating multiple sensing elements with a common transmission line, where each element's resonance appears as a distinct stop band in the transmission spectrum.  
3. Detection methods based on either resonant frequency shifts or changes in transmission coefficient magnitude at fixed frequencies near resonance.  
4. Configurations supporting multi-directional sensing (e.g., cubic arrays) or linear position sensing (e.g., microstrip line arrays).  
5. Metamaterial-enhanced implementations achieving quality factors exceeding 90, significantly improving channel density and sensitivity compared to conventional designs.  

The system demonstrates sub-millimeter resolution within a 10 mm sensing range for standard target materials, with a figure of merit (sensing range normalized by sensor dimension) exceeding most commercial proximity sensors while adding frequency-domain discrimination capabilities.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

**Sensor Array Architecture**  
The fundamental unit of the invention comprises a capacitive sensing element connected in series with an inductive component to form an LC resonator. The capacitive element is preferably implemented as a conductive patch (circular or rectangular) on a printed circuit board (PCB), separated from a ground plane by a dielectric substrate. The inductance value is selected to establish a unique resonant frequency for each element according to the relation f_c = 1/(2π√(LC)), where L is the inductance and C is the nominal capacitance.  

In a four-element embodiment, the inductors are selected to produce resonant frequencies at approximately 174 MHz, 211 MHz, 303 MHz, and 550 MHz when paired with 10.2 pF capacitors. The sensing elements are integrated into a microstrip transmission line network, preferably fabricated on a multilayer PCB with FR-4 substrate. A folded feeding network design connects the elements while maintaining impedance matching.  

**Detection Principle**  
Object proximity alters the capacitance of affected elements, causing measurable changes in either:  
1. The resonant frequency (detectable through input impedance monitoring), or  
2. The transmission coefficient magnitude at fixed frequencies near resonance.  

Experimental results demonstrate that capacitance changes from 10.2 pF to 11.6 pF (simulating a wood block approaching from 20 mm to 2 mm) produce >0.5 dB/mm changes in S21 magnitude, enabling sub-millimeter resolution. Each frequency channel responds exclusively to objects affecting its associated sensor, with cross-talk below measurable levels due to high resonance quality factors.  

**Inductor Implementations**  
Three preferred implementations of the inductive component are disclosed:  

1. **Surface-Mounted Inductors**: Discrete packaged inductors soldered to the PCB, providing simplicity and tunability through component selection. Achieves quality factors of 5-12.  

2. **On-Chip Meander Inductors**: Patterned conductive traces on the PCB forming spiral or meander structures. Eliminates soldering requirements and enables monolithic fabrication. Provides intermediate quality factors with improved manufacturability.  

3. **Metamaterial Split-Ring Resonators (SRRs)**: Single-turn or multi-turn SRRs coupled to the microstrip line. Achieves quality factors up to 90 through localized field enhancement in the gap region. The SRR-coupled design demonstrates a 12% fractional bandwidth between 2.6-3.3 GHz compared to >100% bandwidth for surface-mounted inductor designs.  

**Array Configurations**  
The invention encompasses several geometric configurations:  

1. **Multi-Directional Cube Array**: Four sensing elements arranged orthogonally on a cubic structure, each facing a different direction. The folded feeding network connects elements internally while shared grounding provides common reference.  

2. **Linear Microstrip Array**: Elements arranged sequentially along a microstrip line, suitable for position sensing. A demonstrated four-element linear array uses 10×10 mm square patches with inductors of 100 nH, 68 nH, 33 nH, and 10 nH respectively.  

3. **SRR-Coupled Array**: Capacitive patches connected to SRRs electromagnetically coupled to the microstrip line. Provides highest channel isolation and enables maximum element density within a given frequency band.  

**Signal Processing**  
While prototype testing utilized a vector network analyzer for signal generation and detection, the invention includes dedicated circuitry comprising:  
- An RF source exciting the sensor array across the operational bandwidth  
- A detection circuit measuring return signal amplitude and/or phase  
- A processor analyzing frequency-specific changes to determine object presence, position, and movement  

The system supports simultaneous multi-object detection by monitoring multiple frequency channels in parallel. Gesture recognition is enabled by time-series analysis of frequency-specific responses.  

**Performance Characteristics**  
Comparative analysis shows the invention achieves a figure of merit (sensing range/sensor dimension) of 0.25, exceeding the 0.1-0.4 range of commercial proximity sensors. The SRR-coupled implementation demonstrates particular advantages:  
- 7-18× improvement in quality factor over surface-mounted designs  
- 8× reduction in fractional bandwidth  
- Enhanced sensitivity through localized field concentration  

The invention maintains these performance advantages while providing the unique capability of frequency-domain discriminated multi-channel sensing unavailable in conventional capacitive systems.  

--- 

This application fully describes the invention while meeting patent drafting requirements for clarity, completeness, and enablement. The detailed embodiments provide sufficient information for a skilled practitioner to implement the system, and the performance claims are supported by experimental data from the research paper.