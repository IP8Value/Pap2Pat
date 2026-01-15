# DESCRIPTION

## BENEFIT CLAIMS TO RELATED APPLICATIONS

This application claims the benefit of U.S. Provisional Application No. 63/000,000, filed on April 28, 2020, which is hereby incorporated by reference in its entirety.

## FIELD OF THE INVENTION

The present invention relates generally to optical resonators and, more specifically, to spiral resonators for on-chip laser frequency stabilization. These resonators are designed to provide high coherence and stability in various applications, such as coherent fiber-optic communications, remote sensing, and atomic physics.

## BACKGROUND

Laser frequency stabilization is crucial in many scientific and technological applications. Conventional methods often involve bulky and expensive equipment, limiting their integration into compact systems. On-chip resonators offer a promising solution by providing high coherence and stability in a miniaturized form factor. However, achieving the required performance remains challenging due to various noise sources and environmental factors.

## SUMMARY

The present invention provides spiral resonators for on-chip laser frequency stabilization. The spiral design increases the effective path length within a small footprint, enhancing the stability and coherence of the laser. By using high-Q materials and advanced fabrication techniques, these resonators can achieve phase-noise suppression and linewidth reduction comparable to or better than conventional systems.

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 illustrates an exemplary spiral resonator design for on-chip laser frequency stabilization.
FIG. 2 shows the experimental setup for measuring the performance of the spiral resonator.
FIG. 3 displays phase-noise spectral density functions for different resonators.
FIG. 4 compares the electrical spectrum of heterodyne detection with free-running and locked lasers.
FIG. 5 presents Allan deviation measurements for the spiral resonator.

## DETAILED DESCRIPTION

### Introduction to Spiral Resonators

Spiral resonators are a novel approach to on-chip laser frequency stabilization. These resonators increase the effective path length within a limited area, enhancing the stability and coherence of the laser. By using high-Q materials and advanced fabrication techniques, spiral resonators can achieve phase-noise suppression and linewidth reduction comparable to or better than conventional systems.

### Design and Fabrication

The spiral resonator is designed with a series of concentric rings etched into a silicon substrate. The width and spacing of the rings are optimized to minimize loss and maximize the Q-factor. High-Q materials, such as silica-on-silicon, are used to further improve performance. Advanced fabrication techniques, including deep reactive-ion etching (DRIE) and flame hydrolysis deposition, ensure precise control over the dimensions and quality of the resonator.

### Experimental Setup

To measure the performance of the spiral resonator, an experimental setup is used that includes two fiber lasers locked to separate Pound–Drever–Hall systems. The lasers are heterodyned to produce a beat signal, which is analyzed using an electrical spectrum analyzer, phase-noise analyser, and frequency counter. Acoustic shielding and environmental control measures are implemented to minimize external noise sources.

### Performance Metrics

The performance of the spiral resonator is evaluated based on several metrics, including phase-noise suppression, linewidth reduction, and Allan deviation. Phase-noise spectral density functions are measured over a range of offset frequencies, and the data are compared to free-running lasers and conventional disk resonators. The results show significant improvements in noise suppression and stability for the spiral resonator.

### Phase-Noise Suppression

The phase-noise spectral density function for the heterodyned signals is analyzed both with and without the locking systems engaged. Data are shown for free-running lasers, 3 mm disk resonators, and 1.2 m spiral cases. Within the bandwidth of the feedback control system, an average of 26 dB suppression of fiber-laser phase-noise was measured when the lasers were locked to the 1.2 m spirals. This is a significant improvement over the 10 dB suppression achieved with 3 mm diameter disks.

### Linewidth Reduction

The linewidth reduction is calculated from the phase-noise spectra and compared for different resonators. The free-running lasers have an effective linewidth of 900 Hz, which reduces to 400 Hz when locked to 3 mm disks and further reduces to 100 Hz when locked to 1.2 m spiral resonators. This demonstrates the superior performance of the spiral resonator in reducing linewidth.

### Allan Deviation

Allan deviation measurements are conducted using a frequency counter to confirm the frequency stabilization by the spiral resonators. Over a range of gate times, the Allan deviations of the spiral-locking case are improved compared to the free-running (unlocked) case. At a gate time of 400 μs, a minimum relative Allan deviation of 5.5 × 10^-13 is measured, which is ten times lower than that of the free-running case.

### Mechanically Induced Noise

Measurements of thermo-mechanical-induced noise are also conducted using both disk and spiral resonators. The optomechanical coupling parameter is expected to vary inversely with cavity length, leading to an inverse quadratic dependence on length. This dependence is observed over a range of cavity lengths, and the results show that mechanical noise is significantly reduced for the largest spirals measured (1.2 m path length).

### Future Improvements

Future performance improvements can be achieved by increasing the effective path length of the spiral resonator and using thicker oxides in the fabrication process. Optical-fibre-based reference systems using 1 km fibre delays have attained a phase-noise level of -83 dBc Hz^-1 at 1 kHz offset frequency. In the current chip-based design, lengths in excess of 100 m are feasible, which could further enhance the performance of the spiral resonator.

### Conclusion

Spiral resonators for on-chip laser frequency stabilization offer a promising solution to achieving high coherence and stability in compact systems. By increasing the effective path length within a small footprint and using advanced fabrication techniques, these resonators can achieve phase-noise suppression and linewidth reduction comparable to or better than conventional systems. Future improvements in design and fabrication will further enhance their performance, making them suitable for a wide range of applications.

## CLAIMS

1. A spiral resonator for on-chip laser frequency stabilization comprising:
   - A silicon substrate;
   - A series of concentric rings etched into the substrate to form a spiral path;
   - High-Q materials deposited on the substrate to enhance performance;
   - Advanced fabrication techniques used to control dimensions and quality.

2. The spiral resonator of claim 1, wherein the width and spacing of the rings are optimized to minimize loss and maximize the Q-factor.

3. The spiral resonator of claim 1, further comprising deep reactive-ion etching (DRIE) for precise control over the dimensions.

4. The spiral resonator of claim 1, further comprising flame hydrolysis deposition for high-quality material deposition.

5. A method for measuring the performance of a spiral resonator, comprising:
   - Locking two fiber lasers to separate Pound–Drever–Hall systems;
   - Heterodyning the locked lasers to produce a beat signal;
   - Analyzing the beat signal using an electrical spectrum analyzer, phase-noise analyser, and frequency counter.

6. The method of claim 5, further comprising implementing acoustic shielding and environmental control measures to minimize external noise sources.

7. A system for on-chip laser frequency stabilization, comprising:
   - A spiral resonator as described in claim 1;
   - Two fiber lasers locked to the spiral resonator using Pound–Drever–Hall systems;
   - An experimental setup for measuring phase-noise suppression and linewidth reduction.

8. The system of claim 7, further comprising an electrical spectrum analyzer, phase-noise analyser, and frequency counter for performance analysis.

9. A method for improving the performance of a spiral resonator, comprising:
   - Increasing the effective path length of the spiral;
   - Using thicker oxides in the fabrication process;
   - Implementing advanced noise reduction techniques.

10. The method of claim 9, further comprising achieving phase-noise levels comparable to or better than conventional systems using lengths in excess of 100 m.