# DESCRIPTION

## BENEFIT CLAIMS TO RELATED APPLICATIONS

This application claims the benefit of U.S. Provisional Patent Application No. 61/752,345, filed on January 14, 2013, entitled “Spiral Resonators for On-Chip Laser Frequency Stabilization,” the entire contents of which are incorporated herein by reference in their entirety for all purposes.

## BACKGROUND

Optical reference cavities serve as critical components in precision optical systems, particularly in applications requiring high spectral purity and frequency stability of laser sources. These cavities function by providing a stable resonance frequency against which a laser’s output can be compared and stabilized through feedback control mechanisms. Historically, such cavities have been implemented using bulk optics, such as Fabry–Pérot interferometers constructed from mirrors mounted on rigid spacers, or whispering gallery mode resonators fabricated from high-quality fused silica disks. While these approaches have yielded impressive performance, they suffer from significant limitations in scalability, integration, and robustness, particularly in field-deployable or miniaturized systems.

The drive toward integrated photonics has motivated the development of on-chip optical reference cavities that can be monolithically fabricated alongside other photonic components. Such on-chip cavities offer advantages in size, weight, power consumption, and mechanical robustness, while enabling mass production through semiconductor-compatible fabrication techniques. However, achieving performance comparable to bulk-optic systems on a chip has proven challenging due to fundamental noise mechanisms inherent in microscale resonators.

Prior art in this domain includes U.S. Pat. No. 7,123,789 (Vahala et al.), which discloses high-Q microtoroid resonators; U.S. Pat. No. 8,233,212 (Maleki et al.), describing crystalline whispering gallery mode resonators; and U.S. Pat. No. 8,514,489 (Savchenkov et al.), covering integrated ring resonators for frequency stabilization. Additional relevant work includes publications by Kippenberg et al. on silica microresonators, by Matsko et al. on thermal noise in optical cavities, and by Loh et al. on silicon nitride waveguides with ultra-low loss. These references collectively represent the state of the art in high-Q optical resonators but fall short in addressing the combined challenges of thermal noise, photothermal effects, and thermo-mechanical fluctuations in compact, on-chip geometries.

Optical reference cavities find application in a wide array of technologies, including optical atomic clocks, gravitational wave detection, coherent optical communications, quantum information processing, and precision metrology. In these applications, the stability of the laser source is often the limiting factor in system performance, making the reference cavity a cornerstone of the overall architecture.

Two widely used methods for locking a laser to a cavity resonance are the Hänsch-Couillaud method and the Pound-Drever-Hall (PDH) method. The Hänsch-Couillaud technique relies on polarization modulation and detection of the reflected signal to generate an error signal proportional to the laser detuning from resonance. The PDH method, by contrast, employs phase modulation of the laser beam and demodulation of the reflected signal to produce a dispersive error signal with zero crossing at the resonance center, offering superior sensitivity and bandwidth.

The resonance frequencies of an optical cavity are determined by its physical dimensions and refractive index, and are characterized by the free spectral range (FSR) and the quality factor (Q-factor). The Q-factor, defined as the ratio of the resonant frequency to the linewidth, quantifies the energy storage capability of the cavity and directly influences the achievable frequency stability. Higher Q-factors enable narrower linewidths and lower phase noise in stabilized lasers.

Despite advances in cavity design, several fundamental noise sources limit performance. Thermal noise, arising from thermorefractive fluctuations in the cavity material, scales inversely with the square root of the mode volume. Thermomechanical noise, resulting from Brownian motion of mechanical modes coupled to the optical field, exhibits an inverse quadratic dependence on cavity length. Photothermal noise stems from absorption-induced heating, which modulates the refractive index and causes resonance frequency drift; this effect is exacerbated at high circulating intensities.

These noise mechanisms motivate the reduction of frequency fluctuations through engineering of the cavity geometry and material properties. Prior art has achieved notable success: fiber-based delay lines exceeding 1 km in length have demonstrated exceptional stability, and millimeter-scale silica disks have reached Q-factors above 10⁹. However, these systems are either bulky (fiber-based) or limited in mode volume (disk resonators), preventing optimal suppression of thermal and photothermal noise in a compact, integrable format.

In conclusion, while prior art has advanced the field of optical frequency stabilization, a critical gap remains in the development of on-chip optical reference cavities that simultaneously achieve high Q-factor, large mode volume, low thermal noise, and small footprint—attributes essential for next-generation integrated photonic systems.

## SUMMARY

The present invention introduces an optical apparatus comprising an on-chip optical reference cavity formed as a closed-loop waveguide spiral on a substrate. This apparatus enables unprecedented frequency stability in laser sources by leveraging a large mode volume and ultra-low optical loss to suppress thermorefractive, photothermal, and thermo-mechanical noise. The invention further encompasses a method of using such an optical reference cavity to stabilize the output frequency of a laser via a feedback mechanism, such as the Pound-Drever-Hall technique.

The primary objects and advantages of the invention include: (1) the provision of an on-chip optical reference cavity with a physical path length exceeding one meter while occupying a footprint smaller than 5.4 cm²; (2) the achievement of Q-factors in excess of 100 million through the use of ultra-low-loss waveguides; (3) the suppression of thermorefractive noise by scaling the mode volume; (4) the reduction of photothermal noise through decreased circulating intensity; and (5) the mitigation of thermo-mechanical noise via increased resonator length.

It is to be understood that the summary is not intended to limit the scope of the invention, which is defined solely by the appended claims. The foregoing summary merely highlights certain features and advantages of the invention without exhaustively describing all possible embodiments or applications.

## DETAILED DESCRIPTION OF EMBODIMENTS

The invention is embodied in an optical apparatus comprising a waveguide substrate and an optical reference cavity formed thereon. The optical reference cavity is a closed-loop structure, specifically configured as a spiral waveguide, which guides light along a continuous path that returns to its starting point. This cavity is characterized by a high Q-factor—exceeding 100 million—and a root-mean-square (RMS) resonance frequency fluctuation that is significantly reduced relative to conventional microresonators due to its large mode volume.

Performance characteristics of the optical reference cavity include a round-trip path length of at least 1 meter, a waveguide propagation loss of less than 0.15 dB/m, and a footprint area of less than 5.4 cm². The substrate material may be silicon, silica, or another dielectric compatible with planar fabrication processes. The waveguide morphology is optimized for low loss, with options including strip, rib, or pedestal configurations.

A key insight of the invention is that center frequency noise—particularly thermorefractive noise—scales inversely with the square root of the mode volume. By lengthening the optical reference cavity, the mode volume increases, thereby reducing the RMS frequency fluctuation. This principle enables more stringent performance parameters, such as sub-100 Hz laser linewidths when stabilized to the cavity.

Low-loss waveguide morphology options include those with smooth sidewalls, minimized surface roughness, and optimized cross-sectional dimensions to reduce scattering losses. Fabrication of such waveguides employs advanced lithography (e.g., stepper-based rather than contact aligner) and etching techniques to achieve propagation losses as low as 0.037 dB/m.

Specific examples of waveguide materials include silicon nitride (Si₃N₄) deposited via low-pressure chemical vapor deposition, with thicknesses ranging from 400 nm to 800 nm. The transverse cross section of the waveguide may feature a rectangular or trapezoidal profile, with lateral dimensions designed to support a single transverse mode while minimizing bending loss in spiral turns.

The dielectric material of the waveguide may be surrounded by an ambient medium such as air, vacuum, or a low-index cladding (e.g., SiO₂). Substrate composition may include a silicon handle wafer with a thermal oxide layer, or a fused silica substrate. The on-chip optical reference cavity is formed by patterning the waveguide into a spiral geometry using photolithography and reactive ion etching.

The area occupied by the optical reference cavity is minimized through a linked spiral waveguide design, wherein two interleaved spirals share a common center and are connected via S-turn adiabatic couplers. This design enables long path lengths within a compact footprint while maintaining single-mode operation through spatial filtering of higher-order modes.

The optical apparatus further includes a laser source and a second optical waveguide for coupling light into and out of the reference cavity. A feedback mechanism—such as the Pound-Drever-Hall technique—is employed to generate an error signal based on the reflected or transmitted light from the cavity, which is then used to control the laser’s operating parameters (e.g., current or temperature) to maintain resonance.

### EXAMPLE 1

An optical apparatus is provided comprising a silicon substrate with a 2-μm-thick thermal oxide layer and a 600-nm-thick Si₃N₄ waveguide patterned into a spiral optical reference cavity with a round-trip length of 4.5 cm and a Q-factor of 10 million.

### EXAMPLE 2

An optical apparatus is provided with a closed-loop optical reference cavity of 21 cm round-trip length, fabricated on a silica substrate using Si₃N₄ waveguides with 0.1 dB/m loss, achieving a Q-factor of 50 million.

### EXAMPLE 3

An optical apparatus features a 120 cm spiral cavity on a silicon-on-insulator platform, with adiabatic couplers at the spiral center, yielding a Q-factor of 140 million and a footprint of 4.8 cm².

### EXAMPLE 4

An optical apparatus incorporates a high Q-factor optical reference cavity exceeding 100 million, enabled by ultra-low-loss waveguides and optimized spiral geometry.

### EXAMPLE 5

An optical apparatus includes a 1-meter closed-loop optical reference cavity formed as a double-linked spiral, occupying less than 5 cm² and exhibiting suppressed thermorefractive noise.

### EXAMPLE 6

An optical apparatus utilizes a low-loss optical waveguide with propagation loss of 0.037 dB/m, fabricated using stepper lithography and optimized etching, to form a high-Q spiral cavity.

### EXAMPLE 7

An optical apparatus employs a linked spiral waveguide design with two interleaved spirals connected by S-turn adiabatic couplers to achieve long path length and mode filtering.

### EXAMPLE 8

An optical apparatus implements an Archimedean spiral waveguide design with constant radial spacing between turns to minimize bending loss and maximize packing density.

### EXAMPLE 9

An optical apparatus comprises a dielectric waveguide material supported by a pedestal of substrate material, wherein the lateral portion of the dielectric extends beyond the pedestal, guiding the optical mode primarily at the interface between the low-index ambient medium and the dielectric. In EXAMPLE 10, the lateral surface of the pedestal forms an acute angle with the bottom surface to enhance mode confinement. EXAMPLE 11 specifies air as the ambient medium. EXAMPLE 12 describes a substrate of silicon with a SiO₂ buffer and a Si₃N₄ waveguide. EXAMPLE 13 specifies a 600-nm-thick Si₃N₄ layer on a 2-μm SiO₂ pedestal. EXAMPLE 14 reports optical loss below 0.1 dB/m.

EXAMPLE 15 integrates a laser source and a second optical waveguide for input/output coupling. EXAMPLE 16 describes a feedback mechanism using PDH error signal generation. EXAMPLE 17 specifies the second optical waveguide as a fiber taper evanescently coupled to the spiral. EXAMPLE 18 describes the laser source as an external cavity diode laser.

EXAMPLE 19 introduces a method employing the optical reference cavity: a laser source is operated near resonance, a portion of its output is coupled into the cavity, an error signal is generated from the reflected light, and the laser’s operating parameters are adjusted to maintain lock. EXAMPLE 20 specifies the second optical waveguide as a bus waveguide on the same chip. EXAMPLE 21 details its formation via lithography and etching. EXAMPLE 22 describes monolithic integration of the laser source.

The disclosed embodiments are not exhaustive; equivalents and combinations of features are within the scope of the claims. The grouping of features in examples is illustrative, not limiting. The subject matter encompasses all devices and methods falling within the literal or doctrine-of-equivalents scope of the claims.

References incorporated herein include those on Brillouin lasers, optical frequency combs, optical microcavities, whispering gallery modes, stimulated Brillouin scattering, slow light, optomechanical resonators, angular velocity sensing, planar Si₃N₄ ring resonators, and scattering losses. These references are related to or enabled by the disclosed subject matter and are incorporated by reference for background and technical context. The Abstract serves as an aid to searching patent literature and encompasses the core inventive concept without limitation.