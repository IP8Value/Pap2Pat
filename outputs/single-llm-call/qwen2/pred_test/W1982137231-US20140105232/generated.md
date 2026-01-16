# DESCRIPTION

## BENEFIT CLAIMS TO RELATED APPLICATIONS

This application claims the benefit of U.S. Provisional Application No. 63/XXXXXXX, filed [Date], entitled "Spiral Resonators for On-Chip Laser Frequency Stabilization," the disclosure of which is hereby incorporated by reference in its entirety.

## BACKGROUND

Laser frequency stabilization is a critical requirement in various scientific and technological applications, including precision spectroscopy, optical clocks, and coherent communication systems. The stability of a laser locked to a reference cavity is influenced by the quality factor (Q) of the cavity and the signal-to-noise ratio (SNR) of the detected laser signal. However, technical noise sources such as acceleration, acoustics, and thermorefractive noise can significantly degrade the stability of the laser. Traditional reference cavities, such as those made from bulk materials, often suffer from limitations in terms of size, cost, and integration with other components.

Recent advancements in photonic integrated circuits (PICs) have opened new avenues for developing compact, high-performance reference cavities. One promising approach involves the use of spiral resonators fabricated on silicon chips. These resonators offer high Q factors and large mode volumes, which can significantly reduce thermorefractive and photo-thermal noise. Additionally, the compact nature of these devices makes them suitable for integration into chip-scale systems, enabling a wide range of applications.

## SUMMARY

The present invention relates to spiral resonators for on-chip laser frequency stabilization. Specifically, the invention provides a method and apparatus for fabricating and utilizing spiral resonators that exhibit high Q factors and large mode volumes, thereby reducing thermorefractive and photo-thermal noise. The spiral resonators are designed to be integrated into photonic integrated circuits (PICs) and can be used to stabilize laser frequencies with high precision and stability.

In one embodiment, the invention provides a spiral resonator comprising a silicon chip, a spiral waveguide formed on the silicon chip, and a fibre taper for coupling light into and out of the spiral waveguide. The spiral waveguide is designed to have a high Q factor and a large mode volume, which minimizes the impact of thermorefractive and photo-thermal noise. The spiral resonator can be used to lock a laser to a stable frequency reference, thereby improving the stability and coherence of the laser.

In another embodiment, the invention provides a method for fabricating a spiral resonator. The method includes the steps of depositing a low-loss waveguide material on a silicon substrate, patterning the waveguide material to form a spiral waveguide, and forming a fibre taper for coupling light into and out of the spiral waveguide. The method can be used to produce spiral resonators with Q factors exceeding 100 million and mode volumes suitable for reducing thermorefractive and photo-thermal noise.

The invention also provides a system for laser frequency stabilization. The system includes a laser, a spiral resonator as described above, and a feedback control system for locking the laser to the spiral resonator. The feedback control system can use techniques such as Pound-Drever-Hall (PDH) locking to achieve high stability and coherence in the laser output.

## DETAILED DESCRIPTION OF EMBODIMENTS

### EXAMPLE 1

In a first embodiment, a spiral resonator is fabricated on a silicon chip using a low-loss waveguide material. The waveguide material is deposited on a silicon substrate using a process such as plasma-enhanced chemical vapor deposition (PECVD) or flame hydrolysis. The waveguide material is then patterned using photolithography and etching techniques to form a spiral waveguide with a round-trip physical path length of 120 cm. The spiral waveguide is designed to have a high Q factor and a large mode volume, which reduces the impact of thermorefractive and photo-thermal noise.

A fibre taper is formed at the upper-right corner of the chip to couple light into and out of the spiral waveguide. The fibre taper is aligned with the waveguide using a precision alignment system to ensure efficient coupling. The Q factor of the resonator is measured by monitoring the transmitted optical power on the fibre taper while scanning an external cavity semiconductor laser across a free-spectral range (FSR) of the resonator. The measured FSR of the device is 173 MHz, which agrees well with the expected FSR based on the round-trip length. The resonator exhibits a Q factor of 140 million, demonstrating the high performance of the spiral resonator.

### EXAMPLE 2

In a second embodiment, the spiral resonator is used to stabilize a fibre laser. The fibre laser is locked to the spiral resonator using a Pound-Drever-Hall (PDH) locking system. The PDH locking system includes a phase modulator, a photodetector, and a feedback control loop. The phase modulator introduces a small frequency modulation to the laser output, which is detected by the photodetector. The feedback control loop adjusts the laser frequency to maintain resonance with the spiral resonator.

The phase-noise spectral density function of the stabilized laser is measured using an electrical spectrum analyzer and a phase-noise analyzer. The measurements show a significant reduction in phase noise compared to the free-running laser. Specifically, the phase noise is suppressed by an average of 26 dB within the bandwidth of the feedback control system. The Allan deviation of the stabilized laser is also measured using a frequency counter, and a minimum relative Allan deviation of 5.5 × 10^-13 is observed at a gate time of 400 μs. These results demonstrate the high stability and coherence of the laser when locked to the spiral resonator.

### EXAMPLE 3

In a third embodiment, the spiral resonator is used to compare the performance of different resonator designs. The spiral resonator is compared to conventional disk resonators of varying diameters (3 mm, 7.5 mm, and 15 mm). The phase-noise spectral density function is measured for each resonator, and the results show that the spiral resonator provides better noise suppression at lower offset frequencies. Specifically, the 3 mm disk resonator exhibits a degradation in phase noise at offset frequencies less than 1 kHz, which is consistent with the thermal corner frequency observed in other silica-based resonators. In contrast, the spiral resonator maintains low phase noise at these frequencies, indicating better immunity to photo-thermal noise.

### EXAMPLE 4

In a fourth embodiment, the spiral resonator is used to measure the effect of thermo-mechanical noise. The optomechanical coupling parameter is expected to vary inversely with cavity length, leading to an inverse quadratic dependence of phase noise on length. This dependence is observed over a range of cavity lengths using the Hänsch Couillard technique. Spectral features believed to be thermally excited mechanical resonances are observed at offset frequencies greater than 1 MHz, and these features diminish in amplitude for the largest spirals measured (1.2 m path length). The results confirm that the spiral resonator provides enhanced immunity to thermo-mechanical noise.

### EXAMPLE 5

In a fifth embodiment, the spiral resonator is used to demonstrate the potential for chip-integrated frequency synthesis. The spiral resonator is combined with a microcomb generator to produce a broadband, low-phase-noise signal. The microcomb generator is used to generate a comb of equally spaced optical frequencies, which are then divided down to the radio frequency (RF) domain using electronic division. The resulting RF signal exhibits close-to-carrier phase noise of ~-100/f^3 dBc Hz^-1, which is competitive with state-of-the-art oven-controlled crystal oscillators. This demonstrates the potential for the spiral resonator to serve as a frequency reference in chip-integrated systems.

### EXAMPLE 6

In a sixth embodiment, the spiral resonator is used to improve the coherence of a fibre laser for coherent communication systems. The fibre laser is locked to the spiral resonator using a PDH locking system, and the coherence of the laser is measured using an interferometer. The results show a significant improvement in coherence compared to the free-running laser, with a reduction in the effective linewidth by a factor of 10. This improvement in coherence is beneficial for applications such as coherent fibre-optic communications, where high coherence is required to achieve high data rates and long transmission distances.

### EXAMPLE 7

In a seventh embodiment, the spiral resonator is used to enhance the stability of a laser for remote sensing applications. The laser is locked to the spiral resonator using a PDH locking system, and the stability of the laser is measured using a frequency counter. The results show a significant improvement in stability compared to the free-running laser, with a minimum relative Allan deviation of 5.5 × 10^-13 at a gate time of 400 μs. This improvement in stability is beneficial for applications such as remote sensing, where high stability is required to achieve accurate measurements.

### EXAMPLE 8

In an eighth embodiment, the spiral resonator is used to improve the coherence of a laser for atomic physics experiments. The laser is locked to the spiral resonator using a PDH locking system, and the coherence of the laser is measured using an interferometer. The results show a significant improvement in coherence compared to the free-running laser, with a reduction in the effective linewidth by a factor of 10. This improvement in coherence is beneficial for applications such as atomic physics, where high coherence is required to achieve precise measurements of atomic transitions.

### EXAMPLE 9

In a ninth embodiment, the spiral resonator is used to demonstrate the potential for further performance improvements. The waveguide material is deposited using a flame hydrolysis process, which allows for the formation of thicker oxides. The thicker oxides increase the mode volume of the resonator, leading to a 1,000-fold increase in mode volume relative to the current results. The increased mode volume further reduces the impact of thermorefractive and photo-thermal noise, leading to even higher stability and coherence in the laser output. This demonstrates the potential for the spiral resonator to achieve state-of-the-art performance in laser frequency stabilization.

## CONCLUSION

The present invention provides a novel approach to laser frequency stabilization using spiral resonators fabricated on silicon chips. The spiral resonators exhibit high Q factors and large mode volumes, which reduce the impact of thermorefractive and photo-thermal noise. The invention can be used to stabilize laser frequencies with high precision and stability, making it suitable for a wide range of applications, including precision spectroscopy, optical clocks, coherent communication systems, remote sensing, and atomic physics. The compact and integrable nature of the spiral resonators also makes them attractive for chip-scale systems, enabling the development of advanced photonic integrated circuits (PICs).