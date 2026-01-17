# DESCRIPTION

## BACKGROUND

The field of precision measurement has seen significant advancements in recent years, particularly in the areas of gravimetry and inertial sensing. Traditional methods for measuring gravitational acceleration and inertial forces often rely on mechanical sensors, which can be susceptible to environmental disturbances such as ambient vibrations. These disturbances can significantly degrade the accuracy and stability of the measurements, making it challenging to achieve high precision in real-world applications.

One of the most promising approaches to overcoming these limitations is the use of atom interferometry. Atom interferometers leverage the quantum properties of atoms to achieve extremely high sensitivity and precision in measuring gravitational and inertial forces. However, even atom interferometers are not immune to the effects of ambient vibrations. When the phase reference for the interferometer is disturbed by vibrations, the resulting phase shifts can exceed the 2π phase ambiguity, leading to random readouts and a loss of coherence in the interference pattern.

To address this issue, researchers have explored various methods to mitigate the impact of ambient vibrations. One such method involves the use of optomechanical resonators, which can detect and correct for the phase shifts induced by vibrations. By combining an atom interferometer with an optomechanical resonator, it is possible to achieve a significant improvement in the stability and accuracy of the measurements.

This invention describes a novel method for enhancing the performance of an atom interferometer by integrating it with an optomechanical resonator. The optomechanical resonator serves as a high-sensitivity accelerometer that can detect the vibrations affecting the interferometer. By applying the detected vibrations to correct the phase shifts in the atom interferometer, the overall stability and precision of the measurements are significantly improved.

## SUMMARY OF THE EMBODIMENTS

The present invention relates to a method and apparatus for enhancing the performance of an atom interferometer by integrating it with an optomechanical resonator. The optomechanical resonator is used to detect and correct for the phase shifts induced by ambient vibrations, thereby improving the stability and precision of the atom interferometer.

In one embodiment, the invention includes an atom interferometer configured to measure gravitational acceleration or other inertial forces. The atom interferometer operates by splitting, redirecting, and recombining matter waves of atoms using stimulated two-photon Raman transitions. The phase shift induced by the acceleration of the atoms is measured to determine the gravitational acceleration.

The invention further includes an optomechanical resonator attached to the mirror providing the phase reference for the atom interferometer. The optomechanical resonator is designed to detect the vibrations affecting the mirror and, consequently, the phase reference of the atom interferometer. The resonator is capable of measuring the displacement of the test mass caused by the vibrations and converting this displacement into an acceleration signal.

The acceleration signal from the optomechanical resonator is then used to correct the phase shifts in the atom interferometer. This is achieved by applying the detected acceleration to the phase correction algorithm of the atom interferometer. By doing so, the phase shifts induced by the vibrations are compensated, and the coherence of the interference pattern is maintained.

The invention provides several advantages over existing methods. First, it significantly improves the short-term stability of the atom interferometer by reducing the impact of ambient vibrations. Second, it allows for continuous, uninterrupted measurements over extended periods, which is otherwise impossible with traditional methods. Third, the integration of the optomechanical resonator with the atom interferometer is straightforward and can be implemented with minimal hardware changes, making it suitable for a wide range of applications.

In another embodiment, the invention can be extended to other types of atom interferometric sensors and even laser interferometers. The method is not limited to gravimetry and can be applied to any scenario where high precision and stability are required in the presence of ambient vibrations.

## DETAILED DESCRIPTION OF THE EMBODIMENTS

### Overview of the System

The invention combines an atom interferometer with an optomechanical resonator to enhance the performance of the interferometer in the presence of ambient vibrations. The atom interferometer is configured to measure gravitational acceleration or other inertial forces by splitting, redirecting, and recombining matter waves of atoms using stimulated two-photon Raman transitions. The optomechanical resonator is attached to the mirror providing the phase reference for the atom interferometer and is designed to detect the vibrations affecting the mirror. The detected vibrations are used to correct the phase shifts in the atom interferometer, thereby improving the stability and precision of the measurements.

### Atom Interferometer

The atom interferometer is based on a Kasevich-Chu interferometer, which is a type of Mach-Zehnder interferometer for atoms. In this setup, a cloud of cold atoms, typically rubidium-87 (87Rb), is subjected to a sequence of laser pulses that manipulate the atoms' internal and external degrees of freedom. The sequence consists of a π/2 pulse, followed by a π pulse, and then another π/2 pulse. These pulses split, redirect, and recombine the matter waves of the atoms, creating an interference pattern that is sensitive to the acceleration experienced by the atoms.

The phase shift induced by the acceleration of the atoms is given by the equation:

\[
\Delta \phi = k_{\text{eff}} \cdot a \cdot T
\]

where \( k_{\text{eff}} \) is the effective wavevector of the Raman transition, \( a \) is the acceleration, and \( T \) is the time between the π/2 and π pulses. The phase shift is measured by detecting the relative population of the atoms in the two output ports of the interferometer using state-selective fluorescence detection.

### Optomechanical Resonator

The optomechanical resonator is a key component of the invention, designed to detect the vibrations affecting the mirror providing the phase reference for the atom interferometer. The resonator consists of a test mass and a mirror, both of which are part of a high-finesse optical cavity. The test mass is supported by a stiff, u-shaped flexible mount, allowing it to move in response to vibrations. The mirror forms one end of the optical cavity, and the other end is formed by the flat tip of a polarization-maintaining fiber.

The resonator is operated under normal atmospheric conditions and is attached to a two-inch square mirror that retroreflects the light pulses driving the atom interferometer. The acceleration-sensitive axis of the resonator is aligned collinearly with the retroreflector's normal vector to ensure that the vibrations affecting the mirror are accurately detected.

The motion of the test mass is read out using a fiber-based optical setup. A tunable laser operating at a wavelength near 1560 nm is used to interrogate the resonator. The laser light is split into two paths using a 90:10 splitter, with the majority of the light directed towards the resonator. The light reflected off the resonator is detected using a polarizing beam splitter and two photodetectors. The differential signal from the photodetectors is used to cancel common-mode laser intensity noise and to measure the displacement of the test mass.

The displacement of the test mass is converted into an acceleration signal using the relationship:

\[
X(\omega) = \frac{A(\omega)}{\omega^2 - \omega_0^2 + i\gamma\omega}
\]

where \( X(\omega) \) is the displacement of the test mass, \( A(\omega) \) is the acceleration, \( \omega_0 \) is the resonance frequency of the resonator, and \( \gamma \) is the damping coefficient. The acceleration signal is then used to correct the phase shifts in the atom interferometer.

### Phase Correction Algorithm

The phase correction algorithm is a crucial part of the invention, responsible for applying the detected acceleration to correct the phase shifts in the atom interferometer. The algorithm operates by first filtering the acceleration signal to remove low-frequency drifts and high-frequency noise. High-pass filters at 0.8 Hz are used to suppress low-frequency drifts, and a digital low-pass filter at 50 Hz is applied to match the corner frequency of the atom interferometer.

The filtered acceleration signal is then sampled digitally over 60 ms centered around the central light pulse of each interferometer cycle. The phase correction is calculated using the acceleration sensitivity function, which describes the atom interferometer's phase response to the detected acceleration. The phase correction is applied to the interferometer signal to compensate for the phase shifts induced by the vibrations.

### Performance and Applications

The integration of the optomechanical resonator with the atom interferometer significantly improves the short-term stability of the measurements. In a 22-hour-long, interruption-free measurement series, the method reduces the ambient noise by a factor of 8, as measured by the Allan deviation. This improvement in stability reduces the measurement time necessary to achieve a target instability by a factor of 64.

The invention has a wide range of applications, including gravimetry, inertial navigation, and airborne gravimetry. The high dynamic range and large bandwidth of the hybrid sensor make it suitable for use in environments with large inertial noise, where traditional methods are limited by the need for bulky vibration isolation systems. The compact size and low power consumption of the optomechanical resonator also make it ideal for integration into portable and miniaturized sensor systems.

### Future Enhancements

Future enhancements to the invention include optimizing the performance of both the atom interferometer and the optomechanical resonator. For the atom interferometer, improvements can be made by increasing the interrogation time \( T \) to 35 ms, using higher-order Bragg processes to transfer 4 \( k_{\text{eff}} \), and reducing the phase noise to 3 mrad. For the optomechanical resonator, the resonance frequency can be tuned to 1500 Hz to increase the bandwidth, and the optical finesse can be improved to 1600 by high-reflectivity coating. These optimizations are expected to lower the intrinsic noise to 6 × 10^-8 m/s^2 Hz^-1/2, which is comparable to the noise levels achieved in quiet environments with active vibration isolation.

### Conclusion

In conclusion, the invention provides a novel method for enhancing the performance of an atom interferometer by integrating it with an optomechanical resonator. The optomechanical resonator detects and corrects for the phase shifts induced by ambient vibrations, significantly improving the stability and precision of the measurements. The invention has broad applications in gravimetry, inertial navigation, and other fields requiring high-precision measurements in the presence of environmental disturbances. The compact and low-power nature of the optomechanical resonator makes it suitable for integration into portable and miniaturized sensor systems, opening up new possibilities for practical applications.