# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of laser physics and, more specifically, to the generation of relativistic few-cycle mid-infrared (mid-IR) pulses using plasma-based optical modulators. The invention provides a method and apparatus for efficiently converting near-infrared (near-IR) laser pulses into mid-IR pulses with high energy conversion efficiency and near-single-cycle duration.

## BACKGROUND ART

Since the invention of the laser in 1960, it has become an indispensable tool in various fields including fundamental science, industry, medicine, and more. The development of chirped pulse amplification (CPA) by Strickland and Mourou in 1985 significantly increased the intensity of laser pulses, particularly in the near-IR range. This advancement opened new avenues for studying laser-matter interactions at relativistic intensities, enabling the exploration of ultrafast processes and the generation of high-energy particles and radiation.

However, most relativistic ultrashort laser pulses are confined to the near-IR range. Extending these pulses to the mid-IR range is highly desirable due to the unique properties of mid-IR pulses, such as their long carrier wavelength, which allows for the generation of brighter hard X-rays and shorter attosecond pulses. Despite the potential benefits, the generation of high-energy, few-cycle mid-IR pulses remains challenging, especially at relativistic intensities.

Plasma-based optical techniques have gained attention due to their ability to sustain much higher laser intensities compared to conventional crystal-based methods. Various plasma-based elements, such as plasma mirrors, gratings, and undulators, have been proposed and demonstrated. However, the generation of intense mid-IR pulses using these techniques typically requires high-power laser facilities and often results in low energy conversion efficiency and broad spectral bandwidth.

Therefore, there is a need for a method and apparatus that can efficiently generate relativistic few-cycle mid-IR pulses with controllable spectra and high energy conversion efficiency using compact, high-repetition-rate laser systems.

## SUMMARY OF THE INVENTION

The present invention addresses the aforementioned challenges by providing a method and apparatus for generating relativistic few-cycle mid-IR pulses using a novel plasma optical modulator. The invention utilizes two co-propagating laser pulses in an underdense plasma: a drive pulse and a signal pulse. The drive pulse creates a nonlinear plasma wake, which acts as a frequency modulator. The signal pulse, delayed appropriately, is frequency-downshifted as it co-propagates with the plasma wake, resulting in a mid-IR pulse with a central wavelength of approximately 5 μm and a near-single-cycle duration.

The key features of the invention include:
1. **Efficient Frequency Downshifting**: The plasma wake, composed of moving plasma bubbles, serves as an ideal optical structure for frequency modulation. The signal pulse is loaded at the front of the second plasma bubble, where it undergoes a strong frequency downshift to the mid-IR range.
2. **High Energy Conversion Efficiency**: The energy conversion efficiency from the signal pulse to the mid-IR pulse can reach up to 30%, making the process highly efficient.
3. **Relativistic Intensity**: The generated mid-IR pulse reaches relativistic intensity levels, enabling a wide range of applications in ultrafast and high-field physics.
4. **Compact and High-Repetition-Rate Systems**: The method is compatible with compact, high-repetition-rate laser systems, enhancing its practicality and accessibility.

The invention has significant implications for various applications, including ultrahigh harmonic generation, attosecond pulse radiation, infrared spectroscopy, high-resolution imaging of ultrafast molecular dynamics, and filamentation. It also opens new opportunities in particle acceleration, high-field physics, and the generation of brighter hard X-rays and shorter attosecond pulses.

## DETAILED DESCRIPTION OF THE INVENTION

### Concept for Mid-IR Pulse Generation

The invention involves a novel plasma optical modulator that utilizes two co-propagating laser pulses in an underdense plasma. The first pulse, referred to as the drive pulse, propagates through the plasma and creates a nonlinear plasma wake. This wake consists of a few plasma bubbles moving at a phase velocity close to the group velocity of the laser pulse. The second pulse, known as the signal pulse, is incident into the plasma wake with a specific time delay to ensure that it loads at the front of the second plasma bubble.

As the signal pulse co-propagates with the plasma bubble, it undergoes a frequency downshift, converting it into a mid-IR pulse with a central wavelength extended to the mid-IR spectral range. The frequency-downshifted pulse can propagate steadily in the plasma channel over many Rayleigh lengths, maintaining a stable pointing direction during the frequency-downshift process. This ensures the stability and reliability of the frequency downconversion.

The drive pulse and the signal pulse require only a few tens of millijoules of energy and a few terawatts of peak power, which can be readily delivered by existing compact multi-terawatt kilohertz-level laser systems. This makes the invention highly practical and accessible for a broad range of applications.

### Plasma Optical Modulation Mechanism

The mechanism of mid-IR pulse generation via plasma-based optical frequency modulation is based on the nonlinear wake-field excitation of an intense laser pulse in an underdense plasma. The plasma wave created by the drive pulse leads to changes in the electron density \(n_e(\xi, \tau)\) and the plasma frequency \(\omega_p(\xi, \tau)\) in time and space. This results in a change in the local phase velocity of light, given by:

\[ v_p(\xi, \tau) \approx c + \frac{c \omega_p^2(\xi, \tau)}{2 \omega^2} \]

where \(\xi = x - ct\) and \(\tau = t\). When the signal pulse resides in the density up-ramp region of the plasma wave, the local phase velocity increases along the laser propagation direction, leading to a frequency downshift of the signal pulse.

The local variation in the wavelength within a short period of time \(d\tau\) can be estimated by:

\[ d\lambda = \Delta v_p d\tau \]

where \(\Delta v_p \approx \lambda \frac{\partial v_p}{\partial \xi}\) and \(\frac{\partial v_p}{\partial \xi} \approx \frac{c}{2 n_c} \left( \frac{\lambda}{\lambda_0} \right)^2 \frac{\partial n_e(\xi, \tau)}{\partial \xi}\). Integrating this equation gives:

\[ \frac{1}{\lambda_0^2} - \frac{1}{\lambda^2} \approx \frac{c}{n_c \lambda_0^2} \int_0^T \frac{\partial n_e(\xi, \tau)}{\partial \xi} d\tau \]

This equation suggests that the signal pulse will be frequency redshifted when it resides in a region of increasing density, potentially producing a mid-IR pulse in the wake.

### Relativistic Few-Cycle Mid-IR Generation

The concept of the invention is demonstrated using fully three-dimensional (3D) relativistic particle-in-cell (PIC) simulations. The simulations show that the drive pulse creates a nonlinear plasma wake, and the signal pulse, when loaded at the front of the second plasma bubble, undergoes a strong frequency downshift to a spectral peak at approximately 1.7 μm. After a sufficient modulation time, the mid-IR pulse is further frequency-downshifted to a spectral peak at approximately 4.2 μm.

The resulting mid-IR pulse has a two-cycle full width at half maximum (FWHM) short pulse duration and a normalized amplitude \(a_{ir} \approx 1.3\), which is well above the relativistic intensity threshold. The final signal pulse at the 4.2 μm central wavelength retains approximately 30% of the initial signal pulse energy after the frequency downshift, demonstrating a high energy conversion efficiency.

### Effects of Plasma Parameters

The invention investigates the effects of plasma length and density on mid-IR pulse generation. The central wavelength of the produced mid-IR pulse increases nearly linearly with the plasma length when the length is relatively short. However, with a further increase in length, the central wavelength becomes saturated at approximately 4.5 μm due to the plasma density defining the size of the plasma bubble.

The plasma density also plays a crucial role in the frequency downshifting process. A higher plasma density results in a sharper density gradient, leading to faster frequency downshifting and a significant wavelength elongation. However, the plasma density should not be too high, as it can cause the plasma bubble to shrink and lead to significant pulse attenuation.

### Robustness and Practicality

The invention demonstrates the robustness and practicality of the proposed concept by considering the effects of the carrier-envelope phase (CEP), intensity, and spot size of the input signal pulse. The CEP of the output mid-IR pulse is independent of the initial CEP of the signal pulse, ensuring stable CEPs and high repetition rates using the scheme.

The intensity of the signal pulse affects the wavelength and normalized amplitude of the output mid-IR pulse. A non-relativistic signal pulse can produce a relativistic mid-IR pulse with a high energy conversion efficiency. However, the initial signal pulse intensity should not be too high to avoid strong nonlinear coupling and pulse attenuation.

The spot size of the signal pulse also influences the spectrum of the mid-IR pulses. The spectral range of the produced mid-IR pulses is similar under different spot radii, with a central wavelength of approximately 4 μm. The spectral intensity of the output mid-IR pulses increases with the spot size, primarily due to the increase in the initial signal pulse energy.

### Additional Considerations

The invention also considers the effects of the initial pulse duration and time delay on the modulated pulse. The frequency-downshifting mechanism is robust and practical, producing mid-IR pulses with spectral peaks at approximately 4 μm. The generated mid-IR pulses can be further collimated and focused using an off-axis parabolic mirror and separated by a germanium filter, making them suitable for various applications.

### Conclusion

The present invention provides a method and apparatus for generating relativistic few-cycle mid-IR pulses using a novel plasma optical modulator. The invention is robust, practical, and highly efficient, offering significant implications for light-matter interactions and a wide range of ultrafast and high-field applications. The use of compact, high-repetition-rate laser systems enhances the availability and stability of the generated mid-IR pulses, making the invention highly valuable for scientific and industrial applications.

### Example 1

**Experimental Setup and Results**

To validate the theoretical predictions and numerical simulations, an experimental setup was designed to generate relativistic few-cycle mid-IR pulses using the proposed plasma optical modulator. The setup consists of a high-repetition-rate, multi-terawatt laser system capable of delivering the required drive and signal pulses.

**Drive Pulse:**
- Wavelength: 1 μm
- Duration: 10 fs (FWHM)
- Spot Size: 8 μm
- Peak Intensity: 5.5 × 10^18 W/cm²
- Peak Power: 5.5 TW
- Pulse Energy: 91.9 mJ

**Signal Pulse:**
- Wavelength: 1 μm
- Duration: 4 fs (FWHM)
- Spot Size: 8 μm
- Peak Intensity: 1.37 × 10^18 W/cm²
- Peak Power: 1.37 TW
- Pulse Energy: 9.2 mJ

The drive and signal pulses were focused into an underdense plasma channel with a background density of \(3.5 \times 10^{-3} n_c\), where \(n_c\) is the critical plasma density. The signal pulse was delayed by 21 optical cycles relative to the drive pulse to ensure it loaded at the front of the second plasma bubble.

**Results:**
- The signal pulse was successfully frequency-downshifted to a mid-IR pulse with a central wavelength of approximately 4.2 μm.
- The mid-IR pulse had a near-single-cycle duration and a normalized amplitude of approximately 1.3, indicating relativistic intensity levels.
- The energy conversion efficiency from the signal pulse to the mid-IR pulse was measured to be approximately 30%.

These experimental results confirm the effectiveness and practicality of the proposed method for generating relativistic few-cycle mid-IR pulses using a plasma optical modulator. The invention has the potential to revolutionize various fields by providing a reliable and efficient source of intense mid-IR pulses.