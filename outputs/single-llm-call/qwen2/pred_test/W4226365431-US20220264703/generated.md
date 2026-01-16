# DESCRIPTION

## BACKGROUND

Temperature manipulation in biological systems is crucial for various biomedical applications, including neuroscience research, cancer hyperthermia treatment, expedited wound healing, temperature-controlled drug release, and bioanalytical techniques such as polymerase chain reaction (PCR) and temperature gradient focusing (TGF). Traditional methods of localized heating, such as ohmic heating, dielectric heating, and magnetic heating, have limitations in terms of specificity, efficiency, and spatial resolution. Ohmic heating requires direct contact between heating elements and bio-samples, dielectric heating is often unsuitable due to the similar dielectric properties of targeted cells and surrounding tissues, and magnetic heating, while more specific, faces challenges in achieving high heating efficiency and precise spatial control at the cellular level.

To address these challenges, a fully integrated magnetic microheater array with closed-loop temperature regulation is presented. This invention leverages the ferromagnetic resonance of magnetic nanoparticles (MNP) at GHz microwave frequencies, offering significant advantages over existing methods. The high operating frequency allows for reduced magnetic field strength, enabling a compact and integrated solution. Additionally, the ability to manipulate local magnetic fields using on-chip inductors results in improved spatial resolution at the sub-millimeter scale, enhancing the precision of localized heating.

## BRIEF SUMMARY OF THE INVENTION

The invention pertains to a fully integrated magnetic microheater array capable of closed-loop temperature regulation and sub-millimeter spatial resolution. The microheater array operates based on the ferromagnetic resonance of magnetic nanoparticles (MNP) at GHz microwave frequencies. Each pixel in the array consists of a stacked oscillator and an electro-thermal feedback loop. The stacked oscillator enables significantly higher magnetic field strength with a single inductor footprint, eliminating the need for additional RF amplifiers and reducing pixel area and dc power consumption. The electro-thermal feedback loop controls the biasing voltage of the tail transistor to achieve precise temperature regulation. The invention is particularly useful for biomedical applications requiring localized heating, such as cancer hyperthermia treatment and magnetogenetics.

## DETAILED DESCRIPTION

### Overview of the Invention

The invention is a fully integrated magnetic microheater array designed to achieve closed-loop temperature regulation and sub-millimeter spatial resolution. The array operates by utilizing the ferromagnetic resonance of magnetic nanoparticles (MNP) at GHz microwave frequencies. This approach offers several advantages over traditional heating methods, including reduced magnetic field strength, compact integration, and improved spatial resolution.

### Magnetic Nanoparticles (MNP) Heating Mechanisms

Magnetic nanoparticles (MNP) are microscopic magnetic materials with diameters less than 100 nm. They exhibit distinct magnetic properties compared to bulk magnetic materials and can be dispersed in biological fluids. The heating mechanisms of MNP include Néel relaxation, Brownian relaxation, and ferromagnetic resonance. Ferromagnetic resonance occurs when the frequency of the external magnetic field matches the precession frequency of the magnetic moment (Larmor frequency), leading to heat generation through the absorption and dissipation of the external magnetic field's power.

### Electro-Thermal Multiphysics Modeling

The heating process is governed by the magnetic loss equation and the heat transfer equation. These equations are coupled by the power loss term, which acts as a volumetric heat source. Numerical solutions are obtained using finite-element modeling (FEM) simulators, such as COMSOL Multiphysics, to evaluate the localized heating process. The design of the on-chip inductor is critical for determining the local magnetic field and temperature distributions. Optimal inductor geometry is determined based on the trade-off between temperature and magnetic field distribution, inductance, and quality factor.

### Integrated Microheater Array Architecture

The proof-of-concept microheater array chip consists of 12 pixels, each with a size of 0.6 × 0.7 mm². The stacked oscillators in the array are designed with different frequency tuning ranges to accommodate a wide range of MNP with varying ferromagnetic resonant frequencies. The stacked oscillators in the first three rows of the array are designed with three different frequency tuning ranges (1.2-1.6 GHz, 1.5-2.1 GHz, and 2.0-2.6 GHz, respectively). The stacked oscillators in the last row are the same as those in the second row, except their outputs are capacitively coupled to open-drain buffers for testing purposes.

### Stacked Oscillator Design

The stacked oscillator topology is designed to achieve a large RF output swing using a single inductor footprint. Multiple transistors are connected in series to distribute the voltage stress and generate a high RF output swing. The design ensures a robust oscillation startup condition and high dc-to-RF efficiency. The small-signal equivalent circuit model of the cross-coupled stacked-transistor pair is analyzed to derive the loop gain. The dc-to-RF efficiency is optimized using a load-pull methodology, considering the breakdown limit of stacked transistors and the oscillation startup condition. The final implementation of the four-stacked and five-stacked oscillators results in 45% simulated dc-to-RF efficiency.

### Frequency Tuning

Frequency tuning of the stacked oscillator is enabled by a 4-bit binary-weighted capacitor bank. The overall frequency tuning range is divided into three sub-ranges, and the stacked oscillators in different rows of the array are assigned with different sub-ranges. The capacitances of the capacitor bank and the size of the switches are optimized to ensure a constant output RF swing over the entire frequency range.

### Temperature Sensing and Control Path

The temperature sensing and control path in each pixel senses the local temperature and generates the biasing voltage for the tail transistor of the stacked oscillator for closed-loop temperature control. The first stage is a Proportional-To-Absolute-Temperature (PTAT) temperature sensor array, with diode pairs placed at the corners of the oscillator inductor to avoid sensing the ohmic loss generated by the transistors. The PTAT output is amplified and buffered to regulate the biasing voltage of the tail transistor. The temperature sensing and control path can be configured into three modes: closed-loop, open-loop, and off. The electro-thermal feedback loop is designed to ensure a large loop gain and a low impedance at the biasing voltage to filter out coupling from the strong oscillation swing.

### Measurement Results

The integrated microheater array chip is fabricated in the GlobalFoundries 45-nm CMOS SOI technology. The chip is characterized for the electrical performance of the stacked oscillator, the temperature sensing and control circuit, and a localized heating demonstration. The measured RF swing of the stacked oscillators is very close to the simulation results, and the oscillation amplitude can be backed-off by reducing the biasing voltage of the tail transistor. The temperature sensing and control path is characterized in a temperature chamber, showing good linearity and alignment with the simulation results. The localized heating performance is validated using PDMS membranes mixed with and without MNP, demonstrating efficient heating only in the area above the inductors and precise temperature regulation.

### Conclusion

This invention presents a fully integrated magnetic microheater array based on the ferromagnetic resonance of MNP at GHz microwave frequencies. The array offers the highest spatial resolution, the lowest dc power consumption, and the best dc-to-RF energy efficiency. The precise closed-loop temperature regulation makes it suitable for a wide range of biomedical applications requiring localized heating, such as cancer hyperthermia treatment and magnetogenetics. The compact and integrated design of the microheater array represents a significant advancement in the field of localized heating technology.

### Examples

#### Example 1: Cancer Hyperthermia Treatment

In a clinical setting, the magnetic microheater array can be used to treat cancer by raising the local temperature of a tumor to 43-45 °C. The high spatial resolution ensures that only the tumor is heated, minimizing damage to surrounding healthy tissues. The closed-loop temperature regulation maintains the desired temperature, enhancing the effectiveness of the treatment.

#### Example 2: Magnetogenetics

In neuroscience research, the magnetic microheater array can be used to activate thermal-sensitive ion channels in neurons. By precisely controlling the local temperature, the array can stimulate action potentials in nearby neurons, providing a powerful tool for studying neural circuits and behaviors.

#### Example 3: Wound Healing

The magnetic microheater array can be used to expedite wound healing by locally heating the wound site. The high spatial resolution ensures that only the wound area is heated, promoting faster healing and reducing the risk of infection.

These examples illustrate the versatility and potential impact of the magnetic microheater array in various biomedical applications.