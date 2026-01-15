# Patent Application

## Title: Method for Direct Growth of Single-Crystalline III–V Semiconductors on Amorphous Substrates

## Background

### Field of the Invention
The present invention relates to a method for directly growing single-crystalline III–V semiconductors on amorphous substrates. This technique eliminates the need for lattice-matched substrates, enabling ubiquitous integration of high-quality III–V semiconductors for various applications.

### Description of Related Art
Conventional methods for growing III–V semiconductors require lattice-matched substrates to ensure crystal quality. However, this limitation restricts the choice of substrates and increases manufacturing costs. There is a need for a method that can grow high-quality single-crystalline III–V semiconductors on amorphous substrates.

## Summary of the Invention
The invention provides a method for directly growing optoelectronic-quality single-crystalline III–V semiconductors on amorphous substrates. This method uses thermal liquid phase epitaxy (TLP) to confine the growth of III–V materials, ensuring that the resulting crystals match the shape and quality of the original patterns.

## Detailed Description

### Patterning and Growth of InP
First, a clean Si wafer with a 50-nm thick thermal oxide is lithographically patterned with the desired InP shape. A thin 1–10 nm thick MoOx layer is evaporated, followed by evaporation of In of the desired thickness and a 10–100-nm thick SiOx layer. The substrate chuck is cooled to <150 K using liquid N2 to obtain a smooth In film. After liftoff, angled evaporation coats the exposed side regions of the In with SiOx.

### Growth Process
The growth process involves placing the patterned wafer in a hot-wall CVD tube furnace or a cold-wall CVD system. 10% PH3 in H2 is used as the phosphorous source and diluted to the desired concentration. The samples are grown for 10–20 min at pressures of 100–300 Torr and growth temperatures ranging between 500 and 535 °C. For doping studies, 10% GeH4 in H2 is used as the Ge dopant source.

### EBSD Characterization
EBSD characterization is performed using an FEI Quanta SEM with an Oxford Instruments EBSD detector. Analysis of the maps is done using Oxford Aztec and Tango software programs. Orientation maps are generated and plotted using the inverse pole figure color scheme. Twin boundary removal is achieved by ignoring <111> 60° rotational boundaries within the crystals.

### Photoluminescence Spectra and Imaging
Photoluminescence spectra are taken using a HORIBA LabRAM HR800 tool with a 532 nm excitation wavelength. For photoluminescence imaging, a red LED is used as the excitation light source, and images are captured by an Andor silicon CCD camera through an optical microscope with a GaAs wafer to filter out irradiation wavelengths.

### Electron Concentration Extraction
The electron concentration, n, can be approximated using the equation:
\[ n \approx 2.6 \times 10^{17} \left( \frac{\Delta E}{\text{eV}} \right)^3 \]
where ΔE is the shift of the photoluminescence peak energy from an undoped reference (1.34 eV).

### Urbach Tail Fitting
The absorption at the band edge is related to the photoluminescence spectra by the van Roosbroeck–Schockley equation:
\[ \alpha(hv) = A \exp\left(\frac{E_g - hv}{E_0}\right) \]
where \( E_0 \), the slope at the absorption band edge, is the Urbach tail parameter.

### Device Fabrication
InP microwires with dimensions of 1 × 50 μm and thickness of 125 nm are grown using TLP crystal growth. Photolithography defines the source/drain contacts, followed by evaporation of 3/10/40 nm of Ge/Au/Ni and liftoff. The source/drain contacts are annealed at 375 °C for 5 min to improve contact resistance. 10 nm of ZrO2 is deposited via atomic layer deposition, followed by photolithography to define the gate electrode.

### Sentaurus Simulations
Detailed semi-classical drift-diffusion simulations are carried out using the Sentaurus Device simulator to model device performance. The parameter extraction involves matching the subthreshold region utilizing InP/ZrO2 surface interface traps and gate work function as fitting parameters. The mobility and series resistance of the device are extracted by minimizing the least squares error for all IDS-VDS curves.

## Claims
1. A method for directly growing single-crystalline III–V semiconductors on amorphous substrates, comprising:
   - Providing a clean Si wafer with a 50-nm thick thermal oxide.
   - Lithographically patterning the wafer with desired InP shapes.
   - Evaporating a thin MoOx layer and In of desired thickness.
   - Coating the exposed side regions of the In with SiOx.
   - Performing TLP growth in a CVD furnace using 10% PH3 in H2 as the phosphorous source.

2. The method of claim 1, further comprising doping the III–V semiconductor using 10% GeH4 in H2 as the dopant source.

3. The method of claim 1, wherein EBSD characterization is performed to analyze crystal orientation and quality.

4. The method of claim 1, further comprising photoluminescence spectroscopy to measure electron concentration and Urbach tail parameter.

5. A device fabricated using the III–V semiconductor grown by the method of claim 1, comprising:
   - InP microwires with defined source/drain contacts.
   - A ZrO2 gate dielectric.
   - A top gate electrode for operation as a MOSFET or photo-MOSFET.

## Conclusion
The invention provides a robust and scalable method for directly growing single-crystalline III–V semiconductors on amorphous substrates. This technique opens new avenues for integrating high-quality III–V materials in various applications, including optoelectronics and photovoltaics.