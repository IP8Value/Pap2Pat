# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to high voltage thin film transistors (HVTFTs) and, more specifically, to HVTFTs with improved thermal stability and high voltage blocking capability. The invention is particularly useful in applications requiring reliable high voltage operation, such as photovoltaic (PV) inverters in building-integrated photovoltaics (BIPV) systems and self-powered smart glass technologies.

## BACKGROUND OF THE INVENTION

High voltage thin film transistors (HVTFTs) are essential components in various electronic devices, especially those requiring high voltage and high power applications. Traditional HVTFTs often suffer from limitations such as non-uniform electrical field distribution, poor thermal stability, and insufficient blocking voltage. These issues can lead to device failure and reduced performance, making them unsuitable for advanced applications like PV inverters in BIPV systems.

One common design for HVTFTs is the rectangular channel structure, which introduces non-uniform electrical field distribution with the highest field located at the corners of the channel. This non-uniformity limits the blocking voltage of the devices, thereby reducing their reliability and efficiency. Additionally, the thermal stability of HVTFTs is crucial for high voltage applications, as temperature variations can significantly affect the device's performance and lifespan.

To address these challenges, researchers have explored various materials and design modifications. For instance, doping ZnO with a small amount of Mg to form Mg0.03Zn0.97O (MZO) has been shown to enhance thermal stability. However, further improvements are needed to achieve the desired performance levels for high voltage applications.

## SUMMARY

The present invention provides a high voltage thin film transistor (HVTFT) with improved thermal stability and high voltage blocking capability. The HVTFT includes a ring structure design to reduce the electric field crowding effect, a Mg0.03Zn0.97O (MZO) channel layer to enhance thermal stability, and a modulation-doped ultra-thin MZO transition layer (MZO-TL) to improve the interface properties between the channel and the gate dielectric layer.

The ring structure design ensures a more uniform electrical field distribution, reducing the risk of device failure due to high field concentrations. The MZO channel layer, doped with a small amount of Mg, significantly enhances the thermal stability of the HVTFT, making it suitable for high temperature environments. The modulation-doped MZO-TL acts as a diffusion barrier, preventing the interdiffusion of Zn and Si across the interface, which reduces interface trap density and trapped charges. This improvement leads to a steeper subthreshold slope, higher on-current, and increased blocking voltage.

The HVTFT of the present invention can operate at high drain bias conditions with a high on/off ratio and stable performance, making it ideal for use in PV inverters in BIPV systems and self-powered smart glass technologies.

## DETAILED DESCRIPTION OF THE INVENTION

### Material Preparation and Device Fabrication Process

The HVTFTs of the present invention are fabricated on 0.4 mm thick commercial glass substrates. The fabrication process involves several steps to ensure the desired material and structural properties. Initially, a 50 nm chromium (Cr) layer is deposited by sputtering and patterned using a dry etching process to serve as the bottom gate electrode. Subsequently, a 200 nm SiO2 layer is deposited by plasma-enhanced chemical vapor deposition (PECVD) as the gate dielectric layer.

The channel layer is then deposited using metal-organic chemical vapor deposition (MOCVD) at 400°C. Three types of channel layers, each with a thickness of 50 nm, are deposited on the SiO2 layer: (i) pure ZnO, (ii) Mg0.03Zn0.97O (MZO), and (iii) modulation-doped Mg0.03Zn0.97O (m-MZO). In the m-MZO HVTFT, a 10 nm modulation-doped MgyZn1−yO transition layer (MZO-TL) is inserted between the MZO channel layer and the SiO2 dielectric layer. The Mg composition (y) in the MZO-TL decreases from the side adjacent to SiO2 (y = 1) to the other side adjacent to the channel (y = 0.03).

The source and drain metallization, consisting of 100 nm titanium (Ti) and 50 nm gold (Au), is deposited using electron beam evaporation, followed by a normal lift-off process. A photoresist film is coated on top of the TFT channel to serve as a passivation layer, preventing ambient absorption/desorption during electrical testing. HVTFTs with three different channel lengths are fabricated, with the channel lengths/gate-to-drain offset lengths being 10/5 μm, 15/10 μm, and 25/20 μm for nominal, longer, and longest HVTFTs, respectively. The gate-to-source offset is kept the same at 3 μm.

### Device Testing Conditions

The electrical measurements under low bias conditions are conducted using an HP-4156C with an HP-41501B Pulse Generator. The maximum voltage of the HP-4156C electrical testing system is limited to 200 V, and the system has a current resolution of 1 × 10−15 A. This setup is used for all transfer characteristic measurements. For electrical measurements under high bias conditions, a high voltage testing system is built based on a Tektronix 370 with the probe station. The current resolution of the Tektronix 370 is 1 × 10−6 A, and it is used primarily for testing blocking voltages. To avoid problems with arcing and tracking due to environmental conditions, the devices are immersed in Fluorinert FC-40 during high voltage measurements. The electrical measurements at different temperatures are conducted using an Agilent 1500B, and all measurements are performed in a light-tight probe station.

### Material Characterizations

The structural and interfacial properties of the HVTFTs are analyzed using various techniques. Transmission electron microscopy (TEM) and energy-dispersive X-ray spectroscopy (EDS) are employed to study the cross-sectional images and elemental composition of the interface regions. X-ray photoelectron spectroscopy (XPS) is used to estimate the atomic percentages of different elements in the interface regions, and depth profiles are obtained using in-situ sputtering processes.

### Device Design

The HVTFT of the present invention features a ring structure design to reduce the electric field crowding effect. The ring structure ensures a more uniform electrical field distribution from drain to source, with the highest field being approximately 50% less than in the rectangular counterpart. The HVTFT has a bottom gate inverted-staggered configuration and includes two offset regions: gate to drain and gate to source. The MZO channel layer, doped with a small amount of Mg, enhances the thermal stability of the device. The modulation-doped MZO-TL inserted between the MZO channel layer and the SiO2 dielectric layer acts as a diffusion barrier, preventing the interdiffusion of Zn and Si across the interface.

### Transfer Characteristics and Thermal Stability

The transfer characteristics of the HVTFTs with different channel materials and structures are evaluated to assess their performance. Compared to pure ZnO HVTFTs, MZO HVTFTs exhibit a better subthreshold slope (S.S.) and on-current. The thermal stability of the MZO HVTFT is significantly improved, with a threshold voltage shift (∆Vth) of −6 V at temperatures increasing from 294 K to 367 K, compared to a shift of −10.5 V in pure ZnO HVTFTs. The activation energy of drain currents is extracted from Arrhenius plots, indicating that the MZO HVTFT has a lower trap density and better thermal stability.

When comparing MZO HVTFTs with m-MZO HVTFTs, the latter shows an order of magnitude higher on-current and a steeper S.S. The activation energy of drain currents in m-MZO HVTFTs suggests a nearly 40% lower total trap density, primarily due to the reduction in interface trap density achieved through the modulation-doped MZO-TL.

### High Voltage Blocking Capability

The high voltage blocking capability of the HVTFTs is a critical parameter for their application in high voltage devices. The MZO HVTFT fails to block high voltages, with the drain leakage current increasing abruptly and the device burning down at VDS = 90 V. In contrast, the m-MZO HVTFT maintains a low drain leakage current of 10−12 A even at VDS = 200 V, demonstrating superior blocking capability.

The trade-off between blocking capability and driving capability is observed in m-MZO HVTFTs with different channel lengths. As the channel length increases, the blocking voltage increases, but the on-current decreases. The nominal (L = 10 μm), longer (L = 15 μm), and longest (L = 25 μm) m-MZO HVTFTs have blocking voltages of 300 V, 427 V, and 609 V, respectively, with corresponding on-currents of 3.5 × 10−5 A, 6.61 × 10−6 A, and 4.57 × 10−6 A, respectively. The m-MZO HVTFT with a channel length of 25 μm can operate at a drain bias of 200 V with a blocking capability over 600 V, making it suitable for use in PV inverters in BIPV systems.

### Statistical Data of Electrical Performance

Statistical data of the electrical performance of the HVTFTs are collected to ensure consistency and reliability. The transfer characteristics at normal bias (drain bias = 10 V) among three m-MZO HVTFTs with different channel lengths show a trade-off between blocking capability and driving capability. The output characteristics of the m-MZO HVTFT with a channel length of 10 μm demonstrate better saturation behavior at low gate bias and a kink effect at high gate bias, which may be related to the channel length modulation-induced self-heating effect.

### Simulation of Electrical Field Distribution

To understand the fundamental cause of the improvement in blocking voltage, simulations of the electrical field distribution in the HVTFTs are conducted using SILVACO software. The simulations reveal that the MZO HVTFT possesses extra positive oxide charges in comparison to the m-MZO HVTFT. The reduction of the maximum electrical field near the interface in the m-MZO HVTFT allows it to operate at higher drain bias, enabling higher blocking voltage. The MZO-TL in the m-MZO HVTFT acts as a barrier against Zn diffusion, reducing interface states and trapped charges, which leads to a decrease in the maximum electric field in the channel.

In summary, the HVTFT of the present invention, featuring a ring structure design, MZO channel layer, and modulation-doped MZO-TL, demonstrates improved thermal stability, high voltage blocking capability, and reliable performance under high bias conditions. These features make the HVTFT suitable for advanced applications in PV inverters in BIPV systems and self-powered smart glass technologies.