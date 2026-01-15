Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FIELD  

The present invention relates generally to the field of energy harvesting, and more particularly to systems and methods for harvesting vibrational energy using piezoelectric materials. Energy harvesting involves capturing ambient energy from the environment and converting it into usable electrical energy. This technology has significant applications in powering wireless sensor networks, structural health monitoring systems, and various low-power electronic devices where battery replacement or recharging is impractical. The invention specifically addresses the challenge of efficiently harvesting vibrational energy across a range of frequencies by employing tunable piezoelectric vibration energy harvesters (PVEHs) that can actively adjust their resonant frequency to match ambient vibration conditions.  

## BACKGROUND  

Piezoelectric vibration energy harvesting has emerged as a promising approach for converting mechanical vibrations into electrical energy. Piezoelectric materials generate an electrical charge when subjected to mechanical stress, making them particularly suitable for vibration-based energy harvesting applications. However, conventional piezoelectric energy harvesters suffer from a critical limitation: they demonstrate optimal energy conversion efficiency only when operating at their natural resonant frequency. In real-world applications, ambient vibration frequencies often vary significantly over time, rendering fixed-frequency harvesters inefficient for many practical scenarios.  

Various approaches have been attempted to address this frequency-matching challenge. Passive methods, such as introducing multiple cantilevers with different natural frequencies or using nonlinear oscillators, have shown limited success in broadening the operational bandwidth. Active tuning methods that modify the mass or stiffness of the harvester typically require substantial external power input, often negating the energy benefits of harvesting. There exists a need for an energy-efficient active tuning mechanism that can dynamically adjust the resonant frequency of a piezoelectric harvester while maintaining net positive energy output.  

## SUMMARY  

The present invention discloses novel piezoelectric vibration energy harvesting (PVEH) devices and methods that overcome the limitations of conventional approaches. The disclosed systems incorporate smart materials to enable efficient frequency tuning while maintaining high energy conversion efficiency.  

The invention describes PVEH devices with tunable resonant frequencies achieved through the integration of Ionic Polymer Metal Composite (IPMC) actuators. These actuators apply controlled forces to the piezoelectric cantilever, effectively modifying its stiffness and consequently its resonant frequency. The tuning mechanism operates with minimal power consumption, ensuring that the energy expended for frequency adjustment is substantially less than the additional energy harvested through optimized frequency matching.  

Key components of the invention include a piezoelectric transducer, preferably implemented as a Macro Fiber Composite (MFC) for its flexibility and durability, mounted on a cantilever beam. The system incorporates one or more IPMC devices positioned to interact with the cantilever. When activated by a low voltage input (typically 1-4V), the IPMC devices bend and apply a controlled force to the cantilever, changing its effective stiffness and resonant frequency.  

The invention further describes stacked IPMC configurations that provide enhanced actuation capabilities. In various embodiments, multiple IPMC strips may be arranged in diamond-shaped configurations or other geometric patterns to optimize force application. The cantilever design may incorporate varying thickness along its length to facilitate effective frequency tuning.  

The system includes a sensor circuit that monitors output power and facilitates closed-loop control of the frequency tuning process. Experimental results demonstrate that the disclosed devices can achieve significant net power gains, with harvested power substantially exceeding the power required for frequency adjustment. The output power is conditioned through a full wave bridge (FWB) rectifier circuit for practical use.  

The invention encompasses multiple methods of energy harvesting, including processes for determining optimal tuning parameters, closed-loop control algorithms for autonomous frequency matching, and techniques for maximizing net energy output across varying ambient vibration conditions.  

## DETAILED DESCRIPTION  

### Introduction  

Piezoelectric vibration energy harvesters (PVEHs) represent a significant advancement in energy harvesting technology, particularly for applications requiring autonomous power sources. The disclosed systems, methods, and devices address the fundamental challenge of frequency matching in vibration energy harvesting through innovative approaches to resonant frequency tuning.  

A critical aspect of the invention involves using harvested power to actively tune the resonant frequency of the PVEH device. This self-sustaining approach ensures that the energy required for tuning is drawn from the harvested energy itself, creating an autonomous system capable of adapting to changing environmental conditions. The tuning mechanism maintains net positive energy output throughout its operation.  

The invention utilizes Macro Fiber Composite (MFC) piezoelectric materials, which offer several advantages over conventional piezoelectric ceramics. MFCs consist of rectangular piezoelectric fibers embedded in an epoxy matrix, sandwiched between adhesive layers and polyimide films. This construction provides exceptional flexibility, durability, and electromechanical coupling efficiency. The interdigitated electrode pattern characteristic of MFCs enables efficient poling and charge collection, making them particularly suitable for energy harvesting applications.  

Ionic Polymer Metal Composites (IPMCs) serve as the active tuning elements in the disclosed systems. IPMCs are electroactive polymers consisting of an ionic polymer membrane (typically Nafion) with metal electrodes deposited on both surfaces. When subjected to a low voltage (1-4V), hydrated cations within the polymer matrix migrate toward the cathode, causing bending deformation. This actuation mechanism operates at very low power levels, making IPMCs ideal for integration with energy harvesting systems. The force generated by IPMC actuation can be precisely controlled by adjusting the input voltage, enabling fine-tuning of the harvester's resonant frequency.  

### Example 1  

PVEH device 100 represents a fundamental embodiment of the invention. This device comprises a cantilever beam structure with a piezoelectric transducer, preferably an MFC, bonded to its surface in a unimorph configuration. The cantilever includes a thicker root section for structural support and a thinner extension section to facilitate vibration.  

A tuning member consisting of one or more IPMC strips is positioned to interact with the cantilever. In the inactive state, the IPMC maintains a position separated from the cantilever. When activated by an applied voltage, the IPMC bends to contact the cantilever, applying a controlled force that modifies the system's stiffness. This interaction shifts the resonant frequency of the device to match ambient vibration conditions.  

The degree of frequency tuning can be precisely controlled by adjusting the voltage applied to the IPMC. Experimental results demonstrate that increasing the IPMC actuation voltage from 1.5V to 2.1V can shift the resonant frequency by approximately 2Hz while consuming minimal power (typically less than 6μW). The power harvested at the tuned frequency substantially exceeds the tuning power expenditure, resulting in significant net energy gain.  

### Example 2  

PVEH device 200 illustrates an enhanced embodiment incorporating multiple IPMC elements in a stacked configuration. This design increases the available actuation force while maintaining low power consumption. The stacked IPMC arrangement may take various geometric forms, including but not limited to parallel strips joined at their tips or diamond-shaped configurations.  

In one implementation, two IPMC strips are connected at their distal ends, forming a V-shape when viewed from the side. This configuration doubles the available actuation force while maintaining the same voltage requirements as a single IPMC. The increased force enables greater frequency tuning range and improved stability during operation.  

The device includes a mounting fixture that allows adjustment of the IPMC's initial position relative to the cantilever. This adjustability enables optimization of the system for different frequency ranges and vibration environments. The fixture may incorporate precision mechanisms for fine positional control, ensuring reproducible tuning characteristics.  

### Example 3  

Energy harvesting system 300 represents a complete implementation including power conditioning and control components. This system integrates the tunable PVEH device with a control circuit and energy storage elements to create a self-sustaining power source.  

The tuning member in this embodiment comprises multiple IPMC devices arranged to provide balanced force application to the cantilever. The control circuit monitors the harvested power and adjusts the IPMC actuation voltage to maintain optimal frequency matching. The system includes energy storage components such as capacitors or batteries to buffer the harvested energy.  

A full wave bridge (FWB) rectifier circuit conditions the alternating current output from the piezoelectric transducer into direct current suitable for powering electronic devices or charging storage elements. The FWB circuit incorporates diodes and capacitors selected to minimize power losses while handling the expected voltage and current levels.  

The system features a tunable load member that allows adjustment of the electrical load presented to the piezoelectric transducer. This adjustability enables optimization of power transfer under varying operating conditions. Multiple IPMC strips may be independently controlled to achieve precise frequency tuning across a broad range.  

### Example 4  

This example illustrates specific dimensional configurations of PVEH devices that have demonstrated effective performance in experimental testing. One preferred embodiment utilizes an aluminum cantilever beam with a two-section design: a root section measuring 15cm × 3.5cm × 0.64mm and an extension section measuring 10.5cm × 2cm × 0.37mm, with a 3cm overlap between sections.  

The piezoelectric transducer in this embodiment is an MFC M8514-P2 with an active area of 85mm × 14mm, providing a free strain of 630ppm and a blocking force of 76N. The IPMC elements measure 30mm × 5mm × 0.2mm, with actuation voltages ranging from 0.5V to 4V.  

Experimental results with these dimensions demonstrate resonant frequency tuning from 5.9Hz to 8.0Hz, with corresponding power output increasing from 29μW to 64.19μW. The power required for tuning ranges from 0.765μW to 5.88μW, resulting in substantial net power gains at all tuned frequencies.  

### Example 5  

This example demonstrates the resonance frequency tuning capabilities of the disclosed PVEH devices through experimental data. Testing shows that applying 1.5V to the IPMC actuator shifts the resonant frequency from 5.9Hz to 7.57Hz, while consuming only 0.765μW of power. The harvested power increases from 29μW to 52.03μW during this tuning, resulting in a net power gain of 51.265μW.  

Further frequency tuning to 8.0Hz requires 5.88μW of actuation power while yielding 64.19μW of harvested power, for a net gain of 58.31μW. These results demonstrate that the tuning mechanism provides substantial energy benefits across the entire operational frequency range.  

The relationship between actuation voltage and frequency shift is nearly linear within the tested range, enabling predictable control of the tuning process. The system maintains this linearity across different environmental conditions and device configurations.  

### Example 6  

A method of energy harvesting using the disclosed systems involves several key steps. First, the ambient vibration frequency is detected by analyzing the output of the piezoelectric transducer across a range of excitation frequencies. The frequency producing maximum output is identified as the current resonant frequency of the system.  

If this resonant frequency does not match the dominant ambient vibration frequency (as determined by power output analysis), the control system activates the IPMC tuning mechanism. The IPMC receives a voltage input calculated to produce the required frequency shift, based on predetermined calibration data.  

The system continuously monitors the harvested power and adjusts the tuning as needed to maintain optimal frequency matching. A portion of the harvested energy powers the tuning mechanism, while the remainder is available for use by connected devices or storage in batteries or capacitors.  

The method includes provisions for handling varying vibration conditions, including situations where the ambient frequency changes gradually or abruptly. The control algorithm ensures stable operation while maximizing net energy harvest under all conditions.  

### General Considerations  

As used in this disclosure, singular forms such as "a," "an," and "the" include plural referents unless the context clearly dictates otherwise. For example, reference to "an IPMC device" includes embodiments containing one or more such devices.  

The terms "comprises," "comprising," "includes," or "including" when used in this specification specify the presence of stated features but do not preclude the presence or addition of one or more other features.  

The term "coupled" as used herein refers to any connection, coupling, link, or interaction between components, whether direct or indirect, and whether mechanical, electrical, or otherwise. Components may be coupled through intermediate elements.  

The disclosure of this patent is not limited to the particular configurations, dimensions, or materials described herein. These details are provided as examples, and one skilled in the art will recognize that numerous variations and modifications are possible while remaining within the scope of the invention.  

The systems and methods described may be implemented in various forms and scales, from macroscopic devices to MEMS-scale implementations. The principles of operation remain consistent across these implementations.  

The disclosed technology finds application in numerous fields including structural health monitoring, wireless sensor networks, wearable electronics, and IoT devices. The ability to harvest energy efficiently across varying frequency ranges makes the invention particularly valuable in environments with changing vibration characteristics.