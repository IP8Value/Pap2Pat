# DESCRIPTION

## FIELD

The present invention relates to a tunable piezoelectric vibration energy harvester (TPVEH) that utilizes the actuation properties of Ionic Polymer Metal Composite (IPMC) to actively tune the resonant frequency of the harvester. The invention is particularly useful in applications where ambient vibrations vary over a wide frequency range, such as in structural health monitoring (SHM) and the Internet of Things (IoT).

## BACKGROUND

The development of microsystems has led to the realization of very low power MEMS (Micro Electro Mechanical Systems) based sensors. Deploying a large number of these sensors on structures enables more comprehensive and interactive Structural Health Monitoring (SHM) for diagnosis and prognosis in critical structures. Enabling autonomous power sources for these sensors through energy harvesting from ambient conditions is of great research interest. Among the various forms of ambient energy, vibrations are prevalent in many systems and structures during their operational lifecycle. Therefore, vibration-based energy harvesting, especially using piezoelectric materials, has been extensively studied.

Piezoelectric vibration energy harvesters (PVEHs) have demonstrated good conversion efficiency. However, they typically have a poor response at frequencies other than their resonant frequency. To address this limitation, various methods have been explored to broaden the bandwidth of PVEHs. These methods include mechanically tuning the harvester's resonance frequency by changing the mass or stiffness of the system. For example, introducing notches to apply axial preload, using axially compressed piezoelectric bimorphs, and adding different masses to cantilevers have been reported. However, these methods often have limited tunable frequency ranges and may require significant power input.

The present invention provides a novel design that overcomes these limitations by integrating IPMC with a piezoelectric material to achieve active tuning over a wider frequency range with minimal power consumption.

## SUMMARY

The present invention provides a tunable piezoelectric vibration energy harvester (TPVEH) that includes a cantilever beam, a piezoelectric material bonded to the cantilever beam, and an Ionic Polymer Metal Composite (IPMC) actuator. The IPMC actuator is configured to apply a load to the cantilever beam, thereby changing the resonant frequency of the harvester. The IPMC actuator is powered by a low voltage, which allows for precise control of the applied load and, consequently, the resonant frequency of the harvester.

The TPVEH is designed to operate efficiently over a wide range of ambient vibration frequencies, making it suitable for applications where the ambient frequency varies. The power required to actuate the IPMC is significantly lower than the power output of the PVEH, ensuring a net power gain. Additionally, the IPMC can be easily miniaturized and integrated into the harvester design, making it suitable for various applications, including structural health monitoring and the Internet of Things (IoT).

## DETAILED DESCRIPTION

### Introduction

The present invention addresses the challenge of tuning the resonant frequency of a piezoelectric vibration energy harvester (PVEH) to match varying ambient vibration frequencies. Traditional PVEHs have a narrow bandwidth and perform optimally only at their resonant frequency. This limitation restricts their applicability in environments where ambient vibrations vary over a wide frequency range. The invention introduces an active tuning mechanism using Ionic Polymer Metal Composite (IPMC) to broaden the bandwidth of the PVEH while maintaining high efficiency.

### Example 1

In one embodiment, the TPVEH includes an aluminum cantilever beam with two sections: a thicker root section and a thinner extension section. The piezoelectric material, specifically Macro Fiber Composites (MFC), is bonded to the thicker section of the cantilever beam in a unimorph configuration. The MFC used in this embodiment has an active length of 85 mm, an active width of 14 mm, a free strain of 630 ppm, and a blocking force of 76 N.

An IPMC actuator is positioned to apply a load to the cantilever beam at a specific point. The IPMC actuator is composed of an ionic polymer, such as Nafion, with platinum or gold coatings on its surfaces. When a low voltage (1 to 4 V) is applied to the IPMC, the positively charged hydrated cations in the membrane network migrate to the negative electrode, causing the IPMC to bend. This bending applies a compressive load to the cantilever beam, thereby changing its resonant frequency.

The experimental setup involves an electrodynamic exciter to simulate vibrations and a function generator to provide varying frequencies. The output of the PVEH is connected to a full-wave rectifier and then to an oscilloscope for measurement. The results show that by applying a low voltage to the IPMC, the resonant frequency of the PVEH can be shifted from 5.9 Hz to 8 Hz, with a corresponding increase in power output from 29 µW to 64.19 µW. The power spent on actuating the IPMC is significantly lower than the power output of the PVEH, indicating a net power gain.

### Example 2

In another embodiment, the TPVEH uses a different MFC type, M8507P2, with an active length of 85 mm, an active width of 7 mm, a free strain of 605 ppm, and a blocking force of 38 N. The MFC is bonded to an aluminum substrate in a unimorph configuration. The IPMC actuator is positioned at a variable distance (h) from the cantilever surface to study the effect of the loading point on the resonant frequency.

The experimental setup is similar to Example 1, with the addition of a fixture to adjust the distance (h) of the IPMC tip from the cantilever surface. The results show that the resonant frequency of the PVEH increases as the distance (h) decreases. For example, with an IPMC input voltage of 1.5 V, the resonant frequency shifts from 8.4 Hz to 9.9 Hz as the distance (h) is reduced from 30 mm to 10 mm. The net power output of the PVEH is maintained between 25.69 µW and 33.11 µW, demonstrating the effectiveness of the tuning mechanism.

### Example 3

In a further embodiment, the TPVEH is designed to operate autonomously by integrating a low-power microcontroller. The microcontroller monitors the ambient vibration frequency and adjusts the input voltage to the IPMC to maintain the optimal resonant frequency of the PVEH. The ambient frequency is detected from the voltage output of the PVEH, and a lookup table is used to identify the appropriate input voltage for the IPMC.

The microcontroller can be powered by the harvested energy, creating a closed-loop system. This design ensures that the PVEH remains efficient even when the ambient frequency changes, making it ideal for long-term applications in structural health monitoring and IoT.

### Example 4

In another embodiment, the TPVEH is optimized for miniaturization and integration into MEMS devices. The IPMC actuator is designed with multiple electrodes to enable precise control of the applied load. The thickness of the Nafion film and the cation concentration can be adjusted during fabrication to tailor the actuation properties of the IPMC to the specific requirements of the PVEH.

The experimental results demonstrate that the miniaturized TPVEH maintains high efficiency and a wide tunable frequency range. The power required to actuate the IPMC is minimal, ensuring a net power gain. This design is particularly suitable for applications where space and weight are critical, such as in biomedical devices.

### Example 5

In an embodiment focused on nanoscale applications, the TPVEH integrates IPMC with piezoelectric nano-fibers. The IPMC is designed with silver nano-powders to enhance its actuation properties. The nanoscale TPVEH is capable of harvesting energy from low-frequency vibrations and can be used in various biomedical applications, such as implantable sensors and wearable devices.

The experimental results show that the nanoscale TPVEH achieves a wide tunable frequency range and high efficiency. The power required to actuate the IPMC is extremely low, making it suitable for applications where power consumption is a critical factor.

### Example 6

In a final embodiment, the TPVEH is designed for high-frequency applications. The cantilever beam is made of a high-strength material, such as titanium, to withstand high-frequency vibrations. The MFC is optimized for high-frequency operation, and the IPMC actuator is designed to apply a rapid and precise load to the cantilever beam.

The experimental results demonstrate that the high-frequency TPVEH maintains high efficiency and a wide tunable frequency range. The power required to actuate the IPMC is minimal, ensuring a net power gain. This design is suitable for applications where high-frequency vibrations are prevalent, such as in aerospace and automotive industries.

### General Considerations

The present invention provides a tunable piezoelectric vibration energy harvester (TPVEH) that utilizes the actuation properties of Ionic Polymer Metal Composite (IPMC) to actively tune the resonant frequency of the harvester. The key features of the invention include:

1. **Wide Tunable Frequency Range**: The TPVEH can be tuned to a wide range of ambient vibration frequencies, making it suitable for various applications.
2. **Low Power Consumption**: The power required to actuate the IPMC is significantly lower than the power output of the PVEH, ensuring a net power gain.
3. **Miniaturization**: The IPMC can be easily miniaturized and integrated into the harvester design, making it suitable for applications where space and weight are critical.
4. **Autonomous Operation**: The TPVEH can be designed to operate autonomously by integrating a low-power microcontroller, which monitors the ambient frequency and adjusts the input voltage to the IPMC.
5. **Versatility**: The TPVEH can be optimized for various applications, including structural health monitoring, the Internet of Things (IoT), biomedical devices, and high-frequency applications.

The invention offers a significant improvement over existing PVEHs by providing a robust and efficient solution for energy harvesting in environments with varying ambient vibration frequencies.