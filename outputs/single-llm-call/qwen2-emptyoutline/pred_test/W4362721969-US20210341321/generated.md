# DESCRIPTION

## TECHNICAL FIELD OF THE INVENTION

The present invention relates to a method and apparatus for measuring fluid flow rates in microfluidic channels. More specifically, the invention pertains to a sensor that utilizes the deformation of a thin silicon nitride membrane to detect and measure fluid flow rates. The sensor employs a microwave resonator to monitor changes in the capacitance caused by the membrane deformation, thereby providing a highly sensitive and accurate means of flow rate measurement.

## BACKGROUND OF THE INVENTION (PRIOR ART)

Accurate and reliable measurement of fluid flow rates in microfluidic channels is crucial for various applications, including biomedical research, chemical analysis, and environmental monitoring. Traditional methods often rely on external pumps and commercial flow sensors, which are not well-suited for integration at the microscale due to their size, complexity, and inherent limitations such as delays and fluctuations.

Thermal flow sensors, while compact and easy to fabricate, suffer from reduced accuracy when the fluid composition changes, as their operation depends on the heat capacity of the fluid. Coriolis flowmeters, although more accurate and capable of handling a wide dynamic range, are complex and expensive, making them less practical for widespread use in microfluidic systems.

Passive flow sensing techniques, such as capacitive sensors and cantilever-based sensors, have been explored. Capacitive sensors measure the drag force in microfluidic channels, while cantilever-based sensors correlate flow rate changes with cantilever deflection. However, these methods often require the fluid to pass through a mechanical structure, which can introduce additional complexities and limitations.

Electrical flow sensors operating in the radio frequency (RF) domain have also been developed. These sensors typically exploit the deflection of a thin polydimethylsiloxane (PDMS) layer and measure this deflection using a microwave sensor. While achieving high sensitivity, these sensors suffer from long recovery times due to the relaxation timescale of the elastomeric PDMS layer.

The present invention addresses these limitations by providing a robust, integrated flow sensor that combines the mechanical properties of a silicon nitride membrane with the sensitivity of a microwave resonator. This combination allows for the precise measurement of both liquid and gas flow rates in microfluidic channels, with fast response times and high durability.

## SUMMARY OF THE INVENTION

The present invention provides a method and apparatus for measuring fluid flow rates in microfluidic channels using a thin silicon nitride membrane integrated with a microwave resonator. The key aspects of the invention are as follows:

1. **Membrane Deformation for Flow Rate Sensing**: The invention utilizes the deformation of a thin silicon nitride membrane to detect fluid flow rates. The membrane, which forms part of the microfluidic channel, deforms in response to fluid flow, causing a change in the capacitance of a microwave resonator placed on the membrane. This change in capacitance is directly related to the flow rate and can be accurately measured.

2. **Microwave Resonator for High Sensitivity**: The microwave resonator, designed as a coplanar waveguide (CPW) with a 50 Ω impedance, is used to monitor the changes in capacitance. The resonator's frequency shift, which is proportional to the membrane deformation, is tracked using a custom electronic circuit based on phase-locked loops. This method provides high sensitivity and fast response times.

3. **Dual Sensing Mechanisms**: The invention offers two primary sensing mechanisms:
   - **Frequency Shift Sensing**: The primary method involves measuring the shift in the resonance frequency of the microwave resonator as the membrane deforms. This method is suitable for a wide range of flow rates and provides high sensitivity.
   - **Pulsation Period Sensing**: At certain flow rates, the membrane exhibits periodic pulsations. The period of these pulsations can be measured using the phase response of the microwave sensor, providing an additional means of flow rate measurement, particularly useful at very low flow rates.

4. **Robust and Versatile Design**: The silicon nitride membrane is chosen for its excellent mechanical properties, including durability and the ability to measure both liquid and gas flow rates. The ceramic nature of the membrane ensures long-term stability and reliability, making the sensor suitable for a variety of microfluidic applications.

5. **Fast Recovery Times**: Unlike sensors using elastomeric materials, the silicon nitride membrane has fast recovery times, allowing for rapid and continuous flow rate measurements.

The invention thus provides a highly sensitive, accurate, and robust method for measuring fluid flow rates in microfluidic channels, addressing the limitations of existing technologies and enabling new applications in microfluidics.

## DETAILED DESCRIPTION OF THE INVENTION

### Fabrication of the Flow Sensor

The flow sensor is fabricated using a silicon wafer with a 500 µm Si substrate, a 2 µm SiO2 middle layer, and a 220 nm silicon nitride top layer. The fabrication process involves several steps:

1. **Backside Etching**:
   - Photolithography is performed on the backside of the wafer to define an etch window.
   - Inductively coupled plasma (ICP) etching is used to create the etch window.
   - The wafer is then subjected to KOH wet etching overnight, which etches through the Si and SiO2 layers, leaving a 220 nm thick silicon nitride membrane on the front surface.

2. **Microwave Resonator Fabrication**:
   - Photolithography is performed on the front side of the wafer to define the pattern for the coplanar waveguide (CPW) microwave resonator.
   - A 100 nm gold layer is deposited to form the signal and ground electrodes of the CPW resonator.
   - The CPW resonator is designed to have a 50 Ω impedance to match the impedance of the electronic measurement system.

3. **Microfluidic Channel Fabrication**:
   - A negative photoresist (SU-8) is used to fabricate the molds for the microfluidic channels.
   - Cured PDMS (ratio 10:1) is poured onto the molds and baked at 80°C.
   - The microchannels, typically 300 µm in width and 150 µm in depth, are peeled off from the mold and bonded to the chip using a plasma cleaning process.
   - The microchannels are aligned with the silicon nitride membrane and gold electrodes.

### Operation of the Flow Sensor

The flow sensor operates by monitoring the deformation of the silicon nitride membrane caused by fluid flow. The key operational steps are as follows:

1. **Flow Rate Measurement by Frequency Shift**:
   - When fluid flows through the microchannel, it causes the silicon nitride membrane to deform.
   - The deformation changes the distance between the signal and ground electrodes of the CPW resonator, altering the capacitance.
   - The change in capacitance results in a shift in the resonance frequency of the microwave resonator.
   - The resonance frequency shift is tracked using a custom electronic circuit based on phase-locked loops (PLL).
   - The frequency shift is directly proportional to the flow rate, allowing for accurate measurement.

2. **Flow Rate Measurement by Pulsation Period**:
   - At certain flow rates, the membrane exhibits periodic pulsations.
   - These pulsations are detected by monitoring the phase response of the microwave resonator.
   - The period of the pulsations is related to the flow rate, providing an additional means of measurement.
   - The pulsation period increases as the flow rate decreases, offering a sensitive method for low flow rate measurements.

### Experimental Setup and Data Acquisition

The experimental setup consists of two major subsystems: a microfluidic flow controller and an electronic measurement system. The chip is placed under a microscope stage to observe the mechanical deformations of the membrane. The flow rate is controlled using a controllable pressure pump, and the flow is monitored using a commercial thermal flow sensor. The data acquisition is conducted using a custom-built LabVIEW program, which records the electronic data and flow rate values simultaneously every 50 milliseconds.

### Results and Discussion

#### Liquid Flow Rate Experiments

In the first set of experiments, the shift in the resonance frequency of the microwave sensor was measured as a function of the flow rate. The results showed a clear correlation between the flow rate and the frequency shift, with a short-term sensitivity level of 0.5 µL/min. The sensor demonstrated a time resolution of several seconds when the flow rate was increased stepwise. When the flow was stopped, a downward frequency drift was observed, lasting approximately 50 seconds. However, additional flow rate measurements could still be conducted during the recovery period.

#### Gas Flow Rate Experiments

The sensor was also tested with pressurized air. For the first air flow experiment, a membrane with dimensions of 1.7 mm length and 1 mm width was used. The resonance frequency was approximately 2.55 GHz, and an almost linear relationship was observed for higher pressure values. At low pressures, the trend deviated slightly from the linear response. For higher air flow rates, a device with an increased channel width of 1.3 mm was fabricated, and a monotonically rising frequency response was observed as the applied pressure increased. The device with a 220 nm silicon nitride membrane could endure an applied pressure up to 4 bar.

#### Pulsation Period Experiments

At constant liquid flows, periodic deformations of the membrane were observed under optical microscopy. The period of these pulsations, known as the pulse interval, was found to depend on the flow rate. For a given constant flow rate, the pulse interval remained constant. As the flow rate was set to different constant values, the pulse interval changed accordingly. The experiments were conducted without any cyclic mechanisms, and the results verified that membrane pulsations still occur with a constant pressure source.

### Conclusion

In conclusion, the present invention demonstrates a highly sensitive and accurate method for measuring fluid flow rates in microfluidic channels using a thin silicon nitride membrane integrated with a microwave resonator. The sensor provides two primary sensing mechanisms—frequency shift and pulsation period—enabling precise measurements over a wide range of flow rates. The robust design of the silicon nitride membrane ensures long-term stability and reliability, making the sensor suitable for various microfluidic applications. The invention addresses the limitations of existing technologies and opens new possibilities for advanced microfluidic systems.