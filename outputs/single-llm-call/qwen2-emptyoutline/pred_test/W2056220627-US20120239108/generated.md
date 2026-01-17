# DESCRIPTION

## GOVERNMENT FUNDING

The invention described herein was not developed with any government funding.

## TECHNICAL FIELD

The present invention relates to the field of neuromodulation devices, specifically to methods and systems for improving the energy efficiency of implanted pulse generators (IPGs) used in electrical stimulation of neural tissue. The invention addresses the need for reducing power consumption in IPGs, thereby extending the operational life of the device and enhancing the overall efficacy of neuromodulation therapies.

## BACKGROUND

Neuromodulation devices have been widely used for treating various neurological disorders, including deep brain stimulation (DBS), spinal cord stimulation (SCS), and cochlear stimulation. These devices typically consist of an implanted pulse generator (IPG) that delivers electrical stimuli to excitable tissue through electrodes. The primary goal of such stimulation is to generate action potentials in axons, which modulate the release of neurotransmitters in specific parts of the nervous system.

However, the basic hardware and stimulus waveforms used in clinical practice have remained relatively unchanged for decades. The energy efficiency of these devices is a critical concern, as the implanted battery's lifespan directly impacts the device's longevity and the need for surgical replacement. The power consumption of the stimulation circuitry, which typically accounts for the largest share of the energy usage, is a key area for optimization.

Traditional constant-current stimulators use a fixed compliance voltage to deliver the necessary current across the tissue load. This fixed high compliance voltage is often excessive for normal operating conditions, leading to inefficiencies. Additionally, the shape and duration of the stimulus waveform can significantly affect energy consumption. Recent studies have suggested that non-rectangular waveforms, such as centered-triangular pulses, can achieve energy savings compared to traditional rectangular pulses.

There is a need for improved methods and systems that optimize the energy efficiency of IPGs by exploiting the biophysical properties of the stimulated tissue and refining the electronic circuit design. The present invention addresses this need by providing a novel approach to reduce power consumption in neuromodulation devices.

## SUMMARY

The present invention provides a method and system for improving the energy efficiency of implanted pulse generators (IPGs) used in electrical stimulation of neural tissue. The invention involves the use of an adjustable compliance voltage in the pulse generator circuitry and the implementation of a biophysically optimized stimulus waveform, such as a centered-triangular pulse, to reduce energy consumption.

In one embodiment, the invention includes a method for optimizing the energy efficiency of an IPG. The method comprises the steps of:
1. Determining the minimum compliance voltage required to maintain the desired current output during stimulation.
2. Adjusting the compliance voltage to the determined minimum value.
3. Selecting a stimulus waveform shape that is optimized for energy efficiency, such as a centered-triangular waveform.
4. Determining the optimal pulse width for the selected waveform based on the biophysical properties of the target neural tissue.
5. Delivering the optimized stimulus waveform with the adjusted compliance voltage to the neural tissue.

In another embodiment, the invention includes a system for optimizing the energy efficiency of an IPG. The system comprises:
1. An adjustable compliance voltage circuit configured to dynamically adjust the compliance voltage based on the real-time requirements of the stimulation.
2. A waveform generator configured to produce a stimulus waveform shape that is optimized for energy efficiency, such as a centered-triangular waveform.
3. A pulse width controller configured to determine the optimal pulse width for the selected waveform based on the biophysical properties of the target neural tissue.
4. An electrode configured to deliver the optimized stimulus waveform with the adjusted compliance voltage to the neural tissue.

The invention further includes a computer-implemented method for simulating the energy consumption of different stimulus waveforms and pulse widths to determine the optimal parameters for a given neurostimulation application. The method involves:
1. Modeling the biophysical properties of the target neural tissue.
2. Simulating the energy consumption for various stimulus waveforms and pulse widths.
3. Identifying the waveform and pulse width combination that minimizes energy consumption.

The invention provides significant advantages over existing neuromodulation devices by reducing power consumption, extending the operational life of the IPG, and enhancing the overall efficacy of the therapy. The use of an adjustable compliance voltage and an optimized stimulus waveform can lead to substantial energy savings, making the invention particularly useful for applications such as deep brain stimulation (DBS), spinal cord stimulation (SCS), and cochlear stimulation.

## DETAILED DESCRIPTION

The present invention provides a method and system for improving the energy efficiency of implanted pulse generators (IPGs) used in electrical stimulation of neural tissue. The invention combines the use of an adjustable compliance voltage in the pulse generator circuitry with the implementation of a biophysically optimized stimulus waveform to reduce energy consumption. The following detailed description outlines the components and methods of the invention.

### Adjustable Compliance Voltage Circuit

The adjustable compliance voltage circuit is a key component of the invention. Traditional constant-current stimulators use a fixed compliance voltage to deliver the necessary current across the tissue load. This fixed high compliance voltage is often excessive for normal operating conditions, leading to inefficiencies. The adjustable compliance voltage circuit dynamically adjusts the compliance voltage to the minimum value required to maintain the desired current output during stimulation.

#### Operation of the Adjustable Compliance Voltage Circuit

1. **Determination of Minimum Compliance Voltage:**
   - The circuit continuously monitors the load impedance and the current output.
   - The minimum compliance voltage required to maintain the desired current output is determined in real-time.
   - The compliance voltage is adjusted to this minimum value to ensure efficient operation.

2. **Dynamic Adjustment:**
   - The circuit includes a feedback mechanism to dynamically adjust the compliance voltage based on changes in the load impedance and current requirements.
   - This ensures that the compliance voltage remains at the optimal level throughout the stimulation process.

3. **Energy Savings:**
   - By minimizing the compliance voltage, the circuit significantly reduces the power consumption of the stimulation circuitry.
   - This leads to extended battery life and improved overall efficiency of the IPG.

### Biophysically Optimized Stimulus Waveform

The invention also involves the use of a biophysically optimized stimulus waveform to further reduce energy consumption. Traditional neuromodulation devices typically use rectangular waveforms, which are not the most energy-efficient. The invention utilizes a centered-triangular waveform, which has been shown to achieve significant energy savings.

#### Characteristics of the Centered-Triangular Waveform

1. **Waveform Shape:**
   - The centered-triangular waveform consists of a ramp-up phase followed by a symmetric ramp-down phase.
   - This shape is designed to exploit the biophysical properties of the stimulated tissue, leading to more efficient action potential generation.

2. **Energy Efficiency:**
   - Studies have shown that the centered-triangular waveform can achieve energy savings of up to 12% compared to traditional rectangular waveforms.
   - The waveform injects more charge at similar pulse widths but consumes less energy for a given level of injected charge.

3. **Optimal Pulse Width:**
   - The optimal pulse width for the centered-triangular waveform is determined based on the biophysical properties of the target neural tissue.
   - For large diameter axons, the optimal pulse width is typically around 200 µs.
   - For small diameter axons, the optimal pulse width is longer, around 670 µs.

### System Components

The system for optimizing the energy efficiency of an IPG includes the following components:

1. **Adjustable Compliance Voltage Circuit:**
   - This circuit dynamically adjusts the compliance voltage to the minimum value required for efficient stimulation.
   - It includes a feedback mechanism to monitor and adjust the compliance voltage in real-time.

2. **Waveform Generator:**
   - The waveform generator produces the biophysically optimized stimulus waveform, such as a centered-triangular waveform.
   - It can be programmed to generate different waveform shapes and pulse widths based on the specific requirements of the stimulation application.

3. **Pulse Width Controller:**
   - The pulse width controller determines the optimal pulse width for the selected waveform based on the biophysical properties of the target neural tissue.
   - It can be calibrated using computer simulations or in vivo experiments to identify the most energy-efficient pulse width for a given application.

4. **Electrode:**
   - The electrode is configured to deliver the optimized stimulus waveform with the adjusted compliance voltage to the neural tissue.
   - It can be a monopolar or bipolar electrode, depending on the specific application.

### Method for Optimizing Energy Efficiency

The method for optimizing the energy efficiency of an IPG involves the following steps:

1. **Determine Minimum Compliance Voltage:**
   - Measure the load impedance and current output during stimulation.
   - Calculate the minimum compliance voltage required to maintain the desired current output.
   - Adjust the compliance voltage to this minimum value.

2. **Select Biophysically Optimized Waveform:**
   - Choose a stimulus waveform shape that is optimized for energy efficiency, such as a centered-triangular waveform.
   - Ensure that the waveform generator is configured to produce the selected waveform.

3. **Determine Optimal Pulse Width:**
   - Use computer simulations or in vivo experiments to determine the optimal pulse width for the selected waveform based on the biophysical properties of the target neural tissue.
   - Calibrate the pulse width controller to deliver the optimal pulse width.

4. **Deliver Optimized Stimulus:**
   - Deliver the optimized stimulus waveform with the adjusted compliance voltage to the neural tissue using the electrode.
   - Monitor the energy consumption and adjust the parameters as needed to ensure efficient operation.

### Computer-Implemented Method for Simulation

The invention further includes a computer-implemented method for simulating the energy consumption of different stimulus waveforms and pulse widths to determine the optimal parameters for a given neurostimulation application. The method involves the following steps:

1. **Model Biophysical Properties:**
   - Create a computational model of the target neural tissue, including the biophysical properties of the axons.
   - Define the parameters of the model, such as axon diameter, tissue resistance, and electrode-tissue interface characteristics.

2. **Simulate Energy Consumption:**
   - Simulate the energy consumption for various stimulus waveforms and pulse widths using the computational model.
   - Calculate the energy required to recruit a specific percentage of the neural population for each waveform and pulse width combination.

3. **Identify Optimal Parameters:**
   - Identify the waveform and pulse width combination that minimizes energy consumption.
   - Use the identified parameters to configure the IPG for optimal energy efficiency.

### Advantages of the Invention

The invention provides several significant advantages over existing neuromodulation devices:

1. **Reduced Power Consumption:**
   - The use of an adjustable compliance voltage and a biophysically optimized stimulus waveform significantly reduces the power consumption of the IPG.
   - This leads to extended battery life and reduced need for surgical replacements.

2. **Improved Efficacy:**
   - The optimized stimulus waveform and pulse width enhance the efficacy of the neuromodulation therapy by more efficiently generating action potentials in the target neural tissue.
   - This can lead to better clinical outcomes and improved quality of life for patients.

3. **Flexibility:**
   - The system can be adapted to different neurostimulation applications by adjusting the compliance voltage, waveform shape, and pulse width based on the specific requirements of the target tissue.
   - This makes the invention versatile and applicable to a wide range of neuromodulation therapies.

### Conclusion

The present invention provides a method and system for improving the energy efficiency of implanted pulse generators (IPGs) used in electrical stimulation of neural tissue. By combining the use of an adjustable compliance voltage circuit with a biophysically optimized stimulus waveform, the invention significantly reduces power consumption, extends the operational life of the IPG, and enhances the overall efficacy of the therapy. The invention is particularly useful for applications such as deep brain stimulation (DBS), spinal cord stimulation (SCS), and cochlear stimulation, and has the potential to revolutionize the field of neuromodulation.