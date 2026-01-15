# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to imaging methods for quantifying Fractional Flow Reserve (FFR) using multi-dimensional phase-contrast magnetic resonance (PC-MR) sequences. The method and system described herein provide a noninvasive approach to determine the pressure gradient within blood vessel segments, which is crucial for diagnosing cardiovascular disease.

## BACKGROUND

Current methods for assessing the functional severity of coronary stenosis often rely on invasive techniques such as fractional flow reserve (FFR). These methods involve catheterization and can pose significant risks to patients, including complications from the procedure itself. Noninvasive alternatives like computed tomography angiography (CTA) have limitations in accurately quantifying pressure gradients due to their reliance on indirect measurements and the potential for radiation exposure. Therefore, there is a need for a robust, noninvasive method that can accurately determine FFR and provide detailed information about blood flow dynamics.

## SUMMARY

The present invention introduces a method for quantifying Fractional Flow Reserve (FFR) using a multi-dimensional phase-contrast magnetic resonance (PC-MR) sequence. This method involves acquiring high-resolution velocity field data within the coronary arteries, which is then used to determine the pressure gradient within a blood vessel segment. The pressure gradient is correlated to an FFR value, providing a noninvasive means of assessing the functional severity of stenosis.

The invention also describes an MRI system capable of executing the multi-dimensional PC-MR sequence and includes a non-transitory machine-readable medium with instructions for performing the method. The MRI system is equipped with advanced hardware and software components to ensure accurate and reliable data acquisition and processing. The processor within the system performs complex calculations, including the application of Navier-Stokes equations to derive pressure gradients from velocity field data.

## DETAILED DESCRIPTION

### Definitions

For the purposes of this application, several terms are defined as follows:
- **Fractional Flow Reserve (FFR):** A measure of the functional severity of a coronary artery stenosis, calculated as the ratio of maximal flow in the presence of the stenosis to normal maximal flow.
- **Phase-Contrast Magnetic Resonance (PC-MR):** An MRI technique that measures blood flow velocities by encoding velocity information into the phase of the MR signal.
- **Pressure Gradient:** The change in pressure over a distance, which is used to quantify the resistance to blood flow within a vessel segment.
- **Volume of Interest (VOI):** A defined region within an image where specific measurements are taken.

### FFR Technique

Fractional Flow Reserve (FFR) is a critical parameter for assessing the functional impact of coronary artery stenosis. Traditionally, FFR is measured invasively using pressure wires and catheters. However, this approach poses risks to patients and can be costly. The present invention provides a noninvasive alternative by utilizing multi-dimensional PC-MR sequences to quantify blood flow velocities and derive pressure gradients within the coronary arteries.

### Advantages of MRI over CT

Magnetic Resonance Imaging (MRI) offers several advantages over Computed Tomography (CT) for quantifying FFR. MRI does not involve ionizing radiation, making it safer for repeated use in patient monitoring. Additionally, MRI provides higher spatial and temporal resolution, allowing for more accurate measurement of blood flow velocities and pressure gradients. The ability to perform multi-dimensional PC-MR sequences further enhances the precision and reliability of FFR measurements.

### Method of Using MRI for Quantifying FFR

The method involves acquiring velocity field data using a multi-dimensional phase-contrast magnetic resonance (PC-MR) sequence. This sequence is designed to measure blood flow velocities in multiple directions within a defined volume of interest (VOI). The acquired data is then processed to calculate the pressure gradient within the vessel segment, which is subsequently used to determine the FFR value.

### Multi-Dimensional PC-MR Sequence

The multi-dimensional PC-MR sequence acquires velocity field data at multiple cardiac phases and respiratory states. This is achieved through ECG-triggering and navigator-gating techniques, ensuring that the data is collected during periods of minimal motion. The sequence measures velocities in all three directions (vx, vy, vz) for a single cross-section per acquisition, with 4-5 consecutive slices obtained in the proximal left anterior descending artery (LAD).

### Calculation of Pressure Gradient Using Navier-Stokes Equations

The Navier-Stokes equations are used to derive the pressure gradient from the velocity field data. These equations describe the motion of fluid and can be applied to blood flow within the coronary arteries. The processor within the MRI system performs these calculations, taking into account the viscosity of blood and the geometry of the vessel segment.

### Image Reconstruction Using Generic Fourier Transform Methods

The acquired k-space data is reconstructed using generic Fourier transform methods to generate velocity images. These images provide a visual representation of the flow velocities within the coronary arteries. The reconstruction process ensures that the data is accurately represented, allowing for precise measurement of velocity derivatives and pressure gradients.

### Derivation of 4D Flow Velocity Field

The multi-dimensional PC-MR sequence provides time-resolved (4D) flow velocity data, which captures the dynamic changes in blood flow throughout the cardiac cycle. This 4D flow velocity field is essential for accurately quantifying the pressure gradient within the vessel segment and determining FFR values.

### Calculation of Velocity Derivatives and Pressure Gradient Field

The processor calculates the derivatives of the velocity field to determine the spatial gradients of blood flow velocities. These velocity derivatives are then used to compute the pressure gradient field, which provides a detailed map of the pressure distribution within the coronary arteries. The pressure gradient field is crucial for identifying areas of high resistance and quantifying the severity of stenosis.

### Obtaining Transtenotic Pressure Difference

The transtenotic pressure difference (ΔP) is calculated by integrating the pressure gradient over the length of the vessel segment containing the stenosis. This value is then used to determine the FFR, which provides a quantitative measure of the functional impact of the stenosis on blood flow.

### Volume of Interest in Subject

The volume of interest (VOI) is defined as the region within the coronary arteries where velocity field data is acquired and processed. The VOI is carefully selected to ensure that it includes the entire length of the vessel segment being assessed, from the proximal to the distal end.

### Imaging Parameters

The imaging parameters for the multi-dimensional PC-MR sequence are optimized to provide high-resolution velocity field data. These parameters include:
- In-plane resolution: 0.58-0.67 mm
- Slice thickness: 3.2 mm
- Flip angle: 15°
- Acquisition time per phase: 65-71 ms, with the first phase strictly coinciding with the quiescent period
- Scan time: 1-3 minutes per slice

### Cardiac Phase

The multi-dimensional PC-MR sequence is designed to acquire data at multiple cardiac phases, including mid-diastole and end-expiration. This ensures that the velocity field data is captured during periods of minimal motion, improving the accuracy of the measurements.

### Scan Time

The scan time for each slice ranges from 1 to 3 minutes, depending on the imaging parameters and the complexity of the VOI. The total scan time is minimized by optimizing the acquisition window and using ECG-triggering and navigator-gating techniques.

### Acquisition Window

The acquisition window is carefully selected to ensure that data is collected during periods of minimal motion. This is achieved through ECG-triggering, which synchronizes the acquisition with the cardiac cycle, and navigator-gating, which monitors respiratory motion and triggers acquisitions during quiescent periods.

### ECG-Triggering and Navigator-Gating

ECG-triggering and navigator-gating are essential for ensuring accurate data acquisition. ECG-triggering synchronizes the acquisition with the R-wave of the electrocardiogram (ECG), while navigator-gating monitors respiratory motion and triggers acquisitions during periods of minimal movement. This combination of techniques minimizes motion artifacts and improves the quality of the velocity field data.

### MRI System

The MRI system used for executing the multi-dimensional PC-MR sequence is a 3T MAGNETOM Verio (Siemens) equipped with advanced hardware and software components. The system includes:
- A high-performance gradient coil for generating strong, uniform magnetic fields
- Advanced RF coils for signal acquisition
- High-speed data acquisition and processing units
- User-friendly interface for controlling the imaging sequence

### Processor and Its Functions

The processor within the MRI system performs several critical functions, including:
- Data acquisition and preprocessing
- Application of Navier-Stokes equations to derive pressure gradients
- Image reconstruction using Fourier transform methods
- Calculation of velocity derivatives and pressure gradient fields
- Integration of pressure gradients to obtain transtenotic pressure differences
- Determination of FFR values

### Computer and Its Functions

The computer system associated with the MRI system includes:
- High-performance computing hardware for data processing
- Advanced software for image analysis and visualization
- User interfaces for controlling the imaging sequence and viewing results
- Storage capabilities for archiving acquired data and processed images

### Non-Transitory Machine-Readable Medium

A non-transitory machine-readable medium is provided with instructions for performing the method of quantifying FFR using multi-dimensional PC-MR sequences. The medium may include:
- Software applications for controlling the MRI system
- Algorithms for processing velocity field data
- User guides and documentation for system operation

### Method for Diagnosing Cardiovascular Disease

The method described herein can be used to diagnose cardiovascular disease by quantifying FFR values in patients with suspected coronary artery stenosis. The noninvasive nature of the method makes it suitable for routine clinical use, allowing for early detection and monitoring of disease progression.

### Stenosis

Stenosis refers to a narrowing or blockage within a blood vessel. The severity of stenosis can be classified as:
- **Mild:** 0-49% narrowing
- **Moderate:** 50-69% narrowing
- **Severe:** 70-100% narrowing

### Alternative Imaging Systems

While the present invention focuses on using MRI for quantifying FFR, alternative imaging systems such as ultrasound and computed tomography (CT) may also be used. However, MRI offers several advantages in terms of spatial resolution, lack of ionizing radiation, and the ability to perform multi-dimensional PC-MR sequences.

### Equivalents to Methods and Materials

Various equivalents to the methods and materials described herein are within the scope of this invention. For example, different MRI systems with similar capabilities may be used, and variations in imaging parameters and data processing techniques may be employed without departing from the spirit of the invention.

## EXAMPLES

### Example 1

A 3D phase-contrast magnetic resonance (PC-MR) sequence was designed to acquire velocity field data within the coronary arteries. The acquisition window was set to capture data during mid-diastole and end-expiration, with ECG-triggering and navigator-gating techniques used to minimize motion artifacts. Imaging parameters included an in-plane resolution of 0.58-0.67 mm, a slice thickness of 3.2 mm, and a flip angle of 15°. The acquisition time per phase was 65-71 ms, with the first phase strictly coinciding with the quiescent period. The scan time for each slice ranged from 1 to 3 minutes.

### Example 2

Phantom studies were conducted using a flow phantom containing gadolinium-doped water flowing at 300 mL/min through a silicone tubing with an internal diameter of 4.8 mm and a 40% stenosis. The PC-MR sequence was used to acquire velocity field data, which was then processed to calculate the pressure gradient within the stenotic region. The results demonstrated excellent agreement with theoretical values, validating the accuracy of the method.

### Example 3

A clinical study was conducted on a cohort of patients with suspected coronary artery disease. Multi-dimensional PC-MR sequences were used to quantify FFR values in the proximal left anterior descending artery (LAD). The noninvasive nature of the method allowed for repeated measurements over time, enabling clinicians to monitor disease progression and assess the effectiveness of treatment.

### Conclusion

The present invention provides a noninvasive method for quantifying Fractional Flow Reserve (FFR) using multi-dimensional phase-contrast magnetic resonance (PC-MR) sequences. The method offers several advantages over traditional invasive techniques, including higher spatial resolution, lack of ionizing radiation, and the ability to perform time-resolved flow measurements. This approach has the potential to revolutionize the diagnosis and management of cardiovascular disease by providing accurate, reliable FFR values in a clinical setting.