Below is the drafted patent application following the provided outline with formal patent language:

---

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

Off-resonance blurring presents a significant challenge in radial ultrashort echo time magnetic resonance imaging (MRI) sequences, particularly in techniques such as zero echo time (ZTE) and sweep imaging with Fourier transformation. This blurring arises from fast signal decay of extremely short T2* spins and becomes more pronounced at higher static magnetic field strengths due to increased chemical shifts and accelerated T2* decay. While higher readout bandwidths can mitigate this issue, conventional ZTE techniques impose limitations by requiring identical excitation and readout bandwidths. This constraint leads to elevated specific absorption rate (SAR) and radiofrequency (RF) peak power, which hinder clinical applicability.  

Pointwise-encoding time reduction with radial acquisition (PETRA) was developed to address the slow transmit/receive (T/R) switching limitations of clinical MRI scanners by combining ZTE with single-point imaging (SPI). However, PETRA still suffers from drawbacks when higher bandwidths are employed, including an expanded missing k-space center region and prolonged SPI acquisition times. These inefficiencies highlight the need for an improved solution that maintains high readout bandwidth while reducing excitation bandwidth to alleviate SAR and RF peak power constraints.  

## SUMMARY OF THE INVENTION  

The present invention introduces a gradient-modulated PETRA (GM-PETRA) technique that decouples excitation and readout bandwidths, enabling independent control over these parameters. By ramping up gradient amplitudes after excitation, GM-PETRA achieves higher readout bandwidths while maintaining lower excitation bandwidths, thereby reducing SAR and RF peak power. This approach also minimizes the missing k-space center region, improving imaging efficiency.  

Key advantages of GM-PETRA include enhanced tolerance to off-resonance blurring and short T2* signal decay, reduced artifacts from chemical shift and susceptibility differences, and improved visualization of fine anatomical structures. The technique is compatible with existing clinical MRI systems without requiring hardware modifications, making it a practical solution for high-field imaging applications.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Conventional PETRA Technique  

The conventional PETRA technique acquires k-space data using a hybrid radial and SPI sampling strategy. Signal acquisition begins immediately after a hard pulse excitation, followed by a T/R switching delay. Gradients with constant amplitude (Gex) are applied during readout, and missing k-space center points are acquired using SPI. The size of the missing region is determined by the product of the excitation bandwidth and the time delay before readout (tk).  

### Gradient-Modulated PETRA Technique  

In GM-PETRA, the gradient amplitude is modulated after the receiver gate is activated, ramping up from Gex to a maximum value (Gmax). This modulation enables higher readout bandwidths while maintaining a lower excitation bandwidth. A short delay time (td) with flat gradients is inserted to facilitate SPI acquisitions for missing k-space points.  

### Amplitude-Modulated Gradient  

The amplitude-modulated gradient in GM-PETRA allows flexible control over readout bandwidth independent of excitation bandwidth. By increasing the gradient amplitude during readout, k-space is sampled more rapidly, reducing off-resonance blurring and preserving short T2* signals.  

### Data Acquisition Process  

Data acquisition in GM-PETRA involves the following steps:  
1. Application of a hard pulse with a low excitation bandwidth.  
2. Activation of the receiver gate after a T/R switching delay.  
3. Ramping up of gradient amplitude from Gex to Gmax during readout.  
4. Insertion of a delay time (td) for SPI acquisitions to fill the missing k-space center region.  

### Relationship Between RF Peak Power and SAR  

The RF peak power (B1max) and SAR are proportional to the excitation bandwidth (BWex) and inversely proportional to the pulse width (pw). By reducing BWex while maintaining high readout bandwidth, GM-PETRA significantly lowers SAR and RF peak power, addressing critical limitations in clinical applications.  

### K-Space Sampling Pattern  

GM-PETRA employs a hybrid k-space sampling pattern combining radial trajectories for peripheral k-space and SPI for the center region. The radial sampling is accelerated by the ramped gradient, while SPI ensures complete coverage of the central k-space.  

### Missing Central K-Space Points  

The number of missing central k-space points in GM-PETRA is determined by the product of tk and BWex. By keeping BWex low, the missing region is minimized, reducing the required SPI acquisition time.  

### SPI Acquisitions  

SPI acquisitions in GM-PETRA are performed during the delay time (td) with stepwise reduction of gradient amplitude. This ensures accurate sampling of the k-space center, which is critical for high-fidelity image reconstruction.  

### Advantages of Gradient-Modulated PETRA  

GM-PETRA offers several advantages over conventional PETRA, including:  
- Reduced off-resonance blurring and improved image sharpness.  
- Lower SAR and RF peak power, enabling safer clinical use.  
- Faster k-space sampling, preserving short T2* signals.  
- Compatibility with standard MRI hardware.  

### Example Implementation  

An exemplary implementation of GM-PETRA involves the following parameters:  
- Excitation bandwidth: 60 kHz.  
- Maximum readout bandwidth: 125–200 kHz.  
- Flip angle: 5°.  
- Repetition time (TR): 5 ms.  
- Total acquisitions: 65,536 (including 123 SPI acquisitions).  

### Example Images  

Images acquired using GM-PETRA demonstrate superior visualization of fine anatomical structures, such as the inner ear and apple mesocarp, compared to conventional PETRA and 3D gradient-echo (GRE) techniques. GM-PETRA effectively reduces blurring artifacts caused by susceptibility differences and short T2* decay.  

### Summary of Benefits  

GM-PETRA provides a robust solution for high-field MRI by combining the benefits of high readout bandwidth with low excitation bandwidth. This innovation addresses key limitations of existing techniques, offering improved image quality, reduced SAR, and enhanced clinical feasibility.  

### Example: Ex Vivo Equine Knee Imaging  

#### GM-PETRA Sequence Parameters  

For ex vivo equine knee imaging, GM-PETRA was implemented with the following parameters:  
- Excitation bandwidth: 60 kHz.  
- Maximum readout bandwidth: 125 kHz.  
- Flip angle: 5°.  
- TR: 5 ms.  
- Slew rate: 100 mT/m/ms.  

#### PETRA Sequence Parameters  

Conventional PETRA was performed with excitation bandwidths of 60 kHz and 120 kHz for comparison. The 120 kHz acquisition required 925 SPI acquisitions due to the larger missing k-space region.  

#### Gradient Modulation  

In GM-PETRA, the gradient amplitude was ramped from 60 kHz to 125 kHz during readout, with a slew rate of 100 mT/m/ms to minimize eddy current effects.  

#### Image Reconstruction Process  

Image reconstruction was performed offline using a custom program written in C++/CUDA. K-space sampling density was compensated using iterative density correction, and the data were reconstructed using nonuniform fast Fourier transform.  

#### Example Images  

GM-PETRA images exhibited reduced off-resonance blurring and improved edge sharpness compared to conventional PETRA. The technique also demonstrated superior sensitivity to short T2* signals, as evidenced by the visualization of fine structures in the equine knee.  

#### Comparison of GM-PETRA and PETRA Images  

Side-by-side comparison revealed that GM-PETRA outperformed PETRA in terms of artifact reduction and structural detail. The 120 kHz PETRA acquisition showed comparable off-resonance performance but incurred higher SAR and longer SPI acquisition times.  

#### Advantages of GM-PETRA  

GM-PETRA’s ability to decouple excitation and readout bandwidths provides significant advantages, including:  
- Reduced SAR and RF peak power.  
- Shorter SPI acquisition times.  
- Improved image quality.  

#### MRI System  

The invention is implemented on a 7T MRI scanner (Siemens, Erlangen, Germany) equipped with a head coil featuring 8 transceiver and 24 receiver channels.  

#### Operator Workstation  

The operator workstation interfaces with the MRI system to control sequence parameters, monitor imaging progress, and manage data storage.  

#### Pulse Sequence Server  

The pulse sequence server executes the GM-PETRA sequence, coordinating gradient modulation, RF excitation, and data acquisition.  

#### Data Acquisition Server  

The data acquisition server receives and processes raw k-space data from the receiver coils, ensuring accurate sampling and minimal noise.  

#### Data Processing Server  

The data processing server performs offline reconstruction, including density correction and nonuniform fast Fourier transform.  

#### Data Store Server  

The data store server archives reconstructed images and raw data for subsequent analysis and review.  

#### Physiological Acquisition Controller  

The physiological acquisition controller monitors subject vitals during scanning, ensuring safety and compliance with SAR limits.  

#### Scan Room Interface Circuit  

The scan room interface circuit facilitates communication between the MRI system and peripheral devices, such as gradient amplifiers and RF coils.  

#### Patient Positioning System  

The patient positioning system ensures accurate and reproducible subject placement within the MRI bore, optimizing image quality.  

#### RF System  

The RF system generates and receives RF pulses, with B1+ shimming optimized for uniform excitation over the region of interest.  

#### Gradient System  

The gradient system provides the modulated gradients required for GM-PETRA, with precise control over amplitude and slew rate.  

#### Magnet Assembly  

The magnet assembly generates the high static magnetic field (7T) necessary for high-resolution imaging.  

---  

This patent application provides a comprehensive description of the invention, its advantages, and exemplary implementations, adhering to the outlined structure and formal patent language.