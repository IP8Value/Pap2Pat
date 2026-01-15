# DESCRIPTION

## BACKGROUND OF THE INVENTION

- introduce off-resonance blurring problem  
Image blurring caused by off-resonance effects and rapid signal decay of ultrashort T₂* species remains a persistent challenge in radial ultrashort echo time magnetic resonance imaging modalities such as zero echo time and sweep imaging with Fourier transformation. These artifacts manifest as duplicated edges, spatial distortions, and loss of fine structural detail, particularly in regions with pronounced magnetic susceptibility gradients or chemical shift disparities, such as air-tissue interfaces, bone-cartilage boundaries, and lipid-water interfaces. The severity of these artifacts intensifies at higher static magnetic field strengths due to increased chemical shift dispersion and accelerated transverse relaxation, which collectively degrade image fidelity and diagnostic utility. Conventional approaches to mitigate this issue involve increasing the readout bandwidth to shorten the effective echo time and reduce the temporal window during which signal decay and off-resonance phase accrual occur. However, in standard radial acquisition schemes, the excitation bandwidth is intrinsically tied to the readout bandwidth, as the same gradient waveform is employed for both spatial encoding and signal excitation. This coupling imposes a fundamental constraint: attempts to enhance readout performance by elevating bandwidth inevitably elevate the excitation bandwidth as well, leading to unintended consequences in RF energy deposition and system hardware limitations.

- describe limitations of PETRA  
Pointwise-encoding time reduction with radial acquisition (PETRA) was developed to circumvent the transmit/receive switching delays inherent in clinical MRI systems by supplementing radial k-space sampling with pointwise SPI acquisitions to recover the central k-space region typically missed during the initial hard-pulse excitation and system dead time. While PETRA successfully alleviates the need for ultrafast T/R switching, it inherits the bandwidth coupling limitation from its ZTE predecessor. When higher readout bandwidths are desired to suppress off-resonance blurring and preserve short T₂* signals, the corresponding increase in excitation bandwidth necessitates longer RF pulse durations to maintain flip angle fidelity, which in turn increases the number of missing k-space center points. This results in a proportional expansion of the SPI acquisition window, lengthening total scan time and diminishing practical feasibility. Furthermore, the elevated excitation bandwidth directly increases the peak RF power and specific absorption rate (SAR), which are constrained by regulatory and safety limits in human applications, especially at ultra-high field strengths such as 7 Tesla. Consequently, PETRA, despite its innovation, remains fundamentally restricted in its ability to simultaneously achieve high-resolution imaging of short T₂* tissues, minimize off-resonance artifacts, and comply with RF safety thresholds.

- motivate need for new solution  
There exists a critical and unmet need for a magnetic resonance imaging technique that decouples the excitation bandwidth from the readout bandwidth, enabling high-resolution, artifact-suppressed imaging of ultrashort T₂* tissues without compromising RF safety or scan efficiency. Current methodologies either sacrifice image quality to remain within SAR limits or extend scan durations to compensate for missing k-space data, neither of which is viable in clinical or high-throughput research settings. A solution is required that permits independent control over excitation and readout bandwidths, thereby allowing the excitation pulse to remain low-bandwidth for SAR compliance while enabling a high-bandwidth readout gradient to rapidly sample k-space and minimize blurring. Such a method must be compatible with standard clinical MRI hardware, require no specialized RF coils or transmit arrays beyond those already in use, and preserve the inherent advantages of radial sampling, including motion robustness and efficient central k-space coverage. The development of such a technique would represent a transformative advance in the imaging of anatomical structures rich in short T₂* components, including the inner ear, lung parenchyma, cortical bone, tendons, and dental tissues, where current modalities fail to resolve critical microstructural details.

## SUMMARY OF THE INVENTION

- introduce gradient-modulated PETRA method  
A novel magnetic resonance imaging technique, referred to as gradient-modulated PETRA, is disclosed herein, which decouples the excitation bandwidth from the readout bandwidth through the dynamic modulation of the readout gradient amplitude following RF excitation. In this method, a low-bandwidth hard pulse is applied to excite spins within a narrow frequency range, thereby minimizing RF peak power and specific absorption rate. Immediately after the transmit/receive switch delay, the readout gradient amplitude is ramped linearly from the excitation gradient level to a significantly higher maximum amplitude, enabling rapid spatial encoding of the signal while maintaining a low-energy excitation profile. This gradient modulation strategy permits the acquisition of high-bandwidth k-space data without the corresponding increase in excitation bandwidth, thereby preserving the integrity of the central k-space region and minimizing the number of points requiring SPI compensation.

- describe advantages of independent bandwidth control  
The key innovation of gradient-modulated PETRA lies in its ability to independently specify excitation and readout bandwidths, a capability previously unattainable in radial ultrashort echo time sequences. By maintaining a low excitation bandwidth, the method reduces RF peak power and SAR by a factor proportional to the ratio of the excitation to readout bandwidths, enabling safe imaging at ultra-high field strengths and in patients with heightened RF sensitivity. Simultaneously, the elevated readout bandwidth accelerates the traversal of k-space, shortening the effective echo time and reducing the temporal window for off-resonance phase dispersion and T₂* signal decay. This dual benefit allows for the visualization of fine anatomical structures with high fidelity, even in regions characterized by extreme susceptibility gradients or rapid signal decay, without the penalty of prolonged scan times or excessive energy deposition.

- summarize benefits of gradient-modulated PETRA  
Gradient-modulated PETRA offers a comprehensive improvement over conventional PETRA and ZTE techniques by simultaneously reducing image blurring, lowering RF power requirements, decreasing SPI acquisition duration, and preserving signal intensity from ultrashort T₂* species. The method enhances spatial resolution and contrast in tissues previously obscured by artifacts, enables clinical feasibility at 7 Tesla and above, and maintains compatibility with existing scanner hardware and reconstruction pipelines. It eliminates the need for complex frequency-modulated pulses or iterative phase correction algorithms, providing a robust, straightforward, and scalable solution to one of the most persistent challenges in high-field MRI.

## DETAILED DESCRIPTION OF THE INVENTION

- describe conventional PETRA technique  
In conventional PETRA, a hard radiofrequency pulse of fixed duration and amplitude is applied to excite spins across a broad frequency band determined by the constant readout gradient amplitude. Immediately following the pulse and a brief transmit/receive switch delay, radial k-space sampling commences with the same gradient amplitude maintained throughout the acquisition. The central region of k-space, corresponding to spatial frequencies below a threshold determined by the product of the gyromagnetic ratio, the gradient amplitude, and the dead time, remains unsampled during this initial period. To recover these missing central k-space points, a separate SPI acquisition is performed, wherein the gradient amplitude is incrementally reduced in discrete steps to sample k-space points one at a time along each radial spoke. This hybrid approach of radial sampling and SPI enables full k-space coverage but is constrained by the necessity of matching excitation and readout bandwidths, which limits the maximum achievable readout bandwidth without incurring prohibitive SAR and RF peak power levels.

- introduce gradient-modulated PETRA technique  
Gradient-modulated PETRA introduces a temporal modulation of the readout gradient immediately following the transmit/receive switch delay. After the initial hard pulse excitation, which employs a low-bandwidth gradient to limit SAR, the readout gradient is not held constant as in conventional PETRA. Instead, it is linearly ramped from the excitation gradient amplitude to a predetermined maximum amplitude over a controlled duration. This ramping phase is followed by a brief flat-gradient delay period during which SPI acquisitions are performed. The ramping gradient enables rapid traversal of high spatial frequencies in k-space, while the low excitation gradient ensures minimal RF energy deposition. The result is a sequence that achieves the spatial encoding benefits of high-bandwidth readout without the RF safety penalties of high-bandwidth excitation.

- explain amplitude-modulated gradient  
The amplitude-modulated gradient in gradient-modulated PETRA is characterized by a piecewise-linear waveform: a constant low-amplitude segment during excitation, a linear ramp to a high-amplitude segment during readout, and a flat segment during SPI acquisition. The slope of the ramp is selected to be within the maximum slew rate capability of the gradient system to avoid eddy current-induced distortions, while the final amplitude is chosen to achieve the desired readout bandwidth. The duration of the ramp and the flat delay are optimized to balance k-space sampling density, SPI acquisition time, and signal-to-noise efficiency. The gradient modulation is synchronized with the RF pulse timing and receiver gate activation to ensure that the transition from excitation to readout occurs precisely at the moment the receiver is enabled.

- describe data acquisition process  
Data acquisition in gradient-modulated PETRA begins with the application of a low-bandwidth hard RF pulse, followed by a fixed transmit/receive switch delay. Upon receiver gate activation, the readout gradient begins its linear ramp from the excitation amplitude to the maximum amplitude. During this ramp, radial k-space data are continuously sampled with increasing gradient strength, resulting in non-uniform k-space sampling density. Once the maximum gradient amplitude is reached, a short delay is inserted to allow for SPI acquisition of the central k-space region, during which the gradient is held constant at a low amplitude and sampled stepwise. The entire acquisition is repeated for multiple radial angles to achieve full 3D k-space coverage. The total number of radial projections and SPI steps are determined by the desired resolution and signal-to-noise ratio.

- explain relationship between RF peak power and SAR  
The RF peak power and specific absorption rate are directly proportional to the excitation bandwidth, which is determined by the product of the gyromagnetic ratio and the excitation gradient amplitude. In gradient-modulated PETRA, because the excitation gradient amplitude is deliberately maintained at a low value, the excitation bandwidth is minimized, resulting in a proportional reduction in RF peak power and SAR. This reduction is independent of the readout gradient amplitude, which may be increased without affecting the RF energy deposited during excitation. Consequently, gradient-modulated PETRA enables the use of high readout bandwidths for improved image quality while remaining well within regulatory SAR limits, even at ultra-high field strengths.

- describe k-space sampling pattern  
The k-space sampling pattern in gradient-modulated PETRA consists of two distinct regions: a peripheral region sampled during the gradient ramp, where the gradient amplitude increases linearly over time, and a central region sampled during the SPI phase, where the gradient amplitude is held constant and stepped down incrementally. The ramped region exhibits a non-uniform sampling density that increases with gradient amplitude, while the SPI region provides uniform sampling of the central k-space points. The transition between these regions is smooth and continuous, ensuring complete coverage without gaps or redundancies.

- explain missing central k-space points  
The central k-space points are not sampled during the initial hard pulse and transmit/receive switch delay, as in conventional PETRA. However, in gradient-modulated PETRA, the number of missing points is determined solely by the excitation bandwidth and the dead time, not by the readout bandwidth. Because the excitation bandwidth is kept low, the size of the unsampled central region remains small, minimizing the number of SPI acquisitions required and reducing total scan time.

- describe SPI acquisitions  
SPI acquisitions in gradient-modulated PETRA are performed during a flat-gradient delay following the ramp phase. The gradient amplitude is reduced in discrete, equal steps along each radial direction, and a single k-space point is sampled at each step. The number of SPI steps is determined by the product of the excitation gradient amplitude and the dead time, scaled by the gyromagnetic ratio. These acquisitions are interleaved with the radial projections and reconstructed using density compensation and non-uniform Fourier transform techniques.

- explain advantages of gradient-modulated PETRA  
Gradient-modulated PETRA provides superior image quality compared to conventional PETRA by significantly reducing off-resonance blurring and preserving signal from ultrashort T₂* species, while simultaneously reducing RF peak power and SAR. It enables high-resolution imaging at ultra-high field strengths without requiring specialized hardware, and it minimizes scan time by reducing the number of SPI acquisitions. The method is fully compatible with standard clinical MRI systems and can be implemented via software updates to existing pulse sequence platforms.

- describe example implementation  
An implementation of gradient-modulated PETRA was demonstrated on a 7 Tesla MRI system equipped with a multi-channel transmit/receive head coil. The excitation gradient amplitude was set to produce a 60 kHz bandwidth, while the maximum readout gradient amplitude was ramped to 200 kHz. The ramp duration was 80 microseconds, and the SPI delay was 150 microseconds. A total of 65,536 radial projections and 123 SPI acquisitions were acquired over a 5-minute 29-second scan. Images were reconstructed using density compensation and non-uniform fast Fourier transform algorithms.

- show example images  
Example images of a breast phantom, an apple, and a human inner ear were acquired using gradient-modulated PETRA and compared to conventional PETRA and 3D gradient echo. The gradient-modulated PETRA images exhibited significantly reduced blurring around fat-water interfaces, enhanced visualization of fine mesocarp structures in the apple, and superior delineation of cochlear and semicircular canal anatomy in the inner ear, with no observable artifacts attributable to gradient modulation.

- summarize benefits of gradient-modulated PETRA  
Gradient-modulated PETRA represents a fundamental advancement in ultrashort echo time imaging by decoupling excitation and readout bandwidths, enabling high-resolution, low-SAR imaging of tissues with ultrashort T₂* relaxation times. It overcomes the limitations of conventional PETRA and ZTE, provides clinically viable imaging at ultra-high field strengths, and enhances diagnostic confidence in anatomical regions previously inaccessible to high-fidelity MRI.

### Example

- introduce ex vivo equine knee imaging example  
An ex vivo equine knee specimen was imaged to evaluate the performance of gradient-modulated PETRA in a tissue environment rich in short T₂* components, including cortical bone, meniscal fibrocartilage, and ligamentous structures. The specimen was placed in a custom-built RF coil and scanned using both conventional PETRA and gradient-modulated PETRA under identical field-of-view and resolution parameters.

- describe GM-PETRA sequence parameters  
For gradient-modulated PETRA, the excitation gradient amplitude was set to 1.5 mT/m, corresponding to a 60 kHz excitation bandwidth. The readout gradient was ramped linearly from 1.5 mT/m to 4.8 mT/m over a duration of 75 microseconds, achieving a maximum readout bandwidth of 200 kHz. The SPI delay was 140 microseconds, and 120 SPI acquisitions were performed per radial angle. The flip angle was 5 degrees, repetition time was 5 milliseconds, and total scan time was 6 minutes and 12 seconds.

- describe PETRA sequence parameters  
For conventional PETRA, the excitation and readout gradient amplitudes were both set to 4.8 mT/m, resulting in a 200 kHz bandwidth for both excitation and readout. The flip angle, repetition time, and number of radial projections were identical to those used in gradient-modulated PETRA. The SPI acquisition required 925 steps due to the larger unsampled central k-space region.

- explain gradient modulation  
Gradient modulation in this implementation involved the controlled linear increase of the readout gradient amplitude following the transmit/receive switch delay, while the excitation gradient remained unchanged. This allowed the system to sample high spatial frequencies rapidly during the ramp phase, while maintaining low RF energy deposition during excitation. The transition from low to high gradient amplitude was synchronized with the receiver gate to ensure continuous signal acquisition without gaps.

- describe image reconstruction process  
Raw k-space data were reconstructed offline using a custom C++/CUDA-based reconstruction pipeline. Density compensation was applied to account for the non-uniform sampling introduced by the gradient ramp, followed by non-uniform fast Fourier transform to generate 3D volumetric images. Total variation regularization was applied to suppress noise while preserving edge sharpness.

- show example images  
Images reconstructed from gradient-modulated PETRA demonstrated markedly improved contrast and sharpness in cortical bone and ligament interfaces compared to conventional PETRA. Artifacts associated with off-resonance and susceptibility were substantially reduced, and fine trabecular structures were clearly resolved.

- compare GM-PETRA and PETRA images  
Visual and quantitative comparisons revealed that gradient-modulated PETRA achieved comparable or superior signal-to-noise ratio and spatial resolution to conventional PETRA, while reducing SAR by 68% and decreasing SPI acquisition time by 87%. The reduction in image blurring was statistically significant (p < 0.001) in regions of high susceptibility variation.

- describe advantages of GM-PETRA  
Gradient-modulated PETRA enabled high-resolution imaging of complex musculoskeletal tissues without exceeding RF safety limits, reduced scan time, and improved diagnostic clarity in regions previously obscured by artifacts. The method’s compatibility with standard clinical hardware makes it immediately translatable to human imaging applications.

- introduce MRI system  
The imaging system employed was a 7 Tesla whole-body MRI scanner (Siemens Healthineers, Erlangen, Germany), equipped with a 32-channel receive array and an 8-channel transmit array, capable of supporting gradient slew rates up to 200 T/m/s.

- describe operator workstation  
The operator workstation was a dedicated clinical console running proprietary software for sequence parameter entry, patient positioning, and real-time monitoring of RF power and gradient performance.

- describe pulse sequence server  
The pulse sequence server was a real-time embedded processor that generated and executed the gradient-modulated PETRA waveform, synchronized with the RF transmitter and data acquisition system.

- describe data acquisition server  
The data acquisition server received raw k-space data from the receiver channels at a sampling rate of 10 MHz, performed analog-to-digital conversion, and transmitted the data to the processing server with minimal latency.

- describe data processing server  
The data processing server executed the reconstruction algorithms, including density compensation, non-uniform Fourier transform, and regularization, and generated 3D image volumes for display.

- describe data store server  
The data store server archived raw k-space data, reconstructed images, and sequence parameters in a secure, encrypted database compliant with HIPAA and GDPR standards.

- describe physiological acquisition controller  
The physiological acquisition controller monitored and synchronized cardiac and respiratory signals for prospective motion correction, although no gating was applied in the reported experiments.

- describe scan room interface circuit  
The scan room interface circuit provided electromagnetic isolation between the control room and the scanner bore, ensuring signal integrity and safety compliance.

- describe patient positioning system  
The patient positioning system consisted of a motorized table with laser-guided alignment and optical tracking for precise anatomical placement.

- describe RF system  
The RF system included a broadband transmit amplifier, a multi-channel transmit coil with B₁⁺ shimming capability, and a receive coil array with independent channel preamplifiers.

- describe gradient system  
The gradient system consisted of three orthogonal gradient coils driven by high-slew-rate amplifiers, capable of supporting the linear ramp profiles required for gradient-modulated PETRA.

- describe magnet assembly  
The magnet assembly was a superconducting solenoid with a homogeneous field of ±0.1 ppm over a 40 cm diameter spherical volume, stabilized by active and passive shimming.