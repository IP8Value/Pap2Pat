# DESCRIPTION

## PRIORITY DATA

- claim priority to previous applications  
- incorporate previous applications by reference  

This patent application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/478,901, filed on June 12, 2023, and U.S. Provisional Patent Application No. 63/512,887, filed on October 17, 2023, both of which are hereby incorporated by reference in their entirety. The disclosures contained within these provisional applications, including but not limited to system architectures, illumination schemes, computational reconstruction algorithms, and calibration methodologies, are integral to the invention described herein and form the foundational basis for the claims presented in this application. All subject matter disclosed in the referenced provisional applications is hereby fully incorporated into this specification as if originally set forth herein, ensuring continuity of disclosure and preserving the earliest possible effective filing date for all inventive elements. Any reference to prior art, experimental data, or technical implementations contained in the provisional applications is to be understood as part of the enabling disclosure of the present invention, and no limitation is intended by the omission of redundant detail in this filing. The invention disclosed herein builds upon and extends the scope of the earlier-filed applications by introducing novel system configurations, enhanced parallel processing architectures, and improved aberration correction techniques that collectively enable high-throughput, multi-modal imaging at unprecedented scale and fidelity.

## FEDERALLY SPONSORED RESEARCH OR DEVELOPMENT

- disclose government funding  

This invention was developed with partial support under Award Number R01GM142012 from the National Institute of General Medical Sciences, National Institutes of Health. The United States Government has certain rights in this invention pursuant to the terms of the aforementioned award. The views, opinions, and findings contained in this disclosure are those of the inventors and should not be construed as an official position of the Department of Health and Human Services or the National Institutes of Health. No endorsement by the federal government is implied or intended. All experimental data, system designs, and algorithmic implementations described herein were conducted under the auspices of this federally funded research program, and the resulting technological advancements represent a direct outcome of the objectives and deliverables established under the grant agreement. The invention has been reduced to practice using resources and infrastructure supported by federal funds, and all intellectual property rights arising from this work are subject to the provisions of the Bayh-Dole Act and applicable federal regulations governing inventions made with federal support.

## TECHNICAL FIELD

- define technical field of digital imaging  

The present invention relates generally to the field of digital imaging, specifically to high-throughput, parallelized imaging systems designed for the simultaneous acquisition of high-resolution, aberration-corrected optical data from multi-well plate formats commonly used in biomedical research and pharmaceutical screening. More particularly, the invention pertains to an integrated imaging platform that combines Fourier ptychographic microscopy with fluorescence imaging in a massively parallel architecture, enabling the rapid, label-free, and computationally refocused visualization of cellular morphology and molecular expression across hundreds of biological samples in a single acquisition cycle. The system is engineered to overcome the inherent limitations of conventional well plate readers and scanning microscopes by leveraging computational imaging techniques to achieve superior resolution, extended depth of field, and robust aberration correction without requiring mechanical re-focusing or precision optical alignment for each individual well. The invention is particularly suited for applications in cell biology, drug discovery, high-content screening, and live-cell phenotyping where large-scale, quantitative imaging of adherent cell populations under controlled conditions is required.

## BACKGROUND

- describe limitations of multi-well plate readers  
- motivate need for image information  
- describe conventional imaging techniques  

Traditional multi-well plate readers are widely employed in high-throughput biological assays due to their speed, simplicity, and cost-effectiveness, yet they are fundamentally limited to providing bulk optical measurements such as absorbance or integrated fluorescence intensity, offering no spatial information regarding the distribution, morphology, or behavior of individual cells within each well. This lack of imaging capability renders them incapable of distinguishing between heterogeneous cell populations, detecting subtle morphological changes, or identifying localized patterns of protein expression, all of which are critical for meaningful biological interpretation. While the demand for image-based analysis has grown substantially in fields such as cancer research, neurobiology, and regenerative medicine, existing imaging systems that attempt to bridge this gap typically rely on a single microscope objective that mechanically scans across the plate, sequentially imaging each well. These systems, though capable of delivering high-resolution images, suffer from prohibitively long acquisition times—often exceeding eight minutes per plate—making them incompatible with the throughput requirements of large-scale screening workflows. Furthermore, conventional imaging platforms are highly sensitive to variations in well depth, plate warping, and liquid meniscus formation, which induce defocus and optical aberrations that degrade image quality. Manual or mechanical refocusing is often required per well, further slowing the process and introducing inconsistency. Even advanced systems that incorporate autofocus mechanisms remain constrained by the physical limits of optical depth of field and the mechanical inertia of moving objectives, resulting in a fundamental trade-off between image fidelity and throughput that has long impeded the adoption of imaging in routine high-throughput screening environments.

## SUMMARY

- introduce imaging system  
- describe illumination system  
- describe optical system  
- describe imaging system  
- describe plate receiver system  
- describe controller and image reconstruction process  

The present invention introduces a novel imaging system capable of simultaneously acquiring high-resolution, aberration-corrected bright-field and fluorescence images from all wells of a 96-well plate in under two minutes, without mechanical scanning or manual refocusing. The system comprises a compact, parallelized architecture featuring 96 independently aligned imaging channels, each consisting of a custom-designed miniature objective, a consumer-grade CMOS image sensor, and a dedicated illumination path, all arranged in a fixed grid corresponding to the well layout of a standard 96-well plate. An LED matrix positioned above the plate provides structured illumination, with each LED element capable of illuminating multiple wells simultaneously through a shared illumination geometry that ensures uniform and non-overlapping excitation patterns across the entire plate. The optical system employs finite-conjugate lens designs with a 4× magnification and a numerical aperture of 0.23, optimized for compactness and cost-effective manufacturing using injection-molded plastic elements. A precision plate receiver system holds the multi-well plate in a fixed, rigid orientation relative to the imaging array, eliminating the need for dynamic focusing by maintaining a consistent object-to-sensor distance across all channels. The system further incorporates a centralized controller that orchestrates the sequential activation of illumination sources, synchronizes image capture across all 96 sensors, and executes a parallelized Fourier ptychographic reconstruction algorithm to computationally synthesize high-resolution, aberration-corrected images from the acquired low-resolution intensity data. This reconstruction process leverages iterative phase retrieval techniques to recover both amplitude and phase information, effectively extending the depth of field beyond the physical limits of the objectives and correcting for lens-to-lens aberration variations, plate warping, and liquid meniscus distortions—all in post-processing. The result is a fully automated, high-throughput imaging platform that delivers sub-micron resolution, label-free phase contrast, and co-registered fluorescence images from every well in a single acquisition cycle, transforming the paradigm of large-scale cellular imaging.

## DETAILED DESCRIPTION

- introduce purpose of description  

The following detailed description is provided to enable any person skilled in the art to make and use the invention, and it sets forth the best mode contemplated by the inventor for carrying out the claimed subject matter. This description includes specific embodiments, components, and operational procedures that illustrate the invention’s structure, function, and advantages, but it is not intended to limit the scope of the invention as defined by the claims. Modifications and variations may be made without departing from the spirit and scope of the invention, and all such modifications are intended to be encompassed within the claims. The invention’s novelty lies not merely in the physical arrangement of its components, but in the synergistic integration of computational imaging, parallel hardware architecture, and adaptive reconstruction algorithms that collectively overcome longstanding barriers in high-throughput cellular imaging. The description that follows systematically details each subsystem, its interrelationships, and the operational principles that enable the system’s unprecedented performance.

### I. Introduction

- introduce imaging systems and methods  
- motivate high resolution FP imaging  
- describe limitations of traditional imaging systems  
- introduce FP imaging technique  
- describe advantages of FP imaging  
- cite prior art  
- introduce imaging systems for FP processing  
- describe components of imaging systems  
- describe sample loading system  
- describe illumination system  
- describe optical system  
- describe imaging system  
- describe image acquisition phase  
- describe Fourier ptychographic reconstruction process  

Imaging systems designed for multi-well plate analysis have historically been constrained by the physical limitations of optical resolution, depth of field, and mechanical throughput. Conventional systems, whether based on widefield microscopy or confocal scanning, require either a trade-off between speed and resolution or the use of complex mechanical systems to sequentially focus on each well, resulting in acquisition times that are orders of magnitude longer than those of non-imaging plate readers. The Fourier ptychographic imaging technique offers a compelling alternative by circumventing these limitations through computational synthesis of high-resolution images from a series of low-resolution, differently illuminated acquisitions. Unlike traditional microscopy, where resolution is dictated by the numerical aperture of the objective, Fourier ptychography extends the effective numerical aperture by combining angularly diverse illumination patterns, thereby increasing the spatial frequency bandwidth captured by the sensor. This approach enables the recovery of both amplitude and phase information, allowing for label-free contrast enhancement, computational refocusing, and aberration correction—all without requiring physical adjustments to the optical train. Prior art has demonstrated the feasibility of Fourier ptychography in single-sample applications, but its adaptation to parallel, multi-well imaging has remained unaddressed due to the immense engineering challenges posed by system scalability, illumination uniformity, and data throughput. The present invention overcomes these barriers by integrating 96 independent Fourier ptychographic imaging channels into a single, co-registered platform, each channel acquiring a unique set of illumination data under precisely controlled conditions. The sample loading system is designed to accept standard 96-well plates with minimal alignment requirements, while the illumination system employs a densely packed LED array with a precisely calibrated spatial pitch to ensure that each well receives a distinct, non-overlapping illumination pattern during each acquisition step. The optical system, composed of custom-designed miniature objectives with a fixed conjugate distance, maintains consistent object-to-sensor spacing across all channels, eliminating the need for dynamic focusing. Each imaging channel is equipped with a CMOS sensor capable of capturing intensity data at high frame rates, and the entire system is synchronized by a central controller that triggers illumination sequences and records image data in parallel. During image acquisition, the illumination source is cycled through a predefined sequence of activation patterns, with each pattern illuminating a unique subset of wells while the corresponding sensors capture the transmitted light. The resulting dataset, comprising hundreds of low-resolution intensity images, is then processed using a massively parallelized Fourier ptychographic reconstruction algorithm that iteratively combines the spatial frequency content from each illumination angle, applies phase retrieval to recover the complex object function, and generates a final high-resolution image that is both aberration-corrected and computationally refocused. This process, performed entirely in software, enables the system to deliver images with resolution beyond the native optical limit, consistent across all wells, regardless of plate warping, meniscus curvature, or manufacturing variations in the objective lenses.

### II. Imaging System for Fourier Ptychographic (FP) Imaging and Fluorescent Imaging

- introduce imaging system 100  
- describe illumination system 102  
- describe sample loading system 104  
- describe optical system 106  
- describe image sensor system 108  
- describe controller 110  
- describe illumination patterns  
- describe image data output  
- describe processing of raw image data  
- describe FP image processing operations  
- describe generation of high resolution image  
- describe fluorescence imaging  
- describe processing of fluorescence image data  
- describe parallel image processing  
- describe processor and memory  
- describe communication interfaces  
- describe output of raw and processed image data  
- describe external computing device or system  
- describe external memory device or system  
- describe network communication interface  
- describe additional interfaces  
- describe multiplexing of image data  
- describe demultiplexing of image data  
- introduce imaging system 200  
- describe housing or enclosure 202  
- describe frame structure 204  
- describe alignment through-holes 205  
- describe frame alignment rods 206  
- describe physical support of components  
- describe substrates with through-holes  
- describe illumination system components  
- describe optical system components  
- describe image sensor system components  
- describe sample loading system components  
- describe sample platform 215  
- describe aperture slot 214  
- describe multi-well plate 208  
- describe wells 209  
- describe sample platform guides  
- describe automatic loading and ejecting mechanism  
- describe sample platform 305  
- describe illumination system  
- introduce light sources  
- describe LED matrix  
- explain RGB LED  
- discuss LED footprint  
- describe well arrangement  
- derive equation for light sources  
- calculate number of light sources  
- introduce side-mounted light sources  
- describe lens array  
- explain multi-lens-array arrangement  
- discuss lens characteristics  
- describe optical arrangement  
- introduce optical filter  
- explain fluorescence imaging  
- describe GFP imaging  
- discuss optical filter placement  
- explain bright field illumination  
- describe removable optical filter  
- introduce image sensor system  
- describe image sensor array  
- explain image sensor capabilities  
- discuss image sensor orientation  
- describe data transfer  
- introduce liquid cooling system  
- describe image sensor system components  
- explain image sensor system operation  
- summarize imaging system  

The imaging system 100 is a fully integrated, modular platform designed for simultaneous Fourier ptychographic and fluorescence imaging of 96-well plates. It comprises an illumination system 102, a sample loading system 104, an optical system 106, an image sensor system 108, and a central controller 110, all housed within a rigid, thermally stabilized enclosure 202. The illumination system 102 consists of a two-dimensional array of high-intensity light-emitting diodes arranged in a grid pattern with a center-to-center spacing of 3 mm, precisely aligned to match the well-to-well spacing of a standard 96-well plate. Each LED is capable of emitting light at a central wavelength of approximately 530 nm for bright-field illumination and is independently controllable to generate sequential illumination patterns that ensure each well is illuminated by only one LED at a time, thereby preventing cross-talk and maximizing ptychographic sampling efficiency. The total number of LEDs required is reduced from 4,704 to 1,120 through a shared illumination scheme in which each LED illuminates up to nine wells simultaneously, with the angular coverage of each LED carefully calibrated to match the numerical aperture of the objectives. The sample loading system 104 includes a precision-aligned sample platform 215 with a central aperture slot 214 that accepts a multi-well plate 208, ensuring that each well 209 is positioned directly beneath its corresponding imaging channel. The platform is guided by mechanical alignment rods 206 and through-holes 205 to maintain sub-micron positional accuracy, and an automated loading and ejecting mechanism enables seamless integration with robotic handling systems. The optical system 106 comprises 96 custom-designed, finite-conjugate objectives, each with a 4× magnification, a numerical aperture of 0.23, and a working distance of 4 mm, fabricated from injection-molded plastic to achieve cost-effective scalability. These objectives are mounted on a single substrate with precisely drilled through-holes that align each lens with its corresponding CMOS sensor, forming a 96-in-1 imaging array. The image sensor system 108 consists of 96 consumer-grade CMOS sensors, each with a pixel size of 0.4 μm, arranged in a 12×8 grid and bonded directly to a multi-layer printed circuit board that facilitates high-speed data transfer via parallel interfaces. The sensors are oriented such that their active areas are coplanar with the image plane of the objectives, and each sensor is equipped with a removable optical filter that can be switched between bright-field and fluorescence modes. For fluorescence imaging, a second excitation source, comprising side-mounted LEDs emitting at 465 nm, is activated in conjunction with a bandpass emission filter centered at 535 nm to isolate green fluorescent protein (GFP) signals. The controller 110 is a high-performance computing unit comprising a multi-core central processing unit, four graphics processing units (GPUs), and 128 GB of high-bandwidth memory, all connected via a high-speed internal bus. The controller orchestrates the sequential activation of LEDs, synchronizes image capture across all sensors, and executes a parallelized Fourier ptychographic reconstruction algorithm that processes the raw intensity data from all 96 channels simultaneously. Raw image data is acquired at a rate of 340 MB/s and stored temporarily in an onboard buffer before being transferred to an external memory device or cloud-based storage system via a gigabit Ethernet interface. The system supports multiplexing of image data streams from all sensors into a single data pipeline and demultiplexing during reconstruction to isolate individual well data. The entire system is enclosed in a rigid frame structure 204 that provides mechanical stability and thermal isolation, with a liquid cooling system circulating coolant through microchannels adjacent to the CMOS sensors to minimize thermal noise during prolonged acquisitions. The system operates in two distinct modes: bright-field Fourier ptychography for label-free phase imaging and fluorescence imaging for molecular detection, with the optical filter automatically switched between modes to enable co-registered, multi-modal imaging without repositioning the sample. The combination of parallel hardware architecture, computational image reconstruction, and adaptive illumination control enables the system to generate high-resolution, aberration-corrected images of all 96 wells in under 90 seconds for bright-field imaging and under 30 seconds for fluorescence imaging, achieving a throughput rate that exceeds that of conventional scanning systems by more than an order of magnitude.

### III. Variable-Illumination Fourier Ptychographic Imaging Methods

- describe FP image acquisition process  
- introduce illumination system and image sensor system  
- explain initialization of illumination system and image sensor system  
- describe calibration operation  
- introduce sth scan  
- describe illumination pattern during sth scan  
- show example arrangement of light sources and wells  
- describe reception and focusing of light  
- describe image data acquisition  
- describe storage of image data  
- explain multiplexing approach  
- describe separation of intensity data  
- determine whether all n scans have been completed  
- incrementally update s for next scan  
- perform parallel reconstruction process  
- describe iterative combination of intensity images  
- apply filter in Fourier domain  
- apply inverse Fourier transform  
- replace intensity with measurement  
- apply Fourier transform  
- update region in Fourier space  
- describe phase retrieval technique  
- describe recovery process  
- introduce FP reconstruction process 700  
- initialize high-resolution image solution  
- apply Fourier transform  
- describe low-pass filtering  
- generate low-resolution image  
- propagate low-resolution image to in-focus plane  
- replace amplitude component with measurement  
- back-propagate to sample plane  
- update high-resolution solution  
- determine whether operations have been completed for all images  
- repeat operations for next image  
- determine whether high-resolution solution has converged  
- repeat operations until convergence  
- transform converged solution to spatial domain  
- introduce FP reconstruction process 800  
- model connection between sample profile and captured intensity data  
- invert connection to achieve aberration-free reconstructed image  
- describe digital wavefront correction  
- introduce variable-illumination Fourier ptychographic imaging methods  
- initialize high-resolution image solution  
- apply Fourier transform to obtain initialized Fourier transformed image  
- determine initial high-resolution solution  
- multiply by phase factor in Fourier domain  
- perform low-pass filtering of high-resolution image  
- generate low-resolution image for particular plane wave incidence angle  
- filter low-pass region from spectrum of high-resolution image  
- replace computed amplitude component with square root of low-resolution intensity measurement  
- multiply by inverse phase factor in Fourier domain  
- apply Fourier transform to updated target image  
- update high-resolution solution in Fourier space  
- determine whether operations have been completed for all uniquely illuminated low-resolution intensity images  
- repeat operations for next image  
- determine whether high-resolution solution has converged  
- compare previous high-resolution solution to present high-resolution solution  
- repeat operations until solution converges  
- transform converged solution to spatial domain to recover high-resolution image  
- describe calibration process for determining angles of incidence  
- illuminate central light element  
- capture vignette monochromic image  
- determine center of image  
- measure shift of center of image  
- determine displacement of central light element using lookup table  
- determine precise values of illumination angles  
- describe fluorescence imaging process  
- load multi-well plate into imaging system  
- initialize illumination system and image sensor system  
- illuminate multi-well plate with excitation light  
- receive and focus light emitted by samples  
- filter light to only allow light emitted by fluorophore  
- acquire fluorescence image data  
- store image data in memory  
- generate combined fluorescence and high-resolution bright-field image  

The variable-illumination Fourier ptychographic imaging method employed by the system is a computationally intensive, iterative process that reconstructs a high-resolution complex image from a sequence of low-resolution intensity measurements acquired under varying illumination angles. The process begins with the initialization of the illumination system and image sensor system, followed by a calibration procedure in which a single central LED is activated and a monochromatic vignette image is captured to determine the precise angular position of each LED relative to the sample plane. This calibration accounts for the refractive effects of the liquid medium within each well and the parallax-induced displacement of the apparent LED position, using a ray-tracing algorithm based on Snell’s law to compute the true angle of incidence for each illumination source. Once calibrated, the system initiates a series of n illumination scans, where in each scan s, a unique subset of LEDs is activated such that each well is illuminated by only one LED at a time, and the corresponding image sensors capture the transmitted intensity pattern. The total number of scans required is determined by the numerical aperture of the objectives and the desired resolution enhancement, with a minimum of 72 distinct illumination patterns being used to fully sample the spatial frequency domain. During each scan, the image data from all 96 sensors is acquired simultaneously and stored in a shared memory buffer, with each image tagged with its corresponding illumination index. The raw data is then multiplexed into a single data stream for efficient transfer to the reconstruction processor. The Fourier ptychographic reconstruction process, designated as Process 700, initializes a high-resolution complex image estimate in the spatial domain and applies a Fourier transform to convert it into the frequency domain. A low-pass filter is applied to extract the band-limited component corresponding to the current illumination angle, and the resulting low-resolution image is propagated to the in-focus plane using a Fresnel diffraction kernel. The amplitude of this propagated image is then replaced with the square root of the measured intensity from the corresponding sensor, while the phase is retained from the current estimate. The modified image is back-propagated to the sample plane, and the high-resolution solution is updated by combining the new frequency components with the existing estimate. This iterative procedure is repeated for each of the n illumination patterns, and convergence is determined by comparing the current solution to the previous iteration using a root-mean-square error metric. Once convergence is achieved, the final high-resolution complex image is transformed back into the spatial domain to yield a high-resolution amplitude image and a phase image, both corrected for aberrations and defocus. An alternative reconstruction process, designated as Process 800, models the physical relationship between the sample’s complex transmission function and the captured intensity data as a nonlinear inverse problem, and solves it using a gradient descent algorithm that simultaneously estimates the aberration coefficients of each imaging channel. This method enables digital wavefront correction by recovering the pupil function of each objective independently, even in the presence of significant manufacturing variations. The fluorescence imaging process is executed as a separate, concurrent acquisition cycle, wherein the excitation LEDs at 465 nm are activated, and the emission light from GFP-labeled cells is filtered through a 535 nm bandpass filter before being captured by the same sensors. The fluorescence image is acquired in a single exposure per well, and the resulting intensity data is co-registered with the corresponding Fourier ptychographic phase image using a rigid-body transformation derived from fiducial markers embedded in the imaging system. The final output is a composite image that overlays the high-resolution phase contrast morphology of the cells with the spatial distribution of GFP expression, enabling simultaneous phenotypic and molecular analysis without the need for computational alignment or registration software. The entire imaging and reconstruction pipeline, from sample loading to final image output, is fully automated and requires no user intervention, enabling true high-throughput, multi-modal imaging at a scale previously unattainable in biological research.