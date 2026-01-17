# DESCRIPTION

## PRIORITY DATA

This application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Application No. 63/XXXXXXX, filed on [Date], which is hereby incorporated by reference in its entirety.

## FEDERALLY SPONSORED RESEARCH OR DEVELOPMENT

This invention was made with government support under Grant No. [Grant Number] awarded by [Agency Name]. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates to high-throughput imaging systems, and more specifically, to a parallel imaging system for Fourier ptychographic (FP) imaging and fluorescent imaging of multi-well plates.

## BACKGROUND

Multi-well plate readers are widely used in large format cell culture experiments for performing fluorescence or absorbance measurements. However, these readers provide only gross characterization of the samples, lacking the detailed morphological and functional information that can be obtained through microscopy. To address this limitation, imaging well plate microscopy systems have been developed, allowing for the collection of detailed microscopy images of cell cultures. These systems, however, suffer from a significant throughput bottleneck compared to non-imaging well plate readers due to the finite data throughput rate of a single scanning microscope column.

Parallel imaging approaches have been proposed to overcome this throughput gap, but they face several engineering challenges, including ensuring simultaneous focus across all wells, managing the physical constraints of the objectives, and handling variations in well curvature and well-to-lens distance. Conventional microscopy techniques struggle to address these challenges effectively, leading to suboptimal image quality and reliability.

Fourier ptychographic microscopy (FPM) offers a promising solution to these challenges. FPM combines multiple low-resolution images taken under different illumination angles to computationally reconstruct a high-resolution image. This method can correct for various aberrations, including defocus and lens-to-lens variations, and extend the effective depth of field of the imaging system. Despite its potential, the application of FPM to high-throughput imaging of multi-well plates has not been fully realized.

## SUMMARY

The present invention provides a high-throughput imaging system for Fourier ptychographic (FP) imaging and fluorescent imaging of multi-well plates. The system includes a 96-in-1 parallel imaging module with 96 repeating units of compact microscopes, each consisting of a custom-designed objective and a CMOS sensor. The microscopes are arranged to simultaneously image all wells of a 96-well plate, significantly improving the imaging throughput.

The system utilizes a Fourier ptychographic microscopy (FPM) method to overcome the challenges of defocus, lens aberrations, and well-to-lens distance variations. The FPM method involves collecting a sequence of transmission images of the sample under varying illumination angles and computationally stitching the images to reconstruct a high-resolution, aberration-free image. The system also supports fluorescence imaging at the native resolution of the objectives.

Key features of the invention include:
- **Parallel Imaging Module**: A 96-in-1 sensor board incorporating 96 individual sets of CMOS sensors and microscope objectives, each aligned to a corresponding well of the 96-well plate.
- **Custom-Designed Objectives**: Microscope objectives with a 4× magnification, a working distance of 4 mm, and a numerical aperture (NA) of 0.23, designed to provide a balance between imaging field-of-view and resolution.
- **FPM Method**: A computational method for reconstructing high-resolution images from multiple low-resolution images taken under different illumination angles, correcting for various aberrations and extending the effective depth of field.
- **Fluorescence Imaging**: Support for fluorescence imaging at the native resolution of the objectives, with automatic co-registration and co-alignment of FPM and fluorescence images.
- **Data Processing Pipeline**: A GPU-accelerated data processing pipeline for efficient image reconstruction and analysis.

The invention addresses the throughput gap between well plate readers and imaging systems, providing high-resolution, aberration-free images of cell cultures in a high-throughput manner. This system is particularly useful for applications in biomedical and pharmaceutical research, where rapid and detailed imaging of large numbers of samples is essential.

## DETAILED DESCRIPTION

### I. Introduction

The present invention relates to a high-throughput imaging system for Fourier ptychographic (FP) imaging and fluorescent imaging of multi-well plates. The system, referred to as the 96 Eyes, is designed to overcome the throughput limitations of conventional imaging well plate microscopy systems by utilizing a parallel imaging approach. The 96 Eyes system employs 96 repeating units of compact microscopes, each consisting of a custom-designed objective and a CMOS sensor, to simultaneously image all wells of a 96-well plate. The system leverages the Fourier ptychographic microscopy (FPM) method to correct for various aberrations and extend the effective depth of field, ensuring high-resolution, aberration-free images.

### II. Imaging System for Fourier Ptychographic (FP) Imaging and Fluorescent Imaging

The 96 Eyes system is designed to provide high-throughput imaging of multi-well plates by addressing the key challenges of defocus, lens aberrations, and well-to-lens distance variations. The system includes the following components:

#### 1. Parallel Imaging Module
The parallel imaging module consists of a 96-in-1 sensor board incorporating 96 individual sets of CMOS sensors and microscope objectives. Each set is aligned to a corresponding well of the 96-well plate, allowing for simultaneous imaging of all wells. The sensor board is interfaced with four frame grabber boards, which connect to a workstation for data processing.

#### 2. Custom-Designed Objectives
The microscope objectives are custom-designed to provide a 4× magnification, a working distance of 4 mm, and a numerical aperture (NA) of 0.23. The objectives are designed with a finite conjugate optical configuration, resulting in a fixed object-to-image distance of 48 mm. The use of plastic-molded lenses allows for cost-effective production, although it introduces variations in lens-to-lens aberrations. The FPM method is used to correct these aberrations during image reconstruction.

#### 3. FPM Method
The FPM method involves collecting a sequence of transmission images of the sample under varying illumination angles. The illumination is provided by an LED matrix at the top of the system, with the light transmitting through the target 96-well plate. The transmission through each well is collected by the corresponding objective and projected onto a camera sensor chip. The collected images are computationally stitched in the spatial frequency domain using the Fourier ptychographic phase retrieval algorithm to reconstruct a high-resolution image. The FPM method can correct for various aberrations, including defocus, astigmatism, and spherical aberration, and extend the effective depth of field of the imaging system.

#### 4. Fluorescence Imaging
The system supports fluorescence imaging at the native resolution of the objectives. A pair of liquid-guided excitation sources projects homogenized light beams from both sides of the culture plate. The direct transmitted light is blocked by the internal aperture of the microscope objectives, and the residual scattered light is further attenuated by an emission filter. The plates are scanned with a z-axis piezo flexure scanning stage to capture a z-stack of fluorescence images, and the sharpest image is selected from the z-stack.

### III. Variable-Illumination Fourier Ptychographic Imaging Methods

The 96 Eyes system utilizes a variable-illumination Fourier ptychographic imaging method to overcome the challenges of defocus, lens aberrations, and well-to-lens distance variations. The method involves the following steps:

#### 1. Illumination Scheme
The FPM illumination is provided by an LED matrix at the top of the system. The light transmits through the target 96-well plate, and the transmission through each well is collected by the corresponding objective and projected onto a camera sensor chip. The 96 objectives and camera chips are housed on a customized 96-in-1 sensor board. The camera sensor board is interfaced with four frame grabber boards, which connect to the workstation. A piezo-electric z-axis stage is used to hold the well plate in place and provide z-axis translation as needed.

#### 2. Data Acquisition
The data acquisition process involves illuminating the object with a sequence of spatially distributed light sources. The illumination system is shared among multiple imaging sensors, each of which captures individual specimens independently. The system utilizes a high-density layout of the mini-microscopes, where a single LED can illuminate up to nine wells at the same time. The working wavelength is chosen to match the passband of the fluorescence emission filter. The image acquisition and data transfer are performed in a massively parallel manner, minimizing camera idling time.

#### 3. Image Reconstruction
The collected images are computationally stitched in the spatial frequency domain using the Fourier ptychographic phase retrieval algorithm. The algorithm corrects for various aberrations, including defocus, astigmatism, and spherical aberration, and extends the effective depth of field of the imaging system. The reconstructed images provide high-resolution, aberration-free views of the cell cultures.

#### 4. Aberration Correction
The FPM method is particularly effective at characterizing and correcting the aberrations of the optical system on the fly. The system can correct up to 30 coefficients of the Zernike polynomial decomposition, significantly diminishing the impact of lens-to-lens variations and well-to-lens distance variations. The FPM method also allows for computational refocusing and aberration correction post data acquisition, providing a marked benefit over conventional microscopy systems.

#### 5. Extended Depth of Field
The FPM method can extend the effective depth of field of the imaging system beyond the limits dictated by the objective. This is particularly useful for addressing the well-to-lens distance variations and the natural geometrical warping of the cell culture plates. The system can tolerate an extended depth variation of ±50 μm, covering around 90% of all UV-Star plates.

#### 6. Fluorescence Image Processing
For fluorescence imaging, the plates are scanned with a z-axis piezo flexure scanning stage to capture a z-stack of fluorescence images. The sharpest image is selected from the z-stack, and the two image channels (FPM and fluorescence) are automatically co-registered and co-aligned. The GPU-accelerated data processing pipeline ensures efficient image reconstruction and analysis.

The 96 Eyes system represents a significant advancement in high-throughput imaging of multi-well plates, providing high-resolution, aberration-free images of cell cultures in a rapid and reliable manner. This system is particularly useful for applications in biomedical and pharmaceutical research, where detailed and rapid imaging of large numbers of samples is essential.