# DESCRIPTION  

## BACKGROUND  

The present invention relates to an information processing system for visual motion detection, particularly a system and method that leverages phase information in images to detect motion efficiently and robustly. Visual motion detection is a critical function for both biological and artificial vision systems, enabling tasks such as navigation, object tracking, and scene understanding. In biological systems, motion detection is performed rapidly and in parallel, often beginning in early stages of visual processing. For example, in vertebrates, direction-selective ganglion cells in the retina detect motion within just a few synaptic steps, while in insects, motion-sensitive neurons are found only three synapses away from photoreceptors. These biological systems achieve high efficiency by processing motion in continuous time and in analog domains before converting signals to discrete spikes.  

Existing computer-based motion detection techniques, such as optical flow algorithms, often rely on estimating spatial changes between consecutive image frames. While these methods can produce accurate results, they are computationally intensive and may not be suitable for real-time applications. Biological models of motion detection, such as the Reichardt detector, motion energy detector, and Barlow-Levick model, offer simpler architectures but still face limitations in computational efficiency and robustness under varying conditions.  

A key limitation of conventional motion detection methods is their reliance on intensity-based processing, which can be sensitive to noise, illumination changes, and low-contrast conditions. Phase-based approaches, however, offer inherent advantages, as phase information is less affected by intensity variations and can encode structural features such as edges and textures more robustly. Prior research has demonstrated that images can be faithfully reconstructed using only their global phase, suggesting that phase information alone carries sufficient detail for visual processing tasks.  

The present invention addresses these limitations by introducing a novel motion detection algorithm based on local phase information. This approach mimics the efficiency of biological motion detection by operating in continuous time and leveraging parallel processing. Unlike traditional methods that focus on velocity estimation, the disclosed system emphasizes motion localization and direction detection, making it particularly suitable for real-time applications in surveillance, robotics, and autonomous systems.  

## SUMMARY  

The invention provides a method and system for detecting motion in visual scenes using local phase information derived from images or video sequences. The method involves computing the Short-Time Fourier Transform (STFT) of the visual input using overlapping window functions, extracting local phase components, and analyzing temporal changes in these phases to detect motion.  

The system includes a processor configured to divide the visual input into overlapping blocks, apply a window function (e.g., a Gaussian window) to each block, and compute the local phase via the STFT. A temporal high-pass filter is then applied to the phase components to isolate changes indicative of motion. The Radon transform is subsequently used to analyze the structure of these phase changes, enabling robust motion detection even in noisy or low-contrast conditions.  

A key innovation of this method is its reliance on local phase rather than intensity or amplitude information. This allows the system to detect motion invariantly under varying lighting conditions and contrast levels. Additionally, the algorithm is highly parallelizable, making it suitable for implementation on digital signal processors (DSPs), graphics processing units (GPUs), or field-programmable gate arrays (FPGAs).  

The invention also encompasses a computer-readable medium storing instructions that, when executed by a processor, perform the disclosed motion detection method. This medium may be integrated into embedded systems, cameras, or autonomous vehicles to enable real-time motion analysis.  

## DETAILED DESCRIPTION  

The invention leverages phase information in images to detect motion efficiently. Phase, as opposed to amplitude, encodes structural features of visual scenes, such as edges and textures, and is less sensitive to variations in illumination or contrast. The method involves analyzing both global and local phase components to determine motion characteristics.  

### Global Phase of Images  

The global phase of an image is derived from its Fourier transform. Given a real-valued image \( u(x, y) \), its Fourier transform \( \hat{U}(\omega_x, \omega_y) \) can be expressed in polar coordinates as \( \hat{A}(\omega_x, \omega_y) e^{j\hat{\phi}(\omega_x, \omega_y)} \), where \( \hat{A} \) is the global amplitude and \( \hat{\phi} \) is the global phase. The global phase represents the offset of sinusoids of different frequencies across the entire image and plays a crucial role in image representation.  

### Local Phase of Images  

Local phase is computed using the Short-Time Fourier Transform (STFT), which applies a window function to restrict the Fourier analysis to a specific region of the image. For a window function \( w(x, y) \) centered at \( (x_0, y_0) \), the STFT is given by:  

\[ U(\omega_x, \omega_y, x_0, y_0) = \int_{\mathbb{R}^2} u(x, y) w(x - x_0, y - y_0) e^{-j(\omega_x (x - x_0) + \omega_y (y - y_0))} dx \, dy. \]  

The local phase \( \phi(\omega_x, \omega_y, x_0, y_0) \) is extracted from the polar representation of the STFT, \( A(\omega_x, \omega_y, x_0, y_0) e^{j\phi(\omega_x, \omega_y, x_0, y_0)} \). This phase component captures local structural features, such as edges and textures, within the windowed region.  

When the window function is Gaussian, the STFT can be interpreted as the response of Gabor receptive fields, which are commonly used to model biological vision systems. The local phase thus provides a biologically plausible representation of visual information.  

### Reconstruction of Images from Local Phase  

A significant aspect of the invention is the ability to reconstruct images using only local phase information, demonstrating the sufficiency of phase for visual representation. By solving a set of linear equations derived from phase measurements, the image can be reconstructed up to a constant scale. This reconstruction algorithm involves:  

1. Expressing the image as a sum of basis functions in a Reproducing Kernel Hilbert Space.  
2. Formulating orthogonality conditions based on local phase measurements.  
3. Solving for the coefficients of the basis functions to reconstruct the image.  

This process confirms that local phase alone contains sufficient information for visual processing tasks, including motion detection.  

### The Global Phase Equation for Translational Motion  

For a translating visual stimulus \( u(x, y, t) = u(x - s_x(t), y - s_y(t), 0) \), the derivative of the global phase with respect to time is given by:  

\[ \frac{d\hat{\phi}(\omega_x, \omega_y, t)}{dt} = -\omega_x v_x(t) - \omega_y v_y(t), \]  

where \( v_x(t) \) and \( v_y(t) \) are the instantaneous velocities. This equation establishes a direct relationship between global phase changes and motion.  

### The Local Phase Equation for Translational Motion  

Similarly, the local phase change for a translating stimulus within a windowed region is:  

\[ \frac{d\phi_{00}}{dt}(\omega_x, \omega_y, t) = -v_x(t) \omega_x - v_y(t) \omega_y + \mathfrak{v}_{00}(\omega_x, \omega_y, t), \]  

where \( \mathfrak{v}_{00} \) is a term accounting for windowing effects. The dominant terms \( -v_x(t) \omega_x - v_y(t) \omega_y \) indicate that local phase changes are primarily driven by motion.  

### The Block Structure for Computing the Local Phase  

To implement the method efficiently, the visual input is divided into overlapping blocks, each processed independently. Gaussian windows are applied to each block, and the local phase is computed using the 2D Fast Fourier Transform (FFT). The temporal derivative of the phase is then obtained via high-pass filtering.  

Key steps include:  
1. Defining Gaussian windows centered at regular intervals.  
2. Computing the 2D FFT of each windowed block to extract local phase.  
3. Applying a temporal high-pass filter to isolate phase changes due to motion.  
4. Denoising phase measurements by weighting them according to local amplitude.  

This block-based approach enables parallel processing, making the algorithm suitable for real-time implementation.  

### The Phase-Based Detector  

The phase-based motion detector employs the Radon transform to analyze the structure of phase changes across frequency components. The Radon transform computes line integrals of the phase derivative, revealing linear patterns indicative of motion. The Phase Motion Indicator (PMI) is then derived as:  

\[ \text{PMI}_{kl} = \max_{\theta \in [0, \pi)} \sum_\rho \left| \frac{(\mathcal{R} \frac{d\phi_{kl}}{dt})(\rho, \theta, t)}{\mathfrak{c}(\rho, \theta)} \right|, \]  

where \( \mathfrak{c}(\rho, \theta) \) is a correction term. A high PMI value indicates motion in the corresponding block.  

The direction of motion is computed from the angle \( \theta \) that maximizes the PMI, providing a coarse estimate of motion orientation.  

### Examples of Phase-Based Motion Detection  

The algorithm has been tested on various video sequences, including highway surveillance, train station footage, thermal imaging, and low-light conditions. Results demonstrate robust motion detection even in low-contrast or noisy environments. For instance:  
- In highway surveillance videos, the algorithm successfully detected moving vehicles and distinguished them from stationary objects.  
- In thermal videos, motion was accurately identified despite significant background noise.  
- In low-light conditions, the algorithm maintained performance due to the phase-based approach's invariance to intensity scaling.  

Comparative studies with Reichardt and Barlow-Levick detectors showed superior performance of the phase-based method, particularly in low-contrast scenarios.  

### Applications to Motion Segmentation  

The detected motion signals can be used for segmenting moving objects from the background. By applying adaptive thresholding to the PMI values, moving objects are isolated without requiring additional cues such as color or texture. This approach has been compared to optic flow-based segmentation, with the phase-based method demonstrating comparable or better performance in various test cases.  

In summary, the disclosed invention provides a robust, efficient, and biologically inspired method for motion detection using local phase information. Its parallelizable structure and invariance to lighting conditions make it particularly suitable for real-time applications in computer vision and autonomous systems.