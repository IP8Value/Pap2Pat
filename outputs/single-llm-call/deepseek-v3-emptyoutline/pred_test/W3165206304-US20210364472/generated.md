Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

### 1. Field of the Invention  

The present invention relates generally to non-destructive testing methods for composite materials, and more particularly to an improved ultrasonic testing system and methodology for detecting and accurately sizing foreign object defects within carbon fiber reinforced polymer (CFRP) laminates. The invention provides enhanced signal processing techniques that enable precise edge detection and dimensional quantification of manufacturing defects with unprecedented accuracy compared to existing non-destructive evaluation methods.  

### 2. Description of Related Art  

Composite materials, particularly carbon fiber reinforced polymers, have become increasingly prevalent in aerospace, automotive, and other high-performance applications due to their exceptional strength-to-weight ratios and anisotropic properties. However, these materials are susceptible to various manufacturing defects including delaminations, broken fibers, inclusions, and foreign object debris (FOD). Such defects can significantly reduce the service life of components by creating stress concentration points within the composite structure.  

Current non-destructive testing (NDT) methods for composite inspection include shearography, thermography, X-ray computed tomography, acoustography, terahertz imaging, and ultrasound. Among these, ultrasonic testing has emerged as the most widely adopted technique due to its safety, portability, relatively low cost, and ease of use. Conventional ultrasonic methods for defect detection typically rely on amplitude-based threshold techniques such as the 6 dB drop method, where the defect boundary is defined as the point where signal amplitude decreases to 50% of its maximum value over the defect.  

Prior attempts to improve defect detection accuracy have employed various signal processing approaches. Benammar et al. demonstrated depth detection of delaminations within 4-8% accuracy using specialized signal processing. Poudel et al. utilized fuzzy logic and artificial neural networks to identify Teflon inclusions, though without providing sizing accuracy data. Hasiotis et al. successfully traced defect shapes but significantly overestimated sizes in CFRP laminates.  

More advanced techniques have been proposed, including wavelet transform algorithms and signal correlation methods. Mohammadkhani et al. reported a 43.8% overestimation in sizing Teflon inclusions, while Ma et al. demonstrated errors ranging from 0.175 mm to 0.475 mm in defect dimension measurements. These approaches, while representing improvements over conventional methods, still exhibit significant limitations in sizing accuracy, particularly for small defects and in woven composite systems where structural noise presents additional challenges.  

The present invention addresses these limitations through novel signal processing methodologies that combine advanced filtering techniques with gradient-based edge detection, enabling significantly improved sizing accuracy for foreign object defects in composite materials.  

## SUMMARY OF THE INVENTION  

The present invention provides an advanced ultrasonic inspection system and methodology for detecting and accurately sizing foreign object defects in composite materials, particularly carbon fiber reinforced polymer laminates. The system employs a custom ultrasonic immersion scanning apparatus utilizing high-frequency (7.5 MHz) spherically focused transducers in pulse-echo configuration, requiring only single-sided access to the test specimen.  

Key innovations of the invention include:  

1. A specialized signal preprocessing approach that incorporates front-wall echo alignment through polynomial surface fitting and time-shifting of ultrasonic waveforms to compensate for surface variations and leveling inconsistencies.  

2. An enhanced maximum gated amplitude (MGA) method for c-scan image generation that improves defect visualization by operating on raw signal data rather than absolute values.  

3. A two-dimensional Gaussian filtering technique applied to ultrasonic scan data to reduce noise and improve boundary resolution, with optimized parameters for defect edge enhancement.  

4. A Fourier transform-based interpolation method that increases effective scan resolution while minimizing computational overhead.  

5. A novel Maximum Gradient Transition (MGT) edge detection algorithm that calculates the magnitude of the gradient across interpolated scan data to precisely identify defect boundaries based on maximum signal intensity transitions.  

The combined implementation of these techniques enables unprecedented accuracy in foreign object defect sizing, with experimental results demonstrating an average error of only 0.11 mm (0.004 in) in diameter measurement across a range of defect sizes from 1.59 mm to 12.7 mm. This represents a threefold improvement over previously published methods and maintains consistent accuracy across various depths within the laminate structure.  

The invention finds particular utility in quality control during composite manufacturing and in-service inspection of critical aerospace components, where accurate quantification of manufacturing defects is essential for structural integrity assessment and remaining life prediction.  

## DETAILED DESCRIPTION  

The present invention provides a comprehensive methodology for ultrasonic inspection of composite materials with enhanced defect detection and sizing capabilities. The system architecture comprises several integrated components that work in concert to achieve superior measurement accuracy.  

The ultrasonic scanning apparatus consists of a custom immersion tank system with precision translation stages capable of 0.005 mm/step resolution. A high-frequency (7.5 MHz) spherically focused transducer with 38.1 mm focal length is employed, driven by a high-voltage pulser/receiver system operating at 200 V with 65 ns pulse width. Data acquisition occurs at 160 MHz sampling rate to ensure adequate temporal resolution of ultrasonic waveforms.  

Signal preprocessing begins with front-wall echo detection and alignment. For each scan location (x₁,x₂), the system identifies the first signal excursion above a predetermined threshold and subsequently locates the first peak as the definitive front-wall echo reference. A third-order polynomial surface is then fitted to these reference points across the scan area, and all waveforms are time-shifted such that the front-wall echo occurs at a consistent reference time t₀. This alignment compensates for minor surface variations and ensures proper registration of internal features.  

Following alignment, the system implements an enhanced Maximum Gated Amplitude (MGA) method to generate c-scan images. The gate position is determined based on the known laminate layup sequence, centered on the interface where foreign objects are located. Unlike conventional MGA approaches that use absolute signal values, the present invention operates on raw signal data, preserving important phase information that contributes to defect boundary definition. The gate width is set to three-quarters of the peak-to-peak distance between plies to optimize defect signal capture while minimizing noise inclusion.  

To further improve image quality, the invention applies a two-dimensional Gaussian filter to the MGA-processed data. The filter parameters are carefully selected with standard deviations σ₁ = σ₂ = 5/3 of the scan step size, providing optimal noise reduction without excessive blurring of defect edges. The Gaussian kernel is truncated at ±5 steps from the center, maintaining computational efficiency while preserving >99.5% of the filter's integral value.  

Following Gaussian filtering, the system implements a Fourier transform-based interpolation scheme to increase effective spatial resolution. The data is upsampled fivefold in both x₁ and x₂ dimensions using sequential one-dimensional Fourier interpolation along each axis. This approach provides superior accuracy compared to conventional interpolation methods, particularly for spatial frequencies below the Nyquist limit of the original scan.  

The core innovation of the invention lies in the Maximum Gradient Transition (MGT) edge detection algorithm. The system calculates the magnitude of the gradient G(x₁,x₂) across the interpolated c-scan image using fourth-order accurate central difference approximations for partial derivatives. The gradient magnitude highlights regions of rapid signal intensity change, corresponding to defect boundaries.  

To precisely locate the defect edge, the MGT algorithm begins with manual selection of an interior point within the defect region. From this seed point, the system projects outward along the direction of maximum gradient increase, identifying the peak gradient magnitude as the definitive edge location. This process is repeated at multiple angular positions around the defect to construct a complete boundary profile. For regions where the gradient peak is broad (due to finite sampling resolution), the centroid of the peak region is selected as the edge location.  

The resulting boundary points are processed to calculate defect area and effective diameter. Experimental validation demonstrates the system's exceptional accuracy, with average diameter measurement errors of just 0.11 mm across all defect sizes tested. The methodology shows particular effectiveness for small defects (1.59 mm diameter), where conventional techniques typically exhibit proportionally greater errors.  

Depth compensation is implemented through adaptive gain adjustment based on defect location within the laminate. Defects near the scan surface (3rd-4th ply interface) receive reduced gain to prevent signal saturation, while deeper defects (9th-10th ply interface) maintain standard gain settings. This optimization ensures consistent measurement accuracy regardless of defect depth.  

The complete system provides a robust solution for foreign object defect characterization in composite materials, offering significant improvements over existing ultrasonic inspection methods. The invention's combination of advanced signal processing techniques enables reliable detection and accurate sizing of defects as small as 1.59 mm diameter, with potential applications in both manufacturing quality control and in-service inspection regimes.