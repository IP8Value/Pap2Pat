# DESCRIPTION

## BACKGROUND

An information processing system is a structured framework designed to receive, transform, and output data in a manner that supports decision-making, analysis, or control. Such systems operate across multiple levels of abstraction, ranging from the algorithmic level—where procedural logic defines how information is processed—to the physical circuit level, where the algorithm is realized through hardware such as biological neural circuits or silicon-based digital signal processors. The distinction between these levels is critical for designing efficient, scalable, and biologically plausible computational models, particularly in the domain of visual perception. In this context, motion detection represents a fundamental task that underpins navigation, object recognition, and environmental interaction in both biological organisms and artificial systems.

Visual motion detection is essential for survival across species, enabling rapid responses to dynamic stimuli in the environment. Biological visual systems have evolved highly efficient neural architectures to detect motion with minimal latency and high reliability. In vertebrates, direction-selective ganglion cells (DSGCs) in the retina initiate motion signaling within just a few synaptic layers from photoreceptors. Similarly, in insects such as flies, direction-selective neurons appear in the optic lobe after only three synapses from the photoreceptor layer. This minimal synaptic depth suggests that early motion detection relies on relatively simple, yet highly effective, analog computations that occur before the onset of spiking activity. These biological circuits operate in parallel across the visual field, enabling concurrent processing of motion signals at multiple spatial locations—a feature that contributes significantly to their speed and robustness.

In contrast, computer-based motion detection algorithms, while often accurate, frequently suffer from high computational complexity that impedes real-time performance. Traditional approaches such as optic flow estimation compute dense velocity fields by solving partial differential equations or minimizing energy functions across image sequences. These methods, though powerful, demand substantial memory and processing resources, making them unsuitable for embedded systems, real-time robotics, or low-power applications. Furthermore, many such algorithms are sensitive to changes in illumination, contrast, and noise, which degrade performance in real-world conditions.

Several biologically inspired models have attempted to bridge this gap. The Reichardt detector, originally proposed to explain motion perception in insects, uses a correlation-based mechanism involving delayed and non-delayed signals from adjacent photoreceptors followed by a multiplication operation. The motion energy model, developed for mammalian vision, employs spatiotemporal filters and a squaring nonlinearity to compute motion energy, and has been shown to be mathematically equivalent to the Reichardt model under certain conditions. The Barlow-Levick model, based on retinal physiology in rabbits, uses asymmetric inhibition to suppress responses to motion in the null direction. While these models capture key aspects of biological motion detection, they do not fully exploit the representational power of phase information in visual signals.

Existing biological models, though insightful, largely ignore the role of phase in motion encoding. Phase, as a component of complex signal representations, has historically been underutilized in linear signal processing due to its nonlinear nature. However, recent advances in computational neuroscience and signal processing have demonstrated that phase carries rich structural information about visual scenes, including edge localization, texture, and motion. This insight motivates a reevaluation of motion detection strategies that leverage phase dynamics rather than intensity gradients alone.

## SUMMARY

The present invention provides a method for detecting visual motion in a video sequence by analyzing changes in the local phase of image signals. The method comprises receiving a time-varying visual signal, partitioning the signal into overlapping spatial blocks, applying a windowed Fourier transform to each block to obtain local phase information, computing the temporal derivative of the local phase, and applying a Radon transform to the phase derivative to generate a Phase Motion Indicator (PMI). Motion is detected in a block when the PMI exceeds a predetermined threshold, and the direction of motion is determined from the orientation that maximizes the Radon transform response.

In another aspect, the invention provides a system for motion detection comprising a processor configured to execute the aforementioned method. The system includes a memory storing instructions for dividing the visual field into blocks, applying Gaussian windows, performing two-dimensional Fast Fourier Transforms (2D FFTs), extracting phase components, applying temporal high-pass filtering, computing Radon transforms, and evaluating PMI values. The system is optimized for parallel execution, enabling real-time processing of high-resolution video streams.

In yet another aspect, the invention provides a non-transitory computer-readable medium encoded with instructions that, when executed by a processor, cause the processor to perform the steps of the motion detection method. The medium may be implemented in various forms, including solid-state memory, optical discs, or cloud-based storage, and is compatible with general-purpose computing platforms as well as specialized hardware such as Field-Programmable Gate Arrays (FPGAs) or Graphics Processing Units (GPUs).

## DETAILED DESCRIPTION

### Introduce phase information in images

Phase information in images refers to the angular component of the complex representation of a signal in the frequency domain. When an image is transformed via a complex-valued operator such as the Fourier transform or Short-Time Fourier Transform (STFT), the resulting coefficients can be expressed in polar form as amplitude and phase. While amplitude encodes the strength of frequency components, phase encodes their spatial alignment and relative timing. It has been established that phase, even in the absence of amplitude, can preserve much of the structural content of an image, including edges, textures, and object boundaries. This property makes phase a powerful descriptor for visual tasks, particularly those requiring robustness to intensity variations.

### Motivate local phase information for motion detection

Global phase, derived from the full-image Fourier transform, is insufficient for detecting localized motion because it integrates information across the entire visual field. In natural scenes, motion is often confined to specific regions—for example, a moving vehicle against a static background. To detect such localized motion, it is necessary to analyze phase information at a local scale. Local phase, computed over spatially restricted regions using windowed transforms, provides a spatially resolved representation of phase dynamics that correlates directly with local motion. By tracking how local phase evolves over time, the system can infer the presence, direction, and coherence of motion within each region.

### Define global and local phase of images

The global phase of an image is defined as the phase component of its two-dimensional Fourier transform. Given a real-valued image \( u(x, y) \), its Fourier transform \( \hat{U}(\omega_x, \omega_y) \) is a complex function that can be written as \( \hat{A}(\omega_x, \omega_y) e^{j\hat{\phi}(\omega_x, \omega_y)} \), where \( \hat{A} \) is the global amplitude and \( \hat{\phi} \) is the global phase. In contrast, the local phase is obtained by applying a windowed Fourier transform, such as the STFT, to localized regions of the image. For a window function \( w(x - x_0, y - y_0) \) centered at \( (x_0, y_0) \), the STFT yields a complex coefficient \( U(\omega_x, \omega_y, x_0, y_0) = A(\omega_x, \omega_y, x_0, y_0) e^{j\phi(\omega_x, \omega_y, x_0, y_0)} \), where \( \phi \) is the local phase.

### Discuss amplitude and phase representation of images

Images can be fully reconstructed from their amplitude and phase spectra when both are available. However, research has shown that phase alone—either global or local—contains sufficient information to reconstruct recognizable images, albeit with potential scaling ambiguities. This underscores the informational richness of phase and justifies its use as a primary feature in visual processing algorithms. Unlike amplitude, which is sensitive to absolute intensity, phase is invariant to uniform scaling and robust to contrast changes, making it particularly suitable for motion detection in varying lighting conditions.

### Global Phase of Images

The global phase of an image is formally defined as the argument of the complex-valued Fourier transform of the image. Mathematically, for an image \( u(x, y) \in \mathbb{R}^2 \), the Fourier transform is \( \hat{U}(\omega_x, \omega_y) = \int_{\mathbb{R}^2} u(x, y) e^{-j(\omega_x x + \omega_y y)} dx\,dy \). Expressing \( \hat{U} \) in polar form yields \( \hat{U} = \hat{A} e^{j\hat{\phi}} \), where \( \hat{\phi}(\omega_x, \omega_y) = \arg(\hat{U}(\omega_x, \omega_y)) \) is the global phase. This phase encodes the spatial offsets of sinusoidal components across the entire image and is directly related to image translation via the Fourier shift theorem.

### Local Phase of Images

#### Introduce Short-Time Fourier Transform (STFT)

The Short-Time Fourier Transform (STFT) extends the Fourier transform to localized analysis by multiplying the image with a window function before transformation. For a window \( w(x, y) \) centered at \( (x_0, y_0) \), the STFT is defined as  
\[
U(\omega_x, \omega_y, x_0, y_0) = \int_{\mathbb{R}^2} u(x, y) w(x - x_0, y - y_0) e^{-j(\omega_x (x - x_0) + \omega_y (y - y_0))} dx\,dy.
\]

#### Define local amplitude and phase of images

The STFT output is a complex number whose magnitude \( A(\omega_x, \omega_y, x_0, y_0) \) is the local amplitude and whose argument \( \phi(\omega_x, \omega_y, x_0, y_0) \) is the local phase. These quantities describe the strength and alignment of frequency components within the windowed region.

#### Relate STFT to Gabor receptive fields

When the window \( w \) is Gaussian, the STFT is equivalent to the response of a complex Gabor receptive field, commonly used to model simple cells in the primary visual cortex. A Gabor filter combines a Gaussian envelope with a complex sinusoid, providing joint localization in space and frequency.

#### Discuss reconstruction of images from local phase

An image can be reconstructed from its local phase measurements by solving a system of linear constraints derived from the orthogonality of the image to sine-modulated windowed functions. This reconstruction is unique up to a scalar multiple if the number of phase measurements exceeds the dimensionality of the image space minus one.

#### Formulate local phase encoding of images

Let \( u(x, y) \) belong to a space of trigonometric polynomials spanned by basis functions \( e_{l_x l_y}(x, y) = e^{j(l_x \Omega_x x / L_x + l_y \Omega_y y / L_y)} \). The local phase measurements \( \phi_{kl,mn} \) from Gabor filters yield linear equations \( \Phi \mathbf{c} = 0 \), where \( \mathbf{c} \) contains the expansion coefficients of \( u \).

#### Define basis functions of Reproducing Kernel Hilbert Space

The image space is a Reproducing Kernel Hilbert Space (RKHS) with inner product defined over a periodic domain. The basis functions \( e_{l_x l_y} \) form an orthonormal set under this inner product, enabling projection-based reconstruction.

#### Represent bank of Gabor receptive fields

A bank of Gabor receptive fields is constructed by translating and modulating a base Gaussian window:  
\[
h_{kl,mn}(x, y) = w(x - k b_0, y - l b_0) e^{-j(\omega_{x_m}(x - k b_0) + \omega_{y_n}(y - l b_0))},
\]  
where \( b_0 \) is the spatial sampling interval and \( \omega_{x_m} = m \omega_0 \).

#### Compute responses of Gabor receptive fields

The response of each Gabor filter to the image is a complex number \( A_{kl,mn} e^{j\phi_{kl,mn}} \), obtained by inner product integration.

#### Extract local amplitude and phase information

The magnitude and argument of the filter response yield the local amplitude and phase, respectively.

#### Formulate reconstruction algorithm from local phase

The reconstruction solves \( \Phi \mathbf{c} = 0 \) for \( \mathbf{c} \), where \( \Phi \) encodes the sine-modulated window integrals. The solution lies in the null space of \( \Phi \).

#### Discuss orthogonality of image to space spanned by functions

The image is orthogonal to all functions of the form \( w(x - k b_0, y - l b_0) \sin(\omega_{x_m}(x - k b_0) + \omega_{y_n}(y - l b_0) + \phi_{kl,mn}) \), leading to the linear constraints.

#### Provide example of reconstruction from local phase

Experimental reconstruction of a 64×64 image from local phase achieved a signal-to-noise ratio of 44.48 dB, demonstrating high fidelity.

#### Discuss necessary condition for perfect reconstruction

Perfect reconstruction (up to scale) requires at least \( (2L_x + 1)(2L_y + 1) - 1 \) phase measurements, ensuring the null space of \( \Phi \) is one-dimensional.

#### Discuss alternative way to obtain unique reconstruction

Including an additional constraint, such as the mean image intensity, ensures uniqueness without increasing the number of phase measurements.

### The Global Phase Equation for Translational Motion

For a globally translating image \( u(x - s_x(t), y - s_y(t), 0) \), the time derivative of the global phase satisfies  
\[
\frac{d\hat{\phi}(\omega_x, \omega_y, t)}{dt} = -\omega_x v_x(t) - \omega_y v_y(t),
\]  
where \( v_x, v_y \) are velocity components. This follows directly from the Fourier shift theorem and shows that global phase rate encodes motion velocity.

### The Local Phase Equation for Translational Motion

#### Discuss local motion detection using STFT

Local motion is detected by applying STFT to windowed regions, yielding local phase \( \phi_{00}(\omega_x, \omega_y, t) \).

#### Define local phase of u(x,y,t) using STFT

The STFT of the time-varying image \( u(x, y, t) \) with a window centered at the origin gives \( A_{00} e^{j\phi_{00}} \).

#### Relate change in local phase to visual motion

The temporal derivative of local phase reflects motion within the window. For a rigidly translating pattern, \( d\phi_{00}/dt \approx -\omega_x v_x - \omega_y v_y \).

#### Discuss invariance of local phase to intensity scaling

Local phase is unaffected by multiplicative intensity changes, making it robust to illumination variations.

#### Derive local phase equation for translational motion

For strictly translational motion within the window,  
\[
\frac{d\phi_{00}}{dt} = -\omega_x v_x - \omega_y v_y + \mathfrak{v}_{00},
\]  
where \( \mathfrak{v}_{00} \) is a residual term that is small for dominant motion features like edges.

#### Discuss added term in local phase equation

The term \( \mathfrak{v}_{00} \) arises from boundary effects and non-uniform motion but is negligible when the moving feature dominates the window response.

### The Block Structure for Computing the Local Phase

#### Define Gaussian windows

Gaussian windows \( (\mathcal{T}_{kl}w)(x, y) = \exp(-((x - x_k)^2 + (y - y_l)^2)/(2\sigma^2)) \) are centered at \( x_k = k b_0, y_l = l b_0 \).

#### Compute 2D Fourier transform of windowed video signal

The windowed signal \( u(x, y, t) (\mathcal{T}_{kl}w)(x, y) \) is transformed via 2D FFT to obtain \( A_{kl} e^{j\phi_{kl}} \).

#### Evaluate integral using 2D FFT

Discrete implementation uses 32×32 pixel blocks with \( \sigma = 4 \) pixels, leveraging efficient FFT algorithms.

#### Process each block independently

Each block is processed in parallel, enabling scalable real-time computation.

#### Discuss window size and object motion detection

Smaller windows increase spatial resolution but reduce noise robustness; larger windows improve signal-to-noise ratio but may blend multiple motions.

#### Illustrate block structure with example

A 64×64 image divided into four 32×32 blocks with overlapping Gaussian windows demonstrates the tiling strategy.

#### Describe de-noising of phase measurements

Phase derivatives are weighted by local amplitude to suppress noise:  
\[
\frac{d\phi_{kl}}{dt} \cdot \frac{A_{kl}}{(1/M^2)\sum A_{kl} + \epsilon}.
\]

### The Phase-Based Detector

#### Provide embodiment of block FFT based algorithm

The algorithm applies Gaussian windows, 2D FFT, phase extraction, temporal high-pass filtering, and Radon transform per block to compute PMI.

### Radon Transform on the Change of Phases

#### Compute Radon transform of phase derivative

The Radon transform integrates the phase derivative \( d\phi_{kl}/dt \) along lines in the frequency domain:  
\[
(\mathcal{R} d\phi_{kl}/dt)(\rho, \theta, t) = \int \frac{d\phi_{kl}}{dt}(\rho \cos\theta - s \sin\theta, \rho \sin\theta + s \cos\theta, t) ds.
\]

#### Define Radon transform

The Radon transform projects a 2D function onto 1D lines parameterized by distance \( \rho \) and angle \( \theta \).

#### Discuss linear structure of phase derivative

For rigid motion, \( d\phi_{kl}/dt \) is linear in \( \omega_x, \omega_y \), so its Radon transform is a ramp function.

#### Compute Radon transform for blocks exhibiting motion

The transform yields a strong response along the line orthogonal to the motion direction.

#### Define correction term

A correction term \( \mathfrak{c}(\rho, \theta) \) accounts for variable line lengths within the circular frequency domain.

#### Compute PMI

The Phase Motion Indicator is  
\[
\text{PMI}_{kl} = \max_{\theta \in [0, \pi)} \sum_{\rho} \left| \frac{(\mathcal{R} d\phi_{kl}/dt)(\rho, \theta, t_0)}{\mathfrak{c}(\rho, \theta)} \right|.
\]

#### Discuss PMI computation

High PMI indicates coherent linear structure in phase derivative, characteristic of motion.

#### Compute direction of motion

The motion direction is \( \hat{\theta}_{kl} = \alpha_{kl} + \pi \cdot \text{sign}(\sum_{\rho > 0} \cdots)/2 \), where \( \alpha_{kl} \) maximizes the Radon response.

#### Discuss direction of motion computation

The sign of the Radon integral determines the motion polarity (forward/backward along the axis).

#### Illustrate phase-motion detection algorithm

A schematic shows input video → block division → Gaussian windowing → 2D FFT → phase extraction → temporal high-pass → Radon transform → PMI → motion detection.

#### Describe algorithm implementation

Implemented in PyCUDA on GPU, achieving real-time performance for HD video.

#### Discuss parallel computing capabilities

Each block is independent, enabling massive parallelization across GPU cores.

#### Illustrate algorithm operation

Step-by-step visualization on a highway video shows PMI heat maps aligning with moving vehicles.

#### Divide algorithm into two parts

Part 1: local phase change extraction; Part 2: motion detection via PMI.

#### Discuss first part of algorithm

Includes Gaussian windowing, 2D FFT, and temporal high-pass filtering to isolate phase dynamics.

#### Apply Gaussian window

Each block is multiplied by a Gaussian to localize spatial analysis.

#### Compute local phase

Phase is extracted from the complex FFT output.

#### Employ temporal high-pass filter

A high-pass filter approximates the time derivative of phase, removing slow drifts.

#### Discuss second part of algorithm

Radon transform and PMI computation identify motion-coherent blocks.

#### Evaluate PMI

PMI quantifies the strength of linear structure in phase derivative.

#### Detect motion

Blocks with PMI above threshold are flagged as containing motion.

#### Compute direction of motion

Direction is inferred from the Radon angle that maximizes PMI.

#### Discuss algorithm parallelization

All blocks are processed simultaneously, with no inter-block communication.

#### Discuss extension to higher dimensions

The method extends to 3D+time data by using 3D FFT and 3D Radon transform over planes.

### Examples of Phase-Based Motion Detection

#### Introduce highway video

A surveillance video of a highway with moderate noise and high contrast.

#### Illustrate motion detection results

PMI heat maps accurately highlight moving cars and swaying trees.

#### Discuss aperture problem

Long horizontal edges yield ambiguous downward motion due to lack of terminators.

#### Introduce low-contrast video

Same scene with reduced contrast (intensity range [0.2, 0.4]).

#### Show motion detection results on low-contrast video

Phase-based detector maintains performance, while Reichardt and Barlow-Levick degrade significantly.

#### Introduce train station video

Indoor surveillance with mixed lighting and moving pedestrians.

#### Illustrate motion detection results

People are reliably detected despite low illumination in background regions.

#### Introduce thermal video

Noisy thermal imagery with moving subjects.

#### Show motion detection results on thermal video

Increased threshold compensates for noise, preserving motion detection.

#### Introduce winterstreet video

Nighttime highway scene with snow and uneven lighting.

#### Illustrate motion detection results

Cars are detected, though low-illumination regions show reduced sensitivity.

#### Discuss noise suppression trade-off

Higher thresholds reduce false positives but may miss dim objects.

#### Introduce motion segmentation

Using motion detections to segment moving objects from background.

#### Discuss block size reduction

Refining segmentation with 16×16 blocks around initial 32×32 detections.

#### Illustrate motion segmentation results

Moving cars are segmented with reasonable boundary accuracy.

#### Discuss applicability to various videos

Algorithm performs consistently across diverse conditions: daylight, night, thermal, low-contrast.

#### Discuss computer-implemented operations

All steps are computable via standard DSP operations: FFT, filtering, matrix multiplication.

#### Discuss scope of disclosure

The invention encompasses all embodiments that use local phase dynamics and Radon-based motion indicators for motion detection.