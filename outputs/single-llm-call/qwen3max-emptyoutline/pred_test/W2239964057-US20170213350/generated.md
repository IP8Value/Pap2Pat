# DESCRIPTION

## BACKGROUND

The detection of visual motion constitutes a fundamental capability in both biological and artificial vision systems, serving critical roles in navigation, object tracking, threat avoidance, and scene understanding. In natural environments, visual motion arises from the relative movement between an observer and objects within a scene or from the independent motion of objects themselves. Biological organisms, ranging from insects to mammals, have evolved highly efficient neural architectures capable of detecting motion with remarkable speed, sensitivity, and energy efficiency. These biological systems often perform motion detection in the early stages of visual processing—such as in the retina of vertebrates or the optic lobe of flies—with minimal synaptic delays and without reliance on complex computational frameworks. This suggests that effective motion detection can be achieved through relatively simple, parallel, and analog-like signal processing mechanisms.

In contrast, conventional computer vision approaches to motion detection have largely relied on optical flow estimation techniques, which compute dense velocity fields by solving partial differential equations derived from brightness constancy assumptions across consecutive image frames. While such methods can yield accurate motion vectors under ideal conditions, they are computationally intensive, often requiring iterative optimization, regularization, and significant memory bandwidth. These characteristics render them poorly suited for real-time applications, especially in resource-constrained environments such as embedded systems, autonomous drones, or neuromorphic hardware. Moreover, optical flow algorithms are sensitive to changes in illumination, low contrast, and noise—conditions under which biological systems remain robust.

Alternative biologically inspired models, such as the Reichardt detector and the Barlow-Levick model, offer simpler architectures based on spatiotemporal filtering, correlation, and inhibition. The Reichardt detector, for instance, computes motion by correlating delayed signals from adjacent photoreceptors, effectively implementing a spatiotemporal cross-correlation that is sensitive to direction. The motion energy model, closely related to the Reichardt detector, uses quadrature pairs of spatiotemporal filters followed by a squaring nonlinearity to extract motion energy. While these models capture essential aspects of biological motion detection and are computationally lighter than full optical flow, they still operate primarily on intensity-based signals and may suffer from contrast dependence and limited robustness under varying lighting conditions.

A largely underexplored yet powerful representation in visual signal processing is phase information. In Fourier analysis, any image can be decomposed into amplitude and phase components in the frequency domain. Historically, amplitude has been emphasized due to its direct relationship with signal energy, while phase was often considered secondary. However, seminal work has demonstrated that phase alone—particularly global phase—contains sufficient information to reconstruct recognizable images, underscoring its critical role in encoding structural and spatial relationships. More recently, local phase, derived from time-frequency or space-frequency representations such as the Short-Time Fourier Transform (STFT) or Gabor transforms, has been shown to encode fine-grained features like edges, corners, and texture boundaries with high fidelity, independent of local intensity variations.

Despite these insights, phase-based approaches have seen limited adoption in motion detection, partly due to the nonlinear nature of phase extraction and the perceived complexity of phase manipulation. Nevertheless, the intrinsic properties of phase—its invariance to uniform intensity scaling, robustness to contrast changes, and sensitivity to subpixel displacements—make it uniquely suited for motion analysis. Furthermore, the mathematical relationship between spatial translation and linear phase shift in the Fourier domain provides a direct theoretical foundation for linking motion to phase dynamics. Extending this principle locally enables the detection of motion occurring in specific regions of a visual field, rather than globally across the entire image.

Existing phase-based motion techniques, such as phase correlation for image registration or phase-based optical flow, typically operate on discrete frame pairs and focus on estimating precise displacement vectors. These methods, while effective for certain tasks, do not fully exploit the continuous-time, parallel, and localized nature of phase evolution during motion. Moreover, they often lack mechanisms to distinguish true motion-induced phase changes from those caused by noise or non-rigid deformations, especially in low-amplitude regions where phase becomes unstable.

There remains a need for a motion detection system that combines the biological plausibility of early visual processing with the computational efficiency required for real-time implementation on modern hardware. Such a system should leverage the representational power of local phase, operate continuously in time, support massive parallelism, and provide robust detection of motion location and direction under diverse imaging conditions—including low contrast, noise, and varying illumination—without requiring explicit velocity estimation or complex optimization. The present invention addresses these needs by introducing a novel phase-based motion detection framework grounded in the temporal dynamics of local phase, enhanced by geometric analysis via the Radon transform, and structured for efficient parallel computation using block-based Fast Fourier Transforms (FFTs).

## SUMMARY

The present invention provides a method and system for detecting visual motion in a sequence of images or video frames by analyzing the temporal evolution of local phase information derived from spatially localized frequency representations of the visual input. Unlike conventional motion detection approaches that rely on intensity gradients, optical flow constraints, or correlation-based mechanisms, the invention exploits the fundamental relationship between spatial translation and linear phase shift in the frequency domain, extended to local image regions through windowed Fourier analysis. This approach enables robust, real-time detection of motion occurrence and direction with high sensitivity to subpixel displacements and immunity to uniform changes in illumination or contrast.

The core of the invention lies in the observation that when a rigid structure moves within a localized region of the visual field, the temporal derivative of the local phase—computed via a Short-Time Fourier Transform (STFT) with a spatial window such as a Gaussian function—exhibits a characteristic linear structure in the frequency domain. Specifically, the rate of change of local phase at each spatial frequency component is approximately proportional to the dot product of the motion velocity vector and the spatial frequency vector. This linear dependency creates a planar structure in the three-dimensional space of (ω_x, ω_y, dϕ/dt), which can be efficiently detected using integral geometry techniques.

To operationalize this insight, the invention partitions the input image into overlapping spatial blocks, each associated with a localized window function. For each block, a two-dimensional Fast Fourier Transform (FFT) is applied to the windowed image region to obtain complex-valued spectral coefficients. The phase of these coefficients is extracted and temporally differentiated—either through analytical differentiation or high-pass filtering—to yield a map of local phase changes over the frequency grid of the block. Critically, this phase change map is weighted by the corresponding local amplitude to suppress noise in regions where the signal energy is low, thereby enhancing reliability.

The invention then applies the Radon transform to the amplitude-weighted phase change map within a bounded circular frequency domain. The Radon transform integrates the phase change values along straight lines parameterized by angle and distance from the origin. Due to the linear structure induced by translational motion, this integration yields strong responses along the orientation corresponding to the motion direction, while random noise or non-motion-related phase fluctuations produce diffuse, low-magnitude responses. From the Radon-transformed data, a Phase Motion Indicator (PMI) is computed for each block as the maximum absolute integrated response over all possible orientations. If the PMI exceeds a predetermined threshold, motion is declared present in that spatial block.

Furthermore, the orientation yielding the maximum PMI directly encodes the direction of motion. By examining the sign of the integrated response on either side of the zero-crossing line in the frequency domain, the invention resolves the directional ambiguity inherent in unsigned magnitude responses, thereby providing a signed estimate of motion direction. This directional information is inherently coarse but sufficient for many applications such as motion segmentation, event detection, or attentional guidance.

The entire process is structured to maximize parallelism: each spatial block is processed independently, and within each block, the FFT and subsequent operations are vectorized. This architecture maps naturally onto modern parallel computing platforms, including Graphics Processing Units (GPUs), Field-Programmable Gate Arrays (FPGAs), and multi-core processors, enabling real-time performance even at high-definition resolutions. The algorithm requires no iterative optimization, no storage of historical frames beyond a minimal temporal buffer for differentiation, and no explicit modeling of object shape or background statistics.

In addition to motion detection, the invention demonstrates that local phase alone—without amplitude information—is sufficient to reconstruct bandlimited visual scenes up to a constant scale factor. This reconstruction capability underscores the informational richness of phase and validates its use as a primary representation for visual processing. The motion detection method described herein thus operates on a representation that is both theoretically complete and practically efficient.

The invention further reveals a deep connection between the proposed phase-based detector and classical biological motion models. By expressing the temporal derivative of phase in terms of real and imaginary parts of the analytic signal, the numerator of the derivative corresponds to a second-order Volterra kernel acting on quadrature-filtered inputs—a structure mathematically analogous to the Reichardt detector and motion energy model. This equivalence suggests that biological systems may implicitly perform phase-based computations through their quadrature filter arrangements and nonlinear interactions.

Experimental validation on standard video datasets—including highway surveillance, indoor scenes, thermal imagery, and low-light conditions—demonstrates superior performance compared to Reichardt and Barlow-Levick detectors, particularly under low-contrast and noisy conditions. The phase-based method maintains consistent detection accuracy across a wide range of illumination levels due to its contrast invariance, whereas intensity-based detectors degrade quadratically or linearly with contrast reduction. Moreover, the detected motion signals are effectively used for motion segmentation tasks, achieving results comparable to or better than those obtained using optical flow, despite using only motion cues and no color or background modeling.

In summary, the invention provides a novel, efficient, and biologically plausible framework for visual motion detection based on the temporal dynamics of local phase. It offers advantages in robustness, parallelizability, contrast invariance, and subpixel sensitivity, making it suitable for deployment in real-world applications ranging from autonomous vehicles and robotics to biomedical imaging and neuromorphic engineering.

## DETAILED DESCRIPTION

### Global Phase of Images

The global phase of an image refers to the angular component of its two-dimensional Fourier transform when expressed in polar coordinates. Given a real-valued image \( u(x, y) \) defined over the spatial domain \( \mathbb{R}^2 \), its Fourier transform is a complex-valued function \( \hat{U}(\omega_x, \omega_y) \) given by:

\[
\hat{U}(\omega_x, \omega_y) = \int_{\mathbb{R}^2} u(x, y) e^{-j(\omega_x x + \omega_y y)} \, dx \, dy,
\]

where \( j \) denotes the imaginary unit, and \( (\omega_x, \omega_y) \in \mathbb{R}^2 \) represents spatial frequencies. This complex spectrum can be decomposed into magnitude and phase as:

\[
\hat{U}(\omega_x, \omega_y) = \hat{A}(\omega_x, \omega_y) e^{j \hat{\phi}(\omega_x, \omega_y)},
\]

where \( \hat{A}(\omega_x, \omega_y) \geq 0 \) is the global amplitude (or magnitude spectrum), and \( \hat{\phi}(\omega_x, \omega_y) \in [0, 2\pi) \) is the global phase. The global phase encodes the relative positioning of sinusoidal components across the entire image and is critically responsible for preserving structural information such as edges, shapes, and object boundaries. It is well established in the literature that global phase alone—when combined with a generic or even flat amplitude spectrum—can reconstruct images that retain recognizable content, whereas amplitude-only reconstructions appear as unstructured noise. This phenomenon highlights the dominant role of phase in representing the semantic and geometric structure of natural images.

The global phase is particularly sensitive to spatial translations. According to the Fourier shift theorem, if an image \( u(x, y) \) is translated by \( (s_x, s_y) \) to produce \( u(x - s_x, y - s_y) \), its Fourier transform becomes:

\[
\hat{U}(\omega_x, \omega_y) e^{-j(\omega_x s_x + \omega_y s_y)}.
\]

Thus, the global phase shifts linearly with the translation vector, with the amount of shift at each frequency being proportional to the dot product of the frequency vector \( (\omega_x, \omega_y) \) and the displacement vector \( (s_x, s_y) \). This linear relationship forms the theoretical basis for phase-based motion analysis. However, because the global phase integrates information over the entire image, it cannot localize motion to specific regions; it only reflects the net effect of all movements in the scene. Consequently, while global phase is informative about overall image shifts—such as those caused by camera ego-motion—it is insufficient for detecting localized object motion, which is the primary focus of the present invention. Therefore, the concept of global phase serves as a foundational principle that motivates the development of a localized counterpart capable of spatially resolving motion events.

### Local Phase of Images

To overcome the spatial non-locality of global phase, the invention employs the concept of local phase, derived from a space-frequency representation of the image. Specifically, the Short-Time (or Short-Space) Fourier Transform (STFT) is used to analyze the image in localized spatial neighborhoods. For a given window function \( w(x, y) \)—typically a smooth, compactly supported function such as a Gaussian—the STFT of the image \( u(x, y) \) centered at spatial location \( (x_0, y_0) \) is defined as:

\[
U(\omega_x, \omega_y, x_0, y_0) = \int_{\mathbb{R}^2} u(x, y) w(x - x_0, y - y_0) e^{-j(\omega_x (x - x_0) + \omega_y (y - y_0))} \, dx \, dy.
\]

This operation effectively restricts the Fourier analysis to the region where the window \( w \) is non-negligible, thereby associating each frequency component with a specific spatial location. The resulting complex-valued function can again be expressed in polar form:

\[
U(\omega_x, \omega_y, x_0, y_0) = A(\omega_x, \omega_y, x_0, y_0) e^{j \phi(\omega_x, \omega_y, x_0, y_0)},
\]

where \( A \) is the local amplitude and \( \phi \) is the local phase. The local phase \( \phi(\omega_x, \omega_y, x_0, y_0) \) captures the phase offset of sinusoidal components within the windowed region around \( (x_0, y_0) \), making it sensitive to local features such as edges, corners, and texture patterns.

When the window function \( w \) is Gaussian, the STFT corresponds to the response of a bank of complex Gabor filters, which are widely used in modeling receptive fields of simple and complex cells in the primary visual cortex (V1). A complex Gabor filter has the form:

\[
h(x, y) = e^{-(x^2 + y^2)/(2\sigma^2)} e^{-j(\omega_x x + \omega_y y)},
\]

where \( \sigma \) controls the spatial extent of the receptive field. The real and imaginary parts of this filter form a quadrature pair, enabling the extraction of both amplitude and phase information. The local phase computed via such filters is known to be indicative of feature type and position, independent of local contrast or illumination, due to its normalization by amplitude in the underlying analytic signal representation.

Importantly, the invention demonstrates that local phase alone—without any amplitude information—is sufficient to reconstruct the original image up to a constant scale factor, provided the image is bandlimited and sufficiently sampled. This is achieved by formulating the phase measurements as linear constraints on the image coefficients in a trigonometric polynomial basis. Specifically, if the image is modeled as a finite sum of complex exponentials, the condition that the imaginary part of the windowed Fourier transform vanishes (after phase alignment) yields a homogeneous system of linear equations. The solution to this system, lying in the null space of a measurement matrix constructed from the known phases, recovers the image coefficients. This reconstruction property validates the informational completeness of local phase and justifies its use as a primary representation for motion analysis.

### The Global Phase Equation for Translational Motion

Consider a time-varying visual stimulus \( u(x, y, t) \) that undergoes pure translational motion, such that:

\[
u(x, y, t) = u(x - s_x(t), y - s_y(t), 0),
\]

where \( s_x(t) = \int_0^t v_x(\tau) d\tau \) and \( s_y(t) = \int_0^t v_y(\tau) d\tau \) are the cumulative displacements in the x and y directions, and \( v_x(t), v_y(t) \) are the instantaneous velocity components. Taking the 2D spatial Fourier transform at each time \( t \), the Fourier shift theorem implies:

\[
\hat{U}(\omega_x, \omega_y, t) = \hat{U}(\omega_x, \omega_y, 0) e^{-j(\omega_x s_x(t) + \omega_y s_y(t))}.
\]

Differentiating the global phase \( \hat{\phi}(\omega_x, \omega_y, t) \) with respect to time yields:

\[
\frac{d\hat{\phi}}{dt}(\omega_x, \omega_y, t) = -\omega_x v_x(t) - \omega_y v_y(t).
\]

This equation establishes a direct linear relationship between the temporal derivative of global phase and the motion velocity vector. Each frequency component contributes a phase rate proportional to its projection onto the velocity vector. Thus, in the case of global translation, the entire phase spectrum evolves in a coordinated, linear fashion dictated by the motion parameters. However, this global relationship does not hold for scenes containing multiple independently moving objects, as the phase changes become spatially heterogeneous. Therefore, while this equation provides theoretical motivation, the invention extends this principle to the local domain to handle complex, real-world motion scenarios.

### The Local Phase Equation for Translational Motion

For localized translational motion within a windowed region, the invention derives an analogous relationship for the local phase. Assuming that within the support of a window centered at \( (x_0, y_0) \), the image undergoes rigid translation:

\[
u(x, y, t) = u(x - s_x(t), y - s_y(t), 0), \quad \text{for } (x, y) \in \text{supp}(w(x - x_0, y - y_0)),
\]

the local phase \( \phi(\omega_x, \omega_y, x_0, y_0, t) \) satisfies:

\[
\frac{d\phi}{dt}(\omega_x, \omega_y, x_0, y_0, t) = -\omega_x v_x(t) - \omega_y v_y(t) + \mathfrak{v}(\omega_x, \omega_y, x_0, y_0, t),
\]

where \( \mathfrak{v} \) is a residual term that accounts for boundary effects, non-rigidity, and windowing artifacts. Through simulation and analysis, it is observed that for prominent features such as moving edges, the dominant terms \( -\omega_x v_x - \omega_y v_y \) govern the phase dynamics, while \( \mathfrak{v} \) remains relatively small. This near-linearity in the frequency domain is the key insight exploited by the invention.

The local phase derivative is computed either by numerical differentiation of the unwrapped phase over time or, more robustly, by applying a temporal high-pass filter to the raw phase signal to approximate the derivative while mitigating noise. Additionally, to address phase instability in low-amplitude regions—where small perturbations can cause large phase jumps—the derivative is weighted by the local amplitude normalized by the mean amplitude in the block. This denoising step ensures that only reliable phase changes contribute to motion detection.

### The Block Structure for Computing the Local Phase

To enable efficient and parallel computation, the invention partitions the input image into a grid of overlapping spatial blocks. Each block is associated with a Gaussian window \( w_{kl}(x, y) = \exp(-((x - x_k)^2 + (y - y_l)^2)/(2\sigma^2)) \), where \( (x_k, y_l) = (k b_0, l b_0) \) are the centers of the windows, \( b_0 \) is the inter-block spacing, and \( \sigma \) is the standard deviation controlling the window size. The overlap between blocks ensures smooth spatial coverage and mitigates edge effects.

For each block \( (k, l) \), the windowed image patch is zero-padded to a size \( M \times M \) (typically a power of two, e.g., 32×32) to facilitate efficient computation via the 2D Fast Fourier Transform (FFT). The FFT yields complex coefficients \( U_{kl}(\omega_{x_m}, \omega_{y_n}, t) \) at discrete frequencies \( \omega_{x_m} = m \omega_0 \), \( \omega_{y_n} = n \omega_0 \), where \( \omega_0 = 2\pi/M \) and \( m, n \in \{-M/2, \dots, M/2 - 1\} \). The local phase \( \phi_{kl}(\omega_{x_m}, \omega_{y_n}, t) \) is extracted as the argument of these coefficients.

The block size \( M \) and window standard deviation \( \sigma \) are chosen based on the expected size of moving objects: smaller windows enhance spatial resolution but are more noise-sensitive, while larger windows improve signal-to-noise ratio but may blend multiple motions. The independence of block processing allows the entire pipeline to be parallelized across thousands of threads on a GPU, with each thread handling one block.

### The Phase-Based Detector

The phase-based motion detector operates in two stages. First, for each block, the temporal derivative of local phase is computed and denoised using amplitude weighting. Second, the Radon transform is applied to the resulting phase change map to detect the presence and direction of motion.

The Radon transform integrates the phase derivative \( d\phi_{kl}/dt \) over straight lines in the frequency domain. For a circular domain \( C = \{ (\omega_x, \omega_y) : \omega_x^2 + \omega_y^2 < \pi^2 \} \), the Radon transform at angle \( \theta \) and offset \( \rho \) is:

\[
\left( \mathcal{R} \frac{d\phi_{kl}}{dt} \right)(\rho, \theta, t) = \int_{\mathbb{R}} \frac{d\phi_{kl}}{dt}(\rho \cos\theta - s \sin\theta, \rho \sin\theta + s \cos\theta, t) \cdot 1_C(\cdot) \, ds.
\]

Under pure translational motion, this integral yields a response proportional to \( \rho (-v_x \cos\theta - v_y \sin\theta) \), scaled by a correction factor \( \mathfrak{c}(\rho, \theta) \) accounting for line length within \( C \). The Phase Motion Indicator (PMI) is then defined as:

\[
\text{PMI}_{kl} = \max_{\theta \in [0, \pi)} \sum_{\rho} \left| \frac{ \left( \mathcal{R} (d\phi_{kl}/dt) \right)(\rho, \theta, t) }{ \mathfrak{c}(\rho, \theta) } \right|.
\]

A high PMI indicates coherent linear structure in the phase derivative, characteristic of motion. The optimal angle \( \alpha_{kl} \) maximizing the PMI gives the motion orientation, and the sign of the integrated response determines the direction along that axis.

### Radon Transform on the Change of Phases

The Radon transform serves as a geometric detector of linear structures in the frequency-phase derivative space. Unlike direct thresholding of phase changes—which is vulnerable to noise—the Radon transform accumulates evidence along potential motion directions, enhancing signal-to-noise ratio. The bounded circular domain ensures that only physically meaningful frequencies (below the Nyquist limit) are considered. The correction factor \( \mathfrak{c}(\rho, \theta) \) normalizes for the varying lengths of integration paths, ensuring consistent response magnitudes across orientations.

This approach effectively converts the problem of motion detection into a line detection problem in a transformed domain, leveraging well-established integral geometry. The computational cost of the Radon transform is manageable due to the small block size (e.g., 32×32), and fast implementations using interpolation or Fourier slice theorem can be employed.

### Examples of Phase-Based Motion Detection

The invention has been validated on diverse video sequences. In a highway surveillance video, the phase-based detector accurately identified moving vehicles and swaying trees, even under camera jitter and low-contrast conditions. Compared to Reichardt and Barlow-Levick detectors, it maintained higher detection rates at reduced contrast levels due to its phase-based, contrast-invariant nature. In thermal and low-light videos, increased noise was handled by raising the PMI threshold, demonstrating adaptability.

Motion segmentation using the detected blocks—refined with smaller 16×16 blocks at boundaries—successfully isolated moving objects. When compared to optical flow-based segmentation, the phase method performed comparably or better, particularly for slow-moving objects that optical flow might miss due to thresholding on velocity magnitude.

These examples confirm that the invention provides a robust, efficient, and biologically plausible solution for real-time motion detection across challenging imaging conditions.