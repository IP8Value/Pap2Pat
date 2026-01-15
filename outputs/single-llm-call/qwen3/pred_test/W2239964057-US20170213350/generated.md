## BACKGROUND

- introduce information processing system  
An information processing system operates at multiple levels of abstraction, each defining distinct functional roles in the transformation of sensory input into meaningful output. At the algorithmic level, the system is characterized by its procedural logic—how data is transformed through a sequence of operations independent of physical implementation. At the physical circuit level, this logic is realized through hardware components, whether biological neural circuits or silicon-based digital processors. The design of such systems benefits from aligning algorithmic principles with the constraints and capabilities of their physical substrate. In the context of visual motion detection, this duality is particularly salient, as biological systems achieve high-speed, low-power motion perception through parallel analog processing, while conventional computational models rely on iterative, resource-intensive numerical computations. The challenge lies in bridging these domains: developing an algorithm that emulates the efficiency of biological motion detection while remaining implementable on standard digital hardware. This alignment enables the creation of systems that are not only accurate but also scalable, energy-efficient, and suitable for real-time deployment in dynamic environments.

- motivate visual motion detection  
Visual motion detection is a fundamental capability essential for survival across a wide range of animal species, from insects to primates. The ability to perceive movement in the visual field enables organisms to navigate complex environments, avoid predators, track prey, and coordinate locomotion. In vertebrates, motion signals are extracted as early as the retina, where specialized ganglion cells respond selectively to directional motion before any higher cortical processing occurs. Similarly, in insects, motion detection emerges after only a few synaptic connections from photoreceptors, indicating that the underlying mechanisms are streamlined, efficient, and optimized for speed rather than complexity. These biological systems operate continuously in analog form, leveraging the temporal dynamics of light intensity changes without relying on discrete sampling or spiking events. This suggests that motion detection is not merely a post-processing task but a primary sensory function that must be computed in real time under stringent temporal constraints. For artificial systems, replicating this efficiency is critical for applications in autonomous navigation, surveillance, robotics, and human-machine interaction, where latency, power consumption, and robustness to environmental variability are paramount.

- describe biological motion detection  
Biological motion detection has been extensively studied through models that capture the neural architecture and computational principles underlying motion perception. The Reichardt detector, originally proposed for insect vision, operates by correlating signals from adjacent photoreceptors with a temporal delay, effectively detecting spatiotemporal correlations indicative of motion direction. The motion energy model, derived from studies in mammalian visual cortex, employs spatiotemporal filters that respond to oriented patterns moving in specific directions, followed by a nonlinear squaring operation to compute motion energy. The Barlow-Levick model introduces inhibitory mechanisms that suppress responses to motion in the null direction, enhancing directional selectivity. These models share common features: they operate in parallel across the visual field, utilize analog signal processing, and rely on local interactions rather than global computations. Importantly, they do not require explicit estimation of velocity or displacement but instead detect the presence and direction of motion through changes in phase, amplitude, or correlation over time. The simplicity of these circuits—often implemented with just a few synaptic connections—suggests that motion detection is not a high-level cognitive function but a foundational sensory process, optimized for speed and robustness under noisy and variable conditions.

- limitations of computer-based motion detection  
Conventional computer-based motion detection algorithms, such as those based on optic flow, rely on estimating spatial intensity gradients across consecutive image frames and solving for velocity fields under assumptions of brightness constancy and spatial smoothness. While these methods can achieve high accuracy under controlled conditions, they suffer from significant computational complexity, requiring iterative optimization, dense pixel-wise comparisons, and large memory buffers to store intermediate states. These demands make real-time implementation on embedded or mobile platforms challenging, especially at high resolutions or frame rates. Furthermore, optic flow methods are highly sensitive to illumination changes, low contrast, motion blur, and textureless regions, leading to unreliable or noisy estimates. They also lack the inherent parallelism of biological systems, often requiring sequential processing that introduces latency. Additionally, these algorithms typically aim to compute precise velocity vectors, which introduces unnecessary complexity when the goal is merely to detect the presence and direction of motion. The reliance on intensity-based features renders them vulnerable to variations in lighting, color, and contrast, limiting their robustness in real-world scenarios where such conditions are unpredictable.

- summarize existing biological models  
Existing biological models of motion detection—Reichardt, motion energy, and Barlow-Levick—provide compelling frameworks for understanding how motion is extracted at the earliest stages of visual processing. Each model exploits temporal delays, spatial filtering, and nonlinear operations to achieve directional selectivity. The Reichardt detector uses delayed correlation between neighboring spatial channels, while the motion energy model employs spatiotemporal receptive fields tuned to specific motion directions. The Barlow-Levick model incorporates asymmetric inhibition to suppress responses to non-preferred motion. Despite their differences, all three models share a reliance on local, parallel processing and a tolerance for intensity variations through normalization or gain control. However, none of these models explicitly leverage phase information, which has been shown in recent studies to be a dominant carrier of structural content in visual scenes. While these models successfully explain directional selectivity, they do not account for the remarkable robustness of biological motion detection under low contrast or noisy conditions, nor do they provide a clear pathway for efficient hardware implementation. The absence of a phase-based mechanism in these models represents a gap in understanding how motion signals are encoded and decoded in natural visual systems.

## SUMMARY

- outline method for motion detection  
A method for detecting visual motion is disclosed that leverages the temporal evolution of local phase information in a video sequence to identify regions of motion without relying on intensity-based gradients or explicit velocity estimation. The method operates by dividing the visual field into overlapping spatial blocks, each processed independently to compute the local phase of the intensity signal using a Short-Time Fourier Transform with a Gaussian window. The temporal derivative of the local phase is then calculated for each spatial-frequency component within each block, providing a motion-sensitive signal that is invariant to global intensity scaling and robust to noise. A Radon transform is applied to the phase derivative across the spatial-frequency domain of each block to extract a motion indicator that quantifies the presence and orientation of coherent motion. Motion is detected when this indicator exceeds a predefined threshold, and the direction of motion is determined from the angular location of the maximum response in the Radon domain. The entire process is structured to enable massive parallelism, with each block processed independently, making the method highly suitable for implementation on parallel computing architectures such as GPUs or FPGAs.

- describe system and computer readable medium  
The system comprises a video input interface, a memory unit for storing video frames, and a processing unit configured to execute a sequence of computational operations for motion detection. The processing unit is programmed to apply a Gaussian window to each overlapping block of the video sequence, compute the two-dimensional Fourier transform of each windowed block, and extract the local phase component from the complex-valued frequency representation. A temporal high-pass filter is applied to the phase signal to isolate changes attributable to motion, and the resulting phase derivative is subjected to a Radon transform over a circular frequency domain to compute a Phase Motion Indicator for each block. The system further includes a thresholding module that compares the Phase Motion Indicator against a preset value to determine motion presence, and a direction estimation module that computes the orientation of motion from the angular coordinate of the maximum Radon response. The system is implemented on a computer-readable medium containing executable instructions that, when executed by a processor, cause the system to perform the steps of the motion detection method. The medium may be embodied in volatile or non-volatile memory, firmware, or hardware logic, and may be distributed across multiple processing units for parallel execution.

## DETAILED DESCRIPTION

- introduce phase information in images  
Phase information in images encodes the spatial arrangement of frequency components that define structural features such as edges, contours, and textures. Unlike amplitude, which reflects the magnitude of intensity variations, phase determines the relative timing and alignment of sinusoidal components that collectively reconstruct the image’s geometry. In natural scenes, phase carries the majority of perceptually relevant information, as demonstrated by image reconstruction experiments where only phase is retained and amplitude is randomized. This property suggests that phase is not merely a mathematical byproduct of the Fourier transform but a primary carrier of visual structure. In dynamic visual scenes, changes in phase over time correspond to the motion of these structural elements, making phase a natural and efficient signal for detecting motion without requiring explicit intensity comparisons. By focusing on phase rather than amplitude, motion detection becomes invariant to global illumination changes and contrast variations, aligning more closely with the robustness observed in biological visual systems.

- motivate local phase information for motion detection  
Global phase information, while informative about the overall structure of an image, lacks spatial specificity and cannot localize motion to particular regions of the visual field. Local phase, derived through windowed Fourier analysis, provides a spatially resolved representation of phase that varies independently across different regions of the image. This spatial localization is critical for detecting motion of individual objects within a scene, where multiple independent motions may occur simultaneously. The local phase of a region changes predictably when a rigid edge or texture moves across the windowed area, producing a coherent, directional pattern in the frequency domain. This pattern is distinct from random noise or illumination fluctuations, which tend to induce diffuse, non-structured changes in phase. By analyzing the temporal evolution of local phase, motion can be detected without relying on intensity gradients, making the method robust under low-contrast, noisy, or non-uniform lighting conditions. Furthermore, the local nature of the computation allows for parallel processing, mirroring the distributed architecture of biological motion detectors.

- define global and local phase of images  
The global phase of an image is defined as the phase component of its full two-dimensional Fourier transform, representing the collective phase relationship of all spatial frequencies across the entire image domain. In contrast, the local phase is defined as the phase component of the Short-Time Fourier Transform, computed over a localized region of the image defined by a window function centered at a specific spatial coordinate. While the global phase captures the overall spatial offset of the image, the local phase captures the phase structure within a restricted spatial neighborhood, enabling the detection of local features such as edges and their motion. The local phase is thus a spatially indexed function that varies with position and time, forming a four-dimensional representation (x, y, ωx, ωy, t) that encodes both the spatial frequency content and its temporal evolution within each localized region.

- discuss amplitude and phase representation of images  
Images can be represented in polar form as the product of amplitude and phase components derived from a complex-valued transform such as the Fourier or Short-Time Fourier Transform. The amplitude represents the strength of each frequency component, while the phase encodes the spatial positioning of these components. Traditionally, image reconstruction has relied on both amplitude and phase, as the amplitude provides the magnitude of energy and the phase provides the structural alignment. However, recent theoretical and empirical work has demonstrated that phase alone can reconstruct images with high fidelity, indicating that amplitude is often redundant for structural representation. In motion detection, amplitude serves a secondary role: it provides a measure of signal confidence, allowing for noise suppression by weighting phase changes according to local amplitude. This dual representation enables a robust motion signal that is sensitive to structural motion while rejecting noise in low-amplitude regions.

### Global Phase of Images.

- define global phase of images  
The global phase of an image is the phase angle of its two-dimensional Fourier transform, computed over the entire spatial extent of the image without spatial localization. It reflects the cumulative phase relationship among all frequency components and determines the global alignment of the image’s structural features. When an image undergoes a uniform translation, the global phase shifts linearly with respect to spatial frequency, following the Fourier shift theorem. This relationship forms the basis for global motion estimation techniques but is insufficient for detecting localized motion, as it averages motion signals across the entire scene, obscuring the presence of independent moving objects.

### Local Phase of Images

- introduce Short-Time Fourier Transform (STFT)  
The Short-Time Fourier Transform is a time-frequency analysis technique that applies a sliding window to localize the Fourier transform in space. For each spatial position, the image is multiplied by a window function centered at that position, and the Fourier transform is computed over the windowed region. This produces a four-dimensional representation of the image in terms of spatial location and frequency content, enabling the analysis of how spectral components vary locally across the visual field. The STFT is particularly well-suited for motion detection because it preserves both spatial and frequency information simultaneously, allowing phase changes to be tracked in localized regions over time.

- define local amplitude and phase of images  
The local amplitude of an image is the magnitude of the Short-Time Fourier Transform at a given spatial location and frequency, representing the strength of the frequency component within the windowed region. The local phase is the phase angle of the same transform, representing the spatial alignment of the frequency component relative to the window center. Together, they form a complex-valued representation of the image’s local structure, where amplitude indicates signal confidence and phase indicates structural orientation. The local phase is the key variable for motion detection, as its temporal derivative is directly influenced by the motion of structural features within the window.

- relate STFT to Gabor receptive fields  
The Short-Time Fourier Transform with a Gaussian window is mathematically equivalent to the response of a bank of complex-valued Gabor receptive fields, which are widely used to model simple cells in the primary visual cortex. Each Gabor filter consists of a sinusoidal plane wave modulated by a Gaussian envelope, producing a spatially localized frequency detector tuned to a specific orientation and scale. The STFT can thus be interpreted as a parallel array of Gabor filters, each responding to a unique combination of spatial location, frequency, and phase. This equivalence provides a direct neurobiological interpretation of the method, aligning the computational architecture with known mechanisms of early visual processing.

- discuss reconstruction of images from local phase  
It is possible to reconstruct an image from its local phase information alone, up to a constant scale, by solving a system of linear constraints derived from the orthogonality of the image to a set of sine-modulated Gabor functions. The reconstruction relies on the fact that the phase determines the spatial alignment of frequency components, and when these alignments are known across multiple spatial locations and frequencies, the original image can be recovered by finding a signal that satisfies all phase-induced orthogonality conditions. This demonstrates that phase is not merely a secondary descriptor but a sufficient carrier of structural information, supporting the use of phase as the primary signal for motion detection.

- formulate local phase encoding of images  
Local phase encoding is formulated as a mapping from the image domain to a high-dimensional phase space, where each point corresponds to the phase value at a specific spatial location and frequency component. This encoding transforms the image into a set of phase measurements that are invariant to intensity scaling and sensitive to structural displacement. The encoding is achieved by applying a bank of Gaussian-windowed Gabor filters across the image, computing the phase of each filter response, and storing these values as a phase tensor indexed by spatial position and frequency. This tensor serves as the input to the motion detection algorithm.

- define basis functions of Reproducing Kernel Hilbert Space  
The space of bandlimited images under consideration forms a Reproducing Kernel Hilbert Space, where the basis functions are complex exponentials modulated by periodic boundary conditions. These basis functions span the space of all possible images that can be represented within a finite bandwidth, and they serve as the mathematical foundation for the reconstruction of images from phase measurements. The inner product in this space defines the orthogonality relationships that constrain the image reconstruction problem, ensuring that the solution is unique up to a scalar factor.

- represent bank of Gabor receptive fields  
The bank of Gabor receptive fields is represented as a set of complex-valued functions, each defined by a center location, orientation, spatial frequency, and scale. These functions are arranged in a regular grid across the image plane, with overlapping windows to ensure complete spatial coverage. Each Gabor function acts as a matched filter for a specific local frequency and orientation, producing a complex response that is decomposed into amplitude and phase. The collective response of the bank forms a dense, redundant representation of the image’s local structure, enabling robust phase-based analysis.

- compute responses of Gabor receptive fields  
The responses of the Gabor receptive fields are computed by convolving the image with each filter in the bank, yielding a complex-valued output for each spatial location and frequency channel. This convolution is efficiently implemented using the Fast Fourier Transform, where the image and each Gabor filter are transformed into the frequency domain, multiplied, and then inverse-transformed back to the spatial domain. The result is a four-dimensional tensor of complex values, from which amplitude and phase are extracted for each spatial-frequency-location combination.

- extract local amplitude and phase information  
Local amplitude and phase information are extracted from the complex-valued responses of the Gabor receptive fields by computing the magnitude and argument of each complex sample. The amplitude is used to weight the phase derivative in subsequent steps, suppressing noise in low-energy regions. The phase is retained as the primary signal for motion detection, as its temporal evolution encodes the displacement of structural features. This extraction is performed in real time for each video frame, producing a continuous stream of phase measurements that are processed to detect motion.

- formulate reconstruction algorithm from local phase  
The reconstruction algorithm from local phase is formulated as a linear system of equations derived from the orthogonality condition that the image must be orthogonal to a set of sine-modulated Gabor functions. Each phase measurement imposes a linear constraint on the coefficients of the image’s basis expansion. The solution is found by computing the null space of a matrix formed from these constraints, yielding the image coefficients that best satisfy all phase measurements. This algorithm demonstrates that phase alone can reconstruct the image, validating its use as a sufficient representation for motion detection.

- discuss orthogonality of image to space spanned by functions  
The image is constrained to be orthogonal to the space spanned by sine-modulated Gabor functions whose phase is fixed by the measured local phase values. This orthogonality condition arises because the phase measurement implies that the image has zero projection onto a specific sine component, effectively removing degrees of freedom from the solution space. The number of such constraints must exceed the number of unknown image coefficients to ensure a unique solution, up to a scalar factor. This mathematical constraint underpins the reconstruction algorithm and confirms that phase information contains sufficient structure to define the image.

- provide example of reconstruction from local phase  
An example of image reconstruction from local phase alone is demonstrated using a natural image, where the amplitude of the Gabor responses is discarded and only the phase is retained. The reconstruction algorithm is applied to recover the image, resulting in a high-fidelity approximation with a signal-to-noise ratio exceeding 44 dB. The reconstructed image retains edges, textures, and structural details despite the absence of amplitude information, confirming that phase is the dominant carrier of visual structure.

- discuss necessary condition for perfect reconstruction  
A necessary condition for perfect reconstruction of an image from local phase is that the number of phase measurements exceeds the number of degrees of freedom in the image representation minus one. This ensures that the system of linear constraints is sufficiently overdetermined to yield a unique solution in the null space of the constraint matrix. If the number of measurements is insufficient, multiple images may satisfy the phase constraints, leading to ambiguity in reconstruction.

- discuss alternative way to obtain unique reconstruction  
An alternative method to ensure a unique reconstruction is to include an additional constraint, such as the mean intensity of the image, as a known reference value. This constraint fixes the scalar ambiguity inherent in phase-only reconstruction, allowing the recovered image to be scaled to match the original intensity range. This approach is particularly useful in practical applications where absolute intensity values are available or can be estimated.

### The Global Phase Equation for Translational Motion

- derive global phase equation for translational motion  
For a visual stimulus undergoing uniform translational motion, the global phase of its Fourier transform evolves linearly with time, proportional to the product of spatial frequency and velocity components. This relationship is derived from the Fourier shift theorem, which states that a spatial translation corresponds to a linear phase shift in the frequency domain. The temporal derivative of the global phase is therefore equal to the negative dot product of the frequency vector and the velocity vector, providing a direct link between motion and phase dynamics. This equation forms the theoretical foundation for global motion estimation but is insufficient for detecting localized motion within complex scenes.

### The Local Phase Equation for Translational Motion

- discuss local motion detection using STFT  
Local motion detection is achieved by applying the Short-Time Fourier Transform to localized regions of the visual field, allowing the phase dynamics to be analyzed independently in each region. Unlike global phase, which averages motion across the entire scene, local phase responds only to motion occurring within the window support. This enables the detection of multiple independent motions simultaneously, as each window captures the phase changes induced by objects moving within its spatial bounds.

- define local phase of u(x,y,t) using STFT  
The local phase of a spatiotemporal intensity function u(x,y,t) is defined as the phase component of the Short-Time Fourier Transform computed with a Gaussian window centered at each spatial location (x₀,y₀). The transform is evaluated at each time point, producing a time-varying phase function ϕ(x₀,y₀,ωₓ,ωᵧ,t) that describes how the phase of local frequency components evolves over time due to motion.

- relate change in local phase to visual motion  
The temporal derivative of the local phase is directly related to the velocity of motion within the window. For a rigidly translating edge or texture, the phase derivative exhibits a linear dependence on spatial frequency, with the slope determined by the velocity vector. This relationship is analogous to the global phase equation but is localized, allowing motion to be detected even in the presence of background clutter or non-uniform illumination.

- discuss invariance of local phase to intensity scaling  
The local phase is invariant to uniform scaling of intensity, meaning that changes in illumination or contrast do not alter the phase value. This property makes the phase-based motion detector robust under varying lighting conditions, where traditional intensity-based methods fail. The phase reflects only the spatial structure of the image, not its absolute brightness, ensuring consistent motion detection regardless of environmental illumination.

- derive local phase equation for translational motion  
The local phase equation for translational motion is derived by applying the chain rule to the time derivative of the STFT phase, yielding a term proportional to the velocity vector and an additional residual term that accounts for non-uniform intensity changes within the window. For rigid motion, the residual term is negligible, and the phase derivative closely approximates the global phase equation, localized to the window region.

- discuss added term in local phase equation  
The added term in the local phase equation arises from non-uniform intensity variations within the window, such as edges terminating within the window or non-rigid deformations. While this term can introduce noise, it is typically small compared to the velocity-dependent term for strong, coherent motion. Its presence is mitigated by the Radon transform, which exploits the structured nature of the velocity term to distinguish true motion from noise.

### The Block Structure for Computing the Local Phase

- define Gaussian windows  
Gaussian windows are defined as two-dimensional functions with a bell-shaped envelope, centered at regular intervals across the image plane. The windows overlap to ensure complete spatial coverage and are chosen for their optimal balance between spatial and frequency localization. Their finite support ensures that the Short-Time Fourier Transform is computed over a localized region, enabling the extraction of local phase information.

- compute 2D Fourier transform of windowed video signal  
The two-dimensional Fourier transform of each windowed video block is computed using the Fast Fourier Transform algorithm, which efficiently converts the spatial domain signal into its frequency domain representation. The result is a complex-valued matrix for each block and time point, from which amplitude and phase are extracted.

- evaluate integral using 2D FFT  
The integral defining the Short-Time Fourier Transform is evaluated numerically by discretizing the image into pixels and applying the 2D FFT to each windowed block. This approach leverages the computational efficiency of the FFT, reducing the complexity of the convolution operation from O(N⁴) to O(N² log N), making real-time processing feasible.

- process each block independently  
Each windowed block is processed independently, with no communication between blocks during the computation of phase or its derivative. This independence enables massive parallelism, as each block can be assigned to a separate processing unit, allowing the entire algorithm to scale with the number of available processors.

- discuss window size and object motion detection  
The size of the Gaussian window determines the spatial resolution of motion detection. Smaller windows provide finer localization but are more susceptible to noise, while larger windows enhance robustness but may encompass multiple moving objects, complicating direction estimation. The window size is chosen to match the expected size of moving objects, ensuring optimal sensitivity and specificity.

- illustrate block structure with example  
An example of the block structure is illustrated using a 64×64 pixel image divided into overlapping 32×32 pixel blocks, each centered at intervals of six pixels. The Gaussian windows extend beyond the block boundaries, ensuring that the effective receptive field is fully captured within the FFT window. The overlapping arrangement ensures that motion at any location is detected by multiple blocks, enhancing reliability.

- describe de-noising of phase measurements  
De-noising is performed by weighting the phase derivative by the local amplitude, normalizing it by the average amplitude within the block. This suppresses phase measurements in low-energy regions where noise dominates, ensuring that only reliable phase changes contribute to motion detection.

### The Phase-Based Detector

- provide embodiment of block FFT based algorithm  
An embodiment of the phase-based motion detector is implemented as a block-based FFT algorithm that processes video frames in real time. Each frame is divided into overlapping blocks, each subjected to a Gaussian window and a 2D FFT. The phase is extracted, temporally filtered, and subjected to a Radon transform to compute a Phase Motion Indicator. Motion is detected when the indicator exceeds a threshold, and direction is determined from the angular coordinate of the maximum Radon response.

### Radon Transform on the Change of Phases

- compute Radon transform of phase derivative  
The Radon transform is applied to the phase derivative over the spatial-frequency domain of each block, integrating the phase change along lines of constant orientation. This transform collapses the two-dimensional phase derivative into a one-dimensional function of angle and radial distance, revealing the dominant motion direction as a peak in the Radon domain.

- define Radon transform  
The Radon transform of a function f(x,y) is defined as the integral of f along straight lines parameterized by distance ρ from the origin and angle θ. For the phase derivative, this transform accumulates phase changes along lines of constant frequency orientation, enhancing structured motion signals while suppressing isotropic noise.

- discuss linear structure of phase derivative  
For rigid translational motion, the phase derivative exhibits a linear structure in the frequency domain, proportional to the dot product of frequency and velocity. This structure is aligned along a single direction perpendicular to the motion vector, producing a coherent ridge in the Radon transform. This linear structure is absent in noise, allowing the Radon transform to distinguish true motion from random fluctuations.

- compute Radon transform for blocks exhibiting motion  
For each block, the Radon transform is computed over a circular domain of spatial frequencies, and the resulting function is analyzed for peaks. A strong peak indicates the presence of coherent motion, while a flat response indicates no motion or noise.

- define correction term  
The correction term accounts for the varying length of integration lines in the bounded frequency domain, ensuring that the Radon transform is normalized across all angles. This term is computed analytically based on the geometry of the circular domain and the window size.

- compute PMI  
The Phase Motion Indicator is computed as the maximum value of the normalized Radon transform over all angles, providing a scalar measure of motion strength for each block.

- discuss PMI computation  
The PMI is computed by evaluating the Radon transform at discrete angular intervals and selecting the maximum value. This value is thresholded to determine whether motion is present in the block.

- compute direction of motion  
The direction of motion is determined by the angle at which the PMI is maximized, corresponding to the orientation of the motion-induced ridge in the Radon domain.

- discuss direction of motion computation  
The direction is computed as the angle θ that maximizes the Radon transform, adjusted by a sign determination based on the polarity of the phase derivative. This ensures that the direction is unambiguously assigned, even when the phase derivative is symmetric.

- illustrate phase-motion detection algorithm  
The algorithm is illustrated as a pipeline: input video → windowing → 2D FFT → phase extraction → temporal filtering → Radon transform → PMI computation → motion detection → direction estimation.

- describe algorithm implementation  
The algorithm is implemented on a GPU using CUDA, with each block processed in parallel. The 2D FFT, phase extraction, and Radon transform are performed using optimized libraries, ensuring real-time performance even at HD resolutions.

- discuss parallel computing capabilities  
The algorithm is inherently parallel, as each block is processed independently. This allows for linear scaling with the number of processing units, enabling deployment on embedded systems, FPGAs, or multi-core CPUs.

- illustrate algorithm operation  
The algorithm operates continuously on incoming video frames, updating the motion detection output at each frame. The output is a binary motion map and a direction map, both updated in real time.

- divide algorithm into two parts  
The algorithm is divided into two parts: the first computes local phase changes, and the second detects motion using the Radon transform.

- discuss first part of algorithm  
The first part applies Gaussian windows to each block, computes the 2D FFT, extracts the phase, and applies a temporal high-pass filter to isolate motion-induced phase changes.

- apply Gaussian window  
Each video block is multiplied by a two-dimensional Gaussian window to localize the frequency analysis.

- compute local phase  
The phase of the FFT output is computed for each frequency component within the block.

- employ temporal high-pass filter  
A high-pass filter is applied to the phase time series to remove slow drifts and retain only transient changes indicative of motion.

- discuss second part of algorithm  
The second part computes the Radon transform of the phase derivative, evaluates the PMI, detects motion, and computes direction.

- evaluate PMI  
The Phase Motion Indicator is computed as the maximum of the normalized Radon transform over all angles.

- detect motion  
Motion is detected if the PMI exceeds a predefined threshold.

- compute direction of motion  
The direction is assigned as the angle corresponding to the maximum PMI.

- discuss algorithm parallelization  
The algorithm is fully parallelizable, with each block processed independently. This allows for deployment on massively parallel architectures, achieving real-time performance even at high resolutions.

- discuss extension to higher dimensions  
The method extends naturally to three or more dimensions by applying the same principles to volumetric data, such as 3D video or medical imaging sequences. The Radon transform operates over planes or hyperplanes, and the FFT is computed in higher dimensions using standard algorithms.

### Examples of Phase-Based Motion Detection

- introduce highway video  
A highway surveillance video is processed to demonstrate motion detection under high-contrast, moderate-noise conditions. Moving vehicles are clearly detected, while stationary elements such as road markings and trees remain unactivated.

- illustrate motion detection results  
The motion detection results show high PMI values over moving vehicles and low values over static background, with direction arrows aligned with vehicle motion.

- discuss aperture problem  
The algorithm exhibits the aperture problem, where motion along an edge is perceived as perpendicular to the edge orientation. This is consistent with biological motion detection and does not impair detection accuracy.

- introduce low-contrast video  
A low-contrast version of the highway video is created by reducing intensity range. Despite reduced contrast, motion detection remains robust, with PMI values retaining over 50% of their full-contrast values.

- show motion detection results on low-contrast video  
Motion detection results on the low-contrast video show consistent detection of vehicles, while traditional intensity-based methods fail to detect motion in many regions.

- introduce train station video  
A train station video with mixed lighting and pedestrian motion is processed. The algorithm detects moving individuals and trains with high accuracy.

- illustrate motion detection results  
The motion detection map highlights moving people and trains, with minimal false positives from background noise.

- introduce thermal video  
A thermal video with high background noise is processed. The detection threshold is increased to suppress noise while preserving motion signals.

- show motion detection results on thermal video  
Moving objects such as vehicles and people are detected despite high noise levels, demonstrating the algorithm’s resilience.

- introduce winterstreet video  
A nighttime winter street video with uneven illumination is processed. Motion is detected on illuminated portions of vehicles, with reduced sensitivity in shadowed regions.

- illustrate motion detection results  
Motion is detected on the upper-right side of the road where illumination is sufficient, while lower-left regions show reduced detection due to low amplitude.

- discuss noise suppression trade-off  
A higher threshold improves noise suppression but reduces sensitivity to weak motion. The algorithm balances this trade-off by normalizing phase changes with local amplitude.

- introduce motion segmentation  
Motion segmentation is performed by applying a higher threshold to isolate salient moving objects and then refining boundaries using smaller blocks.

- discuss block size reduction  
Reducing block size from 32×32 to 16×16 pixels improves boundary localization, allowing more precise segmentation of moving objects.

- illustrate motion segmentation results  
Segmentation results show clean outlines of moving vehicles and pedestrians, with minimal background contamination.

- discuss applicability to various videos  
The algorithm is successfully applied to diverse video types, including daylight, thermal, low-contrast, and noisy scenes, demonstrating broad applicability.

- discuss computer-implemented operations  
All operations are implemented as computer-executable instructions, enabling deployment on standard digital hardware.

- discuss scope of disclosure  
The disclosed method encompasses all implementations of phase-based motion detection using local phase analysis, Radon transform, and block-based parallel processing, whether implemented in software, firmware, or hardware.