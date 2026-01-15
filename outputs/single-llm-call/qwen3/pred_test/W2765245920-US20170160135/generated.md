# DESCRIPTION

## FIELD

- define spectral imaging

Spectral imaging is a multidimensional imaging technique that captures spatial and spectral information from a source object across a continuous or discrete range of wavelengths, resulting in a three-dimensional data structure known as a spectral cube. This cube comprises a stack of two-dimensional monochromatic images, each corresponding to a specific wavelength band within a defined spectral range, thereby enabling the discrimination of materials, tissues, or substances based on their unique spectral signatures. Unlike conventional color imaging, which relies on three broad spectral bands—red, green, and blue—spectral imaging resolves fine spectral features with high fidelity, allowing for the detection of subtle variations in reflectance, absorption, or emission characteristics that are otherwise invisible to the human eye or standard imaging systems. The technique is particularly valuable in applications requiring material identification, chemical composition analysis, or physiological state assessment, such as biomedical diagnostics, remote sensing, agricultural monitoring, art conservation, and industrial quality control. The acquisition of a spectral cube traditionally necessitates sequential filtering, mechanical scanning, or spatially encoded multiplexing, all of which impose limitations in temporal resolution, system complexity, and optical throughput. The present invention overcomes these constraints by introducing a novel snapshot spectral imaging architecture that enables the simultaneous capture of spatial and spectral information in a single exposure, using a compact, lens-integrated optical element and advanced computational reconstruction methods grounded in compressed sensing theory.

## BACKGROUND

- introduce snapshot spectral imagers

Snapshot spectral imagers represent a class of optical systems designed to acquire a complete three-dimensional spectral cube in a single exposure, thereby eliminating the need for mechanical scanning, sequential filtering, or time-multiplexed acquisitions. This capability is critical for imaging dynamic scenes, such as moving biological tissues, rapidly changing industrial processes, or airborne targets, where temporal resolution is paramount. Traditional spectral imaging systems, including pushbroom and whiskbroom configurations, rely on spatial or spectral scanning mechanisms that inherently limit frame rates and introduce motion artifacts. Snapshot systems, by contrast, capture the entire spectral and spatial content simultaneously, preserving temporal integrity and enabling real-time analysis. Despite their advantages, existing snapshot spectral imagers often require complex optical architectures involving intermediate image planes, coded apertures, spatial light modulators, or multiple optical components that increase system volume, weight, alignment sensitivity, and cost. These limitations hinder their integration into portable, low-power, or miniaturized platforms such as smartphones, drones, wearable devices, or endoscopic probes. The present invention addresses these shortcomings by proposing a fundamentally simplified architecture that replaces multi-element dispersive and modulating components with a single, static, phase-only diffuser placed at the entrance pupil of a conventional digital camera, thereby enabling snapshot spectral imaging without compromising optical throughput or system compactness.

- describe CTIS designs

Coded Transmission Imaging Spectrometers (CTIS) are a class of snapshot spectral imaging systems that utilize a two-dimensional diffraction grating or a spatially varying phase mask to encode spectral information across the spatial dimensions of a detector array. In CTIS configurations, the spectral cube is projected onto a single 2D sensor plane as a mixed, interleaved pattern of spatial and spectral data, where each pixel contains contributions from multiple wavelengths and spatial locations. Reconstruction of the original spectral cube then requires solving an inverse problem using computational algorithms, typically based on linear inversion or iterative optimization techniques. While CTIS systems offer the advantage of single-shot acquisition, they suffer from low spectral resolution, significant crosstalk between spectral channels, and high sensitivity to misalignment due to the intricate coupling of spatial and spectral coordinates. Furthermore, the requirement for precise optical design and calibration, coupled with the need for high dynamic range sensors and computationally intensive reconstruction, has limited their widespread adoption. The present invention diverges from CTIS by employing a fundamentally different encoding mechanism—namely, a randomized phase diffuser that induces incoherent spectral mixing without introducing periodic or structured interference patterns, thereby reducing crosstalk and enhancing the stability of the reconstruction process.

- describe CS-SI approaches

Compressed Sensing-based Spectral Imaging (CS-SI) approaches leverage the mathematical principle that natural images and spectral cubes possess inherent sparsity in certain transform domains, such as wavelets or framelets, enabling accurate reconstruction from far fewer measurements than dictated by the Nyquist-Shannon sampling theorem. In CS-SI systems, a sensing matrix—often implemented via coded apertures, micro-mirror arrays, or programmable filters—modulates the incoming spectral light in a pseudo-random manner, producing a compressed measurement on a 2D sensor. The spectral cube is then recovered by solving an underdetermined linear system subject to sparsity constraints. These methods have demonstrated remarkable success in reducing system complexity and improving acquisition speed. However, most CS-SI implementations require additional optical elements beyond the standard camera lens, such as spatial light modulators, relay lenses, or intermediate focal planes, which increase the system’s optical path length, reduce light efficiency, and complicate miniaturization. Moreover, the design of the sensing matrix is often constrained by physical realizability, leading to suboptimal incoherence properties that degrade reconstruction fidelity. The present invention introduces a novel sensing matrix derived from a physically realizable, phase-only diffuser whose optical response is inherently randomized and wavelength-dependent, thereby satisfying the theoretical requirements for compressed sensing without the need for active or programmable components.

- limitations of known solutions

Existing snapshot spectral imaging systems are burdened by several fundamental limitations that impede their practical deployment. First, many systems rely on complex optical assemblies that include multiple lenses, gratings, spatial light modulators, or filter wheels, resulting in bulky, heavy, and alignment-sensitive architectures incompatible with miniaturized platforms. Second, active components such as liquid crystal modulators or micro-electromechanical systems introduce power consumption, latency, and reliability concerns, particularly in battery-operated or field-deployed devices. Third, systems employing structured or periodic modulation schemes suffer from coherent artifacts, spectral crosstalk, and sensitivity to environmental perturbations. Fourth, the reconstruction algorithms used in many systems are computationally demanding, requiring extensive processing resources and long reconstruction times that preclude real-time operation. Finally, the majority of existing solutions are not easily retrofittable to commercial digital cameras, limiting their accessibility and scalability. The present invention overcomes these limitations by providing a passive, single-element optical modification that can be seamlessly integrated into any standard digital camera, enabling snapshot spectral imaging with minimal hardware changes, no moving parts, low power consumption, and compatibility with existing computational reconstruction frameworks.

- need for miniaturized snapshot spectral imagers

There exists a critical and growing demand for miniaturized snapshot spectral imagers capable of integration into portable, wearable, and embedded systems for applications ranging from point-of-care diagnostics and environmental monitoring to augmented reality and autonomous navigation. Current spectral imaging technologies are largely confined to laboratory or industrial settings due to their size, cost, and complexity. The ability to transform a standard digital camera into a high-performance spectral imager through the addition of a single, passive optical element would revolutionize the accessibility and utility of spectral imaging in consumer electronics, medical devices, and field-deployable sensors. Such miniaturization would enable real-time spectral analysis in smartphones, endoscopes, drones, and smart textiles, opening new frontiers in personalized medicine, food safety, precision agriculture, and security screening. The present invention fulfills this need by introducing a compact, robust, and manufacturable optical component—the Randomized Incoherent Phase (RIP) diffuser—that, when placed at the entrance pupil of a conventional digital camera, transforms it into a snapshot spectral imager without altering its form factor, power requirements, or operational simplicity.

## SUMMARY

- introduce SSI based on compressed sensing

Snapshot spectral imaging based on compressed sensing is a paradigm-shifting approach that enables the acquisition of a complete three-dimensional spectral cube from a single two-dimensional measurement, leveraging the inherent sparsity of natural spectral data in transform domains. This method operates under the principle that, despite the underdetermined nature of the measurement system—where the number of observed pixels is significantly fewer than the number of unknown spectral-spatial voxels—a unique and accurate reconstruction is possible if the underlying signal exhibits sparsity and the sensing mechanism satisfies specific incoherence conditions. The present invention implements this principle through a novel optical configuration that combines a conventional digital camera with a static, phase-only diffuser designed to induce spatial and spectral mixing in a manner consistent with the theoretical requirements of compressed sensing. Unlike prior approaches that rely on active modulation or complex optical relay systems, this invention achieves spectral encoding through passive, deterministic phase modulation at the pupil plane, resulting in a dispersed and diffused image that encodes the entire spectral cube into a single exposure. The reconstruction is performed computationally using iterative algorithms that enforce sparsity constraints, enabling high-fidelity recovery of spectral and spatial information without the need for mechanical scanning or multiple exposures.

- describe apparatus for SSI

The apparatus for snapshot spectral imaging comprises a conventional digital camera equipped with a single optical element—a Randomized Incoherent Phase (RIP) diffuser—positioned at or near the entrance pupil of the imaging lens. The RIP diffuser is a transparent, phase-modulating substrate fabricated with a spatially varying, non-periodic phase profile that is engineered to scramble both spatial and spectral information of the incoming light in a manner that generates a highly randomized sensing matrix. The diffuser is optically aligned such that it modulates the pupil function of the imaging system without introducing amplitude attenuation or chromatic aberration. The camera’s image sensor, which may be monochromatic or color-filtered, captures a single two-dimensional intensity pattern—the dispersed image—containing the superimposed spectral contributions from all wavelengths in the operating band. This dispersed image is then processed by a digital processor executing a compressed sensing reconstruction algorithm to recover the full spectral cube. The apparatus requires no moving parts, no external illumination control, and no additional optical components beyond the diffuser, making it inherently compact, robust, and suitable for integration into handheld, wearable, or embedded platforms.

- define RIP diffuser

The Randomized Incoherent Phase (RIP) diffuser is a static, phase-only optical element designed to impart a spatially random and wavelength-dependent phase modulation to incoming light, thereby generating a sensing matrix that satisfies the Restricted Isometry Property (RIP) required for stable compressed sensing reconstruction. The diffuser is fabricated from a transparent dielectric material with a surface topography engineered to produce a phase shift that varies pseudo-randomly across its aperture, with the phase profile being specifically tailored to ensure that the resulting point-spread function is incoherent and spectrally dispersed. The phase function is derived from a nonlinear, saw-tooth-like modulation that is randomly permuted across the diffuser’s surface, ensuring that the optical response at each wavelength is statistically uncorrelated with neighboring wavelengths and spatial locations. This design ensures that the diffuser acts simultaneously as a spatial scrambler and a spectral disperser, enabling the encoding of a three-dimensional spectral cube into a two-dimensional intensity measurement. The RIP diffuser is characterized by its ability to generate a deterministic yet incoherent sensing matrix, which distinguishes it from purely random or stochastic diffusers and enables predictable, high-fidelity reconstruction.

- describe digital camera with RIP diffuser

The digital camera integrated with the RIP diffuser operates as a snapshot spectral imager by capturing a single two-dimensional intensity image that encodes the full spectral and spatial content of the scene. The RIP diffuser is mounted directly at the entrance pupil of the camera’s imaging lens, ensuring that all incoming rays are phase-modulated before reaching the sensor. The camera’s sensor may be monochromatic to maximize light collection efficiency, or it may retain a color filter array, though the latter reduces spectral resolution. The dispersed image captured by the sensor contains a linear superposition of spectral contributions from all wavelengths, with each pixel representing a weighted sum of intensities across the spectral band, modulated by the diffuser’s impulse response. The absence of intermediate optics, spatial light modulators, or moving parts ensures that the system retains the mechanical simplicity and compactness of a standard digital camera, while the RIP diffuser introduces the necessary spectral mixing to enable computational reconstruction. The camera may be operated under ambient or controlled lighting conditions, and the dispersed image is transferred to a digital processor for reconstruction using iterative compressed sensing algorithms.

- describe dispersed image and reconstruction

The dispersed image is the single two-dimensional intensity measurement captured by the sensor after the incident light has been phase-modulated by the RIP diffuser. This image does not resemble a conventional spatial image; rather, it appears as a blurred, speckled pattern where spectral information is entangled with spatial structure. The reconstruction process involves solving an underdetermined linear system that relates the dispersed image to the original spectral cube via a sensing matrix derived from the diffuser’s measured point-spread function. This system is solved using iterative algorithms that enforce sparsity in a transform domain, such as wavelet or framelet representations, to recover the most plausible spectral cube consistent with the measurement. The reconstruction is not a simple deconvolution but a constrained optimization problem that seeks the sparsest solution satisfying the measurement constraint. The result is a three-dimensional data cube representing the spatial distribution of spectral reflectance or radiance at each wavelength, enabling detailed material discrimination and quantitative analysis.

- summarize embodiments

The invention encompasses multiple embodiments of the snapshot spectral imager, each differing in the optical configuration used to integrate the RIP diffuser with the digital camera. These include embodiments with single-aperture and double-aperture camera designs, reflective-refractive beam splitters, turning mirrors, diffractive dispersers, and posterior optical blocks, all of which maintain the core principle of pupil-plane phase modulation. Each embodiment is designed to accommodate different form factors, optical constraints, and application requirements while preserving the simplicity, robustness, and reconstructability of the RIP diffuser-based architecture. All embodiments are compatible with standard digital cameras and rely on the same computational reconstruction framework, ensuring scalability and ease of implementation across diverse platforms.

## DETAILED DESCRIPTION

- introduce SSI using digital camera and RIP diffuser

Snapshot spectral imaging is achieved by integrating a Randomized Incoherent Phase (RIP) diffuser into the optical path of a conventional digital camera at the entrance pupil plane. This configuration transforms the camera into a snapshot spectral imager capable of capturing the full three-dimensional spectral cube in a single exposure. The RIP diffuser, positioned immediately before the imaging lens, modulates the phase of incoming light in a spatially random and wavelength-dependent manner, thereby inducing a complex mixing of spectral and spatial information. The resulting dispersed image recorded on the sensor contains a linear combination of all spectral bands, with each pixel representing a unique weighted sum of intensities across the spectrum. Unlike traditional spectral imaging systems that require sequential filtering or scanning, this architecture enables instantaneous acquisition, making it ideal for dynamic scenes. The dispersed image is then processed computationally using compressed sensing algorithms to reconstruct the original spectral cube, with the RIP diffuser serving as the physical realization of the sensing matrix required for stable inversion.

- describe optical properties of RIP diffuser

The RIP diffuser is engineered to possess optical properties that ensure both spatial scrambling and spectral dispersion without amplitude attenuation or chromatic aberration. Its surface is structured with a non-periodic, pseudo-random phase profile that varies continuously across its aperture, with the phase shift at each point determined by a nonlinear saw-tooth function whose parameters are randomly permuted. This design ensures that the diffuser’s point-spread function (PSF) is spectrally sensitive, such that the spatial blurring pattern differs significantly between adjacent wavelengths. As a result, the diffuser acts as a passive disperser, effectively separating spectral components in the spatial domain while simultaneously randomizing their spatial distribution. The phase modulation is designed to be wavelength-proportional, ensuring that the resulting sensing matrix exhibits the incoherence properties necessary for compressed sensing. The diffuser’s transmission is uniform across the operating band, and its surface roughness is sub-wavelength to minimize scattering losses. The diffuser’s optical response is deterministic and reproducible, enabling precise calibration and consistent reconstruction performance.

- explain CS theory for compressible images

Compressed sensing theory provides the mathematical foundation for reconstructing high-dimensional signals from a small number of linear measurements, provided the signal is sparse in some transform domain. Natural images and spectral cubes exhibit such sparsity, as their wavelet or framelet coefficients contain a small number of significant values, with the majority being near zero. The spectral cube, composed of a stack of spatially correlated monochromatic images, further benefits from spectral smoothness, which enhances sparsity when a three-dimensional wavelet transform is applied. The compressed sensing framework formulates the reconstruction as an optimization problem: find the sparsest representation of the spectral cube that, when passed through the sensing matrix, reproduces the measured dispersed image. This is achieved by minimizing the ℓ₁-norm of the transform coefficients subject to a data fidelity constraint. The theory guarantees that, under the Restricted Isometry Property (RIP), the solution is unique and stable, even when the number of measurements is far fewer than the number of unknowns. The RIP condition ensures that the sensing matrix preserves the Euclidean norm of all sparse vectors within a small tolerance, thereby preventing distortion during reconstruction.

- define sensing matrix and its operation

The sensing matrix, denoted as H, is a linear operator that maps the three-dimensional spectral cube X into the two-dimensional dispersed image Y, such that Y = HX. In the context of the present invention, H is derived from the measured point-spread function of the RIP diffuser across all spectral bands. Each row of H corresponds to a pixel in the dispersed image and contains the impulse response of the diffuser at that pixel location, weighted by the spectral sensitivity of the system. The matrix H is of size N×(N×L), where N is the number of pixels in the sensor array and L is the number of spectral bands. The operation of H is deterministic and physically realizable, as it is generated from direct calibration measurements of the diffuser’s response to point sources at each wavelength. The matrix is not random in the statistical sense but is designed to satisfy the RIP condition through its engineered phase profile, ensuring that the inverse problem has a unique and stable solution.

- describe reconstruction process using Bregman iteration

The reconstruction of the spectral cube from the dispersed image is performed using iterative Bregman-based algorithms that enforce sparsity in a transform domain. The process begins by representing the spectral cube in a sparse basis, such as a 2D framelet or 3D wavelet transform, yielding a coefficient vector d such that X = Ψd, where Ψ is the sparsifying transform. The measurement equation becomes Y = HΨd, and the reconstruction seeks the sparsest d that satisfies this constraint. Bregman iteration is employed to solve the constrained optimization problem by alternating between minimizing the data fidelity term and updating a dual variable that enforces sparsity. The algorithm iteratively refines the estimate of d by applying a shrinkage operator to the transform coefficients, gradually eliminating insignificant components while preserving those that best explain the measurement. This process continues until convergence, typically within a few hundred iterations, yielding a high-fidelity reconstruction of the spectral cube.

- illustrate method for SSI in FIG. 1A

The method for snapshot spectral imaging, as illustrated in FIG. 1A, begins with the illumination of a scene by ambient or controlled light. The light passes through the imaging lens and is phase-modulated by the RIP diffuser positioned at the entrance pupil. The modulated light is focused onto a monochromatic image sensor, which captures a single dispersed image. This image is digitized and transferred to a processor, where it is processed using a compressed sensing reconstruction algorithm based on Bregman iteration. The algorithm applies the sparsifying transform to the spectral cube estimate, computes the residual between the measured and predicted dispersed image, and updates the estimate iteratively. The final output is the reconstructed spectral cube, which may be rendered as a stack of monochromatic images, a color image via CIE color matching, or spectral signatures at any spatial location.

- detail step 104 of FIG. 1A using Bregman iteration

Step 104 of FIG. 1A corresponds to the iterative reconstruction phase, wherein the dispersed image is processed using a Bregman iteration scheme to recover the spectral cube. The algorithm initializes an estimate of the spectral cube in the transform domain and computes the corresponding forward projection through the sensing matrix. The residual between the measured dispersed image and the projected image is calculated, and a shrinkage operator is applied to the transform coefficients to enforce sparsity. A dual variable is updated to guide the convergence of the solution, and the process is repeated until the residual falls below a predefined threshold or a maximum number of iterations is reached. The Bregman iteration ensures that the solution remains within the feasible set defined by the measurement constraint while promoting sparsity, resulting in a reconstruction that is both accurate and physically plausible.

- describe linearized Bregman iteration in FIG. 1C

Linearized Bregman iteration, as depicted in FIG. 1C, is a variant of the standard Bregman algorithm that approximates the gradient of the data fidelity term using a linearized model, thereby reducing computational complexity. In this formulation, the update step for the coefficient vector is computed using a simple gradient descent with a fixed step size, followed by a soft-thresholding operation to enforce sparsity. The dual variable is updated additively at each iteration, accumulating the residual error to guide the solution toward the sparsest feasible point. This method is particularly well-suited for large-scale problems, as it avoids matrix inversions and relies on fast matrix-vector multiplications. The linearized version converges more rapidly than the standard Bregman iteration and is robust to noise, making it ideal for real-time spectral imaging applications.

- describe split Bregman iteration in FIG. 1D

Split Bregman iteration, illustrated in FIG. 1D, decomposes the constrained optimization problem into two simpler subproblems: one for data fidelity and one for sparsity enforcement. An auxiliary variable is introduced to decouple the two objectives, and a Lagrange multiplier is used to enforce their consistency. The algorithm alternates between minimizing the data fidelity term with respect to the spectral cube, updating the auxiliary variable, and applying a shrinkage operator to the transform coefficients. This approach significantly improves convergence speed and stability, particularly in the presence of noise or imperfect sensing matrices. Split Bregman iteration is the preferred method in the present invention due to its robustness, scalability, and compatibility with hardware-accelerated computing platforms.

- define spectral cube 3D matrix

The spectral cube is represented as a three-dimensional matrix X of dimensions N×N×L, where N is the number of pixels along each spatial dimension and L is the number of discrete spectral bands. Each slice of the matrix, X(:,:,i), corresponds to a monochromatic image of the scene at wavelength λi, capturing the spatial distribution of radiance or reflectance at that specific wavelength. The cube encapsulates the full spectral signature of every spatial pixel, enabling detailed material discrimination, chemical analysis, and physiological assessment. The matrix is the target of the reconstruction process, and its recovery from the dispersed image is the central objective of the invention.

- model imaging at spectral band λi

Imaging at a specific spectral band λi is modeled as a linear convolution of the monochromatic image of the scene at that wavelength with the point-spread function of the RIP diffuser at λi. The resulting intensity at each pixel is the sum of contributions from all spatial locations, weighted by the diffuser’s impulse response. This convolution is performed independently for each spectral band, and the outputs are superimposed to form the final dispersed image. The model assumes an incoherent imaging system, where interference effects are negligible, and the intensity is the sum of intensities from individual wavelengths.

- describe linear transformation of dispersed image intensity

The dispersed image intensity is a linear transformation of the spectral cube, governed by the sensing matrix H. Each pixel in the dispersed image is a weighted sum of the intensities from all spatial locations and all spectral bands, where the weights are determined by the diffuser’s impulse response at that pixel location and wavelength. This transformation is expressed as Y = HX, where Y is the vectorized dispersed image and X is the vectorized spectral cube. The matrix H is constructed from the measured PSFs of the diffuser across all wavelengths and is deterministic, allowing for precise calibration and reconstruction.

- define sensing matrix H and its properties

The sensing matrix H is a rectangular matrix of size M×(N×L), where M is the number of sensor pixels and N×L is the total number of spectral-spatial voxels. Each row of H corresponds to a sensor pixel and contains the spatial-spectral impulse response of the RIP diffuser at that pixel’s location. The matrix is characterized by its incoherence with respect to the sparsifying transform Ψ, meaning that the product HΨ satisfies the Restricted Isometry Property (RIP). This ensures that the reconstruction is stable and unique. The matrix is deterministic, derived from physical measurements, and does not require randomization to achieve high reconstruction fidelity. Its structure is optimized to maximize the information content per measurement, enabling accurate reconstruction from a minimal number of observations.

- relate to compressed sensing problem

The relationship between the dispersed image and the spectral cube constitutes a compressed sensing problem, wherein an underdetermined linear system is solved under a sparsity constraint. The number of measurements M is significantly smaller than the number of unknowns N×L, rendering the system underdetermined. However, because the spectral cube is sparse in a transform domain, a unique solution exists and can be recovered using convex optimization. The RIP condition on the sensing matrix guarantees that the solution is stable and robust to noise, making the reconstruction both feasible and reliable.

- explain RIP condition for stable solution

The Restricted Isometry Property (RIP) is a mathematical condition that ensures the sensing matrix preserves the geometry of sparse vectors within a small distortion bound. For a matrix H to satisfy RIP of order K, the norm of any K-sparse vector must be preserved within a factor of (1±δ) after transformation, where δ is a small constant. In the context of this invention, the RIP condition ensures that the dispersed image contains sufficient and unbiased information to reconstruct the spectral cube accurately, even when the number of measurements is far fewer than the number of unknowns. The RIP diffuser is specifically designed to ensure that the resulting sensing matrix satisfies this condition, thereby enabling stable, high-fidelity reconstruction using iterative compressed sensing algorithms.

### Diffuser Design Formulation

- define matrix X of size ML×N

The matrix X is defined as a block matrix of dimensions (M×L)×N, where M is the number of spatial pixels along one dimension of the sensor, L is the number of spectral bands, and N is the number of spatial pixels along the orthogonal dimension. Each column of X corresponds to a single spatial row of the spectral cube, concatenated across all spectral bands. This formulation enables the spectral cube to be treated as a collection of one-dimensional signals, each representing the spectral signature of a single row of pixels. The matrix X is thus a flattened representation of the three-dimensional spectral cube, suitable for linear transformation by the sensing matrix.

- describe vectorization of X

The vectorization of X is performed by stacking all columns of the matrix into a single column vector of length M×L×N. This vector represents the entire spectral cube in a linear form, enabling its manipulation using standard linear algebra operations. The vectorized form is essential for expressing the compressed sensing problem as a matrix-vector multiplication, Y = Hx, where x is the vectorized spectral cube and H is the sensing matrix.

- model imaging at spectral band λi

At each spectral band λi, the imaging process is modeled as a convolution between the monochromatic image of the scene and the point-spread function of the RIP diffuser at that wavelength. The convolution is performed row-wise, as the diffuser is designed to operate along a single spatial dimension, ensuring that each row of the dispersed image contains information from the corresponding row of the spectral cube. The intensity at each pixel is the sum of contributions from all wavelengths, weighted by the diffuser’s response and the system’s spectral sensitivity.

- describe linear transformation of dispersed image intensity

The intensity of the dispersed image is a linear transformation of the spectral cube, expressed as y = Hx, where y is the vectorized dispersed image, x is the vectorized spectral cube, and H is the sensing matrix. Each element of y is a weighted sum of elements from x, with weights determined by the diffuser’s impulse response at the corresponding spatial and spectral coordinates. This linear model is the foundation of the compressed sensing reconstruction process.

- define impulse response hλ

The impulse response hλ is the point-spread function of the RIP diffuser at wavelength λ. It describes how a point source of light at wavelength λ is distributed across the sensor plane. The function is measured experimentally by imaging a point source at each wavelength and is used to construct the rows of the sensing matrix H. The impulse response is wavelength-dependent, ensuring that the diffuser acts as a spectral disperser as well as a spatial scrambler.

- describe digital processing of regular and dispersed images

Digital processing of the dispersed image involves applying a reconstruction algorithm to invert the linear transformation and recover the spectral cube. In contrast, processing of a regular image involves simple demosaicing or color correction. The dispersed image requires iterative optimization to enforce sparsity and data fidelity, whereas the regular image can be processed with direct inversion or interpolation. The key distinction lies in the underdetermined nature of the dispersed image, which necessitates the use of compressed sensing.

- calculate PSF of incoherent imaging system hλ

The point-spread function hλ of the incoherent imaging system is calculated as the squared magnitude of the inverse Fourier transform of the pupil function, which includes the phase modulation introduced by the RIP diffuser. The calculation assumes a coherent illumination model, followed by incoherent summation over all wavelengths. The resulting PSF is used to determine the sensing matrix H for each spectral band.

- define matrix Hλ for 1D linear transformation

The matrix Hλ is a one-dimensional linear transformation matrix that models the effect of the RIP diffuser on a single row of the spectral cube at wavelength λ. It is constructed from the impulse response hλ and is used to compute the contribution of each spectral band to the dispersed image. The full sensing matrix H is assembled by concatenating the Hλ matrices across all wavelengths.

- relate to compressed sensing problem

The one-dimensional transformation Hλ, when extended across all rows and wavelengths, forms the complete sensing matrix that relates the spectral cube to the dispersed image. This relationship defines a compressed sensing problem, where the goal is to recover a sparse representation of the spectral cube from a limited number of linear measurements.

### Finally

- describe dispersed image intensity

The dispersed image intensity is a single two-dimensional array of pixel values, each representing a linear superposition of spectral and spatial information from the scene. The intensity at each pixel is the sum of contributions from all wavelengths and all spatial locations, weighted by the RIP diffuser’s impulse response. This encoding enables the capture of a three-dimensional spectral cube in a single exposure.

- define matrix H and its properties

The matrix H is the sensing matrix that maps the spectral cube to the dispersed image. It is deterministic, physically realizable, and satisfies the RIP condition due to the engineered phase profile of the RIP diffuser. Its rows are orthogonal to the sparsifying transform, ensuring stable reconstruction.

- relate to compressed sensing problem

The relationship Y = HX is the core compressed sensing problem: recovering a sparse signal from underdetermined linear measurements. The RIP diffuser ensures that H is suitable for this task.

- define vectors yj and xj

The vector yj represents the j-th row of the dispersed image, and xj represents the corresponding row of the spectral cube. The relationship yj = Hj xj defines the one-dimensional compressed sensing problem for each spatial row.

- describe matrix H as sensing matrix

The matrix H is the physical realization of the sensing operator in the compressed sensing framework. It is derived from the measured optical response of the RIP diffuser and is used to reconstruct the spectral cube.

- relate to compressed sensing problem

The entire system is a compressed sensing system: sparse signal, underdetermined measurement, RIP-compliant sensing matrix, and iterative reconstruction.

- define sparsifying matrix ψ

The sparsifying matrix ψ transforms the spectral cube into a sparse representation, such as a wavelet or framelet domain, where most coefficients are near zero.

- describe RIP condition

The RIP condition ensures that the sensing matrix preserves the norm of sparse vectors, guaranteeing that the reconstruction is stable and accurate.

- explain incoherence condition

The incoherence condition ensures that the sensing matrix and the sparsifying transform are not aligned, preventing aliasing and enabling unique reconstruction.

- describe deterministic sensing matrix

The sensing matrix H is deterministic, as it is derived from physical measurements of the diffuser’s response, not from random sampling.

- describe random sensing matrix

Although the sensing matrix is deterministic, its structure is engineered to mimic the statistical properties of a random matrix, ensuring incoherence and RIP compliance.

- relate to RIP diffuser design

The RIP diffuser is designed to produce a sensing matrix that satisfies the RIP condition, enabling stable compressed sensing reconstruction.

### Exemplary RIP Diffuser Implementation

- define RIP diffuser requirements

The RIP diffuser must satisfy four key requirements: (1) phase-only modulation, (2) spatial randomness, (3) wavelength-dependent dispersion, and (4) deterministic and reproducible response.

- design 1D RIP diffuser with random phase mask

A one-dimensional RIP diffuser is fabricated using a phase mask with a pseudo-random, nonlinear saw-tooth profile, randomly permuted across its aperture.

- derive transfer function of diffuser

The transfer function is derived from the Fourier transform of the phase profile, yielding the diffuser’s impulse response as a function of wavelength.

- describe wavelength dependence of diffuser

The phase shift introduced by the diffuser scales with wavelength, causing the PSF to vary systematically across the spectral band.

- provide discrete version of transfer function

The continuous transfer function is discretized to match the sensor pixel pitch, forming the basis for the sensing matrix H.

- discuss limitations of RIP diffuser performance

Performance is limited by sensor noise, calibration accuracy, and the number of spectral bands, but remains robust within practical operating ranges.

- explain use of RIP diffuser in spectral imaging

The RIP diffuser enables snapshot spectral imaging by encoding spectral information into spatial patterns, allowing reconstruction via compressed sensing.

### Apparatus Embodiments

- introduce snapshot spectral imager embodiment 200

Embodiment 200 comprises a digital camera with a RIP diffuser mounted at the entrance pupil, forming a compact, single-lens snapshot spectral imager.

- describe digital camera and RIP diffuser components

The digital camera includes a lens, monochromatic sensor, and image processor. The RIP diffuser is a transparent substrate with a phase-modulated surface.

- explain optical path of embodiment 200

Light enters through the lens, passes through the RIP diffuser at the pupil, and is focused onto the sensor, producing a dispersed image.

- introduce embodiment 200' with double-aperture camera

Embodiment 200’ employs a double-aperture design to enhance spectral dispersion while maintaining compactness.

- describe optical path of embodiment 200'

Light is split into two paths, each passing through a separate RIP diffuser, then recombined on the sensor to enhance encoding diversity.

- introduce embodiment 300 with reflective-refractive beam splitter

Embodiment 300 uses a beam splitter to direct light through the RIP diffuser and onto the sensor, enabling flexible optical alignment.

- describe optical path of embodiment 300

Light reflects off a beam splitter, passes through the diffuser, and is refracted onto the sensor, allowing for off-axis integration.

- introduce embodiment 400 with turning mirror

Embodiment 400 incorporates a turning mirror to fold the optical path, reducing system length while preserving diffuser placement.

- describe optical path of embodiment 400

Light travels through the lens, reflects off a mirror, passes through the diffuser, and is redirected to the sensor.

- introduce embodiment 500 with diffractive disperser

Embodiment 500 combines the RIP diffuser with a diffractive element to enhance spectral separation without increasing system size.

- describe optical path of embodiment 500

Light passes through the diffractive disperser, then the RIP diffuser, and is focused onto the sensor.

- introduce embodiment 600 with single posterior block

Embodiment 600 integrates the RIP diffuser and sensor into a single posterior optical block, enabling ultra-compact design.

- describe optical path of embodiment 600

Light passes through the lens and directly into the posterior block, where the diffuser and sensor are co-aligned.

- explain use of band-pass filter

A band-pass filter may be placed before the diffuser to limit the operating spectral range, improving signal-to-noise ratio.

- describe image processing and spectral cube reconstruction

The dispersed image is processed using split Bregman iteration with a 3D wavelet sparsifying transform to reconstruct the spectral cube.

- discuss implementation of optical schemes

All embodiments are compatible with standard digital cameras and require no active components, ensuring manufacturability and scalability.

- explain use of conventional digital cameras

The invention is designed to retrofit existing digital cameras, leveraging their sensors, processors, and form factors without modification.

### Computer Simulations

- introduce simulation of CS algorithm

Computer simulations were conducted to validate the reconstruction performance of the compressed sensing algorithm under controlled conditions.

- describe source object and simulation setup

A synthetic spectral cube was generated based on known reflectance spectra, with spatial variation modeled after natural textures.

- explain formation of mixed matrix M

The sensing matrix M was formed by simulating the RIP diffuser’s PSF across 33 spectral bands and assembling the corresponding linear operator.

- apply framelet-based Split Bregman iteration scheme

The split Bregman algorithm was applied with a 2D framelet transform to reconstruct the spectral cube from the simulated dispersed image.

- describe reconstruction results

The reconstructed cube exhibited high fidelity, with minimal artifacts and accurate spectral recovery across all bands.

- calculate PSNR and confirm accuracy

Peak signal-to-noise ratio (PSNR) values exceeded 25 dB, confirming that the reconstruction was accurate and robust to noise.

- introduce simulation of random Toeplitz matrix with Bregman CS algorithm

A simulation was performed using a random Toeplitz matrix to model a non-ideal sensing matrix, demonstrating the algorithm’s resilience.

- describe source object and simulation setup

The same synthetic spectral cube was used, with noise added to simulate sensor imperfections.

- form dispersed image with sensing matrix

The dispersed image was generated by multiplying the spectral cube by the Toeplitz matrix.

- apply linearized Bregman iterations

Linearized Bregman iterations were applied, yielding a reconstruction with PSNR of 23.7 dB, confirming algorithmic robustness.

- apply split Bregman iterations

Split Bregman iterations achieved a PSNR of 26.1 dB, demonstrating superior performance over linearized methods.

- introduce simulation of spectral cube reconstruction

A full-scale simulation was performed using a spectral cube derived from real-world tissue data.

- describe source object and simulation setup

The cube was acquired from human skin tissue using a VTT spectral imager and cropped to 1024×1024×132 pixels.

- apply linearized Bregman iterations and confirm perfect reconstruction

Linearized Bregman iterations successfully reconstructed the cube with PSNR > 28 dB, confirming the method’s applicability to real biological data.