Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION  

## FEDERALLY-SPONSORED RESEARCH AND DEVELOPMENT  

The United States Government has rights in this invention pursuant to funding provided by the Naval Innovative Science and Engineering Program at the Space and Naval Warfare Systems Center Pacific.  

## BACKGROUND  

Acoustic source tracking is a critical capability for various applications including environmental monitoring, surveillance, and underwater navigation. The ability to accurately locate and track sound sources in complex environments enables improved situational awareness and decision-making. Traditional approaches to acoustic source localization face significant challenges due to multipath propagation, environmental uncertainties, and computational complexity. These limitations motivate the development of advanced signal processing techniques that can overcome the shortcomings of conventional methods while maintaining computational tractability for real-time operation.  

## DETAILED DESCRIPTION OF SOME EMBODIMENTS  

The following detailed description illustrates embodiments of the invention by way of example and not by way of limitation. The term "coupled" as used herein refers to any connection, either direct or indirect, that enables functional interconnection between components. The term "connected" refers to a direct physical or electrical connection between elements. The terms "comprises" and "includes" specify the presence of stated features but do not preclude the presence of additional elements. The indefinite articles "a" and "an" refer to one or more of the specified element.  

The invention provides a sparsity-driven approach for acoustic source localization and tracking that overcomes limitations of conventional methods. Source location maps (SLMs) are generated to represent potential source positions, where only locations corresponding to actual sources contain non-zero values. This sparse representation enables efficient processing while maintaining localization accuracy.  

An iterative solver based on the proximal gradient (PG) method is developed to construct the SLMs. The PG approach provides computational efficiency while handling the non-linearities inherent in acoustic propagation models. The system addresses the critical need for accurate localization and tracking of acoustic sources, particularly in challenging underwater environments where multipath effects and environmental uncertainties degrade performance.  

Underwater source localization presents unique challenges due to complex sound propagation characteristics. The invention specifically addresses limitations of conventional matched-field processing (MFP), which suffers from sensitivity to environmental mismatch and difficulty handling multiple sources. While MFP generates ambiguity surfaces by matching measured data to modeled replicas, artifacts in these surfaces often obscure true source locations.  

Matched-field tracking (MFT) extends MFP by connecting peaks across sequential ambiguity surfaces, but this approach becomes computationally intractable for multiple sources and fails to effectively incorporate prior information. Bayesian methods offer theoretical advantages but prove impractical due to excessive computational demands from high-dimensional state spaces. Sparsity-driven Kalman-filter approaches similarly struggle with the dimensionality of typical localization grids.  

The invention overcomes these limitations through a novel sparsity-driven framework that maintains computational efficiency while improving tracking accuracy. The system employs a relevance vector machine approach within a carefully designed sparse estimation framework. A key innovation involves enforcing coherence across frequency-specific SLMs while preserving their sparse structure.  

Figure 1 illustrates the system architecture through block diagram 10, showing the interconnection of major components. Acoustic sensor array 50 comprises multiple hydrophones that collect time-series data, which is transformed to the frequency domain via short-time Fourier transform (STFT). The system models acoustic propagation through the environment to characterize the relationship between source locations and received signals.  

The spectral passive-acoustic tracking problem is formulated as estimating source locations given Fourier coefficient measurements across multiple frequencies. Signal transmission and processing components generate model-based predictions that inform the tracking algorithm. The operational environment and corresponding SLMs are visualized to demonstrate system performance.  

To address nonlinearities in acoustic propagation, the invention proposes a model that linearizes the relationship between source locations and measurements. The measurement vector yf(t) at frequency ωf and time t follows the equation yf(t) = Pf sf(t) + εf(t), where Pf contains replica vectors, sf(t) represents source coefficients, and εf(t) denotes noise.  

Group sparsity and temporal dependency are incorporated through regularization terms that promote row-wise sparsity in the coefficient matrix S(t) while encouraging temporal consistency with previous estimates. The iterative estimator for S(t) minimizes a cost function combining measurement fidelity, group sparsity promotion, and temporal smoothing terms.  

The estimation problem is formulated as a convex optimization problem after transforming complex-valued variables to real-valued representations. The structure of regression coefficient matrix S enables efficient computation, where whole rows correspond to specific grid locations. SLMs are constructed by aggregating row norms across frequencies.  

Interior point methods prove impractical for solving the optimization problem due to high dimensionality. Instead, the invention develops a specialized PG solver that exploits problem structure for computational efficiency. The PG approach rewrites the problem to enhance resilience to model mismatch through carefully designed regularization terms.  

The PG algorithm employs a majorization-minimization framework using majorizer H(S;Z) that upper bounds the original cost function. Figure 8 illustrates the majorizer construction, showing how it facilitates efficient optimization. Each PG iteration performs parallel updates across grid locations, with closed-form solutions enabled by the problem structure.  

Key steps in the PG iteration include computing gradient-descent directions for each row of S, followed by proximal operations that enforce sparsity. The proximal operator takes the form of a group soft-thresholding operation that zeros out rows below a magnitude threshold. The complete PG algorithm is summarized in pseudocode form, showing the sequence of parallelizable operations.  

Termination criteria ensure convergence to a solution with desired accuracy, while an accelerated PG variant provides faster convergence when needed. The system implements the method through modular components that can be deployed on various computing platforms. Information storage media maintain tracking history and environmental parameters, while output devices display SLMs and estimated tracks.  

Performance evaluation using the SWellEX-3 dataset demonstrates the algorithm's effectiveness. The test environment features complex bathymetry and a moving acoustic source, with replicas computed using the KRAKEN normal-mode propagation model. Comparison with conventional MFT shows the invention's superior ability to maintain continuous tracks despite environmental variability.  

For scenarios with multiple sources, the system successfully distinguishes and tracks separate trajectories. Range and depth estimates show good agreement with ground truth, though depth resolution presents greater challenges. The framework extends naturally to other acoustic environments and can incorporate spatially distributed arrays for improved localization accuracy.  

The claims define the scope of protection for this invention, covering the sparsity-driven tracking methodology, system implementations, and specific algorithmic innovations that enable robust performance in challenging acoustic environments.