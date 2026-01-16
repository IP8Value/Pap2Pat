Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION  

## FEDERALLY-SPONSORED RESEARCH AND DEVELOPMENT  

The invention described herein was made with government support under the Naval Innovative Science and Engineering Program at the Space and Naval Warfare Systems Center Pacific. The government has certain rights in the invention.  

## BACKGROUND  

Localization and tracking of acoustic sources underwater presents significant challenges due to the complex interactions of sound propagation in aquatic environments. Conventional passive sonar systems utilize hydrophone arrays to detect acoustic signals and estimate source locations. However, shallow-water environments introduce multipath propagation effects that create constructive and destructive acoustic interference at the hydrophones, complicating the localization process.  

Traditional matched-field processing (MFP) techniques rely on acoustic propagation models to predict pressure measurements (replicas) at hydrophones for potential source locations on a predefined grid. These techniques match predicted replicas against actual measurements to generate ambiguity surfaces indicating probable source locations. While effective in some scenarios, MFP suffers from several limitations including sensitivity to environmental model mismatch, difficulty handling multiple sources, and computational complexity that prevents real-time operation.  

Matched-field tracking (MFT) extends MFP by constructing sequences of ambiguity surfaces to track source trajectories. However, MFT approaches suffer from combinatorial growth in possible tracks, requiring restrictive kinematic assumptions and becoming computationally intractable for multiple sources. Bayesian tracking methods have been proposed but remain impractical for real-time applications due to their computational demands.  

There exists a need for improved underwater acoustic source tracking systems that can:  
1) Handle multiple sources simultaneously  
2) Operate effectively in real-time  
3) Maintain accuracy despite environmental model mismatch  
4) Reduce computational complexity compared to existing approaches  

## DETAILED DESCRIPTION OF SOME EMBODIMENTS  

The present invention provides a novel sparsity-driven framework for broadband underwater acoustic source localization and tracking that addresses the limitations of prior systems. The disclosed method generates source localization maps (SLMs) that coherently process broadband acoustic measurements while enforcing sparsity constraints and temporal consistency across estimates.  

At each time instant t, the system processes acoustic measurements y_f(t) collected by a hydrophone array across multiple frequencies {ω_f}^F_{f=1}. The measurements are modeled as:  

y_f(t) = Σ_{k=1}^K s_{k,f}(t)p_{k,f} + ε_f(t)  

where p_{k,f} represents the normalized replica vector for a source at location r_k(t) transmitting at frequency ω_f, s_{k,f}(t) is the corresponding source spectrum coefficient, and ε_f(t) denotes additive noise.  

To enable practical implementation, the invention introduces a grid G = {r_g}^G_{g=1} of tentative source locations. The measurement model then becomes:  

y_f(t) = Σ_{g=1}^G s_{g,f}(t)p_{g,f} + ε_f(t)  

where most s_{g,f}(t) coefficients are zero except those corresponding to actual source locations. The complete broadband source localization problem is formulated as estimating the matrix S(t) = [s_{g,f}(t)] that is group-sparse across rows (locations) while maintaining temporal consistency with prior estimates S(t-1).  

The core innovation involves an iterative proximal gradient (PG) solver that efficiently estimates S(t) by solving:  

Ŝ(t) = argmin_S Σ_{f=1}^F (1/2)||y_f(t) - P_f s_f||^2_2 + (μ/2)Σ_{g=1}^G ||v_g||_2 + (λ/2)Σ_{f=1}^F ||s_f - ŝ_f(t-1)||^2_2  

where P_f = [p_{1,f},...,p_{G,f}] contains replicas for frequency ω_f, v_g combines real and imaginary parts of location g's coefficients across frequencies, and λ, μ are tuning parameters controlling temporal consistency and sparsity respectively.  

The PG solver operates by iteratively computing:  

1) Gradient descent steps for each frequency component:  
w_g^{[i-1]}(t) = v_g^{[i-1]}(t) - (1/L_h)∇h(v_g^{[i-1]}(t))  

where L_h is the Lipschitz constant and ∇h is the gradient of the differentiable portion of the cost function.  

2) Proximal updates enforcing group sparsity:  
v_g^{[i]}(t) = (1 - μ/(L_h||w_g^{[i-1]}(t)||_2))_+ w_g^{[i-1]}(t)  

This closed-form update simultaneously zeros out insignificant locations while preserving and refining estimates for probable source positions.  

Key advantages of the disclosed method include:  

1) **Broadband Processing**: Coherently combines information across multiple frequencies to improve localization accuracy.  

2) **Sparsity Enforcement**: Explicitly models and exploits the inherent sparsity of the problem (few active sources among many candidate locations).  

3) **Temporal Consistency**: Incorporates prior estimates to maintain track continuity and improve robustness to model mismatch.  

4) **Computational Efficiency**: The PG solver's structure enables parallel updates and rapid convergence to useful solutions within practical iteration counts.  

5) **Multiple Source Handling**: The group sparsity framework naturally accommodates tracking of multiple simultaneous sources.  

The system has been successfully tested using data from the SWellEX-3 experiment, demonstrating accurate tracking of a towed broadband source in shallow water environments. Additional tests simulating two simultaneous sources show promising results in maintaining separate tracks. Performance exceeds conventional MFT approaches while requiring substantially less computational resources.  

The invention further encompasses methods for dynamically adjusting tuning parameters (λ, μ) during operation to optimize performance under varying environmental conditions. Additional embodiments incorporate techniques for online model adaptation to reduce mismatch between assumed and actual propagation conditions.  

This novel framework for sparse, broadband underwater acoustic tracking represents a significant advance over prior systems, enabling real-time, multi-source localization with improved accuracy and reduced computational burden. The invention has broad applications in naval surveillance, environmental monitoring, and underwater navigation systems.