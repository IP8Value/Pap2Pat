# DESCRIPTION

## BACKGROUND OF THE INVENTION

The accurate quantification of nucleic acids in biological samples is a foundational requirement across numerous fields of molecular biology, clinical diagnostics, and genomic research. Among the most critical applications is the detection and characterization of copy number variations (CNVs)—structural alterations in the genome involving gains or losses of DNA segments typically larger than 500 base pairs. CNVs are now recognized as major contributors to human genetic diversity and have been implicated in a wide spectrum of diseases, including neurodevelopmental disorders, cancer, and autoimmune conditions. Consequently, robust, precise, and scalable methods for CNV detection are essential for both research and clinical practice.

Traditional approaches to CNV analysis have relied heavily on array-based technologies such as array comparative genomic hybridization (array-CGH) and high-density single nucleotide polymorphism (SNP) microarrays. While these platforms offer genome-wide coverage and high throughput, they suffer from limited resolution—often unable to detect CNVs smaller than tens of kilobases—and reduced sensitivity in regions of complex genomic architecture or low probe density. Moreover, these methods are indirect; they infer copy number from hybridization intensity signals, which can be confounded by sequence composition bias, cross-hybridization, and normalization artifacts.

Real-time quantitative polymerase chain reaction (qPCR) provides an alternative that is sequence-specific, relatively inexpensive, and widely accessible. However, qPCR’s ability to discriminate between copy number states is constrained by its reliance on relative quantification and amplification efficiency assumptions. In practice, qPCR struggles to reliably distinguish differences smaller than a two-fold change in target concentration, which corresponds to the difference between one and two copies in a diploid genome. This limitation renders it inadequate for precise CNV analysis, especially in mosaic or heterogeneous samples where fractional copy number changes are common.

Digital PCR (dPCR) emerged as a transformative technology that overcomes many of these limitations by partitioning a PCR reaction into thousands of individual reactions, each containing zero, one, or few template molecules. By applying Poisson statistics to the fraction of positive partitions (those yielding amplification), dPCR enables absolute quantification of nucleic acid targets without the need for standard curves or reference samples. This approach significantly enhances precision, sensitivity, and dynamic range compared to qPCR. Early implementations of dPCR used manual serial dilution and endpoint detection in multiwell plates, but these were labor-intensive and prone to pipetting errors.

The advent of integrated nanofluidic platforms, such as the digital array developed by Fluidigm Corporation, has revolutionized dPCR by automating sample partitioning at the nanoliter scale. The digital array biochip contains 765 nanoliter-volume reaction chambers per panel, fabricated using microelectromechanical systems (MEMS) technology. Integrated microvalves and channels enable precise loading of sample and reagent mixtures into each chamber, ensuring uniform partitioning and minimizing cross-contamination. Following thermal cycling on a real-time PCR instrument such as the BioMark system, fluorescence imaging allows automated counting of positive chambers for one or more targets using dedicated analysis software.

Despite these technological advances, the full analytical potential of the digital array for CNV studies has been hindered by a lack of rigorous statistical frameworks for interpreting the raw count data. Specifically, while it is straightforward to estimate the concentration of a single target from the number of positive chambers, the problem becomes significantly more complex when estimating the ratio of two concentrations—as required in CNV analysis where a test gene is compared to a single-copy reference gene. Existing methods often rely on simplified approximations or fail to provide statistically valid confidence intervals for the estimated ratio, limiting their reliability in diagnostic or regulatory contexts.

Prior art includes a Bayesian approach proposed by Warren et al., which models the number of molecules per chamber using a uniform prior distribution up to an assumed maximum (e.g., 4000 molecules) and computes posterior probabilities via combinatorial enumeration. While this method yields credible intervals, it requires subjective specification of prior distributions and does not directly address the frequentist confidence interval for a ratio of concentrations—a standard requirement in many scientific and regulatory settings. Furthermore, the Bayesian framework does not naturally extend to multiplexed assays with independent amplification kinetics, which is a key advantage of the digital array platform.

Thus, there remains a critical unmet need for a mathematically rigorous, assumption-minimal, and computationally tractable method to estimate the ratio of true molecular concentrations from digital array data and to compute associated confidence intervals that reflect the inherent stochasticity of molecular partitioning and detection. Such a method must account for the nonlinear relationship between observed positive counts and underlying concentration, handle the propagation of uncertainty from two independent measurements, and remain valid across a wide dynamic range of input concentrations—from sub-single-copy to hundreds of copies per panel. The present invention fulfills this need by providing a novel statistical framework grounded in classical estimation theory, Poisson process modeling, and numerical integration techniques, enabling accurate and reliable CNV determination on digital array platforms.

## SUMMARY OF THE INVENTION

The present invention provides a comprehensive method and associated computational framework for determining the ratio of true molecular concentrations of two nucleic acid sequences in a biological sample using digital PCR performed on a nanofluidic digital array, along with statistically rigorous confidence intervals for said ratio. The invention specifically addresses the problem of copy number variation (CNV) analysis by enabling precise, absolute quantification of a target gene relative to a single-copy reference gene, thereby facilitating the detection of genomic gains or losses with high accuracy and reproducibility.

In accordance with the invention, a DNA sample is partitioned into a known number of discrete reaction chambers—typically 765 per panel on a commercial digital array chip—such that each chamber receives a random subset of the total molecules according to a Poisson distribution. Multiplex PCR is then performed within each chamber using sequence-specific primers and fluorescent probes for both the target gene and the reference gene, allowing independent detection of amplification events for each sequence. After thermal cycling, the number of chambers positive for the target gene (denoted H₁) and the number positive for the reference gene (denoted H₂) are counted using automated imaging and analysis software.

The core innovation lies in the mathematical transformation and statistical inference applied to these raw counts. The invention establishes that the probability p of a chamber being positive for a given gene is related to the true average concentration λ (in molecules per chamber) by the equation p = 1 − e⁻λ. From the observed proportion of positive chambers, p̂ = H/C (where C is the total number of chambers), an unbiased estimator of λ is derived as λ̂ = −ln(1 − p̂). This estimator accounts for the nonlinearity inherent in the partitioning process, wherein multiple molecules in a single chamber do not produce additional positive signals beyond the first.

To quantify uncertainty, the invention computes a confidence interval for λ based on the binomial sampling distribution of H. Given that H follows a binomial distribution with parameters C and p, the standard error of p̂ is √[p̂(1 − p̂)/C]. For large C—which is satisfied by digital arrays with hundreds to thousands of chambers—the sampling distribution of p̂ is approximately normal. Using this approximation, a (1−α) confidence interval for p is constructed as p̂ ± z_(α/2)√[p̂(1 − p̂)/C], where z_(α/2) is the critical value from the standard normal distribution (e.g., 1.96 for 95% confidence). This interval is then transformed via the inverse relationship λ = −ln(1 − p) to yield a confidence interval [λ_low, λ_high] for the true concentration.

The principal advancement of the invention is the extension of this framework to the ratio r = λ₁/λ₂ of two independent concentrations. Recognizing that λ̂₁ and λ̂₂ are derived from independent binomial processes, the invention provides two complementary approaches for computing a confidence interval [r_low, r_high] for r. The first is a direct analytical approximation based on a generalization of Fieller’s Theorem, adapted to accommodate asymmetric confidence intervals for λ₁ and λ₂. This method constructs a confidence region in the (λ₁, λ₂) plane as the union of four quadrant-wise elliptical segments defined by the upper and lower bounds of each concentration’s confidence interval. The slopes of lines tangent to this region and passing through the origin define the lower and upper bounds of the ratio’s confidence interval.

The second and preferred approach is a numerical algorithm that makes no distributional assumptions beyond independence. This algorithm constructs empirical sampling distributions (e.g., histograms) for λ̂₁ and λ̂₂ from the observed data or via resampling. It then computes the joint distribution of the pair (λ̂₁, λ̂₂) under the assumption of independence. The sampling distribution of the ratio estimator r̂ = λ̂₁/λ̂₂ is obtained by integrating the joint distribution over angular sectors (or “wedges”) in the (λ₁, λ₂) plane corresponding to narrow intervals of r. Specifically, for each candidate ratio value r*, the algorithm accumulates the probability mass in the region where λ₁/λ₂ ≈ r*, effectively performing a change of variables from Cartesian to polar-like coordinates. The resulting distribution q(r̂) is then used to determine the central (1−α) interval that contains the true ratio with the specified confidence level.

The invention further includes methods for handling edge cases, such as when the estimated concentration of the reference gene is very low, which could lead to unbounded confidence intervals. In such scenarios, the algorithm adaptively refines the histogram binning for the low-concentration estimate to improve numerical stability and accuracy.

Validation of the invention was performed through extensive computer simulations and controlled wet-lab experiments. In simulation studies involving 50,000 digital array panels with a known input ratio of 2:1, the computed 95% confidence intervals contained the true ratio in 94.9% of cases, confirming the statistical validity of the method. Experimental validation used a spike-in system where a synthetic RPP30 DNA construct was titrated into human genomic DNA at known ratios (1:1, 1:1.5, 1:2, 1:2.5, 1:3, and 1:3.5) relative to the endogenous single-copy RNase P gene. Using five panels per condition (totaling 3,825 chambers), the invention accurately estimated all target ratios, with the known values consistently falling within the computed 95% confidence intervals. Moreover, increasing the number of panels (and thus total chambers) led to narrower confidence intervals and improved discrimination between adjacent ratios, demonstrating the scalability and precision of the approach.

In summary, the invention provides a complete, end-to-end solution for CNV analysis on digital array platforms, comprising: (1) a statistically sound estimator for absolute molecular concentration from positive chamber counts; (2) a method for computing confidence intervals for single concentrations; (3) a robust numerical algorithm for estimating the ratio of two concentrations and its confidence interval without restrictive distributional assumptions; and (4) practical guidelines for experimental design and data interpretation. This framework transforms the digital array from a qualitative or semi-quantitative tool into a rigorously quantitative platform for genomic analysis, with broad applicability in research, clinical diagnostics, and biopharmaceutical development.

## DETAILED DESCRIPTION OF SPECIFIC EMBODIMENTS

The present invention is best understood by reference to specific embodiments that illustrate its implementation in the context of copy number variation (CNV) analysis using a nanofluidic digital array platform. The following description details the experimental workflow, mathematical derivations, computational algorithms, and validation procedures that constitute the invention. While the examples focus on human genomic DNA and specific gene targets, the principles disclosed herein are universally applicable to any nucleic acid quantification scenario involving digital PCR with partitioned reactions.

**Experimental Setup and Sample Preparation**

In a representative embodiment, genomic DNA is extracted from a biological source, such as a human cell line (e.g., Coriell NA10860), using standard purification methods. A synthetic DNA construct, designed to mimic a fragment of the human RPP30 gene, is prepared by chemical synthesis (e.g., from Integrated DNA Technologies). The RPP30 construct serves as the test target whose copy number is to be varied relative to a stable reference. The reference gene is selected to be a well-characterized single-copy locus in the human genome; in this case, RNase P is used, consistent with established practices in nucleic acid quantification.

A series of DNA mixtures is prepared by spiking known molar amounts of the RPP30 synthetic construct into a fixed quantity of human genomic DNA. The spike-in levels are chosen to simulate diploid genomic copy numbers of 2, 3, 4, 5, 6, and 7 for RPP30, corresponding to RPP30:RNase P ratios of 1:1, 1.5:1, 2:1, 2.5:1, 3:1, and 3.5:1, respectively. Each mixture is then subjected to digital PCR on a Fluidigm digital array chip.

For each panel of the digital array, a 10-µL reaction mix is assembled containing: 1× TaqMan Universal PCR Master Mix (Applied Biosystems), 1× RNase P-VIC TaqMan assay (comprising VIC-labeled probe and gene-specific primers), 1× RPP30-FAM TaqMan assay (with FAM-labeled probe and primers at 900 nM primer and 200 nM probe concentrations), 1× Sample Loading Reagent (Fluidigm), and the DNA mixture adjusted to contain approximately 1,100–1,300 copies of the RNase P gene per panel. Of this 10-µL mix, 4.59 µL is loaded onto the digital array chip, which automatically partitions the volume into 765 individual reaction chambers, each with a volume of approximately 6 nL. Thus, the total analyzed volume per panel is 4.59 µL, and the effective concentration of molecules is referenced to this partitioned volume.

The chip is then thermocycled on a BioMark real-time PCR system using the following protocol: an initial denaturation at 95°C for 10 minutes, followed by 40 cycles of 95°C for 15 seconds (denaturation) and 60°C for 1 minute (annealing and extension). Fluorescence signals for both FAM (RPP30) and VIC (RNase P) channels are recorded at the end of each cycle. Post-amplification, the Digital PCR Analysis software (Fluidigm) processes the fluorescence trajectories to classify each chamber as positive or negative for each target based on threshold-crossing criteria. The output is two integers per panel: H₁ (number of FAM-positive chambers for RPP30) and H₂ (number of VIC-positive chambers for RNase P).

**Mathematical Framework for Concentration Estimation**

The invention begins with the fundamental relationship between the observed number of positive chambers and the true molecular concentration. Consider a single gene target. Let C denote the total number of chambers in a panel (C = 765 in the standard digital array). Let λ represent the true average number of target molecules per chamber, i.e., the concentration in molecules per 6 nL. Under the assumption that molecules are randomly and independently distributed among chambers—a valid assumption given the nanoliter-scale mixing and absence of aggregation—the number of molecules K in any given chamber follows a Poisson distribution with parameter λ:

P(K = k) = (λᵏ e⁻λ) / k! for k = 0, 1, 2, …

A chamber is scored as positive if it contains at least one molecule (K ≥ 1). Therefore, the probability p that a chamber is positive is:

p = P(K ≥ 1) = 1 − P(K = 0) = 1 − e⁻λ.

This equation establishes a deterministic, monotonic relationship between λ and p. Given an observed count H of positive chambers out of C total, the sample proportion p̂ = H/C serves as an unbiased estimator of p. Substituting p̂ into the above relationship yields an estimator for λ:

λ̂ = −ln(1 − p̂) = −ln(1 − H/C).

This estimator is consistent and asymptotically unbiased as C → ∞. It correctly accounts for the fact that chambers containing multiple molecules still contribute only one positive count, thereby avoiding the underestimation that would occur if λ were naively equated to H.

**Confidence Interval for Single Concentration**

To assess the uncertainty in λ̂, the invention derives a confidence interval based on the sampling variability of H. Since each chamber is an independent Bernoulli trial with success probability p, H ~ Binomial(C, p). For large C (which holds for C ≥ 765), the Central Limit Theorem ensures that p̂ is approximately normally distributed with mean p and variance p(1−p)/C. A (1−α) confidence interval for p is therefore:

p̂ ± z_(α/2) √[p̂(1 − p̂)/C],

where z_(α/2) is the (1−α/2) quantile of the standard normal distribution (e.g., 1.96 for α = 0.05).

Because λ is a smooth, strictly increasing function of p (since dλ/dp = 1/(1−p) > 0), the confidence interval for λ is obtained by applying the same transformation to the endpoints of the p-interval:

λ_low = −ln(1 − [p̂ + z_(α/2) √(p̂(1 − p̂)/C)]),

λ_high = −ln(1 − [p̂ − z_(α/2) √(p̂(1 − p̂)/C)]).

Note that due to the convexity of the −ln(1−p) function, the resulting λ-interval is asymmetric around λ̂, with the upper bound typically farther from the estimate than the lower bound—a feature that accurately reflects the skewness of the underlying sampling distribution.

**Ratio Estimation and Confidence Interval Construction**

The primary objective in CNV analysis is to estimate r = λ₁/λ₂, the ratio of true concentrations of the test gene (λ₁) and reference gene (λ₂). Let H₁ and H₂ be the observed positive counts for the two genes, assumed independent because the PCR amplifications are performed with distinct primer-probe sets and detected in separate fluorescence channels. The point estimate is simply r̂ = λ̂₁/λ̂₂.

The challenge lies in constructing a valid confidence interval for r. The invention provides two methods:

*Analytical Approximation via Generalized Fieller’s Theorem*:  
Fieller’s Theorem traditionally provides confidence intervals for the ratio of two normally distributed variables. Although λ̂₁ and λ̂₂ are not exactly normal, their distributions are approximately so for large C. The invention extends this by accommodating asymmetric confidence intervals for λ₁ and λ₂. Let [λ₁_low, λ₁_high] and [λ₂_low, λ₂_high] be the (1−α) confidence intervals for the two concentrations. Define the half-widths as W_L = λ̂₁ − λ₁_low, W_R = λ₁_high − λ̂₁, H_B = λ̂₂ − λ₂_low, and H_T = λ₂_high − λ̂₂.

The confidence region in the (λ₁, λ₂) plane is approximated as the union of four elliptical quadrants centered at (λ̂₁, λ̂₂), with semi-axes (W_L, H_B), (W_R, H_B), (W_L, H_T), and (W_R, H_T). The lines through the origin tangent to this region have slopes given by solving the quadratic equation derived from the condition that the distance from the origin to the ellipse equals zero discriminant. The resulting formulas for the lower and upper bounds are:

r_low = [ (λ̂₁λ̂₂ − W_L H_T) − √( (λ̂₁ H_T − λ̂₂ W_L)² + (W_L H_T)(4λ̂₁λ̂₂ − W_L H_T) ) ] / (λ̂₂² − H_T²),

r_high = [ (λ̂₁λ̂₂ + W_R H_B) + √( (λ̂₁ H_B + λ̂₂ W_R)² − (W_R H_B)(4λ̂₁λ̂₂ + W_R H_B) ) ] / (λ̂₂² − H_B²),

with appropriate adjustments when denominators approach zero. This method is computationally efficient and sufficiently accurate for most practical purposes.

*Numerical Integration Algorithm (Preferred Embodiment)*:  
To avoid distributional assumptions entirely, the invention implements a numerical algorithm that constructs the sampling distribution of r̂ directly. The steps are as follows:

1. From the observed H₁ and H₂, compute λ̂₁ and λ̂₂.
2. Generate empirical sampling distributions f₁(λ₁) and f₂(λ₂) for the two concentrations. This can be done by:
   a. Using the exact binomial likelihood: for each possible h ∈ {0,1,…,C}, compute p = h/C, then λ = −ln(1−p), and assign probability P(H = h) = CCh pʰ (1−p)^(C−h).
   b. Or, for computational efficiency, approximate f₁ and f₂ as normal distributions with means λ̂₁, λ̂₂ and variances derived from the delta method: Var(λ̂) ≈ p̂(1−p̂)/(C(1−p̂)²).
3. Assuming independence, form the joint distribution f(λ₁, λ₂) = f₁(λ₁) × f₂(λ₂).
4. To compute the density q(r) of the ratio r = λ₁/λ₂, perform a change of variables. For a grid of r values, integrate f(λ₁, λ₂) over the region where λ₁/λ₂ ∈ [r − Δr/2, r + Δr/2] for small Δr. Numerically, this is implemented by summing f(λ₁, λ₂) over all (λ₁, λ₂) pairs in a histogram that satisfy r − ε < λ₁/λ₂ < r + ε for a small tolerance ε.
5. Normalize q(r) to obtain a proper probability density.
6. Determine the (1−α) confidence interval [r_low, r_high] as the shortest interval containing 100(1−α)% of the probability mass under q(r).

This algorithm is robust, handles asymmetry and non-normality naturally, and provides exact results given sufficient computational resolution. Special care is taken when λ̂₂ is near zero: in such cases, the bin size for f₂ is reduced adaptively to prevent numerical instability, and if the confidence interval for λ₂ includes zero, the ratio interval is reported as unbounded on the relevant side.

**Validation and Performance Characteristics**

The invention was validated through both simulation and empirical experiments. In simulation, 70,000 panels were generated with λ = 400/765 ≈ 0.523 molecules per chamber (corresponding to ~400 molecules per panel). The observed distribution of H matched the theoretical binomial prediction, and 95% of the computed λ confidence intervals contained the true λ, confirming calibration.

For ratio estimation, 50,000 panels were simulated with a true ratio r = 2. Using the numerical algorithm, 94.9% of the 95% confidence intervals contained r = 2, demonstrating nominal coverage.

Empirically, six mixtures with known RPP30:RNase P ratios (1 to 3.5) were tested in quintuplicate (5 panels each). For each mixture, pooled analysis was performed by summing H₁ and H₂ across P = 1 to 5 panels. Results showed:
- All known ratios fell within the computed 95% confidence intervals.
- Confidence interval width decreased with increasing P, as expected from increased total chamber count (C_total = 765P).
- With P = 1, intervals for ratios 2 and 2.5 overlapped, limiting discriminability.
- With P ≥ 3, intervals were non-overlapping, enabling clear distinction between adjacent copy number states.

These results confirm that the invention enables accurate, precise, and scalable CNV detection on digital array platforms.

**Implementation and Practical Considerations**

The methods of the invention are implemented in software that interfaces with digital array analysis outputs. Input consists of H₁, H₂, and C (number of chambers). The software computes λ̂₁, λ̂₂, r̂, and the confidence intervals using either the analytical or numerical method, with the latter as default. Users can specify confidence level (e.g., 90%, 95%, 99%) and choose pooling across multiple panels.

Practical recommendations include:
- Ensuring that the number of molecules per panel is within the optimal range (~100–1000) to balance signal (sufficient positives) and resolution (minimizing multi-molecule chambers).
- Using a validated single-copy reference gene.
- Performing technical replicates (multiple panels) to improve precision.
- Avoiding extreme ratios where one target is near the limit of detection.

In conclusion, the detailed embodiments described herein demonstrate that the invention provides a mathematically rigorous, experimentally validated, and practically implementable solution for CNV analysis using digital PCR on nanofluidic arrays, significantly advancing the state of the art in genomic quantification.