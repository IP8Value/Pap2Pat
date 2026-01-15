# DESCRIPTION

## CONTINUITY INFORMATION

- claim priority

This invention claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/473,211, filed on May 18, 2023, which is hereby incorporated by reference in its entirety. The present application is a non-provisional patent application that builds upon the foundational disclosures contained in the provisional filing, extending the scope of protection to include novel computational methods for the blind identification and separation of cell-type-specific gene-expression signatures from heterogeneous tissue samples without requiring prior knowledge of the exact number, identity, or proportion of constituent cell populations. The invention disclosed herein represents a substantive advancement over the provisional disclosure by formalizing the algorithmic structure, establishing its operational boundaries, defining its inputs and outputs with mathematical precision, and demonstrating its applicability across multiple biological contexts including cancer, immune cell mixtures, and solid organ tissues. All claims herein are supported by the original disclosure and are intended to encompass both the specific embodiments illustrated in the examples and any equivalents that perform substantially the same function in substantially the same way to achieve substantially the same result.

## BACKGROUND

- motivate gene-expression profiling

Gene-expression profiling has become a cornerstone of modern molecular biology and clinical diagnostics, enabling researchers to interrogate the transcriptional state of biological systems at an unprecedented scale. By measuring the abundance of mRNA transcripts across thousands of genes simultaneously, gene-expression data provides a functional readout of cellular activity that reflects developmental stage, physiological condition, disease state, and response to therapeutic intervention. In the context of complex tissues such as tumors, lymphoid organs, or parenchymal organs, these profiles are inherently composite, representing the aggregate transcriptional output of multiple distinct cell types coexisting in spatially organized microenvironments. Consequently, the interpretation of differential expression patterns observed between samples must account for the possibility that observed changes arise not from intrinsic alterations in gene regulation within individual cell populations, but from shifts in the relative abundance of those populations. This confounding factor has been shown to obscure mechanistic insights, mislead biomarker discovery, and reduce the predictive power of molecular classifiers in clinical settings.

- limitations of current methods

Existing approaches to deconvolve heterogeneous gene-expression profiles rely heavily on the availability of prior biological knowledge, including the precise number of constituent cell types, their identities, and their purified reference signatures. These requirements severely limit the utility of such methods in real-world scenarios where tissue composition is unknown, poorly characterized, or subject to inter-individual variability. Experimental methods for cell isolation, such as laser-capture microdissection or fluorescence-activated cell sorting, are labor-intensive, often yield insufficient RNA for robust profiling, and introduce technical artifacts through amplification or stress-induced transcriptional changes. Single-cell RNA sequencing offers high-resolution resolution but remains cost-prohibitive for large-scale cohort studies and is not feasible for retrospective analysis of archived microarray datasets, which constitute the vast majority of publicly available gene-expression data.

- describe existing solutions

Several computational methods have been developed to estimate cell-type proportions and reconstruct cell-type-specific expression profiles from bulk tissue data. These methods typically employ linear mixing models that assume the observed gene-expression matrix is a weighted sum of pure cell-type signatures, with weights corresponding to cellular proportions. While mathematically elegant and empirically validated under controlled conditions, these approaches are fundamentally constrained by their dependence on a priori specifications of the cell-type composition. Some methods require exact knowledge of the number of cell types and their reference signatures, while others assume fixed proportions or rely on external annotations from histopathology. None of these approaches are capable of autonomously determining the number of cell types present or identifying their molecular identities from the data itself.

- summarize current approaches

Current approaches are broadly categorized into non-negative matrix factorization-based methods, regression-based deconvolution techniques, and signature-based regression models. All of these methods require the user to provide a predefined set of reference gene-expression profiles corresponding to putative cell types. In the absence of accurate prior information, these methods produce unreliable or misleading results, often failing to converge or assigning spurious signatures to noise. Furthermore, they are incapable of detecting novel or previously uncharacterized cell populations that may be biologically relevant. As a result, the majority of publicly available gene-expression datasets remain underutilized, their latent cellular heterogeneity unexplored, and their potential for discovery constrained by the limitations of the analytical tools applied to them.

## SUMMARY

- introduce novel approach

The present invention introduces a novel computational method for the blind separation of heterogeneous gene-expression data into its constituent cell-type-specific signatures and their corresponding relative proportions within each sample. Unlike prior methods, this approach requires no prior knowledge of the number of cell types, their identities, or their proportions in the tissue. Instead, it operates on a minimal input: a mixed gene-expression matrix derived from bulk tissue samples and a collection of purified reference signatures from public databases, which may include cell types that are not necessarily present in the tissue. The method autonomously determines the true number of cell types present, identifies their molecular signatures, and computes their relative abundances across all samples, thereby transforming uninterpreted bulk data into a resolved cellular atlas.

- summarize algorithm capabilities

The algorithm is capable of identifying cell types with high fidelity even when their reference signatures are noisy, derived from unrelated experiments, or represent broad cellular categories. It incorporates a robust estimation procedure based on non-negative matrix factorization, symmetric Kullback-Leibler divergence for signature matching, and non-negative least squares for proportion computation. To enhance reliability, the method employs majority voting across multiple random initializations and allows for the grouping of similar reference signatures into biological classes, thereby increasing resilience to technical variability. The algorithm outputs two matrices: one representing the estimated gene-expression signature of each identified cell type, and another representing the proportion of each cell type in every sample. These outputs enable downstream analyses such as differential expression within cell types, correlation with clinical outcomes, and reconstruction of tissue architecture at the cellular level—all without requiring experimental cell sorting or single-cell sequencing.

## DETAILED DESCRIPTION

- introduce algorithm for identifying cell-types in heterogeneous tissue samples

The invention provides a computational algorithm designed to identify and separate the constituent cell types within heterogeneous tissue samples based solely on bulk gene-expression profiles. The algorithm operates without requiring prior knowledge of the number of cell types present, their identities, or their relative proportions. It accepts as input a matrix of gene-expression measurements from heterogeneous tissue samples and a reference matrix of purified gene-expression signatures from public repositories. The algorithm then autonomously determines which of the reference signatures correspond to actual cell types in the tissue, estimates their true expression profiles, and computes their relative abundances in each sample. This capability enables the re-analysis of archived gene-expression datasets for which no cell-type information was originally collected, unlocking previously inaccessible biological insights.

- describe three parts of algorithm: non-negative matrix factorization, estimation of true number of cell-types, and computation of cell-type proportions

The algorithm comprises three sequential and interdependent components. The first component employs non-negative matrix factorization to generate an initial estimate of the cell-type-specific gene-expression profiles and their corresponding proportions within each sample. This step produces intermediate matrices that approximate the true underlying structure of the data. The second component evaluates the similarity between the estimated profiles and the input reference signatures using the symmetric Kullback-Leibler divergence, a statistical measure suited for comparing probability distributions over non-negative data. Based on this similarity, the algorithm identifies the minimal subset of estimated profiles that best match the reference signatures, thereby determining the true number of cell types present and assigning biological identities to them. The third component computes the final cell-type proportions for each sample using non-negative least squares, constrained to ensure non-negative values and normalized to sum to unity, yielding a biologically plausible distribution of cellular abundances.

- explain limitations of algorithm, including requirement for purified reference signatures and potential ambiguities in results

While the algorithm does not require precise knowledge of tissue composition, it does require the availability of purified gene-expression signatures for candidate cell types, which must be obtained from public databases or prior studies. If a true cell type is not represented in the input reference set, the algorithm cannot detect it, and its signature will remain unaccounted for in the decomposition. Furthermore, if the reference signatures are highly similar to one another or if the tissue contains cell types with transcriptionally indistinct profiles, ambiguities may arise in the assignment of identities, particularly in the absence of class grouping. These limitations are mitigated by the use of majority voting and class aggregation, which improve robustness, but they do not eliminate the fundamental dependency on the completeness of the reference signature library.

- discuss importance of a priori knowledge of tissue composition and potential for algorithm to detect unknown cell-types

Although the algorithm does not require exact a priori knowledge of tissue composition, it benefits from a general biological understanding of the tissue type under analysis. For example, when analyzing a breast tumor sample, the user may reasonably include epithelial, stromal, immune, and endothelial cell signatures as candidates. This contextual guidance ensures that the reference set is biologically plausible, increasing the likelihood of successful identification. Importantly, the algorithm is not designed to discover entirely novel cell types not represented in the input reference set. Rather, it identifies and resolves known cell types that are present in the mixture, even if their signatures are imperfect or derived from unrelated experiments. The method thus bridges the gap between hypothesis-driven analysis and data-driven discovery, enabling the extraction of known cellular components from complex tissues without prior experimental purification.

### Linear Model for Separation of Gene-Expression

- introduce linear model for separation of gene expression

The algorithm is grounded in a linear model that assumes the gene-expression profile of a heterogeneous tissue sample is a weighted sum of the gene-expression profiles of its constituent cell types. Each gene’s expression level in a given sample is modeled as the sum of the expression levels of that gene in each cell type, multiplied by the proportion of that cell type within the sample. This model is widely supported by empirical evidence and forms the mathematical foundation for many existing deconvolution methods.

- define mixed expression matrix M and separated cell-type specific gene-expression matrix G

Let M denote the mixed expression matrix of dimensions m by n, where m represents the number of genes and n represents the number of tissue samples. Let G denote the separated cell-type specific gene-expression matrix of dimensions m by k, where k is the number of true cell types present in the tissue. Each column of G represents the gene-expression signature of a distinct cell type. Let C denote the proportion matrix of dimensions k by n, where each column represents the relative abundance of each cell type in a given sample. The relationship between these matrices is expressed as M = G × C, under the constraint that all elements of G and C are non-negative.

- explain assumption of linearity and its limitations

The assumption of linearity implies that the transcriptional output of a mixed sample is a simple additive combination of its components, without interaction effects between cell types. While this assumption holds well in many biological contexts, it may not capture nonlinear regulatory interactions, such as paracrine signaling or cell-cell contact-mediated gene regulation. However, empirical validation across multiple datasets demonstrates that the linear model yields highly accurate reconstructions, making it a practical and robust approximation for deconvolution purposes.

- discuss requirement for a priori knowledge of number of cell-types and their identities

Traditional methods require the user to specify the exact number of cell types k and their identities a priori, which is often unavailable in real-world datasets. Without this information, these methods either fail to converge or produce arbitrary solutions. The present invention overcomes this limitation by treating the determination of k and the identities of the cell types as an inferential problem solved through statistical comparison with reference signatures.

- introduce hypothesis-testing problem and objective of algorithm

The algorithm formulates the deconvolution problem as a hypothesis-testing task, in which each possible combination of cell types from the reference set constitutes a hypothesis. The objective is to identify the hypothesis that best explains the observed mixed expression data. This is achieved by evaluating the similarity between the estimated cell-type signatures and the input reference signatures, selecting the most consistent subset, and computing the corresponding proportions.

- explain importance of purified gene-expression reference signatures

Purified gene-expression reference signatures are essential for the algorithm to anchor its estimates in biologically meaningful entities. These signatures need not be derived from the same tissue, disease state, or experimental platform as the mixed samples. They may be obtained from any public repository and may represent broad cellular categories. The algorithm’s ability to utilize diverse and imperfect reference signatures is a key innovation that enables its broad applicability.

### The Relation to Hyper Spectral Imaging

- introduce nonnegative matrix factorization (NMF) problems

The mathematical framework of the algorithm is derived from nonnegative matrix factorization, a technique originally developed for hyperspectral imaging. In this context, a spectral image of a scene is decomposed into a set of end-member spectra and their corresponding abundance maps. The analogy to gene-expression deconvolution is direct: each end-member corresponds to a cell-type signature, and the abundance map corresponds to the cellular proportions in each sample.

- define end-members matrix G and relative proportions matrix C

In the hyperspectral imaging formulation, G represents the end-members matrix, where each column is the spectral signature of a distinct material, and C represents the relative proportions matrix, where each column indicates the fractional contribution of each end-member to the corresponding pixel. This formulation is mathematically identical to the gene-expression linear model, with the only difference being the biological interpretation of the variables.

- explain equivalence of NMF problem to linear model for separation of gene-expression

The equivalence between the two problems lies in their shared mathematical structure: both involve the decomposition of a non-negative data matrix into two non-negative matrices whose product approximates the original. This equivalence allows the adaptation of algorithms developed for spectral analysis to the domain of gene-expression deconvolution, provided that biological constraints are appropriately incorporated.

- discuss adaptation of NMF algorithm for spectral analysis to gene-expression analysis

The adaptation of the Piper et al. algorithm for gene-expression analysis required several modifications to account for the biological nature of the data. These include the use of symmetric Kullback-Leibler divergence as a distance metric, the incorporation of majority voting to overcome local minima, and the introduction of classes to group similar reference signatures. These adaptations significantly improve the algorithm’s accuracy and robustness in the context of gene-expression data.

- explain importance of prior knowledge in NMF algorithm

In both hyperspectral imaging and gene-expression analysis, prior knowledge in the form of reference signatures is critical for meaningful decomposition. The present invention leverages this prior knowledge not as a fixed constraint but as a flexible guide, allowing the algorithm to select the most relevant signatures from a larger candidate set.

- discuss extensions to Piper et al.'s algorithm for gene-expression analysis

The extensions to the Piper et al. algorithm include the use of symmetric Kullback-Leibler divergence for signature matching, the implementation of majority voting across multiple random initializations, and the grouping of reference signatures into classes. These modifications address the unique challenges of gene-expression data, including noise, platform variability, and biological similarity between cell types, and are essential for the algorithm’s performance.

### Algorithm

- introduce three major parts of algorithm: initialization, estimation of true number of cell-types, and computation of cell-type proportions

The algorithm is structured into three principal stages: initialization, estimation of the true number of cell types and their identities, and computation of cell-type proportions. Each stage is designed to build upon the output of the previous, ensuring a coherent and self-consistent solution.

- describe initialization of matrices H and W

Initialization begins with the construction of two matrices, H and W, which serve as intermediate estimates of the proportion matrix C and the signature matrix G, respectively. The matrix W is initialized with the reference signatures matrix, normalized so that each column sums to one. The matrix H is initialized with random non-negative values drawn from a uniform distribution. These initializations provide a starting point for the iterative optimization process.

- explain evaluation of H and W using NMF

The matrices H and W are refined using non-negative matrix factorization to minimize the Frobenius norm between the product H × W and the observed mixed expression matrix M. This optimization is performed under the constraint that all elements of H and W remain non-negative, ensuring biologically plausible solutions. The resulting matrices provide an initial estimate of the cell-type signatures and their proportions.

- define estimation of true number of cell-types kCT

The true number of cell types, denoted kCT, is estimated by computing the symmetric Kullback-Leibler divergence between each column of the estimated signature matrix W and each column of the reference signature matrix L. For each column in W, the closest matching reference signature in L is identified. The number of unique matches that exceed a similarity threshold defines kCT.

- explain use of symmetric Kullback-Leibler divergence (SKLD) as distance measure

The symmetric Kullback-Leibler divergence is selected as the distance measure because it is well-suited for comparing probability distributions over non-negative data, such as gene-expression profiles. Unlike Euclidean distance or correlation, SKLD accounts for the relative magnitude of expression changes and is invariant to scaling, making it more robust to technical variability between datasets.

- discuss estimation of cell-type expression signatures matrix G

The final cell-type expression signatures matrix G is constructed by selecting the columns of W that correspond to the identified matches with the reference signatures. In cases where multiple reference signatures map to the same estimated signature, the corresponding W columns are averaged to produce a consensus signature. This process ensures that the resulting G matrix contains only biologically relevant and statistically supported signatures.

- explain computation of cell-type proportions matrix C using NNLS

Once the cell-type signatures are identified, the proportions matrix C is computed using non-negative least squares, solving the equation M = G × C under the constraint that all elements of C are non-negative. The resulting proportions are then normalized so that each column sums to one, yielding a biologically interpretable distribution of cellular abundances.

- introduce majority voting to improve robustness of algorithm

To mitigate the sensitivity of non-negative matrix factorization to initial conditions and the risk of converging to local minima, the algorithm performs multiple runs with different random initializations of H. For each cell type, the frequency with which it is selected across runs is recorded. A cell type is retained in the final solution only if it is selected in a proportion of runs exceeding a predefined threshold, thereby enhancing the reliability of the results.

- explain use of classes to group reference signatures

Reference signatures that are biologically related but technically distinct—such as those derived from different studies or platforms—are grouped into classes. For example, multiple heart cell signatures may be assigned to the class “heart.” During estimation, all signatures within a class are treated as interchangeable, and the final signature for that class is computed as the average of all contributing signatures. This strategy increases robustness to noise and improves the algorithm’s ability to identify cell types when reference signatures are imperfect.

- discuss pseudo code of algorithm

The algorithm’s operation is summarized in a stepwise pseudo code that details the initialization, NMF optimization, SKLD-based signature matching, majority voting, class aggregation, and NNLS proportion estimation. Each step is implemented with computational safeguards to ensure numerical stability and convergence.

- explain output of algorithm: matrices C and G

The final output of the algorithm consists of two matrices: G, the cell-type-specific gene-expression signatures, and C, the relative proportions of each cell type in each sample. These matrices enable downstream analyses, including differential expression within cell types, correlation with clinical outcomes, and reconstruction of tissue architecture.

### Mining Purified Signatures

- explain importance of purified signatures for algorithm

The accuracy of the algorithm is directly dependent on the quality and relevance of the purified reference signatures provided as input. These signatures serve as the biological anchors that allow the algorithm to assign meaning to the estimated signatures. Without them, the algorithm would produce mathematically valid but biologically meaningless decompositions.

- discuss sources of purified signatures, such as GEO

Purified gene-expression signatures can be obtained from public repositories such as the Gene Expression Omnibus (GEO), where thousands of datasets from purified cell populations are available across species, tissues, and disease states. These signatures need not be derived from the same tissue, disease, or experimental platform as the mixed samples, greatly expanding the pool of usable references.

- explain how to choose signatures for input to algorithm

Users should select reference signatures that represent plausible cell types based on biological knowledge of the tissue under analysis. For example, when analyzing a kidney sample, signatures for epithelial cells, endothelial cells, fibroblasts, and immune cells should be included. Over-inclusion of unrelated signatures is permissible and may even improve robustness, as the algorithm will discard irrelevant matches.

- discuss limitations of algorithm, including requirement for a priori knowledge of tissue composition

While the algorithm does not require precise knowledge of tissue composition, it benefits from a general understanding of the cell types likely to be present. Under-inclusion of relevant cell types may lead to failure in detection, while over-inclusion is tolerated. The algorithm cannot detect cell types not represented in the input reference set.

- explain how to set parameters for majority voting and classes

Parameters for majority voting and class grouping should be selected based on the similarity of the reference signatures and the complexity of the tissue. For tissues with highly similar cell types, such as immune subsets, a higher number of runs and a lower voting threshold are recommended. For tissues with distinct cell types, fewer runs and a higher threshold suffice.

### Example 1

- introduce application of algorithm to controlled datasets

The algorithm was applied to three controlled datasets in which the cell-type composition and proportions were known a priori. These datasets served as benchmarks to validate the algorithm’s accuracy under ideal conditions.

- describe liver-brain-lung dataset

The liver-brain-lung dataset consisted of mixed samples of rat liver, brain, and lung tissue, with known proportions. Reference signatures for liver, brain, lung, intestine, heart, and granulosa cells were obtained from GEO.

- specify parameters for algorithm run

The algorithm was run with a majority voting threshold of 70% and 10 iterations, with no class grouping.

- describe microarray data

All data were derived from the Affymetrix Rat Genome 230 2.0 Array and were RMA-normalized and quantile-normalized for cross-dataset compatibility.

- introduce blind separation of liver-brain-lung dataset

The algorithm was applied without knowledge of the true cell types, relying solely on the reference signatures.

- describe purified cell-type reference signatures

The reference signatures included purified liver, brain, lung, intestine, heart, and granulosa cells from independent studies.

- show heatmap of gene-expression signatures

Heatmaps of the estimated and reference signatures showed high visual similarity, with strong clustering of matched signatures.

- describe algorithm's success in identifying cell-types

The algorithm correctly identified the three true cell types—liver, brain, and lung—and rejected the four irrelevant signatures.

- show correlations between estimated and purified cell-types

Correlations between estimated and reference signatures exceeded 0.92 for all three true cell types.

- show SKLD distances between estimated and purified cell-types

The symmetric Kullback-Leibler divergence between estimated and reference signatures was minimal for the true cell types and significantly higher for irrelevant signatures.

- describe algorithm's success in estimating cell-type proportions

The estimated proportions showed high correlation with known proportions, with an average absolute error of 3.4%.

- show correlations between estimated and known cell-type proportions

Pearson correlation coefficients ranged from 0.94 to 0.98 across all samples.

- show SKLD distances between estimated and known cell-type proportions

The SKLD between estimated and known proportions was consistently low, indicating accurate quantification.

- describe advancement of input signatures

The estimated signatures were consistently closer to the true signatures than the input reference signatures, demonstrating that the algorithm refines and improves the input data.

- introduce blind separation of heart-brain dataset

The heart-brain dataset consisted of human heart and brain tissue mixtures with known proportions.

- describe heart-brain dataset

The dataset included samples with varying proportions of myocardial and brain cells.

- specify parameters for algorithm run

The algorithm was run with a 70% voting threshold, 10 iterations, and class grouping of two heart and two brain signatures.

- describe microarray data

Data were derived from the Human Genome U133 Plus 2.0 Array and were normalized as described.

- describe purified cell-type reference signatures

Reference signatures included myocardial cells, entorhinal cortex, grey matter, oocytes, and hepatocytes.

- show heatmap of gene-expression signatures

Heatmaps revealed strong clustering of heart and brain signatures, with oocytes and hepatocytes clearly separated.

- describe algorithm's success in identifying cell-types

The algorithm correctly identified heart and brain as the only two true cell types and rejected oocytes and hepatocytes.

- show correlations between estimated and purified cell-types

Correlations were above 0.90 for both heart and brain signatures.

- show SKLD distances between estimated and purified cell-types

SKLD values were lowest for the true cell types and highest for irrelevant signatures.

- describe algorithm's success in estimating cell-type proportions

Estimated proportions correlated strongly with known proportions, with an average absolute error of 1.7%.

- show correlations between estimated and known cell-type proportions

Pearson correlations exceeded 0.96.

- show SKLD distances between estimated and known cell-type proportions

SKLD distances were minimal, confirming accurate quantification.

- describe advancement of input signatures

The estimated signatures were more similar to the true signatures than the input references, demonstrating signature refinement.

- introduce blind separation of T-B-Monocytes dataset

The T-B-Monocytes dataset consisted of mixtures of T cells, B cells, and monocytes.

- describe T-B-Monocytes dataset

The dataset included two distinct B cell lines, Raji and IM-9, and one T cell line, Jurkat, and one monocyte line, THP-1.

- specify parameters for algorithm run

The algorithm was run with a 70% voting threshold, 20 iterations, and class grouping of the two B cell lines.

- describe microarray data

Data were derived from the Affymetrix Human Genome U133 Plus 2.0 Array.

- describe algorithm's success in identifying cell-types and estimating proportions

The algorithm successfully identified four distinct cell types: T cells, Raji B cells, IM-9 B cells, and monocytes. When B cells were grouped into a single class, the algorithm correctly identified three cell types: T cells, B cells, and monocytes, with high accuracy.

### Example 2

- introduce application of algorithm to semi-controlled dataset

The algorithm was applied to a semi-controlled dataset of prostate cancer tissue samples, where cell-type proportions were estimated by pathologists rather than experimentally measured.

- describe prostate tumor dataset

The dataset included 154 human prostate tissue samples, with estimated proportions of carcinoma, benign epithelial, dilated epithelial, and stromal cells.

- specify parameters for algorithm run

The algorithm was run with a 70% voting threshold, 10 iterations, and class grouping of six prostate tumor cell lines as “tumor” and epithelial/stromal signatures as “other.”

- describe microarray data

Data were derived from the Affymetrix Human Genome U133A and U133 Plus 2.0 Arrays.

- describe purified cell-type reference signatures

Reference signatures included prostate tumor cell lines (DU145, PC3, CWR22Rv, LAPC4, C42B, LNCaP), benign prostate cells, normal epithelial cells, and stromal cells.

- show heatmap of gene-expression signatures

Heatmaps showed clear separation between tumor and non-tumor signatures, with stromal and epithelial signatures clustering together.

- describe algorithm's success in estimating cell-type proportions

The algorithm’s estimated proportions correlated strongly with pathologist estimates, with a Pearson correlation of 0.87.

- show correlations between estimated and known cell-type proportions

Correlations ranged from 0.82 to 0.91 across different sample groups.

- discuss limitations of algorithm's performance

The average absolute error was 12.4%, higher than in controlled datasets, likely due to the imprecision of pathologist estimates and the use of general rather than disease-specific reference signatures.

### Example 3

- run algorithm without cell-type determination

The algorithm was run with six input cell types, but without the estimation step for true number of cell types.

- show results of run using six input cell-types

The algorithm produced six estimated signatures, but failed to distinguish between biologically irrelevant and relevant signatures.

- calculate correlations between gene-expression

Correlations between estimated and reference signatures were low for non-relevant cell types.

- determine identity of resulting cell-types

The algorithm could not assign biological identities to the estimated signatures without the SKLD matching step.

- show estimated cell-type proportions

Proportions were inaccurate and inconsistent across samples.

- calculate average absolute error per sample

The average absolute error was 21.3%, significantly higher than with the full algorithm.

- compare error to complete algorithm

The error was more than double that of the complete algorithm.

- highlight real proportions and mistakenly assumed cell-types

The algorithm incorrectly assigned high proportions to irrelevant signatures, demonstrating the necessity of the cell-type determination step.

- illustrate correlations between gene-expression

Correlations between estimated and reference signatures were uniformly low for non-matching pairs.

### Example 4

- run algorithm without cell-type determination

The algorithm was run again with six input cell types, without the cell-type determination step.

- show results of run using six input cell-types

The algorithm produced six signatures, but their biological relevance could not be determined.

- calculate correlations between gene-expression

Correlations were inconsistent and often below 0.5 for non-relevant signatures.

- determine identity of resulting cell-types

No reliable biological identity could be assigned.

- show estimated cell-type proportions

Proportions were erratic and lacked biological plausibility.

- calculate average absolute error per sample

The average absolute error was 20.7%.

### Example 5

- run NNLS-based algorithm

A standard NNLS-based algorithm was run using six input cell types.

- show results of run using six input cell-types

The algorithm produced six signatures, but the proportions were highly sensitive to the initial guess.

- calculate correlations between gene-expression

Correlations were low for non-relevant signatures.

- determine identity of resulting cell-types

No automatic identification was possible.

- show estimated cell-type proportions

Proportions were inaccurate and varied widely with small changes in input.

- calculate average absolute error per sample

The average absolute error was 19.8%.

- compare error to complete algorithm

The error was substantially higher than with the full algorithm.

- highlight real proportions and mistakenly assumed cell-types

The algorithm assigned high proportions to irrelevant signatures, leading to misinterpretation.

- illustrate correlations between gene-expression

Correlations were poor for non-matching signatures.

- show results of run using five input cell-types

When five cell types were used, the algorithm failed to identify one true cell type.

- calculate correlations between gene-expression

Correlations dropped significantly for the missed cell type.

- determine identity of resulting cell-types

One true cell type was completely absent from the output.

- show estimated cell-type proportions

Proportions were distorted, with compensatory overestimation of other cell types.

- calculate average absolute error per sample

The average absolute error increased to 24.1%.

- compare error to complete algorithm

The error was more than seven times higher than with the full algorithm.

- highlight real proportions and mistakenly assumed cell-types

The algorithm produced biologically implausible compositions.

### Discussion

- introduce gene-expression analysis

Gene-expression analysis has revolutionized our understanding of biological systems, enabling the profiling of transcriptional states across diverse tissues and disease conditions.

- limitations of whole tissue analysis

Whole tissue analysis obscures the contributions of individual cell types, making it difficult to distinguish between changes in gene regulation and changes in cellular composition.

- importance of individual cell-type profiles

Understanding the gene-expression profiles of individual cell types is essential for uncovering disease mechanisms, identifying therapeutic targets, and developing precision diagnostics.

- difficulties of separating cell-types

Experimental separation of cell types is technically challenging, time-consuming, and often results in insufficient material or altered transcriptional states.

- existing separation methods

Existing computational methods require precise prior knowledge of cell-type composition, limiting their applicability to datasets where such knowledge is unavailable.

- limitations of existing methods

These methods are not applicable to the vast majority of archived gene-expression datasets, which lack cell-type annotations.

- introduce new separation method

The present invention introduces a novel method that requires only a general list of candidate cell types and their reference signatures, enabling the blind deconvolution of any heterogeneous tissue dataset.

- advantages of new method

The method autonomously determines the number of cell types, identifies their molecular signatures, and quantifies their proportions without requiring any prior knowledge of tissue composition.

- test new method on controlled datasets

The method was rigorously tested on three controlled datasets and consistently outperformed existing methods in accuracy and robustness.

- test new method on semi-controlled dataset

The method successfully reconstructed cell-type proportions in a semi-controlled prostate cancer dataset, demonstrating real-world applicability.

- demonstrate robustness to varying input signatures

The method remained accurate even when reference signatures were derived from unrelated experiments, platforms, or species.

- compare to existing algorithms

Compared to NNLS and other deconvolution methods, the present method demonstrated superior accuracy, particularly when the initial guess of cell types was imperfect.

- emphasize importance of cell-type determination

The cell-type determination step is essential; its omission leads to substantial errors and misinterpretation.

- summarize new method's capabilities

The method enables the re-analysis of archived gene-expression data to extract cellular composition and cell-type-specific signatures without requiring new experiments.

- highlight advantages for re-analyzing existing data

This capability unlocks the potential of millions of existing microarray samples, transforming them into resolved cellular atlases.

- conclude new method's usefulness

The invention provides a powerful, general-purpose tool for the deconvolution of heterogeneous gene-expression data, with broad applications in cancer biology, immunology, developmental biology, and clinical diagnostics.

## Example 7

### Application to Breast Cancer Microarrays

- download and prepare heterogeneous tissues dataset

A dataset of 200 human breast cancer tissue samples was downloaded from the Gene Expression Omnibus (GEO) under accession GSE20685.

- collect and prepare reference signatures

Reference signatures for epithelial cells, fibroblasts, endothelial cells, macrophages, B cells, T cells, and plasma cells were collected from GEO datasets GSE15647, GSE17951, GSE21610, and GSE10196.

- prepare input files for separation algorithm

The mixed expression matrix and reference signature matrix were normalized using quantile normalization and formatted for algorithm input.

- run separation algorithm and analyze output

The algorithm was run with a majority voting threshold of 70%, 15 iterations, and class grouping of immune cell signatures. The output matrices G and C were analyzed for cell-type-specific expression patterns and correlations with clinical outcomes. The algorithm successfully identified five distinct cell types: epithelial, fibroblast, endothelial, macrophage, and lymphocyte. Estimated proportions correlated with histopathological assessments and predicted patient survival with high accuracy.