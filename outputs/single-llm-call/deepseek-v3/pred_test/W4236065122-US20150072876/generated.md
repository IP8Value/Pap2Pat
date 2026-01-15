Below is the complete patent application following the provided outline and research paper content:

---

# DESCRIPTION  

## CONTINUITY INFORMATION  

The present application claims priority to U.S. Provisional Patent Application No. [XXXXXX], filed on [DATE], the contents of which are hereby incorporated by reference in their entirety.  

## BACKGROUND  

Gene-expression profiling of heterogeneous tissues presents significant challenges due to the inherent complexity of cellular composition within such tissues. Current methodologies for analyzing gene-expression data often fail to account for the diverse cell populations present, leading to results that may reflect differences in cell-type proportions rather than meaningful biological variations in gene expression. Existing experimental techniques for separating cell-types, such as laser-capture microdissection and flow cell sorting, are labor-intensive, require specialized equipment, and often yield insufficient quantities of RNA for downstream applications. Furthermore, single-cell RNA sequencing, while promising, remains cost-prohibitive for large-scale studies.  

Current computational approaches for separating gene-expression profiles from heterogeneous tissues rely on linear models that assume known proportions or identities of cell-types. These methods, while effective in controlled settings, are limited by their dependence on a priori knowledge of tissue composition, which is often unavailable for publicly available datasets. Existing algorithms require input parameters such as the number of cell-types, their identities, or their relative proportions—information that is rarely accessible for most gene-expression studies. As a result, these methods cannot be broadly applied to reanalyze existing datasets where such details were not experimentally determined.  

## SUMMARY  

The present invention provides a novel computational algorithm for blindly identifying and separating cell-types within heterogeneous gene-expression datasets without requiring prior knowledge of tissue composition. The algorithm leverages non-negative matrix factorization (NMF) and symmetric Kullback-Leibler divergence (SKLD) to estimate the number of cell-types, their identities, their gene-expression signatures, and their relative proportions within each sample.  

Key capabilities of the algorithm include:  
1. **Blind separation** of mixed gene-expression data without prior knowledge of cell-type composition.  
2. **Robust estimation** of cell-type identities using purified reference signatures from public repositories.  
3. **Improved accuracy** through majority voting and class-based grouping of similar reference signatures.  
4. **Broad applicability** to existing datasets, enabling reanalysis of archived gene-expression profiles.  

## DETAILED DESCRIPTION  

### Introduction to the Algorithm  

The algorithm comprises three major components:  
1. **Initialization**: The algorithm begins with an initial estimate of possible cell-types and their reference signatures.  
2. **Estimation of True Cell-Types**: Using NMF and SKLD, the algorithm identifies the actual number of cell-types present and their identities.  
3. **Computation of Proportions**: Non-negative least squares (NNLS) is employed to determine the relative proportions of each cell-type per sample.  

The algorithm addresses limitations of existing methods by incorporating majority voting to enhance robustness and class-based grouping to handle noisy or similar reference signatures.  

### Linear Model for Separation of Gene-Expression  

The algorithm is based on a linear model where the mixed gene-expression matrix **M** is decomposed into cell-type-specific expression matrix **G** and proportions matrix **C**:  

\[ M = G \cdot C \]  

Key assumptions and constraints include:  
- **Non-negativity**: All entries in **G** and **C** must be non-negative.  
- **Normalization**: Columns of **G** sum to their mean to ensure comparability.  
- **Reference Signatures**: Purified gene-expression profiles for candidate cell-types are required but need not be study-specific.  

### The Relation to Hyper Spectral Imaging  

The algorithm adapts techniques from hyper-spectral imaging, where NMF is used to decompose mixed spectral data into constituent materials. In this context:  
- **End-members** correspond to cell-types.  
- **Spectral signatures** are analogous to gene-expression profiles.  
- **Proportions** represent the relative abundance of each cell-type.  

Modifications for gene-expression analysis include the use of SKLD for distance measurement and majority voting to resolve ambiguities in similar cell-types.  

### Algorithm  

#### Initialization  
The algorithm initializes matrices **H** (proportions) and **W** (expression) using random values and input reference signatures.  

#### Estimation of **kCT** (True Number of Cell-Types)  
The symmetric Kullback-Leibler divergence (SKLD) is used to match estimated profiles to reference signatures, identifying the most probable cell-types.  

#### Computation of Proportions  
NNLS solves for matrix **C**, representing cell-type proportions per sample, under non-negativity constraints.  

#### Majority Voting and Classes  
- **Majority voting** improves robustness by aggregating results from multiple runs.  
- **Classes** group similar reference signatures to handle biological variability.  

### Mining Purified Signatures  

Purified reference signatures are sourced from public repositories (e.g., GEO) and selected based on biological relevance. The algorithm accommodates multiple signatures per cell-type, enhancing accuracy.  

### Example 1: Liver-Brain-Lung Dataset  

The algorithm was tested on a controlled dataset of rat liver, brain, and lung cell mixtures. Key results include:  
- Accurate identification of the three cell-types (correlation > 0.9).  
- Precise estimation of proportions (average error: 3.4%).  
- Improved signature accuracy compared to input references.  

### Example 2: Prostate Tumor Dataset  

Application to a semi-controlled dataset demonstrated:  
- Correlation with pathologist-estimated proportions.  
- Robustness to noisy reference signatures.  

### Discussion  

The algorithm enables reanalysis of existing gene-expression data by eliminating the need for prior knowledge of tissue composition. Its advantages over existing methods include:  
- **Blind separation** without predefined parameters.  
- **Flexibility** in reference signature selection.  
- **Scalability** to large datasets.  

Future applications include cancer subtyping and biomarker discovery.  

---

This patent application provides a comprehensive and standalone description of the invention, adhering to formal patent language and structure. Let me know if you'd like any refinements or additional details.