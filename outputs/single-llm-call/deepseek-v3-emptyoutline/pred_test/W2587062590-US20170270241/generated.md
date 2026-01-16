Below is the complete patent application drafted according to the provided outline and research paper.  

---

# DESCRIPTION  

## GOVERNMENT FUNDING  
This invention was made with government support under [Grant Number] awarded by [Agency]. The government has certain rights in the invention.  

## BACKGROUND  
Single-cell RNA sequencing (scRNA-seq) has revolutionized the study of cellular heterogeneity, enabling the identification of rare cell populations and lineage trajectories. However, existing computational methods for analyzing scRNA-seq data suffer from significant limitations, including poor handling of dropout events (technical zeros), inability to quantify cellular differentiation potential, and lack of intuitive visualization of developmental trajectories. Current approaches such as principal component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), and diffusion maps fail to accurately resolve lineage hierarchies due to their reliance on holistic representations rather than biologically interpretable parts-based decomposition.  

Furthermore, conventional methods do not provide a quantitative measure of cellular plasticity or differentiation potential, which is critical for understanding progenitor states in developmental biology and regenerative medicine. There remains an unmet need for a computational framework that can robustly decompose scRNA-seq data, quantify cellular multipotency, and reconstruct lineage trajectories with high accuracy.  

## SUMMARY  
The present invention provides a novel computational pipeline, termed **dpath**, for analyzing single-cell RNA sequencing data. The dpath pipeline comprises four key innovations:  

1. **Weighted Poisson Non-Negative Matrix Factorization (wp-NMF)** – A decomposition method that accounts for dropout events by weighting gene expression probabilities, improving the resolution of metagene signatures.  
2. **Metagene Entropy** – A quantitative metric to assess cellular differentiation potential based on the uncertainty of metagene expression profiles.  
3. **Self-Organizing Map (SOM)-Based Metacell Landscape** – A two-dimensional visualization framework that organizes cells into metacells with similar expression patterns, enabling intuitive lineage reconstruction.  
4. **Random Walk with Restart (RWR) Algorithm** – A prioritization method to infer progenitor and committed states by modeling transitions on a heterogeneous metacell-metagene graph.  

The dpath pipeline outperforms existing methods (e.g., PCA, Monocle, Wishbone) in resolving lineage hierarchies, as demonstrated by its application to Etv2-expressing cells during murine embryogenesis. Experimental validation confirmed that dpath accurately identifies progenitor states, predicts differentiation trajectories, and reveals key regulatory pathways such as Sonic Hedgehog (SHH) signaling in haemato-endothelial development.  

## DETAILED DESCRIPTION  

### Definitions  
- **Metagene**: A non-negative linear combination of genes representing a distinct transcriptional program.  
- **Metacell**: A cluster of cells with similar metagene expression profiles, organized on a SOM.  
- **Metagene Entropy**: A measure of cellular differentiation potential, calculated as the entropy of metagene coefficients.  
- **Dropout Event**: A technical artifact in scRNA-seq where a truly expressed gene fails to be detected.  
- **RWR (Random Walk with Restart)**: A graph-based algorithm to prioritize cellular states by simulating transitions between metacells and metagenes.  

### ASPECTS OF THE INVENTION  
1. **wp-NMF for Dropout-Aware Decomposition**  
   - The invention employs a weighted Poisson model to decompose scRNA-seq data into metagenes, where each entry is weighted by its likelihood of being a true expression event rather than a dropout.  
   - The expected gene expression is modeled as:  

     \[  
     \mu_{nm} = \sum_{k=1}^K U_{nk} V_{km}  
     \]  

     where \(U_{nk}\) is the metagene basis, \(V_{km}\) is the metagene coefficient, and \(K\) is the number of metagenes.  

2. **Metagene Entropy as a Differentiation Metric**  
   - Cells with high metagene entropy exhibit multipotency, while those with low entropy are lineage-committed.  
   - Entropy is computed as:  

     \[  
     H_m = -\sum_{k=1}^K V_{km} \log V_{km}  
     \]  

3. **SOM-Based Lineage Visualization**  
   - Cells are mapped to a hexagonal grid where neighboring metacells share similar expression profiles.  
   - Progenitor states occupy central regions with high entropy, while committed states localize to peripheries.  

4. **RWR for Trajectory Inference**  
   - A transition matrix models probabilities between metacells and metagenes.  
   - Developmental trajectories are inferred as shortest paths from high-entropy to low-entropy states.  

### Materials and Methods  
**Cell Isolation and scRNA-seq**  
- Etv2-EYFP+ cells were isolated from murine embryos at E7.25, E7.75, and E8.25 via fluorescence-activated cell sorting (FACS).  
- Libraries were prepared using Fluidigm C1 and sequenced on Illumina MiSeq (75-bp paired-end).  

**Computational Pipeline**  
1. **Data Preprocessing**  
   - Genes with high technical noise or low variability were filtered.  
   - Expression values were log-transformed and normalized.  

2. **wp-NMF Optimization**  
   - The objective function:  

     \[  
     \max_{U,V} \sum_{n,m} \pi_{nm} \left( X_{nm} \log \mu_{nm} - \mu_{nm} \right)  
     \]  

     where \(\pi_{nm}\) is the dropout probability.  

3. **SOM Clustering**  
   - Metacells were generated using the **kohonen** R package with a 15×15 grid.  

4. **RWR Prioritization**  
   - Steady-state probabilities were computed using the **igraph** package (restart probability = 0.85).  

### Data Availability  
The scRNA-seq data are deposited in the NCBI Sequence Read Archive (PRJNA350294). The dpath software is available as an R package (Supplementary Software 1).  

### Results  
- **Metagene Decomposition**: wp-NMF resolved four metagenes corresponding to endothelial (MG1), haematopoietic (MG2), endocardial (MG3), and mesodermal (MG4) lineages.  
- **Lineage Trajectories**: dpath reconstructed paths from T+ progenitors to committed endothelial, blood, and endocardial cells, validated by immunohistochemistry.  
- **SHH Pathway**: RWR identified Sonic Hedgehog signaling as critical for haemato-endothelial differentiation, confirmed via agonist/antagonist experiments.  
- **Performance Benchmarking**: dpath achieved superior pseudotime accuracy (Kendall’s τ = 0.798) compared to Monocle (0.213) and Wishbone (0.375).  

### DISCUSSION  
The dpath pipeline represents a paradigm shift in scRNA-seq analysis by integrating dropout-aware decomposition, entropy-based differentiation scoring, and graph-based trajectory inference. Its applications span developmental biology, cancer genomics, and regenerative medicine, where resolving lineage hierarchies is paramount.  

**Advantages Over Prior Art**  
1. **Robustness to Dropouts**: wp-NMF outperforms PCA in noisy data.  
2. **Quantitative Plasticity Measure**: Metagene entropy provides the first computational metric for progenitor potential.  
3. **Scalability**: The R implementation processes thousands of cells efficiently.  

**Industrial Applicability**  
- **Stem Cell Therapy**: Identifying potent progenitors for tissue engineering.  
- **Cancer Research**: Tracing clonal evolution in tumors.  
- **Drug Discovery**: Prioritizing pathways regulating lineage commitment.  

---  

This patent application provides a comprehensive description of the dpath pipeline, enabling its legal protection and commercialization. Let me know if you require modifications or additional details.