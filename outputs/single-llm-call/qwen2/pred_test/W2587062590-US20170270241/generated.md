# DESCRIPTION

## GOVERNMENT FUNDING

This invention was made with government support under [Grant Number] awarded by [Funding Agency]. The government has certain rights in the invention.

## BACKGROUND

Single-cell RNA sequencing (scRNA-seq) has revolutionized the field of developmental biology by enabling the detailed characterization of gene expression at the single-cell level. However, the analysis of scRNA-seq data is challenging due to the high dimensionality and sparsity of the data, as well as the presence of dropout events, which are common in single-cell experiments. Traditional methods for dimensionality reduction and clustering, such as principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE), often fail to accurately capture the complex biological processes underlying cellular differentiation and lineage commitment.

The invention described herein addresses these challenges by providing a novel computational pipeline, referred to as "dpath," which is specifically designed to analyze scRNA-seq data. The dpath pipeline decomposes the expression profile matrix of single-cell RNA-seq into metagenes using weighted Poisson non-negative matrix factorization (wp-NMF). It then maps cells into metacells using a self-organizing map (SOM) algorithm and prioritizes cells with respect to specific cellular states using a random walk with restart (RWR) algorithm on a heterogeneous metagene–metacell graph. Finally, the pipeline ranks genes for cellular states according to their expression profiles.

## SUMMARY

The present invention provides a method and system for analyzing single-cell RNA sequencing (scRNA-seq) data to identify and prioritize cellular states and gene expression patterns. The method includes the following steps:

1. Decomposing the expression profile matrix of scRNA-seq data into metagenes using weighted Poisson non-negative matrix factorization (wp-NMF).
2. Mapping cells into metacells using a self-organizing map (SOM) algorithm.
3. Prioritizing cells with respect to specific cellular states using a random walk with restart (RWR) algorithm on a heterogeneous metagene–metacell graph.
4. Ranking genes for cellular states according to their expression profiles.

The invention further provides a computer-readable medium containing instructions for performing the method and a system for implementing the method.

## DETAILED DESCRIPTION

### Definitions

- **Metagene**: A latent variable representing a combination of genes that captures the expression patterns of a specific cellular state or lineage.
- **Metacell**: A group of cells with similar metagene expression profiles, organized on a two-dimensional (2D) hexagonal grid using a self-organizing map (SOM) algorithm.
- **Weighted Poisson Non-Negative Matrix Factorization (wp-NMF)**: A method for decomposing the expression profile matrix of scRNA-seq data into metagenes, accounting for dropout events by using a weighted Poisson model.
- **Random Walk with Restart (RWR)**: An algorithm used to prioritize cells with respect to specific cellular states by simulating a random walk on a graph.
- **Entropy**: A measure of disorder or uncertainty, used in the context of this invention to quantify the differentiation potential of a cell based on its metagene expression profile.

### ASPECTS OF THE INVENTION

The invention provides a comprehensive pipeline for analyzing scRNA-seq data, which includes the following aspects:

1. **Decomposition of Expression Profiles**:
   - The expression profile matrix of scRNA-seq data is decomposed into metagenes using wp-NMF. This step accounts for dropout events by assigning different weights to each entry in the matrix based on the likelihood of being a dropout event.
   - The wp-NMF method models the expected gene expression level as a linear combination of non-negative metagene basis and coefficients, providing a parts-based representation of gene expression profiles.

2. **Mapping Cells into Metacells**:
   - The cells are mapped into metacells using a SOM algorithm. The SOM organizes the cells on a 2D hexagonal grid, where neighboring metacells have similar metagene expression profiles.
   - The SOM provides an intuitive way to visualize the distribution of cellular states and reduces the impact of dominant lineages in the analysis.

3. **Prioritizing Cellular States**:
   - A RWR algorithm is used to prioritize cells with respect to specific cellular states. The algorithm simulates a random walk on a heterogeneous metagene–metacell graph, where the transition probabilities are based on the metagene coefficients and the metagene entropy.
   - The metagene entropy is used as a measure of the differentiation potential of a cell, with higher entropy indicating a higher level of cellular plasticity.

4. **Ranking Genes for Cellular States**:
   - Genes are ranked for specific cellular states based on the correlation between their expression levels in metacells and the steady-state probabilities obtained from the RWR algorithm.
   - The enrichment score of a gene is defined as the sum of steady-state probabilities, weighted by the observed expression levels.

### Materials and Methods

#### Cell Isolation
Etv2-EYFP embryos were harvested from time-mated females at E7.25, E7.75, or E8.25 and screened using microscopy for EYFP expression. Embryos were dissociated with TrypLE Express, and EYFP-positive cells were sorted by FACS. FACS-sorted cells were loaded onto a Fluidigm 10–17 μm integrated fluidics circuit for capture, viability screening, lysis, and library amplification.

#### Single-Cell RNA-Seq
Libraries were analyzed for cDNA content, and wells containing a single viable cell with at least 0.2 ng/μL cDNA were chosen for sequencing. Sequencing was performed using 75-bp paired-end reads on a MiSeq, generating 202K–1,910K paired-end reads per cell. Cells with fewer than 100K paired reads were removed, resulting in 281 cells for analysis. Transcripts per million (TPM) estimates were obtained using TopHat and Cufflinks. Genes with high technical noise and ubiquitously expressed genes were removed, leaving 5,799 genes for further analysis.

#### Weighted Poisson Non-Negative Matrix Factorization
The expression profile matrix Xnm was decomposed into metagenes using wp-NMF. The expected log-transformed TPM μnm was modeled as a linear combination of non-negative metagene basis Unk and coefficients Vkm. The weighted log-likelihood function was used to account for dropout events, and the objective function was optimized using a gradient ascent method and multiplicative rules.

#### Clustering Cells into Metacells Using a SOM
Cells were mapped into metacells using a SOM algorithm. The input space for the SOM was the mean metagene expression profiles from multiple runs of wp-NMF. The R package kohonen was used to fit the SOM model, and the metacells were organized on a 15 × 15 2D hexagonal grid.

#### Partitioning SOM Using PAM
The SOM was partitioned into multiple segments using the Partitioning Around Medoids (PAM) algorithm. The number of desired clusters could be specified, or the SOM could be partitioned into the maximum number of clusters, ensuring that each metacell cluster was connected on the SOM.

#### Constructing a Heterogeneous Metagene–Metacell Graph
A transition probability matrix was used to characterize the hierarchical relationships among metacells and between metacells and metagenes. The transition probability matrix was defined based on the metagene coefficients and the metagene entropy. The RWR algorithm was used to prioritize metacells with respect to specific cellular states.

#### Prioritizing Metacells with Respect to Cellular States
The RWR algorithm was used to prioritize metacells with respect to specific cellular states. The random walker started from the vertex representing the metagene(s) and moved to neighboring metacells based on the transition probabilities. The steady-state probabilities were used to rank the metacells.

#### Gene Enrichment Score
Genes were ranked for specific cellular states based on the correlation between their expression levels in metacells and the steady-state probabilities obtained from the RWR algorithm. The enrichment score of a gene was defined as the sum of steady-state probabilities, weighted by the observed expression levels.

### Data Availability

The single-cell RNA-seq data supporting the findings of this study have been deposited in the NCBI Sequence Read Archive database with the project accession number PRJNA350294. The dpath pipeline is implemented as an R package and is available for download. All other relevant data are available from the authors upon request.

### Results

The dpath pipeline was applied to scRNA-seq data from 281 Etv2-EYFP+ cells captured from E7.25, E7.75, and E8.25. The pipeline successfully decomposed the expression profile matrix into four metagenes, which represented the endothelial, haematopoietic, endocardial, and mesodermal lineages. The metagene coefficients were used to map the cells into metacells on a 15 × 15 2D hexagonal grid, and the metacell landscape was visualized to reveal the lineage relationships.

The RWR algorithm was used to prioritize cells with respect to specific cellular states, and the metagene entropy was used to quantify the differentiation potential of the cells. The pipeline accurately predicted the developmental trajectories from progenitor to committed states for the endothelial, haematopoietic, and endocardial lineages. The gene expression patterns and pathway enrichment analysis validated the biological significance of the predicted trajectories.

### DISCUSSION

The dpath pipeline provides a powerful tool for analyzing scRNA-seq data and identifying the hierarchical relationships of cellular states. The use of wp-NMF for decomposing the expression profile matrix, the SOM for mapping cells into metacells, and the RWR algorithm for prioritizing cellular states are novel contributions to the field of single-cell analysis. The pipeline has been successfully applied to scRNA-seq data from Etv2-EYFP+ cells, revealing the dynamic expression patterns of genes and the developmental trajectories of haematopoietic, endothelial, and endocardial lineages.

The dpath pipeline is particularly useful for identifying progenitor and committed states, as well as for ranking genes for specific cellular states. The method has been validated using publicly available scRNA-seq data sets and has been shown to outperform existing methods in terms of accuracy and resolution. The availability of the dpath pipeline as an R package will enable researchers to apply this method to a wide range of scRNA-seq data sets, facilitating the discovery of new biological insights and the development of novel therapeutic strategies.