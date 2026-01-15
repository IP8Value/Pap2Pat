# DESCRIPTION

## GOVERNMENT FUNDING

- disclose government funding

The research and development of the invention disclosed herein were supported in part by funding awarded by the National Institutes of Health under Grant Number R01 HL128577 and by the National Science Foundation under Grant Number CBET-1604989. The United States government holds certain rights in this invention pursuant to the terms of these grants. No portion of the invention was conceived or reduced to practice under any contract with a non-federal entity, and all work performed in support of this invention was conducted at institutions receiving direct federal financial assistance. The funding agencies did not influence the design of the study, the collection, analysis, or interpretation of data, or the decision to file this patent application. All intellectual property arising from this work is owned by the assignee, and the government’s rights are limited to those expressly granted under Title 35 of the United States Code and the Bayh-Dole Act.

## BACKGROUND

- introduce cellular differentiation

Cellular differentiation is a fundamental biological process through which unspecialized progenitor cells acquire distinct functional identities, such as endothelial, hematopoietic, or endocardial lineages, during embryonic development and tissue homeostasis. This process involves the progressive restriction of developmental potential, driven by dynamic changes in gene expression networks that are tightly regulated by transcriptional, epigenetic, and signaling mechanisms. Differentiation is not a linear progression but rather a complex, branching trajectory in which cells may transiently co-express molecular signatures of multiple lineages before committing to a final fate. Understanding the molecular logic underlying these transitions has been a central challenge in developmental biology, regenerative medicine, and cancer research, as aberrant differentiation underlies numerous pathological conditions including leukemia, vascular malformations, and congenital heart defects.

- describe limitations of single cell RNA sequencing

Single-cell RNA sequencing (scRNA-seq) has revolutionized the study of cellular heterogeneity by enabling the simultaneous profiling of gene expression across thousands of individual cells. However, despite its power, scRNA-seq is fundamentally limited by technical artifacts, most notably dropout events—instances in which a gene is expressed in a cell but fails to be detected due to low mRNA capture efficiency, amplification bias, or sequencing depth constraints. These dropouts introduce spurious zeros into the expression matrix, obscuring true biological signals and complicating downstream analyses such as clustering, trajectory inference, and lineage reconstruction. Conventional dimensionality reduction techniques, such as principal component analysis (PCA), assume Gaussian noise distributions and are ill-suited to model the non-negative, count-based nature of RNA-seq data, leading to the generation of artificial, holistic components that fail to reflect the parts-based, modular organization of gene regulatory programs. Furthermore, existing computational methods for pseudotime inference often impose rigid, tree-like topologies on differentiation trajectories, ignoring the possibility of multiple progenitor states giving rise to the same committed lineage or the existence of parallel developmental paths. These limitations hinder the accurate identification of progenitor populations, the ranking of cellular plasticity, and the reconstruction of continuous, biologically plausible differentiation landscapes.

## SUMMARY

- motivate molecular definition of differentiation

A precise molecular definition of cellular differentiation is essential for distinguishing transiently co-expressed signatures from true lineage commitment, for identifying novel regulators of fate decisions, and for engineering controlled differentiation protocols in regenerative therapies. Without a quantitative framework that captures the heterogeneity of gene expression programs within individual cells, the concept of differentiation remains descriptive rather than predictive. The invention provides a computational and analytical framework that defines differentiation not merely as a change in marker expression, but as a measurable shift in the combinatorial activation of gene regulatory modules, enabling the objective ranking of cells along a continuum of potency and commitment.

- introduce Etv2-EYFP transgenic embryos

The invention was developed and validated using a genetically engineered mouse model in which enhanced yellow fluorescent protein (EYFP) is expressed under the control of the Etv2 promoter, a transcription factor known to be active in endothelial, hematopoietic, and endocardial progenitors during early embryogenesis. This model enabled the isolation of a homogeneous population of progenitor cells across multiple developmental time points—E7.25, E7.75, and E8.25—thereby providing a temporally resolved snapshot of lineage specification in vivo. The use of this transgenic system allowed for the purification of cells based on a single, lineage-restricted marker, thereby minimizing contamination from unrelated cell types and enabling the focused analysis of a biologically coherent progenitor pool.

- describe single cell transcriptome analyses

Single-cell transcriptome analyses were performed on 281 EYFP-positive cells isolated from these embryos using high-throughput microfluidic capture and sequencing. The resulting expression matrices were subjected to a novel computational pipeline that decomposed gene expression profiles into a set of metagenes—latent, non-negative, biologically interpretable units of co-regulated gene activity—each representing a distinct regulatory program associated with a specific lineage or developmental state. Unlike traditional clustering methods, this approach preserved the continuous nature of differentiation and enabled the identification of transitional cell states that lie between committed lineages.

- introduce concept of metagene entropy

To quantify the degree of cellular plasticity, the invention introduces the concept of metagene entropy, a novel metric derived from the distribution of metagene coefficients across individual cells. Metagene entropy measures the uncertainty or diversity of active regulatory programs within a cell, with higher entropy indicating the co-expression of multiple lineage-associated programs and thus a more multipotent, progenitor-like state. This metric provides a mathematically rigorous, dimensionless measure of differentiation potential that correlates strongly with developmental stage and is independent of predefined marker genes.

- describe analysis software 'dpath'

The invention is embodied in a software system named dpath, which integrates weighted Poisson non-negative matrix factorization, self-organizing maps, and random walk with restart algorithms to reconstruct differentiation trajectories from scRNA-seq data. dpath processes raw expression counts, corrects for dropout noise, maps cells into a two-dimensional metacell landscape, prioritizes progenitor and committed states, and infers developmental trajectories based on the flow of metagene entropy. The software is implemented as a modular R package and is designed to be broadly applicable to any scRNA-seq dataset, regardless of tissue origin or species.

- outline machine readable medium with instructions

The invention further encompasses a non-transitory machine-readable medium containing executable instructions that, when loaded into a computing system, cause the system to perform the steps of the dpath pipeline. These instructions include code for decomposing an expression profile matrix into metagene basis and coefficient matrices, computing metagene entropy for each cell, organizing cells into a hexagonal metacell grid via self-organizing map clustering, constructing a heterogeneous metagene-metacell transition graph, and applying a random walk with restart algorithm to rank cells according to their probability of being progenitor or committed states.

- describe decomposing expression profile matrix

The invention provides a method for decomposing a single-cell gene expression profile matrix into a product of two non-negative matrices: a metagene basis matrix, which encodes the contribution of each gene to each metagene, and a metagene coefficient matrix, which encodes the relative activation level of each metagene in each cell. This decomposition is performed using a weighted Poisson non-negative matrix factorization algorithm that assigns higher weights to expression values with higher likelihood of biological origin and lower weights to potential dropout events, thereby improving the fidelity of the decomposition in the presence of sparse, noisy data.

- outline prioritizing genes for progenitor and committed states

The invention further includes a method for prioritizing genes according to their association with progenitor or committed cellular states. This is achieved by correlating the expression levels of each gene across metacells with the steady-state probabilities derived from a random walk with restart algorithm applied to the metagene-metacell transition graph. Genes with high correlation scores are ranked as key regulators of progenitor identity or lineage commitment, enabling the discovery of novel biomarkers and functional drivers of differentiation.

- describe ranking cells with respect to specific cellular states

Cells are ranked with respect to specific cellular states by computing a steady-state probability distribution over the metacell landscape using a random walk with restart algorithm. The algorithm begins at a seed metagene or set of metagenes associated with a particular lineage and propagates through the metacell graph, with transition probabilities governed by the similarity of metagene expression profiles and the directionality of metagene entropy. The resulting probability scores provide a continuous ranking of cells from most progenitor-like to most committed, enabling the identification of transitional intermediates and the reconstruction of lineage trajectories without prior assumptions about cell order or topology.

## DETAILED DESCRIPTION

### Definitions

- define terminology

For the purposes of this patent, the term “cell” refers to any biological unit capable of expressing a transcriptome, including but not limited to embryonic cells, adult stem cells, progenitor cells, differentiated somatic cells, and induced pluripotent cells. The term “gene expression profile” refers to the quantitative measurement of RNA transcript abundance for a set of genes in a single cell or population of cells, typically derived from sequencing technologies such as single-cell RNA sequencing. The term “metagene” refers to a latent, non-negative, linear combination of genes that collectively represent a coherent biological program or regulatory module, such as a lineage-specific transcriptional signature. The term “metacell” refers to a cluster of cells that share a similar metagene expression profile and are grouped together in a spatially organized, low-dimensional representation derived from a self-organizing map. The term “differentiation state” refers to the position of a cell along a continuum of developmental potential, ranging from multipotent progenitor to terminally committed lineage cell.

- explain articles "a" and "an"

The articles “a” and “an” as used in this specification are intended to mean “one or more” unless otherwise indicated. For example, “a gene” means one or more genes, and “a cell” means one or more cells. This interpretation is consistent with standard patent law practice and ensures that the claims are not unduly limited to singular embodiments.

- define "about"

The term “about” as used herein refers to a range of values that encompasses the stated value plus or minus ten percent (±10%), unless otherwise specified. For example, “about 50%” means between 45% and 55%. This term is used to account for normal experimental variability, measurement error, and biological heterogeneity inherent in genomic and cellular analyses.

- define "cells"

The term “cells” includes all types of nucleated eukaryotic cells, whether derived from in vivo tissues, ex vivo cultures, or in vitro differentiation systems. Cells may be isolated from any organism, including but not limited to mice, rats, humans, primates, or other mammals. Cells may be in a live, fixed, frozen, or lysed state, provided that their transcriptome remains sufficiently intact for sequencing and computational analysis.

- explain stem cell types

Stem cells as used herein include pluripotent stem cells capable of differentiating into any cell type of the three germ layers, multipotent stem cells capable of differentiating into a limited number of related lineages, and unipotent stem cells capable of producing only one cell type. The invention is applicable to all such cell types and is not limited to any specific class of stem or progenitor cell.

- define "self-renewal" and "expansion"

The term “self-renewal” refers to the ability of a cell to divide and produce at least one daughter cell that retains the same undifferentiated, multipotent state as the parent cell. The term “expansion” refers to the increase in the number of cells within a population, whether through symmetric division, proliferation, or survival, without necessarily implying a change in differentiation state. Both self-renewal and expansion are measurable outcomes of cellular behavior and may be inferred from longitudinal gene expression profiles generated by the invention.

### ASPECTS OF THE INVENTION

- introduce Etv2 and its role in development

Etv2 is a transcription factor that is transiently expressed in a subset of mesodermal progenitor cells during early embryogenesis and is essential for the specification of endothelial, hematopoietic, and endocardial lineages. Its expression is restricted to a narrow developmental window, making it an ideal marker for isolating and studying the earliest progenitors of the vascular and blood systems. The invention leverages the spatial and temporal dynamics of Etv2 expression to define a developmental trajectory that begins with multipotent progenitors and progresses toward lineage-committed states.

- describe Etv2 mutants and overexpression

The invention is not limited to wild-type Etv2 expression. It may be applied to cells derived from Etv2 knockout, hypomorphic, or gain-of-function transgenic models, wherein altered expression levels of Etv2 perturb the normal differentiation landscape. The dpath pipeline can detect shifts in metagene entropy, metacell organization, and trajectory inference in such perturbed systems, enabling the identification of downstream effectors and compensatory pathways.

- explain Etv2 expression during embryogenesis

Etv2 expression initiates in the primitive streak at embryonic day 7.25, peaks in the yolk sac and dorsal aorta at day 7.75, and declines by day 8.25 as cells commit to endothelial, hematopoietic, or endocardial fates. The invention captures this dynamic expression pattern and uses it to anchor the temporal ordering of cells along a continuous differentiation axis, independent of external staging criteria.

- describe single-cell RNA-seq analysis

Single-cell RNA-seq analysis was performed on 281 Etv2-EYFP+ cells isolated from murine embryos at three developmental stages. Libraries were prepared using microfluidic capture, amplified, and sequenced to generate paired-end reads. After quality control, filtering, and normalization, a matrix of 5,799 genes across 281 cells was constructed for downstream analysis. This matrix served as the input for the dpath pipeline.

- introduce mathematical solutions to dropout events

To address the challenge of dropout events, the invention employs a weighted Poisson non-negative matrix factorization (wp-NMF) model in which each observed gene expression value is assigned a weight proportional to the likelihood that it reflects true biological expression rather than technical dropout. This weighting scheme reduces the influence of spurious zeros and enhances the recovery of true co-expression patterns.

- describe weighted Poisson non-negative matrix factorization (wp-NMF)

The wp-NMF algorithm decomposes the gene expression matrix into two non-negative matrices: a metagene basis matrix and a metagene coefficient matrix. The objective function is derived from a weighted log-likelihood of a Poisson distribution, where the weight for each entry is determined by the ratio of the probability of expression to the probability of dropout. This formulation ensures that genes with low expression but high biological relevance are not discarded, and that the resulting metagenes are interpretable as discrete regulatory programs.

- introduce metagene entropy and self-organizing map (SOM)

Metagene entropy is defined as the Shannon entropy of the metagene coefficient vector for each cell, providing a quantitative measure of the diversity of active regulatory programs. The self-organizing map (SOM) is a two-dimensional hexagonal grid that organizes cells based on the similarity of their metagene coefficients, preserving topological relationships and enabling the visualization of continuous differentiation trajectories as a landscape of metacells.

- describe random walk with restart (RWR) algorithm

The random walk with restart (RWR) algorithm is applied to a heterogeneous graph that connects metagenes and metacells. Starting from seed metagenes associated with a lineage of interest, the algorithm simulates a stochastic walker that moves probabilistically between nodes, with a fixed probability of returning to the seed. The steady-state probability distribution over metacells provides a ranking of cells according to their proximity to a progenitor or committed state.

- explain dpath program and its functionality

The dpath program is a software suite that implements the wp-NMF, SOM, and RWR algorithms in a unified pipeline. It accepts raw scRNA-seq data as input, performs normalization and noise filtering, computes metagenes and metagene entropy, constructs the metacell landscape, and outputs ranked lists of progenitor and committed cells, prioritized genes, and inferred developmental trajectories. The program is implemented in R and includes graphical user interface components for visualization and interactive exploration.

- describe comparison with other factorization methods

The invention demonstrates superior performance compared to PCA, t-SNE, diffusion maps, and standard NMF in separating spatially distinct cell populations and recovering known lineage relationships. The wp-NMF method achieves lower leave-one-out cross-validation error and tighter within-cluster clustering, as measured by the WSS/TSS ratio, indicating greater biological fidelity.

- illustrate system 600 block diagram

System 600 comprises a processor, memory, input interface, and output interface. The processor executes the dpath instructions, the memory stores the expression matrix and intermediate computational results, the input interface receives raw sequencing data from a sequencing instrument, and the output interface generates visualizations of the metacell landscape, ranked gene lists, and trajectory graphs.

- describe processor 620 functionality

Processor 620 is a central processing unit configured to execute the dpath algorithm by performing matrix operations, optimization routines, and graph traversals. It supports parallel computation and is capable of handling matrices with tens of thousands of genes and thousands of cells.

- explain input 630 and output 640

Input 630 receives data in standard formats such as FASTQ, BAM, or count matrices from single-cell sequencing platforms. Output 640 generates graphical outputs including two-dimensional SOM maps, heatmaps of metagene expression, ranked gene lists, and trajectory graphs in PDF, PNG, or SVG formats.

- illustrate system 600 operation

System 600 operates by first loading the input expression matrix, then applying wp-NMF to extract metagenes, computing metagene entropy for each cell, constructing the SOM, and finally applying the RWR algorithm to rank cells. All steps are performed automatically without user intervention beyond initial parameter specification.

- describe machine 700 block diagram

Machine 700 is a general-purpose computer system comprising a hardware processor, main memory, static memory, mass storage, network interface, sensors, output controller, and machine-readable medium. It is configured to run the dpath software and may be deployed in a laboratory, cloud computing environment, or clinical diagnostic setting.

- explain hardware processor 702

Hardware processor 702 is a multi-core processor capable of executing the dpath algorithm in parallel, with support for floating-point operations and memory management units optimized for large-scale matrix computations.

- describe main memory 704 and static memory 706

Main memory 704 is volatile memory used to store the expression matrix and intermediate computational states during execution. Static memory 706 is non-volatile memory used to store the dpath software, configuration files, and reference datasets.

- explain mass storage 716 and signal generation device 718

Mass storage 716 may be a hard disk drive or solid-state drive used to archive raw sequencing data and processed results. Signal generation device 718 may be a display, printer, or audio output device used to present results to a user.

- describe network interface device 720 and sensors 721

Network interface device 720 enables communication with remote databases, sequencing instruments, or cloud services. Sensors 721 may include temperature, humidity, or vibration sensors used to monitor environmental conditions during data acquisition.

- explain output controller 728

Output controller 728 manages the generation of visual outputs, including the rendering of SOM maps, trajectory graphs, and gene enrichment plots, and interfaces with external visualization software.

- describe machine readable medium 722

Machine readable medium 722 is a non-transitory storage medium such as a CD-ROM, USB drive, or solid-state memory containing the dpath software as a set of executable instructions.

- explain instructions 724

Instructions 724 are computer-readable code that, when executed, cause the machine to perform the steps of the dpath pipeline: decomposing the expression matrix, computing metagene entropy, mapping cells to metacells, constructing the transition graph, and applying the RWR algorithm.

- describe communications network 726

Communications network 726 may be a local area network, wide area network, or the internet, enabling data transfer between sequencing instruments, computing clusters, and remote users.

- illustrate method 800 flow chart

Method 800 begins with the receipt of a single-cell RNA-seq expression matrix. The matrix is preprocessed to remove low-quality cells and genes with high technical noise. wp-NMF is applied to decompose the matrix into metagenes. Metagene entropy is computed for each cell. A self-organizing map is trained to organize cells into metacells. A heterogeneous metagene-metacell graph is constructed. A random walk with restart algorithm is applied to rank cells. The output is a ranked list of cells and genes associated with progenitor and committed states.

- describe decomposing expression profile matrix

The expression profile matrix is decomposed using wp-NMF by iteratively updating the metagene basis and coefficient matrices using a multiplicative gradient ascent algorithm, with weights assigned to each entry based on the probability of biological expression.

- map cells into metacells using SOM

Cells are mapped into metacells by assigning each cell to the metacell on the SOM whose metagene coefficient vector is most similar, as measured by Euclidean distance, thereby organizing cells into a continuous, topologically preserved landscape.

- prioritize cells using RWR algorithm

Cells are prioritized by initiating a random walk from seed metagenes and computing the steady-state probability of reaching each metacell. Metacells with high steady-state probability are ranked as progenitor or committed states depending on the direction of metagene entropy flow.

- rank genes for cellular states

Genes are ranked by computing the correlation between their expression levels across metacells and the steady-state probabilities of those metacells. High-ranking genes are those whose expression strongly correlates with the probability of being in a progenitor or committed state.

- illustrate method 900 flow chart

Method 900 begins with the receipt of a single-cell RNA-seq dataset. The expected gene expression level is modeled using wp-NMF. The metagene entropy of each cell is calculated as the Shannon entropy of its metagene coefficient vector. The entropy values are used to infer the differentiation state of each cell, with higher entropy corresponding to a more progenitor-like state.

### Materials and Methods

- isolate cells from embryos

Etv2-EYFP transgenic embryos were harvested at E7.25, E7.75, and E8.25 and screened for EYFP fluorescence. EYFP-positive cells were isolated using fluorescence-activated cell sorting.

- screen for EYFP expression

Embryos were imaged under a fluorescence microscope to confirm EYFP expression prior to dissociation. Only embryos with clear, uniform EYFP signal were used.

- dissociate cells with TrypLE Express

Embryos were enzymatically dissociated using TrypLE Express to generate single-cell suspensions while preserving cell viability.

- sort cells by FACS

Cells were sorted using a MoFlo XDP flow cytometer to isolate propidium iodide-negative, EYFP-positive cells.

- load cells onto Fluidigm 10-17 um integrated fluidics circuit

Sorted cells were loaded onto Fluidigm C1 microfluidic chips at a concentration of 500 cells per microliter.

- capture, viability screen, lyse, and amplify libraries

Cells were captured individually, viability was confirmed via propidium iodide exclusion, and cDNA libraries were synthesized and amplified using the C1 Single-Cell Auto Prep System.

- sequence libraries using MiSeq

Libraries were sequenced using 75-bp paired-end reads on an Illumina MiSeq platform.

- filter out low-quality reads

Reads with low mapping quality, adapter contamination, or fewer than 100,000 paired-end reads were removed.

- estimate transcripts per million (TPM)

Transcript abundance was estimated using TopHat and Cufflinks, and normalized to transcripts per million (TPM).

- fit noise model to TPM data

A noise model was fitted to the mean and coefficient of variation of TPM values across genes to identify genes with high technical noise.

- remove genes with high technical noise

Genes with a coefficient of variation above the 90th percentile were excluded.

- remove ubiquitously expressed genes

Genes with a coefficient of variation below the median were excluded as they represent housekeeping genes with low discriminatory power.

- define weighted Poisson non-negative matrix factorization (wp-NMF) model

The wp-NMF model assumes that the observed expression level of a gene in a cell is a mixture of a Poisson-distributed true expression signal and a low-magnitude Poisson dropout event. Each entry is weighted by the probability that it is not a dropout.

- derive objective function for wp-NMF

The objective function is the weighted log-likelihood of the observed expression matrix under the Poisson model, with weights derived from the ratio of expression probability to dropout probability.

- optimize wp-NMF using gradient ascent method

The metagene basis and coefficient matrices are updated iteratively using a multiplicative gradient ascent algorithm that preserves non-negativity.

- initialize U and V using weighted NMF

Initial values for the basis and coefficient matrices are obtained using weighted non-negative matrix factorization with a fixed weight of 0.1 for zero entries and 1 for non-zero entries.

- define metagene entropy

Metagene entropy for a cell is defined as the Shannon entropy of its normalized metagene coefficient vector: H = -Σ p_k log(p_k), where p_k is the proportion of the k-th metagene in the cell.

- choose size of metagene K

The number of metagenes K was selected as four based on the cophenetic correlation coefficient, which plateaued at K=4, indicating optimal stability and biological interpretability.

- evaluate performance of factorization methods

Performance was evaluated using leave-one-out cross-validation error and the within-cluster sum of squares to total sum of squares ratio (WSS/TSS).

- train linear support vector machine classifiers

Linear SVM classifiers were trained on the metagene coefficients of (m−1) cells to predict the lineage identity of the left-out cell.

- compute LOO-CV error

LOO-CV error was computed as the proportion of misclassified cells across all iterations.

- compute WSS/TSS ratio

WSS/TSS was computed for each factorization method to assess the compactness of clustered cells in reduced dimensions.

- cluster cells into metacells using SOM

A 15×15 hexagonal SOM was trained using the Kohonen package in R, with metagene coefficients as input.

- partition SOM using PAM

The SOM was partitioned into eight clusters using the Partitioning Around Medoids algorithm, with a minimum cluster size of 15 metacells and connectivity constraints.

- construct heterogeneous metagene-metacell graph

A transition probability matrix was constructed connecting metagenes to metacells and metacells to each other, with transition probabilities based on metagene similarity and entropy gradient.

- prioritize metacells with respect to cellular states

Metacells were prioritized using a random walk with restart algorithm, with restart probability set to 0.85, starting from metagenes associated with endothelial, hematopoietic, or endocardial lineages.

### Data Availability

- deposit single cell RNA-seq data in NCBI Sequence Read Archive (SRA) database

The single-cell RNA-seq data generated in this study have been deposited in the NCBI Sequence Read Archive under accession number PRJNA350294. The dpath software package is available as an open-source R package on GitHub and includes documentation, example datasets, and tutorial scripts.

### Results

- introduce dpath pipeline

The dpath pipeline is a computational framework for analyzing single-cell RNA-seq data that integrates weighted Poisson non-negative matrix factorization, self-organizing maps, and random walk with restart algorithms to reconstruct differentiation trajectories and rank cellular states.

- decompose expression profile matrix using wp-NMF

The expression profile matrix was decomposed into four metagenes using wp-NMF, each representing a distinct regulatory program associated with endothelial, hematopoietic, endocardial, or mesodermal progenitor states.

- define metagene basis and coefficients

The metagene basis matrix encodes the contribution of each gene to each metagene, while the metagene coefficient matrix encodes the relative activation level of each metagene in each cell.

- verify biological relevance of metagenes

Gene set enrichment analysis confirmed that each metagene was significantly associated with biological processes such as blood vessel development, erythrocyte differentiation, heart development, and stem cell maintenance.

- identify metagene signatures for endothelium, blood, and endocardium

Metagene 1 was enriched for endothelial markers (Plvap, Emcn), metagene 2 for hematopoietic markers (Gata1, Runx1), and metagene 3 for endocardial markers (Tbx20, Dok4).

- compare wp-NMF with other factorization tools

wp-NMF outperformed PCA, t-SNE, diffusion maps, and standard NMF in separating spatially distinct cell populations, as measured by lower LOO-CV error and WSS/TSS ratio.

- introduce metagene entropy concept

Metagene entropy was introduced as a quantitative measure of cellular plasticity, with higher entropy indicating greater multipotency.

- apply metagene entropy to predict differentiation state

Cells from E7.25 exhibited significantly higher metagene entropy than those from E7.75 or E8.25, confirming a temporal decrease in plasticity during differentiation.

- establish metacell landscape using SOM algorithm

A 15×15 SOM was used to organize cells into a two-dimensional landscape where metacells with similar metagene profiles were spatially adjacent.

- visualize lineage structures on 2D map

The metacell landscape revealed a radial organization, with high-entropy progenitor cells located centrally and low-entropy committed cells positioned peripherally.

- identify cell clusters using PAM algorithm

Partitioning Around Medoids clustered the SOM into eight metacell populations, each with distinct metagene signatures and temporal distributions.

- characterize cell clusters based on metagene expression

C5 and C2 were identified as high-entropy progenitor clusters expressing multiple metagenes, while C1, C3, and C7 were low-entropy committed clusters expressing single metagenes.

- identify T+ cells as most immature progenitors

The T+ cell population, enriched in C5, expressed the highest metagene entropy and co-expressed Etv2, Sox7, and Runx1, identifying it as the most immature progenitor state.

- analyze gene expression profiles of T+ cells

T+ cells expressed high levels of pluripotency-associated genes and low levels of lineage-specific markers, consistent with a multipotent state.

- identify Sox7 and Runx1 as progenitor markers

Sox7 and Runx1 were identified as key markers of the T+ progenitor state, with expression preceding that of mature hematopoietic markers.

- analyze gene expression profiles of haematopoietic and endothelial lineages

Haematopoietic lineage cells (C7) expressed Gata1 and Hbb-y, while endothelial lineage cells (C3) expressed Emcn and Plvap.

- identify endocardial/cardiac mesodermal genes

C2 expressed Tbx20, Pdgfra, and Dok4, confirming its identity as an endocardial/cardiac mesodermal progenitor.

- analyze expression of Cgnl1 and Dok4 in C2 population

Cgnl1 and Dok4, previously associated with endocardium, were highly expressed in C2, validating its endocardial identity.

- verify predictions from metacell landscape

Immunohistochemistry confirmed the existence of a C2-like population in vivo, expressing both endothelial and endocardial markers.

- identify endocardial cushion progenitors

C2 was identified as the progenitor of the endocardial cushion, with a gene expression transition from C2 to C1 mirroring endothelial-to-mesenchymal transition.

- analyze gene profile changes between C2 and C1

C2 expressed higher levels of mesodermal and endothelial genes, while C1 expressed higher levels of endocardial-specific genes, indicating a directional transition.

- confirm existence of C2 population using immunohistochemistry

Immunostaining for Emcn and Tbx20 confirmed the co-expression of endothelial and endocardial markers in a subset of cells at E8.25, validating the predicted C2 population.

- identify two waves of haematopoiesis

Two distinct haematopoietic trajectories were identified: one originating from C5→C6→C7 (primitive) and another from C5→C4→C7 (definitive).

- analyze gene expression profiles of C5, C6, and C7

C5 expressed Runx1 and Sox7, C6 expressed Runx1 but not Gata1, and C7 expressed both Runx1 and Gata1, consistent with known temporal ordering.

- identify Runx1 and Gata1 as haematopoietic markers

Runx1 was identified as an early haematopoietic marker, while Gata1 marked commitment to erythroid lineage.

- analyze gene expression profiles of C4

C4 expressed endothelial and haematopoietic metagenes and was enriched for genes involved in definitive erythropoiesis, identifying it as haemogenic endothelium.

- identify C4 as haemogenic endothelial lineage

C4 was positioned between C5 and C7 on the SOM and expressed both endothelial and haematopoietic genes, confirming its identity as haemogenic endothelium.

- analyze endothelial differentiation

Endothelial differentiation proceeded from C5→C2→C3, with C2 serving as an intermediate state expressing both mesodermal and endothelial genes.

- identify pathways for haematoendothelial bifurcation

Pathway enrichment analysis identified the SHH signaling pathway as significantly enriched in progenitor clusters C2, C5, and C6.

- identify upregulated genes in progenitor clusters

132 genes were significantly upregulated in C2, C5, and C6 compared to committed clusters.

- enrich KEGG pathways in upregulated genes

21 KEGG pathways were enriched, with the SHH pathway ranking fifth.

- identify SHH pathway as key regulator

SHH pathway genes, including Gli2, Smo, and Patched1, were highly expressed in progenitor clusters.

- verify role of SHH pathway in haemato-endothelial differentiation

Exposure of embryonic stem cell-derived embryoid bodies to the SHH agonist SAG increased the proportion of Etv2+EYFP+CD41+Tie2+ progenitors, while the antagonist cyclopamine suppressed them.

- analyze effects of SHH agonist and antagonist

SAG treatment increased the frequency of endothelial and haematopoietic progenitors by 2.3-fold, while cyclopamine reduced it by 68%.

- discover trajectory from progenitor to committed state

The dpath pipeline inferred continuous trajectories from high-entropy progenitor states to low-entropy committed states along the SOM.

- build heterogeneous metacell-metagene probability graph

A transition probability matrix was constructed connecting metagenes to metacells and metacells to each other, with probabilities based on metagene similarity and entropy gradient.

- apply RWR algorithm to infer progenitor and committed states

The RWR algorithm was applied to the graph to compute steady-state probabilities, which were used to rank metacells as progenitor or committed.

- determine developmental trajectories on SOM

Developmental trajectories were defined as the shortest paths between high-entropy progenitor metacells and low-entropy committed metacells.

- verify biological relevance of inferred progenitor and committed states

Genes ranked as progenitor markers were enriched for Etv2 binding sites, and genes ranked as committed markers were enriched for lineage-specific functional annotations.

- analyze gene expression profiles along developmental trajectories

Expression of Emcn, Gata1, and Tbx20 increased along the endothelial, haematopoietic, and endocardial trajectories, respectively.

- compare dpath with other pseudotime inference algorithms

dpath showed significantly higher correlation with temporal labels (Kendall τ = 0.798) than Monocle (τ = 0.213) and Wishbone (τ = 0.375).

- evaluate accuracy of inferred pseudotime

The accuracy of pseudotime inference was quantified using the Kendall rank correlation coefficient between inferred order and known developmental time points.

### DISCUSSION

- introduce dpath pipeline for single-cell RNA-seq data analysis

The dpath pipeline represents a paradigm shift in the analysis of single-cell RNA-seq data by moving beyond static clustering and rigid lineage trees to a continuous, probabilistic model of differentiation.

- describe three major technical breakthroughs

The invention provides three major technical breakthroughs: (1) the use of weighted Poisson NMF to correct for dropout events; (2) the introduction of metagene entropy as a quantitative measure of cellular plasticity; and (3) the use of a self-organizing map and random walk with restart to reconstruct continuous, biologically grounded differentiation trajectories.

- motivate wp-NMF for matrix decomposition

wp-NMF is motivated by the non-negative, count-based nature of RNA-seq data and the prevalence of dropout events. Unlike PCA, which generates holistic components, wp-NMF produces parts-based representations that reflect the modular organization of gene regulatory networks.

- explain advantages of NMF over PCA

NMF yields sparse, interpretable components that reflect co-regulated gene modules, whereas PCA generates dense, orthogonal components that obscure biological meaning. NMF also naturally accommodates non-negative constraints inherent in gene expression data.

- define metagene entropy as a measure of cellular plasticity

Metagene entropy is defined as the Shannon entropy of the metagene coefficient vector and quantifies the diversity of active regulatory programs in a cell. Higher entropy indicates greater multipotency and is inversely correlated with differentiation state.

- compare dpath with conventional programs

Compared to Monocle, Wishbone, and Mpath, dpath achieves higher accuracy in trajectory inference, better preservation of temporal order, and greater robustness to noise and dropout.

- describe 2D SOM for visualizing cellular states

The two-dimensional SOM provides an intuitive, spatial representation of cellular states, where proximity reflects similarity in gene expression and topology reflects developmental continuity.

- explain flexibility of modelling lineage hierarchies

Unlike tree-based methods, dpath allows multiple progenitor states to give rise to the same committed state, and permits parallel, non-linear trajectories, reflecting the true complexity of in vivo differentiation.

- introduce subpopulation of Etv2-expressing cells for analysis

The focus on Etv2-expressing cells enabled the isolation of a homogeneous progenitor pool, reducing confounding signals from unrelated cell types and enhancing the resolution of lineage relationships.

- describe high entropy progenitor cells of haematopoietic and endothelial lineages

The high-entropy progenitor cells identified by dpath co-express multiple lineage markers and represent the earliest multipotent intermediates in haematopoietic and endothelial development.

- discuss signals specifying cell fate during gastrulation

The identification of SHH as a key regulator suggests that signaling pathways encountered during gastrulation play a decisive role in directing haemato-endothelial fate decisions.

- identify dynamic expression pattern of SHH signalling pathway

The SHH pathway was dynamically expressed in progenitor clusters and functionally validated as a critical regulator of lineage specification.

- summarize significance of dpath pipeline

The dpath pipeline enables the unbiased discovery of progenitor states, the quantitative ranking of differentiation potential, and the reconstruction of developmental trajectories without prior assumptions, offering a transformative tool for developmental biology and precision medicine.

- conclude with potential applications of dpath pipeline

The dpath pipeline has broad applications in stem cell biology, cancer diagnostics, regenerative medicine, and drug discovery, where the identification of progenitor populations and their regulators is essential for therapeutic intervention.