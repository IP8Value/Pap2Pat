# DESCRIPTION

## FEDERAL FUNDING ACKNOWLEDGEMENT

This invention was made with government support under National Institutes of Health/National Cancer Institute grants U24 CA143882, R01 CA170550, U01 CA184826, U24 CA 210969, and National Institutes of Health/National Human Genome Research Institute grant R01 HG006705. The government has certain rights in the invention.

## FIELD OF THE INVENTION

The present invention relates to the field of molecular biology and epigenetics, particularly to methods and systems for identifying and characterizing partially methylated domains (PMDs) and highly methylated domains (HMDs) in the genome. More specifically, the invention provides a novel approach to predict and analyze DNA hypomethylation in large genomic regions, particularly focusing on solo-WCGW CpGs, which are highly susceptible to hypomethylation.

## BACKGROUND

DNA methylation is a critical epigenetic modification that plays a crucial role in gene regulation, genomic stability, and cellular differentiation. Aberrant DNA methylation patterns, particularly hypomethylation, are associated with various diseases, including cancer. Partially methylated domains (PMDs) and highly methylated domains (HMDs) are large genomic regions with distinct methylation states. PMDs are characterized by lower DNA methylation levels compared to HMDs. Understanding the mechanisms and patterns of DNA methylation in these domains is essential for elucidating the molecular basis of diseases and developing targeted therapies.

Previous studies have suggested that DNA methylation is influenced by local sequence context, including local CpG density and the nucleotides flanking the CpG dinucleotide. Specifically, CpGs flanked by adenine (A) or thymine (T) on both sides (WCGW tetranucleotides) are more prone to hypomethylation. Additionally, the timing of DNA replication and the presence of the histone mark H3K36me3 are known to affect DNA methylation levels.

However, current methods for identifying and characterizing PMDs and HMDs are limited in their ability to accurately predict hypomethylation-prone regions across different cell types and developmental stages. There is a need for a robust and comprehensive approach to identify and analyze PMDs and HMDs, particularly focusing on solo-WCGW CpGs, which are highly susceptible to hypomethylation.

## SUMMARY OF THE INVENTION

The present invention provides a method for identifying and characterizing partially methylated domains (PMDs) and highly methylated domains (HMDs) in the genome. The method includes the following steps:

1. **Sequencing**: Performing whole-genome bisulfite sequencing (WGBS) on a biological sample to obtain methylation data.
2. **Data Processing**: Aligning the sequencing reads to a reference genome and extracting methylation levels for each CpG site.
3. **Identifying Solo-WCGW CpGs**: Defining solo-WCGW CpGs as those with no neighboring CpGs within a specified window and flanked by adenine (A) or thymine (T) on both sides.
4. **Defining PMDs and HMDs**: Using a statistical method, such as a Gaussian mixture model, to classify genomic regions into PMDs and HMDs based on the standard deviation (SD) of methylation levels of solo-WCGW CpGs.
5. **Analyzing Replication Timing and H3K36me3**: Correlating the methylation levels of solo-WCGW CpGs with replication timing and the presence of the H3K36me3 histone mark to understand the factors influencing methylation.
6. **Age-Related Hypomethylation**: Investigating the association between the degree of PMD hypomethylation and donor age to identify age-related changes in DNA methylation.

The invention further provides a system for implementing the method, including a computer-readable medium storing instructions for performing the steps of the method and a processor for executing the instructions.

## DETAILED DESCRIPTION OF THE INVENTION

### Terms (Definitions)

- **Whole-Genome Bisulfite Sequencing (WGBS)**: A high-throughput sequencing technique that converts unmethylated cytosines to uracil, allowing the identification of methylated cytosines in the genome.
- **Solo-WCGW CpGs**: CpG dinucleotides with no neighboring CpGs within a specified window and flanked by adenine (A) or thymine (T) on both sides.
- **Partially Methylated Domains (PMDs)**: Large genomic regions with lower DNA methylation levels compared to the surrounding regions.
- **Highly Methylated Domains (HMDs)**: Large genomic regions with higher DNA methylation levels compared to the surrounding regions.
- **Standard Deviation (SD)**: A measure of the amount of variation or dispersion of a set of values.
- **Replication Timing**: The timing of DNA replication during the S phase of the cell cycle.
- **H3K36me3**: A histone mark characterized by trimethylation of lysine 36 on histone H3, which is associated with actively transcribed gene bodies.

### Example 1

**Sequencing and Data Processing**

A biological sample, such as a tumor or normal tissue, is subjected to whole-genome bisulfite sequencing (WGBS) to obtain methylation data. The sequencing reads are aligned to the human reference genome (GRCh37) using a suitable alignment tool, such as BSmap. Methylation levels for each CpG site are extracted using a tool like Bis-SNP, which distinguishes between C-to-T mutations and bisulfite conversion by examining the complementary strand. CpGs with fewer than 10 reads' coverage are excluded from the analysis.

### Example 2

**Identifying Solo-WCGW CpGs**

From the methylation data, solo-WCGW CpGs are defined as those with no neighboring CpGs within a window of ±35 bp and flanked by adenine (A) or thymine (T) on both sides. These CpGs are highly susceptible to hypomethylation and provide a sensitive marker for identifying PMDs and HMDs.

### Example 3

**Defining PMDs and HMDs**

Using a Gaussian mixture model, genomic regions are classified into PMDs and HMDs based on the standard deviation (SD) of methylation levels of solo-WCGW CpGs. The model assumes two subpopulations of 100-kb bins—those located in PMDs with higher cross-sample SDs and those located in HMDs with lower cross-sample SDs. The final threshold for classifying PMDs from HMDs is determined to be 0.125. The more conservative sets of "common PMDs" and "common HMDs" are defined by the criteria that SD > 0.15 and SD < 0.10, respectively.

### Example 4

**Analyzing Replication Timing and H3K36me3**

The methylation levels of solo-WCGW CpGs are correlated with replication timing and the presence of the H3K36me3 histone mark to understand the factors influencing methylation. Replication timing data is obtained from wavelet-transformed signals, and H3K36me3 data is quantified using the percentage of base overlaps of each window with gapped peaks. The analysis reveals that solo-WCGW CpGs overlapping H3K36me3 are highly methylated, irrespective of replication timing, while those not overlapping H3K36me3 are strongly replication-timing dependent.

### Example 5

**Age-Related Hypomethylation**

The degree of PMD hypomethylation is investigated in different primary cell types to identify age-related changes in DNA methylation. Methylation levels of solo-WCGW CpGs within common PMDs are compared between samples from newborns and elderly donors. The analysis shows a strong association between the degree of PMD hypomethylation and donor age, with older samples exhibiting more extensive hypomethylation.

### Example 6

**Application in Low Coverage and Single-Cell Studies**

The method is applied to low coverage and single-cell WGBS data to demonstrate its utility in these settings. Down-sampled bulk WGBS data and low-coverage single-cell WGBS data are analyzed using the same approach, and the results show that solo-WCGW CpGs allow accurate PMD structure to be determined with average genomic read coverage as low as 0.05×.

### Example 7

**Comparative Analysis Across Developmental Lineages**

The method is used to compare PMD structure across different developmental lineages, including gametes, early developmental stages, and post-natal tissues. The analysis reveals a progressive emergence of PMD/HMD structure along organismal development, with a substantial degree of similarity between PMD structure in brain tissues and other lineages.

### Example 8

**Link to Mitotic Cell Division in Cancer**

The method is applied to a large cohort of tumors from The Cancer Genome Atlas (TCGA) to investigate the link between PMD hypomethylation and mitotic cell division in cancer. The analysis shows that higher genome-wide somatic mutation densities are significantly associated with deeper PMD hypomethylation, suggesting that mitotic turnover may underlie both somatic mutation and PMD hypomethylation. Additionally, tumors with deeper PMD hypomethylation have more LINE-1 insertions, further supporting the link between ongoing cell proliferation and PMD hypomethylation.