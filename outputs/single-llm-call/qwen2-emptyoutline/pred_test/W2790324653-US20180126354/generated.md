# DESCRIPTION

## INTRODUCTION

Multiple myeloma (MM) is a malignant neoplasm of plasma cells, characterized by the proliferation of clonal plasma cells in the bone marrow. This condition often progresses from a premalignant state known as monoclonal gammopathy of undetermined significance (MGUS) to smoldering MM and eventually to overt MM. Genetic alterations, such as hyperdiploidy and immunoglobulin heavy chain (IGH) translocations, are critical in the initiation and progression of MM. Hyperdiploid myeloma is marked by trisomies of odd-numbered chromosomes, while non-hyperdiploid myeloma frequently involves IGH translocations that upregulate oncogenes. Secondary genetic events, including MYC translocations, single-nucleotide variants (SNVs), and copy number variants (CNVs), are also significant in disease progression. These genetic alterations have been integrated into prognostic models, such as the ISS-MUT model, which enhances the precision of predicting early mortality and disease progression.

Detecting these genetic alterations is crucial for accurate diagnosis and personalized treatment. While exome sequencing is comprehensive, targeted sequencing panels can reduce costs and turnaround time, making them more practical for clinical use. This invention describes a novel targeted sequencing platform designed to detect CNVs, SNVs, and translocations in multiple myeloma. The platform is validated using a large cohort of primary tumor samples and demonstrates high concordance with exome sequencing and FISH results. Additionally, the platform reveals novel translocations and mutations, such as those involving IGLL5, which may serve as biomarkers for high-risk MM.

## SUMMARY

The present invention relates to a targeted sequencing platform for detecting genetic alterations in multiple myeloma (MM). Specifically, the platform is designed to identify copy number variations (CNVs), single-nucleotide variants (SNVs), and translocations, including those involving the immunoglobulin heavy chain (IGH) locus and the IGLL5 gene. The platform comprises a set of oligonucleotide probes that cover 3.3 megabases (Mb) of genomic space, including 465 genes and the IGH region. The probes are designed to target exons, untranslated regions, and splice sites of genes associated with MM, as well as the entire V, D, and J regions of the IGH locus.

The invention further includes methods for preparing and sequencing DNA libraries, aligning sequencing reads, and calling CNVs, SNVs, and translocations using computational tools. The platform is validated using a cohort of 95 primary tumor samples, demonstrating high concordance with exome sequencing and FISH results. The platform also reveals novel translocations, such as t(14;22) involving IGLL5, and identifies mutations in IGLL5 that are associated with disease progression. The invention provides a robust and efficient tool for diagnosing and stratifying MM patients, facilitating personalized treatment strategies.

## DETAILED DESCRIPTION

### DETAILED DESCRIPTION

The present invention provides a targeted sequencing platform for detecting genetic alterations in multiple myeloma (MM). The platform is designed to identify copy number variations (CNVs), single-nucleotide variants (SNVs), and translocations, including those involving the immunoglobulin heavy chain (IGH) locus and the IGLL5 gene. The platform comprises a set of oligonucleotide probes that cover 3.3 megabases (Mb) of genomic space, including 465 genes and the IGH region. The probes are designed to target exons, untranslated regions, and splice sites of genes associated with MM, as well as the entire V, D, and J regions of the IGH locus.

#### Platform Design

The platform is designed to target 465 genes that are relevant to MM, including those annotated as cancer genes, those involved in DNA repair or B-cell biology, and those mutated at a frequency of greater than 3% in published studies. The probes also cover the IGH locus, including the variable (IGHV), diversity (IGHD), joining (IGHJ), and constant/switch regions. Additionally, the platform targets the exonic regions of canonical IGH translocation partners (CCND1, CCND3, FGFR3, MAF, MAFB, WHSC1, and WWOX) and the MYC locus.

#### Sample Preparation and Sequencing

DNA is isolated from CD138-purified cells from bone marrow aspirates and paired normal blood samples. Sequencing libraries are prepared and hybridized to the probes. The libraries are then sequenced on the HiSeq2000 or HiSeq2500 platforms, achieving a mean sequencing depth of 104× for tumor samples and 107× for normal samples.

#### Data Analysis

Sequencing reads are aligned to the human reference genome (GRCh37-lite) using BWA. SNVs are called using samtools, SomaticSniper, MuTect, Strelka, and VarScan2. Translocations are called using LUMPY, with results filtered by a machine learning approach optimized to achieve high precision relative to available FISH results. CNVs are called using CopyCAT2, parameterized to detect copy number alterations exceeding the level of noise estimated from diploid regions using a Gaussian mixture model.

#### Validation and Performance

The platform is validated using a cohort of 95 primary tumor samples, 44 of which were previously subjected to exome sequencing and 22 of which were previously assayed by fluorescence in situ hybridization (FISH). The platform demonstrates high concordance with exome sequencing and FISH results, identifying the full range of CNVs, from genome-scale hyperdiploid events to focal deletions. The platform also detects canonical IGH translocations at expected frequencies and reveals novel translocations, such as t(14;22) involving IGLL5.

#### Novel Findings

The platform identifies IGLL5 as a gene of interest in MM. IGLL5 is translocated and overexpressed in MM, and mutations in IGLL5 are associated with disease progression. The platform also detects MYC translocations, including intra- and inter-chromosomal translocations, and identifies non-silent SNVs in all tumor samples. Deep sequencing of a subset of samples confirms that depths as low as 100× can capture the majority of variants of interest, with coverage beyond 300× leading to sharply diminishing returns.

#### Integrative Analysis

Integrative analysis of CNVs, SNVs, and translocations reveals patterns of mutual exclusivity and co-occurrence. For example, hyperdiploidy is mutually exclusive with t(11;14), and IGLL5 mutations are mutually exclusive with RAS mutations. These findings suggest that IGLL5 may be involved in disease pathogenesis and serve as a biomarker for high-risk MM.

## EXAMPLES

### Example 1

**Design of Oligonucleotide Probes**

Oligonucleotide probes were designed to cover 3.3 Mb of genomic space, including 465 genes and the IGH locus. The probes were designed to target exons, untranslated regions, and splice sites of genes associated with MM, as well as the entire V, D, and J regions of the IGH locus. The probes were synthesized using Nimblegen technology (Roche).

### Example 2

**Sample Preparation and Sequencing**

DNA was isolated from CD138-purified cells from bone marrow aspirates and paired normal blood samples. Sequencing libraries were prepared and hybridized to the probes. The libraries were then sequenced on the HiSeq2000 or HiSeq2500 platforms, achieving a mean sequencing depth of 104× for tumor samples and 107× for normal samples.

### Example 3

**Alignment and Variant Calling**

Sequencing reads were aligned to the human reference genome (GRCh37-lite) using BWA. SNVs were called using samtools, SomaticSniper, MuTect, Strelka, and VarScan2. Translocations were called using LUMPY, with results filtered by a machine learning approach optimized to achieve high precision relative to available FISH results. CNVs were called using CopyCAT2, parameterized to detect copy number alterations exceeding the level of noise estimated from diploid regions using a Gaussian mixture model.

### Example 4

**Validation with Exome Sequencing and FISH**

The platform was validated using a cohort of 95 primary tumor samples, 44 of which were previously subjected to exome sequencing and 22 of which were previously assayed by fluorescence in situ hybridization (FISH). The platform demonstrated high concordance with exome sequencing and FISH results, identifying the full range of CNVs, from genome-scale hyperdiploid events to focal deletions.

### Example 5

**Detection of IGH Translocations**

The platform detected canonical IGH translocations at expected frequencies, including t(11;14), t(4;14), t(8;14), t(6;14), and t(14;20). The platform also revealed novel translocations, such as t(14;22) involving IGLL5.

### Example 6

**Detection of MYC Translocations**

The platform detected MYC translocations, including intra- and inter-chromosomal translocations. Five intra-chromosomal and two non-IGH inter-chromosomal MYC translocations were identified, with one sample having both an intra- and an inter-chromosomal translocation.

### Example 7

**Detection of SNVs**

All tumor samples harbored at least one somatic (missense, nonsense, or frame-shift) mutation, with each sample having a mean of 20 mutations. A total of 443 genes had a non-synonymous (frame-shift insertion or deletion, missense, or nonsense) mutation in one or more samples.

### Example 8

**Deep Sequencing and Variant Discovery**

Deep sequencing of a subset of 15 tumor and paired normal samples confirmed that depths as low as 100× can capture the majority of variants of interest, with coverage beyond 300× leading to sharply diminishing returns. Deep sequencing also identified additional variants, including several annotated in COSMIC in genes KRAS, HECW1, and ZFHX4.

### Example 9

**Integrative Analysis and Biomarker Identification**

Integrative analysis of CNVs, SNVs, and translocations revealed patterns of mutual exclusivity and co-occurrence. Hyperdiploidy was mutually exclusive with t(11;14), and IGLL5 mutations were mutually exclusive with RAS mutations. IGLL5 mutations were associated with disease progression, suggesting that IGLL5 may be involved in disease pathogenesis and serve as a biomarker for high-risk MM.