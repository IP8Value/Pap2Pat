Below is the patent application drafted according to the provided outline and research paper. The application has been structured to follow the outline precisely while incorporating the technical details from the research paper in formal patent language.

---

# DESCRIPTION  

## FIELD  

The present invention relates generally to the field of genetic testing and pharmacogenomics. More specifically, the invention provides novel methods and systems for high-throughput genotyping of pharmacogenes, particularly those prone to structural variations and polymorphisms, using high-throughput sequencing (HTS) data. The invention enables accurate allelic decomposition of genes, including identification of novel star-alleles, structural variants, and hybrid gene configurations, thereby improving clinical decision-making in drug metabolism and personalized medicine.  

## BACKGROUND  

Genetic testing has become an essential tool in modern medicine, particularly in pharmacogenomics, where understanding an individual's genetic makeup can guide drug selection and dosing. High-throughput sequencing (HTS) technologies, such as Illumina sequencing, have revolutionized genetic analysis by enabling rapid and cost-effective sequencing of entire genomes or targeted gene panels. However, the analysis of HTS data presents significant computational challenges, particularly for genes with high allelic variability, structural rearrangements, or pseudogene homology.  

Current computational tools for analyzing HTS data are limited in their ability to resolve complex structural variations, such as gene deletions, duplications, and hybridizations with pseudogenes. Existing methods, such as VariationHunter, HYDRA, and Pindel, focus primarily on detecting large-scale structural variants in uniquely mappable regions of the genome. These tools fail to accurately reconstruct the sequence content of structurally altered genes, particularly those with multiple copies or high homology to pseudogenes.  

A critical limitation of existing tools is their inability to perform allelic decomposition—determining the exact sequence composition of each copy of a gene in a sample. This is especially problematic for pharmacogenes, such as CYP2D6 and CYP2A6, which are highly polymorphic and frequently undergo structural rearrangements. Current genotyping assays, such as Affymetrix DMET+ and Illumina ADME arrays, are limited to detecting predefined variants and cannot identify rare or novel alleles, which may have significant clinical implications.  

The importance of genotyping ADME (absorption, distribution, metabolism, and excretion) genes cannot be overstated, as these genes regulate the metabolism of over 90% of clinically prescribed drugs. For example, CYP2D6 is involved in the metabolism of 20-25% of drugs, and its allelic variants can significantly impact drug efficacy and toxicity. Existing array-based genotyping assays are limited in scope and accuracy, often failing to detect rare or structurally complex alleles.  

Targeted genotyping platforms, such as the PGRNseq capture panel, have been developed to address some of these limitations. PGRNseq targets 84 pharmacogenes, including CYP2D6, and provides high-depth sequencing at a lower cost than whole-genome sequencing (WGS). However, analyzing PGRNseq data remains challenging due to non-uniform coverage and the complexity of genotyping structurally variant genes.  

Algorithmic challenges in ADME genotyping include resolving mapping ambiguities, detecting structural variants, and accurately assigning mutations to specific gene copies. No existing tool comprehensively addresses these challenges, leading to gaps in clinical genotyping and suboptimal drug therapy decisions.  

## SUMMARY  

The present invention provides methods and systems for genotyping pharmacogenes using high-throughput sequencing (HTS) data. The invention enables accurate allelic decomposition of genes by identifying nucleic acid sequence variants, detecting structural variants, and assigning star-alleles to each gene copy.  

Key aspects of the invention include:  
1. **Receiving HTS Data**: The method begins with receiving HTS data from a target sample, such as whole-genome sequencing (WGS) or targeted capture sequencing (e.g., PGRNseq).  
2. **Alignment to Reference Genome**: Target sample reads are aligned to a reference genome allele database, which includes known star-alleles and structural variants for the gene of interest.  
3. **Variant Identification**: Nucleic acid sequence variants, including single nucleotide variants (SNVs) and indels, are identified from the aligned reads.  
4. **Structural Variant Detection**: Structural variants, such as gene deletions, duplications, and hybridizations with pseudogenes, are detected using coverage analysis and combinatorial optimization.  
5. **Gene-Disrupting Mutations**: Mutations impacting protein function (gene-disrupting mutations) are identified and used to infer major star-alleles.  
6. **Star-Allele Selection**: Reference star-alleles are selected based on the detected mutations, and the genotype associated with the selected star-alleles is called.  
7. **Genotype Refinement**: Neutral mutations are used to refine the genotype, distinguishing between minor star-alleles and resolving ambiguities.  
8. **Scalability**: The method can be repeated for multiple genes simultaneously and is implemented using a suitably programmed computer system.  

The invention is applicable to various types of HTS data, including WGS, whole-exome sequencing (WES), and targeted capture sequencing. The system for predicting genotypes includes modules for sequence alignment, variant calling, structural variant detection, and genotype refinement, all integrated into a user-friendly interface.  

## DETAILED DESCRIPTION  

The invention introduces a novel computational tool, referred to herein as "Aldy," for allelic decomposition of genes using HTS data. Aldy addresses the limitations of existing structural variation discovery tools by providing a comprehensive framework for genotyping structurally altered and polymorphic genes.  

### Limitations of Existing Tools  
Existing tools for structural variant detection, such as VariationHunter and Pindel, are limited to identifying large-scale structural alterations in uniquely mappable regions. These tools cannot reconstruct the sequence content of genes with multiple copies or high homology to pseudogenes. For example, CYP2D6 frequently hybridizes with its pseudogene, CYP2D7, forming complex hybrid alleles that are difficult to detect with current methods.  

### Motivation for Aldy  
The need for Aldy arises from the inability of existing tools to accurately genotype pharmacogenes, particularly those with structural variations. Aldy combines read alignment, copy number estimation, and combinatorial optimization to resolve these challenges, enabling precise allelic decomposition.  

### PGRNseq Capture Protocol  
Aldy is optimized for use with the PGRNseq capture protocol, which targets 84 pharmacogenes, including CYP2D6 and CYP2A6. PGRNseq provides high-depth sequencing (average 500× coverage) at a lower cost than WGS. However, PGRNseq data exhibits non-uniform coverage, complicating structural variant detection. Aldy incorporates coverage normalization algorithms to address this issue.  

### Advantages Over WGS and WES  
While WGS and WES provide broad genomic coverage, they are costly and computationally intensive. Aldy leverages targeted capture sequencing (e.g., PGRNseq) to achieve high-depth coverage of pharmacogenes at a fraction of the cost. Additionally, Aldy's algorithms are specifically designed to handle the non-uniform coverage of targeted capture data.  

### Challenges in Genotyping ADME Genes  
Genotyping ADME genes, such as CYP2D6, is challenging due to their high polymorphism and structural variability. For example, CYP2D6 has over 100 known star-alleles, many of which result from structural rearrangements with CYP2D7. Aldy addresses these challenges by integrating copy number estimation, structural variant detection, and star-allele identification into a unified framework.  

### Star-Allele Nomenclature  
Aldy employs the star-allele nomenclature system, where major star-alleles (e.g., *2, *4) are defined by gene-disrupting mutations, and minor star-alleles (e.g., *2A, *2B) are extensions of major alleles with neutral mutations. This system allows for precise characterization of gene copies and their functional impact.  

### Genotyping Steps  
Aldy performs genotyping through the following steps:  
1. **Read Alignment and Mutation Detection**: HTS reads are aligned to the reference genome, and mutations in the target gene region are identified.  
2. **Copy Number and Structural Variation Estimation**: The copy number of the gene is estimated, and structural variants (e.g., deletions, duplications, fusions) are detected.  
3. **Major Star-Allele Identification**: Major star-alleles are inferred based on gene-disrupting mutations.  
4. **Genotype Refinement**: Neutral mutations are used to assign minor star-alleles and resolve ambiguities.  

### Example: CYP2D6 Genotyping  
CYP2D6 is used as an exemplary gene due to its clinical importance and structural complexity. Aldy detects CYP2D6 star-alleles, including hybrid configurations with CYP2D7, by solving combinatorial optimization problems to infer the most parsimonious allelic decomposition.  

### System Architecture  
Aldy is implemented as a computer system comprising:  
- A **sequence aligner** for mapping reads to the reference genome.  
- A **variant identifier** for calling SNVs and indels.  
- A **structural variant identifier** for detecting copy number variations and hybrid genes.  
- A **genotype caller** for assigning star-alleles.  
- A **genotype refiner** for resolving ambiguities using neutral mutations.  

The system operates efficiently, analyzing high-coverage BAM files in under a minute on a standard laptop.  

## EXAMPLES  

### Data Sets  
Aldy was validated on three data sets:  
1. **96 Coriell Cell Lines**: Sequenced using PGRNseq v.2, with validated CYP2D6 genotypes.  
2. **137 GeT-RM Samples**: Sequenced using PGRNseq v.1, with genotypes for 10 ADME genes.  
3. **25 Illumina WGS Samples**: From the Platinum Genome and 1000 Genomes projects.  

### Performance  
Aldy achieved >99% accuracy in genotyping CYP2D6 and other ADME genes, outperforming existing tools like Cypiripi and Astrolabe. Notably, Aldy correctly identified complex structural variants and novel star-alleles missed by other methods.  

### Novel Alleles  
Aldy detected novel major star-alleles, such as a CYP2D6*10-like allele in sample NA17012 and novel DPYD alleles in samples NA07357 and NA24027. These findings demonstrate Aldy's ability to discover previously uncharacterized variants.  

## DISCUSSION  

Aldy represents a significant advancement in pharmacogene genotyping, enabling accurate identification of structurally complex alleles. Its low computational overhead and high accuracy make it suitable for clinical applications, where rapid and reliable genotyping is critical.  

Aldy's performance on Illumina WGS and PGRNseq data demonstrates its versatility and robustness. The tool's ability to resolve Mendelian inconsistencies and detect novel alleles highlights its superiority over existing methods.  

Future work may extend Aldy to incorporate long-read sequencing data, further improving its ability to resolve complex structural variants.  

---

This patent application provides a comprehensive and standalone description of the invention, adhering to the provided outline and incorporating all technical details from the research paper. Let me know if you'd like any modifications or additional details.