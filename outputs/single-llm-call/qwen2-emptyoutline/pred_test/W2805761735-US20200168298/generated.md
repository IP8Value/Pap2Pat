# DESCRIPTION

## FIELD

The present invention relates to a method and system for performing allelic decomposition of highly polymorphic and structurally variant genes using high-throughput sequencing (HTS) data. Specifically, the invention provides a computational framework for accurately identifying the sequence content and structural variations of genes, particularly those with multiple copies and highly homologous pseudogenes, such as CYP2D6.

## BACKGROUND

High-throughput sequencing (HTS) technologies have revolutionized genomics research and clinical genomic testing. HTS data offer the potential to determine the exact sequence composition of each copy of a gene of interest. However, the computational challenges are significant, especially for genes that are highly polymorphic, have multiple copies, and are subject to structural alterations. Existing tools are limited in their ability to handle these complexities, often failing to accurately reconstruct the sequence content of such genes.

Current computational tools for structural variation detection focus on identifying large-scale structural alterations in uniquely mappable regions of the genome. Tools for copy number alteration detection assume that gene duplications or deletions affect the entire gene, but do not reconstruct the exact sequence content. No existing tool can identify and reconstruct the sequence content of genes that have been subject to partial duplications, deletions, or fusions with highly homologous pseudogenes.

The invention addresses these limitations by providing a novel combinatorial framework that can perform allelic decomposition of any gene of interest in HTS data. This framework can handle genes that differ from the reference genome by single nucleotide variants (SNVs), short indels, full gene duplications or deletions, partial gene duplications or deletions, and fusions with highly homologous pseudogenes.

## SUMMARY

The present invention provides a method and system for performing allelic decomposition of highly polymorphic and structurally variant genes using high-throughput sequencing (HTS) data. The method includes the following steps:

1. **Read Alignment and Mutation Detection**: Aligning HTS reads to a reference genome and identifying mutations present in the target gene region.
2. **Copy Number and Structural Variation Estimation**: Identifying the copy number of the gene and detecting various structural variations.
3. **Major Star-Allele Identification**: Establishing the major star-allele of each gene copy.
4. **Genotype Refining**: Assigning neutral mutations to each major star-allele and ranking each allelic configuration.

The invention also provides a system for implementing the method, including a processor and a memory storing instructions for executing the method. The system is capable of processing HTS data to accurately reconstruct the sequence content and structural variations of genes, particularly those with multiple copies and highly homologous pseudogenes.

## DETAILED DESCRIPTION

### CYP2D6

CYP2D6 is a highly polymorphic gene that plays a crucial role in the metabolism of approximately 25% of clinically prescribed drugs. The gene is located in close proximity to highly homologous pseudogenes, such as CYP2D7 and CYP2D8, which can form various structural rearrangements and copy number variations. These structural alterations can lead to the formation of hybrid genes, making allelic decomposition challenging.

### HTS Data

HTS data provide the means to determine the exact sequence composition of each copy of a gene of interest. The data are generated using high-throughput sequencing technologies, such as Illumina platforms, and are typically stored in SAM/BAM file formats. The data include reads that are aligned to a reference genome, allowing for the identification of mutations and structural variations.

### Alignment/Read Mapping

The first step in the method is to align HTS reads to a reference genome. This is typically done using a read mapper, such as BWA or CORA, followed by local indel realignment using the Genome Analysis Toolkit (GATK). The alignment process identifies the positions of reads in the reference genome and helps in detecting mutations and structural variations.

### Sequence Variant Calling

After read alignment, the method involves identifying mutations present in the target gene region. This includes single nucleotide variants (SNVs) and short indels. The mutations are detected by comparing the aligned reads to the reference genome and identifying positions where the reads differ from the reference.

### Detecting Structural Variants

Structural variations, such as deletions, duplications, and fusions, are detected by analyzing the read alignments. The method uses a combination of discordantly mapping paired-end reads, split-read mappings, and de novo assembled contigs to identify structural variations. The structural variations are then characterized to determine their exact sequence content and impact on the gene.

### Coverage Normalization

Coverage normalization is a critical step in the method. It involves estimating the normalized copy number of the gene and its pseudogenes at each position. This is done by calculating the coverage of reads at each position and normalizing it by the expected coverage. The normalized copy number is used to estimate the aggregate copy number of each exon and intron of the gene.

### Major Star-Allele Identification

The major star-allele of each gene copy is identified by analyzing the gene-disrupting mutations detected in the sample. The method filters out major star-alleles that do not match the observed mutations and structural configurations. The goal is to find a set of major star-alleles that most closely match the observed set of gene-disrupting mutations.

### Genotype Refining

The final step in the method is genotype refining, which involves assigning neutral mutations to each major star-allele. This step helps in distinguishing between major star-alleles that share common gene-disrupting mutations. The method uses a quadratic integer programming (QIP) approach to minimize the difference between the observed neutral mutations and the mutations assigned to each major star-allele. The final genotype is determined by selecting the configuration with the lowest score.

## Complexity

The computational complexity of the method varies depending on the specific steps involved. The read alignment and mutation detection steps are generally polynomial in time complexity. The copy number and structural variation estimation steps, as well as the major star-allele identification and genotype refining steps, are NP-hard in the general case. However, the method utilizes state-of-the-art integer programming solvers, such as Gurobi or SCIP, to efficiently solve these problems in practice.

## Systems

The invention also provides a system for implementing the method. The system includes a processor and a memory storing instructions for executing the method. The system is capable of processing HTS data to accurately reconstruct the sequence content and structural variations of genes. The system can be implemented on a variety of computing platforms, including desktop computers, servers, and cloud-based systems.

## Examples

### Example 1: CYP2D6 Genotyping

In this example, the method is applied to genotyping the CYP2D6 gene using HTS data from a set of 96 cell lines sequenced via the PGRNseq v.2 protocol. The method accurately identifies the sequence content and structural variations of CYP2D6, including deletions, duplications, and fusions with CYP2D7. The results are validated using PCR-based genotyping panels, and the method outperforms existing tools in terms of accuracy and computational efficiency.

### Example 2: Whole-Genome Sequencing

In this example, the method is applied to whole-genome sequencing (WGS) data from 25 Illumina samples. The method accurately identifies the sequence content and structural variations of key pharmacogenes, including CYP2D6, CYP2A6, and CYP2C19. The results are validated using genotypes from the literature, and the method demonstrates high accuracy and consistency with Mendelian laws of inheritance.

### DISCUSSION

The invention provides a novel and efficient method for performing allelic decomposition of highly polymorphic and structurally variant genes using HTS data. The method addresses the limitations of existing tools by accurately reconstructing the sequence content and structural variations of genes, particularly those with multiple copies and highly homologous pseudogenes. The method is applicable to a wide range of genes and can be used in various applications, including clinical decision-making and personalized medicine.

The invention also provides a system for implementing the method, making it accessible to researchers and clinicians. The system is designed to be user-friendly and can be integrated into existing workflows for HTS data analysis. Future work will focus on expanding the method to handle novel alleles and sub-alleles, as well as incorporating additional data sources to improve the accuracy of genotyping.