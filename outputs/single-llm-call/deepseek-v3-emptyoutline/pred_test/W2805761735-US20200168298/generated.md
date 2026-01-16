Below is the drafted patent application following the provided outline and incorporating the research paper's content:

---

# DESCRIPTION  

## FIELD  
The present invention relates to the field of computational genomics, specifically to methods and systems for allelic decomposition of structurally altered and polymorphic genes using high-throughput sequencing (HTS) data. More particularly, the invention provides a combinatorial framework for accurately determining the sequence content of each copy of a gene, including genes subject to structural variations such as deletions, duplications, and hybridizations with pseudogenes. The invention is particularly useful for genotyping pharmacogenes and other clinically relevant genes, enabling precise drug metabolism predictions and personalized medicine applications.  

## BACKGROUND  
Current computational tools for analyzing HTS data are limited in their ability to resolve the allelic composition of genes that exhibit structural variations or high polymorphism. Existing methods focus on detecting large-scale structural alterations in uniquely mappable genomic regions or estimating gene copy numbers under the assumption that duplications or deletions affect entire genes. However, these tools fail to reconstruct the exact sequence content of structurally altered genes, particularly those with multiple copies or highly homologous pseudogenes.  

For example, pharmacogenes such as CYP2D6 and CYP2A6 are highly polymorphic and prone to structural rearrangements, including hybridizations with pseudogenes. Traditional genotyping assays, such as PCR-based panels or microarray technologies, are limited to detecting predefined variants and cannot identify novel or rare alleles. This gap in capability hinders accurate clinical genotyping, which is critical for drug response prediction and personalized treatment plans.  

There is thus a pressing need for a computational framework capable of resolving the allelic composition of structurally altered genes from HTS data, including whole-genome sequencing (WGS) and targeted capture sequencing (e.g., PGRNseq). The present invention addresses this need by providing a novel combinatorial optimization approach for allelic decomposition, enabling the identification of known and novel alleles with high accuracy and efficiency.  

## SUMMARY  
The invention provides a computational framework, referred to herein as "Aldy," for performing allelic decomposition of any gene of interest using HTS data. The framework is capable of resolving genes that differ from the reference genome by single nucleotide variants (SNVs), short indels, full or partial gene duplications/deletions, and hybridizations with pseudogenes.  

Key features of the invention include:  
1. **Alignment and Mutation Detection**: HTS reads are aligned to a reference genome, and mutations in the target gene region are identified.  
2. **Copy Number and Structural Variation Estimation**: The copy number of the gene is determined, and structural variations (e.g., deletions, duplications, fusions) are identified using an integer linear programming (ILP) formulation.  
3. **Major Star-Allele Identification**: The major star-allele of each gene copy is inferred by matching observed gene-disrupting mutations to known alleles or constructing novel alleles when necessary.  
4. **Genotype Refinement**: Neutral mutations are assigned to major star-alleles to refine the genotype, enabling the identification of minor star-alleles and novel hybrid configurations.  

The invention is implemented as a software tool that operates efficiently on standard computing hardware, requiring less than a minute to analyze high-coverage sequencing data. It has been validated on large datasets, including PGRNseq and WGS samples, demonstrating superior accuracy compared to existing methods.  

## DETAILED DESCRIPTION  

### CYP2D6  
CYP2D6 is a highly polymorphic pharmacogene involved in the metabolism of 20–25% of clinically prescribed drugs. Its genotyping is complicated by structural variations, including hybridizations with the pseudogene CYP2D7. The invention's framework is exemplified using CYP2D6 but is applicable to any gene with similar challenges.  

### HTS Data  
The invention processes HTS data from whole-genome or targeted sequencing platforms (e.g., Illumina HiSeq, PGRNseq). Input data includes aligned reads in SAM/BAM format and a gene-specific database containing known alleles, structural variations, and pseudogene information.  

### Alignment/Read Mapping  
Reads are mapped to the reference genome using standard alignment tools (e.g., BWA, GATK). Local indel realignment is performed to improve mutation detection accuracy.  

### Sequence Variant Calling  
Single nucleotide variants (SNVs) and short indels are identified from the aligned reads. Variant calling accounts for coverage non-uniformity in targeted sequencing data.  

### Detecting Structural Variants  
Structural variations (e.g., deletions, duplications, fusions) are detected by analyzing coverage patterns and breakpoint regions. The invention employs a combinatorial approach to resolve ambiguous mappings in repetitive or homologous regions.  

### Coverage Normalization  
Read coverage is normalized to account for technical biases, enabling accurate copy number estimation. Regions with identical sequences between the gene and pseudogene are masked to prevent misalignment artifacts.  

### Major Star-Allele Identification  
Major star-alleles are identified by matching observed gene-disrupting mutations to known alleles or constructing novel alleles when no match is found. An ILP formulation minimizes discrepancies between observed and expected mutation profiles.  

### Genotype Refining  
Neutral mutations are assigned to major star-alleles to infer minor star-alleles. A quadratic integer programming (QIP) formulation optimizes the assignment, penalizing deviations from known allele definitions while allowing for novel allele discovery.  

## Complexity  
The invention's computational complexity arises from the NP-hard nature of copy number estimation and star-allele identification. However, state-of-the-art solvers (e.g., Gurobi, SCIP) enable efficient practical implementation. Limitations due to short read lengths are mitigated by masking uninformative regions and leveraging known breakpoints.  

## Systems  
The invention is implemented as a software system comprising:  
1. A read alignment module for processing HTS data.  
2. A mutation detection module for identifying SNVs and indels.  
3. A structural variation detection module for resolving copy number changes and hybrid configurations.  
4. A star-allele identification module for inferring major and minor alleles.  
5. A genotype refinement module for final genotype assignment.  

The system operates on standard computing hardware and outputs genotypes in a standardized format compatible with clinical reporting.  

## Examples  
### Example 1: CYP2D6 Genotyping  
The invention was applied to 96 PGRNseq v.2 samples, accurately identifying all CYP2D6 genotypes, including novel hybrid alleles. Validation against PCR-based panels confirmed 100% concordance, outperforming existing tools (Astrolabe, Cypiripi).  

### Example 2: Novel Allele Discovery  
Novel major star-alleles were identified in GeT-RM samples, including a CYP2D6*10-like allele with a previously unreported mutation (c.77 G>A). The invention's combinatorial approach enabled unambiguous allele assignment despite high sequence homology with CYP2D7.  

### DISCUSSION  
The invention represents a significant advancement in computational genotyping, enabling accurate allelic decomposition of structurally altered genes. Its clinical utility is demonstrated in pharmacogenomics, where precise genotype predictions inform drug dosing and selection. Future enhancements may incorporate long-read sequencing data to resolve ambiguous configurations and expand the database of known alleles.  

--- 

This draft adheres to the provided outline and incorporates the research paper's technical content while maintaining formal patent language and standalone clarity. Let me know if you'd like any refinements or expansions.