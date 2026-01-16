Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the field of genomic analysis, particularly to methods for detecting copy number variations (CNVs) in single cells. More specifically, the invention provides a novel bioinformatics pipeline incorporating GC bias correction, binary segmentation algorithms, and dynamic threshold determination to accurately identify CNVs from whole genome amplification (WGA) data obtained through low-coverage massively parallel sequencing. The disclosed method demonstrates superior sensitivity and specificity compared to existing techniques, making it particularly suitable for clinical applications such as pre-implantation genetic diagnosis, non-invasive prenatal testing, and cancer heterogeneity research.  

## BACKGROUND  

Copy number variations (CNVs) play a critical role in human disease pathogenesis, including developmental disorders, cancer progression, and other genetic abnormalities. Traditional methods for CNV detection, such as array comparative genomic hybridization (aCGH), have been widely employed but suffer from significant limitations when applied to single-cell analysis. Whole genome amplification (WGA) introduces substantial biases during the amplification process, particularly in GC-rich and GC-poor regions, leading to false positive and false negative CNV calls. These biases stem from differential polymerase processivity and DNA priming efficiency across genomic regions with varying GC content.  

Existing computational methods for CNV detection from sequencing data, such as SegSeq, rely on comparative genomic strategies that are inherently limited when analyzing single-cell data due to the absence of matched controls and the compounding effects of WGA artifacts. The technical challenges are further exacerbated in clinical applications where sample quantities are extremely limited, such as in pre-implantation genetic diagnosis or circulating tumor cell analysis. There remains an unmet need for a robust analytical framework capable of accurately detecting CNVs in single cells while compensating for the technical artifacts introduced during WGA and low-coverage sequencing.  

## SUMMARY  

The present invention provides a comprehensive bioinformatics pipeline for accurate CNV detection in single cells through the following key innovations:  

First, the method implements a weighted GC bias correction strategy that substantially eliminates amplification artifacts by normalizing read counts according to local GC content. This correction employs a GC-related weighting coefficient calculated from sequencing data, achieving over 99.9% reduction in GC bias as measured by a novel GC-bias index.  

Second, the pipeline incorporates a binary segmentation algorithm that precisely localizes CNV breakpoints through an iterative merging process. The algorithm first identifies candidate breakpoints through statistical testing of read count differences and then refines these predictions by successively merging adjacent segments until reaching optimal segmentation.  

Third, the invention introduces a dynamic threshold determination system that automatically adjusts calling criteria based on local GC content. This adaptive approach significantly reduces false positive calls in both GC-rich and GC-poor regions compared to fixed-threshold methods.  

Validation across seven single-cell samples demonstrated 99.63% sensitivity and 97.71% specificity for CNVs larger than 1 Mb, with accurate detection of events as small as 3.94 Mb. The method outperforms existing techniques by maintaining high specificity (97.71% vs 16.49%) while achieving comparable sensitivity. Additional simulations confirmed robust performance across CNV sizes from 500 kb to 5 Mb, with breakpoint localization precision within 70 kb.  

## DETAILED DESCRIPTION  

### Whole Genome Amplification and Sequencing  

Single cells are isolated from peripheral blood, blastocysts, or other tissues using micromanipulation techniques. Following lysis, whole genome amplification is performed using degenerate oligonucleotide primer PCR (DOP-PCR) or similar methods. The amplified DNA is then processed for library preparation and sequenced using massively parallel sequencing platforms, typically generating 10-15 million single-end 50 bp reads per cell at 4-9.5% genome coverage.  

### Data Preprocessing and Alignment  

Raw sequencing reads undergo quality control and adapter removal before alignment to a reference genome (e.g., HG18) using short read aligners such as SOAP2. The alignment parameters permit a maximum of two mismatches while filtering out PCR duplicates and non-uniquely mapped reads. The resulting alignment files provide the basis for subsequent analysis.  

### Dynamic Window Selection  

The reference genome is divided into approximately 18,743 non-overlapping observation windows averaging 150 kb in size. Window boundaries are dynamically determined through in silico simulation to ensure each window contains a comparable number of uniquely mappable 50 bp reads (typically 140,000 simulated reads per window). This approach normalizes for variations in genome mappability while maintaining sufficient resolution for CNV detection.  

### GC Bias Quantification and Correction  

For each observation window, the method calculates:  
1) The relative read number (RRN) as the quotient between observed reads and the genome-wide average  
2) The GC-bias index measuring deviation between observed and expected RRN values  

A loess regression model fits RRN values against GC content at 0.5% increments to predict expected read counts. The weighted correction strategy then applies GC-specific normalization factors to each window, computed as:  

w_ij = E[RRN|GC_seq_i, GC_ref_j]  

where GC_seq_i and GC_ref_j represent the sequencing-derived and reference genome GC content for window i in sample j. Corrected read counts (CRN) are calculated by multiplying raw counts by their corresponding weights followed by global normalization.  

### Binary Segmentation Algorithm  

The breakpoint detection algorithm proceeds through two phases:  

**Initialization:**  
1) Compute significance of read count differences across each window boundary using a runs test  
2) Select the top 3,000 most significant candidate breakpoints based on p-values  

**Iterative Merging:**  
1) For each candidate breakpoint, evaluate the difference between adjacent segments  
2) Remove the breakpoint with least significant difference (highest p-value)  
3) Repeat until all remaining breakpoints meet significance threshold (p < 0.05)  

The final segmentation provides precise localization of CNV boundaries while accounting for local variations in data quality.  

### Dynamic Thresholding  

Final CNV calls are made by comparing segment mean CRN values to GC-specific thresholds:  
1) Deletion thresholds set at 5th percentile of CRN distribution for each GC bin  
2) Duplication thresholds set at 95th percentile  

This adaptive approach prevents systematic errors in GC-extreme regions while maintaining sensitivity across the genome.  

### Performance Validation  

The method was validated using:  
1) Seven single-cell samples with known CNVs (confirmed by SNP array or WGS)  
2) In silico simulations of 500 kb - 5 Mb CNVs  

Results demonstrated:  
- 99.63% sensitivity and 97.71% specificity for >1 Mb CNVs  
- 94% sensitivity for 3 Mb CNVs in simulations  
- 95% specificity for 750 kb events  
- Median breakpoint precision of 70 kb  

Comparative analysis showed substantial improvement over SegSeq, particularly in specificity (97.71% vs 16.49%) due to effective GC bias correction.  

## INDUSTRIAL APPLICABILITY  

The present invention has broad applications across biomedical research and clinical diagnostics:  

**Reproductive Medicine:**  
- Pre-implantation genetic diagnosis/screening (PGD/PGS) for embryo selection  
- Non-invasive prenatal testing using fetal nucleated red blood cells  

**Oncology:**  
- Circulating tumor cell analysis for cancer screening and monitoring  
- Tumor heterogeneity studies at single-cell resolution  

**Genetic Disorders:**  
- Diagnosis of microdeletion/microduplication syndromes  
- Investigation of somatic mosaicism in developmental disorders  

The method's robustness with low-coverage sequencing (4-10%) makes it particularly suitable for clinical applications where sample quantity is limited. Further optimization through increased sequencing depth or targeted enrichment can enhance resolution for smaller CNVs (<1 Mb).  

The pipeline can be implemented as software for sequencing centers and clinical laboratories, integrated into existing analysis workflows for single-cell genomics. Commercial applications include diagnostic kits combining wet-lab protocols with the computational analysis method.