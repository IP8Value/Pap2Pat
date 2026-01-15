Here is the patent application following your outline and research paper:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to a method and system for determining copy number variations (CNVs) in a genome sample. More particularly, the invention provides an improved computational approach for detecting CNVs in single cells or trace nucleic acid samples while correcting for whole genome amplification (WGA) biases. The method involves sequencing a genome sample, aligning sequencing results to a reference genome, determining breakpoints through statistical analysis, establishing detection windows, and determining CNVs based on normalized read distributions. The system implements this method through specialized computational units and computer-readable media.  

## BACKGROUND  

Current methods for detecting CNVs in single cells, such as array comparative genomic hybridization (aCGH), suffer from significant limitations due to biases introduced during whole genome amplification. These biases are particularly pronounced in GC-rich and GC-poor regions of the genome, leading to false CNV signals and reduced detection accuracy. While massively parallel sequencing (MPS) offers advantages over aCGH, existing computational approaches still fail to adequately account for WGA-induced biases when analyzing single-cell sequencing data. There remains an unmet need for improved methods that can accurately detect CNVs in single cells while correcting for amplification biases, particularly those related to GC content. The present invention addresses this need by providing a novel computational pipeline that incorporates GC bias correction, dynamic thresholding, and advanced statistical segmentation to improve CNV detection sensitivity and specificity.  

## SUMMARY  

The invention provides a method for determining copy number variation in a genome sample. The method comprises sequencing a genome sample to generate sequencing data, aligning the sequencing data to a reference genome sequence, determining breakpoints in the reference genome sequence based on the distribution of aligned reads, establishing detection windows between breakpoints, calculating a first parameter based on reads falling within each detection window, and determining whether a copy number variation exists by comparing the first parameter to a preset threshold.  

The sequencing step involves extracting genomic DNA from a biological sample, which may be a single cell isolated from peripheral blood, blastocyst, or other sources. The genome sample undergoes whole genome amplification before library construction and sequencing. Sequencing may be performed using massively parallel sequencing platforms to generate short reads, typically 50-100 bp in length.  

Alignment of sequencing results involves mapping reads to a reference genome while filtering out non-uniquely aligned reads and PCR duplicates. The distribution of aligned reads across the reference genome is analyzed to identify regions with statistically significant changes in read density, indicating potential breakpoints.  

Breakpoint determination employs a binary segmentation algorithm that calculates p-values for candidate breakpoints and iteratively merges adjacent segments until reaching a termination threshold. Detection windows are then defined between validated breakpoints.  

The first parameter represents a normalized read count within each detection window, corrected for GC bias through a weighted adjustment strategy. GC correction involves calculating relative read numbers, determining GC-dependent weighting coefficients, and applying these to obtain corrected read counts. The normalized read counts are compared to dynamic thresholds that account for local GC content, improving detection accuracy.  

The invention further provides a system for determining copy number variation comprising a sequencing apparatus configured to sequence a genome sample and a computational unit configured to perform alignment, breakpoint detection, GC correction, and CNV calling as described above. The system may include specialized subunits for sequencing library preparation, data processing, and statistical analysis.  

Additionally, the invention provides a computer-readable medium storing instructions that, when executed, cause a processor to align sequencing data to a reference genome, determine read distributions, identify breakpoints, establish detection windows, normalize read counts, and determine CNVs based on comparison to dynamic thresholds. The computer-readable medium preserves the order of computational steps to ensure proper execution of the method.  

## DETAILED DESCRIPTION  

The following detailed description provides a comprehensive explanation of the invention's components and methodologies.  

**Definition of Copy Number Variation**  
Copy number variation refers to differences in the number of copies of a particular genomic region between individuals or cells. These variations may include deletions (reduced copy number), duplications (increased copy number), or more complex rearrangements. The present invention provides an improved method for detecting such variations, particularly in single-cell samples where amplification biases pose significant challenges.  

**Method of Determining Copy Number Variation**  
The method begins with sequencing a genome sample extracted from a biological specimen. Biological samples may include single cells isolated from peripheral blood, blastocysts, or other tissues. For single-cell analysis, cells are typically isolated using micromanipulation techniques and lysed to release genomic DNA.  

Whole genome amplification is performed on the extracted DNA using methods such as degenerate oligonucleotide primer PCR (DOP-PCR). The amplified DNA is then used to construct a sequencing library with appropriate adapters for massively parallel sequencing. Sequencing generates millions of short reads (typically 50-100 bp) that collectively provide coverage across the genome.  

**Alignment of Sequencing Data**  
Sequencing reads are aligned to a reference genome sequence using mapping algorithms that allow for a limited number of mismatches. Only uniquely aligned reads are retained for subsequent analysis, while PCR duplicates and non-unique mappings are filtered out. The distribution of aligned reads across the reference genome is analyzed by dividing the genome into primary observation windows of approximately 150 kb.  

**Breakpoint Determination**  
Candidate breakpoints are identified by calculating p-values for the differences in read counts between adjacent windows using statistical tests such as run tests. A binary segmentation algorithm is employed to iteratively merge adjacent segments and refine breakpoint locations until reaching a termination threshold. This process results in the identification of statistically significant breakpoints that define potential CNV boundaries.  

**GC Correction and Normalization**  
To correct for WGA-induced GC bias, the method calculates relative read numbers (RRN) for each window and applies GC-dependent weighting coefficients. The correction accounts for both under-amplification in GC-poor regions and over-amplification in GC-rich regions. Corrected read numbers are normalized to produce uniform distributions across varying GC contents.  

**Detection Window Establishment**  
Final detection windows are defined between validated breakpoints. The method calculates a first parameter representing the normalized read count within each window and compares this to dynamic thresholds that vary with local GC content. This comparison determines whether a CNV exists and identifies its type (deletion or duplication).  

**System Implementation**  
The system comprises a sequencing apparatus for generating sequencing data and computational units for data analysis. The computational units include specialized modules for alignment, breakpoint detection, GC correction, and CNV calling. The system may be implemented as an integrated platform or distributed across multiple devices.  

**Computer-Readable Medium**  
The invention includes non-transitory computer-readable media storing instructions for executing the method. The medium preserves the computational workflow from alignment through CNV detection, ensuring reproducible analysis.  

**Examples**  
Example 1 demonstrates the method using whole genome amplified samples sequenced at low coverage (4-9.5%). Data analysis shows high concordance with known CNVs, validating the method's accuracy. Example 2 compares results with and without amplification, highlighting the importance of GC correction in single-cell analysis.  

## INDUSTRIAL APPLICABILITY  

The invention has broad applicability in genetic diagnostics and research. It enables accurate CNV detection in single cells for applications including preimplantation genetic diagnosis, non-invasive prenatal testing, and cancer heterogeneity studies. The method's improved sensitivity and specificity compared to existing techniques make it particularly valuable for clinical applications where accurate CNV detection is critical. The system can be implemented in diagnostic laboratories and research facilities, providing a robust platform for single-cell genomic analysis. The computer-readable medium facilitates standardized implementation across different settings, ensuring consistent performance. The invention represents a significant advance in the field of genomic analysis by overcoming key limitations of current single-cell CNV detection methods.