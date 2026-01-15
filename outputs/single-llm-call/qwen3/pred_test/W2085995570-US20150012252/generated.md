# DESCRIPTION

## TECHNICAL FIELD

- relate to method and system

The present invention relates to a method and system for the detection of copy number variations in genomic samples derived from single cells or trace amounts of nucleic acid material. Specifically, the invention provides a computationally enhanced bioinformatics pipeline that enables accurate, high-specificity identification of copy number variations through low-coverage whole genome sequencing, while effectively mitigating amplification biases inherent in whole genome amplification protocols. The system is particularly suited for clinical applications requiring precise genetic analysis of limited biological material, including pre-implantation genetic diagnosis, non-invasive prenatal testing, and cancer diagnostics based on circulating tumor cells or biopsy-derived single cells. The method integrates sequencing data alignment, GC content normalization, breakpoint detection via binary segmentation, and dynamic thresholding to produce reliable copy number profiles without reliance on paired control samples or high sequencing depth. The system further encompasses a computer-readable medium configured to execute the computational steps of the pipeline, enabling automated, reproducible, and scalable analysis across diverse clinical and research settings.

## BACKGROUND

- motivate need for improvement

Current methods for detecting copy number variations in single cells are limited by the pervasive influence of whole genome amplification artifacts, which introduce systematic distortions in sequencing read distribution that mimic true genomic alterations. These distortions are predominantly driven by variations in guanine-cytosine (GC) content across the genome, where regions of low or high GC content are either under-amplified or over-amplified during PCR-based amplification processes, leading to false-positive or false-negative calls in copy number analysis. Conventional approaches, such as array comparative genomic hybridization and comparative sequencing methods like SegSeq, attempt to correct for these artifacts by employing control samples or fixed thresholds, but these strategies remain ineffective in the context of single-cell analysis due to the absence of matched reference material and the stochastic nature of amplification bias. Furthermore, fixed thresholds fail to account for regional variation in amplification efficiency, resulting in high false discovery rates, particularly in GC-extreme chromosomal regions such as chromosome 13 and chromosome 19. The inability to distinguish true biological variation from technical noise has severely constrained the clinical utility of single-cell genomics in applications such as pre-implantation genetic screening, where misdiagnosis can lead to the transfer of embryos with pathogenic copy number alterations. There exists, therefore, a critical unmet need for a method that corrects amplification bias at the data level, identifies breakpoints with high precision, and dynamically adapts detection thresholds based on local genomic context to ensure both sensitivity and specificity in low-coverage sequencing scenarios.

## SUMMARY

- introduce method
- sequence genome sample
- align sequencing result
- determine breakpoints
- determine detection window
- determine first parameter
- determine copy number variation
- introduce system
- describe system components
- introduce computer readable medium
- describe computer readable medium function

The present invention introduces a method for determining copy number variation in a genome sample derived from a single cell or trace nucleic acid material, comprising the steps of sequencing a genome sample obtained from a biological specimen, aligning the resulting sequencing reads to a reference human genome sequence, determining the distribution of uniquely aligned reads across non-overlapping genomic windows, calculating a relative number of reads for each window, correcting the relative number of reads for GC content bias by applying a weighted normalization factor derived from empirical read density across GC-matched regions, determining a normalized number of reads for each window, performing a run test to identify candidate breakpoints between adjacent windows, iteratively removing candidate breakpoints with the highest p-values until all remaining breakpoints exhibit p-values below a predefined termination threshold, defining detection windows as genomic intervals bounded by successive screened candidate breakpoints, determining a first parameter representing the mean normalized read count within each detection window, and comparing the first parameter to a preset threshold to determine the presence or absence of copy number variation. The invention further introduces a system for performing the method, comprising a sequencing apparatus configured to generate low-coverage sequencing data from amplified single-cell genomes, a breakpoint determining unit operatively coupled to the sequencing apparatus and configured to execute the alignment, normalization, and segmentation algorithms, a GC content calculation module that divides the reference genome into regions based on GC content and computes mean relative read values for each region, a detection window determination unit that identifies intervals between screened breakpoints, and a copy number variation determination unit that evaluates the first parameter against a significance boundary to classify genomic regions as normal, deleted, or duplicated. The invention further encompasses a computer-readable medium containing instructions that, when executed by a processor, cause the system to align sequencing results to the reference genome, determine the distribution of reads across the genome, identify breakpoints based on statistical deviation in read density, define detection windows from screened breakpoints, compute the first parameter from reads within those windows, and determine the presence of copy number variation by comparing the first parameter to a predetermined threshold, thereby enabling automated, high-accuracy genomic analysis from minimal input material.

## DETAILED DESCRIPTION

- define copy number variation
- introduce method of determining copy number variation
- describe sequencing genome sample
- explain extracting genome sample from biological sample
- detail types of biological samples
- describe isolating single cell from biological sample
- explain methods of sequencing genome sample
- detail constructing sequencing-library
- describe sequencing constructed sequencing-library
- explain lysing single cell to release whole genome
- detail methods of amplifying single cell whole genome
- describe sequencing whole genome sequencing-library
- explain lengths of sequencing data
- align sequencing result to reference genome sequence
- determine distribution of reads in reference genome sequence
- calculate total number of sequencing data
- select uniquely aligned reads
- determine breakpoints in reference genome sequence
- divide reference genome sequence into primary windows
- determine reads falling into primary windows
- determine number of reads at both sides of site
- perform correlation analysis
- determine p value of site
- determine final p value
- calculate relative number of reads
- perform run test
- correct relative number of reads for GC content
- determine normalized number of reads
- perform run test on normalized number of reads
- calculate GC content of each primary window
- divide GC content into regions
- calculate mean value of relative number of reads
- determine corrected relative number of reads
- determine normalized number of reads
- define formula for Z
- define formula for mean
- define formula for SD
- eliminate bias of genome amplification
- determine possibility of copy number variation
- determine detection windows
- determine candidate breakpoints
- determine p value of each candidate breakpoint
- remove candidate breakpoint with maximal p value
- perform step 8 until p values are smaller than terminate p value
- determine region between two successive screened candidate breakpoints as detection window
- obtain p value of candidate breakpoint using run test
- determine final p value
- determine first parameter based on reads falling in detection window
- determine whether copy number variation presents in genome sample
- compare first parameter to preset threshold
- determine type of copy number variation
- set boundary of significance
- determine copy number variation in genome sample
- solve problem of analyzing single cell or trace of nucleic acid sample
- avoid bias to analyzing copy number variation
- improve detection efficiency
- introduce different indexes during constructing sequencing-library
- improve efficiency of determining copy number variation
- provide genetic counseling and basis for clinic decision
- prevent implantation of embryo with lesion
- describe system for determining copy number variation
- configure sequencing apparatus
- extract genome sample from biological sample
- amplify genome sample
- construct sequencing-library
- sequence sequencing-library
- determine whether copy number variation presents in genome sample
- align sequencing result to reference genome sequence
- define detailed description
- describe breakpoint determining unit
- calculate GC content
- divide GC content into regions
- calculate mean value of relative number of reads
- determine corrected relative number of reads
- determine normalized number of reads
- determine detection window
- determine possibility of copy number variation
- screen breakpoints
- determine p value of candidate breakpoint
- remove candidate breakpoint with maximal p value
- repeat step 12 until p values of rest of candidate breakpoints smaller than terminate p value
- determine final p value
- determine detection window based on screened candidate breakpoints
- determine first parameter based on reads falling in detection window
- determine whether copy number variation presents based on difference between first parameter and preset threshold
- describe computer readable medium
- preserve order in computer readable medium
- align sequencing result to reference genome sequence
- determine distribution of reads in reference genome sequence
- determine breakpoints based on distribution of reads
- determine detection window based on breakpoints
- determine first parameter based on reads falling in detection window
- determine whether copy number variation presents based on difference between first parameter and preset threshold
- describe general method
- amplify whole genome sample
- sequence amplified whole genome
- align reads to standard human genome reference sequence
- calculate relative number of reads
- correct and normalize data
- determine breakpoints
- screen breakpoints
- determine final p value
- determine detection window and verify detection window
- describe example 1
- perform whole genome amplification
- perform sequencing and data analysis

### Example 2

- repeat experiment without amplification
- compare and discuss results

Copy number variation refers to a structural alteration in the genome wherein a segment of DNA, typically greater than one kilobase in length, is present in a number of copies that deviates from the diploid state, resulting in either deletion or duplication of genetic material. The method of determining such variation begins with the extraction of a genome sample from a biological specimen, which may include peripheral blood, amniotic fluid, chorionic villi, blastocyst trophectoderm cells, polar bodies, or circulating tumor cells. The biological sample is processed to isolate a single cell, which is then lysed using alkaline buffer to release the entire genomic DNA. The released genome is subjected to whole genome amplification using degenerate oligonucleotide primer PCR, which generates sufficient DNA for downstream sequencing while introducing GC-dependent amplification biases. The amplified DNA is then used to construct a sequencing library with an insert size of approximately 350 base pairs, which is subsequently sequenced using a massively parallel sequencing platform to generate single-end reads of 50 base pairs in length. The resulting sequencing data, typically comprising between 4% and 9.5% genome coverage, is aligned to the human reference genome (e.g., hg18 or hg38) using a short-read aligner that permits a maximum of two mismatches per read. Only uniquely mapped reads are retained for analysis, and PCR duplicates are removed to eliminate technical artifacts. The reference genome is divided into non-overlapping primary windows of approximately 150 kilobases in size, each designed to contain a comparable number of expected sequencing reads based on simulated mapping. The number of aligned reads falling within each window is counted, and the relative number of reads is calculated as the ratio of observed reads to the global mean across autosomal windows. To correct for GC-induced amplification bias, the GC content of each window is computed and categorized into 1% increments. For each GC bin, the mean relative read number is determined across all windows sharing that GC content, and a weighting coefficient is derived to adjust the relative read number of each window toward its expected value under uniform amplification. This yields a corrected relative number of reads, which is further normalized to produce a normalized number of reads with a mean of one and a standard deviation calibrated to the data distribution. A run test is performed on the normalized read values to assess local continuity and identify statistically significant transitions indicative of breakpoints. Candidate breakpoints are identified as the 3,000 windows with the lowest p-values from the run test. An iterative screening process is then applied: the candidate breakpoint with the highest p-value is removed, and the adjacent segments are merged; this process repeats until all remaining candidate breakpoints exhibit p-values below a termination threshold determined empirically from control samples. The genomic regions bounded by successive screened breakpoints are designated as detection windows. Within each detection window, the first parameter is calculated as the mean normalized read count, which reflects the relative copy number state of that region. A preset threshold, defined as the 5th and 95th percentiles of normalized read values in diploid regions, establishes the boundary of significance: values below the lower threshold indicate deletion, and values above the upper threshold indicate duplication. The method thereby enables the determination of copy number variation without requiring a matched control sample, eliminates the confounding effects of amplification bias, and achieves high specificity and sensitivity even at low sequencing depth. The system for performing this method includes a sequencing apparatus configured to generate the sequencing data, a breakpoint determining unit that executes the alignment, normalization, and segmentation algorithms, and a computer-readable medium storing program instructions that, when executed, automate the entire analytical workflow. The method has been validated on single cells from individuals with known copy number variations, achieving 99.63% sensitivity and 97.71% specificity for variants exceeding one megabase in size. In clinical applications, this method provides a reliable basis for genetic counseling, enables the prevention of embryo implantation with pathogenic lesions, and supports non-invasive cancer screening through analysis of circulating tumor cells. In a comparative experiment conducted without whole genome amplification, the same analytical pipeline applied to unamplified genomic DNA from bulk tissue yielded identical copy number profiles, confirming that the correction strategy effectively neutralizes amplification artifacts and that the detected variations reflect true biological states rather than technical noise.

## INDUSTRIAL APPLICABILITY

- claim industrial applicability
- define embodiment scope

The method and system of the present invention are industrially applicable in clinical diagnostics, reproductive medicine, oncology, and forensic genetics, where accurate detection of copy number variations from limited biological material is essential. The invention may be implemented in diagnostic laboratories equipped with next-generation sequencing platforms and computational infrastructure, enabling routine analysis of single cells from embryos, fetal nucleated red blood cells, or tumor biopsies. The computer-readable medium containing the executable instructions may be distributed as software modules compatible with standard bioinformatics environments, and the system components may be integrated into automated diagnostic platforms for high-throughput screening. The embodiment scope encompasses all methods and systems that perform the steps of sequencing a genome sample, aligning reads to a reference genome, correcting for GC content bias using a weighted normalization derived from empirical read density, identifying breakpoints via iterative run test screening, defining detection windows, computing a first parameter based on normalized read counts, and determining copy number variation by comparison to a dynamically established significance threshold. The invention is not limited to any specific sequencing platform, reference genome version, or biological sample type, provided that the core analytical steps are implemented as described. The invention provides a commercially viable solution for pre-implantation genetic diagnosis, non-invasive prenatal testing, and early cancer detection, thereby fulfilling a critical need in precision medicine.