# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of molecular biology and genomics, specifically to methods and systems for detecting copy number variations (CNVs) in single cells using low coverage whole genome sequencing (WGS) and bioinformatics analysis. The invention addresses the challenges associated with whole genome amplification (WGA) biases and provides a robust pipeline for accurate CNV detection, which is particularly useful in clinical applications such as pre-implantation genetic diagnosis (PGD), pre-implantation genetic screening (PGS), and non-invasive prenatal diagnosis.

## BACKGROUND

Copy number variations (CNVs) are genomic alterations involving the deletion or duplication of large segments of DNA. These variations have been implicated in various diseases, including 22q11.2 deletion/duplication syndrome, Cri-du-Chat syndrome, and certain types of cancer. Single-cell analysis has emerged as a powerful tool for understanding disease etiology and heterogeneity, offering insights that are not possible with bulk cell population studies. Techniques such as whole genome amplification (WGA) and array comparative genomic hybridization (aCGH) have been widely used for CNV analysis in single cells. However, WGA-induced biases, particularly those related to GC content, can significantly affect the accuracy and reliability of CNV detection.

Massively parallel sequencing (MPS) has revolutionized genomics research, providing a more comprehensive and accurate approach to CNV detection compared to traditional methods. Despite its advantages, MPS data from single cells can still be confounded by WGA biases. Therefore, there is a need for a robust bioinformatics pipeline that can effectively correct for these biases and accurately detect CNVs in single cells.

## SUMMARY

The present invention provides a method and system for detecting copy number variations (CNVs) in single cells using low coverage whole genome sequencing (WGS) and a comprehensive bioinformatics pipeline. The method includes the following steps:

1. **Whole Genome Amplification (WGA):** Isolating and amplifying DNA from single cells using WGA techniques.
2. **Sequencing:** Generating sequencing reads from the amplified DNA using massively parallel sequencing (MPS).
3. **GC Bias Correction:** Developing a weighted correction strategy to remove GC biases introduced during WGA. This involves calculating a GC-related weighting coefficient and adjusting the relative read number (RRN) to obtain a corrected relative read number (CRN).
4. **Binary Segmentation Algorithm:** Employing a binary segmentation algorithm to accurately locate CNV breakpoints. This algorithm initializes candidate breakpoints and iteratively merges adjacent segments to optimize breakpoint localization.
5. **Dynamic Threshold Determination:** Defining a dynamic threshold for final signal filtering to minimize false signals and ensure accurate CNV detection. The threshold is determined based on the distribution of CRNs with the same sequencing GC content.

The invention also includes a system for implementing the above method, comprising a computer-readable medium storing instructions for executing the bioinformatics pipeline and a processor for running the instructions.

## DETAILED DESCRIPTION

### Example 2

The present invention is directed to a method and system for detecting copy number variations (CNVs) in single cells using low coverage whole genome sequencing (WGS) and a comprehensive bioinformatics pipeline. The method and system address the challenges associated with whole genome amplification (WGA) biases, particularly those related to GC content, and provide a robust and accurate approach for CNV detection.

#### Whole Genome Amplification (WGA)

The first step in the method involves isolating and amplifying DNA from single cells using WGA techniques. Single cells are typically isolated from peripheral blood (PB) or blastocysts. The isolated cells are subjected to a standard degenerate oligonucleotide primer PCR (DOP-PCR) or a similar WGA method to amplify the DNA. The quality of the WGA products is verified using PCR with primers for housekeeping genes.

#### Sequencing

The amplified DNA is then used to prepare a sequencing library, which is sequenced using massively parallel sequencing (MPS) technologies, such as Illumina sequencing. The sequencing generates single-end (SE) 50 bp reads, which are aligned to the human reference genome (HG18, NCBI Build36) using a short-read aligner like SOAP2. Low-quality alignments, such as PCR duplicates and non-unique alignments, are removed to ensure high-quality data for subsequent analysis.

#### GC Bias Correction

One of the key challenges in WGA is the introduction of biases, particularly those related to GC content. These biases can lead to over-amplification or under-amplification of GC-poor or GC-rich regions, affecting the accuracy of CNV detection. To address this issue, the method includes a weighted correction strategy to remove GC biases.

The relative read number (RRN) is defined as the quotient between the number of reads in each observation window and the average number of reads across all windows. The effect of GC bias on the RRN is quantified using a GC-bias index, which measures the average deviation between the observed RRN and its expected value. A weighted correction strategy is then applied, where the sequencing reads within each window are assigned a weight based on the GC content. The corrected reads number (CRN) is calculated by adjusting the RRN using the GC-related weighting coefficient. This correction significantly reduces the impact of GC biases, improving the accuracy of CNV detection.

#### Binary Segmentation Algorithm

After correcting for GC biases, a binary segmentation algorithm is employed to accurately locate CNV breakpoints. The algorithm consists of two main steps: initialization and iterative merging.

**Initialization:** The significance of differences between the two sides of each window is calculated using a run-test. The p-values for each window are determined, and the top 3,000 windows with the smallest p-values are selected as initial candidate breakpoints.

**Iterative Merging:** Each candidate breakpoint is associated with a left and right segment. The difference between the left and right segments is estimated using a run-test, and the p-value for each breakpoint is calculated. The breakpoint with the most insignificant difference (largest p-value) is removed, and the segments on either side are merged. This process is repeated until the p-value of each breakpoint is below a predefined threshold.

#### Dynamic Threshold Determination

To minimize false signals and ensure accurate CNV detection, a dynamic threshold is determined for the average CRN between two breakpoints. The threshold is calculated based on the distribution of CRNs with the same sequencing GC content. Specifically, the lower and upper quantiles (alpha = 0.05) of CRNs with the same GC content are used as the deletion and duplication cutoff thresholds, respectively. This dynamic thresholding approach helps to filter out false signals and improve the specificity of CNV detection.

#### Evaluation of Sensitivity and Specificity

The performance of the method is evaluated using a set of test samples, including single cells isolated from PB and blastocysts with confirmed CNV results. The sensitivity and specificity of the method are calculated by comparing the detected CNVs with the confirmed results. The method demonstrates high sensitivity (99.63%) and specificity (97.71%) for detecting CNVs larger than 1 Mb.

#### Simulations and Breakpoint Precision Analysis

To further validate the method, simulations are performed in silico to evaluate the sensitivity and specificity of CNV detection on a call level. CNVs ranging from 500 kb to 5 Mb are simulated on the YH genome, and the method is applied to detect these simulated CNVs. The sensitivity and specificity are calculated, and the precision of breakpoint localization is assessed by measuring the minimum distance between the predicted and simulated CNV breakpoints. The results show that the method can accurately detect CNVs and localize breakpoints with high precision.

## INDUSTRIAL APPLICABILITY

The method and system of the present invention have significant industrial applicability in various fields, including:

1. **Clinical Diagnostics:** The method can be used for pre-implantation genetic diagnosis (PGD) and pre-implantation genetic screening (PGS) to detect CNVs in single cells isolated from embryos. This can help identify genetic abnormalities and improve the success rates of in vitro fertilization (IVF) procedures.

2. **Non-Invasive Prenatal Diagnosis:** The method can be applied to non-invasive prenatal diagnosis using single fetal nucleated red blood cells (NRBCs) isolated from maternal blood. This approach offers a safer and more convenient alternative to invasive prenatal testing methods.

3. **Cancer Research:** The method can be used to study somatic CNVs in cancer cells, providing insights into tumor heterogeneity and evolution. This can aid in the development of personalized cancer treatments and improve patient outcomes.

4. **Genetic Research:** The method can be applied to various genetic research studies, including the identification of CNVs associated with complex diseases and the exploration of genetic diversity in populations.

In conclusion, the present invention provides a robust and accurate method for detecting CNVs in single cells using low coverage whole genome sequencing and a comprehensive bioinformatics pipeline. The method addresses the challenges associated with WGA biases and offers significant advantages over existing methods, making it a valuable tool in clinical diagnostics and genetic research.