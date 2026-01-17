# DESCRIPTION

## BACKGROUND

Non-invasive prenatal screening (NIPS) has revolutionized the field of prenatal diagnostics by utilizing cell-free DNA (cfDNA) present in maternal plasma to screen for fetal chromosomal aneuploidies, microdeletions, and monogenic disorders. Current NIPS approaches primarily rely on low-coverage whole-genome sequencing (WGS) or targeted sequencing of single nucleotide polymorphisms (SNPs) to infer fetal chromosome copy numbers. However, these methods face several limitations, including allelic hybridization bias, maternal copy number variations (CNVs), and absence of heterozygosity (AOH), which can confound the detection of fetal variants. Additionally, the low fetal fraction (FF) in maternal plasma, typically around 10% during the second trimester, poses a significant challenge for accurate variant detection.

To address these limitations, a novel NIPS approach has been developed. This approach employs a coordinated allele-aware target enrichment (COATE-seq) method followed by next-generation sequencing (NGS) to comprehensively analyze fetal chromosomal aneuploidies, microdeletions, and monogenic disorders. COATE-seq reduces allelic hybridization bias, thereby improving the signal-to-noise ratio and enhancing the accuracy of fetal variant detection. Furthermore, this method integrates multiple genomic cues, including read depth (RD), allelic fraction (AF), and cfDNA fragment length, to genetically deconvolute the fetal and maternal cfDNA admixture. This comprehensive analysis addresses the limitations of current NIPS methods and provides a more robust and accurate screening tool for a wide range of genetic disorders.

### SUMMARY

The present invention relates to a method for non-invasive prenatal screening (NIPS) using a coordinated allele-aware target enrichment (COATE-seq) followed by next-generation sequencing (NGS). The method involves the following steps:
1. Extracting cell-free DNA (cfDNA) from maternal plasma.
2. Performing COATE-seq to reduce allelic hybridization bias and enrich for target genomic regions.
3. Conducting NGS to generate sequencing data.
4. Analyzing the sequencing data using a combination of read depth (RD), allelic fraction (AF), and cfDNA fragment length to detect fetal chromosomal aneuploidies, microdeletions, and monogenic disorders.
5. Genetically deconvoluting the fetal and maternal cfDNA admixture to improve the accuracy of variant detection.

The method is particularly useful for overcoming the limitations of current NIPS approaches, such as allelic hybridization bias, maternal CNVs, and AOH, and provides a more comprehensive and accurate screening tool for a broad spectrum of genetic disorders.

## DETAILED DESCRIPTION

### Computer System

The method for non-invasive prenatal screening (NIPS) can be implemented using a computer system comprising one or more processors, memory, and storage devices. The computer system is configured to execute software modules for performing the following tasks:
1. **Data Acquisition**: Collecting and processing raw sequencing data from NGS.
2. **Data Preprocessing**: Aligning reads to a reference genome, removing duplicates, and normalizing read depths.
3. **Variant Calling**: Identifying variants in the sequencing data using algorithms that account for read depth, allelic fraction, and cfDNA fragment length.
4. **Quality Control**: Ensuring the accuracy and reliability of the variant calls by filtering out low-quality data and identifying potential confounders such as maternal CNVs and AOH.
5. **Variant Interpretation**: Annotating and interpreting the identified variants to determine their clinical significance.

The computer system may also include user interfaces for inputting patient data, visualizing results, and generating reports. The system can be integrated with existing clinical workflows and databases to facilitate the seamless integration of NIPS results into patient care.

### Other Embodiments

While the invention has been described with reference to specific embodiments, it will be apparent to those skilled in the art that various modifications and variations can be made without departing from the spirit and scope of the invention. For example, the method can be adapted to target different genomic regions or to screen for additional genetic disorders. The COATE-seq probes can be designed to target specific SNPs or regions of interest, and the NGS platform can be adjusted to accommodate different sequencing depths and read lengths. Additionally, the computational algorithms can be optimized to improve the accuracy and efficiency of variant detection and interpretation.

## EXAMPLES

### Example 1: Capture of DNA with the Target Probe

In this example, the COATE-seq method was used to capture and enrich for target genomic regions in maternal plasma cfDNA. The COATE-seq probes were designed to minimize allelic hybridization bias by selecting nucleotides at the target SNP locus that result in the minimum melting temperature (Tm) difference for pairing with both the reference and alternative alleles. The probes were hybridized to the cfDNA at 65°C for 16 hours, followed by washing and purification steps to recover the enriched DNA. The recovered DNA was then used for NGS library preparation and sequencing.

### Example 2: Sequencing

The enriched cfDNA was sequenced using the MGISEQ-2000 platform with 2×100 paired-end reads. The raw sequencing data was processed using a custom bioinformatics pipeline that included quality control, alignment to the human reference genome (GRCh38), and deduplication of reads. The resulting BAM files were used for variant calling and downstream analysis.

### Example 3: The Coordinative Allele-Aware Target Enrichment Improves Capture Homogeneity of Alleles in Target Region

To evaluate the performance of COATE-seq, the allelic fraction (AF) at maternal heterozygous loci was compared between COATE-seq and conventional target enrichment (CON-seq) methods. The COATE-seq method significantly reduced the allelic bias, with the median AF approaching 0.5, indicating a more balanced capture of both reference and alternative alleles. The coefficient of variation (CV) of AF was also significantly lower in COATE-seq, demonstrating improved homogeneity of allele capture.

### Example 4: Determination of the Negative Threshold of Trisomy 21 Syndrome

To establish the negative threshold for detecting trisomy 21 (T21), a cohort of 104 samples with known fetal percentages ranging from 4.0% to 5.8% was analyzed. The samples included cases of T21, T18, T13, 22q11.2 deletion, and a monogenic variant (FGFR3: c.1138G>A). The overall detection rate for the target diseases at the case level was 99.1%, indicating that the lower detection limit of the test was 4.0%.

### Example 5: Determination of the Positive Threshold of Trisomy 21 Syndrome

To determine the positive threshold for T21, the Z-score for chromosome 21 was calculated for each sample. The Z-score was based on the ratio of reads mapping to chromosome 21 relative to the total reads mapping to reference chromosomes. A Z-score cutoff of 3 was used to identify positive cases of T21. The positive threshold was validated using a separate cohort of 724 samples, and the method achieved a sensitivity of 100% and a positive predictive value (PPV) of 80.0%.

### Example 6: Detection of Trisomy 21 Syndrome in Maternal Plasma

In a clinical validation study, 1129 samples from pregnant women were tested using the COATE-seq method. The samples were analyzed for common chromosomal aneuploidies (T21, T18, T13), microdeletions (e.g., 22q11.2 deletion), and monogenic disorders (e.g., FGFR3: c.1138G>A). Among the 70 positive cases identified, 38 were T21, 10 were T18, 6 were T13, 8 were microdeletions, and 8 were monogenic disorders. The test achieved a sensitivity of 100% and a specificity of 99.3%.

### Example 7: Detection of Trisomy in which Homologous Chromosome Recombination has Occurred

To detect trisomies associated with homologous chromosome recombination, the method was applied to 73 aneuploidy samples. The analysis revealed that 44 out of 73 (60.3%) cases had detectable recombinants, with the majority of T21 cases (63.8%) having maternal meiosis I (MI) nondisjunction (NDJ). The method accurately characterized the parental and meiotic origin of the aneuploidies, providing valuable insights into the mechanisms of meiotic errors.

### Example 8: Detection of Chromosome Microdeletion (Example of DiGeorge)

The method was also used to detect chromosome microdeletions, such as the 22q11.2 deletion associated with DiGeorge syndrome. In a cohort of 1129 samples, 8 cases of 22q11.2 deletion were identified, all of which were confirmed by invasive testing. The method accurately detected the microdeletion and provided information on the parental and meiotic origin of the deletion.

### Example 9: Detection of Dominant Monogenic Variation (FGFR3:.pG380R)

To detect dominant monogenic variations, the method was applied to 28 plasma cfDNA samples and their respective amniocytes. The method used a combination of allele count distribution (ACD) and fetal-maternal insert-size distribution (FMID) filters to identify de novo and paternally inherited variants. In this validation set, the method achieved a sensitivity of 99.5% and a positive predictive value (PPV) of 99.9% for detecting the FGFR3: c.1138G>A variant associated with achondroplasia. The method accurately identified the variant and provided information on its fetal origin.