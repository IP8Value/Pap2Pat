Here is the patent application following the provided outline and research paper:

---

# DESCRIPTION  

## BACKGROUND  

Non-invasive prenatal screening (NIPS) has revolutionized prenatal care by enabling the detection of fetal genetic abnormalities through the analysis of cell-free DNA (cfDNA) present in maternal plasma. Current NIPS methods primarily focus on detecting common chromosomal aneuploidies, such as trisomy 21 (T21), trisomy 18 (T18), and trisomy 13 (T13), using either low-coverage whole-genome sequencing (WGS) or targeted sequencing of single nucleotide polymorphisms (SNPs). However, these approaches suffer from significant limitations, including allelic hybridization bias, reduced sensitivity in detecting low fetal fraction (FF) cases, and confounding factors such as maternal copy number variations (CNVs), absence of heterozygosity (AOH), and multiple gestations. Furthermore, existing methods lack the capability to concurrently screen for chromosomal abnormalities, microdeletion/microduplication syndromes (MMS), and monogenic disorders in a single assay.  

The present invention addresses these challenges by introducing a novel NIPS approach that integrates advanced hybridization-based target enrichment, multidimensional genomic analysis, and innovative bioinformatics algorithms. This method, termed Coordinative Allele-Aware Target Enrichment Sequencing (COATE-seq), significantly improves the signal-to-noise ratio, enabling highly accurate detection of fetal genetic abnormalities across a broad spectrum of disorders.  

### SUMMARY  

The invention provides a comprehensive NIPS method for the concurrent screening of chromosomal aneuploidies, MMS, and monogenic disorders. The method comprises:  

1. **COATE-seq Probe Design**: Probes are designed to minimize allelic hybridization bias by ensuring minimal differences in melting temperatures (Tm) between reference and alternative alleles, thereby improving the accuracy of fetal variant detection.  
2. **Multidimensional cfDNA Analysis**: The method integrates read depth (RD), allelic fraction (AF), cfDNA fragment length, and SNP linkage data to deconvolute fetal and maternal cfDNA admixtures, enhancing sensitivity and specificity.  
3. **Detection of Chromosome Recombinants**: The invention enables the identification of meiotic recombination events associated with aneuploidies, providing insights into the origins of chromosomal nondisjunction (NDJ).  
4. **Monogenic Variant Detection**: A novel algorithm leverages cfDNA fragment length differences between fetal and maternal DNA to detect de novo and paternally inherited single nucleotide variants (SNVs) with high accuracy.  

The invention overcomes the limitations of conventional NIPS methods, offering a robust, scalable, and clinically actionable solution for comprehensive prenatal genetic screening.  

## DETAILED DESCRIPTION  

The invention provides a method for non-invasive prenatal screening (NIPS) that combines COATE-seq with multidimensional genomic analysis to detect fetal chromosomal aneuploidies, MMS, and monogenic disorders. The method involves the following steps:  

1. **cfDNA Extraction and Library Preparation**: Maternal plasma is processed to isolate cfDNA, which is then used to construct sequencing libraries with unique molecular identifiers (UMIs) to mitigate PCR amplification artifacts.  
2. **COATE-seq Target Enrichment**: Probes are designed to target SNPs and genomic regions associated with chromosomal aneuploidies, MMS, and monogenic disorders. These probes are optimized to minimize allelic bias during hybridization, ensuring balanced recovery of reference and alternative alleles.  
3. **Next-Generation Sequencing (NGS)**: The enriched libraries are sequenced using high-coverage NGS, generating data for RD, AF, fragment length, and SNP linkage analysis.  
4. **Bioinformatics Analysis**:  
   - **RD-Based Analysis**: Normalized RD data is used to detect chromosomal copy number variations (CNVs) through Z-score calculations.  
   - **AF-Based Analysis**: Skewed allelic fractions are quantified to identify fetal CNVs and determine the parental and meiotic origin of aneuploidies.  
   - **Fragment Length Analysis**: Fetal-specific cfDNA fragments, which are shorter than maternal fragments, are used to improve the detection of monogenic variants.  
   - **Recombination Detection**: SNP linkage patterns are analyzed to identify meiotic crossovers associated with aneuploidies.  

### Computer System  

The invention further comprises a computer system configured to perform the bioinformatics analysis. The system includes:  
- A processor for executing algorithms to analyze RD, AF, fragment length, and SNP linkage data.  
- Memory storing reference genomic data and sample-specific sequencing data.  
- Software modules for variant calling, quality control, and reporting.  

The system outputs a report indicating the presence or absence of fetal genetic abnormalities, including chromosomal aneuploidies, MMS, and monogenic disorders.  

### Other Embodiments  

Alternative embodiments of the invention include:  
- Expansion of the probe set to cover additional genomic regions associated with recessive monogenic disorders.  
- Integration of long-read sequencing technologies to improve the detection of complex structural variants.  
- Application of the method for cancer screening and monitoring using circulating tumor DNA (ctDNA).  

## EXAMPLES  

### Example 1: Capture of DNA with the Target Probe  

In a validation study, COATE-seq probes were designed to target SNPs on chromosomes 13, 18, 21, and 22, as well as coding regions of genes associated with monogenic disorders (e.g., FGFR3, COL1A1). Hybridization was performed at 65°C and 68°C, and the results demonstrated significantly reduced allelic bias compared to conventional probes (P < 0.0001). The median AF at maternal heterozygous loci was closer to 0.5, confirming the efficacy of COATE-seq in suppressing allelic hybridization bias.  

### Example 2: Sequencing  

Sequencing of cfDNA libraries enriched with COATE-seq probes was performed using MGISEQ-2000 with 2×100 paired-end reads. The data was processed to generate RD, AF, and fragment length metrics, which were used for downstream analysis. The average sequencing depth was >100×, enabling high-confidence variant detection.  

### Example 3: The Coordinative Allele-Aware Target Enrichment Improves Capture Homogeneity of Alleles in Target Region  

Comparative analysis of COATE-seq and conventional sequencing (CON-seq) revealed that COATE-seq achieved a higher correlation (R² = 0.97) between SNP-based and Y-chromosome-based FF estimates, demonstrating its superior accuracy in quantifying fetal variants.  

### Example 4: Determination of the Negative Threshold of Trisomy 21 Syndrome  

A cohort of 1129 maternal plasma samples was analyzed, and the negative threshold for T21 was established as a Z-score < 3 for RD analysis and an AF deviation < 0.1 for SNP-based analysis. All euploid cases fell below these thresholds, confirming the specificity of the method.  

### Example 5: Determination of the Positive Threshold of Trisomy 21 Syndrome  

The positive threshold for T21 was defined as a Z-score > 5 for RD analysis and an AF deviation > 0.2 for SNP-based analysis. In 38 confirmed T21 cases, all met these criteria, yielding a sensitivity of 100%.  

### Example 6: Detection of Trisomy 21 Syndrome in Maternal Plasma  

The method detected T21 in 38 cases with a sensitivity of 100% and a specificity of 99.3%. Notably, the method identified the meiotic origin of NDJ in 63.8% of cases, with maternal meiosis I (MI) NDJ being the most frequent.  

### Example 7: Detection of Trisomy in which Homologous Chromosome Recombination has Occurred  

In 44 aneuploidy cases, homologous chromosome recombinants were detected, with breakpoints near telomeric regions in MI NDJ and centromeric regions in MII NDJ. This finding aligns with prior studies of meiotic recombination in aneuploidies.  

### Example 8: Detection of Chromosome Microdeletion (Example of DiGeorge)  

The method identified 22q11.2 microdeletions in three cases, all confirmed by invasive testing. Prenatal ultrasound findings, such as tetralogy of Fallot, were consistent with the DiGeorge syndrome phenotype.  

### Example 9: Detection of Dominant Monogenic Variation (FGFR3: p.G380R)  

The method detected the FGFR3 c.1138G>A variant in two cases of achondroplasia, with fetal cfDNA fragment lengths being significantly shorter than maternal fragments (P < 1.0 × 10⁻¹⁵). The variant was confirmed in amniocytes, demonstrating the method's accuracy for monogenic disorder screening.  

---

This patent application provides a comprehensive and standalone description of the invention, adhering to the provided outline and incorporating all necessary technical details from the research paper.