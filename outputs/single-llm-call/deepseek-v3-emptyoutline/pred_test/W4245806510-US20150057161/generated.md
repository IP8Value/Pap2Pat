Below is the drafted patent application following the provided outline and incorporating the invention described in the research paper. The application is written in formal patent language and adheres to the required structure.

---

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

Macrophages are pivotal immune cells that play a central role in tissue homeostasis, pathogen defense, and inflammatory responses. These cells exhibit remarkable plasticity, allowing them to adopt distinct functional phenotypes in response to environmental stimuli. The classical activation of macrophages, often referred to as the M1-like phenotype, is induced by pro-inflammatory signals such as interferon-gamma (IFN-γ), lipopolysaccharide (LPS), or tumor necrosis factor-alpha (TNF-α). In contrast, alternative activation, or the M2-like phenotype, is driven by cytokines such as interleukin-4 (IL-4) and interleukin-13 (IL-13).  

Historically, the classification of macrophage polarization states has relied on a limited set of surface markers and cytokine profiles. However, these markers often fail to capture the full complexity of macrophage biology, particularly in human systems. Microarray-based transcriptome analysis has been widely used to study macrophage polarization, but this technology suffers from limitations in dynamic range, sensitivity, and the inability to detect alternative splicing events or novel transcripts.  

Recent advances in RNA sequencing (RNA-seq) have revolutionized transcriptome analysis by offering superior resolution, broader dynamic range, and the ability to detect novel transcripts and splice variants. Despite these advantages, the application of RNA-seq to systematically characterize human macrophage polarization and identify novel markers has not been fully explored. There remains a critical need for high-resolution transcriptomic data to better understand macrophage biology and to identify robust biomarkers for distinguishing M1-like and M2-like macrophages in clinical and research settings.  

## SHORT DESCRIPTION OF THE INVENTION  

The present invention provides a method for characterizing macrophage polarization states using high-resolution RNA sequencing (RNA-seq). The invention encompasses the identification of novel transcriptomic signatures, splice variants, and surface markers that distinguish M1-like and M2-like macrophages with unprecedented accuracy.  

Key aspects of the invention include:  
1. A method for generating high-resolution transcriptome profiles of M1-like and M2-like macrophages using RNA-seq.  
2. The identification of novel differentially expressed genes, including members of the apolipoprotein L (APOL) family and leukocyte immunoglobulin-like receptor (LILR) family, which serve as robust markers for macrophage polarization.  
3. The discovery of alternative splicing events and transcript variants that are specific to M1-like or M2-like macrophages.  
4. The characterization of novel surface markers, such as CD120b (TNFR2), SLAMF7, CD1a, CD1b, CD93, and CD226, which enable improved discrimination between macrophage subsets.  

The invention further provides diagnostic and therapeutic applications, including the use of these markers to monitor macrophage polarization in disease states, such as chronic inflammation, autoimmune disorders, and cancer.  

## DETAILED DESCRIPTION OF THE INVENTION  

The present invention is based on the discovery that RNA-seq provides a superior method for analyzing macrophage polarization compared to traditional microarray techniques. By applying RNA-seq to human macrophages polarized under M1-like and M2-like conditions, the inventors have uncovered novel transcriptional networks, splice variants, and surface markers that were previously undetectable using conventional methods.  

### Generation of Polarized Macrophages  
Human CD14+ monocytes are isolated from peripheral blood mononuclear cells (PBMCs) and differentiated into immature macrophages using granulocyte-macrophage colony-stimulating factor (GM-CSF) or macrophage colony-stimulating factor (M-CSF). Polarization into M1-like macrophages is achieved by stimulation with IFN-γ, LPS, or TNF-α, while M2-like polarization is induced by IL-4 or IL-13. The resulting macrophages exhibit distinct transcriptional and phenotypic profiles, as validated by flow cytometry and quantitative PCR (qPCR).  

### RNA-Seq Analysis  
Total RNA is extracted from polarized macrophages and subjected to RNA-seq using Illumina-based sequencing. The resulting data are analyzed to quantify gene expression levels, identify differentially expressed genes, and detect alternative splicing events. Key bioinformatics tools, such as Cufflinks and Cuffdiff, are employed to compare transcriptomes between M1-like and M2-like macrophages.  

### Novel Transcriptional Signatures  
RNA-seq reveals a significantly higher number of differentially expressed genes compared to microarray analysis, including genes with low expression levels or those not represented on microarrays. Notably, the apolipoprotein L (APOL) family and leukocyte immunoglobulin-like receptor (LILR) family are identified as novel markers of macrophage polarization. These genes exhibit strong differential expression between M1-like and M2-like macrophages and are implicated in immune regulation and pathogen defense.  

### Alternative Splicing and Transcript Variants  
The invention further identifies differential usage of splice variants in polarized macrophages. For example, the gene encoding PDZ and LIM domain 7 (PDLIM7) exhibits distinct transcript variants in M1-like and M2-like macrophages, suggesting a role for alternative splicing in macrophage functional specialization.  

### Surface Marker Discovery  
By focusing on the human surfaceome, the invention identifies novel surface markers that distinguish M1-like and M2-like macrophages. These include CD120b (TNFR2) and SLAMF7 as M1-specific markers, and CD1a, CD1b, CD93, and CD226 as M2-specific markers. These markers enable improved discrimination of macrophage subsets and have potential applications in diagnostics and therapeutics.  

### Applications  
The invention has broad applications in research and medicine, including:  
- Monitoring macrophage polarization in inflammatory diseases, autoimmune disorders, and cancer.  
- Developing targeted therapies that modulate macrophage function.  
- Improving the characterization of macrophage subsets in clinical samples.  

### EXAMPLES  

#### Example 1  

**Isolation and Polarization of Human Macrophages**  
CD14+ monocytes were isolated from healthy donor buffy coats using CD14-specific magnetic beads. The monocytes were cultured in RPMI1640 medium supplemented with 10% fetal calf serum (FCS) and differentiated into immature macrophages using GM-CSF (500 U/ml) for 3 days. Polarization into M1-like macrophages was induced by treatment with IFN-γ (200 U/ml) for an additional 3 days, while M2-like polarization was achieved using IL-4 (1,000 U/ml). The resulting macrophages were characterized by flow cytometry for surface markers such as CD64 (M1 marker) and CD23 (M2 marker).  

#### Example 2  

**RNA-Seq Library Preparation and Sequencing**  
Total RNA was extracted from M1-like and M2-like macrophages using TRIZOL and purified using the MinElute Reaction Cleanup Kit. RNA-seq libraries were prepared using the Illumina TruSeq RNA Sample Preparation Kit, and sequencing was performed on an Illumina HiScanSQ system. Paired-end reads (100 bp) were generated and aligned to the human reference genome (hg19) using CASAVA 1.8.  

#### Example 3  

**Identification of Differentially Expressed Genes**  
RNA-seq data were analyzed to identify genes differentially expressed between M1-like and M2-like macrophages. A fold-change threshold of ≥2 and a p-value <0.05 were applied. The analysis revealed 1,736 genes upregulated in M1-like macrophages and 822 genes upregulated in M2-like macrophages. Key findings included the upregulation of APOL and LILR family genes in M1-like macrophages.  

#### Example 4  

**Detection of Alternative Splicing Events**  
Cufflinks and Cuffdiff were used to analyze alternative splicing in polarized macrophages. The gene PDLIM7 exhibited differential usage of transcript variants, with PDLIM7 v1 predominantly expressed in M1-like macrophages and PDLIM7 v2 in M2-like macrophages. This was confirmed by isoform-specific qPCR.  

#### Example 5  

**Validation of Novel Surface Markers**  
Flow cytometry was used to validate the expression of novel surface markers identified by RNA-seq. CD120b (TNFR2) and SLAMF7 were confirmed as M1-specific markers, while CD1a, CD1b, CD93, and CD226 were validated as M2-specific markers.  

#### Example 6  

**Network Analysis of M1-Associated Genes**  
A priori knowledge-based network analysis was performed using EGAN software. The M1-associated network revealed clusters of immunomodulatory genes, including APOL and LILR family members, which were not detected by microarray analysis.  

#### Example 7  

**Comparison of RNA-Seq and Microarray Data**  
RNA-seq data were compared to microarray data from the same samples. RNA-seq identified significantly more differentially expressed genes and provided a broader dynamic range, particularly for low-abundance transcripts.  

#### Example 8  

**Clinical Applications of Macrophage Markers**  
The novel markers identified in this invention were applied to clinical samples from patients with inflammatory diseases. Elevated expression of M1-specific markers (e.g., SLAMF7) was observed in tissue macrophages from patients with graft rejection, supporting the utility of these markers in diagnostic applications.  

---  

This patent application provides a comprehensive and detailed description of the invention, including methodologies, novel discoveries, and applications. The examples demonstrate the practical implementation and validation of the invention.