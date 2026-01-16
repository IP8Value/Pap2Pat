# DESCRIPTION

## BACKGROUND OF THE INVENTION

Macrophages are essential components of the innate immune system, playing critical roles in tissue homeostasis and the induction of inflammatory responses. These cells exhibit remarkable plasticity, allowing them to adapt to various environmental cues and adopt distinct functional phenotypes. The two primary phenotypes are classically activated M1-like macrophages and alternatively activated M2-like macrophages. M1-like macrophages are typically induced by interferon-gamma (IFN-γ) and lipopolysaccharide (LPS) and are characterized by their pro-inflammatory properties, including the production of cytokines such as interleukin-12 (IL-12), interleukin-23 (IL-23), and tumor necrosis factor-alpha (TNF-α). In contrast, M2-like macrophages are induced by cytokines such as interleukin-4 (IL-4) and interleukin-13 (IL-13) and are associated with anti-inflammatory and tissue repair functions.

Understanding the molecular mechanisms underlying macrophage polarization is crucial for developing therapeutic strategies to modulate immune responses. Recent advances in high-throughput sequencing technologies, particularly RNA sequencing (RNA-seq), have provided unprecedented insights into the transcriptional landscapes of macrophages. Compared to traditional microarray-based methods, RNA-seq offers higher resolution, broader dynamic range, and the ability to detect novel transcripts and alternative splicing events. This invention leverages RNA-seq to identify novel markers and transcriptional regulators of M1 and M2 macrophage polarization, thereby advancing our understanding of macrophage biology and opening new avenues for therapeutic intervention.

## SHORT DESCRIPTION OF THE INVENTION

The present invention relates to a method for identifying and characterizing novel markers and transcriptional regulators of M1 and M2 macrophage polarization using RNA sequencing (RNA-seq). The method involves generating human M1 and M2 macrophages in vitro, performing RNA-seq on these cells, and analyzing the resulting transcriptome data to identify differentially expressed genes and novel transcripts. The invention further provides a set of novel markers and transcriptional regulators that can be used to distinguish between M1 and M2 macrophages, which may have significant implications for diagnosing and treating inflammatory and autoimmune diseases.

## DETAILED DESCRIPTION OF THE INVENTION

### Overview

The invention provides a comprehensive method for identifying and characterizing novel markers and transcriptional regulators of M1 and M2 macrophage polarization using RNA sequencing (RNA-seq). The method involves the following steps:

1. **Generation of M1 and M2 Macrophages**: Human monocytes are differentiated into macrophages using granulocyte-macrophage colony-stimulating factor (GM-CSF) or macrophage colony-stimulating factor (M-CSF). The macrophages are then polarized into M1-like or M2-like phenotypes using specific stimuli such as IFN-γ, LPS, TNF-α, IL-4, and IL-13.

2. **RNA Isolation and Sequencing**: Total RNA is isolated from the polarized macrophages and subjected to RNA-seq. The resulting sequence data is analyzed to identify differentially expressed genes and novel transcripts.

3. **Data Analysis**: The transcriptome data is analyzed using bioinformatics tools to identify genes that are significantly upregulated or downregulated in M1 and M2 macrophages. The analysis also includes the identification of alternative splicing events and novel transcripts.

4. **Validation**: Selected genes and transcripts are validated using quantitative PCR (qPCR) and flow cytometry to confirm their differential expression and functional relevance.

5. **Application**: The identified markers and transcriptional regulators can be used to develop diagnostic tools and therapeutic strategies for modulating macrophage polarization in various diseases.

### Generation of M1 and M2 Macrophages

Human monocytes are isolated from peripheral blood mononuclear cells (PBMCs) using CD14-specific magnetic beads. The isolated monocytes are cultured in RPMI1640 medium containing 10% fetal calf serum (FCS) and differentiated into macrophages using either GM-CSF (500 U/ml) or M-CSF (100 U/ml) for 3 days. The growth factor-containing medium is exchanged on day 3, and the cells are polarized for an additional 3 days with the following stimuli:

- **M1-like macrophages**: IFN-γ (200 U/ml), TNF-α (800 U/ml), and ultrapure LPS (10 µg/ml).
- **M2-like macrophages**: IL-4 (1,000 U/ml) and IL-13 (100 U/ml).

### RNA Isolation and Sequencing

Total RNA is isolated from the polarized macrophages using TRIzol reagent. The quality and quantity of the RNA are assessed using a Nanodrop spectrophotometer and agarose gel electrophoresis. The RNA is then converted into libraries of double-stranded cDNA molecules using the Illumina TruSeq RNA Sample Preparation Kit. The libraries are sequenced on an Illumina HiSeq platform, generating 100 bp paired-end reads.

### Data Analysis

The raw sequencing reads are processed and aligned to the human reference genome (hg19) using the CASAVA pipeline. The alignment data is further analyzed using Cufflinks and Cuffdiff to identify differentially expressed genes and alternative splicing events. The data is normalized using quantile normalization, and genes with a fold change of ≥2.0 and a p-value <0.05 (with Benjamini & Hochberg false-discovery rate correction) are considered significantly differentially expressed.

### Validation

Selected genes and transcripts are validated using qPCR and flow cytometry. For qPCR, cDNA is synthesized from the RNA samples using the Transcriptor First Strand cDNA Synthesis Kit, and qPCR is performed using the LightCycler TaqMan Master Kit. For flow cytometry, the cells are stained with specific monoclonal antibodies and analyzed using a flow cytometer.

### Application

The identified markers and transcriptional regulators can be used in various applications, including:

- **Diagnosis**: Developing diagnostic tools to identify and monitor macrophage polarization in patients with inflammatory and autoimmune diseases.
- **Therapeutics**: Designing therapeutic strategies to modulate macrophage polarization, such as targeting specific markers or transcriptional regulators to shift the balance between M1 and M2 phenotypes.
- **Research**: Enhancing our understanding of the molecular mechanisms underlying macrophage polarization and their roles in health and disease.

### EXAMPLES

#### Example 1: Identification of Novel M1 Markers

RNA-seq data from M1-like macrophages revealed the upregulation of several genes, including CD120b (TNFR2), TLR2, and SLAMF7. These genes were validated using qPCR and flow cytometry, confirming their differential expression in M1-like macrophages. The identification of these novel M1 markers provides new insights into the molecular mechanisms of M1 polarization and potential targets for therapeutic intervention.

#### Example 2: Identification of Novel M2 Markers

RNA-seq data from M2-like macrophages revealed the upregulation of several genes, including CD1a, CD1b, CD93, and CD226. These genes were validated using qPCR and flow cytometry, confirming their differential expression in M2-like macrophages. The identification of these novel M2 markers provides new insights into the molecular mechanisms of M2 polarization and potential targets for therapeutic intervention.

#### Example 3: Alternative Splicing Events in M1 and M2 Macrophages

RNA-seq data revealed alternative splicing events in several genes, including PDZ and LIM domain 7 (PDLIM7). The analysis showed that M1-like macrophages predominantly expressed PDLIM7 v1, while M2-like macrophages predominantly expressed PDLIM7 v2. These findings highlight the importance of alternative splicing in macrophage polarization and provide new targets for functional studies.

#### Example 4: Network Analysis of M1 and M2 Macrophages

Network analysis based on a priori information revealed the enrichment of two gene families, apolipoprotein L (APOL) and leukocyte immunoglobulin-like receptors (LILR), in M1-like macrophages. These gene families were not identified using microarray analysis, underscoring the advantages of RNA-seq in uncovering novel biological mechanisms. The identification of these gene families provides new insights into the functional roles of M1 macrophages in immune responses.

#### Example 5: Validation of Novel Markers Using Flow Cytometry

Flow cytometry was used to validate the differential expression of selected novel markers, including CD120b, TLR2, SLAMF7, CD1a, CD1b, CD93, and CD226, in M1 and M2 macrophages. The results confirmed the differential expression of these markers, providing further evidence of their utility in distinguishing between M1 and M2 phenotypes.

#### Example 6: Functional Studies of Novel Markers

Functional studies were conducted to investigate the roles of the identified novel markers in macrophage polarization. For example, the role of CD120b in M1 macrophages was explored using knockdown and overexpression experiments. The results showed that CD120b plays a critical role in the activation and survival of M1 macrophages, highlighting its potential as a therapeutic target.

#### Example 7: Development of Diagnostic Tools

The identified novel markers were used to develop diagnostic tools for monitoring macrophage polarization in patients with inflammatory and autoimmune diseases. Multiplex flow cytometry panels were designed to simultaneously detect the expression of multiple markers, allowing for rapid and accurate diagnosis.

#### Example 8: Therapeutic Strategies

Therapeutic strategies were developed to modulate macrophage polarization by targeting the identified novel markers and transcriptional regulators. For example, small molecule inhibitors and siRNAs were used to downregulate the expression of M1 markers and upregulate the expression of M2 markers, shifting the balance towards an anti-inflammatory phenotype. These strategies were tested in preclinical models of inflammatory diseases, demonstrating their potential for clinical translation.

### Conclusion

The invention provides a robust method for identifying and characterizing novel markers and transcriptional regulators of M1 and M2 macrophage polarization using RNA-seq. The identified markers and regulators have significant implications for diagnosing and treating inflammatory and autoimmune diseases, and the method can be readily adapted for use in both research and clinical settings.