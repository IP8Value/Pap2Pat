# DESCRIPTION

## TECHNOLOGY FIELD

The present invention relates to the field of oncology and specifically to the identification and use of long non-coding RNAs (lncRNAs) as therapeutic targets for enhancing the efficacy of radiotherapy in the treatment of malignant glioma. The invention provides methods for identifying lncRNAs that sensitize glioma cells to radiation, pharmaceutical compositions comprising antisense oligonucleotides (ASOs) targeting these lncRNAs, and systems for evaluating the therapeutic efficacy and potential toxicity of these ASOs in a three-dimensional (3D) human brain organoid model.

## BACKGROUND

Malignant glioma, a primary cancer of the central nervous system (CNS), is a fatal diagnosis for most patients. Despite surgical intervention and adjuvant therapies such as fractionated radiation, the median survival for adults with glioblastoma (GBM) is only 14 months. In children, the most common malignant glioma is diffuse intrinsic pontine glioma (DIPG), which is primarily treated with radiotherapy. However, the median survival for DIPG patients is only 9-10 months, and few patients survive more than 2 years after diagnosis. While radiation is a critical component of the treatment of both adult and pediatric malignant gliomas, the toxicity of radiation to normal brain cells limits the total dose that can be delivered, and glioma cells that survive radiation often lead to tumor recurrence.

Long non-coding RNAs (lncRNAs) are transcripts longer than 200 nucleotides that do not encode proteins. Certain lncRNAs play key roles in the pathogenesis of cancer and exhibit highly cell type-specific expression and function. Systematic functional screens are necessary to identify lncRNAs that can sensitize cancer cells to radiation. CRISPR-based technologies, such as CRISPR interference (CRISPRi), have enabled genome-scale screens of gene function in mammalian cells, including the identification of non-coding genes that modify cellular phenotypes. However, the potential of lncRNAs to increase the efficacy of ionizing radiation in cancer therapy has not been systematically studied at a large scale.

## SUMMARY

The present invention provides a method for identifying lncRNAs that sensitize glioma cells to radiotherapy. The method involves performing a CRISPRi-based radiation modifier screen to identify lncRNAs that modify cell growth in the presence of radiation. Specifically, the invention identifies lncRNA Glioma Radiation Sensitizers (lncGRS) that, when knocked down, inhibit the growth of glioma cells and sensitize these cells to the therapeutic effects of radiation.

The invention further provides pharmaceutical compositions comprising antisense oligonucleotides (ASOs) targeting lncGRS, which are effective in inhibiting the growth of glioma cells and sensitizing these cells to radiation therapy. The ASOs are designed to degrade complementary RNAs via a ribonuclease H-based mechanism and are suitable for use in treating human CNS diseases.

Additionally, the invention provides a three-dimensional (3D) human brain organoid model of malignant glioma for evaluating the therapeutic efficacy and potential toxicity of lncGRS-targeting ASOs. The model is assembled from mature neural cell types derived from induced pluripotent stem cells (iPSCs) and supports the growth of human glioma cells in a 3D tissue environment, mimicking the mature brain tissue of glioma patients.

## DETAILED DESCRIPTION OF EMBODIMENTS

### Definitions

- **lncRNA (long non-coding RNA):** A transcript longer than 200 nucleotides that does not encode for a protein.
- **CRISPRi (CRISPR interference):** A technique that uses a catalytically inactive Cas9 (dCas9) fused to a repressor domain (e.g., KRAB) to repress transcription of target genes.
- **lncGRS (lncRNA Glioma Radiation Sensitizer):** A lncRNA that, when knocked down, inhibits the growth of glioma cells and sensitizes these cells to the therapeutic effects of radiation.
- **ASO (antisense oligonucleotide):** A short, synthetic nucleic acid sequence designed to bind to a specific target RNA and induce its degradation via a ribonuclease H-based mechanism.
- **MBO (Mature Brain Organoid):** A 3D tissue model assembled from mature neural cell types derived from iPSCs, used to evaluate the therapeutic efficacy and potential toxicity of lncGRS-targeting ASOs.

### Method for Identification of a Radiotherapy Sensitizer

The method for identifying lncRNAs that sensitize glioma cells to radiotherapy involves the following steps:

1. **Cell Line Preparation:** Establish a glioma cell line (e.g., U87) stably expressing dCas9-KRAB, a fusion protein of catalytically inactive Cas9 and the KRAB repressor domain.
2. **Library Construction:** Construct a CRISPRi Non-Coding Library targeting 5689 lncRNA loci, with 10 sgRNAs per lncRNA transcriptional start site (TSS) and 1202 non-targeting control sgRNAs.
3. **Screen Design:** Infect the U87-dCas9-KRAB cells with the CRISPRi library, select for infected cells with puromycin, and treat the cells with 8 Gy fractionated radiation (2 Gy doses every other day).
4. **Data Collection:** Perform targeted next-generation sequencing of sgRNA barcodes at the beginning and end of the screen to identify lncRNA hits that modify cell growth in the presence of radiation.
5. **Hit Selection:** Analyze the data to identify lncRNA hits that negatively affect cell culture growth when combined with radiation. Remove neighbor hits (lncRNAs within 1 kb of an expressed protein-coding gene) from further analysis.
6. **Validation:** Validate the top hits by comparing their screen scores in the presence and absence of radiation. Prioritize lncRNAs that are expressed in primary glioma cells and have a higher sensitizer score (ratio of the radiation modifier screen score to the growth screen score).

### Methods of Diagnosis and Treatment

The invention provides methods for diagnosing and treating malignant glioma using lncGRS-targeting ASOs. The methods include:

1. **Diagnosis:** Identify the expression levels of lncGRS in patient samples using quantitative PCR (qPCR) or other molecular biology techniques. Elevated levels of lncGRS may indicate a higher likelihood of response to lncGRS-targeting ASOs.
2. **Treatment:** Administer lncGRS-targeting ASOs to patients with malignant glioma. The ASOs can be delivered intravenously or directly into the cerebrospinal fluid. The treatment can be combined with standard radiotherapy to enhance the therapeutic effects of radiation.

### Pharmaceutical Compositions

The invention provides pharmaceutical compositions comprising antisense oligonucleotides (ASOs) targeting lncGRS. The ASOs are designed to degrade complementary RNAs via a ribonuclease H-based mechanism and are effective in inhibiting the growth of glioma cells and sensitizing these cells to radiation therapy. The pharmaceutical compositions may include:

- **Active Ingredient:** lncGRS-targeting ASOs
- **Excipients:** Suitable carriers, diluents, and excipients for intravenous or intrathecal administration
- **Formulations:** Solutions, suspensions, or lyophilized powders for reconstitution

### Three-Dimensional Human Brain Organoid of Malignant Glioma

The invention provides a three-dimensional (3D) human brain organoid model of malignant glioma for evaluating the therapeutic efficacy and potential toxicity of lncGRS-targeting ASOs. The model is assembled from mature neural cell types derived from induced pluripotent stem cells (iPSCs) and supports the growth of human glioma cells in a 3D tissue environment. The steps for generating the MBO include:

1. **iPSC Differentiation:** Differentiate iPSCs into mature astrocytes (iAstrocytes) and mature cortical neurons (i3Neurons) using established protocols.
2. **Organoid Assembly:** Combine iAstrocytes and i3Neurons in a 1:1 ratio to form mature brain organoids (AN-MBOs). Alternatively, use iAstrocytes alone to form astrocyte mature brain organoids (A-MBOs).
3. **Tumor Seeding:** Seed RFP-labeled glioma cells onto the surface of the MBOs and allow the tumor cells to grow invasively within the organoid tissue.
4. **ASO Treatment:** Transfect the MBOs with lncGRS-targeting ASOs at regular intervals (e.g., every 7 days) and monitor the growth of the RFP-labeled tumors using fluorescence microscopy.
5. **Radiation Therapy:** Evaluate the therapeutic efficacy of lncGRS-targeting ASOs in combination with fractionated radiation therapy by treating the MBOs with clinically relevant doses of radiation.

### Systems

The invention provides systems for implementing the methods described herein. The systems may include:

1. **CRISPRi Screening Platform:** A high-throughput platform for performing CRISPRi-based radiation modifier screens to identify lncGRS.
2. **ASO Delivery System:** A system for delivering lncGRS-targeting ASOs to patient samples or 3D brain organoids, including devices for intravenous or intrathecal administration.
3. **3D Brain Organoid Model:** A system for generating and maintaining 3D human brain organoids, including equipment for culturing iPSCs, differentiating them into mature neural cell types, and assembling the organoids.
4. **Imaging and Analysis System:** A system for monitoring the growth of RFP-labeled tumors in 3D brain organoids, including fluorescence microscopes and image analysis software.

## EXAMPLES

### Example 1

#### CRISPRi-Based Radiation Modifier Screen

1. **Cell Line Preparation:** U87 cells were stably transfected with a plasmid expressing dCas9-KRAB.
2. **Library Construction:** A CRISPRi Non-Coding Library targeting 5689 lncRNA loci was constructed, with 10 sgRNAs per lncRNA TSS and 1202 non-targeting control sgRNAs.
3. **Screen Design:** U87-dCas9-KRAB cells were infected with the CRISPRi library, selected with puromycin, and treated with 8 Gy fractionated radiation (2 Gy doses every other day).
4. **Data Collection:** Genomic DNA was harvested at the beginning and end of the screen, and sgRNA barcodes were sequenced.
5. **Hit Selection:** Data analysis identified 467 lncRNA hits that modified cell growth in the presence of radiation. Neighbor hits were removed, and 33 lncRNA hits were identified as sensitizers.
6. **Validation:** The top 9 lncGRS candidates were validated by comparing their screen scores in the presence and absence of radiation. lncGRS-1 (CTC-338M12.4) was prioritized for further study.

#### ASO-Mediated Knockdown of lncGRS-1

1. **ASO Design:** Locked nucleic acid ASOs targeting lncGRS-1 were designed and synthesized.
2. **Knockdown Efficiency:** ASOs were transfected into patient-derived GBM (SF10360) and DIPG (SF8628) cells, and knockdown efficiency was confirmed by qPCR.
3. **Growth Inhibition:** ASO-mediated knockdown of lncGRS-1 inhibited the growth of GBM and DIPG cells, with an average reduction of 80% in cell proliferation.
4. **Toxicity Assessment:** ASO-mediated knockdown of lncGRS-1 did not affect the proliferation or viability of normal human astrocytes (NHAs).

#### 3D Brain Organoid Model

1. **iPSC Differentiation:** WTC11 iPSCs were differentiated into mature astrocytes (iAstrocytes) and mature cortical neurons (i3Neurons).
2. **Organoid Assembly:** AN-MBOs were generated by combining iAstrocytes and i3Neurons in a 1:1 ratio.
3. **Tumor Seeding:** RFP-labeled U87 cells were seeded onto the surface of AN-MBOs and allowed to grow invasively.
4. **ASO Treatment:** ASOs targeting lncGRS-1 were transfected into the AN-MBOs every 7 days, and the growth of RFP-labeled tumors was monitored using fluorescence microscopy.
5. **Radiation Therapy:** AN-MBOs were treated with clinically relevant doses of fractionated radiation (18 Gy in 9 fractions or 12 Gy in 6 fractions) in combination with lncGRS-1-targeting ASOs.
6. **Efficacy Evaluation:** The combination of lncGRS-1-targeting ASOs and radiation therapy significantly reduced the tumor burden in AN-MBOs compared to radiation alone.

This example demonstrates the effectiveness of the CRISPRi-based radiation modifier screen in identifying lncGRS-1 as a glioma-specific therapeutic target and the utility of the 3D brain organoid model for evaluating the therapeutic efficacy and potential toxicity of lncGRS-1-targeting ASOs.