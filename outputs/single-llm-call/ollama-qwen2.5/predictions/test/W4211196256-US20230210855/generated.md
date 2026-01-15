### 1. RNA Isolation and Purification

3T3-L1 cells were seeded in 6-well plates at 70% confluency and differentiated for 8 days. Differentiated cells were treated with vehicle (DMSO) or the MCT1 inhibitor AZD6965 (1 µM) for 24 h. RNA was isolated using TRIzol, followed by purification with the RNeasy® Plus Mini Kit. RNA quality and quantity were assessed using a Nanodrop 1000 Spectrophotometer.

### 2. mRNA Sequencing

RNA samples from treated and control cells were sent to Novogene for mRNA sequencing on the Illumina NovaSeq platform, generating paired-end 150 bp reads. Raw reads were aligned to the mm10 mouse genome using STAR software, achieving a mapping rate over 80%. Gene expression levels were quantified by FPKM values.

### 3. Differential Expression Analysis

Differential gene expression analysis was performed using DESeq2 in R. Adjusted p-values (padj) were calculated using the Benjamini and Hochberg method to control for false discovery rates. Genes with padj < 0.05 were considered significantly differentially expressed.

### 4. Ingenuity Pathway Analysis

Transcript FPKM values and significance from RNA-seq data were uploaded to QIAGEN's Ingenuity Pathway Analysis software. Annotated genes were mapped, and molecular functions analysis identified significantly altered biological functions (padj < 0.05).

### 5. Confocal Microscopy

Differentiated 3T3-L1 cells treated with vehicle or AZD6965 were fixed, permeabilized, and blocked. Cells were stained with Ki67 antibody, Alexa Fluor 647 secondary antibody, BODIPY for lipid droplets, and DAPI for nuclei. Images were acquired on a Leica TCS SP8 confocal microscope at 630× magnification.

### 6. Ki67 Expression Quantification

Slides prepared as described in the confocal microscopy section were imaged using the Cytation 5 Cell Imaging Multi-Mode reader. Ki67 expression was quantified by measuring fluorescence intensity and deep red fluorescent (Alexa647) positive pixels, normalized to the total number of nuclei.

### 7. Proliferation Assay

3T3-L1 cells were seeded in black, clear-bottom 96-well plates and differentiated. Cells were treated with various concentrations of AZD3965 for 24 h, 48 h, or 72 h. Cell viability was measured using the CyQUANT™ NF Cell Proliferation assay according to the manufacturer's protocol.

### 8. Glucose Uptake Assay

Differentiated 3T3-L1 cells were treated with or without AZD6965 for 24 h, 48 h, or 72 h. Cells were incubated in serum-free media overnight and then in glucose-free media for 2 h. Glucose uptake was stimulated with insulin (175 nM) or vehicle for 30 min and measured using the Glucose Uptake-Glo™ Assay.

### 9. Hyperplasia Assay

Differentiated 3T3-L1 cells were treated with or without AZD6965 for 24 h, 48 h, or 72 h. Media was changed to normal growth media with or without insulin (100 nM) and cultured for an additional 72 h. Intracellular triglycerides were measured using the Triglyceride-Glo™ Assay.

### 10. Statistical Analysis

Data were analyzed using Prism 8 software. Significance was determined using Student’s t-test or one-way ANOVA as appropriate, with a significance threshold of p < 0.05. Data represent the average of three experimental replicates ± SEM unless otherwise stated.