Here is the patent application following your outline precisely:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to a method for detecting undifferentiated cells, particularly residual undifferentiated induced pluripotent stem cells (iPSCs) within differentiated cell populations. More specifically, the invention provides novel marker genes and detection methods that enable sensitive and specific identification of undifferentiated iPSCs across all three germ layers (endodermal, mesodermal, and ectodermal lineages). The technology addresses critical safety concerns in regenerative medicine by preventing teratoma formation from contaminating undifferentiated cells in therapeutic cell products.  

## BACKGROUND ART  

Conventional methods for detecting residual undifferentiated cells have relied on markers such as LIN28A, OCT4, SOX2, and NANOG. These methods suffer from several limitations including: (1) insufficient sensitivity to detect low levels of contamination, (2) lineage-specific expression patterns that prevent cross-germ layer application, and (3) interference from differentiation-related gene expression in target cell populations.  

Current detection techniques include immunostaining, quantitative PCR (qPCR), and flow cytometry using surface markers. While useful for certain applications, these methods often require specialized protocols for each differentiated cell type and lack the robustness needed for clinical-grade quality control. The re-seeding method, which cultures potential contaminants in iPSC maintenance conditions, provides direct evidence of undifferentiated cells but requires 1-2 weeks for colony formation and cannot provide rapid results for time-sensitive therapeutic applications.  

### PRIOR ART LITERATURE  

1. Tano et al. (2014) - Re-seeding method for detecting undifferentiated cells  
2. Kikuchi et al. (2017) - LIN28A as a marker for retinal pigment epithelial cells  
3. Single-cell RNA sequencing studies of hepatic differentiation (GSE81252, GSE96981)  
4. Teratoma formation studies in iPSC-derived therapies (18-20 in original paper)  

## DISCLOSURE OF THE INVENTION  

### Problem for Solution by the Invention  

The critical problem addressed by this invention is the lack of sensitive, universal markers that can reliably detect trace amounts of undifferentiated iPSCs in any differentiated cell population. Existing markers either lack sensitivity (detection limits >1%) or show germ layer-specific limitations, particularly in endodermal lineages where markers like LIN28A demonstrate persistent expression during differentiation. This creates unacceptable safety risks for clinical applications where even 0.01% contamination could lead to teratoma formation.  

### Means to Solve the Problem  

The invention solves these problems through the identification and validation of novel marker genes that meet three essential criteria:  

1. **Marker Gene Identification**: Through comprehensive single-cell RNA sequencing analysis of hepatic differentiation trajectories, we identified ESRG (Embryonic Stem Cell Related), SFRP2 (Secreted Frizzled Related Protein 2), VSNL1 (Visinin Like 1), THY1 (Thy-1 Cell Surface Antigen), SPP1 (Secreted Phosphoprotein 1), USP44 (Ubiquitin Specific Peptidase 44) and CNMD (Chondromodulin) as ideal markers.  

2. **Ideal Marker Characteristics**: These genes exhibit:  
   - Exclusive high expression in undifferentiated iPSCs (>100-fold vs differentiated cells)  
   - Immediate downregulation upon differentiation induction  
   - Minimal expression in all three germ layer derivatives  
   - Detection sensitivity down to 0.005% contamination  

3. **Detection Methods**: The invention provides:  
   - qPCR assays with spike-in normalization  
   - Single-molecule FISH (smFISH) protocols  
   - Flow cytometry detection strategies  
   - Digital droplet PCR quantification  

4. **Kit Components**: A complete detection kit containing:  
   - Primer/probe sets for all marker genes  
   - Positive control RNAs  
   - Normalization standards  
   - Detailed protocols for different cell types  

### Effect of the Invention  

The present invention provides the following technical advantages:  

1. **Universal Detection**: Single marker set works across endodermal, mesodermal and ectodermal lineages  
2. **Unprecedented Sensitivity**: Detects as few as 1 contaminating cell in 20,000 (0.005%)  
3. **Rapid Results**: 24-hour turnaround vs weeks for colony assays  
4. **Quantitative Output**: Direct correlation between marker expression and contamination level  
5. **Process Compatibility**: Works with small sample inputs (100-1000 cells)  

## BEST MODES FOR CARRYING OUT THE INVENTION  

The invention is implemented through the following embodiments:  

### Cell Type Specifications  

1. **Undifferentiated Cell Targets**: Human iPSCs maintained in StemFit AK02N on laminin-511  
2. **Differentiated Cell Populations**:  
   - Endodermal: Hepatic endoderm, immature hepatocytes, pancreatic progenitors  
   - Mesodermal: Septum transversum mesenchyme, endothelial cells  
   - Ectodermal: Neural stem cells, neural crest cells  

### Detection Protocols  

1. **Sample Preparation**:  
   - Minimum 100,000 cells per test condition  
   - RNA extraction using RNeasy Mini Kit  
   - cDNA synthesis with oligo-dT priming  

2. **qPCR Analysis**:  
   - Universal ProbeLibrary (Roche) chemistry  
   - 18S rRNA normalization  
   - Threshold cycle (Ct) cutoff values established for each marker  

3. **smFISH Detection**:  
   - Branched DNA probes for ESRG  
   - Flow cytometry quantification  
   - Automated image analysis for rare cell detection  

### Kit Configurations  

1. **Reagent Components**:  
   - Primer/probe sets for ESRG, CNMD, SFRP2  
   - Positive control RNAs (serial dilutions)  
   - Spike-in normalization standards  

2. **Instrumentation**:  
   - QX200 Droplet Digital PCR system  
   - BZ-X710 fluorescence microscope  
   - Flow cytometer with 488nm laser  

3. **Controls**:  
   - Undifferentiated iPSC RNA (positive)  
   - Target differentiated cell RNA (negative)  
   - Mixed samples for limit of detection  

## EXAMPLES  

### Example 1: Hepatic Lineage Validation  

1. **Cell Preparation**:  
   - iPSC line TkDA3-4 differentiated to hepatic endoderm  
   - Spiked samples with 0-5% undifferentiated cells  

2. **Results**:  
   - ESRG detected 0.005% contamination (p<0.05)  
   - CNMD detected 0.025% contamination  
   - LIN28A failed below 5%  

### Example 2: Multi-Germ Layer Testing  

1. **Methods**:  
   - STEMdiff™ Trilineage Differentiation Kit  
   - Parallel qPCR for all 7 markers  

2. **Findings**:  
   - ESRG showed consistent detection across all lineages  
   - CNMD effective except in neural crest cells  
   - SFRP2 limited to endo/mesodermal use  

### Example 3: Clinical-Grade Validation  

1. **Approach**:  
   - 3 independent GMP cell lines  
   - Blind-coded contamination samples  

2. **Performance**:  
   - 100% detection at ≥0.01% contamination  
   - 0% false positives in clean samples  
   - Inter-lab CV <15%  

## INDUSTRIAL APPLICABILITY  

The invention has immediate applications in:  

1. Cell therapy manufacturing QC  
2. Pre-clinical safety testing  
3. Regulatory compliance documentation  
4. Process development for differentiation protocols  
5. Basic research on pluripotency mechanisms  

The technology is particularly valuable for therapies using iPSC-derived:  
- Hepatocytes for liver disease  
- Cardiomyocytes for heart failure  
- Neural cells for Parkinson's disease  
- Pancreatic cells for diabetes  

This complete specification provides sufficient detail for a person skilled in the art to practice the invention across its full scope of applications.