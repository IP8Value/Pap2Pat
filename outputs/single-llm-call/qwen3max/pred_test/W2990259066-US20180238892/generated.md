# DESCRIPTION

## FIELD OF INVENTION

The present invention resides in the field of molecular oncology and precision medicine, specifically relating to methods and systems for diagnosing, prognosing, and treating cancer through the evaluation of gene expression signatures associated with the BCL2 antagonist of cell death (BAD) apoptosis pathway. More particularly, the invention provides a novel BAD Pathway Gene Expression Signature (BPGES) score derived from principal component analysis of a defined set of genes functionally linked to BAD phosphorylation status and apoptotic regulation. The invention further encompasses methods for using this signature to stratify patients based on cancer subtype, predict clinical outcomes such as overall survival, assess sensitivity to chemotherapeutic agents, and guide targeted therapeutic interventions aimed at modulating BAD phosphorylation to enhance treatment efficacy, particularly in aggressive malignancies such as triple-negative breast cancer.

## BACKGROUND OF THE INVENTION

Genome-wide approaches to cancer research have revolutionized our understanding of tumor biology by enabling comprehensive profiling of molecular alterations that drive carcinogenesis, progression, and therapeutic resistance. These high-throughput methodologies, including transcriptomic, genomic, and proteomic analyses, have facilitated the identification of disease-specific molecular signatures that can serve as biomarkers for diagnosis, prognosis, and prediction of treatment response. Despite these advances, many aggressive cancers—particularly those lacking established therapeutic targets—remain difficult to manage due to limited options for personalized intervention. Triple-negative breast cancer (TNBC), which lacks expression of estrogen receptor (ER), progesterone receptor (PR), and HER2, exemplifies this unmet clinical need, as it is associated with poor prognosis, high recurrence rates, and reliance on non-specific cytotoxic chemotherapy.

Central to the regulation of programmed cell death is the BCL-2 family of proteins, which includes both pro-apoptotic and anti-apoptotic members whose interactions determine cellular fate. Among these, BAD (BCL2 antagonist of cell death) functions as a critical rheostat of apoptosis by binding to and neutralizing anti-apoptotic proteins such as BCL-2 and BCL-xL, thereby freeing pro-apoptotic effectors like BAX to induce mitochondrial outer membrane permeabilization and caspase activation. The activity of BAD is tightly regulated by post-translational modifications, particularly phosphorylation at key serine residues (e.g., Ser-112, Ser-136, Ser-155 in mouse; corresponding to Ser-75, Ser-99, Ser-118 in human), which promote its sequestration by 14-3-3 scaffold proteins and prevent interaction with BCL-2/BCL-xL. Phosphorylation is mediated by multiple kinases—including AKT (via the PI3K/AKT pathway), PKA, RSK, and CDK1—while dephosphorylation is catalyzed by phosphatases such as PP2C (PPM1A), PP2A, and calcineurin. Dysregulation of this balance toward hyperphosphorylation of BAD has been implicated in chemoresistance across various cancers.

Given the critical role of the BAD pathway in determining cellular susceptibility to apoptosis, there exists a pressing need for robust, clinically applicable tools that can assess the functional state of this pathway in patient tumors. Such tools would enable improved cancer diagnosis, risk stratification, prediction of therapeutic response, and rational selection of combination therapies designed to restore apoptotic sensitivity. The present invention addresses this need by providing a quantitative gene expression signature that reflects the integrated activity of the BAD pathway and correlates with clinically relevant endpoints in cancer.

## SUMMARY OF INVENTION

The present invention is motivated by the recognition that comprehensive characterization of the BAD apoptosis pathway offers a powerful framework for understanding cancer biology and guiding therapeutic decisions. To this end, the inventors developed a BAD Pathway Gene Expression Signature (BPGES) using principal component analysis (PCA) applied to a curated set of 16 genes directly involved in regulating BAD phosphorylation and apoptotic function. This signature includes BAX, BCL2, EGFR, PDK1, PIK3CA, PIK3CB, PPP1CA, PPP2CA, PPP3CA, PPM1A, YWHAB, YWHAE, YWHAG, YWHAH, YWHAQ, and YWHAZ—genes encoding components of upstream signaling cascades (e.g., PI3K/AKT, PKA), phosphatases, kinases, and 14-3-3 scaffolding proteins that collectively govern BAD activity.

Analysis of clinico-genomic datasets revealed that the BPGES score strongly differentiates triple-negative breast cancer (TNBC) from hormone receptor–positive (non-TNBC) tumors, with TNBC exhibiting significantly higher BPGES scores indicative of a pro-survival, anti-apoptotic state. Moreover, elevated BPGES was associated with reduced overall survival in two independent cohorts—the Moffitt Cancer Center Total Cancer Care (TCC) dataset and The Cancer Genome Atlas (TCGA)—demonstrating its prognostic utility. Importantly, the BPGES also correlated with in vitro sensitivity to a broad range of cytotoxic drugs in the NCI60 cancer cell line panel, suggesting its predictive value for chemotherapy response.

The invention further introduces a method for evaluating BAD protein phosphorylation status, particularly at Ser-136 (an AKT site), which was found to be enriched in TNBC tissues and positively correlated with BPGES. Building on these findings, the inventors demonstrate that pharmacological inhibition of BAD-phosphorylating kinases—specifically AKT with perifosine or PKA with H89—selectively sensitizes TNBC cells, but not ER+/PR+ cells, to cisplatin-induced cytotoxicity, and that this effect is dependent on the presence of BAD protein.

Accordingly, the invention provides: (1) a method for diagnosing cancer by determining a subject’s BPGES score from a biological sample and comparing it to a reference threshold; (2) a method for determining patient survival probability based on BPGES; (3) a method for identifying cancer sensitivity to chemotherapeutic agents using BPGES or pBAD levels; (4) a method for treating cancer by administering a kinase inhibitor (e.g., AKT or PKA inhibitor) in combination with chemotherapy to patients with high BPGES or elevated pBAD; and (5) a method for monitoring neoplasia progression through serial assessment of BPGES. The BAD pathway gene selection is grounded in established biochemical knowledge of BAD regulation, ensuring biological relevance and clinical interpretability of the signature.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### Definitions

For the purposes of this invention, numerical designations refer to specific amino acid residues in the mouse BAD protein (e.g., Ser-112, Ser-136, Ser-155), which correspond to human Ser-75, Ser-99, and Ser-118, respectively; however, for consistency and cross-reactivity of detection reagents, the mouse nomenclature is retained throughout. The terms “about” and “approximately” denote values within ±10% of the stated number, unless otherwise specified by context. Concentration and amount ranges are inclusive of endpoints and encompass any intermediate value. An “agent” refers to any compound, molecule, or biologic capable of modulating gene or protein expression or activity, including small molecules, antibodies, siRNAs, and peptides. A “subject” is a human or non-human mammal diagnosed with or suspected of having cancer.

Gene biomarkers herein are nucleic acid sequences whose expression levels correlate with a biological state of interest. Diagnosis of disease involves identifying the presence or type of cancer based on molecular profiling. Prognosis refers to predicting the likely course or outcome of disease, including survival. Susceptibility to treatment denotes the likelihood of therapeutic response. Evaluation of treatment efficacy involves assessing changes in biomarker levels post-intervention. “Expression level” means the quantity of RNA transcript or protein product derived from a gene. Detecting biomarker expression may involve hybridization, amplification, sequencing, or immunoassay techniques. Measuring gene expression includes microarray, RNA-seq, RT-qPCR, and Nanostring platforms. Quantifying transcription levels entails normalization to housekeeping genes or total RNA. Protein expression is measured via Western blot, ELISA, immunohistochemistry (IHC), or mass spectrometry.

“Diagnosing” or “diagnosis” means determining the nature or cause of a disease condition. “Prognosis” is the prediction of disease trajectory. “Risk” or “susceptibility” indicates increased likelihood of adverse outcome or treatment resistance. “Treatment” or “treating” encompasses administration of therapeutic agents to ameliorate disease. A “biomarker” is a measurable indicator of a biological state. Biomarker measurement levels may be categorical (high/low) or continuous. A “biological state” includes health, disease, or response to therapy. Biological state measurement involves comparing biomarker levels to baseline or reference values. A “cell” or “cells” refers to any biological unit, including tumor cells in tissue or culture. A “BAD pathway gene” is any gene encoding a protein that directly or indirectly regulates BAD phosphorylation or function.

The BAD pathway gene expression signature score (BPGES) is a composite metric derived from PCA of selected BAD pathway genes. Calculation involves mean-centering and variance-scaling of expression data, followed by projection onto the first principal component (PC1): BPGES = ∑(w_i × x_i), where w_i is the loading coefficient for gene i and x_i is its normalized expression. The BAD Pathway Gene Expression Signature comprises the 16 aforementioned genes. A “sample” is any biological material (e.g., tumor tissue, blood, cell line) suitable for molecular analysis. A “therapeutically effective amount” is a dose sufficient to produce a desired clinical effect. “Baseline level” refers to pre-treatment biomarker measurement. “Neoplasia” denotes abnormal tissue growth, including benign, pre-cancerous, pre-invasive, and invasive forms. “Pre-cancerous” indicates histologic changes with malignant potential (e.g., ductal carcinoma in situ). “Pre-invasive” describes lesions confined to epithelium without stromal invasion. “Invasive” signifies penetration into surrounding tissue. A “gene expression product” is RNA or protein. A “polynucleotide” is a nucleic acid polymer. “Nucleic acid” includes DNA and RNA. A “polypeptide” is an amino acid chain. An “isolated polynucleotide” is separated from its natural environment. A “probe set” is a collection of oligonucleotides targeting a specific transcript.

The BAD apoptosis pathway is motivated by its central role in integrating survival signals to control cell death. BAD pathway expression reflects the net activity of kinases and phosphatases governing BAD phosphorylation. Elevated BPGES correlates with cancer development and progression, particularly in TNBC. BAD protein functions as a pro-apoptotic sensitizer by displacing BAX from BCL-2/BCL-xL. Phosphorylation inactivates BAD by promoting 14-3-3 binding. BAD-BCL-2 interaction is disrupted upon BAD dephosphorylation. Phosphorylation is regulated by growth factor signaling (e.g., PI3K/AKT, PKA). The PI3K/AKT pathway, frequently hyperactivated in cancer, phosphorylates BAD at Ser-136. AKT-mediated phosphorylation inhibits BAD’s pro-apoptotic function. The PKA pathway, activated by cAMP, phosphorylates BAD at Ser-112 and Ser-155. PP2C phosphatase dephosphorylates BAD, promoting apoptosis. PP2C exhibits tumor suppressor activity. CDK1 phosphorylates BAD during mitosis, linking cell cycle to apoptosis regulation. CDK1 drives G2/M transition. CDK1 also phosphorylates BCL-2, modulating its anti-apoptotic activity. The BAD pathway contributes to cancer development by suppressing apoptosis. It promotes cancer progression by enhancing survival under stress. It mediates resistance to therapy by blunting drug-induced apoptosis. It influences overall survival, as shown in clinical datasets. BAD phosphorylation status is a determinant of treatment response. The pathway aids in cancer diagnosis by distinguishing molecular subtypes. In summary, the BAD pathway is a master regulator of apoptotic competence with profound implications for oncology.

### Methods

Cell lines were obtained from the American Type Culture Collection and cultured in RPMI or DMEM supplemented with 10% fetal bovine serum, sodium pyruvate, nonessential amino acids, and mycoplasma inhibitors. Cultures were maintained at 37°C in 5% CO₂ and tested for mycoplasma every six months.

### BAD Pathway Expression

The BAD Pathway Gene Expression Signature score (BPGES) is defined as the first principal component derived from PCA of 16 BAD pathway genes. The PCA method involves selecting probe sets for these genes, mean-centering and scaling expression values, and computing PC1 as a weighted sum of gene expressions. BPGES calculation yields a single numeric score per sample representing overall pathway activity. PCA model interpretation reveals that high BPGES reflects coordinated upregulation of kinases (e.g., PI3K, EGFR) and 14-3-3 proteins alongside downregulation of phosphatases (e.g., PPM1A), consistent with BAD inactivation. Evaluation across cancer datasets—including TCC, TCGA, and CCLE—confirms BPGES elevation in TNBC versus non-TNBC. Analysis of normal versus pathologic tissues shows minimal BPGES in healthy breast, underscoring its cancer specificity. BPGES varies across cancer types, with highest levels in basal-like carcinomas. External validation in clinico-genomic datasets demonstrates reproducibility. Association with patient survival is assessed via Kaplan-Meier curves stratified by median BPGES, revealing significantly worse overall survival in high-BPGES groups. Log-rank tests confirm statistical significance in both TCC and TCGA cohorts.

### Analysis of Genomic and Chemo-Sensitivity Data for NCI60 Cancer Cell Lines

Genomic and chemo-sensitivity data from the NCI60 panel were analyzed to correlate BPGES with in vitro drug response. High BPGES was associated with resistance to multiple cytotoxic agents, supporting its role as a predictor of chemosensitivity.

### Modulating BAD Phosphorylation by siRNA Transfection

siRNA transfection was performed using Nucleofector technology with cell-type–specific buffers. Pooled siRNAs targeting BAD or non-targeting controls were introduced into 4×10⁶ cells. Transfection efficiency was evaluated by Western blot showing >70% knockdown of BAD protein relative to controls.

### MTS Cell Proliferation Assays

MTS assays measured cell viability after drug exposure. Cells were plated in 96-well plates, treated with cisplatin ± kinase inhibitors for 72 hours, then incubated with MTS reagent. Absorbance at 490 nm was read, and percent survival was calculated relative to untreated controls.

### Western Blot Analysis

Western blotting assessed protein expression and phosphorylation. Cells were lysed in SDS buffer, proteins separated by PAGE, transferred to PVDF membranes, blocked, and probed with primary antibodies overnight. After secondary antibody incubation, chemiluminescent detection was performed. Densitometry quantified band intensity normalized to GAPDH or histone H3.

### Immunofluorescence Microscopy

Immunofluorescence microscopy visualized subcellular protein localization. Cells were fixed, permeabilized, blocked, and incubated with primary antibodies, followed by fluorophore-conjugated secondaries. Fluorescence intensity was quantified using image analysis software.

### Antibodies

Antibodies used included: phospho-AKT-Ser473 (#4060, Cell Signaling), AKT (#4691, Cell Signaling), phospho-PKA (#5661, Cell Signaling), PKA (#4782, Cell Signaling), GAPDH (#MAB374, Millipore), phospho-BAD-Ser112 (A00295, Genscript), phospho-BAD-Ser136 (A01156, Genscript), and phospho-BAD-Ser155 (AB28825, Abcam).

## Results

### BAD Pathway Expression is Associated with Cancer Development, Progression, Relapse, and Survival

Statistically significant differences in BPGES were observed between TNBC and non-TNBC in TCC (P=4.8×10⁻¹⁹) and TCGA (P=1.89×10⁻²⁶) datasets. BPGES was markedly elevated in invasive cancer compared to normal tissue. High BPGES correlated with shorter overall survival in combined breast cancer cohorts but not within TNBC alone, suggesting its greatest prognostic power in heterogeneous populations. Analysis across multiple cancer types confirmed BPGES enrichment in aggressive subtypes.

### BAD Pathway Expression is Associated with In-Vitro Sensitivity to a Broad Range of Cytotoxic Drugs

In NCI60 cell lines, BPGES inversely correlated with sensitivity to DNA-damaging agents, microtubule inhibitors, and topoisomerase inhibitors, indicating that BAD pathway activation confers broad chemoresistance.

### BAD Pathway Expression is Associated with In Vitro Levels of Phosphorylated BAD Protein

A significant positive correlation was found between BPGES and phospho-BAD-Ser136 levels (r²=0.1569, P=0.003) in tumor tissues. Chemo-sensitive cell pairs exhibited lower pBAD than resistant counterparts, linking pathway activity to functional protein modification.

### Modulation of BAD Phosphorylation Affects Proliferation of Cancer Cell Lines

siRNA-mediated depletion of CDK1 or PP2C altered BAD phosphorylation and reduced proliferation in TNBC lines, demonstrating that targeted modulation of BAD regulators impacts cancer cell growth.

### BAD Phosphorylation Status and Cancer Development

Immortalized breast epithelial cells showed low pBAD and kinase expression, whereas cancer lines exhibited elevated pBAD-Ser136 and active AKT/PKA, implicating BAD phosphorylation as a hallmark of malignant transformation.

## Conclusion

The results demonstrate that the BAD pathway gene expression signature (BPGES) is a robust biomarker for cancer diagnosis, prognosis, and prediction of therapeutic response. Its association with triple-negative status, survival outcomes, and chemosensitivity underscores its clinical relevance. Furthermore, the dependence of kinase inhibitor–mediated chemosensitization on BAD protein validates the pathway as a therapeutically actionable target. These findings support the development of BPGES-based diagnostic assays and combination therapies targeting BAD phosphorylation to improve outcomes in aggressive cancers, particularly TNBC.