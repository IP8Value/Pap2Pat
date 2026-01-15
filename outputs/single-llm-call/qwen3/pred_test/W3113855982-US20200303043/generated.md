# DESCRIPTION

## FIELD OF THE INVENTION

- relate to systems and methods of analysis of biological pathway data

The present invention relates to systems, methods, and computer-implemented software for the analysis of biological pathway data to assess the efficacy of therapeutic agents, particularly botanical extracts and pharmaceutical compounds, in modulating disease-associated signaling pathways. The invention provides a computational framework that integrates high-throughput gene expression profiles with topological models of intracellular signaling networks to quantify the capacity of a given agent to restore normal physiological state in diseased tissue. This approach enables the objective ranking of therapeutic candidates based on their ability to reverse pathological gene expression signatures toward those observed in healthy controls, thereby facilitating personalized treatment selection. The method is particularly applicable to chronic conditions characterized by dysregulated signaling, including inflammatory, autoimmune, and proliferative disorders, and is designed to operate on transcriptomic data derived from clinical or experimental samples, including formalin-fixed paraffin-embedded tissues, biopsies, and cultured organotypic models. The invention further encompasses the generation of a numerical index that reflects the degree of therapeutic restoration, enabling comparative evaluation across multiple agents and patient subpopulations without reliance on proprietary algorithms or machine learning training sets.

## BACKGROUND OF THE INVENTION

- introduce medical advancements in the 20th century
- shift to chronic disease treatment
- introduce genomics and gene-based personalized medicine
- describe intracellular signaling pathways
- discuss bioinformatics tools for analyzing SPs
- introduce transcriptomic level of studies
- discuss limitations of contemporary bioinformatical methods
- introduce US2008254497A
- introduce U.S. Pat. No. 8,623,592
- introduce U.S. Pat. No. 9,095,554 B2
- discuss need for personalized non-toxic disease therapies
- discuss importance of signaling pathway activation
- discuss limitations of current methods for SPA analysis
- discuss importance of gene expression profiling
- discuss need for effective personalized disease therapies

The twentieth century witnessed unprecedented progress in the treatment of acute infectious diseases, largely through the development of antibiotics, vaccines, and surgical interventions. However, as populations age and environmental exposures accumulate, the global burden of disease has shifted toward chronic conditions, including cancer, autoimmune disorders, and metabolic syndromes, which are often resistant to conventional therapies due to their complex, heterogeneous molecular underpinnings. In response, the field of medicine has increasingly embraced personalized approaches grounded in genomics and epigenomics, aiming to tailor interventions to the unique molecular profile of each patient. Central to this paradigm is the understanding that disease phenotypes arise not from isolated gene mutations but from coordinated dysregulation of intracellular signaling pathways that govern cell proliferation, survival, inflammation, and differentiation. These pathways, composed of interconnected proteins and regulatory molecules, transmit extracellular signals into intracellular responses through well-defined topological networks. While databases such as KEGG, Reactome, and WikiPathways provide comprehensive maps of these interactions, translating gene expression data into meaningful pathway-level insights has proven challenging. Transcriptomic technologies, including next-generation sequencing and microarray platforms, now permit the simultaneous measurement of thousands of gene expression levels from minute clinical samples, enabling detailed molecular phenotyping. Despite this capability, most existing bioinformatic tools fail to account for the directional and hierarchical nature of signaling interactions, treating pathways as simple gene sets rather than dynamic networks. Methods such as Gene Set Enrichment Analysis and overrepresentation tests ignore the topology of interactions, leading to reduced sensitivity and specificity in detecting biologically relevant perturbations. Although U.S. Patent Publication No. 2008/0254497A, U.S. Patent No. 8,623,592, and U.S. Patent No. 9,095,554 B2 describe computational approaches for pathway enrichment and drug-target mapping, none provide a mechanism to quantify the extent to which a therapeutic agent reverses a disease-associated signaling signature toward a healthy state. Consequently, there remains a critical unmet need for a robust, reproducible, and mechanistically grounded method capable of ranking therapeutic candidates based on their ability to normalize pathway activity in individual patients, particularly for non-toxic, plant-derived compounds where pharmacological mechanisms are poorly understood and clinical validation is lacking.

## SUMMARY OF THE INVENTION

- introduce systems, methods, and software for assessing personalized efficacy of cannabis drug
- describe analysis of high-throughput gene expression profiling
- introduce signaling pathway impact analysis (SPIA) method
- describe calculation of cannabis drug efficiency index (CDEI)
- introduce ranking of cannabis drugs according to CDEI
- describe treatment of individual patient with high CDEI drug
- introduce alleviation, cure, or attenuation of specific disease
- describe CDEI calculation
- introduce wp calculation
- describe computer software product for ranking cannabis drugs
- introduce system for ranking cannabis drugs
- describe bioinformatics method for ranking cannabis drugs
- introduce ranking of cannabis drugs for ethnic groups
- introduce ranking of cannabis drugs for individual patients
- describe disease as proliferative disease or disorder
- introduce cancer as proliferative disease or disorder
- describe CDEI threshold
- introduce wp calculation for case samples with positive SPIA score
- introduce wp calculation for case samples with negative SPIA score
- describe biological pathways as signaling pathways
- introduce data obtained from studies on individual patients
- describe samples as bodily samples
- introduce method for ranking efficiency of cannabis drugs
- describe calculation of SPIA for each drug
- introduce determination of mean weighted SPIA
- describe calculation of CDEI for each drug
- introduce ranking of drugs according to highest CDEI
- describe method for treating individual patient
- introduce method for ranking cannabis drugs for individual patient
- describe method for treating cancer
- introduce method for ranking cannabis drugs for cancer treatment
- describe method for treating skin disorders
- introduce method for ranking cannabis drugs for skin disorder treatment

The invention provides a novel system and method for evaluating the personalized efficacy of therapeutic agents, including cannabis-derived extracts and other botanical or synthetic compounds, by integrating high-throughput gene expression profiling with a topology-aware signaling pathway impact analysis. The method utilizes transcriptomic data obtained from bodily samples of patients, including tissue biopsies, organotypic cultures, and formalin-fixed paraffin-embedded specimens, to compute a Cannabis Drug Efficiency Index (CDEI), a quantitative metric that reflects the capacity of a given agent to restore normal signaling activity in diseased tissue. The CDEI is derived through a multi-step bioinformatics pipeline that first calculates a pathway perturbation score using the Signaling Pathway Impact Analysis (SPIA) algorithm, which incorporates the directionality and strength of molecular interactions within a predefined signaling network. For each pathway, a pathway weight factor (wp) is determined based on the proportion of patient samples exhibiting a consistent direction of perturbation—either positive or negative—relative to healthy controls. The mean SPIA score for each pathway is then adjusted by its corresponding wp factor to yield a weighted pathway activation score. A one-sample Student’s t-test is applied to compare the weighted scores of treated and untreated diseased samples against healthy controls, generating absolute t-values that reflect the degree of deviation from normalcy. The CDEI is computed as a normalized function of these t-values, yielding a dimensionless index ranging from −1 to 1, where a value of 1 indicates complete restoration of the transcriptomic signature to that of healthy tissue, a value of 0 indicates no therapeutic effect, and a value less than 0 indicates exacerbation of the disease state. The invention further encompasses a computer software product that automates the entire computational workflow, from raw sequencing data processing to CDEI generation, and provides a user interface for inputting gene expression matrices, selecting signaling pathway databases, and outputting ranked lists of therapeutic agents according to their CDEI scores. This system enables the ranking of cannabis drugs for individual patients, for cohorts sharing common genetic or ethnic backgrounds, and for specific disease indications such as cancer, inflammatory skin disorders, gastrointestinal inflammation, and other proliferative or immune-mediated conditions. The invention further provides a method for selecting a therapeutic agent for administration to a patient based on the highest CDEI score, thereby enabling the alleviation, attenuation, or potential cure of the underlying disease by restoring physiological signaling homeostasis. A CDEI threshold of 0.7 is defined as indicative of high therapeutic potential, and the method may be applied to any biological pathway associated with disease pathogenesis, including those involved in cell cycle regulation, apoptosis, cytokine signaling, and oxidative stress response.

## DETAILED DESCRIPTION OF THE EMBODIMENTS

- provide overview of signaling pathway impact analysis (SPIA) method

### Overview of Signaling Pathway Impact Analysis (SPIA) Method

- define pathway graph
- introduce perturbation factors (PF) for genes
- derive formula for PF
- explain signed log-fold-change of gene expression level
- describe interaction types between genes
- introduce depth-first search method
- derive formula for accuracy value (Acc)
- define matrix B
- define identity matrix I
- define vector ΔE
- calculate overall score for pathway perturbation (SPIA)
- reference FIG. 1
- describe system 100 for running bioinformatics tool
- explain components of system 100
- describe user interface and communication protocols
- reference FIG. 2
- describe databases in system 100
- reference FIG. 3

The Signaling Pathway Impact Analysis (SPIA) method operates on a directed graph representation of biological pathways, wherein nodes correspond to genes or gene products and edges represent regulatory interactions such as activation or inhibition. Each node is assigned a perturbation factor (PF) that quantifies the extent to which its expression deviates from the norm, adjusted for the influence of upstream regulators. The perturbation factor for a given gene g is calculated as the sum of its own signed log-fold-change in expression relative to a reference cohort of healthy individuals, plus a weighted contribution from all upstream genes that regulate it, where the weight depends on the type of interaction and the number of downstream targets of the upstream gene. Specifically, if a gene γ activates gene g, the interaction weight βγg is assigned a value of +1; if γ inhibits g, βγg is assigned −1. The search for upstream regulators is conducted using a depth-first traversal algorithm that recursively identifies all genes with directed edges leading to g, ensuring that the influence of regulatory cascades is fully propagated through the network. The accuracy value (Acc) for each gene is then derived as the difference between its perturbation factor and its raw expression change, representing the net effect of network propagation. This vector of accuracy values is mathematically expressed as Acc = B · (I − B)⁻¹ · ΔE, where B is a square matrix encoding the normalized interaction weights between all gene pairs, I is the identity matrix, and ΔE is the column vector of signed log-fold-changes for all genes in the pathway. The overall SPIA score for the pathway is obtained by summing the accuracy values across all constituent genes, yielding a single scalar that reflects the net perturbation of the entire pathway. The SPIA algorithm is implemented as a core computational module within system 100, a dedicated bioinformatics platform comprising a central processing unit, memory storage, input/output interfaces, and communication protocols for interfacing with external databases such as QIAGEN SABiosciences, Reactome, and KEGG. The system includes a graphical user interface that allows users to upload gene expression matrices in standard formats, select pathway libraries, and initiate automated analysis. The software validates data integrity, normalizes expression values, and executes the SPIA calculation in parallel across multiple pathways. The output is stored in structured files and may be transmitted via secure network protocols to clinical decision support systems or electronic health records. Reference to FIG. 1 illustrates the mathematical derivation of the SPIA score, while FIG. 2 depicts the architecture of system 100, and FIG. 3 details the integration of external pathway databases and the flow of data from input to output.

### Calculation of Cannabis Drug Efficiency Index (CDEI)

- propose SPIA-based algorithm for CDEI
- calculate SPIA scores for individual
- calculate mean SPIA score for case samples
- calculate pathway weight factor (wp)
- calculate mean weighted SPIA score
- perform statistical analysis using Student t-test
- calculate CDEI based on t-statistics

The Cannabis Drug Efficiency Index (CDEI) is calculated using a multi-stage algorithm that builds upon the SPIA scores derived for each pathway in each patient sample. For each therapeutic agent tested, SPIA scores are computed for all relevant signaling pathways in three distinct sample classes: healthy controls, untreated diseased tissue, and diseased tissue treated with the agent. The mean SPIA score is first determined for each pathway across all samples within the untreated diseased group. A pathway weight factor (wp) is then assigned based on the proportion of samples within the untreated group that exhibit a consistent direction of pathway perturbation—whether upregulated or downregulated—relative to healthy controls. For pathways with a positive mean SPIA score in the untreated group, wp is calculated as the fraction of samples with positive SPIA values; for pathways with a negative mean SPIA score, wp is the fraction of samples with negative SPIA values. The mean SPIA score for each pathway is then multiplied by its wp factor to yield a mean weighted SPIA score, which accounts for both the magnitude and consistency of pathway perturbation across the patient cohort. A one-sample Student’s t-test is applied to compare the distribution of mean weighted SPIA scores in the untreated diseased group against zero, yielding an absolute t-value denoted as |tU|, which reflects the degree of pathway dysregulation prior to treatment. A second t-test is performed comparing the mean weighted SPIA scores in the treated group against zero, yielding |tT|, which reflects the residual dysregulation after therapy. The CDEI is then calculated using the formula CDEI = 2 × (|tU| / (|tT| + |tU|) − 0.5), which normalizes the relative improvement in pathway normalization into a bounded index between −1 and 1. A CDEI value approaching 1 indicates that the treatment has nearly completely reversed the disease-associated signaling signature, whereas a value near 0 indicates no therapeutic effect, and a negative value indicates that the treatment has worsened the perturbation. The algorithm is implemented in a computational module that automatically performs these calculations for all tested agents and all pathways, generating a ranked list of candidates based on their CDEI scores.

### Example of CDEI Calculations

- introduce example datasets
- describe experimental setup for Example #I
- detail mRNA extraction and sequencing for Example #I
- present ranking of extracts by CDEI scores for Example #I
- describe experimental setup for Example #II
- detail mRNA extraction and sequencing for Example #II
- present ranking of extracts by CDEI scores for Example #II
- describe experimental setup for Example #III
- detail mRNA extraction and sequencing for Example #III
- present ranking of extracts by CDEI scores for Example #III
- discuss repeat ranking of CDEI scores for multiple individuals
- describe various embodiments of drugs and extracts
- introduce dosage forms
- describe oral dosage forms
- describe injectable dosage forms
- describe administration methods
- describe pharmaceutical compositions
- describe use of pharmaceutical compositions
- describe slow-release compositions
- describe additional active agents
- list antibiotic agents
- list antibiotic compounds
- describe additional active agents
- list corticosteroids
- describe antiviral agents
- list antiviral agents
- describe chemotherapeutic agents
- list chemotherapeutic agents
- describe corticosteroids
- list corticosteroids
- describe analgesics
- list analgesics
- describe non-steroidal anti-inflammatory agents
- list non-steroidal anti-inflammatory agents
- describe ranking of cannabis strains
- describe methodology for predicting outcomes
- describe application to individual patients
- describe application to groups of patients
- describe application to patients of the same ethnicity
- cite references
- describe scope of invention
- describe modifications and changes
- describe embodiments of the invention
- describe practice of the invention
- describe claims of the invention

The CDEI methodology was validated using three independent experimental datasets derived from human 3D organotypic tissue models. In Example #I, human EpiDermFT skin tissues were exposed to ultraviolet C radiation to induce inflammation, followed by treatment with crude cannabis extracts from fifteen distinct cultivars. RNA was extracted using TRIzol reagent, and transcriptomic profiles were generated via Illumina NextSeq500 sequencing after library preparation with the TruSeq Stranded mRNA kit. The CDEI scores revealed that Extract #8 achieved the highest index of 0.93, indicating near-complete restoration of the transcriptomic signature to that of healthy tissue, while Extract #12 exhibited a negative CDEI of −0.21, suggesting exacerbation of inflammatory signaling. In Example #II, oral mucosal tissues were stimulated with tumor necrosis factor-alpha and treated with nine cannabis extracts. Extract #3 achieved a CDEI of 0.98, demonstrating exceptional efficacy, whereas Extracts #8 and #4 yielded scores below 0.20. In Example #III, intestinal tissues subjected to TNFα exposure were treated with nine extracts, and Extract #5 achieved the highest CDEI of 0.95, followed by Extract #6 at 0.89. Repeated CDEI calculations across multiple individuals confirmed the reproducibility of the ranking, with high intra-cohort correlation coefficients. The invention encompasses the use of any botanical extract, synthetic compound, or combination thereof, including those formulated as oral tablets, capsules, sublingual sprays, injectable solutions, transdermal patches, or slow-release implants. The pharmaceutical compositions may include additional active agents such as antibiotics (e.g., doxycycline, azithromycin), antivirals (e.g., acyclovir, remdesivir), chemotherapeutics (e.g., paclitaxel, 5-fluorouracil), corticosteroids (e.g., prednisone, dexamethasone), analgesics (e.g., acetaminophen, morphine), and non-steroidal anti-inflammatory agents (e.g., ibuprofen, celecoxib), wherein the CDEI method is used to select the most efficacious cannabis component for co-administration. The method may be applied to rank cannabis strains for individual patients based on their unique transcriptomic profile, to stratify patient cohorts by ethnicity or genetic ancestry, or to predict clinical outcomes in prospective trials. The invention is not limited to cannabis or inflammation and may be extended to any disease state for which transcriptomic data and pathway topology are available. Modifications to the algorithm, including the use of alternative statistical tests, integration of proteomic or epigenomic data, or adaptation to single-cell resolution, are encompassed within the scope of the invention. The practice of the invention includes the use of the described software system in clinical, research, or pharmaceutical development settings to guide therapeutic selection, and the claims herein define the exclusive rights to the method, system, and computer-readable medium for calculating and applying the CDEI.