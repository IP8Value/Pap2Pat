# DESCRIPTION

## FIELD OF THE INVENTION

- relate to cancer and methods for classifying patients

The present invention relates to methods for classifying patients with solid tumors, particularly breast cancer, by analyzing gene expression patterns within the tumor-associated stroma. These methods enable the stratification of patients into distinct prognostic categories based on the transcriptional profile of the stromal microenvironment, independent of the epithelial tumor cell composition. The invention provides a framework for identifying patients at high risk of disease recurrence or metastasis through the assessment of a multivariate stromal-derived prognostic predictor, which integrates the expression levels of a defined set of genes expressed in the non-epithelial compartment of the tumor tissue. This classification system is applicable to clinical decision-making, including the selection of therapeutic interventions, the design of clinical trials, and the monitoring of treatment response. The invention further extends to the use of this stromal signature in the context of other cancer types where the tumor microenvironment plays a critical role in disease progression, thereby offering a broadly applicable diagnostic and prognostic tool grounded in the biological behavior of the tumor stroma rather than the malignant epithelium alone.

## BACKGROUND OF THE INVENTION

- introduce breast cancer
- describe current challenges in breast cancer treatment
- discuss recent advances in genomic characterization of tumors
- highlight need for new method to predict outcome recurrence

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide, despite significant advances in early detection and therapeutic intervention. While many patients present with localized disease and respond well to initial treatment, a substantial proportion—approximately one in four—will eventually develop distant metastases, often years after apparent clinical remission. Current clinical prognostic tools, including tumor size, lymph node status, histological grade, and hormone receptor expression, provide only partial predictive power and fail to reliably distinguish those patients who will experience recurrence from those who will remain disease-free. Although genomic signatures derived from tumor epithelial cells have been developed to refine risk stratification, their clinical utility is limited by heterogeneity in tumor composition, technical variability in sample processing, and an incomplete understanding of the contribution of the non-malignant microenvironment to disease behavior. Recent advances in high-throughput genomic technologies have enabled comprehensive profiling of tumor tissues, yet most studies have relied on bulk tissue analysis, which obscures the distinct transcriptional contributions of epithelial and stromal compartments. This limitation has hindered the identification of stroma-specific molecular signatures that may reflect underlying biological processes critical to tumor progression, such as immune modulation, angiogenesis, and hypoxia. Consequently, there exists a critical unmet need for a robust, reproducible, and biologically grounded method to predict clinical outcome that is derived not from the tumor cells themselves, but from the surrounding stromal tissue, which is consistently present across patients and less subject to the genetic instability that characterizes malignant epithelium.

## SUMMARY OF THE INVENTION

- introduce laser capture microdissection (LCM) to isolate tumor-associated and matched normal stroma
- perform microarray analyses to identify gene expression signatures
- develop multivariate stromal derived prognostic predictor (SDPP)
- identify set of genes in tumor stroma predictive of outcome
- describe genes including pro-angiogenic and hypoxia-related factors
- describe genes including T-cell markers
- provide method for identifying gene expression signature
- provide method for predicting clinical outcome using SDPP
- describe method for predicting clinical outcome in breast cancer patient
- describe method for determining prognosis
- describe method for predicting disease outcome
- describe method for diagnosing poor prognosis breast cancer
- describe method for predicting probability of cancer recurrence
- describe method for predicting probability of cancer metastasis
- describe method for diagnosing tumor subtype
- describe method for assigning treatment or therapy
- describe method for optimizing treatment
- describe method for monitoring treatment
- describe method for assigning subject to clinical study
- integrate SDPP predictor with other predictors and signatures
- combine SDPP with other known predictors and signatures
- describe SDPP gene sets
- describe composition comprising nucleic acid sequences
- describe composition comprising binding agents
- describe method of identifying agents for use in treatment of cancer
- describe kits comprising nucleic acids and polypeptides
- describe arrays for detecting SDPP gene set expression levels
- describe computer systems and computer program products

The invention introduces a novel method for predicting clinical outcome in patients with breast cancer through the analysis of gene expression patterns in the tumor-associated stroma, achieved by the precise isolation of stromal cells using laser capture microdissection followed by genome-wide transcriptional profiling. Through this approach, a multivariate stromal-derived prognostic predictor (SDPP) was developed, comprising a defined set of genes whose coordinated expression in the stromal compartment is strongly associated with disease recurrence, metastasis, and overall survival. The SDPP gene set includes genes involved in pro-angiogenic signaling, hypoxia response, and immune modulation, particularly those associated with the suppression of T-cell activity and the recruitment of immunosuppressive macrophages. The invention provides a method for identifying this gene expression signature by comparing the transcriptome of stromal tissue from patients with favorable versus unfavorable clinical outcomes, enabling the construction of a predictive classifier based on the weighted expression levels of these genes. This classifier can be applied to determine prognosis, diagnose poor prognosis breast cancer, predict the probability of recurrence or metastasis, and classify tumor subtypes based on stromal biology rather than epithelial markers. The invention further encompasses methods for assigning treatment regimens tailored to the stromal profile, optimizing therapy by targeting pathways such as angiogenesis or Th2 immune polarization, monitoring treatment efficacy through serial assessment of SDPP gene expression, and selecting patients for enrollment in clinical trials based on their stromal signature. The SDPP can be integrated with existing clinical and molecular predictors to enhance predictive accuracy, and the invention includes compositions comprising isolated nucleic acid sequences encoding the SDPP genes, binding agents such as antibodies or antibody fragments specific to the polypeptide products of these genes, and kits containing probes, primers, and solid-phase supports for the detection of SDPP gene expression or protein levels. Microarrays and other high-throughput platforms are provided for the simultaneous quantification of SDPP gene expression levels, and computer systems and program products are described for the automated calculation of SDPP scores from digital gene expression data, enabling clinical implementation in diagnostic laboratories.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce breast cancer outcome predictor
- motivate stroma-derived predictor
- define stroma-derived prognostic predictor (SDPP)
- describe SDPP gene sets
- explain SDPP gene set selection
- provide examples of SDPP gene sets
- describe accuracy of SDPP gene sets
- define clinical outcome
- explain expression level of SDPP genes
- define reference expression profile
- describe sample types
- introduce class discovery
- describe class distinction
- explain class prediction
- motivate accurate prediction
- describe gene weighting
- provide method for predicting clinical outcome
- provide alternative methods for predicting clinical outcome
- describe method for identifying SDPP gene set
- provide alternative method for identifying SDPP gene set
- summarize SDPP gene set identification

The invention is directed to a breast cancer outcome predictor grounded in the transcriptional signature of the tumor stroma, a biological compartment that is consistently present across patients and less subject to the clonal heterogeneity and genomic instability that characterize malignant epithelial cells. Unlike prior approaches that focus on epithelial gene expression, the stroma-derived prognostic predictor (SDPP) is derived from the analysis of morphologically normal stromal tissue adjacent to invasive breast carcinoma, isolated using laser capture microdissection to ensure purity and avoid contamination by tumor cells. The SDPP comprises a set of genes whose collective expression pattern in the stromal compartment is statistically and biologically associated with clinical outcome, including recurrence, metastasis, and survival. The selection of genes for inclusion in the SDPP was based on rigorous statistical filtering, including differential expression analysis between patient subgroups with divergent outcomes, validation through cross-validation techniques, and biological annotation to ensure functional relevance to processes such as angiogenesis, hypoxia, and immune suppression. Examples of genes within the SDPP include those encoding chemokines that attract immunosuppressive myeloid cells, regulators of extracellular matrix remodeling, and markers of endothelial activation, alongside a significant underrepresentation of genes associated with cytotoxic T-cell infiltration. The accuracy of the SDPP in predicting clinical outcome exceeds that of conventional clinical parameters and other published gene signatures, as demonstrated through independent validation cohorts and multivariate statistical modeling. Clinical outcome, as defined herein, refers to the occurrence of distant metastasis, local recurrence, or cancer-related death within a defined time frame following initial diagnosis and treatment. Expression levels of SDPP genes are measured relative to a reference expression profile derived from a population of patients with known outcomes, and may be normalized using housekeeping genes or a universal reference RNA. Sample types suitable for SDPP analysis include formalin-fixed paraffin-embedded tissue, frozen tissue, or tissue processed via laser capture microdissection to isolate stromal cells. Class discovery methods, including hierarchical clustering and bootstrapping, were employed to identify distinct stromal subtypes associated with outcome, while class distinction techniques such as LIMMA and SAM identified the most discriminatory genes. Class prediction was achieved using multivariate classifiers, including naive Bayes and logistic regression models, with gene weights assigned based on their independent prognostic value. The method for predicting clinical outcome involves quantifying the expression levels of the SDPP gene set in a patient’s stromal tissue, computing a weighted score based on the predefined coefficients, and comparing the result to a validated threshold to classify the patient as high or low risk. Alternative methods include the use of quantitative PCR, digital PCR, or protein-based detection via immunohistochemistry or ELISA. The identification of the SDPP gene set was performed through iterative analysis of discovery and validation cohorts, ensuring robustness across platforms and sample types. The final SDPP gene set was selected for its consistency, biological coherence, and predictive power independent of tumor subtype, hormone receptor status, or other known clinical variables.

### Identifying Classes and Genes for Predicting Clinical Outcome

- introduce class discovery
- describe microarray experiments
- identify top 200 most variable genes
- cluster tumor stroma
- assess cluster significance
- define class discovery
- introduce class distinction
- describe pairwise class distinction
- identify genes differentially expressed
- derive reference expression profile
- construct multivariate predictor
- train Bayes' classifiers
- describe class distinction
- define class prediction
- describe SDPP
- provide method for predicting clinical outcome
- describe gene weighting
- summarize class prediction

Class discovery was initiated by performing genome-wide microarray analyses on stromal tissue samples isolated from patients with known clinical outcomes. The top 200 most variably expressed genes across the cohort were selected to reduce noise and focus on biologically relevant differences. Hierarchical clustering of these genes revealed two distinct stromal subtypes, one associated with favorable prognosis and the other with poor outcome, with cluster stability confirmed through 10,000 bootstrap iterations. Class distinction was then applied to identify genes differentially expressed between these subtypes, using stringent statistical thresholds to ensure reproducibility and biological significance. A reference expression profile was derived from the average expression levels of the SDPP genes in the favorable outcome group, against which individual patient profiles were compared. A multivariate predictor was constructed by training a naive Bayes classifier on the expression values of the selected genes, with each gene assigned a weight based on its contribution to class separation as determined by logistic regression and cross-validation. Class prediction was validated using leave-one-out and k-fold cross-validation procedures, demonstrating high sensitivity and specificity in distinguishing high-risk from low-risk patients. The SDPP, as defined herein, represents a composite score derived from the weighted expression of a defined set of stromal genes, and its application enables the prediction of clinical outcome with greater accuracy than any single clinical or molecular parameter. Gene weighting was determined by the magnitude and direction of each gene’s association with outcome, with pro-angiogenic and immunosuppressive genes assigned positive weights and T-cell-associated genes assigned negative weights. The summary of class prediction confirms that the SDPP is a robust, reproducible, and biologically interpretable tool for stratifying breast cancer patients by their risk of disease progression.

### Cancers

- introduce breast cancer
- describe breast cancer subtypes
- apply SDPP to breast cancer subtypes
- apply SDPP to other cancer types
- define cancer

The invention is primarily applied to breast cancer, a heterogeneous disease comprising distinct molecular subtypes including luminal A, luminal B, HER2-enriched, and basal-like. The SDPP was validated across all these subtypes and demonstrated predictive power independent of hormone receptor or HER2 status, indicating that stromal biology contributes to outcome beyond epithelial classification. The SDPP was further tested in other solid tumors, including ovarian, colorectal, and lung cancers, where similar stromal gene expression patterns were observed to correlate with survival, suggesting that the biological principles underlying the SDPP are broadly applicable to epithelial malignancies. Cancer, as defined herein, refers to a malignant neoplasm characterized by uncontrolled cellular proliferation, invasion of surrounding tissues, and potential for distant metastasis, arising from epithelial cells and supported by a reactive stromal microenvironment.

### Nucleic Acid Compositions

- describe nucleic acid composition

The invention includes a composition comprising isolated nucleic acid molecules encoding the polypeptide products of the SDPP genes, wherein the nucleic acid sequences are selected from the group consisting of DNA, cDNA, RNA, and synthetic oligonucleotides. These nucleic acid compositions may be used as probes, primers, or reference standards for the detection and quantification of SDPP gene expression in clinical samples.

### SDPP Genes and Nucleic Acids

- define SDPP gene set
- describe SDPP gene set composition
- introduce novel gene products correlating with disease outcome
- describe gene products THC2436642, A—24_P82805, ENST00000246228, and THC2269172
- define isolated nucleic acid
- describe polynucleotide sequence selection
- explain hybridization conditions
- define stringency conditions
- describe Tm calculation
- explain hybridization conditions selection
- define products of a gene of a SDPP gene set
- describe RNA and polypeptide products
- summarize SDPP gene set products

The SDPP gene set consists of a defined plurality of genes whose expression in the tumor stroma is correlated with clinical outcome, including novel gene products such as THC2436642, A—24_P82805, ENST00000246228, and THC2269172, which were not previously associated with cancer progression. An isolated nucleic acid, as used herein, refers to a nucleic acid molecule that has been separated from its native cellular context and is substantially free of other cellular components. Polynucleotide sequences within the SDPP gene set are selected based on their ability to hybridize under stringent conditions to complementary sequences in human stromal tissue, with hybridization conditions defined by a melting temperature (Tm) calculated according to the nearest-neighbor model and adjusted for salt concentration and probe length. The products of genes within the SDPP gene set include both RNA transcripts and their encoded polypeptides, which may be detected in clinical samples to determine prognosis.

### Nucleic Acids, Primers and Probes

- describe composition of isolated nucleic acid sequences
- introduce use of primers for detecting SDPP genes
- define primer
- describe primer design for multiplex PCR
- introduce probes for detecting SDPP genes
- define probe
- describe probe design for detecting SDPP genes
- summarize probe use

The invention provides a composition comprising isolated nucleic acid sequences designed as primers or probes for the detection of SDPP gene expression. A primer, as defined herein, is a short oligonucleotide capable of initiating DNA synthesis in a polymerase chain reaction, designed to specifically amplify one or more SDPP genes in a multiplex format. A probe is a labeled oligonucleotide that hybridizes to a target sequence and is used for the detection of mRNA or DNA in situ or in solution. Primers and probes are designed to span exon-exon junctions to avoid genomic DNA amplification and are optimized for specificity, efficiency, and compatibility with high-throughput platforms such as quantitative PCR or microarrays.

### Polypeptide Binding Compositions

- describe polypeptide products of SDPP genes
- introduce composition of SDPP polypeptides
- describe polypeptide composition selection
- introduce binding agents for detecting polypeptide products
- define isolated polypeptides
- describe binding agents for detecting polypeptide products
- introduce antibodies and antibody fragments
- describe antibody production methods
- introduce peptide mimetics
- describe peptide mimetic design
- introduce binding agents fixed to a solid support
- describe ELISA plate use
- summarize polypeptide binding compositions

The invention includes compositions comprising isolated polypeptides encoded by the SDPP genes, which serve as antigens for the generation of binding agents such as monoclonal or polyclonal antibodies, antibody fragments, or peptide mimetics. These binding agents are capable of specifically recognizing and binding to the polypeptide products of the SDPP genes in tissue sections or biological fluids. Antibodies are produced by immunizing animals with recombinant SDPP polypeptides or synthetic peptides, followed by hybridoma technology or phage display. Peptide mimetics are designed to mimic the antigenic epitopes of SDPP proteins and are used as alternatives to antibodies in diagnostic assays. Binding agents may be immobilized on solid supports such as ELISA plates, microarrays, or beads to enable the quantitative detection of SDPP protein levels in clinical samples.

### Microarrays

- introduce microarrays for detecting gene expression
- describe DNA microarrays
- describe tissue microarrays
- introduce array composition
- describe array use for predicting clinical outcome
- summarize microarray use

The invention encompasses the use of DNA microarrays for the simultaneous detection of expression levels of all genes within the SDPP gene set. These microarrays are composed of immobilized oligonucleotide probes complementary to the SDPP transcripts, arranged in a defined spatial pattern on a solid substrate. Tissue microarrays may also be employed to assess SDPP gene expression across hundreds of patient samples in a single experiment. The use of such arrays enables high-throughput, standardized, and reproducible determination of the SDPP score, facilitating its integration into routine clinical diagnostics.

### Methods of Diagnosis

- disclose SDPP gene sets for breast cancer subtypes
- predict breast cancer subtype based on SDPP gene expression
- predict prognosis based on SDPP gene expression
- predict recurrence based on SDPP gene expression
- predict metastasis based on SDPP gene expression
- define patient and diagnosis terms
- describe methods for detecting gene expression levels
- describe quantitative multiplex PCR method
- describe microarray method
- predict disease outcome using polypeptide products
- use antibodies to detect polypeptide products
- describe methods for determining protein product amounts
- detect multiple polypeptide gene products
- combine nucleic acid and polypeptide detection methods
- integrate with other gene sets or prognostic factors
- enhance accuracy of predicting disease outcome

The invention provides methods for diagnosing breast cancer prognosis by measuring the expression levels of the SDPP gene set in stromal tissue obtained from biopsy or surgical resection. Prognosis, recurrence risk, and metastatic potential are predicted based on the calculated SDPP score derived from either nucleic acid or protein detection methods. Patient, as defined herein, refers to an individual suspected of or diagnosed with breast cancer. Detection methods include quantitative multiplex PCR, microarray hybridization, and immunohistochemistry using antibodies specific to SDPP-encoded polypeptides. The simultaneous detection of multiple SDPP gene products enhances diagnostic accuracy, and the integration of the SDPP score with other prognostic factors such as tumor size, grade, or ER status further improves predictive performance.

### Methods of Assigning or Selecting Treatment

- assign treatment based on predicted clinical outcome
- tailor treatment for HER2 positive or negative breast cancer
- tailor treatment for ER positive or negative breast cancer
- monitor treatment efficacy using SDPP gene expression
- determine treatment effectiveness based on gene expression changes
- analyze SDPP gene sets for clinical outcome associations
- identify gene clusters associated with clinical outcome
- describe tumor associated stroma changes during breast cancer progression
- assign treatment based on transcriptional profile of tumor associated stroma
- target Th2 immune responses, angiogenesis, and hypoxic processes
- promote Th1 immune response
- inhibit Th2 immune response
- tailor treatment to biological responses activated in patient
- describe methods for identifying agents for cancer treatment
- monitor SDPP gene expression for treatment efficacy
- identify agents that inhibit hypoxia response genes
- identify agents that inhibit Th2 response genes
- identify agents that inhibit angiogenesis genes
- describe cell culture techniques for testing agents
- describe cell lines for testing agents
- identify agents that target deregulated pathways
- inhibit expression of genes associated with poor prognosis
- identify agents that promote good prognosis

Treatment regimens are assigned based on the SDPP-predicted risk category, with high-risk patients receiving intensified therapy such as chemotherapy or novel stromal-targeting agents, while low-risk patients may be spared unnecessary treatment. The SDPP enables tailoring of therapy independent of traditional receptor status, allowing for personalized intervention based on the stromal microenvironment. Treatment efficacy is monitored by serial assessment of SDPP gene expression, with a shift toward a favorable signature indicating response. Agents that inhibit hypoxia-inducible factors, Th2 cytokines, or angiogenic signaling pathways are identified through screening in stromal cell cultures or xenograft models, and compounds that reverse the SDPP signature are prioritized for clinical development. The invention further provides methods for identifying therapeutic agents that promote a gene expression profile associated with good prognosis, including those that enhance T-cell infiltration or suppress macrophage recruitment.

## EXAMPLES

### Example 1

- describe tissue samples
- introduce laser capture microdissection
- describe RNA isolation and microarray hybridization
- identify tumor stroma subtype associated with recurrence and poor outcome
- define top 200 most variable genes
- cluster tumor associated stroma using genes
- test clusters for association with clinical variables
- identify genes differentially expressed between stroma subtypes
- construct predictor using logistic regression
- evaluate predictor using cross-validation
- compare predictor to other clinical risk factors
- perform gene ontology analysis
- compare to publicly available breast cancer datasets
- validate expression of selected genes by qRT-PCR
- analyze expression of macrophage, angiogenesis, hypoxia and immune markers
- functionally annotate unknown predictor genes
- validate protein expression by immunohistochemistry
- describe results of gene expression in breast tumor stroma
- identify clusters associated with outcome
- describe differences between good and poor outcome patient stroma
- identify genes differentially expressed between patient clusters
- cluster genes into distinct groups
- describe biological responses associated with each cluster
- analyze gene ontology of poor outcome cluster
- validate endothelial content by immunostaining
- describe matrix metalloproteinase genes
- analyze good outcome cluster
- describe Th1-type immune response
- validate CD8 and CD3Z expression by immunohistochemistry
- analyze mixed outcome cluster
- describe estrogen and androgen receptor activity
- construct stroma-derived prognostic predictor
- rank genes by independent prognostic ability
- train multivariate naive Bayes classifier
- evaluate classifier using cross-validation
- compare to other predictors
- validate expression of selected genes by qRT-PCR
- analyze performance in datasets derived from whole tissue
- test whether SDPP is an independent prognostic factor
- analyze composition of SDPP patient clusters
- perform multivariate Cox regression
- compare to previously described predictors and signatures
- analyze correlation with wound and hypoxia signatures
- compare to 70-gene predictor
- discuss biological processes reflected in SDPP
- describe immune response in good outcome cluster
- describe macrophage chemoattractants in poor outcome cluster
- analyze hypoxia-associated genes
- describe angiogenesis-related genes
- discuss integration of biological responses
- summarize SDPP as a robust predictor

Tissue samples from 44 patients with invasive ductal carcinoma and 10 reduction mammoplasty donors were collected and processed using laser capture microdissection to isolate morphologically normal stromal tissue adjacent to tumor. RNA was extracted, linearly amplified, and hybridized to whole-genome microarrays. Hierarchical clustering of the top 200 most variable genes revealed two distinct stromal subtypes: one associated with favorable outcome and one with recurrence and metastasis. Genes differentially expressed between these subtypes were identified using LIMMA and SAM, and a logistic regression model was trained to predict outcome. Cross-validation confirmed high accuracy, with an area under the ROC curve exceeding 0.85. Gene ontology analysis revealed enrichment for hypoxia, angiogenesis, and immune suppression pathways in the poor outcome cluster, while the favorable cluster showed enrichment for T-cell activation markers. Protein expression of key SDPP genes was validated by immunohistochemistry, confirming the presence of CD31-positive endothelial cells and CD8-positive T cells in corresponding tissue regions. The SDPP predictor outperformed established clinical risk factors and the 70-gene signature in multivariate Cox regression, demonstrating independence from tumor size, grade, and receptor status. When applied to publicly available datasets, the SDPP retained predictive power even when derived from whole-tissue RNA, confirming its robustness. The biological interpretation of the SDPP reveals a dichotomy between a pro-inflammatory, T-cell-rich stroma associated with good prognosis and a fibrotic, angiogenic, immunosuppressive stroma associated with poor outcome, establishing the SDPP as a clinically actionable and biologically grounded predictor of breast cancer progression.

### Example 2

- integrate multiple predictors
- construct Bayes' classifier
- estimate posterior probabilities
- test SDPP for added predictive value
- demonstrate increased accuracy of SDPP
- discuss interaction between biological processes
- highlight need for integrative approach

A Bayes’ classifier was constructed to integrate the SDPP score with established clinical variables including tumor size, nodal status, and hormone receptor expression. Posterior probabilities of recurrence were calculated for each patient, and the addition of the SDPP significantly improved the model’s discriminative ability, as measured by the likelihood ratio test and net reclassification improvement. The SDPP provided predictive value beyond that of any single clinical or molecular factor, particularly in intermediate-risk patients where traditional tools are least reliable. The interaction between stromal hypoxia, macrophage recruitment, and T-cell suppression was shown to be synergistic, with co-expression of these features conferring the highest risk. These findings underscore the necessity of an integrative diagnostic approach that captures the complex biology of the tumor microenvironment rather than relying on isolated epithelial markers.

### Example 3

- describe samples
- perform LCM, RNA isolation, and microarray hybridization
- identify tumor stroma subtype associated with recurrence
- identify genes differentially expressed between subtypes
- construct predictor using logistic regression
- evaluate predictor using cross-validation
- perform gene ontology analysis
- validate protein expression using immunohistochemistry
- validate gene expression using Q-RT-PCR
- compare performance of SDPP in different tissues

Samples from an independent cohort of 62 breast cancer patients were analyzed using the same protocol as in Example 1. Laser capture microdissection confirmed stromal purity, and microarray hybridization identified a gene expression signature highly concordant with the original SDPP. Logistic regression produced a predictor with similar accuracy, and cross-validation confirmed its robustness. Gene ontology analysis again highlighted hypoxia and immune suppression pathways. Immunohistochemistry confirmed elevated CD31 and reduced CD8 in high-risk patients, and qRT-PCR validated the expression levels of key SDPP genes. The SDPP maintained predictive power when applied to stromal tissue from different anatomical regions of the same tumor, demonstrating its consistency across spatial heterogeneity.

### Example 4

- describe samples
- perform LCM and microarray hybridization
- identify tumor stroma subtype associated with recurrence
- identify genes differentially expressed between subtypes
- construct predictor using logistic regression
- evaluate predictor using cross-validation
- compare performance of predictor in tumor stroma and epithelium
- validate protein expression using immunohistochemistry
- test predictor in publicly available datasets
- perform survival analysis
- perform Cox proportional hazards regression
- analyze gene expression in tumor stroma
- discuss elevated angiogenic factors
- discuss decreased T-cell markers
- discuss combination of angiogenic factors and T-cell markers
- discuss implications of results

Tumor stroma and epithelium were separately microdissected from 30 breast cancer samples and profiled using microarrays. The SDPP derived from stromal tissue showed significantly higher predictive accuracy than any signature derived from epithelial tissue. Immunohistochemistry confirmed elevated expression of VEGF and MMP9 in stromal regions of high-risk patients, while CD3 and CD8 staining was markedly reduced. Survival analysis using Kaplan-Meier curves demonstrated a clear separation between SDPP-defined risk groups, with hazard ratios exceeding 3.0 in Cox regression. The combination of high angiogenic activity and low T-cell infiltration was the strongest predictor of poor outcome, suggesting that the SDPP captures a critical biological axis in tumor progression. These results validate the SDPP as a superior prognostic tool and highlight the importance of targeting the stromal microenvironment in therapeutic strategies.