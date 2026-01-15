## BACKGROUND

- The early detection of cancer remains a critical unmet need in modern medicine, as the majority of cancer-related deaths occur in patients diagnosed at advanced stages when therapeutic options are limited and survival rates are significantly diminished. Despite decades of public health initiatives promoting population-wide screening, adherence to current modalities such as colonoscopy, fecal occult blood testing, and imaging-based surveillance remains suboptimal due to invasiveness, discomfort, logistical burden, and cost. These limitations result in delayed diagnoses, particularly for cancers with insidious progression, such as colorectal, pancreatic, and ovarian malignancies, where asymptomatic early-stage disease often goes undetected until metastatic spread has occurred. Consequently, there is an urgent and persistent demand for non-invasive, highly accurate, and scalable screening tools capable of identifying cancer at its most treatable phase.

- Current screening methods for epithelial cancers are constrained by inherent biological and technical limitations. Colonoscopy, while effective, requires bowel preparation, sedation, and specialized personnel, leading to low compliance rates. Fecal immunochemical tests, though widely adopted, exhibit insufficient sensitivity for precancerous adenomas and early-stage carcinomas, particularly those with low shedding rates. Imaging modalities such as CT or MRI are costly, expose patients to ionizing radiation, and lack the specificity required for population-level deployment. Moreover, existing biomarkers, including carcinoembryonic antigen and CA19-9, lack the sensitivity and specificity necessary for reliable early detection, frequently yielding false positives due to benign inflammatory conditions or false negatives in low-tumor-burden states. These shortcomings underscore the inadequacy of current approaches in capturing the complex, systemic biological changes that accompany the earliest phases of malignancy.

- Blood-based diagnostic tests represent a transformative paradigm in cancer screening, offering the potential for minimally invasive, repeatable, and scalable detection through the analysis of molecular signatures released into circulation. Unlike tissue biopsies, liquid biopsies can capture dynamic, real-time information about tumor biology and host response without requiring surgical intervention. The advent of next-generation sequencing has enabled the interrogation of cell-free nucleic acids—particularly cell-free DNA—that originate from both tumor and non-tumor cells. These circulating analytes reflect not only direct tumor-derived alterations but also systemic physiological perturbations induced by the presence of malignancy, including immune activation, stromal remodeling, and epigenetic reprogramming of circulating cell populations. This broader biological signal offers a more comprehensive window into disease state than mutation-centric approaches alone.

- The analysis of biological analytes in blood, however, presents formidable challenges due to the extreme complexity and low abundance of disease-relevant signals. Tumor-derived DNA typically constitutes less than one percent of total cell-free DNA in early-stage cancer, often obscured by a background of DNA derived from hematopoietic, endothelial, and apoptotic non-malignant cells. Furthermore, biological noise from clonal hematopoiesis, age-related genomic instability, and inter-individual variation in fragmentation patterns complicates signal extraction. Technical variability introduced during sample collection, storage, library preparation, and sequencing further obscures true biological signals, leading to batch effects and institutional biases that can confound machine learning models if not rigorously controlled. These challenges necessitate analytical frameworks that are not only sensitive to subtle molecular changes but also robust to confounding sources of variation.

- Machine learning has emerged as a powerful tool for uncovering complex, non-linear patterns within high-dimensional biological data, enabling the integration of diverse molecular features into unified diagnostic models. Unlike traditional hypothesis-driven approaches that rely on predefined biomarkers, machine learning algorithms can learn predictive relationships directly from genome-wide measurements without prior assumptions about the nature or origin of the signal. This capability is particularly valuable in cancer diagnostics, where the pathophysiology involves multifactorial interactions between tumor cells and the host microenvironment. Recent advances in computational biology have demonstrated that machine learning models trained on whole-genome sequencing profiles can detect cancer with high accuracy by identifying subtle, distributed patterns in fragment length, methylation, and copy number variation that are imperceptible to conventional analysis.

- Despite the promise of machine learning in diagnostics, there remains a critical need for novel methods that overcome the limitations of existing approaches by integrating multi-analyte signals, rigorously controlling for confounders, and operating effectively at the low tumor fractions characteristic of early-stage disease. Current methods often fail to generalize across diverse populations, are overly reliant on tumor-specific mutations that are absent in early lesions, or lack the sensitivity required for population screening. There is an unmet requirement for a scalable, blood-based diagnostic system that leverages the full spectrum of circulating analytes—nucleic acids, proteins, and metabolites—through a unified machine learning framework capable of distinguishing cancer from non-cancer states with high accuracy, even in the absence of overt tumor-derived genomic alterations. Such a method would represent a paradigm shift in early cancer detection, transforming liquid biopsy from a tool for monitoring advanced disease into a preventive screening modality.

## BRIEF SUMMARY

- A machine learning approach is disclosed for the classification of cancer status in a subject based on the integrated analysis of multiple classes of cell-free biological analytes derived from a blood sample. This method leverages high-dimensional molecular data to identify patterns indicative of malignancy that are not discernible through single-analyte or mutation-focused assays, enabling the detection of early-stage epithelial cancers with high sensitivity and specificity.

- The method focuses on the non-cellular portion of circulation, specifically analyzing cell-free nucleic acids, proteins, metabolites, and carbohydrates that are released into the bloodstream as a consequence of tumor-host interactions. These analytes collectively reflect a systemic biological response to malignancy, including epigenetic reprogramming, immune modulation, and tissue remodeling, rather than relying solely on tumor-derived mutations or copy number alterations.

- The method employs a trained machine learning classifier that receives a feature vector derived from quantitative measurements of multiple analyte classes, processes the vector through a predictive algorithm, and outputs a classification indicating the likelihood of cancer presence. The classifier is trained on a dataset of biological samples from subjects with and without cancer, using known clinical labels to optimize its ability to distinguish malignant from non-malignant states.

- The method involves assaying multiple classes of molecules including nucleic acids, polyamino acids, carbohydrates, and metabolites, each contributing orthogonal biological information to the overall diagnostic signal. These analytes are measured using standardized, scalable assays that generate quantitative data suitable for computational integration.

- The features for the machine learning model include quantitative measurements of fragment length distributions, methylation levels at CpG sites, read counts across annotated genomic regions, protein concentrations, autoantibody reactivity profiles, glycan structures, and metabolic pathway abundances. Each feature is derived from a specific analytical assay and represents a measurable biological property associated with cancer biology.

- A feature vector is prepared by aggregating all measured values from the multiple analyte classes into a single, standardized numerical representation. Each element of the vector corresponds to a specific analyte measurement, and the vector is normalized to account for technical variation, sample volume, and biological covariates such as age and sex.

- The machine learning model is loaded into computer memory from a non-transitory storage medium, where it has been previously trained using a set of training vectors derived from biological samples with known cancer status. The model contains learned weights and decision boundaries that enable it to interpret the feature vector and generate a classification output.

- The feature vector is input into the machine learning model, which applies a mathematical transformation based on the learned parameters to compute a decision score. This score reflects the probability that the subject has cancer, and is compared against a pre-defined threshold to generate a binary or probabilistic classification.

- The output classification indicates whether the subject is likely to have cancer, and may further specify the likelihood of cancer type, stage, or tissue of origin. The classification is generated without requiring prior knowledge of the specific molecular alterations present in the sample, relying instead on the collective pattern of analyte measurements.

- The classes of molecules assayed include polynucleotides such as cell-free DNA and cell-free RNA, polypeptides such as circulating proteins and autoantibodies, carbohydrates such as glycosylated epitopes and glycosaminoglycans, and metabolites such as amino acid derivatives and lipid species. Each class contributes distinct biological information to the diagnostic signature.

- Examples of nucleic acids include fragmented cell-free DNA, microRNAs, long non-coding RNAs, and mitochondrial DNA. These molecules are analyzed for abundance, fragmentation patterns, methylation status, and sequence variants.

- Examples of polyamino acids include circulating proteins such as ZNF700, p53, CEA, and CA19-9, as well as autoantibodies directed against tumor-associated antigens. These are measured using immunoassays or mass spectrometry to determine concentration and reactivity profiles.

- Examples of carbohydrates include sialylated glycans, N-linked oligosaccharides, and heparan sulfate fragments. These are detected using lectin arrays, mass spectrometry, or enzymatic digestion followed by chromatographic separation.

- Examples of metabolites include lactate, ketone bodies, amino acid ratios, and bile acid derivatives. These are quantified using gas or liquid chromatography coupled with mass spectrometry to assess metabolic dysregulation associated with malignancy.

- The plurality of assays includes low-coverage whole-genome sequencing, targeted bisulfite sequencing, small RNA sequencing, quantitative immunoassays, glycan profiling, and metabolomic profiling. Each assay is optimized for sensitivity, throughput, and compatibility with clinical sample types.

- Examples of assays include polymerase chain reaction for microRNA detection, enzyme-linked immunosorbent assay for protein quantification, liquid chromatography-mass spectrometry for metabolite analysis, and hybridization-based capture for methylation profiling.

- The biological sample is classified by comparing its feature vector to a reference model trained on samples from cancer and non-cancer subjects. The classification is based on the aggregate pattern of analyte measurements rather than any single biomarker.

- Examples of machine learning algorithms include logistic regression, support vector machines, random forests, gradient boosting, and deep neural networks. The model may be trained using supervised learning with labeled training samples to optimize classification accuracy.

- The specified property is the presence or absence of cancer, and the classification output may further indicate tumor burden, stage, tissue of origin, or predicted responsiveness to therapy.

- Examples of clinically-diagnosed disorders include colorectal cancer, pancreatic adenocarcinoma, ovarian cancer, lung adenocarcinoma, hepatocellular carcinoma, and esophageal squamous cell carcinoma. The method is applicable to any epithelial malignancy detectable through circulating analytes.

- Responsiveness to treatment is inferred by comparing the feature vector to models trained on subjects who have responded or not responded to specific chemotherapeutic agents, immunotherapies, or targeted therapies.

- Continuous measurement of patient trait is enabled by repeated sampling and reclassification over time, allowing for dynamic monitoring of disease progression, treatment efficacy, or recurrence.

- The system for performing classifications includes a receiver for acquiring biological sample data, a feature module for generating feature vectors, an analysis module for applying the machine learning model, a labeling module for assigning classification outputs, a comparator module for evaluating model performance, a training module for updating the model with new data, and an output module for delivering diagnostic results.

- The receiver is a hardware or software interface that ingests raw assay data from sequencing platforms, mass spectrometers, or immunoassay readers and converts it into digital measurements.

- The feature module processes raw data into normalized, standardized features by applying GC correction, depth normalization, length adjustment, and batch effect removal.

- The analysis module executes the machine learning model, computes the decision score, and determines the classification based on a pre-defined threshold.

- The labeling module assigns a categorical or probabilistic label to the classification output, such as “cancer detected,” “low probability,” or “tumor fraction estimated at 1.2%.”

- The comparator module evaluates the model’s performance against ground truth labels, calculates metrics such as area under the curve and sensitivity at fixed specificity, and flags performance degradation.

- The training module updates the machine learning model with new labeled samples, re-optimizes weights, and validates performance on held-out data to ensure continual improvement.

- The output module delivers the classification result to a clinician, electronic health record system, or patient portal in a human-readable format.

- The system for classifying subjects includes a classification circuit comprising a processor, memory, and software instructions that execute the machine learning model on input feature vectors.

- The non-transitory computer-readable medium stores the machine learning model, training data, calibration parameters, and executable code for performing the classification method on a computing device.

## TERMS

- The terms “a,” “an,” and “the” refer to one or more of the referenced element, unless the context clearly indicates otherwise. For example, “a biological sample” may include one or more individual samples processed in parallel.

- The term “or” is used in the inclusive sense, meaning that any combination of the listed elements may be present, including all of them, unless explicitly stated otherwise.

- The phrase “based on” means that the determination, calculation, or classification is influenced by or derived from the specified factor, but may also incorporate additional variables or conditions.

- The term “area under the curve” (AUC) refers to the area under the receiver operating characteristic curve, which quantifies the ability of a diagnostic test to distinguish between two classes across all possible thresholds. An AUC of 1.0 indicates perfect discrimination, while an AUC of 0.5 indicates no discrimination.

- Receiver operating characteristic (ROC) curves are graphical representations of the trade-off between sensitivity and specificity across varying classification thresholds. The curve is constructed by plotting the true positive rate against the false positive rate at different decision thresholds.

- The term “cancer” and “cancerous” refer to a pathological condition characterized by uncontrolled cell proliferation, invasion of surrounding tissues, and potential for metastasis. This includes carcinomas, sarcomas, lymphomas, and other malignant neoplasms of epithelial origin.

- The term “cancer-free” refers to a subject who does not have a diagnosis of cancer at the time of sample collection and who has no evidence of malignancy based on standard clinical and imaging criteria.

- A “genetic variant” or “variant” is a difference in nucleotide sequence between an individual’s genome and a reference genome. Variants may include single-nucleotide polymorphisms, insertions, deletions, or structural variations.

- A “germline variant” is a genetic variant present in all somatic cells of an individual and inherited from a parent, as opposed to a somatic variant acquired during life.

- “Input features” or “features” are quantitative measurements derived from biological assays that serve as inputs to a machine learning model. Each feature represents a measurable biological property, such as read count, methylation level, or protein concentration.

- A “machine learning model” or “model” is a computational algorithm trained on labeled data to recognize patterns and make predictions. The model contains parameters learned from training data that enable it to classify new, unseen samples.

- Model training involves adjusting the parameters of a machine learning algorithm to minimize the difference between predicted outputs and known true labels in a training dataset.

- A “marker” or “marker protein” is a biological molecule whose presence, absence, or concentration is associated with a specific physiological or pathological state, such as cancer.

- Marker detection refers to the measurement of a marker using a specific assay, such as immunoassay, sequencing, or mass spectrometry, to determine its abundance or modification state.

- “Non-cancerous tissue” refers to tissue that lacks malignant transformation and does not exhibit the histological, molecular, or functional characteristics of cancer.

- “Normal tissue” or “healthy tissue” refers to tissue that exhibits typical structure, gene expression, and function for the organ or cell type in the absence of disease.

- “Polynucleotides,” “nucleotide,” “nucleic acid,” and “oligonucleotides” refer to polymers of nucleotide monomers, including DNA and RNA, whether single-stranded or double-stranded, natural or synthetic.

- Polynucleotide structure includes the sequence of nucleotide bases, secondary folding, methylation patterns, and fragmentation profiles that influence biological function and detection.

- A “polypeptide,” “protein,” or “peptide” refers to a chain of amino acids linked by peptide bonds, whether naturally occurring or modified, and includes full-length proteins, proteolytic fragments, and post-translationally modified variants.

- Protein modifications include phosphorylation, glycosylation, acetylation, ubiquitination, methylation, and cleavage events that alter protein function, stability, or immunogenicity.

- A “prediction” is a computational estimate of a biological state, such as cancer presence, derived from a machine learning model based on input features.

- Predictive methods involve the use of statistical or algorithmic models to infer an outcome from observable data, without direct measurement of the outcome itself.

- “Prognosis” refers to the prediction of the likely course and outcome of a disease, including likelihood of recurrence, progression, or survival.

- “Specificity” is the proportion of true negative cases correctly identified by a diagnostic test, calculated as the number of true negatives divided by the sum of true negatives and false positives.

- “Sensitivity” is the proportion of true positive cases correctly identified by a diagnostic test, calculated as the number of true positives divided by the sum of true positives and false negatives.

- A “structural variation” (SV) is a large-scale alteration in the genome, including deletions, duplications, inversions, translocations, or copy number changes exceeding 50 base pairs.

- A “subject” is a human individual from whom a biological sample is obtained for diagnostic purposes.

- Subject characteristics include age, sex, ethnicity, medical history, and clinical stage, which may be used as covariates in model training or analysis.

- A “training sample” is a biological sample with a known clinical label, such as cancer or non-cancer, used to train a machine learning model.

- A “training vector” is a feature vector derived from a training sample, comprising all measured analyte values used to teach the model the relationship between molecular patterns and clinical status.

- “Tumor,” “neoplasia,” “malignancy,” or “cancer” refer to an abnormal growth of cells with uncontrolled proliferation, invasion, and potential for metastasis.

- “Tumor burden” refers to the total amount of cancerous tissue in the body, often estimated by the proportion of tumor-derived DNA in circulation.

- A “nucleic acid sample” is a preparation of DNA or RNA isolated from a biological fluid, such as plasma or serum, containing cell-free molecules.

- A “barcode” is a unique nucleotide sequence added to a molecule during library preparation to identify its origin, enabling multiplexing and error correction.

- A “barcode sequence” is a short, defined nucleotide string used to tag individual molecules for tracking during sequencing and analysis.

- “Tagmentation” or “ligation reaction” refers to enzymatic processes used to fragment and adapter-ligate nucleic acids for sequencing, such as Tn5 transposase-mediated fragmentation.

- “Nucleic acid amplification” refers to the enzymatic replication of nucleic acid molecules, such as PCR or isothermal amplification, to increase detectable signal.

- An “amplified product” is the result of nucleic acid amplification, containing multiple copies of a target sequence suitable for detection or sequencing.

## DETAILED DESCRIPTION

- Medical diagnostic methods are disclosed that utilize machine learning to analyze multi-analyte profiles from biological samples for the detection of cancer. These methods differ from conventional approaches by integrating heterogeneous molecular data into a unified predictive framework, enabling the identification of subtle, systemic signatures of malignancy that are not detectable through single-analyte biomarkers.

- Machine learning approaches enable the discovery of complex, non-linear relationships between molecular features and disease status without requiring prior hypotheses about the biological origin of the signal. This is particularly valuable in early-stage cancer, where tumor-derived alterations are sparse and confounded by background noise.

- The advantages of these methods over other diagnostic approaches include higher sensitivity for early-stage disease, reduced reliance on tumor-specific mutations, scalability to population screening, and the ability to incorporate orthogonal signals from multiple analyte classes. Unlike mutation-focused liquid biopsies, these methods do not require ultra-deep sequencing and can be implemented at lower cost and with smaller blood volumes.

- The non-cellular portion of circulation, including cell-free DNA, RNA, proteins, and metabolites, reflects systemic biological changes induced by cancer, such as immune activation, stromal remodeling, and epigenetic reprogramming. These changes are detectable even when tumor-derived DNA is present at extremely low fractions, making them ideal targets for early detection.

- Applications of these methods include population-based cancer screening, monitoring of treatment response, early detection of recurrence, and stratification of patients for personalized therapy. The method is applicable to a broad range of epithelial cancers, including colorectal, pancreatic, ovarian, and lung cancers.

### I. CIRCULATING ANALYTES AND CELLULAR DECONSTRUCTION WITH BIOLOGICAL ASSAYS

- Cost-effective assays are essential for population-level screening, requiring high throughput, minimal sample volume, and compatibility with clinical workflows. Analytes are defined as measurable molecular entities derived from biological fluids, including nucleic acids, proteins, carbohydrates, and metabolites.

- DNA analytes include fragmented cell-free DNA, mitochondrial DNA, and viral or bacterial DNA. Types of DNA analytes include those with cancer-associated copy number variations, methylation alterations, and fragmentation patterns.

- RNA analytes include messenger RNA, microRNA, long non-coding RNA, and circular RNA. Types of RNA analytes include those derived from tumor cells, immune cells, or stromal cells, and those exhibiting differential expression in cancer.

- Polyamino acid analytes include circulating proteins, peptides, and autoantibodies. Types include tumor-associated antigens, immune checkpoint proteins, and cytokines.

- Other analytes include carbohydrates such as glycosylated epitopes and metabolites such as amino acid derivatives and lipid species. Types include sialylated glycans, bile acids, and ketone bodies.

- The combination of analytes enhances diagnostic accuracy by capturing complementary biological signals. Selection of analyte combinations is based on biological plausibility, assay feasibility, and machine learning performance metrics.

- Examples of analyte combinations include cell-free DNA methylation plus microRNA expression plus autoantibody reactivity, or protein concentration plus metabolite ratio plus fragment length distribution.

- The importance of analyte selection lies in the ability to maximize diagnostic signal while minimizing noise, redundancy, and technical variability.

### II. SAMPLE PREPARATION

- A biological sample is obtained from a subject, typically a venous blood draw, and processed to isolate cell-free nucleic acids using magnetic bead-based extraction. The sample is then processed to purify nucleic acid molecules by removing cellular debris and high molecular weight genomic DNA.

- Analytes are separated using size-exclusion chromatography, differential centrifugation, or selective precipitation. Higher molecular weight nucleic acid molecules are removed by size selection or enzymatic digestion.

- Nucleic acid molecules are modified by oxidation, fragmentation, or chemical conversion to enhance detection. Oxidation of cytosine residues is performed to distinguish 5-hydroxymethylcytosine from 5-methylcytosine.

- Nucleic acid molecules are tagged or barcoded using adapter ligation or transposase-mediated tagging to enable multiplexed sequencing and error correction.

- The sample is partitioned into aliquots for parallel analysis of different analyte classes. Cellular DNA is separated from cell-free DNA using density gradient centrifugation or selective lysis.

- Cellular components are detected via flow cytometry or microscopy to confirm purity. Nucleic acid molecules are quantified using fluorometric assays or digital PCR.

- Blood samples are obtained from healthy individuals and individuals with cancer, with clinical labels confirmed by histopathology and imaging.

- The presence of adenocarcinoma and colorectal cancer is detected by comparing molecular profiles to reference databases. Differentiation between cancer stages and tumor sizes is achieved through machine learning models trained on stage-specific signatures.

- A sequencing library is prepared by adding adapter sequences, incorporating molecular barcodes, and performing end-repair and ligation. The library is amplified to generate sufficient material for sequencing.

- Nucleic acid molecules are treated for methylation analysis by bisulfite conversion, which deaminates unmethylated cytosine to uracil while leaving methylated cytosine unchanged.

- 5-hydroxymethylcytosine is converted to 5-formylcytosine and 5-carboxylcytosine for differential detection. Sequencing is performed using high-throughput platforms such as Illumina NovaSeq.

- Sequencing libraries are prepared using targeted or whole-genome approaches. Nucleic acid molecules are amplified using PCR or isothermal methods.

- Targeted sequencing is performed using hybridization capture or amplicon-based panels. Whole-genome sequencing is performed at low coverage (approximately 5–10X) to enable cost-effective population screening.

- Biological information is prepared by aligning sequencing reads to a reference genome and generating quantitative features such as read counts per gene or methylation level per CpG site.

- Sequencing information is processed to remove low-quality reads, duplicates, and artifacts. Assays are performed on biological samples using standardized protocols.

- Assays are selected based on the machine learning model’s feature requirements. Biological assays are performed on different portions of the same sample to preserve material.

- Feature data for machine learning analysis is generated by aggregating quantitative measurements into a feature vector. Assays and machine learning models are integrated into a unified diagnostic pipeline.

- Sample preparation is motivated by the need to detect copy number variation, which is a hallmark of genomic instability in cancer. Copy number variation detection is performed using read depth analysis across genomic bins.

- Genome-wide detection of copy number alterations is achieved by comparing observed read counts to expected values based on GC content and mappability.

- Chromosomal instability analysis is performed using fragmentation endpoint analysis and length mixture modeling to detect tumor-specific fragmentation patterns.

- Manual inspection of large-scale CNVs is performed to validate algorithmic calls. Changes in gene expression are motivated by the observation that tumor microenvironment alters the transcriptome of circulating cells.

- Microarray analysis is used to measure gene expression levels in cell-free RNA. Metrics of cfDNA concentration include total yield, fragment size distribution, and nucleosome periodicity.

- Somatic mutation analysis is performed using deep sequencing of targeted regions. Low-coverage whole genome sequencing is used for CNV detection, while deep WGS and targeted sequencing are used for mutation detection.

- Somatic mutation analysis features include variant allele frequency, clonality, and mutational signatures. Transcription factor profiling is performed by inferring binding sites from nucleosome positioning patterns.

- Inference of transcription factor binding is performed by analyzing nucleosome depletion at known binding sites. Nucleosome signatures at transcription factor binding sites are detected using shallow WGS data profiles.

- Transcription factor binding site plasticity is assessed by comparing patterns across cancer and non-cancer samples. cfDNA fragmentation patterns are analyzed for nucleosome footprints.

- Hematopoietic transcription factor-nucleosome footprints are identified using curated lists of binding sites. Accessibility scores and z-scores are calculated to quantify chromatin openness.

- A method for diagnosing a disease is introduced, involving generation of a coverage pattern for a transcription factor, processing the pattern to remove noise, comparing the signal to a reference signal from healthy controls, and diagnosing the disease based on deviation from normal.

- Inferring chromosome structure and chromatin state is performed using assays such as Hi-C and cfHi-C. Predicting chromatin state of genes is done using probabilistic graphical models that relate accessibility to gene expression.

- Expression of genes is controlled by the access of cellular machinery to regulatory regions. A tissue-of-origin assay is introduced to infer the anatomical source of tumor-derived material.

- Cell-type-of-origin inference is motivated by the observation that different cancers release distinct epigenetic signatures. Genetic features for inference include methylation patterns, fragmentation profiles, and nucleosome positioning.

- Reference population values are prepared from healthy donors. Sample values are prepared from cancer patients. Matrix multiplication and parameter optimization are performed to estimate cell-type proportions.

- The type and proportion of cell types are determined using deconvolution algorithms. A method of processing a sample is introduced, involving sequencing information, preparation of first, second, and third arrays of values corresponding to different analyte classes.

- Methylation sequencing is performed using enzymatic methyl sequencing or bisulfite conversion. Whole genome bisulfite sequencing is used to map methylation at single-base resolution.

- Modification of nucleic acid molecules includes bisulfite treatment, oxidation, and enzymatic cleavage. Methylation analysis metrics include beta values, delta methylation, and differentially methylated region scores.

- A machine learning approach for nucleosome positioning is introduced, involving alignment of fragments to nucleosome-free regions and calculation of periodicity scores.

- A method for determining genetic sequence features is introduced, optionally including enrichment for CpG islands or promoter regions. Differentially methylated regions analysis is performed using statistical tests such as Wilcoxon rank-sum.

- Haplotype block assays are performed to detect linkage disequilibrium patterns associated with cancer. cfRNA assays are performed using small RNA sequencing and alignment to reference transcriptomes.

- RNA sequencing and alignment are performed using tools such as STAR or HISAT2. RNA fragments are counted and normalized using transcripts per million.

- Sample preparation involves aggregating reads for microRNA detection. Direct detection methods include hybridization to complementary probes.

- Hybridization-based RNA assays use locked nucleic acid probes. In situ hybridization protocol involves fixation, permeabilization, probe hybridization, and signal amplification.

- Probe requirements include specificity, melting temperature, and avoidance of secondary structure. PCR reaction is performed using primers specific to cancer-associated transcripts.

- Quantitative PCR methods include SYBR Green and TaqMan assays. Fluorogenic quantitative PCR uses hydrolysis probes for real-time detection.

- Other suitable amplification methods include loop-mediated isothermal amplification and rolling circle amplification. RNA markers associated with cancer include miR-21, miR-155, and MALAT1.

- Poly-amino acid and autoantibody assays are performed using immunoassay or mass spectrometry. Protein assays use enzyme-linked immunosorbent assay or mass spectrometry-based quantification.

- Immunoassay methods include sandwich ELISA, electrochemiluminescence, and lateral flow. Protein data normalization is performed using total protein concentration or spike-in controls.

- Cancer-associated peptide and protein sequences include ZNF700, p53, MUC1, and CA19-9. Cancer-associated peptide or protein markers are identified by differential expression analysis.

- Autoantibody detection is performed using protein microarrays or immunosorbent assays. Immunosorbent assay methods involve immobilizing antigens and detecting bound antibodies.

- Protein microarrays are printed with hundreds of tumor-associated antigens. Metrics for autoantibody assay include signal-to-noise ratio and reactivity index.

- Autoantibody markers are associated with cancer subtypes or stages. Tumor-associated antigens include ZNF700, which is specifically recognized by autoantibodies in colorectal cancer.

- Anti-p53 antibody assay is performed using recombinant p53 protein as capture antigen. Carbohydrate assays are performed using lectin arrays or mass spectrometry.

- Methods for measuring carbohydrates include glycan profiling by HPLC and glycosylation site mapping. Metrics from carbohydrate assays include sialylation index and fucosylation ratio.

### III. EXAMPLE SYSTEMS

- System architecture is introduced, comprising a network of interconnected devices for sample processing, data acquisition, and computational analysis. Data analysis is performed within measurement devices, such as sequencers and mass spectrometers, or on remote computing hardware.

- Software code is executed on computing hardware to implement the machine learning model. Modules and devices are defined as discrete functional units, including receivers, feature modules, analysis modules, and output modules.

- The data receiving module ingests raw data from sequencing platforms, immunoassay readers, and metabolomic instruments. The data pre-processing module performs normalization, batch correction, and outlier removal.

- The data analysis module for genomic data performs read alignment, feature counting, and methylation calling. The data interpretation module applies the machine learning model to generate classification outputs.

- The machine learning model implementation is stored as executable code on a non-transitory medium. The data visualization module generates ROC curves, feature importance plots, and diagnostic probability distributions.

- Computer systems for implementing methods include processors, memory, storage, and input/output interfaces. Computational analysis on nucleic acid sequencing data includes variant calling, copy number estimation, and fragmentation pattern analysis.

- Variant identification is performed using probabilistic modeling to distinguish somatic from germline variants. Statistical modeling methods include logistic regression and Bayesian inference.

- Mechanistic modeling methods simulate biological processes such as DNA fragmentation and methylation dynamics. Network modeling methods represent interactions between genes, proteins, and metabolites.

- Statistical inferences methods calculate confidence intervals and p-values for feature significance. Non-limiting examples of analysis methods include principal component analysis, hierarchical clustering, and support vector machines.

- Germline variation and somatic mutation are distinguished using population databases and matched normal tissue. Natural or normal variations include single-nucleotide polymorphisms and common structural variants.

- Acquired or abnormal variations include tumor-specific mutations and copy number alterations. Distinguishing between germline variants is performed using matched tissue controls or population allele frequencies.

- Identified variants are used for healthcare improvement by enabling early intervention, personalized therapy, and risk stratification. System 100 for performing methods includes a computer system 101 with components such as CPU, memory, and storage.

- Measurement devices 151, 152, or 153 include sequencers, mass spectrometers, and immunoassay readers. Computer system 101 operations include data ingestion, preprocessing, model execution, and result output.

- Network 130 enables distributed computing across multiple institutions. Cloud computing platforms provide scalable computational resources for model training and validation.

- CPU 105 executes machine-readable instructions stored in memory. Storage unit 115 stores sequencing data, model parameters, and patient records.

### IV. MACHINE LEARNING TOOLS

- Machine learning is introduced for assessing assay effectiveness by evaluating the contribution of individual analytes to diagnostic performance. Statistical learning and regression analysis are used to quantify the relationship between feature values and clinical labels.

- Cross-validation paradigm is employed to estimate model performance on unseen data. Simple to complex and small to large models are compared to identify the optimal balance of accuracy and generalizability.

- Machine learning techniques for commercial testing modalities include logistic regression, random forest, and gradient boosting. Threshold check for assay performance ensures minimum sensitivity and specificity before clinical deployment.

- Desired minimum accuracy is 80%, and desired minimum AUC is 0.85. Subset selection of assays is performed based on cost, throughput, and performance to optimize the diagnostic panel.

- Machine learning techniques for data processing include dimensionality reduction, feature scaling, and outlier detection. Dimension reduction methods include principal component analysis and t-distributed stochastic neighbor embedding.

- Logistic regression and other machine learning methods are used for classification. Supervised machine learning methods require labeled training data; unsupervised methods identify patterns without labels.

- Training samples and known labels are used to teach the model to distinguish cancer from non-cancer states. Optimization of model parameters is performed using grid search or Bayesian optimization.

- Use of machine learning models includes diagnosis, prognosis, recurrence prediction, and treatment stratification.

### V. SELECTION OF INPUT FEATURES

- Feature space generation involves defining the set of measurable variables derived from biological assays. Example features include read counts per gene, methylation beta values, protein concentration, glycan abundance, and fragment length.

- Genetic sequence features include copy number variation, mutation burden, and nucleosome positioning. Methylation status is quantified as the proportion of methylated cytosines at CpG sites.

- Feature selection identifies invariant features that do not vary between classes and varying features that distinguish cancer from non-cancer. Read counts are analyzed for differential abundance using statistical tests.

- Read counts are compared between cancer and non-cancer samples using Wilcoxon rank-sum or DESeq2. Statistical metrics include fold change, p-value, and false discovery rate.

- Features for training are selected based on significance, biological relevance, and model performance. A feature vector is created by concatenating selected features into a single numerical array.

- Indices are associated with feature vector elements to maintain feature identity. A matrix is stored at each index to represent multiple samples. Summary statistics are generated for each feature, including mean, median, and standard deviation.

- Features are concatenated or merged to form a unified vector. Feature engineering involves creating derived features such as ratios, interactions, or polynomial expansions.

- Weights are applied to features to emphasize or de-emphasize their contribution. Weights are learned during training using optimization algorithms. Feature vector size is reduced using dimensionality reduction techniques.

### VI. USE OF MACHINE LEARNING MODEL FOR MULTI-ANALYTE ASSAYS

- A biological sample is received and separated into multiple portions for parallel analysis of nucleic acids, proteins, carbohydrates, and metabolites. Features for each assay are identified based on prior knowledge and exploratory analysis.

- Assays are performed on each portion to obtain measured values such as read counts, protein concentrations, and glycan intensities. Measured values are normalized and standardized to remove technical variation.

- A feature vector is formed by combining all normalized values into a single array. The machine learning model is loaded from memory, having been previously trained on labeled samples.

- The model is trained using training vectors derived from biological samples with known cancer status. The feature vector is input into the model, which computes a decision score.

- An output classification is obtained, indicating the likelihood of cancer. Principal component analysis is used to visualize high-dimensional data in two or three dimensions.

- Models are updated using raw features to incorporate new data and improve performance. Treatment is provided based on classification, such as referral for colonoscopy or initiation of surveillance.

### VII. CLASSIFIER GENERATION

- Informative features correlating with cancer status are identified using statistical tests and machine learning importance scores. Features are sorted by degree of correlation with disease class.

- Correlation strength is determined using Pearson or Spearman coefficients. Machine learning techniques such as random forest and gradient boosting are used to rank feature importance.

- Class distinction is defined as the separation between cancer and non-cancer samples. Disease class distinction includes early-stage versus late-stage, or tumor type classification.

- Examples of cancer types include colorectal, pancreatic, ovarian, and lung cancer. Unknown class is ascertained by comparing the feature vector to reference models.

- Sample is classified into disease class using a decision threshold. A classifier is created for distinguishing individuals with and without cancer.

- The classifier is integrated into the machine learning model as a decision boundary. Feature vector is generated from measured values using standardized protocols.

- Machine learning model is trained using training vectors derived from labeled samples. Model is loaded into computer memory for diagnostic use.

- System for classifying subjects includes a classification circuit comprising processor, memory, and software. Components of classification system include receiver, feature module, analysis module, and output module.

- Types of machine learning classifiers include logistic regression, support vector machine, random forest, and neural network. Threshold of linear classifier is optimized to maximize sensitivity at fixed specificity.

- Multi-analyte assay data is normalized using z-score or quantile normalization. Linear classifier is used for diagnostic or prognostic call by computing a weighted sum of features.

- Data space is split into two disjoint halves using a threshold value. Biomarker profile is evaluated using linear classifier by comparing decision score to pre-defined cut-off.

- Cut-off threshold is interpreted as responsiveness or resistance to therapy. Weights and cut-off threshold are derived from training data using optimization algorithms.

- Partial Least Squares Discriminant Analysis is used to maximize separation between classes. Quantitative assay data is converted into prognosis by mapping decision score to survival probability.

- Methods for performing classification include logistic regression, support vector machine, and ensemble methods. Prediction method is trained using training data and optimized for performance.

- Transformation or pre-processing steps include normalization, log-transformation, and imputation. Weighted sum of pre-processed feature values is computed.

- Weighted sum is compared to threshold value to make classification from measured values.

### VIII. CANCER DIAGNOSIS AND DETECTION

- Cancer diagnosis and detection are achieved by applying a trained machine learning algorithm to a feature vector derived from a biological sample. Predictive analytics using AI-based approaches enable classification without prior knowledge of tumor genotype.

- Prediction algorithm is applied to generate a likelihood of cancer based on the aggregate pattern of analyte measurements. Machine learning predictor is trained using datasets derived from biological samples of cancer and non-cancer subjects.

- Training datasets are generated by measuring multiple analytes in each sample and assigning clinical labels. Features and labels are defined as input variables and ground truth outcomes.

- Characteristics of features include quantitative, continuous, and multi-dimensional nature. Labels are binary (cancer/non-cancer) or multi-class (stage, type).

- Training sets are selected by random sampling or proportionate sampling to ensure representation across age, sex, and ethnicity. Training sets are balanced across data to prevent bias.

- Machine learning predictor is trained until accuracy conditions are met, such as AUC > 0.85 and sensitivity > 80% at 85% specificity. Diagnostic accuracy measures include AUC, sensitivity, specificity, and positive predictive value.

- Method for identifying cancer in a subject involves providing a biological sample comprising cell-free nucleic acid molecules. Sequencing is performed to generate sequencing reads.

- Sequencing reads are aligned to a reference genome using alignment software. Quantitative measure of sequencing reads is generated by counting fragments per gene or per genomic bin.

- Trained algorithm is applied to generate likelihood of cancer. Predetermined conditions for accuracy include minimum read depth, sample purity, and batch consistency.

- Examples of predetermined conditions include minimum 5 million reads per sample and absence of hemolysis. Monitoring progression of disease is performed by repeated sampling and reclassification.

- Tissue-of-origin of cancer is determined by comparing methylation and fragmentation patterns to reference profiles. Tumor burden is estimated using machine learning models trained on IchorCNA or similar algorithms.

- Treatment responsiveness is introduced as a predictive output. Predictive classifiers for treatment responsiveness are trained on samples from responders and non-responders.

- Drug target of a condition is determined by correlating feature patterns with known drug mechanisms. Efficacy of a drug is determined by comparing pre- and post-treatment feature vectors.

- Sample is classified into a class of disease based on molecular signature. Individual is determined to belong to a phenotypic class based on feature profile.

- Biomarkers for predicting prognosis are identified by correlating feature values with survival time. Population is classified based on treatment responsiveness to enable stratified therapy.

- Chemotherapeutic agents include platinum-based drugs, taxanes, and antimetabolites. Examples of treatments for which population may be stratified include immune checkpoint inhibitors and PARP inhibitors.

### IX. INDICATIONS

- Biological condition refers to any physiological state detectable through circulating analytes, including cancer, inflammation, metabolic disorder, or autoimmune disease.

- Examples of biological conditions include colorectal cancer, pancreatic ductal adenocarcinoma, ovarian cancer, inflammatory bowel disease, type 2 diabetes, and neurodegenerative disorders.

- Unknown biological condition is motivated by the ability of machine learning to detect novel disease signatures without prior knowledge. Colon cancer is introduced as a primary indication, with stages I, II, III, and IV defined by TNM classification.

- Examples of colon cancer stages include Stage I (T1-2 N0 M0), Stage II (T3-4 N0 M0), Stage III (any T N1-2 M0), and Stage IV (any T any N M1).

- Conditions that can be inferred include precancerous lesions, metastatic spread, and recurrence. Examples of cancers include breast, lung, liver, and gastric cancer.

- Examples of gut-associated diseases include Crohn’s disease, ulcerative colitis, and celiac disease. Examples of immune-mediated inflammatory diseases include rheumatoid arthritis and systemic lupus erythematosus.

- Examples of neurological diseases include Alzheimer’s disease, Parkinson’s disease, and multiple sclerosis. Examples of kidney diseases include chronic kidney disease and renal cell carcinoma.

- Examples of prenatal diseases include preeclampsia and fetal growth restriction. Examples of metabolic diseases include obesity, insulin resistance, and non-alcoholic fatty liver disease.

- Diagnosis of cancer is performed by comparing feature vector to trained model. Examples of cancers that can be inferred include colorectal, pancreatic, ovarian, and lung cancer.

- Examples of gut-associated diseases that can be inferred include inflammatory bowel disease and diverticulitis. Examples of immune-mediated inflammatory diseases that can be inferred include lupus and vasculitis.

- Examples of neurological diseases that can be inferred include Alzheimer’s and ALS. Examples of kidney diseases that can be inferred include renal cell carcinoma and glomerulonephritis.

- Examples of prenatal diseases that can be inferred include preeclampsia and intrauterine growth restriction. Examples of metabolic diseases that can be inferred include prediabetes and metabolic syndrome.

- Specific details of particular examples are combined to create comprehensive diagnostic profiles. References to patents and publications are incorporated by reference.

- Scope of invention includes all methods, systems, and compositions for cancer detection using multi-analyte machine learning. Modifications and variations are contemplated within the spirit of the invention.

- Non-limiting examples include use in screening, monitoring, and prognosis. Incorporation of patents and publications is made to support technical details.

- Indications are defined as the clinical applications for which the method is intended. Machine learning techniques are applied to detect, classify, and monitor disease.

- Threshold check is performed to ensure diagnostic performance meets regulatory standards. Assay engineering procedure involves iterative selection of analytes and model refinement.

- Hierarchy of samples is motivated by the need to validate performance across diverse populations. Multi-analyte approach integrates nucleic acids, proteins, and metabolites.

- Sample collection is performed using standardized protocols. Sample splitting enables parallel analysis of multiple analyte classes.

- Molecule analysis is performed using high-throughput assays. Assay results analysis involves feature extraction and machine learning classification.

- Iterative flow is introduced, comprising initialization, cohort design, sample acquisition, assay performance, data transmission, filtering, feature extraction, cost/loss selection, model selection, feature selection, training, assessment, and feedback.

- Initialization phase involves defining clinical question and selecting sample cohort. Cohort design includes inclusion and exclusion criteria.

- Sample acquisition is performed under ethical approval. Initial assay performance is evaluated using pilot data.

- Data transmission occurs via secure network. Data filter module removes low-quality samples.

- Feature extraction generates numerical representations of analytes. Cost/loss selection balances performance with economic feasibility.

- Model selection chooses optimal algorithm and parameters. Feature selection identifies most informative analytes.

- Training module optimizes model weights. Assessment module evaluates performance on held-out data.

- Final assay is validated in independent cohort. Feedback loop incorporates new data to update model.

- Assay identification and sample identification are performed using barcodes and metadata. Iterative process continues until performance stabilizes.

- Optional modules include demographic adjustment, batch correction, and longitudinal tracking.

- Conclusion of indications is that the method is applicable to a broad range of epithelial cancers and related conditions.

- Multi-analyte assay design is optimized through iterative refinement. Overall process flow for designing multi-analyte assay involves receiving training samples with multiple classes of molecules.

- Features for each assay and training sample are identified. Sets of measured values are obtained for each assay and training sample.

- Sets of measured values are analyzed to obtain training vectors. Training vectors are operated on using machine learning model.

- Output labels are compared to known labels of training samples. Iterative search is performed for optimal parameters of machine learning model.

- Parameters of machine learning model and set of features are provided as output. Method for identifying cancer in a subject involves providing biological sample comprising cell-free nucleic acid molecules.

- Cell-free nucleic acid molecules are sequenced to generate sequencing reads. Sequencing reads are aligned to reference genome.

- Quantitative measure of sequencing reads is generated at genomic regions. Trained algorithm is applied to generate likelihood of subject having cancer.

- Results for different analytes and corresponding best performing model are analyzed. Results of different models with different dimensional reduction are compared.

- Feature column corresponding to different combinations of analytes is evaluated. Five-fold cross-validation is performed to obtain AUC information.

- Classification performance for different analytes is shown. Individual assays for classification of biological samples are investigated.

- Blood sample is separated into different portions for multiple assays. Classes of molecules including cell-free DNA, cell-free miRNA, and circulating proteins are analyzed.

- Low-coverage whole-genome sequencing and whole-genome bisulfite sequencing are performed on cell-free DNA. Cell-free microRNA is assessed by small-RNA sequencing.

- Levels of circulating proteins are measured by quantitative immunoassay. Sequenced reads are aligned to human reference genome.

- Reads are analyzed to produce vectors per sample. Measured values are filtered to identify significant differences.

- PCA analysis is performed for each analyte. Machine learning model is applied to classification.

- cf-DNA low coverage whole genome sequencing is performed. Sequence reads are counted for each annotated region.

- Read counts are normalized in various ways. Distribution of high tumor fraction samples across clinical stage is shown.

- CNV plots for individuals with high tumor fraction are displayed. Methylation analysis is performed using differentially methylated regions for CpG sites.

- CpG methylation analysis at LINE-1 sites is shown. Micro-RNA analysis is performed using expression data as features.

- cf-miRNA sequencing analysis is performed. Micro-RNAs are ranked by expression. cf-miRNA profiles in individuals with CRC are described.

- Micro-RNAs are motivated as potential CRC biomarkers. Results for different analytes and corresponding best performing model are summarized.

- Results of different models with different dimensional reduction are analyzed. Method for identifying cancer in a subject is summarized.

- Protein data is introduced. Protein data is normalized using spike-in controls. Standard curve is generated for concentration calculation.

- Concentration relationship is calculated from calibration curve. Protein biomarker distribution is shown. Significantly different levels are identified using statistical tests.

- Protein measurements are compared between groups. Distinction among ANOVA plots is observed. Principal component analysis is performed.

- Protein concentrations are vectorized. Proteins with most variation are identified. PCA is performed on cell-free DNA.

- Genes with most variance are identified. PCA output is shown. Distance between high and low tumor fraction is separated.

- Samples are classified using machine learning. Differentiation between classes is maximized using dimensionality reduction.

- Measured values are filtered to remove noise. Hi-C-like structure is identified. Genome sequence is segmented.

- Correlation between bins is calculated. Heatmap is generated. cfDNA-specific co-releasing patterns are identified.

- Three-dimensional proximity of chromatin is inferred. Genome-wide map is generated. Sample collection and preprocessing are described.

- Tissue-of-origin analysis is introduced. Compartment of cfHi-C data is modeled. Genomic regions are filtered.

- Eigenvalues are transformed. Constrained optimization problem is solved. Tumor fraction is defined.

- ichorCNA analysis is performed. Sequencing protocol is described. Normalized fragmentation score is calculated.

- Pearson correlation coefficient is calculated. Hi-C and cfHi-C are compared. Degree of similarity is quantified.

- Compartment A/B is called. Application is expanded to single-sample level. Kolmogorov-Smirnov test is applied.

- Internal library preparation bias is ruled out. Technical bias is ruled out. LOWESS method is applied.

- Genomic DNA is used as negative control. GBM regression tree is applied. Effect of G+C % and mappability is tested.

- Effect of bin size is tested. Effect of sequencing depth is tested. Data is analyzed at different sample sizes.

- Data is analyzed at different pathological conditions. Principal component analysis is applied. Canonical correlation analysis is performed.

- Eigenvalue is correlated with DNase-seq signal. Reference Hi-C panel is generated. Cell-specific correlation patterns are determined.

- Artifacts during library preparation are ruled out. Accuracy of approach is quantified. Tumor fraction is compared with ichorCNA.

- Hypothesis is tested at single-sample level. Detection of cancer using artificial intelligence is described.

- Human genome regions are annotated. Feature set is generated from annotated regions. Feature set is preprocessed.

- Sex chromosomes are removed. Poor-quality genomic bins are removed. Features are normalized for length.

- Depth normalization is performed. GC correction is applied. Cross-validation procedure is described.

- k-batch validation is motivated. k-batch validation is described. Balanced k-batch validation is described.

- Ordered k-batch validation is described. Training schemas are illustrated. k-batch with institutional downsampling is applied.

- Model training is performed. Data is transformed. Data is standardized. Dimensionality is reduced.

- Classifier hyperparameters are optimized. Performance metrics are reported. Bootstrapping is performed.

- Important features are identified. Feature distributions are analyzed. Population demographics are described.

- k-fold cross-validation is evaluated. k-batch cross-validation is evaluated. Balanced k-batch cross-validation is evaluated.

- Ordered k-batch cross-validation is evaluated. Performance is analyzed by population. Performance is analyzed by CRC stage.

- Performance is analyzed by tumor fraction. Performance is analyzed by age. Performance is analyzed by gender.

- Highly important features are identified. Feature significance is analyzed. Copy number distributions are described.

- Highly important features are used for diagnostic call. Performance is evaluated on other cancer types.

- Classification framework is described. Performance is analyzed on smaller datasets. Results are discussed.

- Importance of controlling for confounding factors is discussed. Experimental design is discussed. Computational approaches are discussed.

- cfDNA count-profile representation is discussed. Tumor fraction and clinical cancer stage are discussed. Signals in the models are discussed.

- Sequencing depth is discussed. Sample collection is described. Cell-free DNA extraction is described. Sequencing is performed.

- Reads aligning to annotated protein-coding genes are extracted. Read counts are normalized. Machine learning models are trained.

- Training schemas are illustrated. Classification performance is shown. Threshold for sensitivity is defined.

- Batch-to-batch technical variability is evaluated. Institution-specific differences are evaluated.

- Prototype blood-based CRC screening test is introduced. Gene expression prediction model is introduced.

- Methods for generating predictions are described. De-identified plasma samples are obtained. Plasma samples are separated based on CRC stage.

- Prediction model is trained. V-plots are derived. Footprinting is performed.

- Average V-plot of an expressed gene is shown. Wavelet compression and smoothing are applied.

- Logistic regression coefficients are learned. Presence or absence of accessible chromatin is measured.

- Classification accuracy is evaluated. CNV-based tumor fraction estimation is augmented.

- Computer system is described. Subsystems are utilized. Subsystems are connected via system bus.

- Control logic is implemented. Software components are encoded. Software components are transmitted.