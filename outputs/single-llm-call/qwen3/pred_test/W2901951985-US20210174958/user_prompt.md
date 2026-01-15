Here is the outline of the desired patent application.
Per bullet point, write roughly 800 words.

Example outline (bullet points are the lines starting with '- '):
## DESCRIPTION OF THE INVENTION
- describe discovery of ODAM protein in human epithelial cancers
- describe method for aiding in diagnosis and management of cancer
- describe specific embodiments of the invention
- describe methods for determining presence of ODAM or anti-ODAM antibodies

In the example above, each line beginning with '- ' is a bullet point.

```md
# DESCRIPTION

## BACKGROUND

- motivate cancer screening
- describe limitations of current screening methods
- highlight promise of blood-based tests
- discuss challenges of analyzing biological analytes
- introduce machine learning in cancer diagnostics
- state need for new methods

## BRIEF SUMMARY

- introduce machine learning approach
- describe focus on non-cellular portion of circulation
- summarize method of using classifier
- describe assaying multiple classes of molecules
- identify features for machine learning model
- prepare feature vector
- load machine learning model
- input feature vector into model
- obtain output classification
- describe classes of molecules
- provide examples of nucleic acids
- provide examples of polyamino acids
- provide examples of carbohydrates
- provide examples of metabolites
- describe plurality of assays
- provide examples of assays
- describe classifying biological sample
- provide examples of machine learning algorithms
- describe specified property
- provide examples of clinically-diagnosed disorders
- describe responsiveness to treatment
- describe continuous measurement of patient trait
- introduce system for performing classifications
- describe receiver
- describe feature module
- describe analysis module
- describe labeling module
- describe comparator module
- describe training module
- describe output module
- introduce system for classifying subjects
- describe classification circuit
- describe non-transitory computer-readable medium

## TERMS

- define "a", "an", "the"
- define "or"
- define "based on"
- define "area under the curve" (AUC)
- describe ROC curves
- define "cancer" and "cancerous"
- define "cancer-free"
- define "genetic variant" (or "variant")
- define "germline variant"
- define "input features" (or "features")
- define "machine learning model" (or "model")
- describe model training
- define "marker" or "marker proteins"
- describe marker detection
- define "non-cancerous tissue"
- define "normal tissue" or "healthy tissue"
- define "polynucleotides", "nucleotide", "nucleic acid", and "oligonucleotides"
- describe polynucleotide structure
- define "polypeptide" or "protein" or "peptide"
- describe protein modifications
- define "prediction"
- describe predictive methods
- define "prognosis"
- define "specificity"
- define "sensitivity"
- define "structural variation" (SV)
- define "subject"
- describe subject characteristics
- define "training sample"
- describe training vector
- define "tumor", "neoplasia", "malignancy" or "cancer"
- define "tumor burden"
- describe nucleic acid sample
- define "barcode"
- describe barcode sequence
- describe tagmentation or ligation reaction
- describe nucleic acid amplification
- describe amplified product

## DETAILED DESCRIPTION

- introduce medical diagnostic methods
- describe machine learning approaches
- highlight advantages over other methods
- motivate non-cellular portion of immune system
- summarize applications of methods

### I. CIRCULATING ANALYTES AND CELLULAR DECONSTRUCTION WITH BIOLOGICAL ASSAYS

- introduce importance of cost-effective assays
- define analytes
- describe DNA analytes
- specify types of DNA analytes
- describe RNA analytes
- specify types of RNA analytes
- describe polyamino acid analytes
- specify types of polyamino acid analytes
- describe other analytes
- specify types of other analytes
- motivate combination of analytes
- describe selection of analyte combinations
- provide examples of analyte combinations
- conclude importance of analyte selection

### II. SAMPLE PREPARATION

- obtain biological sample
- process sample to purify nucleic acid molecule
- separate analytes from sample
- remove higher molecular weight nucleic acid molecules
- modify nucleic acid molecule
- oxidize nucleic acid molecule
- tag or barcode nucleic acid molecule
- partition sample
- separate cellular DNA from cfDNA
- detect cellular components
- quantify nucleic acid molecules
- obtain blood samples from healthy and cancer individuals
- detect presence of AA and CRC
- differentiate between stages and sizes of cancer
- prepare library for sequencing
- add adapter sequence to nucleic acid molecule
- incorporate molecular barcode
- generate sequencing library
- treat nucleic acid molecule for methylation analysis
- deaminate unmethylated cytosine bases
- convert 5hmC to 5-formylcytosine and 5-carboxylcytosine
- sequence nucleic acid molecule
- prepare sequencing library
- amplify nucleic acid molecule
- perform targeted sequencing
- perform whole-genome sequencing
- prepare biological information
- prepare sequencing information
- perform assays on biological sample
- select assays based on machine learning model
- perform biological assays on different portions of sample
- generate feature data for machine learning analysis
- integrate assays and machine learning model
- introduce sample preparation
- motivate copy number variation
- describe copy number variation detection
- describe genome-wide detection of copy number alterations
- describe chromosomal instability analysis
- describe Length Mixture Model and Fragment Endpoint Analysis
- describe manual inspection of large-scale CNV
- motivate changes in gene expression
- describe microarray analysis
- describe metrics of cfDNA concentration
- introduce somatic mutation analysis
- describe low-coverage whole genome sequencing
- describe deep WGS and targeted sequencing
- describe somatic mutation analysis features
- introduce transcription factor profiling
- describe inference of transcription factor binding
- describe nucleosome signatures at Transcription Factor Binding Sites
- describe shallow WGS data profiles
- describe transcription factor binding site plasticity
- describe cfDNA fragmentation patterns
- describe hematopoietic transcription factor-nucleosome footprints
- describe curated list of transcription factor binding sites
- describe accessibility score and z-score statistics
- introduce method for diagnosing a disease
- describe generating a coverage pattern for a transcription factor
- describe processing the coverage pattern
- describe comparing the signal to a reference signal
- describe diagnosing the disease
- introduce inferring chromosome structure/chromatin state
- describe assays for inferring three-dimensional structure of a genome
- describe predicting chromatin state of genes
- describe probabilistic graphical model
- describe expression of genes controlled by access of cellular machinery
- introduce tissue of origin assay
- motivate cell-type-of-origin inference
- describe genetic features for cell-type-of-origin inference
- prepare reference population values
- prepare sample values
- perform matrix multiplication and parameter optimization
- estimate cell-type proportion
- determine type and proportion of cell types
- introduce method of processing a sample
- provide sequencing information
- prepare first array of values
- prepare second array of values
- prepare third array of values
- introduce methylation sequencing
- describe enzymatic methyl sequencing
- perform bisulfite conversion
- introduce whole genome bisulfite sequencing
- describe modification of nucleic acid molecule
- introduce methylation analysis metrics
- introduce machine learning approach for nucleosome positioning
- introduce method for determining genetic sequence feature
- introduce method for determining genetic sequence feature with optional enrichment
- introduce differentially methylated regions analysis
- introduce haplotype block assay
- introduce cfRNA assays
- describe RNA sequencing and alignment
- count and normalize RNA fragments
- introduce sample preparation
- aggregate reads for microRNA detection
- describe direct detection methods
- outline hybridization-based RNA assays
- detail in situ hybridization protocol
- specify probe requirements
- describe PCR reaction
- outline quantitative PCR methods
- describe fluorogenic quantitative PCR
- list other suitable amplification methods
- specify RNA markers associated with cancer
- introduce poly-amino acid and autoantibody assays
- describe protein assays using immunoassay or mass spectrometry
- outline immunoassay methods
- describe protein data normalization
- list cancer-associated peptide and protein sequences
- specify cancer-associated peptide or protein markers
- describe autoantibody detection
- outline immunosorbent assay methods
- describe protein microarrays
- specify metrics for autoantibody assay
- associate autoantibody markers with cancer subtypes or stages
- specify tumor-associated antigens
- describe ZNF700 as a capture antigen
- describe anti-p53 antibody assay
- introduce carbohydrate assays
- describe methods for measuring carbohydrates
- specify metrics from carbohydrate assays

### III. EXAMPLE SYSTEMS

- introduce system architecture
- describe data analysis in measurement devices
- outline software code execution on computing hardware
- define modules and devices/computers
- describe data receiving module
- outline data pre-processing module operations
- describe data analysis module for genomic data
- outline data interpretation module methods
- describe machine learning model implementation
- outline data visualization module methods
- describe computer systems for implementing methods
- introduce computational analysis on nucleic acid sequencing data
- describe variant identification using probabilistic modeling
- outline statistical modeling methods
- describe mechanistic modeling methods
- outline network modeling methods
- describe statistical inferences methods
- outline non-limiting examples of analysis methods
- describe germline variation and somatic mutation
- outline natural or normal variations
- describe acquired or abnormal variations
- outline distinguishing between germline variants
- describe using identified variants for healthcare improvement
- introduce system 100 for performing methods
- describe computer system 101 components
- outline measurement devices 151, 152, or 153
- describe computer system 101 operations
- outline network 130 for distributed computing
- describe cloud computing platforms
- outline CPU 105 execution of machine-readable instructions
- describe storage unit 115 for storing files and data

### IV. MACHINE LEARNING TOOLS

- introduce machine learning for assay effectiveness assessment
- describe statistical learning and regression analysis
- outline cross-validation paradigm
- describe simple to complex and small to large models
- outline machine learning techniques for commercial testing modalities
- describe threshold check for assay performance
- outline desired minimum accuracy and AUC
- describe subset selection of assays based on cost and performance
- outline machine learning techniques for data processing
- describe dimension reduction methods
- outline logistic regression and other machine learning methods
- describe supervised and unsupervised machine learning methods
- outline training samples and known labels
- describe optimization of model parameters
- outline use of machine learning models for various purposes

### V. SELECTION OF INPUT FEATURES

- describe feature space generation
- list example features
- explain genetic sequence features
- describe methylation status
- describe feature selection
- identify invariant features
- identify varying features
- analyze read counts
- compare read counts
- use statistical metrics
- select features for training
- create feature vector
- associate indices with feature vector
- store matrix at index
- generate summary statistics
- concatenate features
- merge features
- engineer features
- apply weights to features
- learn weights during training
- reduce feature vector size

### VI. USE OF MACHINE LEARNING MODEL FOR MULTI-ANALYTE ASSAYS

- receive biological sample
- separate sample into portions
- identify features for each assay
- perform assays on portions
- obtain measured values
- form feature vector
- load machine learning model
- train model using training vectors
- input feature vector into model
- obtain output classification
- provide classification output
- use principal component analysis
- update models using raw features
- provide treatment based on classification

### VII. CLASSIFIER GENERATION

- identify informative features correlating with class distinction
- sort features by correlation degree
- determine correlation strength
- use machine learning techniques
- define class distinction
- specify disease class distinction
- provide examples of cancer types
- ascertain unknown class
- classify sample into disease class
- create classifier for distinguishing individuals
- integrate classifier into machine learning model
- input feature vector into machine learning model
- generate feature vector from measured values
- train machine learning model using training vectors
- load machine learning model into computer memory
- provide system for classifying subjects
- specify components of classification system
- list types of machine learning classifiers
- optimize threshold of linear classifier
- normalize multi-analyte assay data
- use linear classifier for diagnostic or prognostic call
- split data space into two disjoint halves
- define threshold value for biomarker
- evaluate biomarker profile using linear classifier
- compare decision score to pre-defined cut-off score
- interpret cut-off threshold responsiveness or resistance
- derive weights and cut-off threshold from training data
- use Partial Least Squares Discriminant Analysis (PLS-DA)
- convert quantitative assay data into prognosis
- list methods for performing classification
- train prediction method using training data
- optimize prediction method for training set
- perform transformation or pre-processing steps
- form weighted sum of pre-processed feature values
- compare weighted sum to threshold value
- make classification from measured values

### VIII. CANCER DIAGNOSIS AND DETECTION

- introduce cancer diagnosis and detection
- describe predictive analytics using AI-based approaches
- apply prediction algorithm to generate diagnosis
- train machine learning predictor using datasets
- generate training datasets from biological samples
- define features and labels for training datasets
- describe characteristics of features and labels
- select training sets by random sampling
- select training sets by proportionate sampling
- balance training sets across data
- train machine learning predictor until accuracy conditions met
- describe diagnostic accuracy measures
- provide method for identifying cancer in a subject
- provide biological sample comprising cell-free nucleic acid molecules
- sequence cfNA molecules to generate sequencing reads
- align sequencing reads to a reference genome
- generate quantitative measure of sequencing reads
- apply trained algorithm to generate likelihood of cancer
- describe predetermined conditions for accuracy
- provide examples of predetermined conditions
- describe monitoring progression of disease
- determine tissue-of-origin of cancer
- estimate tumor burden in subject
- introduce treatment responsiveness
- describe predictive classifiers for treatment responsiveness
- determine drug target of a condition or disease
- determine efficacy of a drug designed to treat a disease
- classify sample into a class of disease
- determine whether individual belongs to a phenotypic class
- identify biomarkers for predicting prognosis of patients
- classify population based on treatment responsiveness
- describe chemotherapeutic agents
- provide examples of treatments for which population may be stratified

### IX. INDICATIONS

- define biological condition
- specify examples of biological conditions
- describe unknown biological condition
- motivate machine learning for unknown biological condition
- introduce colon cancer
- describe stages of colon cancer
- specify examples of colon cancer stages
- introduce conditions that can be inferred
- specify examples of cancers
- specify examples of gut-associated diseases
- specify examples of immune-mediated inflammatory diseases
- specify examples of neurological diseases
- specify examples of kidney diseases
- specify examples of prenatal diseases
- specify examples of metabolic diseases
- describe diagnosis of cancer
- specify examples of cancers that can be inferred
- specify examples of gut-associated diseases that can be inferred
- specify examples of immune-mediated inflammatory diseases that can be inferred
- specify examples of neurological diseases that can be inferred
- specify examples of kidney diseases that can be inferred
- specify examples of prenatal diseases that can be inferred
- specify examples of metabolic diseases that can be inferred
- describe combining specific details of particular examples
- incorporate references by reference
- describe scope of invention
- describe modifications and variations
- describe non-limiting examples
- describe incorporation of patents and publications
- introduce indications
- describe machine learning techniques
- outline threshold check
- describe assay engineering procedure
- motivate hierarchy of samples
- describe multi-analyte approach
- illustrate sample collection
- describe sample splitting
- outline molecule analysis
- describe assay results analysis
- introduce iterative flow
- describe initialization phase
- outline cohort design
- describe sample acquisition
- outline initial assay performance
- describe data transmission
- introduce data filter module
- describe feature extraction
- outline cost/loss selection
- describe model selection
- outline feature selection
- describe training module
- introduce assessment module
- describe final assay
- outline feedback loop
- describe assay identification
- outline sample identification
- describe iterative process
- outline optional modules
- conclude indications
- introduce multi-analyte assay design
- motivate iterative process for assay selection
- describe overall process flow for designing multi-analyte assay
- receive training samples with multiple classes of molecules
- identify features for each assay and training sample
- obtain sets of measured values for each assay and training sample
- analyze sets of measured values to obtain training vectors
- operate on training vectors using machine learning model
- compare output labels to known labels of training samples
- iteratively search for optimal parameters of machine learning model
- provide parameters of machine learning model and set of features
- describe method for identifying cancer in a subject
- provide biological sample comprising cell-free nucleic acid molecules
- sequence cell-free nucleic acid molecules to generate sequencing reads
- align sequencing reads to reference genome
- generate quantitative measure of sequencing reads at genomic regions
- apply trained algorithm to generate likelihood of subject having cancer
- describe results for different analytes and corresponding best performing model
- analyze results of different models with different dimensional reduction
- describe feature column corresponding to different combinations of analytes
- perform 5× cross-validation to obtain AUC information
- show classification performance for different analytes
- analyze individual assays for classification of biological samples
- separate blood sample into different portions for multiple assays
- investigate classes of molecules including cell-free DNA, cell-free miRNA, and circulating proteins
- perform low-coverage whole-genome sequencing and whole-genome bisulfite sequencing on cell-free DNA
- assess cell-free microRNA by small-RNA sequencing
- measure levels of circulating proteins by quantitative immunoassay
- align sequenced reads to human reference genome
- analyze reads to produce vectors per sample
- filter measured values to identify significant differences
- perform PCA analysis for each analyte
- apply machine learning model to classification
- describe cf-DNA low coverage whole genome sequencing
- count sequence reads for each annotated region
- normalize read counts in various ways
- show distribution of high tumor fraction samples across clinical stage
- show CNV plots for individuals with high tumor fraction
- describe methylation analysis
- use differentially methylated regions for CpG sites
- show CpG methylation analysis at LINE-1 sites
- describe micro-RNA analysis
- use expression data for micro-RNAs as features
- show cf-miRNA sequencing analysis
- rank order micro-RNAs by expression
- describe cf-miRNA profiles in individuals with CRC
- motivate use of micro-RNAs as potential CRC biomarkers
- describe results for different analytes and corresponding best performing model
- analyze results of different models with different dimensional reduction
- summarize method for identifying cancer in a subject
- introduce protein data
- normalize protein data
- generate standard curve
- calculate concentration relationship
- show protein biomarker distribution
- identify significantly different levels
- describe protein measurements
- compare protein levels
- observe distinction among ANOVA plots
- perform principal component analysis
- vectorize protein concentrations
- identify proteins with most variation
- perform PCA on cell-free DNA
- identify genes with most variance
- show PCA output
- separate distance between high and low tumor fraction
- classify samples
- maximize differentiation between classes
- use dimensionality reduction
- filter out measured values
- identify Hi-C-like structure
- segment genome sequence
- calculate correlation between bins
- generate heatmap
- identify cfDNA-specific co-releasing patterns
- infer three-dimensional proximity of chromatin
- generate genome-wide map
- describe sample collection and preprocessing
- introduce tissue-of-origin analysis
- model compartment of cfHi-C data
- filter genomic regions
- transform eigenvalues
- solve constrained optimization problem
- define tumor fraction
- perform ichorCNA analysis
- describe sequencing protocol
- calculate normalized fragmentation score
- calculate Pearson correlation coefficient
- compare Hi-C and cfHi-C
- quantify degree of similarity
- call compartment A/B
- expand application to single-sample level
- use Kolmogorov-Smirnov test
- rule out internal library preparation bias
- rule out technical bias
- apply LOWESS method
- use genomic DNA as negative control
- apply GBM regression tree
- test effect of G+C % and mappability
- test effect of bin size
- test effect of sequencing depth
- analyze data at different sample sizes
- analyze data at different pathological conditions
- apply principal component analysis
- apply canonical correlation analysis
- correlate eigenvalue with DNase-seq signal
- generate reference Hi-C panel
- determine cell-specific correlation patterns
- rule out artifacts during library preparation
- quantify accuracy of approach
- compare tumor fraction with ichorCNA
- test hypothesis at single-sample level
- describe detection of cancer using artificial intelligence
- annotate human genome regions
- generate feature set from annotated regions
- preprocess feature set
- remove sex chromosomes
- remove poor-quality genomic bins
- normalize features for length
- perform depth normalization
- apply GC correction
- describe cross-validation procedure
- motivate k-batch validation
- describe k-batch validation
- describe balanced k-batch validation
- describe ordered k-batch validation
- illustrate training schemas
- apply k-batch with institutional downsampling
- describe model training
- transform data
- standardize data
- reduce dimensionality
- optimize classifier hyperparameters
- report performance metrics
- describe bootstrapping
- identify important features
- analyze feature distributions
- describe population demographics
- evaluate k-fold cross-validation
- evaluate k-batch cross-validation
- evaluate balanced k-batch cross-validation
- evaluate ordered k-batch cross-validation
- analyze performance by population
- analyze performance by CRC stage
- analyze performance by tumor fraction
- analyze performance by age
- analyze performance by gender
- identify highly important features
- analyze feature significance
- analyze copy number distributions
- describe use of highly important features
- evaluate performance on other cancer types
- describe classification framework
- analyze performance on smaller datasets
- describe results
- discuss importance of controlling for confounding factors
- discuss experimental design
- discuss computational approaches
- discuss cfDNA count-profile representation
- discuss tumor fraction and clinical cancer stage
- discuss signals in the models
- discuss sequencing depth
- describe sample collection
- describe cell-free DNA extraction
- describe sequencing
- extract reads aligning to annotated protein-coding genes
- normalize read counts
- train machine learning models
- illustrate training schemas
- show classification performance
- define threshold for sensitivity
- evaluate batch-to-batch technical variability
- evaluate institution specific differences
- describe prototype blood-based CRC screening test
- introduce gene expression prediction model
- describe methods for generating predictions
- obtain de-identified plasma samples
- separate plasma samples based on CRC stage information
- train prediction model
- derive V-plots
- perform footprinting
- show average V-plot of an expressed gene
- apply wavelet compression and smoothing
- learn logistic regression coefficients
- measure presence or absence of accessible chromatin
- evaluate classification accuracy
- augment CNV based tumor fraction estimation
- describe computer system
- utilize subsystems
- connect subsystems via system bus
- implement control logic
- encode software components
- transmit software components
```

You need to draft a complete patent application that strictly follows the outline's section order and headings. Do not skip any bullet points. Use formal patent language. The generated patent must not be shorter than the research paper in word count.

Here is the research paper that describes the invention:

```md
# Background

Despite the public health emphasis on population-level cancer screening in recent decades, adherence remains lower than desired [1], and cancer is often detected too late for successful treatment. For example, nearly 60% of colorectal cancer (CRC) cases, and approximately 80% of pancreatic cancer cases, are detected after regional or distant metastases [2]. Although the burden of CRC has been decreasing, CRC remains the third leading cause of cancer-related deaths in men and women in the United States [2]. Current cancer screening methods are often invasive, inconvenient, expensive, and/or have suboptimal clinical performance (i.e., sensitivity or specificity), particularly for early-stage disease and precancerous lesions [3].

Recently, blood-based screening tests for cancer have been proposed in an effort to address some of the aforementioned challenges. One key area of both academic and commercial interest is circulating cell-free DNA (cfDNA), which includes both tumor-derived DNA (so-called “circulating tumor DNA”, or ctDNA) and DNA derived from non-tumor cells, such as hematopoietic and stromal cells, to supplement or replace existing cancer screening methods.

Different screening approaches using cfDNA are being explored, and some have hypothesized that ctDNA-only based “liquid biopsies” may enable sensitive and specific early detection of cancer ([4–7]. ctDNA has unique characteristics of tumor DNA, such as cancer-associated mutations, translocations, and/or large chromosomal copy number variants (CNVs), not typically present in the cfDNA of healthy patients [8]. In addition, ctDNA fragments appear to be shorter on average than cfDNA found in healthy subjects [9]. However, others have questioned whether such an approach is feasible for routine screening, given biological (e.g., clonal hematopoiesis of indeterminate potential (CHIP)), technical (e.g., limits of detection and variable levels of tumor fraction (TF) observed in cancer patients), and practical (e.g., blood volume requirements and cost) considerations [10–12]. In patients with cancer, ctDNA generally represents a small fraction of all cfDNA, ranging from ≥5–10% in late-stage disease to ≤0.01–1.0% in early-stage disease, and even lower in premalignant conditions [13]. These limitations are particularly important in early-stage cancer when the tumor is small and the shedding of DNA into the blood may be minimal. Indeed, many previous cfDNA studies have had stage distributions meaningfully different from those seen in screening populations [14–16].

An alternative to detection based solely on ctDNA is to look more broadly at cfDNA—both tumor derived and non-tumor derived—and changes that early-stage cancer may induce in blood. There is growing evidence of interactions between cancerous cells and other cells, including fibroblasts, platelets, and immune cells, especially within the tumor microenvironment. These include findings of “tumor education”, such as changes in gene expression that may reflect interaction with a tumor and/or ingestion of tumor-related molecules [17]. For example, platelets in patients with cancer harbor different patterns of messenger RNA (mRNA) than platelets in healthy individuals [18]. There are also reports of changes in immune-cell apoptosis patterns in patients with cancer [19], suggesting global changes in hematopoietic cell populations that may reflect altered physiological states. For instance, low relative levels of circulating lymphocytes versus monocytes may be correlated with poor cancer prognosis [20]. It is possible to detect such changes in cell populations from cfDNA because cfDNA fragmentation and methylation patterns can recapitulate expected cellular epigenetic states [15, 16, 21–23].

Because it is still unknown to what extent circulating cells in patients with early-stage cancer are educated by the tumor microenvironment (i.e., how changes in cellular state are explicitly reflected in the billions of base pairs of cfDNA), the ability to identify disease-relevant patterns in cfDNA requires unbiased methods that can identify patterns in high-dimensional space. Given a large enough sample size, machine learning (ML) may provide a toolset by which to learn disease-related patterns from whole-genome signals directly from patients with and without early-stage cancer. However, the primary challenge in an assay with many measured variables is to identify relevant, low-dimensional features that generalize to the screening population [24, 25]. As a corollary, it is necessary to mitigate potential confounding variables, defined as variables that are correlated with the clinical label which in this case is the disease label. For example, batch effects or institutional processing effects can be a significant variable that correlates with non-cancerous and cancerous samples.

Here we develop and implement a computational approach for representing and learning associations between cfDNA profiles and cancer status, with a focus on the importance of accounting for known confounding variables. Using this approach, we report classification results for a large cohort of non-cancer controls and early-stage CRC patients.

# Methods

## Sample Collection

Human EDTA plasma samples were acquired from 546 patients diagnosed with CRC (Table 1). As controls, plasma samples from 271 unique patients without a current CRC diagnosis were also acquired. In total, 817 de-identified plasma samples were collected from institutions and commercial biobanks located in the United States, Germany, and Scotland. Patient age, gender, and cancer stage (where available) were obtained for each sample. Samples were included in the intended use (IU) age range analysis for CRC only if the patient’s age at time of collection was known to be between 50 and 84, inclusive. Plasma was stored at − 80 °C. Table 1Clinical characteristics and demographics of CRC patients and non-cancer controlsCRC N = 546Control N = 271Total Samples N = 817GenderFemale N (%)264 (48%)182 (67%)446 (55%)Male N (%)282 (52%)84 (31%)366 (45%)Unknown05 (2%)5 (< 1%)StageI172 (32%)N/AN/AII266 (49%)III98 (18%)IV6 (1%)Unknown4 (< 1%)Age (yrs)Median (IQR)71 (63–80)60 (53–67)68 (59–77)

## Laboratory Processing, Bioinformatics, And Featurization

Detailed descriptions of laboratory processing and sequencing, bioinformatics analysis, data preprocessing, classifier training, and validation methods (including measuring and controlling for confounding factors) are provided in the Additional file 1: Supplemental Methods.

Briefly, cfDNA was extracted from 250 μl plasma using the MagMAX cfDNA Isolation Kit (Applied Biosystems), converted into libraries using the NEBNext Ultra II DNA Library Prep Kit (New England Biolabs), and paired-end sequenced on the Illumina platform. Reads were aligned to the human genome using BWA-MEM 0.7.15 [26]; all datasets passing quality control (based on acceptable GC bias, sufficient number of reads, and no evidence of contamination or sample swaps) continued to featurization. Aligned reads were transformed into per-sample feature vectors by counting the number of fragments appearing in protein-coding genes. Features were normalized per-sample by dividing by the trimmed mean (excluding top and bottom 10% of counts) over all features and applying Loess GC bias correction [27]. Categorical features used in certain experiments (binned age, sex, and institution) were featurized using a one-hot encoding. Tumor fraction was estimated in each sample using IchorCNA [28] from read counts in 50-kilobase (kb) bins across the entire genome. Prior to the authors’ method, 50 kb-bin counts were normalized using tangent normalization with a panel of non-cancer control and process control samples.

## Model Training

ML models were trained and evaluated using cross-validation (CV) procedures as follows. Each feature, which is a preprocessed read count, was standardized by subtracting the mean and dividing by the standard deviation after large outliers were replaced with 99th percentile value. Dimension-reduction methods including principal component analysis (PCA) and truncated singular-value decomposition (SVD) were then optionally applied to the standardized data. Two classification methods were considered for training (logistic regression and support vector machine (SVM)) with hyperparameters chosen based on random search within the training data of each fold; the test folds were reserved exclusively for the average test performance estimate described in the following section. The detailed procedure used for model selection is described in the Additional file 1: Supplemental Methods. The best model was selected based on k-fold CV, and the methods were subsequently applied to other CV procedures. All methods were implemented by Scikit-learn [29].

## Validation And Confounder Control

We used CV to estimate a classification model’s performance on new, previously unseen data. Five different CV procedures were used to obtain estimates of model performance. All CV procedures shared in common the partitioning of the data into multiple independent subsets, or “folds,” with individual folds held out and used to assess the performance of models trained on the remaining data (Fig. 1a). The principal difference among CV procedures was how individual samples were partitioned into folds (Fig. 1b). The procedures included k-fold, in which samples were partitioned at random (stratified by class label of cancer or not cancer); binned-age, in which partitions were defined based on age; k-batch, in which partitions were defined by processing batch; balanced k-batch, in which partitions were defined by processing batch with additional downsampling to stratify by institutional source; and ordered k-batch, in which samples were partitioned by date of laboratory processing. Details of each method are provided in the Additional file 1: Supplemental Methods. Fig. 1Model training overview and CV procedures. a All methods were trained on k-fold, and the best performing method was chosen to train models for the other cross-validation procedures. Diagram describes individual steps in common to all methods. Models are trained on a given dataset and set of methods (i.e., dimension reduction and classification) and then evaluated, resulting in a performance estimate. b Illustration of CV procedures for k-fold, k-batch, ordered k-batch, and balanced k-batch. Each square represents a single sample, with the fill color indicating class label, the border color representing a confounding factor like institution, and the number indicating processing batch. Each column represents a possible fold constructed for the given CV procedure. The dashed line separates the test set of samples held out from the training set

Five folds (k = 5) were used for CV of all models except for binned-age (which has a fixed number of bins). Reported performance metrics are mean area under the curve (AUC), where curve is the receiver operating characteristic (ROC) curve, and mean sensitivity at 85% specificity, each calculated across all test folds.

# Results

Paired-end whole-genome sequencing (WGS) was performed on plasma cfDNA obtained from 271 non-cancer control subjects and 546 CRC patients (Table 1). The patient population was approximately equally split by gender (55% female, 45% male), and consisted of 80% early-stage (stages I and II) patients. The non-cancer control population skewed younger (median age = 60; interquartile range [IQR] = 53–67) than the cancer population (median age = 71; IQR = 63–80, p < 0.01, Mann-Whitney U-test) (Table 1).

WGS data were converted into input features for the classification model by counting the number of fragments overlapping each annotated protein-coding gene (i.e., each gene corresponded to a single bin) and then normalizing to account for feature length, mappability, read depth, and sequence-content biases. The gene-based featurization was designed to simultaneously capture both copy number changes as well as epigenetic signals reflected in cfDNA fragmentation patterns across genes [15].

## Performance Of Confounder Covariates

Before assessing classification performance, models were trained using confounding variables as inputs to validate our CV stratification methods. In k-fold CV, binned age, batch, processing date, and institution confounders achieved mean AUCs of 0.71, 0.72, 0.69, and 0.87, respectively, when tested individually as the only input features to the classification model (Table 2, Additional file 2: Figure S1). When evaluated using CV methods tailored specifically to address them, these same input features (i.e., confounder variables) had no predictive power (i.e., AUCs of approximately 0.50). These results demonstrate that binned-age, k-batch, balanced k-batch, and ordered k-batch CV effectively assess performance while controlling for their respective confounder variables (Table 2, Additional file 2: Figure S1). Table 2Performance Evaluation of Known ConfoundersConfounderk-fold CV AUC (95% CI)k-fold CV Sensitivity at 85% Specificity (95% CI)Confounder CV methodConfounder CV AUC (95% CI)Age0.71 (0.64–0.77)44% (29–57%)Binned-age0.50 (0.50–0.50)Batch0.72 (0.69–0.75)43% (31–53%)k-batch0.50 (0.50–0.50)Processing Date0.69 (0.64–0.74)38% (25–49%)Ordered k-batch0.48 (0.43–0.52)Institution0.87 (0.84–0.90)74% (72–77%)Balanced k-batch0.51 (0.28–0.74)Performance evaluation of known confounders alone to predict cancer with either k-fold or the CV procedure designed to control for the confounder. Confidence intervals are calculated from bootstrapped distributions of the metric across folds

## Performance By Cross-Validation Procedures

After initial model selection via k-fold CV performance, we additionally applied each previously introduced CV procedure to the same methods to estimate the generalizability of performance when controlling for particular confounder variables individually (Table 3). The method selected by k-fold CV performance was no dimensionality reduction and SVM classification. Evaluation by standard k-fold CV achieved a mean AUC of 0.92 (95% bootstrap confidence interval (CI) of 0.91–0.93) with a mean sensitivity of 85% (95% CI = 83–86%) at 85% nominal specificity. Using binned-age CV to control for age achieved mean AUC of 0.91 (95% CI = 0.89–0.94) with a mean sensitivity of 79% (95% CI = 73–87%) at 85% specificity. We controlled batch-to-batch technical variability using k-batch CV and process variability using ordered k-batch CV, which achieved mean AUC of 0.91 (95% CI = 0.88–0.94) and 0.90 (95% CI = 0.83–0.94) and sensitivity at 85% specificity of 85% (95% CI = 80–89%) and 73% (95% CI = 53–88%), respectively. The larger variance observed in ordered k-batch may be attributed (at least in part) to higher standard deviation in test fold sizes (80.8) when compared to standard deviation of test folds of k-batch (35.0) (Additional file 6: Table S1). Finally, we applied balanced k-batch CV to control for possible institution-specific differences in population or sample handling. Despite training on a significantly reduced dataset (average of 263. 6 samples per fold in training versus 653. 6 samples per fold with k-fold or k-batch as seen in Additional file 6: Table S1), the balanced k-batch CRC model achieved a mean AUC of 0.83 (95% CI = 0.79–0.86) with a mean sensitivity of 71% (95% CI = 63–76%) at 85% specificity (Table 3). Figure 2 shows ROC curves for each CV procedure. Table 3CRC Performance by Validation ProcedureValidationMean AUC (95% CI)Mean Sensitivity at 85% Specificity (95% CI)k-fold0.92 (0.91–0.93)85% (83–86%)Binned-age0.91 (0.89–0.94)79% (73–87%)k-batch0.91 (0.88–0.94)85% (80–89%)Ordered k-batch0.90 (0.83–0.94)73% (53–88%)Balanced k-batch0.83 (0.79–0.86)71% (63–76%)CRC performance by cross-validation procedure in 50–84 year-old patients. Confidence intervals are calculated from bootstrapped distributions of the metric across folds Fig. 2Colorectal cancer classification performance (ROC curves) by each cross-validation method. Average of all folds drawn in solid blue; random chance is represented as dashed red; ROCs for each fold drawn behind. a k-fold, b binned age, c k-batch, d ordered k-batch, and e balanced k-batch

## Detailed Comparison Of Performance By Clinical Parameters

Additionally, we investigated the sensitivity of our method, trained using each CV procedure, to relevant clinical parameters. Figure 3a illustrates sensitivity as a function of clinical stage. All validation methods achieved similar distributions of sensitivity across stages I through III, and consistently classified stage IV cancer correctly. Stage II samples, which represent the majority of our data, performed consistently well. We also evaluated age, which is a known confounder. The AUC performance increased with age in nearly all validations (Fig. 3b). Taken as a whole, the results are consistent with the general notion that cancer is an age-related disease. Performance for males and females was comparable across validation types (Fig. 3c), even in spite of the observed imbalance in non-cancer controls (Table 1). Fig. 3Classification performance for colorectal cancer within the IU age range across all validation methods. N is number of samples, [cancer, controls]. The average of all folds is represented by the colored bars; the 95% bootstrap confidence intervals are represented by the solid black lines. a Sensitivity at 85% nominal specificity by CRC stage across all CV procedures. b AUC by age bins across all CV procedures. c AUC by gender across all CV procedures. d AUC by an IchorCNA-based estimated TF across all CV procedures

Tumor fraction (TF), defined as the fraction of cfDNA originating from tumor cells, has been implicated as a critical parameter for the design of blood-based cancer screens [4, 6, 12, 30, 31]. As high-depth mutation detection information is not available in our data, we estimated TF from observed copy number variation using IchorCNA [28]. Nearly all of the non-cancer control samples (> 99%) were estimated to have a TF below 0.8% (Additional file 3: Figure S2).

Figure 3d displays our CRC model’s AUC as a function of an IchorCNA-based estimated TF. Observed performance increased with increasing TF, which is consistent with the hypothesis that an ML-based method may be able to detect tumor-derived signal; however, performance remained better than chance even in the lowest TF bin. To investigate whether the ML model may detect signal beyond tumor-derived CNVs, we used an IchorCNA-based estimated TF alone to predict cancer. This method achieved AUC of 0.67 in the IU age range, lower than results from the ML model under any analyzed CV scheme (Table 3), consistent with the possibility that the ML model used non-tumor-derived signal (i.e., beyond IchorCNA-detectable CNVs) (Additional file 4: Figure S3).

To address decreased classifier performance due to smaller sample sizes in training (i.e., balanced k-batch), the CRC dataset was downsampled. Additional file 5: Figure S4 illustrates the non-linear relationship between the total number of samples used for training and the measured sensitivity. These results suggest that the lower performance observed using balanced k-batch is explained, at least in part, by the smaller size of the training dataset.

# Discussion

Our results show promising preliminary performance for early-stage (i.e., stages I and II) CRC detection using blood. To our knowledge, this multicenter, international study represents the largest study to date using only cfDNA WGS in patients for the early detection of CRC. We have demonstrated that it is possible to take an ML-based approach to learn the relationship between a patient’s cfDNA profile and cancer diagnosis, with 85% sensitivity at 85% specificity in CRC using standard k-fold cross-validation; application of other rigorous and novel CV strategies specifically designed to control for known confounding variables yielded 71–85% sensitivity at 85% specificity.

In this work, we focused our approach on cfDNA count profiles across the whole human genome (~ 3200 Mb) at relatively low depth (~9X), as opposed to existing liquid biopsy approaches that assess small regions (< 2 Mb) of the genome at very high depth (~ 60,000X) to detect tumor-derived mutations. In particular, we applied ML methods to perform unbiased discovery of signals of varying origin that may inform on the presence of a tumor (including both tumor-derived CNVs as well as potentially non-tumor-derived signals such as changes in the epigenetic states of circulating immune cells) vis-à-vis focusing on only tumor-derived mutations. This parallels previous research in non-invasive prenatal testing (NIPT): Kim et al. demonstrated that an ML-based regression algorithm operating on genome-wide count data was able to accurately estimate fetal fraction in the cfDNA of pregnant women, without the detection of single-nucleotide polymorphism differences between mother and fetus [32]. Additionally, unlike liquid biopsy approaches using ultra-high-depth sequencing, the use of relatively low depth meaningfully decreases the cost of testing and permits the use of reasonable blood volumes, both of which will ultimately be required for population-level screening [12]. Finally, approaches focused on mutation detection alone can miss certain types of tumor-derived signals (e.g., genome-wide CNVs and epigenetic modifications), which are by definition most scarce in early-stage (i.e., non-metastatic) disease and pre-cancerous lesions, the detection of which is the goal of cancer screening programs.

While we have not yet directly determined the exact contributions to classifier performance from tumor- versus non-tumor-derived sources, several lines of evidence suggest that both may be present. First, while the observed relationship between AUC and inferred TF (Fig. 3d) indicates that at least some of the classification power is likely attributable to the ability of the model to identify samples with abundant ctDNA, the ability to correctly classify samples with lower TF and/or early-stage disease suggests that ctDNA alone cannot fully account for classification performance. Second, a CRC classification model based solely on an estimated TF (inferred from CNV calls) performs relatively poorly, with an AUC lower than all tested CV results for the ML method, suggesting that non-CNV sources may contribute to our ML-based classifier. Future research will focus on better understanding the underlying biology of the classifier, as well as assessing potential improvements in model performance from the addition of other analytes and ML method development, including confounder mitigation.

In the presence of inadequately controlled confounders, ML methods are prone to learn irrelevant associations; this poses a critical challenge for the use of ML for biomarker discovery [33–36]. Certain confounders can be mitigated “up front” through experimental design (e.g., demographic biases and institution bias) or operational quality control (e.g., identification of known parameter drift). This can help minimize the dependence between class label and any potential noise-inducing variable but incurs an additional cost in time and/or operational expense. However, perfect control of confounders at the design stage is not realistic: Some variables may be intrinsically confounding in the population of interest (e.g., cancer incidence increases with age), and there are modes of variation which may exist but which may not be known a priori and therefore mitigated post hoc (e.g., batch-to-batch variability in sequencing).

A key contribution of this work is the presentation and analysis of cross-validation techniques specifically tailored to go beyond traditional k-fold validation to measure and mitigate a number of pervasive confounding effects in biomarker discovery: k-batch and ordered k-batch for different scales of process variability in time, respectively, and balanced k-batch for institution-specific biases. We found that standard k-fold CV can have higher performance than confounder-controlled CV methods, consistent with the historical difficulty in reproducing discovery studies. We believe that explicit stratification for technical and biological confounders may be used as standard practice to better evaluate the generalizability of early discovery results.

The current study has a number of potential limitations. First, because samples were obtained retrospectively, breaks in the chain of custody may have led to sample and labelling errors, which would impede the ability of an ML method to adequately learn. Additionally, the presence of CNVs in a small number of control samples (Additional file 3: Figure S2) has been previously observed in other cohorts and may be due to malignant or benign causes [5, 14, 37]; further follow up was not possible in this cohort. Another limitation of our study is that TF was estimated using copy-number inference from moderate-coverage whole-genome sequencing, which as implemented here has a limit of detection of 0.8% for TF; by contrast, targeted mutation detection would allow more sensitive characterization of TF. Finally, cross validation is a widely used approach for assessment of classification performance and, as implemented here, can address known individual confounders. However, cross-validation procedures do not control for all confounders simultaneously, and the use of an independent test set is ultimately needed to evaluate the generalizability of these results. Prospective studies are underway to validate classifier performance in an independent cohort and verify generalization predictions from confounder-controlled CV.

Although this study focused on CRC, this study approach is directly applicable to other cancers and indeed to other pathological and physiological conditions. Our approach extracts signals from certain biological states and can apply them to better understand others; however, full development and validation of classifiers to address different clinical and non-clinical applications will require additional samples in those specific populations. Unlike targeted mutation approaches which require identification of disease-specific targets, this whole genome approach allows for the unbiased discovery of signals which are not disease-specific and could even be extended to the assessment and monitoring of non-disease states. Additionally, this approach should be able to detect unique epigenetic patterns for other diseases, thereby providing specificity by differentiating CRC from other cancers [38]. Efforts are currently underway to evaluate these hypotheses.

# Conclusions

In summary, this study presents a novel representation of cfDNA and an analysis framework that demonstrates promising initial results for the detection of early-stage CRC based on a minimally invasive blood test. Prospective validation of this approach is currently underway, as is the incorporation of other cell-free, blood-based analytes (e.g., proteins) that may contribute orthogonal signals to further improve classifier performance.

# Additional Files

##
```
