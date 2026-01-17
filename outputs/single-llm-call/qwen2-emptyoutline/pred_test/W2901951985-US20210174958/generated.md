# DESCRIPTION

## BACKGROUND

Despite significant advancements in public health initiatives aimed at promoting cancer screening, adherence rates remain suboptimal, and cancer is frequently detected at advanced stages, reducing the chances of successful treatment. For instance, nearly 60% of colorectal cancer (CRC) cases and approximately 80% of pancreatic cancer cases are identified after regional or distant metastases. While the overall burden of CRC has been declining, it remains the third leading cause of cancer-related deaths in both men and women in the United States. Traditional cancer screening methods are often invasive, inconvenient, expensive, and have suboptimal clinical performance, particularly for early-stage disease and precancerous lesions.

Recent developments in blood-based screening tests for cancer have shown promise in addressing these challenges. One notable area of interest is the use of circulating cell-free DNA (cfDNA), which includes both tumor-derived DNA (ctDNA) and DNA from non-tumor cells. ctDNA has unique characteristics, such as cancer-associated mutations, translocations, and large chromosomal copy number variants (CNVs), that are not typically present in the cfDNA of healthy individuals. However, the feasibility of using ctDNA for routine screening is limited by biological, technical, and practical considerations. For example, ctDNA generally represents a small fraction of all cfDNA, especially in early-stage disease, making its detection challenging.

An alternative approach is to examine cfDNA more broadly, including both tumor-derived and non-tumor-derived components, and to identify changes induced by early-stage cancer in the blood. There is growing evidence of interactions between cancerous cells and other cells, such as fibroblasts, platelets, and immune cells, within the tumor microenvironment. These interactions can lead to changes in gene expression and cellular state that are reflected in cfDNA. Machine learning (ML) techniques can be used to identify disease-relevant patterns in high-dimensional cfDNA data, even in the presence of confounding variables.

## BRIEF SUMMARY

The present invention relates to a method and system for detecting early-stage colorectal cancer (CRC) using circulating cell-free DNA (cfDNA) and machine learning (ML) techniques. The method involves collecting plasma samples from patients, extracting cfDNA, and converting it into feature vectors by counting the number of fragments overlapping annotated protein-coding genes. These features are then normalized and used to train ML models, such as logistic regression and support vector machines (SVMs), to classify samples as cancerous or non-cancerous. The invention also includes various cross-validation (CV) procedures to control for confounding variables, ensuring the robustness and generalizability of the model. The method has demonstrated promising performance in detecting early-stage CRC, with high sensitivity and specificity, making it a valuable tool for non-invasive cancer screening.

## TERMS

- **Circulating Cell-Free DNA (cfDNA):** DNA fragments found in the bloodstream that originate from both healthy and cancerous cells.
- **Circulating Tumor DNA (ctDNA):** A subset of cfDNA that originates from tumor cells and contains cancer-specific genetic alterations.
- **Machine Learning (ML):** A subset of artificial intelligence that involves algorithms capable of learning from and making predictions on data.
- **Cross-Validation (CV):** A statistical method used to assess the performance of a predictive model by dividing the data into training and testing sets.
- **Confounding Variables:** Factors that can influence the outcome of a study and may lead to biased results if not properly controlled.
- **Feature Vector:** A numerical representation of the input data used in ML models.
- **Principal Component Analysis (PCA):** A technique used to reduce the dimensionality of a dataset while retaining most of the variance.
- **Support Vector Machine (SVM):** A supervised learning model used for classification and regression analysis.
- **Receiver Operating Characteristic (ROC) Curve:** A graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
- **Area Under the Curve (AUC):** A measure of the performance of a binary classifier, with a higher AUC indicating better performance.

## DETAILED DESCRIPTION

### I. CIRCULATING ANALYTES AND CELLULAR DECONSTRUCTION WITH BIOLOGICAL ASSAYS

The invention leverages the analysis of circulating cell-free DNA (cfDNA) to detect early-stage colorectal cancer (CRC). cfDNA is a mixture of DNA fragments that circulate in the bloodstream and can originate from both healthy and cancerous cells. The presence of cancer can alter the composition and characteristics of cfDNA, making it a valuable biomarker for cancer detection. The method involves the following steps:

1. **Sample Collection:** Plasma samples are collected from patients suspected of having CRC and from healthy controls. The samples are stored at −80°C to preserve the integrity of the cfDNA.
2. **cfDNA Extraction:** cfDNA is extracted from the plasma using a kit such as the MagMAX cfDNA Isolation Kit. The extracted cfDNA is then converted into libraries using a library preparation kit, such as the NEBNext Ultra II DNA Library Prep Kit.
3. **Sequencing:** The cfDNA libraries are sequenced using a next-generation sequencing (NGS) platform, such as the Illumina platform, to generate paired-end reads.
4. **Bioinformatics Analysis:** The sequencing reads are aligned to the human genome using tools like BWA-MEM. The aligned reads are then transformed into feature vectors by counting the number of fragments overlapping annotated protein-coding genes. These features are normalized to account for various biases, such as feature length, mappability, read depth, and sequence content.
5. **Machine Learning Model Training:** The normalized feature vectors are used to train ML models, such as logistic regression and support vector machines (SVMs), to classify samples as cancerous or non-cancerous. Various dimensionality reduction techniques, such as principal component analysis (PCA) and truncated singular-value decomposition (SVD), can be applied to the data before training the models.
6. **Cross-Validation:** To ensure the robustness and generalizability of the model, various CV procedures are employed to control for confounding variables. These procedures include k-fold CV, binned-age CV, k-batch CV, ordered k-batch CV, and balanced k-batch CV.

### II. SAMPLE PREPARATION

The sample preparation process is crucial for the accurate and reliable detection of early-stage CRC using cfDNA. The following steps are involved:

1. **Plasma Collection:** Human EDTA plasma samples are collected from patients diagnosed with CRC and from healthy controls. The samples are acquired from various institutions and commercial biobanks located in the United States, Germany, and Scotland.
2. **Storage:** The plasma samples are stored at −80°C to prevent degradation of the cfDNA.
3. **cfDNA Extraction:** cfDNA is extracted from 250 μL of plasma using the MagMAX cfDNA Isolation Kit. The extraction process ensures the recovery of high-quality cfDNA suitable for downstream analysis.
4. **Library Preparation:** The extracted cfDNA is converted into libraries using the NEBNext Ultra II DNA Library Prep Kit. This step involves the ligation of adaptors to the cfDNA fragments and PCR amplification to generate sufficient material for sequencing.
5. **Sequencing:** The prepared libraries are sequenced on the Illumina platform using paired-end sequencing. The sequencing depth is optimized to balance the need for high coverage with the cost and practicality of the test.

### III. EXAMPLE SYSTEMS

The invention can be implemented using various systems and platforms to facilitate the detection of early-stage CRC. Example systems include:

1. **Laboratory Setup:** A well-equipped laboratory with the necessary instruments for sample collection, cfDNA extraction, library preparation, and sequencing. Key equipment includes centrifuges, thermal cyclers, and NGS platforms.
2. **Data Analysis Pipeline:** A robust bioinformatics pipeline for processing the sequencing data, including alignment, feature extraction, normalization, and ML model training. The pipeline can be implemented using open-source tools and custom scripts.
3. **Cloud Computing Platform:** A cloud-based platform for storing and analyzing large datasets. The platform provides scalable computing resources and secure data storage, enabling efficient processing of the data.
4. **User Interface:** A user-friendly interface for clinicians and researchers to upload samples, monitor the progress of the analysis, and interpret the results. The interface can be web-based or integrated into existing electronic health record (EHR) systems.

### IV. MACHINE LEARNING TOOLS

The invention utilizes advanced machine learning (ML) tools to analyze the cfDNA data and classify samples as cancerous or non-cancerous. The following ML techniques are employed:

1. **Feature Engineering:** The raw sequencing data is processed to extract meaningful features. This involves counting the number of fragments overlapping annotated protein-coding genes and normalizing the counts to account for various biases.
2. **Dimensionality Reduction:** Techniques such as PCA and truncated SVD are applied to reduce the dimensionality of the feature space while retaining the most informative features.
3. **Model Selection:** Multiple ML models, including logistic regression and SVMs, are trained and evaluated using cross-validation (CV) procedures. Hyperparameters are tuned using random search within the training data of each fold.
4. **Cross-Validation:** Various CV procedures are used to control for confounding variables and ensure the robustness of the model. These procedures include k-fold CV, binned-age CV, k-batch CV, ordered k-batch CV, and balanced k-batch CV.

### V. SELECTION OF INPUT FEATURES

The selection of input features is a critical step in the ML pipeline. The following criteria are used to select relevant features:

1. **Gene-Based Featurization:** The number of fragments overlapping annotated protein-coding genes is counted and normalized to account for feature length, mappability, read depth, and sequence content.
2. **Categorical Features:** Categorical features such as binned age, sex, and institution are featurized using one-hot encoding.
3. **Tumor Fraction Estimation:** The tumor fraction (TF) is estimated using IchorCNA from read counts in 50-kilobase (kb) bins across the entire genome. The TF is used as an additional feature in the ML model.
4. **Feature Normalization:** All features are standardized by subtracting the mean and dividing by the standard deviation. Large outliers are replaced with the 99th percentile value to prevent skewing the data.

### VI. USE OF MACHINE LEARNING MODEL FOR MULTI-ANALYTE ASSAYS

The ML model developed in this invention can be extended to multi-analyte assays, combining cfDNA analysis with other blood-based biomarkers to improve the accuracy and specificity of cancer detection. The following multi-analyte approaches are considered:

1. **Protein Biomarkers:** Blood-based protein biomarkers can be measured using immunoassays and combined with cfDNA data to enhance the classification performance.
2. **Epigenetic Markers:** Epigenetic modifications, such as DNA methylation patterns, can be analyzed to provide additional insights into the cellular state and improve the detection of early-stage cancer.
3. **Metabolomic Profiling:** Metabolomic profiling can be used to identify changes in metabolic pathways associated with cancer and integrate this information with cfDNA data.
4. **Multi-Omics Integration:** A comprehensive multi-omics approach can be employed, integrating data from genomics, transcriptomics, proteomics, and metabolomics to develop a more robust and accurate cancer detection model.

### VII. CLASSIFIER GENERATION

The generation of the classifier involves the following steps:

1. **Training Data Preparation:** The normalized feature vectors are split into training and testing sets. The training set is used to train the ML models, while the testing set is used to evaluate the performance of the models.
2. **Model Training:** Multiple ML models, such as logistic regression and SVMs, are trained using the training data. Hyperparameters are optimized using random search within the training data of each fold.
3. **Cross-Validation:** Various CV procedures are used to control for confounding variables and ensure the robustness of the model. The performance of the models is evaluated using metrics such as AUC and sensitivity at 85% specificity.
4. **Model Selection:** The best-performing model is selected based on the results of the CV procedures. The selected model is then validated using an independent test set to confirm its generalizability.

### VIII. CANCER DIAGNOSIS AND DETECTION

The invention provides a method for the early diagnosis and detection of colorectal cancer (CRC) using circulating cell-free DNA (cfDNA) and machine learning (ML) techniques. The method has demonstrated promising performance in detecting early-stage CRC, with high sensitivity and specificity. The following results highlight the effectiveness of the method:

1. **Performance Metrics:** The method achieved a mean AUC of 0.92 with a mean sensitivity of 85% at 85% specificity using standard k-fold cross-validation. When controlling for confounding variables using binned-age, k-batch, ordered k-batch, and balanced k-batch CV, the method maintained high performance, with AUCs ranging from 0.83 to 0.91.
2. **Clinical Parameters:** The method performed well across different clinical parameters, including cancer stage, age, and gender. The performance increased with age, consistent with the age-related nature of cancer. The method also showed good performance in classifying stage II samples, which represent the majority of the data.
3. **Tumor Fraction:** The method was effective in classifying samples with low tumor fraction (TF), suggesting that it can detect non-tumor-derived signals that are indicative of cancer. The performance increased with increasing TF, but remained better than chance even in the lowest TF bin.

### IX. INDICATIONS

The invention is particularly useful for the early detection of colorectal cancer (CRC) in high-risk populations. The following indications are considered:

1. **Population-Level Screening:** The method can be used for population-level screening of individuals aged 50-84 years, who are at increased risk of developing CRC. The non-invasive nature of the test makes it suitable for widespread use.
2. **High-Risk Individuals:** The method can be used to screen high-risk individuals, such as those with a family history of CRC, Lynch syndrome, or inflammatory bowel disease. Early detection in these individuals can significantly improve outcomes.
3. **Monitoring Treatment Response:** The method can be used to monitor the response to treatment in patients with CRC. Changes in cfDNA profiles can provide insights into the effectiveness of the treatment and guide clinical decisions.
4. **Recurrence Monitoring:** The method can be used to monitor for recurrence in patients who have undergone treatment for CRC. Regular testing can help detect early signs of recurrence and enable timely intervention.

In conclusion, the invention provides a novel and effective method for the early detection of colorectal cancer using circulating cell-free DNA (cfDNA) and machine learning (ML) techniques. The method has demonstrated high sensitivity and specificity, making it a valuable tool for non-invasive cancer screening and monitoring. Prospective validation studies are underway to further confirm the generalizability and clinical utility of the method.