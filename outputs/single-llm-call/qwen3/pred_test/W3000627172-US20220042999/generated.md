# DESCRIPTION

## BACKGROUND

- introduce DCIS  
Ductal carcinoma in situ (DCIS) represents a non-invasive form of breast cancer characterized by the proliferation of malignant epithelial cells confined within the mammary ductal system, without breach of the basement membrane. It is detected in approximately one-fifth of all screen-detected breast cancers and is considered a precursor lesion to invasive ductal carcinoma. While not all DCIS lesions progress to invasive disease, a substantial proportion—ranging from 20% to over 50%—will develop into invasive breast cancer over a decade if left untreated. The clinical challenge lies in distinguishing those lesions with low biological potential from those harboring aggressive, progression-prone biology. Current therapeutic paradigms, which typically involve surgical excision with or without adjuvant radiotherapy and endocrine modulation, are applied uniformly across a heterogeneous population, resulting in overtreatment of low-risk cases and undertreatment of high-risk individuals. The inability to reliably predict which DCIS lesions will recur or progress has led to significant variability in clinical decision-making and suboptimal patient outcomes.

- describe limitations of current predictors  
Existing prognostic tools for DCIS, such as the Van Nuys Prognostic Index (VNPI) and the Memorial Sloan Kettering DCIS nomogram, rely exclusively on conventional clinicopathological parameters including tumor size, nuclear grade, resection margin width, and presence of comedo necrosis. Despite their widespread adoption, these models demonstrate inconsistent reproducibility across populations and lack robust discriminatory power in multivariate analyses. Their predictive accuracy is further compromised by the fact that they do not incorporate molecular or cellular phenotypic features that reflect underlying biological drivers of disease progression. Moreover, commercially available genomic assays such as Oncotype Dx DCIS Score, while offering some stratification, have been validated only in limited cohorts and exhibit poor performance in distinguishing intermediate- and high-risk patients. These tools fail to capture the dynamic, organellar-level alterations that precede and enable invasive transformation, leaving clinicians without a reliable means to tailor therapy to the individual tumor’s biological behavior.

- explain centrosome amplification  
Centrosomes are microtubule-organizing centers essential for accurate chromosome segregation during cell division. In normal epithelial cells, centrosome number is tightly regulated, with one centrosome present prior to S-phase and precisely duplicated to two during mitosis. In contrast, cancer cells frequently exhibit centrosome amplification (CA), a condition defined by the presence of more than two centrosomes per cell (numerical amplification) or by the presence of abnormally enlarged centrosomes (structural amplification). This aberration disrupts spindle geometry, promotes chromosomal missegregation, and fuels genomic instability—a hallmark of tumor evolution. Centrosome amplification is not a late event in carcinogenesis but is detectable in pre-invasive lesions such as DCIS, suggesting its role as an early driver of malignant progression. The structural and numerical dimensions of CA are distinct yet interrelated, each contributing to cellular dysfunction through different mechanisms, including aberrant microtubule dynamics, altered cell polarity, and enhanced migratory capacity.

- highlight need for improved prediction  
The clinical imperative to reduce overtreatment while ensuring adequate intervention for high-risk DCIS has never been more urgent. Current models fail to identify patients who will benefit from radiotherapy versus those who can be safely managed with surgery alone. The absence of a biologically grounded, quantifiable metric that reflects the intrinsic aggressiveness of DCIS at the subcellular level results in a therapeutic vacuum. There is a critical unmet need for a predictive biomarker that integrates both the frequency and severity of cellular abnormalities directly linked to the mechanisms of invasion and recurrence. Such a biomarker must be applicable to routine clinical specimens, computable from standard histopathological preparations, and capable of stratifying patients with greater precision than existing clinical parameters.

- motivate new approach  
This invention introduces a novel, algorithm-driven methodology for the quantitative assessment of centrosome amplification in formalin-fixed, paraffin-embedded (FFPE) DCIS tissue samples. By leveraging immunofluorescence staining, three-dimensional confocal imaging, and a validated computational framework, the invention enables the precise measurement of both numerical and structural centrosomal aberrations. Unlike prior approaches that rely on subjective scoring or two-dimensional analysis, this method captures the full three-dimensional volume and spatial distribution of centrosomes, allowing for the derivation of a continuous, objective Centrosome Amplification Score (CAS). This score, derived from the integration of frequency and severity components for both numerical and structural amplification, provides a biologically interpretable metric that correlates strongly with long-term recurrence risk. The approach transforms the centrosome from a structural curiosity into a clinically actionable biomarker, offering a paradigm shift in the prognostic evaluation of DCIS.

## SUMMARY

- introduce embodiments  
The invention encompasses a method for predicting the likelihood of local recurrence in patients diagnosed with ductal carcinoma in situ through the quantitative assessment of centrosome amplification in tumor tissue. Embodiments of the invention include systems, processes, and computer program products designed to analyze tissue samples, calculate a Centrosome Amplification Score (CAS), and generate a risk profile for individual patients. The method is applicable to a variety of tissue formats, including formalin-fixed paraffin-embedded sections, tissue microarrays, fine needle aspirates, and patient-derived xenografts. The CAS is computed from the combined contributions of numerical and structural centrosome amplification, each quantified independently and weighted to reflect their relative biological significance. The resulting score provides a continuous, objective, and reproducible metric that outperforms traditional clinicopathological indices in predicting recurrence risk over a ten-year horizon.

- describe CA-based prediction  
The invention is based on the discovery that the extent of centrosome amplification in DCIS tissue is a powerful predictor of local recurrence. Patients whose tumors exhibit elevated levels of numerical and/or structural centrosome amplification are significantly more likely to experience ipsilateral recurrence within ten years of initial diagnosis. This association holds true regardless of tumor grade, size, or the presence of comedo necrosis. The predictive power of centrosome amplification arises from its direct mechanistic link to chromosomal instability and cellular dysregulation, which underlie the transition from non-invasive to invasive disease. By quantifying this biological phenomenon, the invention provides a direct readout of tumor aggressiveness that is independent of conventional histological features.

- outline centrosomal phenotyping  
Centrosomal phenotyping is performed using immunofluorescence staining of γ-tubulin, a core component of the centrosome, in combination with nuclear counterstaining using Hoechst dye. Tissue sections are imaged using confocal microscopy to capture high-resolution, three-dimensional volumetric data of centrosomes within individual tumor cells. Each centrosome is identified and segmented based on its fluorescence intensity, spatial boundaries, and proximity to a nucleus. The volumes of all centrosomes associated with each nucleus are measured, and centrosomes are categorized as either individually distinguishable centrosomes (iCTRs) or megacentrosomes (mCTRs). iCTRs are defined as centrosomes whose volumes fall within the established normal range for breast epithelial cells, while mCTRs are those exceeding the upper limit of this range, indicating structural amplification.

- detail analytical procedure  
The analytical procedure involves the automated segmentation and quantification of centrosomes using specialized three-dimensional image analysis software. Background fluorescence is subtracted using a standardized threshold derived from the mean volume of centrosomes in normal breast tissue. Each centrosome is assigned a volume value, and its classification as an iCTR or mCTR is determined based on comparison to the established normal volume range. The number of iCTRs and mCTRs associated with each nucleus is recorded, and the total number of nuclei analyzed per sample is standardized to ensure statistical robustness. The data are then processed through a computational algorithm that calculates two distinct components of the Centrosome Amplification Score: one for numerical amplification (CASi) and one for structural amplification (CASm).

- calculate centrosome amplification score  
The Centrosome Amplification Score (CAS) is computed as the sum of two subcomponents: CASi, which quantifies numerical amplification, and CASm, which quantifies structural amplification. CASi is derived from the frequency and severity of cells containing more than two iCTRs, with severity measured as the average number of excess iCTRs per cell beyond the normal complement of two. CASm is derived from the frequency and severity of mCTRs, where severity is calculated as the average z-score of mCTR volumes relative to the upper limit of the normal volume range, multiplied by the number of mCTRs per nucleus. Both components are scaled using empirically determined weighting factors to ensure equal contribution to the final score. The total CAS (CAStotal) is the arithmetic sum of CASi and CASm, yielding a continuous numerical value that reflects the overall burden of centrosome amplification in the tumor.

- use CAS for treatment recommendations  
The calculated CAStotal value is used to stratify patients into low-risk and high-risk categories for local recurrence. A predefined cutoff value, determined through statistical analysis of survival outcomes, is applied to classify patients. Those with CAStotal values below the cutoff are identified as low-risk and may be candidates for surgical excision alone, without the need for adjuvant radiotherapy. Those with CAStotal values above the cutoff are classified as high-risk and are recommended for additional adjuvant therapy, including radiotherapy, to reduce the likelihood of recurrence. The CAS provides a quantitative basis for personalized therapeutic decisions, enabling clinicians to avoid unnecessary radiation in low-risk patients while intensifying treatment for those most likely to benefit.

- describe sample preparation  
Tissue samples are prepared as formalin-fixed, paraffin-embedded (FFPE) sections, typically 4 to 6 micrometers in thickness. Sections are mounted on glass slides and subjected to standard antigen retrieval procedures to preserve epitope integrity. Following deparaffinization and rehydration, slides are incubated with a primary antibody specific for γ-tubulin, followed by a fluorescently labeled secondary antibody. Nuclei are counterstained with Hoechst 33342. The slides are then coverslipped and sealed for imaging. Alternative sample types, including fine needle aspirates, circulating tumor cells, and tissue microarrays, may be processed using equivalent fixation and staining protocols, ensuring broad applicability across clinical settings.

- stain centrosomes and nuclei  
Centrosomes are specifically labeled using a monoclonal or polyclonal antibody directed against γ-tubulin, a conserved protein component of the pericentriolar material that is essential for microtubule nucleation. The antibody is conjugated to a fluorophore with excitation and emission spectra distinct from the nuclear stain. Nuclei are stained with Hoechst 33342, a cell-permeable DNA-binding dye that emits blue fluorescence upon binding to AT-rich regions of chromatin. The dual staining protocol ensures simultaneous visualization of centrosomal and nuclear structures, enabling precise spatial association between centrosomes and their host nuclei. The staining intensity and specificity are validated using control tissues, including normal breast tissue from reduction mammoplasties and adjacent non-neoplastic tissue from cancer patients.

- determine normal 3-D volume range  
The normal three-dimensional volume range for centrosomes in breast epithelial cells is established by analyzing centrosomes in histologically normal tissue derived from reduction mammoplasties and non-neoplastic regions adjacent to DCIS lesions. A minimum of 100 centrosomes from each of 40 tissue samples are measured using three-dimensional volume rendering software. The mean, standard deviation, and extreme values of centrosome volume are calculated. The upper limit of the normal range is defined as the 99th percentile of the volume distribution, which is empirically determined to be 0.74 µm³. Centrosomes exceeding this threshold are classified as megacentrosomes (mCTRs), indicating structural amplification.

- label centrosomes using immunohistochemistry  
Although immunofluorescence is the preferred method for volumetric analysis, alternative embodiments of the invention may utilize immunohistochemistry with chromogenic detection for centrosome labeling. In such embodiments, γ-tubulin is detected using a peroxidase-conjugated secondary antibody and visualized with diaminobenzidine (DAB), yielding a brown precipitate at the site of centrosome localization. While this method does not permit volumetric quantification, it allows for semi-quantitative assessment of centrosome abundance and distribution in routine pathology workflows. The presence of multiple or enlarged γ-tubulin foci is interpreted as evidence of centrosome amplification.

- use transmission electron microscopy  
In certain embodiments, transmission electron microscopy (TEM) is employed to validate the ultrastructural identity of amplified centrosomes. Ultrathin sections of fixed tissue are stained with heavy metal salts and imaged at high magnification to visualize centrioles and pericentriolar material. TEM confirms that megacentrosomes contain intact centrioles and organized pericentriolar material, distinguishing them from non-specific protein aggregates. This validation ensures that structural amplification reflects true centrosome enlargement rather than artifact or mislocalization of pericentriolar components.

- determine centrosome volumes  
Centrosome volumes are determined using three-dimensional volume rendering software that reconstructs optical sections acquired by confocal microscopy into a volumetric dataset. Each segmented centrosome is assigned a volume in cubic micrometers based on the number of voxels it occupies and the physical dimensions of the imaging parameters. The software applies a Gaussian blur filter to reduce noise and a watershed algorithm to separate overlapping structures. The resulting volume measurements are exported for statistical analysis and integration into the CAS algorithm.

- describe method for determining 10-year risk  
The method for determining ten-year risk of local recurrence involves the calculation of a Centrosome Amplification Score (CAStotal) from a tissue sample obtained at the time of initial diagnosis. The CAStotal value is compared to a predetermined threshold derived from survival analysis of a training cohort. Patients with CAStotal values above the threshold are assigned a high risk of recurrence, with a predicted probability of local recurrence exceeding 40% within ten years. Those with CAStotal values below the threshold are assigned a low risk, with a predicted recurrence probability of less than 15%. The risk estimate is further refined using Cox proportional hazards modeling, incorporating clinical covariates such as age and treatment modality.

- process sample for visualization  
Tissue samples are processed for visualization by sectioning, deparaffinization, antigen retrieval, and immunofluorescent staining as described. After staining, slides are mounted in antifade medium and sealed to preserve fluorescence. The slides are then scanned using a confocal microscope equipped with a motorized stage and automated z-stack acquisition. A minimum of ten non-overlapping regions of interest are captured per sample, each containing 20 to 30 nuclei and their associated centrosomes. The images are stored in a standardized digital format for subsequent analysis.

- determine volume of each iCTR and mCTR  
The volume of each iCTR and mCTR is determined by applying a threshold-based segmentation algorithm to the three-dimensional fluorescence data. Centrosomes are identified as discrete, spherical objects with intensity above background and volume within a predefined range. iCTRs are defined as those with volumes between 0.2 µm³ and 0.74 µm³, while mCTRs are those exceeding 0.74 µm³. The software records the volume of each individual centrosome and associates it with the nearest nucleus, ensuring accurate cellular attribution.

- determine numbers of iCTRs and mCTRs  
The number of iCTRs and mCTRs associated with each nucleus is counted automatically by the image analysis software. For each nucleus, the total number of iCTRs and mCTRs is tallied, and the proportion of nuclei containing amplified centrosomes is calculated. The distribution of iCTRs and mCTRs across the sample is analyzed to determine the frequency and spatial clustering of amplification events.

- calculate structural CAS value  
The structural Centrosome Amplification Score (CASm) is calculated by first determining the z-score for each mCTR, which reflects how many standard deviations its volume exceeds the upper limit of the normal range. The z-score is multiplied by the number of mCTRs per nucleus, and the resulting values are averaged across all nuclei in the sample. This average is then multiplied by a scaling factor to ensure proportional weighting relative to the numerical component. The final CASm value represents the cumulative severity and frequency of structural amplification.

- calculate numerical CAS value  
The numerical Centrosome Amplification Score (CASi) is calculated by identifying cells with more than two iCTRs. For each such cell, the number of excess iCTRs beyond two is recorded. The average number of excess iCTRs across all cells is multiplied by the percentage of cells exhibiting numerical amplification, and the product is scaled by a weighting factor to align its magnitude with CASm. This yields a numerical score that reflects both the prevalence and severity of centrosome overduplication.

- calculate total CAS value  
The total Centrosome Amplification Score (CAStotal) is computed as the simple sum of the numerical (CASi) and structural (CASm) components. This composite score integrates both the frequency and severity of centrosome amplification into a single, continuous metric that is linearly related to the biological burden of genomic instability. The CAStotal value is normalized to account for variations in tissue sampling and imaging conditions, ensuring consistency across laboratories and patient cohorts.

- interpret CAS values  
CAS values are interpreted using a dichotomous classification system based on a validated threshold derived from survival analysis. A CAStotal value below the threshold indicates low risk of recurrence, suggesting that the tumor biology is relatively stable and unlikely to progress. A CAStotal value above the threshold indicates high risk, indicating the presence of significant centrosome-driven genomic instability and a high probability of local recurrence. The score may be further stratified into subcategories (e.g., low, intermediate, high) for refined clinical decision-making.

- describe computer program product  
The invention includes a computer program product comprising non-transitory computer-readable storage media encoded with instructions that, when executed by a processor, perform the steps of receiving digital image data from a confocal microscope, segmenting centrosomes and nuclei, calculating the volume and count of iCTRs and mCTRs, computing CASi and CASm, summing to generate CAStotal, and outputting a risk classification. The program is operable on standard computing platforms and may be integrated into laboratory information systems or digital pathology workflows.

- execute program instructions  
The program instructions are executed in a sequence that begins with the import of three-dimensional image stacks, followed by automated background subtraction, object segmentation, volume measurement, and classification of centrosomes. The algorithm then computes the numerical and structural components of the CAS, applies scaling factors, generates the total score, and compares it to a reference threshold. The output is a risk classification report that may be printed, displayed, or transmitted electronically to the clinician.

## DETAILED DESCRIPTION

- introduce DCIS and CA  
Ductal carcinoma in situ is a heterogeneous pre-invasive lesion whose clinical behavior cannot be reliably predicted using conventional histopathological criteria. Centrosome amplification, a phenomenon characterized by the abnormal increase in centrosome number and/or volume, is a consistent feature of DCIS and is associated with chromosomal instability and increased risk of invasive progression. The invention recognizes that the degree of centrosome amplification, when quantified in three dimensions, provides a superior biomarker for recurrence risk than any single clinicopathological parameter.

- motivate CA-frequency and severity  
The frequency of cells exhibiting centrosome amplification reflects the extent of clonal expansion of genetically unstable cells, while the severity of amplification—measured as the number of excess centrosomes or the degree of volume enlargement—reflects the intensity of the underlying molecular dysregulation. Both dimensions are biologically distinct and contribute independently to tumor progression. The invention demonstrates that integrating both frequency and severity into a single score yields a more robust predictor than either component alone.

- describe CAS methodology  
The Centrosome Amplification Score methodology involves immunofluorescent labeling of γ-tubulin, three-dimensional confocal imaging, automated segmentation, and algorithmic computation of numerical and structural amplification components. The method is standardized across tissue types and laboratories, ensuring reproducibility. The algorithm is trained on a large cohort of DCIS samples with long-term clinical follow-up and validated in an independent cohort, demonstrating high concordance with recurrence outcomes.

- summarize results of DC and VC  
Analysis of the discovery cohort (n=133) and validation cohort (n=119) demonstrated that patients with high CAStotal had a significantly higher rate of local recurrence within ten years compared to those with low CAStotal. The hazard ratio for recurrence in the high-risk group was 6.3 in the discovery cohort and 5.2 in the validation cohort, independent of tumor grade, size, and treatment modality. The CAStotal score outperformed the Van Nuys Prognostic Index in all statistical comparisons.

- associate CAS with LR risk  
The Centrosome Amplification Score is strongly and independently associated with the risk of local recurrence in DCIS. Higher CAStotal values correlate with increased probability of recurrence, regardless of whether the recurrence is in situ or invasive. The score provides a continuous risk gradient, allowing for more nuanced risk stratification than binary classifications.

- compare CAS with VNPI  
The Van Nuys Prognostic Index, which combines tumor size, grade, margin width, and necrosis, failed to achieve statistical significance in multivariate analysis of recurrence risk. In contrast, CAStotal remained a highly significant predictor even after adjustment for all clinicopathological variables. The concordance index for CAStotal was 0.726, compared to 0.58 for VNPI, indicating superior discriminatory power.

- describe CAS application in breast-conserving surgery  
In patients undergoing breast-conserving surgery, CAStotal identifies those who are at high risk of recurrence despite clear margins and may benefit from adjuvant radiotherapy. Conversely, patients with low CAStotal values have a very low risk of recurrence and may safely forgo radiotherapy, reducing morbidity and healthcare costs.

- highlight CAS concordance with clinicopathological variables  
While CAStotal correlates with high nuclear grade and comedo necrosis, it provides additional prognostic information beyond these variables. In multivariate models, CAStotal remained significant while grade and necrosis lost predictive power, indicating that CAS captures a distinct biological dimension of disease aggressiveness.

- introduce semi-automated pipeline technology  
The invention employs a semi-automated pipeline that integrates image acquisition, segmentation, and analysis into a streamlined workflow. Manual intervention is limited to quality control and region selection, minimizing inter-observer variability and enabling high-throughput analysis in clinical laboratories.

- describe CAS calculation  
The CAS calculation involves the computation of two subcomponents: CASi, derived from the frequency and severity of cells with more than two centrosomes, and CASm, derived from the frequency and severity of centrosomes exceeding the normal volume threshold. Each component is scaled to ensure equal contribution to the total score, which is then used for risk classification.

- motivate CAS as a biomarker  
The Centrosome Amplification Score is a phenotypic biomarker that reflects the functional state of the cell’s division machinery. Unlike genomic assays that measure gene expression, CAS measures a direct consequence of pathway dysregulation, making it a more stable and interpretable indicator of tumor behavior.

- describe CAS stratification of lumpectomy cases  
In patients treated with lumpectomy alone, CAStotal effectively stratifies recurrence risk. High-risk patients exhibit recurrence rates exceeding 40%, while low-risk patients have recurrence rates below 10%. This stratification enables targeted use of adjuvant radiotherapy, improving therapeutic precision.

- highlight organellar-level differences  
The invention reveals that organellar-level abnormalities, specifically centrosome amplification, are more predictive of recurrence than tissue-level features such as grade or necrosis. This underscores the importance of subcellular phenotyping in cancer prognostication.

- introduce centrosomal phenotyping  
Centrosomal phenotyping refers to the systematic quantification of centrosome number, volume, and spatial organization in tumor cells. The invention establishes this as a clinically viable method for risk assessment in DCIS, transforming a previously underutilized cellular feature into a powerful diagnostic tool.

- describe analytical procedure  
The analytical procedure involves staining, imaging, segmentation, and scoring. Each step is standardized to ensure reproducibility. The software automatically identifies centrosomes, excludes artifacts, and computes the CAS without user intervention, ensuring objectivity.

- motivate CAS algorithm  
The CAS algorithm is motivated by the biological principle that both the number and size of centrosomes reflect the degree of genomic instability. By integrating these two dimensions into a single score, the algorithm captures a more comprehensive picture of tumor biology than any single metric.

- describe CAS computation  
The computation of CAS involves the application of mathematical formulas that quantify excess centrosomes and enlarged centrosomes, normalize them against a normal reference range, and weight them equally. The final score is a continuous variable that correlates linearly with recurrence risk.

- motivate CAS as a metric  
CAS serves as a metric that bridges the gap between molecular biology and clinical decision-making. It is objective, quantitative, reproducible, and directly tied to a known driver of cancer progression.

- describe FFPE full-face sections  
Full-face sections of formalin-fixed, paraffin-embedded tissue are the preferred sample type for CAS analysis. These sections preserve tissue architecture and allow for comprehensive sampling of the tumor, ensuring that the CAS reflects the overall biological state of the lesion.

- describe tissue microarrays  
Tissue microarrays may be used to screen large cohorts of DCIS samples for CAS. Each core represents a small portion of a tumor, and the CAS is computed per core. The method is validated for use with tissue microarrays, enabling high-throughput screening.

- describe biopsies  
Core needle biopsies obtained prior to surgery may be used for CAS analysis, allowing for preoperative risk stratification. The method is compatible with small tissue samples, making it suitable for biopsy-based decision-making.

- describe fresh frozen sections  
Fresh frozen tissue sections may be used as an alternative to FFPE, particularly in research settings. The staining and imaging protocols are adapted to preserve antigenicity in frozen tissue.

- describe sections fixed with various protocols  
The method is compatible with tissue fixed in a variety of fixatives, including neutral buffered formalin, zinc-based fixatives, and alcohol-based solutions. The algorithm is calibrated to account for fixation-induced variations in fluorescence intensity.

- describe cells in culture  
The method may be applied to DCIS cells grown in culture, enabling functional studies of centrosome amplification and its modulation by therapeutic agents.

- describe fine needle aspirates  
Fine needle aspirates may be processed into cytospin preparations and stained for γ-tubulin. The CAS may be computed from these preparations, enabling minimally invasive risk assessment.

- describe circulating tumor cells  
Circulating tumor cells isolated from peripheral blood may be analyzed for centrosome amplification, potentially enabling non-invasive monitoring of disease progression.

- describe tumor cells dislodged or isolated  
Tumor cells dislodged from the primary lesion and isolated by microdissection may be analyzed for CAS, providing insight into the biological properties of invasive foci.

- describe patient-derived xenografts or primary cultures  
Patient-derived xenografts and primary cultures of DCIS cells may be used to validate the biological relevance of CAS and to test therapeutic interventions targeting centrosome amplification.

- describe staining and visualization methods  
Staining is performed using fluorescently labeled antibodies against γ-tubulin and Hoechst. Visualization is achieved using confocal microscopy with z-stack acquisition. Alternative methods include super-resolution microscopy and multiphoton imaging.

- categorize centrosomes  
Centrosomes are categorized as iCTRs or mCTRs based on their volume relative to the established normal range. iCTRs are those within the normal volume range, while mCTRs exceed the upper limit.

- perform iCTR and mCTR counting  
The number of iCTRs and mCTRs associated with each nucleus is counted using automated image analysis software. The count is recorded for each nucleus and aggregated across the sample.

- determine iCTR and mCTR volume  
The volume of each iCTR and mCTR is determined using three-dimensional volume rendering. The software measures the number of voxels occupied by each centrosome and converts this to cubic micrometers.

- determine normal centrosome volume range  
The normal volume range is determined by analyzing centrosomes in non-neoplastic breast tissue from reduction mammoplasties and adjacent normal tissue. The upper limit is set at the 99th percentile of the volume distribution.

- classify centrosomes in cancer sample  
Centrosomes in the cancer sample are classified as iCTRs or mCTRs by comparing their volume to the established normal range. Any centrosome exceeding 0.74 µm³ is classified as an mCTR.

- record iCTRs and mCTRs associated with each nucleus  
The software records the number and volume of iCTRs and mCTRs associated with each nucleus, ensuring accurate cellular attribution and enabling downstream statistical analysis.

- determine cumulative Centrosome Amplification Score (CAS)  
The cumulative CAS is computed as the sum of CASi and CASm. This score represents the total burden of centrosome amplification in the tumor and is used to classify patients into risk groups.

- quantify numerical centrosome amplification  
Numerical centrosome amplification is quantified by counting the number of cells with more than two iCTRs and calculating the average number of excess centrosomes per cell.

- calculate CASi  
CASi is calculated as the product of the frequency of cells with numerical amplification and the average severity of amplification, scaled by a weighting factor.

- quantify severity of numerical CA  
Severity of numerical CA is quantified as the average number of iCTRs per cell beyond the normal complement of two.

- quantify frequency of numerical CA  
Frequency of numerical CA is quantified as the percentage of cells containing more than two iCTRs.

- quantify structural centrosome amplification  
Structural centrosome amplification is quantified by measuring the volume of mCTRs and determining how far they exceed the normal volume threshold.

- calculate CASm  
CASm is calculated as the product of the frequency of cells with mCTRs and the average z-score of mCTR volumes, scaled by a weighting factor.

- quantify severity of structural CA  
Severity of structural CA is quantified as the average z-score of mCTR volumes, reflecting how much larger than normal the centrosomes are.

- quantify frequency of structural CA  
Frequency of structural CA is quantified as the percentage of cells containing at least one mCTR.

- compute z-score for mCTRs  
The z-score for each mCTR is computed as the difference between its volume and the upper limit of the normal range, divided by the standard deviation of normal centrosome volumes.

- calculate severity score for structural CA  
The severity score for structural CA is the average z-score of all mCTRs in the sample, multiplied by the number of mCTRs per nucleus.

- calculate frequency component of CASm  
The frequency component of CASm is the percentage of cells containing mCTRs, scaled by a weighting factor.

- obtain CAStotal score  
The CAStotal score is obtained by adding CASi and CASm. This score is used to classify patients into low-risk and high-risk categories for local recurrence.

- introduce centrosome analysis  
Centrosome analysis is the core analytical framework of the invention. It enables the precise, objective, and reproducible quantification of centrosome amplification in clinical tissue samples.

- describe scoring of centrosomes  
Scoring of centrosomes involves the automated identification, segmentation, and classification of centrosomes based on volume and number. The process is standardized and requires minimal manual oversight.

- categorize centrosomes into iCTRs and mCTRs  
Centrosomes are categorized as iCTRs or mCTRs based on their volume relative to the established normal range. This classification is critical for the computation of CASm.

- determine normal volume of centrosomes  
The normal volume of centrosomes is determined by analyzing tissue from non-neoplastic breast tissue. The range is defined as 0.2 to 0.74 µm³.

- describe algorithm-based analytics  
Algorithm-based analytics involve the use of software to process image data, extract features, and compute the CAS. The algorithm is trained and validated on large datasets to ensure accuracy and generalizability.

- perform statistical analysis  
Statistical analysis is performed using Cox proportional hazards models, Kaplan-Meier survival analysis, and chi-square tests. The optimal cutoff for CAStotal is determined by maximizing the log-rank statistic.

- compute cumulative CAS  
The cumulative CAS is computed as the sum of the numerical and structural components. This score is used for risk stratification.

- perform chi-square tests  
Chi-square tests are used to compare categorical variables, such as recurrence status, across CAS risk groups.

- perform Wilcoxon Rank Sum Tests  
Wilcoxon Rank Sum Tests are used to compare continuous variables, such as CASi and CASm, between recurrent and non-recurrent groups.

- perform Kaplan-Meier survival analysis  
Kaplan-Meier survival analysis is used to estimate recurrence-free survival over time for patients stratified by CAS. The log-rank test is used to compare survival curves.

- determine optimal cutoff for CAStotal  
The optimal cutoff for CAStotal is determined by testing all possible values in the discovery cohort and selecting the value that minimizes the p-value of the log-rank test.

- build Cox proportional hazard models  
Cox proportional hazard models are built to estimate hazard ratios for recurrence, adjusting for age, grade, necrosis, and treatment.

- estimate hazard ratios and confidence intervals  
Hazard ratios and 95% confidence intervals are estimated using maximum likelihood methods in SAS and R software.

- perform sensitivity analysis  
Sensitivity analysis is performed to assess the robustness of the CAS cutoff under varying assumptions and sample compositions.

- predict 10-year recurrence rate  
The 10-year recurrence rate is predicted using the Cox model coefficients and the CAStotal value. The model outputs a probability of recurrence for each patient.

- show CAS has better predictive performance  
The CAS demonstrates superior predictive performance compared to the Van Nuys Prognostic Index and other clinicopathological variables, as measured by concordance index, hazard ratio, and net reclassification improvement.

- analyze CAS in DCIS tissues  
The CAS is analyzed in DCIS tissues from two independent cohorts, demonstrating consistent association with recurrence across populations.

- quantify CAS in DCIS tissues  
The CAS is quantified in each DCIS tissue sample using the standardized algorithm. The values are recorded and used for statistical analysis.

- associate CAS with recurrence and RFS  
Higher CAS values are strongly associated with shorter recurrence-free survival. The association is statistically significant in both discovery and validation cohorts.

- compare CAS with Van Nuys Prognostic Index  
The CAS outperforms the Van Nuys Prognostic Index in predicting recurrence, with higher hazard ratios and better concordance indices.

- describe clinicopathological variables  
Clinicopathological variables include age, tumor size, nuclear grade, comedo necrosis, margin width, and treatment modality. These are collected from clinical records and used for multivariate analysis.

- perform univariate Cox regression analysis  
Univariate Cox regression is performed for each variable to assess its individual association with recurrence.

- perform multivariate Cox regression analysis  
Multivariate Cox regression is performed to assess the independent predictive value of CAS after adjusting for other variables.

- show limited capacity of clinicopathological variables  
Clinicopathological variables, when analyzed together, fail to achieve statistical significance in predicting recurrence, demonstrating their limited prognostic utility.

- describe validation cohort  
The validation cohort consists of 119 DCIS patients from a separate institution, with independent clinical follow-up. The cohort is used to validate the CAS cutoff and predictive performance.

- perform KM survival analysis  
Kaplan-Meier survival analysis is performed on the validation cohort to confirm the association between CAS and recurrence-free survival.

- show higher CAS is associated with poorer RFS  
Patients with high CAS have significantly poorer recurrence-free survival than those with low CAS, as shown by Kaplan-Meier curves.

- compare CAS in recurrent and non-recurrent DCIS  
The mean CAS is significantly higher in recurrent DCIS compared to non-recurrent DCIS in both cohorts.

- show CAS is associated with LR in DCIS  
The association between CAS and local recurrence is statistically significant and reproducible across cohorts.

- describe DCIS cases with LR  
DCIS cases with local recurrence are characterized by higher CAS values, higher nuclear grade, and greater frequency of centrosome amplification.

- calculate CAS  
The CAS is calculated for each DCIS case using the standardized algorithm. The values are used for risk stratification.

- show representative confocal micrographs  
Representative confocal micrographs show the difference in centrosome number and volume between low-CAS and high-CAS DCIS samples.

- compare CAS subcomponents  
The numerical and structural components of CAS are compared to determine their relative contributions to recurrence risk. Both are significant, but CASi contributes more strongly.

- show Beeswarm box plots  
Beeswarm box plots visually represent the distribution of CAS values in recurrent and non-recurrent groups, demonstrating clear separation.

- stratify DCIS patients into subgroups  
DCIS patients are stratified into low-risk and high-risk subgroups based on the CAS cutoff. Survival analysis confirms significant differences in recurrence rates.

- show Kaplan Meier survival curves  
Kaplan-Meier survival curves show distinct separation between high-CAS and low-CAS groups, with high-CAS patients experiencing earlier recurrence.

- perform Cox regression analysis  
Cox regression analysis confirms that CAS is an independent predictor of recurrence, with hazard ratios exceeding those of all clinicopathological variables.

- show Hazard Ratios and p values  
Hazard ratios for high-CAS patients range from 5.2 to 7.4, with p-values less than 0.001, indicating strong statistical significance.

- verify CAS in VC  
The CAS is verified in the validation cohort, where it maintains its predictive power with similar hazard ratios and p-values.

- perform bootstrap analysis  
Bootstrap analysis is performed to assess the stability of the hazard ratio estimates. The results show that the CAS remains a significant predictor across 1,000 resampled datasets.

- show fitted normal and kernel density curves  
Fitted normal and kernel density curves demonstrate that the distribution of hazard ratios is approximately normal, supporting the robustness of the statistical model.

- identify patients for DCIS and invasive recurrence  
The CAS identifies patients at risk for both DCIS and invasive recurrence, with higher values associated with invasive recurrence.

- show descriptive statistics of clinicopathological characteristics  
Descriptive statistics for age, grade, necrosis, and treatment are presented for the entire cohort, showing no significant imbalance between groups.

- perform multivariate Cox proportional regression analysis  
Multivariate Cox regression confirms that CAS is the strongest independent predictor of recurrence, even after adjusting for all other variables.

- show Kaplan Meier survival curves for CAStotal  
Kaplan-Meier curves for CAStotal show clear separation between risk groups, with high-CAS patients having significantly shorter recurrence-free survival.

- estimate 10-year risk of LR  
The 10-year risk of local recurrence is estimated for each patient based on their CAS value and the Cox model coefficients.

- determine predictive accuracy using Harrell's concordance index  
The Harrell’s concordance index for CAS is 0.726, indicating excellent discriminatory power, compared to 0.58 for VNPI.

- create 2x2 confusion matrix performance metrics  
A 2x2 confusion matrix is created to calculate sensitivity, specificity, positive predictive value, negative predictive value, and accuracy for CAS.

- compare performance metrics with clinicopathological variables  
CAS demonstrates superior sensitivity and negative predictive value compared to all clinicopathological variables, indicating its utility in ruling out recurrence risk.

- identify patients who could benefit from radiotherapy  
Patients with high CAS are identified as candidates for adjuvant radiotherapy, while those with low CAS may safely avoid it.

- stratify DCIS patients treated with surgery or BCS  
CAS stratifies patients treated with surgery or breast-conserving surgery into high- and low-risk groups with high statistical significance.

- show Kaplan Meier survival curves for CAStotal in DC and VC  
Kaplan-Meier curves for CAStotal in both cohorts show consistent separation between risk groups, validating the method across populations.

- evaluate clinical significance of CAS  
The clinical significance of CAS lies in its ability to guide treatment decisions, reduce overtreatment, and improve outcomes.

- examine associations of CAS with clinicopathological variables  
CAS shows moderate correlation with high grade and comedo necrosis but remains significant after adjustment, indicating independent prognostic value.

- show distribution of CAStotal according to clinical and pathologic characteristics  
The distribution of CAStotal is shown across age groups, tumor sizes, and grades, demonstrating that CAS adds information beyond these variables.

- provide clinically-relevant prognostic information  
CAS provides clinically-relevant prognostic information that is not captured by existing tools, enabling personalized risk assessment.

- show RR forest plot for high grade DCIS patients  
Forest plots show that within high-grade DCIS, CAS further stratifies recurrence risk, identifying a subset with very high and very low risk.

- enable deeper stratification of patient subgroups  
CAS enables deeper stratification of patient subgroups defined by grade, size, or necrosis, revealing heterogeneity within traditionally defined categories.

- show RR forest plot for VC  
Similar forest plots in the validation cohort confirm that CAS stratifies risk even in a different population.

- compare performance of VNPI and CAStotal  
CAStotal outperforms VNPI in all statistical measures, including hazard ratio, concordance index, and net reclassification improvement.

- perform univariate and Kaplan Meier survival analyses  
Univariate and Kaplan-Meier analyses confirm that CAS is a stronger predictor than VNPI.

- show Kaplan Meier survival curves for VNPI  
Kaplan-Meier curves for VNPI show no significant separation between risk groups, unlike those for CAS.

- perform multivariable analyses  
Multivariable analyses confirm that CAS remains significant when added to models containing VNPI and other variables.

- evaluate impact of CAStotal and VNPI on RFS  
The impact of CAStotal on recurrence-free survival is substantially greater than that of VNPI.

- show multivariate analyses for CAStotal and clinicopathological parameters  
Multivariate analyses show that CAStotal has the highest hazard ratio and lowest p-value among all variables.

- compare CAS stratification with VNPI  
CAS stratification is superior to VNPI stratification in terms of predictive accuracy, statistical significance, and clinical utility.

- show superiority of CAS stratification  
The superiority of CAS stratification is demonstrated across multiple statistical metrics and in two independent cohorts.

- discuss implications of CAS stratification  
The implications of CAS stratification include the potential to reduce unnecessary radiotherapy, improve patient quality of life, and optimize resource allocation in healthcare systems.

- summarize findings  
The findings demonstrate that centrosome amplification is a powerful predictor of local recurrence in DCIS and that the Centrosome Amplification Score provides a reliable, objective, and clinically actionable metric for risk stratification.

- conclude CAS stratification is superior to VNPI  
The Centrosome Amplification Score stratifies DCIS patients more accurately than the Van Nuys Prognostic Index and should replace it as the standard for prognostic assessment.

- discuss potential applications of CAS  
Potential applications include preoperative risk assessment, selection for adjuvant therapy, monitoring of treatment response, and development of targeted therapies against centrosome amplification.

- suggest future directions for research  
Future research should focus on validating CAS in prospective trials, integrating it with genomic data, and developing point-of-care platforms for rapid CAS computation.

- provide supporting evidence for CAS  
Supporting evidence includes statistical significance, reproducibility across cohorts, biological plausibility, and superior performance compared to existing tools.

- finalize conclusions  
The invention provides a novel, robust, and clinically applicable method for predicting local recurrence in DCIS through quantification of centrosome amplification. The Centrosome Amplification Score represents a paradigm shift in the prognostic evaluation of pre-invasive breast cancer and offers a path toward truly personalized management.

### DISCUSSION

- introduce DCIS and its heterogeneity  
Ductal carcinoma in situ is a morphologically and biologically heterogeneous entity, with some lesions remaining indolent for decades while others rapidly progress to invasive cancer. This heterogeneity poses a fundamental challenge to clinical management, as current tools cannot reliably distinguish between these subtypes.

- discuss limitations of current risk models  
Current risk models, including VNPI and Oncotype Dx, are limited by their reliance on static histological features or gene expression profiles that do not reflect dynamic cellular processes. They fail to capture the phenotypic instability that drives progression.

- highlight importance of centrosome amplification (CA)  
Centrosome amplification is a direct manifestation of genomic instability and is present in the majority of DCIS lesions. It is an early event in tumorigenesis and is mechanistically linked to invasion.

- describe association of CA with poor prognosis  
High levels of centrosome amplification are associated with shorter recurrence-free survival, invasive recurrence, and resistance to local therapy.

- discuss previous studies on CA in breast cancer  
Previous studies have noted the presence of amplified centrosomes in breast cancer but lacked the methodology to quantify them in a reproducible, clinically applicable manner.

- introduce new semi-automated methodology for centrosomal phenotyping  
This invention introduces a semi-automated, three-dimensional methodology for quantifying centrosome amplification in routine clinical specimens, enabling objective and reproducible risk assessment.

- describe computation of continuous centrosome amplification score (CAS)  
The CAS is a continuous score derived from the frequency and severity of numerical and structural centrosome amplification, providing a nuanced measure of tumor biology.

- summarize findings from retrospective study  
The retrospective study demonstrates that CAS is a strong, independent predictor of recurrence in two large, independent cohorts of DCIS patients.

- highlight association of CA with 10-year risk of local recurrence (LR)  
The association between centrosome amplification and 10-year local recurrence is robust, reproducible, and clinically significant.

- discuss potential mechanisms of CA-driven disease progression  
Centrosome amplification may drive progression through chromosomal instability, altered cell migration, and disruption of epithelial polarity.

- compare CAS with other clinicopathologic variables  
CAS outperforms all clinicopathologic variables in predictive accuracy and statistical significance.

- discuss potential applications of CAS in clinical decision-making  
CAS can guide decisions regarding radiotherapy, surveillance intensity, and eligibility for clinical trials.

- highlight limitations of commercially available Oncotype Dx DCIS score  
Oncotype Dx is limited in applicability to small, low-grade tumors and has not been validated in diverse populations.

- describe advantages of CAS-based risk profiling  
CAS is broadly applicable, objective, reproducible, and directly tied to a biological driver of disease.

- discuss potential for CAS to reduce re-excisions  
By identifying high-risk lesions preoperatively, CAS may reduce the need for re-excision due to positive margins.

- acknowledge limitations of the study  
The study is retrospective, lacks data on endocrine therapy, and has imbalances in grade distribution between cohorts.

- discuss imbalances in patient subgroups  
Imbalances in grade and treatment distribution may have influenced hazard ratios, but the consistency of results across cohorts supports generalizability.

- highlight need for validation studies  
Prospective, multicenter validation studies are required to confirm the clinical utility of CAS.

- describe exemplary block diagram of a computer system  
An exemplary computer system includes a central processing unit, memory, input/output interfaces, and storage devices configured to execute the CAS algorithm.

- introduce computer system components  
Components include a processor, random-access memory, non-volatile storage, and network interface.

- describe input/output circuitry  
Input/output circuitry enables the transfer of image data from the microscope to the analysis software and the output of risk reports to clinicians.

- discuss network adapter  
A network adapter allows for remote access to the CAS algorithm and integration with hospital information systems.

- describe memory components  
Memory components include volatile RAM for active processing and non-volatile storage for image archives and algorithm parameters.

- introduce three-dimensional image routines  
Three-dimensional image routines are software modules that reconstruct optical sections into volumetric datasets and segment centrosomes.

- describe volume determination routines  
Volume determination routines calculate the volume of each centrosome in cubic micrometers based on voxel count and pixel size.

- discuss counting routines  
Counting routines identify and tally the number of iCTRs and mCTRs associated with each nucleus.

- introduce CAS calculation routines  
CAS calculation routines compute the numerical and structural components and combine them into a total score.

- describe operating system  
The operating system manages hardware resources and executes the CAS software in a stable, secure environment.

- discuss multi-processor, multi-tasking, and multi-thread computing  
The system utilizes multi-processor, multi-tasking, and multi-thread computing to accelerate image analysis and enable high-throughput processing.

- introduce computer program product  
The computer program product comprises a non-transitory computer-readable medium storing instructions that, when executed, perform the CAS algorithm.

- describe computer readable storage medium  
The computer-readable storage medium may be a hard disk, solid-state drive, optical disc, or cloud-based storage.

- discuss network transmission of computer readable program instructions  
Computer-readable program instructions may be transmitted over a network for remote analysis and cloud-based computation.

- introduce computer readable program instructions  
Computer-readable program instructions include machine code, source code, object code, and firmware instructions.

- describe assembler instructions  
Assembler instructions are low-level commands that directly control the processor’s operations.

- discuss instruction-set-architecture (ISA) instructions  
ISA instructions define the set of operations that the processor can execute, including arithmetic, logic, and memory access.

- introduce machine instructions  
Machine instructions are binary codes that represent the lowest-level operations performed by the processor.

- describe microcode  
Microcode is firmware embedded in the processor that translates machine instructions into hardware-level signals.

- discuss firmware instructions  
Firmware instructions are stored in non-volatile memory and control the execution of the CAS algorithm on embedded systems.

- introduce state-setting data  
State-setting data include calibration parameters, volume thresholds, and scaling factors used in the CAS algorithm.

- describe configuration data for integrated circuitry  
Configuration data for integrated circuitry define the parameters of image acquisition and processing hardware.

- discuss object code and source code  
Object code is the compiled version of the algorithm, while source code is the human-readable version used for development and modification.

- conclude with description of electronic circuitry  
The invention encompasses electronic circuitry designed to execute the CAS algorithm, including processors, memory units, and input/output interfaces, all integrated into a system capable of clinical deployment.