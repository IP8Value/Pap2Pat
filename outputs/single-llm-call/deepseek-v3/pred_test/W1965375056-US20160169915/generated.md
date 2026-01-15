Here is the patent application following the provided outline:

# DESCRIPTION  

## CONTINUING APPLICATION DATA  

The present application claims benefit under 35 U.S.C. §119(e) of U.S. Provisional Patent Application No. 62/XXXXXX, filed Month Day, Year, entitled "METHODS FOR DETERMINING RISK OF AUTISM SPECTRUM DISORDER USING METABOLOMIC SIGNATURES", the disclosure of which is incorporated herein by reference in its entirety.  

## BACKGROUND  

Autism spectrum disorder (ASD) is a complex neurodevelopmental condition characterized by persistent deficits in social communication and interaction across multiple contexts, as well as restricted, repetitive patterns of behavior, interests, or activities. The diagnostic criteria for ASD encompass a broad range of symptoms and levels of disability, from mild to severe impairment. Current epidemiological studies estimate that approximately 1 in 68 children in the United States meet diagnostic criteria for ASD, representing a significant public health concern.  

Current diagnostic methods for ASD rely primarily on behavioral observations and developmental assessments, typically administered by trained clinicians. These methods include standardized instruments such as the Autism Diagnostic Observation Schedule (ADOS) and the Autism Diagnostic Interview-Revised (ADI-R). While these tools represent the gold standard for clinical diagnosis, they suffer from several critical limitations. First, behavioral diagnosis cannot typically be made with confidence until approximately 4 years of age, missing a critical window for early intervention during the first years of life when the brain demonstrates maximal plasticity. Second, the diagnostic process requires specialized training and can be time-consuming, often resulting in significant delays between initial parental concerns and formal diagnosis. Third, the subjective nature of behavioral assessments can lead to variability in diagnostic accuracy across clinicians and settings.  

The development of objective biological markers for ASD risk could address these limitations by enabling earlier identification of at-risk children, facilitating timely intervention during critical periods of neurodevelopment. Early behavioral intervention has been shown to significantly improve cognitive, language, and adaptive outcomes in children with ASD. Furthermore, reliable biomarkers could help stratify the heterogeneous ASD population into more homogeneous subgroups, potentially enabling more targeted therapeutic approaches.  

## SUMMARY OF THE INVENTION  

The present invention provides methods for identifying a metabolomic signature associated with autism spectrum disorder (ASD) through comprehensive analysis of biosamples using multiple analytical platforms. In one embodiment, the method comprises assaying biosamples obtained from subjects diagnosed with ASD using gas chromatography-mass spectrometry (GC-MS) to generate first metabolic profiles. Parallel analysis is performed on biosamples from neurotypical control subjects using identical GC-MS parameters to generate control metabolic profiles. Statistical comparison of these profiles identifies metabolites demonstrating differential abundance between ASD and control groups.  

In a complementary approach, the invention employs untargeted liquid chromatography-high resolution mass spectrometry (LC/HRMS) to analyze biosamples from ASD and control subjects. This high-resolution analytical technique provides detection of a broad range of metabolites with precise mass measurement, enabling identification of additional differentially abundant metabolites not detected by GC-MS. The metabolites identified through both GC-MS and LC/HRMS analyses are combined to form a comprehensive panel of potential biomarkers.  

The invention further provides methods for selecting a subset of metabolites from this combined panel that demonstrate statistically significant abundance differences between ASD and control groups. Statistical methods including univariate analysis (e.g., Welch's t-tests with Benjamini-Hochberg correction for multiple comparisons) and multivariate machine learning approaches (e.g., support vector machines, partial least squares discriminant analysis) are employed to identify the most informative metabolites. The resulting metabolomic signature comprises a defined set of metabolites that collectively provide optimal discrimination between ASD and neurotypical individuals.  

In alternative embodiments, the invention provides methods for identifying metabolomic signatures through analysis using two or more complementary analytical methodologies, which may include but are not limited to: GC-MS, LC-MS with various chromatographic separations (e.g., C8 reverse phase, hydrophilic interaction chromatography), and different ionization modes (positive and negative electrospray ionization). Each methodology contributes unique metabolic information, and the combination of data from multiple platforms enhances the breadth and reliability of the resulting signature.  

The invention encompasses variations in the types of biosamples analyzed, including but not limited to blood plasma, serum, urine, cerebrospinal fluid, and saliva. Preferred embodiments utilize blood plasma due to its rich metabolic content and relative ease of collection. The molecular weight range of metabolites included in the signature spans from approximately 50 to 1500 Daltons, encompassing small molecule metabolites such as amino acids, organic acids, lipids, and other intermediary metabolites.  

The statistical methods for selecting metabolites may include various approaches such as univariate filtering based on p-values and fold change, multivariate analysis considering covariance between metabolites, and machine learning algorithms that evaluate the collective predictive power of metabolite combinations. The resulting signatures may comprise varying numbers of metabolites, with preferred embodiments including between 80 and 160 metabolites that demonstrate robust classification performance.  

Specific examples of metabolomic signatures disclosed include those comprising decreased levels of homocitrulline, increased levels of glutaric acid, increased saccharopine, increased 5-aminovaleric acid, increased lactate, and increased succinate. These metabolites represent various biochemical pathways including mitochondrial energy metabolism, amino acid metabolism, and gut microbiome-related pathways.  

The invention further provides methods for assessing risk of ASD in a subject by quantifying metabolites from a biosample and comparing the levels to reference values established from neurotypical controls. The assessment may involve analysis of individual metabolites, combinations of multiple metabolites, or ratios between specific metabolites. The methods are applicable to subjects across a broad age range, with particular utility for children between 2 and 6 years of age to enable early identification.  

Additional embodiments provide methods for stratifying ASD risk assessment based on phenotypic subpopulations, including differentiation between high-functioning and low-functioning autism presentations. The methods may incorporate clinical covariates such as cognitive function, language ability, or presence of comorbid conditions to refine metabolic risk assessment.  

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS  

The present invention provides a metabolomics-based approach for identifying metabolic biomarkers associated with autism spectrum disorder (ASD). This approach employs multiple high-resolution mass spectrometry techniques to comprehensively profile small molecule metabolites in biological samples. The analytical platforms include gas chromatography-mass spectrometry (GC-MS) and liquid chromatography-high resolution mass spectrometry (LC-HRMS), with the latter utilizing both reverse-phase (C8) and hydrophilic interaction liquid chromatography (HILIC) separations coupled to positive and negative electrospray ionization.  

The GC-MS method provides robust detection and quantification of volatile and semi-volatile metabolites, particularly those amenable to derivatization, including many organic acids, sugars, and amino acids. The LC-HRMS methods expand metabolite coverage to include a broader range of polar and nonpolar compounds, with high mass accuracy enabling precise molecular formula determination. Tandem mass spectrometry (MS/MS) methods are employed to confirm metabolite identities through fragmentation pattern analysis.  

Data acquisition for each sample involves multiple analytical runs to capture comprehensive metabolic information. For LC-HRMS analyses, samples are analyzed using four separate methods: C8 chromatography with positive ionization (C8pos), C8 chromatography with negative ionization (C8neg), HILIC with positive ionization (HILICpos), and HILIC with negative ionization (HILICneg). This orthogonal approach maximizes metabolite detection coverage across diverse chemical classes.  

The invention employs advanced statistical methods for data analysis, including univariate analysis to identify individual metabolites showing significant abundance differences between ASD and control groups, multivariate analysis to detect patterns of covariance among metabolites, and machine learning approaches to develop predictive models. Statistical models are developed using training sets of samples with known diagnoses, and model performance is validated using independent sample sets withheld from the model development process.  

Key analytical steps include:  
1. Data preprocessing to align retention times, group mass features, and normalize abundance values  
2. Quality control filtering to remove low-quality data and artifacts  
3. Univariate statistical testing to identify differentially abundant metabolites  
4. Multivariate modeling to evaluate metabolite combinations  
5. Machine learning approaches (e.g., support vector machines, partial least squares discriminant analysis) to develop classification algorithms  
6. Model validation using independent sample sets  
7. Metabolite identification through accurate mass measurement and MS/MS fragmentation  

The training set comprises biosamples from carefully characterized subjects, including both ASD and typically developing controls matched for age, sex, and other relevant covariates. The validation set serves as an independent test of model performance, providing estimates of classification accuracy, sensitivity, and specificity.  

The invention defines several key terms:  
- "Metabolite" refers to any small molecule (typically <1500 Da) involved in or produced by metabolism  
- "Feature" refers to a chromatographic peak defined by specific mass-to-charge ratio and retention time  
- "Biomarker" indicates a metabolite or combination of metabolites associated with a biological state  
- "Metabolic signature" denotes a defined set of metabolites that collectively characterize a biological condition  

The blood-based diagnostic methods disclosed offer significant advantages over current behavioral diagnosis, particularly in enabling earlier identification of ASD risk. Early diagnosis facilitates timely intervention during critical periods of neurodevelopment, potentially improving long-term outcomes.  

The metabolic biomarkers identified through this approach reflect underlying biochemical disturbances in ASD, including:  
- Mitochondrial dysfunction (evidenced by alterations in TCA cycle intermediates)  
- Amino acid metabolism abnormalities  
- Oxidative stress pathways  
- Gut microbiome influences  

Specific metabolites of interest include:  
- Decreased homocitrulline, suggesting altered urea cycle function  
- Increased glutaric acid, indicating possible lysine metabolism disruption  
- Altered branched-chain amino acids, reflecting potential mitochondrial dysfunction  
- Changes in fatty acid levels, possibly related to oxidative stress  

The invention provides metabolomic signatures comprising various numbers of metabolites, ranging from minimal sets of 5-10 key biomarkers to comprehensive panels of 80-160 metabolites. Larger signature panels generally provide higher classification accuracy, while smaller sets offer practical advantages for clinical implementation.  

Metabolite quantification can be performed using either absolute quantification (with internal standards) or relative quantification (compared to control samples). Statistical models may incorporate various approaches to combining metabolite data, including:  
- Simple abundance thresholds for individual metabolites  
- Weighted combinations of multiple metabolites  
- Ratios between specific metabolite pairs  
- Multivariate pattern recognition algorithms  

The invention encompasses kits for implementing the diagnostic methods, including:  
- Sample collection materials (e.g., blood collection tubes)  
- Metabolite extraction reagents  
- Internal standards for quantification  
- Chromatographic columns and mobile phases  
- Reference standards for key metabolites  
- Software for data analysis and interpretation  

Example kits may target specific metabolite panels, such as those including homocitrulline, glutaric acid, and other key biomarkers. The kits may be configured for use with different analytical platforms, including both physical separation methods (chromatography) and non-physical separation methods (direct infusion mass spectrometry).  

The diagnostic methods demonstrate strong performance characteristics, with example embodiments achieving:  
- Classification accuracy of 81% in independent validation  
- Sensitivity of 85% for ASD detection  
- Specificity of 75% for distinguishing from typical development  

These performance metrics exceed current genomic-based approaches for ASD risk assessment and provide a foundation for further refinement through larger validation studies.  

### EXAMPLES  

**Example 1: Subject Recruitment and Sample Collection**  
The study population comprised children aged 4-6 years, including 52 with ASD and 30 typically developing controls. ASD diagnosis was confirmed using DSM-IV criteria, ADOS, and ADI-R assessments. Exclusion criteria included known genetic syndromes, neurological conditions, and significant medical comorbidities. Blood samples were collected after overnight fast using standardized protocols to minimize pre-analytical variability. Plasma was separated and stored at -80°C until analysis.  

**Example 2: Metabolomic Profiling Methods**  
Five analytical methods were employed:  
1. GC-MS for analysis of volatile metabolites  
2. C8 LC-HRMS (positive ionization) for nonpolar metabolites  
3. C8 LC-HRMS (negative ionization) for acidic metabolites  
4. HILIC LC-HRMS (positive ionization) for polar metabolites  
5. HILIC LC-HRMS (negative ionization) for polar acidic metabolites  

LC-HRMS analyses used a QTOF mass spectrometer with mass accuracy <5 ppm. GC-MS used a TOF instrument with electron ionization.  

**Example 3: Data Processing and Analysis**  
Raw data were processed using XCMS for feature detection and alignment. Quality control filters removed low-quality features. Statistical analysis identified 179 differentially abundant features (p<0.05) between ASD and controls. Machine learning models were developed using these features, with the top 80-feature SVM model achieving 81% accuracy in independent validation.  

**Example 4: Key Metabolic Findings**  
Confirmed metabolite changes in ASD included:  
- Decreased homocitrulline (fold change 0.76, p=0.002)  
- Increased glutaric acid (fold change 1.32, p=0.008)  
- Increased saccharopine (fold change 1.28, p=0.01)  
- Altered TCA cycle intermediates (succinate, citrate)  
- Changes in branched-chain amino acids  

**Example 5: Pathway Analysis**  
Metabolites mapped to several key pathways:  
- Mitochondrial energy metabolism  
- Amino acid biosynthesis and degradation  
- Urea cycle  
- Fatty acid oxidation  
- Gut microbiome metabolism  

These findings support the involvement of multiple physiological systems in ASD pathophysiology and provide targets for further investigation and potential therapeutic intervention.  

The examples demonstrate the feasibility of metabolomic approaches for ASD biomarker discovery and highlight specific metabolites and pathways worthy of further investigation. The methods provide a framework for developing objective diagnostic tools and advancing understanding of ASD biology.