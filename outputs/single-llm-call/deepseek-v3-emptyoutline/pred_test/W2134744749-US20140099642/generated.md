Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates generally to the field of non-invasive prenatal genetic testing. More specifically, the invention provides methods, systems, and computer-readable media for detecting fetal chromosomal abnormalities through analysis of cell-free fetal DNA in maternal plasma using massively parallel sequencing (MPS) technology. The invention particularly addresses the technical challenges of GC-content bias correction and statistical modeling to improve the accuracy of detecting both autosomal and sex chromosomal aneuploidies.  

## BACKGROUND ART  

Current prenatal diagnostic methods for detecting chromosomal abnormalities fall into two categories: invasive procedures and non-invasive screening. Invasive methods such as amniocentesis and chorionic villus sampling carry risks including miscarriage, while traditional non-invasive methods like maternal serum screening and ultrasound have limited sensitivity and specificity.  

The discovery of cell-free fetal DNA (cff-DNA) in maternal plasma opened new possibilities for non-invasive testing. However, the low fetal DNA fraction (typically 5-10% of total cell-free DNA) presents significant technical challenges for reliable detection of fetal chromosomal abnormalities. While massively parallel sequencing technologies have shown promise for detecting certain trisomies, existing approaches suffer from limitations including GC-content bias, inadequate statistical models, and difficulty detecting sex chromosomal abnormalities.  

Prior attempts to address these limitations have had mixed success. Some methods apply GC-bias correction but fail to account for the complex relationship between GC content and sequencing coverage across different chromosomes. Other approaches use oversimplified statistical models that reduce detection accuracy, particularly for sex chromosomes. There remains an unmet need for a comprehensive method that overcomes these limitations while maintaining high sensitivity and specificity across all clinically relevant chromosomal abnormalities.  

## SUMMARY OF THE INVENTION  

The present invention provides an advanced methodology for non-invasive detection of fetal chromosomal abnormalities that addresses the limitations of prior approaches. The method, termed NIFTY (Non-Invasive Fetal Trisomy) test, incorporates several key innovations:  

First, the invention establishes a sophisticated GC-correction model that accounts for the non-linear relationship between GC content and sequencing coverage. This model recognizes that chromosomes behave differently based on their inherent GC content, with positive correlation for high-GC chromosomes (>41% GC), negative correlation for low-GC chromosomes (<41% GC), and minimal correlation for chromosomes near the 41% threshold.  

Second, the invention implements a novel binary hypothesis statistical framework that improves detection accuracy. This approach performs two complementary t-tests (euploid vs. trisomic hypotheses) and calculates a logarithmic likelihood odds ratio for confident aneuploidy determination.  

Third, the invention provides specialized methods for sex chromosomal abnormality detection, including a double-threshold gender classification system and differential analysis of X and Y chromosome data. This enables detection of conditions such as Turner syndrome (45,X), Klinefelter syndrome (47,XXY), and XYY syndrome with high accuracy.  

The complete system includes optimized laboratory protocols for plasma DNA processing, massively parallel sequencing procedures, and a comprehensive bioinformatics pipeline. Clinical validation with 903 samples demonstrated 100% sensitivity for trisomies 13, 18, and 21, and 85.7% sensitivity for sex chromosomal abnormalities, with specificity exceeding 99% in all cases.  

## DETAILED DESCRIPTION OF THE INVENTION  

### I. DEFINITIONS  

As used throughout this specification, the following terms shall have the meanings specified:  

"Cell-free fetal DNA (cff-DNA)" refers to extracellular fetal DNA fragments present in maternal plasma, typically ranging from approximately 100-300 base pairs in length.  

"Massively parallel sequencing (MPS)" refers to high-throughput DNA sequencing technologies capable of simultaneously determining millions to billions of DNA sequences, including but not limited to Illumina sequencing platforms.  

"GC content" means the percentage of nitrogenous bases in a DNA sequence that are either guanine (G) or cytosine (C).  

"Relative reads coverage" refers to the normalized representation of a chromosome's sequence data compared to the genome-wide average, calculated as the observed reads count for a chromosome divided by the average reads count across all autosomes.  

"Z-score" represents the number of standard deviations a particular measurement differs from the expected mean value in a reference population.  

"Binary hypothesis testing" refers to the statistical comparison of two competing models (euploid vs. aneuploid) to determine which better explains the observed sequencing data.  

### II. ESTABLISHING A RELATIONSHIP BETWEEN COVERAGE DEPTH AND GC CONTENT  

The invention establishes a sophisticated model of the relationship between sequencing coverage depth and GC content that accounts for chromosome-specific behaviors. Analysis of 300 control pregnancies revealed three distinct patterns:  

For chromosomes with average GC content exceeding 41% (e.g., chromosomes 19 and 22), sequencing coverage shows a strong positive correlation with GC content. This reflects preferential amplification of GC-rich regions during library preparation and sequencing.  

For chromosomes with average GC content below 41% (e.g., chromosomes 4 and 13), sequencing coverage demonstrates a significant negative correlation with GC content. This inverse relationship results from underrepresentation of AT-rich sequences in the sequencing process.  

Chromosomes with GC content near 41% (e.g., chromosome 21) show minimal correlation between coverage and GC content, as they fall near the equilibrium point of these competing effects.  

The invention employs LOESS (Locally Estimated Scatterplot Smoothing) regression to model these relationships, creating chromosome-specific correction factors. This approach differs from prior global GC-correction methods by recognizing that different chromosomes require distinct correction models based on their inherent GC characteristics.  

The GC-correction model is built using a universal unique reads set - a collection of sequences that map unambiguously to single genomic locations. This eliminates alignment ambiguity and provides a consistent basis for GC-content analysis across samples.  

### III. DETERMINING A FETAL GENETIC ABNORMALITY  

The invention provides a comprehensive statistical framework for fetal aneuploidy detection comprising three main components: fetal DNA fraction estimation, autosomal abnormality detection, and sex chromosomal abnormality detection.  

Fetal DNA fraction estimation utilizes the differential representation of X and Y chromosomes between male and female fetuses. For male fetuses, the concentration (ε) is calculated as:  

εi,Y = (cri,Y - cr'i,Y,f)/(cr'i,Y,m - cr'i,Y,f)  

where cri,Y is the observed relative coverage of chromosome Y, cr'i,Y,f is the expected female value, and cr'i,Y,m is the expected male value. Similar calculations using chromosome X data provide cross-validation.  

Autosomal abnormality detection employs a novel binary hypothesis approach. Two t-tests are performed:  

1. H0: Euploid vs. H1: Trisomic  
ti,j,first = (cri,j - cr'i,j)/sdj  

2. H0: Trisomic vs. H1: Euploid  
ti,j,second = (cri,j - cr'i,j(1 + εi/2))/sdj  

The logarithmic likelihood odds ratio (Li,j) between these hypotheses provides a robust aneuploidy call when Li,j > 1.  

Sex chromosomal abnormality detection involves:  

1. Precise fetal gender determination using both threshold-based (cri,Y < 0.04 for female, >0.051 for male) and logistic regression methods.  

2. For female fetuses: XXX/XO detection via t-test comparing X chromosome coverage to female references.  

3. For male fetuses: XXY/XYY detection through combined analysis of X and Y chromosome data and concordance of fetal fraction estimates.  

### IV. COMPUTER READABLE MEDIUM AND SYSTEM FOR DIAGNOSIS OF A FETAL GENETIC ABNORMALITY  

The invention provides a computer system comprising:  

1. A sequencing interface module that receives raw sequencing data from MPS platforms.  

2. An alignment module that maps sequences to a reference genome using the universal unique reads set.  

3. A GC-correction module that applies chromosome-specific coverage normalization based on LOESS regression models.  

4. A statistical analysis module that performs fetal fraction estimation and binary hypothesis testing.  

5. A classification module that determines chromosomal abnormality status and generates diagnostic reports.  

The system is implemented on non-transitory computer-readable media containing executable instructions that, when processed by one or more processors, perform the analytical methods described herein. The system architecture allows for parallel processing of multiple samples and integration with laboratory information management systems.  

### V. EXAMPLES  

Example 1: Detection of Trisomy 21  

Analysis of a maternal plasma sample from gestational week 18 showed:  

- chr21 relative coverage: 1.08 (expected euploid: 1.00)  
- Binary hypothesis testing:  
  - tfirst = 4.2 (p < 0.0001)  
  - tsecond = -3.1 (p = 0.002)  
  - Likelihood ratio: 8.7  

The sample was correctly identified as trisomy 21, confirmed by karyotyping.  

Example 2: Detection of 47,XXY (Klinefelter Syndrome)  

Analysis of a maternal plasma sample showed:  

- Gender classification: male (cri,Y = 0.12, logistic p = 0.99)  
- X chromosome analysis:  
  - t-score = 3.4  
  - εX ≈ 0 (consistent with XXY)  
- Diagnosis: 47,XXY, confirmed by postnatal testing.  

Example 3: Quality Control Failure  

One sample showed:  

- chr18 relative coverage: 1.15  
- Low fetal fraction (ε = 2.1%)  
- Binary hypothesis testing inconclusive (Li,j = 0.3)  

The system flagged this as a quality control failure due to insufficient fetal DNA, preventing a false positive call.  

The complete methodology has been validated on 903 clinical samples with the following performance:  

- Trisomy 21: 100% sensitivity (16/16), 100% specificity  
- Trisomy 18: 100% sensitivity (12/12), 99.7% specificity  
- Trisomy 13: 100% sensitivity (2/2), 100% specificity  
- 45,X: 75% sensitivity (3/4), 99.9% specificity  
- 47,XXY/47,XYY: 100% sensitivity (3/3), 100% specificity  

These results demonstrate significant improvement over prior methods, particularly for sex chromosome abnormalities where conventional approaches often fail.