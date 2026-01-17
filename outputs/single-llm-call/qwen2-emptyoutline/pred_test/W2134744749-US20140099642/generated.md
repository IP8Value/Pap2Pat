# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of medical diagnostics, particularly to methods and systems for noninvasive prenatal testing (NIPT) using cell-free fetal DNA (cff-DNA) in maternal plasma. More specifically, the invention provides an advanced GC-correlation methodology for massively parallel sequencing (MPS)-based detection of fetal aneuploidies, including both autosomal and sex chromosomal abnormalities.

## BACKGROUND ART

Down syndrome (Trisomy 21), Edward syndrome (Trisomy 18), and Patau syndrome (Trisomy 13) are the most clinically significant autosomal aneuploidies, with an incidence as high as one in 160 live births. Turner’s syndrome (45, X), Klinefelter’s syndrome (47, XXY), and XYY syndrome are common sex chromosomal aneuploidies associated with reproductive loss, infertility, and language development delays. These conditions occur in one out of 500 male births and one out of 850 female births.

Conventional prenatal diagnostic methods, such as karyotyping, fluorescence in situ hybridization (FISH), and quantitative fluorescence-polymerase chain reaction (QF-PCR), rely on invasive procedures that carry potential risks for miscarriage. Noninvasive screening methods using maternal serum markers and ultrasound scans offer less risk but have limited sensitivity and specificity.

The discovery of cell-free fetal DNA (cff-DNA) in maternal plasma by Lo et al. in 1997 opened new avenues for noninvasive prenatal testing. Cff-DNA can be detected as early as four gestational weeks and clears rapidly from the maternal circulation after delivery. However, the low fraction of fetal DNA in maternal plasma (5% to 10%) makes it challenging to detect genetic variations in the fetus using conventional molecular techniques.

Recent advances in massively parallel sequencing (MPS) technology have enabled noninvasive detection of fetal aneuploidies in a clinical setting. Studies by Chiu et al. and Ehrich et al. have demonstrated the reliability of MPS-based approaches for detecting trisomy 21. However, these methods have shown less success in detecting trisomy 18 and trisomy 13, possibly due to GC-bias introduced during sample preparation or sequencing.

Quake et al. developed a method to correct for GC-bias, significantly improving the sensitivity of MPS-based approaches for detecting trisomy 18 and trisomy 13. Another study reported the possibility of detecting sex chromosomal aneuploidies using an internal chromosome control approach. Despite these advancements, there remains a need for a more robust and accurate method to detect both autosomal and sex chromosomal aneuploidies.

## SUMMARY OF THE INVENTION

The present invention addresses the limitations of existing methods by providing an advanced GC-correlation methodology for noninvasive fetal trisomy (NIFTY) testing using massively parallel sequencing (MPS). The invention includes a comprehensive bioinformatics pipeline that integrates short reads alignment, GC content correction, fetal DNA concentration estimation, t-test of a binary hypothesis, and fetal gender classification.

The invention comprises the following steps:

1. **Sample Preparation**: Collecting maternal blood samples, isolating plasma, and extracting cell-free DNA.
2. **Library Construction**: Constructing sequencing libraries from the extracted cell-free DNA.
3. **Sequencing**: Performing massively parallel sequencing of the libraries.
4. **Data Analysis**:
   - **Short Reads Alignment**: Aligning the sequencing reads to a reference genome.
   - **GC Content Correction**: Correcting for GC-bias in the sequencing data.
   - **Fetal DNA Concentration Estimation**: Estimating the concentration of fetal DNA in the maternal plasma.
   - **T-Test of a Binary Hypothesis**: Performing statistical tests to detect aneuploidies.
   - **Fetal Gender Classification**: Determining the sex of the fetus.

The invention also includes a computer-readable medium and a system for implementing the NIFTY test, enabling high sensitivity and specificity in the detection of both autosomal and sex chromosomal aneuploidies.

## DETAILED DESCRIPTION OF THE INVENTION

### I. DEFINITIONS

- **Cell-free fetal DNA (cff-DNA)**: DNA fragments derived from the fetus that circulate in the maternal bloodstream.
- **Massively Parallel Sequencing (MPS)**: High-throughput DNA sequencing technology capable of generating millions of reads simultaneously.
- **GC-bias**: Systematic errors in sequencing data due to the preferential amplification or sequencing of DNA fragments with certain GC content.
- **K-mer**: A substring of length k from a DNA sequence.
- **Relative Reads Coverage**: The ratio of the number of sequencing reads mapped to a specific chromosome to the total number of reads.
- **Standard Deviation (SD)**: A measure of the amount of variation or dispersion in a set of values.
- **Coefficient of Variation (CV)**: The ratio of the standard deviation to the mean, expressed as a percentage.
- **Binary Hypothesis**: A statistical test involving two mutually exclusive hypotheses, typically a null hypothesis and an alternative hypothesis.
- **Logarithmic Likelihood Odds Ratio**: A measure of the strength of evidence favoring one hypothesis over another.

### II. ESTABLISHING A RELATIONSHIP BETWEEN COVERAGE DEPTH AND GC CONTENT

To investigate the relationship between GC content and sequencing bias, we analyzed 300 control pregnancies with normal karyotypes. We plotted the relative reads coverage for each chromosome against the corresponding GC content. The relative reads coverage of different chromosomes was strongly related to the inherent chromosomal GC content, and the correlation varied among the chromosomes.

Chromosomes with an average GC content greater than 41% showed a significant positive correlation between reads coverage and GC content, while chromosomes with an average GC content less than 41% showed a significant negative correlation. Chromosomes with average GC content close to 41% showed no correlation between reads coverage and GC content.

To further investigate the effect of GC content on reads coverage, we classified all unique 35-mers in the genome into 36 levels based on the number of guanine (G) and cytosine (C) bases. We used the 35-mer counts to cluster the chromosomes according to their GC levels. Chromosomes 19 and 22 clustered together due to their higher inherent GC content, while chromosomes 4 and 13 clustered together due to their lower inherent GC content.

The differences in inherent GC content combined with sequencer-related GC-bias explained the significant correlation between reads coverage and corresponding GC content. For example, chromosome 13 has a relatively low GC content, and the PCR and sequencing process enriched chromosomes with higher GC content, leading to relatively low reads coverage for chromosome 13 and thus a negative correlation between reads coverage and GC content.

### III. DETERMINING A FETAL GENETIC ABNORMALITY

#### A. Sample Preparation and Library Construction

We recruited 903 pregnant women with ages ranging from 20 to 45 years. The gestational ages varied from 10 to 34 weeks, covering the first to the third trimesters. Based on the results of full karyotyping using amniotic fluid, 866 of the fetuses were euploid, and 37 were aneuploid. We obtained 2–4 million reads for each sample. After alignment and filtering, the average data volume for aneuploidy detection was 1.7 million uniquely aligned reads.

We collected 5 mL of peripheral venous blood from the pregnant women in EDTA tubes. The tubes were centrifuged at 1,600 × g for 10 minutes within four hours of collection. Plasma was transferred to microcentrifuge tubes and centrifuged at 16,000 × g for 10 minutes to remove residual cells. Cell-free plasma was stored at −80°C until DNA extraction. Each plasma sample was frozen and thawed only once.

For massively parallel genomic sequencing, DNA fragments from 600 μL of maternal plasma were used for library construction according to a modified protocol from Illumina. End-repairing of maternal plasma DNA fragments was performed using T4 DNA polymerase, Klenow polymerase, and T4 polynucleotide kinase. A-base tailing adapters were ligated to the DNA fragments. Standard multiplex primers were introduced by 17-cycle PCR. The libraries were analyzed for size distribution by Agilent Bioanalyzer and quantified using real-time PCR. Thirty-six-cycle single-end multiplex sequencing and 50-cycle single-end multiplex sequencing were used for the Illumina GAIIx and Illumina HiSeq 2000 platforms, respectively.

#### B. High-Effective Alignment with Universal Unique Reads Set

Computationally, we incised the human reference genome (HG 18, NCBI build 36) into k-mers (k refers to the length of the sequencing reads) and aligned the k-mers back to the reference genome. All k-mers that could be uniquely mapped to a single position on the reference genome, the unique mapping reads, were named the universal unique reads set. We selected the sequencing reads that could be mapped with 0-mismatch to the universal unique reads set (i.e., the tag) for our analysis.

#### C. K-Mer Coverage and GC-Correlation

We computed the k-mer coverage for each chromosome and every sample as \(C_{i,j} = \frac{n_{i,j}}{N_{i}}\), where \(i\) is the ID of control samples, \(j\) is the chromosome ID, \(n_{i,j}\) is the number of unique reads mapped onto chromosome \(j\) from sample \(i\), and \(N_{i}\) is the total number of unique reads for chromosome \(j\). Because of the differences among the samples, we normalized the data and computed the relative k-mer coverage for each sample as \(r_{i,j} = \frac{C_{i,j}}{C_{i}}\), where \(\bar{C}_{i} = \frac{1}{22} \sum_{j=1}^{22} C_{i,j}\) is the average k-mer coverage of the 22 autosomes in the \(i\)-th sample.

Given the unclear mechanism of GC-bias, we performed a Losses regression to fit the relative k-mer coverage to the corresponding GC content. We denoted the fitted relative k-mer coverage as \(c'_{r,i,j} = f_j(GC_{i,j})\). The fitted value, which we used as the theoretical value, was vital to our statistical model for cff-DNA concentration estimation and aneuploidy detection.

Because we used a male/female data set, we had different fitted values for the analysis of sex chromosomes. We calculated the fitted relative k-mer values for the sex chromosome analysis as follows:

- \(c'_{r,i,j,m} = f_{j,m}(GC_{i,j})\) for the fitted relative k-mer coverage from a regression of an adult male data set.
- \(c'_{r,i,j,f} = f_{j,f}(GC_{i,j})\) for the fitted relative k-mer coverage from a regression of a fetal-female data set.

#### D. Cff-DNA Concentration Estimation

Using the gender difference to compute the relative k-mer coverage of the sex chromosome, we estimated the cff-DNA concentrations, denoted as \(\varepsilon\). Subscripts corresponding to chromosome IDs indicate concentrations estimated from different chromosomes:

- \(\varepsilon_{i,Y} = \frac{c_{r,i,Y} - c'_{r,i,Y,f}}{c'_{r,i,Y,m} - c'_{r,i,Y,f}}\) is the estimation using the data for chromosome Y.
- \(\varepsilon_{i,X} = \frac{c_{r,i,X} - c'_{r,i,Y,f}}{c'_{r,i,Y,m} - c'_{r,i,Y,f}}\) is the estimation using data for chromosome X.

#### E. Autosomal Aneuploidy Detection with Binary Hypothesis

We developed a binary hypothesis strategy to achieve higher sensitivity and specificity. We performed two Student’s t-tests based on null/alternative hypotheses and subsequently calculated the relative logarithmic likelihood odds ratio. The null and alternative hypotheses are shown below.

For the first test:

- \(H_0\) (null hypothesis): The fetal chromosome is euploid.
- \(H_1\) (alternative hypothesis): The fetal chromosome is trisomic.

The first t-value is calculated as \(t_{i,j,\text{first}} = \frac{c_{r,i,j} - c'_{r,i,j}}{sd_j}\).

For the second test:

- \(H_0\) (null hypothesis): The test fetal chromosome is trisomic.
- \(H_1\) (alternative hypothesis): The test fetal chromosome is euploid.

The second t-value is calculated as \(t_{i,j,\text{second}} = \frac{c_{r,i,j} - c'_{r,i,j}(1 + \varepsilon_i / 2)}{sd_j}\).

The logarithmic likelihood odds ratio between our binary hypotheses is defined as:

\[ L_{i,j} = \log \left( \frac{p(t_{i,j,\text{first}}, \text{DOF} | H_0)}{p(t_{i,j,\text{second}}, \text{DOF} | H_1)} \right) \]

where DOF is the degree of freedom. We used \(|t_{i,j,\text{first}}| > 3\) and \(|t_{i,j,\text{second}}| < 3\) as warning criteria. From the logarithmic likelihood odds ratio, we could make a confident judgment of autosomal aneuploidy if \(L_{i,j} > 1\).

#### F. Fetal Gender Classification and Sex Chromosomal Aneuploidy Detection

We developed a double standard strategy with an experimental threshold and logistic regression to detect fetal gender. The k-mer coverage on chromosome Y was an ideal choice for distinguishing genders. Based on the 300 reference controls, we considered \(c_{r,i,Y} < 0.04\) the threshold for identifying a female fetus, while we regarded samples with \(c_{r,i,Y} > 0.051\) as having a male fetus. We considered samples with \(0.04 < c_{r,i,Y} < 0.051\) to be gender-uncertain.

Additionally, we developed a logistic regression strategy to improve the specificity of gender determination. We computed the probability (\(p_i\)) that a fetus was male by the following formula:

\[ \text{logit}(p_i) = \ln \left( \frac{p_i}{1 - p_i} \right) = \beta_0 + \beta_1 c_{r,i,X} + \beta_2 c_{r,i,Y} \]

where the parameters (\(\beta_0, \beta_1, \beta_2\)) were determined by regression using the 300 reference controls mentioned above.

We regarded samples with \(p_i > 0.8\) as having male fetuses, samples with \(p_i < 0.3\) as having female fetuses, and the remaining samples as being gender-uncertain.

After gender classification, we performed XXX and XO detection on samples with a female fetus and XXY and XYY detection on samples with a male fetus.

For samples with a female fetus, we performed a t-test for chromosome abnormality detection:

\[ t_{i,X} = \frac{c_{r,i,X} - c'_{r,i,X,f}}{sd_{X,f}} \]

where \(sd_{X,f}\) is the standard deviation of \(c_{r,i,X,f} - c'_{r,i,X,f}\) calculated from the reference controls with female fetuses; we expected \(sd_{X,f}\) to equal zero. We considered samples with \(t_{i,X} > 2.5\) or \(t_{i,X} < -2.5\) to be XXX or XO.

For a male fetus, we first supposed that chromosome Y is monosomic and extrapolated the fitted k-mer coverage for chromosome X, with the fetal DNA fraction estimated only by the k-mer coverage of chromosome Y. We calculated the t-score by the following formula:

\[ t_{i} = \frac{c_{r,i,X} - c'_{r,i,X,f}(1 - \frac{\varepsilon_{i,Y}}{2})}{sd_{X,f}} \]

where \(\varepsilon_{i,Y}\) is the estimated cff-DNA concentration using chromosome Y data, and \(sd_{X,f}\) is the standard deviation of \(c_{r,i,X,f} - c'_{r,i,X,f}\) calculated from the reference controls carrying female fetuses with an expectation of zero.

We regarded samples with \(t_i > 2.5\) as being XXY or XYY. Additionally, the cff-DNA concentration estimated by chromosome X and Y independently is a combined marker for sex chromosomal aneuploidy detection, especially XXY and XYY. For an XXY sample, not only was \(t_i > 2.5\) but also the cff-DNA concentration estimated by chromosome X was nearly zero, with a confidence interval from −0.03 to 0.03. For an XYY sample, not only was \(t_i > 2.5\), but the R-value (Ratio of the cff-DNA concentration estimated by chromosome Y to that estimated by chromosome X) was nearly two, reflecting the fact that there were two copies of chromosome Y and only a single copy of chromosome X.

### IV. COMPUTER READABLE MEDIUM AND SYSTEM FOR DIAGNOSIS OF A FETAL GENETIC ABNORMALITY

The present invention also includes a computer-readable medium and a system for implementing the NIFTY test. The computer-readable medium contains instructions that, when executed by a processor, cause the processor to perform the steps of the NIFTY test, including:

1. **Sample Preparation and Library Construction**: Instructions for collecting maternal blood samples, isolating plasma, and extracting cell-free DNA.
2. **Sequencing**: Instructions for performing massively parallel sequencing of the libraries.
3. **Data Analysis**:
   - **Short Reads Alignment**: Instructions for aligning the sequencing reads to a reference genome.
   - **GC Content Correction**: Instructions for correcting for GC-bias in the sequencing data.
   - **Fetal DNA Concentration Estimation**: Instructions for estimating the concentration of fetal DNA in the maternal plasma.
   - **T-Test of a Binary Hypothesis**: Instructions for performing statistical tests to detect aneuploidies.
   - **Fetal Gender Classification**: Instructions for determining the sex of the fetus.

The system for implementing the NIFTY test includes:

- **Sample Collection Module**: For collecting maternal blood samples.
- **DNA Extraction Module**: For isolating plasma and extracting cell-free DNA.
- **Library Construction Module**: For constructing sequencing libraries.
- **Sequencing Module**: For performing massively parallel sequencing.
- **Data Analysis Module**: For aligning reads, correcting for GC-bias, estimating fetal DNA concentration, performing t-tests, and classifying fetal gender.
- **Output Module**: For providing the results of the NIFTY test.

### V. EXAMPLES

#### Example 1: Detection of Trisomy 13, 18, and 21

We enrolled 903 pregnant women with ages ranging from 20 to 45 years. The gestational ages varied from 10 to 34 weeks. Based on the results of full karyotyping using amniotic fluid, 866 of the fetuses were euploid, and 37 were aneuploid. The cases of aneuploidy included two cases of trisomy 13, 12 cases of trisomy 18, and 16 cases of trisomy 21.

The NIFTY test performed with 100% sensitivity and specificity for the detection of trisomy 13 (two out of two) and trisomy 21 (16 out of 16). For trisomy 18, the NIFTY test detected 12 of 12 cases and identified 890 of 891 healthy controls, indicating 100% sensitivity and 99.7% specificity, corresponding to zero false negative results and a false positive rate of 0.3%.

#### Example 2: Detection of Sex Chromosomal Abnormalities

The NIFTY test correctly detected sex chromosomal abnormalities. For Turner’s syndrome, the NIFTY test identified three out of four XO cases but failed to detect the mosaic 45, X case, which was in gestational week 25 and had a normal karyotype in 46% of the cells sampled. Thus, the sensitivity and specificity of our approach for the detection of Turner’s syndrome were 75% and 99.9%, respectively; in other words, the false negative rate was 25% and the false positive rate was 0.1% for 45, X detection using the NIFTY test. The test performed with 100% sensitivity and specificity for the detection of XXY (two out of two) or XYY (one out of one).

The NIFTY test correctly identified the sex of approximately 99.9% of the 896 fetuses, 443 male and 452 female, which did not have sex chromosomal aneuploidies. The NIFTY test was inconclusive for one fetus that was determined to be 46, XX by karyotyping.

#### Example 3: Comparison with Other Aneuploidy Detection Approaches

To evaluate the performance of the NIFTY test in the detection of fetal aneuploidy, we compared it with the performance of three other previously reported approaches to analyze our 903 cases, with full karyotyping of the same 300 euploid cases. Chiu et al. used the standard z-score approach without any GC-bias removal to detect Down syndrome. Chen et al. developed a z-score approach with a different GC-bias removal strategy, which we named the “GC-correct z-score approach.” Lau et al. previously demonstrated an internal chromosome control-based z-score approach.

We used the coefficient of variation (CV) to evaluate the performances of these four approaches. The CV for the standard z-score approach was larger than that for other approaches among clinically relevant chromosomes (13, 18, and 21). Thus, the standard z-score approach has a low sensitivity for the detection of trisomies 13 and 18. The performance of the GC-correct z-score approaches and our NIFTY test were close, both demonstrating over 99% sensitivity and specificity for the detection of trisomy 13, 18, and 21. It was difficult to precisely detect sex chromosomal aneuploidies using the GC-correct z-score approach due to fetal gender confusion. The internal chromosome control approach displayed larger CV values for chromosomes 13, 18, and 21 and had a higher risk of false negatives related to XXY and XYY detection. In contrast, the NIFTY test had increased accuracy in the detection of sex chromosomal aneuploidies, such as XO, XXY, and XYY.

#### Example 4: Robust Data Quality Control of the GC-Correlation T-Test

Several indicators were used to judge the quality of the sequence data. We classified these indicators into two categories: direct and indirect. The indirect indicators of the accuracy of the NIFTY test came from the sequencing procedure: Q20% refers to the fraction of bases within the sequenced reads with an Illumina quality score greater than 20, and the PCR duplication rate refers to the fraction of the reads sharing the same start position and end positions on the reference genome. The direct indicators came from the data analysis procedure and included the number of unique reads, the genome-wide average GC content, and the consistency between the test samples and the reference controls.

The NIFTY test performed with 100% sensitivity and 99.9% specificity for the detection of autosomal aneuploidies and 85.7% sensitivity and 99.9% specificity for the detection of sex chromosomal aneuploidies. The quality-control procedure, which uses the estimation of cff-DNA concentration as a key index, improves the accuracy of fetal aneuploidy detection.

#### Example 5: Large-Scale Clinical Validation

To precisely estimate the sensitivity and specificity of our procedure, large-scale, multi-center clinical trials will be required in the future. The NIFTY test has been validated in a cohort of 903 participants, but the sample size in this study was a limiting factor because the incidence of aneuploidies in the general population is low. Future studies should focus on developing an unbiased method to precisely estimate the fraction of cff-DNA in the maternal plasma, especially for female fetuses.

The NIFTY test is a robust and accurate methodology for detecting fetal aneuploidies using MPS. This is the first study to systematically identify sex chromosomal aneuploidies with maternal plasma DNA sequencing. We hope the use of this method in clinical practices will contribute to reducing the number of birth defects.