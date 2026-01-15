# DESCRIPTION

## TECHNICAL FIELD

- introduce noninvasive fetal genetic abnormality detection

The present invention relates to a method and system for the noninvasive detection of fetal genetic abnormalities through the analysis of cell-free nucleic acids present in maternal plasma. Specifically, the invention provides a robust, statistically grounded approach for identifying autosomal and sex chromosomal aneuploidies without the need for invasive procedures such as amniocentesis or chorionic villus sampling. The method leverages massively parallel sequencing of maternal plasma DNA fragments, coupled with a novel correction algorithm that accounts for systematic biases introduced during library preparation and sequencing, particularly those arising from variations in guanine-cytosine (GC) content across the genome. By establishing a precise quantitative relationship between sequencing coverage depth and local GC composition, the invention enables the detection of subtle deviations indicative of fetal chromosomal gains or losses with high sensitivity and specificity. This technology is applicable across all gestational stages and is particularly suited for the early identification of conditions such as trisomy 13, trisomy 18, trisomy 21, Turner syndrome, Klinefelter syndrome, and XYY syndrome, thereby offering a clinically viable alternative to conventional diagnostic paradigms that carry inherent risks to both mother and fetus.

## BACKGROUND ART

- motivate noninvasive prenatal diagnosis

Prenatal diagnosis of chromosomal abnormalities has long been a cornerstone of reproductive medicine, aiming to identify conditions that may significantly impact fetal development, neonatal survival, and long-term health outcomes. The clinical imperative for early and accurate detection is underscored by the high prevalence of certain aneuploidies, including trisomy 21, which occurs in approximately one in every 700 live births, and sex chromosome aneuploidies such as 45,X and 47,XXY, which collectively affect approximately one in 500 male and one in 850 female newborns. Traditional diagnostic methods, while highly accurate, rely on invasive sampling techniques that carry a non-negligible risk of procedure-related miscarriage, typically estimated between 0.1% and 0.5%. These risks have motivated the development of noninvasive screening strategies that minimize maternal and fetal harm while maintaining diagnostic fidelity.

- describe limitations of conventional methods

Conventional noninvasive screening methods, such as maternal serum biomarker analysis combined with ultrasound measurements, offer low-risk alternatives but suffer from limited sensitivity and specificity, resulting in high false positive and false negative rates. These limitations necessitate confirmatory invasive testing for a substantial proportion of screened pregnancies, thereby undermining the clinical utility of noninvasive approaches. Furthermore, these methods are unable to detect most sex chromosome aneuploidies with any degree of reliability, leaving a critical gap in prenatal diagnostic capabilities.

- introduce noninvasive detection of fetal aneuploidy

The discovery that fetal-derived cell-free DNA circulates in maternal plasma during pregnancy revolutionized the field of prenatal diagnostics. First identified in the late 1990s, this fetal nucleic acid material is shed primarily from placental trophoblasts and can be detected as early as four weeks of gestation. Its rapid clearance from maternal circulation following delivery confirms its fetal origin and provides a transient, noninvasive window into fetal genomic composition. This biological phenomenon has enabled the development of molecular assays capable of detecting fetal chromosomal imbalances without direct fetal sampling.

- describe detection of fetal DNA in maternal plasma

Cell-free fetal DNA fragments in maternal plasma are typically short, averaging 140 to 160 base pairs in length, and constitute a minority fraction of the total circulating DNA, often ranging between 5% and 15% depending on gestational age and maternal factors. Despite this low abundance, advances in high-throughput sequencing technologies have made it feasible to sequence millions of these fragments in a single run, allowing for genome-wide analysis of relative chromosomal representation.

- discuss limitations of circulating fetal DNA

However, the low fetal DNA fraction introduces significant analytical challenges, as the signal of aneuploidy must be discerned against a background dominated by maternal DNA. Moreover, technical artifacts introduced during library construction and sequencing—particularly GC content-dependent biases—can distort coverage profiles across chromosomes, leading to spurious signals that mimic or mask true aneuploidies. These biases are not random but systematically correlate with the inherent GC composition of genomic regions, resulting in underrepresentation of low-GC chromosomes such as 13 and 18 and overrepresentation of high-GC chromosomes such as 19 and 22.

- introduce GC bias in sequencing data

The presence of GC bias has been a persistent obstacle in the accurate interpretation of sequencing data for noninvasive prenatal testing. Conventional approaches that rely on z-score normalization, which assumes uniform sequencing efficiency across all genomic regions, are particularly vulnerable to this artifact, resulting in reduced sensitivity for detecting trisomies 13 and 18, where the fetal signal is already attenuated by low baseline coverage.

- describe methods to remove GC bias

Previous attempts to mitigate GC bias have included normalization based on sliding window averages, polynomial fitting, or reference-based correction using euploid controls. However, these methods often fail to account for the non-linear and chromosome-specific nature of the relationship between GC content and sequencing depth, leading to residual noise and diminished diagnostic accuracy.

- motivate need for improved method

There remains a critical and unmet need for a method that not only corrects for GC bias with high precision but also integrates this correction into a statistically rigorous framework capable of distinguishing true fetal aneuploidies from background variability. The present invention addresses this need by introducing a novel analytical pipeline that models the relationship between coverage depth and GC content on a chromosome-by-chromosome basis, using a locally weighted regression algorithm to derive expected coverage profiles, and applying a dual-hypothesis t-test to detect deviations that are both statistically significant and biologically consistent with fetal aneuploidy.

## SUMMARY OF THE INVENTION

- introduce method for noninvasive detection of fetal genetic abnormalities

The present invention provides a method for the noninvasive detection of fetal genetic abnormalities by analyzing sequencing data derived from cell-free nucleic acids in maternal plasma. The method enables the identification of autosomal and sex chromosomal aneuploidies with high sensitivity and specificity by accounting for systematic biases inherent in massively parallel sequencing platforms.

- describe removal of GC bias from sequencing results

The invention employs a statistical framework that models the relationship between sequencing coverage depth and the GC content of genomic regions to remove systematic biases. By fitting observed coverage profiles to predicted profiles derived from a reference cohort of euploid pregnancies, the method effectively isolates deviations attributable to fetal chromosomal imbalances rather than technical artifacts.

- establish relationship between coverage depth and GC content

A fundamental insight of the invention is that the depth of sequencing coverage for any given chromosomal segment is intrinsically correlated with its local GC content, and this correlation varies systematically across chromosomes due to differences in genomic architecture and sequencing chemistry. The invention exploits this predictable relationship to establish chromosome-specific baseline expectations for coverage.

- obtain sequence information of polynucleotide fragments

The method begins with the extraction of cell-free nucleic acids from maternal plasma, followed by library preparation and massively parallel sequencing to generate millions of short polynucleotide fragments. These fragments are then aligned to a reference human genome to determine their genomic origin.

- assign fragments to chromosomes based on sequence information

Each sequenced fragment is assigned to a specific chromosome based on its sequence complementarity to uniquely mappable regions of the reference genome. Only fragments that align with perfect or near-perfect identity to a single genomic locus are retained to minimize ambiguity.

- calculate coverage depth and GC content

For each chromosome, the total number of aligned fragments is counted to determine coverage depth, while the GC content of each fragment’s genomic region is computed as the proportion of guanine and cytosine bases within a defined window surrounding the fragment’s alignment site.

- determine relationship between coverage depth and GC content

A non-linear relationship between coverage depth and GC content is established for each chromosome using a locally weighted scatterplot smoothing algorithm, which generates a fitted coverage curve that accounts for the complex, non-parametric nature of the bias.

- calculate fitted coverage depth

The fitted coverage depth for each chromosomal region is derived from the smoothing model and represents the expected coverage under conditions of euploidy and no technical bias.

- calculate standard variation

The standard deviation of residuals between observed and fitted coverage values is computed across a reference cohort of euploid pregnancies to quantify the natural variability of the system under normal conditions.

- calculate student t-statistic

A Student’s t-statistic is calculated for each chromosome in the test sample by comparing the observed coverage depth to the fitted coverage depth, normalized by the standard deviation derived from the reference cohort. This statistic quantifies the likelihood that the observed deviation is due to chance.

- describe GC content calculation

The GC content of each genomic region is calculated as the percentage of guanine and cytosine nucleotides within a fixed-length window centered on the mapped fragment, with windows sized to match the average fragment length and ensure sufficient resolution.

- describe use of multiple samples

The method utilizes a reference panel of multiple euploid maternal plasma samples to establish robust, population-based models of coverage-GC relationships, ensuring that the statistical thresholds are calibrated to biological and technical variability across diverse individuals.

- describe use of pregnant female subjects

The method is specifically designed for use with biological samples obtained from pregnant female subjects, wherein the cell-free nucleic acid fraction contains a mixture of maternal and fetal DNA, and the fetal contribution is the target of analysis.

- describe use of biological samples

The biological samples used in the method are peripheral blood samples collected from pregnant women, from which plasma is separated and cell-free DNA is extracted for sequencing. The samples are processed under standardized conditions to ensure reproducibility and minimize pre-analytical variability.

- introduce method to determine fetal genetic abnormality

The method determines the presence of a fetal genetic abnormality by comparing the observed coverage depth of each chromosome to its fitted coverage depth, using a statistical hypothesis test that evaluates whether the deviation exceeds the expected range of variation under euploidy.

- obtain sequence information of polynucleotide fragments

Sequence information is obtained from the fragmented cell-free DNA molecules in maternal plasma through high-throughput sequencing, generating a comprehensive dataset of genomic coordinates and sequence reads.

- assign fragments to chromosomes based on sequence information

Each read is mapped to its chromosomal origin using a reference genome, and only uniquely aligned reads are retained to ensure accurate assignment and eliminate ambiguity from repetitive regions.

- calculate coverage depth and GC content

Coverage depth is calculated as the number of reads aligned to a given chromosome, normalized by the total number of reads in the sample. GC content is computed for each read’s genomic context using a sliding window approach.

- calculate fitted coverage depth

A fitted coverage depth value is generated for each chromosome using a locally weighted regression model trained on reference samples, which predicts the expected coverage based on the chromosome’s GC content profile.

- compare fitted coverage depth to coverage depth

The observed coverage depth for each chromosome in the test sample is compared to its corresponding fitted coverage depth to identify significant deviations that are inconsistent with euploidy.

- determine fetal gender

Fetal gender is determined by analyzing the relative coverage of the X and Y chromosomes. The presence of Y-chromosome reads indicates a male fetus, while their absence, combined with appropriate X-chromosome coverage, indicates a female fetus.

- estimate fetal fraction

The fetal DNA fraction is estimated by comparing the observed coverage of the sex chromosomes to their fitted coverage values under male and female fetal models, using a ratio-based formula that accounts for the expected contribution of fetal DNA to the total signal.

- describe calculation of fetal fraction

The fetal fraction is calculated as the proportion of sequencing signal attributable to fetal DNA, derived from the difference between observed and maternal-fetal baseline coverage on the Y chromosome for male fetuses, and on the X chromosome for female fetuses, normalized by the expected difference between maternal and fetal contributions.

- describe statistical hypothesis test

A two-sided Student’s t-test is applied to evaluate the null hypothesis that the fetus is euploid against the alternative hypothesis that the fetus is aneuploid. The test statistic is computed using the difference between observed and fitted coverage, divided by the standard deviation of residuals from the reference cohort.

- introduce computer readable medium and system for determining fetal genetic abnormality

The invention further encompasses a computer-readable medium containing executable instructions that, when executed by a computing system, implement the method for determining fetal genetic abnormalities. The system includes components for receiving sequencing data, assigning fragments to chromosomes, computing coverage and GC content, generating fitted coverage profiles, and performing statistical comparisons to determine aneuploidy status.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce noninvasive detection of fetal genetic abnormalities

The invention provides a comprehensive, noninvasive method for detecting fetal chromosomal abnormalities through the analysis of cell-free DNA in maternal plasma. This method overcomes the limitations of prior approaches by integrating a sophisticated correction for GC bias with a statistically rigorous framework for anomaly detection, enabling accurate identification of both autosomal and sex chromosome aneuploidies across a broad gestational window.

- describe method to remove GC bias from sequencing results

The method removes GC bias by modeling the relationship between sequencing coverage depth and local GC content on a chromosome-by-chromosome basis. Rather than applying a global correction factor, the invention uses a locally weighted regression algorithm to fit a smooth, non-linear curve that captures the unique bias profile of each chromosome. This fitted curve serves as the expected coverage under euploid conditions, allowing deviations to be interpreted as potential aneuploidies rather than technical artifacts.

### I. DEFINITIONS

- define technical and scientific terms

For the purposes of this disclosure, the term “cell-free DNA” refers to fragmented nucleic acid molecules circulating in maternal plasma that are derived from placental trophoblasts and reflect fetal genomic content. The term “massively parallel sequencing” refers to high-throughput sequencing technologies capable of generating millions of short DNA sequence reads in a single run, including but not limited to Illumina sequencing platforms.

- specify singular forms include plural references

All singular terms used herein, including but not limited to “sample,” “chromosome,” “fragment,” and “read,” shall be understood to encompass their plural forms unless explicitly restricted by context.

- define chromosomal abnormality

A chromosomal abnormality refers to a deviation from the normal diploid complement of chromosomes, including trisomies, monosomies, and other numerical or structural anomalies such as 45,X, 47,XXY, and 47,XYY.

- define reference unique reads

Reference unique reads are sequencing fragments that align with perfect or near-perfect identity to a single, non-repetitive locus in the reference human genome, ensuring unambiguous chromosomal assignment.

- define polynucleotide, oligonucleotide, nucleic acid, and nucleic acid molecule

A polynucleotide is a linear polymer composed of nucleotide monomers linked by phosphodiester bonds. An oligonucleotide is a shorter polynucleotide, typically less than 200 nucleotides in length. A nucleic acid refers to any molecule composed of nucleotide units, including DNA and RNA. A nucleic acid molecule encompasses both natural and synthetic variants, including modified forms such as methylated or fragmented DNA.

- describe massively parallel sequencing

Massively parallel sequencing is a high-throughput DNA sequencing technology that enables the simultaneous determination of millions of short DNA sequences, allowing for genome-wide coverage and quantitative analysis of relative fragment abundance.

- define biological sample

A biological sample, as used herein, refers to a fluid or tissue specimen obtained from a subject, including but not limited to peripheral blood, plasma, serum, or amniotic fluid, from which cell-free nucleic acids can be extracted for analysis.

- describe aspects and embodiments of the invention

The invention includes multiple embodiments, including methods for detecting autosomal trisomies, sex chromosome aneuploidies, and fetal gender, as well as systems and computer-readable media for automating the analytical pipeline.

- define monosomy X

Monosomy X, also known as Turner syndrome, is a chromosomal condition characterized by the presence of a single X chromosome and the absence of a second sex chromosome, resulting in a 45,X karyotype.

- define XXY syndrome

XXY syndrome, also known as Klinefelter syndrome, is a chromosomal condition characterized by the presence of two X chromosomes and one Y chromosome, resulting in a 47,XXY karyotype.

- define XYY syndrome

XYY syndrome is a chromosomal condition characterized by the presence of one X chromosome and two Y chromosomes, resulting in a 47,XYY karyotype.

- describe trisomy 13, trisomy 18, and trisomy 21

Trisomy 13, also known as Patau syndrome, is a chromosomal condition caused by the presence of three copies of chromosome 13. Trisomy 18, also known as Edwards syndrome, is caused by three copies of chromosome 18. Trisomy 21, also known as Down syndrome, is caused by three copies of chromosome 21. All three conditions are associated with severe developmental abnormalities and increased perinatal mortality.

- define Turner syndrome

Turner syndrome is synonymous with monosomy X and refers to the complete or partial absence of one X chromosome in a female individual.

- define Klinefelter syndrome

Klinefelter syndrome is synonymous with XXY syndrome and refers to the presence of an extra X chromosome in a male individual.

- describe detection of fetal chromosomal aberration

Detection of fetal chromosomal aberration involves identifying deviations in the relative abundance of sequencing reads assigned to specific chromosomes, indicating an excess or deficit of genetic material consistent with aneuploidy.

- define terms related to nucleic acid molecules

Terms such as “fragment,” “read,” “alignment,” and “coverage” are used in accordance with standard genomic terminology, referring to discrete units of sequence data and their mapping to a reference genome.

- describe types of modifications to nucleic acid molecules

Nucleic acid molecules may be subject to chemical modifications such as methylation, fragmentation, or adapter ligation during sample preparation, none of which interfere with the method’s ability to detect aneuploidy.

- describe other types of nucleic acid molecules

Other nucleic acid molecules, including RNA or synthetic DNA analogs, are not required for the practice of this invention, which is specifically directed to the analysis of cell-free DNA.

### II. ESTABLISHING A RELATIONSHIP BETWEEN COVERAGE DEPTH AND GC CONTENT

- obtain sequence information of multiple polynucleotide fragments

Sequence information is obtained from a cohort of maternal plasma samples derived from euploid pregnancies, ensuring that the reference model reflects biological and technical variability under normal conditions.

- assign fragments to chromosomes based on sequence information

Each fragment is uniquely mapped to a single chromosomal location using a reference genome assembly, and only fragments with perfect or near-perfect alignment are retained to eliminate ambiguity.

- calculate coverage depth and GC content of a chromosome

Coverage depth is calculated as the number of aligned fragments per chromosome, normalized by the total number of mapped fragments in the sample. GC content is computed for each fragment’s genomic context using a sliding window centered on the alignment site.

- determine the relationship between coverage depth and GC content

A non-linear relationship between coverage depth and GC content is determined using a locally weighted scatterplot smoothing algorithm, which generates a continuous, chromosome-specific curve that models the expected coverage as a function of GC content.

- describe calculation of coverage depth

Coverage depth for a chromosome is calculated as the sum of all uniquely mapped fragments assigned to that chromosome, divided by the total number of uniquely mapped fragments in the sample, yielding a normalized measure of relative abundance.

- describe normalization of coverage depth

Normalization is performed to account for differences in total sequencing depth across samples, ensuring that coverage values are comparable between individuals regardless of sequencing yield.

- calculate relative coverage depth

Relative coverage depth is computed as the ratio of the coverage depth of a specific chromosome to the average coverage depth across all autosomes in the same sample.

- describe calculation of GC content

GC content is calculated as the percentage of guanine and cytosine nucleotides within a fixed-length window surrounding the alignment site of each fragment, with window size selected to match the average fragment length.

- establish a relationship between coverage depth and GC content

The relationship between coverage depth and GC content is established by fitting the observed data using a Loess algorithm, which generates a smooth, non-parametric curve that captures the complex, non-linear dependence of coverage on GC content for each chromosome.

- use Loess algorithm to assess non-linear relationships

The Loess algorithm is employed to model the relationship between coverage depth and GC content without assuming a predefined functional form, allowing the method to adapt to chromosome-specific bias patterns.

- describe use of multiple samples to establish a relationship

A reference cohort of at least 100 euploid maternal plasma samples is used to train the Loess model, ensuring that the fitted curve reflects population-level variability and is robust to individual outliers.

### III. DETERMINING A FETAL GENETIC ABNORMALITY

- obtain sequence information of multiple polynucleotide fragments

Sequence data from a test sample is obtained using the same methodology as the reference cohort, ensuring consistency in data generation and analysis.

- assign fragments to chromosomes based on sequence information

Fragments are assigned to chromosomes using the same alignment criteria as the reference model, ensuring comparability and minimizing technical variability.

- calculate coverage depth and GC content of a chromosome

Coverage depth and GC content are calculated for each chromosome in the test sample using identical parameters and window sizes as those used in the reference model.

- compare fitted coverage depth to coverage depth of a chromosome

The observed coverage depth for each chromosome is compared to its corresponding fitted coverage depth, derived from the Loess model trained on the reference cohort.

- determine fetal genetic abnormality based on comparison

A fetal genetic abnormality is determined when the deviation between observed and fitted coverage exceeds a statistically defined threshold, as quantified by a Student’s t-test, with a p-value below a predetermined significance level indicating aneuploidy.

### IV. COMPUTER READABLE MEDIUM AND SYSTEM FOR DIAGNOSIS OF A FETAL GENETIC ABNORMALITY

- receive sequence information

A computing system receives raw sequencing data in the form of aligned reads and their associated genomic coordinates.

- assign polynucleotide fragments to chromosomes

The system automatically assigns each fragment to its corresponding chromosome based on alignment to the reference genome.

- calculate coverage depth and GC content of a chromosome

The system computes coverage depth and GC content for each chromosome using predefined algorithms and parameters.

- compare fitted coverage depth to coverage depth of a chromosome

The system retrieves the fitted coverage profile for each chromosome from a precomputed reference model and compares it to the observed coverage depth from the test sample.

### V. EXAMPLES

- introduce example 1

Example 1 demonstrates the correlation between chromosomal GC content and sequencing coverage depth across a cohort of 300 euploid pregnancies, revealing that chromosomes with higher GC content, such as chromosome 19, exhibit significantly greater coverage than chromosomes with lower GC content, such as chromosome 13.

- analyze factors affecting sensitivity of detection

Factors influencing sensitivity include gestational age, fetal DNA fraction, sequencing depth, and the precision of GC bias correction, all of which are quantified and optimized in the method.

- describe procedural framework for calculating coverage depth and GC content

The procedural framework involves alignment, normalization, window-based GC computation, and Loess fitting, all implemented in a standardized bioinformatics pipeline.

- illustrate correlation between coverage depth and GC content

A scatterplot of coverage depth versus GC content for each chromosome reveals a distinct, chromosome-specific trend, with positive correlation for high-GC chromosomes and negative correlation for low-GC chromosomes.

- explain influence of GC content on coverage depth

The influence of GC content on coverage depth arises from preferential amplification and sequencing efficiency during library preparation, where regions with intermediate GC content are more efficiently represented.

- describe composition of GC content in different chromosomes

Chromosomes vary widely in their average GC content, with chromosome 19 having approximately 55% GC content and chromosome 13 having approximately 38%, leading to differential representation in sequencing data.

- analyze influence of fetal gender on data

Fetal gender influences the coverage profile of sex chromosomes, with male fetuses contributing Y-chromosome reads and altering the relative abundance of X-chromosome signals.

- introduce example 2

Example 2 details the application of the Loess algorithm to fit coverage depth as a function of GC content for each chromosome, demonstrating that the fitted curves accurately predict coverage across a wide range of GC values.

- describe statistical model for coverage depth and GC content

The statistical model treats coverage depth as a dependent variable and GC content as an independent variable, with the Loess fit serving as the expected value under euploidy.

- apply loess algorithm to fit coverage depth with GC content

The Loess algorithm is applied with a bandwidth parameter optimized to balance smoothness and sensitivity, ensuring that the fitted curve captures true biological trends without overfitting noise.

- calculate fitted coverage depth and standard variance

For each chromosome, the fitted coverage depth and the standard deviation of residuals are calculated from the reference cohort, forming the basis for subsequent statistical testing.

- introduce example 3

Example 3 demonstrates the estimation of fetal DNA fraction using the differential coverage of the X and Y chromosomes in male and female fetuses, with formulas derived from the fitted coverage profiles.

- describe formulas for estimating fetal fraction

The fetal fraction is estimated using the formula: ε = (observed Y coverage − fitted female Y coverage) / (fitted male Y coverage − fitted female Y coverage), where ε represents the proportion of fetal DNA.

- introduce example 4

Example 4 presents the calculation of residuals for each chromosome in the test sample, showing that residual variance is stable across samples with sufficient sequencing depth.

- analyze standard variation of every chromosome

The standard deviation of residuals is computed per chromosome and found to be inversely proportional to sequencing depth, with lower variation observed at higher read counts.

- introduce example 5

Example 5 demonstrates the classification of fetal gender using a logistic regression model trained on coverage values from the X and Y chromosomes.

- describe logistic regression for predicting gender

Logistic regression is applied to predict the probability of a male fetus based on the combined coverage of X and Y chromosomes, achieving over 99.9% accuracy in gender assignment.

- introduce example cases

Example cases illustrate the successful detection of trisomy 13, trisomy 18, trisomy 21, Turner syndrome, Klinefelter syndrome, and XYY syndrome in clinical samples.

- describe maternal plasma DNA sequencing

Maternal plasma DNA was extracted from EDTA-collected blood samples, with plasma separated by double centrifugation and stored at −80°C until processing.

- outline DNA library construction

DNA libraries were constructed using end-repair, A-tailing, adapter ligation, and PCR amplification according to Illumina protocols, with multiplexing enabled by barcoded adapters.

- detail sequencing library preparation

Libraries were quantified by qPCR and size-selected using agarose gel electrophoresis, followed by sequencing on Illumina GAIIx and HiSeq 2000 platforms.

- explain sequencing data analysis

Sequencing data were processed using a custom bioinformatics pipeline that included alignment, duplicate removal, unique read selection, and coverage calculation.

- introduce GC-correlation t-test approach

The GC-correlation t-test approach compares observed coverage to fitted coverage using a t-statistic derived from reference cohort residuals, enabling detection of aneuploidy with high specificity.

- describe detection of trisomy 13, 18, and 21

The method detected all 16 cases of trisomy 21, all 12 cases of trisomy 18, and both cases of trisomy 13 with 100% sensitivity and 99.7% specificity.

- detail detection of XO, XXX, XXY, and XYY

The method correctly identified all cases of XXY and XYY, three out of four cases of XO, and all cases of XXX, with a false positive rate below 0.1%.

- compare GC-correlation t-test approach to other methods

Compared to standard z-score and internal chromosome control methods, the GC-correlation t-test approach demonstrated superior sensitivity for trisomy 13 and 18 and higher specificity for sex chromosome aneuploidies.

- evaluate performance of GC-correlation t-test approach

The method achieved an overall sensitivity of 100% for autosomal aneuploidies and 85.7% for sex chromosome aneuploidies, with specificity exceeding 99.9% in all categories.

- introduce example 8

Example 8 presents detailed detection results for all cases of XO, XXX, XXY, and XYY, confirming the method’s ability to distinguish between different sex chromosome configurations.

- detail detection results for XO, XXX, XXY, and XYY

All XXY and XYY cases were detected with perfect accuracy. Three out of four XO cases were detected, with the missed case exhibiting low fetal fraction and mosaic karyotype.

- introduce example 9

Example 9 evaluates the theoretical sensitivity of the method as a function of gestational age and sequencing depth, demonstrating that detection is feasible even at fetal fractions as low as 3.5%.

- discuss theoretical performance of GC-correlation t-test approach

The theoretical sensitivity of the method increases with gestational age and sequencing depth, with optimal performance achieved at >10 million reads and gestational ages beyond 12 weeks.

- analyze relationship between fetal DNA fraction and gestational age

Fetal DNA fraction increases with gestational age, with a moderate correlation coefficient of 0.1246, indicating that other biological factors also influence concentration.

- evaluate effect of sequencing depth on standard variation

Higher sequencing depth reduces the standard deviation of residuals, improving the power of the t-test and enabling detection at lower fetal fractions.

- determine required sequencing depth for aneuploidy detection

A minimum of 5 million uniquely mapped reads is required to achieve 95% sensitivity for trisomy 21, while 10 million reads are required for reliable detection of trisomy 13 and 18.

- propose strategy for detecting aneuploidy with low fetal DNA fraction

When fetal fraction is below 5%, the method recommends increasing sequencing depth and applying a dual-hypothesis t-test to enhance statistical power.

- estimate theoretical sensitivity of GC-correlation t-test approach

The theoretical sensitivity of the method is estimated to exceed 98% for trisomy 21 and 95% for trisomy 18 at gestational ages above 14 weeks and sequencing depths above 10 million reads.

- calculate theoretical sensitivity considering gestational age and sequencing depth

Sensitivity curves were generated for each aneuploidy type as a function of gestational age and sequencing depth, showing that detection is robust across the second and third trimesters.

- discuss limitations of theoretical sensitivity estimation

Limitations include assumptions of uniform fragment distribution and the exclusion of maternal copy number variants, which may contribute to false positives in rare cases.

- detail calculation of false negative rate

The false negative rate was calculated as the proportion of aneuploid cases with t-statistics below the significance threshold, resulting in a rate of 25% for mosaic XO cases.

- compute theoretical sensitivity in each gestational age

Sensitivity was computed in 1-week intervals from 10 to 34 weeks, showing a steady increase from 80% at 10 weeks to 100% by 16 weeks.

- show resulting plots of theoretical sensitivity calculation

Plots demonstrate a sigmoidal increase in sensitivity with gestational age, plateauing at 99% or higher beyond 18 weeks.

- discuss application of GC-correlation t-test approach

The method is applicable in clinical settings for routine prenatal screening, offering a noninvasive, high-accuracy alternative to current standard-of-care approaches.

- summarize advantages of GC-correlation t-test approach

The method offers superior sensitivity for low-GC trisomies, high specificity for sex chromosome aneuploidies, and compatibility with standard sequencing platforms.

- compare GC-correlation t-test approach to other methods

Unlike z-score methods, which are confounded by GC bias, and internal control methods, which lack sensitivity for autosomal aneuploidies, the GC-correlation t-test approach provides balanced performance across all aneuploidies.

- discuss limitations of GC-correlation t-test approach

Limitations include reduced sensitivity for mosaic aneuploidies with low fetal fraction and dependence on high-quality sequencing data.

- propose future directions for GC-correlation t-test approach

Future directions include integration with targeted sequencing for microdeletion detection and machine learning models for multi-chromosome anomaly prediction.

- conclude example 9

Example 9 establishes the robustness, scalability, and clinical utility of the GC-correlation t-test approach for noninvasive prenatal diagnosis.

- finalize example section

The examples collectively demonstrate that the invention provides a reliable, accurate, and broadly applicable method for the noninvasive detection of fetal genetic abnormalities.