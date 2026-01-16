# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of noninvasive prenatal screening (NIPS) using cell-free DNA (cfDNA) from maternal plasma. Specifically, the invention pertains to a method and system for detecting fetal chromosome aneuploidies, such as trisomy 21, trisomy 18, and trisomy 13, with enhanced specificity and positive predictive value (PPV). The invention utilizes advanced sequencing techniques, proprietary bioinformatics, and biostatistical methods to minimize false-positive results and improve the reliability of NIPS.

## BACKGROUND

Noninvasive prenatal screening (NIPS) using cell-free DNA (cfDNA) from maternal plasma has emerged as a highly effective method for detecting fetal aneuploidies, particularly trisomy 21 (Down syndrome), trisomy 18 (Edwards syndrome), and trisomy 13 (Patau syndrome). Traditional methods, such as maternal serum screening and nuchal translucency testing, have higher rates of false-positive results, leading to unnecessary invasive procedures that carry risks of miscarriage and other complications. NIPS, on the other hand, offers higher detection rates and lower false-positive rates, making it a preferred option for high-risk pregnancies.

Several NIPS assays are currently available, including those based on single-nucleotide polymorphism (SNP) analysis, chromosome-specific sequencing (CSS), and massive parallel shotgun sequencing (MPSS). Despite their high sensitivity and specificity, these assays still face challenges, particularly in terms of false-positive results, which affect the positive predictive value (PPV) of the test. The PPV is influenced by the prevalence of the disorder and the specificity of the test. For example, even a test with 99.9% specificity can yield PPVs of 90% for trisomy 21, 67% for trisomy 18, and 53% for trisomy 13 in high-risk populations.

False-positive results in NIPS can be attributed to various biological and technical factors, including confined placental mosaicism, vanishing twin syndrome, fetal or maternal mosaicism, maternal tumors, and maternal duplications. Among these, maternal duplications have been identified as a significant contributor to false-positive results. To address these issues and improve the PPV of NIPS, there is a need for a more refined and accurate assay.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for noninvasive prenatal screening (NIPS) using cell-free DNA (cfDNA) from maternal plasma to detect fetal chromosome aneuploidies with enhanced specificity and positive predictive value (PPV). The invention includes the following key features:

1. **Automated Sample Processing**: The method involves the automated extraction of cfDNA from maternal plasma, ensuring consistent and high-yield preparation of cfDNA. This step is crucial for reducing variability and improving the reliability of the assay.

2. **Next-Generation Sequencing (NGS)**: The cfDNA is sequenced using high-throughput next-generation sequencing (NGS) technology, specifically the HiSeq platform from Illumina. The sequencing process is optimized using "Version 4" chemistry, which enhances the quality and quantity of sequencing data.

3. **Proprietary Bioinformatics and Biostatistical Analysis**: The invention employs a proprietary bioinformatics pipeline to analyze the sequencing data. This pipeline includes GC sequence bias correction, normalization of bin read counts, and calculation of chromosome-specific Z-scores. The Z-scores are used to discriminate between affected and unaffected pregnancies with high accuracy.

4. **Detection of Maternal Duplications and Global Copy Number Changes**: The method includes a step to identify and exclude false-positive results caused by maternal duplications and global copy number changes. This is achieved by examining the entire genome of positive NIPS cases and using chromosomal ideograms to ensure that the entire chromosome is duplicated and not just a small portion.

5. **Fetal Fraction Estimation**: The invention provides methods for estimating the fetal fraction using both X and Y chromosome read counts. This estimation is crucial for interpreting the NIPS results and ensuring the accuracy of the assay.

6. **Sex Chromosome Aneuploidy Detection**: The method is capable of detecting sex chromosome aneuploidies, such as Turner syndrome (45,X) and Klinefelter syndrome (47,XXY), with high accuracy. The detection is based on the analysis of read counts within specific bins on the X and Y chromosomes.

7. **Clinical Implementation**: The invention is designed for clinical implementation, with a Z-score cutoff of ≤4 for unaffected pregnancies and >8 for affected pregnancies. The method also includes procedures for handling cases with intermediate Z-scores and low fetal fractions, ensuring that false-positive results are minimized.

The invention significantly improves the PPV of NIPS by addressing the key factors that contribute to false-positive results, thereby providing a more reliable and accurate screening tool for fetal aneuploidies.

## DETAILED DESCRIPTION

### Automated Sample Processing

The method begins with the automated extraction of cell-free DNA (cfDNA) from maternal plasma. Whole blood is collected in Cell-Free DNA BCT blood collection tubes and processed within 4 days of draw. The plasma is isolated using a Tecan EVO 200 liquid handler, which performs a series of centrifugation and transfer steps to ensure high-yield and consistent cfDNA preparation. The cfDNA is then extracted using DynaMax chemistry and the Kingfisher Flex Purification System, following the manufacturer's recommendations.

### Next-Generation Sequencing (NGS)

The extracted cfDNA is converted into sequencing-ready libraries using the NEBNext® Ultra™ DNA Library Prep Kit for Illumina®. During PCR, a 10-bp barcode is added to each sample using a reverse PCR primer. The PCR products are purified using Agencourt AMPure XP PCR Purification beads and quantified using the Quant-It PicoGreen dsDNA Assay Kit. The libraries are pooled and sequenced on a HiSeq2500 system using single-read 36 cycles followed by 10 cycles to sequence the index. The sequencing data are streamed to an Isilon server, where the bioinformatics analysis pipeline is initiated.

### Proprietary Bioinformatics and Biostatistical Analysis

The bioinformatics pipeline includes several key steps to ensure accurate and reliable results:

1. **Bin Read Count Data Normalization**: The raw sequencing data are normalized by scaling the bin read counts by the total autosomal read counts of the sample. This step ensures that the data are comparable across different samples.

2. **GC Sequence Bias Correction**: GC content can introduce biases in sequencing data, particularly for chromosomes 13 and 18, which are GC-rich. The invention uses a published R-script for GC correction, followed by statistical smoothing using a proprietary algorithm. This correction significantly improves the discrimination between affected and unaffected pregnancies.

3. **Chromosome Representation and Z-Score Calculation**: Chromosome representations are calculated as the ratio of the total read counts for each chromosome to the sum of the total read counts for all autosomes. The Z-score for each chromosome is then calculated using the formula:
   \[
   Z = \frac{x - \mu}{\sigma}
   \]
   where \( x \) is the sample chromosome representation, \( \mu \) is the chromosome representation plate median, and \( \sigma \) is the chromosome representation median absolute deviation (MAD).

4. **Fetal Fraction Estimation**: Fetal fractions are estimated using both X and Y chromosome read counts. For male fetuses, the fetal fraction is calculated using Y chromosome-specific sequences. For female fetuses, a proprietary bioinformatics approach is used to estimate the fetal fraction based on the X chromosome representation.

### Detection of Maternal Duplications and Global Copy Number Changes

To minimize false-positive results, the method includes a step to identify and exclude cases with maternal duplications and global copy number changes. This is achieved by examining the entire genome of positive NIPS cases and using chromosomal ideograms to ensure that the entire chromosome is duplicated and not just a small portion. If a small portion of the chromosome is duplicated, the result is flagged as a potential maternal duplication, and further investigation is conducted using maternal DNA analysis.

### Sex Chromosome Aneuploidy Detection

The method is capable of detecting sex chromosome aneuploidies, such as Turner syndrome (45,X) and Klinefelter syndrome (47,XXY). The detection is based on the analysis of read counts within specific bins on the X and Y chromosomes. The method calculates the X-chromosome-based fetal fraction estimate and the Y-chromosome-based fetal fraction estimate to determine the presence of sex chromosome aneuploidies.

### Clinical Implementation

The invention is designed for clinical implementation, with a Z-score cutoff of ≤4 for unaffected pregnancies and >8 for affected pregnancies. Cases with intermediate Z-scores (between 3 and 8) are reviewed to ensure that the entire chromosome is duplicated and not just a small portion. Low fetal fractions (below 5%) prompt a request for a new sample. The method also includes procedures for handling cases with uninformative DNA patterns or other technical issues.

### Initial Clinical Data

The method has been validated using a series of verification and validation samples, including known unaffected and aneuploid pregnancies. The results demonstrate 100% discrimination between affected and unaffected pregnancies, with no unaffected pregnancy having a Z score >4 and no affected pregnancy having a Z score <8. The method has also been tested in clinical settings, with a positive predictive value (PPV) of 98% for trisomy 21, 92% for trisomy 18, and 69% for trisomy 13. The method has successfully detected maternal duplications and global copy number changes, leading to a reduction in false-positive results.

## EXAMPLES

### Example 1: Assay Development

The assay was developed to improve the specificity and positive predictive value (PPV) of noninvasive prenatal screening (NIPS) for fetal chromosome aneuploidies. The development process involved the following steps:

1. **Sample Collection and Processing**: Whole blood was collected from pregnant women and processed to isolate plasma. The cfDNA was extracted using automated methods to ensure high yield and consistency.

2. **Library Preparation and Sequencing**: The cfDNA was converted into sequencing-ready libraries and sequenced using the HiSeq2500 system. The sequencing data were streamed to an Isilon server for analysis.

3. **Bioinformatics and Statistical Analysis**: The raw sequencing data were normalized and corrected for GC bias. Chromosome representations and Z-scores were calculated to discriminate between affected and unaffected pregnancies.

4. **Validation Studies**: The assay was validated using a series of verification and validation samples, including known unaffected and aneuploid pregnancies. The results demonstrated 100% discrimination between affected and unaffected pregnancies.

### Example 2: Assay Verification and Validation

The assay was verified and validated using a series of samples, including known unaffected and aneuploid pregnancies. The verification and validation studies involved the following steps:

1. **Verification Study**: A series of 2,085 samples, including trisomy 21 (n = 69), trisomy 18 (n = 20), and trisomy 13 (n = 17), were tested. No unaffected pregnancy had a Z score >4, and no affected pregnancy had a Z score <8.

2. **Validation Study**: A validation set comprising 552 samples, including trisomy 21 (n = 21), trisomy 18 (n = 10), trisomy 13 (n = 1), and XO (n = 1), was analyzed. Again, no unaffected pregnancy had a Z score >4, and no affected pregnancy had a Z score <8.

3. **Combined Analysis**: The results from the verification and validation studies were combined for analysis. The effects of GC correction were evaluated, and the assay provided 100% discrimination between affected and unaffected pregnancies.

### Example 3: Clinical Implementations

The assay has been implemented in a clinical setting, with the following results:

1. **Positive Predictive Values (PPVs)**: The PPV for trisomy 21 was 98%, for trisomy 18 was 92%, and for trisomy 13 was 69%. These values are significantly higher than those reported for first-generation NIPS tests.

2. **Maternal Duplications and Global Copy Number Changes**: The assay successfully detected and excluded cases with maternal duplications and global copy number changes, leading to a reduction in false-positive results.

3. **Fetal Fraction Estimation**: The method accurately estimated fetal fractions using both X and Y chromosome read counts, ensuring the reliability of the NIPS results.

4. **Sex Chromosome Aneuploidy Detection**: The method detected sex chromosome aneuploidies, such as Turner syndrome and Klinefelter syndrome, with high accuracy.

5. **Clinical Follow-Up**: The method has been used to screen over 10,000 clinical samples, with a positive rate of 1.8%. The causes of unreportable results, such as low fetal fraction and uninformative DNA patterns, were identified and addressed.

## EQUIVALENTS

While specific embodiments of the invention have been described, it is understood that various modifications and substitutions can be made without departing from the spirit and scope of the invention. For example, alternative sequencing platforms and bioinformatics tools can be used to achieve similar results. The invention is intended to cover all such modifications and equivalents, as defined by the appended claims.