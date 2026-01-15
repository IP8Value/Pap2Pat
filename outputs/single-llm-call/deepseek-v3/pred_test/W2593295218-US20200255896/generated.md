Here is the patent application following your outline and research paper:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates generally to the field of non-invasive prenatal screening (NIPS). More specifically, the invention provides improved methods for detecting chromosomal aneuploidies in a fetus by analyzing cell-free DNA (cfDNA) from maternal plasma. The disclosed techniques significantly reduce false-positive diagnoses by implementing advanced bioinformatics approaches that account for maternal genetic variations, including microduplications and global copy number abnormalities. The invention enhances the positive predictive value (PPV) of NIPS through refined statistical modeling, GC bias correction, and comprehensive chromosomal analysis using next-generation sequencing (NGS) technologies.  

## BACKGROUND  

Current methods for non-invasive prenatal screening suffer from limitations that reduce their clinical utility. While NIPS provides higher detection rates than traditional maternal serum screening, false-positive results remain problematic due to biological and technical factors. Biological contributors include confined placental mosaicism, vanishing twin syndrome, fetal or maternal mosaicism, and maternal chromosomal duplications. Technical issues arise from sequencing biases, particularly in GC-rich regions of chromosomes 13 and 18. These limitations lead to reduced positive predictive values, especially for trisomy 13 and 18, where PPVs can fall below 50%. Existing NIPS assays also fail to adequately distinguish fetal aneuploidies from maternal genetic variations, resulting in unnecessary invasive confirmatory procedures. There exists a critical need for improved NIPS methods that minimize false positives while maintaining high sensitivity for detecting fetal chromosomal abnormalities.  

## SUMMARY OF THE INVENTION  

The invention provides a method for detecting false-positive diagnoses of chromosomal aneuploidy in non-invasive prenatal screening. The method comprises dividing a chromosome of interest into discrete bins and obtaining a bin-specific test parameter for each bin. An ideogram of the chromosome is then plotted to visualize the distribution of bin-specific parameters across the chromosomal regions. The method detects false-positive diagnoses by analyzing the consistency of bin-specific parameters across the chromosome. The steps are repeated for confirming the presence or absence of aneuploidy in the chromosome.  

A substantial portion of the chromosome is analyzed by calculating a chromosome representation value based on the bin-specific parameters. This value is compared to reference values from known euploid samples to determine deviations indicative of aneuploidy. The method improves the positive predictive value of NIPS by distinguishing true fetal aneuploidies from maternal genetic variations. The bin-specific approach enables detection of localized abnormalities that might otherwise be misinterpreted as whole-chromosome aneuploidies.  

The invention further comprises calculating chromosome-specific Z-scores by comparing sample data to reference distributions. Elevated Z-scores trigger examination of chromosomal ideograms to verify whether abnormalities affect the entire chromosome or discrete regions. This analysis prevents false-positive calls due to maternal microduplications or deletions. The method also incorporates GC bias correction algorithms to account for sequencing artifacts in GC-rich regions, particularly for chromosomes 13 and 18.  

## DETAILED DESCRIPTION  

The invention relates to methods for analyzing fetal chromosomal abnormalities through non-invasive prenatal screening. A normal human karyotype consists of 46 chromosomes, with 22 pairs of autosomes and one pair of sex chromosomes. Ploidy refers to the number of complete chromosome sets in a cell, with haploid cells containing one set and diploid cells containing two sets. Aneuploidy describes an abnormal number of chromosomes, which may result from nondisjunction during meiosis or mitosis. Common aneuploidies include trisomy (an extra chromosome) and monosomy (a missing chromosome). Mosaicism occurs when an individual has cells with different chromosomal complements, while fetal aneuploidy specifically refers to chromosomal abnormalities in a developing fetus.  

Non-invasive prenatal testing (NIPT) analyzes cell-free DNA (cfDNA) present in maternal blood, which includes both maternal and fetal DNA components. Cell-free fetal DNA (cffDNA) originates primarily from placental trophoblasts and constitutes a fraction of the total cfDNA, known as the fetal fraction (ff). Invasive prenatal examinations, such as amniocentesis or chorionic villus sampling, carry risks of procedure-related complications and are typically reserved for confirmatory testing after positive NIPS results.  

Chromosomal duplications involve extra copies of chromosomal segments, while deletions involve missing segments. These variations may originate from unequal crossing over during meiosis or errors in DNA replication. Genes are functional units of DNA that encode RNA or polypeptides. Copy number variations (CNVs) are differences in the number of copies of specific DNA segments between individuals. Microduplications and microdeletions are small-scale CNVs that may be pathogenic or benign.  

Next-generation sequencing (NGS) technologies enable high-throughput analysis of cfDNA. The invention utilizes sequencing methods such as shotgun massively parallel sequencing, sequencing-by-synthesis with reversible dye terminators, sequencing-by-ligation, and single molecule sequencing. Specific platforms include the Ion Torrent system, 454 sequencing, Helicos single-molecule sequencing, and the MiSeq personal sequencing system. Sequencing libraries are prepared by attaching adapter sequences to DNA fragments, which are then clonally amplified and sequenced.  

Sequence reads are aligned to bins of a reference genome, where each bin represents a discrete genomic region. Bin read counts are normalized and scaled to account for technical variations. High-order artifacts are corrected using principal component analysis (PCA) modeling. The method calculates bin-specific test parameters by comparing sample read counts to reference distributions. Chromosomal representation values are determined by aggregating bin-specific data across entire chromosomes.  

Z-scores quantify deviations from expected chromosomal representations, with positive values indicating duplications and negative values indicating deletions. The invention interprets Z-scores in the context of chromosomal ideograms, which map DNA sequences to specific chromosomal regions. This approach improves positive predictive value by distinguishing whole-chromosome aneuploidies from localized variations.  

The method evaluates fetal fraction using multiple approaches. For male-bearing pregnancies, fetal fraction is estimated based on Y chromosome overrepresentation. For female-bearing pregnancies, a regularized regression model analyzes X chromosome underrepresentation. Fetal fraction estimates are used to exclude samples with insufficient fetal DNA for reliable analysis.  

The invention detects various chromosomal abnormalities, including trisomy, monosomy, partial duplications or deletions, and mosaicism. It attributes abnormalities to the fetal genome by distinguishing them from maternal variations. The method calculates chromosome-specific Z-scores for multiple chromosomes simultaneously, enabling comprehensive karyotype analysis. Maternal contributions are recognized and excluded to prevent false-positive diagnoses.  

The bin-specific approach pinpoints genetic variations to discrete chromosomal regions. Consistency of bin-specific parameters across a chromosome confirms whole-chromosome abnormalities, while localized deviations suggest maternal microduplications or microdeletions. Large-scale differences in chromosomal representation indicate fetal aneuploidies, which are confirmed through secondary analysis.  

The invention improves the positive predictive value of NIPS by analyzing ideograms for consistency across chromosomal regions. It calculates standard deviations of residues to assess data quality and reliability. The method may be combined with ultrasonographic diagnosis or conventional first/second trimester screening for comprehensive prenatal assessment.  

Various sequencing technologies are employed, including reversible dye-terminator sequencing, sequencing-by-ligation, and single molecule real-time (SMRT) sequencing. The invention corrects GC-sequencing biases using local polynomial regression and loess functions. Different sequencing platforms and methods may be utilized while maintaining the core analytical approach.  

## EXAMPLES  

### Example 1: Assay Development  

Patient samples were collected with informed consent following institutional review board approval. Blood samples were drawn into cell-free DNA BCT tubes and processed within four days. Plasma was isolated through centrifugation at 2,500 x g for 10 minutes followed by 3,200 x g for 20 minutes. Cell-free DNA was extracted from 4 mL plasma using magnetic bead-based purification.  

Sequencing libraries were prepared using the NEBNext Ultra DNA Library Prep Kit with incorporation of 10-bp barcode sequences. PCR amplification was performed for 10 cycles with annealing at 65°C. Purified PCR products were quantified using PicoGreen fluorescence and normalized to 2 nM concentrations. Libraries were pooled and sequenced on a HiSeq2500 system with 36-base single reads and 10-base index reads.  

Fetal fraction estimation for male fetuses utilized Y chromosome-specific sequences, while female fetal fractions were calculated using X chromosome underrepresentation methods. GC content correction was implemented through local polynomial regression fitting of bin read counts against GC percentages.  

### Example 2: Assay Verification and Validation  

The assay was verified using 2,085 samples including 69 trisomy 21, 20 trisomy 18, and 17 trisomy 13 cases. Validation employed 552 samples with 21 trisomy 21, 10 trisomy 18, and 1 trisomy 13 cases. No unaffected samples showed Z-scores >4, while all affected samples had Z-scores >8. GC correction significantly improved discrimination for chromosomes 13 and 18, which have high GC content.  

Twin gestation samples (n=115) showed complete discrimination between affected and unaffected pregnancies, with all trisomy cases displaying Z-scores >11. Fetal sex determination achieved 99.7% accuracy in 372 samples.  

### Example 3: Clinical Implementations  

Clinical implementation used Z-score cutoffs of ≤4 for negative and >8 for positive results. Intermediate Z-scores (4-8) triggered ideogram analysis to detect maternal microduplications. Among 10,713 clinical samples, 180 showed abnormal results (1.8% positive rate). Maternal microduplications were identified in cases with intermediate Z-scores through chromosomal ideogram analysis and confirmed by microarray.  

The assay detected cases of mosaicism and translocations, including a Robertsonian translocation with Z-score 30.78. Maternal global copy number abnormalities were identified through genome-wide Z-score analysis. Positive predictive values were 98% for trisomy 21, 92% for trisomy 18, and 69% for trisomy 13. Sex chromosome aneuploidies showed 86% PPV for Turner syndrome.  

## EQUIVALENTS  

The invention is not limited to the specific embodiments described herein. Functionally equivalent methods of binning chromosomes, calculating Z-scores, or estimating fetal fractions are encompassed. Markush groups include all possible combinations of described elements. Numerical ranges include all values within the stated boundaries and their equivalents. The invention incorporates prior art techniques for DNA sequencing and statistical analysis while providing novel combinations that improve NIPS performance.