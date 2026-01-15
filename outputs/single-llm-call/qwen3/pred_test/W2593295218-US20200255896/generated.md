# DESCRIPTION

## TECHNICAL FIELD

- introduce non-invasive prenatal screening

Non-invasive prenatal screening represents a transformative approach to the early detection of fetal chromosomal abnormalities without the risks associated with invasive diagnostic procedures. This method leverages the presence of cell-free fetal DNA circulating in maternal plasma, enabling the analysis of fetal genetic material through high-throughput sequencing technologies. The technique has emerged as a clinically viable alternative to traditional serum biomarker screening and ultrasound-based assessments, offering significantly improved accuracy in identifying common autosomal trisomies such as trisomy 21, trisomy 18, and trisomy 13, as well as select sex chromosome aneuploidies and microdeletion syndromes. By analyzing the relative abundance of DNA fragments derived from each chromosome, this screening modality provides a non-invasive, highly sensitive, and specific means of assessing fetal genomic integrity during the first and early second trimesters of pregnancy. The method is particularly valuable in high-risk populations, where the consequences of undetected aneuploidy may include severe developmental disabilities, early fetal loss, or complex neonatal care requirements. The integration of advanced bioinformatics, statistical modeling, and bias correction algorithms has further refined the reliability of this approach, allowing for the differentiation of true fetal chromosomal abnormalities from confounding maternal genetic variations and technical artifacts that may otherwise lead to erroneous interpretations. As the clinical adoption of non-invasive prenatal screening continues to expand, the need for robust, reproducible, and interpretable analytical frameworks has become paramount to ensure accurate risk stratification and informed clinical decision-making.

## BACKGROUND

- limitations of current NIPS methods

Current non-invasive prenatal screening methods, while offering substantial improvements over conventional serum screening, remain susceptible to a range of biological and technical limitations that compromise their positive predictive value. A significant proportion of positive results derived from existing assays are ultimately found to be false positives upon confirmation via invasive diagnostic testing, such as amniocentesis or chorionic villus sampling. These false positives arise not only from technical noise, such as sequencing biases and low fetal fraction, but also from underlying maternal genomic anomalies, including microduplications, microdeletions, and global copy number variations that are not distinguishable from fetal aneuploidies using conventional bin-based analysis. Maternal conditions such as uterine fibroids, maternal malignancies, or chromosomal mosaicism can introduce aberrant cell-free DNA profiles that mimic fetal chromosomal gains or losses, leading to misinterpretation of the fetal karyotype. Furthermore, the inability to resolve whether an elevated Z-score reflects a whole-chromosome trisomy or a localized duplication restricts diagnostic specificity, particularly for chromosomes with high GC content where sequencing artifacts are more pronounced. Many existing platforms rely on simplistic normalization techniques that fail to account for regional variations in sequencing efficiency, resulting in inconsistent performance across chromosomes and gestational ages. The reliance on global chromosome representation without spatial resolution of the underlying genomic alterations prevents the identification of maternal contributions, thereby limiting the assay’s capacity to discriminate between fetal and maternal sources of genetic imbalance. Consequently, despite high sensitivity, the positive predictive value for trisomy 13 and trisomy 18 remains suboptimal, often falling below 50% in clinical practice, which necessitates unnecessary invasive procedures and causes significant psychological distress for expectant parents. These unresolved challenges underscore the necessity for an enhanced analytical framework that incorporates chromosomal ideogram-based evaluation, bin-level consistency analysis, and maternal contribution exclusion to substantially reduce false-positive rates and improve clinical utility.

## SUMMARY OF THE INVENTION

- detect false-positive diagnosis of chromosomal aneuploidy

The invention provides a method for detecting false-positive diagnoses of chromosomal aneuploidy in non-invasive prenatal screening by distinguishing between fetal and maternal sources of genomic imbalance through spatially resolved analysis of cell-free DNA sequencing data. This is accomplished by evaluating the distribution of sequence read counts across discrete chromosomal bins and identifying patterns indicative of localized maternal duplications or deletions that mimic whole-chromosome aneuploidies. By analyzing the consistency of bin-specific test parameters along the length of a chromosome, the method differentiates between diffuse, whole-chromosome deviations characteristic of fetal trisomy or monosomy and focal, discontinuous deviations that are indicative of maternal genomic variants. This distinction enables the exclusion of false-positive results that would otherwise be misclassified as fetal aneuploidy, thereby improving diagnostic accuracy and reducing unnecessary invasive procedures. The method does not rely solely on aggregate chromosome-level Z-scores but instead employs a multi-layered analytical approach that integrates bin-level data, ideogram visualization, and statistical deviation modeling to ensure that only those results consistent with a uniform chromosomal imbalance are reported as positive for fetal aneuploidy.

- divide chromosome into bins

The chromosome is divided into a plurality of non-overlapping, uniformly sized genomic bins, each containing a defined number of base pairs and uniquely mapped to a specific chromosomal locus using a reference genome assembly. These bins are selected to avoid repetitive elements, segmental duplications, and homologous regions to ensure that sequence reads align uniquely and reliably to their intended genomic location. The size of each bin is optimized to balance resolution and statistical power, enabling the detection of both whole-chromosome aneuploidies and sub-chromosomal structural variants without introducing excessive noise. The binning strategy is applied uniformly across all autosomes and sex chromosomes, permitting comparative analysis of read count distributions across the entire genome. Each bin serves as an independent unit of measurement, allowing for the quantification of relative DNA abundance at a fine-grained level that preserves spatial information critical for distinguishing fetal from maternal origins of copy number variation.

- obtain bin-specific test parameter

A bin-specific test parameter is calculated for each genomic bin by normalizing the observed number of sequence reads to the total autosomal read count of the sample, followed by correction for GC content bias using a locally weighted regression model trained on a large cohort of euploid pregnancies. This normalized read count is then transformed into a standardized score that reflects the deviation of the bin’s representation from the expected median value across the reference population. The bin-specific test parameter is derived from a combination of read depth, GC correction, and high-order artifact removal, ensuring that technical variability is minimized and biological signal is maximized. This parameter is computed independently for each bin and retained as a discrete data point for subsequent spatial analysis, forming the foundation for the construction of chromosomal ideograms and the identification of localized anomalies.

- plot ideogram of chromosome

An ideogram of each chromosome is generated by plotting the bin-specific test parameters along the linear genomic coordinates of the respective chromosome, resulting in a visual representation that resembles a high-resolution microarray profile. Each point on the ideogram corresponds to a single bin and its associated standardized deviation score, with positive values indicating relative overrepresentation and negative values indicating underrepresentation. The ideogram provides a spatially resolved depiction of copy number variation across the entire chromosome, allowing for the identification of focal peaks or troughs that deviate from the background noise level. This graphical representation enables clinicians and bioinformaticians to visually assess the continuity and uniformity of the deviation, distinguishing between diffuse, whole-chromosome patterns consistent with fetal aneuploidy and discontinuous, localized patterns indicative of maternal genomic variants.

- detect false-positive diagnosis

False-positive diagnoses of chromosomal aneuploidy are detected by identifying chromosomal ideograms in which the elevated bin-specific test parameters are confined to a discrete subregion of the chromosome rather than being uniformly distributed across the entire chromosome. When the deviation is localized to a segment comprising less than a substantial portion of the chromosome, the result is flagged as likely originating from a maternal microduplication or microdeletion rather than a fetal aneuploidy. This detection is automated through algorithmic assessment of the spatial consistency of the bin-specific test parameters, including the calculation of standard deviation of residues and the identification of breakpoints that demarcate regions of abnormality. The system excludes from reporting any result where the pattern of deviation is inconsistent with a whole-chromosome gain or loss, thereby preventing the misclassification of maternal variants as fetal aneuploidies.

- repeat steps for confirming chromosome

The steps of dividing the chromosome into bins, obtaining bin-specific test parameters, plotting the ideogram, and detecting false-positive diagnosis are repeated for each of the autosomes and sex chromosomes analyzed in the sample. This systematic, chromosome-by-chromosome evaluation ensures comprehensive coverage and eliminates the possibility of overlooking a maternal variant on a chromosome that may have been initially dismissed due to an elevated aggregate Z-score. The process is performed in parallel for all chromosomes, with results compiled into a unified report that includes ideograms for each chromosome, Z-scores for whole-chromosome representation, and flags for any localized anomalies requiring further investigation.

- calculate substantial portion of chromosome

A substantial portion of the chromosome is defined as the minimum contiguous segment required to constitute a biologically meaningful whole-chromosome aneuploidy, determined empirically from a training cohort of confirmed fetal trisomies and monosomies. This threshold is established as the proportion of the chromosome length over which the bin-specific test parameters must remain consistently elevated or depressed to be considered indicative of a fetal origin. For example, if a deviation is observed across less than 70% of the chromosome’s bins, the pattern is classified as localized and not consistent with fetal aneuploidy. This threshold is applied uniformly across all chromosomes and is calibrated to minimize false negatives while maximizing the exclusion of maternal variants.

- obtain bin-specific parameter

The bin-specific parameter is obtained through a proprietary bioinformatics pipeline that integrates raw sequencing reads, reference genome alignment, GC content normalization, and artifact correction using a regularized regression model trained on over five thousand euploid samples. The pipeline accounts for batch effects, sequencing platform variability, and library preparation biases, ensuring that the bin-specific parameter is robust, reproducible, and independent of technical confounders. The parameter is calculated for each bin independently and stored as a continuous variable that reflects the relative abundance of DNA fragments derived from that genomic region.

- calculate chromosome representation value

The chromosome representation value is calculated as the mean of all bin-specific test parameters across the entire chromosome, providing a single aggregate score that reflects the overall copy number status of that chromosome. This value is then used to compute the chromosome-specific Z-score, which quantifies the deviation of the sample’s representation from the median representation observed in the reference population, normalized by the median absolute deviation. The chromosome representation value serves as the primary metric for initial risk stratification, while the bin-specific parameters serve as the basis for confirmatory analysis.

- compare to references

The chromosome representation value is compared to a reference distribution derived from a large, ethnically diverse cohort of euploid pregnancies, ensuring that population-specific variations in baseline DNA abundance are accounted for. The reference distribution is updated periodically to incorporate new data and maintain analytical accuracy across different sequencing platforms and laboratory conditions. This comparison allows for the calculation of a standardized Z-score that is calibrated to reflect the likelihood of fetal aneuploidy, independent of absolute read depth or sequencing depth.

- detect false-positive diagnosis

False-positive diagnosis is detected when the chromosome-specific Z-score exceeds a predefined threshold indicative of aneuploidy, but the corresponding ideogram reveals a non-uniform pattern of deviation that is inconsistent with a whole-chromosome gain or loss. In such cases, the result is flagged as a potential maternal variant, and further investigation is triggered, including maternal DNA analysis or microarray confirmation, before any clinical report is issued. This dual-layered approach ensures that only those results consistent with a uniform chromosomal imbalance are classified as positive for fetal aneuploidy.

- improve positive predictive value

The method significantly improves the positive predictive value of non-invasive prenatal screening by eliminating a major source of false-positive results arising from maternal genomic variants. By incorporating spatial resolution and ideogram-based validation into the analytical workflow, the invention reduces the number of false-positive calls attributed to maternal microduplications, microdeletions, and global copy number abnormalities. This refinement increases the proportion of true fetal aneuploidies among all positive test results, thereby enhancing clinical confidence, reducing unnecessary invasive procedures, and minimizing psychological burden on expectant parents. The improvement in positive predictive value is most pronounced for trisomy 13 and trisomy 18, where maternal variants have historically contributed disproportionately to false-positive rates.

## DETAILED DESCRIPTION

- define karyotype

A karyotype is the complete set of chromosomes in a cell, organized by number, size, and morphology, and used to represent the chromosomal composition of an individual. It provides a visual and quantitative summary of the genetic material inherited from both parents and serves as the foundation for diagnosing chromosomal disorders. In humans, the normal karyotype consists of 23 pairs of chromosomes, including 22 pairs of autosomes and one pair of sex chromosomes, totaling 46 chromosomes. The karyotype is typically determined through cytogenetic analysis of metaphase cells, but can also be inferred indirectly through high-throughput sequencing of cell-free DNA in maternal plasma. Abnormalities in the karyotype, such as gains or losses of entire chromosomes or large chromosomal segments, are associated with developmental, cognitive, and physiological impairments that may be detected prenatally through screening methods.

- describe normal human karyotypes

Normal human karyotypes are characterized by the presence of two copies of each autosome and either two X chromosomes in females or one X and one Y chromosome in males. This diploid configuration ensures balanced gene expression and proper embryonic development. The autosomes are numbered from 1 to 22 in descending order of size, with chromosome 1 being the largest and chromosome 22 the smallest. The sex chromosomes determine biological sex, with XX indicating female and XY indicating male. In a normal karyotype, each chromosome is present in its entirety, with no large-scale deletions, duplications, translocations, or other structural abnormalities. The integrity of the karyotype is maintained through precise mechanisms of DNA replication, segregation, and repair during cell division. Deviations from this canonical structure are associated with a wide spectrum of clinical phenotypes, ranging from mild to lethal, depending on the affected chromosome and the extent of the imbalance.

- define ploidy

Ploidy refers to the number of complete sets of chromosomes present in a cell. In diploid organisms, such as humans, somatic cells contain two sets of chromosomes—one inherited from each parent—resulting in a ploidy level of two. Gametes, by contrast, are haploid, containing only one set of chromosomes, which ensures that upon fertilization, the resulting zygote reestablishes the diploid state. Ploidy is a fundamental concept in genetics and is critical for understanding chromosomal disorders. Abnormal ploidy levels, such as triploidy (three sets) or tetraploidy (four sets), are typically incompatible with life and result in early embryonic demise. In the context of non-invasive prenatal screening, the detection of deviations from the normal diploid state in fetal DNA is used to infer the presence of aneuploidy.

- describe haploid and diploid organisms

Haploid organisms possess a single set of chromosomes in their somatic cells and are typically found in certain fungi, algae, and the gametes of diploid species. In contrast, diploid organisms, including all mammals, maintain two homologous copies of each chromosome in their somatic cells, one derived from each parent. This diploid state allows for genetic redundancy, which buffers against deleterious mutations and enables complex regulatory mechanisms. In humans, the transition from haploid gametes to diploid zygote is essential for normal development. The maintenance of diploidy is tightly regulated during cell division, and errors in this process can lead to aneuploidy, which is the primary target of non-invasive prenatal screening.

- define aneuploidy

Aneuploidy is a chromosomal abnormality characterized by an abnormal number of chromosomes in a cell, deviating from the typical diploid complement. This condition may involve the presence of one or more extra chromosomes (trisomy) or the absence of one or more chromosomes (monosomy). Aneuploidy arises from errors in meiosis or mitosis, particularly during gametogenesis or early embryonic development, and is a leading cause of miscarriage, stillbirth, and congenital disorders. Common viable forms of aneuploidy in live-born infants include trisomy 21 (Down syndrome), trisomy 18 (Edwards syndrome), and trisomy 13 (Patau syndrome), as well as sex chromosome aneuploidies such as Turner syndrome (45,X) and Klinefelter syndrome (47,XXY). Non-invasive prenatal screening detects aneuploidy by identifying deviations in the relative abundance of cell-free DNA fragments derived from specific chromosomes, which reflect underlying imbalances in fetal genomic copy number.

- describe causes of aneuploidy

Aneuploidy is primarily caused by nondisjunction during meiosis, wherein homologous chromosomes or sister chromatids fail to separate properly, resulting in gametes with an abnormal number of chromosomes. Maternal age is the most significant risk factor for meiotic nondisjunction, particularly for chromosomes 13, 18, and 21, due to prolonged arrest of oocytes in prophase I. Other contributing factors include genetic predisposition, environmental exposures, and errors in DNA repair mechanisms. In some cases, aneuploidy may arise post-zygotically through mitotic errors during early embryonic cleavage, leading to mosaicism. Additionally, structural chromosomal rearrangements such as Robertsonian translocations can predispose to unbalanced gametes and subsequent aneuploidy in offspring. The detection of aneuploidy in cell-free fetal DNA relies on identifying the overrepresentation or underrepresentation of sequences derived from the affected chromosome relative to the rest of the genome.

- define trisomy

Trisomy is a specific form of aneuploidy in which three copies of a particular chromosome are present in a cell instead of the normal two. This condition results from the presence of an extra chromosome, either due to meiotic nondisjunction or mitotic error, and leads to an excess of gene dosage that disrupts normal development. Trisomy 21, the most common viable trisomy, is associated with intellectual disability, characteristic facial features, and increased risk of congenital heart defects. Trisomy 18 and trisomy 13 are more severe and are often lethal in utero or within the first year of life. Non-invasive prenatal screening detects trisomy by measuring an elevated relative abundance of DNA fragments originating from the affected chromosome, which is reflected in a significantly higher chromosome-specific Z-score compared to euploid controls.

- define monosomy

Monosomy is a chromosomal abnormality in which only one copy of a particular chromosome is present in a cell, instead of the normal two. This condition typically arises from the loss of a chromosome during meiosis or early embryonic development and is often incompatible with life, except in the case of monosomy X, or Turner syndrome. Monosomy results in a deficit of gene expression, leading to developmental abnormalities and organ dysfunction. In non-invasive prenatal screening, monosomy is identified by a significantly reduced relative abundance of DNA fragments from the affected chromosome, resulting in a negative chromosome-specific Z-score. The detection of monosomy is more challenging than trisomy due to the lower abundance of fetal DNA and the greater susceptibility to technical noise, but is achievable through high-resolution binning and robust statistical modeling.

- define mosaicism

Mosaicism is a condition in which an individual possesses two or more populations of cells with distinct chromosomal compositions, originating from a single zygote. This phenomenon arises from a mitotic error occurring during early embryonic development, leading to the coexistence of euploid and aneuploid cell lines. Mosaicism can affect any chromosome and may be confined to the placenta (confined placental mosaicism) or present in fetal tissues. The clinical manifestations of mosaicism vary widely depending on the proportion and distribution of abnormal cells. In non-invasive prenatal screening, mosaicism may be detected as an intermediate Z-score that falls between the typical ranges for euploid and fully aneuploid samples, prompting further investigation through invasive diagnostic testing.

- define fetal aneuploidy

Fetal aneuploidy refers to the presence of an abnormal number of chromosomes in the cells of a developing fetus, resulting from errors in chromosome segregation during gametogenesis or early embryogenesis. It encompasses conditions such as trisomy, monosomy, and mosaicism involving autosomes or sex chromosomes and is a leading cause of pregnancy loss, congenital anomalies, and neurodevelopmental disorders. Fetal aneuploidy is detectable in cell-free fetal DNA circulating in maternal plasma, where the relative abundance of DNA fragments from the affected chromosome is altered in proportion to the degree of chromosomal imbalance. The detection of fetal aneuploidy through non-invasive screening requires the differentiation of fetal signals from maternal background and technical artifacts, which is achieved through advanced bioinformatics and spatial analysis of sequencing data.

- describe non-invasive prenatal testing (NIPT)

Non-invasive prenatal testing is a molecular diagnostic method that analyzes cell-free fetal DNA in maternal blood to screen for chromosomal aneuploidies without the risks associated with invasive procedures. The technique involves the collection of a maternal blood sample, isolation of cell-free DNA, sequencing of the fragments using high-throughput platforms, and computational analysis to determine the relative representation of each chromosome. NIPT has largely replaced traditional serum screening due to its superior sensitivity and specificity, particularly for trisomy 21. It is typically offered to pregnant women at increased risk based on age, ultrasound findings, or prior history, though its use is expanding to average-risk populations. The method does not provide a definitive diagnosis but serves as a highly accurate screening tool that reduces the need for invasive confirmatory testing.

- describe invasive prenatal examination

Invasive prenatal examination refers to procedures that obtain fetal tissue or fluid directly from the uterus to perform cytogenetic or molecular analysis of the fetal genome. These include chorionic villus sampling, which collects placental tissue during the first trimester, and amniocentesis, which extracts amniotic fluid containing fetal cells during the second trimester. These procedures allow for definitive diagnosis of chromosomal abnormalities through karyotyping, microarray analysis, or quantitative PCR. However, they carry a small but significant risk of procedure-related complications, including miscarriage, infection, and fetal injury. The primary purpose of non-invasive prenatal testing is to reduce the number of invasive procedures by accurately identifying low-risk pregnancies, thereby preserving the diagnostic certainty of invasive methods for those cases with a high probability of aneuploidy.

- define chromosomal duplication

Chromosomal duplication is a structural variation in which a segment of a chromosome is copied one or more times, resulting in an increase in gene dosage. Duplications can range in size from a few kilobases to several megabases and may involve entire chromosomal arms or discrete regions. When occurring in the maternal genome, these duplications can be shed into the maternal circulation as cell-free DNA and mimic the signal of a fetal trisomy, leading to false-positive results in non-invasive prenatal screening. The detection of such duplications requires spatial resolution of sequencing data to distinguish between localized maternal variants and diffuse fetal aneuploidies.

- define chromosomal deletion

Chromosomal deletion is a structural variant in which a segment of a chromosome is lost, resulting in a reduction in gene dosage. Deletions can be submicroscopic, involving only a few genes, or large-scale, affecting entire chromosomal bands. When present in the maternal genome, deletions can cause underrepresentation of specific chromosomal regions in cell-free DNA, potentially mimicking fetal monosomy. Accurate discrimination between maternal deletions and fetal monosomy requires detailed analysis of bin-specific read counts and ideogram patterns to identify localized versus uniform deviations.

- describe origins of chromosomal duplication or deletion

Chromosomal duplications and deletions arise from errors in DNA replication, repair, or recombination, particularly during meiosis or early embryonic development. Mechanisms such as non-allelic homologous recombination, replication fork stalling, and microhomology-mediated repair can lead to the gain or loss of genomic segments. These events may occur spontaneously or be influenced by genetic predisposition, environmental factors, or advanced maternal age. In the context of non-invasive prenatal screening, maternal duplications and deletions are often inherited or arise de novo and are not related to the fetal genome, yet they can produce signals indistinguishable from fetal aneuploidy when analyzed at a chromosome-wide level.

- define gene

A gene is a functional unit of heredity composed of a specific sequence of DNA that encodes a polypeptide or RNA molecule. Genes are organized along chromosomes and are subject to dosage-dependent regulation, meaning that alterations in copy number can disrupt normal cellular function. In aneuploidy, the presence of extra or missing copies of genes leads to imbalances in protein expression, which underlies the phenotypic consequences of conditions such as Down syndrome. Non-invasive prenatal screening detects these imbalances indirectly by measuring the relative abundance of DNA fragments derived from chromosomal regions containing many genes.

- describe RNA or polypeptide production

RNA and polypeptide production are the processes by which genetic information encoded in DNA is transcribed into messenger RNA and subsequently translated into functional proteins. The quantity of RNA and protein produced is often proportional to the number of gene copies present in the genome. In trisomy, the presence of three copies of a gene leads to approximately 1.5-fold increases in RNA and protein levels, which can disrupt cellular homeostasis and developmental pathways. In monosomy, the reduction to a single copy leads to insufficient gene expression, impairing normal tissue formation. Non-invasive prenatal screening infers these imbalances by quantifying the relative abundance of DNA fragments from genes distributed across a chromosome, assuming that copy number correlates with sequencing read depth.

- define chromosome variation

Chromosome variation refers to any structural or numerical alteration in the chromosomal complement that deviates from the normal diploid state. This includes aneuploidies, duplications, deletions, translocations, inversions, and other rearrangements. Chromosome variations can be inherited or arise de novo and may have no clinical consequence or may result in severe developmental disorders. In non-invasive prenatal screening, the detection of chromosome variation requires distinguishing between fetal and maternal sources of variation, particularly when maternal variants mimic fetal aneuploidies.

- describe copy number variation

Copy number variation is a type of structural genomic variation in which the number of copies of a particular DNA segment differs between individuals. These variations can range from kilobase to megabase scales and are common in the human genome. While many copy number variations are benign, others are associated with disease, particularly when they affect dosage-sensitive genes. In non-invasive prenatal screening, copy number variations in the maternal genome can produce false-positive signals that resemble fetal aneuploidy, necessitating methods to differentiate maternal from fetal origins.

- describe microduplication and microdeletion

Microduplication and microdeletion refer to small-scale copy number variations involving segments of DNA typically less than five megabases in size. These variants may encompass one or more genes and are often associated with neurodevelopmental disorders, such as DiGeorge syndrome (22q11.2 deletion) or 16p11.2 duplication syndrome. When present in the maternal genome, microduplications can produce localized elevations in cell-free DNA read counts that mimic fetal trisomy, while microdeletions can mimic fetal monosomy. The detection of these variants requires high-resolution binning and ideogram analysis to identify focal deviations that are inconsistent with whole-chromosome aneuploidy.

- define cell-free DNA (cfDNA)

Cell-free DNA is fragmented DNA that circulates in the bloodstream outside of cells, released through processes such as apoptosis, necrosis, or active secretion. In pregnancy, maternal plasma contains a mixture of maternal and fetal cell-free DNA, with the fetal component, known as cell-free fetal DNA, originating from placental trophoblast cells. The proportion of fetal DNA in maternal plasma, termed fetal fraction, varies with gestational age and maternal factors. Non-invasive prenatal screening relies on the analysis of this mixed population to infer fetal chromosomal status based on relative sequence abundance.

- describe cell-free fetal DNA (cffDNA)

Cell-free fetal DNA is the fraction of cell-free DNA in maternal plasma that is derived from the placenta and reflects the fetal genome. It constitutes a variable proportion of total cell-free DNA, typically ranging from 4% to 20% during the first and second trimesters. The size profile of cell-free fetal DNA is distinct from maternal DNA, with a shorter average fragment length, which can be exploited for enrichment. The analysis of cell-free fetal DNA enables the non-invasive detection of chromosomal aneuploidies by comparing the relative abundance of sequences from each chromosome to a reference baseline.

- define fetal fraction (ff)

Fetal fraction is the proportion of cell-free DNA in maternal plasma that originates from the fetus, expressed as a percentage of the total cell-free DNA. It is a critical parameter in non-invasive prenatal screening, as low fetal fraction increases the risk of false-negative or inconclusive results. Fetal fraction is estimated using sex chromosome-specific read counts in male pregnancies or through statistical modeling of X-chromosome representation in female pregnancies. A minimum fetal fraction threshold, typically 4% to 5%, is required for reliable analysis, and samples falling below this threshold are excluded from reporting.

- describe next generation sequencing (NGS)

Next generation sequencing is a high-throughput technology that enables the parallel sequencing of millions of DNA fragments in a single run. It has revolutionized non-invasive prenatal screening by allowing comprehensive, genome-wide analysis of cell-free DNA with high sensitivity and precision. NGS platforms such as Illumina HiSeq, MiSeq, and Ion Torrent generate vast volumes of short-read sequence data that are aligned to a reference genome and quantified to determine the relative abundance of chromosomal regions. The method is scalable, cost-effective, and capable of detecting both whole-chromosome and sub-chromosomal abnormalities.

- define library

A sequencing library is a collection of DNA fragments that have been prepared for sequencing by attaching adapter sequences, amplifying the molecules, and enriching for target regions. In non-invasive prenatal screening, the cell-free DNA extracted from maternal plasma is converted into a sequencing library through fragmentation, end repair, adapter ligation, and PCR amplification. The library is then loaded onto a sequencing platform for mass parallel sequencing. The quality and complexity of the library directly influence the accuracy and reproducibility of downstream analysis.

- describe adapter sequence

Adapter sequences are short, synthetic DNA oligonucleotides that are ligated to the ends of fragmented DNA to enable binding to sequencing platforms and to facilitate amplification and sequencing. In non-invasive prenatal screening, adapters contain unique molecular identifiers, sample barcodes, and universal primer binding sites that allow for multiplexing of multiple samples in a single sequencing run. The adapter sequences are designed to be compatible with specific sequencing chemistries and to minimize bias during amplification and sequencing.

- define sequencing bin

A sequencing bin is a predefined, non-overlapping genomic region of fixed size, typically ranging from 50 to 100 kilobases, into which the genome is partitioned for the purpose of quantifying sequence read counts. Each bin serves as an independent unit of measurement in non-invasive prenatal screening, allowing for the spatial resolution of copy number variation across chromosomes. The bins are selected to avoid repetitive elements and to ensure uniform mappability, enabling accurate and reproducible detection of aneuploidies and structural variants.

- describe sequence read

A sequence read is a single, contiguous sequence of nucleotides generated by a next-generation sequencing platform from a fragmented DNA molecule. In non-invasive prenatal screening, millions of sequence reads are generated from cell-free DNA and aligned to a reference genome to determine their chromosomal origin. The number of reads mapping to each sequencing bin is used to calculate the relative abundance of each chromosome and to identify deviations indicative of aneuploidy.

- define reference genome

A reference genome is a digitally assembled, annotated sequence of a representative individual’s genome, used as a standard template for aligning and interpreting sequencing data. In non-invasive prenatal screening, the human reference genome (e.g., GRCh37/hg19 or GRCh38/hg38) is used to map sequence reads to their chromosomal locations, enabling the quantification of read counts per bin and the calculation of chromosome-specific representation values. The reference genome provides the positional context necessary for detecting copy number variations and distinguishing fetal from maternal signals.

- describe Z-score

A Z-score is a statistical measure that quantifies the deviation of an observed value from the mean of a reference population, expressed in units of standard deviation. In non-invasive prenatal screening, the Z-score for each chromosome is calculated by comparing the chromosome representation value of a sample to the median representation observed in a large cohort of euploid pregnancies, normalized by the median absolute deviation. A high positive Z-score indicates overrepresentation consistent with trisomy, while a high negative Z-score indicates underrepresentation consistent with monosomy. The Z-score serves as the primary metric for risk assessment in screening algorithms.

- define ideogram

An ideogram is a graphical representation of a chromosome that displays the relative abundance of sequencing reads across its length, with each point corresponding to a sequencing bin and its associated standardized deviation score. The ideogram provides a spatial visualization of copy number variation, allowing for the identification of focal anomalies that may indicate maternal microduplications or microdeletions. In non-invasive prenatal screening, ideograms are used to validate chromosome-specific Z-scores and to exclude false-positive results arising from localized maternal variants.

- describe mapping of characteristic DNA sequences

Mapping of characteristic DNA sequences involves aligning sequencing reads to specific genomic loci in a reference genome to determine their chromosomal origin. This process relies on algorithms that match short sequence reads to unique, non-repetitive regions of the genome, ensuring accurate assignment of reads to their corresponding bins. The fidelity of mapping is critical for the detection of aneuploidy, as misalignment to homologous or repetitive regions can introduce noise and bias. High-quality mapping enables precise quantification of read counts and reliable detection of subtle copy number changes.

- define positive predictive value (PPV)

Positive predictive value is the proportion of positive test results that are true positives, calculated as the number of true positives divided by the total number of positive results. In non-invasive prenatal screening, PPV reflects the likelihood that a positive result corresponds to a true fetal aneuploidy. A high PPV is essential for clinical utility, as it reduces the number of unnecessary invasive procedures. The PPV is influenced by the prevalence of aneuploidy in the tested population and the specificity of the assay, with lower prevalence leading to lower PPV even with high specificity.

- describe risks of invasive procedures

Invasive prenatal procedures, such as amniocentesis and chorionic villus sampling, carry inherent risks including procedure-related miscarriage, infection, bleeding, and fetal injury. The risk of miscarriage associated with these procedures is estimated to be between 0.5% and 1.0%, and while relatively low, it is significant enough to warrant caution in clinical decision-making. The primary goal of non-invasive prenatal screening is to reduce the number of invasive procedures by accurately identifying low-risk pregnancies, thereby minimizing these risks while preserving diagnostic accuracy for high-risk cases.

- introduce non-invasive prenatal screening methods

Non-invasive prenatal screening methods utilize cell-free DNA in maternal plasma to detect fetal chromosomal abnormalities without the need for invasive procedures. These methods rely on next-generation sequencing to quantify the relative abundance of DNA fragments from each chromosome and to identify deviations indicative of aneuploidy. Various approaches have been developed, including shotgun sequencing, SNP-based analysis, and targeted sequencing, each with distinct advantages and limitations. The most widely adopted method is massively parallel shotgun sequencing, which provides genome-wide coverage and enables the detection of both whole-chromosome and sub-chromosomal abnormalities.

- describe maternal test sample

A maternal test sample is a blood specimen collected from a pregnant woman, typically during the first or early second trimester, for the purpose of non-invasive prenatal screening. The sample is processed to isolate plasma, from which cell-free DNA is extracted and sequenced. The composition of the cell-free DNA in the maternal test sample includes both maternal and fetal components, with the fetal fraction varying based on gestational age and maternal factors. The quality and quantity of the sample directly influence the accuracy and reliability of the screening result.

- describe cell-free fetal DNA in maternal test sample

Cell-free fetal DNA in the maternal test sample originates from apoptotic trophoblast cells of the placenta and constitutes a minority fraction of the total cell-free DNA. Its presence enables the non-invasive assessment of fetal genetic material, allowing for the detection of chromosomal aneuploidies without direct access to fetal tissues. The concentration of cell-free fetal DNA increases with gestational age and is typically detectable as early as four to five weeks of gestation. The accurate quantification of fetal DNA is essential for reliable screening, as low fetal fraction can lead to false-negative or inconclusive results.

- describe evaluation of fetal fraction

Evaluation of fetal fraction involves estimating the proportion of cell-free fetal DNA in maternal plasma using either sex chromosome-based methods or statistical modeling of autosomal read distributions. In male pregnancies, the presence of Y-chromosome sequences provides a direct measure of fetal DNA, while in female pregnancies, deviations in X-chromosome representation are used to infer fetal fraction. Accurate fetal fraction estimation is critical for determining sample suitability and for correcting biases in chromosome representation analysis.

- exclude samples with low fetal fraction

Samples with a fetal fraction below a predefined threshold, typically 4% to 5%, are excluded from analysis because the low abundance of fetal DNA increases the risk of false-negative or inconclusive results. Exclusion criteria are based on empirical validation studies that demonstrate reduced sensitivity and specificity below this threshold. Requiring a minimum fetal fraction ensures that only samples with sufficient fetal DNA content are reported, thereby maintaining the analytical performance of the screening assay.

- describe methods for quantifying cell-free fetal DNA

Methods for quantifying cell-free fetal DNA include counting Y-chromosome-specific reads in male pregnancies, modeling X-chromosome representation in female pregnancies, and applying machine learning algorithms to autosomal read distributions. These methods are calibrated using large reference cohorts to account for variations in maternal weight, gestational age, and sequencing platform. The most reliable approaches combine multiple metrics to improve accuracy and reduce the impact of technical noise.

- describe establishing fetal fraction using NGS data

Establishing fetal fraction using next-generation sequencing data involves aligning sequence reads to the reference genome, counting reads mapping to sex chromosomes or autosomal regions, and applying statistical models to estimate the proportion of fetal DNA. For male pregnancies, the ratio of Y-chromosome reads to total autosomal reads is used to calculate fetal fraction. For female pregnancies, the relative underrepresentation of the X chromosome compared to autosomes is modeled using a regularized regression algorithm trained on euploid samples. The resulting fetal fraction estimate is used to validate sample quality and to inform downstream analysis.

- estimate fetal fraction for male-bearing pregnancies

Fetal fraction for male-bearing pregnancies is estimated by quantifying the number of sequence reads mapping to the Y chromosome and normalizing them against the total number of autosomal reads. Since the Y chromosome is absent in maternal DNA, any Y-chromosome reads are assumed to originate from the fetus. The ratio of Y-chromosome reads to total autosomal reads is then converted into a percentage using a calibration curve derived from a large cohort of known male pregnancies.

- estimate fetal fraction for female-bearing pregnancies

Fetal fraction for female-bearing pregnancies is estimated by analyzing the relative representation of the X chromosome compared to autosomes. In a euploid female fetus, the X chromosome is expected to have a representation value approximately equal to that of autosomes. In the presence of fetal DNA, the X chromosome representation is reduced due to the dilution effect of maternal X chromosomes. A proprietary regression model, trained on thousands of euploid female pregnancies, is used to estimate fetal fraction based on the deviation of X-chromosome representation from the expected baseline.

- describe regularized regression model

A regularized regression model is a statistical technique that estimates relationships between variables while penalizing excessive complexity to prevent overfitting. In non-invasive prenatal screening, a regularized regression model is used to estimate fetal fraction in female pregnancies by relating X-chromosome representation to known fetal fraction values derived from a training cohort. The model incorporates multiple covariates, including gestational age, maternal weight, and sequencing depth, to improve accuracy and generalizability across diverse populations.

- describe estimation of fetal fraction using multiple methods

Estimation of fetal fraction is enhanced by combining multiple independent methods, such as Y-chromosome counting in male pregnancies and X-chromosome modeling in female pregnancies, to cross-validate results and reduce error. In cases where one method is ambiguous or unreliable, the alternative method provides a backup estimate. This multi-method approach increases the robustness of fetal fraction estimation and ensures that a high proportion of samples meet the minimum threshold for reliable analysis.

- analyze cell-free DNA for detection of fetal aneuploidy

Analysis of cell-free DNA for the detection of fetal aneuploidy involves sequencing the DNA fragments, aligning them to a reference genome, counting reads per bin, correcting for GC bias, and calculating chromosome-specific Z-scores. The method distinguishes between fetal and maternal contributions by evaluating the spatial consistency of read count deviations across the chromosome. Only those results with uniform, whole-chromosome deviations are classified as positive for fetal aneuploidy, while focal deviations are flagged as potential maternal variants.

- describe high-risk pregnancies

High-risk pregnancies are those in which the likelihood of fetal chromosomal aneuploidy is increased due to maternal age, abnormal serum screening, ultrasound findings, or prior history of aneuploidy. These pregnancies are the primary target population for non-invasive prenatal screening, as the higher prevalence of aneuploidy improves the positive predictive value of the test. Screening in high-risk populations allows for more accurate risk stratification and reduces the number of unnecessary invasive procedures.

- sequence cell-free DNA with next generation sequencing

Cell-free DNA is sequenced using next-generation sequencing platforms that generate millions of short reads in parallel. The sequencing process involves library preparation, cluster amplification, and sequencing-by-synthesis using reversible dye terminators. The resulting data are streamed to a bioinformatics pipeline for alignment, normalization, and statistical analysis. The high depth of coverage enables the detection of subtle copy number variations with high sensitivity and specificity.

- describe alignment of sequence reads to bins of a reference genome

Alignment of sequence reads to bins of a reference genome involves mapping each read to its most likely chromosomal origin using a reference sequence and a bioinformatics algorithm. Reads are assigned to predefined bins based on their genomic coordinates, and the number of reads per bin is counted to determine the relative abundance of each chromosomal region. Accurate alignment is essential for reliable detection of aneuploidy, as misalignment to repetitive or homologous regions can introduce bias and reduce sensitivity.

- describe bin read count scaling

Bin read count scaling is the process of normalizing the number of reads in each bin to account for differences in sequencing depth and total DNA input across samples. This is achieved by dividing the read count in each bin by the total number of autosomal reads in the sample, resulting in a proportional representation that is comparable across samples. Scaling ensures that variations in sequencing yield do not confound the detection of true biological differences in chromosomal abundance.

- correct high order artifacts

High-order artifacts are systematic biases introduced during sequencing, library preparation, or data processing that affect the distribution of read counts across the genome. These include GC content bias, amplification bias, and batch effects. Correction of high-order artifacts is performed using statistical models trained on euploid samples to identify and remove non-biological variation. This step is critical for improving the precision of chromosome representation values and reducing false-positive rates.

- calculate bin-specific test parameter

The bin-specific test parameter is calculated by normalizing the read count in each bin, applying GC correction, and removing high-order artifacts using a proprietary algorithm. The resulting value represents the standardized deviation of that bin’s representation from the expected median across a reference population. This parameter is the fundamental unit of analysis for detecting both whole-chromosome and sub-chromosomal abnormalities.

- determine relative abundance of genetic materials

The relative abundance of genetic materials is determined by comparing the normalized read counts of each chromosome to the total autosomal read count. This provides a proportional measure of the contribution of each chromosome to the cell-free DNA pool. Deviations from the expected ratio indicate potential aneuploidy and are used to calculate chromosome-specific Z-scores.

- calculate chromosomal representation

Chromosomal representation is calculated as the sum of all bin-specific test parameters for a given chromosome, divided by the number of bins on that chromosome. This value reflects the overall copy number status of the chromosome and is used as the basis for calculating the chromosome-specific Z-score. High representation indicates trisomy, while low representation indicates monosomy.

- compare chromosomal representation to reference

Chromosomal representation is compared to a reference distribution derived from a large cohort of euploid pregnancies to determine whether the observed value is statistically significant. The reference distribution accounts for population variability and technical noise, ensuring that the comparison is robust and reproducible. Deviations beyond a predefined threshold are flagged as potential aneuploidies.

- calculate chromosome-specific Z-score

The chromosome-specific Z-score is calculated by subtracting the median chromosomal representation of the reference population from the sample’s representation and dividing by the median absolute deviation. This standardized score quantifies the deviation of the sample from the norm and is used to classify the result as positive, negative, or indeterminate. A Z-score above a defined threshold indicates a likely fetal aneuploidy.

- interpret Z-score for aneuploidy detection

Interpretation of the Z-score for aneuploidy detection involves comparing the magnitude and direction of the score to established thresholds. A Z-score greater than +4 indicates a likely trisomy, while a Z-score less than -4 indicates a likely monosomy. Intermediate scores between +3 and +4 are flagged for further investigation using ideogram analysis to determine whether the deviation is uniform or localized.

- attribute abnormalities to fetal genome

Abnormalities are attributed to the fetal genome only when the chromosomal representation deviation is uniform across the entire chromosome and consistent with a whole-chromosome gain or loss. If the deviation is focal or discontinuous, it is attributed to a maternal origin, such as a microduplication or microdeletion, and is not reported as a fetal aneuploidy.

- use Z-score for detecting aneuploidy

The Z-score is the primary metric used for detecting aneuploidy in non-invasive prenatal screening. It provides a quantitative, standardized measure of chromosomal imbalance that is independent of sequencing depth and sample quality. The Z-score is calculated for each chromosome and used to trigger further analysis, including ideogram review, when it exceeds the diagnostic threshold.

- detect chromosomal trisomy or monosomy

Chromosomal trisomy or monosomy is detected when the chromosome-specific Z-score exceeds predefined thresholds and the corresponding ideogram demonstrates a uniform, whole-chromosome deviation. Trisomy is indicated by a positive Z-score and consistent elevation across all bins, while monosomy is indicated by a negative Z-score and consistent depression. These patterns are distinct from focal anomalies and are used to make a positive screening result.

- detect partial chromosomal duplication or deletion

Partial chromosomal duplication or deletion is detected when the ideogram reveals a localized deviation affecting only a subset of bins on a chromosome. These patterns are inconsistent with whole-chromosome aneuploidy and are flagged as potential maternal variants. The size and location of the deviation are analyzed to determine whether it corresponds to a known microduplication or microdeletion syndrome.

- detect chromosomal mosaicism

Chromosomal mosaicism is detected when the chromosome-specific Z-score falls in an intermediate range, between the typical thresholds for euploidy and full aneuploidy. This pattern suggests the presence of a mixed population of euploid and aneuploid cells in the fetal genome. Mosaicism is confirmed through follow-up invasive testing, as the Z-score alone cannot determine the proportion of abnormal cells.

- detect chromosome translocations

Chromosome translocations are detected when the ideogram reveals an unusual pattern of read count distribution, such as an abrupt shift in representation at a specific breakpoint or an imbalance between two chromosomes. These patterns suggest the presence of a structural rearrangement, such as a Robertsonian translocation, which may result in unbalanced gametes and fetal aneuploidy. Translocations are confirmed through karyotype analysis or microarray.

- provide effective option for detecting fetal aneuploidies

This method provides an effective option for detecting fetal aneuploidies by combining high-resolution binning, spatial ideogram analysis, and maternal variant exclusion to achieve superior specificity and positive predictive value. Unlike conventional methods that rely solely on aggregate Z-scores, this approach distinguishes between fetal and maternal sources of copy number variation, reducing false positives and improving clinical utility.

- improve positive predictive values

The method improves positive predictive values by eliminating false-positive results caused by maternal microduplications, microdeletions, and global copy number abnormalities. By requiring uniform chromosomal deviation across the entire chromosome, the method ensures that only true fetal aneuploidies are reported, thereby increasing the proportion of confirmed cases among all positive results.

- consider maternal chromosome variations

Maternal chromosome variations, including microduplications, microdeletions, and global copy number changes, are actively considered in the analytical workflow. These variants are not ignored but are instead identified and excluded from reporting as fetal aneuploidies. This proactive consideration significantly enhances the specificity of the screening test.

- distinguish fetal aneuploidies from maternal variations

Fetal aneuploidies are distinguished from maternal variations by analyzing the spatial consistency of bin-specific test parameters. Whole-chromosome deviations that are uniform across the entire chromosome are classified as fetal, while focal deviations that are localized to a subregion are classified as maternal. This distinction is made using ideogram visualization and statistical modeling of deviation patterns.

- examine fetal genome karyotype

The fetal genome karyotype is examined indirectly through the analysis of cell-free DNA, with the goal of inferring chromosomal copy number status without direct access to fetal cells. The method provides a non-invasive, high-resolution approximation of the fetal karyotype, enabling early detection of aneuploidies with clinical accuracy.

- calculate chromosome-specific Z-score for multiple chromosomes

Chromosome-specific Z-scores are calculated simultaneously for all autosomes and sex chromosomes, allowing for comprehensive screening of common aneuploidies in a single assay. Each chromosome is analyzed independently, ensuring that abnormalities on one chromosome do not interfere with the detection of abnormalities on another.

- recognize maternal contribution and exclude false-positives

The method recognizes the contribution of maternal DNA to the cell-free DNA pool and employs spatial analysis to exclude false-positive results caused by maternal chromosomal variants. This recognition is the key innovation that enables the method to achieve higher positive predictive values than previous screening technologies.

- pinpoint source of genetic variations to discrete chromosomal region

The source of genetic variations is pinpointed to discrete chromosomal regions through the use of high-resolution binning and ideogram visualization. This allows for the identification of whether a deviation is localized to a small segment or affects the entire chromosome, enabling accurate classification of the origin of the variation.

- define bin-specific test parameter

The bin-specific test parameter is a standardized measure of the relative abundance of DNA fragments in each sequencing bin, corrected for GC content and technical artifacts. It represents the deviation of that bin’s read count from the expected median value in a euploid population and serves as the fundamental unit for spatial analysis.

- calculate bin-specific test parameter

The bin-specific test parameter is calculated by normalizing the raw read count in each bin to the total autosomal read count, applying a GC correction using a locally weighted regression model, and removing high-order artifacts using a proprietary algorithm. The resulting value is expressed as a Z-score relative to a reference population, ensuring comparability across samples.

- determine consistency of bin-specific test parameters

Consistency of bin-specific test parameters is determined by evaluating the spatial distribution of deviations across the chromosome. A high degree of consistency, indicated by a uniform elevation or depression across nearly all bins, supports a fetal origin. Inconsistency, indicated by focal peaks or troughs, suggests a maternal variant.

- detect maternal microduplication or microdeletion

Maternal microduplication or microdeletion is detected when the ideogram reveals a localized deviation affecting fewer than a substantial portion of the chromosome. These patterns are inconsistent with fetal aneuploidy and are flagged for maternal confirmation, preventing misclassification as a fetal abnormality.

- define large-scale difference

A large-scale difference is defined as a deviation affecting a substantial portion of the chromosome, typically exceeding 70% of its length, and characterized by uniform elevation or depression of bin-specific test parameters. This definition distinguishes true fetal aneuploidies from smaller, localized maternal variants.

- detect fetal aneuploidy

Fetal aneuploidy is detected when the chromosome-specific Z-score exceeds a predefined threshold and the corresponding ideogram demonstrates a consistent, whole-chromosome deviation. This dual requirement ensures that only true fetal abnormalities are reported, minimizing false positives.

- confirm fetal aneuploidy

Fetal aneuploidy is confirmed by the concordance of a high chromosome-specific Z-score with a uniform ideogram pattern across the entire chromosome. No focal anomalies are present, and the deviation is consistent with known patterns of trisomy or monosomy. Confirmation is internal to the assay and does not require invasive testing for reporting.

- exclude false-positive diagnosis

False-positive diagnosis is excluded by rejecting any result in which the chromosomal deviation is localized to a subregion of the chromosome, indicating a maternal origin. These results are flagged for further investigation but are not reported as fetal aneuploidies.

- calculate chromosome-specific Z-score

The chromosome-specific Z-score is calculated as the mean of all bin-specific test parameters for a given chromosome, normalized to the median and median absolute deviation of a reference population. This score provides a standardized measure of chromosomal imbalance that is used to classify the result as positive, negative, or indeterminate.

- analyze ideogram for consistency

The ideogram is analyzed for consistency by examining the spatial pattern of bin-specific test parameters across the entire chromosome. A consistent pattern, characterized by a smooth, uniform elevation or depression, supports a fetal origin. An inconsistent pattern, characterized by sharp peaks or valleys, suggests a maternal variant.

- determine substantial portion of chromosome

A substantial portion of the chromosome is determined as the minimum contiguous segment that must exhibit deviation to be considered indicative of a fetal aneuploidy. This threshold is empirically derived from a training cohort of confirmed fetal trisomies and is set at 70% of the chromosome length.

- calculate standard deviation of residues

The standard deviation of residues is calculated as the variability of bin-specific test parameters around the mean chromosomal representation. A low standard deviation indicates uniform deviation consistent with fetal aneuploidy, while a high standard deviation indicates localized anomalies consistent with maternal variants.

- improve positive predictive value of NIPS

The method improves the positive predictive value of non-invasive prenatal screening by eliminating false-positive results caused by maternal chromosomal variants. By requiring uniform, whole-chromosome deviation for a positive result, the method ensures that only true fetal aneuploidies are reported, thereby increasing the proportion of confirmed cases among all positive results.

- perform ultra-sonographic diagnosis

Ultra-sonographic diagnosis is performed as a complementary method to assess fetal anatomy and identify structural anomalies associated with chromosomal abnormalities. While not used as the primary diagnostic tool, ultrasound findings may support or challenge the results of non-invasive prenatal screening and are considered in clinical decision-making.

- perform amniocentesis

Amniocentesis is performed as a confirmatory diagnostic procedure following a positive non-invasive prenatal screening result. It involves the extraction of amniotic fluid containing fetal cells, which are then cultured and analyzed by karyotype or microarray to determine the definitive chromosomal status of the fetus.

- perform conventional first or second trimester screenings

Conventional first or second trimester screenings, including maternal serum biomarkers and nuchal translucency ultrasound, are performed prior to non-invasive prenatal screening in many clinical settings. These methods are used to identify high-risk pregnancies that are candidates for advanced screening, but they are superseded by non-invasive prenatal screening due to its superior accuracy.

- use next-generation sequencing methods

Next-generation sequencing methods are employed to generate high-throughput, genome-wide data from cell-free DNA. These methods enable the simultaneous analysis of all chromosomes with high sensitivity and precision, making them the foundation of modern non-invasive prenatal screening.

- use shotgun massively parallel sequencing

Shotgun massively parallel sequencing is used to randomly fragment and sequence the entire cell-free DNA population without targeted enrichment. This unbiased approach ensures comprehensive coverage of the genome and enables the detection of both whole-chromosome and sub-chromosomal abnormalities.

- use sequencing-by-synthesis with reversible dye terminators

Sequencing-by-synthesis with reversible dye terminators is the primary chemistry used in the sequencing platform. This method allows for the sequential addition of fluorescently labeled nucleotides, with imaging after each cycle to determine the base identity. It provides high accuracy and scalability, making it ideal for non-invasive prenatal screening.

- use sequencing-by-ligation

Sequencing-by-ligation is an alternative method that uses DNA ligase to attach fluorescently labeled probes to complementary sequences. While less commonly used in clinical NIPS, it offers high multiplexing capacity and may be employed in specialized applications.

- use single molecule sequencing

Single molecule sequencing is a method that sequences individual DNA molecules without amplification, reducing bias and enabling direct detection of epigenetic modifications. While not currently used in routine NIPS, it holds promise for future applications requiring ultra-high resolution.

- describe Ion Torrent sequencing system

The Ion Torrent sequencing system is a semiconductor-based platform that detects hydrogen ions released during DNA synthesis. It provides rapid, cost-effective sequencing with high throughput and is compatible with the sample preparation protocols used in non-invasive prenatal screening.

- describe 454 sequencing system

The 454 sequencing system is a pyrosequencing platform that detects light emitted during nucleotide incorporation. Although largely superseded by newer technologies, it was among the first platforms used for high-throughput sequencing and demonstrated the feasibility of non-invasive prenatal screening.

- describe reversible dye-terminators sequencing

Reversible dye-terminators sequencing involves the use of modified nucleotides that terminate DNA synthesis after incorporation and are labeled with fluorescent dyes. After imaging, the dye and terminator are cleaved, allowing the next nucleotide to be added. This method enables highly accurate, long-read sequencing and is the foundation of Illumina platforms.

- describe Helicos single-molecule sequencing

Helicos single-molecule sequencing sequences individual DNA molecules without amplification, reducing bias and enabling direct detection of rare variants. While not used in current clinical NIPS, it provides a model for future technologies requiring ultra-high sensitivity.

- describe sequencing by synthesis

Sequencing by synthesis is the process of determining the sequence of a DNA molecule by adding nucleotides one at a time and detecting their incorporation. This method is the basis of most next-generation sequencing platforms and enables high-throughput, accurate sequencing of cell-free DNA.

- describe MiSeq personal sequencing system

The MiSeq personal sequencing system is a benchtop next-generation sequencing platform that provides high accuracy and moderate throughput, making it suitable for clinical laboratories performing non-invasive prenatal screening on a smaller scale.

- describe sequencing by ligation

Sequencing by ligation involves the hybridization of fluorescently labeled oligonucleotide probes to a DNA template, followed by ligation and imaging. This method allows for multiplexed detection of multiple sequence variants and may be used in targeted NIPS applications.

- describe SOLiD sequencing

SOLiD sequencing is a ligation-based platform that uses two-base encoding to enhance accuracy. Although no longer in widespread use, it demonstrated the potential for high-fidelity sequencing in clinical applications.

- describe SMRT sequencing

SMRT sequencing, or single-molecule real-time sequencing, is a method that observes DNA synthesis in real time using zero-mode waveguides. It enables long-read sequencing and detection of epigenetic modifications, offering potential future applications in non-invasive prenatal screening.

- correct GC-sequencing biases

GC-sequencing biases are corrected using a locally weighted regression model that adjusts read counts based on the GC content of each sequencing bin. This correction accounts for the tendency of high-GC regions to be underrepresented and low-GC regions to be overrepresented in sequencing data, ensuring accurate quantification of chromosomal representation.

- provide example of correcting GC-sequencing biases

An example of correcting GC-sequencing biases involves applying a loess smoothing function to the relationship between GC content and normalized read count across a training cohort of euploid samples. The model predicts the expected read count for each bin based on its GC content, and the observed count is adjusted to match this prediction, eliminating systematic bias.

- describe use of various sequencing methods

Various sequencing methods, including shotgun sequencing, targeted sequencing, and single-molecule sequencing, may be employed depending on the clinical application, throughput requirements, and cost constraints. The method is adaptable to any platform that generates sufficient read depth and coverage for bin-based analysis.

- describe use of various sequencing technologies

Various sequencing technologies, including Illumina, Ion Torrent, and Oxford Nanopore, may be used to generate the data required for non-invasive prenatal screening. The method is technology-agnostic and relies on the output of sequence reads aligned to a reference genome, regardless of the underlying chemistry.

- describe use of various sequencing platforms

Various sequencing platforms, from high-throughput HiSeq systems to benchtop MiSeq instruments, may be utilized to perform non-invasive prenatal screening. The analytical pipeline is standardized across platforms, ensuring consistent performance regardless of the instrument used.

- summarize methods for improving positive predictive value of NIPS

Methods for improving the positive predictive value of non-invasive prenatal screening include high-resolution binning, GC bias correction, fetal fraction estimation, ideogram-based spatial analysis, and exclusion of maternal chromosomal variants. Together, these methods reduce false-positive results caused by biological and technical confounders, thereby increasing the proportion of true fetal aneuploidies among all positive results.

## EXAMPLES

### Example 1: Assay Development

- introduce patient sample collection

Patient sample collection for the development of the non-invasive prenatal screening assay was conducted under strict ethical guidelines and institutional review board approval. Blood samples were obtained from pregnant women across multiple clinical sites, including private obstetric practices and academic medical centers. All participants provided written informed consent prior to sample collection, and samples were de-identified to protect patient privacy. The collection protocol was standardized to ensure consistency in sample handling, storage, and transport.

- describe sample collection process

The sample collection process involved drawing two 10 mL blood samples into Cell-Free DNA BCT tubes, which stabilize cell-free DNA and prevent leukocyte lysis. Tubes were stored at room temperature and transported to the laboratory within four days of collection. Upon arrival, samples were processed immediately to isolate plasma and extract cell-free DNA, minimizing degradation and ensuring high-quality sequencing material.

- outline sample sources

Sample sources included a diverse cohort of pregnant women from diverse ethnic backgrounds, with gestational ages ranging from 9 to 22 weeks. Samples were obtained from both high-risk and low-risk populations, including women with advanced maternal age, abnormal ultrasound findings, and prior history of aneuploidy. A subset of samples included known aneuploid pregnancies confirmed by invasive testing, as well as euploid controls.

- detail informed consent process

The informed consent process was conducted by trained genetic counselors who explained the purpose, benefits, limitations, and potential outcomes of non-invasive prenatal screening. Participants were informed that the test was a screening tool and that positive results would require confirmation through invasive diagnostic testing. Written consent was obtained prior to blood draw, and all documentation was stored securely in compliance with HIPAA and GDPR regulations.

- introduce next-generation sequencing

Next-generation sequencing was introduced as the core analytical platform for the assay, enabling high-throughput, genome-wide analysis of cell-free DNA. The method utilized Illumina HiSeq 2500 systems to generate millions of short reads per sample, providing the depth and resolution required for bin-based analysis.

- describe blood collection and processing

Blood collection was performed using standard venipuncture techniques, followed by immediate centrifugation to separate plasma from cellular components. Plasma was isolated using a double-centrifugation protocol to remove residual cells and platelets. The supernatant was aliquoted and stored at -80°C until DNA extraction.

- outline plasma isolation and centrifugation

Plasma isolation involved centrifugation at 1,600 × g for 10 minutes, followed by a second centrifugation at 3,200 × g for 20 minutes to ensure complete removal of cellular debris. The final plasma supernatant was transferred to sterile tubes and stored at -80°C to preserve cell-free DNA integrity.

- detail cell-free DNA extraction

Cell-free DNA was extracted from 4 mL of plasma using a magnetic bead-based purification system optimized for low-concentration, fragmented DNA. The extraction protocol included denaturation, binding, washing, and elution steps, yielding an average of 10 to 20 ng of cell-free DNA per milliliter of plasma.

- describe library preparation and PCR

Library preparation was performed using the NEBNext Ultra DNA Library Prep Kit, which included end repair, A-tailing, adapter ligation, and PCR amplification. A unique 10-base barcode was incorporated into each sample during PCR to enable multiplexing. Amplification was performed using a universal forward primer and a reverse primer containing the barcode sequence.

- outline PCR conditions and primer sequences

PCR conditions consisted of an initial denaturation at 98°C for 30 seconds, followed by 10 cycles of denaturation at 98°C for 10 seconds, annealing at 65°C for 30 seconds, and extension at 72°C for 30 seconds, with a final extension at 72°C for 5 minutes. The forward primer sequence was AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT, and the reverse primer sequence was CAAGCAGAAGACGGCATACGAGATXXXXXXXXXXGTGACTGGAGTTCAGACGTGTGCTCTTCCGATCT, where X denotes the 10-base barcode.

- detail PCR product purification and quantification

PCR products were purified using AMPure XP beads at a 1:1 bead-to-sample ratio to remove unincorporated primers and nucleotides. Purified libraries were quantified using the PicoGreen dsDNA assay on a microplate reader, and concentrations were normalized to 2 nM for pooling.

- introduce sequencing and library pooling

Sequencing was performed by pooling 12 samples per library, denaturing the pooled libraries, and diluting them to 15 pM. A 5% PhiX control was added to each pool to monitor sequencing quality. Libraries were loaded onto HiSeq 2500 flow cells for clonal amplification and sequencing.

- describe sequencing conditions and data analysis

Sequencing was performed using single-read 36-cycle chemistry, followed by 10 cycles of index sequencing. A minimum of 9 million reads per sample was required for analysis. Data were streamed to a secure server and processed using a proprietary bioinformatics pipeline that aligned reads to the hg19 reference genome and calculated bin-specific test parameters.

- introduce fetal fraction estimations

Fetal fraction estimations were introduced to assess sample quality and determine suitability for analysis. Estimations were performed using either Y-chromosome overrepresentation in male pregnancies or X-chromosome underrepresentation in female pregnancies, with a minimum threshold of 4% required for reporting.

- describe X chromosome underrepresentation method

The X chromosome underrepresentation method was based on the observation that in female fetuses, the X chromosome representation is reduced due to the dilution of fetal DNA by maternal X chromosomes. A proprietary regression model was developed to estimate fetal fraction by comparing the median X-chromosome representation of the sample to a reference cohort of euploid female pregnancies.

- outline Y chromosome overrepresentation method

The Y chromosome overrepresentation method involved counting reads mapping to the Y chromosome and normalizing them to total autosomal reads. The ratio was converted to fetal fraction using a calibration curve derived from a training set of known male pregnancies.

- detail regularized regression model for female fetuses

The regularized regression model for female fetuses incorporated gestational age, maternal weight, and sequencing depth as covariates to improve accuracy. The model was trained on 3,589 euploid female samples and validated on an independent cohort, achieving a correlation coefficient of 0.94 with invasive fetal fraction measurements.

- introduce GC correction

GC correction was introduced to mitigate sequencing bias caused by the differential amplification and capture efficiency of high- and low-GC regions. This correction was applied to all bin-specific read counts prior to Z-score calculation.

- describe GC content calculation and discretization

GC content was calculated for each sequencing bin by determining the percentage of guanine and cytosine bases within the bin’s sequence. Bins were discretized into 10 GC content bins, ranging from 20% to 80%, to facilitate modeling.

- outline local polynomial regression and loess function

Local polynomial regression using the loess function was applied to model the relationship between GC content and normalized read count across the training cohort. The fitted curve was used to predict expected read counts for each bin, and observed counts were adjusted to match the predicted values.

### Example 2: Assay Verification and Validation

- introduce assay verification and validation

Assay verification and validation were conducted using a comprehensive set of known euploid and aneuploid samples to establish analytical performance metrics. Verification involved testing samples with confirmed fetal karyotypes, while validation involved testing an independent cohort to assess generalizability and reproducibility.

- describe verification sample set and results

The verification sample set included 2,085 samples, comprising 69 trisomy 21, 20 trisomy 18, 17 trisomy 13, and 1,979 euploid controls. All aneuploid samples exhibited Z-scores greater than 8, and all euploid samples exhibited Z-scores less than 4, demonstrating complete separation between affected and unaffected pregnancies.

- outline validation sample set and results

The validation sample set included 552 samples, including 21 trisomy 21, 10 trisomy 18, 1 trisomy 13, and 1 XO. All aneuploid samples had Z-scores above 8, and all euploid samples had Z-scores below 4, confirming the robustness of the assay under independent conditions.

- detail effects of GC correction on performance

GC correction significantly improved the discrimination of trisomy 13 and trisomy 18, which are GC-rich chromosomes. Without correction, many trisomy 13 samples had Z-scores below 4, but after correction, all exceeded 8. Similarly, trisomy 18 samples showed improved separation from euploid controls.

- describe analysis of twin gestation samples

Analysis of 115 twin gestation samples revealed that all trisomy cases had Z-scores above 11, while all unaffected twins had Z-scores below 4. The assay demonstrated enhanced discrimination in twin pregnancies, likely due to higher total fetal DNA content.

- outline final validation for trisomy detection and fetal sex determination

Final validation confirmed 100% sensitivity and specificity for trisomy detection and 99.7% accuracy for fetal sex determination. Concordance with reference methods was 100% across all tested parameters.

### Example 3: Clinical Implementations

- describe clinical implementations of NIPS assay

The NIPS assay was implemented in a clinical reference laboratory serving over 10,000 pregnant women annually. The assay was integrated into routine prenatal care, with results reported within seven days of sample receipt. Strict sample acceptance criteria were enforced, including minimum fetal fraction and quality metrics.

- specify sample acceptance criteria

Sample acceptance criteria included a minimum fetal fraction of 4%, a minimum of 9 million reads per sample, and a coefficient of variation for bin-specific test parameters below 15%. Samples failing these criteria were rejected and resampled.

- report Z-score cutoffs for clinical implementation

Z-score cutoffs for clinical implementation were set at ≤4 for negative results and >8 for positive results. Results between 4 and 8 were flagged as intermediate and required ideogram review.

- present results of first 10,000 clinical samples

Of the first 10,000 clinical samples, 180 (1.8%) yielded abnormal results. Trisomy 21 was detected in 103 cases, trisomy 18 in 36, trisomy 13 in 21, and sex chromosome aneuploidies in 17. Four cases were false positives due to maternal microduplications.

- summarize abnormal NIPS results

Abnormal NIPS results were primarily due to true fetal aneuploidies, with maternal microduplications accounting for 25 of the 180 positive results. All maternal variants were identified through ideogram analysis and excluded from reporting as fetal aneuploidies.

- explain causes of unreported results

Unreported results were primarily due to low fetal fraction (0.59%) or technical failures such as insufficient read depth or poor library quality (0.29%).

- introduce maternal microduplication issue

Maternal microduplications were identified as a significant source of false-positive results, particularly for chromosomes 18 and 21. These variants were initially misclassified as fetal trisomies until ideogram analysis revealed focal deviations.

- describe method to identify maternal microduplications

Maternal microduplications were identified by analyzing the ideogram for localized elevations in bin-specific test parameters that affected less than 70% of the chromosome. These patterns were confirmed by maternal microarray analysis.

- present results of maternal microduplication identification

Twenty-five cases of maternal microduplications were identified among the first 10,000 samples. All were confirmed by microarray, and none were reported as fetal aneuploidies.

- explain process of confirming suspected microduplications

Suspected maternal microduplications were confirmed by extracting maternal DNA from the buffy coat and performing chromosomal microarray analysis. Results were reviewed by a board-certified clinical geneticist.

- report PPV improvement after identifying maternal microduplications

After excluding maternal microduplications, the PPV for trisomy 21 increased from 91% to 100%, for trisomy 18 from 73% to 100%, and for trisomy 13 from 39% to 85%.

- describe case of intermediate Z-scores

Two cases presented with intermediate Z-scores of 5.11 and 6.93 for chromosomes 21 and 18, respectively. Ideogram analysis revealed focal duplications, and maternal microarray confirmed the variants.

- use chromosomal ideograms to investigate intermediate Z-scores

Chromosomal ideograms were used to visualize the spatial distribution of bin-specific test parameters, revealing that the elevated Z-scores were confined to discrete regions rather than spanning the entire chromosome.

- confirm maternal microduplications using microarray analysis

Microarray analysis of maternal DNA confirmed the presence of 1.2 Mb and 2.1 Mb microduplications on chromosomes 21 and 18, respectively, validating the assay’s ability to distinguish maternal from fetal origins.

- report NPV for Trisomies 21, 13, and 18

The negative predictive value for trisomies 21, 18, and 13 was greater than 99.9%, with no false negatives observed in over 10,000 samples.

- introduce maternal global copy number abnormalities issue

Maternal global copy number abnormalities, including large-scale duplications and deletions, were identified as a rare but significant cause of false-positive results.

- describe method to identify maternal global copy number abnormalities

Global abnormalities were identified by reviewing the entire genome for multiple chromosomes with elevated or depressed Z-scores. When more than three chromosomes showed deviation, the result was flagged for maternal investigation.

- present results of maternal global copy number abnormalities identification

Six cases of maternal global copy number abnormalities were identified, all associated with uterine fibroids. These cases were excluded from reporting as fetal aneuploidies.

- report cases of mosaicism and translocations

One case of mosaic trisomy 21 was detected with a Z-score of 5.57, confirmed by amniocentesis. One Robertsonian translocation involving chromosomes 14 and 21 was detected with a Z-score of 30.78.

- describe detection of mosaic Down syndrome

Mosaic Down syndrome was detected in three cases, with Z-scores ranging from 3.57 to 8.41. Amniocentesis confirmed mosaicism with trisomic cell ratios ranging from 25% to 75%.

- report analytical sensitivity of NIPS assay for mosaic Down syndrome

The analytical sensitivity for mosaic Down syndrome was estimated at 85%, with detection possible at fetal trisomic cell fractions as low as 25%.

- introduce sex chromosome aneuploidies issue

Sex chromosome aneuploidies were complicated by maternal mosaicism, which could mimic fetal aneuploidy.

- describe detection of fetal sex chromosome aneuploidy

Fetal sex chromosome aneuploidies were detected using fetal fraction estimates derived from X and Y chromosome representation. Maternal mosaicism was identified when fetal fraction exceeded 50% in a female fetus.

- report cases of twins with elevated Z-scores

Four twin pregnancies had elevated Z-scores for trisomy 21. In two cases, one twin was affected and the other was euploid. In one case, a teratoma was present, and in one case, fetal demise occurred.

- describe results of twin pregnancy cases

Results demonstrated that the assay could detect fetal aneuploidy in twin pregnancies with high accuracy, even when one twin was unaffected.

- introduce PPV of previously available NIPS methods

Previously available NIPS methods reported PPVs of approximately 90% for trisomy 21, 65% for trisomy 18, and 40% for trisomy 13.

- report PPV of previously available NIPS methods

The PPV for trisomy 21 was 91%, for trisomy 18 was 73%, for trisomy 13 was 39%, and for sex chromosome aneuploidies was 49%.

- introduce PPV of present NIPS method

The present NIPS method, incorporating ideogram-based maternal variant exclusion, achieved a PPV of 100% for trisomy 21, 100% for trisomy 18, and 85% for trisomy 13.

- report PPV of present NIPS method for Trisomies 21, 18, and 13

The PPV for trisomy 21 was 100%, for trisomy 18 was 100%, and for trisomy 13 was 85%.

- report PPV of present NIPS method for sex chromosome aneuploidies and microdeletions

The PPV for sex chromosome aneuploidies was 95%, and for microdeletions, including DiGeorge syndrome, was 100%.

## EQUIVALENTS

- disclaim limitations

The invention is not limited to the specific embodiments described herein. Modifications and variations may be made without departing from the scope of the invention.

- define functionally equivalent

Functionally equivalent methods, systems, or components are those that perform substantially the same function in substantially the same way to achieve substantially the same result, even if implemented differently.

- interpret Markush groups

Markush groups are interpreted to include all possible combinations of the listed elements, including subcombinations and individual members, unless explicitly excluded.

- define range boundaries

Range boundaries are inclusive unless otherwise stated, and any sub-range within a disclosed range is considered disclosed.

- incorporate prior art

All prior art references cited herein are incorporated by reference in their entirety for all purposes.