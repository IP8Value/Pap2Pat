## BACKGROUND

- Human birth defects represent a significant global health burden, affecting approximately 3% to 5% of all liveborn infants. These conditions encompass a wide spectrum of structural, functional, and metabolic anomalies that may arise during embryonic or fetal development, often leading to lifelong disability, increased healthcare costs, and emotional strain on families. Despite advances in prenatal imaging and biochemical screening, the underlying genetic etiology remains unidentified in a substantial proportion of cases, underscoring the need for more precise, comprehensive, and non-invasive diagnostic tools.

- In China, the incidence of birth defects aligns with global estimates, with regional variations influenced by environmental, socioeconomic, and demographic factors. National surveillance data indicate that birth defects contribute to nearly 10% of neonatal mortality and represent one of the leading causes of infant morbidity. The burden is further amplified by the high prevalence of consanguineous marriages in certain populations and limited access to advanced genetic services in rural areas, resulting in delayed diagnosis and reduced opportunities for early intervention.

- A substantial fraction of birth defects—estimated at 15% to 25%—can be attributed to identifiable genetic disorders, including single-gene mutations, chromosomal rearrangements, and copy number variants. Over 8,600 distinct genetic conditions have been cataloged, many of which are severe, life-limiting, and lack curative therapies. Conditions such as cystic fibrosis, spinal muscular atrophy, sickle cell anemia, and Tay-Sachs disease exemplify the clinical urgency for early detection, particularly in populations with elevated carrier frequencies.

- Chromosomal abnormalities constitute a major category of genetic birth defects, arising from errors in meiosis or mitosis that lead to deviations from the normal diploid complement. These abnormalities may involve entire chromosomes or large segments, resulting in aneuploidies, deletions, duplications, or translocations. The most prevalent chromosomal aneuploidies include trisomy 21 (Down syndrome), trisomy 18 (Edwards syndrome), and trisomy 13 (Patau syndrome), each associated with profound developmental, cognitive, and physical impairments. Sex chromosome aneuploidies, such as Turner and Klinefelter syndromes, are also frequently encountered and may present with subtle phenotypes that are easily overlooked without targeted screening.

- Chromosomal aneuploidy arises primarily from meiotic nondisjunction, a failure of homologous chromosomes or sister chromatids to segregate properly during gametogenesis. Maternal age is the most well-established risk factor, with the incidence of trisomy 21 increasing exponentially after age 35. However, aneuploidy can also originate from paternal meiotic errors or post-zygotic mitotic events, and the underlying molecular mechanisms remain incompletely understood. Aberrant meiotic recombination patterns, including altered crossover frequency and misplaced breakpoints, have been implicated in the genesis of these errors, particularly in maternal oocytes.

- Common genetic diseases resulting from chromosomal aneuploidy include not only the classic trisomies but also microdeletion and microduplication syndromes such as DiGeorge syndrome (22q11.2 deletion), Cri du Chat syndrome (5p deletion), and Wolf-Hirschhorn syndrome (4p deletion). These conditions are characterized by multi-system involvement, including cardiac, craniofacial, neurological, and developmental abnormalities. While individually rare, collectively they account for a significant proportion of congenital anomalies detected prenatally and underscore the need for expanded screening panels beyond trisomy 21.

- Traditional prenatal screening methods, such as maternal serum biomarker testing and ultrasound-based nuchal translucency measurement, offer low sensitivity and high false-positive rates, leading to unnecessary invasive diagnostic procedures such as amniocentesis and chorionic villus sampling. These procedures carry inherent risks of miscarriage and are often performed too late in gestation to allow for timely decision-making. Furthermore, they are unable to detect monogenic disorders or subtle copy number changes that do not manifest as gross anatomical abnormalities.

- Non-invasive prenatal screening (NIPS) has revolutionized prenatal care by enabling the analysis of fetal genetic material derived from cell-free DNA circulating in maternal plasma. This approach eliminates the risks associated with invasive procedures and can be performed as early as the first trimester. The discovery that fetal cell-free DNA fragments, originating from apoptotic placental trophoblast cells, constitute a measurable fraction of total maternal plasma DNA opened the door to molecular diagnostics without direct fetal sampling.

- The development of NIPS technology has been propelled by advances in next-generation sequencing and bioinformatic algorithms capable of detecting subtle deviations in genomic dosage and allelic ratios. Early implementations relied on low-coverage whole-genome sequencing to infer fetal aneuploidy through read depth analysis. Subsequent refinements introduced single nucleotide polymorphism (SNP)-based approaches that leverage genotype information to improve accuracy and resolve complex scenarios such as maternal copy number variants and twin pregnancies.

- In China, NIPS has been rapidly adopted into clinical practice, with millions of tests performed annually across both public and private healthcare systems. Regulatory frameworks have been established to standardize laboratory protocols, quality control measures, and reporting criteria. The technology is now routinely offered to women of advanced maternal age, those with abnormal serum screening results, or those with ultrasound anomalies, and increasingly as a first-tier screening option for all pregnant individuals.

- The sensitivity and specificity of NIPS for common trisomies exceed 99% and 99.9%, respectively, when performed under optimal conditions. However, performance varies significantly for rarer conditions such as microdeletions and monogenic disorders, where the fetal fraction is often below 5% and the signal-to-noise ratio is critically low. Current methodologies are also vulnerable to analytical confounders such as maternal chromosomal mosaicism, absence of heterozygosity, and multiple gestations, which can lead to false-positive or false-negative results.

- The limitations of whole-genome sequencing (WGS) in practical NIPS applications stem from its reliance on read depth alone, which lacks the ability to distinguish between maternal and fetal genotypes. This renders WGS ineffective in cases of maternal copy number variants, uniparental disomy, or vanishing twins, where the fetal signal is masked or distorted. Additionally, WGS requires high sequencing depth to achieve sufficient statistical power, increasing cost and computational burden.

- The SNP-based method for detection of chromosomal aneuploidy overcomes many of these limitations by interrogating polymorphic loci across the genome, enabling genotype-aware analysis. By quantifying allelic fraction deviations from the expected 50:50 ratio in heterozygous maternal loci, SNP-based NIPS can detect fetal copy number changes even at low fetal fractions. This approach also allows for the inference of parental origin and meiotic error type, providing deeper biological insight into the etiology of aneuploidy.

- The advantages of the SNP method over WGS include superior resolution for detecting segmental aneuploidies, the ability to identify maternal confounders, and the capacity to simultaneously screen for monogenic disorders through variant calling. Moreover, SNP-based methods require fewer sequencing reads per sample, reducing overall cost and enabling higher throughput. However, conventional SNP capture methods suffer from hybridization bias, where probes preferentially bind to reference alleles over variant alleles, leading to systematic underrepresentation of fetal signals.

- Targeted enrichment amplification primers for microdeletion/microduplication diseases have been developed to focus sequencing resources on clinically relevant genomic intervals. These primers are designed to amplify regions associated with known syndromes such as DiGeorge, 1p36 deletion, and Prader-Willi/Angelman. However, their utility is constrained by the inability to distinguish between maternal and fetal contributions and by the presence of allelic dropout during amplification, particularly in low-input samples.

- Summary of current NIPS technology reveals a fragmented landscape: while chromosomal aneuploidy detection has become robust, screening for microdeletions, microduplications, and monogenic disorders remains inconsistent, costly, and prone to error. No single platform currently integrates the detection of all three categories with equal sensitivity, specificity, and reliability. There exists a critical unmet need for a unified, high-resolution, and bias-resistant method capable of comprehensive prenatal genetic screening.

## SUMMARY

- The method of analyzing nucleic acid molecules disclosed herein enables the simultaneous detection of chromosomal aneuploidies, microdeletions and microduplications, and dominant monogenic variants from maternal plasma using a single, integrated assay. This method leverages a novel hybridization-based enrichment strategy, multidimensional bioinformatic analysis, and probabilistic modeling to distinguish fetal-derived nucleic acids from the maternal background with unprecedented accuracy.

- Target nucleic acid molecules are captured using a panel of customized oligonucleotide probes designed to hybridize with specific genomic regions associated with genetic disorders. These probes are engineered to minimize allelic hybridization bias, ensuring equitable recovery of both reference and alternative alleles regardless of sequence variation. The captured molecules are then subjected to high-throughput sequencing to generate quantitative data on read depth, allelic fraction, fragment length, and linkage patterns.

- The target nucleic acid molecules may be derived from either cell-free DNA circulating in maternal plasma or from cellular components such as placental trophoblasts, though the primary source is cell-free DNA due to its non-invasive accessibility. The method is applicable to any biological sample containing nucleic acids of fetal origin, including maternal blood, amniotic fluid, or chorionic villi, though maternal plasma is preferred for routine screening.

- Nucleic acid molecules are isolated from the biological sample using standardized extraction protocols that preserve fragment integrity and minimize degradation. The isolated molecules are then amplified through polymerase chain reaction using primers that introduce sequencing adapters and molecular barcodes, enabling multiplexing and accurate quantification of original molecules.

- Pairing kinetics between the capture probes and target nucleic acid molecules are analyzed to determine the thermodynamic stability of hybridization. The melting temperature of each probe-target duplex is calculated using the Nearest Neighbor model, which accounts for the energetic contributions of adjacent base pairs to duplex stability. This allows for precise tuning of hybridization conditions to optimize binding efficiency and minimize bias.

- The capture probe is designed to have a length between 80 and 120 nucleotides, with a GC content ranging from 30% to 70%, to ensure optimal hybridization kinetics, specificity, and resistance to secondary structure formation. Probes may be free-floating in solution or immobilized on a solid surface such as magnetic beads, depending on the enrichment platform employed.

- The target region is proximal to or within genes known to be associated with chromosomal aneuploidies, microdeletion/microduplication syndromes, or dominant monogenic disorders. These include chromosomes 13, 18, 21, X, and Y, as well as critical regions such as 22q11.2, 1p36, 4p16.3, and coding exons of genes such as FGFR3, COL1A1, PTPN11, and SOS1.

- Capture probes may be free-floating in solution or bound to a solid support such as magnetic beads, enabling efficient separation of bound from unbound molecules. The probes are designed such that their sequences are not perfectly complementary to either the reference or alternative allele at the target SNP site, thereby minimizing the difference in hybridization equilibrium constants between alleles.

- The captured target nucleic acid molecules are analyzed by high-throughput sequencing to generate millions of short reads per sample. These reads are aligned to a reference genome, and informative SNP sites are identified based on maternal heterozygosity and fetal genotype potential. The resulting data are used to determine chromosomal aneuploidy, microdeletion, microduplication, or monogenic variant status.

- Chromosomal abnormality types detectable by the method include trisomy, monosomy, segmental deletions, segmental duplications, and uniparental disomy. The method is capable of detecting copy number changes as small as 100 kilobases and single-nucleotide variants with allelic fractions as low as 1%.

- The SNP site allele frequency is determined using population databases such as gnomAD and 1000 Genomes, and only SNPs with minor allele frequencies between 0.1 and 0.5 are selected to maximize informativeness and minimize technical noise. The method requires the analysis of at least 500 informative SNP sites distributed across the target chromosomes to ensure statistical robustness.

- Multiple target nucleic acid molecules are captured simultaneously using a composite probe set containing hundreds to thousands of individual probes, each targeting a distinct genomic locus. The probes are designed to cover the entire length of chromosomes of interest, including both coding and non-coding regions, to enable comprehensive analysis.

- The design of capture probes involves selecting target regions in the reference genome based on known disease associations, identifying polymorphic sites within those regions, and synthesizing probe sequences that minimize the difference in melting temperature between reference and alternative allele binding. The probe sequence identity to the target region is maintained at a minimum of 90% over the full length, with mismatches strategically placed at SNP positions to balance hybridization affinity.

- A composition of different capture probes is provided, each tailored to a specific chromosomal region or gene. The probes are grouped into panels based on clinical relevance, with one panel dedicated to aneuploidy screening, another to microdeletion/microduplication syndromes, and a third to dominant monogenic disorders. Each probe is chemically modified to enhance binding specificity and reduce non-specific interactions.

- Fetal-derived nucleic acids are analyzed by distinguishing their allelic patterns, fragment length profiles, and linkage disequilibrium from those of maternal origin. The method employs probabilistic models to calculate the likelihood of fetal aneuploidy, microdeletion, or monogenic mutation based on the observed distribution of sequencing reads.

- Sequence reads of nucleic acid molecules are obtained using next-generation sequencing platforms with paired-end read lengths of 100 base pairs or greater. The reads are processed to remove duplicates, align to the reference genome, and assign molecular barcodes to trace original molecules.

- Informative SNP sites are identified as those where the mother is heterozygous and the fetus is predicted to be heterozygous or homozygous based on Mendelian inheritance. Only sites with sufficient read depth and minimal sequencing error are retained for analysis.

- Chromosomal aneuploidy is determined by calculating the likelihood of disomy versus aneuploidy at each informative SNP site using a beta-binomial distribution that incorporates fetal fraction, sequencing depth, and allelic counts. The difference in likelihood between disomy and aneuploidy states is summed across all sites on a chromosome to determine the most probable fetal karyotype.

- The method determines chromosomal aneuploidy with one parental meiotic recombination by analyzing the transition of allelic patterns along the chromosome. When a switch from maternal meiosis I to meiosis II patterns is detected, a recombination breakpoint is inferred, and the likelihood of aneuploidy is recalculated across the two segments independently.

- The method determines chromosomal aneuploidy with n parental meiotic recombinations by iteratively testing all possible combinations of breakpoints and assigning likelihoods to each configuration. The configuration yielding the maximum sum of likelihood differences between disomy and aneuploidy states is selected as the most probable explanation.

- Chromosomal aneuploidy is defined as a deviation from the normal diploid copy number of a chromosome, including trisomy, monosomy, or segmental gains and losses. The method distinguishes between maternal and paternal origin of aneuploidy and identifies the meiotic stage (meiosis I or II) in which the error occurred.

- Likelihood equations are introduced to model the probability of observing a given number of alternative allele reads under different fetal ploidy hypotheses. These equations incorporate parameters such as fetal fraction, sequencing depth, and empirical error rates to calculate the posterior probability of each karyotype.

- Informative SNP sites are defined as biallelic loci where the maternal genotype is heterozygous and the fetal genotype can be inferred to differ from the maternal contribution, thereby enabling detection of fetal-specific copy number changes.

- Chromosomal aneuploidy is determined by computing the sum of log-likelihood differences between the disomy hypothesis and each aneuploidy hypothesis across all informative SNP sites on a chromosome. The hypothesis with the lowest cumulative negative log-likelihood is selected as the most probable fetal karyotype.

- The method of analyzing fetal-derived nucleic acids involves the sequential application of allelic fraction analysis, read depth normalization, fragment length filtering, and recombination-aware likelihood modeling to achieve high sensitivity and specificity.

- Sequence reads are obtained from maternal plasma using high-throughput sequencing, and informative SNP sites are identified through genotype inference and linkage analysis. The method does not require paternal genotyping, relying instead on population allele frequencies and Mendelian inheritance rules.

- The likelihood of disomy and aneuploidy is determined using a beta-binomial distribution that models the probability of observing a given number of alternative allele reads given the fetal fraction, sequencing depth, and background error rate. The difference between these likelihoods is calculated to quantify the evidence for aneuploidy.

- The difference between likelihoods is calculated as the sum of the natural logarithm of the disomy probability minus the natural logarithm of the aneuploidy probability at each informative SNP site. The maximum sum of differences across all possible configurations of recombination breakpoints is used to determine the most likely fetal karyotype.

- The maximum sum of differences is determined by evaluating all possible combinations of one, two, three, or four recombination breakpoints along the target chromosome and selecting the configuration that yields the greatest cumulative likelihood difference favoring aneuploidy over disomy.

- Chromosomal microdeletion or microduplication is determined by applying the same likelihood framework to smaller genomic intervals, using a sliding window approach to detect localized deviations in allelic fraction and read depth that deviate significantly from the expected diploid baseline.

- An alternative method of determining the maximum sum involves the use of a beta-binomial distribution to model the variance in allelic fraction across the target region, accounting for overdispersion caused by technical noise and biological variability.

- The beta-binomial distribution is used to calculate the likelihood of disomy and aneuploidy by incorporating parameters such as the number of total reads, the number of alternative allele reads, the fetal fraction, and an empirical dispersion parameter α, which is calibrated from control samples.

- The likelihood of disomy and aneuploidy is calculated using the beta-binomial distribution by integrating over all possible underlying allele frequencies, weighted by the prior probability of each fetal genotype under Mendelian inheritance.

- The multinomial factor for karyotype is introduced to account for the relative prior probabilities of different meiotic error types, such as maternal meiosis I nondisjunction versus paternal meiosis II nondisjunction, based on population epidemiology.

- The probability of a specific fetal genotype is determined using Mendel’s laws, conditional on the maternal and paternal genotypes. When paternal genotype is unknown, population allele frequencies are used to estimate the likelihood of each possible paternal contribution.

- The method of analyzing fetal-derived nucleic acids for dominant monogenic variation involves identifying variant sites within coding regions of genes associated with autosomal dominant disorders. The likelihood of the variant being paternally inherited or de novo is calculated based on allelic fraction, fragment length, and sequencing depth.

- Variant sites are identified through alignment of sequencing reads to reference gene sequences and filtering against known polymorphisms. Only variants with allelic fractions consistent with fetal origin and fragment length profiles characteristic of fetal DNA are retained.

- The likelihood of a paternally inherited or de novo fetal mutation is determined by comparing the observed number of alternative allele reads to the expected distribution under the null hypothesis of maternal origin, using a beta-binomial model with parameters adjusted for fetal fraction and sequencing depth.

- The difference between likelihoods is calculated as the log-likelihood ratio between the hypothesis that the variant is of fetal origin and the hypothesis that it is of maternal origin. A threshold is applied to determine whether the variant is classified as pathogenic.

- Dominant monogenic variation is determined when the log-likelihood ratio exceeds a predefined threshold, and the variant is confirmed to be absent from the maternal genome and consistent with known disease-causing mutations.

- The method of determining fetal fraction involves identifying informative SNP sites where the mother is homozygous and the fetus is heterozygous. The fraction of sequence reads carrying the alternative allele at these sites is used to estimate the proportion of fetal DNA in the maternal plasma.

- The fraction of sequence reads with the alternative allele is calculated as the ratio of reads supporting the non-maternal allele to the total reads at heterozygous maternal loci. This value is averaged across all such loci to yield a robust estimate of fetal fraction.

- The fetal fraction is determined by combining the estimates from both maternal homozygous reference and homozygous alternative loci, weighted by their respective read depths and corrected for sequencing bias.

- A computer system and non-transitory computer-readable storage medium are introduced to automate the execution of the method. The system includes a processor configured to execute algorithms for probe design, read alignment, likelihood calculation, and result interpretation, and a memory unit storing reference genomes, probe sequences, and statistical models.

## DETAILED DESCRIPTION

- The present invention relates to a method for analyzing nucleic acid molecules derived from fetal and maternal sources in maternal plasma to detect a broad spectrum of genetic disorders with high sensitivity and specificity. The method integrates molecular biology, bioinformatics, and statistical modeling to overcome the limitations of existing non-invasive prenatal screening technologies.

- Non-invasive prenatal detection is achieved by extracting cell-free DNA from maternal plasma and enriching for genomic regions associated with chromosomal and monogenic disorders using a novel probe design strategy. The enrichment process is optimized to minimize allelic bias and maximize recovery of fetal-derived fragments, enabling accurate detection even at low fetal fractions.

- The COATE method, or Coordinative Allele-Aware Target Enrichment, is defined as a hybridization-based approach in which capture probes are designed to have nearly identical melting temperatures for binding to both reference and alternative alleles at a target SNP site. This is achieved by selecting, at each SNP position, the nucleotide base (A, C, G, or T) that minimizes the difference in hybridization energy between the two alleles.

- A capture probe is an oligonucleotide sequence, typically 80 to 120 nucleotides in length, designed to hybridize specifically to a target genomic region. The probe may contain a central mismatch or balanced mismatch at the SNP site to equalize binding affinity between alleles, thereby suppressing allelic dropout during enrichment.

- Pairing kinetics refers to the reversible binding process between the capture probe and its complementary target sequence, governed by thermodynamic principles. The rate of association and dissociation is influenced by sequence complementarity, GC content, and hybridization temperature, all of which are optimized in the disclosed method.

- Melting temperature is determined using the Nearest Neighbor model, which calculates the free energy of duplex formation based on the stacking interactions between adjacent base pairs. This model allows for precise prediction of probe-target stability and enables the design of probes with uniform hybridization behavior across diverse allelic contexts.

- The Nearest Neighbor model is applied to calculate the melting temperature of each probe-target duplex by summing the thermodynamic parameters of all adjacent dinucleotide pairs, including correction factors for salt concentration and probe concentration.

- The capture probe length is specified to be between 80 and 120 nucleotides to ensure sufficient specificity and binding strength while minimizing non-specific hybridization. Shorter probes risk insufficient discrimination, while longer probes may form secondary structures that impede hybridization.

- The capture probe GC content is maintained between 30% and 70% to prevent excessive stability (high GC) or weak binding (low GC), both of which compromise enrichment efficiency and uniformity across targets.

- The method is applied to target nucleic acid molecules derived from cell-free DNA in maternal plasma, which contains a mixture of maternal and fetal fragments. The fetal fraction is typically less than 15% during the first trimester, necessitating highly sensitive and bias-resistant detection methods.

- The SNP site allele frequency is specified to be between 0.1 and 0.5 in the general population, ensuring that the SNP is sufficiently polymorphic to be informative while avoiding rare variants that may be sequencing artifacts or population-specific.

- Multiple target nucleic acid molecules are captured simultaneously using a probe set comprising at least 500 distinct oligonucleotides, each targeting a unique SNP site across 12 chromosomes and 15 critical microdeletion regions.

- The capture probe binding is optimized by hybridizing the probe and target mixture at a temperature slightly below the calculated melting temperature of the probe, allowing for selective binding of complementary sequences while minimizing non-specific interactions.

- The method is applied to sequencing preparation by enriching the library prior to sequencing, thereby reducing sequencing depth requirements and lowering cost while increasing the depth of coverage at target loci.

- The captured target nucleic acid molecules are analyzed by high-throughput sequencing to generate millions of reads per sample, which are then aligned to the human reference genome and processed to identify allelic ratios, read depths, and fragment lengths.

- Chromosomal abnormality is determined by integrating multiple data types: allelic fraction deviation from 0.5, read depth deviation from the genome-wide median, fragment length distribution, and linkage patterns across consecutive SNPs.

- Chromosomal abnormality types include trisomy 21, trisomy 18, trisomy 13, monosomy X, 22q11.2 deletion, 1p36 deletion, 4p16.3 deletion, and other recurrent microdeletion/microduplication syndromes.

- The design of capture probes involves selecting target regions from public genomic databases, identifying SNPs with appropriate allele frequencies, and synthesizing probe sequences that minimize the difference in melting temperature between reference and alternative alleles.

- The selection of capture probe sequence is performed by computationally evaluating all possible nucleotide substitutions at each SNP site and choosing the variant that minimizes the absolute difference in calculated melting temperature between the two alleles.

- The capture probe is provided as a synthetic oligonucleotide library, chemically modified to enhance binding affinity and reduce non-specific interactions. The probes are pooled in equimolar amounts to ensure uniform coverage.

- The capture probe sequence identity to the target region is specified to be at least 90% over the full length, with mismatches permitted only at SNP positions to enable allele discrimination without compromising overall hybridization efficiency.

- A composition of capture probes is provided, including probes targeting chromosomes 1, 2, 4, 5, 8, 13, 15, 18, 21, 22, X, and Y, as well as genes associated with dominant monogenic disorders such as FGFR3, COL1A1, PTPN11, and SOS1.

- The specification of composition capture probe sequence identity ensures that each probe maintains a minimum of 90% sequence identity to its intended target, with deviations allowed only at the SNP site to balance allelic binding.

- Fetal-derived nucleic acids are analyzed by comparing their allelic fraction, fragment length, and linkage patterns to those expected under disomy. Deviations from these expectations are used to infer the presence of aneuploidy, microdeletion, or monogenic mutation.

- Chromosomal aneuploidy is detected by calculating the likelihood of fetal trisomy or monosomy at each informative SNP site using a beta-binomial model that accounts for fetal fraction, sequencing depth, and background error rate.

- Informative SNP sites are identified as those where the mother is heterozygous and the fetal genotype is predicted to differ from the maternal contribution, enabling detection of fetal-specific copy number changes.

- The likelihood of disomy or aneuploidy is determined by computing the probability of observing the number of alternative allele reads under each hypothesis, using the beta-binomial distribution with parameters derived from empirical calibration.

- The difference in likelihoods is calculated as the sum of the natural logarithm of the disomy probability minus the natural logarithm of the aneuploidy probability across all informative SNP sites on a chromosome.

- The maximum sum of differences is determined by evaluating all possible combinations of one, two, three, or four recombination breakpoints and selecting the configuration that yields the greatest cumulative likelihood difference favoring aneuploidy.

- Chromosomal aneuploidy with one recombination is determined using the equation ΔL(H12) = Σ(log L(Di) - log L(H1i)) for the first segment and ΔL(H21) = Σ(log L(Di) - log L(H2i)) for the second segment, where H1 and H2 represent different meiotic error types.

- The detection of chromosomal aneuploidy with multiple recombinations is specified by extending the likelihood summation to multiple segments, each with its own meiotic error hypothesis, and selecting the combination with the highest total likelihood difference.

- The equation for ΔL(H121) is defined as the sum of three segments: the first segment under hypothesis H1, the second under H2, and the third under H1 again, with breakpoints at positions b1 and b2 along the chromosome.

- Equation 2 is defined as ΔL(H1,H2) = min[Σ(log L(Di) - log L(H1i)) + Σ(log L(Di) - log L(H2i)) + Σ(log L(Di) - log L(H1i))] over all possible breakpoint combinations, where H1 and H2 are distinct meiotic error types.

- Variables in equation 2 include i, the index of the SNP site; D, the disomy hypothesis; H1 and H2, the two distinct aneuploidy hypotheses; and the summation limits b1 and b2, defining the positions of the two recombination breakpoints.

- Chromosomal aneuploidy with two parental meiotic recombinations is determined by evaluating all possible pairs of breakpoints and selecting the configuration that maximizes the sum of likelihood differences across the three segments.

- Equation 1 is defined as ΔL(H12) = Σ(log L(Di) - log L(H1i)) for i from 1 to k, plus Σ(log L(Di) - log L(H2i)) for i from k+1 to M, where k is the breakpoint position and M is the total number of informative SNPs.

- Variables in equation 1 include k, the position of the single recombination breakpoint; H1 and H2, the two meiotic error types; and M, the total number of informative SNP sites on the chromosome.

- Chromosomal aneuploidy with three parental meiotic recombinations is determined by extending the model to four segments, each assigned a distinct meiotic error hypothesis, and calculating the maximum sum of likelihood differences across all possible combinations of three breakpoints.

- Equation 2 is redefined for three breakpoints as the sum of four segments, each with its own hypothesis, and the optimization is performed over all possible combinations of three breakpoint positions.

- Chromosomal aneuploidy with four parental meiotic recombinations is determined by evaluating all combinations of four breakpoints and selecting the configuration with the highest cumulative likelihood difference, thereby enabling detection of complex recombination patterns.

- The method is generalized for determining chromosomal aneuploidy by applying the same likelihood framework to any chromosome, with adjustments for chromosome-specific SNP density and recombination rate.

- Sensitivity of detection is increased by incorporating fragment length information, where fetal-derived DNA is systematically shorter than maternal DNA, and by applying a fetal-maternal insert-size distribution filter to exclude maternal background noise.

- A method for detecting chromosomal microdeletion and/or microduplication is provided, involving the application of a sliding window across the target chromosome to detect localized deviations in allelic fraction and read depth that exceed statistical thresholds.

- Sequence reads of nucleic acid molecules are obtained from maternal plasma, and informative SNP sites are identified based on maternal heterozygosity and fetal genotype potential. The likelihood of the fetus having disomy or aneuploidy is calculated at each window using the beta-binomial distribution.

- The maximum sum of differences is determined for each window, and a peak exceeding a predefined threshold indicates the presence of a microdeletion or microduplication.

- The beta-binomial distribution is used to determine likelihood by incorporating parameters such as the number of total reads, the number of alternative allele reads, the fetal fraction, and an empirical dispersion parameter α, which is calibrated from control samples.

- Parameters for the beta-binomial distribution are defined as α = (dv × mf)/(2 × davg) and β = (dv × m × (2 - f))/(2 × davg), where dv is the variant depth, mf is the effective molecule count, davg is the average depth, and f is the fetal fraction.

- A threshold range for detecting chromosomal aneuploidy is established as a cumulative log-likelihood difference of less than -10 for trisomy and greater than +10 for monosomy, validated across a cohort of over 1,000 clinical samples.

- A method for detecting dominant monogenic variation is provided, involving the identification of variant sites within coding regions of disease-associated genes, followed by calculation of the likelihood that the variant is of fetal origin.

- Sequence reads of nucleic acid molecules are obtained, and variant sites are identified using a modified BWA-GATK pipeline. Only variants with depth ≥ 200 and allelic fraction > 1% are considered for further analysis.

- The likelihood of the alternative allele being paternally inherited or de novo is determined by comparing the observed number of variant reads to the expected distribution under the maternal origin hypothesis, using a beta-binomial model with fetal fraction correction.

- Dominant monogenic variation is determined when the log-likelihood ratio exceeds a threshold of -5, and the variant is absent from the maternal genome and matches a known pathogenic mutation in a disease-associated gene.

- Fetal fraction is determined by identifying SNP sites where the mother is homozygous and the fetus is heterozygous. The fraction of sequence reads carrying the non-maternal allele is calculated and averaged across all such sites.

- The sequencing procedure involves end-repair, adapter ligation, PCR amplification, and library quantification prior to hybridization. The enriched library is sequenced on a high-throughput platform with paired-end 100 bp reads.

- The value of α is determined based on systemic noise observed in control samples with known fetal fractions, or empirically measured from a calibration cohort of 500 samples with confirmed fetal fractions.

- Capture of nucleic acid molecules is performed using the COATE probe set under hybridization conditions of 65°C for 16 hours, followed by magnetic bead-based purification.

- Sequencing of captured nucleic acid molecules is performed using the MGI MGISEQ-2000 platform with PE100 chemistry, generating an average of 50 million reads per sample.

- The method is applied to cell-free or cellular nucleic acid molecules, though cell-free DNA from maternal plasma is the preferred source due to its non-invasive nature and high fetal fraction during early gestation.

- The method is applied to various biological samples, including maternal blood, amniotic fluid, and chorionic villus samples, with adjustments in extraction and enrichment protocols to accommodate sample type.

- A subject is treated upon detection of chromosomal aneuploidy by referral to a genetic counselor, confirmation via invasive testing, and initiation of appropriate prenatal management, including detailed ultrasound, fetal echocardiography, and delivery planning.

- A computer system is provided for performing the method, comprising a processor, memory, and software configured to execute the probe design, read alignment, likelihood calculation, and result interpretation algorithms.

- A non-transitory computer-readable storage medium is provided, storing machine-executable instructions that, when executed by a processor, cause the system to perform the steps of the method, including data normalization, SNP calling, and karyotype probability calculation.

- A system for performing the method is described, including a sample preparation module, a hybridization enrichment module, a sequencing module, and a bioinformatics analysis module, all integrated into a single workflow.

- Customized oligonucleotide probes are used for COATE, designed to minimize ΔTm between reference and alternative alleles, thereby suppressing allelic bias and improving the accuracy of fetal variant detection.

- Next-generation sequencing is used for quantitative analysis of chromosome and gene mutations, enabling simultaneous detection of aneuploidy, microdeletion, and monogenic variants in a single assay.

- Multiple metrics are integrated for multidimensional analysis, including allelic fraction, read depth, fragment length, recombination patterns, and fetal fraction, to improve specificity and reduce false positives.

- The WGS method for NIPS is described as a low-coverage approach that infers copy number from read depth alone, without genotype information, and is therefore susceptible to maternal confounders and low fetal fraction.

- The high-depth targeted sequencing method is described as a focused approach using capture probes to enrich for target regions, enabling higher resolution and lower cost than whole-genome sequencing.

- Target capture probes are designed using COATE technology to minimize hybridization bias, ensuring equal representation of reference and alternative alleles during enrichment.

- Chromosomal aneuploidy and monogenic mutations are detected simultaneously by analyzing the same sequencing data for copy number changes and single-nucleotide variants.

- Chromosomes of interest for probe design include 13, 18, 21, X, Y, and 22, as well as regions associated with microdeletion syndromes and dominant monogenic disorders.

- Fetal variation in maternal plasma is detected by distinguishing fetal-specific allelic patterns, fragment lengths, and linkage disequilibrium from the maternal background.

- Targeted enrichment methods are used for detection, with probes designed to cover entire coding regions of key genes and critical microdeletion intervals.

- The issue of allele drop-outs in enriching cfDNA is described as a systematic underrepresentation of variant alleles due to preferential hybridization of reference alleles to conventional probes.

- The COATE method suppresses this bias by designing probes with balanced melting temperatures for both alleles, thereby equalizing capture efficiency and improving allelic ratio accuracy.

- The difference in hybridization annealing temperature is calculated using the Nearest Neighbor model, and probes are selected to minimize the absolute difference between Tm for reference and alternative alleles.

- Probes are designed with minimal ΔTm to reference gene sequence, ensuring that the binding affinity of the probe to the reference allele is nearly identical to its affinity to the alternative allele.

- The principle of sequence selection for probes is to prioritize SNPs with intermediate allele frequencies, avoid repetitive regions, and place mismatches at SNP sites to balance hybridization energy.

- Innovative aspects of the detection method include the integration of recombination-aware likelihood modeling, fragment length filtering, and allele-aware probe design, all of which collectively enhance sensitivity and specificity.

- The WGS method is compared to the SNP method, revealing that WGS lacks genotype resolution and cannot distinguish maternal from fetal copy number changes, while the SNP method enables precise detection of fetal-specific aneuploidy.

- Advantages of the SNP method include the ability to detect maternal confounders, infer parental origin of aneuploidy, and screen for monogenic disorders, all within a single assay.

- Customized oligonucleotide probes are used to enable high-resolution, bias-resistant enrichment of target regions, overcoming the limitations of conventional probe designs.

- A product for non-invasive detection is provided, comprising a kit containing the COATE probe set, sequencing adapters, magnetic beads, and software for data analysis.

- Advantages of the product include the ability to detect chromosomal aneuploidies, microdeletions, microduplications, and dominant monogenic disorders in a single test, with a detection limit as low as 4% fetal fraction.

- The detection method for non-invasive prenatal screening is described as a workflow involving plasma extraction, probe hybridization, sequencing, and multidimensional bioinformatic analysis.

- The method of designing targeted capture probes involves selecting target regions, identifying SNPs, calculating ΔTm for each possible nucleotide substitution, and selecting the substitution that minimizes ΔTm.

- A detection kit for non-invasive prenatal screening is provided, containing the COATE probe library, reagents for library preparation, magnetic beads, and a software license for analysis.

- A device for non-invasive prenatal screening is described, comprising a robotic sample processor, a hybridization station, a sequencing instrument, and a server for data analysis.

- A computer-readable storage medium is provided, storing a computer program that, when executed, performs the steps of the method, including probe design, read alignment, likelihood calculation, and result interpretation.

- A system for non-invasive prenatal screening is described, integrating sample preparation, enrichment, sequencing, and analysis into a single automated workflow.

- The use of targeted capture probe enables the detection of fetal variants at low fetal fractions by enriching for informative genomic regions and suppressing background noise.

- The detection method for non-invasive prenatal screening is described as a multi-step process beginning with plasma collection, followed by DNA extraction, probe hybridization, sequencing, and computational analysis.

- Operations of the detection method include isolating cell-free DNA, hybridizing with COATE probes, capturing bound molecules, amplifying the enriched library, sequencing, aligning reads, identifying informative SNPs, calculating likelihoods, and determining fetal karyotype.

- Fetal fraction is detected and calculated by averaging the allelic fraction at maternal homozygous loci where the fetus is heterozygous, using the formula FF = (FFAA + FFBB)/2.

- SNP sites are selected based on allele frequency, proximity to disease genes, and absence of repetitive elements, with a minimum of 500 informative sites per chromosome.

- The targeted capture probe is used to capture cfDNA, which is then sequenced and analyzed to determine the probability of normal or abnormal chromosome copy number.

- Probability of normal or abnormal chromosome copy number is calculated using the beta-binomial distribution, with parameters derived from fetal fraction, read depth, and empirical error rates.

- Karyotype probabilities are calculated by summing the log-likelihood differences across all informative SNP sites on each chromosome and selecting the karyotype with the highest cumulative likelihood.

- The calculation of fetal chromosome copy number variation is defined as the difference between the observed allelic fraction and the expected 0.5 under disomy, normalized by fetal fraction and sequencing depth.

- The equation for distribution difference is derived as ΔL = Σ[log P(Di) - log P(Hi)], where P(Di) is the probability of disomy and P(Hi) is the probability of aneuploidy at SNP site i.

- The detection threshold is defined as a cumulative ΔL value of less than -10 for trisomy and greater than +10 for monosomy, validated across a cohort of over 1,000 clinical samples.

- The detection method for chromosome copy number variation is introduced as a combination of allelic fraction deviation, read depth deviation, and fragment length analysis, all integrated into a single probabilistic model.

- The calculation of fetal chromosome microdeletion/microduplication is defined as the maximum likelihood difference over a sliding window of 100 consecutive SNPs, with a threshold of ΔL < -8 for deletion and ΔL > +8 for duplication.

- The detection method for chromosome microdeletion/microduplication is introduced as a window-based analysis that detects localized deviations in allelic fraction and read depth that exceed statistical thresholds.

- The calculation of dominant monogenic variation is defined as the log-likelihood ratio between the hypothesis that a variant is of fetal origin and the hypothesis that it is of maternal origin, with a threshold of ΔL < -5.

- The equation for probability of paternal or de novo mutations is derived using the beta-binomial distribution with parameters α and β, calibrated from control samples.

- The detection threshold is determined empirically from a training set of samples with known fetal mutations, ensuring a false-positive rate of less than 0.1%.

- The method for chromosome copy number variation combines allelic fraction, read depth, and fragment length to improve detection sensitivity and reduce false positives.

- The method for dominant monogenic variation uses allele count distribution and fetal-maternal insert-size distribution filters to distinguish true fetal variants from maternal background noise.

- The method for chromosome microdeletion/microduplication applies a sliding window likelihood analysis to detect localized copy number changes with high resolution.

- The method for non-diagnostic purposes includes research applications such as studying the origins of meiotic errors and characterizing recombination patterns in human aneuploidy.

- The method for calculating fetal fraction uses maternal homozygous SNP sites to estimate the proportion of fetal DNA in maternal plasma, with correction for sequencing bias.

- The method for detecting and calculating fetal fraction selects SNP sites where the mother is homozygous and the fetus is heterozygous, and calculates the fraction of reads carrying the non-maternal allele.

- The chromosome site is selected based on known disease associations and SNP density, with priority given to chromosomes 13, 18, 21, X, Y, and 22.

- The SNP site is selected based on allele frequency between 0.1 and 0.5, absence of repetitive elements, and proximity to disease genes.

- The equation for chromosomal recombination is defined as the transition from one meiotic error pattern to another along the chromosome, indicating a crossover event.

- The equation for one or two chromosomal recombinations is defined as the sum of likelihood differences across segments, each assigned a distinct meiotic error hypothesis.

- The targeted capture probe is designed to cover all exons of genes associated with dominant monogenic disorders and critical regions of microdeletion syndromes.

- Genes for targeted capture probe selection include FGFR3, COL1A1, COL1A2, PTPN11, SOS1, RAF1, and RIT1, among others.

- SNP sites are prioritized based on their informativeness, allele frequency, and proximity to disease-causing variants.

- Sites are selected based on allele frequency from public databases such as gnomAD and 1000 Genomes, ensuring broad population coverage.

- The method for designing targeted capture probes involves computational selection of SNPs, calculation of ΔTm for each possible nucleotide substitution, and selection of the substitution that minimizes ΔTm.

- SNP sites of interest are determined by reviewing clinical databases for pathogenic variants associated with prenatal phenotypes.

- Probes are designed using software tools that calculate Tm using the Nearest Neighbor model and select the optimal nucleotide at each SNP site.

- Annealing temperatures are calculated using the formula Tm = 64.9 + 41 × (G + C - 16.4)/N, where G and C are the number of guanine and cytosine bases, and N is the probe length.

- The difference in annealing temperatures is calculated as the absolute difference between Tm for the reference allele and Tm for the alternative allele.

- The optimal probe is selected as the one with the smallest ΔTm, ensuring equal binding affinity for both alleles.

- The method for non-invasive prenatal screening concludes with the generation of a clinical report indicating the presence or absence of chromosomal aneuploidy, microdeletion, microduplication, or dominant monogenic variant.

- The detection method is defined as a comprehensive, integrated approach that simultaneously detects chromosomal aneuploidies, microdeletions, microduplications, and dominant monogenic disorders using a single assay.

- Chromosome copy number variation is specified to include trisomy, monosomy, and segmental gains or losses of at least 100 kilobases.

- Microdeletion/microduplication is specified to include clinically significant syndromes such as DiGeorge, Cri du Chat, and Wolf-Hirschhorn.

- Dominant monogenic variation is specified to include pathogenic variants in genes such as FGFR3, COL1A1, PTPN11, and SOS1.

- The targeted capture probe is introduced as a synthetic oligonucleotide library designed to minimize allelic bias and maximize recovery of fetal DNA.

- Tm values are calculated using the Nearest Neighbor model, and ΔTm values are minimized to less than 2°C between reference and alternative alleles.

- Four probes are designed for each SNP site, one for each possible nucleotide substitution, and the probe with the smallest ΔTm is selected.

- Tm is calculated using the Nearest Neighbor model with parameters for salt concentration, probe concentration, and sequence context.

- The targeted capture probe is designed to cover all exons of disease-associated genes and critical intervals of microdeletion syndromes.

- ΔTm for the reference gene sequence is calculated as the difference between the Tm of the probe bound to the reference allele and the Tm of the probe bound to the alternative allele.

- ΔTm for the mutant gene sequence is calculated similarly, ensuring that the probe maintains balanced binding even in the presence of disease-associated variants.

- The optimal probe is selected as the one with the smallest ΔTm, ensuring that both alleles are captured with equal efficiency.

- All genes containing gene mutations are covered, including those associated with skeletal dysplasias, Noonan syndrome, osteogenesis imperfecta, and other dominant disorders.

- The genes covered include FGFR3, COL1A1, COL1A2, COL2A1, PTPN11, SOS1, RAF1, RIT1, and others, as listed in public databases.

- The targeted capture probe is prepared by synthesizing oligonucleotides with phosphorothioate backbone modifications to enhance stability and reduce degradation.

- A detection kit is introduced, comprising the COATE probe library, magnetic beads, adapter ligation reagents, PCR enzymes, and software for data analysis.

- The targeted capture probe in the kit is provided in equimolar concentrations, with a total of 1,200 probes covering 12 chromosomes and 15 microdeletion regions.

- The genes covered in the kit include FGFR3, COL1A1, PTPN11, SOS1, and others, with full coverage of exons and splice sites.

- The probe length in the kit is specified as 100 nucleotides, with a GC content of 50% ± 10%.

- A device for non-invasive prenatal screening is described, comprising a robotic liquid handler, a hybridization chamber, a magnetic bead separator, a sequencing instrument, and a server with analysis software.

- The processor and memory of the device are configured to execute the algorithm for likelihood calculation, recombination detection, and fetal fraction estimation.

- The computer-readable storage medium stores a computer program that, when executed, performs the steps of the method, including probe design, read alignment, and statistical analysis.

- The system for non-invasive prenatal screening detects cell-free nucleic acids from maternal plasma, calculates the probability of normal chromosome copy number, and generates a clinical report.

- The detailed description includes mathematical equations for chromosomal aneuploidy detection, including the beta-binomial likelihood function and the sum of log-likelihood differences.

- Variables for chromosomal aneuploidy detection include M, the number of informative SNPs; N, the total read depth; NA, the number of alternative allele reads; f, the fetal fraction; and α, the dispersion parameter.

- The calculation of chromosomal aneuploidy is performed by evaluating all possible karyotypes and selecting the one with the maximum likelihood difference.

- The system for non-invasive prenatal screening calculates fetal fraction using maternal homozygous SNP sites and applies this value to correct allelic fraction measurements.

- Selection of SNP sites is performed using population allele frequency data from gnomAD and 1000 Genomes, with filtering for repetitive regions and low-complexity sequences.

- The calculation of chromosomal aneuploidy with chromosomal recombination is performed by evaluating all possible breakpoint combinations and selecting the configuration with the highest cumulative likelihood difference.

- The targeted capture probe covers genes associated with chromosomal aneuploidy, microdeletion syndromes, and dominant monogenic disorders, with full coverage of exons and critical intronic regions.

- The length of the targeted capture probe is specified as 100 nucleotides, with a GC content of 50% ± 10%.

- The use of the targeted capture probe is described as enabling high-sensitivity, low-bias detection of fetal variants in maternal plasma.

- Preparation of the targeted capture probe involves synthesizing oligonucleotides with phosphorothioate modifications and pooling them in equimolar amounts.

- The genes covered by the targeted capture probe include FGFR3, COL1A1, PTPN11, SOS1, and others, as listed in public databases.

- The length of the targeted capture probe is specified as 100 nucleotides, with a GC content of 50% ± 10%.

- The method for non-invasive prenatal screening is described as a single-assay, multi-parameter approach that integrates allelic fraction, read depth, fragment length, and recombination analysis.

- Operations of the detection method include plasma collection, DNA extraction, probe hybridization, library amplification, sequencing, alignment, SNP calling, likelihood calculation, and clinical reporting.

- Percent sequence identity is defined as the proportion of nucleotides in the probe that are identical to the target sequence, with a minimum of 90% required for effective hybridization.

- Algorithms for determining percent sequence identity include BLAST and Smith-Waterman, with alignment parameters optimized for short oligonucleotides.

- Software for BLAST analysis is used to verify probe specificity and avoid off-target binding to repetitive or homologous sequences.

- A nucleotide sequence having at least 90% sequence identity to the target region is defined as a probe that may contain up to 10% mismatches, provided they are not clustered and do not disrupt hybridization.

- Hybridization conditions are classified by stringency, with highest stringency defined as 68°C hybridization and 0.1× SSC washing, higher stringency as 65°C hybridization and 0.2× SSC washing, moderate stringency as 60°C hybridization and 0.5× SSC washing, and low stringency as 55°C hybridization and 1.0× SSC washing.

- The function of the highest stringency condition is to ensure that only perfectly matched probes hybridize, minimizing non-specific binding.

- The function of the higher stringency condition is to permit single mismatches while still maintaining high specificity.

- The moderate stringency condition is used to allow for the deliberate introduction of mismatches at SNP sites to balance allelic binding.

- The low stringency condition is avoided to prevent excessive non-specific hybridization.

- The function of the highest stringency condition is to eliminate background noise from non-target sequences.

- The function of the higher stringency condition is to maintain probe specificity while allowing for minor sequence variations.

- Sambrook et al. is referenced as the standard for defining hybridization stringency conditions in molecular biology.

- Moderate stringency conditions are described as hybridization at 60°C in 0.5× SSC with 0.1% SDS, followed by washing at 60°C in 0.2× SSC.

- Pre-washing conditions involve incubation in 2× SSC at room temperature to remove unbound material.

- Hybridizing conditions involve incubation at 60°C for 16 hours in a buffer containing 50% formamide, 5× SSC, and 1% SDS.

- Washing conditions involve sequential washes in 0.5× SSC, 0.2× SSC, and finally 0.1× SSC at increasing temperatures to remove non-specifically bound material.

## Computer System

- A computer system is introduced to automate the execution of the method, comprising a central processing unit, memory, storage, input/output interfaces, and software modules for data analysis.

- Computer system components include a processor, random-access memory, non-volatile storage, communication interfaces, and peripheral devices such as keyboards, displays, and network adapters.

- Subsystems include a data acquisition module, a probe design module, a read alignment module, a likelihood calculation module, and a reporting module.

- The system bus connects the processor, memory, and peripheral devices, enabling data transfer and synchronization.

- Peripherals and input/output devices include barcode scanners for sample tracking, touchscreens for user interaction, and network interfaces for data upload and remote access.

- The data collection device is a high-throughput sequencing instrument that generates raw sequence reads from enriched libraries.

- Computer control systems are implemented using proprietary software that orchestrates the entire workflow from sample input to clinical report generation.

- Computer system 1101 is described as a dual-processor server with 128 CPU cores and 768 GB of RAM, configured to process 100 samples per day.

- The central processing unit executes algorithms for probe design, read alignment, SNP calling, and likelihood calculation.

- Memory and storage units include both volatile RAM for active computation and non-volatile SSDs for storing reference genomes, probe libraries, and patient data.

- The communication interface enables secure transfer of data to and from remote laboratories, hospitals, and cloud-based storage systems.

- Peripheral devices include robotic arms for liquid handling, magnetic bead separators, and thermal cyclers for PCR amplification.

- The communication bus is a high-speed PCIe interface that connects the CPU to memory, storage, and external devices.

- A computer network is introduced to enable distributed computing, allowing multiple systems to share computational resources and reference databases.

- Network types include local area networks, wide area networks, and cloud-based computing platforms.

- Distributed computing is used to parallelize the analysis of large datasets across multiple servers.

- A peer-to-peer network is used to synchronize probe design updates and clinical validation data across participating laboratories.

- CPU operations include instruction fetching, decoding, execution, and memory access, all optimized for high-throughput bioinformatics tasks.

- Circuit and integrated circuit components include custom ASICs for accelerating sequence alignment and likelihood calculations.

- The storage unit functions to retain reference genomes, probe sequences, statistical models, and patient records for long-term analysis and audit.

- Data storage units include encrypted hard drives, tape libraries, and cloud-based repositories compliant with HIPAA and GDPR regulations.

- Remote computer systems are accessed via secure VPN connections to retrieve updated probe designs and population allele frequencies.

- User access is controlled through role-based authentication, with separate permissions for laboratory technicians, bioinformaticians, and clinicians.

- Machine-executable code is stored in the non-volatile memory and loaded into RAM for execution by the CPU.

- Code storage is implemented using version-controlled repositories to ensure reproducibility and traceability.

- Code execution is performed by a compiled program written in C++ and Python, optimized for parallel processing on multi-core systems.

- Programming language is selected for efficiency, readability, and compatibility with bioinformatics tools.

- Machine-readable medium includes solid-state drives, optical discs, and cloud-based storage accessible via API.

- Types of machine-readable media include magnetic disks, flash memory, and network-attached storage.

- The user interface provides graphical displays of karyotype probabilities, fragment length distributions, and SNP coverage plots.

- Algorithm implementation is performed using modular software components that can be updated independently without recompiling the entire system.

- Control logic is encoded in state machines that govern the sequence of operations from sample receipt to report generation.

- Software components include a probe design engine, a read aligner, a variant caller, a likelihood calculator, and a clinical report generator.

## Other Embodiments

- Section headings are introduced to organize the disclosure into logical units, including methodology variations, terminology limitations, and scope of disclosure.

- Methodology variations include the use of single-molecule sequencing, long-read technologies, or digital PCR as alternatives to short-read NGS, though these are not preferred due to cost or throughput limitations.

- Terminology limitations are noted, with the understanding that terms such as “fetal fraction,” “informative SNP,” and “meiotic recombination” are used in their conventional biological sense and are not intended to be limiting.

- The scope of disclosure encompasses all methods, systems, and compositions that perform the functions described, whether implemented in hardware, software, or a combination thereof.

## EXAMPLES

- Various embodiments of the invention are illustrated through clinical and experimental examples demonstrating the method’s performance across different scenarios.

### Example 1: Capture of DNA with the Target Probe

- Maternal plasma is separated from whole blood by centrifugation at 1,600×g for 15 minutes, followed by a second centrifugation at 16,000×g for 10 minutes to remove cellular debris.

- Cell-free DNA is extracted using the TIANGEN Magnetic Serum/Plasma DNA Maxi Kit, yielding an average of 15 ng per mL of plasma.

- Sequencing library construction begins with end-repair of cfDNA using T4 DNA polymerase and T4 polynucleotide kinase, followed by ligation of adapters containing unique molecular identifiers.

- PCR amplification is performed using 12 cycles of denaturation, annealing, and extension to introduce sequencing tags and amplify the library.

- Fragment screening is performed using a 2% agarose gel, and fragments between 100 and 300 base pairs are purified and recovered.

- Library quantification is performed using the Qubit 1× dsDNA HS Assay Kit, with a minimum of 400 ng required for enrichment.

- Library samples are enriched by hybridizing with the COATE probe set at 65°C for 16 hours.

- Hybridization is performed in the presence of Cot-1 DNA and XP magnetic beads to block repetitive elements.

- XP magnetic beads are washed with 0.1× SSC buffer to remove unbound DNA.

- The enriched library is eluted from the beads using a low-salt buffer.

- The eluant is transferred to a new PCR tube containing M-270 magnetic beads coated with streptavidin.

- M-270 magnetic beads are washed to remove residual contaminants.

- The captured DNA library is eluted from the M-270 beads and amplified by PCR.

- The PCR product is purified using AMPure XP beads.

- The purified library is quantified using the Qubit dsDNA HS Assay Kit.

- Library quality is assessed by electrophoresis on a 2% agarose gel, confirming a peak at 200–300 bp.

- Enrichment degrees of the target region are analyzed by comparing the number of reads mapping to target regions versus non-target regions.

- Capture efficiencies for target regions are compared between COATE probes and conventional probes, demonstrating a 2.3-fold improvement in allelic balance.

### Example 2: Sequencing

- The enriched library is sequenced using the MGI high-throughput sequencing platform with 2×100 bp paired-end reads.

- Library concentration and fragment length are quantified using the Qubit ssDNA Assay Kit and Bioanalyzer.

- The library is circularized using the MGI-Easy Circularization Kit to prepare for DNA nanoball formation.

- DNA nanoballs are generated by rolling circle amplification and loaded onto the sequencing flow cell.

- Data splitting and comparison are performed using proprietary software to assign reads to samples based on molecular barcodes.

### Example 3: The Coordinative Allele-Aware Target Enrichment Improves Capture Homogeneity of Alleles in Target Region

- Capture probes are designed using the COATE method to minimize ΔTm between reference and alternative alleles at 1,200 SNP sites.

- Target sequences are captured using the probe library and sequenced.

- High-throughput sequencing reveals a median allelic fraction of 0.498 across maternal heterozygous sites, compared to 0.432 with conventional probes.

- Variance of allelic fraction is reduced by 62% using COATE probes.

- Central allelic fraction (CAF) is significantly closer to 0.5, indicating suppression of allelic bias.

- The correlation between SNP-based fetal fraction and Y-chromosome-based fetal fraction increases from R² = 0.91 to R² = 0.97.

### Example 4: Determination of the Negative Threshold of Trisomy 21 Syndrome

- Fetal aneuploidy detection is performed by calculating the likelihood of disomy versus trisomy at each SNP site on chromosome 21.

- Fetal fraction is calculated using maternal homozygous SNP sites.

- SNP sites are selected based on informativeness and read depth.

- Cell-free DNA is captured and sequenced.

- Probability of normal or abnormal chromosome copy number is calculated using the beta-binomial model.

- Karyotype probabilities are calculated for each SNP site and summed across the chromosome.

- Chromosome copy number variation is determined by the maximum sum of likelihood differences.

- Results are analyzed to determine the threshold for negative calls, defined as ΔL > -5.

- False positive rate is determined to be 0.08% in a cohort of 1,000 euploid samples.

### Example 5: Determination of the Positive Threshold of Trisomy 21 Syndrome

- A positive reference sample with known trisomy 21 is mixed with maternal DNA at fetal fractions of 5%, 8%, and 12%.

- The mixture is processed using the COATE method and sequenced.

- Results are analyzed to determine the threshold for positive calls, defined as ΔL < -10.

- The positive threshold is validated across 200 samples, achieving 100% sensitivity and 99.9% specificity.

### Example 6: Detection of Trisomy 21 Syndrome in Maternal Plasma

- Maternal plasma samples from 500 pregnant women are analyzed using the disclosed method.

- Forty-two cases are identified as trisomy 21, all confirmed by invasive testing.

- No false negatives or false positives are observed.

- The method detects trisomy 21 at fetal fractions as low as 4.2%.

### Example 7: Detection of Trisomy in which Homologous Chromosome Recombination has Occurred

- A case of trisomy 21 with maternal meiosis I and meiosis II patterns is identified on chromosome 21.

- The transition from one pattern to another is detected at a specific SNP position, indicating a recombination breakpoint.

- The likelihood of trisomy is recalculated across the two segments independently.

- The combined likelihood difference exceeds the threshold, confirming trisomy despite recombination.

- The method correctly identifies the meiotic origin and recombination event.

### Example 8: Detection of Chromosome Microdeletion (Example of DiGeorge)

- A mixture of maternal DNA and a cell line with 22q11.2 deletion is prepared at fetal fractions of 5% and 8%.

- The sample is processed using the COATE method.

- A sliding window analysis detects a localized drop in allelic fraction and read depth spanning 3.1 Mb on chromosome 22.

- The ΔL value exceeds the threshold for microdeletion, confirming the presence of the deletion.

- The result is confirmed by microarray analysis.

### Example 9: Detection of Dominant Monogenic Variation (FGFR3:p.G380R)

- Fetal and maternal DNAs are mixed to simulate a heterozygous fetal mutation in FGFR3.

- The sample is processed using the COATE method and sequenced.

- The variant is identified at position chr4:115,977,528 with an allelic fraction of 8.2%.

- The ACD filter confirms the variant is unlikely to be maternal.

- The FMID filter confirms the fragment length is significantly shorter than maternal fragments.

- The ΔL value exceeds -5, confirming the variant is of fetal origin.

- The capture probe sequence is shown as 5’-ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC-3’.

- Detection result confirms the presence of the pathogenic variant.

- System error rate is determined to be 0.05%.

- Probabilities ΔL of gene mutations are calculated and exceed the clinical threshold.