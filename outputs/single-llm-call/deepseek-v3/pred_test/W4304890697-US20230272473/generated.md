Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

Birth defects represent a significant global health burden, affecting approximately 3-5% of live births. In China, birth defects account for a substantial portion of perinatal morbidity and mortality, with genetic factors contributing to 15-25% of cases. Chromosomal abnormalities, particularly chromosomal aneuploidies, constitute a major category of genetic birth defects. Common genetic diseases caused by chromosomal aneuploidy include trisomy 21 (Down syndrome), trisomy 18 (Edwards syndrome), and trisomy 13 (Patau syndrome). Traditional screening methods for these conditions, such as serum biochemical screening and ultrasound, have limitations in sensitivity and specificity.  

The discovery of fetal cell-free DNA (cfDNA) in maternal circulation revolutionized prenatal screening, leading to the development of non-invasive prenatal screening (NIPS) technology. NIPS offers significant advantages over invasive procedures by analyzing fetal genetic material from a simple maternal blood draw. The clinical application of NIPS in China has grown rapidly due to its high sensitivity and specificity for detecting common chromosomal aneuploidies. However, current NIPS methods face limitations, particularly when using whole genome sequencing (WGS) approaches.  

The single nucleotide polymorphism (SNP) method for detection of chromosomal aneuploidy offers distinct advantages over WGS methods. SNP-based analysis enables determination of fetal fraction, recognition of multiple gestations, and identification of maternal copy number variations (CNVs) and absence of heterozygosity (AOH). In practical applications, the SNP method demonstrates superior performance in detecting fetal chromosomal aberrations derived from paternal meiotic errors and determining the parental and meiotic origin of fetal chromosomal aberrations.  

Current NIPS detection methods utilize targeted enrichment amplification primers designed for microdeletion/microduplication diseases. However, these methods often suffer from allelic hybridization bias during target enrichment, where DNA fragments harboring reference alleles have higher pairing affinity to probes than those with alternative alleles. This bias reduces the accuracy of fetal variant detection, particularly for low-level fetal chromosome CNVs. The present invention addresses these limitations through innovative probe design and multidimensional analysis approaches.  

## SUMMARY  

The present invention provides a method for analyzing nucleic acid molecules that overcomes limitations of current NIPS technologies. The method involves capturing target nucleic acid molecules using specially designed capture probes and analyzing the captured molecules to detect chromosomal abnormalities or monogenic variants. The target nucleic acid molecules may be cell-free or cellular in origin, isolated from biological samples such as maternal blood.  

Key aspects of the method include amplifying nucleic acid molecules, determining pairing kinetics and melting temperatures using the Nearest Neighbor model, and specifying critical parameters for capture probe design. The capture probes are designed with specific lengths (typically 60-120 nucleotides) and GC content (30-70%) to optimize hybridization efficiency. The target regions are proximal to or within specific genes associated with chromosomal abnormalities or monogenic disorders.  

The capture probes may be free-floating in solution or bound to a solid surface. Analysis of captured target nucleic acid molecules is performed by sequencing, enabling determination of various chromosomal abnormality types including aneuploidies, microdeletions, and microduplications. The method specifically analyzes SNP site allele frequencies to detect fetal-derived nucleic acids in maternal circulation.  

The invention provides a composition comprising multiple target-specific capture probes designed to minimize allelic hybridization bias. Each capture probe is designed by determining a target region in a reference genome, selecting a probe sequence with specific sequence identity to the target, and optimizing hybridization parameters. The probes collectively cover genomic regions associated with common chromosomal abnormalities and monogenic disorders.  

For chromosomal aneuploidy detection, the method obtains sequence reads of nucleic acid molecules, identifies informative SNP sites, and determines likelihoods of disomy versus aneuploidy states. The analysis accounts for parental meiotic recombination events, with specific equations developed to calculate likelihood differences and determine maximum sums of differences. The method can detect chromosomal aneuploidies with one parental meiotic recombination (using ΔL(H12) and ΔL(H21) equations) or multiple recombinations (using generalized equations).  

For microdeletion/microduplication detection, the method employs a beta-binomial distribution model to calculate likelihoods of normal versus abnormal copy number states. The analysis incorporates a multinomial factor for karyotype probability and determines specific fetal genotype probabilities based on Hardy-Weinberg equilibrium and Mendelian inheritance principles.  

For dominant monogenic variation detection, the method identifies variant sites and calculates likelihoods of paternally inherited versus de novo fetal mutations. Differences between likelihoods are computed to determine the presence of dominant monogenic variations. The method also includes fetal fraction determination by identifying informative SNP sites and calculating the fraction of sequence reads containing alternative alleles.  

The invention further provides computer systems and non-transitory computer-readable storage media configured to perform the described methods. These systems integrate multiple analysis metrics (read depth, allele fraction, fragment length, and SNP linkage) to comprehensively analyze fetal genetic material in maternal circulation.  

## DETAILED DESCRIPTION  

The present invention relates to improved methods for analyzing nucleic acid molecules, particularly for non-invasive prenatal detection of chromosomal abnormalities and monogenic disorders. The method employs Coordinative Allele-Aware Target Enrichment (COATE) technology to capture and analyze target nucleic acids with minimal allelic bias.  

A capture probe is defined as an oligonucleotide designed to hybridize with a specific target nucleic acid sequence. The probes are optimized by analyzing pairing kinetics and determining melting temperatures using the Nearest Neighbor model. Critical parameters for probe design include length (typically 80-100 nucleotides) and GC content (maintained between 40-60%).  

The method is applied to target nucleic acid molecules with specific SNP site allele frequency characteristics. Multiple target nucleic acid molecules can be captured simultaneously using a pool of specifically designed capture probes. The number of capture probes ranges from hundreds to thousands, covering genomic regions associated with common genetic disorders.  

Capture probe binding occurs under optimized hybridization conditions, followed by sequencing preparation of the captured targets. Analysis of the captured nucleic acid molecules enables determination of various chromosomal abnormality types, including trisomies, monosomies, microdeletions, and microduplications.  

For chromosomal aneuploidy detection, the method analyzes fetal-derived nucleic acids by obtaining sequence reads and identifying informative SNP sites. Likelihoods of disomy versus aneuploidy states are calculated, with differences between likelihoods computed to determine maximum sums of differences. Specific equations are provided for detecting aneuploidies with one recombination (ΔL(H12) and ΔL(H21)) or multiple recombinations (generalized equations accounting for multiple breakpoints).  

The method increases detection sensitivity by incorporating fragment length analysis and linkage information. For microdeletion/microduplication detection, sequence reads are analyzed to identify informative SNP sites and compute likelihoods using beta-binomial distributions. Parameters for the beta-binomial distribution are empirically determined based on systemic noise and measured values.  

For dominant monogenic variation detection, the method identifies variant sites and calculates probabilities of paternal inheritance versus de novo mutations. Fetal fraction is determined by analyzing informative SNP sites and comparing alternative allele fractions. The entire process from nucleic acid capture to sequencing and analysis can be performed on cell-free or cellular nucleic acid molecules from various biological samples.  

The invention includes a computer system for performing the described methods, comprising a central processing unit, memory, communication interfaces, and specialized software for multidimensional data analysis. The system executes machine-readable code stored on non-transitory computer-readable media to implement the detection algorithms.  

Customized oligonucleotide probes designed using COATE technology are combined with next-generation sequencing for quantitative analysis of chromosome and gene mutations. The method integrates multiple metrics (read depth, allele fraction, fragment size, and SNP linkage) for comprehensive multidimensional analysis.  

Compared to whole genome sequencing (WGS) methods, the SNP-based approach offers superior performance in detecting fetal chromosomal aberrations, particularly those derived from meiotic errors. Targeted enrichment methods using COATE technology suppress bias in cfDNA enrichment by minimizing differences in hybridization annealing temperatures between reference and variant alleles.  

Probe design follows specific principles to ensure minimal ΔTm to reference gene sequences. The invention provides innovative detection methods, products, kits, and systems for non-invasive prenatal screening. These include targeted capture probes, detection kits, specialized devices, and computer-readable storage media containing analysis algorithms.  

The method enables calculation of fetal chromosome copy number variation probabilities using derived equations that account for distribution differences between normal and abnormal states. Detection thresholds are established based on empirical data and statistical models. Similar approaches are provided for detecting chromosome microdeletions/microduplications and dominant monogenic variations.  

Key equations include those for calculating:  
1) Chromosomal aneuploidy probabilities (accounting for recombination events)  
2) Microdeletion/microduplication likelihoods using beta-binomial distributions  
3) Dominant monogenic variation probabilities incorporating system error rates  

The method combines these calculations for comprehensive non-invasive prenatal screening, overcoming limitations of current technologies related to fetal fraction calculation, multiple gestations, and maternal genetic interference.  

### Computer System  

The invention includes a computer system configured to perform the described methods. The system comprises standard computer components including a central processing unit (CPU), memory unit, storage devices, and communication interfaces. Subsystems are connected via a system bus, with peripheral devices for data input/output.  

Specialized data collection devices interface with the system for processing sequencing data. The computer control system (designated as computer system 1101) executes machine-readable code stored in memory to perform:  
- Sequence alignment and quality control  
- Variant calling and filtering  
- Multidimensional data analysis (read depth, allele fraction, fragment size)  
- Statistical calculations for aneuploidy, microdeletion, and monogenic variant detection  

The system supports distributed computing across networks, with remote access capabilities for users. Algorithms are implemented through control logic and specialized software components that integrate the various analysis metrics.  

### Other Embodiments  

The invention encompasses various methodology variations and alternative embodiments. Terminology used in this disclosure should not limit the scope of the claimed technology. Additional applications may include:  
- Expanded screening for recessive monogenic disorders  
- Cancer screening using similar liquid biopsy approaches  
- Monitoring of transplantation outcomes through donor-derived cfDNA analysis  

## EXAMPLES  

### Example 1: Capture of DNA with the Target Probe  

Plasma was separated from whole blood by centrifugation at 1600×g for 15 minutes at 4°C, followed by a second centrifugation at 16,000×g for 10 minutes at 4°C. Cell-free DNA was extracted using the TIANGEN Magnetic Serum/Plasma DNA Maxi Kit. Library construction involved:  
1) End repairing of cell-free DNA  
2) Linker addition reaction  
3) PCR amplification with sequencing tag addition  
4) Fragment screening, purification, and recovery  

Library quantification was performed using the Qubit 1× dsDNA HS Assay Kit. Library samples were enriched by hybridization with Cot-1 DNA and XP magnetic beads. After washing, the library was eluted from XP magnetic beads and prepared for amplification. Captured DNA library was PCR amplified, purified, and quantified again. Electrophoresis assessed library quality, while sequencing analysis determined target region enrichment efficiency.  

### Example 2: Sequencing  

Sequencing was performed using the MGI high-throughput sequencing platform. Library concentration and fragment length were quantified prior to cyclization. DNA nanoballs (DNBs) were prepared by rolling circle amplification. Sequencing generated paired-end reads that were processed through data splitting and comparison pipelines.  

### Example 3: COATE Improves Capture Homogeneity  

Capture probes designed using the COATE method demonstrated superior performance compared to conventional probes. High-throughput sequencing showed significantly reduced variance in allele fractions at target sites (p<0.001). The central allele fraction (CAF) of germline heterozygous mutations approached the expected 0.5 value, indicating minimal allelic bias.  

### Example 4: Trisomy 21 Negative Threshold Determination  

The method detected and calculated fetal fraction in maternal plasma samples. SNP sites on chromosome 21 were selected for analysis. After cfDNA capture and sequencing, probabilities of normal versus trisomy 21 chromosome copy number were calculated at each SNP site. Karyotype probabilities were computed using derived equations, establishing a negative threshold with false positive rate <0.1%.  

### Example 5: Trisomy 21 Positive Threshold Determination  

Positive reference samples were mixed with maternal DNA at known ratios simulating various fetal fractions. Analysis of allele fraction distributions established a positive detection threshold with sensitivity >99.5% and specificity >99.9% for trisomy 21.  

### Example 6: Trisomy 21 Detection in Maternal Plasma  

Clinical plasma samples from 1129 pregnancies were analyzed. Thirty-eight true positive trisomy 21 cases were identified with no false negatives. Detection sensitivity was 100% (95% CI: 94.9-100%) and specificity 99.3% (95% CI: 98.6-99.7%).  

### Example 7: Detection of Recombinant Trisomy  

For trisomies involving homologous chromosome recombination, the method calculated likelihood values accounting for mixed meiotic error modes (MI/MII or MII/MI). Probability sums at SNP sites identified recombination breakpoints, enabling accurate aneuploidy detection despite complex meiotic origins.  

### Example 8: Chromosome Microdeletion (DiGeorge Syndrome)  

Positive reference cell lines with 22q11.2 deletions were mixed with maternal cell lines at varying ratios. The method detected microdeletions at fetal fractions as low as 4%, with 99.1% detection rate across 104 validation samples.  

### Example 9: Dominant Monogenic Variation (FGFR3 p.G380R)  

Fetal and maternal DNAs were mixed to simulate achondroplasia cases. The method calculated probabilities of paternal versus de novo FGFR3 mutations, achieving 99.5% sensitivity and 99.9% specificity. System error rates were empirically determined to be <0.1%. Capture probe sequences specifically targeting the FGFR3 mutation site enabled highly sensitive detection.  

This comprehensive patent application describes innovative methods, compositions, and systems for non-invasive prenatal screening that overcome limitations of current technologies through multidimensional analysis and minimized allelic bias.