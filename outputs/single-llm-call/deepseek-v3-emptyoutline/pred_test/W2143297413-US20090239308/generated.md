Here is the complete patent application following the provided outline and incorporating the research paper's invention:

---

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates generally to the field of molecular biology and genetic analysis, specifically to systems and methods for accurately determining copy number variations (CNVs) in DNA samples. CNVs, which represent gains or losses of genomic regions ranging from 500 bases to several kilobases in size, have been increasingly recognized as significant contributors to human genetic diversity and disease susceptibility. Traditional methods for CNV detection, such as array-based comparative genomic hybridization (array-CGH) and high-density single nucleotide polymorphism (SNP) microarrays, while high-throughput, suffer from limitations in resolution and sensitivity.  

Real-time polymerase chain reaction (PCR), while sequence-specific and relatively easy to perform, lacks sufficient discriminating power to detect differences beyond two-fold changes. Digital PCR techniques have emerged as a promising alternative, offering single-molecule sensitivity through limiting dilutions of target DNA followed by amplification. However, conventional digital PCR methods face challenges in scalability, throughput, and accurate quantification, particularly when analyzing multiple targets simultaneously.  

There exists a pressing need in the art for a robust, high-throughput platform capable of precise CNV quantification with single-molecule resolution, while maintaining operational simplicity and flexibility for analyzing multiple genomic targets in parallel. The present invention addresses these needs through novel integration of nanofluidic partitioning technology with advanced statistical analysis methods, enabling unprecedented accuracy in CNV determination.  

## SUMMARY OF THE INVENTION  

The present invention provides a comprehensive system and method for determining copy number variations in DNA samples with high precision and reliability. At the core of the invention is a nanofluidic biochip apparatus, termed a "digital array," which partitions DNA molecules into hundreds of nanoliter-volume reaction chambers for parallel amplification and detection. This physical partitioning approach eliminates the need for serial dilutions required in conventional digital PCR, while providing statistically robust sampling of target molecules.  

The invention encompasses several key innovations: First, a novel nanofluidic chip architecture incorporating integrated channels and valves that precisely partition mixtures of sample DNA and PCR reagents into 765 discrete reaction chambers per panel, each with a volume of approximately 6 nanoliters. Second, a multiplexed detection system capable of simultaneously quantifying multiple target sequences within the same partitioned reaction volume, thereby eliminating pipetting errors associated with separate reactions. Third, a sophisticated mathematical framework and associated computational algorithms for deriving true molecular concentrations from observed positive chamber counts, including calculation of confidence intervals for both absolute concentrations and concentration ratios.  

The method of the invention involves preparing a reaction mixture containing universal PCR master mix, sequence-specific primers and probes for both target and reference genes, and the DNA sample of interest. This mixture is loaded into the digital array and partitioned into hundreds of nanoliter-scale reaction chambers. Following thermocycling, the presence or absence of amplification products in each chamber is detected through fluorescence signals corresponding to the target and reference sequences. The counts of positive chambers for each target are then processed through the novel statistical algorithms to determine both the absolute concentration of each sequence and the ratio between target and reference sequences, which directly reflects the copy number variation in the sample.  

The invention provides significant advantages over existing technologies, including: (1) elimination of serial dilution requirements through physical partitioning; (2) simultaneous quantification of multiple targets in the same reaction volume; (3) robust statistical methods providing confidence intervals for concentration estimates; (4) single-molecule sensitivity enabling detection of subtle CNVs; and (5) flexibility to analyze various genomic targets using the same platform. These innovations make the invention particularly valuable for research and clinical applications requiring precise CNV detection, such as cancer genomics, genetic disease diagnosis, and pharmacogenomics.  

## DETAILED DESCRIPTION OF SPECIFIC EMBODIMENTS  

The present invention is implemented through several interconnected components and methods, which are described in detail below with reference to specific embodiments.  

**Nanofluidic Digital Array Apparatus**  
The digital array comprises a biochip fabricated from optically transparent materials (e.g., glass or polymer) containing integrated fluidic channels and valves. Each panel of the array contains precisely 765 reaction chambers, each with a volume of 6 nanoliters, resulting in a total reaction volume of 4.59 microliters per panel. The chambers are arranged in a geometrically optimized pattern to ensure uniform partitioning of the sample mixture. Integrated pneumatic valves control the flow of sample and reagents into the chambers, ensuring complete and consistent filling. The array is designed to be compatible with standard thermocycling instruments and fluorescence detection systems.  

In a preferred embodiment, the digital array is configured to operate with the BioMark real-time PCR system (Fluidigm Corporation), allowing simultaneous thermal cycling and fluorescence detection across all chambers. The system captures fluorescence signals at each cycle, with distinct fluorophores (e.g., FAM and VIC) used to label different target sequences. Detection optics are aligned to resolve fluorescence from individual chambers, enabling binary determination (positive/negative) for each target in each chamber.  

**Sample Preparation and Loading**  
The method begins with preparation of a 10 microliter reaction mixture containing: 1× TaqMan Universal PCR master mix, sequence-specific primers (typically 900 nM) and probes (typically 200 nM) for both target and reference genes, 1× sample loading reagent, and DNA sample containing approximately 1,100-1,300 copies of the reference gene. The reference gene is typically RNase P, a single-copy gene in the human genome, though other suitable reference genes may be used.  

Of this mixture, 4.59 microliters is loaded into each panel of the digital array. The integrated fluidic system partitions this volume uniformly across all 765 chambers, resulting in random distribution of DNA molecules according to Poisson statistics. This partitioning represents a critical improvement over serial dilution methods, as it ensures statistical robustness while maintaining single-molecule sensitivity.  

**Thermocycling Protocol**  
Following loading, the digital array undergoes thermal cycling under the following conditions: initial denaturation at 95°C for 10 minutes, followed by 40 cycles of two-step PCR consisting of 15 seconds at 95°C for denaturation and 1 minute at 60°C for annealing and extension. Throughout the cycling process, fluorescence signals from all chambers are monitored at each cycle endpoint. The two-step cycling protocol enhances specificity while maintaining efficiency, particularly important for the nanoliter-scale reactions.  

**Data Acquisition and Initial Processing**  
Following amplification, the Digital PCR Analysis software processes the fluorescence data to determine positive chambers for each target. A chamber is scored as positive for a particular target if its fluorescence intensity exceeds a predetermined threshold specific to that fluorophore. This binary determination (positive/negative) for each chamber forms the primary data for subsequent statistical analysis.  

**Mathematical Framework for Concentration Estimation**  
The invention incorporates a novel mathematical framework for deriving true molecular concentrations from the observed counts of positive chambers. The relationship between the true concentration λ (molecules per chamber) and the probability p of a chamber being positive is given by:  

p = 1 - e^(-λ)  

This follows from modeling the number of molecules per chamber as a Poisson process. For a panel with C chambers (typically 765) and H observed positive chambers, the estimator p̂ = H/C has expectation p and variance p(1-p)/C. Through the inverse relationship:  

λ̂ = -ln(1 - p̂)  

we obtain an estimator for the true concentration. The sampling distribution of λ̂ can be derived from that of p̂ through transformation of variables, enabling calculation of confidence intervals.  

For a 95% confidence interval on p̂:  

[p̂ - z_c√(p̂(1-p̂)/C), p̂ + z_c√(p̂(1-p̂)/C)]  

where z_c = 1.96. This transforms to a 95% confidence interval for λ̂:  

[-ln(1 - (p̂ - z_c√(p̂(1-p̂)/C))), -ln(1 - (p̂ + z_c√(p̂(1-p̂)/C)))]  

**Ratio Estimation and Confidence Intervals**  
For CNV analysis, the critical parameter is the ratio r = λ_1/λ_2 between concentrations of target and reference sequences. The invention provides methods to estimate this ratio and determine its confidence interval through several approaches:  

1. **Direct Method**: Using the estimators λ̂_1 and λ̂_2, the ratio estimator is simply r̂ = λ̂_1/λ̂_2. A confidence interval can be obtained through Fieller's theorem, which accounts for the covariance between the two concentration estimates.  

2. **Numerical Integration Method**: More accurate confidence intervals are obtained by numerically integrating the joint probability distribution of λ̂_1 and λ̂_2 over regions corresponding to specific ratio values. This method makes no assumptions about normality and can handle asymmetric distributions.  

The numerical method proceeds by:  
(a) Constructing fine-grained histograms of the sampling distributions for λ̂_1 and λ̂_2  
(b) For each possible ratio value r, integrating the joint probability density over all (λ_1, λ_2) pairs satisfying λ_1/λ_2 = r  
(c) Accumulating these probabilities to build the sampling distribution q(r̂)  
(d) Determining confidence intervals from percentiles of q(r̂)  

**Experimental Validation**  
The invention has been experimentally validated using synthetic DNA constructs spiked into human genomic DNA at known ratios. A 65-base oligonucleotide matching part of the human RPP30 gene was used as the target sequence, while RNase P served as the reference. Mixtures were prepared with RPP30:RNase P ratios of 1:1, 1:1.5, 1:2, 1:2.5, 1:3, and 1:3.5, simulating samples with 2 to 7 copies of RPP30 per diploid cell.  

Analysis of these mixtures demonstrated accurate ratio estimation across the entire range, with the known ratio falling within the 95% confidence interval in all cases. Precision improved with increasing numbers of panels, with clear separation between different ratios achieved when using three or more panels (totaling ≥2,295 chambers).  

**Applications and Advantages**  
The invention finds particular utility in:  
- Cancer genomics, where CNVs of oncogenes and tumor suppressors are common  
- Genetic disease diagnosis, particularly for disorders caused by gene dosage imbalances  
- Pharmacogenomics, where gene copy number may affect drug metabolism  
- Basic research into genome structure and variation  

Key advantages include:  
- Single-molecule sensitivity without requiring serial dilutions  
- Simultaneous multiplex analysis of multiple targets  
- Built-in quality control through confidence interval estimation  
- Flexibility to analyze various targets using the same platform  
- Compatibility with standard laboratory equipment  

While particular embodiments have been described, the invention encompasses various modifications and alternative implementations that maintain the core principles of nanofluidic partitioning coupled with advanced statistical analysis for CNV determination.  

--- 

This completes the patent application following the specified outline while incorporating all essential elements from the research paper in proper patent format. The application provides comprehensive coverage of the invention's background, summary, and detailed embodiments while maintaining formal patent language throughout.