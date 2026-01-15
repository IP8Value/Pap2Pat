Here is the complete patent application following the provided outline:

# DESCRIPTION  

The present application incorporates by reference in their entirety prior U.S. Provisional Patent Application No. 62/XXXXXX, filed Month Day, Year, and U.S. Non-Provisional Patent Application No. XX/XXX,XXX, filed Month Day, Year.  

## RIGHTS OF THE GOVERNMENT  

The invention described herein was made with government support under Grant/Contract No. XXXXX awarded by [Government Agency Name]. The government has certain rights in the invention.  

## FIELD OF THE INVENTION  

The present invention relates generally to the field of infectious disease diagnostics and more specifically to compositions and methods for the rapid, sensitive, and specific detection of Francisella tularensis bacteria, including differentiation between virulent and non-virulent subspecies and subtypes.  

## BACKGROUND OF THE INVENTION  

Tularemia is a severe zoonotic disease caused by the facultative intracellular pathogen Francisella tularensis. This organism is considered one of the most infectious bacteria known, with an infectious dose of fewer than 10 organisms required to cause disease. Due to its extreme virulence and potential for aerosol dissemination, F. tularensis is classified as a Tier 1 select agent by the U.S. government, representing the highest risk category for biological threats to public health and national security.  

Four subspecies of F. tularensis have been identified: F. tularensis subsp. tularensis (Type A), F. tularensis subsp. holarctica (Type B), F. tularensis subsp. mediasiatica, and F. tularensis subsp. novicida. The Type A strains are further divided into subtypes A.I and A.II, with subtype A.I being the most virulent. Three of these subspecies (tularensis, holarctica, and mediasiatica) are classified as select agents, while F. tularensis subsp. novicida is not.  

Transmission of F. tularensis occurs through multiple routes including inhalation, ingestion, and cutaneous exposure. The bacterium can infect over 250 animal species and persists in environmental reservoirs including arthropod vectors (ticks, mosquitoes, deer flies), contaminated water, and soil. Pneumonic tularemia resulting from inhalation of contaminated aerosols is the most severe disease manifestation, with untreated cases progressing to hematogenous spread and acute renal failure. Fatality rates as high as 35% have been reported for infections caused by the hypervirulent subtype A.I strains.  

Current diagnostic methods for F. tularensis suffer from several limitations. While PCR-based assays can identify F. tularensis at the species level, they generally cannot differentiate between the virulent select agent subspecies and the less virulent non-select agent strains. Existing qPCR assays that attempt subspecies differentiation require complex scoring matrices and demonstrate inconsistent performance due to sporadic amplification failures and nonspecific amplification. Furthermore, environmental Francisella-like organisms and tick endosymbionts such as F. persica share genomic content with F. tularensis, leading to potential misidentification.  

There remains an urgent need in public health and defense applications for rapid, accurate diagnostic assays that can definitively identify F. tularensis subspecies and subtypes, particularly the hypervirulent select agent strains. Such assays would enable appropriate medical management and facilitate epidemiological investigations during outbreaks.  

## SUMMARY OF THE INVENTION  

The present invention provides novel compositions and methods for the rapid, sensitive, and specific detection of Francisella tularensis, including differentiation between virulent select agent subspecies and non-select agent strains. The invention encompasses a comprehensive system of singleplex and multiplex qPCR assays capable of identifying all known F. tularensis subspecies and subtypes with high specificity and sensitivity.  

The detection method comprises several key components. First, a universal 16S ribosomal DNA (rDNA) assay (U16S) serves as an endogenous control to verify the presence of bacterial DNA in test samples. This assay targets a conserved region of the 16S rRNA gene modified to accommodate Francisella species and related organisms.  

For species-level detection of all F. tularensis strains, the invention provides the 4Pan1 assay targeting a 116-bp region within the ostA gene encoding an organic solvent tolerance protein. This assay detects all four F. tularensis subspecies without cross-reactivity with near neighbors or environmental Francisella-like organisms.  

Differentiation between virulent select agent subspecies and non-select agent strains is achieved through the 3Pan assay, which targets an 83-bp region within a hypothetical protein gene (FTL_1858) present only in the three select agent subspecies (tularensis, holarctica, and mediasiatica). This assay specifically excludes F. tularensis subsp. novicida.  

For identification of the hypervirulent subtype A.I strains, the A1d assay targets a 114-bp region within an oxidoreductase gene (FTT_0516) unique to this clade. The A2c assay identifies subtype A.II strains through detection of a 101-bp region within the mviN gene (FTW_1702).  

Subspecies-specific detection is provided by several assays: The B2 assay targets an 80-bp region within a hypothetical protein gene (FTS_0806) specific to F. tularensis subsp. holarctica (Type B). The M3 assay detects F. tularensis subsp. mediasiatica through a 112-bp region within a major facilitator superfamily transporter gene (FTM_1104). The N1 assay identifies F. tularensis subsp. novicida by targeting a 140-bp region within a metabolite:H+ symporter family protein gene (FTN_0003).  

The invention further provides optimized multiplex qPCR platforms combining these assays in strategic configurations. A Tier 1 multiplex platform combines the U16S, 4Pan1, 3Pan, and A1d assays to rapidly determine whether a sample contains: 1) a hypervirulent select agent A.I strain (all four targets positive), 2) another select agent strain (U16S, 4Pan1, and 3Pan positive; A1d negative), or 3) a non-select agent strain (only U16S and 4Pan1 positive).  

A Tier 2 multiplex platform combines the U16S, A2c, B2, and M3 assays to further characterize select agent strains that are not subtype A.I. This configuration enables differentiation between subtype A.II, Type B, and mediasiatica strains.  

The assays demonstrate exceptional sensitivity with limits of detection (LOD) ranging from 2-7 fg (1-3 genomic copies) in singleplex format and 10-100 fg (5-49 genomic copies) in multiplex configurations. All assays show perfect specificity when tested against inclusivity panels containing all F. tularensis subspecies and exclusivity panels containing near neighbors and environmental samples.  

Additional embodiments include sequencing-based confirmation methods using barcoded primers compatible with next-generation sequencing platforms. The assays are compatible with standard real-time PCR instruments including the Applied Biosystems 7500 Fast Dx system and 3M Integrated Cycler.  

The invention provides several advantages over existing methods: 1) comprehensive coverage of all F. tularensis subspecies and subtypes in a single testing platform, 2) elimination of complex scoring matrices through strategically designed multiplex configurations, 3) superior sensitivity enabling detection of just 1-3 genomic copies, 4) robust performance across different instrumentation platforms, and 5) flexibility for both clinical diagnostic and environmental surveillance applications.  

## DETAILED DESCRIPTION OF THE INVENTION  

The present invention provides a comprehensive system for detecting and differentiating Francisella tularensis subspecies and subtypes through optimized qPCR assays and strategic multiplex configurations. The system encompasses both singleplex assays for maximum sensitivity and multiplex platforms for efficient screening and characterization.  

The detection process begins with obtaining a suspect specimen, typically from clinical, environmental, or cultured sources. Genomic DNA is extracted from the specimen using standard methods such as the Gentra Puregene yeast/bacteria kit or cetyltrimethylammonium bromide (CTAB) extraction. DNA quantity and quality are assessed by spectrophotometry and gel electrophoresis.  

The universal 16S rDNA (U16S) assay serves as an endogenous control to verify the presence of bacterial DNA in test samples. This assay targets a modified version of conserved 16S rRNA gene sequences, producing a 159-bp amplicon detectable in all bacterial species. The assay uses optimized primer concentrations of 0.5 μM each and a hydrolysis probe concentration of 0.2 μM, with a limit of detection (LOD) of 0.1 pg bacterial DNA.  

For species-level detection of F. tularensis, the 4Pan1 assay targets a 116-bp region within the ostA gene (FTT_0467). This assay uses forward primer at 0.5 μM, reverse primer at 0.75 μM, and hydrolysis probe at 0.3 μM, achieving an LOD of 3 fg (~2 genomic copies). The 4Pan1 assay detects all four F. tularensis subspecies without cross-reactivity with near neighbors.  

Differentiation between virulent select agent subspecies (tularensis, holarctica, mediasiatica) and non-select agent novicida strains is accomplished through the 3Pan assay. This assay targets an 83-bp region within hypothetical protein gene FTL_1858, using primers at 0.5 μM each and probe at 0.4 μM, with an LOD of 5 fg (~3 genomic copies).  

The hypervirulent subtype A.I is identified by the A1d assay targeting a 114-bp region within oxidoreductase gene FTT_0516. This assay uses primers at 0.5 μM each and probe at 0.2 μM, with an LOD of 7 fg (~3 genomic copies). Subtype A.II strains are detected by the A2c assay targeting a 101-bp region within mviN gene FTW_1702, using forward primer at 0.5 μM, reverse primer at 0.25 μM, and probe at 0.3 μM, with an LOD of 5 fg.  

Subspecies-specific detection includes:  
- B2 assay for F. tularensis subsp. holarctica (Type B): Targets 80-bp in FTS_0806, primers at 0.5 μM each, probe at 0.2 μM, LOD 5 fg  
- M3 assay for F. tularensis subsp. mediasiatica: Targets 112-bp in FTM_1104, forward primer 1.0 μM, reverse 0.75 μM, probe 0.3 μM, LOD 2 fg  
- N1 assay for F. tularensis subsp. novicida: Targets 140-bp in FTN_0003, forward primer 1.0 μM, reverse 0.75 μM, probe 0.3 μM, LOD 3 fg  

The Tier 1 multiplex platform combines U16S, 4Pan1, 3Pan, and A1d assays to provide:  
1) Positive for all four targets: Hypervirulent select agent A.I strain  
2) Positive for U16S, 4Pan1, 3Pan; negative for A1d: Other select agent strain  
3) Positive for U16S and 4Pan1 only: Non-select agent novicida strain  

The Tier 2 multiplex platform combines U16S, A2c, B2, and M3 assays to differentiate select agent strains that are not subtype A.I.  

All assays demonstrate linear standard curves with R² values ≥0.976 in singleplex format and ≥0.961 in multiplex configurations. The assays are compatible with direct analysis from colony material by suspending several CFUs in water, heating to 98°C for 10 minutes, and using 2 μl in a 20 μl reaction.  

For sequencing confirmation, barcoded primers compatible with Ion Torrent PGM sequencing are used. Sequencing provides strain-level discrimination by identifying single nucleotide polymorphisms (SNPs) in the amplified regions.  

### Example 1  

Analytical sensitivity was determined using 10-fold serial dilutions of F. tularensis genomic DNA from 1 ng to 0.1 fg. The LOD ranges for singleplex assays were:  
- U16S: 0.1 pg  
- 4Pan1: 3 fg  
- 3Pan: 5 fg  
- A1d: 7 fg  
- A2c: 5 fg  
- B2: 5 fg  
- M3: 2 fg  
- N1: 3 fg  

Standard curves showed excellent linearity with R² values:  
- 4Pan1: 0.998  
- 3Pan: 0.999  
- A1d: 0.976  
- A2c: 0.999  
- B2: 0.998  
- M3: 0.997  
- N1: 1.000  

Multiplex Tier 1 platform LODs:  
- U16S: 50 fg (7500 Fast Dx), 100 fg (3M Integrated Cycler)  
- 4Pan1: 10 fg  
- 3Pan: 30 fg  
- A1d: 30 fg  

Multiplex Tier 2 platform LODs:  
- U16S: 50 fg  
- A2c: 10 fg  
- B2: 30 fg  
- M3: 10 fg  

### Example 2  

The multiplex Tier 1 real-time qPCR assay was developed to detect all bacteria (U16S), all F. tularensis (4Pan1), select agent subspecies (3Pan), and hypervirulent A.I strains (A1d). Optimal 10× stock compositions were:  
- U16S: 5 μM forward primer, 5 μM reverse primer, 2 μM probe  
- 4Pan1: 5 μM forward, 7.5 μM reverse, 3 μM probe  
- 3Pan: 5 μM forward, 5 μM reverse, 4 μM probe  
- A1d: 5 μM forward, 5 μM reverse, 2 μM probe  

The multiplex Tier 2 assay for differentiating select agent strains used:  
- U16S: 5 μM forward, 5 μM reverse, 2 μM probe  
- A2c: 5 μM forward, 2.5 μM reverse, 3 μM probe  
- B2: 5 μM forward, 5 μM reverse, 2 μM probe  
- M3: 10 μM forward, 7.5 μM reverse, 3 μM probe  

Specificity testing against inclusivity panels (all F. tularensis subspecies) and exclusivity panels (near neighbors and environmental samples) showed 100% accuracy. No cross-reactivity was observed with Francisella philomiragia, F. persica, or tick endosymbionts.  

The assays provide significant advantages for molecular diagnostic testing by:  
1) Enabling rapid identification of hypervirulent strains requiring immediate intervention  
2) Differentiating select agent from non-select agent strains for regulatory compliance  
3) Providing sensitive detection from limited sample material  
4) Offering flexible singleplex or multiplex configurations for various testing needs  

While particular embodiments of the invention have been described, the scope of the invention is not limited to these specific examples and encompasses all variations and modifications within the spirit of the disclosure.