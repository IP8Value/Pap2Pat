# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to methods and compositions for the direct and quantitative sequencing of cytosine modifications in DNA, specifically 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), 5-formylcytosine (5fC), and 5-carboxylcytosine (5caC). The methods described herein are bisulfite-free and provide a subtraction-free approach for the specific and accurate detection of these modifications, thereby overcoming the limitations associated with traditional bisulfite-based methods.

## BACKGROUND

The primary DNA sequence, composed of the four-letter alphabet G, C, A, and T, carries the genetic information essential for life. Chemical modifications of DNA bases, such as 5-methylcytosine (5mC) and 5-hydroxymethylcytosine (5hmC), add an extra layer of information that plays crucial roles in various biological processes, including gene regulation and normal development. 5-Hydroxymethylcytosine (5hmC) is converted from 5mC by the ten-eleven translocation (TET) family of dioxygenases and is particularly enriched in neuronal cells. Further oxidation by TET enzymes results in the formation of 5-formylcytosine (5fC) and 5-carboxylcytosine (5caC), which are intermediates in the active DNA demethylation pathway.

Traditional methods for detecting cytosine modifications, such as bisulfite sequencing (BS), have several limitations. BS involves harsh chemical treatments that degrade DNA and reduce sequence complexity. Recent advances have led to the development of bisulfite-free methods, such as APOBEC-coupled epigenetic sequencing (ACE-seq) and Enzymatic Methyl-seq (EM-seq), which improve upon BS but still suffer from indirect detection issues. TET-assisted pyridine borane sequencing (TAPS) is a promising bisulfite-free method that directly detects 5mC and 5hmC without causing DNA damage.

However, distinguishing between 5mC and 5hmC typically requires performing two separate assays and subtracting the results, which can introduce errors and require higher sequencing depth. There is a need for a subtraction-free, bisulfite-free method that can directly and accurately detect 5mC and 5hmC, as well as 5fC and 5caC, in a whole-genome context.

## SUMMARY OF THE INVENTION

The present invention provides methods and compositions for the direct and quantitative sequencing of cytosine modifications in DNA, specifically 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), 5-formylcytosine (5fC), and 5-carboxylcytosine (5caC). The methods are bisulfite-free and subtraction-free, offering improved accuracy and reduced DNA damage compared to traditional methods.

In one aspect, the invention provides a method for bisulfite-free 5mC-specific sequencing, referred to as TAPS with β-glucosyltransferase (βGT) blocking (TAPSβ). The method includes the steps of:
1. Blocking 5hmC with βGT to prevent its oxidation.
2. Oxidizing 5mC to 5-carboxylcytosine (5caC) using TET proteins.
3. Reducing 5caC to dihydrouracil (DHU) using pyridine borane.
4. Amplifying and sequencing the modified DNA, where DHU is read as thymine (T).

In another aspect, the invention provides a method for bisulfite-free 5hmC-specific sequencing, referred to as chemical-assisted pyridine borane sequencing (CAPS). The method includes the steps of:
1. Oxidizing 5hmC to 5-formylcytosine (5fC) using potassium ruthenate (K2RuO4).
2. Reducing 5fC to dihydrouracil (DHU) using 2-methylpyridine borane (pic-borane).
3. Amplifying and sequencing the modified DNA, where DHU is read as thymine (T).

In yet another aspect, the invention provides a method for bisulfite-free 5fC and 5caC-specific sequencing, referred to as pyridine borane sequencing (PS). The method includes the steps of:
1. Reducing 5fC and 5caC to dihydrouracil (DHU) using pyridine borane.
2. Amplifying and sequencing the modified DNA, where DHU is read as thymine (T).

Additionally, the invention provides a method for 5caC-specific sequencing, referred to as pyridine borane sequencing for 5caC (PS-c). The method includes the steps of:
1. Blocking 5fC with O-ethylhydroxylamine.
2. Reducing 5caC to dihydrouracil (DHU) using pyridine borane.
3. Amplifying and sequencing the modified DNA, where DHU is read as thymine (T).

The methods of the invention provide high conversion rates, low false-positive rates, and excellent sequencing quality, making them suitable for whole-genome applications. The invention also includes compositions and kits for performing the methods, as well as computer-readable media and systems for analyzing the sequencing data.

## DETAILED DESCRIPTION OF THE INVENTION

### TAPS with β-Glucosyltransferase (βGT) Blocking (TAPSβ)

#### Overview

TAPSβ is a bisulfite-free method for 5mC-specific sequencing. It utilizes β-glucosyltransferase (βGT) to selectively block 5hmC, preventing its oxidation by TET proteins. The unblocked 5mC is then oxidized to 5-carboxylcytosine (5caC) and reduced to dihydrouracil (DHU) using pyridine borane. The modified DNA is amplified and sequenced, where DHU is read as thymine (T).

#### Method Steps

1. **Blocking 5hmC with βGT**:
   - Ligated DNA is incubated with βGT and UDP-glucose to selectively label 5hmC with glucose, preventing its oxidation.
   - The reaction mixture typically includes 50 mM HEPES buffer (pH 8), 25 mM MgCl2, 200 μM UDP-Glc, and 10 U of βGT for 1 hour at 37°C.
   - The 5hmC-blocked DNA is purified using Ampure XP beads.

2. **Oxidizing 5mC to 5caC**:
   - The 5hmC-blocked DNA is incubated with TET proteins to oxidize 5mC to 5caC.
   - The reaction mixture typically includes 50 mM HEPES buffer (pH 8.0), 100 μM ammonium iron (II) sulfate, 1 mM α-ketoglutarate, 2 mM ascorbic acid, 1 mM dithiothreitol, 100 mM NaCl, 1.2 mM ATP, and 4 μM mTet1CD for 80 minutes at 37°C.
   - Proteinase K is added to the reaction and incubated for 1 hour at 50°C to inactivate the TET proteins.
   - The oxidized DNA is purified using Ampure XP beads and subjected to a second round of TET oxidation to ensure complete oxidation.

3. **Reducing 5caC to DHU**:
   - The double-oxidized DNA is incubated with pyridine borane to reduce 5caC to DHU.
   - The reaction mixture typically includes 600 mM NaAc (pH 4.3) and 1 M pyridine borane.
   - The reaction is incubated at 37°C and 850 r.p.m. for 16 hours and purified using a Zymo-IC column with Oligo Binding Buffer.

4. **Amplifying and Sequencing**:
   - The modified DNA is amplified using a KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - The PCR product is purified using 1× Ampure XP beads and quantified using a Qubit dsDNA HS Assay Kit.
   - The libraries are sequenced on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

### Chemical-Assisted Pyridine Borane Sequencing (CAPS)

#### Overview

CAPS is a bisulfite-free method for 5hmC-specific sequencing. It involves the chemical oxidation of 5hmC to 5-formylcytosine (5fC) using potassium ruthenate (K2RuO4) and the reduction of 5fC to dihydrouracil (DHU) using 2-methylpyridine borane (pic-borane). The modified DNA is amplified and sequenced, where DHU is read as thymine (T).

#### Method Steps

1. **Denaturing and Oxidizing 5hmC to 5fC**:
   - Ligated DNA is denatured in 0.05 M NaOH for 30 minutes at 37°C.
   - The denatured DNA is oxidized with K2RuO4 in a two-step process to ensure complete oxidation.
   - The reaction mixture typically includes 1× oxidant (prepared by diluting 10× oxidant with distilled water) and is incubated at 37°C and 850 r.p.m. for 1 hour, followed by the addition of additional oxidant and incubation for another hour.
   - The oxidized DNA is purified using a Bio-Rad Micro Bio-Spin P-6 SSC column.

2. **Reducing 5fC to DHU**:
   - The oxidized DNA is incubated with pic-borane to reduce 5fC to DHU.
   - The reaction mixture typically includes 0.6 M MES (pH 5.2) and 0.2 M pic-borane.
   - The reaction is incubated at 37°C and 850 r.p.m. for 2 hours and purified using a Zymo-IC column with Oligo Binding Buffer.

3. **Amplifying and Sequencing**:
   - The modified DNA is amplified using a KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - The PCR product is purified using 1× Ampure XP beads and quantified using a Qubit dsDNA HS Assay Kit.
   - The libraries are sequenced on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

### Pyridine Borane Sequencing (PS)

#### Overview

PS is a bisulfite-free method for 5fC and 5caC-specific sequencing. It involves the reduction of 5fC and 5caC to dihydrouracil (DHU) using pyridine borane. The modified DNA is amplified and sequenced, where DHU is read as thymine (T).

#### Method Steps

1. **Reducing 5fC and 5caC to DHU**:
   - Ligated DNA is incubated with pyridine borane to reduce 5fC and 5caC to DHU.
   - The reaction mixture typically includes 0.6 M NaAc (pH 4.3) and 1 M pyridine borane.
   - The reaction is incubated at 37°C and 850 r.p.m. for 16 hours and purified using a Zymo-IC column with Oligo Binding Buffer.

2. **Amplifying and Sequencing**:
   - The modified DNA is amplified using a KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - The PCR product is purified using 1× Ampure XP beads and quantified using a Qubit dsDNA HS Assay Kit.
   - The libraries are sequenced on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

### Pyridine Borane Sequencing for Carboxylcytosine (PS-c)

#### Overview

PS-c is a bisulfite-free method for 5caC-specific sequencing. It involves the blocking of 5fC with O-ethylhydroxylamine and the reduction of 5caC to dihydrouracil (DHU) using pyridine borane. The modified DNA is amplified and sequenced, where DHU is read as thymine (T).

#### Method Steps

1. **Blocking 5fC with O-ethylhydroxylamine**:
   - Ligated DNA is incubated with O-ethylhydroxylamine to block 5fC.
   - The reaction mixture typically includes 10 mM O-ethylhydroxylamine and 100 mM MES buffer (pH 5.0).
   - The reaction is incubated at 37°C and 850 r.p.m. for 4 hours and purified using Ampure XP beads.

2. **Reducing 5caC to DHU**:
   - The 5fC-blocked DNA is incubated with pyridine borane to reduce 5caC to DHU.
   - The reaction mixture typically includes 0.6 M NaAc (pH 4.3) and 1 M pyridine borane.
   - The reaction is incubated at 37°C and 850 r.p.m. for 16 hours and purified using a Zymo-IC column with Oligo Binding Buffer.

3. **Amplifying and Sequencing**:
   - The modified DNA is amplified using a KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - The PCR product is purified using 1× Ampure XP beads and quantified using a Qubit dsDNA HS Assay Kit.
   - The libraries are sequenced on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

### Data Analysis

#### Data Preprocessing

Sequencing reads are trimmed using Trim Galore! to remove adaptors and low-quality bases. Trimmed reads are mapped to a genome combining spike-in sequences and the mm9 mouse genome using BWA mem. PCR duplicates are removed using the MarkDuplicate function of Picard. Reads with MAPQ < 10 are excluded from methylated site calling. Modified bases are called using asTair, and raw signals are calculated as the ratio between C and C+T at each site. Regions known to be prone to mapping artifacts and known single nucleotide variants of the E14 cell line are excluded from subsequent analysis.

#### Pairwise Comparisons

Pairwise comparisons of TAPSβ, CAPS, and other methods are performed using published datasets. The Pearson correlation coefficient (Pearson’s r) is calculated using the R function cor. Scatterplots with smoothed densities are visualized using the smoothScatter function in R.

#### Coverage Analysis

Coverage analysis is performed to compare CAPS and ACE-seq. The CpG island annotation is downloaded from UCSC. Each CpG island is evenly binned into ten windows, and the 4-kb flanking regions are binned into 20 windows. The average coverage is calculated using Bedtools map, and the coverage at each site is normalized by the ratio of overall coverage between the two datasets.

#### Estimation of 5hmC Using Maximum Likelihood

The maximum likelihood methylation levels (MLML) estimation method is applied to estimate 5hmC levels from TAPS and TAPSβ. Sites with a minimum coverage of 5 are used for the analysis, and sites with at least one conflict are excluded.

#### Statistical Test of 5hmC

The binomial test is used to call 5hmC at sites with a minimal coverage of five reads. The probability p of the binomial distribution is the false-positive rate of CAPS, calculated from the unmodified control DNA. Cytosines with Benjamini–Hochberg (BH) adjusted p-value < 0.05 are used for downstream analysis.

#### Quantifying Enrichment of 5hmCGs in Regulatory Elements

The list of putative genomic regulatory elements is downloaded from the ENCODE project. High-confidence 5hmCG sites are annotated using bedtools intersect, and the number of 5hmCG sites falling into each category is counted. The enrichment of 5hmCG in each element class is investigated by sampling a set of CG sites for ten times to generate a background distribution of CG sites across element categories.

### Kits and Compositions

The invention also provides kits and compositions for performing the methods described herein. The kits may include reagents such as β-glucosyltransferase (βGT), TET proteins, potassium ruthenate (K2RuO4), 2-methylpyridine borane (pic-borane), pyridine borane, O-ethylhydroxylamine, and other necessary buffers and solutions. The kits may also include instructions for performing the methods and analyzing the sequencing data.

### Computer-Readable Media and Systems

The invention further provides computer-readable media and systems for analyzing the sequencing data generated by the methods described herein. The computer-readable media may include software for data preprocessing, pairwise comparisons, coverage analysis, estimation of 5hmC using maximum likelihood, statistical testing of 5hmC, and quantifying the enrichment of 5hmCGs in regulatory elements. The systems may include hardware and software components for storing and processing the sequencing data.

## EXAMPLES

### Example 1: TAPSβ for Bisulfite-Free 5mC-Specific Sequencing

#### Materials and Methods

- **DNA Sample**: Mouse embryonic stem cells (mESCs) genomic DNA (gDNA).
- **Reagents**: β-glucosyltransferase (βGT), TET proteins, pyridine borane, KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit, Qubit dsDNA HS Assay Kit, Ampure XP beads, Zymo-IC column, Oligo Binding Buffer.

#### Procedure

1. **Blocking 5hmC with βGT**:
   - Incubate 100 ng of ligated DNA with βGT and UDP-glucose in a 50 μL reaction mixture for 1 hour at 37°C.
   - Purify the 5hmC-blocked DNA using Ampure XP beads.

2. **Oxidizing 5mC to 5caC**:
   - Incubate the 5hmC-blocked DNA with TET proteins in a 50 μL reaction mixture for 80 minutes at 37°C.
   - Add Proteinase K and incubate for 1 hour at 50°C.
   - Purify the oxidized DNA using Ampure XP beads and subject to a second round of TET oxidation.

3. **Reducing 5caC to DHU**:
   - Incubate the double-oxidized DNA with pyridine borane in a 50 μL reaction mixture for 16 hours at 37°C.
   - Purify the modified DNA using a Zymo-IC column with Oligo Binding Buffer.

4. **Amplifying and Sequencing**:
   - Amplify the modified DNA using the KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - Purify the PCR product using 1× Ampure XP beads and quantify using the Qubit dsDNA HS Assay Kit.
   - Sequence the libraries on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

#### Results

- **Conversion Rates**: High 5mC conversion rate (97.6%) and low false-positive rate (0.24%) were achieved in TAPSβ.
- **Correlation Analysis**: Good correlation between TAPSβ and published 5mC data of mESCs by reduced representation oxBS-seq (Pearson’s r = 0.77) and whole-genome oxBS-seq (Pearson’s r = 0.72).
- **Sequencing Quality**: TAPSβ showed much improved sequencing quality evidenced by higher mapping rate (90.7%) compared to RRoxBS-seq (66.2–68.2%) and oxBS-seq (21.4–26.1%).

### Example 2: CAPS for Bisulfite-Free 5hmC-Specific Sequencing

#### Materials and Methods

- **DNA Sample**: Mouse embryonic stem cells (mESCs) genomic DNA (gDNA).
- **Reagents**: Potassium ruthenate (K2RuO4), 2-methylpyridine borane (pic-borane), KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit, Qubit dsDNA HS Assay Kit, Ampure XP beads, Zymo-IC column, Oligo Binding Buffer.

#### Procedure

1. **Denaturing and Oxidizing 5hmC to 5fC**:
   - Denature 100 ng of ligated DNA in 0.05 M NaOH for 30 minutes at 37°C.
   - Oxidize the denatured DNA with K2RuO4 in a two-step process.
   - Purify the oxidized DNA using a Bio-Rad Micro Bio-Spin P-6 SSC column.

2. **Reducing 5fC to DHU**:
   - Incubate the oxidized DNA with pic-borane in a 50 μL reaction mixture for 2 hours at 37°C.
   - Purify the modified DNA using a Zymo-IC column with Oligo Binding Buffer.

3. **Amplifying and Sequencing**:
   - Amplify the modified DNA using the KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - Purify the PCR product using 1× Ampure XP beads and quantify using the Qubit dsDNA HS Assay Kit.
   - Sequence the libraries on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

#### Results

- **Conversion Rates**: High 5hmC conversion rate (83.1%) and low false-positive rate (0.72%) were achieved in CAPS.
- **Correlation Analysis**: Good correlation between CAPS and published 5hmC data of mESCs by TAB-seq (Pearson’s r = 0.79) and ACE-seq (Pearson’s r = 0.67).
- **Sequencing Quality**: CAPS outperformed TAB-seq and ACE-seq in sequencing metrics, including higher mapping rate and better base quality.

### Example 3: PS for Bisulfite-Free 5fC/5caC-Specific Sequencing

#### Materials and Methods

- **DNA Sample**: Mouse embryonic stem cells (mESCs) genomic DNA (gDNA).
- **Reagents**: Pyridine borane, KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit, Qubit dsDNA HS Assay Kit, Ampure XP beads, Zymo-IC column, Oligo Binding Buffer.

#### Procedure

1. **Reducing 5fC and 5caC to DHU**:
   - Incubate 100 ng of ligated DNA with pyridine borane in a 50 μL reaction mixture for 16 hours at 37°C.
   - Purify the modified DNA using a Zymo-IC column with Oligo Binding Buffer.

2. **Amplifying and Sequencing**:
   - Amplify the modified DNA using the KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - Purify the PCR product using 1× Ampure XP beads and quantify using the Qubit dsDNA HS Assay Kit.
   - Sequence the libraries on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

#### Results

- **Conversion Rates**: High 5caC conversion rate (93.8%) and good 5fC conversion rate (76.8%) were achieved in PS.
- **False-Positive Rate**: Low false-positive rate (0.27%) in PS.
- **Enrichment Analysis**: 5fC/5caC signals were enriched at H3K4me1, H3K4me3 regions, promoters, and enhancers compared to repressed regions or heterochromatin.

### Example 4: PS-c for Bisulfite-Free 5caC-Specific Sequencing

#### Materials and Methods

- **DNA Sample**: Mouse embryonic stem cells (mESCs) genomic DNA (gDNA).
- **Reagents**: O-ethylhydroxylamine, pyridine borane, KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit, Qubit dsDNA HS Assay Kit, Ampure XP beads, Zymo-IC column, Oligo Binding Buffer.

#### Procedure

1. **Blocking 5fC with O-ethylhydroxylamine**:
   - Incubate 100 ng of ligated DNA with O-ethylhydroxylamine in a 50 μL reaction mixture for 4 hours at 37°C.
   - Purify the 5fC-blocked DNA using Ampure XP beads.

2. **Reducing 5caC to DHU**:
   - Incubate the 5fC-blocked DNA with pyridine borane in a 50 μL reaction mixture for 16 hours at 37°C.
   - Purify the modified DNA using a Zymo-IC column with Oligo Binding Buffer.

3. **Amplifying and Sequencing**:
   - Amplify the modified DNA using the KAPA HiFi HotStart Uracil+ ReadyMix PCR Kit for 4 cycles.
   - Purify the PCR product using 1× Ampure XP beads and quantify using the Qubit dsDNA HS Assay Kit.
   - Sequence the libraries on a NovaSeq 6000 platform (150 bp paired-end reads) with no PhiX added.

#### Results

- **Conversion Rates**: High 5caC conversion rate (95.3%) and low false-positive rate (0.22%) were achieved in PS-c.
- **Enrichment Analysis**: 5caC signals were enriched at H3K4me1, H3K4me3 regions, promoters, and enhancers compared to repressed regions or heterochromatin.

### Conclusion

The methods and compositions of the present invention provide a comprehensive solution for the direct and quantitative sequencing of cytosine modifications in DNA. TAPSβ, CAPS, PS, and PS-c are bisulfite-free and subtraction-free, offering improved accuracy, reduced DNA damage, and excellent sequencing quality. These methods are suitable for whole-genome applications and provide valuable insights into the distribution and function of 5mC, 5hmC, 5fC, and 5caC in various biological processes.