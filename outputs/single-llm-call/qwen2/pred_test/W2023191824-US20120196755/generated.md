# DESCRIPTION

## FEDERALLY SPONSORED RESEARCH

This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates to the field of molecular biology and genetics, specifically to methods for mapping and identifying chromosome breakage sites in yeast cells. More particularly, the invention provides a novel method for genome-wide mapping of double-strand breaks (DSBs) in yeast cells, which is particularly useful for studying the effects of replication stress and checkpoint deficiencies on chromosome stability.

## BACKGROUND

Chromosome stability is crucial for the proper functioning and survival of cells. Double-strand breaks (DSBs) in DNA are a major form of chromosomal instability that can lead to various genetic disorders and cancer. Understanding the mechanisms that cause DSBs and identifying the sites of these breaks is essential for developing strategies to prevent and treat such conditions.

Replication stress, often caused by nucleotide depletion or checkpoint deficiencies, can lead to the formation of single-stranded DNA (ssDNA) at replication forks. This ssDNA can subsequently become a site of DSBs, leading to chromosome fragility. While several methods exist for mapping DSBs, such as ChIP-chip, ChIP-Seq, and γ-H2AX foci detection, these methods have limitations in terms of specificity and accuracy, especially when dealing with in vitro-generated breaks.

The present invention addresses these limitations by providing a novel method for genome-wide mapping of DSBs in yeast cells. This method involves preparing genomic DNA from yeast cells embedded in agarose plugs, labeling the DNA ends in gel, and cohybridizing the labeled DNA to microarrays. The method is highly specific and minimizes the generation of in vitro breaks, making it a valuable tool for studying chromosome fragility in various genetic backgrounds and under different conditions of replication stress.

## SUMMARY

The present invention provides a method for genome-wide mapping of double-strand breaks (DSBs) in yeast cells. The method includes the following steps:

1. Preparing genomic DNA from yeast cells embedded in agarose plugs.
2. Labeling the DNA ends in gel using a labeling kit that incorporates Cy-conjugated dUTP at DNA ends.
3. Eluting the labeled DNA from the agarose plugs.
4. Fragmenting the eluted DNA to reduce the average fragment size.
5. Co-hybridizing the labeled DNA to microarrays.
6. Analyzing the microarray data to identify the sites of DSBs.

The invention also provides a method for identifying the correlation between the sites of DSBs and the locations of single-stranded DNA (ssDNA) in yeast cells. This method involves:

1. Mapping ssDNA in yeast cells using random-primed labeling without denaturation of genomic template DNA.
2. Comparing the ssDNA profiles with the DSB profiles to identify the correlation between the two.

The invention further provides a method for validating the identified DSB sites using indirect end-labeling techniques.

The methods of the present invention are particularly useful for studying the effects of replication stress and checkpoint deficiencies on chromosome stability. The invention can be applied to various genetic backgrounds and conditions, including but not limited to mec1 mutants, rad53 mutants, and temperature-sensitive mec1 alleles.

## DETAILED DESCRIPTION

### Preparation of Genomic DNA

Genomic DNA is prepared from yeast cells embedded in agarose plugs. The yeast cells are first synchronized in G1 phase using the mating pheromone alpha factor. The cells are then released into S phase in the presence or absence of hydroxyurea (HU) to induce replication stress. Samples are collected at various time points during HU exposure and recovery from HU.

The cells are embedded in agarose plugs and processed for cell wall disruption and protein degradation. The agarose plugs are pre-equilibrated in 10 mM Tris-HCl (pH 8.0) and 0.1 mM EDTA, followed by equilibration in 50 mM Tris-HCl (pH 6.8), 5 mM MgCl2, and 10 mM β-mercaptoethanol. The agarose plugs are then used for in-gel labeling of ssDNA or DSBs.

### In-Gel Labeling of ssDNA

For in-gel labeling of ssDNA, the agarose plugs are pre-equilibrated in 10 mM Tris-HCl (pH 8.0) and 0.1 mM EDTA, followed by equilibration in 50 mM Tris-HCl (pH 6.8), 5 mM MgCl2, and 10 mM β-mercaptoethanol. The labeling mix, containing 50 mM Tris-HCl (pH 6.8), 5 mM MgCl2, 10 mM β-mercaptoethanol, 0.24 mM of each of dATP, dCTP, and dGTP, 0.12 mM of dTTP, 0.12 mM Cy5 or Cy3-dUTP, 250 μg/ml random hexamers, and 150 units of Klenow (exonuclease deficient), is added to the agarose plugs. The plugs are incubated at 37°C in the dark for 2 hours. The labeled DNA is then electroeluted from the agarose plugs and purified using standard procedures.

### In-Gel Labeling of DSBs

For in-gel labeling of DSBs, the agarose plugs are pre-equilibrated in 10 mM Tris-HCl (pH 8.0) and 0.1 mM EDTA, followed by equilibration in 1× End-Repair buffer (Epicentre). The End-Repair labeling mix, containing 1× End-Repair buffer, 1 mM ATP, 0.24 mM of each of dATP, dCTP, and dGTP, 0.12 mM of dTTP, 0.12 mM Cy5 or Cy3-dUTP, and 3 μl of End-Repair enzyme mix, is added to the agarose plugs. The plugs are incubated at room temperature in the dark for 1 hour. The labeled DNA is then electroeluted from the agarose plugs and purified using standard procedures.

### Microarray Analysis

The labeled DNA from the experimental and control samples is cohybridized to Agilent G4493A yeast 4x44K ChIP to chip DNA microarrays. Data extraction is performed using Agilent’s Feature Extraction software. The ratio of background-subtracted fluorescent signals from the experimental to the control sample is calculated for each probe. The resulting ratios for all the probe locations on each chromosome are normalized to the total amount of signals in each fluorescent channel and smoothed with a 6 kb window using a Lowess smoothing algorithm.

### Validation of DSB Sites

To validate the identified DSB sites, indirect end-labeling techniques are used. DNA samples are prepared by spheroplasting after cells are embedded in agarose as described above, differentially end-labeled with Cy-dyes, and cohybridized to the microarray. The relative amount of DSBs in the experimental sample is quantified as the ratio of fluorescent signal from the experimental sample to that from the control.

### Correlation Between ssDNA and DSB Sites

The ssDNA and DSB profiles are compared to identify the correlation between the two. The correlation is quantified by calculating the correlation coefficient between the ssDNA and DSB profiles. The significance of the correlation is assessed using random simulation tests.

### Application to Different Genetic Backgrounds

The methods of the present invention can be applied to various genetic backgrounds, including but not limited to mec1 mutants, rad53 mutants, and temperature-sensitive mec1 alleles. The methods are particularly useful for studying the effects of replication stress and checkpoint deficiencies on chromosome stability.

## EXAMPLES

### Example 1

#### Mapping DSBs in mec1 Cells Exposed to HU

1. **Preparation of Genomic DNA**: Yeast cells (mec1-1) were synchronized in G1 phase using alpha factor and released into S phase in the presence of 200 mM HU. Samples were collected at 0 hours (HU 0hr) and 1 hour (HU 1hr) after HU addition. The cells were then allowed to recover in fresh medium without HU for 1 hour (R 1hr). Genomic DNA was prepared from the samples and embedded in agarose plugs.

2. **In-Gel Labeling of DSBs**: The agarose plugs were pre-equilibrated and labeled with Cy5 or Cy3-dUTP as described in the detailed description. The labeled DNA was electroeluted and purified.

3. **Microarray Analysis**: The labeled DNA from the HU 1hr and R 1hr samples was cohybridized to Agilent microarrays. Data extraction and analysis were performed as described in the detailed description.

4. **Results**: The DSB profiles of the HU 1hr and R 1hr samples were compared. The R 1hr sample showed significant DSBs at multiple sites in the genome, while the HU 1hr sample did not. The breakage sites were correlated with the locations of ssDNA near checked origins of replication.

### Example 2

#### Mapping DSBs in mec1-4 Temperature-Sensitive Mutant

1. **Preparation of Genomic DNA**: Yeast cells (mec1-4) were synchronized in G1 phase using alpha factor and released into S phase at the restrictive temperature (37°C). Samples were collected at 40 minutes (mid-S phase) after release. Genomic DNA was prepared from the samples and embedded in agarose plugs.

2. **In-Gel Labeling of ssDNA and DSBs**: The agarose plugs were pre-equilibrated and labeled with Cy5 or Cy3-dUTP as described in the detailed description. The labeled DNA was electroeluted and purified.

3. **Microarray Analysis**: The labeled DNA from the mid-S phase sample was cohybridized to Agilent microarrays. Data extraction and analysis were performed as described in the detailed description.

4. **Results**: The ssDNA profile of the mid-S phase sample showed distinct patterns of elevated levels of ssDNA near replication termini. The DSB profile of the same sample showed significant DSBs at the same regions, confirming that ssDNA formation is a precursor to chromosome breakage in the mec1-4 mutant.

These examples demonstrate the utility of the methods of the present invention for mapping and identifying DSBs in yeast cells under various conditions of replication stress and checkpoint deficiencies. The methods provide a powerful tool for studying chromosome stability and the mechanisms underlying chromosome fragility.