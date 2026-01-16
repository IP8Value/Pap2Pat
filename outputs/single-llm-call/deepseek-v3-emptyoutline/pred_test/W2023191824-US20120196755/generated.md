Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FEDERALLY SPONSORED RESEARCH  

The present invention was made without federal funding or government support. All rights are retained by the inventors and assignees.  

## TECHNICAL FIELD  

The present invention relates to the field of molecular biology and genetics, specifically to methods for genome-wide mapping of chromosome breakage and single-stranded DNA (ssDNA) formation. More particularly, the invention provides novel techniques for identifying sites of chromosomal fragility and replication stress through in-gel labeling methodologies coupled with microarray analysis. These methods enable high-resolution detection of double-strand breaks (DSBs) and ssDNA regions, which are critical for understanding replication fork instability and genome maintenance mechanisms.  

## BACKGROUND  

Chromosomal instability is a hallmark of many diseases, including cancer, and is often associated with defects in DNA replication and repair. Replication stress, induced by factors such as hydroxyurea (HU) or other replication inhibitors, leads to the stalling and collapse of replication forks, resulting in the formation of ssDNA and DSBs. Traditional methods for detecting DSBs, such as γ-H2AX staining or Spo11 binding assays, suffer from limitations including false positives and inability to directly label terminal DSBs. Similarly, existing ssDNA mapping techniques lack the resolution and specificity needed to accurately correlate ssDNA formation with subsequent chromosomal breakage.  

Prior art methods for genome-wide break mapping rely on indirect detection strategies that do not label DSBs directly, leading to potential inaccuracies. Furthermore, these methods are often confounded by in vitro artifacts introduced during DNA isolation and processing. There exists a pressing need for a robust, high-resolution methodology that can directly label and map DSBs and ssDNA in a genome-wide manner, enabling precise identification of fragile sites and replication stress responses.  

## SUMMARY  

The present invention provides novel methods for genome-wide mapping of chromosome breakage and ssDNA formation, addressing the limitations of existing techniques. The invention comprises an in-gel labeling approach that minimizes in vitro artifacts by processing DNA within agarose plugs, followed by differential labeling of experimental and control samples using fluorescently conjugated nucleotides. The labeled DNA is then cohybridized to microarrays, allowing for high-resolution detection of breakage sites and ssDNA regions.  

Key aspects of the invention include:  
1. **In-gel ssDNA labeling**: A method for labeling ssDNA gaps within agarose-embedded DNA using exonuclease-deficient polymerases, enabling precise mapping of replication fork-associated ssDNA.  
2. **In-gel DSB labeling**: A technique for end-labeling DSBs within agarose plugs using end-repair enzymes, facilitating direct detection of chromosomal breakage sites.  
3. **Microarray-based analysis**: Integration of labeled DNA samples with microarray hybridization and data smoothing algorithms to generate high-resolution breakage and ssDNA profiles.  

The invention further demonstrates that ssDNA formation precedes chromosomal breakage, providing a predictive tool for identifying fragile genomic regions. Applications include research in cancer biology, genotoxic stress responses, and therapeutic development targeting replication stress.  

## DETAILED DESCRIPTION  

The invention provides a comprehensive methodology for genome-wide mapping of chromosomal breakage and ssDNA formation, as detailed below.  

### In-Gel ssDNA Labeling  

The method involves embedding cells in agarose plugs, followed by spheroplasting to release genomic DNA while minimizing mechanical shearing. The agarose-embedded DNA is equilibrated in labeling buffer, and ssDNA gaps are labeled using an exonuclease-deficient polymerase (e.g., Klenow or Sequenase) in the presence of fluorescently conjugated dUTP (e.g., Cy3 or Cy5). The labeling reaction is performed in-gel to prevent artifactual ssDNA generation. After labeling, DNA is electroeluted from the agarose, fragmented by sonication, and purified for microarray hybridization.  

### In-Gel DSB Labeling  

For DSB mapping, genomic DNA is similarly embedded in agarose plugs and processed to minimize in vitro breakage. DSB ends are labeled in-gel using an end-repair enzyme mix (e.g., T4 DNA polymerase and polynucleotide kinase) with fluorescent dUTP incorporation. This method efficiently labels both 3′ and 5′ overhangs, enabling direct detection of DSBs. Labeled DNA is eluted, fragmented, and cohybridized with control DNA to microarrays.  

### Microarray Analysis  

Experimental and control DNA samples are differentially labeled and cohybridized to high-density microarrays (e.g., Agilent yeast 4x44K arrays). Fluorescence ratios are calculated for each probe, and data is smoothed using a Lowess algorithm with a 6 kb window. Peaks in the breakage or ssDNA profiles are identified statistically, and chromosomal coordinates of significant sites are mapped.  

### Applications and Validation  

The invention has been validated using:  
1. **Restriction enzyme-digested DNA**: BamHI and FspI digestion confirmed precise mapping of known cleavage sites.  
2. **HO endonuclease-induced breaks**: Detection of HO cut sites demonstrated method specificity for in vivo DSBs.  
3. **Replication stress models**: ssDNA and breakage mapping in mec1 mutants exposed to HU revealed correlations between fork progression and chromosomal fragility.  

## EXAMPLES  

### Example 1  

**Genome-wide ssDNA Mapping in mec1 Mutants Exposed to HU**  

Mec1 cells were synchronized in G1 with alpha factor, released into S phase in the presence of 200 mM HU, and sampled at 0 and 1 hr. Cells were embedded in agarose, and ssDNA was labeled in-gel with Cy5-dUTP using Klenow polymerase. Control G1 DNA was labeled with Cy3-dUTP. Labeled DNA was hybridized to microarrays, and ssDNA profiles showed persistent gaps near replication origins after HU removal, correlating with subsequent breakage sites.  

### Example 2  

**Chromosome Breakage Mapping During Recovery from HU**  

Mec1 cells were treated with HU for 1 hr, followed by recovery in fresh medium. DNA was labeled in-gel using the end-repair method with Cy5-dUTP, while control DNA was labeled with Cy3. Microarray analysis revealed breakage sites near checked (late-firing) origins, with significant correlation to prior ssDNA profiles (R = 0.64). Breakage was absent in wild-type cells, confirming checkpoint dependence.  

The invention thus provides a powerful tool for studying replication stress and chromosomal instability, with broad applications in research and therapeutics.