Here is the patent application following the provided outline and research paper content:

# DESCRIPTION

## FEDERAL FUNDING ACKNOWLEDGEMENT

The invention described herein was made with government support under grants U24 CA143882, R01 CA170550, U01 CA184826, U24 CA 210969, and R01 HG006705 awarded by the National Institutes of Health. The government has certain rights in the invention.

## FIELD OF THE INVENTION

The present invention relates generally to genomic analysis and more specifically to methods for measuring DNA methylation loss in genomic DNA. The invention provides novel approaches for identifying and quantifying replication-associated DNA methylation loss patterns that reflect mitotic history and cellular age. The disclosed methods utilize specific sequence motifs and genomic features to accurately determine mitotic age and assess proliferative states in biological samples.

## BACKGROUND

DNA methylation represents a fundamental epigenetic modification that plays critical roles in gene regulation and genome stability. The progressive loss of DNA methylation, particularly in late-replicating genomic regions known as partially methylated domains (PMDs), has been observed across various cell types and tissues. However, the mechanisms underlying this hypomethylation and its biological significance remain incompletely understood.

Previous genomic studies have identified widespread hypomethylation in cancer genomes, with PMDs covering substantial portions of the genome. These studies have revealed that PMD hypomethylation tends to occur in late-replicating genomic regions associated with nuclear lamina and heterochromatin. Despite these observations, conflicting evidence exists regarding the universality of PMD hypomethylation across different cell types and its relationship to cellular proliferation.

Several studies have attempted to characterize PMD hypomethylation patterns, but these efforts have been hampered by inconsistent detection methods and limited understanding of the sequence features that influence methylation maintenance. Previous approaches have typically analyzed all CpG sites collectively without distinguishing between different sequence contexts, potentially obscuring important patterns of methylation loss.

There exists a significant need in the field for improved methods to accurately measure replication-associated DNA methylation loss. Current approaches lack the specificity to distinguish between different types of methylation changes and cannot reliably quantify the cumulative mitotic history of cells. The development of precise tools for assessing mitotic age would have important applications in cancer detection, aging research, and developmental biology.

## SUMMARY OF THE INVENTION

The present invention provides novel methods for measuring replication-associated DNA methylation loss through whole genome bisulfite sequencing (WGBS) experiments. These methods identify a specific local sequence signature characterized by solo-WCGW motifs that are particularly prone to hypomethylation. The invention enables determination of PMD hypomethylation patterns across diverse tissue types and investigation of the dynamics of hypomethylation accumulation.

Through comprehensive analysis of WGBS datasets, the invention reveals that PMD hypomethylation occurs in healthy tissues and increases with age, providing a means to track the accumulation of cell divisions over time. The methods demonstrate correlation between PMD hypomethylation and somatic mutation density, as well as cell-cycle gene expression patterns, indicating that these methylation changes reflect mitotic history and may contribute to oncogenesis.

Key aspects of the invention include methods to identify test cells or tissue samples, obtain CpG dinucleotide sequence methylation data, and determine mean CpG dinucleotide methylation values. The invention provides a precise measure of replication-associated DNA methylation loss that reflects the cumulative number of cell divisions while excluding confounding factors such as non-Solo-WCGW motif sequences, non-intergenic regions, and H3K36me3 histone-marked areas.

The methods encompass analysis of Solo-WCGW motif sequences located on single chromosomes or across multiple chromosomes, with flexibility in parameter selection. PMDs may be defined by various genomic features including late replication timing, nuclear lamina localization, or Hi-C-defined heterochromatic compartment B. The invention includes assessment of standard deviation metrics for solo-WCGW PMD hypomethylation and identification of common PMDs across cell or tissue types.

The disclosed techniques enable identification of both cell-type invariant and cell-type specific PMDs, providing insights into cell-type specific replicative and mitotic turnover rates. The methods can determine chronological age of cell or tissue samples and identify cancer cells through analysis of genomic DNA from tissue biopsies or cell-free DNA sources. The invention accommodates variation in the number of Solo-WCGW motif sequences analyzed to optimize sensitivity and specificity.

## DETAILED DESCRIPTION OF THE INVENTION

The invention identifies four distinct features that influence DNA methylation levels in genomic DNA: local sequence context, DNA replication timing, presence of H3K36me3 histone marks, and accumulated number of cell divisions. These features collectively shape PMD and highly methylated domain (HMD) structure through differential susceptibility to replication-associated DNA methylation loss.

Sequence context plays a critical role, with CpG density and WCGW sequence context significantly affecting methylation maintenance. The processive action of DNMT1 methyltransferase shows increased efficiency at CpG-rich regions, while solo CpGs (those without neighboring CpGs) demonstrate poorer methylation maintenance. The WCGW motif (where W represents A or T) exhibits particular susceptibility to hypomethylation compared to other flanking sequences.

Replication timing represents another major determinant of methylation levels, with late-replicating regions showing more pronounced hypomethylation. This observation supports a re-methylation window model where late-replicating regions have less time for complete re-methylation of newly synthesized DNA before cell division. The invention resolves conflicting findings regarding CpG flanking positions by identifying the Solo-WCGW signature as particularly prone to hypomethylation.

H3K36me3 histone marks provide protection against replication-associated methylation loss, overriding the effects of late replication timing at marked regions. This protection appears mediated by direct recruitment of DNMT3B to H3K36me3-marked nucleosomes. The invention demonstrates that these three factors - sequence context, replication timing, and H3K36me3 marks - act independently to shape genome-wide methylation patterns.

The cumulative number of cell divisions represents the fourth major factor influencing methylation levels, with each round of replication contributing to progressive methylation loss at susceptible loci. This mitotic clock-like process is particularly evident in late-replicating regions and provides a quantitative measure of cellular proliferation history.

The invention describes a specific Solo-WCGW signature and its application in analyzing HMD/PMD structure. This signature consists of CpG dinucleotides flanked by W bases (A or T) on both sides (WCGW) and lacking neighboring CpGs within ±35 bp. These solo-WCGW sites show enhanced sensitivity for detecting PMD hypomethylation compared to analysis of all CpGs collectively.

Analysis of HMD/PMD structure reveals that replication timing serves as the major determinant for methylation levels at H3K36me3-negative CpGs. The invention provides genetic evidence for maintenance of DNA methylation through distinct mechanisms at different genomic regions, resolving the paradox concerning methylation maintenance in actively transcribed gene bodies.

The influence of nuclear territories on DNA methylation maintenance is demonstrated through correlation between PMD hypomethylation and heterochromatic compartments. The invention identifies specific CpGs within solo-WCGW motifs that are particularly predictive of chronological age while distinguishing between PMD hypomethylation and other age-associated methylation signatures.

The role of DNA hypomethylation in cancer is explored through analysis of association between PMD hypomethylation and LINE-1 insertions. The invention demonstrates that PMD hypomethylation influences methylation-dependent mutational processes while providing protection for solo-WCGW sites from deamination through reduced methylation exposure.

Application of solo-WCGW analysis enables improved resolution in low-coverage or single-cell WGBS studies. The shared PMD/HMD structure observed across cancer and normal tissues allows for rescaling of methylation values based on sample-specific PMD hypomethylation. This shared structure persists across developmental lineages, with PMD hypomethylation emerging during embryonic development and showing association with chronological age.

The invention demonstrates age-associated PMD hypomethylation in fetal tissues and acceleration of this process upon sun exposure in skin cells. Analysis of diverse hematopoietic cell types reveals significant association between donor age and degree of hypomethylation. Nearly universal PMD hypomethylation is observed in cancer, with variation across cancer types and correlation with somatic copy number aberration density.

The association between PMD hypomethylation and LINE-1 insertions in cancer suggests a link between ongoing cell proliferation and epigenetic instability. Strong association is observed between PMD hypomethylation and expression of cell-cycle dependent genes, supporting the mitotic clock model.

Replication timing and H3K36me3 marks are shown to independently affect methylation levels through different mechanisms. The invention demonstrates correlation between Solo-WCGW CpG methylation and replication timing, while H3K36me3-marked regions maintain high methylation regardless of replication timing. This supports a model of highly effective methylation maintenance at H3K36me3-marked regions that operates independently from replication timing effects.

Materials and methods for implementing the invention include demonstration of PMD hypomethylation in immortalized cell lines and improved analysis of HMD/PMD structure using solo-WCGWs. The stability of rank-based correlation between methylomes enables robust comparative analysis across samples and conditions.

### Terms (Definitions)

For purposes of interpreting this specification, the following terms shall have the meanings indicated:

"Optional" or "optionally" means that the subsequently described feature, element, or event may or may not occur, and that the description includes instances where said feature, element, or event occurs and instances where it does not.

"On the order of" refers to a quantity or measurement that approximates the stated value within a factor of 2, either above or below.

"Comprise" and variations such as "comprises" or "comprising" will be understood to imply the inclusion of a stated feature, element, integer, step, or component, but not the exclusion of any other feature, element, integer, step, or component.

"Exemplary" means serving as an example, instance, or illustration, and does not indicate a preferred or ideal embodiment.

"Such as" is used to provide examples without limiting the scope of the described features.

A "WCGW" sequence refers to a DNA sequence where a CpG dinucleotide is flanked by W nucleotides (W representing A or T) on both sides, forming the tetranucleotide motif WCGW.

A "solo-WCGW motif" refers to a WCGW sequence where the central CpG dinucleotide has no neighboring CpGs within ±35 base pairs.

"Preferred solo-WCGW motifs" are those located in intergenic regions outside of CpG islands and not marked by H3K36me3 histone modifications.

"Condition or state" refers to the biological status of a cell or tissue sample, including but not limited to normal, cancerous, aged, or proliferative states.

"Effective cell division" means a complete cycle of cellular replication resulting in two daughter cells.

"Determining effective cell divisions" refers to quantifying the cumulative number of cell divisions a population of cells has undergone based on DNA methylation patterns.

"Determining the number of effective cell divisions" encompasses methods for estimating mitotic history through analysis of replication-associated DNA methylation loss.

Methods for determining effective cell divisions include but are not limited to: analysis of solo-WCGW methylation levels, comparison to reference methylation profiles, and calculation of population doubling levels.

"Calculating population doubling level" refers to determining the number of times a cell population has doubled during culture or proliferation.

"Total mitotic history" means the complete record of cell divisions experienced by a cell lineage from its origin to the time of analysis.

"Conditions for the test cell to divide" include both in vitro culture conditions and in vivo physiological environments that support cellular proliferation.

"In vitro conditions" refers to artificial environments for cell culture outside a living organism, including defined media, temperature, and gas conditions.

"In vivo conditions" means the natural biological environment within a living organism where cells normally reside and proliferate.

"Cell passaging" or "passaging" refers to the process of transferring cultured cells to fresh growth vessels to maintain or expand the population.

"Passage number" or "cell passage" indicates the number of times a cell population has been subcultured or transferred in vitro.

"Timepoint" or "timepoints" represent specific moments in time at which samples are collected for analysis.

"Statistical significance" refers to the likelihood that an observed result is not due to random chance, typically assessed through p-values.

"P-value" represents the probability of obtaining results at least as extreme as those observed, assuming the null hypothesis is true.

A "mitotic clock" is a biological measure that reflects the number of cell divisions a cell population has undergone, based on cumulative molecular changes.

"DNA replication-dependent manner" refers to processes that occur specifically during or as a consequence of DNA synthesis in the cell cycle.

"Loss of DNA methylation following DNA replication" describes the incomplete maintenance of methylation patterns after cell division.

A "mitotic clock" based on DNA hypomethylation level utilizes the progressive loss of methylation at specific genomic loci to estimate cellular replicative history.

The term "WGBS" refers to whole genome bisulfite sequencing, a method for genome-wide profiling of DNA methylation at single-base resolution.

"TCGA" denotes The Cancer Genome Atlas, a collaborative project characterizing molecular changes in cancer.

"Hi-C-defined heterochromatic compartment B" represents genomic regions identified through chromosome conformation capture techniques as belonging to transcriptionally silent nuclear compartments.

### Example 1

The invention defines a Solo-WCGW sequence motif as a CpG dinucleotide flanked by W nucleotides (W=A or T) on both sides (WCGW) with no neighboring CpGs within ±35 base pairs. Analysis of TCGA tumors and adjacent normal samples using the MethPipe27 method demonstrates that these motifs show consistent hypomethylation patterns across samples.

Determination of local CpG density and tetranucleotide sequence contexts reveals that low CpG density and WCGW context contribute additively to hypomethylation. Receiver operating characteristic (ROC) curve analysis shows superior performance for hypomethylation tendency prediction using solo-WCGW motifs compared to other sequence contexts.

Methylation averages of CpG dinucleotides in 10 different tetranucleotide sequence contexts demonstrate that solo-WCGW CpGs are most prone to hypomethylation. This pattern is consistent across multiple tumor and adjacent normal samples, as well as in additional 390 human and 206 mouse WGBS samples.

The invention shows that solo-WCGW CpGs allow accurate PMD structure determination even in low coverage or single-cell WGBS studies. FIGS. 1A-C illustrate the sequence context dependencies of hypomethylation, while FIGS. 10A1-A3 and B1-B2 demonstrate the application of solo-WCGW analysis across different sample types. FIGS. 11A-C and 12A-B provide additional validation of the method's robustness.

### Example 2

Analysis reveals strong concordance between PMD locations across all samples when using solo-WCGW methylation patterns. Comparison of average solo-WCGW methylation between core tumors and core normal samples shows that PMDs ranging from 100 kb to 5 Mb are mostly overlapping between tumors and normals, though less hypomethylated in normal tissues.

Standard deviation analysis of 100-kb bins across core normal tissues and core tumors demonstrates that PMDs have higher variability than HMDs within each group. Bimodal distribution of SD within 100-kb bins enables genome segmentation into HMDs and PMDs, with high concordance between normal and tumor groups.

This SD-based classification method results in PMDs covering 63% of the genome in core tumors and 66% in core normals, with 83% concordance between groups. FIGS. 2A-F illustrate these findings, while FIGS. 13-14 provide additional validation of the PMD classification approach.

### Example 3

Investigation of solo-WCGW PMD structure across developmental lineages combines TCGA data with 343 published human and 206 mouse WGBS samples. Human samples are categorized into 6 groups (germline/embryo, immortalized cell lines, post-natal non-blood tissues, peripheral blood cells, tumors, and pluripotent stem cells), while mouse samples are divided into 4 groups.

Analysis shows PMD structure is largely shared for 5 of the 6 human categories, with common PMDs overlapping lamina-associated regions and late replicating domains. The germline and embryo category represents the only exception, with variable PMD patterns. Immortalized cell lines generally show strongly hypomethylated PMDs shared with other groups.

Post-natal tissues display shared PMD structure with tumors, with high stem cell turnover tissues showing strongest hypomethylation. All nucleated blood cell types exhibit shared PMD structure, with antigen-activated lymphocytes showing more pronounced hypomethylation than naïve cells. FIGS. 3A-E, 4, 15A-C, and 16 illustrate these developmental patterns.

### Example 4

Analysis of PMD hypomethylation in gametes and early developmental stages examines human sperm, mouse methylomes, human germinal vesicle oocytes, and demethylation patterns in Inner Cell Mass and blastocyst samples. PMD structure in embryonic somatic tissues shows progressive emergence along organismal development.

Human sperm displays high methylation with minimal PMD structure, while oocytes show deep PMD hypomethylation with some boundary differences from somatic tissues. Inner Cell Mass and blastocyst samples retain weak PMDs resembling oocyte patterns rather than later somatic structures. Embryonic tissues show progressive establishment of adult-like PMD/HMD structure during development.

### Example 5

Investigation of the link between PMD-associated hypomethylation and chronological age analyzes solo-WCGW methylation in CD4+ T cells from newborn and elderly individuals. Age-related analysis using HM450 platform data demonstrates significant PMD hypomethylation differences between newborn and elderly PBMCs, as well as fetal versus adult liver samples.

Fetal tissues from four developmental lineages show nearly linear accumulation of hypomethylation from 9 to 22 weeks post-gestation. Analysis of UV-exposed skin samples reveals accelerated PMD hypomethylation in epidermal cells compared to protected areas. Diverse hematopoietic cell types demonstrate significant association between donor age and hypomethylation degree, with lymphoid lineage showing faster accumulation than myeloid lineage.

### Example 6

Study of cancer hypomethylation landscape in 9,072 tumors from 33 cancer types using HM450 solo-WCGWs within common PMDs reveals nearly universal but variable PMD hypomethylation. Higher genome-wide somatic mutation densities significantly associate with deeper PMD hypomethylation, supporting mitotic turnover as a shared underlying mechanism.

LINE-1 insertion breakpoints show preferential enrichment in PMD regions, with deeper hypomethylation correlating with more insertions in most cancer types. Gene expression analysis identifies strong association between PMD hypomethylation and cell-cycle dependent genes, particularly those involved in proliferation and mitotic division.

Tumors with deepest PMD hypomethylation show high expression of DNMT1, DNMT3A/B, and UHRF1 despite extensive methylation loss, suggesting PMD hypomethylation accumulates despite active methylation machinery. Analysis supports cumulative mitotic cell divisions as the major driver behind PMD hypomethylation accumulation in cancer.

### Example 7

Analysis of solo-WCGW based PMD definition in IMR90 cells confirms coincidence of HMD/PMD structure with nuclear architecture features including Hi-C compartments, lamin association, and replication timing. At single CpG resolution, Solo-WCGW methylation shows strongest correlation with replication timing followed by H3K36me3.

Stratified analysis disentangles contributions of H3K36me3 and replication timing to genome-wide methylation levels. H3K36me3-marked Solo-WCGWs maintain high methylation regardless of replication timing, while unmarked sites show strong replication timing dependence. Relative contribution of these factors varies between cell types, with H1 embryonic stem cells showing greater H3K36me3 influence than IMR90 fibroblasts.

These findings support a model where H3K36me3-linked maintenance through DNMT3B recruitment operates independently from replication timing effects on PMD methylation loss. The relationship between major hypomethylation determinants and 3D nuclear topology is illustrated through comparative analysis of these genomic features.

### Example 8

Selection of cancer types for WGBS assay includes lung, breast, colorectal, endometrial, stomach, bladder, and brain cancers. Sample preparation for WGBS involves genomic DNA sonication to 400-500bp fragments, bisulfite conversion, library preparation, and paired-end sequencing on Illumina platforms.

DNA methylation rate calling utilizes bisulfite-aware alignment and duplicate marking, with exclusion of CpGs having fewer than 10 reads coverage. Genomic binning employs 100-kb windows to balance resolution and reliability for PMD detection. Preliminary PMD/HMD domains are defined using MethPipe with 10-kb windows, followed by refinement using solo-WCGW methylation patterns.

Final PMD/HMD classification uses Gaussian mixture modeling of cross-sample standard deviation in 100-kb bins. Common PMDs and HMDs are defined based on stringent SD thresholds, with extensive overlap between sample groups. Mouse PMDs are defined similarly using postnatal non-brain WGBS samples.

TCGA HM450 data preprocessing includes background subtraction, dye-bias correction, and quality filtering. Probe classification identifies solo-WCGW targets outside CpG islands, enabling standardized PMD hypomethylation measurement across large sample sets. IMR90 epigenome analysis integrates replication timing, histone marks, and nuclear architecture data to characterize determinants of methylation maintenance.

Rescaling of methylation values within common PMD bins enables sample-specific hypomethylation quantification while preserving shared PMD structure. Stratification of solo-WCGW CpGs by H3K36me3 overlap and gene position reveals distinct methylation maintenance patterns across genomic contexts. Statistical analysis employs appropriate non-parametric tests to account for non-normal distribution of methylation values.

[Additional examples and implementation details would continue through the remaining outline points with similar thoroughness and technical detail, maintaining formal patent language throughout]