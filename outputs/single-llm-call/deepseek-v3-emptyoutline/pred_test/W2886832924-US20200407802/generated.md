Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FEDERAL FUNDING ACKNOWLEDGEMENT  
The invention described herein was made with government support under grant numbers U24 CA143882, R01 CA170550, U01 CA184826, U24 CA 210969, and R01 HG006705 awarded by the National Institutes of Health. The government has certain rights in the invention.  

## FIELD OF THE INVENTION  
The present invention relates generally to the field of epigenetics and genomic analysis. More specifically, the invention pertains to methods for identifying and analyzing partially methylated domains (PMDs) in genomic DNA, particularly through the characterization of solo-WCGW CpG sites that exhibit differential methylation patterns associated with cell division history, developmental stage, and disease states such as cancer.  

## BACKGROUND  
DNA methylation represents a critical epigenetic modification that regulates gene expression and maintains genomic stability. While methylation patterns have been extensively studied in promoter regions and CpG islands, large-scale methylation changes across broad genomic domains remain poorly understood. Previous studies have identified partially methylated domains (PMDs) as regions of the genome exhibiting reduced methylation levels, particularly in cancer cells and certain normal tissues. However, existing methods for PMD detection suffer from limitations in sensitivity, specificity, and applicability across different sample types and sequencing depths.  

Current approaches to analyzing DNA methylation patterns typically examine all CpG sites collectively, without distinguishing between different sequence contexts that may influence methylation maintenance. This lack of discrimination leads to reduced sensitivity in detecting PMD structure, particularly in samples with subtle hypomethylation patterns or when working with low-coverage sequencing data. Furthermore, existing methods fail to adequately account for the relationship between PMD hypomethylation and cell division history, limiting their utility as biomarkers of cellular aging and proliferative history.  

There exists an unmet need for improved methods of PMD detection and analysis that provide enhanced sensitivity across diverse biological samples, enable accurate assessment of methylation patterns even with limited sequencing data, and facilitate the correlation of PMD hypomethylation with critical biological processes including development, aging, and disease progression. The present invention addresses these needs through novel approaches focusing on specific CpG sequence contexts that serve as optimal markers for PMD analysis.  

## SUMMARY OF THE INVENTION  
The present invention provides novel methods and systems for analyzing DNA methylation patterns, particularly focusing on the identification and characterization of partially methylated domains (PMDs) through examination of specific CpG sequence contexts. The invention is based on the discovery that CpG sites with particular sequence features - specifically those with low local CpG density and flanked by A:T bases (WCGW context) - show differential susceptibility to methylation loss and serve as superior markers for PMD detection.  

Key aspects of the invention include:  
1. Methods for identifying PMDs by analyzing methylation patterns at solo-WCGW CpG sites, defined as WCGW-context CpGs with no neighboring CpGs within ±35 base pairs.  
2. Techniques for classifying genomic regions as PMDs or highly methylated domains (HMDs) based on the standard deviation of methylation levels at solo-WCGW CpGs across multiple samples.  
3. Applications of solo-WCGW CpG analysis for detecting PMD structure in low-coverage sequencing data, including single-cell whole genome bisulfite sequencing (scWGBS).  
4. Methods for correlating PMD hypomethylation patterns with biological parameters including cell division history, developmental stage, and disease states.  
5. Systems for identifying cancer-associated methylation changes by comparing PMD hypomethylation between tumor and normal samples.  

The invention provides significant advantages over existing methods, including enhanced sensitivity for PMD detection, applicability to diverse sample types and sequencing depths, and the ability to quantify methylation changes associated with critical biological processes. These improvements enable novel applications in cancer detection, aging research, developmental biology, and epigenetic biomarker discovery.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Terms (Definitions)  
For purposes of this invention, the following terms shall have the meanings specified:  

"Partially Methylated Domain (PMD)" refers to a genomic region spanning at least 100 kilobases that exhibits reduced DNA methylation levels compared to other genomic regions in the same sample, typically showing average methylation below 70% in cancer samples.  

"Highly Methylated Domain (HMD)" refers to a genomic region spanning at least 100 kilobases that maintains high DNA methylation levels, typically showing average methylation above 70% in cancer samples.  

"Solo-WCGW CpG" denotes a cytosine-phosphate-guanine (CpG) dinucleotide that meets two criteria: (1) it is flanked by adenine or thymine bases on both sides in the sequence context WCGW, where W represents A or T; and (2) it has no other CpG dinucleotides within ±35 base pairs in the genomic sequence.  

"Replication timing" refers to the temporal order in which genomic regions are replicated during S phase of the cell cycle, typically classified as early, mid, or late replicating regions.  

"H3K36me3" denotes trimethylation of lysine 36 on histone H3, a chromatin modification associated with actively transcribed gene bodies.  

### Example 1  
**Identification of Solo-WCGW CpGs as Superior Markers for PMD Detection**  

The invention provides methods for identifying PMDs through analysis of methylation patterns at solo-WCGW CpG sites. In representative embodiments, whole genome bisulfite sequencing (WGBS) data is analyzed from a set of tumor and normal samples. Using this approach, methylation levels are determined for all CpG sites in the genome, with particular attention to sequence context and local CpG density.  

Analysis reveals that CpG sites with the WCGW sequence context (flanked by A/T bases) show significantly greater hypomethylation in PMDs compared to other sequence contexts. Furthermore, WCGW CpGs with no neighboring CpGs within ±35 base pairs ("solo-WCGW") exhibit the most pronounced hypomethylation in PMDs. These solo-WCGW CpGs represent approximately 13% of all CpGs in the human genome and serve as highly sensitive markers for PMD detection.  

The enhanced sensitivity of solo-WCGW CpGs for PMD detection is demonstrated through comparative analysis of tumor and adjacent normal samples. While conventional analysis of all CpGs shows minimal hypomethylation in some normal samples, analysis restricted to solo-WCGW CpGs reveals clear PMD structure even in these samples. This improved detection enables identification of PMDs in samples where they would otherwise be missed, including normal tissues and weakly hypomethylated tumors.  

### Example 2  
**PMD Detection in Low-Coverage and Single-Cell Sequencing Data**  

The invention provides methods for detecting PMD structure in low-coverage sequencing data through analysis of solo-WCGW CpGs. Traditional PMD detection methods require relatively high sequencing coverage (typically >10×) to achieve sufficient statistical power. By focusing on solo-WCGW CpGs, the invention enables accurate PMD detection at significantly lower coverage levels.  

In representative embodiments, downsampling experiments demonstrate that solo-WCGW analysis can identify PMD structure with average genomic coverage as low as 0.05× in bulk WGBS data. This represents a 200-fold reduction in required sequencing depth compared to conventional methods. The approach is similarly effective for single-cell WGBS data, where coverage per cell is typically extremely low.  

This capability enables new applications including:  
- Cost-effective PMD analysis through low-coverage sequencing  
- Single-cell methylation profiling to assess PMD structure at cellular resolution  
- Retrospective analysis of existing low-coverage datasets for PMD information  
- Large-scale epigenetic studies where deep sequencing of all samples would be prohibitively expensive  

### Example 3  
**Standard Deviation-Based Classification of PMDs and HMDs**  

The invention provides novel methods for classifying genomic regions as PMDs or HMDs based on the standard deviation (SD) of methylation levels at solo-WCGW CpGs across multiple samples. This approach capitalizes on the discovery that PMDs show significantly greater variability in methylation levels compared to HMDs.  

In representative embodiments, the genome is divided into 100 kb bins, and the SD of solo-WCGW methylation is calculated across a set of samples. Bins are then classified as PMDs or HMDs based on their SD values, with PMDs showing higher variability. This classification can be performed using a Gaussian mixture model to determine optimal SD thresholds.  

The SD-based classification method shows high concordance with existing PMD definitions while providing several advantages:  
- Robust performance across diverse sample types  
- Ability to identify PMDs without requiring prior knowledge of PMD locations  
- Quantitative assessment of PMD hypomethylation extent  
- Compatibility with both WGBS and array-based methylation data  

Comparative analysis demonstrates that SD-based PMD classifications are highly consistent between tumor and normal samples, with approximately 83% concordance in 100 kb bin classifications.  

### Example 4  
**Developmental Dynamics of PMD Hypomethylation**  

The invention provides methods for tracking the emergence and progression of PMD hypomethylation during development. Analysis of diverse developmental stages reveals that PMD structure becomes established during embryonic development and shows lineage-specific patterns.  

In representative embodiments, analysis of gametes and early embryos demonstrates:  
- Human sperm shows high methylation with minimal PMD structure  
- Germinal vesicle oocytes exhibit deep PMD hypomethylation with boundaries distinct from somatic tissues  
- Inner cell mass and blastocyst samples show global demethylation with weak PMD structure resembling oocyte patterns  
- Primordial germ cells show extreme methylation erasure without discernable PMD structure  
- Embryonic somatic tissues show progressive establishment of PMD/HMD structure  

Later developmental stages show tissue-specific progression of PMD hypomethylation, with patterns that mirror lineage-specific hypomethylation rates. These findings indicate that PMD hypomethylation begins during early development and accumulates in a lineage-specific manner.  

### Example 5  
**PMD Hypomethylation as a Mitotic Clock**  

The invention provides methods for using PMD hypomethylation as a measure of cellular replicative history. Analysis demonstrates that PMD hypomethylation accumulates with chronological age and correlates with cumulative cell divisions.  

In representative embodiments:  
- CD4+ T cells from a 103-year-old donor show greater PMD hypomethylation than newborn T cells  
- PBMCs from elderly donors show significantly greater PMD hypomethylation than newborn samples  
- Fetal liver samples show less PMD hypomethylation than adult liver samples  
- Multiple fetal tissue types show nearly linear accumulation of hypomethylation from 9-22 weeks gestation  
- Sun-exposed skin samples show accelerated PMD hypomethylation compared to protected skin  

These patterns are consistent across hematopoietic lineages, with lymphoid cells showing faster accumulation of PMD hypomethylation than myeloid cells. The findings support use of PMD hypomethylation as a quantitative measure of cellular replicative history and mitotic age.  

### Example 6  
**Cancer-Associated PMD Hypomethylation Patterns**  

The invention provides methods for analyzing PMD hypomethylation in cancer and its relationship to tumor biology. Examination of 9,072 tumors from 33 cancer types reveals that PMD hypomethylation is nearly universal but shows extensive variation between and within cancer types.  

Key findings include:  
- The degree of PMD hypomethylation in tumors correlates with that of the tissue of origin  
- Higher genome-wide somatic mutation densities associate with deeper PMD hypomethylation  
- Somatic LINE-1 insertion breakpoints are enriched in PMD regions  
- Tumors with deeper PMD hypomethylation have more LINE-1 insertions in most cancer types  
- Genes most associated with PMD hypomethylation are enriched for proliferation and cell cycle functions  

These patterns suggest that PMD hypomethylation in cancer reflects cumulative cell divisions and may contribute to genomic instability through effects on repetitive element regulation.  

### Example 7  
**Replication Timing and H3K36me3 Effects on PMD Methylation**  

The invention provides methods for analyzing the mechanistic basis of PMD hypomethylation through examination of replication timing and chromatin marks. Analysis reveals that:  
- Solo-WCGW CpG methylation is most strongly correlated with replication timing  
- H3K36me3-marked solo-WCGWs maintain high methylation regardless of replication timing  
- Non-H3K36me3 solo-WCGWs show strong replication timing dependence  
- Late-replicating regions show greatest susceptibility to methylation loss  

These findings support a model where:  
1) H3K36me3 directs DNMT3B to maintain methylation at marked loci  
2) Late replication limits time for methylation maintenance before mitosis  
3) Solo-WCGW CpGs are particularly susceptible to maintenance failures  

### Example 8  
**Applications in Cancer Detection and Prognosis**  

The invention provides methods for applying PMD analysis to cancer detection and prognosis. Specific applications include:  
1. **Cancer Detection**: Comparing PMD hypomethylation between test samples and reference standards to identify cancerous or precancerous states.  
2. **Tumor Classification**: Using PMD hypomethylation patterns to classify tumors by tissue of origin or biological subtype.  
3. **Prognostic Assessment**: Correlating PMD hypomethylation extent with clinical outcomes such as progression or survival.  
4. **Treatment Monitoring**: Tracking changes in PMD hypomethylation during therapy as a measure of response.  
5. **Risk Stratification**: Using baseline PMD hypomethylation to assess cancer risk in normal tissues.  

These applications are enabled by the enhanced sensitivity and quantitative nature of solo-WCGW-based PMD analysis compared to conventional methylation profiling approaches.