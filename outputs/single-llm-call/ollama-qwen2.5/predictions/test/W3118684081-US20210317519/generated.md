### TAPSβ Data Preprocessing and Analysis
Sequencing reads were trimmed with Trim Galore! v0.3.1 to remove adaptors and low-quality bases. Trimmed reads were mapped to a combined genome of spike-in sequences and the mm9 mouse genome using BWA mem v.0.7.12. PCR duplicates were removed, and reads with MAPQ < 10 were excluded. Modified bases were called by asTair v3.3.1, and raw signals were calculated as the ratio of C to C+T at each site.

### Coverage Analysis of CAPS and ACE-seq
CpG island annotation was downloaded from UCSC. Each CpG island was evenly binned into ten windows, and 4-kb flanking regions were binned into 20 windows. The average coverage was calculated using Bedtools map, and coverage at each site was normalized by the ratio of overall coverage between CAPS and ACE-seq datasets.

### Estimation of 5hmC Using Maximum Likelihood
The MLML estimation method was applied to sites with a minimum coverage of 5 reads. Sites with conflicts were excluded. The average levels of 5mC and 5hmC estimated by MLML were tiled into 1-kb bins for visualization on the mm9 genome using the Integrative Genomics Viewer.

### Quantifying Enrichment of 5hmCGs in Regulatory Elements
The list of genomic regulatory elements was downloaded from a GitHub repository. High-confidence 5hmCG sites (BH-adjusted p-value < 0.05 and coverage ≥5 reads) were annotated using bedtools intersect. The number of 5hmCG sites falling into each category was counted, and enrichment was assessed by comparing to a background distribution generated from random sampling.

### Genome-Wide Analysis of PS and PS-C
ChIP-seq data for histone modifications H3K4me1 and H3K4me3 were downloaded from the ENCODE project. The centers of broad peaks or predicted regions were used, and average modification levels were calculated by tiling 3-kb flanking regions into 100-bp bins. Profiles were visualized using deepTools 3.3.0 to analyze genome-wide patterns of PS and PS-C.

### Pairwise Comparisons of TAPSβ
Replicate one of RRoxBS-seq data was used due to the highest read count among four replicates. Three whole-genome oxBS-seq replicates were pooled for correlation analysis. Sites with a minimum coverage of ten reads were used, and Pearson's r was calculated using R functions cor and smoothScatter.

### Pairwise Comparisons of CAPS
To compare CAPS with ACE-seq and TAB-seq, raw 5hmCG signals (C/(C+T)) were calculated within 10-kb genomic bins. The TAPS−TAPSβ subtraction signal was calculated as the average estimated 5hmC levels from MLML output, facilitating a comprehensive comparison of methylation levels across different datasets.

### Statistical Test of 5hmC
The binomial test was used to call 5hmC at sites with a minimum coverage of five reads. The probability p was the false-positive rate (0.0072) of CAPS, calculated from unmodified control DNA. Cytosines with BH-adjusted p-value < 0.05 were selected for downstream analysis and further enrichment studies.

### Reporting Summary
Further information on research design is available in the Nature Research Reporting Summary linked to this article, providing detailed methodologies and data sources used in the study.
```
