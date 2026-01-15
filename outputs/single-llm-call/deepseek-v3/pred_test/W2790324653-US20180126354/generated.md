Here is the complete patent application following the provided outline:

# DESCRIPTION

## INTRODUCTION

Multiple myeloma represents a malignant proliferation of plasma cells within the bone marrow, characterized by heterogeneous genetic alterations that drive disease pathogenesis and progression. The disease typically evolves from an asymptomatic premalignant condition known as monoclonal gammopathy of undetermined significance (MGUS), which may progress through smoldering myeloma before manifesting as symptomatic multiple myeloma requiring therapeutic intervention. The premalignant phase harbors initiating genetic events that can be broadly categorized into two primary molecular subtypes exhibiting minimal overlap. The first subtype, hyperdiploid multiple myeloma, demonstrates characteristic trisomies of specific odd-numbered chromosomes. The second subtype frequently involves translocations of the immunoglobulin heavy chain (IGH) locus that dysregulate oncogenes through juxtaposition with potent B-cell-specific enhancer elements. 

Numerous secondary genetic alterations emerge during disease progression, including MYC translocations, single nucleotide variants affecting RAS pathway genes, and various copy number alterations. These secondary events contribute to the molecular heterogeneity observed in multiple myeloma and influence clinical outcomes. Recent advances have enabled the development of prognostic models that integrate molecular profiling with traditional staging systems, significantly improving risk stratification accuracy. Such models demonstrate the clinical utility of comprehensive genomic characterization in multiple myeloma management.

Current approaches for detecting myeloma-associated mutations include whole exome sequencing and targeted sequencing panels. While exome sequencing provides broad coverage, targeted methods offer advantages including reduced computational burden, faster turnaround times, and deeper sequencing at fixed costs. Existing targeted platforms have focused primarily on detecting single nucleotide variants and copy number alterations, with limited capacity for simultaneous translocation detection. There remains an unmet need for integrated platforms capable of comprehensively profiling all major mutation classes relevant to multiple myeloma pathogenesis and progression.

## SUMMARY

The present invention provides a novel capture-based sequencing approach specifically designed for multiple myeloma genomic profiling. This platform offers significant advantages over existing methods through its ability to simultaneously detect copy number variants, single nucleotide variants, and translocations using a single integrated assay. The invention enables personalized treatment planning by providing comprehensive molecular characterization of individual tumors. 

Key aspects of the platform include a custom-designed oligonucleotide probe set targeting approximately 3.3 megabases of genomic space encompassing 465 genes relevant to multiple myeloma pathogenesis. The probe design incorporates strategic coverage of the IGH locus and its canonical translocation partners, along with tiling across the MYC locus to facilitate detection of secondary translocations. The platform demonstrates superior performance compared to previous targeted sequencing approaches through optimized probe configuration and bioinformatics analysis pipelines.

The method involves several key steps: preparation of DNA sequencing libraries from both tumor and matched normal cells, hybridization of these libraries to the custom capture array, deep sequencing of captured fragments, and sophisticated bioinformatic analysis to identify somatic alterations. The platform specifically enables detection of chromosome-level, arm-level, and focal copy number alterations through analysis of sequencing depth ratios. Translocation detection employs specialized algorithms tuned to minimize false positives while maintaining sensitivity.

The oligonucleotide array comprises probes targeting exonic regions of canonical IGH translocation partners including CCND1, CCND3, FGFR3, MAF, MAFB, and WHSC1. Additional probes tile across the MYC locus to capture both intra- and inter-chromosomal rearrangements. The array design also includes comprehensive coverage of genes involved in DNA repair pathways, B-cell biology, and frequently mutated in hematologic malignancies. In total, the platform detects alterations in at least 400 genes, with preferred embodiments targeting between 465 and 467 genes.

The DNA capture array composition includes biotinylated oligonucleotide probes complementary to targeted genomic regions. These probes hybridize specifically to their cognate sequences in the prepared sequencing libraries, enabling selective enrichment prior to sequencing. The platform demonstrates particular utility in detecting canonical IGH translocation partners along with additional genes including ATM, BRCA2, CLIP1, CSMD3, and EP400. Through this comprehensive design, the invention provides unprecedented capability for integrative analysis across mutation types in multiple myeloma.

## DETAILED DESCRIPTION

The invention provides a DNA capture array and associated methods for identifying multiple myeloma-associated mutations. The array comprises oligonucleotide probes designed to hybridize with specific genomic regions of interest. The method involves preparing DNA sequencing libraries from both tumor cells and matched normal cells, followed by hybridization to the capture array and deep sequencing. 

Sequencing libraries are prepared using standard molecular biology techniques, with optional incorporation of molecular barcodes to facilitate multiplexing. The DNA capture array contains probes targeting all exonic regions of the 465 selected genes, along with tiled coverage of the IGH and MYC loci. Hybridization conditions are optimized to maximize specificity while maintaining sensitivity for variant detection.

Following hybridization and capture, sequencing is performed to achieve maximum average depth, typically exceeding 100x coverage. The resulting data undergoes sophisticated bioinformatic analysis to identify somatic variants present in tumor but not normal cells. This analysis pipeline provides distinct advantages over whole-exome sequencing approaches, particularly in detection sensitivity for copy number alterations and translocations.

The platform enables simultaneous detection of single nucleotide variants, copy number changes, and translocations from a single assay. Data analysis incorporates multiple complementary algorithms to maximize detection accuracy across variant types. The resulting mutational profiles facilitate prognostic assessment and therapeutic selection based on the specific molecular features of each tumor.

Integration with gene expression profiling enhances the clinical utility of the platform. The invention further provides methods for determining mutual exclusivity and co-occurrence patterns among different mutation classes. Specific analyses include testing mutual exclusivity between NRAS, KRAS, and IGLL5 mutations, as well as between hyperdiploidy and non-MYC IGH translocations. The platform also enables investigation of co-occurrence relationships between copy number alterations and single nucleotide variants.

### DETAILED DESCRIPTION

The custom capture sequencing platform is designed to target approximately 3.3 Mb of genomic space encompassing 465 genes and the IGH region. Probe design specifically facilitates detection of chromosome-level, arm-level, and focal copy number alterations through optimized coverage density. Probes targeting the IGH locus and MYC locus are strategically positioned to maximize translocation detection sensitivity.

The platform hypothesizes that endonucleolytic cleavage of free DNA ends precedes fusion with partner chromosomes in translocation events. Probe design accounts for this by positioning probes both inside and outside relevant genomic elements. Automated dual-indexed library construction enables efficient sample multiplexing while maintaining data quality.

Library pools are hybridized with biotinylated probe sets under optimized conditions. Post-capture library quantification employs quantitative PCR to ensure appropriate sequencing representation. Sequencing is performed on high-throughput platforms such as the HiSeq2000 or HiSeq2500, generating paired-end reads of sufficient length for accurate variant calling.

Read alignment against the human reference genome (GRCh37-lite) utilizes the Burrows-Wheeler Aligner (BWA) for optimal performance. Single nucleotide variant calling incorporates multiple complementary algorithms including samtools, SomaticSniper, MuTect, Strelka, and VarScan2 to maximize sensitivity and specificity. 

Copy number variant detection employs CopyCAT2 software parameterized with a Gaussian mixture model to distinguish true alterations from noise. Analytical pipelines include specific steps to exclude samples with suboptimal quality metrics from copy number analysis. Focal and arm-level copy number alterations are annotated with relevant gene information to facilitate biological interpretation.

Translocation detection utilizes the LUMPY algorithm with additional machine learning-based filtering to reduce false positives. For IGH translocations, a support vector machine (SVM) classifier is trained using available fluorescence in situ hybridization (FISH) data to optimize precision. MYC translocations are filtered using a separately trained SVM model with manually defined decision boundaries.

The platform includes specialized methods for mapping IGH constant, switch, and enhancer regions using public genome resources. Breakpoint validation employs PCR amplification of derivative chromosomes followed by Sanger sequencing. Novel translocation partners are prioritized based on proximity to cancer-associated genes and supporting RNA expression data.

Somatic single nucleotide variant detection involves rigorous quality filtering and annotation. Variants are classified by functional impact and assessed for potential deleteriousness using PolyPhen-2 and SIFT algorithms. Integration of results from multiple variant callers improves overall detection accuracy.

The platform enables investigation of clonal architecture through comparison of variant allele frequencies at different sequencing depths. Downsampling analyses demonstrate that the majority of biologically relevant variants can be detected at moderate sequencing depths (100-300x), with diminishing returns at higher coverage levels.

Integrative analysis across mutation types employs statistical methods to identify significant patterns of mutual exclusivity and co-occurrence. These analyses control for potential confounding factors such as hypermutator phenotypes. Survival analysis incorporates clinical outcome data to assess the prognostic significance of specific molecular features.

## EXAMPLES

The following examples illustrate the design and application of the oligonucleotide probe array for targeted sequence capture in multiple myeloma.

### Example 1

The platform was designed with oligonucleotide probes specifically targeting multiple myeloma-associated genomic regions. The custom capture sequencing platform successfully detected copy number variants, single nucleotide variants, and translocations in primary myeloma samples. Probe configuration enabled comprehensive assessment of all major mutation classes from a single assay.

### Example 2

Implementation of the platform involved sequencing 95 paired tumor and normal DNA samples. The method achieved average sequencing depths exceeding 100x across all targets, with excellent coverage uniformity. Comparison with existing exome sequencing and FISH data demonstrated high concordance for both single nucleotide variants and translocations.

### Example 3

Analysis identified copy number alterations with established prognostic significance in multiple myeloma. The platform detected the full spectrum of copy number changes from whole-chromosome gains to focal deletions. Specific findings included chromosome-level hyperdiploidy events and focal deletions encompassing tumor suppressor genes such as BRCA2.

### Example 4

The platform successfully detected IGH translocations using the optimized LUMPY/SVM pipeline. Validation against existing FISH data demonstrated 100% precision with 64% recall for known IGH translocations. The method identified canonical IGH translocations at frequencies consistent with published literature.

### Example 5

Novel IGH translocations were prioritized based on supporting read evidence and proximity to cancer-associated genes. A validated t(14;22) translocation juxtaposing IGH with IGLL5 was characterized in detail. Breakpoint analysis revealed potential involvement of a super-enhancer region that may influence expression of nearby genes.

### Example 6

The platform detected both intra- and inter-chromosomal MYC translocations using the specialized filtering approach. Putative MYC rearrangements were called in 6% of samples, consistent with expected frequencies. Intra-chromosomal events frequently involved the PVT1 and POU5F1B loci as previously reported.

### Example 7

Non-silent single nucleotide variants were identified in all tumor samples analyzed. The platform detected an average of 20 somatic mutations per sample, with nearly all samples harboring at least one predicted deleterious variant. Frequently mutated genes included known drivers such as KRAS and NRAS.

### Example 8

Deep sequencing of selected samples demonstrated that most biologically relevant variants could be detected at moderate sequencing depths. Analysis of variant allele frequencies revealed that the majority of mutations were clonal, with relatively few subclonal variants identified even at extremely high coverage levels.

### Example 9

Integrative analysis across mutation types revealed significant patterns of mutual exclusivity and co-occurrence. Notable findings included mutual exclusivity between IGLL5 mutations and RAS pathway alterations, as well as co-occurrence of specific copy number alterations. IGLL5 mutations showed significant association with disease progression in survival analysis.