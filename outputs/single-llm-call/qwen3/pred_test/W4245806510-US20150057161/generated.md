# DESCRIPTION

- introduce invention

The present invention relates to a novel and highly refined method for identifying, characterizing, and distinguishing between functionally distinct states of human macrophages through high-resolution transcriptome analysis. This invention enables the precise classification of macrophage polarization states—specifically M1-like and M2-like phenotypes—by leveraging next-generation RNA sequencing to uncover previously undetectable molecular signatures at the level of gene expression, alternative splicing, promoter usage, and transcript isoform diversity. Unlike prior approaches that relied on limited panels of surface markers or low-resolution microarray technologies, the invention provides a comprehensive, systems-level view of macrophage biology, revealing not only novel marker genes but also intricate regulatory mechanisms that govern functional polarization. The method is applicable to clinical diagnostics, immunotherapy monitoring, and precision medicine, offering a robust platform for the identification of macrophage subsets in human tissues and peripheral blood with unprecedented accuracy and biological relevance.

## BACKGROUND OF THE INVENTION

- introduce macrophages

Macrophages are essential innate immune cells that reside in virtually all tissues and play a central role in maintaining homeostasis, clearing cellular debris, initiating inflammatory responses, and orchestrating tissue repair. These cells exhibit remarkable plasticity, adapting their functional state in response to environmental cues such as cytokines, microbial products, and damage-associated signals. Their ability to adopt distinct phenotypes under different stimuli has long been recognized, with the classical M1-like and alternative M2-like activation states serving as paradigmatic models for pro-inflammatory and anti-inflammatory/resolving functions, respectively. However, the molecular underpinnings of these states remain incompletely defined, particularly in human systems, where phenotypic heterogeneity is compounded by donor variability, tissue-specific influences, and overlapping marker expression.

- describe M1-like macrophages

M1-like macrophages are typically induced by interferon-gamma in combination with microbial ligands such as lipopolysaccharide or tumor necrosis factor-alpha. These cells are characterized by a robust pro-inflammatory phenotype, including the secretion of interleukin-12, interleukin-23, interleukin-1beta, interleukin-6, and tumor necrosis factor-alpha. They exhibit enhanced microbicidal activity, promote Th1-type adaptive immunity, and are associated with host defense against intracellular pathogens and tumor surveillance. Surface markers such as CD64, CD86, and CD16 have been historically used to identify M1-like macrophages, yet these markers are neither exclusive nor consistently expressed across all experimental or clinical contexts.

- describe M2-like macrophages

M2-like macrophages arise in response to interleukin-4, interleukin-13, immune complexes, or glucocorticoids, and are associated with tissue remodeling, angiogenesis, parasite containment, and resolution of inflammation. These cells produce anti-inflammatory cytokines such as interleukin-10 and express surface molecules including CD23, CD163, and CD206. While useful in some settings, these markers are often shared with other myeloid populations, including dendritic cells and regulatory monocytes, leading to ambiguity in phenotypic classification. The functional diversity within the M2 subset further complicates the use of single markers, necessitating a more nuanced, multi-dimensional approach to characterization.

- discuss transcriptional reprogramming

The functional polarization of macrophages is driven by extensive transcriptional reprogramming, wherein signaling pathways converge on transcription factors that activate or repress large gene networks. This reprogramming involves not only changes in overall gene abundance but also the dynamic regulation of alternative promoters, transcription start sites, and splicing isoforms, all of which contribute to functional diversity. Previous attempts to map these changes have been constrained by the technical limitations of older gene expression platforms, which lack the sensitivity and resolution to detect subtle yet biologically significant shifts in transcript structure and abundance.

- introduce RNA sequencing

RNA sequencing represents a transformative advancement in transcriptome analysis, enabling the unbiased, high-throughput quantification of all RNA molecules within a cell. Unlike prior methods, RNA sequencing captures not only the presence and quantity of transcripts but also their structural variants, including alternative splice junctions, fusion transcripts, and non-coding RNAs. This capability allows for the detection of regulatory events that are invisible to conventional gene expression profiling tools, thereby providing a more complete picture of cellular state.

- compare RNA-seq to microarray analysis

Microarray technology, once the gold standard for transcriptome profiling, relies on hybridization to pre-defined probes, limiting its scope to annotated genes and exons. It suffers from poor dynamic range, high background noise, and an inability to distinguish between closely related isoforms or novel transcripts. In contrast, RNA sequencing provides a continuous, digital readout of transcript abundance with a detection range spanning several orders of magnitude, enabling the identification of low-abundance transcripts and subtle expression differences that are critical for distinguishing functionally distinct cell states.

- summarize limitations of prior art

Prior art in macrophage classification has been fundamentally constrained by its reliance on a narrow set of surface markers and low-resolution transcriptomic data. These approaches fail to capture the full complexity of macrophage polarization, often misclassifying heterogeneous populations or overlooking key regulatory mechanisms. Furthermore, microarray-based studies have produced inconsistent results across laboratories due to platform variability, probe design limitations, and insufficient biological replication. As a result, the field lacks a standardized, high-fidelity method for defining human macrophage phenotypes in clinical and research settings.

## SHORT DESCRIPTION OF THE INVENTION

- motivate need for high-resolution transcriptome data

There exists a critical and unmet need for a method capable of resolving the molecular heterogeneity of human macrophages with high precision, particularly in contexts where functional state dictates disease outcome, such as cancer, chronic inflammation, and autoimmune disorders. Conventional approaches are inadequate for capturing the nuanced transcriptional programs that define macrophage polarization, especially when subtle but biologically significant changes occur in isoform usage, promoter selection, or non-coding transcript expression.

- describe application of RNA-seq to macrophage polarization

The invention applies RNA sequencing to systematically profile the transcriptomes of human macrophages polarized toward M1-like and M2-like states under physiologically relevant conditions. By analyzing multiple biological replicates from primary human monocytes, the invention reveals a comprehensive landscape of gene expression, alternative splicing, and transcriptional regulation that was previously inaccessible.

- summarize new insights into human macrophage biology

The invention uncovers novel biological insights, including the identification of previously unrecognized macrophage-specific transcripts, the discovery of differential promoter usage and alternative transcription start sites between polarization states, and the characterization of splice variants that are selectively expressed in M1-like or M2-like macrophages. These findings redefine the molecular architecture of macrophage polarization and establish a new framework for functional classification.

- introduce method for identifying M1-like and M2-like macrophages

The invention introduces a method for identifying M1-like and M2-like macrophages by detecting a panel of signature transcripts identified through RNA sequencing, including novel marker genes, splice variants, and regulatory elements. This method enables accurate classification based on transcriptomic profiles rather than limited surface marker expression.

- describe preferred embodiment of method

In a preferred embodiment, the method comprises isolating peripheral blood mononuclear cells from a human subject, differentiating CD14-positive monocytes into macrophages using granulocyte-macrophage colony-stimulating factor, polarizing the macrophages with interferon-gamma or interleukin-4, extracting total RNA, generating sequencing libraries, performing paired-end RNA sequencing, aligning reads to the human reference genome, quantifying transcript abundance, and identifying differentially expressed genes and isoforms using bioinformatic tools such as Cufflinks and Cuffdiff.

- describe another preferred embodiment of method

In another preferred embodiment, the method utilizes a hybridization-based assay designed to detect the presence or absence of specific RNA transcripts or splice variants identified by RNA sequencing, enabling rapid, cost-effective, and clinically deployable classification of macrophage polarization states in diagnostic settings.

- summarize invention

The invention provides a transformative method for the precise identification and classification of human macrophage polarization states through high-resolution transcriptome analysis, enabling the discovery of novel biomarkers, regulatory mechanisms, and diagnostic signatures that surpass the limitations of prior art.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce macrophages

Macrophages are terminally differentiated myeloid cells derived from circulating monocytes that play indispensable roles in immune surveillance, tissue homeostasis, and inflammatory regulation. Their functional diversity is not static but dynamically shaped by signals from the microenvironment, resulting in a spectrum of activation states that cannot be adequately captured by single-marker definitions.

- motivate RNA-seq

The application of RNA sequencing to macrophage biology overcomes the inherent limitations of microarray platforms by providing a comprehensive, quantitative, and isoform-resolved view of the transcriptome. This technology enables the detection of transcripts with low abundance, novel splice variants, and non-annotated transcripts, all of which are critical for understanding the molecular logic of macrophage polarization.

- describe RNA-seq results

RNA sequencing of M1-like and M2-like macrophages revealed a significantly broader dynamic range of gene expression compared to microarray data, with over 1,700 genes differentially expressed between M1-like and M2-like states at a fold-change threshold of 1.5, compared to fewer than 900 genes detected by microarray under identical conditions. The increased sensitivity of RNA sequencing enabled the identification of numerous low-abundance transcripts that were previously undetectable.

- identify novel marker genes

The invention identifies a set of novel marker genes that are differentially expressed between M1-like and M2-like macrophages, including CD120b, SLAMF7, CD1a, CD1b, CD93, and CD226. These genes are not only statistically significant but also biologically relevant, as they encode cell surface proteins that can be readily detected by flow cytometry or immunohistochemistry, enabling direct translation into diagnostic assays.

- describe differential promoter usage

RNA sequencing analysis revealed that multiple genes exhibit differential promoter usage between M1-like and M2-like macrophages. For example, the gene encoding PDLIM7 utilizes distinct transcription start sites in each polarization state, resulting in the production of isoforms with divergent functional properties. This finding demonstrates that macrophage polarization is not merely a matter of gene on/off switching but involves complex regulatory rewiring at the promoter level.

- describe alternative transcription start sites

The invention demonstrates that alternative transcription start sites are frequently employed in a polarization-specific manner. Genes such as APOL3 and LILRB1 show preferential usage of upstream or downstream transcription start sites in M1-like versus M2-like macrophages, leading to the production of protein isoforms with distinct signaling capacities. These events are invisible to microarray platforms, which target only predefined exonic regions.

- describe differential splicing variants

Differential splicing is a central feature of macrophage polarization, with over 20 genes exhibiting alternative splicing patterns between M1-like and M2-like states. The invention identifies specific splice variants of PDLIM7, where the v1 isoform is enriched in M1-like macrophages and the v2 isoform in M2-like macrophages. This isoform switching correlates with functional differences in cytoskeletal organization and transcriptional regulation, suggesting a direct mechanistic link between splicing and polarization.

- discuss PDLIM7

The gene encoding PDLIM7, a PDZ and LIM domain-containing scaffold protein, is one of the most striking examples of polarization-dependent alternative splicing identified by the invention. RNA sequencing revealed that while total PDLIM7 expression is only modestly altered, the relative abundance of its three major isoforms shifts dramatically between M1-like and M2-like macrophages. This isoform-specific regulation was validated by isoform-specific qPCR and is not detectable by conventional microarray probes.

- discuss RNA-seq advantages

RNA sequencing provides unparalleled advantages over microarray analysis, including higher sensitivity, broader dynamic range, the ability to detect novel transcripts and splice variants, and the capacity to quantify transcript abundance without reliance on pre-defined probes. These features collectively enable the invention to uncover molecular signatures of macrophage polarization that were previously inaccessible.

- discuss microarray limitations

Microarray technology is fundamentally limited by its dependence on hybridization to pre-designed probes, which restricts detection to annotated exons and known transcripts. The technology suffers from high background noise, poor reproducibility across platforms, and an inability to resolve alternative splicing or promoter usage. Consequently, microarray data frequently misclassify macrophage states and miss critical regulatory events.

- describe cell surface marker identification

The invention systematically interrogates the human surfaceome to identify cell surface proteins that are differentially expressed between M1-like and M2-like macrophages. Through RNA sequencing, the invention identifies CD120b, SLAMF7, CD1a, CD1b, CD93, and CD226 as novel markers with high specificity and robust expression differences, enabling their use in multiplexed flow cytometry panels for clinical phenotyping.

- introduce method of aspect (1)

In a first aspect, the invention provides a method for identifying M1-like and M2-like macrophages by measuring the expression levels of a panel of RNA transcripts selected from the group consisting of CD120b, SLAMF7, CD1a, CD1b, CD93, CD226, and splice variants of PDLIM7, using RNA sequencing or quantitative PCR.

- describe primer design

For quantitative PCR-based detection, primers are designed to span splice junctions or target exon-exon boundaries unique to specific isoforms, ensuring selective amplification of the polarization-associated transcript variant. Primer sequences are optimized for efficiency and specificity, with amplicons ranging from 80 to 150 base pairs to ensure compatibility with clinical diagnostic platforms.

- introduce method of aspect (2)

In a second aspect, the invention provides a method for detecting the presence of M1-like or M2-like macrophages using a hybridization array containing probes complementary to the novel marker transcripts and splice variants identified by RNA sequencing.

- describe probe design

Probes are designed to be 40 to 60 nucleotides in length and are positioned to uniquely hybridize to splice junctions, alternative exons, or untranslated regions specific to M1-like or M2-like transcripts. Each probe is chemically modified for enhanced binding stability and labeled with a fluorescent or chemiluminescent tag for signal detection.

- introduce method of aspect (3)

In a third aspect, the invention provides a method for classifying macrophage polarization using a microarray-based platform comprising immobilized oligonucleotide probes specific to the novel marker transcripts and splice variants identified by RNA sequencing.

- describe hybridization array

The hybridization array contains a multiplexed set of probes targeting the 20 most discriminative transcripts identified by RNA sequencing, including CD120b, SLAMF7, CD1a, CD1b, CD93, CD226, and PDLIM7 isoforms. Hybridization is performed under stringent conditions to minimize cross-reactivity, and signal intensity is quantified using high-resolution scanners calibrated for low-abundance transcript detection.

- introduce method of aspect (4)

In a fourth aspect, the invention provides a method for identifying M1-like or M2-like macrophages using binding molecules that specifically recognize the protein products of the novel marker genes.

- describe binding molecules

The binding molecules include monoclonal antibodies, single-domain antibodies, or antibody fragments directed against CD120b, SLAMF7, CD1a, CD1b, CD93, and CD226. These molecules are conjugated to fluorophores, enzymes, or magnetic beads for use in flow cytometry, immunohistochemistry, or magnetic separation assays.

- introduce method of aspect (5)

In a fifth aspect, the invention provides a method for predicting macrophage polarization state by integrating transcriptomic data from multiple genes into a computational algorithm that assigns a polarization score based on the relative expression levels of the identified marker panel.

- discuss CD120b

CD120b, also known as tumor necrosis factor receptor 2, is preferentially expressed on M1-like macrophages and is involved in sustained NF-kB activation and cell survival signaling. Its identification as a novel M1 marker enables the distinction of pro-inflammatory macrophages in contexts where CD86 expression is ambiguous.

- discuss SLAMF7

SLAMF7 is a signaling lymphocytic activation molecule that is highly upregulated in M1-like macrophages and is associated with enhanced antigen presentation and co-stimulatory function. Its expression is not observed in M2-like macrophages, making it a highly specific marker for inflammatory macrophage states.

- discuss CD1a and CD1b

CD1a and CD1b, traditionally associated with dendritic cells, are found to be significantly upregulated in M2-like macrophages, suggesting a role in lipid antigen presentation during tissue repair. This finding challenges conventional assumptions about the cellular identity of these molecules and expands their diagnostic utility.

- discuss CD93

CD93 is a transmembrane glycoprotein involved in phagocytosis and endothelial adhesion, and its expression is markedly elevated in M2-like macrophages. The invention demonstrates that CD93 is not merely a marker of monocyte differentiation but is actively regulated during polarization, with implications for its role in resolving inflammation.

- discuss CD226

CD226, originally characterized as a T-cell activation molecule, is identified herein as a novel M2-associated surface marker. Its expression on macrophages correlates with migratory and tissue-repair functions, suggesting a previously unrecognized role in myeloid cell behavior during resolution of inflammation.

- discuss RNA-seq benefits

The use of RNA sequencing in this invention provides a comprehensive, unbiased, and quantitative framework for macrophage classification that is not constrained by prior assumptions about marker expression. It enables the discovery of novel regulatory mechanisms, improves diagnostic accuracy, and facilitates the development of targeted therapeutic interventions.

- conclude RNA-seq advantages

In summary, RNA sequencing provides a superior platform for macrophage phenotyping by revealing the full complexity of transcriptomic regulation, including alternative splicing, promoter usage, and isoform switching. The invention leverages these advantages to establish a new standard for the identification and classification of human macrophage polarization states.

### EXAMPLES

- abbreviations

For the purposes of clarity and consistency, the following abbreviations are used throughout: PBMC for peripheral blood mononuclear cells, M-CSF for macrophage colony-stimulating factor, GM-CSF for granulocyte-macrophage colony-stimulating factor, IFN-γ for interferon-gamma, LPS for lipopolysaccharide, IL-4 for interleukin-4, IL-13 for interleukin-13, RPKM for reads per kilobase per million mapped reads, TSS for transcription start site, CDS for coding sequence, FACS for fluorescence-activated cell sorting, qPCR for quantitative polymerase chain reaction, PCA for principal component analysis, FC for fold change, and EGAN for exploratory gene association network.

- isolate peripheral blood mononuclear cells

Peripheral blood mononuclear cells were isolated from buffy coats obtained from healthy donors following density gradient centrifugation using Pancoll. Cells were washed in phosphate-buffered saline and counted using a hemocytometer. Viability was confirmed to exceed 95% by trypan blue exclusion.

- isolate CD14+ monocytes

CD14-positive monocytes were purified from PBMCs using magnetic-activated cell sorting with anti-CD14-conjugated microbeads according to the manufacturer’s protocol. Purity was routinely confirmed by flow cytometry to exceed 95%.

- generate macrophages

CD14-positive monocytes were cultured in RPMI-1640 medium supplemented with 10% fetal calf serum and either 500 U/mL GM-CSF or 100 U/mL M-CSF for three days to generate immature macrophages. Medium was replaced daily to maintain cytokine concentration.

- polarize macrophages

Immature macrophages were polarized for an additional three days with 200 U/mL IFN-γ and 10 µg/mL LPS to induce M1-like polarization, or with 1,000 U/mL IL-4 and 100 U/mL IL-13 to induce M2-like polarization. Control cells were maintained in medium without polarizing stimuli.

- stain cells with monoclonal antibodies

Cells were harvested, washed, and incubated with Fc receptor-blocking reagent prior to staining with a panel of fluorochrome-conjugated monoclonal antibodies targeting CD1a, CD1b, CD93, CD226, CD120b, SLAMF7, CD64, CD86, CD23, and HLA-DR.

- perform flow cytometry

Stained cells were analyzed on a BD LSR II flow cytometer. Data were processed using FlowJo software, and median fluorescence intensity was calculated for each marker. Gating was performed on live, single cells, and isotype controls were used to define background fluorescence.

- isolate RNA

Macrophages were lysed in TRIzol reagent, and total RNA was extracted according to the manufacturer’s protocol. RNA integrity was assessed using an Agilent Bioanalyzer, and only samples with an RNA integrity number greater than 8.0 were used for downstream analysis.

- perform quantitative PCR

Reverse transcription was performed using the Transcriptor First Strand cDNA Synthesis Kit. Quantitative PCR was carried out using LightCycler TaqMan Master Mix on a LightCycler 480 II instrument. GAPDH was used as a reference gene, and relative expression was calculated using the 2−ΔΔCT method.

- perform microarray-based transcriptional profiling

Total RNA was purified using the MinElute Reaction Cleanup Kit, and biotin-labeled cRNA was generated using the TargetAmp Nano-g Biotin-cRNA Labeling Kit. Samples were hybridized to Human HT-12V3 Beadchips and scanned on an Illumina HiScanSQ system.

- analyze microarray data

Raw intensity data were normalized using quantile normalization in the limma package of Bioconductor. Genes with a coefficient of variation less than 0.5 were excluded. Differentially expressed genes were identified using Student’s t-test with Benjamini-Hochberg correction for multiple testing.

- perform RNA-seq

RNA-seq libraries were prepared using the Illumina TruSeq RNA Sample Preparation Kit. Paired-end 100-bp sequencing was performed on an Illumina HiScanSQ platform. Reads were aligned to the human reference genome hg19 using TopHat and Bowtie.

- analyze RNA-seq data

Reads were quantified using Cufflinks and Cuffdiff. RPKM values were calculated for RefSeq transcripts. Differential expression was defined as a fold-change greater than 1.5 and a p-value less than 0.05 after multiple testing correction.

- perform a priori information-based network analysis

Network graphs were generated using EGAN software, incorporating prior knowledge from pathway databases. Genes with fold-change thresholds of FC >4 for M1 and FC >2.5 for M2 were used to construct primary networks.

- perform statistical analysis

Statistical comparisons between groups were performed using paired or unpaired Student’s t-tests as appropriate. Linear regression was used to correlate fold-change values between RNA-seq and microarray data. All analyses were conducted using SPSS version 19.0.

- conclude examples

The examples demonstrate that RNA sequencing provides a superior platform for macrophage phenotyping, enabling the identification of novel markers, splice variants, and regulatory mechanisms that are inaccessible to microarray technology.

### Example 1

- introduce macrophage model system

A well-controlled in vitro model system was established using GM-CSF-differentiated human monocytes polarized with IFN-γ or IL-4 to generate M1-like and M2-like macrophages, respectively. This model was selected for its reproducibility and physiological relevance to inflammatory conditions.

- compare M1 and M2 polarization

M1-like macrophages exhibited elevated expression of CD64, CXCL10, and TNF, while M2-like macrophages showed increased expression of CD23, CCL18, and CLEC4A, consistent with prior literature. However, surface marker expression varied depending on the differentiation cytokine used, highlighting the need for transcriptome-based classification.

- analyze surface marker expression

Flow cytometry revealed that CD64 was a reliable M1 marker, while CD23 was a robust M2 marker. However, CD86 expression was inconsistent across donors, and CD163 was expressed in both states, underscoring the limitations of single-marker approaches.

- discuss limitations of M1 and M2 polarization

The binary M1/M2 classification is an oversimplification of macrophage biology. The invention demonstrates that polarization is a continuous spectrum governed by complex transcriptional networks, and that transcriptome-based profiling is necessary to capture this heterogeneity.

### Example 2

- introduce microarray-based gene expression profiling

Microarray analysis was performed on seven biological replicates of unpolarized, M1-like, and M2-like macrophages. Principal component analysis revealed clear segregation of samples by polarization state, confirming the validity of the model.

- perform PCA and hierarchical clustering

PCA showed that the first two principal components accounted for over 70% of variance, with M1 and M2 samples clustering distinctly. Hierarchical clustering confirmed the reproducibility of gene expression patterns across donors.

- analyze known M1 and M2 macrophage markers

Known markers such as FCGR1A, CXCL9, IL1B, and CCL17 were significantly upregulated in M1 and M2 states, respectively. However, several established markers showed low fold-changes or inconsistent expression, limiting their diagnostic utility.

- generate M1 and M2 associated networks

Network analysis using EGAN revealed interconnected gene modules enriched in M1 and M2 states. However, many hub genes were not detected due to low expression or probe design limitations.

- discuss limitations of microarray data

Microarray data failed to detect several key genes, including APOL3 and LILRB1, which were identified by RNA sequencing. This omission significantly reduced the biological insight derived from the network analysis.

### Example 3

- introduce RNA-seq for transcriptome analysis

RNA sequencing was performed on three biological replicates of M1-like and M2-like macrophages. The average read depth exceeded 15 million reads per sample, providing sufficient coverage for isoform-level analysis.

- analyze RNA-seq data for M1 and M2 macrophages

RNA sequencing confirmed all known M1 and M2 markers and identified over 900 additional differentially expressed genes not detected by microarray. The dynamic range of expression was significantly greater in RNA-seq data.

- compare RNA-seq and microarray data

A strong correlation was observed between RNA-seq and microarray data for highly expressed genes, but RNA-seq revealed significant differential expression in genes with low abundance or complex splicing patterns that were invisible to microarrays.

### Example 4

- analyze differential expression using RNA-seq

Using a fold-change threshold of 1.5 and a p-value of 0.05, RNA sequencing identified 1,736 genes differentially expressed in M1-like versus M2-like macrophages, compared to only 834 genes detected by microarray.

- compare RNA-seq and microarray data for differential expression

RNA-seq detected fourfold more genes with fold-changes greater than 4, demonstrating its superior sensitivity. Genes such as DUOX1 and GBP7 were exclusively identified by RNA sequencing.

- discuss advantages of RNA-seq over microarray

The increased dynamic range, lower background noise, and ability to detect novel transcripts make RNA sequencing a superior tool for identifying subtle yet biologically significant differences in macrophage polarization.

- analyze fold-change distribution of RNA-seq data

The fold-change distribution in RNA-seq data spanned six orders of magnitude, compared to only four in microarray data, reflecting the broader detection capability of sequencing.

- discuss limitations of microarray data

Microarray data exhibited high variance for low-abundance transcripts and failed to detect many functionally relevant genes due to probe design constraints and hybridization artifacts.

### Example 5

- analyze exon resolution transcriptome analysis

RNA sequencing enabled exon-level analysis of macrophage markers such as CD68, CD64, and CD23. Expression patterns across individual exons were visualized and compared to microarray, qPCR, and FACS data.

- visualize RNA-seq data for macrophage markers

For CD64, RNA sequencing revealed complete absence of expression in M2-like macrophages across all exons, whereas microarray data showed only modest reduction. For CD23, RNA sequencing confirmed high expression in M2-like macrophages with isoform-specific variation.

- compare RNA-seq data with array, qPCR, and FACS data

RNA sequencing showed superior concordance with qPCR and FACS data for splice variants and low-abundance transcripts, whereas microarray data exhibited significant discrepancies, particularly for genes with alternative promoters.

### Example 6

- apply network analysis to RNA-seq data

Network analysis using EGAN was performed on RNA-seq data, revealing two novel gene clusters: the apolipoprotein L family and the leukocyte immunoglobulin-like receptor family.

- visualize array-based gene expression on RNA-seq network

When microarray expression values were overlaid onto the RNA-seq-derived network, only 73% of M1 network nodes and 54% of M2 network nodes were detectable, demonstrating the superior information content of RNA sequencing.

- discuss advantages of RNA-seq over microarray for network analysis

RNA sequencing enabled the identification of central regulatory hubs and novel gene families that were completely absent from microarray-based networks, providing deeper biological insight into macrophage polarization.

### Example 7

- identify splice variants and RNA chimaera

Analysis of RNA-seq data using Cufflinks and Cuffdiff revealed 9 genes with alternative promoters, 28 genes with alternative transcription start sites, and 20 genes with differential coding sequence usage between M1-like and M2-like macrophages.

- analyze alternative promoters, TSS, and CDS

The gene PDLIM7 exhibited differential usage of three coding sequence variants, with v1 enriched in M1-like macrophages and v2 in M2-like macrophages. Alternative promoters were identified upstream of CD120b and SLAMF7.

- validate alternative splicing using qPCR

Isoform-specific qPCR confirmed the RNA-seq findings, demonstrating that the v1 and v2 isoforms of PDLIM7 are differentially regulated in a polarization-dependent manner, with no detectable signal using conventional qPCR primers.

### Example 8

- identify new markers for M1 and M2 macrophages

The invention identifies CD120b, SLAMF7, CD1a, CD1b, CD93, and CD226 as novel markers for M1-like and M2-like macrophages. These markers are expressed at the protein level and are detectable by flow cytometry.

- validate new markers using RNA-seq data

All six markers showed statistically significant differential expression in RNA-seq data, with fold-changes exceeding 2.5 and p-values below 0.01. Validation by qPCR and FACS confirmed their specificity and robustness across multiple donors.