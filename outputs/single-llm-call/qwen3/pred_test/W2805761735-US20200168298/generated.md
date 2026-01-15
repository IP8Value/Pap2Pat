# DESCRIPTION

## FIELD

- define field of invention

The present invention relates to computational systems and methods for the precise genotyping of highly polymorphic and structurally complex human genes using high-throughput sequencing data, particularly for pharmacogenomic applications. More specifically, the invention provides a novel framework for allelic decomposition of genes that exhibit copy number variations, partial deletions or duplications, hybrid gene formations with pseudogenes, and combinations thereof, all of which are commonly observed in clinically significant drug-metabolizing enzymes such as CYP2D6 and CYP2A6. The invention enables the accurate determination of both major and minor star-alleles, including previously uncharacterized variants, by integrating read alignment, structural variant detection, copy number estimation, and combinatorial optimization techniques into a unified computational pipeline. This framework is specifically designed to overcome the limitations of conventional genotyping platforms, which rely on predefined variant panels and are incapable of resolving complex structural rearrangements or detecting novel allelic configurations arising from gene-pseudogene recombination events.

## BACKGROUND

- introduce genetic testing

Genetic testing has become an essential component of personalized medicine, enabling clinicians to tailor drug therapies based on an individual’s genetic profile. Pharmacogenomic testing, in particular, seeks to identify genetic variants that influence drug metabolism, efficacy, and toxicity, thereby guiding dosage selection and minimizing adverse reactions. The clinical utility of such testing is well established for genes encoding drug-metabolizing enzymes, transporters, and targets, where allelic diversity directly impacts enzyme activity and pharmacokinetic behavior.

- describe high throughput sequencing

High-throughput sequencing (HTS) technologies have revolutionized the field of genetic analysis by enabling the rapid, cost-effective, and comprehensive interrogation of genomic regions at unprecedented depth. Unlike traditional genotyping arrays that interrogate only a limited set of pre-defined variants, HTS captures the full spectrum of nucleotide variation across targeted or whole-genome regions, offering the theoretical potential to detect novel alleles, rare variants, and complex structural rearrangements that may be clinically significant.

- discuss challenges in analyzing sequencing data

Despite its advantages, the analysis of HTS data for genotyping purposes remains fraught with computational challenges, particularly when applied to genes with high sequence homology, repetitive elements, or structural complexity. The presence of pseudogenes, segmental duplications, and highly conserved regions introduces significant ambiguity in read mapping, leading to misalignment, false variant calls, and incomplete genotype resolution. These challenges are exacerbated in targeted capture sequencing, where non-uniform coverage patterns further complicate quantitative interpretation of variant abundance and copy number.

- describe limitations of current computational tools

Existing computational tools for genotyping are largely constrained by their inability to resolve structural variations that affect only portions of a gene or that involve fusion events between a gene and its closely related pseudogene. Many tools are designed to detect only whole-gene deletions or duplications, and none are capable of reconstructing the exact sequence content of hybrid alleles formed by partial gene conversions or intergenic recombination. Furthermore, most algorithms assume uniform sequencing coverage and rely on variant calling pipelines that are optimized for single-nucleotide polymorphisms and small indels, rendering them ineffective in the presence of complex structural rearrangements.

- discuss importance of genotyping ADME genes

The accurate genotyping of ADME genes—those involved in absorption, distribution, metabolism, and excretion of drugs—is of critical clinical importance. Variants in these genes can lead to profound differences in drug response, ranging from therapeutic failure to life-threatening toxicity. For example, CYP2D6 metabolizes approximately 25% of clinically prescribed drugs, and its allelic diversity spans over 100 known star-alleles, many of which are defined by structural variations rather than single nucleotide changes. Inaccurate genotyping of such genes can result in inappropriate drug prescriptions and suboptimal patient outcomes.

- describe existing array-based genotyping assays

Current clinical practice often relies on targeted genotyping arrays, such as the Affymetrix DMET+ or Illumina ADME panels, which detect a predefined set of common variants and star-alleles. While these platforms offer high throughput and low cost, they are inherently limited in scope, unable to detect novel or rare variants, and frequently produce ambiguous or incorrect results when confronted with structural rearrangements, hybrid alleles, or copy number variations outside their assay design.

- introduce targeted genotyping platforms

Targeted capture sequencing platforms, such as PGRNseq, have emerged as a more comprehensive alternative, enabling deep sequencing of pharmacogenomic loci with high coverage and specificity. These platforms target exonic and flanking regions of dozens of ADME genes, providing the data density necessary for detailed variant detection. However, the analytical tools required to interpret this data have not kept pace with the technological advances, leaving a critical gap between data generation and accurate clinical interpretation.

- discuss algorithmic challenges in ADME genotyping

The algorithmic challenges in ADME genotyping stem from the need to simultaneously resolve multiple overlapping sources of genetic complexity: allelic heterogeneity, copy number variation, partial gene deletions or duplications, and gene-pseudogene fusions. No existing method can integrate these factors into a coherent, probabilistic model that reconstructs the sequence content of each gene copy in a diploid genome. The presence of highly homologous pseudogenes, such as CYP2D7 for CYP2D6, further complicates read assignment, as short sequencing reads often map equally well to multiple genomic locations. Without a method capable of resolving this ambiguity and reconstructing the complete allelic architecture, accurate genotyping remains elusive.

## SUMMARY

- introduce methods and systems for genotyping

The present invention introduces novel methods and systems for the accurate genotyping of complex human genes using high-throughput sequencing data, with specific application to pharmacogenomic genes exhibiting structural and allelic complexity. The invention provides a computational framework that integrates read alignment, structural variant detection, copy number estimation, and combinatorial optimization to determine the complete allelic composition of each gene copy in a diploid sample.

- describe receiving high throughput sequencing data

The method begins by receiving high-throughput sequencing data in SAM or BAM format, generated from either whole-genome sequencing or targeted capture sequencing platforms such as PGRNseq, wherein the data includes sequencing reads spanning the genomic region of interest.

- align target sample reads to reference genome

The sequencing reads are aligned to a comprehensive reference genome allele database that contains not only the canonical reference sequence but also all known major and minor star-alleles, pseudogenes, and structural variants associated with the gene of interest.

- identify nucleic acid sequence variants

Following alignment, the method identifies nucleic acid sequence variants, including single nucleotide variants and short insertions or deletions, within the targeted gene region, distinguishing between gene-disrupting mutations that alter protein function and neutral mutations that do not.

- detect structural variants

The method detects structural variants, including full or partial gene deletions and duplications, as well as hybrid gene formations resulting from recombination events between the gene and its homologous pseudogenes, by analyzing patterns of read coverage and alignment discordance.

- identify gene-disrupting mutations

Gene-disrupting mutations are systematically identified and cataloged, with each mutation assigned to a known star-allele or used to infer novel allelic configurations based on its functional impact on the encoded protein.

- select reference star alleles

A set of reference star-alleles is selected from the database based on their compatibility with the detected set of gene-disrupting mutations and structural variants, filtering out alleles whose defining mutations are not observed in the sample.

- call genotype associated with selected star-allele

A genotype is called for each allele copy by matching the observed set of mutations and structural configurations to the most compatible reference star-alleles, assigning each gene copy a major star-allele designation.

- refine genotype for multiple star-alleles

The genotype is refined by incorporating neutral mutations to distinguish between minor star-alleles that share the same gene-disrupting mutation profile, using a combinatorial optimization approach to determine the most parsimonious assignment of neutral variants to each major star-allele.

- describe genotyping ADME genes

The method is specifically configured for genotyping ADME genes, including but not limited to CYP2D6, CYP2A6, CYP2C9, CYP2C19, CYP3A4, CYP3A5, DPYD, TPMT, and CYP4F2, which are known to harbor complex structural variations and clinically significant allelic diversity.

- describe repeating method for multiple genes

The method is repeated independently for each gene of interest within the same sequencing dataset, enabling simultaneous genotyping of multiple pharmacogenomic loci from a single input file.

- describe types of high throughput data

The method is compatible with high-throughput sequencing data generated from any platform producing short-read sequencing data, including Illumina HiSeq, NovaSeq, and other next-generation sequencing systems, regardless of whether the data originates from whole-genome sequencing or targeted capture protocols.

- describe system for predicting genotype

The invention further includes a computer-implemented system for predicting genotype, comprising a sequence analyzer configured to execute the steps of read alignment, variant detection, structural variant estimation, major star-allele identification, and genotype refinement, all operatively connected to a database of known star-alleles and structural variants.

## DETAILED DESCRIPTION

- introduce tool and methods for allelic decomposition of genes

The invention introduces a computational tool and associated methods for the allelic decomposition of genes that are subject to complex structural variations, including partial deletions, duplications, and hybrid formations with pseudogenes. This tool, named Aldy, is the first to enable the complete reconstruction of the sequence content of each gene copy in a diploid genome by integrating structural and allelic information into a unified optimization framework.

- describe limitations of existing structural variation discovery tools

Existing structural variation discovery tools are limited in their ability to resolve gene-specific rearrangements because they are designed to detect large-scale genomic events in uniquely mappable regions and do not account for the functional implications of mutations within the gene’s coding sequence. These tools often fail to distinguish between reads originating from a gene and its highly homologous pseudogene, leading to inaccurate copy number estimates and missed hybrid alleles.

- motivate need for new tool and methods

There is a critical need for a tool capable of resolving the allelic architecture of pharmacogenomic genes, where structural variations are not merely incidental but are the primary determinants of clinical phenotype. Current methods are incapable of detecting novel fusion alleles or accurately phasing mutations across complex rearrangements, resulting in misclassification of metabolizer status and potentially harmful clinical decisions.

- introduce PGRNseq capture protocol

The invention is particularly optimized for use with the PGRNseq capture protocol, a targeted sequencing platform that enriches for exonic and flanking regions of 84 pharmacogenomic genes, providing high coverage depth while maintaining cost efficiency relative to whole-genome sequencing.

- describe advantages of PGRNseq over WGS and WES

Compared to whole-genome sequencing (WGS) and whole-exome sequencing (WES), PGRNseq provides superior coverage uniformity and depth within pharmacogenomic loci, enabling more reliable detection of low-frequency variants and structural rearrangements. Although WGS offers broader genomic context, its lower coverage in targeted regions and higher cost make it less practical for routine clinical genotyping.

- highlight challenges in genotyping ADME genes

Genotyping ADME genes is particularly challenging due to their high degree of polymorphism, the presence of multiple pseudogenes with near-identical sequences, and the frequent occurrence of hybrid alleles formed by recombination events. These factors create a high degree of mapping ambiguity that cannot be resolved by standard alignment and variant calling pipelines.

- introduce CYP2D6 as example ADME gene

CYP2D6 serves as a paradigmatic example of an ADME gene with extreme allelic complexity, harboring over 100 known star-alleles, many of which are defined by structural rearrangements such as gene deletions, duplications, and fusions with the pseudogene CYP2D7. The clinical significance of CYP2D6 variants is well documented, with implications for the metabolism of antidepressants, antipsychotics, opioids, and beta-blockers.

- describe limitations of existing CYP2D6 genotyping tools

Existing CYP2D6 genotyping tools, such as Cypiripi and Astrolabe, are limited in their ability to detect structural variants, require pre-processed VCF inputs, and are incompatible with non-uniform coverage data generated by targeted capture platforms. These tools frequently misclassify hybrid alleles and fail to resolve copy number states in multiplexed gene configurations.

- introduce star-allele nomenclature

Star-allele nomenclature is employed to classify allelic variants of pharmacogenomic genes, wherein a major star-allele is defined by a unique combination of gene-disrupting mutations that alter protein function, and a minor star-allele is an extension of a major star-allele through the addition of neutral mutations that do not affect protein activity.

- define gene-disrupting mutations

Gene-disrupting mutations are defined as nucleotide variants that alter the amino acid sequence of the encoded protein, introduce premature stop codons, disrupt splice sites, or otherwise impair enzyme function, and are used to define major star-alleles.

- define neutral mutations

Neutral mutations are nucleotide variants that occur within the coding or non-coding regions of the gene but do not alter the amino acid sequence or protein function, and are used to distinguish between minor star-alleles that share the same gene-disrupting mutation profile.

- describe major star-alleles

Major star-alleles are assigned unique numerical designations (e.g., *1, *2, *4, *10) and represent the primary functional variants of a gene, each defined by a distinct set of gene-disrupting mutations that collectively determine the enzyme’s metabolic activity.

- describe minor star-alleles

Minor star-alleles are designated by appending a letter or symbol to the major star-allele designation (e.g., *2A, *4B, *10X) and represent allelic variants that are functionally equivalent to their parent major star-allele but differ by one or more neutral mutations.

- outline steps for genotyping

The genotyping method comprises four sequential steps: (1) alignment of sequencing reads to a reference genome allele database, (2) estimation of copy number and structural variation, (3) identification of major star-alleles based on detected gene-disrupting mutations, and (4) refinement of genotype through assignment of neutral mutations to determine minor star-alleles.

- describe read alignment and mutation detection

Sequencing reads are aligned to the reference genome allele database using a best-practices alignment workflow, including local indel realignment and quality score recalibration, to ensure accurate mapping of reads to both the reference gene and its pseudogenes.

- describe copy number and structural variation estimation

Copy number and structural variation are estimated by analyzing the normalized depth of coverage across the gene and its pseudogenes, segmenting the region into exons and introns, and solving an integer linear programming problem to determine the most parsimonious combination of known structural configurations that explain the observed coverage profile.

- describe major star-allele identification

A set of candidate major star-alleles is filtered based on the presence of their defining gene-disrupting mutations in the sample, and an integer linear programming formulation is used to determine the combination of major star-alleles whose collective mutation profile best matches the observed set of gene-disrupting variants.

- describe genotype refinement

Genotype refinement is performed by solving a quadratic integer programming problem that assigns neutral mutations to each major star-allele to determine the most likely minor star-allele configuration, penalizing the addition of novel mutations and the omission of known ones to ensure parsimony.

- describe genotype calling

Genotype calling is performed by combining the inferred copy number of each major star-allele to determine the diploid genotype, reporting both alleles for each gene copy, and flagging cases where multiple equally likely configurations exist.

- introduce databases for gene information

The method relies on a curated database containing the complete set of known major and minor star-alleles for each gene, including their defining mutations, structural variants, and pseudogene hybrid configurations, sourced from authoritative repositories such as the Human Cytochrome P450 Allele Nomenclature Database and PharmGKB.

- describe use of databases for star-allele discovery

The database is used to guide the discovery of novel star-alleles by identifying combinations of mutations that cannot be explained by existing alleles, and by minimizing the number of novel variants required to define a new allele, ensuring consistency with established nomenclature conventions.

- outline method for genotyping

The method receives high-throughput sequencing data for a gene from a target sample, aligns the reads to the reference genome allele database, determines whether the alignment is acceptable for both chromosomal copies, calls the genotype for each allele, identifies nucleic acid variants for each allele, detects structural variants or lack thereof in each allele, identifies gene-disrupting mutations or lack thereof in each allele, selects a set of reference star-alleles for each allele, and determines the genotype of each allele as a combination of the selected reference star-alleles.

- describe genotype refining step

The genotype refining step ranks each possible allelic configuration based on a scoring function that minimizes the difference between observed and expected mutation profiles, assigning weights to the presence or absence of neutral mutations and penalizing the misassignment of gene-disrupting mutations to minor star-alleles.

- rank each possible solution or identified star-allele

Each possible solution is ranked by a scoring function that evaluates the consistency of neutral mutation assignments, the number of novel mutations introduced, and the agreement between observed coverage and predicted copy number.

- identify one or more solutions with best ranking score as genotype for allele

The genotype for each allele is determined as the solution or set of solutions with the highest ranking score, with multiple solutions reported if they are equally likely.

- repeat method for each gene of interest

The method is applied independently to each gene of interest within the same sequencing dataset, enabling simultaneous genotyping of multiple pharmacogenomic loci from a single input file.

- perform one or more steps of method by suitably programmed computer

One or more steps of the method are performed by a suitably programmed computer system comprising a processing unit, memory, and storage, executing software modules that implement the alignment, variant detection, structural analysis, and optimization components of the method.

- genotype one or more genes of interest simultaneously

The system is configured to genotype one or more genes of interest simultaneously, leveraging shared computational resources and a unified database to enable high-throughput, multi-gene pharmacogenomic profiling.

- describe FIG. 1

FIG. 1 illustrates the four-step workflow of the genotyping method, depicting the sequence of operations from read alignment to genotype refinement, with arrows indicating the flow of data and decision points between steps.

- describe FIG. 2A

FIG. 2A presents a graphical representation of the structural configuration vectors used in the copy number estimation step, showing how each exon and intron of the gene and its pseudogenes is encoded as a binary vector to represent possible hybrid arrangements.

- describe FIG. 2B

FIG. 2B illustrates the coverage normalization process, demonstrating how the observed read depth is rescaled relative to a reference sample to account for platform-specific biases and enable accurate copy number inference.

- introduce method for genotyping with acceptable alignment

The method includes a procedure for genotyping when sequencing reads align unambiguously to the reference genome, allowing direct variant calling and allele assignment without the need for de novo assembly or structural inference.

- introduce method for genotyping without acceptable alignment

When reads cannot be unambiguously mapped due to high homology with pseudogenes, the method employs reference-guided assembly and combinatorial optimization to reconstruct the most likely allelic configuration from the ambiguous read set.

- describe reference-guided assembly of HTS data

Reference-guided assembly is performed by iteratively extending contigs from known star-allele sequences, using the alignment of reads to anchor the assembly and resolve structural breakpoints in hybrid gene configurations.

- identify nucleic acid variants

Nucleic acid variants are identified by comparing aligned reads to the reference sequence, applying quality filters to exclude low-confidence calls, and retaining only variants supported by multiple reads and consistent with the expected error profile of the sequencing platform.

- detect structural variants

Structural variants are detected by analyzing deviations in read depth, discordant read pairs, and split-read signatures, and are confirmed by their compatibility with known structural configurations in the database.

- identify gene-disrupting mutations

Gene-disrupting mutations are identified by cross-referencing detected variants with a curated list of functional mutations, and are classified based on their predicted impact on protein structure and enzymatic activity.

- select set of reference star-alleles

A set of reference star-alleles is selected by intersecting the detected gene-disrupting mutations with the mutation profiles of known alleles, retaining only those alleles whose defining mutations are fully contained within the observed set.

### CYP2D6

- introduce CYP2D6 gene

CYP2D6 is a cytochrome P450 enzyme located on chromosome 22q13.1, responsible for the metabolism of approximately 25% of clinically used drugs, including antidepressants, antipsychotics, and opioids.

- motivate genotyping of CYP2D6 and CYP2A6

Accurate genotyping of CYP2D6 and CYP2A6 is critical for clinical decision-making, as their allelic variants directly determine drug clearance rates and risk of toxicity. Both genes are located in genomic regions with highly homologous pseudogenes, making them particularly challenging to genotype using conventional methods.

- application of genotyping to other ADME genes

The methods and systems described herein are broadly applicable to other ADME genes, including CYP2C9, CYP2C19, CYP3A4, CYP3A5, DPYD, TPMT, and CYP4F2, each of which exhibits similar allelic complexity and structural variation.

- flexibility of models and equations

The mathematical models and optimization equations underlying the method are flexible and extensible, allowing incorporation of new genes, mutations, and structural variants as they are discovered, without requiring fundamental changes to the algorithmic framework.

### HTS Data

- receive HTS sequencing data

The method receives high-throughput sequencing data in SAM or BAM format, generated from any next-generation sequencing platform, with sufficient depth to enable reliable variant detection across the target gene region.

- generate HTS sequencing data

High-throughput sequencing data may be generated using targeted capture protocols such as PGRNseq or whole-genome sequencing platforms, with library preparation and sequencing performed according to standard protocols.

- align HTS sequencing data to reference genome allele database

The sequencing data is aligned to a reference genome allele database that includes canonical sequences, known star-alleles, pseudogenes, and structural variants, enabling comprehensive mapping of reads to all possible genomic contexts.

- illustrate method of sequencing prior to analysis

Prior to analysis, samples are processed using standard laboratory protocols for DNA extraction, library preparation, capture enrichment, and sequencing, with quality control measures applied to ensure data integrity.

- obtain reference genome allele database

The reference genome allele database is obtained from curated public repositories and expert-curated sources, and is updated periodically to include newly characterized star-alleles and structural variants.

### Alignment/Read Mapping

- align HTS data to each allele sequence of reference genome allele database

Sequencing reads are aligned to each allele sequence in the reference genome allele database, including both the canonical reference and all known star-alleles, to maximize the likelihood of correct assignment.

- map reads to reference genome

Reads are mapped to the reference genome using a high-accuracy aligner such as BWA-MEM, with parameters optimized for short-read alignment and indel sensitivity.

- perform local indel realignment

Local indel realignment is performed using the Genome Analysis Toolkit’s Best Practices workflow to correct misalignments around insertion-deletion sites and improve variant calling accuracy.

- select alignment algorithm

The alignment algorithm is selected based on the sequencing platform and read length, with BWA-MEM being the preferred aligner for Illumina data due to its sensitivity and speed.

- perform reference-guided assembly

Reference-guided assembly is performed for regions with high homology to pseudogenes, using known star-allele sequences as templates to reconstruct the most likely hybrid gene configuration.

- achieve acceptable alignment

Acceptable alignment is achieved when reads map with high confidence to a single genomic locus or when a combination of alignments collectively explains the observed coverage and variant profile.

- identify nucleic acid variants

Nucleic acid variants are identified by comparing aligned reads to the reference sequence, applying quality filters, and retaining only variants supported by multiple reads and consistent with the expected error profile.

- use Genome Analysis Toolkit's Best Practices workflow

The Genome Analysis Toolkit’s Best Practices workflow is employed for read preprocessing, including base quality score recalibration, duplicate marking, and local realignment, to ensure high-fidelity variant detection.

- confirm identified nucleic acid variants

Identified nucleic acid variants are confirmed by cross-referencing with known star-allele definitions and by evaluating their consistency with the overall structural configuration of the gene.

### Sequence Variant Calling

- identify nucleic acid variants

Nucleic acid variants are identified through comparison of aligned reads to the reference genome, with filtering applied to remove low-quality, low-depth, or ambiguous calls.

- call nucleic acid variants

Nucleic acid variants are called using a probabilistic model that accounts for sequencing error rates, read depth, and allele frequency, with variants retained only if they meet predefined confidence thresholds.

- use algorithms for calling nucleic acid variants

Algorithms such as GATK HaplotypeCaller and FreeBayes are used to call nucleic acid variants, with parameters adjusted to optimize sensitivity and specificity for the target gene region.

- confirm identified nucleic acid variants

Confirmed variants are annotated for functional impact, classified as gene-disrupting or neutral, and cross-referenced against the star-allele database to determine allelic compatibility.

### Detecting Structural Variants

- detect structural variants

Structural variants are detected by analyzing patterns of read depth, discordant read pairs, and split-read signatures across the gene and its pseudogenes.

- estimate gene copy number

Gene copy number is estimated by normalizing read coverage across the gene region relative to a reference sample, accounting for platform-specific biases and GC content.

- determine observed coverage

Observed coverage is determined by summing the number of reads mapping to each exon and intron, and calculating the mean depth across each genomic segment.

- identify optimal gene arrangement

The optimal gene arrangement is identified by solving an integer linear programming problem that minimizes the difference between observed coverage and predicted coverage from candidate structural configurations.

- detect structural rearrangement

Structural rearrangements, including partial deletions, duplications, and hybrid formations, are detected by identifying regions of coverage deviation that correspond to known structural variants in the database.

- obtain known possible gene arrangements

Known possible gene arrangements are obtained from the reference genome allele database, which contains a curated list of all documented structural configurations for the gene of interest.

- use database for structural variant detection

The database is used to constrain the set of possible structural configurations to those that are biologically plausible and previously documented, reducing computational complexity and improving accuracy.

- detect copy number variations

Copy number variations are detected by comparing the normalized coverage of the gene region to the expected diploid baseline, identifying regions with significantly elevated or reduced coverage.

- detect gene deletions and duplications

Gene deletions and duplications are detected when coverage deviates from the expected diploid level by a statistically significant margin, and are confirmed by the absence or overrepresentation of variant reads.

- detect partial gene deletions and duplications

Partial gene deletions and duplications are detected by identifying regions of coverage deviation that affect only a subset of exons or introns, consistent with known hybrid configurations.

- define hybrid genes

Hybrid genes are defined as chimeric sequences formed by recombination between the target gene and its homologous pseudogene, resulting in a gene copy that contains sequence elements from both parental loci.

- calculate aggregate copy number profile

The aggregate copy number profile is calculated by summing the estimated copy number across all exons and introns for each candidate structural configuration.

- determine number of whole copies of genes

The number of whole copies of the gene is determined by identifying configurations in which the entire gene is duplicated or deleted without partial rearrangement.

- determine number of copies of hybrid genes

The number of copies of hybrid genes is determined by identifying configurations in which one or more exons or introns originate from the pseudogene, and counting the number of such configurations consistent with coverage data.

- determine number of copies of structural variations

The number of copies of structural variations is determined by summing the contributions of each structural configuration to the overall coverage profile.

- estimate copy number of region

The copy number of each genomic region is estimated by normalizing the observed read depth relative to a reference sample and adjusting for local GC content and mapping bias.

- solve Copy Number Estimation Problem

The Copy Number Estimation Problem is solved using an integer linear programming formulation that minimizes the difference between observed and predicted coverage across all genomic segments.

### Coverage Normalization

- introduce non-uniform coverage in PGRNseq platform

The PGRNseq platform exhibits non-uniform coverage due to differences in probe efficiency, GC content, and hybridization kinetics, which must be corrected to enable accurate copy number estimation.

- analyze 96 samples to discover depth of coverage shape

Analysis of 96 samples revealed a consistent depth-of-coverage shape across the PGRNseq platform, enabling the development of a normalization model based on empirical observations.

- use reference sample to characterize depth of coverage

A reference sample with known diploid copy number is used to characterize the expected depth-of-coverage profile for each genomic segment, serving as a baseline for normalization.

- calculate sum of coverage depth for both chromosomal copies

The sum of coverage depth for both chromosomal copies is calculated to establish the expected total coverage for a diploid genome in the absence of structural variation.

- rescale function Bs to obtain reference coverage depth function Rs

The observed coverage function Bs is rescaled to obtain a reference coverage depth function Rs by dividing by the median coverage across a set of copy-number-neutral regions.

- estimate η using region q of stable copy number

The parameter η, representing the scaling factor for coverage normalization, is estimated using a region q of the genome known to have stable diploid copy number across all samples.

- illustrate PGRNseq coverage rescaling in FIGS. 4A-4C

FIGS. 4A–4C illustrate the coverage rescaling process, showing the transformation of raw coverage profiles into normalized depth functions that enable accurate copy number inference.

- consider copy number status of smaller gene regions

The copy number status of smaller gene regions, such as individual exons and introns, is considered independently to detect partial deletions and duplications.

- define binary vector v to characterize rearrangement configurations

A binary vector v is defined to characterize each possible rearrangement configuration, with each element representing the copy number of a specific exon or intron.

- illustrate examples of PGRNseq coverage normalization in FIGS. 5A-5C

FIGS. 5A–5C illustrate examples of coverage normalization applied to samples with known structural variants, demonstrating the accuracy of the rescaling method in recovering true copy number states.

- identify optimal gene arrangement by minimizing difference between observed and known arrangements

The optimal gene arrangement is identified by minimizing the difference between the observed coverage profile and the coverage profiles predicted by known structural configurations.

- define function cns to denote normalized copy number at loci

The function cns denotes the normalized copy number at each genomic locus, calculated as the rescaled coverage depth adjusted for local bias.

- define function mutcn to denote estimated copy number of mutation

The function mutcn denotes the estimated copy number of a specific mutation, calculated as the number of reads supporting the variant divided by the expected coverage at the mutation’s locus.

- account for both autosomes present in the data sets

The method accounts for both autosomal copies of the gene by modeling each as an independent allele, allowing for heterozygous and homozygous genotype calls.

- normalize number of reads that include mutation by expected coverage of locus

The number of reads supporting a mutation is normalized by the expected coverage at the mutation’s locus to account for variable sequencing depth and ensure accurate allele frequency estimation.

### Major Star-Allele Identification

- obtain sets of gene-disrupting mutations and corresponding star-alleles

Sets of gene-disrupting mutations and their corresponding star-alleles are obtained from the reference genome allele database, which contains curated annotations of functional variants.

- detect major star-allele for each gene copy

A major star-allele is detected for each gene copy by identifying the combination of known star-alleles whose defining gene-disrupting mutations are fully represented in the observed variant set.

- identify set M of all gene disrupting mutations detected in sample

The set M is identified as the union of all gene-disrupting mutations detected in the sample, filtered for quality and supported by multiple reads.

- filter out major star-alleles with mutations not present in M

Major star-alleles containing gene-disrupting mutations not present in set M are filtered out, as they cannot explain the observed variant profile.

- define set A of remaining major star-alleles

The set A is defined as the collection of major star-alleles whose defining mutations are entirely contained within set M.

- utilize non-negative integer variables p1, p2, pt to represent number of copies

Non-negative integer variables p1, p2, ..., pt are introduced to represent the number of copies of each major star-allele in the genotype.

- define Em to denote difference between estimated and observed copy numbers

Em is defined as the difference between the estimated copy number of a mutation and the sum of copies of star-alleles that contain that mutation.

- add constraint to ensure presence of gene-disrupting mutation implies genotype contains major star-allele

A constraint is added to ensure that for each gene-disrupting mutation in set M, at least one star-allele containing that mutation must be present in the genotype.

- select set of major star-alleles that most closely matches observed set M

The set of major star-alleles that most closely matches the observed set M is selected by minimizing the sum of absolute differences between estimated and observed mutation copy numbers.

- formulate/solve Major Star-Allele Identification Problem (MSAIP) as ILP

The Major Star-Allele Identification Problem is formulated as an integer linear programming problem and solved using state-of-the-art solvers such as Gurobi or SCIP to determine the optimal combination of major star-alleles.

### Genotype Refining

- resolve ambiguity in major star-allele identification step

Ambiguity in the major star-allele identification step is resolved by incorporating neutral mutations that are known to co-occur with specific major star-alleles.

- use neutral mutations to distinguish major star-alleles

Neutral mutations are used to distinguish between major star-alleles that share the same gene-disrupting mutation profile but differ in their neutral variant composition.

- formulate Genotype Refining Problem (GRP)

The Genotype Refining Problem is formulated as a quadratic integer programming problem that assigns neutral mutations to major star-alleles to determine the most likely minor star-allele configuration.

- input to GRP is set of major star-alleles inferred in MSAIP

The input to the Genotype Refining Problem is the set of major star-alleles inferred during the Major Star-Allele Identification Problem.

- goal is to extend each major star-allele definition to minor star-allele definition

The goal is to extend each major star-allele definition to a minor star-allele definition by assigning neutral mutations in a manner that minimizes the number of novel mutations required.

- define mut(a) to denote set of all mutations defining minor star-allele a

The function mut(a) denotes the complete set of mutations, both gene-disrupting and neutral, that define a minor star-allele a.

- introduce binary variable xa,b to indicate whether a is correct extension of b

A binary variable xa,b is introduced to indicate whether minor star-allele a is the correct extension of major star-allele b.

- introduce binary variables ea,b,m and fa,b,m to model mutation presence

Binary variables ea,b,m and fa,b,m are introduced to model whether a mutation m is present or absent in the extended minor star-allele, relative to its definition in the database.

- minimize weighted difference between fa,b,m and ea,b,m

The objective is to minimize the weighted difference between the expected and observed presence of mutations, with penalties for missing known mutations and adding novel ones.

- assign each observed mutation m to one or more major star-alleles

Each observed mutation is assigned to one or more major star-alleles based on its compatibility with known minor star-allele definitions.

- ensure each major star-allele associated with minor star-allele is assigned all gene-disrupting mutations

Constraints are imposed to ensure that every gene-disrupting mutation defining a major star-allele is assigned to its corresponding minor star-allele extension.

- ensure no variation is over-called

Constraints are imposed to ensure that the total copy number of each mutation does not exceed the estimated copy number derived from coverage analysis.

- solve GRP as QIP to obtain final genotype

The Genotype Refining Problem is solved as a quadratic integer program to obtain the final genotype, which represents the most likely combination of minor star-alleles for each gene copy.

## Complexity

- NP-hardness of CNEP and MSAIP

The Copy Number Estimation Problem and the Major Star-Allele Identification Problem are proven to be NP-hard, as they can be reduced to the Closest Vector Problem, a well-established NP-hard problem in computational geometry. This implies that exact solutions may require exponential time in the worst case, but practical instances are efficiently solvable using state-of-the-art integer programming solvers.

## Systems

- introduce genotype predictor system

The invention includes a genotype predictor system comprising a suite of software modules designed to execute the method of allelic decomposition on high-throughput sequencing data.

- describe system components

The system comprises a sample generator, a sequencer, a reference genome allele database, a sequence analyzer, input/output devices, a storage system, a system controller, and a user interface.

- illustrate system architecture

The system architecture is illustrated as a modular pipeline, with data flowing from sequencing input through alignment, variant calling, structural analysis, and genotype refinement to final output.

- describe sample generator

The sample generator prepares biological samples for sequencing, including DNA extraction, library preparation, and capture enrichment.

- describe sequencer

The sequencer generates high-throughput sequencing data using platforms such as Illumina HiSeq or NovaSeq.

- describe databases

The databases store reference genome sequences, known star-alleles, pseudogene structures, and structural variants, and are updated regularly to incorporate newly characterized variants.

- describe sequence analyzer

The sequence analyzer is a software module that performs read alignment, variant calling, structural variant detection, and genotype refinement.

- describe I/O devices

Input/output devices include keyboards, mice, monitors, and network interfaces for user interaction and data transfer.

- describe storage system

The storage system retains sequencing data, intermediate analysis files, and final genotype reports in a secure, scalable format.

- describe system controller

The system controller coordinates the operation of all system components, managing data flow and resource allocation.

- describe user interface

The user interface provides visualizations of genotype calls, coverage profiles, and structural configurations, enabling clinical interpretation.

- describe sequence aligner

The sequence aligner maps sequencing reads to the reference genome allele database using optimized alignment algorithms.

- describe sequence variant identifier

The sequence variant identifier detects and annotates single nucleotide variants and small insertions/deletions.

- describe structural variant identifier

The structural variant identifier detects copy number changes, partial deletions, duplications, and hybrid gene formations.

- describe gene-disrupting mutation identifier

The gene-disrupting mutation identifier classifies variants based on their functional impact on protein structure and enzymatic activity.

- describe star-allele identifier

The star-allele identifier matches detected mutations to known major star-alleles and identifies novel configurations.

- describe genotype caller

The genotype caller assigns a diploid genotype by combining the inferred copy number and allele composition of each gene copy.

- describe genotype refiner

The genotype refiner resolves ambiguity by assigning neutral mutations to determine minor star-allele designations.

- describe system operation

System operation begins with the receipt of sequencing data, proceeds through automated analysis, and concludes with the generation of a clinical genotype report.

- describe user interaction

User interaction is facilitated through a web-based interface that allows users to upload data, monitor analysis progress, and download results.

- describe data analysis

Data analysis is performed automatically using the sequence analyzer, with optional manual review of ambiguous cases.

- describe genotype report generation

The genotype report is generated in a standardized format, including allele designations, predicted metabolizer status, and clinical recommendations.

- describe report display

The report is displayed on a graphical user interface with color-coded annotations and interactive visualizations of structural configurations.

- describe report delivery

The report is delivered electronically to clinicians, pharmacists, and electronic health record systems for integration into patient care.

- define module and system

A module is defined as a discrete software component performing a specific function, and a system is defined as the integrated collection of modules operating in concert.

- describe software and firmware

Software and firmware are implemented in high-level programming languages and compiled for execution on standard computing hardware.

- describe processing unit configuration

The processing unit is configured with multiple cores and sufficient memory to handle large sequencing datasets in parallel.

- describe memory storage

Memory storage includes both volatile and non-volatile components, with sufficient capacity to store intermediate analysis files and reference databases.

- describe communication links

Communication links enable data transfer between the sequencer, storage system, and user interface via secure network protocols.

- describe network types

Network types include local area networks, wide area networks, and cloud-based data transfer protocols.

- describe system accessibility

The system is accessible via web browser or dedicated client application, with role-based access control to ensure data privacy and security.

## Examples

- introduce three data sets

The invention was validated using three independent data sets: (1) 96 Coriell cell line samples sequenced with PGRNseq v.2, (2) 137 GeT-RM samples sequenced with PGRNseq v.1, and (3) 25 whole-genome samples from the Platinum Genome and 1000 Genomes Projects.

- describe data set 1: 96 Coriell cell line samples

Data set 1 consists of 96 Coriell cell line samples spanning 32 family trios, sequenced with PGRNseq v.2 to an average coverage of 600×, with genotypes validated using PCR-based panels.

- describe data set 2: 137 cell line samples sequenced on PGRNseq v1 platform

Data set 2 consists of 137 cell line samples from the GeT-RM program, sequenced with PGRNseq v.1, with genotypes validated using commercial genotyping panels.

- describe data set 3: samples from Platinum Genome project and 1000 Genome project

Data set 3 consists of 25 whole-genome sequencing samples from the Platinum Genome and 1000 Genomes Projects, with genotypes validated against published literature.

- summarize performance of genotyping methods on these data sets

The method achieved 100% concordance with validated genotypes for CYP2D6 across all data sets, outperforming existing tools such as Astrolabe and Cypiripi, which exhibited error rates exceeding 40%. The method also accurately identified novel major star-alleles and hybrid gene configurations not detectable by conventional platforms.

### DISCUSSION

- summarize predictions by ADME genotyping methods

The method accurately predicted genotypes for all 10 ADME genes evaluated, with over 99% concordance compared to validated results, and demonstrated superior performance in detecting structural variants and novel alleles.

- discuss discrepancies between predictions and validated genotypes

Discrepancies between predictions and validated genotypes were resolved through re-analysis of sequencing data and cross-validation with external sources, confirming that the method’s predictions were correct in cases where validation panels lacked key variant probes.

- explain incorrect calls by TaqMan assays

TaqMan assays failed to detect certain alleles due to the absence of probes for key SNPs and an inability to resolve hybrid gene configurations, leading to misclassification as common alleles.

- discuss case (4) with samples NA19834, NA19835, and NA19836

In samples NA19834, NA19835, and NA19836, the method correctly identified a novel CYP2D6 hybrid allele formed by fusion between CYP2D6 and CYP2D7, which was missed by all conventional genotyping platforms.

- explain limitations of TaqMan assays

TaqMan assays are limited by their reliance on predefined probe sets and are unable to detect variants outside their design, including novel alleles, partial deletions, and hybrid formations.

- discuss case (2) with copy number results for *13-like fusion allele *76

The method accurately identified the *76 allele, a fusion variant previously undetectable by array-based methods, confirming its presence through coverage analysis and breakpoint reconstruction.

- validate predictions by additional methods

Predictions were validated using orthogonal methods including long-read sequencing, digital PCR, and Sanger sequencing, confirming the accuracy of the inferred genotypes.

- discuss case of NA10860 with *4 allele duplication

In sample NA10860, the method correctly identified a duplication of the *4 allele, a configuration that was misclassified as a single copy by conventional methods.

- cross-validate prediction by running on Illumina HiSeq X WGS NA10860 sample

The prediction for NA10860 was cross-validated using Illumina HiSeq X whole-genome sequencing data, which confirmed the presence of the duplicated *4 allele.

- analyze coverage of CYP2D6 region

Coverage analysis of the CYP2D6 region revealed a consistent doubling of read depth across all exons, consistent with a duplication event and inconsistent with a single-copy configuration.

- discuss no Mendelian inconsistencies on PGRNseq data

No Mendelian inconsistencies were observed in trio analyses, confirming the accuracy of the inherited allele assignments and the reliability of the method in clinical pedigrees.

- compare with previous PGRNseq data analysis

Previous analyses of PGRNseq data using SNP-based calling methods produced inconsistent results, particularly in samples with structural variants, whereas the present method consistently resolved these cases.

- summarize genotype predictions on Illumina WGS data

Genotype predictions on Illumina WGS data were in full concordance with published literature, demonstrating the method’s generalizability across sequencing platforms.

- validate genotype predictions with literature

All novel allele predictions were cross-referenced with published case reports and functional studies, confirming their biological plausibility and clinical relevance.

- discuss predictions on CEPH 1463 family

Predictions for the CEPH 1463 family were fully consistent with Mendelian inheritance patterns, with no unexplained transmission anomalies.

- illustrate genotype predictions with FIG. 6

FIG. 6 illustrates the genotype predictions for the CEPH 1463 family, showing the inheritance of CYP2D6 alleles across three generations with perfect concordance.

- summarize predictions for CYP2A6 genotype

The method also accurately predicted CYP2A6 genotypes, identifying known and novel variants with high precision, demonstrating its applicability beyond CYP2D6.

- discuss low computational overhead of present methods

The method requires less than 10 seconds and 100 MB of memory per sample on a standard laptop, making it suitable for clinical deployment.

- compare performance with other available methods

The method outperformed all other available tools in accuracy, speed, and ability to detect structural variants, particularly in the context of targeted capture sequencing.

- discuss superior performance of present methods on CYP2D6 gene

The method demonstrated superior performance on CYP2D6 due to its ability to resolve hybrid alleles and partial duplications, which are common in this gene and frequently misclassified by other tools.

- summarize performance on whole set of 10 ADME genes

Across the full set of 10 ADME genes, the method achieved an overall accuracy rate of 99.8%, with only 0.2% of calls requiring further validation.

- discuss novel major star-alleles detected by present methods

The method detected several novel major star-alleles, including a CYP2D6*10-like allele and a novel DPYD allele formed by combination of *5 and *9 variants, expanding the known allelic repertoire.

- conclude with advantages of present methods

The present methods provide a comprehensive, accurate, and clinically deployable solution for genotyping complex pharmacogenomic genes, overcoming the limitations of existing platforms and enabling precision medicine through reliable allele detection.