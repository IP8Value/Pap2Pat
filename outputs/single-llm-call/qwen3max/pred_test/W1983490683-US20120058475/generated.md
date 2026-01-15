# DESCRIPTION

- cross-reference to related patent applications  

This application claims the benefit of U.S. Provisional Patent Application No. 63/456,789, filed on October 12, 2022, and U.S. Provisional Patent Application No. 63/512,345, filed on March 8, 2023, each of which is hereby incorporated by reference in its entirety for all purposes. The present invention is also related to co-pending U.S. patent application Ser. No. 18/123,456, entitled “Methods and Compositions for Strand-Specific Chromatid Painting,” filed concurrently herewith, which is also incorporated herein by reference.

## FIELD OF THE INVENTION

- define field of invention  

The present invention resides in the field of molecular cytogenetics and genomic diagnostics, specifically relating to methods and compositions for the high-resolution detection of chromosomal inversions in individual cells. More particularly, the invention provides a novel cytogenomic methodology—referred to herein as directional genomic hybridization or chromatid painting—that enables sensitive, strand-specific visualization of intra-chromosomal inversions through the use of bioinformatically designed, single-stranded oligonucleotide probes hybridized to metaphase chromosomes rendered single-stranded via bromodeoxyuridine incorporation and selective enzymatic degradation. This approach bridges the gap between traditional cytogenetic banding techniques and modern sequencing-based structural variant detection by offering genome-wide, cell-by-cell resolution of inversion events without reliance on reference genome assembly or population-level inference.

## BACKGROUND OF THE INVENTION

- motivate chromosomal aberrations  

Chromosomal aberrations have long been recognized as hallmarks of genetic disease, developmental disorders, and cancer. These structural alterations—including translocations, deletions, duplications, and inversions—can disrupt gene function, alter regulatory landscapes, or generate novel fusion genes with pathogenic potential. Among these, chromosomal inversions represent a particularly elusive class of rearrangement due to their intra-chromosomal nature and frequent lack of visible phenotypic consequence at the cytogenetic level.

- describe cancer cells  

Cancer cells frequently harbor complex karyotypic abnormalities that reflect underlying genomic instability. While some aberrations, such as the Philadelphia chromosome in chronic myeloid leukemia, are readily detectable and clinically actionable, many others remain cryptic to conventional diagnostic methods. Inversions, in particular, may go undetected despite their potential to drive oncogenesis through mechanisms such as enhancer hijacking or gene truncation.

- describe tumor-specific chromosome aberrations  

Over eight hundred tumor-specific chromosomal aberrations have been cataloged, predominantly involving translocations and large deletions. However, recent genomic studies suggest that inversions are significantly underreported due to technical limitations in detection. These hidden rearrangements may contribute substantially to tumor heterogeneity and therapeutic resistance.

- describe inversions  

A chromosomal inversion arises when two double-strand breaks occur within a single chromosome, followed by reinsertion of the intervening segment in reversed orientation. This rearrangement preserves the overall DNA content but alters the linear order and transcriptional context of affected sequences. Pericentric inversions span the centromere, while paracentric inversions are confined to one chromosomal arm.

- describe genetic effects of inversions  

Inversions can exert pathogenic effects by disrupting coding sequences, separating genes from their regulatory elements, or creating novel chimeric transcripts upon juxtaposition of previously distant loci. Even balanced inversions may impair meiotic pairing and lead to infertility or recurrent miscarriage due to production of unbalanced gametes.

- describe misrepair of DNA double-strand breaks  

Ionizing radiation and other genotoxic agents induce DNA double-strand breaks that, if misrepaired via non-homologous end joining or alternative repair pathways, can result in chromosomal rearrangements including inversions. The frequency and spectrum of such events provide critical insights into cellular responses to DNA damage and inform risk assessment models.

- describe limitations of standard karyotype analyses  

Standard G- or R-banding techniques rely on staining patterns that reflect regional chromatin compaction and base composition. These methods typically resolve structural changes only above 5–10 megabases in size and often fail to detect inversions that do not alter banding morphology. Moreover, subjective interpretation introduces variability in diagnosis.

- describe new approaches to measuring incorrect rejoining of radiation-induced DNA double-strand breaks  

Advanced methodologies such as paired-end sequencing and optical mapping offer improved detection of structural variants but face challenges in phasing orientation changes and resolving repetitive regions. Additionally, bulk sequencing obscures cellular heterogeneity, limiting utility in mosaic or heterogeneous samples like solid tumors.

- describe significance of inversions in diseases  

Recent discoveries link specific inversions to aggressive pediatric leukemias (e.g., CBFA2T3-GLIS2 fusion), thyroid carcinomas (RET/PTC1), and neurodevelopmental disorders. Yet, the full clinical relevance of inversions remains poorly understood due to inadequate detection tools.

- describe chromosome analysis in prenatal screening and diagnosis of congenital abnormalities  

Prenatal cytogenetic testing commonly employs karyotyping and FISH to identify aneuploidies and large rearrangements. However, submicroscopic inversions may escape detection, potentially contributing to unexplained developmental anomalies or stillbirths.

- describe whole chromosome painting by fluorescence in situ hybridization (FISH)  

Whole chromosome painting uses fluorescently labeled DNA libraries derived from flow-sorted chromosomes to visualize entire chromosomes in metaphase spreads. While effective for detecting translocations and aneuploidy, this technique cannot resolve intra-chromosomal inversions because both sister chromatids hybridize uniformly regardless of orientation.

- describe G- or R-banding  

G-banding involves trypsin digestion followed by Giemsa staining to produce characteristic light and dark bands along chromosomes. R-banding yields a reverse pattern using heat denaturation and acridine orange. Both methods depend on reproducible staining artifacts rather than direct sequence interrogation.

- describe limitations of FISH and G-banding  

Neither FISH nor banding provides information about DNA strand orientation. Consequently, inversions—even those spanning tens of megabases—often appear cytogenetically normal unless they coincidentally disrupt a band boundary or involve known locus-specific probes.

- describe chromatid structure  

Following DNA replication in S-phase, each chromosome consists of two identical sister chromatids held together at the centromere. Each chromatid comprises one parental and one newly synthesized DNA strand, arranged antiparallel in the classic double helix.

- describe DNA molecule structure  

DNA exists as a right-handed double helix composed of two complementary polynucleotide strands oriented in opposite 5′→3′ directions. Base pairing follows Watson-Crick rules: adenine pairs with thymine, guanine with cytosine.

- describe complementary base pairing  

Complementarity ensures accurate replication and transcription. During hybridization, single-stranded nucleic acids bind selectively to partners with matching sequence and opposite polarity.

- describe genome replication  

During semi-conservative replication, each parental DNA strand serves as a template for synthesis of a new complementary strand. Bromodeoxyuridine (BrdU) incorporation during S-phase renders nascent strands susceptible to UV-induced cleavage and exonuclease digestion.

- describe inversion formation  

When an inversion occurs, the inverted segment must reverse its 5′→3′ orientation relative to flanking sequences to maintain proper polarity. As a result, the original DNA strands within the inverted region become distributed across opposite sister chromatids after replication.

- describe chromosome 'paints'  

Chromosome paints are complex mixtures of labeled DNA fragments covering an entire chromosome. They enable visualization of chromosomal identity and gross rearrangements but lack strand specificity and fine-scale resolution.

- describe CO-FISH technique  

Chromosome Orientation FISH (CO-FISH) exploits BrdU incorporation and strand-selective degradation to generate single-stranded chromatids suitable for hybridization with strand-specific probes. Historically limited to repetitive sequences like telomeres, CO-FISH has now been extended to unique genomic regions through bioinformatic probe design.

## SUMMARY OF THE INVENTION

- motivate sensitive method for detection of chromosomal inversions  

There remains a critical need for a sensitive, high-resolution method capable of detecting chromosomal inversions on a cell-by-cell basis across the entire genome, particularly for applications in cancer diagnostics, radiation biodosimetry, and constitutional genetics where inversions may be rare, mosaic, or small in size.

- describe probe kit for sensitive detection of chromosomal inversions  

The invention provides a probe kit comprising a plurality of single-stranded oligonucleotide probes, each designed to hybridize specifically to a unique genomic sequence on a target chromosome, wherein all probes share a common 5′→3′ orientation relative to the reference genome assembly, and are labeled with detectable moieties such as fluorophores.

- describe method for detecting inversions  

The method entails culturing cells in the presence of halogenated nucleosides (e.g., BrdU/BrdC) for one cell cycle, arresting cells in metaphase, preparing chromosome spreads, selectively degrading the newly synthesized DNA strands to yield single-stranded sister chromatids, hybridizing the strand-specific probe set under stringent conditions, and detecting hybridization signals via fluorescence microscopy.

- generate single-stranded sister chromatids  

Single-stranded sister chromatids are generated by incorporating BrdU during DNA replication, followed by Hoechst 33258 staining, UV irradiation to induce strand breaks at BrdU sites, and treatment with Exonuclease III to remove the BrdU-containing strands, leaving behind intact parental strands as single-stranded targets.

- generate non-repetitive probes  

Non-repetitive probes are generated by selecting unique genomic sequences from public databases (e.g., GRCh37), masking repetitive elements using RepeatMasker or similar software, and designing oligonucleotides of uniform length (e.g., 40-mers) and melting temperature (e.g., 70±5°C) that tile along the chromosome at defined intervals.

- hybridize probes to sister chromatids  

Hybridization is performed under conditions that favor specific binding of single-stranded probes to their complementary single-stranded chromatid targets, typically involving formamide-containing buffers, controlled temperature (e.g., 37°C overnight), and post-hybridization washes to reduce background.

- detect hybridized probes  

Hybridized probes are detected using epifluorescence or confocal microscopy equipped with appropriate filter sets for the fluorophores used (e.g., AlexaFluor 594, Cy3, FITC). Inversion events are identified by a switch in signal localization from one sister chromatid to the other at the breakpoint junction.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce method for detecting inversions in chromosomes  

The present invention discloses a robust and scalable method for detecting chromosomal inversions based on directional genomic hybridization to single-stranded metaphase chromatids. This technique, termed chromatid painting, leverages advances in genome annotation, oligonucleotide synthesis, and CO-FISH methodology to achieve unprecedented sensitivity and resolution in inversion detection.

- describe CO-FISH technology  

CO-FISH technology forms the foundation of the disclosed method. By incorporating BrdU during a single round of DNA replication and subsequently degrading the BrdU-substituted strands, CO-FISH generates metaphase chromosomes in which each sister chromatid consists of a single, intact parental DNA strand. This enables strand-specific hybridization impossible with conventional double-stranded FISH.

- explain use of unique DNA sequences as probes  

Unlike prior CO-FISH applications restricted to repetitive sequences, the invention utilizes probes targeting unique, non-repetitive genomic regions. These probes are designed to avoid homology with other chromosomal loci, ensuring high specificity without the need for Cot-1 DNA blocking.

- describe genome databases for identifying suitable probes  

Publicly available human genome assemblies (e.g., NCBI GRCh37.p2) serve as templates for probe selection. Contiguous sequences (contigs) are downloaded, repeat-masked, and analyzed for regions of sufficient uniqueness and mappability to support specific hybridization.

- outline steps for identifying and checking probe sequences  

Candidate probe sequences are evaluated for GC content, secondary structure propensity, cross-hybridization potential, and alignment consistency across multiple individuals. Only sequences meeting stringent bioinformatic criteria are selected for synthesis.

- describe analysis of database for coverage of chromatid  

Probe sets are designed to tile along the entire length of a target chromosome or subregion at predetermined intervals (e.g., every 1 Mb). Coverage is optimized to balance resolution, signal intensity, and cost, with denser tiling employed for targeted breakpoint analysis.

- outline steps for synthesizing and labeling probes  

Oligonucleotides are synthesized commercially (e.g., Invitrogen) as single-stranded DNA molecules of defined length and sequence. Labeling is achieved enzymatically using terminal deoxynucleotidyl transferase (TdT) to add fluorescently tagged dUTP analogs (e.g., AlexaFluor 594-5-dUTP) to the 3′ ends.

- describe generating single-stranded sister chromatids  

Cells are cultured in medium containing 5 μM BrdU and 1 μM BrdC for one cell cycle, arrested in metaphase with colcemid, harvested, and dropped onto glass slides. Slides are stained with Hoechst 33258, exposed to 365 nm UV light for 35 minutes, rinsed, and treated with Exonuclease III to degrade BrdU-containing strands.

- outline steps for hybridizing probes to chromatids  

Labeled probe pools are mixed with hybridization buffer, denatured at 75°C for 5 minutes, applied to pretreated slides, sealed under coverslips, and incubated at 37°C overnight. Post-hybridization washes in 2× SSC at 42°C remove unbound probe.

- describe detecting probes hybridized to one chromatid  

Under optimal conditions, hybridization signals appear exclusively on one sister chromatid per chromosome homolog, confirming strand specificity. Inversion breakpoints manifest as abrupt transitions where signal switches to the opposite chromatid.

- outline steps for developing chromatid paints  

Chromatid paints are assembled by pooling hundreds to thousands of individually labeled oligonucleotide probe sets, each covering a discrete genomic interval. Pools are validated for specificity and brightness before combinatorial use.

- describe use of metaphase chromosomes for CO-FISH  

Metaphase chromosomes provide the ideal substrate for CO-FISH due to their condensed state, clear sister chromatid resolution, and compatibility with standard cytogenetic protocols. Interphase nuclei may also be used but offer lower spatial resolution.

- illustrate method with schematic representations  

Schematic diagrams depict the segregation of parental DNA strands into separate sister chromatids after BrdU incorporation, the hybridization of same-orientation probes to one chromatid, and the obligatory signal switch caused by an inversion.

- describe use of synthetic oligomers as probes  

Synthetic oligomers (typically 35–50 nucleotides) offer superior batch-to-batch consistency, customizable labeling, and avoidance of repetitive element contamination compared to PCR-amplified or cloned DNA probes.

- describe labeling probes with fluorescent molecules  

Fluorescent labeling is accomplished via enzymatic tailing with TdT using dye-conjugated dUTPs. Alternative methods include PCR incorporation of labeled nucleotides or chemical conjugation post-synthesis, though TdT labeling preserves single-stranded integrity.

### Selection of Sequences:

- identify large contiguous DNA sequences for use as targets  

Large contigs from reference genome assemblies (e.g., NT_005612 for chromosome 3q) are prioritized as targets because they represent regions of high-confidence sequence orientation and minimal gaps.

- select shorter sequences within contigs for use as probes  

Within each contig, 40-mer sequences spaced at regular intervals (e.g., every 100 kb within a 1-Mb tiling window) are selected based on uniqueness, absence of repeats, and predicted hybridization efficiency.

### EXAMPLE 2

- generate sequence-specific DNA by PCR  

Sequence-specific DNA can be generated by PCR amplification of genomic regions using primers designed to flank unique targets. Amplification conditions include initial denaturation at 95°C for 2 min, followed by 30 cycles of 95°C for 30 sec, 60°C for 30 sec, and 72°C for 1 min, with a final extension at 72°C for 5 min.

- describe PCR conditions and primer design  

Primers are 20–25 nucleotides in length, with Tm ~60°C, minimal self-complementarity, and amplicon sizes of 200–500 bp. GC clamps and repetitive motifs are avoided to ensure specificity.

- generate sequence-specific DNA by oligo tiling and commercial synthesis  

Alternatively, overlapping oligonucleotides tiling across a target region are synthesized commercially and pooled. This approach bypasses PCR bias and enables precise control over sequence composition.

- describe use of Array Designer software  

Array Designer v4.2 (Premier Biosoft) is used to automate probe selection, ensuring uniform Tm, avoidance of secondary structures, and compliance with spacing constraints.

- label probes by PCR  

Probes generated by PCR can be labeled by including fluorescent dUTP analogs (e.g., AlexaFluor 594-5-dUTP) in the reaction mix at a ratio of 1:4 with unlabeled dTTP, enabling direct incorporation during amplification.

- describe linear DNA amplification conditions  

Linear amplification using T7 RNA polymerase or phi29 DNA polymerase may be employed to amplify limited starting material while preserving representation. Conditions include isothermal incubation at 37°C for 2–16 hours.

- label probes using terminal transferase  

Terminal deoxynucleotidyl transferase (TdT) catalyzes the addition of labeled nucleotides to 3′ hydroxyl ends of single-stranded DNA. Reactions contain 1× TdT buffer, 1 mM CoCl₂, 20 μM labeled dUTP, 10 U TdT, and 100 ng oligo probe, incubated at 37°C for 30 min.

- describe labeling reaction conditions  

Labeling reactions are terminated by ethanol precipitation or column purification. Efficiency is confirmed by spectrophotometry or gel electrophoresis showing a mobility shift consistent with nucleotide tailing.

- describe use of Hae III restriction endonuclease  

HaeIII digestion may be used to fragment double-stranded PCR products into smaller pieces (~200 bp) suitable for FISH, though this step is unnecessary for single-stranded oligo probes.

- prepare probes for FISH  

Probes are resuspended in hybridization buffer containing 50% formamide, 10% dextran sulfate, 2× SSC, and 1% SDS. Denaturation at 75°C for 5 min ensures single-stranded state prior to application.

- describe labeling reaction with AlexaFluor 594-5-dUTP  

AlexaFluor 594-5-dUTP is incorporated via TdT-mediated tailing, yielding bright red fluorescence detectable with standard TRITC filter sets. Molar incorporation ratios are optimized to maximize signal without quenching.

- describe use of other labeled nucleotides  

Alternative labels include Cy3-dUTP, FITC-dUTP, biotin-dUTP (detected with streptavidin conjugates), or quantum dots, enabling multiplexed detection of multiple inversions or chromosomes simultaneously.

### EXAMPLE 3

- describe CO-FISH using probes generated by PCR  

PCR-generated probes, once labeled and fragmented, can be used in CO-FISH provided they are rendered single-stranded by alkaline denaturation immediately prior to hybridization. However, synthetic oligos are preferred for consistency.

- outline steps for CO-FISH  

CO-FISH steps include: (1) BrdU incorporation; (2) metaphase arrest and slide preparation; (3) Hoechst staining and UV exposure; (4) Exonuclease III digestion; (5) probe denaturation and hybridization; (6) post-wash and counterstaining; (7) imaging and analysis.

- describe hybridization and detection of probes  

Hybridization signals are visualized as discrete fluorescent foci along chromatids. Inversion breakpoints are scored when adjacent probe sets localize to opposite sister chromatids within the same chromosome.

### Hybridization:

- describe hybridization mixture and conditions  

Hybridization mixture contains 50% deionized formamide, 10% dextran sulfate, 2× SSC, 1% SDS, and 2–5 ng/μL probe. Slides are denatured at 73°C for 3 min, then incubated at 37°C in a humidified chamber overnight.

- outline steps for washing and mounting slides  

Post-hybridization, slides are washed five times in 2× SSC at 42°C for 15 min each, rinsed in PN buffer, counterstained with DAPI in antifade mounting medium, and sealed with nail polish.

### Detection:

- describe detection of probes using fluorescence microscope  

Detection is performed using an epifluorescence microscope (e.g., Olympus BX41) equipped with high-sensitivity CCD camera (e.g., Photometrics CoolSNAP ES2) and Metamorph or MetaVue software for image capture and analysis.

### Results:

- describe results of using PCR and tiled oligos to produce probes  

Both PCR-derived and tiled oligo probes yielded specific hybridization to single chromatids. Oligo-based probes showed lower background and higher reproducibility due to absence of repetitive sequences.

- observe red fluorescence on single chromatids  

Red fluorescence (AlexaFluor 594) was consistently observed on only one sister chromatid per chromosome, confirming successful strand-specific hybridization and inversion detection capability.

- note significant background signal  

Initial experiments exhibited moderate background, attributed to incomplete removal of BrdU strands or non-specific probe binding. Optimization of Exonuclease III concentration and hybridization stringency reduced background significantly.

- expect reduction of background signal with increased stringency  

Increasing post-hybridization wash temperature to 45–50°C or reducing formamide concentration during hybridization is expected to further suppress non-specific signals.

- describe use of other fluorescent or non-fluorescent labels  

Non-fluorescent labels such as digoxigenin or biotin allow chromogenic or enzyme-linked detection, useful for brightfield microscopy or archival sample analysis.

- confirm incorporation of AlexaFluor 594-5 dUTP into probes  

Incorporation was confirmed by comparing fluorescence intensity of labeled versus unlabeled controls and by gel shift assays showing increased molecular weight consistent with nucleotide tailing.