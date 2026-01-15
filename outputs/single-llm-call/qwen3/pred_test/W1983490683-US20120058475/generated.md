# DESCRIPTION

- cross-reference to related patent applications

This application claims no benefit of priority to any previously filed patent application and constitutes an original disclosure of novel methods and compositions for the detection of chromosomal inversions using directional genomic hybridization. No prior provisional or non-provisional applications have been filed by the inventors or assignees that disclose the specific combination of single-stranded chromatid-specific probes, bioinformatically selected non-repetitive oligonucleotide sequences, and strand-specific hybridization protocols described herein. All aspects of the invention, including probe design criteria, labeling techniques, metaphase chromosome preparation methods, and inversion detection criteria, are disclosed for the first time in this specification. This application is not a continuation, divisional, or continuation-in-part of any earlier-filed application, nor does it incorporate by reference any prior application in whole or in part. The invention described herein is entirely novel and non-obvious in light of the state of the art as of the filing date.

## FIELD OF THE INVENTION

- define field of invention

The present invention resides in the field of molecular cytogenetics and diagnostic oncology, specifically in the development of high-resolution, cell-by-cell methods for detecting chromosomal inversions in human and non-human eukaryotic cells. The invention provides a novel approach to identifying intra-chromosomal rearrangements that are otherwise cryptic to conventional banding techniques, fluorescence in situ hybridization using whole-chromosome paints, and genomic sequencing platforms. The methods and compositions disclosed herein enable the visualization of inverted DNA segments at resolutions far exceeding those achievable through traditional cytogenetic analyses, thereby facilitating the detection of structural variants associated with cancer, developmental disorders, and genomic instability. The invention is particularly useful in clinical diagnostics, environmental mutagenesis studies, and comparative genomics, where precise identification of inversion breakpoints and their cellular prevalence is critical for accurate risk assessment, disease classification, and therapeutic decision-making.

## BACKGROUND OF THE INVENTION

- motivate chromosomal aberrations

Chromosomal aberrations represent a fundamental class of genomic alterations that underlie a wide spectrum of human diseases, including malignancies, congenital syndromes, and infertility. These structural changes disrupt gene dosage, regulatory elements, and transcriptional architecture, often leading to the formation of oncogenic fusion proteins or the dysregulation of tumor suppressor genes. While translocations and deletions have been extensively characterized, inversions—rearrangements in which a segment of a chromosome is reversed in orientation—remain significantly underdetected due to technical limitations in existing methodologies. Their cryptic nature, particularly when small or lacking visible banding disruptions, has obscured their true frequency and biological significance in human pathology.

- describe cancer cells

Cancer cells frequently exhibit complex genomic instability characterized by recurrent structural rearrangements that drive clonal evolution, therapeutic resistance, and metastatic potential. These alterations are not random but often recur in specific genomic loci, suggesting selective pressure for functional consequences such as aberrant gene expression or loss of tumor suppression. The persistence of inversion events in tumor populations indicates that they are not merely bystander effects of DNA damage but may confer a proliferative advantage, particularly when they juxtapose regulatory elements with proto-oncogenes or disrupt tumor suppressor loci.

- describe tumor-specific chromosome aberrations

Tumor-specific chromosomal aberrations include translocations, deletions, amplifications, and inversions, each contributing distinct molecular phenotypes. While translocations such as BCR-ABL in chronic myeloid leukemia and MYC-IGH in Burkitt lymphoma are well documented, inversions remain underrepresented in clinical databases due to their invisibility under standard cytogenetic assays. Many tumor genomes harbor inversions that are functionally significant yet undetected, leading to misclassification of rearrangements as “complex” or “uncharacterized” structural variants.

- describe inversions

An inversion occurs when two double-strand breaks occur within a single chromosome, followed by the reinsertion of the intervening segment in reverse orientation. This rearrangement preserves overall chromosome length and copy number, rendering it invisible to array-based comparative genomic hybridization and often to standard karyotyping. The inversion does not alter the total amount of genetic material but reverses the linear order of genes and regulatory sequences, potentially disrupting enhancer-promoter interactions or creating novel fusion transcripts.

- describe genetic effects of inversions

The genetic consequences of inversions include the separation of genes from their native regulatory landscapes, the juxtaposition of unrelated sequences to generate chimeric transcripts, and the creation of novel splice variants. Inversions may also interfere with meiotic pairing, leading to gamete aneuploidy and infertility. In somatic cells, inversions can activate oncogenes by placing them under the control of strong enhancers or silence tumor suppressors by relocating them to heterochromatic regions.

- describe misrepair of DNA double-strand breaks

Misrepair of DNA double-strand breaks is a primary mechanism underlying chromosomal inversions. When two breaks occur in close spatial proximity on the same chromosome, the cellular repair machinery may erroneously rejoin the ends in an inverted configuration rather than restoring the original orientation. This misrepair is particularly prevalent following exposure to ionizing radiation, chemotherapeutic agents, or endogenous oxidative stress, and is a hallmark of genomic instability in cancer cells.

- describe limitations of standard karyotype analyses

Standard karyotype analyses rely on G- or R-banding patterns to detect structural abnormalities, but these methods are limited to resolving rearrangements larger than approximately 5–10 megabases. Smaller inversions, especially those lacking distinct banding discontinuities, remain undetectable. Furthermore, banding patterns are subject to inter-laboratory variability and require highly skilled interpretation, making them unsuitable for high-throughput or automated screening.

- describe new approaches to measuring incorrect rejoining of radiation-induced DNA double-strand breaks

Recent advances in sequencing technologies have enabled the detection of inversion breakpoints at nucleotide resolution, but these approaches are constrained by their inability to assess cellular heterogeneity, require large amounts of DNA, and are not amenable to single-cell analysis. Additionally, sequencing-based methods are prone to misassembly in regions of high homology or repetitive content, leading to false negatives in inversion detection.

- describe significance of inversions in diseases

Inversions have been implicated in a growing number of diseases, including pediatric acute leukemia, thyroid carcinoma, and neurodevelopmental disorders. For example, inv(16)(p13.3q24.3) generates the CBFA2T3-GLIS2 fusion gene, defining a high-risk subtype of acute myeloid leukemia. The failure to detect such inversions in routine diagnostics results in misclassification, inappropriate risk stratification, and suboptimal treatment selection.

- describe chromosome analysis in prenatal screening and diagnosis of congenital abnormalities

Prenatal screening for chromosomal abnormalities traditionally relies on karyotyping and FISH, but inversions below the resolution threshold of these methods remain undetected, leading to unexplained cases of fetal anomalies, miscarriage, or developmental delay. A method capable of detecting small inversions at the single-cell level would significantly enhance the diagnostic yield of prenatal cytogenetic testing.

- describe whole chromosome painting by fluorescence in situ hybridization (FISH)

Whole chromosome painting employs fluorescently labeled DNA probes derived from entire chromosomes to visualize gross rearrangements. However, because these probes hybridize to both sister chromatids simultaneously, they cannot distinguish orientation changes within a chromosome, rendering them incapable of detecting inversions.

- describe G- or R-banding

G- and R-banding produce alternating light and dark bands along chromosomes based on differential staining of AT- and GC-rich regions. While useful for identifying large-scale rearrangements, these techniques lack the resolution to detect inversions smaller than several megabases and are highly dependent on chromosome condensation quality and technician expertise.

- describe limitations of FISH and G-banding

Both FISH and G-banding are fundamentally limited by their inability to resolve orientation changes within a chromosome. FISH probes bind to both strands of a chromosome, masking any inversion-induced polarity reversal. G-banding relies on visual pattern recognition, which is insensitive to inversions that do not alter banding density or distribution. Neither method provides strand-specific information.

- describe chromatid structure

Each chromosome at metaphase consists of two sister chromatids, each derived from a single DNA molecule replicated during S-phase. The two chromatids are held together at the centromere and contain complementary polynucleotide strands that are antiparallel in orientation. The sequence directionality of each chromatid is preserved from the original parental DNA strand, creating a directional asymmetry that can be exploited for inversion detection.

- describe DNA molecule structure

DNA is a double-stranded helix composed of two antiparallel polynucleotide chains held together by complementary base pairing. Each strand has a defined 5′ to 3′ directionality, and the sequence of nucleotides along a given strand determines its hybridization specificity. Inversions reverse the 5′→3′ orientation of a segment, thereby altering the directional context of the underlying sequence.

- describe complementary base pairing

Complementary base pairing between adenine and thymine, and guanine and cytosine, governs the specificity of hybridization between nucleic acid strands. This principle underlies all molecular detection methods, including FISH, and is exploited in the present invention to ensure that probes bind only to sequences with perfect complementarity and correct orientation.

- describe genome replication

During genome replication, each parental DNA strand serves as a template for the synthesis of a new complementary strand. The resulting daughter chromosomes each contain one original strand and one newly synthesized strand. When bromodeoxyuridine is incorporated during replication, the nascent strands become susceptible to selective degradation, leaving behind single-stranded chromatids with preserved directional polarity.

- describe inversion formation

Inversion formation occurs when two double-strand breaks occur on the same chromosome, and the intervening segment is excised, rotated 180 degrees, and reinserted. This process reverses the 5′→3′ orientation of the DNA sequence within the inverted region, while preserving the overall continuity of the chromosome. The inversion does not alter the total sequence content but changes the spatial and directional relationship of flanking sequences.

- describe chromosome 'paints'

Chromosome paints are pools of fluorescently labeled DNA probes designed to hybridize to entire chromosomes or chromosomal regions. Conventional chromosome paints are composed of double-stranded DNA fragments and bind to both sister chromatids, rendering them incapable of detecting orientation changes.

- describe CO-FISH technique

Chromosome orientation fluorescence in situ hybridization (CO-FISH) is a technique that exploits the differential susceptibility of bromodeoxyuridine-substituted DNA strands to enzymatic degradation. By selectively removing the newly synthesized strands, CO-FISH generates metaphase chromosomes with single-stranded chromatids, enabling strand-specific hybridization. However, prior CO-FISH methods were limited to repetitive sequences and lacked the resolution to detect inversions in unique genomic regions.

## SUMMARY OF THE INVENTION

- motivate sensitive method for detection of chromosomal inversions

There exists a critical need for a sensitive, high-resolution, and unbiased method capable of detecting chromosomal inversions at the single-cell level across the entire genome. The invention fulfills this need by combining strand-specific hybridization with bioinformatically designed, non-repetitive, sequence-directed probes that reveal inversions as discrete switches in hybridization signal between sister chromatids, enabling detection of inversions as small as a few kilobases with high specificity and reproducibility.

- describe probe kit for sensitive detection of chromosomal inversions

The invention provides a probe kit comprising a plurality of synthetic, single-stranded oligonucleotide probes, each designed to hybridize to a unique, non-repetitive genomic sequence with defined 5′→3′ orientation. The probes are labeled with detectable moieties and are pooled to cover entire chromosomes or targeted regions. The kit further includes reagents for generating single-stranded chromatids, hybridization buffers, washing solutions, and mounting media optimized for directional hybridization.

- describe method for detecting inversions

The method comprises culturing cells in the presence of bromodeoxyuridine to label nascent DNA strands, arresting cells in metaphase, selectively degrading the newly synthesized strands to expose single-stranded chromatids, hybridizing the directional probes to the exposed chromatids, and detecting hybridization signals using fluorescence microscopy. An inversion is identified when a probe signal switches from one sister chromatid to the other, indicating a reversal in the 5′→3′ orientation of the underlying DNA segment.

- generate single-stranded sister chromatids

Single-stranded sister chromatids are generated by incorporating bromodeoxyuridine into the DNA of dividing cells during a single cell cycle, followed by staining with Hoechst 33258, exposure to ultraviolet light to induce strand breaks at bromodeoxyuridine sites, and treatment with exonuclease III to selectively digest the bromodeoxyuridine-containing strands, leaving behind the original parental strands as single-stranded templates for hybridization.

- generate non-repetitive probes

Non-repetitive probes are generated by selecting unique genomic sequences from publicly available genome databases, excluding regions with homology to repetitive elements such as LINEs, SINEs, and satellite DNA, and designing oligonucleotides of uniform length and melting temperature to ensure consistent hybridization behavior.

- hybridize probes to sister chromatids

The probes are hybridized to the single-stranded chromatids under stringent conditions that permit binding only to perfectly complementary sequences with matching 5′→3′ orientation, ensuring that signal is observed only when the probe sequence aligns correctly with the target strand.

- detect hybridized probes

Hybridized probes are detected using fluorescence microscopy equipped with filters specific to the labeling fluorophores. Signal localization on a single chromatid confirms normal orientation; a switch in signal between sister chromatids indicates the presence of an inversion.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce method for detecting inversions in chromosomes

The method for detecting chromosomal inversions involves the preparation of metaphase chromosomes from cells that have undergone a single round of DNA replication in the presence of bromodeoxyuridine. Following selective degradation of the newly synthesized strands, the remaining chromatids are single-stranded and retain the original polarity of the parental DNA. Hybridization of directional probes to these chromatids reveals inversions as abrupt transitions in signal localization between sister chromatids, providing a direct visual readout of orientation changes.

- describe CO-FISH technology

Chromosome orientation fluorescence in situ hybridization (CO-FISH) is the foundational technology upon which this invention is built. Unlike conventional FISH, CO-FISH exploits the differential stability of bromodeoxyuridine-substituted DNA to generate single-stranded chromatids. This enables the use of strand-specific probes that bind only to sequences oriented in the correct 5′→3′ direction, thereby revealing inversions as polarity switches.

- explain use of unique DNA sequences as probes

The invention utilizes unique, non-repetitive DNA sequences as probes to ensure specificity and avoid cross-hybridization. These sequences are selected from regions of the genome devoid of repetitive elements and are designed to span contiguous stretches of DNA with known orientation, enabling the construction of high-resolution maps of chromosomal structure.

- describe genome databases for identifying suitable probes

Publicly accessible genome databases, including the Human Genome Reference Assembly (GRCh37 and GRCh38), are used to identify suitable probe targets. Bioinformatic tools filter out repetitive sequences, assess GC content, and confirm uniqueness by BLAST analysis against the entire genome to ensure probe specificity.

- outline steps for identifying and checking probe sequences

The steps include downloading contiguous genomic sequences, masking repetitive regions using RepeatMasker or similar software, selecting oligonucleotides of 35–50 nucleotides in length with melting temperatures between 65°C and 75°C, verifying uniqueness via BLAST, and confirming absence of secondary structure using folding algorithms.

- describe analysis of database for coverage of chromatid

To ensure complete coverage of a target chromosome, the genome is divided into overlapping or adjacent regions, each targeted by a distinct probe set. The spacing between probe sets is optimized to balance resolution and practicality, with intervals ranging from 1 Mb for genome-wide screening to 10 kb for high-resolution targeted analysis.

- outline steps for synthesizing and labeling probes

Oligonucleotides are chemically synthesized using solid-phase phosphoramidite chemistry, purified by HPLC, and labeled with fluorescent nucleotides such as AlexaFluor 594-5-dUTP using terminal transferase or PCR-based labeling. Labeled probes are pooled in defined ratios to ensure uniform signal intensity.

- describe generating single-stranded sister chromatids

Metaphase spreads are prepared from cells arrested in mitosis and treated with Hoechst 33258 and ultraviolet light to induce strand breaks at bromodeoxyuridine sites. Exonuclease III is then applied to digest the bromodeoxyuridine-containing strands, leaving behind the complementary strands as single-stranded targets for hybridization.

- outline steps for hybridizing probes to chromatids

Hybridization mixtures containing probe pools, formamide, dextran sulfate, and SSC are denatured at 75°C and applied to pretreated slides. Slides are sealed, incubated overnight at 37°C, and washed under stringent conditions to remove non-specific binding.

- describe detecting probes hybridized to one chromatid

Fluorescence microscopy is used to visualize probe signals. A continuous signal along one chromatid indicates normal orientation; a signal switch to the sister chromatid indicates an inversion. Signal intensity, continuity, and localization are analyzed using automated image capture software.

- outline steps for developing chromatid paints

Chromatid paints are developed by pooling hundreds to thousands of individually labeled oligonucleotide probes targeting unique sequences along a chromosome. Each probe set is validated for strand specificity before pooling. The final paint is applied to metaphase spreads to generate whole-chromosome or sub-chromosomal directional maps.

- describe use of metaphase chromosomes for CO-FISH

Metaphase chromosomes are the preferred substrate for CO-FISH because they are maximally condensed, spatially resolved, and contain two clearly distinguishable sister chromatids. Their structural integrity allows for precise localization of hybridization signals and accurate inversion mapping.

- illustrate method with schematic representations

Schematic representations depict the progression from bromodeoxyuridine incorporation to strand degradation, probe hybridization, and inversion detection. Arrows indicate 5′→3′ orientation, and color switches between chromatids illustrate inversion breakpoints.

- describe use of synthetic oligomers as probes

Synthetic oligomers are preferred over genomic DNA fragments due to their uniform length, defined sequence, and absence of repetitive elements. They enable high specificity, reproducible hybridization kinetics, and compatibility with high-throughput manufacturing.

- describe labeling probes with fluorescent molecules

Probes are labeled with fluorophores such as AlexaFluor 488, Cy3, Cy5, or Texas Red using terminal transferase-mediated tailing or PCR incorporation of labeled nucleotides. Labeling efficiency is confirmed by spectrophotometry and gel electrophoresis.

### Selection of Sequences:

- identify large contiguous DNA sequences for use as targets

Large contiguous DNA sequences are identified from genome assemblies using tools such as UCSC Genome Browser or Ensembl. Regions with minimal gaps, high sequence completeness, and low repeat content are prioritized as targets for probe design.

- select shorter sequences within contigs for use as probes

Within each contig, 40-mer oligonucleotides are selected at intervals of 1 Mb or less, ensuring that each probe is unique, has a melting temperature within ±5°C of the target, and is oriented in the same 5′→3′ direction as the underlying genomic sequence.

### EXAMPLE 2

- generate sequence-specific DNA by PCR

Sequence-specific DNA is amplified from genomic DNA using primers designed to flank unique regions. PCR conditions include denaturation at 95°C, annealing at 60°C, and extension at 72°C for 30 cycles.

- describe PCR conditions and primer design

Primers are 20–25 nucleotides in length, with GC content of 45–55%, and lack secondary structure. Melting temperatures are calculated using the nearest-neighbor method and adjusted to ensure specificity.

- generate sequence-specific DNA by oligo tiling and commercial synthesis

Oligo tiling involves designing overlapping or adjacent oligonucleotides that collectively cover a target region. These are synthesized commercially using high-fidelity phosphoramidite chemistry and pooled in equimolar ratios.

- describe use of Array Designer software

Array Designer software is used to design oligonucleotides with uniform melting temperatures, minimal self-complementarity, and optimal spacing. The software filters out repetitive sequences and predicts hybridization efficiency.

- label probes by PCR

Labeled probes are generated by PCR using labeled nucleotides such as AlexaFluor 594-5-dUTP. The reaction includes Taq polymerase, dNTPs, and a 1:5 ratio of labeled to unlabeled nucleotides.

- describe linear DNA amplification conditions

Linear amplification is performed using a single primer and 20 cycles of amplification to avoid exponential amplification artifacts. Reaction conditions include 1× PCR buffer, 200 μM dNTPs, 1.5 mM MgCl₂, and 0.5 U/μL Taq polymerase.

- label probes using terminal transferase

Terminal transferase adds fluorescently labeled nucleotides to the 3′ ends of oligonucleotides. The reaction is performed at 37°C for 30 minutes in the presence of Co²⁺ and labeled dUTP.

- describe labeling reaction conditions

Labeling reactions contain 1 μg of oligonucleotide, 10 U terminal transferase, 100 μM AlexaFluor 594-5-dUTP, 1× reaction buffer, and 1 mM CoCl₂. Reactions are stopped by heat inactivation at 70°C.

- describe use of Hae III restriction endonuclease

Hae III is used to fragment genomic DNA into smaller pieces for probe generation. Digestion conditions include 1 U enzyme per μg DNA, 37°C for 1 hour, followed by heat inactivation.

- prepare probes for FISH

Probes are ethanol precipitated, resuspended in hybridization buffer, and denatured prior to application to slides. Probe concentration is adjusted to 1–5 ng/μL for optimal signal-to-noise ratio.

- describe labeling reaction with AlexaFluor 594-5 dUTP

AlexaFluor 594-5-dUTP is incorporated into probes during PCR or terminal transferase reactions. Incorporation is confirmed by absorbance at 594 nm and fluorescence microscopy.

- describe use of other labeled nucleotides

Other labeled nucleotides, including Cy3-dUTP, FITC-dUTP, and biotin-16-dUTP, are used interchangeably depending on detection requirements. Non-fluorescent labels such as digoxigenin are used for indirect detection with enzyme-conjugated antibodies.

### EXAMPLE 3

- describe CO-FISH using probes generated by PCR

CO-FISH is performed using probes generated by PCR amplification of unique genomic regions. Probes are labeled with AlexaFluor 594-5-dUTP and hybridized to bromodeoxyuridine-treated metaphase spreads.

- outline steps for CO-FISH

Slides are pretreated with Hoechst 33258 and UV light, digested with exonuclease III, hybridized with probe mix, washed, counterstained with DAPI, and imaged using a fluorescence microscope.

- describe hybridization and detection of probes

Hybridization occurs overnight at 37°C. Washes remove non-specific binding. Fluorescence signals are captured using a high-resolution camera and analyzed for signal switches between sister chromatids.

### Hybridization:

- describe hybridization mixture and conditions

The hybridization mixture contains 50% formamide, 10% dextran sulfate, 2× SSC, 10% SDS, and 1–5 ng/μL of probe pool. Slides are denatured at 73°C for 3 minutes and hybridized at 37°C for 12–16 hours.

- outline steps for washing and mounting slides

Slides are washed five times in 2× SSC at 42°C for 15 minutes each, followed by a rinse in 0.1× SSC. Slides are air dried, counterstained with DAPI, and mounted in antifade medium.

### Detection:

- describe detection of probes using fluorescence microscope

Fluorescence signals are detected using an Olympus BX41 microscope equipped with appropriate filter sets for AlexaFluor 594, Cy3, and DAPI. Images are captured using a CoolSNAP ES2 camera and analyzed with Metavue software.

### Results:

- describe results of using PCR and tiled oligos to produce probes

Both PCR-amplified and synthetically tiled probes produced strong, specific signals on single chromatids. Signal intensity was comparable between methods, with tiled oligos showing slightly higher uniformity.

- observe red fluorescence on single chromatids

Red fluorescence from AlexaFluor 594-labeled probes was observed exclusively on one chromatid of each target chromosome, confirming strand specificity.

- note significant background signal

Initial hybridizations exhibited moderate background signal, which was reduced by increasing stringency of washes and optimizing probe concentration.

- expect reduction of background signal with increased stringency

Increased stringency, including higher formamide concentration and elevated wash temperatures, significantly reduced non-specific binding without compromising signal intensity.

- describe use of other fluorescent or non-fluorescent labels

Other fluorophores, including Cy5, FITC, and Oregon Green, produced equivalent results. Non-fluorescent labels such as biotin and digoxigenin enabled detection via enzyme-linked secondary reagents.

- confirm incorporation of AlexaFluor 594-5 dUTP into probes

Incorporation was confirmed by spectrophotometric analysis, which showed absorbance peaks at 594 nm, and by gel electrophoresis, which demonstrated increased molecular weight of labeled probes.