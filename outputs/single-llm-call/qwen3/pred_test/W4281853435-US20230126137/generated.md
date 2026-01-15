# DESCRIPTION

## BACKGROUND

- introduce RNA binding proteins

RNA binding proteins constitute a vast and functionally diverse class of molecular regulators that govern nearly every aspect of RNA metabolism, from transcription and splicing to transport, stability, localization, and translation. These proteins recognize specific RNA sequences, secondary structures, or chemical modifications through conserved RNA-binding domains, enabling them to exert precise temporal and spatial control over gene expression. Their activity is not static but dynamically modulated by post-translational modifications, cellular stress, developmental cues, and disease states. Dysregulation of RNA binding protein function has been implicated in a wide spectrum of human pathologies, including neurodegenerative disorders such as amyotrophic lateral sclerosis and frontotemporal dementia, as well as numerous cancers, where aberrant RNA processing can drive oncogenic transformation, metastasis, and therapeutic resistance. Despite their critical biological roles, the full repertoire of RNA binding proteins in the human genome remains incompletely characterized, with emerging evidence suggesting that at least 15% of all protein-coding genes may possess RNA-binding capacity. Traditional methods for identifying RNA targets of these proteins, such as RNA immunoprecipitation and its more refined variant, crosslinking and immunoprecipitation, have enabled significant advances in transcriptome-wide mapping. However, these approaches remain labor-intensive, low-throughput, and constrained by the requirement for large quantities of starting material, limiting their utility in clinical contexts where tissue samples are scarce or heterogeneous. The need for a scalable, sensitive, and multiplexable platform capable of interrogating hundreds of RNA binding proteins simultaneously from minimal biological inputs has become a central challenge in molecular biology and precision medicine.

## SUMMARY

- introduce method of identifying RNA-protein complexes

The present invention provides a novel method for identifying RNA molecules bound by RNA binding proteins in a high-throughput, multiplexed, and highly sensitive manner, enabling comprehensive profiling of RNA-protein interactions without the need for gel-based size selection or individual immunoprecipitations. This method leverages the covalent conjugation of unique oligonucleotide barcodes to antibodies specific for individual RNA binding proteins, thereby enabling the direct ligation of bound RNA molecules to their cognate protein-capturing entities in close spatial proximity. The resulting chimeric RNA-DNA molecules serve as molecular records of specific RNA-protein interactions, preserving the identity of both the bound RNA and the RNA binding protein that captured it. This approach eliminates the need for SDS-PAGE and membrane transfer steps, which are traditionally required to isolate protein-RNA complexes by size, thereby reducing procedural complexity, minimizing technical variability, and significantly accelerating workflow timelines. The method further enables the simultaneous interrogation of multiple RNA binding proteins within a single experimental sample, dramatically reducing the quantity of biological material required per target and facilitating the analysis of rare or clinically derived specimens such as tumor biopsies, circulating tumor cells, or archived formalin-fixed tissues.

- describe contacting RNA sample with RNA binding proteins

The method begins with the exposure of a biological sample containing intact RNA-protein complexes to a panel of oligonucleotide-conjugated antibodies, each specific for a distinct RNA binding protein. Prior to exposure, cells or tissues are subjected to UV crosslinking to covalently stabilize transient RNA-protein interactions, ensuring that only direct, in vivo binding events are captured. The crosslinked sample is then lysed under denaturing conditions that preserve RNA integrity while disrupting noncovalent protein-protein and protein-RNA interactions, thereby enriching for complexes that were stabilized by crosslinking. The lysate is incubated with the panel of oligonucleotide-conjugated antibodies, which selectively bind to their target RNA binding proteins, thereby capturing the RNA molecules that are physically associated with each protein at the time of crosslinking. This binding occurs in solution, allowing for uniform and simultaneous capture of multiple RNA binding protein-RNA complexes within a single reaction vessel.

- describe ligating RNA sample to oligo conjugated entities

Following immunoprecipitation and stringent washing to remove non-specifically bound material, the RNA molecules that are covalently linked to their cognate RNA binding proteins are brought into close proximity with the oligonucleotide barcode conjugated to the antibody. A proximity-dependent ligation reaction is then performed using a T4 RNA ligase under optimized conditions that favor intermolecular ligation between the 3′ end of the crosslinked RNA and the 5′ end of the oligonucleotide barcode. This ligation event is spatially constrained by the physical tethering of the RNA to its binding protein via the antibody, ensuring that only RNA molecules directly bound to the target protein are ligated to the corresponding barcode. The resulting chimeric RNA-DNA molecules thus encode both the identity of the RNA binding protein (via the barcode sequence) and the sequence of the bound RNA transcript.

- describe amplifying chimeric RNA or DNA molecules

The chimeric RNA-DNA molecules are then reverse transcribed into complementary DNA using a primer complementary to the oligonucleotide barcode sequence. The resulting cDNA is subjected to a series of enzymatic treatments including end repair, adapter ligation, and polymerase chain reaction amplification to generate a sequencing library compatible with high-throughput next-generation sequencing platforms. The amplification step is performed using primers that flank the barcode-RNA junction, ensuring that only successfully ligated chimeric molecules are amplified. To mitigate the effects of PCR duplication, unique molecular identifiers are incorporated into the barcode design, allowing for computational deduplication of reads derived from the same original RNA molecule. This results in a library that faithfully represents the abundance and identity of RNA molecules bound by each targeted RNA binding protein.

- describe sequencing chimeric RNA or DNA molecules

The amplified libraries are sequenced using high-depth, paired-end sequencing on an Illumina platform, generating millions of reads per sample. Each read contains a segment corresponding to the oligonucleotide barcode, which identifies the RNA binding protein, followed by a segment derived from the bound RNA transcript. The sequencing depth is sufficient to detect low-abundance RNA targets and to distinguish true binding events from background noise, even in multiplexed experiments involving ten or more RNA binding proteins simultaneously.

- describe identifying computationally chimeric RNA or DNA molecules

Bioinformatic analysis is performed to demultiplex the sequencing reads based on the barcode sequence, assigning each read to its corresponding RNA binding protein. Adapter sequences and unique molecular identifiers are trimmed, and reads are aligned to the reference genome to determine the genomic origin of the bound RNA. Reads that map uniquely to the genome and contain a clear barcode-RNA junction are retained as high-confidence binding events. Peak calling algorithms are applied to identify genomic regions enriched for chimeric reads, and statistical models are used to compare the enrichment of each RBP’s targets against a complementary control derived from the aggregate reads of all other RBPs in the multiplexed sample, thereby eliminating background noise without requiring separate input controls.

- describe isolating RNA-protein complex

The RNA-protein complex is isolated through immunoprecipitation using antibody-conjugated magnetic beads, which provide a solid-phase support for efficient capture, washing, and elution of bound complexes. The beads are pre-conjugated with the oligonucleotide-barcoded antibodies, ensuring that each bead population is uniquely identifiable by its barcode sequence. After crosslinking and lysis, the sample is incubated with the bead-bound antibodies under conditions that preserve RNA integrity and promote specific binding. Following extensive washing to remove non-specifically bound material, the RNA-protein complexes remain stably associated with the beads, ready for proximity ligation and downstream processing.

- describe lysing cells prior to isolating RNA-protein complex

Prior to isolation of the RNA-protein complex, cells are lysed using a buffer containing detergents, salts, and RNase inhibitors to disrupt cellular membranes while preserving RNA-protein interactions. The lysate is then subjected to sonication to shear genomic DNA and reduce viscosity, followed by treatment with RNase I and Turbo DNase to partially digest RNA and remove non-crosslinked RNA, respectively. This step ensures that only RNA molecules covalently crosslinked to their binding proteins remain intact and available for subsequent capture and ligation, thereby enhancing the specificity of the method.

- describe immunoprecipitation of RNA-protein complex

Immunoprecipitation is performed by incubating the lysate with antibody-conjugated magnetic beads under gentle rotation at 4°C for an extended period, typically overnight. The beads are then collected using a magnetic separator and washed multiple times with buffers of increasing stringency, including high-salt and lithium chloride-containing solutions, to remove non-specifically bound proteins and RNA. The specificity of immunoprecipitation is further enhanced by the use of validated, high-affinity antibodies previously characterized for RNA immunoprecipitation applications.

- describe generating oligo conjugated antibodies

Oligonucleotide-conjugated antibodies are generated through a two-step click chemistry reaction. First, antibodies are modified with dibenzocyclooctyne (DBCO) groups via NHS ester chemistry targeting lysine residues. Separately, oligonucleotide barcodes are functionalized with azide moieties. The DBCO-modified antibodies are then reacted with azide-labeled oligonucleotides under mild aqueous conditions, resulting in a stable triazole linkage between the antibody and the barcode. Unreacted reagents are removed by buffer exchange using size-exclusion chromatography, yielding pure, functional oligo-conjugated antibodies ready for use in immunoprecipitation.

- describe crosslinking RNA-protein complex

Crosslinking is achieved by exposing intact cells or tissues to ultraviolet light at a wavelength of 254 nm and a dose of approximately 400 mJ/cm². This induces the formation of covalent bonds between RNA nucleotides and nearby amino acid residues in RNA binding proteins, effectively “freezing” transient interactions that occur in vivo. Crosslinking is performed on ice to minimize non-specific damage and is followed by immediate cell harvesting and freezing to preserve the integrity of the crosslinked complexes.

- describe enriching chimeric RNA or DNA molecules

Enrichment of chimeric RNA or DNA molecules is achieved through a combination of proximity ligation, stringent washing, and selective amplification. Only those RNA molecules that are physically tethered to the oligonucleotide barcode via the RNA binding protein are ligated to the barcode, ensuring that non-specific RNA is excluded. Subsequent reverse transcription and PCR amplification further enrich for these chimeric molecules, while the use of unique molecular identifiers allows for accurate quantification and removal of PCR duplicates. The final library is purified using size-selective bead-based cleanup to remove adapter dimers and other artifacts.

- describe providing kit for identifying RNA targets

The invention further provides a commercial kit for performing the method, comprising pre-conjugated oligonucleotide-barcoded antibodies for a panel of RNA binding proteins, ligation and reverse transcription enzymes, buffer systems optimized for proximity ligation, magnetic beads, oligonucleotide adapters, primers for PCR amplification, and detailed protocols for sample preparation, library construction, and data analysis. The kit is designed for use with minimal input material and includes all necessary reagents for multiplexed profiling of up to twenty RNA binding proteins in a single experiment, enabling researchers to rapidly generate comprehensive RNA-protein interaction maps without requiring specialized instrumentation or extensive optimization.

## DETAILED DESCRIPTION

- define terms and phrases used in the application

### Definitions

- define eCLIP

eCLIP refers to an enhanced version of crosslinking and immunoprecipitation that incorporates size selection via SDS-PAGE, the use of size-matched input controls, and standardized bioinformatic pipelines to identify RNA binding sites of RNA binding proteins with high specificity and reproducibility.

- define "about" or "approximately"

The terms “about” or “approximately” when used in reference to a numerical value indicate that the stated value may vary by ±10% unless otherwise specified, accounting for normal experimental variability in measurement, preparation, or instrumentation.

- define "including" and variations

The term “including” and its grammatical variations, such as “includes” or “included,” are used in an open-ended sense to mean “including but not limited to,” and do not exclude additional elements, steps, or components not explicitly listed.

- define "comprising"

The term “comprising” is used in its open-ended sense to mean that the described composition, method, or system includes the recited elements but may also include additional elements not explicitly mentioned.

- define "having"

The term “having” is used synonymously with “comprising” and indicates the presence of the specified features, elements, or steps without excluding the presence of others.

- define "includes"

The term “includes” is used interchangeably with “comprising” and denotes that the listed elements are part of a broader set that may contain additional components or steps.

- define "example"

The term “example” is used to illustrate a specific embodiment of the invention and is not intended to limit the scope of the invention to the details provided therein.

- define "preferably" and variations

The term “preferably” and its variations, such as “more preferably” or “most preferably,” indicate optional features that enhance the utility, efficiency, or performance of the invention but are not essential to its practice.

- define "comprising" in context of process and compound/composition/device

In the context of a process, “comprising” means that the steps recited are required, but additional steps may be performed. In the context of a compound, composition, or device, “comprising” means that the recited components are essential, but other components may be present without departing from the scope of the invention.

### Methods

- introduce method of identifying RNA molecules bound by RNA binding proteins

The method of identifying RNA molecules bound by RNA binding proteins involves the covalent stabilization of in vivo RNA-protein interactions through UV crosslinking, followed by selective capture of these complexes using oligonucleotide-conjugated antibodies, proximity-based ligation of bound RNA to a unique barcode, and high-throughput sequencing to decode the identity of both the RNA and the RNA binding protein responsible for its capture.

- describe contacting RNA sample with RNA binding protein

A biological sample containing RNA binding proteins and their associated RNA transcripts is contacted with a panel of antibodies, each conjugated to a unique oligonucleotide barcode, under conditions that preserve RNA integrity and promote specific binding to the target RNA binding protein.

- describe ligating RNA sample to oligo conjugated entity

After immunoprecipitation and washing, the RNA molecules covalently bound to their cognate RNA binding proteins are ligated to the oligonucleotide barcode via proximity-dependent RNA ligation, creating a chimeric molecule that encodes both the RNA sequence and the identity of the binding protein.

- describe amplifying chimeric RNA or DNA molecules by PCR

The chimeric RNA molecules are reverse transcribed into cDNA and amplified using polymerase chain reaction with primers that flank the barcode-RNA junction, ensuring selective amplification of only those molecules that have undergone successful ligation.

- describe sequencing PCR products

The amplified PCR products are sequenced using high-throughput next-generation sequencing to generate millions of reads, each containing a barcode sequence that identifies the RNA binding protein and a transcript-derived sequence that identifies the bound RNA.

- describe identifying computationally chimeric RNA molecules

Bioinformatic analysis is performed to demultiplex reads by barcode, align transcript sequences to the genome, and identify genomic regions enriched for chimeric reads, distinguishing true binding events from background noise using statistical models.

- describe using specific sequences capable of identifying original complex

The oligonucleotide barcodes contain unique nucleotide sequences that serve as molecular tags, enabling unambiguous identification of the RNA binding protein that captured each RNA molecule, even in multiplexed experiments involving multiple targets.

- describe using randomized sequence to determine uniqueness or PCR duplicate

Unique molecular identifiers are incorporated into the barcode design as randomized nucleotide stretches, allowing for the computational identification and removal of PCR duplicates derived from the same original RNA molecule.

- describe isolating RNA-protein complex

The RNA-protein complex is isolated by incubating the lysed sample with magnetic beads coated with oligonucleotide-conjugated antibodies, followed by magnetic separation and stringent washing to remove non-specifically bound material.

- describe lysing cells prior to isolating complex

Cells are lysed using a detergent-based buffer containing RNase inhibitors and protease inhibitors, followed by sonication and enzymatic digestion to reduce viscosity and remove non-crosslinked RNA, ensuring that only crosslinked RNA-protein complexes remain for capture.

- describe using antibody as oligo conjugated entity

An antibody specific for a given RNA binding protein is covalently linked to an oligonucleotide barcode via click chemistry, enabling the antibody to simultaneously capture the target protein and tag the associated RNA with a unique identifier.

- describe using recombinant Fab, nanobody, or aptamer as oligo conjugated entity

Alternative binding entities such as recombinant Fab fragments, nanobodies, or RNA aptamers may be conjugated to oligonucleotide barcodes in place of full-length antibodies, offering advantages in size, stability, or specificity for certain RNA binding proteins.

- describe using bead as oligo conjugated entity

Magnetic beads themselves may be functionalized with oligonucleotide barcodes and coated with antibodies, serving as both the capture platform and the carrier of the barcode, simplifying workflow and reducing non-specific binding.

- describe selecting specific RNA binding protein

The RNA binding protein to be interrogated is selected based on its known or suspected role in RNA regulation, disease pathology, or biological pathway of interest, and a validated antibody specific for that protein is obtained for conjugation.

- describe selecting chemical crosslink agent

UV light at 254 nm is selected as the crosslinking agent due to its efficiency in forming covalent bonds between RNA and proximal amino acids, though alternative crosslinkers such as psoralen or formaldehyde may be employed depending on the nature of the target interaction.

- describe conjugating oligo to entity using amine or thiol reactive probe

Oligonucleotides are conjugated to antibodies or other entities using NHS ester chemistry targeting lysine residues or maleimide chemistry targeting cysteine residues, ensuring site-specific and stable linkage.

- describe using click chemistry reaction

Click chemistry, specifically copper-free azide-alkyne cycloaddition, is employed to link azide-modified oligonucleotides to DBCO-modified antibodies, enabling efficient, bioorthogonal conjugation under physiological conditions.

- describe removing unreacted probes

Unreacted oligonucleotides and conjugation reagents are removed by buffer exchange using size-exclusion chromatography or dialysis, ensuring that only fully conjugated, functional entities are used in the assay.

- describe using antibody-coupled magnetic bead

Antibody-coupled magnetic beads are used to facilitate rapid separation and washing of immunoprecipitated complexes using a magnetic separator, enabling high-throughput processing and minimizing sample loss.

- describe generating oligo conjugated antibody

An antibody is modified with DBCO groups via NHS ester chemistry, then reacted with azide-labeled oligonucleotides under mild conditions to produce a stable, covalent conjugate that retains both antigen-binding specificity and barcode functionality.

- describe contacting RNA sample with RBP to form complex

The lysed biological sample is incubated with oligonucleotide-conjugated antibodies to allow formation of immune complexes between the antibody, its target RNA binding protein, and any RNA molecules crosslinked to that protein.

- describe isolating complex using labeled antibody

The immune complexes are isolated by capturing the antibody-conjugated beads using a magnetic separator, followed by multiple washes to remove non-specifically bound material, leaving only the RNA binding protein-RNA-oligo complex intact.

- describe ligating RBP bound RNA molecule to oligo on antibody

A proximity-dependent RNA ligation reaction is performed using T4 RNA ligase to covalently join the 3′ end of the crosslinked RNA molecule to the 5′ end of the oligonucleotide barcode attached to the antibody.

- describe amplifying enriched chimeric RNA molecules by PCR

The chimeric RNA-oligo molecules are reverse transcribed and amplified using PCR with primers complementary to the barcode and adapter sequences, enriching for only those molecules that underwent successful ligation.

- describe sequencing PCR products

The PCR-amplified libraries are sequenced on a high-throughput platform to generate reads that contain both the barcode sequence and the transcript-derived sequence, enabling identification of bound RNA and its cognate RNA binding protein.

- describe identifying computationally chimeric RNA molecules

Computational pipelines are used to demultiplex reads by barcode, trim adapters and UMIs, align transcript sequences to the genome, and identify statistically significant peaks of enrichment, distinguishing true binding events from background noise.

- describe generating antibody conjugated to oligonucleotide barcode

Antibodies are chemically modified to carry a unique oligonucleotide barcode via click chemistry, enabling each antibody to serve as both a capture agent and a molecular tag for the RNA it binds.

- describe providing RNA and incubating with conjugated antibody

A lysate containing RNA and RNA binding proteins is incubated with a panel of oligonucleotide-conjugated antibodies, allowing each antibody to capture its target protein and the RNA molecules bound to it.

- describe ligating RNA molecule to oligo on antibody

After immunoprecipitation and washing, a proximity ligation reaction is performed to covalently link the 3′ end of the crosslinked RNA to the 5′ end of the oligonucleotide barcode attached to the antibody.

- describe amplifying enriched chimeric RNA molecules by PCR

The ligated RNA-oligo complexes are reverse transcribed and amplified using PCR with primers flanking the barcode-RNA junction, ensuring that only successfully ligated molecules are amplified.

- describe sequencing PCR products

The amplified products are sequenced using next-generation sequencing to obtain reads that contain both the barcode sequence and the RNA sequence, enabling identification of the bound RNA and its cognate RNA binding protein.

- describe identifying computationally chimeric RNA molecules

Bioinformatic analysis is performed to demultiplex reads, align transcript sequences, and identify enriched genomic regions, with statistical models used to distinguish true binding events from background noise.

- describe combining multiple antibodies in multiplexed mixture

Multiple oligonucleotide-conjugated antibodies, each specific for a different RNA binding protein and bearing a unique barcode, are combined into a single multiplexed mixture, enabling simultaneous profiling of multiple targets from a single sample.

### Kits

- introduce kits containing components for methods and assays

The invention provides a comprehensive kit for performing the method of identifying RNA-protein interactions, containing all necessary reagents, consumables, and instructions for sample preparation, library construction, and data analysis.

- describe kit components, including unconjugated oligos and ligase

The kit includes unconjugated oligonucleotides for custom barcode design, T4 RNA ligase for proximity ligation, and T4 polynucleotide kinase for RNA end repair.

- describe kit components, including RNA binding proteins and antibodies

The kit includes a panel of validated, oligonucleotide-conjugated antibodies specific for a set of RNA binding proteins of interest, as well as unconjugated antibodies for control experiments.

- describe kit components, including conjugation reagents and beads

The kit contains DBCO-NHS and azide-modified oligonucleotides for custom conjugation, magnetic beads pre-functionalized with streptavidin or anti-IgG, and buffer systems optimized for conjugation and ligation.

- describe kit components, including buffers and reagents

The kit includes lysis buffers, high-salt and low-salt wash buffers, RNase inhibitors, protease inhibitors, and buffers for reverse transcription, cDNA ligation, and PCR amplification.

- describe kit components, including adapters and primers

The kit provides sequencing adapters, index primers for dual indexing, and PCR primers designed to amplify the barcode-RNA junction for sequencing library construction.

## EXAMPLES

- provide examples

### Example 1

- describe method for identifying RNA targets

The method for identifying RNA targets begins with the conjugation of oligonucleotide barcodes to antibodies specific for RNA binding proteins using click chemistry. Antibodies are purified and modified with DBCO-NHS to introduce reactive groups, while oligonucleotides are functionalized with azide moieties. The two components are mixed and allowed to react overnight, after which unreacted reagents are removed by desalting. The resulting oligo-conjugated antibodies are coupled to magnetic beads, which are then used to immunoprecipitate RNA binding protein-RNA complexes from UV-crosslinked cells. After washing, RNA ends are repaired, and proximity ligation is performed to join the bound RNA to the barcode. Protein and antibody peptides are digested with proteinase K, and the RNA is reverse transcribed into cDNA. Adapter ligation, PCR amplification, and library purification are performed, followed by sequencing on an Illumina NextSeq 2000. Reads are demultiplexed by barcode, aligned to the human genome, and analyzed to identify enriched binding sites. Results demonstrate that the method recapitulates known RNA binding profiles with high fidelity, matching the specificity and sensitivity of traditional eCLIP while requiring significantly less input material and eliminating gel-based steps.

- conjugate oligonucleotide to antibody

Oligonucleotides are conjugated to antibodies via DBCO-azide click chemistry, resulting in stable, covalent linkage without compromising antibody function.

- purify antibody

Antibodies are purified using protein A/G affinity chromatography and buffer-exchanged into PBS to ensure compatibility with conjugation chemistry.

- attach oligo to antibody through click chemistry

Azide-modified oligonucleotides are reacted with DBCO-modified antibodies under mild aqueous conditions at room temperature for 16 hours, yielding a homogeneous conjugate.

- crosslink cells or tissues with RNA binding proteins

Cells are exposed to 254 nm UV light at 400 mJ/cm² on ice to covalently crosslink RNA binding proteins to their bound RNA transcripts.

- lyse cells and apply to beads

Cells are lysed in detergent-containing buffer, sonicated, and treated with RNase I and DNase to remove non-crosslinked material before being applied to antibody-conjugated magnetic beads.

- wash to remove background

Beads are washed with high-salt, lithium chloride, and low-salt buffers to remove non-specifically bound material while preserving crosslinked complexes.

- repair 3’ RNA ends

RNA 3′ ends are repaired using T4 polynucleotide kinase to generate uniform ends suitable for ligation.

- perform proximity-based intermolecular ligation

T4 RNA ligase is used to ligate the 3′ end of the crosslinked RNA to the 5′ end of the oligonucleotide barcode attached to the antibody.

- wash to remove background

Post-ligation washes are performed to remove unligated RNA and enzyme, ensuring only chimeric molecules remain.

- digest RNA binding protein and antibody peptides

Proteinase K is used to digest the RNA binding protein and antibody, releasing the chimeric RNA-oligo molecule for downstream processing.

- reverse transcribe RNA molecules

Reverse transcription is performed using a primer complementary to the oligonucleotide barcode to generate cDNA.

- clean up cDNA

cDNA is purified using silica-based columns to remove enzymes and nucleotides.

- perform second adapter ligation

A second adapter is ligated to the cDNA to enable PCR amplification and sequencing.

- PCR amplify and clean up libraries

Libraries are amplified using indexed primers and purified using AMPure XP beads.

- sequence libraries

Libraries are sequenced on an Illumina NextSeq 2000 at a depth of 25 million reads per barcode.

- analyze data

Reads are demultiplexed, trimmed, aligned to the human genome, and analyzed for enrichment using CLIPper and custom statistical models.

- trim UMIs and sequencing adapters

Unique molecular identifiers and adapter sequences are removed using bioinformatic tools to enable accurate deduplication and alignment.

- identify RNA targets

Genomic regions enriched for chimeric reads are identified as binding sites for the corresponding RNA binding protein.

- prepare lysis buffer

Lysis buffer is prepared containing Tris, NaCl, Igepal, SDS, and sodium deoxycholate to disrupt membranes while preserving RNA-protein interactions.

- thaw high-salt buffer and no-salt buffer

Buffers are thawed and pre-warmed to room temperature prior to use in washing steps.

- mix magnetic beads with secondary antibody

Magnetic beads are incubated with secondary antibody to facilitate antibody coupling.

- wash magnetic beads

Beads are washed with lysis buffer to remove unbound antibody.

- couple primary antibody to magnetic beads

Primary antibody is added to the beads and rotated for one hour to allow immobilization.

- perform immunoprecipitation

The lysate is added to the bead-antibody mixture and rotated overnight at 4°C.

- prepare no-salt buffer

No-salt buffer is prepared for initial washing steps to minimize non-specific binding.

- prepare high-salt buffer plus

High-salt buffer supplemented with LiCl is prepared for stringent washing.

- perform first immunoprecipitation wash

Beads are washed three times with high-salt buffer to remove non-specifically bound material.

- repeat first immunoprecipitation wash

The high-salt wash is repeated to further reduce background.

- perform RNA end repair

T4 PNK is added to repair 3′ ends of RNA for efficient ligation.

- perform second immunoprecipitation wash

Beads are washed with low-salt buffer to remove residual salts.

- prepare barcode chimeric ligation master mix

Master mix containing T4 ligase, ATP, PEG, and buffer is prepared for proximity ligation.

- perform barcode chimeric ligation

The master mix is added to the beads and incubated for 45 minutes at room temperature.

- perform proteinase digestion

Proteinase K is added to digest proteins and release chimeric RNA-oligo molecules.

- prepare proteinase master mix

Master mix containing proteinase K and SDS is prepared for efficient digestion.

- clean samples with RNA clean and concentrator kit

Samples are purified using a silica-column-based RNA cleanup kit.

- elute RNA samples

RNA is eluted in nuclease-free water.

- wash RNA samples

Washes are performed with ethanol to remove contaminants.

- dry spin RNA samples

Samples are centrifuged to remove residual ethanol.

- elute and store RNA samples

Eluted RNA is stored at −80°C until reverse transcription.

- perform RNA sample preparation

RNA is quantified and assessed for integrity prior to reverse transcription.

- perform reverse transcription of RNA

Reverse transcription is performed using Superscript III and a barcode-complementary primer.

- perform cDNA end repair of samples

cDNA ends are repaired using T4 polynucleotide kinase and Klenow fragment.

- perform cDNA sample bead cleanup

cDNA is purified using magnetic beads.

- perform cDNA ligation on beads

Adapter ligation is performed directly on beads to minimize loss.

- perform ligated cDNA sample cleanup

Ligated cDNA is purified using AMPure XP beads.

- perform cDNA sample quantification by qPCR

qPCR is performed using barcode-specific primers to quantify library yield.

- perform PCR amplification of cDNA and dual index addition

PCR amplification is performed using indexed primers to add sample identifiers.

- perform AMPure library PCR product cleanup

PCR products are purified using magnetic beads.

- analyze library length and concentration via Agilent Tapestation

Libraries are assessed for size distribution and concentration.

- perform agarose gel extraction if necessary

If adapter dimers are present, libraries are gel-purified.

- sequence libraries on an Illumina Nextseq 2000

Libraries are sequenced at high depth.

- map sequencing reads to the human genome

Reads are aligned using STAR aligner.

- split reads into each barcode/RBP computationally

Reads are demultiplexed by barcode to assign each read to its cognate RNA binding protein.

- illustrate results of the protocol in FIG. 6

FIG. 6 illustrates the workflow of the method, highlighting the replacement of SDS-PAGE with proximity ligation.

- illustrate results of the protocol in FIG. 7

FIG. 7 shows genome browser tracks comparing binding sites identified by ABC and eCLIP.

- illustrate results of the protocol in FIG. 8

FIG. 8 demonstrates similar library complexity between ABC and eCLIP.

- illustrate results of the protocol in FIG. 9

FIG. 9 shows enrichment of known motifs in ABC-derived peaks.

### Example 3

- prepare cell pellets

Cell pellets are generated by washing adherent cells with cold PBS and pelleting by centrifugation.

- validate cell viability

Cell viability is confirmed using trypan blue exclusion prior to crosslinking.

- wash cells

Cells are washed twice with cold PBS to remove serum and media components.

- crosslink cells

Cells are exposed to UV light at 254 nm for 400 mJ/cm² on ice.

- prepare oligo for conjugation

Oligonucleotides are synthesized with azide modifications and purified by HPLC.

- conjugate barcodes onto beads

Barcodes are conjugated to beads via click chemistry and validated by fluorescence labeling.

- prepare library

Library preparation reagents are prepared in aliquots and stored at −80°C.

- prepare lysis mix

Lysis buffer is prepared with protease and RNase inhibitors.

- lyse cells

Cells are resuspended in lysis buffer and sonicated.

- sonicate cells

Sonication is performed in cycles to shear DNA and reduce viscosity.

- add RNase-I and Turbo DNase

RNase I and Turbo DNase are added to digest non-crosslinked RNA and DNA.

- incubate and fragment RNA

Incubation is performed at 37°C for five minutes to achieve partial RNA fragmentation.

- pellet cellular debris

Cellular debris is pelleted by centrifugation.

- transfer supernatant

Supernatant is carefully transferred to a new tube for immunoprecipitation.

- prepare lysis buffer and high-salt buffer

Buffers are prepared and aliquoted for use in washing steps.

- couple primary antibody to beads

Antibodies are incubated with magnetic beads for one hour.

- wash beads

Beads are washed with lysis buffer to remove unbound antibody.

- immunoprecipitate

Lysate is added to beads and rotated overnight.

- wash immunoprecipitate

Beads are washed with high-salt and low-salt buffers.

- prepare for first IP wash

Wash buffers are pre-warmed and aliquoted.

- perform first IP wash

Beads are washed three times with high-salt buffer.

- prepare for RNA end repair

T4 PNK buffer and enzyme are prepared.

- perform RNA end repair

RNA ends are repaired using T4 PNK.

- perform second IP wash

Beads are washed with low-salt buffer.

- prepare for barcode chimeric ligation

Ligation master mix is prepared with T4 ligase and PEG.

- prepare Chimeric Ligation master mix

Master mix is prepared in a single tube to minimize variability.

- perform IP tube preparation

IP tubes are labeled and prepared for ligation.

- add Chimeric Ligation master mix to IP tubes

Master mix is added to each IP tube and incubated for 45 minutes.

- perform bead separation and washing

Beads are separated magnetically and washed with high- and low-salt buffers.

- prepare Proteinase master mix

Proteinase K is mixed with SDS and Tris buffer.

- add Proteinase master mix to IP tubes

Proteinase mix is added and incubated at 37°C and 50°C.

- incubate IP tubes

Tubes are incubated with agitation for 40 minutes total.

- clean all samples with Silane beads

Samples are cleaned using Silane-coated beads for cDNA capture.

- prepare Reverse Transcription master mix

RT master mix is prepared with primer, dNTPs, and enzyme.

- add Reverse Transcription master mix to RNA samples

RT mix is added to eluted RNA and incubated at 54°C.

- incubate RNA samples

Incubation is performed for 20 minutes to generate cDNA.

- perform cDNA End Repair

cDNA ends are repaired using T4 PNK and Klenow.

- perform cDNA Sample Bead Cleanup

cDNA is purified using magnetic beads.

- prepare cDNA Ligation master mix

Adapter ligation master mix is prepared with T4 ligase.

- add cDNA Ligation master mix to cDNA samples

Ligation is performed on beads to minimize loss.

- incubate cDNA samples

Ligation is performed overnight at room temperature.

- perform Ligated cDNA Sample Cleanup

cDNA is purified using AMPure XP beads.

- prepare qPCR master mix

qPCR master mix is prepared with Luna dye and primers.

- perform qPCR

qPCR is performed to quantify library yield.

- record qPCR Ct values

Ct values are recorded for each sample.

- prepare PCR amplification reaction mix

PCR mix is prepared with indexed primers and Q5 polymerase.

- calculate PCR cycles

Cycles are calculated based on qPCR quantification.

- perform PCR amplification

PCR is performed to amplify the library.

- perform AMPure Library PCR Product Cleanup

Products are purified using magnetic beads.

- analyze library length and concentration

Library size and concentration are measured using Tapestation.

- sequence libraries

Libraries are sequenced on an Illumina platform.

- map sequencing reads to human genome

Reads are aligned using STAR.

- split reads into each barcode/RBP

Reads are demultiplexed by barcode sequence.

- illustrate results of protocol using Table 16

Table 16 summarizes the number of reads and peaks per RBP.

- illustrate results of protocol using Table 17

Table 17 compares peak overlap between ABC and eCLIP.

- illustrate schematic diagrams of on bead proximity ligation

Schematics illustrate the spatial constraint of ligation on the bead surface.

- illustrate binding sites for various RBPs

Binding sites are mapped across transcript features for each RBP.

- illustrate bar graphs for each RBP

Bar graphs show enrichment of binding in specific genomic regions.

### Example 4

- introduce protocol for amplifying and quantifying individual barcode

A protocol is provided for the amplification and quantification of individual barcode-derived libraries to enable precise normalization and comparison across multiplexed samples.

- list reagents

Reagents include T4 ligase, reverse transcriptase, PCR master mix, indexed primers, and magnetic beads.

- describe amplification

Amplification is performed using indexed primers to add sample identifiers and enable multiplexed sequencing.

- describe cleanup

Cleanup is performed using AMPure XP beads to remove primer dimers and unincorporated nucleotides.

- describe quantification

Quantification is performed using qPCR with barcode-specific primers to determine optimal PCR cycles.

- describe PCR amplification of cDNA and dual index addition

Dual indexing is performed using forward and reverse primers containing unique barcodes for sample multiplexing.

- describe PCR amplification reaction mix preparation

Reaction mix is prepared with Q5 polymerase, dNTPs, and indexed primers.

- describe PCR amplification cycling conditions

Cycling conditions include initial denaturation at 98°C, followed by 10–15 cycles of denaturation, annealing, and extension.

- describe AMPure library PCR product cleanup

PCR products are purified using magnetic beads to remove excess primers and salts.

- describe library length and concentration analysis

Library size and concentration are assessed using Tapestation.

- describe sequencing and read mapping

Libraries are sequenced and aligned to the human genome using STAR.

- describe results illustration in FIG. 11

FIG. 11 illustrates the correlation between barcode signal and input RNA abundance.

- describe results illustration in FIG. 12

FIG. 12 demonstrates the reproducibility of binding site identification across replicates.