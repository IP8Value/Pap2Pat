# DESCRIPTION

- incorporate references

The present invention is grounded in a comprehensive body of scientific literature concerning the molecular mechanisms of epigenetic regulation, particularly as they pertain to DNA methylation and its implications in oncogenesis. Numerous prior studies have established the centrality of DNA methylation in the control of gene expression during normal development and its frequent dysregulation in malignant transformation [1–3]. Foundational works by Jones and Baylin have delineated the role of aberrant promoter methylation in silencing tumor suppressor genes, while more recent contributions by Esteller and Herman have expanded the clinical relevance of methylation markers as diagnostic and prognostic tools [4,5]. The methodological evolution of methylation detection, from early restriction enzyme-based assays to bisulfite conversion coupled with PCR amplification, has been extensively documented in the literature [6,7]. Seminal contributions by Warnecke et al. first identified the phenomenon of PCR bias favoring unmethylated templates during amplification of bisulfite-converted DNA, a limitation that has persisted despite widespread adoption of methylation-independent primer strategies [8]. Subsequent investigations by Hesson et al. and Cui et al. have corroborated the prevalence of this bias and its capacity to distort quantitative assessments of methylation levels [9,10]. The development of high-resolution melting analysis (HRM) as a post-PCR analytical technique has provided a sensitive, closed-tube platform for methylation quantification, yet its accuracy remains contingent upon the fidelity of primer design [11,12]. The present invention builds upon these foundational insights by introducing a novel paradigm for the design and application of methylation-independent primers that actively correct for amplification bias through strategic incorporation of CpG dinucleotides and precise modulation of annealing temperature, thereby enabling reliable, reproducible, and quantitatively accurate detection of methylated alleles in clinical specimens.

## FIELD OF INVENTION

- relate to method of detecting methylated CpG-containing nucleic acids

The present invention relates to a method for detecting methylated CpG-containing nucleic acids in biological samples, particularly for the purpose of diagnosing, monitoring, or prognosticating neoplastic and epigenetic disorders. More specifically, the invention pertains to a polymerase chain reaction-based technique that employs methylation-independent oligonucleotide primers designed to hybridize to bisulfite-modified DNA templates containing CpG dinucleotides, wherein the primers incorporate one or more CpG sites at defined positions to modulate hybridization stringency and thereby correct for inherent amplification bias. The method enables the proportional amplification of both methylated and unmethylated alleles under optimized thermal cycling conditions, allowing subsequent analysis of methylation status through melting curve analysis, sequencing, or other post-amplification techniques without the need for methylation-specific primers or complex post-PCR processing. The invention is particularly applicable to clinical diagnostics involving formalin-fixed paraffin-embedded tissues, blood-derived cell-free DNA, and other minimally invasive sample types where accurate quantification of methylation levels is critical for clinical decision-making.

## BACKGROUND OF INVENTION

- introduce DNA methylation

### Importance of DNA Methylation

- define DNA methylation

DNA methylation is a biochemical modification involving the covalent addition of a methyl group to the fifth carbon of a cytosine ring within a DNA molecule, typically occurring at cytosine residues preceding guanine residues in a CpG dinucleotide context. This epigenetic mark is catalyzed by DNA methyltransferase enzymes and is stably inherited through cell division, serving as a fundamental regulator of genomic function.

- motivate DNA methylation

DNA methylation plays a pivotal role in the regulation of gene expression, chromatin structure, genomic imprinting, X-chromosome inactivation, and suppression of transposable elements. Its precise spatiotemporal patterning is essential for normal cellular differentiation and tissue-specific gene activity during embryonic development and adult homeostasis.

- describe methylation reaction

The methylation reaction proceeds through the enzymatic transfer of a methyl group from S-adenosyl methionine to the C5 position of cytosine, resulting in the formation of 5-methylcytosine. This modification does not alter the primary nucleotide sequence but profoundly influences the accessibility of DNA to transcriptional machinery and chromatin-modifying complexes.

- introduce CpG islands

CpG islands are genomic regions characterized by an elevated frequency of CpG dinucleotides relative to the genome-wide average, often spanning promoter regions and first exons of housekeeping and tumor suppressor genes. These regions are typically unmethylated in normal somatic cells, maintaining an open chromatin configuration conducive to transcriptional activity.

- describe methylation of CpG islands

In pathological states, particularly in cancer, CpG islands associated with tumor suppressor genes frequently undergo aberrant hypermethylation, leading to transcriptional silencing and loss of function. This epigenetic alteration is not a random event but occurs in a gene- and tissue-specific manner, contributing directly to the acquisition of malignant phenotypes.

- relate methylation to gene function

Methylation of CpG islands in promoter regions impedes the binding of transcription factors and recruits methyl-CpG-binding domain proteins that facilitate the assembly of repressive chromatin complexes, including histone deacetylases and histone methyltransferases, thereby establishing a transcriptionally inert state.

- describe abnormal methylation in cancer

Aberrant DNA methylation is a hallmark of virtually all human cancers, manifesting as global hypomethylation accompanied by focal hypermethylation of specific tumor suppressor gene promoters. These changes often precede genetic mutations and are detectable in early-stage lesions and even in circulating cell-free DNA.

- introduce methylation as hallmark of cancer

The presence of tumor-specific methylation patterns has emerged as a robust biomarker for cancer detection, classification, risk stratification, and monitoring of therapeutic response. Unlike genetic alterations, methylation marks are reversible and thus represent attractive targets for epigenetic therapies.

- describe methylation patterns in tumour types

Distinct methylation signatures have been identified across diverse tumor types, including colorectal, breast, lung, ovarian, and hematological malignancies, enabling molecular subtyping and the development of tissue-of-origin diagnostics. For instance, hypermethylation of the MGMT promoter in glioblastoma predicts responsiveness to alkylating agents, while silencing of MLH1 in endometrial cancer correlates with microsatellite instability.

- motivate methylation detection

Accurate quantification of methylation levels is essential for distinguishing biologically significant epigenetic alterations from background noise, particularly in heterogeneous samples such as tumor biopsies or liquid biopsies containing admixed normal cells.

- introduce existing methodologies

Existing methodologies for methylation detection include methylation-specific PCR, pyrosequencing, bisulfite sequencing, methylation-sensitive restriction enzyme digestion, and high-resolution melting analysis.

- limitations of existing methodologies

Traditional methylation-specific PCR is inherently qualitative and prone to false positives due to incomplete bisulfite conversion or primer specificity issues. Bisulfite sequencing, while comprehensive, is costly and low-throughput. Pyrosequencing requires specialized instrumentation and is limited in multiplexing capacity. Restriction enzyme-based methods are constrained by the availability of suitable cutting sites and are insensitive to partial methylation. High-resolution melting analysis, though sensitive and rapid, has historically suffered from poor reproducibility when primers are designed to exclude CpG dinucleotides, due to preferential amplification of unmethylated templates.

## SUMMARY OF INVENTION

- introduce method for detecting methylated CpG-containing nucleic acids

The present invention introduces a novel method for detecting methylated CpG-containing nucleic acids by employing methylation-independent oligonucleotide primers that contain one or more CpG dinucleotides strategically positioned within their sequence, combined with precise modulation of the annealing temperature during polymerase chain reaction amplification. This approach enables the proportional amplification of both methylated and unmethylated alleles, thereby correcting for the inherent PCR bias that otherwise favors unmethylated templates.

- describe aspects of the invention

Key aspects of the invention include the deliberate inclusion of CpG dinucleotides in primer sequences at positions distal to the 3′ end to avoid complete methylation specificity, the optimization of annealing temperature to fine-tune hybridization stringency, and the use of standardized thermal cycling conditions that permit quantitative assessment of methylation levels through melting curve analysis. The method is applicable to a broad range of biological samples and enables the detection of methylation at low abundance levels, even in the presence of a vast excess of unmethylated DNA.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce invention for DNA methylation pattern identification

The invention provides a robust, quantitative, and reproducible method for identifying DNA methylation patterns in CpG-rich regions of the genome by overcoming the longstanding limitation of PCR bias in methylation-independent amplification. Through the strategic design of primers containing controlled numbers of CpG dinucleotides and the empirical determination of optimal annealing temperatures, the method achieves balanced amplification of methylated and unmethylated alleles, thereby enabling accurate estimation of methylation proportions in heterogeneous samples. This innovation transforms methylation-independent PCR from a qualitative screening tool into a reliable quantitative platform suitable for clinical diagnostics and longitudinal monitoring.

### DEFINITIONS

- define amplification

Amplification refers to the enzymatic replication of a specific nucleic acid sequence through repeated cycles of denaturation, primer hybridization, and extension, resulting in exponential increase in the number of target molecules.

- describe template copying process

The template copying process involves the binding of complementary oligonucleotide primers to single-stranded DNA templates, followed by the sequential addition of deoxynucleotide triphosphates by a DNA polymerase enzyme to synthesize a new strand complementary to the template.

- define double stranded polynucleotide

A double-stranded polynucleotide is a macromolecular structure composed of two complementary strands of nucleic acids held together by hydrogen bonding between paired nitrogenous bases, forming a stable helical conformation.

- define gene

A gene is a functional unit of heredity comprising a specific sequence of nucleotides that encodes a functional product, such as a protein or RNA molecule, and includes regulatory regions necessary for its expression.

- define nucleotide

A nucleotide is the fundamental building block of nucleic acids, consisting of a phosphate group, a pentose sugar, and a nitrogenous base.

- describe nucleotide structure

The nucleotide structure comprises a five-carbon sugar (deoxyribose in DNA or ribose in RNA), a phosphate moiety attached to the 5′ carbon, and a nitrogenous base—adenine, thymine, cytosine, or guanine in DNA—linked to the 1′ carbon of the sugar.

- define nucleotides

Nucleotides are the monomeric units that polymerize to form polynucleotide chains, with the sequence of bases encoding genetic information.

- describe base pairing

Base pairing refers to the specific hydrogen bonding between complementary nitrogenous bases: adenine with thymine (or uracil in RNA) and cytosine with guanine, ensuring fidelity in DNA replication and transcription.

- define oligonucleotide

An oligonucleotide is a short, synthetic sequence of nucleotides, typically between 15 and 50 bases in length, designed to hybridize to a complementary target sequence.

- describe oligonucleotide structure

An oligonucleotide structure consists of a linear chain of nucleotides connected by phosphodiester bonds between the 3′ hydroxyl group of one sugar and the 5′ phosphate of the next, with exposed 5′ and 3′ termini.

- define polynucleotide

A polynucleotide is a long, continuous chain of nucleotides linked by phosphodiester bonds, forming the backbone of DNA or RNA molecules.

- describe polynucleotide structure

The polynucleotide structure is characterized by a sugar-phosphate backbone with nitrogenous bases projecting laterally, forming a helical conformation in double-stranded contexts and adopting complex secondary structures in single-stranded forms.

- define dinucleotide

A dinucleotide is a molecule composed of two nucleotides joined by a single phosphodiester bond.

- describe CpG dinucleotide

A CpG dinucleotide is a sequence motif in which a cytosine nucleotide is immediately followed by a guanine nucleotide, linked by a phosphodiester bond, and is the primary site of DNA methylation in mammalian genomes.

- define methylation status

Methylation status refers to the presence or absence of methyl groups at cytosine residues within CpG dinucleotides, indicating whether a given site is methylated, unmethylated, or partially methylated.

- describe methylation status detection

Methylation status detection involves the identification and quantification of methylated cytosines in a DNA sample, typically following bisulfite conversion, which differentiates methylated cytosines from unmethylated ones through sequence alteration.

- define PCR bias

PCR bias refers to the non-uniform amplification efficiency between two or more allelic templates during polymerase chain reaction, leading to disproportionate representation of one variant over another in the final amplified product.

- describe PCR bias effects

PCR bias effects manifest as underrepresentation of methylated templates in amplification reactions using methylation-independent primers, resulting in underestimation of true methylation levels and potential failure to detect clinically relevant epigenetic alterations.

### Samples

- describe sample sources

Biological samples suitable for the method include fresh or frozen tissue, formalin-fixed paraffin-embedded tissue sections, blood, serum, plasma, cerebrospinal fluid, urine, sputum, and cell lines derived from neoplastic or normal tissues.

- list preferred sample sources

Preferred sample sources include formalin-fixed paraffin-embedded tumor biopsies, peripheral blood mononuclear cells, and cell-free DNA isolated from plasma or serum.

- describe sample selection

Sample selection is guided by the clinical context, with priority given to specimens containing sufficient nucleic acid yield and integrity to permit reliable bisulfite conversion and amplification.

- list specific sample sources

Specific sample sources encompass colorectal adenocarcinoma tissue, breast carcinoma biopsies, lung tumor specimens, ovarian cancer ascites, and leukemic blasts.

- describe nucleic acid extraction

Nucleic acid extraction is performed using commercially available kits or phenol-chloroform protocols, with optional enrichment for low-abundance targets through carrier DNA or size selection.

- describe direct sample use

Direct sample use is feasible in cases where the concentration of target nucleic acid is sufficiently high, eliminating the need for prior extraction and reducing procedural variability.

- describe target nucleic acid

The target nucleic acid is a region of genomic DNA containing one or more CpG islands associated with genes known to be epigenetically regulated in disease, such as MGMT, BNIP3, CDKN2A, or RASSF1A.

- describe nucleic acid forms

Nucleic acid forms include double-stranded genomic DNA, single-stranded cDNA, and fragmented cell-free DNA, all of which are amenable to bisulfite conversion and subsequent amplification.

- describe nucleic acid mixtures

Nucleic acid mixtures may contain admixed DNA from normal and malignant cells, necessitating the method’s ability to detect low levels of methylation against a background of unmethylated sequences.

- describe genomic nucleic acid preparation

Genomic nucleic acid preparation involves the isolation of high-molecular-weight DNA free of protein and RNA contaminants, followed by quantification and assessment of purity by spectrophotometry or fluorometry.

- describe nucleic acid amount

The amount of nucleic acid used per reaction ranges from 1 ng to 100 ng, with optimal sensitivity achieved at 10 ng of input DNA.

### Modification of DNA

- describe DNA modification step

The DNA modification step involves the chemical conversion of unmethylated cytosine residues to uracil through treatment with sodium bisulfite, while methylated cytosines remain unaltered.

- describe agent for modifying unmethylated cytosine

Sodium bisulfite is the preferred agent for modifying unmethylated cytosine, as it selectively deaminates cytosine to uracil under acidic conditions and elevated temperature.

- describe sodium bisulfite reaction

The sodium bisulfite reaction proceeds through the addition of bisulfite ions to the C5–C6 double bond of cytosine, followed by hydrolytic deamination to form uracil-bisulfite adducts, which are subsequently desulfonated to yield uracil.

- describe PCR amplification of modified nucleic acid

Following bisulfite conversion, the modified nucleic acid is amplified using methylation-independent primers designed to bind to sequences containing both C and T residues derived from original cytosines, allowing amplification of both methylated and unmethylated templates.

### Methylation-Independent Primer

- define methylation-independent primer

A methylation-independent primer is an oligonucleotide sequence designed to hybridize to both methylated and unmethylated versions of a target DNA sequence following bisulfite conversion, without exhibiting preferential binding to either.

- describe hybridization properties

The hybridization properties of the methylation-independent primer are engineered to permit stable binding to templates containing either cytosine or thymine at CpG-derived positions, depending on the methylation status of the original sequence.

- explain CpG dinucleotide role

The inclusion of one or more CpG dinucleotides within the primer sequence enables differential hybridization stability between methylated and unmethylated templates based on the presence or absence of a methyl group on the cytosine residue.

- describe primer design considerations

Primer design considerations include the number and position of CpG dinucleotides, the melting temperature of the primer, the avoidance of secondary structures, and the minimization of primer-dimer formation.

- specify primer length

The primer length is between 18 and 30 nucleotides, with optimal performance observed at 22 to 26 nucleotides.

- provide specific primer length examples

Specific primer length examples include sequences of 22, 24, and 26 nucleotides, each designed to yield a salt-adjusted melting temperature of approximately 65°C.

- describe method for determining methylation status

Methylation status is determined by analyzing the melting profile of the amplified product, wherein the presence of methylated cytosines results in a higher melting temperature compared to unmethylated sequences.

- specify use of methylation-independent primer

The methylation-independent primer is used in conjunction with a thermal cycling protocol that includes a variable annealing temperature to modulate the stringency of hybridization and thereby correct for amplification bias.

- describe CpG island hybridization

Hybridization occurs within or adjacent to CpG islands, where the density of CpG dinucleotides provides sufficient sequence complexity for specific primer binding.

- specify CpG dinucleotide presence

The presence of one to three CpG dinucleotides within the primer sequence is sufficient to enable bias correction without conferring methylation specificity.

- provide CpG dinucleotide location options

CpG dinucleotides may be located anywhere within the primer sequence except within the final three nucleotides at the 3′ end.

- describe 5'-end CpG dinucleotide location

A CpG dinucleotide located near the 5′ end of the primer contributes minimally to hybridization stability and is preferred for maintaining methylation independence.

- specify CpG site positioning

CpG sites are positioned at least four nucleotides from the 3′ end to prevent complete exclusion of unmethylated templates during amplification.

- describe specific hybridization requirements

Hybridization requires a salt-adjusted melting temperature between 58°C and 68°C, with optimal stringency achieved at 60–65°C.

- specify primer identity to target template

The primer must exhibit at least 90% sequence identity to the bisulfite-converted target template to ensure efficient binding and amplification.

- describe primer-template mismatch effects

Mismatched bases introduced by bisulfite conversion reduce hybridization stability, and the strategic placement of CpG dinucleotides compensates for this destabilization in methylated templates.

- explain annealing temperature modulation

Annealing temperature modulation allows fine-tuning of primer binding affinity, enabling selective amplification of methylated templates at higher temperatures and balanced amplification at intermediate temperatures.

- describe stringency of hybridization modulation

Stringency of hybridization is modulated by varying the annealing temperature, buffer salt concentration, and magnesium ion content to achieve desired amplification profiles.

- specify buffer composition and salt concentration effects

Buffer composition containing 10–50 mM KCl and 1.5–3.0 mM MgCl₂ enhances primer binding specificity and reduces non-specific amplification.

- describe temperature modulation preference

Temperature modulation is preferred in the range of 55°C to 70°C, with incremental increases of 1–2°C used to determine the optimal annealing temperature for bias correction.

- specify SEQ ID NO. primer selection

The methylation-independent primers are selected from the group consisting of SEQ ID NO: 128–151, each corresponding to a specific gene locus and optimized for methylation bias correction.

- specify preferred SEQ ID NO. primer selection

Preferred primer pairs include SEQ ID NO: 138 and 139 for MGMT, SEQ ID NO: 142 and 143 for BNIP3, and SEQ ID NO: 130 and 131 for RASSF1A.

- describe SEQ ID NO. 132-139 primer selection

SEQ ID NO: 132–139 comprise primer sequences designed for amplification of CpG islands in the promoter regions of tumor suppressor genes, each containing one or two CpG dinucleotides positioned at least five nucleotides from the 3′ end.

- specify SEQ ID NO. 138 and 139 primer selection

SEQ ID NO: 138 and 139 are used as a primer pair for the MGMT gene, with a salt-adjusted Tm of 64.5°C and one CpG dinucleotide located at position 12 from the 5′ end.

- describe SEQ ID NO. 142 and 143 primer selection

SEQ ID NO: 142 and 143 are used as a primer pair for the BNIP3 gene, containing two CpG dinucleotides and exhibiting a Tm of 63.8°C, enabling robust amplification of methylated alleles at an annealing temperature of 62°C.

- describe SEQ ID NO. 144-151 primer selection

SEQ ID NO: 144–151 are primer pairs designed for the detection of methylation in CDKN2A, MLH1, and APC, each containing one CpG dinucleotide and demonstrating consistent bias correction across a range of DNA inputs.

- specify SEQ ID NO. 128 and 129 primer selection

SEQ ID NO: 128 and 129 are used for the detection of methylation in the promoter of CDH1, with a single CpG dinucleotide at position 14 from the 5′ end and a Tm of 64.2°C.

- specify SEQ ID NO. 130 and 131 primer selection

SEQ ID NO: 130 and 131 are used for the detection of methylation in RASSF1A, with a CpG dinucleotide at position 10 from the 5′ end and a Tm of 63.5°C.

- specify SEQ ID NO. 132 and 133 primer selection

SEQ ID NO: 132 and 133 are used for the detection of methylation in DAPK1, with a CpG dinucleotide at position 16 from the 5′ end and a Tm of 65.1°C.

- specify SEQ ID NO. 134 and 135 primer selection

SEQ ID NO: 134 and 135 are used for the detection of methylation in SOCS1, with two CpG dinucleotides and a Tm of 64.8°C.

- specify SEQ ID NO. 136 and 137 primer selection

SEQ ID NO: 136 and 137 are used for the detection of methylation in FHIT, with a single CpG dinucleotide at position 13 from the 5′ end and a Tm of 63.9°C.

- specify SEQ ID NO. 138 and 139 primer selection

SEQ ID NO: 138 and 139 are used for the detection of methylation in MGMT, with a CpG dinucleotide at position 12 from the 5′ end and a Tm of 64.5°C, enabling detection of methylation levels as low as 1%.

- specify SEQ ID NO. 140 and 141 primer selection

SEQ ID NO: 140 and 141 are used for the detection of methylation in TIMP3, with a CpG dinucleotide at position 11 from the 5′ end and a Tm of 64.0°C.

- specify SEQ ID NO. 142-151 primer selection

SEQ ID NO: 142–151 are used for the detection of methylation in multiple tumor suppressor genes, each optimized for a specific CpG island and validated across clinical specimen types.

- define methylation-independent primer

A methylation-independent primer is an oligonucleotide that binds to both methylated and unmethylated bisulfite-converted templates with sufficient affinity to permit amplification, and whose binding affinity is modulated by annealing temperature to correct for amplification bias.

- specify target genes

Target genes include MGMT, BNIP3, RASSF1A, CDKN2A, DAPK1, SOCS1, FHIT, CDH1, TIMP3, MLH1, APC, and RARB.

- list genes for specific embodiment

In a specific embodiment, the target gene is MGMT.

- list genes for another embodiment

In another embodiment, the target gene is BNIP3.

- list genes for further embodiment

In a further embodiment, the target gene is RASSF1A.

- list genes for preferred embodiment

In a preferred embodiment, the target gene is MGMT.

- list genes for another preferred embodiment

In another preferred embodiment, the target gene is BNIP3.

- specify target polynucleotide sequences

Target polynucleotide sequences are selected from the group consisting of SEQ ID NO: 1–127, each representing a bisulfite-converted region of a gene promoter.

- list genes for specific embodiment with SEQ ID NO.

In a specific embodiment, the gene is MGMT, and the target sequence is SEQ ID NO: 120.

- list genes for another embodiment with SEQ ID NO.

In another embodiment, the gene is BNIP3, and the target sequence is SEQ ID NO: 121.

- list genes for another embodiment with SEQ ID NO.

In another embodiment, the gene is RASSF1A, and the target sequence is SEQ ID NO: 122.

- list genes for another embodiment with SEQ ID NO.

In another embodiment, the gene is CDKN2A, and the target sequence is SEQ ID NO: 123.

- list genes for another embodiment with SEQ ID NO.

In another embodiment, the gene is DAPK1, and the target sequence is SEQ ID NO: 124.

- list genes for another embodiment with SEQ ID NO.

In another embodiment, the gene is SOCS1, and the target sequence is SEQ ID NO: 125.

- describe significance of MGMT gene methylation

Methylation of the MGMT gene promoter is a clinically validated biomarker predictive of response to alkylating chemotherapy in glioblastoma and other central nervous system tumors, and its accurate quantification is critical for therapeutic decision-making.

### Amplifying Step

- introduce amplifying step

The amplifying step involves the polymerase chain reaction of bisulfite-modified DNA using methylation-independent primers under controlled thermal cycling conditions.

- define methylation-independent oligonucleotide primer

The methylation-independent oligonucleotide primer is a synthetic DNA sequence designed to hybridize to both methylated and unmethylated templates following bisulfite conversion, with one or more CpG dinucleotides incorporated to enable temperature-dependent bias correction.

- describe polymerisation reaction

The polymerisation reaction is catalyzed by a thermostable DNA polymerase, which extends the primer along the template strand by sequentially incorporating complementary deoxynucleotide triphosphates.

- list suitable enzymes for polymerisation

Suitable enzymes for polymerisation include Taq DNA polymerase, Stoffel fragment, Platinum Taq, and hot-start polymerases such as KAPA HiFi or Q5.

- specify PCR as preferred method

Polymerase chain reaction is the preferred method for amplification due to its sensitivity, speed, and compatibility with high-resolution melting analysis.

- describe melting step

The melting step involves heating the reaction mixture to a temperature sufficient to dissociate double-stranded DNA into single strands, typically between 92°C and 98°C.

- specify melting temperature range

The melting temperature range is 94°C to 96°C, with a duration of 10 to 30 seconds.

- specify incubation time for melting

The incubation time for melting is 15 seconds per cycle.

- describe annealing step

The annealing step involves lowering the temperature to allow primer hybridization to the complementary template strand.

- specify annealing temperature range

The annealing temperature range is 55°C to 70°C, with optimal correction of bias achieved between 60°C and 65°C.

- specify incubation time for annealing

The incubation time for annealing is 20 to 40 seconds per cycle.

- describe elongation step

The elongation step involves the extension of the primer by the DNA polymerase to synthesize a complementary strand.

- specify elongating temperature range

The elongating temperature range is 68°C to 72°C.

- specify incubation time for elongation

The incubation time for elongation is 20 to 60 seconds per kilobase of amplicon.

- describe cycling process

The cycling process consists of repeated cycles of melting, annealing, and elongation, with the number of cycles adjusted to remain within the exponential amplification phase.

- specify number of cycles

The number of cycles is between 35 and 45, with 40 cycles being optimal for detection of low-abundance methylated alleles.

- describe PCR machine and thermal cycler

The PCR machine is a thermal cycler equipped with precise temperature control and real-time fluorescence detection capabilities.

- specify dyes for real-time PCR

Dyes suitable for real-time PCR include SYBR Green I, EvaGreen, and LCGreen Plus.

- describe real-time PCR and qPCR

Real-time PCR and quantitative PCR refer to the continuous monitoring of amplicon accumulation during amplification using fluorescent dyes or probes.

- describe multiplex PCR

Multiplex PCR enables simultaneous amplification of multiple target sequences in a single reaction using distinct primer pairs labeled with different fluorophores.

- specify variants of PCR technique

Variants of PCR technique include nested PCR, digital PCR, and touchdown PCR, each of which may be adapted for use with the methylation-independent primer set.

- conclude amplifying step

The amplifying step results in the production of a homogeneous population of amplicons suitable for downstream methylation analysis.

### Analysis of Amplified CpG-Containing Nucleic Acids

- introduce analysis of amplified nucleic acids

The analysis of amplified CpG-containing nucleic acids involves the interrogation of the melting behavior of the PCR product to determine the proportion of methylated and unmethylated sequences.

- describe conversion of unmethylated cytosine to uracil

Unmethylated cytosine residues are converted to uracil during sodium bisulfite treatment, while methylated cytosines remain unchanged.

- specify sodium bisulphite as modifying agent

Sodium bisulfite is the sole modifying agent used in the method, as it provides complete and specific conversion of unmethylated cytosines.

- describe PCR-mediated conversion of uracils to thymine

During PCR amplification, uracil residues are recognized as thymine by the DNA polymerase, resulting in the incorporation of adenine opposite the uracil, thereby converting the original cytosine to a thymine in the amplified product.

- explain difference in nucleic acid sequence

The difference in nucleic acid sequence between methylated and unmethylated alleles arises from the retention of cytosine in methylated templates and its replacement by thymine in unmethylated templates after bisulfite conversion and PCR.

- describe analysis of amplified nucleic acid

Analysis of the amplified nucleic acid is performed by high-resolution melting curve analysis, which detects subtle differences in melting temperature caused by sequence heterogeneity.

- list methods for analysis

Methods for analysis include melting curve analysis, high-resolution melting analysis, sequencing, restriction digestion, and capillary electrophoresis.

- specify melting curve analysis

Melting curve analysis is performed by gradually increasing the temperature of the PCR product while monitoring fluorescence to detect the dissociation of double-stranded DNA.

- describe high resolution melting analysis

High-resolution melting analysis utilizes specialized instrumentation capable of acquiring fluorescence data at high temperature resolution, typically 0.02°C increments, to resolve subtle differences in melting profiles.

- introduce melting curve analysis

Melting curve analysis is based on the principle that the thermal stability of a DNA duplex is determined by its base composition and sequence, with methylated sequences exhibiting higher melting temperatures due to the presence of C–G base pairs.

- describe principle of melting curve analysis

The principle of melting curve analysis is that the melting temperature of a PCR product is directly influenced by the number of C–G base pairs retained from methylated cytosines, allowing the relative abundance of methylated alleles to be inferred from the shape and position of the melting curve.

- specify temperature range for melting curve analysis

The temperature range for melting curve analysis is from 65°C to 95°C, with data acquisition occurring at 0.02°C increments.

- specify temperature transitions

Temperature transitions are performed at a rate of 0.1°C per second to ensure equilibrium between melting and re-annealing events.

- describe measurement of fluorescence

Fluorescence is measured using a saturating DNA-binding dye that emits signal only when intercalated into double-stranded DNA.

- specify normalization of melting curves

Melting curves are normalized by setting the baseline fluorescence at the lowest temperature to 100% and the fluorescence at the highest temperature to 0%, enabling direct comparison between samples.

- describe calculation of line of best fit

A line of best fit is calculated using software algorithms to model the derivative of the melting curve, identifying the peak melting temperature and curve shape characteristics.

- specify platforms for melting curve analysis

Platforms suitable for melting curve analysis include the LightCycler 480, CFX96 Touch, and HR-1 instruments.

- describe in-tube methylation assay

The in-tube methylation assay is a closed-system method that requires no post-PCR handling, minimizing contamination risk and enabling high-throughput screening.

- specify estimation of relative amount of methylated CpG-containing nucleic acid

The relative amount of methylated CpG-containing nucleic acid is estimated by comparing the melting profile of the test sample to a series of standard dilutions containing known proportions of methylated and unmethylated DNA.

- describe comparison with standard samples

Comparison with standard samples is performed by overlaying the melting curves of test samples with those of reference mixtures containing 0%, 1%, 5%, 10%, 25%, 50%, 75%, and 100% methylated DNA.

- specify standard samples

Standard samples include commercially available fully methylated DNA and unmethylated DNA from peripheral blood mononuclear cells.

- describe determination of methylation status

Methylation status is determined by the position and shape of the melting peak, with a shift toward higher temperatures indicating a higher proportion of methylated alleles.

- conclude analysis of amplified CpG-containing nucleic acids

The analysis of amplified CpG-containing nucleic acids provides a quantitative, reproducible, and highly sensitive measure of methylation status without the need for sequencing or enzymatic digestion.

### Nucleic Acid Sequencing

- define nucleic acid sequencing

Nucleic acid sequencing is the process of determining the precise order of nucleotides within a DNA molecule.

- describe dideoxy sequencing method

The dideoxy sequencing method, also known as Sanger sequencing, involves the use of chain-terminating dideoxynucleotides to generate a set of DNA fragments of varying lengths, which are separated by capillary electrophoresis to deduce the sequence.

- explain Sanger method or chain termination method

The Sanger method or chain termination method relies on the incorporation of dideoxynucleotides during DNA synthesis, which lack a 3′ hydroxyl group and thus prevent further elongation, resulting in a ladder of fragments that correspond to each nucleotide position.

### Primer Extension

- describe primer extension method

The primer extension method involves the hybridization of a labeled oligonucleotide primer adjacent to a CpG site, followed by single-base extension using a DNA polymerase and fluorescently labeled dideoxynucleotides to determine the methylation status of the targeted cytosine.

### Restriction Enzyme Digestion

- introduce restriction enzyme digestion

Restriction enzyme digestion is a method that exploits the differential cleavage of DNA by enzymes sensitive to cytosine methylation.

- describe exonucelases and endonucleases

Exonucleases degrade nucleic acids from the ends, while endonucleases cleave internal phosphodiester bonds; the method employs endonucleases that recognize methylation-sensitive restriction sites.

- explain specific conversion of unmethylated cytosines to thymines

The conversion of unmethylated cytosines to thymines via bisulfite treatment disrupts the recognition sequence of methylation-sensitive restriction enzymes, preventing cleavage of unmethylated templates.

- describe disruption of restriction endonuclease site

Disruption of the restriction endonuclease site occurs when a CpG dinucleotide within the recognition sequence is converted from C to T, altering the enzyme’s ability to bind and cleave.

- list specific restriction endonucleases

Specific restriction endonucleases include HpaII, MspI, BstUI, and AciI, each of which exhibits differential cleavage activity based on methylation status.

- describe analysis of digested nucleic acid sample

Analysis of the digested nucleic acid sample is performed by gel electrophoresis, where the presence or absence of cleavage products indicates methylation status.

- explain gel electrophoresis

Gel electrophoresis separates DNA fragments by size under an electric field, with smaller fragments migrating farther through an agarose or polyacrylamide matrix.

- describe modified and amplified nucleic acid

The modified and amplified nucleic acid is subjected to restriction digestion following PCR, with the resulting fragment pattern interpreted to infer methylation levels.

- explain restriction endonuclease site disruption

Restriction endonuclease site disruption occurs when bisulfite conversion alters the sequence such that the enzyme no longer recognizes its cleavage site, thereby preserving the integrity of unmethylated amplicons.

- list more specific restriction endonucleases

More specific restriction endonucleases include HhaI, Sau3AI, and NcoI, each with distinct methylation sensitivity profiles.

- describe subsequent analysis

Subsequent analysis involves quantification of cleaved versus uncleaved fragments using densitometry or capillary electrophoresis.

- explain gel electrophoresis

Gel electrophoresis is performed using ethidium bromide or SYBR Safe staining, with band intensity correlating to the relative abundance of methylated and unmethylated alleles.

### Disorders

- introduce disorders

The method is applicable to the detection of epigenetic alterations associated with a wide range of human disorders, particularly malignancies.

- CpG methylation indicative of disorders

Aberrant CpG methylation is indicative of neoplastic transformation, developmental disorders, and neurodegenerative diseases.

- protooncogenes and cancer risk

Demethylation of protooncogene promoters can lead to their overexpression and increased cancer risk.

- demethylation of transposable elements and cancer risk

Global hypomethylation of transposable elements promotes genomic instability and is associated with increased cancer risk.

- tumour suppressor genes and cancer risk

Hypermethylation of tumor suppressor gene promoters results in their silencing and contributes to tumor initiation and progression.

- list of disorders

Disorders include breast cancer, bladder cancer, ovarian cancer, melanoma, prostate cancer, lung cancer, colon cancer, endometrial cancer, leukemia, gastric cancer, cervical cancer, and imprinting disorders such as Beckwith-Wiedemann syndrome and Angelman syndrome.

- breast cancer

Hypermethylation of BRCA1, RASSF1A, and CDH1 is associated with sporadic breast cancer.

- bladder cancer

Methylation of CDKN2A and RARB is a frequent event in urothelial carcinoma.

- ovarian cancer

Methylation of MLH1 and HIC1 is observed in epithelial ovarian cancer.

- melanoma

Methylation of RASSF1A and PTEN is linked to melanoma progression.

- prostate cancer

Methylation of GSTP1 is a hallmark of prostate adenocarcinoma.

- lung cancer

Methylation of APC and CDKN2A is prevalent in non-small cell lung cancer.

- colon cancer

Methylation of MLH1 and MGMT is associated with microsatellite instability and chemotherapy response.

- endometrial cancer

Methylation of PTEN and RASSF1A is common in endometrioid endometrial carcinoma.

- leukaemia

Methylation of CDKN2B and DAPK1 is observed in acute myeloid leukemia.

- gastric cancer and cervical cancer

Methylation of CDH1 and FHIT is frequent in gastric and cervical cancers.

- imprinting disorders

Imprinting disorders such as Prader-Willi and Angelman syndromes are associated with aberrant methylation at imprinted loci.

### Kit

- introduce kit

The invention further provides a kit for performing the method of detecting methylated CpG-containing nucleic acids.

- methylation-independent oligonucleotide primer

The kit contains one or more methylation-independent oligonucleotide primers selected from SEQ ID NO: 128–151.

- reference sample

The kit includes a reference sample comprising fully methylated and unmethylated DNA controls.

- control CpG-containing nucleic acid

The kit contains control CpG-containing nucleic acid derived from commercially available methylated and unmethylated DNA standards.

- additional reagents

Additional reagents include sodium bisulfite conversion reagents, PCR master mix, DNA intercalating dye, and nuclease-free water.

- instructions for performance and interpretation

The kit includes detailed instructions for bisulfite conversion, PCR setup, thermal cycling parameters, and interpretation of melting curve profiles.

- software for calculation and interpretation

The kit includes software for automated analysis of melting curves, calculation of methylation percentages, and generation of standard curves.

- specific embodiments of CpG-containing nucleic acid and primer

Specific embodiments include primer pairs for MGMT (SEQ ID NO: 138 and 139), BNIP3 (SEQ ID NO: 142 and 143), and RASSF1A (SEQ ID NO: 130 and 131), each accompanied by corresponding control templates.

### Use

- use of methylation-independent oligonucleotide primer

The methylation-independent oligonucleotide primer is used for the detection of methylation status of CpG-containing nucleic acids in clinical and research settings, enabling accurate quantification of epigenetic alterations without PCR bias.

### EXAMPLES

### Example 1

- modify CpG containing nucleic acid

CpG-containing nucleic acid extracted from colorectal cancer tissue was subjected to sodium bisulfite modification using a commercial kit.

- perform bisulfite conversion

Bisulfite conversion was performed according to manufacturer’s protocol, with incubation at 98°C for 10 minutes followed by 55°C for 2.5 hours.

- amplify modified CpG containing nucleic acids

The modified DNA was amplified using primers designed according to conventional guidelines, resulting in preferential amplification of unmethylated alleles.

- test PCR bias toward unmethylated allele

Melting curve analysis revealed a single low-temperature peak, indicative of dominant unmethylated amplification.

- redesign primers to address limitations

Primers were redesigned to include one CpG dinucleotide at position 12 from the 5′ end, with a Tm of 64.5°C.

- show improved sensitivity of assay

Re-amplification using the redesigned primers at an annealing temperature of 62°C yielded a biphasic melting curve, indicating the presence of both methylated and unmethylated alleles.

- illustrate melting profiles for PCR product

Melting profiles demonstrated a clear shift in peak temperature corresponding to methylation levels as low as 1%.

- discuss results of redesigned primers

The redesigned primers corrected PCR bias and enabled accurate quantification of methylation levels previously undetectable with conventional primers.

### Example 2

- validate approach using four assays

The redesigned primer set was validated using four independent assays: melting curve analysis, pyrosequencing, bisulfite sequencing, and MethylLight.

- test redesigned primers with defined mixtures

Defined mixtures of methylated and unmethylated DNA ranging from 0% to 100% were amplified using the new primers.

- show amplification of methylated allele

The methylated allele was consistently amplified across all dilutions, with a strong correlation between melting temperature and methylation percentage.

- discuss importance of primer design

The results underscored the critical importance of CpG inclusion in primer design for bias correction.

- highlight advantages of modified MS-MCA methodology

The modified methylation-sensitive melting curve analysis demonstrated superior sensitivity, reproducibility, and cost-effectiveness compared to existing methods.

- describe features of the method

Key features include closed-tube analysis, no post-PCR handling, and compatibility with standard PCR instrumentation.

### Example 3

- introduce methylation-sensitive high resolution melting (MS-HRM)

Methylation-sensitive high-resolution melting is introduced as a novel approach for sensitive and high-throughput assessment of methylation.

- describe MS-HRM as a new approach for sensitive and high-throughput assessment of methylation

MS-HRM combines the bias-correcting primer design with high-resolution melting analysis to enable quantitative detection of methylation at low abundance levels.

- describe DNA samples and controls

DNA samples were extracted from colorectal cancer tissues and cell lines, with CpGenome Universal Methylated DNA and peripheral blood mononuclear cell DNA used as positive and negative controls.

- extract DNA from colorectal cancer samples

DNA was extracted using a silica-column-based method, with quantification by Qubit.

- purify DNA from cell lines

Cell lines were lysed, and DNA was purified using phenol-chloroform extraction.

- use CpGenome Universal Methylated DNA as a positive/methylated control

CpGenome Universal Methylated DNA was used as a 100% methylated reference.

- use DNA from peripheral blood mononuclear cells as a negative/unmethylated reference

Peripheral blood mononuclear cell DNA served as the unmethylated reference.

- create range of methylated and unmethylated allele dilutions

Serial dilutions of methylated DNA in unmethylated DNA were prepared to generate standards from 0% to 100%.

- perform bisulphite modification of DNA

Bisulfite modification was performed using the EZ DNA Methylation-Direct Kit.

- describe PCR amplification and high resolution melting analysis

PCR amplification was performed using the MGMT-specific primer pair (SEQ ID NO: 138 and 139), followed by high-resolution melting on a LightCycler 480.

- perform PCR amplification

PCR amplification was performed with 40 cycles, 62°C annealing temperature, and EvaGreen dye.

- perform high resolution melting analysis

High-resolution melting was performed with a ramp rate of 0.02°C/s from 65°C to 95°C.

- normalize melting curves

Melting curves were normalized using the instrument’s built-in algorithm.

- describe MGMT MethylLight assay

MGMT MethylLight assay was performed using TaqMan probes as a reference method.

- perform MGMT MethylLight assay

MethylLight assays were conducted in triplicate, with results analyzed using standard curves.

- design primers for MS-HRM assays

Primers were designed to amplify a 98-bp region of the MGMT promoter.

- describe sensitivity of MS-HRM assay

The MS-HRM assay detected methylation levels as low as 0.5%.

- test sensitivity of MGMT MS-HRM assay

The MGMT MS-HRM assay demonstrated 100% concordance with MethylLight at methylation levels above 5%.

- redesign primers to amplify shorter fragments

Primers were redesigned to amplify a 60-bp fragment (MGMT MS-HRM2) and a 45-bp fragment (MGMT MS-HRM3).

- test MGMT MS-HRM2 and MGMT MS-HRM3 assays

Both assays maintained sensitivity and reproducibility, with MGMT MS-HRM3 showing improved signal-to-noise ratio.

- profile methylation content of samples by MS-HRM

MS-HRM profiles were generated for 19 colorectal cancer samples.

- test consistency of normalized melting profiles

Normalized melting profiles were highly consistent across replicates, with coefficient of variation less than 5%.

- describe limitations of short products

Short amplicons (<50 bp) exhibited reduced resolution between methylated and unmethylated peaks.

- describe limitations of long products

Long amplicons (>150 bp) showed broader melting transitions, reducing precision.

- validate MS-HRM results against MethylLight assay

MS-HRM results showed 94% concordance with MethylLight across all samples.

- apply MGMT MS-HRM assay to cell lines

Eight cell lines were tested, with methylation levels ranging from 0% to 85%.

- test DNA from eight cell lines

All cell lines showed expected methylation patterns consistent with published data.

- apply MGMT MS-HRM assay to clinical specimens

Nineteen colorectal cancer specimens were analyzed.

- test panel of 19 colorectal cancer samples

Twelve samples showed methylation levels above 10%, with three showing levels below 5%.

- verify accuracy of MS-HRM approach

Accuracy was verified by bisulfite sequencing of selected samples.

- develop BNIP3 MS-HRM assay

A BNIP3 MS-HRM assay was developed using primer pair SEQ ID NO: 142 and 143.

- test BNIP3 MS-HRM assay

The assay demonstrated high sensitivity and reproducibility.

- apply BNIP3 MS-HRM assay to cell lines

Six cell lines were tested, with methylation levels ranging from 0% to 70%.

- apply BNIP3 MS-HRM assay to clinical specimens

Twenty-two clinical specimens showed methylation levels correlating with tumor stage.

- discuss limitations of other methods

Traditional MSP showed false positives, pyrosequencing required expensive equipment, and bisulfite sequencing was low-throughput.

- discuss strengths and weaknesses of genomic sequencing

Genomic sequencing provides base-pair resolution but is costly and impractical for routine diagnostics.

- discuss strengths and weaknesses of pyrosequencing

Pyrosequencing is quantitative but limited in multiplexing and requires specialized instrumentation.

- discuss strengths and weaknesses of methylation-specific PCR

MSP is simple but qualitative and prone to false positives due to incomplete conversion.

### Example 4

- introduce DNA methylation analysis

DNA methylation analysis is a cornerstone of epigenetic diagnostics in oncology.

- motivate melting curve assays

Melting curve assays offer a rapid, closed-tube alternative to sequencing and restriction-based methods.

- describe bisulfite modification

Bisulfite modification was performed using the EZ DNA Methylation-Direct Kit, with modifications to incubation time and temperature to optimize conversion efficiency.

- explain PCR amplification

PCR amplification was performed using a hot-start Taq polymerase to minimize non-specific amplification.

- introduce DNA melting

DNA melting refers to the dissociation of double-stranded DNA into single strands upon heating.

- describe melting profile

The melting profile is a plot of fluorescence versus temperature, reflecting the thermal stability of the amplicon.

- explain fluorescence detection

Fluorescence detection is mediated by a saturating dye that binds preferentially to double-stranded DNA.

- motivate high resolution melting

High-resolution melting provides superior resolution of heteroduplexes and methylation differences compared to conventional melting.

- introduce methylation-sensitive high resolution melting

Methylation-sensitive high-resolution melting is a method that exploits the differential melting behavior of methylated and unmethylated amplicons to quantify methylation levels.

- describe primer design

Primer design followed the guidelines of including one to three CpG dinucleotides, avoiding the 3′ end, and matching Tm within 1°C.

- list materials for bisulfite modification

Materials included sodium bisulfite, hydroquinone, and sodium acetate.

- describe instrumentation for melting analysis

Instrumentation included the LightCycler 480 II with high-resolution melting module.

- list DNA saturating dyes

Dyes included EvaGreen, LCGreen Plus, and SYBR Green I.

- describe sodium bisulfite treatment

Sodium bisulfite treatment was performed in a thermocycler with a 10-minute denaturation step followed by 2.5 hours at 55°C.

- outline bisulfite conversion steps

Steps included denaturation, bisulfite incubation, desulfonation, and DNA recovery.

- provide protocol for bisulfite conversion

Protocol included 100 ng DNA, 200 µL bisulfite solution, 1-hour incubation at 98°C, 2.5 hours at 55°C, and purification using magnetic beads.

- introduce primer design for PCR amplification

Primer design for PCR amplification required the inclusion of CpG dinucleotides and avoidance of secondary structures.

- limitations of traditional primer design rules

Traditional rules that exclude CpG dinucleotides result in PCR bias and underestimation of methylation.

- propose new guidelines for MIP primer design

New guidelines propose inclusion of one to three CpG dinucleotides, placement at least five nucleotides from the 3′ end, and Tm of 64–66°C.

- importance of annealing temperature manipulation

Annealing temperature manipulation is critical for bias correction, with optimal temperatures determined empirically.

- good practices for primer design

Good practices include checking for hairpins, dimers, and cross-hybridization using OligoAnalyzer.

- predict melting behavior of sequence of interest

Melting behavior was predicted using MeltWin and Primer3.

- tools for predicting melting profiles

Tools included OligoCalc, IDT OligoAnalyzer, and NEB Tm Calculator.

- predict melting temperature of methylated/unmethylated PCR amplicons

Melting temperatures of methylated and unmethylated amplicons were predicted to differ by 1–3°C.

- importance of melting temperature differences

Melting temperature differences of at least 1°C are required for reliable discrimination.

- PCR amplification of bisulfite-modified DNA

PCR amplification was performed using 10 ng DNA, 0.5 µM primers, and 1.5 mM MgCl₂.

- differences in PCR reagents and suppliers

Different polymerase suppliers yielded variable amplification efficiency, with KAPA HiFi showing superior performance.

- importance of hot start protocols

Hot start protocols prevent non-specific amplification during reaction setup.

- role of Mg+2 concentration

Mg²⁺ concentration affects primer binding and polymerase activity, with 1.5 mM being optimal.

- importance of empirically adjusting Mg+2 concentration

Mg²⁺ concentration must be empirically optimized for each primer pair.

- bisulfite template input for PCR amplification

Input of 10 ng bisulfite-modified DNA yielded optimal signal-to-noise ratios.

- importance of high-quality DNA for bisulfite modification

High-quality, intact DNA is essential for complete bisulfite conversion.

- use of carrier DNA for higher recovery rates

Carrier DNA (e.g., glycogen) improved recovery of low-input samples.

- sensitivity of melting assay correlated to input DNA

Sensitivity increased with higher DNA input, with 1 ng being the lower limit of detection.

- PCR amplification parameters

Parameters included 40 cycles, 15-second denaturation, 30-second annealing, and 30-second extension.

- importance of stopping PCR before plateau phase

Stopping PCR before the plateau phase ensures quantitative accuracy.

- re-annealing of PCR product

Re-annealing of PCR product during melting analysis allows formation of heteroduplexes, enhancing resolution.

- design of temperature gradient for melting analyses

Temperature gradients of 0.02°C/s were used for high-resolution melting.

- acquisition of fluorescence

Fluorescence was acquired continuously during the melting ramp.

- importance of precise and accurate fluorescence acquisition

Precise acquisition is essential for reproducible curve shapes.

- settings for data collection on HRM instruments

Settings included 10 acquisitions per degree and 10-second equilibration between steps.

- analysis of results

Results were analyzed using derivative peak analysis and curve normalization.

- derivative peak analysis

Derivative peak analysis identified the melting temperature as the point of maximum fluorescence change.

- direct visualization of melting

Direct visualization confirmed the presence of multiple peaks in heterogeneous samples.

- normalization of HRM curves

Normalization was performed using the instrument’s built-in algorithm.

- estimation of methylation levels

Methylation levels were estimated by comparing sample curves to a standard curve generated from known dilutions.

- notes on bisulfite-modified DNA storage

Bisulfite-modified DNA should be stored at –20°C and used within one week.

- notes on standards for MS-HRM analyses

Standards must be prepared fresh and stored in aliquots to prevent degradation.

- notes on PCR bias and sensitivity of detection

PCR bias remains a critical factor, and the method’s sensitivity is directly dependent on primer design and annealing temperature optimization.

### Example 5

- introduce MS-HRM protocol

The MS-HRM protocol integrates methylation-independent primer design with high-resolution melting analysis for quantitative methylation detection.

- describe PCR amplification of bisulfite modified DNA templates

PCR amplification was performed using 10 ng bisulfite-modified DNA, 0.5 µM primers, and 1.5 mM MgCl₂ in a 25 µL reaction volume.

- explain high resolution analysis of PCR product

High-resolution analysis was performed using a LightCycler 480 with 0.02°C temperature increments.

- outline methodology for single locus methylation studies

The methodology enables single-locus methylation studies with high sensitivity and throughput.

- motivate primer design for correction of PCR bias

Primer design is motivated by the need to correct for PCR bias that otherwise obscures low-level methylation.

- describe PCR bias and its effects

PCR bias leads to underrepresentation of methylated alleles, resulting in false-negative results.

- outline primer design guidelines

Guidelines include inclusion of one to three CpG dinucleotides, placement at least five nucleotides from the 3′ end, and Tm of 64–66°C.

- specify primer design rules

Primer design rules prohibit CpG dinucleotides within the last three nucleotides and require matching Tm within 1°C.

- list materials needed

Materials included sodium bisulfite, DNA extraction kit, PCR master mix, EvaGreen dye, and nuclease-free water.

- specify reagents required

Reagents required were Taq polymerase, MgCl₂, dNTPs, and primer pairs.

- describe bisulfite modification kit

The EZ DNA Methylation-Direct Kit was used for bisulfite modification.

- list DNA intercalating dyes

Dyes included EvaGreen, LCGreen Plus, and SYBR Green I.

- specify Taq polymerase and master mixes

KAPA HiFi HotStart ReadyMix was the preferred polymerase.

- list equipment needed

Equipment included a thermal cycler, high-resolution melting fluorimeter, and centrifuge.

- describe high resolution melting fluorimeter

The LightCycler 480 II was used for melting analysis.

- outline bisulfite modification of genomic DNA

Genomic DNA was treated with sodium bisulfite according to manufacturer’s instructions.

- caution against bisulfite modified template degradation

Bisulfite-modified DNA is fragile and should be handled with care to prevent fragmentation.

- describe DNA saturating dyes

DNA saturating dyes bind stoichiometrically to double-stranded DNA and fluoresce upon binding.

- specify Mg+2 concentration

Mg²⁺ concentration was optimized at 1.5 mM for all primer pairs.

- describe methylated and unmethylated references

Methylated reference was CpGenome Universal Methylated DNA; unmethylated reference was peripheral blood mononuclear cell DNA.

- outline dilution series of methylated in unmethylated controls

Dilution series ranged from 0% to 100% methylated DNA in 5% increments.

- describe equipment needed for MS-HRM experiments

Equipment included a thermal cycler, fluorimeter, and computer with analysis software.

- outline procedure for MS-HRM

Procedure included DNA extraction, bisulfite conversion, PCR amplification, and high-resolution melting analysis.

- describe DNA template extraction

DNA was extracted using a silica-column method with proteinase K digestion.

- outline sodium bisulfite modification of DNA

DNA was incubated with bisulfite solution at 98°C for 10 minutes, then at 55°C for 2.5 hours.

- describe HRM scans and data analysis

HRM scans were performed with 0.02°C increments, and data were analyzed using the instrument’s software.

- outline troubleshooting and anticipated results

Troubleshooting included checking for primer dimers, optimizing Mg²⁺ concentration, and verifying bisulfite conversion efficiency. Anticipated results included distinct melting peaks corresponding to methylation levels.