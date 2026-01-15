# DESCRIPTION

## FIELD OF THE INVENTION

The present invention pertains to the field of enzymology and industrial biotechnology, specifically relating to lipases derived from fungal sources and engineered variants thereof. More particularly, the invention concerns modified polynucleotides encoding lipase proenzymes or mature lipases with altered propeptide regions that confer improved functional properties such as enhanced thermostability, increased expression yield, or modulated inhibitory behavior. The invention further encompasses recombinant nucleic acid constructs, expression vectors, host cells engineered to produce such lipase variants, methods for their production, and their incorporation into detergent and cleaning compositions. The disclosed technology enables the rational design of lipases with tailored performance characteristics for use in a wide range of industrial applications, including but not limited to laundry detergents, dishwashing formulations, textile processing, and biofuel production.

## BACKGROUND OF THE INVENTION

Lipases (triacylglycerol acylhydrolases, EC 3.1.1.3) are enzymes that catalyze the hydrolysis of ester bonds in triglycerides, releasing free fatty acids, di- and monoglycerides, and glycerol. Among the diverse families of lipases, class 3 lipases—characterized by membership in the α/β hydrolase fold superfamily and the presence of a canonical Ser-His-Asp catalytic triad—are widely distributed across eukaryotes and prokaryotes. Fungal class 3 lipases, such as those from *Rhizomucor miehei* (RmL) and *Thermomyces lanuginosus* (TLL), are of significant commercial interest due to their robust activity under alkaline conditions, compatibility with surfactants, and utility in detergent formulations. These enzymes typically exist in two conformational states: a closed, inactive form where the active site is buried beneath a surface loop termed the “lid,” and an open, active form induced upon interaction with lipid-water interfaces.

A key feature of many class 3 lipases is the presence of an N-terminal propeptide that is cleaved during maturation. This propeptide functions as an intramolecular chaperone, facilitating correct folding of the nascent polypeptide in the secretory pathway. Historically, structural information on lipase propeptides has been lacking, with crystallographic studies limited to the mature enzyme domains. Consequently, the molecular mechanism by which the propeptide influences folding, stability, and activity remained speculative. Although prior art has reported that mutations in the propeptide region can affect lipase activity—suggesting a regulatory role—no high-resolution structural data existed to elucidate how the propeptide interacts with the mature domain or whether it exerts an inhibitory function post-folding.

Limitations in current lipase production systems include low expression yields in heterologous hosts, premature activation leading to cellular toxicity, and suboptimal stability under industrial process conditions. Conventional approaches to improve lipase performance have focused on engineering the mature catalytic domain, often at the expense of expression efficiency or structural integrity. Moreover, the absence of structural insight into propeptide-lipase interactions has hindered rational design efforts targeting this regulatory region. Thus, there remains a critical need for lipase variants with engineered propeptides that decouple folding assistance from inhibitory constraints, thereby enabling higher production titers and tunable activity profiles without compromising enzyme stability.

## SUMMARY OF THE INVENTION

The present invention addresses the aforementioned limitations by providing novel lipase variants wherein specific alterations are introduced into the propeptide region to modulate its interaction with the mature lipase domain. These modifications are designed based on the first high-resolution crystal structures of a class 3 prolipase—specifically, the *Rhizomucor miehei* lipase (ProRmL)—which reveal that the propeptide binds directly over the active site and lid region, sterically occluding substrate access and stabilizing the closed, inactive conformation. This structural insight demonstrates that the propeptide serves not only as a folding chaperone but also as a potent inhibitor of enzymatic activity until proteolytic cleavage occurs late in the secretion pathway.

Accordingly, the invention provides modified polynucleotides encoding lipase variants comprising substitutions, insertions, deletions, or combinations thereof within the propeptide sequence, particularly at residues involved in contact with the mature domain’s lid or hydrophobic anchor regions. Such alterations are shown to reduce inhibitory binding affinity, enhance thermostability, increase expression yield in fungal hosts such as *Aspergillus oryzae*, or prevent premature autoprocessing. In one embodiment, a deletion of residues 95–96 (corresponding to the N-terminus of the mature enzyme) generates a stable prolipase (ProRmL-del) resistant to cleavage, facilitating structural and kinetic studies. In another embodiment, point mutations such as L81V weaken propeptide-mature domain interactions, resulting in elevated lipolytic activity. The invention thus enables the deliberate engineering of propeptide-lipase interfaces to achieve desired functional outcomes, offering a new paradigm for lipase optimization beyond traditional active-site mutagenesis.

## DESCRIPTION OF THE INVENTION

### Definition

As used herein, the term “lipase” refers to a polypeptide exhibiting triacylglycerol lipase activity (EC 3.1.1.3), specifically hydrolyzing ester bonds in water-insoluble triglycerides at a lipid-water interface. “Lipase activity” denotes the catalytic conversion of triglycerides to free fatty acids and partial glycerides, measurable by standard assays such as the GLAD assay using 4-methylumbelliferyl oleate as substrate. “Lipase inhibitory activity” describes the capacity of a propeptide or fragment thereof to reduce or abolish lipase activity when bound to the mature enzyme, as demonstrated by decreased fluorescence in kinetic assays or absence of hydrolysis zones in zymograms. A “coding sequence” is a nucleotide sequence that encodes a polypeptide, excluding untranslated regions. “Control sequences” encompass promoters, terminators, leaders, polyadenylation signals, and other regulatory elements necessary for transcription and translation in a given host. “Expression” refers to the production of a polypeptide from its encoding nucleic acid in a host cell. An “expression vector” is a recombinant DNA molecule containing a coding sequence operably linked to control sequences for expression in a host. A “fragment” is a portion of a full-length polypeptide retaining at least one functional property. A “host cell” is a prokaryotic or eukaryotic cell transformed with a nucleic acid construct for protein production. An “improved property” includes enhanced thermostability, increased specific activity, higher expression yield, or reduced inhibition. A “mature polypeptide” is the enzymatically active form of a lipase following removal of signal peptide and propeptide. A “mature polypeptide coding sequence” encodes this mature form. A “mutant” is a variant differing from a parent sequence by one or more amino acid alterations. A “nucleic acid construct” is a DNA molecule assembled for expression, comprising a coding sequence and control elements. “Operably linked” means that a coding sequence is positioned relative to control sequences to enable transcription and/or translation. A “parent or parent lipase” is the unmodified lipase from which variants are derived. “Sequence identity” is the percentage of identical residues between two aligned sequences over a defined region.

### Conventions for Designation of Variants

Variants are designated according to standard nomenclature indicating the parent residue, position, and substituted residue (e.g., L81V). Amino acid sequences are aligned using structure-guided or sequence-based methods to identify corresponding residues across homologs. Corresponding residues are determined by structural superposition or by alignment using multiple sequence comparison algorithms. Multiple sequence alignments are generated using programs such as MUSCLE or Clustal Omega to identify conserved regions and variable positions. Pairwise sequence comparisons employ algorithms like Needleman-Wunsch or Smith-Waterman to assess local or global similarity. Probabilistic representations, including hidden Markov models (HMMs), are used to define polypeptide families and infer functional residues. Homology models are constructed using templates from known structures to predict variant conformations. MUSCLE v3.8.31 is employed for high-accuracy multiple sequence alignment, particularly for distantly related lipase homologs.

### Muscle v3.8.31 Basic Usage

MUSCLE v3.8.31 is executed via command line with default parameters for progressive alignment followed by iterative refinement. Input consists of a FASTA-formatted file containing amino acid sequences. The program outputs a multiple sequence alignment in various formats (e.g., CLUSTAL, FASTA) suitable for phylogenetic analysis or structural mapping.

### Common Options (for a Complete List Please See the User Guide):

The input file option (-in) specifies the path to the FASTA file containing sequences to align. The output file option (-out) defines the destination and format of the alignment result. The find diagonals option (-diags) accelerates alignment by identifying conserved diagonal segments in the dynamic programming matrix. The maximum number of iterations option (-maxiters) sets the upper limit for refinement cycles, typically 8 for accuracy. The maximum time to iterate option (-maxhours) constrains computational runtime. The log option (-log) records processing details to a specified file for reproducibility.

### Polynucleotides

The invention provides polynucleotides encoding lipase variants comprising a propeptide and mature domain. The propeptide of RmL spans residues 30–94, while the mature polypeptide comprises residues 95–363. Variants exhibit at least 80% sequence identity to the wild-type propeptide or mature domain. Structural analysis identifies propeptide contact zones involving residues 37–88, which interact with the lid (residues 178–186) and hydrophobic anchor (e.g., I250, F345, V348) of the mature enzyme. Alterations include substitutions (e.g., L81V), insertions, or deletions within these zones. Deletions may remove residues 95–96 to prevent cleavage. Such modifications reduce inhibitory binding, enhance thermostability (as measured by Tm shifts in thermal shift assays), or increase expression yield. Contact zones in the mature protein include the lid helix and adjacent hydrophobic patches. Alterations here may indirectly affect propeptide binding or lid dynamics. The total number of alterations ranges from 1 to 20, preferably 1–5. The net effect is a lipase variant with decoupled folding and inhibition, enabling high-yield production of stable, activatable enzymes.

### Nucleic Acid Constructs

Nucleic acid constructs of the invention comprise a polynucleotide encoding a lipase variant operably linked to control sequences. Promoters for bacterial hosts include lac, trp, or T7; for filamentous fungi, gpdA or amyB; for yeast, ADH1 or GAL1. Transcription terminators include bacterial rrnB, fungal cbh1, or yeast CYC1. mRNA stabilizers such as the *Aspergillus* alcA 3′ UTR enhance transcript longevity. Leaders for fungi include the *Aspergillus* glucoamylase leader; for yeast, the α-factor leader. Polyadenylation sequences include SV40 for mammals, but for fungi, the trpC terminator suffices. Signal peptides direct secretion: PelB for bacteria, α-factor for yeast, and native RmL signal (residues 1–29) or *Aspergillus* glucoamylase signal for fungi. Propeptide coding sequences are included to ensure proper folding. The construct is optimized for codon usage in the intended host.

### Expression Vectors

Recombinant expression vectors contain origins of replication, selectable markers, and the nucleic acid construct. Vector components include bacterial (e.g., pUC19 ori), yeast (2μ or CEN/ARS), or fungal (AMA1) replication origins. Vectors are constructed by ligation or Gibson assembly. Compatibility with host cells is ensured by matching replication and selection systems. Autonomously replicating vectors maintain episomally; others integrate via homologous recombination using flanking sequences (e.g., pyrG for *Aspergillus*) or non-homologous end joining. Origins include pMB1 for *E. coli*, 2μ for *S. cerevisiae*, and AMA1 for *Aspergillus*.

### Host Cells

Recombinant host cells include prokaryotes (e.g., *Bacillus subtilis*, *Streptomyces lividans*) and eukaryotes (e.g., *Saccharomyces cerevisiae*, *Aspergillus niger*). Gram-positive bacteria like *Bacillus* spp. are preferred for secretion. Gram-negative hosts include *E. coli*. Fungal hosts include yeasts (*Pichia pastoris*) and filamentous fungi (*Trichoderma reesei*). DNA is introduced via transformation: CaCl₂ for *E. coli*, PEG-mediated protoplast fusion for fungi, or electroporation for yeast.

### Methods of Production

Host cells are cultivated in nutrient media under conditions inducing expression. Variants are recovered from culture supernatants or lysates. Detection employs SDS-PAGE, Western blot, or activity assays. Recovery uses filtration, centrifugation, or chromatography. Purification includes hydrophobic interaction (decylamine agarose), ion exchange (UNO Q), or size exclusion (Superdex 200). Electrophoretic procedures (Native PAGE, zymogram) confirm activity loss in propeptide-bound forms. Differential solubility, extraction, or SDS-PAGE may be used. The host cell itself may serve as an immobilized enzyme source. Production is optimized by media composition, temperature, and induction timing. Lipase preparations show increased lipolytic activity due to reduced autoinhibition. Nucleotide alterations are introduced via site-directed mutagenesis. Contact zones are identified structurally. Sequence identity of variants is ≥80%. Detergent compositions incorporate variants at 0.001–10 mg/g. Activation occurs upon dilution or pH shift. Improved properties include thermostability (Tm >60°C), specific activity (>1000 LU/mg), and expression yield (>1 g/L). Surfactants include anionic (LAS), nonionic (AEO), cationic (quaternary ammonium), and amphoteric (betaines). Builders (citrate, zeolites), bleach systems (peracids, hydrogen peroxide), enzymes (proteases, amylases), and polymers (soil release) are added. Multi-compartment pouches separate incompatible components. Lipase particles are spray-dried or granulated (size 100–1000 μm, protein content 1–50%). Water-soluble films (PVOH) encapsulate doses. Manufacturing involves casting, filling, and heat sealing.

## EXAMPLES

### Materials

Chemicals were from Sigma-Aldrich or Thermo Fisher. Enzymes from New England Biolabs. Media components from Difco. A. oryzae ToC1512 was used as host.

### Example 1: Variant Generation, Transformation and Expression

The RmL gene (UniProt P19515) was cloned into pDAU2 vector under amyB promoter. Site-directed mutagenesis introduced deletions (Δ95–96) or substitutions (L81V). Plasmids were transformed into E. coli DH5α, verified by sequencing, then into A. oryzae via protoplast fusion. Transformants were grown in M400 media at 34°C for 4 days.

### Purification of RmL WT Controls: RmL WT and RmL WT Pur

Culture supernatant was clarified by Seitz/Whatman filtration, then sterile-filtered. Applied to decylamine column; flow-through (ProRmL) and eluate (RmL WT) were collected. RmL WT was further purified by HIC. Native PAGE and zymogram confirmed activity.

### Propeptide Variant Generation

PCR with overlapping primers introduced mutations. DpnI digested template DNA. E. coli transformants yielded plasmids confirmed by sequencing. A. oryzae expression produced variants analyzed by SDS-PAGE and MS.

### Example 2: Native PAGE

Native PAGE (12% resolving gel) separated proteins. Zymogram overlay with olive oil/brilliant green revealed hydrolysis halos only for mature RmL, not ProRmL.

### Example 3: Temperature Stability Assay (TSA)

SYPRO Orange dye bound to unfolding protein. Tm was calculated from fluorescence vs. temperature curves. ProRmL-del showed Tm = 62°C vs. 58°C for RmL WT.

### Example 4: Zymogram Analysis

As in Example 2, no activity was detected for ProRmL or ProRmL-del, confirming inhibition.

### Example 5: Decylamine Agarose Chromatography

Decylamine column separated ProRmL (flow-through) from RmL (eluted with water). Elution profiles confirmed differential binding.

### Example 6: Relative Wash Performance

AMSA used turmeric-stained cotton in model detergent B. Fluorescence decrease measured cleaning. Propeptide variants showed up to 20% improvement over reference lipase at 0.5 ppm dosage.