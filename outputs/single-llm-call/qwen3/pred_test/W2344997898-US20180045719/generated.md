# DESCRIPTION

## TECHNICAL FIELD

- define technical field of virus-like particles

The present invention relates to the field of molecular biology and proteomics, specifically to the use of virus-like particles as a platform for the isolation, stabilization, and analysis of protein-protein interactions and protein-small molecule interactions under native cellular conditions. The invention provides a novel biochemical system in which intracellular protein complexes are encapsulated within self-assembling, non-infectious virus-like particles that are secreted from mammalian cells, thereby preserving the structural and functional integrity of transient, low-affinity, or context-dependent interactions that are typically lost during conventional cell lysis-based methodologies. The virus-like particles are engineered to incorporate bait proteins fused to a viral structural polypeptide, enabling the selective recruitment and entrapment of interacting partners within the lumen or on the surface of the particle. This system enables the capture of protein complexes in their physiological state, without disruption of subcellular architecture, membrane potential, or post-translational modification environments. The invention further extends to the use of such virus-like particles for the identification of cellular targets of small molecules, including drugs and metabolites, through the incorporation of bivalent molecular baits that simultaneously engage a viral scaffold and a small molecule of interest. The resulting particle-based enrichment platform offers a lysis-independent, high-fidelity alternative to traditional co-immunoprecipitation, affinity purification-mass spectrometry, and yeast two-hybrid systems, particularly for the study of dynamic, weak, or membrane-associated interactions that are refractory to conventional biochemical approaches.

## BACKGROUND

- motivate molecular interactions

Protein-protein interactions are fundamental to nearly all biological processes, including signal transduction, metabolic regulation, gene expression, and cellular organization. The accurate identification and characterization of these interactions are essential for understanding disease mechanisms, identifying therapeutic targets, and elucidating the functional architecture of cellular networks. Despite their biological significance, many protein interactions remain undetected due to their transient nature, low binding affinity, or dependence on specific subcellular contexts that are disrupted during experimental manipulation.

- describe protein-protein interactions

Protein-protein interactions occur through a variety of molecular interfaces, often involving structured domains, disordered regions, or post-translationally modified residues. These interactions can be stable and long-lived, forming core components of macromolecular assemblies, or highly dynamic, serving as transient signaling nodes that respond to environmental cues such as ligand binding, phosphorylation, or changes in ion concentration. The stability and specificity of these interactions are frequently influenced by the local microenvironment, including pH, ionic strength, redox state, and proximity to membranes or organelles.

- limitations of yeast two-hybrid

The yeast two-hybrid system, while widely used for binary interaction screening, suffers from significant limitations, including the requirement for nuclear localization of interacting proteins, the artificial reconstitution of transcriptional activation domains, and the frequent occurrence of false positives due to non-physiological protein folding or auto-activation. Moreover, this system is largely incompatible with membrane proteins, cytosolic proteins requiring specific chaperones, or interactions dependent on post-translational modifications not present in yeast.

- introduce phage display

Phage display has been employed to screen peptide or protein libraries for binding partners, offering high throughput and the ability to select for high-affinity binders. However, this method relies on the expression of proteins in a bacterial system, which lacks the eukaryotic machinery necessary for proper folding, glycosylation, or assembly of many human proteins. Furthermore, phage display cannot recapitulate the spatial and temporal dynamics of interactions occurring within intact mammalian cells.

- limitations of phage display

The inability of phage display to preserve native conformational states, the absence of cellular context, and the frequent misfolding of complex eukaryotic proteins render this method unsuitable for the discovery of physiologically relevant interactions. Additionally, phage display is generally restricted to binary interactions and cannot capture multi-protein complexes or interactions requiring co-factors.

- describe Mappit system

The MAPPIT system is a mammalian two-hybrid method that utilizes cytokine receptor signaling to detect protein interactions through luciferase reporter activation. While it operates in a human cellular context and can detect some weak interactions, it is limited by its reliance on artificial fusion architectures, the requirement for specific signaling pathways, and the potential for background activation due to non-specific recruitment of signaling components.

- limitations of Mappit system

MAPPIT is constrained by its dependence on a single signaling cascade, which limits its applicability to proteins not compatible with the receptor scaffold. It also suffers from low sensitivity for interactions that do not induce strong transcriptional activation, and it cannot be used to detect interactions involving membrane proteins that are not properly trafficked to the plasma membrane.

- describe cytoplasmic interaction trap

Cytoplasmic interaction trap systems attempt to detect interactions within the cytosol by reconstituting split reporter enzymes. However, these systems still require the reassembly of artificial fragments and are subject to false positives from non-specific complementation or overexpression artifacts.

- limitations of genetic systems

All genetic interaction systems—yeast two-hybrid, MAPPIT, and cytoplasmic traps—are fundamentally limited by their reliance on reporter gene expression, which introduces a layer of indirect detection that may not reflect true physical proximity. These systems are also highly sensitive to expression levels, often requiring non-physiological overexpression that can induce artificial interactions or mask endogenous ones.

- introduce biochemical strategies

Biochemical strategies such as co-immunoprecipitation and affinity purification coupled with mass spectrometry have become the gold standard for identifying protein interaction networks. These methods rely on the physical extraction of protein complexes from cell lysates using antibodies or affinity tags.

- describe co-purification strategies

Co-purification strategies involve the lysis of cells under defined buffer conditions, followed by the capture of a tagged bait protein and its associated partners using immobilized ligands such as antibodies or affinity resins. The captured complexes are then eluted and analyzed by mass spectrometry to identify interacting proteins.

- limitations of co-purification strategies

The major limitation of co-purification lies in the lysis step itself, which disrupts the native subcellular environment, leading to the dissociation of weak or transient complexes, the mislocalization of membrane-associated proteins, and the introduction of non-specific contaminants from disrupted organelles. Lysis conditions are highly variable and must be empirically optimized for each complex, making large-scale studies labor-intensive and inconsistent.

- describe classical approach to identify target proteins

The classical approach to identifying small molecule targets involves chemical proteomics, wherein biotinylated or fluorescent derivatives of the molecule are used to pull down interacting proteins from cell lysates, followed by mass spectrometric identification.

- limitations of classical approach

This approach is confounded by the need to chemically modify the small molecule, which may alter its binding properties, cellular uptake, or target specificity. Furthermore, lysate-based methods fail to capture interactions that require intact cellular architecture, such as those involving membrane proteins, lipid rafts, or spatially restricted complexes. Background binding from abundant cellular proteins further obscures true targets, necessitating extensive filtering that may eliminate genuine low-abundance interactors.

## BRIEF SUMMARY

- motivate need for new method

There exists a critical need in the field of proteomics for a method that enables the capture of protein complexes and protein-small molecule interactions under native, non-lytic conditions, preserving the structural, spatial, and temporal integrity of biological interactions that are otherwise lost during conventional biochemical extraction. Current methods are either too artificial, too disruptive, or too insensitive to reliably detect weak, transient, or context-dependent interactions.

- describe new method derived from Virotrap

The present invention provides a novel method for the isolation of protein complexes and small molecule targets through the use of engineered virus-like particles that are produced intracellularly and sequester interacting proteins within a protective, lipid-bilayered shell. This method involves the expression of a fusion protein comprising a viral structural polypeptide and a bait protein, which directs the assembly of virus-like particles that encapsulate endogenous proteins interacting with the bait. The particles are then secreted from the cell and captured using surface tags, enabling the purification of intact complexes without cell lysis.

- summarize advantages of new method

This approach eliminates the need for detergent-based lysis, preserves the native state of membrane-associated and labile complexes, reduces background contamination through stringent control comparisons, and enables the detection of interactions that are inaccessible to genetic or biochemical methods. The system is scalable, compatible with high-throughput screening, and applicable to both protein-protein and protein-small molecule interactions.

- introduce artificial VLPs

The invention further encompasses the use of artificial virus-like particles engineered with modified structural proteins, alternative surface ligands, or compartment-specific targeting motifs to expand the range of interactions that can be captured, including those originating from the plasma membrane, endoplasmic reticulum, or other subcellular compartments previously inaccessible to this platform.

## DETAILED DESCRIPTION

### Definitions

- define terms for patent scope

For the purposes of this patent application, the term “comprising” is used in its open-ended sense, meaning that the described composition or method may include additional elements beyond those explicitly recited, without excluding other components. The term “a” or “an” refers to one or more of the recited element, unless the context clearly indicates otherwise.

- explain non-limiting nature of drawings

The drawings provided in this specification are intended to illustrate exemplary embodiments of the invention and are not intended to limit the scope of the claims. Variations in structure, composition, or method steps that do not depart from the essential principles of the invention are encompassed within the scope of the claims.

- define "comprising" and "a" or "an"

As previously defined, “comprising” denotes inclusion without exclusivity, and “a” or “an” denotes singular or plural reference depending on context, with the understanding that the invention may encompass multiple instances of the recited component unless otherwise specified.

- explain use of "first", "second", etc.

The terms “first,” “second,” “third,” and similar ordinal designations are used for the purpose of distinguishing between multiple elements or steps and do not imply any temporal, hierarchical, or preferential order unless explicitly stated.

- provide definitions for terms of art

Terms of art used herein are defined according to their conventional meaning in the fields of molecular biology, virology, and proteomics, unless otherwise explicitly defined herein.

- define "virus-like particle" (VLP)

A virus-like particle, or VLP, is a non-infectious, self-assembling nanostructure composed of one or more viral structural proteins that mimic the morphology of a native virus but lack viral genetic material. In the context of this invention, VLPs are produced intracellularly in mammalian cells and contain encapsulated protein complexes or small molecule-binding partners.

- define "VLP-forming polypeptide"

A VLP-forming polypeptide is a protein or polypeptide sequence derived from a viral structural protein, such as the GAG protein of retroviruses, that is capable of self-assembling into a virus-like particle when expressed in a suitable host cell.

- define "fusion protein"

A fusion protein is a chimeric polypeptide created by the covalent linkage of two or more distinct protein domains or sequences, such as a VLP-forming polypeptide fused to a bait protein, such that both domains retain their functional properties.

- define "polypeptide"

A polypeptide refers to a linear chain of amino acids linked by peptide bonds, whether naturally occurring, synthetic, or recombinantly produced, and includes full-length proteins, fragments, and modified variants thereof.

- define "fusion construct"

A fusion construct is a nucleic acid molecule encoding a fusion protein, operably linked to regulatory elements that enable its expression in a host cell.

- define "recruiting element"

A recruiting element is a molecular moiety, such as a tag, ligand, or antibody-binding domain, that is incorporated into the VLP or its components to facilitate the selective capture or purification of the particle from a complex mixture.

- define "bait"

A bait is a protein, peptide, or small molecule derivative that is used to capture interacting partners, either through direct binding to a protein or through a molecular linker that enables the recruitment of a target molecule.

- define "small molecule"

A small molecule is a low molecular weight organic compound, typically less than 900 Daltons, that interacts with a biological target, including drugs, metabolites, natural products, or synthetic compounds.

- explain "interacts with"

To “interact with” means to engage in a direct or indirect physical association, including binding, complex formation, or spatial proximity sufficient to result in co-entrapment within a virus-like particle under physiological conditions.

- define "recruited to"

To be “recruited to” refers to the process by which a protein or molecule is brought into proximity with a bait or VLP-forming polypeptide through specific molecular recognition, resulting in its entrapment within the forming particle.

- describe VLP derivation

The VLPs described herein are derived from retroviral structural proteins, particularly the GAG polyprotein from human immunodeficiency virus type 1, which retains its ability to assemble into particles when expressed in mammalian cells.

- describe VLP-forming polypeptide modifications

The VLP-forming polypeptide may be modified by the addition of epitope tags, signal peptides, or membrane-targeting sequences to enhance particle secretion, stability, or purification efficiency.

- describe fusion protein embodiments

Fusion proteins may comprise a VLP-forming polypeptide fused at its N-terminus, C-terminus, or internal loop region to a bait protein, with linkers of varying length and flexibility to optimize folding and interaction accessibility.

- describe VLP-forming polypeptide embodiments

The VLP-forming polypeptide may be derived from GAG, M, or other structural proteins of retroviruses, paramyxoviruses, or other enveloped viruses, and may include mutations that enhance particle yield, reduce immune recognition, or increase stability.

- describe small molecule embodiments

Small molecules may include pharmaceutical agents such as simvastatin, tamoxifen, or reversine, or their derivatives, conjugated to a linker molecule such as methotrexate or polyethylene glycol to enable dual engagement with a VLP-forming scaffold and a cellular target.

- describe recruiting element embodiments

Recruiting elements may include FLAG, MYC, VSV-G, or E-tag epitopes, or ligands such as streptavidin-binding peptides, that are fused to the VLP surface or to a co-expressed glycoprotein to enable affinity capture.

- describe bait embodiments

Baits may include signaling proteins such as HRAS, MYD88, FADD, or A20, or enzymes such as eDHFR, and may be expressed as wild-type or mutant variants to probe interaction specificity.

- describe VLP composition

The VLPs comprise a lipid bilayer envelope derived from the host cell membrane, an inner protein shell formed by the VLP-forming polypeptide, and entrapped proteins or small molecule complexes that are recruited during particle assembly.

- describe fusion construct composition

The fusion construct comprises a promoter, a coding sequence for the fusion protein, and a polyadenylation signal, all operably linked in an expression vector suitable for transfection into mammalian cells.

- describe VLP-forming polypeptide interactions

The VLP-forming polypeptide interacts with itself and with cellular proteins involved in particle assembly, including host factors such as ALIX or TSG101, to facilitate budding and secretion.

- describe recruiting element interactions

The recruiting element interacts with an affinity matrix, such as anti-FLAG antibody-conjugated beads, to enable the selective isolation of VLPs from culture supernatant.

- describe bait interactions

The bait interacts with endogenous cellular proteins or small molecules that are present in the same subcellular compartment during VLP assembly, resulting in their co-packaging.

- describe VLP entrapped complex

The VLP entrapped complex comprises the bait protein, its interacting partners, and any associated small molecules, all enclosed within the VLP structure, protected from extracellular degradation.

- describe VLP formation

VLP formation occurs spontaneously upon expression of the VLP-forming polypeptide in a mammalian cell, with the recruitment of interacting proteins occurring during or shortly after particle assembly at the plasma membrane.

- describe VLP composition variations

Variations in VLP composition may include the incorporation of heterologous glycoproteins such as VSV-G, modifications to the lipid content, or the inclusion of fluorescent reporters for visualization.

- describe prey protein embodiments

Prey proteins may be cytosolic, nuclear, or membrane-associated, including transmembrane receptors, kinases, phosphatases, or transcription factors, provided they are accessible during VLP biogenesis.

- describe VLP-forming polypeptide interactions variations

Variations in VLP-forming polypeptide interactions may include the co-expression of wild-type GAG to modulate particle size, or the substitution of GAG with other viral scaffolds such as M from influenza or p24 from HIV-2.

- describe VLP entrapped complex variations

The entrapped complex may contain single or multiple prey proteins, post-translationally modified species, or small molecule ligands, depending on the bait and cellular context.

- describe VLP detection methods

VLPs may be detected by electron microscopy, nanoparticle tracking analysis, Western blotting for structural proteins, or fluorescence microscopy if labeled with a reporter such as EGFP.

- describe VLP analysis methods

Analysis of VLPs may involve mass spectrometry, proteomic profiling, immunoblotting, or functional assays to identify entrapped proteins or small molecule targets.

- describe VLP purification methods

Purification may be achieved through ultracentrifugation, size-exclusion chromatography, or affinity capture using antibody-coated beads or lectin resins.

- describe VLP enrichment methods

Enrichment may be performed by sequential binding to affinity matrices, depletion of abundant contaminants, or differential centrifugation to isolate particles from vesicles or debris.

- describe VLP isolation methods

Isolation methods include filtration, precipitation with polyethylene glycol, or immunoaffinity capture using magnetic beads functionalized with antibodies against surface tags.

- describe VLP analysis variations

Analysis variations may include quantitative proteomics, cross-linking mass spectrometry, or interaction network mapping to distinguish specific from non-specific interactors.

### EXAMPLES

- describe plasmid generation

Plasmids encoding the fusion constructs were generated by PCR amplification of the GAG coding sequence from a packaging plasmid and subsequent cloning into a mammalian expression vector via In-Fusion or Gateway recombination, resulting in constructs expressing GAG fused to various bait proteins including HRAS, A20, and eDHFR.

- describe VLP production

HEK293T cells were transfected with plasmids encoding the fusion constructs and, where applicable, VSV-G and epitope-tagged surface proteins. Supernatants were collected 24 to 48 hours post-transfection and processed for particle isolation.

- describe VLP analysis

VLPs were analyzed by Western blotting for the presence of GAG, bait proteins, and candidate interactors, with particle integrity confirmed by electron microscopy and particle size distribution measured by nanoparticle tracking analysis.

- describe mass spectrometry analysis

Proteins entrapped within VLPs were digested with trypsin, and resulting peptides were analyzed by nano-LC-MS/MS using a Q Exactive mass spectrometer, with data searched against human and viral protein databases using Mascot software.

- describe VLP trapping of simvastatin binders

Cells expressing eDHFR-GAG were treated with a bivalent molecule consisting of methotrexate linked to simvastatin, resulting in the specific entrapment of HMG-CoA reductase and UBIAD1 within VLPs, as confirmed by mass spectrometry.

- describe results of VLP trapping experiment

The VLP trapping experiment revealed consistent enrichment of known simvastatin targets and novel interactors such as SQLE, while control experiments with DMSO or unmodified bait showed no such enrichment, demonstrating specificity.

### Example 1

- describe VLP trapping of simvastatin binders

VLPs were produced in cells expressing eDHFR-GAG and exposed to a methotrexate-simvastatin conjugate. After purification via anti-FLAG affinity capture, entrapped proteins were identified by mass spectrometry, revealing HMGCR as the top hit.

- describe results of VLP trapping experiment

HMGCR was identified with multiple unique peptides across biological replicates, and its enrichment was absent in control experiments lacking the bivalent molecule or expressing untagged GAG.

- describe protein identification methods

Proteins were identified using Mascot database searches with a false discovery rate below 1%, and only proteins identified by at least two unique peptides in two independent experiments were considered confident interactors.

- describe results of protein identification

HMGCR, UBIAD1, and SQLE were consistently identified as specific interactors, while abundant cellular proteins such as actin and albumin were excluded by comparison to 19 control VLP experiments.

### Example 2

- perform viral trapping experiment

A second viral trapping experiment was conducted using A20 as bait, with and without TNFα stimulation. VLPs were purified, and mass spectrometry revealed enrichment of RIPK1, TRADD, and TNFR1 only in the stimulated condition, demonstrating dynamic complex capture.

### Example 3

- perform viral trapping experiment

A third experiment was performed using FADD as bait, resulting in the entrapment of CASP8 and other known apoptotic regulators, confirming the system’s ability to capture well-characterized complexes.

### Example 4

- describe MASPIT assay protocol

MASPIT assays were performed by co-transfecting HEK293T cells with eDHFR bait, HSD17B4 prey, and a luciferase reporter. Cells were treated with MTX-tamoxifen, and luciferase activity was measured after 24 hours.

- confirm interaction between tamoxifen and HSD17B4

Luciferase activity increased significantly only in the presence of the bivalent tamoxifen conjugate and the HSD17B4 prey, confirming a direct or proximal interaction between tamoxifen and HSD17B4, independent of VLP formation.