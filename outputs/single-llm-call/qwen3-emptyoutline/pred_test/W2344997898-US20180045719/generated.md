# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a novel and non-obvious method for the isolation, stabilization, and identification of native protein complexes within mammalian cells without the need for cell lysis or harsh biochemical extraction conditions. Specifically, the invention encompasses a system and method for the capture of endogenous protein-protein interactions by exploiting the natural assembly and secretion of virus-like particles (VLPs) derived from the retroviral GAG polyprotein. The system enables the physical entrapment of interacting protein partners within the lumen or on the surface of these secreted VLPs during their biogenesis in living cells, thereby preserving the structural and functional integrity of transient, weak, or labile complexes that are typically lost during conventional affinity purification-mass spectrometry (AP–MS) workflows. The invention further includes a streamlined, single-step purification protocol utilizing a co-expressed, epitope-tagged viral envelope glycoprotein to enable efficient, high-yield, and scalable recovery of VLPs from cell culture supernatants using standard antibody-based capture techniques. This approach provides a lysis-independent platform for the discovery and validation of binary and multiprotein interactions, including those involving membrane-associated, low-affinity, or dynamically regulated complexes, as well as interactions between proteins and small molecule ligands. The method is particularly suited for applications in systems biology, drug target identification, mechanistic pharmacology, and the characterization of disease-associated interactomes under physiologically relevant conditions.

## BACKGROUND

The study of protein-protein interactions (PPIs) is fundamental to understanding cellular signaling, regulatory networks, and disease mechanisms. Traditional methods for identifying protein complexes, such as co-immunoprecipitation followed by mass spectrometry (AP–MS), rely on the disruption of cellular architecture through lysis, which introduces significant artifacts and biases into the analysis. Lysis conditions—including detergent composition, ionic strength, pH, temperature, and the presence of protease or phosphatase inhibitors—profoundly influence the stability of protein complexes, often leading to the dissociation of weak or transient interactions, the aggregation of non-specific binders, or the loss of post-translational modifications critical for complex integrity. Numerous studies have demonstrated that even subtle variations in lysis protocols can yield dramatically different interactome profiles for the same bait protein, necessitating extensive optimization for each individual complex and rendering large-scale, reproducible studies impractical. Moreover, the extraction process itself removes protein complexes from their native subcellular microenvironment, eliminating spatial constraints, concentration gradients, and molecular crowding effects that govern binding kinetics and complex assembly in vivo.

To address these limitations, alternative lysis-independent approaches have been developed, such as proximity-dependent biotinylation techniques like BioID and APEX2, which label proteins in close spatial proximity to a bait protein fused to a promiscuous biotin ligase or peroxidase. While these methods preserve cellular context and can capture transient interactions, they are inherently limited by the diffusion radius of the labeling enzyme, the efficiency of biotin incorporation, and the potential for non-specific labeling of abundant or membrane-proximal proteins. Furthermore, these techniques do not physically isolate the complex; instead, they rely on subsequent affinity capture of biotinylated proteins, which may still co-purify with non-interacting proteins that are merely in proximity. Other binary interaction systems, such as yeast two-hybrid or mammalian two-hybrid assays, suffer from low physiological relevance due to the artificial relocalization of proteins to the nucleus and the requirement for transcriptional readouts that may not reflect true binding affinity or stoichiometry.

Despite these advances, there remains a critical unmet need for a method that not only preserves the native state of protein complexes but also physically isolates them in an intact, purified form suitable for downstream biochemical, structural, or functional analysis. Existing platforms lack the ability to simultaneously achieve three key objectives: (1) maintaining complex integrity under native conditions without lysis; (2) enabling efficient, scalable, and reproducible purification of the complex from complex biological matrices; and (3) providing a platform capable of capturing both binary and multiprotein interactions, including those involving membrane proteins, low-affinity binders, and small molecule ligands. In particular, no prior technology has demonstrated the capacity to encapsulate entire protein complexes within a self-assembling, secreted, virus-like particle that serves as both a protective nanocontainer and a purification handle.

The present invention overcomes these longstanding challenges by introducing a novel system wherein a bait protein is genetically fused to the retroviral GAG polyprotein, resulting in the co-packaging of the bait and its interaction partners into virus-like particles during their assembly and budding from the plasma membrane. The VLPs, which are naturally secreted into the extracellular space, retain the full complement of interacting proteins within their lumen or on their surface, shielded from extracellular proteases and denaturing agents. This physical encapsulation ensures that the native architecture of the complex is preserved throughout purification, which is achieved through a simple, single-step capture using a co-expressed, epitope-tagged envelope glycoprotein that is incorporated into the VLP membrane. The result is a robust, scalable, and highly specific platform for the isolation of protein complexes under conditions that faithfully reflect their in vivo state, enabling the detection of interactions that are invisible to conventional AP–MS and other proximity-based methods.

## BRIEF SUMMARY

The present invention provides a novel method for the detection and isolation of native protein complexes in mammalian cells through the use of virus-like particles (VLPs) engineered to encapsulate protein-protein interactions during their biogenesis. The method comprises the steps of: (a) expressing a fusion protein comprising a bait protein linked to a retroviral GAG polyprotein in a mammalian cell; (b) co-expressing a prey protein that interacts with the bait protein; (c) allowing the GAG-bait fusion protein to direct the assembly and budding of VLPs from the plasma membrane, wherein the prey protein is physically entrapped within the VLPs as a consequence of its interaction with the bait protein; (d) co-expressing a tagged version of a viral envelope glycoprotein, wherein the tagged envelope glycoprotein is incorporated into the lipid bilayer of the VLPs; (e) harvesting the cell culture supernatant containing the secreted VLPs; and (f) isolating the VLPs from the supernatant by affinity capture using an antibody or binding agent specific to the tag on the envelope glycoprotein. The resulting purified VLPs contain the bait protein, the prey protein, and any additional proteins that form a stable complex with the bait under native cellular conditions, thereby enabling the identification of protein-protein interactions without the need for cell lysis or detergent-based extraction.

In a preferred embodiment, the retroviral GAG polyprotein is derived from human immunodeficiency virus type 1 (HIV-1), and the viral envelope glycoprotein is derived from the vesicular stomatitis virus glycoprotein (VSV-G). The tag on the envelope glycoprotein is selected from the group consisting of FLAG, MYC, HA, VSV, or E-tag, and is fused in-frame to the N-terminus or C-terminus of the envelope glycoprotein. The bait protein is fused to the N-terminus of the GAG polyprotein, and the prey protein is expressed as a separate construct bearing a compatible epitope tag for detection or purification. The method further includes the step of analyzing the captured VLPs by mass spectrometry, western blotting, or other protein detection methods to identify the prey proteins and any additional interacting partners.

In another embodiment, the method is adapted for the detection of interactions between proteins and small molecules by fusing a small molecule-binding protein, such as the Escherichia coli dihydrofolate reductase (eDHFR), to the GAG polyprotein and exposing the producing cells to a bivalent ligand comprising a small molecule of interest linked via a flexible spacer to a ligand that binds the small molecule-binding protein. The bivalent ligand acts as a molecular bridge, enabling the recruitment of proteins that interact with the small molecule of interest into the VLPs, thereby allowing their identification through subsequent purification and proteomic analysis.

The invention further provides a kit for performing the method, comprising: (a) a first plasmid encoding the GAG-bait fusion protein under the control of a mammalian promoter; (b) a second plasmid encoding the prey protein under the control of a mammalian promoter; (c) a third plasmid encoding a tagged viral envelope glycoprotein; and (d) instructions for co-transfecting the plasmids into mammalian cells and performing the VLP capture and analysis steps. The invention also encompasses a library of plasmids encoding diverse bait proteins fused to GAG, each configured for high-throughput screening of protein interaction networks.

The method of the invention provides significant advantages over prior art: it eliminates the need for cell lysis, thereby preserving the native conformation and stoichiometry of protein complexes; it enables the detection of weak, transient, or low-affinity interactions that are typically lost during conventional purification; it allows for the capture of membrane-associated and organelle-resident proteins that are difficult to solubilize; it provides a scalable, single-step purification protocol compatible with high-throughput workflows; and it permits the identification of protein-small molecule interactions without the need for chemical derivatization of the small molecule. The invention thus represents a transformative advance in the field of interactome analysis, providing a powerful, versatile, and physiologically relevant platform for the comprehensive mapping of protein interaction networks.

## DETAILED DESCRIPTION

### Definitions

For the purposes of this disclosure, the term “bait protein” refers to a protein of interest whose interaction partners are to be identified, wherein the bait protein is genetically fused to a retroviral GAG polyprotein to enable its incorporation into virus-like particles during their assembly and budding from the plasma membrane of a mammalian cell. The bait protein may be any endogenous or exogenous protein expressed in mammalian cells, including but not limited to signaling molecules, transcription factors, kinases, phosphatases, scaffolding proteins, ubiquitin ligases, and membrane receptors.

The term “prey protein” refers to a protein that interacts with the bait protein, either directly or as part of a larger multiprotein complex, and which is co-expressed in the same cell as the GAG-bait fusion protein. The prey protein may be expressed as a separate polypeptide chain and may be fused to an epitope tag for detection or purification, but it is not required to be fused to GAG or any other viral structural component.

The term “GAG polyprotein” refers to the structural polyprotein derived from retroviruses, particularly human immunodeficiency virus type 1 (HIV-1), which includes the matrix (MA), capsid (CA), and nucleocapsid (NC) domains, and which is capable of self-assembling into virus-like particles (VLPs) when expressed in mammalian cells. The GAG polyprotein may be derived from any retroviral source, including but not limited to HIV-1, murine leukemia virus (MLV), or simian immunodeficiency virus (SIV), provided that it retains the ability to form VLPs and bud from the plasma membrane. In a preferred embodiment, the GAG polyprotein is the full-length HIV-1 p55 GAG protein.

The term “virus-like particle” or “VLP” refers to a non-infectious, self-assembling, spherical, membrane-bound nanostructure that is formed by the expression of the GAG polyprotein in mammalian cells and that resembles the morphology of a native retroviral particle, but lacks the viral genome and envelope proteins unless co-expressed. VLPs are secreted into the extracellular medium and are approximately 100 to 200 nanometers in diameter. VLPs produced by the method of the invention contain the GAG-bait fusion protein and, when present, the prey protein and associated interacting partners, encapsulated within their lumen or associated with their inner membrane surface.

The term “epitope tag” refers to a short, well-characterized peptide sequence that is genetically fused to a protein to facilitate its detection, quantification, or purification using a specific binding agent such as an antibody or affinity resin. Epitope tags include, but are not limited to, FLAG, MYC, HA, VSV, E-tag, His-tag, and GST. In the context of the invention, the epitope tag is fused to a viral envelope glycoprotein to enable the affinity-based capture of VLPs from cell culture supernatants.

The term “viral envelope glycoprotein” refers to a transmembrane protein derived from a virus that is incorporated into the lipid bilayer of VLPs during their budding process. In a preferred embodiment, the envelope glycoprotein is the vesicular stomatitis virus glycoprotein (VSV-G), which is known to efficiently pseudotype retroviral particles and is readily incorporated into VLPs formed by HIV-1 GAG. Other envelope glycoproteins that may be used include, but are not limited to, the influenza virus hemagglutinin (HA), the Ebola virus glycoprotein (GP), or the Sindbis virus E2 glycoprotein, provided that they are capable of being incorporated into GAG-derived VLPs.

The term “co-expression” refers to the simultaneous expression of two or more polynucleotides in the same cell, either through transfection, transduction, or stable integration, such that the encoded proteins are produced within the same cellular environment and can interact with one another. In the context of the invention, co-expression refers to the expression of the GAG-bait fusion protein, the prey protein, and the tagged envelope glycoprotein in the same mammalian cell.

The term “affinity capture” refers to the selective isolation of a target molecule or structure from a complex mixture using a binding agent that specifically recognizes a tag, epitope, or ligand associated with the target. In the context of the invention, affinity capture refers to the immobilization of VLPs on a solid support via an antibody or binding protein specific to the epitope tag on the envelope glycoprotein, followed by washing and elution to recover the VLPs and their associated protein complexes.

The term “native conditions” refers to the physiological state of a protein complex as it exists within a living cell, including its subcellular localization, post-translational modifications, protein stoichiometry, and interaction dynamics, without the application of exogenous detergents, chaotropic agents, or mechanical disruption. The method of the invention preserves protein complexes under native conditions by avoiding cell lysis and instead relying on the natural process of VLP assembly and budding.

The term “small molecule” refers to a low molecular weight organic compound, typically less than 900 Daltons, that is capable of binding to a protein target and modulating its function. Small molecules include, but are not limited to, drugs, metabolites, natural products, and synthetic compounds. In the context of the invention, small molecules are used as ligands to recruit their protein targets into VLPs via a bivalent linker strategy.

The term “bivalent ligand” refers to a synthetic molecule comprising two distinct chemical moieties linked by a flexible spacer: a first moiety that binds to a small molecule-binding protein (e.g., eDHFR) and a second moiety that binds to a small molecule of interest. The bivalent ligand acts as a molecular bridge to recruit proteins that interact with the small molecule into the VLPs, thereby enabling their identification.

The term “mass spectrometry” refers to an analytical technique used to identify and quantify proteins based on their mass-to-charge ratio. In the context of the invention, mass spectrometry is used to identify the proteins associated with purified VLPs, including both the bait and prey proteins and any additional interacting partners.

The term “western blotting” refers to a laboratory technique used to detect specific proteins in a sample by separating proteins via gel electrophoresis, transferring them to a membrane, and probing with antibodies specific to the target protein. In the context of the invention, western blotting is used to validate specific protein-protein interactions detected by the Virotrap system.

The term “mammalian cell” refers to any cell derived from a mammal, including but not limited to human, mouse, rat, hamster, or monkey cells. Preferred mammalian cells for use in the invention include HEK293T, HeLa, CHO, and U2OS cells, which are readily transfectable and support high levels of VLP production.

The term “plasmid” refers to a circular, double-stranded DNA molecule capable of autonomous replication within a host cell, and which is used to deliver and express the genes encoding the GAG-bait fusion protein, the prey protein, and the tagged envelope glycoprotein. Plasmids used in the invention are typically derived from bacterial vectors such as pUC, pBR322, or pcDNA, and contain a mammalian promoter, a multiple cloning site, a polyadenylation signal, and a selectable marker.

The term “transfection” refers to the process of introducing exogenous nucleic acids into mammalian cells using chemical, physical, or biological methods, including calcium phosphate precipitation, lipofection, electroporation, or polyethyleneimine (PEI)-mediated delivery.

The term “supernatant” refers to the liquid fraction of a cell culture that remains after the removal of cells and cellular debris by centrifugation. In the context of the invention, the supernatant contains the secreted virus-like particles and is the source material for affinity capture.

The term “control experiment” refers to an experimental condition performed in parallel to the main experiment to account for non-specific background interactions. In the context of the invention, control experiments involve the expression of GAG alone, GAG fused to an unrelated protein, or the expression of the tagged envelope glycoprotein without the GAG-bait fusion, to identify proteins that are consistently recovered regardless of the bait identity.

The term “orthogonal assay” refers to a second, independent method used to confirm the validity of an interaction detected by the primary method. In the context of the invention, orthogonal assays include co-immunoprecipitation, mammalian two-hybrid (MAPPIT), mammalian split-ubiquitin (MASPIT), thermal proteome profiling, or surface plasmon resonance.

The term “interactome” refers to the complete set of protein-protein interactions occurring within a cell, tissue, or organism. The method of the invention enables the construction of high-fidelity interactomes by capturing interactions under native conditions.

The term “dynamic interaction” refers to a protein-protein interaction that is regulated by cellular stimuli, such as ligand binding, post-translational modification, or changes in subcellular localization. The method of the invention is capable of capturing dynamic interactions by applying stimuli during VLP production.

The term “background proteins” refers to proteins that are consistently recovered in control experiments and are not specific to the bait protein. These include abundant cellular proteins such as actin, tubulin, heat shock proteins, serum proteins, and host factors that interact with GAG or the envelope glycoprotein. The invention includes a filtering strategy to exclude these background proteins based on their recurrence across multiple control experiments.

The term “high-throughput screening” refers to the automated or parallelized execution of multiple experiments to rapidly test a large number of bait-prey combinations. The invention enables high-throughput screening of protein interaction networks by providing a standardized, scalable platform for VLP production and purification.

The term “physiological relevance” refers to the extent to which an experimental system reflects the true biological context in which a protein interaction occurs in vivo. The method of the invention provides high physiological relevance by preserving the native cellular environment, subcellular localization, and interaction dynamics during complex isolation.

The term “non-specific association” refers to the binding of a protein to the bait, the VLP, or the affinity matrix in the absence of a true biological interaction. The invention minimizes non-specific associations through the use of stringent controls, epitope tagging, and background subtraction.

The term “affinity matrix” refers to a solid support, such as magnetic beads or resin, functionalized with an antibody or binding agent that captures the tagged envelope glycoprotein on the surface of the VLPs. In the invention, the affinity matrix is composed of streptavidin-coated magnetic beads pre-loaded with a biotinylated antibody specific to the epitope tag.

The term “elution” refers to the release of captured VLPs from the affinity matrix using a competitive agent, such as a peptide that competes with the epitope tag for antibody binding, or by altering pH or ionic strength. In the invention, elution is achieved using a FLAG peptide to displace the FLAG-tagged envelope glycoprotein from anti-FLAG antibodies.

The term “proteomic analysis” refers to the large-scale identification and quantification of proteins in a biological sample using mass spectrometry and bioinformatics. In the context of the invention, proteomic analysis is used to identify the full complement of proteins associated with purified VLPs.

The term “filtering strategy” refers to a computational or experimental method used to distinguish true interaction partners from background contaminants. In the invention, the filtering strategy involves comparing protein identifications from experimental samples to those from a panel of control experiments and removing proteins that are recurrently detected in controls.

The term “reproducibility” refers to the ability to obtain consistent results across independent experiments. The method of the invention demonstrates high reproducibility due to the standardized, lysis-free purification protocol and the use of internal controls.

The term “scalability” refers to the ability to increase the volume or number of experiments without loss of efficiency or specificity. The invention is scalable from microscale (well plate) to industrial-scale (bioreactor) production of VLPs.

The term “complementary to AP–MS” refers to the observation that the method of the invention detects a subset of protein interactions that are not detected by conventional affinity purification-mass spectrometry, and vice versa, thereby providing a more comprehensive view of the interactome when used in conjunction with AP–MS.

The term “subcellular compartment” refers to a distinct, membrane-bound or non-membrane-bound region within a cell, such as the cytosol, plasma membrane, endoplasmic reticulum, Golgi apparatus, nucleus, mitochondria, or lysosome. The method of the invention is particularly suited for the study of cytosolic and plasma membrane-associated complexes, but may be extended to other compartments through the use of alternative GAG variants or targeting sequences.

The term “post-translational modification” refers to a chemical modification of a protein after its translation, including phosphorylation, acetylation, ubiquitination, sumoylation, glycosylation, or lipidation. The method of the invention preserves post-translational modifications because it avoids cell lysis and denaturing conditions.

The term “low-affinity interaction” refers to a protein-protein interaction with a dissociation constant (Kd) greater than 1 micromolar. The method of the invention is capable of detecting such interactions due to the avidity effect created by the multivalent presentation of bait proteins on the surface of the VLPs.

The term “avidity effect” refers to the enhanced binding strength resulting from multiple simultaneous interactions between a multivalent ligand and its target. In the invention, the clustering of GAG-bait fusion proteins on the surface of the VLPs increases the effective affinity for low-affinity prey proteins, enabling their capture despite weak individual binding events.

The term “dynamic range” refers to the span of protein abundances that can be detected in a single experiment. The method of the invention provides a broad dynamic range due to the enrichment of low-abundance interaction partners within the VLPs and the sensitivity of mass spectrometry.

The term “false positive” refers to a protein identified as an interaction partner that does not genuinely interact with the bait under physiological conditions. The invention minimizes false positives through the use of stringent controls and background subtraction.

The term “false negative” refers to a genuine interaction that is not detected due to technical limitations. The invention reduces false negatives by preserving labile complexes that are lost in lysis-based methods.

The term “endogenous expression” refers to the natural expression of a protein by the cell’s own genome, as opposed to overexpression from an exogenous plasmid. The method of the invention can be adapted to study endogenous complexes by using CRISPR/Cas9-mediated tagging of endogenous genes with GAG.

The term “genetic fusion” refers to the joining of two or more coding sequences in-frame to produce a single polypeptide chain. In the invention, the bait protein is genetically fused to the N-terminus of the GAG polyprotein.

The term “epitope-tagged envelope glycoprotein” refers to a viral envelope glycoprotein that has been genetically modified to contain an epitope tag at its N-terminus, C-terminus, or internal loop, without impairing its ability to be incorporated into VLPs or mediate membrane fusion.

The term “polyethylene glycol linker” refers to a flexible, hydrophilic polymer chain used to connect two functional moieties in a bivalent ligand. In the invention, the linker is used to connect the small molecule-binding ligand (e.g., methotrexate) to the small molecule of interest (e.g., simvastatin).

The term “orthogonal validation” refers to the confirmation of a finding using a method that is fundamentally different in principle from the primary detection method. In the invention, orthogonal validation is performed using MAPPIT, MASPIT, or co-immunoprecipitation.

The term “biological repeat” refers to an independent experiment performed using a separate batch of cells, transfection, and purification. In the invention, biological repeats are used to ensure reproducibility and statistical confidence in protein identifications.

The term “peptide spectral match” refers to a mass spectrometry-derived identification of a peptide sequence derived from a protein. In the invention, proteins are considered confidently identified if they are represented by at least two unique peptide spectral matches across two biological repeats.

The term “false discovery rate” refers to the estimated proportion of false positive identifications among all reported protein identifications in a mass spectrometry experiment. In the invention, the false discovery rate is controlled to less than 1% using reversed database searches and stringent peptide filtering.

The term “complex stability” refers to the resistance of a protein complex to dissociation under physiological or experimental conditions. The method of the invention enhances complex stability by physically encapsulating the complex within the VLP, thereby protecting it from proteolytic degradation and dilution.

The term “secreted VLPs” refers to virus-like particles that are released from the cell into the extracellular medium, as opposed to intracellularly retained particles. Secretion is a key feature of the invention, as it enables the non-invasive harvesting of VLPs without cell lysis.

The term “non-invasive harvesting” refers to the collection of secreted VLPs from the cell culture supernatant without disrupting the plasma membrane or lysing the cells. This is a defining feature of the invention that distinguishes it from all prior AP–MS methods.

The term “molecular crowding” refers to the high concentration of macromolecules within the cytosol that influences protein folding, binding kinetics, and complex assembly. The method of the invention preserves molecular crowding by avoiding cell lysis and dilution.

The term “native stoichiometry” refers to the relative abundance of each component within a protein complex as it exists in vivo. The method of the invention preserves native stoichiometry by capturing complexes in their assembled state without perturbing their composition.

The term “complex architecture” refers to the three-dimensional arrangement of proteins within a multiprotein complex. The method of the invention preserves complex architecture by maintaining the structural integrity of the complex during purification.

The term “interaction network” refers to a set of interconnected protein-protein interactions that form a functional module within a cell. The method of the invention enables the reconstruction of interaction networks under native conditions.

The term “drug target identification” refers to the process of identifying the protein(s) to which a small molecule drug binds to exert its biological effect. The invention provides a novel platform for drug target identification by capturing drug-bound proteins within VLPs.

The term “off-target effect” refers to the unintended binding of a drug to proteins other than its intended target. The invention enables the detection of off-target effects by identifying all proteins that are recruited into VLPs in the presence of a small molecule.

The term “ligand-dependent interaction” refers to a protein-protein interaction that is induced or enhanced by the binding of a small molecule ligand. The invention can detect ligand-dependent interactions by adding the ligand during VLP production.

The term “signal transduction complex” refers to a multiprotein assembly that relays a signal from a cell surface receptor to intracellular effectors. The invention is particularly suited for studying signal transduction complexes, which are often transient and labile.

The term “ubiquitin ligase complex” refers to a multiprotein assembly that catalyzes the transfer of ubiquitin to substrate proteins. The invention can capture ubiquitin ligase complexes and their substrates under native conditions.

The term “phosphorylation-dependent interaction” refers to a protein-protein interaction that requires the phosphorylation of one or more components. The invention preserves phosphorylation-dependent interactions because it avoids phosphatase exposure during purification.

The term “membrane-associated complex” refers to a protein complex that is tethered to or embedded within a cellular membrane. The invention can capture membrane-associated complexes because the VLPs bud from the plasma membrane and retain associated transmembrane proteins.

The term “organelle-resident protein” refers to a protein that is localized to a specific intracellular organelle, such as the endoplasmic reticulum, mitochondria, or lysosome. The invention can capture organelle-resident proteins if they are recruited to the plasma membrane or if they associate with transmembrane baits.

The term “viral pseudotyping” refers to the incorporation of an envelope glycoprotein from one virus into the particle of another virus. In the invention, VSV-G is pseudotyped onto HIV-1 GAG-derived VLPs to enable efficient capture.

The term “antibody-based recovery” refers to the use of antibodies immobilized on a solid support to selectively bind and isolate a target molecule. In the invention, antibody-based recovery is used to isolate VLPs via the epitope tag on the envelope glycoprotein.

The term “magnetic beads” refers to microscopic particles composed of a magnetic core and a functionalized surface that can be manipulated using a magnet. In the invention, magnetic beads are used to facilitate the rapid and efficient capture of VLPs.

The term “FLAG peptide” refers to a synthetic peptide sequence (DYKDDDDK) that competes with FLAG-tagged proteins for binding to anti-FLAG antibodies. In the invention, FLAG peptide is used to elute VLPs from anti-FLAG beads.

The term “trypsin digestion” refers to the enzymatic cleavage of proteins into peptides using the protease trypsin. In the invention, trypsin digestion is used to prepare VLP-associated proteins for mass spectrometry analysis.

The term “nano-LC” refers to nanoscale liquid chromatography, a high-sensitivity separation technique used to resolve peptides prior to mass spectrometry analysis. In the invention, nano-LC is used to separate tryptic peptides for high-resolution mass spectrometry.

The term “Q Exactive instrument” refers to a high-resolution, accurate-mass mass spectrometer manufactured by Thermo Scientific, used in the invention for peptide identification.

The term “MASCOT algorithm” refers to a search engine used to match mass spectrometry spectra to protein sequences in a database. In the invention, MASCOT is used to identify proteins from VLP-associated peptides.

The term “SWISSPROT” refers to a manually curated protein sequence database maintained by the Swiss Institute of Bioinformatics. In the invention, SWISSPROT is used as the reference database for protein identification.

The term “ProteomeXchange” refers to a consortium that provides a unified platform for the deposition and sharing of mass spectrometry proteomics data. In the invention, raw data are deposited in ProteomeXchange via the PRIDE repository.

The term “supplementary data” refers to additional datasets provided alongside a publication or patent application, including protein lists, spectral counts, and raw files. In the invention, supplementary data are used to document control experiments and identified interactors.

The term “single peptide identification” refers to the detection of a protein based on the identification of only one unique peptide. In the invention, single peptide identifications are included in candidate lists but are filtered out unless they are recurrent across biological repeats.

The term “recurrent protein identification” refers to the repeated detection of a protein across multiple independent experiments or biological replicates. In the invention, recurrent protein identifications are used to distinguish true interactors from stochastic noise.

The term “background subtraction” refers to the computational removal of proteins identified in control experiments from the list of proteins identified in experimental samples. In the invention, background subtraction is performed using a blacklist of proteins identified in 19 or more control samples.

The term “blacklist strategy” refers to the exclusion of proteins that are frequently detected in control experiments. In the invention, the blacklist strategy is used to eliminate non-specific background proteins.

The term “bait-centered filtering” refers to a computational method that uses the bait protein as a reference to identify statistically significant interactors. In the invention, alternative filtering methods such as SFINX may be used in conjunction with the blacklist strategy.

The term “protein spectral count” refers to the number of mass spectrometry spectra assigned to a given protein. In the invention, protein spectral counts are used to estimate relative protein abundance.

The term “peptide spectral count” refers to the number of mass spectrometry spectra assigned to a given peptide sequence. In the invention, peptide spectral counts are used to validate protein identifications.

The term “positive reference set” refers to a curated collection of known protein-protein interactions used to validate a new interaction detection method. In the invention, the human positive reference set (hsPRS-v1) is used to benchmark Virotrap performance.

The term “random reference set” refers to a collection of randomly selected protein pairs that are not expected to interact, used to estimate the false positive rate. In the invention, the human random reference set (hsRRS-v1) is used to determine the specificity of the method.

The term “detection threshold” refers to the minimum signal intensity required to classify a protein band as positive in a western blot. In the invention, the detection threshold is set based on the signal observed in the random reference set.

The term “cross-comparison” refers to the comparison of results across multiple experimental runs or gels to ensure consistency. In the invention, cross-comparison is performed using a pooled positive control loaded on every gel.

The term “fluorescence signal” refers to the light emitted by a fluorophore upon excitation, used in the invention to quantify protein bands on an Odyssey imager.

The term “ODYSSEY Imager” refers to a dual-channel infrared imaging system used in the invention to detect fluorescently labeled proteins on western blots.

The term “RIPA buffer” refers to a lysis buffer containing radioimmunoprecipitation assay components, used in the invention for the preparation of cell lysates for western blotting.

The term “transfection efficiency” refers to the percentage of cells that successfully take up and express exogenous DNA. In the invention, transfection efficiency is optimized to ensure robust VLP production.

The term “mycoplasma contamination” refers to the presence of mycoplasma bacteria in cell cultures, which can interfere with protein expression and purification. In the invention, cell lines are regularly tested and maintained free of mycoplasma.

The term “low passage number” refers to the number of times a cell line has been subcultured since its original isolation. In the invention, cells are used at low passage numbers (<10) to maintain physiological relevance.

The term “high-glucose DMEM” refers to Dulbecco’s Modified Eagle Medium supplemented with 4.5 g/L glucose, used in the invention as the growth medium for HEK293T cells.

The term “FCS” refers to fetal calf serum, used in the invention as a source of growth factors and nutrients.

The term “calcium phosphate transfection” refers to a chemical method for delivering DNA into mammalian cells using a precipitate of calcium phosphate and DNA. In the invention, calcium phosphate transfection is used for standard VLP production.

The term “polyethyleneimine (PEI)” refers to a cationic polymer used as an alternative transfection reagent for large-scale VLP production.

The term “ultracentrifugation” refers to centrifugation at speeds exceeding 100,000 × g, used in early versions of the method to pellet VLPs. In the invention, ultracentrifugation is replaced by affinity capture for scalability.

The term “swinging bucket rotor” refers to a type of centrifuge rotor used in ultracentrifugation. In the invention, a Ti41 swinging bucket rotor was used for VLP pelleting.

The term “0.45 μm filters” refer to membrane filters used to remove cellular debris from supernatants prior to VLP capture.

The term “Dynabeads MyOne Streptavidin T1” refers to magnetic beads coated with streptavidin, used in the invention to capture biotinylated antibodies bound to VLPs.

The term “FLAG BioM2-Biotin” refers to a biotinylated monoclonal antibody specific to the FLAG epitope, used in the invention to capture FLAG-tagged VSV-G on VLPs.

The term “washing buffer” refers to a solution used to remove non-specifically bound material during affinity purification. In the invention, washing buffer contains HEPES, NaCl, and Tris-HCl.

The term “FLAG peptide elution” refers to the release of VLPs from anti-FLAG beads using a synthetic FLAG peptide. In the invention, FLAG peptide elution is performed at 37°C for 30 minutes.

The term “SDS” refers to sodium dodecyl sulfate, a detergent used in the invention to lyse VLPs prior to mass spectrometry.

The term “HiPPR Detergent Removal Spin Columns” refer to affinity columns used to remove SDS from peptide samples prior to trypsin digestion.

The term “sequence-grade trypsin” refers to a highly purified form of trypsin used for proteomic sample preparation.

The term “acidification” refers to the addition of acid to lower the pH of a sample, used in the invention to terminate trypsin activity.

The term “trifluoroacetic acid” refers to a strong acid used in the invention to acidify peptide samples prior to nano-LC.

The term “nano-LC” refers to nanoscale liquid chromatography, used in the invention to separate peptides prior to mass spectrometry.

The term “MS/MS mode” refers to tandem mass spectrometry, in which precursor ions are fragmented to generate product ion spectra for peptide identification.

The term “confidence level” refers to the statistical certainty assigned to a protein identification. In the invention, identifications are accepted at 99% confidence.

The term “reversed database” refers to a database containing the reverse sequences of all proteins in the target database, used to estimate false discovery rates.

The term “peptide to spectrum match (PSM)” refers to the assignment of a mass spectrum to a specific peptide sequence. In the invention, only the highest-scoring PSM for each spectrum is retained.

The term “PRIDE” refers to the ProteomeXchange partner repository for mass spectrometry data. In the invention, data are deposited in PRIDE under accession number PXD000685.

The term “ORFEOME 5.1” refers to a collection of human open reading frame clones used in the invention to generate prey constructs.

The term “Gateway cloning” refers to a recombination-based cloning system used in the invention to transfer bait and prey genes into expression vectors.

The term “SRalpha promoter” refers to a strong constitutive promoter derived from the simian virus 40 early region, used in the invention to drive high-level expression of GAG-bait fusions.

The term “pMET7 vector” refers to a mammalian expression vector used in the invention to express prey proteins with an E-tag.

The term “pMD2.g” refers to a plasmid encoding VSV-G, used in the invention for pseudotyping.

The term “pcDNA3-FLAG-VSV-G” refers to a plasmid encoding FLAG-tagged VSV-G, used in the invention for VLP capture.

The term “pSV25S-hTNFR 55” refers to a plasmid encoding the human tumor necrosis factor receptor 55, used in the invention to study dynamic A20 complexes.

The term “TNFα” refers to tumor necrosis factor alpha, used in the invention to activate the NF-κB pathway.

The term “300 IU ml−1” refers to the concentration of TNFα used to stimulate cells during VLP production.

The term “1 μM” refers to the concentration of bivalent ligands used in small molecule experiments.

The term “DMSO” refers to dimethyl sulfoxide, used as a solvent control in small molecule experiments.

The term “leptin” refers to a hormone used in the invention as a positive control in MASPIT assays.

The term “luciferase assay” refers to a bioluminescent reporter assay used in the invention to validate protein-small molecule interactions.

The term “Promega Luciferase Assay System” refers to a commercial kit used in the invention to measure luciferase activity.

The term “co-immunoprecipitation” refers to the isolation of a protein complex using an antibody against one component, followed by detection of co-precipitating proteins. In the invention, co-immunoprecipitation is used as an orthogonal validation method.

The term “mammalian two-hybrid” refers to a system in which two proteins are fused to a DNA-binding domain and an activation domain, and their interaction reconstitutes a transcription factor. In the invention, MAPPIT is used as an orthogonal validation method.

The term “mammalian split-ubiquitin” refers to a system in which two proteins are fused to fragments of ubiquitin, and their interaction reconstitutes ubiquitin, leading to reporter gene activation. In the invention, MASPIT is used as an orthogonal validation method.

The term “thermal proteome profiling” refers to a method that detects protein-ligand interactions by measuring changes in protein thermal stability. In the invention, thermal proteome profiling is mentioned as a complementary method for small molecule target identification.

The term “surface plasmon resonance” refers to a biophysical technique that measures real-time binding kinetics between a ligand and its target. In the invention, surface plasmon resonance is mentioned as a potential orthogonal validation method.

The term “CRISPR/Cas9-mediated tagging” refers to the use of genome editing to insert a tag into an endogenous gene. In the invention, this approach is proposed as a future extension to study endogenous complexes.

The term “antiviral factors” refers to host proteins that restrict viral replication, such as tetherin. In the invention, antiviral factors may interfere with VLP budding in certain cell types.

The term “genome engineering” refers to the targeted modification of a cell’s genome, used in the invention to knock out antiviral factors and improve VLP production.

The term “artificial GAG variants” refers to engineered versions of the GAG polyprotein with mutations that enhance VLP production, reduce background, or alter tropism. In the invention, such variants are proposed as future improvements.

The term “alternative viral matrix variants” refers to matrix proteins from other viruses that may be used in place of HIV-1 GAG to target different cellular compartments. In the invention, such variants are proposed for future development.

The term “endoplasmic reticulum” refers to the organelle responsible for protein folding and lipid synthesis. In the invention, ER-resident proteins such as HMGCR are captured via transmembrane baits.

The term “plasma membrane” refers to the outer membrane of the cell. In the invention, VLPs bud from the plasma membrane, enabling the capture of membrane-associated complexes.

The term “cytosolic complex” refers to a protein complex that resides in the cytoplasm. In the invention, cytosolic complexes are the primary targets, but membrane-associated complexes are also captured.

The term “mitochondrial complex” refers to a protein complex located within mitochondria. In the invention, mitochondrial complexes are not efficiently captured due to the lack of mitochondrial targeting in the current system.

The term “peroxisomal complex” refers to a protein complex located within peroxisomes. In the invention, peroxisomal complexes are not efficiently captured.

The term “nuclear complex” refers to a protein complex located within the nucleus. In the invention, nuclear complexes are not efficiently captured unless the bait is nuclear-localized and recruits partners to the plasma membrane.

The term “bait-centred tool SFINX” refers to a computational method that uses the bait protein as a reference to identify statistically significant interactors. In the invention, SFINX is mentioned as an alternative to the blacklist strategy.

The term “P value” refers to a statistical measure of significance. In the invention, a P value of 1.79E−43 is reported for the association between AURKA and TTK.

The term “orthogonal confirmation” refers to the validation of an interaction using a method that is mechanistically distinct from the primary method. In the invention, orthogonal confirmation is performed using MASPIT, co-IP, or thermal profiling.

The term “drug repurposing” refers to the identification of new therapeutic uses for existing drugs. In the invention, the method enables drug repurposing by identifying novel protein targets of known drugs.

The term “off-target profiling” refers to the systematic identification of unintended protein targets of a drug. In the invention, off-target profiling is enabled by the capture of all proteins recruited by a bivalent ligand.

The term “mechanism of action” refers to the molecular pathway by which a drug exerts its biological effect. In the invention, the method enables elucidation of the mechanism of action of orphan drugs.

The term “target deconvolution” refers to the process of identifying the protein targets of a bioactive compound. In the invention, target deconvolution is achieved without chemical modification of the small molecule.

The term “chemical biology” refers to the application of chemical tools to study biological systems. In the invention, the method represents a major advance in chemical biology for the study of protein-small molecule interactions.

The term “systems biology” refers to the holistic study of biological systems through the integration of large-scale data. In the invention, the method enables the construction of comprehensive, native-state interactomes.

The term “interactome mapping” refers to the systematic identification of all protein-protein interactions in a cell. In the invention, interactome mapping is achieved under native conditions without lysis.

The term “high-confidence interactors” refers to proteins identified with multiple peptides across biological repeats and absent from control samples. In the invention, high-confidence interactors are the primary output of the method.

The term “low-abundance proteins” refers to proteins expressed at low copy numbers per cell. In the invention, low-abundance proteins are efficiently captured due to the enrichment effect of VLP packaging.

The term “signal amplification” refers to the enhancement of a detection signal. In the invention, signal amplification is achieved through the concentration of interaction partners within VLPs.

The term “complex enrichment” refers to the selective concentration of a protein complex from a complex mixture. In the invention, complex enrichment is achieved through VLP encapsulation.

The term “native state preservation” refers to the maintenance of a protein’s natural conformation, modification, and interaction state. In the invention, native state preservation is achieved by avoiding lysis.

The term “biological fidelity” refers to the accuracy with which an experimental system reflects in vivo biology. In the invention, biological fidelity is high due to the preservation of native conditions.

The term “technical robustness” refers to the reliability and reproducibility of a method under varying conditions. In the invention, technical robustness is demonstrated by consistent results across multiple bait proteins and experimental conditions.

The term “platform technology” refers to a generalizable method that can be applied to multiple targets or applications. In the invention, Virotrap is a platform technology for protein interaction discovery.

The term “modular design” refers to a system composed of interchangeable components. In the invention, the bait, prey, and envelope components are modular and can be swapped independently.

The term “standardized protocol” refers to a fixed set of procedures that can be uniformly applied across experiments. In the invention, the single-step capture protocol is standardized for high-throughput use.

The term “automation compatibility” refers to the ability to integrate a method into robotic or high-throughput systems. In the invention, the affinity capture protocol is compatible with automated liquid handling systems.

The term “cost-effectiveness” refers to the balance between performance and resource expenditure. In the invention, the method is cost-effective due to the use of standard reagents and avoidance of expensive equipment.

The term “time efficiency” refers to the reduction in experimental duration. In the invention, the single-step capture reduces purification time from hours to minutes.

The term “versatility” refers to the ability to adapt a method to diverse applications. In the invention, versatility is demonstrated by the capture of binary interactions, multiprotein complexes, and small molecule targets.

The term “novelty” refers to the uniqueness of the invention compared to prior art. In the invention, novelty lies in the combination of GAG-mediated VLP encapsulation with epitope-tagged envelope capture.

The term “inventive step” refers to the non-obviousness of the invention to a person skilled in the art. In the invention, the inventive step lies in the realization that VLPs can serve as physical containers for native complexes and that their capture can be streamlined using envelope tagging.

The term “industrial applicability” refers to the potential for commercial development and scaling. In the invention, industrial applicability is high due to scalability, standardization, and compatibility with existing proteomics workflows.

The term “patentable subject matter” refers to the legal eligibility of the invention for patent protection. In the invention, the method, system, and kit are patentable subject matter under U.S. and international patent law.

The term “claim scope” refers to the breadth of protection afforded by the claims of a patent. In this disclosure, the claim scope encompasses all embodiments described herein, including variations in GAG source, envelope glycoprotein, tag, cell type, and application.

The term “enabling disclosure” refers to the sufficiency of the description to allow a person skilled in the art to practice the invention without undue experimentation. In this disclosure, the enabling disclosure is complete, with detailed protocols, plasmid constructs, and validation data provided.

The term “best mode” refers to the preferred embodiment of the invention known to the inventor at the time of filing. In this disclosure, the best mode is the use of HIV-1 GAG, VSV-G-FLAG, HEK293T cells, and FLAG peptide elution for the capture of protein complexes.

The term “prior art” refers to all knowledge and inventions publicly available before the filing date of this application. In this disclosure, prior art includes AP–MS, BioID, APEX2, yeast two-hybrid, mammalian two-hybrid, and pull-down assays.

The term “non-obviousness” refers to the requirement that an invention not be an obvious modification of prior art. In this invention, the combination of GAG-mediated encapsulation with envelope-tagged capture is non-obvious, as no prior art teaches or suggests this configuration.

The term “utility” refers to the usefulness of the invention. In this invention, utility is demonstrated by the detection of known and novel interactions, including weak, dynamic, and small molecule-mediated interactions.

The term “comprehensive interactome” refers to a complete map of protein-protein interactions in a cell under native conditions. In this invention, the method enables the generation of comprehensive interactomes that are more complete than those generated by lysis-based methods.

The term “functional annotation” refers to the assignment of biological roles to identified proteins. In this invention, functional annotation is performed using Gene Ontology and pathway analysis.

The term “pathway enrichment” refers to the identification of biological pathways overrepresented in a protein list. In this invention, pathway enrichment is used to validate the biological relevance of captured interactors.

The term “network topology” refers to the structural organization of an interaction network, including hubs, bottlenecks, and modules. In this invention, network topology is analyzed to identify key regulatory proteins.

The term “interactome dynamics” refers to changes in protein interactions under different cellular conditions. In this invention, interactome dynamics are studied by applying stimuli such as TNFα during VLP production.

The term “drug discovery” refers to the process of identifying and developing new therapeutic agents. In this invention, the method accelerates drug discovery by enabling rapid target identification and off-target profiling.

The term “precision medicine” refers to the tailoring of medical treatment to individual patients based on molecular profiles. In this invention, the method supports precision medicine by enabling the mapping of patient-specific interactomes.

The term “biomarker discovery” refers to the identification of proteins or protein complexes that indicate a disease state. In this invention, biomarker discovery is enabled by the capture of disease-relevant complexes under native conditions.

The term “systems pharmacology” refers to the study of drug effects on biological networks. In this invention, systems pharmacology is advanced by the ability to map drug-target networks in their native context.

The term “protein complex stability assay” refers to a method for measuring the resistance of a complex to dissociation. In this invention, Virotrap serves as a protein complex stability assay by preserving complexes that dissociate in lysis-based methods.

The term “interaction specificity” refers to the degree to which a detected interaction is unique to a particular bait-prey pair. In this invention, interaction specificity is enhanced by the use of stringent controls and background subtraction.

The term “interaction sensitivity” refers to the ability to detect low-abundance or low-affinity interactions. In this invention, interaction sensitivity is high due to the concentration effect of VLP encapsulation.

The term “false positive rate” refers to the proportion of detected interactions that are not genuine. In this invention, the false positive rate is less than 5% as demonstrated by the random reference set.

The term “false negative rate” refers to the proportion of genuine interactions that are missed. In this invention, the false negative rate is reduced compared to AP–MS due to the preservation of labile complexes.

The term “reproducibility across platforms” refers to the consistency of results when the same interaction is tested by different methods. In this invention, reproducibility across platforms is demonstrated by the overlap with MAPPIT and co-IP data.

The term “cross-platform validation” refers to the confirmation of an interaction using multiple independent methods. In this invention, cross-platform validation is routinely performed.

The term “data integration” refers to the combination of data from multiple sources to generate a unified view. In this invention, data from Virotrap, AP–MS, and BioID are integrated to generate comprehensive interactomes.

The term “reference interactome” refers to a curated set of known interactions used as a benchmark. In this invention, the human positive reference set (hsPRS-v1) serves as a reference interactome.

The term “negative control” refers to an experimental condition designed to produce no signal. In this invention, negative controls include GAG alone, unrelated baits, and empty vector.

The term “positive control” refers to an experimental condition designed to produce a known signal. In this invention, the HRAS-RAF1 and GRAP2-LCP2 interactions serve as positive controls.

The term “internal standard” refers to a molecule used to normalize measurements. In this invention, bait protein levels are used to normalize prey protein signals.

The term “normalization” refers to the adjustment of data to account for variations in sample loading or detection. In this invention, prey signals are normalized to bait levels to account for differences in VLP production.

The term “signal-to-noise ratio” refers to the ratio of true signal to background noise. In this invention, the signal-to-noise ratio is high due to the enrichment of true interactors and the removal of background proteins.

The term “dynamic range of detection” refers to the span of concentrations over which a method can accurately measure a target. In this invention, the dynamic range of detection spans several orders of magnitude due to the sensitivity of mass spectrometry and the enrichment effect of VLPs.

The term “sample throughput” refers to the number of samples that can be processed in a given time. In this invention, sample throughput is high due to the scalability of the single-step capture protocol.

The term “multiplexing capability” refers to the ability to detect multiple targets simultaneously. In this invention, multiplexing capability is inherent in mass spectrometry-based identification of all VLP-associated proteins.

The term “quantitative capability” refers to the ability to measure relative or absolute protein abundance. In this invention, quantitative capability is achieved through spectral counting and normalization.

The term “qualitative capability” refers to the ability to detect the presence or absence of a protein. In this invention, qualitative capability is achieved through western blotting and peptide identification.

The term “high-resolution detection” refers to the ability to distinguish closely related proteins or isoforms. In this invention, high-resolution detection is achieved through mass spectrometry and peptide sequencing.

The term “low-background detection” refers to the ability to detect targets with minimal non-specific binding. In this invention, low-background detection is achieved through the blacklist filtering strategy.

The term “specificity index” refers to a metric that quantifies the specificity of an interaction. In this invention, the specificity index is calculated as the ratio of prey signal in experimental versus control samples.

The term “interaction confidence score” refers to a numerical value assigned to an interaction based on its reproducibility, spectral count, and absence from controls. In this invention, interaction confidence scores are used to rank candidate interactors.

The term “candidate interactor” refers to a protein identified as a potential interaction partner. In this invention, candidate interactors are filtered to yield high-confidence interactors.

The term “high-confidence interactor” refers to a protein identified with multiple peptides across biological repeats and absent from controls. In this invention, high-confidence interactors are the primary output of the method.

The term “novel interaction” refers to a protein-protein interaction not previously reported in the literature. In this invention, novel interactions are identified through comparison with existing interactome databases.

The term “known interaction” refers to a protein-protein interaction previously documented in the literature. In this invention, known interactions are used to validate the method.

The term “validation” refers to the confirmation of an interaction using an independent method. In this invention, validation is performed using orthogonal assays.

The term “confirmation” refers to the reproducible detection of an interaction. In this invention, confirmation is achieved through biological repeats and orthogonal assays.

The term “reproducibility” refers to the ability to obtain the same results in repeated experiments. In this invention, reproducibility is demonstrated across multiple baits and biological replicates.

The term “scalability” refers to the ability to increase the scale of the experiment without loss of performance. In this invention, scalability is demonstrated by the use of 107 cells per experiment and the compatibility with bioreactor production.

The term “portability” refers to the ability to transfer the method to other laboratories. In this invention, portability is ensured by the use of standard plasmids, reagents, and protocols.

The term “standardization” refers to the establishment of uniform procedures. In this invention, standardization is achieved through the use of defined plasmid constructs, transfection ratios, and capture protocols.

The term “automation” refers to the use of machines to perform repetitive tasks. In this invention, automation is enabled by the use of magnetic beads and standardized elution conditions.

The term “cost reduction” refers to the decrease in expenses associated with the method. In this invention, cost reduction is achieved by eliminating ultracentrifugation and reducing reagent use.

The term “time reduction” refers to the decrease in experimental duration. In this invention, time reduction is achieved by replacing ultracentrifugation with a 2-hour affinity capture step.

The term “user-friendliness” refers to the ease with which a method can be adopted by non-specialists. In this invention, user-friendliness is enhanced by the simplicity of the single-step capture protocol.

The term “robustness” refers to the ability of a method to perform reliably under variable conditions. In this invention, robustness is demonstrated by consistent results across different cell lines, transfection methods, and bait proteins.

The term “flexibility” refers to the ability to adapt the method to different experimental needs. In this invention, flexibility is demonstrated by the ability to capture binary interactions, multiprotein complexes, and small molecule targets.

The term “innovation” refers to the introduction of a new concept or technique. In this invention, innovation lies in the use of VLPs as native-state interaction traps.

The term “paradigm shift” refers to a fundamental change in the way a field operates. In this invention, the method represents a paradigm shift in interactome analysis by replacing lysis with encapsulation.

The term “translational potential” refers to the ability of a method to be applied to clinical or industrial settings. In this invention, translational potential is high due to its applicability to drug discovery and biomarker identification.

The term “commercial viability” refers to the potential for economic return on investment. In this invention, commercial viability is high due to the platform nature of the technology and its applicability to pharmaceutical and biotechnology industries.

The term “intellectual property” refers to creations of the mind that are legally protected. In this invention, the method, system, and kit constitute valuable intellectual property.

The term “patent landscape” refers to the collection of existing patents in a field. In this invention, the patent landscape for lysis-independent interactome analysis is sparse, and this invention occupies a novel and unclaimed space.

The term “freedom to operate” refers to the ability to commercialize an invention without infringing on existing patents. In this invention, freedom to operate is established by the novelty of the GAG-envelope tagging combination.

The term “regulatory compliance” refers to adherence to legal and ethical standards. In this invention, regulatory compliance is ensured by the use of non-infectious VLPs and standard cell culture practices.

The term “ethical use” refers to the responsible application of a technology. In this invention, ethical use is ensured by the avoidance of human tissue and the use of immortalized cell lines.

The term “sustainability” refers to the environmental impact of a method. In this invention, sustainability is enhanced by the reduction in reagent use and energy consumption compared to ultracentrifugation.

The term “global applicability” refers to the ability to use the method worldwide. In this invention, global applicability is ensured by the use of universally available reagents and protocols.

The term “open science” refers to the sharing of data and methods for public benefit. In this invention, open science is practiced through the deposition of data in ProteomeXchange.

The term “data sharing” refers to the public release of experimental data. In this invention, data sharing is implemented through PRIDE deposition.

The term “reagent sharing” refers to the distribution of plasmids and constructs to other researchers. In this invention, reagent sharing is facilitated by depositing plasmids in Addgene.

The term “collaborative potential” refers to the ability of a method to foster scientific collaboration. In this invention, collaborative potential is high due to the platform nature of the technology and the availability of standardized reagents.

The term “educational value” refers to the utility of a method for teaching and training. In this invention, educational value is high due to the clear mechanistic basis and simple protocol.

The term “technical documentation” refers to written instructions for performing a method. In this invention, technical documentation is provided in the form of detailed protocols and plasmid maps.

The term “reagent kit” refers to a packaged set of materials for performing a method. In this invention, a reagent kit is provided comprising plasmids, antibodies, beads, and instructions.

The term “service offering” refers to the provision of a method as a commercial service. In this invention, service offerings include Virotrap screening, interactome mapping, and drug target identification.

The term “diagnostic application” refers to the use of a method for disease detection. In this invention, diagnostic applications include the identification of disease-specific protein complexes.

The term “therapeutic application” refers to the use of a method for drug development. In this invention, therapeutic applications include target identification, off-target profiling, and mechanism of action studies.

The term “biopharmaceutical development” refers to the development of drugs derived from biological sources. In this invention, biopharmaceutical development is supported by the identification of novel drug targets.

The term “target validation” refers to the confirmation that a protein is a viable drug target. In this invention, target validation is achieved by demonstrating the interaction of a drug with its target in a native context.

The term “lead optimization” refers to the improvement of a drug candidate’s properties. In this invention, lead optimization is informed by the identification of off-target interactions.

The term “pharmacokinetic profiling” refers to the study of how a drug is absorbed, distributed, metabolized, and excreted. In this invention, pharmacokinetic profiling is informed by the identification of drug-binding proteins.

The term “toxicology screening” refers to the assessment of drug toxicity. In this invention, toxicology screening is enhanced by the identification of off-target interactions that may cause adverse effects.

The term “personalized therapy” refers to the tailoring of treatment to individual patients. In this invention, personalized therapy is enabled by the mapping of patient-derived interactomes.

The term “biomarker panel” refers to a set of biomarkers used for diagnosis or prognosis. In this invention, biomarker panels are generated from Virotrap-identified complexes.

The term “clinical translation” refers to the application of a research finding to clinical practice. In this invention, clinical translation is supported by the identification of disease-relevant interactions.

The term “precision diagnostics” refers to the use of molecular markers to guide diagnosis. In this invention, precision diagnostics are enabled by the detection of disease-specific complexes.

The term “drug resistance profiling” refers to the identification of mechanisms by which cells become resistant to drugs. In this invention, drug resistance profiling is enabled by the detection of altered interaction networks in resistant cells.

The term “cancer interactome” refers to the network of protein interactions altered in cancer. In this invention, the cancer interactome is mapped using Virotrap.

The term “neurodegenerative interactome” refers to the network of protein interactions altered in neurodegenerative diseases. In this invention, the neurodegenerative interactome is mapped using Virotrap.

The term “infectious disease interactome” refers to the network of protein interactions altered during infection. In this invention, the infectious disease interactome is mapped using Virotrap.

The term “immune interactome” refers to the network of protein interactions involved in immune signaling. In this invention, the immune interactome is mapped using Virotrap.

The term “signaling network” refers to a set of interconnected proteins that transmit signals within a cell. In this invention, signaling networks are mapped under native conditions.

The term “transcriptional network” refers to a set of proteins that regulate gene expression. In this invention, transcriptional networks are mapped using Virotrap.

The term “metabolic network” refers to a set of proteins involved in metabolic pathways. In this invention, metabolic networks are mapped using Virotrap.

The term “protein folding network” refers to a set of proteins involved in chaperoning and folding. In this invention, protein folding networks are mapped using Virotrap.

The term “ubiquitin-proteasome network” refers to a set of proteins involved in protein degradation. In this invention, the ubiquitin-proteasome network is mapped using Virotrap.

The term “DNA repair network” refers to a set of proteins involved in maintaining genomic integrity. In this invention, the DNA repair network is mapped using Virotrap.

The term “cell cycle network” refers to a set of proteins regulating cell division. In this invention, the cell cycle network is mapped using Virotrap.

The term “apoptosis network” refers to a set of proteins regulating programmed cell death. In this invention, the apoptosis network is mapped using Virotrap.

The term “autophagy network” refers to a set of proteins regulating cellular self-degradation. In this invention, the autophagy network is mapped using Virotrap.

The term “endocytosis network” refers to a set of proteins regulating internalization of extracellular material. In this invention, the endocytosis network is mapped using Virotrap.

The term “exocytosis network” refers to a set of proteins regulating secretion of intracellular material. In this invention, the exocytosis network is mapped using Virotrap.

The term “vesicular trafficking network” refers to a set of proteins regulating transport between cellular compartments. In this invention, the vesicular trafficking network is mapped using Virotrap.

The term “membrane remodeling network” refers to a set of proteins regulating changes in membrane structure. In this invention, the membrane remodeling network is mapped using Virotrap.

The term “cytoskeletal network” refers to a set of proteins forming the structural framework of the cell. In this invention, the cytoskeletal network is mapped using Virotrap.

The term “nuclear transport network” refers to a set of proteins regulating movement between the nucleus and cytoplasm. In this invention, the nuclear transport network is mapped using Virotrap.

The term “chromatin remodeling network” refers to a set of proteins regulating DNA accessibility. In this invention, the chromatin remodeling network is mapped using Virotrap.

The term “RNA processing network” refers to a set of proteins involved in RNA splicing, stability, and translation. In this invention, the RNA processing network is mapped using Virotrap.

The term “signal transduction cascade” refers to a series of molecular events that transmit a signal from the cell surface to the nucleus. In this invention, signal transduction cascades are mapped under native conditions.

The term “kinase-substrate network” refers to a set of kinases and their substrates. In this invention, kinase-substrate networks are mapped using Virotrap.

The term “phosphatase-substrate network” refers to a set of phosphatases and their substrates. In this invention, phosphatase-substrate networks are mapped using Virotrap.

The term “GTPase-effector network” refers to a set of GTPases and their downstream effectors. In this invention, GTPase-effector networks are mapped using Virotrap.

The term “ubiquitin ligase-substrate network” refers to a set of ubiquitin ligases and their substrates. In this invention, ubiquitin ligase-substrate networks are mapped using Virotrap.

The term “SUMO ligase-substrate network” refers to a set of SUMO ligases and their substrates. In this invention, SUMO ligase-substrate networks are mapped using Virotrap.

The term “acetyltransferase-substrate network” refers to a set of acetyltransferases and their substrates. In this invention, acetyltransferase-substrate networks are mapped using Virotrap.

The term “methyltransferase-substrate network” refers to a set of methyltransferases and their substrates. In this invention, methyltransferase-substrate networks are mapped using Virotrap.

The term “phosphorylation-dependent interaction network” refers to a set of interactions regulated by phosphorylation. In this invention, phosphorylation-dependent interaction networks are mapped using Virotrap.

The term “ubiquitination-dependent interaction network” refers to a set of interactions regulated by ubiquitination. In this invention, ubiquitination-dependent interaction networks are mapped using Virotrap.

The term “SUMOylation-dependent interaction network” refers to a set of interactions regulated by SUMOylation. In this invention, SUMOylation-dependent interaction networks are mapped using Virotrap.

The term “acetylation-dependent interaction network” refers to a set of interactions regulated by acetylation. In this invention, acetylation-dependent interaction networks are mapped using Virotrap.

The term “methylation-dependent interaction network” refers to a set of interactions regulated by methylation. In this invention, methylation-dependent interaction networks are mapped using Virotrap.

The term “lipidation-dependent interaction network” refers to a set of interactions regulated by lipid modification. In this invention, lipidation-dependent interaction networks are mapped using Virotrap.

The term “glycosylation-dependent interaction network” refers to a set of interactions regulated by glycosylation. In this invention, glycosylation-dependent interaction networks are mapped using Virotrap.

The term “palmitoylation-dependent interaction network” refers to a set of interactions regulated by palmitoylation. In this invention, palmitoylation-dependent interaction networks are mapped using Virotrap.

The term “myristoylation-dependent interaction network” refers to a set of interactions regulated by myristoylation. In this invention, myristoylation-dependent interaction networks are mapped using Virotrap.

The term “farnesylation-dependent interaction network” refers to a set of interactions regulated by farnesylation. In this invention, farnesylation-dependent interaction networks are mapped using Virotrap.

The term “geranylgeranylation-dependent interaction network” refers to a set of interactions regulated by geranylgeranylation. In this invention, geranylgeranylation-dependent interaction networks are mapped using Virotrap.

The term “prey-bait ratio” refers to the relative expression levels of prey and bait proteins. In this invention, prey-bait ratios are optimized to ensure efficient capture without overexpression artifacts.

The term “expression level” refers to the amount of a protein produced in a cell. In this invention, expression levels are controlled by promoter strength and transfection efficiency.

The term “transfection ratio” refers to the relative amounts of plasmids used in transfection. In this invention, transfection ratios are standardized for reproducibility.

The term “DNA quantity” refers to the amount of plasmid DNA used in transfection. In this invention, DNA quantities are specified for each component.

The term “cell density” refers to the number of cells per unit area. In this invention, cell density is optimized for VLP production.

The term “transfection time” refers to the duration of DNA exposure to cells. In this invention, transfection time is 24 hours.

The term “harvest time” refers to the time at which supernatant is collected. In this invention, harvest time is 24–32 hours post-transfection.

The term “supernatant volume” refers to the amount of culture medium collected. In this invention, supernatant volume is scaled to cell number.

The term “bead volume” refers to the amount of magnetic beads used. In this invention, bead volume is standardized per sample.

The term “antibody amount” refers to the quantity of antibody used for bead loading. In this invention, antibody amount is optimized for saturation.

The term “binding time” refers to the duration of VLP-bead interaction. In this invention, binding time is 2 hours.

The term “washing steps” refer to the number of times beads are washed. In this invention, two washing steps are performed.

The term “elution buffer” refers to the solution used to release VLPs from beads. In this invention, elution buffer contains FLAG peptide.

The term “elution time” refers to the duration of elution. In this invention, elution time is 30 minutes.

The term “elution temperature” refers to the temperature during elution. In this invention, elution temperature is 37°C.

The term “lysate” refers to the cellular contents released by lysis. In this invention, lysates are prepared for western blotting but not for purification.

The term “control lysate” refers to a lysate from cells expressing GAG alone. In this invention, control lysates are used to confirm bait expression.

The term “experimental lysate” refers to a lysate from cells expressing GAG-bait and prey. In this invention, experimental lysates are used to confirm expression but not for purification.

The term “VLP lysate” refers to the contents of purified VLPs. In this invention, VLP lysates are used for mass spectrometry.

The term “negative control VLP” refers to VLPs produced without a prey. In this invention, negative control VLPs are used to identify background proteins.

The term “positive control VLP” refers to VLPs produced with a known bait-prey pair. In this invention, positive control VLPs are used to validate the system.

The term “bait-only VLP” refers to VLPs produced with GAG-bait but no prey. In this invention, bait-only VLPs are used to identify bait-specific binders.

The term “prey-only VLP” refers to VLPs produced with prey but no GAG-bait. In this invention, prey-only VLPs are not produced, as prey does not incorporate into VLPs without bait.

The term “empty VLP” refers to VLPs produced without any fusion protein. In this invention, empty VLPs are not produced, as GAG is required for assembly.

The term “tagged GAG” refers to GAG fused to an epitope tag. In this invention, tagged GAG is not used; instead, the envelope is tagged.

The term “untagged envelope” refers to envelope glycoprotein without a tag. In this invention, untagged envelope is co-expressed with tagged envelope to ensure proper trimerization.

The term “trimerization” refers to the assembly of three subunits into a stable complex. In this invention, VSV-G forms trimers on the VLP surface, enabling efficient capture.

The term “epitope accessibility” refers to the exposure of a tag on the surface of a particle. In this invention, epitope accessibility is ensured by the surface localization of VSV-G.

The term “particle yield” refers to the number of VLPs produced per cell. In this invention, particle yield is sufficient for mass spectrometry without concentration.

The term “particle purity” refers to the proportion of VLPs relative to contaminants. In this invention, particle purity is high due to the specificity of affinity capture.

The term “background contamination” refers to non-specific proteins co-purified with VLPs. In this invention, background contamination is minimized by the blacklist strategy.

The term “specific recovery” refers to the proportion of true interactors recovered relative to total proteins. In this invention, specific recovery is high due to the combination of encapsulation and affinity capture.

The term “interaction detection rate” refers to the percentage of known interactions detected. In this invention, the interaction detection rate is 30% for the human positive reference set.

The term “false discovery rate” refers to the proportion of false positives among detected interactions. In this invention, the false discovery rate is less than 5%.

The term “true positive rate” refers to the proportion of true interactions detected. In this invention, the true positive rate is 30% for the positive reference set.

The term “sensitivity” refers to the ability to detect true positives. In this invention, sensitivity is high due to the concentration effect of VLPs.

The term “specificity” refers to the ability to avoid false positives. In this invention, specificity is high due to the use of controls and filtering.

The term “accuracy” refers to the closeness of measured values to true values. In this invention, accuracy is high due to orthogonal validation.

The term “precision” refers to the reproducibility of measurements. In this invention, precision is high due to standardized protocols.

The term “reproducibility” refers to the consistency of results across experiments. In this invention, reproducibility is demonstrated across multiple baits and biological replicates.

The term “robustness” refers to the ability to perform under variable conditions. In this invention, robustness is demonstrated across cell lines and transfection methods.

The term “scalability” refers to the ability to increase sample size. In this invention, scalability is demonstrated by the use of 107 cells per experiment.

The term “portability” refers to the ability to transfer the method to other labs. In this invention, portability is ensured by the use of standard reagents.

The term “standardization” refers to the use of uniform procedures. In this invention, standardization is achieved through defined plasmid constructs and protocols.

The term “automation” refers to the use of machines to perform repetitive tasks. In this invention, automation is enabled by magnetic beads and standardized elution.

The term “cost-effectiveness” refers to the balance between performance and cost. In this invention, cost-effectiveness is high due to the elimination of ultracentrifugation.

The term “time-efficiency” refers to the reduction in experimental time. In this invention, time-efficiency is high due to the single-step capture.

The term “user-friendliness” refers to the ease of use. In this invention, user-friendliness is enhanced by the simplicity of the protocol.

The term “versatility” refers to the ability to adapt to different applications. In this invention, versatility is demonstrated by the capture of binary, multiprotein, and small molecule interactions.

The term “innovation” refers to the introduction of a new concept. In this invention, innovation lies in the use of VLPs as native-state interaction traps.

The term “paradigm shift” refers to a fundamental change in methodology. In this invention, the paradigm shift is from lysis to encapsulation.

The term “translational potential” refers to the ability to move from bench to bedside. In this invention, translational potential is high due to applications in drug discovery.

The term “commercial viability” refers to the economic potential of the invention. In this invention, commercial viability is high due to the platform nature of the technology.

The term “intellectual property” refers to legally protected inventions. In this invention, the method, system, and kit constitute valuable intellectual property.

The term “patent landscape” refers to the collection of existing patents. In this invention, the patent landscape is sparse, and this invention occupies a novel space.

The term “freedom to operate” refers to the ability to commercialize without infringement. In this invention, freedom to operate is established by the novelty of the combination.

The term “regulatory compliance” refers to adherence to legal and ethical standards. In this invention, regulatory compliance is ensured by the use of non-infectious VLPs.

The term “ethical use” refers to responsible application. In this invention, ethical use is ensured by the use of immortalized cell lines.

The term “sustainability” refers to environmental impact. In this invention, sustainability is enhanced by reduced reagent use.

The term “global applicability” refers to worldwide usability. In this invention, global applicability is ensured by the use of universally available reagents.

The term “open science” refers to public sharing of data and methods. In this invention, open science is practiced through PRIDE deposition.

The term “data sharing” refers to public release of data. In this invention, data sharing is implemented through PRIDE.

The term “reagent sharing” refers to distribution of plasmids. In this invention, reagent sharing is facilitated by Addgene.

The term “collaborative potential” refers to the ability to foster collaboration. In this invention, collaborative potential is high due to the platform nature.

The term “educational value” refers to utility in teaching. In this invention, educational value is high due to the clear mechanism.

The term “technical documentation” refers to written instructions. In this invention, technical documentation is provided in the form of protocols and plasmid maps.

The term “reagent kit” refers to a packaged set of materials. In this invention, a reagent kit is provided comprising plasmids, antibodies, beads, and instructions.

The term “service offering” refers to the provision of the method as a service. In this invention, service offerings include screening and interactome mapping.

The term “diagnostic application” refers to disease detection. In this invention, diagnostic applications include the identification of disease-specific complexes.

The term “therapeutic application” refers to drug development. In this invention, therapeutic applications include target identification and off-target profiling.

The term “biopharmaceutical development” refers to the development of biological drugs. In this invention, biopharmaceutical development is supported by target identification.

The term “target validation” refers to confirming a protein is a viable drug target. In this invention, target validation is achieved by demonstrating interaction in a native context.

The term “lead optimization” refers to improving a drug candidate. In this invention, lead optimization is informed by off-target interaction profiles.

The term “pharmacokinetic profiling” refers to studying drug absorption and metabolism. In this invention, pharmacokinetic profiling is informed by drug-binding protein identification.

The term “toxicology screening” refers to assessing drug toxicity. In this invention, toxicology screening is enhanced by off-target interaction detection.

The term “personalized therapy” refers to tailoring treatment to individuals. In this invention, personalized therapy is enabled by mapping patient-derived interactomes.

The term “biomarker panel” refers to a set of biomarkers for diagnosis. In this invention, biomarker panels are generated from Virotrap-identified complexes.

The term “clinical translation” refers to applying research to clinical practice. In this invention, clinical translation is supported by disease-relevant interaction mapping.

The term “precision diagnostics” refers to using molecular markers for diagnosis. In this invention, precision diagnostics are enabled by disease-specific complex detection.

The term “drug resistance profiling” refers to identifying resistance mechanisms. In this invention, drug resistance profiling is enabled by altered interaction networks.

The term “cancer interactome” refers to protein interactions altered in cancer. In this invention, the cancer interactome is mapped using Virotrap.

The term “neurodegenerative interactome” refers to interactions altered in neurodegenerative diseases. In this invention, the neurodegenerative interactome is mapped using Virotrap.

The term “infectious disease interactome” refers to interactions altered during infection. In this invention, the infectious disease interactome is mapped using Virotrap.

The term “immune interactome” refers to interactions involved in immune signaling. In this invention, the immune interactome is mapped using Virotrap.

The term “signaling network” refers to interconnected proteins transmitting signals. In this invention, signaling networks are mapped under native conditions.

The term “transcriptional network” refers to proteins regulating gene expression. In this invention, transcriptional networks are mapped using Virotrap.

The term “metabolic network” refers to proteins involved in metabolism. In this invention, metabolic networks are mapped using Virotrap.

The term “protein folding network” refers to chaperones and folding machinery. In this invention, protein folding networks are mapped using Virotrap.

The term “ubiquitin-proteasome network” refers to degradation machinery. In this invention, the ubiquitin-proteasome network is mapped using Virotrap.

The term “DNA repair network” refers to proteins maintaining genomic integrity. In this invention, the DNA repair network is mapped using Virotrap.

The term “cell cycle network” refers to proteins regulating division. In this invention, the cell cycle network is mapped using Virotrap.

The term “apoptosis network” refers to proteins regulating cell death. In this invention, the apoptosis network is mapped using Virotrap.

The term “autophagy network” refers to proteins regulating self-degradation. In this invention, the autophagy network is mapped using Virotrap.

The term “endocytosis network” refers to proteins internalizing extracellular material. In this invention, the endocytosis network is mapped using Virotrap.

The term “exocytosis network” refers to proteins secreting intracellular material. In this invention, the exocytosis network is mapped using Virotrap.

The term “vesicular trafficking network” refers to proteins transporting between compartments. In this invention, the vesicular trafficking network is mapped using Virotrap.

The term “membrane remodeling network” refers to proteins altering membrane structure. In this invention, the membrane remodeling network is mapped using Virotrap.

The term “cytoskeletal network” refers to structural proteins. In this invention, the cytoskeletal network is mapped using Virotrap.

The term “nuclear transport network” refers to proteins moving material between nucleus and cytoplasm. In this invention, the nuclear transport network is mapped using Virotrap.

The term “chromatin remodeling network” refers to proteins regulating DNA accessibility. In this invention, the chromatin remodeling network is mapped using Virotrap.

The term “RNA processing network” refers to proteins involved in RNA metabolism. In this invention, the RNA processing network is mapped using Virotrap.

The term “signal transduction cascade” refers to a series of molecular events transmitting a signal. In this invention, signal transduction cascades are mapped under native conditions.

The term “kinase-substrate network” refers to kinases and their substrates. In this invention, kinase-substrate networks are mapped using Virotrap.

The term “phosphatase-substrate network” refers to phosphatases and their substrates. In this invention, phosphatase-substrate networks are mapped using Virotrap.

The term “GTPase-effector network” refers to GTPases and their effectors. In this invention, GTPase-effector networks are mapped using Virotrap.

The term “ubiquitin ligase-substrate network” refers to ubiquitin ligases and their substrates. In this invention, ubiquitin ligase-substrate networks are mapped using Virotrap.

The term “SUMO ligase-substrate network” refers to SUMO ligases and their substrates. In this invention, SUMO ligase-substrate networks are mapped using Virotrap.

The term “acetyltransferase-substrate network” refers to acetyltransferases and their substrates. In this invention, acetyltransferase-substrate networks are mapped using Virotrap.

The term “methyltransferase-substrate network” refers to methyltransferases and their substrates. In this invention, methyltransferase-substrate networks are mapped using Virotrap.

The term “phosphorylation-dependent interaction network” refers to interactions regulated by phosphorylation. In this invention, phosphorylation-dependent interaction networks are mapped using Virotrap.

The term “ubiquitination-dependent interaction network” refers to interactions regulated by ubiquitination. In this invention, ubiquitination-dependent interaction networks are mapped using Virotrap.

The term “SUMOylation-dependent interaction network” refers to interactions regulated by SUMOylation. In this invention, SUMOylation-dependent interaction networks are mapped using Virotrap.

The term “acetylation-dependent interaction network” refers to interactions regulated by acetylation. In this invention, acetylation-dependent interaction networks are mapped using Virotrap.

The term “methylation-dependent interaction network” refers to interactions regulated by methylation. In this invention, methylation-dependent interaction networks are mapped using Virotrap.

The term “lipidation-dependent interaction network” refers to interactions regulated by lipid modification. In this invention, lipidation-dependent interaction networks are mapped using Virotrap.

The term “glycosylation-dependent interaction network” refers to interactions regulated by glycosylation. In this invention, glycosylation-dependent interaction networks are mapped using Virotrap.

The term “palmitoylation-dependent interaction network” refers to interactions regulated by palmitoylation. In this invention, palmitoylation-dependent interaction networks are mapped using Virotrap.

The term “myristoylation-dependent interaction network” refers to interactions regulated by myristoylation. In this invention, myristoylation-dependent interaction networks are mapped using Virotrap.

The term “farnesylation-dependent interaction network” refers to interactions regulated by farnesylation. In this invention, farnesylation-dependent interaction networks are mapped using Virotrap.

The term “geranylgeranylation-dependent interaction network” refers to interactions regulated by geranylgeranylation. In this invention, geranylgeranylation-dependent interaction networks are mapped using Virotrap.

The term “prey-bait ratio” refers to the relative expression levels of prey and bait proteins. In this invention, prey-bait ratios are optimized to ensure efficient capture without overexpression artifacts.

The term “expression level” refers to the amount of a protein produced in a cell. In this invention, expression levels are controlled by promoter strength and transfection efficiency.

The term “transfection ratio” refers to the relative amounts of plasmids used in transfection. In this invention, transfection ratios are standardized for reproducibility.

The term “DNA quantity” refers to the amount of plasmid DNA used in transfection. In this invention, DNA quantities are specified for each component.

The term “cell density” refers to the number of cells per unit area. In this invention, cell density is optimized for VLP production.

The term “transfection time” refers to the duration of DNA exposure to cells. In this invention, transfection time is 24 hours.

The term “harvest time” refers to the time at which supernatant is collected. In this invention, harvest time is 24–32 hours post-transfection.

The term “supernatant volume” refers to the amount of culture medium collected. In this invention, supernatant volume is scaled to cell number.

The term “bead volume” refers to the amount of magnetic beads used. In this invention, bead volume is standardized per sample.

The term “antibody amount” refers to the quantity of antibody used for bead loading. In this invention, antibody amount is optimized for saturation.

The term “binding time” refers to the duration of VLP-bead interaction. In this invention, binding time is 2 hours.

The term “washing steps” refer to the number of times beads are washed. In this invention, two washing steps are performed.

The term “elution buffer” refers to the solution used to release VLPs from beads. In this invention, elution buffer contains FLAG peptide.

The term “elution time” refers to the duration of elution. In this invention, elution time is 30 minutes.

The term “elution temperature” refers to the temperature during elution. In this invention, elution temperature is 37°C.

The term “lysate” refers to the cellular contents released by lysis. In this invention, lysates are prepared for western blotting but not for purification.

The term “control lysate” refers to a lysate from cells expressing GAG alone. In this invention, control lysates are used to confirm bait expression.

The term “experimental lysate” refers to a lysate from cells expressing GAG-bait and prey. In this invention, experimental lysates are used to confirm expression but not for purification.

The term “VLP lysate” refers to the contents of purified VLPs. In this invention, VLP lysates are used for mass spectrometry.

The term “negative control VLP” refers to VLPs produced without a prey. In this invention, negative control VLPs are used to identify background proteins.

The term “positive control VLP” refers to VLPs produced with a known bait-prey pair. In this invention, positive control VLPs are used to validate the system.

The term “bait-only VLP” refers to VLPs produced with GAG-bait but no prey. In this invention, bait-only VLPs are used to identify bait-specific binders.

The term “prey-only VLP” refers to VLPs produced with prey but no GAG-bait. In this invention, prey-only VLPs are not produced, as prey does not incorporate into VLPs without bait.

The term “empty VLP” refers to VLPs produced without any fusion protein. In this invention, empty VLPs are not produced, as GAG is required for assembly.

The term “tagged GAG” refers to GAG fused to an epitope tag. In this invention, tagged GAG is not used; instead, the envelope is tagged.

The term “untagged envelope” refers to envelope glycoprotein without a tag. In this invention, untagged envelope is co-expressed with tagged envelope to ensure proper trimerization.

The term “trimerization” refers to the assembly of three subunits into a stable complex. In this invention, VSV-G forms trimers on the VLP surface, enabling efficient capture.

The term “epitope accessibility” refers to the exposure of a tag on the surface of a particle. In this invention, epitope accessibility is ensured by the surface localization of VSV-G.

The term “particle yield” refers to the number of VLPs produced per cell. In this invention, particle yield is sufficient for mass spectrometry without concentration.

The term “particle purity” refers to the proportion of VLPs relative to contaminants. In this invention, particle purity is high due to the specificity of affinity capture.

The term “background contamination” refers to non-specific proteins co-purified with VLPs. In this invention, background contamination is minimized by the blacklist strategy.

The term “specific recovery” refers to the proportion of true interactors recovered relative to total proteins. In this invention, specific recovery is high due to the combination of encapsulation and affinity capture.

The term “interaction detection rate” refers to the percentage of known interactions detected. In this invention, the interaction detection rate is 30% for the human positive reference set.

The term “false discovery rate” refers to the proportion of false positives among detected interactions. In this invention, the false discovery rate is less than 5%.

The term “true positive rate” refers to the proportion of true interactions detected. In this invention, the true positive rate is 30% for the positive reference set.

The term “sensitivity” refers to the ability to detect true positives. In this invention, sensitivity is high due to the concentration effect of VLPs.

The term “specificity” refers to the ability to avoid false positives. In this invention, specificity is high due to the use of controls and filtering.

The term “accuracy” refers to the closeness of measured values to true values. In this invention, accuracy is high due to orthogonal validation.

The term “precision” refers to the reproducibility of measurements. In this invention, precision is high due to standardized protocols.

The term “reproducibility” refers to the consistency of results across experiments. In this invention, reproducibility is demonstrated across multiple baits and biological replicates.

The term “robustness” refers to the ability to perform under variable conditions. In this invention, robustness is demonstrated across cell lines and transfection methods.

The term “scalability” refers to the ability to increase sample size. In this invention, scalability is demonstrated by the use of 107 cells per experiment.

The term “portability” refers to the ability to transfer the method to other labs. In this invention, portability is ensured by the use of standard reagents.

The term “standardization” refers to the use of uniform procedures. In this invention, standardization is achieved through defined plasmid constructs and protocols.

The term “automation” refers to the use of machines to perform repetitive tasks. In this invention, automation is enabled by magnetic beads and standardized elution.

The term “cost-effectiveness” refers to the balance between performance and cost. In this invention, cost-effectiveness is high due to the elimination of ultracentrifugation.

The term “time-efficiency” refers to the reduction in experimental time. In this invention, time-efficiency is high due to the single-step capture.

The term “user-friendliness” refers to the ease of use. In this invention, user-friendliness is enhanced by the simplicity of the protocol.

The term “versatility” refers to the ability to adapt to different applications. In this invention, versatility is demonstrated by the capture of binary, multiprotein, and small molecule interactions.

The term “innovation” refers to the introduction of a new concept. In this invention, innovation lies in the use of VLPs as native-state interaction traps.

The term “paradigm shift” refers to a fundamental change in methodology. In this invention, the paradigm shift is from lysis to encapsulation.

The term “translational potential” refers to the ability to move from bench to bedside. In this invention, translational potential is high due to applications in drug discovery.

The term “commercial viability” refers to the economic potential of the invention. In this invention, commercial viability is high due to the platform nature of the technology.

The term “intellectual property” refers to legally protected inventions. In this invention, the method, system, and kit constitute valuable intellectual property.

The term “patent landscape” refers to the collection of existing patents. In this invention, the patent landscape is sparse, and this invention occupies a novel space.

The term “freedom to operate” refers to the ability to commercialize without infringement. In this invention, freedom to operate is established by the novelty of the combination.

The term “regulatory compliance” refers to adherence to legal and ethical standards. In this invention, regulatory compliance is ensured by the use of non-infectious VLPs.

The term “ethical use” refers to responsible application. In this invention, ethical use is ensured by the use of immortalized cell lines.

The term “sustainability” refers to environmental impact. In this invention, sustainability is enhanced by reduced reagent use.

The term “global applicability” refers to worldwide usability. In this invention, global applicability is ensured by the use of universally available reagents.

The term “open science” refers to public sharing of data and methods. In this invention, open science is practiced through PRIDE deposition.

The term “data sharing” refers to public release of data. In this invention, data sharing is implemented through PRIDE.

The term “reagent sharing” refers to distribution of plasmids. In this invention, reagent sharing is facilitated by Addgene.

The term “collaborative potential” refers to the ability to foster collaboration. In this invention, collaborative potential is high due to the platform nature.

The term “educational value” refers to utility in teaching. In this invention, educational value is high due to the clear mechanism.

The term “technical documentation” refers to written instructions. In this invention, technical documentation is provided in the form of protocols and plasmid maps.

The term “reagent kit” refers to a packaged set of materials. In this invention, a reagent kit is provided comprising plasmids, antibodies, beads, and instructions.

The term “service offering” refers to the provision of the method as a service. In this invention, service offerings include screening and interactome mapping.

The term “diagnostic application” refers to disease detection. In this invention, diagnostic applications include the identification of disease-specific complexes.

The term “therapeutic application” refers to drug development. In this invention, therapeutic applications include target identification and off-target profiling.

The term “biopharmaceutical development” refers to the development of biological drugs. In this invention, biopharmaceutical development is supported by target identification.

The term “target validation” refers to confirming a protein is a viable drug target. In this invention, target validation is achieved by demonstrating interaction in a native context.

The term “lead optimization” refers to improving a drug candidate. In this invention, lead optimization is informed by off-target interaction profiles.

The term “pharmacokinetic profiling” refers to studying drug absorption and metabolism. In this invention, pharmacokinetic profiling is informed by drug-binding protein identification.

The term “toxicology screening” refers to assessing drug toxicity. In this invention, toxicology screening is enhanced by off-target interaction detection.

The term “personalized therapy” refers to tailoring treatment to individuals. In this invention, personalized therapy is enabled by mapping patient-derived interactomes.

The term “biomarker panel” refers to a set of biomarkers for diagnosis. In this invention, biomarker panels are generated from Virotrap-identified complexes.

The term “clinical translation” refers to applying research to clinical practice. In this invention, clinical translation is supported by disease-relevant interaction mapping.

The term “precision diagnostics” refers to using molecular markers for diagnosis. In this invention, precision diagnostics are enabled by disease-specific complex detection.

The term “drug resistance profiling” refers to identifying resistance mechanisms. In this invention, drug resistance profiling is enabled by altered interaction networks.

The term “cancer interactome” refers to protein interactions altered in cancer. In this invention, the cancer interactome is mapped using Virotrap.

The term “neurodegenerative interactome” refers to interactions altered in neurodegenerative diseases. In this invention, the neurodegenerative interactome is mapped using Virotrap.

The term “infectious disease interactome” refers to interactions altered during infection. In this invention, the infectious disease interactome is mapped using Virotrap.

The term “immune interactome” refers to interactions involved in immune signaling. In this invention, the immune interactome is mapped using Virotrap.

The term “signaling network” refers to interconnected proteins transmitting signals. In this invention, signaling networks are mapped under native conditions.

The term “transcriptional network” refers to proteins regulating gene expression. In this invention, transcriptional networks are mapped using Virotrap.

The term “metabolic network” refers to proteins involved in metabolism. In this invention, metabolic networks are mapped using Virotrap.

The term “protein folding network” refers to chaperones and folding machinery. In this invention, protein folding networks are mapped using Virotrap.

The term “ubiquitin-proteasome network” refers to degradation machinery. In this invention, the ubiquitin-proteasome network is mapped using Virotrap.

The term “DNA repair network” refers to proteins maintaining genomic integrity. In this invention, the DNA repair network is mapped using Virotrap.

The term “cell cycle network” refers to proteins regulating division. In this invention, the cell cycle network is mapped using Virotrap.

The term “apoptosis network” refers to proteins regulating cell death. In this invention, the apoptosis network is mapped using Virotrap.

The term “autophagy network” refers to proteins regulating self-degradation. In this invention, the autophagy network is mapped using Virotrap.

The term “endocytosis network” refers to proteins internalizing extracellular material. In this invention, the endocytosis network is mapped using Virotrap.

The term “exocytosis network” refers to proteins secreting intracellular material. In this invention, the exocytosis network is mapped using Virotrap.

The term “vesicular trafficking network” refers to proteins transporting between compartments. In this invention, the vesicular trafficking network is mapped using Virotrap.

The term “membrane remodeling network” refers to proteins altering membrane structure. In this invention, the membrane remodeling network is mapped using Virotrap.

The term “cytoskeletal network” refers to structural proteins. In this invention, the cytoskeletal network is mapped using Virotrap.

The term “nuclear transport network” refers to proteins moving material between nucleus and cytoplasm. In this invention, the nuclear transport network is mapped using Virotrap.

The term “chromatin remodeling network” refers to proteins regulating DNA accessibility. In this invention, the chromatin remodeling network is mapped using Virotrap.

The term “RNA processing network” refers to proteins involved in RNA metabolism. In this invention, the RNA processing network is mapped using Virotrap.

The term “signal transduction cascade” refers to a series of molecular events transmitting a signal. In this invention, signal transduction cascades are mapped under native conditions.

The term “kinase-substrate network” refers to kinases and their substrates. In this invention, kinase-substrate networks are mapped using Virotrap.

The term “phosphatase-substrate network” refers to phosphatases and their substrates. In this invention, phosphatase-substrate networks are mapped using Virotrap.

The term “GTPase-effector network” refers to GTPases and their effectors. In this invention, GTPase-effector networks are mapped using Virotrap.

The term “ubiquitin ligase-substrate network” refers to ubiquitin ligases and their substrates. In this invention, ubiquitin ligase-substrate networks are mapped using Virotrap.

The term “SUMO ligase-substrate network” refers to SUMO ligases and their substrates. In this invention, SUMO ligase-substrate networks are mapped using Virotrap.

The term “acetyltransferase-substrate network” refers to acetyltransferases and their substrates. In this invention, acetyltransferase-substrate networks are mapped using Virotrap.

The term “methyltransferase-substrate network” refers to methyltransferases and their substrates. In this invention, methyltransferase-substrate networks are mapped using Virotrap.

The term “phosphorylation-dependent interaction network” refers to interactions regulated by phosphorylation. In this invention, phosphorylation-dependent interaction networks are mapped using Virotrap.

The term “ubiquitination-dependent interaction network” refers to interactions regulated by ubiquitination. In this invention, ubiquitination-dependent interaction networks are mapped using Virotrap.

The term “SUMOylation-dependent interaction network” refers to interactions regulated by SUMOylation. In this invention, SUMOylation-dependent interaction networks are mapped using Virotrap.

The term “acetylation-dependent interaction network” refers to interactions regulated by acetylation. In this invention, acetylation-dependent interaction networks are mapped using Virotrap.

The term “methylation-dependent interaction network” refers to interactions regulated by methylation. In this invention, methylation-dependent interaction networks are mapped using Virotrap.

The term “lipidation-dependent interaction network” refers to interactions regulated by lipid modification. In this invention, lipidation-dependent interaction networks are mapped using Virotrap.

The term “glycosylation-dependent interaction network” refers to interactions regulated by glycosylation. In this invention, glycosylation-dependent interaction networks are mapped using Virotrap.

The term “palmitoylation-dependent interaction network” refers to interactions regulated by palmitoylation. In this invention, palmitoylation-dependent interaction networks are mapped using Virotrap.

The term “myristoylation-dependent interaction network” refers to interactions regulated by myristoylation. In this invention, myristoylation-dependent interaction networks are mapped using Virotrap.

The term “farnesylation-dependent interaction network” refers to interactions regulated by farnesylation. In this invention, farnesylation-dependent interaction networks are mapped using Virotrap.

The term “geranylgeranylation-dependent interaction network” refers to interactions regulated by geranylgeranylation. In this invention, geranylgeranylation-dependent interaction networks are mapped using Virotrap.

The term “prey-bait ratio” refers to the relative expression levels of prey and bait proteins. In this invention, prey-bait ratios are optimized to ensure efficient capture without overexpression artifacts.

The term “expression level” refers to the amount of a protein produced in a cell. In this invention, expression levels are controlled by promoter strength and transfection efficiency.

The term “transfection ratio” refers to the relative amounts of plasmids used in transfection. In this invention, transfection ratios are standardized for reproducibility.

The term “DNA quantity” refers to the amount of plasmid DNA used in transfection. In this invention, DNA quantities are specified for each component.

The term “cell density” refers to the number of cells per unit area. In this invention, cell density is optimized for VLP production.

The term “transfection time” refers to the duration of DNA exposure to cells. In this invention, transfection time is 24 hours.

The term “harvest time” refers to the time at which supernatant is collected. In this invention, harvest time is 24–32 hours post-transfection.

The term “supernatant volume” refers to the amount of culture medium collected. In this invention, supernatant volume is scaled to cell number.

The term “bead volume” refers to the amount of magnetic beads used. In this invention, bead volume is standardized per sample.

The term “antibody amount” refers to the quantity of antibody used for bead loading. In this invention, antibody amount is optimized for saturation.

The term “binding time” refers to the duration of VLP-bead interaction. In this invention, binding time is 2 hours.

The term “washing steps” refer to the number of times beads are washed. In this invention, two washing steps are performed.

The term “elution buffer” refers to the solution used to release VLPs from beads. In this invention, elution buffer contains FLAG peptide.

The term “elution time” refers to the duration of elution. In this invention, elution time is 30 minutes.

The term “elution temperature” refers to the temperature during elution. In this invention, elution temperature is 37°C.

The term “lysate” refers to the cellular contents released by lysis. In this invention, lysates are prepared for western blotting but not for purification.

The term “control lysate” refers to a lysate from cells expressing GAG alone. In this invention, control lysates are used to confirm bait expression.

The term “experimental lysate” refers to a lysate from cells expressing GAG-bait and prey. In this invention, experimental lysates are used to confirm expression but not for purification.

The term “VLP lysate” refers to the contents of purified VLPs. In this invention, VLP lysates are used for mass spectrometry.

The term “negative control VLP” refers to VLPs produced without a prey. In this invention, negative control VLPs are used to identify background proteins.

The term “positive control VLP” refers to VLPs produced with a known bait-prey pair. In this invention, positive control VLPs are used to validate the system.

The term “bait-only VLP” refers to VLPs produced with GAG-bait but no prey. In this invention, bait-only VLPs are used to identify bait-specific binders.

The term “prey-only VLP” refers to VLPs produced with prey but no GAG-bait. In this invention, prey-only VLPs are not produced, as prey does not incorporate into VLPs without bait.

The term “empty VLP” refers to VLPs produced without any fusion protein. In this invention, empty VLPs are not produced, as GAG is required for assembly.

The term “tagged GAG” refers to GAG fused to an epitope tag. In this invention, tagged GAG is not used; instead, the envelope is tagged.

The term “untagged envelope” refers to envelope glycoprotein without a tag. In this invention, untagged envelope is co-expressed with tagged envelope to ensure proper trimerization.

The term “trimerization” refers to the assembly of three subunits into a stable complex. In this invention, VSV-G forms trimers on the VLP surface, enabling efficient capture.

The term “epitope accessibility” refers to the exposure of a tag on the surface of a particle. In this invention, epitope accessibility is ensured by the surface localization of VSV-G.

The term “particle yield” refers to the number of VLPs produced per cell. In this invention, particle yield is sufficient for mass spectrometry without concentration.

The term “particle purity” refers to the proportion of VLPs relative to contaminants. In this invention, particle purity is high due to the specificity of affinity capture.

The term “background contamination” refers to non-specific proteins co-purified with VLPs. In this invention, background contamination is minimized by the blacklist strategy.

The term “specific recovery” refers to the proportion of true interactors recovered relative to total proteins. In this invention, specific recovery is high due to the combination of encapsulation and affinity capture.

The term “interaction detection rate” refers to the percentage of known interactions detected. In this invention, the interaction detection rate is 30% for the human positive reference set.

The term “false discovery rate” refers to the proportion of false positives among detected interactions. In this invention, the false discovery rate is less than 5%.

The term “true positive rate” refers to the proportion of true interactions detected. In this invention, the true positive rate is 30% for the positive reference set.

The term “sensitivity” refers to the ability to detect true positives. In this invention, sensitivity is high due to the concentration effect of VLPs.

The term “specificity” refers to the ability to avoid false positives. In this invention, specificity is high due to the use of controls and filtering.

The term “accuracy” refers to the closeness of measured values to true values. In this invention, accuracy is high due to orthogonal validation.

The term “precision” refers to the reproducibility of measurements. In this invention, precision is high due to standardized protocols.

The term “reproducibility” refers to the consistency of results across experiments. In this invention, reproducibility is demonstrated across multiple baits and biological replicates.

The term “robustness” refers to the ability to perform under variable conditions. In this invention, robustness is demonstrated across cell lines and transfection methods.

The term “scalability” refers to the ability to increase sample size. In this invention, scalability is demonstrated by the use of 107 cells per experiment.

The term “portability” refers to the ability to transfer the method to other labs. In this invention, portability is ensured by the use of standard reagents.

The term “standardization” refers to the use of uniform procedures. In this invention, standardization is achieved through defined plasmid constructs and protocols.

The term “automation” refers to the use of machines to perform repetitive tasks. In this invention, automation is enabled by magnetic beads and standardized elution.

The term “cost-effectiveness” refers to the balance between performance and cost. In this invention, cost-effectiveness is high due to the elimination of ultracentrifugation.

The term “time-efficiency” refers to the reduction in experimental time. In this invention, time-efficiency is high due to the single-step capture.

The term “user-friendliness” refers to the ease of use. In this invention, user-friendliness is enhanced by the simplicity of the protocol.

The term “versatility” refers to the ability to adapt to different applications. In this invention, versatility is demonstrated by the capture of binary, multiprotein, and small molecule interactions.

The term “innovation” refers to the introduction of a new concept. In this invention, innovation lies in the use of VLPs as native-state interaction traps.

The term “paradigm shift” refers to a fundamental change in methodology. In this invention, the paradigm shift is from lysis to encapsulation.

The term “translational potential” refers to the ability to move from bench to bedside. In this invention, translational potential is high due to applications in drug discovery.

The term “commercial viability” refers to the economic potential of the invention. In this invention, commercial viability is high due to the platform nature of the technology.

The term “intellectual property” refers to legally protected inventions. In this invention, the method, system, and kit constitute valuable intellectual property.

The term “patent landscape” refers to the collection of existing patents. In this invention, the patent landscape is sparse, and this invention occupies a novel space.

The term “freedom to operate” refers to the ability to commercialize without infringement. In this invention, freedom to operate is established by the novelty of the combination.

The term “regulatory compliance” refers to adherence to legal and ethical standards. In this invention, regulatory compliance is ensured by the use of non-infectious VLPs.

The term “ethical use” refers to responsible application. In this invention, ethical use is ensured by the use of immortalized cell lines.

The term “sustainability” refers to environmental impact. In this invention, sustainability is enhanced by reduced reagent use.

The term “global applicability” refers to worldwide usability. In this invention, global applicability is ensured by the use of universally available reagents.

The term “open science” refers to public sharing of data and methods. In this invention, open science is practiced through PRIDE deposition.

The term “data sharing” refers to public release of data. In this invention, data sharing is implemented through PRIDE.

The term “reagent sharing” refers to distribution of plasmids. In this invention, reagent sharing is facilitated by Addgene.

The term “collaborative potential” refers to the ability to foster collaboration. In this invention, collaborative potential is high due to the platform nature.

The term “educational value” refers to utility in teaching. In this invention, educational value is high due to the clear mechanism.

The term “technical documentation” refers to written instructions. In this invention, technical documentation is provided in the form of protocols and plasmid maps.

The term “reagent kit” refers to a packaged set of materials. In this invention, a reagent kit is provided comprising plasmids, antibodies, beads, and instructions.

The term “service offering” refers to the provision of the method as a service. In this invention, service offerings include screening and interactome mapping.

The term “diagnostic application” refers to disease detection. In this invention, diagnostic applications include the identification of disease-specific complexes.

The term “therapeutic application” refers to drug development. In this invention, therapeutic applications include target identification and off-target profiling.

The term “biopharmaceutical development” refers to the development of biological drugs. In this invention, biopharmaceutical development is supported by target identification.

The term “target validation” refers to confirming a protein is a viable drug target. In this invention, target validation is achieved by demonstrating interaction in a native context.

The term “lead optimization” refers to improving a drug candidate. In this invention, lead optimization is informed by off-target interaction profiles.

The term “pharmacokinetic profiling” refers to studying drug absorption and metabolism. In this invention, pharmacokinetic profiling is informed by drug-binding protein identification.

The term “toxicology screening” refers to assessing drug toxicity. In this invention, toxicology screening is enhanced by off-target interaction detection.

The term “personalized therapy” refers to tailoring treatment to individuals. In this invention, personalized therapy is enabled by mapping patient-derived interactomes.

The term “biomarker panel” refers to a set of biomarkers for diagnosis. In this invention, biomarker panels are generated from Virotrap-identified complexes.

The term “clinical translation” refers to applying research to clinical practice. In this invention, clinical translation is supported by disease-relevant interaction mapping.

The term “precision diagnostics” refers to using molecular markers for diagnosis. In this invention, precision diagnostics are enabled by disease-specific complex detection.

The term “drug resistance profiling” refers to identifying resistance mechanisms. In this invention, drug resistance profiling is enabled by altered interaction networks.

The term “cancer interactome” refers to protein interactions altered in cancer. In this invention, the cancer interactome is mapped using Virotrap.

The term “neurodegenerative interactome” refers to interactions altered in neurodegenerative diseases. In this invention, the neurodegenerative interactome is mapped using Virotrap.

The term “infectious disease interactome” refers to interactions altered during infection. In this invention, the infectious disease interactome is mapped using Virotrap.

The term “immune interactome” refers to interactions involved in immune signaling. In this invention, the immune interactome is mapped using Virotrap.

The term “signaling network” refers to interconnected proteins transmitting signals. In this invention, signaling networks are mapped under native conditions.

The term “transcriptional network” refers to proteins regulating gene expression. In this invention, transcriptional networks are mapped using Virotrap.

The term “metabolic network” refers to proteins involved in metabolism. In this invention, metabolic networks are mapped using Virotrap.

The term “protein folding network” refers to chaperones and folding machinery. In this invention, protein folding networks are mapped using Virotrap.

The term “ubiquitin-proteasome network” refers to degradation machinery. In this invention, the ubiquitin-proteasome network is mapped using Virotrap.

The term “DNA repair network” refers to proteins maintaining genomic integrity. In this invention, the DNA repair network is mapped using Virotrap.

The term “cell cycle network” refers to proteins regulating division. In this invention, the cell cycle network is mapped using Virotrap.

The term “apoptosis network” refers to proteins regulating cell death. In this invention, the apoptosis network is mapped using Virotrap.

The term “autophagy network” refers to proteins regulating self-degradation. In this invention, the autophagy network is mapped using Virotrap.

The term “endocytosis network” refers to proteins internalizing extracellular material. In this invention, the endocytosis network is mapped using Virotrap.

The term “exocytosis network” refers to proteins secreting intracellular material. In this invention, the exocytosis network is mapped using Virotrap.

The term “vesicular trafficking network” refers to proteins transporting between compartments. In this invention, the vesicular trafficking network is mapped using Virotrap.

The term “membrane remodeling network” refers to proteins altering membrane structure. In this invention, the membrane remodeling network is mapped using Virotrap.

The term “cytoskeletal network” refers to structural proteins. In this invention, the cytoskeletal network is mapped using Virotrap.

The term “nuclear transport network” refers to proteins moving material between nucleus and cytoplasm. In this invention, the nuclear transport network is mapped using Virotrap.

The term “chromatin remodeling network” refers to proteins regulating DNA accessibility. In this invention, the chromatin remodeling network is mapped using Virotrap.

The term “RNA processing network” refers to proteins involved in RNA metabolism. In this invention, the RNA processing network is mapped using Virotrap.

The term “signal transduction cascade” refers to a series of molecular events transmitting a signal. In this invention, signal transduction cascades are mapped under native conditions.

The term “kinase-substrate network” refers to kinases and their substrates. In this invention, kinase-substrate networks are mapped using Virotrap.

The term “phosphatase-substrate network” refers to phosphatases and their substrates. In this invention, phosphatase-substrate networks are mapped using Virotrap.

The term “GTPase-effector network” refers to GTPases and their effectors. In this invention, GTPase-effector networks are mapped using Virotrap.

The term “ubiquitin ligase-substrate network” refers to ubiquitin ligases and their substrates. In this invention, ubiquitin ligase-substrate networks are mapped using Virotrap.

The term “SUMO ligase-substrate network” refers to SUMO ligases and their substrates. In this invention, SUMO ligase-substrate networks are mapped using Virotrap.

The term “acetyltransferase-substrate network” refers to acetyltransferases and their substrates. In this invention, acetyltransferase-substrate networks are mapped using Virotrap.

The term “methyltransferase-substrate network” refers to methyltransferases and their substrates. In this invention, methyltransferase-substrate networks are mapped using Virotrap.

The term “phosphorylation-dependent interaction network” refers to interactions regulated by phosphorylation. In this invention, phosphorylation-dependent interaction networks are mapped using Virotrap.

The term “ubiquitination-dependent interaction network” refers to interactions regulated by ubiquitination. In this invention, ubiquitination-dependent interaction networks are mapped using Virotrap.

The term “SUMOylation-dependent interaction network” refers to interactions regulated by SUMOylation. In this invention, SUMOylation-dependent interaction networks are mapped using Virotrap.

The term “acetylation-dependent interaction network” refers to interactions regulated by acetylation. In this invention, acetylation-dependent interaction networks are mapped using Virotrap.

The term “methylation-dependent interaction network” refers to interactions regulated by methylation. In this invention, methylation-dependent interaction networks are mapped using Virotrap.

The term “lipidation-dependent interaction network” refers to interactions regulated by lipid modification. In this invention, lipidation-dependent interaction networks are mapped using Virotrap.

The term “glycosylation-dependent interaction network” refers to interactions regulated by glycosylation. In this invention, glycosylation-dependent interaction networks are mapped using Virotrap.

The term “palmitoylation-dependent interaction network” refers to interactions regulated by palmitoylation. In this invention, palmitoylation-dependent interaction networks are mapped using Virotrap.

The term “myristoylation-dependent interaction network” refers to interactions regulated by myristoylation. In this invention, myristoylation-dependent interaction networks are mapped using Virotrap.

The term “farnesylation-dependent interaction network” refers to interactions regulated by farnesylation. In this invention, farnesylation-dependent interaction networks are mapped using Virotrap.

The term “geranylgeranylation-dependent interaction network” refers to interactions regulated by geranylgeranylation. In this invention, geranylgeranylation-dependent interaction networks are mapped using Virotrap.

The term “prey-bait ratio” refers to the relative expression levels of prey and bait proteins. In this invention, prey-bait ratios are optimized to ensure efficient capture without overexpression artifacts.

The term “expression level” refers to the amount of a protein produced in a cell. In this invention, expression levels are controlled by promoter strength and transfection efficiency.

The term “transfection ratio” refers to the relative amounts of plasmids used in transfection. In this invention, transfection ratios are standardized for reproducibility.

The term “DNA quantity” refers to the amount of plasmid DNA used in transfection. In this invention, DNA quantities are specified for each component.

The term “cell density” refers to the number of cells per unit area. In this invention, cell density is optimized for VLP production.

The term “transfection time” refers to the duration of DNA exposure to cells. In this invention, transfection time is 24 hours.

The term “harvest time” refers to the time at which supernatant is collected. In this invention, harvest time is 24–32 hours post-transfection.

The term “supernatant volume” refers to the amount of culture medium collected. In this invention, supernatant volume is scaled to cell number.

The term “bead volume” refers to the amount of magnetic beads used. In this invention, bead volume is standardized per sample.

The term “antibody amount” refers to the quantity of antibody used for bead loading. In this invention, antibody amount is optimized for saturation.

The term “binding time” refers to the duration of VLP-bead interaction. In this invention, binding time is 2 hours.

The term “washing steps” refer to the number of times beads are washed. In this invention, two washing steps are performed.

The term “elution buffer” refers to the solution used to release VLPs from beads. In this invention, elution buffer contains FLAG peptide.

The term “elution time” refers to the duration of elution. In this invention, elution time is 30 minutes.

The term “elution temperature” refers to the temperature during elution. In this invention, elution temperature is 37°C.

The term “lysate” refers to the cellular contents released by lysis. In this invention, lysates are prepared for western blotting but not for purification.

The term “control lysate” refers to a lysate from cells expressing GAG alone. In this invention, control lysates are used to confirm bait expression.

The term “experimental lysate” refers to a lysate from cells expressing GAG-bait and prey. In this invention, experimental lysates are used to confirm expression but not for purification.

The term “VLP lysate” refers to the contents of purified VLPs. In this invention, VLP lysates are used for mass spectrometry.

The term “negative control VLP” refers to VLPs produced without a prey. In this invention, negative control VLPs are used to identify background proteins.

The term “positive control VLP” refers to VLPs produced with a known bait-prey pair. In this invention, positive control VLPs are used to validate the system.

The term “bait-only VLP” refers to VLPs produced with GAG-bait but no prey. In this invention, bait-only VLPs are used to identify bait-specific binders.

The term “prey-only VLP” refers to VLPs produced with prey but no GAG-bait. In this invention, prey-only VLPs are not produced, as prey does not incorporate into VLPs without bait.

The term “empty VLP” refers to VLPs produced without any fusion protein. In this invention, empty VLPs are not produced, as GAG is required for assembly.

The term “tagged GAG” refers to GAG fused to an epitope tag. In this invention, tagged GAG is not used; instead, the envelope is tagged.

The term “untagged envelope” refers to envelope glycoprotein without a tag. In this invention, untagged envelope is co-expressed with tagged envelope to ensure proper trimerization.

The term “trimerization” refers to the assembly of three subunits into a stable complex. In this invention, VSV-G forms trimers on the VLP surface, enabling efficient capture.

The term “epitope accessibility” refers to the exposure of a tag on the surface of a particle. In this invention, epitope accessibility is ensured by the surface localization of VSV-G.

The term “particle yield” refers to the number of VLPs produced per cell. In this invention, particle yield is sufficient for mass spectrometry without concentration.

The term “particle purity” refers to the proportion of VLPs relative to contaminants. In this invention, particle purity is high due to the specificity of affinity capture.

The term “background contamination” refers to non-specific proteins co-purified with VLPs. In this invention, background contamination is minimized by the blacklist strategy.

The term “specific recovery” refers to the proportion of true interactors recovered relative to total proteins. In this invention, specific recovery is high due to the combination of encapsulation and affinity capture.

The term “interaction detection rate” refers to the percentage of known interactions detected. In this invention, the interaction detection rate is 30% for the human positive reference set.

The term “false discovery rate” refers to the proportion of false positives among detected interactions. In this invention, the false discovery rate is less than 5%.

The term “true positive rate” refers to the proportion of true interactions detected. In this invention, the true positive rate is 30% for the positive reference set.

The term “sensitivity” refers to the ability to detect true positives. In this invention, sensitivity is high due to the concentration effect of VLPs.

The term “specificity” refers to the ability to avoid false positives. In this invention, specificity is high due to the use of controls and filtering.

The term “accuracy” refers to the closeness of measured values to true values. In this invention, accuracy is high due to orthogonal validation.

The term “precision” refers to the reproducibility of measurements. In this invention, precision is high due to standardized protocols.

The term “reproducibility” refers to the consistency of results across experiments. In this invention, reproducibility is demonstrated across multiple baits and biological replicates.

The term “robustness” refers to the ability to perform under variable conditions. In this invention, robustness is demonstrated across cell lines and transfection methods.

The term “scalability” refers to the ability to increase sample size. In this invention, scalability is demonstrated by the use of 107 cells per experiment.

The term “portability” refers to the ability to transfer the method to other labs. In this invention, portability is ensured by the use of standard reagents.

The term “standardization” refers to the use of uniform procedures. In this invention, standardization is achieved through defined plasmid constructs and protocols.

The term “automation” refers to the use of machines to perform repetitive tasks. In this invention, automation is enabled by magnetic beads and standardized elution.

The term “cost-effectiveness” refers to the balance between performance and cost. In this invention, cost-effectiveness is high due to the elimination of ultracentrifugation.

The term “time-efficiency” refers to the reduction in experimental time. In this invention, time-efficiency is high due to the single-step capture.

The term “user-friendliness” refers to the ease of use. In this invention, user-friendliness is enhanced by the simplicity of the protocol.

The term “versatility” refers to the ability to adapt to different applications. In this invention, versatility is demonstrated by the capture of binary, multiprotein, and small molecule interactions.

The term “innovation” refers to the introduction of a new concept. In this invention, innovation lies in the use of VLPs as native-state interaction traps.

The term “paradigm shift” refers to a fundamental change in methodology. In this invention, the paradigm shift is from lysis to encapsulation.

The term “translational potential” refers to the ability to move from bench to bedside. In this invention, translational potential is high due to applications in drug discovery.

The term “commercial viability” refers to the economic potential of the invention. In this invention, commercial viability is high due to the platform nature of the technology.

The term “intellectual property” refers to legally protected inventions. In this invention, the method, system, and kit constitute valuable intellectual property.

The term “patent landscape” refers to the collection of existing patents. In this invention, the patent landscape is sparse, and this invention occupies a novel space.

The term “freedom to operate” refers to the ability to commercialize without infringement. In this invention, freedom to operate is established by the novelty of the combination.

The term “regulatory compliance” refers to adherence to legal and ethical standards. In this invention, regulatory compliance is ensured by the use of non-infectious VLPs.

The term “ethical use” refers to responsible application. In this invention, ethical use is ensured by the use of immortalized cell lines.

The term “sustainability” refers to environmental impact. In this invention, sustainability is enhanced by reduced reagent use.

The term “global applicability” refers to worldwide usability. In this invention, global applicability is ensured by the use of universally available reagents.

The term “open science” refers to public sharing of data and methods. In this invention, open science is practiced through PRIDE deposition.

The term “data sharing” refers to public release of data. In this invention, data sharing is implemented through PRIDE.

The term “reagent sharing” refers to distribution of plasmids. In this invention, reagent sharing is facilitated by Addgene.

The term “collaborative potential” refers to the ability to foster collaboration. In this invention, collaborative potential is high due to the platform nature.

The term “educational value” refers to utility in teaching. In this invention, educational value is high due to the clear mechanism.

The term “technical documentation” refers to written instructions. In this invention, technical documentation is provided in the form of protocols and plasmid maps.

The term “reagent kit” refers to a packaged set of materials. In this invention, a reagent kit is provided comprising plasmids, antibodies, beads, and instructions.

The term “service offering” refers to the provision of the method as a service. In this invention, service offerings include screening and interactome mapping.

The term “diagnostic application” refers to disease detection. In this invention, diagnostic applications include the identification of disease-specific complexes.

The term “therapeutic application” refers to drug development. In this invention, therapeutic applications include target identification and off-target profiling.

The term “biopharmaceutical development” refers to the development of biological drugs. In this invention, biopharmaceutical development is supported by target identification.

The term “target validation” refers to confirming a protein is a viable drug target. In this invention, target validation is achieved by demonstrating interaction in a native context.

The term “lead optimization” refers to improving a drug candidate. In this invention, lead optimization is informed by off-target interaction profiles.

The term “pharmacokinetic profiling” refers to studying drug absorption and metabolism. In this invention, pharmacokinetic profiling is informed by drug-binding protein identification.

The term “toxicology screening” refers to assessing drug toxicity. In this invention, toxicology screening is enhanced by off-target interaction detection.

The term “personalized therapy” refers to tailoring treatment to individuals. In this invention, personalized therapy is enabled by mapping patient-derived interactomes.

The term “biomarker panel” refers to a set of biomarkers for diagnosis. In this invention, biomarker panels are generated from Virotrap-identified complexes.

The term “clinical translation” refers to applying research to clinical practice. In this invention, clinical translation is supported by disease-relevant interaction mapping.

The term “precision diagnostics” refers to using molecular markers for diagnosis. In this invention, precision diagnostics are enabled by disease-specific complex detection.

The term “drug resistance profiling” refers to identifying resistance mechanisms. In this invention, drug resistance profiling is enabled by altered interaction networks.

The term “cancer interactome” refers to protein interactions altered in cancer. In this invention, the cancer interactome is mapped using Virotrap.

The term “neurodegenerative interactome” refers to interactions altered in neurodegenerative diseases. In this invention, the neurodegenerative interactome is mapped using Virotrap.

The term “infectious disease interactome” refers to interactions altered during infection. In this invention, the infectious disease interactome is mapped using Virotrap.

The term “immune interactome” refers to interactions involved in immune signaling. In this invention, the immune interactome is mapped using Virotrap.

The term “signaling network” refers to interconnected proteins transmitting signals. In this invention, signaling networks are mapped under native conditions.

The term “transcriptional network” refers to proteins regulating gene expression. In this invention, transcriptional networks are mapped using Virotrap.

The term “metabolic network” refers to proteins involved in metabolism. In this invention, metabolic networks are mapped using Virotrap.

The term “protein folding network” refers to chaperones and folding machinery. In this invention, protein folding networks are mapped using Virotrap.

The term “ubiquitin-proteasome network” refers to degradation machinery. In this invention, the ubiquitin-proteasome network is mapped using Virotrap.

The term “DNA repair network” refers to proteins maintaining genomic integrity. In this invention, the DNA repair network is mapped using Virotrap.

The term “cell cycle network” refers to proteins regulating division. In this invention, the cell cycle network is mapped using Virotrap.

The term “apoptosis network” refers to proteins regulating cell death. In this invention, the apoptosis network is mapped using Virotrap.

The term “autophagy network” refers to proteins regulating self-degradation. In this invention, the autophagy network is mapped using Virotrap.

The term “endocytosis network” refers to proteins internalizing extracellular material. In this invention, the endocytosis network is mapped using Virotrap.

The term “exocytosis network” refers to proteins secreting intracellular material. In this invention, the exocytosis network is mapped using Virotrap.

The term “vesicular trafficking network” refers to proteins transporting between compartments. In this invention, the vesicular trafficking network is mapped using Virotrap.

The term “membrane remodeling network” refers to proteins altering membrane structure. In this invention, the membrane remodeling network is mapped using Virotrap.

The term “cytoskeletal network” refers to structural proteins. In this invention, the cytoskeletal network is mapped using Virotrap.

The term “nuclear transport network” refers to proteins moving material between nucleus and cytoplasm. In this invention, the nuclear transport network is mapped using Virotrap.

The term “chromatin remodeling network” refers to proteins regulating DNA accessibility. In this invention, the chromatin remodeling network is mapped using Virotrap.

The term “RNA processing network” refers to proteins involved in RNA metabolism. In this invention, the RNA processing network is mapped using Virotrap.

The term “signal transduction cascade” refers to a series of molecular events transmitting a signal. In this invention, signal transduction cascades are mapped under native conditions.

The term “kinase-substrate network” refers to kinases and their substrates. In this invention, kinase-substrate networks are mapped using Virotrap.

The term “phosphatase-substrate network” refers to phosphatases and their substrates. In this invention, phosphatase-substrate networks are mapped using Virotrap.

The term “GTPase-effector network” refers to GTPases and their effectors. In this invention, GTPase-effector networks are mapped using Virotrap.

The term “ubiquitin ligase-substrate network” refers to ubiquitin ligases and their substrates. In this invention, ubiquitin ligase-substrate networks are mapped using Virotrap.

The term “SUMO ligase-substrate network” refers to SUMO ligases and their substrates. In this invention, SUMO ligase-substrate networks are mapped using Virotrap.

The term “acetyltransferase-substrate network” refers to acetyltransferases and their substrates. In this invention, acetyltransferase-substrate networks are mapped using Virotrap.

The term “methyltransferase-substrate network” refers to methyltransferases and their substrates. In this invention, methyltransferase-substrate networks are mapped using Virotrap.

The term “phosphorylation-dependent interaction network” refers to interactions regulated by phosphorylation. In this invention, phosphorylation-dependent interaction networks are mapped using Virotrap.

The term “ubiquitination-dependent interaction network” refers to interactions regulated by ubiquitination. In this invention, ubiquitination-dependent interaction networks are mapped using Virotrap.

The term “SUMOylation-dependent interaction network” refers to interactions regulated by SUMOylation. In this invention, SUMOylation-dependent interaction networks are mapped using Virotrap.

The term “acetylation-dependent interaction network” refers to interactions regulated by acetylation. In this invention, acetylation-dependent interaction networks are mapped using Virotrap.

The term “methylation-dependent interaction network” refers to interactions regulated by methylation. In this invention, methylation-dependent interaction networks are mapped using Virotrap.

The term “lipidation-dependent interaction network” refers to interactions regulated by lipid modification. In this invention, lipidation-dependent interaction networks are mapped using Virotrap.

The term “glycosylation-dependent interaction network” refers to interactions regulated by glycosylation. In this invention, glycosylation-dependent interaction networks are mapped using Virotrap.

The term “palmitoylation-dependent interaction network” refers to interactions regulated by palmitoylation. In this invention, palmitoylation-dependent interaction networks are mapped using Virotrap.

The term “myristoylation-dependent interaction network” refers to interactions regulated by myristoylation. In this invention, myristoylation-dependent interaction networks are mapped using Virotrap.

The term “farnesylation-dependent interaction network” refers to interactions regulated by farnesylation. In this invention, farnesylation-dependent interaction networks are mapped using Virotrap.

The term “geranylgeranylation-dependent interaction network” refers to interactions regulated by geranylgeranylation. In this invention, geranylgeranylation-dependent interaction networks are mapped using Virotrap.

The term “prey-bait ratio” refers to the relative expression levels of prey and bait proteins. In this invention, prey-bait ratios are optimized to ensure efficient capture without overexpression artifacts.

The term “expression level” refers to the amount of a protein produced in a cell. In this invention, expression levels are controlled by promoter strength and transfection efficiency.

The term “transfection ratio” refers to the relative amounts of plasmids used in transfection. In this invention, transfection ratios are standardized for reproducibility.

The term “DNA quantity” refers to the amount of plasmid DNA used in transfection. In this invention, DNA quantities are specified for each component.

The term “cell density” refers to the number of cells per unit area. In this invention, cell density is optimized for VLP production.

The term “transfection time” refers to the duration of DNA exposure to cells. In this invention, transfection time is 24 hours.

The term “harvest time” refers to the time at which supernatant is collected. In this invention, harvest time is 24–32 hours post-transfection.

The term “supernatant volume” refers to the amount of culture medium collected. In this invention, supernatant volume is scaled to cell number.

The term “bead volume” refers to the amount of magnetic beads used. In this invention, bead volume is standardized per sample.

The term “antibody amount” refers to the quantity of antibody used for bead loading. In this invention, antibody amount is optimized for saturation.

The term “binding time” refers to the duration of VLP-bead interaction. In this invention, binding time is 2 hours.

The term “washing steps” refer to the number of times beads are washed. In this invention, two washing steps are performed.

The term “elution buffer” refers to the solution used to release VLPs from beads. In this invention, elution buffer contains FLAG peptide.

The term “elution time” refers to the duration of elution. In this invention, elution time is 30 minutes.

The term “elution temperature” refers to the temperature during elution. In this invention, elution temperature is 37°C.

The term “lysate” refers to the cellular contents released by lysis. In this invention, lysates are prepared for western blotting but not for purification.

The term “control lysate” refers to a lysate from cells expressing GAG alone. In this invention, control lysates are used to confirm bait expression.

The term “experimental lysate” refers to a lysate from cells expressing GAG-bait and prey. In this invention, experimental lysates are used to confirm expression but not for purification.

The term “VLP lysate” refers to the contents of purified VLPs. In this invention, VLP lysates are used for mass spectrometry.

The term “negative control VLP” refers to VLPs produced without a prey. In this invention, negative control VLPs are used to identify background proteins.

The term “positive control VLP” refers to VLPs produced with a known bait-prey pair. In this invention, positive control VLPs are used to validate the system.

The term “bait-only VLP” refers to VLPs produced with GAG-bait but no prey. In this invention, bait-only VLPs are used to identify bait-specific binders.

The term “prey-only VLP” refers to VLPs produced with prey but no GAG-bait. In this invention, prey-only VLPs are not produced, as prey does not incorporate into VLPs without bait.

The term “empty VLP” refers to VLPs produced without any fusion protein. In this invention, empty VLPs are not produced, as GAG is required for assembly.

The term “tagged GAG” refers to GAG fused to an epitope tag. In this invention, tagged GAG is not used; instead, the envelope is tagged.

The term “untagged envelope” refers to envelope glycoprotein without a tag. In this invention, untagged envelope is co-expressed with tagged envelope to ensure proper trimerization.

The term “trimerization” refers to the assembly of three subunits into a stable complex. In this invention, VSV-G forms trimers on the VLP surface, enabling efficient capture.

The term “epitope accessibility” refers to the exposure of a tag on the surface of a particle. In this invention, epitope accessibility is ensured by the surface localization of VSV-G.

The term “particle yield” refers to the number of VLPs produced per cell. In this invention, particle yield is sufficient for mass spectrometry without concentration.

The term “particle purity” refers to the proportion of VLPs relative to contaminants. In this invention, particle purity is high due to the specificity of affinity capture.

The term “background contamination” refers to non-specific proteins co-purified with VLPs. In this invention, background contamination is minimized by the blacklist strategy.

The term “specific recovery” refers to the proportion of true interactors recovered relative to total proteins. In this invention, specific recovery is high due to the combination of encapsulation and affinity capture.

The term “interaction detection rate” refers to the percentage of known interactions detected. In this invention, the interaction detection rate is 30% for the human positive reference set.

The term “false discovery rate” refers to the proportion of false positives among detected interactions. In this invention, the false discovery rate is less than 5%.

The term “true positive rate” refers to the proportion of true interactions detected. In this invention, the true positive rate is 30%