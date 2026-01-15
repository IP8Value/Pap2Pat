# DESCRIPTION

The type III secretion system (T3SS) is a highly specialized protein export apparatus employed by numerous Gram-negative bacterial pathogens to directly inject effector proteins into the cytosol of eukaryotic host cells. This molecular syringe-like nanomachine enables bacteria to subvert host cell signaling pathways, manipulate immune responses, and establish infection. Unlike conventional secretion systems that release proteins extracellularly, the T3SS functions as a contact-dependent translocon, forming a continuous channel from the bacterial cytoplasm through the host plasma membrane. The structural core of this system is the needle complex—a supramolecular assembly embedded in both bacterial membranes and extending outward as a hollow needle filament. Given its central role in virulence across multiple pathogenic species, the T3SS represents a compelling target for novel anti-infective strategies.

Conventional antibiotic therapies face mounting challenges due to the rapid emergence of multidrug-resistant bacterial strains, rendering many first-line treatments ineffective. Moreover, broad-spectrum antibiotics indiscriminately disrupt commensal microbiota, leading to secondary complications such as opportunistic infections and dysbiosis. Critically, traditional antibiotics exert strong selective pressure that accelerates resistance development. In contrast, antivirulence approaches that disarm pathogens without killing them—such as inhibition of the T3SS—offer a promising alternative by reducing pathogenicity while minimizing ecological disruption and resistance selection.

This context motivates the targeting of the T3SS as a therapeutic strategy. By blocking the delivery of effector proteins, T3SS inhibitors can neutralize bacterial virulence without affecting bacterial viability, thereby preserving the host microbiome and reducing the evolutionary incentive for resistance. Such inhibitors could serve as standalone therapeutics or as adjuncts to conventional antibiotics, enhancing treatment efficacy while mitigating resistance risks.

Type III secretion systems are evolutionarily conserved among diverse Gram-negative pathogens and share a common architectural blueprint. The system comprises a basal body spanning the inner and outer bacterial membranes, a hollow needle protruding from the bacterial surface, and a translocon pore inserted into the host cell membrane. Assembly occurs in a hierarchical, stepwise manner: first, the basal body forms; then, the needle polymerizes; finally, upon host cell contact, translocon proteins are secreted to complete the conduit into the host cytosol.

Numerous clinically significant pathogens rely on T3SS for infection, including *Salmonella enterica* serovar Typhimurium, *Shigella flexneri*, *Yersinia pestis*, enteropathogenic *Escherichia coli* (EPEC), enterohemorrhagic *E. coli* (EHEC), and *Vibrio cholerae*. Each employs its T3SS to deliver a unique repertoire of effectors that hijack host cellular processes—ranging from actin cytoskeleton remodeling to suppression of inflammatory responses.

Secreted effectors are bacterial proteins translocated directly into host cells via the T3SS. These molecules function as molecular mimics or enzymatic modulators that interfere with host signaling cascades. For example, *Salmonella* SptP acts as a GTPase-activating protein to reverse cytoskeletal rearrangements induced during invasion, while SipB forms part of the translocon and triggers pyroptosis in macrophages.

The needle complex is the structural cornerstone of the T3SS, composed of multiple protein subunits organized into concentric rings and a central channel. In *S. typhimurium* SPI-1 T3SS, the base includes PrgH, PrgK, and InvG, while the needle filament is formed by PrgI and the inner rod by PrgJ. Cryo-electron microscopy has revealed a cylindrical architecture approximately 25 nm wide and 50–80 nm long, featuring inner and outer rings connected by a neck region.

Assembly of the needle complex proceeds through defined intermediates. Initial formation of the inner membrane ring (composed of PrgH and PrgK) is followed by recruitment of the outer membrane secretin InvG. Subsequent polymerization of PrgI forms the needle, and only after full assembly does the system switch substrate specificity to secrete effectors rather than structural components.

Chaperones are small, acidic proteins that bind newly synthesized effectors or translocators in the bacterial cytoplasm, preventing premature interactions or degradation. They facilitate efficient targeting to the T3SS export apparatus and maintain substrates in a secretion-competent state.

Class I chaperones specifically bind effector proteins and typically form homodimers that cradle a single effector polypeptide. Examples include SicP for SptP in *Salmonella* and CesT for Tir in EPEC. These chaperones recognize N-terminal secretion signals and often mask hydrophobic regions of their cognate effectors.

Removal of chaperones occurs at the base of the T3SS, where ATPase activity and unfolding machinery dissociate the chaperone-effector complex, allowing the effector to be threaded through the secretion channel. This step is essential for productive translocation and is tightly coupled to the energization of the export process.

Prior art methods for studying T3SS function have relied heavily on genetic knockouts, secretion assays via Western blotting, and structural analyses using electron microscopy. However, these approaches are low-throughput, labor-intensive, and poorly suited for drug screening.

WO2009/145829 discloses methods for identifying T3SS inhibitors using reporter gene fusions linked to T3SS promoters, but this indirect approach measures transcriptional activation rather than actual secretion or assembly.

WO2009/137133 describes assays based on β-lactamase fusion proteins to detect translocation into host cells, which requires eukaryotic co-culture and is not amenable to high-throughput screening of compound libraries.

WO2009/061491 proposes fluorescence-based assays for monitoring T3SS activity but lacks specificity for individual structural or secretory steps and suffers from background interference.

Gauthier et al. (2005) introduced a secretion ELISA for *Yersinia* Yop proteins but did not address chaperone-bound intermediates or needle complex assembly, limiting its utility for mechanistic inhibitor profiling.

The object of the present invention is to provide robust, quantitative, high-throughput compatible methods for identifying and characterizing inhibitors of the T3SS by directly measuring key functional and structural events: effector-chaperone interactions, protein secretion, and needle complex assembly.

The novel method comprises two principal steps: (1) detection of effector proteins either free or bound to their cognate chaperones in bacterial culture supernatants or lysates, and (2) assessment of structural integrity and assembly of the needle complex.

The first step involves capturing a specific component—either a chaperone or an effector—onto a solid support and detecting the presence of its binding partner using labeled antibodies. This enables quantification of secretion competence and chaperone-effector dynamics.

Detection methods include enzyme-linked immunosorbent assays (ELISA), AlphaScreen, DELFIA, and FRET-based formats, all adapted to T3SS-specific targets.

ELISA is a widely used immunoassay that relies on antigen-antibody binding for detection. In the context of T3SS, it can be configured to measure secreted effectors or chaperone complexes.

A conventional ELISA protocol involves immobilizing a capture reagent (e.g., antibody or purified protein) onto a microtiter plate, blocking non-specific sites, incubating with sample, washing away unbound material, adding a detection antibody conjugated to an enzyme, and measuring signal via colorimetric, chemiluminescent, or fluorescent readout.

Immobilization of the capture reagent is typically achieved by passive adsorption to polystyrene surfaces or covalent coupling via amine-reactive chemistry. Alternatively, affinity tags (e.g., His-tag/Ni-NTA) enable oriented immobilization.

Blocking agents such as bovine serum albumin (BSA), casein, or non-fat dry milk are used to occupy remaining protein-binding sites on the plate surface, minimizing non-specific binding during subsequent steps.

Incubation of the sample—comprising bacterial culture supernatant, lysate, or purified fractions—allows the target analyte to bind the immobilized capture reagent under controlled conditions of time, temperature, and buffer composition.

Washing removes unbound proteins and contaminants, followed by addition of a labeled detection antibody that binds a distinct epitope on the captured analyte, enabling specific signal generation.

Labels include enzymes (horseradish peroxidase, alkaline phosphatase), fluorophores, or chemiluminescent moieties. Detection methods range from absorbance spectroscopy to luminescence or fluorescence plate readers.

Conjugation of labels to antibodies is performed using standard bioconjugation techniques, such as NHS-ester chemistry for amine coupling or maleimide-thiol reactions for site-specific labeling.

Measurement of bound antibody is proportional to the amount of target present in the sample, allowing quantitative assessment of secretion levels or complex formation.

The second step of the method evaluates the structural assembly of the needle complex by detecting interactions between core components such as PrgH, PrgK, InvG, and PrgI.

A preferred embodiment of the first step is a secretion ELISA designed to detect effector proteins—such as SptP—while still bound to their class I chaperone SicP, reflecting a pre-secretion intermediate state.

Detection of effector bound to chaperone provides insight into whether a test compound blocks early recognition, chaperone loading, or later export steps.

An ELISA-type assay is configured by immobilizing purified SicP onto the plate, incubating with culture supernatant or lysate, and detecting bound SptP using anti-SptP antibodies.

Immobilization of chaperone ensures specific capture of chaperone-effector complexes, distinguishing them from free effectors or degraded fragments.

Detection of effector confirms the presence and stability of the complex, with signal intensity correlating to functional T3SS activity.

A secretion ELISA variant measures total secreted effectors in culture supernatants after induction of T3SS expression, providing a direct readout of export competence.

Advantages of the secretion ELISA include high sensitivity, compatibility with 96- or 384-well formats, no requirement for host cells, and applicability to diverse bacterial species.

The method is broadly applicable to other T3SS-encoding bacteria by substituting species-specific effectors, chaperones, and structural proteins. For instance, YopE/SycE in *Yersinia*, EspB/CesD in EPEC, or ExoS/PcrH in *Pseudomonas*.

Examples of effector proteins include SptP, SipA, SipB, SipC, and SopE in *Salmonella*; YopH, YopE, YopM in *Yersinia*; Tir, EspF, Map in EPEC; and VopS, VopQ in *Vibrio*.

The type III protein secretion system, as described, is a multi-component nanomachine requiring precise coordination of structural and regulatory elements.

Effector and chaperone proteins function as a unit: chaperones stabilize effectors and present them to the export gate, ensuring ordered secretion.

Examples of effector-chaperone pairs include SptP/SicP, SseF/SscA, YopE/SycE, Tir/CesT, and ExoS/PcrH.

The method for determining chaperone existence or engagement involves co-immunoprecipitation or sandwich ELISA using matched antibody pairs or tagged components.

Activating the T3SS is essential for meaningful assay readouts, as basal expression is often low. Induction can be achieved by environmental cues (e.g., low oxygen, high osmolarity) or genetic overexpression of master regulators like HilA.

Methods for activating T3SS include growth in SPI-1-inducing media (e.g., LB + 0.3M NaCl, pH 7.0, standing culture), or use of arabinose-inducible *hilA* plasmids.

A high-throughput screening method is enabled by adapting the ELISA format to automated liquid handling and plate reading, facilitating rapid evaluation of thousands of compounds.

Alpha technology (Amplified Luminescent Proximity Homogeneous Assay) is a bead-based proximity assay that generates signal only when donor and acceptor beads are brought into close proximity (<200 nm) by a biomolecular interaction.

The principle of AlphaScreen relies on singlet oxygen diffusion: laser excitation of donor beads produces singlet oxygen, which triggers chemiluminescence in nearby acceptor beads if a binding event bridges them.

A sandwich format assay uses one bead coated with capture molecule (e.g., anti-chaperone antibody) and another with detection molecule (e.g., anti-effector antibody), producing signal only when both bind the same complex.

Direct detection uses labeled proteins, while indirect detection employs primary and secondary antibodies, offering flexibility in assay design.

Alternative assay types include DELFIA (Dissociation-Enhanced Lanthanide Fluoroimmunoassay), which uses time-resolved fluorescence of europium or terbium chelates for ultra-sensitive detection with minimal background.

DELFIA assay involves capture on solid phase, incubation with sample, addition of lanthanide-labeled detection antibody, and enhancement solution that dissociates and stabilizes the fluorescent lanthanide ion for measurement.

A method for monitoring assembly of structural components assesses interactions between needle complex proteins such as PrgH-PrgK, PrgH-InvG, or PrgH-PrgI.

A structure assay, such as a structure ELISA, immobilizes one structural component (e.g., His-tagged PrgH) and detects binding of another (e.g., PrgI) using specific antibodies.

The principle of structure ELISA is analogous to secretion ELISA but targets protein-protein interactions within the assembled basal body or needle.

An example of structure ELISA involves capturing poly-histidine-tagged PrgH on Ni-NTA plates and detecting attached PrgI needle subunits using anti-PrgI antibodies, thereby reporting on needle-base connectivity.

Alternative tags for structural components include FLAG, HA, Strep-tag, or GST, enabling flexible immobilization strategies depending on expression system and solubility.

Analysis of a test compound’s effect on ATPase activity—critical for substrate unfolding and export—can be performed using malachite green phosphate detection or radiolabeled ATP hydrolysis assays, though this is ancillary to the primary structural and secretion readouts.

The structure assay is applicable to other bacteria by substituting orthologous proteins: e.g., EscJ/EscC in EPEC, MxiJ/MxiD in *Shigella*, or YscJ/YscC in *Yersinia*.

Identification of inhibitors of T3SS is achieved by comparing signal intensity in compound-treated versus untreated samples: reduced effector secretion or disrupted structural interactions indicate inhibitory activity.

A method for testing inhibitors against other bacteria involves expressing heterologous T3SS components in surrogate hosts or using native pathogens under biosafety-appropriate conditions.

Modeling of test compounds into structural components uses available crystal structures (e.g., PrgH C-terminal domain, EscJ) to predict binding pockets at protein-protein interfaces critical for assembly.

De novo design of inhibitors leverages structural insights to create small molecules or peptides that disrupt key interactions, such as PrgH-InvG or PrgH-PrgK binding.

Computer-based design employs molecular docking, virtual screening, and molecular dynamics simulations using software such as AutoDock, Rosetta, or Schrödinger Suite.

X-ray crystallography provides high-resolution atomic structures of target domains or complexes, enabling structure-guided inhibitor optimization.

Computer programs for designing inhibitors include MOE, Glide, GOLD, and CHARMM, which evaluate binding affinity, pharmacophore fit, and ADMET properties.

Crystallization methods involve vapor diffusion, microbatch, or lipidic cubic phase techniques to obtain diffraction-quality crystals of target proteins or complexes.

Evaluation of candidate compounds includes dose-response curves, IC50 determination, counter-screens against unrelated secretion systems, and cytotoxicity assays.

Toxicity testing in animal disease models assesses systemic safety and tolerability of lead compounds prior to efficacy studies.

Pharmacokinetics and toxicity testing determine absorption, distribution, metabolism, excretion, and maximum tolerated dose in rodents.

Animal infection models recapitulate human disease and allow evaluation of therapeutic efficacy in vivo.

Testing in EPEC/EHEC animal models (e.g., rabbit ileal loop or mouse colonization models) assesses impact on attaching/effacing lesion formation and bacterial shedding.

Testing in *Salmonella typhimurium* animal models evaluates reduction in intestinal invasion, systemic spread, and inflammation.

The murine typhoid model—intraperitoneal or oral infection of susceptible mice with *S. typhimurium*—mimics human typhoid fever and is ideal for assessing systemic T3SS inhibition.

Testing compounds in the murine typhoid model involves pretreatment or co-administration with bacteria, followed by monitoring of survival, weight loss, and bacterial loads.

Organ colonization rates in spleen, liver, and mesenteric lymph nodes serve as quantitative endpoints for efficacy, with reduced CFUs indicating successful T3SS inhibition.

Compounds identified by this method hold significant potential as antibacterial therapeutics that attenuate virulence without promoting resistance, offering a sustainable alternative to traditional antibiotics.

In conclusion, the described method provides a comprehensive, modular platform for identifying and characterizing T3SS inhibitors through direct measurement of secretion and structural assembly, enabling rational development of next-generation antivirulence drugs.

## EXAMPLE 1

### Secretion ELISA for Detecting the Effector Protein SptP when Bound to its Cognate Chaperone SicP

To implement the secretion ELISA for detecting SptP bound to SicP, recombinant glutathione S-transferase (GST)-tagged SicP was expressed in *Escherichia coli* and purified using glutathione-Sepharose affinity chromatography. The purified GST-SicP was then used to precoat high-binding polystyrene microtiter plates by overnight incubation at 4°C in carbonate-bicarbonate buffer (pH 9.6). Following coating, plates were blocked with 3% bovine serum albumin (BSA) in phosphate-buffered saline (PBS) for one hour at room temperature to prevent non-specific binding. Culture supernatants from *Salmonella typhimurium* strains grown under SPI-1-inducing conditions were clarified by centrifugation and filtration, then added to the coated wells and incubated for two hours at 37°C. After extensive washing with PBS containing 0.05% Tween-20, bound SptP was detected using a rabbit polyclonal anti-SptP primary antibody, followed by a horseradish peroxidase (HRP)-conjugated goat anti-rabbit secondary antibody. The enzymatic reaction was initiated by adding tetramethylbenzidine (TMB) substrate, and the reaction kinetics were monitored spectrophotometrically at 650 nm. The reaction was stopped after a defined interval by addition of 2N sulfuric acid, and the final absorbance was measured at 450 nm. Signal intensity correlated directly with the amount of SptP-SicP complex present in the supernatant, providing a quantitative readout of T3SS-mediated effector-chaperone secretion competence.

## EXAMPLE 2

### Structure ELISA to Detect the Attachment of PrgI to the PrgH-Containing Base Element

For the structure ELISA assessing PrgI attachment to the base, PrgH was engineered with a C-terminal poly-histidine tag and expressed in a *prgH* deletion strain of *Salmonella typhimurium* under its native promoter. The protein was purified from detergent-solubilized membranes using Ni-NTA affinity chromatography. Purified His-PrgH was immobilized onto Ni-NTA-coated microtiter plates by incubation in binding buffer (20 mM Tris-HCl pH 7.5, 300 mM NaCl, 0.05% Tween-20) for one hour at room temperature. Plates were then blocked with 5% non-fat dry milk in PBS. Intact needle complexes, purified from wild-type or test-compound-treated bacteria, were added to the wells and allowed to bind via the His-PrgH anchor. After washing, the presence of attached PrgI needle subunits was detected using a mouse monoclonal anti-PrgI antibody, followed by an HRP-conjugated anti-mouse secondary antibody. TMB substrate was added, and the colorimetric reaction was allowed to proceed for a fixed duration before being stopped with sulfuric acid. Absorbance was measured at 450 nm, with higher values indicating intact PrgH-PrgI interactions and thus preserved needle complex assembly. Disruption of this interaction by test compounds resulted in significantly reduced signal, identifying structural inhibitors.

## EXAMPLE 3

### Immunodetection of the Translocators SipB and SipC (Western Blot)

To validate secretion phenotypes, *Salmonella typhimurium* cultures were grown under SPI-1-inducing conditions to mid-log phase. Bacterial cells were removed by centrifugation at 10,000 × g for 10 minutes, and the resulting supernatant was filtered through a 0.22 µm membrane. Proteins in the supernatant were precipitated by addition of trichloroacetic acid (TCA) to a final concentration of 10%, incubated on ice for 30 minutes, and pelleted by centrifugation at 16,000 × g for 15 minutes. The protein pellet was washed twice with cold acetone to remove residual TCA, air-dried, and resuspended in SDS-PAGE sample buffer containing β-mercaptoethanol. Samples were heated at 95°C for 5 minutes, then loaded onto a 12% polyacrylamide gel for electrophoretic separation. Proteins were transferred to a polyvinylidene difluoride (PVDF) membrane using semi-dry electroblotting. The membrane was blocked with 5% non-fat milk in Tris-buffered saline with Tween-20 (TBST), then probed overnight at 4°C with primary antibodies against SipB and SipC. After washing, HRP-conjugated secondary antibodies were applied for one hour at room temperature. Protein bands were visualized by chemiluminescence using an ECL reagent, and the signal was captured on X-ray film or a digital imager. The presence and intensity of SipB and SipC bands confirmed functional T3SS secretion, while absence or reduction indicated inhibitory effects of test compounds.