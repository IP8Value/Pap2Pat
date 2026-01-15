## FIELD OF THE INVENTION

- define genetic engineering field

The present invention resides in the field of genetic engineering, specifically in the design and implementation of synthetic biological systems for the discovery, production, and optimization of biologically active small molecules that modulate enzyme function. This field encompasses the deliberate manipulation of genetic circuits, metabolic pathways, and regulatory elements within microbial hosts to achieve programmable biochemical outcomes. The invention integrates principles of synthetic biology, enzyme kinetics, optogenetics, and high-throughput screening to construct genetically encoded systems that couple cellular viability to the presence or absence of specific enzymatic activities. These systems are engineered to detect intracellular modulation of target enzymes through transcriptional readouts, luminescent or fluorescent signals, or growth-dependent phenotypes, thereby enabling the identification of novel inhibitors or activators without prior knowledge of their chemical structure. The invention further extends to the directed evolution of enzymes and their substrates through iterative cycles of mutagenesis and selection, allowing for the optimization of binding affinity, selectivity, and functional dynamics under controlled environmental conditions. By embedding complex biochemical objectives into the genome of a living cell, this approach transforms microbial systems into autonomous biosensors and bio-factories capable of navigating vast chemical spaces to uncover molecules with therapeutic potential.

## BACKGROUND

- motivate phosphatase control

Protein tyrosine phosphatases represent a critical class of regulatory enzymes whose activity is tightly coupled to the control of intracellular signaling networks governing cell growth, metabolism, immune response, and differentiation. Dysregulation of these enzymes has been implicated in a broad spectrum of human diseases, including type 2 diabetes, obesity, autoimmune disorders, and multiple forms of cancer. Despite their therapeutic relevance, the development of selective, cell-permeable inhibitors of protein tyrosine phosphatases has proven exceptionally challenging due to the highly conserved, positively charged architecture of their catalytic sites, which favor nonselective interactions with polar or charged molecules. Traditional drug discovery approaches relying on high-throughput screening of synthetic compound libraries have yielded few viable candidates, largely because such molecules often lack the necessary membrane permeability or fail to distinguish between closely related phosphatase isoforms. Furthermore, the dynamic and spatially regulated nature of phosphatase activity—particularly in subcellular compartments such as the endoplasmic reticulum or plasma membrane—demands inhibitors capable of modulating enzyme function with precision in both time and space. Conventional pharmacological tools are ill-suited to address these requirements, as they typically act globally and irreversibly, disrupting physiological feedback loops and inducing off-target effects. The absence of effective modulators has thus hindered both the validation of phosphatases as drug targets and the mechanistic elucidation of their roles in disease. There exists, therefore, a compelling need for novel strategies that enable the discovery of structurally diverse, selective, and functionally tunable inhibitors capable of engaging phosphatases through non-canonical binding modes, including allosteric pockets, and that can be produced in a scalable, biosynthetic manner within engineered microbial systems.

## SUMMARY OF THE INVENTION

- relate to genetic engineering

This invention pertains to a novel framework in genetic engineering that enables the systematic discovery, evolution, and production of biologically active small molecules through the integration of genetically encoded detection systems with biosynthetic metabolic pathways. The system is designed to link the functional inhibition of a target enzyme to the survival or detectable output of a microbial host, thereby transforming the cell into an autonomous biosensor capable of identifying molecular solutions to a predefined biochemical objective. This approach circumvents the limitations of traditional screening methods by embedding the criteria for molecular efficacy—such as potency, selectivity, and intracellular bioavailability—directly into the genetic circuitry of the organism. The invention leverages the natural capacity of microorganisms to synthesize structurally complex and diverse chemical scaffolds, particularly terpenoids, polyketides, and alkaloids, and couples their biosynthesis to the modulation of enzyme activity through transcriptional or phenotypic readouts. By doing so, the invention enables the discovery of bioactive molecules that might otherwise remain undetected due to low natural abundance, structural complexity, or poor solubility in conventional assay formats.

- construct operons to produce biologically active agents

The invention provides for the construction of synthetic operons that coordinate the expression of multiple genes required for the biosynthesis of biologically active agents, including terpenoids, polyketides, and alkaloids, under the control of inducible or constitutive promoters. These operons are engineered to function within a bacterial host, such as Escherichia coli, and are designed to maximize flux through precursor pathways, minimize metabolic burden, and enhance the yield of target molecules. Each operon comprises a set of heterologous genes encoding enzymes responsible for the synthesis of isoprenoid precursors, such as the mevalonate pathway from Saccharomyces cerevisiae, alongside terpene synthases, cytochrome P450s, halogenases, methyltransferases, or other tailoring enzymes that introduce structural diversity into the core scaffold. The operons are assembled using modular cloning techniques, including Golden Gate or Gibson assembly, to ensure precise orientation and stoichiometric balance of gene expression. The resulting constructs enable the production of structurally varied molecules in a single microbial strain, allowing for the rapid generation of chemical libraries that reflect the combinatorial potential of enzymatic biosynthesis.

- describe fusion proteins

The invention further encompasses the design and implementation of fusion proteins that serve as molecular switches or sensors for enzymatic activity. These fusion proteins are constructed by genetically linking a target enzyme, such as protein tyrosine phosphatase 1B, to a reporter domain that transduces conformational or catalytic changes into a measurable signal. The fusion architecture includes a substrate recognition domain, a DNA-binding domain, and an anchoring unit for RNA polymerase, all arranged to enable a phosphorylation-dependent interaction that governs transcriptional activation. In one embodiment, the fusion protein comprises a Src homology 2 (SH2) domain fused to a transcriptional repressor, which binds to a phosphorylated substrate domain linked to the omega subunit of RNA polymerase. When the target enzyme dephosphorylates the substrate, the interaction is disrupted, suppressing expression of a downstream gene of interest. Conversely, inhibition of the enzyme restores the interaction and activates transcription. This architecture allows for the direct coupling of enzyme inhibition to cellular growth or luminescence, forming the basis of a genetically encoded biosensor.

- introduce substrate recognition domain

The substrate recognition domain is a critical component of the genetically encoded detection system, designed to undergo site-specific post-translational modification by a kinase or phosphatase. In the disclosed invention, this domain is derived from a naturally occurring phosphoprotein sequence, such as MidT or p130cas, and is engineered to contain a single tyrosine residue that serves as the exclusive site of phosphorylation by Src kinase. The sequence is optimized to maximize specificity for the kinase while minimizing off-target modifications by endogenous host kinases. The phosphorylated state of this domain enables high-affinity binding to the SH2 domain of the fusion protein, triggering a conformational change that permits RNA polymerase recruitment and transcriptional initiation. Mutations within this domain, such as tyrosine-to-phenylalanine substitutions, abolish phosphorylation and serve as negative controls, confirming the dependence of the system on enzymatic activity.

- introduce DNA-binding domain

The DNA-binding domain is fused to the SH2 domain and is derived from the 434 phage cI repressor, which recognizes a specific operator sequence upstream of the gene of interest. This domain serves as the molecular interface between the phosphorylation-dependent protein-protein interaction and transcriptional regulation. When the substrate recognition domain is phosphorylated and bound to the SH2 domain, the DNA-binding domain is brought into proximity with the operator, facilitating the recruitment of RNA polymerase through its anchoring unit. The binding affinity of the DNA-binding domain for its operator is tuned through mutagenesis to ensure robust transcriptional activation only upon successful complex formation, thereby reducing background noise and enhancing signal-to-noise ratio. This domain is essential for converting a transient biochemical event into a stable, heritable phenotypic output.

- introduce anchoring unit for RNA polymerase

The anchoring unit for RNA polymerase is a genetically encoded module that facilitates the recruitment of the transcriptional machinery to the promoter region. In the disclosed invention, this unit is implemented as the omega subunit of RNA polymerase, fused directly to the substrate recognition domain. This arrangement ensures that when the phosphorylated substrate engages the SH2-DNA-binding domain fusion, the RNA polymerase is physically tethered to the promoter, enabling transcription initiation. The anchoring unit is chosen for its minimal interference with native transcriptional processes and its compatibility with bacterial host machinery. Its inclusion eliminates the need for additional transcriptional activators, streamlining the system and enhancing its portability across different microbial strains.

- describe reporter gene

The reporter gene is operably linked downstream of the promoter and operator elements and encodes a protein whose expression can be readily quantified. In one embodiment, the reporter gene is luxAB, encoding bacterial luciferase, which produces light upon substrate addition, enabling real-time, non-destructive monitoring of enzyme inhibition. In another embodiment, the reporter gene encodes an antibiotic resistance protein, such as spectinomycin resistance (SpecR), which permits selective growth of cells only when the target enzyme is inhibited. This dual functionality allows for both high-throughput screening via luminescence and selection-based evolution via antibiotic resistance. The choice of reporter gene is determined by the desired application, whether it be rapid detection, long-term selection, or quantitative kinetic analysis.

- introduce first enzyme

The first enzyme in the system is a protein tyrosine kinase, specifically Src kinase, which is responsible for phosphorylating the substrate recognition domain. The enzyme is expressed from a separate plasmid under the control of an inducible promoter, allowing for temporal regulation of phosphorylation. The inclusion of Src kinase ensures that the system remains dependent on a defined, exogenous enzymatic activity, minimizing interference from endogenous host kinases. The kinase is co-expressed with its chaperone, Cdc37, to ensure proper folding and activity in the bacterial cytoplasm.

- introduce second enzyme

The second enzyme is a protein tyrosine phosphatase, specifically protein tyrosine phosphatase 1B (PTP1B), which dephosphorylates the substrate recognition domain and thereby suppresses transcriptional activation. The phosphatase is expressed from a second plasmid under an inducible promoter, enabling precise control over its intracellular concentration. The enzyme is selected for its relevance to human disease, its well-characterized structure, and its suitability for allosteric inhibition. The system is designed such that inhibition of PTP1B—whether by small molecules, mutations, or environmental perturbations—restores the phosphorylation-dependent interaction and activates the reporter gene.

- describe system of proteins

The system of proteins described herein constitutes a genetically encoded biosensor that integrates kinase, phosphatase, substrate, SH2 domain, DNA-binding domain, and RNA polymerase anchoring unit into a single, self-contained regulatory circuit. This system operates as a molecular logic gate, where the output—gene expression—is contingent upon the balance of phosphorylation and dephosphorylation events. The entire system is encoded on one or more plasmids that are stably maintained in a bacterial host, allowing for scalable, reproducible, and modular implementation. The system’s sensitivity, dynamic range, and specificity are tunable through mutagenesis of individual components, promoter strength, ribosome binding sites, and chaperone co-expression.

- introduce protein phosphatase

The protein phosphatase employed in the invention is a member of the protein tyrosine phosphatase family, with protein tyrosine phosphatase 1B (PTP1B) being the primary exemplar. PTP1B is a key regulator of insulin and leptin signaling pathways and is implicated in the pathogenesis of type 2 diabetes, obesity, and HER2-positive breast cancer. Its catalytic domain is structurally representative of the broader PTP family, making it an ideal model system for the development of broadly applicable discovery platforms. The invention enables the detection of inhibitors that bind to both the active site and distal allosteric pockets, including regions that are not amenable to traditional screening methods.

- introduce protein tyrosine phosphatase

The protein tyrosine phosphatase is a catalytic enzyme that hydrolyzes phosphate groups from tyrosine residues on substrate proteins. In the context of this invention, it functions as the molecular target whose inhibition is linked to cellular survival or reporter expression. The enzyme is expressed in a bacterial host that lacks endogenous homologs with significant sequence similarity, thereby minimizing off-target effects. The phosphatase is engineered to be compatible with the fusion protein architecture, ensuring that its catalytic activity directly modulates the phosphorylation state of the substrate domain and, consequently, the transcriptional output of the system.

- introduce protein tyrosine phosphatase 1B

Protein tyrosine phosphatase 1B (PTP1B) is the preferred embodiment of the phosphatase used in the invention. It is selected for its well-documented role in metabolic and oncogenic signaling, its structural accessibility to allosteric modulation, and its amenability to crystallographic and kinetic analysis. The invention demonstrates that PTP1B inhibition can be detected with high sensitivity and specificity using the genetically encoded system, and that inhibitors identified through this platform exhibit biological activity in mammalian cells, including enhanced insulin receptor phosphorylation. The system is further extended to other PTPs, including TC-PTP, PTPN2, PTPN6, and PTPN12, demonstrating its generalizability across the phosphatase family.

- describe reporter protein

The reporter protein is a detectable gene product whose expression level correlates directly with the degree of phosphatase inhibition. In one embodiment, the reporter protein is bacterial luciferase (LuxAB), which generates bioluminescence in the presence of its substrate, allowing for real-time, quantitative measurement of enzyme inhibition. In another embodiment, the reporter protein is spectinomycin resistance protein, which enables selective growth of cells under antibiotic pressure only when the phosphatase is inhibited. The choice of reporter protein determines whether the system is used for screening, selection, or kinetic analysis.

- introduce LuxAB bioreporters

LuxAB bioreporters are employed in the invention as a sensitive, non-invasive means of detecting phosphatase inhibition. These bioreporters consist of the luciferase genes from Photorhabdus luminescens, which catalyze the oxidation of a long-chain aldehyde to produce visible light. The intensity of luminescence is proportional to the level of transcriptional activation, which in turn reflects the degree of phosphatase inhibition. LuxAB is particularly advantageous because it requires no exogenous substrates beyond those naturally present in the cell, allows for continuous monitoring, and is compatible with high-throughput microplate readers.

- introduce fluorescent protein

A fluorescent protein is alternatively employed as a reporter in the invention, enabling spatial and temporal resolution of enzyme activity within single cells. The fluorescent protein is selected from a family of variants including mClover, mNeonGreen, or mScarlet, each offering distinct excitation and emission spectra for multiplexed detection. The fluorescent signal is quantified using flow cytometry or fluorescence microscopy, allowing for the isolation of high-performing clones and the analysis of heterogeneity in population responses.

- introduce mClover

mClover is a bright, photostable green fluorescent protein used in the invention as a reporter for phosphatase inhibition. Its high quantum yield and rapid maturation make it ideal for use in bacterial systems where rapid signal generation is required. When fused to the transcriptional output of the detection system, mClover expression provides a quantitative readout of enzyme inhibition that can be measured by flow cytometry or confocal microscopy, enabling single-cell resolution and the identification of rare, high-potency producers.

- introduce antibiotic resistance

Antibiotic resistance is employed as a phenotypic selection marker in the invention to enable the enrichment of microbial strains capable of producing potent phosphatase inhibitors. The resistance gene, such as spectinomycin resistance (SpecR), is placed under the control of the phosphatase-regulated promoter. Only cells that inhibit the phosphatase survive under selective antibiotic pressure, allowing for the isolation of clones from large libraries without the need for chemical extraction or purification. This strategy enables the direct evolution of inhibitor-producing pathways through iterative rounds of mutagenesis and selection.

- introduce decoy protein fusion

A decoy protein fusion is optionally included in the system to compete with the substrate recognition domain for binding to the SH2 domain, thereby increasing the dynamic range of the sensor. The decoy is a mutated version of the substrate domain that binds the SH2 domain with high affinity but cannot be phosphorylated, effectively sequestering the SH2 domain and raising the threshold for transcriptional activation. This design enhances the sensitivity of the system to weak inhibitors by reducing basal signaling and increasing the signal-to-noise ratio.

- introduce third enzyme

The third enzyme is a tailoring enzyme, such as a cytochrome P450, halogenase, or methyltransferase, that modifies the core scaffold produced by the terpene synthase to generate structural diversity. These enzymes are co-expressed with the biosynthetic operon to functionalize the terpenoid backbone with hydroxyl, halogen, or methyl groups, thereby expanding the chemical space accessible to the screening system. The third enzyme is selected for its compatibility with the host organism and its ability to act on the primary terpenoid product without disrupting cellular viability.

- describe substrate domains

The substrate domains used in the invention are short peptide sequences derived from natural phosphoproteins that contain a single tyrosine residue susceptible to phosphorylation by Src kinase. These domains are engineered for high specificity, minimal cross-reactivity, and optimal binding kinetics to the SH2 domain. Examples include MidT, p130cas, and other Src substrates, each of which has been optimized through mutagenesis to enhance phosphorylation efficiency and reduce background dephosphorylation by endogenous phosphatases.

- introduce light modulated enzyme

A light-modulated enzyme is introduced as an embodiment of the invention to enable spatiotemporal control of phosphatase activity using optogenetic tools. This enzyme is a chimeric protein comprising the catalytic domain of PTP1B fused to a photosensitive LOV2 domain from Avena sativa phototropin 1. Upon exposure to blue light, the LOV2 domain undergoes a conformational change that either activates or inhibits the phosphatase domain, thereby permitting precise, reversible control of enzyme activity with light.

- introduce protein-LOV2 chimera

The protein-LOV2 chimera is a genetically encoded fusion protein in which the catalytic domain of a phosphatase is linked to the LOV2 photosensory domain via a flexible peptide linker. The linker is optimized to transmit conformational changes from the LOV2 domain to the phosphatase active site, resulting in light-dependent modulation of catalytic activity. The chimera is expressed in bacterial or mammalian cells and enables the optical control of intracellular signaling pathways with high temporal and spatial precision.

- introduce PTP1B-LOV2 chimera

The PTP1B-LOV2 chimera is a specific embodiment of the protein-LOV2 fusion in which the catalytic domain of protein tyrosine phosphatase 1B is fused to the LOV2 domain. This chimera exhibits reduced phosphatase activity in the dark and increased activity upon blue light illumination, or vice versa, depending on the orientation and length of the linker. The construct enables the optical perturbation of insulin signaling pathways in mammalian cells and serves as a tool for probing the physiological consequences of localized phosphatase inhibition.

- describe toxic product

A toxic product is optionally introduced into the system to create a negative selection pressure that eliminates cells producing non-inhibitory or off-target compounds. The toxic product is encoded by a gene such as SacB, which converts sucrose into levan, a compound toxic to Gram-negative bacteria. Cells expressing the biosynthetic pathway but failing to inhibit the phosphatase accumulate the toxic product and die, while those producing effective inhibitors survive. This dual-selection strategy enhances the specificity of the screen by removing false positives.

- introduce SacB

SacB is a levansucrase enzyme from Bacillus subtilis that catalyzes the synthesis of levan from sucrose. In the invention, SacB is expressed under the control of a constitutive promoter in strains harboring the phosphatase detection system. When sucrose is added to the growth medium, cells that fail to inhibit the phosphatase produce levan and die, while those producing effective inhibitors survive. This negative selection mechanism complements the positive selection conferred by antibiotic resistance, enabling the simultaneous enrichment of high-potency, low-toxicity producers.

- describe expression vector

The expression vector is a plasmid-based system designed to stably maintain and express the components of the detection and biosynthetic systems in a bacterial host. The vector contains multiple cloning sites, inducible promoters (e.g., arabinose-inducible pBAD or IPTG-inducible lacUV5), ribosome binding sites optimized for bacterial translation, and antibiotic resistance markers for selection. The vector is compatible with standard cloning techniques and can be co-transformed with other plasmids to assemble multi-component systems.

- describe bacterial cell

The bacterial cell used in the invention is a genetically tractable strain of Escherichia coli, such as DH10B, BL21(DE3), or s1030, engineered to support the expression of heterologous enzymes, tolerate terpenoid production, and maintain plasmid stability. The strain is selected for its low endogenous phosphatase activity, high transformation efficiency, and compatibility with inducible expression systems. The cell serves as the chassis for the entire detection and biosynthesis platform.

- detect inhibitors of an enzyme

The invention provides a method for detecting inhibitors of a target enzyme by linking enzyme inhibition to a measurable cellular phenotype, such as growth, luminescence, or fluorescence. The method involves transforming a bacterial cell with a genetically encoded system that couples the inhibition of the enzyme to the expression of a reporter gene. Cells that produce or are exposed to an inhibitor of the enzyme exhibit a detectable signal, enabling high-throughput screening of chemical libraries or biosynthetic gene clusters.

- describe system for detecting inhibitors

The system for detecting inhibitors comprises a genetically encoded circuit that includes a kinase, a substrate recognition domain, a phosphatase, an SH2 domain, a DNA-binding domain, an RNA polymerase anchoring unit, and a reporter gene. The system is configured such that inhibition of the phosphatase restores phosphorylation-dependent transcriptional activation, leading to reporter expression. The system is modular, allowing for the substitution of different phosphatases, kinases, or reporters to adapt to new targets.

- evolve inhibitors of an enzyme

The invention provides a method for evolving inhibitors of an enzyme through iterative cycles of mutagenesis and selection. A library of biosynthetic gene variants is generated by error-prone PCR or site-saturation mutagenesis of the terpene synthase or tailoring enzyme genes. The library is introduced into the detection system, and cells are subjected to selective pressure, such as antibiotic resistance or luminescence-based sorting. Surviving or high-signal clones are isolated, their genes are sequenced, and the process is repeated to enrich for higher-potency inhibitors.

- describe method for evolving inhibitors

The method for evolving inhibitors involves constructing a library of variant biosynthetic pathways, transforming the library into a detection system, applying selective pressure to enrich for inhibitor-producing clones, isolating and sequencing the active variants, and re-introducing them into the system for further rounds of evolution. This iterative process enables the optimization of inhibitor potency, selectivity, and biosynthetic yield without prior structural knowledge of the target.

- detect selective inhibitors

The invention enables the detection of selective inhibitors by constructing parallel detection systems for closely related enzymes, such as PTP1B and TC-PTP. By comparing the growth or signal output of cells exposed to the same compound across the two systems, selective inhibitors can be identified as those that confer a significant advantage in one system but not the other.

- evolve selective inhibitors

Selective inhibitors are evolved by performing parallel directed evolution campaigns on two detection systems, each targeting a different enzyme. Clones that exhibit strong activity in the target system but weak activity in the off-target system are selected and subjected to further rounds of mutagenesis. This strategy enriches for inhibitors that distinguish between highly homologous enzymes.

- evolve photoswitchable enzymes

Photoswitchable enzymes are evolved by generating libraries of chimeric proteins comprising a phosphatase domain fused to a photosensory domain, such as LOV2 or phytochrome. The library is screened under alternating light and dark conditions to isolate variants whose activity is reversibly modulated by light. These variants are then subjected to further rounds of mutagenesis to optimize dynamic range, recovery time, and spectral sensitivity.

- describe method for evolving photoswitchable enzymes

The method for evolving photoswitchable enzymes involves constructing a library of phosphatase-photoreceptor fusions, transforming the library into a bacterial host, and screening for light-dependent changes in reporter gene expression. Clones exhibiting a significant difference in signal between light and dark conditions are isolated, their genes are sequenced, and the process is repeated to refine the photoswitching properties.

- evolve selective mutants of an enzyme

Selective mutants of an enzyme are evolved by introducing random mutations into the enzyme’s coding sequence and screening for variants that exhibit altered substrate specificity, inhibitor sensitivity, or catalytic efficiency. The mutants are selected based on their ability to alter the output of the detection system, enabling the identification of gain-of-function or loss-of-function alleles.

- describe method for evolving selective mutants

The method for evolving selective mutants involves generating a random mutagenesis library of the target enzyme, expressing the library in the detection system, and selecting for clones that exhibit altered reporter output under defined conditions. The selected mutants are characterized for biochemical properties, and the process is repeated to accumulate beneficial mutations.

- evolve substrate domains selective for an enzyme

Substrate domains selective for a particular enzyme are evolved by introducing mutations into the substrate recognition sequence and selecting for variants that are preferentially phosphorylated or dephosphorylated by the target enzyme. This enables the creation of highly specific sensors for individual phosphatase isoforms.

- describe method for evolving substrate domains

The method for evolving substrate domains involves generating a library of random mutations in the substrate recognition sequence, expressing the library in a detection system, and selecting for clones that exhibit maximal signal output in response to the target enzyme’s activity. The selected sequences are then tested for specificity against related enzymes.

- use microbial biosensor

The invention provides for the use of a microbial biosensor to detect the presence of small molecule modulators of enzyme activity. The biosensor is a genetically engineered bacterial strain that expresses a detection system coupled to a biosynthetic pathway. The biosensor responds to the presence of an inhibitor by producing a quantifiable signal, such as luminescence or antibiotic resistance.

- describe method of using microbial biosensor

The method of using the microbial biosensor involves transforming the biosensor strain with a library of biosynthetic gene clusters or chemical compounds, incubating the culture under selective conditions, and measuring the output signal. Clones exhibiting a strong signal are isolated and their genetic or chemical content is analyzed to identify the active modulator.

- provide variants of chemical structures

The invention provides for the generation of structural variants of terpenoid scaffolds through the combinatorial action of engineered biosynthetic enzymes. These variants differ in stereochemistry, ring topology, functional group placement, and hydrophobicity, expanding the chemical diversity accessible to the screening system.

- describe method for providing variants

The method for providing variants involves expressing a library of tailoring enzymes—such as cytochrome P450s, halogenases, or methyltransferases—in conjunction with a terpene synthase. The resulting products are extracted and analyzed for structural diversity, and variants with enhanced inhibitory activity are selected for further optimization.

- describe fusion protein DNA construct

The fusion protein DNA construct is a recombinant DNA molecule encoding a chimeric protein composed of a phosphatase domain, a photosensory domain, a linker region, and optionally a reporter domain. The construct is cloned into an expression vector under the control of an inducible promoter and is designed for seamless assembly using modular cloning techniques.

- define fusion protein

A fusion protein is a single polypeptide chain composed of two or more distinct protein domains encoded by separate genes, joined together by a synthetic linker. In the invention, fusion proteins are used to couple enzymatic activity to transcriptional regulation or optical control.

- describe protein phosphatase

A protein phosphatase is an enzyme that catalyzes the removal of phosphate groups from tyrosine residues on substrate proteins. In the invention, protein phosphatases serve as the molecular targets for inhibitor discovery and optical modulation.

- describe protein light switch

A protein light switch is a genetically encoded photosensitive domain that undergoes a conformational change upon exposure to light, thereby modulating the activity of a fused enzymatic domain. In the invention, the LOV2 domain serves as a light switch for phosphatase activity.

- describe method of using fusion protein

The method of using the fusion protein involves expressing the fusion protein in a bacterial or mammalian cell, exposing the cell to light of a specific wavelength, and measuring the resulting change in enzyme activity or downstream signaling output.

- describe controlling cell movement

The invention enables the control of cell movement by linking phosphatase activity to the regulation of cytoskeletal proteins. For example, inhibition of PTP1B can alter the phosphorylation state of focal adhesion kinases, thereby modulating cell migration.

- describe controlling cell signaling

The invention enables the control of intracellular signaling pathways by optogenetically modulating phosphatase activity. For example, illumination of cells expressing a PTP1B-LOV2 chimera can be used to reversibly activate or inhibit insulin signaling.

- describe modulatory effect

The modulatory effect refers to the ability of a small molecule or light stimulus to alter the catalytic activity of the target enzyme, either by binding to an allosteric site or by inducing a conformational change in the enzyme structure.

- describe illumination

Illumination refers to the application of light of a specific wavelength to activate or inhibit a photoswitchable enzyme. In the invention, blue light (450–470 nm) is used to activate the LOV2 domain, inducing a conformational change that alters phosphatase activity.

- describe light-induced conformational change

The light-induced conformational change is a structural rearrangement in the photosensory domain, such as LOV2, that occurs upon absorption of light. This change is transmitted through a linker to the fused enzymatic domain, altering its catalytic efficiency.

- describe altering catalytic activity

Altering catalytic activity refers to the modulation of the enzyme’s ability to bind substrate or catalyze a reaction. In the invention, this is achieved through allosteric inhibition by small molecules or through light-induced conformational changes in photoswitchable chimeras.

- describe method for detecting small molecule modulator

The method for detecting a small molecule modulator involves expressing the fusion protein in a microbial host, exposing the host to a library of test compounds, illuminating the cells, and measuring the resulting change in reporter gene expression. Compounds that enhance or suppress the light-induced signal are identified as modulators.

- describe providing fusion protein

Providing the fusion protein involves cloning the DNA sequence encoding the chimeric protein into an expression vector and transforming it into a suitable host organism for expression and purification.

- describe expressing fusion protein

Expressing the fusion protein involves culturing the transformed host under conditions that induce transcription and translation of the fusion gene, resulting in the production of the chimeric protein.

- describe contacting with small molecule test compound

Contacting the fusion protein with a small molecule test compound involves incubating the expressed protein or the host cell expressing it with a library of chemical compounds under controlled conditions.

- describe illuminating fusion protein

Illuminating the fusion protein involves exposing the host cell or purified protein to light of a specific wavelength to trigger the conformational change in the photosensory domain.

- describe measuring visual readout

Measuring the visual readout involves quantifying the expression of a reporter gene, such as luciferase or fluorescent protein, using a plate reader, flow cytometer, or microscope.

- describe identifying small molecule test compound

Identifying the small molecule test compound involves isolating and characterizing the compound responsible for the observed modulation of the reporter signal, using chromatography, mass spectrometry, or nuclear magnetic resonance.

- describe using modulatory small molecule test compound

Using the modulatory small molecule test compound involves administering the compound to a biological system, such as a mammalian cell culture or animal model, to elicit a desired physiological response, such as enhanced insulin signaling or reduced tumor growth.

- describe treating patient

Treating a patient involves administering a therapeutically effective dose of a small molecule inhibitor identified by the invention to a subject suffering from a disease associated with aberrant phosphatase activity, such as type 2 diabetes, obesity, or cancer.

- describe disease associated with phosphatase

Diseases associated with phosphatase activity include type 2 diabetes, obesity, HER2-positive breast cancer, autoimmune disorders, and neurodegenerative conditions, all of which involve dysregulated tyrosine phosphorylation signaling.

- describe photoswitchable protein tyrosine phosphatase enzyme construct

The photoswitchable protein tyrosine phosphatase enzyme construct is a genetically encoded fusion protein comprising the catalytic domain of PTP1B linked to the LOV2 photosensory domain, such that its enzymatic activity is reversibly controlled by blue light.

- describe N-terminal alpha helix

The N-terminal alpha helix is a structural element of the LOV2 domain that undergoes a light-induced unfolding transition, which is transmitted to the fused phosphatase domain to modulate its activity.

- describe C-terminal allosteric domain region

The C-terminal allosteric domain region of PTP1B is a structurally dynamic region distal to the active site that undergoes conformational rearrangement upon binding of allosteric inhibitors, such as amorphadiene or bisabolene.

- describe biosensor for enzyme activity

The biosensor for enzyme activity is a genetically encoded system that translates the biochemical activity of a target enzyme into a quantifiable cellular output, such as luminescence, fluorescence, or growth.

- describe substrate domain

The substrate domain is a peptide sequence that is phosphorylated by a kinase and dephosphorylated by a phosphatase, serving as the molecular link between enzymatic activity and transcriptional regulation.

- describe substrate recognition domain

The substrate recognition domain is a protein module, such as an SH2 domain, that binds specifically to the phosphorylated form of the substrate domain, initiating a downstream signaling cascade.

- describe first fluorescent protein

The first fluorescent protein is a donor fluorophore, such as mCerulean, fused to the substrate domain, used in a Förster resonance energy transfer (FRET) pair to monitor conformational changes in the detection system.

- describe second fluorescent protein

The second fluorescent protein is an acceptor fluorophore, such as mCitrine or mClover, fused to the SH2 domain, forming a FRET pair with the first fluorescent protein to report on binding events.

- describe genetically encoded system for detecting small molecules

The genetically encoded system for detecting small molecules is a synthetic biological circuit that links the presence of a small molecule modulator to the expression of a reporter gene through the modulation of enzyme activity.

- describe first region

The first region is a DNA sequence encoding the phosphatase enzyme, operably linked to a promoter and ribosome binding site.

- describe second region

The second region is a DNA sequence encoding the kinase enzyme, operably linked to a separate promoter and ribosome binding site.

- describe operator

The operator is a DNA sequence recognized by the DNA-binding domain of the fusion protein, located upstream of the reporter gene.

- describe binding site for RNA polymerase

The binding site for RNA polymerase is a promoter sequence that recruits the transcriptional machinery and is positioned adjacent to the operator.

- describe gene of interest

The gene of interest is the reporter gene, such as luxAB or SpecR, whose expression is controlled by the phosphorylation-dependent interaction of the fusion protein components.

- describe method for using genetically encoded system

The method for using the genetically encoded system involves transforming a microbial host with the system, exposing it to a library of compounds or gene clusters, and selecting for cells exhibiting a detectable output signal indicative of enzyme inhibition.

- describe providing genetically encoded system

Providing the genetically encoded system involves constructing the DNA components of the detection circuit and assembling them into a plasmid vector for delivery into a host cell.

- describe transforming bacteria

Transforming bacteria involves introducing the plasmid containing the genetically encoded system into a bacterial host using chemical or electroporation methods.

- describe observing expression of gene of interest

Observing expression of the gene of interest involves measuring the level of reporter protein produced, using luminescence, fluorescence, or antibiotic resistance as a proxy for phosphatase inhibition.

- describe pathway that generates linear isoprenoid precursors

The pathway that generates linear isoprenoid precursors is the mevalonate pathway, which converts acetyl-CoA into isopentenyl diphosphate and dimethylallyl diphosphate, the building blocks for terpenoid biosynthesis.

- describe gene for terpene synthase

The gene for terpene synthase encodes an enzyme that catalyzes the cyclization or rearrangement of isoprenoid diphosphates into structurally diverse terpenoid scaffolds.

- describe plurality of E. coli bacteria

A plurality of E. coli bacteria is used to screen a library of biosynthetic gene clusters, each expressing a different terpene synthase or tailoring enzyme, to identify novel phosphatase inhibitors.

- describe extracting terpenoids

Extracting terpenoids involves harvesting the microbial culture, adding an organic solvent such as hexane, and isolating the terpenoid products from the aqueous phase.

- describe identifying terpenoids

Identifying terpenoids involves analyzing the extracted compounds using gas chromatography-mass spectrometry (GC-MS) to determine their chemical structure and purity.

- describe purifying terpenoids

Purifying terpenoids involves chromatographic separation using silica gel or preparative HPLC to isolate individual compounds for biochemical testing.

- describe treating mammalian cell culture

Treating mammalian cell culture involves exposing cells, such as HEK293T or insulin-responsive adipocytes, to purified terpenoid inhibitors and measuring downstream signaling effects, such as insulin receptor phosphorylation.

- describe measuring biochemical effect

Measuring biochemical effect involves quantifying changes in phosphorylation status, gene expression, or metabolic output in response to inhibitor treatment using Western blotting, ELISA, or metabolomics.

- describe quantifying modulatory effect

Quantifying modulatory effect involves determining the concentration of inhibitor required to achieve half-maximal modulation of enzyme activity (IC50) and assessing selectivity across related enzymes.

- define genetically encoded detection operon system

A genetically encoded detection operon system is a set of co-regulated genes assembled on a single DNA construct that links the inhibition of a target enzyme to a selectable or detectable phenotypic output.

- describe method of using inhibitor detection operon

The method of using the inhibitor detection operon involves introducing the operon into a microbial host, exposing the host to a compound library, and selecting for growth or signal output that correlates with enzyme inhibition.

- detail first region of DNA in operable combination

The first region of DNA in operable combination encodes the phosphatase enzyme and is placed under the control of a constitutive promoter.

- describe second region of DNA in operable combination

The second region of DNA in operable combination encodes the kinase enzyme and is placed under the control of an inducible promoter.

- outline method of using genetically encoded detection operon system

The method involves transforming a bacterial host with the operon, inducing expression of the kinase, and applying selective pressure to identify clones that survive or fluoresce due to phosphatase inhibition.

- describe mevalonate-terpene pathway operon

The mevalonate-terpene pathway operon is a synthetic DNA construct that encodes the mevalonate pathway from yeast and a terpene synthase from a plant or fungal source, enabling the production of terpenoids in E. coli.

- detail fourth DNA sequence under control of fifth promoter

The fourth DNA sequence encodes a tailoring enzyme, such as a cytochrome P450, and is placed under the control of a second inducible promoter to enable timed functionalization of the terpenoid scaffold.

- describe transfecting bacteria with inhibitor detection operon

Transfecting bacteria with the inhibitor detection operon involves introducing the plasmid containing the detection circuit into a bacterial strain using electroporation or chemical transformation.

- describe transfecting bacteria with mevalonate pathway operon

Transfecting bacteria with the mevalonate pathway operon involves co-transforming the detection system with the biosynthetic operon to enable simultaneous inhibitor production and detection.

- describe transfecting bacteria with fourth DNA sequence

Transfecting bacteria with the fourth DNA sequence involves introducing a plasmid encoding a tailoring enzyme to generate structural variants of the terpenoid scaffold.

- describe growing bacteria cells expressing three genes of interest

Growing bacteria cells expressing three genes of interest involves culturing the transformed strain under conditions that induce expression of the phosphatase, kinase, and biosynthetic enzymes.

- describe isolating protein phosphatase inhibitor molecules

Isolating protein phosphatase inhibitor molecules involves extracting and purifying the terpenoid compounds produced by the engineered strain and testing them for inhibitory activity in biochemical assays.

- describe treating mammalian cell cultures

Treating mammalian cell cultures involves exposing cells to purified inhibitor molecules and measuring changes in phosphorylation status or metabolic output.

- describe reducing activity of protein phosphatase enzyme

Reducing activity of the protein phosphatase enzyme is achieved by administering an inhibitor identified by the invention, which binds to the enzyme and suppresses its catalytic function.

- describe protein phosphatase enzyme

The protein phosphatase enzyme is a tyrosine-specific hydrolase that removes phosphate groups from tyrosine residues, regulating signaling pathways involved in metabolism, growth, and immune response.

- describe terpene synthase enzyme

The terpene synthase enzyme catalyzes the cyclization of isoprenoid precursors into structurally diverse terpenoid scaffolds, serving as the primary generator of chemical diversity in the invention.

- describe genes of interest

The genes of interest are those encoding the phosphatase, kinase, biosynthetic enzymes, and reporter proteins that together form the core of the genetically encoded system.

- describe method of using genetically encoded system for detecting small molecules

The method involves transforming a microbial host with the detection system, exposing it to a library of compounds or biosynthetic gene clusters, and selecting for cells exhibiting a detectable output signal indicative of enzyme inhibition.

- describe genetically encoded pathway for polyketide biosynthesis

The genetically encoded pathway for polyketide biosynthesis is a synthetic operon that encodes modular polyketide synthases and tailoring enzymes, enabling the production of polyketide inhibitors of phosphatase activity.

- describe method of using genetically encoded system for detecting small molecules and genetically encoded pathway for polyketide biosynthesis

The method involves combining the detection system with a polyketide biosynthetic pathway to screen for novel polyketide inhibitors of phosphatase activity.

- describe method of using genetically encoded system for detecting small molecules and genetically encoded pathway for alkaloid biosynthesis

The method involves combining the detection system with an alkaloid biosynthetic pathway to screen for novel alkaloid inhibitors of phosphatase activity.

- describe engineered bacteria cell line

The engineered bacteria cell line is a genetically modified strain of E. coli that stably expresses the detection system and biosynthetic pathway, enabling continuous production and screening of phosphatase inhibitors.

- describe phosphatase inhibitor molecule produced by bacterium

The phosphatase inhibitor molecule produced by the bacterium is a terpenoid compound, such as amorphadiene or bisabolene, that binds to PTP1B and inhibits its catalytic activity.

- describe bacteria strain producing phosphatase inhibitor molecule

The bacteria strain producing the phosphatase inhibitor molecule is a genetically engineered E. coli strain harboring the mevalonate pathway and a terpene synthase gene, along with the phosphatase detection system.

- describe terpenoid molecule

The terpenoid molecule is a hydrocarbon scaffold derived from isoprenoid precursors, such as amorphadiene or δ-cadinene, that exhibits inhibitory activity against protein tyrosine phosphatases.

- describe inducible promoter

An inducible promoter is a DNA sequence that drives gene expression only in the presence of a specific chemical inducer, such as arabinose or IPTG, allowing for temporal control of enzyme expression.

## DEFINITIONS

- define operon

An operon is a functional unit of DNA containing a cluster of genes under the control of a single promoter, typically transcribed into a single mRNA molecule and coordinately regulated.

- define phosphorylation-regulating enzymes

Phosphorylation-regulating enzymes are proteins that catalyze the addition or removal of phosphate groups from tyrosine, serine, or threonine residues on substrate proteins, thereby modulating their activity, localization, or interactions.

- define phosphorylation

Phosphorylation is the covalent attachment of a phosphate group to a protein, typically at tyrosine, serine, or threonine residues, which alters the protein’s conformation, activity, or binding partners.

- define optogenetic actuator

An optogenetic actuator is a genetically encoded protein or protein fusion that responds to light by undergoing a conformational change that modulates its biological activity, enabling precise spatiotemporal control of cellular processes.

- define dynamic range

Dynamic range refers to the ratio between the maximum and minimum measurable output of a biosensor, reflecting its sensitivity and ability to distinguish between low and high levels of modulation.

- define operably linked

Operably linked refers to the physical and functional connection of genetic elements such that the expression or activity of one element is directly influenced by another, as in a promoter driving the transcription of a downstream gene.

- define other terms

Other terms used herein are defined according to their conventional meanings in the fields of molecular biology, synthetic biology, enzymology, and medicinal chemistry, as understood by a person of ordinary skill in the art.

## DETAILED DESCRIPTION OF INVENTION

- relate invention to genetic engineering

The invention is fundamentally rooted in the principles of genetic engineering, wherein biological systems are rationally redesigned to perform novel functions not found in nature. By integrating synthetic gene circuits with metabolic pathways and optogenetic tools, the invention transforms microbial cells into programmable platforms for the discovery and optimization of bioactive molecules. The system is not merely a screening tool but a self-sustaining biosynthetic and biosensing platform capable of evolving new molecular functions through iterative selection.

- introduce operons for producing biologically active agents

The invention introduces synthetic operons that coordinate the expression of multiple biosynthetic enzymes to produce structurally complex and pharmacologically relevant agents, including terpenoids, polyketides, and alkaloids. These operons are modular, scalable, and compatible with high-throughput cloning, enabling the rapid construction of diverse chemical libraries.

- describe systems for detecting and constructing biologically active agents

The systems described herein are dual-purpose: they detect the presence of bioactive agents through phenotypic readouts and simultaneously construct those agents through engineered biosynthetic pathways. This integration eliminates the need for chemical extraction prior to screening, enabling direct selection of high-potency producers.

- provide microbial operons for identifying small molecule inhibitors/modulators

Microbial operons are provided that link the inhibition of a target enzyme to cellular survival or reporter expression, enabling the identification of small molecule inhibitors or modulators without prior knowledge of their chemical structure.

- describe use of inhibitors/modulator molecules in treating diseases

The inhibitors and modulators identified by the invention are used to treat diseases associated with dysregulated phosphatase activity, including type 2 diabetes, obesity, and cancer, by restoring normal signaling dynamics in affected tissues.

- introduce Protein tyrosine phosphatase 1B (PTP1B) as a valuable model system

Protein tyrosine phosphatase 1B (PTP1B) is introduced as a valuable model system due to its well-characterized structure, its central role in metabolic signaling, and its susceptibility to allosteric inhibition, making it an ideal target for the development of broadly applicable discovery platforms.

- describe PTP1B as an experimentally tractable model system

PTP1B is experimentally tractable because its catalytic domain can be expressed in bacterial systems, crystallized with high resolution, and modulated by small molecules with measurable effects on cellular signaling.

- describe PTP1B as an enzyme for which optical modulation is contemplated

PTP1B is selected as a target for optical modulation because its allosteric sites are amenable to conformational control, and its inhibition can be precisely timed and localized using light-activated chimeras.

- relate PTP1B to spatial regulation and intracellular signaling

PTP1B is spatially regulated by its association with the endoplasmic reticulum, where it dephosphorylates receptor tyrosine kinases such as the insulin receptor. The invention enables the study of this spatial regulation by optogenetically controlling PTP1B activity in subcellular compartments.

### I. Protein Tyrosine Phosphatases (PTPs) and Protein Tyrosine Kinases (PTKs) in Relation to Disease.

- introduce PTPs and PTKs in relation to disease

Protein tyrosine phosphatases and protein tyrosine kinases function as opposing regulators of tyrosine phosphorylation, a key post-translational modification that governs cell growth, differentiation, and survival. Dysregulation of this balance is a hallmark of numerous diseases, including cancer, diabetes, and autoimmune disorders.

- describe PTPs and PTKs as contributing to anomalous signaling events

PTPs and PTKs contribute to anomalous signaling events when their expression, localization, or activity is altered by mutation, overexpression, or environmental perturbation, leading to sustained activation or suppression of downstream pathways.

- describe use of light as photoswitchable constructs for controlling PTPs and PTKs

Light is used as a non-invasive, spatiotemporally precise trigger to control the activity of photoswitchable PTP and PTK constructs, enabling the dissection of signaling dynamics in live cells.

- describe photoswitchable constructs of PTPs and PTKs

Photoswitchable constructs of PTPs and PTKs are fusion proteins comprising a catalytic domain linked to a light-sensitive domain, such as LOV2 or phytochrome, that modulates enzyme activity upon illumination.

- describe use of photoswitchable constructs for identifying specific alleles of PTPs and/or PTKs

Photoswitchable constructs are used to identify alleles of PTPs and PTKs that exhibit altered light sensitivity, enabling the discovery of gain-of-function or loss-of-function variants.

- describe use of photoswitchable constructs for screening and testing molecules

Photoswitchable constructs are used in high-throughput screens to identify small molecules that stabilize or destabilize the light-induced conformational state, thereby enhancing or suppressing photoswitching efficiency.

- introduce Fan et al. reference

Fan et al. demonstrated the use of LOV2 domains to control kinase activity, providing a foundational framework for the development of photoswitchable phosphatases.

- describe LOV2 conjugates

LOV2 conjugates are fusion proteins in which the LOV2 photosensory domain is genetically fused to a target enzyme, enabling light-dependent regulation of catalytic activity.

- introduce WO2011133493 reference

WO2011133493 describes fusion proteins comprising a kinase and a LOV domain for optical control of signaling pathways, establishing the precedent for optogenetic enzyme modulation.

- introduce WO2012111772 reference

WO2012111772 describes polypeptides for optical control of calcium signaling, demonstrating the versatility of light-sensitive domains in synthetic biology.

- introduce U.S. Pat. No. 8,859,232 reference

U.S. Pat. No. 8,859,232 describes fusion proteins comprising protein light switches for the control of gene expression, providing a basis for the integration of optogenetic tools into detection systems.

- describe methods of photomanipulating protein function

Methods of photomanipulating protein function involve the use of genetically encoded photosensitive domains to reversibly alter enzyme activity, protein-protein interactions, or subcellular localization in response to light.

- introduce A. Protein Tyrosine Phosphatases (PTPs)

Protein tyrosine phosphatases are a family of enzymes that catalyze the dephosphorylation of tyrosine residues, playing critical roles in signal transduction and homeostasis.

- describe PTPs as regulatory enzymes

PTPs are regulatory enzymes that counterbalance the activity of protein tyrosine kinases, ensuring precise control over cellular signaling networks.

- describe detailed biophysical studies of PTP IB

Detailed biophysical studies of PTP1B have revealed its allosteric network, conformational dynamics, and structural plasticity, providing a foundation for the design of inhibitors and photoswitches.

- describe allosteric communication in PTPs

Allosteric communication in PTPs involves the transmission of structural changes from distal binding sites to the catalytic center, enabling regulation by small molecules that do not compete with substrate.

- describe results of X-ray crystallography and molecular dynamics simulations

X-ray crystallography and molecular dynamics simulations have identified novel allosteric pockets on PTP1B, including a hydrophobic cleft adjacent to the α7 helix, which is targeted by terpenoid inhibitors.

- describe kinetic studies of PTPs

Kinetic studies of PTPs have established their substrate specificity, catalytic efficiency, and inhibition profiles, enabling the quantitative assessment of inhibitor potency and mechanism.

- describe allosteric network resolved in this study

The allosteric network resolved in this study includes a set of residues connecting the α7 helix to the WPD loop, mediating the transmission of conformational changes induced by inhibitor binding.

- describe new sites for targeting allosteric inhibitors of PTPs

New sites for targeting allosteric inhibitors of PTPs include the hydrophobic cleft formed upon reorganization of the α7 helix, a region not targeted by conventional substrate analogs.

- describe functional influence of disease-associated mutations

Disease-associated mutations in PTPs alter their allosteric communication, substrate specificity, or stability, providing mechanistic insights into pathogenesis and opportunities for targeted intervention.

### II. Optogenetic Actuators.

- introduce optogenetic actuators

Optogenetic actuators are genetically encoded tools that enable the precise control of biological processes using light, offering advantages over chemical inducers in terms of speed, reversibility, and spatial resolution.

- describe biochemical events under optical control

Biochemical events under optical control include enzyme activation, protein-protein interactions, and gene expression, all of which can be modulated with millisecond precision using light-sensitive domains.

- motivate limitations of existing technologies

Existing technologies for controlling enzyme activity, such as chemical inducers or temperature shifts, lack the spatial and temporal precision required to dissect complex signaling networks.

- describe observational interference

Observational interference arises when the act of measuring a biological process alters its behavior, a limitation overcome by the non-invasive nature of optogenetic readouts.

- describe illuminating half the story

Illuminating half the story refers to the fact that while light can activate a protein, it does not necessarily reveal the downstream consequences of that activation, necessitating complementary biosensors.

- describe limited palette of actuators

The limited palette of optogenetic actuators has historically restricted the range of biological processes that can be controlled, but the invention expands this palette by introducing photoswitchable phosphatases.

- introduce photoswitchable constructs

Photoswitchable constructs are fusion proteins that reversibly change their activity in response to light, enabling dynamic control of enzyme function.

- describe advantages over other technologies

Advantages over other technologies include the ability to control activity with high spatial precision, the reversibility of modulation, and the absence of exogenous chemical inducers.

- describe reference WO2013016693

WO2013016693 describes bacteriophytochrome-based photoactivated fusion proteins, demonstrating the feasibility of red-light-controlled systems.

- describe limitations of existing approaches

Limitations of existing approaches include the requirement for exogenous chromophores, slow recovery kinetics, and poor expression in bacterial hosts.

- introduce "cage-free" approach

The "cage-free" approach refers to the use of genetically encoded photosensitive domains that do not require exogenous cofactors, enabling robust expression and function in diverse hosts.

- describe current strategies for optical control

Current strategies for optical control rely on photo-caged amino acids or synthetic photoswitches, which are limited by poor cellular uptake and metabolic instability.

- describe limitations of cage-based systems

Cage-based systems are limited by the need for chemical synthesis, poor bioavailability, and irreversible photolysis, making them unsuitable for long-term or repeated use.

- introduce genetically encoded photoswitchable phosphatase

The genetically encoded photoswitchable phosphatase is a fusion protein comprising PTP1B and the LOV2 domain, enabling reversible, light-dependent control of phosphatase activity.

- describe PTP1B-regulated processes

PTP1B-regulated processes include insulin signaling, leptin signaling, and HER2-mediated oncogenesis, all of which are amenable to optical perturbation.

- describe photoswitchable phosphatases

Photoswitchable phosphatases are engineered enzymes whose catalytic activity is modulated by light, enabling precise temporal and spatial control of tyrosine phosphorylation dynamics.

- state hypothesis

The hypothesis is that fusion of the LOV2 domain to PTP1B will induce a light-dependent conformational change that alters its catalytic activity, enabling optical control of downstream signaling.

- describe experimental approach

The experimental approach involves constructing a library of PTP1B-LOV2 fusions with varying linker lengths and orientations, expressing them in E. coli, and screening for light-dependent changes in reporter gene expression.

- describe kinetic assays and binding studies

Kinetic assays and binding studies are performed to quantify the extent of light-induced modulation, determine the IC50 of inhibition, and assess substrate specificity.

- describe crystallographic and spectroscopic analyses

Crystallographic and spectroscopic analyses are used to determine the structural basis of light-induced conformational change and to validate the mechanism of allosteric modulation.

- describe extension to STEP and PTK6

The approach is extended to other phosphatases and kinases, including STEP and PTK6, demonstrating the generalizability of the photoswitchable design.

- introduce photoswitchable variant of PTP1B

The photoswitchable variant of PTP1B is a chimeric protein in which the LOV2 domain is fused to the C-terminus of PTP1B, resulting in a light-dependent increase in phosphatase activity.

- describe LOV2 domain

The LOV2 domain is a blue-light-sensitive flavoprotein from Avena sativa that undergoes a conformational change upon illumination, leading to unfolding of the Jα helix.

- describe PTP1B-LOV2 chimeras

PTP1B-LOV2 chimeras are fusion proteins in which the catalytic domain of PTP1B is linked to the LOV2 domain via a flexible peptide linker, enabling light-dependent modulation of enzyme activity.

- describe light-dependent catalytic activity

Light-dependent catalytic activity refers to the increase or decrease in phosphatase activity observed upon illumination, which is reversible and repeatable over multiple cycles.

- describe mutational analysis

Mutational analysis is performed to identify residues in the linker or LOV2 domain that enhance the dynamic range or recovery time of the photoswitch.

- describe dynamic range

Dynamic range is the ratio of enzyme activity in the light versus the dark, with higher values indicating greater sensitivity to optical control.

- describe FIG. 1C

FIG. 1C illustrates the luminescence output of the detection system in response to varying concentrations of inducer and light, demonstrating the correlation between phosphatase inhibition and reporter expression.

- describe different crossover points

Different crossover points refer to the positions at which the LOV2 domain is fused to PTP1B, each yielding a distinct level of light-induced modulation.

- describe different partitioning of the linker

Different partitioning of the linker refers to the length and composition of the peptide sequence connecting the LOV2 and PTP1B domains, which affects the efficiency of conformational transmission.

- describe Jalpha helix

The Jα helix is a C-terminal helical element of the LOV2 domain that unfolds upon illumination, transmitting structural change to the fused enzyme.

- describe A'alpha helix

The A’alpha helix is a structural element adjacent to the LOV2 chromophore that stabilizes the dark state and influences the kinetics of photoactivation.

- describe alpha7 helix of PTP1B

The alpha7 helix of PTP1B is a C-terminal structural element that forms part of the allosteric binding pocket and undergoes conformational rearrangement upon terpenoid binding.

- describe combination of sites

Combination of sites refers to the simultaneous mutation of multiple residues in the linker, LOV2, and PTP1B domains to optimize photoswitching performance.

- describe limitations of native AsLOV2 domain

The native AsLOV2 domain exhibits slow recovery kinetics and limited dynamic range, necessitating engineering for improved performance.

- describe efforts to engineer improved variants

Efforts to engineer improved variants include directed evolution of the LOV2 domain to accelerate dark-state recovery and increase light-induced conformational change.

- describe exemplary linkers

Exemplary linkers include glycine-serine repeats of varying lengths, such as (GGGGS)3 or (EAAAK)2, which provide flexibility or rigidity as needed.

- describe exemplary mutations

Exemplary mutations include A450L and V486I in the LOV2 domain, which enhance chromophore stability and increase the magnitude of conformational change.

- describe biophysical studies

Biophysical studies include circular dichroism, fluorescence spectroscopy, and single-molecule FRET to characterize the structural dynamics of the photoswitch.

- describe development of a photoswitchable variant of PTP1B

The development of a photoswitchable variant of PTP1B involved iterative rounds of linker optimization, mutagenesis, and screening to achieve a dynamic range of over 10-fold between light and dark states.

- Optogenetic Actuators

Optogenetic actuators are central to the invention, enabling the precise, reversible, and non-invasive control of enzyme activity with light.

- Target PTP1BPS with dynamic range and recovery time

PTP1BPS is targeted for optimization to achieve a dynamic range greater than 10-fold and a recovery time under 10 minutes, enabling repeated optical perturbations.

- Motivate PTP1BPS for optical control of cell signaling

PTP1BPS is motivated as a tool for optical control of cell signaling because it enables the perturbation of insulin and leptin pathways with high spatiotemporal precision.

- Derive biophysical properties of PTP1BPS

Biophysical properties of PTP1BPS, including extinction coefficient, quantum yield, and thermal relaxation rate, are derived using spectroscopic and kinetic analyses.

- Characterize PTP1B-Substrate and PTP1B-Protein Interactions

PTP1B-substrate and PTP1B-protein interactions are characterized using surface plasmon resonance and co-immunoprecipitation to assess the impact of photoswitching on binding affinity.

- Analyze kinetic studies of PTP1BW and PTP1BPS

Kinetic studies of wild-type PTP1B (PTP1BW) and the photoswitchable variant (PTP1BPS) reveal differences in catalytic efficiency and inhibitor sensitivity.

- Compare substrate specificity of PTP1BW and PTP1BPS

Substrate specificity is compared using a panel of phosphopeptide substrates, revealing that photoswitching does not alter the enzyme’s inherent substrate preference.

- Investigate substrate-dependence of photoswitchability

Substrate-dependence of photoswitchability is investigated by testing the effect of illumination on inhibition by different classes of inhibitors, revealing that photoswitching is independent of substrate identity.

- Assess protein-protein interactions of PTP1BW and PTP1BPS

Protein-protein interactions are assessed using bacterial two-hybrid assays, demonstrating that photoswitching does not disrupt interactions with binding partners such as IRS-1.

- Biostructural characterization of PTP1BPS

Biostructural characterization of PTP1BPS is performed using X-ray crystallography and NMR spectroscopy to visualize the light-induced conformational change.

- Use X-ray crystallography to study PTP1BPS structure

X-ray crystallography reveals that illumination induces a shift in the position of the Jα helix, which propagates to the catalytic site through the α7 helix.

- Use NMR spectroscopy to study PTP1BPS catalytic activity

NMR spectroscopy confirms that the catalytic site remains structurally intact during photoswitching, supporting a purely allosteric mechanism of modulation.

- Exemplary Imaging Methodology to Study Subcellular Signaling Events

Exemplary imaging methodology includes confocal microscopy and FRET-based biosensors to visualize the spatiotemporal dynamics of phosphatase activity in live cells.

- Hypothesize subcellular localization of PTPs and PTKs

It is hypothesized that PTPs and PTKs exert their effects in specific subcellular compartments, such as the endoplasmic reticulum or plasma membrane, and that their spatial organization is critical for signaling fidelity.

- Develop approach for studying spatially localized signaling events

An approach is developed to localize PTP1BPS to specific organelles using targeting sequences, enabling the study of compartment-specific signaling.

- Localize PTP1BPS in living cells

PTP1BPS is localized to the endoplasmic reticulum using a KDEL retention signal, enabling the optical control of insulin receptor dephosphorylation at its physiological site.

- Control PTP1BPS in living cells using confocal microscopy

Confocal microscopy is used to illuminate specific regions of the cell and measure the resulting changes in FRET signal, demonstrating spatially resolved control of phosphatase activity.

- Develop FRET-based sensor for protein phosphorylation

A FRET-based sensor is developed by fusing a phospho-tyrosine binding domain to a donor fluorophore and a phospho-tyrosine-containing substrate to an acceptor fluorophore.

- Optimize FRET sensor for Src kinase activity

The FRET sensor is optimized for Src kinase activity by tuning the linker length and fluorophore pairing to maximize signal change upon phosphorylation.

- Use reaction-diffusion model to study spatially distinct subpopulations

A reaction-diffusion model is used to simulate the diffusion of phosphorylated substrates and phosphatases, predicting the formation of spatially distinct signaling zones.

- Investigate relationships between PTP1BPS activation and sensor phosphorylation

The relationship between PTP1BPS activation and sensor phosphorylation is investigated by co-expressing the sensor and the photoswitch, revealing a direct, reversible correlation.

- Image analysis to estimate ER in different regions of irradiation

Image analysis is used to quantify the intensity of FRET signal in illuminated versus non-illuminated regions, demonstrating localized control of phosphatase activity.

- Spatial Regulation and Intracellular Signaling

Spatial regulation and intracellular signaling are central to the invention, as the ability to control enzyme activity in specific subcellular locations enables the dissection of signaling networks.

- Hypothesize role of PTP1B in tumorigenesis

It is hypothesized that PTP1B promotes tumorigenesis by dephosphorylating receptor tyrosine kinases at the endoplasmic reticulum, dampening anti-proliferative signals.

- Develop tools to investigate differential influence of PTP1B subpopulations

Tools are developed to selectively inhibit or activate PTP1B in different subcellular compartments, enabling the determination of which pool contributes most to oncogenic signaling.

- Network biology to study spatial context in signaling networks

Network biology is applied to model the interactions between PTP1B, its substrates, and its regulators, revealing key nodes that are sensitive to spatial perturbation.

- Generalize approach to Protein Tyrosine Phosphatases and Kinases

The approach is generalized to other PTPs and PTKs, demonstrating that the fusion of photosensitive domains can be applied to a broad range of signaling enzymes.

- Develop photoswitchable variants of STEP and PTK6

Photoswitchable variants of STEP and PTK6 are developed using the same design principles, expanding the toolkit for optical control of signaling.

- Measure substrate specificities of photoswitchable chimeras

Substrate specificities are measured using peptide arrays, confirming that photoswitching does not alter the intrinsic selectivity of the enzymes.

- Collect crystal structures of optimal chimeras

Crystal structures of optimal chimeras are collected to guide future engineering efforts and validate the structural basis of photoswitching.

- Exemplary photoswitch construct sequences for mammalian cells

Exemplary sequences for mammalian expression include codon-optimized PTP1B-LOV2 fusions under the control of a CMV promoter, with nuclear export signals for cytoplasmic localization.

- Exemplary FRET sensors for monitoring PTP1B activity

Exemplary FRET sensors include a phospho-tyrosine binding domain fused to mCerulean and a phosphopeptide substrate fused to mCitrine, with a linker optimized for maximal dynamic range.

- Exemplary mammalian expression vector for expressing photoswitch construct

An exemplary mammalian expression vector includes a CMV promoter, a Kozak sequence, a PTP1B-LOV2 fusion, a polyA signal, and a puromycin resistance gene for selection.

- Contemplative embodiments for invadopodia formation and EGFR regulation

Contemplative embodiments include the use of PTP1BPS to control invadopodia formation in cancer cells and to regulate EGFR signaling in response to light.

- Determine role of cytosolic PTP1B in invadopodia formation

The role of cytosolic PTP1B in invadopodia formation is determined by expressing a cytosolic-targeted PTP1BPS and illuminating cells during matrix degradation assays.

- Analyze cooperative contribution of PTP1B and PTK6 to EGFR regulation

The cooperative contribution of PTP1B and PTK6 to EGFR regulation is analyzed by co-expressing photoswitchable versions of both enzymes and measuring EGFR phosphorylation under alternating light conditions.

### III. Genetically Encoded System for Constructing and Detecting Biologically Active Agents: Microbial Inhibitor Screening Systems.

- introduce operons for specific purposes

Operons are introduced for the specific purpose of linking the biosynthesis of terpenoid scaffolds to the inhibition of phosphatase activity, enabling direct selection of high-potency producers.

- describe genetic operons for insertion

Genetic operons are designed for insertion into bacterial chromosomes or plasmids using homologous recombination or transposon-based integration, ensuring stable inheritance.

- link enzyme activity to cellular luminescence, fluorescence, or growth

Enzyme activity is linked to cellular luminescence, fluorescence, or growth by placing the reporter gene under the control of a promoter activated only when the phosphatase is inhibited.

- detect biologically active molecules and non-native biologically active metabolites

The system detects both endogenous and non-native biologically active metabolites, including terpenoids, polyketides, and alkaloids, regardless of their chemical structure.

- enable detection of metabolites that inhibit/activate a protein of interest

The system enables the detection of metabolites that either inhibit or activate a protein of interest, depending on the configuration of the detection circuit.

- describe genetic operons for detecting and/or evolving biologically active metabolites

Genetic operons are described that combine biosynthetic pathways with detection circuits, enabling both the detection and the directed evolution of bioactive metabolites.

- modify operons for use in detecting and/or evolving biologically active metabolites

Operons are modified by replacing the reporter gene with a toxin gene for negative selection, or by introducing multiple promoters for combinatorial expression.

- install metabolic pathways responsible for making natural metabolites

Metabolic pathways responsible for making natural metabolites, such as the mevalonate pathway or the polyketide synthase cluster, are installed into the host strain to enable biosynthesis.

- enable detection of metabolite-based biological activities

The system enables the detection of metabolite-based biological activities by coupling their production to a selectable phenotype, such as antibiotic resistance.

- modify methods of evolving molecules from Moses et al.

Methods of evolving molecules from Moses et al. are modified by integrating the detection system as a selection pressure, enabling the evolution of inhibitors without chemical screening.

- modify methods of evolving molecules from Badran et al.

Methods of evolving molecules from Badran et al. are modified by replacing chemical extraction with in vivo selection, enabling the evolution of inhibitors directly in the producing organism.

- construct drug leads using microbial hosts

Drug leads are constructed using microbial hosts by combining biosynthetic pathways with detection systems, enabling the rapid generation of novel chemical entities.

- address the development of low-cost pharmaceuticals

The invention addresses the development of low-cost pharmaceuticals by enabling the microbial production of complex inhibitors that would be prohibitively expensive to synthesize chemically.

- use biology for the systematic construction of new molecules

Biology is used for the systematic construction of new molecules by leveraging the catalytic diversity of enzymes to generate chemical structures that are inaccessible to traditional synthesis.

- accelerate the rate and lower the cost of pharmaceutical development

The invention accelerates the rate and lowers the cost of pharmaceutical development by eliminating the need for chemical synthesis, purification, and high-throughput screening.

- optimize pharmacological properties of drug leads

Pharmacological properties of drug leads are optimized by evolving the biosynthetic pathway to introduce functional groups that enhance solubility, membrane permeability, or metabolic stability.

- develop protein inhibitors, particularly for natural products

Protein inhibitors are developed, particularly for natural products, by using the detection system to screen libraries of biosynthetic gene clusters for inhibitors of phosphatase activity.

- use enzymes to construct terpenoid inhibitors

Enzymes are used to construct terpenoid inhibitors by expressing terpene synthases and tailoring enzymes in a detection system, enabling the direct selection of high-potency producers.

- study the molecular-level origin and thermodynamic basis of affinity and activity

The molecular-level origin and thermodynamic basis of affinity and activity are studied by measuring binding constants, enthalpies, and entropies of inhibitor binding to PTP1B.

- develop selective inhibitors of protein tyrosine phosphatase 1B (PTP1B)

Selective inhibitors of PTP1B are developed by screening libraries of terpenoid-producing strains and comparing their activity against closely related phosphatases.

- hypothesize about abietic acid as an allosteric inhibitor of PTP1B

It is hypothesized that abietic acid, a diterpenoid, acts as a weak allosteric inhibitor of PTP1B by binding to a hydrophobic cleft adjacent to the α7 helix.

- make structural variants of abietadiene

Structural variants of abietadiene are made by expressing mutant cytochrome P450s that hydroxylate or halogenate the terpenoid scaffold at specific positions.

- determine free energies, enthalpies, and entropies of binding

Free energies, enthalpies, and entropies of binding are determined using isothermal titration calorimetry and surface plasmon resonance to quantify the thermodynamic basis of inhibition.

- examine the molecular basis and thermodynamic origin of affinity and activity

The molecular basis and thermodynamic origin of affinity and activity are examined by correlating structural features of the terpenoid with binding parameters.

- develop a set of empirical relationships that predict how mutations influence product attributes

A set of empirical relationships is developed that predict how mutations in the terpene synthase or tailoring enzyme influence the structure, potency, and selectivity of the final product.

- evolve high-affinity terpenoid inhibitors of PTP1B

High-affinity terpenoid inhibitors of PTP1B are evolved by iterative rounds of mutagenesis and selection, resulting in compounds with IC50 values below 10 μM.

- identify structure-activity relationships that enable the evolution of terpenoid inhibitors

Structure-activity relationships are identified by correlating terpenoid structure with inhibitory potency, revealing that hydrophobicity and ring topology are key determinants of activity.

- develop a biophysical framework for using protein crystal structures to identify enzymes

A biophysical framework is developed that uses protein crystal structures to identify residues in the binding pocket that are amenable to mutagenesis for enhanced inhibitor affinity.

- apply the technology to develop treatments for diabetes, obesity, and cancer

The technology is applied to develop treatments for diabetes, obesity, and cancer by identifying and optimizing inhibitors of PTP1B and other disease-relevant phosphatases.

- introduce synthetic biology approach

A synthetic biology approach is introduced that integrates genetic circuits, metabolic pathways, and evolutionary selection to discover and optimize bioactive molecules.

- describe limitations of current pathway discovery and optimization methods

Current pathway discovery and optimization methods are limited by their reliance on chemical synthesis, low throughput, and inability to link production to function.

- propose strategy for using synthetic biology to build new molecular function

A strategy is proposed that uses synthetic biology to build new molecular function by coupling biosynthesis to phenotypic selection, enabling the discovery of molecules with desired biological activities.

- describe operon for detecting molecules that inhibit protein of interest

The operon for detecting molecules that inhibit a protein of interest consists of a phosphatase, a kinase, a substrate domain, an SH2 domain, a DNA-binding domain, and a reporter gene, all arranged to activate transcription only when the phosphatase is inhibited.

- discuss advantages of methods and systems over other systems for detecting small molecule inhibitors

Advantages include the ability to screen entire biosynthetic gene clusters without prior knowledge of the compound, the elimination of chemical extraction, and the direct selection of high-potency producers.

- compare with U.S. Pat. No. 6,428,951

U.S. Pat. No. 6,428,951 describes a bacterial two-hybrid system for detecting protein-protein interactions, but does not link inhibition of enzyme activity to cellular survival.

- discuss PCA strategy for detecting interactions of proteins with small molecules

PCA strategy involves splitting a reporter protein into two fragments that reassemble only when a small molecule binds two proteins, but is limited by low signal and high background.

- describe WO2004048549 and its limitations

WO2004048549 describes a system for detecting small molecule binding using split luciferase, but requires chemical induction and lacks the ability to evolve inhibitors.

- discuss advantages of methods and systems over other systems for detecting small molecule inhibitors

The methods and systems of the invention offer superior sensitivity, scalability, and the ability to evolve inhibitors directly in the producing organism.

- describe technology for evolving proteins with different affinities for other proteins/peptide substrates

Technology for evolving proteins with different affinities involves directed mutagenesis of the SH2 domain or substrate sequence, followed by selection for enhanced binding.

- discuss discovery of metabolites with targeted biological activities but unknown structures

The invention enables the discovery of metabolites with targeted biological activities but unknown structures, by selecting for a phenotypic output without requiring prior chemical characterization.

- describe high-throughput metabolic engineering

High-throughput metabolic engineering is achieved by combining combinatorial gene assembly with automated screening, enabling the rapid testing of thousands of biosynthetic variants.

- identify new inhibitors as a starting point

New inhibitors are identified as starting points for lead optimization by isolating the biosynthetic genes from high-performing strains and expressing them in heterologous hosts.

- describe metabolic engineering of E. coli to produce abietadiene

Metabolic engineering of E. coli to produce abietadiene involves introducing the mevalonate pathway and a diterpene synthase from Abies grandis.

- assess ability of minor structural perturbations of abietadiene derivatives to yield improved inhibitors

Minor structural perturbations of abietadiene derivatives, such as hydroxylation or methylation, are assessed for their ability to improve inhibitor potency and selectivity.

- functionalize abietadiene using mutants of cytochrome P450bm3

Abietadiene is functionalized using mutants of cytochrome P450bm3 that introduce hydroxyl groups at specific carbon positions, enhancing solubility and binding affinity.

- perform biostructural analyses of PTP1B

Biostructural analyses of PTP1B include X-ray crystallography, molecular dynamics simulations, and NMR spectroscopy to map the binding sites of terpenoid inhibitors.

- develop high-throughput screens for detecting inhibitors

High-throughput screens are developed using microplate readers to measure luminescence or fluorescence in thousands of microbial cultures simultaneously.

- provide structurally varied terpenoids with different affinities for the allosteric binding pocket

Structurally varied terpenoids are provided that bind to the allosteric pocket with different orientations, affinities, and thermodynamic signatures.

- describe research plan for using enzymes to build selective terpenoid inhibitors of PTP1B

The research plan involves expressing a library of terpene synthases and tailoring enzymes, screening for PTP1B inhibition, isolating the active genes, and evolving them for enhanced potency.

- hypothesize structural changes affecting affinity of ligands for PTP1B

It is hypothesized that structural changes in the terpenoid scaffold, such as ring expansion or stereochemical inversion, will affect affinity by altering shape complementarity with the allosteric pocket.

- describe strategy for generating terpenoids that differ in stereochemistry, shape, and size

The strategy involves expressing mutant terpene synthases that generate alternative cyclization products, such as tricyclic or branched scaffolds, differing in stereochemistry and size.

- generate terpenoids that differ in size using mutations that increase/decrease the volume of the active sites of ABS

Terpenoids that differ in size are generated by mutating residues in the active site of abietadiene synthase to expand or contract the cavity, yielding C15 or C20 terpenoids.

- isolate and characterize new terpenoids

New terpenoids are isolated using hexane extraction and characterized using GC-MS and NMR to determine their chemical structure.

- measure free energy, enthalpy, and entropy of binding to PTP1B

Free energy, enthalpy, and entropy of binding are measured using isothermal titration calorimetry to quantify the thermodynamic basis of inhibition.

- hydroxylate and halogenate terpenoids using mutants of cytochrome P450 BM3 and CYP720B4

Hydroxylate and halogenate terpenoids using mutants of cytochrome P450 BM3 and CYP720B4 to introduce polar functional groups that enhance solubility and binding.

- construct variants with hydroxyl or carboxyl groups at different positions

Variants with hydroxyl or carboxyl groups at different positions are constructed by expressing tailoring enzymes with altered regioselectivity.

- use mutants of tryptophan 6-halogenase and vanadium haloperoxidase to generate halogenated ligands

Mutants of tryptophan 6-halogenase and vanadium haloperoxidase are used to generate brominated or chlorinated terpenoid ligands with enhanced binding affinity.

- screen and characterize halogenated diterpenoids

Halogenated diterpenoids are screened for PTP1B inhibition and characterized for their selectivity against TC-PTP and other phosphatases.

- discuss advantages of methods and systems over other systems for detecting small molecule inhibitors

The methods and systems of the invention offer unparalleled scalability, sensitivity, and the ability to evolve inhibitors directly in the producing organism, without chemical synthesis.

- describe high-throughput screens for terpenoids with a targeted activity

High-throughput screens for terpenoids with targeted activity are performed using luminescence-based detection in 96- or 384-well plates.

- discuss identification of new inhibitors

New inhibitors are identified by comparing the growth or signal output of strains expressing different biosynthetic pathways.

- describe metabolic engineering of E. coli to produce abietadiene

Metabolic engineering of E. coli to produce abietadiene involves introducing the mevalonate pathway and a diterpene synthase from Abies grandis.

- assess ability of minor structural perturbations of abietadiene derivatives to yield improved inhibitors

Minor structural perturbations of abietadiene derivatives, such as hydroxylation or methylation, are assessed for their ability to improve inhibitor potency and selectivity.

- functionalize abietadiene using mutants of cytochrome P450bm3

Abietadiene is functionalized using mutants of cytochrome P450bm3 that introduce hydroxyl groups at specific carbon positions, enhancing solubility and binding affinity.

- perform biostructural analyses of PTP1B

Biostructural analyses of PTP1B include X-ray crystallography, molecular dynamics simulations, and NMR spectroscopy to map the binding sites of terpenoid inhibitors.

- develop high-throughput screens for detecting inhibitors

High-throughput screens are developed using microplate readers to measure luminescence or fluorescence in thousands of microbial cultures simultaneously.

- provide structurally varied terpenoids with different affinities for the allosteric binding pocket

Structurally varied terpenoids are provided that bind to the allosteric pocket with different orientations, affinities, and thermodynamic signatures.

- describe research plan for using enzymes to build selective terpenoid inhibitors of PTP1B

The research plan involves expressing a library of terpene synthases and tailoring enzymes, screening for PTP1B inhibition, isolating the active genes, and evolving them for enhanced potency.

- hypothesize structural changes affecting affinity of ligands for PTP1B

It is hypothesized that structural changes in the terpenoid scaffold, such as ring expansion or stereochemical inversion, will affect affinity by altering shape complementarity with the allosteric pocket.

- describe strategy for generating terpenoids that differ in stereochemistry, shape, and size

The strategy involves expressing mutant terpene synthases that generate alternative cyclization products, such as tricyclic or branched scaffolds, differing in stereochemistry and size.

- summarize advantages of methods and systems over other systems for detecting small molecule inhibitors

The methods and systems of the invention offer unparalleled scalability, sensitivity, and the ability to evolve inhibitors directly in the producing organism, without chemical synthesis.

### IV. Evolving High-Affinity Terpenoid Inhibitors of PTP1B.

- develop high-throughput screens for PTP1B inhibitors

High-throughput screens for PTP1B inhibitors are developed using luminescence-based detection in 96-well plates, enabling the testing of thousands of microbial strains in parallel.

- use site-saturation and random mutagenesis to evolve new inhibitors

Site-saturation and random mutagenesis are used to evolve new inhibitors by introducing mutations into the terpene synthase or tailoring enzyme genes, followed by selection for enhanced inhibition.

- introduce biological selection method for rapid screening

A biological selection method is introduced that uses spectinomycin resistance as a readout for phosphatase inhibition, enabling the rapid isolation of high-potency producers.

- describe operon construction for linking inhibitor potency to cell growth

Operon construction involves placing the spectinomycin resistance gene under the control of a promoter that is only activated when PTP1B is inhibited, thereby linking inhibitor potency to cell growth.

- outline components of operon for PTP1B inhibition

The operon for PTP1B inhibition includes the phosphatase, kinase, substrate, SH2 domain, DNA-binding domain, and spectinomycin resistance gene, all assembled on a single plasmid.

- develop luminescence-based system for operon optimization

A luminescence-based system is developed to optimize the operon by measuring the dynamic range and signal-to-noise ratio of the reporter.

- introduce FRET sensor for PTP1B activity

A FRET sensor is introduced to monitor PTP1B activity in real time by fusing a phospho-tyrosine binding domain to a donor fluorophore and a substrate to an acceptor fluorophore.

- describe FRET sensor construction for monitoring PTP1B activity

The FRET sensor is constructed by cloning the phospho-tyrosine binding domain of PTP1B fused to mCerulean and a phosphopeptide substrate fused to mCitrine, with a flexible linker.

- outline FACS-based screen for PTP1B inhibition

An FACS-based screen is outlined in which cells expressing the biosynthetic pathway and the FRET sensor are sorted based on their fluorescence ratio, enriching for high-inhibitor producers.

- introduce FRET sensor for changes in PTP1B conformation

A FRET sensor is introduced to detect conformational changes in PTP1B by fusing a donor fluorophore to the N-terminus and an acceptor fluorophore to the C-terminus.

- describe FRET experiment for detecting PTP1B conformational changes

The FRET experiment involves measuring the change in emission ratio upon addition of inhibitor, revealing a light-dependent conformational shift.

- outline FACS-based screen for PTP1B conformational changes

The FACS-based screen involves sorting cells based on their FRET ratio, enriching for variants that exhibit large conformational changes in response to inhibitor binding.

- introduce binding-induced changes in tryptophan fluorescence

Binding-induced changes in tryptophan fluorescence are introduced as a readout for inhibitor binding, as the terpenoid inhibitors quench the intrinsic fluorescence of PTP1B.

- describe tryptophan fluorescence-based screen for PTP1B inhibition

The tryptophan fluorescence-based screen involves measuring the decrease in fluorescence intensity upon addition of terpenoid inhibitors, enabling the quantification of binding affinity.

- outline mutagenesis strategies for evolving terpenoid inhibitors

Mutagenesis strategies include error-prone PCR of the terpene synthase gene, site-saturation mutagenesis of the active site, and combinatorial assembly of tailoring enzyme variants.

- introduce site-saturation mutagenesis for identifying plastic residues

Site-saturation mutagenesis is introduced to identify plastic residues in the terpene synthase active site that can be mutated to generate new terpenoid scaffolds.

- describe error-prone PCR for generating mutated terpenoid pathways

Error-prone PCR is used to generate random mutations in the terpenoid biosynthetic pathway, creating a library of variants for screening.

- develop biophysical framework for identifying enzymes capable of synthesizing inhibitors

A biophysical framework is developed that uses protein structure and docking simulations to predict which enzymes are likely to produce high-affinity inhibitors.

- analyze relationships between binding pockets

Relationships between binding pockets are analyzed by comparing the shape, hydrophobicity, and electrostatic potential of the allosteric site across different phosphatases.

- construct matrices for comparing binding pocket properties

Matrices are constructed to compare binding pocket properties across multiple phosphatases, enabling the prediction of inhibitor selectivity.

- validate and extend approach to identify promising active site motifs

The approach is validated by identifying known inhibitors and extended to predict novel motifs that confer high-affinity binding.

### VI. Evolving Optogenetic Actuators: Photoswitchable Constructs.

- introduce optogenetic actuators

Optogenetic actuators are introduced as genetically encoded tools for the precise, reversible, and non-invasive control of enzyme activity using light.

- describe limitations of blue and green light

Limitations of blue and green light include poor tissue penetration and phototoxicity, motivating the development of red- or infrared-light-responsive systems.

- propose photoswitchable enzymes stimulated by red or infrared light

Photoswitchable enzymes stimulated by red or infrared light are proposed by fusing PTP1B to bacteriophytochrome domains, which absorb in the far-red spectrum.

- describe operon to evolve photoswitchable constructs

An operon is described that links the expression of a phosphatase-phytochrome fusion to a reporter gene, enabling selection for light-dependent activity.

- outline control strategy of operon

The control strategy involves expressing the fusion protein under a constitutive promoter and selecting for cells that exhibit high reporter output only under red light.

- describe PTP1B suppression of transcription

PTP1B suppresses transcription by dephosphorylating the substrate domain, preventing the SH2-mediated recruitment of RNA polymerase.

- explain difference in growth rates

Difference in growth rates is explained by the fact that cells producing effective inhibitors survive under antibiotic pressure, while those producing inactive compounds die.

- describe initial experiments with Lux-based luminescence

Initial experiments with Lux-based luminescence demonstrated that light-dependent modulation of PTP1B activity could be detected as a change in bioluminescence.

- outline goals of operon development

Goals of operon development include achieving a dynamic range of 10-fold, a recovery time under 10 minutes, and compatibility with red-light activation.

- describe FRET sensors

FRET sensors are described as tools for monitoring conformational changes in the photoswitchable enzyme in real time.

- outline use of FRET to monitor PTP1B activity

FRET is used to monitor PTP1B activity by measuring the change in emission ratio upon illumination, revealing a direct correlation between light and enzyme state.

- describe optimization of sensor

The sensor is optimized by tuning the linker length, fluorophore pairing, and expression level to maximize signal change and minimize background.

- outline directed evolution of phosphatases and kinases

Directed evolution of phosphatases and kinases is outlined as a strategy to improve photoswitching performance through iterative rounds of mutagenesis and selection.

- describe hypothesis of phytochrome proteins

The hypothesis is that phytochrome proteins, which undergo large conformational changes upon red-light absorption, can be fused to PTP1B to create red-light-responsive switches.

- outline experimental approach

The experimental approach involves constructing a library of PTP1B-BphP1 fusions, expressing them in E. coli, and screening for light-dependent changes in reporter output.

- describe construction of library of PTP1B-phytochrome chimeras

The library is constructed by fusing the PTP1B catalytic domain to the PAS and GAF domains of BphP1 using Golden Gate assembly.

- outline screening of library for functional mutants

The library is screened by exposing colonies to red light and measuring luminescence, selecting for clones that show a strong response.

- describe kinetic and biostructural characterization of mutants

Kinetic and biostructural characterization is performed to determine the dynamic range, recovery time, and structural basis of photoswitching.

- outline extension of approach to STEP and PTK6

The approach is extended to STEP and PTK6 by constructing similar fusions and screening for red-light-dependent modulation.

- describe development of synthetic operon for evolving PTP1B-BphP1 chimeras

A synthetic operon is developed that includes the PTP1B-BphP1 fusion, a constitutive promoter, and a spectinomycin resistance gene under the control of the detection circuit.

- outline components of operon

Components of the operon include the phosphatase-phytochrome fusion, the detection circuit, and the antibiotic resistance gene.

- describe use of operon to evolve photoswitchable PTP1B-BphP1 chimeras

The operon is used to evolve photoswitchable chimeras by performing multiple rounds of mutagenesis and selection under red light.

- outline advantages of using operons

Advantages of using operons include the ability to evolve inhibitors directly in the producing organism and the elimination of chemical extraction.

- describe directed evolution of PTP1B-BphP1 chimeras

Directed evolution of PTP1B-BphP1 chimeras results in variants with enhanced dynamic range, faster recovery, and improved expression.

- outline extension of approach to other enzymes

The approach is extended to other enzymes, including kinases, proteases, and acetyltransferases, demonstrating broad applicability.

## ABBREVIATIONS

- define abbreviations

Abbreviations used herein are defined as follows: PTP, protein tyrosine phosphatase; PTK, protein tyrosine kinase; LOV2, light-oxygen-voltage-sensing domain 2; B2H, bacterial two-hybrid; GC-MS, gas chromatography-mass spectrometry; FRET, Förster resonance energy transfer; IC50, half-maximal inhibitory concentration; IPTG, isopropyl β-D-1-thiogalactopyranoside; DMSO, dimethyl sulfoxide; PTP1B, protein tyrosine phosphatase 1B; TC-PTP, T-cell protein tyrosine phosphatase; PTPN2, protein tyrosine phosphatase non-receptor type 2; PTPN6, protein tyrosine phosphatase non-receptor type 6; PTPN12, protein tyrosine phosphatase non-receptor type 12; PTP1BPS, photoswitchable variant of PTP1B; BphP1, bacteriophytochrome 1; FPP, farnesyl diphosphate; GGPP, geranylgeranyl diphosphate; P450, cytochrome P450; BSA, bovine serum albumin; TCEP, tris(2-carboxyethyl)phosphine; HEPES, 4-(2-hydroxyethyl)-1-piperazineethanesulfonic acid; LB, Luria-Bertani; FBS, fetal bovine serum; ELISA, enzyme-linked immunosorbent assay; SD, standard deviation; SEM, standard error of the mean.

- list protein tyrosine phosphatases

Protein tyrosine phosphatases include PTP1B, TC-PTP, PTPN2, PTPN6, PTPN12, SHP1, SHP2, STEP, PTK6, and CD45.

- list other abbreviations

Other abbreviations include: DNA, deoxyribonucleic acid; RNA, ribonucleic acid; mRNA, messenger RNA; cDNA, complementary DNA; PCR, polymerase chain reaction; RBS, ribosome binding site; ORF, open reading frame; GFP, green fluorescent protein; YFP, yellow fluorescent protein; CFP, cyan fluorescent protein; mClover, monomeric Clover fluorescent protein; LuxAB, bacterial luciferase; SpecR, spectinomycin resistance; SacB, levansucrase; KDEL, Lys-Asp-Glu-Leu endoplasmic reticulum retention signal; CMV, cytomegalovirus promoter; pBAD, arabinose-inducible promoter; pTRC, trc promoter; pET, T7 promoter-based expression vector; Addgene, plasmid repository; PDB, Protein Data Bank; NMR, nuclear magnetic resonance; MD, molecular dynamics; SIM, selected ion monitoring.

## Examples

- illustrate statistical analysis

Statistical analysis is illustrated by reporting the mean and standard deviation of triplicate measurements, with significance determined by Student’s t-test or ANOVA, where p < 0.05 is considered statistically significant.

- exemplify estimation of IC50

IC50 is exemplified by fitting dose-response curves to a four-parameter logistic model using GraphPad Prism, with 95% confidence intervals calculated using nonlinear regression.

- discuss scope of invention

The scope of the invention encompasses any method, system, or composition of matter that links enzyme inhibition to a detectable cellular output, including but not limited to phosphatases, kinases, and other signaling enzymes, and includes any microbial host, biosynthetic pathway, optogenetic actuator, or reporter gene that enables the discovery, evolution, or production of bioactive molecules.