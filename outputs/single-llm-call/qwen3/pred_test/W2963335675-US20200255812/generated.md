# DESCRIPTION

## FIELD OF THE INVENTION

- define field of invention

The present invention relates to the field of enzymology and industrial biotechnology, specifically to modified lipases with altered functional properties arising from the presence or absence of their native propeptide domains. The invention encompasses novel polynucleotides, polypeptides, expression systems, and detergent formulations that exploit the inhibitory function of the propeptide region in class 3 lipases to control enzymatic activity during production, storage, and application. This technology is particularly relevant to the formulation of cleaning compositions, where precise temporal activation of lipase activity is critical for performance, stability, and compatibility with other detergent components.

## BACKGROUND OF THE INVENTION

- introduce lipases
- limitations of lipase production
- summarize prior art

Lipases are hydrolytic enzymes that catalyze the cleavage of ester bonds in triglycerides and other lipid substrates, playing essential roles in biological systems and industrial processes such as detergent manufacturing, food processing, and biodiesel production. Among the most industrially utilized lipases are those belonging to class 3, which are characterized by an α/β-hydrolase fold and a catalytic triad composed of serine, aspartate, and histidine residues. These enzymes are typically secreted as precursor proteins containing an N-terminal signal peptide and a propeptide region that is cleaved during or after secretion to yield the mature, active enzyme. While the mature forms of such lipases have been extensively studied and engineered for enhanced activity, thermostability, and solvent tolerance, the functional role of the propeptide has remained largely unexplored in practical applications. Prior art has focused on optimizing the mature enzyme domain through random mutagenesis or rational design, often overlooking the regulatory influence of the propeptide. Some studies have noted that mutations within the propeptide can affect enzyme activity, but none have disclosed the structural basis for this effect or leveraged it to engineer controlled activation. Furthermore, conventional production methods result in premature activation of lipase during fermentation, leading to self-degradation, reduced yield, and instability in formulation. The absence of a mechanism to suppress lipase activity until the desired point of use has been a persistent limitation in the development of high-performance detergent systems requiring delayed or triggered enzymatic action.

## SUMMARY OF THE INVENTION

- motivate need for improved lipases
- introduce modified polynucleotides

There exists a compelling need for lipases that remain inert during production, purification, and storage, yet activate efficiently under defined environmental conditions such as those encountered during a washing cycle. Such enzymes would enable higher expression yields, improved formulation stability, and more precise control over substrate hydrolysis, thereby enhancing cleaning performance while minimizing unintended side reactions. The present invention addresses this need by introducing modified polynucleotides encoding lipase variants in which the propeptide domain is retained in a non-cleaved or partially cleaved state, thereby maintaining the enzyme in a catalytically inhibited conformation. These polynucleotides are designed to express full-length prolipase proteins or propeptide-mature enzyme complexes that exhibit minimal activity until the propeptide is removed or displaced under specific conditions, such as elevated pH, temperature, or the presence of surfactants. The invention further provides methods for stabilizing these inhibited forms during manufacturing and for reactivating them in situ, thereby enabling the development of next-generation detergent compositions with superior performance profiles.

## DESCRIPTION OF THE INVENTION

### Definition

- define lipase
- define lipase activity
- define lipase inhibitory activity
- define coding sequence
- define control sequences
- define expression
- define expression vector
- define fragment
- define host cell
- define improved property
- define mature polypeptide
- define mature polypeptide coding sequence
- define mutant
- define nucleic acid construct
- define operably linked
- define parent or parent lipase
- define sequence identity

A lipase is an enzyme that catalyzes the hydrolysis of ester bonds in lipids, particularly triglycerides, under physiological or industrial conditions. Lipase activity refers to the catalytic capacity of the enzyme to cleave ester linkages in lipid substrates, measurable by the release of fatty acids or fluorescent products such as 4-methylumbelliferone. Lipase inhibitory activity denotes the ability of a polypeptide domain, such as a propeptide, to bind to the mature lipase and suppress its catalytic function without covalent modification. A coding sequence is a nucleic acid sequence that encodes a polypeptide, including all codons necessary for translation into the full-length protein. Control sequences are regulatory DNA elements, such as promoters, enhancers, terminators, and ribosome binding sites, that govern the transcription and translation of a coding sequence in a host cell. Expression refers to the process by which a gene is transcribed into mRNA and translated into a functional polypeptide. An expression vector is a recombinant nucleic acid construct designed to facilitate the replication and expression of a coding sequence in a host cell. A fragment is a portion of a nucleic acid or polypeptide sequence that retains at least one biological or structural function of the full-length molecule. A host cell is a living cell capable of receiving and expressing a foreign nucleic acid construct, including prokaryotic and eukaryotic organisms such as bacteria, yeast, and filamentous fungi. An improved property refers to a measurable enhancement in a characteristic of the lipase, such as increased thermostability, enhanced expression yield, prolonged storage stability, or controlled activation kinetics. A mature polypeptide is the functional form of a lipase after removal of the signal peptide and, optionally, the propeptide. A mature polypeptide coding sequence is the nucleic acid sequence encoding the mature polypeptide, excluding the sequences for the signal peptide and propeptide. A mutant is a variant of a native polypeptide or polynucleotide sequence that differs by one or more substitutions, insertions, or deletions. A nucleic acid construct is a synthetic or recombinant assembly of nucleic acid segments, including coding sequences and control elements, assembled for the purpose of expression in a host cell. Operably linked describes the physical and functional connection between a coding sequence and one or more control sequences such that the control sequences direct the expression of the coding sequence. A parent or parent lipase is the native, unmodified lipase from which a mutant is derived. Sequence identity is the percentage of identical residues between two aligned polypeptide or nucleic acid sequences, calculated over a defined region of comparison.

### Conventions for Designation of Variants

- describe conventions for designation of variants
- align amino acid sequences
- determine corresponding amino acid residue
- use multiple sequence comparison
- use pairwise sequence comparison algorithms
- use probabilistic representations of polypeptide families
- generate homology models
- describe muscle v3.8.31 basic usage

Variants of the lipase are designated according to standardized nomenclature in which the position of each residue is numbered relative to the first amino acid of the mature polypeptide or the full-length precursor, as appropriate. Amino acid sequences are aligned using computational methods to identify homologous regions and determine corresponding residues across different lipase variants. Multiple sequence comparisons are performed to identify conserved structural motifs, particularly within the propeptide and lid regions. Pairwise sequence comparison algorithms, such as Needleman-Wunsch and Smith-Waterman, are employed to quantify similarity and identify optimal alignments. Probabilistic representations of polypeptide families, including hidden Markov models, are used to predict structural and functional impacts of mutations. Homology models are generated to predict the three-dimensional structure of variants when experimental structures are unavailable. The program MUSCLE version 3.8.31 is used for multiple sequence alignment, with default parameters for gap opening and extension penalties, and iterative refinement to maximize alignment accuracy. The output is visually inspected and manually adjusted where necessary to preserve biologically relevant structural features, particularly around the catalytic triad and propeptide-binding interface.

### Muscle v3.8.31 Basic Usage

- describe muscle v3.8.31 basic usage

MUSCLE version 3.8.31 is a multiple sequence alignment tool optimized for speed and accuracy in aligning large sets of protein or nucleic acid sequences. The program is executed via command-line interface, requiring an input file in FASTA format containing the sequences to be aligned. Default parameters are employed unless otherwise specified, including three iterative refinement cycles and a maximum of 16 iterations. The algorithm proceeds through three stages: draft progressive alignment, refinement of pairwise alignments, and final refinement using a tree-based approach. The output is generated in multiple formats, including CLUSTAL, FASTA, and PHYLIP, and is used as input for downstream structural and functional analyses. The program is capable of handling sequences with high divergence and is particularly effective in aligning lipase propeptide regions due to its sensitivity to conserved secondary structure elements.

### Common Options (for a Complete List Please See the User Guide):

- describe input file option
- describe output file option
- describe find diagonals option
- describe maximum number of iterations option
- describe maximum time to iterate option
- describe log option

The input file option specifies the path to a file containing the sequences to be aligned, formatted in FASTA with unique identifiers for each sequence. The output file option defines the destination and format of the resulting alignment, supporting formats such as CLUSTAL, FASTA, and PHYLIP. The find diagonals option enables the algorithm to identify conserved local regions of high similarity before global alignment, improving accuracy in divergent sequences. The maximum number of iterations option sets the upper limit for refinement cycles, with a default of 16 to balance computational efficiency and alignment quality. The maximum time to iterate option restricts the total runtime of the alignment process, preventing excessive computation on large datasets. The log option generates a detailed text file recording the progression of the alignment, including iteration counts, convergence metrics, and runtime statistics, facilitating reproducibility and troubleshooting.

### Polynucleotides

- define polynucleotide encoding lipase
- describe propeptide and mature protein of lipase
- specify sequence identity of propeptide and mature protein
- identify lipase contact zones in propeptide
- describe alterations in propeptide
- specify positions of alterations in propeptide
- describe substitutions in propeptide
- describe insertions in propeptide
- describe deletions in propeptide
- specify number of alterations in propeptide
- describe effect of alterations on lipase association
- describe effect of alterations on thermostability
- describe propeptide contact zones in mature protein
- describe alterations in mature protein
- specify positions of alterations in mature protein
- describe effect of alterations on lipase production
- summarize polynucleotide variants

A polynucleotide encoding a lipase is a nucleic acid sequence that directs the synthesis of a lipase polypeptide, including the coding regions for the signal peptide, propeptide, and mature enzyme. The propeptide is an N-terminal segment that, when bound to the mature protein, inhibits lipase activity by occluding the active site and stabilizing the closed conformation of the lid domain. The mature protein is the catalytically active core of the lipase following removal of the signal peptide and, optionally, the propeptide. The sequence identity between the propeptide and mature protein regions is typically less than 20%, with the propeptide exhibiting low primary sequence conservation but high structural complementarity to the mature enzyme. Lipase contact zones in the propeptide include residues involved in hydrophobic packing, hydrogen bonding, and salt bridge formation with the lid and substrate-binding regions of the mature enzyme. Alterations in the propeptide may include substitutions, insertions, or deletions that modulate the affinity or kinetics of propeptide binding. Specific positions subject to alteration include residues 76, 77, 81, 88, and 97, corresponding to key interaction sites with the lid helix and hydrophobic anchor residues. Substitutions may involve replacement of hydrophobic residues with smaller or polar residues to reduce binding affinity, while insertions or deletions may disrupt the structural continuity of the propeptide. The number of alterations in the propeptide ranges from one to ten, with each alteration independently or synergistically affecting the stability of the inhibited complex. Alterations that weaken propeptide-lipase association lead to faster activation kinetics, while those that enhance binding prolong inhibition. These modifications can also influence thermostability, with tighter binding generally increasing the melting temperature of the complex. Propeptide contact zones in the mature protein include residues I183, V348, F345, and L349, which form a hydrophobic cluster that is shielded by the propeptide. Alterations in the mature protein may be introduced at positions adjacent to these contact zones to modulate accessibility or electrostatic complementarity. Positions such as 184, 186, 345, and 348 are targeted to fine-tune the interaction interface. Such alterations can improve lipase production by reducing aggregation during expression or by facilitating purification through differential binding to affinity matrices. Polynucleotide variants encompass full-length prolipase constructs, propeptide-deleted mutants, and chimeric constructs in which the propeptide is replaced with heterologous inhibitory domains.

### Nucleic Acid Constructs

- define nucleic acid constructs
- describe polynucleotide encoding variant
- specify control sequences for expression
- describe promoters for bacterial host cells
- describe promoters for filamentous fungal host cells
- describe promoters for yeast host cells
- describe transcription terminators
- specify terminators for bacterial host cells
- specify terminators for filamentous fungal host cells
- specify terminators for yeast host cells
- describe mRNA stabilizer regions
- describe leaders for filamentous fungal host cells
- describe leaders for yeast host cells
- describe polyadenylation sequences
- specify polyadenylation sequences for filamentous fungal host cells
- specify polyadenylation sequences for yeast host cells
- describe signal peptide coding regions
- specify signal peptides for bacterial host cells
- specify signal peptides for filamentous fungal host cells
- specify signal peptides for yeast host cells
- describe propeptide coding sequences
- summarize nucleic acid constructs

A nucleic acid construct is a recombinant DNA molecule comprising a polynucleotide encoding a lipase variant operably linked to control sequences necessary for expression in a host cell. The polynucleotide may encode a full-length prolipase, a mature lipase with an intact propeptide, or a fusion protein in which the propeptide is fused to a heterologous tag. Control sequences include promoters, terminators, leaders, and polyadenylation signals tailored to the host organism. For bacterial host cells, promoters such as lac, tac, T7, or aprE are used to drive high-level expression. In filamentous fungal hosts such as Aspergillus oryzae, promoters derived from the amylase or glucoamylase genes are preferred for strong, secretory expression. In yeast hosts such as Saccharomyces cerevisiae or Pichia pastoris, the alcohol oxidase (AOX1) or glyceraldehyde-3-phosphate dehydrogenase (GAP) promoters are commonly employed. Transcription terminators ensure proper mRNA processing and stability; for bacterial hosts, the T7 terminator or rrnB terminator is used; for filamentous fungi, the trpC or gpd terminator is preferred; and for yeast, the CYC1 or ADH1 terminator is utilized. mRNA stabilizer regions, such as 5′ and 3′ untranslated sequences from highly expressed genes, are included to enhance transcript half-life. Leaders for filamentous fungal hosts include the α-amylase or acid protease signal sequences, while for yeast, the α-factor preproleader is employed. Polyadenylation sequences for filamentous fungi include the trpC or gpd 3′ UTR, while for yeast, the ADH1 or PGK 3′ UTR is used. Signal peptide coding regions direct secretion of the lipase and are derived from native fungal or bacterial sources, such as the signal peptide of Aspergillus oryzae α-amylase or Bacillus subtilis aprE. Propeptide coding sequences are derived from the native lipase or engineered variants thereof. Nucleic acid constructs may be linear or circular, integrated or episomal, and are designed for stable expression in industrial host strains.

### Expression Vectors

- define recombinant expression vectors
- describe vector components
- explain vector construction
- discuss vector compatibility with host cells
- describe autonomously replicating vectors
- explain integration into host cell genome
- discuss homologous recombination
- describe non-homologous recombination
- explain origins of replication
- provide examples of bacterial origins of replication
- provide examples of yeast origins of replication
- provide examples of filamentous fungal origins of replication

A recombinant expression vector is a DNA molecule engineered to carry and express a target polynucleotide in a host cell. Vector components include an origin of replication, selectable marker, promoter, coding sequence, terminator, and optional enhancer or fusion tags. Vector construction involves the ligation of these components into a plasmid or viral backbone using restriction enzymes or recombination-based cloning methods. Compatibility with the host cell is determined by the origin of replication and the presence of host-specific regulatory elements. Autonomously replicating vectors, such as pUC or pET plasmids for bacteria, or 2μ plasmids for yeast, replicate independently of the host chromosome. Integration into the host genome is achieved through homologous recombination, where vector sequences flanking the insert share identity with genomic loci, or through non-homologous recombination, which relies on random insertion mediated by transposons or endogenous repair mechanisms. Origins of replication include ColE1 for Escherichia coli, pBR322 for Gram-negative bacteria, and pTA1 for Bacillus species. In yeast, the 2μ circle origin supports high-copy replication, while in filamentous fungi, the AMA1 or pyrG locus enables stable episomal maintenance. These vectors are selected based on host compatibility, copy number, and ease of selection to ensure robust and scalable production of the lipase variant.

### Host Cells

- define recombinant host cells
- describe host cell types
- discuss prokaryotic host cells
- describe Gram-positive bacteria
- describe Gram-negative bacteria
- provide examples of Bacillus cells
- provide examples of Streptococcus cells
- provide examples of Streptomyces cells
- describe eukaryotic host cells
- discuss fungal host cells
- describe yeast host cells
- provide examples of yeast cells
- describe filamentous fungal host cells
- provide examples of filamentous fungal cells
- explain DNA introduction into host cells
- describe transformation methods
- provide examples of transformation methods for various host cells

A recombinant host cell is a cell that has been genetically modified to express a foreign polynucleotide encoding a lipase variant. Host cell types include prokaryotic and eukaryotic organisms, each selected for their expression capacity, secretion efficiency, and scalability. Prokaryotic host cells include Gram-positive bacteria such as Bacillus subtilis, Bacillus licheniformis, and Bacillus amyloliquefaciens, which secrete proteins efficiently into the culture medium. Gram-negative bacteria such as Escherichia coli and Pseudomonas putida are used for intracellular expression or when periplasmic secretion is desired. Streptomyces species, including Streptomyces lividans and Streptomyces coelicolor, are employed for complex secondary metabolite production. Eukaryotic host cells include fungi and yeasts, which offer post-translational modifications and robust secretion pathways. Fungal host cells, particularly filamentous fungi such as Aspergillus oryzae, Aspergillus niger, and Trichoderma reesei, are preferred for industrial lipase production due to their high secretion titers and compatibility with fermentation systems. Yeast host cells such as Saccharomyces cerevisiae, Pichia pastoris, and Hansenula polymorpha are used for rapid, high-density expression. DNA is introduced into host cells via transformation methods including electroporation, chemical transformation, protoplast fusion, and Agrobacterium-mediated transfer. For Bacillus species, natural competence or electroporation is employed; for E. coli, calcium chloride or electroporation is standard; for yeast, lithium acetate or electroporation is used; and for filamentous fungi, protoplast transformation or particle bombardment is preferred. Each method is optimized for the host cell wall composition and membrane permeability to ensure efficient uptake and stable integration of the expression construct.

### Methods of Production

- cultivate host cell
- recover variant
- detect variant using methods known in the art
- recover variant from nutrient medium
- purify variant by chromatography
- purify variant by electrophoretic procedures
- purify variant by differential solubility
- purify variant by SDS-PAGE
- purify variant by extraction
- use host cell as source of variant
- optimize lipase production
- generate lipase preparations with increased lipolytic activity
- alter one or more nucleotides in polynucleotide
- identify lipase contact zones
- select alteration of one or more nucleotides
- determine sequence identity of lipase contact zones
- prepare detergent composition
- activate or reactivate lipase
- increase activity of lipase
- isolate propeptide variant
- use isolated propeptide variant for modifying lipase
- modify lipase
- alter property of lipase
- describe improved properties
- incorporate additional components in detergent composition
- select suitable component materials
- determine levels of incorporation
- describe surfactants
- select suitable anionic detersive surfactants
- select suitable non-ionic detersive surfactants
- select suitable cationic detersive surfactants
- describe mid-chain branched detersive surfactants
- provide non-limiting examples of surfactants
- define cationic detersive surfactants
- specify formula for quaternary ammonium compounds
- provide examples of cationic surfactants
- define amphoteric/zwitterionic surfactants
- specify examples of amine oxides and betaines
- describe anionic surfactants and their neutralization
- specify agents for neutralization
- describe semipolar surfactants
- specify examples of amine oxides
- describe surfactant systems
- specify preferred weight ratios of anionic to nonionic surfactant
- describe soap
- specify examples of soap
- describe fatty acids
- specify preferred fatty acids
- describe hydrotropes
- describe builders
- introduce cleaning composition
- describe co-builders
- list examples of co-builders
- introduce chelating agents and crystal growth inhibitors
- describe chelating agents and crystal growth inhibitors
- list examples of chelating agents and crystal growth inhibitors
- introduce bleach component
- describe bleach component
- introduce pre-formed peracids
- describe pre-formed peracids
- list examples of pre-formed peracids
- introduce sources of hydrogen peroxide
- describe sources of hydrogen peroxide
- list examples of sources of hydrogen peroxide
- introduce bleach activators
- describe bleach activators
- list examples of bleach activators
- describe diacyl peroxides
- describe tetraacyl peroxides
- introduce bleach catalysts
- describe bleach catalysts
- list examples of bleach catalysts
- describe iminium cations and polyions
- list examples of iminium cations and polyions
- introduce bleach catalysts
- list modified amine oxygen transfer catalysts
- list modified amine oxide oxygen transfer catalysts
- list N-sulphonyl imine oxygen transfer catalysts
- list N-phosphonyl imine oxygen transfer catalysts
- list N-acyl imine oxygen transfer catalysts
- list thiadiazole dioxide oxygen transfer catalysts
- list perfluoroimine oxygen transfer catalysts
- list cyclic sugar ketone oxygen transfer catalysts
- describe preferred bleach catalyst structures
- describe bleach component composition
- describe peracid sources
- describe peracid and/or bleach activator amounts
- describe metal-containing bleach catalysts
- describe manganese compound catalysts
- describe cobalt bleach catalysts
- describe transition metal complex catalysts
- describe photobleaches
- describe fabric hueing agents
- list small molecule dyes
- list polymeric dyes
- describe preferred hueing dyes
- describe whitening agents
- describe dye clay conjugates
- list suitable dye clay conjugates
- describe bleach catalysts in combination with peracid sources
- describe bleach catalysts in combination with bleach activators
- describe bleach catalysts in combination with organic peroxyacids
- describe bleach catalysts in combination with hydrogen peroxide sources
- describe bleach catalysts in combination with perhydrolase enzymes
- describe bleach catalysts in combination with esters
- describe bleach catalysts in combination with bleach activators and peracid sources
- describe bleach catalysts in combination with bleach activators and hydrogen peroxide sources
- describe bleach catalysts in combination with bleach activators and perhydrolase enzymes
- describe bleach catalysts in combination with bleach activators and esters
- describe exemplary bleaching systems
- define pigments
- list pigments
- describe encapsulates
- define encapsulate components
- describe encapsulate core materials
- describe encapsulate shell materials
- specify encapsulate properties
- describe encapsulate manufacturing
- describe formaldehyde scavengers
- describe deposition aids
- list deposition aid polymers
- describe perfumes
- list perfume raw materials
- describe encapsulated perfume particles
- describe perfume particle components
- describe perfume particle manufacturing
- describe pre-complexed perfumes
- describe perfume-polyamine complexes
- describe Schiff base formation
- describe perfume composition
- describe perfume concentration
- describe perfume application
- define polymers
- list examples of polymers
- describe amphiphilic cleaning polymers
- describe alkoxylated grease cleaning polymers
- describe alkoxylated polycarboxylates
- describe isoprenoid-derived surfactants
- describe carboxylate polymers
- describe soil release polymers
- describe cellulosic polymers
- describe enzymes
- list examples of enzymes
- introduce enzymes
- describe cellulases
- list examples of cellulases
- describe proteases
- list examples of proteases
- describe subtilases
- list examples of subtilases
- describe metalloproteases
- list examples of metalloproteases
- describe lipases and cutinases
- list examples of lipases and cutinases
- describe amylases
- list examples of amylases
- describe hybrid alpha-amylase
- list examples of hybrid alpha-amylase
- describe additional amylases
- list examples of additional amylases
- describe peroxidases/oxidases
- list examples of peroxidases
- describe commercially available cellulases
- describe commercially available proteases
- describe commercially available lipases
- describe commercially available amylases
- describe commercially available peroxidases
- summarize enzyme properties
- discuss enzyme compatibility
- discuss enzyme effectiveness
- conclude enzyme selection
- define haloperoxidase enzyme
- classify haloperoxidases by specificity
- describe chloroperoxidase
- specify vanadium haloperoxidase
- combine vanadate-containing haloperoxidase with chloride ion source
- isolate haloperoxidases from fungi
- isolate haloperoxidases from bacteria
- specify Curvularia sp. as haloperoxidase source
- specify Drechslera hartlebii as haloperoxidase source
- define laccase enzyme
- classify laccase enzymes by EC number
- specify microbial origin of laccase enzymes
- derive laccase from Aspergillus
- derive laccase from Neurospora
- derive laccase from Coprinopsis
- derive laccase from Myceliophthora
- specify pectate lyases and mannanases
- formulate detergent additive
- produce non-dusting granulates
- stabilize liquid enzyme preparations
- define dye transfer inhibiting agents
- specify polymeric dye transfer inhibiting agents
- define brighteners
- specify C.I. fluorescent brightener 260
- formulate silicate salts
- specify dispersants
- introduce cationic polymers
- describe complex coacervates
- introduce nonionic polymers
- describe conditioning agents
- specify silicone conditioning agents
- specify organic conditioning oils
- introduce hygiene and malodour agents
- describe probiotics
- introduce suds boosters
- describe suds suppressors
- specify monocarboxylic fatty acid suds suppressors
- specify silicone suds suppressors
- specify monostearyl phosphate suds suppressors
- specify hydrocarbon suds suppressors
- specify alcohol suds suppressors
- describe pH range of compositions
- describe temperature range of compositions
- introduce form of detergent composition
- describe solid or liquid cleaning compositions
- describe gel or paste compositions
- describe soap bar compositions
- describe regular or compacted powder compositions
- describe granulated solid compositions
- describe homogenous or multilayer tablet compositions
- describe pouch compositions
- describe single or multi-compartment unit dose forms
- introduce lipase particles
- describe lipase crystals or precipitate
- describe spray or freeze-dried lipase
- describe granulated lipase
- specify particle size of lipase particles
- specify lipase protein content
- introduce water-soluble film
- describe PVOH film
- describe optional ingredients for water-soluble film
- introduce PVOH resin properties
- describe molecular weight range
- list optional additive ingredients
- discuss plasticizers for PVOH
- describe surfactants for water-soluble films
- introduce defoamers for water-soluble films
- describe processes for making water-soluble articles
- outline casting process for water-soluble film
- describe enzyme addition and stirring process
- discuss drying conditions for maintaining enzyme activity
- describe film uses, including pouch formation
- introduce multi-compartment pouches and packets
- describe compartment arrangements and sizes
- discuss compartment geometries and designs
- describe film selection for pouches and packets
- outline equipment and methods for making pouches and packets
- describe vertical form filling, horizontal form filling, and rotary drum filling
- discuss mold sizes and shapes
- introduce partitioning walls in multi-compartment pouches
- describe composition selection for pouches and packets
- discuss heat application for thermoforming
- describe wetting methods for film
- outline vacuum drawing and mold filling
- discuss closing and sealing methods
- describe heat sealing temperatures
- introduce solvent welding and wet sealing
- describe cutting methods for pouches and packets
- outline process for making multi-compartment pouches
- describe forming recesses in compartments
- discuss filling and closing second compartments
- describe sealing and cutting multi-compartment pouches
- introduce alternative process for making multi-compartment pouches
- describe forming first compartment
- discuss filling first compartment
- outline deforming second film for second compartment
- describe filling second compartment
- discuss sealing second compartment
- describe placing second compartment onto first compartment
- outline sealing and cutting multi-compartment pouches
- discuss selecting forming machines
- describe feed stations for manufacturing multi-compartment pouches
- discuss incorporating different compositions
- conclude manufacturing process

The lipase variant is produced by cultivating a recombinant host cell under controlled conditions in a nutrient medium optimized for protein expression and secretion. The host cell is grown in a bioreactor at a temperature between 25°C and 37°C, with pH maintained between 5.5 and 8.0, and aeration sufficient to support high cell density. After a fermentation period of 24 to 96 hours, the culture supernatant is harvested and clarified by filtration or centrifugation. The lipase variant is recovered from the supernatant using chromatographic techniques, including hydrophobic interaction chromatography, ion exchange chromatography, or affinity chromatography with decylamine resin, which selectively binds the mature enzyme while allowing the propeptide-bound complex to remain in the flow-through. The variant may be detected using zymogram analysis, native PAGE, or fluorescence-based activity assays. Purification is further refined by differential solubility, electrophoretic separation, or extraction with organic solvents. The purified lipase variant may be used directly as a component of a detergent composition or formulated into a solid or liquid preparation. Lipase production is optimized by altering one or more nucleotides in the polynucleotide encoding the propeptide or mature enzyme to enhance secretion yield, reduce aggregation, or modulate activation kinetics. Contact zones between the propeptide and mature enzyme are identified through structural analysis, and alterations are selected to either stabilize or destabilize the inhibitory complex. Sequence identity of these contact zones is determined by alignment with homologous lipases to ensure conservation of key residues. The detergent composition is prepared by incorporating the lipase variant with surfactants, builders, bleach components, and other additives. Activation of the lipase is triggered by environmental conditions such as pH shift, temperature increase, or surfactant-induced displacement of the propeptide. Improved properties include enhanced storage stability, delayed activation, increased specific activity upon activation, and compatibility with high-alkaline formulations. Additional components such as surfactants, builders, and bleach catalysts are incorporated at levels determined by efficacy and formulation stability. Anionic surfactants such as linear alkylbenzene sulfonate and sodium lauryl sulfate are selected for their cleaning power, while non-ionic surfactants such as alcohol ethoxylates provide emulsification. Cationic surfactants, including quaternary ammonium compounds of the formula R1R2R3R4N⁺X⁻, are used for fabric conditioning. Mid-chain branched surfactants improve solubility in hard water. Soap, derived from fatty acids such as stearic or palmitic acid, may be included in bar formulations. Hydrotropes such as sodium xylene sulfonate enhance solubility, while builders like sodium carbonate and zeolites sequester metal ions. Chelating agents such as ethylenediaminetetraacetic acid and citric acid prevent precipitation. Bleach components include hydrogen peroxide sources, peracids such as peroxycarboxylic acids, and bleach activators such as tetraacetylethylenediamine. Bleach catalysts, including manganese and cobalt complexes, accelerate peracid formation. Enzymes such as cellulases, proteases, and amylases are co-formulated for synergistic cleaning. The lipase variant may be incorporated into water-soluble films made of polyvinyl alcohol, with molecular weights between 10,000 and 150,000 Da, plasticized with glycerol or ethylene glycol, and dried under controlled humidity to preserve enzyme activity. The film may be formed into single or multi-compartment pouches using vertical or horizontal form-fill-seal equipment, with each compartment containing a different detergent component. The lipase variant is formulated as granules, crystals, or spray-dried particles with particle sizes between 50 and 500 micrometers and protein content of 5% to 30% by weight. The composition is stable at pH 7 to 11 and temperatures up to 60°C, and is suitable for use in laundry, dishwashing, and industrial cleaning applications.