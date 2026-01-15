# DESCRIPTION

## FIELD OF THE INVENTION

- define nucleic acid memory

Nucleic acid memory refers to a class of information storage systems that utilize synthetic or naturally derived nucleic acid molecules as physical media for the encoding, preservation, and retrieval of digital data. Unlike conventional electronic or magnetic storage devices, nucleic acid memory leverages the inherent chemical stability, extreme information density, and molecular specificity of DNA and related polymers to store binary information in the sequence or spatial arrangement of nucleotides. This form of memory is not dependent on electrical charge, magnetic domains, or optical pits, but instead encodes data through the presence, absence, or configuration of nucleotide strands within a structured molecular architecture. The system is designed for long-term archival applications where data integrity, environmental resilience, and minimal energy consumption are paramount. Nucleic acid memory operates independently of traditional computing architectures, enabling data to be stored in a chemically stable, non-volatile state for decades or longer under ambient conditions, and retrieved through molecular recognition, hybridization, or optical imaging techniques without requiring continuous power. The architecture of such memory systems may involve linear strands, self-assembled nanostructures, or spatially organized arrays, each capable of hosting multiple bits of information per molecular unit through combinatorial encoding schemes. This invention specifically defines nucleic acid memory as a platform wherein digital information is embedded not merely in the sequence of nucleotides, but in the three-dimensional spatial organization of oligonucleotide elements relative to a programmable scaffold, enabling high-density, addressable, and error-resilient data storage through direct optical interrogation.

## BACKGROUND OF THE INVENTION

- motivate archival memory materials

The exponential growth of global digital data has outpaced the capacity and sustainability of existing archival storage technologies, creating an urgent need for alternative materials capable of preserving vast quantities of information over centuries with minimal energy expenditure and physical footprint. Magnetic tape, the current industry standard for long-term data archiving, suffers from inherent limitations including mechanical degradation, susceptibility to electromagnetic interference, limited areal density, and the need for climate-controlled environments to maintain data integrity over time. As data volumes continue to expand at rates exceeding Moore’s Law, the energy cost of maintaining data centers, the physical space required for storage arrays, and the frequency of media replacement have become economically and environmentally unsustainable. The development of a new class of archival materials that can store petabytes of data in a gram-scale volume, remain stable under ambient conditions for millennia, and require no active power for preservation is therefore not merely advantageous but essential for the future of digital heritage, scientific recordkeeping, and institutional data retention.

- describe limitations of magnetic tape

Magnetic tape, while widely adopted for offline archival storage, is fundamentally constrained by its two-dimensional data density, which has approached physical limits due to grain size restrictions and signal-to-noise degradation at high track densities. Its reliance on mechanical read/write heads introduces latency, wear, and vulnerability to physical damage from handling, vibration, or environmental humidity. Furthermore, magnetic media require periodic migration to newer formats to avoid obsolescence, a process that is both costly and error-prone. Data retention on magnetic tape is typically guaranteed for only 10 to 30 years under ideal conditions, necessitating frequent duplication and re-storage, which compounds energy consumption and operational overhead. The infrastructure required to support tape libraries—including robotic arms, climate-controlled vaults, and specialized readers—adds significant capital and maintenance burdens. These limitations render magnetic tape increasingly inadequate for the long-term, low-maintenance, high-density storage demands of modern data archiving.

- introduce DNA as alternative

Deoxyribonucleic acid (DNA) presents a compelling alternative to conventional storage media due to its unparalleled information density, chemical durability, and compatibility with biological systems. A single gram of DNA can theoretically store up to 215 petabytes of data, surpassing the areal density of any existing storage technology by several orders of magnitude. DNA is chemically stable under dry, cool, and dark conditions, with estimated half-lives of over a million years when protected from hydrolytic and oxidative degradation. Unlike magnetic or solid-state media, DNA does not degrade through repeated read cycles, requires no electrical power for storage, and is immune to electromagnetic pulses or data corruption from power loss. Moreover, DNA is a universal molecular language, compatible with existing biological synthesis and sequencing platforms, and can be replicated using enzymatic amplification, enabling scalable duplication without technological obsolescence. These properties position DNA as a uniquely suited medium for archival storage of critical data, including cultural, scientific, and governmental records intended for preservation across generations.

- summarize DNA nanotechnology

DNA nanotechnology has emerged as a powerful framework for engineering molecular-scale structures with precise geometric control, enabled by the predictable base-pairing rules of Watson-Crick complementarity. Through the design of synthetic oligonucleotides, researchers can program DNA to self-assemble into complex two- and three-dimensional architectures, including lattices, cages, and origami-like nanostructures, with nanometer-level precision. These structures are formed by hybridizing a long single-stranded scaffold with hundreds of short synthetic staple strands, each designed to bind specific regions of the scaffold and fold it into a predetermined shape. The resulting assemblies exhibit addressable surface sites, spatial regularity, and structural robustness, making them ideal scaffolds for organizing molecular components with sub-10-nanometer resolution. This capability has been leveraged to create molecular sensors, drug delivery vehicles, and molecular computers, but has not been fully exploited for digital data storage until the advent of systems that encode binary information not in sequence but in spatial configuration. DNA nanotechnology thus provides the structural foundation for a new generation of nucleic acid memory systems that combine molecular precision with digital encoding.

## BRIEF SUMMARY OF THE INVENTION

- introduce nucleic acid architectures

Nucleic acid architectures are precisely engineered molecular assemblies formed through the programmed hybridization of oligonucleotide strands into stable, predictable, and addressable nanostructures. These architectures are constructed using the principles of Watson-Crick base pairing to create defined geometries, such as two-dimensional lattices, three-dimensional polyhedra, or planar origami sheets, each composed of a scaffold strand and multiple staple strands that bind at specific locations to induce folding. The resulting structures possess spatially distinct, chemically unique sites that can be selectively modified to serve as physical representations of binary states, enabling the storage of digital information in the three-dimensional arrangement of molecular components rather than in the linear sequence of nucleotides. Such architectures are self-assembling, scalable, and compatible with high-throughput synthesis, making them ideal platforms for dense, stable, and retrievable data storage systems.

- describe data strands and addressable locations

Data strands are oligonucleotide sequences that are incorporated into nucleic acid architectures to encode binary information through their presence or absence at predefined, spatially fixed locations. Each data strand is designed with two functional domains: a structural domain that hybridizes with the scaffold to anchor the strand at a specific site within the architecture, and a data domain that extends beyond the surface of the structure and serves as a docking site for imager strands. The data domain may be present or absent, corresponding to a binary 1 or 0, respectively. The spatial arrangement of these data strands across the architecture forms an indexed matrix of binary values, where each position corresponds to a unique bit in the encoded message. The addressability of these locations is ensured by the precise positioning of staple strands during origami folding, allowing each data strand to be uniquely identified by its location within the nanostructure, independent of its nucleotide sequence.

- explain reading digital information

Digital information stored within nucleic acid architectures is read by optically detecting the presence or absence of data strands using super-resolution imaging techniques that resolve individual molecular binding events below the diffraction limit of light. Fluorescently labeled imager strands, complementary to the data domains of the data strands, are introduced into solution and transiently hybridize with their targets, producing brief, localized fluorescence signals known as blinking events. These events are captured over thousands of frames using a sensitive camera and localized with nanometer precision using computational algorithms. The spatial coordinates of each blinking event are mapped to the known geometry of the architecture, allowing the reconstruction of a binary matrix corresponding to the original data encoding. The absence of a signal at a predicted location indicates a binary 0, while the presence of a signal indicates a binary 1, enabling the direct optical retrieval of digital information without the need for nucleic acid sequencing.

- describe encoding binary data

Binary data is encoded by selecting, from a library of pre-synthesized staple strands, those that contain or omit a data domain extension at each addressable site within the nucleic acid architecture. Each site in the architecture corresponds to a single bit, and the selection of extended or unextended strands determines whether that bit is set to 1 or 0. The entire message is partitioned into overlapping data droplets, each assigned to a distinct nucleic acid architecture, with additional bits dedicated to indexing, orientation, parity, and checksum functions to enable error detection and correction. The encoding process is performed algorithmically, using fountain codes to generate redundant droplets and bi-level parity schemes to detect and correct bit errors within each architecture, ensuring that the complete message can be recovered even if some architectures are partially corrupted or lost during imaging.

- outline integrated memory platform

The integrated memory platform comprises a unified system for writing, storing, and reading digital information using nucleic acid architectures as the physical medium. Data is written by synthesizing and selecting specific staple strands to form architectures with desired data domain configurations. These architectures are then deposited onto a solid substrate, such as a glass coverslip, and stored in a dry, ambient environment indefinitely. To read the data, the substrate is imaged using super-resolution microscopy, and the resulting fluorescence signals are processed by a decoding algorithm that reconstructs the binary matrix, corrects errors using parity and checksum bits, and reassembles the original message from overlapping droplets. The platform is self-contained, requiring no continuous power for storage, and enables high-density, error-resilient, and durable data retention through molecular-scale encoding and optical retrieval.

- introduce error-correcting algorithms

Error-correcting algorithms are integral to the reliability of nucleic acid memory systems, compensating for imperfections in strand incorporation, fluorophore bleaching, background noise, and imaging artifacts that may corrupt individual bits during data retrieval. These algorithms employ a multi-layered approach combining fountain codes with bi-level parity and checksum encoding to detect and correct errors at both the local (per-architecture) and global (message-wide) levels. Fountain codes allow the message to be reconstructed from any sufficient subset of droplets, even if some are entirely missing, while parity and checksum bits enable the correction of bit-flip errors within each architecture. The combination of these methods ensures that the complete message can be recovered with near-perfect fidelity even when a significant proportion of the encoded structures contain multiple errors.

- summarize error correction methods

Error correction is achieved through a hierarchical algorithm that first evaluates each nucleic acid architecture for inconsistencies between the expected and observed parity and checksum values, assigns error weights to each bit based on its contribution to these discrepancies, and performs a greedy search to flip the most probable erroneous bits until the parity and checksum constraints are satisfied. If the architecture cannot be fully corrected, it is passed to the fountain code decoder, which reconstructs the original message by combining data from multiple partially corrupted droplets. The method is robust to high error rates, capable of recovering the full message even when up to 70% of the architectures contain more than five errors each, and requires no prior knowledge of the data content to perform decoding, making it suitable for blind, automated retrieval in archival applications.

## DETAILED DESCRIPTION

- define conventions for technical terms

### Definitions

- define singular and plural terms

In this disclosure, singular terms encompass their plural forms unless explicitly stated otherwise. For example, reference to a “strand” includes one or more strands, and reference to a “nucleotide” includes one or more nucleotides. The use of a singular noun in the context of a system, method, or apparatus is intended to cover both individual and collective instances unless the context clearly requires otherwise.

- define "or" and "and/or"

The term “or” is used in its inclusive sense, meaning that one or more of the listed alternatives may be selected. The term “and/or” is synonymous with “or” and is used interchangeably to indicate that any combination of the listed elements may be present, including all elements simultaneously. For example, “A or B” and “A and/or B” both encompass the possibilities of A alone, B alone, or both A and B together.

- define numeric ranges

Numeric ranges disclosed herein, such as “between 1 and 10,” include all integers and non-integers within the stated bounds, unless otherwise specified. Ranges are inclusive of the endpoints unless explicitly stated to be exclusive. For example, a range of “1 to 10” includes 1, 10, and all values in between, including fractional values such as 5.5 or 9.2.

- define "about"

The term “about” when used in conjunction with a numerical value indicates that the stated value may vary by ±10% unless otherwise indicated. This variation accounts for experimental tolerances, measurement uncertainty, and minor deviations inherent in biological or chemical synthesis processes. For example, “about 10 nm” encompasses values from 9 nm to 11 nm.

- define "non-covalent" and "covalent"

A non-covalent interaction refers to a reversible molecular association formed through hydrogen bonding, van der Waals forces, electrostatic attraction, or hydrophobic effects, without the sharing of electrons between atoms. A covalent interaction refers to a chemical bond formed by the sharing of electron pairs between atoms, resulting in a stable, irreversible linkage under physiological conditions. In this invention, nucleic acid strands are primarily held together by non-covalent interactions, while modifications such as fluorophore conjugation may involve covalent attachment.

- define "structural strand"

A structural strand is an oligonucleotide sequence that hybridizes to a scaffold strand to induce the folding of a nucleic acid architecture, but does not contain an extended data domain. Structural strands serve to define the overall geometry and topology of the architecture without contributing directly to the encoding of binary data.

- define "brick" and "nucleotide brick"

A brick is a modular, pre-designed nucleic acid unit that contains a defined sequence of nucleotides and is capable of self-assembling with other bricks to form larger architectures. A nucleotide brick is a synthetic oligonucleotide strand designed with specific binding domains that enable it to hybridize with complementary bricks in a predictable spatial orientation, forming a repeating structural motif within a larger assembly.

- define "nucleotide"

A nucleotide is a molecular subunit of nucleic acids, consisting of a nitrogenous base, a five-carbon sugar, and one or more phosphate groups. In this invention, nucleotides include naturally occurring deoxyadenosine, deoxyguanosine, deoxycytidine, and deoxythymidine, as well as synthetic analogs modified for enhanced stability, fluorescence, or hybridization kinetics.

- define "nucleotide duplex"

A nucleotide duplex is a double-stranded structure formed by the complementary base pairing of two oligonucleotide strands through Watson-Crick hydrogen bonding. The duplex may be fully complementary or partially mismatched, and may be stabilized by magnesium ions or other cations to enhance thermal stability.

- define "nucleotide origami" and "origami"

Nucleotide origami refers to a self-assembled nanostructure formed by folding a long single-stranded DNA scaffold into a predetermined shape using hundreds of short staple strands. Origami is a shorthand term for nucleotide origami and denotes any such architecture created through the programmed hybridization of scaffold and staple strands to form a stable, addressable, two- or three-dimensional structure.

- define "scaffold"

A scaffold is a long, single-stranded nucleic acid molecule, typically derived from a viral genome, that serves as the backbone for the assembly of a nucleic acid architecture. The scaffold is hybridized with multiple staple strands that bind at specific intervals to induce folding into a desired geometry.

- define "staple" and "staple strand"

A staple is a short, synthetic oligonucleotide strand designed to bind to multiple regions of a scaffold strand, thereby stabilizing specific folds and defining the final architecture. A staple strand is a single molecule of such a staple, and may be modified to include a data domain extension for encoding binary information.

- define "nanobreadboard", "breadboard", "substrate", and "template"

A nanobreadboard is a solid surface, such as a glass coverslip, upon which nucleic acid architectures are deposited for imaging or further processing. A breadboard is a general term referring to any substrate that supports the immobilization of molecular structures. A substrate is a solid support material, such as glass, silicon, or polymer, that facilitates the spatial organization of architectures for optical interrogation. A template is a pre-defined geometric pattern or reference structure used to align and interpret the spatial arrangement of data domains within an architecture.

- define "architecture" and "nucleic acid architecture"

An architecture is a three-dimensional molecular structure formed through the programmed self-assembly of nucleic acid strands. A nucleic acid architecture is an architecture composed entirely or predominantly of DNA or RNA molecules, designed to exhibit a stable, reproducible, and addressable geometry suitable for the encoding and retrieval of digital information.

- define "self-assembly"

Self-assembly is the spontaneous organization of molecular components into a defined structure through non-covalent interactions, without external direction beyond the initial design of the components. In this invention, nucleic acid architectures self-assemble upon mixing scaffold and staple strands under controlled thermal conditions.

- define FRET, RET, and EET

FRET refers to Förster Resonance Energy Transfer, a non-radiative energy transfer between two fluorophores in close proximity. RET refers to Resonance Energy Transfer, a broader term encompassing FRET and other mechanisms of energy transfer between chromophores. EET refers to Excitation Energy Transfer, a process in which excitation energy is transferred from one chromophore to another without emission of a photon. These phenomena are not utilized in this invention for data encoding but may be employed in related systems for signal amplification or multiplexing.

- define "dye", "chromophore", and "fluorophore"

A dye is a chemical compound capable of absorbing and emitting light at specific wavelengths. A chromophore is the part of a molecule responsible for its color and light absorption properties. A fluorophore is a chromophore that emits light upon excitation, and is used in this invention to label imager strands for super-resolution imaging.

- define "indexed array"

An indexed array is a spatially organized matrix of data sites, each associated with a unique identifier that enables the reconstruction of the data’s geometric context. In this invention, each nucleic acid architecture contains an index bit sequence that identifies its position within the overall data set, allowing for the correct assembly of droplets during decoding.

- define "archival storage", "long-term storage", and "stable storage"

Archival storage refers to the preservation of data for extended periods, typically decades or longer, with minimal intervention. Long-term storage denotes data retention over time scales exceeding the operational lifespan of conventional media. Stable storage refers to data retention under ambient environmental conditions without the need for refrigeration, vacuum sealing, or active power. This invention provides a system for archival, long-term, and stable storage of digital information using nucleic acid architectures.

- define "binary string"

A binary string is a sequence of bits, each having a value of 0 or 1, representing digital information. In this invention, binary strings are encoded into the spatial configuration of data strands within nucleic acid architectures.

- define "bit"

A bit is the smallest unit of digital information, represented by a binary value of 0 or 1. In this invention, each bit corresponds to the presence or absence of a data strand at a specific location within a nucleic acid architecture.

- define "byte"

A byte is a unit of digital information consisting of eight bits. In this invention, bytes are encoded as groups of eight bits within a data droplet, corresponding to a single ASCII character or other data unit.

- define "checksum bit"

A checksum bit is a bit derived from a mathematical function applied to a set of data bits, used to verify the integrity of the data during retrieval. In this invention, checksum bits are computed over subsets of data and parity bits to detect and correct errors within each architecture.

- define "data bit"

A data bit is a bit that encodes a portion of the original digital message, as opposed to a parity, index, or checksum bit. In this invention, data bits are encoded by the presence or absence of extended staple strands at designated positions within the architecture.

- define "data strand" and "information-bearing particles"

A data strand is an oligonucleotide strand that contains a data domain extension, enabling it to encode a binary value through its presence or absence. Information-bearing particles refer to nucleic acid architectures that contain one or more data strands and serve as physical carriers of digital information.

- define "decoding algorithm"

A decoding algorithm is a computational procedure that reconstructs digital information from raw imaging data by identifying spatial patterns, correcting errors using parity and checksum bits, and reassembling overlapping data droplets into the original message. In this invention, the decoding algorithm operates without prior knowledge of the encoded data and is capable of recovering the full message even when a majority of the architectures contain multiple errors.

### Nucleic Acid Architecture

- introduce nucleic acid architecture

Nucleic acid architectures are self-assembled nanostructures formed through the precise hybridization of a long scaffold strand with multiple short staple strands, resulting in stable, addressable, and geometrically defined forms such as rectangles, grids, or three-dimensional polyhedra. These architectures are designed with spatially fixed sites where staple strands may be extended or left unmodified, enabling the encoding of binary data through molecular topology rather than sequence.

- describe Watson-Crick pairing

Watson-Crick pairing refers to the specific hydrogen bonding between complementary nucleobases—adenine with thymine (or uracil in RNA), and guanine with cytosine—that governs the formation of double-stranded nucleic acid structures. This pairing is the fundamental mechanism by which scaffold and staple strands associate to fold nucleic acid architectures with predictable geometry.

- list methods of designing architectures

Architectures are designed using computational software that simulates strand hybridization, predicts folding pathways, and optimizes staple sequences to minimize off-target binding. Design methods include grid-based modeling, sequence optimization algorithms, and thermodynamic modeling to ensure efficient folding and structural stability.

- describe nucleobase composition

The nucleobase composition of the nucleic acid strands consists of the four canonical bases—adenine, thymine, guanine, and cytosine—arranged in sequences that maximize hybridization specificity and minimize secondary structure formation. Synthetic analogs may be incorporated to enhance stability or fluorescence properties.

- list natural nucleobases

Natural nucleobases include adenine, thymine, guanine, and cytosine, each capable of forming specific Watson-Crick base pairs with their complementary partners.

- list synthetic nucleobases

Synthetic nucleobases include, but are not limited to, 2-aminoadenine, 5-methylcytosine, inosine, and dU, which may be incorporated to modulate hybridization kinetics, increase thermal stability, or reduce enzymatic degradation.

- describe nucleotide analogs

Nucleotide analogs are chemically modified nucleotides in which the sugar, phosphate, or base moiety has been altered to enhance properties such as nuclease resistance, fluorescence, or binding affinity. Examples include locked nucleic acids (LNAs), peptide nucleic acids (PNAs), and 2′-O-methyl RNA.

- list examples of nucleotide analogs

Examples of nucleotide analogs include locked nucleic acid (LNA), 2′-fluoro RNA, 2′-O-methyl RNA, phosphorothioate backbone modifications, and 5-propynyl uracil.

- describe polymerization of nucleotides

Polymerization of nucleotides occurs enzymatically using DNA polymerases or chemically via solid-phase synthesis to produce oligonucleotides of defined length and sequence. In this invention, strands are synthesized chemically using automated phosphoramidite chemistry.

- describe design of oligomers

Oligomers are designed with specific lengths and sequences to ensure selective hybridization to the scaffold and to minimize cross-talk between adjacent binding sites. Design parameters include melting temperature, GC content, and avoidance of self-complementarity.

- introduce software for designing architectures

Software tools such as caDNAno, CanDo, and oxDNA are used to model the three-dimensional structure of nucleic acid architectures, simulate folding pathways, and generate staple strand sequences for synthesis.

- describe nucleotide brick molecular canvases

Nucleotide brick molecular canvases are modular, repeating units of nucleic acid architecture that serve as building blocks for larger assemblies. Each brick contains a defined set of binding interfaces that allow it to associate with neighboring bricks in a predictable orientation.

- describe single-pot synthesis

Single-pot synthesis refers to the simultaneous production of multiple oligonucleotide strands in a single reaction vessel, enabling high-throughput, cost-effective fabrication of staple libraries for nucleic acid architecture assembly.

- describe serial fluidic flow assembly

Serial fluidic flow assembly is a method of sequentially introducing staple strands into a reaction chamber under controlled flow conditions to facilitate stepwise folding of complex architectures with minimal misfolding.

- introduce origami approach

The origami approach involves the folding of a long single-stranded DNA scaffold into a desired shape using hundreds of short staple strands, each designed to bind to multiple regions of the scaffold and induce precise bends and twists.

- describe scaffold nucleic acid strand

The scaffold nucleic acid strand is a long, single-stranded DNA molecule, typically derived from the M13 bacteriophage genome, that serves as the structural backbone of the origami. Its length and sequence are chosen to accommodate the desired number of staple binding sites.

- describe staple strands

Staple strands are short, synthetic oligonucleotides that bind to specific regions of the scaffold strand, holding it in a folded conformation. Each staple strand is designed to hybridize with two or more non-contiguous regions of the scaffold, thereby stabilizing the overall architecture.

- describe single-stranded tiles (SSTs)

Single-stranded tiles are short DNA strands that self-assemble into larger lattices through complementary end-to-end hybridization, forming repetitive, two-dimensional patterns useful for high-density data encoding.

- describe nucleic acid bricks

Nucleic acid bricks are modular, pre-designed oligonucleotide units that contain multiple binding domains and can be assembled into larger architectures through programmed hybridization, enabling scalable and reproducible fabrication of complex structures.

- describe addressability of nanostructures

Addressability refers to the ability to uniquely identify and access individual molecular sites within a nanostructure. In this invention, each data strand is positioned at a fixed coordinate within the architecture, allowing its presence or absence to be mapped to a specific bit location.

- describe single-stranded oligomers

Single-stranded oligomers are short, linear nucleic acid sequences that serve as either scaffold strands, staple strands, or imager strands. In this invention, they are synthesized with defined lengths and sequences to ensure specific hybridization and minimal secondary structure.

- describe length of single-stranded DNA

The length of single-stranded DNA used in this invention ranges from 20 to 100 nucleotides for staple strands, and from 7,000 to 8,000 nucleotides for scaffold strands, optimized for folding efficiency and structural stability.

- describe two-dimensional architectures

Two-dimensional architectures are planar nucleic acid structures, such as rectangular grids or hexagonal lattices, formed by the folding of a scaffold strand into a flat sheet. These architectures are particularly suited for high-density data encoding due to their uniform surface topology and ease of imaging.

- describe three-dimensional architectures

Three-dimensional architectures are folded structures with volume, such as cubes, tetrahedra, or toroids, formed by introducing additional staple strands that induce curvature and stacking. These architectures offer increased surface area and potential for multiplexed encoding but require more complex design and folding protocols.

- describe attachment to substrates

Nucleic acid architectures are attached to solid substrates such as glass coverslips via electrostatic interactions, chemical functionalization, or physical adsorption, ensuring immobilization during imaging while preserving accessibility of data domains to imager strands.

- introduce dNAM origami

dNAM origami refers to a nucleic acid architecture specifically designed to encode digital information through the spatial arrangement of extended staple strands, enabling optical readout via super-resolution microscopy without sequencing.

- describe data strands and imager strands

Data strands are staple strands extended with a sequence complementary to a fluorescently labeled imager strand. Imager strands are short, dye-labeled oligonucleotides that transiently hybridize with data strands, producing localized fluorescence signals that are detected and localized to reconstruct the binary data matrix.

- describe data density and encoding

Data density is determined by the spacing between data domains, which in this invention is approximately 10 nanometers, allowing for approximately 1000 bits per square micrometer. Encoding is achieved by selecting extended or unextended staple strands to represent binary 1s and 0s, respectively, across a matrix of addressable sites.

### Dyes

- introduce dyes

Dyes are organic or inorganic molecules capable of absorbing and emitting light at specific wavelengths, used in this invention to label imager strands for detection via super-resolution microscopy.

- describe chromophores

Chromophores are molecular moieties responsible for light absorption and color, and in this invention, they are covalently attached to imager strands to enable fluorescence-based detection of hybridization events.

- describe fluorophores

Fluorophores are chromophores that emit light upon excitation, and in this invention, they are selected for high photostability, brightness, and blinking kinetics compatible with DNA-PAINT imaging.

- bind dyes to imager strands

Dyes are covalently conjugated to the 5′ or 3′ terminus of imager strands during solid-phase synthesis, ensuring consistent labeling efficiency and minimal interference with hybridization.

- single dye bound to imager strand

Each imager strand carries a single dye molecule, ensuring that each hybridization event produces a single, resolvable fluorescence signal.

- multiple dyes bound at multiple turns

Multiple dyes may be attached to different positions along a single imager strand to enhance signal intensity, though in this invention, single-dye labeling is preferred to ensure unambiguous localization.

- dyes same within dNAM origami

All imager strands used within a single dNAM origami experiment are labeled with the same dye to ensure uniform excitation and detection characteristics.

- dyes multiplexed using orthogonal binding sequences

Multiple dyes may be used simultaneously by pairing each with a unique imager strand sequence that binds only to a specific data domain, enabling multiplexed encoding of multiple bits per site.

- increase data density using multiplexing

Multiplexing increases data density by allowing multiple bits to be encoded at a single spatial location through the use of orthogonal binding sequences and spectrally distinct dyes.

- list xanthene derivatives

Xanthene derivatives include fluorescein, rhodamine, and tetramethylrhodamine.

- list cyanine derivatives

Cyanine derivatives include Cy3, Cy5, Cy7, and IR-786.

- list squaraine derivatives

Squaraine derivatives include SQ-1, SQ-2, and SQ-3.

- list naphthalene derivatives

Naphthalene derivatives include napthofluorescein and naphthylamine.

- list coumarin derivatives

Coumarin derivatives include 7-hydroxycoumarin and 4-methylcoumarin.

- list oxadiazole derivatives

Oxadiazole derivatives include 2,5-diphenyloxadiazole and 3,4-diphenyl-1,2,5-oxadiazole.

- list anthracene derivatives

Anthracene derivatives include 9,10-diphenylanthracene and 9-cyanoanthracene.

- list pyrene derivatives

Pyrene derivatives include pyrene-1-carboxylic acid and 1-pyrenemethanol.

- list oxazine derivatives

Oxazine derivatives include Nile Red and Oxazine 170.

- list acridine derivatives

Acridine derivatives include acridine orange and proflavine.

- list arylmethine derivatives

Arylmethine derivatives include malachite green and crystal violet.

- list tetrapyrrole derivatives

Tetrapyrrole derivatives include porphyrin, chlorophyll, and bilirubin.

- list dipyrromethene derivatives

Dipyrromethene derivatives include BODIPY and boron-dipyrromethene.

- list commercial dyes

Commercial dyes include Alexa Fluor, ATTO, LI-COR IRDyes, Dyomic, and WellRED.

- list Freedom Dyes

Freedom Dyes are a proprietary class of fluorophores optimized for DNA-PAINT imaging with high blinking rates and low background.

- list Alexa Fluor Dyes

Alexa Fluor Dyes include Alexa Fluor 488, Alexa Fluor 555, and Alexa Fluor 647.

- list LI-COR IRDyes

LI-COR IRDyes include IRDye 680 and IRDye 800CW.

- list ATTO Dyes

ATTO Dyes include ATTO 488, ATTO 550, and ATTO 647N.

- list Rhodamine Dyes

Rhodamine Dyes include Rhodamine 6G, Rhodamine B, and Rhodamine 101.

- list WellRED Dyes

WellRED Dyes include WellRED 570 and WellRED 670.

- list Dyomic Dyes

Dyomic Dyes include Dyomic 488, Dyomic 594, and Dyomic 670.

- describe dye modifications

Dyes may be modified with polyethylene glycol chains, hydrophilic groups, or charged moieties to enhance solubility, reduce aggregation, and improve binding kinetics.

- adjust solubility

Solubility is adjusted by attaching hydrophilic linkers such as PEG or charged amino groups to the dye molecule to prevent nonspecific adsorption to surfaces.

- adjust hydrophobicity

Hydrophobicity is reduced by incorporating polar functional groups to prevent dye aggregation and improve dispersion in aqueous buffers.

- adjust symmetry

Symmetry is modified to alter the dipole moment and fluorescence quantum yield, enhancing signal intensity and photostability.

- adjust placement of dye

Placement of the dye is optimized to minimize steric hindrance with hybridization and to ensure that the dye is positioned away from the nucleic acid backbone to reduce quenching.

- motivate use of dyes

Dyes enable the optical detection of individual hybridization events at the nanoscale, allowing for the direct, non-destructive readout of binary data without sequencing.

- describe data density increase

The use of dyes enables the detection of single-molecule binding events, allowing data to be encoded at sub-10-nanometer spacing, significantly increasing areal data density compared to bulk optical methods.

- describe multiplexing

Multiplexing allows multiple bits to be encoded at a single spatial location by using orthogonal imager strands labeled with spectrally distinct dyes, each binding to a unique data domain sequence.

- describe binding additional dyes

Additional dyes may be bound to the same imager strand or to separate strands to increase signal intensity or enable multi-bit encoding per site.

- describe use of multiple chromophores

Multiple chromophores may be employed to encode additional information through spectral signatures, enabling higher-order bit encoding beyond binary.

- describe use of orthogonal binding sequences

Orthogonal binding sequences are designed to bind only to their intended target and not to any other sequence, ensuring specificity and minimizing cross-talk between data domains.

- describe data storage on dNAM origami

Data storage on dNAM origami is achieved by encoding binary states as the presence or absence of extended staple strands at addressable locations, with each state read via transient hybridization of dye-labeled imager strands.

- describe data density increase

By placing data domains at 10-nanometer intervals and using super-resolution imaging, data density exceeds 300 Gbit/cm², surpassing magnetic tape by an order of magnitude.

- describe use of dyes in data storage

Dyes serve as the optical transducers that convert molecular hybridization events into detectable signals, enabling the non-destructive, high-resolution readout of stored data.

- describe use of dyes in dNAM architecture

In dNAM architecture, dyes are used exclusively on imager strands to label data domains, ensuring that the architecture itself remains unmodified and reusable for multiple read cycles.

- describe use of dyes in NAM architecture

In general nucleic acid memory architectures, dyes may be used to label either the data strands or auxiliary imager strands to enable optical retrieval of encoded information.

- describe use of dyes in data encoding

Dyes do not encode data directly but enable the detection of data strand presence or absence, thereby facilitating the conversion of molecular topology into binary information.

- describe use of dyes in data decoding

During decoding, dye signals are localized and mapped to known architecture geometries to reconstruct the binary matrix, which is then processed by error-correcting algorithms to recover the original message.

- describe use of dyes in data recovery

Dyes enable the recovery of data even when individual strands are partially degraded or misincorporated, as long as sufficient signals remain to reconstruct the binary pattern.

- describe use of dyes in NAM and dNAM

In both nucleic acid memory and digital nucleic acid memory systems, dyes serve as the critical interface between molecular storage and optical readout, enabling high-fidelity, non-sequencing-based data retrieval.

- introduce nucleotides as storage media

Nucleotides serve as the fundamental storage medium in this invention, with digital information encoded in the spatial configuration of oligonucleotide strands rather than their sequence.

- describe environmental benefits

The use of nucleotides as a storage medium offers significant environmental benefits, including low energy consumption during storage, biodegradability under natural conditions, and elimination of toxic heavy metals and rare earth elements used in conventional storage media.

- motivate biodegradable nature

The biodegradable nature of nucleic acids ensures that stored data does not contribute to long-term electronic waste, and that archived media can safely decompose after their useful life.

- describe protection methods

Protection methods include encapsulation within silica nanoparticles, coating with polymer films, or embedding in inert matrices to shield nucleic acid architectures from hydrolytic and enzymatic degradation.

- introduce silica nanoparticles

Silica nanoparticles provide a protective shell around nucleic acid architectures, enhancing stability under ambient conditions and preventing aggregation or surface adsorption.

- describe stability at ambient temperatures

Nucleic acid architectures stored in silica-coated form remain stable for years at ambient temperatures, with minimal degradation of hybridization capacity or fluorescence signal.

- differentiate from other nucleotide storage systems

Unlike prior nucleic acid storage systems that rely on sequencing for data retrieval, this invention decouples storage from sequencing by encoding data in spatial architecture and retrieving it optically.

- describe data encryption

Data encryption is achieved by encoding information in non-obvious spatial patterns, requiring knowledge of the architecture design and decoding algorithm to reconstruct the message.

- describe miniaturization possibilities

The molecular-scale nature of nucleic acid architectures enables miniaturization down to the nanometer level, allowing for storage densities far exceeding those of conventional media.

- introduce product tracking

Product tracking is enabled by embedding unique digital identifiers into nucleic acid architectures, which can be read optically to verify authenticity, provenance, or expiration.

- describe regulatory approval labeling

Regulatory approval labels may be encoded into nucleic acid architectures as machine-readable codes, allowing for non-invasive verification of compliance without altering physical packaging.

- differentiate from current nucleotide strands

Current nucleotide storage systems encode data in sequence and require sequencing for retrieval; this invention encodes data in spatial configuration and retrieves it via optical imaging.

- introduce temporary or short-term data storage

This system may also be adapted for temporary or short-term storage by using thermally labile nucleotide modifications that degrade predictably over time, enabling time-sensitive data retention.

- describe heat sensitive nature

Certain nucleotide analogs and dye-linker chemistries are engineered to degrade at elevated temperatures, allowing for time-limited data retention suitable for expiration-based applications.

- describe degradation detection

Degradation is detected by reduced signal intensity, increased background noise, or failure to reconstruct the expected binary matrix, indicating expiration or tampering.

- describe age detection

Age detection is achieved by measuring the cumulative degradation of signal fidelity, which correlates with elapsed time under defined environmental conditions.

- introduce sugar modification

Sugar modifications, such as 2′-O-methyl or locked nucleic acid (LNA) substitutions, are introduced to tune the thermal stability and degradation kinetics of the nucleic acid strands.

- describe stability tuning

Stability is tuned by varying the degree of sugar modification, phosphate backbone alteration, or nucleobase substitution to achieve desired retention times under ambient or accelerated conditions.

- introduce NAM and dNAM systems

NAM and dNAM systems refer to nucleic acid memory and digital nucleic acid memory platforms, respectively, wherein data is encoded in either sequence or spatial architecture, with dNAM specifically utilizing addressable nanostructures for optical readout.

- describe data encoding

Data encoding is performed algorithmically by mapping binary strings to the presence or absence of extended staple strands at predefined locations within a nucleic acid architecture.

- describe device and computer processor configuration

The system includes a computer processor configured to execute machine-readable instructions for generating encoding algorithms, processing imaging data, and performing error correction, with memory units storing the decoding software and architecture templates.

- describe additional tasks

Additional tasks include image alignment, drift correction, localization clustering, and binary matrix reconstruction, all performed by software running on a general-purpose or specialized computing system.

- introduce data reading system

The data reading system comprises a super-resolution microscope, a high-sensitivity camera, a laser excitation source, and a computer processor configured to analyze blinking events and reconstruct binary data.

- describe microscope and computer processor configuration

The microscope is configured in total internal reflection fluorescence (TIRF) mode with an oil-immersion objective and an EMCCD camera, and is coupled to a computer processor with sufficient memory and processing power to handle thousands of image frames and perform real-time localization.

- describe image capture and data strand identification

Image capture involves recording tens of thousands of frames to detect transient hybridization events, and data strand identification is achieved by clustering localized signals and mapping them to the known geometry of the architecture.

- describe symbol generation and compilation

Symbol generation refers to the conversion of localized fluorescence events into binary values, and compilation refers to the aggregation of these values into a complete binary matrix for decoding.

- describe nucleotide synthesis device selection

Nucleotide synthesis devices are selected based on throughput, fidelity, and cost, with automated phosphoramidite synthesizers preferred for large-scale staple strand production.

- describe microscopy options

Microscopy options include DNA-PAINT, STORM, PALM, and SIM, with DNA-PAINT preferred for its high resolution, low background, and compatibility with transient hybridization.

- introduce information storage applications

Information storage applications include archival storage of scientific data, cultural heritage records, government documents, and cryptographic keys, where long-term integrity and low energy are critical.

- introduce computer systems using NAM or dNAM

Computer systems using NAM or dNAM include hybrid architectures where nucleic acid memory serves as a non-volatile storage tier, accessed via optical readout and interfaced with electronic processors for data retrieval.

- describe computer system components

Computer system components include a central processing unit, memory units, input/output interfaces, communication modules, and a nucleic acid memory reader subsystem.

- describe CPU and memory interaction

The CPU executes decoding algorithms to retrieve data from nucleic acid memory, transferring recovered information to volatile memory for processing, and storing results in electronic storage media.

- describe communication interface and peripheral devices

Communication interfaces enable data transfer between the nucleic acid memory system and external networks, while peripheral devices include robotic sample handlers, environmental sensors, and calibration standards.

- describe network and distributed computing

Network and distributed computing enable multiple nucleic acid memory repositories to be accessed remotely, with data retrieval coordinated across geographically dispersed storage nodes.

- describe CPU execution of machine-readable instructions

The CPU executes machine-readable instructions stored in non-transitory memory to perform encoding, decoding, error correction, and data management tasks.

- describe storage unit and data storage

The storage unit contains the physical nucleic acid architectures deposited on substrates, and data storage refers to the retention of binary information in their spatial configuration.

- describe remote computer system communication

Remote communication allows for the transmission of encoding parameters, architecture templates, and decoding algorithms to remote nucleic acid memory systems for data retrieval.

- describe machine-executable code and programming language

Machine-executable code is written in high-level programming languages such as Python, C++, or MATLAB, compiled into binary instructions executable by the processor.

- describe electronic storage medium and machine-readable medium

Electronic storage medium refers to non-volatile memory such as SSDs or hard drives, and machine-readable medium includes any physical or digital medium capable of storing instructions for execution by a processor.

- describe tangible and intangible storage media

Tangible storage media include glass coverslips with deposited architectures, while intangible storage media refer to digital copies of encoding parameters and decoding algorithms.

- describe user interface and output data

The user interface allows for input of data to be encoded, selection of architecture parameters, and visualization of recovered data, with output provided as text, files, or digital records.

- describe algorithm implementation

Algorithm implementation includes the software-based execution of fountain codes, parity checks, checksum validation, and greedy error correction to reconstruct the original message.

- describe software execution

Software execution occurs on a general-purpose computer system, with modules for image processing, localization, matrix reconstruction, and message decoding running in sequence.

- introduce experimental examples

Experimental examples demonstrate the encoding, storage, and retrieval of digital messages using nucleic acid architectures under controlled laboratory conditions.

## EXAMPLES

- introduce dNAM concept

The dNAM concept is a digital nucleic acid memory system in which binary data is encoded in the spatial configuration of extended staple strands within DNA origami nanostructures, and retrieved via super-resolution optical imaging without sequencing.

### Example 1

- introduce dNAM approach

The dNAM approach utilizes DNA origami as a programmable substrate for the spatial encoding of binary information, where each bit is represented by the presence or absence of a data domain extension on a staple strand.

- describe DNA origami nanostructures

DNA origami nanostructures are rectangular, two-dimensional lattices approximately 90 by 70 nanometers in size, formed by folding a 7,000-nucleotide scaffold with hundreds of staple strands, each designed to bind at specific locations.

- explain binary states definition

Binary states are defined by the presence (1) or absence (0) of a 10-nucleotide data domain extension on each staple strand, which serves as a docking site for a fluorescently labeled imager strand.

- describe data encoding process

The data encoding process converts a digital message into a binary string, partitions it into 16-bit droplets, and maps each droplet onto a unique origami design using a fountain code algorithm that generates 15 overlapping droplets.

- introduce error-correcting algorithms

Error-correcting algorithms combine fountain codes with bi-level parity and checksum encoding to detect and correct bit errors within each origami and recover the full message even when multiple droplets are corrupted.

- describe fountain codes

Fountain codes generate redundant data droplets by XORing random combinations of message segments, allowing the original message to be reconstructed from any sufficient subset of droplets.

- explain bi-level parity codes

Bi-level parity codes compute parity bits over both row and column subsets of the data matrix, enabling detection of single-bit errors and identification of likely error locations.

- describe error detection scheme

The error detection scheme calculates checksum values over subsets of data and parity bits, comparing them to stored values to identify inconsistencies that indicate bit flips.

- introduce dNAM-specific information encoding

dNAM-specific information encoding includes the allocation of bits for index, orientation, parity, and checksum functions, ensuring that each origami can be uniquely identified and its data corrected independently.

- describe decoding algorithm

The decoding algorithm first corrects errors within each origami using a greedy search based on parity and checksum weights, then uses fountain code decoding to reconstruct the original message from the corrected droplets.

- report prototype results

A prototype system successfully encoded and recovered the message “Data is in our DNA!” from 15 origami structures, achieving 100% recovery from a single imaging session despite an average of 7.3 errors per origami.

- describe message recovery process

The message recovery process involves imaging the origami mixture with DNA-PAINT, localizing fluorescence signals, mapping them to a binary matrix, correcting errors using parity and checksum bits, and reassembling the droplets using fountain code decoding.

- introduce quality control of dNAM

Quality control of dNAM is performed using atomic force microscopy to verify the structural integrity and correct placement of data domains on each origami design.

- describe automated image processing algorithms

Automated image processing algorithms align and classify individual origami structures, average multiple images to improve signal-to-noise, and identify misfolded or incomplete architectures.

- validate origami synthesis by AFM

Atomic force microscopy confirms that all 15 origami designs were synthesized with the expected honeycomb structure and properly positioned data domains.

- describe further AFM analysis

Further AFM analysis reveals that approximately 60% of origami are oriented with data domains facing the solution, enabling imager strand binding, while the remainder are adsorbed in orientations that hinder detection.

- investigate variance in error rates

Variance in error rates is investigated by resynthesizing the most error-prone origami design, revealing that error rates are consistent across batches and likely due to stochastic folding or purification conditions rather than design flaws.

- describe data encoding/decoding strategy

The data encoding/decoding strategy employs a 48-bit matrix per origami, with 16 bits for data, 4 for index, 4 for orientation, 4 for checksum, and 20 for parity, enabling robust recovery even with high error rates.

- evaluate decoding algorithm performance

The decoding algorithm successfully recovered the full message with 97.5% reliability at 7.4 errors per origami and 100% reliability with 14 or more correctly decoded droplets.

- describe template matching strategy

Template matching compares observed origami images to pre-defined design templates to identify the most likely match and quantify errors, though this method is not used for actual data recovery.

- determine error rates

Error rates are determined by comparing the binary matrix reconstructed from imaging to the expected matrix, counting false positives and false negatives per origami.

- describe error correction scheme

The error correction scheme assigns weights to each bit based on its contribution to parity and checksum violations, then performs a greedy search to flip the most probable erroneous bits until all constraints are satisfied.

- report message recovery results

Message recovery results show that the full message was recovered in all three independent experiments, even when as few as 750 of the 4,500 imaged origami were successfully decoded.

- describe sampling analysis of dNAM

Sampling analysis demonstrates that 750 successfully decoded origami are sufficient to recover the full message with 100% probability, given the redundancy provided by the fountain code.

- determine number of origami needed

The number of origami needed for full recovery is determined by random subsampling of decoded matrices and iterative decoding, revealing that 750 is the threshold for guaranteed recovery.

- describe simulations of dNAM

Simulations confirm that the system can recover messages up to 64 kilobytes in size, with recovery rates remaining above 95% even when 80% of origami contain more than five errors.

- evaluate encoding scheme efficiency

The encoding scheme achieves an areal data density of 330 Gbit/cm² after accounting for redundancy, surpassing magnetic tape by an order of magnitude.

- evaluate error correction algorithm

The error correction algorithm reduces the number of required origami for recovery by enabling correction of up to nine errors per architecture, significantly lowering the storage overhead.

- describe discussion of results

The results demonstrate that dNAM is a viable, scalable, and robust platform for archival data storage, combining the molecular density of DNA with the speed and non-destructiveness of optical readout.

- compare dNAM to magnetic tape

dNAM achieves an areal density of 330 Gbit/cm² compared to 31 Gbit/cm² for the most advanced magnetic tape, and offers superior durability under ambient conditions.

- discuss durability of DNA

DNA is estimated to remain stable for millions of years under optimal conditions, far exceeding the 10–30 year lifespan of magnetic tape.

- describe current limitations of dNAM

Current limitations include slow write speeds due to oligonucleotide synthesis, limited read throughput due to imaging time, and sensitivity to environmental noise.

- propose improvements to dNAM

Improvements include increasing origami size to encode more bits per structure, optimizing staple sequences to reduce misfolding, and integrating multiplexed dyes to increase bit depth.

- discuss coordinated effort for advancements

Advancements in dNAM will require coordinated efforts in DNA synthesis, imaging technology, algorithm development, and substrate engineering to scale to practical data capacities.

- describe fountain code algorithm

The fountain code algorithm generates droplets by XORing random combinations of message segments drawn from a Soliton distribution, ensuring optimal redundancy and recovery probability.

- explain robustness to lost packets

The algorithm is robust to lost packets because recovery requires only a sufficient number of droplets, not all, and the order of reception is irrelevant.

- discuss reducing error rates

Reducing error rates through improved synthesis and imaging will allow fewer bits to be allocated to redundancy, increasing information density.

- describe defect analysis

Defect analysis reveals that inactive but incorporated data strands contribute more to errors than unincorporated strands, suggesting that hybridization efficiency is a key bottleneck.

- propose sequence optimization

Sequence optimization involves redesigning staple strands to minimize secondary structure, improve binding kinetics, and reduce off-target hybridization.

- discuss use of larger DNA origami

Larger DNA origami would allow more data bits per structure, reducing the total number of origami needed for a given message and improving overall system efficiency.

- describe conclusion of dNAM

dNAM represents a paradigm shift in digital storage by decoupling data encoding from sequencing and enabling high-density, durable, and optically retrievable archival memory.

- summarize dNAM as a technology platform

dNAM is a scalable, non-volatile, and environmentally sustainable technology platform for long-term digital storage, combining the molecular precision of DNA nanotechnology with the speed and reliability of optical imaging and algorithmic error correction.

## Materials and Methods

- outline materials and methods

The materials and methods described herein include the synthesis of oligonucleotides, assembly of DNA origami, preparation of imaging substrates, performance of DNA-PAINT microscopy, and implementation of computational algorithms for data recovery.

### Buffers

- describe buffer composition

The deposition buffer consists of 0.5× TBE and 18 mM MgCl₂, while the imaging buffer contains the deposition buffer supplemented with 60 nM PCD, 1 mM Trolox, 3 nM imager strands, and 10 mM PCA.

### Encoding Algorithm

- introduce fountain code

The fountain code algorithm divides the input message into segments and generates droplets by XORing random combinations of segments, with the number of segments per droplet drawn from a Soliton distribution.

- describe droplet formation

Droplet formation involves selecting a random number of segments according to the Soliton distribution and combining them via XOR to form a single droplet, which is then encoded onto a unique origami.

- explain Soliton distribution

The Soliton distribution ensures that a sufficient number of single-segment droplets are generated to initiate the decoding process, while also producing multi-segment droplets to provide redundancy.

- detail matrix construction

Matrix construction involves mapping each droplet to a 6×8 grid on the origami, assigning positions for data, index, orientation, parity, and checksum bits according to a predefined layout.

- illustrate encoding process

The encoding process begins with ASCII conversion of the message, followed by segmentation, droplet generation, matrix assignment, and synthesis of corresponding staple strands.

### DNA Origami Folding

- describe origami design

Origami designs are generated using caDNAno software, with 48 staple binding sites arranged in a 6×8 grid, spaced 10 nm apart, and extended at positions corresponding to binary 1s.

- outline folding protocol

Folding is performed by mixing scaffold and staple strands in a 1:10 ratio in TAE buffer with 18 mM MgCl₂, heating to 90°C, and cooling slowly over 12 hours to 25°C.

### Glass Coverslip Preparation

- clean coverslips

Coverslips are sonicated in Liquinox and nano-pure water, then dried at 40°C for 30 minutes.

- deposit fiducial markers

Fiducial markers are deposited by incubating coverslips with 0.2 pM gold nanoparticles for 10 minutes.

- store coverslips

Coverslips are stored at 40°C until use to prevent contamination.

### Fluorescence Microscopy

- describe DNA-PAINT imaging

DNA-PAINT imaging is performed using a Nikon Eclipse Ti2 microscope in TIRF mode with a 100× oil-immersion objective and an EMCCD camera.

- outline image acquisition

Images are acquired over 40,000 frames at 300 ms exposure, with a 561 nm laser exciting Cy3B-labeled imager strands.

### DNA-PAINT Fluorophore Localization

- identify localization centers

Localization centers are identified using the ThunderSTORM plugin in ImageJ, which detects and fits individual blinking events to Gaussian point spread functions.

### Localization Data Processing

- select localization clusters

Clusters are selected by sampling random points and accepting those with high local density and low surrounding noise.

- fit clusters to grid

Clusters are fitted to a 6×8 grid using maximum likelihood estimation, optimizing position, rotation, and background intensity.

- define likelihood function

The likelihood function models the probability of observing localized signals given a grid of emitters with known intensities and background.

- optimize likelihood function

Optimization is performed using the L-BFGS-B algorithm to minimize the negative log-likelihood.

- filter signals

Signals not aligned to the grid are filtered out to eliminate noise from fiducial markers or misfolded origami.

- assign binary values

Binary values are assigned by comparing signal intensity to an empirically derived threshold.

- decode binary matrix data

The binary matrix is passed to the decoding algorithm for error correction and message reconstruction.

### Decoding Algorithm

- introduce error correction scheme

The error correction scheme computes parity and checksum values, assigns weights to bits based on their contribution to discrepancies, and performs a greedy search to correct errors.

- calculate parity and checksum values

Parity and checksum values are calculated over predefined subsets of the binary matrix using XOR and modular arithmetic.

- determine error weights

Error weights are calculated by summing the number of parity violations associated with each bit.

- calculate overall matrix weight

The overall matrix weight is the sum of all bit weights normalized by the number of correctly matched parity bits.

- perform greedy search

The greedy search iteratively flips the bit with the highest weight, recalculates the matrix weight, and repeats until no further improvement is possible or the maximum number of flips is reached.

- correct errors

Errors are corrected by flipping the selected bits and revalidating parity and checksum constraints.

- extract droplet and index data

Droplet and index data are extracted from the corrected matrix and passed to the fountain code decoder.

- decode fountain code

The fountain code decoder reconstructs the original message by iteratively XORing droplets to recover individual message segments.

- recover full message

The full message is recovered when all message segments are identified and concatenated.

## Data Simulation Test

- simulate origami data with random messages and errors

Random binary messages of 160 to 12,800 bits are generated, encoded into origami using the same algorithm as experimental data, and subjected to 0–9 random bit flips per origami to simulate imaging errors.

### Code Availability

- provide source codes for encoding, decoding, and localization algorithms

Source code for encoding, decoding, and localization algorithms is publicly available at a designated repository under an open-source license.

### AUTHOR CONTRIBUTIONS

- list author contributions to the study

All authors contributed to the design, execution, analysis, and writing of the study, with specific roles in algorithm development, microscopy, synthesis, and data interpretation.

### Supplemental Materials and Methods

- describe encoding/decoding algorithms

Supplemental materials detail the mathematical formulation of the fountain code, parity calculation, and greedy error correction algorithm.

- describe atomic force microscopy analysis

AFM analysis was performed using a Bruker Dimension Icon microscope in tapping mode, with images processed using Nanoscope Analysis software.

### Supplemental Results

- evaluate dna-paint resolution

DNA-PAINT resolution was evaluated at 5 nm, sufficient to resolve data domains spaced 10 nm apart.

- analyze proximity error

Proximity error was analyzed by measuring the distance between neighboring localizations and comparing to expected spacing.

- investigate error locations

Error locations were mapped to determine whether errors clustered at origami edges or were randomly distributed.

- perform atomic force microscopy imaging

AFM imaging confirmed the presence of data domains and the structural integrity of all 15 origami designs.

- analyze afm images

AFM images were quantified for height, shape, and domain placement, confirming design fidelity.

## Data Simulation Test

- simulate origami data with random messages and errors

Random binary messages of 160 to 12,800 bits were generated, encoded into origami using the same algorithm as experimental data, and subjected to 0–9 random bit flips per origami to simulate imaging errors. The decoding algorithm successfully recovered all messages with up to 7.4 errors per origami, demonstrating robustness to noise. Recovery rates dropped below 50% only when error rates exceeded 9 bits per origami, confirming the algorithm’s resilience under extreme conditions. Simulations confirmed that the system could store up to 64 kilobytes of data with current architecture dimensions, and that increasing origami size would linearly increase capacity. The simulation framework validated the theoretical limits of the encoding scheme and guided experimental design parameters.