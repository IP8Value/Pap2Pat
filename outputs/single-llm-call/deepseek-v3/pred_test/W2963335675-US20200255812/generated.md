Here is the patent application following the provided outline:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of biotechnology and enzymology, specifically to modified polynucleotides encoding lipases with improved properties. More particularly, the invention pertains to novel variants of Rhizomucor miehei lipase (RmL) comprising alterations in the propeptide region that confer enhanced enzymatic characteristics. The invention further encompasses methods for producing said lipase variants, nucleic acid constructs and expression vectors for their recombinant production, host cells transformed with such vectors, and industrial applications of the improved lipases in detergent compositions and other formulations requiring lipolytic activity.  

## BACKGROUND OF THE INVENTION  

Lipases (triacylglycerol acylhydrolases, EC 3.1.1.3) constitute a class of enzymes that catalyze the hydrolysis of ester bonds in triglycerides, playing crucial roles in lipid metabolism across biological systems. These enzymes belong to the α/β hydrolase superfamily and typically feature a catalytic triad composed of serine, histidine, and aspartate residues. Class 3 lipases, including those from fungal sources such as Rhizomucor miehei, are of particular industrial importance due to their stability and activity under diverse conditions.  

Conventional lipase production faces several limitations, including low expression yields, inadequate thermostability, and suboptimal activity in industrial formulations. The natural maturation process of lipases often involves propeptides that function as intramolecular chaperones, facilitating proper folding of the mature enzyme. However, these propeptides may also inhibit enzymatic activity until appropriate activation occurs, presenting challenges for industrial enzyme production.  

Prior art has described various approaches to lipase optimization, including protein engineering of mature enzyme domains and fermentation process improvements. However, the structural and functional roles of lipase propeptides remain poorly understood, with limited exploration of propeptide engineering as a strategy for enhancing lipase performance. There exists a need in the art for lipase variants with improved properties that address these limitations while maintaining or enhancing catalytic efficiency.  

## SUMMARY OF THE INVENTION  

The present invention addresses the need for improved lipase enzymes by providing modified polynucleotides encoding lipase variants with enhanced properties. Through extensive structural and functional characterization of the Rhizomucor miehei lipase system, the inventors have identified that strategic modifications in the propeptide region can significantly influence lipase performance characteristics.  

The invention particularly encompasses lipase variants comprising alterations in the propeptide contact zones that interact with the mature enzyme domain. These modifications have been shown to affect critical enzyme properties including but not limited to thermostability, specific activity, and resistance to inhibition. The invention further provides nucleic acid constructs, expression vectors, and host cells for producing these improved lipase variants, along with methods for their use in various industrial applications.  

## DESCRIPTION OF THE INVENTION  

### Definition  

For purposes of the present invention, the following terms shall have the meanings specified:  

The term "lipase" refers to any enzyme classified under EC 3.1.1.3 that catalyzes the hydrolysis of ester bonds in triglycerides. In the context of this invention, it particularly encompasses class 3 lipases from fungal sources, including Rhizomucor miehei lipase (RmL) and variants thereof.  

The term "lipase activity" denotes the catalytic capacity of an enzyme to hydrolyze ester bonds in triglyceride substrates, measurable by standard enzymatic assays such as those employing p-nitrophenyl esters or triglyceride emulsions as substrates.  

The term "lipase inhibitory activity" refers to the capacity of a molecule, particularly a propeptide, to reduce or prevent the catalytic activity of a lipase enzyme through binding interactions.  

The term "coding sequence" indicates a nucleotide sequence that encodes a polypeptide, including both the mature lipase and its propeptide when present.  

The term "control sequences" encompasses all regulatory nucleic acid sequences necessary for expression of a coding sequence, including promoters, transcription terminators, and other regulatory elements.  

The term "expression" denotes the process by which a polynucleotide is transcribed into mRNA and translated into a polypeptide within a host cell.  

The term "expression vector" refers to a DNA construct containing a coding sequence operably linked to control sequences that enable its expression in a host cell.  

The term "fragment" indicates a portion of a polypeptide or polynucleotide that retains at least one functional characteristic of the full-length molecule.  

The term "host cell" encompasses any prokaryotic or eukaryotic cell capable of expressing a heterologous polynucleotide, including bacterial, yeast, and filamentous fungal cells.  

The term "improved property" refers to any enhanced characteristic of a lipase variant compared to its parent enzyme, including but not limited to increased thermostability, specific activity, or resistance to inhibition.  

The term "mature polypeptide" denotes the fully processed, functional form of a lipase enzyme following removal of any signal peptide and propeptide.  

The term "mature polypeptide coding sequence" indicates the nucleotide sequence encoding only the mature lipase polypeptide, excluding sequences encoding signal peptides or propeptides.  

The term "mutant" refers to a polypeptide or polynucleotide comprising at least one alteration compared to a parent sequence.  

The term "nucleic acid construct" indicates a man-made nucleic acid molecule comprising a coding sequence operably linked to control sequences for expression.  

The term "operably linked" describes the functional joining of nucleic acid sequences such that the regulatory sequences control expression of the coding sequence.  

The term "parent" or "parent lipase" refers to the original, unmodified lipase from which variants are derived.  

The term "sequence identity" denotes the percentage of identical residues in an alignment of two polypeptide or polynucleotide sequences, calculated using standard algorithms such as BLAST with default parameters.  

### Conventions for Designation of Variants  

The present invention employs systematic conventions for designating lipase variants. Amino acid alterations are indicated by the original amino acid followed by its position and the new amino acid, using standard one-letter codes. For example, L81V indicates replacement of leucine at position 81 with valine.  

Sequence alignments between parent lipases and variants are performed using multiple sequence comparison methods to identify corresponding amino acid residues. The MUSCLE algorithm (v3.8.31) serves as the primary tool for generating multiple sequence alignments, though other alignment methods may also be employed.  

Corresponding amino acid residue positions are determined by structural superposition when three-dimensional structures are available, or by sequence alignment when only sequence data exists. Multiple sequence comparison utilizes the full-length sequences of related lipases to establish conserved regions and identify variable positions suitable for modification.  

Pairwise sequence comparison algorithms such as Needleman-Wunsch or Smith-Waterman may supplement multiple sequence alignments for specific applications. Probabilistic representations of polypeptide families, including hidden Markov models, provide additional tools for identifying conserved patterns and predicting functional residues.  

Homology modeling techniques enable prediction of three-dimensional structures for variants based on known parent structures. These models facilitate rational design of modifications by visualizing potential structural impacts of amino acid changes.  

### Muscle v3.8.31 Basic Usage  

The MUSCLE algorithm (v3.8.31) represents a preferred method for generating multiple sequence alignments in the context of the present invention. The basic command line usage involves specifying input and output files:  

`muscle -in <inputfile> -out <outputfile>`  

This command performs progressive alignment of sequences in the input file and writes the result to the specified output file. The algorithm proceeds through three main stages: draft progressive alignment, improved progressive alignment, and refinement.  

### Common Options (for a Complete List Please See the User Guide)  

The input file option (`-in`) specifies the file containing unaligned sequences in FASTA format. The output file option (`-out`) designates where the alignment result will be written.  

The find diagonals option (`-diags`) enables diagonal optimization for improved speed with similar sequences. The maximum number of iterations option (`-maxiters`) controls how many refinement iterations are performed, with typical values ranging from 2 to 16.  

The maximum time to iterate option (`-maxhours`) limits the total computation time in hours. The log option (`-log`) writes progress information to a specified log file for monitoring alignment status.  

### Polynucleotides  

The invention encompasses polynucleotides encoding lipase variants with improved properties. These polynucleotides may encode the full proenzyme (including propeptide and mature protein) or only the mature lipase. The propeptide and mature protein regions exhibit specific sequence identities to their parent sequences while containing strategic modifications.  

Key contact zones between the propeptide and mature lipase have been identified, particularly in regions surrounding residue Leu81 of the propeptide. Alterations in these contact zones, including substitutions, insertions, or deletions, can significantly influence lipase properties.  

Preferred modifications include substitutions at positions corresponding to Leu81 in RmL, which interact with the lid region of the mature enzyme. These alterations affect the equilibrium between open and closed conformations of the lipase, thereby modulating activity and stability.  

The number of alterations in the propeptide typically ranges from 1 to 10 amino acid changes, with preferred embodiments containing 1-5 modifications. These alterations may include substitutions, insertions, or deletions, with substitutions being particularly preferred.  

The effects of propeptide alterations include modified association kinetics between propeptide and mature lipase, altered thermostability profiles, and changes in lipase production yields. Corresponding modifications in the mature protein's propeptide contact zones can further optimize these effects.  

### Nucleic Acid Constructs  

The invention provides nucleic acid constructs comprising polynucleotides encoding lipase variants operably linked to control sequences for expression. These constructs include various regulatory elements selected based on the host cell system.  

For bacterial host cells, suitable promoters include T7, lac, trp, and SP6 promoters. Filamentous fungal host cells may utilize promoters such as the TAKA amylase, glucoamylase, or cellobiohydrolase promoters. Yeast host cells may employ the GAL1, ADH1, or TPI1 promoters.  

Transcription terminators are similarly host-specific, with bacterial terminators including rrnB and T7 terminators, fungal terminators such as the TAKA amylase terminator, and yeast terminators including the ADH1 and CYC1 terminators.  

Additional regulatory elements include mRNA stabilizer regions, leaders for efficient translation initiation, and polyadenylation sequences. Signal peptide coding regions direct secretion of expressed lipases, with options including pelB for bacteria, glucoamylase for fungi, and α-factor for yeast.  

### Expression Vectors  

Recombinant expression vectors according to the invention incorporate the nucleic acid constructs described above within suitable vector backbones. These vectors may be autonomously replicating or designed for genomic integration, depending on the host system.  

Vector components typically include selectable markers (e.g., antibiotic resistance genes), origins of replication, and optionally reporter genes. Construction involves standard molecular biology techniques such as restriction digestion, ligation, and recombination.  

Compatibility with host cells determines vector design, with bacterial vectors (e.g., pET, pUC derivatives) differing from fungal (e.g., pTAKA) or yeast (e.g., pYES) vectors. Integration into host cell genomes may occur via homologous or non-homologous recombination mechanisms.  

Origins of replication vary by host, including ColE1 for E. coli, 2μ for S. cerevisiae, and AMA1 for Aspergillus. These elements ensure proper vector maintenance and copy number control in the respective host systems.  

### Host Cells  

The invention provides recombinant host cells transformed with the expression vectors described above. These host cells may be prokaryotic or eukaryotic, with preferred embodiments including Gram-positive bacteria (e.g., Bacillus subtilis), Gram-negative bacteria (e.g., E. coli), yeast (e.g., S. cerevisiae), and filamentous fungi (e.g., Aspergillus oryzae).  

DNA introduction methods vary by host cell type, including transformation, electroporation, and conjugation for bacteria; protoplast transformation and Agrobacterium-mediated transformation for fungi; and lithium acetate transformation for yeast.  

### Methods of Production  

Lipase variants are produced by cultivating transformed host cells under conditions inducing lipase expression, followed by recovery and purification of the enzyme. Cultivation conditions are optimized for each host cell type, considering factors such as temperature, pH, aeration, and media composition.  

Recovery methods include centrifugation or filtration to separate cells from culture supernatant, followed by concentration steps such as ultrafiltration. Purification employs techniques including hydrophobic interaction chromatography, ion exchange chromatography, and affinity chromatography.  

Lipase activity is detected using standard assays measuring hydrolysis of synthetic substrates (e.g., p-nitrophenyl esters) or natural triglycerides. Purified enzymes may be formulated as liquid preparations, lyophilized powders, or granulates for specific applications.  

Optimization of lipase production involves modifying cultivation conditions, media composition, and induction parameters. Genetic modifications may include altering gene copy number, optimizing codon usage, or introducing regulatory mutations to enhance expression.  

### Detergent Compositions  

The lipase variants of the invention are particularly suited for incorporation into detergent compositions. Such compositions typically include surfactants (anionic, nonionic, cationic, or amphoteric), builders, co-builders, and additional enzymes.  

Preferred surfactant systems combine anionic and nonionic surfactants at weight ratios between 10:1 and 1:10. Builders may include zeolites, silicates, or polycarboxylates, while co-builders encompass phosphonates or citrates.  

Bleach systems may incorporate pre-formed peracids, hydrogen peroxide sources, or bleach activators such as TAED or NOBS. Additional components include enzymes (proteases, amylases), polymers, perfumes, and fabric hueing agents.  

The lipase variants are incorporated at levels between 0.0001% and 10% by weight of protein, depending on the specific formulation and application requirements. The compositions may take various forms including powders, liquids, gels, or unit dose pouches.  

## EXAMPLES  

### Materials  

All chemicals and materials were obtained from commercial suppliers unless otherwise specified. Restriction enzymes and DNA modification enzymes were from New England Biolabs. Oligonucleotides were synthesized by Integrated DNA Technologies. Culture media components were from Difco. Chromatography media included Butyl-Toyopearl (Tosoh Bioscience) and DEAE Sepharose (GE Healthcare).  

### Example 1: Variant Generation, Transformation and Expression  

The RmL gene was cloned into expression vector pTAKA and transformed into Aspergillus oryzae. Site-directed mutagenesis was performed to generate propeptide variants using overlap extension PCR. Transformants were selected and cultured in M400 medium at 34°C for protein expression.  

### Purification of RmL WT Controls: RmL WT and RmL WT Pur  

Culture supernatant was clarified by filtration and applied to a decylamine agarose column. The flow-through containing ProRmL was collected, while mature RmL was eluted with water. Further purification employed ion exchange chromatography on UNO Q resin. Protein purity was assessed by Native PAGE and zymogram analysis.  

### Propeptide Variant Generation  

PCR-based site-directed mutagenesis introduced specific alterations in the propeptide region. Variants were transformed into E. coli for plasmid propagation, then into A. oryzae for expression. Mutations were confirmed by DNA sequencing. Expressed variants were analyzed for lipase activity and stability.  

### Example 2: Native PAGE  

Proteins were analyzed on 12% native polyacrylamide gels with Tris-glycine buffer system. Gels were either stained with Coomassie Brilliant Blue or used for zymogram analysis on agarose/olive oil plates to assess lipase activity.  

### Example 3: Temperature Stability Assay (TSA)  

Protein thermal stability was measured using SYPRO Orange dye in a real-time PCR instrument. Melting temperatures (Tm) were determined from fluorescence curves, with higher Tm values indicating greater thermostability.  

### Example 4: Zymogram Analysis  

Native PAGE gels were overlaid on agarose containing olive oil and brilliant green dye. Active lipases appeared as clear zones against the stained background after incubation at 37°C.  

### Example 5: Decylamine Agarose Chromatography  

ProRmL variants were purified using decylamine agarose columns equilibrated with 1M NaCl in HEPES buffer. Proteins were eluted with decreasing salt gradients or water steps. Fractions were analyzed by SDS-PAGE and activity assays.  

### Example 6: Relative Wash Performance  

Lipase variants were tested in Automatic Mechanical Stress Assay (AMSA) using turmeric-stained cotton cloth. Wash performance was measured spectrophotometrically at 460 nm and expressed relative to a reference lipase. Tests were conducted in glycine buffers and model detergent formulations at varying temperatures and pH values.