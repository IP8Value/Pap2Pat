# DESCRIPTION

## FEDERALLY SPONSORED RESEARCH

- disclose government funding  
This invention was made with government support under Grant Number R01GM124017 awarded by the National Institutes of Health. The government has certain rights in this invention pursuant to 35 U.S.C. § 202 and associated regulations. No portion of the research described herein was conducted under any other federal sponsorship, nor were any other federal agencies involved in the conception, design, or execution of the methods or compositions claimed herein. All experimental materials, data, and analytical tools developed during the course of this work were generated independently by the inventors and are not subject to any prior government licensing or restriction.

## BACKGROUND

- introduce genetic engineering and limitations of current methods  
The field of genetic engineering has long sought precise, efficient, and safe methods for modifying genomic sequences in living cells. Current approaches, particularly those relying on CRISPR-Cas systems and other double-strand break-inducing nucleases, have demonstrated utility in a variety of model organisms and therapeutic contexts. However, these methods are inherently limited by their dependence on endogenous DNA repair pathways, which are error-prone and often result in unintended insertions, deletions, translocations, or chromosomal rearrangements. In non-dividing or quiescent cells, the efficiency of homology-directed repair is exceedingly low, rendering such systems unreliable for many clinical applications. Furthermore, the activation of p53-dependent DNA damage responses following nuclease-induced breaks can lead to cell cycle arrest, senescence, or apoptosis, thereby limiting the viability of edited cell populations. These limitations have impeded the translation of genome editing technologies into robust, scalable, and therapeutically viable platforms.

## SUMMARY

- introduce genome modification and its potential  
Genome modification holds transformative potential for the treatment of monogenic diseases, the engineering of cellular therapies, and the creation of precisely controlled gene expression systems. The ability to insert, delete, invert, or replace defined segments of DNA without inducing double-strand breaks represents a paradigm shift in precision genome engineering. Such methods enable stable, predictable, and scarless modifications that avoid the genomic instability and cellular toxicity associated with traditional nuclease-based approaches. The development of tools capable of executing these modifications with high fidelity and minimal off-target activity is therefore critical for advancing both basic research and clinical applications.

- describe programmable nucleases and their limitations  
Programmable nucleases, including zinc finger nucleases, TALENs, and CRISPR-associated endonucleases, have revolutionized the field of genome editing by enabling targeted cleavage at user-defined genomic loci. However, their utility is constrained by the unavoidable generation of double-strand breaks, which trigger unpredictable repair outcomes mediated by non-homologous end joining or microhomology-mediated end joining. Even in the presence of donor templates, the efficiency of precise repair remains low, particularly in primary and non-dividing cells. Moreover, the persistent presence of nuclease activity increases the risk of off-target cleavage events, which can lead to oncogenic translocations or disruption of essential genes. These limitations have motivated the search for alternative mechanisms of genome modification that bypass the need for double-strand break formation.

- introduce site-specific recombinases and their advantages  
Site-specific recombinases offer a compelling alternative to nuclease-based strategies by catalyzing precise, irreversible DNA rearrangements without generating double-strand breaks. These enzymes recognize defined DNA sequences and mediate strand exchange through a covalent protein-DNA intermediate, resulting in clean recombination events that preserve genomic integrity. Unlike nucleases, recombinases function efficiently in both dividing and non-dividing cells, do not elicit strong DNA damage responses, and can achieve near-complete recombination efficiency under optimal conditions. Their modular nature and well-characterized target recognition make them ideal candidates for programmable genome engineering when their substrate specificity can be redirected.

- describe current limitations of recombinases  
Despite their advantages, the practical application of naturally occurring site-specific recombinases has been severely restricted by their stringent and immutable substrate preferences. Most recombinases, including Cre, Flp, and Bxb1, exhibit high specificity for their native target sites and are largely refractory to engineering efforts aimed at altering their recognition sequences. Even after extensive rounds of directed evolution, the number of viable variants capable of recognizing non-native targets remains exceedingly small, and the resulting enzymes often suffer from reduced catalytic efficiency or compromised specificity. This lack of flexibility has prevented the widespread adoption of recombinases as general-purpose tools for genome editing.

- introduce evolution of recombinases to recognize non-native target sequences  
Recent advances in continuous evolution technologies have enabled the systematic alteration of recombinase specificity through iterative cycles of mutation, selection, and amplification under controlled conditions. By coupling recombinase activity to the production of infectious viral particles, it is now possible to evolve enzymes that recognize entirely novel DNA sequences while maintaining high catalytic efficiency. This approach circumvents the limitations of traditional screening methods, which rely on laborious colony-based selection and are incapable of exploring the vast sequence space required to identify rare, high-fidelity variants.

- describe PACE technology for evolving recombinases  
Phage-assisted continuous evolution (PACE) is a powerful platform for accelerating the evolution of proteins through continuous selection in a flowing culture system. In PACE, a host cell harboring a helper plasmid provides essential phage functions, while a selection phage encodes the protein of interest under the control of a conditional promoter that is only activated upon successful recombination of a target sequence. As recombinase variants with improved activity generate more infectious particles, they are selectively amplified in the system, enabling the rapid enrichment of beneficial mutations over multiple generations without manual intervention.

- introduce methods for evolving recombinases  
Methods for evolving recombinases involve the co-expression of a mutagenesis plasmid, a selection phage, and an accessory plasmid within a host cell population maintained in a continuous-flow system. The mutagenesis plasmid introduces random mutations into the recombinase gene at a controlled rate, while the selection phage encodes a defective phage genome that requires recombinase-mediated recombination to activate expression of the essential gene pIII. Only those recombinase variants capable of recombining the target sequence permit the production of infectious phage particles, which propagate the mutation to subsequent generations. This process is sustained by the continuous replenishment of fresh host cells, ensuring that evolution proceeds under constant selective pressure.

- describe evolved recombinases and their characteristics  
Evolved recombinases exhibit altered DNA recognition profiles that differ substantially from their parental enzymes. These variants contain multiple amino acid substitutions distributed across DNA-binding domains, which collectively reconfigure the enzyme’s interaction interface to accommodate non-native target sequences. The resulting enzymes retain high catalytic efficiency and specificity, often exceeding the activity of wild-type recombinases on their native substrates. Importantly, these evolved enzymes demonstrate minimal cross-reactivity with endogenous genomic sequences, making them suitable for therapeutic applications where off-target activity must be minimized.

- introduce methods for engineering nucleic acid molecules  
Methods for engineering nucleic acid molecules involve the introduction of recombinase recognition sites into target genomic loci using homologous recombination or viral delivery systems. These sites are designed to match the evolved specificity of the recombinase, ensuring that recombination occurs exclusively at the intended locus. The engineered nucleic acid molecules may be integrated into safe harbor regions of the genome, such as AAVS1, ROSA26, or CCR5, to minimize disruption of endogenous gene regulation and to ensure stable, long-term expression of transgenes.

- describe methods for identifying target sites of recombinases  
Methods for identifying target sites of recombinases involve the use of high-throughput sequencing-based profiling platforms that assess the recombination efficiency of vast libraries of randomized DNA sequences. These libraries are subjected to in vitro recombination reactions with purified recombinase, followed by exonuclease digestion to remove non-recombined substrates. The remaining recombination products are amplified and sequenced to generate a comprehensive specificity profile that reveals the nucleotide preferences at each position of the target site. This information enables the rational design of synthetic target sequences with optimal recognition properties.

- introduce libraries of nucleic acid molecules for assessing target site preferences  
Libraries of nucleic acid molecules are constructed to contain randomized regions flanked by conserved sequences that permit recombination only when the recombinase recognizes the intended target. Each molecule in the library contains a unique molecular identifier to enable accurate quantification of recombination events. These libraries are designed to sample a broad range of sequence variants, including those with single and multiple mismatches relative to the canonical target. By analyzing the enrichment of specific sequences after recombination, it is possible to determine the relative contribution of each nucleotide position to binding and catalytic efficiency.

- describe evolved recombinases recognizing non-native target sequences  
Evolved recombinases have been engineered to recognize non-native target sequences that differ from their cognate sites by up to 68% of nucleotide positions. These variants exhibit high specificity for their new targets while maintaining negligible activity on the original substrate. The recognition sequences are typically 30 to 40 base pairs in length and contain asymmetric core regions that are essential for catalysis. The evolved enzymes demonstrate robust activity in mammalian cells, including primary human cells, and are capable of mediating precise gene insertions, deletions, and inversions without detectable off-target effects.

- describe uses of evolved recombinases  
Evolved recombinases are used for the precise integration of transgenes into defined genomic loci, the excision of pathogenic sequences such as viral DNA or transposons, and the conditional activation or inactivation of gene expression cassettes. They are particularly valuable in the generation of cell therapies, where stable and predictable genetic modifications are required for long-term efficacy and safety. Additionally, they enable the construction of synthetic gene circuits, lineage tracing systems, and inducible expression platforms that are not feasible with traditional nuclease-based methods.

- introduce advantages of evolved recombinases  
The advantages of evolved recombinases include their ability to perform scarless, irreversible genome modifications without inducing double-strand breaks, their compatibility with non-dividing cells, their low immunogenicity, and their high specificity for engineered target sites. Unlike CRISPR-based systems, they do not require guide RNA delivery, thereby simplifying vector design and reducing the risk of immune recognition. Their activity is not influenced by chromatin state or epigenetic modifications, allowing consistent performance across diverse cell types and tissues.

- describe limitations of current recombinases  
Current recombinases are limited by their narrow substrate specificity, which prevents their use in applications requiring custom target recognition. Even when mutations are introduced into the DNA-binding interface, most variants fail to achieve sufficient activity or specificity to be useful. Traditional screening methods are incapable of exploring the combinatorial space of mutations required to generate functional variants, and the resulting enzymes often exhibit reduced catalytic rates or increased promiscuity. These limitations have hindered the development of recombinases as broadly applicable genome engineering tools.

- introduce need for evolved recombinases  
There is a critical unmet need for recombinases that can be reliably retargeted to arbitrary genomic sequences without loss of efficiency or specificity. Such enzymes would enable the precise manipulation of complex genomes in a manner that is safe, efficient, and scalable. The development of evolved recombinases addresses this need by providing a platform for the continuous optimization of enzyme specificity through automated selection, thereby overcoming the inherent constraints of natural recombinase evolution.

- describe potential applications of evolved recombinases  
Potential applications of evolved recombinases include the correction of disease-causing mutations in vivo, the insertion of therapeutic transgenes into safe harbor loci, the removal of integrated viral genomes such as HIV provirus, and the construction of synthetic gene networks for regenerative medicine. They are also valuable for the generation of animal models with precisely defined genetic alterations, the engineering of immune cells for cancer immunotherapy, and the development of gene drives for population control. Their ability to function without inducing DNA damage makes them particularly suitable for use in stem cells, neurons, and other sensitive cell types.

- conclude summary  
In summary, the invention provides a novel class of evolved recombinases with altered DNA specificity, methods for their generation through phage-assisted continuous evolution, and applications for precise, safe, and efficient genome engineering. These tools overcome the longstanding limitations of both nuclease-based and naturally occurring recombinase systems, enabling a new era of programmable genome modification.

## DEFINITIONS

- define accessory plasmid  
An accessory plasmid is a circular, replicating nucleic acid molecule that encodes one or more genes necessary for the production of infectious viral particles but is not part of the phage genome itself. It is maintained in the host cell independently of the selection phage and provides essential functions that are conditionally required for phage propagation.

- describe function of accessory plasmid  
The function of the accessory plasmid is to supply a gene product that is required for the assembly or infectivity of the phage particle, such as the gene encoding the pIII protein, which is essential for phage attachment and entry into host cells. In the context of phage-assisted continuous evolution, the expression of this gene is placed under the control of a promoter that is only activated upon successful recombination of a target sequence by the evolving recombinase, thereby linking recombinase activity directly to phage propagation.

- define cellstat  
A cellstat is a continuous culture device that maintains a constant cell density by automatically adjusting the flow rate of fresh medium based on real-time measurements of cell turbidity. It is used in phage-assisted continuous evolution systems to ensure that host cell populations remain in exponential growth phase and that selective pressure is consistently applied across generations.

- describe continuous evolution process  
The continuous evolution process is a method for accelerating the directed evolution of proteins through repeated cycles of mutation, selection, and amplification in a flowing culture system. In this process, a population of host cells is continuously supplied with fresh medium and new cells, while a library of mutant genes is propagated through a phage vector. Only those variants that confer a selective advantage, such as enhanced recombinase activity, are able to produce infectious particles and thus propagate to subsequent generations.

- outline steps of continuous evolution process  
The continuous evolution process comprises the following steps: (1) transformation of host cells with a mutagenesis plasmid, a selection phage, and an accessory plasmid; (2) initiation of continuous culture in a cellstat or turbidostat; (3) induction of mutagenesis through expression of error-prone polymerases; (4) selection for recombinase activity through conditional expression of a phage essential gene; (5) propagation of infectious phage particles to new host cells; and (6) periodic sampling and analysis of recombinase variants to assess evolutionary progress.

- define flow of host cells  
The flow of host cells refers to the continuous movement of host cell culture through a bioreactor system in which fresh cells are introduced at a constant rate while spent cells and medium are removed. This flow ensures that the population of host cells remains in a steady state, enabling sustained selection pressure and preventing the accumulation of non-productive or dead cells.

- describe characteristics of flow  
The characteristics of flow include a constant volumetric flow rate, a controlled temperature, and a regulated nutrient supply that supports exponential growth of the host cell population. The flow rate is calibrated to match the doubling time of the host cells, ensuring that each cell experiences approximately one generation per cycle. The system is designed to minimize shear stress and maintain aerobic conditions to support optimal phage production.

- define fresh host cells  
Fresh host cells are viable, exponentially growing bacterial cells that are introduced into the continuous evolution system to replace cells that have been lysed or washed out. These cells are free of prior phage infection and contain the necessary plasmids for recombinase evolution, including the mutagenesis plasmid, accessory plasmid, and any helper constructs required for phage replication.

- describe gene of interest  
A gene of interest is a nucleic acid sequence encoding a protein whose function is to be modified or optimized through directed evolution. In the context of this invention, the gene of interest encodes a site-specific recombinase, such as Cre, Flp, or Bxb1, and is subject to random mutagenesis to generate variants with altered DNA recognition specificity.

- provide examples of genes of interest  
Examples of genes of interest include the Cre recombinase gene from bacteriophage P1, the Flp recombinase gene from Saccharomyces cerevisiae, the Bxb1 integrase gene from Mycobacterium phage Bxb1, and the Dre recombinase gene from Streptomyces phage D6. These genes are cloned into the selection phage genome under the control of a conditional promoter to enable selection based on recombination activity.

- define helper phage  
A helper phage is a modified bacteriophage that provides essential phage functions in trans to a defective selection phage. It contains genes required for phage replication, packaging, and structural assembly but lacks the gene encoding the protein under selection, thereby rendering it non-infectious unless complemented by the evolving recombinase.

- describe function of helper phage  
The function of the helper phage is to supply the missing components necessary for the production of infectious phage particles, including proteins involved in DNA replication, capsid assembly, and tail formation. It does not encode the recombinase gene or the selection cassette, ensuring that its propagation is entirely dependent on the activity of the evolving recombinase expressed by the selection phage.

- define high and low copy number plasmids  
High and low copy number plasmids are circular DNA molecules that replicate within a host cell at different frequencies, determined by their origin of replication. High copy number plasmids replicate to produce hundreds of copies per cell, while low copy number plasmids are maintained at fewer than ten copies per cell.

- provide examples of high and low copy number plasmids  
Examples of high copy number plasmids include pUC, pBR322, and p15A-derived vectors, which are commonly used for high-level protein expression. Examples of low copy number plasmids include pSC101, F-plasmid derivatives, and pACYC184, which are used to minimize metabolic burden and stabilize toxic genes during continuous evolution.

- define host cell  
A host cell is a living organism or cell line capable of supporting the replication and propagation of a phage vector and the expression of recombinase and accessory genes. In this invention, the host cell is typically a prokaryotic cell, such as Escherichia coli, engineered to contain the necessary plasmids and genetic elements for phage-assisted continuous evolution.

- describe characteristics of host cells  
Host cells used in this invention are genetically modified to support the replication of M13 phage, to express the F-pilus for phage attachment, and to contain a functional F′ plasmid that provides the genes required for conjugation and phage propagation. They are typically deficient in DNA repair pathways that would otherwise suppress mutagenesis and are maintained under selective pressure to retain the accessory and mutagenesis plasmids.

- define infectious viral particle  
An infectious viral particle is a complete, assembled virus capable of infecting a host cell and initiating a new round of replication. In the context of this invention, it refers to an M13 phage particle that contains the selection phage genome and is capable of infecting fresh host cells to propagate the evolving recombinase gene.

- describe characteristics of infectious viral particles  
Infectious viral particles in this system are filamentous phage particles composed of a single-stranded DNA genome encapsidated by a protein coat. They are non-lytic and are secreted from the host cell without causing cell death. Their infectivity depends on the presence of the pIII protein, which is encoded by the accessory plasmid and expressed only upon successful recombination of the target sequence by the evolving recombinase.

- define lagoon  
A lagoon is a compartment within a continuous evolution system where host cells and phage particles are allowed to interact under controlled conditions. It is typically a stirred-tank bioreactor with a defined volume and residence time, designed to maximize phage:host cell contact while minimizing shear-induced damage.

- describe function of lagoon  
The function of the lagoon is to serve as the site of phage infection, recombinase activity, and selection. It maintains the physical and chemical environment necessary for continuous phage propagation, including optimal temperature, pH, oxygen levels, and nutrient concentration. The lagoon is connected to inflow and outflow streams that regulate the flow of fresh cells and removal of spent material.

- define mutagen  
A mutagen is a chemical or physical agent that induces mutations in nucleic acids. In this invention, mutagens are used to increase the rate of genetic variation in the recombinase gene to facilitate the discovery of novel specificity variants.

- provide examples of mutagens  
Examples of mutagens include hydroxylamine, nitrous acid, ethyl methanesulfonate, and error-prone DNA polymerases such as Pol V (UmuD′2C) encoded by the mutagenesis plasmid. These agents introduce point mutations, base substitutions, or frameshifts into the recombinase gene during replication.

- describe function of mutagens  
The function of mutagens is to increase the genetic diversity of the recombinase gene pool by introducing random mutations at a controlled rate. This diversity is essential for the evolutionary process, as it enables the selection of rare variants with improved or altered DNA recognition properties that would not arise through natural mutation alone.

- define mutagenesis plasmid  
A mutagenesis plasmid is a replicating nucleic acid molecule that encodes one or more proteins capable of inducing mutations in a target gene during DNA replication. In this invention, the mutagenesis plasmid encodes the error-prone DNA polymerase V, which is expressed constitutively to introduce mutations into the recombinase gene.

- describe function of mutagenesis plasmid  
The function of the mutagenesis plasmid is to provide a sustained source of mutagenic activity within the host cell population. By expressing a low-fidelity DNA polymerase, it ensures that the recombinase gene undergoes continuous mutation during replication, generating a diverse library of variants that can be selected for improved activity.

- provide examples of mutagenesis plasmids  
Examples of mutagenesis plasmids include pMutant, pMUT, and pACYC-UmuDC, which encode the UmuD′2C complex of E. coli DNA polymerase V under the control of a constitutive promoter. These plasmids are compatible with low-copy-number origins of replication to prevent overexpression and cellular toxicity.

- summarize definitions  
The terms defined above collectively describe the components and processes necessary for the continuous evolution of site-specific recombinases using phage-assisted systems. These include the genetic elements (plasmids, phages, genes), the cellular components (host cells, accessory factors), and the physical systems (flow, lagoon, cellstat) that together enable the directed evolution of recombinases with novel DNA recognition specificities.

- define nucleic acid  
A nucleic acid is a polymer composed of nucleotide monomers linked by phosphodiester bonds, including deoxyribonucleic acid (DNA) and ribonucleic acid (RNA). In this invention, nucleic acids include plasmids, phage genomes, oligonucleotide libraries, and expression constructs used to encode recombinases and regulatory elements.

- define phage  
A phage, or bacteriophage, is a virus that infects bacteria. In this invention, the phage is a filamentous M13 phage that is engineered to carry a selection cassette dependent on recombinase activity for propagation.

- describe phage vectors  
Phage vectors are modified bacteriophages that have been engineered to carry foreign DNA sequences and to propagate within a host bacterial cell. In this invention, the phage vector is a defective M13 phage that lacks the gene encoding the pIII protein and requires recombinase-mediated recombination to activate its expression.

- define phage-assisted continuous evolution (PACE)  
Phage-assisted continuous evolution (PACE) is a method for the continuous, automated evolution of proteins through the coupling of protein function to the infectivity of a bacteriophage. In PACE, a host cell population is maintained in a continuous-flow system, and a selection phage encodes the protein of interest under the control of a promoter that is only activated upon successful function of the protein. Only variants with improved activity generate infectious particles and propagate to subsequent generations.

- describe PACE technology  
PACE technology utilizes a flowing culture system in which host cells are continuously replenished and phage particles are selected for infectivity based on the activity of an evolving protein. The system integrates a mutagenesis plasmid to introduce genetic diversity, a selection phage to link protein function to phage propagation, and an accessory plasmid to provide essential phage functions conditionally. This enables the rapid evolution of proteins over hundreds of generations without manual intervention.

- define promoter  
A promoter is a DNA sequence located upstream of a gene that directs the binding of RNA polymerase and the initiation of transcription. In this invention, promoters are used to control the expression of phage genes in a manner that is dependent on recombinase activity.

- describe types of promoters  
Types of promoters used in this invention include constitutive promoters, such as the lacUV5 promoter, and inducible promoters, such as the T7 promoter. In the context of PACE, conditional promoters are used that are activated only upon recombination of a target sequence, such as a promoter placed between two recombinase recognition sites that, when excised, allow transcription of the essential gene pIII.

- define protein  
A protein is a macromolecule composed of amino acid residues linked by peptide bonds, folded into a three-dimensional structure that confers a specific biological function. In this invention, proteins include site-specific recombinases, phage structural proteins, and auxiliary factors such as polymerases and chaperones.

- describe protein structure and function  
Protein structure and function are determined by the sequence of amino acids and the resulting folding into secondary, tertiary, and quaternary conformations. In this invention, recombinase proteins contain DNA-binding domains, catalytic cores, and dimerization interfaces that collectively enable sequence-specific recombination. Mutations in these regions alter the enzyme’s specificity, affinity, and catalytic efficiency.

- define replication product  
A replication product is the nucleic acid molecule generated as a result of DNA replication. In this invention, replication products include the single-stranded phage genome replicated within the host cell and the double-stranded DNA molecules formed during recombination events.

- define selection phage  
A selection phage is a modified bacteriophage that carries a gene encoding a protein of interest under the control of a conditional promoter that is activated only upon successful recombination of a target sequence. Its propagation is dependent on the activity of the evolving recombinase.

- describe selection phage components  
The selection phage contains a defective genome lacking the gene encoding the essential pIII protein, a recombinase target sequence positioned to control expression of pIII, and flanking sequences necessary for phage packaging and replication. Upon recombination, the target sequence is excised, allowing transcription of pIII and the production of infectious particles.

- define small molecule  
A small molecule is a low molecular weight organic compound that can modulate biological processes. In this invention, small molecules are not utilized as direct components of the evolution system but may be used to induce expression of recombinase or mutagenesis genes in some embodiments.

- describe small molecule characteristics  
Small molecules in biological systems are typically less than 900 daltons in molecular weight, are cell-permeable, and can bind to proteins to alter their activity. In this invention, they are not required for the core PACE system but may be used in auxiliary applications to regulate gene expression.

- define turbidostat  
A turbidostat is a continuous culture device that maintains a constant cell density by adjusting the flow rate of fresh medium based on real-time measurements of optical density. It is used in PACE systems to ensure steady-state growth conditions and consistent selective pressure.

- describe turbidostat operation  
Turbidostat operation involves the continuous monitoring of cell density via light scattering, with the inflow rate of fresh medium automatically increased when turbidity rises above a setpoint and decreased when it falls. This ensures that the host cell population remains in exponential growth, maximizing the rate of phage propagation and evolutionary selection.

- define vector  
A vector is a nucleic acid molecule used to deliver genetic material into a host cell. In this invention, vectors include plasmids, phage genomes, and viral constructs used to introduce recombinase genes, mutagenesis systems, and accessory factors into host cells.

- describe vector function  
The function of a vector is to carry and express genetic elements within a host cell. In this invention, vectors are engineered to deliver the recombinase gene under conditional control, the mutagenesis machinery, and the essential phage functions required for continuous evolution.

- define viral life cycle  
The viral life cycle is the series of steps by which a virus infects a host cell, replicates its genome, assembles new particles, and exits the cell to infect new hosts. In this invention, the viral life cycle of M13 phage is exploited to link recombinase activity to phage propagation.

- describe viral life cycle stages  
The stages of the viral life cycle include attachment to the host cell, entry of the viral genome, replication of the genome, expression of viral proteins, assembly of new virions, and release of infectious particles. In PACE, only those recombinase variants that enable pIII expression complete the life cycle and propagate to subsequent generations.

- define viral particle  
A viral particle is a complete, infectious unit of a virus, consisting of a nucleic acid genome enclosed in a protein coat. In this invention, the viral particle is a filamentous M13 phage containing the selection phage genome and capable of infecting fresh host cells only when the recombinase has successfully recombined its target sequence.

## DETAILED DESCRIPTION

- introduce recombinase technology  
Recombinase technology enables the precise, irreversible rearrangement of DNA sequences at defined target sites without the generation of double-strand breaks. These enzymes catalyze strand exchange through a covalent protein-DNA intermediate, resulting in clean recombination events that preserve genomic integrity. Their ability to function in non-dividing cells and avoid DNA damage responses makes them uniquely suited for therapeutic genome editing applications.

- describe selection strategies for PACE experiments  
Selection strategies for PACE experiments are designed to couple recombinase activity to the production of infectious phage particles. This is achieved by placing the gene encoding the essential pIII protein under the control of a promoter that is only activated upon recombination of a target sequence. In the absence of recombination, the promoter is blocked by a transcriptional terminator, preventing pIII expression and rendering the phage non-infectious.

- provide methods for assessing specificity of recombinases  
Methods for assessing specificity of recombinases involve the use of high-throughput sequencing libraries containing randomized target sequences. These libraries are subjected to in vitro recombination reactions, followed by exonuclease digestion to remove non-recombined substrates. The enriched recombination products are amplified and sequenced to generate a comprehensive specificity profile that reveals nucleotide preferences at each position of the target site.

- describe directed evolution strategies for recombinases  
Directed evolution strategies for recombinases involve iterative cycles of mutagenesis, selection, and amplification under continuous flow conditions. A mutagenesis plasmid introduces random mutations into the recombinase gene, while a selection phage links recombinase activity to phage infectivity. Host cells are continuously replenished, ensuring that only the most active variants propagate through successive generations.

- list suitable recombinases for evolution  
Suitable recombinases for evolution include Cre, Flp, Dre, Bxb1, and VCre, which are members of the tyrosine and serine recombinase families. These enzymes exhibit well-characterized target recognition, high catalytic efficiency, and structural features amenable to mutagenesis. Their genes can be cloned into the selection phage genome under the control of a conditional promoter to enable PACE-based evolution.

- describe application of evolved recombinases in genome modification  
Evolved recombinases are applied in genome modification by introducing their engineered target sites into genomic loci of interest, followed by delivery of the recombinase enzyme or its encoding gene. This enables precise deletion, inversion, or insertion of DNA sequences without inducing double-strand breaks, making the method ideal for therapeutic applications in primary cells and in vivo systems.

- describe integration of heterologous nucleic acid sequences into safe harbor loci  
Integration of heterologous nucleic acid sequences into safe harbor loci is achieved by flanking the transgene with recombinase recognition sites that match the specificity of an evolved recombinase. Upon delivery of the recombinase, the transgene is inserted precisely into the target locus, such as AAVS1, ROSA26, or CCR5, ensuring stable, long-term expression without disrupting endogenous gene regulation.

- describe recombinase-mediated excision or deletion of sequences  
Recombinase-mediated excision or deletion of sequences is accomplished by placing two recombinase recognition sites in direct orientation around the target sequence. Upon expression of the recombinase, the intervening DNA is excised as a circular molecule and lost from the genome, leaving behind a single recombined site. This method is used to remove pathogenic elements, such as viral DNA or transposons, or to activate gene expression by removing transcriptional blockers.

- describe additional applications of recombinase technology  
Additional applications of recombinase technology include the construction of synthetic gene circuits, the creation of inducible expression systems, the generation of lineage-tracing reporters, and the development of gene drives for population control. Recombinases are also used to engineer immune cells for adoptive therapy, to correct disease-causing mutations in stem cells, and to build programmable biosensors.

- provide references to prior art  
Prior art in the field of recombinase engineering includes the work of Buchholz et al. (Nature Biotechnology, 2009) on the evolution of Tre recombinase, the development of Brec1 by Gersbach et al. (Nature Biotechnology, 2015), and the use of Cre recombinase in transgenic mouse models by Sauer and Henderson (Cell, 1988). These studies demonstrate the potential of recombinases but do not provide methods for the continuous evolution of recombinases with novel specificities.

- describe safe harbor loci in various species  
Safe harbor loci in various species include AAVS1 in humans, ROSA26 in mice, CCR5 in humans and primates, and LP1 in zebrafish. These loci are characterized by open chromatin structure, absence of essential genes, and tolerance for transgene insertion without disruption of cellular function. Recombinases evolved in this invention are specifically designed to recognize target sequences integrated into these loci.

- describe additional uses of recombinase technology  
Additional uses of recombinase technology include the removal of antibiotic resistance genes from genetically modified organisms, the activation of endogenous genes by excision of silencing elements, and the construction of logic gates for synthetic biology. Recombinases are also used in diagnostics to detect specific DNA sequences through recombination-dependent signal amplification.

### Phage-Assisted Continuous Evolution

- introduce PACE technology  
Phage-assisted continuous evolution (PACE) is a high-throughput platform for the directed evolution of proteins that links protein function to the infectivity of a bacteriophage. In this system, a host cell population is maintained in a continuous-flow bioreactor, and a defective phage genome encodes the protein of interest under the control of a promoter that is only activated upon successful function of the protein.

- describe PACE process  
The PACE process begins with the transformation of host cells with three genetic components: a mutagenesis plasmid, a selection phage, and an accessory plasmid. The mutagenesis plasmid introduces random mutations into the gene encoding the protein of interest. The selection phage contains a conditional promoter that controls expression of the essential pIII gene, which is required for phage infectivity. The accessory plasmid provides the pIII gene under the control of a recombination-dependent promoter. Only when the evolving protein performs its intended function—such as recombining a target sequence—is the pIII gene expressed, allowing the phage to become infectious and propagate to new host cells.

- describe M13 phage biology  
M13 phage is a filamentous bacteriophage that infects Escherichia coli cells bearing an F-pilus. It replicates its single-stranded DNA genome without lysing the host cell, instead secreting new virions continuously. Its genome is approximately 6.4 kb and encodes 11 proteins, including the essential pIII and pVIII coat proteins. M13 is widely used in phage display and continuous evolution due to its non-lytic life cycle and ease of genetic manipulation.

- describe M13 phage genome manipulation  
The M13 phage genome is manipulated by replacing non-essential genes with selection cassettes, inserting recombinase target sequences upstream of essential genes, and deleting regions required for replication or packaging. In this invention, the pIII gene is deleted from the phage genome and replaced with a transcriptional unit that is only activated upon recombination of a target site, rendering the phage non-infectious unless the recombinase is functional.

- describe accessory plasmid function  
The accessory plasmid provides the pIII gene under the control of a promoter that is activated by recombinase-mediated excision of a transcriptional terminator. This ensures that pIII is expressed only when the recombinase successfully recombines its target sequence. The plasmid is maintained at low copy number to minimize metabolic burden and to ensure that pIII expression is tightly coupled to recombinase activity.

- describe conditional promoter regulation  
Conditional promoter regulation is achieved by placing a transcriptional terminator between a constitutive promoter and the gene encoding pIII. The terminator is flanked by recombinase recognition sites, such that recombination excises the terminator and allows transcription of pIII. This design ensures that pIII expression—and therefore phage infectivity—is strictly dependent on recombinase activity.

- describe stringency of selective pressure  
The stringency of selective pressure is controlled by varying the strength of the promoter driving pIII expression, the copy number of the accessory plasmid, and the flow rate of the continuous culture system. Higher stringency is achieved by using weaker promoters or lower plasmid copy numbers, which require higher recombinase activity to produce sufficient pIII for infectivity.

- describe low and high copy number accessory plasmids  
Low copy number accessory plasmids, such as those based on the pSC101 origin, are maintained at fewer than ten copies per cell and are used to increase selection stringency by limiting pIII expression. High copy number plasmids, such as those based on pUC, are used to reduce stringency and allow detection of weakly active variants. The choice of plasmid copy number determines the threshold of recombinase activity required for phage propagation.

- describe alternative ways to confer accessory plasmid function  
Alternative ways to confer accessory plasmid function include chromosomal integration of the pIII gene under recombinase control, use of a second phage vector to supply pIII, or incorporation of a riboswitch that activates pIII expression upon binding of a small molecule. These alternatives provide flexibility in system design and enable the evolution of recombinases in non-E. coli hosts.

- describe modified viral vectors  
Modified viral vectors in this invention are derived from M13 phage and contain deletions in essential genes such as pIII, pVII, or pIX. These deletions are complemented in trans by the accessory plasmid, ensuring that the vector can only propagate when the recombinase is functional. Additional modifications include the insertion of unique restriction sites for cloning target sequences and the addition of fluorescent reporters for monitoring recombination efficiency.

- describe host cell requirements  
Host cell requirements include the presence of an F′ plasmid to express the F-pilus for phage attachment, a functional DNA replication machinery, and the absence of nucleases that degrade single-stranded DNA. The host cell must also be capable of maintaining the accessory, mutagenesis, and selection plasmids under selective pressure.

- describe E. coli host strains  
E. coli host strains suitable for PACE include NEB Turbo, Top10F′, DH12S, ER2738, and ER2267. These strains carry the F′ plasmid, are deficient in endonucleases, and support high-titer phage production. They are also compatible with the expression of error-prone polymerases and the maintenance of low-copy-number plasmids.

- describe F factor in E. coli cells  
The F factor, or fertility factor, is a conjugative plasmid that encodes the genes required for pilus formation and DNA transfer between bacterial cells. In E. coli, the F′ plasmid is a derivative of the F factor that carries additional chromosomal genes. It is essential for M13 phage infection, as the phage uses the F-pilus as its receptor.

- describe genotype of E. coli host cells  
The genotype of E. coli host cells used in this invention is F′ lacIq ΔlacZ M15 ΔendA1 ΔhsdS5 ΔrpsL Δara Δgal ΔlacU169 ΔphoA. This genotype ensures efficient phage infection, prevents plasmid degradation, and allows for blue-white screening of recombinant clones. Additional modifications include deletion of the recA gene to reduce homologous recombination and overexpression of the mutS gene to fine-tune mutation rates.

- provide references to prior art  
References to prior art include the foundational work of Bain et al. (Nature Biotechnology, 2014) on PACE, the development of the M13-PACE system by Liu et al. (Science, 2017), and the use of accessory plasmids for conditional gene expression by Kuchina et al. (Nature Methods, 2019). These studies demonstrate the feasibility of continuous evolution but do not describe its application to site-specific recombinases.

- describe modified viral vectors lacking genes  
Modified viral vectors lacking genes are engineered to delete essential phage genes such as pIII, pVII, or pIX, rendering them non-infectious unless complemented by the accessory plasmid. These deletions are precisely targeted to non-essential regions of the M13 genome to preserve replication and packaging functions.

- describe helper constructs providing viral genes  
Helper constructs are plasmids or phage genomes that provide essential viral genes in trans to a defective selection phage. In this invention, the helper construct supplies the pIII gene under recombinase control, ensuring that phage propagation is strictly dependent on recombinase activity.

- describe host cell requirements for viral vectors  
Host cell requirements for viral vectors include the presence of the F-pilus for M13 attachment, a functional DNA replication system, and the absence of restriction-modification systems that degrade foreign DNA. The cells must also be competent for plasmid transformation and capable of sustaining continuous culture under flow conditions.

- describe E. coli host cells for M13-PACE  
E. coli host cells for M13-PACE are engineered to express the F-pilus, to carry the accessory and mutagenesis plasmids, and to be deficient in DNA repair pathways that would suppress mutagenesis. These cells are maintained in a turbidostat or cellstat to ensure exponential growth and consistent selective pressure.

- describe genotype of E. coli host cells for M13-PACE  
The genotype of E. coli host cells for M13-PACE is F′ lacIq ΔlacZ M15 ΔendA1 ΔhsdS5 ΔrpsL Δara Δgal ΔlacU169 ΔphoA ΔrecA ΔmutS. This genotype ensures efficient phage infection, prevents recombination of the selection cassette, and allows for controlled mutagenesis through the expression of error-prone polymerases.

- provide references to prior art  
References to prior art include the characterization of E. coli strains for PACE by Bain et al. (Nature Biotechnology, 2014), the optimization of F′ plasmids for phage propagation by Liu et al. (Science, 2017), and the use of ΔrecA strains to stabilize evolved genes by Kuchina et al. (Nature Methods, 2019).

### Methods for Evolving Recombinases

- introduce methods for evolving recombinases  
Methods for evolving recombinases involve the continuous propagation of mutant recombinase genes within a flowing culture of host cells, where recombinase activity is linked to the production of infectious phage particles. This process enables the rapid enrichment of variants with improved or altered DNA recognition specificity.

- describe contacting host cells with phage vectors  
Host cells are contacted with phage vectors by introducing a suspension of selection phage into a culture of host cells carrying the accessory and mutagenesis plasmids. The phage particles attach to the F-pilus and inject their single-stranded genome into the cell, where it is converted to double-stranded form and replicated.

- describe incubating host cells under conditions allowing mutation  
Host cells are incubated under conditions that promote mutagenesis, including the expression of error-prone DNA polymerase V from the mutagenesis plasmid. The temperature is maintained at 37°C, and the culture is aerated to ensure optimal growth and mutation rates.

- describe isolating replicated phage vectors  
Replicated phage vectors are isolated from the culture supernatant by centrifugation and filtration to remove bacterial cells, followed by precipitation with polyethylene glycol and sodium chloride. The resulting phage pellet is resuspended in buffer and used to infect fresh host cells.

- describe expression construct in host cells  
The expression construct in host cells consists of a recombinase gene cloned into the selection phage genome under the control of a promoter that is blocked by a transcriptional terminator flanked by recombinase recognition sites. Upon recombination, the terminator is excised, allowing transcription of the recombinase gene and pIII.

- describe recombination of recombinase target sequences  
Recombination of recombinase target sequences occurs when the recombinase protein binds to its cognate sites and catalyzes strand exchange, resulting in the excision of the transcriptional terminator. This event activates expression of pIII, enabling the production of infectious phage particles.

- describe excision of transcriptional terminator  
Excision of the transcriptional terminator is a key step in the selection process. The terminator is flanked by two recombinase recognition sites in direct orientation. When the recombinase acts on these sites, the intervening DNA is looped out and degraded, allowing RNA polymerase to access the pIII promoter.

- describe expression of genes for infectious phage particles  
Expression of genes for infectious phage particles occurs only after excision of the transcriptional terminator, which allows transcription of the pIII gene. The pIII protein is incorporated into the phage coat, enabling the particle to bind to the F-pilus of new host cells and initiate a new round of infection.

- describe mutated recombinase with higher efficiency  
A mutated recombinase with higher efficiency contains amino acid substitutions that enhance its binding affinity, catalytic rate, or specificity for the target sequence. These mutations are enriched over multiple generations of PACE and are identified by sequencing the recombinase gene from propagated phage.

- describe recombinase target sequences in target cells  
Recombinase target sequences in target cells are synthetic DNA sequences integrated into the genome or delivered on plasmids. They are designed to match the specificity of the evolved recombinase and are flanked by homologous arms for integration or contain regulatory elements for conditional expression.

- describe negative selection for undesired recombinase activity  
Negative selection for undesired recombinase activity is achieved by linking off-target recombination to the expression of a dominant-negative pIII protein. When the recombinase acts on non-cognate sites, the dominant-negative pIII is produced, which interferes with phage assembly and reduces infectivity.

- describe dominant-negative pIII protein  
The dominant-negative pIII protein is a mutant form of pIII that retains the ability to incorporate into the phage coat but disrupts particle assembly or infectivity. It is expressed under the control of a promoter activated by recombination at off-target sites, thereby selecting against recombinase variants with promiscuous activity.

- describe mutagenesis plasmid  
The mutagenesis plasmid is a low-copy-number plasmid encoding the UmuD′2C complex of E. coli DNA polymerase V under the control of a constitutive promoter. It introduces point mutations into the recombinase gene during replication, generating the genetic diversity necessary for evolution.

- describe continuous replenishment of host cells  
Continuous replenishment of host cells is achieved by maintaining a constant inflow of fresh, uninfected cells into the culture system while removing spent medium and lysed cells. This ensures that each generation of phage infects a new population of host cells, preventing the accumulation of non-productive variants.

### Evolved Recombinases

- define evolved recombinases  
Evolved recombinases are site-specific recombinases that have been subjected to continuous directed evolution and contain amino acid substitutions that alter their DNA recognition specificity. These enzymes are capable of recombining target sequences that differ substantially from their native substrates.

- describe amino acid sequence  
The amino acid sequence of evolved recombinases contains multiple substitutions relative to the wild-type enzyme, particularly in regions involved in DNA binding and catalysis. These substitutions are distributed across helices, loops, and β-sheets that contact the DNA backbone or bases.

- specify mutations  
Mutations in evolved recombinases include substitutions at positions such as R259A, G90D, Q94L, E262G, and K244A in Cre, which collectively reconfigure the DNA-binding interface to accommodate non-native sequences. These mutations are identified by sequencing the recombinase gene from evolved phage populations.

- describe recombinase target sequence  
The recombinase target sequence is a 30- to 40-base pair DNA sequence containing two inverted repeats flanking an asymmetric core region. It is designed to be recognized exclusively by the evolved recombinase and is not found in the human genome or other model organisms.

- specify differences from canonical sequence  
The evolved recombinase target sequence differs from the canonical loxP sequence by up to 68% of nucleotide positions. These differences are concentrated in the half-site regions, while the core sequence is preserved to maintain catalytic competence.

- describe length of target sequence  
The length of the recombinase target sequence is between 30 and 40 base pairs, with each half-site being 13 to 15 base pairs in length and the core region being 6 to 8 base pairs. This length ensures high specificity while maintaining sufficient sequence diversity for evolutionary optimization.

- specify type of recombinase  
The type of recombinase used in this invention is a tyrosine recombinase derived from Cre, Flp, or Bxb1. These enzymes catalyze recombination through a transesterification mechanism involving a conserved tyrosine residue that forms a covalent intermediate with the DNA backbone.

- describe canonical target sequence  
The canonical target sequence for Cre recombinase is loxP, a 34-base pair sequence consisting of two 13-base pair inverted repeats flanking an 8-base pair asymmetric core. This sequence is recognized with high specificity by wild-type Cre but is altered in evolved variants to enable recognition of novel targets.

- specify mutations in amino acid sequence  
Mutations in the amino acid sequence of evolved recombinases include substitutions at residues that directly contact DNA, such as Arg259, Gln90, and Glu262, as well as residues that stabilize the protein fold or mediate dimerization. These mutations are identified by comparing the sequences of evolved clones to the parental enzyme.

- describe recognition of target sequence  
Recognition of the target sequence by evolved recombinases is mediated by a combination of direct hydrogen bonding, shape complementarity, and water-mediated interactions. The enzyme binds the target sequence with high affinity and catalyzes recombination with efficiency comparable to or exceeding that of wild-type Cre on loxP.

- specify location of target sequence  
The location of the target sequence is within a safe harbor genomic locus, such as AAVS1, ROSA26, or CCR5, or on an episomal vector. The sequence is flanked by homologous arms for integration or by regulatory elements for conditional expression.

- describe safe harbor genomic locus  
A safe harbor genomic locus is a region of the genome that is permissive for transgene insertion without disrupting endogenous gene expression or regulatory elements. Examples include AAVS1 in humans, ROSA26 in mice, and LP1 in zebrafish. These loci are characterized by open chromatin, high transcriptional activity, and low risk of insertional mutagenesis.

- specify pharmaceutical composition  
A pharmaceutical composition comprises a nucleic acid encoding an evolved recombinase, a delivery vehicle such as an adeno-associated virus or lipid nanoparticle, and a pharmaceutically acceptable carrier. The composition is formulated for intravenous, intramuscular, or local administration.

- describe administration of composition  
Administration of the composition is performed by injection into a subject, with dosage determined by body weight, target tissue, and disease state. The recombinase is delivered in a form that enables transient expression, minimizing the risk of immune response or off-target activity.

- specify use of composition  
The use of the composition is for the precise correction of disease-causing mutations, the insertion of therapeutic transgenes, or the excision of pathogenic DNA sequences such as integrated viral genomes. It is particularly useful in the treatment of monogenic disorders, cancer, and chronic viral infections.

### Methods For Recombinase-Mediated Genetic Engineering

- describe method for engineering nucleic acid molecule  
The method for engineering a nucleic acid molecule involves introducing a recombinase recognition site into a target genomic locus using homologous recombination or viral delivery, followed by expression of an evolved recombinase that recognizes the site with high specificity.

- specify contacting nucleic acid molecules with recombinase  
Contacting nucleic acid molecules with recombinase involves delivering the recombinase as a protein, mRNA, or encoding gene into a cell containing the target sequence. The recombinase binds to its recognition site and catalyzes recombination, resulting in deletion, inversion, or insertion of the intervening DNA.

- describe recombination of target sequences  
Recombination of target sequences occurs when the recombinase binds to two recognition sites in direct or inverted orientation and mediates strand exchange. This results in the precise excision, inversion, or integration of the DNA segment between the sites, depending on their orientation.

- specify differences from canonical target sequence  
The target sequence differs from the canonical loxP sequence by at least 20% of nucleotide positions, with substitutions concentrated in the half-site regions. These differences are sufficient to prevent recognition by wild-type recombinases but are optimized for high-efficiency recombination by the evolved enzyme.

- describe administration of composition  
Administration of the composition is performed by systemic or localized delivery using viral vectors, lipid nanoparticles, or electroporation. The recombinase is expressed transiently to minimize off-target effects and immune recognition.

- specify use of method  
The use of the method is for the therapeutic correction of genetic diseases, the engineering of cell therapies, or the construction of synthetic gene circuits. It is particularly valuable in applications requiring scarless, irreversible genome modifications without double-strand breaks.

### Methods for Evaluating the Specificity of Recombinases

- describe method for identifying target site  
The method for identifying a target site involves exposing a library of randomized DNA sequences to the recombinase, followed by exonuclease digestion to remove non-recombined substrates. The enriched products are amplified and sequenced to determine the nucleotide preferences at each position of the target site.

- specify providing recombinase  
Providing recombinase involves purifying the enzyme from bacterial lysates or expressing it in vitro using cell-free transcription-translation systems. The recombinase is used at a concentration that ensures saturation of the target library without inducing non-specific recombination.

- describe contacting recombinase with library  
Contacting recombinase with library involves incubating the purified enzyme with a pool of synthetic oligonucleotides containing randomized regions flanked by conserved sequences. The library contains up to 10^11 unique sequences, ensuring comprehensive sampling of possible target variants.

- specify structure of candidate nucleic acid molecules  
The structure of candidate nucleic acid molecules consists of a double-stranded DNA oligonucleotide with a randomized region of 15–20 base pairs flanked by fixed sequences that permit recombination only if the recombinase recognizes the target. Each molecule contains a unique molecular identifier for accurate quantification.

- describe recombination of target sequences  
Recombination of target sequences occurs when the recombinase binds to its cognate site and catalyzes strand exchange, resulting in a recombined product that is resistant to exonuclease digestion. Non-recombined molecules are degraded, leaving only functional substrates for amplification.

- specify identifying recombinase target sites  
Identifying recombinase target sites involves sequencing the recombined products and aligning them to the wild-type sequence to determine the enrichment of each nucleotide at each position. Sites with high enrichment scores are considered high-affinity targets.

- describe determining sequence of recombined molecule  
Determining the sequence of the recombined molecule is performed by high-throughput sequencing using Illumina MiSeq or NovaSeq platforms. The reads are demultiplexed, aligned to the reference sequence, and analyzed for enrichment of specific nucleotides at each position.

- specify enriching amplified molecules  
Enriching amplified molecules involves performing quantitative PCR to ensure that the recombined products are amplified within the linear range and that background amplification from non-recombined molecules is minimized.

- describe sequencing recombined molecule  
Sequencing the recombined molecule is performed using paired-end or single-end reads of 250–300 bases, with sufficient depth to detect rare variants. The data are analyzed using custom software to calculate enrichment scores and specificity profiles.

- specify evaluating specificity of recombinase  
Evaluating specificity of recombinase involves comparing the enrichment profile of the evolved enzyme to that of the wild-type enzyme. A high specificity score indicates that the enzyme preferentially recombines the intended target sequence with minimal activity on off-target sites.

### Libraries for Assessing Recombinase Target Site Preferences

- describe library of nucleic acid molecules  
A library of nucleic acid molecules is a heterogeneous pool of synthetic DNA sequences containing randomized regions designed to sample the full range of possible target sites for a recombinase. Each molecule contains a unique molecular identifier and is flanked by conserved sequences that permit recombination only upon recognition by the enzyme.

- specify structure of candidate nucleic acid molecules  
The structure of candidate nucleic acid molecules consists of a hairpin oligonucleotide with a randomized region of 15–20 base pairs flanked by fixed sequences that serve as priming sites for extension. The randomized region corresponds to the half-site of the recombinase target, allowing the assessment of binding preferences at each position.

- describe loop sequence  
The loop sequence is the unpaired region of the hairpin that connects the two complementary arms. It is designed to be stable at the annealing temperature and to facilitate primer extension without interfering with recombinase binding.

- specify number of different half-site sequences  
The library contains at least 10^9 different half-site sequences, each differing from the canonical sequence by up to seven nucleotide substitutions. This ensures comprehensive coverage of the sequence space and enables the identification of rare, high-affinity variants.

- describe use of library  
The use of the library is to generate a high-resolution specificity profile of the recombinase, revealing which nucleotides are preferred, tolerated, or disfavored at each position of the target site. This information is used to design synthetic target sequences with optimal recognition properties.

### Vectors and Reagents

- provide selection phage with phage genome deficient in gene required for infectious phage particles  
The selection phage is an M13 phage genome that has been modified to lack the gene encoding the pIII protein, rendering it non-infectious unless complemented by the accessory plasmid. The pIII gene is placed under the control of a promoter that is activated only upon recombination of a target sequence.

- describe gene encoding recombinase of interest  
The gene encoding the recombinase of interest is cloned into the selection phage genome downstream of the conditional promoter. It is expressed only after recombination excises the transcriptional terminator, ensuring that phage propagation is strictly dependent on recombinase activity.

- detail multiple cloning site for insertion of nucleic acid sequence  
The multiple cloning site is a region of the selection phage genome containing a series of unique restriction enzyme sites, including EcoRI, BamHI, and XhoI, that allow for the insertion of recombinase target sequences or regulatory elements.

- describe M13 phage genome with genes required for phage life cycle  
The M13 phage genome contains genes required for phage life cycle, including those encoding the pVIII coat protein, pVI and pVII for DNA packaging, and pII and pXI for replication. These genes are retained in the selection phage to ensure proper assembly and propagation.

- provide helper phage complementing selection phage genome  
The helper phage is a modified M13 phage that provides the pIII gene in trans to the selection phage. It contains a functional pIII gene under the control of a constitutive promoter and is used to initiate the first round of infection.

- describe mutagenesis plasmid with gene expression cassette  
The mutagenesis plasmid contains a gene expression cassette encoding the UmuD′2C complex of E. coli DNA polymerase V under the control of a constitutive promoter. It is maintained at low copy number to ensure controlled mutagenesis without cellular toxicity.

- detail components of E. coli translesion synthesis polymerase V  
The components of E. coli translesion synthesis polymerase V include the UmuD′ and UmuC proteins, which form a heterotrimeric complex that introduces point mutations during DNA replication. The complex is expressed from the mutagenesis plasmid to generate genetic diversity in the recombinase gene.

- describe deoxyadenosine methylase and hemimethylated-GATC binding domain  
The deoxyadenosine methylase and hemimethylated-GATC binding domain are not utilized in this invention. The system operates independently of methylation status, and all DNA substrates are synthesized as unmethylated oligonucleotides to avoid interference with recombination efficiency.

### Expression Constructs

- provide nucleic acids encoding recombinases  
Nucleic acids encoding recombinases include DNA sequences derived from Cre, Flp, Bxb1, Dre, and VCre, cloned into expression vectors under the control of inducible or constitutive promoters. These constructs are used to deliver the recombinase to target cells for genome engineering.

- describe heterologous promoter controlling expression  
A heterologous promoter controlling expression is a promoter from a different organism or system that drives expression of the recombinase gene in the host cell. Examples include the CMV promoter for mammalian cells, the T7 promoter for bacterial expression, and the EF1α promoter for stem cells.

- detail expression construct, e.g., plasmid, viral vector, or linear expression construct  
The expression construct is a plasmid, viral vector, or linear DNA fragment containing the recombinase gene, a promoter, a polyadenylation signal, and a selectable marker. It is delivered to cells via electroporation, lipofection, or viral transduction.

- describe nucleic acid or expression construct in cell, tissue, or organism  
The nucleic acid or expression construct is introduced into a cell, tissue, or organism by direct injection, systemic delivery, or ex vivo transduction. The recombinase is expressed transiently to mediate precise genome modification without permanent integration of the delivery vector.

- define vector and its uses  
A vector is a nucleic acid molecule used to deliver genetic material into a host cell. In this invention, vectors include plasmids, adeno-associated viruses, lentiviruses, and lipid nanoparticles used to deliver recombinase genes or mRNA for genome engineering.

- describe non-viral and viral vectors  
Non-viral vectors include plasmids, synthetic nanoparticles, and electroporation devices, which are safe and scalable but often have low delivery efficiency. Viral vectors include AAV, lentivirus, and adenovirus, which provide high transduction efficiency but may elicit immune responses.

- detail regulatory sequences, e.g., promoters, enhancers, and transcriptional termination sequences  
Regulatory sequences include promoters such as CMV, EF1α, and CAG; enhancers such as the woodchuck hepatitis virus post-transcriptional regulatory element; and transcriptional termination sequences such as the SV40 polyA signal. These elements are used to control the timing, level, and duration of recombinase expression.

- describe operably linked regulatory or control sequences  
Operably linked regulatory or control sequences are DNA elements that are functionally connected to the recombinase gene to ensure its proper expression. For example, a promoter is operably linked to the recombinase coding sequence such that transcription initiates at the promoter and proceeds through the gene.

- provide cells expressing evolved recombinase  
Cells expressing evolved recombinase are generated by transducing or transfecting primary human cells, stem cells, or immortalized cell lines with a vector encoding the evolved recombinase. These cells are used for ex vivo genome engineering or as models for in vivo delivery studies.

### Host Cells

- provide host cells for continuous evolution processes  
Host cells for continuous evolution processes are E. coli strains carrying the F′ plasmid, the accessory plasmid, and the mutagenesis plasmid. These cells are maintained in a continuous-flow bioreactor to enable sustained selection for recombinase activity.

- describe host cells for phage-assisted continuous evolution processes  
Host cells for phage-assisted continuous evolution processes are E. coli strains such as NEB Turbo, Top10F′, and ER2738 that express the F-pilus, lack restriction systems, and support high-titer phage production. They are engineered to contain the accessory and mutagenesis plasmids under antibiotic selection.

- detail accessory plasmid with gene required for infectious phage particles  
The accessory plasmid contains the pIII gene under the control of a recombinase-dependent promoter. It is maintained at low copy number to ensure that pIII expression is tightly coupled to recombinase activity.

- describe host cell providing phage functions  
The host cell provides phage functions by expressing the F-pilus for phage attachment and the replication machinery for phage genome synthesis. It also supplies the nucleotides, enzymes, and energy required for phage assembly.

- provide modified viral vectors lacking gene required for infectious viral particles  
Modified viral vectors lack the gene encoding pIII and are non-infectious unless complemented by the accessory plasmid. They are used to deliver the recombinase gene under the control of a conditional promoter.

- describe host cell comprising gene required for infectious viral particles  
The host cell comprises the pIII gene on the accessory plasmid, which is expressed only upon recombination of the target sequence. This ensures that phage propagation is strictly dependent on recombinase activity.

- detail host cell comprising helper construct providing viral genes  
The host cell comprises a helper construct that provides essential phage genes such as pVIII and pVI in trans to the selection phage. This construct is maintained on a separate plasmid or integrated into the chromosome.

- describe prokaryotic host cell, e.g., bacterial cell  
The prokaryotic host cell is an E. coli strain that supports the replication of M13 phage and the expression of recombinase and accessory genes. It is grown in rich medium under constant aeration and flow.

- provide E. coli host cell  
The E. coli host cell is strain NEB Turbo, which carries the F′ plasmid, is deficient in endonucleases, and supports high-titer phage production. It is transformed with the accessory, mutagenesis, and selection plasmids.

- describe eukaryotic host cell, e.g., yeast cell, insect cell, or mammalian cell  
Eukaryotic host cells are not used in the PACE system but are used for testing recombinase activity in therapeutic applications. Examples include HEK293T cells, iPSCs, and primary T cells.

- detail viral vector and host cell combinations  
Viral vector and host cell combinations include AAV9 delivery to hepatocytes, lentivirus delivery to hematopoietic stem cells, and lipid nanoparticle delivery to neurons. These combinations are optimized for tissue-specific transduction and transient expression.

- describe phage and host cell combinations  
Phage and host cell combinations include M13 phage and E. coli strain ER2738, which is engineered to express the F-pilus and carry the accessory and mutagenesis plasmids. This combination supports continuous evolution of recombinases over hundreds of generations.

- provide E. coli host strains, e.g., NEB Turbo, Top10F′, DH12S, ER2738, ER2267  
E. coli host strains suitable for PACE include NEB Turbo, Top10F′, DH12S, ER2738, and ER2267. These strains carry the F′ plasmid, are deficient in restriction systems, and support high-titer phage production.

- describe Fertility factor and its role in conjugation  
The Fertility factor is a conjugative plasmid that encodes the genes required for pilus formation and DNA transfer. In this invention, the F′ plasmid is essential for M13 phage infection, as the phage uses the F-pilus as its receptor.

- provide host cells for M13-PACE, e.g., E. coli cells with F′ plasmid  
Host cells for M13-PACE are E. coli cells carrying the F′ plasmid, the accessory plasmid, and the mutagenesis plasmid. These cells are maintained in a continuous-flow system to enable sustained selection for recombinase activity.

## EXAMPLES

- develop PACE selection for altering DNA specificity of Cre recombinase  
A PACE system was developed to evolve Cre recombinase to recognize a non-native target sequence, RosaLox, which differs from loxP at 18 nucleotide positions. The selection phage was constructed by inserting two RosaLox sites flanking a transcriptional terminator upstream of the pIII gene. The accessory plasmid provided pIII under constitutive control. Host cells were transformed with the mutagenesis plasmid and maintained in a turbidostat at 37°C with a 1.5-hour residence time.

- generate Cre deletion-dependent Accessory Plasmid (AP)  
A Cre deletion-dependent accessory plasmid was generated by placing the pIII gene under the control of a promoter that is only activated upon excision of a 200-bp fragment flanked by loxP sites. In the absence of Cre activity, the promoter is blocked, preventing pIII expression and phage propagation.

- design selection system for selecting recombinase activity  
The selection system was designed such that only recombinase variants capable of recombining RosaLox could excise the terminator and activate pIII expression. Phage particles produced by active variants infected fresh host cells, propagating the mutation to subsequent generations.

- perform PACE experiment with host cells harboring AP  
A PACE experiment was performed by inoculating a culture of E. coli ER2738 with selection phage and maintaining it in a turbidostat for 120 hours. Fresh cells were continuously supplied, and phage was harvested every 24 hours for sequencing.

- validate selection strategy using Cre-selection phagemid  
The selection strategy was validated by introducing a plasmid encoding wild-type Cre into the host cells. This resulted in rapid enrichment of phage particles, confirming that the system was responsive to recombinase activity.

- test enrichment of Cre-encoding selection phagemid  
Deep sequencing of phage populations revealed a 10^5-fold enrichment of phage encoding evolved Cre variants after 120 hours, compared to the initial population. The dominant variant contained seven amino acid substitutions.

- evolve inactive Cre proteins to recognize native LoxP site  
A library of inactive Cre mutants was subjected to PACE using a loxP-flanked terminator. After 72 hours, variants capable of recombining loxP were enriched, demonstrating the system’s ability to restore function.

- select for recombinases recognizing Rosa26 locus  
A target sequence derived from the human ROSA26 locus was designed and inserted into the selection phage. After 100 hours of PACE, variants were identified that recombined the Rosa26-derived sequence with high efficiency and minimal off-target activity.

- design retargeting evolution strategy  
A retargeting evolution strategy was designed by introducing a series of mutations into the Cre gene that were predicted to alter DNA recognition. These mutations were introduced via site-directed mutagenesis and subjected to PACE under increasing stringency.

- evolve Cre towards asymmetric RosaLox site  
The RosaLox site was designed to be asymmetric, with distinct half-sites. PACE was performed under conditions that favored recombination of the asymmetric sequence. After 96 hours, a variant was identified that recombined RosaLox with 92% efficiency and showed no activity on loxP.

- illustrate evolution of recombinases recognizing L1 intermediate target sequence  
An intermediate target sequence, L1, was designed to contain partial homology to both loxP and RosaLox. PACE was performed using L1 as the selection target. After 80 hours, a variant was identified that recombined L1 with high efficiency and showed intermediate activity on both loxP and RosaLox.

- assess recombinase activity on wild-type LoxP and L1 target sequences  
The evolved recombinase was tested on wild-type loxP and L1 sequences in vitro. Activity on loxP was reduced by 100-fold, while activity on L1 was increased 50-fold, demonstrating successful retargeting.

- identify converged mutations in evolved clones  
Deep sequencing of evolved clones revealed that mutations at positions R259A, G90D, Q94L, and E262G were consistently enriched across independent PACE runs, indicating convergence to a functional solution.

- observe retention of activity on wild-type LoxP site  
The evolved recombinase retained low but detectable activity on wild-type loxP, which was eliminated by introducing a second round of negative selection using a dominant-negative pIII construct linked to loxP recombination.

- measure activity on RosaLox site in mammalian cells  
The evolved recombinase was delivered to HEK293T cells via AAV9 and tested for recombination of a RosaLox-flanked reporter cassette. Recombination efficiency was 87%, with no detectable off-target events by whole-genome sequencing.

- design negative PACE selection strategy  
A negative PACE selection strategy was designed by linking recombination at off-target sites to the expression of a dominant-negative pIII protein. This ensured that recombinase variants with promiscuous activity were selected against.

- link unwanted recombinase activity to production of dominant negative pIIIneg  
The dominant-negative pIIIneg protein was created by fusing a truncated pIII sequence to a dimerization domain. When expressed, it incorporated into phage particles and disrupted assembly, reducing infectivity by 99%.

- adjust selection stringency using different promoters  
Selection stringency was adjusted by replacing the pIII promoter with weaker variants, such as the lacUV5 promoter. This increased the threshold for phage propagation and selected for higher-affinity variants.

- illustrate results of negative selection  
After three rounds of negative selection, the evolved recombinase showed no detectable activity on loxP, AAVS1, or CCR5, while maintaining 90% activity on RosaLox.

- describe in vitro method to measure specificity of recombinases  
An in vitro method was developed using a library of 10^10 randomized sequences flanked by fixed recombination arms. The library was incubated with purified recombinase, followed by exonuclease digestion and high-throughput sequencing.

- develop recombinase profiling workflow  
The recombinase profiling workflow involved synthesizing hairpin oligonucleotides, extending them with Klenow fragment, incubating with recombinase, digesting with exonucleases, amplifying by PCR, and sequencing on an Illumina MiSeq.

- extend library oligos across randomized portion  
Library oligos were extended across the randomized region using Klenow fragment to generate double-stranded substrates required for recombination. The extension reaction was optimized to ensure uniform coverage of all sequence variants.

- treat with exonucleases to remove unreacted oligos  
The reaction was treated with exonucleases I, III, and V to degrade non-recombined substrates. Only recombined products with hairpins on both ends were resistant to digestion.

- PCR amplify and prepare for high-throughput sequencing  
The recombined products were amplified using primers containing Illumina adapters and barcodes. The library was quantified and sequenced to a depth of 10^7 reads per sample.

- generate sequence logos for post-selection abundance of DNA bases  
Sequence logos were generated using custom Python software to visualize the enrichment of each nucleotide at each position. The logos revealed strong preferences for specific bases at positions 5, 6, 10, and 16.

- calculate overall specificity score  
The overall specificity score was calculated as the geometric mean of enrichment ratios across all positions. The evolved recombinase achieved a score of 4.2, compared to 0.8 for wild-type Cre on RosaLox.

- analyze sequence logos for subset of sequences with mismatch  
Analysis of sequences containing single mismatches revealed that the enzyme tolerated substitutions at positions 7 and 12 but strongly disfavored changes at positions 5 and 10.

- observe relaxed specificity for distal bases  
The enzyme showed relaxed specificity for distal bases beyond position 17, indicating that the recognition interface is focused on the central half-site region.

- identify residues making specific contacts at position 8  
Structural modeling identified Gln90 and Arg259 as residues making direct hydrogen bonds with the base at position 8. Mutations at these positions altered specificity at this site.

- perform parallel profiling experiments using evolved Cre mutants  
Parallel profiling was performed on 12 evolved Cre mutants, each containing different combinations of mutations. The results revealed that mutations at Gln90 and Glu262 were synergistic in enhancing specificity for RosaLox.

- analyze specificity of mutants for LoxP and RosaLox oligos  
The specificity of each mutant was measured on both loxP and RosaLox substrates. The best variant showed a 200-fold preference for RosaLox over loxP.

- observe loss of preference at position 8  
The evolved recombinase lost preference for the canonical base at position 8, consistent with the introduction of a glycine substitution that altered the DNA-binding interface.

- identify unique binding mode for positions 12 and 13  
The evolved recombinase adopted a novel binding mode at positions 12 and 13, involving a water-mediated hydrogen bond network not observed in wild-type Cre.

- investigate relationship between DNA-binding protein and nucleic acid substrate  
The relationship between the recombinase and its substrate was investigated using molecular dynamics simulations. The simulations revealed that the evolved enzyme formed a more extensive network of van der Waals contacts with the DNA backbone.

- profile specificity of single mutants of Cre residues contacting DNA  
Single mutants of residues contacting DNA were profiled using Rec-seq. The results showed that mutations at Gln90 and Arg259 had the largest impact on specificity, while mutations at Lys244 had minimal effect.

- study specificity of relatives of Cre using profiling methods  
The specificity of Dre and VCre recombinases was profiled using the same method. Both enzymes showed asymmetric recognition patterns, with strong preferences at positions 6, 7, and 12, consistent with their native target sequences.