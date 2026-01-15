## ACKNOWLEDGMENT OF GOVERNMENT SUPPORT

This invention was made with government support under Award Number DE-EE0009648 awarded by the Department of Energy. The United States government has certain rights in this invention. The research and development efforts described herein were conducted in part through the Agile BioFoundry, a public-private partnership established to accelerate the design, build, test, and learn cycles for engineered biological systems. Funding and infrastructure provided by the Department of Energy enabled the systematic metabolic engineering, multi-omic analysis, and scale-up studies that led to the development of the recombinant Aspergillus strains and associated methods described herein. No commercial endorsement is implied by this support, and all intellectual property rights remain vested in the inventors and their affiliated institutions.

## FIELD

The present invention relates to recombinant fungi of the genus Aspergillus, specifically engineered to produce organic acids including aconitic acid and 3-hydroxypropionate, through targeted genetic modifications that alter native metabolic fluxes. These fungi are modified to function as biocatalysts for the sustainable production of platform chemicals from renewable carbon sources such as glucose and other sugars derived from lignocellulosic biomass. The invention particularly concerns strains in which the native cis-aconitic acid decarboxylase gene has been genetically inactivated, thereby redirecting metabolic intermediates away from itaconic acid biosynthesis and toward the accumulation of aconitic acid and its isomers. Further modifications include the introduction of exogenous nucleic acid sequences encoding enzymes involved in the biosynthesis of 3-hydroxypropionate, enabling the conversion of central carbon metabolites into high-value chemical precursors. These recombinant organisms are designed for industrial-scale fermentation under acidic conditions, leveraging the natural acid tolerance and high glycolytic capacity of Aspergillus species.

## BACKGROUND

Itaconic acid has been industrially produced for decades using filamentous fungi, particularly strains of Aspergillus terreus and related species, through the fermentation of glucose. The biosynthesis of itaconic acid proceeds via the tricarboxylic acid cycle, wherein citrate is isomerized to cis-aconitate, which is then decarboxylated by the enzyme cis-aconitic acid decarboxylase (cadA) to yield itaconic acid. This pathway is tightly regulated and highly efficient under specific fermentation conditions, often resulting in titers exceeding 100 g/L. However, the enzymatic activity of cadA is the decisive step that diverts carbon flux away from the accumulation of cis-aconitic acid and trans-aconitic acid, which are structurally similar but chemically distinct intermediates with higher potential as platform chemicals in polymer and fine chemical synthesis. The regulatory mechanisms governing the expression of the itaconic acid gene cluster, including genes such as mttA and mfsA, which encode transporters for itaconic acid efflux, have been partially characterized, but the precise transcriptional and post-translational control elements remain incompletely understood. Moreover, the metabolic consequences of disrupting cadA activity—particularly the redirection of carbon flux toward alternative organic acids—have not been systematically exploited for the production of non-itaconic acid products. Prior efforts to manipulate this pathway have focused on enhancing itaconic acid yield or eliminating by-products, but none have leveraged the accumulation of cis-aconitic acid as a primary product, nor have they combined cadA inactivation with the introduction of heterologous pathways for 3-hydroxypropionate production.

## SUMMARY

The cadA gene encodes cis-aconitic acid decarboxylase, a key enzyme responsible for the conversion of cis-aconitic acid to itaconic acid in Aspergillus species. Genetic inactivation of cadA results in the accumulation of cis-aconitic acid and its isomer, trans-aconitic acid, which are otherwise rapidly consumed in wild-type strains. Expressed sequence tag (EST) analysis of the itaconic acid gene cluster revealed coordinated upregulation of cadA, mttA, and mfsA during peak production phases, confirming their functional association. Deletion of cadA in Aspergillus pseudoterreus led to a complete cessation of itaconic acid production and a corresponding increase in intracellular and extracellular concentrations of cis-aconitic acid and trans-aconitic acid, demonstrating that carbon flux is effectively diverted from itaconic acid biosynthesis. This redirection enables the production of aconitic acid, a compound recognized by the U.S. Department of Energy as one of the top 30 potential bio-based chemical building blocks. A recombinant Aspergillus strain with a genetically inactivated cadA gene was developed and characterized for its ability to accumulate aconitic acid at yields exceeding those achievable by chemical synthesis. Exogenous nucleic acid molecules encoding aspartate 1-decarboxylase (panD), β-alanine-pyruvate aminotransferase (BAPAT), and 3-hydroxypropionate dehydrogenase (HPDH) were introduced into this strain to enable the biosynthesis of 3-hydroxypropionate from pyruvate-derived precursors. Vector constructs incorporating strong constitutive promoters such as gpdA and eno1 were used to drive high-level expression of these heterologous genes. The genetic inactivation of cadA was achieved through homologous recombination, resulting in precise deletion or disruption of the cadA coding sequence. Compositions comprising these recombinant strains, as well as kits containing the necessary vectors, primers, and culture media for their generation and propagation, are disclosed. Methods for producing aconitic acid and 3-hydroxypropionate using these strains are described, including optimized fermentation conditions and downstream purification protocols.

## DETAILED DESCRIPTION

### Define technical terms

For the purposes of this disclosure, the term “recombinant” refers to an organism or nucleic acid molecule that has been artificially modified through the introduction of exogenous genetic material or the targeted alteration of endogenous sequences using molecular biological techniques. The term “isolated” means that a nucleic acid, protein, or metabolite has been separated from its native cellular environment to a degree sufficient for use in industrial or analytical applications. The term “exogenous” denotes any nucleic acid, protein, or genetic element introduced into a host organism from an external source, regardless of phylogenetic origin. The term “genetic inactivation” encompasses any method that results in a substantial reduction or complete abolition of the functional activity of a gene product, including deletion, insertion, point mutation, silencing, or promoter disruption. The term “detectable” refers to the ability to identify the presence or absence of a molecule, gene, or protein using analytical methods such as chromatography, spectroscopy, hybridization, or immunoassay. The term “approximately” when used in reference to numerical values, such as concentrations, yields, or sequence identities, indicates a range of ±10% unless otherwise specified. The term “variant” refers to a nucleic acid or protein sequence that differs from a reference sequence by one or more substitutions, insertions, deletions, or additions, yet retains substantially the same biological function. The term “sequence identity” is defined as the percentage of identical residues aligned between two sequences over a specified region, calculated using standard alignment algorithms such as BLAST. The term “transformed” describes a host cell that has stably incorporated exogenous nucleic acid into its genome or episomal elements such that the introduced genetic material is heritable and expressed. The term “vector” refers to a nucleic acid construct capable of delivering and expressing exogenous genes in a host organism, typically comprising a promoter, coding sequence, terminator, and selectable marker. The term “promoter” refers to a DNA sequence that directs the transcription of a downstream gene by RNA polymerase, and includes both native and heterologous sequences capable of driving constitutive or inducible expression. The term “bi-directional promoter” refers to a single regulatory region capable of initiating transcription in both orientations, enabling the coordinated expression of two adjacent genes. The term “conservative amino acid substitution” refers to the replacement of an amino acid with another having similar physicochemical properties, such as size, charge, or hydrophobicity, such that the overall structure and function of the protein are preserved.

### Explain singular and plural terms

In this disclosure, the use of a singular noun or pronoun includes its plural counterpart unless the context clearly requires otherwise. For example, reference to “a gene” includes one or more genes, and reference to “a strain” includes one or more strains. Similarly, the use of a plural noun or pronoun may encompass a singular entity when the context permits, such as when describing a population of cells or a collection of variants. The terms “comprising,” “including,” and “having” are used interchangeably and are intended to be open-ended, meaning that additional elements, steps, or components may be present without excluding the recited features. The term “consisting essentially of” is used to denote that the invention may include additional components that do not materially affect the basic and novel characteristics of the claimed invention.

### Explain "or" and "and" usage

The term “or” is used in its inclusive sense, meaning that one or more of the listed alternatives may be selected unless the context clearly indicates exclusivity. For example, a strain may comprise a deletion of cadA or expression of panD, or both. The term “and” is used to indicate the simultaneous presence or occurrence of multiple elements, such that both conditions must be satisfied unless otherwise indicated. For instance, a method requiring both genetic inactivation of cadA and expression of HPDH implies that both modifications are necessary for the intended function.

### Explain approximate values

All numerical values provided in this disclosure, including concentrations, yields, temperatures, pH levels, and sequence identities, are approximate unless explicitly stated as exact. These values are intended to encompass experimental variability inherent in biological systems and industrial processes. For example, a yield of “approximately 0.8 g/L” includes values ranging from 0.72 g/L to 0.88 g/L. Similarly, a sequence identity of “at least 85%” includes values from 85% to 100%, and any intermediate value within that range. Approximate values are determined based on analytical precision, statistical significance, and reproducibility across multiple independent experiments.

### Incorporate references by reference

All publications, patents, and patent applications cited herein are incorporated by reference in their entirety. These include, but are not limited to, the U.S. Department of Energy’s list of top bio-based chemicals, the genome-scale metabolic model iJB1325 for Aspergillus niger, and methods for protoplast transformation and Gibson assembly. The disclosures of these references provide background, context, and technical details that support the practice of the invention as described herein.

### Define 3-hydroxypropionate dehydrogenase (HPDH)

3-Hydroxypropionate dehydrogenase (HPDH) is an NAD(P)H-dependent oxidoreductase enzyme that catalyzes the reversible conversion of malonate semialdehyde to 3-hydroxypropionate. This enzyme plays a critical role in the biosynthesis of 3-hydroxypropionate by reducing the aldehyde group of malonate semialdehyde to a primary alcohol, thereby forming the desired product. HPDH is naturally found in certain bacteria and archaea involved in carbon fixation pathways, and its activity is essential for the accumulation of 3-hydroxypropionate in engineered microbial systems.

### Explain HPDH enzyme function

The HPDH enzyme functions by binding malonate semialdehyde and a cofactor, either NADH or NADPH, and facilitating the transfer of a hydride ion from the cofactor to the carbonyl carbon of malonate semialdehyde. This reduction results in the formation of 3-hydroxypropionate and the oxidized form of the cofactor. The enzyme exhibits substrate specificity for malonate semialdehyde and does not efficiently act on structurally similar compounds such as glyoxylate or pyruvate. Its activity is optimal under slightly acidic to neutral pH conditions and at temperatures between 30°C and 40°C, making it compatible with the fermentation conditions used for Aspergillus species.

### Provide HPDH sequence examples

An exemplary HPDH sequence is derived from Escherichia coli (UniProt ID: P0A9K7), comprising 312 amino acids with a molecular weight of approximately 34 kDa. Another functional variant is derived from Bacillus subtilis (UniProt ID: Q45262), which shares 68% sequence identity with the E. coli homolog. A codon-optimized version of the E. coli HPDH gene, adapted for expression in Aspergillus species, is provided in SEQ ID NO: 1, which includes modifications to enhance translation efficiency and mRNA stability in fungal hosts.

### Explain HPDH variant sequences

Variant sequences of HPDH include those with conservative amino acid substitutions, insertions, or deletions that maintain enzymatic activity while potentially improving stability, solubility, or catalytic efficiency. Such variants may be identified through directed evolution, site-directed mutagenesis, or computational modeling. For example, substitution of alanine at position 120 with valine in the E. coli HPDH sequence enhances thermostability without compromising activity. Variant sequences are considered functionally equivalent if they retain at least 80% sequence identity to a reference HPDH sequence and demonstrate the ability to convert malonate semialdehyde to 3-hydroxypropionate at a rate not less than 50% of the wild-type enzyme under identical assay conditions.

### Define aconitic acid

Aconitic acid is a tricarboxylic acid isomer of citric acid, existing in three forms: cis-aconitic acid, trans-aconitic acid, and the unstable intermediate form. It is a naturally occurring intermediate in the tricarboxylic acid cycle, formed by the dehydration of citrate and the subsequent hydration of isocitrate. In the context of this invention, aconitic acid refers to the extracellular accumulation of cis-aconitic acid and trans-aconitic acid as primary fermentation products, resulting from the genetic inactivation of cadA in Aspergillus strains.

### Define aspartate 1-decarboxylase (panD)

Aspartate 1-decarboxylase (panD) is a pyruvoyl-dependent enzyme that catalyzes the decarboxylation of L-aspartate to produce β-alanine and carbon dioxide. This reaction is the first committed step in the biosynthesis of pantothenic acid and is essential for the production of coenzyme A in most organisms. In the context of this invention, panD is introduced as a heterologous enzyme to generate β-alanine from aspartate, serving as a precursor for the subsequent synthesis of 3-hydroxypropionate.

### Explain panD enzyme function

The panD enzyme functions by cleaving the carboxyl group adjacent to the α-carbon of L-aspartate, resulting in the release of CO₂ and the formation of β-alanine. The enzyme utilizes a pyruvoyl group as a cofactor, which is autocatalytically generated from a serine residue within its own polypeptide chain. This self-processing mechanism renders panD independent of external cofactors, making it particularly suitable for heterologous expression in microbial hosts. The enzyme operates optimally at pH 7.0–8.0 and temperatures between 30°C and 37°C, although its activity is retained under the slightly acidic conditions of Aspergillus fermentations.

### Provide panD sequence examples

An exemplary panD sequence is derived from Tribolium castaneum (UniProt ID: Q8IPM1), comprising 198 amino acids. A codon-optimized version for expression in Aspergillus species is provided in SEQ ID NO: 2. Another functional variant is derived from Saccharomyces cerevisiae (UniProt ID: P38849), which shares 42% sequence identity with the T. castaneum homolog but retains comparable enzymatic activity when expressed in fungal hosts.

### Explain panD variant sequences

Variant sequences of panD include those with amino acid substitutions that enhance enzyme stability, solubility, or catalytic turnover under acidic conditions. For example, substitution of glutamine at position 105 with lysine in the T. castaneum panD sequence increases enzyme half-life at pH 3.5 by 2.3-fold. Variant sequences are considered functionally equivalent if they retain at least 75% sequence identity to a reference panD sequence and produce β-alanine at a rate not less than 60% of the wild-type enzyme under identical assay conditions.

### Define β-alanine-pyruvate aminotransferase (BAPAT)

β-Alanine-pyruvate aminotransferase (BAPAT) is a pyridoxal phosphate-dependent transaminase that catalyzes the reversible transfer of an amino group from β-alanine to pyruvate, yielding malonate semialdehyde and alanine. This enzyme is a critical component of the 3-hydroxypropionate biosynthetic pathway, linking β-alanine metabolism to the formation of the direct precursor to 3-hydroxypropionate.

### Explain BAPAT enzyme function

BAPAT facilitates the transamination reaction between β-alanine and pyruvate, resulting in the formation of malonate semialdehyde and alanine. The enzyme requires pyridoxal phosphate as a cofactor, which forms a Schiff base with the amino group of β-alanine, enabling the transfer of the amino moiety to pyruvate. BAPAT exhibits broad substrate specificity but demonstrates highest catalytic efficiency with β-alanine and pyruvate. Its activity is optimal at pH 7.5–8.5 and temperatures between 30°C and 40°C, although it retains sufficient activity under the acidic conditions of Aspergillus fermentations when expressed at high levels.

### Provide BAPAT sequence examples

An exemplary BAPAT sequence is derived from Bacillus cereus (UniProt ID: Q818D1), comprising 412 amino acids. A codon-optimized version for expression in Aspergillus species is provided in SEQ ID NO: 3. Another functional variant is derived from Corynebacterium glutamicum (UniProt ID: Q9K8Q0), which shares 58% sequence identity with the B. cereus homolog and demonstrates comparable activity in fungal expression systems.

### Explain BAPAT variant sequences

Variant sequences of BAPAT include those with amino acid substitutions that improve enzyme kinetics, stability, or expression levels under acidic conditions. For example, substitution of glycine at position 187 with arginine enhances substrate binding affinity for pyruvate by 35%. Variant sequences are considered functionally equivalent if they retain at least 70% sequence identity to a reference BAPAT sequence and catalyze the formation of malonate semialdehyde at a rate not less than 65% of the wild-type enzyme under identical assay conditions.

### Define cadA (cis-aconitic acid decarboxylase)

CadA, or cis-aconitic acid decarboxylase, is a zinc-dependent enzyme that catalyzes the decarboxylation of cis-aconitic acid to produce itaconic acid. This enzyme is the defining component of the itaconic acid biosynthetic pathway in Aspergillus species and is responsible for diverting carbon flux away from the tricarboxylic acid cycle toward the production of itaconic acid.

### Explain cadA enzyme function

The cadA enzyme functions by binding cis-aconitic acid in its active site and facilitating the elimination of a carboxyl group as carbon dioxide, resulting in the formation of itaconic acid. The reaction proceeds through a zinc-stabilized enolate intermediate, and the enzyme exhibits high specificity for cis-aconitic acid over its isomers. CadA is expressed under conditions of high glucose availability and low pH, and its activity is tightly regulated at the transcriptional level by the itaconic acid gene cluster. Inactivation of cadA results in the accumulation of cis-aconitic acid and trans-aconitic acid, which are otherwise consumed in wild-type strains.

### Provide cadA sequence examples

An exemplary cadA sequence is derived from Aspergillus pseudoterreus (GenBank accession: MN891234), comprising 1,026 nucleotides encoding a 342-amino acid protein. A second functional variant is derived from Aspergillus terreus (GenBank accession: AF421567), which shares 94% nucleotide identity with the A. pseudoterreus sequence. The cadA gene is flanked by upstream and downstream regulatory regions that are essential for its expression, as detailed in SEQ ID NO: 4 and SEQ ID NO: 5.

### Explain cadA variant sequences

Variant sequences of cadA include those with deletions, insertions, or point mutations that abolish enzymatic activity while preserving the structural integrity of the surrounding genomic locus. For example, a 48-base pair deletion within the catalytic zinc-binding domain results in a frameshift and premature stop codon, leading to complete loss of function. Variant sequences are considered functionally inactivated if they reduce cadA activity to less than 5% of wild-type levels under standard fermentation conditions.

### Define detectable

Detectable refers to the ability to identify the presence or absence of a molecule, gene, protein, or metabolite using analytical methods with a signal-to-noise ratio sufficient to distinguish it from background. In this invention, detectable levels of aconitic acid, 3-hydroxypropionate, or heterologous proteins are defined as concentrations or expression levels that can be reliably quantified using high-performance liquid chromatography, mass spectrometry, or immunoblotting techniques.

### Define exogenous

Exogenous refers to any nucleic acid, protein, or genetic element that is introduced into a host organism from an external source, including but not limited to genes from bacterial, fungal, plant, or synthetic origins. In this invention, exogenous nucleic acid molecules include the panD, BAPAT, and HPDH genes, which are not native to Aspergillus pseudoterreus but are functionally expressed in the recombinant strains.

### Explain genetic enhancement or up-regulation

Genetic enhancement or up-regulation refers to the increase in the expression level of a gene or the activity of its encoded protein, achieved through methods such as promoter replacement, gene copy number amplification, or deletion of repressor elements. In this invention, up-regulation of panD, BAPAT, and HPDH is achieved by placing their coding sequences under the control of strong constitutive promoters such as gpdA and eno1, resulting in significantly higher transcript and protein levels compared to native expression.

### Explain genetic inactivation or down-regulation

Genetic inactivation or down-regulation refers to the reduction or complete elimination of the functional activity of a gene or its encoded protein, achieved through methods such as gene deletion, insertion mutagenesis, antisense RNA expression, RNA interference, or promoter disruption. In this invention, genetic inactivation of cadA is achieved by homologous recombination-mediated replacement of the cadA coding region with a selectable marker, resulting in the complete absence of itaconic acid production.

### Summarize gene expression

Gene expression in the recombinant strains of this invention is driven by constitutive promoters that remain active throughout the growth and production phases of fermentation. The expression of heterologous genes is stable over multiple generations, and transcript levels are quantified using quantitative RT-PCR. Protein levels are confirmed by targeted proteomics and immunoblotting, demonstrating consistent expression of panD, BAPAT, and HPDH in the absence of inducers.

### Define mutation

Mutation refers to any heritable change in the nucleotide sequence of a gene or genomic region. In this invention, mutations include deletions, insertions, substitutions, and inversions introduced to inactivate cadA or to enhance the expression of heterologous genes. Mutations are confirmed by DNA sequencing and are stably maintained in the absence of selective pressure.

### Describe genetic inactivation

Genetic inactivation of cadA is achieved by replacing the entire coding sequence with a selectable marker gene, such as ptrA or hph, via homologous recombination. Alternatively, a portion of the cadA coding region is disrupted by insertion of a transposon or by introduction of a stop codon via site-directed mutagenesis. Inactivation is confirmed by the absence of itaconic acid production and by the presence of cis-aconitic acid and trans-aconitic acid in the culture supernatant.

### Define isolated

Isolated refers to a nucleic acid, protein, or organism that has been purified or separated from its native environment to a degree sufficient for industrial or analytical use. In this invention, isolated recombinant Aspergillus strains are free from contamination by other microbial species and contain only the intended genetic modifications.

### Describe isolation methods

Isolation methods include serial dilution and plating on selective media, filtration, centrifugation, and lyophilization. For recombinant strains, isolation is performed by selecting colonies that grow on media containing antibiotics corresponding to the selectable marker used during transformation. Strains are further verified by PCR and metabolite profiling.

### Define promoter

A promoter is a DNA sequence located upstream of a gene that directs the initiation of transcription by RNA polymerase. In this invention, promoters are selected for their ability to drive high-level, constitutive expression in Aspergillus species under acidic fermentation conditions.

### Describe promoter elements

Promoter elements include core promoter regions such as TATA boxes, initiator elements, and upstream activating sequences that bind transcription factors. In this invention, the gpdA promoter from Aspergillus niger contains a conserved GC-rich region that enhances transcriptional activity under high-glucose conditions.

### Describe bi-directional promoters

Bi-directional promoters are regulatory sequences capable of initiating transcription in both directions, enabling the simultaneous expression of two adjacent genes. In this invention, a bi-directional terminator from the A. niger elf3 gene is used to drive expression of two heterologous genes in opposite orientations.

### Provide examples of promoters

Examples of promoters used in this invention include the gpdA promoter from Aspergillus niger, the eno1 promoter from Aspergillus niger, and the tdh promoter from Aspergillus nidulans. Each promoter drives high-level expression of heterologous genes under standard fermentation conditions.

### Define recombinant

Recombinant refers to an organism or nucleic acid molecule that has been artificially modified through the introduction or alteration of genetic material using molecular biological techniques. In this invention, recombinant Aspergillus strains contain exogenous genes and/or inactivated endogenous genes that alter their metabolic output.

### Describe recombinant nucleic acid molecules

Recombinant nucleic acid molecules in this invention include plasmids and expression cassettes containing the panD, BAPAT, and HPDH genes under the control of fungal promoters, as well as deletion constructs designed to replace the cadA gene with selectable markers.

### Describe recombinant organisms

Recombinant organisms in this invention are Aspergillus pseudoterreus strains that have been genetically modified to inactivate cadA and express heterologous genes for 3-hydroxypropionate biosynthesis. These strains are capable of producing aconitic acid and 3-hydroxypropionate at industrially relevant titers.

### Define sequence identity/similarity

Sequence identity refers to the percentage of identical residues between two aligned sequences, while sequence similarity includes both identical and functionally equivalent residues. In this invention, sequence identity is calculated using BLAST with default parameters, and sequences with at least 70% identity are considered homologous.

### Describe methods of sequence alignment

Sequence alignment is performed using the Needleman-Wunsch global alignment algorithm or the BLAST local alignment algorithm. Alignments are performed using the NCBI BLASTP or BLASTN tools with default gap penalties and substitution matrices.

### Describe BLAST

BLAST, or Basic Local Alignment Search Tool, is a bioinformatics algorithm used to compare nucleotide or protein sequences against databases to identify regions of similarity. In this invention, BLAST is used to identify homologs of panD, BAPAT, HPDH, and cadA across fungal and bacterial genomes.

### Provide examples of BLAST options

BLAST options used in this invention include the BLOSUM62 substitution matrix, an E-value threshold of 1e-10, and word size of 3 for protein searches. For nucleotide searches, the Megablast algorithm is used with a word size of 28.

### Describe calculation of sequence identity

Sequence identity is calculated as the number of identical residues divided by the total number of aligned residues, multiplied by 100. Gaps are not counted in the denominator. For example, if two sequences align over 100 residues and 85 are identical, the sequence identity is 85%.

### Describe rounding of sequence identity values

Sequence identity values are rounded to the nearest whole number. For example, a value of 84.7% is rounded to 85%, and 84.4% is rounded to 84%.

### Describe comparisons of amino acid sequences

Amino acid sequence comparisons are performed using Clustal Omega or MUSCLE alignment tools, followed by manual inspection of conserved domains. Functional equivalence is determined by conservation of catalytic residues and overall structural topology.

### Describe homologs

Homologs are sequences derived from a common ancestor and sharing significant sequence similarity and functional similarity. In this invention, panD, BAPAT, and HPDH homologs from bacterial and fungal sources are identified and tested for functionality in Aspergillus hosts.

### Describe degeneracy of genetic code

The degeneracy of the genetic code refers to the fact that multiple codons can encode the same amino acid. In this invention, codon optimization exploits this degeneracy to replace rare fungal codons with preferred codons, thereby enhancing translation efficiency.

### Describe homologous nucleic acid sequences

Homologous nucleic acid sequences are those that share significant sequence similarity and likely common evolutionary origin. In this invention, homologous sequences of cadA are identified across Aspergillus species and used to design deletion constructs.

### Describe variant proteins or nucleic acid molecules

Variant proteins or nucleic acid molecules are those that differ from a reference sequence by one or more substitutions, insertions, or deletions, yet retain the biological function of the original molecule. In this invention, variant sequences of panD, BAPAT, HPDH, and cadA are tested for functionality in recombinant strains.

### Define transformed

Transformed refers to a host cell that has stably incorporated exogenous nucleic acid into its genome or episomal elements such that the introduced genetic material is heritable and expressed. In this invention, transformed Aspergillus strains are selected based on antibiotic resistance and confirmed by PCR and metabolite profiling.

### Describe transformation methods

Transformation methods include protoplast transformation using polyethylene glycol, Agrobacterium-mediated transformation, and electroporation. In this invention, protoplast transformation is used to introduce deletion constructs and expression cassettes into Aspergillus pseudoterreus.

### Define vector

A vector is a nucleic acid construct capable of delivering and expressing exogenous genes in a host organism. In this invention, vectors include plasmids containing promoters, coding sequences, terminators, and selectable markers for use in Aspergillus transformation.

### Describe vector elements

Vector elements include a selectable marker (e.g., ptrA), a promoter (e.g., gpdA), a coding sequence (e.g., HPDH), a terminator (e.g., trpC), and a replication origin. In this invention, vectors are designed for integration into the cadA locus via homologous recombination.

### Overview

Aspergillus pseudoterreus is a filamentous fungus capable of producing high titers of organic acids under acidic fermentation conditions. The native production of itaconic acid is mediated by the cadA gene, which converts cis-aconitic acid to itaconic acid. Deletion of cadA results in the accumulation of cis-aconitic acid and trans-aconitic acid, which are otherwise not detectable in wild-type strains. Glucose is metabolized through glycolysis to produce citrate, which enters the tricarboxylic acid cycle and is converted to cis-aconitic acid. The mttA and mfsA transporters are responsible for the efflux of itaconic acid, but their expression is abolished in cadA deletion strains. The ΔcadA strain serves as a biocatalyst for the production of aconitic acid, a compound with higher value than itaconic acid in polymer synthesis. EST analysis confirms that the itaconic acid gene cluster is transcriptionally silent in the ΔcadA strain. The ΔcadA strain produces aconitic acid at yields exceeding 40 g/L, surpassing chemical synthesis methods. The strain is isolated and characterized as a novel recombinant Aspergillus fungus with genetic inactivation of cadA.

### Recombinant ΔcadA Fungi

Recombinant ΔcadA fungi are Aspergillus pseudoterreus strains in which the cadA gene has been genetically inactivated, resulting in the accumulation of cis-aconitic acid and trans-aconitic acid. Genetic inactivation is achieved through homologous recombination, deletion of the cadA coding region, or insertion of a stop codon. Aspergillus strains used include ATCC® 32359™ and its derivatives. Methods for genetic inactivation include protoplast transformation, Agrobacterium-mediated transfer, and CRISPR-Cas9 editing. Reduced cadA activity is defined as less than 5% of wild-type enzyme activity. Examples of cadA gene inactivation include deletion of exons 2–4, insertion of a ptrA cassette, and point mutation of the zinc-binding motif. Aconitic acid production is increased by at least 10-fold compared to wild-type strains. Additional genes inactivated include Apald6, which encodes malonate semialdehyde dehydrogenase, to prevent degradation of 3-hydroxypropionate. Aconitic acid transporters are not required for accumulation, as the acid diffuses passively under acidic conditions. Exogenous nucleic acid molecules encoding panD, BAPAT, and HPDH are introduced to enable 3-hydroxypropionate biosynthesis. Expression of these genes is driven by the gpdA and eno1 promoters. Production of 3-hydroxypropionate is achieved at titers exceeding 0.8 g/L. Methods of functionally deleting cadA include deletion of the coding region, mutation of promoter elements, and antisense RNA expression. An inactivated or functionally deleted cadA gene is defined as one that produces no detectable itaconic acid. Mutation of control elements includes deletion of the TATA box or upstream activating sequences. Deletion of the coding region removes the entire open reading frame. Insertional mutation introduces a selectable marker into the cadA locus. Genetic inactivation by vector transformation uses homologous recombination to replace cadA with a marker gene. The cre-lox system is used for site-specific recombination to excise marker genes after integration. Replacement of cadA with a marker gene is performed using a linear DNA fragment with 1 kb homology arms. Antisense technology involves expression of RNA complementary to cadA mRNA. Gene silencing is achieved using RNA interference constructs. Protoplast transformation involves enzymatic removal of the cell wall using lysing enzymes. Protoplast preparation includes incubation with Novozym 234 and osmotic stabilizers. Transformation of protoplasts is performed using polyethylene glycol and calcium chloride. Measuring gene inactivation is performed by PCR amplification of the cadA locus and sequencing. Nucleic acid hybridization techniques include Southern blotting with cadA-specific probes. qRT-PCR detects absence of cadA mRNA. Immunohistochemical and biochemical techniques confirm absence of cadA protein. Measuring aconitic acid production is performed using spectrophotometric assays and HPLC. LC and HPLC methods use C18 columns and UV detection at 210 nm. cadA sequences are provided in SEQ ID NO: 4. Variants of cadA sequences include those with conservative substitutions. Conservative amino acid substitutions preserve enzyme structure. The cadA gene encodes a 342-amino acid protein. The cadA protein catalyzes the decarboxylation of cis-aconitic acid. cadA sequences are identified by BLAST against fungal genomes. Methods for identifying cadA sequences include PCR with degenerate primers. panD, BAPAT, and HPDH sequences are provided in SEQ ID NO: 2, SEQ ID NO: 3, and SEQ ID NO: 1. panD, BAPAT, and HPDH proteins function in the 3-hydroxypropionate biosynthetic pathway. panD, BAPAT, and HPDH sequences are identified by homology searches. Methods for identifying panD, BAPAT, and HPDH sequences include BLASTP and HMMER. Variants of panD, BAPAT, and HPDH sequences include those with codon optimization. Codon optimization enhances expression in Aspergillus. Conservative amino acid substitutions are used to improve enzyme stability. panD, BAPAT, and HPDH gene expression is driven by constitutive promoters. Up-regulation of panD, BAPAT, and HPDH genes is achieved by promoter replacement. Transformation of Aspergillus is performed using protoplasts or Agrobacterium. Methods for increasing panD, BAPAT, and HPDH expression include gene copy number amplification. The cre-lox system enables marker excision after integration. Site-specific recombination is mediated by cre recombinase. Transgene generation involves assembly of promoter-coding sequence-terminator cassettes. Transgene expression is confirmed by RT-PCR and immunoblotting. The trpC transcriptional terminator ensures proper mRNA processing. The ptrA sequence confers resistance to pyrithiamine. Transgene sequences are integrated into the cadA locus. Measuring gene expression is performed by qRT-PCR and RNA-seq. Immunohistochemical techniques detect heterologous proteins in mycelia. Biochemical techniques include enzyme assays using NADPH consumption. Measuring 3-HP production is performed by HPLC and GC-MS. ΔcadA fungus identification is confirmed by PCR and metabolite profiling. panD, BAPAT, and HPDH peptide detection is performed by targeted proteomics. 3-HP production detection is confirmed by comparison to authentic standards. Genetic inactivation of cadA is confirmed by absence of itaconic acid. Combination of cadA inactivation and panD, BAPAT, and HPDH expression enables simultaneous production of aconitic acid and 3-hydroxypropionate.

### Methods of Producing Aconitic Acid

ΔcadA fungi are cultured in minimal medium containing glucose as the sole carbon source. Culturing conditions include a temperature of 30°C, agitation at 200 rpm, and pH maintained at 2.8–3.2. Culture media for aconitic acid production include 100 g/L glucose, 2.36 g/L (NH₄)₂SO₄, and trace minerals. Temperature and pressure conditions are ambient pressure at 30°C. Isolation and purification of aconitic acid involve filtration, acid precipitation, and crystallization. Timing for aconitic acid isolation is at 7–10 days post-inoculation. Culture containers include shake flasks and stirred-tank bioreactors. Inoculation is performed with 2×10⁶ spores per mL. Examples of aconitic acid production yields include 42 g/L in shake flasks and 48 g/L in bioreactors. Yields are 15-fold higher than in wild-type cadA fungi, which produce less than 3 g/L.

### Methods of Producing 3-HP

ΔcadA fungi expressing panD, BAPAT, and HPDH are cultured in minimal medium supplemented with ammonium sulfate and trace metals. Culturing conditions include a temperature of 30°C, agitation at 400 rpm, and pH maintained at 2.8. Culture media for 3-HP production include 100 g/L glucose, 2.36 g/L (NH₄)₂SO₄, and 0.13 g/L CaCl₂·2H₂O. Temperature and pressure conditions are ambient pressure at 30°C. Isolation and purification of 3-HP involve filtration, ion-exchange chromatography, and lyophilization. Timing for 3-HP isolation is at 7–9 days post-inoculation. Culture containers include 0.5 L and 20 L bioreactors. Inoculation is performed with 2×10⁶ spores per mL. Examples of 3-HP production yields include 0.88 g/L in the ΔApald6 strain and 0.27 g/L in the parent strain. Yields are 3.3-fold higher than in wild-type cadA fungi, which produce no detectable 3-HP.

### Compositions and Kits

Compositions comprise isolated recombinant Aspergillus fungi with genetic inactivation of cadA and expression of panD, BAPAT, and HPDH. Kits comprise vectors for cadA inactivation and heterologous gene expression, selectable markers, primers for strain verification, and culture media for fermentation. Exemplary mediums for compositions and kits include minimal medium with glucose and ammonium sulfate, agar plates with pyrithiamine, and cryopreservation solutions containing glycerol.

### Example 1

Materials and methods include strain ATCC® 32359™, vector pZD4028, and growth conditions in minimal medium at 30°C. Medium composition includes 100 g/L glucose, 2.36 g/L (NH₄)₂SO₄, and trace elements. Transformation of A. pseudoterreus protoplasts is performed using polyethylene glycol. Construction of deletion mutants is achieved by fusion PCR and homologous recombination. Fusion PCR products are assembled using Gibson Assembly. Overlap PCR is used to generate deletion cassettes. Transformation protocol includes protoplast preparation, incubation with DNA, and recovery on selective media. Protoplast preparation involves enzymatic digestion with Novozym 234. Washing and conditioning of protoplasts is performed in osmotic stabilizer. Transformation reaction is incubated at 30°C for 48 hours. Dry mass measurement is performed by lyophilization. HPLC analysis is performed using a Bio-Rad HPX-87H column. RNA isolation and transcript analysis are performed using TRIzol and qRT-PCR. Quantitative real-time RT-PCR is performed using SYBR Green and actin as a reference gene.

### Example 2

Expression profile of the itaconic acid gene cluster is analyzed using EST data from Aspergillus pseudoterreus. EST data analysis reveals coordinated expression of tf, cadA, mttA, and mfsA during peak itaconic acid production. Expression patterns show that cadA is the most highly expressed gene in the cluster, with transcript levels 12-fold higher than baseline.

### Example 3

Transformation system development includes protoplast transformation and Agrobacterium-mediated delivery. Generation of recombinant knockout strains is performed using homologous recombination. Effect of tf, cadA, mttA, and mfsA deletion on itaconic acid production shows that deletion of cadA abolishes itaconic acid production, while deletion of tf reduces production by 80%.

### Example 4

Production kinetics are tested in shake flasks and bioreactors. IA yield is analyzed by HPLC. IA yield in Δtf strain is 20% of wild type. Δtf strain characteristics include reduced growth rate and altered morphology. Example 5 investigates tf gene deletion effects. mRNA levels are measured by qRT-PCR. tf gene deletion consequences include downregulation of cadA and mttA. Example 6 deletes cadA gene. Aconitic acid production is analyzed by LC-MS. ΔcadA strain characteristics include accumulation of cis-aconitic acid and trans-aconitic acid. Example 7 describes materials and methods for cloning. DNA fragments are isolated by PCR. Fragments are assembled into plasmid DNA using Gibson Assembly. Invention scope is claimed for recombinant Aspergillus strains with inactivated cadA and expressed panD, BAPAT, and HPDH.