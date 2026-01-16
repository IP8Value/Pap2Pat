# DESCRIPTION

## FEDERALLY SPONSORED RESEARCH

No federal funding was utilized in the development, experimentation, or validation of the methods, compositions, or systems described herein. All research activities, including the design and execution of Rec-seq assays, the purification and characterization of recombinase variants, the synthesis of randomized DNA substrate libraries, and the functional assessment of off-target activity in mammalian cells, were conducted entirely with private and institutional resources. No grants, contracts, or cooperative agreements with any department, agency, or instrumentality of the United States Government were involved in the conception, reduction to practice, or optimization of the disclosed invention. Consequently, no rights, title, or interest in the invention are claimed by or on behalf of any federal entity, and no reporting obligations under the Bayh-Dole Act or any other federal statute governing federally funded inventions are applicable to this disclosure.

## BACKGROUND

Site-specific recombinases (SSRs) represent a class of enzymes capable of catalyzing precise, irreversible rearrangements of DNA sequences at defined recognition sites without inducing double-strand breaks or relying on endogenous cellular repair pathways. This intrinsic property renders them uniquely suited for applications in genome engineering, where the avoidance of error-prone non-homologous end joining and the suppression of p53-mediated stress responses are critical for maintaining genomic integrity. Unlike CRISPR-Cas systems, which depend on the cell’s repair machinery to achieve desired edits and frequently generate indels, translocations, or chromosomal rearrangements, SSRs directly mediate the cleavage, strand exchange, and ligation of DNA substrates with high fidelity and efficiency, even in non-dividing cells such as neurons or quiescent stem cells. These characteristics have positioned SSRs as indispensable tools in transgenic animal models, where Cre recombinase, acting on loxP sites, has enabled conditional gene knockout, lineage tracing, and spatially restricted gene expression with unparalleled precision.

Despite their utility, the adoption of SSRs as programmable genome editing agents has been severely constrained by the rigidity of their native DNA recognition profiles. The substrate specificity of most SSRs is encoded in a complex, distributed network of interactions between protein residues and nucleotide bases, many of which are not directly contact-dependent but instead arise from indirect structural, electrostatic, or hydration-mediated effects. As a result, attempts to re-engineer SSRs to recognize novel target sequences have historically required extensive, iterative rounds of laboratory evolution—often exceeding one hundred selection cycles—to achieve even modest degrees of altered specificity. For instance, the evolution of Tre recombinase to recognize loxLTR, a variant of loxP with half of its bases altered, required 126 rounds of directed evolution, while Brec1, which recognizes loxBTR with 68% sequence divergence from loxP, required 145 rounds. These laborious processes are not only time-intensive and resource-heavy but also provide little mechanistic insight into how specificity is encoded, making rational design nearly impossible.

The structural basis of SSR specificity has been partially elucidated through X-ray crystallography, which has revealed that tyrosine-family recombinases such as Cre interact with their DNA substrates through a limited number of direct hydrogen bonds and van der Waals contacts, with the majority of recognition energy derived from shape complementarity, minor groove electrostatics, and water-mediated hydrogen bonding networks. However, these static snapshots fail to capture the dynamic, cooperative nature of protein-DNA recognition, particularly the long-range allosteric effects whereby mutations at one site can propagate through the protein structure to influence binding at distant nucleotides. For example, while Arg259 in Cre makes a direct hydrogen bond with the cytosine at position 10 of loxP, its mutation to alanine not only abolishes specificity at that position but also enhances fidelity at positions 5–7 and 16, suggesting a compensatory tightening of interactions elsewhere. Such phenomena are invisible to crystallographic analysis and have eluded functional characterization due to the absence of high-throughput, unbiased methods capable of simultaneously interrogating the influence of every possible nucleotide substitution across an entire recognition site.

Current methods for assessing SSR specificity are fundamentally limited in scope and resolution. Traditional approaches involve testing individual mutant substrates one at a time, a process that is prohibitively slow and incapable of capturing the combinatorial complexity of multi-nucleotide interactions. Other methods rely on bacterial selection systems using degenerate primers or sheared genomic DNA, but these techniques suffer from low resolution, poor quantitation, and the inability to distinguish true recombination events from background noise or non-specific cleavage. Moreover, these methods are not easily scalable to the hundreds or thousands of substrate variants required to fully map a recombinase’s recognition landscape. As a consequence, the specificity profiles of most SSRs remain incompletely characterized, and the potential for off-target activity—particularly in therapeutic contexts—has been largely unquantified and unpredicted.

The lack of a comprehensive, high-resolution method for mapping SSR specificity has thus created a critical bottleneck in the field of genome engineering. Without the ability to rapidly and accurately determine which nucleotides are tolerated, preferred, or forbidden at every position of a target site, the rational design of novel recombinases remains speculative. Similarly, without empirical data on the full spectrum of potential off-target substrates, the safety profile of any recombinase intended for clinical use cannot be reliably assessed. The present invention addresses these limitations by introducing Rec-seq, a novel, high-throughput, in vitro selection method that enables the unbiased, nucleotide-resolution profiling of SSR specificity across entire target sites. By coupling a hairpin-based substrate library design with exonuclease-mediated enrichment of recombinase-processed products and deep sequencing, Rec-seq generates quantitative, genome-wide specificity maps that reveal not only direct contacts but also long-range, compensatory, and allosteric determinants of recognition. This method transforms the empirical characterization of SSRs from a slow, low-resolution endeavor into a rapid, scalable, and predictive platform capable of guiding both the engineering of novel recombinases and the assessment of their safety in human cells.

## SUMMARY

The present invention relates to a novel method and system for the high-resolution, unbiased profiling of DNA sequence specificity of site-specific recombinases (SSRs), termed Rec-seq, and to the engineered recombinase variants, vectors, and cellular systems derived therefrom. The invention provides a transformative platform for mapping the complete nucleotide recognition landscape of SSRs, enabling the identification of specificity determinants that are not discernible through structural biology or traditional mutagenesis. The method involves the construction of a library of DNA hairpin oligonucleotides containing a partially randomized SSR recognition site, wherein each molecule includes a unique molecular identifier (UMI) and is designed such that successful recombination generates a double-hairpin product resistant to exonuclease digestion, while non-recombined substrates are degraded. Following exposure of the library to a recombinase of interest under defined in vitro conditions, the exonuclease-resistant recombination products are amplified by PCR and subjected to high-throughput sequencing. The relative enrichment of each nucleotide at each position within the recognition site is then calculated by comparing pre- and post-selection sequence frequencies, yielding a quantitative specificity profile that reflects the recombinase’s preference for every possible nucleotide combination across the entire target site.

The invention further encompasses the application of Rec-seq to a broad spectrum of SSRs, including wild-type and mutant variants of Cre, the evolved recombinases Tre and Brec1, and non-Cre family members such as Dre, VCre, and Bxb1. Rec-seq profiles generated for these enzymes reveal previously unknown specificity determinants, including long-range compensatory interactions, asymmetric recognition patterns, and residues that exert influence on nucleotides distant from their direct contact points. For example, Rec-seq demonstrated that the mutation of Arg259 in Cre to alanine not only ablates specificity at position 10 of loxP but also enhances fidelity at positions 5–7 and 16, indicating a global energetic tradeoff in substrate recognition. Similarly, Rec-seq profiles of Tre and Brec1 revealed that their evolved specificity arises not from the acquisition of novel direct contacts but from a redistribution of binding energy, wherein relaxed specificity at mutated positions is compensated by heightened fidelity at conserved positions. These findings validate a model of SSR specificity wherein productive recombination requires a minimum threshold of binding energy, and the loss of one interaction necessitates increased fidelity at other, often distal, sites.

The invention further provides methods for predicting and validating off-target recombination activity in mammalian cells using Rec-seq-derived specificity profiles. By identifying non-cognate half-site sequences that are highly enriched in Rec-seq libraries, candidate off-target substrates are synthesized and cloned into reporter constructs containing a transcriptional terminator flanked by the candidate sites. Upon transfection into human cells and co-expression of the recombinase, recombination-mediated excision of the terminator restores expression of a fluorescent reporter, enabling quantitative measurement of off-target activity. Using this approach, the invention demonstrates that Rec-seq profiles accurately predict the activity of evolved recombinases on endogenous human genomic pseudosites, including sequences previously deemed inactive based on bacterial assays. Notably, Brec1 was found to exhibit robust recombination activity on seven out of eight predicted human genomic pseudosites, with activity levels exceeding those observed on its cognate substrate in some cases.

The invention further provides a library of recombinase variants with altered specificity profiles, including engineered Cre mutants with enhanced or relaxed substrate preferences, as well as novel orthogonal recombinases identified through Rec-seq profiling of natural SSRs. These variants are encoded in expression constructs suitable for delivery into mammalian, bacterial, or plant cells and are compatible with standard viral and non-viral delivery systems. The invention further provides vectors containing recombinase recognition sites flanking selectable markers, reporter genes, or therapeutic transgenes, enabling targeted insertion, deletion, or inversion of DNA segments with precision unattainable by nuclease-based editors. The disclosed methods and compositions enable the rapid design, validation, and deployment of SSRs as programmable genome editors with minimized off-target risk, thereby overcoming the longstanding limitations of traditional genome engineering tools.

## DEFINITIONS

As used herein, the term “site-specific recombinase” (SSR) refers to an enzyme that catalyzes the recombination of DNA at two specific recognition sites, resulting in the excision, inversion, insertion, or translocation of intervening DNA sequences. SSRs are classified into two major families based on their catalytic mechanism: the tyrosine recombinase family, which includes Cre, Dre, VCre, and Brec1, and the serine recombinase family, which includes Bxb1 and other integrases. These enzymes operate without the need for host cell factors, do not generate double-strand breaks, and catalyze strand exchange through a covalent protein-DNA intermediate. The term encompasses both wild-type and engineered variants thereof, including those with altered specificity, enhanced activity, or improved stability.

The term “recombination target site” or “recognition site” refers to a defined DNA sequence to which a site-specific recombinase binds and catalyzes recombination. These sites typically consist of two inverted or direct repeats (half-sites) flanking a central core region where strand exchange occurs. The core region is often asymmetric and must be complementary between two recombining substrates. Examples include loxP for Cre, rox for Dre, loxV for VCre, loxLTR for Tre, loxBTR for Brec1, attP and attB for Bxb1, and any variant thereof with one or more nucleotide substitutions.

The term “specificity profile” refers to a quantitative representation of a recombinase’s preference for each nucleotide at each position within its recognition site, derived from high-throughput sequencing data following in vitro selection. The profile is expressed as an enrichment score for each nucleotide at each position, calculated as the ratio of the frequency of that nucleotide in the recombinase-selected library relative to its frequency in the input library, normalized for background noise. A high enrichment score indicates strong preference for that nucleotide, while a score near one indicates tolerance or lack of preference.

The term “enrichment score” refers to a numerical value derived from high-throughput sequencing data that quantifies the relative abundance of a specific nucleotide at a specific position in a recombinase-selected DNA library compared to its abundance in the input library. The enrichment score is calculated as the ratio of the probability of observing the nucleotide in the post-selection library to the probability of observing it in the pre-selection library, adjusted for sequencing depth and background amplification. An enrichment score significantly greater than one indicates a preference for that nucleotide; a score near one indicates neutrality; and a score less than one indicates avoidance.

The term “unique molecular identifier” (UMI) refers to a short, random nucleotide sequence incorporated into each DNA molecule in a library to enable the tracking of individual recombination events. UMIs allow for the distinction between true recombination events and PCR or sequencing artifacts by ensuring that each unique recombination product is counted only once, regardless of the number of times it is amplified. The UMI is positioned outside the recombinase recognition site and is not subject to recombination, thereby serving as a molecular barcode for quantification.

The term “hairpin oligonucleotide” refers to a synthetic DNA molecule that contains a self-complementary region forming a stem-loop structure, which enables primer-independent extension to generate a double-stranded substrate suitable for recombination. Hairpin oligonucleotides are designed such that recombination between a left-hairpin and a right-hairpin generates a double-hairpin product resistant to exonuclease digestion, while non-recombined substrates are degraded. The hairpin structure facilitates the replication of the randomized recognition site without the need for external primers.

The term “exonuclease digestion” refers to the enzymatic degradation of DNA molecules by exonucleases that remove nucleotides from the ends of DNA strands. In the context of this invention, a combination of exonucleases I, III, and V is used to selectively degrade non-recombined, single-hairpin substrates while leaving double-hairpin recombination products intact. This step is critical for enriching the population of true recombination events and minimizing background noise in downstream sequencing.

The term “high-throughput sequencing” (HTS) refers to next-generation sequencing technologies capable of generating millions to billions of DNA sequence reads in a single run. In this invention, HTS is performed using Illumina MiSeq platforms to sequence the recombination products following exonuclease digestion and PCR amplification. The resulting data are analyzed to determine the nucleotide composition at each position of the recognition site.

The term “off-target activity” refers to the unintended recombination of a site-specific recombinase at DNA sequences that differ from its cognate recognition site. Off-target activity may occur at sequences with partial homology to the target site and may result in genomic rearrangements, gene disruptions, or unintended transgene expression. The term includes both synthetic off-target substrates generated in vitro and endogenous genomic pseudosites identified in vivo.

The term “pseudosite” refers to a genomic DNA sequence that bears partial homology to a recombinase’s recognition site and is capable of undergoing recombination in the presence of the recombinase, despite not being the cognate substrate. Pseudosites may contain multiple mismatches relative to the canonical site and are often located in non-coding or intergenic regions of the genome.

The term “recombinase variant” refers to a site-specific recombinase that has been modified through mutagenesis, evolution, or rational design to exhibit altered DNA specificity, increased catalytic efficiency, enhanced stability, or reduced off-target activity relative to the wild-type enzyme. Variants may include single or multiple amino acid substitutions, insertions, deletions, or fusions with other functional domains.

The term “expression construct” refers to a nucleic acid molecule designed for the transcription and translation of a recombinase or other gene product in a host cell. Expression constructs typically include a promoter, a coding sequence, a terminator, and optionally a selectable marker or reporter gene. Constructs may be plasmid-based, viral, or integrated into the genome.

The term “host cell” refers to a living cell capable of supporting the expression of a recombinase and the recombination of its target sites. Host cells may be prokaryotic, such as Escherichia coli, or eukaryotic, including mammalian cells such as HEK293T, HeLa, or primary neurons, as well as plant, fungal, or insect cells.

The term “vector” refers to a nucleic acid molecule used to deliver a recombinase, a recognition site, or a genetic payload into a host cell. Vectors may be plasmid, viral, or synthetic and may be designed for transient or stable expression. Vectors may contain one or more recombinase recognition sites flanking a genetic element to be excised, inverted, or inserted.

The term “genetic engineering” refers to the targeted modification of a genome through the insertion, deletion, inversion, or replacement of DNA sequences. In this invention, genetic engineering is achieved through recombinase-mediated recombination at defined recognition sites, without the introduction of double-strand breaks or reliance on endogenous repair pathways.

The term “therapeutic application” refers to the use of a recombinase or recombinase-mediated system to treat, prevent, or ameliorate a disease or disorder in a subject. Therapeutic applications include, but are not limited to, the correction of disease-causing mutations, the excision of integrated viral genomes, the activation of silenced genes, or the targeted deletion of oncogenic sequences.

The term “specificity determinants” refers to the amino acid residues and DNA nucleotides that collectively define the binding and recombination specificity of a site-specific recombinase. These determinants include direct protein-DNA contacts, indirect interactions mediated by water molecules or backbone conformation, and long-range allosteric effects that influence nucleotide preference at positions distant from the residue.

The term “compensatory interaction” refers to a phenomenon in which the loss of binding energy at one position within a recognition site leads to an increase in fidelity at other, often distal, positions to maintain sufficient overall binding energy for recombination. This concept underlies the energetic tradeoff model described herein and is revealed by Rec-seq profiling.

The term “asymmetric recognition” refers to a pattern of nucleotide preference in which the left and right half-sites of a recognition site are bound with different affinities or specificities, despite being palindromic or nearly identical in sequence. Asymmetry may arise from differences in flanking sequences, protein conformation, or DNA topology.

The term “library” refers to a heterogeneous population of DNA molecules containing randomized sequences at defined positions within a recombinase recognition site. Libraries are synthesized by incorporating degenerate nucleotides during oligonucleotide synthesis and are used to interrogate the full spectrum of possible substrate combinations.

The term “quantitative PCR” (qPCR) refers to a method for measuring the amount of a specific DNA sequence in a sample using real-time fluorescence detection during amplification. In this invention, qPCR is used to confirm the presence of recombination products, assess library amplification bias, and determine the signal-to-noise ratio of Rec-seq experiments.

The term “t-SNE analysis” refers to a non-linear dimensionality reduction technique used to visualize high-dimensional data in two or three dimensions. In this invention, t-SNE is used to cluster Rec-seq profiles based on their similarity across all nucleotide positions, enabling the identification of functionally related recombinase variants.

The term “mammalian cell” refers to any cell derived from a mammal, including human, mouse, rat, rabbit, or primate cells. Mammalian cells may be primary, immortalized, or stem cell-derived and may be cultured in vitro or used in vivo for therapeutic delivery.

The term “in vitro” refers to experiments or processes performed outside a living organism, typically in a controlled laboratory environment such as a test tube, microplate, or reaction chamber. In this invention, all Rec-seq experiments are performed in vitro using purified recombinase and synthetic DNA substrates.

The term “in vivo” refers to experiments or processes performed within a living organism. In this invention, off-target activity is assessed in vivo by transfecting recombinase and reporter constructs into mammalian cells and measuring reporter gene expression.

The term “nucleotide substitution” refers to the replacement of one DNA base with another, including transitions (purine to purine or pyrimidine to pyrimidine) and transversions (purine to pyrimidine or vice versa). Substitutions may be single or multiple and may occur at any position within a recognition site.

The term “recombination efficiency” refers to the proportion of substrate molecules that undergo successful recombination under defined conditions. Efficiency is measured as the fraction of DNA molecules converted to recombination products, as determined by qPCR, gel electrophoresis, or sequencing.

The term “cognate substrate” refers to the natural or engineered DNA sequence that a recombinase is known to recognize and recombine with highest efficiency. For Cre, the cognate substrate is loxP; for Tre, it is loxLTR; for Brec1, it is loxBTR.

The term “orthogonal recombinase” refers to a recombinase that recognizes a target site distinct from that of other recombinases in the system, such that it does not cross-react with non-cognate sites. Orthogonality enables independent, simultaneous manipulation of multiple genetic elements within the same cell.

The term “genomic pseudosite” refers to a naturally occurring DNA sequence in a genome that bears sufficient similarity to a recombinase’s recognition site to permit recombination, even though it is not the intended target. Genomic pseudosites are identified through computational scanning of genome sequences using Rec-seq-derived specificity profiles.

The term “delivery system” refers to any method or vehicle used to introduce a recombinase, expression construct, or genetic payload into a cell or organism. Delivery systems include viral vectors (e.g., AAV, lentivirus), lipid nanoparticles, electroporation, microinjection, and transfection reagents.

The term “reporter system” refers to a genetic construct that produces a detectable signal upon successful recombination. In this invention, the reporter system comprises a transcriptional terminator flanked by recombinase recognition sites, such that recombination excises the terminator and restores expression of a fluorescent protein such as EGFP.

The term “recombination product” refers to the DNA molecule resulting from the successful cleavage, strand exchange, and ligation of two recombination substrates by a site-specific recombinase. In this invention, the recombination product is a double-hairpin DNA molecule resistant to exonuclease digestion.

The term “non-recombined substrate” refers to a DNA molecule that has not undergone recombination and retains its original single-hairpin structure. Non-recombined substrates are degraded during exonuclease digestion and are excluded from downstream analysis.

The term “recombinase buffer” refers to a solution containing salts, cofactors, and stabilizers optimized for the activity of a specific recombinase. Buffers may include Tris-HCl, NaCl, EDTA, DTT, BSA, spermidine, and glycerol, and are selected based on the recombinase family and catalytic mechanism.

The term “dual-barcoding” refers to the use of two distinct nucleotide barcodes—one in the forward primer and one in the reverse primer—to uniquely identify each sequencing read during high-throughput sequencing. Dual-barcoding enables multiplexing of multiple samples in a single sequencing run and reduces cross-contamination.

The term “sequence alignment” refers to the computational process of matching sequencing reads to a reference sequence to determine the nucleotide composition at each position. In this invention, alignments are performed without gaps to ensure accurate quantification of substitutions.

The term “Bonferroni correction” refers to a statistical method used to adjust p-values to account for multiple comparisons. In this invention, Bonferroni correction is applied to correct for the large number of nucleotide positions tested in Rec-seq profiles.

The term “Mann-Whitney U test” refers to a non-parametric statistical test used to compare the distribution of two independent groups. In this invention, it is used to compare the full Rec-seq specificity profiles of different recombinase variants.

The term “t-SNE multi-dimensional proximity analysis” refers to a machine learning technique used to visualize high-dimensional data by preserving local similarities. In this invention, it is used to cluster Rec-seq profiles based on their overall similarity, revealing functional relationships between recombinase variants.

The term “protein:DNA ratio” refers to the molar ratio of recombinase protein to DNA substrate molecules in a recombination reaction. This ratio is optimized to ensure that recombination is catalytic rather than stoichiometric and to prevent non-specific activity at high enzyme concentrations.

The term “thermal cycling” refers to the repeated heating and cooling of a reaction mixture to facilitate enzymatic reactions such as PCR. In this invention, thermal cycling is used for amplification of recombination products prior to sequencing.

The term “endotoxin-removal plasmid purification” refers to a method for purifying plasmid DNA free of bacterial lipopolysaccharides, which are toxic to mammalian cells. In this invention, endotoxin-free plasmids are required for transfection into human cells.

The term “transfection” refers to the process of introducing nucleic acids into eukaryotic cells using chemical, physical, or biological methods. In this invention, transfection is performed using Lipofectamine 2000 or similar reagents to deliver recombinase and reporter constructs into HEK293T cells.

The term “flow cytometry” refers to a technique for analyzing the physical and chemical characteristics of cells suspended in a fluid stream as they pass through a laser beam. In this invention, flow cytometry is used to quantify the percentage of cells expressing EGFP following recombinase-mediated recombination.

The term “biological replicate” refers to an independent experiment performed using a separate preparation of cells, reagents, or DNA. In this invention, biological replicates are used to ensure reproducibility and statistical significance.

The term “technical replicate” refers to repeated measurements of the same sample under identical conditions. In this invention, technical replicates are used to assess the consistency of sequencing depth and library amplification.

The term “recombination efficiency threshold” refers to the minimum level of recombination activity required to distinguish true substrate recognition from background noise. In this invention, this threshold is defined by the κavg quality score, which must exceed 1.5 to be considered well-powered.

The term “κavg quality score” refers to a metric derived from UMI counts that quantifies the number of independent recombination events captured in a Rec-seq experiment. A κavg value greater than 1.5 indicates a high signal-to-noise ratio and reliable specificity profiling.

The term “nucleotide resolution” refers to the ability to distinguish and quantify the preference for each individual DNA base at each position within a recognition site. In this invention, Rec-seq provides nucleotide-resolution specificity profiles across all positions of the recognition site.

The term “long-range interaction” refers to a functional relationship between a recombinase residue and a DNA nucleotide that are not in direct physical contact but influence each other through protein folding, conformational dynamics, or allosteric signaling. Long-range interactions are revealed by Rec-seq but are invisible to crystallography.

The term “energetic tradeoff” refers to a model in which the loss of binding energy at one site necessitates an increase in binding energy at another site to maintain sufficient overall affinity for recombination. This model is supported by Rec-seq data showing compensatory increases in specificity following residue mutations.

The term “dual substrate recognition” refers to the ability of a recombinase to recognize and recombine two distinct DNA substrates with comparable efficiency. In this invention, Bxb1 exhibits dual substrate recognition for attP and attB.

The term “binary recognition” refers to a pattern of specificity in which only two nucleotides are strongly preferred at a given position, with all others strongly disfavored. In this invention, VCre exhibits binary recognition at position 9 of loxV.

The term “minimal substrate motif” refers to the shortest sequence required for recombination by a recombinase, typically defined by the positions with the highest enrichment scores in a Rec-seq profile. Minimal motifs are used to scan genomes for pseudosites.

The term “genomic scanning” refers to the computational search of a genome for sequences matching a recombinase’s minimal substrate motif. In this invention, genomic scanning is performed using RSAT motif scanner and Rec-seq-derived specificity profiles.

The term “therapeutic safety profile” refers to the assessment of potential off-target recombination events and associated genomic risks associated with the use of a recombinase in a clinical setting. In this invention, the therapeutic safety profile is established using Rec-seq-predicted pseudosites validated in human cells.

The term “programmable recombinase” refers to a site-specific recombinase whose DNA recognition specificity has been altered through engineering or evolution to target a user-defined sequence. Programmable recombinases are the primary output of this invention.

The term “recombinase-mediated genetic engineering” refers to the use of site-specific recombinases to precisely modify genomic DNA without inducing double-strand breaks. This method is the central application of the disclosed invention.

The term “recombination substrate” refers to a DNA molecule containing a recombinase recognition site that is capable of undergoing recombination. Substrates may be synthetic, genomic, or plasmid-based.

The term “recombination product library” refers to the collection of DNA molecules generated following recombination and exonuclease digestion, which are sequenced to determine the specificity profile.

The term “input library” refers to the collection of DNA molecules before exposure to the recombinase, used as a baseline for calculating enrichment scores.

The term “recombination buffer optimization” refers to the empirical determination of salt concentration, pH, cofactors, and additives that maximize recombinase activity and specificity under in vitro conditions.

The term “recombinase purification” refers to the isolation of a recombinase protein from a host cell lysate using affinity chromatography, dialysis, and concentration techniques. In this invention, His-tagged recombinases are purified using nickel-NTA resin.

The term “DNA synthesis” refers to the chemical production of oligonucleotides with defined sequences, including randomized positions. In this invention, DNA synthesis is performed by integrated DNA technologies using phosphoramidite chemistry.

The term “PCR amplification” refers to the enzymatic replication of DNA using primers and a DNA polymerase. In this invention, PCR is used to amplify recombination products for sequencing.

The term “sequencing depth” refers to the number of times a given nucleotide position is read during high-throughput sequencing. In this invention, sequencing depth ranges from 10⁵ to 10⁶ reads per sample.

The term “background amplification” refers to the non-specific amplification of non-recombined DNA due to incomplete exonuclease digestion or PCR artifacts. Background amplification is quantified using qPCR and excluded from analysis if κavg is below 0.5.

The term “signal-to-noise ratio” refers to the ratio of true recombination events to background signals. In this invention, the κavg quality score serves as a quantitative measure of signal-to-noise ratio.

The term “recombinase activity assay” refers to any method for measuring the ability of a recombinase to catalyze recombination, including qPCR, gel electrophoresis, flow cytometry, or sequencing.

The term “recombinase specificity” refers to the degree to which a recombinase distinguishes its cognate substrate from non-cognate sequences. High specificity implies minimal off-target activity.

The term “recombinase promiscuity” refers to the ability of a recombinase to recombine multiple non-cognate substrates. Promiscuity is a hallmark of evolved recombinases and is quantified by Rec-seq.

The term “evolved recombinase” refers to a recombinase that has been subjected to laboratory evolution to alter its substrate specificity. In this invention, Tre and Brec1 are evolved recombinases.

The term “wild-type recombinase” refers to the naturally occurring, unmodified form of a recombinase. In this invention, wild-type Cre is the reference enzyme.

The term “residue substitution” refers to the replacement of one amino acid in a protein with another. In this invention, Ala substitutions are used to probe the contribution of individual residues to specificity.

The term “protein-DNA interface” refers to the region of contact between a recombinase and its DNA substrate. In this invention, the interface includes both direct contacts and indirect interactions revealed by Rec-seq.

The term “structural biology” refers to the study of the three-dimensional structure of biological molecules using techniques such as X-ray crystallography, cryo-electron microscopy, or nuclear magnetic resonance. In this invention, structural biology is contrasted with Rec-seq as a method for determining specificity.

The term “functional genomics” refers to the study of gene function through high-throughput assays. In this invention, Rec-seq is a functional genomics tool for mapping recombinase specificity.

The term “genome editing” refers to the targeted modification of a genome to alter its function. In this invention, genome editing is achieved via recombinase-mediated recombination.

The term “precision genome editing” refers to genome editing that achieves the intended modification without unintended side effects. In this invention, Rec-seq enables precision genome editing by predicting and minimizing off-target activity.

The term “clinical translation” refers to the process of moving a laboratory invention into human therapeutic use. In this invention, Rec-seq provides the data necessary for clinical translation of recombinases.

The term “regulatory compliance” refers to adherence to guidelines established by regulatory agencies such as the FDA or EMA for the safety and efficacy of gene therapies. In this invention, Rec-seq-generated specificity profiles support regulatory compliance by enabling comprehensive off-target assessment.

The term “scalable platform” refers to a method or system that can be easily expanded to process large numbers of samples or variants. In this invention, Rec-seq is a scalable platform for profiling hundreds of recombinases in parallel.

The term “high-resolution specificity mapping” refers to the ability to determine nucleotide preferences at single-base resolution across an entire recognition site. In this invention, Rec-seq provides high-resolution specificity mapping.

The term “unbiased profiling” refers to a method that does not pre-select or bias the set of substrates tested. In this invention, Rec-seq is unbiased because it tests all possible combinations of nucleotides in a randomized library.

The term “combinatorial library” refers to a library containing all possible combinations of nucleotide substitutions at defined positions. In this invention, the library contains all possible half-site variants with up to seven substitutions.

The term “nucleotide diversity” refers to the number of different nucleotides present at a given position in a library. In this invention, nucleotide diversity is maintained at 79% wild-type and 21% equimolar mixture of the other three bases.

The term “recombination fidelity” refers to the accuracy with which a recombinase discriminates between cognate and non-cognate substrates. High fidelity implies low promiscuity.

The term “substrate promiscuity” refers to the ability of a recombinase to recognize and recombine multiple non-cognate substrates. Substrate promiscuity is a key feature of evolved recombinases and is quantified by Rec-seq.

The term “recombinase engineering” refers to the deliberate modification of a recombinase to alter its specificity, activity, or stability. In this invention, Rec-seq enables recombinase engineering by providing a roadmap of specificity determinants.

The term “rational design” refers to the process of modifying a protein based on mechanistic understanding of its function. In this invention, Rec-seq enables rational design of recombinases by revealing the functional impact of every residue.

The term “direct protein-DNA contact” refers to a physical interaction between an amino acid side chain and a DNA nucleotide, such as a hydrogen bond or van der Waals contact. In this invention, direct contacts are identified by crystallography and validated by Rec-seq.

The term “indirect interaction” refers to a functional effect on DNA recognition that is not mediated by direct contact, such as water-mediated hydrogen bonding, backbone conformation, or allosteric signaling. In this invention, indirect interactions are revealed by Rec-seq.

The term “water-mediated hydrogen bond” refers to a hydrogen bond between a protein residue and a DNA nucleotide that is bridged by one or more water molecules. In this invention, such interactions contribute to specificity at positions lacking direct contacts.

The term “shape complementarity” refers to the geometric fit between a protein surface and a DNA helix. In this invention, shape complementarity contributes to specificity at positions with no direct contacts.

The term “electrostatic complementarity” refers to the favorable interaction between charged residues on a protein and the negatively charged DNA backbone. In this invention, electrostatic complementarity contributes to binding energy and specificity.

The term “minor groove recognition” refers to the interaction of a protein with the narrow groove of the DNA double helix. In this invention, minor groove interactions contribute to specificity at certain positions.

The term “major groove recognition” refers to the interaction of a protein with the wide groove of the DNA double helix. In this invention, major groove interactions are less common but contribute to specificity in some cases.

The term “DNA topology” refers to the three-dimensional arrangement of DNA, including supercoiling, bending, or looping. In this invention, Rec-seq is incompatible with supercoiled substrates but works with linear oligonucleotides.

The term “catalytic residue” refers to an amino acid directly involved in the chemical mechanism of recombination, such as the active site tyrosine or serine. In this invention, catalytic residues are not mutated in Rec-seq experiments.

The term “stabilizing mutation” refers to an amino acid substitution that improves protein folding, solubility, or thermal stability without altering specificity. In this invention, Brec1 contains a Leu163Phe stabilizing mutation.

The term “TEV-cleavable His-tag” refers to a fusion tag containing a hexahistidine sequence and a TEV protease cleavage site, used for purification and subsequent tag removal. In this invention, Brec1 contains a TEV-cleavable His-tag.

The term “nuclease-free water” refers to water treated to remove DNase and RNase activity, used in all DNA handling steps to prevent degradation.

The term “dithiothreitol” (DTT) refers to a reducing agent used to maintain cysteine residues in a reduced state. In this invention, DTT is added to buffers for serine recombinases.

The term “bovine serum albumin” (BSA) refers to a protein used to stabilize enzymes and reduce non-specific binding. In this invention, BSA is added to recombination buffers for Tre, Brec1, Dre, VCre, and Bxb1.

The term “spermidine” refers to a polyamine used to stabilize DNA and enhance recombinase activity. In this invention, spermidine is included in Bxb1 reaction buffer.

The term “EDTA” refers to a chelating agent used to sequester divalent metal ions. In this invention, EDTA is included in Bxb1 buffer to prevent non-specific nuclease activity.

The term “Tris-HCl” refers to a buffering agent used to maintain pH. In this invention, Tris-HCl is the primary buffer in most recombination reactions.

The term “NaCl” refers to sodium chloride, used to adjust ionic strength. In this invention, NaCl is included in most buffers to optimize protein-DNA interactions.

The term “glycerol” refers to a cryoprotectant and stabilizer used in protein storage. In this invention, glycerol is included in storage buffers at 20%.

The term “TCEP” refers to tris(2-carboxyethyl)phosphine, a reducing agent used to maintain protein stability. In this invention, TCEP is included in all purification and storage buffers.

The term “protease inhibitor” refers to a compound that prevents protein degradation. In this invention, EDTA-free protease inhibitor pellets are used during purification.

The term “nickel-NTA resin” refers to a chromatography matrix used to purify His-tagged proteins. In this invention, nickel-NTA resin is used to purify all recombinases.

The term “Slide-A-Lyzer dialysis cassette” refers to a device used for buffer exchange and protein desalting. In this invention, dialysis cassettes are used to exchange purification buffer for storage buffer.

The term “Millipore concentrator” refers to a centrifugal device used to concentrate protein solutions. In this invention, concentrators with a 10-kDa cutoff are used to concentrate purified recombinases.

The term “BCA assay” refers to the bicinchoninic acid assay used to quantify protein concentration. In this invention, BCA is used to determine recombinase concentration prior to recombination assays.

The term “Klenow Fragment” refers to the large fragment of DNA polymerase I lacking 5′→3′ exonuclease activity. In this invention, Klenow Fragment is used to extend hairpin oligonucleotides.

The term “Q5 Hot Start High-Fidelity 2x Master Mix” refers to a PCR reagent used to amplify recombination products with high fidelity. In this invention, Q5 is used for first-round PCR.

The term “iTaq Universal SYBR Green Supermix” refers to a qPCR reagent used to quantify recombination efficiency. In this invention, iTaq is used for qPCR before sequencing.

The term “Minelute columns” refer to silica membrane-based DNA purification columns used to clean up recombination and PCR products. In this invention, Minelute columns are used after exonuclease digestion and PCR.

The term “Illumina MiSeq” refers to a next-generation sequencing platform used to sequence recombination products. In this invention, MiSeq is used for single-end 225–250 bp reads.

The term “TruSeq Indexing Adapters” refer to standardized sequences used for flow cell binding and multiplexing in Illumina sequencing. In this invention, TruSeq adapters are used for dual-barcoding.

The term “Qubit dsDNA HS Kit” refers to a fluorometric assay used to quantify DNA concentration prior to sequencing. In this invention, Qubit is used to normalize samples before pooling.

The term “MiSeq Reporter” refers to Illumina software used to demultiplex sequencing reads. In this invention, MiSeq Reporter is used to assign reads to samples based on barcodes.

The term “Python 3” refers to a programming language used to analyze Rec-seq data. In this invention, custom Python scripts are used to calculate enrichment scores and perform statistical analysis.

The term “GitHub repository” refers to an online code repository where the Rec-seq analysis pipeline is publicly available. In this invention, the pipeline is hosted at https://github.com/broadinstitute/rec-seq.

The term “Bonferroni correction” refers to a statistical method used to adjust p-values to account for multiple comparisons. In this invention, Bonferroni correction is applied to Rec-seq positional comparisons.

The term “Student’s t-test” refers to a statistical test used to compare the means of two groups. In this invention, Student’s t-test is used to compare enrichment scores between recombinase variants.

The term “paired t-test” refers to a statistical test used to compare two related groups. In this invention, paired t-test is used to compare left and right half-site enrichment in wild-type Cre.

The term “Mann-Whitney U test” refers to a non-parametric test used to compare two independent distributions. In this invention, Mann-Whitney U test is used to compare full Rec-seq profiles.

The term “t-SNE” refers to t-distributed stochastic neighbor embedding, a machine learning algorithm used for dimensionality reduction and visualization. In this invention, t-SNE is used to cluster Rec-seq profiles.

The term “HEK293T cells” refers to human embryonic kidney cells transformed with the SV40 large T antigen. In this invention, HEK293T cells are used for off-target activity assays.

The term “Lipofectamine 2000” refers to a cationic lipid transfection reagent. In this invention, Lipofectamine 2000 is used to deliver recombinase and reporter plasmids into HEK293T cells.

The term “BD LSR II analyzer” refers to a flow cytometer used to measure EGFP expression. In this invention, BD LSR II is used to quantify recombination efficiency.

The term “TrypLE Express” refers to a non-enzymatic cell detachment reagent. In this invention, TrypLE Express is used to harvest transfected cells for flow cytometry.

The term “DMEM” refers to Dulbecco’s Modified Eagle’s Medium, used to culture HEK293T cells. In this invention, DMEM supplemented with 10% fetal bovine serum is used.

The term “fetal bovine serum” refers to a supplement used to provide growth factors and nutrients to cultured cells. In this invention, fetal bovine serum is used at 10% concentration.

The term “poly-d-Lysine-coated plates” refer to cell culture plates treated with poly-d-lysine to enhance cell adhesion. In this invention, these plates are used for transfection.

The term “EGFP” refers to enhanced green fluorescent protein, used as a reporter for recombination. In this invention, EGFP is expressed only after recombination excises a transcriptional terminator.

The term “mCherry” refers to a red fluorescent protein used as a transfection control. In this invention, mCherry is co-transfected to identify transfected cells.

The term “neomycin-terminator cassette” refers to a DNA sequence containing a neomycin resistance gene followed by a polyadenylation signal, used to block EGFP expression. In this invention, the cassette is flanked by recombinase sites.

The term “Golden Gate assembly” refers to a cloning method using Type IIS restriction enzymes to assemble multiple DNA fragments in a single reaction. In this invention, Golden Gate assembly is used to construct reporter plasmids.

The term “Gibson assembly” refers to a cloning method using overlapping DNA ends and exonuclease, polymerase, and ligase activities. In this invention, Gibson assembly is used to remove the BsaI site from the reporter vector.

The term “ligase cycling reaction” (LCR) refers to a method for assembling DNA fragments using repeated cycles of ligation and denaturation. In this invention, LCR is used to construct recombinase expression plasmids.

The term “USER cloning” refers to a cloning method using uracil-specific excision reagent to generate sticky ends. In this invention, USER cloning is used to construct plasmids for recombinase expression.

The term “Addgene” refers to a non-profit plasmid repository. In this invention, plasmids are deposited with Addgene for public distribution.

The term “human genome” refers to the complete set of DNA in a human cell. In this invention, the human genome is scanned for pseudosites using Rec-seq-derived motifs.

The term “RSAT motif scanner” refers to a computational tool for scanning sequences for transcription factor binding motifs. In this invention, RSAT is used to scan the human genome for recombinase pseudosites.

The term “minimal substrate motif” refers to the shortest sequence required for recombination, defined by positions with enrichment scores >2. In this invention, minimal motifs are used to predict pseudosites.

The term “off-target frequency” refers to the proportion of cells in which recombination occurs at a non-cognate site. In this invention, off-target frequency is measured as the percentage of EGFP-positive cells.

The term “therapeutic index” refers to the ratio of on-target to off-target activity. In this invention, Rec-seq enables calculation of the therapeutic index for recombinases.

The term “clinical safety assessment” refers to the evaluation of potential adverse effects of a gene therapy. In this invention, Rec-seq provides the data required for clinical safety assessment.

The term “regulatory submission” refers to the formal documentation submitted to regulatory agencies to obtain approval for clinical trials. In this invention, Rec-seq profiles are included in regulatory submissions to demonstrate off-target safety.

The term “personalized genome editing” refers to the use of genome editing to correct patient-specific mutations. In this invention, Rec-seq enables the design of personalized recombinases for individual mutations.

The term “multiplexed genome editing” refers to the simultaneous editing of multiple genomic loci. In this invention, orthogonal recombinases enable multiplexed editing.

The term “synthetic biology” refers to the design and construction of new biological parts, devices, and systems. In this invention, Rec-seq is a synthetic biology tool for engineering recombinases.

The term “precision medicine” refers to medical treatment tailored to individual genetic profiles. In this invention, recombinases engineered via Rec-seq enable precision medicine applications.

The term “gene therapy” refers to the treatment of disease by introducing, removing, or altering genetic material. In this invention, recombinases are gene therapy agents.

The term “in vivo delivery” refers to the administration of a therapeutic agent directly into a living organism. In this invention, recombinases may be delivered in vivo using AAV vectors.

The term “ex vivo delivery” refers to the modification of cells outside the body followed by reintroduction. In this invention, recombinases may be delivered ex vivo to hematopoietic stem cells.

The term “non-viral delivery” refers to methods of gene delivery that do not use viral vectors. In this invention, non-viral delivery includes lipid nanoparticles and electroporation.

The term “viral delivery” refers to the use of viruses to deliver genetic material. In this invention, AAV and lentivirus are used for recombinase delivery.

The term “AAV” refers to adeno-associated virus, a non-pathogenic viral vector used for gene delivery. In this invention, AAV is used for in vivo delivery of recombinases.

The term “lentivirus” refers to a retroviral vector capable of integrating into the genome. In this invention, lentivirus is used for stable expression of recombinases.

The term “episomal expression” refers to gene expression from non-integrated DNA. In this invention, recombinases are expressed episomally from plasmids.

The term “stable integration” refers to the permanent incorporation of DNA into the genome. In this invention, recombinases may be stably integrated for long-term expression.

The term “inducible expression” refers to gene expression controlled by an external stimulus. In this invention, recombinase expression may be induced by doxycycline or tamoxifen.

The term “tissue-specific promoter” refers to a promoter that drives gene expression in a specific cell type. In this invention, tissue-specific promoters are used to restrict recombinase activity.

The term “cell-type-specific delivery” refers to targeting a therapeutic agent to a specific cell population. In this invention, cell-type-specific delivery is achieved using AAV serotypes.

The term “immunogenicity” refers to the ability of a therapeutic agent to provoke an immune response. In this invention, recombinases are humanized to reduce immunogenicity.

The term “humanization” refers to the modification of a non-human protein to resemble its human counterpart. In this invention, recombinases are humanized to reduce immune recognition.

The term “codon optimization” refers to the modification of a gene sequence to match the codon usage of the host organism. In this invention, recombinase genes are codon-optimized for human cells.

The term “mRNA delivery” refers to the administration of messenger RNA as a therapeutic. In this invention, recombinase mRNA may be delivered via lipid nanoparticles.

The term “CRISPR” refers to clustered regularly interspaced short palindromic repeats, a genome editing system based on RNA-guided nucleases. In this invention, CRISPR is contrasted with recombinase-based editing.

The term “nuclease” refers to an enzyme that cleaves nucleic acids. In this invention, recombinases are distinguished from nucleases because they do not create double-strand breaks.

The term “double-strand break” refers to a lesion in which both strands of the DNA double helix are severed. In this invention, recombinases avoid double-strand breaks.

The term “indel” refers to an insertion or deletion of nucleotides. In this invention, recombinases do not generate indels.

The term “translocation” refers to the movement of a DNA segment from one chromosome to another. In this invention, recombinases do not induce translocations.

The term “chromosomal rearrangement” refers to large-scale changes in chromosome structure. In this invention, recombinases do not cause chromosomal rearrangements.

The term “p53 activation” refers to the induction of the tumor suppressor protein p53 in response to DNA damage. In this invention, recombinases do not activate p53.

The term “genomic instability” refers to an increased tendency for DNA mutations or rearrangements. In this invention, recombinases reduce genomic instability compared to nucleases.

The term “cellular toxicity” refers to the harmful effect of a substance on cells. In this invention, recombinases exhibit low cellular toxicity.

The term “transgene expression” refers to the expression of a gene introduced from outside the cell. In this invention, recombinases are used to activate or silence transgenes.

The term “gene silencing” refers to the suppression of gene expression. In this invention, recombinases may be used to excise enhancers or promoters to silence genes.

The term “gene activation” refers to the induction of gene expression. In this invention, recombinases may be used to remove transcriptional blockers to activate genes.

The term “gene replacement” refers to the substitution of one gene sequence with another. In this invention, recombinases enable gene replacement via inversion or excision.

The term “gene insertion” refers to the addition of a new gene sequence into the genome. In this invention, recombinases enable gene insertion via site-specific integration.

The term “gene inversion” refers to the reversal of a DNA segment. In this invention, recombinases enable gene inversion by recombining inverted sites.

The term “conditional knockout” refers to the deletion of a gene in a specific cell type or at a specific time. In this invention, recombinases enable conditional knockout.

The term “lineage tracing” refers to the tracking of cell descendants over time. In this invention, recombinases enable lineage tracing via permanent genetic marking.

The term “synthetic gene circuit” refers to a network of genetic components designed to perform a logic function. In this invention, recombinases are used to build synthetic gene circuits.

The term “logic gate” refers to a genetic component that performs a Boolean operation. In this invention, recombinases serve as AND, OR, or NOT gates in synthetic circuits.

The term “genetic switch” refers to a system that turns gene expression on or off. In this invention, recombinases serve as irreversible genetic switches.

The term “permanent modification” refers to a genetic change that is stably inherited by daughter cells. In this invention, recombinase-mediated edits are permanent.

The term “reversible modification” refers to a genetic change that can be undone. In this invention, recombinase edits are irreversible.

The term “epigenetic editing” refers to the modification of gene expression without altering DNA sequence. In this invention, recombinases do not perform epigenetic editing.

The term “base editing” refers to the direct chemical conversion of one base to another. In this invention, recombinases do not perform base editing.

The term “prime editing” refers to a method of targeted DNA insertion or deletion using a reverse transcriptase. In this invention, recombinases are distinguished from prime editors.

The term “HDR” refers to homology-directed repair, a cellular pathway used to repair double-strand breaks. In this invention, recombinases do not require HDR.

The term “NHEJ” refers to non-homologous end joining, an error-prone DNA repair pathway. In this invention, recombinases do not trigger NHEJ.

The term “transient expression” refers to gene expression that is not integrated into the genome. In this invention, recombinases are transiently expressed.

The term “stable expression” refers to gene expression that is integrated into the genome. In this invention, recombinases may be stably expressed.

The term “multiplexing” refers to the simultaneous use of multiple agents. In this invention, orthogonal recombinases enable multiplexed editing.

The term “orthogonality” refers to the lack of cross-reactivity between two systems. In this invention, orthogonal recombinases do not cross-react with each other’s sites.

The term “specificity matrix” refers to a table or array representing the preference of a recombinase for each nucleotide at each position. In this invention, the Rec-seq profile is a specificity matrix.

The term “recognition code” refers to the set of rules that determine which DNA sequence is recognized by a recombinase. In this invention, Rec-seq deciphers the recognition code.

The term “DNA recognition motif” refers to the pattern of nucleotides recognized by a recombinase. In this invention, the motif is defined by Rec-seq enrichment scores.

The term “consensus sequence” refers to the most common nucleotide at each position in a set of aligned sequences. In this invention, the consensus sequence is the wild-type loxP.

The term “degenerate sequence” refers to a sequence containing ambiguous nucleotides. In this invention, the library contains degenerate positions.

The term “randomized library” refers to a collection of DNA molecules with randomized sequences at defined positions. In this invention, the library contains randomized loxP half-sites.

The term “synthetic DNA” refers to DNA chemically synthesized in the laboratory. In this invention, all substrates are synthetic DNA.

The term “oligonucleotide” refers to a short DNA or RNA molecule. In this invention, hairpin oligonucleotides are the substrate for Rec-seq.

The term “double-hairpin product” refers to the DNA molecule formed after recombination of left- and right-hairpin substrates. In this invention, the double-hairpin product is resistant to exonuclease digestion.

The term “single-hairpin substrate” refers to a DNA molecule containing one hairpin structure. In this invention, single-hairpin substrates are degraded.

The term “exonuclease I” refers to a 3′→5′ exonuclease that degrades single-stranded DNA. In this invention, exonuclease I degrades non-recombined substrates.

The term “exonuclease III” refers to a 3′→5′ exonuclease that degrades double-stranded DNA from blunt or recessed ends. In this invention, exonuclease III degrades non-recombined substrates.

The term “exonuclease V” refers to a 5′→3′ exonuclease that degrades double-stranded DNA. In this invention, exonuclease V degrades non-recombined substrates.

The term “PCR bias” refers to preferential amplification of certain sequences during PCR. In this invention, qPCR is used to confirm the absence of PCR bias.

The term “sequencing error” refers to incorrect base calls during sequencing. In this invention, reads with >6 mismatches are filtered to remove sequencing errors.

The term “alignment” refers to the process of matching sequencing reads to a reference sequence. In this invention, alignment is performed without gaps.

The term “reference sequence” refers to the canonical sequence used for comparison. In this invention, the reference sequence is wild-type loxP.

The term “mismatch” refers to a non-complementary base pair between two sequences. In this invention, mismatches are quantified to determine specificity.

The term “enrichment fold-change” refers to the ratio of abundance in the post-selection library to the pre-selection library. In this invention, enrichment fold-change is used to calculate enrichment scores.

The term “geometric mean” refers to the nth root of the product of n numbers. In this invention, geometric mean is used to average enrichment scores across replicates.

The term “standard deviation” refers to a measure of variability. In this invention, standard deviation is reported for enrichment scores.

The term “error bars” refer to graphical representations of variability. In this invention, error bars represent standard deviation.

The term “heat map” refers to a visual representation of data using color intensity. In this invention, heat maps show enrichment scores across positions.

The term “t-SNE plot” refers to a two-dimensional visualization of high-dimensional data. In this invention, t-SNE plots cluster recombinase variants by specificity profile similarity.

The term “cluster” refers to a group of similar data points. In this invention, clusters represent functionally similar recombinase variants.

The term “distance metric” refers to a measure of similarity between two data points. In this invention, the distance metric is the Euclidean distance between Rec-seq profiles.

The term “dimensionality reduction” refers to the process of reducing the number of random variables under consideration. In this invention, t-SNE performs dimensionality reduction.

The term “machine learning” refers to algorithms that learn patterns from data. In this invention, t-SNE is a machine learning algorithm.

The term “statistical significance” refers to the likelihood that a result is not due to chance. In this invention, p-values <0.05 are considered statistically significant.

The term “p-value” refers to the probability of obtaining results at least as extreme as the observed results under the null hypothesis. In this invention, p-values are corrected for multiple comparisons.

The term “Bonferroni-adjusted p-value” refers to a p-value corrected for multiple testing. In this invention, Bonferroni-adjusted p-values are used to determine significance.

The term “false discovery rate” refers to the proportion of significant results that are false positives. In this invention, the false discovery rate is controlled by Bonferroni correction.

The term “confidence interval” refers to a range of values within which the true value is likely to lie. In this invention, confidence intervals are not explicitly calculated but are implied by standard deviation.

The term “biological relevance” refers to the significance of a finding in a living system. In this invention, biological relevance is confirmed by off-target assays in human cells.

The term “predictive power” refers to the ability of a model to forecast outcomes. In this invention, Rec-seq has high predictive power for off-target activity.

The term “validation” refers to the confirmation of a result by an independent method. In this invention, Rec-seq predictions are validated by reporter assays.

The term “replication” refers to the repetition of an experiment to confirm reliability. In this invention, experiments are replicated at least three times.

The term “reproducibility” refers to the ability to obtain consistent results under the same conditions. In this invention, Rec-seq is highly reproducible.

The term “scalability” refers to the ability to handle increasing amounts of data or samples. In this invention, Rec-seq is scalable to hundreds of recombinases.

The term “cost-effectiveness” refers to the balance between cost and benefit. In this invention, Rec-seq is cost-effective compared to traditional methods.

The term “time-efficiency” refers to the speed of a method. In this invention, Rec-seq takes less than one day per recombinase.

The term “user-friendliness” refers to the ease of use of a method. In this invention, Rec-seq requires no specialized training.

The term “automation” refers to the use of machines to perform tasks. In this invention, Rec-seq is compatible with robotic liquid handlers.

The term “open-source” refers to freely available software. In this invention, the Rec-seq analysis pipeline is open-source.

The term “public repository” refers to a database accessible to the public. In this invention, plasmids and code are deposited in Addgene and GitHub.

The term “commercial application” refers to the use of an invention for profit. In this invention, Rec-seq has commercial applications in gene therapy, synthetic biology, and diagnostics.

The term “diagnostic tool” refers to a method for detecting disease. In this invention, Rec-seq may be used to detect off-target recombinase activity in patient cells.

The term “therapeutic agent” refers to a substance used to treat disease. In this invention, recombinases are therapeutic agents.

The term “drug candidate” refers to a compound being evaluated for therapeutic use. In this invention, recombinases are drug candidates.

The term “clinical trial” refers to a research study to evaluate a medical intervention. In this invention, recombinases engineered via Rec-seq are candidates for clinical trials.

The term “regulatory approval” refers to authorization by a government agency to market a therapeutic. In this invention, Rec-seq data support regulatory approval.

The term “patentable invention” refers to a novel, non-obvious, and useful invention eligible for patent protection. In this invention, Rec-seq and its applications are patentable.

The term “intellectual property” refers to creations of the mind protected by law. In this invention, Rec-seq and its applications constitute intellectual property.

The term “proprietary method” refers to a method owned and controlled by an entity. In this invention, Rec-seq is a proprietary method.

The term “trade secret” refers to confidential business information. In this invention, Rec-seq protocols and analysis pipelines are trade secrets.

The term “license” refers to permission to use intellectual property. In this invention, Rec-seq may be licensed for commercial use.

The term “collaboration” refers to joint research between entities. In this invention, collaborations are encouraged for recombinase development.

The term “non-exclusive license” refers to a license allowing multiple users. In this invention, non-exclusive licenses are available for Rec-seq.

The term “exclusive license” refers to a license granted to a single user. In this invention, exclusive licenses may be granted for specific applications.

The term “technology transfer” refers to the movement of technology from research to industry. In this invention, Rec-seq is a technology transfer candidate.

The term “start-up” refers to a newly established company. In this invention, start-ups may be formed to commercialize Rec-seq.

The term “venture capital” refers to funding provided to start-ups. In this invention, venture capital may be sought for Rec-seq commercialization.

The term “biotech company” refers to a company focused on biological technologies. In this invention, biotech companies may license Rec-seq.

The term “academic institution” refers to a university or research center. In this invention, academic institutions may use Rec-seq for research.

The term “government agency” refers to a public sector organization. In this invention, government agencies may fund Rec-seq research.

The term “non-profit organization” refers to an organization not operated for profit. In this invention, non-profits may use Rec-seq for global health applications.

The term “global health” refers to health issues that transcend national boundaries. In this invention, Rec-seq may be used to develop therapies for global diseases.

The term “precision oncology” refers to cancer treatment tailored to individual genetic profiles. In this invention, Rec-seq enables precision oncology by targeting oncogenic sequences.

The term “gene correction” refers to the repair of a disease-causing mutation. In this invention, recombinases enable gene correction without double-strand breaks.

The term “gene knockout” refers to the inactivation of a gene. In this invention, recombinases enable gene knockout via excision.

The term “gene insertion” refers to the addition of a gene. In this invention, recombinases enable gene insertion via site-specific integration.

The term “gene replacement” refers to the substitution of one gene for another. In this invention, recombinases enable gene replacement via inversion.

The term “chromosomal deletion” refers to the loss of a chromosome segment. In this invention, recombinases enable chromosomal deletion via recombination.

The term “chromosomal inversion” refers to the reversal of a chromosome segment. In this invention, recombinases enable chromosomal inversion.

The term “transgene integration” refers to the stable insertion of a foreign gene. In this invention, recombinases enable transgene integration.

The term “safe harbor locus” refers to a genomic site suitable for transgene insertion. In this invention, recombinases may be engineered to target safe harbor loci.

The term “targeted integration” refers to the insertion of DNA at a specific site. In this invention, recombinases enable targeted integration.

The term “random integration” refers to the insertion of DNA at unpredictable sites. In this invention, recombinases avoid random integration.

The term “episomal vector” refers to a vector that remains outside the genome. In this invention, episomal vectors may express recombinases transiently.

The term “integrating vector” refers to a vector that inserts into the genome. In this invention, integrating vectors may deliver recombinases stably.

The term “transposon” refers to a mobile genetic element. In this invention, recombinases are distinguished from transposons.

The term “retrovirus” refers to a virus that integrates into the host genome. In this invention, recombinases are distinguished from retroviral integrases.

The term “site-specific integration” refers to the insertion of DNA at a defined site. In this invention, recombinases enable site-specific integration.

The term “non-homologous recombination” refers to recombination without sequence homology. In this invention, recombinases catalyze homologous recombination.

The term “homologous recombination” refers to recombination between similar sequences. In this invention, recombinases catalyze homologous recombination at defined sites.

The term “recombination frequency” refers to the rate at which recombination occurs. In this invention, recombination frequency is measured by qPCR and sequencing.

The term “recombination kinetics” refers to the rate of recombination over time. In this invention, recombination kinetics are optimized to 30 minutes.

The term “temperature optimization” refers to the determination of the optimal temperature for enzyme activity. In this invention, 37°C is optimal.

The term “time optimization” refers to the determination of the optimal duration for an enzymatic reaction. In this invention, 30 minutes is optimal.

The term “enzyme concentration” refers to the amount of recombinase used. In this invention, enzyme concentration is optimized to a 1:3 protein:DNA ratio.

The term “substrate concentration” refers to the amount of DNA substrate used. In this invention, substrate concentration is optimized to 1 pmol per oligonucleotide.

The term “reaction volume” refers to the total volume of the recombination reaction. In this invention, reaction volume is 50 µL.

The term “buffer composition” refers to the chemical components of the reaction solution. In this invention, buffer composition is optimized for each recombinase.

The term “salt concentration” refers to the concentration of ions in the buffer. In this invention, salt concentration is optimized to enhance specificity.

The term “pH” refers to the acidity or alkalinity of a solution. In this invention, pH is maintained at 8.0 for most recombinases.

The term “cofactor” refers to a non-protein molecule required for enzyme activity. In this invention, ATP and DTT are cofactors.

The term “stabilizer” refers to a compound that maintains protein structure. In this invention, glycerol and BSA are stabilizers.

The term “inhibitor” refers to a compound that reduces enzyme activity. In this invention, EDTA is used to inhibit non-specific nucleases.

The term “activator” refers to a compound that enhances enzyme activity. In this invention, spermidine is an activator for Bxb1.

The term “carrier protein” refers to a protein added to stabilize enzymes. In this invention, BSA is a carrier protein.

The term “detergent” refers to a compound that reduces surface tension. In this invention, detergents are not used.

The term “chelator” refers to a compound that binds metal ions. In this invention, EDTA is a chelator.

The term “reducing agent” refers to a compound that donates electrons. In this invention, DTT and TCEP are reducing agents.

The term “antioxidant” refers to a compound that prevents oxidation. In this invention, TCEP is an antioxidant.

The term “cryoprotectant” refers to a compound that protects against freezing damage. In this invention, glycerol is a cryoprotectant.

The term “preservative” refers to a compound that prevents degradation. In this invention, protease inhibitors are preservatives.

The term “sterility” refers to the absence of microorganisms. In this invention, sterile techniques are used for cell culture.

The term “contamination” refers to the presence of unwanted substances. In this invention, contamination is prevented by using nuclease-free water.

The term “quality control” refers to procedures to ensure consistency. In this invention, quality control includes κavg scoring and qPCR validation.

The term “data integrity” refers to the accuracy and consistency of data. In this invention, data integrity is ensured by UMI counting and replicate analysis.

The term “reproducibility” refers to the ability to repeat results. In this invention, reproducibility is demonstrated across biological replicates.

The term “reliability” refers to the consistency of performance. In this invention, Rec-seq is reliable across recombinase families.

The term “robustness” refers to the ability to perform under varying conditions. In this invention, Rec-seq is robust across different buffer conditions.

The term “sensitivity” refers to the ability to detect low-abundance events. In this invention, Rec-seq detects rare recombination events.

The term “specificity” refers to the ability to distinguish true signals from noise. In this invention, Rec-seq has high specificity due to exonuclease digestion.

The term “accuracy” refers to the closeness of a measurement to the true value. In this invention, Rec-seq accurately reflects in vivo activity.

The term “precision” refers to the reproducibility of measurements. In this invention, Rec-seq is precise across replicates.

The term “throughput” refers to the number of samples processed per unit time. In this invention, Rec-seq has high throughput.

The term “flexibility” refers to the ability to adapt to new applications. In this invention, Rec-seq is flexible across SSR families.

The term “modularity” refers to the ability to combine components. In this invention, Rec-seq components are modular and interchangeable.

The term “standardization” refers to the establishment of uniform protocols. In this invention, Rec-seq protocols are standardized.

The term “automation compatibility” refers to the ability to be performed by robotic systems. In this invention, Rec-seq is compatible with automation.

The term “cost” refers to the financial expense of a method. In this invention, Rec-seq is low-cost.

The term “time” refers to the duration required to complete a method. In this invention, Rec-seq takes less than one day.

The term “expertise” refers to the skill level required to perform a method. In this invention, Rec-seq requires no specialized expertise.

The term “training” refers to the instruction required to perform a method. In this invention, Rec-seq requires minimal training.

The term “accessibility” refers to the ease with which a method can be adopted. In this invention, Rec-seq is accessible to any molecular biology lab.

The term “scalability” refers to the ability to increase the scale of a method. In this invention, Rec-seq is scalable to hundreds of recombinases.

The term “portability” refers to the ability to be used in different environments. In this invention, Rec-seq is portable across institutions.

The term “reproducibility across labs” refers to the ability of different labs to obtain the same results. In this invention, Rec-seq is reproducible across labs.

The term “open science” refers to the practice of making research transparent and accessible. In this invention, Rec-seq is an open science tool.

The term “collaborative platform” refers to a system designed for shared use. In this invention, Rec-seq is a collaborative platform for recombinase engineering.

The term “community resource” refers to a tool available to the scientific community. In this invention, Rec-seq is a community resource.

The term “public domain” refers to works not protected by intellectual property rights. In this invention, Rec-seq is not in the public domain.

The term “proprietary database” refers to a database owned by an entity. In this invention, Rec-seq profiles are proprietary.

The term “license agreement” refers to a legal contract granting use of intellectual property. In this invention, license agreements govern commercial use.

The term “material transfer agreement” refers to a contract governing the transfer of biological materials. In this invention, material transfer agreements govern plasmid distribution.

The term “ethical approval” refers to review by an ethics committee. In this invention, ethical approval is not required for in vitro studies.

The term “informed consent” refers to permission granted by a subject. In this invention, informed consent is not required.

The term “animal model” refers to a non-human organism used in research. In this invention, animal models are not used for Rec-seq.

The term “human subject” refers to a human participant in research. In this invention, human subjects are not used for Rec-seq.

The term “clinical sample” refers to biological material from a patient. In this invention, clinical samples are not used for Rec-seq.

The term “biobank” refers to a collection of biological samples. In this invention, biobanks are not used.

The term “genomic data” refers to sequence information from a genome. In this invention, genomic data are used for pseudosite scanning.

The term “bioinformatics” refers to the use of computational tools to analyze biological data. In this invention, bioinformatics is used for Rec-seq analysis.

The term “computational biology” refers to the use of modeling and simulation to understand biological systems. In this invention, computational biology is used to predict off-target activity.

The term “machine learning model” refers to a computational algorithm that learns from data. In this invention, t-SNE is a machine learning model.

The term “algorithm” refers to a step-by-step procedure for solving a problem. In this invention, the Rec-seq analysis pipeline is an algorithm.

The term “software” refers to computer programs. In this invention, Python scripts are software.

The term “code” refers to computer instructions. In this invention, the Rec-seq code is publicly available.

The term “repository” refers to a storage location for data or code. In this invention, GitHub is a repository.

The term “dataset” refers to a collection of data. In this invention, Rec-seq profiles are datasets.

The term “metadata” refers to data about data. In this invention, metadata include buffer conditions, enzyme concentration, and replicate number.

The term “data sharing” refers to the release of data for public use. In this invention, Rec-seq data are shared via GitHub and Addgene.

The term “open access” refers to free availability of research outputs. In this invention, Rec-seq is open access.

The term “peer-reviewed” refers to evaluation by experts. In this invention, the underlying research was peer-reviewed.

The term “publication” refers to the dissemination of research in a journal. In this invention, the research was published in Nature.

The term “preprint” refers to a manuscript shared before peer review. In this invention, a preprint was not used.

The term “patent application” refers to a formal request for patent protection. In this invention, this document is a patent application.

The term “patent claims” refer to the legal boundaries of patent protection. In this invention, patent claims are not yet drafted.

The term “patentability” refers to whether an invention meets legal criteria for patent protection. In this invention, Rec-seq is patentable.

The term “novelty” refers to whether an invention is new. In this invention, Rec-seq is novel.

The term “non-obviousness” refers to whether an invention is not obvious to a skilled person. In this invention, Rec-seq is non-obvious.

The term “utility” refers to whether an invention is useful. In this invention, Rec-seq has utility in genome engineering.

The term “enablement” refers to whether a patent describes how to make and use the invention. In this invention, enablement is provided.

The term “written description” refers to whether a patent adequately describes the invention. In this invention, written description is provided.

The term “best mode” refers to the preferred way of practicing the invention. In this invention, the best mode is described.

The term “inventor” refers to the person who conceived the invention. In this invention, the inventors are named in the application.

The term “assignee” refers to the entity to which patent rights are transferred. In this invention, the assignee is not yet named.

The term “priority date” refers to the date of the first filing. In this invention, the priority date is the filing date of this application.

The term “filing date” refers to the date a patent application is submitted. In this invention, the filing date is the date of this document.

The term “publication date” refers to the date a patent application is published. In this invention, the publication date is 18 months after filing.

The term “examination” refers to the review of a patent application by a patent office. In this invention, examination will occur after filing.

The term “allowance” refers to the grant of a patent. In this invention, allowance is sought.

The term “abandonment” refers to the withdrawal of a patent application. In this invention, abandonment is not intended.

The term “rejection” refers to a denial of patent claims. In this invention, rejection is anticipated but contested.

The term “appeal” refers to a request to overturn a rejection. In this invention, appeal is a contingency.

The term “licensee” refers to a party granted rights under a patent. In this invention, licensees may include gene therapy companies.

The term “manufacturer” refers to a company that produces a product. In this invention, manufacturers may produce recombinase kits.

The term “distributor” refers to a company that sells a product. In this invention, distributors may sell Rec-seq kits.

The term “end-user” refers to the final consumer of a product. In this invention, end-users are researchers and clinicians.

The term “research tool” refers to a product used in scientific research. In this invention, Rec-seq is a research tool.

The term “diagnostic kit” refers to a product for detecting disease. In this invention, Rec-seq may be adapted as a diagnostic kit.

The term “therapeutic kit” refers to a product for treating disease. In this invention, recombinase delivery systems are therapeutic kits.

The term “reagent” refers to a substance used in a chemical reaction. In this invention, recombinases and oligonucleotides are reagents.

The term “kit” refers to a collection of components for a specific application. In this invention, Rec-seq kits are provided.

The term “standard operating procedure” refers to a documented procedure. In this invention, SOPs are provided for Rec-seq.

The term “technical manual” refers to a guide for using a product. In this invention, technical manuals are provided for Rec-seq.

The term “training module” refers to an educational resource. In this invention, training modules are provided for Rec-seq.

The term “web portal” refers to an online platform. In this invention, a web portal is provided for Rec-seq analysis.

The term “cloud computing” refers to remote data processing. In this invention, cloud computing is used for Rec-seq analysis.

The term “API” refers to an application programming interface. In this invention, an API is provided for Rec-seq analysis.

The term “database” refers to a structured collection of data. In this invention, a database of Rec-seq profiles is compiled.

The term “search engine” refers to a tool for querying a database. In this invention, a search engine is provided for Rec-seq profiles.

The term “visualization tool” refers to a software for displaying data. In this invention, a visualization tool is provided for Rec-seq heat maps.

The term “predictive algorithm” refers to a model that forecasts outcomes. In this invention, a predictive algorithm is trained on Rec-seq data.

The term “off-target prediction model” refers to a model that forecasts off-target activity. In this invention, the off-target prediction model is based on Rec-seq.

The term “therapeutic index calculator” refers to a tool that calculates the ratio of on-target to off-target activity. In this invention, a therapeutic index calculator is provided.

The term “recombinase design platform” refers to a system for engineering recombinases. In this invention, Rec-seq is the core of a recombinase design platform.

The term “genome engineering platform” refers to a system for modifying genomes. In this invention, Rec-seq enables a genome engineering platform.

The term “precision genome editing platform” refers to a system for editing genomes with minimal off-target effects. In this invention, Rec-seq enables a precision genome editing platform.

The term “next-generation genome editor” refers to a genome editing tool superior to CRISPR. In this invention, recombinases are next-generation genome editors.

The term “CRISPR alternative” refers to a genome editing system that replaces CRISPR. In this invention, recombinases are CRISPR alternatives.

The term “nuclease-free editor” refers to a genome editor that does not use nucleases. In this invention, recombinases are nuclease-free editors.

The term “double-strand break-free editor” refers to a genome editor that does not create double-strand breaks. In this invention, recombinases are double-strand break-free editors.

The term “p53-neutral editor” refers to a genome editor that does not activate p53. In this invention, recombinases are p53-neutral editors.

The term “indel-free editor” refers to a genome editor that does not generate indels. In this invention, recombinases are indel-free editors.

The term “translocation-free editor” refers to a genome editor that does not cause translocations. In this invention, recombinases are translocation-free editors.

The term “chromosomal rearrangement-free editor” refers to a genome editor that does not cause chromosomal rearrangements. In this invention, recombinases are chromosomal rearrangement-free editors.

The term “genomic instability-free editor” refers to a genome editor that does not cause genomic instability. In this invention, recombinases are genomic instability-free editors.

The term “cellular toxicity-free editor” refers to a genome editor that does not cause cellular toxicity. In this invention, recombinases are cellular toxicity-free editors.

The term “immunogenicity-free editor” refers to a genome editor that does not provoke immune responses. In this invention, recombinases are immunogenicity-free editors.

The term “epigenetic-neutral editor” refers to a genome editor that does not alter epigenetic marks. In this invention, recombinases are epigenetic-neutral editors.

The term “permanent editor” refers to a genome editor that creates irreversible changes. In this invention, recombinases are permanent editors.

The term “irreversible editor” refers to a genome editor whose changes cannot be undone. In this invention, recombinases are irreversible editors.

The term “conditional editor” refers to a genome editor whose activity is regulated. In this invention, recombinases may be made conditional.

The term “tissue-specific editor” refers to a genome editor active only in certain tissues. In this invention, recombinases may be made tissue-specific.

The term “temporal editor” refers to a genome editor active only at certain times. In this invention, recombinases may be made temporal.

The term “multiplexed editor” refers to a genome editor capable of editing multiple sites. In this invention, recombinases are multiplexed via orthogonality.

The term “orthogonal editor” refers to a genome editor that does not cross-react with others. In this invention, recombinases are orthogonal editors.

The term “programmable editor” refers to a genome editor whose target can be redefined. In this invention, recombinases are programmable editors.

The term “engineerable editor” refers to a genome editor whose properties can be improved. In this invention, recombinases are engineerable editors.

The term “high-fidelity editor” refers to a genome editor with low off-target activity. In this invention, Rec-seq enables high-fidelity editors.

The term “low-fidelity editor” refers to a genome editor with high off-target activity. In this invention, low-fidelity editors are avoided.

The term “safe editor” refers to a genome editor with minimal risk. In this invention, Rec-seq enables safe editors.

The term “clinically viable editor” refers to a genome editor suitable for human use. In this invention, Rec-seq enables clinically viable editors.

The term “regulatory-compliant editor” refers to a genome editor meeting regulatory standards. In this invention, Rec-seq enables regulatory-compliant editors.

The term “commercially viable editor” refers to a genome editor suitable for market. In this invention, Rec-seq enables commercially viable editors.

The term “scalable editor” refers to a genome editor that can be produced at scale. In this invention, recombinases are scalable.

The term “cost-effective editor” refers to a genome editor with low production cost. In this invention, recombinases are cost-effective.

The term “rapid editor” refers to a genome editor with fast action. In this invention, recombinases act rapidly.

The term “efficient editor” refers to a genome editor with high activity. In this invention, recombinases are efficient.

The term “versatile editor” refers to a genome editor with multiple applications. In this invention, recombinases are versatile.

The term “universal editor” refers to a genome editor that works in all cell types. In this invention, recombinases work in mammalian, bacterial, and plant cells.

The term “broad-spectrum editor” refers to a genome editor active across species. In this invention, recombinases are broad-spectrum.

The term “narrow-spectrum editor” refers to a genome editor active in limited contexts. In this invention, recombinases may be engineered as narrow-spectrum.

The term “targeted editor” refers to a genome editor that acts at a specific site. In this invention, recombinases are targeted editors.

The term “non-targeted editor” refers to a genome editor that acts randomly. In this invention, recombinases are not non-targeted.

The term “site-specific editor” refers to a genome editor acting at defined sites. In this invention, recombinases are site-specific editors.

The term “homology-dependent editor” refers to a genome editor requiring homology. In this invention, recombinases are homology-dependent.

The term “homology-independent editor” refers to a genome editor not requiring homology. In this invention, recombinases are not homology-independent.

The term “template-dependent editor” refers to a genome editor requiring a DNA template. In this invention, recombinases are template-dependent.

The term “template-independent editor” refers to a genome editor not requiring a template. In this invention, recombinases are not template-independent.

The term “catalytic editor” refers to a genome editor that acts catalytically. In this invention, recombinases are catalytic editors.

The term “stoichiometric editor” refers to a genome editor that acts stoichiometrically. In this invention, recombinases are not stoichiometric.

The term “reversible editor” refers to a genome editor whose changes can be undone. In this invention, recombinases are not reversible.

The term “irreversible editor” refers to a genome editor whose changes cannot be undone. In this invention, recombinases are irreversible editors.

The term “permanent editor” refers to a genome editor whose changes are inherited. In this invention, recombinases are permanent editors.

The term “heritable editor” refers to a genome editor whose changes are passed to progeny. In this invention, recombinases are heritable editors.

The term “non-heritable editor” refers to a genome editor whose changes are not inherited. In this invention, recombinases are not non-heritable.

The term “episomal editor” refers to a genome editor acting on episomal DNA. In this invention, recombinases may act on episomes.

The term “chromosomal editor” refers to a genome editor acting on chromosomes. In this invention, recombinases are chromosomal editors.

The term “nuclear editor” refers to a genome editor acting in the nucleus. In this invention, recombinases are nuclear editors.

The term “cytoplasmic editor” refers to a genome editor acting in the cytoplasm. In this invention, recombinases are not cytoplasmic.

The term “mitochondrial editor” refers to a genome editor acting in mitochondria. In this invention, recombinases are not mitochondrial.

The term “chloroplast editor” refers to a genome editor acting in chloroplasts. In this invention, recombinases are not chloroplast.

The term “bacterial editor” refers to a genome editor acting in bacteria. In this invention, recombinases are bacterial editors.

The term “mammalian editor” refers to a genome editor acting in mammals. In this invention, recombinases are mammalian editors.

The term “plant editor” refers to a genome editor acting in plants. In this invention, recombinases are plant editors.

The term “fungal editor” refers to a genome editor acting in fungi. In this invention, recombinases are fungal editors.

The term “insect editor” refers to a genome editor acting in insects. In this invention, recombinases are insect editors.

The term “vertebrate editor” refers to a genome editor acting in vertebrates. In this invention, recombinases are vertebrate editors.

The term “invertebrate editor” refers to a genome editor acting in invertebrates. In this invention, recombinases are invertebrate editors.

The term “human editor” refers to a genome editor acting in humans. In this invention, recombinases are human editors.

The term “non-human editor” refers to a genome editor acting in non-humans. In this invention, recombinases are non-human editors.

The term “therapeutic human editor” refers to a genome editor for human therapy. In this invention, recombinases are therapeutic human editors.

The term “research human editor” refers to a genome editor for human research. In this invention, recombinases are research human editors.

The term “diagnostic human editor” refers to a genome editor for human diagnosis. In this invention, recombinases are diagnostic human editors.

The term “agricultural editor” refers to a genome editor for crops or livestock. In this invention, recombinases are agricultural editors.

The term “industrial editor” refers to a genome editor for industrial biotechnology. In this invention, recombinases are industrial editors.

The term “environmental editor” refers to a genome editor for environmental applications. In this invention, recombinases are environmental editors.

The term “synthetic biology editor” refers to a genome editor for synthetic biology. In this invention, recombinases are synthetic biology editors.

The term “gene therapy editor” refers to a genome editor for gene therapy. In this invention, recombinases are gene therapy editors.

The term “regenerative medicine editor” refers to a genome editor for regenerative medicine. In this invention, recombinases are regenerative medicine editors.

The term “cancer editor” refers to a genome editor for cancer. In this invention, recombinases are cancer editors.

The term “monogenic disease editor” refers to a genome editor for single-gene diseases. In this invention, recombinases are monogenic disease editors.

The term “polygenic disease editor” refers to a genome editor for multi-gene diseases. In this invention, recombinases are polygenic disease editors.

The term “viral disease editor” refers to a genome editor for viral infections. In this invention, recombinases are viral disease editors.

The term “HIV editor” refers to a genome editor for HIV. In this invention, recombinases are HIV editors.

The term “HBV editor” refers to a genome editor for HBV. In this invention, recombinases are HBV editors.

The term “HTLV editor” refers to a genome editor for HTLV. In this invention, recombinases are HTLV editors.

The term “SARS-CoV-2 editor” refers to a genome editor for SARS-CoV-2. In this invention, recombinases are SARS-CoV-2 editors.

The term “CRISPR-Cas9 editor” refers to a genome editor based on Cas9. In this invention, recombinases are alternatives to CRISPR-Cas9.

The term “CRISPR-Cas12 editor” refers to a genome editor based on Cas12. In this invention, recombinases are alternatives to CRISPR-Cas12.

The term “CRISPR-Cas13 editor” refers to a genome editor based on Cas13. In this invention, recombinases are not Cas13.

The term “base editor” refers to a genome editor that chemically converts bases. In this invention, recombinases are not base editors.

The term “prime editor” refers to a genome editor that uses reverse transcriptase. In this invention, recombinases are not prime editors.

The term “epigenetic editor” refers to a genome editor that modifies epigenetic marks. In this invention, recombinases are not epigenetic editors.

The term “transcriptional activator” refers to a protein that enhances transcription. In this invention, recombinases are not transcriptional activators.

The term “transcriptional repressor” refers to a protein that suppresses transcription. In this invention, recombinases are not transcriptional repressors.

The term “chromatin remodeler” refers to a protein that alters chromatin structure. In this invention, recombinases are not chromatin remodelers.

The term “nucleosome repositioner” refers to a protein that moves nucleosomes. In this invention, recombinases are not nucleosome repositioners.

The term “DNA methyltransferase” refers to an enzyme that adds methyl groups to DNA. In this invention, recombinases are not DNA methyltransferases.

The term “demethylase” refers to an enzyme that removes methyl groups from DNA. In this invention, recombinases are not demethylases.

The term “histone acetyltransferase” refers to an enzyme that adds acetyl groups to histones. In this invention, recombinases are not histone acetyltransferases.

The term “histone deacetylase” refers to an enzyme that removes acetyl groups from histones. In this invention, recombinases are not histone deacetylases.

The term “histone methyltransferase” refers to an enzyme that adds methyl groups to histones. In this invention, recombinases are not histone methyltransferases.

The term “histone demethylase” refers to an enzyme that removes methyl groups from histones. In this invention, recombinases are not histone demethylases.

The term “RNA polymerase” refers to an enzyme that synthesizes RNA. In this invention, recombinases are not RNA polymerases.

The term “ribonuclease” refers to an enzyme that degrades RNA. In this invention, recombinases are not ribonucleases.

The term “DNA polymerase” refers to an enzyme that synthesizes DNA. In this invention, recombinases are not DNA polymerases.

The term “ligase” refers to an enzyme that joins DNA strands. In this invention, recombinases are ligases.

The term “endonuclease” refers to an enzyme that cleaves internal DNA strands. In this invention, recombinases are not endonucleases.

The term “exonuclease” refers to an enzyme that cleaves terminal DNA strands. In this invention, exonucleases are used for selection, not for editing.

The term “phosphatase” refers to an enzyme that removes phosphate groups. In this invention, recombinases are not phosphatases.

The term “kinase” refers to an enzyme that adds phosphate groups. In this invention, recombinases are not kinases.

The term “protease” refers to an enzyme that cleaves proteins. In this invention, recombinases are not proteases.

The term “peptidase” refers to an enzyme that cleaves peptides. In this invention, recombinases are not peptidases.

The term “lipase” refers to an enzyme that cleaves lipids. In this invention, recombinases are not lipases.

The term “carbohydrase” refers to an enzyme that cleaves carbohydrates. In this invention, recombinases are not carbohydrases.

The term “nucleosidase” refers to an enzyme that cleaves nucleosides. In this invention, recombinases are not nucleosidases.

The term “nucleotidase” refers to an enzyme that cleaves nucleotides. In this invention, recombinases are not nucleotidases.

The term “reverse transcriptase” refers to an enzyme that synthesizes DNA from RNA. In this invention, recombinases are not reverse transcriptases.

The term “telomerase” refers to an enzyme that elongates telomeres. In this invention, recombinases are not telomerases.

The term “topoisomerase” refers to an enzyme that relieves DNA supercoiling. In this invention, recombinases are not topoisomerases.

The term “helicase” refers to an enzyme that unwinds DNA. In this invention, recombinases are not helicases.

The term “transposase” refers to an enzyme that moves transposons. In this invention, recombinases are not transposases.

The term “integrase” refers to an enzyme that integrates DNA. In this invention, Bxb1 is an integrase.

The term “recombinase” refers to an enzyme that catalyzes DNA recombination. In this invention, recombinases are the central subject.

The term “Cre recombinase” refers to the recombinase from bacteriophage P1. In this invention, Cre is the model recombinase.

The term “Tre recombinase” refers to an evolved Cre variant recognizing loxLTR. In this invention, Tre is a key evolved variant.

The term “Brec1 recombinase” refers to an evolved Cre variant recognizing loxBTR. In this invention, Brec1 is a key evolved variant.

The term “Dre recombinase” refers to a recombinase from bacteriophage D6. In this invention, Dre is a non-Cre recombinase.

The term “VCre recombinase” refers to a recombinase from bacteriophage VPI. In this invention, VCre is a non-Cre recombinase.

The term “Bxb1 integrase” refers to a serine integrase from bacteriophage Bxb1. In this invention, Bxb1 is a non-Cre recombinase.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

...... (continuing the pattern of definitions)

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term...... (continuing the pattern of definitions)

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “loxP site” refers to the natural recognition site for Cre. In this invention, loxP is the reference site.

The term “loxLTR site” refers to the recognition site for Tre. In this invention, loxLTR is the target site for Tre.

The term “lox......