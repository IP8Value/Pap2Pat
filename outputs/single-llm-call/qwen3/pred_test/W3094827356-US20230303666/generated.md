## TECHNICAL FIELD

- introduce integrated platform for generating and engineering antibodies

The present invention relates to an integrated, cell-free platform for the generation, selection, and engineering of high-affinity antigen-binding proteins, particularly single-domain antibodies derived from camelid heavy chain variable domains, commonly referred to as VHHs or nanobodies. The platform enables the rapid, scalable, and reproducible production of antibody fragments with tailored binding properties without reliance on immunization of animals or cellular expression systems. It combines synthetic DNA library design, in vitro ribosome display technology, computational clustering of binding sequences, and iterative affinity maturation to yield fully characterized, high-performance antigen binders. The system is particularly suited for the discovery of neutralizing agents against rapidly evolving pathogens, such as SARS-CoV-2 variants, and for the development of diagnostic reagents, therapeutic candidates, and research tools requiring precise molecular recognition. The platform operates entirely in vitro, eliminating biological variability associated with immune responses and enabling direct control over library diversity, selection stringency, and sequence optimization. This represents a paradigm shift from traditional antibody discovery methods by decoupling binding function from biological host constraints and enabling the systematic exploration of vast, fully randomized sequence spaces under defined biochemical conditions.

## BACKGROUND

- motivate need for alternative antibody generation methods
- describe traditional antibody generation methods
- discuss limitations of traditional methods
- introduce in vitro antibody generation methods
- highlight recent advances in antibody library design and construction

The conventional approach to generating therapeutic and diagnostic antibodies relies on immunizing animals, such as llamas, alpacas, or mice, followed by isolation of antigen-specific B cells and cloning of their antibody genes. While effective, this method is time-consuming, labor-intensive, and subject to biological variability, including immune tolerance, limited repertoire diversity, and unpredictable immunogenicity profiles. Furthermore, the resulting antibodies often require extensive humanization to reduce immunogenicity in human patients, a process that can compromise binding affinity and stability. Alternative in vitro methods, such as phage display and yeast display, have been developed to circumvent these limitations by enabling the screening of large synthetic antibody libraries in cell-free systems. However, these methods are constrained by the capacity of cellular hosts to maintain and express large, diverse libraries, typically limited to 10^9–10^11 variants, and are susceptible to biases introduced by host cell physiology, such as protein folding inefficiencies, toxicity, and selection artifacts. Recent advances in ribosome display have overcome some of these constraints by tethering the translated protein to its encoding mRNA via stalled ribosomes, allowing for the selection of libraries exceeding 10^12 unique members without cellular transformation. Despite these improvements, existing in vitro platforms suffer from poor recovery of high-affinity binders due to sequence shuffling during PCR amplification, inadequate control over CDR diversity, and lack of computational methods to distill meaningful binding families from massive sequencing outputs. Moreover, prior synthetic nanobody libraries have relied on biased amino acid distributions derived from natural sequences, inadvertently excluding potentially superior binders that deviate from evolutionary constraints. There remains a critical need for a robust, fully synthetic, and computationally guided platform that enables unbiased exploration of sequence space, efficient recovery of rare but potent binders, and seamless integration of affinity maturation—all within a closed, scalable, and automation-compatible workflow.

## SUMMARY

- introduce antibody or antigen binding fragment
- describe CDRs selected or derived from clusters
- specify SR1, SR2, SR4, SR6, SR8, SR12, SR15, SR18, SR25, SR30
- describe SR6v15, SR6v7, SR38, SR6c3, SR4t13, or SR2c3
- introduce heavy chain antibody or VHH
- specify SR38 binding to N501Y SARS-CoV-2 variant
- describe SR6v15, SR6v7, or SR38
- introduce camelid heavy chain antibodies
- describe humanization of camelid antibodies
- specify framework residues to be humanized
- describe modification of antibody or antigen binding fragment
- introduce fusion protein
- describe fusion to another antibody or antibody fragment
- describe treatment of SARS-CoV-2 infection
- specify administration of SR38 or SR6v15
- describe detection of SARS-CoV-2
- specify detection using SR38 or SR6v15
- introduce method of generating VHH library
- describe PCR amplification and ligation
- specify primer sequences and conditions
- describe method of identifying CDRs

The invention provides an antigen-binding fragment comprising a variable domain derived from a camelid heavy chain antibody, wherein the fragment contains three complementarity-determining regions (CDRs) selected from a computationally clustered population of sequences generated by an in vitro ribosome display platform. The antigen-binding fragment may be a single-domain antibody (sdAb), specifically a VHH domain, exhibiting specific binding to the receptor-binding domain (RBD) of the SARS-CoV-2 spike protein. In specific embodiments, the antigen-binding fragment comprises the CDR sequences of SR1, SR2, SR4, SR6, SR8, SR12, SR15, SR18, SR25, or SR30, each of which was identified through CDR-directed clustering of output sequences from a fully randomized nanobody library. Notably, the variant SR6v15, SR6v7, SR38, SR6c3, SR4t13, or SR2c3 exhibits enhanced binding affinity and neutralization potency against SARS-CoV-2, with SR38 demonstrating a unique preference for binding to the N501Y-containing RBD variant over the wild-type strain. These fragments are derived from camelid heavy chain antibodies, which naturally lack light chains and possess extended CDR3 loops enabling access to cryptic epitopes. The framework regions of these VHH domains may be humanized by substituting key residues, including those at positions 42, 43, 44, and 47 in framework region 2, to reduce immunogenicity while preserving structural integrity and binding function. The antigen-binding fragment may be further modified through covalent conjugation to a carrier protein, fusion to an Fc domain, PEGylation, polysialylation, HESylation, or attachment to a nanoparticle or cholesterol moiety to enhance pharmacokinetics, stability, or delivery. In therapeutic applications, the fragment may be administered to a subject suffering from or at risk of SARS-CoV-2 infection via intranasal, intravenous, or inhalation routes, with dosages ranging from 0.1 mg/kg to 10 mg/kg, depending on formulation and target tissue. For diagnostic purposes, the fragment may be labeled with a detectable moiety such as a fluorophore, enzyme, or radionuclide and employed in immunoassays to detect SARS-CoV-2 antigens in clinical samples, including nasopharyngeal swabs, saliva, or serum. The invention further encompasses a method for generating a VHH library by performing successive rounds of PCR amplification and ligation using a mixture of DNA templates encoding conserved framework regions and oligonucleotides containing fully randomized NNB codons at each position of CDR1, CDR2, and CDR3, with CDR3 randomized first to maximize diversity, followed by CDR1 and then CDR2, according to a defined hierarchy. Primer sequences used in each stage are designed to preserve reading frame and include flanking restriction sites for directional ligation, with PCR performed at an elongation temperature of 65°C to prevent hairpin destabilization. The method further includes identifying CDRs by aligning sequences against consensus framework boundaries derived from natural nanobody sequences, where CDR boundaries are defined by sharp drops in the combined frequency of the two most abundant amino acids at each position, and clustering sequences based on pairwise CDR match scores calculated using the BLOSUM62 substitution matrix, with clusters formed when at least three CDRs exhibit a combined match score exceeding 35.

## DETAILED DESCRIPTION OF THE EXAMPLE EMBODIMENTS

### General Definitions

- define technical terms
- provide references for molecular biology
- define singular and plural forms
- define "optional" and "optionally"
- define numerical ranges
- define "about" and "approximately"
- define biological sample
- define bodily fluid
- define subject, individual, and patient
- describe various embodiments
- incorporate publications by reference

For the purposes of this disclosure, the term “antibody” refers to any immunoglobulin molecule or antigen-binding fragment thereof capable of specifically binding to a target antigen, including full-length antibodies, Fab, Fab′, F(ab′)2, scFv, sdAb, and VHH domains. The term “antigen-binding fragment” encompasses any portion of an antibody that retains the ability to bind an antigen, including those comprising one or more CDRs. The singular form of a term includes the plural unless otherwise indicated, and vice versa. The term “optional” or “optionally” means that the subsequently described event or circumstance may or may not occur, and the description includes instances where the event occurs and instances where it does not. Numerical ranges, such as “from 1 to 10,” include all integers and fractions within the range unless otherwise specified. The terms “about” and “approximately” refer to values that vary by ±10% of the stated value, unless context dictates otherwise. A “biological sample” refers to any material derived from a living organism, including tissues, cells, secretions, or excretions, and includes but is not limited to blood, plasma, serum, saliva, sputum, nasal swabs, cerebrospinal fluid, urine, and feces. A “bodily fluid” is a subset of biological samples that are liquid in nature and originate from within the body, including serum, plasma, lymph, and interstitial fluid. The terms “subject,” “individual,” and “patient” are used interchangeably to refer to any mammal, preferably a human, requiring medical diagnosis, monitoring, or treatment. Embodiments of the invention include compositions comprising the antigen-binding fragments described herein, pharmaceutical formulations containing such fragments, kits for detection or therapy, and methods of use in diagnostics, therapeutics, or research. All publications, patents, and patent applications cited herein are incorporated by reference in their entirety for all purposes.

### OVERVIEW

- introduce cell-free antibody engineering platform
- describe CeVICA platform and its applications

The invention provides a cell-free antibody engineering platform designated CeVICA, which integrates synthetic DNA library construction, ribosome display-based selection, computational clustering of binding sequences, and iterative affinity maturation into a single, automated workflow. CeVICA enables the rapid discovery of high-affinity, high-stability antigen-binding fragments without the need for animal immunization, hybridoma generation, or cellular cloning. The platform begins with a linear DNA library encoding a diverse population of VHH domains with fully randomized CDRs, synthesized via a three-stage PCR and ligation process that preserves sequence integrity and maximizes diversity. Following in vitro transcription and translation, ribosome display links each expressed VHH to its encoding mRNA, allowing for selection against immobilized antigens under controlled conditions. Selected sequences are recovered, reverse-transcribed, and amplified for subsequent rounds of selection, with each round increasing the enrichment of high-affinity binders. High-throughput sequencing of the output library is followed by CDR-directed clustering to identify distinct binding families, enabling efficient prioritization of lead candidates. CeVICA is particularly useful for targeting rapidly mutating pathogens such as SARS-CoV-2, where the ability to rapidly identify cross-reactive or variant-specific binders is critical. Applications include the development of neutralizing therapeutics, diagnostic assays, biosensors, and research reagents for viral detection, immune profiling, and epitope mapping.

### Therapeutic Antibodies or Binding Fragments of an Antibody

- define antibodies
- describe antibody fragments
- introduce antigen-binding fragments
- motivate therapeutic antibodies
- describe neutralizing antibodies
- introduce SARS-CoV-2 variants
- define variants and strains
- discuss clades and lineages
- list SARS-CoV-2 variants
- describe Phylogenetic Assignment of Named Global Outbreak (PANGO) Lineages
- introduce complementarity determining regions (CDRs)
- describe frame regions (FRs)
- motivate heavy chain antibodies
- introduce VHH domains
- describe single-domain antibodies (sdAbs)
- introduce camelid heavy chain antibody domains
- identify CDR clusters
- describe binding and neutralizing activity
- introduce amino acid changes for potency improvement
- describe CDRs and framework substitutions
- introduce mutated antibodies with enhanced neutralizing activity
- define substantially free of non-antibody protein
- introduce monoclonal antibodies
- describe polyclonal antibodies
- introduce binding portions of antibodies
- describe humanized forms of non-human antibodies
- introduce chimeric antibodies
- describe humanized residues in frame regions
- introduce human IGHV gene
- describe humanized frames based on VHHs
- introduce epitope-binding proteins
- list examples of antibody portions
- describe Ig classes and subclasses
- define IgG subclass
- define single-chain immunoglobulin
- define domain
- define constant and variable domains
- define region
- define conformation
- define specific binding
- define affinity
- describe blocking antibodies
- describe agonist and antagonist antibodies
- describe receptor-specific antibodies
- describe ligand-specific antibodies
- describe receptor activation
- describe antibody modifications
- define modified therapeutic antibodies
- describe conjugation to a carrier protein
- describe conjugation to a ligand
- describe conjugation to another antibody
- describe PEGylation
- describe polysialylation
- describe HESylation
- describe recombinant PEG mimetics
- describe Fc fusion
- describe albumin fusion
- describe nanoparticle attachment
- describe nanoparticulate encapsulation
- describe cholesterol fusion
- describe iron fusion
- describe acylation
- describe amidation
- describe glycosylation
- describe side chain oxidation
- describe phosphorylation
- describe biotinylation
- describe addition of a surface active material
- describe addition of amino acid mimetics
- describe addition of unnatural amino acids
- describe analogs
- describe spacers or linkers
- describe PEG-conjugated biomolecules
- describe PEGylation methods
- describe PEG molecule attachment
- describe glycosylation and polysialylation
- define therapeutic antibodies or binding fragments of an antibody
- fusion of albumin to one or more antibodies
- albumin binding strategies
- modification of polypeptide sequences
- conjugate modification
- hesylation modification
- protecting group covalently joined to the N-terminal amino group
- amino protecting groups
- deamination of the N-terminal amino acid
- chemically modified compositions of the antibodies
- polymer selection
- PEGylation, cholesterylation, or palmitoylation
- modification to any amino acid residue
- N-terminus modification
- substitutions of amino acids
- conservative substitutions
- non-conservative substitutions
- production of therapeutic antibodies
- in vitro production
- in vivo production
- antibody derivatives
- covalent attachment of molecules
- glycosylation, acetylation, pegylation, phosphorylation, amidation
- simple binding assays
- detection methods
- affinity biosensor methods
- administration of therapeutic antibodies
- routes of administration
- dosage amounts
- therapeutic regimens
- vector delivery
- define therapeutic antibodies or binding fragments
- describe delivery system comprising vectors or polynucleotide molecules
- define vector
- describe types of vectors
- describe use of antibodies for detection
- describe immunoassay methods
- describe immunoassay formats
- describe labels for detection
- describe methods of detecting labels
- describe platforms and methods for generating antibodies
- describe in vitro platform for generating antibodies
- describe libraries of DNA sequences encoding antibodies
- describe screening library by ribosome display
- describe identifying families of antibodies
- describe affinity maturation
- describe validation of antibodies
- describe VHH library
- describe generating libraries
- describe randomizing CDR regions
- describe PCR and ligation for each CDR
- describe using promoter sequence
- describe transcribing into mRNAs
- describe translating into antibody polypeptide
- describe ribosome display
- describe selecting for translated antibody frameworks
- describe washing and isolating mRNA
- describe converting to cDNA
- describe successive rounds of ribosome display
- describe adjusting stringency
- describe including epitope tag
- describe enriching for full-length mRNA sequences
- describe clustering antibodies having similar CDRs
- describe affinity maturation
- describe validating antibody binding and neutralization activity

The invention provides a therapeutic antibody or antigen-binding fragment comprising a VHH domain engineered to bind with high affinity to the receptor-binding domain of SARS-CoV-2, wherein the fragment is substantially free of non-antibody protein and exhibits neutralizing activity against multiple variants including Alpha, Beta, Gamma, Delta, and Omicron lineages as defined by the Phylogenetic Assignment of Named Global Outbreak (PANGO) system. The fragment may be a monoclonal antibody or a single-domain antibody derived from a camelid heavy chain, wherein the variable domain contains three complementarity-determining regions (CDRs) flanked by four framework regions (FRs) that are structurally homologous to human IGHV3-23 or IGHJ4. The CDRs may be derived from clusters identified through computational analysis of ribosome display output libraries, such as SR6v15, SR6v7, SR38, SR6c3, SR4t13, or SR2c3, each of which demonstrates binding affinity in the low nanomolar range as measured by biolayer interferometry. The fragment may be humanized by substituting framework residues at positions corresponding to VHH hallmark residues—such as F42, G43, W44, and R47—with their human VH counterparts, thereby reducing immunogenic potential while retaining binding function. The fragment may be modified by fusion to an Fc domain, albumin, or a second VHH domain to form a dimer or trimer, enhancing serum half-life and avidity. Such modifications may be achieved through genetic fusion or chemical conjugation using spacers such as glycine-serine linkers. The fragment may be further modified by PEGylation, polysialylation, HESylation, or acylation to improve pharmacokinetics, or by fusion to cholesterol, iron, or lipid moieties to facilitate cellular uptake. The fragment may be formulated for intranasal, intravenous, or inhaled administration, with dosages ranging from 0.1 mg/kg to 10 mg/kg, administered once daily or every 3–7 days depending on the clinical indication. The fragment may be delivered via viral vectors such as AAV or lentivirus encoding the nucleic acid sequence of the VHH, enabling in vivo expression. The fragment may be labeled with a detectable moiety such as horseradish peroxidase, fluorescein, or a radionuclide for use in immunoassays, including ELISA, lateral flow assays, or biosensor platforms. The fragment may be produced in vitro by cell-free expression systems, including PURExpress or wheat germ extracts, or in vivo by transgenic organisms. The fragment may be validated through binding assays, pseudovirus neutralization assays, thermal stability measurements, and refolding assays, demonstrating resistance to denaturation and retention of function after heating. The fragment may be part of a diagnostic kit comprising a solid support coated with the fragment, a detection reagent, and instructions for use in detecting SARS-CoV-2 antigens in clinical specimens. The fragment may be used as a blocking antibody to inhibit viral entry by preventing spike protein interaction with ACE2. The fragment may be engineered to act as an agonist or antagonist of receptor signaling pathways, depending on the epitope targeted. The fragment may be produced as a chimeric molecule by fusing its variable domain to a human constant region, thereby forming a chimeric antibody. The fragment may be modified by the incorporation of unnatural amino acids, phosphorylation, biotinylation, or glycosylation at specific sites to alter function or stability. The fragment may be purified to greater than 95% homogeneity using affinity chromatography, size exclusion, or ion exchange. The fragment may be stored in lyophilized form or in aqueous buffer at 2–8°C for extended periods. The fragment may be used in combination with other antiviral agents, including monoclonal antibodies, small molecule inhibitors, or interferons, to enhance therapeutic efficacy. The fragment may be encoded by a nucleic acid sequence comprising a T7 promoter, a 3×Myc epitope tag, a ribosome-stalling spacer, and a poly-A tail, enabling in vitro transcription and translation for ribosome display. The fragment may be selected from a library generated by randomizing CDR1, CDR2, and CDR3 using NNB codons, with CDR3 randomized first to preserve diversity, followed by CDR1 and then CDR2, and the library may be subjected to multiple rounds of ribosome display under increasing stringency to isolate high-affinity binders. The fragment may be validated by testing its ability to bind to RBD variants including N501Y, E484K, and K417N, and may exhibit enhanced binding to N501Y as demonstrated by SR38. The fragment may be used in a biosensor platform wherein binding is detected by surface plasmon resonance, quartz crystal microbalance, or interferometry. The fragment may be formulated as a nasal spray, inhalable powder, or injectable solution for prophylactic or therapeutic use. The fragment may be part of a delivery system comprising a lipid nanoparticle or polymer-based carrier encapsulating the nucleic acid encoding the fragment for in vivo expression. The fragment may be used in a method of detecting SARS-CoV-2 in a biological sample by contacting the sample with the fragment immobilized on a solid support and detecting bound antigen using a labeled secondary reagent. The fragment may be used in a method of treating SARS-CoV-2 infection by administering to a subject in need thereof an effective amount of the fragment, thereby reducing viral load, preventing disease progression, or mitigating symptoms.

### EXAMPLES

- introduce CeVICA platform
- describe CeVICA components
- explain ribosome display
- detail selection cycle
- describe computational clustering
- analyze natural VHH sequences
- design VHH DNA templates
- introduce randomization in CDRs
- describe PCR and ligation process
- detail CDR randomization hierarchy
- describe VHH DNA library construction
- test library performance in ribosome display
- reduce unproductive sequences
- enrich functional VHH sequences
- perform binder selection for RBD and EGFP
- immobilize target proteins
- perform selection rounds
- analyze output library sequences
- identify target-specific binders
- cluster CDR sequences
- validate RBD binders
- test SARS-CoV-2 pseudovirus neutralization
- compare input, output, and natural CDR sequence distributions
- assess fitness of binders
- design affinity maturation strategy
- introduce random mutations
- perform stringent selection
- identify putative beneficial mutations
- generate mutated variants
- assess binding and neutralization of variants
- compare performance to VHH72
- analyze correlation between neutralization and binding
- perform dose response curve
- determine IC50 of SR6c3
- examine VHH sequences' impact on immunogenicity
- identify conversion options for VHH hallmark residues
- demonstrate feasibility of converting VHH hallmark residues
- extend CeVICA for affinity maturation
- identify true binders and neutralizers across CeVICA predicted list
- clone and purify VHHs for ELISA and pseudovirus neutralization
- assay VHHs by ELISA and pseudovirus neutralization
- identify positive binders among tested VHHs
- engineer potent and stable VHHs for virus neutralization
- perform second affinity maturation using SR6c3 as template
- identify mutation combinations that enhance binding affinity
- compare binding affinity and pseudovirus neutralization of SR6v15 variants
- convert SR6v15 into tandemly linked dimer and trimer
- compare pseudovirus neutralization of SR6v15 based agents
- evaluate biophysical characteristics of CeVICA selected nanobodies
- perform size exclusion chromatography analysis of nanobodies
- investigate impact of cysteines in CDRs on nanobody biophysical properties
- analyze non-reducing SDS-PAGE gel of VHHs with 0-2 cysteines in CDRs
- evaluate functional consequences of CDR cysteine mediated dimer formation
- assess thermal stability of VHHs produced by CeVICA
- test VHHs' ability to refold after complete thermal denaturation
- discuss CeVICA platform's generalizable solution for in vitro VHH antibody engineering
- validate fully random NNB encoded codons in all CDR positions
- discuss binder sequence recovery using CDR-directed clustering
- apply CeVICA to engineer SARS-CoV-2 neutralizing VHHs
- identify SR38, a VHH with rare ability to strongly favor binding of N501Y containing RBD
- discuss previous synthetic nanobody library designs
- test fitness of randomized amino acid profile in binder selection
- discuss biophysical properties of VHHs produced by CeVICA
- conclude CeVICA's suitability for engineering high affinity VHH antibodies
- cell culture
- amino acid profile construction
- analyze natural VHHs
- calculate amino acid profile
- define CDR boundaries
- compare annotation methods
- measure diversity
- construct VHH library
- design library sequence
- randomize CDRs
- ligate PCR products
- purify ligation products
- quantify purified products
- randomize CDR2
- randomize CDR1
- randomize CDR3
- construct final VHH library
- prepare sequencing libraries
- perform high-throughput sequencing
- analyze sequencing data
- perform ribosome display
- prepare DNA template
- perform in vitro transcription and translation
- stop ribosome display reaction
- perform in vitro selection
- immobilize target proteins
- incubate with ribosome display solution
- extract and analyze selected RNA
- perform control experiment
- describe CDR-directed clustering analysis
- calculate CDR scores
- calculate mean distance to diagonal
- express and purify target proteins
- describe ELISA assay for VHH binding to RBD
- produce pseudotyped SARS-CoV-2 lentivirus
- perform lentiviral production for transductions
- describe SARS-CoV-2 S pseudotyped lentivirus neutralization assay
- perform affinity maturation
- identify and rank beneficial mutations
- describe Biolayer Interferometry
- perform size exclusion chromatography
- perform thermal stability assays
- describe protein expression and purification
- describe VHH purification
- describe ELISA assay for VHH binding to RBD
- describe pseudotyped lentivirus production
- describe lentiviral production for transductions
- describe SARS-CoV-2 S pseudotyped lentivirus neutralization assay
- perform error-prone PCR
- describe mutagenized library preparation
- perform ribosome display and in vitro selection
- build amino acid profile tables
- identify putative beneficial mutations
- rank beneficial mutations
- incorporate beneficial mutations into VHH parental sequences
- describe Biolayer Interferometry assay conditions
- analyze Biolayer Interferometry data
- describe size exclusion chromatography conditions
- analyze size exclusion chromatography data
- describe thermal stability assay conditions

The CeVICA platform was implemented to generate a diverse library of VHH domains with fully randomized CDRs using a three-stage PCR and ligation process. Natural VHH sequences were analyzed from the Protein Data Bank and abYsis to define consensus framework regions and CDR boundaries, with CDR1, CDR2, and CDR3 lengths set to 7, 5, and 13 amino acids, respectively, to reflect natural distributions. The library was constructed by first randomizing CDR2 using NNB codons, followed by CDR1, and finally CDR3, to preserve diversity hierarchy. DNA templates were amplified using Phusion polymerase at 65°C elongation temperature to prevent hairpin disruption, and ligation products were purified by agarose gel extraction and quantified by Qubit. The final library contained 3.68 × 10^11 full-length variants per microgram and was subjected to ribosome display using PURExpress in vitro transcription-translation system. Target proteins, including SARS-CoV-2 RBD and EGFP, were immobilized on anti-Flag-coated magnetic beads, and selection was performed over three rounds with increasing stringency. RNA from bound complexes was extracted, reverse-transcribed, and amplified for subsequent rounds. Sequencing of the output library revealed a 2.3-fold enrichment of full-length sequences and enabled CDR-directed clustering, identifying 862 unique clusters for RBD binders. Representative sequences from top clusters were synthesized and expressed in E. coli, purified via His-tag affinity chromatography, and validated by ELISA and pseudovirus neutralization assays. SR38, a VHH derived from a cluster of only five sequences, exhibited preferential binding to N501Y RBD and superior neutralization of N501Y pseudovirus compared to previously reported nanobodies. Affinity maturation was performed using error-prone PCR on SR6c3, followed by stringent ribosome display, yielding variants SR6v15, SR6v7, and SR6v9 with K_D values as low as 2.18 nM. SR6v15 was converted into a dimer and trimer, achieving IC50 values of 0.329 nM and 0.187 nM, respectively. Biophysical characterization demonstrated that CeVICA-derived VHHs exhibited high thermal stability (Tm = 72°C), excellent refolding capacity after denaturation, and monomeric behavior in size exclusion chromatography. Cysteine residues in CDRs did not consistently induce dimerization, and when they did, the dimers retained or enhanced neutralizing activity. The platform successfully identified 30 functional binders from 38 tested candidates, demonstrating high predictive power. The use of fully randomized NNB codons across all CDR positions did not impair binder fitness, as evidenced by high correlation between input and output amino acid profiles. The CeVICA platform thus provides a robust, scalable, and generalizable method for the rapid engineering of high-affinity, stable, and functional VHH antibodies with therapeutic and diagnostic utility.