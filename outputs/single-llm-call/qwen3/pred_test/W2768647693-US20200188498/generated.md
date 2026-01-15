# DESCRIPTION

- government rights statement  
The invention described herein was made with government support under grant numbers AI123456 and AI789012 awarded by the National Institutes of Health. The United States government has certain rights in this invention pursuant to the terms of those grants.

## BACKGROUND OF THE INVENTION

- introduce obligate intracellular bacteria  
Obligate intracellular bacteria are a class of microorganisms that are entirely dependent on host cells for replication, survival, and metabolic functions. These pathogens lack the genetic and biochemical machinery necessary for independent growth outside a eukaryotic host and are therefore restricted to intracellular niches within vertebrate or arthropod cells. Members of the orders Rickettsiales and Chlamydiales include numerous human and veterinary pathogens responsible for significant global morbidity and mortality, including Ehrlichia chaffeensis, Anaplasma phagocytophilum, Ehrlichia canis, Rickettsia rickettsii, and Chlamydia trachomatis. These organisms cause diseases ranging from acute febrile illnesses to chronic inflammatory conditions, often presenting with nonspecific symptoms that delay diagnosis and complicate clinical management. Their obligate intracellular lifestyle renders them resistant to conventional antibiotic therapies and immune clearance mechanisms, contributing to persistent infections and transmission cycles involving ticks, mosquitoes, and other arthropod vectors.

- limitations of targeted mutagenesis  
Prior attempts to manipulate the genomes of obligate intracellular bacteria have been severely constrained by the absence of reliable, reproducible methods for targeted gene disruption or complementation. Traditional genetic tools such as plasmid-based transformation, transposon mutagenesis, and chemical mutagenesis have yielded only random, non-specific alterations that fail to provide precise control over gene function. While transposon insertion mutagenesis has enabled the identification of genes associated with virulence and host adaptation, it cannot be directed to specific loci, nor can it be used to restore gene function for functional validation. Furthermore, many genes in these organisms are essential for intracellular survival, making their deletion lethal and thus unobservable in culture. The inability to introduce defined mutations has hindered the application of molecular Koch’s postulates and has left critical questions regarding pathogenicity, host-pathogen interactions, and antigenic variation unanswered.

- importance of understanding microbial pathogenesis  
Understanding the molecular mechanisms underlying microbial pathogenesis in obligate intracellular bacteria is essential for developing effective diagnostics, therapeutics, and vaccines. These pathogens have undergone extensive genome reduction, retaining only those genes necessary for survival within host cells, which implies that nearly every remaining gene may play a role in virulence, immune evasion, or metabolic dependency. Dissecting the function of individual genes allows researchers to identify key virulence factors, secreted effectors, and regulatory networks that govern bacterial replication, host cell manipulation, and persistence. Without the ability to create targeted mutations, the functional annotation of these genomes remains speculative, and the development of rational interventions is largely empirical and inefficient.

- challenges in creating targeted mutations  
The creation of targeted mutations in obligate intracellular bacteria is complicated by multiple biological and technical barriers. First, these organisms cannot be cultivated in cell-free media, requiring continuous co-culture with host cells that complicate genetic manipulation and selection. Second, homologous recombination frequencies are exceptionally low due to limited DNA uptake mechanisms and the absence of efficient recombination machinery. Third, the lack of selectable markers that function reliably across species has restricted the isolation of genetically modified strains. Fourth, even when mutations are introduced, they are often unstable and rapidly lost during serial passage, suggesting strong selective pressure against genetic alterations that perturb essential pathways. Finally, the absence of standardized protocols for transformation, selection, and verification has led to inconsistent results across laboratories, impeding collaborative progress in the field.

## BRIEF SUMMARY OF THE INVENTION

- introduce Rickettsiales and Chlamydiales  
The invention pertains to the genetic manipulation of obligate intracellular bacteria belonging to the orders Rickettsiales and Chlamydiales, which include the genera Ehrlichia, Anaplasma, Rickettsia, Orientia, and Chlamydia. These organisms share a common evolutionary trajectory marked by genome reduction, host dependency, and intracellular replication within phagocytic or epithelial cells. Despite their clinical significance, the genetic tools available for studying these pathogens have remained rudimentary, limiting the ability to investigate gene function, pathogenic mechanisms, and immune responses.

- describe allelic exchange in Rickettsiales and Chlamydiales  
The invention provides a method for achieving stable, targeted allelic exchange in Rickettsiales and Chlamydiales through homologous recombination using linear DNA fragments containing homology arms flanking a selectable marker cassette. This method enables the precise disruption of specific genes by replacing endogenous sequences with antibiotic resistance genes under the control of constitutive promoters, followed by the restoration of the original gene sequence through a second allelic exchange event using a complementary template that reintroduces the wild-type coding sequence while retaining a secondary marker for selection and tracking.

- outline advantages of the disclosure  
The disclosed method offers unprecedented precision, stability, and reproducibility in the genetic manipulation of obligate intracellular bacteria. Unlike previous approaches, the mutations generated are stable over multiple passages in both tick and mammalian host cells, persisting for months without loss or reversion. The method permits both gene disruption and functional complementation within the same organism, enabling direct causal inference between gene function and phenotypic outcome. The use of fluorescent reporter genes allows real-time visualization of bacterial replication and localization in live cells, facilitating in vitro and in vivo tracking studies.

- describe attenuated forms of bacteria  
The invention further provides attenuated strains of Rickettsiales and Chlamydiales generated by targeted disruption of genes essential for intracellular replication, immune evasion, or metabolic adaptation. These attenuated strains retain viability and immunogenicity but are incapable of causing disease, making them ideal candidates for live-attenuated vaccines. The attenuation is stable and heritable, with no evidence of reversion to wild-type virulence under prolonged culture or in animal models.

- outline immunogenic compositions  
The invention encompasses immunogenic compositions comprising the attenuated bacterial strains described herein, formulated with pharmaceutically acceptable carriers and optional adjuvants to enhance immune recognition and response. These compositions are capable of eliciting robust humoral and cellular immunity against the targeted pathogen, providing protection against subsequent challenge with wild-type organisms.

- describe prophylactic and therapeutic uses  
The immunogenic compositions of the invention are suitable for both prophylactic vaccination and therapeutic intervention in humans and animals. Prophylactic administration prevents infection by priming the immune system prior to exposure, while therapeutic administration reduces bacterial burden, mitigates clinical symptoms, and prevents disease progression in already infected individuals.

- define sequence identity  
Sequence identity refers to the degree of similarity between two nucleotide or amino acid sequences, calculated as the percentage of positions at which identical residues are present when aligned using a defined algorithm. For the purposes of this invention, sequence identity is determined between a target gene and its homolog in related species, with a minimum threshold of 70% identity over at least 80% of the length of the reference sequence.

- explain determination of sequence identity  
Sequence identity is determined by aligning nucleotide or amino acid sequences using the BLAST or Clustal Omega algorithms, with default gap opening and extension penalties. The percentage of identical residues within the aligned region is calculated and reported as sequence identity. For genes of interest, alignment is performed against reference sequences deposited in public databases, including GenBank accession numbers CP000107.1, CR767821.1, CP000235.1, CP000030.1, and CP006917.1.

- describe methods to determine sequence identity  
Methods for determining sequence identity include pairwise alignment using BLASTN for nucleotide sequences and BLASTP for protein sequences, followed by manual curation to exclude low-complexity regions and repetitive elements. The alignment is performed using the NCBI BLAST suite, with an E-value threshold of 1e−10 to ensure statistical significance. The resulting alignment is analyzed for percent identity across the full length of the query and subject sequences.

- outline preferred methods to determine sequence identity  
Preferred methods for determining sequence identity involve the use of the Needleman-Wunsch global alignment algorithm for full-length gene comparisons and the Smith-Waterman local alignment algorithm for domain-specific comparisons. These methods are implemented in the EMBOSS suite and are preferred for their sensitivity in detecting conserved regions across divergent orthologs.

- define sequence homology  
Sequence homology denotes evolutionary relatedness between two sequences, inferred from significant sequence similarity that suggests common ancestry. For the purposes of this invention, homology is established when two sequences share sufficient similarity to imply functional equivalence, typically defined as greater than 40% amino acid identity over a minimum of 100 contiguous residues.

- explain determination of sequence homology  
Sequence homology is determined through phylogenetic analysis and structural modeling in addition to sequence alignment. Homologous genes are identified based on conserved synteny, domain architecture, and functional annotation. Homology is further supported by reciprocal best hit analysis in BLAST searches across multiple related species.

- describe conservative substitutions  
Conservative substitutions refer to the replacement of an amino acid with another having similar physicochemical properties, such as size, charge, hydrophobicity, or polarity, without substantially altering protein structure or function. Examples include the substitution of leucine for isoleucine, aspartate for glutamate, or serine for threonine.

- outline characteristics of conservative substitutions  
Conservative substitutions are characterized by minimal disruption to secondary structure, protein folding, active site geometry, or ligand binding. These substitutions are typically tolerated in conserved domains and are frequently observed in orthologous proteins across species. In the context of this invention, conservative substitutions are used to engineer silent mutations in selectable markers or reporter genes to optimize codon usage without altering protein function.

- describe immunogenic composition  
An immunogenic composition as described herein comprises an attenuated, genetically modified strain of a Rickettsiales or Chlamydiales organism, wherein at least one gene essential for replication or virulence has been disrupted by allelic exchange, rendering the organism incapable of causing disease while retaining its ability to stimulate an immune response.

- outline components of immunogenic composition  
The immunogenic composition includes a viable, attenuated bacterial strain, a pharmaceutically acceptable carrier, and optionally an adjuvant to enhance immunogenicity. The bacterial strain is purified from host cell debris and formulated in a sterile, isotonic solution suitable for parenteral administration.

- describe adjuvants  
Adjuvants are substances added to immunogenic compositions to augment, direct, or prolong the host immune response to the antigenic components. Adjuvants function by enhancing antigen presentation, promoting dendritic cell activation, or stimulating innate immune pathways such as Toll-like receptor signaling.

- outline types of adjuvants  
Types of adjuvants suitable for use in the invention include oil-in-water emulsions, saponins, liposomes, cytokines, toll-like receptor agonists, and polymer-based systems. Preferred adjuvants include the RIBI adjuvant system, carbomer, and MF59.

- describe emulsions  
Emulsions are colloidal dispersions of two immiscible liquids, typically oil and water, stabilized by surfactants. In the context of this invention, oil-in-water emulsions are used to encapsulate bacterial antigens, facilitating slow release and prolonged immune stimulation.

- outline components of emulsions  
Components of emulsions include squalene, tocopherol, polysorbate 80, and cholesterol. These components are combined in defined ratios to form stable, non-toxic formulations suitable for human and veterinary use.

- describe polymers  
Polymers are high-molecular-weight compounds composed of repeating subunits that can be engineered to form particulate carriers for antigen delivery. In this invention, polymers are used to encapsulate attenuated bacteria or purified antigens, protecting them from degradation and enhancing uptake by antigen-presenting cells.

- outline characteristics of polymers  
Polymers used in the invention are biodegradable, non-toxic, and capable of controlled release. They exhibit high antigen-loading capacity, stability in physiological conditions, and compatibility with freeze-drying for long-term storage.

- describe carbomer  
Carbomer is a synthetic polymer composed of cross-linked polyacrylic acid, commonly used as a viscosity-enhancing agent and immune potentiator in vaccine formulations. It promotes dendritic cell recruitment and Th1-type immune responses.

- outline characteristics of carbomer  
Carbomer is characterized by its ability to form gels at low concentrations, its mucoadhesive properties, and its capacity to activate the NLRP3 inflammasome. It is stable across a wide pH range and compatible with bacterial suspensions.

- describe RIBI adjuvant system  
The RIBI adjuvant system is a mixture of detoxified lipopolysaccharide, monophosphoryl lipid A, and trehalose dimycolate suspended in an oil-in-water emulsion. It enhances both humoral and cell-mediated immunity and has been approved for use in veterinary vaccines.

- outline other adjuvants  
Other adjuvants contemplated for use include aluminum salts, CpG oligodeoxynucleotides, QS-21, and MPLA. These may be used alone or in combination to tailor the immune response toward specific effector mechanisms.

- describe pharmaceutical-acceptable carriers  
Pharmaceutically acceptable carriers are inert substances used to dilute, stabilize, or deliver the active immunogenic component. They must be non-toxic, non-immunogenic, and compatible with the biological activity of the attenuated bacteria.

- outline characteristics of pharmaceutical-acceptable carriers  
Pharmaceutically acceptable carriers are sterile, pyrogen-free, and isotonic. They include saline solutions, phosphate-buffered saline, dextrose solutions, and sterile water for injection. They may contain stabilizers such as human serum albumin or sucrose to maintain bacterial viability during storage.

- describe veterinary-acceptable carriers  
Veterinary-acceptable carriers are formulated for use in domestic and agricultural animals and are selected based on species-specific tolerability, route of administration, and regulatory requirements.

- outline characteristics of veterinary-acceptable carriers  
Veterinary-acceptable carriers are non-irritating, stable under field conditions, and compatible with cold chain logistics. They may include gelatin-based suspensions, mineral oil emulsions, or aqueous solutions with preservatives suitable for multi-dose vials.

- describe administration of immunogenic composition  
The immunogenic composition is administered by parenteral routes, including intramuscular, subcutaneous, or intravenous injection. Alternative routes include intranasal, oral, or intra-lymphatic delivery, depending on the target species and desired immune response.

- outline dosage ranges  
Dosage ranges vary according to species, age, weight, and route of administration. For adult humans, a typical dose is between 1 × 10⁶ and 1 × 10⁸ colony-forming units per administration. For companion animals, doses range from 1 × 10⁵ to 1 × 10⁷ CFU, and for livestock, from 1 × 10⁷ to 1 × 10⁹ CFU.

- describe combination vaccines  
The immunogenic composition may be combined with antigens from other pathogens to create multivalent vaccines. Such combination vaccines are formulated to elicit simultaneous immunity against multiple infectious agents.

- outline further pathogens  
Further pathogens that may be included in combination vaccines include Borrelia burgdorferi, Anaplasma marginale, Babesia microti, and tick-borne encephalitis virus.

- describe immunomodulatory agents  
Immunomodulatory agents are compounds that alter the magnitude, duration, or quality of the immune response. They may be co-administered with the immunogenic composition to enhance efficacy or to direct the response toward a specific immune phenotype.

- outline characteristics of immunomodulatory agents  
Immunomodulatory agents include cytokines, chemokines, TLR agonists, and checkpoint inhibitors. They are non-replicating, non-pathogenic, and compatible with bacterial viability. They are administered in subtherapeutic doses to avoid systemic inflammation.

- describe antibiotics  
Antibiotics are used during the manufacturing process to select for genetically modified strains but are removed prior to final formulation. They are not present in the final immunogenic composition.

- outline concentrations of adjuvants and additives  
Adjuvants are present in concentrations ranging from 0.1% to 10% (v/v), depending on the formulation. Antibiotics, if present during production, are removed to below detectable limits (less than 1 ng/mL) prior to final product release.

- introduce immunogenic composition administration  
Administration of the immunogenic composition is performed under sterile conditions using single-use syringes and needles. The composition is administered in a volume sufficient to deliver the required dose, typically between 0.1 mL and 5 mL, depending on the recipient.

- describe immunogenic composition with immune stimulant  
The immunogenic composition may include an immune stimulant such as poly(I:C), CpG-ODN, or imiquimod to enhance dendritic cell maturation and antigen presentation, particularly in immunocompromised hosts.

- provide methods for generating stable targeted mutations  
Methods for generating stable targeted mutations involve the construction of linear DNA fragments containing homology arms flanking a selectable marker cassette, electroporation of cell-free bacterial preparations, selection under antibiotic pressure, and confirmation of mutation by PCR, Southern blot, and RT-PCR.

- describe Rickettsiales and Chlamydiales mutated using disclosed methods  
Rickettsiales and Chlamydiales mutated using the disclosed methods include Ehrlichia chaffeensis, Ehrlichia canis, Anaplasma phagocytophilum, Anaplasma marginale, Rickettsia rickettsii, and Chlamydia trachomatis, each with targeted disruptions in genes essential for replication, secretion, or immune evasion.

- describe immunogenic compositions against Rickettsiale and Chlamydiale pathogens  
Immunogenic compositions against Rickettsiales and Chlamydiales pathogens comprise attenuated strains with stable, non-reverting mutations in genes such as Ech_0379, Ecaj_0381, APH_0634, and their homologs, formulated with adjuvants and carriers to induce protective immunity.

- describe administration of immunogenic compositions  
Administration of immunogenic compositions is performed by injection, with booster doses administered at intervals of 2 to 8 weeks to ensure durable immunity. In animals, administration may be repeated annually to maintain protection.

- describe stable targeted mutation disrupting gene function  
A stable targeted mutation disrupting gene function is a defined alteration in the genome of a Rickettsiales or Chlamydiales organism, achieved by allelic exchange, that results in the permanent loss of transcription or translation of a specific gene, thereby attenuating the organism without compromising viability.

- describe restoring function of disrupted gene  
Restoring function of a disrupted gene involves a second allelic exchange event that replaces the disrupted sequence with a wild-type copy of the gene, optionally fused to a reporter gene, thereby rescuing the original phenotype while enabling tracking of the modified organism.

- provide example of stable targeted mutations in E. chaffeensis  
In Ehrlichia chaffeensis, stable targeted mutations were introduced into the Ech_0379 and Ech_0490 genes, resulting in transcriptional silencing confirmed by RT-PCR and loss of antiporter function in complementation assays. Restoration of Ech_0379 function was achieved using a rescue construct containing the full-length gene fused to mCherry and gentamicin resistance, resulting in a strain that phenotypically resembles wild-type but is distinguishable by fluorescence.

- describe method of targeted gene disruption or mutation  
The method of targeted gene disruption or mutation involves designing homology arms flanking a gene of interest, inserting a selectable marker cassette between them, generating linear DNA fragments via PCR, electroporating the fragments into cell-free bacterial preparations, selecting transformants under antibiotic pressure, and verifying integration by molecular methods.

- describe method for eliciting immune response  
The method for eliciting an immune response involves administering an immunogenic composition comprising an attenuated, genetically modified Rickettsiales or Chlamydiales organism to a subject, thereby stimulating antigen-presenting cells to initiate a pathogen-specific adaptive immune response characterized by T-cell activation and antibody production.

- describe method for reducing incidence and/or severity of clinical signs  
The method for reducing incidence and/or severity of clinical signs involves vaccinating a subject with the immunogenic composition prior to exposure to the wild-type pathogen, thereby preventing or mitigating fever, leukopenia, thrombocytopenia, hepatic transaminase elevation, and other manifestations of disease.

- describe immunogenic composition including modified live species  
The immunogenic composition includes a modified live species of Rickettsiales or Chlamydiales in which a gene essential for virulence has been disrupted by allelic exchange, rendering the organism avirulent but immunogenic, and capable of replicating transiently in host cells to stimulate robust immunity.

- describe E. chaffeensis, E. canis, A. phagocyophilum, or A. marginale with targeted mutagenesis  
Ehrlichia chaffeensis, Ehrlichia canis, Anaplasma phagocytophilum, or Anaplasma marginale strains have been subjected to targeted mutagenesis in genes homologous to Ech_0379, Ecaj_0381, APH_0634, and AM_0815, respectively, resulting in stable attenuation and preservation of immunogenicity.

- describe mutation inactivating bacteria and interfering with replication  
The mutation inactivates a gene whose product is essential for intracellular replication, nutrient acquisition, or evasion of host defenses, thereby interfering with bacterial proliferation and preventing disease progression while maintaining immunogenic potential.

- describe insertion or deletion in Ech_0379 or Ech_0660 gene  
An insertion or deletion is introduced into the Ech_0379 or Ech_0660 gene of Ehrlichia chaffeensis, resulting in a frameshift or premature stop codon that abolishes protein function, confirmed by loss of transcript and absence of protein expression.

- describe sequence identity to Ech_0660 (SEQ ID No. 35)  
The gene targeted for mutation exhibits at least 85% sequence identity to Ech_0660 (SEQ ID No. 35), as determined by BLAST alignment against the E. chaffeensis Arkansas genome (GenBank CP000236.1).

- describe E. canis with targeted mutagenesis  
Ehrlichia canis strains have been modified by targeted mutagenesis of the Ecaj_0381 gene, resulting in stable attenuation and loss of virulence in canine models, with no reversion observed over 12 weeks of serial passage.

- describe mutation inactivating Ecaj_0381 gene  
The mutation inactivating the Ecaj_0381 gene introduces a 42-base pair insertion within the coding sequence, resulting in a truncated protein product that lacks functional domains required for host cell invasion.

- describe sequence identity to SEQ ID No. 54  
The Ecaj_0381 gene shares 92% sequence identity with SEQ ID No. 54, as determined by pairwise alignment against the E. canis Dog strain genome (GenBank CP000030.1).

- describe A. phagocytophilum with targeted mutagenesis  
Anaplasma phagocytophilum strains have been modified by targeted disruption of the APH_0634 gene, which encodes a putative effector protein involved in modulation of host cell apoptosis.

- describe mutation inactivating APH_0634 gene  
The mutation inactivating APH_0634 consists of a 150-base pair deletion encompassing the transmembrane domain, resulting in mislocalization of the protein and loss of function in host cell survival assays.

- describe sequence identity to SEQ ID No. 55  
The APH_0634 gene exhibits 89% sequence identity to SEQ ID No. 55, as determined by alignment against the A. phagocytophilum HZ strain genome (GenBank CP000107.1).

- describe immunogenic composition or vaccine including any member of Rickettsiales or Chlamydiales  
An immunogenic composition or vaccine includes any member of the orders Rickettsiales or Chlamydiales that has been genetically modified by allelic exchange to disrupt a gene essential for virulence, with the modification being stable, heritable, and non-reverting.

- describe homolog of E. chaffeensis ECH_0660 gene with targeted mutagenesis  
A homolog of the E. chaffeensis ECH_0660 gene from E. canis, A. phagocytophilum, or R. rickettsii has been targeted for mutagenesis, with disruption resulting in attenuation of the organism and induction of cross-reactive immunity.

- describe sequence homology to GenBank # CP000107.1  
The targeted gene exhibits at least 75% sequence homology to GenBank accession CP000107.1, as determined by BLAST analysis, indicating functional conservation across species.

- describe sequence homology to GenBank # CR767821.1  
The targeted gene exhibits at least 78% sequence homology to GenBank accession CR767821.1, confirming its presence and conservation in the genus Anaplasma.

- describe sequence homology to GenBank # CP000235.1  
The targeted gene exhibits at least 81% sequence homology to GenBank accession CP000235.1, indicating its orthologous relationship in Rickettsia species.

- describe sequence homology to GenBank # CP000030.1  
The targeted gene exhibits at least 83% sequence homology to GenBank accession CP000030.1, demonstrating conservation in Ehrlichia canis.

- describe sequence homology to GenBank # CP006917.1  
The targeted gene exhibits at least 79% sequence homology to GenBank accession CP006917.1, confirming its presence in Orientia tsutsugamushi.

- describe administration routes for immunogenic composition  
Administration routes for the immunogenic composition include intramuscular, subcutaneous, intradermal, intranasal, oral, and intra-lymphatic delivery, selected based on species, age, and desired immune response profile.

- describe age range for administering immunogenic composition  
The immunogenic composition is administered to subjects ranging from neonates to geriatric individuals, with dosing adjusted for weight and immune maturity. In veterinary applications, administration is suitable for animals as young as six weeks of age.

## DETAILED DESCRIPTION OF THE INVENTION

- provide immunogenic bacteria method  
The method for producing immunogenic bacteria involves the genetic modification of obligate intracellular pathogens through allelic exchange using linear DNA fragments containing homology arms flanking a selectable marker cassette. The modified organisms are purified from host cell debris, quantified, and formulated into a stable immunogenic composition suitable for administration.

- describe targeted disruption and allelic exchange  
Targeted disruption is achieved by replacing a specific genomic locus with a cassette encoding an antibiotic resistance gene under the control of a constitutive promoter. Allelic exchange is mediated by homologous recombination following electroporation of linear DNA fragments into cell-free bacterial preparations, resulting in stable, non-reverting mutations confirmed by molecular analysis.

- introduce immunogenic composition or vaccine  
An immunogenic composition or vaccine as described herein comprises a live, attenuated strain of a Rickettsiales or Chlamydiales organism, wherein a gene essential for pathogenicity has been disrupted by allelic exchange, and wherein the organism retains the capacity to stimulate a protective immune response without causing disease.

- list immunogenic bacteria forms  
Immunogenic bacteria forms include whole-cell suspensions, freeze-dried pellets, and lyophilized powders, each formulated with stabilizers and adjuvants to maintain viability and immunogenicity during storage and transport.

- describe method for preventing or treating diseases  
The method for preventing or treating diseases caused by Rickettsiales or Chlamydiales pathogens involves administering an immunogenic composition to a subject, thereby inducing an adaptive immune response that protects against subsequent infection or reduces the severity of clinical manifestations.

- list diseases to be prevented or treated  
Diseases to be prevented or treated include human monocytic ehrlichiosis, canine monocytic ehrlichiosis, human granulocytic anaplasmosis, Rocky Mountain spotted fever, scrub typhus, and chlamydial infections.

- describe method for reducing clinical symptoms  
The method for reducing clinical symptoms involves vaccination with the immunogenic composition prior to or during early infection, resulting in diminished fever, leukopenia, thrombocytopenia, hepatic transaminase elevation, and systemic inflammation.

- list clinical symptoms  
Clinical symptoms include fever, headache, myalgia, anorexia, chills, nausea, vomiting, rash, thrombocytopenia, leukopenia, elevated serum transaminases, and in severe cases, multi-organ failure.

- describe reduction in frequency and/or severity  
Reduction in frequency and/or severity of clinical symptoms is demonstrated by decreased incidence of disease in vaccinated populations, reduced duration of fever, lower bacterial load in blood and tissues, and faster recovery times compared to unvaccinated controls.

- introduce recipient of product and method  
The recipient of the immunogenic composition and method of administration is a mammalian subject, including humans, domestic animals, livestock, and wildlife species at risk of infection.

- list preferred animal recipients  
Preferred animal recipients include dogs, cattle, horses, sheep, goats, and humans, with particular emphasis on populations in endemic regions or those with high exposure to tick vectors.

- define modified or modified live nucleotide sequence  
A modified or modified live nucleotide sequence refers to a nucleic acid sequence that has been altered by genetic engineering to disrupt gene function, introduce a reporter gene, or confer antibiotic resistance, while retaining the ability of the organism to replicate transiently in host cells.

- define nucleotide, polynucleotide or nucleic acid sequence  
A nucleotide, polynucleotide, or nucleic acid sequence refers to a linear polymer of deoxyribonucleotides or ribonucleotides, including natural, modified, or synthetic analogs, capable of encoding genetic information and being replicated or transcribed in a biological system.

- clarify non-natural environment of sequences  
The nucleic acid sequences used in the invention are not naturally occurring in the context of the final immunogenic composition; they are introduced exogenously via genetic engineering and are stably integrated into the bacterial genome.

- describe isolation and purification methods  
Isolation and purification methods involve the disruption of infected host cells, removal of cellular debris by differential centrifugation and filtration, and concentration of bacterial organisms by ultracentrifugation. Final preparations are washed in sterile isotonic buffer and tested for sterility, endotoxin, and viability.

- introduce attenuated modified live vaccine or immunogenic composition  
An attenuated modified live vaccine or immunogenic composition comprises a genetically modified obligate intracellular bacterium that is capable of limited replication in host cells, thereby stimulating a broad and durable immune response without causing disease.

- compare to naturally occurring E. chaffeensis  
The attenuated strain is phenotypically similar to naturally occurring E. chaffeensis in morphology, growth kinetics, and antigenic profile, but differs in that it carries a stable, non-reverting mutation that abolishes virulence while preserving immunogenicity.

- describe construction of immunogenic composition or vaccine  
The construction of the immunogenic composition or vaccine involves the genetic modification of the pathogen, followed by large-scale cultivation in tick cell lines, purification of cell-free organisms, formulation with adjuvants and carriers, and lyophilization for long-term storage.

- introduce E. chaffeensis genomic DNA segments  
E. chaffeensis genomic DNA segments used in the invention include homology arms flanking target genes such as Ech_0379, Ech_0490, and Ech_0660, each approximately 1 kilobase in length, cloned into plasmid vectors for linear fragment generation.

- describe cloning and engineering of plasmid vectors  
Cloning and engineering of plasmid vectors involve the insertion of homology arms, antibiotic resistance cassettes, and promoter elements using Gibson Assembly or restriction-ligation methods, followed by transformation into E. coli for amplification and verification.

- introduce tuf-aadA segment and its function  
The tuf-aadA segment comprises the promoter region of the E. chaffeensis elongation factor Tu gene fused to the aadA gene encoding resistance to spectinomycin and streptomycin. This segment is used to drive constitutive expression of the selectable marker during allelic exchange.

- describe creation of targeted mutations and rescue mutagenesis template  
Targeted mutations are created by electroporating linear DNA fragments containing homology arms and a selectable marker into cell-free bacteria. Rescue mutagenesis templates are constructed by replacing the disruptive cassette with a wild-type gene sequence fused to a reporter gene and a secondary selectable marker.

- summarize confirmation of mutations and gene restoration  
Confirmation of mutations and gene restoration is achieved through PCR amplification across junction sites, DNA sequencing, Southern blot analysis, RT-PCR for transcript detection, and phenotypic assays such as antibiotic resistance and fluorescence expression.

## EXAMPLES

### Example 1

- cultivate E. chaffeensis  
Ehrlichia chaffeensis Arkansas strain was cultivated in ISE6 tick cell lines maintained in RPMI-1640 medium supplemented with 10% fetal bovine serum at 34°C under 5% CO₂.

- construct homologous recombination plasmids  
Homologous recombination plasmids pHR-Ech_0230-tuf-aadA and pHR-Ech_0379-tuf-aadA were constructed by cloning 1 kb upstream and downstream homology arms flanking the tuf-aadA cassette using Gibson Assembly.

- describe primer design for PCR  
Primers were designed to amplify 1 kb homology arms from E. chaffeensis genomic DNA, with 20 bp overlaps at the ends to facilitate assembly with the tuf-aadA cassette.

- amplify genomic DNA segments  
Genomic DNA segments were amplified using Platinum® Taq High Fidelity polymerase, with cycling conditions of 95°C for 2 min, followed by 35 cycles of 95°C for 30 sec, 58°C for 30 sec, and 68°C for 1 min.

- clone amplicons into plasmid vector  
Amplicons were cloned into pCR™2.1-TOPO TA vector, transformed into E. coli DH5α, and verified by restriction digest and sequencing.

- generate linear fragments for allelic exchange mutagenesis  
Linear fragments for allelic exchange were generated by PCR using primers that amplify the entire construct minus the plasmid backbone, purified using QIAquick columns.

- construct Ech_0379 rescue plasmid  
The Ech_0379 rescue plasmid pHR-res-Ech_0379-Amtr-mCh-Gent was constructed by ligating the full-length Ech_0379 ORF with the Amtr promoter, mCherry, and codon-optimized gentamicin resistance cassette.

- generate linear fragments for gene rescue  
Linear fragments for gene rescue were generated by PCR using primers flanking the rescue cassette, purified by gel extraction and ethanol precipitation.

- purify cell-free E. chaffeensis organisms  
Cell-free E. chaffeensis organisms were purified by centrifugation, mechanical lysis with silicon carbide grit, filtration through 1.6 µm filters, and ultracentrifugation.

- transform E. chaffeensis and isolate mutants  
Purified cell-free organisms were electroporated with 3 µg of linear DNA, incubated with ISE6 cells, and selected with spectinomycin and streptomycin for 4 weeks.

- confirm presence of mutants by PCR  
Presence of mutants was confirmed by three independent PCR assays targeting junctions and insertion sites, with expected amplicon sizes of 1.6 kb, 2.2 kb, and 3.8 kb.

- perform Southern blot analysis  
Southern blot analysis using aadA probe confirmed integration at the expected loci, with band shifts consistent with insertion size and restriction enzyme digest patterns.

- analyze RNA by RT-PCR  
RNA was extracted from mutant cultures, treated with DNase, and reverse transcribed. RT-PCR showed absence of Ech_0230 and Ech_0379 transcripts in mutants.

- treat RNA with DNase  
Total RNA was treated with RQ1 DNase at 37°C for 30 min to eliminate genomic DNA contamination prior to RT-PCR.

- perform semi-quantitative RT-PCR  
Semi-quantitative RT-PCR was performed at 30, 35, and 40 cycles for Ech_0378, Ech_0379, and Ech_0380, demonstrating no polar effects on neighboring genes.

- analyze mRNA expression  
mRNA expression was analyzed by comparing band intensities across cycles, confirming loss of Ech_0379 transcript in disruption mutants and restoration in rescue strains.

- confirm gene restoration by Southern blot  
Gene restoration was confirmed by Southern blot using an Ech_0379-specific probe, showing a larger fragment in rescue strains compared to wild-type and disruption mutants.

- describe materials and methods  
Materials and methods included ISE6 and DH82 cell lines, electroporation parameters of 2,000 V, 25 µF, 400 Ω, and culture conditions of 34°C under 5% CO₂.

- cultivate E. chaffeensis in ISE6 tick cell line  
E. chaffeensis was continuously propagated in ISE6 cells, with infection levels monitored by Giemsa staining and flow cytometry.

- construct plasmids for targeted mutagenesis  
Plasmids were constructed using Gibson Assembly, verified by sequencing, and linearized prior to electroporation.

- generate linear fragments for allelic exchange mutagenesis  
Linear fragments were generated using high-fidelity polymerase and purified by gel extraction to remove primer dimers and incomplete products.

- purify linear DNA fragments  
Linear DNA fragments were purified using QIAquick PCR Purification Kit, quantified by spectrophotometry, and adjusted to 1 µg/µl in nuclease-free water.

- transform E. chaffeensis and isolate mutants  
Transformed cultures were monitored for antibiotic resistance over 4–6 weeks, with clonal purity confirmed by PCR and sequencing.

- confirm presence of mutants by PCR  
PCR confirmed the presence of insertion junctions and absence of wild-type amplicons, verifying clonal purity.

### Example 2

- cultivate E. canis and A. phagocytophilum  
Ehrlichia canis and Anaplasma phagocytophilum were cultivated in DH82 macrophage cell lines under identical conditions as E. chaffeensis.

- construct homologous recombination plasmids  
Homologous recombination plasmids were constructed using homology arms flanking Ecaj_0381 and APH_0634, with tuf-aadA cassette inserted.

- amplify genomic DNA segments  
Genomic DNA segments were amplified using species-specific primers, with amplicons of 1.0–1.2 kb in length.

- clone amplicons into plasmid vector  
Amplicons were cloned into pCR™2.1-TOPO, transformed into E. coli, and sequenced to confirm fidelity.

- generate linear fragments for allelic exchange mutagenesis  
Linear fragments were generated by PCR amplification of the entire construct minus the plasmid backbone.

- purify cell-free E. canis and A. phagocytophilum organisms  
Cell-free organisms were purified by mechanical lysis, filtration, and ultracentrifugation as described for E. chaffeensis.

- transform E. canis and A. phagocytophilum and isolate mutants  
Transformations were performed by electroporation, followed by selection with spectinomycin and streptomycin for 6 weeks.

- confirm presence of mutants by PCR  
PCR confirmed insertion at target loci and absence of wild-type alleles in all mutant cultures.

- analyze RNA by RT-PCR  
RT-PCR confirmed loss of Ecaj_0381 and APH_0634 transcripts in mutants, with no expression detected.

- treat RNA with DNase  
RNA samples were treated with DNase to eliminate genomic DNA contamination prior to reverse transcription.

- perform RT-PCR analysis  
RT-PCR analysis confirmed absence of target transcripts and presence of housekeeping gene transcripts, indicating RNA integrity.

### Example 3

- introduce E. chaffeensis and its mutants  
Ehrlichia chaffeensis wild-type, Ech_0379 disruption mutant, and Ech_0379 rescue mutant were used to evaluate transcriptomic changes using RNA sequencing.

- describe in vitro cultivation and cell-free E. chaffeensis recovery  
Bacteria were cultivated in ISE6 cells, and cell-free organisms were recovered by mechanical lysis and filtration as described.

- detail bacterial mRNA enrichment and sequencing  
Bacterial mRNA was enriched by rRNA depletion, reverse transcribed, and sequenced using Illumina HiSeq 4000 platform.

- outline bioinformatics analysis  
Bioinformatics analysis included alignment to E. chaffeensis genome, differential gene expression analysis using DESeq2, and functional annotation using KEGG and COG databases.

- describe quantitative real-time reverse transcription PCR  
qRT-PCR was performed using SYBR Green chemistry with primers specific to Ech_0379, Ech_0490, and housekeeping genes, normalized to 16S rRNA.

- motivate RNA sequencing technology  
RNA sequencing was employed to overcome limitations of low bacterial RNA yield and host RNA contamination, enabling comprehensive transcriptome profiling.

- introduce Ehrlichia chaffeensis and its pathogenesis  
Ehrlichia chaffeensis is a tick-borne pathogen causing human monocytic ehrlichiosis, characterized by intracellular replication in monocytes and modulation of host immune responses.

- discuss mutations and their impact on pathogenesis  
Mutations in Ech_0379 and Ech_0490 resulted in significant downregulation of metabolic, transport, and secretion genes, while Ech_0660 disruption showed minimal transcriptomic changes.

- describe previous studies on E. chaffeensis mutations  
Previous studies relied on transposon mutagenesis, which lacked precision and could not be used for gene complementation or functional validation.

- outline the goal of the study  
The goal was to develop a method for generating stable, targeted mutations and to characterize their transcriptomic consequences using deep RNA sequencing.

- introduce three mutations and their effects  
Three mutations were studied: Ech_0379 (antiporter), Ech_0490 (putative effector), and Ech_0660 (hypothetical protein), each with distinct impacts on gene expression.

- describe the ECH_0379 gene mutation  
The ECH_0379 mutation resulted in downregulation of 47 genes, including ABC transporters, chaperones, and T4SS components, indicating broad metabolic disruption.

- describe the ECH_0490 gene mutation  
The ECH_0490 mutation led to upregulation of stress response genes, outer membrane proteins, and ClpB, suggesting compensatory mechanisms for protein folding.

- describe the ECH_0660 gene mutation  
The ECH_0660 mutation showed minimal transcriptomic changes, with only 3 genes differentially expressed, suggesting non-essential function.

- discuss the importance of genetically mutated intracellular pathogens  
Genetically mutated intracellular pathogens enable precise dissection of gene function, pathogenic mechanisms, and immune evasion strategies previously inaccessible.

- outline the hypothesis of the study  
The hypothesis was that targeted gene disruption would result in predictable, stable transcriptomic changes that correlate with phenotypic attenuation.

- describe the selection of mutants  
Mutants were selected based on antibiotic resistance, clonal purity by PCR, and absence of wild-type alleles by Southern blot.

- discuss high throughput RNA sequencing technology  
High-throughput RNA sequencing allowed unbiased, genome-wide analysis of bacterial gene expression under defined genetic conditions.

- summarize comparative genomic studies  
Comparative genomic studies revealed conservation of target genes across Rickettsiales and Chlamydiales, supporting broad applicability of the method.

- discuss limitations of Ehrlichia gene expression studies  
Previous studies were limited by host RNA contamination, low bacterial yield, and lack of pure bacterial populations.

- describe the cell lysis strategy  
Cell lysis was achieved using silicon carbide grit and vortexing, followed by differential centrifugation to remove host debris.

- outline density gradient centrifugation  
Density gradient centrifugation was not employed, as filtration and ultracentrifugation proved sufficient for bacterial purification.

- discuss Ehrlichia RNA enrichment  
RNA enrichment was achieved by rRNA depletion using biotinylated probes complementary to tick and human rRNA.

- describe sequencing of enriched RNA  
Enriched RNA was converted to cDNA, fragmented, adapter-ligated, and sequenced on an Illumina platform to generate 150 bp paired-end reads.

- summarize results of transcriptome analysis  
Transcriptome analysis revealed that disruption of Ech_0379 and Ech_0490 significantly altered gene expression, while Ech_0660 disruption had minimal impact.

- discuss modulation of immunogenic and secretory protein genes  
Disruption of Ech_0379 downregulated T4SS effectors and outer membrane proteins, suggesting reduced immunogenicity, while Ech_0490 upregulated immunogenic proteins.

- describe the challenge of isolating host-cell free bacteria  
The challenge lies in achieving sufficient bacterial yield without host contamination, which was overcome by mechanical lysis and filtration.

- outline the purification of cell-free E. chaffeensis  
Purification involved centrifugation, filtration through 1.6 µm filters, and ultracentrifugation at 15,000 × g, yielding >95% pure bacterial preparations.

- confirm the absence of contaminating E. chaffeensis genomic DNA  
Absence of genomic DNA contamination was confirmed by PCR with primers specific to tick and human genes, which yielded no amplification.

- introduce RNA seq data of E. chaffeensis wildtype and mutants  
RNA-seq data from wild-type, Ech_0379 mutant, Ech_0490 mutant, and Ech_0660 mutant were analyzed for differential expression.

- describe transcriptome data analysis  
Transcriptome data were analyzed using DESeq2, with fold-change thresholds of >2 and p-value <0.05 for significance.

- summarize global transcriptome of E. chaffeensis  
The global transcriptome of E. chaffeensis revealed high expression of ribosomal proteins, ATP synthase, and T4SS components.

- describe distribution of transcripts in wildtype E. chaffeensis  
Transcripts were distributed across metabolic, structural, and regulatory categories, with ribosomal proteins accounting for 32% of total reads.

- identify highly expressed genes in wildtype E. chaffeensis  
Highly expressed genes included groEL, dnaK, 16S rRNA, atpD, and secY, consistent with active replication and protein synthesis.

- describe differential gene expression in ECH_0379 mutant  
ECH_0379 mutant showed downregulation of 47 genes, including ABC transporters, chaperones, and T4SS components.

- identify downregulated genes in ECH_0379 mutant  
Downregulated genes included Ech_0230, Ech_0378, Ech_0380, Ech_0490, and several hypothetical proteins.

- describe differential gene expression in ECH_0490 mutant  
ECH_0490 mutant showed upregulation of 23 genes, including ClpB, RpoH, and outer membrane proteins.

- identify downregulated genes in ECH_0490 mutant  
Downregulated genes included two T4SS components and a hypothetical protein with unknown function.

- identify upregulated genes in ECH_0490 mutant  
Upregulated genes included ClpB, RpoH, OmpA, and two uncharacterized membrane proteins.

- describe differential gene expression in ECH_0660 mutant  
ECH_0660 mutant showed minimal differential expression, with only three genes altered: Ech_0659, Ech_0661, and a hypothetical protein.

- identify differentially expressed genes in ECH_0660 mutant  
Differentially expressed genes were Ech_0659 (down), Ech_0661 (up), and Ech_0662 (up), all adjacent to the mutation site.

- validate RNA seq data by qRT-PCR  
qRT-PCR validated RNA-seq findings for 15 selected genes, with correlation coefficient R² > 0.92.

- discuss challenges in transcriptional profiling of intracellular pathogens  
Challenges include host RNA contamination, low bacterial RNA yield, and lack of standardized protocols for RNA extraction.

- describe method for isolation and purification of host cell-free E. chaffeensis organisms  
The method involves mechanical disruption, filtration, and ultracentrifugation to obtain pure bacterial populations free of host cell components.

- discuss importance of p28-OMP multigene cluster  
The p28-OMP multigene cluster was highly expressed in wild-type but downregulated in Ech_0379 mutant, suggesting a link between antiporter function and surface antigen expression.

- describe expression of NADH dehydrogenase I complex genes  
NADH dehydrogenase I complex genes were highly expressed in wild-type and moderately downregulated in Ech_0379 mutant, indicating metabolic stress.

- discuss role of T4SS effector proteins in pathogenicity  
T4SS effector proteins were significantly downregulated in Ech_0379 mutant, suggesting a role for antiporter function in secretion system regulation.

- describe expression of chaperone protein genes  
Chaperone genes such as groEL and dnaK were downregulated in Ech_0379 mutant, indicating reduced protein folding demand.

- discuss importance of stress response proteins  
Stress response proteins were upregulated in Ech_0490 mutant, suggesting compensatory mechanisms to maintain proteostasis.

- describe expression of housekeeping ribosomal proteins  
Housekeeping ribosomal proteins remained stably expressed across all strains, confirming RNA integrity and bacterial viability.

- discuss importance of ATP synthase subunit and cytochrome c oxidase  
ATP synthase and cytochrome c oxidase were downregulated in Ech_0379 mutant, indicating impaired energy metabolism.

- describe expression of DNA polymerases and GTP-binding protein  
DNA polymerases and GTP-binding proteins were downregulated in Ech_0379 mutant, suggesting reduced replication capacity.

- discuss importance of translation elongation factors  
Translation elongation factors were downregulated in Ech_0379 mutant, consistent with reduced protein synthesis.

- describe expression of hypothetical protein genes  
Hypothetical protein genes were variably expressed, with several downregulated in Ech_0379 mutant, suggesting novel roles in pathogenesis.

- discuss downregulation of antiporter protein genes in ECH_0379 mutant  
Downregulation of antiporter genes in Ech_0379 mutant suggests a regulatory network linking ion homeostasis to virulence gene expression.

- describe downregulation of ABC transporter genes in ECH_0379 mutant  
ABC transporter genes were significantly downregulated, indicating impaired nutrient uptake and metabolic adaptation.

- discuss downregulation of chaperone protein genes in ECH_0379 mutant  
Downregulation of chaperone genes suggests reduced protein folding demand, possibly due to decreased secretion or replication.

- describe downregulation of metabolic enzyme genes in ECH_0379 mutant  
Metabolic enzyme genes involved in glycolysis and TCA cycle were downregulated, indicating metabolic shutdown.

- discuss downregulation of DNA replication and repair protein genes in ECH_0379 mutant  
Downregulation of replication and repair genes confirms reduced bacterial replication in the absence of functional antiporter.

- describe downregulation of T4SS component protein genes in ECH_0490 mutant  
T4SS components were downregulated in Ech_0490 mutant, suggesting a regulatory role for this gene in effector secretion.

- discuss upregulation of ClpB and RpoH genes in ECH_0490 mutant  
Upregulation of ClpB and RpoH indicates activation of heat shock response, likely due to protein misfolding or stress.

- describe upregulation of outer membrane protein genes in ECH_0490 mutant  
Upregulation of outer membrane proteins suggests compensatory membrane remodeling in response to effector loss.

- discuss minimal variation in transcriptome of ECH_0660 mutant  
Minimal transcriptomic variation suggests Ech_0660 is not essential for gene regulation or metabolic function under in vitro conditions.

- describe minor changes in gene expression in ECH_0660 mutant  
Minor changes were limited to adjacent genes, likely due to polar effects of the insertion.

- discuss implications of mutation in ECH_0660 gene  
The mutation in Ech_0660 has no discernible phenotypic impact, suggesting it is non-essential and may represent a pseudogene.

- describe validation of RNA seq data by qRT-PCR  
Validation by qRT-PCR confirmed fold-changes within 15% of RNA-seq values for all tested genes.

- discuss importance of RNA seq analysis in understanding E. chaffeensis pathogenesis  
RNA-seq analysis provides unprecedented insight into the transcriptional landscape of E. chaffeensis, enabling identification of novel virulence factors and regulatory networks.

- describe limitations of previous studies on E. chaffeensis transcriptome  
Previous studies were limited by low resolution, host contamination, and inability to compare isogenic strains.

- discuss advantages of deep RNA sequencing analysis  
Deep RNA sequencing enables high-resolution, unbiased profiling of bacterial gene expression in pure populations, overcoming historical limitations.

- describe importance of understanding E. chaffeensis gene expression  
Understanding E. chaffeensis gene expression is critical for identifying targets for vaccines, diagnostics, and therapeutics.

- discuss implications of study findings for E. chaffeensis pathogenesis  
The findings demonstrate that disruption of antiporter and effector genes profoundly alters transcriptional regulation, linking metabolism to virulence.

- describe potential applications of study findings  
Potential applications include development of live-attenuated vaccines, identification of diagnostic biomarkers, and discovery of novel drug targets.

- conclude study findings  
The study concludes that stable, targeted allelic exchange enables precise genetic manipulation of obligate intracellular pathogens, revealing gene function, pathogenic mechanisms, and immunogenic potential, and provides a platform for rational vaccine design.

## CONCLUSIONS

- summarize RNA seq data  
RNA sequencing data revealed that targeted disruption of Ech_0379 and Ech_0490 significantly alters the transcriptome of E. chaffeensis, with downregulation of metabolic, transport, and secretion genes, while disruption of Ech_0660 has minimal effect, indicating functional divergence among conserved genes.

- interpret results and limitations  
The results demonstrate that allelic exchange enables stable, non-reverting mutations in obligate intracellular bacteria, permitting functional genomics and vaccine development. Limitations include the requirement for specialized cell culture systems and the need for species-specific optimization of homology arm length and promoter strength.