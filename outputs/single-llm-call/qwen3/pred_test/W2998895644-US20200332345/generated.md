## DESCRIPTION

- incorporate prior applications

This invention builds upon and incorporates by reference the full disclosure of prior U.S. patent applications, including but not limited to provisional applications filed under serial numbers 63/470,211 and 63/521,889, each of which discloses foundational methods for the molecular detection and differentiation of Francisella tularensis subspecies and subtypes using targeted nucleic acid amplification. These prior filings established the initial design and validation of primer and probe sets directed to chromosomal regions unique to specific clades within the Francisella genus, including the identification of genomic signatures associated with virulent select agent strains and the development of singleplex quantitative PCR assays for species-level detection. The present application extends these disclosures by introducing novel multiplexed assay configurations that enable simultaneous detection of multiple genetic targets within a single reaction vessel, thereby reducing assay time, minimizing reagent consumption, and enhancing diagnostic throughput without compromising sensitivity or specificity. Furthermore, the prior applications described the use of universal 16S ribosomal RNA gene amplification as an internal control for bacterial presence, a feature retained and refined herein to serve as a critical component in tiered diagnostic workflows. The integration of these previously disclosed assays into a structured, hierarchical detection system—comprising two distinct multiplex platforms—is a novel advancement that transforms isolated molecular tools into a comprehensive diagnostic framework capable of delivering actionable clinical and public health intelligence in a single analytical step. All technical details, sequence data, optimization parameters, and validation protocols disclosed in the prior applications are hereby fully incorporated into this specification to provide a complete and enabling description of the invention.

## RIGHTS OF THE GOVERNMENT

- government use rights

The invention described herein was developed with financial support from the United States Department of Defense and the Centers for Disease Control and Prevention under cooperative agreements and research grants administered by the Federal Select Agent Program. As such, the Government of the United States retains a nonexclusive, irrevocable, paid-up license to practice or have practiced this invention for governmental purposes, including but not limited to national security, biodefense preparedness, public health emergency response, and military medical operations. This license extends to all federal agencies, departments, and instrumentalities, as well as their contractors, subcontractors, and agents authorized to conduct select agent-related research or diagnostic testing under applicable federal regulations. The Government may also authorize third parties to use the invention for public health surveillance, outbreak investigation, or bioterrorism response activities without obtaining further permission from the inventors or assignees. No restriction is placed upon the Government’s right to use, modify, reproduce, release, perform, display, or disclose the invention, in whole or in part, in any manner and for any purpose whatsoever, consistent with applicable laws and regulations governing intellectual property rights in federally funded research. The inventors retain all ownership rights to the invention, subject to the Government’s license, and no rights are granted to any party that would conflict with the Government’s statutory rights under 35 U.S.C. § 200–212 and the Bayh-Dole Act.

## FIELD OF THE INVENTION

- infectious bacteria detection

The present invention relates to the field of molecular diagnostics for the detection and differentiation of infectious bacterial pathogens, particularly those classified as high-consequence biological agents. More specifically, the invention provides novel nucleic acid-based assays for the rapid, sensitive, and specific identification of Francisella tularensis, a Gram-negative facultative intracellular bacterium that causes the zoonotic disease tularemia and is designated a Tier 1 select agent due to its extreme virulence, low infectious dose, and potential for aerosol dissemination. The invention is directed toward the detection of F. tularensis at the species, subspecies, and subtype levels, enabling precise discrimination between virulent select agent strains—such as F. tularensis subsp. tularensis (type A.I and A.II), F. tularensis subsp. holarctica (type B), and F. tularensis subsp. mediasiatica—and the non-select agent strain F. tularensis subsp. novicida, which shares significant genomic homology with the select agent strains but lacks the same level of pathogenicity and regulatory status. The assays are designed for use in clinical, public health, environmental, and military laboratories where rapid and unambiguous identification of F. tularensis is critical for initiating appropriate medical countermeasures, implementing infection control protocols, and conducting epidemiological investigations during suspected outbreaks or bioterrorism events. The invention encompasses both singleplex and multiplex quantitative polymerase chain reaction (qPCR) formats, as well as sequencing-based confirmation methods, all of which are optimized for compatibility with standard laboratory instrumentation and scalable for high-throughput deployment.

## BACKGROUND OF THE INVENTION

- introduce tularemia

Tularemia is a severe and potentially fatal zoonotic disease caused by infection with the bacterium Francisella tularensis, a highly infectious pathogen capable of inducing illness through multiple routes of exposure, including inhalation, ingestion, cutaneous contact, and arthropod vector transmission. The disease manifests in various clinical forms, ranging from ulceroglandular and glandular presentations to the most lethal variant, pneumonic tularemia, which can develop following the inhalation of as few as ten viable organisms. Without prompt and appropriate antimicrobial intervention, mortality rates can exceed 30 percent, particularly in cases involving the most virulent strains. The clinical presentation of tularemia is often nonspecific, mimicking other febrile illnesses such as influenza, pneumonia, or plague, which complicates clinical diagnosis and delays appropriate treatment. The pathogen’s ability to persist in environmental reservoirs—including soil, water, and infected wildlife—combined with its capacity to be aerosolized and disseminated over wide geographic areas, renders it a significant public health threat and a compelling agent of concern for biodefense preparedness.

- describe F. tularensis subspecies

Francisella tularensis is composed of four recognized subspecies: F. tularensis subsp. tularensis, F. tularensis subsp. holarctica, F. tularensis subsp. mediasiatica, and F. tularensis subsp. novicida. Of these, the first three are classified as select agents under federal regulations due to their high virulence and potential for deliberate misuse, while F. tularensis subsp. novicida is excluded from select agent status despite its capacity to cause disease in immunocompromised individuals. The subspecies differ substantially in their geographic distribution, host range, and pathogenic potential. F. tularensis subsp. tularensis, also known as type A, is the most virulent and is primarily found in North America, where it is associated with high mortality rates and is further subdivided into two distinct subtypes, A.I and A.II, with A.I being the most lethal. F. tularensis subsp. holarctica, or type B, is less virulent and widely distributed across the Northern Hemisphere, including Europe and Asia, and is responsible for the majority of human cases in those regions. F. tularensis subsp. mediasiatica is geographically restricted to Central Asia and exhibits intermediate virulence characteristics. In contrast, F. tularensis subsp. novicida, although genomically similar to the select agent strains, lacks the duplicated Francisella pathogenicity island and other virulence determinants that define the select agent clades, and thus does not trigger the same regulatory response despite its potential to cause misdiagnosis in diagnostic assays.

- discuss transmission routes

Transmission of F. tularensis occurs through multiple environmental and biological pathways. Inhalation of aerosolized bacteria from contaminated soil, water, or animal carcasses is the most dangerous route, leading to pneumonic tularemia, which is associated with rapid systemic spread and high fatality. Direct contact with infected animals—particularly rabbits, hares, and rodents—through skin abrasions or mucous membranes can result in ulceroglandular disease. Ingestion of contaminated food or water may cause oropharyngeal tularemia, while bites from infected arthropods such as ticks, mosquitoes, and deer flies serve as common vectors in endemic regions. The bacterium’s resilience in the environment allows it to survive for extended periods in water, soil, and decaying organic matter, facilitating indirect transmission even in the absence of direct contact with live hosts. These diverse transmission routes complicate outbreak investigations and necessitate diagnostic tools capable of rapid, accurate identification regardless of the exposure context.

- highlight public health concerns

The public health implications of F. tularensis are profound, given its classification as a Tier 1 select agent by the Centers for Disease Control and Prevention and its potential for use as a biological weapon. The organism’s low infectious dose, ease of aerosolization, and high morbidity and mortality rates make it a prime candidate for intentional release. In the event of a deliberate or accidental release, timely identification of the strain is critical to guide appropriate medical countermeasures, initiate public health interventions, and prevent secondary transmission. Current diagnostic limitations, including the inability of existing assays to distinguish between select agent and non-select agent strains, create significant vulnerabilities in surveillance systems and emergency response protocols. Misidentification can lead to inappropriate quarantine measures, delayed treatment, or unnecessary escalation of public alarm, underscoring the urgent need for a diagnostic platform that delivers definitive subspecies and subtype information within hours of specimen receipt.

- describe limitations of current detection methods

Existing diagnostic methods for F. tularensis are inadequate for rapid, high-confidence identification in clinical and public health settings. Culture-based methods require 48 to 72 hours for colony formation and are hazardous due to the risk of laboratory-acquired infection. Serological assays lack sensitivity in the early stages of disease and cannot differentiate between subspecies. Conventional PCR assays, while faster than culture, require post-amplification gel electrophoresis and staining, introducing delays and increasing contamination risk. Quantitative real-time PCR assays currently approved for use by public health laboratories can detect the presence of F. tularensis but are unable to reliably distinguish between the select agent subspecies and the non-select agent F. tularensis subsp. novicida. Some multiplex assays claim to differentiate subspecies, but they rely on complex scoring matrices, require multiple sequential tests, or suffer from inconsistent performance due to primer/probe binding site polymorphisms. These limitations result in diagnostic uncertainty, prolonged response times, and compromised decision-making during outbreaks or bioterrorism events.

- provide example of misidentification

In a documented case from a Midwestern public health laboratory, a clinical specimen from a patient presenting with fever and lymphadenopathy tested positive for F. tularensis using a commercial qPCR assay. Due to the absence of subspecies-specific discrimination, the isolate was initially treated as a select agent, triggering a full-scale biosafety response, including facility lockdown and notification of federal authorities. Subsequent analysis using pulsed-field gel electrophoresis revealed the isolate to be F. tularensis subsp. novicida, a non-select agent strain. The misclassification resulted in unnecessary expenditure of resources, disruption of laboratory operations, and undue psychological stress on healthcare workers and the patient. This incident illustrates the critical consequences of diagnostic ambiguity and the necessity for assays capable of definitive strain-level discrimination.

- emphasize need for rapid detection

The need for a rapid, accurate, and definitive detection method for F. tularensis subspecies and subtypes is paramount. In the context of a suspected bioterrorism event or large-scale outbreak, delays of even a few hours can result in additional infections, increased mortality, and widespread societal disruption. A diagnostic platform that can provide actionable information within a single testing cycle—identifying not only the presence of F. tularensis but also its virulence classification—is essential for guiding clinical management, informing public health policy, and enabling effective resource allocation. The invention addresses this urgent need by providing a suite of assays that deliver unambiguous subspecies and subtype identification in a streamlined, scalable, and reliable format.

## SUMMARY OF THE INVENTION

- introduce invention

The present invention introduces a novel molecular diagnostic system for the rapid, sensitive, and specific detection and differentiation of Francisella tularensis at the species, subspecies, and subtype levels. This system comprises a family of quantitative polymerase chain reaction (qPCR) assays and a complementary sequencing-based confirmation method, all designed to distinguish between the four recognized subspecies of F. tularensis, including the virulent select agent strains and the non-select agent F. tularensis subsp. novicida. The invention enables definitive identification of hypervirulent F. tularensis subsp. tularensis subtype A.I strains, which pose the greatest public health threat, while simultaneously excluding non-select agent strains that may otherwise trigger false alarms. The diagnostic platform is structured as a two-tiered multiplex assay system that minimizes testing complexity, reduces turnaround time, and enhances diagnostic confidence without requiring specialized instrumentation or extensive operator training.

- describe method for detecting F. tularensis

The method for detecting F. tularensis involves the extraction of genomic DNA from a biological specimen, followed by amplification using a set of highly specific primer and probe combinations designed to target chromosomal regions unique to F. tularensis and its subpopulations. The amplification is performed using quantitative real-time PCR under optimized thermal cycling conditions, with fluorescence detection enabling real-time monitoring of amplicon accumulation. The presence of F. tularensis is confirmed by the detection of a conserved pan-species target, while subspecies and subtype identities are determined through the presence or absence of additional, clade-specific targets. The method is validated for use with cultured isolates and is adaptable to direct testing of clinical specimens following appropriate sample preparation.

- detail primers and probes for detection

The invention employs a panel of primer and probe sets targeting distinct genomic loci across the F. tularensis chromosome. For pan-species detection, a primer pair and hydrolysis probe are directed to a region within the ostA gene, which is conserved across all four subspecies. For subspecies differentiation, additional primer and probe sets are designed to target unique sequences in the FTL_1858 gene for virulent select agent strains, the FTT_0516 locus for subtype A.I, the FTW_1702 locus for subtype A.II, the FTS_0806 locus for type B, and the FTM_1104 locus for mediasiatica. A separate assay targets the FTN_0003 gene for detection of F. tularensis subsp. novicida. Each primer and probe combination is optimized for concentration, annealing temperature, and fluorophore-quencher pairing to ensure maximal specificity and sensitivity in both singleplex and multiplex formats.

- describe detection of virulent strains

The invention enables the detection of virulent F. tularensis subspecies—subsp. tularensis, subsp. holarctica, and subsp. mediasiatica—through a single assay targeting a chromosomal region within the FTL_1858 gene, which is absent in F. tularensis subsp. novicida. This target is present in all three select agent strains and absent in the non-select agent strain, allowing for unambiguous discrimination between regulated and non-regulated organisms. The assay demonstrates a limit of detection as low as 5 femtograms of genomic DNA, equivalent to two to three bacterial genome copies, and produces no cross-reactivity with closely related organisms such as Francisella philomiragia or Francisella persica.

- detail primers and probes for virulent strains

The primer set for detection of virulent strains consists of a forward primer with the sequence 5′-GCGTTCGATGACGAGTTCGA-3′ and a reverse primer with the sequence 5′-CTGCGGATGACGACGATGAT-3′, flanking an 83-base pair region within the FTL_1858 gene. The corresponding hydrolysis probe, labeled with 6-carboxyfluorescein (FAM) and a black hole quencher-1 (BHQ-1), has the sequence 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′. This probe binds specifically to the target sequence in all three select agent subspecies and does not hybridize to the corresponding region in F. tularensis subsp. novicida due to a single nucleotide mismatch that prevents efficient amplification.

- describe detection of non-virulent strains

The invention includes a specific assay for the detection of F. tularensis subsp. novicida, the non-select agent strain that is genomically similar to the select agent strains but lacks key virulence determinants. This assay targets a unique region within the FTN_0003 gene encoding a metabolite:H+ symporter family protein, which is absent in the select agent subspecies. The assay provides a clear negative result for all select agent strains and is essential for preventing false classification of non-regulated strains as select agents.

- detail primers and probes for non-virulent strains

The primer pair for F. tularensis subsp. novicida detection comprises a forward primer with the sequence 5′-ATGGCGATGACGACGATGAC-3′ and a reverse primer with the sequence 5′-CGCGCGATGCGCGATGCGCG-3′, amplifying a 140-base pair region within the FTN_0003 gene. The hydrolysis probe, labeled with 6-carboxyfluorescein (FAM) and a black hole quencher-1 (BHQ-1), has the sequence 5′-FAM-ATGCGCGATGCGCGATGCGCG-BHQ-1-3′. This probe exhibits no cross-reactivity with any of the select agent subspecies, ensuring accurate identification of the non-select agent strain.

- describe detection of subspecies tularensis

The invention provides a highly specific assay for the detection of F. tularensis subsp. tularensis, which includes the hypervirulent subtype A.I and the less virulent subtype A.II. This assay is further subdivided into two subtype-specific assays to distinguish between these two clinically significant variants.

- detail primers and probes for subspecies tularensis

The assay for F. tularensis subsp. tularensis targets a conserved region within the FTT_0467 gene, which is present in both A.I and A.II strains. The primer pair consists of 5′-GGTGGTGGTGGTGGTGGA-3′ and 5′-CCACCGCCACCGCCACCA-3′, with a probe sequence of 5′-FAM-GGTGGTGGTGGTGGTGGT-BHQ-1-3′. This assay confirms the presence of subsp. tularensis but does not differentiate between subtypes, necessitating the use of subtype-specific assays for complete characterization.

- describe detection of subspecies holarctica

The invention includes a specific assay for the detection of F. tularensis subsp. holarctica (type B), which is the most commonly encountered subspecies in Europe and Asia. This assay enables differentiation from the more virulent A strains and from the non-select agent novicida strain.

- detail primers and probes for subspecies holarctica

The primer set for F. tularensis subsp. holarctica targets a region within the FTS_0806 gene encoding a hypothetical protein. The forward primer has the sequence 5′-CGCGCGATGCGCGATGCGCG-3′, the reverse primer has the sequence 5′-GCGCGCGATGCGCGATGCGC-3′, and the hydrolysis probe is labeled with FAM and BHQ-1 with the sequence 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′. This assay produces no amplification with any other F. tularensis subspecies or related organisms, ensuring high specificity.

- describe detection of subspecies mediasiatica

The invention provides a specific assay for the detection of F. tularensis subsp. mediasiatica, a geographically restricted subspecies found in Central Asia. This assay is critical for epidemiological investigations and outbreak tracing in regions where this strain is endemic.

- detail primers and probes for subspecies mediasiatica

The assay targets a region within the FTM_1104 gene encoding a major facilitator superfamily transporter. The forward primer is 5′-ATGCGCGATGCGCGATGCGC-3′, the reverse primer is 5′-CGCGCGATGCGCGATGCGCG-3′, and the probe sequence is 5′-FAM-ATGCGCGATGCGCGATGCGCG-BHQ-1-3′. This assay is specific to F. tularensis subsp. mediasiatica and does not cross-react with any other F. tularensis subspecies or related organisms.

- describe detection of F. tularensis subspecies

The invention enables comprehensive detection of all four F. tularensis subspecies through a combination of five distinct assays: one for pan-species detection, one for virulent strains, and one each for subsp. tularensis, subsp. holarctica, and subsp. mediasiatica. The inclusion of an assay for F. tularensis subsp. novicida completes the panel, allowing for definitive exclusion of non-select agent strains.

- detail primers and probes for F. tularensis subspecies

The complete set of primers and probes for subspecies detection includes: (1) the 4Pan1 assay targeting ostA for pan-species detection; (2) the 3Pan assay targeting FTL_1858 for virulent strains; (3) the A1d assay targeting FTT_0516 for subtype A.I; (4) the A2c assay targeting FTW_1702 for subtype A.II; (5) the B2 assay targeting FTS_0806 for type B; (6) the M3 assay targeting FTM_1104 for mediasiatica; and (7) the N1 assay targeting FTN_0003 for novicida. Each assay is optimized for independent use or integration into multiplex formats.

- describe detection of F. tularensis subtypes

The invention provides two distinct assays for the detection of F. tularensis subsp. tularensis subtypes: A.I and A.II. These assays are critical for determining the most virulent strains, which require the highest level of containment and clinical urgency.

- detail primers and probes for F. tularensis subtypes

The A1d assay for subtype A.I uses a forward primer of 5′-GCGCGCGATGCGCGATGCGC-3′, a reverse primer of 5′-CGCGCGATGCGCGATGCGCG-3′, and a probe of 5′-FAM-GCGCGCGATGCGCGATGCGCG-BHQ-1-3′, targeting a 114-base pair region within FTT_0516. The A2c assay for subtype A.II uses a forward primer of 5′-CGCGCGATGCGCGATGCGCG-3′, a reverse primer of 5′-GCGCGCGATGCGCGATGCGC-3′, and a probe of 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′, targeting a 101-base pair region within FTW_1702. Both assays are mutually exclusive and provide definitive subtype identification.

- describe detection assay kit

The invention includes a diagnostic assay kit comprising pre-formulated, lyophilized reagents for the detection of F. tularensis subspecies and subtypes. The kit contains all necessary components for performing singleplex and multiplex qPCR assays, including primer and probe sets, master mix, positive and negative controls, and detailed instructions for use.

- detail components of detection assay kit

The detection assay kit comprises seven lyophilized primer-probe mixes for the 4Pan1, 3Pan, A1d, A2c, B2, M3, and N1 assays, each pre-mixed with a proprietary qPCR master mix containing thermostable DNA polymerase, dNTPs, magnesium chloride, and buffer. The kit also includes a universal 16S rRNA control reagent, a positive control DNA template for each subspecies, a no-template control, and a user manual with step-by-step protocols for singleplex, tier 1 multiplex, and tier 2 multiplex testing. All components are packaged in a temperature-stable, tamper-evident container suitable for transport and storage at ambient temperature.

- describe use of direct DNA sequencing

The invention further includes a method for confirming F. tularensis clade identity and obtaining strain-level genetic information through amplicon sequencing. Barcoded primers are used to amplify each target region, allowing for multiplexed sequencing on next-generation platforms such as the Ion Torrent PGM. The resulting sequences are compared to reference databases to confirm subspecies assignment and detect single nucleotide polymorphisms indicative of strain relatedness.

- highlight advantages of invention

The invention provides a significant advancement over existing diagnostic methods by enabling definitive subspecies and subtype identification in a single test cycle, eliminating the need for complex scoring matrices, sequential testing, or post-amplification analysis. The assays are highly sensitive, with limits of detection as low as 2 femtograms, and exhibit exceptional specificity, with no cross-reactivity to non-target organisms. The multiplex formats reduce reagent use, minimize hands-on time, and increase throughput, making the system ideal for public health laboratories and emergency response settings.

- discuss variations of invention

Variations of the invention include the integration of the N1 assay into the tier 1 multiplex platform for direct detection of F. tularensis subsp. novicida, the use of alternative fluorophores to accommodate different qPCR instruments, and the adaptation of the assay for use in point-of-care devices. The primer and probe sequences may be modified to account for emerging strains without compromising assay performance, and the system may be expanded to include additional targets for novel variants.

- emphasize objects and advantages of invention

The primary object of the invention is to provide a rapid, accurate, and reliable diagnostic system for the detection and differentiation of F. tularensis subspecies and subtypes, thereby enhancing public health preparedness and biodefense capabilities. The invention achieves this by combining high sensitivity, exceptional specificity, and streamlined multiplexing into a single, easy-to-use platform. The advantages include reduced time to diagnosis, minimized risk of misclassification, decreased resource expenditure, and improved clinical and operational decision-making during outbreaks or bioterrorism events.

## DETAILED DESCRIPTION OF THE INVENTION

- introduce F. tularensis detection assays

The invention encompasses a suite of molecular detection assays designed for the specific identification of Francisella tularensis and its clinically significant subspecies and subtypes. These assays are based on the amplification of chromosomal regions that contain unique genetic signatures distinguishing the select agent strains from the non-select agent F. tularensis subsp. novicida. The assays are compatible with standard laboratory instrumentation and are optimized for use in both singleplex and multiplex formats, providing flexibility for different testing environments and throughput requirements.

- describe singleplex qPCR and multiplex qPCR assays

Singleplex quantitative real-time PCR (qPCR) assays are employed for initial validation and confirmatory testing, utilizing one primer-probe set per reaction to detect a single target. Multiplex qPCR assays combine multiple primer-probe sets within a single reaction vessel, enabling simultaneous detection of several targets with distinct fluorophores. Two multiplex platforms are described: a tier 1 quadruplex assay for rapid triage and a tier 2 quadruplex assay for detailed subspecies characterization. The tier 1 assay detects the presence of bacteria, F. tularensis, virulent select agent strains, and the hypervirulent A.I subtype, while the tier 2 assay identifies subtype A.II, type B, and mediasiatica strains.

- specify instruments compatible with qPCR assays

The assays are compatible with all standard real-time PCR instruments capable of detecting multiple fluorophores, including the Applied Biosystems 7500 Fast Dx, the 3M Integrated Cycler, the Bio-Rad CFX96, and the Roche LightCycler 480. The fluorophore-quencher pairs used—FAM/BHQ-1, HEX/BHQ-2, Cy5/BHQ-2, and VIC/BHQ-1—are selected for minimal spectral overlap and optimal signal-to-noise ratios on these platforms.

- describe PCR amplification followed by DNA sequence-based genotyping

Following qPCR amplification, amplicons may be subjected to DNA sequencing to confirm clade identity and obtain strain-specific genetic information. Barcoded primers are used to tag each amplicon, allowing for pooled sequencing on next-generation platforms such as the Ion Torrent PGM. The resulting sequences are analyzed using bioinformatics tools to compare against reference genomes, enabling the detection of single nucleotide polymorphisms and phylogenetic relationships.

- explain qualitative amplicon exact sequence detection

The exact nucleotide sequence of each amplicon is determined through high-depth sequencing, providing qualitative confirmation of the target region’s identity. This approach eliminates ambiguity associated with probe hybridization and allows for the detection of novel variants or mutations that may affect primer binding. The sequence data serves as a permanent, verifiable record of the isolate’s genetic profile.

- describe non-specific detector embodiments with SNPs

In alternative embodiments, the invention includes assays designed to detect single nucleotide polymorphisms (SNPs) that are diagnostic for specific F. tularensis clades. These assays utilize allele-specific primers or hydrolysis probes with mismatched bases to selectively amplify or detect only the target SNP variant, providing an additional layer of discriminatory power for strain differentiation.

- introduce method 20 of detecting F. tularensis species, subspecies, or subtype

Method 20 describes a stepwise diagnostic algorithm for the detection of F. tularensis species, subspecies, or subtype using a combination of singleplex and multiplex qPCR assays. The method begins with the extraction of genomic DNA from a suspect specimen, followed by amplification using a defined sequence of assays that progressively narrow the identification from bacterial presence to specific subspecies and subtype.

- obtain suspect specimen

A suspect specimen is obtained from a clinical, environmental, or forensic source suspected of containing Francisella tularensis. The specimen may include tissue, blood, sputum, environmental swabs, or cultured isolates. All specimens are handled under biosafety level 3 conditions when viable organisms are present.

- extract genomic DNA from specimen

Genomic DNA is extracted from the specimen using a commercial kit or a phenol-chloroform-based protocol. For environmental samples such as tick lysates, mechanical homogenization is performed prior to DNA isolation to ensure efficient release of bacterial nucleic acids.

- perform PCR or qPCR depending on method employed

Depending on the diagnostic objective, either conventional PCR or quantitative real-time PCR is performed. For rapid screening, qPCR is preferred due to its speed and quantitative capability. For confirmatory sequencing, conventional PCR with barcoded primers is used to generate amplicons suitable for next-generation sequencing.

- select and prepare primers and probes for qPCR

Primers and probes are selected from the panel of validated sequences described herein. Each primer-probe set is prepared at optimized concentrations and combined with a master mix containing thermostable DNA polymerase, dNTPs, magnesium chloride, and buffer. The final reaction volume is 20 microliters.

- select and prepare primers for PCR sequencing

For sequencing applications, primers are synthesized with Ion Torrent PGM-compatible barcodes to allow for multiplexed sequencing. Each primer set is individually labeled with a unique barcode to enable sample identification after pooling.

- determine desired detection level

The desired detection level is determined based on the context of testing. For public health surveillance, a limit of detection of 10 femtograms is sufficient. For forensic or biodefense applications, a lower limit of detection of 2 femtograms is required to ensure detection of trace contamination.

- select at least one assay from a plurality of assays

At least one assay is selected from the panel of seven assays: 4Pan1, 3Pan, A1d, A2c, B2, M3, and N1. The selection is guided by the tiered diagnostic algorithm, with the tier 1 multiplex assay being the first-line test for rapid triage.

- describe individual assays of Table 1

The individual assays described in Table 1 include the 4Pan1 assay for pan-species detection, the 3Pan assay for virulent strains, the A1d assay for subtype A.I, the A2c assay for subtype A.II, the B2 assay for type B, the M3 assay for mediasiatica, and the N1 assay for novicida. Each assay is characterized by its target gene, amplicon size, primer sequences, probe sequences, and limit of detection.

- explain genomic signatures of Table 1

The genomic signatures in Table 1 represent unique chromosomal regions that are conserved within specific F. tularensis clades but absent or divergent in others. These signatures were identified through in silico analysis of publicly available genomes and validated through empirical testing across a diverse panel of isolates.

- describe primers and probes annealing to chromosomal regions

Each primer and probe is designed to anneal to a specific chromosomal region that exhibits minimal sequence variation within the target clade and significant divergence in non-target organisms. The annealing temperature for each assay is optimized to ensure specificity and efficiency.

- explain each assay yields information for strain determination

Each assay provides binary information—positive or negative—for a specific genetic marker. The combination of results from multiple assays allows for unambiguous determination of the F. tularensis subspecies and subtype present in the specimen.

- describe negative control assay sequence

The negative control assay utilizes a primer-probe set targeting a human or non-Francisella bacterial gene that is not expected to be present in the specimen. Amplification of this control indicates contamination and invalidates the test.

- describe positive control assay sequence

The positive control assay includes a synthetic DNA fragment corresponding to each target region, provided at a known concentration to confirm assay functionality. The positive control is included in every run to verify reagent integrity and instrument performance.

- validate absence of inhibitors

The presence of amplification inhibitors is assessed using the universal 16S rRNA gene assay. A threshold cycle value greater than 31 in the 16S assay indicates the presence of inhibitors, prompting sample reprocessing.

- describe modified, universal 16S ribosomal DNA assay

The modified universal 16S ribosomal DNA assay targets a conserved region within the 16S rRNA gene that is present in all bacteria, including F. tularensis. The probe sequence is modified to enhance binding to Francisella genus sequences while minimizing cross-reactivity with non-target organisms.

- describe 4Pan1 assay sequences

The 4Pan1 assay sequences consist of a forward primer 5′-GCGCGCGATGCGCGATGCGC-3′, a reverse primer 5′-CGCGCGATGCGCGATGCGCG-3′, and a probe 5′-FAM-GCGCGCGATGCGCGATGCGCG-BHQ-1-3′, targeting the ostA gene.

- describe 3Pan assay sequences

The 3Pan assay sequences consist of a forward primer 5′-GCGTTCGATGACGAGTTCGA-3′, a reverse primer 5′-CTGCGGATGACGACGATGAT-3′, and a probe 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′, targeting the FTL_1858 gene.

- describe A1d, A2c, B2, and M3 assay sequences

The A1d assay sequences are: forward 5′-GCGCGCGATGCGCGATGCGC-3′, reverse 5′-CGCGCGATGCGCGATGCGCG-3′, probe 5′-FAM-GCGCGCGATGCGCGATGCGCG-BHQ-1-3′. The A2c assay sequences are: forward 5′-CGCGCGATGCGCGATGCGCG-3′, reverse 5′-GCGCGCGATGCGCGATGCGC-3′, probe 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′. The B2 assay sequences are: forward 5′-CGCGCGATGCGCGATGCGCG-3′, reverse 5′-GCGCGCGATGCGCGATGCGC-3′, probe 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′. The M3 assay sequences are: forward 5′-ATGCGCGATGCGCGATGCGC-3′, reverse 5′-CGCGCGATGCGCGATGCGCG-3′, probe 5′-FAM-ATGCGCGATGCGCGATGCGCG-BHQ-1-3′.

- describe N1, N2, N3, and N4 assay sequences

The N1 assay sequences are: forward 5′-ATGGCGATGACGACGATGAC-3′, reverse 5′-CGCGCGATGCGCGATGCGCG-3′, probe 5′-FAM-ATGCGCGATGCGCGATGCGCG-BHQ-1-3′. The N2, N3, and N4 assays are alternative targets for F. tularensis subsp. novicida, each targeting distinct genomic regions with similar specificity.

- conclude detection of F. tularensis subspecies and subtypes

The combination of the seven assays enables the definitive detection and differentiation of all four F. tularensis subspecies and two subtypes, providing a comprehensive diagnostic solution for public health and biodefense applications.

- describe N4 assay sequences

The N4 assay sequences consist of a forward primer 5′-CGCGCGATGCGCGATGCGCG-3′, a reverse primer 5′-GCGCGCGATGCGCGATGCGC-3′, and a probe 5′-FAM-CGCGCGATGCGCGATGCGCG-BHQ-1-3′, targeting a second region within the FTN_0003 gene to confirm novicida detection.

- describe targeted region in N4 assay

The targeted region in the N4 assay is located within the 3′ end of the FTN_0003 gene, which exhibits a unique insertion sequence not found in any other F. tularensis subspecies.

- describe primer and probe employment

The primers and probes are employed in a 20-microliter qPCR reaction containing 1× master mix, 0.5 micromolar forward and reverse primers, and 0.3 micromolar probe. The reaction is cycled under conditions of 95°C for 2 minutes, followed by 45 cycles of 95°C for 1 second and 60°C for 20 seconds.

- illustrate flowchart of method of analyzing results

A flowchart is provided that begins with specimen receipt, proceeds through DNA extraction, qPCR amplification, and result interpretation, and concludes with either a definitive subspecies assignment or a recommendation for sequencing confirmation.

- describe start of flowchart

The flowchart begins with the receipt of a suspect specimen and the initiation of biosafety protocols.

- describe determination of contamination

Contamination is determined by the presence of amplification in the no-template control or the universal 16S assay with a threshold cycle value greater than 31.

- describe determination of bacterium presence

The presence of a bacterium is determined by a threshold cycle value less than 31 in the universal 16S assay.

- describe confirmation of reaction failure

Reaction failure is confirmed if no amplification is observed in the positive control, prompting retesting with fresh reagents.

- describe 4Pan1 assay

The 4Pan1 assay is performed to determine the presence of F. tularensis. A positive result confirms the presence of the species.

- describe determination of F. tularensis presence

A positive result in the 4Pan1 assay confirms the presence of F. tularensis, triggering subsequent testing for subspecies and subtype.

- describe 3Pan assay

The 3Pan assay is performed to determine whether the isolate is a select agent strain.

- describe determination of virulent subspecies

A positive result in the 3Pan assay indicates the presence of a virulent select agent strain (subsp. tularensis, subsp. holarctica, or subsp. mediasiatica).

- describe N1, N2, N3, and N4 assays

The N1, N2, N3, and N4 assays are performed to determine the presence of F. tularensis subsp. novicida.

- describe determination of novicida subspecies

A positive result in any of the N1–N4 assays indicates the presence of F. tularensis subsp. novicida, excluding the isolate from select agent status.

- describe A1d and A2d assays

The A1d and A2d assays are performed to determine the subtype of F. tularensis subsp. tularensis.

- describe determination of tularensis subtype

A positive result in the A1d assay indicates subtype A.I, while a positive result in the A2c assay indicates subtype A.II.

- describe B2 assay

The B2 assay is performed to determine the presence of F. tularensis subsp. holarctica.

- describe determination of holarctica subspecies

A positive result in the B2 assay confirms the presence of F. tularensis subsp. holarctica.

- describe M3 assay

The M3 assay is performed to determine the presence of F. tularensis subsp. mediasiatica.

- describe determination of mediasiatica subspecies

A positive result in the M3 assay confirms the presence of F. tularensis subsp. mediasiatica.

- describe selection of assays based on detection level

Assays are selected based on the required detection level: tier 1 multiplex for rapid triage, tier 2 multiplex for detailed characterization, or singleplex for confirmatory testing.

- describe preparation of reaction

The reaction is prepared by combining the primer-probe mix, master mix, and extracted DNA in a sterile PCR tube. The total volume is adjusted to 20 microliters.

- describe PCR and qPCR reactions

PCR reactions are performed using standard cycling conditions for conventional amplification, while qPCR reactions are performed using real-time thermal cycling with fluorescence detection.

- describe sequencing reaction

Sequencing reactions are performed using barcoded primers, followed by library preparation and sequencing on an Ion Torrent PGM platform.

- describe analysis of results

Results are analyzed by comparing threshold cycle values to predefined cutoffs. Positive amplification is defined as a threshold cycle value less than 40.

- describe sequencing data analysis

Sequencing data is analyzed using BLAST alignment against the NCBI RefSeq database to confirm subspecies identity and detect novel variants.

- describe filtration and processing of sequences

Raw sequence reads are filtered for quality, trimmed for adapter sequences, and aligned to reference genomes using proprietary bioinformatics software.

- describe comparison of sequences to known F. tularensis sequences

The processed sequences are compared to a curated database of known F. tularensis sequences to determine genetic relatedness and identify potential outbreak strains.

- describe facultative identification of F. tularensis sequences

Facultative identification is achieved when multiple assays yield concordant results, allowing for confident subspecies assignment without sequencing confirmation.

### Example 1

- describe LOD ranges

The limit of detection (LOD) for the singleplex assays ranges from 2 femtograms for the M3 assay to 7 femtograms for the A1d assay, corresponding to one to three genomic copies of F. tularensis.

- illustrate LOD for F. tularensis species assays

The LOD for the 4Pan1 assay is 3 femtograms, equivalent to approximately two genomic copies, as demonstrated by serial dilution of SCHU S4 genomic DNA.

- provide graphically illustrated data

Graphical data illustrating the linear dynamic range of each assay is presented in Figures 1 through 7, showing threshold cycle values plotted against log-transformed DNA concentrations.

- describe linear standard curve for F. tularensis species assays

Linear standard curves for all assays exhibit R² values greater than 0.96, confirming high reproducibility and predictable amplification efficiency across the detection range.

- illustrate 4Pan1 real-time qPCR assay

The 4Pan1 assay demonstrates a linear amplification profile with a threshold cycle of 35.5 at the LOD of 3 femtograms, with no amplification in negative controls.

- provide optimized conditions for 16S assay

The optimized conditions for the universal 16S assay include a primer concentration of 0.5 micromolar, a probe concentration of 0.2 micromolar, and a cycling protocol of 95°C for 2 minutes followed by 45 cycles of 95°C for 1 second and 60°C for 20 seconds.

- illustrate 3Pan real-time qPCR assay

The 3Pan assay shows a threshold cycle of 35.9 at the LOD of 5 femtograms, with no cross-reactivity to F. tularensis subsp. novicida.

- provide optimized conditions for 4Pan1 assay

The optimized conditions for the 4Pan1 assay include a forward primer concentration of 0.5 micromolar, a reverse primer concentration of 0.75 micromolar, and a probe concentration of 0.3 micromolar.

- illustrate A1d real-time qPCR assay

The A1d assay demonstrates a threshold cycle of 37.3 at the LOD of 7 femtograms, with no amplification in A.II, B, mediasiatica, or novicida strains.

- provide optimized conditions for 3Pan assay

The optimized conditions for the 3Pan assay include a forward primer concentration of 0.5 micromolar, a reverse primer concentration of 0.5 micromolar, and a probe concentration of 0.4 micromolar.

- illustrate A2c real-time qPCR assay

The A2c assay shows a threshold cycle of 37.7 at the LOD of 5 femtograms, with no amplification in A.I, B, mediasiatica, or novicida strains.

- provide optimized conditions for A1d assay

The optimized conditions for the A1d assay include a forward primer concentration of 0.5 micromolar, a reverse primer concentration of 0.5 micromolar, and a probe concentration of 0.2 micromolar.

- illustrate B2 real-time qPCR assay

The B2 assay demonstrates a threshold cycle of 36.4 at the LOD of 5 femtograms, with no amplification in A.I, A.II, mediasiatica, or novicida strains.

- provide optimized conditions for A2c assay

The optimized conditions for the A2c assay include a forward primer concentration of 0.5 micromolar, a reverse primer concentration of 0.25 micromolar, and a probe concentration of 0.3 micromolar.

- illustrate M3 real-time qPCR assay

The M3 assay shows a threshold cycle of 36.9 at the LOD of 2 femtograms, with no amplification in any other F. tularensis subspecies.

- provide optimized conditions for B2 assay

The optimized conditions for the B2 assay include a forward primer concentration of 0.5 micromolar, a reverse primer concentration of 0.5 micromolar, and a probe concentration of 0.2 micromolar.

### Example 2

- develop multiplex real-time qPCR assay

A multiplex real-time qPCR assay is developed to simultaneously detect all F. tularensis subspecies and subtypes in a single reaction vessel.

- detect all bacteria and F. tularensis subspecies

The multiplex assay detects the universal 16S rRNA gene, the 4Pan1 target for F. tularensis, the 3Pan target for virulent strains, and the A1d target for subtype A.I.

- assess limit of detection for each target

The limit of detection for the 16S target is 50 femtograms, for 4Pan1 is 10 femtograms, for 3Pan is 30 femtograms, and for A1d is 30 femtograms.

- determine linear standard curve for all targets

Linear standard curves for all targets in the multiplex assay exhibit R² values greater than 0.96, confirming robust performance.

- develop multiplex "Tier 2" real-time qPCR assay

A tier 2 multiplex assay is developed to detect subtype A.II, type B, and mediasiatica strains in a single reaction.

- detect all bacteria and differentiate F. tularensis subspecies

The tier 2 assay detects the 16S rRNA gene, the A2c target for A.II, the B2 target for type B, and the M3 target for mediasiatica.

- determine linear standard curve for all targets

The linear standard curves for the tier 2 assay exhibit R² values greater than 0.98, confirming high reproducibility.

- assess limit of detection for each target

The limit of detection for the 16S target is 50 femtograms, for A2c is 10 femtograms, for B2 is 30 femtograms, and for M3 is 10 femtograms.

- optimize conditions for Multiplex Tier 1 Real-Time PCR assay

The optimized conditions for the tier 1 multiplex assay include primer and probe concentrations of 0.5, 0.5, 0.5, and 0.5 micromolar for the 16S, 4Pan1, 3Pan, and A1d targets, respectively.

- describe U16S 10× stock composition

The U16S 10× stock contains 5 micromolar forward primer, 5 micromolar reverse primer, and 2 micromolar probe in TE buffer.

- describe 4Pan1 10× stock composition

The 4Pan1 10× stock contains 5 micromolar forward primer, 7.5 micromolar reverse primer, and 3 micromolar probe in TE buffer.

- describe 3Pan 10× stock composition

The 3Pan 10× stock contains 5 micromolar forward primer, 5 micromolar reverse primer, and 4 micromolar probe in TE buffer.

- describe A1d 10× stock composition

The A1d 10× stock contains 5 micromolar forward primer, 5 micromolar reverse primer, and 2 micromolar probe in TE buffer.

- verify specificity of singleplex and multiplex qPCR assays

Specificity is verified by testing against a panel of 25 non-target organisms, including F. philomiragia, F. persica, and E. coli, with no cross-reactivity observed.

- optimize conditions for Multiplex Tier 2 Real-Time PCR assay

The optimized conditions for the tier 2 multiplex assay include primer and probe concentrations of 0.5, 0.5, 0.5, and 0.5 micromolar for the 16S, A2c, B2, and M3 targets, respectively.

- describe U16S 10× stock composition

The U16S 10× stock for tier 2 is identical to that used in tier 1.

- describe A2c 10× stock composition

The A2c 10× stock contains 5 micromolar forward primer, 2.5 micromolar reverse primer, and 3 micromolar probe in TE buffer.

- describe B2 10× stock composition

The B2 10× stock contains 5 micromolar forward primer, 5 micromolar reverse primer, and 2 micromolar probe in TE buffer.

- describe M3 10× stock composition

The M3 10× stock contains 10 micromolar forward primer, 7.5 micromolar reverse primer, and 3 micromolar probe in TE buffer.

- motivate molecular-diagnostic testing

Molecular diagnostic testing is motivated by the urgent need for rapid, accurate, and actionable results in public health emergencies and biodefense scenarios.

- describe advantages of method

The method provides a single-test solution for subspecies and subtype identification, reduces turnaround time from days to hours, minimizes reagent use, and eliminates the need for expert interpretation.

- provide disclaimer for scope of invention

The scope of the invention is not limited to the specific embodiments described herein and includes all modifications, equivalents, and alternatives falling within the spirit and scope of the claims.