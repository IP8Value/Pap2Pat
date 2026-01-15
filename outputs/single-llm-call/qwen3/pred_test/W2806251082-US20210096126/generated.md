# DESCRIPTION

## BACKGROUND

- motivate RFID applications  
Radio frequency identification (RFID) technology has become an indispensable tool in modern industrial and consumer systems due to its ability to enable wireless, contactless identification and data exchange without requiring line-of-sight visibility. Its applications span across logistics, retail, transportation, and security sectors, where it is used for inventory tracking, access control, toll collection, and asset management. RFID systems consist of a microchip embedded within a substrate, coupled with an antenna that captures electromagnetic energy from a reader device, enabling the chip to transmit stored information back wirelessly. The passive nature of most commercial RFID tags eliminates the need for internal power sources, making them durable, low-maintenance, and suitable for long-term deployment in harsh environments. These characteristics have led to widespread adoption in environments subject to mechanical stress, chemical exposure, temperature extremes, and high humidity, where traditional identification methods such as barcodes or optical labels fail. The global RFID market has demonstrated consistent growth, driven by increasing demand for automation, traceability, and real-time data acquisition, with projections indicating continued expansion into new domains including healthcare, agriculture, and smart manufacturing.

- limitations of RFID in tissue culture  
Despite the proven utility of RFID in macroscopic systems, its application within biological tissue cultures—particularly those involving complex three-dimensional structures such as organoids—has remained largely unexplored and technically unfeasible. Conventional RFID tags are too large, rigid, or chemically incompatible with the delicate physiological conditions required for cell viability, proliferation, and differentiation. Attempts to introduce RFID components into living cells or tissues have been limited to phagocytic cell lines or cancerous models, which lack the structural and functional fidelity necessary for modeling human organ physiology. Moreover, existing methods of chip integration have resulted in cellular toxicity, disruption of tissue architecture, or loss of biological function, rendering these approaches unsuitable for applications requiring long-term culture, repeated passaging, or phenotypic analysis under dynamic conditions. The inability to embed RFID technology within self-organizing, polarized, lumen-containing tissues has prevented the development of a scalable, non-invasive method for uniquely identifying, tracking, and correlating individual organoid lineages with donor-specific genetic backgrounds in pooled experimental settings. This gap has hindered progress in high-throughput screening, personalized medicine, and comparative disease modeling using human-derived organoid systems.

## BRIEF SUMMARY OF THE INVENTION

- summarize invention  
The invention disclosed herein comprises a novel method and composition for integrating a microscale radio frequency identification (RFID) chip directly into human organoids during their self-assembly phase, resulting in a digitally encoded biological structure termed a “RFID-integrated organoid” or “RiO.” This innovation enables the persistent, non-destructive, and wireless identification of individual organoids throughout their lifecycle, from derivation through cryopreservation, thawing, functional assays, and in vivo transplantation. The RFID chip is incorporated during the re-aggregation of precursor cells, where the natural self-organization process facilitates the passive encapsulation of the chip within the developing lumen without compromising tissue integrity, polarity, or physiological function. Each RiO retains full organoid morphology, gene expression profiles, secretory capacity, transport functionality, and metabolic activity equivalent to non-integrated controls. The system permits high-throughput screening of pooled organoid populations by enabling simultaneous phenotypic readouts—such as fluorescence-based biomarker detection—and wireless retrieval of donor-specific identifiers via RFID readers, thereby eliminating manual tracking errors and enabling scalable, cost-effective forward cellomics approaches for precision medicine and drug discovery.

## DETAILED DESCRIPTION

### Definitions

- define terms according to conventional usage  
Terms used herein are to be interpreted in accordance with their ordinary and customary meanings in the fields of molecular biology, tissue engineering, biomedical engineering, and electronics, unless otherwise explicitly defined. All technical and scientific terms used herein have the meanings commonly understood by a person of ordinary skill in the art to which this invention pertains, taking into account the context in which such terms are used.

- specify singular forms to include plural referents  
Whenever a singular form of a noun, verb, or adjective is used herein, it is intended to encompass the plural form unless the context clearly indicates otherwise. For example, reference to “an organoid” includes one or more organoids, and reference to “a cell” includes one or more cells.

- define "about" or "approximately"  
The terms “about” or “approximately” when used in reference to a numerical value or range shall mean ±10% of the stated value, unless otherwise specified. These terms are intended to account for minor variations inherent in measurement, preparation, or environmental conditions that do not materially affect the function or outcome of the invention.

- define "individual," "host," "subject," and "patient"  
The terms “individual,” “host,” “subject,” and “patient” are used interchangeably herein to refer to a human being from whom biological material, such as somatic cells or tissue samples, is derived for the generation of induced pluripotent stem cells or organoids. These terms encompass both healthy individuals and those diagnosed with a disease or disorder, whether genetically inherited, acquired, or idiopathic.

- define "precursor cell"  
A “precursor cell” refers to a progenitor cell that is capable of differentiating into one or more specialized cell types within a specific tissue lineage, including but not limited to definitive endoderm cells, hepatic progenitors, intestinal stem cells, or renal progenitors. Precursor cells may be derived from pluripotent stem cells, adult stem cells, or directly reprogrammed somatic cells.

- describe RFID technology  
RFID technology comprises a passive microchip embedded within a substrate and coupled to a coiled antenna, designed to harvest electromagnetic energy from an external reader and transmit stored data wirelessly in response. The chip contains a memory unit capable of storing a unique identifier, typically encoded as an Electronic Product Code (EPC), and operates without an internal power source. The system functions within a range of millimeters to centimeters depending on frequency, antenna design, and environmental interference.

- describe RFID applications  
RFID applications include asset tracking, inventory management, access control, automated toll collection, and animal identification. In medical contexts, RFID has been applied to pharmaceutical tracking, patient identification, and implantable devices for monitoring medication adherence.

- describe RFID in healthcare  
In healthcare, RFID has been utilized to track medical equipment, manage patient flow in hospitals, and monitor drug administration through ingestible sensors. These applications demonstrate the compatibility of RFID with biological systems under physiological conditions, though prior implementations have not extended to three-dimensional tissue constructs.

- introduce human organoids  
Human organoids are three-dimensional, self-organizing, multicellular structures derived from stem cells that recapitulate key architectural, functional, and genetic features of native human organs. They are generated through directed differentiation and culture under conditions that promote self-assembly, polarization, and lumen formation.

- describe organoid features  
Organoids exhibit cell-type diversity, spatial organization, tissue polarity, and physiological functions such as secretion, absorption, and transport that mirror those of their in vivo counterparts. They are enclosed by a basement membrane and contain specialized luminal compartments that facilitate fluid dynamics and barrier formation.

- describe potential organoid applications  
Organoids serve as physiologically relevant models for studying human development, disease mechanisms, drug toxicity, and personalized therapeutic responses. Their use in precision medicine enables patient-specific drug screening and genetic disease modeling.

- introduce digitized organoids  
Digitized organoids are organoids that have been permanently and non-invasively tagged with a microscale RFID chip, enabling each organoid to be uniquely identified and tracked throughout experimental procedures without altering its biological properties.

- describe digitized organoid features  
Digitized organoids retain all native morphological, molecular, and functional characteristics of their non-integrated counterparts while acquiring a persistent, machine-readable identifier. The embedded chip is spatially confined within the lumen and does not interfere with cell-cell interactions, gene expression, or metabolic activity.

- describe pooled organoid composition  
A pooled organoid composition refers to a mixed population of organoids derived from multiple donors, cultured together in a single vessel, and distinguishable only through their embedded RFID identifiers. This configuration enables high-throughput phenotypic screening under identical environmental conditions.

- describe detectable sensor forms  
Detectable sensor forms include fluorescent reporters, luminescent markers, metabolic dyes, and RFID tags, each capable of providing quantitative or qualitative readouts of biological states such as viability, gene expression, lipid accumulation, or bile transport.

- describe biological parameters  
Biological parameters include gene expression levels, protein secretion rates, transport efficiency, lipid accumulation, iron deposition, metabolic activity, and morphological metrics such as lumen size, epithelial thickness, and cell density.

- describe identifier unique to donor  
Each RFID chip is programmed with a unique identifier that corresponds to the genetic origin of the organoid, allowing for unambiguous linkage between phenotypic data and the donor’s genomic background.

- describe organoid types  
Organoid types include, but are not limited to, esophageal, gastric, small intestinal, colonic, hepatic, biliary, pancreatic, pulmonary, and renal organoids, each derived from appropriate precursor cell populations and maintained under lineage-specific culture conditions.

- describe method of making RFID-integrated organoid  
The method involves the co-aggregation of precursor cells with a microscale RFID chip under conditions that promote self-assembly, wherein the chip is passively internalized into the forming lumen during polarization.

- describe contacting definitive endoderm cells with micro-RFID chip  
Definitive endoderm cells are suspended in a culture medium containing a micro-RFID chip at a concentration sufficient to ensure one chip per aggregate, followed by centrifugation or settling to facilitate physical contact.

- describe co-aggregation of cells and micro-RFID chip  
Co-aggregation occurs during the initial re-aggregation phase, wherein cells adhere to one another and encapsulate the chip within the nascent lumen, driven by cell-cell adhesion forces and extracellular matrix deposition.

- describe formation of organoid with lumen  
The organoid develops a polarized epithelial layer surrounding a central lumen, with the RFID chip stably positioned within the lumenal space, shielded from direct contact with epithelial cells but accessible to luminal fluids.

- describe derivation of spheroid from precursor cell  
A spheroid is formed by culturing dissociated precursor cells in low-adhesion plates under conditions that promote cell-cell aggregation, resulting in a compact, spherical structure prior to polarization.

- describe formation of esophageal organoid  
Esophageal organoids are generated by differentiating pluripotent stem cells through a foregut endoderm intermediate, followed by exposure to retinoic acid and BMP inhibitors to specify esophageal identity.

- describe formation of stomach organoid  
Stomach organoids are derived by patterning definitive endoderm with Wnt and FGF signaling to induce gastric progenitors, followed by culture in Matrigel with gastrin and EGF.

- describe formation of small intestinal organoid  
Small intestinal organoids are generated by activating Wnt and Notch signaling in posterior foregut cells, followed by embedding in Matrigel and culture with Noggin, R-spondin, EGF, and nicotinamide.

- describe formation of colon organoid  
Colon organoids are derived by modulating Wnt and BMP signaling in posterior endoderm, followed by culture in the presence of FGF4, retinoic acid, and TGF-β inhibitors.

- describe formation of hepatic organoid  
Hepatic organoids are generated by sequential activation of Activin A and FGF signaling to induce hepatic endoderm, followed by maturation with HGF and Oncostatin M.

- describe formation of liver organoid  
Liver organoids are formed by maintaining hepatic progenitors in a 3D matrix with hepatocyte growth factors and extracellular matrix components to promote bile duct and parenchymal organization.

- describe formation of bile duct organoid  
Bile duct organoids are derived by inducing cholangiocyte fate through Notch activation and culture in Matrigel with EGF and forskolin to promote cystic lumen formation.

- describe formation of pancreatic organoid  
Pancreatic organoids are generated by directing endoderm toward pancreatic progenitors using retinoic acid and FGF10, followed by culture with EGF, Noggin, and exendin-4.

- describe formation of lung organoid  
Lung organoids are formed by patterning anterior endoderm with FGF and Wnt inhibition to induce lung progenitors, followed by culture in Matrigel with KGF and BMP4.

- describe formation of kidney organoid  
Kidney organoids are generated by differentiating pluripotent cells through mesoderm intermediates using CHIR99021 and FGF9, followed by culture in a 3D matrix with retinoic acid and BMP7.

- describe FGF signaling pathway  
The FGF signaling pathway involves ligand binding to fibroblast growth factor receptors, leading to activation of the RAS-MAPK and PI3K-AKT cascades, which regulate cell proliferation, survival, and differentiation during organogenesis.

- describe FGF pathway activators  
FGF pathway activators include recombinant FGF2, FGF4, FGF7, and FGF10, as well as small molecule agonists that enhance receptor dimerization or downstream signaling.

- describe treatment with FGF signaling pathway activators  
Treatment with FGF pathway activators is performed during early differentiation stages to promote endoderm specification and subsequent organoid patterning, typically at concentrations ranging from 10 to 100 ng/mL.

- describe concentration of signaling molecule  
The concentration of signaling molecules is optimized empirically for each organoid type and is maintained within a range sufficient to induce lineage commitment without inducing aberrant differentiation or toxicity.

- describe media composition  
Media composition includes basal culture medium supplemented with growth factors, small molecule modulators, antibiotics, and extracellular matrix components, tailored to support the survival and differentiation of specific precursor cell types.

- describe Wnt signaling pathway activators  
Wnt signaling pathway activators include CHIR99021, BIO, and recombinant Wnt3a, which stabilize β-catenin and promote progenitor expansion during early organoid development.

- describe BMP activators  
BMP activators include recombinant BMP4 and BMP7, which induce mesodermal and posterior endodermal fates depending on context and concentration.

- describe GSK3 inhibitors  
GSK3 inhibitors, such as CHIR99021, inhibit glycogen synthase kinase 3, thereby promoting β-catenin accumulation and enhancing Wnt pathway activity.

- describe ROCK inhibitor  
ROCK inhibitor, such as Y-27632, is used to suppress actomyosin contractility and improve cell survival during dissociation and re-aggregation.

- describe organoid characterization  
Organoid characterization includes morphological assessment, immunofluorescent staining for lineage-specific markers, quantitative PCR for gene expression, ELISA for secreted proteins, and functional assays for transport and metabolic activity.

- describe RiO features  
RiO features include stable chip integration within the lumen, preservation of epithelial polarity, intact barrier function, normal gene expression profiles, and wireless detectability without loss of viability or function.

- describe precursor cells from multiple individuals  
Precursor cells are derived from induced pluripotent stem cells originating from multiple human donors, each carrying distinct genetic variants, enabling comparative phenotypic analysis across genetic backgrounds.

- describe method of screening cell population  
The method involves pooling RiOs from multiple donors, exposing them to a phenotypic stimulus, measuring a biological readout, and correlating the response with the RFID identifier to link phenotype to genotype.

- describe correlating genotype with phenotype  
Correlation is achieved by linking the unique RFID identifier of each RiO to the genomic data of its donor, enabling precise mapping of phenotypic variations to underlying genetic determinants.

### Examples

- introduce RFID technology  
RFID technology, as employed herein, utilizes passive microchips sized between 400 and 500 micrometers, fabricated with silicon-based circuitry and copper coil antennas, capable of storing a 512-bit unique identifier and transmitting data wirelessly at distances up to two millimeters.

- motivate digitalization of biological tissues  
Digitalization of biological tissues enables the transition from manual, error-prone tracking to automated, high-fidelity data acquisition, facilitating large-scale comparative studies and long-term longitudinal analysis of organoid behavior.

- describe RiO development  
RiO development involves the integration of a micro-RFID chip into human iPSC-derived endoderm spheroids during re-aggregation, followed by polarization in a basement membrane matrix to form a lumen-containing organoid with the chip stably enclosed.

- integrate O-Chip into iPSC-derived organoids  
The O-Chip, measuring 460 × 480 μm, is introduced into dissociated foregut spheroids prior to embedding in laminin-rich matrix, resulting in 95% incorporation efficiency without affecting cell viability or organoid structure.

- describe O-Chip structure and function  
The O-Chip consists of a silicon die with an integrated antenna, memory unit, and RF transceiver, operating at 13.56 MHz, powered by inductive coupling, and capable of transmitting a unique EPC upon interrogation.

- test O-Chip integration into biological tissues  
Integration is confirmed via confocal microscopy showing chip localization within the lumen, absence of tissue damage, and retention of epithelial markers such as E-cadherin and ZO-1.

- generate RiO from different donor-derived iPSC lines  
RiOs are successfully generated from eight distinct human iPSC lines, each carrying unique genetic variants, demonstrating broad applicability across diverse genetic backgrounds.

- compare RiO morphology with Control LO  
RiOs exhibit identical morphology to control liver organoids, with comparable size, epithelial thickness, and lumen formation, as quantified by brightfield and confocal imaging.

- analyze liver specific gene expression in RiO  
qPCR analysis reveals no significant difference in expression levels of ALB, AFP, AAT, HNF4A, or CYP3A4 between RiOs and control liver organoids.

- confirm ALB secretion in RiO  
ELISA measurements show ALB secretion levels of 152.9 ng/mL in RiOs versus 149.8 ng/mL in controls, indicating preserved hepatocyte function.

- analyze bile transport potential in RiO  
CLF and rhodamine123 transport assays demonstrate active luminal accumulation in RiOs, confirming functional expression of BSEP and MDR1 transporters.

- study fat accumulation capacity in RiO  
BODIPY staining reveals lipid droplet accumulation in RiOs following fatty acid exposure, with quantification matching control organoids.

- analyze iron accumulation in RiO  
FeRhoNox staining confirms normal iron uptake and storage capacity in RiOs, with no evidence of chelation or toxicity due to chip presence.

- test O-Chip durability under various conditions  
The O-Chip remains functional after exposure to temperatures ranging from −196°C to 60°C, pH 4–10, ethanol, autoclaving, and paraffin embedding.

- examine cryopreservation potential of RiO  
RiOs survive cryopreservation using slow-freezing protocols with DMSO and sucrose, retaining morphology and chip functionality after thawing.

- develop method for optimal cryopreservation of RiO  
Optimal cryopreservation involves gradual cooling to −80°C over 24 hours, followed by storage in liquid nitrogen, with 90% post-thaw viability and intact RFID signal.

- evaluate RiO tracing capability  
RiOs are successfully tracked in vivo following subcutaneous transplantation in immunodeficient mice, with RFID signals detectable through tissue layers.

- develop device for simultaneous fluorescence and RFID measurement  
A microfluidic device integrates a syringe pump, RFID reader, and fluorescence microscope to capture both phenotypic fluorescence and EPC data as RiOs flow through a capillary channel.

- validate device with RiO phenotyping  
The device successfully correlates BODIPY intensity with EPC identifiers in pooled RiO populations, enabling automated, high-throughput phenotyping.

- motivate use of human stem cell models for precision medicine  
Human stem cell-derived organoids provide genetically accurate models of disease, enabling patient-specific drug testing and mechanistic studies unattainable in animal models.

- describe limitations of manual comparison among different iPSC lines  
Manual comparison of hundreds of iPSC lines is prohibitively labor-intensive, costly, and prone to error due to inconsistent handling, labeling, and tracking.

- introduce RiO-based approach for detecting specific donors  
The RiO system enables unambiguous identification of donor-specific organoids within pooled cultures, eliminating manual tracking and enabling scalable phenotypic screening.

- conduct proof-of-principle phenotypic screen  
A pool of seven RiOs, including two from Wolman disease donors, is exposed to fatty acids; lipid accumulation is quantified via BODIPY, and RFID reading identifies the disease-derived organoids.

- generate pool of frozen RiOs  
A library of cryopreserved RiOs from multiple donors is established, with each vial labeled only by its EPC, enabling long-term storage and retrieval.

- identify specific donor from pool using RFID chip  
After thawing and phenotyping, RFID interrogation unambiguously identifies the Wolman disease-derived RiOs based on elevated lipid accumulation.

- confirm steatosis progression in Wolman-derived LO  
Wolman-derived RiOs exhibit significantly higher lipid accumulation than controls, recapitulating the known pathology of lysosomal acid lipase deficiency.

- conclude RiO-based pooling approach for high-throughput phenotyping  
The RiO pooling strategy enables cost-effective, scalable, and reproducible phenotypic screening across hundreds of genetic variants, transforming organoid-based precision medicine.

- introduce organoid phenotyping strategy  
The strategy combines functional assays with wireless identification to link phenotype to genotype without physical separation or manual labeling.

- describe advantages of RFID integration  
RFID integration eliminates manual errors, enables automation, reduces reagent costs, allows long-term tracking, and supports multiplexed screening in shared culture environments.

- motivate cellomics approach  
Cellomics seeks to correlate cellular phenotypes with genomic variation across large populations, and RFID-integrated organoids provide a scalable platform for this endeavor.

- discuss limitations of conventional laboratory protocols  
Conventional protocols require individual culture of each line, leading to high costs, variable conditions, and limited throughput, rendering population-scale studies impractical.

- highlight cost-effectiveness of RFID-based strategy  
Each RFID chip costs less than $0.20, making population-scale screening orders of magnitude more economical than single-cell genomics or manual tracking methods.

- introduce forward cellomics approach  
Forward cellomics begins with phenotypic screening in pooled populations, then identifies causal genetic variants through linkage to RFID identifiers, reversing the traditional gene-first paradigm.

- describe potential applications of disclosed compositions and methods  
Applications include drug toxicity screening, disease modeling, personalized therapy selection, biobanking, organoid transplantation tracking, and large-scale genetic association studies.

- maintain PSCs  
Human iPSCs are maintained on Matrigel-coated plates in mTeSR1 medium with daily feeding and passaging using EDTA.

- induce definitive endoderm  
Definitive endoderm is induced by treating iPSCs with Activin A (100 ng/mL) and Wnt3a (100 ng/mL) for 48 hours in RPMI/B27 medium.

- generate RFID chip incorporated liver organoid  
Definitive endoderm cells are aggregated with O-Chips in low-adhesion plates, then embedded in Matrigel and matured with HGF and Oncostatin M for 14 days.

- measure albumin secretion level  
Culture supernatants are collected at 24-hour intervals and analyzed via ELISA using a commercial human albumin detection kit.

- perform whole mount immunofluorescence  
Organoids are fixed in 4% PFA, permeabilized with 0.5% Triton X-100, and stained with primary antibodies against ALB and HNF4A, followed by fluorescent secondary antibodies.

- isolate RNA and perform RT-qPCR  
Total RNA is extracted using TRIzol, reverse transcribed with oligo-dT primers, and amplified using SYBR Green master mix with gene-specific primers.

- perform Rhodamine123 and Cholyl-Lysyl-Fluorescein transport assay  
Organoids are incubated with 5 μM rhodamine123 or 10 μM CLF for 30 minutes, washed, and imaged for luminal accumulation over time.

- perform live-cell imaging of lipid accumulation  
Organoids are incubated with 5 μM BODIPY 493/503 for 30 minutes and imaged using a confocal microscope with 488 nm excitation.

- cryopreserve and thaw RiO  
RiOs are suspended in cryomedium (90% FBS, 10% DMSO), cooled at −1°C/min to −80°C, then transferred to liquid nitrogen for storage; thawing is performed at 37°C with immediate dilution in culture medium.

- describe culture conditions for human iPSC lines  
iPSCs are cultured on Matrigel in mTeSR1 medium at 37°C, 5% CO2, with daily medium changes and passaging every 5–7 days using EDTA.

- describe differentiation medium composition  
Differentiation medium consists of RPMI 1640 supplemented with B27, Activin A, CHIR99021, and FGF4, with sequential changes to induce endoderm, hepatic, and mature hepatocyte fates.

- describe hepatocyte culture medium composition  
Hepatocyte medium contains William’s E medium supplemented with HGF, Oncostatin M, dexamethasone, insulin, and transferrin.

- describe significance testing methods  
Statistical significance is determined using two-tailed Student’s t-test or one-way ANOVA with Tukey’s post-hoc test, with p < 0.05 considered significant.

- provide manufacturer information for reagents and equipment  
All reagents are sourced from Thermo Fisher Scientific, R&D Systems, StemCell Technologies, and Sigma-Aldrich. RFID chips are procured from Alien Technology, and microfluidic devices are fabricated using standard photolithography at the MIT Microphotonics Center.