# DESCRIPTION

## BACKGROUND

Thermotoga is a genus of hyperthermophilic, strictly anaerobic bacteria that represent one of the deepest evolutionary branches among known bacterial lineages. These organisms are distinguished by their ability to thrive at temperatures approaching 90°C, a trait shared only with the genus Aquifex among bacteria. Thermotoga species have been isolated from diverse geothermal environments, including marine hydrothermal vents, continental hot springs, and subsurface oil reservoirs, underscoring their ecological adaptability to extreme conditions. Phylogenetic analyses based on 16S ribosomal RNA sequences place Thermotoga near the root of the bacterial tree of life, suggesting an ancient origin during a period when Earth’s surface was significantly hotter and its atmosphere largely anoxic. This evolutionary positioning renders Thermotoga a compelling model system for investigating fundamental biological processes under extreme physicochemical constraints, particularly those related to macromolecular stability, anaerobic metabolism, and early microbial evolution.

The growth conditions required by Thermotoga reflect their extremophilic nature. These organisms grow optimally between 70°C and 90°C in strictly anaerobic environments, utilizing a fermentative metabolism to degrade complex polysaccharides into simpler compounds, with hydrogen gas as a major metabolic byproduct. Their capacity to produce hydrogen from renewable biomass has attracted considerable interest for bioenergy applications, especially in the context of sustainable and carbon-neutral energy production. However, despite their biotechnological promise and phylogenetic significance, progress in harnessing Thermotoga for applied or basic research has been severely hampered by the absence of robust genetic tools. Unlike model mesophiles such as Escherichia coli, which benefit from decades of molecular toolkit development, Thermotoga remains largely refractory to routine genetic manipulation.

Studying Thermotoga is critically important not only for understanding the limits of life under extreme conditions but also for advancing synthetic biology in non-model thermophiles. The development of reliable methods for gene transfer, expression, and stable maintenance in Thermotoga would enable targeted metabolic engineering, functional genomics, and protein engineering studies. Such capabilities could unlock the full potential of Thermotoga as a chassis for high-temperature bioprocessing and provide insights into the evolution of thermostable enzymes and regulatory networks.

Current cultivation methods for Thermotoga suffer from significant limitations that impede genetic experimentation. Traditional approaches rely on anaerobic glove boxes to maintain oxygen-free conditions during plating and colony isolation. While effective, glove boxes are expensive, cumbersome to operate, and ill-suited for precise manipulations such as single-colony picking due to the thick gloves required. Alternative techniques, such as the Hungate roll-tube method, involve sealing cultures in glass tubes under anoxic gas and rolling them to distribute cells in a solidified medium. Although useful for some anaerobes, this method is prone to cross-contamination, difficult to scale, and incompatible with standard microbiological workflows. An overlay technique—where molten agar is poured over pre-inoculated cells—has been used with limited success but still requires anaerobic chambers or gas-purged environments, limiting accessibility.

A further barrier to genetic manipulation lies in the presence of restriction-modification (R-M) systems, which are widespread in prokaryotes and serve as innate immune defenses against foreign DNA. Restriction endonucleases recognize specific short DNA sequences and cleave unmethylated DNA, thereby degrading invading phage genomes or plasmids. Concurrently, modification methyltransferases methylate the same recognition sequences in the host genome, rendering it resistant to cleavage. Type II R-M systems, the most common class used in molecular biology, consist of separate restriction and modification enzymes that act independently but recognize identical sequences. The variability of R-M systems across bacterial strains means that exogenous DNA introduced from E. coli—typically unmethylated at Thermotoga-specific sites—is often degraded upon entry, drastically reducing transformation efficiency.

In Thermotoga, genomic analyses have predicted the presence of multiple methyltransferase genes, though functional assignments for many remain unverified. Recent work has begun to characterize specific R-M systems in Thermotoga, revealing active restriction endonucleases that pose a formidable barrier to heterologous DNA introduction. This underscores the necessity of either inactivating host R-M systems or pre-methylating vectors to match the host’s modification pattern—a challenge that has yet to be systematically addressed in Thermotoga.

Genetic manipulation efforts in Thermotoga have historically relied on cryptic mini-plasmids such as pRQ7, pMC24, and pRKU1, which are nearly identical 846-bp elements encoding a single replication protein and replicating via a rolling-circle mechanism. Leveraging pRQ7, researchers constructed early Thermotoga-E. coli shuttle vectors, pJY1 and pJY2, which conferred transient antibiotic resistance in liquid culture but failed to yield stable transformants on solid media. Despite over 1,200 publications on Thermotoga, no reproducible method for stable genetic transformation had been established prior to the present invention. This gap highlights the urgent need for a tractable gene transfer system that supports not only DNA delivery but also stable maintenance and functional expression of heterologous genes.

The creation of improved Thermotoga-E. coli shuttle vectors represents a pivotal step toward overcoming these barriers. Such vectors must replicate autonomously in both hosts, carry selectable markers functional in Thermotoga, and resist degradation by host R-M systems. The successful development of such a system would mark a transformative advance, enabling the first true genetic engineering of Thermotoga and opening new avenues for both fundamental and applied research.

## SUMMARY

The present invention provides an improved method for the cultivation and genetic manipulation of Thermotoga species, overcoming longstanding technical barriers that have impeded molecular studies in this hyperthermophilic genus. Central to the invention is a novel embedded growth technique that enables high-efficiency plating of Thermotoga under aerobic conditions without the need for anaerobic chambers or specialized gas-handling equipment. This method achieves plating efficiencies approaching 50%, a dramatic improvement over traditional surface plating, thereby facilitating the isolation of single colonies—including genetically modified transformants—for the first time in Thermotoga.

In addition, the invention discloses a stable and functional Thermotoga-E. coli shuttle vector, designated pDH10, which enables the introduction, expression, and stable maintenance of heterologous genes in Thermotoga. The vector combines the replication origin of the native Thermotoga mini-plasmid pRQ7 with a thermostable kanamycin resistance gene driven by a promoter active in Thermotoga, allowing for effective selection in both liquid and solid media. pDH10 has been successfully introduced into Thermotoga sp. RQ7 and Thermotoga maritima via both liposome-mediated transformation and electroporation, yielding viable transformants that stably retain the plasmid even in the absence of selective pressure. Furthermore, the invention identifies kanamycin as a suitable selection marker for specific Thermotoga strains and defines optimal concentrations for reliable selection.

The invention also encompasses the DNA sequences encoding key components of the system, including the kanamycin adenyltransferase gene adapted for thermophilic expression and the pRQ7-derived replication origin. These sequences, along with the engineered vector architecture, constitute a foundational platform for future genetic engineering of Thermotoga and related hyperthermophiles.

## DETAILED DESCRIPTION

### A. Abbreviations

The following abbreviations are used throughout this application: Ap refers to ampicillin; CFU denotes colony forming unit; DNA stands for deoxyribonucleic acid; EDTA is ethylenediaminetetraacetic acid; Kan signifies kanamycin; LB designates Luria Broth; PCR is polymerase chain reaction; Tm refers to Thermotoga maritima; Tn denotes Thermotoga neapolitana; and RQ7 refers to Thermotoga sp. RQ7. These abbreviations are standard in the art and are used consistently herein to enhance clarity and conciseness.

### B. Terms

As used herein, technical terms have the meanings commonly understood by those skilled in the relevant art. Molecular biology terms not otherwise defined are to be interpreted in accordance with standard references such as Sambrook and Russell, Molecular Cloning: A Laboratory Manual, 3rd ed., Cold Spring Harbor Laboratory Press (2001). The singular forms “a,” “an,” and “the” include plural referents unless the context clearly dictates otherwise. The term “about” when used in connection with a numerical value indicates a range of ±10% of the stated value unless otherwise specified.

The phrase “methyltransferase or functional derivative thereof” refers to a polypeptide that retains the enzymatic activity of transferring a methyl group from S-adenosylmethionine (AdoMet) to a specific nucleotide within a DNA recognition sequence. Functional derivatives include naturally occurring variants, engineered mutants, fusion proteins, and truncated forms that preserve catalytic function and substrate specificity. Similarly, “restriction endonuclease or functional derivative thereof” denotes a polypeptide capable of cleaving double-stranded DNA at or near a specific recognition site. Functional derivatives encompass operable fragments, point mutants, chimeric enzymes, and truncated versions that maintain sequence-specific cleavage activity. Operable fragments, mutants, or truncated forms are those that, despite modifications, retain sufficient structural integrity to perform the intended biological function in vivo or in vitro.

### Improved Method for Cultivation of Thermotoga.

The present invention introduces an improved method for the cultivation of Thermotoga that eliminates the need for anaerobic glove boxes or complex gas-handling systems during colony isolation. Previous methods, including the traditional anaerobic glove box approach and the Hungate roll-tube technique, are labor-intensive, prone to contamination, and incompatible with high-throughput workflows. The embedded growth method of the invention overcomes these limitations by suspending diluted Thermotoga cultures directly into molten SVO medium containing a low concentration of Gelrite (0.3%), which is then poured into Petri dishes and allowed to solidify. In this configuration, cells are embedded within the semi-solid matrix, minimizing exposure to atmospheric oxygen while permitting normal growth and colony formation.

SVO medium, as developed by van Ooteghem et al., is prepared by dispensing 50 mL into 100 mL serum bottles, sparging with nitrogen to remove dissolved oxygen, sealing with rubber stoppers, and autoclaving. For embedded plating, double-strength SVO is mixed with an equal volume of Gelrite solution while hot, inoculated with diluted culture, and poured into plates. Incubation at 77°C in a standard anaerobic jar containing a palladium catalyst under a 96:4 N₂:H₂ atmosphere yields visible colonies within 24–48 hours. This method achieves plating efficiencies of approximately 50%, as demonstrated by quantitative colony counts, representing a >10,000-fold improvement over surface spreading under identical conditions.

To facilitate transfer of colonies to liquid culture under aerobic conditions, the invention further provides a soft SVO medium containing 0.075% Gelrite. Single colonies are picked with a loop and pushed to the bottom of culture tubes filled with soft SVO, where an oxygen-depleted microenvironment supports cell viability. After 12–24 hours of incubation, cultures are transferred to liquid SVO using a syringe. This two-step transfer protocol ensures maximum cell survival and enables routine handling on the laboratory benchtop. The scope of the invention extends to any solidifying agent compatible with Thermotoga growth, including agar and alternative gelling polymers, as well as any nutrient medium supporting Thermotoga proliferation. Likewise, the method applies to various vessel types and incubation protocols, provided that anaerobic conditions are maintained during incubation, whether through gas exchange, chemical scavengers, or physical barriers.

### Strains and Cultivation Conditions

The bacterial strains and vectors employed in the invention include Thermotoga neapolitana ATCC 49049, Thermotoga sp. RQ7, Thermotoga maritima MSB8, and E. coli DH5α. T. neapolitana is cultivated at 77°C in SVO medium under strict anaerobic conditions, with growth monitored by optical density at 600 nm. E. coli is grown at 37°C in Luria Broth supplemented with ampicillin (100 μg/mL) when required. For gene expression studies, the CTN_0339 gene was amplified by PCR using specific primers and cloned into the pET-24a(+) vector, which includes a C-terminal His-tag for purification. PCR amplification utilized high-fidelity DNA polymerases and standard cycling conditions. Vector construction involved restriction digestion with appropriate enzymes (e.g., BamHI, EcoRI) and ligation into the target backbone.

The scope of the invention encompasses any primer pair capable of amplifying Thermotoga genes of interest, as well as any expression vector compatible with E. coli or Thermotoga hosts. Epitope tags such as His₆, FLAG, or HA may be incorporated to facilitate detection or purification, and the invention includes all DNA sequences encoding functional Thermotoga proteins or their derivatives, regardless of codon optimization or fusion architecture.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Recombinant CTN_0339 and CTN_0340 proteins were expressed in E. coli and purified using heat treatment followed by centrifugation, exploiting the thermostability of Thermotoga proteins. Cell lysates were heated to 70°C for 20 minutes, denaturing mesophilic host proteins while leaving thermophilic targets intact. The supernatant was analyzed by SDS-PAGE to assess purity and molecular weight. Western blotting confirmed identity using anti-His antibodies. Restriction assays measured DNA cleavage activity by incubating purified CTN_0339 with plasmid substrates, while modification assays tested CTN_0340’s ability to protect DNA from restriction by co-incubating with AdoMet and subsequent challenge with restriction enzymes. The invention includes all methods of protein purification leveraging thermostability, as well as analytical techniques employing epitope tags or activity-based assays.

### Determination of the Cleavage Site of the REase

The cleavage site of the restriction endonuclease was determined by amplifying a target DNA fragment via PCR, digesting it with the purified enzyme, and isolating the smaller cleavage product for sequencing. A vector containing the recognition site was constructed, digested, and subcloned for Sanger sequencing. Analysis revealed a blunt-end cut at the CGCG tetranucleotide, confirming the enzyme as a Type II restriction endonuclease with specificity for this sequence.

### Restriction and Modification Assays of the Thermotoga Proteins

CTN_0339 exhibited robust restriction activity against unmethylated pUC19 DNA, while CTN_0340 conferred protection when pre-incubated with AdoMet, demonstrating that CTN_0339 functions as a restriction endonuclease and CTN_0340 as its cognate methyltransferase. Together, they constitute a functional Type II R-M system.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) following induction with IPTG. Protein expression was confirmed by SDS-PAGE and Western blot, and activity was validated by restriction assays.

### Determination of the Cleavage Site of R.TneDI

R.TneDI was shown to recognize and cleave the CGCG sequence, producing blunt ends. This specificity was confirmed by digesting PCR products containing the site and sequencing the fragments. The enzyme is proposed as a novel molecular tool for DNA manipulation. Host chromosomal DNA is protected by M.TneDI methylation. Orthologs were identified in Thermotogaceae genomes, suggesting recent horizontal gene transfer.

### Antibiotics sensitivity tests

Kanamycin sensitivity was tested using disc diffusion and broth dilution. T. maritima and RQ7 showed clear inhibition zones, while T. neapolitana was less sensitive. Selective concentrations were established as 150 μg/mL in liquid and 250 μg/mL in solid media.

### Construction of pDH10

pDH10 was constructed by amplifying pRQ7 and ligating it into pKT1, which carries a thermostable kan gene under a Thermus promoter. The vector contains ColE1 ori and β-lactamase for E. coli, and pRQ7 ori plus kan for Thermotoga.

### Transformation and selection methods

Liposome-mediated transformation and electroporation (1.8–2.0 kV) successfully delivered pDH10 into Thermotoga. Transformants were selected on kanamycin-containing soft SVO using the embedded method.

### Stability assays

pDH10 was stably maintained in Thermotoga for >72 hours without selection. In E. coli, pDH10 showed enhanced stability over pKT1 due to the pRQ7 insert, not copy number differences.

### Copy number determination

Digestion and gel analysis showed comparable copy numbers between pDH10 and pKT1 in E. coli, confirming that stability enhancement is sequence-dependent, not dosage-related.

## E. Tables

Tables referenced herein include Table 1 (strains and vectors), Table 2 (Thermotoga plasmid stability), and Table 3 (E. coli plasmid stability). These tables provide quantitative data supporting the claims of high transformation efficiency, stable plasmid maintenance, and improved vector performance.