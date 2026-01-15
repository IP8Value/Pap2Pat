# DESCRIPTION

## BACKGROUND

Thermotoga is a genus of hyperthermophilic, strictly anaerobic bacteria that thrive at temperatures approaching 90°C, making them among the most heat-tolerant organisms known. Isolates of Thermotoga have been recovered from diverse geothermal environments, including deep-sea hydrothermal vents, continental hot springs, and subsurface oil reservoirs. Phylogenetic analyses of 16S rRNA sequences place Thermotoga on a deeply branching lineage within the bacterial domain, suggesting an ancient evolutionary origin that may reflect early life conditions on a hot, anoxic Earth. These organisms are metabolically unique in their ability to ferment a broad spectrum of polysaccharides, including cellulose, xylan, and starch, into molecular hydrogen, organic acids, and carbon dioxide. This metabolic profile has positioned Thermotoga as a promising candidate for biohydrogen production through biotechnological engineering. However, despite their ecological and industrial relevance, the genetic manipulation of Thermotoga has remained severely limited due to the absence of reliable tools for stable gene transfer and expression. Prior efforts to introduce heterologous DNA into Thermotoga species relied on liposome-mediated transformation, which yielded transient antibiotic resistance in liquid culture but failed to produce stable, isolatable transformants on solid media. This inability to culture and select genetically modified Thermotoga colonies has hindered functional genomics, metabolic pathway engineering, and mechanistic studies of thermostability in this lineage.

The cultivation of Thermotoga under laboratory conditions presents significant technical challenges. Standard anaerobic techniques, such as the use of glove boxes or Hungate tubes, require specialized equipment, meticulous gas purging, and labor-intensive handling procedures that are incompatible with high-throughput screening or routine colony isolation. The Hungate technique, while effective for maintaining anaerobiosis, involves narrow-necked tubes that increase the risk of cross-contamination and impede precise manipulation of individual colonies. Moreover, the oxygen sensitivity of Thermotoga, even if transient, drastically reduces plating efficiency when cells are exposed during surface spreading or streaking. Traditional solid media formulations using agar have proven inadequate, as agar degrades at elevated temperatures and fails to support consistent colony formation. These limitations have collectively restricted the study of Thermotoga to biochemical characterization, genomic sequencing, and fermentation profiling, leaving a critical gap in the ability to perform targeted genetic interventions.

Restriction-modification (R-M) systems are ubiquitous bacterial defense mechanisms that protect the host genome from foreign DNA by cleaving unmethylated sequences while methylating endogenous DNA to prevent self-destruction. Type II R-M systems, which consist of a restriction endonuclease and a cognate methyltransferase operating as independent enzymes, are particularly valuable in molecular biology due to their sequence specificity and ease of use in DNA manipulation. In Thermotoga, several predicted methyltransferase and restriction endonuclease genes have been identified through genome annotation, yet their functional roles remain uncharacterized. The presence of active R-M systems in Thermotoga strains is suspected to be a major barrier to successful transformation, as incoming plasmid DNA lacking appropriate methylation patterns is likely cleaved upon entry into the cell. This phenomenon has been documented in other anaerobic genera such as Clostridia, where pre-methylation of shuttle vectors increased transformation efficiency by several orders of magnitude. Despite the known presence of cryptic mini-plasmids like pRQ7, pMC24, and pRKU1 in various Thermotoga isolates, attempts to harness these elements for vector development have been largely unsuccessful due to poor stability, low copy number, or incompatibility with standard selection markers.

Previous attempts to construct Thermotoga-E. coli shuttle vectors, such as pJY1 and pJY2, utilized the replication origin of pRQ7 and conferred transient resistance to chloramphenicol or kanamycin in liquid culture. However, no transformants were ever recovered on solid media, indicating that the vectors were either degraded, not stably maintained, or incompatible with colony-forming conditions. The lack of a robust, selectable, and stably maintained genetic system has impeded the development of Thermotoga as a platform for synthetic biology. Consequently, there exists a compelling need for a cultivation method that enables reliable colony formation under ambient atmospheric conditions, coupled with a genetically tractable shuttle vector system that overcomes the dual barriers of oxygen sensitivity and restriction-mediated DNA degradation.

## SUMMARY

A novel method for cultivating Thermotoga species under aerobic conditions has been developed, enabling high-efficiency plating and isolation of viable colonies without the need for anaerobic chambers or specialized gas-handling equipment. This method employs an embedded growth technique wherein bacterial suspensions are mixed with molten, oxygen-restricted medium containing Gelrite as a solidifying agent, allowing cells to be uniformly distributed within a semi-solid matrix that minimizes oxygen diffusion. The invention further includes a soft SVO medium formulation with reduced Gelrite concentration, facilitating the transfer of single colonies from solid to liquid culture under ambient conditions while preserving cell viability. Concurrently, a Thermotoga-E. coli shuttle vector, designated pDH10, has been constructed and validated for stable replication and selection in both host organisms. The vector integrates the replication origin of the Thermotoga cryptic plasmid pRQ7, a thermostable kanamycin resistance gene under a Thermus-derived promoter, and a ColE1 origin with β-lactamase for selection in E. coli. This vector enables the first documented, stable transformation of Thermotoga species via electroporation and liposome-mediated delivery, with transformants maintaining the plasmid over multiple generations in the absence of selective pressure. The invention further encompasses the functional characterization of a Type II restriction-modification system in Thermotoga neapolitana, including the identification of its recognition sequence and the demonstration that methylation of the shuttle vector prior to transformation enhances transformation efficiency. Together, these components form a comprehensive system for the genetic manipulation of Thermotoga, overcoming longstanding technical barriers and enabling future applications in metabolic engineering, enzyme discovery, and evolutionary biology.

## DETAILED DESCRIPTION

### A. Abbreviations

For clarity and consistency, the following abbreviations are used throughout this disclosure: Ap refers to ampicillin; CFU denotes colony forming unit; DNA refers to deoxyribonucleic acid; EDTA refers to ethylenediaminetetraacetic acid; Kan refers to kanamycin; LB refers to Luria Broth; PCR refers to polymerase chain reaction; Tm refers to Thermotoga maritima; Tn refers to Thermotoga neapolitana; RQ7 refers to Thermotoga sp. RQ7; SVO refers to the synthetic medium developed by van Ooteghem et al.; Gelrite refers to gellan gum, a microbial polysaccharide used as a gelling agent; R-M refers to restriction-modification; REase refers to restriction endonuclease; MTase refers to methyltransferase; AdoMet refers to S-adenosyl-L-methionine; ori refers to origin of replication; Apr refers to ampicillin resistance gene; kan refers to kanamycin resistance gene; and pDH10 refers to the Thermotoga-E. coli shuttle vector described herein.

### B. Terms

In the context of this patent application, all technical terms are used in accordance with standard definitions in molecular biology, microbiology, and genetic engineering. The singular form of a term shall include its plural form unless the context clearly indicates otherwise. The term “about” when used in reference to numerical values, such as temperature, concentration, or time, shall mean ±10% of the stated value unless otherwise specified. The phrase “methyltransferase or functional derivative thereof” refers to any polypeptide capable of transferring a methyl group from S-adenosyl-L-methionine to a specific nucleotide sequence in DNA, including full-length native enzymes, truncated variants, fusion proteins, and amino acid substitutions that retain at least 70% of the catalytic activity of the reference enzyme. Functional derivatives include those with conservative amino acid substitutions, insertions, or deletions that do not abolish the enzyme’s ability to recognize and methylate its target sequence. Similarly, the phrase “restriction endonuclease or functional derivative thereof” refers to any enzyme capable of cleaving double-stranded DNA at a specific recognition site, including naturally occurring variants, chimeric constructs, and engineered mutants that retain sequence-specific cleavage activity. Operable fragments, mutants, or truncated forms of the disclosed genes are those that retain the essential catalytic or binding domains required for their biological function, as demonstrated by in vitro or in vivo assays. All DNA sequences referenced herein are intended to include complementary strands, variants with codon optimization for heterologous expression, and sequences with up to 15% nucleotide divergence that maintain functional equivalence.

### Improved Method for Cultivation of Thermotoga.

The invention provides a novel method for cultivating Thermotoga species on solid media under aerobic conditions, overcoming the historical limitations of anaerobic glove boxes and Hungate techniques. Unlike prior methods that required continuous purging with inert gas or sealed chambers, this method utilizes an embedded growth protocol in which a suspension of Thermotoga cells is mixed with molten SVO medium containing 0.3% Gelrite immediately prior to solidification in Petri dishes. The Gelrite matrix physically restricts oxygen diffusion into the medium, creating localized anaerobic microenvironments around embedded cells while permitting ambient handling of the plates. This approach achieves plating efficiencies approaching 50%, enabling the recovery of over 10⁹ colony forming units per milliliter from a single overnight culture, a ten-thousand-fold improvement over conventional surface spreading. The method is applicable to multiple Thermotoga species, including T. neapolitana, T. maritima, and T. sp. RQ7, and does not require specialized equipment beyond standard laboratory incubators and aerobic workbenches. The invention further encompasses the use of soft SVO medium, prepared by dissolving 0.075% Gelrite in liquid SVO, which permits the physical transfer of individual colonies from solid plates to liquid culture using a sterile loop or pipette tip without exposing cells to prolonged oxygen stress. Soft SVO medium also functions as a stable storage medium, preserving cell viability for up to two months at ambient temperature. The scope of this invention includes any solidifying agent capable of forming a stable gel at 77°C, including but not limited to agarose, carrageenan, or other polysaccharide-based gelling agents that resist thermal degradation. The method is applicable to any growth medium suitable for Thermotoga, including modified versions of SVO, TCS, or other defined or complex formulations. The invention further includes any vessel capable of containing the medium during solidification, including Petri dishes, multi-well plates, or microfluidic chambers, and any method of inoculation, including serial dilution, micropipetting, or robotic spotting.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana ATCC 49049, Thermotoga maritima MSB8, and Thermotoga sp. RQ7 as host strains for genetic manipulation. E. coli DH5α and BL21(DE3) are used as cloning and expression hosts, respectively. All Thermotoga strains are cultivated at 77°C in SVO medium supplemented with 150 μg/ml kanamycin for selection in liquid culture and 250 μg/ml kanamycin in soft or solid media. Growth is monitored by measuring optical density at 600 nm using a spectrophotometer calibrated for high-temperature measurements. For expression studies, the gene encoding CTN-0339, predicted to encode a Type II restriction endonuclease, was amplified by PCR using primers designed to flank the open reading frame and cloned into the pET-24a(+) vector downstream of a T7 promoter. The gene encoding CTN-0340, predicted to encode a cognate methyltransferase, was similarly cloned into pET-24a(+) for overexpression in E. coli. The invention includes all primers capable of amplifying the full-length coding sequences of CTN-0339 and CTN-0340, as well as any variant with up to 15% nucleotide divergence that encodes a functionally equivalent protein. The invention further includes any expression vector containing a promoter active in Thermotoga, including but not limited to promoters derived from Thermus thermophilus, Pyrococcus furiosus, or other thermophilic organisms. Epitope tags such as His₆, FLAG, or HA may be fused to the N- or C-terminus of the encoded proteins for purification or detection, and such fusions are encompassed within the scope of this invention.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

The gene products of CTN-0339 and CTN-0340 were expressed in E. coli BL21(DE3) under induction with isopropyl β-D-1-thiogalactopyranoside. Cells were harvested by centrifugation and lysed by sonication in the presence of protease inhibitors. The lysate was subjected to heat treatment at 80°C for 15 minutes to denature and precipitate non-thermostable E. coli proteins, followed by centrifugation to isolate the thermostable supernatant. The resulting protein fraction was analyzed by SDS-PAGE and Western blotting using antibodies specific for His₆-tagged proteins. Restriction assays were performed by incubating purified CTN-0339 with pUC19 DNA and analyzing cleavage patterns via agarose gel electrophoresis. Modification assays were conducted by pre-incubating pUC19 DNA with CTN-0340 and S-adenosyl-L-methionine, followed by digestion with CTN-0339 to assess protection from cleavage. The invention includes any method of protein purification that exploits the thermostability of the encoded enzymes, including heat precipitation, ion exchange chromatography, or affinity chromatography using epitope tags. The invention further encompasses any DNA sequence encoding a functional homolog of CTN-0339 or CTN-0340, as well as any mutant or variant that retains the ability to cleave or methylate the target sequence CGCG.

### Determination of the Cleavage Site of the REase

The recognition site of CTN-0339 was determined by digesting a PCR-amplified fragment of pUC19 with purified enzyme and separating the resulting fragments by gel electrophoresis. The smaller fragment was excised, purified, and sequenced using Sanger sequencing. Sequence alignment revealed that CTN-0339 cleaves between the two cytosine residues within the palindromic sequence 5′-CGCG-3′, generating blunt-ended fragments. The cleavage was confirmed by cloning the digested fragment into a sequencing vector and verifying the precise junctions. The invention includes any method of determining cleavage specificity, including next-generation sequencing of digested genomic DNA, linker ligation assays, or in vitro transcription-coupled cleavage assays. The invention further encompasses the use of CTN-0339 as a molecular tool for precise, blunt-end cleavage of DNA containing the CGCG sequence.

### Restriction and Modification Assays of the Thermotoga Proteins

Functional assays demonstrated that CTN-0339 cleaves unmethylated pUC19 DNA at the CGCG site, producing a single double-strand break. Pre-methylation of pUC19 DNA by CTN-0340 in the presence of S-adenosyl-L-methionine completely protected the DNA from cleavage by CTN-0339, confirming that CTN-0340 is a cognate methyltransferase. The methylation activity was dependent on the presence of AdoMet and was abolished in the absence of cofactor or in the presence of competitive inhibitors. The invention includes the use of CTN-0339 and CTN-0340 as a pair for site-specific DNA cleavage and protection in molecular cloning applications, particularly in systems requiring high fidelity and minimal off-target activity.

### Overexpression of R.TneDI

The gene encoding CTN-0339, designated R.TneDI, was cloned into the expression vector pET-24a(+) and transformed into E. coli BL21(DE3). Induction with IPTG resulted in the accumulation of a single protein band of approximately 32 kDa, consistent with the predicted molecular weight. The protein was purified by heat treatment and affinity chromatography, and its activity was confirmed by restriction assays. The invention includes any E. coli strain capable of expressing R.TneDI, including but not limited to XL1-Blue MRF′, JM109, or other derivatives, and any expression system utilizing a T7, lac, or other inducible promoter.

### Determination of the Cleavage Site of R.TneDI

The recognition sequence of R.TneDI was identified as 5′-CGCG-3′, and cleavage occurs between the two cytosines, yielding blunt ends. The cleavage specificity was confirmed by digesting synthetic oligonucleotides containing the target sequence and analyzing the products by mass spectrometry and sequencing. The invention includes any DNA molecule containing the CGCG sequence that is susceptible to cleavage by R.TneDI, as well as any method of using R.TneDI to generate blunt-ended fragments for cloning, gene editing, or synthetic biology applications.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of R.TneDI was tested against plasmid DNA from multiple sources, including pUC19, pET vectors, and genomic DNA from Thermotoga species. R.TneDI exhibited strict specificity for the CGCG sequence and did not cleave DNA containing methylated cytosines. The methyltransferase M.TneDI, encoded by CTN-0340, was shown to methylate cytosine residues within the CGCG motif, protecting the host genome from self-cleavage. The invention includes the use of R.TneDI and M.TneDI as a self-contained Type II restriction-modification system for genetic engineering applications in thermophilic organisms.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system, and the protein was purified to homogeneity. The purified enzyme exhibited robust activity at 37°C and retained partial activity at 70°C, indicating thermostability. The invention includes any method of overexpression, including fusion tags, chaperone co-expression, or codon optimization for enhanced yield.

### Determination of the Cleavage Site of R.TneDI

The cleavage site of R.TneDI was determined to be blunt-ended at the CGCG sequence. The enzyme does not require divalent cations for activity and is inhibited by high salt concentrations. The invention includes the use of R.TneDI as a precision tool for DNA manipulation in thermophilic systems, particularly where conventional restriction enzymes are unstable or non-specific under high-temperature conditions.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for genetic transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines, generating blunt ends. The invention includes any method of identifying cleavage sites, including ligation-mediated PCR, exonuclease digestion, or hybridization-based mapping.

### Restriction and Modification Assays of the Thermotoga Proteins

The restriction activity of CTN-0339 was confirmed by its ability to cleave pUC19 DNA, while CTN-0340 protected the same DNA from cleavage upon methylation. The invention includes the use of these proteins as a pair for controlled DNA manipulation in thermophilic systems.

### Overexpression of R.TneDI

R.TneDI was overexpressed in E. coli BL21(DE3) using a T7 promoter system. The protein was purified by heat treatment and nickel-affinity chromatography. The invention includes any expression system capable of producing R.TneDI in high yield.

### Determination of the Cleavage Site of R.TneDI

The recognition site of R.TneDI was identified as CGCG, and cleavage occurs between the cytosines, producing blunt ends. The enzyme does not require metal ions and is inhibited by high salt. The invention includes the use of R.TneDI as a molecular tool for precise DNA cleavage.

### Restriction and Modification Assays of the Thermotoga Proteins

The methyltransferase M.TneDI was shown to methylate cytosine residues within the CGCG sequence, protecting host DNA from cleavage by R.TneDI. The invention includes the use of M.TneDI to modify plasmid DNA prior to transformation into Thermotoga.

### Overexpression of R.TneDI

R.TneDI was expressed in E. coli BL21(DE3) and purified to homogeneity. The enzyme retained activity at elevated temperatures, demonstrating its suitability for thermophilic applications.

### Determination of the Cleavage Site of R.TneDI

The cleavage site was confirmed by sequencing of digested DNA fragments. The invention includes any method of determining cleavage specificity, including mass spectrometry or next-generation sequencing.

### Strains and Cultivation Conditions

The invention encompasses the use of Thermotoga neapolitana, Thermotoga maritima, and Thermotoga sp. RQ7 as host strains for transformation. Cultivation is performed at 77°C in SVO medium, with kanamycin used as the selective agent at concentrations of 150 μg/ml in liquid media and 250 μg/ml in solid or soft media. Growth is monitored by OD₆₀₀ measurements, and induction of gene expression is achieved using IPTG or temperature shifts. The invention includes any vector containing the pRQ7 origin of replication, a kanamycin resistance gene under a thermophilic promoter, and a ColE1 origin for propagation in E. coli.

### Purification and Analyses of CTN 0339 and CTN 0340 Gene Products

Cell lysis was performed by sonication in the presence of lysozyme and protease inhibitors. Heat purification at 80°C for 15 minutes removed non-thermostable contaminants, and the supernatant was analyzed by SDS-PAGE and Western blot. Restriction assays confirmed that CTN-0339 cleaves unmethylated DNA at CGCG, while CTN-0340 methylates cytosine residues within the same sequence. The invention includes any method of detecting or quantifying restriction or methylation activity, including radioactive labeling, fluorescence-based assays, or next-generation sequencing.

### Determination of the Cleavage Site of the REase

A PCR product containing the CGCG sequence was digested with purified CTN-0339, and the fragments were sequenced. The cleavage occurred between the two cytosines