Here is the drafted patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

RNA binding proteins (RBPs) play a critical role in regulating gene expression by controlling the rate, location, and timing of RNA maturation. Dysregulation of RBP function has been associated with various genetic and somatic disorders, including neurodegeneration and cancer. Technologies such as RNA immunoprecipitation (RIP) and crosslinking and immunoprecipitation (CLIP) have been developed to identify transcriptome-wide RNA binding sites for RBPs. Enhanced CLIP (eCLIP) has enabled standardized profiling of RNA targets for numerous RBPs across different cell lines. However, current CLIP-based methods face limitations in scalability due to technical complexities, particularly the requirement for SDS-PAGE and nitrocellulose membrane transfer steps to size-select immunoprecipitated protein-RNA complexes. Additionally, each RBP requires a separate immunoprecipitation step, increasing the input material burden when studying multiple RBPs simultaneously.  

## SUMMARY  

The present invention provides a novel method for identifying RNA molecules bound by RNA binding proteins (RBPs) through an improved antibody-barcode eCLIP (ABC) approach. The method involves contacting an RNA sample with one or more RNA binding proteins to form RNA-protein complexes. Oligonucleotide-conjugated entities, such as antibodies, recombinant Fab fragments, nanobodies, or aptamers, are used to label the RNA-protein complexes. The RNA sample is then ligated to the oligonucleotide-conjugated entities, creating chimeric RNA or DNA molecules.  

These chimeric molecules are subsequently amplified using polymerase chain reaction (PCR) and sequenced to identify the RNA targets bound by the RBPs. Computational analysis is performed to identify the chimeric RNA or DNA molecules and determine the original RNA-protein complexes. The method further includes isolating the RNA-protein complexes, which may involve lysing cells prior to isolation and immunoprecipitating the complexes using labeled antibodies or other binding entities.  

The invention also encompasses generating oligonucleotide-conjugated antibodies or other binding entities, which can be achieved through amine or thiol reactive probes or click chemistry reactions. Crosslinking agents may be used to stabilize the RNA-protein complexes, and unreacted probes can be removed to reduce background noise. The method allows for multiplexing, where multiple RBPs can be analyzed simultaneously from a single sample by using distinct barcodes for each RBP.  

Additionally, the invention provides kits containing components necessary for performing the described methods. These kits may include unconjugated oligonucleotides, ligases, RNA binding proteins, antibodies, conjugation reagents, magnetic beads, buffers, adapters, primers, and other reagents required for the assay.  

## DETAILED DESCRIPTION  

### Definitions  

**eCLIP**: Enhanced crosslinking and immunoprecipitation, a method for identifying RNA binding sites of proteins.  

**"About" or "approximately"**: Refers to a value or range that may vary by up to 10% from the stated amount.  

**"Including" and variations**: Means encompassing but not limited to the specified elements.  

**"Comprising"**: Indicates that the described method or composition includes the listed components but may also include additional unspecified elements.  

**"Having"**: Denotes possession of the specified characteristics or components.  

**"Includes"**: Synonymous with "comprising," indicating that other elements may be present in addition to those listed.  

**"Example"**: Refers to an illustrative embodiment of the invention and does not limit the scope of the claims.  

**"Preferably" and variations**: Indicates a preferred but non-limiting embodiment of the invention.  

**"Comprising" in the context of process and compound/composition/device**: Specifies that the process or composition includes the essential features but may also incorporate additional steps or components.  

### Methods  

The invention provides a method for identifying RNA molecules bound by RNA binding proteins (RBPs). The method begins by contacting an RNA sample with one or more RBPs to form RNA-protein complexes. The RNA sample may be derived from cells or tissues, which can be lysed prior to complex isolation. The RNA-protein complexes are then isolated using immunoprecipitation techniques, where antibodies or other binding entities specific to the RBPs are employed.  

Oligonucleotide-conjugated entities, such as antibodies, recombinant Fab fragments, nanobodies, or aptamers, are used to label the RNA-protein complexes. These entities are conjugated to oligonucleotides through chemical reactions, such as amine or thiol reactive probes or click chemistry. Unreacted probes are removed to minimize background interference. The RNA molecules bound to the RBPs are ligated to the oligonucleotides on the conjugated entities, forming chimeric RNA or DNA molecules.  

The chimeric molecules are amplified using PCR, and the resulting products are sequenced to identify the RNA targets. Computational analysis is performed to map the sequenced reads to the genome and identify the original RNA-protein complexes. Unique molecular identifiers (UMIs) and randomized sequences may be used to distinguish between unique molecules and PCR duplicates.  

The method allows for multiplexing by incorporating distinct barcodes for each RBP, enabling simultaneous analysis of multiple RBPs from a single sample. This approach significantly reduces the input material requirement and eliminates the need for separate immunoprecipitation steps for each RBP.  

Crosslinking agents, such as UV light or chemical crosslinkers, may be applied to stabilize the RNA-protein complexes before lysis and immunoprecipitation. Magnetic beads coupled with antibodies or other binding entities can be used to isolate the complexes, followed by washing steps to remove nonspecific interactions.  

The invention also includes generating oligonucleotide-conjugated antibodies or other binding entities. These conjugated entities are produced by reacting antibodies with oligonucleotides bearing complementary reactive groups, such as azide and DBCO for click chemistry. The conjugated entities are purified to remove unreacted components before use in the assay.  

### Kits  

The invention further provides kits for performing the described methods. These kits contain all necessary components, including unconjugated oligonucleotides, ligases, RNA binding proteins, antibodies, conjugation reagents, magnetic beads, buffers, and other reagents. The kits may also include adapters and primers for PCR amplification and sequencing, as well as instructions for performing the assay.  

## EXAMPLES  

### Example 1  

A method for identifying RNA targets of RBPs was performed as follows: Oligonucleotides were conjugated to antibodies using click chemistry. The antibodies were purified to remove unreacted components. Cells or tissues were crosslinked with RBPs, lysed, and applied to magnetic beads coupled with the conjugated antibodies. The beads were washed to remove background noise, and the 3' ends of the RNA molecules were repaired. Proximity-based intermolecular ligation was performed to attach the RNA molecules to the oligonucleotides on the antibodies.  

After washing, the RNA binding proteins and antibody peptides were digested, and the RNA molecules were reverse transcribed into cDNA. The cDNA was cleaned up, and a second adapter was ligated. PCR amplification was performed, and the resulting libraries were sequenced. Data analysis involved trimming UMIs and sequencing adapters, mapping reads to the genome, and identifying RNA targets.  

Lysis buffer, high-salt buffer, and no-salt buffer were prepared, and magnetic beads were coupled with secondary and primary antibodies. Immunoprecipitation was performed, followed by RNA end repair and chimeric ligation. Proteinase digestion was carried out, and samples were cleaned using an RNA clean and concentrator kit. The RNA was eluted, reverse transcribed, and prepared for sequencing.  

### Example 3  

Cell pellets were prepared, and viability was validated. Cells were washed, crosslinked, and lysed. Oligonucleotides were conjugated to beads, and libraries were prepared. Lysis mix was added to the cells, followed by sonication and RNA fragmentation. Cellular debris was pelleted, and the supernatant was transferred for immunoprecipitation.  

Primary antibodies were coupled to beads, and immunoprecipitation was performed. Washes were carried out, and RNA end repair was conducted. Chimeric ligation master mix was prepared and added to the samples. Proteinase digestion was performed, and samples were cleaned using silane beads. Reverse transcription was carried out, followed by cDNA end repair and ligation.  

Quantitative PCR was performed to determine cycle thresholds, and PCR amplification was conducted. The libraries were cleaned up, analyzed for length and concentration, and sequenced. Reads were mapped to the human genome and demultiplexed by barcode.  

### Example 4  

A protocol for amplifying and quantifying individual barcodes was introduced. Reagents were listed, and amplification was performed. Cleanup and quantification steps were carried out, followed by PCR amplification of cDNA with dual index addition. PCR cycling conditions were optimized, and library products were cleaned up.  

Library length and concentration were analyzed, and sequencing was performed. Reads were mapped to the genome, and results were illustrated in figures showing binding sites for various RBPs and bar graphs depicting enrichment levels.  

This concludes the detailed description of the invention, including methods, kits, and exemplary embodiments. The invention provides a scalable and efficient approach for identifying RNA targets of RBPs, overcoming limitations of existing technologies.