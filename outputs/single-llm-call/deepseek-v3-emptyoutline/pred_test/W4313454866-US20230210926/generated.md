Here is the patent application following the provided outline:

# DESCRIPTION  

## STATEMENT REGARDING FEDERALLY SPONSORED R&D  
The invention described herein was made with government support under Grant No. R01-EB018975 awarded by the National Institutes of Health. The government has certain rights in the invention.  

## BACKGROUND  

### Field  
The present invention relates generally to the field of molecular imaging and reporter gene technology. More specifically, the invention pertains to novel acoustic reporter genes (ARGs) that enable non-invasive ultrasound imaging of genetically modified cells in vivo.  

### Description of the Related Art  
Existing reporter gene technologies for non-invasive imaging have significant limitations. Fluorescent proteins require external illumination and have limited tissue penetration depth. Luciferase-based reporters require substrate administration and suffer from low spatial resolution. Previous acoustic reporter genes based on gas vesicle (GV) gene clusters produce only linear ultrasound contrast or require destructive imaging pulse sequences. Moreover, these prior ARGs exhibit poor expression at physiological temperatures (37°C) and impose substantial metabolic burden on host cells. There exists an unmet need for improved ARGs that provide strong non-linear ultrasound contrast, enable long-term expression under physiological conditions, and minimize cellular burden.  

## SUMMARY  
The present invention provides novel acoustic reporter genes (ARGs) derived from genomically mined gas vesicle (GV) gene clusters that overcome limitations of prior reporter technologies. In one embodiment, the invention provides a bacterial ARG (bARG) derived from Serratia sp. 39006 that produces 9-fold stronger non-linear acoustic contrast compared to previous bacterial ARGs when expressed in Escherichia coli. In another embodiment, the invention provides a mammalian ARG (mARG) derived from Anabaena flos-aquae that produces 38-fold stronger non-linear contrast compared to previous mammalian ARGs when expressed in human cells.  

The ARGs of the invention enable non-destructive, real-time ultrasound imaging of genetically modified cells in vivo at depths greater than 1 cm. The bARG embodiment allows visualization of probiotic bacteria colonizing tumors, while the mARG embodiment enables monitoring of tumor cell gene expression patterns and ultrasound-guided biopsy of genetically defined cell populations. The invention further provides methods of using these ARGs for various research and clinical applications requiring non-invasive cellular imaging.  

## DETAILED DESCRIPTION  

### Definitions  
As used herein, the following terms have the following meanings:  

"Acoustic reporter gene" or "ARG" refers to a genetic construct that encodes proteins capable of producing contrast detectable by ultrasound imaging when expressed in cells.  

"Gas vesicle" or "GV" refers to a gas-filled protein nanostructure naturally produced by certain microorganisms that scatters ultrasound waves.  

"Non-linear contrast" refers to ultrasound signals generated through non-linear oscillation or deformation of contrast agents in response to acoustic pressure, enabling their specific detection against background tissue signals.  

"xAM" refers to cross-propagating amplitude modulation, an ultrasound pulse sequence that enhances detection of non-linear contrast agents while suppressing linear background scattering.  

### Acoustic Reporter Genes for Nondestructive In Vivo Imaging  
The invention provides two principal embodiments of improved ARGs:  

1. A bacterial ARG (bARG_Ser) derived from the GV gene cluster of Serratia sp. 39006 (NCBI: txid104623), comprising genes gvpA, gvpN, gvpJ, gvpK, gvpF, gvpG, gvpW, gvpV, and optionally excluding gene Ser39006_001280. This construct demonstrates 9-fold stronger non-linear ultrasound contrast compared to previous bacterial ARGs when expressed in E. coli.  

2. A mammalian ARG (mARG_Ana) derived from the GV gene cluster of Anabaena flos-aquae (NCBI: txid315271), comprising genes gvpA (provided in stoichiometric excess), gvpN, gvpJ, gvpK, gvpF, gvpG, gvpW, and gvpV, while excluding gvpC. This construct demonstrates 38-fold stronger non-linear contrast compared to previous mammalian ARGs when expressed in human cells.  

Both ARG embodiments are optimized for robust expression at 37°C and produce strong non-linear ultrasound contrast detectable using non-destructive imaging sequences like xAM. The bARG_Ser is particularly suited for expression in probiotic bacteria such as E. coli Nissle 1917 and attenuated Salmonella strains, while mARG_Ana is optimized for expression in mammalian cells including cancer cell lines.  

### Methods of Imaging and Treatment  
The invention provides methods for using the disclosed ARGs in various applications:  

For bacterial imaging, the method comprises: (a) genetically modifying bacteria with bARG_Ser; (b) administering the modified bacteria to a subject; (c) allowing bacterial colonization of target tissues; (d) inducing ARG expression; and (e) imaging bacterial distribution using non-linear ultrasound sequences.  

For mammalian cell imaging, the method comprises: (a) genetically modifying cells with mARG_Ana; (b) administering the modified cells to a subject or growing them in vitro; (c) inducing ARG expression; and (d) imaging cell distribution and gene expression patterns using non-linear ultrasound.  

For ultrasound-guided biopsy, the method comprises: (a) identifying mARG_Ana-expressing cells in a tissue using non-linear ultrasound; (b) targeting biopsy instruments to the imaged region in real-time; and (c) collecting tissue samples from the genetically defined region.  

## EXAMPLES  

### Example 1  
**Construction and Characterization of bARG_Ser in E. coli**  

The Serratia sp. 39006 GV gene cluster was cloned into a pBAD vector with L-arabinose-inducible expression. After transformation into E. coli BL21(DE3), cells were grown on dual-layer LB agar plates containing varying concentrations of L-arabinose inducer. Ultrasound imaging using xAM pulse sequences at 1.74 MPa revealed strong non-linear contrast from induced cultures, with signal detectable at inducer concentrations as low as 0.001% L-arabinose.  

Phase contrast microscopy and transmission electron microscopy confirmed abundant GV formation in induced cells. The construct demonstrated stable expression over multiple generations with less than 0.3% loss-of-function mutations observed after 35 generations. Ultrasound signal was detectable at cell densities as low as 10^7 cells/mL using non-destructive xAM imaging and 10^5 cells/mL using destructive BURST imaging.  

## Materials and Methods  

### Genomic Mining of ARG Clusters  
GV gene clusters were identified from 288 known GV-producing organisms through literature review and sequence analysis. Phylogenetic diversity was maximized by selecting clusters from organisms spanning different habitats (halophilic, thermophilic, mesophilic) and taxonomic groups. Fifteen representative clusters were cloned and screened for heterologous expression in E. coli.  

### Bacterial Plasmid Construction and Molecular Biology  
Selected GV gene clusters were PCR-amplified from genomic DNA and cloned into pET28a(+) vectors using Gibson assembly. Modifications included deletion of non-essential genes (e.g., Ser39006_001280) and addition of toxin-antitoxin stability cassettes (Axe-Txe). Constructs were verified by Sanger sequencing before transformation into bacterial hosts.  

### In Vitro Bacterial Expression of ARGs  
Transformed bacteria were grown in LB media with appropriate antibiotics and inducers (IPTG, aTc, or L-arabinose). For solid media expression, cells were plated on dual-layer LB agar plates with inducers in the bottom layer. Liquid cultures were induced at mid-log phase (OD600 0.1-0.3) and harvested at various timepoints for analysis.  

### In Vitro Ultrasound Imaging of Bacteria Expressing ARGs on Solid Media  
Bacterial patches on agar plates were imaged using a custom robotic ultrasound system with xAM pulse sequences (transmit frequency 18 MHz, receive frequency 9 MHz). Images were analyzed for signal-to-background ratio (SBR) using custom MATLAB scripts.  

### In Vitro Ultrasound Imaging of Bacteria Expressing ARGs Suspended in Agarose Phantoms  
Induced bacterial cultures were normalized to specific cell densities (10^5-10^9 cells/mL), suspended in 1% low-melt agarose, and cast in phantom wells. Phantoms were imaged using xAM and BURST sequences at varying acoustic pressures (0.3-3 MPa).  

### Microscopy of Bacteria  
GV expression was confirmed by phase contrast microscopy (Zeiss Axiocam, 40× objective) and transmission electron microscopy (FEI Tecnai T12, 120 kV). Samples were prepared without staining to preserve GV structure.  

### In Vivo Bacterial ARG Expression and Ultrasound Imaging  
MC26 tumor-bearing mice were intravenously injected with EcN bacteria expressing bARG_Ser. After 3 days, GV expression was induced by intraperitoneal L-arabinose injection. Tumors were imaged 24 hours post-induction using pAM and xAM sequences (Vevo 3100, Fujifilm VisualSonics). Signal was localized to tumor cores at tissue depths >1 cm.  

### Histology of Tumors Colonized by Bacteria  
Following ultrasound imaging, tumors were resected, fixed in formalin, and sectioned for H&E staining and anti-E. coli immunohistochemistry (Opal IHC kits). Histology confirmed bacterial localization to viable tumor regions adjacent to necrotic areas.  

### Mammalian Plasmid Construction  
The A. flos-aquae GV gene cluster was codon-optimized and synthesized with WPRE-hGH polyA elements. Polycistronic constructs were created using P2A peptide linkers between assembly factor genes (gvpNJKFGWV). The gvpA gene was provided on a separate plasmid for stoichiometric tuning. Constructs were cloned into PiggyBac transposon vectors for genomic integration.  

This concludes the detailed description of the invention. While particular embodiments have been described, the invention is capable of variations and modifications apparent to those skilled in the art.