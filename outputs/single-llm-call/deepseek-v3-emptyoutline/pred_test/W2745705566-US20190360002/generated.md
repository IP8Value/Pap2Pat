Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of genetic engineering and molecular biology, specifically to novel CRISPR-Cas systems derived from thermophilic microorganisms. More particularly, the invention provides an RNA-guided DNA endonuclease (ThermoCas9) isolated from Geobacillus thermodenitrificans T12, along with methods and compositions for genome editing and gene regulation across a wide temperature range. The ThermoCas9 system demonstrates activity between 20°C and 70°C, enabling its use in both thermophilic and mesophilic organisms where conventional CRISPR-Cas systems are ineffective.  

## BACKGROUND TO THE INVENTION  

CRISPR-Cas systems have revolutionized genetic engineering by enabling precise genome editing in diverse organisms. However, current CRISPR-Cas tools derived from mesophilic organisms exhibit limited functionality at elevated temperatures, restricting their application in thermophilic microorganisms and high-temperature environments. The inability to perform genome editing in thermophiles represents a significant technological gap, as these organisms possess considerable potential for industrial biotechnology applications due to their ability to thrive under harsh conditions.  

Prior attempts to adapt mesophilic Cas9 proteins for high-temperature applications through protein engineering have proven challenging. Furthermore, existing thermophilic CRISPR-Cas systems belong primarily to Class 1, which are complex multi-protein systems less suitable for genome engineering applications compared to the simpler Class 2 systems. The discovery and characterization of a thermostable Type II CRISPR-Cas system from Geobacillus thermodenitrificans T12 addresses these limitations by providing a robust genome editing platform functional across a broad temperature spectrum.  

## SUMMARY OF THE INVENTION  

The present invention provides ThermoCas9, a thermostable RNA-guided DNA endonuclease derived from Geobacillus thermodenitrificans T12, along with associated compositions and methods for genome editing. Key aspects of the invention include:  

1. A Cas9 protein (ThermoCas9) exhibiting nuclease activity between 20°C and 70°C, with optimal activity at thermophilic temperatures.  
2. Identification of the protospacer adjacent motif (PAM) sequence requirements for ThermoCas9 activity, including an 8-nucleotide PAM sequence.  
3. Thermostable single guide RNA (sgRNA) constructs that enhance the stability and activity of ThermoCas9 at elevated temperatures.  
4. Methods for genome editing in thermophilic microorganisms using ThermoCas9, demonstrated in Bacillus smithii ET 138 at 55°C.  
5. Application of ThermoCas9 for genome editing in mesophilic organisms, demonstrated in Pseudomonas putida KT2440 at 37°C.  
6. A catalytically inactive variant of ThermoCas9 (ThermodCas9) for gene silencing applications in thermophiles.  

The ThermoCas9 system represents the first CRISPR-Cas9 genome engineering tool specifically adapted for thermophilic organisms, while maintaining functionality in mesophilic systems. This dual functionality provides unprecedented flexibility for genetic manipulation across diverse temperature conditions.  

## RNA Guides and Target Sequences  

The ThermoCas9 system utilizes an RNA guide comprising a CRISPR RNA (crRNA) and trans-activating CRISPR RNA (tracrRNA), which may be combined into a single guide RNA (sgRNA) construct. The sgRNA comprises a spacer sequence complementary to the target DNA (protospacer) and scaffold elements essential for ThermoCas9 binding and activity.  

### Binding, Cleavage, Marking and Modifying Temperatures  

ThermoCas9 demonstrates binding and cleavage activity across an exceptionally broad temperature range of 20°C to 70°C. The system maintains robust activity at thermophilic temperatures (55-65°C) where conventional Cas9 proteins denature. The temperature range can be divided into distinct operational zones:  

- Low temperature zone (20-30°C): Activity is restricted to targets with optimal PAM sequences  
- Moderate temperature zone (37-45°C): Activity expands to include targets with suboptimal PAM sequences  
- High temperature zone (50-70°C): Maximum activity across diverse PAM sequences while maintaining target specificity  

This temperature-dependent activity profile enables precise control over editing efficiency and specificity by adjusting reaction temperatures.  

### Functional Moieties  

The ThermoCas9 system comprises several functional components:  

1. The ThermoCas9 protein containing RuvC and HNH nuclease domains adapted for thermostability  
2. RNA guide components including:  
   - A spacer sequence (19-30 nucleotides) complementary to the target DNA  
   - A repeat-anti-repeat region essential for ThermoCas9 binding  
   - Scaffold elements including three stem-loop structures that confer thermostability  
3. Target DNA sequences containing:  
   - A protospacer matching the guide RNA spacer  
   - A PAM sequence (5'-NNNNCNRA-3') positioned immediately adjacent to the protospacer  

The functional interplay between these components enables targeted DNA cleavage across the full operational temperature range.  

### Nuclease Activity  

ThermoCas9 exhibits RNA-guided double-stranded DNA cleavage activity mediated by its RuvC and HNH nuclease domains. The cleavage activity shows metal ion dependence, with optimal activity in the presence of Mg²⁺ or Mn²⁺ ions. The nuclease maintains strict target specificity, with reduced activity against mismatched targets particularly at lower temperatures.  

The catalytic residues responsible for nuclease activity have been identified (D8 and H582), enabling creation of catalytically inactive variants (ThermodCas9) for gene silencing applications. The nuclease demonstrates enhanced thermostability when complexed with its cognate sgRNA, maintaining activity after incubation at 70°C for 5 minutes.  

## Expression Vectors  

The invention provides expression vectors for implementing the ThermoCas9 system in both thermophilic and mesophilic organisms. Key vector features include:  

1. ThermoCas9 expression cassettes with temperature-appropriate promoters (e.g., PxylL for thermophiles, Pm for mesophiles)  
2. sgRNA expression modules with constitutive promoters (e.g., Ppta)  
3. Homologous recombination templates for genome editing applications  
4. Selectable markers appropriate for the target organism  

Vectors have been demonstrated in both plasmid and chromosomal integration formats, with the pNW33n backbone proving particularly effective for thermophilic applications. The modular design enables customization for specific host organisms and editing applications.  

## Host Cells  

The ThermoCas9 system has been demonstrated in multiple host cell types:  

1. Thermophilic hosts:  
   - Geobacillus thermodenitrificans T12 (native host)  
   - Bacillus smithii ET 138 (demonstrated at 55°C)  

2. Mesophilic hosts:  
   - Pseudomonas putida KT2440 (demonstrated at 37°C)  
   - Escherichia coli (for protein production)  

The system is expected to function in a broad range of additional bacterial hosts, particularly within the Bacillus and Geobacillus genera, as well as other prokaryotes with appropriate genetic tools.  

## DETAILED DESCRIPTION  

### Example 1: Isolation of Geobacillus thermodenitrificans  

Geobacillus thermodenitrificans strain T12 was isolated from compost samples and identified through 16S rRNA sequencing. The strain grows optimally at 65°C and possesses a Type IIC CRISPR-Cas system encoding the ThermoCas9 protein. Genome sequencing revealed the complete ThermoCas9 coding sequence (1082 amino acids) along with associated CRISPR array and tracrRNA elements. Phylogenetic analysis identified closely related Cas9 orthologs in other thermophilic Bacillus and Geobacillus species, confirming the conservation of this system among thermophiles.  

### Example 2: Defining the Essential Consensus Sequences for Cas9 in Geobacillus thermodenitrificans  

The essential consensus sequences for ThermoCas9 function were determined through bioinformatic analysis and experimental validation. Key sequence elements include:  

1. The Cas9 coding sequence containing conserved RuvC (D8) and HNH (H582) catalytic residues  
2. The tracrRNA sequence comprising:  
   - A 36-nucleotide anti-repeat region  
   - Three stem-loop structures critical for thermostability  
3. The crRNA sequence containing:  
   - A 30-nucleotide spacer region  
   - A 36-nucleotide repeat sequence  

The minimal functional sgRNA was determined to be 190 nucleotides, combining these essential elements with a 5'-GAAA-3' linker. Truncation analysis revealed that removal of the 3' stem-loop significantly reduces thermostability and activity at elevated temperatures.  

### Example 3: Identifying Core Amino Acid Motifs which are Essential for the Function of CAS9 and Those which Confer Thermostability in Thermophilic Cas9 Nucleases  

Comparative sequence analysis between ThermoCas9 and mesophilic Cas9 orthologs identified several thermostability-conferring features:  

1. Increased prevalence of charged residues (D, E, K, R) on the protein surface  
2. Higher ratio of arginine to lysine residues  
3. Enrichment of proline residues in loop regions  
4. Compact REC lobe structure compared to mesophilic Cas9 proteins  

Site-directed mutagenesis confirmed the essential catalytic residues (D8 in RuvC domain, H582 in HNH domain) and identified additional residues critical for thermostability. The D8A/H582A double mutant created a catalytically inactive variant (ThermodCas9) suitable for gene silencing applications.  

### Example 4: Determination of the PAM Sequence of G. thermodenitrificans gtCas9  

The protospacer adjacent motif (PAM) requirement for ThermoCas9 was determined through in vitro cleavage assays using target libraries containing randomized PAM sequences. Deep sequencing of cleaved products revealed a consensus 7-nucleotide PAM sequence: 5'-NNNNCNR-3', with preference for cytosine at positions 1, 3, 4, and 6.  

### Example 5: Target Generation with Randomized PAM  

A target DNA library was generated containing:  
- A constant protospacer sequence matching the guide RNA  
- A randomized 7-nucleotide PAM region at the 3' end  

This library served as substrate for in vitro cleavage assays to empirically determine PAM preferences across the operational temperature range.  

### Example 6: In Vitro Determination of PAM Sequences for gtCas9  

In vitro cleavage assays were performed at temperatures ranging from 20°C to 65°C using the randomized PAM library. Cleaved products were isolated, sequenced, and compared to the input library to identify enriched PAM sequences. Results demonstrated temperature-dependent PAM flexibility:  

- At 20°C: Strict requirement for 5'-CCCCCCA-3'  
- At 37°C: Tolerance for 5'-CNCCNNA-3'  
- At 55°C: Broad tolerance for 5'-NNNNCNR-3'  

### Example 7: In Silico PAM Prediction for gtCas9  

Bioinformatic analysis of spacers in the native G. thermodenitrificans CRISPR array and potential protospacers in viral/plasmid databases provided preliminary PAM predictions. While limited hits were obtained, the in silico analysis supported the importance of cytosine at position 5 and adenine at position 8 in the PAM sequence.  

### Example 8: Determination of 8 Nucleotide Long PAM Sequences for gtCas9  

Extended PAM analysis revealed the significance of an eighth position nucleotide, with adenine strongly preferred. The complete 8-nucleotide PAM sequence was determined to be 5'-NNNNCNRA-3', with the eighth position adenine enhancing targeting efficiency.  

### Example 9: In Vivo Genome Editing of Bacillus smithii ET138 with gtCas9 and 8 Nucleotide Length PAM Sequences  

The ThermoCas9 system was implemented for genome editing in Bacillus smithii ET138 at 55°C using:  

1. A pNW33n-based vector expressing ThermoCas9 and sgRNA  
2. Homologous recombination templates for pyrF gene deletion  
3. Selection based on ThermoCas9 counterselection  

Successful editing was achieved, with 10% of colonies showing complete pyrF deletion. This represents the first demonstration of CRISPR-Cas9 genome editing in an obligate thermophile at its native growth temperature. Parallel experiments in Pseudomonas putida at 37°C achieved 50% editing efficiency, confirming the system's broad temperature applicability.  

The complete ThermoCas9 system, including expression vectors, sgRNA designs, and protocols for implementation in diverse hosts, provides a comprehensive platform for genome engineering across unprecedented temperature ranges. This technology enables new possibilities for genetic manipulation of thermophilic organisms and expands the operational parameters for CRISPR-based applications in industrial and research settings.