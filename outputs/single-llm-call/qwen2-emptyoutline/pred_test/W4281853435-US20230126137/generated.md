# DESCRIPTION

## BACKGROUND

RNA-binding proteins (RBPs) play a crucial role in regulating gene expression by controlling the rate, location, and timing of RNA maturation. Dysregulation of RBP function is associated with various genetic and somatic disorders, including neurodegeneration and cancer. Technologies such as RNA immunoprecipitation (RIP) and Cross-Linking and ImmunoPrecipitation (CLIP) have facilitated the transcriptome-wide identification of RNA binding sites. The enhanced CLIP (eCLIP) method has been instrumental in generating comprehensive datasets profiling targets for numerous RBPs. However, the technical complexity of the eCLIP protocol, particularly the need for SDS-PAGE and nitrocellulose membrane transfer steps, poses significant challenges in scaling up the process. Additionally, the requirement for separate immunoprecipitation (IP) steps for each RBP increases the input material needed, making it impractical for studying a large number of RBPs simultaneously.

To address these limitations, the present invention introduces a novel method called Antibody-Barcoded eCLIP (ABC). This method utilizes DNA-barcoded antibodies to replace the SDS-PAGE and membrane transfer steps, thereby streamlining the process. The DNA barcodes also enable the simultaneous interrogation of multiple RBPs within a single sample, significantly reducing the input requirement per RBP. This innovation aims to accelerate the characterization of RBPs and facilitate broader applications in clinical and research settings.

## SUMMARY

The present invention provides a method for identifying RNA binding sites of multiple RNA-binding proteins (RBPs) simultaneously using Antibody-Barcoded eCLIP (ABC). The method involves the use of DNA-barcoded antibodies to label the immunoprecipitated protein-RNA complexes, allowing for on-bead proximity-based ligations. This eliminates the need for SDS-PAGE and nitrocellulose membrane transfer steps, simplifying the workflow and reducing the time and labor required. The DNA barcodes also enable the identification of different RBPs within the same sample, thereby reducing the input material needed and facilitating the simultaneous analysis of multiple RBPs.

The invention includes the following steps:
1. **Cell Culture and Crosslinking**: Culturing cells and crosslinking them with UV light to fix the protein-RNA interactions.
2. **Lysis and Immunoprecipitation**: Lysing the cells and performing immunoprecipitation using DNA-barcoded antibodies.
3. **Proximity Ligation**: Performing proximity ligation to attach the DNA barcodes to the immunoprecipitated RNA.
4. **Reverse Transcription and Library Preparation**: Converting the RNA into cDNA and preparing the sequencing library.
5. **Sequencing and Data Analysis**: Sequencing the libraries and analyzing the data to identify the RNA binding sites of the RBPs.

The invention further provides kits and methods for performing the ABC protocol, including reagents, buffers, and detailed protocols for each step of the process. The invention also includes methods for analyzing the data to identify significant peaks and compare the results with existing CLIP-based methods.

## DETAILED DESCRIPTION

### Definitions

- **RNA-binding protein (RBP)**: A protein that binds to RNA and plays a role in various aspects of RNA metabolism, including splicing, transport, localization, and stability.
- **Cross-Linking and ImmunoPrecipitation (CLIP)**: A method used to identify the binding sites of RBPs on RNA by crosslinking the proteins to RNA, immunoprecipitating the complexes, and sequencing the bound RNA.
- **Enhanced CLIP (eCLIP)**: An optimized version of CLIP that uses a standardized protocol to generate high-quality datasets.
- **Antibody-Barcoded eCLIP (ABC)**: A novel method that uses DNA-barcoded antibodies to simplify the eCLIP protocol and enable the simultaneous analysis of multiple RBPs.
- **DNA Barcode**: A short DNA sequence attached to an antibody to label the immunoprecipitated protein-RNA complexes.
- **Proximity Ligation**: A technique used to attach the DNA barcodes to the immunoprecipitated RNA by bringing the barcodes and RNA into close proximity.
- **Library Preparation**: The process of converting the immunoprecipitated RNA into a form suitable for sequencing.
- **Sequencing**: The process of determining the nucleotide sequence of the RNA using high-throughput sequencing technologies.
- **Data Analysis**: The process of analyzing the sequencing data to identify the RNA binding sites of the RBPs and compare the results with existing methods.

### Methods

#### Cell Culture and Crosslinking
Cells are cultured in appropriate media and crosslinked with UV light to fix the protein-RNA interactions. The cells are then harvested, washed, and resuspended in a small volume of cold PBS. The cell suspension is subjected to UV crosslinking (254 nm, 400 mJ/cm²) on ice. After crosslinking, the cells are washed again with cold PBS and pelleted by centrifugation. The cell pellets are flash-frozen on dry ice and stored at -80°C until further use.

#### Oligo Barcoding Prep
A 100 µl sample of 100 µM oligo barcode in PBS is reacted with 10 µl 10 mM azide-NHS in DMSO by rotating at room temperature for 2 hours. Unreacted azide is removed by buffer exchange into PBS using Zeba desalting columns. The azide-labeled barcodes are stored at -20°C.

#### Antibody Barcoding
Antibodies (20 µg) are diluted to 70 µl in PBS and buffer-exchanged into PBS using Zeba desalting columns. Then, 10 µl 10 mM DBCO-NHS is added to the antibodies and allowed to rotate at room temperature for 1 hour. Unreacted DBCO-NHS is removed by buffer exchange into PBS using Zeba desalting columns. The DBCO-labeled antibodies are stored at 4°C. Azide-containing barcodes (6.65 µl) are then reacted with the DBCO-labeled antibodies (70 µl) and allowed to rotate overnight at room temperature. The labeled antibodies are stored at 4°C.

#### Antibody Conjugation Clip
##### IP Bead Conjugation
Lysis buffer (200 µl) is added to 25 µl anti-rabbit Dynabeads, and the beads are washed twice with 500 µl lysis buffer. The beads are resuspended in 50 µl lysis buffer, and 5 µg of the barcoded antibody is added. The mixture is rotated at room temperature for 1 hour. The beads are washed again with 500 µl lysis buffer and resuspended in 50 µl lysis buffer. This procedure is repeated for each barcode and antibody combination.

##### Immunoprecipitation
Frozen cell pellets (10 million cells) are lysed in 1 ml lysis buffer supplemented with protease inhibitor cocktail and RNase inhibitor. The lysate is sonicated for 5 minutes with 30-second on/off cycles at 4°C. The lysate is treated with RNase I and TurboDNase and mixed at 37°C for 5 minutes. Cellular debris is removed by centrifugation at 12,000g for 3 minutes. The supernatant is transferred to a new tube along with 50 µl of each preconjugated antibody-coated magnetic bead (500 µl total for 10plex) and immunoprecipitated overnight by rotation at 4°C. The beads are washed with high salt wash buffer (3×), high salt wash buffer + 80 mM LiCl (1×), and low salt wash buffer (3×).

##### Proximity Ligation
The beads are resuspended in 76 µl T4 PNK reaction buffer, 3 µl T4 PNK, and 1 µl RNase inhibitor and incubated at 37°C for 20 minutes with interval mixing. After PNK treatment, the samples are washed with high salt buffer (1×) and low salt buffer (3×). Proximity barcode ligations are carried out in 150 µl T4 ligation reaction mix at room temperature for 45 minutes with interval mixing. The samples are washed again with high salt buffer (1×) and low salt buffer (2×). Chimeric RNA barcode molecules are eluted from the beads by incubating with 127 µl ProK digestion solution at 37°C for 20 minutes followed by 50°C for 20 minutes with interval mixing. The supernatants are transferred to a clean tube and cleaned up using Zymogen RNA clean and concentrator, eluting in 10 µl.

#### Reverse Transcription and Library Prep
RNA (9 µl) is combined with 1.5 µl reverse transcriptase (RT) primer mix and heated to 65°C for 2 minutes and immediately placed on ice. Then, 9.2 µl RT buffer, 0.2 µl RNase inhibitor, and 0.6 µl Superscript III are added, mixed by pipetting, and reverse transcribed at 54°C for 20 minutes. After RT, excess primers and nucleotides are removed with 2.5 µl ExoSAP-IT at 37°C for 15 minutes, and the RNA is degraded with 1 µl 0.5 M EDTA and 3 µl 1 M NaOH at 70°C for 10 minutes. The sample is pH neutralized with 3 µl 1 M HCl.

#### RT Cleanup
MyOne Silane beads are prepared by adding 5 µl beads to a fresh tube containing 25 µl RLT buffer + 0.01% Tween 20. The tube is placed on a magnet, and the supernatant is removed and replaced with 93 µl RLT buffer + 0.01% Tween 20. The bead preparation is added to the pH-neutralized RT cDNA and incubated at room temperature for 10 minutes. The beads are washed with 300 µl of 80% ethanol twice. The cDNA is eluted in 2.5 µl ssDNA ligation adapter and heated to 70°C for 2 minutes before being placed on ice.

#### ssDNA Ligation
Without removing the beads, 6.5 µl T4 ligase solution, 1 µl T4 ligase, and 0.3 µl deadenylase are added and rotated overnight at room temperature. Bead binding buffer and 45 µl ethanol are added to the ligation mixture to rebind the cDNA to the silane beads for 10 minutes at room temperature. The beads are washed with 300 µl of 80% ethanol twice and allowed to air dry until the beads no longer appear wet. The cDNA is eluted in 25 µl elution buffer.

#### PCR Quantification
cDNA (1 µl) is diluted with 10 µl water, and 1 µl of the diluted cDNA is mixed with 2 µl of each qPCR primer and 5 µl Luna qPCR Master Mix. The samples are processed on a StepOnePlus System. Final libraries are amplified with dual index Illumina primers using Next Ultra II Q5 Master Mix. If necessary, adapter dimers are removed using a Qiagen Gel Extraction kit. The libraries are quantified by Tapestation and sequenced on an Illumina Nextseq 2000, with approximately 25 million reads per barcode (250 million reads for a 10plex).

### Kits
The invention provides kits for performing the ABC protocol, including the following components:
- **Cell Culture Media and Reagents**: DMEM medium, fetal bovine serum (FBS), PBS, protease inhibitor cocktail, RNase inhibitor, RNase I, TurboDNase.
- **Lysis and Wash Buffers**: Lysis buffer, high salt wash buffer, low salt wash buffer.
- **Antibody Barcoding Reagents**: Azide-NHS, DBCO-NHS, Zeba desalting columns.
- **Proximity Ligation Reagents**: T4 PNK, T4 ligase, RNase inhibitor, ProK digestion solution, Zymogen RNA clean and concentrator.
- **Reverse Transcription and Library Prep Reagents**: dNTPs, ABC RT primer, RT buffer, DTT, Superscript III, ExoSAP-IT, MyOne Silane beads, RLT buffer, ssDNA ligation adapter, T4 ligase solution, deadenylase, bead binding buffer, elution buffer, qPCR primers, Luna qPCR Master Mix, dual index Illumina primers, Next Ultra II Q5 Master Mix, Qiagen Gel Extraction kit, Tapestation.

## EXAMPLES

### Example 1
**Evaluation of ABC Singleplex Experiments**
To evaluate the performance of ABC in singleplex mode, we performed duplicate singleplex ABC experiments for the RNA Binding Fox-1 Homolog 2 (RBFOX2) in HEK293XT cells and the Stem-Loop Binding Protein (SLBP) in K562 cells. The library complexity, defined as the number of unique reads, was compared between ABC and eCLIP. The results showed that ABC and eCLIP exhibited similar library complexity, indicating that ABC maintains the efficiency of the original eCLIP method. The read density at known binding sites of RBFOX2 and SLBP was also comparable between ABC and eCLIP, confirming the accuracy of the ABC method.

### Example 3
**Multiplexing Multiple RBPs**
To demonstrate the multiplexing capability of ABC, we selected ten RBPs with diverse known binding preferences and performed triplicate multiplexed ABC experiments in K562 cells. The RBPs included DDX3, EIF3G, IGF2BP2, FAM120A, PUM2, ZC3H11A, LIN28B, SF3B4, and PRPF8. The results showed that ABC generated similar read distributions and peak calling as eCLIP for each RBP. The use of a complement control (CC) derived from the other RBPs in the multiplexed reaction improved the ranking of genic regions known to be preferred by the RBPs, compared to using total RNA-seq as background. The metagene profiles of the enriched peaks for the spliceosomal proteins SF3B4 and PRPF8 were also similar between ABC and eCLIP, further validating the performance of ABC in multiplex mode.

### Example 4
**Comparison of ABC and eCLIP in Clinical Samples**
To assess the applicability of ABC in clinical settings, we applied the method to a limited amount of input material from a patient-derived xenograft (PDX) model of neurodegenerative disease. The results showed that ABC successfully identified the RNA binding sites of multiple RBPs in the PDX sample, demonstrating the potential of ABC for characterizing RBPs in clinically relevant samples where source materials are often limited. The data also showed that ABC performed with comparable sensitivity and specificity to eCLIP, even with reduced input material, highlighting the advantages of the ABC method in clinical research.

The invention thus provides a robust and scalable method for identifying RNA binding sites of multiple RBPs simultaneously, with significant improvements in efficiency and input requirements compared to existing CLIP-based methods. The ABC method is expected to facilitate broader applications in both basic research and clinical settings, contributing to a deeper understanding of the roles of RBPs in gene regulation and disease.