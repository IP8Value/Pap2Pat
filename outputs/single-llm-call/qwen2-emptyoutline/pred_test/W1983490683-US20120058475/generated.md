# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a method for the detection of chromosomal inversions using directional genomic hybridization. Specifically, the invention provides a novel cytogenomics-based methodology known as chromatid painting, which utilizes strand-specific hybridization strategies to achieve high-resolution detection of chromosomal inversions on a cell-by-cell basis. This method significantly enhances the ability to detect and characterize inversions, which are among the most challenging structural rearrangements to identify using traditional cytogenetic techniques.

## BACKGROUND OF THE INVENTION

Chromosomal inversions are intra-chromosomal rearrangements that result from two breaks occurring within the same chromosome, followed by re-insertion of the broken segment in the opposite (inverted) orientation. These inversions are among the most difficult structural rearrangements to detect, primarily due to the limitations of traditional cytogenetic methods. Traditional approaches such as banding analysis and fluorescence in situ hybridization (FISH) can only detect large inversions, typically greater than 10 megabases (Mb), and often fail to identify smaller inversions.

The importance of detecting chromosomal inversions lies in their potential association with various diseases, particularly cancers. For instance, the (p13.3q24.3) pericentric inversion on chromosome 16, discovered through transcriptome sequencing, encodes a fusion protein (CBFA2T3-GLIS2) linked to a particularly aggressive subtype of pediatric acute leukemia. Despite the advancements in molecular methods such as paired-end sequencing, the detection and characterization of inversions remain challenging due to issues like possible mis-assembly of reference genomes and the presence of flanking high-density inverted duplications.

There is a clear need for methodologies that provide higher resolution and sensitivity in detecting chromosomal inversions, while retaining the ability to provide an unbiased view of the entire genome on an individual cell basis. The present invention addresses this need by introducing a novel method called directional genomic hybridization, specifically chromatid painting, which combines the high resolution of molecular approaches with the comprehensive genome-wide view of traditional cytogenetics.

## SUMMARY OF THE INVENTION

The present invention provides a method for detecting chromosomal inversions using directional genomic hybridization, specifically chromatid painting. The method involves the following steps:

1. **Cell Culture and Treatment**: Culturing cells in the presence of bromodeoxyuridine (BrdU) and bromodeoxycytidine (BrdC) to label newly synthesized DNA strands during a single round of replication.
2. **Slide Preparation**: Preparing metaphase spreads from the treated cells and treating the slides to render the sister chromatids single-stranded.
3. **Probe Design and Labeling**: Designing and synthesizing single-stranded oligonucleotide probes that are specific to unique sequences along the chromosome and are labeled with fluorescent tags.
4. **Hybridization**: Hybridizing the labeled probes to the single-stranded chromatids.
5. **Signal Detection and Analysis**: Detecting and analyzing the hybridization signals to identify inversions based on the switching of signals between sister chromatids.

The invention also encompasses the use of high-resolution, strand-specific probe sets targeted to previously verified or suspected inversion breakpoints, as well as the application of the method to comparative and evolutionary studies involving closely related species.

## DETAILED DESCRIPTION OF THE INVENTION

### Selection of Sequences:

The selection of sequences for the design of single-stranded oligonucleotide probes is a critical step in the directional genomic hybridization method. The sequences are chosen from the publicly available NCBI genomic database, ensuring that they are unique and non-repetitive. Bioinformatics tools are used to mask repeat sequences and to design probes that are similar in directionality, length, and melting temperature. The probes are tiled to small target regions at specified non-overlapping locations along the length of the chromosome at approximately 1-Mb intervals. For high-resolution targeted inversion detection, larger probe sets covering larger unique regions (approximately 14-63 kb) are designed and synthesized.

### EXAMPLE 2

**Materials and Methods**:

1. **Cell Culture**:
   - Human cells (normal fibroblasts and lymphocytes), Kasumi-4 leukemia cell line, and immortalized thyroid cells (HTori-3) were cultured in complete media containing 5.0 μM 5-bromo-2-deoxyuridine (BrdU) and 1.0 μM 5-bromo-deoxycytidine (BrdC).
   - Cells were blocked in mitosis using Colcemid at a final concentration of 0.1 μg/ml for 2-4 hours.
   - Mitotic cells were harvested and dropped onto slides using standard cytogenetic protocols.

2. **Directional Strand-Specific Probes**:
   - Contiguous DNA sequences from the NCBI genomic database (GRCh37.p2 primary assembly) were downloaded and masked using Genetic Information Research Institute software to remove repeat sequences.
   - Single-stranded oligonucleotide probes were designed to unique sequences using ARRAY Designer (version 4.2) and proprietary software (KromaTiD Inc.).
   - Oligo design criteria included similar directionality, length (approximately 40-mers), and uniform melting temperatures (70.0 ± 5.0 °C).
   - Over 17,000 individual oligos were synthesized, hydrated, and pooled in subsets of 45 for end-labeling with fluorescent dNTP analogs (Cy-3, Fluorescein) using terminal transferase.
   - Pools of 90 individual labeled oligos constituted a probe set, which together spanned relatively short unique regions (approximately 5-14.5 kb).
   - Complete chromosome 3-specific chromatid paint consisted of 190 probe sets.
   - Targeted probe sets to known inversion breakpoints on chromosome 3 (q21; q26) and chromosome 10 (q11.2; q21) were also generated, consisting of 180-200 individual labeled oligos covering larger unique regions (approximately 14-63 kb).

3. **Single-Stranded Hybridization Pre-Treatment**:
   - Slides were incubated in PN buffer (sodium phosphate) for 10 minutes at room temperature, rinsed in phosphate-buffered saline, and dehydrated through an ethanol series (75%, 85%, and 100%) for 2 minutes each.
   - Slides were air-dried, stained with Hoechst 33258 (0.5 μg/ml in 2× sodium citrate; SSC) for 15 minutes in the dark, then rinsed with deionized distilled water (ddH2O).
   - Slides were air-dried, flooded with 2× SSC, coverslipped, and exposed to 365 nm ultraviolet light (UV Stratalinker 2400) for 35 minutes.
   - Slides were rinsed in ddH2O to remove the coverslip, air-dried, and dehydrated in the ethanol series, as above.

4. **Strand-Specific Hybridization**:
   - For each pre-treated slide, a mixture of hybridization buffer, chromatid paint, and ddH2O was prepared and heated to 75 °C for 5 minutes.
   - The mixture was pipetted onto pre-treated slides, which were coverslipped and sealed with rubber cement.
   - Slides were heated at 73 °C for 3 minutes, then transferred to individual hybridization chambers and incubated at 37 °C overnight.
   - After hybridization, the slides were washed five times in 2× SSC at 42 °C for 15 minutes each.
   - Slides were rinsed in PN buffer, counterstained with DAPI/antifade, and coverslipped.

### EXAMPLE 3

**Results**:

1. **Directional Genomic Hybridization Methodology**:
   - Cells incorporating BrdU/BrdC during a single round of replication have sister chromatids unifilarly substituted.
   - Slides are stained with Hoechst 33258, exposed to UV light to nick the DNA at sites of BrdU incorporation, and treated with Exonuclease III to selectively degrade the newly replicated strands.
   - This strategy renders the entirety of each sister chromatid a single-stranded target for subsequent hybridization.
   - Bioinformatics-based approaches were used to design sequence-specific, single-stranded oligonucleotides with similar directionality.
   - Initial probe sets to unique sequences on chromosome 3q produced strand-specific signals at the predicted locations, confirming the similar 5′→3′ direction of the assembled contigs.
   - A complete chromosome 3 chromatid-specific paint was created, consisting of 17,000 individual (single-stranded) oligonucleotides, which hybridized robustly and specifically to one chromatid of the target chromosome.

2. **Genome-Wide Inversion Discovery**:
   - Inverted segments reinsert themselves into chromosomal DNA in the reversed (or opposite) orientation.
   - Strand-specific probes, all possessing the same 5′→3′ directionality, hybridize only to complementary stretches of single-stranded chromatids, revealing inversions as obligatory color switches of signal from one sister chromatid to the other.
   - Ionizing radiation (IR) was used to demonstrate the utility of chromatid painting for discovering novel inversions.
   - Small and large inversions were observed following exposure of human cells to IR (gamma rays).
   - Detection of much smaller inversions is possible with more densely spaced probes, as demonstrated by a simulated 6 kb inversion within a 10 Mb region of chromosome 3q.

3. **High-Resolution Targeted Inversion Detection**:
   - High-resolution, strand-specific probe sets targeted to previously verified or suspected inversion breakpoints can further augment the detection and characterization of inversions.
   - The Kasumi 4 cell line, derived from a patient with chronic myelogenous leukemia (CML), possesses a large q21; q26 inversion on chromosome 3, which was readily identified using targeted probe sets.
   - The RET/PTC1 rearrangement associated with radiation-induced papillary thyroid carcinoma, involving a q11.2; q21 inversion on chromosome 10, was also detected in HTori-3 immortalized human thyroid cell cultures.

### Hybridization:

The hybridization process is a crucial step in the directional genomic hybridization method. It involves the preparation of a hybridization mixture containing the labeled single-stranded oligonucleotide probes and the application of this mixture to the pre-treated slides. The slides are then subjected to a series of heating and incubation steps to ensure optimal hybridization conditions. After hybridization, the slides are washed to remove unbound probes and counterstained with DAPI/antifade to visualize the hybridization signals.

### Detection:

Detection of chromosomal inversions using directional genomic hybridization is based on the switching of hybridization signals between sister chromatids. Inverted segments reinsert themselves into chromosomal DNA in the reversed (or opposite) orientation, causing the sequence-and strand-specific probes to hybridize to the opposite chromatid. This results in a visible color switch of the hybridization signal, which can be detected and analyzed using fluorescence microscopy. The efficiency of probe hybridization is routinely greater than 90%, ensuring reliable and consistent detection of inversions.

### Results:

The directional genomic hybridization method, specifically chromatid painting, has been successfully applied to detect and characterize chromosomal inversions in various cell types and species. The method has demonstrated high resolution and sensitivity in detecting both large and small inversions, providing a valuable tool for genome-wide inversion discovery and targeted inversion detection. The ability to simultaneously detect both intra-chromosomal (inversions) and inter-chromosomal rearrangements (dicentrics, translocations) further enhances the utility of this method in cytogenetic and genomic studies. The application of chromatid painting to closely related hominoid species has also shown promise in comparative and evolutionary studies, highlighting the broad applicability of this innovative technique.