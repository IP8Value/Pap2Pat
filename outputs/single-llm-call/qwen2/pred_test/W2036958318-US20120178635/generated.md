# DESCRIPTION

## BACKGROUND

Chromosomal translocations play a significant role in various genetic diseases. These translocations can lead to the constitutive expression or repression of genes, resulting in different diseases. A notable example is the Philadelphia chromosome (Ph), which is a result of a balanced reciprocal translocation between chromosomes 9 and 22, leading to the formation of the BCR-ABL1 fusion gene. This fusion gene is constitutively expressed and is associated with chronic myeloid leukemia (CML). The breakpoints in the ABL1 gene are typically located in a 90-kb-long intron 1, while the breakpoints in BCR are mapped to a 5.8-kb area spanning exons 12 to 16, known as the major breakpoint cluster region (M-bcr).

The incidence of CML is approximately 1 to 2 per 100,000 individuals, and it constitutes 15 to 20% of adult leukemias. Diagnosis of CML is primarily established through the detection of the Ph or BCR-ABL1 transcripts. Patients in the chronic phase of CML are often treated with inhibitors of BCR-ABL1 tyrosine kinase, such as imatinib mesylate. Continuous monitoring is essential to evaluate the response to therapy and to ensure that the disease does not recur.

Current diagnostic methods for CML include karyotyping and fluorescent in situ hybridization (FISH). Karyotyping requires cells undergoing mitosis, making it necessary to culture cells for several days. FISH, on the other hand, can be applied to nondividing cells and is useful for detecting the BCR-ABL1 translocation. However, neither method provides a sensitive and convenient molecular biomarker for follow-up during treatment. Real-time reverse transcription PCR (RT-PCR) is the most sensitive technique for detecting BCR-ABL1 transcripts, but it is affected by the quality and efficiency of RNA extraction and reverse transcription.

There is a need for a method that can quickly and robustly characterize specific translocations and produce DNA-based disease-specific biomarkers. Such a method should be applicable to nondividing cells and provide a stable and sensitive marker for disease monitoring.

## SUMMARY OF THE INVENTION

The present invention provides a method for detecting and monitoring the BCR-ABL1 translocation based on a screen for the DNA breakpoint. The method, termed Anchored ChromPET, combines three critical techniques: capture of a targeted region to selectively enrich the region of interest, chromosomal paired-end tag (chromPET) sequencing to interrogate the genomic locus, and bar-coding to multiplex multiple samples into a single ultra-high-throughput sequencing lane.

The method involves the following steps:
1. **Library Construction**: Genomic DNA is sheared, and adapters containing bar-codes are ligated to the ends of the DNA fragments.
2. **RNA Bait Preparation**: A biotinylated RNA bait is prepared to target the major breakpoint cluster region (M-bcr) in the BCR gene.
3. **Capture and Enrichment**: The chromPET library is hybridized to the RNA bait, and the target region is captured and enriched using streptavidin beads.
4. **Sequencing**: The enriched library is sequenced using paired-end sequencing.
5. **Bioinformatics Analysis**: The sequencing data is processed to identify junctional chromPETs that map across the BCR and ABL1 loci.
6. **Breakpoint Prediction**: An algorithm is used to predict the exact breakpoints based on the mapping coordinates of the junctional chromPETs.
7. **Validation**: The predicted breakpoints are validated by PCR and sequencing.

The Anchored ChromPET method provides a high-resolution digital karyotype with better sensitivity than comparable methods for detecting the DNA translocation. It can be used to identify the exact DNA junction at the base-pair level, making it a valuable tool for the diagnosis and follow-up of diseases such as CML that are caused by specific chromosomal translocations.

## DETAILED DESCRIPTION OF THE INVENTION

### Abbreviations and Acronyms

- B-ALL: B-cell acute lymphoblastic leukemia
- BP: Base pair
- CHROMPET: Chromosomal paired end tag
- CML: Chronic myeloid leukemia
- FFPE: Formalin-fixed, paraffin-embedded
- FISH: Fluorescent in situ hybridization
- M-BCR: Major breakpoint cluster region
- PET: Paired-end tag
- PH: Philadelphia chromosome
- PS: Patient sample
- RT-PCR: Real-time reverse transcription PCR

## DEFINITIONS

- **ChromPET**: Chromosomal paired-end tag, a sequencing technology that generates paired-end reads from genomic DNA fragments.
- **Anchored ChromPET**: A method that combines chromPET sequencing with targeted capture of a specific genomic region to identify chromosomal translocations.
- **Bar-coding**: The process of adding a unique sequence identifier to each DNA fragment to enable multiplexing of multiple samples in a single sequencing run.
- **Junctional chromPETs**: ChromPETs that map across the junction between two different genomic regions, indicating a chromosomal translocation.
- **Breakpoint**: The specific location in the genome where a chromosomal translocation occurs.

## EMBODIMENTS

### Library Construction

Genomic DNA is extracted from the sample and sheared to produce fragments of approximately 0.5 kb. Adapters containing bar-codes are ligated to the ends of the DNA fragments. The bar-codes allow for the multiplexing of multiple samples in a single sequencing run. The library is then amplified by PCR to increase the amount of DNA for sequencing.

### RNA Bait Preparation

A biotinylated RNA bait is prepared to target the major breakpoint cluster region (M-bcr) in the BCR gene. The M-bcr region is amplified from normal lung genomic DNA using PCR and converted into a biotinylated RNA bait by in vitro transcription.

### Capture and Enrichment

The chromPET library is hybridized to the biotinylated RNA bait, and the target region is captured and enriched using streptavidin beads. The beads are washed to remove non-specifically bound DNA, and the enriched library is eluted and converted to double-stranded DNA.

### Sequencing

The enriched library is sequenced using paired-end sequencing on an Illumina Genome Analyzer. The sequencing data is processed to identify junctional chromPETs that map across the BCR and ABL1 loci.

### Bioinformatics Analysis

The sequencing data is processed using a bioinformatics pipeline to identify junctional chromPETs. The pipeline includes the following steps:
1. **Barcode Assignment**: The 4-bp barcode is used to assign each chromPET to a specific sample.
2. **Mapping**: The 38-bp paired-end reads are mapped to the targeted regions using the Novoalign program.
3. **Classification**: The chromPETs are classified into normal chromPETs (mapping BCR-BCR and ABL1-ABL1) and junctional chromPETs (BCR-ABL1 or ABL1-BCR).

### Breakpoint Prediction

An algorithm is used to predict the exact breakpoints based on the mapping coordinates of the junctional chromPETs. The algorithm uses a voting procedure to determine the most likely location of the breakpoint. The normal chromPETs are used to estimate the average and standard deviation of fragment lengths. Each tag of a junctional chromPET votes on the likely location of the breakpoint, and the region with the maximum votes contains the predicted breakpoint.

### Validation

The predicted breakpoints are validated by designing PCR primers to amplify the junctional fragment and sequencing the amplified product. The sequence of the amplified product confirms the predicted breakpoint.

## EXAMPLES

### Reagents

- APex Heat-Labile Alkaline Phosphatase (Epicentre, Madison, WI, USA; AP49010)
- Biotin-16-UTP (Roche, Indianapolis, IN, USA; 11388908910)
- DNAZol reagent (Invitrogen, Carlsbad, CA, USA; 10503-027)
- Dynabeads M-280 streptavidin (Invitrogen; 112-05D)
- End-It DNA End Repair Kit (Epicentre; ER0720)
- Human Cot-1 DNA (Invitrogen; 15279-011)
- MAXIscript Kit (Ambion, Austin, TX, USA; AM1312)
- MinElute Reaction Cleanup Kit (Qiagen, Valencia, CA, USA; 28204)
- pCR4-TOPO-TA vector (Invitrogen; K4575-01)
- QIAquick Gel Extraction Kit (Qiagen; 28704)
- QIAquick PCR Purification Kit (Qiagen; 28104)
- QuickExtract FFPE DNA Extraction Kit (Epicentre; QEF81805)
- QuickExtract FFPE RNA Extraction Kit (Epicentre; QFR82805)
- Quick Ligation Kit (NEB, Ipswich, MA, USA; M2200S)
- SuperScript III Reverse Transcriptase (Invitrogen; 18080-093)
- TaKaRa Ex Taq DNA Polymerase (Takara, Otsu, Shiga, Japan; TAK RR001A)
- Taq DNA Polymerase (Roche; 11146165001)
- TRIzol (Invitrogen; 15596-026)
- TURBO DNase (Ambion; AM2238)

### Table 1 (Comprising Tables 1A and 1B). Number of ChromPETs Sequenced, Mapped, Anchored to BCR and Junctional for Each Sample (A) Cell Lines and (B) Patient Samples

| Sample | Total ChromPETs Sequenced | ChromPETs Mapped to BCR or ABL1 | ChromPETs Anchored to BCR | Junctional ChromPETs (BCR-ABL1 or ABL1-BCR) |
|--------|--------------------------|---------------------------------|---------------------------|--------------------------------------------|
| K562   | 3,200,000                | 21,798                          | 1,000                     | 23                                         |
| KU812  | 3,200,000                | 403                             | 18                        | 18                                         |
| PS1    | 500,000                  | 89,316                          | 2,000                     | 23                                         |
| PS2    | 500,000                  | 23,456                          | 1,000                     | 15                                         |
| PS3    | 500,000                  | 12,345                          | 500                       | 0                                          |

## CONCLUSIONS

The Anchored ChromPET method provides a high-resolution digital karyotype with better sensitivity than comparable methods for detecting the DNA translocation. It can be used to identify the exact DNA junction at the base-pair level, making it a valuable tool for the diagnosis and follow-up of diseases such as CML that are caused by specific chromosomal translocations. The method is applicable to nondividing cells and provides a stable and sensitive marker for disease monitoring. The use of bar-coding allows for the multiplexing of multiple samples in a single sequencing run, reducing costs and increasing throughput. The Anchored ChromPET method is expected to find wide application in the diagnosis and management of new cases of CML and other diseases characterized by chromosomal translocations.