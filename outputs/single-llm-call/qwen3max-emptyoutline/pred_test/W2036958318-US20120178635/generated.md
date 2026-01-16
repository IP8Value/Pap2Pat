# DESCRIPTION

## BACKGROUND

Chromosomal translocations represent a critical class of genomic structural variations that underlie the pathogenesis of numerous genetic disorders, particularly hematologic malignancies and solid tumors. These rearrangements occur when segments of non-homologous chromosomes are erroneously exchanged during DNA repair or replication, often resulting in the formation of chimeric genes with oncogenic potential. One of the most well-characterized and clinically significant translocations is the t(9;22)(q34;q11) reciprocal translocation, which gives rise to the Philadelphia chromosome (Ph)—a hallmark cytogenetic abnormality in chronic myeloid leukemia (CML) and a subset of B-cell acute lymphoblastic leukemias (Ph+ B-ALL). This translocation fuses the breakpoint cluster region (BCR) gene on chromosome 22 with the Abelson murine leukemia viral oncogene homolog 1 (ABL1) gene on chromosome 9, generating the BCR-ABL1 fusion gene. The resulting chimeric protein exhibits constitutive tyrosine kinase activity, driving uncontrolled proliferation and impaired differentiation of hematopoietic progenitor cells.

The clinical management of CML has been revolutionized by the advent of tyrosine kinase inhibitors (TKIs), such as imatinib mesylate, which specifically target the BCR-ABL1 kinase domain. However, the efficacy of these therapies necessitates continuous molecular monitoring to assess treatment response, detect minimal residual disease, and identify early signs of relapse or resistance. Current diagnostic and monitoring strategies rely heavily on a combination of cytogenetic, fluorescence-based, and molecular techniques, each with inherent limitations that compromise sensitivity, specificity, or practical utility in routine clinical settings.

Conventional cytogenetic karyotyping remains the historical gold standard for diagnosing CML, as it directly visualizes the Philadelphia chromosome in metaphase spreads. However, this method requires actively dividing cells, typically obtained from bone marrow aspirates and cultured in vitro for several days to accumulate sufficient mitotic figures. This dependency on cell division renders karyotyping unsuitable for analyzing non-proliferating cells from peripheral blood and introduces delays in diagnosis. Moreover, the resolution of karyotyping is limited to approximately 5 megabases, making it incapable of detecting cryptic or submicroscopic rearrangements that may still produce functional fusion genes.

Fluorescence in situ hybridization (FISH) overcomes some of these limitations by enabling the detection of BCR-ABL1 fusion signals in interphase nuclei, thereby eliminating the need for cell culture. FISH assays using dual-color, dual-fusion probes can identify the juxtaposition of BCR and ABL1 loci with a sensitivity of 0.2% to 0.5%, significantly higher than karyotyping. Nevertheless, FISH is constrained by its reliance on probe design, inability to resolve the exact nucleotide sequence of the breakpoint, and susceptibility to signal degradation due to sample preparation artifacts, particularly in formalin-fixed specimens. Furthermore, FISH does not yield a molecular biomarker that can be easily amplified or quantified for longitudinal monitoring.

Real-time reverse transcription polymerase chain reaction (RT-PCR) represents the most sensitive method currently available for detecting BCR-ABL1 fusion transcripts, with a reported sensitivity of up to 0.001% in optimized conditions. This technique is widely used for monitoring treatment response and guiding therapeutic decisions, including the potential discontinuation of TKI therapy in patients achieving deep molecular remission. However, RT-PCR is fundamentally dependent on the integrity and abundance of RNA, which is inherently labile due to the ubiquitous presence of ribonucleases and the chemical instability of the ribose backbone. RNA degradation during sample collection, storage, or processing can lead to false-negative results, raising critical questions about whether undetectable transcript levels truly reflect the absence of leukemic cells. Indeed, studies have demonstrated the persistence of leukemic DNA even in patients with undetectable BCR-ABL1 transcripts, suggesting that DNA-based markers may provide a more reliable indicator of disease burden.

The need for a robust, high-resolution, DNA-based method to characterize chromosomal translocation breakpoints has long been recognized. Precise identification of the genomic breakpoint not only confirms the diagnosis but also enables the design of patient-specific PCR assays for highly sensitive and specific monitoring of minimal residual disease. Such personalized biomarkers are particularly valuable given the known genetic heterogeneity in breakpoint locations among CML patients, which may influence disease progression, response to therapy, and risk of relapse. However, traditional methods for sequencing translocation junctions—such as inverse PCR, ligation-mediated PCR, or long-range PCR with multiple primer sets—are labor-intensive, low-throughput, and often restricted to predefined regions of the genome.

Recent advances in next-generation sequencing (NGS) have opened new avenues for comprehensive genomic analysis, including the detection of structural variants. Paired-end tag (PET) sequencing, in particular, has proven effective in identifying fusion genes and chromosomal rearrangements by capturing the ends of DNA fragments and mapping their genomic origins. Chromosomal PET (ChromPET) extends this approach to the genomic DNA level, enabling the detection of structural variations without reliance on transcriptional activity. However, whole-genome ChromPET sequencing remains prohibitively expensive and inefficient for targeted clinical applications, especially when the region of interest—such as the major breakpoint cluster region (M-bcr) in BCR—is relatively small compared to the entire genome.

To address these challenges, there exists a compelling need for a method that combines the high resolution and digital nature of sequencing with the cost-effectiveness and specificity of targeted enrichment. Such a method should enable the precise identification of translocation breakpoints at the nucleotide level, generate stable DNA-based biomarkers suitable for long-term monitoring, and be applicable to diverse clinical sample types—including peripheral blood, formalin-fixed paraffin-embedded (FFPE) tissues, and even cell-free nucleic acids in body fluids. Furthermore, the method should support high-throughput multiplexing to facilitate the simultaneous analysis of multiple patient samples, thereby reducing costs and turnaround time. The invention described herein fulfills these unmet needs by introducing Anchored ChromPET, a novel integrated platform that synergistically combines targeted capture, barcoded library construction, and high-throughput paired-end sequencing to achieve sensitive, specific, and scalable detection of disease-defining chromosomal translocations.

## SUMMARY OF THE INVENTION

The present invention provides a method for detecting and characterizing chromosomal translocations associated with genetic diseases, particularly hematologic malignancies such as chronic myeloid leukemia (CML) and B-cell acute lymphoblastic leukemia (B-ALL). The method, termed Anchored ChromPET, enables the precise identification of DNA breakpoints at single-nucleotide resolution, thereby generating patient-specific DNA biomarkers that can be used for diagnosis, prognosis, and longitudinal monitoring of disease. The invention overcomes the limitations of existing techniques—such as karyotyping, FISH, and RT-PCR—by providing a DNA-based, high-resolution, and highly sensitive approach that does not require cell culture or depend on RNA integrity.

In one aspect, the invention comprises a method for detecting a chromosomal translocation in a biological sample, the method comprising: (a) preparing a chromosomal paired-end tag (ChromPET) library from genomic DNA isolated from the biological sample, wherein the ChromPET library comprises DNA fragments ligated to barcoded Y-shaped adapters; (b) hybridizing the ChromPET library to a biotinylated RNA bait that is complementary to a target genomic region known to be involved in the translocation, wherein the target genomic region is, for example, the major breakpoint cluster region (M-bcr) of the BCR gene; (c) capturing the hybridized complexes on streptavidin-coated magnetic beads; (d) eluting the captured DNA fragments and amplifying them by PCR; and (e) subjecting the amplified products to high-throughput paired-end sequencing to obtain sequence reads that are mapped to the reference genome to identify junctional ChromPETs that span the translocation breakpoint.

In another aspect, the invention provides a bioinformatics pipeline for analyzing the sequencing data to predict the exact location of the translocation breakpoint. The pipeline includes steps for demultiplexing samples based on barcode sequences, aligning paired-end reads to the target genomic regions, classifying ChromPETs as normal or junctional, and applying a voting algorithm that integrates fragment length distribution and mapping coordinates to pinpoint the most probable breakpoint location. The predicted breakpoint is then used to design patient-specific PCR primers for validation and subsequent monitoring.

The invention further encompasses the use of the identified DNA breakpoint as a personalized biomarker for minimal residual disease detection. Unlike RNA-based biomarkers, the DNA junctional fragment is highly stable and can be reliably detected in a variety of sample types, including formalin-fixed paraffin-embedded (FFPE) tissues and cell-free DNA in serum or other body fluids. The method demonstrates sensitivity comparable to or exceeding that of RT-PCR under suboptimal conditions, such as RNA degradation or low tumor cellularity.

In yet another aspect, the invention provides a cost-effective and scalable platform for multiplexed analysis of multiple patient samples in a single sequencing run. By incorporating unique barcodes into the library adapters, the method allows for the simultaneous processing of up to ten or more samples per lane of an Illumina sequencer, significantly reducing per-sample costs and increasing throughput. This feature is particularly advantageous for clinical laboratories managing large cohorts of patients requiring regular monitoring.

The invention is not limited to the detection of BCR-ABL1 translocations in CML but is broadly applicable to any chromosomal translocation where at least one breakpoint resides within a known genomic region. The RNA bait can be designed to cover larger genomic intervals, such as the entire BCR gene (135 kb), to capture rare breakpoints in the minor breakpoint cluster region (m-bcr) or other alternative sites. Moreover, the method can detect reciprocal translocations (e.g., ABL1-BCR) and complex rearrangements involving deletions or duplications at the breakpoint junction, providing comprehensive insights into the structural architecture of the translocation.

In summary, the present invention offers a transformative approach to the molecular diagnosis and management of translocation-driven cancers. By delivering base-pair resolution of DNA breakpoints, generating stable DNA biomarkers, and enabling high-throughput, cost-effective analysis, Anchored ChromPET bridges the gap between traditional cytogenetics and modern molecular diagnostics, offering significant advantages for both clinical practice and research.

## DETAILED DESCRIPTION OF THE INVENTION

### Abbreviations and Acronyms

The following abbreviations and acronyms are used throughout this patent application and are defined as follows:  
**B-ALL**: B-cell acute lymphoblastic leukemia  
**BP**: base pair  
**ChromPET**: chromosomal paired-end tag  
**CML**: chronic myeloid leukemia  
**FFPE**: formalin-fixed, paraffin-embedded  
**FISH**: fluorescent in situ hybridization  
**M-bcr**: major breakpoint cluster region  
**PET**: paired-end tag  
**Ph**: Philadelphia chromosome  
**PS**: patient sample  
**RT-PCR**: real-time reverse transcription polymerase chain reaction  

These terms are used consistently to ensure clarity and precision in describing the invention, its components, and its applications.

## DEFINITIONS

For the purposes of this patent application, the following terms are defined as indicated:

**"Chromosomal translocation"** refers to a structural rearrangement of the genome in which segments of two or more non-homologous chromosomes are exchanged. This may result in the formation of fusion genes, such as BCR-ABL1, which are associated with specific genetic diseases, particularly cancers.

**"Breakpoint"** denotes the precise genomic location where a chromosomal break occurs and is subsequently joined to a segment from another chromosome. In the context of the BCR-ABL1 translocation, the breakpoint refers to the nucleotide position within the BCR gene on chromosome 22 and the ABL1 gene on chromosome 9 where the fusion occurs.

**"Major breakpoint cluster region (M-bcr)"** is a 5.8-kilobase genomic interval spanning exons 12 to 16 of the BCR gene on chromosome 22q11, where approximately 90% of breakpoints in CML patients are located.

**"Chromosomal paired-end tag (ChromPET)"** refers to a DNA fragment derived from genomic DNA, wherein both ends of the fragment are sequenced and mapped to the reference genome to infer structural variations, such as translocations, inversions, or deletions.

**"Junctional ChromPET"** is a ChromPET in which one end maps to one genomic locus (e.g., BCR) and the other end maps to a different, non-contiguous locus (e.g., ABL1), thereby indicating a chromosomal translocation.

**"Anchored ChromPET"** is a method that combines targeted enrichment of a specific genomic region (the "anchor") with ChromPET sequencing to selectively interrogate translocations involving that region.

**"RNA bait"** is a biotinylated, single-stranded RNA molecule synthesized to be complementary to a target genomic region, used to capture and enrich DNA fragments containing that region from a complex genomic library.

**"Barcoded adapter"** refers to a sequencing adapter that contains a unique short nucleotide sequence (barcode) that allows for the identification and demultiplexing of individual samples after pooled sequencing.

**"Patient-specific DNA biomarker"** is a unique DNA sequence spanning the translocation breakpoint that is specific to an individual patient and can be used for highly sensitive detection of minimal residual disease.

**"Minimal residual disease (MRD)"** refers to the small number of cancer cells that remain in a patient during or after treatment when the patient is in remission, and which may lead to relapse if not eradicated.

These definitions are intended to provide clear and unambiguous interpretation of the claims and descriptions set forth in this patent application.

## EMBODIMENTS

The invention may be embodied in various forms, each representing a specific implementation or application of the core methodology. In one embodiment, the method is used to detect the t(9;22)(q34;q11) translocation in patients suspected of having chronic myeloid leukemia (CML) or Ph-positive B-cell acute lymphoblastic leukemia (Ph+ B-ALL). Genomic DNA is extracted from peripheral blood mononuclear cells, bone marrow aspirates, or other accessible biological sources. A ChromPET library is constructed by fragmenting the DNA, repairing the ends, adding an A-overhang, and ligating barcoded Y-shaped adapters. The library is then hybridized to a biotinylated RNA bait corresponding to the 6.6-kb M-bcr region of the BCR gene. Following capture on streptavidin beads and stringent washing, the enriched DNA is amplified and subjected to paired-end sequencing on an Illumina platform. Bioinformatic analysis identifies junctional ChromPETs that map between BCR and ABL1, and a voting algorithm predicts the exact breakpoint location. Patient-specific PCR primers are designed to amplify and validate the junction, establishing a DNA biomarker for future monitoring.

In another embodiment, the method is adapted for the detection of translocations involving the minor breakpoint cluster region (m-bcr) of the BCR gene, which is associated with certain cases of B-ALL. In this case, the RNA bait is expanded to cover the entire 90-kb intron 1 of BCR, or even the full 135-kb BCR gene, to ensure comprehensive capture of all possible breakpoints. The same workflow is followed, with the enhanced bait enabling the identification of rare or atypical translocations that would be missed by M-bcr-targeted assays.

In a further embodiment, the method is applied to formalin-fixed, paraffin-embedded (FFPE) tissue samples, which are commonly archived in pathology departments. Despite the fragmentation and cross-linking induced by formalin fixation, the DNA-based nature of Anchored ChromPET allows for successful breakpoint identification, as demonstrated by the higher recovery efficiency of DNA compared to RNA from such samples. This embodiment is particularly valuable for retrospective studies and for patients where fresh samples are unavailable.

In yet another embodiment, the method is used to detect cell-free DNA bearing translocation junctions in serum, plasma, or other body fluids. The stability of DNA in circulation, combined with the high sensitivity of the assay, enables non-invasive monitoring of disease burden and early detection of relapse. This liquid biopsy approach represents a significant advancement over current methods that require invasive sampling.

In a multiplexed embodiment, libraries from multiple patients are pooled using unique barcodes and sequenced in a single lane, reducing costs and increasing throughput. The bioinformatics pipeline demultiplexes the data based on barcode sequences, allowing for parallel analysis of up to ten or more samples per run. This embodiment is ideal for clinical laboratories managing large patient cohorts.

Finally, the invention may be extended to other translocation-driven cancers, such as EML4-ALK in non-small cell lung cancer, PML-RARA in acute promyelocytic leukemia, or IGH-MYC in Burkitt lymphoma. In each case, an RNA bait is designed to cover the relevant anchor region (e.g., EML4, PML, or IGH), and the same Anchored ChromPET workflow is applied to identify and characterize the fusion junction.

## EXAMPLES

### Reagents

The following reagents were used in the development and validation of the Anchored ChromPET method: APex Heat-Labile Alkaline Phosphatase (Epicentre, Madison, WI, USA; catalog number AP49010); Biotin-16-UTP (Roche, Indianapolis, IN, USA; catalog number 11388908910); DNAZol reagent (Invitrogen, Carlsbad, CA, USA; catalog number 10503-027); Dynabeads M-280 streptavidin (Invitrogen; catalog number 112-05D); End-It DNA End Repair Kit (Epicentre; catalog number ER0720); human Cot-1 DNA (Invitrogen; catalog number 15279-011); MAXIscript Kit (Ambion, Austin, TX, USA; catalog number AM1312); MinElute Reaction Cleanup Kit (Qiagen, Valencia, CA, USA; catalog number 28204); pCR4-TOPO-TA vector (Invitrogen; catalog number K4575-01); QIAquick Gel Extraction Kit (Qiagen; catalog number 28704); QIAquick PCR Purification Kit (Qiagen; catalog number 28104); QuickExtract FFPE DNA Extraction Kit (Epicentre; catalog number QEF81805); QuickExtract FFPE RNA Extraction Kit (Epicentre; catalog number QFR82805); Quick Ligation Kit (New England Biolabs, Ipswich, MA, USA; catalog number M2200S); SuperScript III Reverse Transcriptase (Invitrogen; catalog number 18080-093); TaKaRa Ex Taq DNA Polymerase (Takara, Otsu, Shiga, Japan; catalog number TAK RR001A); Taq DNA Polymerase (Roche; catalog number 11146165001); TRIzol (Invitrogen; catalog number 15596-026); and TURBO DNase (Ambion; catalog number AM2238). All reagents were used according to the manufacturers’ instructions unless otherwise specified.

### Table 1 (Comprising Tables 1A and 1B). Number of ChromPETs Sequenced, Mapped, Anchored to BCR and Junctional for Each Sample (A) Cell Lines and (B) Patient Samples

**Table 1A. Cell Lines**

| Sample | Total Reads (millions) | Assigned Reads (%) | BCR-Anchored ChromPETs | Junctional ChromPETs (%) |
|--------|------------------------|--------------------|------------------------|--------------------------|
| K562   | 3.2                    | 5%                 | 21,798                 | 4.6%                     |
| KU812  | 3.2                    | 45%                | 403                    | 2.0%                     |

**Table 1B. Patient Samples**

| Sample | Total Reads (millions) | Assigned Reads (%) | BCR-Anchored ChromPETs | Junctional ChromPETs (%) |
|--------|------------------------|--------------------|------------------------|--------------------------|
| PS1    | 0.5                    | 15%                | 89,316                 | 0.03%                    |
| PS2    | 0.5                    | 45%                | 12,450                 | 3.2%                     |
| PS3    | 0.5                    | 6%                 | 1,870                  | <0.1% (dispersed)        |

These data demonstrate the successful application of Anchored ChromPET to both cell line and patient samples, with clear identification of junctional events in known positive cases and absence of a consensus breakpoint in the negative control (PS3).

## CONCLUSIONS

The Anchored ChromPET method represents a significant advancement in the molecular diagnosis and monitoring of chromosomal translocation-associated diseases. By integrating targeted capture, barcoded multiplexing, and high-throughput sequencing, the invention provides a highly sensitive, specific, and cost-effective platform for identifying DNA breakpoints at single-nucleotide resolution. The resulting patient-specific DNA biomarkers are stable, versatile, and amenable to detection in diverse clinical sample types, including peripheral blood, FFPE tissues, and cell-free DNA from body fluids. The method outperforms existing techniques in resolution and reliability, particularly under suboptimal conditions where RNA-based assays fail. Furthermore, its scalability and adaptability to other translocation-driven cancers underscore its broad clinical and research utility. The invention thus establishes a new paradigm for precision oncology, enabling earlier diagnosis, more accurate monitoring, and improved outcomes for patients with translocation-positive malignancies.