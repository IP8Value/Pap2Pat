Here is the patent application following the provided outline:

---

# DESCRIPTION  

## TECHNOLOGY FIELD  

The present invention relates to the field of molecular biology and oncology, specifically to compositions and methods for diagnosing and treating malignant glioma through the modulation of long non-coding RNAs (lncRNAs). More particularly, the invention discloses the identification and therapeutic targeting of lncGRS-1, a lncRNA that sensitizes glioma cells to radiotherapy, as well as novel screening methods utilizing CRISPR interference (CRISPRi) and three-dimensional human brain organoid models.  

## BACKGROUND  

Long non-coding RNAs (lncRNAs) represent a vast class of transcripts exceeding 200 nucleotides in length that do not encode proteins but play critical regulatory roles in cellular processes, including cancer pathogenesis. The human genome encodes thousands of lncRNAs, many of which exhibit highly cell type-specific expression and function, making them attractive candidates for targeted cancer therapy. However, the functional relevance of most lncRNAs remains poorly understood, and systematic approaches to identify lncRNAs that can enhance the efficacy of existing cancer therapies, such as radiation, are lacking.  

Malignant glioma, including glioblastoma (GBM) in adults and diffuse intrinsic pontine glioma (DIPG) in children, is a devastating primary brain cancer with limited treatment options. Despite surgical resection and adjuvant radiotherapy, median survival for GBM patients remains approximately 14 months, while DIPG patients survive only 9–10 months. Radiotherapy, though a cornerstone of glioma treatment, is limited by toxicity to normal brain tissue and the persistence of radiation-resistant tumor cells, leading to recurrence. Therefore, there is an urgent need for therapeutics that selectively enhance radiation sensitivity in glioma cells while sparing normal brain tissue.  

Current high-throughput screening methods for identifying therapeutic targets have primarily focused on protein-coding genes. While CRISPR-based screens have enabled genome-wide functional interrogation, their application to systematically discover lncRNAs that modulate radiation response in glioma has not been explored. Furthermore, conventional preclinical models, such as mouse xenografts, are inadequate for evaluating targets like lncGRS-1, which lacks rodent orthologs, necessitating the development of human-relevant models.  

Organoids, three-dimensional miniature tissue structures that recapitulate key aspects of in vivo organs, have emerged as powerful tools for cancer research. Traditional embryonic brain organoids, however, predominantly consist of immature neural progenitors and lack the cellular diversity of mature brain tissue, limiting their utility for assessing therapeutic toxicity in normal adult brain cells. Thus, there remains a need for improved organoid models that better mimic the mature brain microenvironment for preclinical testing of glioma therapeutics.  

## SUMMARY  

The present invention provides a genome-scale CRISPR interference (CRISPRi) screening platform to systematically identify lncRNAs that sensitize glioma cells to radiotherapy. Using this approach, the inventors discovered lncGRS-1 (CTC-338 M12.4), a primate-conserved lncRNA that, when knocked down, selectively inhibits glioma cell proliferation and enhances radiation sensitivity without harming normal brain cells.  

Key aspects of the invention include:  

1. **CRISPRi Screening Method**: A high-throughput screening method employing CRISPRi to repress lncRNA transcription via catalytically inactive Cas9 (dCas9) fused to the KRAB repressor, enabling the identification of lncRNAs that modify glioma cell response to radiation.  

2. **lncGRS-1 as a Therapeutic Target**: The prioritization of lncGRS-1 as a glioma-specific radiation sensitizer, validated through CRISPRi and antisense oligonucleotide (ASO)-mediated knockdown in patient-derived glioma cells.  

3. **Diagnostic and Therapeutic Methods**: Compositions and methods for diagnosing malignant glioma by detecting lncGRS-1 expression and treating glioma by administering agents that knockdown lncGRS-1, optionally in combination with radiotherapy.  

4. **Three-Dimensional Human Brain Organoid Model**: A novel mature brain organoid (MBO) model assembled from induced astrocytes (iAstrocytes) and neurons (i3Neurons), which supports invasive glioma growth and enables concurrent evaluation of therapeutic efficacy and toxicity.  

5. **Pharmaceutical Compositions**: Formulations comprising ASOs or other nucleic acid-based agents targeting lncGRS-1, optionally combined with pharmaceutically acceptable carriers and/or radiotherapy.  

The invention further encompasses:  
- Methods for identifying radiotherapy sensitizers by exposing test cells to radiation, selecting cells with decreased proliferation, and identifying the targeted lncRNA loci.  
- Methods for diagnosing malignant glioma by hybridizing a sample-derived nucleic acid to a probe complementary to lncGRS-1 and detecting binding.  
- Methods for treating malignant glioma by administering an effective amount of an lncGRS-1-targeting agent, such as an ASO, siRNA, or miRNA, optionally with concurrent radiotherapy.  
- Systems for detecting lncGRS-1, comprising nucleic acid probes and solid supports.  

## DETAILED DESCRIPTION OF EMBODIMENTS  

The following embodiments are provided to illustrate the invention but are not intended to limit its scope.  

### Definitions  

**"Comprising"**: The term "comprising" includes "consisting of" and "consisting essentially of."  

**"Nucleic acid"**: A polynucleotide, including DNA, RNA, and synthetic analogs thereof, such as locked nucleic acids (LNAs) and morpholinos.  

**"Long noncoding RNA (lncRNA)"**: A transcribed RNA molecule >200 nucleotides that does not encode a protein.  

**"Antisense oligonucleotide (ASO)"**: A single-stranded nucleic acid that hybridizes to a target RNA to modulate its expression or function.  

**"CRISPR interference (CRISPRi)"**: A gene repression system utilizing a nuclease-deficient Cas9 (dCas9) fused to a transcriptional repressor (e.g., KRAB) and guided by an sgRNA to a target genomic locus.  

**"Radiotherapy sensitizer"**: An agent that enhances the cytotoxic effects of ionizing radiation on target cells.  

**"Organoid"**: A three-dimensional cell culture model that recapitulates structural and functional features of an organ.  

**"Mature brain organoid (MBO)"**: An organoid comprising postmitotic astrocytes and neurons derived from induced pluripotent stem cells (iPSCs).  

### Method for Identification of a Radiotherapy Sensitizer  

The invention provides a method for identifying lncRNAs that sensitize glioma cells to radiotherapy, comprising:  
1. Providing test cells (e.g., U87-dCas9-KRAB glioma cells) stably expressing dCas9 fused to a transcriptional repressor (e.g., KRAB).  
2. Introducing a library of sgRNAs targeting lncRNA transcriptional start sites into the test cells.  
3. Exposing the cells to fractionated radiation (e.g., 8 Gy delivered in 2 Gy fractions).  
4. Selecting cells exhibiting reduced proliferation relative to controls.  
5. Identifying the lncRNA loci targeted by enriched sgRNAs, thereby identifying radiotherapy sensitizers.  

In one embodiment, the screen identified lncGRS-1 as a top hit, with knockdown synergizing with radiation to inhibit glioma cell growth.  

### Methods of Diagnosis and Treatment  

**Diagnosis**: Malignant glioma is diagnosed by:  
1. Obtaining a biological sample (e.g., tumor biopsy or cerebrospinal fluid) from a subject.  
2. Detecting lncGRS-1 expression by hybridizing the sample RNA to a complementary nucleic acid probe (e.g., SEQ ID NO: 38–46).  
3. Correlating elevated lncGRS-1 levels with glioma diagnosis.  

**Treatment**: Malignant glioma is treated by administering a therapeutically effective amount of an lncGRS-1-targeting agent (e.g., ASO, siRNA) alone or with radiotherapy. In one embodiment, ASOs targeting lncGRS-1 (e.g., SEQ ID NO: 38–46) are administered intrathecally weekly.  

### Pharmaceutical Compositions  

The invention includes pharmaceutical compositions comprising:  
- An lncGRS-1-targeting ASO (e.g., 10–50 nM).  
- A pharmaceutically acceptable carrier (e.g., saline, lipid nanoparticles).  
- Optional adjunctive agents (e.g., chemotherapeutics, radiation modifiers).  

### Three-Dimensional Human Brain Organoid Model  

The MBO model is generated by:  
1. Differentiating iPSCs into iAstrocytes and i3Neurons.  
2. Combining these cells in a 3D culture to form organoids.  
3. Seeding glioma cells (e.g., DIPG SF8628) onto the organoids to model invasive tumor growth.  

This model is useful for screening ASOs or other agents for efficacy and toxicity.  

### Systems  

A system for detecting lncGRS-1 comprises:  
- A nucleic acid probe complementary to lncGRS-1 (e.g., SEQ ID NO: 38–46).  
- A solid support (e.g., microarray, bead) for probe immobilization.  
- Detection reagents (e.g., fluorescent labels).  

## EXAMPLES  

### Example 1: CRISPRi Screen Identifies lncGRS-1  

A genome-scale CRISPRi screen was performed in U87-dCas9-KRAB cells exposed to fractionated radiation (8 Gy in 4 fractions). Screening 5689 lncRNA loci revealed 467 hits modifying radiation response, with lncGRS-1 (CTC-338 M12.4) prioritized due to glioma-specific expression and high sensitizer score.  

### Example 2: lncGRS-1 Knockdown Synergizes with Radiation  

CRISPRi-mediated lncGRS-1 knockdown in U87 cells reduced proliferation by 48% without radiation and synergized with radiation to inhibit growth by 95%. Similar effects were observed in patient-derived GBM and DIPG cells.  

### Example 3: ASO Efficacy in MBOs  

ASOs targeting lncGRS-1 (e.g., SEQ ID NO: 38) knocked down lncGRS-1 by >90% in glioma cells cultured in MBOs, reducing tumor growth and enhancing radiation sensitivity without harming normal organoid cells.  

---  

This patent application provides a comprehensive framework for targeting lncGRS-1 in glioma therapy, supported by innovative screening and preclinical models.