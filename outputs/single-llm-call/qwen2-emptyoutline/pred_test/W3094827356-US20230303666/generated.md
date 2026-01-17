# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of biotechnology, specifically to methods and systems for the in vitro engineering of therapeutic antibodies or binding fragments thereof. More particularly, the invention pertains to an integrated platform for the design, generation, and selection of high-affinity nanobodies (single-domain antibodies) using cell-free ribosome display and computational clustering techniques.

## BACKGROUND

Single-domain antibodies (nanobodies) are derived from the variable heavy chain domains of heavy-chain-only antibodies found in camelids. Due to their small size, high stability, and ease of production, nanobodies have gained significant attention in therapeutic and diagnostic applications. Traditional methods for generating nanobodies often rely on in vivo immunization and phage display, which can be time-consuming and limited in the diversity of the antibody libraries. 

Recent advancements in cell-free protein synthesis and ribosome display have opened new avenues for the rapid and efficient generation of nanobodies. Ribosome display is a powerful technique that allows the in vitro selection of proteins based on their binding properties without the need for cellular transformation. However, existing methods often suffer from issues such as limited library diversity, background noise, and the inability to efficiently recover and characterize binders.

There is a need for an integrated platform that combines the advantages of cell-free displays with advanced computational tools to enhance the efficiency and effectiveness of nanobody engineering. The present invention addresses this need by providing a comprehensive system for the design, generation, and selection of high-affinity nanobodies using cell-free ribosome display and computational clustering.

## SUMMARY

The present invention provides an integrated platform for the in vitro engineering of high-affinity nanobodies (single-domain antibodies) using cell-free ribosome display and computational clustering. The platform, referred to as CeVICA (Cell-Free VHH Integrated Clustering and Analysis), includes the following key components:

1. **Design and Generation of CDR-Randomized VHH Libraries**: The invention involves the design and generation of linear DNA libraries encoding artificial nanobodies with fully randomized complementarity-determining regions (CDRs). The DNA libraries are constructed using a three-stage PCR and ligation process, ensuring high diversity and the inclusion of essential sequence elements for ribosome display.

2. **Optimized Ribosome Display-Based Selection**: The platform employs an optimized ribosome display protocol that links genotype (RNA transcribed from the DNA library) and phenotype (folded nanobody protein tethered to ribosomes). The selection cycles include binding to immobilized targets, reverse transcription-polymerase chain reaction (RT-PCR) of the RNA attached to bound ribosomes, and in vitro transcription/translation for subsequent rounds of display.

3. **Computational Clustering for Binder Prediction**: After selection, the output library is sequenced, and sequences are grouped into clusters based on the similarity of their CDR sequences. Computational clustering identifies unique binding families, enabling the efficient recovery and characterization of high-affinity binders.

4. **Affinity Maturation**: The invention further includes methods for affinity maturation of selected nanobodies using error-prone PCR and additional rounds of ribosome display-based selection. Beneficial mutations are identified and incorporated into the nanobody sequences to enhance binding affinity and neutralization potency.

The CeVICA platform offers a streamlined and scalable approach for the rapid generation of high-affinity nanobodies, making it a valuable tool in the development of therapeutic and diagnostic antibodies.

## DETAILED DESCRIPTION OF THE EXAMPLE EMBODIMENTS

### General Definitions

**Nanobody**: A single-domain antibody derived from the variable heavy chain domain of heavy-chain-only antibodies found in camelids. Nanobodies are characterized by their small size, high stability, and ease of production.

**Complementarity-Determining Regions (CDRs)**: The hypervariable regions of an antibody that are responsible for antigen binding. CDRs are crucial for the specificity and affinity of the antibody.

**Ribosome Display**: An in vitro selection technique that links the genotype (RNA) and phenotype (protein) of a library of proteins by tethering the protein to the ribosome. This allows the selection of proteins based on their binding properties without the need for cellular transformation.

**Computational Clustering**: A bioinformatics technique used to group sequences based on their similarity. In the context of the present invention, computational clustering is used to identify unique binding families from the output library sequences.

### OVERVIEW

The CeVICA platform is an integrated system for the in vitro engineering of high-affinity nanobodies. The platform consists of four main components: (1) design and generation of CDR-randomized VHH libraries, (2) optimized ribosome display-based selection, (3) computational clustering for binder prediction, and (4) affinity maturation. Each component is designed to enhance the efficiency and effectiveness of nanobody engineering, from the initial library design to the final selection and characterization of high-affinity binders.

### Therapeutic Antibodies or Binding Fragments of an Antibody

**Design and Generation of CDR-Randomized VHH Libraries**:
The first step in the CeVICA platform is the design and generation of linear DNA libraries encoding artificial nanobodies with fully randomized CDRs. The DNA libraries are constructed using a three-stage PCR and ligation process, ensuring high diversity and the inclusion of essential sequence elements for ribosome display. The process involves the following steps:

1. **Analysis of Natural Nanobody Sequences**: The design of the nanobody library is guided by the analysis of natural nanobody sequences from the Protein Data Bank (PDB) and abYsis. The analysis reveals the sequence characteristics of the four frame regions and the three CDRs, which are used to define the library design.

2. **Three-Stage PCR and Ligation Process**: The DNA library is constructed in three stages, with each stage randomizing one of the three CDRs. The process involves the use of DNA oligonucleotides with a 5′NNB sequence to introduce randomization in the CDRs and hairpin DNA oligonucleotides to block ligation of one end of the PCR product. The final library contains an upstream T7 promoter, a 3×Myc tag, and a spacer downstream of the nanobody coding region to enable ribosome display.

**Optimized Ribosome Display-Based Selection**:
The second component of the CeVICA platform is the optimized ribosome display-based selection. This process links the genotype (RNA transcribed from the DNA library) and phenotype (folded nanobody protein tethered to ribosomes) to select for high-affinity binders. The selection cycles include the following steps:

1. **Binding to Immobilized Targets**: The displaying ribosome complexes bind to an immobilized target, such as the receptor-binding domain (RBD) of the spike protein of SARS-CoV-2 or enhanced green fluorescent protein (EGFP).

2. **Reverse Transcription-Polymerase Chain Reaction (RT-PCR)**: The RNA attached to the bound ribosomes is reverse-transcribed and amplified by PCR to generate double-stranded DNA.

3. **In Vitro Transcription/Translation**: The double-stranded DNA is then in vitro transcribed and translated to produce a new round of ribosome display.

4. **Sequencing and Clustering**: The double-stranded DNA from any chosen round is sequenced to obtain full-length nanobody sequences. The sequences are then grouped into clusters based on the similarity of their CDR sequences, with each cluster representing a unique binding family.

**Computational Clustering for Binder Prediction**:
The third component of the CeVICA platform is the computational clustering of sequences to predict high-affinity binders. The process involves the following steps:

1. **Sequence Matching and Clustering**: The distribution of sequence match scores (based on BLOSUM62 amino acid pair scores) is analyzed to identify sequences with high similarity in their CDRs. Sequences with a high match score are grouped into clusters.

2. **Cluster Analysis**: The clusters are analyzed to identify unique binding families. Representative sequences from each cluster are selected for further characterization.

**Affinity Maturation**:
The final component of the CeVICA platform is the affinity maturation of selected nanobodies. This process involves the following steps:

1. **Error-Prone PCR**: Error-prone PCR is used to introduce random mutations across the full length of selected nanobody sequences.

2. **Additional Rounds of Ribosome Display-Based Selection**: The mutagenized library is subjected to additional rounds of ribosome display-based selection under stringent conditions to enrich for high-affinity binders.

3. **Identification and Incorporation of Beneficial Mutations**: Beneficial mutations are identified by comparing the amino acid profiles of the pre- and post-affinity maturation libraries. The identified mutations are incorporated into the nanobody sequences to enhance binding affinity and neutralization potency.

### EXAMPLES

**Example 1: Development of CeVICA for In Vitro VHH Domain Antibody Engineering**

**Objective**: To develop an integrated platform (CeVICA) for the in vitro engineering of high-affinity nanobodies using cell-free ribosome display and computational clustering.

**Methods**:
1. **Library Design and Generation**:
   - **Analysis of Natural Nanobody Sequences**: 298 unique camelid nanobody sequences from the Protein Data Bank (PDB) were analyzed to define the sequence characteristics of the four frame regions and the three CDRs.
   - **Three-Stage PCR and Ligation Process**: Linear DNA libraries encoding artificial nanobodies with fully randomized CDRs were constructed using a three-stage PCR and ligation process. The final library contained an upstream T7 promoter, a 3×Myc tag, and a spacer downstream of the nanobody coding region.

2. **Optimized Ribosome Display-Based Selection**:
   - **Binding to Immobilized Targets**: The displaying ribosome complexes were allowed to bind to immobilized targets, such as the RBD of SARS-CoV-2 or EGFP.
   - **RT-PCR and In Vitro Transcription/Translation**: The RNA attached to the bound ribosomes was reverse-transcribed and amplified by PCR to generate double-stranded DNA, which was then in vitro transcribed and translated for subsequent rounds of ribosome display.

3. **Computational Clustering for Binder Prediction**:
   - **Sequence Matching and Clustering**: The distribution of sequence match scores was analyzed to identify sequences with high similarity in their CDRs. Sequences with a high match score were grouped into clusters.
   - **Cluster Analysis**: The clusters were analyzed to identify unique binding families. Representative sequences from each cluster were selected for further characterization.

4. **Affinity Maturation**:
   - **Error-Prone PCR**: Error-prone PCR was used to introduce random mutations across the full length of selected nanobody sequences.
   - **Additional Rounds of Ribosome Display-Based Selection**: The mutagenized library was subjected to additional rounds of ribosome display-based selection under stringent conditions.
   - **Identification and Incorporation of Beneficial Mutations**: Beneficial mutations were identified by comparing the amino acid profiles of the pre- and post-affinity maturation libraries. The identified mutations were incorporated into the nanobody sequences to enhance binding affinity and neutralization potency.

**Results**:
- The CeVICA platform successfully generated a high-diversity nanobody library with fully randomized CDRs.
- The optimized ribosome display-based selection effectively enriched for high-affinity binders.
- Computational clustering identified unique binding families, enabling the efficient recovery and characterization of high-affinity binders.
- Affinity maturation further enhanced the binding affinity and neutralization potency of selected nanobodies.

**Conclusion**:
The CeVICA platform provides a comprehensive and efficient system for the in vitro engineering of high-affinity nanobodies, making it a valuable tool in the development of therapeutic and diagnostic antibodies.

**Example 2: Application of CeVICA for the Identification of SARS-CoV-2 Neutralizing Nanobodies**

**Objective**: To apply the CeVICA platform for the identification of nanobodies that bind to the receptor-binding domain (RBD) of the spike protein of SARS-CoV-2 and neutralize pseudotyped lentiviruses.

**Methods**:
1. **Library Design and Generation**:
   - **Analysis of Natural Nanobody Sequences**: 298 unique camelid nanobody sequences from the Protein Data Bank (PDB) were analyzed to define the sequence characteristics of the four frame regions and the three CDRs.
   - **Three-Stage PCR and Ligation Process**: Linear DNA libraries encoding artificial nanobodies with fully randomized CDRs were constructed using a three-stage PCR and ligation process. The final library contained an upstream T7 promoter, a 3×Myc tag, and a spacer downstream of the nanobody coding region.

2. **Optimized Ribosome Display-Based Selection**:
   - **Binding to Immobilized RBD**: The displaying ribosome complexes were allowed to bind to immobilized RBD of SARS-CoV-2.
   - **RT-PCR and In Vitro Transcription/Translation**: The RNA attached to the bound ribosomes was reverse-transcribed and amplified by PCR to generate double-stranded DNA, which was then in vitro transcribed and translated for subsequent rounds of ribosome display.

3. **Computational Clustering for Binder Prediction**:
   - **Sequence Matching and Clustering**: The distribution of sequence match scores was analyzed to identify sequences with high similarity in their CDRs. Sequences with a high match score were grouped into clusters.
   - **Cluster Analysis**: The clusters were analyzed to identify unique binding families. Representative sequences from each cluster were selected for further characterization.

4. **Affinity Maturation**:
   - **Error-Prone PCR**: Error-prone PCR was used to introduce random mutations across the full length of selected nanobody sequences.
   - **Additional Rounds of Ribosome Display-Based Selection**: The mutagenized library was subjected to additional rounds of ribosome display-based selection under stringent conditions.
   - **Identification and Incorporation of Beneficial Mutations**: Beneficial mutations were identified by comparing the amino acid profiles of the pre- and post-affinity maturation libraries. The identified mutations were incorporated into the nanobody sequences to enhance binding affinity and neutralization potency.

5. **Characterization of Selected Nanobodies**:
   - **ELISA Assay**: ELISA assays were performed to evaluate the binding affinity of selected nanobodies to the RBD of SARS-CoV-2.
   - **Pseudotyped Lentivirus Neutralization Assay**: Pseudotyped lentivirus neutralization assays were performed to evaluate the neutralization potency of selected nanobodies.

**Results**:
- The CeVICA platform successfully identified nanobodies that bind to the RBD of SARS-CoV-2 with high affinity.
- Several nanobodies were found to neutralize SARS-CoV-2 pseudotyped lentiviruses with high potency, including SR6c3 and SR6v15.
- Affinity maturation further enhanced the binding affinity and neutralization potency of selected nanobodies, with SR6v15.d showing an IC50 of 0.329 nM.

**Conclusion**:
The CeVICA platform is effective in the identification and optimization of high-affinity nanobodies that bind to the RBD of SARS-CoV-2 and neutralize pseudotyped lentiviruses. This demonstrates the potential of the CeVICA platform in the development of therapeutic antibodies for viral diseases.