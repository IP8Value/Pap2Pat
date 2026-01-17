# DESCRIPTION

## THE NAMES OF THE PARTIES TO A JOINT RESEARCH AGREEMENT

This invention was made as a result of a joint research agreement between [Institution A] and [Institution B]. The parties to this agreement are [Institution A], located at [Address A], and [Institution B], located at [Address B]. The agreement outlines the collaborative efforts and contributions of both institutions towards the development and commercialization of the invention described herein.

## BACKGROUND OF THE INVENTION

Incisional hernia repair is a significant aspect of general surgical practice, with an incidence ranging from 2 to 11%. Despite advancements in surgical techniques, the recurrence rate remains high, estimated between 10 and 50%. The annual cost for incisional hernia repairs in the United States is approximately $2.5 billion. While the use of prosthetic mesh has reduced recurrence rates, a notable percentage of patients still experience multiple recurrences, ranging from 5 to 20%.

Several risk factors for incisional hernia formation have been identified, including wound infection, abdominal distention, pulmonary complications, male gender, age, and obesity. However, the literature is inconsistent regarding risk factors for recurrent incisional hernias, such as body mass index, ascites, large hernias, continued smoking, occupational lifting, and wound-healing disorders.

Current evidence suggests that incisional hernias are often caused by failures in early surgical wound healing. Collagen I provides tensile strength to connective tissue, while immature collagen III, found in early wounds, is weaker. Studies have shown a decreased collagen I-to-III ratio in patients with direct and indirect hernias compared to controls, indicating a potential high-risk group more susceptible to hernia formation.

Despite the known association of collagen and connective tissue diseases with hernia formation, there is a lack of data on genetic predispositions to hernia formation in otherwise normal patients. The present invention addresses this gap by identifying distinct gene expression profiles in patients with recurrent incisional hernias (RH) compared to normal controls (NC). This identification of genomic profiles aims to provide a deeper understanding of the molecular mechanisms underlying hernia formation and recurrence, potentially leading to the development of targeted therapies and personalized treatment strategies.

## BRIEF SUMMARY OF THE INVENTION

The present invention relates to a method for identifying and characterizing distinct gene expression profiles in patients with recurrent incisional hernias (RH) compared to normal controls (NC). The method involves analyzing skin and fascia samples from patients undergoing laparoscopic repair of recurrent ventral or incisional hernias and comparing them to samples from patients without a history of hernias. The invention specifically identifies the differential expression of genes, particularly GREM1, COL1A1, and COL3A1, which are associated with wound healing and collagen synthesis. These gene expression profiles can serve as biomarkers to stratify patients into different risk groups for hernia development and recurrence, thereby facilitating personalized treatment approaches.

## DETAILED DESCRIPTION OF THE INVENTION

### Abbreviations and Definitions

- **COL1A1**: Collagen, type I, alpha 1
- **COL3A1**: Collagen, type III, alpha 1
- **GREM1**: Gremlin 1, cysteine knot superfamily, homolog (Xenopus laevis)
- **NC**: Normal Control
- **RH**: Recurrent Hernia
- **RNA**: Ribonucleic Acid
- **qPCR**: Quantitative Real-Time Polymerase Chain Reaction
- **PCR Array**: Polymerase Chain Reaction Array
- **GO**: Gene Ontology
- **RIN**: RNA Integrity Number
- **QDA**: Quadratic Discriminant Analysis

### Methods

The invention involves a method for identifying distinct gene expression profiles in patients with recurrent incisional hernias (RH) compared to normal controls (NC). The method includes the following steps:

1. **Patient Selection and Sample Collection**:
   - Patients eligible for the study are 18 years of age or older and have undergone laparoscopic repair of a recurrent ventral or incisional hernia.
   - Patients are excluded if they have a history of steroid use, severe COPD, pulmonary or connective tissue disorders, or are prisoners.
   - Approximately 1 cm² of skin and fascia is removed from the trocar placement site, remote from the hernia or old incisions.
   - Tissue samples are divided and placed in either 10% buffered formalin or RNALater™ RNA Stabilization Reagent (Qiagen, Valencia, CA).
   - Tissue is stored in RNALater™ for up to 48 hours at room temperature.

2. **RNA Isolation and Amplification**:
   - Total RNA is isolated from the skin and fascia specimens using the RNeasy® Lipid Tissue Mini Kit (Qiagen) with a rotor homogenizer and on-column DNase treatment.
   - Total RNA is amplified using the WT-Ovation™ Pico RNA Amplification System protocol (NuGen, San Carlos, CA).

3. **cDNA Labeling, RNA Quantity and Quality, and Microarray**:
   - For each skin and fascia sample, 1.5 μg biotin-labeled, amplified cDNA is hybridized to a Sentrix® Human-6 v.2 Whole Genome Expression BeadChips (Sentrix Human WG-6; Illumina, San Diego, CA).

4. **Validation by Quantitative RT-PCR (qPCR) and PCR Array**:
   - cDNA is generated from 10 ng of the same total RNA samples used for the microarray experiment.
   - qPCR is performed on the StepOne™ Real-Time PCR System (Applied Biosystems, Foster City, CA) using GAPDH as a reference gene.
   - A PCR array, focusing on the expression of 84 key genes related to dysregulated tissue remodeling during wound healing, is performed on the samples using the Human Fibrosis RT2 Profiler™ PCR Array System (SABiosciences, Frederick, MD).

5. **Immunohistochemistry**:
   - Specimens are fixed in 10% buffered formalin, processed, embedded in paraffin, and cut at 4 μm.
   - Immunohistochemistry is performed using the automated horseradish peroxidase Autostainer/Envision Plus method (Dakocytomation, Carpenteria, CA).

6. **Statistical Analysis of Microarray Data**:
   - Analysis of microarray gene expression data is performed using R open-source software.
   - Genes considered "not detectable" across >50% of patient samples are excluded.
   - Differential gene expression analysis is performed using a moderated t-statistic applied to the log2-transformed normalized intensity for each gene.
   - Adjustment for multiple testing is made using the false discovery rate method of Benjamini and Hochberg.
   - Gene ontology (GO) analyses are conducted to identify common functions or descriptive terms that are statistically abundant in the list of differentially expressed genes.

### Kits

The invention also encompasses kits for performing the methods described herein. The kits may include, but are not limited to, the following components:

- **RNA Isolation Kit**: RNeasy® Lipid Tissue Mini Kit (Qiagen)
- **RNA Amplification Kit**: WT-Ovation™ Pico RNA Amplification System (NuGen)
- **cDNA Labeling Kit**: Sentrix® Human-6 v.2 Whole Genome Expression BeadChips (Illumina)
- **qPCR Kit**: SuperScript™ III Platinum® Two-Step qPCR Kit with SYBR® Green (Invitrogen)
- **PCR Array Kit**: Human Fibrosis RT2 Profiler™ PCR Array System (SABiosciences)
- **Immunohistochemistry Kit**: Automated horseradish peroxidase Autostainer/Envision Plus method (Dakocytomation)

## EXAMPLES

### Patient Samples and Tissue Acquisition

Thirty-three patients participated in the study. Eighteen patients with at least one recurrent incisional hernia underwent laparoscopic incisional hernia repair, and fifteen healthy patients who had no hernia history underwent laparoscopic cholecystectomy as controls. Approximately 1 cm² of skin and fascia was removed from the trocar placement site, remote from the hernia or old incisions. Tissue samples were divided and placed in either 10% buffered formalin or RNALater™ RNA Stabilization Reagent (Qiagen, Valencia, CA). Tissue was stored in RNALater™ for up to 48 hours at room temperature.

### RNA Isolation and RNA Amplification

Total RNA was isolated from the skin and fascia specimens using the RNeasy® Lipid Tissue Mini Kit (Qiagen) with a rotor homogenizer and on-column DNase treatment. Total RNA was amplified using the WT-Ovation™ Pico RNA Amplification System protocol (NuGen, San Carlos, CA).

### Immunohistochemistry

Specimens were fixed in 10% buffered formalin, processed, embedded in paraffin, and cut at 4 μm. Immunohistochemistry was performed using the automated horseradish peroxidase Autostainer/Envision Plus method (Dakocytomation, Carpenteria, CA).

### Statistical Analysis of Microarray Data

Analysis of microarray gene expression data was performed using R open-source software. Genes considered "not detectable" across >50% of patient samples were excluded. Differential gene expression analysis was performed using a moderated t-statistic applied to the log2-transformed normalized intensity for each gene. Adjustment for multiple testing was made using the false discovery rate method of Benjamini and Hochberg. Gene ontology (GO) analyses were conducted to identify common functions or descriptive terms that were statistically abundant in the list of differentially expressed genes.

### Demographics

Demographics for the 33 enrolled patients and the subset of 17 patients whose samples were analyzed by microarray are shown in Table 1. The majority (26/33) of enrolled patients were female, and all but one sample analyzed by microarray were from females. The RH and NC groups analyzed by microarray were comparable (p > 0.05) on all demographics except diabetes (p = 0.03) and previous surgery (p = 0.01), neither of which is unexpected in these populations.

| Characteristics | Patients Enrolled | Patients Analyzed by Microarray |
|-----------------|-------------------|---------------------------------|
| RH (n = 18)     | NC (n = 15)       | RH (n = 9)                      | NC (n = 8)                      |
| Sex (M/F)       | 4/14              | 3/12                            | 0/9                             | 1/7                             |
| Age             | 55.2 ± 4.9        | 44.9 ± 14.5                     | 50.9 ± 3.9                      | 39.1 ± 10.2                     |
| BMI             | 36.6 ± 3.05       | 30.5 ± 3.10                     | 39.2 ± 3.14                     | 31.4 ± 3.10                     |
| Smoker          | 8                 | 2                               | 4                               | 2                               |
| Diabetes        | 7                 | 0                               | 5                               | 0                               |
| Previous Surgery| 18                | 6                               | 9                               | 3                               |

### Gene Ontology Analysis of Differentially Expressed Genes

Gene ontology analyses were performed to determine whether there were common functions or descriptive terms that were statistically abundant in the list of differentially expressed genes. In the skin of RH patients, many differentially expressed genes were found to be more abundant than expected in biologic processes such as response to wounding, regulation of immune response, activation of plasma proteins during acute inflammatory response, lipid metabolic process, multicellular organismal development, and cell adhesion.

### COL1/COL3 Ratio by Microarray, PCR Array, and Immunohistochemistry

By microarray, the COL1A1/COL3A1 ratio in the skin of RH patients was slightly lower than NC patients, but was not significant (1.33 vs. 1.46, p = 0.65). Similar but significant results were found for COL1A2/COL3A1 (0.59 vs. 0.79, p = 0.02). Neither of these ratios were statistically different in the fascia. Immunohistochemistry on 5 patients demonstrated slightly greater staining intensity of COL3A1 than COL1A1 in the skin and fascia from RH patients in comparison with NC. Analysis by PCR array revealed that gene expression of COL3A1 was greater than COL1A2 in the skin in both groups. The ratio of COL1A2/COL3A1 was decreased in the RH group as compared to NC (1.51 vs. 2.26, p = 0.058, one-sided t-test).

## Other Embodiments

The invention described herein is not limited to the specific embodiments set forth above. Various modifications and variations can be made without departing from the scope of the invention. For example, the method can be adapted to analyze other types of tissues or conditions related to wound healing and connective tissue disorders. Additionally, the invention can be extended to include the development of therapeutic agents targeting the identified genes, such as GREM1, COL1A1, and COL3A1, to prevent or treat recurrent incisional hernias. The invention also encompasses the use of the identified gene expression profiles as biomarkers for stratifying patients into different risk groups for hernia development and recurrence, thereby facilitating personalized treatment approaches.