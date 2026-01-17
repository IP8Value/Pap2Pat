# DESCRIPTION

## BACKGROUND

Prostate cancer (CaP) is a significant health concern, particularly for men of African ancestry who exhibit higher rates of incidence and mortality compared to their Caucasian counterparts. Genetic and biological alterations in CaP differ between African American (AA) and Caucasian American (CA) men, highlighting the need for comprehensive genomic analyses to understand these disparities. Recent studies have identified frequent alterations in genes such as ERG, PTEN, and SPOP in early-stage CaP, but most of these studies have been conducted primarily in men of European ancestry. This invention addresses the gap by performing whole-genome analyses of prostate cancers from AA and CA men, revealing novel genomic alterations and their implications for disease progression.

## SUMMARY

The present invention relates to the discovery of a novel genomic alteration in the LSAMP locus (3q13.31) in prostate cancers of African American men. This alteration, characterized by deletions and rearrangements, is associated with rapid disease progression and recurrence. The invention also includes methods for detecting these alterations using whole-genome sequencing, fluorescence in situ hybridization (FISH), and other molecular techniques. The identification of LSAMP locus alterations provides a valuable biomarker for diagnosing and prognosticating prostate cancer in AA men, enabling the development of targeted therapeutic strategies.

## DETAILED DESCRIPTION

### DEFINITIONS

- **Prostate Cancer (CaP):** A malignant tumor arising from the prostate gland, often characterized by uncontrolled cell growth and the potential to spread to other parts of the body.
- **African American (AA):** Individuals of African descent living in the United States.
- **Caucasian American (CA):** Individuals of European descent living in the United States.
- **LSAMP Locus:** The region on chromosome 3q13.31 that contains the LSAMP gene, which is involved in cell adhesion and migration.
- **Whole-Genome Sequencing (WGS):** A technique that determines the complete DNA sequence of an organism's genome at a single time.
- **Fluorescence In Situ Hybridization (FISH):** A molecular cytogenetic technique used to detect and locate the presence or absence of specific DNA sequences on chromosomes.
- **Single Nucleotide Variants (SNVs):** Genetic variations involving a single nucleotide change in the DNA sequence.
- **Structural Variations (SVs):** Large-scale genomic alterations, including deletions, insertions, inversions, and translocations.
- **Copy Number Variations (CNVs):** Changes in the number of copies of a particular gene or genomic region.

### Examples

#### Example 1: Identification of LSAMP Locus Alterations in AA Prostate Cancer

**Materials and Methods:**

1. **Prostate Cancer Specimens and Sample Preparation:**
   - Prostate cancer samples were obtained from seven AA and seven CA patients undergoing radical prostatectomy at the Walter Reed National Military Medical Center (WRNMMC). Tumor tissues with primary Gleason pattern 3 were manually dissected under a microscope, ensuring 80-95% tumor cell content. DNA was extracted from these tissues and peripheral blood lymphocytes using the DNeasy Blood and Tissue DNA isolation kit (Qiagen).

2. **Whole-Genome Sequencing:**
   - DNA samples were processed using the Illumina TruSeq DNA PCR-Free Sample Preparation kit, resulting in an average insert size of 310 bp. Paired-end sequence reads of 101 bases were generated using the Genome Analyzer IIX with v5 SBS reagent kits. Data were aligned to the reference genome (GRCh37/hg19) using the ELANDv2e algorithm in the CASAVA v1.8 pipeline.

3. **Variant Calling and Validation:**
   - Somatic variants were called using Strelka, Genomatix Mapper, BreakDancer, cn.MOPs, and Control-FREEC. High-confidence SNVs were validated using Varscan2, MuTect, and Somatic Sniper. Structural variations and copy number variations were also identified and validated using Sanger sequencing.

4. **Detection of LSAMP Locus Rearrangements:**
   - In-depth analysis of the 3q13.31 region revealed deletions and rearrangements in three AA patients. Two patients had deletions of 23 Mb and 1 Mb in the ZBTB20-LSAMP region, while the third patient had a duplication resulting in a novel fusion junction. These alterations were confirmed using RNA-Seq data, targeted genomic sequencing, and 5′-RACE.

**Results:**
- LSAMP locus alterations were significantly more prevalent in AA CaP genomes compared to CA CaP genomes. All three AA patients with LSAMP locus alterations showed disease recurrence after prostatectomy.

#### Example 2: Validation of LSAMP Locus Deletions in an Independent Cohort

**Materials and Methods:**

1. **TCGA Data Analysis:**
   - The frequency of LSAMP locus deletions was assessed in the TCGA prostate cancer SNP data. Patients' ancestry was confirmed using principal component analysis (PCA) with the EIGENSTRAT method. Copy number variations were normalized using the CRMA v2 method, and integer copy number inference was performed with the ASCAT software suite.

2. **FISH Assay:**
   - FISH analysis was performed on tissue microarrays (TMAs) comprising multi-sampled cores from 42 AA and 59 CA patients. Locus-specific and control probes were designed and used to detect deletions at the PTEN and ZBTB20-LSAMP loci. Tumor cells with at least two centromeres were counted, and deletions were called when more than 75% of evaluable tumor cells showed loss of allele.

**Results:**
- LSAMP locus deletions were detected in 27% of AA tumors and 13% of CA tumors in the TCGA cohort, supporting the initial WGS observations. FISH analysis on TMAs further confirmed that LSAMP deletions were more prevalent in AA cases (26%) compared to CA cases (7%), and correlated with biochemical recurrence and pT3 tumors in AA men.

#### Example 3: Mutation Landscape of Prostate Cancers in AA and CA Men

**Materials and Methods:**

1. **SNV Detection and Comparison:**
   - A total of 261 somatically acquired SNVs in the coding sequence of 247 genes were detected from the 14 patients. These SNVs were compared against the COSMIC and TCGA databases to identify previously reported mutations in prostate and other cancers.

2. **Validation of PTEN Deletions:**
   - The virtual absence of PTEN deletions in AA CaP was confirmed using FISH analysis on TMAs. PTEN deletions were less frequent in AA (15%) compared to CA cases (63%), with a larger difference observed in Gleason 6 tumors (7% in AA vs. 53% in CA).

**Results:**
- The mutation landscape of prostate cancers in AA and CA men revealed distinct patterns. Recurrent CaP genomic alterations such as TMPRSS2-ERG fusion, PTEN and CHD1 deletions, and SPOP mutations were confirmed. The virtual absence of PTEN deletions in early-stage AA CaP was a notable finding, suggesting a different genomic profile in this population.

#### Example 4: Clinical Implications and Therapeutic Strategies

**Materials and Methods:**

1. **Clinical Correlation:**
   - The clinical outcomes of patients with LSAMP locus alterations were analyzed. Biochemical recurrence and disease progression were correlated with the presence of LSAMP deletions in AA men.

2. **Therapeutic Targeting:**
   - Potential therapeutic strategies targeting the LSAMP locus were explored. Drugs known to affect cell adhesion and migration were evaluated for their efficacy in inhibiting the growth and spread of prostate cancer cells with LSAMP deletions.

**Results:**
- Patients with LSAMP locus alterations had a higher risk of biochemical recurrence and disease progression. Therapeutic targeting of the LSAMP locus may provide a new approach for managing aggressive prostate cancers in AA men.

### Conclusion

The present invention provides a comprehensive understanding of the genomic alterations in prostate cancers of African American men, particularly the novel deletions and rearrangements in the LSAMP locus. These findings have significant implications for the diagnosis, prognosis, and treatment of prostate cancer in this population. The methods and techniques described herein enable the detection of these alterations and pave the way for the development of targeted therapies.