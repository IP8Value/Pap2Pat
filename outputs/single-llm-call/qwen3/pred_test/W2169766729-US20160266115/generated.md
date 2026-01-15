# DESCRIPTION

## BACKGROUND

- The immune system is a complex, dynamic network of specialized cells, signaling molecules, and tissues that work in concert to defend the body against pathogens, eliminate abnormal or malignant cells, and maintain homeostasis through coordinated activation and suppression. It comprises innate and adaptive components that interact across multiple physiological scales, from molecular recognition events to systemic inflammatory responses. The innate immune system includes phagocytic cells such as neutrophils, monocytes, and macrophages, as well as natural killer cells and myeloid-derived suppressor cells, which provide rapid, non-specific defense mechanisms. The adaptive immune system consists of T lymphocytes, B lymphocytes, and antigen-presenting cells that generate highly specific, long-lasting responses through clonal expansion and immunological memory. These components are not isolated entities but functionally interdependent, with each cell type capable of modulating the activity of others through direct cell-to-cell contact, cytokine secretion, and metabolic competition. The balance between stimulatory and inhibitory signals within this network determines the overall immune tone, which can shift between states of activation, tolerance, or suppression depending on environmental cues, genetic predisposition, and disease burden. Disruption of this delicate equilibrium underlies many pathological conditions, including chronic inflammation, autoimmune disorders, and cancer progression. Traditional approaches to immune assessment have historically focused on isolated cell populations or relative proportions within a single lineage, often overlooking the systemic context in which these cells operate. Such reductionist methods fail to capture the emergent properties of immune network dynamics, leading to inconsistent correlations between immune parameters and clinical outcomes. A more comprehensive framework is required to understand how the collective behavior of leukocyte subsets defines immune status and dictates therapeutic response.

- The immune system comprises a diverse array of cellular and molecular components that collectively orchestrate immune surveillance and response. Key cellular constituents include granulocytes, such as neutrophils, which are the most abundant circulating leukocytes and serve as first responders to infection; monocytes, which differentiate into macrophages and dendritic cells upon tissue entry; lymphocytes, which encompass T cells, B cells, and natural killer cells responsible for adaptive immunity; and regulatory subsets such as CD4+CD25+CD127lo regulatory T cells and CD14+HLA-DRlo/neg immunosuppressive monocytes that modulate immune activation to prevent excessive inflammation. Each of these populations is defined by unique surface marker expression patterns, functional capacities, and developmental trajectories. Granulocytes are characterized by high forward and side scatter properties and express markers such as CD15 and CD66b, while monocytes are identified by CD14 expression and variable HLA-DR levels that reflect their activation state. T lymphocytes are defined by CD3 expression and subdivided into CD4+ helper and CD8+ cytotoxic subsets, with CD4+ T cells further classified into effector, memory, and regulatory phenotypes. B lymphocytes are identified by CD19 and CD20 expression and are responsible for antibody production, whereas natural killer cells co-express CD56 and CD16 and mediate cytotoxicity without prior sensitization. Myeloid-derived suppressor cells, identified by lineage negativity, HLA-DR low/negative expression, and CD33 positivity, represent a heterogeneous population of immature myeloid cells that expand in pathological conditions and actively suppress T cell function. The functional integrity of the immune system depends not only on the absolute numbers of these subsets but also on their relative proportions, spatial distribution, and reciprocal interactions. The interplay between pro-inflammatory and immunosuppressive elements determines whether an immune response is effective, self-limiting, or pathologically suppressed. Understanding these relationships requires a systems-level approach that accounts for the entire leukocyte ecosystem rather than isolated components.

## SUMMARY

- Immune system profiles are comprehensive, quantitative representations of the relative and absolute abundances of circulating leukocyte subsets in an individual, derived from multiparametric flow cytometric analysis of whole blood. These profiles integrate data from multiple immune cell lineages—including granulocytes, monocytes, lymphocytes, T cells, B cells, natural killer cells, regulatory T cells, and immunosuppressive monocytes—into a unified phenotypic signature that reflects the systemic immune state. Unlike conventional metrics that rely on percentages or ratios within a single lineage, immune system profiles capture the full combinatorial landscape of immune cell frequencies per microliter of blood, enabling the identification of recurring patterns across diverse patient populations. These profiles are not disease-specific but represent fundamental states of immune organization that can be shared across unrelated pathologies, such as cancer, acute lung injury, and sepsis. By analyzing these patterns, it becomes possible to classify individuals into distinct immune phenotypes that correlate strongly with clinical outcomes, independent of the underlying diagnosis. This paradigm shift from disease-centric to immune-centric classification provides a novel framework for prognostication, therapeutic selection, and drug development.

- The flow cytometry method employed to assess immune system profiles involves the direct staining of fresh, unprocessed whole blood with a panel of fluorochrome-conjugated monoclonal antibodies targeting key leukocyte surface markers. This approach eliminates the need for density gradient separation or other manipulative steps that may alter cell viability, distribution, or abundance. Cells are labeled with antibodies specific for CD3, CD4, CD8, CD19, CD56, CD14, HLA-DR, CD25, CD127, CD33, and other relevant markers, followed by lysis of red blood cells and fixation. Absolute cell counts per microliter are determined using TruCount™ beads or similar volumetric reference standards, allowing for precise quantification of each subset without reliance on relative gating. Data are acquired on a calibrated flow cytometer and analyzed using standardized gating strategies to ensure reproducibility across laboratories. This method provides a high-dimensional, quantitative snapshot of the peripheral immune landscape, capturing both the magnitude and composition of immune cell populations in a single assay.

- Leukocyte subtypes are quantified as absolute numbers per microliter of blood, including granulocytes, monocytes, lymphocytes, CD3+ T cells, CD4+ T cells, CD8+ T cells, CD19+ B cells, CD56+CD16+ natural killer cells, CD4+CD25+CD127lo regulatory T cells, CD14+HLA-DR+ monocytes, and CD14+HLA-DRlo/neg immunosuppressive monocytes. Each subset is measured independently and then integrated into a composite profile that reflects the total leukocyte architecture. This quantitative approach enables the detection of subtle but biologically significant shifts in immune composition that are invisible when only relative proportions are considered. For example, a patient may exhibit a normal percentage of CD4+ T cells within the lymphocyte compartment, yet have an overall lymphopenia that renders the absolute number of CD4+ cells insufficient to mount an effective immune response. By measuring absolute counts, such critical contextual information is preserved and utilized in downstream analysis.

- A database of immune phenotypes is generated by aggregating quantitative flow cytometry data from hundreds of individuals, including healthy volunteers and patients with diverse pathological conditions. Each entry in the database contains the absolute cell counts for all measured leukocyte subsets, along with metadata such as age, sex, disease diagnosis, treatment history, and clinical outcome. This database serves as a reference repository for identifying patterns of immune organization that recur across individuals, regardless of disease etiology. The inclusion of healthy volunteers provides a baseline for defining normal immune variation, against which pathological deviations can be measured and interpreted.

- Similarity analysis is applied to the database using unsupervised hierarchical clustering and principal component analysis to identify groups of individuals with highly similar immune phenotypes. These analyses reveal clusters of patients whose immune cell distributions are statistically indistinguishable from one another, even when they have different diagnoses. The clustering algorithm operates on normalized cell count data, where each individual’s values are divided by the mean value observed in the healthy volunteer cohort, allowing for cross-subject comparison despite baseline differences in total leukocyte counts. The resulting dendrograms and principal component plots reveal discrete immune profiles that are reproducible and biologically meaningful.

- Immune profiles are defined as groups of at least seven individuals whose immune phenotypes cluster together with minimal dendrogram branch length, indicating high similarity in leukocyte composition. These profiles are not arbitrary groupings but emerge as stable, reproducible patterns from the data, representing distinct states of immune organization. Each profile is characterized by a unique combination of absolute cell counts across multiple lineages, such as elevated granulocytes with depleted lymphocytes, or increased CD14+HLA-DRlo/neg monocytes with reduced CD4+ T cells. These patterns reflect underlying biological states that transcend diagnostic categories and are associated with consistent clinical outcomes.

- An individual’s immune phenotype is compared to the database of immune profiles using distance metrics such as Euclidean or Mahalanobis distance to determine the closest matching profile. This comparison is performed algorithmically, assigning each patient to the profile with the highest similarity score. The assignment is not based on disease diagnosis but solely on the quantitative immune signature, enabling the identification of patients who share immune states despite different clinical presentations.

- Immune status is determined by the assigned immune profile, which provides a functional classification of the patient’s immune system as being in a state of activation, suppression, or homeostasis. Profiles associated with high lymphocyte and low immunosuppressive monocyte counts are classified as favorable, while those with elevated granulocytes, monocytes, and CD14+HLA-DRlo/neg monocytes are classified as suppressive. This classification predicts the likelihood of immune-mediated control of disease and response to immunotherapy.

- Immune profiles are used to predict response to therapy by correlating profile assignment with clinical outcomes in patients who have received immune-modulating treatments such as checkpoint inhibitors, cytokine therapies, or adoptive cell transfer. Patients assigned to favorable immune profiles are more likely to respond to these therapies, while those in suppressive profiles show minimal benefit. This predictive capability allows for stratification of patients prior to treatment, improving trial design and clinical decision-making.

- Subtypes of disease that correlate with specific immune profiles are identified by analyzing the distribution of diagnoses within each profile. For example, glioblastoma, renal cell carcinoma, and non-Hodgkin lymphoma patients are found to cluster into distinct immune profiles, but a subset of patients from each disease group shares the same profile as healthy volunteers. These shared profiles are associated with significantly improved survival, indicating that immune status, rather than tumor type, is the dominant determinant of outcome.

- Pathological subtypes are diagnosed by assigning patients to immune profiles that are statistically associated with specific clinical trajectories. This diagnostic method does not rely on histopathology or molecular markers of the tumor but instead on the systemic immune context. A patient with glioblastoma may be diagnosed as having a “suppressive immune subtype” based on their profile, even if their tumor is molecularly identical to that of a patient with a favorable profile.

- The flow cytometry method for immune phenotype determination is standardized across laboratories using a defined panel of antibodies, gating strategies, and calibration protocols. This ensures that immune profiles are reproducible and comparable across institutions, enabling multi-center validation and clinical deployment.

- Cell numbers are compared to the database of immune profiles using automated algorithms that calculate the Euclidean distance between the patient’s normalized cell counts and the centroid of each profile. The profile with the smallest distance is assigned as the patient’s immune status. This comparison is performed in real time and can be integrated into clinical laboratory information systems.

- Immune system profile is identified by clustering analysis of quantitative flow cytometry data from peripheral blood, resulting in a classification that reflects the patient’s systemic immune state. This profile is independent of disease diagnosis and is determined solely by the relative and absolute abundances of leukocyte subsets.

- Immune profiles are used to identify medical outcomes by correlating profile assignment with survival, progression-free survival, response to therapy, and incidence of complications. Patients in favorable profiles consistently demonstrate longer survival across multiple disease types, while those in suppressive profiles exhibit rapid disease progression and poor response to treatment.

- The method for treating glioblastoma involves determining the patient’s immune system profile via flow cytometry of peripheral blood, classifying the patient into one of five defined immune profiles, and selecting an immunomodulatory intervention based on profile assignment. Patients in favorable profiles may receive standard therapy with immune checkpoint blockade, while those in suppressive profiles may be candidates for myeloid-targeted therapies or cellular reconstitution strategies.

- The method for treating renal cell carcinoma involves determining the patient’s immune system profile via flow cytometry of peripheral blood, classifying the patient into one of five defined immune profiles, and selecting an immunomodulatory intervention based on profile assignment. Patients in favorable profiles may benefit from vascular endothelial growth factor inhibitors combined with checkpoint blockade, while those in suppressive profiles may require prior depletion of immunosuppressive monocytes before immunotherapy.

- The method for treating non-Hodgkin lymphoma involves determining the patient’s immune system profile via flow cytometry of peripheral blood, classifying the patient into one of five defined immune profiles, and selecting an immunomodulatory intervention based on profile assignment. Patients in favorable profiles may respond to anti-CD20 monoclonal antibodies and checkpoint inhibitors, while those in suppressive profiles may require combination therapies targeting myeloid suppression.

- The method for determining immune system profile involves collecting a peripheral blood sample, performing multiparametric flow cytometry using a standardized panel of antibodies, quantifying absolute cell counts per microliter for ten key leukocyte subsets, normalizing these counts to a healthy volunteer reference mean, and applying hierarchical clustering to assign the patient to one of five predefined immune profiles.

- Cell numbers are compared to the database of immune profiles using a computational algorithm that calculates the Euclidean distance between the patient’s normalized cell count vector and the centroid vectors of each profile. The profile with the minimum distance is assigned as the patient’s immune status.

- A human is classified as having an immune system profile when their normalized leukocyte count vector falls within a defined statistical boundary of one of the five pre-established immune profiles, as determined by hierarchical clustering of a reference cohort.

- The likelihood of a medical outcome is assessed by correlating the assigned immune profile with historical survival and response data from the database. Profiles associated with prolonged survival and treatment response are assigned a high probability of favorable outcome, while profiles associated with rapid progression and therapy resistance are assigned a low probability.

- Immune profiles are used to accelerate the testing of immune-modulating drugs by stratifying clinical trial participants based on immune status rather than disease diagnosis. This increases the likelihood of detecting therapeutic efficacy by enriching for patients who are biologically primed to respond, reducing sample size requirements and trial duration.

## DETAILED DESCRIPTION

- Methods and materials for assessing immune system profiles include the use of fresh, unprocessed peripheral blood collected in EDTA tubes, a standardized panel of fluorochrome-conjugated monoclonal antibodies targeting CD3, CD4, CD8, CD19, CD56, CD14, HLA-DR, CD25, CD127, CD33, and lineage markers, TruCount™ beads for absolute quantification, flow cytometers calibrated daily with fluorescent microspheres, and software platforms for data acquisition and analysis such as FlowJo, BD Multiset, and Partek Genomics Suite. All reagents are validated for batch-to-batch consistency, and gating strategies are documented and standardized across participating institutions.

- Flow cytometry is performed on whole blood by adding antibody cocktails directly to 50–100 μL of blood, incubating for 15–20 minutes at room temperature in the dark, lysing red blood cells with ammonium chloride-based lysis buffer, washing with phosphate-buffered saline, and fixing in 4% paraformaldehyde. Data are acquired on a BD FACSCalibur or equivalent flow cytometer using consistent voltage and compensation settings. Absolute cell counts are determined using TruCount™ beads, and percentages are converted to absolute numbers per microliter using the total lymphocyte count as a reference.

- Information about leukocyte subsets is generated by measuring the absolute number of granulocytes, monocytes, lymphocytes, CD3+ T cells, CD4+ T cells, CD8+ T cells, CD19+ B cells, CD56+CD16+ natural killer cells, CD4+CD25+CD127lo regulatory T cells, CD14+HLA-DR+ monocytes, and CD14+HLA-DRlo/neg monocytes per microliter of blood. These values are recorded in a structured database with associated clinical metadata, including age, sex, diagnosis, treatment history, and survival outcome.

- Clustering algorithms are applied to the database using unsupervised hierarchical agglomerative clustering with Euclidean distance and average linkage. Normalized cell count vectors are log-transformed prior to clustering to reduce skewness and enhance separation of low-abundance populations. Principal component analysis is performed to visualize high-dimensional data in two or three dimensions and to identify the most influential variables driving profile separation.

- Normalization of information within the database is performed by dividing each individual’s absolute cell count for a given marker by the mean value of that marker in the healthy volunteer cohort. This creates a dimensionless ratio that reflects the deviation of each patient’s immune state from the norm, enabling cross-subject comparison regardless of baseline leukocyte counts.

- Individuals with similar immune profiles are identified by clustering analysis, which groups patients whose normalized cell count vectors are statistically indistinguishable from one another. These individuals are expected to exhibit similar immune system responses to infection, injury, or therapy due to the shared architecture of their leukocyte ecosystem.

- A healthy immune system profile is defined as a cluster of at least seven healthy volunteers whose normalized cell count vectors form a distinct, non-overlapping group in hierarchical clustering. This profile is characterized by moderate lymphocyte counts, low granulocyte counts, minimal CD14+HLA-DRlo/neg monocytes, and a CD4+/CD14+HLA-DRlo/neg ratio greater than 20.

- The cell counts for a healthy immune system profile are defined as follows: lymphocytes 1,200–2,500 cells/μL, granulocytes 2,500–5,000 cells/μL, monocytes 300–700 cells/μL, CD4+ T cells 500–1,200 cells/μL, CD8+ T cells 300–800 cells/μL, B cells 100–300 cells/μL, natural killer cells 100–300 cells/μL, regulatory T cells 10–50 cells/μL, CD14+HLA-DR+ monocytes 200–500 cells/μL, and CD14+HLA-DRlo/neg monocytes less than 10 cells/μL.

- Immune system profile 1 is defined as a cluster of individuals with lymphocyte and CD4+ T cell counts near or above healthy norms, low granulocyte and monocyte counts, and minimal CD14+HLA-DRlo/neg monocytes. This profile is associated with favorable outcomes across multiple disease states.

- The cell counts for immune system profile 1 are defined as follows: lymphocytes 1,100–2,800 cells/μL, granulocytes 2,000–4,500 cells/μL, monocytes 250–650 cells/μL, CD4+ T cells 450–1,300 cells/μL, CD8+ T cells 250–750 cells/μL, B cells 80–320 cells/μL, natural killer cells 80–300 cells/μL, regulatory T cells 10–60 cells/μL, CD14+HLA-DR+ monocytes 200–550 cells/μL, and CD14+HLA-DRlo/neg monocytes less than 15 cells/μL.

- Immune system profile 2 is defined as a cluster of individuals with moderate lymphopenia, elevated granulocytes, normal monocytes, and low CD14+HLA-DRlo/neg monocytes. This profile is associated with intermediate survival outcomes.

- The cell counts for immune system profile 2 are defined as follows: lymphocytes 800–1,600 cells/μL, granulocytes 4,000–7,000 cells/μL, monocytes 300–700 cells/μL, CD4+ T cells 300–800 cells/μL, CD8+ T cells 200–500 cells/μL, B cells 50–200 cells/μL, natural killer cells 50–200 cells/μL, regulatory T cells 15–70 cells/μL, CD14+HLA-DR+ monocytes 250–600 cells/μL, and CD14+HLA-DRlo/neg monocytes less than 20 cells/μL.

- Immune system profile 3 is defined as a cluster of individuals with elevated granulocytes, elevated monocytes, elevated lymphocytes, elevated regulatory T cells, and elevated CD14+HLA-DRlo/neg monocytes. This profile is associated with poor outcomes in cancer and acute lung injury.

- The cell counts for immune system profile 3 are defined as follows: lymphocytes 1,500–3,200 cells/μL, granulocytes 6,000–10,000 cells/μL, monocytes 800–1,500 cells/μL, CD4+ T cells 600–1,400 cells/μL, CD8+ T cells 400–900 cells/μL, B cells 150–400 cells/μL, natural killer cells 150–400 cells/μL, regulatory T cells 50–150 cells/μL, CD14+HLA-DR+ monocytes 500–1,000 cells/μL, and CD14+HLA-DRlo/neg monocytes 50–200 cells/μL.

- Immune system profile 4 is defined as a cluster of individuals with markedly elevated granulocytes and monocytes, severely depleted lymphocytes, and markedly elevated CD14+HLA-DRlo/neg monocytes. This profile is associated with the poorest survival outcomes.

- The cell counts for immune system profile 4 are defined as follows: lymphocytes 300–800 cells/μL, granulocytes 8,000–15,000 cells/μL, monocytes 1,000–2,000 cells/μL, CD4+ T cells 100–400 cells/μL, CD8+ T cells 100–300 cells/μL, B cells 20–100 cells/μL, natural killer cells 20–80 cells/μL, regulatory T cells 30–100 cells/μL, CD14+HLA-DR+ monocytes 600–1,200 cells/μL, and CD14+HLA-DRlo/neg monocytes 100–300 cells/μL.

- Immune system profile 5 is defined as a cluster of individuals with profound lymphopenia, low granulocytes, low monocytes, and low CD14+HLA-DRlo/neg monocytes. This profile is associated with immune exhaustion and poor response to immunotherapy.

- The cell counts for immune system profile 5 are defined as follows: lymphocytes 100–400 cells/μL, granulocytes 1,000–3,000 cells/μL, monocytes 100–300 cells/μL, CD4+ T cells 50–150 cells/μL, CD8+ T cells 30–100 cells/μL, B cells 10–50 cells/μL, natural killer cells 10–40 cells/μL, regulatory T cells 5–30 cells/μL, CD14+HLA-DR+ monocytes 50–150 cells/μL, and CD14+HLA-DRlo/neg monocytes less than 5 cells/μL.

- Immune system profile is determined by calculating the ratio of each leukocyte subset’s absolute count to the mean value observed in the healthy volunteer cohort, then applying hierarchical clustering to identify the closest matching profile. The patient is assigned to the profile with the smallest Euclidean distance to their normalized vector.

- Flow cytometry is used to determine cell counts by staining whole blood with a standardized panel of antibodies, lysing red blood cells, and acquiring data on a calibrated flow cytometer. Absolute counts are derived using TruCount™ beads or equivalent volumetric reference standards.

- Antibodies and reagents used to perform flow cytometry include anti-CD3 FITC, anti-CD4 PE, anti-CD8 APC, anti-CD19 PerCP, anti-CD56 PE-Cy7, anti-CD14 APC-Cy7, anti-HLA-DR BV421, anti-CD25 PE, anti-CD127 AF700, anti-CD33 PE, and lineage cocktail (CD2, CD3, CD19, CD56, CD14) for exclusion. All antibodies are titrated for optimal signal-to-noise ratio and validated for batch consistency.

- Immunological techniques such as intracellular cytokine staining, proliferation assays, and apoptosis detection are used to validate the functional state of cells within each profile, confirming that the observed phenotypes correspond to biological activity.

- PCR or array technologies are used to determine cell counts by measuring mRNA expression of lineage-specific genes such as CD3E, CD19, CD14, and CD33, and correlating transcript levels with flow cytometry-derived absolute counts. These methods provide orthogonal validation of cell abundance.

- Immune system profiles are associated with medical outcomes by analyzing survival, progression-free survival, response to immunotherapy, and incidence of infection in patients assigned to each profile. Profiles 1 and 2 are associated with prolonged survival, while profiles 3, 4, and 5 are associated with rapid progression and poor response.

- Immune system profiles are associated with response to immune-modulating drugs by comparing clinical outcomes in patients treated with checkpoint inhibitors, cytokines, or cellular therapies based on profile assignment. Patients in profile 1 show the highest response rates, while those in profile 4 show minimal benefit.

- Methods for treating mammals involve determining the immune system profile via flow cytometry, classifying the patient into one of five immune profiles, and administering a therapy selected based on profile-specific biological characteristics. For example, profile 4 patients receive myeloid-targeted agents prior to checkpoint blockade.

- Glioblastoma is treated by determining the immune system profile, identifying patients in profile 4, and administering a CSF1R inhibitor to deplete immunosuppressive monocytes, followed by PD-1 blockade. Patients in profile 1 receive PD-1 blockade alone.

- Renal cell carcinoma is treated by determining the immune system profile, identifying patients in profile 3, and administering a combination of VEGF inhibitor and IL-2, while patients in profile 4 receive a combination of CSF1R inhibitor and CTLA-4 blockade.

- Non-Hodgkin lymphoma is treated by determining the immune system profile, identifying patients in profile 5, and administering CAR T-cell therapy with lymphodepletion, while patients in profile 1 receive anti-CD20 monoclonal antibody with PD-1 blockade.

- Examples of the invention include the analysis of 40 healthy volunteers and 100 patients with glioblastoma, non-Hodgkin lymphoma, renal cell carcinoma, ovarian cancer, and acute lung injury. Hierarchical clustering of their immune phenotypes revealed five distinct immune profiles that correlated with survival independent of diagnosis.

- Further description of the invention includes the use of machine learning algorithms to refine profile boundaries, the integration of cytokine and chemokine data to augment immune profiles, and the development of point-of-care devices for rapid immune profiling in clinical settings.

- The scope of the invention encompasses all methods of determining immune system profiles using quantitative flow cytometry of peripheral blood, all databases of immune phenotypes derived from such methods, all algorithms for profile assignment, and all therapeutic interventions selected based on profile assignment.

- The invention concludes with the recognition that immune status, as defined by systemic leukocyte composition, is a dominant determinant of clinical outcome and therapeutic response, transcending traditional diagnostic categories and providing a new paradigm for precision immunology.

### EXAMPLES

- Patients and healthy volunteers were enrolled under institutional review board approval, including 40 healthy volunteers, 27 glioblastoma patients, 24 non-Hodgkin lymphoma patients, 20 renal cell carcinoma patients, 15 ovarian cancer patients, and 25 acute lung injury patients. All samples were collected prior to initiation of therapy.

- Sample collection involved drawing 5 mL of peripheral blood into EDTA tubes, processing within two hours, and storing at 4°C until staining. Previous results from flow cytometry of these samples had been published but were reanalyzed using the new clustering methodology.

- Specific characteristics of patients included age, sex, tumor stage, prior therapy, steroid use, and survival time. Glioblastoma patients were stratified by dexamethasone use, while acute lung injury patients were stratified by sepsis status.

- Acute lung injury patients were defined by meeting the American-European Consensus Conference criteria, including PaO2/FiO2 ≤300, bilateral infiltrates on chest radiograph, and absence of left atrial hypertension.

- Flow cytometry of whole blood was performed using a standardized panel of antibodies and TruCount™ beads to determine absolute cell counts per microliter. Data were acquired on a BD FACSCalibur flow cytometer and analyzed using FlowJo and Partek Genomics Suite.

- Immune markers identified included CD3, CD4, CD8, CD19, CD56, CD14, HLA-DR, CD25, CD127, CD33, and lineage markers. Antibody reagents were sourced from BD Biosciences, BioLegend, and eBioscience, with validation for lot-to-lot consistency.

- Antibody reagents included anti-CD3 FITC, anti-CD4 PE, anti-CD8 APC, anti-CD19 PerCP, anti-CD56 PE-Cy7, anti-CD14 APC-Cy7, anti-HLA-DR BV421, anti-CD25 PE, anti-CD127 AF700, and anti-CD33 PE.

- Data acquisition and analysis involved setting voltage and compensation using single-stained controls, gating on live, single cells, and using TruCount™ beads to calculate absolute counts. Gating strategies were standardized across all samples.

- Gating strategies included forward/side scatter for granulocyte/monocyte/lymphocyte discrimination, lineage exclusion for myeloid subsets, and sequential gating for CD4+CD25+CD127lo Tregs and CD14+HLA-DRlo/neg monocytes.

- Cell counts were calculated by multiplying the percentage of each subset by the absolute lymphocyte count obtained from TruCount™ beads, then normalizing to the healthy volunteer mean.

- Multiparameter analysis involved log-transforming normalized cell counts and performing hierarchical clustering using Euclidean distance and average linkage in Partek Genomics Suite.

- Hierarchical clustering revealed five distinct immune profiles with minimal internal variation and maximal separation between profiles.

- Principal component analysis confirmed that the first two principal components accounted for over 70% of variance and separated profiles along biologically meaningful axes.

- Immune phenotypes were defined as the complete set of absolute leukocyte counts for each individual, and immune profiles were defined as clusters of at least seven individuals with similar phenotypes.

- Immune profiles were identified across glioblastoma, non-Hodgkin lymphoma, renal cell carcinoma, ovarian cancer, and acute lung injury, with each disease showing a non-random distribution across profiles.

- Distinct immune profiles were identified that were shared across disease types, with profile 1 containing the majority of healthy volunteers and a subset of cancer patients with prolonged survival.

- Immune profiles were compared across diseases using Fisher’s exact test, revealing significant differences in profile distribution between glioblastoma, renal cell carcinoma, and non-Hodgkin lymphoma.

- Immune cell demographics were analyzed by comparing absolute counts of each subset across profiles, revealing that profile 4 had the highest granulocyte and CD14+HLA-DRlo/neg monocyte counts and the lowest lymphocyte counts.

- The uniqueness of immune profiles was confirmed by comparing each profile’s marker distribution to the pooled healthy volunteer cohort, with profile 4 showing the most significant deviation.

- Immune profiles were compared to healthy volunteers using Mann-Whitney U tests, confirming that profiles 3, 4, and 5 were significantly different from healthy controls.

- Average immune phenotypes were reconstructed for each profile by calculating the median cell count for each marker and visualizing as pie charts representing the relative abundance of each subset.

- Relative and absolute values were visualized using heatmaps and box plots, demonstrating that absolute changes in cell counts were more discriminative than relative percentages.

- Immune profiles were correlated with patient outcome using Kaplan-Meier survival analysis, with profile 1 and 2 showing median survival of 915 days versus 379 days for profiles 3–5.

- Survival data were analyzed using Cox proportional hazards models adjusted for age and disease type, confirming immune profile as an independent prognostic factor.

- Related immune markers were identified using hierarchical clustering of 23 markers, revealing that CD14+HLA-DRlo/neg monocytes clustered independently of granulocytes and monocytes.

- Hierarchical clustering identified CD14+HLA-DRlo/neg monocytes as a central node in immune suppression, inversely correlated with CD4+ T cells.

- CD14+HLA-DRlo/neg monocytes were identified as a prognostic biomarker, with a cutoff of 20 cells/μL separating high-risk from low-risk patients.

- The CD4+/CD14+HLA-DRlo/neg ratio was calculated for each patient, with a cutoff of 2.0 providing optimal separation of survival curves.

- Overall survival was analyzed using multivariate Cox regression, confirming that the CD4+/CD14+HLA-DRlo/neg ratio was an independent predictor of survival (p=0.006).

- Comprehensive immune system analysis was performed using a systems biology approach integrating flow cytometry, bioinformatics, and clinical outcomes.

- The technical approach involved standardized flow cytometry, normalized data, hierarchical clustering, and validation in independent cohorts.

- Data analysis was performed using Partek Genomics Suite, Prism, and SAS software, with statistical significance defined as p<0.05.

- Potential applications include patient stratification for clinical trials, real-time immune monitoring during therapy, and development of profile-specific immunotherapies.

- Other embodiments include the use of mass cytometry, single-cell RNA sequencing, and machine learning to expand the number of markers and refine profile boundaries.