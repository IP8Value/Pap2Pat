# DESCRIPTION

## BACKGROUND

Autoimmune diseases, such as multiple sclerosis (MS), pose significant challenges in terms of diagnosis and treatment. These conditions often lead to severe and irreversible damage before they are detected, necessitating the development of tools that can monitor disease activity and predict relapses. Current methods, such as magnetic resonance imaging (MRI), provide limited information about the underlying immunopathology and are not always reliable in predicting disease progression. There is a critical need for a tool that can accurately and non-invasively monitor the immune status within target tissues, such as the central nervous system (CNS), to identify relapse onset and predict therapeutic responses.

The invention disclosed herein addresses this need by providing an implantable subcutaneous immunological niche (IN) that reflects the immune status of the CNS. This IN is designed to detect changes in gene expression and cell populations that are indicative of disease activity and relapse. The IN can be biopsied and analyzed to provide real-time information about the immune response, enabling preemptive interventions to prevent or mitigate relapses.

## SUMMARY

The present invention relates to a method and system for monitoring immune dysregulation in autoimmune diseases, particularly relapsing-remitting multiple sclerosis (RRMS). The invention involves the creation of an implantable subcutaneous immunological niche (IN) that reflects the immune status of the central nervous system (CNS). The IN is composed of porous materials that induce cell ingrowth and vascularization, allowing for the formation of a tissue that mirrors the immunological changes occurring in the CNS.

Key features of the invention include:
1. **Implantable Subcutaneous Immunological Niche (IN):** A microporous scaffold, such as poly(ε-caprolactone) (PCL), is implanted subcutaneously in a subject. The scaffold induces the formation of a vascularized tissue that reflects the immune status of the CNS.
2. **Gene Expression Analysis:** The IN is biopsied and analyzed for gene expression changes using high-throughput techniques such as OpenArray qPCR. A set of 21 genes has been identified as a signature for disease activity and relapse.
3. **Computational Scoring System:** A scoring system based on unsupervised dimensionality reduction (SVD) and supervised machine learning (Bagged Tree) is used to classify the IN as healthy or diseased. The scores are predictive of disease onset and severity.
4. **Preemptive Interventions:** The IN can be used to monitor disease dynamics and predict relapses, enabling preemptive interventions with therapies such as glucocorticoids or tolerogenic nanoparticles.
5. **Monitoring Therapeutic Response:** The IN can also be used to monitor the effectiveness of therapeutic interventions, providing a tool for personalized medicine.

## DETAILED DESCRIPTION

### Implantable Subcutaneous Immunological Niche (IN)

The invention utilizes a microporous scaffold, such as poly(ε-caprolactone) (PCL), which is implanted subcutaneously in a subject. The scaffold is designed to induce cell ingrowth and vascularization, creating a tissue that reflects the immune status of the central nervous system (CNS). The IN contains disease-relevant innate and adaptive immune cells, making it a valuable tool for monitoring immune dysregulation.

#### Scaffold Fabrication and Subcutaneous Implantation

The microporous scaffolds are prepared by mixing PCL with a salt porogen, pressing into molds, and undergoing polymer sintering and porogen dissolution. The scaffolds are implanted subcutaneously in female SJL/J mice, which are anesthetized with isoflurane. The mice receive subcutaneous injections of carprofen (5 mg/kg) immediately before and 24 hours after surgery to manage pain.

### Gene Expression Analysis

The IN is biopsied and analyzed for gene expression changes using high-throughput techniques such as OpenArray qPCR. A set of 21 genes has been identified as a signature for disease activity and relapse. These genes are differentially expressed between control and diseased mice, and their expression patterns are predictive of disease onset and severity.

#### OpenArray High-Throughput RT-qPCR

High-throughput gene expression analysis is performed using OpenArray panels focused on mouse inflammatory pathways. The panels contain 632 validated genes and 16 endogenous controls. RNA is isolated from the IN using the Directzol RNA Miniprep kit, and cDNA is synthesized using the SuperScript™ VILO™ cDNA Synthesis Kit. The OpenArray analysis is performed on a QuantStudio 12k Flex RT-PCR system, and the data are analyzed to identify differentially expressed genes.

### Computational Scoring System

A scoring system based on unsupervised dimensionality reduction (SVD) and supervised machine learning (Bagged Tree) is used to classify the IN as healthy or diseased. The SVD score and Bagged Tree (BT) score are combined to create a comprehensive metric for disease classification. The scores are predictive of disease onset and severity, and they can be used to monitor disease dynamics and predict relapses.

#### Unsupervised Hierarchical Clustering

Unsupervised hierarchical clustering analysis is performed using MATLAB’s clustergram tool to identify genes that cluster together and to visualize the differences between healthy and diseased samples. The genes of interest are identified based on their fold change (FC), expression stability, and elastic net regularization scores.

### Preemptive Interventions

The IN can be used to monitor disease dynamics and predict relapses, enabling preemptive interventions with therapies such as glucocorticoids or tolerogenic nanoparticles. The IN is biopsied at various time points to assess the immune status, and if the scores indicate disease activity, the subject can be treated with a preemptive intervention to prevent or mitigate relapses.

#### Glucocorticoid Treatment

Glucocorticoids, such as dexamethasone, can be administered to subjects when the IN indicates disease activity. The treatment is effective in reducing clinical symptoms and preventing disease onset, but the effects may diminish over time.

#### Tolerogenic Nanoparticles

Tolerogenic nanoparticles, which have been shown to induce tolerance in preclinical models, can also be used as a preemptive intervention. The nanoparticles are administered intravenously, and they are effective in preventing disease onset with a single administration.

### Monitoring Therapeutic Response

The IN can also be used to monitor the effectiveness of therapeutic interventions, providing a tool for personalized medicine. The IN is biopsied and analyzed for gene expression changes to determine the response to therapy. The scores can be used to identify responders and non-responders to a particular treatment, enabling the selection of the most effective therapy for each subject.

#### T-Cell Subsets

The IN can be used to analyze T-cell subsets, such as Treg, Th2, Th1, Th1/17, and Th17, in the spinal cord, inguinal lymph nodes, and spleens. The IN reflects the immunological changes in diseased organs that are not reflected in lymphoid tissues, making it a valuable tool for monitoring the immune response.

### Examples

#### Example 1: Gene Expression Analysis in Adoptive Transfer Model

Microporous PCL scaffolds were implanted subcutaneously in SJL mice 14 days before adoptively transferring either autoreactive T-cells (reactive to PLP139–151) or control T-cells (reactive to OVA323–339). The INs were harvested and analyzed for gene expression using OpenArray qPCR. The analysis revealed 130 differentially expressed genes between control and diseased mice, with 21 genes identified as a signature for disease activity and relapse.

#### Example 2: Computational Scoring System in Adoptive Transfer Model

The 21-gene signature was used to develop a scoring system based on SVD and Bagged Tree. The scores were predictive of disease onset and severity, and they could be used to monitor disease dynamics and predict relapses.

#### Example 3: Gene Expression Analysis in Active Immunization Model

The active immunization model of EAE was used to demonstrate the broader utility of the IN. The INs were harvested and analyzed for gene expression using OpenArray qPCR. The analysis revealed 222 differentially expressed genes between control and diseased mice, with 25 genes identified as a signature for disease activity and relapse.

#### Example 4: Preemptive Interventions with Glucocorticoids

EAE was induced in mice, and the INs were biopsied for analysis 7 days post-transfer. The mice were then treated with a daily intraperitoneal injection of 5 mg/kg dexamethasone from days 7 to 11. The treatment reduced clinical symptoms and prevented disease onset, but the effects diminished after day 14.

#### Example 5: Preemptive Interventions with Tolerogenic Nanoparticles

EAE was induced in mice, and the INs were biopsied for analysis 7 days post-transfer. The mice were then treated with a single intravenous dose of 2.5 mg of antigen-encapsulating PLG nanoparticles. The treatment prevented disease onset through day 18, with only a single administration.

#### Example 6: Monitoring Therapeutic Response with Tolerogenic Nanoparticles

The INs were biopsied and analyzed for gene expression to determine the response to therapy. The effective treatment group (PLP reactive T-cells with PLP particles) had similar signature and clinical scores as the control group. The ineffective treatment group (PLP reactive T-cells with OVA particles) had significantly higher signature and clinical scores relative to the control.

#### Example 7: Gene Expression Analysis in Remission and Relapse

The INs were biopsied at various time points to assess the immune status during remission and relapse. The gene expression patterns in the INs during remission were similar to those of control mice, indicating a return to a healthy state. During relapse, the gene expression patterns returned to those associated with disease onset, indicating a relapse.

#### Example 8: T-Cell Subsets in IN and Spinal Cord

The INs and spinal cords were analyzed for T-cell subsets, such as Treg, Th2, Th1, Th1/17, and Th17. The INs reflected the immunological changes in the spinal cord during EAE onset, highlighting the potential utility of the IN over endogenous lymphoid tissues.

#### Example 9: Longitudinal Monitoring of Disease Dynamics

The INs were biopsied and analyzed for gene expression at various time points to monitor disease dynamics. The scores were used to predict disease onset, monitor remission, and identify relapses, enabling preemptive interventions to prevent or mitigate relapses.

## CONCLUSION

The invention provides a novel and effective method for monitoring immune dysregulation in autoimmune diseases, particularly relapsing-remitting multiple sclerosis (RRMS). The implantable subcutaneous immunological niche (IN) reflects the immune status of the central nervous system (CNS) and can be biopsied and analyzed to provide real-time information about the immune response. The IN can be used to predict disease onset and relapses, enabling preemptive interventions to prevent or mitigate relapses. Additionally, the IN can be used to monitor the effectiveness of therapeutic interventions, providing a tool for personalized medicine.