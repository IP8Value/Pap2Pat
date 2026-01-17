# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a method for predicting the response to treatment with bevacizumab in combination with chemotherapy in patients with breast cancer. More specifically, the invention provides a protein signature score (VEGF inhibition response predictor [ViRP] score) derived from the expression levels of nine specific proteins measured in tumor samples collected prior to neoadjuvant treatment. This score is capable of identifying patients who are likely to achieve a pathologic complete response (pCR) or have a low residual cancer burden (RCB) following treatment with bevacizumab plus chemotherapy.

## BACKGROUND

Treatment of solid tumors using antiangiogenic therapy has been a focus of research for several decades. The discovery of vascular endothelial growth factor A (VEGF-A) as a key driver of tumor angiogenesis led to the development of bevacizumab, a recombinant humanized monoclonal antibody targeting VEGF-A. The addition of bevacizumab to various chemotherapy regimens has shown significant benefits in patients with several types of advanced solid tumors, improving overall survival (OS) and progression-free survival (PFS).

However, despite the initial success of bevacizumab in some patients with breast cancer (BC), randomized trials have not consistently demonstrated improved OS in the general population. This has led to limitations in the clinical use of bevacizumab for BC, particularly in regions outside of Europe. Nonetheless, there is evidence that certain subpopulations of patients do benefit from bevacizumab therapy, highlighting the need for biomarkers to identify these responsive subgroups.

Several biomarkers have been explored, including plasma VEGF-A levels, soluble carbonic anhydrase IX, BRCA1/2 mutations, and DNA methylation signatures. However, none of these biomarkers have been universally accepted for clinical use. There is a critical need for more reliable biomarkers, particularly those based on protein expression, which have shown promise in predicting drug response.

## SUMMARY

The present invention provides a method for predicting the response to treatment with bevacizumab in combination with chemotherapy in patients with breast cancer. The method involves measuring the expression levels of nine specific proteins in tumor samples collected prior to neoadjuvant treatment and calculating a protein signature score (ViRP score) based on these measurements. The ViRP score is capable of identifying patients who are likely to achieve a pathologic complete response (pCR) or have a low residual cancer burden (RCB) following treatment with bevacizumab plus chemotherapy.

The nine proteins included in the ViRP score are:
1. Syk
2. ACC-pS79
3. AKT-pS473
4. Bcl-2
5. CDK2
6. c-Myc
7. p38 MAPK-pT180/Y182
8. p53
9. VEGFR2

The ViRP score is calculated as the sum of the intercept and beta coefficient-weighted expression of these nine proteins. The score demonstrates a significant correlation with relative tumor size after treatment, pCR, and RCB. Patients with a lower ViRP score are more likely to respond positively to the treatment.

The invention also includes a method for validating the ViRP score using mRNA expression data as a surrogate. The mRNA ViRP score is calculated using the corresponding genes from the ViRP signature and demonstrates a high correlation with the original protein ViRP score. The predictive performance of the ViRP score is evaluated using receiver operating characteristic (ROC) curves, and the score is validated in an independent clinical cohort.

## DETAILED DESCRIPTION

### Treatment Regimen

The method of the present invention is designed to predict the response to a specific treatment regimen involving bevacizumab in combination with chemotherapy. Bevacizumab is a recombinant humanized monoclonal antibody that targets VEGF-A, inhibiting angiogenesis and thereby reducing tumor growth. Chemotherapy regimens typically include drugs such as paclitaxel, docetaxel, and carboplatin, which are known to be effective in treating breast cancer.

The treatment regimen involves administering bevacizumab in combination with chemotherapy to patients with breast cancer. Tumor samples are collected prior to the initiation of neoadjuvant treatment (NAT). These samples are then analyzed to measure the expression levels of the nine proteins included in the ViRP score. The ViRP score is calculated based on these measurements and used to predict the likelihood of achieving a pCR or having a low RCB.

### Relative Importance of Proteins and Modelling of Protein Combinations

The relative importance of each protein in the ViRP signature is determined using the R-package relaimpo with the lmg metric. The Syk protein is found to be of high importance, which is consistent with its role in immune cells and its involvement in angiogenesis. The other proteins in the signature also play significant roles in various biological processes related to tumor response and angiogenesis.

The ViRP score is developed using adaptive Lasso regression, which selects proteins based on their association with relative tumor size after treatment. The final signature consists of nine proteins, each with a corresponding beta coefficient. The score is calculated as the sum of the intercept and beta coefficient-weighted expression of these proteins.

The predictive performance of the ViRP score is evaluated using ROC curves, and the score is validated in an independent clinical cohort. The mRNA ViRP score, calculated using the corresponding genes from the ViRP signature, demonstrates a high correlation with the original protein ViRP score and is also validated in an external cohort.

## EXAMPLES

### Example 1: Identification of Ten Protein Prognostic Signature

#### Patient Cohort

To develop the ViRP score, protein expression profiles were established in the NeoAva phase II clinical trial. Patients with human epidermal growth factor receptor 2 (HER2)-negative, previously untreated breast carcinomas with a size of at least 2.5 cm were included. A total of 67 patients were randomly assigned to treatment with chemotherapy (CTx) and 71 patients to treatment with bevacizumab plus chemotherapy (Bev plus CTx). Tumor samples collected prior to treatment were available for protein analysis using reverse-phase protein arrays (RPPA) in the CTx treatment arm (N = 55) and in the Bev plus CTx treatment arm (N = 54).

#### Reverse Phase Protein Arrays (RPPA)

Protein expression profiling was performed using RPPA, which measures the expression levels of 210 cancer-relevant proteins, including 54 phosphorylated proteins. Tumor protein lysates were serially diluted and probed with antibodies, and the relative protein levels were determined and normalized for protein loading. All values were log2 transformed and median-centered across each antibody.

#### Statistical Analysis and Signature Development

Statistical analyses were conducted using R programming language. Low-variance proteins were filtered out using a mixed-model distribution, and adaptive Lasso regression was performed to select proteins for the ViRP score. The final signature consists of nine proteins with corresponding beta coefficients. The ViRP score is calculated as the sum of the intercept and beta coefficient-weighted expression of these proteins.

The ViRP score demonstrates a significant correlation with relative tumor size after treatment, pCR, and RCB. Patients with a lower ViRP score are more likely to respond positively to the treatment. The predictive performance of the ViRP score is evaluated using ROC curves, and the score is validated in an independent clinical cohort.

#### Validation of the ViRP Score

The mRNA ViRP score, calculated using the corresponding genes from the ViRP signature, demonstrates a high correlation with the original protein ViRP score. The mRNA ViRP score is also validated in an external cohort, confirming its predictive performance.

In conclusion, the ViRP score provides a robust and reliable method for predicting the response to treatment with bevacizumab plus chemotherapy in patients with breast cancer. The score can be used to identify patients who are likely to benefit from this treatment regimen, thereby improving clinical outcomes and reducing unnecessary treatments.