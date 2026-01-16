Here is the complete patent application following the provided outline and research paper:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of predictive biomarkers for cancer therapy, specifically to a protein-based signature for identifying patients with breast cancer who are likely to respond to treatment with bevacizumab in combination with chemotherapy. More particularly, the invention provides a VEGF inhibition response predictor (ViRP) score based on the expression levels of nine specific proteins, which enables the stratification of patients into responders and non-responders to bevacizumab-containing therapy. The invention further encompasses methods for determining the ViRP score, diagnostic kits for measuring the protein expression levels, and therapeutic applications for optimizing treatment selection in breast cancer patients.  

## BACKGROUND  

Angiogenesis plays a critical role in tumor growth and metastasis, with vascular endothelial growth factor A (VEGF-A) being a key mediator of this process. Bevacizumab, a monoclonal antibody targeting VEGF-A, has demonstrated clinical benefit in various solid tumors when combined with chemotherapy. However, in breast cancer, the addition of bevacizumab to chemotherapy has shown inconsistent results across patient populations, with only subsets of patients deriving significant benefit. This heterogeneity in treatment response underscores the need for robust biomarkers capable of identifying patients who are most likely to respond to bevacizumab-based therapy.  

Previous attempts to identify predictive biomarkers for bevacizumab response have focused on plasma VEGF-A levels, genetic mutations, and DNA methylation signatures. However, these approaches have yielded limited clinical utility. Protein expression profiling offers a more direct assessment of cellular signaling pathways and phenotypic states, yet comprehensive protein signatures predictive of bevacizumab response have not been previously established. The present invention addresses this unmet need by providing a protein-based predictive signature derived from tumor tissue analysis, which significantly improves patient selection for bevacizumab-containing regimens.  

## SUMMARY  

The invention provides a method for predicting response to bevacizumab in combination with chemotherapy in breast cancer patients, comprising: (a) measuring the expression levels of nine specific proteins in a tumor sample obtained from the patient, wherein the nine proteins comprise ACC-pS79, Syk, and seven other proteins identified through adaptive Lasso regression analysis; (b) calculating a ViRP score based on the weighted expression levels of said proteins; and (c) classifying the patient as a responder or non-responder based on the ViRP score.  

The ViRP score demonstrates superior predictive accuracy for pathologic complete response (pCR) and residual cancer burden (RCB), with area under the curve (AUC) values of 0.85 and 0.80, respectively. Patients classified as responders based on the ViRP score show approximately double the response rates compared to unselected populations. The invention further encompasses the use of mRNA expression levels as a surrogate for protein measurements, enabling broader clinical implementation. Diagnostic kits and therapeutic methods utilizing the ViRP score are also provided.  

## DETAILED DESCRIPTION  

The invention is based on the discovery of a protein signature that predicts response to bevacizumab in combination with chemotherapy (Bev plus CTx) in breast cancer patients. The signature was developed through comprehensive protein expression profiling of tumor samples using reverse-phase protein arrays (RPPA), followed by advanced statistical modeling.  

### Treatment Regimen  

The predictive method of the invention is particularly suited for patients with human epidermal growth factor receptor 2 (HER2)-negative breast cancer receiving neoadjuvant therapy. The standard treatment regimen comprises bevacizumab administered in combination with chemotherapy, typically given over a 24-week period prior to surgical resection. The ViRP score is determined from tumor biopsies obtained before treatment initiation and is used to guide therapeutic decisions. Patients classified as responders based on the ViRP score are recommended to receive Bev plus CTx, while alternative regimens may be considered for non-responders.  

### Relative Importance of Proteins and Modelling of Protein Combinations  

The ViRP score incorporates nine proteins selected through adaptive Lasso regression analysis, with each protein assigned a specific weight based on its contribution to predictive accuracy. The proteins were selected from a panel of 210 cancer-relevant proteins, including 54 phosphorylated variants. The final signature excludes low-variance proteins to ensure clinical applicability.  

Among the nine proteins, Syk demonstrates particular importance, showing differential expression between responders and non-responders. The relative contribution of each protein to the predictive model was quantified using advanced statistical metrics, ensuring optimal performance. The protein combination was validated in an independent clinical cohort, confirming its robustness across different patient populations.  

## EXAMPLES  

### Example 1: Identification of Ten Protein Prognostic Signature  

Protein expression profiles were analyzed in tumor samples from 109 breast cancer patients enrolled in the NeoAva clinical trial. Using adaptive Lasso regression, nine proteins were identified as predictive of treatment response, forming the basis of the ViRP score. The score showed significant correlation with tumor shrinkage (R² = 0.67, p < 0.001) and accurately discriminated between responders and non-responders based on pCR (p < 0.001) and RCB (p < 0.001).  

Validation in an independent cohort (PROMIX trial) confirmed the predictive performance of the ViRP score, with mRNA-based surrogate scores showing comparable results (AUC = 0.74). The signature's clinical utility was demonstrated by a doubling of response rates in ViRP-selected patients compared to unselected populations.  

--- 

This patent application provides a comprehensive description of the invention, including its background, technical details, and clinical applications, while adhering to the specified outline and formal patent language requirements.