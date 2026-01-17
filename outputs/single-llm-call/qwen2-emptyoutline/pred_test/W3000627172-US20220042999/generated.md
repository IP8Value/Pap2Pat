# DESCRIPTION

## BACKGROUND

Breast cancer (BC) is a significant health concern, with approximately 20% of screen-detected cases being ductal carcinoma in situ (DCIS). DCIS is a pre-invasive form of BC characterized by malignant epithelial cells confined to the lumen of a mammary duct, without invasion into the adjacent stroma. Despite its non-invasive nature, 20-53% of women with untreated DCIS may progress to invasive BC over a period of at least 10 years. Current treatment strategies for DCIS primarily involve surgical excision, often accompanied by local radiotherapy (RT) and, in some cases, endocrine blockade. However, these treatments are not without drawbacks, as 10-35% of DCIS patients treated with lumpectomy or breast conservation surgery (BCS) experience local recurrence (LR), with about half of these recurrences manifesting as invasive breast cancer (IBC).

A major challenge in the management of DCIS is the inability to reliably predict which patients are at high risk of recurrence. Current prognostic tools, such as the Van Nuys Prognostic Index (VNPI) and the Memorial Sloan Kettering DCIS nomogram, rely on clinicopathological parameters and lack consistency and reproducibility in risk prediction. Additionally, these tools do not integrate molecular predictors, leading to an underestimation of DCIS heterogeneity. The Oncotype Dx Breast DCIS score, a gene-expression based assay, has shown some value in predicting LR but has been validated in only two cohorts, raising questions about its prognostic reliability.

Intratumoral heterogeneity (ITH) is a characteristic feature of DCIS, and higher ITH is associated with a greater likelihood of LR and invasive BC. Centrosome amplification (CA), which involves an abnormal increase in the number and/or volume of centrosomes, is a well-recognized driver of ITH. CA is an early event in tumorigenesis and is associated with higher tumor grade, larger tumor size, and increased risk of recurrence and metastasis. Despite its potential prognostic value, there is currently no methodology available for the rigorous quantitation of CA phenotypes in clinical tissue samples.

## SUMMARY

The present invention provides a novel methodology for centrosomal phenotyping to quantitatively assess both numerical and structural centrosomal aberrations in clinical tissue samples. The method involves immunofluorescent staining of centrosomes using an antibody against γ-tubulin and co-staining of nuclei with Hoechst. The analytical procedure allows for the robust interrogation of the capacity of centrosomal overload to predict the risk of LR after lumpectomy.

The invention includes an algorithm that quantitates the frequency and severity of centrosomal amplification (CA) in formalin-fixed paraffin-embedded (FFPE) clinical samples. The algorithm computes a centrosome amplification score (CAS) for each sample, which is a composite metric that integrates numerical and structural CA. CAS is a promising prognostic tool that can improve treatment recommendations and identify patients at low risk of recurrence who may not require adjuvant RT.

The key features of the invention are:
1. **Immunofluorescent Staining**: Centrosomes are stained using an antibody against γ-tubulin, and nuclei are stained with Hoechst.
2. **Image Acquisition**: Confocal microscopy is used to acquire high-resolution images of the stained tissue samples.
3. **Data Processing**: Raw 3D image data are processed using 3D volume rendering software to determine the volume of each centrosome.
4. **Categorization of Centrosomes**: Centrosomes are categorized into individually distinguishable centrosomes (iCTRs) and megacentrosomes (mCTRs) based on their volume.
5. **Algorithm-Based Analytics**: A cumulative CAS is computed based on the frequency and severity of numerical and structural CA.
6. **Statistical Analysis**: The CAS is used to stratify patients into high- and low-risk groups for LR, and its prognostic value is validated using statistical methods.

## DETAILED DESCRIPTION

### DISCUSSION

The invention addresses the critical need for a reliable and robust method to predict the risk of local recurrence (LR) in patients with ductal carcinoma in situ (DCIS). The methodology involves a multi-step process that combines immunofluorescent staining, confocal microscopy, and algorithm-based analytics to quantitatively assess centrosomal amplification (CA) in clinical tissue samples.

#### Immunofluorescent Staining and Confocal Microscopy Imaging

1. **Sample Preparation**: Formalin-fixed paraffin-embedded (FFPE) tissue sections of DCIS are prepared using standard methods. The tissue blocks are stored in a tissue bank and are retrieved for analysis.
2. **Staining**: Centrosomes are immunofluorescently stained using an antibody against γ-tubulin, which labels the centrosomes. Nuclei are co-stained with Hoechst, a DNA-binding fluorescent dye.
3. **Imaging**: High-resolution images of the stained tissue samples are acquired using a Zeiss LSM 700 confocal microscope equipped with a 63x oil immersion lens. The imaging parameters are optimized to ensure accurate and consistent results. Laser power and detector gain are adjusted to prevent over- and under-saturation, and the offset is set to minimize background noise.

#### Data Processing and Centrosome Categorization

1. **3D Volume Rendering**: Raw 3D image data are processed using IMARIS Biplane 8.2 3D volume rendering software. Background subtraction is applied to exclude non-specific signals, and the volume of each centrosome is determined.
2. **Categorization**: Centrosomes are categorized into individually distinguishable centrosomes (iCTRs) and megacentrosomes (mCTRs) based on their volume. iCTRs are defined as centrosomes with volumes within the range of centrosome volumes found in normal breast tissue. mCTRs are defined as centrosomes with volumes exceeding the upper limit of the normal range and are considered to represent structurally amplified centrosomes.

#### Algorithm-Based Analytics

1. **Numerical CA Quantitation**: The frequency and severity of numerical CA are quantitated using the following formula:
   \[
   \text{CAS}_i = \left( \frac{\sum_{i=1}^{N} (N_i - R_{\text{th}})}{R} \right) \times \left( \frac{p_i}{\beta_i} \right)
   \]
   where:
   - \( R_{\text{th}} \) is the highest number of centrosomes present in a normal breast cell (2).
   - \( N_i \) is the number of iCTRs in a cell that contains more than 2 iCTRs.
   - \( R \) is the range of values for the number of centrosomes present in a normal cell (2).
   - \( p_i \) is the percentage of cells with more than 2 iCTRs.
   - \( \beta_i \) is a scaling factor (0.1 for breast tissue).

2. **Structural CA Quantitation**: The frequency and severity of structural CA are quantitated using the following formula:
   \[
   \text{CAS}_m = \left( \frac{\sum_{i=1}^{N} \left( \frac{V_i - V_{\text{critical}}}{\sigma_{V_i}} \right) \times N_i}{N} \right) \times \left( \frac{p_m}{\beta_m} \right)
   \]
   where:
   - \( V_i \) is the volume of the \( i \)-th mCTR.
   - \( V_{\text{critical}} \) is the maximum volume of a normal centrosome (0.735 µm³ for breast tissue).
   - \( \sigma_{V_i} \) is the standard deviation of the volume of mCTRs.
   - \( p_m \) is the percentage of cells with mCTRs.
   - \( \beta_m \) is a scaling factor (0.148 for breast tissue).

3. **Composite CAS Calculation**: The cumulative centrosome amplification score (CAS) is computed by summing the numerical and structural CA scores:
   \[
   \text{CAS}_{\text{total}} = \text{CAS}_i + \text{CAS}_m
   \]

#### Statistical Analysis and Validation

1. **Cohort Selection**: The methodology is validated using two large, well-characterized cohorts: a discovery cohort (DC) and a validation cohort (VC). The DC consists of 133 consecutive pure DCIS patients, and the VC consists of 119 consecutive pure DCIS patients. Both cohorts include patients with available tissue samples and clinicopathological data.
2. **Survival Analysis**: The CAS is used to stratify patients into high- and low-risk groups for LR. Kaplan-Meier survival analysis is performed to assess the association between CAS and recurrence-free survival (RFS). The optimal cutoff for CAS is determined using log-rank tests.
3. **Multivariable Cox Regression**: Univariate and multivariable Cox proportional hazard models are built to estimate hazard ratios (HRs) and 95% confidence intervals (CIs) between high- and low-CAS groups, controlling for potential confounders such as age, grade, comedo necrosis, and RT.
4. **Performance Metrics**: The performance of CAS in predicting LR is evaluated using sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), and accuracy. The Harrell's concordance index is used to assess the discriminatory power of the survival model.

#### Clinical Significance

1. **Prognostic Value**: The invention demonstrates that patients with higher CAS are at significantly higher risk of LR compared to those with lower CAS. CAS outperforms traditional clinicopathological variables and the Van Nuys Prognostic Index (VNPI) in predicting LR.
2. **Therapeutic Decision Making**: CAS can be used as a clinical tool to identify patients who can be safely treated with BCS/mastectomy alone and those who will benefit from the inclusion of RT. This can help reduce overtreatment and improve patient outcomes.
3. **Future Directions**: The methodology can be extended to other cancer types with rampant CA and can be refined for broader applicability. Future research will focus on validating the method in external cohorts and understanding the mechanistic role of CA-associated proteins in DCIS progression.

In conclusion, the invention provides a novel and robust methodology for centrosomal phenotyping that can significantly improve the management of DCIS by predicting the risk of LR and guiding therapeutic decisions.