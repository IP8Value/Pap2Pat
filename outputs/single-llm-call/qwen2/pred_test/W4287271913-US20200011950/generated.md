# DESCRIPTION

## FEDERAL FUNDING NOTICE

This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## BACKGROUND

Cancer, particularly glioblastoma (GBM), is a complex and aggressive disease characterized by heterogeneous growth patterns that extend beyond visible surgical margins. The impact of GBM is not confined to the visible tumor but also affects the immediate peri-tumoral and seemingly normal-appearing adjacent regions, a phenomenon known as the "tumor field effect." GBM's infiltrative nature and the mass effect it exerts on surrounding tissues lead to significant alterations in brain structure and function, contributing to poor patient outcomes. Despite advances in radiomics and deep learning, there remains a gap in leveraging the subtle tumor-induced deformations in the brain around the tumor (BAT) as a complementary marker for prognosticating patient survival.

Recent studies have shown the potential of radiomic features and deep learning approaches in survival prediction and response assessment for brain tumors. However, these methods primarily focus on local textural changes within the tumor and peri-tumoral regions, neglecting the biomechanical deformations in the BAT. This oversight limits the comprehensive characterization of the disease and its field effect. To address this gap, the present invention introduces an integrated MRI-based descriptor, termed radiomic-Deformation and Textural Heterogeneity (r-DepTH), which captures both morphological and biomechanical attributes of the tumor and its surroundings.

The r-DepTH descriptor combines measurements of biomechanical deformations in the BAT regions with textural radiomic features from the visible tumor margins and peri-tumoral regions. By integrating these features, the r-DepTH descriptor aims to provide a more comprehensive characterization of the tumor field effect, enabling better risk stratification and prognosticating overall patient survival in GBM.

## DETAILED DESCRIPTION

### Introduction

Glioblastoma (GBM) is one of the most aggressive and lethal forms of brain cancer, characterized by rapid and invasive growth. The tumor's impact extends beyond its visible margins, affecting the surrounding brain tissue through a phenomenon known as the "tumor field effect." This effect includes both morphological changes within the tumor and biomechanical deformations in the brain around the tumor (BAT). These deformations can provide valuable prognostic cues, as more aggressive tumors exert greater pressure on surrounding tissues, leading to pronounced structural changes and worse outcomes.

### Previous Work and Novel Contributions

The concept of characterizing the tumor field effect using routine MRI scans has gained significant attention in recent years. Previous studies have developed deterministic mathematical models to understand the mechanical stress and mass effect caused by GBM tumors. These models consider factors such as cellular motility and diffusion coefficients to account for the tumor's invasiveness and the resulting tissue deformations. Additionally, data-driven approaches, such as radiomic features and deep learning, have been employed to extract texture and shape information from the tumor and peri-tumoral regions for survival prediction.

However, these approaches have primarily focused on local textural changes and have not explicitly accounted for the biomechanical deformations in the BAT. The present invention addresses this gap by introducing the r-DepTH descriptor, which combines biomechanical deformation features from the BAT with textural radiomic features from the tumor and peri-tumoral regions. The r-DepTH descriptor aims to provide a more comprehensive characterization of the tumor field effect, enabling better risk stratification and survival prediction in GBM patients.

### Methodology

#### Notation

An image scene \( I \) is defined as \( I = (C, f) \), where \( I \) is a spatial grid of voxels \( c \in C \) in a 3-dimensional space \( \mathbb{R}^3 \). Each voxel \( c \in C \) is associated with an intensity value \( f(c) \). The sub-volumes \( I_T \), \( I_P \), and \( I_B \) correspond to the intra-tumoral, peritumoral, and surrounding normal parenchymal regions within every image \( I \), respectively, such that \( [I_T, I_P, I_B] \subset I \). The sub-volume \( I_B \) is further divided into uniformly sized annular sub-volumes \( I_{Bj} \), where \( j \) is the number of uniformly-sized annular bands, and \( j \in \{1, \ldots, m\} \), with \( m \) being a user-defined proximity parameter dependent on the distance from the tumor margin.

#### r-DepTH Descriptor

1. **Deformation Heterogeneity Features from the Normal Parenchyma**

   To measure the tissue deformation in the normal appearing brain regions, a healthy T1-weighted MNI atlas \( I_{\text{Atlas}} \) is used. The atlas is non-rigidly aligned to the patient volume \( I \) using the mutual-information-based similarity measure provided in the ANTs (Advanced Normalization Tools) SyN (Symmetric Normalization) toolbox. This toolbox is chosen for its efficiency in mapping brain images containing lesions into healthy templates and handling constrained cost-function masking, where the mapping within a tumor-exclusive region is determined by the solution of the negative tumor mask region \( I_{\text{mask}} \).

   The non-rigid alignment can be formulated as:
   \[
   (I, I_{\text{mask}}) = T_r(I_{\text{Atlas}})
   \]
   where \( T_r(\cdot) \) is the forward transformation of the composite voxel-wise deformation field (including affine components) that maps the displacements of the voxels between the reference and floating volumes. The inverse transformation \( T_r^{-1}(\cdot) \) maps \( I \) to the \( I_{\text{Atlas}} \) space, yielding the tissue deformation of \( I \) with respect to \( I_{\text{Atlas}} \).

   The displacement vector for every voxel \( c \in C_{Bj} \) is given by:
   \[
   (c_t, c_u, c_v) = (c_t, c_u, c_v) + (\delta t, \delta u, \delta v)
   \]
   where \( (\delta t, \delta u, \delta v) \) are the scalar values of the deformation orientations. The magnitude of deformation is calculated using the Euclidean norm of the deformation orientations. First-order statistics (mean, median, standard deviation, skewness, and kurtosis) are then calculated by aggregating the deformation magnitudes \( D(c) \) for every voxel within each annular sub-volume \( I_{Bj} \), yielding a feature descriptor \( F_{Bj} \) for each annular sub-region.

2. **3D COLLAGE Features from Within the Tumor Confinements**

   The 3D COLLAGE descriptor captures intra-tumoral heterogeneity by calculating local per-voxel gradient orientations. For every voxel \( c \), intensity gradients in the X, Y, and Z directions are calculated, followed by centering a 3D window around each voxel to compute the vector gradient matrix \( F \). Two principal orientations, \( \theta(c) \) and \( \phi(c) \), are obtained from \( F \), and two separate co-occurrence matrices, \( M_\theta \) and \( M_\phi \), are computed to capture orientation pairs between voxels in a local neighborhood. From each co-occurrence matrix, 13 Haralick statistics are calculated for every voxel \( c \). First-order statistics (mean, median, standard deviation, skewness, and kurtosis) are then obtained for every voxel within the enhancing lesion \( C_T \) and the T2/FLAIR hyperintense peri-lesional component \( C_P \), yielding feature descriptors \( F_T \) and \( F_P \).

3. **Construction of r-DepTH Descriptor**

   The r-DepTH descriptor is constructed for each patient by concatenating the deformation descriptor \( F_B \) with the COLLAGE texture descriptors \( F_T \) and \( F_P \):
   \[
   F_{\text{rDepTH}} = [F_B, F_T, F_P]
   \]
   The r-DepTH descriptor can be employed within supervised or unsupervised approaches for disease characterization. The algorithm for computing r-DepTH is provided in Algorithm 1.

### Preprocessing

Manual annotations of the MRI slices were performed by three experts using a hand-annotation tool in 3D Slicer. The senior-most expert independently annotated the studies obtained from one institution, while the other two experts supervised the annotation of cases from additional institutions. Disagreements were resolved by consulting the senior-most radiologist. Each tumor was annotated into two regions: the enhancing lesion \( I_T \) and the T2/FLAIR hyperintense peri-lesional component \( I_P \). The 3 MRI sequences (Gd-T1w, T2w, and FLAIR) were co-registered to a brain atlas (MNI152) using the ANTs SyN toolbox. Skull stripping was performed simultaneously during registration, and bias field correction was conducted using a non-parametric non-uniform intensity normalization technique.

### Implementation Details

Deformation magnitudes \( F_{Bj} \) were calculated for 12 annular regions, each 5mm apart, resulting in a 60 × 1 deformation vector. This vector included 5 statistics (mean, median, standard deviation, skewness, and kurtosis) for each of the 12 bands, yielding 60 features corresponding to \( F_B \). Similarly, 130 COLLAGE features were extracted from each of the two compartments \( F_T \) and \( F_P \), resulting in a total of 320 features for the r-DepTH descriptor.

### Survival Risk Assessment

Feature selection and reduction were conducted using the least absolute shrinkage and selection operator (LASSO) along with a Cox regression model. The top features selected by the LASSO model were used to create a continuous survival risk score (Risc) for each patient. The risk score was calculated as:
\[
Risc = \sum_{g=1}^{A} q_g F_g^\alpha
\]
where \( A \) is the number of selected imaging features, \( F_g^\alpha \) is the \( g \)-th feature for \( \alpha = \{T, P, B, \text{rDepTH}\} \), and \( q_g \) is the respective coefficient. Patients were classified into high-risk (H-R) and low-risk (L-R) groups using a grid-search to find an optimal threshold. Kaplan-Meier (KM) survival analysis and log-rank tests were performed to assess the differences in survival rates between the identified groups. Performance measures such as hazard ratios (HR), 95% confidence intervals (CI), and concordance index (C-index) were obtained to evaluate the predictive ability of the survival models.

### Comparative Strategies

To evaluate the efficacy of r-DepTH for GBM survival prediction, the following comparisons were performed:

1. **Clinical Features**: Age, gender, tumor volume, and molecular information (MGMT, IDH, and extent of resection) were evaluated in univariate and multivariate settings.
2. **Collage Features from Tumor and Peri-Tumoral Regions**: The risk score \( Risc(F_{T,P}) \) was calculated using features from \( F_T \) and \( F_P \), along with age and gender.
3. **Deformation Features from Tumor and Peri-Tumoral Regions**: The risk score \( Risc(F_B) \) was calculated using features from \( F_B \), along with age and gender.
4. **State-of-the-Art Radiomics and CNN Approaches**: The performance of r-DepTH was compared to state-of-the-art radiomic and deep learning approaches previously used in the literature for GBM prognosis.

### Results

The LASSO survival analysis using r-DepTH yielded 9 features that demonstrated significant differences between the high-risk and low-risk survival groups on both the training and validation sets. The Kaplan-Meier curves showed significant differences in survival rates, with p-values of \( 3.5 \times 10^{-7} \) and \( 0.0024 \) for the training and validation sets, respectively. The C-index obtained using r-DepTH was 0.80, indicating strong predictive ability.

### Discussion

Highly aggressive tumors like GBM tend to proliferate beyond their visible confines, causing mass effect and structural deformations in the surrounding brain tissue. Previous studies have focused on extracting radiomic features from within the tumor and peri-tumoral regions, but none have explicitly leveraged biomechanical deformation attributes from the BAT. The r-DepTH descriptor addresses this gap by combining deformation features from the BAT with textural radiomic features from the tumor and peri-tumoral regions, providing a more comprehensive characterization of the tumor field effect.

The results of this study demonstrate that the r-DepTH descriptor significantly improves the prognostic model for predicting overall survival in GBM patients. The inclusion of deformation features from the BAT, along with textural features from the tumor and peri-tumoral regions, allows for better risk stratification and survival prediction compared to using clinical variables, radiomic features, or deep learning approaches alone.

### Conclusion

In this work, we introduced the r-DepTH descriptor, an integrated radiomic descriptor that comprehensively characterizes the tumor field effect from the tumor, peri-tumor, and brain around tumor regions. The r-DepTH descriptor combines biomechanical deformation features from the BAT with textural radiomic features from the tumor and peri-tumoral regions, enabling improved risk stratification and survival prediction in GBM patients. Future work will involve integrating direction attributes of tissue deformation and extending the analysis to larger multi-site retrospective cohorts for further validation.