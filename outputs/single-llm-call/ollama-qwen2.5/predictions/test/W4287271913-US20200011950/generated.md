### I. Introduction

Glioblastoma (GBM) is an aggressive brain tumor with poor prognosis. Current survival prediction methods primarily focus on features within the visible tumor and peri-tumoral regions. However, the mass effect of GBM causes significant deformation in surrounding healthy brain tissue, which may provide additional prognostic value. This study introduces r-DepTH, a radiomic descriptor that integrates biomechanical deformation features from the brain around the tumor (BAT) with morphological features from the tumor and peri-tumoral regions.

### II. Methods

r-DepTH was developed by combining 3D COLLAGE descriptors, which measure local heterogeneity via higher-order statistics of local gradient tensors, with deformation features extracted from the BAT region. The deformation features were quantified at various distances (5mm to 60mm) from the tumor boundary. These integrated features were then used to predict overall survival in a GBM cohort using machine learning techniques.

### III. Data and Cohort

The study utilized a multi-institutional cohort of GBM patients with MRI scans acquired on different scanners. The dataset was divided into training (S_t) and test (S_v) sets. Clinical and molecular data, such as IDH mutation status, MGMT methylation, and extent of resection (EOR), were available for a subset of the patients but not all.

### IV. Feature Extraction

COLLAGE features were extracted from the tumor and peri-tumoral regions, capturing textural heterogeneity. Deformation features were calculated by registering pre- and post-treatment MRI scans to quantify the displacement of tissue structures in the BAT region. These deformation magnitudes were then summarized using higher-order statistics (mean, standard deviation, skewness, kurtosis) at different distances from the tumor boundary.

### V. Results

The r-DepTH descriptor demonstrated significant differences between high-risk (H-R) and low-risk (L-R) groups in both S_t and S_v sets. Specifically, deformation features like skewness and kurtosis at 5mm and 10mm were strong predictors of poor survival. The integrated feature set outperformed radiomic and deep learning approaches that only considered intra-tumor and peri-tumoral regions.

### VI. Discussion

Our findings suggest that incorporating biomechanical deformation features from the BAT region improves survival prediction in GBM. Previous studies have shown that textural features from both intra- and peri-tumoral regions enhance diagnostic and prognostic models. By adding deformation features, r-DepTH provides a more comprehensive characterization of the tumor microenvironment, capturing the mass effect's impact on surrounding tissue.

### VII. Conclusion

r-DepTH, an integrated radiomic descriptor combining biomechanical deformation features from the BAT region with morphological features from the tumor and peri-tumor regions, improves survival prediction in GBM patients. Future work will focus on integrating direction attributes of tissue deformations and validating r-DepTH in larger multi-site cohorts and prospectively collected scans.

### VIII. Future Work

Future research will extend r-DepTH by incorporating phase attributes of tissue deformation to build a more comprehensive prognostic signature. Additionally, we plan to validate r-DepTH in large multi-site retrospective cohorts and eventually in prospectively collected scans to establish its utility as a robust prognostic marker for GBM and other solid tumors.