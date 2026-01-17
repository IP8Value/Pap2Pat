# DESCRIPTION

## BACKGROUND OF THE DISCLOSURE

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterized by impairments in social communication and interaction, as well as the presence of restricted, repetitive patterns of behavior. The prevalence of ASD has increased significantly over the past few decades, from 0.64% in 2002 to 1.14% in 2008, surpassing the rates of other developmental disabilities. Despite the high prevalence and the substantial impact on quality of life and healthcare costs, the biochemical basis of ASD remains largely unknown. Current diagnostic methods rely heavily on psychometric tools, but there is a growing interest in identifying biomarkers and developing therapeutic strategies for ASD.

Post-mortem brain analysis has revealed several structural and functional abnormalities associated with ASD, including altered synapse connectivity and plasticity, decreased neuron size and increased neuron density in the amygdala and hippocampus, decreased Purkinje cell size and number in the cerebellum, neuroinflammation, and aberrant activity-dependent transcription and translation. Molecular studies have implicated alterations in Wnt/β-catenin signaling, Ca²⁺ signaling, and glutamatergic/GABAergic signaling in the pathophysiology of ASD. Polyunsaturated fatty acids (PUFAs), particularly docosahexaenoic acid (DHA) and arachidonic acid (AA), are essential components of neuronal phospholipids and play crucial roles in neuroplasticity, neurogenesis, and synaptogenesis. These PUFAs are derived from dietary precursors, such as α-linolenic acid (ALA) and linoleic acid (LA), and have been investigated as potential biomarkers and therapeutic targets for ASD.

Previous studies have reported differences in erythrocyte-membrane and plasma fatty acid profiles between individuals with ASD and neurotypical (NEU) controls. However, the results have been inconsistent, often due to variations in sample size, methodology, and the way data are reported. Some studies have focused on absolute fatty acid concentrations, while others have used relative concentrations, leading to different conclusions. The ability of fatty acid measurements to accurately classify individuals with ASD from NEU controls is a critical factor in determining their utility as biomarkers.

## SUMMARY OF THE DISCLOSURE

The present disclosure relates to a method for assessing the utility of erythrocyte-membrane fatty acid profiles as biomarkers for Autism Spectrum Disorder (ASD). The method involves comparing the levels of specific fatty acids in erythrocyte membranes between a large cohort of individuals with ASD and a control group of neurotypical (NEU) individuals. The invention provides a comprehensive statistical analysis to evaluate the ability of individual and multivariate fatty acid measurements to distinguish between ASD and NEU participants.

The method includes the following steps:
1. **Study Population Selection**: Recruiting a large cohort of individuals with ASD and a control group of NEU individuals, ensuring that both groups are matched for age and gender. Participants are excluded if they have taken nutritional supplements or followed abnormal diets in the past two months.
2. **Fatty Acid Measurement**: Quantifying erythrocyte-membrane fatty acids using a reliable and standardized method, such as gas chromatography with flame ionization detection.
3. **Statistical Analysis**:
   - **Hypothesis Testing**: Conducting hypothesis tests to determine significant differences in mean or median fatty acid concentrations between the ASD and NEU groups.
   - **Classification Analysis**: Using receiver-operating characteristic (ROC) curve analysis to assess the ability of individual fatty acids to classify ASD and NEU participants. Additionally, performing Fisher Discriminant Analysis (FDA) to evaluate the classification performance of multivariate combinations of fatty acids.
4. **Data Interpretation**: Interpreting the results to determine the utility of erythrocyte-membrane fatty acid profiles as biomarkers for ASD. The invention highlights the importance of evaluating biomarkers at the individual level rather than solely relying on population-level differences.

The results of the disclosed method indicate that while some individual fatty acids show statistically significant differences between ASD and NEU groups, they do not provide sufficient discriminatory power to serve as robust biomarkers for ASD. The multivariate classification using FDA shows a moderate improvement but still falls short of the necessary diagnostic accuracy. The invention emphasizes the need for further research and the importance of considering individual-level classification in biomarker studies.

## DETAILED DESCRIPTION

### Study Population Selection

The method begins with the selection of a large and well-characterized study population. The cohort consists of 63 individuals with ASD and 49 NEU individuals, with a median age of 9.7 years and 10.0 years, respectively. Both groups are matched for age and gender to minimize confounding factors. Participants are excluded if they have taken nutritional supplements or followed abnormal diets in the past two months. This exclusion criterion ensures that the fatty acid profiles are not influenced by external factors that could skew the results.

### Fatty Acid Measurement

Erythrocyte-membrane fatty acids are quantified using a reliable and standardized method. The fatty acids are extracted from red blood cells and derivatized to their methyl esters. Gas chromatography with flame ionization detection is employed to measure the concentrations of various fatty acids, including arachidonic acid (AA), dihomo-γ-linoleic acid (DGLA), docosahexaenoic acid (DHA), eicosapentaenoic acid (EPA), elaidic acid, linoleic acid, oleic acid, palmitelaidic acid, palmitic acid, palmitoleic acid, and stearic acid. All fatty acid measurements are normalized by the concentration of total fatty acids in the sample to ensure consistency and comparability.

### Statistical Analysis

#### Hypothesis Testing

The first step in the statistical analysis is to conduct hypothesis tests to determine significant differences in mean or median fatty acid concentrations between the ASD and NEU groups. Individual measurements for each cohort are assessed for normality using the Anderson-Darling test at a significance level of 0.05. If the distributions from both cohorts fail to reject the null hypothesis of the Anderson-Darling test, the F-test for equal variances is performed to decide whether to use a Student’s t-test or Welch’s test. If the distributions from one or more cohorts reject the null hypothesis of the Anderson-Darling test, the two-sample Kolmogorov-Smirnov test is used to test whether the samples come from distributions of the same shape. If the distributions fail to reject the null hypothesis of the Kolmogorov-Smirnov test, the Mann-Whitney U test is used to test for significant differences in median values; otherwise, Welch’s test is used to test for significant differences in mean values. All statistical tests are performed using MATLAB, and the results are visualized using kernel density estimation (KDE).

#### Classification Analysis

Univariate classification for each fatty acid measurement is assessed using receiver-operating characteristic (ROC) curve analysis. The C-statistic, which is the area under the ROC curve, is calculated to quantify the ability of each fatty acid to classify ASD and NEU participants. A C-statistic of 0.5 indicates a random separation, while a C-statistic of 1 indicates a perfect separation. The results show that none of the individual fatty acids provide sufficient discriminatory power to serve as robust biomarkers for ASD.

Multivariate classification is then assessed using Fisher Discriminant Analysis (FDA). All variables presented in the univariate analysis are included in the FDA analysis. The probability distributions of the FDA scores are visualized, and the C-statistic is calculated to evaluate the classification performance of the multivariate model. The results indicate a moderate improvement in classification accuracy but still fall short of the necessary diagnostic accuracy.

### Data Interpretation

The interpretation of the results is a critical step in the method. The invention highlights the importance of evaluating biomarkers at the individual level rather than solely relying on population-level differences. While some individual fatty acids show statistically significant differences between ASD and NEU groups, the overlap in the probability distributions indicates that they do not provide sufficient discriminatory power to serve as robust biomarkers for ASD. The multivariate classification using FDA shows a moderate improvement but still falls short of the necessary diagnostic accuracy.

The invention also emphasizes the need for further research and the importance of considering individual-level classification in biomarker studies. The results suggest that erythrocyte-membrane fatty acid profiles are not promising biomarkers for classifying ASD and NEU children. A repository of individual-level measurements in biomarker studies for ASD, including those reporting negative results, would greatly help the field iterate toward more promising biomarkers for classifying ASD.

### EXAMPLES

#### Example 1: Univariate Classification of Erythrocyte-Membrane Fatty Acids

In this example, the univariate classification of erythrocyte-membrane fatty acids is assessed using ROC curve analysis. The results show that none of the individual fatty acids provide sufficient discriminatory power to serve as robust biomarkers for ASD. For instance, the C-statistic for arachidonic acid (AA) is 0.51, indicating a random separation between ASD and NEU participants. Similar results are observed for other fatty acids, such as DGLA (C-statistic = 0.62), DHA (C-statistic = 0.50), and EPA (C-statistic = 0.51).

#### Example 2: Multivariate Classification Using Fisher Discriminant Analysis (FDA)

In this example, multivariate classification is assessed using Fisher Discriminant Analysis (FDA). All variables from the univariate analysis are included in the FDA analysis. The probability distributions of the FDA scores are visualized, and the C-statistic is calculated to evaluate the classification performance of the multivariate model. The results show a moderate improvement in classification accuracy, with a C-statistic of 0.76. However, this still falls short of the necessary diagnostic accuracy for a robust biomarker.

#### Example 3: Regression Analysis of Fatty Acids with Seafood Intake

In this example, the red blood cell fatty acid compositions are regressed onto the number of seafood meals per month. The results show that increased seafood consumption is modestly correlated with increased DHA and EPA in both ASD and NEU cohorts. For example, the correlation coefficient for DHA is 0.347 in the ASD cohort and 0.477 in the NEU cohort. These results support the link between dietary seafood intake and increased levels of EPA and DHA.

#### Example 4: Comparison with Previous Studies

In this example, the results of the present study are compared with previous studies to highlight the importance of evaluating biomarkers at the individual level. For instance, despite the statistically significant lower AA in the ASD population reported in previous studies, the probability distributions indicate only slight separation between the groups at the individual level (C-statistic = 0.62 and 0.75, respectively). Similar results are observed for DHA, where a statistically significant difference in mean DHA levels does not indicate its usefulness as a biomarker for ASD status.

These examples demonstrate the utility of the disclosed method in assessing the potential of erythrocyte-membrane fatty acid profiles as biomarkers for ASD and highlight the importance of considering individual-level classification in biomarker research.