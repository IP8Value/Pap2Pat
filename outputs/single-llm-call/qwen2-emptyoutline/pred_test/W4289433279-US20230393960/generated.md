# DESCRIPTION

## BACKGROUND

In the realm of algorithmic decision-making, particularly within the domain of machine learning (ML), ensuring fairness and avoiding discriminatory practices are paramount. Anti-discrimination laws in the United States, such as those addressing disparate treatment and disparate impact, play a crucial role in regulating how algorithms interact with protected attributes. Disparate treatment focuses on preventing explicit discrimination by excluding protected attributes from the input data. For instance, in a fraud detection model, one might simply omit sensitive attributes like race or gender to comply with disparate treatment. However, this approach alone is insufficient to address the broader issue of disparate impact, which concerns the uneven outcomes that different protected groups may experience due to the algorithm's decisions.

Disparate impact is more challenging to mitigate because it requires balancing the outcomes across different groups. Various fairness measures, including equal opportunity, demographic parity, and equalized odds, have been proposed to address this issue. Equal opportunity, for example, aims to ensure that the true positive rates (TPR) are the same across different groups, while demographic parity seeks to equalize the overall acceptance or rejection rates. Equalized odds, a more stringent measure, requires both the TPR and false positive rates (FPR) to be equal across groups. Despite their theoretical appeal, these measures often face practical limitations in real-world applications, especially in industries like fraud detection where the cost of false positives and false negatives can vary significantly.

One of the main challenges in implementing these fairness measures is their assumption of binary sensitive attributes, which is not always applicable in scenarios with high-arity attributes (e.g., multiple countries or currencies). Additionally, many existing techniques require multiple sanitizations of the ML model to achieve fairness, which can be computationally expensive and impractical in real-time systems. Moreover, the dynamic nature of data, with daily fluctuations in traffic and fraud patterns, further complicates the enforcement of these fairness criteria.

To address these limitations, this invention introduces a relaxed version of the equalized odds fairness measure and a one-shot fairness heuristic. The relaxed equalized odds measure allows for a more practical and flexible approach to fairness, while the heuristic provides an efficient method to calibrate the model's outputs to conform to the relaxed measure. This innovation is particularly valuable for industries that require real-time decision-making and must handle multiple protected attributes simultaneously.

## SUMMARY

The present invention relates to a method and system for ensuring fairness in machine learning models, particularly in the context of fraud detection and other real-time decision-making applications. The invention introduces a relaxed version of the equalized odds fairness measure, which allows for a more practical and flexible approach to achieving fairness across multiple protected attributes. The relaxed equalized odds measure requires that the false positive rates (FPR) and true positive rates (TPR) across different attribute values lie within a specified range defined by the mean and standard deviation of these rates across all attribute values.

To implement the relaxed equalized odds measure, the invention provides a one-shot fairness heuristic. This heuristic involves the following key steps:

1. **Choice of Constraints**: The heuristic allows the end user to specify whether to enforce similar FPRs, TPRs, or both, depending on the business requirements and the relative costs of false positives and false negatives.

2. **Threshold Grid Initialization**: A linear grid of possible threshold values is initialized to calibrate the decision thresholds of the model across different attribute values.

3. **Performance Computation**: For each threshold value in the grid, the performance metrics (FPR and TPR) are computed for each attribute value. The heuristic then prunes threshold values that result in performance metrics significantly higher than the average.

4. **Selection of Optimal Thresholds**: From the pruned set of threshold values, the heuristic selects the thresholds that maximize a chosen selection metric (e.g., F1 score, F0.5, or F2) for each attribute value. This ensures that the selected thresholds not only conform to the relaxed equalized odds measure but also maintain the model's overall performance.

5. **Multiple Attribute Extension**: The heuristic is extended to handle multiple protected attributes. Two approaches are provided: a strong multiple attribute fairness approach, which ensures fairness across all sub-populations formed by combinations of attribute values, and a weak multiple attribute fairness approach, which ensures fairness across each attribute independently. The strong approach is recommended for a smaller number of attributes, while the weak approach is more suitable for a larger number of attributes.

The invention also includes a method for attribute pruning to reduce the computational complexity of the multiple attribute extension. This method identifies and removes highly dependent attributes, thereby focusing on independent attributes that contribute significantly to the fairness measure.

The proposed method and system offer several advantages over existing fairness techniques. They provide a more practical and flexible approach to achieving fairness, are computationally efficient, and can be applied to a wide range of real-world applications, including fraud detection, income prediction, and criminal recidivism.

## DETAILED DESCRIPTION

### Introduction

The present invention addresses the challenge of ensuring fairness in machine learning models, particularly in scenarios involving high-arity attributes and real-time decision-making. Traditional fairness measures, such as equalized odds, often require multiple sanitizations and assume binary sensitive attributes, making them impractical for many real-world applications. The invention introduces a relaxed version of the equalized odds measure and a one-shot fairness heuristic to overcome these limitations.

### Relaxed Equalized Odds Measure

The classical definition of equalized odds requires that the true positive rates (TPR) and false positive rates (FPR) be the same across different values of a protected attribute. However, this strict requirement is often impractical in real-world settings, especially when dealing with high-arity attributes. To address this, the invention proposes a relaxed version of the equalized odds measure.

Let \( D = \{d_1, d_2, \ldots, d_K\} \) be a protected attribute with arity \( K \). A fraud detector model \( F \) is said to satisfy relaxed equalized odds with respect to attribute \( D \) if the FPR and TPR for each attribute value \( d_i \) lie within a specified range defined by the mean and standard deviation of these rates across all attribute values. Mathematically, this can be expressed as:

\[ \left| \text{FPR}(d_i) - \mu_{\text{FPR}} \right| \leq n \sigma_{\text{FPR}} \]
\[ \left| \text{TPR}(d_i) - \mu_{\text{TPR}} \right| \leq n \sigma_{\text{TPR}} \]

where:
- \( \mu_{\text{FPR}} \) and \( \mu_{\text{TPR}} \) are the average FPR and TPR across all attribute values, respectively.
- \( \sigma_{\text{FPR}} \) and \( \sigma_{\text{TPR}} \) are the standard deviations of the FPR and TPR across all attribute values, respectively.
- \( n \) is a user-defined parameter that controls the strictness of the relaxation.

This relaxed measure captures the core philosophy of equalized odds while being more practical and flexible for real-world applications.

### One-Shot Fairness Heuristic

To implement the relaxed equalized odds measure, the invention provides a one-shot fairness heuristic. The heuristic involves the following key steps:

#### Choice of Constraints

The heuristic allows the end user to specify whether to enforce similar FPRs, TPRs, or both, depending on the business requirements and the relative costs of false positives and false negatives. This flexibility is crucial for different applications. For example, in fraud detection, false positives (accepting a fraudulent transaction) might be more costly than false negatives (declining a genuine transaction), leading the user to focus on similar FPRs.

#### Threshold Grid Initialization

A linear grid of possible threshold values is initialized to calibrate the decision thresholds of the model across different attribute values. For example, the grid might consist of threshold values ranging from 0.6 to 0.9 in increments of 0.01. The choice of the grid depends on the specific application and the desired level of granularity.

#### Performance Computation

For each threshold value in the grid, the performance metrics (FPR and TPR) are computed for each attribute value. This involves evaluating the model's predictions at each threshold and calculating the corresponding FPR and TPR. The heuristic then prunes threshold values that result in performance metrics significantly higher than the average. Specifically, threshold values are pruned if:

\[ \left| \text{FPR}(g) - \mu_{\text{FPR}} \right| > n \sigma_{\text{FPR}} \]
\[ \left| \text{TPR}(g) - \mu_{\text{TPR}} \right| > n \sigma_{\text{TPR}} \]

where \( g \) is a threshold value in the grid.

#### Selection of Optimal Thresholds

From the pruned set of threshold values, the heuristic selects the thresholds that maximize a chosen selection metric for each attribute value. The selection metric can be the F1 score, F0.5, or F2, depending on the user's preference. The F1 score treats false positives and false negatives equally, while F0.5 and F2 weight false positives and false negatives differently, respectively. The optimal thresholds are selected to ensure that the model not only conforms to the relaxed equalized odds measure but also maintains its overall performance.

#### Multiple Attribute Extension

The heuristic is extended to handle multiple protected attributes. Two approaches are provided:

1. **Strong Multiple Attribute Fairness**: This approach ensures that the FPRs and TPRs of every sub-population formed by combinations of attribute values conform to the relaxed equalized odds measure. For example, if the protected attributes are country and currency, the heuristic ensures fairness across all combinations of country and currency values. However, this approach can be computationally expensive, especially with a large number of attributes and high arity.

2. **Weak Multiple Attribute Fairness**: This approach ensures that the FPRs and TPRs per attribute conform to the relaxed equalized odds measure independently. For example, if the protected attributes are country and currency, the heuristic ensures fairness across each attribute independently. This approach is more computationally efficient and is recommended for a larger number of attributes.

### Attribute Pruning

To reduce the computational complexity of the multiple attribute extension, the invention includes a method for attribute pruning. This method identifies and removes highly dependent attributes, thereby focusing on independent attributes that contribute significantly to the fairness measure. The dependence between attributes is determined using the Chi-square statistic, and attributes with a p-value less than or equal to 0.01 are considered statistically independent and retained.

### Case Studies

#### Fraud Detection

The proposed heuristic was applied to a fraud detection model to ensure fairness across different countries. The model was trained to predict whether an online transaction is fraudulent, and the heuristic was used to calibrate the decision thresholds per country. The results showed that the final model with custom thresholds per country significantly reduced bias towards certain countries and conformed to the relaxed equalized odds measure. The mean FPR and TPR across countries lay within two standard deviations, indicating fair performance.

#### Income Prediction

The heuristic was also applied to an income prediction model to ensure equalized cost across genders. The model was trained to predict whether a person's income is above $50,000, and the heuristic was used to calibrate the decision thresholds to ensure similar false negative rates (FNRs) across genders. The results showed that the proposed heuristic achieved similar FNRs for males and females, with comparable or better performance compared to existing fairness techniques.

#### Criminal Recidivism

The heuristic was applied to a criminal recidivism model to ensure fairness across racial groups. The model was trained to predict the likelihood of reoffending, and the heuristic was used to calibrate the decision thresholds to ensure similar false positive rates (FPRs) and false negative rates (FNRs) across racial groups. The results showed that the proposed heuristic achieved similar FPRs and FNRs for African Americans and Caucasians, with comparable or better performance compared to existing fairness techniques.

### Conclusion

The present invention provides a practical and flexible approach to ensuring fairness in machine learning models, particularly in scenarios involving high-arity attributes and real-time decision-making. The relaxed equalized odds measure and the one-shot fairness heuristic offer significant advantages over existing techniques, including computational efficiency and applicability to a wide range of real-world applications. The invention has been validated through detailed case studies in fraud detection, income prediction, and criminal recidivism, demonstrating its effectiveness and robustness.