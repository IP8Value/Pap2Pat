# DESCRIPTION

## BACKGROUND

In the field of healthcare, the development of predictive models for clinical outcomes is critical for improving patient care and reducing adverse events. These models, often based on machine learning algorithms, aim to estimate the probability of a specific outcome, such as mortality or the occurrence of a stroke, based on a patient's clinical and demographic data. However, the accuracy of these models is not the only factor that determines their clinical utility. In many clinical settings, the consequences of a misclassification can be severe, leading to inappropriate treatment decisions that may harm patients. Therefore, it is essential to have a method that can identify when a model's predictions are likely to be unreliable.

Existing methods for estimating prediction reliability can be broadly categorized into model-dependent and model-independent approaches. Model-dependent methods, such as confidence intervals and uncertainty estimates, are specific to the type of classifier used and often require retraining the model with new data. Model-independent methods, on the other hand, can be applied to various models but often require access to the original training dataset, which is not always feasible due to privacy concerns and computational constraints.

The present invention addresses these limitations by providing a method for identifying unreliable predictions that is model-independent, does not require retraining, and can be applied in the setting of significant class imbalance. This method is particularly useful in clinical settings where the outcome of interest is rare, and the consequences of a misclassification are severe.

## SUMMARY

The present invention provides a method for identifying unreliable predictions in clinical risk models. The method is designed to be model-independent, meaning it can be applied to any clinical risk model regardless of the underlying algorithm. The method does not require retraining the model or access to the original training dataset, making it practical for use in clinical settings where such data may not be readily available.

The method involves the following steps:
1. **Risk Model Prediction**: For a given patient, the clinical risk model \( f(\overset{\rightharpoonup}{x}) \) is used to estimate the probability of an adverse outcome, where \( \overset{\rightharpoonup}{x} \) is a vector of the patient's prognostic features.
2. **Generative Model Calculation**: A separate risk metric \( P^G(y = 1 \mid \overset{\rightharpoonup}{x}) \) is calculated using summary statistics from the same training data used to develop the risk model. This metric is derived from generative models that generate feature vectors consistent with patients in the positive and negative classes.
3. **Unreliability Metric Calculation**: The unreliability metric \( U(\overset{\rightharpoonup}{x}) \) is calculated as the absolute difference between the risk model prediction \( f(\overset{\rightharpoonup}{x}) \) and the generative model prediction \( P^G(y = 1 \mid \overset{\rightharpoonup}{x}) \). Specifically, \( U(\overset{\rightharpoonup}{x}) = |P^G(y = 1 \mid \overset{\rightharpoonup}{x}) - f(\overset{\rightharpoonup}{x})| \).
4. **Thresholding**: Patients with high values of \( U(\overset{\rightharpoonup}{x}) \) are identified as belonging to a subgroup where the risk model is likely to have reduced performance. These patients should be treated with caution, and additional data or alternative risk metrics should be considered to arrive at a more accurate assessment of their risk.

The method is particularly useful in the setting of significant class imbalance, where the positive class (e.g., patients with the adverse outcome) is in the minority. In such cases, the method is effective in identifying patient subgroups where the risk model is likely to perform poorly, thereby helping healthcare providers make more informed decisions.

## DETAILED DESCRIPTION

### 1. Overview of the Method

The method for identifying unreliable predictions in clinical risk models is designed to address the limitations of existing reliability estimation methods. It is model-independent, does not require retraining, and can be applied in the setting of significant class imbalance. The method involves the following key components:

#### 1.1. Risk Model Prediction

The first step in the method is to use a clinical risk model \( f(\overset{\rightharpoonup}{x}) \) to estimate the probability of an adverse outcome for a given patient. The risk model takes a vector of the patient's prognostic features \( \overset{\rightharpoonup}{x} \) as input and outputs a risk score \( f(\overset{\rightharpoonup}{x}) \) that ranges from 0 to 1. This score represents the estimated probability that the patient will experience the adverse outcome.

#### 1.2. Generative Model Calculation

The next step is to calculate a separate risk metric \( P^G(y = 1 \mid \overset{\rightharpoonup}{x}) \) using summary statistics from the same training data used to develop the risk model. This metric is derived from generative models that generate feature vectors consistent with patients in the positive and negative classes. Specifically, the generative models are used to estimate the likelihood of the patient's feature vector \( \overset{\rightharpoonup}{x} \) given the positive class \( P(\overset{\rightharpoonup}{x} \mid y = 1) \) and the negative class \( P(\overset{\rightharpoonup}{x} \mid y = 0) \).

The generative model \( P^G(y = 1 \mid \overset{\rightharpoonup}{x}) \) is calculated using Bayes' rule:
\[ P^G(y = 1 \mid \overset{\rightharpoonup}{x}) = \frac{P(\overset{\rightharpoonup}{x} \mid y = 1) P(y = 1)}{P(\overset{\rightharpoonup}{x})} \]
where \( P(y = 1) \) is the prevalence of the positive class in the dataset, and \( P(\overset{\rightharpoonup}{x}) \) is the marginal probability of the feature vector, which can be calculated as:
\[ P(\overset{\rightharpoonup}{x}) = P(\overset{\rightharpoonup}{x} \mid y = 1) P(y = 1) + P(\overset{\rightharpoonup}{x} \mid y = 0) P(y = 0) \]

In practice, the likelihoods \( P(\overset{\rightharpoonup}{x} \mid y = 1) \) and \( P(\overset{\rightharpoonup}{x} \mid y = 0) \) are estimated using a multivariate normal (MVN) distribution, which provides an efficient and widely used mechanism for likelihood estimation. The mean and covariance of the MVN distributions are estimated using the sample mean and sample covariance of the positive and negative patient populations in the training data.

#### 1.3. Unreliability Metric Calculation

The unreliability metric \( U(\overset{\rightharpoonup}{x}) \) is calculated as the absolute difference between the risk model prediction \( f(\overset{\rightharpoonup}{x}) \) and the generative model prediction \( P^G(y = 1 \mid \overset{\rightharpoonup}{x}) \):
\[ U(\overset{\rightharpoonup}{x}) = |P^G(y = 1 \mid \overset{\rightharpoonup}{x}) - f(\overset{\rightharpoonup}{x})| \]

The higher the value of \( U(\overset{\rightharpoonup}{x}) \), the more unreliable the model prediction is considered to be. Patients with high values of \( U(\overset{\rightharpoonup}{x}) \) are identified as belonging to a subgroup where the risk model is likely to have reduced performance.

#### 1.4. Thresholding

To identify the most unreliable predictions, a threshold is applied to the unreliability metric \( U(\overset{\rightharpoonup}{x}) \). Patients with \( U(\overset{\rightharpoonup}{x}) \) values above a certain threshold (e.g., the 99th percentile) are flagged as having unreliable predictions. These patients should be treated with caution, and additional data or alternative risk metrics should be considered to arrive at a more accurate assessment of their risk.

### 2. Application to Specific Risk Models

#### 2.1. GRACE Risk Model

The Global Registry of Acute Coronary Events (GRACE) risk score is a widely used clinical risk model that quantifies the risk of death 6 months after presenting with an acute coronary syndrome. The GRACE score is based on a set of prognostic features, including age, heart rate, systolic blood pressure, Killip class, serum creatinine, cardiac arrest at presentation, ST-segment elevation, and elevated cardiac markers.

To apply the method to the GRACE risk model, the following steps are taken:
1. **Risk Model Prediction**: The GRACE score is converted to a probability using a published nomogram.
2. **Generative Model Calculation**: Separate MVN distributions are fit to the positive and negative patient populations in the GRACE dataset. The mean and covariance of these distributions are estimated using the sample mean and sample covariance.
3. **Unreliability Metric Calculation**: The unreliability metric \( U(\overset{\rightharpoonup}{x}) \) is calculated for each patient in the GRACE dataset using the formula:
   \[ U(\overset{\rightharpoonup}{x}) = |P^G(y = 1 \mid \overset{\rightharpoonup}{x}) - f(\overset{\rightharpoonup}{x})| \]
4. **Thresholding**: Patients with \( U(\overset{\rightharpoonup}{x}) \) values above the 99th percentile are identified as having unreliable predictions.

#### 2.2. Stroke Risk Model

The Stroke risk model is a logistic regression model trained on the GRACE dataset to predict the risk of in-hospital stroke in patients presenting with an acute coronary syndrome. The model is based on a set of 198 features, including laboratory data, patient demographic information, and medications administered during the first hospital day.

To apply the method to the Stroke risk model, the following steps are taken:
1. **Risk Model Prediction**: The Stroke risk model is used to estimate the probability of in-hospital stroke for each patient.
2. **Generative Model Calculation**: Separate MVN distributions are fit to the positive and negative patient populations in the Stroke dataset. The mean and covariance of these distributions are estimated using the sample mean and sample covariance.
3. **Unreliability Metric Calculation**: The unreliability metric \( U(\overset{\rightharpoonup}{x}) \) is calculated for each patient in the Stroke dataset using the formula:
   \[ U(\overset{\rightharpoonup}{x}) = |P^G(y = 1 \mid \overset{\rightharpoonup}{x}) - f(\overset{\rightharpoonup}{x})| \]
4. **Thresholding**: Patients with \( U(\overset{\rightharpoonup}{x}) \) values above the 99th percentile are identified as having unreliable predictions.

### 3. Evaluation of the Method

#### 3.1. Performance Metrics

The performance of the method is evaluated using several metrics, including calibration curves, normalized Brier scores, and area under the receiver operating characteristic curve (AUC).

- **Calibration Curves**: Calibration curves are used to assess the agreement between the predicted risk and the observed risk. Patients are binned based on their predicted risk, and the fraction of patients who experienced the adverse outcome is calculated for each bin. The method is considered well-calibrated if the predicted risk closely matches the observed risk.
- **Normalized Brier Scores**: The Brier score is a measure of the mean squared error between the predicted risk and the true class label. The normalized Brier score is calculated by dividing the Brier score by the Brier score of a null model that predicts every patient to have a risk equal to the prevalence of the outcome in the population. Lower normalized Brier scores indicate better performance.
- **AUC**: The AUC is a measure of the model's discriminatory ability. Higher AUC values indicate better performance.

#### 3.2. Comparison with Trust Score

The performance of the method is also compared with an alternate metric for quantifying the trustworthiness of a given prediction, known as the trust score. The trust score measures the agreement between the classifier and a nearest-neighbor classifier on a testing example. The trust score is calculated as the ratio between the distance to the alternate class and the distance to the predicted class. Low trust scores denote predictions that are untrustworthy.

### 4. Results

#### 4.1. GRACE Risk Model

- **Calibration Curves**: Calibration curves for the GRACE risk model show that predictions in the upper 50th percentile of unreliability values tend to overestimate the patient's risk of death, while predictions in the lower 50th percentile are well-calibrated.
- **Normalized Brier Scores**: The normalized Brier score for predictions in the upper 50th percentile of unreliability values is significantly higher than the normalized Brier score for predictions in the lower 50th percentile, indicating reduced accuracy.
- **AUC**: The AUC for predictions in the upper 50th percentile of unreliability values is significantly lower than the AUC for predictions in the lower 50th percentile, indicating reduced discriminatory ability.

#### 4.2. Stroke Risk Model

- **Calibration Curves**: Calibration curves for the Stroke risk model show that predictions in the upper 50th percentile of unreliability values tend to underestimate the patient's risk of stroke, while predictions in the lower 50th percentile are well-calibrated.
- **Normalized Brier Scores**: The normalized Brier score for predictions in the upper 50th percentile of unreliability values is significantly higher than the normalized Brier score for predictions in the lower 50th percentile, indicating reduced accuracy.
- **AUC**: The AUC for predictions in the upper 50th percentile of unreliability values is significantly lower than the AUC for predictions in the lower 50th percentile, indicating reduced discriminatory ability.

### 5. Conclusion

The present invention provides a method for identifying unreliable predictions in clinical risk models that is model-independent, does not require retraining, and can be applied in the setting of significant class imbalance. The method is effective in identifying patient subgroups where the risk model is likely to perform poorly, thereby helping healthcare providers make more informed decisions. The method has been successfully applied to the GRACE risk model and the Stroke risk model, demonstrating its utility in clinical settings. Future work will focus on applying the method to additional risk models and datasets to further validate its performance.