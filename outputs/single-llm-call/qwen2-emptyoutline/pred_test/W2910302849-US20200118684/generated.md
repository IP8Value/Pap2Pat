# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a method and system for predicting kidney transplant survival using an ensemble of machine learning models. Specifically, the invention involves clustering transplant recipients into different cohorts based on recipient age and applying different predictive models to each cohort to achieve improved accuracy in estimating post-transplant survival.

## BACKGROUND

Kidney transplantation is a critical medical procedure that significantly improves the quality of life for patients with end-stage renal disease. However, the success of a kidney transplant depends on various factors, including the compatibility of the donor and recipient, the recipient's overall health, and the post-transplant care. Accurate prediction of post-transplant survival is essential for optimizing the allocation of donated kidneys and improving patient outcomes.

Current methods for predicting kidney transplant survival, such as the Estimated Post Transplant Survival (EPTS) score, the Recipient Risk Score (RSS), and the Life Years from Transplant (LYFT) model, primarily rely on the Cox proportional hazards model. While these models have been widely used, they often fail to capture the complex interactions and heterogeneity in the data, leading to suboptimal predictions.

Recent advancements in machine learning have shown promise in improving the accuracy of survival predictions. Techniques such as random survival forests and Cox proportional hazards models with regularization have been explored. However, these methods are typically applied uniformly across all recipients, which may not account for the differences in survival patterns across different patient cohorts.

The present invention addresses these limitations by employing an ensemble of machine learning models tailored to specific cohorts of transplant recipients. By clustering recipients based on recipient age and applying different models to each cohort, the invention aims to provide more accurate and reliable predictions of post-transplant survival.

## SUMMARY

The present invention provides a method and system for predicting kidney transplant survival using an ensemble of machine learning models. The method involves the following steps:

1. **Data Collection**: Collecting a dataset of kidney transplant recipients, including both living and deceased donors, pediatric and adult recipients, and censored observations. The dataset should include a comprehensive set of variables related to the recipient, donor, and transplant procedure.

2. **Data Preparation**: Preparing the dataset by handling missing data, grouping categorical variables, and selecting relevant variables. This step ensures that the data is clean and ready for analysis.

3. **Cohort Clustering**: Clustering the recipients into different cohorts based on recipient age. The invention specifically divides the recipients into two cohorts: those aged 50 and under (cohort 1) and those aged 51 and older (cohort 2).

4. **Variable Selection**: Using permutation importance and Lasso regularization to select the most important variables for each cohort. This step helps in identifying the variables that have the greatest impact on post-transplant survival.

5. **Model Building**: Constructing different predictive models for each cohort. For cohort 1, a random survival forest model with conditional inference trees is used. For cohort 2, a Cox proportional hazards model is employed. The choice of model is based on the performance metrics obtained during cross-validation.

6. **Model Evaluation**: Evaluating the performance of the proposed model using metrics such as Harrell’s concordance index and the integrated Brier score. The model is validated using cross-validation and compared to existing models like the EPTS model.

The invention also provides a system for implementing the method, including a data processing module, a cohort clustering module, a variable selection module, a model building module, and a model evaluation module. The system is designed to handle large datasets and provide real-time predictions of post-transplant survival.

## DETAILED DESCRIPTION

### Data Collection

The dataset for the invention is obtained from the United Network for Organ Sharing (UNOS) and includes records of kidney transplant recipients from 1987 to 2014. The dataset encompasses both living and deceased donors, pediatric and adult recipients, and censored observations. Each observation in the dataset includes a wide range of variables, such as recipient age, gender, medical history, donor characteristics, and post-transplant outcomes.

### Data Preparation

#### Handling Missing Data

To address missing data, two approaches are employed:
1. **Imputation by Predictive Mean Matching (PMM)**: This method is used to impute missing values for non-categorical variables.
2. **Removing Missing Data**: For non-categorical variables, observations with missing data are removed. For categorical variables, missing data are labeled as 'unknown'.

#### Grouping Categorical Variables

Some categorical variables in the dataset have a large number of possible values, which can lead to overfitting and increased model variance. To mitigate this, the invention groups different values of categorical variables together based on their effect on the hazard function, controlling for relevant variables. For example, the variable "kidney diagnosis" is reduced from 75 different values to 8.

### Cohort Clustering

The invention clusters the recipients into two cohorts based on recipient age:
1. **Cohort 1**: Recipients aged 50 and under.
2. **Cohort 2**: Recipients aged 51 and older.

This clustering is based on the average split value obtained from 100 survival decision trees, which is rounded to 50 years. The average 5-year survival probabilities for cohorts 1 and 2 are 93% and 80%, respectively, based on the Kaplan-Meier estimate.

### Variable Selection

#### Permutation Importance

The invention uses the Breiman-Cutler permutation importance measure for random survival forests to rank the variables in order of importance. Recipient age is consistently ranked as the most important variable, justifying the cohort clustering approach.

#### Lasso Regularization

To determine the number of top variables to select, the invention applies a Cox model regularized with the Lasso (L1) penalty. The optimal Lasso penalty is determined using 10-fold cross-validation. The number of nonzero coefficients for different penalty values is evaluated, and the Lasso penalty with the fewest nonzero coefficients within one standard deviation of the minimum Partial Likelihood Deviance (PLD) is selected.

### Model Building

#### Cohort 1: Random Survival Forests

For cohort 1, a random survival forest model with conditional inference trees is constructed. The forest consists of 800 trees, with four randomly selected variables considered for each split. The tree split is restricted to occur only if the splitting test statistic exceeds 0.3, allowing the use of smaller trees with the same predictive performance.

#### Cohort 2: Cox Proportional Hazards Model

For cohort 2, a Cox proportional hazards model is used. This model is chosen because it achieves a better concordance index than the random survival forest model for this cohort. The model is fitted using the selected variables and cross-validated to ensure robustness.

### Model Evaluation

The performance of the proposed model is evaluated using two primary metrics:
1. **Harrell’s Concordance Index (C-index)**: This measures the percentage of patient pairs correctly ranked by the model based on their post-transplant survival duration.
2. **Integrated Brier Score**: This measures the accuracy of the model's survival predictions over a given time horizon.

The proposed model is validated using cross-validation and compared to existing models, such as the EPTS model. The results show that the proposed model outperforms the EPTS model and other recent models in the literature, achieving a higher C-index and lower integrated Brier score.

### System Implementation

The invention also provides a system for implementing the method, comprising the following modules:
1. **Data Processing Module**: Handles data collection, cleaning, and preparation.
2. **Cohort Clustering Module**: Clusters recipients into different cohorts based on recipient age.
3. **Variable Selection Module**: Selects the most important variables for each cohort using permutation importance and Lasso regularization.
4. **Model Building Module**: Constructs the random survival forest model for cohort 1 and the Cox proportional hazards model for cohort 2.
5. **Model Evaluation Module**: Evaluates the performance of the proposed model using cross-validation and compares it to existing models.

The system is designed to handle large datasets and provide real-time predictions of post-transplant survival, making it a valuable tool for optimizing kidney allocation and improving patient outcomes.

By leveraging an ensemble of machine learning models tailored to specific cohorts, the invention offers a more accurate and reliable method for predicting kidney transplant survival, ultimately contributing to better patient care and resource allocation in the field of organ transplantation.