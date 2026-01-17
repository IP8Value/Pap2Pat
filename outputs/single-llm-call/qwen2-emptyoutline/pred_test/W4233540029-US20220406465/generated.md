# DESCRIPTION

## TECHNICAL FIELD

The present invention relates generally to the field of healthcare and, more specifically, to a system and method for predicting mental health risk in individuals with diabetes using passive sensing and machine learning. The system leverages data from blood glucose meters, mobile applications, and coaching interactions to identify individuals at risk of mental health issues, such as depression and anxiety, thereby facilitating early intervention and improved health outcomes.

## BACKGROUND

Diabetes mellitus is a chronic condition affecting millions of people worldwide. In the United States alone, approximately 34.2 million individuals are diagnosed with diabetes. A significant portion of these individuals, around 25%, also experience significant depressive symptoms, and up to 40% suffer from generalized anxiety disorder (GAD). The co-occurrence of diabetes and mental health issues poses substantial challenges, including reduced adherence to diabetes treatment recommendations, increased risk of adverse medical outcomes, and higher healthcare costs. Despite the recognized importance of routine mental health evaluations for individuals with diabetes, only a fraction of those in need receive appropriate diagnosis and intervention due to a shortage of mental health professionals, limited mental health knowledge among primary care providers, and a lack of accessible screening tools.

Traditional methods of mental health assessment, such as clinical interviews and self-report questionnaires, are resource-intensive and may not provide real-time insights. Recent advancements in technology, particularly in passive sensing and ecological momentary assessments (EMA), offer promising alternatives. Passive sensing involves the automatic collection of data about an individual without requiring additional effort, while EMA involves repeated sampling of an individual's behavior in real-time within their natural environment. These methods can be integrated with devices and services that individuals with diabetes already use, such as blood glucose meters and mobile health applications, to enable continuous monitoring and timely interventions.

However, existing research on passive sensing and EMA has primarily focused on the general population and has not extensively explored their application in the diabetes population. Moreover, no known studies have utilized blood glucose meters for detecting mental health concerns, despite the frequent interaction of individuals with diabetes with these devices and the known correlations between blood glucose monitoring and psychological effects. The present invention addresses this gap by developing a machine learning model that uses passive sensing signals from blood glucose meters, mobile applications, and coaching interactions to predict mental health risk in individuals with diabetes.

## SUMMARY

The present invention provides a system and method for predicting mental health risk in individuals with diabetes using passive sensing and machine learning. The system collects and processes data from various sources, including blood glucose meters, mobile applications, and coaching interactions, to identify individuals at risk of mental health issues such as depression and anxiety. The method involves the following steps:

1. **Data Collection**: Collecting passive sensing signals from blood glucose meters, mobile applications, and coaching interactions. These signals include demographic data, blood glucose levels, interaction frequencies, and emotional states reported during blood glucose checks.

2. **Data Aggregation**: Aggregating the collected signals over a defined period, such as four weeks, to create participant-period instances. This step ensures that the model has a comprehensive view of the individual's behavior and emotional state.

3. **Feature Engineering**: Extracting relevant features from the aggregated data, including demographic factors, blood glucose metrics, coaching interaction patterns, and event data from mobile applications and web portals.

4. **Model Training**: Training a machine learning model, specifically a LightGBM gradient tree boosting model, on a training dataset. The model is trained to recognize patterns in the passive sensing signals that are indicative of mental health risk.

5. **Model Evaluation**: Evaluating the trained model on separate test datasets to assess its performance in predicting mental health risk. The evaluation includes metrics such as accuracy, precision, recall, and area under the receiver operating characteristic curve (AUC).

6. **Deployment and Monitoring**: Deploying the trained model in a real-world setting to continuously monitor individuals with diabetes and identify those at risk of mental health issues. The system can trigger alerts for healthcare providers to initiate early interventions.

The invention offers several advantages over existing methods of mental health assessment. By leveraging passive sensing and machine learning, the system can provide real-time insights and facilitate early intervention, leading to improved health outcomes and reduced healthcare costs. Additionally, the system is designed to be non-intrusive and can be integrated with existing diabetes management programs, making it a practical solution for widespread adoption.

## DETAILED DESCRIPTION

### System Overview

The system for predicting mental health risk in individuals with diabetes comprises several components working in concert to collect, process, and analyze data. The key components of the system are:

1. **Data Collection Module**: This module collects passive sensing signals from various sources, including:
   - **Blood Glucose Meters**: Collects data on blood glucose levels, the frequency of blood glucose checks, and the emotional state of the individual during these checks.
   - **Mobile Applications and Web Portals**: Tracks the frequency, duration, and consistency of interactions with the diabetes management program, including the use of features such as food tracking, physical activity tracking, and health reminders.
   - **Coaching Interactions**: Records the frequency and duration of interactions with diabetes coaches, as well as the success or failure of these interactions.

2. **Data Aggregation Module**: This module aggregates the collected signals over a defined period, such as four weeks, to create participant-period instances. The aggregation process ensures that the model has a comprehensive view of the individual's behavior and emotional state.

3. **Feature Engineering Module**: This module extracts relevant features from the aggregated data. The features include:
   - **Demographic Factors**: Age, gender, ethnicity, and race.
   - **Blood Glucose Metrics**: Number of blood glucose checks, mean and standard deviation of blood glucose levels, and variations in blood glucose values.
   - **Coaching Interaction Patterns**: Frequency and duration of coaching interactions, success or failure of contacts, and time spent interacting.
   - **Event Data**: Frequency, duration, and consistency of interactions with the mobile application and web portal, including the time of day and day of week information.

4. **Machine Learning Model**: The system employs a LightGBM gradient tree boosting model to predict mental health risk. The model is trained on a training dataset and evaluated on separate test datasets to ensure its performance and generalizability.

5. **Deployment and Monitoring Module**: Once trained, the model is deployed in a real-world setting to continuously monitor individuals with diabetes. The system can trigger alerts for healthcare providers to initiate early interventions for individuals at risk of mental health issues.

### Methodology

#### Data Collection

The data collection module gathers passive sensing signals from the following sources:

- **Blood Glucose Meters**: The blood glucose meter is the most frequent interaction point for individuals with diabetes. The system collects data on the number of times blood glucose is checked, the blood glucose levels, and the emotional state of the individual during these checks. Emotional state data is obtained through prompts asking the individual to report their current emotional state (e.g., "How are you feeling right now?").

- **Mobile Applications and Web Portals**: The mobile application and web portal are integral parts of the diabetes management program. The system tracks the frequency, duration, and consistency of interactions with these platforms, including the use of features such as food tracking, physical activity tracking, and health reminders. Additional data points include the time of day and day of week information associated with these interactions.

- **Coaching Interactions**: Diabetes coaches play a crucial role in supporting individuals with diabetes. The system records the frequency and duration of coaching interactions, as well as the success or failure of these contacts. The time spent interacting with coaches is also captured.

#### Data Aggregation

The data aggregation module combines the collected signals over a defined period, typically four weeks, to create participant-period instances. This aggregation process ensures that the model has a comprehensive view of the individual's behavior and emotional state. The module filters out ineligible participant-period instances, such as those occurring before the individual joined the diabetes management program or those with extended inactivity (30 or more days without any interaction with the program).

#### Feature Engineering

The feature engineering module extracts relevant features from the aggregated data. The features are categorized as follows:

- **Demographic Factors**: Age, gender, ethnicity, and race are included as demographic factors. These factors have been shown to be related to mental health and are important for the model to consider.

- **Blood Glucose Metrics**: The number of blood glucose checks, mean and standard deviation of blood glucose levels, and variations in blood glucose values are extracted as blood glucose metrics. These metrics provide insights into the individual's blood glucose control and its potential impact on mental health.

- **Coaching Interaction Patterns**: The frequency and duration of coaching interactions, success or failure of contacts, and time spent interacting are extracted as coaching interaction patterns. These patterns serve as a proxy for sociability and can indicate the individual's level of engagement with the diabetes management program.

- **Event Data**: The frequency, duration, and consistency of interactions with the mobile application and web portal, including the time of day and day of week information, are extracted as event data. This data provides a broader view of the individual's behavior and engagement with the program.

#### Model Training

The machine learning model is trained on a training dataset to recognize patterns in the passive sensing signals that are indicative of mental health risk. The training dataset is divided into two segments: the training and validation set, and the test sets. The training and validation set consists of 87% of the study population, while the test sets are held separate from model training to evaluate the model's ability to generalize to unseen data.

The core component of the model is the LightGBM gradient tree boosting algorithm. The model is trained on random subsets of the training data, with undersampling of the control instances to address class imbalance. This approach allows the model to fully utilize the entire training dataset and train on multiple perspectives of the data. Soft voting is used to obtain an output prediction for a given instance, where the outputs of each constituent model are averaged to obtain a single aggregate confidence score.

#### Model Evaluation

The trained model is evaluated on separate test datasets to assess its performance in predicting mental health risk. The evaluation includes the following metrics:

- **Accuracy**: The proportion of correct predictions out of all predictions.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positive cases.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
- **Area Under the Receiver Operating Characteristic Curve (AUC)**: A measure of the model's ability to distinguish between positive and negative cases.

The model's performance is evaluated on three distinct test sets, each with a different source of ground truth. The results demonstrate that the model achieves a respectable, generalizable performance, with an AUC of nearly 0.7 on the first test set and exceeding 0.65 across all three sets. The model's precision and recall metrics also show significant improvement over random prediction, indicating its effectiveness in identifying individuals at risk of mental health issues.

#### Deployment and Monitoring

Once trained and evaluated, the model is deployed in a real-world setting to continuously monitor individuals with diabetes. The system triggers alerts for healthcare providers to initiate early interventions for individuals at risk of mental health issues. The deployment and monitoring module ensures that the model remains up-to-date and continues to perform effectively in a dynamic environment.

### Conclusion

The present invention provides a robust and scalable system for predicting mental health risk in individuals with diabetes using passive sensing and machine learning. By leveraging data from blood glucose meters, mobile applications, and coaching interactions, the system can identify individuals at risk of mental health issues, facilitating early intervention and improved health outcomes. The system's non-intrusive nature and integration with existing diabetes management programs make it a practical solution for widespread adoption. Future work will focus on expanding the scope of the system to include additional data sources and further refining the machine learning model to enhance its predictive capabilities.