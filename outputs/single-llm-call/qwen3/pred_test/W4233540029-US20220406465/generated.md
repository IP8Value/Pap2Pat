# DESCRIPTION

## TECHNICAL FIELD

- define technical field

The present invention relates to computer-implemented systems and methods for the passive, non-intrusive detection and prediction of mental health risk in individuals living with diabetes mellitus through the analysis of behavioral, physiological, and interaction-based data streams generated during routine diabetes self-management. Specifically, the invention encompasses a machine learning framework that leverages data collected from diabetes management devices—particularly blood glucose meters—and associated digital platforms, including mobile applications, web portals, and remote coaching services, to identify patterns indicative of emerging or existing mental health conditions such as depression, anxiety, and emotional dysregulation. The system operates without requiring additional patient input beyond normal engagement with existing diabetes care tools, thereby enabling scalable, real-time mental health risk assessment within the context of standard clinical workflows. This approach represents a novel integration of digital health technologies, behavioral analytics, and predictive modeling to address a critical unmet need in chronic disease management: the early identification of psychological comorbidities that significantly impair treatment adherence and clinical outcomes.

## BACKGROUND

- introduce diabetes statistics

Diabetes mellitus affects approximately 34.2 million individuals in the United States alone, representing nearly one in ten members of the population. The prevalence of this chronic condition continues to rise, placing substantial strain on healthcare infrastructure and economic resources. Beyond its physiological consequences, diabetes is strongly associated with a heightened incidence of psychological disorders, including major depressive disorder and generalized anxiety disorder, which collectively affect between 25% and 40% of affected individuals. These mental health comorbidities are not incidental but are deeply interwoven with the daily burdens of disease management, including frequent glucose monitoring, dietary restrictions, medication regimens, and lifestyle modifications.

- describe mental health needs in diabetes patients

Individuals with diabetes who experience concurrent mental health challenges demonstrate significantly reduced adherence to recommended self-care behaviors, including glucose testing, insulin administration, nutritional planning, physical activity, and scheduled medical appointments. This non-adherence directly correlates with poorer glycemic control, increased rates of diabetic complications such as neuropathy, retinopathy, and nephropathy, and higher hospitalization frequencies. Moreover, the psychological distress associated with chronic illness often exacerbates feelings of helplessness, isolation, and emotional fatigue, creating a self-reinforcing cycle that further deteriorates both mental and physical health outcomes.

- discuss limitations of current mental health detection methods

Despite clear clinical guidelines from organizations such as the American Diabetes Association and the United States Preventive Services Task Force recommending routine mental health screening for all individuals with diabetes, fewer than half of those exhibiting symptoms receive a formal diagnosis or intervention. This gap arises from multiple systemic barriers: a shortage of mental health professionals trained in diabetes care, limited time and resources available to primary care providers, inadequate integration of mental health screening tools into routine clinical workflows, and persistent stigma surrounding psychological conditions. Traditional screening methods, such as paper-based questionnaires or brief clinical interviews, are often administered infrequently, rely on self-reporting that may be biased by denial or social desirability, and fail to capture dynamic changes in emotional state over time.

- motivate need for improvement

There is a pressing and unmet need for a scalable, objective, and passive method of mental health risk detection that operates seamlessly within the existing routines of diabetes self-management. Such a system must avoid adding burden to patients or clinicians, provide continuous monitoring rather than intermittent snapshots, and deliver actionable insights in real time. The integration of passive sensing technologies with digital health platforms offers a transformative opportunity to bridge this gap by transforming routine device interactions into meaningful indicators of psychological well-being.

## SUMMARY

- introduce need for mental health risk detection improvement

The present invention addresses the critical need for an automated, non-intrusive, and clinically actionable system capable of detecting early signs of mental health risk in individuals with diabetes through the analysis of passive behavioral and physiological data collected during routine diabetes management activities. Unlike conventional screening tools that require active participation or clinical intervention, this system operates continuously and autonomously, leveraging data already generated as part of standard care.

- describe passive sensing and ecological momentary assessment

The invention employs principles of passive sensing and ecological momentary assessment to capture real-time, context-rich behavioral patterns without requiring additional effort from the individual. Passive sensing involves the unobtrusive collection of data through devices already in regular use, while ecological momentary assessment captures transient emotional and behavioral states as they occur within the natural environment of the individual, thereby minimizing recall bias and enhancing ecological validity.

- motivate use of glucometer data for mental health detection

Crucially, the invention uniquely utilizes blood glucose meter data as a primary source of mental health risk signals. Individuals with diabetes are clinically encouraged to test their glucose levels multiple times daily, creating a high-frequency, longitudinal dataset that includes not only physiological readings but also metadata such as time of day, day of week, frequency of testing, and self-reported emotional states at the time of measurement. These data points, previously overlooked in mental health contexts, have been shown to correlate strongly with psychological well-being, including patterns of avoidance, emotional reactivity, and behavioral inertia.

- describe computer-implemented method for mental health risk prediction

The invention comprises a computer-implemented method for predicting mental health risk in individuals with diabetes by analyzing aggregated input signals derived from multiple data sources, including glucometer usage, coaching interactions, demographic characteristics, and digital platform engagement. These signals are processed through a trained machine learning model that identifies complex, non-linear relationships between behavioral patterns and mental health status, generating a continuous risk score that reflects the likelihood of an emerging or existing mental health condition.

- define glucometer data inputs

Glucometer data inputs include the number of glucose measurements per day, temporal distribution of measurements across hours and days of the week, mean and standard deviation of glucose values, variability in glucose trends over time, and self-reported emotional states (e.g., “well,” “stressed,” “unwell”) recorded at the time of each measurement.

- define demographic data inputs

Demographic data inputs include age, gender, race, ethnicity, body mass index, smoking status, and socioeconomic indicators derived from enrollment or insurance records, each of which has been empirically associated with differential risk profiles for mental health conditions in diabetic populations.

- define mental health status data inputs

Mental health status data inputs are derived from validated ground truth sources, including pharmacy claims for antidepressants, anxiolytics, or antipsychotics; mental health-related diagnostic codes from medical claims; documented mental health interventions such as counseling sessions or referrals; and records of medication refills indicating treatment continuity or discontinuation.

- describe machine learning system training

The machine learning system is trained using supervised learning techniques on a large, retrospective dataset of participant-period instances, each representing a four-week window of behavioral data aggregated from individual users. The training process employs gradient boosting algorithms optimized for class imbalance, with ensemble modeling used to enhance generalizability and reduce overfitting.

- describe machine learning system validation

Validation is performed across multiple held-out test sets, each defined by distinct sources of ground truth—medication refill data, claims data, and combined indicators—to ensure robustness across diverse data environments and population subgroups. Performance is evaluated using metrics including area under the receiver operating characteristic curve, precision, recall, F1 score, and SHAP-based feature importance analysis.

- describe updating machine learning model

The machine learning model is designed for continuous learning, with periodic retraining using newly collected data and feedback loops that incorporate clinician annotations, patient-reported outcomes, and intervention outcomes to refine predictive accuracy over time.

- introduce coaching data inputs

Coaching data inputs include frequency and duration of interactions with diabetes coaches, success and failure rates of outreach attempts, response latency to messages, and the emotional tone of verbal or written exchanges, serving as proxies for social engagement and behavioral activation.

- describe event data inputs

Event data inputs encompass usage patterns of the associated mobile application and web portal, including session frequency, duration, time of day of access, interaction with educational content, response to reminders, and voluntary sharing of health data with family or caregivers.

- describe system for mental health risk prediction

The system comprises a distributed architecture integrating data ingestion modules, preprocessing pipelines, feature extraction engines, machine learning inference engines, and output interfaces, all operating in secure compliance with healthcare privacy regulations.

- define input device functionality

Input devices include cellular-enabled blood glucose meters, mobile applications, web portals, and coaching platforms, each configured to transmit anonymized, encrypted behavioral and physiological data to a centralized data repository.

- define machine learning system functionality

The machine learning system ingests, normalizes, and aggregates input signals into participant-period instances, applies feature engineering to extract temporal, statistical, and contextual patterns, and generates a continuous mental health risk probability score using an ensemble of gradient-boosted decision trees.

- define output device functionality

Output devices include clinician dashboards, automated alerts to care teams, and integrated care coordination platforms, which present risk scores alongside interpretive insights and recommended intervention pathways.

- describe glucometer data inputs

Glucometer data inputs are processed to identify deviations from baseline usage patterns, including prolonged periods of inactivity, clustering of measurements during non-typical hours, increased variability in glucose readings unrelated to dietary or pharmacological changes, and consistent self-reports of emotional distress during testing.

- describe demographic data inputs

Demographic data inputs are normalized and encoded to reflect known risk factors, such as female gender, younger age, Black or Hispanic ethnicity, lower socioeconomic status, and elevated BMI, each contributing weighted parameters to the predictive model.

- describe mental health status data inputs

Mental health status data inputs are labeled using a multi-source consensus approach, where an individual is classified as a case if they have at least one prescription for a psychiatric medication, one mental health diagnosis code, or one documented intervention within the observation window.

- describe system training and validation

System training involves partitioning data into time-based cohorts to prevent temporal leakage, applying random undersampling to balance class distributions, and training an ensemble of 10 LightGBM models with hyperparameters optimized via Bayesian search. Validation is conducted on three independent test sets, each representing different data availability profiles, to ensure generalizability across heterogeneous real-world settings.

## DETAILED DESCRIPTION

- introduce system for mental health risk detection in diabetes patients

The system for mental health risk detection in diabetes patients is a fully integrated, cloud-based platform designed to operate within existing diabetes care ecosystems without requiring additional hardware, software installation, or behavioral modification from the patient. It functions as a passive monitoring layer embedded within digital diabetes management programs, continuously analyzing behavioral patterns to detect deviations indicative of psychological distress.

- describe machine learning system architecture

The machine learning system architecture consists of four primary components: a data ingestion layer that receives encrypted telemetry from glucometers, mobile apps, web portals, and coaching systems; a preprocessing engine that cleans, timestamps, and normalizes raw signals; a feature extraction module that computes temporal, statistical, and contextual features from aggregated participant-period data; and an inference engine that applies an ensemble of gradient-boosted decision trees to generate a continuous mental health risk probability score.

- discuss glucometer data collection

Glucometer data collection occurs automatically during each glucose measurement, with the device recording the time, date, glucose value, and optional self-reported emotional state. These data are transmitted via cellular or Bluetooth connectivity to a secure server, where they are aggregated into daily and weekly behavioral profiles.

- explain coaching data collection

Coaching data collection captures every interaction between the patient and a certified diabetes coach, including initiated calls, responded messages, duration of conversations, response latency, and sentiment analysis of text-based exchanges. These interactions serve as behavioral proxies for social connectedness and emotional engagement.

- describe demographics data collection

Demographics data are collected during enrollment and updated periodically through patient-reported updates or insurance records, including age, gender, race, ethnicity, geographic location, income bracket, and smoking status.

- discuss event data collection

Event data collection tracks all digital interactions with the diabetes management platform, including app logins, web portal visits, response to educational prompts, use of goal-setting tools, and voluntary sharing of health data with family members or caregivers.

- illustrate system components

The system comprises a network of secure data endpoints, a centralized data lake, preprocessing pipelines, feature stores, model servers, and output interfaces, all interconnected through encrypted APIs and operating under HIPAA-compliant protocols.

- describe machine learning model training

Model training is performed on a retrospective dataset of over 124,000 participant-period instances, each representing a four-week window of behavioral data. Labels are derived from pharmacy claims, diagnostic codes, and intervention records. Training employs LightGBM gradient boosting with random undersampling to address class imbalance, and model ensembles are created via bootstrapped subsampling to enhance robustness.

- discuss supervised machine learning approach

The supervised machine learning approach relies on labeled training data to establish mappings between input features and mental health status. The model learns to recognize patterns in behavioral sequences that precede or co-occur with documented mental health diagnoses or medication fills.

- introduce deep learning neural network

While deep learning neural networks were evaluated during model selection, they did not outperform gradient boosting methods in terms of interpretability, computational efficiency, or performance on imbalanced datasets. The final system therefore employs ensemble tree-based models for superior generalization and clinical transparency.

- introduce gradient boosting algorithm

The gradient boosting algorithm employed is LightGBM, chosen for its high accuracy, low memory footprint, and efficient handling of categorical variables. It iteratively corrects prediction errors by training subsequent models on residual errors of prior models, resulting in a highly discriminative ensemble.

- describe training set creation

Training set creation involves defining participant-period instances with a fixed four-week aggregation window, filtering out instances with more than 30 consecutive days of inactivity, and ensuring temporal separation between training and test sets to prevent data leakage.

- discuss MH status data labelling

Mental health status labeling is performed using a multi-source consensus protocol: an individual is classified as a case if they have at least one of the following within the observation window: a filled prescription for a psychiatric medication, a mental health diagnosis code in claims data, or documented participation in a mental health intervention.

- describe MH medication data collection

Mental health medication data are collected from pharmacy claims databases, capturing the type, dosage, and refill history of antidepressants, anxiolytics, mood stabilizers, and antipsychotics.

- describe MH assessment data collection

Mental health assessment data are derived from structured clinical documentation, including completed PHQ-9 or GAD-7 scales, clinician notes indicating mood disorders, and referrals to behavioral health specialists.

- describe MH claims data collection

Mental health claims data are obtained from insurance claims systems, including ICD-10 codes for depression, anxiety, bipolar disorder, and related conditions, as well as CPT codes for psychotherapy and psychiatric consultations.

- describe MH intervention data collection

Mental health intervention data include records of scheduled counseling sessions, telehealth visits, crisis interventions, and referrals to mental health providers initiated through the diabetes program.

- illustrate MH risk input signals in demographics data

Demographic signals associated with mental health risk include female gender, younger age, Black or Hispanic ethnicity, lower socioeconomic status, and active smoking, each contributing a weighted feature vector to the model’s predictive logic.

- illustrate MH risk input signals in glucometer data

Glucometer-based risk signals include decreased testing frequency, increased variability in glucose readings, testing predominantly during late-night or early-morning hours, and consistent self-reports of “unwell” emotional states during measurement events.

- describe glucometer usage patterns

Glucometer usage patterns are analyzed for temporal regularity, including the proportion of tests performed on weekdays versus weekends, the consistency of testing intervals, and the presence of prolonged gaps exceeding seven days without any measurement.

- illustrate MH risk input signals in coaching data

Coaching data reveal risk signals such as reduced frequency of coach-initiated contact, prolonged response times to messages, shorter call durations, and repeated unsuccessful outreach attempts, indicating withdrawal or disengagement.

- describe coaching data analysis

Coaching data are analyzed for interaction frequency, duration, sentiment polarity, and response patterns, with machine learning models identifying deviations from baseline engagement profiles that correlate with declining psychological well-being.

- illustrate MH risk input signals in event data

Event data show risk signals including reduced app usage, avoidance of goal-setting features, infrequent interaction with educational content, and decreased sharing of health data with support networks.

- describe event data analysis

Event data are analyzed for session frequency, duration, time-of-day distribution, and interaction depth, with anomalies such as sudden drops in engagement or prolonged inactivity flagged as potential indicators of depressive withdrawal.

- describe pre-processing of input data

Input data are pre-processed through normalization, outlier removal, temporal alignment, and missing value imputation using forward-fill and interpolation techniques. All data are aggregated into participant-period instances with consistent time windows to ensure model compatibility.

- illustrate system for mental health risk assessment

The system for mental health risk assessment operates as a closed-loop platform that continuously ingests behavioral data, computes risk scores in real time, surfaces alerts to care teams, and incorporates feedback from clinical interventions to refine future predictions.

- describe interaction with diabetes management program

The system is fully integrated into existing diabetes management programs, operating as a passive overlay that enhances—not replaces—current care delivery models. It triggers automated alerts to care coordinators when risk scores exceed predefined thresholds, prompting timely intervention.

- describe data collection from various devices

Data are collected from a network of FDA-cleared devices, including cellular-enabled glucometers, mobile applications, web portals, and coaching platforms, all transmitting encrypted telemetry to a centralized data repository in compliance with HIPAA and GDPR standards.

- describe MH risk input signals analysis

Mental health risk input signals are analyzed using feature engineering techniques that extract statistical moments, temporal autocorrelations, entropy measures, and behavioral rhythms, transforming raw telemetry into discriminative predictors of psychological distress.

- describe MH risk score calculation

The mental health risk score is calculated as a probability value between 0 and 1, derived from the averaged predictions of 10 ensemble LightGBM models, each trained on a randomized subset of the training data. The score reflects the likelihood that the individual is experiencing or will soon experience a clinically significant mental health condition.

- describe MH risk score output

The mental health risk score is output to clinician-facing dashboards, care coordination platforms, and automated alert systems, accompanied by interpretive insights derived from SHAP values that highlight the most influential behavioral signals contributing to the prediction.

- describe feedback process for model refinement

A feedback loop is established wherein clinicians can annotate predicted risk cases as true positives, false positives, or false negatives. These annotations are incorporated into subsequent training cycles to recalibrate model weights and improve predictive accuracy over time.

- describe model update using gradient descent algorithm

Model updates are performed using gradient descent-based optimization, with parameters adjusted iteratively to minimize cross-entropy loss between predicted risk scores and ground truth labels derived from newly labeled data.

- illustrate process for detecting MH risk in diabetes patients

The process begins with the passive collection of behavioral and physiological data during routine diabetes self-management, followed by aggregation into participant-period instances, feature extraction, model inference, risk score generation, and clinical alerting—all occurring without patient awareness or active participation.

- describe collection of MH risk input signals and MH status data

Mental health risk input signals are collected continuously and automatically from glucometers, coaching systems, and digital platforms, while mental health status data are retrospectively obtained from claims, medication records, and clinical documentation to serve as training labels.

- describe training set creation

Training sets are created by aligning behavioral data windows with corresponding mental health status labels, ensuring temporal consistency and excluding incomplete or inactive periods to maintain data integrity.

- describe model validation

Model validation is performed across three independent test sets, each defined by different sources of ground truth, to evaluate generalizability across diverse populations, data availability profiles, and healthcare settings.

- describe model deployment

Model deployment occurs in a secure, scalable cloud environment with real-time inference capabilities, integrated into existing diabetes management platforms via API endpoints that deliver risk scores to care teams without disrupting workflow.

- describe MH risk prediction score calculation

The mental health risk prediction score is calculated as the soft-voted average of 10 independently trained LightGBM models, each contributing a probability estimate that is aggregated into a single, interpretable risk value between 0 and 1.

- describe computer system architecture for implementing the system

The computer system architecture comprises distributed data ingestion nodes, a secure data lake, feature engineering pipelines, model inference servers, and output interfaces, all operating within a containerized microservices framework on a HIPAA-compliant cloud infrastructure, ensuring scalability, redundancy, and regulatory compliance.