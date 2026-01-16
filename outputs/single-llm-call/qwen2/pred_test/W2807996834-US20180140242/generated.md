# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a method and system for predicting and detecting impending relapse in schizophrenia patients using specific Positive and Negative Syndrome Scale (PANSS) items. More specifically, the invention involves identifying and monitoring a subset of PANSS items that exhibit significant changes prior to relapse, thereby enabling early intervention and prevention of full relapse.

## BACKGROUND

Schizophrenia is a chronic and severe mental disorder that affects a significant portion of the global population. The disease course of schizophrenia is often characterized by frequent relapse of psychotic symptoms, leading to treatment resistance, cognitive and functional impairment, decreased quality of life, and increased economic burden. Suicide rates are alarmingly high among schizophrenia patients, with approximately 4.9% of patients eventually taking their own lives. Therefore, relapse prevention is a critical component of schizophrenia patient management.

Effective early detection of relapse through symptom monitoring offers the opportunity for early interventions to prevent full relapse. Traditional methods of symptom monitoring, such as brief assessments during outpatient visits, are insufficient for capturing the early signs of relapse. Comprehensive scales like the PANSS, while effective, are time-consuming and impractical for frequent administration.

Recent advancements in mobile technology have opened new possibilities for continuous and periodic symptom monitoring. Mobile devices and wearable technology can collect both passive sensing data and periodic self-reports, providing a more comprehensive and real-time view of a patient's mental health status. However, the burden of frequent and lengthy self-reports can be overwhelming for patients, particularly those with cognitive impairments.

Identifying the specific PANSS items that are most predictive of impending relapse can significantly enhance the effectiveness of remote monitoring systems. Early warning signs of relapse in schizophrenia have been identified through both prospective and retrospective studies, but the inconsistency in results highlights the need for a more systematic and data-driven approach.

The present invention addresses these challenges by providing a method and system for identifying and monitoring a subset of PANSS items that exhibit significant changes prior to relapse, thereby enabling early detection and intervention.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for predicting and detecting impending relapse in schizophrenia patients using specific Positive and Negative Syndrome Scale (PANSS) items. The method involves analyzing data from relapse-prevention studies to identify individual PANSS items that exhibit significant changes prior to relapse. These items are then used to develop a remote assessment solution for monitoring symptoms and detecting relapse early in schizophrenia patients.

In one aspect, the invention includes a method for predicting and detecting impending relapse in schizophrenia patients, comprising the steps of:
1. Collecting data on individual PANSS items from a plurality of schizophrenia patients over a period of time.
2. Analyzing the collected data to identify a subset of PANSS items that exhibit significant changes prior to relapse.
3. Monitoring the identified subset of PANSS items in a patient to detect changes indicative of impending relapse.
4. Providing an alert or recommendation for intervention based on the detected changes.

In another aspect, the invention includes a system for predicting and detecting impending relapse in schizophrenia patients, comprising:
1. A data collection module for collecting data on individual PANSS items from a plurality of schizophrenia patients over a period of time.
2. An analysis module for analyzing the collected data to identify a subset of PANSS items that exhibit significant changes prior to relapse.
3. A monitoring module for monitoring the identified subset of PANSS items in a patient to detect changes indicative of impending relapse.
4. An alert module for providing an alert or recommendation for intervention based on the detected changes.

The invention further includes a computer-readable medium containing program instructions for causing a computer to perform the method steps described herein.

## DETAILED DESCRIPTION OF THE INVENTION

### PANSS

The Positive and Negative Syndrome Scale (PANSS) is a widely used tool for assessing the severity of symptoms in patients with schizophrenia. The PANSS consists of 30 items grouped into three subscales: positive symptoms (seven items), negative symptoms (seven items), and general psychopathology (sixteen items). Each item is rated on a scale from 1 to 7, with higher scores indicating more severe symptoms.

In the context of the present invention, the PANSS is used to identify specific items that exhibit significant changes prior to relapse. By focusing on these items, the invention aims to provide a more efficient and effective method for monitoring symptoms and detecting impending relapse in schizophrenia patients.

### Identification of Key PANSS Items

The identification of key PANSS items is based on data from relapse-prevention studies. Specifically, data from three randomized, double-blind, placebo-controlled withdrawal studies were pooled to analyze the changes in individual PANSS items prior to relapse. The studies involved patients with a diagnosis of schizophrenia who were treated with paliperidone in various formulations (oral extended-release, 1-month injectable, and 3-month injectable).

The analysis revealed that a subset of seven PANSS items exhibited the most significant increases prior to relapse. These items are:
1. P1 (Delusions)
2. P2 (Conceptual disorganization)
3. P3 (Hallucinations)
4. P4 (Excitement)
5. P6 (Suspiciousness)
6. G2 (Anxiety)
7. G4 (Tension)

These items are primarily from the positive symptoms subscale and the general psychopathology subscale, reflecting the typical expression of relapse as an exacerbation of positive symptoms and the role of affectivity in psychotic relapse.

### Trajectories of PANSS Item Increases

To better understand the timing of symptom changes, the trajectories of the identified PANSS items were modeled using linear and non-linear mixed effect models. The patient observations were aligned by their time of observation as days relative to relapse. The trajectories of the seven key PANSS items suggested that these items started to increase approximately 7 to 10 days before relapse and reached an average increase of 1 point about 0.3 to 1.2 days before relapse.

### Remote Assessment Solution

The identified subset of PANSS items forms the basis for a remote assessment solution for monitoring symptoms and detecting relapse early in schizophrenia patients. The solution can be implemented using mobile devices and wearable technology to collect periodic self-reports and continuous passive sensing data.

#### Data Collection Module
The data collection module is responsible for collecting data on the identified subset of PANSS items from patients. This can be achieved through a mobile application that prompts patients to self-report on the specified items at regular intervals. Additionally, passive sensing data, such as sleep patterns, mobility, and smartphone usage, can be collected to provide a more comprehensive view of the patient's mental health status.

#### Analysis Module
The analysis module processes the collected data to identify changes in the key PANSS items that are indicative of impending relapse. Statistical models, such as the linear and non-linear mixed effect models used in the studies, can be employed to analyze the data and detect significant changes.

#### Monitoring Module
The monitoring module continuously monitors the identified PANSS items and passive sensing data to detect changes indicative of impending relapse. The module can be configured to trigger alerts or recommendations for intervention based on predefined thresholds.

#### Alert Module
The alert module provides notifications to healthcare providers and caregivers when changes in the key PANSS items suggest an impending relapse. The alerts can include recommendations for early intervention, such as adjusting medication or scheduling a follow-up appointment.

### Examples

#### Example 1: Data Collection and Analysis

In a pilot study, a mobile application was developed to collect periodic self-reports on the seven key PANSS items identified in the invention. The application prompted patients to self-report on the items six times daily for a week. The compliance rate was high, with 82% of the participants completing at least 33% of all possible data points. The collected data was analyzed using the non-linear mixed effect model to identify changes in the key PANSS items. The results showed that the identified items started to increase approximately 7 to 10 days before relapse, confirming the findings from the relapse-prevention studies.

#### Example 2: Remote Monitoring System

A remote monitoring system was implemented using the identified subset of PANSS items and continuous passive sensing data. The system included a mobile application for self-reporting and a wearable device for collecting passive sensing data. The data was transmitted to a central server for analysis. The analysis module detected significant changes in the key PANSS items for a patient, triggering an alert to the healthcare provider. The healthcare provider scheduled a follow-up appointment and adjusted the patient's medication, preventing a full relapse.

### PANSS: Methods

The methods for identifying the key PANSS items and modeling their trajectories are based on data from three randomized, double-blind, placebo-controlled withdrawal studies. The studies involved patients with a diagnosis of schizophrenia who were treated with paliperidone in various formulations. The PANSS was administered at regular intervals to assess symptom severity.

#### Data Collection
Data on individual PANSS items were collected from patients who experienced a relapse during the double-blind phase of the studies. The relapse was defined by specific criteria, including psychiatric hospitalization, suicidal or aggressive behavior, and significant increases in PANSS scores.

#### Data Analysis
The collected data was analyzed to identify the subset of PANSS items that exhibited the most significant increases prior to relapse. Linear and non-linear mixed effect models were used to model the trajectories of the identified items, aligning the patient observations by their time of observation as days relative to relapse.

### PANSS: Definition of Relapse

Relapse in the studies was defined by any one of the following criteria:
1. Psychiatric hospitalization (involuntary or voluntary admission to a psychiatric hospital for decompensation of the subject’s schizophrenia symptoms).
2. Deliberate self-injury or aggressive behavior, or suicidal or homicidal ideation and aggressive behavior that was clinically significant.
3. A 25% increase in PANSS total score for two consecutive assessments < 7 days apart for patients who scored > 40 at randomization, or a 10-point increase for patients who scored ≤ 40 at randomization.
4. An increase for two consecutive assessments < 7 days apart in pre-specified individual PANSS item scores (P1 (Delusions), P2 (Conceptual disorganization), P3 (Hallucinations), P6 (Suspiciousness), P7 (Hostility), and G8 (Uncooperativeness)) to ≥ 5 for patients whose score was ≤ 3 at randomization, or to ≥ 6 for patients whose score was four at randomization.

### PANSS: Statistical Analysis

Statistical analysis was performed using linear and non-linear mixed effect models to identify the key PANSS items and model their trajectories. The patient observations were aligned by their time of observation as days relative to relapse. The trajectories of the identified items were compared to determine the timing of symptom changes.

### PANSS: Demographics and Characteristics of Patients Experienced a Relapse During Double-Blind Phase

Among 907 patients who were randomized and included in the analysis, a total of 267 patients experienced a relapse during the double-blind phase of the three studies. The demographic and baseline characteristics of the relapsed patients were compared with those who did not experience a relapse. No significant differences were observed in demographics and baseline characteristics except that the relapsed patients had higher PANSS total scores at the baseline of the double-blind phase.

### Example 5

In a follow-up study, the remote monitoring system was tested in a larger cohort of schizophrenia patients. The system was found to be effective in detecting impending relapse and facilitating early intervention. The compliance rate for self-reporting was high, and the system provided timely alerts to healthcare providers, leading to improved patient outcomes and reduced hospitalizations.

The present invention provides a robust method and system for predicting and detecting impending relapse in schizophrenia patients using specific PANSS items. By focusing on the key items that exhibit significant changes prior to relapse, the invention enables early intervention and prevention of full relapse, thereby improving patient outcomes and reducing the economic burden associated with schizophrenia.