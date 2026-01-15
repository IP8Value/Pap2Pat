# DESCRIPTION

## FIELD OF THE INVENTION

- define field of invention

The present invention relates to a computer-implemented system and method for predicting post-transplant survival rates of prospective organ recipients using an ensemble of machine learning models tailored to distinct patient cohorts. Specifically, the invention provides a predictive algorithm that dynamically selects and applies different statistical models based on the clinical and demographic characteristics of the recipient and, where available, the donor organ. This system is designed to enhance the accuracy of survival probability estimation in the context of organ allocation, thereby improving the efficiency, fairness, and long-term outcomes of transplant decision-making. The invention operates within the domain of medical informatics and transplant logistics, integrating large-scale clinical datasets with advanced computational modeling techniques to generate individualized survival predictions that inform organ allocation protocols. The system is applicable to solid organ transplantation, including but not limited to kidneys, livers, hearts, and lungs, and is particularly suited for use in national or regional organ procurement and transplantation networks that manage large waiting lists and require objective, data-driven prioritization criteria.

## BACKGROUND

- describe organ transplant waiting list

The global demand for solid organ transplants far exceeds the supply of viable donor organs, resulting in extensive waiting lists that span months to years for many patients. In the United States alone, over 100,000 individuals are currently listed for organ transplantation, with kidneys representing the largest proportion of waiting list candidates. The allocation of scarce donor organs is governed by complex policies designed to maximize survival benefit, minimize waitlist mortality, and ensure equitable access. However, the current systems rely on simplified scoring models that fail to capture the nuanced interactions between recipient physiology, donor organ quality, and environmental factors that collectively determine post-transplant outcomes. As a result, many recipients receive organs for which their predicted survival is suboptimal, while others with higher potential for long-term survival remain on the list due to inadequate risk stratification.

- describe limitations of current organ transplant process

Current organ allocation systems, such as the Estimated Post Transplant Survival (EPTS) score and the Life Years from Transplant (LYFT) model, employ parametric statistical methods—primarily Cox proportional hazards regression—that assume linear relationships between covariates and survival outcomes. These models are constrained by rigid functional forms, limited variable selection, and an inability to adapt to heterogeneous subpopulations. For instance, EPTS uses only four recipient variables and excludes donor characteristics entirely, despite evidence that donor age, cause of death, and organ quality significantly influence graft survival. Furthermore, these models apply a single algorithm uniformly across all age groups and disease states, ignoring the fact that the relative importance of clinical variables differs substantially between younger and older recipients. The result is a system that misranks candidates, fails to optimize organ utilization, and perpetuates inefficiencies in the allocation process.

- highlight need for improved system

There is a critical and unmet need for a dynamic, data-driven system capable of generating highly accurate, individualized survival predictions that account for both recipient and donor characteristics, adapt to cohort-specific patterns, and continuously improve through machine learning. Such a system must be able to handle high-dimensional, heterogeneous clinical data, manage missing or incomplete variables without compromising predictive integrity, and provide interpretable outputs that can be integrated into existing clinical workflows. The invention addresses this need by introducing a novel ensemble approach that partitions the recipient population into biologically meaningful subgroups and applies the most statistically robust predictive model to each subgroup, thereby significantly improving ranking accuracy and transplant outcomes.

## SUMMARY

- introduce predictive organ transplant survival rate system

The invention introduces a predictive organ transplant survival rate system that leverages machine learning techniques to estimate the probability of long-term survival following organ transplantation for each prospective recipient. This system is designed to replace or augment existing allocation algorithms by providing more accurate, personalized survival predictions that reflect the complex interplay between recipient physiology, donor organ attributes, and historical transplant outcomes.

- describe method for calculating survival rates

The method for calculating survival rates involves the integration of two distinct predictive models, each optimized for a specific cohort of recipients defined by age and other clinical parameters. For younger recipients, a random survival forest constructed from conditional inference trees is employed to capture non-linear interactions and high-order variable dependencies. For older recipients, a regularized Cox proportional hazards model is applied due to its superior performance in this subgroup. The selection of model is automatic and based on recipient age, with a threshold of 50 years used to partition the population into two cohorts.

- describe receiving datasets

The system receives two primary datasets: a first dataset comprising historical clinical records of prior transplant recipients, including demographic, medical, and transplant-specific variables, as well as their observed survival durations; and a second dataset containing the clinical characteristics of a prospective recipient and, when available, the corresponding donor organ. Both datasets are anonymized and structured to include categorical, ordinal, and continuous variables, with provisions for missing data handled through imputation or exclusion based on variable type.

- describe calculating first set of estimated survival rates

The first set of estimated survival rates is calculated by identifying a subset of previous recipients whose clinical characteristics exhibit partial congruence with those of the prospective recipient. This congruence is determined through a similarity metric that evaluates the match between key variables such as age, comorbidities, diagnosis, and prior transplant history. The survival outcomes of these congruent historical cases are then aggregated to generate a probability distribution of survival over a defined time horizon, typically five years.

- describe calculating second set of estimated survival rates

The second set of estimated survival rates is derived by applying the cohort-specific predictive model—either the random survival forest or the Cox model—to the prospective recipient’s profile. The model incorporates weighted contributions from all relevant variables, with weights dynamically determined through Lasso regularization and permutation importance analysis. The output is a continuous survival probability estimate that reflects the likelihood of the recipient surviving beyond the specified time frame.

- describe generating graph of survival rates

The system generates a graphical representation of the estimated survival rates, depicting the probability of survival over time as a continuous curve, with confidence intervals and comparative benchmarks against population averages. The graph is rendered in real time and includes annotations indicating the relative position of the prospective recipient within the broader cohort of similar candidates.

- describe displaying graph

The generated graph is displayed on a user interface accessible to transplant coordinators, clinicians, and prospective recipients. The display is interactive, allowing users to toggle between recipient-only and recipient-donor combined predictions, adjust time horizons, and view the influence of individual variables on the survival estimate through sensitivity plots.

- describe characteristics of previous recipients and donors

The historical datasets include comprehensive characteristics of previous transplant recipients, such as age, sex, race, body mass index, primary renal diagnosis, dialysis duration, prior transplant history, immunological risk factors, and geographic region. Donor characteristics include age, cause of death, donor risk index, cold ischemia time, HLA mismatch, and organ quality metrics.

- describe characteristics of prospective recipient

The prospective recipient’s characteristics include all variables present in the historical dataset, with additional fields for current clinical status, urgency criteria, and time on the waiting list. Missing values are flagged and handled according to predefined protocols that preserve model integrity.

- describe characteristics of organ

The organ characteristics include donor-specific metrics such as kidney donor risk index, perfusion quality, histological findings, and organ preservation time. These variables are incorporated into the survival model only when available and are weighted according to their historical predictive power.

- describe calculating survival rates based on organ type

The system is modular and can be adapted to different organ types by retraining the ensemble model on organ-specific datasets. Each organ type—kidney, liver, heart, lung—has its own set of validated variables and model parameters, ensuring accurate predictions across the spectrum of transplant indications.

- describe assigning weights to characteristics

Weights are assigned to each characteristic based on their permutation importance and regression coefficients derived from the training phase. Variables with higher predictive value, such as recipient age and donor risk index, are assigned greater weights, while less influential variables are penalized or excluded.

- describe comparing characteristics

The system compares the prospective recipient’s profile against historical cases using a multivariate similarity algorithm that computes a weighted Euclidean distance across all relevant variables. This enables the identification of the most comparable prior recipients for survival rate estimation.

- describe calculating transplant score

A transplant score is calculated as a composite metric derived from the intersection of the first and second sets of estimated survival rates, normalized against population percentiles. This score serves as a prioritization index for organ allocation.

- describe updating predictive algorithm

The predictive algorithm is continuously updated through feedback loops that incorporate new transplant outcomes into the training dataset. Each new outcome, whether positive or negative, is used to retrain the model, adjust variable weights, and refine cohort boundaries.

- describe sending graph to user device

The graph of estimated survival rates is transmitted securely to the user device of the transplant coordinator, physician, or prospective recipient via encrypted communication protocols. The transmission includes metadata indicating the model version, date of calculation, and confidence level.

- describe determining organ recipient

The organ recipient is determined by ranking all eligible candidates according to their transplant score, with the highest-scoring candidate offered the organ first. The system ensures that organ allocation decisions are based on predicted survival benefit rather than waitlist time alone.

- describe calculating survival rates for multiple recipients

The system simultaneously calculates survival rates for all candidates on the waiting list who are compatible with the available organ, enabling a comprehensive comparison of potential outcomes across the entire candidate pool.

- describe identifying organ recipient

The recipient with the highest transplant score is identified as the optimal candidate for the organ, taking into account both predicted survival and compatibility factors such as blood type and tissue match.

- describe generating graph for organ recipient

A customized graph is generated for the identified recipient, highlighting their predicted survival trajectory relative to the cohort average and including a risk assessment for early graft failure.

- describe sending offer to organ recipient

An offer of transplantation is electronically transmitted to the recipient’s secure patient portal, accompanied by the survival graph, a summary of expected outcomes, and instructions for response.

- describe receiving response from organ recipient

The system receives a digital response from the recipient indicating acceptance or denial of the organ offer. This response is logged and used to update the predictive model.

- describe sending graph to server

The survival graph and associated metadata are transmitted to a central server for archival, audit, and model refinement purposes.

- describe determining next organ recipient

If the initial offer is declined, the system automatically identifies the next highest-scoring candidate and initiates the same offer process.

- describe sending offer to next organ recipient

The system generates and transmits a new survival graph and organ offer to the next candidate, repeating the process until an acceptance is received.

- describe receiving response from next organ recipient

The system records the response from the next candidate, updating the database and triggering the next iteration if necessary.

- describe sending graph to server

Each generated graph and associated response is securely stored on the central server, contributing to the longitudinal dataset used for algorithmic refinement.

## DETAILED DESCRIPTION

- introduce system for predictive survival rates of prospective organ transplant recipients

The system for predictive survival rates of prospective organ transplant recipients is a distributed computing architecture comprising user devices, a central computing device, and an external server, all interconnected via secure, encrypted communication channels. The system is designed to operate in real time within hospital transplant centers, regional organ procurement organizations, and national transplant networks, providing clinicians with evidence-based, dynamic survival predictions that inform organ allocation decisions.

- describe system components, including user devices, computing device, and external server

The system comprises three core components: user devices used by clinicians and patients, a central computing device responsible for model execution and data processing, and an external server for data storage, backup, and model training. Each component is configured with redundant security protocols and complies with HIPAA and other regulatory standards for protected health information.

- detail user device architecture

The user device is a mobile or desktop computing platform equipped with a processor, memory, transceiver, input device, output device, and data storage. It runs a dedicated transplant application that interfaces with the central computing device to request survival estimates, display results, and submit recipient responses.

- detail computing device architecture

The computing device is a high-performance server cluster with multiple processors, dedicated memory for model caching, and high-speed data interfaces. It hosts the predictive algorithm, performs real-time calculations, and manages data flow between user devices and the external server.

- describe database and its contents

The database contains historical transplant records from the United Network for Organ Sharing (UNOS), including recipient demographics, pre-transplant clinical status, donor characteristics, surgical details, and long-term survival outcomes. The database is updated quarterly with new data and is anonymized to remove personally identifiable information.

- explain method for calculating estimated survival rates

The method for calculating estimated survival rates involves partitioning the recipient population into two cohorts based on age, applying a random survival forest to the younger cohort and a Lasso-regularized Cox model to the older cohort, and combining the outputs with a similarity-based historical comparison to generate a final survival probability estimate.

- identify characteristics of previous prospective organ transplant recipients

Previous recipients are characterized by age, sex, race, body mass index, primary diagnosis, dialysis duration, HLA mismatch, prior transplant history, geographic region, and comorbid conditions such as diabetes and hypertension.

- identify characteristics of previous organs received by previous prospective organ transplant recipients

Previous organs are characterized by donor age, cause of death, kidney donor risk index, cold ischemia time, perfusion scores, and histological findings from biopsy.

- identify characteristics of previous persons in need of an organ transplant that did not receive the organ transplant

Persons who remained on the waiting list without receiving an organ are characterized by their clinical status at the time of last follow-up, including urgency status, waitlist time, and changes in medical condition.

- describe first set of actual survival rates of previous prospective organ transplant recipients

The first set of actual survival rates consists of observed survival durations for all previous recipients who received a transplant, censored at the end of the observation period or at the time of death or graft failure.

- describe second set of actual survival rates of previous persons in need of an organ transplant that did not receive the organ transplant

The second set of actual survival rates consists of the survival durations of individuals who remained on the waiting list without receiving a transplant, censored at the time of death, removal from the list, or end of observation.

- explain congruence of characteristics between prospective organ transplant recipient and previous persons

Congruence is determined by computing a weighted similarity score between the prospective recipient’s profile and each historical case, using a Mahalanobis distance metric that accounts for variable correlations and standard deviations.

- calculate first set of estimated survival rates based on congruent previous persons

The first set of estimated survival rates is calculated by averaging the survival outcomes of the top 100 most congruent historical recipients, weighted by their similarity scores.

- calculate second set of estimated survival rates based on congruent previous prospective organ transplant recipients

The second set of estimated survival rates is calculated by inputting the prospective recipient’s profile into the cohort-specific predictive model, which outputs a continuous survival probability curve.

- generate graph of estimated survival rates

The graph is generated using a time-axis representation of survival probability, with shaded confidence bands, reference lines for population medians, and markers indicating the recipient’s predicted position.

- send graph to external server for download by user device

The graph is encrypted and transmitted to the external server, where it is queued for download by the user device upon request by an authorized user.

- send instructions to user device to display graph

The system transmits a command to the user device to retrieve and render the graph, along with contextual instructions for interpretation and response.

- receive response from user device, including approval or denial of organ transplant

The system receives a digital response from the user device indicating whether the recipient accepts or declines the organ offer, which is then logged and used to update the predictive model.

- refine predictive algorithm based on new data

New outcomes are incorporated into the training dataset, and the model is retrained using incremental learning techniques to maintain accuracy and adapt to evolving clinical patterns.

- update weights assigned to characteristics

Variable weights are recalculated using updated permutation importance scores and Lasso regression coefficients, ensuring that the model remains responsive to changing predictors of survival.

- determine organ recipient for donor organ

The organ recipient is selected as the candidate with the highest combined transplant score, derived from the intersection of the first and second survival estimates.

- receive first dataset, including characteristics of previous prospective organ transplant recipients

The system receives the first dataset from the external server, which contains anonymized historical transplant records spanning over two decades.

- receive second dataset, including characteristics of prospective organ recipients and organ donor

The second dataset is received in real time from hospital information systems and organ procurement networks, containing the latest clinical data for candidates and available organs.

- calculate first set of estimated survival rates for each prospective organ recipient

For each candidate, the system calculates the first set of survival rates by identifying congruent historical cases and aggregating their outcomes.

- calculate second set of estimated survival rates for each prospective organ recipient

The system applies the cohort-specific model to each candidate to compute the second set of survival rates.

- analyze estimated survival rates to determine prospective organ recipient most likely to live longest

The system ranks all candidates by their combined transplant score and selects the one with the highest predicted survival probability.

- generate graph of estimated survival rates for prospective organ recipient

A personalized survival graph is generated for the selected candidate, including comparative benchmarks and risk annotations.

- send graph and offer of organ to prospective organ recipient

The graph and formal organ offer are transmitted to the recipient’s secure patient portal, requiring authentication for access.

- receive response from prospective organ recipient, including acceptance or denial of organ transplant

The system records the recipient’s response and triggers the next step in the allocation workflow.

- update predictive algorithm based on new data

The recipient’s outcome, whether successful or not, is added to the training dataset, and the model is retrained to improve future predictions.

- determine next prospective organ recipient if organ transplant is denied

If the offer is denied, the system immediately identifies the next highest-ranking candidate and initiates the same offer process.

- generate graph of estimated survival rates for next prospective organ recipient

A new survival graph is generated for the next candidate, reflecting their unique profile and predicted outcomes.

- send graph and offer of organ to next prospective organ recipient

The graph and offer are transmitted securely to the next recipient’s portal, following the same protocol.

- receive response from next prospective organ recipient, including acceptance or denial of organ transplant

The system logs the response and continues the process until an acceptance is received.

- store personally identifiable data for prospective organ recipients

All personally identifiable data is stored in encrypted form on a separate, access-controlled server that is not directly connected to the predictive algorithm, ensuring compliance with privacy regulations.

- provide graphs and information for download by prospective organ recipients

Recipients may download their survival graphs and supporting documentation through a secure portal after authentication using multi-factor identification.

- require matching personally identifiable data for download

Access to graphs and information is granted only after the system verifies the recipient’s identity using a combination of government-issued ID, biometric data, and unique patient identifier.

- describe method for determining predictive organ transplant survival rates

The method involves receiving historical and prospective datasets, calculating two sets of survival estimates using cohort-specific models, generating a graphical representation, transmitting the graph to authorized users, receiving responses, and iteratively refining the model based on new outcomes.

- describe flow chart of method for determining predictive organ transplant survival rates

The flow chart begins with the receipt of historical and prospective datasets, proceeds to cohort assignment based on age, calculates the first and second survival estimates, generates the graph, transmits it to the user device, awaits response, and loops back to update the model and identify the next candidate if necessary.

- describe user device

The user device is a secure, compliant computing platform used by clinicians and patients to interact with the system, capable of rendering graphical outputs and transmitting encrypted responses.

- describe components of user device

The user device includes a central processing unit, volatile and non-volatile memory, a transceiver for wireless communication, an input device such as a touchscreen or keyboard, an output device such as a display screen, and a data storage unit for temporary caching.

- describe memory of user device

The memory of the user device includes RAM for active processing and flash storage for caching recently accessed graphs and model metadata.

- describe transplant app

The transplant app is a native application installed on the user device that facilitates secure communication with the central computing device, renders survival graphs, and collects recipient responses.

- describe data storage devices of user device

Data storage devices include internal solid-state drives and encrypted external storage modules that comply with NIST standards for healthcare data protection.

- describe transceiver of user device

The transceiver supports encrypted Wi-Fi, Bluetooth, and cellular communication protocols and is configured to transmit data only through secure TLS 1.3 connections.

- describe output device of user device

The output device is a high-resolution color display capable of rendering complex survival curves, confidence intervals, and interactive controls.

- describe input device of user device

The input device includes a touchscreen, physical buttons, and voice recognition capabilities to allow for intuitive interaction with the system.

- describe computing device

The computing device is a clustered server environment hosted in a HIPAA-compliant data center, responsible for executing the predictive algorithm and managing data flow.

- describe components of computing device

The computing device includes multiple high-core processors, distributed memory arrays, high-speed network interfaces, redundant power supplies, and thermal management systems.

- describe memory of computing device

The memory of the computing device includes terabytes of RAM for real-time model execution and petabytes of solid-state storage for historical datasets and model checkpoints.

- describe transplant app of computing device

The transplant app is a server-side application written in Python and R, containing the ensemble model, data preprocessing modules, and communication protocols for interfacing with user devices and the external server.

- describe data storage devices of computing device

Data storage devices include encrypted, fault-tolerant storage arrays with automated backup and disaster recovery protocols.

- describe transceiver of computing device

The transceiver supports high-bandwidth, low-latency encrypted communication with user devices and external servers using industry-standard cryptographic protocols.

- describe output device of computing device

The output device is a virtual interface that generates data streams for transmission to user devices and external servers, with no physical display component.

- describe input device of computing device

The input device is a network interface that receives datasets from hospitals, organ procurement organizations, and external databases.

- describe graphical user interface

The graphical user interface is a web-based and mobile-accessible dashboard that displays survival graphs, recipient rankings, and organ offer statuses in real time.

- describe buttons of graphical user interface

Buttons include “Accept Offer,” “Decline Offer,” “View Variable Influence,” “Download Graph,” and “Request Re-evaluation,” each triggering a specific system function.

- describe functionality of buttons

Each button initiates a secure API call to the computing device, which processes the request and updates the system state accordingly.

- describe display of graph

The graph is displayed as a line plot with time on the x-axis and survival probability on the y-axis, with shaded confidence bands and vertical markers indicating median survival for the cohort.

- describe user interaction with graphical user interface

Users may hover over the graph to view survival probabilities at specific time points, click on variable names to see their impact on the prediction, and export the graph as a PDF for clinical documentation.

- define terms used in specification

“Cohort” refers to a subgroup of recipients defined by age and clinical similarity. “Congruence” refers to the degree of similarity between a prospective recipient and historical cases. “Transplant score” is a composite metric combining historical and model-based survival estimates. “Ensemble model” refers to the combination of multiple predictive models applied to distinct subgroups.

- describe scope of disclosed technology

The disclosed technology encompasses the system, method, and algorithm for predicting transplant survival using cohort-specific ensemble modeling, including all implementations, variations, and applications to any solid organ transplant.

- describe relationship between implementations

All implementations share a common architecture and algorithmic framework, differing only in the specific variables, datasets, and organ types used for training and prediction.

- describe patentable scope of disclosed technology

The patentable scope includes the method of partitioning recipients into cohorts based on age, the application of distinct predictive models to each cohort, the combination of model-based and similarity-based survival estimates, the generation of interactive survival graphs, and the iterative updating of the algorithm based on real-time outcomes.