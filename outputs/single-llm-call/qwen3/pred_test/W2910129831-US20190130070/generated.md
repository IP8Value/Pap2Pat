# DESCRIPTION

## TECHNICAL FIELD

- relate to observational databases

The present invention relates to systems, methods, and computer program products for modeling and predicting the progression of chronic diseases through the analysis of longitudinal observational databases derived from heterogeneous clinical sources. Specifically, the invention pertains to the computational integration, filtering, and probabilistic modeling of high-dimensional, irregularly sampled clinical data collected from disease registries, electronic health records, and prospective cohort studies. The invention enables the construction of continuous-time, state-based representations of disease trajectories that capture subtle, multidimensional changes occurring over extended periods, particularly in conditions where traditional biomarkers are absent or insufficient. The disclosed framework is particularly suited for neurodegenerative, metabolic, and other slowly progressive disorders characterized by heterogeneous symptom expression, sparse longitudinal sampling, and lack of standardized staging systems. By transforming fragmented, noisy, and incomplete observational data into structured, interpretable disease states, the invention provides a scalable foundation for clinical decision support, patient stratification, and trial design in both rare and prevalent chronic conditions.

## SUMMARY

- introduce invention concept

The invention introduces a computer-implemented system and method for constructing a discriminative, probabilistic disease progression model from observational clinical data without reliance on predefined staging criteria or known biomarkers. The system leverages continuous-time hidden Markov modeling to infer latent disease states from high-dimensional, temporally irregular clinical observations, enabling the reconstruction of a patient’s progression trajectory across the entire disease continuum—from pre-symptomatic phases through advanced clinical manifestation. Unlike conventional approaches that treat clinical features in isolation or rely on expert-defined thresholds, the invention identifies statistically coherent patterns of co-variation across motor, cognitive, and functional domains to define disease states that reflect underlying biological progression rather than arbitrary clinical cutoffs.

- motivate disease progression model

Disease progression modeling is essential for understanding the natural history of chronic conditions, optimizing timing of interventions, and identifying patients at critical transition points. Accurate progression models facilitate early diagnosis, personalized prognosis, and efficient recruitment for clinical trials. However, the absence of objective, universally accepted biomarkers for many chronic diseases—particularly rare neurodegenerative disorders—has historically limited the development of granular, data-driven staging systems. The invention addresses this gap by deriving disease states directly from observational data, thereby uncovering progression patterns that may be invisible to clinicians relying on subjective or unidimensional assessments.

- limitations of tracking disease progression

Current methods for tracking disease progression are fundamentally constrained by three interrelated limitations. First, clinical observations are typically collected at irregular, infrequent intervals, creating discontinuous and incomplete temporal profiles. Second, individual patient records rarely span the full disease trajectory, necessitating the aggregation of data across cohorts, which introduces alignment and heterogeneity challenges. Third, most existing models analyze single clinical metrics independently, failing to capture the multidimensional, interacting nature of symptom evolution. These limitations result in coarse, static, and often misleading representations of disease course that lack predictive power and clinical utility.

- application of discriminative features

The invention overcomes these limitations by applying a discriminative feature extraction pipeline that identifies the most informative clinical variables for modeling progression, while discarding redundant or non-informative measurements. Through Bayesian latent variable analysis and statistical filtering, the system reduces high-dimensional observational data into a compact set of latent factors that collectively represent the underlying disease process. These discriminative features are then used as inputs to a continuous-time hidden Markov model, ensuring that the resulting progression states are grounded in clinically meaningful variation rather than measurement noise.

- describe disease progression prediction system

The disease progression prediction system comprises a computational architecture that ingests raw observational data, filters and transforms it into a reduced feature space, estimates latent disease states using a probabilistic model, and outputs individualized progression trajectories with associated transition probabilities and expected durations. The system is capable of predicting future disease states, estimating time-to-transition, and identifying patients approaching critical clinical thresholds, such as motor onset in Huntington’s disease, prior to their clinical recognition.

- outline method for generating disease progression model

The method for generating the disease progression model involves four sequential phases: (1) preprocessing and integration of heterogeneous observational datasets; (2) extraction and dimensionality reduction of clinical features using latent variable modeling; (3) determination of the optimal number of disease states via cross-validated likelihood maximization; and (4) estimation of transition dynamics and observational parameters using the expectation-maximization algorithm under a continuous-time hidden Markov model framework. The resulting model is fully parameterized, enabling both population-level inference and individual patient staging.

- describe computer program product

The invention further encompasses a computer program product comprising non-transitory computer-readable storage media encoded with instructions that, when executed by one or more processors, cause the system to perform the steps of the method. The program product is configured for deployment in clinical data repositories, research databases, or cloud-based analytics platforms, and is compatible with standard data formats used in electronic health records and longitudinal studies.

## DETAILED DESCRIPTION

- disclaim limitations of embodiments

It is understood that the embodiments described herein are illustrative and not exhaustive. The invention is not limited to any particular disease, dataset, or computational implementation. Modifications and adaptations to the disclosed system and method may be made without departing from the scope of the invention.

- describe purpose of detailed description

The purpose of this detailed description is to provide a comprehensive, enabling disclosure of the invention, including its components, operational principles, and practical implementations, such that a person skilled in the art may reproduce and utilize the invention without undue experimentation.

- introduce disease progression model

The disease progression model is a probabilistic, continuous-time representation of a chronic disease’s natural history, constructed as a hidden Markov process wherein the true disease state is unobserved and inferred from discrete, noisy clinical measurements. The model defines a finite set of latent states, each characterized by a unique multivariate distribution of clinical features, and transitions between states governed by a generator matrix that encodes instantaneous transition intensities.

- motivate tracking disease progression

Tracking disease progression accurately is critical for timely intervention, patient counseling, and clinical trial design. In conditions such as Huntington’s disease, where symptoms evolve over decades before overt diagnosis, the ability to detect subtle, preclinical changes enables earlier therapeutic intervention and more precise enrollment in prevention trials.

- describe difficulties in tracking disease progression

Difficulties in tracking disease progression arise from the irregular timing of clinical assessments, the high dimensionality and heterogeneity of clinical measures, the presence of missing data, and the lack of consensus on which features are most indicative of underlying progression. Traditional methods that rely on fixed thresholds or univariate trends fail to account for the complex, interacting nature of symptom evolution across domains.

- introduce system 100

System 100 is a computer-implemented architecture designed to generate and apply a disease progression model. It comprises a processor, memory, a system bus, and a set of functional components that sequentially process observational data to produce a refined, state-based representation of disease progression.

- describe system 100 components

System 100 includes a receiving component, a model generation component, an identification component, a ranking component, and a disease progression model. These components operate in concert to ingest, filter, analyze, and output a structured progression model derived from observational data.

- introduce processor 102

Processor 102 is a central processing unit configured to execute program instructions stored in memory 104, coordinating the operation of all system components and performing computationally intensive tasks such as parameter estimation, state inference, and likelihood optimization.

- describe memory 104

Memory 104 is a non-transitory storage medium that holds program instructions, intermediate data structures, input datasets, and the final disease progression model. It includes both volatile and nonvolatile memory elements to support real-time computation and persistent storage.

- introduce system bus 106

System bus 106 is a communication pathway that connects processor 102, memory 104, and external data interfaces, enabling high-speed transfer of data and control signals between system components.

- describe receiving component 108

Receiving component 108 is configured to ingest structured and unstructured observational data from multiple sources, including disease registries, electronic health records, and longitudinal cohort studies. It normalizes data formats, aligns temporal stamps, and handles missing values through imputation or exclusion protocols.

- describe model generation component 110

Model generation component 110 performs feature extraction, dimensionality reduction, and model selection. It applies Bayesian latent variable analysis to extract latent factors from clinical assessments, then determines the optimal number of disease states by maximizing cross-validated log-likelihood across a range of candidate models.

- introduce identification component 112

Identification component 112 identifies the subset of clinical features that most discriminatively differentiate between disease states. It employs statistical tests and information-theoretic measures to rank features by their contribution to state separability and removes redundant or non-informative variables.

- describe ranking component 114

Ranking component 114 quantifies the relative discriminative power of each clinical feature and each latent factor in distinguishing between adjacent disease states. It outputs a ranked list of features ordered by their sensitivity to progression, enabling prioritization for clinical monitoring and trial endpoints.

- describe disease progression model 116

Disease progression model 116 is the final output of system 100, comprising a continuous-time hidden Markov model with estimated transition intensities, initial state probabilities, and observational distributions. It is capable of predicting future disease states for individual patients, estimating time-to-transition, and assigning current disease stages based on observed clinical data.

- describe system 100 functionality

System 100 functions by sequentially processing observational data through its components to generate a probabilistic, state-based representation of disease progression. It transforms raw, heterogeneous clinical records into a unified, interpretable model that captures the natural trajectory of disease, enabling both population-level insights and individualized clinical predictions.

- describe limitations of current state of art

Current state-of-the-art approaches rely on expert-defined staging systems, univariate trend analysis, or cross-sectional comparisons that fail to model the continuous, multidimensional nature of disease progression. These methods are unable to detect preclinical changes, handle irregular sampling, or integrate data across domains, resulting in models that lack sensitivity and generalizability.

- introduce subject innovation

The subject innovation lies in the integration of latent variable modeling with continuous-time hidden Markov processes to construct a data-driven, probabilistic disease progression model from observational data without reliance on predefined clinical benchmarks. This innovation enables the discovery of previously unrecognized disease states and transition patterns, particularly in conditions lacking established biomarkers.

- describe computer processing systems

Computer processing systems employed by the invention include general-purpose processors, application-specific integrated circuits, and distributed computing clusters capable of executing the expectation-maximization algorithm and Viterbi decoding at scale.

- describe computer-implemented methods

Computer-implemented methods of the invention include the steps of data ingestion, feature extraction, state determination, parameter estimation, and individualized staging, all performed through algorithmic execution on digital computing hardware.

- describe apparatus and/or computer program products

The invention includes an apparatus comprising the components of system 100 and a computer program product comprising executable instructions stored on non-transitory media that, when executed, cause the apparatus to perform the disclosed methods.

- describe functionality of system 100

The functionality of system 100 is to transform fragmented, noisy, and high-dimensional clinical observations into a low-dimensional, probabilistic representation of disease progression that is both interpretable and predictive.

- describe FIG. 1

FIG. 1 illustrates the architecture of system 100, depicting the flow of data from receiving component 108 through model generation component 110, identification component 112, ranking component 114, and into disease progression model 116, with processor 102 and memory 104 supporting all operations.

- describe receiving component 108 functionality

Receiving component 108 is configured to accept data in multiple formats, including structured tables from disease registries and unstructured notes from electronic health records, and to align observations by timestamp, standardize units, and perform quality control to exclude incomplete or outlier records.

- describe limitations of observational data

Observational data is inherently limited by selection bias, irregular sampling intervals, missing values, and measurement noise. These limitations render direct analysis of raw data inadequate for modeling continuous progression, necessitating the computational preprocessing and transformation steps implemented by system 100.

- describe model generation component 110 functionality

Model generation component 110 applies latent variable modeling to reduce the dimensionality of clinical features, then performs grid-search optimization to determine the number of disease states that maximizes predictive likelihood on held-out data.

- describe identification component 112 functionality

Identification component 112 employs mutual information and discriminant analysis to isolate clinical features that exhibit significant variation across inferred disease states, eliminating features that contribute minimally to state separation.

- describe ranking component 114 functionality

Ranking component 114 computes the relative entropy or Kullback-Leibler divergence between the distributions of each feature across adjacent disease states, producing a ranked list of features by their ability to distinguish progression.

- describe disease progression model 116 functionality

Disease progression model 116 uses the estimated transition generator matrix and observational parameters to compute the probability of being in any disease state at any time, given a sequence of clinical observations, enabling both retrospective staging and prospective prediction.

- describe FIG. 2

FIG. 2 illustrates the filtering component 202, which is a subcomponent of model generation component 110, responsible for removing redundant, non-informative, or highly correlated clinical assessments prior to latent variable extraction.

- describe FIG. 3

FIG. 3 illustrates the pooling component 302, which aggregates clinical assessments across domains—motor, cognitive, and functional—into a unified latent space, ensuring balanced representation and preventing domain bias in model construction.

- describe composite feature engineering step

The composite feature engineering step involves the creation of latent factors through factor analysis, where each factor represents a latent dimension of disease progression derived from the co-variation of multiple clinical measures, such as motor scores, cognitive test results, and functional capacity indices.

- describe final disease progression model 116 functionality

The final disease progression model 116 is a fully parameterized continuous-time hidden Markov model that outputs, for any given patient, a sequence of inferred disease states, transition probabilities, and expected durations, enabling precise staging and prognosis.

- describe utility-based analysis

Utility-based analysis is employed to evaluate the clinical relevance of each disease state by correlating inferred states with known outcomes such as time to motor diagnosis, functional decline, or institutionalization, thereby validating the model’s predictive utility.

- describe system 100 design flexibility

System 100 is designed with modular components that can be reconfigured for different diseases, data sources, or clinical objectives, allowing adaptation to conditions such as Parkinson’s disease, Alzheimer’s disease, or chronic kidney disease without structural modification.

- describe artificial intelligence employment

Artificial intelligence techniques, including probabilistic graphical models, expectation-maximization, and Bayesian inference, are employed throughout the system to extract patterns from noisy, incomplete data and to infer hidden temporal dynamics.

- describe classification schemes

Classification schemes are used to assign patients to disease states based on posterior probabilities, with optional confidence thresholds to flag uncertain classifications for clinical review.

- describe probabilistic and/or statistical-based analysis

All operations within system 100 are grounded in probabilistic and statistical analysis, ensuring that inferences are quantified with uncertainty estimates and that model parameters are estimated using maximum likelihood and Bayesian methods.

- describe real-world examples of observational data

Real-world examples of observational data include annual clinical assessments from the Enroll-HD registry, longitudinal cognitive testing from PREDICT-HD, functional scores from REGISTRY, and neuroimaging metrics from TRACK-HD.

- illustrate disease registry data

Disease registry data consists of structured, standardized clinical measurements collected at scheduled intervals from large cohorts of genetically confirmed or clinically diagnosed patients, often spanning decades.

- motivate limitations of disease registry data

Limitations of disease registry data include inconsistent data collection protocols across sites, variable follow-up intervals, and incomplete coverage of clinical domains, which necessitate computational harmonization prior to modeling.

- introduce electronic health record

Electronic health records contain rich, unstructured clinical narratives, laboratory results, and billing codes collected during routine clinical care, offering complementary data to structured registries.

- motivate limitations of electronic health record

Limitations of electronic health records include sparse longitudinal sampling, inconsistent terminology, and high rates of missing data, which require advanced imputation and normalization techniques to render them usable for progression modeling.

- illustrate generating refined database

The invention generates a refined database by integrating multiple observational sources, applying feature filtering and latent variable extraction, and producing a reduced, discriminative dataset optimized for disease progression modeling.

- describe initial feature filtering

Initial feature filtering removes clinical assessments with excessive missingness, low variability, or poor reliability, ensuring that only high-quality, informative measures are retained for downstream analysis.

- describe composite feature engineering

Composite feature engineering constructs latent factors that represent underlying disease dimensions by modeling the covariance structure of clinical assessments across motor, cognitive, and functional domains.

- describe disease progression modeling

Disease progression modeling involves the application of a continuous-time hidden Markov model to the engineered features, estimating transition dynamics and observational distributions to define a sequence of latent disease states.

- describe disease stage assignment

Disease stage assignment is performed by applying the Viterbi algorithm to infer the most likely sequence of disease states for each patient, given their observed clinical history and the learned model parameters.

- illustrate input data

Input data consists of longitudinal clinical measurements, timestamps, patient identifiers, and metadata such as genetic status, age, and site of collection.

- describe observational data

Observational data refers to clinical measurements collected during routine care or research visits without experimental intervention, including motor scores, cognitive test results, functional assessments, and behavioral ratings.

- describe knowledge data

Knowledge data includes prior clinical understanding of disease progression, such as the expected order of symptom emergence or known biomarkers, which may be used to constrain model structure but are not required for model generation.

- illustrate analyzing observational data

Analyzing observational data involves aligning time stamps, imputing missing values, normalizing scales, and extracting latent factors to reduce dimensionality while preserving progression-relevant variation.

- describe extracting features related to disease progression

Features related to disease progression are selected based on their temporal stability, correlation with known clinical milestones, and ability to differentiate between inferred disease states.

- describe extracting features for other tasks

Features extracted for other tasks, such as diagnosis or treatment response, are excluded from the progression model unless they also demonstrate discriminative power across disease states.

- illustrate output database

Output database contains the inferred disease states, transition probabilities, expected durations, and ranked feature importance scores for each patient, formatted for clinical or research use.

- describe reduced database

Reduced database refers to the filtered, latent-variable-transformed dataset that serves as input to the disease progression model, containing only the most discriminative features.

- describe discriminative features sub-database

Discriminative features sub-database is a curated subset of the reduced database containing only those features with statistically significant variation across adjacent disease states, as determined by ranking component 114.

- illustrate flow diagram of computer-implemented method

The flow diagram illustrates the sequential steps of data ingestion, filtering, latent feature extraction, state number determination, model parameter estimation, and individualized staging, all performed by system 100.

- build preliminary disease progression model

A preliminary disease progression model is built using a candidate number of states, with parameters estimated via expectation-maximization on a training cohort.

- identify discriminative clinical features

Discriminative clinical features are identified by comparing feature distributions across inferred states and selecting those with the highest between-state variance.

- determine convergence on suitable reduced subset

Convergence on a suitable reduced subset is achieved when further feature removal no longer significantly degrades model likelihood, as determined by cross-validation.

- rank discriminative powers of clinical features

Discriminative powers are ranked using information-theoretic metrics that quantify the ability of each feature to distinguish between adjacent disease states.

- generate final disease progression model

The final disease progression model is generated by re-estimating all parameters using the full dataset and the optimal number of states, producing a robust, generalizable representation of disease progression.

- describe computer-implemented methodologies

Computer-implemented methodologies include the algorithmic execution of latent variable modeling, continuous-time Markov process estimation, and probabilistic inference, all performed on digital computing systems.

- illustrate operating environment

The operating environment includes general-purpose computers, servers, cloud computing platforms, and embedded systems capable of executing the disclosed program instructions.

- describe computer components

Computer components include processors, memory, input/output interfaces, and communication modules that enable data acquisition, processing, and dissemination.

- describe computer components

Computer components further include storage devices, network interfaces, and peripheral devices that support data input, model deployment, and user interaction.

- list storage media

Storage media include hard disk drives, solid-state drives, optical discs, flash memory, and cloud-based storage systems.

- illustrate disk storage

Disk storage is used to persistently store raw observational data, intermediate processing files, and the final disease progression model.

- describe software components

Software components include operating systems, data processing libraries, machine learning frameworks, and custom modules implementing the disclosed methods.

- explain operating system functions

Operating system functions include memory management, process scheduling, file system access, and hardware abstraction to support the execution of the invention’s software components.

- describe system applications

System applications include data ingestion tools, statistical analysis suites, and visualization dashboards that interface with the core progression model.

- illustrate input devices

Input devices include keyboards, mice, touchscreens, and data import interfaces that allow users to initiate model runs or upload datasets.

- describe interface ports

Interface ports include USB, Ethernet, and wireless communication interfaces that enable data transfer between external devices and the system.

- describe output devices

Output devices include monitors, printers, and audio systems that present model outputs, such as disease state assignments or progression trajectories.

- illustrate output adapters

Output adapters convert digital model outputs into formats compatible with electronic health record systems, clinical decision support tools, or research databases.

- describe remote computers

Remote computers may host portions of the system, such as data storage or model training, enabling distributed computation and secure data sharing.

- illustrate network connections

Network connections include local area networks, wide area networks, and secure data exchange protocols that enable communication between system components and external data sources.

- describe network interface

Network interface is a hardware and software component that enables the system to receive observational data from remote databases and transmit model outputs to clinical systems.

- list network technologies

Network technologies include TCP/IP, HTTP, HL7, FHIR, and encrypted data transfer protocols compliant with healthcare privacy regulations.

- describe communication connections

Communication connections include wired and wireless links that facilitate data exchange between system components and external databases or user interfaces.

- illustrate computer program products

Computer program products are tangible, non-transitory media encoded with executable instructions that, when loaded into memory and executed by a processor, cause the system to perform the disclosed methods.

- describe computer readable storage media

Computer readable storage media include magnetic, optical, semiconductor, and cloud-based storage devices capable of retaining program instructions for retrieval and execution.

- list storage medium examples

Storage medium examples include hard drives, SSDs, USB drives, memory cards, and server-based storage arrays.

- describe computer readable program instructions

Computer readable program instructions are sequences of code written in high-level or machine languages that implement the steps of the invention, including data preprocessing, model training, and state inference.

- explain instruction execution

Instruction execution occurs when a processor retrieves program instructions from memory, decodes them, and performs the corresponding computational operations to process data and generate outputs.

- describe electronic circuitry

Electronic circuitry includes logic gates, memory cells, and signal processors that physically implement the computational operations of the invention.

- illustrate flowcharts and block diagrams

Flowcharts and block diagrams illustrate the logical structure and data flow of the invention, depicting the sequence of operations performed by system 100 and its components.

- describe computer implemented processes

Computer implemented processes are automated, algorithmic procedures executed by digital systems to transform input data into output models without human intervention.

- explain distributed computing environments

Distributed computing environments allow the invention to be deployed across multiple interconnected systems, enabling scalable processing of large observational datasets.

- describe computer-related entities

Computer-related entities include hardware components, software modules, data structures, and communication protocols that collectively enable the invention’s functionality.

- define components and systems

Components are modular units of the invention, such as receiving component 108 or model generation component 110, while systems refer to the complete integrated architecture, such as system 100.

- describe apparatus with specific functionality

Apparatuses are physical or virtual devices configured with the components of system 100 to perform the specific function of generating disease progression models from observational data.

- explain virtual machines

Virtual machines are software-emulated computing environments that execute the invention’s program instructions on any compatible host system, enabling portability and scalability.

- clarify inclusive "or" usage

The term “or” as used herein is inclusive unless otherwise indicated, meaning that any combination of the listed elements may be employed.

- define "example" and "exemplary"

The terms “example” and “exemplary” are used to illustrate embodiments and are not intended to imply limitation or preference.

- describe processor types

Processor types include central processing units, graphics processing units, field-programmable gate arrays, and application-specific integrated circuits capable of executing the disclosed algorithms.

- explain integrated circuits

Integrated circuits are semiconductor devices that contain the electronic components necessary to perform the computational functions of the invention.

- describe memory components

Memory components include random access memory, read-only memory, cache memory, and persistent storage devices that hold data and instructions during system operation.

- list memory types

Memory types include DRAM, SRAM, NAND flash, NOR flash, and non-volatile memory modules.

- describe volatile and nonvolatile memory

Volatile memory loses its contents when power is removed, while nonvolatile memory retains data indefinitely without power, and both are employed in the system for different operational needs.

- explain RAM types

RAM types include synchronous dynamic RAM and double data rate RAM, which provide high-speed temporary storage for active computations.

- describe computer-implemented methods

Computer-implemented methods are sequences of algorithmic steps executed by digital processors to achieve the invention’s objectives, including data transformation, model training, and inference.

- explain system combinations

System combinations refer to the flexible integration of the invention’s components with external databases, clinical decision support systems, or research platforms.

- describe computer program products

Computer program products are tangible, non-transitory media encoded with instructions that, when executed, cause a computer to perform the steps of the invention.

- clarify disclosure scope

The disclosure encompasses all embodiments, variations, and adaptations of the invention that fall within the scope of the claims, whether implemented in hardware, software, or a combination thereof.