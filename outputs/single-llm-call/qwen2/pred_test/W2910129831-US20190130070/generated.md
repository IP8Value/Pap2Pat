# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and system for modeling the progression of chronic diseases, particularly neurodegenerative disorders such as Huntington’s Disease (HD), using a Continuous-Time Hidden Markov Model (CTHMM). The invention addresses the challenges associated with tracking disease progression from observational data, including the discrete and irregular nature of observations, the multifaceted nature of symptoms, and the lack of widely accepted biomarkers for many chronic conditions.

## SUMMARY

The invention provides a framework for building probabilistic disease progression models from observational data using a Continuous-Time Hidden Markov Model (CTHMM). The framework includes a method for determining the number of disease states and constructing a probabilistic disease progression model. The CTHMM model segments the progression of the target disease into distinct disease states, capturing typical disease statuses along its natural course. The underlying progression process is modeled as a continuous-time Markov process, parameterized by a transition generator matrix and an initial state probability vector. The method uses the Expectation-Maximization (EM) algorithm to estimate the parameters of the model and the Viterbi algorithm to infer individual state sequences. The invention is particularly useful for understanding the progression of chronic diseases, enabling better staging systems, and facilitating personalized care and intervention development.

## DETAILED DESCRIPTION

The present invention is directed to a method and system for modeling the progression of chronic diseases, particularly neurodegenerative disorders such as Huntington’s Disease (HD), using a Continuous-Time Hidden Markov Model (CTHMM). The invention addresses several key challenges in disease progression modeling, including the discrete and irregular nature of observations, the multifaceted nature of symptoms, and the lack of widely accepted biomarkers for many chronic conditions.

### Continuous-Time Hidden Markov Model (CTHMM)

The CTHMM model assumes that the progression of the target disease can be segmented into \(M\) distinct disease states, where each disease state captures a typical disease status along its natural course. The underlying progression process of the target disease is modeled as a continuous-time Markov process, denoted as \(S(\tau)\), and is parameterized by an \(M \times M\) transition generator matrix \(Q\) and an \(M \times 1\) initial state probability vector \(\pi\). The \((i,j)\)-th element of \(Q\), denoted as \(Q_{(i,j)}\), characterizes the intensity of instantaneous transition from disease state \(i\) to disease state \(j\), for \(i \neq j\). The \(i\)th diagonal element \(Q_{(i,i)} = - \sum_{j \neq i}Q_{(i,j)}\), and the row sums of \(Q\) equal to \(0\). The progression of the target disease is reflected in the transition of disease states. An element \(Q_{(i,j)} = 0\) (for \(i \neq j\)) indicates that patients in disease state \(i\) cannot progress into disease state \(j\) at an instantaneous time. Different types of disease progression can be specified by imposing various constraints on the structure of \(Q\).

For example, a \(Q\) with all elements not equal to \(0\) indicates that a patient in any disease state can progress/recover to any other state. The corresponding model is referred to as the full progression model. A \(Q\) with all the lower triangular elements equal to \(0\) indicates that a disease can only get worse and the progression cannot be reversed. The corresponding model is referred to as the forward progression model. A \(Q\) with only the diagonal line and the first \(L\) upper off-diagonal lines not equal to \(0\) indicates that the disease can only progress to the next \(L\) states at any instantaneous time. The corresponding model is referred to as the \(L\)-th order forward-chain progression model. For disease progression, the most appropriate type of the model (i.e., constraints on \(Q\)) is specified based on existing knowledge of the target disease.

Given \(Q\), the transition probabilities with a time span \(\delta\) can be calculated using the matrix exponential function:
\[ A_{i,j}(\delta) = \expm(\delta Q)_{i,j} \]

### Estimation of Parameters

The Expectation-Maximization (EM) algorithm is used to estimate the parameters of the CTHMM model. The complete likelihood can be written as:
\[ P(Z, S, S(\tau); \Theta) = \prod_{n=1}^{N} \left\{ P(S_{n,1}) \prod_{t=1}^{T_n} P(S_{n,t} | S_{n,t-1}) \prod_{t=0}^{T_n} \prod_{k=1}^{K} P(Z_{n,t,k} | S_{n,t}) \right\} \]
where \(Z\) denotes the observed features, \(S\) denotes the hidden disease states, and \(\Theta\) denotes the collection of parameters in the CTHMM model.

The conditional expectation term can be broken down into two parts:
\[ E_{P(S, S(\tau) | Z; \Theta')} \left[ \log P(Z, S, S(\tau); \Theta) \right] = E_{P(S | Z; \Theta')} \left[ \log \pi + \log P(Z | S) \right] + E_{P(S, S(\tau) | Z; \Theta')} \left[ \log P(S, S(\tau); \Theta) \right] \]

The second term in the expectation can be calculated as:
\[ E_{P(S, S(\tau) | X; \Theta')} \left[ \log P(S, S(\tau); \Theta) \right] = \sum_{\delta, i, j \in [M]} C_{ij}(\delta) \left[ \sum_{k, l \in [M], k \neq l} \log Q_{kl} E(N_{kl}(\delta) | S; Q') - Q_{kl} E(R_k(\delta) | S; Q') \right] \]
where \(C_{ij}(\delta)\) denotes the number of transitions such that \(S_{t-1} = i\), \(S_t = j\), and \(\tau_t - \tau_{t-1} = \delta\), \(N_{kl}(\delta)\) is the number of transitions from state \(k\) to state \(l\) during time interval \(\delta\), and \(R_k(\delta)\) is the total time the Markov process spends in state \(k\) during the time interval \(\delta\).

In the M-step, the transition generator matrix \(Q\) and initial probability \(\pi\) are updated as follows:
\[ Q_{ij} = \frac{\sum_{\delta, k, l \in [M]} E(N_{ij}(\delta) | S(\delta) = l, S(0) = k; Q') C_{kl}(\delta)}{\sum_{\delta, k, l \in [M]} E(R_i(\delta) | S(\delta) = l, S(0) = k; Q') C_{kl}(\delta)} \]
\[ \pi_i = \frac{\sum_{n=1}^{N} P(S_{n,0} = i; \pi', Q')}{\sum_{n, j} P(S_{n,0} = j; \pi', Q')} \]

The parameters in the observational model can be updated using the sufficient statistics. Specifically, under the independent Gaussian model, the parameters \(\mu\) and \(\sigma\) can be updated as follows:
\[ \mu_{m,k} = \frac{\sum_{n,t} P(S_{n,t} = m) Z_{n,t,k}}{\sum_{n,t} P(S_{n,t} = m)} \]
\[ \sigma_{m,k}^2 = \frac{\sum_{n,t} P(S_{n,t} = m) (Z_{n,t,k} - \mu_{m,k})^2}{\sum_{n,t} P(S_{n,t} = m)} \]

### Determining the Number of Disease States \(M\)

The CTHMM model assumes that the number of disease states \(M\) is predetermined. For some of the most studied chronic diseases which have widely accepted staging systems or biomarkers, \(M\) can be determined based on clinical knowledge. However, such knowledge is not available for other diseases, especially those rare and less understood diseases such as HD. A data-driven approach is used to determine \(M\) for these diseases. The dataset is split into a training set and a testing set. A series of CTHMM models with various values of \(M\) are built on the training set. Each model is then applied to the test set to calculate the fitness measure defined as the log-likelihood on the test set given the model. The model with the highest log-likelihood provides the best fit for the data, and its corresponding \(M\) is chosen as the optimal \(M\).

### Individual State Sequences

Individual state sequences can be obtained using the standard Viterbi algorithm. Predictions of future disease states and feature values can be made by leveraging intermediate results from the model. The detailed description of the method is provided in the supplementary material.

### Application to Integrated HD Data

The dataset used in this study was integrated from four large-scale prospective observational studies of HD, namely Enroll-HD, REGISTRY, TRACK-HD/TRACK-ON, and PREDICT-HD. The integrated dataset contains 55,782 observations from 16,653 HD gene expansion carriers (HDGECs) and 2,716 control participants, with an average of 2.9 observations per participant. Several challenges prohibited directly applying the framework to the integrated HD data, including missing values, irrelevant assessments, and the high-dimensional nature of the clinical assessments. To address these issues, the Bayesian Latent Variable Analysis by Ghosh et al. was used to extract latent factor scores to represent the underlying progression trajectories. Specifically, the leading three latent factors were extracted from each of the motor, functional, and cognitive domains and concatenated to form the observed features \(Z\) in the CTHMM model.

### Results

The integrated HD data consists of participants with the number of clinical visits ranging from 1 to 25. Longitudinal information is essential for disease progression modeling, so study visits with missing values and patients with only one clinical visit were excluded. The final HD progression model was built using 8,452 HDGECs with at least 2 observations.

#### Determine the Number of Disease States

The dataset was randomly split into a training set (80%) and a testing set (20%). CTHMM models with \(M\) ranging from 6 to 12 were trained on the training set, and the log-likelihood on the testing set was calculated. The model with 9 states yielded the highest log-likelihood, and thus, the final HD progression model was built with 9 states.

#### Integrated HD Progression Model

The final HD progression model is referred to as the Integrated Huntington’s Disease Progression Model (IHDPM). The IHDPM was compared to the Shoulson and Fahn HD stages, which are defined based on the Total Functional Capacity and only cover periods after motor onset. The IHDPM provides a comprehensive view of the symptom progression in multiple domains and covers periods both before and after motor diagnosis.

The nine states in the IHDPM can be separated into three phases:
1. **Prodromal States (States 1 and 2)**: Most patients had not reached motor onset (DCL < 4).
2. **Transition States (States 3 to 5)**: The majority of patients went through motor onset.
3. **Manifest States (States 6 to 9)**: Most patients had already reached motor onset.

The IHDPM reveals that patients may undergo complex changes around the time of motor diagnosis, and it provides a more nuanced view of the disease progression compared to the Shoulson and Fahn stages.

#### Individual Disease Staging

The state sequences of individual patients can be examined and compared with the Shoulson and Fahn stages. The IHDPM model provides a more granular and detailed progression, enabling better personalized care management and subcohort segmentation for clinical trials.

### Discussion and Conclusion

The invention provides a robust framework for building disease progression models using a Continuous-Time Hidden Markov Model (CTHMM). The method is particularly useful for understanding the progression of chronic diseases, enabling better staging systems, and facilitating personalized care and intervention development. The framework is not limited to HD and can be applied to observational data of other diseases with appropriate adjustments based on clinical knowledge. Quantitative evaluation of the developed model is challenging due to the lack of a proper gold standard, but qualitative validation using existing knowledge and domain expert feedback supports the effectiveness of the model. Future research will focus on validating the model using independent datasets and translating it into clinical practice.