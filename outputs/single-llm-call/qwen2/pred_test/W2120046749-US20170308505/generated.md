# DESCRIPTION

## FIELD OF INVENTION

The present invention relates to a method and system for detecting and predicting critical transitions in complex, heterogeneous networks. More specifically, the invention involves an information dynamic spectrum framework that utilizes transfer entropy (TE) and associative transfer entropy (ATE) to quantify both global and local directional influences in such networks. The framework is capable of identifying early warning signs of system instability and impending critical transitions, thereby providing valuable insights for proactive management and intervention in various domains, including financial markets, biological systems, and technological infrastructures.

## BACKGROUND OF INVENTION

Detection and prediction of emerging tipping points are major challenges in complex systems. While self-organized emerging interactions can facilitate information exchange, they also increase the risk of attack or failure. When a system operates in a high-risk, unstable region, a small perturbation can induce a critical transition that leads to catastrophic failures. Although studies of data-driven computational models, with prior knowledge of individual systems, have greatly advanced the understanding of such emerging phenomena, we remain unequipped to accurately detect and predict tipping points prior to critical transitions.

Current works in emerging tipping point detection focus on homogeneous structured dynamical systems. For example, Scheffer et al. provide a seminal review on early warning signals for detecting critical transitions in the ecological domain. Signals such as increased temporal correlation, skewness, and spatial correlation of population dynamics are used to quantify the phenomena of critical slowing down as early warning indicators. However, these methods do not address heterogeneously networked systems, which are characterized by heavy tails, large fluctuations, scale-free properties, and non-trivial correlations. Consequently, network science has evolved from studying complex systems by modeling single, non-interacting networks to modeling interdependent networks. Buldrev et al. show that a network of networks is extremely vulnerable with respect to random failures: a random removal of a small fraction of the nodes from a network can trigger a catastrophic cascade of failures. Understanding complex interactions and coupling dynamics in large-scale complex networks is crucial for preventing dangerous system behaviors.

The spectral early warning signals (EWS) theory is one of the few attempts to detect critical transitions in heterogeneous networks. The spectral EWS theory states that the covariance spectrum can quantify the phenomenon of critical slowing down in heterogeneous networks by mathematically proving the link between complex network structures and observed time series. However, the symmetric nature of covariance spectrum does not permit the analysis of directional influences among elements.

Transfer entropy (TE) and symbolic transfer entropy (STE) have been proposed to identify directional influence in complex systems. For instance, STE is used to analyze brain activity data for the detection and identification of asymmetric dependences of brain regions in epileptic seizure activity. The transfer entropy matrix (TEM) is used on financial market data to analyze the asymmetrical influence of mature markets on emerging markets. Although these methods have shown promising results, they do not address the changing structures and dynamics of networks over time.

There is a need for a comprehensive framework that can detect and predict critical transitions by quantifying both global and local directional influences in heterogeneous networks. The present invention addresses this need by introducing an information dynamic spectrum framework that utilizes a novel associative transfer entropy (ATE) measure to decompose directional influence into associative states. This framework transforms multivariate time series of a complex system into the spectrum of the transfer entropy matrix (TEM) and the spectrum of the associative transfer entropy matrix (ATEM) to capture information dynamics. Novel spectral radius measures of TEM and ATEM are developed to detect early warning signs of source-driven instability and to reveal the sources and dynamics of directional influences. The invention further includes a method to automate the generation of early warning indicators using natural logarithmic curve modeling.

## SUMMARY OF INVENTION

The present invention provides a method and system for detecting and predicting critical transitions in complex, heterogeneous networks. The method includes the following steps:

1. **Data Collection**: Collecting multivariate time series data from the elements of the complex system.
2. **Transfer Entropy Calculation**: Calculating the transfer entropy (TE) between pairs of time series to quantify directional influence.
3. **Associative Transfer Entropy Calculation**: Decomposing the TE into associative transfer entropy (ATE) to capture the types of directional influences.
4. **Spectral Analysis**: Transforming the TE and ATE into the spectrum of the transfer entropy matrix (TEM) and the associative transfer entropy matrix (ATEM).
5. **Spectral Radius Calculation**: Calculating the spectral radius of the TEM and ATEM to quantify the total amount of entropy transferred in the entire network.
6. **Local Analysis**: Performing local TE and ATE calculations over sliding time windows to capture dynamic changes in information transfer.
7. **Early Warning Indicators**: Generating early warning indicators by analyzing the trajectories of the spectral radius of TEM and ATEM using natural logarithmic curve modeling.
8. **Probabilistic Light Cone Method**: Applying a probabilistic light cone method to predict the trajectories of the spectral radius and detect instability.

The system for implementing the method includes a data collection module, a transfer entropy calculation module, an associative transfer entropy calculation module, a spectral analysis module, a spectral radius calculation module, a local analysis module, an early warning indicator generation module, and a probabilistic light cone method module.

## DETAILED DESCRIPTION

### (2) PRINCIPAL ASPECTS

The principal aspects of the present invention include:

1. **Comprehensive Framework**: The invention provides a comprehensive framework for detecting and predicting critical transitions in complex, heterogeneous networks by quantifying both global and local directional influences.
2. **Novel Measures**: The invention introduces novel measures, including associative transfer entropy (ATE), to decompose directional influence into associative states.
3. **Spectral Analysis**: The invention utilizes spectral analysis to transform TE and ATE into the spectrum of the transfer entropy matrix (TEM) and the associative transfer entropy matrix (ATEM).
4. **Spectral Radius Calculation**: The invention calculates the spectral radius of the TEM and ATEM to quantify the total amount of entropy transferred in the entire network.
5. **Local Analysis**: The invention performs local TE and ATE calculations over sliding time windows to capture dynamic changes in information transfer.
6. **Early Warning Indicators**: The invention generates early warning indicators by analyzing the trajectories of the spectral radius of TEM and ATEM using natural logarithmic curve modeling.
7. **Probabilistic Light Cone Method**: The invention applies a probabilistic light cone method to predict the trajectories of the spectral radius and detect instability.

### (3) INTRODUCTION

The ability to detect and predict critical transitions in complex, heterogeneous networks is crucial for preventing catastrophic failures and ensuring the stability of various systems. Current methods, which focus on homogeneous structured dynamical systems, are insufficient for addressing the complexities of real-world networks. The present invention introduces an information dynamic spectrum framework that utilizes transfer entropy (TE) and associative transfer entropy (ATE) to quantify both global and local directional influences in such networks. This framework provides a comprehensive solution for detecting and predicting critical transitions, enabling proactive management and intervention in various domains.

### (4) SPECIFIC DETAILS OF THE INVENTION

#### Data Collection

The first step in the method is to collect multivariate time series data from the elements of the complex system. This data can be obtained from various sources, such as sensor readings, financial market indices, or biological measurements. The time series data should be sufficiently long and detailed to capture the dynamic behavior of the system.

#### Transfer Entropy Calculation

Transfer entropy (TE) is a directional measure of information flow between a pair of time series. It quantifies how much information is transferred from the current state into the future from one time series to another. The TE from source \( x \) to destination \( y \) with a time lag \( \tau \) in the future is defined as:

\[ TE_{x \to y}(\tau) = \sum_{x(t), y(t), y(t+\tau)} p(y(t+\tau), y(t), x(t)) \log \frac{p(y(t+\tau) | y(t), x(t))}{p(y(t+\tau) | y(t))} \]

where \( p \) denotes the probability distribution.

#### Associative Transfer Entropy Calculation

Since TE only quantifies the net amount of information going from a source to a destination, it does not distinguish the types of effects the information transferred. The invention introduces associative transfer entropy (ATE) to decompose TE by constraining the associated states of processes. ATE is defined as:

\[ ATE_{x \to y}^S(\tau) = \sum_{x(t), y(t), y(t+\tau) \in S} p(y(t+\tau), y(t), x(t)) \log \frac{p(y(t+\tau) | y(t), x(t))}{p(y(t+\tau) | y(t))} \]

where \( S \) is the associate state, a subset of the set of all possible states of \( (y(t+\tau), y(t), x(t)) \). ATE captures information transfer between two variables for a particular association of their states.

#### Spectral Analysis

The TE and ATE are transformed into the spectrum of the transfer entropy matrix (TEM) and the associative transfer entropy matrix (ATEM). The TEM is an \( m \times m \) matrix with the \( (i, j) \)-th entry \( (M)_{ij} = TE_{x_i \to x_j} \). Similarly, the ATEM is an \( m \times m \) matrix with the \( (i, j) \)-th entry \( (M^S)_{ij} = ATE_{x_i \to x_j}^S \).

#### Spectral Radius Calculation

The spectral radius of the TEM and ATEM is calculated to quantify the total amount of entropy transferred in the entire network. The spectral radius of a matrix is the supremum of the absolute values of its eigenvalues. The spectral radius of the TEM and ATEM becomes a function of time when calculated over sliding time windows.

#### Local Analysis

Local TE and ATE calculations are performed over sliding time windows to capture dynamic changes in information transfer. This allows the method to detect transient influence dynamics and provide early warnings of system instability.

#### Early Warning Indicators

The trajectories of the spectral radius of the TEM and ATEM are analyzed using natural logarithmic curve modeling to generate early warning indicators. The convex growth of the spectral radius prior to critical transitions enables the method to predict instability trajectories and provide early indications of critical transitions.

#### Probabilistic Light Cone Method

The probabilistic light cone method is applied to predict the trajectories of the spectral radius and detect instability. This method uses a moving time window over the observed spectral radius time series to derive the unknown coefficients and constants for natural logarithm curves. For a given prediction time point, the method generates a probabilistic light cone based on 95% confidence intervals of predicted trajectories with fitted natural logarithm curves. If the actual TE and ATE trajectories are outside the confidence interval, the method declares instability, serving as an early indicator of a critical transition.

#### Numerical Results

The effectiveness of the invention has been demonstrated through various numerical results. For example, the method has been applied to detect and predict critical transitions in:

1. **Oscillatory Synchronizations of Networked Non-Foster Circuits**: The method successfully identified the onset of synchronization in the positive mode, as indicated by the rapid increase in the ATE+ spectral radius curve.
2. **Pitchfork Bifurcations of Chaotic Systems**: The method detected the transition in advance, as evidenced by the rapid decrease in the spectral radius of the local TEM.
3. **Abrupt Falls of Directional Influences in Latin America Stock Indices and Dow Jones**: The method identified the rapid decrease in the TE and ATE curves before the October 2008 Crash, providing early warnings of the market crash.
4. **Exponential Growth in Wikipedia Editing Behaviors**: The method captured the rapid increase in the ATE+ curve, indicating the exponential growth in system activities.

#### Conclusion

The present invention provides a novel information dynamic spectrum framework for automated detection of critical transitions and identification of directional influences in complex, heterogeneous networks. The framework effectively quantifies associative, asymmetric directional influence, captures system-wide directional influence, and detects instability in systems with dynamic directional influence. The probabilistic light cone method further enhances the predictive capabilities of the framework, making it a valuable tool for proactive management and intervention in various domains. Future work will focus on a complete analysis of the proposed framework and the development of adaptive parameter selection methods for the probabilistic light cone analysis.