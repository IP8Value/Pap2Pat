# DESCRIPTION

## STATEMENT OF FEDERALLY SPONSORED RESEARCH
The research leading to the development of the invention described herein was supported by federal funds. The government may have certain rights in the invention.

## TECHNICAL FIELD
The present invention relates to the field of wireless communication systems, particularly to the use of electrically reconfigurable antennas in multiple-input multiple-output (MIMO) ad hoc networks. More specifically, the invention pertains to methods and systems for selecting antenna configurations in such networks to maximize network sum capacity while mitigating interference.

## BACKGROUND
Recent advancements in the field of ad hoc networks have led to significant improvements in physical layer techniques, including the application of smart antennas and antenna diversity techniques, the development of medium access control protocols for MIMO ad hoc networks, and the creation of adaptive algorithms for antenna beamforming. Directional antennas, such as phased arrays and switchable parasitic elements antennas, have been proposed to reduce interference and maximize network throughput. However, these solutions often face practical limitations, especially in compact portable devices where space constraints hinder the integration of multiple directional antennas.

To address these challenges, electrically reconfigurable antennas have emerged as a promising solution. These antennas can dynamically change their radiation patterns, thereby increasing channel capacity and reducing space occupation. Despite their potential, there has been limited research on the practical implementation and field testing of reconfigurable antennas in multi-link MIMO ad hoc networks. The present invention aims to fill this gap by providing a method and system for efficiently selecting antenna configurations in such networks, thereby maximizing network sum capacity and mitigating interference.

## SUMMARY
The present invention provides a method and system for selecting antenna configurations in a MIMO ad hoc network using electrically reconfigurable antennas. The method includes the steps of defining a set of reconfigurable antenna configurations, estimating the channel matrix for each configuration, and selecting the configuration that maximizes network sum capacity. The system comprises a plurality of nodes, each equipped with reconfigurable antennas, and a controller for managing the configuration selection process.

In one embodiment, the invention provides a distributed configuration selection algorithm that allows each node to independently select its antenna configuration based on local channel information. This algorithm is designed to optimize individual link capacity while ensuring network-wide performance. Additionally, a centralized configuration selection algorithm is provided, which uses global network information to assign the optimal configuration to each node.

The invention also includes a detailed analysis of the performance of two reconfigurable antenna architectures: the reconfigurable printed dipole array (RPDA) and the reconfigurable circular patch antenna (RCPA). The performance metrics considered include spatial correlation, radiation efficiency, and network sum capacity. The results demonstrate that reconfigurable antennas can significantly enhance the performance of MIMO ad hoc networks, particularly when the number of configurations and the diversity between patterns are optimized.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

### I. RECONFIGURABLE ANTENNA ARCHITECTURES
The present invention utilizes two types of reconfigurable antennas: the reconfigurable printed dipole array (RPDA) and the reconfigurable circular patch antenna (RCPA). Both antennas are designed to operate in the 2.4-2.5 GHz frequency band, typical of 802.11-like MIMO networks.

#### A. Reconfigurable Printed Dipole Array (RPDA)
The RPDA consists of two microstrip dipoles separated by a quarter-wavelength distance. Each dipole can be electrically reconfigured in length using PIN diode switches, resulting in two configurations: "long" and "short." Consequently, the RPDA can be configured in four different states: both antennas "long" (l-l), both antennas "short" (s-s), one antenna "short" and the other "long" (s-l), and vice versa (l-s).

The radiation patterns of the RPDA are measured in an anechoic chamber, and the spatial correlation coefficients between the patterns are calculated. The results show that the correlation values between radiation patterns at the two ports of the array are small enough to provide significant diversity gain. However, the level of diversity between the different configurations generated at the same port is not high, indicating that the RPDA offers a large number of radiation patterns with moderate diversity.

#### B. Reconfigurable Circular Patch Antenna (RCPA)
The RCPA consists of a circular patch with a variable radius that can be electrically adjusted by turning switches on and off. The RCPA has two configurations: "Mode 3" and "Mode 4," corresponding to the excitation of the TM31 and TM41 electromagnetic modes, respectively.

The radiation patterns of the RCPA are also measured in an anechoic chamber, and the spatial correlation coefficients are calculated. The results show that the patterns generated at the two ports of the RCPA are spatially orthogonal for both configurations, and the level of diversity between the two configurations is very high. However, the RCPA suffers from low radiation efficiency due to the excitation of higher-order modes on a lossy substrate.

### II. SYSTEM MODEL AND NOTATION
The ad hoc network is modeled as consisting of \( L \) co-located links that interfere with each other. Each link is single-hop, and the transmit-receive pairs are pre-determined. The notation used in the model is as follows:
- \( H_{i}^{rc,j tc} \) denotes the channel between the receiver of link \( i \) and the transmitter of link \( j \), which is a function of the receive configuration \( i^{rc} \) and the transmit configuration \( j^{tc} \).
- \( x_i \) is the signal vector of link \( i \).
- \( (\cdot)^H \) denotes the conjugate transpose.

The input-output relationship for link \( l \) is given by:
\[ y_l = H_l^{rc,l tc} x_l + \sum_{i \in L \setminus l} H_l^{rc,i tc} x_i + n \]
where \( n \) is the noise vector.

The interference plus noise covariance matrix for link \( l \) is:
\[ R_l = \sigma^2 I + \sum_{i \in L \setminus l} H_l^{rc,i tc} H_l^{rc,i tc H} \]

The capacity of link \( l \) is:
\[ C_l = \log_2 \det \left( I + \frac{P_T}{\sigma^2} H_l^{rc,l tc H} R_l^{-1} H_l^{rc,l tc} \right) \]

The sum capacity of the network is:
\[ C_{\text{sum}} = \sum_{l=1}^L C_l \]

### III. ANTENNA CONFIGURATION SELECTION METHODS
Three different cases for using reconfigurable antennas in the network are considered:
1. **Double-Side Reconfigurable Antennas (DSRA)**: Both the receiver and the transmitter of any given link can adapt their configurations.
2. **Receiver-Side Reconfigurable Array (RXRA)**: Only the link receiver is allowed to switch its configuration.
3. **Transmitter-Side Reconfigurable Array (TXRA)**: Only the link transmitter is allowed to switch its configuration.

#### A. Centralized Configuration Selection Technique
A centralized controller with instantaneous knowledge of all communication and interference channels optimizes the sum capacity by solving the following optimization problem:
\[ \max_{c} C_{\text{sum}}(c) \]
where \( c \) is a vector containing the configurations for each node. The centralized controller conducts an exhaustive search over all possible antenna configurations in all network nodes.

#### B. Distributed Configuration Selection Technique
Each link makes its own configuration selection using only local channel information. The optimization problem for link \( l \) is:
\[ \max_{c_l} C_l(c_l) \]
where \( c_l \) is the configuration vector for link \( l \). The distributed technique is an iterative procedure where each link continually updates its configuration selection in response to changes in the interference.

#### C. Single Side Reconfigurable Antennas
When only the receive configurations are allowed to change (RXRA), a change in configuration affects only the link's own capacity, and iterations are not needed. This simplifies the implementation and reduces overhead. The RXRA technique is also desirable because it maximizes both individual link capacity and network sum capacity.

### IV. DATA COLLECTION
The performance of the reconfigurable antennas and configuration selection methods was evaluated through field measurements and electromagnetic ray-tracing simulations in an indoor environment.

#### A. Measurement Setup
Measurements were conducted in the third floor of the Bossone building at Drexel University using the HYDRA Software Defined Radio platform. Two RCPAs and four RPDAs were built and equipped with PIN diodes. The network measurement topology involved three receivers and three transmitters, creating six different network topologies. Small-scale fading effects were captured by moving the receive elements on a robotic antenna positioner. Channel responses were measured and averaged over 100 noisy channel estimates for each subcarrier.

#### B. Simulation Setup
Simulated channels were acquired using the electromagnetic ray tracer FASANT. A 3D model of the hallway on the third floor of the Bossone building was used as the geometry input. The measured 3D radiation patterns of the antennas were used in the simulations. The simulations were conducted by transmitting a single tone at 2.484 GHz to obtain the channel matrices. The sum network capacity was calculated for each configuration selection method.

### V. RESULTS
The performance of the reconfigurable antennas and configuration selection methods was analyzed using cumulative distribution functions (CDFs) of the network sum capacity.

#### A. RCPA
1. **Sum Capacity Results**
   - The CDFs of the network sum capacity using the Centralized and Distributed configuration selection methods showed significant increases in sum capacity compared to non-reconfigurable Mode 3 circular patch antennas.
   - The measured sum capacity increases were greater than those predicted by simulations. For the Centralized DSRA scheme, simulations showed a 50% increase, while measurements showed a 75% increase.
   - The Distributed RXRA technique performed better in measurements, while the Distributed TXRA technique performed better in simulations.

2. **Convergence Properties**
   - The average number of iterations required for convergence was low, with more than 99% of scenarios reaching convergence before the 10th iteration.

#### B. RPDA
1. **Sum Capacity Results**
   - The CDFs of the network sum capacity using the Centralized and Distributed configuration selection methods showed significant increases in sum capacity compared to non-reconfigurable S-S dipole antennas.
   - The measured sum capacity increases were greater than those predicted by simulations. For the Centralized DSRA scheme, simulations showed a 50% increase, while measurements showed a 75% increase.
   - The Centralized DSRA technique performed the best, and the Distributed TXRA technique performed the worst.

2. **Convergence Properties**
   - The iterative configuration selection schemes using RPDAs required more iterations before convergence compared to RCPAs. However, the majority of scenarios still converged within 10 iterations.

#### C. Comparing RCPA with RPDA
- The performance of the RPDA was higher in both percentage increase relative to non-reconfigurable architectures and in absolute sum network capacity values.
- The RPDA's superior performance can be attributed to its larger number of configurations and closer efficiency between configurations.

#### D. Effect of the Number of Configurations
- Reducing the number of configurations available to the RPDA (to two) resulted in a significant decrease in network sum capacity, highlighting the importance of having a large number of configurations.

#### E. Effect of Correlation Between Patterns
- Normalizing the radiation patterns to remove the effects of radiation efficiency showed that uncorrelated patterns significantly improve network sum capacity.

### VI. SOFTWARE IMPLEMENTATION
The software implementation of the configuration selection algorithms involves the following steps:
1. **Channel Estimation**: Each node estimates the channel matrix for all available configurations.
2. **Capacity Calculation**: The capacity for each configuration is calculated using the estimated channel matrix.
3. **Configuration Selection**: The configuration that maximizes the network sum capacity is selected.
4. **Iteration**: For distributed methods, the process is repeated iteratively until convergence.

The software is designed to be flexible and adaptable to different network topologies and reconfigurable antenna architectures. It can be integrated into existing MIMO ad hoc network systems to enhance their performance.

### VII. CONCLUSIONS
The present invention demonstrates the significant benefits of using electrically reconfigurable antennas in MIMO ad hoc networks. The proposed methods and systems for selecting antenna configurations can effectively maximize network sum capacity while mitigating interference. The performance of the reconfigurable antennas is influenced by the number of configurations, the correlation between patterns, and the radiation efficiency. The distributed configuration selection technique, particularly the RXRA method, strikes a good balance between performance and practical implementation constraints. Future research will focus on optimizing the design of reconfigurable antennas and exploring closed-loop power allocation algorithms for further performance improvements.