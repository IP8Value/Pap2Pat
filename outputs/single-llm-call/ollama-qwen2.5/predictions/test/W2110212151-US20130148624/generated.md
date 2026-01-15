- **Introduction**: This paper introduces a novel packet scheduling algorithm for 3GPP LTE systems that focuses on efficient transmit power consumption. The proposed Minimum Transmit Power (MP) algorithm is compared against conventional algorithms like Maximum Throughput (MT), Round Robin (RR), and Proportional Fairness (PF). The MP algorithm considers the ratio of transmit power to the number of transmission bits, aiming to improve both throughput and fairness.

- **System Model**: The system model includes a decoupled packet scheduling framework with Time Domain Packet Scheduler (TDPS) and Frequency Domain Packet Scheduler (FDPS). A classifier differentiates between Real-Time (RT) and Non-Real-Time (NRT) traffic. RT traffic requires guaranteed bit rates and has higher priority, while NRT traffic utilizes remaining power after RT traffic is served.

- **Proposed MP Algorithm**: The proposed MP algorithm optimizes the ratio of transmit power to the number of transmission bits. This metric ensures that UEs with poor channel conditions are not entirely neglected, leading to better fairness and throughput. The algorithm is applied to both TDPS and FDPS stages in the 3GPP LTE system.

- **Simulation Setup**: Simulations were conducted using a 5 MHz bandwidth, with each TTI lasting 1 millisecond. UEs were distributed across cells, and call arrivals followed a Poisson process. Admission control was used to prevent cell congestion. The simulation parameters are detailed in Table 1, including the use of an infinite buffer model and link adaptation based on CQI.

- **Performance Metrics**: Key performance metrics include average user throughput, cell throughput, and fairness. Fairness is defined as the ratio of the best 5% UEs' throughput to the total cell throughput. The simulation results are analyzed for different scenarios, including varying numbers of UEs per cell and call arrival rates.

- **Average User and Cell Throughput**: The MP-MP algorithm outperformed other algorithms in average user and cell throughput. For example, with 25 UEs per cell, the MP-MP algorithm achieved an 18% increase in average user throughput compared to MT-MT. The NRT traffic contributed significantly to this gain, as it can utilize available power more flexibly.

- **Fairness Performance**: The MP-MP algorithm demonstrated superior fairness, with the best 5% UEs occupying less than 10% of the total cell throughput while maintaining high overall throughput. This is a significant improvement over the MT-MT algorithm, where the best 5% UEs dominated approximately 20% of the throughput.

- **Power Efficiency**: The MP-MP algorithm also showed better power efficiency, sustaining more than 10 Mbps average cell throughput with 30 dBm transmit power. It saved about 8 dBm compared to the MT-MT algorithm while maintaining the same throughput level.

- **Conclusion**: The proposed MP-MP algorithm significantly improves both throughput and fairness in 3GPP LTE systems by optimizing the ratio of transmit power to the number of transmission bits. Future work will focus on enhancing CQI reporting schemes to further improve performance.

- **Acknowledgment**: This research was supported by the Industrial Technology Development Program (Project no. KI002143) from the Ministry of Knowledge Economy (MKE) of Korea.