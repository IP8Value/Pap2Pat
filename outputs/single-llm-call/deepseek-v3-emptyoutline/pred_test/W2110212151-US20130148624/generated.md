Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

The present invention relates generally to wireless communication systems, and more particularly to packet scheduling algorithms in 3GPP Long-Term Evolution (LTE) networks. In modern wireless communication systems, efficient resource allocation is critical to achieving high throughput, spectral efficiency, and fairness among user equipment (UEs). Conventional packet scheduling algorithms, such as round robin (RR), maximum throughput (MT), and proportional fairness (PF), have been widely used in LTE networks. However, these algorithms exhibit limitations in terms of power efficiency, particularly in mixed traffic environments where real-time (RT) and non-real-time (NRT) traffic coexist.  

Existing scheduling algorithms primarily focus on channel quality indication (CQI) and throughput maximization but fail to account for the ratio of transmit power to the number of transmission bits. This oversight leads to suboptimal power consumption, especially in scenarios where UEs experience poor channel conditions or when the system operates near its maximum transmit power limit. Furthermore, conventional algorithms often struggle to balance fairness and throughput, resulting in monopolization of resources by UEs with favorable channel conditions.  

There is therefore a need for an improved packet scheduling algorithm that optimizes transmit power efficiency while maintaining high throughput and fairness. The present invention addresses this need by introducing a novel minimum transmit power-based (MP) packet scheduling algorithm that allocates physical resource blocks (PRBs) based on the ratio of transmit power to the number of transmission bits.  

## SUMMARY  

The present invention provides a packet scheduling algorithm for 3GPP LTE networks that improves power efficiency, throughput, and fairness. The algorithm operates in two phases: time domain packet scheduling (TDPS) and frequency domain packet scheduling (FDPS). During TDPS, a scheduling candidate set (SCS) is selected based on metrics such as buffer size, delay, and CQI reports. In the FDPS phase, PRBs are allocated to UEs in the SCS using a novel scheduling metric that minimizes the ratio of transmit power to the number of transmission bits.  

Key aspects of the invention include:  
1. **Power-Efficient Scheduling Metric**: The algorithm prioritizes UEs requiring the least transmit power per bit, ensuring optimal power utilization.  
2. **Mixed Traffic Support**: The invention accommodates both RT and NRT traffic by assigning different priorities and power constraints to each traffic type.  
3. **Improved Fairness**: By considering excess channel gain and required received power per bit, the algorithm prevents resource monopolization and enhances fairness.  
4. **Reduced Complexity**: The scheduling metric is simplified using precalculated values, minimizing computational overhead.  

The invention is particularly advantageous in multicell environments with high UE density, where power efficiency and fairness are critical. Simulation results demonstrate significant improvements in average user throughput, cell throughput, and fairness compared to conventional algorithms.  

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS  

The following detailed description provides exemplary embodiments of the invention, including specific implementations and operational details.  

### System Architecture  

The packet scheduling system operates within an evolved Node B (eNB) in a 3GPP LTE network. The system comprises:  
1. **Classifier**: Classifies incoming traffic into RT and NRT queues, assigning higher priority to RT traffic due to its delay sensitivity.  
2. **Time Domain Packet Scheduler (TDPS)**: Selects a scheduling candidate set (SCS) based on metrics such as CQI, buffer status, and delay. The TDPS reduces complexity by limiting the number of UEs considered in the FDPS phase.  
3. **Frequency Domain Packet Scheduler (FDPS)**: Allocates PRBs to UEs in the SCS using the proposed MP scheduling metric.  

### Minimum Transmit Power-Based (MP) Scheduling Algorithm  

The MP scheduling algorithm allocates PRBs to UEs based on the following scheduling metric:  

\[ M(s, n) = \frac{\Delta_{s,n}}{\omega(b_{s,n})} \]  

where:  
- \( \Delta_{s,n} \) is the excess channel gain, defined as the difference between the actual channel gain and the minimum required channel gain for successful transmission.  
- \( \omega(b_{s,n}) \) is a precalculated function of the number of transmission bits \( b_{s,n} \).  

The metric ensures that PRBs are allocated to UEs with the highest excess channel gain relative to their required received power per bit. This approach minimizes transmit power consumption while maximizing throughput.  

### Implementation Details  

1. **Link Adaptation**: The eNB selects the modulation and coding scheme (MCS) level for each UE-PRB combination based on CQI reports. Higher MCS levels are assigned to UEs with better channel conditions.  
2. **Power Allocation**: The eNB allocates transmit power to UEs in the SCS, prioritizing RT traffic to meet delay constraints. Remaining power is allocated to NRT traffic for throughput maximization.  
3. **Complexity Reduction**: The scheduling metric is simplified by precalculating \( g_{min}(b_{s,n}) \) and \( \omega(b_{s,n}) \), reducing real-time computational load.  

### Simulation Results  

The MP algorithm was evaluated in a 19-cell LTE environment with omnidirectional antennas and uniformly distributed UEs. Key findings include:  
1. **Throughput Improvement**: The MP algorithm achieved an 18% increase in average user throughput compared to the MT algorithm at high UE density.  
2. **Power Efficiency**: The algorithm saved approximately 8 dBm of transmit power while maintaining the same cell throughput as the MT algorithm.  
3. **Fairness Enhancement**: The MP algorithm distributed resources more evenly among UEs, with the best 5% of UEs occupying less than 10% of total cell throughput.  

### Conclusion  

The present invention provides a robust and efficient packet scheduling algorithm for 3GPP LTE networks. By optimizing transmit power consumption and improving fairness, the MP algorithm significantly enhances system performance in mixed traffic environments. Future work may explore advanced CQI reporting schemes to further improve accuracy and efficiency.  

This concludes the detailed description of the exemplary embodiments. The invention is not limited to the specifics described herein and may be adapted for various wireless communication systems without departing from the scope of the claims.