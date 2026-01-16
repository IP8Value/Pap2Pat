# DESCRIPTION

## BACKGROUND

In the field of telecommunications, the 3GPP Long-Term Evolution (LTE) standard has been pivotal in advancing mobile communication technologies. The LTE standard aims to achieve peak data rates of 100 Mbps in the downlink and 50 Mbps in the uplink, enhance throughput at the cell boundary, improve spectral efficiency, and support scalable bandwidth. These goals are crucial for meeting the increasing demands of mobile data traffic and ensuring a robust and efficient communication network.

Packet scheduling plays a central role in achieving these objectives. Various packet scheduling algorithms have been proposed and evaluated, focusing on fairness and throughput maximization. These algorithms include the Round Robin (RR), Proportional Fair (PF), and Maximum Throughput (MT) methods. Each of these algorithms has its strengths and weaknesses. For instance, the RR algorithm ensures fairness by rotating the scheduling among users, while the MT algorithm maximizes system throughput by prioritizing users with the best channel conditions. The PF algorithm strikes a balance between fairness and throughput by considering both the current and past data rates of users.

However, existing algorithms often overlook the importance of power efficiency, which is critical in wireless communication systems where power is a limited resource. This limitation has motivated the development of a novel Minimum Transmit Power-based (MP) packet scheduling algorithm. The MP algorithm aims to achieve power-efficient transmission to User Equipment (UE) while providing both system throughput gains and fairness improvements. This invention addresses the need for a more efficient and balanced approach to packet scheduling in LTE systems.

## SUMMARY

The present invention relates to a novel packet scheduling algorithm for 3GPP LTE systems, specifically designed to optimize power efficiency while maintaining or improving system throughput and fairness. The Minimum Transmit Power-based (MP) packet scheduling algorithm operates by allocating Physical Resource Blocks (PRBs) to UEs based on the ratio of the transmit power to the number of transmission bits. This approach ensures that the PRBs are assigned to UEs that require the least ratio of transmit power per bit, thereby minimizing power consumption and enhancing system performance.

The MP algorithm is particularly useful in mixed traffic systems, where Real-Time (RT) and Non-Real-Time (NRT) traffic coexist. In such systems, the algorithm first allocates PRBs to RT traffic UEs, which have strict delay constraints, and then uses the remaining power to allocate PRBs to NRT traffic UEs, aiming to maximize bit rate. The algorithm leverages Channel Quality Indication (CQI) information to determine the Modulation and Coding Scheme (MCS) level and the transmit power for each UE.

The performance of the MP algorithm has been evaluated through computer simulations, which demonstrate significant improvements in average user and cell throughput, especially at the cell boundary. Additionally, the MP algorithm shows enhanced fairness, as it prevents a few UEs with good channel conditions from monopolizing the spectrum resources. The invention is expected to contribute to the development of more efficient and sustainable LTE systems, addressing the growing demand for high-speed and reliable mobile data services.

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS

### Packet Scheduling Models

The basic structure of the downlink packet scheduler for RT and NRT traffics in the eNodeB (eNB) of the 3GPP LTE system is depicted in Figure 1. The packet scheduler is divided into two phases: Time Domain Packet Scheduling (TDPS) and Frequency Domain Packet Scheduling (FDPS).

#### Classifier

In the context of mixed traffic systems, the classifier plays a crucial role in efficiently managing different types of traffic. The classifier at Layer 2 data buffer classifies the mixed traffic according to the type of traffic, setting independent queues for RT and NRT traffics and assigning different priorities to these queues. RT traffics, such as voice streaming, have a Constant Bit Rate (CBR) feature and require higher priority due to their delay constraints. NRT traffics, like web browsing, have Best Effort (BE) characteristics and aim to maximize bit rate within the available power constraints.

The classifier ensures that RT traffic UEs are allocated PRBs first, followed by NRT traffic UEs. This approach ensures that the delay constraints of RT traffic are met while maximizing the use of the remaining power for NRT traffic. The classifier works in conjunction with the CQI manager, Hybrid Automatic Repeat Request (HARQ), Link Adaptation, and Quality of Service (QoS) manager to ensure efficient packet scheduling.

#### Time Domain Packet Scheduling (TDPS)

The primary goal of the TDPS is to set the Scheduling Candidate Set (SCS). The TDPS does not directly allocate PRBs but restricts the number of UEs for the FDPS to reduce scheduling complexity. The SCS is chosen based on various metrics, including buffer size, delay, and CQI reports. The SCS information is then conveyed to the FDPS, which only considers the UEs restricted by the TDPS as candidates for PRB allocation.

The TDPS takes into account the data in the L2 buffer and HARQ status. If retransmission is requested through HARQ, the UEs requiring retransmission are automatically included in the SCS. This ensures that the TDPS is responsive to the current network conditions and can adapt to changing traffic patterns.

#### Frequency Domain Packet Scheduling (FDPS)

In the FDPS phase, PRBs are directly allocated to the UEs, and their data are transmitted. The FDPS delivers the allocated data to physical layer (L1) devices, which modulate the signals and transmit them through the physical channel. The FDPS considers only the SCS during one Transmission Time Interval (TTI) and completes the scheduling process when all transmit power is consumed.

A UE can load information on multiple PRBs, but a PRB cannot be shared by more than one UE at the same time. The FDPS ensures that the PRBs are allocated efficiently, taking into account the channel conditions and the power constraints of the system.

### Packet-Scheduling Algorithms

#### Conventional Packet-Scheduling Algorithms

Various conventional packet scheduling algorithms have been proposed and evaluated in terms of system throughput and fairness. The Round Robin (RR) algorithm ensures fairness by rotating the scheduling among users, while the Maximum Throughput (MT) algorithm maximizes system throughput by prioritizing users with the best channel conditions. The Proportional Fair (PF) algorithm balances fairness and throughput by considering both the current and past data rates of users.

- **Round Robin (RR) Algorithm**: The RR algorithm uses the elapsed time since the last scheduled time for each UE as the scheduling metric. This ensures that all UEs are given equal opportunities to access the network resources, promoting fairness.
  
- **Maximum Throughput (MT) Algorithm**: The MT algorithm focuses on spectral efficiency and achieves the best system throughput. The scheduling metric is based on the data rate, which is calculated based on the recommended Modulation and Coding Scheme (MCS) level determined by the CQI reported from the UE. The UE with the highest data rate is given the highest priority.

- **Proportional Fair (PF) Algorithm**: The PF algorithm aims to solve the monopolization issue in the MT algorithm by considering the ratio of the current data rate to the past average user data rate. This ensures that UEs with poor channel conditions are given a higher priority, improving fairness.

#### Proposed Minimum Transmit Power-based (MP) Packet-Scheduling Algorithm

The proposed MP packet scheduling algorithm introduces a new metric that considers the ratio of the transmit power to the number of transmission bits. This metric is designed to enhance system performance by optimizing power efficiency while maintaining or improving throughput and fairness.

The MP algorithm selects UEs to be allocated PRBs in ascending order of the ratio of the transmit power \( P_{s,n} \) to the number of transmission bits \( b_{s,n} \):

\[
M(s, n) = \frac{P_{s,n}}{b_{s,n}}
\]

Where:
- \( P_{s,n} \) is the transmit power required for the PRB \( n \) of the UE \( s \).
- \( b_{s,n} \) is the number of transmission bits for the PRB \( n \) of the UE \( s \).

The channel power \( g_{s,n} \) of the PRB \( n \) of the UE \( s \) is used to calculate the transmit power. Assuming that the same MCS level is used for all subcarriers in a PRB, the minimum transmit power \( f(b_{s,n}) \) is given by:

\[
f(b_{s,n}) = \sigma^2_{s,n} \cdot 2^{b_{s,n}}
\]

Where:
- \( \sigma^2_{s,n} \) is the noise variance for the subcarriers in the PRB \( n \) at the UE \( s \).

The excess channel gain \( \Delta_{s,n} \) is defined as:

\[
\Delta_{s,n} = g_{s,n} - g_{\text{min}}(b_{s,n})
\]

Where:
- \( g_{\text{min}}(b_{s,n}) \) is the minimum channel gain required for the successful transmission of \( b_{s,n} \) bits.

The scheduling metric \( M(s, n) \) can be rewritten as:

\[
M(s, n) = \frac{\sigma^2_{s,n} \cdot 2^{b_{s,n}}}{b_{s,n} \cdot \Delta_{s,n}}
\]

The MP scheduler assigns the PRB \( n \) to the UE with the largest excess channel gain compared to the required received power per bit. For UEs with equal excess channel gain, the PRB is assigned to the UE with the smaller received power per bit.

### Simulation Environment

The performance of the proposed MP packet scheduling algorithm was evaluated through computer simulations based on the 3GPP LTE downlink specifications. The simulation environment includes a 19-cell model with wrap-around, where omnidirectional antennas are used, and UEs are uniformly distributed. Calls are generated based on a Poisson arrival rate, and a simple admission control is applied to prevent users from gathering in a few cells. The simulation parameters are as follows:

- Transmission bandwidth: 5 MHz
- Number of PRBs per TTI: 25
- Maximum allowable transmit power: 46 dBm
- Traffic types: RT and NRT
- Guaranteed bit rate (GBR) for RT traffic: 64 kbps
- Infinite buffer model
- Link adaptation based on CQI
- No HARQ scheme applied

### Simulation Results

The proposed MP packet scheduling algorithm was compared with the conventional MT, RR, and PF packet scheduling algorithms. The results demonstrate the superior performance of the MP algorithm in terms of average user and cell throughput, as well as fairness.

#### Average User and Cell Throughput Performance

Figure 3 shows the average user throughput, defined as the ratio of the total throughput in a cell divided by the total number of UEs, with different maximum numbers of UEs in a cell. The MP-MP algorithm achieves even better average UE throughput than the MT-MT algorithm. The MP algorithm's spectral efficiency is more efficient as the maximum number of UEs in a cell increases. When the maximum number of UEs in a cell is 25, the MP-MP algorithm achieves an 18% increase in average user throughput compared to the MT-MT algorithm.

Most of the gain in average user throughput of mixed traffic UEs comes from NRT traffic UEs, which can receive as many available data as possible, while RT traffic UEs do not receive more data than their target data rates. The MP algorithm also shows the best capacity for RT traffic, providing more capacity when applied. Under the same maximum number of UEs in a cell, the MP-MP algorithm shows the best throughput per UE, indicating that better average user throughput occurs with more UEs due to the efficiency of transmit power consumption.

As the call arrival rate increases, the MP-MP algorithm provides more eminent performance. For example, when the call arrival rate is \( 10^{-2} \), the algorithm shows a 6% gain in average cell throughput for total UEs compared to the MT-MT algorithm.

Figure 5 shows the average cell throughput at the cell boundary with call arrival rate. In the simulation, 20% of the UEs were located at the cell boundary, where power efficiency is particularly important. Compared to the RR-RR algorithm, the MP-MP algorithm achieves a 70% gain at the cell boundary for a call arrival rate of \( 10^{-2} \). The improved spectrum efficiency is attributed to the proposed MP scheduling algorithm, which considers the ratio of the transmit power to the number of transmission bits.

Figure 6 shows the average cell throughput with transmit power. The MP-MP algorithm can sustain more than 10 Mbps average cell throughput with 30 dBm. Additionally, the MP-MP algorithm can save about 8 dBm of transmit power compared to the MT-MT algorithm while maintaining the same cell throughput.

#### Fairness Performance

Figure 7 shows the fairness and cell throughput. Fairness is defined as the ratio of the best 5% UEs' throughput to the total cell throughput. The MT-MT algorithm shows the worst fairness, with the best 5% UEs occupying approximately 20% of the total cell throughput. In contrast, the RR-RR and PF-PF algorithms show less than 10 Mbps cell throughput, with the best 5% UEs occupying less than 10% of the cell throughput. The MP-MP algorithm, however, achieves more than 10 Mbps cell throughput with the best 5% UEs occupying less than 10% of the cell throughput, demonstrating better performance in terms of both cell throughput and fairness.

Figure 8 shows the distribution of normalized throughput with respect to the UE index. The normalized throughput is defined as the ratio of the throughput per UE to the total throughput in a cell. The MT-MT algorithm centralizes a large portion of the normalized throughput in only a few UEs with good channel conditions. In contrast, the normalized throughput by the RR-RR, PF-PF, and MP-MP algorithms is fairly distributed, with the MP-MP algorithm showing relatively equal transmission probabilities for all UEs.

Figure 9 shows the distribution of normalized throughput of a UE with the distance from the serving eNB normalized by the cell radius. The distance is the most important factor affecting the channel condition. In the MT-MT and PF-PF algorithms, the normalized throughput is centralized and decreases as the distance from the center increases. However, the normalized throughput in the RR-RR and MP-MP algorithms is randomly spread over the entire region. The reason for this is that the MP algorithm considers the ratio of the transmit power to the number of transmission bits, leading to improved performance in terms of fairness and throughput, especially at the cell boundary.

### Conclusion

In conclusion, the proposed Minimum Transmit Power-based (MP) packet scheduling algorithm for 3GPP LTE systems demonstrates significant improvements in system performance, including average user and cell throughput, especially at the cell boundary, and fairness. The MP algorithm optimizes power efficiency by considering the ratio of the transmit power to the number of transmission bits, ensuring that PRBs are allocated to UEs that require the least ratio of transmit power per bit. This approach not only enhances system throughput but also prevents a few UEs with good channel conditions from monopolizing the spectrum resources, thereby improving fairness.

The simulation results confirm the effectiveness of the MP algorithm in mixed traffic systems, where it first allocates PRBs to RT traffic UEs and then uses the remaining power to allocate PRBs to NRT traffic UEs. The proposed algorithm is expected to contribute to the development of more efficient and sustainable LTE systems, addressing the growing demand for high-speed and reliable mobile data services. Further work includes optimizing the CQI reporting scheme to enhance the accuracy of the CQI information, which is crucial for the performance of the proposed downlink scheduling algorithm.