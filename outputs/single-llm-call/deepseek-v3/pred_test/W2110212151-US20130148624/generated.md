Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION

## BACKGROUND

Wireless communication systems have become ubiquitous in modern society, enabling mobile voice and data services across wide geographic areas. These systems rely on sophisticated scheduling mechanisms to efficiently allocate limited radio resources among multiple user equipment (UE) devices. The downlink scheduler represents a critical component within the evolved Node B (eNB) that determines how physical resource blocks (PRBs) are assigned to UEs for data transmission in the downlink direction.

The importance of the downlink scheduler cannot be overstated, as it directly impacts key system performance metrics including throughput, fairness, spectral efficiency, and power consumption. An effective scheduler must balance competing demands between maximizing overall system capacity while ensuring equitable resource distribution among users with varying channel conditions and quality of service requirements.

Current downlink scheduling approaches exhibit several limitations that constrain system performance. Traditional methods such as the round-robin (RR) approach provide basic fairness by cycling through UEs in fixed order, but achieve poor spectral efficiency. The maximum throughput (MT) method prioritizes users with the best channel conditions to maximize data rates, but can starve cell-edge users with poorer signal quality. The proportional fair (PF) algorithm attempts to balance these concerns by considering both instantaneous channel conditions and historical throughput, but still fails to optimize power efficiency.

The construction of modern downlink schedulers typically involves two operational phases: time domain packet scheduling (TDPS) and frequency domain packet scheduling (FDPS). The TDPS creates a scheduling candidate set (SCS) of UEs based on metrics like buffer status, delay constraints, and channel quality indicators (CQI). The FDPS then performs the actual PRB allocation to these candidate UEs. This two-stage approach helps manage complexity but introduces new challenges in coordinating between phases.

The proportional fair method, while widely adopted, suffers from several inherent limitations. Its focus on throughput fairness fails to consider power efficiency, which becomes increasingly important as networks approach capacity limits. The PF metric also struggles to properly handle mixed traffic scenarios where real-time (RT) and non-real-time (NRT) flows have fundamentally different quality of service requirements. These shortcomings create a clear need for improved scheduling methods that can simultaneously address throughput, fairness, and power efficiency concerns.

## SUMMARY

The present invention addresses these limitations through a novel minimum transmit power-based (MP) packet scheduling algorithm that optimizes power efficiency while maintaining throughput and fairness objectives. Exemplary embodiments of the invention provide a scheduling method that considers the ratio of transmit power to the number of transmission bits as the primary metric for resource allocation.

The object of these exemplary embodiments is to provide a downlink scheduling solution that achieves superior power efficiency compared to conventional approaches while maintaining or improving upon existing throughput and fairness performance. This is accomplished through a scheduling metric that preferentially allocates PRBs to UEs requiring the least transmit power per bit, thereby maximizing the number of bits transmitted per unit of power consumed.

Key advantages of the exemplary embodiments include significant improvements in average user throughput, particularly for non-real-time traffic flows. The invention demonstrates approximately 18% higher average user throughput compared to maximum throughput scheduling when serving 25 users per cell. The method also provides superior fairness characteristics, with cell-edge users experiencing up to 70% higher throughput compared to round-robin approaches while maintaining balanced resource distribution across all users.

The method of scheduling downlink packets according to exemplary embodiments operates by first classifying incoming traffic into real-time and non-real-time categories. For real-time traffic scheduling, the system prioritizes flows based on delay constraints while applying the MP metric. The non-real-time traffic scheduling then utilizes remaining resources, again applying the power-efficient MP allocation principle.

A critical operation involves calculating the scheduling metric M(s,n) for each UE-PRB combination as:

M(s,n) = Δs,n / (f(bs,n)/bs,n)

where Δs,n represents excess channel gain and f(bs,n)/bs,n denotes the required received power per bit. This metric effectively captures the power efficiency of potential allocations while accounting for current channel conditions.

The apparatus for implementing this downlink packet scheduling comprises a classifier module, time domain scheduler, frequency domain scheduler, and specialized MP metric calculation unit. The complete solution provides the aforementioned advantages while maintaining backward compatibility with existing 3GPP LTE systems.

## DETAILED DESCRIPTION OF EXEMPLARY EMBODIMENTS

The following detailed description provides a complete explanation of exemplary embodiments with reference to the accompanying figures. The invention encompasses a downlink packet scheduling method and apparatus particularly suited for 3GPP LTE systems, though the principles apply equally to other wireless standards.

For purposes of this description, "connected" refers to logical association between network elements, while "include" indicates non-exclusive incorporation of components or steps. The scheduling process handles two distinct traffic types: real-time (RT) traffic with strict delay requirements like voice streaming, and non-real-time (NRT) traffic with more flexible timing such as web browsing.

The real-time traffic scheduling operation follows a priority-based approach where RT flows are served before NRT traffic during each transmission time interval (TTI). Within the RT category, the MP metric determines PRB allocation to ensure power-efficient delivery while meeting delay constraints. The system maintains constant bit rate (CBR) characteristics for RT flows through appropriate link adaptation.

For non-real-time traffic scheduling, the system employs remaining resources after RT allocation using the same MP principle. The best-effort nature of NRT traffic allows flexible allocation patterns that maximize throughput while maintaining power efficiency. The scheduler ensures full utilization of available transmit power each TTI by appropriately scaling NTR allocations.

FIG. 2 illustrates the overall structure of the downlink scheduler according to an exemplary embodiment. The scheduler comprises several key components: a classifier (210) that separates RT and NRT traffic into distinct queues, a time domain packet scheduler (220) that creates the scheduling candidate set, and a frequency domain packet scheduler (230) that performs final PRB allocation using the MP metric.

The scheduling metric calculation unit (240) represents a specialized component that computes the MP metric for each potential UE-PRB combination. This unit receives CQI reports from UEs and calculates the excess channel gain Δs,n and required power per bit f(bs,n)/bs,n for each candidate allocation. The real-time traffic scheduling unit (250) and non-real-time traffic scheduling unit (260) then apply these metrics to their respective traffic categories.

The scheduling metric calculation process involves several steps. First, the system determines the minimum channel gain gmin(bs,n) required for successful transmission of bs,n bits through PRB n. This depends on the modulation and coding scheme (MCS) level selected by the link adaptation module. The excess channel gain Δs,n is then calculated as the difference between actual channel gain gs,n and this minimum required value.

FIG. 3 presents a flowchart detailing the complete downlink packet scheduling operation. The process begins by calculating the scheduling metric M(s,n) for all UE-PRB combinations in the scheduling candidate set (310). The system then selects the real-time traffic flow and PRB pair with the highest metric value (320) and allocates the PRB accordingly (330).

If unallocated real-time traffic remains (340), the process repeats from step 320 until all RT demands are satisfied or resources exhausted. For any remaining unallocated PRBs (350), the system applies the same metric-based approach to non-real-time traffic (360), allocating resources until all transmit power is consumed.

The transmission power per bit calculation follows Equation 3:

f(bs,n) = σ²s,n (2^(bs,n/Nsc) - 1)

where σ²s,n represents noise variance and Nsc is the number of subcarriers per PRB. The reception power at the UE is given by Equation 4:

Ps,n = f(bs,n)/gs,n

The complete scheduling method combines these calculations into the MP metric of Equation 5:

M(s,n) = Δs,n / (f(bs,n)/bs,n)

The link adaptation scheme selects MCS levels according to Equation 6:

bs,n = floor(Nsc·log2(1 + gs,nPs,n/σ²s,n))

where floor() ensures integer bit allocations. The excess channel gain Δs,n in Equation 7:

Δs,n = gs,n - gmin(bs,n)

completes the set of key calculations underlying the MP scheduler.

The MP scheduler's operation can be understood through an example scenario with three UEs (k, j, i) operating at different MCS levels. UE k uses high MCS level 1 requiring low power per bit due to excellent channel conditions. UE j operates at intermediate level 2 with channel quality near the minimum threshold, requiring higher power. UE i uses low level 3 but has substantial excess channel gain and low power per bit requirements. The MP scheduler would allocate resources in order of UE i, UE k, and UE j, demonstrating its power-efficient prioritization.

This detailed description illustrates how exemplary embodiments achieve superior performance through careful consideration of power efficiency in the scheduling metric. The complete solution maintains compatibility with existing 3GPP LTE systems while providing measurable improvements in throughput, fairness, and power consumption characteristics.