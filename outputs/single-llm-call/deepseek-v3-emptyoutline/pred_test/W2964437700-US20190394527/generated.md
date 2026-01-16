Here is the complete patent application following the provided outline and incorporating the research paper's invention:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates generally to wireless communication systems, and more particularly to systems and methods for improving quality of experience (QoE) for Hypertext Transfer Protocol (HTTP) Adaptive Streaming (HAS) video services in cellular radio access networks (RANs). The invention specifically addresses the technical challenges of detecting streaming video users experiencing degraded quality and proactively prioritizing their scheduling at base stations (BSs) using machine learning classifiers that operate solely on downlink packet header information.  

## BACKGROUND  

The exponential growth of mobile video traffic has created significant challenges for mobile network operators (MNOs) in maintaining satisfactory user QoE. Current approaches to managing this growth include network upgrades (e.g., 5G deployment), edge caching solutions, and bitrate throttling techniques. However, these solutions fail to address the fundamental scheduling inefficiencies in how Radio Access Networks (RANs) handle HAS traffic.  

Conventional traffic classification methods suffer from several limitations. Rule-based schemes are vulnerable to spoofing attacks, while Deep Packet Inspection (DPI) techniques fail when applied to encrypted payloads. Prior machine learning approaches for traffic classification either require bidirectional flow monitoring (increasing latency and complexity) or rely on non-standard features that are impractical for real-time implementation in base stations.  

In current LTE systems, HAS traffic is typically scheduled as best-effort traffic (QoS class identifiers 8 and 9), leading to suboptimal user experiences during network congestion. Users frequently experience prolonged playback delays, frequent resolution changes, and rebuffering events. Existing solutions attempt to reactively address these issues through client-side Adaptive Bit Rate (ABR) algorithms or network-side approaches that assume ideal knowledge of buffer states and encoding rates - assumptions that don't hold in practical implementations.  

There exists a critical need for a practical, real-time solution that can:  
1) Accurately detect HAS flows using only downlink packet headers  
2) Determine user playback state and video resolution quality with low latency  
3) Proactively prioritize scheduling of users experiencing poor QoE  
4) Maintain acceptable service levels for non-video traffic  
5) Operate within the computational constraints of existing base station hardware  

## SUMMARY  

The present invention provides a comprehensive solution to the above challenges through an experience-centric scheduling framework comprising three key innovations:  

1) **Lightweight Traffic Classification System**: A Random Forest Classifier (RFC)-based system that analyzes only standard IP/TCP headers of downlink packets to:  
   - Identify HAS traffic flows among various service types  
   - Determine client playback state (Buffering or Steady-State)  
   - Detect current video resolution quality (from 144p to 1080p)  

2) **Real-time Feature Extraction Methodology**: A novel feature set derived exclusively from downlink packets that enables:  
   - Service classification using just the first 5 packets of a new flow  
   - Playback state detection with 95% accuracy using 0.1s sampling intervals  
   - Resolution quality detection with 90% accuracy using 1s sampling intervals  

3) **Experience-aware Scheduler**: A modified proportional fair scheduler that:  
   - Applies dynamic prioritization weights based on detected user QoE states  
   - Implements both Strict Priority (SP) and Weighted Proportional Fair (WPF) scheduling policies  
   - Maintains balanced resource allocation between video and non-video traffic  

The invention achieves several technical advantages over prior approaches:  
- Eliminates need for bidirectional flow monitoring or payload inspection  
- Reduces feature extraction latency by using only downlink packets  
- Minimizes computational overhead through optimized Random Forest implementations  
- Enables proactive QoE improvement rather than reactive stalling mitigation  
- Maintains compatibility with existing base station hardware architectures  

Experimental results demonstrate significant QoE improvements, including:  
- 30-50% reduction in initial playback delays  
- 12.1% improvement in worst-case Mean Opinion Scores (MOS)  
- Higher average video resolution quality  
- Minimal impact on non-video traffic throughput  

## DETAILED DESCRIPTION  

The present invention will now be described in detail with reference to the technical implementation and operational characteristics.  

### System Architecture  

The invention operates within a standard cellular network architecture comprising one or more base stations serving multiple user equipment (UE) devices. Each base station implements three key functional modules:  

1) **Traffic Classifier Module**:  
   - Implements a cascade of Random Forest Classifiers (RFCs)  
   - First-stage classifier identifies service type (HAS vs. non-HAS)  
   - Second-stage classifiers determine playback state and resolution  
   - Processes only downlink IP/TCP packet headers  
   - Maintains sub-millisecond classification latency  

2) **Feature Extraction Engine**:  
   - For service classification: Extracts server/client port numbers, PUSH bit counts, and median packet size from first 5 packets  
   - For state detection: Tracks packet counts and sizes over sliding 0.1s windows  
   - For resolution detection: Aggregates state detection features over 1s intervals  
   - Implements memory-efficient sample buffering  

3) **Experience-aware Scheduler**:  
   - Implements modified Proportional Fair (PF) algorithm  
   - Applies dynamic weights based on classifier outputs:  
     * v_j[n] > 1 for users in Buffering state  
     * w_j[n] > 1 for users with resolution below threshold  
   - Supports both Strict Priority (SP) and Weighted PF (WPF) modes  
   - Maintains fairness through α and β tuning parameters  

### Classifier Implementation  

The traffic classification system employs an optimized Random Forest architecture with the following characteristics:  

**Service Classifier**:  
- 10 decision trees with maximum depth of 246 nodes  
- Processes first 5 downlink packets of new flows  
- Feature set: {Source Port, Destination Port, PUSH bit count, Median IP Length}  
- Achieves 98% accuracy in identifying HAS traffic  

**Playback State Classifier**:  
- 10 decision trees with maximum depth of 300 nodes  
- Sampling interval (T_w) = 0.1s  
- Number of samples (n) = 5  
- Features: {Packet count, Total payload size} over sliding window  
- 95% accuracy in distinguishing Buffering vs. Steady-State  

**Resolution Classifier**:  
- 10 decision trees with maximum depth of 541 nodes  
- Sampling interval (T_w) = 1s  
- Number of samples (n) = 15  
- Features: Aggregated state detection features  
- 90% accuracy in resolution detection (6 quality levels)  

The classifiers are designed for efficient hardware implementation:  
- Total memory requirement: < 2.5 MB  
- Computational overhead: < 13 MIPS  
- Compatible with standard Intel x86 and ARM processors  

### Scheduling Algorithms  

The invention implements two experience-aware scheduling variants:  

**Strict Priority (SP) Scheduler**:  
- Assigns infinite weight (v_j = ∞ or w_j = ∞) to unhappy users  
- Always prioritizes unhappy users when detected  
- Provides maximum QoE improvement for video users  
- May impact non-video traffic under heavy load  

**Weighted Proportional Fair (WPF) Scheduler**:  
- Applies fixed weight (v_j = 4 or w_j = 4) to unhappy users  
- Balances QoE improvement with fairness considerations  
- Minimizes impact on non-video traffic  
- More suitable for mixed traffic environments  

The scheduling metric combines conventional PF with QoE weights:  

j* = argmax_j (v_j[n]·w_j[n]·d_j[n]^α / R_j[n])  

Where:  
- d_j[n]: Instantaneous rate for user j at subframe n  
- R_j[n]: Throughput estimate (updated as R_j[n+1] = βR_j[n] + (1-β)d_j[n])  
- α: Fairness parameter (typically α=1)  
- β: Forgetting factor (typically β=0.98)  

### Operational Characteristics  

The system demonstrates several key performance characteristics:  

1) **QoE Improvement**:  
- 30-50% reduction in initial playback delays  
- 0.15-0.44 MOS score improvement (3.5%-12.1%)  
- Higher sustained video resolution quality  

2) **Resource Efficiency**:  
- < 0.05% CPU utilization on Intel i7 processors  
- Memory footprint < 3MB  
- Compatible with existing base station architectures  

3) **Traffic Fairness**:  
- Minimal impact on non-video traffic volume  
- < 30% UPT reduction for worst-case file download users  
- Balanced resource utilization between service classes  

4) **Robustness**:  
- Tolerant to misclassification errors  
- Performance gains persist even at 80% recall/precision  
- Graceful degradation under heavy load  

### Implementation Considerations  

The invention is designed for practical deployment with several implementation optimizations:  

**Feature Extraction**:  
- Uses only standard IP/TCP header fields  
- No payload inspection required  
- Works with encrypted traffic  
- Minimal packet processing overhead  

**Memory Management**:  
- Compact model representations  
- Shared feature buffers between classifiers  
- Efficient sliding window implementation  

**Real-time Operation**:  
- Bounded classification latency  
- Predictable scheduling overhead  
- Support for high flow arrival rates (500+ flows/sec)  

The system has been validated through:  
- Extensive simulation using LTE system models  
- Testing with real video traces (Big Buck Bunny, Elephants Dream)  
- Performance benchmarking against baseline schedulers  

### Alternative Embodiments  

While the preferred embodiment uses Random Forest classifiers, alternative implementations may utilize:  
- Optimized neural networks for classification  
- Different feature sets for state detection  
- Alternative scheduling algorithms incorporating QoE metrics  
- Hybrid approaches combining rule-based and ML classification  

The invention may be extended to:  
- Other video streaming protocols beyond HAS  
- Emerging services like VR/AR streaming  
- Network slicing implementations  
- 5G NR scheduling frameworks  

This detailed description illustrates the innovative aspects and technical advantages of the present invention. The specific implementations and parameters may vary while remaining within the scope of the claimed invention.