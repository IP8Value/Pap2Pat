# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and system for enhancing the Quality of Experience (QoE) of streaming video users in mobile networks. Specifically, the invention pertains to the use of machine learning (ML) classifiers to detect and prioritize streaming video users who are experiencing poor video quality or are awaiting playback, thereby improving their overall streaming experience.

## BACKGROUND

Mobile data traffic continues to grow at a roughly 50% annual rate, primarily driven by the increasing demand for video streaming services. By 2022, streaming video and immersive multimedia traffic are projected to comprise 82% of all mobile data traffic, reaching 225 exabytes per month. This surge in data consumption poses significant challenges for mobile network operators (MNOs) and content providers, who are constrained by limited bandwidth and infrastructure. Some operators offer unlimited streaming video plans, but these often come with de facto limits on streaming video bit rates and differential charging based on the content provider.

To address these challenges, a more holistic, user-experience-based, and end-to-end approach is needed. This approach should include deploying network upgrades (e.g., 5G), bringing video content closer to the end user (e.g., via edge caching), detecting streaming video flows and their underlying application characteristics, and optimizing the scheduling of these flows based on end-user experience. The focus of this invention is on the latter two aspects: detecting unhappy streaming video users and their playback status, and prioritizing their scheduling.

Streaming video delivery is predominantly provided by over-the-top (OTT) services such as Amazon Video, Netflix, and YouTube, which use HTTP Adaptive Streaming (HAS) over the Internet. HAS employs HTTP over Transmission Control Protocol (TCP) for content delivery, but TCP's native rate control mechanisms can lead to increased delay and jitter. Moreover, HAS traffic is typically scheduled as best-effort traffic (Quality of Service (QoS) class identifiers 8 and 9), leading to prolonged playback delay, frequent resolution changes, and rebuffering in congested scenarios. Users experiencing these issues are designated as "unhappy" HAS users.

The key to improving streaming video experience lies in detecting the presence of unhappy users and their playback state, and prioritizing their scheduling. For instance, scheduling a HAS user in the Initial Buffering state more frequently can help them commence video playback sooner, transitioning them to Steady-State where they request new video chunks less often. This, in turn, frees up radio resources for other services and ensures tolerable service degradation.

## SUMMARY

The present invention provides a method and system for enhancing the Quality of Experience (QoE) of streaming video users in mobile networks. The invention utilizes machine learning (ML) classifiers to detect and prioritize streaming video users who are experiencing poor video quality or are awaiting playback. The key aspects of the invention are as follows:

1. **Network Traffic Classification**: The invention employs Random Forest Classifiers (RFCs) to detect the service type of newly detected traffic flows. These classifiers rely on standard IP/TCP headers of downlink packets, ensuring low computational and memory overhead. The classifiers can distinguish between HAS traffic and other types of traffic.

2. **HAS State Detection**: Once HAS traffic is detected, the invention further employs RFCs to determine the playback status (Buffering or Steady-State) and the video resolution quality of the streaming video. These classifiers use temporal statistics derived from downlink packet headers to make their determinations.

3. **Prioritized Scheduling**: The invention integrates the outputs of the ML classifiers into a base station (BS) scheduler. The scheduler prioritizes the scheduling of HAS users based on their detected playback status and video resolution quality. Specifically, HAS users in the Buffering state or those with low video resolution quality are given higher priority to reduce playback delay and improve video quality.

4. **Performance Evaluation**: The invention has been evaluated using both public datasets and experimental testbeds. The ML classifiers achieve high precision and recall scores, and the experience-centric scheduling schemes provide significant improvements in video QoE metrics such as initial playback delay and Mean Opinion Score (MOS).

The invention offers a lightweight and modular solution that can be easily integrated into existing BS implementations, making it a practical and effective approach to enhancing the streaming video experience in mobile networks.

## DETAILED DESCRIPTION

### System Model

The system model consists of one or more base stations (BSs) serving multiple users sharing common radio frequency spectrum. Each user's traffic originates from the internet and is composed of one or more traffic flows. A "flow" is defined as a bi-directional session between two hosts, uniquely identified by the IP five-tuple {protocol (UDP or TCP), client-IP-address, server-IP-address, client-Port, server-Port}. Streaming video users' traffic is hosted at a HAS server, which could be located at a content distribution network or a HTTP proxy server.

#### Primer on HAS

A HAS server stores its video content in multiple encoded bit rates, corresponding to different representation levels, and breaks it into small chunks. Each video chunk corresponds to a few seconds (typically between 1 and 15 seconds) of video playback data. A DASH-compliant server extracts each chunk at runtime based on the client's requested encoding bit rate and transfers the video file to the HAS client sequentially on a chunk-by-chunk basis.

##### Chunk Transfer Mechanism

In a typical HAS session, the client player initiates an HTTP request containing details of the desired video clip for playback. The server transfers to the HAS client a Media Presentation Description (MPD) containing metadata of audio and video representations for the stored video, including available chunk encoding rates. The client player communicates its desired encoding bit rate to the server through a HTTP GET request while requesting a subsequent chunk. The player continuously buffers chunks in its playback buffer; upon buffering sufficient data, it concurrently plays back its video.

##### Adaptive Bit Rate (ABR) Algorithms

The client player employs ABR algorithms to dynamically adapt their encoding bit rate per video chunk. This allows the streaming video encoding rate to adjust to the underlying network conditions, which are affected by various factors such as congestion, channel fading, and interference.

##### HAS Client Player Status

The client player requests new chunks differently depending on the status of its playback buffer. In the Buffering state, the player requests a new chunk as soon as the previous chunk was downloaded, maximizing its chunk request rate to build its playback buffer to a sufficient amount. Subsequently, the player transitions to Steady-State. In Steady-State, the player aims to maintain its playback buffer constant, requesting a new chunk either after a fixed duration or as soon as the previously requested chunk is received.

### Algorithm Flow

The algorithm flow of the experience-centric scheduling solution is as follows:

1. **Traffic Flow Detection**: Upon detecting the arrival of a new traffic flow destined for a particular user, the ML traffic service classifier extracts relevant features from the packets and determines whether the flow is generated from HAS traffic or not.
2. **HAS State Detection**: For detected HAS traffic flows, the ML HAS state classifiers extract features from packet headers and determine the playback status (Buffering or Steady-State) and the video resolution quality at the client player.
3. **User Prioritization**: HAS users detected in the Buffering state or with low video resolution quality are classified as "Unhappy" users. The scheduler prioritizes the radio resource scheduling of Unhappy users. Other users are classified as "Happy" users and receive no such prioritization.

### Machine Learning Traffic Classification

The first step in providing an experience-aware Radio Access Network (RAN) is detecting the type of traffic services and their underlying characteristics at the end-user. This is formulated as a supervised learning problem for classifying the service category per traffic flow and detecting the player status and resolution quality for video flows.

#### Classifier Design

Random Forest Classifiers (RFCs) are used for both classification problems. RFCs are simple ensemble classifiers that train multiple decision trees on various bootstrap samples of the training dataset. The choice of RFCs is motivated by their proven capability in multi-class classification problems and their low computational and memory overhead.

##### Traffic Service Classifier

The traffic service classifier determines the service category of every new traffic flow. It relies on features extracted from the first 5 downlink packet headers of user plane packets belonging to a newly detected traffic flow. The feature set includes:
- Server and Client port numbers (via the TCP Source Port and Destination Port fields)
- Number of packets with PUSH bit set (contained within the TCP Flags field)
- Median IP packet size (derived from the IP Total Length field)

The classifier provides over 95% accuracy for service classification.

##### HAS State Classifier

The HAS state classifier processes packet headers of video flows to detect the client player status (Buffering or Steady-State) and the video resolution quality. The classifier uses temporal statistics derived from downlink packet headers. The feature set is parameterized by the sampling interval duration \( T_w \) and the number of samples \( n \). Each feature sample is a vector of length \( n \) whose entries are obtained by observing user plane packets during the preceding \( n \cdot T_w \) seconds. The classifier uses a sliding window approach to ensure bounded detection latency.

The video resolution classifier extracts features with a longer sampling interval duration, which is an integer multiple of \( T_w \). This ensures that the detection latency is minimized while maintaining high accuracy.

### Evaluation

The performance of the proposed ML traffic classifiers has been evaluated using both public datasets and an experimental testbed. The classifiers achieve high precision and recall scores, demonstrating their effectiveness in real-world scenarios.

#### Public Datasets

- **Traffic Service Classification**: The University of Cambridge dataset is used for traffic service classification. The dataset contains 11 hours of TCP network traffic from 812,000 flows. The RFC achieves a 5-fold cross-validation accuracy of approximately 98%.
- **HAS State Classification**: The dataset published in [33] is used for HAS state classification. It contains packets captured during YouTube video streaming on smartphones. The RFC achieves a 5-fold cross-validation accuracy of 95% for player status classification and 90% for video resolution classification.

#### Experimental Testbed

The classifiers were implemented on a testbed simulating an LTE user plane packet processor. The service classifier achieves a 5-fold cross-validation accuracy of 94%, and the resolution classifier achieves an accuracy of 87% for three resolution classes and 95% for two resolution classes.

### Computation and Memory Requirements

The computational and memory requirements for the classifiers are evaluated to ensure they can be efficiently deployed in practical BS implementations. The proposed traffic profiling solution requires 12.7 million instructions per second (MIPS) and 2.26 MB of storage, equating to approximately 0.05% CPU utilization with an Intel i7 6700K CPU.

### Experience-Centric Traffic Scheduling

The invention adapts the radio resource allocation at a LTE BS scheduler to improve streaming video QoE using the ML classifiers. The scheduler applies traffic-dependent weights on top of the Proportional Fair (PF) scheduling metric to prioritize HAS users based on their user experience.

#### Scheduler Framework

- **Service Category Estimation**: The BS packet processor uses the RFC outputs to provide the scheduler with the estimated service category per traffic flow.
- **Player Status and Resolution Quality Detection**: The scheduler uses the detected player status and video resolution quality to categorize HAS users into Unhappy and Happy states.
- **Weight Assignment**: Unhappy users are assigned higher weights to prioritize their scheduling. The weights are set based on the player state and video resolution quality.

#### Scheduling Schemes

- **Strict Priority (SP) Scheme**: The SP scheduler gives highest priority to scheduling unhappy video users in each subframe by applying prioritization weights of \( w_j = \infty \) or \( v_j = \infty \) on top of their PF metric.
- **Weighted Proportional Fair (WPF) Scheme**: The WPF scheduler applies fixed prioritization weights of \( v_j = 4 \) or \( w_j = 4 \) on top of their PF metric if the user is detected as an unhappy HAS user.
- **Baseline Scheme**: The baseline scheduler implements a PF scheduler that prioritizes all users equally, regardless of whether they are engaged in a HAS session.

### Radio System Simulation

The performance of the experience-aware schedulers is evaluated through radio system simulations. The simulations consider the Baseline, WPF, and SP schedulers and evaluate their impact on video QoE and system performance.

#### Metrics of Interest

- **Traffic Volume**: The total transported data within the simulation time.
- **Initial Playback Delay**: The delay before video playback begins.
- **Video Mean Opinion Score (MOS)**: A subjective measure of video QoE.
- **User Perceived Throughput (UPT)**: The normalized throughput for a user.

#### Physical Layer Modeling

The simulations are conducted in a typical LTE scenario over 10 MHz bandwidth with Time Division Duplex operation. The results are averaged over 16 different drops, each run for 40 seconds.

#### User Activity Model

The simulations model 20 users, with 5 HAS users and 15 File Download users. Each HAS user alternates between watching a HAS video clip and staying idle, ensuring around 70% average resource utilization.

#### HAS Server and Client Player Model

A simple model for a HAS server and client player is implemented. The server loads one chunk worth of video data from stored video traces, and the client player state machine models the Buffering and Steady-State states.

#### Flow Classification Model

The simulations consider both ideal and non-ideal service classification scenarios. In the ideal scenario, the RFC accurately detects the set of HAS users and File Download users. In non-ideal scenarios, the impact of misclassifying HAS users as Download users and vice versa is evaluated.

#### Simulation Results

- **Ideal Flow Classification**: The WPF and SP schemes significantly improve video QoE metrics such as initial playback delay and MOS compared to the baseline scheme.
- **Misclassification Scenarios**: The QoE gains from experience-centric scheduling are sensitive to the video recall score (probability of detecting video flows) but not to the video precision score (fraction of detected video flows that are correctly classified).
- **Traffic Volume**: The experience-optimized scheduling schemes deliver almost identical traffic volume for both HAS and File Download traffic as the baseline scheduling scheme, indicating that the prioritized scheduling of HAS users has a tolerable service-level impact on File Download users.

### Discussion and Future Extensions

The key advantage of the proposed framework is its lightweight and modular nature. The traffic classifier only needs a few downlink packet headers for deriving its features, ensuring low detection latency. The ML classifiers can be easily integrated into existing BS implementations. However, the current framework only detects the service type per flow and differentiates between HAS users and non-HAS users. Future work could extend this framework to passive detection of other service categories such as augmented and virtual reality and derive service-specific QoE information for cellular scheduling.

### Conclusions

The present invention provides an end-to-end experience-centric scheduling framework for improving streaming video quality over cellular radio networks. By inspecting IP/TCP headers of downlink user plane packets, the network can reliably detect the presence of HAS flows and infer their application characteristics. Applying this application layer HAS state information during downlink scheduling of video users significantly improves their QoE. This motivates an end-to-end user application-centric approach towards designing next-generation mobile wireless networks.