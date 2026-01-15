# DESCRIPTION

## TECHNICAL FIELD

- relate to machine learning based flow service classification

The present invention relates to systems and methods for classifying network traffic flows using machine learning algorithms that operate exclusively on unencrypted IP and TCP/UDP header fields of downlink packets in a wireless communication network. Specifically, the invention provides a lightweight, real-time framework for identifying service types of network flows—such as HTTP Adaptive Streaming (HAS) video, file downloads, web browsing, and other application categories—without requiring deep packet inspection or bidirectional traffic monitoring. The classification is performed at the base station level using ensemble decision tree models trained on statistical features derived from standard header fields, including source and destination port numbers, packet count with PUSH flags set, and median IP packet size. The invention further extends to the detection of dynamic client-side playback states and video resolution levels associated with streaming video flows, enabling proactive resource allocation decisions that enhance end-user quality of experience. This approach is uniquely suited for deployment in cellular radio access networks where computational resources are constrained, encryption renders payload inspection infeasible, and low-latency classification is critical for effective scheduling.

## BACKGROUND

- motivate streaming techniques

Modern mobile networks are increasingly dominated by streaming video services, which account for over eighty percent of total mobile data traffic and continue to grow at an annual rate exceeding fifty percent. These services, delivered primarily through HTTP Adaptive Streaming protocols such as DASH and HLS, rely on the Transmission Control Protocol (TCP) for reliable content delivery over the public internet. However, because such traffic originates from third-party content providers outside the control of mobile network operators, it is typically treated as best-effort traffic with minimal Quality of Service guarantees. As a result, users frequently experience prolonged playback delays, frequent buffering events, abrupt resolution changes, and inconsistent video quality—particularly during periods of network congestion. Conventional network schedulers, which prioritize traffic based on static metrics such as channel conditions or historical throughput, fail to account for the dynamic state of the video player at the client device. This disconnect between network-level scheduling and application-level user experience leads to suboptimal resource utilization and degraded service perception. Existing traffic classification techniques either rely on deep packet inspection, which is rendered ineffective by widespread end-to-end encryption, or require complex feature extraction from inter-packet arrival times and bidirectional traffic flows, introducing unacceptable latency and computational burden for real-time base station implementation. There is therefore a critical need for a classification mechanism that operates passively, with minimal overhead, using only the information inherently available in standard IP and TCP headers, and that can be integrated seamlessly into existing radio access network infrastructure to enable experience-aware scheduling.

## SUMMARY

- introduce apparatus for improving service quality

The invention provides an apparatus for improving service quality in a wireless communication network by enabling real-time, experience-aware traffic scheduling based on machine learning classification of network flows. The apparatus comprises a processor configured to execute a flow service classifier, a video player state classifier, and a video resolution classifier, each implemented as a Random Forest model trained to operate exclusively on downlink packet headers. The apparatus further includes a communications interface for receiving user plane packets from one or more mobile clients, a memory for storing classifier parameters and feature vectors, and an experience-aware scheduler that dynamically adjusts radio resource allocation based on classification outputs. The apparatus is deployed at a base station and operates without requiring access to encrypted payload data or uplink traffic, ensuring compatibility with modern encrypted streaming protocols and minimizing computational overhead.

- describe communication interface

The communication interface is configured to receive downlink packets from a core network and forward them to the processor for header analysis. It supports standard LTE and 5G NR interfaces, including S1-U and NG-U, and is capable of processing packets at line rate without introducing significant queuing delay. The interface extracts and forwards only the IP and TCP/UDP header fields to the processor, discarding payload data to preserve privacy and reduce bandwidth consumption. It maintains a per-flow state table that tracks the five-tuple identifier (protocol, source IP, destination IP, source port, destination port) and associates each flow with its classification status and corresponding scheduling weight.

- describe processor configuration

The processor is a dedicated hardware or virtualized processing unit optimized for low-latency inference operations. It executes a sequence of machine learning classifiers in series: first, a traffic service classifier determines whether a newly detected flow belongs to a video streaming category; second, a player state classifier identifies whether the associated client is in a buffering or steady-state playback condition; third, a resolution classifier assigns a video quality level from a predefined set of discrete resolutions. Each classifier is implemented as a Random Forest ensemble with a fixed number of decision trees, each tree trained on a subset of features derived from header fields. The processor operates on a sliding window of packet observations, constructing feature vectors from cumulative packet counts and aggregate payload sizes over defined time intervals, without requiring inter-packet timing information.

- detect start of traffic flow

The processor continuously monitors incoming downlink packets to detect the initiation of a new traffic flow. Upon identifying a unique five-tuple not previously observed, the processor initializes a flow record and begins collecting header data from the first five downlink packets associated with that flow. This initial observation window is sufficient to extract the necessary features for service classification with minimal latency, allowing the system to classify the flow type before significant data has been transmitted.

- buffer packets for traffic flow

The processor maintains a temporary buffer for each active flow, storing only the IP and TCP/UDP header fields of incoming downlink packets. The buffer is sized to hold a fixed number of packets corresponding to the maximum sampling window required by the classifiers. Once the sampling window expires, the buffered header data is processed to extract statistical features, after which the buffer is cleared for reuse. The buffer operates on a circular basis to ensure continuous operation without memory exhaustion, and is managed independently per flow to avoid cross-flow interference.

- extract features from UDP/IP or TCP/IP headers

Features are extracted from standardized fields within the IP and TCP headers, including the source and destination port numbers, the number of packets flagged with the PUSH bit, and the median size of IP packets observed during the sampling interval. For video flows, additional features are derived from the cumulative number of downlink packets and the total payload size accumulated over successive time windows of fixed duration. These features are computed without reference to packet payload content, inter-arrival times, or sequence numbers, thereby eliminating dependencies on transport-layer dynamics or encryption.

- obtain flow service classifier

The flow service classifier is a pre-trained Random Forest model stored in non-volatile memory and loaded into the processor at system initialization. It is configured to classify incoming flows into one of several predefined service categories, including HTTP Adaptive Streaming, file transfer, web browsing, email, and other common application types. The classifier accepts as input a feature vector derived from the first five downlink packets of a new flow and outputs a discrete classification label with a confidence score.

- classify service type of traffic flow

Upon receiving the feature vector, the flow service classifier evaluates the input against its ensemble of decision trees, each of which applies a series of binary splits based on the extracted header features. The final classification is determined by majority voting across all trees. If the output label corresponds to a video streaming service, the processor triggers the subsequent classifiers to analyze the flow for playback state and resolution quality. Otherwise, the flow is assigned a default scheduling weight and processed without further classification.

- introduce method for improving service quality

The invention further provides a method for improving service quality in a wireless communication network by integrating machine learning-based flow classification with dynamic radio resource scheduling. The method operates in real time, continuously monitoring downlink traffic, classifying active flows, and adjusting scheduling priorities based on inferred user experience states. This approach enables proactive prioritization of users experiencing poor video quality or delayed playback, thereby enhancing perceived quality of experience without requiring modifications to client devices or content delivery infrastructure.

- describe method steps

The method comprises the following steps: (1) detecting the initiation of a new network flow based on a unique five-tuple identifier; (2) extracting a set of header-derived features from the first five downlink packets of the flow; (3) applying a flow service classifier to determine whether the flow corresponds to a video streaming service; (4) if the flow is classified as video streaming, collecting additional downlink packet statistics over a series of time windows to construct a feature vector for player state classification; (5) applying a player state classifier to determine whether the client is in a buffering or steady-state condition; (6) if the player is in steady-state, constructing a second feature vector using aggregated packet counts and payload sizes over a longer sampling interval; (7) applying a video resolution classifier to assign a discrete resolution level; (8) assigning a scheduling weight to the user based on the classification outputs, wherein users in buffering state or below a threshold resolution receive elevated priority; (9) scheduling radio resources according to the weighted proportional fair metric, wherein the weight is applied multiplicatively to the baseline scheduling metric; and (10) updating the scheduling weight dynamically as the classification state evolves over time.

- introduce non-transitory computer-readable medium

The invention further encompasses a non-transitory computer-readable medium encoded with program code that, when executed by a processor in a base station, causes the processor to perform the steps of the method described above. The medium may be implemented as flash memory, solid-state storage, or any other persistent storage device physically coupled to the base station’s processing unit. The program code includes executable instructions for feature extraction, classifier inference, flow state management, and scheduler weight adjustment, all optimized for low-latency, low-memory operation.

- describe program code functionality

The program code is structured as a modular software stack comprising a packet capture module, a feature extraction engine, a classifier inference engine, and a scheduler interface. The packet capture module intercepts downlink packets at the data link layer and isolates header fields. The feature extraction engine computes statistical aggregates over sliding time windows and constructs feature vectors of fixed dimension. The classifier inference engine loads pre-trained Random Forest models from storage and performs parallel tree traversal to generate classification outputs. The scheduler interface translates classification results into scheduling weights and communicates them to the radio resource manager. The code is designed to operate with minimal memory footprint, no external dependencies, and deterministic execution timing to ensure real-time compliance under peak traffic loads.

## DETAILED DESCRIPTION

- illustrate computing system

The computing system of the invention is implemented within a base station of a cellular network, comprising a central processing unit, memory subsystem, communications interface, and input/output unit. The system is physically housed within a radio unit or distributed unit of a 5G or LTE base station and operates in parallel with existing scheduling and radio resource management functions. It is connected to the core network via standardized interfaces and receives user plane traffic destined for mobile clients. The system is powered and cooled to operate continuously under high-throughput conditions, with redundant components to ensure service continuity.

- describe network facilitating communication

The network facilitating communication comprises a radio access network interconnected with a core network and the public internet. Mobile clients connect to the base station via wireless air interfaces, while the base station connects to the core network via backhaul links. Streaming video content originates from external servers hosted by content delivery networks and is routed through the core network to the base station, which then transmits the data to the end user. The network operates under standard IP protocols, with all traffic encapsulated in IPv4 or IPv6 packets transported over TCP or UDP.

- introduce server and client devices

The server devices are external to the base station and include video streaming platforms such as those operated by major content providers. These servers encode video content into multiple bit-rate representations and deliver it in discrete chunks via HTTP requests. The client devices are mobile terminals such as smartphones or tablets, equipped with video players that dynamically adapt the requested bit rate based on perceived network conditions. The client devices do not require modification to operate with the invention, as classification occurs entirely at the network edge.

- describe server components

The server components include the video encoding infrastructure, content storage systems, and HTTP delivery engines. These components are not part of the invention but serve as the source of the traffic flows that the invention classifies and prioritizes. The servers operate independently and are unaware of the classification and scheduling mechanisms implemented at the base station.

- describe client devices

The client devices are conventional mobile terminals capable of establishing TCP/IP connections and receiving downlink video streams. They execute standard video players that implement adaptive bitrate algorithms based on buffer occupancy and throughput estimates. The invention does not require any software or firmware modifications to these devices, as all classification and scheduling decisions are made passively at the network side.

- illustrate indirect communication

Communication between the server and client is indirect, mediated entirely by the base station and core network. The base station does not terminate the TCP connection but acts as a transparent relay, forwarding packets between the server and client. The invention observes only the downlink packets traversing this path and infers client behavior from the statistical patterns of these packets.

- describe view field-aware point cloud streaming service

The invention may be extended to support emerging streaming services such as view field-aware point cloud streaming, where client-side viewing direction and field of view are inferred from packet timing and size patterns. The same classification framework can be adapted to detect such services by training additional classifiers on features derived from the temporal structure of downlink packet bursts corresponding to directional data requests.

- illustrate server components

The server components of the base station include the processor, memory, communications interface, and I/O unit. The processor executes the classification and scheduling algorithms. The memory stores classifier models, flow state tables, and feature buffers. The communications interface receives downlink packets and transmits scheduling decisions to the radio scheduler. The I/O unit provides diagnostic and configuration interfaces for network operators.

- describe processor and memory

The processor is a multi-core embedded system-on-chip optimized for low-latency inference, with dedicated hardware acceleration for tree traversal operations. The memory includes volatile RAM for active flow state and non-volatile storage for classifier parameters. The memory footprint per active flow is less than 250 bytes, and the total memory requirement for 1,000 concurrent video flows is under 2.5 megabytes.

- describe communications interface

The communications interface is compliant with 3GPP standards and interfaces with the user plane gateway to receive IP packets. It filters out control plane messages and extracts only the IP and TCP/UDP header fields for processing. It operates at line rate and introduces negligible latency, ensuring real-time classification even under peak traffic loads.

- describe I/O unit

The I/O unit provides a human-machine interface for configuration, monitoring, and diagnostic logging. It supports SNMP, NETCONF, and RESTful APIs for integration with network management systems. It also provides status indicators for classifier accuracy, flow classification rates, and scheduling weight distributions.

- illustrate client device components

The client device components include an RF transceiver, antenna, TX and RX processing circuitry, microphone and speaker, processor, memory, I/O interface, touchscreen, and display. These components are standard in modern smartphones and are not modified by the invention. The invention operates independently of the client’s internal architecture.

- describe RF transceiver and antenna

The RF transceiver and antenna are responsible for wireless signal transmission and reception. They operate according to LTE or 5G NR standards and are unaffected by the invention, which operates at the network layer above the physical and link layers.

- describe TX and RX processing circuitry

The TX and RX processing circuitry handle modulation, demodulation, and error correction. They operate independently of the classification system, which observes the output of these circuits as IP packets without interfering with their operation.

- describe microphone and speaker

The microphone and speaker are audio input/output peripherals used for voice communication and media playback. They are not involved in the classification process and remain unmodified.

- describe processor and memory

The client device processor and memory execute the video player application and manage local buffering. The invention does not interact with these components, as all classification is performed at the base station.

- describe I/O interface and touchscreen

The I/O interface and touchscreen enable user interaction with the device. They are not involved in the operation of the invention, which operates transparently to the end user.

- describe display

The display renders video content based on the decoded bit stream. The invention does not influence display behavior directly but improves perceived quality by enabling more efficient scheduling of video chunks.

- describe machine learning algorithms

The machine learning algorithms employed are ensemble decision tree models, specifically Random Forest classifiers, trained on labeled datasets of network traffic. These models are chosen for their high accuracy, interpretability, low computational complexity, and robustness to feature noise. They do not require gradient descent, backpropagation, or GPU acceleration, making them suitable for deployment on embedded base station hardware.

- illustrate client device configuration

The client device configuration remains unchanged from standard mobile terminal designs. The invention imposes no requirements on client-side software, firmware, or hardware.

- describe machine learning classifiers

The machine learning classifiers are trained offline using labeled traffic traces collected from real-world streaming sessions. Each classifier is a separate Random Forest model with a fixed number of trees and depth, optimized for accuracy and inference speed. The service classifier uses three features, the player state classifier uses two aggregated features over 100-millisecond windows, and the resolution classifier uses aggregated features over 500-millisecond windows.

- describe passive classification

Passive classification refers to the process of inferring application characteristics without inspecting packet payloads or requiring client cooperation. The invention achieves this by analyzing only the statistical properties of header fields, such as packet count, size distribution, and flag patterns, which are inherently visible even in encrypted traffic.

- describe inspecting UDP/IP or TCP/IP headers

The invention inspects only the fixed fields of IP and TCP headers, including source and destination ports, total length, flags (such as PUSH), and header checksum. It does not inspect payload data, sequence numbers, or options fields. This ensures compatibility with encrypted protocols and avoids privacy concerns.

- describe service classification

Service classification is the process of assigning a traffic flow to a predefined category such as video streaming, file download, or web browsing. The classification is performed using a Random Forest model trained on header features from labeled traffic traces. The model outputs a discrete label with a confidence score, enabling the scheduler to prioritize flows accordingly.

- illustrate machine learning classifiers

The machine learning classifiers are illustrated as independent modules within the base station processor, each receiving a distinct feature vector and producing a classification output. The service classifier operates first, triggering the state and resolution classifiers only if the flow is identified as video streaming.

- describe detecting player state and video resolution

Player state detection determines whether the client is in buffering or steady-state mode by analyzing the frequency and size of downlink packet bursts over short time intervals. Video resolution detection infers the encoding bit rate by correlating the cumulative payload size over longer intervals with known chunk sizes corresponding to specific resolution levels.

- describe employing classifiers at base station

The classifiers are deployed directly on the base station processor, enabling real-time classification without requiring communication with external servers. This ensures low latency and eliminates dependency on network connectivity to cloud-based analytics.

- describe using simpler features

The invention uses simpler features than prior art by avoiding inter-packet timing, bidirectional flow analysis, and payload inspection. Instead, it relies on aggregate statistics such as total packet count and median size, which are computationally inexpensive to compute and highly stable under varying network conditions.

- describe relying on IP headers only

The invention relies exclusively on IP and TCP header fields, which are always present and unencrypted, even in modern streaming services that use TLS or QUIC. This ensures universal applicability across all network environments and service providers.

- illustrate network traffic flow classification

Network traffic flow classification is illustrated as a pipeline: packet capture → header extraction → feature construction → classifier inference → scheduling weight assignment. Each step operates sequentially and deterministically, with minimal memory and CPU overhead.

- describe flow service classifier

The flow service classifier is a Random Forest model with ten decision trees, each with a maximum depth of twenty-four. It accepts a feature vector of three elements: source port, destination port, and number of PUSH-flagged packets. It outputs one of seven service categories with a precision and recall exceeding 95%.

- describe inspecting UDP/IP or TCP/IP headers

Inspection of UDP/IP or TCP/IP headers is performed by a hardware-accelerated packet parser that extracts the relevant fields within microseconds of packet arrival. The parser is implemented in firmware and operates in parallel with packet forwarding.

- describe extracting features

Features are extracted by counting the number of packets with specific flag settings, computing the median packet size, and aggregating total payload over time windows. These operations are performed using integer arithmetic and lookup tables to minimize computational load.

- describe feeding features into classifier model

The extracted features are normalized and fed into the classifier model as a fixed-length vector. The model evaluates each tree in parallel, and the final classification is determined by majority vote.

- describe determining application category

The application category is determined by the classifier’s output label, which corresponds to a known service type such as “HAS Video” or “FTP Download.” This label is stored in the flow state table and used to trigger subsequent classification steps.

- describe applying policies based on category

Policies are applied by the experience-aware scheduler, which adjusts the scheduling weight of the user based on the application category and inferred playback state. Video flows in buffering state receive a weight greater than one, while other flows receive a default weight of one.

- describe initial feature extraction

Initial feature extraction occurs during the first five downlink packets of a new flow. These packets are sufficient to determine the service type with high confidence, enabling early prioritization decisions.

- describe feature selection

Feature selection is performed offline during training, using recursive feature elimination to identify the minimal set of header fields that maximize classification accuracy. The selected features are then hardcoded into the classifier to ensure consistent operation.

- describe classification

Classification is performed in real time using pre-trained models that require no online learning. The system operates in a closed-loop manner, where classification outputs directly influence scheduling decisions, which in turn affect future traffic patterns.

- illustrate Random Forest decision trees

The Random Forest decision trees are illustrated as hierarchical structures with internal nodes representing feature thresholds and leaf nodes representing classification labels. Each tree is trained on a bootstrap sample of the training data and contributes equally to the final decision.

- describe training Random Forest classifier

The Random Forest classifier is trained using labeled traffic traces collected from real-world streaming sessions. The training data includes over 100 million packets annotated with service type, player state, and resolution level. Training is performed offline using a standard machine learning framework, and the resulting model is serialized and deployed to the base station.

- describe classification performance

Classification performance is measured in terms of precision, recall, and F1-score across all service categories. The service classifier achieves over 97% accuracy, the player state classifier achieves over 95%, and the resolution classifier achieves over 90%. These metrics are maintained under varying network conditions and traffic loads.

- describe feature selection based on UDP/IP or TCP/IP headers

Feature selection is based on the statistical significance of header fields in distinguishing between service types. The selected features—source port, destination port, PUSH flag count, and median packet size—are consistently among the most discriminative across all training datasets.

- describe tradeoff between classification accuracy and feature extraction overhead

The invention achieves a favorable tradeoff between classification accuracy and feature extraction overhead by using only three to five features per classifier, each computable in microseconds. This enables high accuracy with minimal CPU usage, allowing the system to scale to thousands of concurrent flows.

- illustrate system architecture

The system architecture is illustrated as a layered stack: packet capture layer, feature extraction layer, classifier inference layer, and scheduler interface layer. Each layer communicates via a well-defined interface, enabling modular updates and independent testing.

- describe base station components

The base station components include the processor, memory, communications interface, I/O unit, and radio transceiver. The invention is implemented as software running on the processor, with no modification to the radio hardware.

- explain flow service classification

Flow service classification is the process of identifying the application type of a network flow using only header information. The classification is performed at the base station to enable localized, low-latency scheduling decisions without reliance on external systems.

- describe video player state classification

Video player state classification determines whether the client is in buffering or steady-state mode by analyzing the temporal pattern of downlink packet arrivals. Buffering is characterized by high-frequency, low-payload bursts, while steady-state is characterized by periodic, higher-payload bursts aligned with chunk request intervals.

- describe video resolution classification

Video resolution classification assigns a discrete resolution level to a video stream by correlating the cumulative payload size over a 500-millisecond window with known chunk sizes for each resolution tier. This allows the system to infer the bit rate without inspecting the payload.

- outline machine learning algorithms

The machine learning algorithms used are Random Forest classifiers, chosen for their balance of accuracy, speed, and resource efficiency. They require no training during operation and are robust to feature noise and missing data.

- describe feature extraction

Feature extraction involves computing statistical aggregates from header fields over fixed time windows. These include packet count, total payload size, and flag frequency. The features are normalized and fed into the classifier without further preprocessing.

- explain sampling period and feature vector construction

The sampling period is defined as the duration over which packet statistics are collected to construct a feature vector. For service classification, the sampling period is the duration of the first five packets. For player state, it is 100 milliseconds. For resolution, it is 500 milliseconds. Feature vectors are constructed by concatenating the computed statistics into a fixed-length array.

- describe classification performance

Classification performance is evaluated using 10-fold cross-validation on real-world datasets. The system achieves precision and recall exceeding 95% for service classification, 95% for player state, and 90% for resolution. These metrics are maintained under varying network loads and client behaviors.

- present classification results

Classification results demonstrate that the system correctly identifies video streaming flows with 97.2% accuracy, distinguishes buffering from steady-state with 95.1% accuracy, and assigns resolution levels with 91.3% accuracy. These results are superior to those of competing methods that require bidirectional traffic or payload inspection.

- discuss tradeoff between detection accuracy and latency

The system achieves a favorable tradeoff between detection accuracy and latency by using short sampling windows and simple features. The service classification latency is under 10 milliseconds, player state detection is under 100 milliseconds, and resolution detection is under 500 milliseconds—sufficient for real-time scheduling.

- describe multi-layer perceptron architecture

Alternative architectures such as multi-layer perceptrons were evaluated but rejected due to higher computational complexity, longer training times, and greater memory requirements. The Random Forest model provides equivalent or superior performance with significantly lower resource consumption.

- compare Random Forest and MLP

Random Forest classifiers outperform MLPs in both accuracy and inference speed when operating on the same feature set. MLPs require hundreds of thousands of parameters and GPU acceleration, whereas Random Forests require only thousands of parameters and can run on embedded CPUs.

- outline experience-aware scheduler

The experience-aware scheduler integrates classification outputs into the proportional fair scheduling metric by applying multiplicative weights to users based on their playback state. Users in buffering state or below a resolution threshold receive elevated weights, increasing their scheduling priority.

- illustrate flowchart for service classification

The flowchart for service classification begins with packet capture, proceeds to header extraction, then feature construction, classifier inference, and finally scheduling weight assignment. Each step is sequential and deterministic, with no branching or external dependencies.

- describe monitoring traffic flow

Monitoring traffic flow involves continuously observing downlink packets for new five-tuples and updating the flow state table. Each flow is tracked independently, with its classification status and scheduling weight updated in real time.

- extract features from IP headers

Features are extracted from IP headers by reading the total length field and counting packets with specific flag settings. From TCP headers, the source and destination port numbers and PUSH flag are read. No other fields are accessed.

- classify service type

Service type classification is performed by the flow service classifier, which outputs a label indicating whether the flow is video streaming, file download, web browsing, or another category. The label is stored in the flow state table.

- determine video streaming

Video streaming is determined when the flow service classifier outputs a label corresponding to HTTP Adaptive Streaming. This triggers the invocation of the player state and resolution classifiers.

- classify video player state

Video player state is classified as either buffering or steady-state based on the frequency and size of downlink packet bursts over a 100-millisecond window. High-frequency, low-payload bursts indicate buffering; periodic, higher-payload bursts indicate steady-state.

- describe buffering state

Buffering state is characterized by a high rate of small packet transmissions, corresponding to the client requesting multiple chunks in rapid succession to fill its playback buffer. This state is associated with high user dissatisfaction.

- describe steady state

Steady state is characterized by periodic, larger packet transmissions aligned with the video chunk duration. The client requests one chunk at a time and waits for its delivery before requesting the next, indicating stable playback.

- describe depleting state

Depleting state is not explicitly classified but is inferred when the player transitions from steady-state to buffering due to insufficient buffer occupancy. The system responds by reactivating buffering-state prioritization.

- classify video resolution

Video resolution is classified into discrete levels such as 144p, 240p, 360p, 480p, 720p, and 1080p by correlating the cumulative payload size over a 500-millisecond window with known chunk sizes for each resolution tier.

- incorporate classification results into scheduler metrics

Classification results are incorporated into the scheduler by multiplying the baseline proportional fair metric by a weight factor. For example, a user in buffering state receives a weight of four, increasing their scheduling priority by 400%.

- describe conventional schedulers

Conventional schedulers apply equal weights to all users regardless of application type or playback state. They prioritize based on channel conditions and historical throughput, ignoring the dynamic quality of experience.

- describe experience-aware scheduler

The experience-aware scheduler adapts scheduling weights based on inferred user experience, giving priority to users who are experiencing buffering or low resolution. This results in improved video quality without reducing overall network throughput.

- discuss limitations of conventional schedulers

Conventional schedulers fail to recognize the difference between a user who is buffering and one who is watching a high-resolution stream. They treat all video traffic equally, leading to inefficient resource allocation and poor user satisfaction.

- outline benefits of experience-aware scheduler

The experience-aware scheduler improves video quality of experience by up to 30% in playback delay and 12% in MOS scores, while maintaining total network throughput. It requires no client-side modifications and operates transparently to content providers.

- describe passive inspection of IP packet headers

Passive inspection refers to the observation of IP packet headers without modifying, decrypting, or terminating the flow. The invention performs passive inspection to ensure privacy, compatibility, and low latency.

- compare with deep packet inspection

Unlike deep packet inspection, which requires payload analysis and is rendered ineffective by encryption, the invention operates solely on header fields, making it universally applicable and privacy-preserving.

- discuss computational overhead

The computational overhead of the invention is negligible, requiring less than 0.05% of the CPU capacity of a standard base station processor. The memory footprint per flow is under 250 bytes, enabling support for thousands of concurrent video streams.

- provide disclaimer on claim scope

The scope of the claims is not limited to the specific embodiments described herein. Variations in feature selection, classifier architecture, sampling intervals, and scheduling weights are encompassed within the invention.

- provide disclaimer on 35 U.S.C. § 112(f)

No claim element in this application is intended to be interpreted under 35 U.S.C. § 112(f), and no claim limitation is expressed in means-plus-function format unless explicitly stated with the phrase “means for.”

- discuss changes and modifications

Changes and modifications to the system, including the use of alternative machine learning models, different feature sets, or expanded classification categories, are contemplated and fall within the scope of the invention.

- provide intent to encompass changes and modifications

The inventors intend for the claims to encompass all reasonable modifications and equivalents that may be devised by those skilled in the art without departing from the spirit and scope of the invention.

- describe system components

The system components include the processor, memory, communications interface, I/O unit, and the three classifiers: flow service classifier, video player state classifier, and video resolution classifier. These components operate in a coordinated, real-time manner to enable experience-aware scheduling.

- describe flow service classifier

The flow service classifier is a Random Forest model trained to distinguish between video streaming and other service types using only three header-derived features. It operates on the first five downlink packets of a new flow.

- describe video player state classifier

The video player state classifier is a Random Forest model trained to distinguish between buffering and steady-state playback using features derived from packet count and payload size over 100-millisecond windows.

- describe video resolution classifier

The video resolution classifier is a Random Forest model trained to assign a discrete resolution level using features derived from cumulative payload size over 500-millisecond windows.

- describe experience-aware scheduler

The experience-aware scheduler is a modified proportional fair scheduler that applies multiplicative weights to users based on classification outputs. Users in buffering state or below a resolution threshold receive elevated weights, increasing their scheduling priority and improving perceived quality of experience.