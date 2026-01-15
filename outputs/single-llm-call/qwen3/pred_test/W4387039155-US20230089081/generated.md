# DESCRIPTION

## TECHNICAL FIELD

- relate to electric power grids

The present invention relates to electric power grids, and more specifically to systems and methods for predicting disturbances and impending faults in electrical transmission and distribution networks using machine learning models trained on historical event data derived from existing instrumentation. The invention enables the early detection of incipient faults in power grid infrastructure without requiring the installation of additional sensors or dedicated monitoring hardware. It leverages standard current and voltage measurements collected by existing intelligent electronic devices (IEDs) compliant with IEC 61850 protocols, transforming these measurements into predictive signals that indicate the likelihood of a fault occurring within a defined time horizon. This approach is particularly suited for medium-voltage distribution systems, including both overhead lines and underground or underwater cables, where traditional fault detection methods are either impractical or cost-prohibitive. The invention is applicable to utility-scale power systems, industrial power networks, and smart grid architectures where reliability, operational efficiency, and asset longevity are critical concerns.

## BACKGROUND ART

- introduce smart grids
- discuss big data analytics

Smart grids represent a modern evolution of traditional electrical power systems, integrating digital communication, automation, and real-time data analytics to enhance grid reliability, efficiency, and resilience. These systems rely on widespread deployment of intelligent electronic devices that continuously monitor electrical parameters such as voltage, current, phase angle, and switching status, generating vast volumes of operational data. The advent of big data analytics has enabled utilities to extract meaningful insights from this data, facilitating condition-based maintenance, load forecasting, and anomaly detection. However, existing analytical approaches often focus on post-fault diagnostics or static threshold-based alarm systems that lack predictive capability. While some methods attempt to correlate weather patterns or environmental conditions with fault occurrences, they fail to capture the nuanced, time-dependent electrical signatures that precede mechanical, thermal, or insulation-related failures. Furthermore, many proposed solutions require additional sensor infrastructure, high-bandwidth communication channels, and substantial computational resources, making them unsuitable for widespread deployment in aging or resource-constrained grid segments. The present invention addresses these limitations by introducing a data-efficient, sensor-agnostic framework that derives predictive power from the intrinsic temporal patterns embedded within existing disturbance recordings, thereby enabling proactive grid management without infrastructure overhauls.

## SUMMARY

- introduce fault prediction method
- receive events with time information
- input events to trained predictor
- output predictions for disturbance events
- display predictions for fault detection
- display time windows of different lengths
- indicate occurrence time of disturbance event
- maintain event categories
- determine event category
- input event category to trained predictor
- maintain event weight values
- determine event weight value
- input event weight value to trained predictor
- store events with time information
- input time information to trained predictor
- perform inputting periodically
- introduce event pattern extraction
- acquire event history data
- extract event patterns from history data
- input event patterns to machine learning model
- train model to predict disturbance events
- store trained model as predictor
- filter event patterns for testing/commissioning
- include event categories in event pattern
- include weight values in event pattern
- filter events with low weight values
- introduce apparatus for fault prediction

A fault prediction method is disclosed for electric power grids, comprising the reception of disturbance events each associated with precise time information, wherein each event is characterized by a set of derived electrical features extracted from voltage and current waveforms recorded by existing grid instrumentation. These events are input to a trained machine learning-based predictor that outputs a probability score indicating the likelihood of a disturbance event occurring within a predefined future time horizon. The output is displayed to grid operators as a temporal forecast, enabling proactive intervention prior to actual failure. The method further includes the dynamic adjustment of prediction time windows, allowing for forecasts over intervals ranging from hours to days, and the explicit indication of the estimated time of disturbance occurrence relative to the prediction timestamp. Event categories are maintained to classify disturbances by origin, such as transient overvoltages, load imbalances, or insulation degradation signatures, and each event is assigned a weight value reflecting its relevance to fault prediction. Both the event category and its associated weight value are independently input to the trained predictor to refine predictive accuracy. Events are stored in a structured database with associated timestamps, and the predictor is periodically updated with new event data to maintain model fidelity. Event pattern extraction is performed by acquiring historical event data over extended periods, segmenting the data into temporal windows, and deriving multivariate feature vectors that encapsulate the evolution of electrical conditions preceding known faults. These extracted event patterns are then used to train a long short-term memory neural network to recognize patterns indicative of impending failure. The trained model is stored as a predictive engine and may be selectively filtered for commissioning or testing purposes to exclude low-relevance events. Event categories and weight values are incorporated into the training patterns to enhance discriminative power, and events with weight values below a predefined threshold are excluded from training and prediction cycles. An apparatus for fault prediction is further disclosed, comprising a computing system configured to execute the method, including data acquisition, pattern extraction, model training, and real-time prediction components, all operable within existing grid communication and control infrastructures.

## DETAILED DESCRIPTION OF SOME EMBODIMENTS

- introduce embodiments
- clarify meaning of "an", "one", or "some" embodiment(s)
- describe single units, models, devices, and memory
- explain cloud computing and virtualization
- illustrate system architecture in FIG. 1
- describe logical connections in FIG. 1
- introduce Industrial Internet of Things (IIoT) and smart grid
- describe system 100 components
- introduce power distribution grid 110
- describe intelligent electronic devices (IEDs) 111
- explain IED functions and event generation
- describe event data and time information
- introduce cloud platform 102
- describe data storage 121 and historic event data
- illustrate historic event data structure
- describe event types and additional information
- explain data storage implementation
- introduce control center 130
- describe computing apparatus 132 and user interface 131
- introduce predictor unit 133
- describe predictor unit output and display
- explain disturbance events and evolving faults
- describe warnings and notifications
- introduce offline equipment (apparatus) 140
- describe trainer unit 116
- illustrate information flow in FIG. 2
- describe predictor unit input data and processing
- explain event categorization and weight values
- describe trained machine learning based model 133-1
- illustrate flow chart in FIG. 3
- describe event reception and prediction
- illustrate near real-time predictions in FIGS. 4 and 5
- describe event buffering and periodic input
- explain time interval monitoring and input
- describe output processing and display
- illustrate information flow in FIG. 6
- describe trainer unit operation and machine learning based model training
- acquire event patterns
- perform windowing
- input event patterns to machine learning model
- train machine learning model
- store trained model
- download trained model to predictor unit
- retrain model periodically
- illustrate training functionalities
- acquire event history data
- extract event patterns
- input event patterns to machine learning model
- train machine learning model
- store trained model
- associate events with weight values and categories
- determine weight values and categories
- include weight values and categories in training data
- filter irrelevant event patterns
- filter irrelevant events
- perform online training
- implement functionalities in apparatus
- describe apparatus components
- illustrate apparatus block diagram
- describe interface entities
- describe processing entities
- describe memory
- store algorithms in memory
- describe computer program code
- store computer program code in memory
- describe carrier of computer program
- configure predictor unit and trainer unit
- describe computer or processor
- describe microprocessor
- describe chipset
- describe logic gates
- describe computer processors
- describe application-specific integrated circuits
- describe field-programmable gate arrays

Embodiments of the invention are described herein to illustrate its practical implementation and operational scope, and it is understood that the terms “an,” “one,” or “some” embodiment refer to one or more non-exclusive implementations, not limiting the invention to any single configuration. The invention may be embodied as a single computational unit, a distributed model, a networked device, or a memory-stored algorithm executable on general-purpose or specialized hardware. Cloud computing and virtualization technologies are employed to enable scalable deployment, allowing the training and inference components to operate independently on separate virtualized environments while maintaining secure, low-latency data exchange. The system architecture, as illustrated in FIG. 1, comprises a power distribution grid 110 equipped with intelligent electronic devices (IEDs) 111 that continuously monitor voltage and current waveforms at sampling rates exceeding 4 kHz. These IEDs generate disturbance events in response to predefined thresholds or algorithmic triggers, each event being timestamped and encoded with metadata including phase information, magnitude, duration, and harmonic content. The event data is transmitted via IEC 61850-compliant protocols to a cloud platform 102, where historic event data is stored in a structured database 121 comprising fields for event ID, timestamp, feature vector, category label, and weight value. Each event is categorized based on its waveform morphology and contextual origin, and a weight value is computed to reflect its statistical significance in predicting future faults. The control center 130 includes a computing apparatus 132 interfaced with a user interface 131 and a predictor unit 133, which receives buffered event streams and outputs probabilistic forecasts displayed as time-series graphs indicating rising risk levels preceding known fault events. Disturbance events are recognized as precursors to evolving faults, and the system generates automated warnings and notifications to grid operators prior to equipment failure. An offline apparatus 140, comprising a trainer unit 116, periodically acquires historical event data from the cloud, extracts temporal patterns through sliding windowing techniques, and trains a long short-term memory neural network model 133-1 to map historical sequences to binary fault outcomes. The trained model is then downloaded to the predictor unit for real-time inference. Information flow between components is illustrated in FIG. 2, wherein event categorization and weight assignment occur prior to model input, ensuring that only high-value patterns influence prediction. The predictor unit processes incoming events periodically, buffering them over time intervals of up to one week, and inputs the aggregated feature vectors into the trained model to generate a probability score between zero and one. Near real-time predictions are visualized in FIGS. 4 and 5, demonstrating a consistent rise in predicted probability preceding actual fault events. Output processing includes thresholding, temporal smoothing, and alert generation, with displays showing both current risk levels and projected fault timelines. The trainer unit operates independently, acquiring event history, extracting patterns, and retraining the model on a weekly or monthly basis to adapt to changing grid conditions. Training functionalities include the inclusion of event categories and weight values in the input sequences, filtering out low-weight events to reduce noise, and performing online training when sufficient new data accumulates. The apparatus implementing the method comprises interface entities for data acquisition and user interaction, processing entities for feature extraction and model inference, and memory for storing algorithms, trained models, and computer program code. The computer program code is stored in non-transitory memory and may be distributed via tangible carriers such as optical discs or secure digital downloads. The predictor unit and trainer unit are configurable using general-purpose computers, microprocessors, chipsets, logic gates, application-specific integrated circuits, or field-programmable gate arrays, enabling deployment across diverse hardware platforms from edge devices to cloud servers.