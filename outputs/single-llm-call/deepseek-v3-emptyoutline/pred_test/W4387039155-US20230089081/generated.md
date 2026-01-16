Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates generally to electrical power systems, and more particularly to systems and methods for predicting faults in power transmission and distribution networks. The invention utilizes machine learning techniques to analyze existing current and voltage measurements from power lines, enabling the prediction of potential faults up to one week in advance without requiring additional sensor infrastructure. The disclosed system processes disturbance recordings through advanced feature engineering and long short-term memory (LSTM) neural networks to generate accurate fault probability assessments, significantly improving grid reliability and reducing operational costs associated with unplanned outages.  

## BACKGROUND ART  

Electrical power systems are increasingly complex networks spanning generation, transmission, distribution, and consumption systems. This complexity leads to a higher frequency of faults, primarily manifesting as partial or complete short circuits between power lines or to ground. Conventional approaches to fault prediction have significant limitations that the present invention addresses.  

Existing methods for fault prediction include analysis of partial discharges, which requires expensive measurement equipment and performs poorly in noisy environments or with underground/underwater cables. Other approaches utilize temperature sensors or unmanned aerial vehicle monitoring, but these cannot detect all fault types and require substantial additional infrastructure investment. These conventional methods share several disadvantages: high implementation costs, limited detection capabilities for certain fault types, inability to work with existing sensor infrastructure, and poor performance in challenging environments.  

The present invention overcomes these limitations by utilizing only existing current and voltage measurements through a novel machine learning pipeline. This approach eliminates the need for additional sensors while providing comprehensive fault prediction capabilities across all power line types and configurations. Prior attempts at data-driven fault prediction have failed to achieve the accuracy and lead time demonstrated by the current invention, particularly in processing irregularly sampled time series data from disturbance recordings.  

## SUMMARY  

The invention provides a computer-implemented method for predicting power line faults comprising: receiving time-series measurements of current and voltage from power lines; processing the measurements through a feature engineering pipeline to extract relevant electrical characteristics; filtering the processed data to identify anomaly recordings relevant for fault prediction; analyzing the filtered data through an LSTM neural network architecture; and generating fault probability predictions with up to one week lead time.  

Key innovations include:  
1. A multi-stage feature extraction process that transforms high-dimensional waveform data into approximately 300 descriptive scalar values while preserving critical fault-indicative information  
2. A neural network filtering component that automatically classifies recordings as relevant or irrelevant for fault prediction  
3. An LSTM-based time series analysis architecture specifically optimized for irregularly sampled power system data  
4. A data augmentation system that multiplies training samples through phase permutation techniques  
5. Integration with existing substation measurement infrastructure using IEC 61850 protocols  

The system achieves superior performance metrics compared to conventional approaches, with demonstrated specificity of 0.9127 and recall of 0.6694 in experimental implementations. The invention provides utilities and industrial power system operators with an accurate, cost-effective solution for proactive grid maintenance and outage prevention.  

## DETAILED DESCRIPTION OF SOME EMBODIMENTS  

The fault prediction system operates through several integrated components that process power line measurements into actionable predictions. Figure 1 illustrates the overall system architecture, which will be described in detail below.  

**Data Acquisition and Preprocessing**  
The system utilizes existing instrument transformers and I/O connections at substations to collect current and voltage measurements without additional sensor deployment. Data is transmitted using IEC 61850-9-2LE and IEC 61850 8-1 protocols to edge computing devices that perform initial processing. These edge devices execute high-sensitivity protection algorithms that trigger disturbance recordings when anomalies are detected, significantly reducing data volume compared to continuous cloud streaming.  

Recordings comprise voltage and current waveforms sampled at 4 kHz, capturing detailed transient information while maintaining manageable data sizes. The edge devices upload these disturbance recordings to cloud-based storage for subsequent machine learning processing. This hybrid edge-cloud architecture optimizes computational resource usage while maintaining access to historical data for model training and validation.  

**Feature Engineering Pipeline**  
The high-dimensional waveform data undergoes extensive feature extraction to create compact yet informative representations for machine learning processing. The system calculates multiple electrical characteristics from each disturbance recording, including:  
- Root mean square (RMS) values  
- Impedance measurements  
- Active and reactive power  
- Harmonic components  
- Phase angle relationships  

From these derived signals, the system computes statistical descriptors (maximum, minimum, standard deviation, etc.) to capture the essential information in approximately 300 scalar values per recording. This feature vector serves as a lower-dimensional input to the machine learning model while preserving the critical fault-predictive information contained in the original waveforms. Figure 3 illustrates the complete feature extraction process.  

**Machine Learning Architecture**  
The prediction model processes feature vectors arranged as multidimensional time series with dimensions corresponding to the number of recordings and feature values. The model architecture comprises three principal components:  

1. **Filtering Network**: A fully connected neural network that classifies each recording as relevant or irrelevant for fault prediction. This component is trained on manually labeled samples to distinguish between genuine anomalies and normal system variations, significantly reducing noise in the input data.  

2. **LSTM Layer**: A long short-term memory recurrent neural network that processes the filtered time series data. The LSTM architecture is particularly suited for irregularly sampled time series due to its selective memory mechanisms, which maintain important temporal relationships while ignoring irrelevant variations. The LSTM outputs a consolidated feature vector representing the processed time window.  

3. **Classification Head**: Two fully connected layers with sigmoid activation that convert the LSTM output into a fault probability prediction between 0 and 1. The prediction represents the probability of a fault occurring within the subsequent one-week period.  

The complete model architecture is shown in Figure 5. For systems with multiple measurement locations, the network processes each location's data separately before concatenating the LSTM outputs for final classification, enabling comprehensive analysis of interconnected grid segments.  

**Training and Optimization**  
The system employs several innovative techniques to overcome data scarcity challenges:  

- **Phase Permutation Augmentation**: The training dataset is expanded sixfold by systematically permuting phase assignments (e.g., treating phase 1 as phase 2) while maintaining correct electrical relationships. This augmentation significantly improves model generalization without requiring additional physical measurements.  

- **Hyperparameter Optimization**: Critical parameters including learning rate (optimally 0.00003) and learning rate decay (optimally 0.05) are carefully tuned to balance recall (0.6694) and specificity (0.9127) performance metrics.  

- **Cross-Validation**: The model is evaluated using 5-fold cross-validation to ensure robust performance across different data segments and operating conditions.  

**Operational Implementation**  
In practical deployment, the system makes hourly predictions by analyzing a sliding one-week window of historical data. When the fault probability exceeds a configurable threshold (typically 0.5), operators receive alerts with sufficient lead time for preventive maintenance. Figure 6 demonstrates the system's predictive capability, showing rising probability signals preceding actual faults.  

The invention's technical advantages include:  
- Utilization of existing measurement infrastructure without additional sensor costs  
- Comprehensive fault prediction across overhead, underground, and underwater power lines  
- One-week prediction horizon enabling proactive maintenance planning  
- Computational efficiency allowing deployment on standard hardware  
- Adaptability to diverse grid configurations through modular architecture  

While particular embodiments have been described, the invention encompasses various modifications and alternative implementations within the scope of the claims. The examples provided illustrate the principles of the invention and its practical applications, enabling others skilled in the art to utilize and adapt the invention for specific power system requirements.