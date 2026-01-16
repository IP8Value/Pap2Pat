# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of electrical power systems, particularly to methods and systems for predicting faults in power lines. More specifically, the invention provides a machine learning-based approach for predicting faults in transmission and distribution power lines up to a week before they occur, utilizing existing measurements of current and voltage without requiring the installation of additional sensors.

## BACKGROUND ART

Electrical power systems are increasingly complex and widespread, encompassing generation, transmission, distribution, and load systems. The frequency of faults in these systems, such as partial or complete short circuits of power lines to the ground or among themselves, is on the rise. These faults can lead to significant financial losses and a decrease in the reliability of the electrical system. Utilities and large industrial plants, which often have extensive power line systems, are particularly prone to faults due to various factors, including aging and wear and tear of power lines, unsuitable materials, mechanical failure, degradation of insulation, moisture build-up, electrical overloading, interference from animals or vegetation, and extreme environmental conditions.

Early fault prediction is crucial for grid operators to address potential issues before they escalate into failures. This proactive approach can enhance the overall reliability of the grid, reduce operational costs, and minimize revenue loss while ensuring continuous power delivery to end users. Prior to a fault, the grid often exhibits precursor symptoms, which can be detected using appropriate models.

Several methods have been explored for predicting and analyzing grid faults. One well-known approach involves analyzing partial discharges, which can indicate insulation degradation. However, this method requires expensive measurement tools and is challenging in noisy environments, making it less feasible for large-scale grids and underground or underwater cables. Other methods, such as using temperature sensors on power lines and monitoring with unmanned aerial vehicles, have also been investigated. These methods, however, have limitations in detecting faults in underground and underwater cables and are often limited in their ability to identify faults from various root causes. Additionally, these approaches necessitate the deployment of additional sensors, which can be costly and resource-intensive.

The present invention addresses these limitations by providing a machine learning-based approach that predicts faults in power lines up to a week before they occur, using a purely data-driven method that leverages existing measurements of current and voltage without the need for additional sensors.

## SUMMARY

The present invention provides a method and system for predicting faults in transmission and distribution power lines using a machine learning-based approach. The method includes collecting current and voltage measurements from power lines, performing feature engineering to extract relevant features from the measurements, and using a long short-term memory (LSTM) based deep neural network to predict faults up to a week before they occur. The system utilizes existing measurements and does not require the installation of additional sensors, making it cost-effective and efficient.

The method comprises the following steps:
1. Collecting current and voltage measurements from power lines using existing instrument transformers and I/O connections.
2. Pre-processing the collected data to extract derived features such as RMS, impedance, active and reactive power, harmonics, and phase angles.
3. Arranging the extracted features into a multidimensional time series and using this time series as input to an LSTM-based deep neural network.
4. Training the LSTM-based model to predict faults based on the input time series.
5. Deploying the trained model to make real-time predictions and alert grid operators of potential faults.

The system includes an edge computing device for collecting and pre-processing data, a cloud-based service for storing and transferring data, and a dedicated machine learning service for performing the predictions. The edge device runs high-sensitivity protection-related algorithms to trigger the recording of disturbances, which are then uploaded to the cloud for further processing.

The invention offers several advantages over existing methods, including:
- Early fault prediction up to a week before occurrence.
- Utilization of existing measurements without additional sensors.
- Cost-effective and efficient use of computational resources.
- High specificity and recall in fault prediction, reducing false positives and improving reliability.

## DETAILED DESCRIPTION OF SOME EMBODIMENTS

### Data Collection

The data used for fault prediction is collected from power lines using existing instrument transformers and I/O connections, which are typically part of the protection and control devices in the power system. Current and voltage measurements, along with I/O statuses of primary equipment, are shared using standardized protocols such as IEC 61850-9-2LE and IEC 61850 8-1. An edge computing device collects this data and runs multiple high-sensitivity protection-related algorithms to trigger the recording of disturbances. These disturbance recordings, which include voltage and current waveforms sampled at 4 kHz, are then uploaded to a cloud-based service for monitoring and storage.

### Feature Engineering

The waveform recordings are high-dimensional and contain a large amount of information. To make the data more manageable and relevant for the machine learning model, feature engineering is performed. Each disturbance recording is processed to extract several derived features, such as RMS, impedance, active and reactive power, harmonics, and phase angles. From these signals, representative scalar values are calculated, including maximum, minimum, and standard deviation, to capture as much information as possible with a minimal number of values. This process results in a feature vector with approximately 300 values, serving as a lower-dimensional representation of the waveform recording.

### Machine Learning Model

The feature vectors are arranged to create a multidimensional time series with dimensions (# recordings, # feature values). These time series are then used as input to the forecasting model. The model consists of three main components: a filtering component, an LSTM layer, and a classification head.

1. **Filtering Component**: This is a fully connected neural network that classifies each recording in the input time series as relevant or not relevant to the forecasting process. The filtering component is trained on a subset of manually labeled recordings to distinguish between relevant and irrelevant data. Only relevant recordings are passed on to the next stage of processing.

2. **LSTM Layer**: The LSTM layer is a type of recurrent neural network well-suited for handling irregularly sampled time series data. It processes the input time series and generates a single feature vector. The LSTM layer's ability to selectively remember or forget certain pieces of information from the input sequence makes it effective for fault prediction.

3. **Classification Head**: The output from the LSTM layer is processed through a classification head consisting of two fully connected layers and projected with a sigmoid activation function to a value between 0 and 1, representing the probability of a fault occurring within the next week. If data from multiple connected locations is available, the network can process a time series for each location and concatenate the outputs from the LSTM layer before passing them to the classification head.

### Data Augmentation

To address the limitation of having a small amount of data, various augmentations are applied to the data before training. One significant augmentation involves shuffling the input data such that the three phases change order. This effectively increases the number of samples by six times when all permutations are used, enhancing the model's ability to generalize.

### Model Training and Evaluation

The model is trained using a combination of a training set and a test set, with 5-fold cross-validation to assess its ability to generalize to unseen data. The performance of the model is evaluated using standard metrics such as recall and specificity. Recall measures the proportion of positive samples that the model correctly identifies, while specificity reflects the proportion of negative samples that the model accurately predicts. The goal is to achieve high specificity to avoid unnecessary costs from false positives.

### Results

The results of the model show that it can predict faults up to a week before they occur with a high degree of accuracy. The best recall of 0.6694 was achieved with a learning rate of 0.00003 and a learning rate decay of 0.05, while the highest specificity of 0.9127 was obtained with a learning rate of 0.0003 and a learning rate decay of 0.05. The model's output is visualized in a graph, showing the rising and falling trends before and after faults.

### Comparison to Baseline

A baseline method was developed for comparison, which uses the frequency of recorded anomalies to predict faults. When adjusted to achieve a specificity of 0.9127, the baseline method had a recall of 0.0889, significantly lower than the proposed method's recall of 0.4367 at the same specificity. Adjusting the threshold to achieve a recall of 0.6694 resulted in a specificity of 0.3640, again lower than that of the proposed method.

### Discussion

#### Feature Engineering

Feature engineering plays a crucial role in the effectiveness of the model. By extracting key properties from the waveforms, the model can focus on the most relevant information, making it easier to learn and more data-efficient. Overdoing feature engineering can result in the loss of essential information, as seen in the poor performance of the baseline model. Careful development of appropriate feature engineering is therefore essential.

#### Importance of Filtering

The filtering of recordings is important to exclude unnecessary noise and focus the model's attention on the most relevant information. While a more complex model might be able to ignore certain information, this would come at the cost of increased computational expense and data requirements. The filtering component helps maintain the model's efficiency and effectiveness.

### Conclusion

The present invention provides a novel and effective method and system for predicting faults in power lines using a machine learning-based approach. By leveraging existing measurements and performing advanced feature engineering and filtering, the invention achieves high accuracy in fault prediction, helping grid operators to proactively address potential issues and enhance the reliability of the electrical system. The invention is cost-effective, efficient, and adaptable to various computing environments, making it a valuable tool for modern power grid management.