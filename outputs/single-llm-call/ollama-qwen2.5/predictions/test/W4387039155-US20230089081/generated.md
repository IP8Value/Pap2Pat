# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to electric power grids. Specifically, it addresses systems and methods for predicting faults in smart grids using big data analytics and machine learning techniques.

## BACKGROUND ART

Smart grids are advanced electrical distribution networks that integrate various technologies to improve efficiency, reliability, and sustainability. These grids utilize sensors, communication infrastructure, and data analytics to monitor and manage the flow of electricity. However, one significant challenge is the prediction and prevention of faults, which can lead to power outages and other issues. Traditional methods for fault detection are often reactive and may not provide timely warnings. With the advent of big data and machine learning, there is a growing interest in developing predictive models that can anticipate potential failures before they occur.

## SUMMARY OF THE INVENTION

The present invention provides a system and method for predicting faults in an electric power grid using historical disturbance recordings and machine learning techniques. The system includes sensors for collecting real-time data from the grid, a data processing unit for feature extraction, and a neural network model trained to predict fault occurrences based on the extracted features. The method involves preprocessing the raw waveform data, extracting relevant features, filtering out noise, and using an LSTM-based model to forecast potential faults up to one week in advance.

## BRIEF DESCRIPTION OF THE DRAWINGS

- Figure 1: Schematic diagram of the system architecture.
- Figure 2: Flowchart of the method for fault prediction.
- Figure 3: Feature extraction pipeline.
- Figure 4: Input data arrangement for the LSTM model.
- Figure 5: Architecture of the neural network model.
- Figure 6: Model output visualization during a short time period.

## DETAILED DESCRIPTION OF THE INVENTION

### System Architecture

The system includes several key components:
1. **Sensors**: Deployed throughout the power grid to collect real-time data such as voltage, current, and temperature.
2. **Data Collection Unit**: Aggregates data from sensors and stores it in a database.
3. **Data Processing Unit**: Preprocesses raw waveform data, performs feature extraction, and filters out irrelevant recordings.
4. **Machine Learning Model**: An LSTM-based neural network trained to predict faults based on the processed data.
5. **User Interface**: Provides visualizations and alerts for potential fault occurrences.

### Method for Fault Prediction

1. **Data Collection**: Raw waveform data is collected from sensors at regular intervals.
2. **Preprocessing**: The raw data is cleaned to remove noise and outliers.
3. **Feature Extraction**: Key features are extracted from the waveforms using techniques such as Fourier transforms, wavelet transforms, and statistical measures.
4. **Filtering**: A filtering component classifies each recording as relevant or irrelevant to the forecasting process.
5. **Model Training**: The LSTM model is trained on a dataset of historical recordings labeled with fault occurrences.
6. **Prediction**: The model processes new data in real-time and outputs the probability of a fault occurring within one week.

### Feature Engineering

Feature engineering serves two important purposes: it highlights the most relevant parts of recorded anomalies and reduces the dimensionality of the input data. Without feature extraction, each data point would be high-dimensional, making it more difficult for the model to learn. The extracted features are designed to capture the essential characteristics of the waveforms that are indicative of potential faults.

### Importance of Filtering

Filtering recordings is crucial to focus the model's attention on the most relevant information and avoid unnecessary noise. A filtering component classifies each recording as relevant or irrelevant, ensuring that only pertinent data is used for prediction. This step improves the model's performance by reducing computational complexity and enhancing its ability to generalize.

### Machine Learning Model

The forecasting model consists of three main components:
1. **Filtering Component**: A fully connected neural network that classifies recordings as relevant or not.
2. **LSTM Layer**: Processes the input time series and generates a single feature vector.
3. **Classification Head**: Projects the output to a value between 0 and 1, representing the probability of a fault occurring within one week.

### Evaluation Methodology

The model's performance is evaluated using standard metrics such as recall and specificity. Predictions are made once each hour, and the data is split into training and test sets for validation. Cross-validation techniques are used to ensure the model's ability to generalize to unseen data.

### Results and Discussion

The results show that the LSTM-based model can successfully predict faults based on disturbance recordings. Hyperparameter tuning significantly affects performance, with optimal settings achieving high recall and specificity. The model outperforms baseline methods, demonstrating its effectiveness in fault prediction for power grids.

## CONCLUSIONS

In conclusion, the present invention provides a robust system and method for predicting faults in electric power grids using machine learning. By leveraging historical data and advanced feature engineering techniques, the LSTM-based model can forecast potential faults up to one week in advance, enabling proactive maintenance and reducing the risk of power outages.