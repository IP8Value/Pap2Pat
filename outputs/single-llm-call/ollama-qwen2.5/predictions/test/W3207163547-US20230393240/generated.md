# Patent Application

**Title:** System and Method for People Counting Using Label-Aware Ranked Loss in a Vehicle Cabin

## Background

### Field of Invention

The present invention relates generally to the field of people counting systems using radar sensors, and more particularly to a method and system that employs a label-aware ranked loss (LAR) function to improve prediction accuracy in a vehicle cabin environment.

### Description of Related Art

Existing methods for people counting often rely on visual sensors or simplistic machine learning models. However, these approaches can be limited by occlusions, lighting conditions, and computational complexity. Radar sensors offer a robust alternative due to their ability to penetrate non-metallic materials and operate effectively in various environmental conditions. Nevertheless, the accuracy of radar-based systems is often compromised by suboptimal loss functions used during training. This invention addresses this gap by introducing a novel LAR loss function that enhances the prediction capabilities of people counting models.

## Summary of the Invention

### Overview

The present invention provides a system and method for accurately counting people in a vehicle cabin using a radar sensor and a deep learning model trained with a label-aware ranked (LAR) loss function. The LAR loss function ensures that the embedding space is ordered according to the number of people, thereby improving the predictive performance of the model.

### Key Features

- **Radar Sensor Integration:** Utilizes an Infineon XENSIV 60 GHz radar sensor mounted on the internal-upper side of the front window of the vehicle cabin.
- **Data Preprocessing:** Processes raw radar data to generate frames suitable for input into a deep learning model.
- **LAR Loss Function:** A novel loss function that ranks embeddings according to the number of people and ensures uniform angles between different labels, enhancing prediction accuracy.
- **Temporal Smoothing:** Applies exponential smoothing (ES) to stabilize the output of the model during inference.

## Detailed Description

### Radar Sensor Integration

One Infineon XENSIV 60 GHz radar sensor is utilized for capturing data inside the vehicle cabin. The sensor is positioned on the internal-upper side of the front window, providing a comprehensive view of the interior space. This strategic placement ensures that the radar can effectively detect and track individuals within the cabin.

### Data Preprocessing

The raw radar data is preprocessed to generate frames suitable for input into the deep learning model. Each frame represents a snapshot of the vehicle cabin at a specific point in time. The preprocessing steps include filtering, normalization, and dimensionality reduction to ensure that the data is clean and consistent. This results in 95,000 frames of scenes with zero to five people, divided into recordings with an average length of 350 frames.

### LAR Loss Function

The label-aware ranked (LAR) loss function is a novel approach designed to improve the predictive performance of deep learning models for people counting. The LAR loss ensures that the embedding space is ordered according to the number of people and that embeddings of different labels are separated by uniform angles. This property enhances the model's ability to distinguish between different numbers of people, even in challenging scenarios.

The LAR loss function is defined as follows:

\[
L_{\text{LAR}} = \sum_{i=1}^{N} \sum_{j=1, j \neq i}^{N} \log(\Delta l) \cdot (1 - f_i^T f_j)
\]

where \( N \) is the number of samples, \( f_i \) and \( f_j \) are the normalized feature vectors, and \( \Delta l = |l_a - l_n| \) is the difference between the labels of the anchor and negative samples. The logarithm function ensures numerical stability and assigns smaller values to neighboring labels.

### Temporal Smoothing

To further stabilize the output of the model during inference, exponential smoothing (ES) is applied. ES helps to reduce noise and smooth out fluctuations in the predictions, making the system more reliable in real-world scenarios. The ES formula used is:

\[
y_t = \alpha x_t + (1 - \alpha) y_{t-1}
\]

where \( y_t \) is the smoothed output at time \( t \), \( x_t \) is the raw prediction, and \( \alpha \) is the smoothing factor.

### Training and Evaluation

The deep learning model used for people counting is a convolutional neural network (CNN) with three convolutional layers, each followed by a ReLU activation function and a pooling layer. The final layer is a fully connected ReLU layer that outputs the predicted number of people. The model is trained using the LAR loss function on a dataset split into 76,000 training frames and 19,000 testing frames.

The performance of the model is evaluated using accuracy and accuracy +/-1 metrics. The results show that the proposed method achieves an accuracy of 83.0% and an accuracy +/-1 of 99.9%, representing a significant improvement over existing methods.

## Conclusion

The present invention provides a robust and accurate system for people counting in vehicle cabins using a radar sensor and a deep learning model trained with a label-aware ranked (LAR) loss function. The LAR loss ensures that the embedding space is ordered according to the number of people, enhancing the predictive performance of the model. Temporal smoothing further stabilizes the output during inference, making the system reliable in real-world applications.

## Claims

1. A method for counting people in a vehicle cabin using a radar sensor and a deep learning model trained with a label-aware ranked (LAR) loss function, comprising:
   - Mounting an Infineon XENSIV 60 GHz radar sensor on the internal-upper side of the front window of the vehicle cabin.
   - Capturing raw radar data from the sensor.
   - Preprocessing the raw radar data to generate frames suitable for input into a deep learning model.
   - Training a convolutional neural network (CNN) with the LAR loss function using the preprocessed frames.
   - Applying exponential smoothing (ES) to stabilize the output of the model during inference.

2. The method of claim 1, wherein the LAR loss function is defined as:

\[
L_{\text{LAR}} = \sum_{i=1}^{N} \sum_{j=1, j \neq i}^{N} \log(\Delta l) \cdot (1 - f_i^T f_j)
\]

where \( N \) is the number of samples, \( f_i \) and \( f_j \) are the normalized feature vectors, and \( \Delta l = |l_a - l_n| \) is the difference between the labels of the anchor and negative samples.

3. The method of claim 1, wherein the deep learning model includes three convolutional layers, each followed by a ReLU activation function and a pooling layer, and a fully connected ReLU layer for outputting the predicted number of people.

4. The method of claim 1, further comprising evaluating the performance of the model using accuracy and accuracy +/-1 metrics.

5. A system for counting people in a vehicle cabin, comprising:
   - An Infineon XENSIV 60 GHz radar sensor mounted on the internal-upper side of the front window of the vehicle cabin.
   - A data preprocessing unit for processing raw radar data to generate frames suitable for input into a deep learning model.
   - A convolutional neural network (CNN) trained with a label-aware ranked (LAR) loss function using the preprocessed frames.
   - An exponential smoothing (ES) module for stabilizing the output of the model during inference.

6. The system of claim 5, wherein the LAR loss function is defined as:

\[
L_{\text{LAR}} = \sum_{i=1}^{N} \sum_{j=1, j \neq i}^{N} \log(\Delta l) \cdot (1 - f_i^T f_j)
\]

where \( N \) is the number of samples, \( f_i \) and \( f_j \) are the normalized feature vectors, and \( \Delta l = |l_a - l_n| \) is the difference between the labels of the anchor and negative samples.

7. The system of claim 5, wherein the deep learning model includes three convolutional layers, each followed by a ReLU activation function and a pooling layer, and a fully connected ReLU layer for outputting the predicted number of people.

8. The system of claim 5, further comprising an evaluation module for evaluating the performance of the model using accuracy and accuracy +/-1 metrics.

## Drawings

- **Figure 1:** Block diagram of the system architecture.
- **Figure 2:** Example frames generated from raw radar data.
- **Figure 3:** Visualization of the embedding space before and after applying the LAR loss function.
- **Figure 4:** Performance comparison of different loss functions on the people counting task.

## Abstract

A system and method for accurately counting people in a vehicle cabin using a radar sensor and a deep learning model trained with a label-aware ranked (LAR) loss function. The LAR loss ensures that the embedding space is ordered according to the number of people, enhancing predictive performance. Temporal smoothing stabilizes the output during inference, making the system reliable in real-world applications.

## Keywords

- Radar Sensor
- People Counting
- Label-Aware Ranked Loss (LAR)
- Convolutional Neural Network (CNN)
- Exponential Smoothing (ES)