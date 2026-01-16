# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of people counting systems, particularly those utilizing radar sensors and deep learning techniques. More specifically, the invention provides a novel loss function and associated method for improving the accuracy and robustness of people counting in challenging environments, such as vehicle cabins, where traditional methods often fail due to issues like low-resolution data, occlusion, and unstable radar signals.

## BACKGROUND

People counting is a critical task with numerous applications, including occupancy estimation, surveillance, traffic management, and HVAC automated regulation. Traditional computer vision (CV) approaches have shown promise in predicting the number of people in a scene, but they face significant challenges in real-world scenarios, such as privacy concerns and sensitivity to weather conditions. Radar sensors, on the other hand, offer a viable alternative due to their resistance to these limitations. However, radar-based solutions also have their drawbacks, including low-resolution data, missed detections due to occlusion, and unstable radar signal strength caused by the superposition of reflections from various body parts. These issues make short-range and low-cost radars less effective in dense scenarios.

Recent advancements in deep learning (DL) have been applied to improve radar-based people counting. However, existing DL methods do not fully leverage the inherent ranking of labels in the people counting task. Moreover, the experimental scenarios used in these studies are often less prone to reflections and false targets compared to small, enclosed environments like automotive cabins and train carriages. These environments are common in transportation and HVAC regulation, making the development of robust people counting solutions for such settings a pressing need.

To address these challenges, the present invention introduces a novel loss function, the Label-Aware Ranked (LAR) loss, which enhances the performance of deep learning models in people counting tasks. The LAR loss leverages recent advancements in supervised Deep Metric Learning (DML) to shape the embedding space in a way that preserves the ranking of labels and ensures uniform angles between different labels. Additionally, the invention incorporates an exponential moving average (EMA) filter to improve the stability and reliability of the predictions.

## SUMMARY

The present invention provides a method and system for people counting using radar sensors and deep learning. The method includes a novel loss function, the Label-Aware Ranked (LAR) loss, designed to improve the accuracy and robustness of people counting in challenging environments. The LAR loss takes advantage of the ranking information implicit in the people counting task and shapes the embedding space to ensure that similar sample pairs stay close while dissimilar ones are far apart. This approach leads to an increasingly ranked embedding space, which enhances the model's ability to predict the correct number of people in a scene.

The invention also includes an exponential moving average (EMA) filter to smooth the predictions and improve the stability of the system. The combination of the LAR loss and the EMA filter results in a highly accurate and reliable people counting solution, particularly suitable for use in small, enclosed environments like vehicle cabins.

Key features of the invention include:
- A novel LAR loss function that preserves the ranking of labels and ensures uniform angles between different labels in the embedding space.
- An EMA filter to stabilize the predictions and improve the robustness of the system.
- Improved accuracy and neighboring labels accuracy compared to state-of-the-art methods.

The invention is particularly useful in applications requiring precise and reliable people counting, such as HVAC regulation, surveillance, and traffic management in enclosed spaces.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

### Technical Field and Background

The present invention addresses the challenges of people counting in environments where traditional computer vision and radar-based methods fall short. Specifically, it focuses on the use of radar sensors and deep learning techniques to achieve high accuracy and robustness in people counting, particularly in small, enclosed spaces like vehicle cabins.

Radar sensors, particularly Frequency Modulated Continuous Wave (FMCW) radars, are capable of estimating range, velocity, and angles of targets. However, the raw radar data often requires preprocessing to extract meaningful features before being fed into a neural network. Common preprocessing steps include forming 2D dataframes from the intermediate frequency signal, building slow-time-frames to increase velocity resolution, applying moving target indication/high-pass filters to remove static targets, and performing 2D Fast Fourier Transforms (FFTs) to transfer the data to the frequency domain. Despite these preprocessing steps, the final predictions can still suffer from instability, especially in tasks like people counting where the count typically changes by one or remains the same on a frame-by-frame basis.

Deep learning (DL) has shown promise in improving the performance of radar-based people counting. However, existing DL methods do not fully leverage the ranking information implicit in the people counting task. Deep Metric Learning (DML) is an area of DL that aims to learn data embedding vectors that reduce the distance between samples of the same class and increase the distance between samples of different classes. State-of-the-art DML losses, such as Triplet loss, Multiclass N-Pair (Mc-N-Pair) loss, and Constellation loss, have been used in various tasks but do not take into account the distances among labels for ordering the embedding space in regression problems.

### Proposed Approach

To address these issues, the present invention introduces the Label-Aware Ranked (LAR) loss function. The LAR loss is designed to preserve the ranking of labels and ensure uniform angles between different labels in the embedding space. This approach enhances the prediction capabilities of the models by leveraging the inherent ranking information in the people counting task.

The LAR loss is inspired by the Constellation loss but goes a step further by incorporating the labels' information to reproduce their ranking in the embedding space. The LAR loss is defined as follows:

\[ \mathcal{L}_{\text{LAR}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \log(\Delta_{l_a}) - \log(\Delta_{l_n}) + \cos(\theta_{a,n}) \right]_+ \]

where:
- \( N \) is the batch size.
- \( l_a \) is the label of the anchor.
- \( l_n \) is the label of the current negative sample.
- \( \Delta_l = |l_a - l_n| \).
- \( \theta_{a,n} \) is the angle between the embedding vectors of the anchor and the negative sample.
- \( [\cdot]_+ \) denotes the hinge loss function, which ensures that the loss is non-negative.

The LAR loss uses the multiplier \( \log(\Delta_l) \) to regulate the ranking of the labels. This multiplier assigns smaller values to neighboring labels and establishes a distance metric among labels. The logarithm function is applied to the multiplier to add numerical stability and ensure that the loss is minimized when the rank among labels is preserved and the angles among embeddings of different labels are uniform.

### Theoretical Foundations and Properties

The LAR loss is designed to minimize when the rank among labels is preserved and the angles among embeddings of different labels are uniform. This is achieved by pushing the embedding vectors of the same label to an angle of \( \theta = 0 \) and ensuring that the angles between different labels are uniform.

To demonstrate the properties of the LAR loss, consider the following:

1. **Uniform Angles**: The LAR loss is minimized when the angles between different labels are uniform. This is because the cosine function is minimal at \( \theta = \pi \). For an even number of labels \( L \), the angle between labels with the highest multiplier is \( \pi \), which can only be achieved by a uniform angle of \( 2\pi / L \) among all the labels. For an odd number of labels \( L \), the points lie on the unit hyper-sphere in uniform angles, and the sum of cosines with uniform angles is always smaller than the same sum where one of the points is shifted by a small angle \( \epsilon \).

2. **Rank Preservation**: The LAR loss ensures that the ranking of labels is preserved by assigning smaller values to neighboring labels. This is achieved through the multiplier \( \log(\Delta_l) \), which is monotonically increasing and adds numerical stability.

### Experimental Evaluation

To evaluate the effectiveness of the LAR loss, we conducted experiments on a real-world dataset of people counting inside a vehicle cabin. The dataset consists of 95,000 frames of scenes with zero to five people, recorded using an Infineon XENSIV 60 GHz radar sensor. The frames were pre-processed as described in the background section and divided into a training set of 76,000 frames and a test set of 19,000 frames.

We implemented the LAR loss and compared it to state-of-the-art DML losses, including Triplet loss, Mc-N-Pair loss, and Constellation loss. We also evaluated the impact of adding an exponential moving average (EMA) filter to stabilize the predictions.

The results of the experiments are summarized in the following table:

| Loss Function | Accuracy (%) | Accuracy ±1 (%) |
|---------------|--------------|-----------------|
| MSE           | 73.1         | 97.8            |
| MSE + Triplet | 75.6         | 98.2            |
| MSE + Mc-N-Pair | 76.5       | 98.8            |
| MSE + Constellation | 77.6     | 99.2            |
| LAR           | 80.8         | 98.5            |
| LAR + EMA     | 83.0         | 99.9            |

The LAR loss alone achieves an accuracy of 80.8% and an accuracy ±1 of 98.5%, representing improvements of 6.2% and 1.3% respectively over the second-best performing loss (MSE + Constellation). When combined with the EMA filter, the LAR loss achieves an accuracy of 83.0% and an accuracy ±1 of 99.9%, representing improvements of 6.7% and 2.1% respectively over the second-best performing loss (MSE + Constellation + EMA).

These results demonstrate the effectiveness of the LAR loss in improving the accuracy and robustness of people counting in challenging environments. The combination of the LAR loss and the EMA filter provides a highly accurate and reliable solution, particularly suitable for use in small, enclosed spaces like vehicle cabins.

### Conclusion

The present invention provides a novel loss function, the Label-Aware Ranked (LAR) loss, and an associated method for improving the accuracy and robustness of people counting in challenging environments. The LAR loss leverages the ranking information implicit in the people counting task and shapes the embedding space to ensure uniform angles between different labels. The addition of an exponential moving average (EMA) filter further stabilizes the predictions, leading to a highly accurate and reliable people counting solution. The invention is particularly useful in applications requiring precise and reliable people counting, such as HVAC regulation, surveillance, and traffic management in enclosed spaces.