# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of video anomaly detection (VAD) and, more specifically, to a method and system for zero-shot cross-domain video anomaly detection (zxVAD). The invention addresses the challenge of detecting anomalies in a target domain without any access to the target domain's training data, thereby providing a robust solution for unsupervised VAD in cross-domain settings.

## BACKGROUND

Unsupervised video anomaly detection (VAD) methods have gained significant traction in security and surveillance applications due to their ability to operate without labeled training data containing abnormal events. Traditional unsupervised VAD methods typically require training data from the same domain where the system will be deployed, which can be impractical or impossible in many real-world scenarios. Cross-domain VAD (xVAD) methods aim to address this limitation by leveraging training data from a source domain to detect anomalies in a different target domain. However, existing xVAD methods often require some form of access to the target domain's training data or strong supervision from pre-trained object detectors, which can be time-consuming and resource-intensive.

The present invention introduces a novel framework, Zero-shot Cross-domain Video Anomaly Detection (zxVAD), which eliminates the need for any target domain training data or strong supervision. zxVAD leverages a combination of a future-frame prediction generator, a Normalcy Classifier (NC) module, and an Untrained Convolutional Neural Network (CNN) Anomaly Synthesis (O) module to learn the relative normalcy features of input videos. This approach enhances the model's ability to generalize across new target domains without any adaptation at test time.

## SUMMARY

The present invention provides a method and system for zero-shot cross-domain video anomaly detection (zxVAD). The method includes the following steps:

1. **Data Preparation**: Collecting video data containing only normal events from a source domain.
2. **Generator Training**: Training a future-frame prediction generator using the source domain data. The generator is designed to predict the next frame in a sequence of video frames.
3. **Anomaly Synthesis**: Creating pseudo-abnormal frames using an Untrained Convolutional Neural Network (CNN) Anomaly Synthesis (O) module. The O module localizes objects from both task-relevant (VAD) and task-irrelevant (TI) video frames and pastes them onto normal VAD video frames to create diverse pseudo-anomalies.
4. **Normalcy Classification**: Training a Normalcy Classifier (NC) module to distinguish between the predicted normal future-frame and the pseudo-abnormal frames. The NC module uses novel loss functions to learn the relative normalcy features, focusing on the differences between normal and abnormal frames.
5. **Model Optimization**: Optimizing the generator, discriminator, and normalcy classifier using a combination of loss functions, including normalcy loss, relative normalcy loss, attention affirmation loss, and relative attention affirmation loss.
6. **Anomaly Detection**: Deploying the trained model to detect anomalies in the target domain without any access to the target domain's training data.

The key advantages of the zxVAD framework include:

- **Zero-Shot Learning**: The model can detect anomalies in the target domain without any training data from the target domain.
- **Generalization**: The model generalizes well across different target domains and types of anomalies.
- **Efficiency**: The model requires no strong supervision from pre-trained object detectors, reducing computational overhead and training time.
- **Flexibility**: The model can be trained using a combination of task-relevant and task-irrelevant data, enhancing its robustness and adaptability.

## DETAILED DESCRIPTION

### System Overview

The zxVAD framework consists of several key components working in concert to achieve zero-shot cross-domain video anomaly detection. The system architecture is designed to be modular and scalable, allowing for easy integration and customization.

#### 1. Data Preparation

The first step in the zxVAD process is data preparation. The system collects video data containing only normal events from a source domain. This data serves as the training set for the future-frame prediction generator. Additionally, the system can optionally incorporate task-irrelevant (TI) data, which consists of video frames from non-VAD tasks such as action recognition or video classification. TI data is used to create diverse pseudo-anomalies and to enhance the model's ability to learn normalcy features.

#### 2. Generator Training

The future-frame prediction generator is a crucial component of the zxVAD framework. The generator is trained to predict the next frame in a sequence of video frames based on the previous frames. The generator is designed to learn the normal event patterns in the source domain data. During training, the generator is regularized using the Normalcy Classifier (NC) module and the Untrained CNN Anomaly Synthesis (O) module to ensure that it focuses on the relative normalcy features.

#### 3. Anomaly Synthesis

The Untrained CNN Anomaly Synthesis (O) module is responsible for creating pseudo-abnormal frames. The O module localizes objects from both task-relevant (VAD) and task-irrelevant (TI) video frames and pastes them onto normal VAD video frames to create diverse pseudo-anomalies. This process involves the following steps:

- **Object Localization**: The O module uses an untrained CNN to extract objects from input video frames. The CNN is randomly initialized and does not require any training. The objects are localized using attention maps generated from the CNN's feature maps.
- **Pasting Objects**: The localized objects are pasted onto normal VAD video frames at random locations and sizes to create pseudo-abnormal frames. This process ensures that the pseudo-anomalies are diverse and representative of different types of anomalies.

#### 4. Normalcy Classification

The Normalcy Classifier (NC) module is trained to distinguish between the predicted normal future-frame and the pseudo-abnormal frames. The NC module uses novel loss functions to learn the relative normalcy features, focusing on the differences between normal and abnormal frames. The loss functions include:

- **Normalcy Loss**: This loss function optimizes the NC module to increase the probability that the predicted future-frame is normal and the pseudo-abnormal frame is abnormal.
- **Relative Normalcy Loss**: This loss function maximizes the probability that the predicted future-frame is more normal than the pseudo-abnormal frame, emphasizing the relative difference between normal and abnormal frames.
- **Attention Affirmation Loss**: This loss function ensures that the NC module considers the entire scene in the predicted future-frame to classify it as normal and focuses on the foreign objects in the pseudo-abnormal frame to classify it as abnormal.
- **Relative Attention Affirmation Loss**: This loss function enforces the relative difference in attention maps between normal and pseudo-abnormal frames, enhancing the model's ability to distinguish between normal and abnormal events.

#### 5. Model Optimization

The generator, discriminator, and normalcy classifier are optimized using a combination of loss functions. The optimization process involves the following steps:

- **Generator Loss**: The generator is optimized using a mean square error loss, structural similarity index measure (SSIM), and gradient loss to ensure accurate future-frame prediction.
- **Discriminator Loss**: The discriminator is optimized using a least-square GAN loss to distinguish between the predicted future-frame and the ground truth frame.
- **Normalcy Classifier Loss**: The normalcy classifier is optimized using the normalcy loss, relative normalcy loss, attention affirmation loss, and relative attention affirmation loss to learn the relative normalcy features.

#### 6. Anomaly Detection

Once the model is trained, it can be deployed to detect anomalies in the target domain without any access to the target domain's training data. The system processes input video frames from the target domain and uses the trained generator to predict the next frame in the sequence. The normalcy classifier then evaluates the predicted frame and the input frame to determine if an anomaly is present. If the predicted frame deviates significantly from the input frame, the system flags it as an anomaly.

### Implementation Details

The zxVAD framework is implemented using the PyTorch deep learning library. The generator is an U-Net adapted from existing architectures, with a memory module at its bottleneck to capture long-term dependencies. The discriminator and normalcy classifier are Patch-GAN discriminators. The system is trained using the Adam optimizer with appropriate learning rates and hyperparameters.

### Evaluation and Results

The zxVAD framework is evaluated under various training scenarios, including the use of both task-relevant and task-irrelevant data, and the use of only one of these data types. The performance of zxVAD is compared with existing state-of-the-art xVAD methods using metrics such as the area under the ROC curve (AUC), model storage, total parameters, GPU energy consumption, inference time FPS, and GMACs.

The results demonstrate that zxVAD outperforms existing methods in terms of AUC, efficiency, and generalization ability across different target domains. Specifically, zxVAD shows superior performance in detecting anomalies in the target domain without any adaptation or access to target domain training data. The use of task-irrelevant data further enhances the model's performance, validating the effectiveness of the proposed approach.

### Conclusion

The present invention, Zero-shot Cross-domain Video Anomaly Detection (zxVAD), provides a robust and efficient solution for unsupervised VAD in cross-domain settings. By leveraging a combination of a future-frame prediction generator, a Normalcy Classifier (NC) module, and an Untrained CNN Anomaly Synthesis (O) module, zxVAD eliminates the need for target domain training data and strong supervision, making it a versatile and practical solution for real-world applications. The results demonstrate the potential of task-irrelevant data as a promising direction for addressing the xVAD problem, and the framework's ability to generalize well across different target domains and types of anomalies.