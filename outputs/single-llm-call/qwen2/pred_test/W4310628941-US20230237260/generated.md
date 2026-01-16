# DESCRIPTION

## BACKGROUND

Anomaly detection plays a crucial role in various real-world applications, including identifying manufacturing defects, network security threats, and financial fraud. Traditional anomaly detection methods can be categorized into fully supervised, fully unsupervised, and semi-supervised settings. Fully supervised methods require labeled data for both normal and anomalous samples, which can be costly and time-consuming. Fully unsupervised methods do not require any labeled data but often suffer from performance degradation compared to supervised methods. Semi-supervised methods aim to bridge this gap by leveraging a limited amount of labeled data along with a larger set of unlabeled data.

However, a significant challenge in semi-supervised anomaly detection is the distribution mismatch between labeled and unlabeled data. This mismatch can arise due to various reasons, such as the introduction of new anomaly types, changes in the manufacturing process, or the adversarial nature of financial fraud. Existing semi-supervised methods often assume that labeled and unlabeled data come from the same distribution, which is not always the case in real-world scenarios. This assumption can lead to suboptimal performance and limit the applicability of these methods.

To address these challenges, we propose a novel semi-supervised anomaly detection framework called SPADE (Semi-supervised Pseudo-labeler Anomaly Detection with Ensembling). SPADE is designed to handle distribution mismatches effectively and achieve robust performance in various semi-supervised settings.

## BRIEF SUMMARY

The present invention relates to a semi-supervised anomaly detection framework, SPADE, designed to handle distribution mismatches between labeled and unlabeled data. SPADE introduces a novel pseudo-labeling mechanism using an ensemble of one-class classifiers (OCCs) and combines supervised and self-supervised learning to train a robust anomaly detection model. Key features of SPADE include:

1. **Pseudo-labeling Mechanism**: SPADE uses an ensemble of OCCs to generate pseudo-labels for unlabeled data, reducing the dependence on the labeled data and making the model more robust to distribution shifts.
2. **Combination of Supervised and Self-supervised Learning**: SPADE leverages both labeled and pseudo-labeled data for supervised learning and incorporates self-supervised learning to improve the quality of the learned representations.
3. **Hyperparameter Selection Without Validation Set**: SPADE employs a partial matching method to determine critical hyperparameters without requiring a validation set, which is often unavailable in real-world scenarios with limited labeled data.
4. **Robust Performance**: SPADE demonstrates state-of-the-art performance in various semi-supervised settings, including scenarios with new types of anomalies, labeling based on the easiness of samples, and positive and unlabeled (PU) learning.

## DETAILED DESCRIPTION

### Problem Formulation

We focus on the general semi-supervised anomaly detection problem with distribution mismatch. Given labeled training data \( \mathbf{x}_l \sim P_l(X) \) and unlabeled training data \( \mathbf{x}_u \sim P_u(X) \), where \( P_l(X) \) and \( P_u(X) \) are the feature distributions of the labeled and unlabeled data, respectively. The labels \( y \in Y \) are binary, indicating whether a sample is normal (0) or anomalous (1). The distribution of normal and anomalous samples is imbalanced, with far more normal examples than anomalies, i.e., \( P(y = 0) \gg P(y = 1) \).

Most semi-supervised methods assume that \( P_l(X) = P_u(X) \). However, in this work, we allow for the scenario where \( P_l(X) \neq P_u(X) \). The goal is to construct an anomaly detection model \( f : X \to Y \) that minimizes the test loss \( L(f(\mathbf{x}), y) \) over the union of \( P_l(X) \) and \( P_u(X) \).

### Core Idea and Desiderata

The core idea of SPADE is based on self-training, where a binary classifier is iteratively trained using labeled and pseudo-labeled data. The key component is the pseudo-labeler, which assigns binary labels to unlabeled data. Unlike traditional methods that use a trained binary classifier for pseudo-labeling, SPADE decouples the pseudo-labeler from the binary classifier and builds it using an ensemble of OCCs. This approach prevents overfitting to the small amount of labeled data and makes the model more robust to distribution shifts.

### Building Blocks

SPADE consists of four main components:
1. **Encoder**: \( h : X \to H \) maps the input features \( \mathbf{x} \) into latent representations \( \mathbf{r} = h(\mathbf{x}) \). The encoder can be any neural network architecture, such as a multi-layer perceptron (MLP) for tabular data or a convolutional neural network (CNN) for image data.
2. **Predictor**: \( q : H \to Y \) utilizes the learned representation \( \mathbf{r} \) to output the anomaly scores \( q(\mathbf{r}) \). The anomaly score is determined by the encoder and predictor as \( q(h(\mathbf{x})) \).
3. **Pseudo-labeler**: \( v : H \to \{0, 1, -1\} \) determines the pseudo-labels of the unlabeled data \( \mathbf{x}_u \) using an ensemble of OCCs. \( v(h(\mathbf{x}_u)) = 1/0/-1 \) represents pseudo-anomalous/pseudo-normal/unlabeled.
4. **Projection Head**: \( g : H \to G \) helps in representation learning of the encoder. Various pretext tasks, such as contrastive learning or masked autoencoder, can be used to train the projection head.

### Pseudo-labeling via Consensus

A major novel component of SPADE is the design of the pseudo-labeler. The pseudo-labeler \( v \) is composed of an ensemble of \( K \) OCCs. The pseudo-labels are assigned based on the consensus of the OCCs:
- Positive pseudo-labels (anomalous predictions) are assigned if all OCCs agree on them.
- Negative pseudo-labels (normal predictions) are assigned if all OCCs agree on them.
- Unlabeled data without consensus are annotated as unknown.

### Determining \( \eta_p \) and \( \eta_n \) Using Partial Matching

In SPADE, thresholds \( \eta_p \) and \( \eta_n \) are critical parameters for pseudo-labeling. Instead of treating them as user-defined hyperparameters, SPADE uses a partial matching method to determine these parameters without a validation set. The partial matching method matches the distribution of anomaly scores of the labeled data to that of the unlabeled data to estimate their marginal distribution and determine \( \eta_p \) and \( \eta_n \).

### Loss Functions and Optimization

SPADE trains the anomaly detection model \( q(h(\cdot)) \) using three loss functions:
1. **Binary Cross-Entropy (BCE) on Labeled Data**:
   \[
   \mathcal{L}_{\text{labeled}} = -\frac{1}{|D_l|} \sum_{(\mathbf{x}_i, y_i) \in D_l} \left[ y_i \log q(h(\mathbf{x}_i)) + (1 - y_i) \log (1 - q(h(\mathbf{x}_i))) \right]
   \]
2. **BCE on Pseudo-labeled Data**:
   \[
   \mathcal{L}_{\text{pseudo}} = -\frac{1}{|D_u^{\text{pseudo}}|} \sum_{\mathbf{x}_i \in D_u^{\text{pseudo}}} \left[ v(h(\mathbf{x}_i)) \log q(h(\mathbf{x}_i)) + (1 - v(h(\mathbf{x}_i))) \log (1 - q(h(\mathbf{x}_i))) \right]
   \]
   Here, \( D_u^{\text{pseudo}} \) is the set of unlabeled data with known pseudo-labels.
3. **Self-supervised Loss**:
   \[
   \mathcal{L}_{\text{self}} = \mathcal{L}_{\text{reconstruction}} \text{ or } \mathcal{L}_{\text{contrastive}}
   \]
   The self-supervised loss depends on the application domain and can include reconstruction objectives or contrastive learning objectives.

The overall optimization problem is:
\[
h^*, g^*, q^* = \arg \min_{h, g, q} \alpha \mathcal{L}_{\text{labeled}} + \beta \mathcal{L}_{\text{pseudo}} + \gamma \mathcal{L}_{\text{self}}
\]
where \( \alpha \), \( \beta \), and \( \gamma \) are hyperparameters. In our experiments, we set \( \alpha = 1 \), \( \beta = 1 \), and \( \gamma = 1 \).

### Experiments

We conducted extensive experiments to evaluate the performance of SPADE in various practical settings of semi-supervised learning with distribution mismatch. We used multiple anomaly detection datasets, including image and tabular data types.

#### New Types of Anomalies

Anomalies can evolve over time, leading to distribution shifts between labeled and unlabeled data. We constructed datasets with multiple anomaly types, providing subsets of the anomaly types as labeled data and the rest as unlabeled data. SPADE achieved consistently better performance in all metrics (overall, given, and missed AUC), demonstrating its generalizability to unseen anomalies.

#### Labeling Based on the Easiness of Samples

Human labeling can be more confident on easy samples, leading to distribution mismatches. We simulated this scenario by labeling only the easy-to-label samples and including hard-to-label samples in the unlabeled dataset. SPADE achieved superior or similar anomaly detection performance compared to the best alternatives, highlighting its potential in reducing human labeling costs.

#### Positive and Unlabeled (PU) Learning

In PU settings, only positive samples are labeled, and all other samples are unlabeled. SPADE generalized much better and outperformed all other alternatives, especially in scenarios with new types of anomalies.

#### Time-varying Distributions: Real-world Fraud Detection

We evaluated SPADE on two real-world fraud detection datasets, where anomalies evolve over time. SPADE improved the anomaly detection performance using both labeled data and newly-gathered unlabeled data, demonstrating its effectiveness in dynamic environments.

### Discussions

#### Accuracy of the Pseudo-labels

The accuracy of the pseudo-labeler is crucial for the robustness of semi-supervised anomaly detection. We analyzed the precision of the pseudo-labels for both normal and anomalous samples. The proposed pseudo-labeler achieved robust pseudo-labeling for normal samples, with high precision for anomalous samples when the anomaly scores were above 80%.

#### Ablation Studies

We conducted ablation studies to understand the impact of each component in SPADE. All components contributed significantly to the robust anomaly detection performance, with the self-supervised learning component contributing to 0.018 AUC improvements. The performance was not sensitive to the hyperparameter \( \alpha \), demonstrating the stability of SPADE.

### Conclusions

Semi-supervised anomaly detection is a critical challenge in practice, especially when the distributions of labeled and unlabeled samples are different. SPADE addresses this challenge by introducing a novel pseudo-labeling mechanism using an ensemble of OCCs and combining supervised and self-supervised learning. SPADE also includes a novel approach to pick hyperparameters without a validation set, making it data-efficient. Overall, SPADE consistently outperforms alternatives in various scenarios, achieving significant AUC improvements. Future work can extend this semi-supervised framework to multi-class classification or regression with distribution mismatch.