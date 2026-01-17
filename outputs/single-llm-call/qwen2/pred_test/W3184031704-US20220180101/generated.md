# DESCRIPTION

## TECHNICAL FIELD

The present disclosure relates to the field of self-supervised learning for video representation learning, particularly focusing on a novel method for cooperative contrastive learning (CoCon) that leverages multiple views to improve the quality of learned representations. The disclosed method is applicable to various downstream tasks, including video classification and action recognition, and is designed to overcome the limitations of existing contrastive learning approaches by utilizing inter-view information to uncover implicit relationships between instances.

## DESCRIPTION OF RELATED ART

Recent advancements in self-supervised learning have shown significant promise in visual representation learning, often outperforming their supervised counterparts. In the video domain, self-supervised methods have been developed to exploit the temporal consistency in neighboring frames, typically through future prediction tasks. These approaches can be categorized into two main types: (1) predicting a reconstruction of future frames and (2) predicting features representing the future frames. While these methods have led to improved performance, the quality of the learned features still lags behind that of supervised methods.

One of the primary challenges in self-supervised learning is the lack of labels, which makes it difficult to establish direct associations between different training instances. Prior work has addressed this by learning associations based on structural constraints, such as temporal or spatial proximity. However, these methods often enforce similarity constraints between instances from the same video while pushing instances from other videos far apart, even if they represent the same semantic content. This limitation results in features with limited semantic knowledge and encourages low-level discrimination between different videos.

The concept of utilizing multiple views of information has been well-established in human perception and has been explored in machine learning to improve representation quality. Multi-view learning approaches have utilized core ideas such as contrastive learning and mutual information maximization to enhance representations. Despite these efforts, existing methods still suffer from the drawback of low-level discrimination between similar instances, leading to suboptimal representations.

## BRIEF SUMMARY OF THE DISCLOSURE

The present disclosure addresses the aforementioned challenges by introducing Cooperative Contrastive Learning (CoCon), a novel method for self-supervised video representation learning. CoCon leverages multiple views of the input to overcome the limitations of existing contrastive learning approaches. The main motivation behind CoCon is that each view captures a specific pattern, which can be useful to guide other views and improve representations. By utilizing inter-view information, CoCon avoids the drawback of discriminating similar instances and uncovers implicit relationships between instances in a self-supervised multi-view setting.

CoCon operates by computing view-specific distances and synchronizing them across all views. This is achieved through a consistency loss that encourages the relationships between instances to be the same across views. Additionally, a hinge loss is used to promote well-distributed distances, ensuring that the model can generate both positive and negative pairs. The overall loss combines these components to guide the learning process, resulting in improved representations for downstream tasks such as video classification and action recognition.

The disclosed method is applicable to various views, including RGB frames, optical flow, pose keypoints, and segmentation masks. These views are not independent but are complementary and lead to significant gains in representation quality. The extensible nature of CoCon makes it possible to use the method with any publicly available video dataset and other contrastive learning approaches.

## DETAILED DESCRIPTION

### Introduction

The present disclosure introduces Cooperative Contrastive Learning (CoCon), a novel method for self-supervised video representation learning. CoCon leverages multiple views of the input to improve the quality of learned representations by overcoming the limitations of existing contrastive learning approaches. The main motivation is that each view captures a specific pattern, which can be useful to guide other views and improve representations. By utilizing inter-view information, CoCon avoids the drawback of discriminating similar instances and uncovers implicit relationships between instances in a self-supervised multi-view setting.

### Technical Background

#### Self-Supervised Learning from Videos

Self-supervised learning from videos has gained significant attention due to the rich temporal structure in video data. Recent approaches have attempted to exploit this structure through future prediction tasks, either by predicting the reconstruction of future frames or by predicting features representing the future frames. These methods have shown promising results but still fall short of the performance of supervised methods.

#### Multi-View Learning

Multi-view learning has been explored to improve representation quality by leveraging multiple views of the input. These views can be derived from the original input, such as RGB frames and optical flow, or can be high-level inferred semantics, such as pose keypoints and segmentation masks. Existing multi-view learning approaches have utilized contrastive learning and mutual information maximization to enhance representations. However, these methods still suffer from the drawback of low-level discrimination between similar instances.

### Method

#### Problem Formulation

A video \( V \) is a sequence of \( T \) frames with resolution \( H \times W \) and \( C \) channels, represented as \( \{i_1, i_2, \ldots, i_T\} \), where \( i_t \in \mathbb{R}^{H \times W \times C} \). We partition the video clip \( V \) into \( N \) disjoint blocks, \( V = \{x_1, x_2, \ldots, x_N\} \), where \( x_j \in \mathbb{R}^{K \times H \times W \times C} \) and \( T = N \times K \). A non-linear encoder \( f(\cdot) \) transforms each input block \( x_j \) into its latent representation \( z_j = f(x_j) \). An aggregation function \( g(\cdot) \) takes a sequence \( \{z_1, z_2, \ldots, z_j\} \) as input and generates a context representation \( c_j = g(z_1, z_2, \ldots, z_j) \). In our setup, \( z_j \in \mathbb{R}^{H' \times W' \times D} \) and \( c_j \in \mathbb{R}^D \), where \( D \) represents the embedding size and \( H', W' \) represent down-sampled resolutions.

#### Prediction Task

To learn effective representations, we create a prediction task involving predicting the latent state of future blocks. We define a predictive function \( \phi(\cdot) \) that takes the context representation \( c_t \) as input and predicts the latent state of the future frames. The formulation is given by:

\[
\hat{z}_{t+1} = \phi(c_t)
\]

We then utilize the predicted \( \hat{z}_{t+1} \) to compute the next context representation \( c_{t+1} \). This process can be repeated for multiple steps to capture long-range semantics. In our experiments, we predict the next three blocks using the first five blocks.

#### Contrastive Loss

We use Noise Contrastive Estimation (NCE) to construct a binary classification task where a classifier distinguishes between real and noisy samples. The NCE loss is defined as:

\[
L_{\text{cpc}} = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\sum_{k \neq i} \exp(z_k \cdot z_j / \tau)}
\]

where \( z_i \) and \( z_j \) are feature embeddings for the same instance, and \( \tau \) is a temperature parameter. This loss is used to train the encoders for each view independently.

#### Cooperative Multi-View Learning

To extend the contrastive learning framework to multiple views, we utilize different encoders \( \phi_v \) for each view \( v \). We train these encoders by applying the NCE loss independently for each view. However, to leverage inter-view information, we introduce a consistency loss that synchronizes the distances between instances across views. This is achieved by computing view-specific distances and enforcing a consistency loss between them.

The consistency loss is defined as:

\[
L_{\text{sync}} = \sum_{v_0, v_1} \| W_{v_0} - W_{v_1} \|^2
\]

where \( W_v \) is the graph similarity matrix for view \( v \), computed using a distance metric such as cosine distance. The consistency loss ensures that similar pairs in one view are also similar in other views.

To promote well-distributed distances, we use a hinge loss:

\[
L_{\text{sim}} = \sum_{a, b} \max(0, \mu - D(h_a^v, h_b^v))
\]

where \( h_a^v \) represents the representation for instance \( a \) in view \( v \), and \( \mu \) is a margin parameter. The hinge loss pushes representations of the same instance in different views closer while pushing different instances apart.

The overall cooperative loss is given by:

\[
L_{\text{coop}} = L_{\text{sync}} + \alpha \cdot L_{\text{sim}}
\]

where \( \alpha \) is a hyperparameter that balances the contributions of the consistency and hinge losses. The final loss for the model is:

\[
L = L_{\text{cpc}} + L_{\text{coop}}
\]

### Experiments

#### Datasets

The effectiveness of the disclosed method is validated on several human action datasets, including UCF101, HMDB51, and Kinetics400. UCF101 contains 13,000 videos spanning 101 human action classes, HMDB51 contains 7,000 video clips for 51 classes, and Kinetics400 is a large video dataset with 306,000 video clips from 400 classes.

#### Views

For Kinetics400, we learn encoders for RGB and optical flow using Farneback flow. For UCF101 and HMDB51, we learn encoders for RGB, TVL1 optical flow, pose heatmaps, and human segmentation masks. These views are generated using off-the-shelf detectors without any pre/post-processing.

#### Implementation Details

We use a 3D-ResNet as the encoder \( f(\cdot) \). The video is partitioned into 8 blocks, each containing 5 frames. The predictive task involves predicting the last three blocks using the first five blocks. We use standard data augmentations during training and train the models using the Adam optimizer with an initial learning rate of \( 10^{-3} \). The models are trained on 4 GPUs with a batch size of 16 samples per GPU.

#### Action Classification

The effectiveness of the learned representations is measured using the downstream task of action classification. The self-supervised model weights are used as initialization for supervised learning, and the architecture is fine-tuned end-to-end using class label supervision. The fine-tuned accuracies are reported on UCF101 and HMDB51.

### Results

#### Quantitative Results

We perform extensive ablation studies and experiments on multiple datasets to validate the effectiveness of CoCon. The results demonstrate that CoCon outperforms existing methods, particularly when pre-trained on large datasets like Kinetics400. The use of multiple views leads to significant performance gains, even when the additional views are noisy and derived from the original modality.

#### Qualitative Results

We analyze the learned representations qualitatively by studying the inter-class relationships and action alignment. The results show that CoCon is able to capture higher-order semantics and uncover meaningful relationships between action classes. The t-SNE visualizations and cosine similarity distributions further confirm the effectiveness of the learned representations.

### Conclusion

The present disclosure introduces Cooperative Contrastive Learning (CoCon), a novel method for self-supervised video representation learning. CoCon leverages multiple views of the input to improve the quality of learned representations by overcoming the limitations of existing contrastive learning approaches. The method is applicable to various downstream tasks, including video classification and action recognition, and demonstrates significant performance gains on multiple datasets. The extensible nature of CoCon makes it a versatile tool for enhancing representation quality in self-supervised learning.