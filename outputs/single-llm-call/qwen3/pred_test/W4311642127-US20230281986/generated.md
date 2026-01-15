# DESCRIPTION

## TECHNICAL FIELD

- relate to video anomaly detection

The present invention relates generally to the field of computer vision and artificial intelligence, specifically to systems and methods for unsupervised video anomaly detection in cross-domain settings without requiring access to training data from the target domain. More particularly, the invention encompasses a zero-shot cross-domain video anomaly detection framework that enables the identification of anomalous events in surveillance or monitoring video streams from previously unseen environments, using only normal video data from a source domain and optionally, task-irrelevant video data unrelated to the anomaly detection task. The system operates without any prior knowledge of the types, appearances, or temporal patterns of anomalies in the target domain, and does not require retraining, fine-tuning, or adaptation upon deployment in new environments. This capability is particularly valuable in real-world security, public safety, and industrial monitoring applications where collecting labeled abnormal data is infeasible due to the rarity of anomalies, privacy constraints, or proprietary restrictions on data sharing. The invention leverages a novel combination of generative modeling, relative normalcy learning, and untrained convolutional neural networks to synthesize diverse pseudo-anomalies and train a robust anomaly classifier that distinguishes normal from abnormal patterns based on their relative differences rather than absolute representations of normality alone.

## BACKGROUND

- introduce unsupervised video anomaly detection

Unsupervised video anomaly detection has emerged as a dominant paradigm in automated surveillance systems due to its ability to identify unusual events without requiring annotated examples of abnormal behavior. Traditional approaches rely on modeling the statistical distribution of normal activities observed during training, typically using deep generative models such as autoencoders or generative adversarial networks. These models learn to reconstruct or predict normal video frames with high fidelity, and deviations from this learned normality—measured via reconstruction error or prediction discrepancy—are flagged as anomalies during inference. While effective within the same domain where training data was collected, such methods exhibit severe performance degradation when applied to new environments with different camera angles, lighting conditions, object types, or behavioral contexts. This limitation arises because the learned representations are inherently biased toward the specific distribution of the source domain, failing to generalize to unseen variations that may constitute legitimate anomalies in a new setting.

- limitations of cross-domain video anomaly detection

Recent efforts to address domain generalization in video anomaly detection have proposed cross-domain variants, which attempt to transfer knowledge from a source domain with abundant normal training data to a target domain with no labeled anomalies. However, these approaches remain fundamentally constrained by their reliance on access to at least some portion of target domain training data, either for domain adaptation, feature alignment, or model calibration. Other methods require strong supervision in the form of pre-trained object detectors to localize and manipulate specific entities within frames, introducing computational overhead, dependency on external models, and vulnerability to detector inaccuracies. These requirements render existing cross-domain systems impractical for end-users who need plug-and-play solutions capable of operating immediately upon deployment, without access to target domain data or specialized infrastructure. Moreover, in many real-world scenarios—such as corporate surveillance, private property monitoring, or international border security—sharing video data across domains is legally or ethically prohibited due to privacy regulations, intellectual property concerns, or national security restrictions.

- motivate zero-shot cross domain video anomaly detection

To overcome these limitations, this invention introduces the concept of zero-shot cross-domain video anomaly detection, wherein anomaly detection is performed in a target domain with absolutely no access to any training or validation data from that domain, and with no prior knowledge of the nature or appearance of potential anomalies. This setting represents the most stringent and practical formulation of the problem, as it aligns with the operational reality of deploying surveillance systems in uncharted or restricted environments. The motivation for this paradigm stems from the observation that anomalies are not defined by fixed visual signatures but by contextual deviations from expected behavior. Therefore, a system capable of learning what constitutes normality relative to plausible deviations—rather than memorizing specific normal patterns—can generalize far more effectively across domains. The invention demonstrates that such relative learning can be achieved without target domain data by leveraging task-irrelevant video sources and a novel mechanism for synthesizing diverse pseudo-anomalies using untrained neural networks.

- identify need for improved xVAD systems

There exists a critical and unmet need for video anomaly detection systems that are not only accurate and efficient but also truly domain-agnostic, deployable without adaptation, and operable under strict data access constraints. Current systems fail to meet this need by requiring either target domain data for adaptation or pre-trained object detectors for supervision. The proposed invention fulfills this need by introducing a zero-shot framework that achieves state-of-the-art performance across multiple benchmark datasets without ever observing target domain examples, while simultaneously reducing computational cost, model size, and energy consumption compared to prior approaches. This represents a paradigm shift in the field, moving from domain-specific training to relative normalcy learning as the core principle of anomaly detection.

## SUMMARY

- introduce zero-shot cross-domain video anomaly detection system

The present invention discloses a zero-shot cross-domain video anomaly detection system capable of identifying anomalous events in video sequences from an unseen target domain using only normal video data from a source domain and optionally, task-irrelevant video data from unrelated domains. The system operates without any training, fine-tuning, or adaptation on target domain data and does not require access to pre-trained object detectors or labeled anomaly examples. It achieves this capability through a novel architecture comprising three integrated components: a future frame prediction module, a normalcy classifier module, and an object-aware anomaly synthesis module. Together, these components enable the system to learn the relative distinctions between normal and abnormal patterns by contrasting predicted normal frames with synthetically generated pseudo-abnormal frames, thereby generalizing to unseen anomaly types in new environments.

- collect video from source domain

The system is initialized by collecting a set of video sequences from a source domain, wherein all video content consists exclusively of normal, non-anomalous activities. These videos serve as the sole source of training data for learning the distribution of normal behavior. The source domain may correspond to any surveillance environment, such as a pedestrian walkway, traffic intersection, or factory floor, provided that the videos contain no annotated or visible anomalies. The system does not require any metadata, labels, or contextual information beyond the raw pixel data of the video frames.

- obtain images of foreground objects

Foreground objects are extracted from both the source domain videos and optionally, from task-irrelevant video datasets, using a randomly initialized convolutional neural network that has not been trained for object detection or classification. This network, when applied to a video frame, produces an attention map that highlights regions corresponding to semantically meaningful objects due to inherent architectural biases in deep convolutional layers, even without parameter optimization. These attention maps are thresholded to generate binary masks that isolate foreground objects from the background, enabling their extraction without any supervised training or domain-specific tuning.

- train first neural network for frame prediction

A first neural network, implemented as a generative adversarial network with a memory module, is trained to predict the next frame in a sequence given a temporal window of prior frames. The generator component of this network learns to reconstruct normal video dynamics by minimizing reconstruction error and structural similarity between predicted and ground truth frames. The discriminator component simultaneously learns to distinguish between real and predicted frames, encouraging the generator to produce temporally coherent and visually plausible outputs. A memory module is incorporated to store and retrieve sparse, high-dimensional representations of recurring normal patterns, enhancing the model’s capacity to capture long-term temporal dependencies and complex normal behaviors.

- train second neural network for anomaly classification

A second neural network, designated as the normalcy classifier, is trained to distinguish between predicted normal frames and synthetically generated pseudo-abnormal frames. Unlike traditional classifiers that learn to recognize normal patterns, this classifier is trained to recognize the relative difference between normality and abnormality. It is optimized using four novel loss functions that jointly enforce: (1) correct classification of normal versus pseudo-abnormal frames, (2) increased confidence that normal frames are more normal than pseudo-abnormal frames, (3) spatial attention focused on the entire scene for normal frames and on foreign objects for pseudo-abnormal frames, and (4) enhanced relative similarity between augmented normal frames and original normal frames compared to pseudo-abnormal frames. These losses are computed using attention maps derived from the classifier’s internal feature representations, enabling the system to learn context-aware distinctions that generalize across domains.

- utilize normalcy classifier module

The normalcy classifier module is the central innovation of the system, serving as a relative normalcy learning engine that transforms the anomaly detection problem from one of pattern memorization to one of contextual deviation detection. By comparing predicted normal frames with pseudo-abnormal frames created using untrained object extraction, the classifier learns to identify anomalies not by what they look like, but by how they differ from normality in a relative, context-sensitive manner. This approach eliminates the need for target domain data, as the classifier’s decision boundaries are defined by the diversity of pseudo-anomalies rather than by specific anomaly examples from the target domain.

- perform video anomaly detection

During inference, the system processes video frames from the target domain by first predicting the next frame using the trained generator. The difference between the predicted frame and the actual frame is then evaluated by the normalcy classifier, which outputs a normalcy score indicating the likelihood that the frame conforms to learned relative normalcy patterns. Frames with low normalcy scores are flagged as anomalous. The entire process requires no retraining, no domain adaptation, and no access to target domain data, enabling true zero-shot deployment across any new surveillance environment.

## DETAILED DESCRIPTION

- set context for description

The following detailed description provides a comprehensive exposition of the system architecture, training methodology, and operational principles of the zero-shot cross-domain video anomaly detection invention. The description is presented in a manner consistent with standard patent practice, defining components, relationships, and processes with sufficient specificity to enable reproduction by a person skilled in the art of computer vision and machine learning. All components are implemented as software modules executable on general-purpose computing hardware, and all training and inference procedures are performed using standard numerical optimization techniques and deep learning frameworks.

- define terminology and phrasing

For the purposes of this disclosure, the term “normal frame” refers to any video frame extracted from a source domain video sequence that contains only expected, non-anomalous activities. The term “pseudo-abnormal frame” refers to a synthetic frame created by inserting a foreground object, extracted from any video source using an untrained CNN, into a normal frame at a random location and scale. The term “task-irrelevant dataset” refers to any collection of video data originally curated for purposes other than anomaly detection, such as action recognition or video classification, and which contains no labeled anomalies. The term “zero-shot” denotes the absence of any training, validation, or test data from the target domain during any phase of system development or deployment. The term “relative normalcy” refers to the learned distinction between normal and abnormal patterns based on their comparative differences, rather than on absolute representations of normality.

- clarify representation of systems

The system is represented as a modular architecture comprising interconnected neural network components, each responsible for a distinct function in the anomaly detection pipeline. The generator, discriminator, memory module, normalcy classifier, and anomaly synthesis module are each implemented as separate neural network subcomponents with defined input-output relationships. The system is trained end-to-end using backpropagation, with gradients flowing from the combined loss functions of all components to update shared and independent parameters. The memory module is represented as a learnable key-value store, and the normalcy classifier is implemented as a PatchGAN discriminator with a modified output layer and loss structure. All components are described in terms of their functional behavior, mathematical operations, and interdependencies, without reference to proprietary implementations or specific hardware configurations.

- establish level of detail

The level of detail provided herein is sufficient to enable a person of ordinary skill in the art of machine learning and computer vision to implement the invention without undue experimentation. All architectural choices, loss functions, hyperparameters, and training procedures are described with sufficient specificity to permit replication. Where implementation details are omitted—for instance, exact layer dimensions or optimizer settings—they are either standard in the field or empirically determined and documented in the accompanying figures and supplementary materials. The invention is not limited to any specific implementation, dataset, or hardware platform, and encompasses all variations and modifications that fall within the scope of the claims.

### System Overview

- introduce training system

The training system comprises a unified computational framework that integrates a future frame prediction module, a normalcy classifier module, and an object-aware anomaly synthesis module into a single end-to-end trainable architecture. The system is designed to operate in a zero-shot cross-domain setting, meaning it is trained exclusively on normal video data from a source domain and optionally on task-irrelevant video data, with no access to any data from the target domain. During training, the system learns to distinguish normal video frames from synthetically generated pseudo-anomalies by leveraging relative normalcy learning, thereby acquiring generalizable knowledge of what constitutes deviation from normality without ever observing actual anomalies.

- describe system components

The system consists of five primary components: a generator network, a discriminator network, a memory module, a normalcy classifier network, and an object-aware anomaly synthesis module. The generator and discriminator form a generative adversarial network responsible for modeling the distribution of normal video frames and predicting future frames with high fidelity. The memory module, integrated into the bottleneck of the generator, stores sparse, high-dimensional representations of recurring normal patterns to enhance temporal consistency. The normalcy classifier is a convolutional neural network trained to classify frames as normal or pseudo-abnormal using novel loss functions that enforce relative normalcy distinctions. The object-aware anomaly synthesis module extracts foreground objects from video frames using an untrained CNN and pastes them into normal frames to create diverse pseudo-anomalies for training the classifier.

- explain memory storage

The memory module is implemented as a learnable key-value matrix, where each row corresponds to a memory slot containing a feature vector representing a learned normal pattern. During frame prediction, the generator computes an attention vector that determines which memory slots are most relevant for reconstructing the current frame. The attention vector is constrained to be sparse using a hard-shrinkage function and regularized by minimizing its entropy to encourage selective memory usage. Memory slots are updated during training via gradient descent, allowing the system to adaptively store and retrieve the most discriminative normal patterns observed in the source domain.

- detail processor functionality

The processor functionality is implemented as a sequence of computational operations executed on a graphics processing unit (GPU) or other parallel computing architecture. The processor receives a sequence of input frames, passes them through the generator to predict the next frame, compares the prediction with the ground truth to compute reconstruction loss, extracts foreground objects using the anomaly synthesis module, generates pseudo-abnormal frames, and feeds both normal and pseudo-abnormal frames into the normalcy classifier. The classifier computes four distinct loss terms based on its predictions and attention maps, and the total loss is backpropagated to update the parameters of the generator, discriminator, memory module, and classifier. All operations are performed in mini-batches, with gradients computed using automatic differentiation.

- introduce source domain data

Source domain data consists of video sequences captured in a controlled or monitored environment, such as a pedestrian zone, traffic intersection, or industrial facility, where all recorded activities are known to be normal. The data is collected in raw pixel format without any annotation, labeling, or metadata regarding anomalies. The system does not require knowledge of the environment’s semantics, camera parameters, or lighting conditions. The source domain may include multiple video clips recorded under varying conditions to enhance the diversity of normal behaviors learned by the system.

- describe normal events in source domain

Normal events in the source domain encompass all activities that are statistically frequent, temporally consistent, and contextually appropriate for the recorded environment. These include pedestrian walking, vehicle movement, object manipulation, and other routine behaviors that do not involve sudden disruptions, unusual interactions, or atypical motion patterns. The system does not require explicit definitions of normality; instead, it learns the statistical and structural properties of these activities implicitly through reconstruction and relative classification.

- explain foreground object extraction

Foreground object extraction is performed using a randomly initialized convolutional neural network that has not been trained for object detection or classification. The network processes a video frame and produces a feature tensor, which is then aggregated across channels using spatial channel-wise density averaging to generate an attention map. This attention map is thresholded to produce a binary mask that isolates regions corresponding to semantically meaningful objects. The extraction process requires no training, no labeled data, and no domain-specific tuning, relying instead on the inherent bias of deep convolutional layers to respond more strongly to textured, structured regions (foreground objects) than to homogeneous backgrounds.

- detail training of first neural network

The first neural network, comprising the generator and discriminator, is trained using a least-squares generative adversarial network formulation. The generator receives a sequence of four consecutive frames and predicts the fifth frame, while the discriminator attempts to distinguish between the predicted frame and the actual ground truth frame. Training is optimized by minimizing a composite loss function that includes mean squared error, structural similarity index, gradient loss, and memory regularization terms. The memory module is updated concurrently using a sparsity-inducing shrinkage function and entropy minimization to ensure efficient storage of normal patterns.

- introduce future frame prediction module

The future frame prediction module is the core generative component of the system, responsible for modeling the temporal dynamics of normal video sequences. It consists of a U-Net architecture with skip connections, a memory module at its bottleneck, and a discriminator for adversarial training. The module learns to predict the next frame in a sequence by encoding temporal dependencies through convolutional layers and retrieving relevant normal patterns from memory. Its output is used both for reconstruction loss and as input to the normalcy classifier for relative normalcy learning.

- describe generator and discriminator components

The generator is a convolutional encoder-decoder network with skip connections that maps a sequence of input frames to a predicted future frame. It incorporates a memory module that stores and retrieves latent representations of normal patterns. The discriminator is a PatchGAN architecture that evaluates local image patches to determine whether a frame is real or generated. Both components are trained adversarially, with the generator seeking to produce frames indistinguishable from real ones and the discriminator seeking to correctly classify them.

- explain reconstruction loss and discriminative loss

Reconstruction loss is computed as the sum of mean squared error between predicted and ground truth frames, structural similarity index, and gradient magnitude difference. Discriminative loss is computed using the least-squares adversarial objective, where the discriminator minimizes the squared difference between its predictions and binary labels, and the generator minimizes the squared difference between its predictions and the label for real frames. These losses jointly ensure that the generator produces temporally coherent and visually realistic frames that conform to the statistical distribution of normal activity.

- detail training of second neural network

The second neural network, the normalcy classifier, is trained using four novel loss functions: normalcy loss, relative normalcy loss, attention affirmation loss, and relative attention affirmation loss. These losses are computed using the classifier’s output logits and attention maps derived from its final convolutional layer. The classifier is trained on pairs of normal frames and pseudo-abnormal frames, with the goal of maximizing the relative difference between them. Training is performed using stochastic gradient descent with Adam optimizer, and the classifier’s parameters are updated in conjunction with the generator and discriminator during joint training.

- introduce normalcy classifier module

The normalcy classifier module is a convolutional neural network designed to distinguish between normal frames and pseudo-abnormal frames by learning relative normalcy features. Unlike traditional classifiers that learn to recognize normal patterns, this module learns to recognize the contrast between normality and abnormality. It is trained using attention maps that reveal which regions of the frame influence its decision, enabling it to focus on global scene context for normal frames and on inserted foreign objects for pseudo-abnormal frames.

- describe object-aware anomaly synthesis module

The object-aware anomaly synthesis module is responsible for generating diverse pseudo-abnormal frames by extracting foreground objects from video frames using an untrained CNN and pasting them into normal frames at random locations and scales. The module operates on both source domain frames and task-irrelevant video data, enabling the generation of anomalies containing a wide variety of object types, sizes, and appearances. This diversity ensures that the normalcy classifier learns to detect anomalies beyond those present in the source domain, enhancing generalization to unseen target domains.

- explain pseudo abnormal frame generation

Pseudo-abnormal frames are generated by first extracting a binary mask of foreground objects from a source or task-irrelevant frame using an untrained CNN. The mask is resized to a random scale and positioned at a random location within a normal frame from the source domain. The pixels corresponding to the mask are then copied onto the normal frame, replacing the underlying pixels to create a synthetic anomaly. This process is repeated for each training batch, ensuring that the classifier is exposed to a broad spectrum of possible anomalies during training.

- detail normalcy loss function

The normalcy loss function is a binary cross-entropy loss that encourages the classifier to assign high probability to normal frames and low probability to pseudo-abnormal frames. It is computed as the average of the negative log-likelihood of correct classifications across all training pairs. This loss ensures that the classifier learns to distinguish between the two classes at a basic level, forming the foundation for the more sophisticated relative losses.

- explain relative normalcy loss function

The relative normalcy loss function is a novel loss that enforces the condition that the probability of a normal frame being classified as normal should increase as the probability of a pseudo-abnormal frame being classified as normal decreases. It is computed as the difference between the classifier’s output for normal frames and pseudo-abnormal frames, and is designed to maximize the margin between them. This loss ensures that the classifier does not merely classify frames as normal or abnormal, but learns to rank them in terms of their relative normality.

- introduce attention affirmation loss function

The attention affirmation loss function ensures that the classifier’s attention map for a normal frame is distributed across the entire scene, reflecting holistic understanding of normality, while the attention map for a pseudo-abnormal frame is localized to the region of the inserted object. This is achieved by comparing the classifier’s attention map with a ground truth mask of the inserted object and penalizing deviations. The loss encourages the classifier to base its decision on the correct visual cues, enhancing interpretability and robustness.

- detail relative attention affirmation loss function

The relative attention affirmation loss function extends the attention affirmation loss by enforcing a margin between the similarity of attention maps for normal frames and augmented normal frames (via geometric transformations) and the similarity between normal frames and pseudo-abnormal frames. It is implemented using the ArcFace loss, which treats attention maps as embeddings in a hypersphere and enforces angular margins between classes. This loss enhances intra-class compactness and inter-class separation, ensuring that the classifier learns fine-grained distinctions between subtle deviations and true anomalies.

- explain joint training of neural networks

The generator, discriminator, memory module, and normalcy classifier are trained jointly in an end-to-end manner. Gradients from all loss functions—reconstruction, discriminative, normalcy, relative normalcy, attention affirmation, and relative attention affirmation—are backpropagated simultaneously to update all parameters. This joint optimization ensures that the generator produces frames that are not only realistic but also challenging for the classifier to distinguish from pseudo-abnormal frames, thereby forcing the classifier to learn increasingly robust relative normalcy features.

- describe video anomaly detection process

During inference, the system receives a sequence of video frames from a target domain and predicts the next frame using the trained generator. The predicted frame is compared with the actual frame to compute a reconstruction error. The actual frame is then passed through the normalcy classifier, which outputs a normalcy score. The anomaly score for the frame is computed as a weighted combination of the reconstruction error and the inverse of the normalcy score. Frames with scores exceeding a predefined threshold are flagged as anomalous. No training, adaptation, or data from the target domain is required during this process.

- introduce FIG. 2A

FIG. 2A illustrates the architecture of the future frame prediction module, showing the flow of input frames through the generator, memory module, and discriminator. The generator consists of an encoder, bottleneck with memory, and decoder, with skip connections between corresponding layers. The memory module is depicted as a key-value matrix, and the discriminator is shown as a patch-based classifier evaluating local regions of the predicted frame.

- describe future frame prediction module

The future frame prediction module takes as input a sequence of four consecutive video frames and outputs a single predicted frame. The encoder extracts hierarchical features from the input frames, which are then compressed and stored in the memory module. The decoder reconstructs the predicted frame by retrieving relevant memory slots and upscaling the features. Skip connections preserve spatial details, and adversarial training ensures temporal consistency and visual realism.

- detail generator architecture

The generator architecture is based on a U-Net structure with residual blocks and skip connections. The encoder consists of four convolutional layers with increasing channel depth, followed by a bottleneck containing the memory module. The decoder mirrors the encoder with transposed convolutions and skip connections. The memory module contains 128 slots, each storing a 512-dimensional feature vector. The architecture is designed to balance reconstruction fidelity with computational efficiency.

- explain memory module functionality

The memory module functions as a dynamic, sparse storage mechanism that captures recurring patterns of normal activity. During inference, the attention mechanism computes a relevance score for each memory slot, and only the top-k slots are used for reconstruction. The memory is updated during training via gradient descent, allowing it to adaptively learn the most discriminative normal patterns. The use of a hard-shrinkage function ensures sparsity, preventing overfitting and improving generalization.

- describe decoder architecture

The decoder reconstructs the predicted frame by upsampling the latent representation from the memory module and combining it with features from the encoder via skip connections. Each upsampling layer is followed by a convolutional block that refines spatial details. The final layer applies a tanh activation to constrain pixel values to the range [-1, 1], matching the input normalization.

- introduce FIG. 2B

FIG. 2B illustrates the architecture of the normalcy classifier module, showing its input, internal layers, and output. The classifier is a PatchGAN discriminator with additional attention extraction and loss computation modules. The attention maps are derived from the last convolutional layer and used to compute the four novel loss functions.

- describe normalcy classifier module

The normalcy classifier module is a convolutional network that takes a single frame as input and outputs a scalar normalcy score. It consists of a series of convolutional layers, batch normalization, and ReLU activations, followed by a global average pooling layer and a fully connected output layer. The attention maps are extracted from the final convolutional layer before pooling and are used to compute the attention affirmation and relative attention affirmation losses.

- detail object-aware anomaly synthesis module

The object-aware anomaly synthesis module receives a video frame and an untrained CNN as inputs. The CNN produces an attention map, which is thresholded to generate a binary mask. The mask is resized to a random scale and positioned at a random location within a normal frame. The pixels under the mask are replaced with the corresponding pixels from the source frame, creating a pseudo-abnormal frame. The module operates independently of the classifier and requires no training.

- explain randomly initialized CNN

The randomly initialized CNN is a standard convolutional network, such as ResNet152 or DenseNet161, with weights initialized using standard methods (e.g., He initialization) and left untrained throughout the entire system lifecycle. Despite lacking training, the network’s architecture induces a bias toward highlighting textured, structured regions (foreground objects) due to the hierarchical nature of convolutional filters. This property enables object localization without any labeled data or supervised training.

- describe object-aware cut mix operation

The object-aware cut mix operation is the core mechanism of the anomaly synthesis module. It involves cutting out a foreground object from one frame using a binary mask and pasting it onto another frame at a random location and scale. The operation is performed in pixel space and does not involve any semantic understanding of the object. The randomness in location and scale ensures diversity in pseudo-anomalies, preventing the classifier from learning spurious correlations.

- introduce FIG. 2C

FIG. 2C illustrates the architecture of the normalcy classifier, highlighting the attention extraction process and the four loss functions. The figure shows how attention maps are computed from the classifier’s final convolutional layer and used to compute the normalcy, relative normalcy, attention affirmation, and relative attention affirmation losses.

- describe normalcy classifier architecture

The normalcy classifier architecture consists of five convolutional layers, each followed by batch normalization and ReLU activation. The final layer is a 1×1 convolution that reduces the feature map to a single channel, which is then globally averaged to produce a scalar output. The attention maps are extracted from the feature map immediately before the global average pooling layer.

- detail normalcy loss function

The normalcy loss function is defined as the binary cross-entropy between the classifier’s output and the ground truth labels (1 for normal frames, 0 for pseudo-abnormal frames). It is computed as the mean over all training pairs and serves as the primary classification objective.

- explain relative normalcy loss function

The relative normalcy loss function is computed as the difference between the classifier’s output for normal frames and pseudo-abnormal frames, with a margin enforced to ensure that normal frames are consistently scored higher. It is formulated as a hinge loss that penalizes cases where the normal frame score is not sufficiently greater than the pseudo-abnormal frame score.

- introduce attention affirmation loss function

The attention affirmation loss function ensures that the classifier’s attention map for a normal frame is spatially diffuse, covering the entire scene, while the attention map for a pseudo-abnormal frame is localized to the region of the inserted object. It is computed as the L1 distance between the attention map and a ground truth mask of the object region.

- detail relative attention affirmation loss function

The relative attention affirmation loss function is implemented using the ArcFace loss, which treats attention maps as embeddings in a hypersphere and enforces angular margins between classes. It compares the attention map of a normal frame with that of an augmented normal frame and with that of a pseudo-abnormal frame, ensuring that the former are closer in angular distance than the latter.

- explain attention map extraction

Attention maps are extracted by applying spatial channel-wise density averaging to the feature map from the final convolutional layer of the normalcy classifier. Each spatial location is assigned a value equal to the sum of its feature channels, producing a 2D heatmap that indicates which regions of the frame most influenced the classifier’s decision.

- describe attention affirmation loss function

The attention affirmation loss function penalizes the classifier when its attention map for a normal frame fails to cover the entire scene or when its attention map for a pseudo-abnormal frame fails to localize to the inserted object. It is computed as the sum of two terms: one encouraging global attention for normal frames and another encouraging localized attention for pseudo-abnormal frames.

- introduce FIG. 3

FIG. 3 depicts examples of pseudo-abnormal frames generated by the object-aware anomaly synthesis module, showing foreground objects extracted from task-irrelevant datasets and pasted onto normal frames from the source domain. The figure illustrates the diversity of object types, sizes, and placements used to train the classifier.

- describe pseudo abnormal frame representation

Pseudo-abnormal frames are represented as pixel-level modifications of normal frames, where a foreground object from one frame is inserted into another frame at a random location and scale. The inserted object retains its original pixel values and is blended seamlessly into the background. The resulting frame contains a semantically plausible anomaly that is visually distinct from normal activity but does not correspond to any real-world anomaly.

- detail object localization process

Object localization is performed by applying a randomly initialized CNN to a video frame and computing an attention map via spatial channel-wise summation. The attention map is thresholded at a fixed value (e.g., 0.1) to produce a binary mask. The mask is then resized to match the spatial dimensions of the original frame, and the corresponding region is extracted using element-wise multiplication.

- explain ground truth mask generation

Ground truth masks for pseudo-abnormal frames are generated by recording the exact location, size, and shape of the inserted object during the synthesis process. These masks are used to compute the attention affirmation and relative attention affirmation losses by comparing them with the classifier’s attention maps.

- introduce FIG. 4

FIG. 4 illustrates the complete training workflow of the system, showing the flow of data from source domain videos through the generator, memory module, anomaly synthesis module, and normalcy classifier. The figure highlights the joint training loop and the feedback mechanisms between components.

- describe training system workflow

The training workflow begins with the sampling of a batch of normal frames from the source domain. The generator predicts the next frame, and the discriminator evaluates its realism. Simultaneously, the anomaly synthesis module extracts objects from source or task-irrelevant frames and creates pseudo-abnormal frames. These frames are fed into the normalcy classifier, which computes four loss functions. All losses are combined and backpropagated to update the parameters of all components in a single training step.

- detail future frame prediction process

The future frame prediction process involves encoding a sequence of four frames into a latent representation, retrieving relevant memory slots, and decoding the predicted frame. The process is repeated for each batch during training, with the generator improving its predictions over time through adversarial feedback and reconstruction loss minimization.

- explain normalcy classification process

The normalcy classification process involves passing a frame through the classifier to obtain a normalcy score. The attention map is extracted from the final convolutional layer and used to compute the four loss functions. The classifier is trained to assign high scores to normal frames and low scores to pseudo-abnormal frames, with the relative losses ensuring that the distinction is context-aware and robust.

- describe video anomaly detection output

The output of the video anomaly detection system is a sequence of anomaly scores, one for each frame in the input video. These scores are normalized to the range [0,1], with higher values indicating higher likelihood of anomaly. The system may output a binary decision (anomalous or normal) based on a threshold, or a continuous anomaly curve for temporal analysis.

## CLAIMS

[Claims to be drafted separately in accordance with patent office requirements]