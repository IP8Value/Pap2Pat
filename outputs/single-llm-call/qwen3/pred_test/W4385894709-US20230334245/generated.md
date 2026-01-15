# DESCRIPTION

## CROSS REFERENCE

- claim priority

This application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/XXXXXX, filed on [Insert Filing Date], entitled “Efficient Zero-Shot Text Classification Using Conformal Prediction Framework,” the entire disclosure of which is hereby incorporated by reference in its entirety. The present application is directed to a novel method and system for improving the computational efficiency of zero-shot text classification models through the application of conformal prediction techniques. The invention leverages a two-stage classification architecture wherein a fast, lightweight base classifier is employed to generate a reduced set of plausible class labels, which is then passed to a more accurate but computationally intensive zero-shot model for final prediction. This approach ensures that the predictive performance of the full zero-shot model is preserved while significantly reducing the number of label evaluations required during inference. The framework is model-agnostic and can be applied to any zero-shot classification system that relies on scoring text-label pairs, including those based on natural language inference, next-sentence prediction, prompt-based generation, or in-context learning. The invention further encompasses the use of calibration datasets derived from model-generated labels, eliminating the need for human-annotated training data in practical deployment scenarios. The method described herein enables scalable deployment of zero-shot classifiers in resource-constrained environments, reduces latency in real-time applications, and lowers the environmental cost associated with large-scale inference operations.

## TECHNICAL FIELD

- define technical field

The present invention relates generally to the field of natural language processing and machine learning, and more particularly to systems and methods for efficient zero-shot text classification. Zero-shot text classification refers to the task of assigning predefined semantic categories to textual inputs without requiring task-specific labeled training data. This capability is particularly valuable in dynamic environments where new classes emerge frequently, labeling is prohibitively expensive, or domain adaptation is required across heterogeneous data sources. The invention specifically addresses the computational inefficiencies inherent in current zero-shot classification architectures that rely on cross-encoder models such as those based on natural language inference or next-sentence prediction formulations. These models require a separate forward pass through a large transformer-based language model for each candidate label, resulting in linear scaling of computational cost with the number of possible classes. The invention introduces a conformal prediction-based filtering mechanism that reduces the number of labels evaluated by the primary zero-shot model, thereby enabling faster, more energy-efficient, and scalable deployment without sacrificing classification accuracy or coverage guarantees.

## BACKGROUND

- motivate text classification

Text classification is a foundational task in natural language processing with widespread applications across customer service automation, content moderation, sentiment analysis, intent recognition in virtual assistants, document categorization, and multilingual information retrieval. The ability to classify text into predefined categories without relying on labeled training data has become increasingly important as organizations seek to rapidly adapt to new domains, languages, and evolving terminology. Zero-shot classification models have emerged as a powerful solution to this challenge, enabling systems to generalize to unseen classes by leveraging semantic relationships encoded in pre-trained language models. These models typically represent each class as a natural language hypothesis and evaluate the logical relationship between the input text and each hypothesis using a pre-trained transformer architecture. While highly accurate, this approach incurs substantial computational overhead, especially when the number of possible classes is large, making real-time deployment in high-throughput environments impractical.

- limitations of zero-shot models

Current zero-shot classification models suffer from significant inefficiencies due to their reliance on full self-attention mechanisms that process each text-label pair independently. For datasets with dozens or hundreds of potential labels, this results in hundreds of forward passes through large transformer models, leading to prohibitive inference latency and high energy consumption. Furthermore, the computational burden scales linearly with the number of target classes, rendering these models unsuitable for applications requiring low-latency responses or deployment on edge devices. Existing approaches to accelerate inference, such as pruning, quantization, or distillation, often compromise model accuracy or require retraining, which defeats the purpose of zero-shot generalization. No prior method provides a theoretically grounded, model-agnostic mechanism to reduce the number of labels evaluated during inference while maintaining strict statistical guarantees on classification coverage. As a result, there remains a critical unmet need for an efficient, scalable, and reliable framework that can reduce the computational footprint of zero-shot classification without altering the underlying model architecture or requiring additional labeled data.

## DETAILED DESCRIPTION

- define network

The network architecture of the invention comprises a modular system designed to facilitate efficient zero-shot text classification through conformal prediction. The network includes a first component responsible for generating a reduced set of candidate labels using a fast base classifier, and a second component responsible for performing the final classification using a high-accuracy zero-shot model. These components are interconnected through a communication interface that transmits the input text and the filtered label set between modules. The network may operate on a single computing device or be distributed across multiple interconnected systems, including client devices, cloud servers, and data vendor platforms. The network is configured to receive textual input, process it through the conformal prediction module, and return a final classification result with a statistically bounded error rate. The architecture is designed to be compatible with existing zero-shot classification frameworks and can be integrated as a preprocessing layer without requiring modifications to the underlying language models.

- define module

Each functional unit within the system is implemented as a discrete module, including a conformal prediction module, a base classifier module, a zero-shot classification module, a calibration data processor, and a user interface module. The conformal prediction module receives the input text and computes non-conformity scores across all possible labels using the base classifier’s output. It then determines a reduced label set by comparing these scores against a quantile threshold derived from a calibration dataset. The base classifier module executes a computationally lightweight model, such as a distilled transformer, a token overlap scorer, or a GloVe-based cosine similarity estimator, to generate initial label predictions. The zero-shot classification module receives the reduced label set and performs a full evaluation using a high-capacity model such as BART-large-NLI or BERT-base-NSP. Each module operates independently and may be replaced or upgraded without affecting the overall system architecture, enabling continuous improvement and adaptation to new classification tasks.

- introduce zero-shot classification models

Zero-shot classification models operate by encoding both the input text and each candidate class label into a shared semantic space and computing a similarity or logical relationship score between them. These models typically employ transformer-based architectures that have been pre-trained on massive corpora and fine-tuned on tasks such as natural language inference or sentence pair classification. For each input, the model evaluates the text against every possible label hypothesis, generating a confidence score for each pairing. The label with the highest score is selected as the predicted class. While these models achieve state-of-the-art accuracy, their computational cost grows linearly with the number of labels, making them impractical for applications involving hundreds or thousands of potential categories.

- describe limitations of existing models

Existing zero-shot classification models are constrained by their inability to scale efficiently with the number of target classes. Each label requires a separate forward pass through a large transformer model, resulting in high latency and substantial energy consumption. This limitation restricts their deployment in real-time systems, mobile applications, and environments with limited computational resources. Furthermore, no existing method provides a principled way to reduce the number of labels evaluated without risking the omission of the true class. Traditional pruning techniques rely on heuristic thresholds or statistical approximations that lack theoretical guarantees, leading to unpredictable performance degradation. The absence of a model-agnostic, coverage-preserving filtering mechanism represents a fundamental barrier to the widespread adoption of zero-shot classification in scalable, production-grade systems.

- motivate need for efficient zero-shot classification

The increasing demand for adaptive, low-resource text classification systems necessitates a paradigm shift in how zero-shot models are deployed. As the number of possible classes in real-world applications continues to grow—ranging from intent detection in customer service chatbots to multi-label document categorization in legal and medical domains—the computational cost of evaluating every label becomes prohibitive. Efficient classification is no longer merely a performance optimization but a critical requirement for sustainability, scalability, and user experience. The invention addresses this need by introducing a conformal prediction framework that guarantees classification coverage while reducing the number of labels evaluated by up to 43%, thereby enabling faster inference, lower energy consumption, and broader applicability across resource-constrained environments.

- introduce Conformal Predictor (CP) framework

The invention introduces a conformal predictor framework that operates as a pre-filtering layer for zero-shot classification models. This framework generates a set of candidate labels instead of a single prediction, ensuring that the true label is included within this set with a probability no less than 1−α, where α is a user-defined error rate. The framework is built upon the theoretical foundations of conformal prediction, which provides distribution-free, finite-sample validity guarantees without assumptions about the underlying data distribution. By integrating this framework with a fast base classifier, the invention enables the elimination of unlikely labels prior to evaluation by the primary zero-shot model, significantly reducing computational overhead while preserving accuracy.

- describe CP framework components

The conformal predictor framework comprises three core components: a base classifier, a calibration dataset, and a quantile-based decision rule. The base classifier is a computationally inexpensive model that generates logits or similarity scores for each label given an input text. The calibration dataset consists of input-text pairs and their corresponding model-generated labels, used to compute non-conformity scores that quantify the disagreement between predicted and true labels. The quantile-based decision rule determines a threshold score such that only labels with non-conformity scores below this threshold are retained for evaluation by the zero-shot model. This threshold is computed as the ⌈(n+1)(1−α)⌉/n empirical quantile of the calibration scores, ensuring statistical coverage guarantees.

- introduce fast base classifier

The invention employs a fast base classifier to compute non-conformity scores efficiently. This classifier may be implemented using a distilled BERT model, a token overlap metric, or a cosine similarity measure based on static word embeddings such as GloVe. Unlike the primary zero-shot model, the base classifier is designed for minimal computational cost, enabling rapid evaluation across all candidate labels. The choice of base classifier is flexible and may be selected based on the trade-off between speed and label set size. For example, token overlap provides the fastest inference but yields larger label sets, while a fine-tuned distilled transformer offers smaller label sets at the cost of slightly higher computational overhead.

- describe calibration dataset

The calibration dataset is constructed using model-generated labels from the zero-shot classifier itself, eliminating the need for human-annotated data. This approach ensures that the calibration samples are exchangeable with the test data and aligned with the decision boundary of the primary model. The dataset may be drawn from unlabeled text collected in the target domain and labeled using the zero-shot model’s predictions. The size of the calibration dataset is shown to have minimal impact on performance when a low error rate (e.g., α = 0.01) is selected, allowing for efficient deployment even with small amounts of calibration data.

- generate predictions using base classifier

For each input text, the base classifier generates a vector of scores corresponding to each possible class label. These scores may represent logit values from a neural network, token overlap percentages, or cosine similarities between text and label embeddings. The scores are then used to compute non-conformity scores, which quantify how atypical each label is with respect to the input text under the base classifier’s predictions.

- compute non-conformity scores

Non-conformity scores are computed as the negative of the base classifier’s score for each label. For example, if the base classifier assigns a logit score of 2.3 to the true label, the non-conformity score is −2.3. Higher non-conformity scores indicate greater disagreement between the base classifier’s prediction and the candidate label, suggesting that the label is less plausible. These scores are collected across the calibration dataset to establish a reference distribution.

- describe quantile computation

The quantile threshold q is computed as the ⌈(n+1)(1−α)⌉/n empirical quantile of the non-conformity scores from the calibration dataset, where n is the number of calibration samples and α is the desired error rate. This quantile defines the boundary below which labels are considered sufficiently plausible to be retained for evaluation by the zero-shot model. The use of empirical quantiles ensures that the framework provides distribution-free coverage guarantees without requiring assumptions about the underlying data distribution.

- generate reduced label set

The reduced label set is generated by selecting all labels whose non-conformity scores are less than the computed quantile threshold. This set contains the true label with probability at least 1−α, as guaranteed by the theory of conformal prediction. The size of this set is typically much smaller than the full label set, often reducing the number of labels by 30–45% depending on the base classifier and dataset characteristics.

- use reduced label set with zero-shot model

The reduced label set is passed as input to the zero-shot classification model, which evaluates only the retained labels. This reduces the number of forward passes required, directly decreasing inference latency and computational cost. The zero-shot model retains its original architecture and parameters, ensuring that its predictive accuracy is preserved.

- describe ensemble of class label descriptions

The class label descriptions are represented as natural language phrases that capture the semantic essence of each category. These descriptions are used by the zero-shot model to form hypothesis statements. The invention supports the use of multiple descriptions per label, forming an ensemble that enhances the robustness of the zero-shot model’s scoring mechanism.

- describe cosine-similarity-based non-conformity scores

Cosine-similarity-based non-conformity scores are computed by representing both the input text and each label description as bag-of-words vectors using static word embeddings such as GloVe. The cosine similarity between these vectors is computed, and the non-conformity score is defined as one minus this similarity. This approach is particularly effective for domains with sparse or ambiguous vocabulary, where semantic similarity provides a more robust signal than exact token matching.

- describe distilled BERT-base model

The distilled BERT-base model is employed as a base classifier to generate high-quality, low-latency label predictions. This model is a compressed variant of BERT-base, retaining 95% of the original model’s performance while reducing inference time by over 50%. It is fine-tuned on model-generated labels from the zero-shot classifier and outputs logits that serve as the basis for non-conformity score computation.

- describe another parameter-efficient NLI zero-shot model

In addition to BART-large-NLI, the invention is compatible with other parameter-efficient zero-shot models such as DeBERTa-v3, RoBERTa-base, or T5-small, provided they operate by scoring text-label pairs. The conformal prediction framework is agnostic to the specific architecture of the zero-shot model, allowing for seamless integration with future advancements in zero-shot classification.

- discuss model-agnostic nature of CP framework

The conformal prediction framework is entirely model-agnostic, meaning it can be applied to any zero-shot classification system that outputs a score for each label given an input text. It does not require modifications to the underlying model, retraining, or architectural changes. This makes the invention broadly applicable across domains, languages, and model types, including prompt-based, in-context learning, and image-text classification systems.

- describe application to prompt-based few-shot classification models

The framework can be applied to prompt-based few-shot models that generate verbalizers for each class. By reducing the number of candidate labels, the invention reduces the number of verbalizers that must be evaluated, thereby accelerating inference in models that require autoregressive generation for each label.

- describe application to in-context learning

In in-context learning, the invention reduces the number of training examples included in the prompt by filtering out unlikely classes. This minimizes the length of the prompt, reducing the computational cost of context encoding and improving inference speed without compromising performance.

- describe application to image classification models

The framework extends to image classification by applying conformal prediction to text-based label descriptions of visual classes. For example, in zero-shot image classification using CLIP, the framework reduces the number of text labels evaluated against image embeddings, improving efficiency in large-scale visual categorization tasks.

- introduce computing device for implementing CP framework

The invention may be implemented on a general-purpose computing device comprising a processor, memory, and non-volatile storage. The device executes executable code that implements the conformal prediction module, base classifier, and zero-shot classification model. The device may be a server, desktop computer, mobile device, or embedded system, and may operate independently or as part of a distributed network.

- describe efficient zero-shot classification module

The efficient zero-shot classification module integrates the conformal prediction framework with a zero-shot classification model to produce a final classification result with reduced computational cost. The module receives textual input, generates a reduced label set using the base classifier and calibration data, and passes this set to the zero-shot model for final scoring. The module outputs the predicted label along with a confidence measure derived from the zero-shot model’s scores.

- describe networked system for implementing CP framework

The invention may be deployed in a networked system comprising client devices, server nodes, and data vendor platforms. Client devices send text inputs to a server hosting the conformal prediction module, which returns the classification result. The server may be connected to databases storing calibration datasets, label descriptions, and model weights. Communication between components occurs via secure network protocols, enabling scalable, cloud-based deployment of efficient zero-shot classification services.

### Computer Environment

- introduce computing device

The computing device upon which the invention may be implemented includes a central processing unit, random-access memory, storage devices, input/output interfaces, and a communication interface. The device may be a server, workstation, laptop, smartphone, or embedded system, and is configured to execute the executable code necessary to perform the steps of the conformal prediction framework.

- describe processor and memory

The processor is configured to execute instructions stored in memory, including code for generating non-conformity scores, computing quantile thresholds, filtering label sets, and invoking zero-shot classification models. Memory includes both volatile and non-volatile components, storing calibration data, model weights, and intermediate computational results.

- describe operation of computing device

The computing device operates by receiving input text, loading the base classifier and zero-shot model from memory, generating non-conformity scores using the calibration dataset, computing the quantile threshold, filtering the label set, and returning the final classification result to the user or application.

- describe machine-readable media

Machine-readable media include non-transitory storage devices such as solid-state drives, hard disk drives, optical discs, and flash memory, which store the executable code, calibration datasets, and model parameters required for operation.

- describe executable code

Executable code comprises software instructions that, when executed by a processor, cause the device to perform the steps of the conformal prediction framework, including calibration, quantile computation, label filtering, and zero-shot classification.

- introduce efficient zero-shot classification module

The efficient zero-shot classification module is a software component that orchestrates the interaction between the base classifier, conformal prediction logic, and zero-shot model. It manages data flow, handles error conditions, and ensures compliance with coverage guarantees.

- describe data interface

The data interface receives textual inputs from external sources, such as user queries, documents, or sensor logs, and formats them for processing by the base classifier.

- describe user interface

The user interface allows end users to submit text inputs, view classification results, adjust error rates, and monitor system performance.

- describe communication interface

The communication interface enables the device to transmit and receive data over a network, facilitating integration with cloud services, mobile applications, and enterprise systems.

- introduce conformal prediction module

The conformal prediction module is responsible for computing non-conformity scores, determining the quantile threshold, and generating the reduced label set. It operates independently of the zero-shot model and may be updated or replaced without affecting the overall system.

- describe NLI/NSP classifier module

The NLI/NSP classifier module implements the zero-shot classification model using natural language inference or next-sentence prediction formulations. It receives the reduced label set and computes final scores for each label.

- describe implementation of modules

Modules are implemented as software components using object-oriented programming principles, allowing for modular development, testing, and deployment. Each module exposes well-defined interfaces for data exchange and configuration.

- introduce networked system

The networked system comprises multiple interconnected computing devices, including user devices, data vendor servers, and centralized servers hosting the conformal prediction framework.

- describe user device

The user device is a client endpoint such as a smartphone, tablet, or desktop computer that submits text classification requests and receives results from the server.

- describe data vendor servers

Data vendor servers provide labeled or unlabeled text data used for calibration and model training, and may host precomputed label descriptions and embedding vectors.

- describe server

The server hosts the conformal prediction module, zero-shot classification model, and calibration datasets, and responds to classification requests from user devices.

- describe network

The network is a secure, high-bandwidth communication infrastructure such as the Internet, private enterprise networks, or cellular data networks, enabling data transmission between components.

- describe user interface application

The user interface application is a software program, web service, or mobile app that allows users to interact with the classification system, view results, and configure parameters.

- describe other applications

Other applications include customer service chatbots, content moderation systems, automated document classification tools, and multilingual information retrieval engines.

- describe database

The database stores calibration datasets, label descriptions, model weights, and historical classification logs for auditing and performance monitoring.

- describe network interface component

The network interface component enables communication between the computing device and external systems, supporting protocols such as HTTP, gRPC, or WebSocket.

- describe data vendor server

The data vendor server provides labeled or unlabeled text data for calibration, and may offer APIs for downloading label descriptions and embedding vectors.

- describe database

The database is a structured repository for storing calibration data, model configurations, and performance metrics, accessible via secure query interfaces.

- describe network interface component

The network interface component facilitates secure, low-latency communication between the computing device and remote servers, ensuring reliable data transmission.

- describe server

The server executes the conformal prediction framework and serves classification results to multiple clients simultaneously, supporting load balancing and failover mechanisms.

- describe efficient zero-shot classification module

The efficient zero-shot classification module is a core component of the server, responsible for coordinating the entire classification pipeline and ensuring compliance with statistical guarantees.

- describe network interface component

The network interface component enables the server to receive incoming classification requests and transmit results to client devices over encrypted channels.

### Work Flows

- introduce method for efficient zero-shot text classification

The method for efficient zero-shot text classification comprises the steps of receiving a calibration dataset containing text samples and corresponding model-generated labels, computing non-conformity scores using a base classifier, determining a quantile threshold based on the desired error rate, generating a reduced label set for each input text, and applying the zero-shot classification model to the reduced set to produce a final prediction.

- receive calibration dataset with texts and labels

The calibration dataset is received from a storage medium or network source and contains pairs of input texts and their corresponding labels, which may be generated by a zero-shot model or other classification system.

- generate predicted labels using base classifier model

For each text in the calibration dataset, the base classifier generates a set of predicted scores for all possible labels, which are used to compute non-conformity scores.

- compute non-conformity scores by comparing predicted labels and calibration labels

Non-conformity scores are computed as the negative of the base classifier’s score for the calibration label, quantifying the degree of disagreement between the base classifier’s prediction and the true label.

- compute non-conformity threshold based on non-conformity scores and error rate

The non-conformity threshold is computed as the empirical quantile of the calibration scores corresponding to the desired error rate, ensuring statistical coverage guarantees.

- generate predicted testing label using base classifier model

For each input text in the testing set, the base classifier generates a score for each possible label, which is used to compute non-conformity scores for filtering.

- generate second set of non-conformity scores by comparing predicted testing label and classification labels

The non-conformity scores for the testing set are computed using the same method as for the calibration set, ensuring consistency in the filtering process.

- determine reduced set of classification labels based on non-conformity scores and threshold

The reduced label set is determined by retaining only those labels whose non-conformity scores are below the computed threshold, ensuring that the true label is included with probability at least 1−α.

### Example Data Experiment and Performance

- evaluate CP-based framework on intent classification datasets

The framework was evaluated on three intent classification datasets: SNIPS, ATIS, and HWU64, which contain 7, 17, and 64 labels respectively. The conformal prediction framework consistently reduced the average number of labels evaluated by the zero-shot model while maintaining or improving classification accuracy.

- use moderately sized BART-large as zero-shot classification model

The BART-large-NLI model served as the primary zero-shot classifier, providing high accuracy across all datasets. The conformal prediction framework was applied as a preprocessing layer to reduce the number of labels evaluated by this model.

- use small BERT-base as base classifier in CP framework

A distilled BERT-base model was used as the base classifier, achieving a favorable trade-off between computational efficiency and label set size reduction.

- calibrate CP-Token, CP-Glove, and CP-Distil using training set and validation set

Three base classifiers—CP-Token, CP-Glove, and CP-Distil—were calibrated using the training and validation sets of each dataset. Each was evaluated for empirical coverage and average label set size.

- train CP-CLS base classifier using training set and validation set

The CP-CLS base classifier was fine-tuned on model-generated labels from the zero-shot model, achieving the smallest average label set size across all datasets.

- show empirical coverage and average label set size of four base classifiers

All four base classifiers achieved empirical coverage equal to or greater than the nominal coverage at α = 0.01, demonstrating the validity of the conformal prediction guarantees. CP-CLS consistently produced the smallest label sets.

- compare accuracy, average inference time, and average label set size of CP framework

The CP framework reduced average inference time by up to 25.6% and reduced average label set size by up to 43.38% compared to the full zero-shot model, with no significant loss in accuracy.

- observe CP framework achieves valid coverage

The framework consistently achieved valid coverage across all datasets and base classifiers, confirming the theoretical guarantees of conformal prediction.

- observe CP reduces average number of labels for zero-shot model

The average number of labels evaluated by the zero-shot model was reduced by over 40% in all cases, directly translating to faster inference.

- observe fine-tuning base classifier reduces average number of labels

Fine-tuned base classifiers such as CP-CLS and CP-Distil produced smaller label sets than non-parametric approaches like CP-Token and CP-Glove.

- observe CP-Token achieves best inference time on some datasets

CP-Token achieved the fastest overall inference time on datasets with smaller label spaces due to its minimal computational overhead.

- observe CP-Distil improves inference time on some datasets

CP-Distil improved inference time on datasets with large label spaces by reducing the number of labels evaluated, despite its slightly higher computational cost.

- observe CP improves efficiency on datasets with many labels

The efficiency gains were most pronounced on datasets with 64 labels, demonstrating the framework’s suitability for high-cardinality classification tasks.

- observe CP performs comparable to zero-shot model

Classification accuracy of the CP framework was statistically indistinguishable from that of the full zero-shot model, with some cases showing slight improvements.

- observe CP-based label filtering retains performance of corresponding models

The performance of the zero-shot model was preserved or enhanced after label filtering, suggesting that the framework removes noisy or ambiguous labels.

- conclude CP framework is effective for efficient zero-shot text classification

The conformal prediction framework is a robust, scalable, and model-agnostic solution for improving the efficiency of zero-shot text classification systems. It provides statistically valid coverage guarantees while significantly reducing computational cost, making it suitable for real-world deployment in dynamic, high-throughput environments.