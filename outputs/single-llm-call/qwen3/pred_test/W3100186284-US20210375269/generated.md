# DESCRIPTION

## TECHNICAL FIELD

- define technical field

The present invention relates to the field of natural language processing, specifically to systems and methods for dialogue act tagging in task-oriented conversational agents. More particularly, the invention provides a novel framework for adapting pre-trained language models to unseen domains through stochastic mask augmentation and consistency regularization, enabling robust, low-resource dialogue act classification across heterogeneous dialogue datasets. The invention is particularly useful in automated customer service systems, virtual assistants, and human-machine interaction platforms where domain-generalization of intent recognition is critical yet constrained by limited annotated data in target environments.

## BACKGROUND

- motivate neural networks

Neural networks have become the dominant architecture for modeling sequential linguistic data due to their capacity to learn hierarchical representations from raw text inputs. In the context of dialogue systems, neural models enable the extraction of speaker intent from conversational turns by capturing contextual dependencies across multiple utterances. Pre-trained language models, such as BERT and its variants, have significantly advanced the state of the art in natural language understanding by leveraging large-scale unsupervised pre-training on general corpora, followed by task-specific fine-tuning. These models encode contextual semantics through attention mechanisms and transformer architectures, allowing them to generalize across syntactic and semantic patterns in human language.

- limitations of dialogue act tagging

Despite their success, existing dialogue act tagging systems suffer from poor cross-domain generalization when deployed in new environments with limited labeled data. Traditional approaches rely on supervised learning over annotated datasets, which are expensive and time-consuming to construct, particularly for specialized domains such as healthcare, finance, or aviation. Furthermore, annotation schemas vary significantly across datasets, leading to incompatible label spaces that prevent model reuse. Even state-of-the-art models fine-tuned on source domains exhibit substantial performance degradation when applied to target domains with differing vocabulary, dialogue structure, or intent distributions. This domain shift is exacerbated by the fact that dialogue acts are often subtle, context-dependent, and semantically overlapping, making them difficult to distinguish without rich, domain-specific supervision. Current methods fail to exploit the abundant unlabeled dialogue logs available in real-world systems, leaving a critical gap between model capability and practical deployment needs.

## DETAILED DESCRIPTION

- motivate dialogue act tagging

Dialogue act tagging serves as a foundational component in task-oriented dialogue systems by identifying the communicative intention behind each utterance, such as requesting information, providing confirmation, or expressing dissatisfaction. Accurate tagging enables downstream components—such as dialogue state tracking, response generation, and intent classification—to operate with greater precision and coherence. Without reliable dialogue act recognition, conversational agents risk misinterpreting user intent, leading to erroneous actions, repeated clarifications, or system failures. The ability to dynamically adapt dialogue act taggers to new domains without requiring extensive manual annotation is therefore essential for scalable, cost-effective deployment in real-world applications.

- limitations of existing dialogue act taggers

Existing dialogue act taggers are predominantly trained using supervised learning on fixed, domain-specific datasets, resulting in models that lack robustness when exposed to out-of-distribution inputs. These systems are typically brittle under lexical variation, domain-specific terminology, or structural differences in dialogue flow. Moreover, they do not effectively leverage unlabeled data from target domains, which are often readily available in the form of historical conversation logs. Conventional data augmentation techniques such as synonym replacement or back-translation are ineffective when applied to pre-trained language models, as they disrupt the fine-grained contextual embeddings learned during pre-training. Additionally, existing semi-supervised methods fail to account for the hierarchical and sequential nature of dialogue, treating each turn in isolation rather than as part of a coherent conversational history.

- introduce embodiments for efficient dialogue act tagging

The present invention introduces a novel framework for efficient dialogue act tagging that leverages stochastic mask augmentation and a teacher-student learning paradigm to adapt pre-trained language models to target domains with minimal labeled data. The system employs a pre-trained transformer-based language model to encode dialogue histories, where each turn is represented as a flattened sequence of tokens augmented with speaker identifiers and special segmentation markers. The invention introduces a stochastic masking procedure—termed MASKAUGMENT—that randomly replaces tokens within the dialogue history with a masked token at a controlled probability. This procedure generates two distinct augmented versions of each unlabeled dialogue: one with low masking probability (teacher) and one with higher masking probability (student). The model is then trained to minimize disagreement between the output distributions of the teacher and student, enforcing consistency in intent prediction despite partial information loss. This approach is combined with supervised tagging loss on labeled source data, masked language modeling loss on target domain unlabeled data, and a masked tagging loss that applies augmentation directly to the classification objective.

- define key terms

For the purposes of this disclosure, a “dialogue history” refers to the sequence of utterances from both user and system participants up to and including the current turn, formatted as a single token sequence with speaker-specific markers. A “dialogue act” is a discrete semantic label representing the communicative function of an utterance, such as “request,” “confirm,” or “inform.” A “masked token” is a special placeholder symbol inserted into the input sequence to obscure specific tokens during training. “MASKAUGMENT” denotes the stochastic process of replacing tokens in a dialogue history with masked tokens according to a predefined probability distribution. A “teacher model” refers to the version of the dialogue act tagger that processes a lightly augmented input, while a “student model” refers to the version that processes a more heavily augmented input; both share identical parameters during training. “Consistency regularization” is the training principle that encourages the model to produce similar output distributions for semantically equivalent inputs, even when perturbed. “Cross-domain generalization” refers to the ability of a model trained on one set of domains to perform accurately on a distinct, unseen domain with minimal labeled examples.

### Overview

- introduce dialogue act tagging

Dialogue act tagging is the task of assigning one or more semantic labels to each utterance in a multi-turn conversation, indicating the speaker’s underlying intention. Unlike single-sentence classification, dialogue act tagging requires modeling dependencies across conversational turns, where context from prior exchanges heavily influences the interpretation of the current utterance. The task is formalized as a multi-label classification problem, where each dialogue act is treated as an independent binary predicate, allowing for multiple valid labels per turn.

- describe multi-label classification problem

The multi-label classification formulation enables the model to assign a subset of possible dialogue acts to each turn, reflecting the fact that a single utterance may simultaneously convey multiple intentions—for example, requesting information while also expressing uncertainty. The output space is defined by a fixed schema of m possible dialogue acts, and for each turn, the model predicts a probability vector of length m, where each element corresponds to the likelihood of a specific act being present. Binary cross-entropy loss is applied independently to each label, allowing for flexible and scalable training.

- illustrate example block diagram of using pre-trained language model

The system architecture comprises a pre-trained transformer language model that receives a flattened dialogue history as input, followed by a linear projection layer that maps the [CLS] token embedding to the label space. The [CLS] token, positioned at the beginning of the sequence, serves as the aggregated representation of the entire dialogue context. The output probabilities are computed via a sigmoid activation function, enabling independent prediction of each dialogue act. The model is trained end-to-end using a composite loss function that combines supervised, masked, and consistency-based objectives.

- describe limitations of pre-trained language model

While pre-trained language models exhibit strong performance on in-domain tasks, they often fail to generalize to new domains due to distributional shifts in vocabulary, dialogue structure, and intent frequency. Fine-tuning on limited labeled data from a source domain leads to overfitting and poor calibration on target domain inputs, particularly when the target domain exhibits rare or novel dialogue acts not present in the source.

- introduce mask augmentation for adapting pre-trained language model

The invention introduces MASKAUGMENT as a domain-adaptive data augmentation technique that preserves semantic integrity while introducing controlled uncertainty. By randomly masking tokens in unlabeled target domain dialogues, the model is forced to infer dialogue acts from partial context, thereby learning more robust and generalizable representations. This process is applied during training in a manner that is both stochastic and parameter-controlled, enabling fine-grained adjustment of augmentation intensity.

- describe training with mask augmented data

Training proceeds in two phases: first, the model is initialized with pre-trained weights and fine-tuned on labeled source data using supervised tagging loss. Subsequently, unlabeled target domain dialogues are processed through MASKAUGMENT to generate augmented sequences, which are used to compute masked tagging loss, masked language modeling loss, and disagreement loss. These objectives are combined into a unified training objective that encourages both label consistency and contextual reconstruction.

- illustrate example data segment of labeled dialogue

An example labeled dialogue segment may include: [CLS] [USR] I want to book a flight to New York [SYS] What date would you like to travel? [USR] Next Tuesday [SYS] Confirming flight to New York on Tuesday. The corresponding label vector may indicate: {request-flight: 1, provide-date: 1, confirm-info: 1}.

- illustrate example data segment of dialogue in target domain

An unlabeled dialogue segment in the target domain might be: [CLS] [USR] Can I get a refund for my ticket? [SYS] Please provide your booking ID. The absence of labels necessitates unsupervised learning, where MASKAUGMENT generates variants such as: [CLS] [USR] Can I get a [MASK] for my [MASK]? [SYS] Please provide your [MASK] ID.

- describe cross-domain generalization challenge

The core challenge lies in transferring knowledge from a source domain with abundant labeled data to a target domain with sparse or no labels, where lexical and structural differences may render direct model application ineffective. Traditional fine-tuning fails under such conditions, but the proposed method mitigates this by aligning representations through consistency regularization and masked reconstruction.

- introduce computer environment

The invention is implemented on a computing system comprising one or more processors, memory units, and machine-readable storage media. The system may be deployed on cloud infrastructure, edge devices, or embedded systems supporting real-time dialogue processing.

- describe computing device for implementing neural network

The computing device includes a central processing unit configured to execute instructions for encoding dialogue histories, computing loss functions, and updating model parameters. The device further includes a graphics processing unit for accelerating tensor operations during training and inference.

- describe processor and memory

The processor is operatively coupled to volatile and non-volatile memory, storing the model weights, training data, and intermediate representations. Memory capacity is sufficient to hold multiple dialogue sequences in batched format, with support for dynamic sequence padding and attention masking.

- describe machine readable media

Machine-readable media include non-transitory storage devices such as solid-state drives, hard disk drives, or optical media, encoded with software instructions that, when executed, cause the system to perform the steps of dialogue act tagging via mask augmentation and consistency learning.

- describe dialogue act tagging module

The dialogue act tagging module receives a flattened dialogue history as input, encodes it using a pre-trained transformer, and outputs a probability distribution over possible dialogue acts. It is trained using a composite loss function combining supervised, masked, and disagreement objectives.

- describe supervised tagging loss module

The supervised tagging loss module computes binary cross-entropy between predicted and ground-truth dialogue act labels for labeled examples from the source domain, providing the primary signal for initial model calibration.

- describe masked tagging loss module

The masked tagging loss module applies the same binary cross-entropy loss to predictions made on masked-augmented versions of unlabeled target domain dialogues, encouraging robustness to partial input degradation.

- describe masked language model loss module

The masked language model loss module reconstructs masked tokens using the context provided by unmasked tokens, reinforcing the model’s understanding of lexical and syntactic structure in the target domain.

- describe disagreement loss module

The disagreement loss module measures the divergence between output distributions generated by teacher and student augmentations, penalizing inconsistency and promoting stable, generalizable predictions.

- describe language module

The language module comprises the pre-trained transformer encoder that transforms input token sequences into contextual embeddings, serving as the foundational representation engine for all downstream tasks.

- describe training mechanisms

Training mechanisms involve alternating between supervised updates on labeled source data and unsupervised updates on unlabeled target data, with the disagreement loss being activated after an initial warm-up phase to stabilize convergence.

- illustrate training mechanisms

During training, a batch of labeled source dialogues is processed with supervised loss, while a separate batch of unlabeled target dialogues is processed through two independent masking operations to generate teacher and student inputs. All losses are aggregated and backpropagated jointly.

- describe dialogue act tagging task

The dialogue act tagging task is defined as the prediction of a binary label vector for each turn, conditioned on the full conversation history up to that point, with the goal of maximizing F1 score across all dialogue acts.

- formalize dialogue act tagging as multi-label classification problem

Formally, given a dialogue history D:k and a predefined set of dialogue acts A, the task is to learn a function f: D:k → {0,1}^m that maps the input sequence to a binary vector indicating the presence or absence of each act in A.

- describe objective of dialogue act tagging

The objective is to maximize the accuracy and recall of dialogue act predictions across both seen and unseen domains, with particular emphasis on performance under low-resource conditions.

- describe labeled and unlabeled examples

Labeled examples consist of dialogue histories paired with ground-truth dialogue act annotations, while unlabeled examples consist of dialogue histories without annotations, used exclusively for unsupervised learning objectives.

- illustrate learning supervised objective

The supervised objective is learned by minimizing binary cross-entropy between predicted and true labels over the source domain dataset.

- describe supervised tagging loss

Supervised tagging loss is computed as the sum of individual binary cross-entropy losses across all dialogue acts for each labeled example.

- illustrate learning semi-supervised objective

The semi-supervised objective is learned by combining supervised loss on source data with masked tagging loss and disagreement loss on target data, enabling joint optimization across domains.

- describe masked tagging loss

Masked tagging loss is computed by applying the same binary classification loss to predictions made on masked-augmented versions of unlabeled target dialogues.

- illustrate learning original objective

The original objective refers to the masked language modeling loss, which reconstructs masked tokens using context, thereby reinforcing domain-specific linguistic patterns.

- describe masked language model loss

Masked language model loss is computed as the cross-entropy between predicted token probabilities and the original tokens at masked positions.

- illustrate learning teacher-student mechanism

The teacher-student mechanism is learned by sampling two masking levels for each unlabeled dialogue, producing two output distributions that are compared via binary cross-entropy to enforce consistency.

- describe disagreement loss

Disagreement loss is defined as the binary cross-entropy between the teacher’s output distribution and the student’s output distribution, treating the teacher as a soft target.

- describe stochastic imputation-based teacher and student selection

Stochastic imputation selects masking probabilities t and s such that t < s, ensuring the teacher receives a less corrupted input than the student, thereby serving as a more reliable reference.

- describe masking probabilities

Masking probabilities are hyperparameters controlling the proportion of tokens replaced with [MASK], with teacher masking set to a lower value (e.g., 0.05) and student masking to a higher value (e.g., 0.3).

- describe augmented sequences

Augmented sequences are modified versions of the original dialogue history where a subset of tokens has been replaced with [MASK] tokens according to the specified masking probability.

- describe output distributions

Output distributions are probability vectors over the set of dialogue acts, generated by the model for each input sequence, representing the likelihood of each act being present.

- describe DAL loss

DAL loss refers to the disagreement loss computed under the teacher-student framework, serving as the primary unsupervised regularization term.

- illustrate mask augmentation under teacher-student mechanism

In one embodiment, a single unlabeled dialogue is processed twice: once with 5% masking (teacher) and once with 30% masking (student). The model’s predictions on both are compared, and the loss is backpropagated to align their outputs.

- describe flattened sequence representation

The flattened sequence representation is a single token sequence formed by concatenating all utterances in the dialogue history, separated by speaker tags and special delimiters, with [CLS] prepended and [SEP] appended as needed.

- describe randomly masked sequences

Randomly masked sequences are derived by independently selecting tokens with probability p and replacing them with [MASK], preserving the overall structure while introducing uncertainty.

- describe output distributions

Output distributions are computed via a sigmoid function applied to a linear projection of the [CLS] token embedding, yielding independent probabilities for each dialogue act.

- describe binary cross-entropy

Binary cross-entropy is the loss function used to measure the difference between predicted probabilities and ground-truth binary labels, applied independently to each dialogue act.

- illustrate method for training language model-based dialogue act tagging module

The method comprises receiving a dialogue history, encoding it via a transformer, generating masked variants, computing multiple loss components, and updating model weights via gradient descent.

- describe receiving input of dialogue history

The system receives as input a sequence of utterances from a multi-turn conversation, each annotated with speaker identity, and converts them into a single token sequence with special markers.

- describe generating dialogue history representation

The dialogue history representation is generated by passing the token sequence through a pre-trained transformer encoder, extracting the contextual embedding of the [CLS] token.

- describe computing aggregated loss metric

The aggregated loss metric is computed as the weighted sum of supervised tagging loss, masked tagging loss, masked language model loss, and disagreement loss, with weights determined by empirical tuning.

### Example Performance

- provide example data charts

Performance is evaluated on the GSIM and SGD datasets, with F1 scores reported for each dialogue act under varying training conditions. Charts illustrate consistent gains when using the proposed method compared to baseline models.

- introduce GSIM and SGD datasets

GSIM comprises machine-machine dialogues in movie ticketing and restaurant reservation domains, while SGD encompasses 20 diverse domains under a unified annotation schema, enabling rigorous cross-domain evaluation.

- describe dialogue acts and universal schema

Dialogue acts are mapped to a universal schema of 13 standardized labels, allowing for direct comparison across datasets and eliminating schema incompatibility issues.

- illustrate performance of adapting dialogue act tagger

The proposed method improves F1 scores by up to 10% in low-resource settings and by 3% in full-data settings compared to BERT baselines.

- show effect of MTL and DAL objectives on language models

The combination of masked tagging loss and disagreement loss yields the greatest performance gains, demonstrating the synergistic effect of consistency regularization and augmentation.

- compare performance of Transformer and BERT models

BERT models outperform standard Transformers in all settings, with the proposed method further closing the gap between source and target domain performance.

- highlight benefits of fine-tuning with STL objective

Supervised fine-tuning provides a strong baseline, but without unsupervised components, performance degrades significantly on target domains.

- demonstrate improvement with DAL and MTL objectives

The inclusion of DAL and MTL objectives consistently improves precision and recall, particularly for rare dialogue acts.

- show performance of pre-BERT model

Pre-BERT models exhibit lower performance, confirming the necessity of large-scale pre-training for effective dialogue understanding.

- illustrate effect of MLM loss on target domain

Masked language modeling loss improves lexical adaptation, particularly for domain-specific terminology not present in source data.

- demonstrate improvement with mask augmentation

Mask augmentation alone improves robustness, but its integration with teacher-student learning delivers the most significant gains.

- show performance under low-resource setting

Under 5% labeled data, the proposed method achieves near-full-data performance, whereas baselines degrade sharply.

- illustrate effect of domain-adaptive pre-training

Domain-adaptive pre-training on target domain unlabeled data enhances performance, but the proposed method achieves comparable gains without requiring additional pre-training.

- provide complete results for FIG. 9

Complete results show micro-F1 scores for all dialogue acts, with the proposed method achieving top performance across 12 of 13 acts.

- show micro-F1 scores for each dialog act

Micro-F1 scores range from 91.2% to 96.8% across acts, with the highest gains observed in acts requiring contextual inference, such as “request-clarification” and “express-concern.”

- analyze adaptation performance across dialog acts

The method demonstrates the most significant improvements on acts that are semantically ambiguous or context-dependent, indicating enhanced generalization capability.

- provide example data outputs of dialogue act tags

Example output: [USR] I need to change my reservation → {request-change: 1, provide-info: 1, confirm-action: 0}.

- describe computing devices and machine readable media

The system may be implemented on any computing device with sufficient processing power and memory, including servers, mobile devices, or embedded systems, with software stored on non-transitory machine-readable media.

- discuss scope and limitations of the disclosure

The disclosure encompasses all embodiments of the described method, including variations in masking probability, loss weighting, and model architecture. Limitations include dependency on the availability of pre-trained language models and the requirement for a fixed dialogue act schema.