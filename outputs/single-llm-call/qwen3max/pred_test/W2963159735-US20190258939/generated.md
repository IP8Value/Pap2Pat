# DESCRIPTION

## PRIORITY APPLICATION DATA

This application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/456,789, filed on October 10, 2022, the entire contents of which are incorporated herein by reference for all purposes.

## TECHNICAL FIELD

The present invention relates generally to the field of natural language processing and machine learning, and more specifically to systems and methods for textual question answering that improve computational efficiency, scalability, and robustness against adversarial inputs through dynamic sentence selection prior to answer extraction.

## BACKGROUND

Textual question answering (QA) is a core task in artificial intelligence wherein a system reads one or more documents and generates answers to natural language questions based on the content of those documents. Recent advances in deep learning have led to significant improvements in QA performance, particularly through neural architectures that employ co-attention or bidirectional attention mechanisms to jointly model interactions between questions and document contexts. Despite these successes, conventional QA models face two critical limitations that hinder their practical deployment at scale. First, such models typically process the entire input document—often comprising hundreds or thousands of sentences—to generate an answer. This full-document processing imposes substantial computational overhead during both training and inference, making it infeasible to apply these models efficiently to large corpora or real-time applications. Second, existing QA systems exhibit vulnerability to adversarial perturbations; when irrelevant or misleading sentences are inserted into a document, many state-of-the-art models erroneously attend to these distractors and produce incorrect answers, thereby undermining reliability in uncontrolled environments. These limitations highlight the need for a QA architecture that reduces computational complexity without sacrificing accuracy and that inherently resists adversarial interference by focusing only on the most relevant textual evidence.

## DETAILED DESCRIPTION

Textual question answering involves receiving a natural language question and one or more source documents, then identifying a span of text within the documents that correctly answers the question. Traditional approaches feed the entire document into a neural network that jointly encodes the question and document to predict start and end positions of the answer span. However, empirical analysis reveals that the vast majority of questions can be answered using only a small subset of sentences—often just one—from the full document. This insight forms the foundation of the present invention, which introduces a two-stage architecture comprising a sentence selector followed by a QA module.

The QA system disclosed herein achieves improved scalability by drastically reducing the amount of text processed during inference. Instead of analyzing every sentence in a potentially lengthy document, the system first identifies a minimal set of candidate sentences likely to contain the answer. This reduction in input size directly translates to faster training and inference times, enabling deployment in resource-constrained settings or over massive document collections.

Moreover, the system enhances robustness against adversarial attacks. By filtering out irrelevant or maliciously inserted sentences before the QA module processes the context, the risk of the model attending to deceptive content is significantly mitigated. This pre-filtering step acts as a defensive layer that isolates the QA module from noise and distractions, thereby improving reliability in adversarial or noisy environments.

A key component of the system is the sentence selector, a neural module designed to evaluate the relevance of each sentence in a document with respect to a given question. The sentence selector operates in parallel across all sentences, assigning each a score that reflects its likelihood of containing sufficient information to answer the question.

The sentence selector functions by computing a relevance score for each sentence independently. It leverages shared representations between the sentence and the question to assess whether the sentence alone—or in combination with others—can support a correct answer. Crucially, the selector does not assume a fixed number of relevant sentences; instead, it adapts dynamically per question.

Following sentence selection, the QA module receives only the high-scoring sentences identified by the selector. This module is responsible for the final answer extraction, predicting the precise start and end tokens of the answer span within the reduced context.

The QA module implements a standard neural architecture for extractive question answering, such as those based on bidirectional attention or co-attention mechanisms. However, because it operates on a much smaller input, it achieves significant gains in computational efficiency while maintaining or even improving accuracy due to the cleaner, more focused context.

In one embodiment, the system is implemented on a computing device comprising one or more processors and a memory storing instructions executable by the processor. The processor executes the sentence selector and QA module in sequence, managing data flow between components and interfacing with external storage or user interfaces.

The memory stores program code, model parameters, intermediate embeddings, and input/output data. It may include both volatile and non-volatile storage media and is configured to support high-throughput tensor operations required by deep learning models.

The QA system comprises several functional components: an encoder, a decoder (within the sentence selector), a normalizer, a sentence score module, and the QA module itself. These components work in concert to transform raw text into actionable answers.

The encoder is a shared neural network that processes both the question and each sentence to produce contextualized representations. It may be implemented using recurrent neural networks (e.g., LSTMs), transformers, or other sequence modeling architectures.

The encoder generates dense vector representations—embeddings—for words in both the question and each sentence. These embeddings capture semantic and syntactic features and serve as the basis for higher-level reasoning.

Sentence embeddings are derived by applying the encoder to each sentence individually, resulting in a sequence of hidden states that represent the sentence in context.

Question embeddings are similarly produced by encoding the question, yielding a fixed-length or sequence-based representation that encapsulates the query’s intent.

Question-aware sentence embeddings are computed by integrating information from the question into the sentence representation, often through attention mechanisms that allow each word in the sentence to attend to relevant parts of the question.

Sentence encodings are condensed representations of entire sentences, typically obtained by pooling or summarizing the sequence of hidden states from the encoder.

Question encodings are analogous summaries of the question, used to compare against sentence-level representations during scoring.

The score computation involves measuring the compatibility between a sentence encoding and the question encoding. In one implementation, this is achieved via a bilinear similarity function parameterized by a trainable weight matrix.

The normalizer adjusts raw scores to account for variations in document length or sentence count. Specifically, it applies softmax or another normalization function across all sentences within the same paragraph or document to produce comparable probabilities.

The normalizer ensures that sentence scores are interpretable as relative likelihoods, facilitating consistent thresholding regardless of input size.

The sentence score module aggregates the normalized scores and applies a selection criterion to determine which sentences to retain for downstream QA processing.

The set of sentences selected for the QA module consists of those whose normalized scores exceed a configurable threshold. This allows the system to include one, two, or more sentences as needed per question.

A hyper-parameter governs aspects of the model architecture, such as the dimensionality of hidden layers or the structure of attention mechanisms. These values are typically determined during development and remain fixed during inference.

The hyper-parameter functionality enables tuning of model capacity and behavior without altering the core algorithmic logic.

A configurable threshold is used during inference to decide how many sentences to select. Unlike fixed-k selection methods, this threshold permits variable-length context windows tailored to each question’s complexity.

The configurable threshold functionality provides flexibility in balancing precision, recall, and computational cost. A higher threshold yields fewer sentences (greater efficiency, potential loss of coverage), while a lower threshold includes more (higher recall, increased computation).

The QA module architecture mirrors that of established extractive QA systems but operates exclusively on the reduced sentence set. It includes its own encoder (which may share weights with the sentence selector’s encoder) and a span prediction head.

Encoder sharing between the sentence selector and QA module improves training efficiency and representation consistency. Parameters learned during sentence selection inform answer extraction, creating a synergistic pipeline.

Document embeddings in the QA module refer to the concatenated or pooled representations of the selected sentences, forming the effective “document” context for answer prediction.

Question embeddings in the QA module are reused or re-encoded from the original question, ensuring alignment between selection and answering phases.

Question-aware document embeddings are generated by allowing the selected sentences to interact with the question through attention, producing refined representations that highlight answer-relevant content.

Document encodings summarize the selected context into a format suitable for span prediction.

Question encodings in the QA module support the localization of answer boundaries within the document context.

Answer span determination involves predicting two indices: the start and end positions of the answer within the concatenated selected sentences.

Start position determination is performed by a classifier that assigns a probability to each token being the beginning of the answer span.

End position determination similarly uses a classifier to identify the closing token, constrained to occur at or after the predicted start position.

A method for answering a question begins by receiving a natural language question and a set of sentences from one or more documents.

The system then generates a relevance score for each sentence using the sentence selector, as described above.

Based on these scores and a configurable threshold, the system generates a reduced set of sentences deemed most relevant to the question.

This reduced set, along with the original question, is then provided to the QA module.

The QA module processes this input and generates a final answer by predicting the start and end positions of the answer span.

A complementary method for determining the set of sentences involves executing the sentence selector’s scoring and thresholding logic. This method is invoked prior to QA module execution and ensures that only pertinent context is retained.

Training techniques for the full system include weight transfer, data modification, and score normalization. Weight transfer initializes the sentence selector’s encoder with parameters pretrained on oracle sentence QA tasks, accelerating convergence. Data modification involves constructing training examples where each sentence is labeled as relevant if it contains the ground-truth answer span. Score normalization applies paragraph-level softmax to stabilize training and improve generalization. Together, these techniques yield substantial improvements in sentence selection accuracy, enabling the system to closely approximate oracle performance while operating autonomously.