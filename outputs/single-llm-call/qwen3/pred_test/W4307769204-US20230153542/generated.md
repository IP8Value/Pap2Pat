## CROSS REFERENCE(S)

- claim priority

This invention claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/XXXXXX, filed on [Insert Filing Date], entitled “System and Method for Cross-Lingual Sentence Alignment Using a Normalized BERT-Score Architecture,” the entirety of which is hereby incorporated by reference. The present application is a non-provisional application seeking full patent protection for the novel framework, training methodology, and computational architecture described herein, which were first conceived and reduced to practice in the course of research conducted at [Insert Institution Name]. All subject matter disclosed in the provisional application, including but not limited to the use of a pre-trained multilingual language model, the integration of a supervised BERT-score computation module, the implementation of an in-batch normalization layer, and the training protocol employing contrastive learning with in-batch negatives, is hereby fully asserted as part of the inventive disclosure. No portion of the invention disclosed herein was publicly disclosed, offered for sale, or otherwise made available to the public more than one year prior to the effective filing date of this application. The inventors affirm that all claims presented herein are supported by the original disclosure and are not directed to subject matter that was added after the filing of the provisional application.

## TECHNICAL FIELD

- define technical field

The present invention relates generally to natural language processing systems and, more specifically, to methods and apparatuses for cross-lingual sentence alignment in multilingual environments. The invention provides a computational framework for identifying semantically equivalent sentence pairs across different human languages without requiring parallel training data for every target language pair. This technology is particularly useful in applications involving machine translation, cross-lingual information retrieval, document alignment in multilingual corpora, and quality filtering of automatically mined parallel text from the web. The system leverages pre-trained multilingual language models and introduces architectural innovations that enable high-accuracy alignment even when training is restricted to a minimal set of high-resource language pairs, thereby significantly reducing the dependency on scarce, low-resource parallel datasets. The invention operates in both supervised and zero-shot transfer settings, enabling scalable deployment across hundreds of language pairs without proportional increases in training data or computational cost.

## BACKGROUND

- motivate cross-lingual sentence alignment
- limitations of training data

The alignment of semantically equivalent sentences across different languages is a foundational task in multilingual natural language processing, essential for constructing high-quality parallel corpora to train machine translation systems, enabling cross-lingual knowledge transfer, and supporting applications such as multilingual search engines and document summarization across linguistic boundaries. Traditional approaches to this task have relied heavily on large-scale, manually curated parallel datasets, which are expensive to produce and exist in significant quantities only for a handful of dominant language pairs, such as English-French or English-Chinese. For the vast majority of the world’s languages, especially low-resource ones, such data is either unavailable or of insufficient quality to support robust model training. Alternative unsupervised methods, while avoiding the need for parallel data, suffer from low accuracy on benchmark tasks and lack the precision required for real-world deployment in critical applications. Furthermore, existing supervised models that attempt to generalize across many language pairs often require training on hundreds or even thousands of language pairs, leading to computational inefficiency, model capacity dilution, and diminished performance on individual pairs due to the heterogeneity of linguistic structures and data distributions. These limitations have created a persistent bottleneck in the development of truly global multilingual systems, where the availability of training data remains the primary constraint rather than algorithmic innovation. The present invention addresses these shortcomings by demonstrating that high-fidelity cross-lingual alignment can be achieved through a novel architecture trained on a single rich-resource language pair, eliminating the necessity for extensive data collection while simultaneously achieving state-of-the-art performance across diverse language families.

## DETAILED DESCRIPTION

- define network

The network of the present invention comprises a pre-trained multilingual transformer-based language model, configured to encode sentences from multiple languages into a shared semantic embedding space. This model is initialized with weights derived from a large-scale monolingual pre-training corpus covering over one hundred languages, ensuring that the underlying representations capture cross-lingual semantic relationships prior to any supervised fine-tuning. The network is further augmented with a specialized module for computing pairwise semantic similarity between source and target sentences, which operates on the final layer of the transformer encoder and incorporates token-level cross-attention without requiring full sequence-to-sequence encoding. The entire architecture is designed to be end-to-end trainable using contrastive learning objectives and is optimized for inference efficiency, enabling deployment in real-time systems with minimal latency.

- define module

The module of the present invention is a supervised BERT-score computation unit that calculates semantic similarity between two sentences by measuring the cosine similarity between each token embedding in the source sentence and each token embedding in the target sentence, followed by a precision and recall aggregation that accounts for the maximum similarity match in each direction. This module is distinct from conventional Siamese encoders that rely on mean-pooled sentence representations, as it preserves fine-grained token-level alignment signals while remaining agnostic to sentence order. The module is integrated directly into the forward pass of the network and is trained end-to-end with the transformer backbone, allowing the model to learn how to weight and interpret token-level similarities in a language-agnostic manner.

- define rich-source and low-resource

For the purposes of this invention, a rich-source language pair refers to a pair of languages for which a substantial volume of parallel training data is available, typically exceeding ten thousand aligned sentence pairs, and for which the corresponding monolingual corpora used in pre-training contain millions of sentences. In contrast, a low-resource language pair denotes a pair of languages for which parallel data is sparse, often fewer than one thousand aligned pairs, and for which the monolingual pre-training data is limited or of poor quality. The invention demonstrates that performance on low-resource language pairs can be achieved without direct training on such pairs, provided that the model is trained on one or more rich-source language pairs and that the monolingual pre-training data for the target languages meets a minimum size threshold.

- motivate cross-lingual sentence alignment

Cross-lingual sentence alignment is motivated by the need to identify equivalent semantic content across languages without relying on human annotation or expensive translation services. In practical applications such as web-scale document mining, legal document comparison, or multilingual customer support systems, the ability to automatically detect sentence-level translations enables the construction of parallel corpora at unprecedented scale. The invention addresses the critical challenge that existing methods either require prohibitively large amounts of parallel data or fail to generalize accurately to unseen language pairs, thereby limiting their utility in real-world multilingual environments.

- describe limitations of existing systems

Existing systems for cross-lingual sentence alignment suffer from several critical limitations. First, many rely on a Siamese architecture with mean-pooled sentence representations, which discard fine-grained lexical and syntactic alignment signals. Second, systems that employ full cross-attention between source and target sentences are computationally prohibitive for large-scale inference. Third, most models require training on hundreds of language pairs to achieve robust cross-lingual transfer, which is impractical given the scarcity of parallel data for most languages. Fourth, existing approaches are heavily biased toward English-centered language pairs, resulting in poor performance on non-English-centric alignments. Fifth, no prior system incorporates a normalization mechanism to counteract the popular sentence effect, wherein certain sentences consistently receive high similarity scores regardless of their actual semantic equivalence.

- introduce proposed framework

The proposed framework introduces a novel architecture for cross-lingual sentence alignment that combines a pre-trained multilingual transformer with a supervised BERT-score similarity module and an in-batch normalization layer. The framework is trained using contrastive learning on positive sentence pairs drawn from a single rich-source language pair, with all other sentence pairs in the same batch treated as negative examples. This approach enables the model to learn a global decision boundary for alignment without requiring any parallel data for the target language pairs. The framework is scalable, data-efficient, and achieves state-of-the-art performance on multiple benchmark datasets, including Tatoeba-36 and New-Tatoeba, even when trained on less than one million sentence pairs.

- describe use of pre-trained multi-lingual language model

The framework employs a pre-trained multilingual language model that has been initialized with weights learned from a massive corpus of monolingual text spanning over one hundred languages. This model serves as the sentence encoder, transforming each input sentence into a sequence of contextualized token embeddings. The choice of this model ensures that the system inherits strong cross-lingual representation capabilities without requiring additional pre-training or language-specific adaptations. The model is frozen during initial training phases and later fine-tuned jointly with the similarity and normalization modules, allowing the entire system to adapt its representations to the alignment task while retaining the generalization power of the pre-trained weights.

- describe supervised version of BERT-score

The supervised version of BERT-score is a modified implementation of the original unsupervised metric, adapted for end-to-end training within a neural network framework. Instead of computing similarity scores as a post-processing step, the BERT-score is computed as a differentiable operation within the forward pass of the model, using token-level cosine similarities between source and target embeddings. The precision and recall components are computed as weighted averages of maximum similarities, and the resulting score is treated as a logit for classification. This modification enables the model to learn how to adjust token-level matching behavior during training, leading to more accurate and language-independent alignment decisions.

- describe normalization layer

The normalization layer is an architectural component that subtracts a scaled average of pairwise similarities between each sentence and all other sentences in the target language within the same batch. This operation counteracts the popular sentence effect, wherein certain sentences are erroneously assigned high similarity scores across many candidates due to stylistic or structural biases. The scaling factor is learned during training and applied consistently during both training and inference. The normalization is performed in-batch to maintain computational efficiency and does not require access to external statistics or large reference corpora.

- describe training on rich-resource language pairs

Training is conducted exclusively on parallel sentence pairs drawn from one or more rich-resource language pairs, with a fixed budget of one million aligned examples. The model is not exposed to any low-resource language pairs during training, yet it achieves high performance on evaluation sets containing dozens of low-resource language pairs. This demonstrates that the model learns transferable alignment patterns from rich-resource data, and that the quality of alignment is determined more by the size of the monolingual pre-training corpus than by the diversity of training language pairs.

- describe scaling up with top-k rich-resource language pairs

The framework can be extended by training on the top-k rich-resource language pairs ranked by the size of their parallel corpora. As k increases from one to thirty-two, performance on benchmark datasets improves monotonically, reaching within 0.4 percentage points of the performance achieved when training on all available language pairs. This indicates that marginal gains in alignment accuracy can be obtained by expanding the training set to include additional high-resource pairs, but that the majority of performance is captured by training on a single pair, making the approach highly cost-effective.

- describe training without English as an anchor language

The framework is trained successfully on language pairs that do not include English as either the source or target language. Performance on English-centered evaluation sets remains consistent with models trained on English-centered pairs, demonstrating that the alignment capability is not dependent on English as a pivot language. This enables the deployment of the system in multilingual regions where English is not dominant, and supports the creation of non-English-centered multilingual systems.

- illustrate training framework for alignment model

The training framework receives a batch of source and target sentences, where each source sentence is paired with its true translation as a positive example. All other target sentences in the batch are treated as negative examples. The model computes pairwise BERT-scores between every source and target sentence, applies in-batch normalization, and computes a contrastive loss using cross-entropy over the resulting logits. The model parameters are updated via backpropagation, and this process is repeated over multiple epochs until convergence.

- describe input sentences

Input sentences are tokenized using the vocabulary of the pre-trained multilingual language model and truncated or padded to a maximum length of one hundred tokens. Sentences containing fewer than five tokens are filtered out to remove noise. The input format does not require special tokens such as [CLS] or [SEP], and the source and target sentences are processed independently before being fed into the similarity module.

- describe alignment model

The alignment model is a neural architecture composed of a pre-trained multilingual transformer encoder, a supervised BERT-score computation module, and an in-batch normalization layer. The model outputs a scalar similarity score for each source-target sentence pair, which is used to rank candidate translations. The model is trained to distinguish true translations from distractors using contrastive learning, and it generalizes to unseen language pairs without additional training.

- describe embedding model

The embedding model is the pre-trained multilingual transformer that generates contextualized token embeddings for each input sentence. These embeddings are derived from the final layer of the transformer and are not modified by additional layers. The model’s pre-training on a diverse monolingual corpus ensures that the embeddings capture cross-lingual semantic relationships, enabling the alignment model to operate effectively even with minimal supervised training.

- describe BERT score computation module

The BERT score computation module calculates the pairwise cosine similarity between every token in the source sentence and every token in the target sentence. It then computes precision as the average of the maximum similarities for each source token and recall as the average of the maximum similarities for each target token. The harmonic mean of precision and recall is taken as the final similarity score, which is used as a logit in the classification task.

- describe normalization layer

The normalization layer computes, for each sentence in the source batch, the average similarity score between that sentence and all sentences in the target batch. This average is scaled by a learned parameter and subtracted from each individual similarity score, reducing the influence of sentences that are consistently rated highly due to stylistic bias. The normalization is applied in-batch to ensure computational efficiency and is differentiable, allowing gradients to flow through the operation during training.

- illustrate BERT score computation

The BERT score computation begins by encoding both sentences using the transformer encoder. For each token in the source sentence, the model computes its cosine similarity with every token in the target sentence. The highest similarity for each source token is recorded, and similarly for each target token. The precision score is the mean of the highest source-to-target similarities, and the recall score is the mean of the highest target-to-source similarities. The final BERT score is the harmonic mean of these two values.

- describe recall score computation

Recall is computed as the arithmetic mean of the maximum cosine similarity values obtained between each token in the target sentence and all tokens in the source sentence. This measures how well the target sentence is covered by the source sentence’s vocabulary and semantic structure.

- describe precision score computation

Precision is computed as the arithmetic mean of the maximum cosine similarity values obtained between each token in the source sentence and all tokens in the target sentence. This measures how well the source sentence’s content is captured by the target sentence’s representation.

- describe BERT score computation

The BERT score is computed as the harmonic mean of the precision and recall scores, ensuring that both directional alignment qualities are equally weighted. This score serves as the primary similarity metric used by the model to determine whether two sentences are translations of each other.

- describe normalization step

The normalization step subtracts a scaled average of all pairwise similarities involving a given sentence from its individual similarity score. This adjustment reduces the bias introduced by sentences that are stylistically or structurally common across the corpus, ensuring that the final similarity ranking reflects true semantic equivalence rather than frequency or popularity.

- describe inference stage

During inference, the model receives a source sentence and a set of candidate target sentences. It computes the BERT score for each candidate pair, applies the in-batch normalization using the batch of candidates, and ranks the candidates by their normalized scores. The highest-ranked candidate is selected as the aligned translation. The process is performed in a single forward pass and requires no additional lookup tables or external resources.

- describe contrastive learning approach

The contrastive learning approach treats each aligned sentence pair as a positive example and all other pairs within the same batch as negative examples. The model is trained to assign higher similarity scores to positive pairs and lower scores to negative pairs using a cross-entropy loss over the logits derived from the normalized BERT scores. This approach enables the model to learn a global alignment threshold without requiring explicit negative samples or external datasets.

- describe in-batch negatives

In-batch negatives are all sentence pairs in the training batch that are not the true translation pair. For a batch of N source and N target sentences, there are N² total pairs, of which N are positive and N²−N are negative. The model is trained to distinguish the N positive pairs from the N²−N negative pairs simultaneously, enabling the learning of a global alignment decision boundary.

- describe pairwise semantic similarity computation

Pairwise semantic similarity is computed using the BERT-score module, which evaluates the token-level cosine similarities between every pair of tokens from the source and target sentences. The resulting matrix of similarities is aggregated into a single scalar value using precision and recall metrics, which are then normalized to produce the final similarity score.

- describe contrastive loss computation

The contrastive loss is computed as the cross-entropy loss over the logits derived from the normalized BERT scores. Each positive pair is assigned a label of 1, and all negative pairs are assigned a label of 0. The loss is computed across the entire batch, encouraging the model to maximize the margin between positive and negative scores.

- illustrate method of training aligner model

The method of training begins by sampling a batch of aligned sentence pairs from a rich-resource language pair. Each source sentence is encoded using the pre-trained transformer, and each target sentence is encoded independently. The BERT-score module computes pairwise token-level similarities, and the normalization layer adjusts these scores to counteract bias. The contrastive loss is computed over all pairs in the batch, and gradients are backpropagated to update the transformer weights and the normalization parameter. This process is repeated for multiple epochs until the loss converges.

- receive training dataset

The training dataset consists of aligned sentence pairs from one or more rich-resource language pairs, with a total size not exceeding one million pairs. Each pair is preprocessed to remove sentences with fewer than five tokens, and the dataset is shuffled before each training epoch to ensure uniform exposure to all examples.

- form positive and negative input pairs

Positive input pairs are formed by matching each source sentence with its corresponding translation in the target language. Negative input pairs are formed by pairing each source sentence with every target sentence in the same batch that is not its true translation. This results in one positive pair and N²−N negative pairs per batch of size N.

- compute pairwise token-level similarity

Pairwise token-level similarity is computed by taking the cosine similarity between every token embedding in the source sentence and every token embedding in the target sentence, resulting in a matrix of size M×N, where M and N are the lengths of the source and target sentences, respectively.

- compute loss objective

The loss objective is the cross-entropy loss over the normalized BERT scores, where each positive pair is assigned a label of 1 and all other pairs are assigned a label of 0. The loss is averaged over the entire batch and used to update the model parameters.

- update pre-trained multi-lingual model

The pre-trained multilingual model is updated via stochastic gradient descent with a learning rate of 3e-6 and a batch size of 64. The model weights are fine-tuned jointly with the BERT-score and normalization modules, allowing the entire system to adapt to the alignment task while preserving the cross-lingual representation capabilities of the pre-trained weights.

- perform alignment task

The alignment task is performed by encoding a source sentence and a set of candidate target sentences, computing the normalized BERT score for each candidate, and selecting the candidate with the highest score as the aligned translation. The task is performed without requiring any language-specific rules or external dictionaries.

- describe computing device for implementing cross-lingual sentence alignment

The computing device for implementing cross-lingual sentence alignment comprises a central processing unit, a graphics processing unit with at least 40 gigabytes of memory, and a non-transitory computer-readable storage medium storing instructions that, when executed, cause the device to perform the steps of encoding sentences, computing BERT scores, applying normalization, and performing contrastive learning. The device is configured to operate in both batch and real-time modes, supporting deployment in cloud servers and edge computing environments.

- describe processor and memory

The processor is a high-performance multi-core CPU or GPU capable of executing parallel tensor operations at high throughput. The memory includes high-bandwidth dynamic random-access memory sufficient to store the pre-trained model weights, input sentence embeddings, and intermediate similarity matrices for batches of up to 256 sentence pairs. The memory is organized to minimize data transfer latency between the processor and storage.

- describe paraphrase generation module

The paraphrase generation module is an optional component that generates synthetic parallel sentence pairs by paraphrasing source sentences in the same language and translating them into the target language using external systems. These synthetic pairs are used to augment the training dataset when real parallel data is limited, but are not required for the core functionality of the invention.

- describe data interface and output

The data interface accepts input sentences in plain text or tokenized form and outputs a ranked list of candidate translations with associated similarity scores. The output is formatted as a JSON object containing the source sentence, the top-k aligned candidates, their normalized BERT scores, and a confidence metric derived from the score distribution. The interface supports RESTful API calls and batch processing for integration into downstream applications.

### Example Performance

- describe aligner model training

The aligner model is trained using a single A100 GPU with a batch size of 64 and a learning rate of 3e-6 over three epochs. Training is conducted on a fixed budget of one million aligned sentence pairs drawn from rich-resource language pairs. The model converges within 24 hours and achieves state-of-the-art performance on benchmark datasets without requiring any additional data or architectural complexity.

- introduce OPUS-100 and Tatoeba Challenge datasets

The OPUS-100 dataset provides a collection of parallel sentence pairs across 100 languages, with each language pair capped at one million examples. The Tatoeba Challenge dataset is a benchmark consisting of 36 language pairs, each containing approximately one thousand test sentence pairs, designed to evaluate cross-lingual sentence retrieval performance on both high- and low-resource languages.

- explain data removal from OPUS-100

To ensure clean training data, any sentence pair from OPUS-100 containing fewer than five tokens in either the source or target language is removed. Additionally, any examples that overlap with sentences in the Tatoeba or BUCC evaluation sets are excluded to prevent data leakage.

- show performance results in FIGS. 5-13

Performance results, as illustrated in FIGS. 5 through 13, demonstrate that the aligner model achieves an accuracy of 94.9% on the Tatoeba-36 dataset when trained on all language pairs from OPUS-100, outperforming prior models by 8.0 percentage points. When trained on a single rich-resource language pair, the model achieves performance within 2.0 percentage points of the full model, and when trained on the top 32 rich-resource pairs, performance reaches 94.0%, only 0.4 points below the full model.

- introduce cross-lingual sentence retrieval tasks

Cross-lingual sentence retrieval tasks involve identifying the correct translation of a given sentence from a set of candidate sentences in another language. These tasks are evaluated using ranking metrics such as accuracy at top-1 and mean reciprocal rank, and are commonly assessed on datasets such as Tatoeba and BUCC.

- describe Tatoeba dataset from XTREME benchmark

The Tatoeba dataset from the XTREME benchmark contains 36 language pairs, each with approximately one thousand test sentence pairs. The dataset includes low-resource languages such as Javanese and Swahili, making it a challenging benchmark for cross-lingual transfer.

- introduce v2021-08-07 Tatoeba Challenge dataset

The v2021-08-07 Tatoeba Challenge dataset is an expanded version of the original Tatoeba benchmark, containing 223 language pairs with at least one thousand test examples each. This dataset includes both English-centered and non-English-centered language pairs and is used to evaluate the generalization capability of the aligner model across diverse linguistic contexts.

- explain language pair selection for evaluation

Language pairs are selected for evaluation based on the availability of at least one thousand aligned test examples and the presence of both languages in the CC-100 pre-training corpus. This ensures that the evaluation is conducted on languages for which the model has sufficient pre-training exposure.

- describe BUCC 2018 dataset

The BUCC 2018 dataset consists of four high-resource language pairs (English-German, English-French, English-Russian, and English-Chinese) and is designed as a bitext mining task where the goal is to identify true translations from a large pool of candidate sentences. It is used as a sanity check to verify that the model performs well on high-resource languages.

- introduce baseline models for comparison

Baseline models for comparison include VECO and ERNIE-M, both of which are trained on hundreds of language pairs and large-scale parallel corpora. These models represent the state of the art prior to the invention and are used to demonstrate the data efficiency and performance of the proposed framework.

- describe VECO and ERNIE-M models

VECO is a multilingual alignment model trained on 879 language pairs using 6.4 billion parallel sentences, while ERNIE-M is trained on 96 languages using a combination of public parallel corpora. Both models are built on the XLM-R architecture but require significantly more training data than the proposed framework.

- show basic stats of each model in FIG. 6

FIG. 6 presents a comparison of model statistics, including the number of training language pairs, the volume of parallel data, and the number of parameters. The proposed model achieves superior performance with fewer than one million training examples and a single language pair, in contrast to VECO and ERNIE-M, which require orders of magnitude more data.

- compare aligner model with baseline models

The aligner model outperforms VECO and ERNIE-M on the Tatoeba-36 dataset while using 180 times less training data. It also achieves higher accuracy on non-English-centered language pairs, demonstrating superior generalization and reduced bias toward English.

- show Tatoeba-36 performance in FIG. 5

FIG. 5 shows that the aligner model achieves 94.9% accuracy on Tatoeba-36 when trained on all language pairs, and 92.9% when trained on a single rich-resource pair, outperforming all baseline models by a significant margin.

- show data efficiency of aligner model

The aligner model demonstrates exceptional data efficiency, achieving near-state-of-the-art performance with only one million training examples and a single language pair, whereas prior models require hundreds of millions of examples to reach comparable accuracy.

- show consistent performance across language pairs in FIG. 7

FIG. 7 demonstrates that the aligner model exhibits consistent performance across all 36 language pairs in Tatoeba-36, regardless of whether the language is high-resource or low-resource, indicating that the model’s alignment capability is not dependent on the resource status of the target language.

- analyze performance against data availability in FIG. 8

FIG. 8 shows that performance plateaus after a certain threshold of monolingual pre-training data size is reached, suggesting that the quality of alignment is determined more by the richness of pre-training than by the quantity of parallel training data.

- show performance on non-English-centered language pairs in FIG. 9

FIG. 9 demonstrates that the aligner model trained on English-centered language pairs performs equally well on non-English-centered evaluation pairs, indicating that the model does not rely on English as a pivot language.

- explore X-centered language pairs in FIG. 10

FIG. 10 explores performance across language pairs centered on languages other than English, showing that the model achieves high accuracy even when trained and evaluated on non-English-centered pairs, confirming the language-agnostic nature of the alignment capability.

- discuss limitations and scope of the disclosure

While the invention demonstrates remarkable performance across a wide range of languages and settings, it is limited to languages included in the pre-training corpus and does not support languages with no prior monolingual representation. The framework assumes that sentences are properly tokenized and does not handle code-switching or mixed-script inputs. The scope of the disclosure encompasses all implementations of the described architecture, training methodology, and normalization technique, regardless of the specific pre-trained model used or the programming language employed.