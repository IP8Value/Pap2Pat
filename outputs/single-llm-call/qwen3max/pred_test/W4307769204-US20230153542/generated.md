# DESCRIPTION

## CROSS REFERENCE(S)

This application claims the benefit of priority to U.S. Provisional Patent Application No. 63/XXX,XXX, filed on [Date], the entire disclosure of which is incorporated herein by reference in its entirety for all purposes.

## TECHNICAL FIELD

The present invention relates generally to the field of natural language processing and, more specifically, to systems and methods for cross-lingual sentence alignment using deep learning architectures. The invention provides a novel framework for aligning sentences across different languages without requiring extensive parallel corpora for low-resource language pairs, thereby enabling efficient and accurate cross-lingual retrieval, machine translation, and other multilingual natural language understanding tasks.

## BACKGROUND

Cross-lingual sentence alignment is a foundational task in multilingual natural language processing that involves identifying pairs of sentences in different languages that are mutual translations of one another. This capability is essential for constructing high-quality parallel corpora used in training machine translation systems, performing cross-lingual information retrieval, and enabling zero-shot transfer learning across languages. Traditional approaches to this problem have largely fallen into two categories: fully unsupervised methods that rely solely on monolingual data and pre-trained multilingual embeddings, and fully supervised methods that require large amounts of parallel training data across many language pairs.

Unsupervised methods, while avoiding the need for costly parallel data collection, often suffer from limited accuracy—particularly on low-resource languages—due to the absence of explicit alignment signals during training. Conversely, supervised approaches demand extensive parallel corpora covering numerous language pairs, including those involving low-resource languages, which are inherently difficult to obtain in sufficient quantity and quality. This dichotomy creates a significant practical gap: practitioners either accept suboptimal performance or face prohibitive data acquisition costs.

Moreover, existing supervised models frequently assume English as a central pivot language, limiting their applicability in truly multilingual settings where English may not be involved. Additionally, conventional similarity metrics such as cosine distance between mean-pooled sentence embeddings lack the fine-grained token-level alignment necessary for precise semantic matching, while full cross-attention mechanisms are computationally expensive and sensitive to input order. These limitations hinder the deployment of robust, scalable, and language-agnostic sentence alignment systems in real-world applications such to web-scale bitext mining or human-in-the-loop translation workflows.

## DETAILED DESCRIPTION

The present invention introduces a novel cross-lingual sentence alignment system, referred to herein as OneAligner, which overcomes the aforementioned limitations by leveraging a pre-trained multilingual language model enhanced with a supervised variant of BERT-score and an integrated normalization layer. The core network architecture is based on XLM-RoBERTa (XLM-R), a Transformer-based model pre-trained on the CC-100 corpus covering 100 languages, providing strong initial cross-lingual representations.

A key module within the system is the semantic similarity computation unit, which employs a modified BERT-score mechanism to calculate token-level pairwise cosine similarities between source and target sentences. Unlike standard Siamese architectures that use mean-pooled vectors, this approach retains positional and contextual information by computing alignment scores at the token level after the final Transformer layer, effectively implementing a shallow yet efficient form of cross-attention. This design maintains invariance to input order—i.e., swapping source and target sentences does not affect the output—while significantly improving alignment fidelity compared to pooling-based methods.

The invention distinguishes between "rich-source" and "low-resource" languages based on the availability of monolingual and parallel training data. Critically, the system is designed to operate under a rich-resource-only constraint, meaning it requires parallel data only from high-resource language pairs, yet achieves strong generalization to low-resource languages through effective zero-shot transfer.

The motivation for cross-lingual sentence alignment stems from its utility in downstream applications such as machine translation, cross-lingual question answering, and corpus filtering. Existing systems are limited by their reliance on either massive parallel datasets or weak unsupervised signals, neither of which scales well across the world’s linguistic diversity. The proposed framework addresses this by demonstrating that training on a single rich-resource language pair suffices to achieve near-optimal performance across dozens of languages.

Central to the invention is the use of a pre-trained multilingual language model—specifically XLM-R—as the backbone embedding generator. This model has been trained on vast amounts of monolingual text across 100 languages, enabling it to develop shared semantic representations that facilitate cross-lingual transfer even without explicit translation supervision.

The system incorporates a supervised version of BERT-score, originally developed as an evaluation metric for text generation, repurposed here as a trainable similarity function. During training, BERT-score computes the maximum cosine similarity between each token in the source sentence and all tokens in the target sentence (and vice versa), yielding precision and recall components that are combined into an F1-like score. This score serves as a differentiable proxy for semantic equivalence.

To counteract the “popular sentence effect”—a phenomenon wherein certain sentences exhibit spuriously high similarity scores with many unrelated sentences in the target language—the invention includes a built-in normalization layer. This layer subtracts a scaled average of a sentence’s similarity scores against all other sentences in the opposing language within the same batch. The scaling factor α is empirically tuned (typically α = 0.75) and applied during both training and inference, ensuring consistent score calibration regardless of batch composition.

Training is performed exclusively on rich-resource language pairs, such as English-Spanish or French-German, selected from high-quality parallel corpora like OPUS-100. Remarkably, models trained on just one such pair achieve performance within 2.0 accuracy points of models trained on all available language pairs under the same data budget.

The invention further enhances performance by scaling up to the top-k richest-resource language pairs (e.g., top-8 or top-16), achieving 94.0% accuracy on the Tatoeba-36 benchmark—only 0.4 points below the model trained on all pairs—while still adhering to the rich-resource-only principle.

Notably, the system does not require English as an anchor language. Experiments confirm that models trained on non-English-centered pairs (e.g., Spanish-French) perform comparably on English-centered evaluation tasks, provided both languages involved have sufficient monolingual pretraining data and the parallel training set exceeds a critical size threshold (approximately 10,000–20,000 sentence pairs).

The training framework operates as follows: input sentences from source and target languages are independently tokenized and fed into the shared XLM-R encoder. The resulting contextualized token embeddings are passed to the BERT-score computation module, which calculates pairwise token similarities. These scores undergo in-batch normalization to mitigate bias, then serve as logits for a contrastive classification objective.

During inference, the alignment model receives candidate sentence pairs and outputs normalized similarity scores that can be used for ranking or binary classification (e.g., in BUCC-style bitext mining). The embedding model generates dense vector representations that capture cross-lingual semantics without requiring re-encoding for different language directions.

The BERT score computation module implements both precision and recall variants: precision measures how well target tokens are covered by source tokens, while recall measures coverage in the reverse direction. The final BERT score is typically the harmonic mean (F1) of these two components, though alternative combinations may be used.

The normalization step is integral to the architecture, applied before loss computation and during inference. It ensures that similarity scores reflect genuine semantic correspondence rather than artifacts of sentence frequency or lexical overlap.

At inference time, the system processes batches of sentence pairs, computes normalized BERT scores, and ranks candidates accordingly. For open-domain mining tasks, a global threshold can be applied to accept or reject alignments.

The training methodology employs a contrastive learning approach using in-batch negatives. Given a batch of N aligned sentence pairs, the model treats the N correct alignments as positive examples and the remaining N² − N mismatched pairs as negative examples. This global negative sampling strategy enables the model to learn a universal decision boundary, crucial for real-world scenarios where not every source sentence has a translation.

In-batch negatives are formed implicitly by computing all pairwise similarities within a batch. This eliminates the need for hard negative mining and leverages the natural diversity of sentences in each batch to provide challenging contrastive signals.

Pairwise semantic similarity is computed via the BERT-score mechanism, which aligns tokens based on contextual embeddings from the final layer of XLM-R. This yields a soft, differentiable measure of cross-lingual equivalence.

The contrastive loss is implemented as a cross-entropy objective over the normalized similarity scores, with temperature scaling (typically τ = 5.0) to sharpen the probability distribution. This encourages higher scores for true alignments relative to all others in the batch.

The method of training the aligner model begins by receiving a training dataset comprising parallel sentences from one or more rich-resource language pairs. Positive input pairs are formed from known translations, while negative pairs emerge automatically from cross-combinations within each batch.

The system computes pairwise token-level similarity using cosine distance between contextual embeddings, aggregates these into sentence-level BERT scores, applies in-batch normalization, and computes the contrastive loss. The pre-trained multilingual model parameters are then updated via backpropagation.

Once trained, the model performs the alignment task by scoring new sentence pairs and outputting calibrated similarity values suitable for retrieval or classification.

The invention may be implemented on a computing device comprising one or more processors and memory storing executable instructions. The processor executes a paraphrase generation module (optional) for data augmentation, though the core alignment functionality does not require it. A data interface receives multilingual text inputs, and the output module returns alignment scores or ranked lists.

### Example Performance

The aligner model was trained on a fixed budget of 1 million parallel sentence pairs, sampled from either OPUS-100 or the Tatoeba Challenge dataset (v2021-08-07). From OPUS-100, noisy examples containing fewer than five tokens in either language were removed to ensure data quality.

Performance was evaluated on multiple benchmarks. FIGS. 5–13 illustrate key results, including state-of-the-art accuracy on cross-lingual sentence retrieval tasks. The primary evaluation used the Tatoeba dataset from the XTREME benchmark (referred to as Tatoeba-36), which includes 36 language pairs, many involving low-resource languages.

Additional evaluation employed the newer Tatoeba Challenge dataset (v2021-08-07), which covers 557 languages and over 3,700 bitexts. For fair comparison, only language pairs with at least 1,000 development/test examples were retained, and 1,000 examples per pair were randomly sampled, yielding 223 evaluation pairs—including 49 English-centered, 174 non-English-centered, and 58 low-resource pairs.

The BUCC 2018 dataset was used as a sanity check for bitext mining, requiring a universal threshold to accept/reject alignments—a more realistic setting than pure ranking tasks.

Baseline models included VECO and ERNIE-M, both strong performers on XTREME. Basic statistics for each model are summarized in FIG. 6, showing differences in training data scale and language coverage.

OneAligner outperformed all baselines. On Tatoeba-36, it achieved 94.9% accuracy when trained on all OPUS-100 pairs—8.0 points higher than VECO despite using 180× less parallel data. FIG. 5 displays this result.

When trained on a single rich-resource pair (e.g., English-Spanish), OneAligner reached 92.4–93.0% accuracy, within 2.0 points of the all-pairs model. FIG. 7 shows consistent performance across diverse language pairs, confirming robustness.

Data efficiency was demonstrated by maintaining high accuracy even with reduced training budgets. FIG. 8 reveals that performance plateaus once parallel data exceeds ~20k examples per language, highlighting diminishing returns beyond that point.

Crucially, the model excelled on non-English-centered pairs. FIG. 9 shows comparable accuracy whether training and evaluation involved English or not, provided monolingual pretraining data was sufficient.

Further analysis explored X-centered evaluation (where X is any language). FIG. 10 demonstrates that alignment accuracy for language X correlates strongly with the amount of monolingual data X received during XLM-R pretraining, reinforcing the importance of pretraining scale.

Limitations include dependence on the pretraining corpus (CC-100); languages absent from pretraining cannot be handled without vocabulary expansion. Additionally, the model assumes sentence-level parallelism and may struggle with highly divergent syntactic structures. Nevertheless, the scope of the disclosure encompasses all embodiments described herein, including variations in normalization strength, batch size, and similarity metric formulation.