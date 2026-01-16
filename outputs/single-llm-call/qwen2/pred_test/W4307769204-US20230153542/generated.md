# DESCRIPTION

## CROSS REFERENCE(S)

This patent application claims priority to and the benefit of U.S. Provisional Application No. 63/XXXXXXX, filed on [DATE], which is hereby incorporated by reference in its entirety.

## TECHNICAL FIELD

The present invention relates generally to the field of cross-lingual sentence retrieval and, more specifically, to a method and system for aligning cross-lingual sentences using a neural network model trained on a single language pair.

## BACKGROUND

Cross-lingual sentence retrieval is a critical task in natural language processing (NLP) that involves aligning parallel sentence pairs that are translations of each other from unlabeled multilingual documents. This task is essential for various downstream applications, including Machine Translation, cross-lingual Word Sense Disambiguation, and Quality Estimation. Traditional approaches to cross-lingual sentence retrieval are either entirely unsupervised, which often results in lower accuracy, or heavily supervised, requiring extensive parallel data that may not be readily available for low-resource languages.

Recent advancements in deep learning have led to the development of models that can leverage large amounts of monolingual data to learn robust cross-lingual representations. However, the effectiveness of these models in zero-shot cross-lingual transfer, particularly when trained on a single language pair, remains underexplored. There is a need for a method that can efficiently align cross-lingual sentences using minimal parallel data while maintaining high accuracy.

## DETAILED DESCRIPTION

The present invention provides a method and system for aligning cross-lingual sentences using a neural network model trained on a single language pair. The invention, referred to as OneAligner, is built on top of the XLM-RoBERTa (XLM-R) model, a state-of-the-art transformer-based model pre-trained on a large multilingual corpus. OneAligner introduces several key innovations to enhance the performance of cross-lingual sentence alignment:

### Base Model

OneAligner utilizes XLM-R as its base model. XLM-R is pre-trained on the monolingual CC-100 dataset, which covers 100 languages, and has demonstrated superior performance in various cross-lingual tasks. By leveraging the pre-trained cross-lingual representations, OneAligner can effectively capture the semantic similarities between sentences in different languages.

### Calculation of Semantic Similarity

OneAligner employs a modified version of BERT-score to calculate the semantic similarity between cross-lingual sentences. BERT-score is an unsupervised automatic evaluation metric originally designed to compute the similarity between sentences in the same language. In OneAligner, BERT-score is repurposed to compute cross-lingual semantic similarity. Specifically, given two sequences \( s = \{s_1, s_2, ..., s_M\} \) and \( t = \{t_1, t_2, ..., t_N\} \) in the source and target languages, respectively, the pairwise token-level cosine distance is computed as follows:

\[ F(s, t) = \frac{\sum_{i=1}^{M} \sum_{j=1}^{N} \text{cos}(s_i, t_j)}{M \times N} \]

This method serves as a shallow cross-attention layer that is computationally efficient and agnostic to the order of input sentences.

### In-Batch Normalization

OneAligner incorporates an in-batch normalization step to counteract the "popular sentence effect," where some sentences in one language tend to have a high similarity score with any sentence in the other language. This effect can lead to inaccurate rankings of candidate sentences. To address this issue, OneAligner subtracts a scaled average of similarity scores between each sentence in one language and all sentences in the other. The normalized similarity score is computed as follows:

\[ \text{normalized\_similarity}(S_i, T_j) = f(S_i, T_j) - \alpha \left( \frac{1}{N} \sum_{k=1}^{N} f(S_i, T_k) + \frac{1}{M} \sum_{l=1}^{M} f(S_l, T_j) \right) \]

where \( f \) is the function that computes semantic similarity (BERT-score in this case), and \( \alpha \) is a hyperparameter that determines the normalization strength. This normalization step is built into the model architecture to ensure accurate and reliable alignment.

### Classification with In-Batch Negatives

OneAligner employs a contrastive learning approach to train the model on a classification task with in-batch negatives. The intuition behind this approach is that a pair of sentences that are translations of each other can be interpreted as two "views" of the same underlying semantics. During training, the model computes the pairwise BERT-score between batches of sentences in the source and target languages and applies the in-batch normalization. The resulting similarity scores are treated as logits, and the cross-entropy loss is computed by pairing each positive logit with all negative logits. This approach helps the model establish a global score threshold for aligning cross-lingual sentences.

### Experimental Setup

OneAligner is trained and evaluated on several datasets to validate its performance. The training data is sampled from the OPUS-100 dataset, which contains parallel examples for 100 languages. All experiments are performed under a fixed 1M examples budget to ensure a fair comparison. The model is trained for 3 epochs with a batch size of 64 and evaluated with a batch size of 256. The maximum sequence length for both source and target languages is set to 100.

### Evaluation Data

OneAligner is evaluated on three datasets:
1. **Tatoeba-36**: A subset of the XTREME benchmark containing 36 language pairs, including multiple low-resource ones.
2. **New-Tatoeba**: A combination of development and test sets covering 223 language pairs, including 49 English-centered pairs and 58 low-resource pairs.
3. **BUCC 2018**: A cross-lingual bitext mining task involving 4 language pairs, all of which are highly resource-rich.

### Results and Analysis

#### All Language Pair Performance

Training OneAligner on the entire OPUS-100 dataset achieves state-of-the-art results on the Tatoeba-36 dataset, with an accuracy of 94.9%. This performance is comparable to models trained with 180 times more parallel examples, demonstrating the efficiency and effectiveness of OneAligner.

#### Single Language Pair Performance

OneAligner is also trained on individual rich-resource language pairs. The results show consistent performance across different language pairs, indicating that the model can be fine-tuned with almost any rich-resource language pair to achieve similar performance. The performance is positively correlated with the amount of parallel data up to a certain threshold, beyond which additional data does not significantly improve performance.

#### Language Pairs Not Centered Around English

OneAligner is evaluated on non-English-centered language pairs to assess its generalizability. The results show that the model performs similarly regardless of whether the training language pair is English-centered or not. This indicates that OneAligner can effectively transfer cross-lingual alignment capabilities across different language pairs.

#### Scaling Up the Number of Language Pairs

To further improve performance, OneAligner is trained on increasing numbers of rich-resource language pairs. The results show a monotonic increase in performance as more language pairs are included, reaching an accuracy of 94.0% on Tatoeba-36 when trained on the top-32 rich-resource language pairs. This performance is only 0.4 points off from the best performance when training with all language pairs under the same data budget.

#### BUCC Results

OneAligner is also evaluated on the BUCC 2018 dataset, achieving F1 scores that outperform previous state-of-the-art models. This demonstrates the model's ability to perform cross-lingual transfer effectively, even when the training language pair is not English-centered.

### Conclusion

OneAligner provides a robust and efficient solution for cross-lingual sentence alignment. By leveraging a single language pair and incorporating innovative techniques such as in-batch normalization and contrastive learning, OneAligner achieves state-of-the-art performance on multiple benchmarks. The invention offers a practical and scalable approach to cross-lingual sentence retrieval, making it a valuable tool for various NLP applications.

### Example Performance

OneAligner was tested on the Tatoeba-36 dataset, achieving an accuracy of 94.9%, which is 8.0 points higher than models trained with 180 times more parallel examples. When trained on a single rich-resource language pair, OneAligner matches the performance of a model trained on all language pairs within a 2.0 gap in accuracy. The model also performs well on non-English-centered language pairs and scales effectively with the addition of more rich-resource pairs. These results demonstrate the robustness and efficiency of OneAligner in cross-lingual sentence alignment tasks.