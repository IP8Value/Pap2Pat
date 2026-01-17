# DESCRIPTION

## BACKGROUND

Semantic role labeling (SRL) is a fundamental task in natural language processing (NLP) that involves identifying the predicate-argument structure of sentences based on semantic frames and their roles. SRL is crucial for various downstream NLP tasks, including machine translation, sentiment analysis, and question answering. Traditionally, SRL has been approached using rule-based systems, statistical models, and more recently, deep learning techniques. However, despite the advancements in deep neural models, challenges remain, particularly in handling low-frequency predicates and out-of-domain data.

Deep neural models, such as Long Short-Term Memory (LSTM) networks and Transformer models, have achieved state-of-the-art performance on SRL tasks. These models leverage contextualized word embeddings, such as those provided by ELMo and BERT, to capture rich semantic information. However, these models often struggle with rare or unseen predicates and may not generalize well to out-of-domain data. To address these limitations, researchers have explored memory-based and memory-adaptive methods that incorporate external knowledge or past experiences to improve model performance.

One such approach is the Parameterized Neighborhood Memory Adaptive (PNMA) method, which combines the strengths of deep neural models with memory-based techniques. PNMA leverages a memory of token representations derived from a trained base model to improve the prediction of semantic roles. By utilizing the nearest neighbors of a token in the memory, PNMA can provide additional context and information that the base model might miss. This approach has shown promising results in improving the performance of SRL models, particularly in handling low-frequency predicates and out-of-domain data.

## SUMMARY

The present invention relates to a method and system for improving the performance of semantic role labeling (SRL) models using a Parameterized Neighborhood Memory Adaptive (PNMA) approach. The invention comprises two main phases: memory generation and PNMA training. In the memory generation phase, a memory of token representations is created using a trained base model. In the PNMA training phase, the memory is utilized to compute a parameterized vector representation of the nearest neighbors for each token, which is then used to retrain the classification layers of the base model.

The PNMA method improves the performance of SRL models by leveraging the information contained in the nearest neighbors of token representations. This approach is particularly effective in handling low-frequency predicates and out-of-domain data, where traditional deep neural models may struggle. The invention can be applied to both span-style and dependency-style SRL tasks and has demonstrated state-of-the-art results on standard benchmark datasets.

## DETAILED DESCRIPTION

### Overview

The Parameterized Neighborhood Memory Adaptive (PNMA) method is designed to enhance the performance of semantic role labeling (SRL) models by incorporating a memory of token representations. The method consists of two main phases: memory generation and PNMA training. During the memory generation phase, a memory of token representations is created using a trained base model. In the PNMA training phase, the memory is utilized to compute a parameterized vector representation of the nearest neighbors for each token, which is then used to retrain the classification layers of the base model. This approach leverages the information contained in the nearest neighbors to improve the prediction of semantic roles, particularly for low-frequency predicates and out-of-domain data.

### Base Model for SRL

The base model for SRL is an Alternating LSTM (ALSTM) model, which has been successfully used for SRL tasks. The ALSTM model consists of the following main components:

1. **Word Embeddings**: Numerical vector representations of textual words. Contextualized word embeddings, such as those derived from pretrained language models like ELMo and BERT, can be used to produce predictions for each token.
2. **LSTM Layers**: Recurrent neural network layers that capture the sequential dependencies in the input sentence.
3. **CRF Layer**: A Conditional Random Field (CRF) layer that is trained end-to-end with the other layers to ensure coherent predictions for the entire sequence.

The base model is trained on a labeled SRL dataset, and the final LSTM layer produces a dense representation for each token in the input sentence. These representations are used to populate the memory during the memory generation phase.

### Memory Generation

#### Intuition

The intuition behind the memory generation phase is that the nearest neighbors of a token in the memory contain valuable information about the correct label of the token, even when the base model prediction is incorrect. To test this hypothesis, the base model is trained on a labeled SRL dataset, and the predicted labels for the sentences in the validation set are computed. For each token with an incorrect predicted label, the K nearest neighbors are found, and the rank of the first token with the correct label among the neighbors is recorded. The distribution of these ranks is highly left-skewed, indicating that most tokens labeled incorrectly by the base model have a close neighbor in the memory with the correct label.

#### Memory Generation Process

1. **Training the Base Model**: The base model is trained on the SRL dataset using the Adam optimizer with an initial learning rate of 1e-3, which is decreased by half after epochs 50 and 75. L2 weight decay of 1e-4 is used for regularization. Dropout is applied after each LSTM layer and the word embedding layer.
2. **Populating the Memory**: After training, the base model is used to populate a memory \( M \) with the activations \( h_L(w) \) produced by the final LSTM layer for each token \( w \) in the training set. The memory \( M \) contains a subset of the tokens, typically 15% of the tokens in the training set.
3. **Computing Nearest Neighbors**: For each token \( w \) in the memory, the K nearest neighbors are computed using Euclidean distance. The distance between \( w \) and its i-th nearest neighbor is denoted as \( m_i(w) - h_L(w) \), where \( m_i(w) \) is the representation of the i-th nearest neighbor in the memory.

### PNMA Training

#### Parameterized Vector Representation

In the PNMA training phase, the memory constructed during the memory generation phase is utilized to compute a parameterized vector representation \( n_K(w) \) of the K nearest neighbors for each token \( w \). The parameterized vector representation is defined as:

\[ n_K(w) = \sum_{i=1}^K \eta_i \cdot m_i(w) \]

where \( \eta_i \) is a weight computed using a learned parameter \( n_i \):

\[ \eta_i = \frac{\exp(n_i \cdot (m_i(w) - h_L(w)))}{\sum_{j=1}^K \exp(n_j \cdot (m_j(w) - h_L(w)))} \]

The parameterized vector representation \( n_K(w) \) is a compact weighted representation of the K nearest neighbors, which is then used to retrain the classification layers of the base model.

#### Retraining the Classification Layers

During the PNMA training phase, the parameters associated with the LSTM, connection, and embedding layers in the base model are frozen. The neighborhood parameters \( \{n_i\}_{i=1}^K \) and the parameters in the classification and CRF layers are updated using the parameterized vector representation \( n_K(w) \) and the label of \( w \) in the training set. The model is trained for 20 epochs with a constant learning rate of 4e-4.

At test time, the SRL label predictions are obtained by computing \( n_K(w) \) for each token \( w \) in the test sentence, which requires the computation of the nearest neighbors \( N_K(w) \) using the LSTM representations of \( w \) and the memory \( M \).

### Experimental Evaluation

#### Datasets

The PNMA method is evaluated on both span-style and dependency-style Propbank semantic parsing datasets. For span-style SRL evaluation, the CoNLL2005 and CoNLL2012 datasets are used. For dependency-style SRL evaluation, the English subset of the CoNLL2009 dataset is used. The key statistics of these datasets are provided in Appendix A.

#### Word Embeddings

Experiments are conducted using randomly initialized word embeddings and publicly available pretrained contextualized embeddings from ELMo and BERT. The word embeddings are mapped to a 50-dimensional space, and each LSTM layer has a hidden dimension of 300. The memory \( M \) is populated with 15% of the tokens in the respective training sets.

#### Training Specifics

The base model is trained using the Adam optimizer for 100 epochs with an initial learning rate of 1e-3, which is decreased by half after epochs 50 and 75. L2 weight decay of 1e-4 is used for regularization. Dropout is applied after each LSTM layer and the word embedding layer. The PNMA training phase is conducted for 20 epochs with a constant learning rate of 4e-4.

#### Results

The PNMA method is evaluated using the official CoNLL SRL evaluation scripts. The results are averaged over five runs of model training with random seeds. The PNMA method consistently improves the performance of the base models, achieving state-of-the-art results on both span-style and dependency-style SRL datasets.

For the CoNLL2005 dataset, the PNMA method improves the validation and in-domain WSJ test F1 scores for all cases, with the highest gains observed on the out-of-domain Brown test set. The Base-BERT+PNMA model achieves a 2.0 point increase in F1 score on the out-of-domain Brown test set compared to existing methods.

For the CoNLL2012 dataset, the PNMA method also results in gains across the board, with the best results achieved by the Base-BERT+PNMA model, which improves upon the current state of the art by 0.3 and 0.4 F1 points for the validation and test sets, respectively.

For the CoNLL2009 dataset, the PNMA method outperforms the state-of-the-art syntax-aware and syntax-agnostic SRL models, achieving a significant performance gain of 5.2 absolute F1 points on the out-of-domain Brown set. The PNMA method is also competitive with syntax-aware models, demonstrating its effectiveness in handling low-frequency predicates and out-of-domain data.

### Analysis

#### Role Level Analysis

The effectiveness of the PNMA method is analyzed across individual argument role labels. The PNMA method improves the prediction of role labels for almost all types, with notable improvements for core roles such as A2 and A1. For example, PNMA reduces misclassification of A2 to A1 and LOC, and A1 to A0, A2, and LOC. Among adjunct roles, ADV benefits the most from PNMA, thanks to reduced confusion with DIR, MOD, and A2.

#### Instance Level Analysis

The PNMA method is more likely to change a wrong prediction of the base model into a correct prediction than to change a correct prediction into a wrong prediction. For the CoNLL2005 and CoNLL2009 datasets, PNMA is correct in 68% and 48% of the cases, respectively, where the models disagree, while the base model is correct in only 20% and 12% of these cases. This indicates that the nearest neighbors in the memory contain valuable information about the correct label of the test token, even when the base model prediction is incorrect.

#### Low-Frequency Exceptions

The PNMA method is particularly effective in handling low-frequency predicates, which are important in many real-world use cases. The disagreement between PNMA and the base model is higher for low-frequency predicates, and when the two models disagree, PNMA is correct most of the time. This confirms the effectiveness of PNMA in addressing low-frequency exceptions.

### Error Analysis

For the PNMA method to work effectively, two conditions must be met: there should be samples in the neighborhood of the test sample whose gold label matches that of the test sample, and enough of those neighboring samples should have representations that result in correct label predictions. The PNMA method mostly improves the samples associated with low-frequency predicates and which have moderate representation in the memory. However, there are some samples that lie in the moderate predicate frequency and moderate neighborhood sample regime which are predicted wrong by both models. This is because some of these neighbors have a wrong prediction by the base model, leading to a wrong prediction by the PNMA model. Future work will explore better memory generation techniques to handle such cases.

### Conclusion

The Parameterized Neighborhood Memory Adaptive (PNMA) method is a novel approach to improving the performance of semantic role labeling (SRL) models. By leveraging the information contained in the nearest neighbors of token representations, PNMA addresses the limitations of traditional deep neural models, particularly in handling low-frequency predicates and out-of-domain data. The PNMA method has demonstrated state-of-the-art results on standard benchmark datasets and is a promising direction for future research in SRL.