# DESCRIPTION

## FIELD
The present invention relates to the field of natural language processing, specifically to methods and systems for word sense disambiguation (WSD). More particularly, the invention pertains to a novel approach for leveraging lexical knowledge from multiple word sense inventories to train a general semantic equivalence recognizer, which can be further fine-tuned for specific WSD tasks.

## BACKGROUND
Human language is inherently ambiguous, with words often having multiple meanings depending on their context. Word Sense Disambiguation (WSD) is the task of automatically identifying the correct sense of a word within a given context. This task is crucial for many downstream applications in natural language processing, such as machine translation, information extraction, and sentiment analysis. Traditional WSD approaches have primarily relied on supervised learning techniques, which require large amounts of annotated data. However, these methods often suffer from limitations such as poor performance on rare and zero-shot word senses and dependency on predefined word sense inventories, typically WordNet.

Recent advancements in deep learning and pretraining of language models have led to significant improvements in WSD performance. However, these models still face challenges when dealing with low-resource scenarios and rare word senses. To address these issues, the present invention proposes a method that leverages abundant lexical knowledge from various word sense inventories to train a general semantic equivalence recognizer. This recognizer can then be fine-tuned for specific WSD tasks, leading to improved performance, especially in low-resource settings.

## SUMMARY
The present invention provides a method and system for word sense disambiguation (WSD) that overcomes the limitations of traditional supervised and knowledge-based approaches. The key aspects of the invention are as follows:

1. **Gloss Alignment Algorithm**: A novel algorithm is proposed to align glosses (definition sentences) from different word sense inventories based on their semantic equivalence. This alignment process significantly expands the available lexical knowledge, particularly for rare word senses.

2. **Pretraining a General Semantic Equivalence Recognizer**: Using the aligned glosses, a general semantic equivalence recognizer is pretrained to determine whether a word in a context sentence and a gloss are semantically equivalent. This recognizer can be used directly for WSD tasks or further fine-tuned on task-specific datasets to become an expert model.

3. **Two-Stage Transfer Learning Scheme**: The invention employs a two-stage transfer learning approach. In the first stage, the general model is pretrained on the aligned glosses. In the second stage, the model is fine-tuned on task-specific datasets to achieve state-of-the-art performance on both all-words and low-shot WSD tasks.

4. **Enhanced Performance**: The general model, without fine-tuning, demonstrates strong applicability to low-shot and zero-shot word senses. After fine-tuning, the expert model outperforms previous state-of-the-art models on all-words WSD tasks and significantly improves performance on low-shot WSD tasks.

5. **Application to Other NLP Tasks**: The semantic equivalence knowledge derived from the aligned glosses can also be applied to other natural language processing tasks, such as the Word-in-Context (WiC) task, leading to substantial improvements in performance.

The major contributions of the present invention are:
1. A gloss alignment algorithm that integrates lexical knowledge from different word sense inventories to train a general semantic equivalence recognizer.
2. A two-stage transfer learning scheme that enhances the performance of WSD models, particularly in low-resource scenarios.

## DETAILED DESCRIPTION
### Gloss Alignment Algorithm
The gloss alignment algorithm is designed to align glosses (definition sentences) from different word sense inventories based on their semantic equivalence. This alignment process is crucial for expanding the available lexical knowledge, especially for rare word senses. The algorithm converts the problem of aligning two groups of glosses into a Maximum-weight Bipartite Matching Problem, which can be solved using Linear Programming.

#### Data Collection
Word sense inventory data is collected from multiple sources, including WordNet 3.0 and five professional dictionaries for advanced English learners: Oxford Advanced Learner's Dictionary, Merriam-Webster's Advanced Learner's Dictionary, Collins COBUILD Advanced Dictionary, Cambridge Advanced Learner's Dictionary, and Longman Dictionary of Contemporary English. These dictionaries provide rich sense knowledge and example sentences for different word senses.

In total, the invention collects 557,800 glosses and 469,400 example sentences. Each word sense inventory is a lexical knowledge bank that provides example sentences to illustrate word senses, including those less frequently seen in the real world.

#### Problem Formulation
Given a keyword, the algorithm retrieves two word sense sets \( S_1 \) and \( S_2 \) from two different inventories. Each set consists of a list of definition sentences (glosses). The goal is to find a matching \( f: S_1 \rightarrow S_2 \) that maximizes the total rewards \( \sum_{a \in S_1, f(a) \in S_2} r(a, f(a)) \), where \( r: S_1 \times S_2 \rightarrow \mathbb{R} \) is a reward function based on textual similarity.

To measure the textual similarity between two definition sentences, a pretrained model like SBERT is applied to obtain sentence embeddings. The cosine similarity between the embeddings is used as the reward function.

#### Solving Bipartite Graph Matching by Linear Programming
The Maximum-weight Graph Matching problem can be solved using Linear Programming. For each edge \( (i, j) \) in the bipartite graph, a variable \( x_{ij} \) is introduced. \( x_{ij} = 1 \) if the edge is included in the matching, and \( x_{ij} = 0 \) otherwise. The total weight of the matching is \( \sum_{(i, j) \in S_1 \times S_2} w_{ij} x_{ij} \), where \( w_{ij} \) is the textual similarity score between the \( i \)-th definition sentence in \( S_1 \) and the \( j \)-th definition sentence in \( S_2 \).

Constraints are added to ensure that each vertex is in exactly one edge in the matching:
\[ \sum_{j \in S_2} x_{ij} = 1 \quad \text{for} \quad i \in S_1 \]
\[ \sum_{i \in S_1} x_{ij} = 1 \quad \text{for} \quad j \in S_2 \]

The goal is to find a maximum-weight perfect matching that satisfies these constraints. In the implementation, the gloss alignment solver is applied to all common words shared by two inventories, considering only glosses under the same part-of-speech (POS) category. Overall, the invention obtains 704,000 gloss alignment links.

### Positive and Negative Training Instances
The gloss alignment algorithm provides the linking between word sense sets \( S_1 \) and \( S_2 \). Two glosses \( g \in S_1 \) and \( g' \in S_2 \) have the same meaning if they are aligned by the algorithm or different meanings if they are not aligned. Positive and negative training instances are generated by pairing the definition sentence of \( g \) (or \( g' \)) with each example sentence in \( g \) (or \( g' \)). Pairs are labeled as positive if \( g \) and \( g' \) are aligned or negative otherwise.

In experiments, only aligned gloss pairs with textual similarities above a threshold (e.g., 0.6) are considered to improve the quality of supervision. In total, the invention generates 421,000 positive and 538,000 negative gloss-context pairs across different inventories. Additional pairs are generated by contrasting glosses within each inventory individually, resulting in 1.3 million positive and 418,000 negative gloss-context pairs.

### A Unified Neural Model for Recognizing Semantic Equivalence
The invention introduces a unified neural model for recognizing semantic equivalence. The model architecture is inspired by Blevins and Zettlemoyer (2020) and consists of a semantic encoder and a learning objective.

#### Model Architecture
The model uses a pretrained BERT or RoBERTa model to obtain contextual word representations and sentence representations. Given an input sentence \( S \) padded by the start symbol [CLS] and the end symbol [SEP], the model first obtains \( N \) contextualized embeddings \( \{o_i\}_{i=1}^N \) for all tokens \( \{t_i\}_{i=1}^N \) using BERT. For a context sentence, the contextualized embedding at the target word position is selected. For a gloss sentence, the first output embedding \( o_0 \) (corresponding to the special token [CLS]) is selected as the sentence representation.

The learning objective involves comparing the two representations \( u \) and \( v \) using element-wise difference \( |u - v| \) and element-wise multiplication \( u \cdot v \). These features are concatenated and multiplied with a trained weight matrix \( W_t \in \mathbb{R}^{4n \times 2} \) followed by a softmax prediction layer for binary classification (semantically equivalent or not).

The model is trained using binary cross-entropy loss and the AdamW optimizer with an initial learning rate of \( \{1e-5, 5e-6, 2e-6\} \), 0.2 dropout, a batch size of 64, and 10 training epochs. Two model sizes are considered: SemEq-Base, initialized with the pretrained BERT Base model, and SemEq-Large, initialized with the pretrained RoBERTa Large model.

### Evaluation
#### Accuracy of the Gloss Alignment Algorithm
To evaluate the accuracy of the gloss alignment algorithm, 1,000 gloss pairs from 704,000 alignments are randomly sampled and judged by two human annotators. The annotators agreed on 94% of the 200 gloss pairs they labeled in common, achieving a kappa inter-agreement score of 0.74. The remaining 800 gloss pairs were evenly allocated to the two annotators. The accuracy of the gloss alignment algorithm on each POS type based on human annotations is as follows: Noun (0.90), Verb (0.81), Adjective (0.88), and Adverb (0.85), with an overall accuracy of 0.87. Applying a threshold of 0.6 to alignment results further improves the accuracy to 0.98.

#### Experiments on WSD
The invention evaluates the model on two WSD datasets: the all-words WSD framework established by Raganato et al. (2017b) and the FEWS dataset proposed by Blevins et al. (2021).

**All-Words WSD Tasks**
The all-words WSD framework includes five benchmark datasets from previous Senseval and SemEval competitions. The testing dataset contains 5 benchmark datasets: Senseval-2, Senseval-3, SemEval-07, SemEval-13, and SemEval-15. The build-in training set is SemCor, which contains 226,036 annotated instances. The development set is SemEval-07.

The invention considers two strategies for incorporating rich lexical knowledge into model training:
1. **Data Augmentation**: The build-in training set is augmented with gloss-context pairs generated from the aligned word sense inventories.
2. **Transfer Learning**: The semantic equivalence recognizer is pretrained only using gloss-context pairs from the aligned word sense inventories. The pretrained classifier is a general model capable of deciding semantic equivalence independent of any specific word sense inventories. The general model is then fine-tuned on the build-in training set to become an expert model.

Experimental results show that the general model (SemEq-Base-General) achieves comparable performance with LMMS BERT (based on BERT Large) without using SemCor. After fine-tuning on SemCor, the expert model (SemEq-Base-Expert) improves performance on all-words WSD tasks, achieving an F1 score of 79.9% on the ALL dataset. Increasing the model parameters (SemEq-Large-Expert) further boosts performance to 80.7% on ALL. The SemEq-Large-Expert model outperforms AdaptBERT (the previous best model without using WordNet synset graph information) by 1.2% on ALL and better disambiguates all types of words, including nouns, verbs, adjectives, and adverbs.

**Few-Shot and Zero-Shot WSD Tasks**
The FEWS dataset focuses on low-shot WSD evaluation and covers 35,000 polysemous words and 71,000 senses. The build-in training set consists of 87,000 sentence instances, and the test set includes two evaluation subsets: a few-shot evaluation set and a zero-shot evaluation set.

The invention demonstrates that the general model (SemEq-Base-General) performs well on low-shot and zero-shot word senses, achieving 13.1%, 8.1%, and 5.6% higher performance on 0-shot, 1-2 shot, and 3-5 shot senses, respectively, compared to a model trained only on SemCor. After fine-tuning on SemCor, the expert model (SemEq-Base-Expert) fits the distribution of senses in the real world and achieves better overall performance.

**Experiments on Context-Sensitive Word Meanings**
The Word-in-Context (WiC) task from the SuperGLUE benchmark provides a high-quality dataset for evaluating context-sensitive word meanings. The invention applies the semantic equivalence knowledge from aligned inventories to the WiC task, improving RoBERTa Large's performance by 6% and surpassing the large language model T5-3B by 3.8%.

### Conclusion
The present invention provides a novel approach for word sense disambiguation that leverages lexical knowledge from multiple word sense inventories to train a general semantic equivalence recognizer. The two-stage transfer learning scheme enhances the performance of WSD models, particularly in low-resource scenarios. The general model demonstrates strong applicability to low-shot and zero-shot word senses, and the expert model achieves state-of-the-art performance on all-words and low-shot WSD tasks. The semantic equivalence knowledge derived from the aligned glosses can also be applied to other natural language processing tasks, leading to substantial improvements in performance.

### Ethical Considerations
The data used in this invention belongs to their respective owners. The authors are permitted to use the data under the permission of non-commercial research purposes and following the principle of fair use. The authors will not reproduce, republish, distribute, transmit, or link the data used on any other website without the express permission of the respective owners. The authors bear the responsibility to comply with the rules of copyright holders.