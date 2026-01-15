# DESCRIPTION

## FIELD

- define NLP and WSD

Natural Language Processing (NLP) is a subfield of artificial intelligence concerned with the interaction between computers and human language, enabling machines to understand, interpret, generate, and respond to textual or spoken communication in a manner that is both meaningful and contextually appropriate. Within NLP, Word Sense Disambiguation (WSD) refers to the computational task of identifying the correct meaning or sense of a polysemous word based on its contextual usage within a sentence or discourse. This process is essential because many words in human language possess multiple distinct meanings, and the intended sense can only be determined through analysis of surrounding linguistic cues. WSD serves as a foundational component for higher-level NLP applications such as machine translation, information retrieval, question answering, sentiment analysis, and semantic parsing, where ambiguity in word meaning can lead to significant errors in downstream performance. The challenge of WSD lies not merely in recognizing lexical variation, but in accurately mapping a word’s occurrence in a specific context to its most semantically aligned definition among a set of possible senses, often drawn from structured lexical resources such as dictionaries or ontologies.

## BACKGROUND

- motivate WSD
- limitations of supervised models

Word Sense Disambiguation is a critical problem in computational linguistics because the accurate interpretation of word meaning directly impacts the reliability and robustness of language-based systems. Without effective disambiguation, even state-of-the-art models may misinterpret the intent of a sentence, leading to incorrect inferences, flawed translations, or misleading summaries. For instance, the word “bank” may refer to a financial institution, the edge of a river, or the act of tilting an aircraft—each requiring a distinct semantic representation for proper understanding. While supervised approaches to WSD have achieved notable success by training models on manually annotated corpora such as SemCor, these methods are inherently constrained by the scope and coverage of their training data. Supervised models are typically trained to distinguish between senses defined within a single lexical inventory, most commonly WordNet, and thus lack the ability to generalize to senses that are underrepresented or entirely absent in the training set. This limitation becomes especially pronounced when encountering rare, novel, or domain-specific word senses, which are frequently encountered in real-world applications but rarely appear in curated datasets. Furthermore, supervised models are tightly coupled to the structure and granularity of the sense inventory they are trained on, rendering them inflexible when applied to alternative lexical resources or multilingual settings. As a result, these models exhibit poor zero-shot and few-shot performance, fail to transfer knowledge across inventories, and cannot leverage the rich, complementary semantic information encoded in diverse dictionaries compiled by human lexicographers.

## SUMMARY

- motivate gloss alignment algorithm
- propose gloss alignment algorithm
- application of gloss alignment algorithm
- embodiment of method
- generate aligned inventories
- obtain word in context sentence
- determine semantic equivalence scores
- predict correct sense of word
- generate positive and negative gloss pairs
- determine sentence textual similarity score
- train gloss classifier

The invention is motivated by the observation that different dictionaries, though independently compiled, often express the same underlying meanings of a word through semantically equivalent glosses—definition sentences that capture the intended sense using distinct phrasing. This redundancy across lexical resources presents an opportunity to unify and amplify semantic knowledge beyond the limitations of any single inventory. To exploit this insight, the invention proposes a gloss alignment algorithm that systematically identifies correspondences between glosses from multiple word sense inventories by modeling the alignment as a maximum-weight bipartite matching problem, where edge weights are determined by sentence-level textual similarity computed using pre-trained transformer models. The resulting aligned inventories form a unified semantic knowledge base that enables the generation of high-quality, automatically labeled training pairs for a semantic equivalence recognizer. This recognizer is trained to determine whether a given word in context is semantically equivalent to a candidate gloss, regardless of the source inventory. The method is embodied in a two-stage process: first, aligned inventories are constructed by matching glosses across dictionaries using an optimization framework that maximizes overall textual similarity; second, a neural classifier is trained on positive and negative gloss-context pairs derived from these alignments, where positive pairs consist of a context sentence and its semantically aligned gloss, and negative pairs consist of a context sentence paired with a non-aligned gloss. Semantic equivalence scores are computed by encoding both the context sentence and the gloss into dense vector representations using a pre-trained transformer architecture, followed by a binary classification layer that evaluates their similarity. These scores are then used to predict the correct sense of a target word by ranking all candidate glosses associated with the word and selecting the one with the highest probability. The training of the gloss classifier is enhanced by augmenting the training data with both intra-inventory and cross-inventory pairs, ensuring broad coverage of lexical variation and improved generalization to low-resource senses. Sentence textual similarity is determined using a secondary pre-trained model, such as SBERT, which provides robust, context-aware embeddings for measuring semantic overlap between definition sentences. The resulting classifier, trained on this rich, multi-source supervision, achieves superior performance in word sense disambiguation tasks without requiring task-specific annotations, and can be fine-tuned for domain-specific applications while retaining its generalization capacity.

## DETAILED DESCRIPTION

- introduce word sense predicting model
- describe gloss alignment of word sense inventories
- generate pairs of glosses
- label positive and negative pairs of glosses
- generate pairs of glosses using glosses within each word sense inventory
- obtain context sentence
- train model using training data
- use pre-trained transformers
- generate probability to predict correct sense of word
- evaluate word sense predicting model using WSD Datasets
- generate positive and negative pairs of glosses from built-in training data
- generate aligned inventories using dictionaries
- train transformers using augmented training data
- train general model using aligned inventories
- fine-tune general model on built-in training data
- compare expert model with previous best model
- demonstrate benefits of leveraging multiple word sense inventories
- evaluate word sense predicting model on FEWS dataset
- augment FEWS train set with multiple word sense inventories
- adopt transfer learning strategy on FEWS dataset
- describe aligned word sense inventory
- use dictionaries as word sense inventories
- align parallel glosses from multiple word sense inventories
- determine best matching function using optimization setup
- use Maximum Weighted Bipartite matching
- configure matching function to maximize sentence textual similarity
- describe example setup of Maximum Weighted Bipartite Matching optimization
- use secondary pre-trained model to measure sentence textual similarity
- describe semantic equivalence recognizer model
- train transformers using gloss examples from aligned inventories
- train transformers using augmented training data
- produce output probabilities for glosses semantically equivalent to word in context sentence
- describe word sense predicting model
- introduce process 400
- generate aligned inventories
- obtain word in context sentence
- determine semantic equivalence scores
- predict correct sense of word
- introduce process 500
- collect glosses from first word sense inventory
- collect glosses from second word sense inventory
- determine best match between inventories
- determine sentence textual similarity score
- determine matching function
- generate positive gloss pairs
- generate negative gloss pairs
- introduce process 600
- input context sentence into semantic equivalence recognizer model
- input pairs of glosses into semantic equivalence recognizer model
- identify glosses associated with word in context sentence
- apply trained gloss classifier
- generate probability score for each gloss
- introduce process 700
- input context sentence into semantic equivalence recognizer model
- input pairs of glosses into semantic equivalence recognizer model
- identify glosses associated with word in context sentence
- apply trained gloss classifier

The word sense predicting model is a neural architecture designed to determine the correct semantic sense of a target word within a given context sentence by evaluating its semantic equivalence to candidate glosses drawn from a unified, aligned inventory of lexical definitions. The model operates through a series of interconnected processes that begin with the construction of aligned word sense inventories by integrating multiple authoritative dictionaries, including Oxford Advanced Learner’s Dictionary, Merriam-Webster’s Advanced Learner’s Dictionary, Collins COBUILD Advanced Dictionary, Cambridge Advanced Learner’s Dictionary, Longman Dictionary of Contemporary English, and WordNet. For each word shared across these inventories, glosses are extracted and grouped by part-of-speech category to ensure semantic consistency. A maximum weighted bipartite matching algorithm is then employed to identify the optimal one-to-one correspondence between glosses from different inventories, where the weight of each potential match is determined by the sentence-level textual similarity between two glosses, computed using a pre-trained sentence embedding model such as SBERT. This optimization ensures that glosses expressing the same underlying meaning, despite differing in wording, are matched with high confidence. Once aligned, the system generates positive and negative gloss-context pairs: positive pairs are formed by associating a context sentence from a dictionary’s example usage with its semantically aligned gloss, while negative pairs are formed by pairing the context sentence with a non-aligned gloss from either the same or a different inventory. These pairs are used to train a semantic equivalence recognizer, which employs a transformer-based encoder—initializing from BERT or RoBERTa—to generate contextual representations of both the target word in context and the candidate gloss. The representations are combined using element-wise operations and passed through a binary classification head to produce a probability score indicating whether the gloss is semantically equivalent to the word in context. The model is first trained as a general-purpose classifier using only the aligned inventories, without access to any task-specific annotated data, thereby acquiring broad, transferable semantic knowledge. Subsequently, the general model may be fine-tuned on domain-specific datasets such as SemCor or FEWS to produce an expert model optimized for particular evaluation benchmarks. Evaluation on standard WSD datasets demonstrates that the general model, without fine-tuning, outperforms previous supervised baselines by 13.1% on zero-shot word senses, and the fine-tuned expert model achieves state-of-the-art performance on both all-words WSD and low-shot WSD tasks, surpassing prior models by 1.2% and 4.3%, respectively. The model’s effectiveness is further validated on the WiC benchmark, where it improves RoBERTa Large’s accuracy by 6%, even without using WordNet during training. The entire system is implemented through three core processes: Process 400 generates aligned inventories and computes semantic equivalence scores to predict the correct sense; Process 500 collects glosses from two inventories, computes pairwise similarities, and determines optimal matches to generate positive and negative training pairs; and Process 600 and Process 700 deploy the trained classifier by inputting context sentences and candidate glosses into the semantic equivalence recognizer, identifying the most probable sense by ranking all associated glosses according to their output probabilities. The resulting system enables robust, inventory-agnostic word sense disambiguation that leverages the collective lexical wisdom of multiple expert dictionaries, overcoming the data scarcity and domain rigidity inherent in traditional supervised approaches.