Here is the complete patent application following the provided outline:

# DESCRIPTION

## BACKGROUND

### Field of the Disclosure

The present invention relates generally to the field of machine learning and artificial intelligence, and more specifically to systems and methods for active learning in text classification using deep neural networks. The disclosure particularly focuses on techniques for reducing sampling bias and improving efficiency in creating representative training datasets for deep learning models through optimized active learning strategies.

### Related Art

Traditional approaches to active learning in text classification have employed various strategies to select informative samples from large unlabeled datasets. Early work by Lewis and Gale (1994) demonstrated the effectiveness of uncertainty-based sampling using decision trees, which was later adapted for classifiers like Support Vector Machines (SVMs), Naive Bayes, and k-nearest neighbors. These methods typically relied on greedy uncertainty sampling techniques such as least confidence and entropy measures.

However, these conventional approaches suffered from several limitations, including significant sampling bias, high variance in sample quality, and lack of diversity in selected samples. Subsequent developments attempted to address these issues through diversity measures (Brinker, 2003; Hoi et al., 2006) or ensemble methods (McCallum and Nigam, 1998; Liere and Tadepalli, 1997). Despite these improvements, traditional active learning methods remained computationally expensive and often produced biased samples when applied to large datasets.

With the advent of deep learning, new challenges emerged in applying active learning principles to deep neural networks. Prior work has shown inconsistent results regarding the effectiveness of uncertainty sampling with deep models, with some studies finding it performs no better than random sampling (Sener and Savarese, 2018; Ducoffe and Precioso, 2018) while others demonstrated superior performance (Beluch et al., 2018; Gissin and Shalev-Shwartz, 2019). The translation of active learning principles from shallow to deep models has remained unclear, particularly for large-scale text classification tasks.

## SUMMARY

The present invention provides a novel system and method for deep active learning in text classification that overcomes the limitations of prior approaches. The disclosed technique utilizes uncertainty-based sampling with a single deep learning model to create highly representative training subsets from large text datasets. This approach demonstrates remarkable robustness to sampling bias while achieving significant computational efficiencies compared to conventional methods.

Key innovations of the present invention include: (1) a computationally efficient uncertainty sampling strategy using a single deep model that produces samples with properties comparable to more expensive ensemble approaches; (2) demonstration that actively acquired training sets exhibit minimal class bias and favorable feature bias towards class boundaries; (3) robustness to various algorithmic factors including initial set selection, query size, and query strategy; and (4) the ability to create compressed surrogate training sets (5x-40x smaller than original datasets) that maintain classification accuracy while enabling 25x-200x speedups in model training.

The disclosed method has been empirically validated through extensive experimentation, including over 2,300 active learning trials across eight large text classification datasets ranging from 120,000 to 3.6 million samples. Results show the invention outperforms prior state-of-the-art methods, achieving equivalent accuracy with 4x less training data compared to Bayesian active learning approaches and superior performance at all data sizes compared to diversity-based core-set sampling methods.

## DETAILED DESCRIPTION

The present invention provides a comprehensive system and methodology for deep active learning in text classification applications. The detailed implementation encompasses several key components and novel techniques that collectively address the limitations of prior approaches.

The core of the invention involves an optimized active learning framework that incrementally constructs training sets from large pools of unlabeled text data. At each iteration, the system trains a deep learning model (preferably FastText.zip) on the current labeled set, then uses the model's uncertainty estimates to select the most informative samples for labeling from the remaining pool. This process creates a sequence of increasingly representative training sets while minimizing redundant or biased sampling.

A critical innovation is the use of posterior uncertainty from a single deep model as the acquisition function, which surprisingly provides sampling properties comparable to more expensive ensemble approaches. The system implements this through entropy-based scoring of model outputs, where samples producing high entropy in the model's softmax predictions are prioritized for selection. This approach demonstrates remarkable stability across different initializations and query sizes, unlike traditional methods that show high variance under these conditions.

The invention incorporates several mechanisms to mitigate sampling bias. For class bias, the system automatically monitors and maintains balanced label distributions through real-time measurement of Kullback-Leibler divergence between sample and true distributions. For feature bias, the method naturally favors samples near class boundaries, as evidenced by high overlap (typically 60-80%) with support vectors from SVM models trained on full datasets. This desirable bias towards boundary samples enhances classification performance while avoiding the outlier selection problems common in conventional uncertainty sampling.

The system architecture includes specialized modules for handling various algorithmic factors that impact sampling quality. An initialization module manages the selection of starting labeled sets, with randomization techniques that ensure robustness to initial conditions. A query size optimizer dynamically adjusts batch sizes (typically 0.25%-2% of dataset size) based on dataset characteristics while maintaining stable sampling properties. The strategy selector implements multiple uncertainty measures (entropy, least confidence) with mechanisms to prevent redundancy in selected samples.

A significant advantage of the invention is its ability to generate highly compressed surrogate training sets that maintain the essential characteristics of full datasets. These compact representations (typically 5x-40x smaller than original data) enable dramatic speedups in training large models like ULMFiT, achieving 25x-200x faster training times while maintaining classification accuracy. The system includes verification modules that validate surrogate set quality through comparison with full-dataset training benchmarks.

The implementation also features specialized interfaces for downstream applications, including rapid bootstrapping of large model training and efficient hyperparameter optimization. The system's active learning components are designed for integration with various deep learning architectures, though optimal performance is achieved with FastText.zip due to its combination of speed and accuracy.

Experimental validation across eight large text classification datasets (AG-News, DBPedia, Amazon Reviews, Yelp Reviews, Yahoo Answers, and Sogou News) demonstrates the invention's superior performance over prior methods. The system achieves state-of-the-art accuracy with significantly less training data (12% vs 50% for Bayesian methods) and outperforms diversity-based approaches at all data sizes. The robust sampling properties enable reliable transfer of learned models to new domains and tasks.

The complete system includes supporting infrastructure for large-scale deployment, including distributed computing capabilities for handling massive text corpora, automated labeling interfaces, and monitoring tools for tracking sampling quality throughout the active learning process. Implementation details cover all aspects from data preprocessing and feature extraction to model training and evaluation, with particular attention to computational efficiency in large-scale settings.