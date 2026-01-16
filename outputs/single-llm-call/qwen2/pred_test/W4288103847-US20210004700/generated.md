# DESCRIPTION

## BACKGROUND

### Field of the Disclosure

The present disclosure relates generally to the field of machine learning, specifically to methods and systems for active learning in deep text classification. More particularly, the disclosure pertains to techniques for efficiently constructing representative training datasets for deep neural network (DNN) models using active learning, thereby reducing the amount of labeled data required to achieve high classification accuracy.

### Related Art

Deep neural networks (DNNs) have achieved state-of-the-art performance in various natural language processing (NLP) tasks, including text classification. However, the success of DNNs often relies on large labeled datasets, which can be costly and time-consuming to obtain. Active learning is a promising approach to mitigate this issue by iteratively selecting the most informative samples for labeling, thereby reducing the amount of labeled data needed. Despite its potential, the effectiveness of active learning in the context of deep text classification remains a subject of ongoing research.

Traditional active learning methods for text classification have primarily focused on shallow models such as decision trees, support vector machines (SVMs), and naive Bayes classifiers. These methods often employ uncertainty sampling, where the model selects samples with the highest uncertainty for labeling. However, concerns about sampling bias and the need for diversity measures have led to the development of more complex acquisition functions, such as query-by-committee and core-set sampling.

Recent advancements in deep active learning have adapted these frameworks to train DNNs on large datasets. However, the properties of deep active learning, particularly regarding sampling efficiency and bias, have not been thoroughly investigated. Some studies suggest that uncertainty-based strategies perform no better than random sampling, while others find that ensemble methods and Bayesian approaches outperform single-model strategies. The lack of consensus and the limited scope of existing studies, which often use relatively small datasets, highlight the need for a comprehensive empirical investigation.

## SUMMARY

The present disclosure addresses the challenges and uncertainties in deep active text classification by providing a method and system for efficiently constructing representative training datasets using active learning. The disclosed method leverages uncertainty sampling with a single deep model, such as FastText.zip (FTZ), to create small, surrogate training sets that can achieve similar test accuracy to those trained on full datasets. The method is robust to various algorithmic factors and demonstrates minimal sampling bias, making it suitable for bootstrapping the training of large DNN models.

In one aspect, the disclosure provides a method for active text classification comprising the steps of:
1. Initializing a training set with a small, randomly selected subset of labeled data from a pool of unlabeled data.
2. Training a deep model, such as FastText.zip, on the initial training set.
3. Using an uncertainty-based query strategy, such as entropy, to select a batch of samples from the pool of unlabeled data.
4. Labeling the selected samples and adding them to the training set.
5. Repeating steps 2-4 until a predetermined stopping criterion is met, such as achieving a desired level of classification accuracy or reaching a maximum number of iterations.

In another aspect, the disclosure provides a system for implementing the method, comprising a processor and a memory storing instructions that, when executed by the processor, cause the system to perform the steps of the method.

The disclosed method and system offer several advantages over existing approaches:
- **Efficiency**: The method significantly reduces the amount of labeled data required to achieve high classification accuracy, thereby lowering costs and speeding up the model training process.
- **Robustness**: The method is robust to various algorithmic factors, such as initial set selection, query size, and query strategy, ensuring consistent performance across different datasets and experimental setups.
- **Minimal Bias**: The method demonstrates minimal sampling bias, both in terms of class distribution and feature space, making it suitable for a wide range of applications.
- **Scalability**: The method can handle large datasets and can be used to bootstrap the training of large DNN models, such as ULMFiT, with significant speedups and minimal loss in accuracy.

## DETAILED DESCRIPTION

The detailed description of the invention is provided below, outlining the specific embodiments and implementations of the method and system for active text classification using deep learning.

### Field of the Disclosure

The present disclosure is directed to the field of machine learning, specifically to methods and systems for active learning in deep text classification. The invention aims to address the challenges associated with obtaining large labeled datasets by providing an efficient and robust method for constructing representative training datasets using active learning.

### Background

Deep neural networks (DNNs) have revolutionized the field of natural language processing (NLP) by achieving state-of-the-art performance in various tasks, including text classification. However, the success of DNNs often depends on the availability of large labeled datasets, which can be costly and time-consuming to obtain. Active learning is a technique that can help mitigate this issue by iteratively selecting the most informative samples for labeling, thereby reducing the amount of labeled data required.

Traditional active learning methods for text classification have primarily focused on shallow models such as decision trees, support vector machines (SVMs), and naive Bayes classifiers. These methods often employ uncertainty sampling, where the model selects samples with the highest uncertainty for labeling. However, concerns about sampling bias and the need for diversity measures have led to the development of more complex acquisition functions, such as query-by-committee and core-set sampling.

Recent advancements in deep active learning have adapted these frameworks to train DNNs on large datasets. However, the properties of deep active learning, particularly regarding sampling efficiency and bias, have not been thoroughly investigated. Some studies suggest that uncertainty-based strategies perform no better than random sampling, while others find that ensemble methods and Bayesian approaches outperform single-model strategies. The lack of consensus and the limited scope of existing studies, which often use relatively small datasets, highlight the need for a comprehensive empirical investigation.

### Summary of the Invention

The present invention provides a method and system for efficiently constructing representative training datasets for deep text classification using active learning. The method leverages uncertainty sampling with a single deep model, such as FastText.zip (FTZ), to create small, surrogate training sets that can achieve similar test accuracy to those trained on full datasets. The method is robust to various algorithmic factors and demonstrates minimal sampling bias, making it suitable for bootstrapping the training of large DNN models.

#### Method for Active Text Classification

The method for active text classification comprises the following steps:

1. **Initialization**:
   - Initialize a training set with a small, randomly selected subset of labeled data from a pool of unlabeled data.

2. **Training**:
   - Train a deep model, such as FastText.zip (FTZ), on the initial training set.

3. **Query Selection**:
   - Use an uncertainty-based query strategy, such as entropy, to select a batch of samples from the pool of unlabeled data.

4. **Labeling and Updating**:
   - Label the selected samples and add them to the training set.

5. **Iteration**:
   - Repeat steps 2-4 until a predetermined stopping criterion is met, such as achieving a desired level of classification accuracy or reaching a maximum number of iterations.

#### System for Implementing the Method

The system for implementing the method comprises:
- **Processor**: A computing device capable of executing the steps of the method.
- **Memory**: A storage device for storing the instructions and data required to execute the method.
- **Input/Output Devices**: Devices for receiving user input and displaying results.

### Detailed Description of the Invention

#### Initialization

The method begins by initializing a training set with a small, randomly selected subset of labeled data from a pool of unlabeled data. The size of the initial set can vary depending on the dataset and the desired level of accuracy. For example, the initial set can be 0.5% to 1% of the total dataset.

#### Training

The next step involves training a deep model, such as FastText.zip (FTZ), on the initial training set. FastText.zip is a lightweight and efficient deep learning model that provides competitive results with a significant speedup compared to other deep models. The model is trained using standard techniques, such as stochastic gradient descent (SGD), and the hyperparameters are tuned to optimize performance.

#### Query Selection

Once the model is trained, an uncertainty-based query strategy is used to select a batch of samples from the pool of unlabeled data. The query strategy can be based on various uncertainty measures, such as least confidence, margin, or entropy. In the preferred embodiment, the entropy measure is used, which selects samples with the highest uncertainty in their predicted probabilities.

The size of the query batch can vary depending on the dataset and the desired level of accuracy. For example, the query size can be 0.5% to 1% of the total dataset. The query strategy can also be adjusted to account for different algorithmic factors, such as initial set selection, query size, and query strategy.

#### Labeling and Updating

The selected samples are then labeled and added to the training set. The labeling process can be performed manually by human annotators or automatically using a labeling service. Once the samples are labeled, they are added to the training set, and the model is retrained on the updated training set.

#### Iteration

The process of training the model, selecting a batch of samples, labeling the samples, and updating the training set is repeated until a predetermined stopping criterion is met. The stopping criterion can be based on various factors, such as achieving a desired level of classification accuracy, reaching a maximum number of iterations, or exhausting the pool of unlabeled data.

### Advantages of the Invention

The method and system for active text classification offer several advantages over existing approaches:

- **Efficiency**: The method significantly reduces the amount of labeled data required to achieve high classification accuracy, thereby lowering costs and speeding up the model training process.
- **Robustness**: The method is robust to various algorithmic factors, such as initial set selection, query size, and query strategy, ensuring consistent performance across different datasets and experimental setups.
- **Minimal Bias**: The method demonstrates minimal sampling bias, both in terms of class distribution and feature space, making it suitable for a wide range of applications.
- **Scalability**: The method can handle large datasets and can be used to bootstrap the training of large DNN models, such as ULMFiT, with significant speedups and minimal loss in accuracy.

### Experimental Results

To validate the effectiveness of the method, extensive experiments were conducted on eight large datasets, including AG-News, DBPedia, Amazon Review Polarity, Amazon Review Full, Yelp Review Polarity, Yelp Review Full, Yahoo Answers, and Sogou News. The experiments involved over 2,300 runs, each with different initial sets, query sizes, and query strategies.

The results showed that the method using uncertainty sampling with a single FastText.zip model outperformed other methods, including random sampling, ensemble methods, and diversity-based approaches. The method achieved high classification accuracy with a significant reduction in the amount of labeled data required. Additionally, the method demonstrated minimal sampling bias and high robustness to various algorithmic factors.

### Conclusion

The present invention provides a method and system for efficiently constructing representative training datasets for deep text classification using active learning. The method leverages uncertainty sampling with a single deep model, such as FastText.zip, to create small, surrogate training sets that can achieve similar test accuracy to those trained on full datasets. The method is robust to various algorithmic factors and demonstrates minimal sampling bias, making it suitable for a wide range of applications. The invention offers significant advantages in terms of efficiency, robustness, and scalability, making it a valuable tool for researchers and practitioners in the field of machine learning.