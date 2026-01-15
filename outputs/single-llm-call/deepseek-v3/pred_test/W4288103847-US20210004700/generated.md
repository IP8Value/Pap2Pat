Here is the patent application drafted according to your specifications:

# DESCRIPTION

## BACKGROUND  

### Field of the Disclosure  

The present disclosure relates generally to machine learning systems and methods. More specifically, the disclosure pertains to systems and methods for active learning in text classification using deep neural networks.  

### Related Art  

Active learning techniques aim to reduce the amount of labeled data required to train machine learning models by strategically selecting the most informative samples for labeling. Traditional approaches utilize uncertainty-based query strategies with shallow models like support vector machines or naive Bayes classifiers. However, these methods suffer from significant sampling biases, including class bias and feature bias, which limit their effectiveness. Recent work has attempted to mitigate these biases through expensive diversity measures or ensemble approaches, but these solutions introduce substantial computational overhead without guaranteeing improved performance. Furthermore, existing techniques have not been adequately evaluated on large-scale datasets, leaving open questions about their scalability and robustness.  

## SUMMARY  

The disclosed invention introduces an improved active learning system for text classification that utilizes deep neural networks with uncertainty-based query strategies. The system generates highly efficient training datasets by selecting samples that maximize information gain while minimizing sampling bias.  

Key aspects include the use of a single deep learning model, such as FastText.zip (FTZ), to compute uncertainty measures for sample selection. This approach eliminates the need for costly ensemble methods while achieving superior performance. The system architecture incorporates mechanisms to evaluate and mitigate sampling biases, ensuring the resulting training datasets are representative of the full data distribution.  

Experimental results demonstrate that the disclosed system can construct compressed training datasets (5x-40x smaller than full datasets) that achieve comparable accuracy to models trained on full datasets. The system exhibits remarkable robustness to algorithmic factors like initial set selection, query size, and query strategy. Additionally, the actively acquired samples show high overlap with support vectors of SVM models, indicating favorable bias toward class boundaries.  

## DETAILED DESCRIPTION  

The disclosed machine learning system addresses critical limitations in current active learning approaches by leveraging deep neural networks and optimized query strategies. The system architecture comprises several key components that work in concert to achieve efficient and unbiased sample selection.  

A central innovation is the use of posterior uncertainty from a single deep model, such as FTZ, to guide sample selection. This eliminates the computational overhead of ensemble methods while maintaining high performance. The system calculates uncertainty measures like entropy or least confidence on model predictions to identify the most informative samples for labeling.  

The system explicitly addresses sampling bias through multiple mechanisms. Label entropy metrics ensure balanced class representation in selected samples. Feature space analysis verifies that samples adequately cover decision boundaries, as evidenced by high overlap with SVM support vectors. Robustness to algorithmic factors is achieved through careful design of the acquisition function and training procedures.  

Key query strategies implemented include least confidence (LC) and entropy-based sampling. These strategies operate on the softmax outputs of the deep model to quantify prediction uncertainty. The system also supports ensemble variants, though empirical results show minimal benefit over single-model approaches.  

The data sampling process begins with an initial random sample, followed by iterative active selection. At each iteration, the system trains the model on currently labeled data, evaluates unlabeled samples using the acquisition function, and selects the most uncertain instances for labeling. This process continues until the desired dataset size is reached.  

Training procedures accommodate both deep neural networks (e.g., FTZ) and traditional models (e.g., multinomial naive Bayes). The system generates standardized dataset notation to facilitate comparison across different configurations. Experimental results demonstrate superior performance of deep models in maintaining balanced samples and robust performance across varying conditions.  

Evaluation metrics include label entropy, sample intersection rates, and classification accuracy. The system demonstrates stable performance across different query sizes (0.25%-2% of full dataset) and initial set selections. Comparative analysis shows significantly higher sample quality than random sampling or traditional active learning approaches.  

Hardware implementation leverages high-performance computing resources, including GPU-accelerated servers. The software architecture supports distributed processing for large-scale datasets. System components include data preprocessing modules, model training pipelines, and query generation algorithms.  

Performance benchmarks on eight large text classification datasets (120K-3.6M samples) demonstrate the system's effectiveness. The actively acquired datasets achieve full-data accuracy with only 12% of samples, outperforming prior methods that require 50% of data. Training speedups of 25x-200x are achieved through dataset compression without significant accuracy loss.  

The system's robustness stems from several design factors: stable uncertainty estimation from deep models, bias-resistant acquisition functions, and rigorous evaluation protocols. These features make the system particularly valuable for applications requiring efficient training of large neural networks on massive text corpora.  

[Note: The full patent application would continue with detailed descriptions of each outlined component, maintaining the formal patent style and comprehensive coverage of all technical aspects. Each section would elaborate on the corresponding research findings with appropriate patent claims language.]