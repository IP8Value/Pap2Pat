# DESCRIPTION

## CROSS REFERENCES

This patent application claims priority to and the benefit of U.S. Provisional Patent Application No. 63/XXXXXXX, filed on [DATE], entitled “CONTRASTIVE PARAMETER ENSEMBLING FOR REDUCING HALLUCINATIONS IN ABSTRACTIVE SUMMARIZATION,” the entire disclosure of which is incorporated herein by reference.

## TECHNICAL FIELD

The present invention relates generally to the field of natural language processing (NLP) and, more specifically, to methods and systems for reducing content hallucinations in abstractive summarization models.

## BACKGROUND

Abstractive summarization systems have made significant advancements in generating concise and coherent summaries of long documents. However, a persistent challenge in these systems is the tendency to hallucinate information—i.e., to generate content that is not supported by the source document. This issue is particularly pronounced in models trained on noisy datasets, where the training data itself contains factual errors. Traditional approaches to mitigate hallucinations, such as data filtering, can lead to a reduction in the diversity and quality of the generated summaries. Therefore, there is a need for a method that can effectively reduce hallucinations while maintaining the overall performance of the summarization model.

## DETAILED DESCRIPTION

### Computer and Network Environment

The present invention can be implemented in various computing environments, including but not limited to, cloud computing platforms, distributed computing systems, and local computing devices. The system may include one or more processors, memory, storage devices, input/output interfaces, and network communication interfaces. The processors may be any suitable type of processor, such as a central processing unit (CPU), a graphics processing unit (GPU), or a tensor processing unit (TPU). The memory may include volatile memory (e.g., RAM) and non-volatile memory (e.g., ROM, flash memory). The storage devices may include hard disk drives, solid-state drives, and other types of storage media. The input/output interfaces may include keyboards, mice, touchscreens, and other user interface devices. The network communication interfaces may include wired and wireless network interfaces, such as Ethernet, Wi-Fi, and cellular network interfaces.

The system may be configured to run various software components, including operating systems, application programs, and machine learning frameworks. The machine learning frameworks may include TensorFlow, PyTorch, and Hugging Face Transformers. The system may also include specialized hardware for accelerating machine learning tasks, such as GPUs and TPUs.

### Example Workflows

#### Overview of Contrastive Parameter Ensembling (CaPE)

The present invention introduces Contrastive Parameter Ensembling (CaPE), a method for reducing content hallucinations in abstractive summarization models. CaPE leverages the observation that the level of hallucination in a summarization model correlates with the level of noise in the training data. By training an expert model on clean data and an anti-expert model on noisy data, CaPE adjusts the parameters of a base summarization model to reduce hallucinations while preserving other aspects of the generated summaries, such as fluency and information recall.

#### Workflow for Training and Combining Models

1. **Data Selection**:
   - **Measuring Hallucinations**: To select clean and noisy data, CaPE uses automatic metrics for measuring factual consistency. Two primary metrics are employed: Entity Overlap and Dependency Arc Entailment (DAE).
     - **Entity Overlap**: Measures the token-level overlap of named entities between the summary and the source document. It is calculated as the percentage of named-entity tokens in the summary that are also present in the source document.
     - **DAE**: Measures fine-grained entailment by breaking the summary into smaller claims defined by dependency arcs. It identifies cases of intricate hallucinations, such as incorrect predicates or their arguments, coreference errors, and discourse link errors.
   - **Selecting Clean and Noisy Samples**: Using the entity overlap or DAE error metrics, CaPE selects clean samples with high factual consistency and noisy samples with low factual consistency. The thresholds for these selections are predefined based on the specific requirements of the task.

2. **Training Expert and Anti-expert Models**:
   - **Base Model**: A base summarization model is trained using the entire dataset. This model serves as the foundation for the expert and anti-expert models.
   - **Expert Model**: The base model is fine-tuned on the clean dataset to obtain the expert model. The goal is to enhance the factual consistency of the model while retaining other aspects such as ROUGE scores and information recall.
   - **Anti-expert Model**: The base model is fine-tuned on the noisy dataset to obtain the anti-expert model. The anti-expert model is designed to generate summaries with higher levels of hallucinations.

3. **Combining Models**:
   - **Parameter Adjustment**: The parameters of the base model are adjusted by combining the parameters of the expert and anti-expert models. Specifically, the parameters of the expert model are added to the base model, while the parameters of the anti-expert model are subtracted. This is achieved using a mixing coefficient \( \alpha \), which balances the contributions of the expert and anti-expert models.
   - **Mixing Coefficient**: The mixing coefficient \( \alpha \) is selected to ensure that the CaPE model does not underperform the base model by more than a predefined threshold on metrics such as ROUGE and information recall. This ensures that the reduction in hallucinations does not come at the cost of a significant drop in other performance metrics.

#### Evaluation and Performance

1. **Experimental Setup**:
   - **Datasets**: CaPE is evaluated on two benchmark abstractive summarization datasets: XSUM and CNN/DM. XSUM is highly abstractive and noisy, while CNN/DM is more extractive and contains fewer factual errors.
   - **Metrics**: A diverse set of metrics is used to evaluate the performance of CaPE, including ROUGE-1/2/L, D arc, D sum, E-P src, E-R ref, BERTScore, QEval, MNLI, and QAFactEval. These metrics assess various aspects of the generated summaries, such as factual consistency, fluency, and information recall.

2. **Results**:
   - **Automatic Evaluation**: CaPE consistently outperforms the base model and other baselines on all factual consistency metrics. The performance of CaPE is comparable to the base model on ROUGE scores and information recall, with a maximum 1% drop allowed. CaPE also improves BERTScore precision with respect to the source article, which correlates with human judgments of factuality.
   - **Human Evaluation**: Pairwise comparisons of summaries generated by CaPE and the base model show that CaPE significantly improves factual consistency. Human annotators rated 100 random articles from each of the XSUM and CNN/DM datasets, and the inter-annotator agreement was 0.8385.
   - **Transferability**: CaPE models improve performance on metrics that were not used for training the expert or anti-expert, demonstrating the robustness of the method. Additionally, the experts and anti-experts are interchangeable, allowing for flexible combinations of models.

3. **Computational Efficiency**:
   - **Training Time**: CaPE models only marginally increase the training time required for fine-tuning the expert and anti-expert on a smaller selected subset of training data. The increase in training time is ≤14% compared to the base model.
   - **Inference Time**: CaPE models do not increase the inference time, making them suitable for real-time applications. In contrast, post-processing methods require additional models for correcting summaries, increasing both the training and inference times.

#### Variants of CaPE

- **CaPE P P**: Uses the entity precision-based expert and anti-expert.
- **CaPE DP**: Uses the DAE-based expert and entity precision-based anti-expert.
- **CaPE P D**: Uses the entity precision-based expert and DAE-based anti-expert.
- **CaPE DD**: Uses the DAE-based expert and anti-expert.

Each variant of CaPE provides a different trade-off between factual consistency and other performance metrics. CaPE DP is found to provide the best balance for all performance measures on both datasets.

#### Conclusion

Contrastive Parameter Ensembling (CaPE) is a novel method for reducing content hallucinations in abstractive summarization models. By leveraging the differences between expert and anti-expert models, CaPE effectively reduces hallucinations while maintaining the overall performance of the summarization model. The method is computationally efficient and demonstrates robust performance across various datasets and evaluation metrics.