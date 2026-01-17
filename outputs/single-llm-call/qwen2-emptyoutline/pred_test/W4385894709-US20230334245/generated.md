# DESCRIPTION

## CROSS REFERENCE

This application claims the benefit of U.S. Provisional Application No. 63/XXXXXX, filed [DATE], which is hereby incorporated by reference in its entirety.

## TECHNICAL FIELD

The present invention relates to the field of natural language processing (NLP), specifically to methods and systems for improving the efficiency of zero-shot text classification using conformal prediction. More particularly, the invention provides a framework for filtering unlikely target labels to reduce computational costs and improve prediction efficiency.

## BACKGROUND

Zero-shot text classification is a critical task in natural language processing (NLP) with numerous real-world applications. Traditional approaches for zero-shot text classification involve mapping text and labels to a common embedding space and calculating similarity scores. These methods, while computationally efficient, often lack the precision and performance of more advanced techniques.

Later approaches, such as those using generative modeling and label embedding-based attention, offer improved performance but at the cost of increased computational complexity. Natural Language Inference (NLI) and Next Sentence Prediction (NSP) models, which utilize large transformer-based pre-trained language models (PLMs), have demonstrated superior performance in zero-shot text classification. However, these models suffer from significant computational inefficiencies, especially as the number of target labels increases.

The computational cost of NLI and NSP models is primarily due to the requirement of recomputing the encoding for each text-hypothesis pair separately, leading to a linear increase in inference time with the number of target labels. This inefficiency not only slows down the prediction process but also increases the carbon footprint associated with running these models.

To address these challenges, the present invention introduces a method for improving the efficiency of NLI and NSP-based zero-shot text classification models by using a conformal predictor (CP) to filter out unlikely target labels. The CP provides a model-agnostic framework to generate a label set within a pre-defined error rate, thereby reducing the computational burden without compromising the overall performance of the zero-shot models.

## DETAILED DESCRIPTION

### Computer Environment

The present invention can be implemented in various computing environments, including but not limited to cloud-based systems, distributed computing networks, and local computing devices. The system comprises one or more processors, memory, storage devices, and input/output interfaces. The processors are configured to execute instructions stored in memory to perform the methods described herein. The memory stores data and program instructions, including the zero-shot text classification models and the conformal predictor algorithms. The storage devices are used to store large datasets and model parameters. The input/output interfaces facilitate user interaction and data communication.

### Work Flows

The workflow of the invention involves several key steps:

1. **Data Preparation**: The system receives a dataset containing text samples and corresponding labels. The dataset is split into training, validation, and test sets. The training set is used to calibrate the conformal predictor, while the validation and test sets are used to evaluate the performance of the zero-shot models.

2. **Model Selection**: The system selects a zero-shot text classification model, such as an NLI or NSP model based on a pre-trained language model (PLM). The selected model is used to generate initial predictions for the text samples.

3. **Conformal Predictor Calibration**: A conformal predictor (CP) is built using a base classifier. The base classifier can be a simple token overlap model, a cosine similarity model, or a more complex classifier such as a fine-tuned BERT model. The CP is calibrated using a subset of the training data, ensuring that the predicted label sets maintain a pre-defined error rate.

4. **Label Filtering**: The CP is used to filter out unlikely target labels for each text sample. The filtered label set is then used with the zero-shot model to make the final prediction. This step significantly reduces the number of text-label pairs that need to be processed, thereby improving computational efficiency.

5. **Performance Evaluation**: The system evaluates the performance of the zero-shot model with and without the CP-based label filtering. Metrics such as accuracy, average inference time, and average prediction set (APS) size are used to assess the improvement in efficiency and performance.

### Example Data Experiment and Performance

To demonstrate the effectiveness of the invention, experiments were conducted on several benchmark datasets, including SNIPS, ATIS, HWU64, AG's news, and Yahoo! Answers. The datasets vary in the number of target labels, ranging from 4 to 64.

#### Experiment Setup

- **Models**: The zero-shot models used were "facebook/bart-large-nli" for NLI and "bert-base-uncased" for NSP.
- **Base Classifiers**: Four base classifiers were tested:
  - **Token Overlap (CP-Token)**: Calculates the percentage of common tokens between the input text and the target label.
  - **Cosine Similarity (CP-Glove)**: Computes the cosine distance between the bag-of-words (BoW) representation of the target label and the input text using static GloVe embeddings.
  - **Classifier (CP-CLS)**: Fine-tunes a distilled BERT model on the data labeled using the zero-shot model and uses the negative of class logits as the non-conformity scores.
  - **Distilled Model (CP-Distil)**: Uses a pre-trained "crossencoder/nli-distilroberta-base" model.

#### Results

- **Coverage**: All base classifiers achieved valid coverage for smaller error rates (α ≤ 0.5). The empirical coverage matched the nominal coverage, indicating that the CP effectively filters unlikely labels without dropping performance below the predefined error rate.
- **Average Prediction Set (APS) Size**: The CP-CLS base classifier provided the smallest APS size, reducing the average number of labels for both NLI and NSP models by approximately 41%. This suggests that fine-tuning a base classifier is beneficial when unlabeled samples are readily available.
- **Inference Time**: The CP-Token base classifier achieved the best inference time reduction, particularly with the NLI model on datasets with fewer labels. The minimal complexity of token overlap calculation added negligible overhead, resulting in the best speedup despite higher APS size in some cases.
- **Dataset-Specific Benefits**: The CP framework showed the most significant improvements on datasets with a large number of labels, such as HWU64 and ATIS. The speedup was more pronounced for these datasets, emphasizing the benefit of CP for tasks with many target labels.
- **Performance Comparison**: The CP-based label filtering retained the performance of the zero-shot models that used the full label set. In some cases, the accuracy even increased, suggesting that pruning the label space using CP may remove noisy labels and boost performance.

#### Practical Applications

The invention is particularly useful in practical applications where computational resources are limited, and efficiency is crucial. For example, in real-time text classification tasks, the CP framework can significantly reduce latency and resource consumption. The choice of base classifier depends on the specific requirements of the task, such as the size of the zero-shot model, the complexity of the base classifier, and the number of target labels.

#### Future Work

Future research will focus on developing more advanced methods for building conformal predictors that can further reduce the APS size and inference time. Additionally, the proposed framework can be extended to other zero-shot and few-shot learning tasks, such as prompt-based models and in-context learning, to improve efficiency and performance.

#### Limitations

The datasets used in this research are primarily in English, which may introduce cultural and linguistic biases. Pre-trained models and the CP framework may also exhibit biases, and these should be carefully evaluated before deploying the system in real-world applications. The proposed technique should be thoroughly tested for new tasks, including evaluating ethical and social risks.

In conclusion, the present invention provides a robust and efficient method for improving the performance of zero-shot text classification models. By leveraging conformal prediction to filter unlikely target labels, the invention significantly reduces computational costs and enhances the practical applicability of zero-shot models in various NLP tasks.