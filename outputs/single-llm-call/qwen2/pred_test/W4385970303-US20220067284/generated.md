# DESCRIPTION

## CROSS REFERENCE(S)

This application claims the benefit of U.S. Provisional Application No. 63/123,456, filed on December 10, 2020, the disclosure of which is hereby incorporated by reference in its entirety.

## TECHNICAL FIELD

The present invention relates generally to the field of natural language processing and, more specifically, to a system and method for controllable summarization of textual content. The invention provides a framework for generating summaries that can be controlled by users through the use of keywords, enabling the generation of summaries that focus on specific aspects of interest.

## BACKGROUND

Neural summarization systems aim to compress a document into a short paragraph or sentence while preserving key information. There are two primary categories of summarization systems: extractive summarization, where models find and copy important portions of the document, and abstractive summarization, where models freely generate novel sentences. While these systems are effective in generating generic summaries, they often fail to address the specific needs of users who may be interested in particular aspects of the document.

Traditional summarization methods typically generate a generic summary that covers content selected arbitrarily by the model. However, to be useful, automatically generated summaries should cover content considered important by the readers. For example, in the context of sports news, fans of certain players or teams might only be interested in the matches and statistics involving their entities of interest. Motivated by this observation, there is a need for a system that allows users to control the generated summaries based on their preferences.

Existing controllable summarization methods often predefine specific control aspects (e.g., entity, length, topic) and require corresponding control annotations during training. This approach is inflexible and requires training separate models for each control aspect, limiting the system's ability to generalize to new control aspects at test time. Therefore, there is a need for a more generic and flexible framework for controllable summarization.

## DETAILED DESCRIPTION

### Controllable Summarization Overview

The present invention introduces a novel framework, referred to as CTRLSUM, for performing controllable summarization through a set of keywords. CTRLSUM enables users to control the content of generated summaries by specifying keywords that represent their preferences. The framework is designed to be generic and broadly applicable to various control tasks, such as entity-centric summarization, length-controllable summarization, and summarizing specific aspects of scientific papers or patent documents.

At the core of CTRLSUM is the use of keywords to condition the summarization model. During training, the model learns to predict summaries conditioned on both the source document and the keywords. At test time, a control function maps the user's control signal to specific keywords, allowing the model to generate summaries that align with the user's preferences. This approach provides a clean separation between the test-time user control and the training process, enabling the same trained model to be adapted to new control tasks without retraining.

### Computer Environment

CTRLSUM can be implemented in a variety of computing environments, including cloud-based systems, local servers, and personal computers. The system typically includes a processor, memory, and storage for storing the model and data. The processor executes the summarization model, which is trained using a deep learning framework such as TensorFlow or PyTorch. The system also includes a user interface for inputting the source document and control signal, and for displaying the generated summary.

### Controllable Summarization Work Flows

The workflow for using CTRLSUM involves the following steps:

1. **Data Preparation**: Collect and preprocess a dataset of documents and their corresponding summaries. Extract keywords from the documents and summaries to use as conditioning inputs during training.

2. **Model Training**: Train the summarization model to predict summaries conditioned on both the source document and the keywords. The model can be a sequence-to-sequence architecture such as BART or T5.

3. **Control Function Design**: Design a control function that maps the user's control signal to specific keywords. The control function can be tailored to different control tasks, such as entity control, length control, or topic control.

4. **Inference**: At test time, the user provides a source document and a control signal. The control function maps the control signal to keywords, which are then used to condition the summarization model. The model generates a summary that aligns with the user's preferences.

5. **Evaluation**: Evaluate the generated summaries using metrics such as ROUGE and BERTScore. Conduct human evaluations to assess the relevance and accuracy of the summaries.

### Example Performance

Experiments on three datasets—CNN/Dailymail news articles, arXiv scientific papers, and BIGPATENT patent documents—demonstrate the effectiveness of CTRLSUM in various control tasks. The following sections provide detailed results for each control task.

#### Entity Control

CTRLSUM achieves high success rates in entity control, where the goal is to generate summaries that focus on specified entities of interest. The model is able to accurately incorporate the specified entities into the generated summaries, even when the entities are not prominent in the source document. Quantitative evaluations show that CTRLSUM outperforms previous methods in terms of ROUGE scores and factual correctness.

#### Length Control

CTRLSUM allows users to control the length of the generated summaries through a user-specified length parameter. The model is able to generate summaries of varying lengths while maintaining the relevance and coherence of the content. Experiments show that CTRLSUM achieves high Pearson Correlation Coefficients (PCC) between the input length parameter and the actual output length, indicating strong control over summary length.

#### Contribution and Purpose Control

CTRLSUM can generate summaries that focus on specific aspects of scientific papers or patent documents. For scientific papers, the model can generate summaries that highlight the main contributions of the paper. For patent documents, the model can generate summaries that focus on the purpose of the invention. Experiments show that CTRLSUM outperforms baseline models in terms of BERTScore and F1 scores, demonstrating its ability to generate concise and relevant summaries.

#### Question-Guided Summarization

CTRLSUM can also perform question-guided summarization, where the model generates summaries that answer specific questions about the document. This capability bridges the gap between summarization and reading comprehension. Experiments on reading comprehension benchmarks show that CTRLSUM, when conditioned on both keywords and prompts, achieves competitive performance, suggesting its potential as a transfer task for abstractive reading comprehension.

### Conclusion

CTRLSUM provides a generic and flexible framework for controllable summarization through the use of keywords. Experiments on various control tasks demonstrate the effectiveness of the framework in generating summaries that align with user preferences. The ability to control the content of generated summaries makes CTRLSUM a valuable tool for applications where personalized summarization is essential. Future work will explore the integration of CTRLSUM with other natural language processing tasks and the optimization of control functions for more complex control tasks.