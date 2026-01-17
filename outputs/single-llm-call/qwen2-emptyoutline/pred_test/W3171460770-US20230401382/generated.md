# DESCRIPTION

## FIELD

The field of this invention relates to natural language processing (NLP) and, more specifically, to systems and methods for dynamically updating and training language models to adapt to continuously evolving content. The invention addresses the challenges posed by the rapid emergence of new words and the semantic shift of existing words in dynamic environments such as social media platforms.

## BACKGROUND

Language is a living entity that evolves over time, reflecting changes in society, culture, and technology. This evolution is particularly pronounced in digital communication channels such as social media, where new words and phrases emerge rapidly, and the meanings of existing words can shift significantly. Traditional language models, such as BERT, are typically trained on static datasets and may perform poorly when applied to new, evolving content. This degradation in performance is primarily due to two factors: vocabulary shift and semantic shift.

Vocabulary shift refers to the introduction of new words and the obsolescence of older ones. For instance, during the COVID-19 pandemic, terms like "Covid" and "Zoom" became widely used and were added to the Oxford English Dictionary. Similarly, the usage and context of existing words can evolve, leading to semantic shift. For example, the term "flattening the curve" transitioned from a technical scientific term to a common phrase during the pandemic.

These challenges are particularly acute for pre-trained transformer-based language models like BERT, which rely on a fixed vocabulary and pre-trained embeddings. While BERT has shown remarkable success in various NLP tasks, its performance can degrade significantly when applied to dynamic content. Prior works have explored various strategies to address these issues, including incremental learning and dynamic vocabulary updates, but a comprehensive solution that effectively adapts BERT to continuously evolving content remains elusive.

## SUMMARY

The present invention provides a system and method for dynamically updating and training a language model, such as BERT, to adapt to continuously evolving content. The invention addresses the challenges of vocabulary shift and semantic shift by dynamically updating the model's vocabulary and incrementally pre-training the model with new data.

The invention includes the following key components:

1. **Dynamic Vocabulary Update**: A method for dynamically updating the model's vocabulary by adding emerging wordpieces and removing stale ones. This ensures that the vocabulary remains up-to-date and relevant to the evolving content.

2. **Effective Sampling for Incremental Training**: Three sampling methods for selecting representative examples from new data to incrementally train the model:
   - **Token Embedding Shift Method**: Identifies tokens with significant changes in their embeddings and uses them to sample tweets.
   - **Sentence Embedding Shift Method**: Measures the embedding shift at the sentence level and uses it to sample tweets.
   - **Token MLM Loss Method**: Uses the Masked Language Modeling (MLM) loss to identify hard examples and sample tweets.

3. **Production System Architecture**: A conceptual architecture for deploying the model in a production environment, including continuous monitoring of the model's performance and automatic initiation of incremental training when performance degradation is detected.

The invention significantly reduces the computational cost of training by leveraging incremental learning and effective sampling, while maintaining or improving the model's performance on evolving content. The system is particularly useful for applications involving dynamic content, such as social media monitoring, real-time sentiment analysis, and content moderation.

## DETAILED DESCRIPTION

### Overview

The invention provides a system and method for dynamically updating and training a language model, such as BERT, to adapt to continuously evolving content. The system addresses the challenges of vocabulary shift and semantic shift by dynamically updating the model's vocabulary and incrementally pre-training the model with new data. The key components of the invention include dynamic vocabulary update, effective sampling for incremental training, and a production system architecture for deployment.

### Example Methods

#### Dynamic Vocabulary Update

The dynamic vocabulary update method involves periodically updating the model's vocabulary to reflect the evolving content. This is achieved by adding emerging wordpieces and removing stale ones, ensuring that the vocabulary remains up-to-date and relevant. The process is as follows:

1. **Vocabulary Shift Analysis**: Analyze the vocabulary shift by comparing the top frequent tokens (natural words, wordpieces, and hashtags) from different time periods. This helps identify the most frequent new tokens and the least frequent stale tokens.

2. **Vocabulary Composition for Hashtags**: For hashtag-sensitive tasks, include popular whole hashtags in the vocabulary as intact tokens. This preserves the strong topical information carried by hashtags.

3. **Algorithm for Vocabulary Update**: Implement an algorithm to add the most frequent new wordpieces and remove the least frequent stale ones from the vocabulary. The algorithm ensures that the vocabulary size remains constant for efficient model parameterization.

#### Effective Sampling for Incremental Training

The effective sampling methods are designed to select representative examples from new data to incrementally train the model. The three proposed methods are:

1. **Token Embedding Shift Method**:
   - **Step 1**: Compute the cosine distance between a token's embedding from the updated model and its preceding version.
   - **Step 2**: Identify new tokens in the first iteration and top X tokens with the largest embedding shift in subsequent iterations.
   - **Step 3**: Assign large weights to tweets containing tokens with large embedding shifts and linearly combine embedding cosine distance and normalized tweet length as the sampling weight.

2. **Sentence Embedding Shift Method**:
   - **Step 1**: Measure the embedding shift at the sentence level using the [CLS] token embedding.
   - **Step 2**: Assign larger weights to longer sentences and use the combination of embedding cosine distance and tweet length to perform weighted random sampling.

3. **Token MLM Loss Method**:
   - **Step 1**: Modify the Masked Language Modeling (MLM) loss to identify hard examples by masking out tokens from the last layer of the pre-trained model.
   - **Step 2**: Use the surrounding tokens to predict the masked tokens and compute the MLM loss.
   - **Step 3**: Perform weighted random sampling based on the MLM loss and normalized tweet length.

### Example Devices and Systems

#### Production System Architecture

The production system architecture for deploying the dynamic BERT model includes the following components:

1. **Initial Model Pre-training**:
   - Pre-train the base model using vocabulary and tweets derived from a particular "base" time-period.
   - Fine-tune the base model with task-specific data and deploy it to serve real-time traffic.

2. **Continuous Monitoring**:
   - Continuously monitor the model's performance by computing the token MLM loss on real-time data.
   - Detect performance degradation and trigger incremental training when necessary.

3. **Incremental Training**:
   - Draw hard examples from the new data using the proposed sampling methods.
   - Update the model vocabulary and incrementally pre-train the model with the hard examples.
   - Fine-tune the model for the specific task and deploy the updated model.

4. **Deployment**:
   - Deploy the updated model to serve live traffic.
   - Repeat the process to keep the model up-to-date with the evolving data stream.

### Additional Disclosure

#### Training Cost Savings

Compared to training a base model from scratch, the proposed architecture for training an incremental model significantly reduces the training cost. The cost of incremental training is only 600k steps, saving 2M steps and yielding a cost savings of 76.9% relative to the base model.

#### Evaluation

The model performance is evaluated on two downstream tasks:

1. **Country Hashtag Prediction**:
   - Predict the associated country hashtag for a tweet from a pre-defined country list.
   - Evaluate the model using micro-F1, macro-F1, and accuracy scores.

2. **OffensEval 2019**:
   - Identify if tweets are offensive.
   - Evaluate the model using F1 score and AUC-ROC.

#### Results and Analysis

The experimental results demonstrate the effectiveness of the proposed dynamic modeling and efficient incremental training strategies. The incremental models outperform the base models trained on past data, confirming the necessity of adapting the model to new content. The proposed sampling methods, particularly the Token MLM Loss method, show significant improvements over baseline methods, making them suitable for continuous model adaptation while serving live traffic.

The invention provides a robust solution for dynamically updating and training language models to adapt to continuously evolving content, addressing the challenges of vocabulary shift and semantic shift. This system is particularly valuable for applications involving dynamic content, such as social media monitoring, real-time sentiment analysis, and content moderation.