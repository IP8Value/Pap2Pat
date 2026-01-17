# DESCRIPTION

## BACKGROUND

The field of image captioning has seen significant advancements with the advent of deep learning techniques. Traditional methods often rely on maximizing the textual similarity between generated and reference captions, which can lead to less distinctive and less detailed descriptions of images. Public datasets, such as MS COCO, typically contain reference captions that focus on the most salient objects in images, often neglecting finer details. This limitation can result in models that generate generic and repetitive captions, failing to capture the unique aspects of each image.

To address this issue, recent research has explored the use of multi-modal encoders, such as CLIP (Contrastive Language–Image Pre-training), which are trained on large-scale image-text pairs from the web. CLIP provides a powerful tool for evaluating the relevance between images and text, making it a promising candidate for guiding the generation of more distinctive and detailed captions. Additionally, the need for reference captions during training can be eliminated, reducing the reliance on potentially biased or incomplete reference data.

This invention introduces a novel training strategy for image captioning models by leveraging CLIP's similarity scores as rewards. The method also includes a finetuning strategy for CLIP's text encoder to improve the grammatical correctness of the generated captions. These innovations aim to produce captions that are not only more distinctive but also more accurate and grammatically sound.

## SUMMARY

The present invention relates to a system and method for generating image captions that are more distinctive and detailed. The method involves using the similarity scores from a multi-modal encoder, specifically CLIP, as rewards during the training of an image captioning model. This approach encourages the model to generate captions that capture the fine-grained details of images, rather than focusing solely on the most salient objects.

Furthermore, the invention includes a finetuning strategy for the CLIP text encoder using synthetic negative caption augmentation. This technique helps to improve the grammatical correctness of the generated captions by addressing common issues such as word repetition and other degeneration artifacts.

The invention also introduces a new dataset, FineCapEval, designed for fine-grained caption evaluation. This dataset measures the performance of captioning models in various aspects, including background, object, and relation between objects. The dataset provides a comprehensive framework for evaluating the quality of generated captions.

## DETAILED DESCRIPTION

### Image Search System

The image search system described herein is designed to generate high-quality, distinctive captions for images. The system comprises several key components: a multi-modal encoder (CLIP), a captioning model, and a finetuning mechanism for the text encoder. The following sections detail the operation and benefits of each component.

#### Multi-Modal Encoder (CLIP)

CLIP is a pre-trained multi-modal encoder that has been trained on a large dataset of image-text pairs. It consists of two encoders: an image encoder and a text encoder. The image encoder maps images into a high-dimensional feature space, while the text encoder maps text into the same feature space. The similarity between an image and a caption is then calculated as the dot product of their respective embeddings.

The use of CLIP's similarity scores as rewards during training encourages the captioning model to generate captions that are more relevant and distinctive to the input image. This is achieved by maximizing the multimodal similarity score, which ensures that the generated captions capture the unique aspects of the image, including background details, object attributes, and relationships between objects.

#### Captioning Model

The captioning model is a neural network trained to generate captions for images. The model is optimized using the REINFORCE algorithm with a self-critical baseline. The reward function is defined as the CLIP similarity score between the generated caption and the input image. This training strategy ensures that the model generates captions that are not only relevant but also diverse and detailed.

The captioning model can be implemented using various architectures, such as LSTM (Long Short-Term Memory) or Transformer models. The choice of architecture depends on the specific requirements of the application, such as computational efficiency and the complexity of the images being captioned.

#### Finetuning Mechanism for Text Encoder

To improve the grammatical correctness of the generated captions, the CLIP text encoder is finetuned using synthetic negative caption augmentation. This process involves generating negative captions by applying various operations to the reference captions, such as repeating, removing, inserting, swapping, and shuffling tokens. These negative captions are then used to train the text encoder to recognize and correct grammatical errors.

The finetuning process is crucial for ensuring that the generated captions are not only relevant and distinctive but also grammatically correct. This is particularly important for applications where the quality of the generated text is critical, such as in accessibility tools for the visually impaired.

#### FineCapEval Dataset

FineCapEval is a new dataset introduced to evaluate the performance of captioning models in a fine-grained manner. The dataset consists of 1,000 images, each annotated with phrases describing the background, objects, and relations between objects. Additionally, each image is accompanied by a detailed caption that integrates all three aspects.

The dataset is designed to measure the performance of captioning models in various criteria, including overall caption quality, background description, object description, and relation between objects. The annotations are collected from human annotators, ensuring that the dataset captures a wide range of descriptive elements.

#### Experimental Setup and Evaluation

To validate the effectiveness of the proposed method, experiments were conducted on the MS COCO dataset. The captioning model was trained using different reward functions, including MLE (Maximum Likelihood Estimation), CIDEr, CLIP-S, and a combination of CLIP-S and grammar improvement. The performance of the models was evaluated using various metrics, including n-gram based metrics, embedding-based metrics, text-to-image retrieval scores, and the FineCapEval dataset.

The results showed that the models trained with CLIP-S and CLIP-S+Grammar rewards generated captions that were more distinctive and contained more detailed information compared to the baselines. The text-to-image retrieval scores were even higher than those achieved with reference captions, indicating the superior distinctiveness of the generated captions.

Human evaluations were also conducted to assess the preference for the generated captions. The results showed that human annotators strongly preferred the captions generated by the models trained with CLIP-S+Grammar rewards over those generated by the baselines.

#### Conclusion

The invention provides a novel and effective method for generating high-quality, distinctive image captions. By leveraging the similarity scores from CLIP and finetuning the text encoder, the method ensures that the generated captions are not only relevant and detailed but also grammatically correct. The introduction of the FineCapEval dataset further enhances the evaluation of captioning models, providing a comprehensive framework for assessing their performance in various aspects.

Future work will focus on extending the method to support different writing styles and languages, as well as exploring the use of external data to improve the synthetic augmentation process. The invention has the potential to significantly impact various applications, including image search engines, accessibility tools, and content generation systems.