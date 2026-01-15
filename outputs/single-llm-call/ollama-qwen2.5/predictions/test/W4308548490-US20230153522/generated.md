- **Introduction**: This paper introduces a novel training strategy for image captioning models that leverages CLIP's multimodal similarity score to generate more distinctive and grammatically correct captions. It also presents FineCapEval, a dataset designed for fine-grained caption evaluation, addressing the limitations of reference-based metrics.

- **Background**: Image captioning models often rely on reference captions, which can introduce biases and limit the diversity of generated descriptions. CLIP, a multimodal model trained on large-scale internet data, offers a way to generate more diverse and accurate captions by focusing on the semantic alignment between images and text.

- **Methodology**: The proposed method maximizes the CLIP similarity score during training, eliminating the need for reference captions. Additionally, it fine-tunes CLIP's text encoder to improve grammatical correctness, addressing common issues like word repetition and other degenerations in generated captions.

- **FineCapEval Dataset**: FineCapEval is a dataset designed to evaluate image captions on specific criteria such as background, objects, attributes, and relations. It consists of detailed annotations for each criterion, enabling more nuanced evaluation of caption quality beyond traditional metrics.

- **Experiments**: The experiments demonstrate that models trained with the CLIP reward generate captions with higher text-to-image retrieval scores and better performance on FineCapEval compared to baselines. Human evaluations also show a strong preference for captions generated using the CLIP-S+Grammar reward.

- **Results**: Models trained with the CLIP-S and CLIP-S+Grammar rewards achieve higher multimodal similarity scores and text-to-image retrieval scores, indicating more distinctive and accurate captions. They also perform better on fine-grained evaluation criteria, as shown by both quantitative metrics and human preferences.

- **Human Evaluation**: Human annotators strongly prefer captions generated with the CLIP-S+Grammar reward over those from CIDEr and MLE baselines across all criteria, including overall quality, background, objects, attributes, and relations. This indicates that the proposed method generates more descriptive and grammatically correct captions.

- **Conclusion**: The paper concludes by highlighting the effectiveness of using CLIP's multimodal similarity score for training image captioning models, which results in more diverse and accurate captions. It also emphasizes the importance of fine-grained evaluation datasets like FineCapEval for assessing caption quality.

- **Future Work**: Future research could explore extending the proposed method to different languages and applications, as well as incorporating desired writing styles. Additionally, improving the synthetic augmentation process with advanced linguistic expertise could further enhance grammatical correctness.

- **Ethical Considerations**: The CLIP models used in this work are trained on large-scale web data, which may contain problematic content. The authors emphasize that their method is intended for research purposes and caution against deploying the models without careful consideration of ethical implications.