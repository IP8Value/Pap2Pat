- This study demonstrates how conformal prediction (CP) can enhance zero-shot text classification efficiency by filtering target labels. CP reduces computational costs and maintains performance across various datasets and models, particularly benefiting tasks with large label spaces like HWU64.

- The research highlights that a strong base classifier, such as CP-CLS or CP-Distil, generally provides better average prediction set (APS) sizes, leading to faster zero-shot inference. However, these classifiers can be slower in generating the label set for the 0shot model.

- A simpler and efficient CP base classifier like token overlap matching may achieve the best inference time despite having a higher APS size. This is due to its minimal computational overhead, making it ideal for smaller label spaces or when the zero-shot model is less complex.

- The effectiveness of CP in practical applications depends on the sizes of the zero-shot model, the base classifier, and the label space. For large label spaces or complex zero-shot models, a stronger base classifier like BERT or RoBERTa is recommended. Otherwise, faster but weaker classifiers are more suitable.

- The conformal prediction framework can be extended to other zero-shot or few-shot models, such as prompt-based models that generate verbalizers. It can also filter training examples for in-context learning prompts, reducing computational costs and potentially improving performance by minimizing noisy labels.

- Limitations of the study include the use of English-language datasets, which may introduce cultural, gender, or age biases. Additionally, CP-based label filtering could introduce new biases, though this risk is considered low. Future work will explore methods to further reduce APS sizes and boost zero-shot model performance.

- The authors recommend that any subsequent usage of the proposed technique clearly states its use for label filtering and thoroughly evaluates it for ethical and social risks before applying it to new tasks. This ensures responsible and effective application of CP in practical scenarios.

- Experimental details reveal that the entire training set of intent datasets and a subset of validation sets from topic datasets were used for calibration. The base classifier was fine-tuned using AdamW optimizer, and the best checkpoint was chosen based on highest accuracy on the calibration dataset. This ensures robust performance of the CP framework across different tasks.

- For smaller-sized calibration sets, empirical coverage is worse than nominal coverage, but this difference is negligible at low error rates (α = 0.01). Thus, even with a small calibration set, CP can improve zero-shot classification efficiency without significant performance drops, provided a low α value is used.

- Calibration sets from other datasets can be used for CP if the error rate (α) is set to a low value. However, this results in larger APS sizes compared to using target task-specific calibration data. This flexibility allows for broader application of CP in scenarios where target task data may be limited.