### Metrics Overview

Abstractive text summarization metrics like ROUGE and BERTScore focus on lexical and semantic overlap but fall short in evaluating factuality and faithfulness. This gap has spurred research into new metrics such as entailment and question-answering based evaluations, which aim to better assess factual consistency and hallucination in summaries.

### Contrastive Parameter Ensembling (CaPE)

Contrastive Parameter Ensembling (CaPE) is a method designed to reduce content hallucinations in abstractive summarization models. It involves selecting clean or noisy training samples to fine-tune expert and anti-expert models, respectively. The difference between these parameters is then used to adjust the base summarization model, effectively reducing hallucinations without significantly impacting ROUGE scores and information recall.

### Evaluation Metrics

CaPE leverages a diverse set of metrics for evaluation, including entailment-based measures like FactCC, entity overlap metrics like E-P src, and question-answering based evaluations. These metrics are complemented by traditional ones such as ROUGE and BERTScore to provide a comprehensive assessment of factual consistency and overall summarization quality.

### Data Selection

The performance of CaPE is influenced by the average summary length of the data subset used for training the expert (anti-expert). On datasets with shorter summaries, like XSUM, CaPE tends to reduce the length of generated summaries as α increases. Conversely, on datasets with longer summaries, like CNN/DM, the average summary length increases with higher α values.

### Initialization Methods

Comparing different initialization methods for experts and anti-experts, fine-tuning a base summarization model is more efficient than training BART from scratch. Both methods improve factual consistency metrics, but fine-tuning the base model yields better ROUGE scores, making it a more practical approach for building these models.

### Performance Variations

CaPE outperforms simple parameter ensembling (WiSE-FT) by combining the benefits of both expert and anti-expert models. While individual expert or anti-expert models improve factual consistency metrics, CaPE achieves faster and more balanced performance improvements across all metrics, effectively utilizing both clean and noisy data.

### Related Work

Research on abstractive text summarization has seen a surge in methods aimed at enforcing factual consistency. These include constrained decoding, neurologic decoding, guided generation using language models, and loss truncation techniques. Additionally, post-hoc correction methods like SpanFact and contrast entity generation have been proposed. CaPE stands out by leveraging contrastive parameter ensembling to improve summarization quality.

### Conclusion

CaPE is a robust method for reducing content hallucinations in abstractive summarization models. By fine-tuning expert and anti-expert models on clean and noisy data, respectively, and adjusting the base model parameters accordingly, CaPE effectively enhances factual consistency without significantly compromising ROUGE scores or information recall, as demonstrated on the XSUM and CNN/DM datasets.