# DESCRIPTION

## CROSS REFERENCES

This application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 63/XXXXXX, filed on [Date], entitled “Contrastive Parameter Ensembling for Reducing Hallucinations in Abstractive Summarization Systems,” the entire contents of which are hereby incorporated by reference in their entirety for all purposes.

## TECHNICAL FIELD

The present invention relates generally to the field of artificial intelligence and natural language processing, and more specifically to systems and methods for improving the factual consistency of abstractive text summarization models by reducing hallucinations—i.e., the generation of content that is not supported by or contradicts the source document. The invention further pertains to a novel parameter ensembling framework that leverages both high-quality and low-quality training data to produce summarization models that maintain high levels of informativeness, fluency, and lexical fidelity while significantly improving factual accuracy.

## BACKGROUND

Abstractive summarization, a core task in natural language processing, involves generating concise, human-readable summaries that capture the essence of a source document without merely extracting verbatim phrases. Recent advances in deep neural networks, particularly transformer-based architectures such as BART and T5, have enabled state-of-the-art performance on benchmark datasets like XSUM and CNN/DM. However, these models frequently suffer from hallucinations—generating plausible-sounding but factually incorrect or unsupported content. Such hallucinations undermine the reliability of automated summarization systems in critical applications including journalism, legal documentation, medical reporting, and scientific literature review.

Prior approaches to mitigating hallucinations have largely focused on either post-processing corrections or data filtering. Post-processing methods attempt to revise generated summaries after the fact using auxiliary models or rule-based constraints, but they often introduce computational overhead, reduce inference speed, and may fail to correct complex semantic inconsistencies. Data filtering strategies aim to remove noisy or hallucinated samples from the training corpus before model training, thereby improving the factual quality of the resulting model. However, this approach drastically reduces the size and diversity of the training data, leading to significant degradation in standard summarization metrics such as ROUGE scores and named entity recall. This trade-off between factual consistency and summary quality has remained a persistent challenge in the field.

Moreover, existing evaluation metrics like ROUGE and BERTScore emphasize lexical or semantic overlap with reference summaries but do not adequately assess whether the generated content is factually grounded in the source document. While newer metrics based on entailment, entity overlap, and question answering have shown better correlation with human judgments of factuality, they have not been systematically integrated into the training or model refinement pipeline. Consequently, there remains a need for a computationally efficient, end-to-end framework that leverages the full spectrum of training data—both clean and noisy—to produce summarization models that are both fluent and factually reliable.

## DETAILED DESCRIPTION

The present invention introduces a novel framework termed Contrastive Parameter Ensembling (CaPE), designed to reduce hallucinations in abstractive summarization models while preserving key aspects of summary quality such as informativeness, coherence, and recall. At its core, CaPE operates by constructing three distinct models: a base summarization model trained on the entire dataset, an expert model fine-tuned on a subset of high-factual-consistency (clean) samples, and an anti-expert model fine-tuned on a subset of low-factual-consistency (noisy) samples. The final summarization model is then derived through a linear combination of the parameters of these three models, effectively amplifying the behaviors associated with factual consistency while suppressing those linked to hallucination.

A hallucination in the context of abstractive summarization refers to any generated content that is not entailed by or contradicts the source document. These can range from simple extrinsic errors—such as introducing entities not present in the source—to complex semantic distortions involving incorrect predicates, coreference mismatches, or erroneous discourse relations. Hallucinations are particularly prevalent in models trained on datasets like XSUM, where reference summaries themselves contain factual inaccuracies.

The types of hallucinations addressed by the present invention include out-of-article entity errors (extrinsic hallucinations), predicate-argument mismatches, coreference errors, and discourse-level inconsistencies. These are systematically captured using automatic factual metrics such as entity token overlap precision and Dependency Arc Entailment (DAE). The quality of training data directly influences the propensity of a model to hallucinate; models trained on noisy data inherit and amplify these errors, while those trained exclusively on clean data may sacrifice coverage and diversity.

To address this, the CaPE framework begins with a base summarization model—typically a pre-trained BART or similar architecture—trained on the full training corpus. Using an automated factual metric, the training data is partitioned into clean and noisy subsets. For instance, clean samples may be defined as those with entity precision above a threshold (e.g., 90%) or DAE error counts below a threshold, while noisy samples fall below or above these thresholds, respectively.

The expert model is obtained by fine-tuning the base model on the clean subset, thereby reinforcing behaviors that align with factual consistency. Conversely, the anti-expert is fine-tuned on the noisy subset, intentionally amplifying hallucinatory tendencies. Critically, both models share the same initialization and optimization trajectory as the base model, ensuring that parameter-space operations remain meaningful—a principle supported by recent work in weight averaging and model interpolation.

The ensembling of parameters is performed via the formula:  
θ_CaPE = θ_B + α(θ_E − θ_Ē),  
where θ_B, θ_E, and θ_Ē denote the parameters of the base, expert, and anti-expert models, respectively, and α is a mixing coefficient that controls the strength of the contrastive adjustment. This formulation generalizes prior parameter averaging techniques such as WiSE-FT by explicitly incorporating a negative component (the anti-expert) to subtract undesirable behaviors.

The advantages of CaPE include improved factual consistency across multiple metrics without significant loss in ROUGE or entity recall, computational efficiency during inference (as only one model is deployed), and flexibility in balancing factual accuracy against other summary qualities via the mixing coefficient α. Unlike ensemble methods that average model outputs, CaPE operates in parameter space, avoiding the linear increase in inference cost.

FIG. 1 illustrates the CaPE framework: a base model is trained on all data; clean and noisy subsets are selected using factual metrics; expert and anti-expert models are fine-tuned; and final parameters are ensembled contrastively.

Factual metrics used include entity token overlap precision (E-P_src)—the percentage of named entities in the summary found in the source—and DAE, which evaluates fine-grained entailment by decomposing summaries into dependency arcs and checking their support in the source. Samples are scored using these metrics; those above (below) a threshold are labeled clean (noisy).

The final summarization model is constructed by ensembling parameters with a chosen α. Alternative ensembling methods, such as simple averaging or output-level voting, are less effective due to increased computational cost or lack of contrastive signal.

Experiments were conducted on XSUM and CNN/DM datasets. The computing environment includes a processor, memory, machine-readable media, and a data interface for input/output. The Summarization module comprises submodules: Base Training, Data Filtering, Fine-Tuning, and Mixing Experts. Executable code implements the CaPE workflow.

In a networked system, user devices communicate with servers hosting the Summarization module via a network. Data vendor servers provide training datasets. A database stores models and intermediate data. Network interface components manage data flow.

### Computer and Network Environment

The present invention may be implemented on a computing device comprising one or more processors, a memory unit, and machine-readable media storing executable instructions. The processor and memory are operatively coupled to execute the Summarization module, which includes submodules for Base Training, Data Filtering, Fine-Tuning, and Mixing Experts. Input data includes source documents and reference summaries; output data includes factually consistent summaries.

The computing device may be part of a networked system including user devices, data vendor servers, and a central server. The user device runs a user interface application to submit documents and receive summaries. The server hosts the Summarization module and communicates with data vendor servers that provide training corpora. All components connect via a network, with databases storing models, datasets, and logs. Network interface components facilitate secure, efficient data exchange.

### Example Workflows

An exemplary algorithm for CaPE Summarization begins by receiving a training dataset of document-summary pairs. A base summarization model is trained on all samples. Clean and noisy subsets are selected using factual metrics such as E-P_src or DAE errors. The base model is fine-tuned on the clean subset to produce an expert, and on the noisy subset to produce an anti-expert. Parameters are combined via θ_CaPE = θ_B + α(θ_E − θ_Ē). The final model is stored in a database and transmitted to user devices upon request.

Performance evaluation shows that expert models improve factual metrics, while anti-experts degrade them—confirming their intended roles. Four CaPE variants (e.g., CaPE_PP using entity-based expert/anti-expert, CaPE_DP using DAE expert and entity anti-expert) are tested. Variables include ROUGE, E-R_ref, D_arc, E-P_src, QAFactEval, and MNLI scores.

Validation results demonstrate that CaPE consistently outperforms baseline BART and ensemble models on factual metrics, with minimal ROUGE loss. CaPE_DP achieves the best trade-off. Mixing coefficient α modulates performance: higher α increases factual consistency but may reduce recall. Summary length varies with α, reflecting the average length of training subsets.

On CNN/DM, CaPE maintains high performance due to inherently cleaner data. Fine-tuning the base model is more efficient than retraining from scratch. CaPE surpasses WiSE-FT by leveraging both expert and anti-expert signals. Human evaluations confirm superior factual consistency of CaPE-generated summaries over baselines.