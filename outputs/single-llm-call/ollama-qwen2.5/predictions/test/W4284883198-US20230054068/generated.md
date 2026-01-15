This paper introduces Entity Control Code (ECC) to reduce extrinsic hallucination in abstractive summarization for both supervised and zero-shot settings. ECC uses control codes during training and inference to guide the model towards more faithful summaries without sacrificing quality. The method is applied to BART-large, a pre-trained seq2seq model.

Experiments show ECC effectively reduces entity-level hallucinations across multiple datasets like XSum and PubMed. Compared to baselines, ECC generates more factual outputs while maintaining fluency and salience. The approach outperforms post-processing methods that can introduce new errors.

In the zero-shot setting, ECC improves cross-domain generalization by incorporating domain-specific control codes during pre-training. This allows the model to adapt to unseen domains without fine-tuning. Representative examples demonstrate how ECC corrects hallucinations from vanilla BART outputs.

The paper analyzes why models like BART generate erroneous entities, finding that pre-trained knowledge can lead to incorrect inferences when contextual information is limited or ambiguous. By controlling entity generation with explicit guidance, ECC mitigates this issue.

ECC builds on prior work in controllable summarization and factuality evaluation. It differs by focusing specifically on entity-level hallucinations and using control codes for both training and inference. The method is generally applicable to seq2seq models beyond BART.

The authors acknowledge limitations like not addressing factual statements verifiable independent of source text. They also note potential improvements through more fine-grained control over entity compression rates. Overall, ECC represents a promising step towards more faithful abstractive summarization systems.