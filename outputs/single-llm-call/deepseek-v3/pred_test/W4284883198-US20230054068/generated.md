Here is the complete patent application following your specified outline:

# DESCRIPTION

## TECHNICAL FIELD

The present invention relates generally to machine learning systems for document summarization, and more specifically to systems and methods for improving faithfulness in abstractive summarization through entity coverage control mechanisms. The disclosed techniques address the critical problem of hallucination in neural sequence-to-sequence summarization models by introducing novel control codes based on entity coverage precision metrics. The invention further provides methods for domain-generalizable zero-shot summarization through controllable intermediate fine-tuning pipelines. These technical advancements enable the generation of more accurate and reliable summaries across diverse document types while maintaining high levels of fluency and salience.

## BACKGROUND

Prior approaches to document summarization have faced significant limitations in ensuring the faithfulness of generated outputs. Conventional abstractive summarization systems based on pre-trained language models frequently produce summaries containing hallucinated or fabricated information not present in source documents. Existing solutions to this problem have included post-processing correction methods that replace hallucinated entities and multi-task learning approaches that filter training data. However, these methods introduce new errors or sacrifice training data quality. Other techniques relying on textual entailment, question answering, or natural language inference for faithfulness evaluation add computational complexity without fully solving the hallucination problem at the entity level. The present invention overcomes these limitations by integrating entity coverage precision metrics directly into the model training process through novel control code mechanisms, enabling faithful summary generation without compromising model performance or requiring extensive post-processing.

## DETAILED DESCRIPTION

The following terms are used throughout this disclosure with the specified meanings: "Entity coverage precision" refers to the ratio of named entities in a summary that originate from the source document. "Control code" denotes a special token prepended to input documents that guides the generation process. "Hallucination" describes generated content not supported by the source material. "Zero-shot summarization" indicates the ability to generate summaries for unseen document types without target-specific training.

The disclosed machine learning methods for document summarization employ sequence-to-sequence transformer architectures enhanced with entity coverage precision metrics and control code mechanisms. The system calculates entity coverage precision for each document-summary pair in the training set, quantifying the proportion of summary entities derived from the source document. This precision metric serves as a faithfulness indicator, with higher values corresponding to more faithful summaries.

A key innovation involves the entity coverage precision metric for evaluating summary faithfulness. This metric computes the ratio between the count of named entities in the summary that appear in the source document and the total named entities in the summary. Mathematically, it is expressed as prec_en = |N(s_i) ∩ N(d_i)| / |N(s_i)|, where N(t) represents the set of named entities in text t. This precision measurement provides a quantifiable faithfulness signal for model training.

The pseudo labeling module generates discrete control codes by binning the continuous entity coverage precision values. The system partitions the precision range into k bins with boundaries selected to maintain balanced training example distribution across bins. Each bin receives a unique control code token added to the model vocabulary. During training, the appropriate control code is prepended to each input document, enabling the model to learn generation patterns corresponding to different faithfulness levels.

The summarization module integrates these control codes with a sequence-to-sequence architecture. The model processes both the source document and its prepended control code to generate conditioned outputs. For inference, high-faithfulness control codes ensure minimal entity hallucination while maintaining summary quality. The system employs transformer-based encoder-decoder structures with attention mechanisms, specifically utilizing BART-large as a backbone model while remaining compatible with other seq2seq architectures.

### Systems for Abstractive Summarization

The abstractive summarization system comprises several integrated modules that collectively improve faithfulness and domain adaptability. The pre-processing module handles document ingestion, tokenization, and named entity recognition using specialized NLP toolkits like Stanza and scispaCy depending on domain requirements. This module identifies and extracts entities for subsequent precision calculations.

The entity coverage precision module computes faithfulness metrics for each training example by comparing source and summary entities. It implements the prec_en formula and maintains entity mappings between documents and their reference summaries. This module outputs continuous precision values that feed into the pseudo labeling process.

The pseudo labeling module discretizes continuous precision values into k bins, assigning each a control code token. It employs dynamic bin boundary adjustment to maintain balanced training distributions across bins. The module generates three primary control codes (<FF-low>, <FF-mid>, <FF-high>) representing increasing faithfulness levels, though the system supports arbitrary bin counts.

The summarization module implements the core sequence-to-sequence model architecture. Based on BART-large, it incorporates transformer encoder and decoder components with 336M parameters. The encoder processes document tokens concatenated with control codes, while the decoder generates summary tokens conditioned on both the encoded representation and control signal. The model optimizes a cross-entropy loss function during training.

The system introduces faithfulness control codes as special tokens prepended to input sequences. These discrete-valued signals condition the generation process to favor entity-preserving outputs. During inference, selecting higher-faithfulness codes reduces hallucination while maintaining fluency. The codes remain compatible with existing transformer architectures without requiring structural modifications.

The entity coverage precision metric calculation involves named entity recognition on both source documents and reference summaries. The system counts matching entities between each document-summary pair to compute the precision ratio. This process uses domain-appropriate NER models, such as biomedical-specific recognizers for scientific articles.

Pseudo label generation transforms continuous precision values into discrete control codes through quantile-based binning. The system determines bin boundaries to evenly distribute training examples, then assigns each example to its corresponding bin and associated control code. These codes become additional vocabulary tokens learned during training.

Control code generation produces the special tokens that condition model behavior. The system creates unique tokens for each precision bin and target domain, adding them to the model vocabulary. During training and inference, these tokens prepend input documents to guide generation toward desired faithfulness levels or domain characteristics.

The summarization model training process incorporates control codes as additional input features. The system fine-tunes the base model on document-summary pairs with prepended control codes, teaching the model to associate codes with specific generation patterns. Training uses standard seq2seq objectives with cross-entropy loss computed over summary tokens.

The transformer encoder processes input sequences comprising control codes followed by document tokens. It generates contextual representations that capture both content and control signal information. The encoder stack consists of multiple self-attention layers that build progressively higher-level representations.

The transformer decoder generates output summaries autoregressively while attending to encoder states. It produces each summary token conditioned on previous outputs, encoder representations, and the control code signal. The decoder's cross-attention mechanisms ensure content relevance while maintaining control code influence throughout generation.

Output summary generation occurs through standard autoregressive sampling from the decoder's output distribution. The system can employ various decoding strategies like beam search or nucleus sampling while maintaining the conditioning effect of the control code throughout the generation process.

The training objective minimizes cross-entropy loss between generated and reference summaries while implicitly learning control code associations. The model learns to adjust generation faithfulness based on the prepended code without explicit faithfulness supervision beyond the initial bin assignments.

For domain adaptation, the system employs domain-specific databases and intermediate pre-training pipelines. It processes Wikipedia article dumps to create pseudo training data matching target domain characteristics in length, abstractiveness, and entity distributions. This enables effective zero-shot transfer to new domains.

The Wikipedia corpus serves as a source for generating diverse pseudo training examples. The system extracts articles and creates artificial summaries with controlled properties, building a rich intermediate dataset for cross-domain generalization. This corpus provides broad coverage of entities and topics to support domain transfer.

The intermediate pre-training pipeline fine-tunes models on Wikipedia-derived pseudo data before target domain adaptation. This pipeline creates training instances with controlled summary lengths, document lengths, and abstractiveness levels matching various target domains. The process preserves entity coverage faithfulness through control codes.

Target datasets benefit from the system's zero-shot capabilities through domain control codes. The system generates target-specific pseudo data from Wikipedia, assigns domain control codes, and includes these in intermediate training. During inference, selecting the appropriate domain code steers generation toward target-appropriate outputs.

Abstractiveness level control complements entity coverage precision in the training data construction. The system measures and controls the degree of abstraction in generated pseudo summaries to match target domain expectations, whether highly abstractive (e.g., news) or more extractive (e.g., scientific).

Training instance construction for intermediate pre-training involves generating document-summary pairs from Wikipedia with controlled properties. The system processes articles to create pseudo summaries of specified lengths and abstractiveness levels while maintaining high entity coverage precision through selective entity inclusion.

Pseudo label generation for domain adaptation creates target-specific control codes analogous to faithfulness codes. These domain codes enable steering generation toward particular domains or styles during zero-shot inference. The system supports stacking domain and faithfulness codes for combined control.

Training set construction for intermediate pre-training combines Wikipedia-derived pseudo data with multiple control signals. Each training instance receives both a faithfulness control code based on its entity coverage precision and a domain control code indicating its target domain characteristics.

Summarization model training for zero-shot capability incorporates both domain and faithfulness control codes. The model learns to generate outputs conditioned on stacked control signals, enabling simultaneous control over domain appropriateness and entity faithfulness during inference.

Zero-shot summarization operates by selecting appropriate domain and faithfulness control codes for unseen target domains. The system prepends these codes to input documents, guiding the model to generate domain-appropriate, faithful summaries without target-specific fine-tuning.

Target-specific intermediate data generation creates pseudo training examples mimicking various domain characteristics. The system processes Wikipedia articles to produce datasets with properties matching news, scientific papers, dialogues, or other target types, enabling broad zero-shot capability.

Wikipedia article processing involves extracting clean text, identifying named entities, and generating pseudo summaries with controlled properties. The system measures article characteristics and generates matching summaries while tracking entity coverage to ensure faithfulness.

Summary generation for pseudo training data employs template-based and neural methods to create diverse examples. The system varies phrasing and structure while maintaining content accuracy and specified abstractiveness levels, producing rich training data for domain generalization.

Article generation complements summary creation in pseudo training data construction. The system can generate or modify Wikipedia-style articles to match desired characteristics, creating balanced document-summary pairs for controlled intermediate training.

Training instance construction for zero-shot learning combines generated articles and summaries with multiple control signals. Each instance receives faithfulness codes based on entity coverage and domain codes indicating target characteristics, teaching the model multi-factor control.

Pseudo label generation for zero-shot training creates both faithfulness and domain control codes. These discrete signals enable precise control over generation characteristics during inference, allowing adaptation to unseen domains without additional training.

Training set construction for zero-shot summarization combines diverse pseudo examples with stacked control codes. The resulting dataset enables models to learn complex generation control patterns supporting faithful, domain-appropriate summarization across varied contexts.

Summarization model training for generalizability emphasizes learning from control code signals. The model develops the capacity to adjust faithfulness and domain characteristics based on prepended codes, facilitating effective zero-shot transfer to new summarization tasks.

Zero-shot summarization performance benefits from the system's control code architecture. By prepending appropriate domain and faithfulness codes, the model generates suitable summaries for unseen document types without task-specific fine-tuning, demonstrating cross-domain generalization.

Generalizability across domains results from the system's intermediate pre-training approach. By exposing the model to diverse pseudo training data with explicit control signals, it learns to adapt generation characteristics to varying domain requirements through code selection.

## EXAMPLES

### Example 1: Experimental Methods and Results

The experimental setup evaluated the invention's effectiveness across multiple datasets and compared it against state-of-the-art baselines. Testing covered news (XSum), scientific papers (Pubmed), and dialog (SAMsum) domains to demonstrate broad applicability. The system used BART-large as its backbone model, fine-tuned with control codes on each dataset.

The news dataset evaluation employed the XSum benchmark containing BBC articles and single-sentence summaries. The scientific paper dataset used Pubmed citations and abstracts, while the dialog dataset utilized SAMsum conversations with summaries. Each presented unique challenges in length, style, and entity distributions that tested the system's adaptability.

Comparison with baseline methods included the original BART model, post-processing correction approaches, and data filtering techniques. The evaluation measured both faithfulness improvements and potential quality trade-offs across these competing methods. The system demonstrated superior faithfulness without sacrificing summary quality.

Evaluation metrics included both automated measures and human assessments. Rouge scores (1, 2, and L) quantified summary fluency and salience relative to references. BERTSCORE measured semantic similarity, while the novel Entity Coverage Precision metric specifically evaluated faithfulness at the entity level. FEQA provided additional factual consistency measurement.

The Rouge metric implementation followed standard practices using stemmed, stopword-filtered token matching. Rouge-1 measured unigram overlap, Rouge-2 bigram overlap, and Rouge-L longest common subsequence. These metrics assessed the system's ability to produce fluent, content-rich summaries comparable to references.

BERTSCORE evaluation employed the standard implementation calculating cosine similarity between contextual embeddings of generated and reference summaries. This semantic similarity measure complemented lexical overlap metrics by capturing paraphrase equivalences and meaning preservation.

The Entity Coverage Precision metric specifically addressed faithfulness by computing the ratio of summary entities present in the source document. This direct measurement of entity hallucination provided clear insight into the system's faithfulness improvements over baselines.

FEQA metric implementation used question generation and answering to evaluate factual consistency. The approach generated questions from the summary, answered them using the source document, and measured answer similarity as a proxy for factual alignment between source and summary.

Human evaluation complemented automated metrics with expert judgments of faithfulness and quality. Annotators categorized summaries as faithful, intrinsically hallucinated, or extrinsically hallucinated while also rating overall quality on a three-point scale. This provided real-world performance assessment.

The implementation utilized Huggingface libraries for model training and evaluation. The system built upon the BART-large architecture available through these libraries, ensuring reproducibility and compatibility with standard NLP toolchains. All experiments ran on Tesla A100 GPU clusters.

The BART-large model configuration used 12-layer transformer encoder and decoder stacks with 16 attention heads and 1024-dimensional embeddings. Fine-tuning employed Adam optimization with learning rate 5e-5 and weight decay, running for sufficient epochs to ensure convergence.

Fine-tuning process details included batch size optimization and gradient accumulation settings appropriate for the available GPU memory. The system maintained consistent hyperparameters across experiments to enable fair comparison, with control code count k=3 as the standard configuration.

Entity recognition for precision calculations used the Stanza NLP toolkit trained on OntoNotes for general domains, with scispaCy handling biomedical entity recognition in scientific papers. This dual approach ensured accurate entity identification across diverse text types.

The Stanza NLP toolkit implementation provided state-of-the-art named entity recognition with configurable models. For non-scientific texts, the system used the default OntoNotes-trained model identifying PERSON, ORGANIZATION, and LOCATION entities among others.

The scispaCy toolkit specialized in biomedical and scientific entity recognition, identifying drug compounds, proteins, diseases, and other domain-specific entities. This ensured accurate entity coverage calculations for scientific paper summarization.

Example article and summary pairs illustrated the system's improvements over baselines. One case showed the original BART model hallucinating "Gumtree" as a website name, while the invention correctly preserved the source's "YouTube" reference through entity coverage control.

Performance on downstream datasets demonstrated consistent faithfulness improvements. The system achieved 15-20% higher entity coverage precision than baselines while maintaining comparable Rouge scores, showing hallucination reduction without quality loss.

Comparison with state-of-the-art methods revealed advantages over both post-processing and data filtering approaches. The system avoided post-processing errors and preserved full training data utility while achieving better faithfulness metrics than either alternative.

The Wikipedia intermediate pre-training study showed significant zero-shot performance gains. Models pre-trained with control codes on Wikipedia data outperformed direct fine-tuning approaches when applied to unseen domains, demonstrating effective knowledge transfer.

Zero-shot performance results highlighted the system's cross-domain capabilities. Using domain control codes, the model generated faithful summaries for news, scientific papers, and dialogues without target-specific training, achieving Rouge scores within 5% of fine-tuned baselines.

Human evaluation results confirmed significant reductions in both intrinsic and extrinsic hallucination categories. Expert annotators rated the system's outputs as more faithful than baselines while maintaining equivalent quality scores, validating the automated metric findings.

Entity distribution analysis showed the system neither under-generated nor over-generated entities compared to baselines. Histograms of entity counts per summary demonstrated similar distributions, disproving any potential "safe generation" strategy of simply reducing entity mentions.

Control code investigation revealed expected faithfulness gradations across code levels. High-faithfulness codes produced the most accurate entity preservation, while low codes allowed more abstraction but maintained reasonable faithfulness, providing adjustable control.