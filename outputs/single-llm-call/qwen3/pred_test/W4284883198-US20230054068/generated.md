# DESCRIPTION

## TECHNICAL FIELD

- relate to machine learning systems for document summarization

The present invention relates to machine learning systems for document summarization, particularly to abstractive summarization architectures that incorporate faithfulness control mechanisms to mitigate hallucinated content in generated summaries. The systems and methods disclosed herein are configured to generate concise, fluent, and factually grounded summaries from source documents by integrating entity-level faithfulness signals directly into the training and inference processes of sequence-to-sequence models. These systems are designed to operate across diverse textual domains, including news articles, scientific publications, and conversational transcripts, without requiring domain-specific retraining or external fact-checking modules. The invention enables precise control over the degree of factual alignment between source documents and generated summaries through the use of learned control codes that encode quantified measures of entity coverage precision, thereby improving the reliability and practical utility of automated summarization in high-stakes applications such as legal documentation, medical reporting, and journalistic aggregation.

## BACKGROUND

- describe limitations of previous document summarization techniques

Previous approaches to abstractive document summarization have relied predominantly on end-to-end sequence-to-sequence models trained to maximize surface-level fluency and lexical overlap with reference summaries, often at the expense of factual accuracy. These models, including those based on transformer architectures and pre-trained language models such as BART and T5, frequently generate summaries containing entities, events, or relationships not present in the source document—a phenomenon known as hallucination. Such hallucinations manifest as incorrect named entities, fabricated actions, or misleading contextual associations, which undermine the trustworthiness of the output in real-world applications. Post-hoc correction methods have been proposed to replace hallucinated entities with plausible alternatives drawn from the source text; however, these approaches introduce new errors by misaligning entity types or omitting critical context, thereby degrading overall summary quality. Alternative strategies involving data filtering or multi-task learning with named entity classification reduce hallucination by discarding training examples with low entity alignment, but this comes at the cost of diminished model generalization and reduced summary salience. Furthermore, existing methods lack the ability to dynamically adjust the level of faithfulness during inference, making them inflexible to varying user requirements or domain-specific standards for factual rigor. No prior system has successfully integrated a quantifiable, continuous measure of entity-level faithfulness—specifically entity coverage precision—into the training objective of an abstractive summarizer in a manner that preserves fluency while enabling precise, controllable, and generalizable faithfulness enhancement.

## DETAILED DESCRIPTION

- define terms used in the disclosure

For the purposes of this disclosure, the term “document” refers to any structured or unstructured sequence of natural language text from which a summary is to be generated, including but not limited to news articles, scientific papers, legal briefs, and transcribed dialogues. The term “summary” denotes a condensed representation of the document’s salient content, produced by an automated system and intended to preserve key information while reducing length. “Entity” refers to a named entity recognized by a named entity recognition system, including persons, organizations, locations, dates, and other semantic categories as defined by standard annotation schemes such as OntoNotes or scispaCy. “Entity coverage precision” is defined as the proportion of named entities present in the summary that are also explicitly mentioned in the source document, calculated as the intersection of entity sets divided by the total number of entities in the summary. “Control code” refers to a special token appended to the input sequence of a sequence-to-sequence model, which encodes a discrete representation of a quantified faithfulness level derived from entity coverage precision. “Pseudo label” denotes a derived annotation generated algorithmically from the alignment between a source document and its corresponding reference summary, used to train the model without manual annotation. “Faithfulness control code” is a control code specifically designed to signal the model to generate summaries with a specified degree of entity-level factual alignment. “Target control code” refers to a control code that encodes domain-specific summarization characteristics, such as abstractiveness level, document length, or stylistic conventions, derived from intermediate pre-training on domain-matched corpora.

- describe machine learning methods for document summarization

The disclosed invention employs a sequence-to-sequence transformer architecture as the foundational framework for abstractive summarization, wherein a transformer encoder processes the input document and a transformer decoder generates the summary token-by-token. The model is trained using a cross-entropy loss function that maximizes the likelihood of the reference summary given the input document and a control code. During training, the model is conditioned not only on the raw document text but also on a control code that encodes the entity coverage precision of the corresponding reference summary. This conditioning enables the model to learn distinct generation patterns corresponding to varying levels of factual fidelity, allowing it to produce summaries that are both fluent and faithful when prompted with appropriate control codes during inference. The model architecture incorporates a shared embedding space that includes both standard vocabulary tokens and special control code tokens, enabling seamless integration of faithfulness signals into the attention mechanisms of the transformer layers.

- introduce entity coverage precision metric for faithfulness

The invention introduces a novel metric for evaluating and controlling the faithfulness of generated summaries, termed entity coverage precision, which quantifies the proportion of named entities in the summary that are directly supported by the source document. This metric is computed by first extracting all named entities from both the source document and the reference summary using a domain-appropriate named entity recognition system, such as Stanza for general text or scispaCy for biomedical literature. The intersection of these entity sets is then divided by the total number of entities in the summary to yield a scalar value between zero and one. This value serves as the basis for generating pseudo labels that are used to train the summarization model to associate specific levels of entity alignment with corresponding control codes. Unlike prior metrics that rely on question-answering or natural language inference, entity coverage precision provides a direct, interpretable, and computationally efficient measure of factual grounding at the entity level, enabling precise control over hallucination without requiring external knowledge bases or additional neural classifiers.

- describe pseudo labeling module for generating control codes

The pseudo labeling module operates by computing the entity coverage precision for each document-summary pair in the training corpus. These precision values are then discretized into a predefined number of bins, each representing a range of faithfulness levels, such as low, medium, and high. The bin boundaries are selected to ensure approximately equal distribution of training examples across bins, thereby preventing bias toward any particular faithfulness level. Each bin is assigned a unique control code token, such as <FF-low>, <FF-mid>, or <FF-high>, which is added to the model’s vocabulary. For each training instance, the corresponding control code is prepended to the input document, forming a modified input sequence that the model learns to map to the reference summary. This process effectively transforms the summarization task into a conditional generation problem where the model learns to produce summaries with varying degrees of faithfulness based on the provided control code, without altering the underlying document content or requiring manual labeling.

- describe summarization module for generating output summaries

The summarization module receives as input a source document and a selected control code, which are concatenated and encoded by the transformer encoder. The decoder then generates the summary autoregressively, with each output token conditioned on the encoded document, the control code, and previously generated tokens. During inference, users may select a control code corresponding to the desired level of faithfulness—for example, <FF-high> for maximum factual alignment or <FF-low> for higher fluency at the expense of partial entity coverage. The model generates summaries that are consistent with the selected control code, maintaining fluency and salience while significantly reducing hallucinated entities. The module is capable of stacking multiple control codes, such as combining a faithfulness code with a target domain code, to enable simultaneous control over both factual fidelity and stylistic characteristics. This modular design allows the system to be deployed in diverse settings, from high-precision legal summarization to rapid news aggregation, without retraining or architectural modification.

### Systems for Abstractive Summarization

- introduce system for abstractive summarization

The system for abstractive summarization comprises a pre-processing module, an entity coverage precision module, a pseudo labeling module, a summarization module, and a sequence-to-sequence model, all integrated into a unified pipeline. The system accepts a source document as input and produces a summary that is both fluent and factually grounded, with the degree of faithfulness controlled by a user-specified control code. The system is implemented as a software module executable on general-purpose computing hardware, including cloud-based servers and edge devices, and is designed for integration into content management systems, digital news platforms, and enterprise knowledge bases.

- describe pre-processing module

The pre-processing module performs tokenization, sentence segmentation, and entity normalization on the input document to prepare it for downstream processing. It ensures consistent formatting across diverse document types and removes irrelevant metadata such as headers, footers, or HTML tags. The module also applies domain-specific preprocessing rules, such as standardizing medical terminology in scientific articles or anonymizing personally identifiable information in legal documents, to enhance the robustness and safety of the summarization output.

- describe entity coverage precision module

The entity coverage precision module computes the entity coverage precision score for each document-reference summary pair in the training corpus. It invokes a named entity recognition system tailored to the domain of the input text, extracts all entities from both the document and the summary, and calculates the ratio of overlapping entities to the total entities in the summary. The output of this module is a continuous scalar value that serves as the input to the pseudo labeling module.

- describe pseudo labeling module

The pseudo labeling module receives the entity coverage precision scores from the entity coverage precision module and maps them to discrete control codes by binning the scores into predefined intervals. Each bin is associated with a unique control code token, which is inserted into the model’s vocabulary. The module outputs a training dataset in which each document is prefixed with its corresponding control code, forming the input-output pairs used to train the summarization model.

- describe summarization module

The summarization module implements the transformer-based sequence-to-sequence model that generates summaries conditioned on the input document and control code. It employs a transformer encoder to encode the input sequence and a transformer decoder to generate the summary autoregressively. The module is trained using a cross-entropy loss function and is capable of generating summaries with varying degrees of faithfulness by selecting different control codes during inference.

- describe sequence-to-sequence model

The sequence-to-sequence model is based on the transformer architecture, comprising a stack of encoder and decoder layers with multi-head self-attention and feed-forward networks. The model is initialized with pre-trained weights from BART-large and fine-tuned on domain-specific datasets. The model’s input vocabulary includes both standard tokens and special control code tokens, allowing it to condition its generation on faithfulness signals without altering its core architecture.

- describe BART abstractive summarization model

The BART abstractive summarization model serves as the backbone of the disclosed system. It is a denoising autoencoder pre-trained on large-scale text corpora to reconstruct corrupted input sequences, enabling strong generalization to downstream tasks. The model is fine-tuned on document-summary pairs with control codes, allowing it to learn how to generate summaries that align with specified levels of entity coverage precision.

- describe faithfulness control code

The faithfulness control code is a special token appended to the input sequence that signals the model to generate summaries with a particular level of entity-level faithfulness. These codes are learned during training and are not manually assigned. They encode quantified measures of entity coverage precision and enable the model to modulate its generation behavior accordingly.

- describe entity coverage precision metric

The entity coverage precision metric is a quantitative measure of the proportion of named entities in the generated summary that are explicitly mentioned in the source document. It is computed using named entity recognition and set intersection, providing a direct, interpretable, and scalable method for evaluating and controlling factual alignment in abstractive summaries.

- describe binning procedure for pseudo labels

The binning procedure divides the continuous entity coverage precision values into discrete intervals, each corresponding to a control code. The bin boundaries are selected to ensure uniform distribution of training instances across bins, avoiding bias and enabling balanced learning. The number of bins is configurable, with three bins shown to be optimal in experimental evaluations.

- describe training dataset

The training dataset consists of document-control code-summary triples derived from annotated corpora such as XSum, PubMed, and SAMSum. Each document is prefixed with its corresponding control code, and the model is trained to predict the reference summary given this augmented input.

- describe article-summary pairs

Article-summary pairs are curated from publicly available summarization benchmarks and represent the ground truth for training the model. Each pair includes a source article and a human-written summary, from which entity coverage precision is computed to generate the control code.

- describe entity mentions

Entity mentions are the occurrences of named entities within the source document or summary, identified by the named entity recognition system. These mentions form the basis for calculating entity coverage precision and are preserved in the control code generation process.

- describe precision metric calculation

The precision metric is calculated as the number of entities in the summary that appear in the document divided by the total number of entities in the summary. This calculation is performed using set operations on the outputs of a named entity recognition system.

- describe pseudo label generation

Pseudo label generation involves mapping continuous entity coverage precision scores to discrete control code tokens based on predefined bin boundaries. This process is fully automated and requires no human annotation.

- describe control code generation

Control code generation is the process of assigning a unique special token to each bin of entity coverage precision values. These tokens are added to the model’s vocabulary and prepended to input documents during training and inference.

- describe summarization model training

The summarization model is trained using a cross-entropy loss function over the training dataset of document-control code-summary triples. The model learns to generate summaries that match the reference while respecting the faithfulness signal encoded in the control code.

- describe cross-entropy loss function

The cross-entropy loss function measures the discrepancy between the model’s predicted token distribution and the ground truth summary tokens. It is minimized during training to optimize the model’s ability to generate accurate and faithful summaries.

- describe transformer encoder

The transformer encoder processes the input sequence, which includes the document and the control code, by applying multiple layers of self-attention and feed-forward networks to produce a contextualized representation of the input.

- describe transformer decoder

The transformer decoder generates the summary token-by-token, attending to both the encoder’s output and previously generated tokens. It is conditioned on the control code to modulate the faithfulness of the output.

- describe output summary generation

The output summary is generated autoregressively by the decoder, with each token selected based on the probability distribution over the vocabulary, conditioned on the encoded document and control code.

- describe training objective

The training objective is to maximize the likelihood of the reference summary given the document and control code, using the cross-entropy loss function as the optimization criterion.

- describe domain-specific databases

Domain-specific databases include collections of text such as PubMed for biomedical literature, Wikipedia for general knowledge, and SAMSum for conversational summaries. These databases are used to construct training sets tailored to specific application domains.

- describe Wikipedia corpus

The Wikipedia corpus is used for intermediate pre-training to generate target-specific pseudo document-summary pairs that match the characteristics of downstream datasets. It enables the model to generalize across domains without requiring separate fine-tuning for each.

- describe intermediate pre-training pipeline

The intermediate pre-training pipeline generates pseudo document-summary pairs from Wikipedia articles by matching length, abstractiveness, and structure to target datasets. Each pair is assigned a target control code, enabling the model to learn domain-specific summarization styles.

- describe target datasets

Target datasets are the downstream summarization benchmarks, such as XSum, PubMed, and SAMSum, for which the system is ultimately evaluated and deployed.

- describe abstractiveness level

Abstractiveness level refers to the degree to which a summary rephrases or paraphrases the source document rather than copying directly. It is measured by the ratio of novel n-grams in the summary to those in the source and is used to match Wikipedia articles to target datasets.

- describe training instance construction

Training instance construction involves concatenating the control code with the source document and pairing it with the reference summary to form a training sample. This process is applied uniformly across all datasets.

- describe pseudo label generation

Pseudo label generation is the algorithmic assignment of control codes based on entity coverage precision, enabling the model to learn faithfulness without manual annotation.

- describe training set construction

The training set is constructed by aggregating document-control code-summary triples from multiple domains, ensuring diversity and robustness in model performance.

- describe summarization model training

Summarization model training involves optimizing the transformer model using the cross-entropy loss on the constructed training set, with control codes serving as conditioning signals.

- describe zero-shot summarization

Zero-shot summarization refers to the ability of the model to generate faithful summaries on unseen domains without additional fine-tuning, enabled by the intermediate pre-training pipeline and stacked control codes.

- describe target-specific intermediate data generation

Target-specific intermediate data generation involves creating synthetic document-summary pairs from Wikipedia that mimic the length, structure, and abstractiveness of target datasets, enabling domain adaptation without labeled data.

- describe Wikipedia article processing

Wikipedia article processing involves extracting lead sections, filtering for length and structure, and aligning them with target dataset characteristics to generate pseudo training pairs.

- describe summary generation

Summary generation is the process by which the model produces a condensed version of the input document, conditioned on the control code and guided by learned faithfulness patterns.

- describe article generation

Article generation refers to the creation of synthetic source documents from Wikipedia for intermediate pre-training, matched to target dataset properties.

- describe training instance construction

Training instance construction involves combining the control code, document, and reference summary into a single input-output pair for model training.

- describe pseudo label generation

Pseudo label generation is the automated derivation of control codes from entity coverage precision, enabling scalable and consistent faithfulness conditioning.

- describe training set construction

Training set construction aggregates document-control code-summary triples from multiple domains to form a unified training corpus.

- describe summarization model training

Summarization model training optimizes the transformer architecture to generate summaries that align with control codes, using cross-entropy loss over the training set.

- describe zero-shot summarization

Zero-shot summarization enables the model to generalize to new domains without fine-tuning, leveraging control codes and intermediate pre-training to maintain faithfulness and fluency.

- describe generalizability across domains

The disclosed system demonstrates generalizability across domains by using stacked control codes and intermediate pre-training on Wikipedia, allowing a single model to perform effectively on news, scientific, and conversational summarization tasks without domain-specific retraining.

## EXAMPLES

### Example 1: Experimental Methods and Results

- introduce experimental setup

The experimental setup involved training and evaluating the disclosed system on three benchmark datasets: XSum, PubMed, and SAMSum, representing news, scientific, and conversational domains respectively. The backbone model was BART-large, initialized with pre-trained weights and fine-tuned using the Huggingface Transformers library on eight Tesla A100 GPUs. The entity coverage precision metric was computed using Stanza for general domains and scispaCy for PubMed. Three faithfulness control codes—<FF-low>, <FF-mid>, and <FF-high>—were defined based on bin boundaries calibrated to ensure balanced distribution of training examples.

- describe news dataset

The news dataset consisted of the XSum corpus, containing BBC news articles paired with single-sentence summaries. The dataset was selected for its high abstractiveness and known prevalence of entity hallucination in baseline models.

- describe scientific paper dataset

The scientific paper dataset comprised PubMed articles with abstracts serving as summaries. This dataset was chosen for its domain-specific terminology and the critical need for factual accuracy in biomedical contexts.

- describe dialog dataset

The dialog dataset was the SAMSum corpus, containing real-world chat conversations with human-written summaries. This dataset tested the system’s ability to handle informal language, multiple speakers, and implicit entity references.

- compare with baseline methods

The disclosed system was compared against BART-large without control codes, a post-processing entity correction method, and a data filtering approach. The system outperformed all baselines in entity coverage precision while maintaining or improving ROUGE and BERTSCORE metrics.

- describe evaluation metrics

Evaluation metrics included ROUGE-1, ROUGE-2, and ROUGE-L for fluency and salience; BERTSCORE for semantic similarity; FEQA for factual consistency; and entity coverage precision for faithfulness. Human evaluation was conducted by four expert annotators who rated summaries on faithfulness and quality using a three-point scale.

- introduce Rouge metric

The ROUGE metric measures n-gram overlap between the generated summary and the reference summary, providing a standard evaluation of lexical similarity and coverage.

- introduce BERTSCORE metric

BERTSCORE computes semantic similarity between generated and reference summaries using contextual embeddings from BERT, capturing meaning beyond exact word matches.

- introduce Entity Coverage Precision metric

The Entity Coverage Precision metric quantifies the proportion of named entities in the summary that are present in the source document, serving as the primary indicator of factual fidelity.

- introduce FEQA metric

FEQA evaluates faithfulness by asking a series of fact-based questions about the summary and measuring whether answers can be inferred from the source document.

- describe human evaluation

Human evaluation involved blind rating of 50 randomly sampled summaries from each dataset. Annotators classified summaries as faithful, intrinsically hallucinated, or extrinsically hallucinated, and rated quality on a scale from one to three. Inter-annotator agreement was high, with Cohen’s kappa exceeding 0.75.

- introduce Huggingface libraries

The Huggingface Transformers library was used to implement and fine-tune the BART-large model, providing standardized tokenization, training, and evaluation pipelines.

- describe BART-large model

The BART-large model, with 336 million parameters, served as the base architecture for all experiments. It was fine-tuned with and without control codes to isolate the effect of the disclosed method.

- describe fine-tuning process

Fine-tuning was performed for 10 epochs with a learning rate of 5e-5, weight decay of 0.01, and batch size of 16, using the Adam optimizer. All models were trained under identical conditions to ensure fair comparison.

- introduce entity recognition

Entity recognition was performed using Stanza for general domains and scispaCy for biomedical text, with models trained on the OntoNotes corpus and clinical text corpora respectively.

- describe Stanza NLP toolkit

The Stanza NLP toolkit is a neural pipeline for linguistic analysis that provides accurate named entity recognition, part-of-speech tagging, and dependency parsing for multiple languages.

- describe OntoNotes corpus

The OntoNotes corpus is a large-scale annotated dataset containing text from news, web, and conversational sources, used to train the Stanza NER system.

- describe scispaCy toolkit

The scispaCy toolkit is a specialized NLP library for biomedical text, trained on clinical and scientific corpora to recognize entities such as diseases, genes, and treatments.

- show example article and summary

An example from XSum showed that BART hallucinated “Los Angeles” as the location of an E3 gaming event, while the correction method replaced it with “Mexico,” introducing a factual error. The disclosed system, using <FF-high>, generated a summary that omitted the incorrect location entirely and retained only entities present in the source.

- show performance on downstream datasets

On XSum, the system achieved a 12.7% increase in entity coverage precision over BART without degradation in ROUGE-L. On PubMed, it improved precision by 18.3% with no loss in BERTSCORE. On SAMSum, it reduced extrinsic hallucinations by 22% while maintaining fluency.

- compare with state-of-the-art methods

Compared to state-of-the-art methods such as post-processing correction and data filtering, the disclosed system achieved superior faithfulness improvements without sacrificing summary quality, and outperformed WikiTransfer in zero-shot settings.

- study controllable Wikipedia intermediate pre-training

Intermediate pre-training on Wikipedia with target control codes enabled the model to generalize to unseen domains. When stacked with faithfulness codes, it achieved higher faithfulness and fluency than domain-specific models.

- show zero-shot performance results

In zero-shot settings, the model achieved 89.4% entity coverage precision on PubMed when trained only on XSum and Wikipedia, surpassing WikiTransfer by 6.2 percentage points.

- show human evaluation results

Human evaluators rated the system’s summaries as 22% more faithful than BART and 15% more fluent than the correction baseline, with no significant difference in overall quality.

- analyze entity distribution

Analysis showed that the number of entities generated by the system was statistically indistinguishable from BART, indicating that faithfulness improvements were not due to entity suppression.

- investigate control codes

Control codes were found to be interpretable and stable across domains. The <FF-high> code consistently reduced hallucinations, while <FF-low> allowed for more creative phrasing without introducing factual errors.