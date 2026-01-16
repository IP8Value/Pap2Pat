Here is the complete patent application following the provided outline and incorporating the research paper's content:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates generally to the field of natural language processing (NLP) and abstractive summarization. More specifically, the invention pertains to systems and methods for improving the faithfulness of abstractive summarization outputs by reducing entity-level hallucinations through entity coverage control mechanisms. The disclosed techniques utilize control codes to guide sequence-to-sequence (seq2seq) models during both training and inference phases, enabling the generation of summaries that maintain high factual accuracy without compromising fluency or salience. The invention further extends to zero-shot summarization scenarios through controllable intermediate fine-tuning across multiple domains.  

## BACKGROUND  

Abstractive summarization systems aim to generate concise, fluent summaries that capture the most salient information from source documents. While recent advances in pre-trained language models have improved summary quality, a persistent limitation remains the generation of unfaithful content—particularly through entity hallucinations where the model introduces entities not present in the source material.  

Existing approaches to address hallucination suffer from significant drawbacks:  
- Post-processing correction methods (e.g., replacing hallucinated entities with source document entities) introduce additional errors and increase intrinsic hallucinations.  
- Data filtering and multi-task learning techniques sacrifice training data quantity, degrading summary quality.  
- Domain-specific fine-tuning lacks generalizability, requiring separate models for different applications.  

Prior art fails to provide a unified solution that maintains summary quality while systematically reducing hallucinations across diverse domains. There exists a pressing need for a method that:  
1) Quantitatively measures and controls entity faithfulness during model training  
2) Preserves fluency and salience while minimizing extrinsic hallucinations  
3) Generalizes across domains without requiring target-specific architecture modifications  

## DETAILED DESCRIPTION  

The present invention introduces an Entity Coverage Control (ECC) framework that addresses the above limitations through three key innovations:  

1) **Quantized Faithfulness Guidance**  
For each document-summary pair (d_i, s_i) in the training set D, the system computes entity coverage precision (prec_en) as:  

prec_en = |N(s_i) ∩ N(d_i)| / |N(s_i)|  

where N(t) represents the set of named entities in text t. The prec_en values are quantized into k discrete bins (e.g., k=3 for <FF-low>, <FF-mid>, <FF-high>), with bin boundaries dynamically adjusted to maintain balanced training distribution.  

2) **Control-Conditioned Generation**  
Special control tokens {C_1,...,C_k} are added to the model vocabulary. During training, document d_i is prepended with its corresponding control code C_i, enabling the model to learn faithfulness patterns as:  

p_θ(h_i|d_i, C_i)  

During inference, high-faithfulness code C_k is prepended to all inputs to generate maximally faithful summaries.  

3) **Stackable Domain Adaptation**  
For zero-shot generalization, the invention introduces target control codes {E_1,...,E_l} representing different domains/styles. These stack with faithfulness controls for joint conditioning:  

p_θ(h_i|d_i, E_j, C_k)  

This allows single-model deployment across multiple domains while maintaining faithfulness.  

### Systems for Abstractive Summarization  

Figure 1 illustrates the complete ECC system architecture comprising:  

- **Entity Recognition Module**: Utilizes domain-appropriate NER systems (e.g., Stanza for general text, scispaCy for biomedical) to extract named entities.  
- **Control Code Generator**: Computes prec_en, performs quantization, and assigns control codes.  
- **Controlled Seq2Seq Model**: BART-large architecture modified to process prepended control codes.  
- **Multi-Domain Interface**: Enables dynamic control code stacking for zero-shot applications.  

The system trains on Wikipedia-derived pseudo data matching target domain characteristics (length, abstractiveness) before fine-tuning on specific summarization tasks. During deployment, users may:  
1) Select desired faithfulness level via control code  
2) Optionally specify domain/style code  
3) Receive faithful summaries without entity hallucinations  

## EXAMPLES  

### Example 1: Experimental Methods and Results  

**Implementation**  
The system was implemented using HuggingFace libraries with BART-large (336M parameters) as backbone. Training used 8×A100 GPUs with Adam optimizer (lr=5e-5, weight decay). Entity recognition employed:  
- Stanza NER for general domains (XSum, SAMSum)  
- scispaCy for biomedical texts (PubMed)  

**Benchmark Results**  
Table 2 shows ECC's performance versus baselines:  

| Metric       | BART   | ECC (Ours) |  
|--------------|--------|------------|  
| Entity Prec  | 0.68   | 0.92       |  
| ROUGE-L      | 38.2   | 38.0       |  
| FEQA         | 0.71   | 0.89       |  

Key findings:  
- 35% relative improvement in entity precision  
- Comparable ROUGE scores (Δ < 0.5%)  
- 25% higher factual consistency per FEQA  

**Human Evaluation**  
Four experts evaluated 50 XSum samples (Table 5):  

| Model        | Faithful % | Quality (1-3) |  
|--------------|------------|---------------|  
| BART         | 62%        | 2.1           |  
| ECC          | 88%        | 2.2           |  

The system reduced extrinsic hallucinations from 23% to 6% while maintaining summary quality.  

**Zero-Shot Generalization**  
Table 4 shows cross-domain performance:  

| Model        | Entity Prec | ROUGE-L |  
|--------------|------------|---------|  
| WikiTransfer | 0.85       | 32.1    |  
| ECC          | 0.91       | 34.7    |  

ECC achieved superior faithfulness and salience using a single unified model versus domain-specific baselines.  

**Representative Cases**  
Example outputs (Table 8):  

Source: "Saints captain Anderson scored..."  
- BART: "Gary Anderson led Saints..." (hallucinated)  
- ECC: "The captain scored..." (faithful)  

The system avoided common hallucination patterns while preserving core meaning, demonstrating practical utility for real-world applications requiring high factual accuracy.  

This concludes the detailed description of the invention. The claims follow hereafter.