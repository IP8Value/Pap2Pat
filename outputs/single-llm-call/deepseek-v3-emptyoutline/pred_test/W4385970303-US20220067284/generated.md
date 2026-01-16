Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION  

## CROSS REFERENCE(S)  

This application claims priority to U.S. Provisional Patent Application No. [Application Number], filed [Filing Date], which is hereby incorporated by reference in its entirety.  

## TECHNICAL FIELD  

The present invention relates generally to natural language processing and text summarization. More specifically, the invention relates to systems and methods for generating controllable abstractive summaries of input documents through keyword-based conditioning.  

## BACKGROUND  

Traditional neural summarization systems operate by compressing source documents into generic summaries without user control over content selection. These systems fall into two categories: extractive summarization that copies important document portions, and abstractive summarization that generates novel sentences. While effective for generic summarization, such systems cannot adapt output to specific user needs or preferences.  

Previous attempts at controllable summarization have required predefined control aspects (e.g., entity focus, length) with corresponding training annotations, necessitating separate models for each control dimension. This approach lacks flexibility and cannot generalize to new control requirements at inference time. Keyword-guided summarization methods have been explored for quality improvement in unconstrained settings, but have not been generalized as a framework for multi-aspect control.  

There exists an unmet need for a unified summarization framework that can: 1) accept diverse control signals through a standardized interface, 2) maintain a single model architecture adaptable to multiple control dimensions, and 3) generalize to novel control aspects without retraining. The present invention addresses these limitations through a novel keyword-based conditioning approach.  

## DETAILED DESCRIPTION  

### Controllable Summarization Overview  

The invention provides CTRLSUM, a framework for generating controlled abstractive summaries through keyword conditioning. At training time, the model learns to predict summaries conditioned on both source documents and automatically extracted keywords. During inference, a control function maps user preferences to keywords that steer summary generation.  

Key advantages include:  
1) Clean separation between training procedure and control aspects - the same model supports multiple control dimensions through different control function implementations  
2) Flexible control interface - users can specify entities, length, or other attributes without model retraining  
3) Combination with prompt engineering - keywords can be paired with decoder prompts for specialized control tasks  

The system architecture comprises:  
1) A sequence-to-sequence model (e.g., BART) as the base summarization engine  
2) A keyword extraction module for training data processing  
3) A configurable control function for inference-time user guidance  

### Computer Environment  

The system operates on standard computing hardware with:  
- One or more processors (CPU/GPU/TPU)  
- Memory (RAM, storage)  
- Network interfaces for model serving  

Software components include:  
- Pretrained language models (e.g., BART-large)  
- Keyword extraction models (e.g., BERT-based taggers)  
- Inference servers with API endpoints  

Input documents may be provided through:  
- File uploads  
- Database queries  
- Real-time text streams  

Output summaries are delivered via:  
- REST APIs  
- User interfaces  
- Integrated applications  

### Controllable Summarization Work Flows  

**Training Phase:**  
1) For each document-summary pair, extract keywords by:  
   a) Selecting sentences maximizing ROUGE with reference  
   b) Identifying longest sub-sequences matching the summary  
   c) Removing duplicates and stop words  

2) Apply keyword dropout to prevent over-reliance on keywords  

3) Train model to maximize p(y|x,z) where:  
   - x = source document  
   - z = keyword sequence  
   - y = target summary  

**Inference Phase:**  
1) User provides control signal c (e.g., entity, length value)  

2) Control function g(x,c) generates keywords by:  
   - Direct output for entity control  
   - Length-based extraction for length control  
   - Task-specific prompts for specialized controls  

3) Model generates summary conditioned on [keywords] + [document]  

**Control Aspects:**  
1) Entity-centric: Focus summaries on specified entities  
2) Length-controlled: Generate summaries of specified length  
3) Contribution-focused: Highlight paper contributions  
4) Purpose-focused: Summarize invention purposes  
5) Question-answering: Generate answers to queries  

### Example Performance  

Quantitative evaluations demonstrate:  

**Entity Control:**  
- 95% success rate including specified entities  
- 89% factual consistency for important entities  

**Length Control:**  
- 0.92 Pearson correlation between requested/actual length  
- 15% reduction in length deviation vs baselines  

**Specialized Tasks:**  
- 43.97 ROUGE-1 for contribution summarization  
- 32.43 ROUGE-1 for purpose summarization  

**Unconstrained Setting:**  
- Outperforms BART by 1.2 ROUGE-2 points on news articles  
- Matches state-of-the-art on scientific papers  

Human evaluations show:  
- 4.3/5 control accuracy for entity-focused summaries  
- 4.1/5 relevance for purpose-focused summaries  

The system has been validated on multiple domains including news articles (CNN/DailyMail), scientific papers (arXiv), and patents (BIGPATENT). The same model architecture achieves strong performance across all domains when adapted through control function configuration.