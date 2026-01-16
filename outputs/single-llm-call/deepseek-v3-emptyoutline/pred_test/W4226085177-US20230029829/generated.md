Here is the complete patent application following the provided outline and incorporating the research paper's invention:

# DESCRIPTION  

## BACKGROUND  

The field of information retrieval has traditionally relied on a one-shot query-response paradigm, wherein a user submits a single query and receives a static set of results. This approach fails to account for the inherent ambiguity and evolving nature of human information needs, particularly when dealing with complex queries or exploratory search scenarios. Conventional search systems lack mechanisms to dynamically clarify user intent during an interaction, often returning suboptimal results due to misinterpretation of the original query.  

Prior attempts to address this limitation have focused on either pre-defined clarification question pools or generative models. The former approach suffers from rigidity and limited coverage, while the latter introduces uncontrolled variability and potential irrelevance. Neither method effectively leverages the contextual relationship between user queries, intermediate conversational turns, and the underlying document corpus that ultimately contains the sought-after information.  

Existing systems also fail to properly integrate passage-level content analysis when formulating clarification questions. This represents a significant oversight, as the informational passages most relevant to a user's query often contain crucial indicators for appropriate clarification strategies. The current state of technology therefore lacks a robust, corpus-grounded method for selecting optimal clarification questions during conversational search interactions.  

## SUMMARY  

The present invention discloses a novel system and method for clarification question selection in conversational search environments. At its core, the invention utilizes a dual-index architecture combined with two specialized BERT-based neural models to achieve superior clarification question selection performance.  

The system operates through several key innovations: First, it maintains separate indices for documents and clarification questions, allowing parallel retrieval operations. Second, it implements a sophisticated passage retrieval mechanism that analyzes conversation context through utterance-biased term weighting and sliding window passage extraction. Third, the invention employs two distinct BERT models - one focused on conversation context to clarification question mapping (BERT-C-cq), and another that incorporates retrieved passage content (BERT-C-P-cq). These models are trained using triplet networks that learn fine-grained associations between contexts, passages, and appropriate clarification questions.  

During operation, the system first retrieves relevant passages from the document corpus based on the ongoing conversation context. These passages then inform the retrieval of candidate clarification questions from a dedicated question pool. The two BERT models independently re-rank these candidates, with their scores combined through Comb-SUM fusion to produce the final ranking. This architecture enables the system to leverage both direct context-question relationships and latent patterns revealed through passage content analysis.  

Experimental results demonstrate significant improvements over baseline methods, with the fused approach showing particular effectiveness. The system has been validated across both open-domain information seeking scenarios and task-oriented customer support environments, proving its versatility. Notably, the invention achieves these advances while maintaining computational efficiency, with re-ranking operations completing in 1-2 seconds per conversation on standard GPU hardware.  

## DETAILED DESCRIPTION  

The present invention provides a comprehensive solution for clarification question selection in conversational search systems. The detailed operation of the invention can be understood through examination of its core components and their interactions.  

The system architecture comprises two primary indices: a Documents Index containing the searchable corpus and a Clarification-questions Index housing the pool of potential clarification questions. Both indices utilize advanced information retrieval techniques, including BM25 similarity scoring and specialized field weighting. The Documents Index particularly benefits from an innovative anchor-text enhancement approach, where document representations are augmented with text from associated training dialogues.  

Passage retrieval forms a critical component of the invention's operation. Given a conversation context Cj (comprising the initial query and subsequent utterances), the system first retrieves top-k documents using an utterance-biased extension of the Fixed-Point method for verbose query processing. This approach applies distinct weighting to individual utterances within the conversation, recognizing their varying importance. From these documents, candidate passages are extracted using a sliding window technique with configurable size and overlap parameters.  

Each candidate passage receives a composite score combining: (1) term coverage metrics assessing alignment between passage content and conversation context, utilizing both inverse document frequency (idf) and scaled term frequency (tf) measures; and (2) the retrieval score of its source document. The term coverage calculation employs multiple scoring variants, including BM25-based term weighting and minimum tf scaling, to capture different aspects of relevance.  

The clarification question retrieval phase leverages these passages to identify appropriate candidate questions. For each passage, the system constructs a composite query combining passage content with the conversation context. This query retrieves potential clarification questions from the dedicated index, resulting in passage-specific question rankings.  

The invention's novel re-ranking stage employs two specially-trained BERT models. The first model (BERT-C-cq) focuses on direct relationships between conversation contexts and clarification questions. It is trained using triplet networks that contrast positive examples (actual clarification questions from training data) with randomly-selected negative examples. The second model (BERT-C-P-cq) extends this approach by incorporating passage content, using a [SEP] token to demarcate conversation context from passage text in its input format.  

Both models operate within BERT's token limit constraints through intelligent context truncation - preserving the most recent m utterances that fit within the character limit without mid-utterance truncation. During inference, each candidate clarification question is evaluated by both models, with BERT-C-P-cq processing it in conjunction with its associated passage. The models' scores are combined using Comb-SUM fusion, with optional weighting to emphasize one component over another based on application needs.  

### Experimental Results  

The invention's effectiveness has been rigorously validated through comprehensive testing on two distinct datasets: the open-domain ClariQ benchmark and an internal customer support dataset (Support).  

On the ClariQ development set, the fused BERT approach achieved a Recall@30 score of 0.791, representing a 12% improvement over the initial retrieval baseline (IR-Base) and a 2.5% enhancement over either individual BERT model. The customer support environment showed even more dramatic gains, with the fused system reaching R@30=0.553 compared to the baseline's 0.294 - an 87% relative improvement.  

Analysis of individual cases revealed the complementary strengths of the two BERT models. For the query "all men are created equal," BERT-C-P-cq successfully identified nine relevant clarification questions from the passage content, while BERT-C-cq found only three. This demonstrates the value of passage-informed questioning, particularly for ambiguous queries where multiple interpretations exist.  

The system's computational performance proved practical for real-world deployment. On a 32GB V100 GPU, processing 1000 candidate clarification questions required just 1-2 seconds per conversation. This efficiency stems from careful optimization of the BERT models' sequence lengths (256 tokens for BERT-C-cq and 384 for BERT-C-P-cq) and batch processing of candidates.  

Notably, the invention showed robust performance across both structured (ClariQ) and noisy (Support) conversational environments. In the latter case, a sophisticated question detection pipeline successfully filtered out chit-chat and irrelevant questions by requiring that candidate questions retrieve passages from the conversation's answer document in their top-3 results. This demonstrates the system's adaptability to diverse application scenarios while maintaining high precision in clarification question selection.