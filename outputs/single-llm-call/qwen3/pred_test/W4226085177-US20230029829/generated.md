# DESCRIPTION

## BACKGROUND

- introduce conversational search  
Conversational search represents a paradigm shift in information retrieval, moving beyond static, one-time queries toward dynamic, multi-turn interactions between users and systems. Unlike traditional search engines that rely on keyword matching and document ranking based on isolated queries, conversational search systems engage users through iterative dialogue, adapting responses based on evolving context, intent, and feedback. This approach enables deeper exploration of ambiguous or complex information needs, particularly when users lack precise terminology or when the underlying knowledge domain is multifaceted. In such settings, users often begin with broad or incomplete queries, requiring clarification to narrow scope, disambiguate meaning, or align system understanding with user intent. Conversational search systems are designed to recognize these moments of uncertainty and respond proactively, thereby reducing cognitive load and improving retrieval accuracy. The interaction model is inherently sequential, where each utterance builds upon prior exchanges, allowing the system to refine its understanding incrementally. This paradigm is especially valuable in domains such as healthcare, legal research, technical support, and academic inquiry, where precision and contextual awareness are critical to delivering actionable insights.

- limitations of traditional search  
Traditional search methodologies, rooted in document-centric retrieval models such as TF-IDF or BM25, suffer from fundamental limitations when confronted with ambiguous, incomplete, or context-dependent queries. These systems treat each query as an independent event, disregarding the conversational history that may contain critical clues about user intent. As a result, they frequently return irrelevant or overly broad results, forcing users to refine their queries through repeated iterations—a process that is time-consuming and cognitively taxing. Moreover, traditional models lack the ability to distinguish between surface-level keyword matches and semantically meaningful associations, leading to poor performance on queries that require inferential reasoning or domain-specific nuance. The absence of dialogue history integration also prevents systems from recognizing when a user’s underlying need has shifted, resulting in responses that are technically accurate but contextually inappropriate. Furthermore, these systems cannot proactively identify gaps in user understanding or anticipate follow-up questions, rendering them passive rather than interactive. In high-stakes environments such as clinical diagnostics or regulatory compliance, such limitations can lead to misinformation, delayed decision-making, or missed opportunities for insight.

- describe conversational search paradigm  
The conversational search paradigm is characterized by its reliance on sequential dialogue as the primary interface for information discovery. It treats search not as a single transaction but as an ongoing exchange, where the system actively participates in shaping the inquiry through clarification, elaboration, and redirection. At its core, this paradigm leverages natural language understanding to interpret user utterances within the context of prior exchanges, enabling the system to detect ambiguity, infer latent intent, and generate context-aware responses. A distinguishing feature of this approach is its capacity for mixed-initiative interaction: while users may initiate the dialogue with a query, the system may respond with a clarification question to narrow the scope, thereby guiding the user toward more precise information. This dynamic interplay allows the system to iteratively refine its understanding, aligning retrieved content more closely with the user’s true information need. The paradigm is grounded in the assumption that complex information needs cannot be fully expressed in a single utterance and that meaningful retrieval requires collaborative exploration. To achieve this, systems must maintain state across turns, integrate external knowledge sources, and employ models capable of reasoning about relevance, coherence, and intent over time.

- summarize related art limitations  
Prior approaches to conversational search have primarily focused on either generating clarification questions using neural language models or selecting from a fixed pool of pre-defined questions using heuristic or rule-based methods. While generative models offer flexibility, they often produce syntactically correct but semantically irrelevant or redundant questions, lacking grounding in the underlying document corpus. Conversely, selection-based methods, though more reliable, typically rely on shallow features such as query-question similarity or co-occurrence statistics, ignoring the rich contextual signals provided by retrieved passages. Existing systems frequently treat clarification questions as isolated entities, disconnected from the textual evidence that supports their relevance. This disconnect leads to suboptimal question selection, as the system fails to leverage the semantic alignment between the user’s intent, the retrieved documents, and the potential clarification. Furthermore, most approaches do not jointly model the relationship between conversation history, passage content, and candidate questions, resulting in a fragmented understanding of relevance. The absence of a unified framework that integrates passage-based evidence with conversational context has hindered the development of robust, generalizable clarification selection systems capable of performing effectively across diverse domains and data sources.

## SUMMARY

- introduce computer-implemented method  
A computer-implemented method is disclosed for automatically selecting clarification questions during a conversational search interaction. The method operates by analyzing a sequence of user and system utterances to identify moments of ambiguity and then selecting the most appropriate clarification question from a curated pool of candidates. This selection is informed by deep learning models that evaluate the semantic association between the conversation history and candidate questions, as well as the alignment between those questions and relevant textual passages retrieved from a document corpus. The method enables systems to proactively guide users toward more precise information needs by leveraging contextual and content-based signals, thereby improving retrieval accuracy and reducing user effort.

- receive search conversation  
The method begins by receiving a search conversation comprising a sequence of utterances, each attributed to either a user or an agent. The conversation includes an initial user query followed by zero or more subsequent exchanges, culminating in a point where a clarification question is to be selected. Each utterance is processed as a natural language text string, preserving its temporal order and speaker identity. The system maintains a dynamic representation of the conversation context, incorporating all prior utterances up to the current turn, excluding any clarification question that has yet to be selected.

- retrieve relevant text passages  
The system retrieves a set of relevant text passages from a corpus of documents indexed for content-based retrieval. These passages are extracted using a sliding window technique applied to top-ranked documents identified through an initial retrieval phase based on keyword matching and term weighting derived from the conversation history. Each passage is assigned an initial relevance score based on term coverage, inverse document frequency, and term frequency, adjusted by a document-level score. The resulting set of passages serves as a contextual foundation for evaluating the potential utility of candidate clarification questions.

- retrieve candidate clarification questions  
For each retrieved passage, the system queries a pre-indexed pool of candidate clarification questions by constructing a composite query that combines the passage content with the full conversation history. This query is used to retrieve a ranked list of candidate questions from the clarification index, ensuring that only questions semantically related to both the conversation and the retrieved evidence are considered. The result is a set of candidate clarification questions, each associated with one or more passages, forming a structured candidate space for subsequent ranking.

- rank candidate clarification questions  
The candidate clarification questions are ranked using two distinct deep learning models. The first model evaluates the association between the conversation history and each candidate question, independent of passage content. The second model evaluates the joint association between the conversation history, a specific passage, and the candidate question. Each model outputs a scalar score representing the strength of semantic alignment. These scores are combined using a fusion function to produce a final ranking of candidate questions, prioritizing those that are both conversationally relevant and contextually grounded in retrieved evidence.

- provide highest-ranking candidate question  
The system selects the candidate clarification question with the highest fused score and presents it to the user as the next system utterance in the conversation. This question is designed to resolve ambiguity, refine intent, or narrow the scope of the search in a manner consistent with the retrieved evidence and prior dialogue. The presentation is delivered in natural language, maintaining conversational coherence and user experience.

- introduce system embodiment  
The invention further encompasses a system embodiment comprising one or more computing devices configured to execute the method. The system includes a conversation history module, a document retrieval module, a passage extraction module, a clarification question index, two deep learning models trained to evaluate semantic associations, a scoring and fusion module, and a user interface component. These components operate in concert to receive, process, and respond to conversational inputs in real time.

- describe system components  
The system components are interconnected through a modular architecture that enables scalable deployment across distributed environments. The conversation history module maintains a persistent, time-stamped record of all utterances. The document retrieval module interfaces with a document corpus indexed using standard information retrieval techniques. The passage extraction module applies sliding window segmentation to extract contextually relevant text segments. The clarification question index stores a curated collection of candidate questions, each annotated with metadata for retrieval. The deep learning models are implemented as transformer-based neural networks fine-tuned for sequence classification tasks. The scoring and fusion module computes and combines model outputs using a weighted summation function. The user interface component formats and delivers the selected clarification question to the user via a text-based or voice-based interface.

- introduce computer program product  
The invention further includes a computer program product comprising non-transitory computer-readable storage media encoded with program instructions that, when executed by one or more processors, cause the system to perform the steps of the disclosed method. The program product may be distributed as software, embedded in firmware, or downloaded over a network, and is capable of execution on general-purpose computing devices, cloud servers, or specialized hardware accelerators.

- describe program code execution  
The program code is configured to load the conversation context, invoke the document and clarification question retrieval modules, apply the deep learning models in sequence, fuse their outputs, and select the highest-ranking clarification question. Execution is performed in a deterministic, state-aware manner, ensuring that each step builds upon the output of the prior step. The code is optimized for low-latency inference, enabling real-time interaction without perceptible delay. The program further includes error handling, logging, and monitoring capabilities to ensure operational reliability and diagnostic traceability.

- fuse scores from deep learning models  
The fusion of scores from the two deep learning models is performed using a linear combination function that assigns distinct weights to each model’s output based on empirical performance across training datasets. This fusion strategy capitalizes on the complementary strengths of the models: the first model excels in identifying questions that align with conversational intent, while the second model excels in identifying questions that are substantiated by retrieved textual evidence. The combined score provides a more robust and generalizable ranking than either model alone, reducing the risk of selecting questions that are either too generic or overly dependent on noisy passage content.

- retrieve solution documents  
Solution documents are retrieved from a large-scale document corpus using a disjunctive query over terms extracted from the conversation history. These documents serve as the source material for passage extraction and are ranked using a combination of BM25 similarity and fixed-point term weighting that accounts for utterance structure and sequential emphasis. The top-ranked documents are retained for subsequent processing, ensuring that the system operates on the most relevant content.

- extract candidate text passages  
Candidate text passages are extracted from the retrieved solution documents using a sliding window of fixed size with partial overlap. Each passage is assigned an initial score based on the coverage of conversation terms, weighted by inverse document frequency and scaled term frequency. The final passage score is computed as a linear combination of the initial passage score and the score of its parent document, ensuring that high-quality documents contribute more significantly to the passage ranking.

- retrieve candidate clarification questions  
Candidate clarification questions are retrieved by querying a pre-built index of potential questions using a composite query formed by concatenating the content of a passage with the full conversation history. This ensures that only questions semantically related to both the context and the evidence are considered, filtering out irrelevant or generic alternatives.

- train deep learning models  
The deep learning models are trained using triplet networks on annotated conversational datasets. The first model is trained on triplets consisting of a conversation context, a positive clarification question (i.e., one that was historically selected), and a negative clarification question (randomly sampled from the pool). The second model is trained on triplets that include a passage in addition to the conversation and question, enabling it to learn associations that are grounded in textual evidence. Training is performed using the BERT-base-uncased architecture, fine-tuned with a learning rate of 2e-5 over three epochs, with maximum sequence lengths adjusted to accommodate input constraints.

## DETAILED DESCRIPTION

- introduce computer-implemented method for automatic selection of clarification question  
A computer-implemented method for the automatic selection of clarification questions in a conversational search environment is disclosed, wherein the method operates by integrating conversational context, retrieved textual evidence, and deep learning-based semantic scoring to identify the most informative next question. The method is designed to function in real time, adapting dynamically to the evolving nature of user inquiries and the structure of the underlying document corpus. It is applicable across diverse domains, including customer support, academic research, and technical documentation retrieval, where precision and contextual awareness are paramount.

- describe system and computer program product embodiments  
The invention is embodied in both a system and a computer program product. The system comprises hardware components including processors, memory, network interfaces, and storage devices, coupled with software modules that implement the method’s steps. The computer program product is stored on non-transitory media such as solid-state drives, optical discs, or cloud-based storage, and includes executable instructions that, when loaded and executed, cause the system to perform the method. The program product may be distributed independently or integrated into larger software platforms, such as enterprise search systems or virtual assistant frameworks.

- explain use of deep learning models to rank candidate clarification questions  
The method employs two deep learning models to rank candidate clarification questions, each trained to capture distinct aspects of relevance. The first model learns to associate conversation history with clarification questions without reference to external passages, capturing conversational coherence and intent alignment. The second model learns to associate the conversation history, a specific passage, and a clarification question, capturing the alignment between the question and the evidence supporting it. Together, these models enable the system to distinguish between questions that are merely plausible and those that are both contextually appropriate and substantiated by content.

- describe first model outputting score denoting strength of association between candidate clarification question and search conversation  
The first model, designated as BERT-C-cq, is a transformer-based neural network fine-tuned to compute a scalar score representing the strength of semantic association between a candidate clarification question and the full conversation history. The model receives as input a concatenation of the conversation utterances and the candidate question, separated by special tokens, and outputs a probability score indicating the likelihood that the question is the appropriate next utterance. The model is trained using triplet loss, where positive examples are drawn from historical clarification questions and negative examples are randomly sampled from the pool.

- describe second model outputting score denoting strength of association between candidate clarification question, search conversation, and text passage  
The second model, designated as BERT-C-P-cq, is a transformer-based neural network fine-tuned to compute a scalar score representing the strength of semantic association between a candidate clarification question, the conversation history, and a specific retrieved text passage. The model receives as input a concatenation of the conversation, the passage, and the candidate question, separated by special tokens, and outputs a score reflecting how well the question resolves ambiguity within the context of the passage. This model is trained using triplets that include a passage retrieved for the conversation, ensuring that the model learns to ground its selections in document-based evidence.

- discuss fusion of scores from both models  
The scores generated by the two models are fused using a linear combination function that assigns weights based on empirical performance observed during training. The fusion function is designed to maximize recall at top ranks by leveraging the complementary strengths of the models: the first model captures conversational relevance, while the second captures content grounding. The fused score provides a more reliable ranking than either model alone, reducing false positives and improving the precision of selected clarification questions.

- describe retrieval of candidate clarification questions  
Candidate clarification questions are retrieved by constructing a query that combines the content of a retrieved passage with the conversation history and submitting it to a pre-indexed database of potential clarification questions. This retrieval step ensures that only questions semantically related to both the context and the evidence are considered, filtering out generic or irrelevant alternatives. The retrieval process is implemented using a standard inverted index with BM25 scoring, enhanced by utterance-biased term weighting derived from the conversation structure.

- introduce block diagram of exemplary configuration for training deep learning models  
A block diagram of the training system illustrates the flow of data from annotated conversation datasets through preprocessing, model initialization, training, and evaluation. The diagram includes components for data ingestion, passage extraction, triplet generation, model training using triplet loss, and performance validation. The training system operates offline, producing model weights that are later deployed in the inference system.

- describe training system components  
The training system comprises a dataset ingestion module, a passage extraction module, a triplet generation module, a model training module, and a validation module. The dataset ingestion module loads annotated conversations from structured files. The passage extraction module identifies relevant text segments from associated documents. The triplet generation module constructs positive and negative examples for training. The model training module fine-tunes BERT architectures using triplet loss. The validation module evaluates model performance on held-out datasets using metrics such as Recall@30.

- describe conversational search system components  
The conversational search system comprises a conversation history manager, a document retrieval engine, a passage extractor, a clarification question index, two deep learning models, a score fusion module, and a user interface. These components operate in sequence: the conversation history is processed to retrieve documents, passages are extracted, candidate questions are retrieved, scores are computed and fused, and the highest-ranking question is presented to the user.

- describe clarification-question selection module  
The clarification-question selection module is a software component responsible for orchestrating the entire selection process. It coordinates the retrieval of documents and passages, invokes the deep learning models, fuses their outputs, and selects the top-ranked question. The module is designed for low-latency inference and is optimized for deployment in cloud-based or edge computing environments.

- discuss training system operation  
The training system operates by first loading annotated conversations and their associated documents. It then extracts passages, identifies clarification questions, and generates triplets for training. The system trains the two models independently, using triplet loss functions to optimize for semantic alignment. Model performance is evaluated on validation sets, and the best-performing weights are saved for deployment.

- describe conversational search system operation  
During operation, the conversational search system receives a user query, retrieves top documents, extracts passages, retrieves candidate questions, computes scores using the two models, fuses the scores, and selects the highest-ranking question. All steps are executed in real time, with the system maintaining state across conversational turns to ensure continuity and coherence.

- introduce flowchart of method for training one or two models  
A flowchart illustrates the steps involved in training the two models: loading conversations, extracting passages, labeling clarification questions, generating triplets, initializing BERT models, fine-tuning with triplet loss, validating performance, and saving model weights. The flowchart distinguishes between the training procedures for the first and second models, highlighting the inclusion of passage content in the second model’s input.

- obtain H2H conversations  
Human-to-human (H2H) conversations are obtained from annotated datasets, including both open-domain and task-oriented domains. These conversations include user queries, agent responses, and labeled clarification questions, serving as the ground truth for training.

- obtain text passages relevant to each H2H conversation  
For each H2H conversation, relevant text passages are extracted from associated documents using a sliding window technique. Each passage is aligned with the conversation and labeled with its parent document, forming the basis for evidence-grounded training.

- label clarification questions and answers in H2H conversations  
Clarification questions are manually or algorithmically labeled based on their presence in the conversation and their association with a subsequent document. Only questions that lead to a relevant document are retained as positive examples.

- discuss scoring solution documents based on relevancy to H2H conversation  
Solution documents are scored using BM25 and fixed-point term weighting, adjusted by utterance position to reflect the increasing importance of later utterances in the conversation. Documents with higher scores are prioritized for passage extraction.

- discuss utterance-biased extension for enhanced word-weighting  
An utterance-biased extension is applied to enhance term weighting by assigning greater weight to terms appearing in later utterances, reflecting their increased relevance to the evolving intent. This adjustment improves the accuracy of both document retrieval and passage scoring.

- retrieve top-r text passages for each H2H conversation  
For each H2H conversation, the top-r passages are retrieved based on their combined document and passage scores. These passages form the evidence base for training the second model.

- retrieve candidate clarification questions for each text passage  
For each retrieved passage, a query is constructed by concatenating the passage with the conversation history and used to retrieve candidate clarification questions from the index. The top-k candidates are retained for training and inference.

- discuss creating training sets for first and second models  
Training sets are created by forming triplets for each conversation: a positive clarification question (the one historically selected), a negative question (randomly sampled), and the conversation context. For the second model, each triplet includes a passage. The training sets are balanced to ensure equal representation of positive and negative examples.

- train first model based on first training set  
The first model is trained using the first training set, which consists of conversation-question triplets without passage content. Training is performed using triplet loss with BERT-base-uncased, optimizing for the ability to distinguish relevant from irrelevant clarification questions based solely on conversational context.

- train second model based on second training set  
The second model is trained using the second training set, which includes passage content in each triplet. Training is performed using the same architecture and loss function, but with input sequences that incorporate passage text, enabling the model to learn associations grounded in document evidence.

- discuss training first model using triplet network  
The first model is trained using a triplet network architecture, where each input consists of a conversation context, a positive clarification question, and a negative clarification question. The model is optimized to minimize the distance between the conversation and the positive question while maximizing the distance to the negative question, using a margin-based triplet loss function.

- discuss training second model using triplet network  
The second model is similarly trained using a triplet network, but each input includes a passage concatenated with the conversation context. This enables the model to learn how clarification questions relate to the content of retrieved documents, improving the grounding of selected questions in evidence.

- introduce method for selecting suitable clarification question during search conversation  
A method for selecting a suitable clarification question during a live search conversation is disclosed, comprising the steps of receiving the conversation history, retrieving relevant documents, extracting passages, retrieving candidate questions, scoring them using two deep learning models, fusing the scores, and selecting the highest-ranked question for presentation.

- receive search conversation  
The method receives a search conversation as a sequence of utterances, each tagged with speaker identity and timestamp. The conversation is processed in real time, with the system maintaining a dynamic context window of the most recent utterances.

- retrieve text passages relevant to search conversation  
Text passages are retrieved from a document corpus using a disjunctive query over conversation terms, followed by sliding window extraction and scoring based on term coverage and document relevance.

- retrieve top-m solution documents  
The top-m solution documents are retrieved using BM25 scoring, with term weights adjusted by utterance position to reflect their increasing relevance in the conversation.

- extract candidate text passages from solution documents  
Candidate text passages are extracted from the top-m documents using overlapping sliding windows of fixed size. Each passage is scored based on term coverage and document relevance, with a linear combination used to compute the final score.

- assign initial score to each candidate text passage  
Each candidate passage is assigned an initial score based on the sum of weighted term frequencies and inverse document frequencies of terms appearing in the conversation, with additional weighting applied to terms in later utterances.

- calculate final text passage score  
The final passage score is computed as a linear combination of the initial passage score and the score of its parent document, with a fixed weight of 0.5 assigned to each component.

- select top-r text passages  
The top-r passages are selected based on their final scores, ensuring that only the most relevant evidence is used in subsequent steps.

- retrieve candidate clarification questions for each text passage  
For each selected passage, a query is constructed by concatenating the passage with the conversation history and used to retrieve candidate clarification questions from the index.

- rank candidate clarification questions using one or both trained models  
Candidate clarification questions are ranked using the first trained model, the second trained model, or both. Each model outputs a score indicating the strength of association, and the scores are combined using a fusion function.

- discuss ranking by first trained model  
The first trained model ranks each candidate question based on its alignment with the conversation history, independent of passage content. Higher scores indicate greater conversational relevance.

- discuss ranking by second trained model  
The second trained model ranks each candidate question based on its alignment with both the conversation history and the associated passage. Higher scores indicate greater evidential grounding.

- fuse scores from both models  
The scores from the two models are fused using a weighted sum, with weights determined empirically during training. The fused score provides a comprehensive measure of both conversational relevance and content grounding.

- select top-ranking candidate clarification question  
The candidate clarification question with the highest fused score is selected as the next system utterance.

- provide top-ranking candidate clarification question to user  
The selected clarification question is presented to the user in natural language, maintaining conversational flow and coherence.

- discuss optional presentation of multiple top-ranking candidate clarification questions  
In optional embodiments, the system may present multiple top-ranking clarification questions to the user, allowing for user-driven selection or providing alternative interpretations of intent.

- describe conversational search system operation  
The conversational search system operates continuously, receiving user inputs, retrieving and processing documents, extracting passages, retrieving and scoring candidate questions, and delivering the highest-ranked question as the next system utterance.

- describe clarification-question selection module operation  
The clarification-question selection module executes the full selection pipeline, coordinating retrieval, scoring, fusion, and presentation. It is designed for low-latency, high-throughput operation and is capable of scaling to support thousands of concurrent conversations.

- discuss training system implementation  
The training system is implemented as a batch-processing pipeline running on distributed computing clusters, with data stored in structured formats and models trained using GPU-accelerated frameworks.

- discuss conversational search system implementation  
The conversational search system is implemented as a microservice architecture, with each component deployed independently and communicating via API endpoints. The system is containerized for scalability and deployed on cloud infrastructure.

- discuss clarification-question selection module implementation  
The clarification-question selection module is implemented as a stateful service that maintains conversation context across turns, invokes retrieval and scoring components, and returns the selected question. It is optimized for real-time inference with sub-second response times.

- discuss hardware and software components  
The system comprises standard computing hardware including CPUs, GPUs, memory, and storage, coupled with software components including operating systems, libraries, and application code. The software stack includes Python, PyTorch, Hugging Face Transformers, Apache Lucene, and custom modules for passage extraction and fusion.

- discuss operating system and software components  
The operating system is a Linux-based distribution, with software components including a web server, message queue, database, and model inference engine. All components are containerized using Docker and orchestrated using Kubernetes.

- discuss additional components and modules  
Additional components include logging, monitoring, and analytics modules that track system performance, user satisfaction, and model drift. These modules support continuous improvement and adaptive retraining.

- conclude description of patent application  
The disclosed invention provides a novel, robust, and scalable method for selecting clarification questions in conversational search systems by integrating deep learning models with evidence-based reasoning. The method significantly improves upon prior art by grounding question selection in both conversational context and retrieved textual evidence, enabling more accurate, context-aware, and user-aligned interactions.

### Experimental Results

- evaluate method 200 and method 300  
Method 200, which employs only the first model (BERT-C-cq), and Method 300, which employs the fused model (BERT-fusion), were evaluated on two datasets: ClariQ and Support. Method 300 consistently outperformed Method 200, demonstrating the value of incorporating passage-based evidence into the selection process.

- introduce ClariQ dataset  
The ClariQ dataset is a publicly available collection of open-domain conversational search interactions, each consisting of a user query, a clarification question, and a user response. The dataset is annotated with high-quality clarification questions selected by human annotators and is designed to evaluate systems on their ability to select contextually appropriate follow-up questions.

- introduce Support dataset  
The Support dataset consists of internal customer support conversations between users and technical agents, annotated with clarification questions identified using rule-based filtering and document alignment. The dataset is characterized by noisy, informal language and reflects real-world usage patterns.

- describe differences between datasets  
The ClariQ dataset is clean, structured, and domain-general, while the Support dataset is noisy, domain-specific, and contains conversational artifacts such as pleasantries and disfluencies. The differences between the datasets demonstrate the method’s generalizability across varied environments.

- describe experimental setup  
Experiments were conducted using Apache Lucene for document indexing, BERT-base-uncased for model training, and PyTorch for implementation. The sliding window size was set to 512 characters, and hyperparameters were tuned using development sets. Evaluation used Recall@30 as the primary metric.

- represent documents using two fields  
Documents were represented using two fields: text (the document body) and anchor (associated conversation snippets). For ClariQ, only the text field was used; for Support, both fields were combined.

- use sliding window for text passage retrieval  
Passages were extracted using a sliding window of 512 characters with 50% overlap, ensuring broad coverage of contextual information while maintaining manageable input sizes for the BERT models.

- set hyperparameters  
Hyperparameters included a learning rate of 2e-5, three training epochs, a maximum sequence length of 256 for the first model and 384 for the second, and a batch size of 16. All models were trained on NVIDIA V100 GPUs.

- use PyTorch Hugging Face implementation of BERT  
The BERT models were implemented using the Hugging Face Transformers library, with bert-base-uncased as the base architecture, fine-tuned on the training data for both models.

- fine-tune BERT models  
Both models were fine-tuned using triplet loss, with positive and negative samples generated according to the training protocol. Training converged within three epochs, with validation performance stabilizing after two.

- retrieve initial candidate clarifications  
Initial candidate clarification questions were retrieved using BM25 over the concatenated passage-conversation query, with a maximum of 1,000 candidates per passage.

- report results of development sets  
On the ClariQ development set, BERT-fusion achieved Recall@30 of 0.791, outperforming BERT-C-cq (0.770) and BERT-C-P-cq (0.768). On the Support development set, BERT-fusion achieved 0.552, outperforming BERT-C-cq (0.538) and BERT-C-P-cq (0.535).

- compare BERT rankers to IR-Base  
Both BERT models significantly outperformed IR-Base, which used only BM25 for ranking clarification questions. On Support, BERT-C-cq improved Recall@30 by 82% over IR-Base.

- observe similar results from BERT models  
The two BERT models performed similarly on both datasets, indicating that neither model alone dominates the other. Their complementary nature justifies the fusion strategy.

- fuse scores for further improvement  
Fusing the scores of the two models resulted in a consistent 2–3% improvement in Recall@30, demonstrating that the models capture distinct aspects of relevance.

- report official ClariQ results on test set  
On the official ClariQ test set, the BERT-fusion system ranked fourth overall and second among independent teams, demonstrating competitive performance without exploiting dataset-specific biases.

- describe optional embodiments of the invention  
Optional embodiments include systems that present multiple clarification questions, systems that retrain models periodically using new user interactions, and systems that integrate feedback loops to refine question selection over time.

- define system, method, and computer program product  
The invention is defined as a system, a method, and a computer program product, each comprising the components and steps described herein. The system performs the method, and the computer program product causes the system to perform the method when executed.

- describe computer readable storage medium  
The computer readable storage medium is a non-transitory physical medium, such as a hard disk drive, solid-state drive, optical disc, or flash memory, encoded with program instructions that, when executed, cause a computing device to perform the method.

- list examples of computer readable storage medium  
Examples include magnetic disks, optical discs, solid-state drives, USB flash drives, and cloud-based storage volumes accessible via network protocols.

- describe computer readable program instructions  
The computer readable program instructions are encoded in a programming language such as Python or C++, compiled or interpreted, and structured to implement the steps of the method when loaded into memory and executed by a processor.

- download instructions from network  
The program instructions may be downloaded from a remote server over a network, such as the Internet, and stored locally for execution.

- execute instructions on computing device  
The instructions are executed on a computing device comprising one or more processors, memory, and input/output interfaces, enabling real-time operation in production environments.

- describe flowchart and block diagram illustrations  
Flowcharts and block diagrams are provided to illustrate the method and system architecture, showing the sequence of operations and component interactions. These illustrations are integral to the disclosure and support the claims.

- implement functions using computer readable program instructions  
All functions described herein, including retrieval, scoring, fusion, and selection, are implemented using computer readable program instructions stored on non-transitory media.

- describe scope of numerical values  
Numerical values such as window sizes, learning rates, and recall thresholds are provided as illustrative examples and are not limiting. The invention encompasses any values that achieve comparable performance within the described framework.

- clarify terminology and inconsistencies  
Terminology such as “utterance,” “passage,” and “clarification question” is used consistently throughout. Any inconsistencies in prior art references are explicitly avoided in this disclosure to ensure clarity and legal precision.