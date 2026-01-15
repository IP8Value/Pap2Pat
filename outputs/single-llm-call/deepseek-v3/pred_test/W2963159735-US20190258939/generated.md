Here is the patent application following the provided outline and research paper:

# DESCRIPTION  

## PRIORITY APPLICATION DATA  

The present application claims priority to U.S. Provisional Patent Application No. [APPLICATION NUMBER], filed on [DATE], entitled "[TITLE OF PRIOR APPLICATION]", the entirety of which is incorporated herein by reference.  

## TECHNICAL FIELD  

The present invention relates generally to the field of natural language processing and machine learning. More specifically, the invention pertains to systems and methods for scalable and robust textual question answering (QA) using dynamic sentence selection.  

## BACKGROUND  

Textual question answering (QA) represents a critical challenge in natural language processing, wherein a machine must comprehend a given document and generate an accurate response to a posed question. Recent advancements in QA models have been driven by the availability of diverse datasets and sophisticated neural architectures, particularly those employing coattention or bidirectional attention mechanisms to establish codependent representations of documents and questions.  

Despite these advancements, conventional QA models suffer from significant limitations. First, processing lengthy documents or multiple documents requires learning extensive contextual dependencies, leading to computational inefficiency and scalability challenges. Second, existing models exhibit vulnerability to adversarial inputs, often focusing on irrelevant portions of the context and producing incorrect answers. These shortcomings underscore the need for a QA system that balances scalability with robustness.  

## DETAILED DESCRIPTION  

The present invention introduces a novel QA system designed to overcome the limitations of conventional approaches by dynamically selecting a minimal set of sentences necessary to answer a given question. The system comprises two primary components: a sentence selector and a QA module, which operate in tandem to optimize efficiency and accuracy.  

### Introduction to Textual Question Answering  
Textual question answering involves processing a document and generating an answer to a question based on the document's content. The invention improves upon existing methods by reducing the computational burden associated with processing entire documents while maintaining high accuracy.  

### QA System Scalability  
The system achieves scalability by dynamically selecting only those sentences relevant to answering the question, rather than processing the entire document. Empirical analysis demonstrates that most questions can be answered using one or two sentences, significantly reducing processing time and resource consumption.  

### QA System Robustness  
The system enhances robustness by minimizing exposure to adversarial sentences. By filtering out irrelevant or misleading content, the system reduces the likelihood of incorrect answers resulting from adversarial inputs.  

### Introduction to Sentence Selector  
The sentence selector is a critical component that evaluates and scores each sentence in a document relative to the posed question. The selector operates in parallel, assigning a score to each sentence indicating its relevance to the question.  

### Sentence Selector Functionality  
The sentence selector employs an encoder-decoder architecture. The encoder, shared with the QA module, generates sentence encodings and question encodings. The decoder computes bilinear similarities between these encodings to produce a score for each sentence.  

### Introduction to QA Module  
The QA module processes the selected sentences to generate an answer. The module leverages state-of-the-art neural architectures, such as bidirectional attention mechanisms, to determine the most probable answer span within the selected sentences.  

### QA Module Functionality  
The QA module receives the question and the selected sentences, encodes them, and computes an answer span by identifying start and end positions within the sentences. The module's efficiency is enhanced by its ability to operate on a reduced set of sentences.  

### Computing Device Architecture  
The system is implemented on a computing device comprising a processor, memory, and storage. The processor executes instructions stored in memory to perform sentence selection and QA tasks, while the storage retains trained model parameters and datasets.  

### Processor Functionality  
The processor performs parallel computations to score sentences and generate answers. Its architecture is optimized for high-throughput processing of natural language data.  

### Memory Functionality  
The memory stores intermediate representations, such as sentence embeddings and question encodings, during processing. It also caches frequently accessed data to expedite computations.  

### QA System Components  
The system includes an encoder, decoder, and scoring modules. The encoder transforms input text into numerical representations, while the decoder generates scores and answers.  

### Introduction to Encoder  
The encoder processes sentences and questions into fixed-length vector representations. It employs recurrent neural networks (RNNs) or transformers to capture sequential dependencies.  

### Encoder Functionality  
The encoder computes sentence embeddings and question-aware sentence embeddings by applying learned transformations to input text. These embeddings are subsequently used by the decoder.  

### Introduction to Decoder  
The decoder generates scores for sentences and determines answer spans. It utilizes bilinear attention mechanisms to compare sentence and question representations.  

### Decoder Functionality  
The decoder computes similarity scores between sentence encodings and question encodings. These scores guide the selection of relevant sentences and the identification of answer spans.  

### Sentence Embeddings  
Sentence embeddings are dense vector representations that capture semantic and syntactic features of sentences. They are generated by the encoder and used for scoring.  

### Question Embeddings  
Question embeddings represent the posed question in a vector space, enabling comparison with sentence embeddings. They are derived using the same encoder as sentence embeddings.  

### Question-Aware Sentence Embeddings  
These embeddings combine sentence and question information, allowing the model to assess the relevance of a sentence to the question. They are computed using attention mechanisms.  

### Sentence Encodings  
Sentence encodings are final representations produced by the encoder, incorporating contextual information from the entire sentence.  

### Question Encodings  
Question encodings are comprehensive representations of the question, used to compute relevance scores with sentences.  

### Score Computation  
Scores are computed using bilinear functions that measure the alignment between sentence encodings and question encodings. Higher scores indicate greater relevance.  

### Introduction to Normalizer  
The normalizer adjusts scores across sentences within the same paragraph to ensure consistent scaling. It mitigates biases introduced by varying sentence lengths or structures.  

### Normalizer Functionality  
The normalizer applies softmax or other normalization techniques to scores, ensuring fair comparison between sentences.  

### Introduction to Sentence Score Module  
The sentence score module aggregates normalized scores and determines which sentences to select. It operates dynamically, selecting a variable number of sentences per question.  

### Sentence Score Module Functionality  
The module applies a configurable threshold to normalized scores, retaining only those sentences exceeding the threshold. This approach adapts to the complexity of each question.  

### Set of Sentences Selection  
The selected sentences form a minimal set sufficient to answer the question. The size of this set varies depending on the question's requirements.  

### Introduction to Hyper-Parameter  
A hyper-parameter controls the threshold for sentence selection. It is tuned during training to optimize performance.  

### Hyper-Parameter Functionality  
The hyper-parameter balances precision and recall in sentence selection, ensuring high-quality inputs for the QA module.  

### Introduction to Configurable Threshold  
The threshold is dynamically adjusted based on the question's complexity, allowing the system to select more sentences for challenging questions.  

### Configurable Threshold Functionality  
The threshold is optimized to minimize the number of selected sentences while maximizing answer accuracy.  

### QA Module Architecture  
The QA module shares its encoder with the sentence selector, reducing redundancy and improving efficiency.  

### Encoder Sharing  
Shared weights between the selector and QA module ensure consistency in text representations and reduce training overhead.  

### Document Embeddings  
For multi-document QA tasks, the system generates document-level embeddings to prioritize relevant documents before sentence selection.  

### Question Embeddings  
Question embeddings are reused across documents to maintain consistency in relevance scoring.  

### Question-Aware Document Embeddings  
These embeddings combine document and question information, guiding the selection of relevant documents.  

### Document Encodings  
Document encodings summarize entire documents, enabling efficient comparison with questions.  

### Question Encodings  
Reused question encodings ensure uniform processing across multiple documents.  

### Answer Span Determination  
The QA module identifies the answer span by predicting start and end positions within selected sentences.  

### Start Position Determination  
The module predicts the most likely starting token for the answer using a softmax distribution over sentence tokens.  

### End Position Determination  
Similarly, the module predicts the ending token, ensuring the answer span is contiguous and accurate.  

### Method for Answering a Question  
The system first receives a question and a set of sentences. It then generates scores for each sentence, selects a subset, and produces an answer.  

### Receiving Question and Sentences  
Inputs are preprocessed and encoded into numerical representations for downstream processing.  

### Generating Scores  
The sentence selector computes relevance scores for each sentence relative to the question.  

### Generating Set of Sentences  
Sentences exceeding a threshold are selected for further processing by the QA module.  

### Receiving Set of Sentences and Question  
The QA module processes the selected sentences and the question to generate an answer.  

### Generating Answer  
The module predicts an answer span within the selected sentences, providing the final output.  

### Method for Determining Set of Sentences  
The sentence selector normalizes scores, applies a threshold, and dynamically selects sentences based on question complexity.  

### Training Techniques  
The system employs weight transfer, data modification, and score normalization to enhance selector performance. Weight transfer initializes the encoder with pretrained QA model weights, data modification augments training examples, and score normalization ensures consistent scoring across paragraphs.  

This concludes the detailed description of the invention. The claims section will further define the scope of protection sought for this innovative QA system.