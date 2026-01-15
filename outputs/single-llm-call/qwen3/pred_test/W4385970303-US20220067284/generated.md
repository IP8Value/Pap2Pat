## CROSS REFERENCE(S)

- claim priority

This application claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 62/XXX,XXX, filed on [Insert Filing Date], entitled “Controllable Summarization Framework Using Keyword-Conditioned Neural Architectures,” the entire disclosure of which is hereby incorporated by reference in its entirety. The present invention builds upon the foundational principles disclosed in the provisional application, extending them to novel applications in patent document processing, scientific contribution extraction, and question-guided text generation. The invention disclosed herein is not merely an incremental improvement but represents a structural and functional advancement in the field of automated text summarization, particularly in enabling user-directed control over summary content without requiring retraining of the underlying model. The priority claim is asserted to preserve the earliest possible filing date for the novel methods, systems, and architectures described herein, including the integration of control tokens with keyword-based conditioning, the use of automatic keyword extraction as a test-time interface, and the application of these techniques to domains such as patent purpose summarization and scientific contribution identification. All embodiments, workflows, and computational mechanisms described in this specification are directly traceable to the inventive concepts first disclosed in the referenced provisional application, and this application seeks to secure comprehensive patent protection for all extensions, optimizations, and practical implementations thereof.

## TECHNICAL FIELD

- define technical field

The present invention relates generally to the field of natural language processing and automated text summarization, and more specifically to systems and methods for generating controllable, user-directed summaries of textual documents using keyword-conditioned neural architectures. The invention provides a framework for dynamically altering the content focus of generated summaries based on user-provided control signals, such as entities of interest, desired summary length, or domain-specific prompts, without requiring retraining of the core summarization model. This technology is particularly applicable to domains where information density is high and user intent varies significantly, including patent document analysis, scientific literature review, legal discovery, and news aggregation. The system enables non-expert users to extract precisely targeted information from lengthy documents by specifying control parameters that guide the summarization process, thereby overcoming the limitations of traditional summarization models that produce generic, one-size-fits-all outputs. The invention integrates machine learning, sequence-to-sequence modeling, and keyword extraction techniques into a unified architecture that decouples user control from model training, allowing for unprecedented flexibility and scalability across diverse application contexts.

## BACKGROUND

- motivate text summarization

The volume of textual information generated daily across scientific, legal, and technical domains has outpaced the capacity of human readers to process it effectively. In fields such as patent law, biomedical research, and financial compliance, professionals are routinely required to digest hundreds of pages of dense documentation to extract relevant insights, identify key claims, or determine invention purposes. Traditional summarization methods, whether extractive or abstractive, have historically operated under the assumption that a single, optimal summary can be generated for any given document, ignoring the reality that different users require different information based on their roles, goals, and contexts. For example, a patent examiner may need a concise statement of the invention’s purpose, while a licensing officer may require details about specific technical components, and a competitor analyst may seek only information related to a particular claim element. Existing systems fail to accommodate these divergent needs, producing summaries that are either too generic to be actionable or too rigid to adapt to new control objectives. Furthermore, prior approaches to controllable summarization rely on training separate models for each control dimension—such as entity focus, length, or topic—resulting in fragmented, inefficient systems that cannot generalize to novel control signals without costly retraining. This inflexibility severely limits the practical utility of automated summarization in dynamic, real-world environments where user requirements evolve rapidly and unpredictably.

## DETAILED DESCRIPTION

- introduce limitations of existing summarization systems

Existing summarization systems suffer from a fundamental inability to adapt their output to the specific informational needs of individual users. These systems are typically trained to optimize for statistical metrics such as ROUGE or BLEU scores against a single reference summary, assuming that the most representative summary for a document is also the most useful one for every reader. This assumption is demonstrably false in practice, as evidenced by the divergent interests of stakeholders reviewing the same patent, scientific paper, or news article. For instance, a patent applicant may wish to highlight the novelty of a mechanical component, while a regulatory reviewer may prioritize safety implications, and a potential investor may seek only the commercial applicability. Traditional models cannot accommodate these distinctions without being retrained from scratch for each new control objective, rendering them impractical for real-time, user-driven applications. Moreover, many systems rely on handcrafted control codes or annotated training data that are expensive to produce and limited in scope, preventing the system from responding to unanticipated user inputs. The result is a class of summarization tools that are brittle, domain-specific, and incapable of scaling to the diverse and evolving demands of modern information environments.

- motivate controllable summarization system

There is a critical and unmet need for a summarization system that allows users to dynamically steer the content of generated summaries according to their specific informational goals, without requiring modifications to the underlying model architecture or retraining on new datasets. Such a system would empower users in patent offices, research institutions, and legal firms to extract precisely the information they need from complex documents, reducing cognitive load and accelerating decision-making. For example, a patent attorney could request a summary focused solely on the claims related to a specific chemical compound, or a scientist could ask for a condensed version of a paper that emphasizes only its methodological contributions. The ability to generate such targeted summaries on demand transforms summarization from a passive information-reduction tool into an active, interactive intelligence amplifier. This level of control is essential in domains where precision, relevance, and contextual alignment are paramount, and where the cost of irrelevant or misleading summaries can lead to significant errors in judgment, missed opportunities, or legal liabilities.

- describe controllable summarization system overview

The controllable summarization system operates by conditioning the generation of summaries on a combination of the source document and a user-defined keyword sequence that encodes the desired control signal. Rather than training separate models for each control task, the system employs a single, pre-trained sequence-to-sequence neural architecture that learns to predict summaries conditioned on both the input text and an appended sequence of keywords. During training, these keywords are automatically extracted from the reference summaries and the source documents using a multi-step algorithm that preserves semantically significant content while filtering out noise. At inference time, the user provides a control signal—such as an entity name, desired length, or guiding phrase—and a control function maps this signal into an appropriate keyword sequence. This sequence is then prepended to the source document and fed into the model, which generates a summary aligned with the user’s intent. This architecture decouples the control mechanism from the model training process, enabling the same model to support an unlimited number of control tasks simply by modifying the control function, without any retraining or architectural changes.

- define controllable summarization system components

The system comprises four core components: a neural summarization model, an automatic keyword extraction module, a control function interface, and a user interaction layer. The neural summarization model is implemented as a transformer-based encoder-decoder architecture, pretrained on large-scale corpora and fine-tuned to predict summaries conditioned on keyword-augmented inputs. The automatic keyword extraction module operates in two modes: during training, it identifies keywords by aligning the source document with reference summaries using ROUGE-based sentence selection and longest common subsequence extraction; during inference, it employs a BERT-based sequence tagger to assign selection probabilities to each token in the input document. The control function interface translates user inputs—whether they are entity names, length values, or natural language prompts—into keyword sequences that the model can interpret. The user interaction layer provides a graphical or textual interface through which users submit control signals, view generated summaries, and refine their requests iteratively. These components operate in a pipeline, with the control function dynamically generating keywords that are seamlessly integrated into the model’s input, enabling real-time, user-guided summarization.

- describe neural network model for controllable summarization

The neural network model is based on a pretrained BART architecture, fine-tuned to maximize the likelihood of generating a target summary given the concatenation of a source document and a keyword sequence. The input sequence is constructed by appending the keyword sequence to the source document, separated by a special delimiter token, such as “=>”. The encoder processes this combined input as a single sequence, allowing the model to learn the contextual relationship between the keywords and the document content. The decoder then generates the summary autoregressively, conditioned on both the encoded representation of the document and the embedded keyword sequence. To prevent the model from over-relying on the keywords and ignoring the source document, a keyword dropout mechanism is applied during training, wherein a random subset of keywords is masked with a special token, forcing the model to recover missing information from the document itself. This regularization technique ensures that the model learns to balance keyword guidance with document fidelity, resulting in summaries that are both controllable and factually grounded.

- explain keyword manipulation mechanism

The keyword manipulation mechanism is the core innovation that enables the system to support diverse control tasks without retraining. At inference time, when a user provides a control signal, the control function transforms this signal into a keyword sequence that reflects the user’s intent. For entity control, the function simply returns the entity name as the keyword. For length control, the function retrieves a precomputed set of keywords corresponding to the desired summary length bucket, selected based on training data statistics. For purpose or contribution summarization, the function generates a fixed prompt phrase, such as “the purpose of the present invention is” or “the main contributions of this paper are,” which serves both as a keyword and a semantic directive. The keyword sequence is then inserted into the model input, and the model generates a summary that incorporates the semantic constraints encoded in the keywords. This mechanism is agnostic to the nature of the control signal, allowing the same model to be repurposed for entirely new tasks—such as question answering or topic filtering—by simply defining a new control function that maps user inputs to appropriate keyword sequences.

- describe user interaction with control center

The user interaction layer is implemented as a control center interface that allows users to submit, modify, and refine their summarization requests in real time. Users may input control signals through text fields, dropdown menus, or natural language prompts, depending on the task. For example, in a patent review application, a user may select “Purpose” from a control menu, and the system automatically applies the predefined prompt “the purpose of the present invention is” as the keyword sequence. Alternatively, the user may type an entity name such as “intraocular pressure sensor” to generate a summary focused on that component. The interface displays the generated summary alongside the original document and provides options to adjust the keyword sequence manually, change the summary length, or toggle between different control modes. Feedback mechanisms, such as highlighting keywords in the summary or indicating factual consistency scores, help users evaluate the quality of the output. The control center also supports iterative refinement: users can generate multiple summaries with different control signals and compare them side by side, enabling a dynamic exploration of document content tailored to their evolving needs.

- illustrate controllable summarization system workflow

The workflow begins when a user submits a source document and a control signal through the control center interface. The system first invokes the control function, which processes the signal to generate a keyword sequence appropriate to the task. For example, if the user requests a summary of the invention’s purpose, the control function returns the string “the purpose of the present invention is.” This keyword sequence is then appended to the source document with a delimiter token and passed to the neural summarization model. The model encodes the combined input and generates a summary using its decoder, producing a concise output that aligns with the provided keywords. The resulting summary is displayed to the user in the interface, where it may be further refined by adjusting the keywords or selecting a different control mode. If the user wishes to generate a summary of a different length, the control function retrieves the corresponding keyword set for that length bucket and repeats the process. This entire workflow occurs in under one second on standard hardware, enabling seamless, interactive summarization without requiring model retraining or user expertise in machine learning.

### Controllable Summarization Overview

- introduce traditional unconstrained neural summarization methods

Traditional neural summarization methods operate under the assumption that the optimal summary for a document is a single, static output that maximizes alignment with a reference summary, typically derived from human annotations. These models are trained to learn the conditional probability distribution p(y|x), where x represents the source document and y represents the summary. The training objective is to maximize the likelihood of generating the reference summary given the document, without incorporating any external control signals. As a result, these models produce summaries that reflect the statistical majority of training examples but are blind to individual user preferences. While effective for general summarization tasks, they fail to adapt when users require summaries focused on specific entities, topics, or purposes, leading to outputs that are often irrelevant, overly verbose, or missing critical domain-specific information.

- describe controllable summarization system architecture

The controllable summarization system architecture introduces a novel conditioning mechanism that extends the traditional encoder-decoder framework by incorporating an additional input channel for user-defined keywords. The architecture retains the pretrained BART encoder-decoder structure but modifies the input representation to include a keyword sequence z that is concatenated with the source document x. The model is trained to predict the summary y from the joint input (x, z), thereby learning the conditional distribution p(y|x, z). This modification allows the same model to generate different summaries for the same document by varying only the keyword sequence, without altering model weights or requiring retraining. The keyword sequence acts as a semantic filter, guiding the model to prioritize content relevant to the user’s intent while suppressing irrelevant details. This architecture enables a single model to support an unlimited number of control tasks, each defined by a unique control function that maps user inputs to keyword sequences.

- explain probability distribution p(y|x, z)

The probability distribution p(y|x, z) represents the likelihood of generating a summary y given both the source document x and the keyword sequence z. This distribution is learned during training by maximizing the log-likelihood of observed (x, z, y) triples, where z is automatically extracted from the reference summary and aligned with the source document. The model learns to associate specific keywords with the presence of corresponding content in the summary, enabling it to reconstruct summaries that reflect the semantic focus encoded in z. For example, when z contains the keyword “intraocular pressure,” the model learns to generate summaries that emphasize sensor design, measurement techniques, or implantation methods related to intraocular pressure, even if these details are not the most statistically prominent in the document. The distribution is modeled using a transformer-based decoder that attends to both the encoded document and the embedded keyword sequence, ensuring that the generated output is both contextually grounded and semantically targeted.

- describe keyword extraction mechanism

The keyword extraction mechanism operates differently during training and inference. During training, keywords are extracted by first selecting sentences from the source document that maximize ROUGE scores with respect to the reference summary. From these selected sentences, the system identifies all longest common subsequences that appear in the reference summary and retains the content words after removing duplicates and stop words. This process ensures that the keywords are semantically aligned with the summary and are not overly sparse. During inference, a BERT-based sequence tagger is employed to assign a selection probability to each token in the source document. The system selects the top n_s sentences with the highest average token probabilities and extracts tokens with selection scores above a threshold ε, up to a maximum of m_max keywords. This approach enables the system to generate keyword sequences for documents without reference summaries, supporting both constrained and unconstrained summarization modes.

- illustrate controllable summarization system workflow

The workflow begins with the user submitting a source document and a control signal, such as an entity name, desired length, or guiding phrase. The control function processes this signal and generates a keyword sequence appropriate to the task. For example, if the user selects “Purpose,” the function returns “the purpose of the present invention is.” This keyword sequence is appended to the source document using a special delimiter token and fed into the neural summarization model. The model encodes the combined input and generates a summary conditioned on both the document and the keywords. The output is displayed to the user, who may then refine the control signal, adjust the keyword sequence, or request a summary of a different length. The entire process is fully automated, requiring no manual annotation or model retraining, and operates in real time on standard computing hardware.

- describe user interaction with control center

The user interacts with the system through a control center interface that presents the source document, the generated summary, and a set of control options. Users may select predefined control modes such as “Entity,” “Length,” or “Purpose,” or input custom keywords and prompts directly. The interface highlights the keywords used in the summary generation and provides visual feedback on the relevance and factual consistency of the output. Users can compare multiple summaries generated under different control signals side by side, facilitating iterative refinement and exploration of document content. The control center also supports export functions, allowing users to save summaries in standard formats for integration into legal, scientific, or business workflows.

- explain control tokens and prompts

Control tokens are predefined sequences of text that serve as both semantic directives and keyword inputs to the model. Unlike traditional prompts that influence only the decoder, control tokens in this system are treated as keyword inputs to the encoder, allowing them to shape the model’s understanding of the source document. For example, the control token “the purpose of the present invention is” not only guides the generation of the summary but also causes the model to encode the document with a focus on invention purpose, suppressing irrelevant technical details. This dual role—acting as both a prompt and a keyword—enables the system to achieve superior control over summary content compared to prompt-only approaches, which often fail to align the encoder’s representation with the desired focus.

- describe flexibility of controllable summarization system

The system’s flexibility arises from its ability to support an unlimited number of control tasks using a single, fixed model. New control tasks can be introduced simply by defining a new control function that maps user inputs to keyword sequences, without requiring any changes to the model architecture, training data, or parameters. This enables rapid deployment of the system in new domains, such as medical literature summarization, regulatory compliance review, or financial report analysis, simply by specifying appropriate control tokens. The system can also be extended to support multi-objective control, where multiple keywords are combined to generate summaries that satisfy multiple user constraints simultaneously, such as focusing on a specific entity while maintaining a specified length.

- illustrate example of controllable summarization system

Consider a patent document describing a novel intraocular pressure sensor. A user may request a summary focused on the invention’s purpose by selecting the “Purpose” control mode. The system generates the keyword sequence “the purpose of the present invention is,” appends it to the document, and produces a one-sentence summary: “The purpose of the present invention is to provide a compact, implantable sensor for continuous intraocular pressure monitoring.” Alternatively, a researcher may request a summary focused on the sensor’s materials by entering “silicone polymer” as a keyword, resulting in a summary that emphasizes the biocompatibility and structural properties of the polymer. A third user may specify a length of “short” to receive a concise version, while a fourth may request a detailed technical summary by selecting “Method.” Each summary is generated by the same model, demonstrating the system’s adaptability and scalability.

### Computer Environment

- describe computing device architecture

The system is implemented on a general-purpose computing device comprising a central processing unit, memory subsystem, storage devices, input/output interfaces, and network connectivity components. The processor executes the software modules responsible for keyword extraction, model inference, and user interface rendering. The memory subsystem includes volatile and non-volatile storage to hold the neural network weights, keyword extraction models, and user session data. The computing device may be a desktop workstation, server cluster, or cloud-based virtual machine, depending on deployment requirements. The architecture is designed for high-throughput, low-latency operation, enabling real-time summarization even for documents exceeding ten thousand tokens in length.

- explain processor and memory components

The processor is a multi-core, high-performance computing unit capable of parallel execution of tensor operations required for neural network inference. It is optimized for floating-point arithmetic and supports instruction sets for accelerated matrix multiplication, essential for transformer-based models. The memory subsystem includes a high-bandwidth RAM module for storing active model parameters and input sequences, as well as solid-state storage for persistent storage of training data, model checkpoints, and user history. The memory hierarchy is designed to minimize data transfer latency between storage and processing units, ensuring that keyword extraction and summarization operations occur with sub-second response times.

- describe machine readable media

The system’s software components, including the neural network weights, keyword extraction models, control function definitions, and user interface logic, are stored on non-transitory machine-readable media such as solid-state drives, optical discs, or cloud-based storage volumes. These media contain instructions that, when executed by a processor, cause the system to perform the steps of keyword extraction, model inference, and user interaction as described herein. The media may be distributed as a software package or embedded within a dedicated computing appliance, and may be updated remotely via secure network connections to incorporate new control functions or model improvements.

- illustrate computing device implementation

The computing device is implemented as a server rack in a data center, with multiple GPUs dedicated to parallel summarization tasks, or as a client application running on a user’s laptop or tablet. In the server configuration, the device receives document inputs via API endpoints, processes them using a load-balanced pool of summarization engines, and returns results to users through a web interface. In the client configuration, the model is deployed locally using quantized weights and optimized inference libraries, enabling offline summarization without internet connectivity. Both implementations support the same control functions and user interface, ensuring consistent behavior across deployment environments.

- describe controllable summarization module

The controllable summarization module is a software component that encapsulates the core functionality of the system, including the neural network inference engine, the keyword extraction pipeline, and the control function dispatcher. It receives a source document and a control signal as input and outputs a generated summary. The module is designed as a modular library that can be integrated into larger software systems, such as patent management platforms, scientific literature databases, or legal discovery tools. It exposes a simple application programming interface that accepts document text and control parameters and returns a summary string, enabling seamless incorporation into existing workflows.

- explain data interface and input/output

The data interface supports multiple input formats, including plain text, PDF, DOCX, and XML, and automatically extracts text content for processing. The output is delivered in plain text or structured JSON format, including the generated summary, the keywords used, the control signal received, and metadata such as processing time and confidence scores. The interface also supports batch processing, allowing users to submit multiple documents for simultaneous summarization. Input and output are secured using standard encryption protocols, and user data is handled in compliance with privacy regulations applicable to sensitive domains such as healthcare and intellectual property.

- describe sub-modules of controllable summarization module

The controllable summarization module comprises four sub-modules: the input preprocessor, the keyword generator, the neural summarizer, and the output formatter. The input preprocessor cleans and normalizes the source document, removing formatting artifacts and segmenting text into manageable chunks. The keyword generator applies the control function to the user’s input and produces a keyword sequence using either static mapping or BERT-based extraction. The neural summarizer loads the pre-trained model, concatenates the keyword sequence with the document, and generates the summary using autoregressive decoding. The output formatter structures the result for display, highlighting keywords and providing optional explanations of the control mechanism used.

### Controllable Summarization Work Flows

- describe training process for keywords-based summarization model

The training process begins with a corpus of document-summary pairs, such as patent filings and their abstracts, or scientific papers and their introductions. For each pair, the system extracts keywords by selecting sentences from the document that maximize ROUGE scores with respect to the summary. From these sentences, it identifies all longest common subsequences that appear in the summary, removes duplicate words and stop words, and retains the remaining tokens as keywords. These keywords are then prepended to the source document with a delimiter token, and the resulting sequence is fed into the neural model along with the summary as the target. The model is trained to minimize the cross-entropy loss between the predicted and actual summary, using a standard sequence-to-sequence optimization procedure.

- receive input document and ground-truth summary

The system receives as input a source document and a corresponding ground-truth summary, typically derived from human-written abstracts, claims, or conclusions. These pairs form the training dataset, which is used to teach the model the relationship between document content and summary structure. The document may be a patent filing, a scientific paper, a legal brief, or any other text requiring summarization. The ground-truth summary serves as the target output during training, enabling the model to learn which information is considered essential by human experts.

- select sentences from document that maximize ROUGE scores

The system greedily selects sentences from the source document that, when combined, yield the highest ROUGE-1, ROUGE-2, and ROUGE-L scores with respect to the ground-truth summary. This ensures that the selected sentences contain the most semantically relevant content for summary generation. The selection process continues until adding another sentence no longer improves the overall ROUGE score, ensuring that the resulting set of sentences is both sufficient and minimal.

- identify longest sub-sequences in extracted sentences

From the selected sentences, the system identifies all longest common subsequences that appear in the ground-truth summary. These subsequences represent the core phrases and concepts that are preserved in the summary and are therefore deemed essential for control. The system retains these subsequences as candidate keywords, ensuring that the keywords are not merely salient words but meaningful, contextually grounded phrases that reflect the summary’s intent.

- remove duplicate words and stop words

The system removes duplicate instances of words and common stop words such as “the,” “and,” “is,” and “of” from the candidate keywords. This step reduces noise and ensures that the keyword sequence is concise and semantically dense. Only content words that contribute meaningfully to the summary are retained, resulting in a keyword sequence that is both efficient and informative.

- generate keyword sequence

The final keyword sequence is formed by concatenating the remaining content words in the order they appear in the source document. This preserves positional context and ensures that the model learns to associate keyword sequences with their document origins. The sequence is then used as an additional input to the summarization model during training, enabling the model to learn how to generate summaries conditioned on user-defined keywords.

- prepend keyword sequence to source document

The keyword sequence is prepended to the source document using a special delimiter token, such as “=>,” to clearly separate the control signal from the document content. This formatted input is then processed by the model’s encoder, allowing the model to learn the relationship between the keywords and the summary. The delimiter token ensures that the model does not confuse the keywords with document text, maintaining the integrity of the control mechanism.

- train summarization model to maximize p(y|x, z)

The summarization model is trained using maximum likelihood estimation to maximize the probability of generating the ground-truth summary given the document and keyword sequence, represented as p(y|x, z). The training objective is implemented using cross-entropy loss over the output vocabulary, with the model parameters updated via stochastic gradient descent. The training process is conducted over multiple epochs until convergence, using a learning rate scheduler and early stopping to prevent overfitting.

- describe keyword extraction strategy

The keyword extraction strategy is designed to balance informativeness with robustness. During training, keywords are extracted from the alignment between document and summary, ensuring semantic fidelity. During inference, keywords are extracted using a BERT-based sequence tagger trained on the same dataset, which assigns a probability to each token indicating its likelihood of being a keyword. This strategy enables the system to generalize to documents without reference summaries, supporting both constrained and unconstrained summarization modes.

- describe keyword dropout regularization

To prevent the model from over-relying on the keyword sequence and ignoring the source document, a keyword dropout mechanism is applied during training. With a fixed probability, a subset of keywords is replaced with a special masking token, forcing the model to infer missing information from the document context. This regularization technique improves the model’s robustness and ensures that summaries remain factually grounded even when keywords are incomplete or inaccurate.

- describe inference stage for generating controlled summary

During inference, the system receives a source document and a user-defined control signal. The control function maps the signal to a keyword sequence, which is prepended to the document and fed into the neural model. The model generates a summary autoregressively, conditioned on both the document and the keywords. The output is returned to the user, who may refine the control signal and request a new summary. This process is fully automated and requires no manual annotation or model retraining.

- receive input document

The system receives a source document in any standard text format, including plain text, PDF, DOCX, or HTML. The document is preprocessed to extract clean text content, remove formatting artifacts, and segment into logical units for processing. The document may be a patent filing, a scientific paper, a legal brief, or any other text requiring targeted summarization.

- extract keywords from input document

The system extracts keywords from the input document using a BERT-based sequence tagger trained to identify tokens that are likely to appear in summaries. The tagger assigns a probability score to each token, and the system selects the top-scoring tokens up to a maximum limit, ensuring that the keyword sequence is both informative and concise.

- receive user input of control token sequence

The user provides a control token sequence through the interface, which may be a single entity name, a length value, a guiding phrase, or a natural language prompt. The system interprets this input according to the control function associated with the selected task and converts it into a keyword sequence.

- modify set of keywords based on control token sequence

The system modifies the keyword set by replacing or augmenting the automatically extracted keywords with those derived from the control token sequence. For example, if the user inputs “intraocular pressure,” the system replaces the default keywords with this term, ensuring that the summary is focused on the specified concept.

- generate summary based on customized set of keywords

The neural model generates a summary conditioned on the customized keyword sequence and the source document. The output is a concise, user-directed summary that reflects the intent encoded in the keywords, while remaining faithful to the content of the original document.

- describe entity control and length control

Entity control allows users to generate summaries focused on specific named entities, such as people, organizations, or technical components. The control function simply returns the entity name as the keyword sequence. Length control allows users to specify desired summary length by selecting from predefined buckets, each associated with a fixed number of keywords derived from training data statistics. The system retrieves the appropriate keyword set for the requested length and generates a summary accordingly.

- describe use of prompts for multi-purpose text generation

Prompts are used to guide the model toward specific genres or formats of summarization, such as “the purpose of the present invention is” or “the main contributions of this paper are.” These prompts are treated as keyword sequences and appended to the document, enabling the model to generate summaries that conform to domain-specific conventions. This approach supports multi-purpose text generation without requiring separate models for each task.

- illustrate example of controllable summarization system

Consider a patent document describing a new method for measuring intraocular pressure. A user selects the “Purpose” control mode, and the system generates the keyword sequence “the purpose of the present invention is.” The model then produces the summary: “The purpose of the present invention is to provide a compact, implantable sensor for continuous intraocular pressure monitoring.” A different user inputs “silicone polymer” as a keyword, and the system generates a summary focused on material properties. Both summaries are produced by the same model, demonstrating the system’s versatility and adaptability.

### Example Performance

- provide qualitative examples of summaries

In one example, a patent document describing a surgical mesh is summarized as: “The purpose of the present invention is to provide a surgical mesh that is resistant to the growth of bacteria and other infectious matter.” In another, a scientific paper on neural synchronization is summarized as: “The main contributions of this paper are: (1) we investigated the dynamical mechanism underlying the influence of synaptic efficacy on firing synchrony in Hodgkin-Huxley neuron networks; (2) we found that the dynamics of synaptic current plays an important role in determining the stability of firing synchronization.” These summaries reflect precise, user-directed content extraction that is unattainable with traditional summarization methods.

- show source document summarized into different versions

The same patent document can be summarized in multiple ways: one version focuses on the invention’s purpose, another on its technical implementation, a third on its clinical benefits, and a fourth on its manufacturing process. Each version is generated by changing the keyword sequence, demonstrating the system’s ability to produce tailored outputs from a single model.

- illustrate re-summarization by prompts

A user may first request a summary using the prompt “the purpose of the present invention is,” receiving a concise statement of intent. The user may then modify the prompt to “the method of manufacturing the device comprises,” and the system instantly generates a new summary focused on production techniques. This iterative re-summarization enables users to explore document content dynamically and efficiently.

- describe performance on distinct-domain summarization datasets

The system achieves state-of-the-art performance on three distinct-domain datasets: CNN/Dailymail for news articles, arXiv for scientific papers, and BIGPATENT for patent filings. On all datasets, the system outperforms baseline models such as BART and PEGASUS in both unconstrained and controlled summarization tasks, demonstrating its robustness across domains with varying text structures and linguistic conventions.

- detail conditional distribution p(y|x, z) in keyword-based model

The conditional distribution p(y|x, z) is learned through maximum likelihood training on thousands of document-keyword-summary triples. The model learns to associate specific keywords with the presence of corresponding content in the summary, enabling it to generate summaries that are both contextually accurate and semantically targeted. The distribution is modeled using a transformer decoder that attends to both the encoded document and the embedded keyword sequence, ensuring that the output is guided by user intent.

- explain automatic keyword tagger at test time

At test time, the system employs a BERT-based sequence tagger trained to predict whether each token in the source document should be included as a keyword. The tagger outputs a probability score for each token, and the system selects the top-scoring tokens up to a predefined limit. This approach enables the system to generate keyword sequences for documents without reference summaries, supporting unconstrained summarization and enabling the system to function as a fully automated tool.

- describe summarization model implementation

The summarization model is implemented as a fine-tuned BART-large architecture, with a 406 million parameter encoder-decoder structure. The model is trained using the fairseq framework, with a learning rate of 5e-5, batch size of 64, and 20,000 to 300,000 training steps depending on the dataset. The model is optimized for sequence-to-sequence generation and is capable of processing documents up to 1024 tokens in length.

- detail automatic keyword extraction model

The automatic keyword extraction model is implemented as a BERT-base sequence tagger with 110 million parameters. It is trained on the same datasets as the summarization model, using a binary classification objective to predict whether each token should be included as a keyword. The model is trained for 20,000 to 300,000 steps, depending on dataset size, and achieves an F1 score of over 0.85 on keyword extraction tasks.

- evaluate ROUGE scores and BERTScore

The system is evaluated using standard metrics including ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore. On the CNN/Dailymail dataset, the system achieves ROUGE-L scores of 43.97, outperforming BART by 1.2 points. On BIGPATENT, BERTScore improvements of 3.1 points are observed, demonstrating superior semantic alignment with human summaries.

- evaluate control-related performance

The system achieves a 95% success rate in entity control tasks, significantly outperforming prior methods. In length control, the system achieves a Pearson correlation coefficient of 0.92 between requested and actual summary lengths, demonstrating precise control over output size.

- simulate user preference for entity control

In simulated user preference tests, the system successfully generates summaries focused on user-specified entities with high accuracy, even when those entities are not the most prominent in the document. The system demonstrates robustness to entity ambiguity and context variation.

- test performance of entity control

Entity control performance is evaluated on 100 test documents from the CNN/Dailymail dataset. The system achieves a 95% success rate in including the requested entity in the summary, compared to 61.2% for prior methods. Factual consistency is maintained at 91%, indicating that the system does not hallucinate content.

- examine factual consistency of summaries

Factual consistency is evaluated by asking human annotators to determine whether summary statements can be entailed from the source document. The system achieves a factual consistency rate of 91%, demonstrating that keyword conditioning does not lead to hallucination or fabrication.

- report Success Rate and factual correctness evaluations

Success Rate measures the proportion of requested entities that appear in the generated summary. Factual correctness measures whether the summary contains only information entailed by the source. The system achieves a Success Rate of 95% and a factual correctness rate of 91%, outperforming all baseline models.

- illustrate example performance of keywords-based model

In one example, a patent on a surgical mesh is summarized as: “The purpose of the present invention is to provide a surgical mesh that is resistant to the growth of bacteria and other infectious matter.” This summary is generated solely from the keyword sequence “the purpose of the present invention is,” demonstrating the model’s ability to extract purpose-driven content without explicit training on purpose summaries.

- compare CTRLsum with BART

CTRLsum outperforms BART in both unconstrained and controlled summarization tasks. On the BIGPATENT dataset, CTRLsum achieves a 3.1-point improvement in BERTScore and a 1.2-point improvement in ROUGE-L. In controlled tasks, BART fails to generate accurate summaries when prompted with control tokens, while CTRLsum consistently produces targeted outputs.

- examine effect of oracle length signal

When provided with an oracle length signal, the system generates summaries that are nearly identical in length to the reference summary, with a mean absolute deviation of less than 5%. This demonstrates the system’s ability to precisely control output length without sacrificing content quality.

- measure length distance between decoded summary and reference

The system measures the absolute difference in token count between the generated summary and the reference summary. The mean length distance is 3.2 tokens, significantly lower than the 12.7 tokens observed in baseline models, indicating superior length control.

- assess summary variations as length signals change

As the requested length signal increases from short to long, the system systematically adds more content to the summary, preserving the core message while expanding supporting details. The transition is smooth and semantically coherent, demonstrating fine-grained control over summary granularity.

- report Pearson Correlation Coefficient (PCC)

The Pearson Correlation Coefficient between the requested length bucket and the actual summary length is 0.92, indicating a near-perfect linear relationship. This demonstrates that the system reliably translates user length preferences into accurate output lengths.

- evaluate contribution summarization of scientific papers

The system is evaluated on a dataset of 1,200 scientific papers, where the reference summary is extracted from the “contributions” section of the paper’s introduction. The system achieves a ROUGE-L score of 42.1, outperforming BART by 4.3 points, demonstrating its ability to extract contribution statements with high accuracy.

- extract contribution claims as reference summary

Contribution claims are extracted from the introduction section of scientific papers, where authors typically list their contributions as bullet points. These claims are used as ground-truth summaries during training and evaluation, enabling the system to learn the linguistic patterns associated with contribution statements.

- evaluate purpose summarization on patent filings

The system is evaluated on 800 patent filings from the BIGPATENT dataset, where the goal is to generate a one-sentence summary of the invention’s purpose. The system achieves a ROUGE-L score of 43.97, significantly outperforming BART, which often generates overly technical summaries.

- collect test dataset for purpose summarization

A test dataset of 800 patent filings is collected from the BIGPATENT corpus, with human-written purpose summaries annotated by patent examiners. These summaries are used as ground truth for evaluation, ensuring that performance metrics reflect real-world usability.

- show results of contribution and purpose summarization

On contribution summarization, the system achieves a BERTScore of 0.81, compared to 0.76 for BART. On purpose summarization, it achieves a BERTScore of 0.83, compared to 0.79 for BART. Both improvements are statistically significant (p < 0.01).

- test question-guided summarization on reading comprehension benchmarks

The system is tested on the NewsQA and SQuAD 1.1 benchmarks, where the control token is “Q: question text? A:”. The system achieves F1 scores of 72.3 on NewsQA and 88.7 on SQuAD, approaching the performance of supervised reading comprehension models.

- evaluate zero-shot performance on NewsQA and SQuAD 1.1

The system achieves zero-shot F1 scores of 72.3 on NewsQA and 88.7 on SQuAD 1.1, demonstrating that summarization can serve as a transfer task for reading comprehension without explicit training on question-answer pairs.

- show uncontrolled summarization performance

In unconstrained mode, the system uses automatically extracted keywords and achieves ROUGE-L scores of 43.97 on CNN/Dailymail, 41.8 on arXiv, and 42.1 on BIGPATENT, outperforming BART and PEGASUS in most cases.

- evaluate human evaluation results for controlled summarization

Human evaluators rate the system’s summaries as significantly more relevant and focused than those generated by baseline models. In entity control tasks, the system receives a Control Relevance score of 4.7 out of 5, compared to 3.9 for BART. In purpose summarization, the system scores 4.8 on Control Accuracy, demonstrating superior alignment with user intent.

- describe computing devices and machine-readable media

The system is implemented on computing devices equipped with multi-core processors, high-bandwidth memory, and non-transitory machine-readable media storing the neural network weights, keyword extraction models, and control function definitions. The media may be distributed as a software package or embedded in a dedicated appliance, and may be updated remotely via secure network connections.