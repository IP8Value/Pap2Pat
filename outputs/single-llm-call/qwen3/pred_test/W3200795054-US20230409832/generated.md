# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR (IF APPLICABLE)

- disclose prior art  
Prior to the present invention, methods for generating contrastive explanations in natural language processing systems relied primarily on direct perturbation of input text without leveraging semantically structured attribute classifiers to guide the generation process. Existing approaches, including Generate Your Counterfactuals (GYC) and Minimal Contrastive Editing (MICE), produced textual modifications aimed at flipping model predictions by replacing, inserting, or deleting words based on gradient-based importance scores or fine-tuned language models. These methods did not incorporate external or internal attribute classifiers to identify latent subtopics or conceptual dimensions that distinguish between classification outcomes. While such techniques achieved limited success in generating fluent and minimal perturbations, they failed to provide interpretable insights into why a model classified an input as belonging to one category rather than another. Furthermore, prior systems required dataset-specific fine-tuning of language models, imposed rigid constraints on target classes, and lacked mechanisms to quantify the addition or removal of abstract semantic features. No prior system integrated a multi-objective optimization framework that simultaneously enforced prediction flip, attribute change, fluency, and edit minimality using independently trained attribute classifiers derived from heterogeneous data sources. The present invention overcomes these limitations by introducing a novel framework that uses attribute classifiers to guide contrastive text generation, thereby enabling not only more interpretable and reliable explanations but also transferability across domains without retraining the underlying classification model.

## BACKGROUND OF THE INVENTION

- motivate natural language processing  
The increasing deployment of natural language processing systems in high-stakes decision-making environments—such as customer service automation, legal document review, financial risk assessment, and public policy analysis—has underscored the critical need for transparent and trustworthy AI. Black-box models, including deep neural networks, have demonstrated superior performance in classifying and generating human language; however, their opaque decision mechanisms hinder accountability, regulatory compliance, and user trust. In applications where explanations are legally mandated or ethically necessary, such as in automated hiring or credit scoring, stakeholders require not only to know what decision was made but also to understand why alternative outcomes were not selected. Traditional feature attribution methods, such as saliency maps or SHAP values, identify influential words but fail to articulate the conceptual shift that would alter the model’s prediction. Contrastive explanations, which answer “Why not Y instead of Z?”, offer a more intuitive form of reasoning aligned with human cognition. Yet, existing methods for generating such explanations in text remain superficial, often producing semantically incoherent or overly disruptive edits that obscure the underlying logic of the model. Without a mechanism to anchor perturbations in meaningful, interpretable attributes—such as topics, sentiments, or domain-specific concepts—these explanations lack depth and fail to serve as tools for model auditing, bias detection, or user empowerment. The present invention addresses this gap by introducing a systematic, attribute-guided approach to contrastive explanation generation that transforms textual perturbation from a mechanical search into a semantically informed reasoning process.

## SUMMARY

- outline perturbed text generation  
The present invention discloses a computer-implemented method for generating contrastive textual explanations by identifying and manipulating latent semantic attributes that influence machine learning model predictions. The method receives an input text classified by a black-box model into a first category and generates a perturbed version of the text that is predicted by the model to belong to a second, contrasting category. The perturbation is not randomly generated but is guided by a set of pre-trained attribute classifiers that detect the presence or absence of semantically meaningful subtopics within the text. The system identifies key words in the input text that are most influential to the model’s classification, replaces or removes them with masked tokens, and employs a masked language model to propose candidate replacements. Each candidate perturbation is evaluated against an objective function that balances four criteria: the likelihood of prediction flip, the number of attribute changes, the semantic fluency of the resulting text, and the minimal edit distance from the original input. The system then selects the perturbation that maximizes this composite objective, producing a contrastive explanation that not only alters the classification but also explicitly identifies which attributes were added or removed to achieve the change. This approach enables the generation of human-interpretable, conceptually grounded explanations that reveal the hidden decision logic of the model, thereby enhancing transparency, trust, and diagnostic utility without requiring access to the model’s internal parameters or retraining.

## DETAILED DESCRIPTION

- define computer program product  
The present invention encompasses a computer program product comprising a non-transitory computer-readable storage medium having program instructions embodied therewith, the program instructions being executable by a processor to perform a method for generating contrastive textual explanations. The program instructions, when executed, cause the processor to receive an input text classified by a machine learning model into a first class, determine a set of attribute classifiers trained to detect the presence or absence of semantically relevant subtopics in text, identify a subset of words in the input text that contribute most significantly to the model’s classification decision, generate a plurality of candidate perturbations by masking the identified words and substituting them with alternative terms using a pre-trained masked language model, evaluate each candidate perturbation using a multi-objective scoring function that quantifies prediction flip, attribute change, fluency, and edit distance, and output a final perturbed text along with a set of attributes that were added or removed to effect the classification change.

- describe computer readable storage medium  
The computer readable storage medium is a physical, non-transitory device capable of storing digital data for retrieval and execution by a computing system. It may be implemented as a hard disk drive, solid-state drive, optical disc, flash memory, or any other tangible medium that retains data in the absence of power. The storage medium is not a transitory signal or propagated wave but a physical component of a computing apparatus, configured to hold the program instructions and associated data structures required to execute the contrastive explanation generation method. The storage medium may be local to the computing device or remotely accessible via a network, provided that the instructions are loaded into volatile memory prior to execution.

- list examples of computer readable storage medium  
Examples of computer readable storage media include, but are not limited to, magnetic storage devices such as hard disk drives and floppy disks, optical storage devices such as CD-ROMs, DVDs, and Blu-ray discs, solid-state storage devices such as USB flash drives, SD cards, and SSDs, and non-volatile memory modules such as EEPROM and PROM. The medium may also include distributed storage systems, such as network-attached storage or cloud-based storage services, provided that the program instructions are ultimately downloaded and executed on a local processor.

- explain computer readable program instructions  
The computer readable program instructions comprise a sequence of executable commands that, when loaded into a processor’s memory and executed, cause the system to perform the steps of the disclosed method. These instructions are written in a programming language compatible with the target computing environment and may be compiled or interpreted prior to execution. The instructions include modules for text preprocessing, attribute classifier invocation, masked language modeling, objective function evaluation, and output generation. The program instructions are designed to operate independently of the underlying classification model, making the system adaptable to any black-box text classifier without requiring internal access or modification.

- describe downloading instructions from storage medium  
The program instructions may be downloaded from the computer readable storage medium to a volatile memory component of a computing device, such as random access memory, prior to execution. The downloading process may occur over a network connection, such as the Internet or a local area network, or through direct physical transfer via removable media. Once loaded into memory, the instructions are executed by the central processing unit to carry out the operations of the contrastive explanation generation system.

- outline network components for instruction transmission  
The transmission of program instructions from a remote storage location to a local computing device may involve network components such as routers, switches, firewalls, and communication protocols including TCP/IP, HTTP, and FTP. The instructions may be packaged in data packets and transmitted across wired or wireless networks, including cellular, Wi-Fi, or satellite connections. The receiving device may authenticate the source of the instructions and verify their integrity using cryptographic signatures before loading them into memory for execution.

- specify types of computer readable program instructions  
The computer readable program instructions may include machine code, bytecode, interpreted scripts, or high-level source code. They may be implemented as software modules written in Python, Java, C++, or other general-purpose programming languages. The instructions may also include configuration files, model weights, and metadata required for the operation of attribute classifiers and masked language models. The instructions are structured to be modular, allowing for independent updates to the language model, attribute classifiers, or objective function without requiring a complete system reinstallation.

- explain execution of instructions on user's computer  
When executed on a user’s computing device, the program instructions initiate a sequence of operations that begins with the ingestion of an input text, followed by the application of feature attribution techniques to identify influential words. The system then invokes pre-trained attribute classifiers to assess the semantic profile of the input, generates candidate perturbations using a masked language model, evaluates each candidate against a multi-objective function, and selects the optimal perturbation. The final output is displayed to the user as a modified text accompanied by a list of added and removed attributes, enabling the user to understand the conceptual basis for the model’s classification change.

- describe remote computer execution  
The program instructions may also be executed on a remote server or cloud-based computing system. In such embodiments, the user’s device transmits the input text over a network to the remote system, which performs the full contrastive explanation generation process and returns the perturbed text and associated attributes as a response. This architecture allows for centralized model management, reduced computational burden on client devices, and secure handling of sensitive input data.

- introduce electronic circuitry execution  
In alternative embodiments, the program instructions may be implemented directly in electronic circuitry, such as field-programmable gate arrays or application-specific integrated circuits, configured to perform the steps of the method in hardware. Such implementations offer reduced latency and energy efficiency, making them suitable for real-time applications in embedded systems or edge computing environments.

- describe flowchart and block diagram illustrations  
The method may be represented by flowcharts and block diagrams that illustrate the sequential and parallel operations involved in the contrastive explanation generation process. These diagrams depict the input stage, the attribute classification module, the perturbation generation module, the objective function evaluation unit, and the output interface. Each block corresponds to a distinct computational step, and the arrows indicate the flow of data and control between components.

- explain implementation of functions/acts  
Each function or act described in the flowchart is implemented by one or more program instructions stored on the computer readable storage medium. The implementation may involve subroutine calls, object-oriented methods, or procedural logic, depending on the programming paradigm. The functions are designed to be deterministic, reproducible, and verifiable, ensuring consistent behavior across multiple executions.

- describe computer readable storage medium with instructions  
The computer readable storage medium contains a complete set of instructions necessary to execute the entire method without requiring external dependencies beyond standard libraries and pre-trained models. The medium may be distributed as a software package, firmware update, or downloadable application, and may include documentation, licensing terms, and version control metadata.

- outline loading instructions onto computer  
The instructions are loaded onto the computer by means of a boot sequence, operating system loader, or application runtime environment. The loading process ensures that all required components—attribute classifiers, language models, and optimization routines—are initialized in memory and validated for integrity before execution begins.

- explain series of operational steps  
The series of operational steps includes receiving an input text, determining influential words using gradient-based attribution, applying masked language modeling to generate candidate substitutions, evaluating each candidate using a composite objective function, and selecting the perturbation that best satisfies the criteria of prediction flip, attribute change, fluency, and minimal edit distance. The process is iterative and adaptive, allowing for multiple rounds of perturbation if necessary to achieve a valid contrast.

- describe flowchart and block diagram functionality  
The flowchart and block diagram functionality provides a visual representation of the system architecture, enabling developers and auditors to verify the correctness of the implementation. Each block corresponds to a specific computational module, and the interconnections reflect the data dependencies and control flow inherent in the method.

- introduce computing environment  
The computing environment in which the invention operates includes a central processing unit, memory, input/output interfaces, and network connectivity. The environment may be a personal computer, server, mobile device, or embedded system, provided that it supports the execution of the program instructions and the loading of the necessary models and classifiers.

- describe computing device components  
The computing device includes a processor, memory, storage, input devices such as keyboards or touchscreens, output devices such as displays or speakers, and communication interfaces such as Ethernet or Wi-Fi modules. The device may be standalone or connected to a networked infrastructure, and may be configured to operate in real-time or batch mode.

- outline network components  
The network components include routers, switches, firewalls, load balancers, and communication protocols that facilitate data exchange between client devices and remote servers. These components ensure secure, reliable, and low-latency transmission of input texts and output explanations across distributed systems.

- explain CAT program functionality  
The CAT program is a software application that implements the method of contrastive attributed explanation generation. It receives an input text and a classification model, invokes attribute classifiers to extract semantic features, generates candidate perturbations using a masked language model, evaluates them using a multi-objective function, and outputs a contrastive explanation with annotated attribute changes. The program is modular, allowing for the substitution of different attribute classifiers, language models, or optimization parameters without altering the core architecture.

- describe generating perturbed text  
The generation of perturbed text involves identifying salient words in the input, replacing them with masked tokens, and using a pre-trained masked language model to propose plausible substitutions. The system evaluates each substitution for its likelihood of flipping the classification, its impact on attribute scores, its fluency as measured by language model perplexity, and its edit distance from the original. The best candidate is selected according to the composite objective function.

- outline determining classifiers for text data  
The classifiers are determined by training binary or multiclass models on external datasets to detect the presence or absence of semantic attributes such as topics, sentiments, or domain-specific concepts. These classifiers are independent of the target classification model and may be trained on unrelated datasets, enabling transferability across domains.

- explain classifier model functionality  
Each classifier model takes as input a text and outputs a confidence score indicating the degree to which a specific attribute is present. These scores are used to quantify the change in semantic profile between the original and perturbed texts, forming a critical component of the objective function.

- describe mask module functionality  
The mask module identifies the most influential words in the input text using feature attribution techniques and replaces them with a [MASK] token. The module may mask single words or multiple words in sequence, depending on the complexity of the classification decision, and generates a set of masked variants for subsequent perturbation.

- outline determining important words  
Important words are determined by computing attribution scores using methods such as Integrated Gradients or LIME, which quantify the contribution of each word to the model’s prediction. Words with scores above a predefined threshold are selected for masking.

- explain generating candidate perturbations  
Candidate perturbations are generated by substituting the masked tokens with alternative words proposed by a masked language model. The system retrieves the top-k most likely replacements and evaluates each for its effect on classification, attribute scores, fluency, and edit distance.

- describe determining edit distance  
Edit distance is determined using the word-level Levenshtein distance, which counts the minimum number of insertions, deletions, or substitutions required to transform the original text into the perturbed version. The distance is normalized by the length of the input to ensure comparability across texts of varying size.

- outline language model functionality  
The language model is a pre-trained masked language model such as BERT or RoBERTa, which predicts the most probable words to fill masked positions based on contextual embeddings. The model is used to generate fluent, grammatically correct perturbations that preserve the syntactic structure of the input.

- explain selecting candidate perturbation  
The candidate perturbation is selected by maximizing a composite objective function that balances the prediction flip, the number of attribute changes, the fluency of the output, and the edit distance. The selection is performed by evaluating each candidate against the function and choosing the one with the highest score.

- describe objective function  
The objective function is a weighted sum of four terms: a contrastive score that penalizes failure to flip the classification, an attribute change score that encourages minimal but meaningful attribute shifts, a fluency score that rewards high language model likelihood, and an edit distance penalty that discourages excessive modification. The weights are determined during hyperparameter tuning and remain fixed during deployment.

- outline maximizing objective function  
Maximization of the objective function is achieved by evaluating all candidate perturbations and selecting the one that yields the highest value. The process is deterministic and does not require iterative optimization, enabling efficient execution even on resource-constrained devices.

- explain selecting perturbation with maximum objective function  
The perturbation with the maximum objective function value is selected because it best satisfies the competing goals of prediction change, semantic relevance, linguistic fluency, and textual minimalism. This selection ensures that the output explanation is both effective and interpretable.

- conclude detailed description  
The detailed description above provides a comprehensive disclosure of the invention, including its components, operations, and modes of implementation. The invention is not limited to the specific embodiments described but encompasses all variations and equivalents that fall within the scope of the claims.

- define contrastive attribute text (CAT) program  
The contrastive attribute text (CAT) program is a software system designed to generate human-interpretable contrastive explanations for black-box text classifiers by leveraging attribute classifiers to guide perturbation generation. The program operates without requiring access to the internal parameters of the target model and produces explanations that include both the modified text and the semantic attributes responsible for the classification change.

- describe objective function E.1  
The objective function, denoted as E.1, is defined as a weighted combination of four components: a contrastive term that ensures the perturbed text is classified differently from the original, an attribute change term that minimizes the number of attributes altered while ensuring at least one is changed, a fluency term that penalizes low-probability sentences according to a language model, and an edit distance term that penalizes deviations from the original text. The weights are tuned to balance interpretability and fidelity.

- explain attribute classifiers  
Attribute classifiers are machine learning models trained to detect the presence or absence of semantic attributes such as topics, sentiments, or domain-specific concepts in text. These classifiers are independent of the target classification model and may be trained on external datasets, enabling the CAT program to operate across diverse domains without retraining.

- detail edit distance calculation  
Edit distance is calculated as the word-level Levenshtein distance between the original and perturbed texts, representing the minimum number of word insertions, deletions, or substitutions required to convert one into the other. The distance is normalized by the total number of words in the original text to produce a dimensionless metric.

- quantify fluency of generated sentence  
Fluency is quantified by computing the masked language modeling loss of the perturbed sentence using a pre-trained language model such as GPT-2 or BERT. The fluency score is the ratio of the loss of the perturbed sentence to the loss of the original sentence, with a value closer to one indicating higher fluency.

- introduce hyperparameters  
Hyperparameters include weights assigned to each component of the objective function, thresholds for attribute detection, and the number of candidate perturbations generated per mask. These values are determined once per dataset through qualitative evaluation and remain fixed during deployment.

- outline embodiment of CAT program  
An embodiment of the CAT program operates on a server system that receives text inputs from client devices, processes them using pre-loaded attribute classifiers and language models, and returns contrastive explanations via a web interface. The program is designed for scalability, supporting batch processing of multiple inputs simultaneously.

- illustrate example perturbations  
An example perturbation transforms the sentence “Many technologies may be a waste of time and money” into “Many technologies jobs may be a waste of time and money,” with the attribute change indicating the addition of “employment” and the removal of “finance.” This change flips the classification from Sci-Tech to Business, and the attribute annotations reveal the conceptual basis for the model’s decision.

- describe table of results  
A table of results presents quantitative metrics across multiple datasets, including flip rate, edit distance, fluency, and content preservation, demonstrating that the CAT program outperforms prior methods in all metrics except fluency on one dataset, where it remains competitive without requiring fine-tuning.

- explain classifier model bias detection  
The CAT program enables bias detection by revealing which attributes are systematically associated with certain classifications. For instance, if gender-related attributes consistently appear in perturbations that flip a hiring classifier’s decision, this indicates potential bias in the model’s training data.

- illustrate operational processes of CAT program  
The operational processes include receiving an input text, determining influential words, masking those words, generating candidate perturbations, evaluating each candidate using the objective function, selecting the optimal perturbation, and outputting the result with attribute annotations.

- receive text input  
The system receives an input text from a user interface, file, or API endpoint. The text is preprocessed to remove formatting and tokenize into words for analysis.

- determine classifiers for text data  
The system invokes a set of pre-trained attribute classifiers to compute confidence scores for each attribute in the input text, producing a semantic profile that characterizes the text’s conceptual content.

- insert masks for important words  
The system inserts [MASK] tokens in place of the most influential words as determined by feature attribution methods, creating a masked version of the input for perturbation.

- generate candidate perturbations  
The system uses a masked language model to generate a set of candidate replacements for each masked token, producing a combinatorial set of potential perturbations.

- determine edit distance  
For each candidate, the system calculates the word-level Levenshtein distance between the original text and the candidate perturbation.

- determine fluency of candidate perturbations  
The system computes the language model likelihood of each candidate perturbation to assess its grammatical and semantic fluency.

- select candidate that maximizes objective function  
The system evaluates each candidate using the composite objective function and selects the one that achieves the highest score.

- provide perturbed text and attribute data  
The system outputs the final perturbed text alongside a list of attributes that were added (+) or removed (−) to effect the classification change.

- illustrate system architecture  
The system architecture comprises an input module, an attribution module, a masking module, a perturbation generation module, an attribute evaluation module, an objective function evaluator, and an output module, all interconnected to enable end-to-end contrastive explanation generation.

- identify important words  
Important words are identified by computing attribution scores using gradient-based methods and selecting those above a threshold value.

- perturb important words  
Important words are perturbed by replacing them with masked tokens and generating candidate substitutions using a masked language model.

- evaluate language model  
The language model is evaluated for its ability to reconstruct fluent text from masked inputs, ensuring that generated perturbations are grammatically sound.

- reclassify perturbed text  
The perturbed text is passed to the target classification model to confirm that the prediction has flipped to the desired contrasting class.

- illustrate components of computing device  
The computing device includes a processor, memory, storage, input/output interfaces, and network connectivity, all configured to execute the program instructions and support the CAT program’s operations.

- describe computer-readable storage media  
Computer-readable storage media include physical devices such as hard drives, SSDs, optical discs, and flash memory that store the program instructions and data required for the CAT program to function. These media are non-transitory and retain data without power.