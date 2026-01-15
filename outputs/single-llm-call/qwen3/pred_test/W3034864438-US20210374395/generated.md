# DESCRIPTION

## FIELD

- relate machine-learned models

The present invention relates to computer-implemented systems and methods for extracting structured information from form-like documents using machine-learned models that learn dense, context-aware representations of extraction candidates and their associated field types. These models are trained to recognize and distinguish between candidate text portions based on their spatial relationships with neighboring textual elements, their positional coordinates within the document layout, and the semantic characteristics of the target field to which they may belong. The system operates without reliance on predefined templates or rule-based heuristics, instead leveraging neural architectures that encode both local textual context and global spatial configuration into unified embedding spaces. By learning to differentiate between candidate values and their corresponding field semantics through similarity-based scoring, the invention enables robust generalization across unseen document layouts, varying vendor formats, and diverse document types such as invoices, receipts, purchase orders, and insurance forms. The machine-learned models are designed to process inputs derived from optical character recognition outputs, transforming raw text and bounding box data into interpretable, discriminative representations that capture the nuanced patterns of information organization inherent in structured forms. This approach fundamentally departs from conventional extraction techniques that depend on fixed templates or manual rule authoring, offering a scalable, adaptive, and data-driven solution capable of operating in dynamic business environments where document variety and volume are high.

## BACKGROUND

- introduce templatic documents

Templatic documents constitute a pervasive class of structured records used across commercial, financial, and administrative domains, including invoices, tax forms, insurance claims, purchase orders, and shipping manifests. These documents are characterized by consistent field layouts that recur across multiple instances, even when produced by different entities, and typically contain predefined information such as dates, monetary amounts, identifiers, addresses, and quantities. Despite their structural regularity, the visual presentation of these fields varies significantly between sources, with differences in font, alignment, spacing, and positional arrangement rendering traditional rule-based extraction systems brittle and error-prone. The underlying intent of such documents is to convey a standardized set of data elements, yet the absence of a universal markup language or semantic encoding forces automated systems to infer meaning from visual and textual cues alone.

- describe challenges of extracting information

Extracting accurate information from templatic documents presents substantial challenges due to the lack of consistent syntactic structure, the prevalence of non-textual layout elements such as tables and grid lines, and the frequent use of scanned or rasterized formats that obscure machine-readable metadata. Unlike natural language text, which follows grammatical conventions and sequential coherence, templatic documents often organize information in fragmented, non-linear arrangements, where the meaning of a field depends critically on its proximity to a key phrase or its relative position within a visual hierarchy. Traditional approaches relying on keyword matching, regular expressions, or hand-crafted heuristics fail to generalize beyond known templates and are highly sensitive to minor variations in formatting, OCR errors, or document orientation. Furthermore, the absence of explicit semantic tags or hierarchical markup prevents the application of web-based extraction techniques such as wrapper induction or DOM parsing, which assume structured HTML or XML input.

- limitations of current techniques

Current techniques for information extraction from form-like documents are largely confined to template-matching frameworks that require prior exposure to each document format, or to rule-based systems that depend on manually annotated patterns for each field. These methods are labor-intensive to maintain, scale poorly with increasing document diversity, and exhibit poor robustness when confronted with new vendors, updated layouts, or degraded image quality. Even advanced approaches that incorporate computer vision or deep learning often rely on pixel-level analysis or grid-based representations that are computationally expensive and require high-resolution input, limiting their applicability to low-quality scans or mobile-captured documents. Moreover, many existing models treat extraction as a sequence labeling problem, ignoring the critical role of spatial context and failing to distinguish between semantically similar but contextually distinct candidates—such as an invoice date versus a due date—based on their surrounding textual environment.

- difficulties of extracting information

The fundamental difficulty lies in the ambiguity of candidate values: a single document may contain multiple dates, currency amounts, or identifiers, each of which could plausibly correspond to a target field, yet only one is correct. Distinguishing the true value requires understanding not only the content of the candidate but also its relationship to nearby key phrases, such as “Invoice Date” or “Total Due,” which serve as contextual anchors. These key phrases are often positioned above, below, or to the side of the target value, and their location relative to the candidate is not fixed across documents. Additionally, key phrases themselves exhibit lexical variation—“Date,” “Dated,” “Invoice #,” “PO #”—yet remain semantically aligned with specific fields. Extracting information accurately therefore demands a model capable of integrating textual semantics, spatial geometry, and field-specific prior knowledge into a unified decision framework, a challenge that has remained unsolved by prior methods lacking the capacity to learn generalized, transferable representations of extraction candidates.

## SUMMARY

- introduce computer-implemented method

The present invention introduces a computer-implemented method for extracting structured information from form-like documents by leveraging a machine-learned scoring model that evaluates candidate text portions in relation to target field types. The method operates by first identifying candidate text segments corresponding to predefined field types such as dates, currency amounts, or identifiers, then computing a similarity-based score for each candidate relative to each target field using learned embeddings of both the candidate and the field. The scoring model processes spatial and textual features derived from the document’s optical character recognition output, encoding the candidate’s position, its surrounding neighborhood of text, and the semantic identity of the target field into a joint embedding space. The highest-scoring candidate for each field is selected as the extracted value, enabling accurate, template-independent information extraction without reliance on manual rules or pre-registered document templates.

- describe extracting candidate text portions

The method begins by extracting candidate text portions from a document image using a type-specific candidate generator that identifies all occurrences of predefined data types such as dates, numbers, currency values, email addresses, and alphanumeric identifiers. Each candidate is associated with its textual content and its precise two-dimensional position within the document, derived from bounding box coordinates normalized to the document’s page dimensions. These candidates are generated independently of any specific field, ensuring that all plausible instances of a given type are considered for every field of the same type, thereby maximizing recall while deferring disambiguation to the subsequent scoring stage.

- generate input feature vectors

For each candidate, the method generates a multi-dimensional input feature vector that encodes the candidate’s absolute position, the relative positions of its neighboring text portions, and the textual content of those neighbors. The neighborhood is defined as a region extending upward and to the left of the candidate, encompassing all text portions whose bounding boxes overlap with a predefined zone. Each neighbor’s text is tokenized and mapped to a learned word embedding, while its relative position is encoded using a nonlinear positional embedding that captures fine-grained spatial distinctions such as alignment on the same line versus adjacent lines. The candidate’s own position is embedded via a linear transformation, and the target field type is represented using a separate embedding table that maps each field to a dense vector.

- determine scores and assign text portions

The input feature vector is processed by a neural scoring model that computes a similarity score between the candidate’s encoded representation and the field’s learned embedding. This score reflects the likelihood that the candidate corresponds to the target field, based on learned patterns of co-occurrence and spatial association. The model employs self-attention mechanisms to dynamically weight the influence of each neighbor based on its relevance to the candidate, producing a context-aware neighborhood encoding. The final score is derived from the cosine similarity between the candidate encoding and the field encoding, rescaled to a probability value between zero and one. Candidates are then assigned to their respective field types according to the highest computed score, with only one candidate selected per field, enabling accurate, end-to-end extraction of structured data from previously unseen document formats.

## DETAILED DESCRIPTION

- introduce system for extracting information from form-like documents

The invention comprises a system for extracting structured information from form-like documents that operates through a sequence of interconnected components designed to process document images, generate candidate text portions, compute field-specific similarity scores, and assign extracted values with high accuracy. The system is capable of ingesting both native digital documents and scanned images, converting them into a uniform representation of text and spatial coordinates via optical character recognition. It does not require prior knowledge of document templates or manual annotation of field locations, instead learning generalizable patterns of information organization from a small set of labeled examples. The architecture is modular, allowing for easy adaptation to new document types and target schemas without re-engineering the underlying extraction logic.

- describe end-to-end trainable system using machine learning models

The system is fully end-to-end trainable, employing neural networks that learn to map raw document inputs to structured outputs through supervised optimization. All components—including candidate generation, feature encoding, neighborhood aggregation, and scoring—are integrated into a single trainable framework that minimizes a binary cross-entropy loss between predicted and ground-truth field assignments. The model learns to distinguish between correct and incorrect extractions by identifying latent patterns in the spatial and textual relationships between candidates and their associated key phrases, enabling it to generalize to document layouts not encountered during training. Training is performed using a diverse corpus of annotated documents, with the model’s parameters updated iteratively to maximize extraction accuracy across unseen templates.

- explain robustness to native digital documents and scanned images

The system exhibits robust performance across both native digital documents and scanned images, as it operates on the output of a standard optical character recognition service, which abstracts away differences in document origin. Whether the input is a PDF generated by accounting software or a low-resolution photograph of a paper invoice, the system receives the same sequence of text tokens and bounding boxes, ensuring consistent behavior regardless of input quality. The use of normalized coordinates and embedding-based representations further enhances robustness by decoupling the model from pixel-level details, making it resilient to variations in resolution, orientation, lighting, and compression artifacts.

- describe machine learning model learning dense representation for extraction candidates

The core of the invention is a machine learning model that learns a dense, low-dimensional representation for each extraction candidate, capturing both its textual neighborhood and spatial context in a manner that is invariant to document layout. This representation is constructed by encoding the candidate’s position, its surrounding text tokens, and the relative positions of those tokens into a unified embedding space. Through the use of self-attention mechanisms, the model dynamically adjusts the influence of each neighbor based on its relevance to the candidate, effectively filtering out irrelevant or misleading text. The resulting candidate encoding is designed to cluster with other candidates that belong to the same field, even when those candidates appear in vastly different document formats.

- explain desirable property of positive and negative examples forming separable clusters

A key desirable property of the learned representations is that positive examples—candidates correctly assigned to a field—form tightly clustered regions in the embedding space, while negative examples—incorrect candidates—are distributed more sparsely and remain distant from the field’s learned embedding. This clustering behavior emerges naturally from the training objective, which encourages the model to minimize the distance between a field and its true candidates while maximizing the distance to all other candidates. As a result, the model implicitly learns the semantic and spatial signatures of each field, enabling accurate discrimination even when key phrases are ambiguous or partially obscured.

- describe generating scores for candidates relative to field types in target schema

The system generates a score for each candidate relative to each field type specified in a target schema, using a similarity metric computed between the candidate’s encoded representation and the field’s learned embedding. These scores are computed independently for each candidate-field pair, allowing the model to evaluate all possible assignments simultaneously. The target schema defines the expected fields for a given document type—for example, invoice_date, total_amount, and vendor_name—and the system generates a score for every candidate of the corresponding type against every field in the schema. This enables the system to resolve ambiguities by selecting the candidate with the highest score for each field, even when multiple candidates of the same type exist.

- explain assigning candidates to field types based on scores

Candidates are assigned to field types based on the highest computed score for each field, with only one candidate permitted per field. This assignment is deterministic and does not require post-processing heuristics or conflict resolution rules. The scoring mechanism inherently resolves ambiguity by leveraging the learned relationships between candidates and fields, ensuring that the most contextually appropriate candidate is selected even in the presence of competing values. The system guarantees that no field is left unassigned if a plausible candidate exists, and that no field receives multiple assignments, thereby producing clean, structured output suitable for downstream automation.

- describe using extracted information for automated actions

The extracted information is used to trigger automated business actions such as invoice processing, payment scheduling, tax reporting, and inventory reconciliation. Once values are assigned to their respective fields, they are transmitted to enterprise systems such as accounting software, procurement platforms, or workflow engines, where they are validated, stored, and acted upon without human intervention. This enables organizations to reduce manual data entry, minimize errors, accelerate transaction cycles, and improve compliance by ensuring consistent and accurate data capture across thousands of documents.

- motivate workflows for business processes including form-like documents

Form-like documents are central to numerous business workflows, including accounts payable, human resources, logistics, and regulatory reporting. The manual processing of these documents is costly, time-consuming, and prone to error, particularly in organizations that receive high volumes from diverse vendors. Automating the extraction of information from these documents reduces operational overhead, accelerates decision-making, and improves auditability by creating a consistent, traceable record of data capture. The invention enables this automation at scale, without requiring organizations to maintain custom extraction rules for each vendor or document format.

- explain expense and time reduction through automated processing

By eliminating the need for manual data entry and reducing reliance on brittle rule-based systems, the invention significantly reduces both the financial expense and temporal cost associated with document processing. Organizations that previously required dedicated staff to review and input data from hundreds of invoices per day can now deploy the system to process thousands of documents with minimal supervision. The reduction in labor costs is complemented by a decrease in processing latency, enabling faster payments, improved cash flow, and enhanced supplier relationships.

- describe document analysis system identifying document types

The invention includes a document analysis system that identifies the type of a received document—such as invoice, receipt, or purchase order—by analyzing its layout, textual content, and structural patterns. This identification step determines which target schema should be applied for extraction, ensuring that the correct set of fields is evaluated for each document. The system is capable of recognizing document types even when they lack explicit headers or metadata, relying instead on learned patterns derived from training data.

- explain associating document types with target schemas

Each identified document type is associated with a predefined target schema that specifies the expected fields and their corresponding data types. For example, an invoice schema may include fields such as invoice_number, invoice_date, total_amount, and vendor_address, while a receipt schema may include date, total, and merchant_name. The association between document type and schema is stored in a lookup table and dynamically applied during processing, allowing the system to adapt its extraction behavior to the nature of the incoming document.

- describe target schema including expected fields and information

The target schema defines the set of fields that must be extracted from a document, along with their expected data types and semantic roles. Each field is associated with a type such as date, currency, integer, or alphanumeric string, which guides the candidate generation process. The schema is designed to reflect the information requirements of downstream systems and is configurable to meet the needs of different business units or regulatory jurisdictions.

- explain receiving image of document and determining document type

The system receives a digital image of a document—whether scanned, photographed, or rendered from a digital file—and applies a document classification model to determine its type. This classification is based on learned features extracted from the document’s layout, text density, and common field patterns, enabling the system to route the document to the appropriate extraction pipeline without user input.

- describe analyzing image to extract text portions

The image is processed using an optical character recognition engine that outputs a hierarchical representation of text, including individual characters, words, lines, and blocks, each associated with a bounding box in the document’s coordinate space. These text portions are analyzed to determine their content, position, and relative location to other portions, forming the basis for candidate generation and subsequent scoring.

- explain extracting data from text portions including content and location

The system extracts both the textual content and spatial coordinates of each recognized text portion, normalizing the coordinates to the document’s page dimensions to ensure device- and resolution-independence. This data is used to construct candidate sets for each field type and to compute the positional features required for the scoring model.

- describe determining field types expected in document based on target schema

Once the document type is identified, the system retrieves the corresponding target schema and identifies all field types that must be extracted. For each field type, the system invokes the appropriate candidate generator to identify all text portions in the document that match the expected data type.

- explain determining candidate text portions for each field type

For each field type in the target schema, the system determines a set of candidate text portions by applying type-specific detectors that identify all instances of dates, currency amounts, integers, and other predefined types. These candidates are not yet assigned to fields; they are simply potential values that may correspond to any field of the same type.

- describe analyzing text portions to determine content type

Each candidate text portion is analyzed to determine its content type, such as whether it represents a date, a currency value, or an alphanumeric identifier. This analysis is performed using pre-trained detectors that rely on regular expressions, statistical models, or neural classifiers trained on labeled datasets of common data types.

- explain generating scores for candidate text portions using machine-learned model

Each candidate text portion is evaluated by a machine-learned scoring model that computes a similarity score between the candidate’s encoded representation and the embedding of each target field. The model takes as input the candidate’s position, its neighborhood of surrounding text, and the identity of the target field, producing a score that reflects the likelihood of correct assignment.

- describe machine-learned model taking field type and position information as input

The machine-learned model is designed to accept as input both the field type and the spatial position of the candidate, encoding each into separate embedding spaces. The field type is embedded using a lookup table that maps each field name to a dense vector, while the candidate’s position is encoded using a linear transformation of its normalized coordinates.

- explain generating embeddings for field type and candidate text portion

The model generates a field embedding by retrieving the pre-trained vector associated with the target field name, and a candidate embedding by combining the embedded representations of the candidate’s position, its surrounding text tokens, and the relative positions of those tokens. These embeddings are then used to compute a similarity score that reflects the match between the candidate and the field.

- describe generating neighborhood candidate position embedding

The model generates a neighborhood candidate position embedding by encoding the relative positions of all text portions within the candidate’s neighborhood zone. Each neighbor’s position is transformed using a nonlinear embedding function that captures fine-grained spatial relationships, such as whether a neighbor is on the same line, above, or diagonally adjacent.

- explain using self-attention layers to update neighbor encodings

Self-attention layers are employed to allow each neighbor’s encoding to be updated based on its relationship with all other neighbors in the neighborhood. This enables the model to downweight irrelevant or misleading neighbors and emphasize those that are most indicative of the correct field assignment, such as a key phrase located near the candidate.

- describe combining candidate position embedding and neighborhood embedding

The candidate position embedding and the aggregated neighborhood embedding are concatenated and passed through a projection layer to produce a unified candidate encoding. This encoding encapsulates both the candidate’s location and its contextual environment, enabling the model to distinguish between candidates that may have identical text but different spatial contexts.

- explain generating overall score for candidate text portion

The overall score for the candidate text portion is generated by computing the cosine similarity between the candidate encoding and the field embedding, followed by a linear rescaling to produce a probability value between zero and one. This score represents the model’s confidence that the candidate corresponds to the target field.

- describe selecting candidate text portion based on generated scores

The candidate text portion with the highest score for each field is selected as the extracted value. This selection is made independently for each field, ensuring that the system assigns at most one value per field and avoids conflicts between overlapping candidates.

- explain transmitting selected values to central server for use and analysis

The selected values are transmitted to a central server or enterprise system where they are validated, stored in a database, and used to trigger downstream business processes such as payment initiation, inventory updates, or tax filing. The system may also log the extraction results for auditing, model retraining, or performance monitoring.

- describe three general principles informing document analysis system organization

The design of the document analysis system is informed by three general principles. First, each field corresponds to a well-understood data type, allowing candidate generation to be restricted to plausible values. Second, each field instance is associated with a key phrase that provides contextual guidance, enabling the model to distinguish between semantically similar candidates. Third, the vocabulary of key phrases is drawn from a small, field-specific set of terms, making the problem tractable with limited training data.

- describe pipeline stages including document ingestion, text recognition, and candidate generation

The system operates through a pipeline comprising three primary stages: document ingestion, text recognition, and candidate generation. In the ingestion stage, the document image is received and preprocessed. In the text recognition stage, optical character recognition is applied to extract text and bounding boxes. In the candidate generation stage, type-specific detectors identify all potential values for each field type.

- explain score generation stage and assigning candidate text portion to field

In the score generation stage, the machine-learned model computes a similarity score for each candidate-field pair. In the assignment stage, the highest-scoring candidate is selected for each field, producing a final set of extracted values.

- define scorer system

The scorer system is the core component of the invention, responsible for computing the similarity scores between candidate text portions and target field types. It is implemented as a neural network that learns to map input features—candidate position, neighborhood text, and field identity—into a joint embedding space where similarity reflects correctness.

- describe scorer system functionality

The scorer system receives as input a candidate and a target field, generates embeddings for each, computes a neighborhood encoding using self-attention, and produces a scalar score indicating the likelihood of correct assignment. It is trained using binary cross-entropy loss over labeled examples and optimized to maximize extraction accuracy.

- determine features associated with candidate text portion

The features associated with a candidate text portion include its absolute position, the textual content of its neighbors, the relative positions of those neighbors, and the identity of the target field. These features are encoded into embeddings and processed by the neural model.

- define neighborhood zone

The neighborhood zone is a spatial region extending upward and to the left of the candidate, defined by a fixed height and width relative to the document page. Any text portion whose bounding box overlaps with this zone is considered a neighbor.

- identify nearby text portions

Nearby text portions are identified by computing the intersection between their bounding boxes and the neighborhood zone. Only those with more than half overlap are included as neighbors.

- encode neighbor text portions

Each neighbor’s text is tokenized and mapped to a learned word embedding, while its relative position is encoded using a nonlinear positional embedding. These embeddings are combined into a vector representation for each neighbor.

- represent position of candidate text portion

The position of the candidate text portion is represented as a pair of normalized coordinates (x, y), indicating the centroid of its bounding box relative to the document page dimensions.

- calculate relative position of neighbor text portion

The relative position of a neighbor is calculated as the difference between its normalized coordinates and those of the candidate, producing a vector that encodes direction and distance.

- embed information associated with inputs

All input information—including candidate position, neighbor text, neighbor position, and field identity—is embedded into dense vector representations using learned embedding tables and transformation layers.

- generate intermediate representation of each input

Each input is transformed into an intermediate representation through linear and nonlinear layers, producing embeddings that capture both semantic and spatial characteristics.

- employ embedding table for field

An embedding table maps each field name to a unique dense vector, allowing the model to learn a distinct representation for each field type.

- generate initial neighbor embeddings

Initial neighbor embeddings are generated by combining the word embedding and positional embedding of each neighbor, resulting in a 2d-dimensional vector for each.

- transform initial neighbor embeddings

The initial neighbor embeddings are transformed using linear projection matrices to produce query, key, and value vectors for self-attention computation.

- obtain attention weight vector

An attention weight vector is obtained for each neighbor by computing the dot product between its query vector and all key vectors, followed by softmax normalization.

- encode neighbor text portions using self-attention

Each neighbor’s encoding is updated by computing a weighted sum of all neighbor value vectors, using the attention weights to determine influence.

- project self-attended neighbor encodings

The self-attended neighbor encodings are projected into a higher-dimensional space using a ReLU-activated linear layer, then projected back to the original dimension.

- form single encoding by combining neighbor encodings

The individual neighbor encodings are combined using max-pooling to produce a single, invariant neighborhood encoding that summarizes the contextual information surrounding the candidate.

- obtain candidate encoding

The candidate encoding is obtained by concatenating the neighborhood encoding with the candidate position embedding and applying a linear projection to reduce dimensionality.

- generate score for candidate text portion

The score for the candidate text portion is generated by computing the cosine similarity between the candidate encoding and the field embedding, then rescaling the result to lie within the range [0,1].

- compute cosine similarity

Cosine similarity is computed as the dot product of the candidate encoding and the field embedding, normalized by their respective L2 norms.

- rescale cosine similarity to generate score

The cosine similarity is rescaled using a linear transformation to ensure that the output score lies within the range [0,1], making it interpretable as a probability.

- train scorer system using binary cross-entropy

The scorer system is trained using binary cross-entropy loss, where the target label is 1 if the candidate matches the ground truth for the field and 0 otherwise. The model is optimized using the Rectified Adam optimizer with a learning rate of 0.001.

- describe technical effects and benefits

The technical effects of the invention include the ability to extract structured information from form-like documents with high accuracy across unseen templates, reduced dependency on manual rule authoring, and improved scalability in high-volume document processing environments. The benefits include significant reductions in labor costs, faster processing times, fewer errors, and enhanced compliance through consistent data capture.

- introduce document analysis system

The document analysis system is a comprehensive framework for automated information extraction from templatic documents, comprising components for document ingestion, text recognition, candidate generation, scoring, and assignment.

- describe components of document analysis system

The system comprises a document ingestion module, an optical character recognition engine, a candidate generator, a scorer system, and an assignment engine. Each component is designed to operate in sequence, with outputs from one serving as inputs to the next.

- describe candidate generation system

The candidate generation system identifies all text portions in a document that match predefined data types, producing a set of potential values for each field type without assigning them to specific fields.

- determine target schema

The target schema is determined based on the identified document type and contains a list of expected fields and their associated data types.

- extract candidate text portions

Candidate text portions are extracted using type-specific detectors that identify dates, currency amounts, numbers, and other structured data elements.

- analyze content of text portions

The content of each text portion is analyzed to determine its data type, ensuring that only plausible candidates are considered for each field.

- generate score for each candidate text portion

Each candidate text portion is scored against each target field using the machine-learned scoring model, producing a likelihood value for each candidate-field pair.

- use machine-learned model to generate score

The machine-learned model generates the score by comparing the candidate’s encoded representation with the field’s embedding, using cosine similarity and self-attention to capture contextual relationships.

- generate embeddings for input data

Embeddings are generated for the candidate’s position, its neighborhood of text, and the target field, each using learned transformation functions.

- compare embeddings to generate scores

The candidate embedding and field embedding are compared using cosine similarity to generate a score that reflects their semantic and spatial alignment.

- generate field type embedding

The field type embedding is retrieved from a learned embedding table that maps each field name to a dense vector representation.

- generate candidate position embedding

The candidate position embedding is generated by applying a linear transformation to the normalized coordinates of the candidate’s bounding box centroid.

- generate neighborhood candidate position embedding

The neighborhood candidate position embedding is generated by encoding the relative positions of all neighbors using a nonlinear positional embedding function.

- use self-attention layers to obtain word embeddings

Self-attention layers are applied to the neighbor embeddings to allow each neighbor’s representation to be updated based on its relationship with all other neighbors.

- generate attention weight vector

An attention weight vector is generated for each neighbor by computing the softmax-normalized dot product between its query vector and all key vectors.

- combine candidate position embedding and neighborhood embedding

The candidate position embedding and the max-pooled neighborhood embedding are concatenated and passed through a linear projection layer to form the final candidate encoding.

- compare candidate encoding to field encoding

The candidate encoding is compared to the field encoding using cosine similarity to determine their degree of alignment.

- select candidate text portion based on score

The candidate with the highest score for each field is selected as the extracted value.

- assign candidate text portion to field type

Each selected candidate is assigned to its corresponding field type, producing a structured output record.

- store assigned values for later use

The assigned values are stored in a structured database or transmitted to an enterprise system for further processing.

- transmit data to second computing system

The extracted data is transmitted to a second computing system, such as an accounting platform or workflow engine, where it triggers automated actions.

- map candidate text portions to actions

Each extracted value is mapped to a specific action in a business workflow, such as initiating a payment, updating an inventory record, or generating a tax report.

- describe system for extracting text information

The system for extracting text information is a fully automated, machine-learned pipeline that transforms document images into structured data without human intervention.

- divide document analysis process into components

The document analysis process is divided into four components: document ingestion, text extraction, candidate selection, and scoring and assignment.

- describe text extraction system

The text extraction system receives a document image and outputs a list of text portions with their content and bounding box coordinates.

- extract characters from document

Characters are extracted from the document using an optical character recognition engine that identifies individual glyphs and their positions.

- group characters into text portions

Characters are grouped into words, lines, and blocks based on spatial proximity and reading order.

- determine content of each text portion

The content of each text portion is determined by concatenating the recognized characters in reading order.

- determine position of each text portion

The position of each text portion is determined by computing the centroid of its bounding box and normalizing it to the document page dimensions.

- determine relative position of text portions

The relative position of text portions is computed as the difference between their normalized coordinates, enabling the model to understand spatial relationships.

- describe candidate selection system

The candidate selection system identifies all text portions that match the data type of each target field, producing a set of potential values for each field.

- determine target schema for document

The target schema is determined by matching the document type to a predefined schema in a lookup table.

- identify candidate text portions for each field

Candidate text portions are identified for each field by applying type-specific detectors to the extracted text.

- use scoring model to generate score for each candidate

Each candidate is scored against each field using the machine-learned scoring model, which computes a similarity between the candidate and field embeddings.

- assign candidate text portion to field based on score

The candidate with the highest score for each field is assigned to that field, producing a final set of extracted values.

- describe score model

The score model is a neural network that computes a similarity score between a candidate and a field based on their embeddings and neighborhood context.

- generate features associated with candidate text portion

Features include the candidate’s position, the text of its neighbors, and the relative positions of those neighbors.

- encode neighbor text portions using vocabulary

Neighbor text portions are encoded using a learned vocabulary of the most frequent tokens, with out-of-vocabulary tokens mapped to a special token.

- define score model

The score model is defined as a neural architecture that embeds the candidate, its neighborhood, and the target field, then computes a cosine similarity between the candidate and field encodings.

- transform initial neighbor embeddings

Initial neighbor embeddings are transformed using linear projection matrices to produce query, key, and value vectors for self-attention.

- obtain attention weight vector

An attention weight vector is obtained by applying softmax to the dot products between query and key vectors.

- encode neighbor text portions

Neighbor text portions are encoded by computing a weighted sum of value vectors using the attention weights.

- project self-attended neighbor encodings

Self-attended neighbor encodings are projected into a higher-dimensional space and then back to the original dimension using ReLU-activated linear layers.

- form neighborhood encoding

The neighborhood encoding is formed by applying max-pooling across all neighbor encodings to produce a single, position-invariant representation.

- obtain candidate encoding

The candidate encoding is obtained by concatenating the neighborhood encoding with the candidate position embedding and applying a linear projection.

- generate score

The score is generated by computing the cosine similarity between the candidate encoding and the field embedding, then rescaling it to [0,1].

- train score model

The score model is trained using binary cross-entropy loss on labeled examples of correct and incorrect candidate-field assignments.

- depict example document

An example document is depicted as a scanned invoice with text portions annotated by bounding boxes, showing the location of candidate dates, amounts, and key phrases such as “Invoice Date” and “Total Due.”

- identify position of candidate text portion

The position of each candidate text portion is identified by its normalized centroid coordinates within the document page.

- identify text of neighbor text portions

The text of neighbor text portions is identified by extracting the recognized words adjacent to the candidate within the neighborhood zone.

- generate score

The score is generated by the model based on the alignment between the candidate’s encoding and the field’s embedding.

- depict example process for identifying neighbor text portions

An example process is depicted showing the neighborhood zone extending upward and to the left of a candidate, with overlapping text portions marked as neighbors.

- identify center position of candidate text portion

The center position of the candidate text portion is identified as the centroid of its bounding box.

- identify text portions as neighboring

Text portions are identified as neighboring if their bounding boxes overlap with the neighborhood zone by more than half.

- depict block diagram of multi-step document analysis model

A block diagram depicts the sequential flow from document image to extracted values, showing the text extraction module, candidate generation module, scoring model, and assignment engine.

- describe text extraction model

The text extraction model is implemented as an optical character recognition service that outputs text and bounding boxes from document images.

- describe candidate selection model

The candidate selection model identifies all text portions matching predefined data types for each field in the target schema.

- train machine-learned models

Machine-learned models are trained using labeled datasets of documents with ground-truth field assignments, optimized to minimize extraction error.

- depict flowchart illustrating example method for extracting text

A flowchart illustrates the method steps: receiving a document image, extracting text portions, generating candidates, computing scores using the scoring model, assigning candidates to fields, and transmitting results.

- obtain image of document

The method begins by obtaining a digital image of a document, whether scanned or rendered from a native file format.

- extract candidate text portions

Candidate text portions are extracted by applying type-specific detectors to the recognized text.

- generate input feature vector

An input feature vector is generated for each candidate, encoding its position, neighborhood text, and neighbor positions.

- process input feature vector using machine-learned scoring model

The input feature vector is processed by the machine-learned scoring model to compute a similarity score between the candidate and each target field.

- determine score for each candidate text portion

A score is determined for each candidate text portion relative to each field type in the target schema.

- assign candidate text portions to field type

Each candidate text portion is assigned to the field type with which it achieves the highest score.

- map assigned candidate text portions to action

The assigned values are mapped to corresponding actions in a business workflow, such as payment initiation or record update.

- describe variations of embodiments

Variations of the embodiment include the use of different embedding dimensions, alternative pooling strategies, inclusion of additional features such as font size or text color, and adaptation to new document types through transfer learning. The system may be deployed on-premises or in the cloud, and may be integrated with enterprise resource planning systems, accounting software, or robotic process automation platforms.