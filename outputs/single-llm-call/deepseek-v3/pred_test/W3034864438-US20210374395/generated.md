Here is the patent application drafted according to your outline and guidelines:

# DESCRIPTION  

## FIELD  

The present invention relates generally to machine-learned models for information extraction. More specifically, the invention relates to computer-implemented methods and systems for extracting structured information from templatic documents using learned representations of extraction candidates. The disclosed techniques employ machine learning models to analyze both textual content and spatial layout information when processing form-like documents such as invoices, receipts, purchase orders, and other business documents. The system generates dense vector representations for candidate text portions and compares these representations against learned field embeddings to determine the most appropriate field assignments for extracted information.  

## BACKGROUND  

Templatic documents present unique challenges for automated information extraction systems. Unlike natural language text organized in sentences and paragraphs, form-like documents rely heavily on spatial layout elements such as tables, grids, and specific formatting conventions to convey information. Current techniques for processing such documents suffer from several limitations. Traditional approaches that work well on HTML documents or web pages are not directly applicable since templatic documents typically exist as PDFs or scanned images without markup language structure.  

Existing methods for information extraction from form-like documents face significant difficulties in handling document variations. While template-based systems can extract information from known document layouts, they fail to generalize to unseen templates. Recent attempts to combine layout features with text signals either require computationally expensive image processing or rely on heuristics that limit accuracy. The current state of technology lacks robust solutions that can automatically adapt to different document layouts while maintaining high extraction accuracy.  

## SUMMARY  

The present invention introduces a computer-implemented method for extracting information from form-like documents using machine learning techniques. The method involves analyzing an input document to extract candidate text portions corresponding to potential field values in a target schema. For each candidate text portion, the system generates input feature vectors incorporating both textual content and spatial position information.  

A machine-learned scoring model processes these feature vectors to determine scores indicating how well each candidate text portion matches various field types in the target schema. The model learns dense vector representations for both the candidate text portions and the field types themselves. By comparing these representations, the system assigns each candidate text portion to the most appropriate field type based on the generated scores.  

The scoring model employs several innovative techniques including self-attention mechanisms for neighbor text analysis, nonlinear positional embeddings, and specialized pooling operations. These components work together to create robust representations that capture both semantic meaning and spatial relationships within the document. The system outputs structured data by assigning the highest-scoring candidate text portions to their respective fields in the target schema.  

## DETAILED DESCRIPTION  

The document analysis system of the present invention provides an end-to-end trainable solution for extracting information from form-like documents. The system demonstrates particular robustness in handling both native digital documents and scanned images through its innovative use of machine learning techniques. A key aspect of the invention involves the machine learning model learning dense representations for extraction candidates that capture both textual and spatial characteristics.  

The system exhibits the desirable property that positive and negative examples form separable clusters in the learned representation space. This clustering behavior enables effective scoring of candidates relative to field types in the target schema. The assignment of candidates to field types occurs based on these computed scores, with the system selecting the most appropriate matches. The extracted information can then be used to trigger automated actions in downstream business processes.  

The invention specifically addresses workflows involving form-like documents common in business operations. By automating the extraction process, the system significantly reduces both expense and time associated with manual document processing. The document analysis system first identifies document types and associates them with appropriate target schemas. Each target schema includes expected fields and the type of information they should contain.  

When processing a document, the system receives an image representation and determines the document type. Analysis of the image yields extracted text portions with both content and location information. For each field type expected in the document according to the target schema, the system determines candidate text portions that potentially match. The system analyzes these text portions to determine their content type and generates scores using the machine-learned model.  

The machine-learned model takes both field type information and position data as input to generate embeddings. These embeddings include representations for the field type, the candidate text portion position, and neighborhood candidate positions. The system employs self-attention layers to update encodings of neighboring text portions based on their relationships. By combining candidate position embeddings with neighborhood embeddings, the model generates an overall score for each candidate text portion.  

Selection of candidate text portions occurs based on these generated scores, with the system transmitting selected values to central servers for further use and analysis. The organization of the document analysis system follows three general principles that inform its design. First, each field corresponds to a well-understood type with specific characteristics. Second, each field instance associates with a key phrase that indicates its purpose. Third, these key phrases draw from a small, field-specific vocabulary.  

The processing pipeline includes several stages: document ingestion, text recognition, candidate generation, score generation, and field assignment. The scorer system forms a core component of this pipeline, determining features associated with candidate text portions and analyzing their neighborhoods. The system defines neighborhood zones around candidates and identifies nearby text portions for analysis.  

Encoding of neighbor text portions involves representing their positions relative to the candidate and embedding this information through specialized transformations. The system employs embedding tables for field types and generates initial neighbor embeddings that undergo further processing. Through self-attention mechanisms, the system obtains attention weight vectors that help encode neighbor text portions more effectively.  

Projection of self-attended neighbor encodings produces refined representations that combine to form a single encoding for the entire neighborhood. The system obtains a candidate encoding by combining this neighborhood representation with the candidate's own position embedding. Scoring occurs through computation of cosine similarity between candidate encodings and field embeddings, with subsequent rescaling to generate final scores.  

Training of the scorer system utilizes binary cross-entropy loss to optimize performance. The document analysis system provides several technical effects and benefits, including improved accuracy in information extraction and better generalization to unseen document templates. The system components work together to analyze documents, determine target schemas, extract candidate text portions, and generate appropriate scores.  

The machine-learned model generates embeddings for input data and compares these to produce scores. Field type embeddings capture expected characteristics, while candidate position embeddings represent spatial information. Neighborhood candidate position embeddings incorporate context from surrounding text. Self-attention layers process these inputs to obtain meaningful word embeddings that reflect document structure.  

By generating attention weight vectors and combining position embeddings with neighborhood information, the system creates comprehensive representations. Comparison of candidate encodings to field encodings enables accurate scoring and assignment. Selected candidate text portions map to appropriate field types and store for later use or transmission to other computing systems.  

The system divides document analysis into logical components including text extraction and candidate selection. The text extraction system processes documents to identify characters and group them into meaningful text portions. Analysis determines both content and position characteristics for these portions, including their relative placements.  

Candidate selection involves determining the target schema and identifying appropriate candidates for each field. The scoring model evaluates these candidates based on generated features and assigns them to fields according to computed scores. The score model itself generates features associated with candidates and encodes neighbor text portions using a defined vocabulary.  

Transformation of initial neighbor embeddings through attention mechanisms produces refined representations. The system obtains attention weight vectors that guide encoding of neighbor text portions. Projection operations create self-attended neighbor encodings that combine to form neighborhood representations. These combine with candidate encodings to generate final scores through specialized similarity calculations.  

Training of the score model optimizes its ability to distinguish between appropriate and inappropriate field assignments. Example documents demonstrate the system's operation, showing identification of candidate positions and neighbor text portions. The scoring process evaluates these elements to determine the best matches for each field in the target schema.  

Neighbor identification involves determining center positions of candidates and establishing nearby text portions as neighbors. A multi-step document analysis model coordinates these operations through specialized components including text extraction and candidate selection models. Training procedures ensure the machine-learned components operate effectively together.  

The method for extracting text involves obtaining document images, extracting candidate portions, and generating input feature vectors. Processing these vectors through the machine-learned scoring model produces scores that guide assignment of candidates to field types. Mapping of assigned candidates to automated actions completes the information extraction process.  

Variations of these embodiments include alternative approaches to neighbor identification, different scoring mechanisms, and modified training procedures. The system maintains its core functionality while adapting to specific use cases or document types. These variations demonstrate the flexibility of the underlying invention while preserving its innovative approach to information extraction from form-like documents.