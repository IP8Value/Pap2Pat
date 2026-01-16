Here is the complete patent application following your outline and guidelines:

# DESCRIPTION  

## FIELD  

The present invention relates generally to automated information extraction from form-like documents, and more particularly to systems and methods for learning representations of extraction candidates to accurately identify field values across diverse document templates. The disclosed techniques are applicable to processing various structured documents including but not limited to invoices, purchase orders, tax forms, insurance quotes, receipts, and other templatic documents where information follows predictable patterns despite layout variations.  

This technological field intersects document processing, computer vision, natural language processing, and machine learning. The invention provides specific technical solutions to longstanding problems in automated form processing, including robust handling of spatial relationships between field values and their identifying key phrases, generalization to unseen document templates, and meaningful representation learning for extraction candidates. The disclosed methods significantly improve upon conventional template-based approaches and heuristic methods through novel neural network architectures that jointly model textual and spatial features.  

## BACKGROUND  

Current approaches for extracting structured information from form-like documents suffer from several technical limitations. Traditional methods rely heavily on manual template creation or brittle rule-based systems that cannot adapt to layout variations across vendors or document versions. While optical character recognition (OCR) technology can convert scanned documents into machine-readable text, the resulting output lacks semantic understanding of field relationships.  

Prior attempts to automate form processing have employed either natural language processing techniques optimized for prose (which fails to capture critical spatial relationships in forms) or computer vision approaches that treat documents as raw pixels (which is computationally expensive and ignores valuable textual signals). Existing wrapper induction techniques developed for HTML documents cannot be directly applied to PDFs or scanned images where markup information is unavailable.  

Recent hybrid approaches combining text and layout features still face fundamental challenges in generalizing to unseen document templates. These systems typically require extensive per-template training data and fail to leverage common patterns that persist across different form types. There remains an unmet need for systems that can accurately extract field values from novel document layouts after training on limited examples.  

The technical problems addressed by this invention include: (1) how to effectively represent the spatial relationships between field values and their identifying key phrases across varying document layouts; (2) how to learn transferable representations of extraction candidates that capture these relationships; and (3) how to score candidates based on their similarity to learned field representations while maintaining robustness to OCR errors and document noise.  

## SUMMARY  

The present invention provides a novel system and method for extracting structured information from form-like documents using learned representations of extraction candidates. At a high level, the system operates through three main technical components: (1) a candidate generation module that identifies potential field values based on type detectors; (2) a neural scoring model that learns meaningful representations of candidates and fields; and (3) an assignment module that selects the most likely candidate for each target field.  

Key technical innovations include:  

A representation learning framework where the system jointly learns embeddings for extraction candidates (encoding their spatial neighborhoods) and for target fields (encoding their characteristic key phrases). The similarity between these embeddings determines candidate scores, enabling generalization across templates.  

A neural network architecture that processes both textual and spatial features of candidate neighborhoods through specialized embedding layers and self-attention mechanisms. This architecture captures critical spatial relationships between field values and their identifying key phrases while remaining robust to layout variations.  

A complete pipeline that handles document ingestion (including OCR processing), candidate generation using type-specific detectors, neural scoring of candidates, and final assignment of field values. The pipeline is designed to maximize recall during candidate generation while relying on the neural scorer for precision.  

Technical advantages over prior approaches include:  

1. Template-independent operation through learned representations that capture common patterns across document types  
2. Effective combination of textual and spatial signals through novel neural architectures  
3. Interpretable candidate embeddings that cluster meaningfully by field type  
4. Robust performance on both digital and scanned documents  
5. Efficient processing through selective candidate generation and neural scoring  

The system demonstrates particular effectiveness in processing invoices and receipts, where it significantly outperforms baseline methods on field extraction tasks. However, the core techniques are broadly applicable to any domain involving structured information extraction from form-like documents.  

## DETAILED DESCRIPTION  

The following sections provide comprehensive technical details of the invention's components and operation.  

### Document Ingestion and OCR Processing  

The system accepts both native digital documents (PDFs, Word files, etc.) and scanned images as input. For non-image inputs, the system first renders each page to an image representation to normalize processing. An OCR engine then extracts text content along with spatial positioning information.  

The OCR output organizes text hierarchically, with characters grouped into words, words into paragraphs, and paragraphs into blocks. Each element is associated with a bounding box in the 2D coordinate space of the document page. This spatial information is preserved through subsequent processing stages and forms a critical input to the neural scoring model.  

### Candidate Generation  

Candidate generation leverages the observation that most fields in form-like documents correspond to well-defined types (dates, currency amounts, etc.). The system employs specialized detectors for each supported type:  

- Date detectors using pattern matching and validation rules  
- Currency amount detectors recognizing monetary formats  
- Alphanumeric detectors for IDs and codes  
- Address, email, and other specialized detectors as needed  

Each target field in the schema is associated with one or more candidate generators based on its type. For example, all dates in a document become candidates for date-type fields like "invoice_date" and "due_date." The candidate generation phase prioritizes high recall, as the system cannot recover field values missed at this stage.  

### Neural Scoring Model Architecture  

The scoring model represents the core innovation of the system. It takes as input a candidate (including its text and position) and a target field, outputting a score between 0 and 1 representing the likelihood that the candidate is the correct extraction for that field.  

The model architecture comprises several key components:  

**Neighborhood Definition**  
For each candidate, the system defines a neighborhood zone extending leftward and upward in the document (typically covering about 10% of page height). Text tokens overlapping this zone are considered neighbors. The model captures both the text of these neighbors and their relative positions to the candidate.  

**Feature Embedding Layers**  
1. Neighbor text embeddings: Words are embedded using a learned vocabulary table (with special tokens for numbers, rare words, etc.)  
2. Position embeddings: Relative positions are embedded through nonlinear transformations capturing fine spatial relationships  
3. Candidate position embedding: Absolute position is embedded through a linear layer  
4. Field embedding: Each target field has its own learned embedding  

**Self-Attention Mechanism**  
Neighbor embeddings are processed through a self-attention layer that captures interactions between neighbors. This allows the model to identify important key phrases even when they aren't the closest text to the candidate.  

**Candidate Encoding**  
The system combines attended neighbor embeddings through max pooling (preserving the most salient features) and concatenates this with the candidate's position embedding. A projection layer produces the final candidate encoding.  

**Scoring Function**  
The score is computed as the cosine similarity between the candidate encoding and field embedding, rescaled to [0,1]. This approach causes positive examples to cluster around their field embeddings in the representation space.  

### Training and Optimization  

The model is trained as a binary classifier using cross-entropy loss. Positive examples are candidates matching ground truth field values; negatives are other candidates for the same field. Training incorporates several important techniques:  

1. Negative downsampling to balance the dataset (typically keeping ≤40 negatives per positive)  
2. Positional dropout to improve robustness to layout variations  
3. Rectified Adam optimization with careful learning rate selection  
4. Validation-based early stopping  

The model learns to position field embeddings such that they attract positive candidates while repelling negatives (especially positives for other fields). This creates well-separated clusters in the embedding space.  

### Assignment Module  

After scoring all candidates for a document, the system assigns at most one candidate per field. Simple thresholding can be used, or more sophisticated approaches like non-maximum suppression for overlapping candidates. The system can also enforce domain-specific constraints (e.g., that "due_date" must follow "invoice_date").  

### Experimental Results  

Testing on invoice and receipt datasets demonstrated significant improvements over baselines:  

1. 10+ point F1 score improvements compared to bag-of-words and MLP baselines  
2. Robust generalization to unseen document templates  
3. Clear clustering of candidate embeddings by field type  
4. Effective handling of spatial relationships between fields and key phrases  

The system showed particular strength on common fields like dates and amounts, while performance on rare fields improved with additional training data. The self-attention mechanism proved critical, contributing ~1.7 F1 points to overall performance.  

### Implementation Considerations  

The system can be implemented using standard deep learning frameworks (TensorFlow, PyTorch). For production use, several optimizations are available:  

1. Batch processing of documents for efficient GPU utilization  
2. Caching of OCR results for repeated documents  
3. Pre-computation of candidate features where possible  
4. Distributed scoring for high-volume applications  

The technology is particularly valuable for enterprise document processing workflows involving large volumes of invoices, receipts, or similar forms. It reduces manual data entry while providing more accurate and consistent results than rule-based systems.  

This detailed description covers the novel technical aspects of the invention. Further embodiments may extend the core techniques to additional document types, field categories, and use cases while maintaining the essential innovation of representation learning for form extraction.