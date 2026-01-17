# DESCRIPTION

## FIELD

The field of the present invention pertains to the automated extraction of structured information from form-like documents, such as invoices, purchase orders, tax forms, and insurance quotes. Specifically, the invention involves a method and system for learning a representation of extraction candidates to accurately identify and extract target fields from such documents, even when the documents belong to unseen templates.

## BACKGROUND

Form-like documents are ubiquitous in business workflows, yet the current methods for processing them often rely on manual effort or brittle heuristics, which are error-prone and inefficient. Traditional approaches to information extraction, such as those designed for natural text, are not directly applicable to form-like documents due to their reliance on spatial layout elements and the lack of explicit markup. Recent advancements in natural language processing (NLP) and computer vision have shown promise, but they often struggle to generalize to unseen templates and fail to leverage the unique characteristics of form-like documents.

The primary challenge in extracting structured information from form-like documents is the variability in layout and presentation. Different vendors may present the same information in diverse ways, making it difficult to apply a single extraction rule. Additionally, the spatial relationships between fields and their key phrases are critical for accurate extraction, but these relationships are not always straightforward and can vary across templates.

To address these challenges, the present invention introduces a novel approach based on representation learning. This approach generates extraction candidates for each target field using their associated types and then uses a neural network model to learn dense representations for each candidate. The similarity between the candidate and field representations is used to score the candidate, enabling the system to generalize to unseen templates and achieve high extraction accuracy.

## SUMMARY

The present invention provides a method and system for extracting structured information from form-like documents using representation learning. The method comprises the following steps:

1. **Ingestion**: The system ingests both native digital and scanned documents, rendering them to images and using optical character recognition (OCR) to extract all text and associated bounding boxes.

2. **Candidate Generation**: For each target field in the schema, the system generates extraction candidates using a library of detectors for common types such as dates, currency amounts, integers, addresses, and email addresses. These detectors are designed to have high recall to ensure that potential candidates are not missed.

3. **Scoring and Assignment**: The system scores each candidate using a neural scoring model that learns separate embeddings for the candidate and the field. The similarity between these embeddings determines the score, which is used to select the most likely candidate for each field.

The neural scoring model is designed to learn meaningful candidate representations by incorporating spatial and textual information from the candidate's neighborhood. The model uses self-attention mechanisms to capture interactions between neighboring text tokens and their positions, ensuring that the learned representations are robust and generalizable.

The invention further includes a detailed description of the neural scoring model, including the candidate features, embeddings, and the scoring mechanism. The model is trained using a binary cross-entropy loss function and evaluated on datasets from different domains, such as invoices and receipts, to demonstrate its effectiveness in generalizing to unseen templates.

## DETAILED DESCRIPTION

### Ingestion

The system is capable of ingesting both native digital documents and scanned images. Each document is rendered to an image, and a cloud-based OCR service is used to extract all the text in the document. The OCR result is arranged in a hierarchical structure, with individual characters at the leaf level and words, paragraphs, and blocks at higher levels. Each node in the hierarchy is associated with bounding boxes represented in the 2D Cartesian plane of the document page. The words in a paragraph are arranged in reading order, as are the paragraphs and blocks themselves.

### Candidate Generation

The system generates extraction candidates for each target field in the schema using the field type. For example, all dates in an invoice become candidates for date fields such as invoice_date, due_date, and delivery_date. The candidate generators are designed to have high recall to ensure that potential candidates are not missed. Each field type is associated with one or more candidate generators, which use a cloud-based entity extraction service to detect spans of the OCR text that are instances of the corresponding type.

### Scoring and Assignment

The core of the extraction system is the neural scoring model, which takes as input the target field and the extraction candidate and produces a prediction score in the range [0, 1]. The model is trained as a binary classifier, with the target label determined by whether the candidate matches the ground truth for that document and field.

#### Candidate Features

The model learns a representation of a candidate that captures its neighborhood. The essential features of a candidate include the text tokens that appear nearby, along with their positions. A neighborhood zone is defined around the candidate, extending all the way to the left of the page and about 10% of the page height above it. Any text tokens whose bounding boxes overlap by more than half with the neighborhood zone are considered neighbors. The position of the candidate and each of its neighbors is represented using the 2D Cartesian coordinates of the centroids of their respective bounding boxes, normalized by dividing by the corresponding page dimensions. The relative position of a neighbor is calculated as the difference between its normalized 2D coordinates and those of the candidate. An additional feature is the absolute position of the candidate itself.

#### Embeddings

Each of the candidate features is embedded separately. The neighboring text tokens are embedded using a word embedding table. Each neighbor's relative position is embedded through a nonlinear positional embedding consisting of two ReLU-activated layers with dropout. The candidate position feature is embedded using a linear layer. The field to which a candidate belongs is also embedded using an embedding table.

The neighbor embeddings are combined using self-attention mechanisms to capture interactions between neighbors. The self-attended neighbor encodings are then projected to a larger dimensional space and back to the original dimension. The neighborhood encoding is obtained by max-pooling the neighbor encodings, ensuring that the encoding is invariant to the order of the neighbors.

The candidate encoding is formed by concatenating the neighborhood encoding with the candidate position embedding and projecting it back down to the original dimension using a ReLU-activated linear layer.

#### Candidate Scoring

The candidate encoding is designed to contain all relevant information about the candidate, including its position and its neighborhood. The model is trained as a binary classifier to score a candidate according to how likely it is to be the true extraction value for some field and document. The final score is a linear rescaling of the cosine similarity between the candidate encoding and the field embedding, normalized to the range [0, 1].

### Datasets

The performance of the model was evaluated using datasets from two different domains: invoices and receipts. The invoice dataset consists of two corpora, Invoices1 and Invoices2, with different templates and fields. The receipt dataset is a publicly available corpus of scanned receipts with ground truth extraction results for fields such as address, company, date, and total.

### Experiments

The model was trained using the Rectified Adam optimizer with a learning rate of 0.001 for 50 epochs. The training, validation, and test splits were used to train the model, select the best model based on validation performance, and report performance metrics, respectively.

The model was compared to two baseline models: a bag-of-words (BoW) baseline and a multilayer perceptron (MLP) baseline. The BoW baseline incorporates only the neighboring tokens of a candidate, while the MLP baseline uses the same input features as the proposed model, including the relative positions of the candidate's neighbors. Both baselines follow the representation learning approach, encoding the candidate and the field separately and using the cosine distance between the candidate and field encodings as the final score.

The results show that the proposed model significantly outperforms the baselines in both scorer ROC AUC and end-to-end maximum F1 score. The improvement is particularly notable for fields that occur frequently in invoices and have a small number of negatives for each positive. The model also performs well on the receipt dataset, demonstrating its ability to generalize to unseen templates.

### Meaningful Internal Representations

The internal representations learned by the model were visualized using t-SNE. The representations of date candidates were colored based on their ground truth labels, showing distinct clusters for different fields such as invoice_date, due_date, and delivery_date. The field embeddings were found to lie close to the clusters of positive examples for their respective fields, demonstrating the model's ability to learn meaningful and interpretable representations.

### Conclusion and Future Work

The present invention provides a novel and effective method for extracting structured information from form-like documents using representation learning. The system is capable of generalizing to unseen templates and achieving high extraction accuracy. Future work will focus on extending the system to handle repeated fields and developing domain-specific candidate generators. Additionally, the learned candidate representations will be explored for transfer learning to new domains and few-shot settings.