# DESCRIPTION

## BACKGROUND

The medical domain is characterized by a vast and rapidly expanding body of knowledge. This knowledge is dispersed across various sources, including textbooks, encyclopedias, guidelines, electronic medical records, and online resources. The sheer volume and complexity of this information pose significant challenges for healthcare professionals, who must stay updated and make informed decisions in clinical practice. Traditional methods of information retrieval and processing are often inadequate to handle the scale and intricacy of medical data.

Recent advancements in information extraction technologies, particularly in the realm of relation extraction, offer promising solutions to these challenges. Relation extraction involves identifying and classifying relationships between entities mentioned in text. In the medical domain, these relationships can be crucial for clinical decision-making, such as determining the appropriate treatment for a disease or identifying the causes of a condition.

However, developing effective relation extraction systems for the medical domain presents several challenges. First, the medical domain is highly specialized, requiring the identification of a comprehensive set of relations that cover the diverse needs of clinical practice. Second, the scale of medical text is enormous, necessitating efficient and scalable relation detection methods. Third, the scarcity of labeled training data, due to the high cost of manual annotation, often leads to overfitting in relation extraction models.

To address these challenges, the present invention introduces a novel method for medical relation extraction that leverages both labeled and unlabeled data, utilizes efficient parsing and classification techniques, and focuses on key relations that are essential for clinical decision-making. This method significantly improves the accuracy and efficiency of relation extraction in the medical domain, thereby enhancing the ability of healthcare professionals to access and utilize critical medical information.

## SUMMARY

The present invention relates to a method and system for extracting medical relations from large-scale text corpora. The method addresses the challenges of identifying a comprehensive set of relations, efficiently detecting relations in large amounts of text, and overcoming the limitations of insufficient labeled training data. Specifically, the invention provides a manifold model that integrates both labeled and unlabeled data to enhance the performance of relation extraction.

The key aspects of the invention include:

1. **Identification of Key Medical Relations**: The invention identifies a set of "super relations" that are crucial for clinical decision-making, such as "treats," "causes," and "contraindicates." These relations are selected based on an analysis of real-world clinical questions and the tasks performed by healthcare professionals.

2. **Efficient Relation Detection**: The invention employs parsing adaptation and the use of linear classifiers to speed up the relation detection process. This allows the method to handle large-scale medical text corpora efficiently.

3. **Utilization of Unlabeled Data**: The invention uses a manifold model that leverages both labeled and unlabeled data to prevent overfitting and improve the robustness of the relation extraction model. The model encourages examples with similar content to be assigned similar scores, thereby respecting the topology of the data manifold.

4. **Training Data Collection**: The invention provides a method for collecting training data for each relation, including a combination of labeled and unlabeled data. The method uses K-medoids clustering to select representative sentences for annotation, minimizing the human labeling effort.

5. **Knowledge Base Construction**: The invention constructs a new medical relation knowledge base by applying the trained relation detectors to a large medical corpus. The knowledge base stores the extracted relations in a structured format, providing a valuable resource for clinical decision support.

The invention significantly outperforms existing state-of-the-art approaches in medical relation extraction, offering faster and more accurate results. The method is particularly useful in scenarios where labeled training data is limited, making it a valuable tool for healthcare professionals and researchers in the medical domain.

## DETAILED DESCRIPTION

### Identification of Key Medical Relations

The first step in building a relation extraction system for the medical domain is to identify the relations that are most relevant for clinical decision-making. Based on an analysis of real-world clinical questions and the tasks performed by healthcare professionals, the invention focuses on seven key relations, referred to as "super relations":

1. **Treats**: Indicates that a particular treatment is effective for a specific disease.
2. **Causes**: Identifies the factors that cause a disease.
3. **Contraindicates**: Specifies that a particular treatment is not suitable for a specific condition.
4. **Symptoms**: Lists the symptoms associated with a disease.
5. **Diagnosis Tests**: Identifies the tests used to diagnose a disease.
6. **Prevents**: Indicates that a particular measure can prevent a disease.
7. **Locations**: Specifies the anatomical locations where a disease or symptom is observed.

These super relations cover the most common clinical tasks, such as therapy selection, diagnosis, etiology, and prognosis. By focusing on these relations, the invention ensures that the extracted information is directly applicable to clinical practice.

### Efficient Relation Detection

To efficiently detect relations in large-scale medical text, the invention employs several techniques:

1. **Parsing Adaptation**: The invention uses a specialized parser, MedicalESG, which is an adaptation of the English Slot Grammar (ESG) to the medical domain. MedicalESG is designed to handle the unique linguistic structures and terminologies found in medical text. It is approximately 10 times faster than MetaMap, a widely used tool for medical entity detection, while producing similar parsing results.

2. **Linear Classifiers**: The invention replaces non-linear classifiers with linear classifiers to speed up the relation detection process. Linear classifiers are computationally efficient and can handle large datasets without significant performance degradation.

3. **Semantic Typing**: The invention uses the semantic types defined in the Unified Medical Language System (UMLS) to categorize relation arguments. Each argument is associated with one or more UMLS semantic types, which provide a consistent categorization of medical concepts. This helps in accurately identifying and classifying the entities involved in the relations.

### Utilization of Unlabeled Data

One of the key innovations of the invention is the use of a manifold model that integrates both labeled and unlabeled data. The manifold model leverages the topology of the data manifold to improve the performance of the relation extraction model. The model encourages examples with similar content to be assigned similar scores, thereby preventing overfitting and enhancing the robustness of the model.

The manifold model is formalized as follows:

Given a dataset \( X = \{x_1, x_2, \ldots, x_m\} \) represented as a feature-instance matrix, and a desired label vector \( Y = \{y_1, y_2, \ldots, y_l\} \) where \( l \leq m \), the goal is to construct a mapping function \( f \) that projects any example \( x_i \) to a new space where \( f^T x_i \) matches \( x_i \)'s desired label \( y_i \). Additionally, the model aims to preserve the manifold topology of the dataset, ensuring that similar examples (both labeled and unlabeled) get similar scores.

The cost function \( C(f) \) is defined as:
\[ C(f) = \sum_{i=1}^{l} \alpha_i (f^T x_i - y_i)^2 + \mu f^T L f \]

Where:
- \( \alpha_i \) is a user-specified parameter representing the weight of label \( y_i \).
- \( \mu \) is a weight scalar.
- \( L \) is the graph Laplacian matrix modeling the data manifold.

The first term of \( C(f) \) penalizes the difference between the mapping result of \( x_i \) and its desired label \( y_i \). The second term encourages the neighborhood relationship within \( X \) to be preserved in the mapping. The solution to the problem is given by:
\[ f = (X(A + \mu L)X^T)^+ X A V^T \]

Where \( (X(A + \mu L)X^T)^+ \) represents the pseudo inverse.

### Training Data Collection

Collecting training data for each relation is a critical step in the relation extraction process. The invention provides a method for collecting both labeled and unlabeled data, minimizing the human labeling effort:

1. **Distant Supervision**: The invention uses distant supervision to collect a large amount of noisy relation data. This involves parsing the medical corpus to identify sentences containing the terms associated with the CUI pairs in the UMLS knowledge base. While this approach results in a large amount of data, it also introduces noise in the form of false positives.

2. **K-Medoids Clustering**: To reduce the noise and minimize the human labeling effort, the invention applies K-medoids clustering to the sentences associated with each super relation. The cluster centers are selected as the most representative sentences for annotation. The number of clusters is chosen based on the number of sentences collected for each relation, typically ranging from 3,000 to 6,000.

3. **Labeling and Annotation**: The selected cluster centers are manually annotated by human experts to assign positive or negative labels. The remaining sentences are held as unlabeled data for further experiments.

4. **Negative Training Set Growth**: To grow the size of the negative training set, the invention adds a small number of the most representative examples from each unrelated UMLS relation to the training set as negative examples. This results in more than 10,000 extra negative examples for each relation.

### Knowledge Base Construction

The invention constructs a new medical relation knowledge base by applying the trained relation detectors to a large medical corpus. The knowledge base stores the extracted relations in a structured format, providing a valuable resource for clinical decision support.

The steps involved in constructing the knowledge base are:

1. **Relation Detection**: The trained relation detectors are applied to the medical corpus to extract relations. Each relation is represented as a tuple (relation name, argument 1, argument 2, confidence), where the confidence is computed based on the relation detector confidence score and the relation popularity in the corpus.

2. **Data Integration**: The extracted relations are combined and stored in the knowledge base. The knowledge base covers all super relations and provides a comprehensive resource for clinical decision-making.

3. **Performance Evaluation**: The new knowledge base is evaluated against existing knowledge bases, such as the UMLS Metathesaurus, using an answer generation task on a set of clinical questions. The results demonstrate that the new knowledge base outperforms existing resources in terms of knowledge coverage and accuracy.

### Experimental Results

The invention was evaluated using a cross-validation test on a dataset of medical relations. The dataset includes seven super relations, and the evaluation was conducted using a 5-fold cross-validation. The F1 scores reported are the average of all five rounds.

The invention was compared against three state-of-the-art approaches:

1. **SVM with Convolution Tree Kernels**: This approach uses tree kernels to capture parse tree structure-related features.
2. **Linear Regression**: This approach uses linear regression to model the relation extraction task.
3. **SVM with Linear Kernels**: This approach uses linear SVM to model the relation extraction task.

The results show that the invention significantly outperforms the baseline approaches in terms of F1 scores. The manifold model, which integrates both labeled and unlabeled data, achieves the best performance, demonstrating the effectiveness of the invention in handling the challenges of medical relation extraction.

### Conclusion

The present invention provides a novel method and system for medical relation extraction that addresses the challenges of identifying key relations, efficiently detecting relations in large-scale text, and overcoming the limitations of insufficient labeled training data. The invention leverages a manifold model that integrates both labeled and unlabeled data, resulting in faster and more accurate relation extraction. The constructed knowledge base serves as a valuable resource for clinical decision support, enhancing the ability of healthcare professionals to access and utilize critical medical information.