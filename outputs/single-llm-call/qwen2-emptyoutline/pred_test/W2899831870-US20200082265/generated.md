# DESCRIPTION

## TECHNICAL FIELD

The present disclosure relates to the field of machine learning, particularly to methods and systems for learning deep graph representations. More specifically, the invention pertains to a framework for inductive graph representation learning that generalizes for across-network transfer learning tasks. The disclosed framework, referred to as DeepGL, is designed to overcome the limitations of existing node embedding methods by learning relational functions that can be applied to any arbitrary graph, thereby enabling inductive transfer learning.

## BACKGROUND

Machine learning tasks, such as node and link classification, anomaly detection, link prediction, dynamic network analysis, community detection, role discovery, visualization, network alignment, and graph similarity, heavily rely on effective data representation. Traditional methods often require manual feature engineering, which is labor-intensive and may not generalize well across different datasets. Recent advancements in graph representation learning, such as DeepWalk and node2vec, have automated the process of learning node embeddings by leveraging random walks and the skip-gram model. However, these methods have several limitations:

1. **Non-Inductive Nature**: Existing methods learn node embeddings that are specific to the graph they are trained on and do not generalize to other networks, limiting their applicability for transfer learning tasks.
2. **Dense Representations**: The learned embeddings are typically dense, consuming significant memory resources, especially for large graphs.
3. **Lack of Interpretability**: The features learned by these methods are often difficult to interpret, which is increasingly important in practical applications.
4. **Inefficiency**: These methods can be computationally expensive, with runtimes that are orders of magnitude slower compared to more efficient algorithms.

To address these limitations, the present invention introduces DeepGL, a novel framework for inductive graph representation learning. DeepGL overcomes the aforementioned challenges by learning relational functions that generalize across different graphs, supporting inductive transfer learning tasks. Additionally, DeepGL is designed to be space-efficient, interpretable, and computationally efficient.

## DETAILED DESCRIPTION

### Overview of DeepGL

DeepGL is a general, expressive, and flexible deep graph representation learning framework that addresses the limitations of existing methods. The core idea behind DeepGL is to learn relational functions that can be applied to any arbitrary graph, enabling inductive transfer learning. These relational functions are composed of relational feature operators applied to base features derived from the graph structure and attributes.

### Base Graph Features

The first step in the DeepGL framework is to derive a set of base graph features using the graph topology and any available attributes. These base features serve as the foundation for learning deeper and more discriminative features. Common base features include:

- **Degree Features**: In-degree, out-degree, total degree, and weighted degree for each node or edge.
- **Egonet Features**: Features derived from the ego-network of a node or edge, which includes the set of nodes and edges within a certain distance (e.g., 1-hop).
- **Graphlet Frequencies**: Counts of small induced subgraphs (graphlets) around each node or edge, capturing higher-order structural patterns.
- **Attribute Features**: Any initial attributes provided as input, such as node labels or edge weights.

These base features are concatenated into a feature matrix \( \mathbf{X} \), which is then used as the starting point for learning deeper features.

### Relational Function Space & Expressivity

DeepGL defines a space of relational functions that can be expressed and searched over. A relational function is a composition of relational feature operators applied to an initial base feature. Formally, a relational function \( f \) of order \( h \) is defined as:

\[ f = F_h(F_{h-1}(\ldots F_1(x) \ldots)) \]

where \( x \) is an initial base feature and \( F_i \) are relational feature operators. The expressivity of DeepGL depends on several components:

- **Base Features**: The initial features derived from the graph structure and attributes.
- **Relational Feature Operators**: A set of operators \( F = \{F_1, F_2, \ldots, F_K\} \) that can be applied to the feature values of neighboring graph elements.
- **Sets of Related Graph Elements**: The sets \( S \) of related graph elements (e.g., 1-hop neighbors) used with each relational feature operator.
- **Depth**: The number of times each relational function is composed with another.

### Composing Relational Functions

The space of relational functions searched by DeepGL is defined compositionally in terms of the relational feature operators. For example, given an initial base feature \( x \), a relational function can be constructed by applying a sequence of relational feature operators:

\[ x' = F_k(F_j(F_i(x))) \]

This composition allows DeepGL to capture increasingly complex and higher-order subgraph patterns. The framework is flexible and can incorporate a wide variety of relational feature operators, such as mean, sum, and product, to derive rich and expressive features.

### Searching the Relational Function Space

DeepGL employs a general and flexible framework for learning a hierarchical graph representation. The process involves the following steps:

1. **Deriving Base Features**: Compute the initial base features using the graph structure and attributes.
2. **Transforming Feature Vectors**: Optionally transform the feature vectors to reduce noise and enhance interpretability (e.g., using logarithmic binning).
3. **Learning Feature Layers**: Iteratively derive new feature layers by applying relational feature operators to the features learned in the previous layer.
4. **Evaluating and Pruning Features**: Evaluate the importance of the features in each layer and prune redundant or noisy features.
5. **Convergence Check**: Terminate the learning process when no new features emerge or the maximum number of layers is reached.

### Feature Diffusion

DeepGL introduces the concept of feature diffusion, where the feature matrix at each layer can be smoothed using a feature diffusion process. This process helps to regularize the learned features and improve their generalizability. For example, the feature matrix \( \mathbf{X} \) can be diffused using the adjacency matrix \( \mathbf{A} \) and the degree matrix \( \mathbf{D} \):

\[ \mathbf{X}^{(t)} = \mathbf{D}^{-1} \mathbf{A} \mathbf{X}^{(t-1)} \]

This diffusion process can be repeated for a fixed number of iterations or until convergence, effectively smoothing the feature vectors by the features of related graph elements.

### Supervised Representation Learning

DeepGL can be extended for supervised representation learning by incorporating an appropriate objective function. For example, the framework can be modified to find a set of features that maximize relevancy (predictive quality) with respect to the target labels while minimizing redundancy among the features. The objective function can be formulated as:

\[ \max_{x_i} \left( \alpha I(y; x_i) - (1 - \alpha) \sum_{x_j \in X} I(x_i; x_j) \right) \]

where \( I \) is a measure such as mutual information, \( y \) is the target label, \( X \) is the set of selected features, and \( \alpha \) is a hyperparameter balancing relevance and redundancy.

### Time and Space Complexity

The time complexity of learning edge and node features using DeepGL is linear in the number of edges, making it scalable for large graphs. Specifically, the time complexity is \( O(KFM) \), where \( K \) is the number of relational feature operators, \( F \) is the number of relational functions learned, and \( M \) is the number of edges. The space complexity of the learned sparse feature matrix \( \mathbf{X} \) is \( O(F \lceil aN \rceil) \), where \( a \) is the bin size of the logarithmic binning transformation and \( N \) is the number of nodes.

### Inductive Relational Functions

One of the key advantages of DeepGL is its ability to learn inductive relational functions that can be applied to any arbitrary graph. This enables across-network transfer learning tasks, such as graph matching, similarity, and across-network classification. Given a set of learned relational functions \( F \), the time complexity for extracting these functions on another graph is \( O(FM) \), which is linear in the number of edges.

### Experimental Evaluation

The effectiveness of DeepGL has been demonstrated through extensive experiments on a variety of graph-based learning tasks, including link classification, across-network transfer learning, and link prediction. DeepGL consistently outperforms state-of-the-art methods, achieving significant improvements in predictive performance, space efficiency, and runtime. Key findings include:

- **Predictive Performance**: DeepGL achieves an average gain of 18.09% to 20.80% in AUC for link classification tasks.
- **Space Efficiency**: DeepGL learns sparse feature representations that require up to 6x less memory compared to existing methods.
- **Runtime Efficiency**: DeepGL is up to 182 times faster than node2vec and 106 times faster than LINE for large graphs.
- **Interpretability**: The learned features are more interpretable, providing insights into the structural roles of nodes and edges in the graph.

### Conclusion

DeepGL is a powerful and flexible framework for inductive graph representation learning that addresses the limitations of existing methods. By learning relational functions that generalize across different graphs, DeepGL enables inductive transfer learning tasks and provides a more interpretable and efficient alternative to traditional node embedding methods. The framework is designed to be space-efficient, computationally efficient, and expressive, making it suitable for a wide range of applications and graph types.