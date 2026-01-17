# DESCRIPTION

## BACKGROUND

Synthetic lethality (SL) is a phenomenon where the simultaneous disruption of two non-essential genes leads to cell death. This concept has significant implications in cancer therapy, as it can guide the development of combination therapies that selectively target cancer cells while sparing healthy cells. However, identifying SL gene pairs in humans is challenging due to the vast number of possible gene pairs and the ethical and practical constraints of experimental validation. Traditional methods, such as genetic homology, structural similarity, and functional similarity, have limitations in predicting SL across species. This patent application introduces a novel method, Species INdependent TRAnslation (SINaTRA), which leverages network connectivity profiles to predict SL in any species with an available protein-protein interaction (PPI) network. SINaTRA significantly outperforms existing methods and can be applied to species without known SL pairs, making it a powerful tool for advancing cancer research and drug discovery.

## SUMMARY

The present invention provides a method for predicting synthetic lethality (SL) in any species using protein-protein interaction (PPI) network data. The method, termed Species INdependent TRAnslation (SINaTRA), involves the following steps:
1. **Network Construction**: Constructing a PPI network for the source species (e.g., S. cerevisiae) and the target species (e.g., S. pombe, M. musculus, or H. sapiens).
2. **Parameter Calculation**: Calculating network parameters for each gene and gene pair in the PPI networks.
3. **Normalization**: Normalizing the network parameters to make them comparable across species.
4. **Model Training**: Training a machine learning model using the normalized network parameters and experimentally validated SL pairs from the source species.
5. **Prediction**: Applying the trained model to the target species to predict SL pairs.
6. **Post-Processing**: Filtering predicted SL pairs using additional biological priors to reduce false positives.

The invention further includes a database of predicted SL pairs in humans, which can be used to inform cancer combination therapy and other biomedical applications.

## DETAILED DESCRIPTION

### Network Construction

The first step in the SINaTRA method is to construct a PPI network for the source species and the target species. The PPI network is a graph where nodes represent genes and edges represent physical interactions between gene products. PPI data can be obtained from various databases, such as BioGRID. The network is pruned to contain one connected component to ensure that all nodes are reachable from each other.

### Parameter Calculation

For each gene and gene pair in the PPI networks, a set of network parameters is calculated. These parameters include, but are not limited to:
- Degree centrality: The number of edges connected to a node.
- Betweenness centrality: The number of shortest paths passing through a node.
- Closeness centrality: The average length of the shortest paths from a node to all other nodes.
- Eigenvector centrality: The influence of a node in the network.
- Clustering coefficient: The degree to which nodes in the network tend to cluster together.
- Shortest path: The shortest path between two nodes.
- Communicability: The ease with which information can flow between two nodes.
- Shared neighbors: The number of common neighbors between two nodes.
- Shared non-neighbors: The number of nodes that are not neighbors of both nodes.
- Shared 2nd-degree neighbors: The number of nodes that are neighbors of neighbors of both nodes.

These parameters are calculated using network analysis tools, such as NetworkX in Python.

### Normalization

The network parameters are normalized to make them comparable across species. Four normalization strategies are considered:
- **Regular normalization**: Dividing each value by the maximum value of that parameter.
- **Rank normalization**: Ranking all values from smallest to largest and dividing by the total number of genes or gene pairs.
- **Tied-rank normalization**: Assigning the median rank to all equal values and normalizing by the total number of genes or gene pairs.
- **Quantile normalization**: Up-sampling networks with fewer nodes/edges to match the distribution of the larger network.

Rank normalization is found to be the most effective and is used in the final model.

### Model Training

A machine learning model is trained using the normalized network parameters and experimentally validated SL pairs from the source species. The model is trained using a random forest classifier, which is accurate and efficient on large datasets and resistant to overfitting. The model is trained using five-fold cross-validation, where 80% of the data is used for training and 20% for testing.

### Prediction

The trained model is applied to the target species to predict SL pairs. Each gene pair in the target species is assigned a SINaTRA score between 0 and 1, representing the likelihood of an SL relationship. The model can be applied to any species with an available PPI network, including species without known SL pairs.

### Post-Processing

To reduce false positives, predicted SL pairs are filtered using additional biological priors. For example, gene pairs where both genes are homozygous for deleterious mutations in at least one patient are filtered out as confirmed non-SL pairs. The remaining high-confidence SL pairs are those with SINaTRA scores above a specified threshold (e.g., 0.95) that are not filtered out by the genetic screen.

### Example 1

**Application to S. pombe**

1. **Network Construction**: Construct a PPI network for S. cerevisiae and S. pombe.
2. **Parameter Calculation**: Calculate network parameters for each gene and gene pair in both networks.
3. **Normalization**: Normalize the network parameters using rank normalization.
4. **Model Training**: Train a random forest classifier using the normalized network parameters and experimentally validated SL pairs from S. cerevisiae.
5. **Prediction**: Apply the trained model to S. pombe to predict SL pairs.
6. **Post-Processing**: Filter predicted SL pairs using a database of homozygous deleterious mutations in S. pombe.

### Example 2

**Application to Mice**

1. **Network Construction**: Construct a PPI network for S. cerevisiae and M. musculus.
2. **Parameter Calculation**: Calculate network parameters for each gene and gene pair in both networks.
3. **Normalization**: Normalize the network parameters using rank normalization.
4. **Model Training**: Train a random forest classifier using the normalized network parameters and experimentally validated SL pairs from S. cerevisiae.
5. **Prediction**: Apply the trained model to M. musculus to predict SL pairs.
6. **Post-Processing**: Filter predicted SL pairs using a database of homozygous deleterious mutations in M. musculus.

### Example 3

**Application to Humans**

1. **Network Construction**: Construct a PPI network for S. cerevisiae and H. sapiens.
2. **Parameter Calculation**: Calculate network parameters for each gene and gene pair in both networks.
3. **Normalization**: Normalize the network parameters using rank normalization.
4. **Model Training**: Train a random forest classifier using the normalized network parameters and experimentally validated SL pairs from S. cerevisiae.
5. **Prediction**: Apply the trained model to H. sapiens to predict SL pairs.
6. **Post-Processing**: Filter predicted SL pairs using a database of homozygous deleterious mutations in humans. High-confidence SL pairs are those with SINaTRA scores above 0.95 that are not filtered out by the genetic screen.

The SINaTRA method provides a robust and species-independent approach to predicting synthetic lethality, which can significantly advance cancer research and drug discovery. The database of predicted SL pairs in humans can be used to identify novel cancer combination therapies and other biomedical applications.