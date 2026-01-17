# DESCRIPTION

## BACKGROUND

The invention pertains to the field of bioinformatics and gene expression analysis, specifically addressing the challenge of inferring gene-gene interactions that are specifically associated with a phenotype, such as a particular cancer. Traditional methods for inferring gene interactions from microarray data, such as Bayesian networks, pairwise mutual information, and graphical Gaussian models, often fail to identify interactions that are uniquely relevant to the phenotype. These methods typically construct interaction networks from a single set of microarray data, which can include general biological functions unrelated to the phenotype of interest.

The present invention introduces a novel methodology for identifying gene pairs that are synergistically linked with respect to a phenotype, without the need for prior biological knowledge. This is achieved by using an information-theoretic measure called synergy, which quantifies the cooperative effect of two genes in providing information about the phenotype. The synergy of a gene pair is defined as the difference between the mutual information of the pair with respect to the phenotype and the sum of the individual mutual informations of the genes with respect to the phenotype. This approach allows for the identification of gene pairs that are cooperatively associated with the phenotype, even if the individual genes are not individually correlated with the phenotype.

The invention further provides a computational methodology for estimating synergy directly from continuous gene expression data, using a dendrogram-based clustering algorithm. This method overcomes the limitations of binarizing expression data, which can lead to loss of information and arbitrary thresholding. The methodology is applied to a publicly available prostate cancer microarray dataset, identifying gene pairs with high synergy and validating their biological significance.

## SUMMARY

The present invention provides a method for identifying gene pairs that are synergistically linked with respect to a phenotype, such as a particular cancer, from gene expression data. The method comprises the following steps:

1. **Data Preparation**: Obtain a set of gene expression data comprising samples from both healthy and diseased conditions.
2. **Entropy Calculation**: Compute the entropy of the class label (healthy vs. diseased) for each gene and for each pair of genes using a dendrogram-based clustering algorithm.
3. **Conditional Entropy Calculation**: Calculate the conditional entropy of the class label given the expression levels of each gene and each pair of genes.
4. **Mutual Information Calculation**: Determine the mutual information between the expression levels of each gene and the class label, and between the expression levels of each pair of genes and the class label.
5. **Synergy Calculation**: Compute the synergy of each gene pair using the formula:
   \[
   \text{Syn}(G_1, G_2; C) = I(G_1, G_2; C) - [I(G_1; C) + I(G_2; C)]
   \]
   where \(I\) denotes mutual information, \(G_1\) and \(G_2\) are the expression levels of the two genes, and \(C\) is the class label.
6. **Statistical Validation**: Perform permutation experiments to assess the statistical significance of the highest-synergy gene pairs.
7. **Network Construction**: Construct a synergy network by including gene pairs with statistically significant synergy values.
8. **Biological Interpretation**: Analyze the synergy network to identify potential pathways and biological mechanisms associated with the phenotype.

The invention further provides a software tool for implementing the above method, which can be used to analyze gene expression data from various diseases and conditions.

## DETAILED DESCRIPTION

### 1. Data Preparation

The method begins with the preparation of gene expression data. The data should include samples from both healthy and diseased conditions, preferably with a balanced number of samples in each group. The gene expression data can be obtained from microarray experiments or other high-throughput technologies. The data should be preprocessed to normalize the expression levels, such as using the Robust Multi-array Average (RMA) method.

### 2. Entropy Calculation

The entropy of the class label (healthy vs. diseased) is calculated for each gene and each pair of genes. The entropy of a cluster of samples is defined as:
\[
h(Q) = -Q \log_2 Q - (1 - Q) \log_2 (1 - Q)
\]
where \(Q\) is the relative frequency of diseased samples in the cluster. The entropy of a partition of the full set of samples into disjoint clusters is the weighted average of the entropies of all clusters, where the weights are the relative memberships of the clusters.

### 3. Conditional Entropy Calculation

The conditional entropy of the class label given the expression levels of each gene and each pair of genes is calculated using a dendrogram-based clustering algorithm. The UPGMA (Unweighted Pair Group Method with Arithmetic Mean) algorithm is used to create a dendrogram of the samples based on the expression levels of the selected genes. The conditional entropy is then computed as the average entropy of the clusters formed by cutting the dendrogram at a specified distance \(D^*\).

### 4. Mutual Information Calculation

The mutual information between the expression levels of each gene and the class label, and between the expression levels of each pair of genes and the class label, is calculated using the formula:
\[
I(G_1, G_2; C) = H(C) - H(C | G_1, G_2)
\]
where \(H(C)\) is the entropy of the class label and \(H(C | G_1, G_2)\) is the conditional entropy of the class label given the expression levels of the two genes.

### 5. Synergy Calculation

The synergy of each gene pair is calculated using the formula:
\[
\text{Syn}(G_1, G_2; C) = I(G_1, G_2; C) - [I(G_1; C) + I(G_2; C)]
\]
This measure quantifies the cooperative effect of the two genes in providing information about the phenotype. Positive synergy indicates that the two genes are synergistically linked with respect to the phenotype.

### 6. Statistical Validation

To assess the statistical significance of the highest-synergy gene pairs, permutation experiments are performed. Two types of permutations are used:
- **Permutation A**: Randomly shuffle the class labels of the samples.
- **Permutation B**: Independently shuffle the expression values of each gene within the healthy and diseased samples.

For each permutation, the highest-synergy gene pairs are identified, and the distribution of these values is used to estimate the significance of the highest-synergy gene pairs in the actual data. The Gumbel distribution is used to estimate the p-values, and the false discovery rate (FDR) is controlled to adjust for multiple comparisons.

### 7. Network Construction

A synergy network is constructed by including gene pairs with statistically significant synergy values. The network is represented as a graph where nodes are genes and edges represent synergistic interactions. The network can be visualized and analyzed to identify potential pathways and biological mechanisms associated with the phenotype.

### 8. Biological Interpretation

The synergy network is analyzed to identify potential pathways and biological mechanisms associated with the phenotype. For example, the network may reveal that certain genes are synergistically linked with respect to prostate cancer, and these genes may be involved in pathways related to oxidative stress, apoptosis, and cell proliferation. The biological interpretation of the network can provide insights into the molecular mechanisms underlying the phenotype and can guide further experimental studies.

### Example Application

The method was applied to a publicly available prostate cancer microarray dataset consisting of 102 samples, 50 of which were healthy and 52 of which were cancerous. The gene pairs with the highest synergy values were identified, and their biological significance was validated. The results showed that the gene RBP1 (cellular retinol-binding protein-1) is synergistically linked with many other genes, many of which are ribosomal genes. The findings support the hypothesis that prostate cancer is linked with cellular damage from oxidative stress combined with the inhibition of apoptotic mechanisms.

### Conclusion

The present invention provides a novel method for identifying gene pairs that are synergistically linked with respect to a phenotype, such as a particular cancer. The method uses an information-theoretic measure of synergy and a dendrogram-based clustering algorithm to estimate synergy directly from continuous gene expression data. The method is computationally efficient and can be applied to large datasets. The resulting synergy network can provide valuable insights into the molecular mechanisms underlying the phenotype and can guide further experimental studies.