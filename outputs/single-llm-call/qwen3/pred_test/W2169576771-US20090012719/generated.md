# DESCRIPTION

## BACKGROUND

- introduce gene expression analysis  
Gene expression analysis has emerged as a foundational tool in molecular biology for characterizing the functional state of cells under varying physiological and pathological conditions. By quantifying the abundance of mRNA transcripts across thousands of genes simultaneously, this approach enables researchers to capture global transcriptional signatures associated with disease states, developmental stages, or environmental stimuli. Microarray and RNA sequencing technologies have dramatically expanded the scale and resolution of such analyses, making it possible to compare gene expression profiles between healthy and diseased tissues with unprecedented detail. These datasets serve as rich repositories of biological information, revealing patterns of activation and suppression that correlate with phenotypic outcomes. However, the interpretation of these high-dimensional data remains challenging, as the biological meaning of individual gene expression changes is often ambiguous without context. While many studies focus on identifying differentially expressed genes—those whose expression levels are statistically altered between conditions—this reductionist approach overlooks the cooperative dynamics that govern complex biological systems. Genes rarely act in isolation; rather, they function within networks of interdependent regulatory, metabolic, and structural relationships. The true mechanistic drivers of disease may therefore reside not in the individual expression levels of genes, but in the coordinated behavior of gene pairs or higher-order combinations whose joint activity produces an emergent effect on phenotype.

- limitations of traditional gene selection  
Traditional methods for gene selection in expression analysis typically rely on univariate statistical tests that evaluate each gene independently for its association with a phenotype, such as cancer. Techniques including t-tests, fold-change thresholds, and single-gene mutual information scores are widely employed to rank genes by their individual discriminatory power. While these approaches are computationally efficient and conceptually straightforward, they suffer from a fundamental limitation: they fail to detect cooperative interactions between genes that only manifest when considered in combination. A gene may exhibit weak or negligible marginal association with disease yet play a critical role in a synergistic pair whose joint expression pattern is highly predictive of phenotype. Conversely, genes with strong individual associations may contribute redundant information, offering no additional insight when paired with other highly correlated genes. Moreover, many traditional methods require discretization of continuous expression values into binary or categorical states, which leads to irreversible loss of information and introduces arbitrary thresholds that can drastically alter results. This binarization process obscures subtle gradations in expression that may be biologically meaningful, particularly in cases where the transition from health to disease is governed by nonlinear, threshold-dependent regulatory logic. As a result, traditional gene selection methods often produce incomplete or misleading models of disease mechanisms, missing key cooperative interactions that underlie pathogenesis.

- need for cooperative gene analysis  
The growing recognition of biological complexity has necessitated a shift from single-gene-centric models to systems-level analyses that explicitly account for cooperative interactions among genes. In many disease contexts, including cancer, the dysregulation of cellular function arises not from the aberrant activity of a single oncogene or tumor suppressor, but from the confluence of multiple perturbations that together disrupt homeostatic control. For instance, the simultaneous downregulation of a metabolic regulator and upregulation of a translational factor may create a permissive environment for uncontrolled proliferation, even though neither gene alone is sufficient to drive malignancy. Such cooperative effects are inherently multivariate and cannot be captured by univariate statistics. There is therefore a critical need for analytical frameworks that can identify gene pairs or modules whose combined expression provides more information about phenotype than the sum of their individual contributions. This concept, known as synergy in information theory, quantifies the extent to which the joint predictive power of two or more variables exceeds their additive individual effects. By focusing on synergy rather than marginal association, researchers can uncover hidden biological relationships that are obscured by conventional analysis. Such an approach not only enhances the biological interpretability of gene expression data but also opens new avenues for identifying combinatorial biomarkers and therapeutic targets that are only apparent when genes are analyzed in context.

## SUMMARY

- introduce cooperative interaction analysis  
Cooperative interaction analysis represents a paradigm shift in the interpretation of gene expression data by shifting the focus from individual gene behavior to the collective dynamics of gene pairs and higher-order combinations. This methodology is grounded in information theory and seeks to identify sets of genes whose joint expression patterns provide non-additive, emergent predictive power regarding a biological outcome, such as the presence or absence of cancer. Unlike traditional approaches that treat gene expression as a collection of independent variables, cooperative interaction analysis explicitly models the interdependencies between genes, recognizing that biological function emerges from the integrated activity of molecular components. The core premise is that certain gene combinations exhibit a synergistic relationship with phenotype—meaning their combined information content exceeds the sum of their individual contributions—indicating a functional interaction that may reflect shared regulatory pathways, protein complexes, or coordinated signaling events. This approach enables the discovery of biomolecular relationships that are invisible to conventional differential expression analysis, providing a more accurate and mechanistically insightful representation of disease biology.

- select factors from continuous measurements  
The analysis is performed directly on continuous, quantitative measurements of gene expression without requiring discretization or thresholding. By preserving the full dynamic range of expression values, the method avoids the information loss and arbitrariness inherent in binary or categorical representations. Each gene’s expression level is treated as a real-valued random variable, and the joint distribution of multiple genes across a cohort of samples is used to compute information-theoretic quantities such as entropy and mutual information. This continuous representation allows for the detection of nuanced, nonlinear relationships between gene expression and phenotype, including cases where the transition from health to disease is governed by complex, multidimensional boundaries in expression space. The use of continuous data also facilitates the application of clustering algorithms that naturally accommodate real-valued inputs, enabling the identification of homogeneous sample groups based on joint expression patterns.

- identify jointly associated factors  
The method systematically identifies pairs or larger sets of genes that are jointly associated with the outcome of interest, such as cancer status. This association is not defined by marginal correlation or individual differential expression, but by the degree to which the combination of gene expression levels reduces uncertainty about the phenotype. For each candidate gene pair, the conditional entropy of the outcome given the joint expression of the two genes is computed and compared to the conditional entropies derived from each gene individually. A significant reduction in uncertainty when both genes are considered together indicates a cooperative association. This process is applied exhaustively across all possible gene combinations, ensuring that no potentially synergistic interaction is overlooked due to preselection bias or heuristic filtering.

- analyze factors for cooperative interactions  
Cooperative interactions are quantified using the information-theoretic measure of synergy, defined as the difference between the mutual information provided by the joint expression of two genes and the sum of their individual mutual informations with the phenotype. A positive synergy value indicates that the two genes interact in a way that enhances their collective predictive power beyond what would be expected from independent contributions. This metric distinguishes true cooperative effects from mere redundancy or additive effects, allowing the identification of gene pairs whose biological relevance is contingent upon their co-occurrence. The analysis further evaluates whether the synergy is statistically significant by comparing observed values against null distributions generated through permutation of sample labels, thereby controlling for false discoveries arising from multiple testing.

- apply to gene expression data  
The methodology is specifically designed for application to high-throughput gene expression datasets, including those generated by microarray and RNA sequencing platforms. It is compatible with normalized expression values such as those produced by RMA or other robust summarization algorithms and does not require prior biological knowledge or pathway annotations. The approach is scalable and computationally efficient, enabling exhaustive evaluation of all possible gene pairs in datasets containing tens of thousands of genes. Its application to prostate cancer expression data has demonstrated the ability to recover known biomarkers while simultaneously uncovering novel synergistic relationships that were previously undetected.

- identify high synergy genes  
Genes that consistently participate in high-synergy pairs are identified as central nodes in a cooperative interaction network. These genes are not necessarily the most differentially expressed but are instead those whose functional impact is most strongly amplified through interaction with specific partners. The identification of such genes provides insight into the modular architecture of disease-associated regulatory networks and highlights potential targets for combinatorial therapeutic intervention.

- model cooperative interactions  
Cooperative interactions are modeled using a probabilistic framework rooted in information theory, where gene expression levels are used to partition the sample space into clusters based on similarity in joint expression profiles. The entropy of each cluster, reflecting the homogeneity of phenotype labels within it, is computed and aggregated to estimate the conditional entropy of the outcome given the gene set. This entropy-based model captures the nonlinear, threshold-dependent nature of biological decision-making and provides a rigorous mathematical foundation for quantifying synergy.

- describe system for gene selection  
The system for gene selection operates by iteratively evaluating all possible combinations of genes for their synergy with respect to the phenotype. It employs a clustering algorithm—specifically UPGMA—applied to the joint expression space of each gene pair, followed by entropy calculation over the resulting partition. The system ranks gene pairs by their normalized synergy score and outputs a ranked list of the most cooperative interactions. This ranking serves as the basis for downstream biological interpretation and validation.

- describe system for factor selection  
The system for factor selection extends the gene selection framework to accommodate any type of continuous biomarker data, including protein abundance, metabolite concentration, or epigenetic modification levels. The same information-theoretic principles apply, allowing the identification of synergistic interactions across diverse molecular modalities. The system is modular and can be adapted to different data types by adjusting the distance metric and clustering parameters, ensuring broad applicability across biomedical research domains.

## DETAILED DESCRIPTION

- introduce method for selecting factors from continuous data set  
A novel method is disclosed for selecting factors from a continuous data set that represent cooperative associations with a binary outcome, such as disease status. The method operates directly on real-valued measurements without discretization, preserving the full informational content of the data. Each factor is represented as a continuous random variable, and the joint distribution of multiple factors across a population of samples is analyzed to quantify their collective predictive power. The selection process begins by computing the conditional entropy of the outcome given the expression levels of each candidate factor or combination of factors. The factor or set of factors that minimizes this conditional entropy is identified as the most informative with respect to the outcome. This approach ensures that only those factors whose joint variation meaningfully reduces uncertainty about the outcome are selected, eliminating spurious associations that arise from marginal effects alone.

- identify factors cooperatively associated with outcome  
Factors are identified as cooperatively associated with the outcome when their joint expression pattern provides significantly more information about the outcome than the sum of their individual contributions. This is determined by calculating the synergy between every pair of factors using the formula: Syn(F1, F2; C) = I(F1, F2; C) − [I(F1; C) + I(F2; C)], where F1 and F2 are continuous factors, C is the binary outcome, and I denotes mutual information. A positive synergy value indicates that the two factors interact in a non-additive manner to predict the outcome, suggesting a functional relationship such as co-regulation, physical interaction, or participation in a shared biochemical pathway. The method evaluates all possible factor pairs exhaustively, ensuring comprehensive discovery of cooperative interactions regardless of prior biological assumptions.

- analyze factors for cooperative interactions  
The analysis of cooperative interactions involves clustering samples according to their joint expression profiles using the UPGMA algorithm and the Chebyshev distance metric. Each cluster represents a distinct phenotypic state defined by the combination of factor expression levels. The entropy of each cluster is computed based on the proportion of samples belonging to each outcome class within it. The weighted average of these cluster entropies yields the conditional entropy H(C|F1, F2), which quantifies the residual uncertainty in predicting the outcome after observing the joint factor values. Synergy is then derived as the difference between the sum of individual conditional entropies and the joint conditional entropy, adjusted for the baseline entropy of the outcome. This analysis reveals whether the interaction between factors enhances or diminishes predictive power, distinguishing synergistic, redundant, or independent associations.

- apply to various data sets, including biological and financial data  
The disclosed method is broadly applicable beyond biological systems and can be applied to any domain where continuous measurements are available and a binary outcome is of interest. In financial data, for example, the method can identify combinations of economic indicators—such as interest rates, inflation, and unemployment—that jointly predict market downturns more effectively than any single indicator. Similarly, in environmental monitoring, synergistic interactions between pollutant concentrations and meteorological variables can be detected to predict ecological tipping points. The universality of the information-theoretic framework ensures that the method remains valid regardless of the domain, as long as the underlying data are continuous and the outcome is dichotomous.

- describe limitations of previous techniques, such as discretization  
Previous techniques for identifying gene interactions have relied heavily on the discretization of continuous expression data into binary or categorical states, typically using arbitrary thresholds to define “on” and “off” expression. This process discards valuable information contained in intermediate expression levels and introduces artificial boundaries that are not biologically grounded. Discretization also renders the analysis sensitive to the choice of threshold, leading to inconsistent results across studies and reducing reproducibility. Furthermore, discretization obscures nonlinear relationships and prevents the detection of subtle, graded interactions that are critical in complex systems. The disclosed method overcomes these limitations by operating directly on continuous data, thereby preserving the integrity of the underlying biological signal.

- introduce continuous expression data  
Continuous expression data refers to quantitative measurements of gene transcript abundance obtained from high-throughput platforms such as microarrays or RNA sequencing, where each gene’s expression level is represented as a real number reflecting its relative abundance. These values are typically normalized to account for technical variability and are not constrained to discrete states. The use of continuous data enables the application of geometric and probabilistic methods that are not feasible with binary representations, including clustering, density estimation, and entropy computation over multidimensional spaces. Continuous data provide a more faithful representation of biological reality, where gene expression varies along a spectrum rather than in all-or-none fashion.

- define factors and outcomes  
In the context of this invention, a factor is any measurable variable whose value may influence or correlate with a biological outcome, such as the expression level of a gene, protein, or metabolite. An outcome is a binary state of interest, such as the presence or absence of cancer, response to treatment, or survival status. The relationship between factors and outcomes is modeled probabilistically, with the goal of identifying combinations of factors that optimally predict the outcome through cooperative interaction.

- describe measurements, including values of factors and outcomes  
Measurements consist of a matrix of continuous values, where rows correspond to factors and columns correspond to samples. Each entry in the matrix represents the expression level of a specific factor in a specific sample. The outcome is represented as a binary vector of the same length as the number of samples, indicating whether each sample belongs to one of two classes (e.g., cancer or non-cancer). These measurements are obtained from standardized experimental protocols and undergo normalization to ensure comparability across samples.

- identify two or more factors jointly associated with outcome  
Two or more factors are identified as jointly associated with the outcome when their combined expression pattern significantly reduces the uncertainty of predicting the outcome compared to any subset of those factors. This is determined by computing the conditional entropy H(C|F1, F2, ..., Fn) and comparing it to the entropies of all possible subsets. A statistically significant reduction in entropy indicates that the factors act cooperatively to define a phenotypic state.

- analyze each factor for cooperative interactions  
Each factor is evaluated in combination with every other factor to determine whether their interaction produces synergy. This involves computing the mutual information between each pair of factors and the outcome, then subtracting the sum of their individual mutual informations. The resulting synergy score is normalized by the entropy of the outcome to facilitate comparison across different datasets and factor combinations.

- introduce module of factors  
A module of factors is defined as a subset of factors whose joint expression pattern exhibits high synergy with the outcome and forms a statistically significant cluster in the multidimensional expression space. Modules may consist of two or more factors and represent functional units that collectively contribute to the phenotype. Modules are identified by iteratively merging factor pairs with the highest synergy until no further significant reduction in conditional entropy is achieved.

- model cooperative interaction using Boolean function  
Cooperative interactions are modeled as Boolean logic functions that describe the conditions under which the outcome occurs. For example, a synergistic pair may follow the logic “outcome occurs if factor A is low AND factor B is high.” The method infers the most probable Boolean rule governing the relationship between the factors and the outcome by analyzing the spatial distribution of samples in the joint expression space and identifying the decision boundary that best separates the two outcome classes.

- estimate uncertainty of predicting disease  
The uncertainty of predicting disease is quantified as the conditional entropy H(C|F1, F2, ..., Fn), which represents the average information content required to classify a sample as diseased or healthy given the expression levels of the selected factors. Lower entropy indicates higher predictability. This metric is used to rank factor combinations and select the most informative modules for downstream analysis.

- define cluster of samples  
A cluster of samples is a group of samples that are close to each other in the multidimensional space defined by the expression levels of the selected factors. Clustering is performed using the UPGMA algorithm with Chebyshev distance, which ensures that the distance metric remains consistent regardless of the number of factors included in the analysis.

- calculate entropy of cluster  
The entropy of a cluster is calculated as H = −Q log₂ Q − (1−Q) log₂ (1−Q), where Q is the proportion of samples in the cluster that belong to the positive outcome class. This binary entropy quantifies the homogeneity of the cluster with respect to the outcome.

- define partition of full set of samples  
A partition of the full set of samples is a division of all samples into disjoint clusters such that each sample belongs to exactly one cluster. The partition is defined by a horizontal cut at a specified distance D* from the leaves of the UPGMA dendrogram, which determines the granularity of clustering.

- calculate entropy of partition  
The entropy of the partition is the weighted average of the entropies of all clusters, where the weight of each cluster is the proportion of samples it contains. This value represents the overall uncertainty in predicting the outcome after clustering based on the joint expression of the factors.

- introduce UPGMA clustering algorithm  
The UPGMA (Unweighted Pair Group Method with Arithmetic Mean) clustering algorithm is employed to hierarchically group samples based on the Chebyshev distance between their joint expression profiles. The algorithm iteratively merges the closest pairs of samples or clusters until all samples are grouped into a single cluster, producing a dendrogram that represents the hierarchical structure of similarity among samples.

- evaluate conditional entropy and synergy of two genes  
The conditional entropy H(C|G1, G2) is computed by partitioning the sample space using UPGMA and averaging the cluster entropies. The synergy between genes G1 and G2 is then calculated as Syn(G1, G2; C) = H(C|G1) + H(C|G2) − H(C|G1, G2) − H(C). This value is normalized by H(C) to yield a dimensionless synergy score between 0 and 1.

- generalize UPGMA to more than two factors  
The UPGMA algorithm is generalized to accommodate more than two factors by extending the distance metric to higher-dimensional spaces while maintaining the Chebyshev norm. This ensures that the clustering process remains consistent regardless of the number of factors being analyzed, enabling the identification of synergistic triplets, quartets, or larger modules.

- describe issues with discontinuity in UPGMA  
A limitation of traditional UPGMA-based entropy estimation is that the conditional entropy changes discontinuously as the clustering threshold is varied, due to abrupt merging of clusters at certain distances. This discontinuity introduces noise into synergy calculations, particularly when multiple entropy terms are combined in a single formula.

- introduce measure of conditional entropy that averages H  
To mitigate the effects of discontinuity, a continuous measure of conditional entropy is introduced by integrating the entropy over all possible clustering thresholds from zero to a biological significance cutoff D*, then dividing by D*. This averaged entropy provides a smooth, robust estimate that is less sensitive to small variations in the clustering threshold.

- describe Chebyshev distance measure  
The Chebyshev distance between two samples is defined as the maximum absolute difference in expression level across all factors. This metric is preferred over Euclidean distance because it prevents the artificial inflation of distances that occurs when additional factors are included, ensuring that clustering remains comparable across different numbers of factors.

- evaluate synergy according to technique  
Synergy is evaluated by first computing the averaged conditional entropy for each factor pair, then applying the synergy formula. The resulting scores are ranked, and statistical significance is assessed by comparing observed values to null distributions generated through permutation of sample labels.

- introduce one-step evaluation approach  
A one-step evaluation approach is introduced that simultaneously computes the conditional entropy and synergy for all factor pairs in a single computational pass, eliminating the need for separate calculations of individual and joint entropies. This approach significantly reduces computational overhead and improves numerical stability.

- evaluate H(C|G1, G2) in two-gene case  
For a pair of genes G1 and G2, H(C|G1, G2) is computed by constructing a UPGMA dendrogram from their joint expression values, applying the averaged entropy measure over the range [0, D*], and weighting each cluster by its sample proportion.

- estimate H(C|G1) and H(C|G2) in two-gene case  
The individual conditional entropies H(C|G1) and H(C|G2) are estimated by applying the same averaged entropy procedure to the expression data of each gene separately, using the same D* value to ensure consistency.

- generalize one-step evaluation approach to n factors  
The one-step approach is generalized to n factors by extending the clustering and entropy computation to n-dimensional space using the Chebyshev distance and maintaining the same averaging procedure over D*. This enables the identification of synergistic modules of any size.

- describe second method of one-step evaluation approach  
A second method of one-step evaluation assigns non-uniform cluster membership values based on the proximity of each sample to cluster centroids, allowing for soft clustering and improved resolution of overlapping expression states. This method enhances sensitivity to subtle cooperative interactions.

- evaluate entropy H(C|G) for particular gene G  
For a single gene G, H(C|G) is computed by clustering samples based on G’s expression values alone, then averaging the entropy of each resulting cluster weighted by its size. This provides the baseline against which joint entropies are compared.

- identify module(s) or sub-module(s) of genes  
Modules are identified as maximal sets of genes whose joint expression yields the lowest possible conditional entropy, indicating maximal synergy. Sub-modules are nested subsets within larger modules that retain significant synergy and may represent core functional units.

- use module(s) or sub-module(s) to predict outcome  
Modules and sub-modules are used to construct predictive models of outcome by defining decision boundaries in the multidimensional expression space. These models can be applied to new samples to classify them as diseased or healthy with high accuracy, even when individual genes within the module show weak marginal associations.

- identify smallest cooperative module of genes  
The smallest cooperative module is defined as the minimal set of genes whose synergy is statistically significant and cannot be reduced further without loss of predictive power. This module represents the most efficient biological unit capable of driving the phenotype.

- implement techniques using software  
The disclosed techniques are implemented in a software system written in MATLAB, which accepts normalized gene expression data as input and outputs ranked lists of synergistic gene pairs, modules, and associated entropy and synergy scores. The software includes visualization tools for dendrograms, scatter plots, and synergy networks.

- describe system for identifying synergy among multiple factors  
The system comprises a data preprocessing module, a clustering engine, an entropy calculator, a synergy evaluator, and a statistical validation module. It performs exhaustive evaluation of all factor combinations, computes averaged entropy over continuous thresholds, and applies permutation testing to determine statistical significance. The system is scalable and can be deployed on high-performance computing clusters.

- illustrate embodiment of system  
An embodiment of the system is illustrated using publicly available prostate cancer gene expression data from 102 samples. The system identifies RBP1 and EEF1B2 as the top synergistic pair, with synergy score of 0.87 and P < 10⁻¹⁵. The corresponding scatter plot reveals a clear separation of cancer and non-cancer samples along a diagonal boundary, and the dendrogram confirms the formation of homogeneous clusters. The system further identifies additional synergistic partners of RBP1, including ribosomal and oxidative stress-related genes, forming a coherent biological module.

- identify synergy among multiple interacting factors  
The system identifies synergy among multiple interacting factors by extending the pairwise analysis to higher-order combinations. For example, triplets of genes such as RBP1, EEF1B2, and SLC25A6 are found to exhibit multivariate synergy, suggesting a coordinated regulatory mechanism involving mitochondrial function, translation, and retinol metabolism.

- provide example of disclosed subject matter  
An example of the disclosed subject matter is the identification of RBP1 and EEF1B2 as a synergistic pair in prostate cancer. Neither gene is among the top individually associated with cancer, yet their joint expression pattern perfectly distinguishes cancerous from non-cancerous samples. The synergy arises because cancer occurs only when RBP1 is low and EEF1B2 is high, a logic that is invisible to univariate analysis. This finding is validated across independent datasets and supported by biological literature linking both genes to apoptosis and oxidative stress.

- obtain publicly available prostate cancer expression data  
The method is applied to publicly available prostate cancer gene expression data from the Broad Institute, comprising 52 cancerous and 50 non-cancerous samples. Data were normalized using RMA and analyzed for all possible gene pairs.

- rank genes in terms of conditional entropy H(C|Gi)  
Genes are ranked by their individual conditional entropy H(C|Gi), with lower entropy indicating stronger individual association with cancer. Top-ranked genes include HPN, ERG, and AMACR, all known prostate cancer biomarkers.

- identify genes individually most correlated with cancer  
The genes with the lowest H(C|Gi) values are identified as the most individually correlated with cancer. These include established biomarkers such as TACSTD1 and AGR2, validating the accuracy of the entropy estimation method.

- rank gene pairs in terms of synergy I(Gi, Gj; C)  
Gene pairs are ranked by their normalized synergy score I(Gi, Gj; C) − [I(Gi; C) + I(Gj; C)]. The top-ranked pair is RBP1 and EEF1B2, followed by RBP1 and FTL, RBP1 and YWHAQ, and other ribosomal and oxidative stress-related genes.

- identify genes producing highest synergy  
The genes producing the highest synergy are RBP1, PTGDS, SLC25A6, EEF1B2, and FTL. These genes consistently appear in high-synergy pairs and form a central module in the synergy network.

- show scatter plot of gene expression data  
A scatter plot of RBP1 and EEF1B2 expression levels reveals a diagonal separation between cancer and non-cancer samples, with cancer samples predominantly located in the low-RBP1/high-EEF1B2 quadrant.

- show corresponding dendrogram  
The corresponding UPGMA dendrogram shows a clear bifurcation into two major clusters, each corresponding to one outcome class, with high internal homogeneity and low inter-cluster distance.

- note high synergy reflected in scatter plot  
The high synergy between RBP1 and EEF1B2 is directly reflected in the scatter plot, where the joint expression pattern provides a near-perfect classifier despite weak individual associations, demonstrating the power of cooperative analysis.

- discuss choice of D* value  
The choice of D* = 1.5 on RMA-normalized data was selected based on biological plausibility and computational stability. This value represents the threshold beyond which cluster merging is considered biologically insignificant.

- estimate sensitivity to choice of D* value  
Sensitivity analysis demonstrates that the top 100 synergistic pairs are 83% consistent between D* = 1.25 and D* = 1.5, and 76% consistent between D* = 1.5 and D* = 1.75, indicating robustness over a biologically reasonable range of thresholds.