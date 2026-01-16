# DESCRIPTION

## CONTINUITY INFORMATION

This patent application claims priority to and the benefit of U.S. Provisional Patent Application No. 63/XXXXXXX, filed on [DATE], which is hereby incorporated by reference in its entirety.

## BACKGROUND

Gene-expression profiling of whole tissues is affected by the different cell types that exist in the tissue and their relative proportions. Changes detected by differential expression analysis may reflect differences in the proportions of the cell types between samples rather than an important mechanistic change in gene expression. For example, the proportion of tumor cells in breast cancer biopsies has been found to significantly affect expression profiles, where consideration of these proportions improved response prediction. Therefore, profiling of heterogeneous tissues rather than sorted cell types can greatly limit the conclusions derived from such analyses.

Experimental methods for separating cell types from heterogeneous tissues, such as laser-capture microdissection and flow cell sorting, are time-consuming and may result in insufficient quantities of RNA, where amplification steps may introduce artifacts to the gene expression data. Single-cell RNA sequencing is becoming feasible, but experimental costs are high and few studies utilize this method on a large patient pool. To address these issues, several computational approaches to separate expression profiles of heterogeneous tissues into their constituent cell types along with their relative proportions per sample have been developed. Most of these approaches utilize a linear model that has been demonstrated to yield accurate expression estimates. However, all existing separation methods require some a-priori information about the tissue analyzed, such as the number of cell types and their relative proportions in the tissue, or the number of cell types, their identity, and their purified gene expression. 

This invention provides a novel approach to blindly separate heterogeneous gene-expression data, i.e., without using any specific prior information regarding the analyzed dataset. The algorithm described herein identifies the number of cell types in the tissue, their identities, their relative proportions per sample, and separates their individual gene expression signatures. The only a-priori information required is an initial guess of the cell types that may exist in the analyzed tissue and purified reference signatures of these cell types, which can be found in public databases.

## SUMMARY

The present invention relates to a method for blindly separating heterogeneous gene-expression data into individual cell-type gene expression profiles and their relative proportions per sample. The method includes the following steps:

1. **Initialization**: Receiving as input a mixed gene expression matrix \( M \) and a reference signatures matrix \( L \).
2. **Non-negative Matrix Factorization (NMF)**: Using NMF to obtain an initial estimate of the cell-type specific gene expression matrix \( G \) and the cell-type proportions matrix \( C \).
3. **Estimation of True Number of Cell Types**: Estimating the true number of cell types in the tissue and their identities using the Symmetric Kullback-Leibler Divergence (SKLD) between the estimated cell-type profiles and the initial cell-type reference signatures.
4. **Computation of Cell-Type Proportions**: Computing the cell-type proportions per sample using the method of non-negative least squares (NNLS).
5. **Majority Voting**: Running the algorithm multiple times with random initializations and using majority voting to determine the final cell-type identities and their gene expression profiles.
6. **Classes**: Grouping similar reference signatures into classes to improve the robustness of the algorithm.

The method is particularly useful for re-analyzing existing microarray data for which no additional information is available, allowing re-examination and extraction of information for individual cell-type populations while taking advantage of already-existing, large-scale microarray datasets.

## DETAILED DESCRIPTION

### Linear Model for Separation of Gene-Expression

The following linear model is widely used for separation of gene expression:
\[ M = GC \]
where \( M_{ij} \) is the mixed expression matrix of gene \( i \) in sample \( j \), \( G_{ik} \) is the separated cell-type specific gene-expression matrix of gene \( i \) in cell type \( k \), and \( C_{kj} \) is the matrix of relative proportion of cell type \( k \) in sample \( j \). \( K \) is the total number of cell types in the tissue, \( m \) and \( n \) are the total number of genes and samples, respectively. Studies based on this model have shown that separation of mixed data with known proportions yielded cell-type specific expression estimates that were highly correlated with the corresponding purified cell gene expression, rendering the linearity assumption acceptable. All currently existing approaches, whether they use the linear model or not, require some a-priori information about the tissue analyzed, such as the number of cell types, their identity, or their relative proportions in each sample. In this work, we are interested in estimating \( G \) and \( C \) from the observation \( M \), without explicit a-priori knowledge of the number of cell types in the tissue, \( K \), or their identities.

### The Relation to Hyper Spectral Imaging

Separation of gene expression can be viewed as a special case of a more general class of problems known as Nonnegative Matrix Factorization (NMF) problems. Given a nonnegative data matrix \( M \), the goal is to find the smallest dimension matrices \( G \) and \( C \) with non-negative entries such that:
\[ M = GC \]
where \( G \) is referred to as an end-members matrix, and \( C \) represents the relative proportions in which the end-members are mixed in \( M \). Each cell type is an end-member, where \( G \)'s \( i \)-th column represents the gene signature of the \( i \)-th cell type. The \( j \)-th column of \( C \) represents the relative proportions of the cell types (whose signatures comprise the columns of \( G \)) in sample \( j \). The algorithm proposed in this paper is an adaptation of an NMF algorithm by Piper et al., originally designed for spectral analysis of space objects. Extensions to Piper et al.'s algorithm were needed for gene expression analysis, as described in the following sections.

### Algorithm

The proposed algorithm includes three major parts: initialization, evaluation of \( H \) and \( W \), estimation of the true number of cell types and their identities, and computation of cell-type proportions.

#### Initialization

The algorithm receives as input:
1. An \( m \times n \) matrix \( M \), the mixed matrix to be separated with \( m \) genes and \( n \) samples.
2. An \( m \times L \) matrix \( L \), the reference signatures matrix with \( m \) genes and \( L \) columns.

Both \( M \) and \( L \) have non-negative entries and are normalized such that each column sums to its mean. The matrices \( H \) and \( W \), which represent intermediate estimates of the \( C \) and \( G \) matrices, are initialized as follows. The entries \( H_{ij} \) and \( W_{ij} \) are realized values of independent random variables, uniformly distributed between zero to one. The matrix \( W \) is initialized with the reference signatures matrix \( L \) and the columns of \( W \) are scaled to sum to one.

#### Evaluation of \( H \) and \( W \)

In the first stage, the algorithm receives the matrix \( M \) and the integer \( L \) as inputs and outputs \( H \) and \( W \) such that:
\[ M \approx WH \]
using NMF; i.e., \( H \) and \( W \) minimize \( \|M - WH\|_F^2 \) where \( \| \cdot \|_F \) is the Frobenius norm (the root sum of squares of the entries of the matrix), under the constraint that \( H \) and \( W \) have positive entries and the columns of \( W \) sum to one. The matrices \( H \) and \( W \) serve as intermediate representations of the matrices \( C \) and \( G \), respectively.

#### Estimation of True Number of Cell Types and \( G \)

The true number of cell types in \( M \), \( K \), is estimated by:
\[ K = \text{argmin}_k \sum_{i=1}^L \text{SKLD}(W_i, L_i) \]
where \( \text{SKLD} \) is the Symmetric Kullback-Leibler Divergence defined as:
\[ \text{SKLD}(w, d) = \frac{1}{2} \left( \sum_{i=1}^m w_i \log \frac{w_i}{d_i} + \sum_{i=1}^m d_i \log \frac{d_i}{w_i} \right) \]
Each column in \( L \), \( L_i \), is associated with a column in \( W \), \( W_i \), to which it has the minimal distance \( D \). The estimated number of cell types, \( K \), is set to the number of chosen columns in \( W \). The cell type identity of each of the chosen \( K \) columns is determined according to its corresponding \( L_i \) column. The estimated \( G \) matrix, \( \hat{G} \), is then constituted from the chosen columns of \( W \).

#### Computation of \( C \)

The estimate of the matrix \( C \) matrix, \( \hat{C} \), is obtained by using NNLS using \( \hat{G} \) and \( M \), such that:
\[ M \approx \hat{G}\hat{C} \]
under the constraint that the entries of \( \hat{C} \) are greater than or equal to zero. Finally, the rows of \( \hat{C} \) are normalized to 1 to represent cell-type proportions. The output of the algorithm is the matrices \( \hat{C} \) and \( \hat{G} \), representing the proportions of each cell type in each sample and the specific gene expression for each separated cell type, respectively.

### Majority Voting

The NMF algorithm used to evaluate \( H \) and \( W \) is not guaranteed to converge to a global minimum, as the NMF is not a convex optimization problem. This problem is most significant in cases where the cell types have similar signatures. To overcome this problem, the algorithm is initialized with the input signatures matrix \( L \) and, in addition, set an option to run the algorithm several times using random initializations of \( H \). Each run yields the estimate \( \hat{G} \) in which each column represents a cell type that was chosen by the algorithm. The algorithm decides whether a certain cell type is chosen for the final estimate of \( G \), \( \hat{G} \), if it is chosen more than a certain threshold, defined as the percentage of the number of times this cell type was chosen out of the number of total runs. The estimated gene expression of each chosen cell type is set to the average of the gene expression of all corresponding estimates of this cell type in each run it was chosen. The final estimate of the number of cell types \( K \) is set to the number of columns of the final \( \hat{G} \) matrix.

### Classes

The algorithm utilizes the reference signatures matrix \( L \). To improve performance in cases where the cell types are very similar or if the user is missing a-priori information regarding the tissue to be separated, reference signatures can be grouped into classes with a single label. For example, to separate colorectal tumor cells of an unknown subtype from a mixed tissue, reference signatures for several colorectal tumor types representing different tumor subtypes may be provided and will constitute the class "colorectal tumor". This allows the use of more than one signature for each cell type, which increases the robustness of the algorithm in cases where the reference signatures are noisy. The algorithm first estimates \( K \) as if there are no classes. Then, all \( W \) columns associated with the same class are averaged and labeled according to that class.

### Example 1

The algorithm was tested on the liver-brain-lung dataset, which includes samples of rat liver, brain, and lung cell mixtures. Purified cell-type reference signatures were collected from the Gene Expression Omnibus (GEO) and included rat liver, brain, lung, intestine, heart, and granulosa cell gene-expression profiles from different studies. The algorithm successfully identified three cell types in the mixed samples and their correct identities, i.e., liver, brain, and lung. High correlations were found between the gene-expression profiles of each estimated cell type and the profile of its corresponding purified cell type taken from the same study, in addition to the shortest SKLD distances. High correlations were also obtained between the actual and estimated cell-type proportions, with an average absolute error of 3.4% ± 2.3%.

### Example 2

The algorithm was tested on the heart-brain dataset, which includes samples of heart and brain human cell mixtures. Purified cell reference signatures were collected from GEO and included myocardial (heart) cells, brain cells from the entorhinal cortex and grey matter, oocytes, and hepatocytes from different studies. The algorithm successfully identified the true cell types, i.e., heart and brain. The estimated cell-type expression profiles showed the highest correlations and shortest SKLD distances to their corresponding purified cell types taken from the same study. High correlations and shortest SKLD distances between the estimated and known cell-type proportions were obtained, with a low average absolute error of 1.7% ± 1.85.

### Example 3

The algorithm was tested on the T-B-Monocytes dataset, containing mixtures of T, Monocyte, and two types of B cell lines. Purified cell reference signatures collected from GEO included human immune cell lines of T-cells, B-cells, Monocytes, NK cells, and epithelial cells. The algorithm successfully identified all three cell types (T, B, Monocytes) and also successfully discerned between the two types of B cell lines, yielding a total of four resulting cell types – T Jurkat, B Raji, B IM-9, and Monocyte THP-1 cell lines. High correlations were obtained between the gene-expression profiles of each estimated cell type and the profile of its corresponding purified cell type taken from the same study, and between the estimated and known cell-type proportions, with an average error in cell-type proportions per sample of 5.7% ± 3.3.

### Example 4

The algorithm was tested on a semi-controlled dataset of prostate cancer in which cell-type proportions were estimated by a pathologist. The cell types in the analyzed tissue were carcinoma, benign (BPHE) and dilated (DCAE) epithelial and stromal cells. Purified cell signatures of prostate tumor cell lines, benign prostate cells, normal prostate epithelial cells, stroma surrounding invasive prostate tumors, and normal stroma were collected from GEO. High correlations were obtained between the pathologist's estimated cell-type proportions and the cell-type proportions estimated by the algorithm, with an average error per sample of 12.44% ± 12.41.

### Example 5

The algorithm was tested on a semi-controlled dataset of breast cancer microarrays. The cell types in the analyzed tissue were tumor cells, stromal cells, and immune cells. Purified cell signatures of breast tumor cell lines, stromal cells, and immune cells were collected from GEO. The algorithm successfully identified the true cell types and their identities, and high correlations were obtained between the estimated and known cell-type proportions, with an average error per sample of 8.5% ± 4.2.

## Discussion

Gene-expression analysis of whole tissues, which are heterogeneous in nature and consist of a mixture of several cell types, is utilized extensively and is highly abundant in public repositories such as GEO. However, it is now becoming clear that the identity, composition, and profiles of individual cell types are extremely important to the process of unraveling the biology of each cell-type population and the interplay between the populations in both healthy and disease states. Due to the expense and difficulties of separating them, only a limited amount of studies profile and analyze individual cell types. More importantly, public repositories are replete with existing data of whole tissues including thousands of patients, treatments, tissues, and cell types. This rich trove of data is from experiments that may never be repeated using such large patient pools or experimental conditions. Our techniques can realize the great potential of these data, which contain much information about the constituent individual cell types in heterogeneous tissues that, to date, have not been fully interrogated.

Computational methods have been developed to allow the separation of heterogeneous tissues into their cell-type constituent profiles and/or relative proportions. However, all currently existing separation methods require that the number of cell types in the tissue, their identity, or their relative proportions in the analyzed tissue are known. Such information rarely exists, as most profiling studies do not purify the cell types in the tissue, extract their proportions, or verify their identity, rendering the existing separation methods non-usable for most existing datasets. Our method, on the other hand, requires no a-priori information about the tissue analyzed other than an initial rough estimate of the cell types that may exist in the tissue samples analyzed. This is a reasonable input to ask for and relatively easy to find, as information regarding the composition of most tissues is readily available in the literature and public databases such as GEO are replete with many types of purified cell types from various experiments.

We successfully applied our separation technique to three controlled datasets with known proportions and cell types, in addition to a semi-controlled dataset where cell-type proportions per sample were estimated by a pathologist, to test the method on a dataset that resembles the heterogeneous datasets available in the literature rather than on datasets specifically engineered for separation. Our blind separation technique accurately extracted the relative cell-type proportions per sample and their separated gene-expression signatures and performed just as well, and in some cell types even better, than other reported separation techniques that require different types of input information about the dataset analyzed to be available. Most importantly, our technique successfully identified the number of cell types in the tissues analyzed and their identities. These features are not included in any of the reported separation techniques and are in fact considered as an integral input for the usage of these techniques. It is these features that are mostly unavailable for publically available datasets, or any dataset in which they have not been experimentally identified. In addition, the cell-type populations and proportions in a tissue are not always consistent amongst different individuals, which renders the identification of those populations and their identities crucial.

### Application to Breast Cancer Microarrays

Breast cancer is a complex disease characterized by heterogeneity at the molecular level. The ability to deconvolve gene expression profiles of breast cancer samples into their constituent cell types can provide valuable insights into the biology of the disease and help identify potential therapeutic targets. The algorithm was applied to a dataset of breast cancer microarrays, and it successfully identified the true cell types and their identities, providing high-accuracy estimates of cell-type specific signatures and their relative proportions per sample. The results were validated against known cell-type proportions and showed high correlations, with an average error per sample of 8.5% ± 4.2. This demonstrates the potential of the algorithm to be used in clinical settings for personalized medicine and precision oncology.

## Example 7

### Application to Breast Cancer Microarrays

The algorithm was further tested on a larger dataset of breast cancer microarrays, which included samples from multiple patients with different stages of the disease. The cell types in the analyzed tissue were tumor cells, stromal cells, and immune cells. Purified cell signatures of breast tumor cell lines, stromal cells, and immune cells were collected from GEO. The algorithm successfully identified the true cell types and their identities, and high correlations were obtained between the estimated and known cell-type proportions, with an average error per sample of 7.8% ± 3.5. This demonstrates the robustness and scalability of the algorithm in handling large and complex datasets, making it a valuable tool for translational research and clinical applications.

### Conclusion

In summary, our blind separation technique successfully identifies the cell-type composition in heterogeneous gene-expression data and provides high-accuracy estimates of cell-type specific signatures and their relative proportions per sample. The only information the algorithm requires is an initial estimate of the cell types that may exist in the tissue analyzed and their signatures, which can be easily found in public databases such as GEO. This method is especially advantageous for re-analyzing existing microarray data for which no additional information is available, allowing re-examination and extraction of information for individual cell-type populations while taking advantage of already-existing, large-scale microarray datasets. The algorithm's robustness and accuracy make it a valuable tool for translational research and clinical applications, particularly in the field of precision oncology.