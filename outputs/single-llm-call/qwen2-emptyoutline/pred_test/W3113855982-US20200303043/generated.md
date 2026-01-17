# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a method for assessing the efficiency of cannabis drugs in treating various conditions, particularly inflammation, by analyzing changes in signaling pathway activation (SPA) using high-throughput gene expression data. The invention provides a novel approach, the Cannabis Drug Efficiency Index (CDEI), which integrates transcriptomic data with signaling pathway topology to predict and rank the efficacy of cannabis extracts or other botanical compounds.

## BACKGROUND OF THE INVENTION

In the field of personalized medicine, the development of diagnostic tools for characterizing individual responses to drugs has become increasingly important. Intracellular signaling pathways (SPs) play a crucial role in regulating numerous biological processes, including development, growth, aging, and disease. High-throughput transcriptomic methods, such as next-generation sequencing (NGS) and microarray analysis, have enabled the routine determination of gene expression levels, providing valuable insights into the activation of signaling pathways.

However, existing bioinformatics tools for analyzing signaling pathway activation (SPA) have limitations. Many methods are either proprietary, rely on machine learning with limited transparency, or fail to provide a comprehensive and quantitative estimation of pathway activation. This has hindered the ability to trace overall pathway activation signatures and quantitatively estimate the extent of SPA.

The present invention addresses these limitations by introducing a method for quick, informative, and large-scale screening of changes in SPA in cells and tissues. This method, the Cannabis Drug Efficiency Index (CDEI), utilizes high-throughput gene expression data and signaling pathway topology to assess the efficiency of cannabis drugs in treating various conditions, such as inflammation. The CDEI method provides a novel and effective approach for predicting and ranking the efficacy of cannabis extracts or other botanical compounds, thereby facilitating the development of personalized and non-toxic disease therapies.

## SUMMARY OF THE INVENTION

The present invention provides a method for assessing the efficiency of cannabis drugs in treating various conditions, particularly inflammation, by analyzing changes in signaling pathway activation (SPA) using high-throughput gene expression data. The method, referred to as the Cannabis Drug Efficiency Index (CDEI), integrates transcriptomic data with signaling pathway topology to predict and rank the efficacy of cannabis extracts or other botanical compounds.

The CDEI method involves the following steps:
1. **Data Collection**: Obtain high-throughput gene expression data from cells or tissues of individual patients and healthy individuals.
2. **Signaling Pathway Impact Analysis (SPIA)**: Convert the gene expression data into pathway perturbation scores using the SPIA method, which takes into account the topology of signaling pathways.
3. **Pathway Weight Calculation**: Calculate the pathway weight (wp) factor based on the mean SPIA scores of the case samples.
4. **Adjustment of Mean SPIA Scores**: Adjust the mean SPIA scores of each pathway by the weight factor.
5. **Statistical Testing**: Perform Student’s t-test to compare the adjusted mean SPIA scores of the case samples with the control samples.
6. **CDEI Calculation**: Calculate the CDEI for each drug for a specific disease using the t-values from the t-tests.
7. **Drug Ranking**: Rank the drugs according to their CDEI values to identify the most efficient drugs for treating the condition.

The CDEI method provides a comprehensive and quantitative approach for assessing the efficiency of cannabis drugs, enabling the selection of the most effective treatments for individual patients. The method is particularly useful for evaluating the anti-inflammatory properties of cannabis extracts and can be extended to other conditions and diseases.

## DETAILED DESCRIPTION OF THE EMBODIMENTS

### Overview of Signaling Pathway Impact Analysis (SPIA) Method

The Signaling Pathway Impact Analysis (SPIA) method is a key component of the CDEI approach. SPIA converts high-throughput gene expression data into pathway perturbation scores by considering the topology of signaling pathways. The method involves the following steps:

1. **Graph Representation**: Represent the signaling pathway as a graph \( G(V, E) \), where \( V \) is the set of graph nodes (genes) and \( E \) is the set of graph edges (interactions between genes).
2. **Perturbation Factor Calculation**: Calculate the perturbation factor (PF) for each gene in the pathway using the formula:
   \[
   PF(g) = \Delta E(g) + \sum_{\gamma \in U_g} \beta_{\gamma g} \cdot \frac{PF(\gamma)}{n_{\text{down}}(\gamma)}
   \]
   where:
   - \( \Delta E(g) \) is the signed log-fold-change of the gene \( g \) expression level in a given sample compared to the average value for the pool of normal samples.
   - \( U_g \) is the set of upstream genes for gene \( g \).
   - \( n_{\text{down}}(\gamma) \) is the number of downstream genes for gene \( \gamma \).
   - \( \beta_{\gamma g} \) is the interaction type between \( \gamma \) and \( g \): \( \beta_{\gamma g} = 1 \) if \( \gamma \) activates \( g \), and \( \beta_{\gamma g} = -1 \) if \( \gamma \) inhibits \( g \).

3. **Accuracy Vector Calculation**: Calculate the accuracy vector \( \mathbf{Acc} \) using the formula:
   \[
   \mathbf{Acc} = \mathbf{B} \cdot (\mathbf{I} - \mathbf{B})^{-1} \cdot \mathbf{\Delta E}
   \]
   where:
   - \( \mathbf{B} \) is the interaction matrix.
   - \( \mathbf{I} \) is the identity matrix.
   - \( \mathbf{\Delta E} \) is the vector of log-fold-changes for all genes in the pathway.

4. **Pathway Perturbation Score Calculation**: Calculate the overall score for pathway perturbation using the formula:
   \[
   \mathbf{SPIA} = \sum_{g} \mathbf{Acc}(g)
   \]

The SPIA method provides a robust and accurate way to assess the activation of signaling pathways based on gene expression data, taking into account the complex interactions between genes.

### Calculation of Cannabis Drug Efficiency Index (CDEI)

The Cannabis Drug Efficiency Index (CDEI) is a metric that quantifies the efficiency of a cannabis drug in treating a specific condition. The CDEI is calculated using the following steps:

1. **SPIA Calculation for Each Drug and Pathway**: Obtain the SPIA scores for each drug for each biological pathway using the SPIA method described above.
2. **Pathway Weight Calculation**: Calculate the pathway weight (wp) factor for each pathway based on the mean SPIA scores of the case samples:
   - For pathways with a positive mean SPIA score of the case samples:
     \[
     wp = \frac{\text{number of case samples with positive SPIA score}}{\text{total number of case samples}}
     \]
   - For pathways with a negative mean SPIA score of the case samples:
     \[
     wp = \frac{\text{number of case samples with negative SPIA score}}{\text{total number of case samples}}
     \]

3. **Adjustment of Mean SPIA Scores**: Adjust the mean SPIA scores of each pathway by the weight factor:
   \[
   \text{SPIA}_\mu = \text{mean(SPIA)} \cdot wp
   \]

4. **Statistical Testing**: Perform Student’s t-test to compare the adjusted mean SPIA scores of the case samples with the control samples. Calculate the t-values for the untreated case (U) and treated case (T) samples:
   - \( |t_U| \): Absolute t-value for the Student’s t-test for U-vs.-C profiles.
   - \( |t_T| \): Absolute t-value for the Student’s t-test for T-vs.-C profiles.

5. **CDEI Calculation**: Calculate the CDEI for each drug for a specific disease using the formula:
   \[
   \text{CDEI} = 2 \left( \frac{|t_U|}{|t_T| + |t_U|} - 0.5 \right)
   \]

The CDEI metric has the following properties:
- CDEI is a value between -1 and 1.
- CDEI is 0 if \( |t_T| \) and \( |t_U| \) are the same, indicating no drug efficiency.
- CDEI is 1 if \( |t_T| \) is 0, indicating perfect efficiency.
- CDEI is a value greater than 0 if \( |t_T| \) is smaller than \( |t_U| \), indicating positive efficiency.
- CDEI is a value less than 0 if \( |t_T| \) is larger than \( |t_U| \), indicating negative efficiency.

### Example of CDEI Calculations

The CDEI method was validated using several datasets, including transcriptomic data from human EpiDermFT 3D skin tissues, EpiOral tissues, and EpiIntestinal tissues. The following examples illustrate the application of the CDEI method to assess the anti-inflammatory properties of cannabis extracts.

#### Example #1: EpiDermFT 3D Skin Tissues

In this experiment, human EpiDermFT 3D skin tissues were exposed to UVC to induce inflammation and then treated with extracts of several cannabis cultivars. The untreated sample (U) had DMSO added to the media instead of extracts, and the control (C) sample had not been exposed to UVC. The CDEI calculations revealed that Extract #8 was the most efficient in restoring the transcriptome response after UVC exposure, with a CDEI score of 0.85. Extract #4 was less efficient, with a CDEI score of 0.30, while Extract #13 was not efficient, with a CDEI score of -0.10. Extract #12 was actually harmful, with a CDEI score of -0.50.

#### Example #2: EpiOral Tissues

In this experiment, human EpiOral tissues were treated with TNFα to induce inflammation and then treated with various cannabis extracts. The control sample was exposed to DMSO only. The CDEI calculations revealed that Extract #3 was the most efficient, with a CDEI score of 0.98. Extracts #5, #9, and #2 were also quite efficient, with CDEI scores of 0.90, 0.88, and 0.87, respectively. Extracts #8 and #4 were not very efficient, with CDEI scores of 0.10 and 0.16, respectively.

#### Example #3: EpiIntestinal Tissues

In this experiment, human EpiIntestinal tissues were treated with TNFα and then treated with various cannabis extracts. The control sample was exposed to DMSO only. The CDEI calculations revealed that Extract #5 was the most efficient, with a CDEI score of 0.95, followed by Extract #6, with a CDEI score of 0.85.

These examples demonstrate the effectiveness of the CDEI method in assessing the efficiency of cannabis extracts in reducing inflammation in different tissues. The CDEI method provides a quantitative and reliable approach for predicting and ranking the efficacy of cannabis drugs, facilitating the development of personalized and non-toxic disease therapies.