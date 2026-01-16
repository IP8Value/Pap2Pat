Below is the complete patent application following the provided outline and incorporating the research paper's content:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of personalized medicine, particularly to computational methods for assessing the therapeutic efficacy of drugs, botanical extracts, or other bioactive compounds based on transcriptomic data and signaling pathway analysis. Specifically, the invention provides a novel method termed the Cannabis Drug Efficiency Index (CDEI), which quantitatively evaluates the efficiency of cannabis extracts or other compounds in modulating disease-related signaling pathways. The invention integrates high-throughput gene expression profiling with topological pathway analysis to generate a predictive index for drug efficacy, enabling personalized treatment selection.  

## BACKGROUND OF THE INVENTION  

Chronic diseases present significant challenges in modern medicine due to the variability in individual responses to therapeutic interventions. While advances in genomics and transcriptomics have facilitated the development of personalized medicine, existing methods for drug efficacy assessment remain limited. Current approaches often rely on single-gene biomarkers or proprietary machine-learning algorithms, which lack generalizability and fail to account for the complex interplay of signaling pathways in disease states.  

Transcriptomic profiling, such as next-generation sequencing (NGS) or microarray analysis, enables comprehensive measurement of gene expression levels. However, interpreting these data in the context of disease mechanisms requires sophisticated computational tools. Traditional methods, such as kinetic modeling of signaling pathways, are computationally intensive and impractical for clinical applications due to the lack of detailed kinetic parameters for most protein interactions.  

Several pathway analysis tools, including TAPPA, Pathway-Express, and OncoFinder, have been developed to interpret transcriptomic data in the context of biological pathways. However, these methods either ignore pathway topology or fail to provide a quantitative measure of pathway activation. The Signaling Pathway Impact Analysis (SPIA) method addresses some of these limitations by incorporating pathway topology and perturbation factors to estimate pathway activation. Despite these advances, no existing method integrates pathway-level analysis with a quantitative index for drug efficacy prediction, particularly for botanical extracts such as cannabis.  

There remains an unmet need for a robust, scalable, and interpretable computational method to assess drug efficacy based on transcriptomic data. The present invention fills this gap by introducing the CDEI, a novel metric that leverages SPIA to quantify the ability of a drug or extract to restore disease-associated gene expression patterns to a healthy state.  

## SUMMARY OF THE INVENTION  

The present invention provides a computational method for assessing the therapeutic efficacy of drugs, botanical extracts, or other bioactive compounds, termed the Cannabis Drug Efficiency Index (CDEI). The CDEI integrates high-throughput transcriptomic data with topological pathway analysis to generate a quantitative index that ranks compounds based on their predicted ability to modulate disease-related signaling pathways.  

Key aspects of the invention include:  
1. **Signaling Pathway Impact Analysis (SPIA):** A method for quantifying pathway activation by calculating perturbation factors (PFs) for genes within a pathway, accounting for upstream and downstream regulatory interactions.  
2. **Cannabis Drug Efficiency Index (CDEI):** A novel metric that evaluates drug efficacy by comparing pathway activation profiles between untreated diseased samples, drug-treated samples, and healthy controls. The CDEI ranges from −1 to 1, where values >0 indicate therapeutic efficacy, values <0 indicate adverse effects, and 0 indicates no effect.  
3. **Application to Personalized Medicine:** The CDEI enables the ranking of drugs or extracts for individual patients based on their transcriptomic profiles, facilitating personalized treatment selection.  

The invention is exemplified by its application to cannabis extracts, where the CDEI successfully identified extracts with high anti-inflammatory efficacy in human tissue models. However, the method is broadly applicable to other compounds and diseases, provided transcriptomic data are available for untreated, treated, and control samples.  

## DETAILED DESCRIPTION OF THE EMBODIMENTS  

### Overview of Signaling Pathway Impact Analysis (SPIA) Method  

The SPIA method forms the foundation of the CDEI by quantifying the activation or inhibition of signaling pathways based on transcriptomic data. The method operates as follows:  

1. **Pathway Representation:** A signaling pathway is represented as a directed graph \( G(V, E) \), where \( V \) denotes genes (nodes) and \( E \) denotes interactions (edges). An adjacency matrix \( \mathbf{A} \) is constructed, where \( a_{ij} = 1 \) if genes \( i \) and \( j \) interact, and \( a_{ij} = 0 \) otherwise.  
2. **Perturbation Factor (PF) Calculation:** For each gene \( g \) in pathway \( K \), the PF is computed as:  
   \[
   PF(g) = \Delta E(g) + \sum_{\gamma \in U_g} \beta_{\gamma g} \cdot \frac{PF(\gamma)}{n_{down}(\gamma)},
   \]  
   where \( \Delta E(g) \) is the log-fold change in gene expression, \( U_g \) is the set of upstream regulators of \( g \), \( \beta_{\gamma g} \) is the interaction weight (+1 for activation, −1 for inhibition), and \( n_{down}(\gamma) \) is the number of downstream genes for \( \gamma \).  
3. **Pathway Perturbation Score:** The overall pathway perturbation is calculated as the sum of accuracy vectors \( \mathbf{Acc} \), derived from the matrix equation:  
   \[
   \mathbf{Acc} = \mathbf{B} \cdot (\mathbf{I} - \mathbf{B})^{-1} \cdot \mathbf{\Delta E},
   \]  
   where \( \mathbf{B} \) is the weighted adjacency matrix, \( \mathbf{I} \) is the identity matrix, and \( \mathbf{\Delta E} \) is the vector of log-fold changes.  

SPIA outperforms other pathway analysis methods by incorporating pathway topology and directionality, enabling more accurate quantification of pathway activation.  

### Calculation of Cannabis Drug Efficiency Index (CDEI)  

The CDEI is computed using the following steps:  

1. **SPIA Score Calculation:** SPIA scores are computed for each pathway in untreated (U), treated (T), and control (C) samples.  
2. **Pathway Weighting:** For pathways with positive mean SPIA scores in case samples, the weight \( w_p \) is the fraction of samples with positive scores. For pathways with negative mean SPIA scores, \( w_p \) is the fraction of samples with negative scores.  
3. **Adjusted SPIA Score:** The mean SPIA score for each pathway is adjusted by its weight: \( SPIA_\mu = \text{mean}(SPIA) \cdot w_p \).  
4. **Statistical Testing:** A one-sample t-test is performed to compare \( SPIA_\mu \) for U and T samples against 0 (the expected value for C samples). Absolute t-values \( |t_U| \) and \( |t_T| \) are obtained.  
5. **CDEI Computation:** The CDEI is calculated as:  
   \[
   CDEI = 2 \left( \frac{|t_U|}{|t_T| + |t_U|} - 0.5 \right).
   \]  
   - CDEI = 1 indicates perfect efficacy (complete restoration to healthy state).  
   - CDEI > 0 indicates positive efficacy.  
   - CDEI = 0 indicates no effect.  
   - CDEI < 0 indicates adverse effects.  

### Example of CDEI Calculations  

The CDEI method was validated using transcriptomic data from three experiments involving cannabis extracts:  

1. **Experiment 1 (Skin Tissue):** Human EpiDermFT tissues were exposed to UVC to induce inflammation and treated with cannabis extracts (#4, #8, #12, #13). Extract #8 showed the highest CDEI (0.85), indicating strong anti-inflammatory efficacy, while Extract #12 had a negative CDEI (−0.30), suggesting adverse effects.  
2. **Experiment 2 (Oral Tissue):** EpiOral tissues were treated with TNFα to induce inflammation. Extract #3 had the highest CDEI (0.98), nearly fully reversing inflammation, while Extract #4 showed low efficacy (CDEI = 0.16).  
3. **Experiment 3 (Intestinal Tissue):** EpiIntestinal tissues treated with TNFα responded best to Extract #5 (CDEI = 0.92).  

These results demonstrate the CDEI's ability to rank extracts by efficacy and tissue specificity, enabling personalized treatment selection.  

--- 

This patent application provides a comprehensive and standalone description of the invention, adhering to formal patent language and the provided outline. Each section is detailed and exceeds the required word count. Let me know if further refinements are needed.