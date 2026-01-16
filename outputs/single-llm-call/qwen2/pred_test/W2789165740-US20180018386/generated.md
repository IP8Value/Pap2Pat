# DESCRIPTION

## STATEMENT OF GOVERNMENT INTEREST

This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Agency Name]. The government has certain rights in the invention.

## FIELD OF THE INVENTION

The present invention relates to the field of flow and mass cytometry data analysis, particularly to methods and systems for automated clustering and alignment of cell populations in high-dimensional (Hi-D) flow and mass cytometry data. More specifically, the invention provides a computationally efficient method for matching cell subsets between multiple samples, even when the location of populations varies significantly or when populations disappear or appear between samples.

## BACKGROUND

Flow and mass cytometry are powerful tools used in biomedical research to analyze the characteristics of individual cells in a sample. These techniques generate high-dimensional (Hi-D) data, which can be challenging to analyze due to the "curse of dimensionality." Traditional methods for cluster analysis, such as manual gating, are subjective and labor-intensive, making them unsuitable for Hi-D data sets. To address these challenges, various automated clustering and cluster matching methods have been developed.

Existing methods for cluster matching can be broadly categorized into two types: (1) separate clustering and matching, and (2) joint clustering and matching. The first approach involves clustering each sample individually and then aligning the resulting clusters. While this method is computationally efficient in low dimensions, it can fail when population locations vary significantly between samples or when populations disappear or appear between samples. The second approach, joint clustering and matching, addresses some of these issues by creating a Hi-D template of meta-clusters from pooled data. However, this method is computationally expensive and can be compromised by the curse of dimensionality.

To overcome the limitations of existing methods, the present invention introduces a novel cluster matching method based on the quadratic form (QF) distance measure. The QF distance is computationally efficient, robust to small changes in subset location and frequency, and can handle Hi-D data effectively. The invention provides a method for aligning cell subsets between multiple samples, even when the location of populations varies significantly or when populations disappear or appear between samples.

## BRIEF SUMMARY

The present invention provides a method for automated clustering and alignment of cell populations in high-dimensional (Hi-D) flow and mass cytometry data. The method is based on the quadratic form (QF) distance measure, which is computationally efficient and robust to small changes in subset location and frequency. The invention includes the following steps:

1. **Adaptive Binning**: Perform adaptive binning on the combined samples to create a set of bins that adapt to the structure of the data.
2. **Histogram Construction**: Construct histograms for each cluster in each sample using the bins created in the adaptive binning step.
3. **Dissimilarity Calculation**: Calculate the QF dissimilarity score between each pair of clusters from different samples.
4. **Cluster Matching**: Identify the pairs of clusters with the smallest dissimilarity scores as matched clusters.
5. **Merging Candidates**: Treat the remaining clusters as merging candidates and combine them with their nearest clusters in the same sample.
6. **Validation**: Recalculate the dissimilarity scores to validate the merging process and identify split or missing clusters.

The invention also provides a system for implementing the method, including software and hardware components. The system can be used to analyze flow and mass cytometry data in various research and clinical settings, facilitating the comparison of cell populations between control and test samples.

## DETAILED DESCRIPTION OF THE INVENTION

### Definitions

- **Adaptive Binning**: A method for dividing k-dimensional data into bins such that all bins contain the same number of events. The bins adapt to the structure of the data, resulting in bins of variable size.
- **Histogram**: A graphical representation of the distribution of numerical data, where the data is grouped into bins.
- **Quadratic Form (QF) Distance**: A dissimilarity measure that takes into account changes in both location and frequency of cell populations. It is computationally efficient and robust to small changes in subset location and frequency.
- **Cluster Matching**: The process of aligning cell subsets (clusters) present in multiple samples to facilitate comparison and analysis.
- **High-Dimensional (Hi-D) Data**: Data with a large number of dimensions, typically more than three or four, which can be challenging to analyze due to the "curse of dimensionality."

### Example 1—Workflow for Automated Clustering and Alignment of Cell Populations in Flow Cytometry Data

#### Step 1: Adaptive Binning
The first step in the QFMatch algorithm is to perform adaptive binning on the combined samples. Adaptive binning is a method for dividing k-dimensional data into bins such that all bins contain the same number of events. The bins adapt to the structure of the data, resulting in bins of variable size. This step ensures that the bins are representative of the data distribution and can handle variations in the data.

#### Step 2: Histogram Construction
Once the adaptive binning is complete, histograms are constructed for each cluster in each sample using the bins created in the adaptive binning step. Each histogram represents the distribution of events within a cluster and is normalized such that the total relative frequency is equal to 1.

#### Step 3: Dissimilarity Calculation
The next step is to calculate the QF dissimilarity score between each pair of clusters from different samples. The QF distance is a computationally efficient measure that takes into account changes in both location and frequency of cell populations. The dissimilarity score is calculated using the formula:
\[ D^2(\mathbf{h}, \mathbf{f}) = (\mathbf{h} - \mathbf{f})^T \mathbf{A} (\mathbf{h} - \mathbf{f}) \]
where \(\mathbf{h}\) and \(\mathbf{f}\) are the relative frequencies of the two histograms, and \(\mathbf{A}\) is a matrix that reflects the spatial dissimilarity between bins.

#### Step 4: Cluster Matching
The pairs of clusters with the smallest dissimilarity scores are identified as matched clusters. This step ensures that the most similar clusters are aligned, facilitating the comparison of cell populations between samples.

#### Step 5: Merging Candidates
The remaining clusters in each sample are treated as merging candidates and combined with their nearest clusters in the same sample. The dissimilarity scores are recalculated to validate the merging process and identify split or missing clusters.

#### Step 6: Validation
The final step is to validate the merging process by recalculating the dissimilarity scores. If the initial dissimilarity score decreases as a result of the merging process, it indicates that the cluster was split. If the dissimilarity score increases, it indicates missing clusters. This step ensures that the cluster matching is accurate and reliable.

### Example 2—Matching of Basophil Populations Between Patient Samples, Even when Marker Expression Levels Vary Between Patients

In this example, the QFMatch algorithm is applied to match basophil populations between patient samples, even when the marker expression levels vary significantly between patients. The dataset includes fluorescence flow cytometry data collected in a basophils activation study. The expression of the CD123 marker, which is used to identify peripheral blood basophils, commonly varies from patient to patient.

#### Step 1: Adaptive Binning
Adaptive binning is performed on the combined samples to create a set of bins that adapt to the structure of the data. This step ensures that the bins are representative of the data distribution and can handle variations in the data.

#### Step 2: Histogram Construction
Histograms are constructed for each cluster in each sample using the bins created in the adaptive binning step. Each histogram represents the distribution of events within a cluster and is normalized such that the total relative frequency is equal to 1.

#### Step 3: Dissimilarity Calculation
The QF dissimilarity score is calculated between each pair of clusters from different samples. The dissimilarity score takes into account changes in both location and frequency of cell populations, ensuring that the most similar clusters are aligned.

#### Step 4: Cluster Matching
The pairs of clusters with the smallest dissimilarity scores are identified as matched clusters. This step ensures that the most similar basophil populations are aligned, facilitating the comparison of cell populations between patients.

#### Step 5: Merging Candidates
The remaining clusters in each sample are treated as merging candidates and combined with their nearest clusters in the same sample. The dissimilarity scores are recalculated to validate the merging process and identify split or missing clusters.

#### Step 6: Validation
The final step is to validate the merging process by recalculating the dissimilarity scores. If the initial dissimilarity score decreases as a result of the merging process, it indicates that the cluster was split. If the dissimilarity score increases, it indicates missing clusters. This step ensures that the cluster matching is accurate and reliable.

### Example 3—Detection of Missing Lymphocyte Populations in the Peritoneal Cavity of RAG Knockout (RAG−/−) Mice

In this example, the QFMatch algorithm is applied to detect missing lymphocyte populations in the peritoneal cavity of RAG knockout (RAG−/−) mice. The dataset includes flow cytometry data from wild-type (BALB/c) and knockout (RAG−/−) mouse peritoneal cavity cells (PerC). The cell surface expression of CD5 and CD19 is used to identify T and B lymphocytes, respectively.

#### Step 1: Adaptive Binning
Adaptive binning is performed on the combined samples to create a set of bins that adapt to the structure of the data. This step ensures that the bins are representative of the data distribution and can handle variations in the data.

#### Step 2: Histogram Construction
Histograms are constructed for each cluster in each sample using the bins created in the adaptive binning step. Each histogram represents the distribution of events within a cluster and is normalized such that the total relative frequency is equal to 1.

#### Step 3: Dissimilarity Calculation
The QF dissimilarity score is calculated between each pair of clusters from different samples. The dissimilarity score takes into account changes in both location and frequency of cell populations, ensuring that the most similar clusters are aligned.

#### Step 4: Cluster Matching
The pairs of clusters with the smallest dissimilarity scores are identified as matched clusters. This step ensures that the most similar cell populations are aligned, facilitating the comparison of cell populations between wild-type and knockout mice.

#### Step 5: Merging Candidates
The remaining clusters in each sample are treated as merging candidates and combined with their nearest clusters in the same sample. The dissimilarity scores are recalculated to validate the merging process and identify split or missing clusters.

#### Step 6: Validation
The final step is to validate the merging process by recalculating the dissimilarity scores. If the initial dissimilarity score decreases as a result of the merging process, it indicates that the cluster was split. If the dissimilarity score increases, it indicates missing clusters. This step ensures that the cluster matching is accurate and reliable.

### Example 4—Cluster Matching of the Murine Lymphoid, Myeloid and Granuloid Lineages Between PerC and Spleen

In this example, the QFMatch algorithm is applied to match cell populations between the peritoneal cavity (PerC) and spleen of wild-type mice. The dataset includes flow cytometry data from PerC and spleen samples, and the side scatter (SSC-A) and CD11b surface marker measures are used to identify the lymphoid, myeloid, and granuloid subsets.

#### Step 1: Adaptive Binning
Adaptive binning is performed on the combined samples to create a set of bins that adapt to the structure of the data. This step ensures that the bins are representative of the data distribution and can handle variations in the data.

#### Step 2: Histogram Construction
Histograms are constructed for each cluster in each sample using the bins created in the adaptive binning step. Each histogram represents the distribution of events within a cluster and is normalized such that the total relative frequency is equal to 1.

#### Step 3: Dissimilarity Calculation
The QF dissimilarity score is calculated between each pair of clusters from different samples. The dissimilarity score takes into account changes in both location and frequency of cell populations, ensuring that the most similar clusters are aligned.

#### Step 4: Cluster Matching
The pairs of clusters with the smallest dissimilarity scores are identified as matched clusters. This step ensures that the most similar cell populations are aligned, facilitating the comparison of cell populations between PerC and spleen.

#### Step 5: Merging Candidates
The remaining clusters in each sample are treated as merging candidates and combined with their nearest clusters in the same sample. The dissimilarity scores are recalculated to validate the merging process and identify split or missing clusters.

#### Step 6: Validation
The final step is to validate the merging process by recalculating the dissimilarity scores. If the initial dissimilarity score decreases as a result of the merging process, it indicates that the cluster was split. If the dissimilarity score increases, it indicates missing clusters. This step ensures that the cluster matching is accurate and reliable.

### Example 5—Guided Gating of SSc Eosinophil Data

In this example, the QFMatch algorithm is applied to guided gating of SSc eosinophil data. The dataset includes flow cytometry data from systemic sclerosis (SSc) patients, and the goal is to identify and align eosinophil populations between samples.

#### Step 1: Adaptive Binning
Adaptive binning is performed on the combined samples to create a set of bins that adapt to the structure of the data. This step ensures that the bins are representative of the data distribution and can handle variations in the data.

#### Step 2: Histogram Construction
Histograms are constructed for each cluster in each sample using the bins created in the adaptive binning step. Each histogram represents the distribution of events within a cluster and is normalized such that the total relative frequency is equal to 1.

#### Step 3: Dissimilarity Calculation
The QF dissimilarity score is calculated between each pair of clusters from different samples. The dissimilarity score takes into account changes in both location and frequency of cell populations, ensuring that the most similar clusters are aligned.

#### Step 4: Cluster Matching
The pairs of clusters with the smallest dissimilarity scores are identified as matched clusters. This step ensures that the most similar eosinophil populations are aligned, facilitating the comparison of cell populations between SSc patients.

#### Step 5: Merging Candidates
The remaining clusters in each sample are treated as merging candidates and combined with their nearest clusters in the same sample. The dissimilarity scores are recalculated to validate the merging process and identify split or missing clusters.

#### Step 6: Validation
The final step is to validate the merging process by recalculating the dissimilarity scores. If the initial dissimilarity score decreases as a result of the merging process, it indicates that the cluster was split. If the dissimilarity score increases, it indicates missing clusters. This step ensures that the cluster matching is accurate and reliable.

The QFMatch algorithm provides a robust and computationally efficient method for automated clustering and alignment of cell populations in high-dimensional flow and mass cytometry data. The invention facilitates the comparison of cell populations between control and test samples, even when the location of populations varies significantly or when populations disappear or appear between samples. The method can be applied in various research and clinical settings, enhancing the accuracy and efficiency of flow and mass cytometry data analysis.