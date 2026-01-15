Here is the patent application following your outline:

# DESCRIPTION  

## STATEMENT OF GOVERNMENT INTEREST  

The invention described herein was made with government support under Grant No. [Grant Number] awarded by [Agency Name]. The government has certain rights in the invention.  

## FIELD OF THE INVENTION  

The present invention relates generally to the fields of data analysis and visualization, particularly in the context of biological sample comparison. More specifically, the invention pertains to automated methods for matching clusters in high-dimensional cytometry data, including flow cytometry and mass cytometry. The invention further relates to systems and methods for visualizing multidimensional data to guide sequential gating processes in cytometry analysis.  

## BACKGROUND  

The need for accurate and efficient sample comparison in biomedical studies has become increasingly critical with the advent of high-throughput data collection techniques. Most flow and mass cytometry applications rely on comparisons between control and test samples to identify dissimilarities arising from drug treatments, disease progression, or therapeutic responses. Traditionally, such comparisons have been performed by manually gating data into arbitrary clusters, a method that is both subjective and labor-intensive, particularly when applied to high-dimensional datasets.  

Manual gating approaches suffer from significant limitations, including subjectivity, inconsistency, and an inability to scale with modern high-dimensional data. These limitations have motivated the development of automated clustering and cluster matching methods. However, both cluster identification and cluster matching are highly challenging due to the "curse of dimensionality," a well-known statistical problem that compromises both statistical validity and computational performance in high-dimensional data analysis.  

Existing methods for cluster matching fall into two broad categories, each with distinct limitations. The first approach involves clustering one sample at a time and subsequently aligning the cell subsets present in multiple samples. While this method allows for fast computational implementations in low dimensions, it fails when population locations vary significantly between samples or when populations disappear or appear between samples. Furthermore, this approach is particularly vulnerable to the curse of dimensionality when applied in high-dimensional settings.  

The second approach attempts to address these limitations by creating a high-dimensional template of meta-clusters in which all sample data are pooled, simultaneously clustered, and then matched. While this method alleviates some problems associated with the first approach, it remains computationally expensive and relies heavily on fitting mathematical models to datasets. The feasibility of such model fitting is dramatically affected by the curse of dimensionality, as the number of parameter combinations increases exponentially with dimensionality.  

These limitations highlight the need for improved cluster matching methods that can accommodate real-world flow and mass cytometry data. There is a particular need for methods that can handle cases where population locations vary significantly between samples or where populations disappear or appear between samples. Additionally, there is a need for user guidance in sequential gating processes to improve the accuracy and efficiency of cytometry data analysis.  

## BRIEF SUMMARY  

The present invention provides a novel method for multidimensional cluster matching in cytometry data. The method introduces a multivariate extension of the quadratic form (QF) distance for comparing flow cytometry samples, overcoming key limitations of existing approaches. The QF distance possesses properties that make it particularly suitable for biological comparisons, including metric properties, continuity, computational efficiency, and insensitivity to small changes caused by instrument noise.  

The invention further provides a system for matching clusters in cytometry data, comprising memory and storage for data, a processor, and various modules including an adaptive binning module, a dissimilarity module, a matching and merging candidate identification module, and a determination module. These modules work in concert to perform multivariate adaptive binning, generate histograms for identified clusters, determine dissimilarity scores, and identify matched clusters and merging candidates.  

Additionally, the invention includes a method for rendering a graphical user interface to guide sequential gating processes. The interface comprises a first interactive display and a second interactive display, which may include two-dimensional plots, single parameter charts or graphs, and graphical indications of thresholds. The interface allows users to select guidance features and modify displays accordingly, facilitating more accurate and efficient gating.  

The method for matching clusters in cytometry data involves several key steps. First, sample data is obtained, and clusters are identified within the data. Multivariate adaptive binning is then performed on the combined samples, and the resulting binning pattern is applied separately to each sample. Histograms are generated for the identified clusters, and dissimilarity scores are calculated for all combinations of clusters using the QF distance measure.  

The QF dissimilarity score is calculated using a spatial dissimilarity matrix that reflects the Euclidean distance between bin centers. This approach ensures that the score increases not only with the magnitude of differences between histograms but also with the spatial distance of non-zero elements, providing a biologically meaningful quantification of differences between samples.  

Matched clusters are identified based on the lowest dissimilarity scores, and merging candidates are determined for remaining clusters. The method further involves recalculating dissimilarity scores after merging candidates are combined with their nearest clusters, allowing for the identification of split or missing clusters. This process accommodates cases where population locations vary significantly between samples or where populations disappear or appear between samples.  

The invention also provides methods for performing multivariate adaptive binning in k-dimensions, dividing data into k-dimensional bins, and determining dissimilarity scores for each combination of clusters. The system calculates additional information, such as relative frequencies and distances between geometric means, to facilitate cluster matching and track population changes.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Definitions  

For purposes of this invention, the following terms shall have the meanings set forth below:  

An "item" refers to any biological particle that may be analyzed using cytometry, including but not limited to cells, cellular components, or synthetic particles.  

"Gating" refers to the process of selecting subsets of items based on their measured characteristics in cytometry data.  

A "gate" is a boundary or threshold applied to cytometry data to define a subset of items.  

A "marker" is a detectable molecule used to identify specific characteristics of an item, typically through binding to specific targets.  

A "reagent" is any chemical substance used to prepare samples for cytometry analysis.  

A "stain" refers to the application of a marker to a sample to enable detection of specific characteristics.  

A "staining reagent" is a composition comprising one or more markers used for staining samples.  

Where the context permits, singular forms shall include plural forms and vice versa. The phrase "in one embodiment" as used herein does not necessarily refer to the same embodiment, though it may.  

"Cluster matching" refers to the process of identifying corresponding clusters across different cytometry samples.  

"Clustering" refers to the process of grouping items in cytometry data based on their measured characteristics.  

"Aligning cell subsets" refers to the process of identifying corresponding cell populations across different cytometry samples.  

"Population matching" refers to the process of identifying corresponding cell populations across different cytometry samples, encompassing both separate clustering and matching approaches and joint clustering and matching approaches.  

"Separate clustering and matching" refers to approaches where clustering is performed independently on each sample before attempting to match clusters across samples.  

"Joint clustering and matching" refers to approaches where samples are pooled and simultaneously clustered to create meta-clusters that serve as templates for matching.  

A "dissimilarity measure" is a quantitative representation of the difference between two clusters or samples.  

"Distance metrics" are mathematical formulations used to quantify dissimilarity between clusters or samples.  

The "Earth Mover's Distance" (EMD) is a specific distance metric that measures the minimum amount of "work" required to transform one distribution into another.  

The "quadratic form distance measure" is a distance metric that compares two histograms using a spatial dissimilarity matrix.  

The "QF dissimilarity score" is a specific implementation of the quadratic form distance measure, calculated as described herein.  

"Multi-dimensional cluster matching" refers to the process of matching clusters across samples using multiple dimensions of cytometry data simultaneously.  

"High-dimensional cluster matching" refers to multi-dimensional cluster matching where the number of dimensions is sufficiently large to present computational challenges.  

"Two-dimensional density-based merging" refers to a clustering method that identifies clusters based on density in two-dimensional projections of cytometry data.  

"Cluster identification methods" encompass any techniques used to identify distinct cell populations in cytometry data.  

"Template formation" refers to the process of creating a reference set of clusters for matching purposes.  

"Meta-cluster formation" refers to the process of creating higher-level clusters that represent distinct biologically-relevant cell types across multiple samples.  

"Higher-level template formation" refers to the process of creating templates at increasing levels of abstraction for matching purposes.  

A "resulting template" is the final output of template formation processes, used for matching clusters across samples.  

"Relative frequency" refers to the proportion of events in a cluster compared to the total number of events in a sample.  

A "spatial dissimilarity matrix" is a matrix used in calculating the QF dissimilarity score, where elements reflect the spatial distance between bins.  

The invention provides a comprehensive system and method for cluster matching in cytometry data. The system comprises several key components, including an adaptive binning module that performs multivariate adaptive binning on combined samples, a dissimilarity module that calculates QF dissimilarity scores, a matching and merging candidate identification module that identifies matched clusters and merging candidates, and a determination module that determines whether merging candidates correspond to split or missing clusters.  

The method involves obtaining first sample data and second sample data, identifying clusters in each sample, performing multivariate adaptive binning on the combined data, applying the combined binning pattern separately to each sample, generating histograms for identified clusters, determining dissimilarity scores for combinations of clusters, identifying matched clusters and merging candidates, determining the lowest dissimilarity score for each cluster, and identifying whether merging candidates correspond to split or missing clusters.  

The QF dissimilarity score is calculated using the equation:  
D²(h,f) = (h-f)ᵀA(h-f) = ΣΣaᵢⱼ(hᵢ-fᵢ)(hⱼ-fⱼ)  
where h and f are relative frequency vectors for two histograms, and A = [aᵢⱼ] is a spatial dissimilarity matrix with elements:  
aᵢⱼ = 1 - dMᵢⱼ/dmax  
where dMᵢⱼ is the Euclidean distance between centers of mass of bins i and j, and dmax is the maximum value of all dMᵢⱼ.  

The system further includes modules for rendering a graphical user interface, comprising a first interactive display module and a second interactive display module. These modules enable the display of two-dimensional plots, single parameter charts or graphs, and graphical indications of thresholds. The interface allows users to select guidance features and modify displays accordingly, facilitating more accurate and efficient gating.  

The invention may be implemented in various computing environments, including cloud computing environments with software as a service (SaaS) delivery models. The system may comprise digital electronic circuitry, computer hardware, firmware, software, or combinations thereof. The invention may be embodied as a computer program product stored on a machine-readable medium, including computer storage media and communication media.  

The system may be implemented using special purpose logic circuitry, such as an FPGA (field programmable gate array) or an ASIC (application-specific integrated circuit). The computing system may include a processor, main memory, static memory, and a bus connecting various components. User interface components may include a video display unit, alphanumeric input device, UI navigation device, and disk drive unit. Network interface devices may enable communication with other devices via a communications network.  

### Example 1—Workflow for Automated Clustering and Alignment of Cell Populations in Flow Cytometry Data  

In this example, the invention is applied to automate clustering and alignment of cell populations in flow cytometry data. The data preprocessing step involves compensating the data, applying a Logicle transformation, and clustering the transformed data using a two-dimensional density-based merging algorithm.  

The method 100 for cluster matching is then applied to the preprocessed data. Adaptive binning is performed on the combined samples, and the resulting binning pattern is applied separately to each sample. Histograms are generated for identified clusters, and QF dissimilarity scores are calculated for all combinations of clusters.  

The results demonstrate successful alignment of cell populations across samples, even when population locations vary significantly or when populations disappear or appear between samples. The method accurately identifies matched clusters and correctly classifies merging candidates as either split or missing clusters.  

### Example 2—Matching of Basophil Populations Between Patient Samples, Even when Marker Expression Levels Vary Between Patients  

This example demonstrates the invention's ability to match basophil populations between patient samples despite significant variations in marker expression levels. Data collection involves staining peripheral blood samples with a panel including CD123, followed by flow cytometry analysis.  

The method 100 is applied to preprocessed data from multiple patients. The results show successful alignment of basophil populations despite variations in CD123 expression levels (MFI ranging from 1033 to 6672) and differences in basophil frequencies between patients. The method accurately matches clusters while correctly identifying variations in geometric means between matched populations.  

### Example 3—Detection of Missing Lymphocyte Populations in the Peritoneal Cavity of RAG Knockout (RAG−/−) Mice  

This example demonstrates the invention's ability to detect missing lymphocyte populations in knockout mice. Wild-type (BALB/c) and knockout (RAG−/−) mouse peritoneal cavity cells are analyzed based on surface expression of CD5 and CD19.  

Application of method 100 reveals the complete absence of T and B lymphocytes in RAG−/− mice, as evidenced by unmatched clusters that cannot be merged without increasing dissimilarity scores. The results confirm the expected biological phenotype and demonstrate the method's sensitivity in detecting missing populations.  

### Example 4—Cluster Matching of the Murine Lymphoid, Myeloid and Granuloid Lineages Between PerC and Spleen  

This example applies the invention to match immune cell subsets between peritoneal cavity (PerC) and spleen samples in mice. A 12-parameter flow cytometry panel identifies lymphoid, myeloid, and granuloid subsets.  

Method 100 successfully aligns shared immune cell subsets between PerC and spleen while correctly identifying tissue-specific subsets as unmatched. The results demonstrate the method's ability to handle complex matching tasks involving multiple cell lineages and significant tissue differences.  

### Example 5—Guided Gating of SSc Eosinophil Data  

This example demonstrates the invention's guidance window functionality for gating eosinophil data. The guidance window provides interactive displays showing two-dimensional plots and single parameter charts, allowing users to select subsets and modify gating thresholds.  

The results show effective use of the guidance window for eosinophil gating, including overlay of sample data with FMO control sample data. The interface facilitates accurate gating by providing real-time visual feedback and threshold adjustments.  

The invention has been described with reference to various specific and preferred embodiments and techniques. However, it should be understood that many variations and modifications may be made while remaining within the spirit and scope of the invention.