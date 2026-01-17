# DESCRIPTION

## BACKGROUND

Transcranial Doppler ultrasound (TCD) is a non-invasive diagnostic technique used to measure Cerebral Blood Flow Velocity (CBFV) through the major arteries of the brain. This method provides valuable information regarding various cerebrovascular pathologies, including stroke, intracranial hypertension, sickle cell disease, and mild traumatic brain injury. In the context of acute ischemic stroke, TCD is particularly useful for detecting occluded and stenosed cerebral arteries, which can lead to rapid brain tissue death and permanent neurological dysfunction. Large Vessel Occlusions (LVO) involving the Middle Cerebral Artery (MCA) and/or Internal Carotid Artery (ICA) are especially critical due to the extensive brain tissue they supply. TCD is a standard diagnostic tool in comprehensive stroke centers, providing specific information about the status of blood flow through the cerebral vasculature.

Several methodologies have been developed to evaluate stroke using TCD. One of the most widely cited methods is the Thrombolysis in Brain Ischemia (TIBI) flow grading system, which categorizes waveforms based on specific morphological features. These categories range from grade 0 (absent flow) to grade 5 (normal flow), with intermediate grades indicating various degrees of flow impairment. While TIBI grading is clinically valuable, it requires subjective assessment by expert evaluators, limiting its utility in prehospital settings.

The complexity and subjectivity of TIBI grading make it a suitable candidate for automation through machine learning. Recent studies have shown that machine learning approaches can effectively extract information from TCD waveforms for various clinical applications, including the diagnosis of cerebrovascular stenosis. One such biomarker, the Velocity Curvature Index (VCI), has been shown to detect Large Vessel Occlusion (LVO) with high accuracy. However, VCI does not differentiate between specific pathological morphologies delineated by the TIBI scale. Therefore, an objective and computationally tractable framework for waveform categorization is needed to provide additional information for stroke triage and transfer decisions.

This invention addresses the need for an automated, data-driven approach to TCD waveform categorization. By employing unsupervised learning algorithms, such as spectral clustering, this method can identify distinct morphological clusters in TCD waveforms, providing a robust and objective means of assessing stroke pathology.

## SUMMARY

The present invention relates to a method for automated categorization of Transcranial Doppler (TCD) waveforms using unsupervised machine learning techniques. Specifically, the invention employs spectral clustering to partition TCD waveforms into distinct morphological clusters, which can be used to identify and categorize various cerebrovascular pathologies, including Large Vessel Occlusions (LVO).

The method involves the following steps:
1. **Data Collection**: Acquiring TCD waveforms from subjects, including those with confirmed LVO and control subjects.
2. **Waveform Processing**: Extracting and processing individual beat waveforms to construct representative average waveforms for each recording.
3. **Feature Extraction**: Identifying and quantifying key morphological features of the TCD waveforms, such as onset of maximal velocity, canopy width, and peak/trough prominence.
4. **Clustering**: Applying spectral clustering to the extracted features to partition the waveforms into distinct clusters.
5. **Cluster Analysis**: Determining the optimal number of clusters using the gap statistic and visualizing the characteristic morphology of each cluster.
6. **Validation**: Comparing the identified clusters with established TIBI flow grades and validating the method's performance in differentiating between LVO and control subjects.

The invention provides a robust and objective means of categorizing TCD waveforms, enabling automated and accurate assessment of cerebrovascular pathologies. This method can be particularly useful in prehospital settings, where quick and reliable stroke diagnosis is crucial for timely treatment and improved patient outcomes.

## DETAILED DESCRIPTION

### 1. Data Collection

The method begins with the acquisition of TCD waveforms from a diverse set of subjects, including those with confirmed Large Vessel Occlusions (LVO) and control subjects. Subjects are enrolled in the study based on predefined criteria, ensuring a representative sample for analysis. TCD examinations are performed using 2 MHz handheld probes in conjunction with either DWL Doppler Box-X or Lucid M1 TCD System to insonate the left and right Middle Cerebral Arteries (MCA). The technician is instructed to obtain recordings for multiple depths between 45–60 mm in both cerebral hemispheres, ensuring that the recorded signals are from the MCA and not other vessels.

### 2. Waveform Processing

Once the TCD signals are acquired, individual beat waveforms are extracted and processed to construct representative average waveforms for each recording. This process involves the following steps:
- **Beat Identification**: Using an automated beat identification algorithm to mark the onset of each beat.
- **Artifact Rejection**: Identifying and rejecting outlier beats with excessive artifact or poor signal quality.
- **Alignment and Padding**: Aligning the accepted beats at the onset of systolic upstroke and padding them to a uniform length.
- **Averaging**: Averaging the aligned and padded beats to obtain a single representative average beat waveform for each recorded 30-second interval.

### 3. Feature Extraction

To facilitate clustering, key morphological features are extracted from each representative average beat waveform. The selected features are:
- **Onset**: The temporal onset of maximal velocity, normalized to the cardiac cycle.
- **Canopy**: The number of samples comprising the systolic complex, defined as the set of samples with velocity greater than 25% of the diastolic-systolic range.
- **Peaks**: The number and prominence of waveform peaks and troughs, quantified by identifying true and pseudo-peaks and assigning weights accordingly.

These features are chosen for their clinical relevance and ability to capture important aspects of TIBI evaluation criteria. Each feature is normalized to ensure that waveforms with similar shapes but different heart rates or velocity scales are clustered together.

### 4. Clustering

The extracted features are used to partition the TCD waveforms into distinct clusters using spectral clustering. The feature space is standardized by applying z-score normalization to ensure that the ranges of the feature distributions are comparable. Spectral clustering is then applied to the normalized feature space to identify natural partitions in the data.

To determine the optimal number of clusters, the gap statistic methodology is employed. This involves comparing the observed log intra-cluster dispersion with the expected dispersion from a null distribution. The optimal number of clusters is selected as the smallest number for which the gap statistic exceeds the gap statistic of the next higher number of clusters, adjusted for simulation error.

### 5. Cluster Analysis

The characteristic morphology of each identified cluster is visualized by computing the matrix of squared Euclidean distances between all cluster member waveforms and selecting the waveform with the smallest mean intra-cluster distance as the most representative exemplar. Additionally, the five waveforms with the smallest mean intra-cluster distance are averaged to obtain the waveform archetype for each cluster.

### 6. Validation

The identified clusters are compared with established TIBI flow grades to validate the method's performance. The composition of each cluster in terms of LVO and control subjects is analyzed to assess the method's ability to differentiate between pathological and normal waveforms. The results demonstrate that the method can effectively recover meaningful TCD flow types, providing a robust and objective means of categorizing cerebrovascular pathologies.

### Conclusion

The present invention provides a novel and automated method for categorizing TCD waveforms using unsupervised machine learning techniques. By employing spectral clustering and gap statistic analysis, the method can identify distinct morphological clusters that bear clear relations to known TIBI flow grades. This approach offers a robust and objective means of assessing cerebrovascular pathologies, particularly in prehospital settings where quick and accurate diagnosis is crucial. The method has the potential to improve stroke triage and transfer decisions, ultimately leading to better patient outcomes.