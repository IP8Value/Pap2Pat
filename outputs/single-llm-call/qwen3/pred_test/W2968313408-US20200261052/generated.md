# DESCRIPTION

## BACKGROUND

- introduce TCD and its applications

Transcranial Doppler ultrasound is a noninvasive diagnostic technique that measures cerebral blood flow velocity through the major intracranial arteries using low-frequency acoustic energy transmitted through the temporal bone window. This method provides real-time, continuous assessment of hemodynamic patterns in the cerebral circulation and is widely employed in clinical settings for the evaluation of cerebrovascular conditions including acute ischemic stroke, intracranial hypertension, sickle cell disease, and traumatic brain injury. The pulsatile nature of the resulting velocity waveform reflects the cardiac cycle and is sensitive to changes in vascular resistance, compliance, and patency. In the context of stroke care, transcranial Doppler is particularly valuable for detecting occlusions or severe stenoses in the middle cerebral artery and internal carotid artery, which are responsible for the majority of large-vessel ischemic events. The ability to rapidly assess cerebral perfusion without ionizing radiation or contrast agents makes transcranial Doppler an ideal tool for emergency triage, intraoperative monitoring, and longitudinal follow-up in neurocritical care. Its portability and compatibility with bedside use further enhance its utility in prehospital, emergency department, and intensive care unit environments. Despite its widespread adoption, the interpretation of transcranial Doppler waveforms has historically relied on subjective visual inspection by trained clinicians, leading to variability in diagnostic accuracy and limiting its potential for standardized, scalable deployment in resource-constrained or non-specialist settings.

## SUMMARY

- process signal to identify morphological variables

The system processes raw ultrasound data to extract a series of quantifiable morphological variables from each cerebral blood flow velocity waveform, enabling objective characterization of hemodynamic patterns without reliance on clinician interpretation. Each waveform is first isolated from the continuous signal stream by identifying individual cardiac cycles through automated detection of systolic onset points, followed by rejection of waveforms containing excessive noise or artifact based on cross-correlation and beat-length consistency metrics. Accepted waveforms are then aligned at their systolic onset and averaged to produce a representative beat for each recording depth, ensuring robust signal fidelity while minimizing transient artifacts. The averaged waveform is resampled to a uniform temporal resolution and normalized in both time and velocity dimensions to eliminate confounding effects of heart rate variability and absolute velocity magnitude, thereby preserving only the shape characteristics that reflect underlying vascular physiology. This preprocessing pipeline ensures that subsequent analysis is focused exclusively on morphological structure rather than extraneous signal properties.

- categorize waveform based on morphological variables

The normalized waveforms are analyzed to compute three distinct morphological variables: onset, canopy, and peaks. The onset variable quantifies the temporal position of maximal systolic velocity relative to the start of the cardiac cycle, capturing the timing of flow acceleration. The canopy variable measures the duration of the systolic phase during which velocity exceeds a defined threshold relative to the diastolic-systolic range, reflecting the extent of the systolic complex. The peaks variable evaluates the number and prominence of secondary velocity maxima and minima within the systolic and early diastolic phases, assigning weighted contributions to both clearly defined peaks and subtle inflections that may indicate disturbed flow dynamics. These three variables collectively form a three-dimensional feature space that encodes the essential morphological signatures of cerebral flow patterns. Each waveform is then mapped into this space as a single point, allowing for systematic comparison across subjects and conditions.

- visualize categorized waveform

The categorized waveforms are rendered in a three-dimensional graphical representation where each axis corresponds to one of the morphological variables. Cluster membership is visually encoded through color and spatial density, enabling immediate recognition of distinct waveform families. Representative waveforms from each cluster are displayed alongside their corresponding coordinates in the feature space, allowing clinicians to correlate abstract numerical groupings with tangible waveform shapes. The visualization interface permits interactive rotation, zooming, and filtering by subject group or clinical condition, facilitating intuitive exploration of relationships between morphology and pathology.

- determine probability of waveform belonging to category

For each input waveform, the system computes a probabilistic assignment to each of the predefined categories based on its position within the feature space and the statistical distribution of known cluster members. This probability is derived from a kernel density estimation of the cluster boundaries, accounting for overlapping regions and fuzzy transitions between categories. The output includes a ranked list of category likelihoods, with the most probable classification highlighted and accompanied by a confidence score. This probabilistic framework allows for nuanced decision-making, particularly in cases where waveform morphology falls near the boundary between two clinically distinct categories.

- define morphological variables

The morphological variables are rigorously defined as quantitative descriptors derived from the temporal and amplitude characteristics of the cerebral blood flow velocity waveform. The onset variable is calculated as the normalized time point at which the maximum velocity occurs within the cardiac cycle, expressed as a fraction of total systolic duration. The canopy variable is determined by counting the number of discrete time samples in which velocity exceeds 25% of the difference between systolic peak and diastolic minimum, thereby capturing the breadth of the systolic complex. The peaks variable is computed as the sum of weighted contributions from all local maxima and minima identified in the waveform, where true peaks (zero-crossings of the first derivative) are assigned a weight of one, and pseudo-peaks (inflections with near-zero derivative magnitude) are assigned weights inversely proportional to the magnitude of their slope, with a threshold of 0.01 used to define the boundary between significant and negligible inflections.

- map variables to axes

The three morphological variables—onset, canopy, and peaks—are mapped to the three orthogonal axes of a Cartesian coordinate system, forming a three-dimensional feature space where each point represents a unique waveform. The onset variable is assigned to the x-axis, the canopy variable to the y-axis, and the peaks variable to the z-axis. This spatial arrangement enables geometric interpretation of waveform relationships, where proximity in space corresponds to morphological similarity. The axes are scaled to equalize the variance of each variable across the dataset, ensuring that no single feature dominates the clustering outcome due to differences in measurement range.

- categorize variables

Each morphological variable is partitioned into discrete ranges based on empirical distributions observed across the training cohort. These ranges are not predefined by clinical heuristics but are instead derived from the natural clustering structure of the data. For onset, ranges correspond to early, intermediate, and late systolic velocity maxima; for canopy, they reflect narrow, moderate, and wide systolic complexes; and for peaks, they distinguish weak, moderate, and strong secondary structures. These partitions are used to define the boundaries of the final categories, which are themselves determined by the joint distribution of all three variables.

- define categories and corresponding areas

Four distinct categories are defined in the three-dimensional feature space, each corresponding to a cluster of waveforms with shared morphological characteristics. Category I encompasses waveforms with early onset, wide canopy, and strong peaks, representing normal flow. Category II includes waveforms with late onset, wide canopy, and strong peaks, indicative of altered peripheral resistance. Category III consists of waveforms with early onset, narrow canopy, and weak peaks, strongly associated with large vessel occlusion. Category IV comprises waveforms with late onset, wide canopy, and weak peaks, corresponding to blunted flow patterns. Each category is represented as a contiguous region in the feature space, bounded by the statistical extent of its cluster members and defined by a probability density threshold that ensures high within-category homogeneity.

- categorize waveform based on variable values

A novel waveform is categorized by first computing its three morphological variables and then determining its position within the predefined three-dimensional feature space. The system evaluates the Euclidean distance between the waveform’s coordinates and the centroid of each category, as well as its likelihood under the multivariate Gaussian distribution fitted to each cluster. The category with the highest posterior probability is assigned as the classification, with a confidence score derived from the ratio of the highest to second-highest likelihood. This process is fully automated and requires no manual input, enabling real-time classification during live data acquisition.

- relate categories to blood flow and pathologies

Category I is associated with normal cerebral perfusion, characterized by robust pulsatility and timely systolic acceleration, consistent with intact vascular compliance and unobstructed flow. Category II is linked to conditions of increased downstream resistance or altered reflection patterns, such as chronic hypertension or distal stenosis, where systolic timing is delayed but amplitude remains preserved. Category III is strongly correlated with acute large vessel occlusion, where early systolic acceleration is preserved but subsequent flow is suppressed due to proximal obstruction, resulting in a truncated, low-amplitude waveform. Category IV corresponds to blunted flow patterns typically seen in moderate stenosis or impaired cardiac output, where systolic acceleration is delayed and secondary peaks are diminished. These associations are validated by comparison with gold-standard imaging and clinical outcomes, establishing a direct link between morphology and pathophysiology.

- describe device for visualizing categorization

A dedicated device is configured to receive raw ultrasound data from a transcranial Doppler probe, process the signal to extract morphological variables, classify the waveform into one of four categories, and display the results in real time on a high-resolution touchscreen interface. The device includes an integrated processor, memory storage for reference cluster models, and a user interface that overlays the classified waveform with its position in the three-dimensional feature space, along with a confidence metric and clinical interpretation. The system is designed for use in emergency, prehospital, and critical care settings, providing immediate, objective feedback to guide triage, transport, and treatment decisions.

## DETAILED DESCRIPTION

- introduce detailed description

The following section provides a comprehensive description of the systems, methods, and components employed to automate the classification of transcranial Doppler waveforms based on morphological characteristics. This description includes the acquisition of raw ultrasound data, the preprocessing and normalization of waveforms, the extraction of morphological variables, the application of unsupervised clustering algorithms, the derivation of cluster archetypes, and the implementation of a real-time visualization and classification system. All components are integrated into a cohesive workflow that transforms subjective visual assessment into an objective, reproducible, and scalable diagnostic tool.

- provide context for appended drawings

The appended drawings illustrate the structural and functional components of the system, including the probe configuration, robotic positioning mechanism, controller architecture, and graphical user interface. Figure 1 depicts the three morphological variables overlaid on a representative waveform. Figure 2 shows the gap statistic analysis used to determine the optimal number of clusters. Figure 3 presents the three-dimensional feature space partitioned into four clusters, with corresponding archetype waveforms. Figure 4 displays the normalized waveforms of each cluster, demonstrating their distinct morphological signatures. Figure 5 illustrates the hardware configuration of the visualization device, including the headset, probe, and controller. Figure 6 outlines the method for automated waveform categorization. Figure 7 details the process for visualizing categorized waveforms with probability scores.

- describe systems and methods for automated flow type classification

The system employs a fully automated pipeline for the classification of cerebral blood flow velocity waveforms into morphologically distinct categories. The process begins with the acquisition of ultrasound signals from the middle cerebral artery using a 2 MHz transducer. The raw Doppler signal is digitized and processed to isolate individual cardiac cycles, which are then averaged to produce a representative waveform for each recording depth. These waveforms undergo temporal and velocity normalization to eliminate confounding factors related to heart rate and absolute velocity. Three morphological variables—onset, canopy, and peaks—are computed from each normalized waveform and used to construct a three-dimensional feature space. Spectral clustering with a radial basis kernel is applied to partition the feature space into clusters, with the optimal number of clusters determined using the gap statistic criterion. Cluster archetypes are derived by averaging the five most representative waveforms within each cluster. These archetypes serve as reference templates for classifying new waveforms.

- introduce objective assessment of TCD morphology using machine learning

The system introduces an objective, data-driven framework for assessing transcranial Doppler morphology that replaces subjective visual grading with automated, statistically validated classification. Unlike traditional methods that rely on clinician interpretation of TIBI flow grades, this approach derives categories directly from the empirical distribution of waveform morphology across a diverse cohort of patients and controls. The use of unsupervised learning ensures that the resulting categories reflect natural groupings in the data rather than predefined clinical assumptions. This method eliminates inter-rater variability and enables consistent application across clinical settings, regardless of operator experience.

- extract morphological variables from waveforms

Morphological variables are extracted from each normalized waveform using algorithmic procedures that quantify temporal and amplitude features without requiring the presence of identifiable peaks. The onset variable is determined by identifying the sample point corresponding to the maximum velocity within the systolic phase and normalizing it relative to the total systolic duration. The canopy variable is computed as the number of consecutive samples in which velocity exceeds 25% of the diastolic-systolic range. The peaks variable is calculated by identifying true peaks (zero-crossings of the first derivative) and pseudo-peaks (inflections with derivative magnitude below 0.01), assigning weights of one and (1 – derivative magnitude / 0.01), respectively, and summing all contributions. These variables are designed to be robust to noise, artifact, and waveform distortion.

- describe spectral clustering for identifying groups

Spectral clustering is employed to identify groups of waveforms with similar morphological characteristics without assuming convexity or uniform density of clusters. The algorithm constructs a similarity matrix based on the Euclidean distances between all waveform points in the three-dimensional feature space, then applies eigenvalue decomposition to project the data into a lower-dimensional space where clusters are more readily separable. The resulting clusters are assigned using k-means clustering in the transformed space. The radial basis kernel ensures that nonlinear relationships between variables are preserved, allowing for accurate identification of complex, non-linearly separable morphological patterns.

- discuss TCD morphological variability beyond TIBI flow grades

The analysis reveals that transcranial Doppler waveform morphology encompasses more than the four categories defined by the TIBI scale. While categories corresponding to normal (TIBI 5) and blunted flow (TIBI 2) are clearly identified, two additional morphological patterns emerge that do not align with existing classifications. One pattern, characterized by early systolic onset, narrow canopy, and weak peaks, is strongly associated with large vessel occlusion and may represent a previously unrecognized subtype of occlusive pathology. Another pattern, with late onset and weak peaks, exhibits overlap with TIBI grade 4 but lacks the velocity asymmetry required for traditional grading. These findings demonstrate that the TIBI scale, while clinically useful, does not fully capture the spectrum of morphological variation present in cerebral flow.

- introduce Velocity Curvature Index (VCI) for detecting Large Vessel Occlusion (LVO)

The Velocity Curvature Index is a previously established metric that quantifies the degree of curvature deviation in the systolic upstroke of the waveform, providing a reliable indicator of large vessel occlusion. While VCI effectively identifies waveforms with pathological curvature, it does not differentiate between distinct morphological subtypes of occlusion or distinguish occlusion from other causes of flow disturbance, such as stenosis or intracranial hypertension. The current system complements VCI by providing categorical classification that identifies not only the presence of abnormal flow but also its specific morphological signature, thereby enhancing diagnostic specificity.

- describe limitations of VCI for differentiating pathological morphologies

The Velocity Curvature Index is sensitive to the shape of the systolic upstroke but insensitive to features occurring in the mid-to-late systolic and diastolic phases, such as the presence or absence of secondary peaks, canopy width, or timing of maximal velocity. As a result, VCI cannot distinguish between waveforms with similar curvature but different downstream resistance patterns, nor can it differentiate occlusion from stenosis when both produce similar upstroke distortion. This limitation necessitates the use of additional morphological descriptors to fully characterize the nature of the underlying pathology.

- introduce objective waveform categorization for stroke etiology

The system introduces an objective, data-driven method for categorizing transcranial Doppler waveforms based on their morphological signatures, enabling direct inference of stroke etiology. By clustering waveforms from patients with confirmed large vessel occlusion and control subjects, the system identifies four distinct morphological categories, each with a unique association to underlying vascular pathology. This approach allows for automated classification of novel waveforms into these categories, providing clinicians with actionable information regarding the likely cause of flow disturbance without requiring expert interpretation.

- describe data-driven approach to waveform categorization

The categorization process is entirely data-driven, meaning that categories are not imposed by clinical heuristics but are instead derived from the intrinsic structure of the waveform dataset. This approach avoids the biases inherent in subjective grading systems and allows for the discovery of novel morphological patterns that may not have been previously recognized. The use of unsupervised learning ensures that the resulting categories reflect natural groupings in the data, enhancing generalizability across populations and clinical settings.

- apply unsupervised learning algorithm to dataset

An unsupervised learning algorithm, specifically spectral clustering, is applied to a dataset comprising 996 normalized waveforms from patients with large vessel occlusion, in-hospital controls, and out-of-hospital controls. The algorithm identifies four distinct clusters in the three-dimensional feature space formed by the onset, canopy, and peaks variables. Each cluster represents a unique morphological phenotype, and the membership of each waveform is determined probabilistically based on its position relative to cluster centroids and density distributions.

- employ spectral clustering for identifying clusters

Spectral clustering is implemented using a radial basis kernel and default parameters from the scikit-learn library. The similarity matrix is constructed from pairwise Euclidean distances in the feature space, and eigenvalue decomposition is performed to project the data into a space where clusters are linearly separable. The number of clusters is determined using the gap statistic, which compares the observed within-cluster dispersion to that expected under a null distribution. The optimal number of clusters is selected as the smallest k for which the gap statistic exceeds the gap of k+1 minus its standard error.

- compare TCD waveform morphology across subject groups

Waveform morphology is compared across three subject groups: patients with large vessel occlusion, in-hospital controls without occlusion, and out-of-hospital healthy controls. The largest cluster (Type I) is predominantly composed of controls and exhibits normal morphology. The smallest cluster (Type III) is almost exclusively composed of LVO patients and demonstrates a distinct pattern of early onset, narrow canopy, and weak peaks. The remaining clusters show intermediate compositions, with Type II associated primarily with controls and Type IV with a mix of LVO and control subjects. These comparisons reveal that morphological patterns are not uniformly distributed across clinical groups and that specific combinations of variables are strongly predictive of pathology.

- acquire CBFV signals using ultrasound probes

Cerebral blood flow velocity signals are acquired using handheld 2 MHz transcranial Doppler probes positioned over the temporal bone window to insonate the middle cerebral artery. Recordings are made at depths between 45 and 60 mm to ensure consistent vessel targeting. Each recording lasts 30 seconds, during which time at least 15 individual cardiac cycles are captured. Signals are digitized at 125 Hz and stored for offline analysis. The probe is held in place by manual operator control or automated robotic positioning, ensuring stable acoustic coupling and consistent signal quality.

- describe automated process for signal acquisition

The automated signal acquisition process begins with the detection of the optimal insonation depth using real-time velocity feedback and signal-to-noise ratio metrics. Once a stable signal is identified, the system initiates a 30-second recording and simultaneously monitors for motion artifact or signal dropout. If artifact is detected, the system prompts the operator to reposition the probe or automatically adjusts the probe position via robotic control. Accepted recordings are automatically segmented into individual beats, and outlier beats are rejected using an iterated interquartile range algorithm based on cross-correlation and beat length consistency.

- extract individual beat waveforms from recorded depths

Individual cardiac cycles are extracted from the continuous signal using an automated algorithm that identifies the onset of systolic upstroke as the point of maximum positive derivative preceding a velocity peak. Each beat is isolated as a segment spanning from one systolic onset to the next. Beats with excessive noise, irregular duration, or low correlation to the median beat are excluded. Only recordings containing at least 15 accepted beats from a single depth are retained for further analysis.

- align and average accepted beats for representative waveform

Accepted beats are temporally aligned at their systolic onset and padded with their final value to match the length of the longest beat in the ensemble. The aligned beats are then averaged point-by-point to produce a single representative waveform for each recording depth. This averaging process reduces the impact of transient noise and enhances the signal-to-noise ratio, producing a waveform that accurately reflects the underlying hemodynamic pattern.

- resample waveforms to match native sampling rate

Waveforms acquired at different sampling rates are resampled to a uniform rate of 125 Hz using cubic spline interpolation. This ensures that all waveforms have identical temporal resolution, enabling direct comparison and clustering without distortion due to sampling differences.

- smooth waveforms to reduce high-frequency noise

Each waveform is convolved with a 90-millisecond Hanning window to attenuate high-frequency noise components that exceed the physiological range of cerebral flow dynamics. This smoothing operation preserves the overall shape of the waveform while eliminating spurious fluctuations unrelated to vascular physiology.

- normalize waveforms with respect to time and velocity

Each waveform is normalized in time by resampling to 100 equally spaced points, effectively standardizing the cardiac cycle duration. Velocity normalization is performed by subtracting the minimum velocity and dividing by the difference between maximum and minimum velocity, scaling the waveform to the range [0,1]. This dual normalization ensures that clustering is based solely on waveform shape, independent of heart rate or absolute velocity magnitude.

- extract three morphological features from each waveform

Three morphological features—onset, canopy, and peaks—are extracted from each normalized waveform using the procedures described above. These features are computed independently and stored as a three-dimensional vector for each waveform, forming the basis for subsequent clustering and classification.

- define onset variable for absolute peak onset

The onset variable is defined as the normalized time point at which the maximum velocity occurs within the systolic phase, expressed as a fraction of the total systolic duration. This variable captures the timing of flow acceleration and is sensitive to delays caused by increased resistance or stenosis.

- define canopy variable for systolic canopy length

The canopy variable is defined as the number of discrete time samples in which the velocity exceeds 25% of the difference between systolic peak and diastolic minimum. This variable quantifies the breadth of the systolic complex and is sensitive to blunting or attenuation of flow.

- define peaks variable for number/prominence of auxiliary peaks

The peaks variable is defined as the sum of weights assigned to all local maxima and minima in the waveform, where true peaks (zero-crossings of the first derivative) are assigned a weight of one, and pseudo-peaks (inflections with derivative magnitude below 0.01) are assigned weights inversely proportional to their slope. This variable captures the complexity of the waveform and is sensitive to the presence or absence of secondary structures.

- use spectral clustering to identify groups in feature space

The three-dimensional feature space formed by the onset, canopy, and peaks variables is subjected to spectral clustering using a radial basis kernel. The algorithm identifies four distinct clusters, each representing a unique morphological phenotype. Cluster membership is determined probabilistically, and the optimal number of clusters is confirmed using the gap statistic.

- derive beat archetypes for each resultant cluster

For each cluster, the five waveforms with the smallest mean intra-cluster distance are selected and averaged to produce a representative archetype. These archetypes serve as morphological templates for classifying new waveforms and are displayed in the user interface alongside the classification results.

- describe categorization of onset variable

The onset variable is categorized into three ranges: early (0.1–0.3), intermediate (0.3–0.5), and late (0.5–0.7), based on the empirical distribution observed in the training cohort. These ranges are used to define the boundaries of the morphological categories in conjunction with the canopy and peaks variables.

- describe categorization of canopy variable

The canopy variable is categorized into three ranges: narrow (10–25 samples), moderate (25–40 samples), and wide (40–60 samples), reflecting the breadth of the systolic complex. These ranges are used to distinguish between normal, blunted, and occluded flow patterns.

- describe categorization of peaks variable

The peaks variable is categorized into three ranges: weak (0–1.5), moderate (1.5–3.0), and strong (3.0–5.0), based on the summed weights of true and pseudo-peaks. These ranges reflect the complexity of the waveform and are used to differentiate between normal, stenotic, and occluded morphologies.

- illustrate waveform translation to three-dimensional representation

Each waveform is translated into a point in a three-dimensional space where the x-axis represents onset, the y-axis represents canopy, and the z-axis represents peaks. This representation allows for direct visualization of morphological relationships and enables the identification of clusters that are not apparent in one- or two-dimensional analyses.

- depict waveform data with various morphological features

The three-dimensional feature space is depicted with color-coded points representing each waveform, with cluster membership indicated by hue. Archetypal waveforms are displayed adjacent to their corresponding cluster centroids, allowing for direct comparison between abstract data points and their physical waveform manifestations.

- describe clusters of waveforms

Four clusters are identified, each with distinct morphological characteristics. Cluster I exhibits early onset, wide canopy, and strong peaks, corresponding to normal flow. Cluster II exhibits late onset, wide canopy, and strong peaks, associated with altered resistance. Cluster III exhibits early onset, narrow canopy, and weak peaks, strongly associated with large vessel occlusion. Cluster IV exhibits late onset, wide canopy, and weak peaks, corresponding to blunted flow.

- define morphological variables

The morphological variables are defined as quantitative, algorithmically derived descriptors of waveform shape that capture the essential hemodynamic features relevant to cerebrovascular pathology. These variables are independent of absolute velocity, heart rate, or signal amplitude, ensuring that classification is based solely on morphology.

- partition feature space via spectral clustering

The feature space is partitioned into four non-overlapping regions using spectral clustering, with each region corresponding to a distinct morphological category. The boundaries between regions are determined by the statistical distribution of cluster members and are not fixed by clinical assumptions.

- derive archetypal waveforms for each cluster

Archetypal waveforms are derived by averaging the five most representative waveforms within each cluster. These archetypes serve as reference templates for classifying new waveforms and are displayed in the user interface as visual benchmarks.

- visualize cluster morphologies

Cluster morphologies are visualized through three-dimensional scatter plots, color-coded by cluster membership, and through side-by-side displays of archetypal waveforms. These visualizations enable clinicians to intuitively understand the relationship between abstract data points and physical waveform shapes.

- describe features of representative waveforms

The representative waveforms for each cluster exhibit consistent morphological features: Cluster I has a sharp, early systolic peak with multiple distinct secondary peaks; Cluster II has a delayed peak but retains strong secondary structures; Cluster III has a sharp early peak but lacks subsequent structures; Cluster IV has a delayed peak and minimal secondary structures.

- depict categories of waveforms in 3D cluster space

The four categories are depicted as distinct, non-convex regions in the three-dimensional feature space, with overlapping boundaries between adjacent clusters. The spatial relationships between clusters reflect their morphological similarities and differences.

- describe fuzzy boundary between clusters

The boundaries between Cluster II and Cluster IV are fuzzy, primarily due to overlap in the peaks variable. Waveforms with intermediate peak prominence may be assigned to either cluster with moderate probability, reflecting the clinical ambiguity of certain flow patterns.

- discuss limitations of clustering

The clustering approach is limited by its reliance on three morphological variables, which may not capture all aspects of waveform morphology. Additionally, the absence of waveforms corresponding to TIBI grades 0 and 1 limits the system’s ability to detect absent or minimal flow. Future iterations may incorporate additional features derived from spectral analysis or M-mode imaging.

- depict waveforms with different morphological features

Waveforms from each cluster are displayed side-by-side, highlighting the differences in onset timing, canopy width, and peak structure. These visual comparisons demonstrate the distinct morphological signatures associated with each category.

- compute representative waveforms for each category

Representative waveforms are computed as the average of the five most central waveforms within each cluster, ensuring that the archetype reflects the most typical morphology of the group.

- describe features of waveforms in each category

Cluster I waveforms exhibit normal timing, broad systolic complexes, and multiple secondary peaks. Cluster II waveforms show delayed systolic timing but retain normal peak structure. Cluster III waveforms have normal early acceleration but lack subsequent structures, indicating proximal occlusion. Cluster IV waveforms show delayed timing and diminished peak complexity, consistent with blunted flow.

- depict histograms of probability distributions

Histograms of the probability distributions for each cluster are displayed, showing the likelihood of a waveform belonging to each category. These histograms are used to generate confidence scores for classification decisions.

- describe gap-statistic disparity for clusters

The gap statistic reveals a clear maximum at four clusters, with diminishing returns for additional clusters. The disparity between the observed and expected log dispersion confirms that four clusters optimally represent the underlying structure of the data.

- determine optimal number of clusters

The optimal number of clusters is determined as four, based on the gap statistic criterion, which selects the smallest k for which the gap statistic exceeds the gap of k+1 minus its standard error.

- relate clusters to TIBI flow grades

Cluster I corresponds to TIBI grade 5 (normal flow), Cluster IV corresponds to TIBI grade 2 (blunted flow), and Cluster III corresponds to a novel morphology not fully captured by existing TIBI criteria. Cluster II does not have a direct TIBI analogue but may reflect a subtype of stenotic or resistance-related flow.

- discuss association of clusters with control subjects and LVO

Cluster I is predominantly composed of control subjects, Cluster III is almost exclusively composed of LVO patients, and Clusters II and IV contain mixtures of both groups, reflecting the heterogeneity of flow patterns in non-occlusive conditions.

- discuss limitations of TIBI flow grades

The TIBI scale is limited by its reliance on subjective visual interpretation, its inability to distinguish between morphologically distinct occlusion patterns, and its exclusion of waveforms with absent or minimal pulsatility. The current system overcomes these limitations by providing an objective, data-driven classification framework.

- propose extension of clustering framework

The clustering framework may be extended to include additional morphological variables derived from spectral analysis, M-mode imaging, or velocity asymmetry between hemispheres. Future versions may also incorporate longitudinal data to track morphological changes over time.

- describe potential applications of clustering framework

Potential applications include prehospital stroke triage, automated stroke center triage protocols, real-time monitoring in the intensive care unit, and training tools for non-specialist clinicians. The system may also be integrated into telemedicine platforms to enable remote interpretation of TCD waveforms.

- introduce waveform visualization system

A dedicated waveform visualization system is described, comprising a headset device, a robotic probe positioning system, a controller, an output display, an input interface, and a network interface. The system is designed for seamless integration into clinical workflows and enables real-time, objective classification of transcranial Doppler waveforms.

- describe headset device

The headset device is a lightweight, adjustable frame that securely holds the transcranial Doppler probe in a fixed position relative to the patient’s temporal window. The headset includes a robotic positioning system that automatically adjusts probe angle and depth to maintain optimal signal quality.

- describe controller

The controller receives raw ultrasound data from the probe, performs signal processing to extract morphological variables, applies the clustering algorithm to classify the waveform, and transmits the results to the output device. The controller includes a processor, memory, and software for executing the classification pipeline.

- describe output device

The output device is a high-resolution touchscreen display that presents the classified waveform, its position in the three-dimensional feature space, the assigned category, and the probability of classification. The display also includes clinical interpretations and recommendations based on the classification.

- describe input device

The input device is a touchscreen interface or physical buttons that allow the operator to initiate recordings, adjust parameters, or override classifications. The input device also permits entry of patient identifiers and clinical notes.

- describe network interface

The network interface enables secure transmission of data to hospital information systems, electronic health records, and remote diagnostic centers. The interface supports HIPAA-compliant communication protocols and encrypted data transfer.

- describe probe configuration

The probe is a 2 MHz transcranial Doppler transducer with a concave surface designed to focus acoustic energy into the middle cerebral artery. The probe is coupled to the robotic positioning system via a threaded interface that allows for precise axial and angular adjustments.

- describe automatic location of middle cerebral artery

The system automatically locates the middle cerebral artery by scanning a predefined spatial grid over the temporal window and selecting the depth and angle that maximizes signal amplitude and signal-to-noise ratio.

- describe positioning of probes

Probes are positioned using either manual operator control or automated robotic adjustment. The robotic system uses real-time feedback to maintain optimal probe alignment throughout the recording.

- describe manual operation of probes

In manual mode, the operator adjusts the probe position using a joystick or touchpad interface, with real-time visual feedback indicating signal quality and depth.

- conclude description of waveform visualization system

The waveform visualization system integrates all components into a unified platform that enables objective, automated classification of transcranial Doppler waveforms in real time, enhancing diagnostic accuracy and accessibility.

- describe controller 830

Controller 830 receives raw ultrasound data from the probe, executes signal processing algorithms to extract morphological variables, applies the clustering model to classify the waveform, and generates output signals for display. The controller includes a microprocessor, memory storage for cluster models, and software for real-time computation.

- receive ultrasound data

The controller receives digitized ultrasound data from the probe via a high-speed digital interface, ensuring minimal latency between signal acquisition and processing.

- generate CBFV waveforms

The controller generates cerebral blood flow velocity waveforms by demodulating the Doppler signal and applying filtering and envelope detection algorithms to extract the velocity profile.

- perform signal processing functions

The controller performs all signal processing functions, including beat detection, artifact rejection, alignment, averaging, resampling, smoothing, and normalization.

- determine morphological indicators

The controller computes the onset, canopy, and peaks variables for each waveform and maps them into the three-dimensional feature space.

- display morphological indicators

The controller transmits the morphological indicators and classification results to the output device for display to the operator.

- describe output device 845

Output device 845 is a high-resolution touchscreen display that presents the classified waveform, its position in the feature space, the assigned category, and the probability of classification. The display also includes clinical interpretations and recommendations.

- describe headset device 810

Headset device 810 includes a robotic positioning system 814 that automatically adjusts the probe position to maintain optimal signal quality. The headset is designed for comfort and stability during prolonged recordings.

- include robotics 814

Robotics 814 is a multi-degree-of-freedom positioning system that translates and rotates the probe in three-dimensional space to maintain optimal insonation angle and depth.

- translate probe 805

The robotics system translates the probe along the x, y, and z axes to achieve optimal alignment with the middle cerebral artery.

- move probe 805 with respect to head

The probe is moved with respect to the patient’s head using servo motors controlled by real-time feedback from the signal quality metrics.

- include multiple degree of freedom TCD transducer positioning system

The positioning system includes six degrees of freedom: three translational and three rotational, enabling precise control of probe orientation and depth.

- describe end of probe 805

The end of the probe includes a concave acoustic surface that focuses the ultrasound beam into the middle cerebral artery, improving signal penetration and clarity.

- couple to robotics 814

The probe is coupled to the robotics system via a threaded interface that allows for secure attachment and precise mechanical control.

- include concave surface

The concave surface of the probe focuses the acoustic energy into a narrow beam, enhancing signal-to-noise ratio and reducing artifacts from adjacent vessels.

- describe first end of probe 805

The first end of the probe contains the piezoelectric transducer that emits and receives acoustic energy.

- emit acoustic energy

The transducer emits pulsed ultrasound waves at 2 MHz and receives the Doppler-shifted echoes reflected from moving red blood cells.

- describe second end of probe 805

The second end of the probe is mechanically coupled to the robotics system and includes a threaded section for secure attachment.

- include threaded section

The threaded section allows the probe to be screwed into the robotic mount, ensuring stable alignment and preventing dislodgement during movement.

- describe structural support 816

Structural support 816 is a rigid frame that supports the patient’s head and maintains the headset device in a fixed position relative to the skull.

- support head of patient

The structural support is contoured to fit the patient’s head and includes padding for comfort and stability.

- support headset device 810

The structural support is rigidly connected to the headset device, ensuring that probe position remains stable during robotic adjustments.

- describe input device 850

Input device 850 is a touchscreen interface or physical buttons that allow the operator to initiate recordings, adjust parameters, or override classifications.

- describe network interface 860

Network interface 860 enables secure transmission of data to hospital information systems and remote diagnostic centers using encrypted, HIPAA-compliant protocols.

- describe controller 830 operations

Controller 830 executes input commands, performs signal processing, determines morphological indicators, applies clustering algorithms, and transmits results to the output device.

- describe controller 830

Controller 830 is a dedicated embedded system with real-time processing capabilities, optimized for low-latency execution of the classification pipeline.

- introduce robotic control circuit 840

Robotic control circuit 840 is a microcontroller that receives commands from the main controller and drives the servo motors of the robotic positioning system.

- explain control of probe 805

The robotic control circuit receives position commands from the main controller and adjusts the probe’s orientation and depth using closed-loop feedback based on signal quality metrics.

- describe method 900 of categorizing a waveform

Method 900 comprises the steps of receiving an ultrasound signal, extracting individual beats, averaging accepted beats, normalizing the waveform, computing the onset, canopy, and peaks variables, mapping the waveform into the three-dimensional feature space, determining cluster membership using spectral clustering, assigning a category, and displaying the result with a probability score.

- process signal containing ultrasound data

The signal is processed to remove noise, detect individual cardiac cycles, and reject artifacts.

- identify morphological variables of waveform

The onset, canopy, and peaks variables are computed from the normalized waveform.

- identify categories corresponding to morphological variables

The system references a precomputed model that maps combinations of morphological variables to predefined categories.

- categorize waveform as belonging to one of the categories

The waveform is assigned to the category with the highest posterior probability.

- visualize one or more categories

The system displays the waveform alongside its position in the three-dimensional feature space and the assigned category.

- display probability of waveform belonging to categories

The system displays a probability distribution across all categories, with the most likely category highlighted.

- map morphological variables to axes

The onset variable is mapped to the x-axis, the canopy variable to the y-axis, and the peaks variable to the z-axis.

- categorize waveform based on morphological variables

The waveform is categorized based on its position in the three-dimensional feature space relative to the predefined cluster boundaries.

- describe examples of categorizing waveform

A waveform with early onset, wide canopy, and strong peaks is categorized as Type I (normal flow). A waveform with late onset, wide canopy, and weak peaks is categorized as Type IV (blunted flow).

- describe correspondence of categories to blood flow

Type I corresponds to normal flow, Type II to altered resistance, Type III to large vessel occlusion, and Type IV to blunted flow.

- describe overlap of designated areas of categories

The boundaries between Type II and Type IV exhibit overlap due to similarity in canopy and onset, with differentiation primarily determined by the peaks variable.

- describe examples of blood flow and pathologies

Normal flow is associated with healthy vasculature, blunted flow with moderate stenosis, and the novel Type III morphology with proximal occlusion.

- describe automated algorithm for performing method

The automated algorithm is implemented in software running on the controller and executes all steps of the method without operator intervention.

- describe displaying waveform and samples

The system displays the classified waveform alongside a gallery of archetype waveforms from each category, enabling visual comparison.

- describe method 100 for visualizing a waveform

Method 100 comprises collecting ultrasound data, automatically identifying morphological variables, referencing predetermined categories, determining classification probability, assigning a category, and displaying the categorized waveform with its probability score.

- collect data about waveform using ultrasound insonation

Ultrasound data is collected using a transcranial Doppler probe positioned over the temporal window.

- automatically identify morphological variables of waveform

The system computes the onset, canopy, and peaks variables from the normalized waveform.

- automatically reference predetermined categories

The system accesses a precomputed clustering model to determine the most likely category for the waveform.

- determine probability of waveform belonging to categories

The system calculates the posterior probability of the waveform belonging to each category using kernel density estimation.

- categorize waveform as belonging to one of the categories

The waveform is assigned to the category with the highest probability.

- display categorized waveform and probability

The system displays the waveform, its category, and a confidence score on the output device.

- describe examples of morphological variables

Examples include the timing of systolic peak, the duration of the systolic complex, and the number of secondary peaks.

- describe scope of claims and equivalents

The claims encompass all methods, systems, and apparatuses that implement the disclosed process for automated classification of transcranial Doppler waveforms based on morphological variables, including equivalents that perform substantially the same function in substantially the same way to achieve substantially the same result.

- describe implementation of logical blocks and circuits

The logical blocks and circuits described herein may be implemented using general-purpose processors, digital signal processors, application-specific integrated circuits, field-programmable gate arrays, or any combination thereof. The methods may be implemented in software, firmware, hardware, or any combination thereof.