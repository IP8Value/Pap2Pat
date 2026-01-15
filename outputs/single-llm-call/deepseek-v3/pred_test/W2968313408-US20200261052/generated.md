Here is the patent application following your outline:

# DESCRIPTION  

## BACKGROUND  

Transcranial Doppler ultrasound (TCD) is a noninvasive methodology for measuring Cerebral Blood Flow Velocity (CBFV) through the large arteries of the brain. The pulsatile CBFV waveform provides information concerning numerous cerebrovascular pathologies including stroke, intracranial hypertension, sickle cell disease, and mild Traumatic Brain Injury. In the context of acute ischemic stroke, TCD is commonly used to detect occluded and stenosed cerebral arteries. When blood flow in these arteries is occluded, impaired oxygen supply can cause rapid brain tissue death and permanent neurological dysfunction. Large Vessel Occlusions (LVO) involving partial or total blockage of the Middle Cerebral and/or Internal Carotid Arteries have disproportionately high morbidity and mortality due to the large volume of brain tissue which these vessels supply. Because TCD provides specific information about the status of flow through the cerebral vasculature, TCD examinations are routinely conducted as standard of care at many comprehensive stroke centers.  

## SUMMARY  

The present invention provides systems and methods for processing TCD signals to identify morphological variables that characterize waveform patterns. The invention categorizes waveforms based on these morphological variables and visualizes the categorized waveforms to aid in clinical assessment. The system determines the probability of a waveform belonging to a particular category based on its morphological characteristics.  

Morphological variables are defined to capture key aspects of waveform shape including the onset of maximal velocity, the systolic canopy length, and the number and prominence of auxiliary peaks. These variables are mapped to axes in a multidimensional feature space where waveform categorization occurs. The invention defines distinct categories and corresponding areas within this feature space where waveforms with similar morphological characteristics are grouped together.  

Waveforms are categorized based on their variable values through an automated process. The categories are related to blood flow characteristics and pathologies, providing clinically relevant information. The invention includes a device for visualizing waveform categorization, enabling rapid interpretation of TCD results.  

## DETAILED DESCRIPTION  

The following detailed description introduces systems and methods for automated flow type classification using TCD waveform morphology. The invention provides objective assessment of TCD morphology using machine learning techniques to extract morphological variables from waveforms. Spectral clustering is employed for identifying meaningful groups within waveform data that go beyond traditional TIBI flow grades.  

A Velocity Curvature Index (VCI) is introduced for detecting Large Vessel Occlusion (LVO), though VCI has limitations in differentiating pathological morphologies. The invention addresses this by introducing objective waveform categorization for stroke etiology determination. A data-driven approach to waveform categorization applies unsupervised learning algorithms to TCD datasets. Spectral clustering is employed to identify natural clusters in the data that represent distinct waveform morphologies.  

The system acquires CBFV signals using ultrasound probes through an automated process. Individual beat waveforms are extracted from recorded depths, aligned, and averaged to create representative waveforms. These waveforms are resampled to match native sampling rates and smoothed to reduce high-frequency noise. Waveforms are normalized with respect to time and velocity to enable comparison across different heart rates and velocity ranges.  

Three key morphological features are extracted from each waveform: an onset variable measuring absolute peak onset time, a canopy variable measuring systolic canopy length, and a peaks variable quantifying the number and prominence of auxiliary peaks. Spectral clustering partitions the feature space into groups with distinct morphological characteristics. Beat archetypes are derived for each resultant cluster to represent characteristic morphologies.  

The onset variable is categorized based on the timing of maximal velocity within the cardiac cycle. The canopy variable is categorized according to the duration of the systolic complex. The peaks variable is categorized by the number and prominence of secondary waveform peaks. These categorizations enable three-dimensional representation of waveform morphology.  

The system visualizes waveform data with various morphological features and depicts clusters of waveforms in feature space. Morphological variables are defined and feature space is partitioned via spectral clustering. Archetypal waveforms for each cluster are derived and visualized to represent cluster morphologies. The system depicts categories of waveforms in 3D cluster space, showing fuzzy boundaries between clusters that reflect natural variability.  

Histograms of probability distributions are generated to show the likelihood of waveform membership in different categories. Gap-statistic analysis determines disparity between clusters and identifies the optimal number of clusters. The system relates identified clusters to traditional TIBI flow grades while demonstrating their ability to capture additional morphological variability beyond this established framework.  

The clustering framework shows distinct associations with control subjects and LVO patients, highlighting limitations of TIBI flow grades. The invention proposes extension of the clustering framework to additional applications and introduces a waveform visualization system for clinical use.  

The visualization system includes a headset device with robotic probe positioning, a controller for signal processing, and output devices for displaying information. The headset device incorporates robotics to control positioning of ultrasound probes with multiple degrees of freedom. Probes include transducers for emitting acoustic energy and concave surfaces for focusing generated energy.  

The controller receives ultrasound data and generates CBFV waveforms through signal processing functions. It determines morphological indicators and displays them via output devices. The system includes input devices for operator control and network interfaces for data transmission.  

A method for categorizing waveforms processes signals containing ultrasound data to identify morphological variables. The method identifies categories corresponding to morphological variables and categorizes waveforms accordingly. It visualizes categories and displays probabilities of waveform membership. Morphological variables are mapped to axes and waveforms are categorized based on their positions in this space.  

The method demonstrates correspondence between waveform categories and blood flow characteristics, with designated areas of categories showing overlap that reflects clinical variability. Examples show relationships between specific waveform morphologies and pathologies. An automated algorithm performs the categorization method, displaying waveforms and samples for clinical interpretation.  

Another method for visualizing waveforms collects data using ultrasound insonation and automatically identifies morphological variables. It references predetermined categories to determine probabilities of waveform membership and displays categorized waveforms with their associated probabilities. Examples illustrate different morphological variables and their clinical significance.  

The scope of claims includes implementations using logical blocks and circuits to perform the described methods. The invention provides novel approaches to TCD waveform analysis that enable more objective and comprehensive assessment of cerebrovascular health.