Here is the complete patent application following the provided outline:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of medical diagnostics and personalized cancer therapy. More specifically, the invention pertains to a novel system and method for assessing the viability of cancer cells in response to targeted therapeutic agents using multifrequency impedance spectroscopy combined with machine learning algorithms. The disclosed technology enables rapid, label-free classification of live and dead tumor cells, facilitating real-time evaluation of drug efficacy in targeted cancer treatments.  

## BACKGROUND OF THE INVENTION  

Cancer remains a leading cause of mortality worldwide, with current treatment modalities including surgery, chemotherapy, radiation therapy, and targeted therapies. Conventional chemotherapy suffers from significant limitations due to its non-specific mechanism of action, resulting in systemic toxicity and adverse effects on healthy tissues. Targeted therapies, wherein antineoplastic agents are conjugated to antibodies specific to tumor cell surface markers, offer a promising alternative by selectively delivering cytotoxic payloads to malignant cells while sparing normal tissues.  

A critical challenge in implementing targeted cancer therapies lies in accurately predicting patient response to specific drug candidates. Current methods for assessing cell viability, such as trypan blue exclusion assays or optical techniques, suffer from limitations including the need for cell staining, bulky instrumentation, and inability to perform downstream molecular analyses on labeled samples. Microfluidic impedance cytometry has emerged as a label-free alternative for single-cell analysis, but existing implementations typically rely on single-frequency measurements, which lack the discriminatory power needed for robust classification of cell viability states.  

There exists an unmet need for a rapid, accurate, and label-free system capable of evaluating cancer cell response to targeted therapeutic agents at the point-of-care. Such a system would enable personalized treatment selection and real-time monitoring of therapeutic efficacy, significantly improving clinical outcomes in cancer management.  

## SUMMARY OF THE INVENTION  

The present invention provides a comprehensive solution to the aforementioned challenges through an integrated system combining multifrequency impedance cytometry with machine learning-based classification algorithms. The invention encompasses a microfluidic device featuring microfabricated electrodes capable of simultaneously measuring cellular impedance across a broad frequency spectrum (300 kHz to 30 MHz). As cells flow through the microchannel, their dielectric properties are interrogated at multiple discrete frequencies, generating a comprehensive electrical fingerprint that reflects cell viability status.  

Key innovations of the present invention include:  
1. A multifrequency impedance measurement system employing simultaneous acquisition at strategically selected frequency bands to capture both membrane-related (low frequency) and intracellular (high frequency) dielectric properties.  
2. A machine learning framework utilizing support vector machines (SVMs) with Gaussian kernels to classify cells as live or dead based on extracted impedance features, including amplitude and phase changes across multiple frequencies.  
3. An optimized microfluidic chip design incorporating gold electrodes on a glass substrate with precisely controlled channel dimensions (100 μm width × 30 μm height) to ensure consistent single-cell analysis.  
4. A data processing pipeline featuring advanced signal conditioning algorithms for detrending, denoising, and feature extraction from raw impedance signals.  

The system demonstrates exceptional classification accuracy (>95%) when utilizing both amplitude and phase change features across multiple frequencies, significantly outperforming conventional single-frequency approaches. This technological advancement enables rapid, label-free assessment of cancer cell viability in response to targeted therapeutic agents, facilitating personalized treatment selection and real-time monitoring of drug efficacy.  

## DETAILED DESCRIPTION OF THE INVENTION  

### A. Methods and Systems for Classifying Biological Particles  

The present invention provides a comprehensive system for classifying biological particles, particularly cancer cells, based on their viability status following exposure to therapeutic agents. The system architecture comprises three principal components: (1) a microfluidic impedance cytometer, (2) a multifrequency measurement subsystem, and (3) a machine learning classification module.  

The microfluidic impedance cytometer forms the core physical platform for cell analysis. The device incorporates a polydimethylsiloxane (PDMS) microchannel bonded to a glass substrate containing patterned gold electrodes. The electrode configuration consists of two parallel sensing elements (20 μm width) separated by a 25 μm gap, positioned perpendicular to the fluid flow direction. A 10 nm chromium adhesion layer ensures robust bonding between the gold electrodes and glass substrate. The microchannel dimensions (100 μm width × 30 μm height) are optimized to facilitate single-cell analysis while minimizing signal interference from multiple simultaneous cell transits.  

The multifrequency measurement subsystem employs a lock-in amplifier capable of simultaneous acquisition at four discrete frequencies within the 300 kHz to 30 MHz range. This subsystem applies an alternating current excitation across the sensing electrodes and measures the resultant impedance modulation as cells traverse the detection zone. The system specifically utilizes 500 kHz as a baseline frequency for all measurements, supplemented by higher frequencies (20 MHz, 25 MHz, and 30 MHz) to probe intracellular properties.  

The machine learning classification module implements a support vector machine (SVM) algorithm with Gaussian kernel for robust cell classification. The module processes two primary features extracted from the impedance signals: amplitude change (ΔA) and phase change (Δθ). These features are calculated for each frequency channel by comparing the baseline signal to the peak response during cell transit. The SVM classifier is trained using labeled datasets comprising known live and dead cell populations, with features normalized to ensure consistent scaling across frequency bands.  

System operation follows a standardized protocol:  
1. Cancer cell samples are prepared in phosphate-buffered saline (PBS) at optimal concentration (~400 cells/μL).  
2. The microfluidic channel is primed with PBS and flow is established via gravity-driven pressure differential.  
3. Cells are introduced through the inlet and their transit through the detection zone generates multifrequency impedance signals.  
4. Signal processing algorithms extract amplitude and phase change features for each cell at all measurement frequencies.  
5. The trained SVM classifier assigns viability status (live/dead) to each cell based on the multidimensional feature vector.  
6. Aggregate viability statistics are computed and presented as a percentage of dead cells in the population.  

The system's performance has been validated using T47D breast cancer cells as a model system, demonstrating classification accuracy exceeding 95% when utilizing combined amplitude and phase features across four frequency bands. This represents a significant improvement over conventional single-frequency approaches and establishes the invention as a robust platform for therapeutic efficacy assessment.  

### B. Definitions  

For purposes of interpreting this disclosure, the following terms shall have the meanings set forth below:  

"Multifrequency impedance spectroscopy" refers to the measurement of a particle's electrical properties at multiple discrete frequencies simultaneously, encompassing both amplitude and phase characteristics of the impedance response.  

"Microfluidic impedance cytometer" denotes a device comprising microfabricated electrodes integrated with a microfluidic channel, designed to measure electrical properties of particles in flow at single-cell resolution.  

"Amplitude change (ΔA)" represents the difference between baseline impedance magnitude and peak impedance magnitude during cell transit, measured in decibels or arbitrary units.  

"Phase change (Δθ)" indicates the angular displacement between the excitation signal and measured response during cell transit, measured in degrees or radians.  

"Support vector machine (SVM)" refers to a supervised machine learning algorithm that constructs hyperplanes in high-dimensional space for classification tasks, particularly employing Gaussian kernel functions for nonlinear separation of feature spaces.  

"Label-free" describes analytical techniques that do not require chemical staining, fluorescent tagging, or other exogenous markers to characterize biological samples.  

"Targeted cancer therapy" encompasses therapeutic approaches wherein cytotoxic agents are specifically delivered to tumor cells through molecular targeting mechanisms, such as antibody-drug conjugates binding to cell surface markers.  

"Viability" indicates the physiological state of a cell, with "live" referring to metabolically active, intact cells and "dead" referring to cells undergoing apoptosis or necrosis.  

"Activated matriptase" refers to the catalytically active form of the membrane-bound serine protease overexpressed in various epithelial cancers, serving as a molecular target for therapeutic antibodies.  

"Gaussian kernel" denotes a radial basis function used in machine learning to transform input features into higher-dimensional space for nonlinear classification, defined by the equation K(x,y) = exp(-γ||x-y||²) where γ is the kernel parameter.  

"Double-layer capacitance (Cdl)" represents the capacitive effect arising from ionic charge separation at the electrode-electrolyte interface, influencing low-frequency impedance measurements.  

"Solution resistance (Rs)" refers to the bulk resistive component of the electrolyte medium between measurement electrodes.  

"Coupling capacitance (Ccell)" indicates the capacitive coupling between electrodes through the measured cell or particle.  

--- 

This complete patent application thoroughly describes the invention while adhering to the specified outline structure and maintaining formal patent language throughout. Each section provides comprehensive technical details while ensuring the document stands alone without reference to the original research paper.