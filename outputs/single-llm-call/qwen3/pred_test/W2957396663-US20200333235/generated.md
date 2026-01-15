# DESCRIPTION

## FIELD OF THE INVENTION

- define field of invention

The present invention relates to systems and methods for the classification of biological particles, particularly viable and nonviable cancer cells, through the measurement and analysis of their dielectric properties using multifrequency impedance cytometry in combination with machine learning algorithms. This technology is specifically designed for rapid, label-free assessment of cellular response to targeted therapeutics in clinical and point-of-care settings. The invention finds application in oncology diagnostics, personalized medicine, and real-time monitoring of drug efficacy, enabling the differentiation of live from dead cells without the need for fluorescent labeling, dye-based assays, or bulky optical instrumentation. By leveraging the intrinsic electrical signatures of cells under alternating current fields across a broad spectrum of frequencies, the disclosed system provides a robust, scalable, and cost-effective platform for evaluating tumor cell viability following exposure to antibody-conjugated anticancer agents.

## BACKGROUND OF THE INVENTION

- introduce circulating cancer cells

Circulating cancer cells, derived from primary or metastatic tumors, represent a critical biomarker for disease progression, therapeutic response, and prognosis in oncology. These cells enter the bloodstream through invasion of surrounding tissues and vascular structures, serving as precursors to distant metastasis. Their presence and viability are directly correlated with disease aggressiveness and treatment resistance, making their accurate detection and characterization essential for clinical decision-making.

- describe limitations of current detection methods

Current methodologies for assessing cancer cell viability predominantly rely on dye-based techniques such as trypan blue exclusion, flow cytometry with annexin V/propidium iodide staining, or colorimetric assays like MTS and MTT. These approaches require chemical labeling, extensive sample preparation, and specialized optical instrumentation, which collectively introduce significant delays, increase costs, and preclude downstream molecular analysis due to cellular damage or fixation. Moreover, these methods are not easily adaptable to point-of-care environments due to their reliance on laboratory infrastructure and trained personnel.

- discuss importance of CTC analysis

Analysis of circulating tumor cells remains a cornerstone of precision oncology, as it enables real-time monitoring of tumor dynamics without the need for invasive tissue biopsies. The ability to quantify viable versus nonviable tumor cells following therapeutic intervention provides direct insight into drug efficacy, allowing clinicians to tailor treatment regimens based on individual patient responses. This capability is particularly valuable in cases where tumors exhibit heterogeneity or develop resistance to targeted therapies.

- summarize existing CTC detection technologies

Existing technologies for CTC detection include immunomagnetic separation, microfluidic filtration, and optical imaging systems such as optical coherence tomography and fluorescence-activated cell sorting. While these platforms offer varying degrees of sensitivity and specificity, they are often limited by low throughput, high reagent consumption, complex operational protocols, and an inability to simultaneously assess multiple dielectric parameters of individual cells.

- highlight challenges in CTC detection

Key challenges in CTC detection include the rarity of target cells in blood samples, the structural and phenotypic heterogeneity among tumor cells, and the lack of universal surface markers that distinguish viable from nonviable populations. Additionally, most current methods fail to capture dynamic changes in cell integrity that occur during early stages of apoptosis or necrosis, resulting in delayed or inaccurate assessments of therapeutic response.

- emphasize need for improved methods

There exists a critical and unmet need for a rapid, label-free, and quantitative method capable of distinguishing viable from nonviable cancer cells based on intrinsic biophysical properties, without reliance on exogenous labels or complex sample processing. Such a method must be compatible with small sample volumes, operate at the single-cell level, and deliver results in a clinically relevant timeframe to support real-time therapeutic adjustments.

## SUMMARY OF THE INVENTION

- introduce system for classifying biological particles

The present invention introduces a novel system for classifying biological particles, particularly cancer cells, by measuring their impedance response across multiple frequencies and applying machine learning models to interpret the resulting data patterns. This system enables the automated, label-free discrimination of viable and nonviable cells based on their dielectric behavior under alternating electric fields.

- describe impedance response measurement

The system measures the impedance response of individual biological particles as they pass through a microfluidic channel equipped with microelectrodes, applying a range of alternating current frequencies simultaneously. The impedance signal is captured as a function of frequency, yielding amplitude and phase components that reflect changes in cellular membrane integrity, cytoplasmic conductivity, and overall dielectric structure.

- determine physical properties of impedance response data

Physical properties of the impedance response are extracted from the raw signal, including the magnitude of amplitude change and the angular shift in phase relative to the baseline signal observed in the absence of cells. These features are calculated for each detected particle across a spectrum of frequencies, generating a multidimensional signature unique to the cell’s physiological state.

- classify biological particles using machine learning

The extracted features are input into a trained machine learning model that has been optimized to distinguish between viable and nonviable cell populations. The model identifies patterns in the amplitude and phase spectra that correlate with cell death, enabling high-accuracy classification without prior knowledge of cell type or surface marker expression.

- introduce system for determining biological particle type

The invention further encompasses a system for determining the type of biological particle based on its multifrequency impedance signature, enabling not only viability classification but also potential identification of cell lineage or pathological state through comparative analysis against reference datasets.

- describe impedance response measurement

Impedance response is measured using a lock-in amplifier connected to gold microelectrodes patterned within a microfluidic channel. The system applies a multiplexed sinusoidal voltage across a frequency range extending from 300 kHz to 30 MHz, capturing real and imaginary components of the current response for each particle passage.

- determine physical properties of impedance response data

Physical properties are derived through digital signal processing, including detrending to remove baseline drift and noise filtering to isolate single-cell events. Amplitude change is computed as the peak deviation from the baseline voltage, while phase change is determined from the arctangent of the ratio of imaginary to real signal components at each frequency.

- determine biological particle type using machine learning

A supervised machine learning classifier is trained using labeled datasets of known viable and nonviable cells, learning to map the multidimensional impedance feature space to discrete biological classifications. The classifier outputs a probability score indicating the likelihood that a given particle is viable or nonviable.

- introduce method of classifying biological particles

A method is disclosed for classifying biological particles by exposing them to a controlled electric field across multiple frequencies, recording the resulting impedance waveform, extracting amplitude and phase features, and applying a trained machine learning algorithm to assign a classification label based on previously established patterns.

- describe impedance response measurement

The impedance response is measured in real time as cells flow through a microfluidic constriction between two planar electrodes, with each cell causing a transient perturbation in the current flow that is captured as a discrete event in the time-series signal.

- determine physical properties of impedance response data

The physical properties of each event are quantified by comparing the signal envelope and phase trajectory at multiple discrete frequencies, generating a feature vector that encapsulates the cell’s dielectric fingerprint.

- classify biological particles using machine learning

The feature vectors are processed by a machine learning classifier trained on a diverse set of cell states, enabling the system to generalize across cell lines, patient samples, and varying degrees of viability without requiring re-calibration or re-labeling.

## DETAILED DESCRIPTION OF THE INVENTION

- classify biological particles

### A. Methods and Systems for Classifying Biological Particles

- introduce system for classifying biological particles

The system comprises a microfluidic chip with integrated microelectrodes, a multifrequency impedance measurement unit, a data acquisition module, and a computing system executing a machine learning classifier. The components are configured to operate in sequence, enabling continuous, real-time classification of biological particles as they transit the sensing region.

- describe system components

The microfluidic chip is fabricated from polydimethylsiloxane bonded to a glass substrate patterned with gold electrodes, forming a flow channel with a constriction that ensures single-cell passage. The impedance measurement unit includes a lock-in amplifier capable of generating and detecting sinusoidal signals across a range of frequencies from 300 kHz to 30 MHz. The computing system includes a processor, memory, and software for signal processing and classification.

- outline system functionality

Upon introduction of a biological sample into the microfluidic channel, cells are driven by gravity flow past the electrodes. As each cell passes through the sensing region, it modulates the ionic current, producing a transient impedance change that is recorded across multiple frequencies. The raw data is processed to extract amplitude and phase features, which are then classified by a pre-trained machine learning model.

- specify biological particles

The biological particles classified by the system include mammalian cells, particularly epithelial cancer cells such as breast, lung, ovarian, and prostate carcinoma cells, as well as hematological malignancies including lymphoma and myeloma cells.

- provide examples of biological particles

Examples of biological particles include T47D breast cancer cells, MCF-7 cells, HCT116 colon cancer cells, and Jurkat lymphoma cells, both in viable and nonviable states induced by drug exposure, heat shock, or chemical lysis.

- describe cell collection

Cells are collected from culture media or clinical samples such as blood, ascites, or tumor digests, centrifuged to remove debris, and resuspended in a physiologically compatible buffer such as phosphate-buffered saline to maintain viability during measurement.

- specify cell types

The system is capable of distinguishing between live cells with intact membranes and dead cells with compromised membrane integrity, regardless of the mechanism of cell death, including apoptosis, necrosis, or autophagy.

- describe impedance response measurement

Impedance measurements are performed using a sinusoidal excitation signal applied simultaneously at four or more discrete frequencies, with 500 kHz serving as a reference frequency. The real and imaginary components of the current are recorded at each frequency, allowing for the construction of complex impedance spectra for each cell event.

- outline physical property determination

Physical properties are determined by calculating the difference between the peak signal and the baseline signal for amplitude change, and by computing the phase shift relative to the excitation waveform for phase change, using the arctangent of the ratio of the imaginary to real components.

- describe machine learning model application

The machine learning model receives the feature vector for each cell and assigns a classification label—viable or nonviable—based on learned patterns derived from training data. The model operates without requiring prior knowledge of cell surface markers or molecular identity.

- specify machine learning models

The machine learning models include support vector machines with radial basis function kernels, neural networks with multiple hidden layers, logistic regression classifiers, and k-nearest neighbors algorithms, all trained on normalized feature matrices derived from amplitude and phase data.

- describe cancer diagnosis

The system enables cancer diagnosis by determining the proportion of viable tumor cells in a sample following therapeutic exposure, thereby indicating whether the administered treatment is effective. A high percentage of nonviable cells correlates with a favorable therapeutic response.

- specify cancer types

The system is applicable to epithelial carcinomas including breast, lung, colorectal, and ovarian cancers, as well as hematological malignancies such as B-cell lymphoma and multiple myeloma, particularly those expressing activated matriptase or other targetable surface proteases.

- describe advantages over existing methods

The disclosed system offers significant advantages over existing methods by eliminating the need for staining, reducing sample volume requirements to microliters, enabling real-time analysis, and providing higher accuracy through the integration of multidimensional dielectric signatures with machine learning.

- outline label-free and cost-effective approach

The method is entirely label-free, requiring no fluorescent dyes, antibodies, or enzymatic reagents, thereby reducing reagent costs and simplifying workflow. The microfabricated chip is disposable and inexpensive to produce, making the system suitable for widespread clinical deployment.

- describe point-of-care applications

The system is designed for point-of-care use in oncology clinics, pathology laboratories, and hospital wards, where rapid assessment of drug efficacy can guide immediate therapeutic decisions without the need for centralized laboratory infrastructure.

- introduce multi-frequency impedance cytometry

Multi-frequency impedance cytometry enables the simultaneous capture of cellular dielectric properties across a broad frequency spectrum, revealing information about both membrane and intracellular characteristics that are inaccessible at single frequencies.

- describe impedance cytometry experiments

Impedance cytometry experiments are conducted by flowing cell suspensions through the microfluidic channel while recording impedance signals at discrete frequencies, with each cell producing a distinct event corresponding to its unique dielectric signature.

- specify cell preparation

Cells are prepared by washing and resuspending in isotonic buffer at a concentration of approximately 400 cells per microliter, ensuring single-file passage through the sensing region and minimizing overlapping events.

- describe impedance cytometry measurements

Measurements are performed inside a Faraday cage to minimize electromagnetic interference, with data acquired at a sampling rate sufficient to resolve individual cell transits, typically exceeding 10 kHz per channel.

- outline circuit model

The electrical behavior of the system is modeled as a parallel combination of solution resistance and coupling capacitance, with double-layer capacitances at each electrode interface, forming a circuit that accurately represents the impedance perturbations caused by cell passage.

- describe data post-processing

Data post-processing involves detrending to remove low-frequency drift, bandpass filtering to eliminate noise, and peak detection algorithms to isolate individual cell events from the continuous signal stream.

- extract amplitude change feature

The amplitude change feature is extracted by identifying the maximum deviation of the signal envelope from the baseline for each cell event, normalized to account for variations in flow rate and electrode spacing.

- extract phase change feature

The phase change feature is derived by computing the angular difference between the phase of the signal peak and the phase of the baseline signal at each frequency, providing a sensitive indicator of changes in cellular permittivity and conductivity.

- present scatter plots of amplitude change

Scatter plots of amplitude change at 500 kHz versus 20 MHz reveal distinct clustering of viable and nonviable cell populations, with nonviable cells exhibiting significantly reduced amplitude modulation due to loss of membrane integrity.

- present scatter plots of phase change

Scatter plots of phase change at 500 kHz versus 30 MHz demonstrate a consistent shift in phase behavior between viable and nonviable cells, with viable cells showing positive phase shifts at high frequencies and nonviable cells exhibiting attenuated or inverted responses.

- introduce machine learning analysis

Machine learning analysis is employed to identify nonlinear decision boundaries between cell classes, leveraging the full multidimensional feature space to maximize classification accuracy beyond what is achievable with manual thresholding.

- describe machine learning model training

The machine learning model is trained using labeled datasets generated from control samples of 100% viable and 100% nonviable cells, with additional training performed on mixed populations to enhance generalizability across varying viability percentages.

- specify machine learning algorithms

Algorithms include support vector machines with Gaussian kernels, deep neural networks with three hidden layers, logistic regression with L2 regularization, and k-nearest neighbors with k=7, all evaluated for performance using cross-validation.

- describe classification results

Classification results demonstrate accuracy exceeding 95% when both amplitude and phase features are combined, with sensitivity and specificity both above 95%, outperforming traditional dye-based methods in both precision and reproducibility.

- outline system advantages

The system offers advantages including high throughput, minimal sample consumption, rapid analysis time under five minutes, compatibility with clinical workflows, and the ability to preserve cells for downstream molecular analysis.

- describe system applications

Applications include monitoring response to targeted antibody-drug conjugates, evaluating resistance development in real time, screening patient-derived xenografts, and guiding personalized therapy selection in oncology clinics.

- specify system benefits

Benefits include reduced false positives, elimination of reagent costs, compatibility with automated systems, and the potential for integration into handheld diagnostic devices for use in resource-limited settings.

- conclude system description

The system represents a paradigm shift in cellular analysis by replacing chemical labeling with physical characterization, enabling a new standard for rapid, accurate, and cost-effective cancer diagnostics.

- introduce classification of biological particles

Classification of biological particles is achieved through the synergistic integration of microfluidic impedance sensing and machine learning, allowing for the differentiation of cell states based on intrinsic biophysical properties rather than extrinsic markers.

- motivate dielectric properties

Dielectric properties of cells are inherently linked to their structural and functional integrity, with viable cells exhibiting distinct capacitive and resistive behaviors compared to nonviable cells due to membrane integrity, cytoplasmic composition, and organelle organization.

- describe amplitude and phase spectra

The amplitude spectrum reflects changes in ionic resistance caused by cell volume and membrane capacitance, while the phase spectrum captures the time-delayed response of the cell’s dielectric components, providing complementary information for classification.

- apply machine learning model

The machine learning model is applied to the combined amplitude and phase feature matrix to identify complex, nonlinear patterns that distinguish viable from nonviable cells with high confidence, even in heterogeneous populations.

- introduce classification learner toolbox

The classification learner toolbox is employed to systematically evaluate multiple algorithms, optimize hyperparameters, and validate model performance using stratified k-fold cross-validation to ensure robustness across diverse datasets.

- describe feature matrix construction

The feature matrix is constructed by compiling amplitude and phase values for each cell event at four or more frequencies, resulting in an eight-dimensional input vector for each particle, normalized to zero mean and unit variance.

- normalize feature matrix

Normalization is performed using z-score standardization to ensure equal weighting of features and to mitigate the influence of instrumental drift or sample concentration variations.

- train classifier

The classifier is trained on a dataset comprising over 1,000 labeled events from multiple cell lines and viability conditions, with validation performed on independent test sets to prevent overfitting.

- introduce neural network technology

Neural network technology is utilized to model highly nonlinear relationships between impedance features and cell viability, with architectures including fully connected layers, dropout regularization, and rectified linear unit activation functions.

- describe neural network architecture

The neural network architecture comprises an input layer with eight nodes corresponding to the amplitude and phase features, two hidden layers with 64 and 32 nodes respectively, and an output layer with a sigmoid activation function to produce a probability score for viability.

- train neural network

The neural network is trained using the Adam optimizer with a learning rate of 0.001, batch size of 32, and early stopping based on validation loss, achieving convergence within 100 epochs.

- describe neural network application

The trained neural network is deployed in real time on the computing module, where it processes incoming impedance events and outputs a viability classification within milliseconds of detection.

- introduce unsupervised machine learning

Unsupervised machine learning techniques, including hierarchical clustering and principal component analysis, are employed to explore novel cell subpopulations and detect previously unrecognized phenotypes in untreated or resistant samples.

- describe clustering techniques

Clustering techniques group cells based on similarity in impedance signatures, revealing subpopulations that may represent early apoptotic states or drug-resistant variants not distinguishable by traditional methods.

- compare machine learning classifiers

Comparative analysis demonstrates that support vector machines with Gaussian kernels achieve the highest accuracy for binary classification, while neural networks offer superior generalization in heterogeneous clinical samples.

- describe logistic regression

Logistic regression is employed as a baseline classifier, providing interpretable coefficients that quantify the relative contribution of each feature to viability prediction.

- describe K Nearest Neighbors

K Nearest Neighbors is used to classify cells based on proximity to labeled training examples in the feature space, offering a non-parametric approach that adapts to local data density.

- describe Support Vector Machine

Support Vector Machine is the primary classifier, utilizing a radial basis function kernel to map the feature space into a higher-dimensional space where linear separation of viable and nonviable cells is achieved with maximal margin.

- classify white blood cell sub-types

The system is further adaptable to classify white blood cell subtypes, including neutrophils, lymphocytes, and monocytes, based on their distinct dielectric signatures, expanding its utility beyond oncology to immunology and hematology.

- describe SVM classifier

The SVM classifier is trained using a radial basis function kernel with a regularization parameter of 10 and a gamma value of 0.1, achieving optimal performance on the combined amplitude and phase feature set.

- describe electrode fabrication

Electrodes are fabricated on fused silica substrates using photolithography, followed by electron beam metal evaporation of a 10-nm chromium adhesion layer and a 100-nm gold layer, with liftoff processing to define the final electrode geometry.

- describe photolithography process

Photolithography involves spin-coating of photoresist, soft baking, ultraviolet exposure through a chromium mask, development, and hard baking to create a patterned resist layer that defines the electrode layout.

- describe electron beam metal evaporation

Electron beam metal evaporation is performed under high vacuum conditions to deposit thin, uniform layers of chromium and gold with precise thickness control, ensuring low electrical resistance and high surface stability.

- describe liftoff processing

Liftoff processing involves dissolving the photoresist in acetone, leaving behind only the metal deposited on the exposed substrate regions, resulting in clean, well-defined microelectrodes with minimal residue.

- describe microfluidic channel fabrication

Microfluidic channels are fabricated using soft lithography, with SU-8 photoresist patterned on a silicon wafer to create a master mold, followed by pouring and curing of polydimethylsiloxane to form the channel structure.

- describe soft lithography

Soft lithography enables the replication of micron-scale features from a master mold into elastomeric materials, allowing for rapid, low-cost production of microfluidic devices with high fidelity.

- describe SU-8 photo-patterning

SU-8 photo-patterning involves spin-coating, soft baking, UV exposure through a photomask, development in SU-8 developer, and hard baking to create a robust, high-aspect-ratio mold for PDMS replication.

- describe PDMS channel fabrication

PDMS channels are cured at 80°C for two hours, peeled from the mold, punched with inlet and outlet ports, and bonded to the electrode chip using oxygen plasma treatment to form a sealed, hydrophilic flow path.

- describe cell culture

Cell culture is performed in RPMI 1640 medium supplemented with fetal bovine serum, with cells maintained in logarithmic growth phase prior to drug exposure and impedance measurement.

- describe computer system and network

The system includes a client computer for user interface and data visualization, and a server computer for model training and data storage, connected via a secure communications network to enable remote diagnostics and cloud-based analysis.

- describe client computer and server computer

The client computer runs a graphical interface for sample input and result display, while the server computer hosts the trained machine learning models and processes incoming data for classification and reporting.

- describe communications network

The communications network is a secured, encrypted connection compliant with HIPAA and GDPR standards, enabling transmission of patient data and classification results between clinical sites and diagnostic centers.

- describe data module and computing module

The data module captures and preprocesses impedance signals, while the computing module executes the machine learning classifier and outputs a classification result in real time.

- describe internal structure of computer

The internal structure of the computer includes a central processing unit, random-access memory, non-volatile storage, and input/output interfaces, all interconnected via a system bus to enable high-speed data transfer and real-time processing.

- describe system bus and I/O device interface

The system bus facilitates communication between the processor, memory, and peripheral devices, while the I/O interface connects the impedance measurement unit, display, and network adapter to the computing core.

- describe computer program product

A computer program product is disclosed, comprising a non-transitory computer-readable medium storing executable instructions that, when executed by a processor, cause the system to perform the steps of impedance measurement, feature extraction, and machine learning-based classification of biological particles.

### B. Definitions

- define subject and patient

The term “subject” refers to a mammalian organism, and “patient” refers to a human subject under medical evaluation or treatment, including individuals suspected of having cancer or undergoing therapeutic intervention.

- specify vertebrates included

The subject may include any vertebrate, including but not limited to humans, non-human primates, dogs, cats, rabbits, and mice, with particular application to human patients in clinical settings.

- define normal, control, or reference subject

A “normal,” “control,” or “reference” subject is an individual without a diagnosed malignancy or with a known benign condition, used to establish baseline impedance signatures for comparison.

- define sample, test sample, and patient sample

A “sample” refers to any biological material containing cells, a “test sample” is a sample exposed to a therapeutic agent, and a “patient sample” is a sample derived directly from a subject’s bodily fluid or tissue.

- specify sample types

Sample types include whole blood, peripheral blood mononuclear cells, ascites fluid, pleural effusion, tumor digestates, and culture media containing suspended cells.

- describe sample preparation

Sample preparation involves centrifugation, washing in phosphate-buffered saline, resuspension at a defined cell concentration, and filtration to remove clumps or debris prior to introduction into the microfluidic system.

- define biological sample

A “biological sample” is any material derived from a living organism that contains one or more cells, including cells in suspension, tissue homogenates, or bodily fluids.

- specify cell types, tissues, and bodily fluids

Cell types include epithelial, hematopoietic, and mesenchymal cells; tissues include tumor biopsies and lymph nodes; bodily fluids include blood, lymph, cerebrospinal fluid, and pleural fluid.

- describe methods for collecting and processing samples

Samples are collected via venipuncture, fine-needle aspiration, or surgical resection, processed within two hours of collection, and maintained at 4°C until analysis to preserve cell viability.

- define determining, measuring, assessing, and assaying

The terms “determining,” “measuring,” “assessing,” and “assaying” are used interchangeably to refer to the quantitative evaluation of a biological parameter, including the classification of cell viability.

- define diagnosis

“Diagnosis” refers to the identification of the presence, type, or stage of a disease, including cancer, based on the classification of viable versus nonviable tumor cells in a patient sample.

- describe diagnostic methods

Diagnostic methods include the use of the disclosed system to classify cell viability following therapeutic exposure, with results interpreted in the context of clinical history and imaging findings.

- define prognosis

“Prognosis” refers to the prediction of the likely course and outcome of a disease, determined by the proportion of nonviable cells after drug exposure, with higher proportions indicating favorable prognosis.

- describe determining prognosis

Prognosis is determined by comparing the percentage of nonviable cells in a test sample to a predefined threshold derived from clinical validation studies, with values above the threshold indicating a high likelihood of therapeutic success.

- specify singular and plural forms

All singular terms include their plural forms unless context dictates otherwise, and vice versa, to encompass all possible embodiments within the scope of the claims.

- define including, comprising, containing, and having

The terms “including,” “comprising,” “containing,” and “having” are used synonymously and are intended to be non-exclusive, allowing for the presence of additional elements not explicitly listed.

- describe phrases in one embodiment

Phrases such as “in one embodiment,” “in another embodiment,” or “in a preferred embodiment” are used to describe optional features that may be combined or substituted without departing from the scope of the invention.

- define and/or and I

The term “and/or” means either or both, and the term “I” is not used in this context except as a grammatical pronoun in narrative descriptions.

- define substantially

The term “substantially” refers to a degree of completeness or accuracy that is sufficient to achieve the intended purpose, including variations of up to ±10% from a stated value.

- define approximately or about

The terms “approximately” or “about” indicate a range of ±5% around a stated numerical value, unless otherwise specified, to account for experimental variability and measurement tolerance.

- describe ranges of values

Ranges of values are inclusive of their endpoints unless otherwise indicated, and any range disclosed encompasses all sub-ranges within it.

- describe each in reference to a collection

Each element or step described herein is intended to be applicable individually or in combination with any other element or step, forming a collection of embodiments within the scope of the invention.

- specify examples and exemplary language

Examples and exemplary language are provided for illustrative purposes only and are not intended to limit the scope of the invention, which is defined solely by the claims.