Here is the complete patent application following your outline precisely:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates generally to the field of cancer diagnostics and personalized medicine. More specifically, the invention pertains to systems and methods for classifying biological particles, particularly circulating tumor cells (CTCs), through impedance response measurements and machine learning analysis. The disclosed technology enables label-free, rapid assessment of cell viability in response to targeted cancer therapies by analyzing dielectric properties across multiple frequencies. This represents a significant advancement over conventional staining-based viability assays by providing a cost-effective, portable solution for point-of-care cancer management.  

## BACKGROUND OF THE INVENTION  

Circulating cancer cells have emerged as critical biomarkers for cancer diagnosis, prognosis, and treatment monitoring. These rare cells detach from primary tumors and enter the bloodstream, carrying valuable information about tumor biology and therapeutic response. Current detection methods for circulating tumor cells face substantial limitations in sensitivity, specificity, and practicality. Traditional approaches relying on immunostaining or fluorescence labeling suffer from antibody cross-reactivity, sample processing artifacts, and inability to perform downstream molecular analysis after staining.  

The analysis of CTCs holds tremendous clinical importance for understanding metastatic potential, evaluating treatment efficacy, and enabling personalized therapeutic strategies. Existing CTC detection technologies including CellSearch® and microfluidic isolation platforms demonstrate poor recovery rates (typically <80%) and require complex instrumentation. These systems struggle with the fundamental challenges of CTC detection: extreme rarity in blood (1-10 CTCs per mL among billions of blood cells), phenotypic heterogeneity, and morphological similarity to normal blood cells.  

There remains an urgent need for improved methods that can overcome these limitations while providing rapid, cost-effective analysis at the point-of-care. Current technologies fail to meet the requirements for widespread clinical adoption due to their reliance on bulky equipment, expensive reagents, and labor-intensive protocols. The present invention addresses these challenges through an innovative combination of multi-frequency impedance cytometry and machine learning classification.  

## SUMMARY OF THE INVENTION  

The present invention introduces a novel system for classifying biological particles based on their dielectric properties. The system measures impedance response across multiple frequencies as particles flow through a microfluidic channel, capturing comprehensive electrical signatures. Physical properties including amplitude change and phase change are extracted from the impedance response data at discrete frequencies ranging from 300 kHz to 30 MHz. These features form a multidimensional dataset that enables highly accurate classification through machine learning algorithms.  

In one embodiment, the invention provides a system for determining biological particle type through label-free impedance analysis. The system comprises microfabricated electrodes integrated with a microfluidic channel, a multi-frequency lock-in amplifier for impedance measurement, and computational modules for data processing and classification. The system extracts both amplitude and phase spectra from the impedance response, which reflect the dielectric properties of cell membranes and intracellular components. Machine learning models including support vector machines (SVMs) with Gaussian kernels are trained to distinguish particle types based on these spectral features.  

The invention further discloses a method of classifying biological particles that includes the steps of: measuring impedance response at multiple frequencies simultaneously, determining physical properties from the response data, normalizing the features into a matrix, and applying trained machine learning classifiers. This method achieves superior accuracy (>95%) in distinguishing live from dead cancer cells, enabling rapid assessment of therapeutic efficacy. The label-free nature of the technology preserves cells for downstream analysis while eliminating costly staining reagents and complex optical systems.  

## DETAILED DESCRIPTION OF THE INVENTION  

### A. Methods and Systems for Classifying Biological Particles  

The present invention provides an integrated system for classifying biological particles based on their dielectric properties. The system comprises three main components: a microfluidic impedance cytometer, a multi-frequency measurement module, and machine learning classification software. The microfluidic component features gold electrodes fabricated on glass substrates using photolithography and electron beam evaporation, with precisely defined electrode gaps (25 μm) optimized for single-cell analysis. The microfluidic channel, fabricated in PDMS through soft lithography, has dimensions (100 μm width × 30 μm height) that enable hydrodynamic focusing of cells past the detection electrodes.  

The system operates by measuring impedance changes as cells transit through the detection region between electrodes. Unlike conventional single-frequency approaches, the invention simultaneously applies and measures response at four discrete frequencies (e.g., 500 kHz, 20 MHz, 25 MHz, and 30 MHz). This multi-frequency strategy captures both membrane characteristics (low frequency response) and intracellular properties (high frequency response) in a single measurement. The system utilizes a lock-in amplifier to precisely measure amplitude and phase changes caused by cells modulating the electric field.  

For biological particle analysis, the system is particularly suited for classifying cancer cell viability in response to targeted therapies. In one embodiment, T47D breast cancer cells are treated with anti-matriptase-conjugated drugs and analyzed to determine the percentage of viable versus apoptotic cells. The system processes the raw impedance data through several stages: detrending to remove baseline drift, denoising to improve signal quality, feature extraction (amplitude change and phase change at each frequency), and normalization to create an 8-dimensional feature vector (amplitude and phase at four frequencies).  

Machine learning models are trained on known samples (100% viable and 100% non-viable cells) to establish classification boundaries in the feature space. The invention employs several classifier types including support vector machines (SVMs) with Gaussian kernels, logistic regression, and neural networks. In preferred embodiments, SVMs achieve >95% accuracy by mapping features into higher-dimensional space using kernel functions. The trained models can then predict viability percentages in test samples with varying ratios of live/dead cells (e.g., 90%, 82%, 50% viable).  

Key advantages over existing technologies include: 1) elimination of labeling reagents and associated costs, 2) preservation of cells for downstream molecular analysis, 3) reduced sample volume requirements (<50 μL), 4) rapid analysis (<5 minutes), and 5) compatibility with point-of-care implementation. The system's microfluidic design enables gravity-driven flow without pumps, simplifying operation while minimizing electronic noise. Electrode fabrication using gold ensures corrosion resistance and stable performance over repeated measurements.  

The multi-frequency impedance cytometry approach provides several technical innovations. First, the simultaneous measurement at strategically selected frequencies enables comprehensive dielectric profiling not achievable with single-frequency systems. Second, the combination of amplitude and phase features at these frequencies creates a highly discriminative feature space for machine learning. Third, the integration of microfabricated electrodes with PDMS microfluidics allows for low-cost, disposable cartridges suitable for clinical use.  

For cancer diagnosis applications, the system can classify various tumor types including B-cell lymphoma, multiple myeloma, and epithelial carcinomas. The technology is particularly valuable for assessing patient-specific response to targeted therapies by analyzing tumor cells exposed to drug candidates ex vivo. This enables personalized treatment selection while avoiding systemic toxicity from ineffective drugs.  

### B. Definitions  

As used throughout this specification, the following terms shall have the meanings specified:  

The terms "subject" and "patient" refer to a human or other vertebrate including mammals such as dogs, cats, livestock, and laboratory animals.  

A "normal", "control", or "reference" subject refers to an individual not afflicted with the condition being studied, providing baseline measurements for comparison.  

The terms "sample", "test sample", and "patient sample" refer to biological material obtained from a subject, including but not limited to blood, tissue biopsies, pleural effusions, or dissociated tumor specimens. Sample preparation may include centrifugation, filtration, or other processing steps to isolate cells of interest.  

A "biological sample" encompasses any material containing biological particles for analysis, including cell suspensions, bodily fluids, or tissues. The sample may contain various cell types such as circulating tumor cells, leukocytes, or cultured cell lines.  

The terms "determining", "measuring", "assessing", and "assaying" refer to both qualitative and quantitative analysis of sample properties using the disclosed methods.  

"Diagnosis" refers to identifying the presence, nature, or characteristics of a pathological condition. Diagnostic methods include detecting specific cell types or viability states indicative of disease or treatment response.  

"Prognosis" refers to predicting the likely course or outcome of a disease, including response to particular therapies. Determining prognosis may involve analyzing cell viability patterns after drug exposure.  

The singular forms "a", "an", and "the" include plural referents unless the context clearly dictates otherwise. The term "or" is inclusive, meaning "and/or".  

The terms "including", "comprising", "containing", and "having" are open-ended and indicate the presence of stated features but permit additional elements.  

"Substantially" means within 10% of the stated value or characteristic. "Approximately" or "about" refers to ±15% of the specified measurement.  

Ranges of values are inclusive of all intermediate values and endpoints. References to "each" in relation to a collection apply to every member individually where applicable.  

Examples and exemplary language ("e.g.", "such as") are merely illustrative and do not limit the scope of the claimed invention unless expressly stated.