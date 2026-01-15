# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of medical diagnostics and imaging, particularly to methods and apparatus for quantitative analysis of tissue perfusion in peripheral tissues. More specifically, the invention provides a novel approach to measuring functional tissue perfusion rates by analyzing the spatiotemporal dynamics of indocyanine green (ICG) fluorescence following intravenous administration. This technique enables the generation of high-resolution perfusion maps and predictive tissue necrosis probability maps, which are valuable for assessing vascular insufficiency, guiding therapeutic interventions, and evaluating treatment efficacy in conditions such as peripheral arterial disease, diabetic complications, and post-ischemic recovery. The invention is applicable in both preclinical research and clinical settings, offering a non-invasive, cost-effective, and quantitatively robust alternative to existing perfusion imaging modalities.

## BACKGROUND

Laser Doppler imaging (LDI) has long been regarded as a standard tool for in vivo blood flow measurement in superficial tissues. However, LDI suffers from significant limitations that restrict its utility in quantitative perfusion analysis. The technique provides only relative, rather than absolute, measurements of blood flow, making inter-individual comparisons unreliable. Moreover, LDI is highly sensitive to motion artifacts and ambient light interference, and its spatial resolution is insufficient to distinguish between microvascular and macrovascular perfusion compartments. These shortcomings render LDI inadequate for predicting tissue viability or monitoring subtle changes in perfusion during therapeutic interventions.

X-ray blood vessel imaging, including conventional angiography and micro-CT angiography, offers high-resolution structural visualization of the vasculature. While useful for identifying anatomical occlusions or stenoses, these modalities do not provide functional information about actual tissue perfusion. Blood may be rerouted through collateral vessels that are not apparent in static structural images, leading to inaccurate prognostic assessments. Furthermore, X-ray-based techniques involve ionizing radiation and often require contrast agents that may pose risks to patients with renal impairment.

Indocyanine green (ICG) angiography has emerged as a clinically approved method for visualizing vascular architecture using near-infrared (NIR) fluorescence. ICG binds to serum albumin upon intravenous injection and remains intravascular under normal physiological conditions, enabling real-time imaging of blood flow. However, traditional ICG angiography is primarily qualitative, relying on maximal fluorescence intensity as a surrogate for perfusion. This approach is confounded by variations in tissue optical properties, depth of vasculature, and systemic pharmacokinetics, limiting its quantitative accuracy.

The ICG elimination test, historically used to assess liver function, measures the rate at which ICG is cleared from systemic circulation via hepatic excretion. While this provides information about overall hepatic capacity, it does not yield regional perfusion data for peripheral tissues. Moreover, the elimination kinetics are influenced by multiple factors—including anesthesia, cardiac output, and liver health—making them unsuitable as a direct indicator of local tissue perfusion.

A critical shortcoming of ICG dynamics in prior art is the lack of a mathematical framework to translate temporal fluorescence profiles into absolute perfusion rates. Previous studies attempted to correlate peak fluorescence intensity or time-to-peak with perfusion but failed to account for systemic variations in ICG clearance and vascular input functions. As a result, these methods could not support comparative analyses across subjects or time points.

Liver function tests, while essential for assessing hepatic health, are entirely unrelated to peripheral tissue perfusion and cannot inform decisions regarding vascular insufficiency or tissue viability. The absence of a reliable, quantitative, and non-invasive method for measuring functional perfusion in peripheral tissues has thus remained a significant unmet need in both clinical practice and biomedical research. There is a compelling motivation to develop a new perfusion measurement method that overcomes the limitations of existing technologies by providing absolute, spatially resolved, and physiologically meaningful perfusion metrics derived from ICG fluorescence dynamics.

## DISCLOSURE

### Technical Problem

The technical problem addressed by the present invention is the inability of existing imaging modalities to provide quantitative, spatially resolved, and functionally relevant measurements of tissue perfusion in peripheral tissues. Current methods either offer only qualitative or relative assessments, lack the sensitivity to predict tissue necrosis, or fail to distinguish between microvascular and macrovascular perfusion components. This deficiency impedes accurate diagnosis of vascular insufficiency, objective evaluation of therapeutic interventions, and reliable prognosis prediction in ischemic conditions.

### Technical Solution

The present invention proposes a tissue perfusion analysis apparatus and associated method that leverages the spatiotemporal dynamics of indocyanine green (ICG) fluorescence to compute absolute perfusion rates on a per-pixel basis. By mathematically modeling the relationship between ICG pharmacokinetics and tissue perfusion, and by normalizing for systemic variations in ICG clearance using subject-specific parameters, the invention enables the construction of high-resolution perfusion maps and predictive tissue necrosis probability maps. This solution provides a quantitative, non-invasive, and clinically translatable approach to functional perfusion assessment.

## BEST MODE

### Introduction to Analysis Apparatus

The best mode of carrying out the invention involves a tissue perfusion analysis apparatus comprising an imaging system, a data processing unit, and an output interface. The apparatus is designed to acquire time-series near-infrared (NIR) fluorescence images following intravenous injection of ICG, process the image data to extract perfusion-relevant parameters, and generate quantitative perfusion maps and necrosis probability predictions.

### Photodetector

The apparatus includes a photodetector, preferably a charge-coupled device (CCD) or complementary metal-oxide-semiconductor (CMOS) camera, optimized for NIR wavelengths. The photodetector is equipped with a band-pass filter centered at approximately 830 nm to selectively capture ICG fluorescence emission while rejecting excitation light and background noise.

### Numerical Conversion Means

The acquired analog fluorescence signals are converted into digital pixel intensity values through an analog-to-digital converter. These numerical values represent the relative fluorescence intensity at each spatial location over time and serve as the raw input for subsequent perfusion calculations.

### Perfusion Rate Calculation Means

A dedicated processing module implements a mathematical model derived from Fick’s principle and first-order pharmacokinetics to compute the perfusion rate for each pixel in the region of interest. The model incorporates the time-to-peak (Tmax) of ICG fluorescence and the systemic ICG half-life, obtained from a reference region such as the trunk, to calculate absolute perfusion rates expressed as a percentage of vascular volume exchanged per minute (%/min).

### Output Means

The calculated perfusion rates are rendered through an output means, such as a display monitor or printer, in the form of pseudocolor-coded perfusion maps. Additionally, the system can generate and output tissue necrosis probability maps based on empirically derived correlations between perfusion rates and observed necrosis outcomes.

### ICG Fluorescence Dynamics

Following intravenous bolus injection, ICG circulates bound to albumin and exhibits characteristic fluorescence dynamics in tissue regions. In well-perfused tissues, ICG arrives rapidly, peaks quickly, and clears exponentially. In ischemic tissues, arrival is delayed, peak intensity is reduced, and clearance is prolonged due to diminished blood flow.

### Tmax Calculation

For each pixel in the region of interest, the time-to-peak (Tmax) is determined as the time point at which the first derivative of the fluorescence intensity curve equals zero. This parameter is critical for perfusion rate computation, as it reflects the balance between inflow and outflow kinetics.

### Simulation of ICG Dynamics

Computational simulations based on compartmental modeling confirm that varying perfusion rates produce distinct ICG temporal profiles. These simulations validate the theoretical relationship between Tmax, systemic ICG half-life, and local perfusion rate, ensuring the model’s biological plausibility.

### Relationship Between Tmax and Perfusion Rate

The invention establishes a quantitative inverse relationship between Tmax and perfusion rate: lower perfusion results in longer Tmax. By incorporating the systemic ICG half-life (τ), the perfusion rate (P) is calculated using the derived equation:  
P = (1/τ) × ln(1 + τ / Tmax).  
This equation normalizes for inter-subject variability in hepatic clearance and hemodynamics.

### Perfusion Map

A perfusion map is constructed by assigning a color to each pixel based on its calculated perfusion rate. High perfusion rates (e.g., >300%/min) are represented in red, moderate rates in yellow/green, and low rates (<50%/min) in blue, enabling intuitive visual interpretation.

### Color Representation Method

The color representation employs a continuous pseudocolor scale calibrated to known perfusion thresholds. This allows clinicians to instantly recognize regions at risk of necrosis, typically those with perfusion rates below 20%/min.

### Tissue Necrosis Probability Map

Based on longitudinal studies correlating early post-operative perfusion rates with necrosis outcomes on postoperative day 7, an inverse sigmoidal function is used to convert perfusion rates into necrosis probabilities. This function is implemented in software to generate a predictive necrosis map.

### Measurement Apparatus

The complete measurement apparatus includes a light source, optical filters, a detector, and an analysis unit. It is configured for non-contact, in vivo imaging of small animals or human extremities.

### Light Source

A near-infrared light-emitting diode (LED) array emitting at 760 nm is used to excite ICG. The light source is arranged to provide uniform illumination over the target tissue area.

### Filter

An 830-nm band-pass filter is positioned in front of the detector to isolate ICG fluorescence emission, minimizing contamination from reflected excitation light and autofluorescence.

### Detector

The detector captures sequential fluorescence images at high temporal resolution (e.g., 1-second intervals) to accurately resolve the rapid dynamics of ICG in normal tissues.

### Analysis Apparatus

The analysis apparatus comprises a computer running specialized software that performs image registration, region-of-interest masking, Tmax extraction, perfusion calculation, and map generation.

### ICG Injection

A standardized bolus of ICG (e.g., 0.1 mL of 400 µmol/L) is administered intravenously, typically via the tail vein in murine models or peripheral vein in humans.

### Light Radiation

Immediately after injection, the tissue is illuminated with 760-nm light, and fluorescence emission is captured continuously for 10–12 minutes.

### Fluorescence Detection

The detector records the spatiotemporal evolution of ICG fluorescence, producing a time-series image stack where each frame represents the fluorescence distribution at a specific time point.

### Data Processing

Image processing algorithms align frames, subtract background, and extract fluorescence intensity curves for each pixel. A reference region (e.g., trunk) is used to determine the systemic ICG half-life (τ).

### Perfusion Rate Calculation

Using the derived equation and measured Tmax and τ values, the perfusion rate is computed for every pixel in the ischemic and normal limb regions.

### Output of Perfusion Rates

Numerical perfusion rates are displayed as histograms or tabular data, allowing statistical comparison between groups or time points.

### Perfusion Map Construction

Pixel-wise perfusion rates are mapped to a color scale and overlaid on a silhouette or grayscale image of the tissue, creating an intuitive visual representation of perfusion heterogeneity.

### Tissue Necrosis Probability Prediction

The necrosis probability for each pixel is calculated using the inverse sigmoidal function and displayed as a separate color-coded map, enabling preemptive identification of at-risk tissue.

### Measurement Method

The overall method involves ICG injection, dynamic imaging, data extraction, mathematical modeling, and predictive mapping.

### ICG Blood Vessel Image Diagram

Initial frames of the time-series can be used to generate a static angiogram showing vascular architecture, though this is secondary to the dynamic perfusion analysis.

### Tmax Acquisition

Tmax is automatically determined by locating the maximum of the fluorescence intensity curve for each pixel, validated by zero-crossing of the first derivative.

### Perfusion Rate Calculation

The core innovation lies in the transformation of Tmax and τ into an absolute perfusion metric, enabling cross-subject and longitudinal comparisons.

### Output of Perfusion Rates

Results are exported in standard formats (e.g., DICOM, CSV) for integration with electronic health records or research databases.

### Perfusion Map Construction

Maps are generated in real-time or post-processing, with options for zoom, pan, and region-specific statistics.

### Tissue Necrosis Probability Prediction

The predictive map is validated against histological and clinical outcomes, demonstrating high sensitivity and specificity for necrosis forecasting.

### Data Processing

All computations are performed using optimized algorithms in C++ or MATLAB, ensuring rapid turnaround even for high-resolution datasets.

### Software Implementation

The software includes a graphical user interface for protocol setup, real-time monitoring, and result visualization, suitable for both research and clinical environments.

### Output Device

Results are displayed on high-resolution monitors and can be printed or saved for documentation and telemedicine applications.

### MODE FOR INVENTION

The invention may be embodied in various configurations, including handheld devices for bedside use, integrated surgical imaging systems, or preclinical research platforms. Alternative implementations may employ different fluorophores, multi-spectral imaging, or machine learning enhancements to refine perfusion estimates.

## Comparative Example 1

### Prediction for Tissue Necrosis through Doppler Imaging

In experimental studies using a murine hindlimb ischemia model, laser Doppler imaging (LDI) was employed to assess blood flow immediately after surgery. While LDI detected a general reduction in perfusion in ischemic limbs compared to contralateral controls, it failed to distinguish between animals that would later develop severe necrosis versus those that would recover fully. The blood flow ratios (ischemic/normal) showed overlapping distributions across outcome groups, with no statistically significant correlation to necrosis severity scores on postoperative day 7. This limitation arises because LDI measures relative flux rather than absolute perfusion and cannot resolve spatial heterogeneity within the ischemic region. Consequently, LDI lacks the sensitivity required for individualized prognosis prediction, underscoring the need for the quantitative method disclosed herein.

## Example 1

### Establishment of Method of Measuring Perfusion Using ICG

The foundation of the invention is a mathematical equation derived from first principles of tracer kinetics and Fick’s law of diffusion. Assuming steady-state vascular volume and first-order ICG clearance, the fluorescence intensity in a tissue region is modeled as the convolution of the vascular input function (VIF) and the tissue impulse response. The VIF is approximated as a monoexponential decay with time constant τ, reflecting systemic ICG half-life. For an ischemic region with perfusion rate P, the fluorescence intensity FI(t) is given by:  
FI(t) = Vd × ∫₀ᵗ [exp(−(t−u)/τ) × P × exp(−P u)] du,  
where Vd is vascular density. Solving this integral yields:  
FI(t) = Vd × [exp(−P t) − exp(−t/τ)] / (1/τ − P).  
The time-to-peak (Tmax) occurs when dFI/dt = 0, leading to:  
exp(−P Tmax) × P = exp(−Tmax/τ) / τ.  
Rearranging gives:  
P = (1/τ) × ln(1 + τ / Tmax).  
This equation was validated through ICG fluorescence experiments in murine models. Normal hindlimbs exhibited rapid Tmax (~20 s) and high perfusion rates (~475%/min), while ischemic limbs showed delayed Tmax (>60 s) and low perfusion (<60%/min). In silico simulations of ischemic tissue ICG dynamics with varying P values reproduced the observed fluorescence curves, confirming model accuracy. The concept of perfusion rate as %/min provides a physiologically intuitive metric. The correlation coefficient between Tmax and perfusion rate was found to be strongly inverse and nonlinear, justifying the logarithmic relationship in the final equation. Thus, the relationship between Tmax and perfusion rate is rigorously defined, enabling pixel-wise perfusion quantification.

## Example 2

### Measurement of Perfusion Using Indocyanine Green and the Construction of Perfusion Map and Tissue Necrosis Probability Map Based on Correlation Coefficient

A murine hindlimb ischemia model was used to establish a blood perfusion reduction paradigm. Immediately after femoral artery ligation, ICG perfusion imaging was performed, and perfusion rates were calculated for each pixel in the ischemic limb. Regions were classified as necrotic or non-necrotic on postoperative day 7 based on visual inspection. Analysis revealed that pixels with perfusion rates below 20%/min had a >90% probability of necrosis, while those above 60%/min rarely necrosed. A logistic regression model yielded an inverse sigmoidal relationship:  
Necrosis Probability = 1 / (1 + exp(a × (P − b))),  
where P is perfusion rate, and a, b are fitted parameters. Using this function, a tissue necrosis probability map was constructed by assigning a probability value to each pixel based on its perfusion rate. The map was color-coded (e.g., blue for low probability, red for high), and validation against actual necrosis patterns showed >85% spatial concordance. This demonstrates that early perfusion measurement can reliably predict tissue fate, enabling preemptive intervention.

## INDUSTRIAL APPLICABILITY

The perfusion measurement method disclosed herein has broad industrial applicability in medical diagnostics, drug development, and surgical planning. In clinical settings, it can guide revascularization procedures in peripheral artery disease, monitor flap viability in reconstructive surgery, and assess diabetic foot ulcer risk. In pharmaceutical research, it enables objective quantification of proangiogenic drug efficacy in preclinical models. The apparatus can be integrated into operating rooms, wound care centers, and imaging suites, offering a cost-effective, non-invasive, and quantitative alternative to current perfusion assessment tools. Its compatibility with existing ICG protocols and regulatory approval status facilitates rapid clinical translation.