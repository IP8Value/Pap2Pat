# DESCRIPTION

## TECHNICAL FIELD

- introduce tissue perfusion analysis

Tissue perfusion analysis constitutes a critical diagnostic modality for evaluating the functional integrity of vascular networks in peripheral and deep tissues, particularly under pathological conditions such as ischemia, diabetic microangiopathy, or atherosclerotic occlusion. The ability to quantitatively measure the rate at which blood flows through capillary beds and collateral vessels provides indispensable insight into tissue viability, therapeutic response, and prognostic outcomes. Conventional methods for assessing perfusion often rely on structural imaging of vasculature, which fails to capture dynamic functional parameters essential for predicting tissue survival. This invention introduces a novel, non-invasive, and quantitative method for tissue perfusion analysis based on the spatiotemporal dynamics of indocyanine green (ICG) fluorescence following intravenous administration. The method enables precise, pixel-level quantification of perfusion rates in living tissues through mathematical modeling of ICG pharmacokinetics, allowing for the generation of high-resolution perfusion maps and predictive tissue necrosis probability maps. The system is particularly suited for use in preclinical research and clinical settings where real-time, cost-effective, and repeatable assessment of micro- and macrovascular function is required, without the need for ionizing radiation, expensive imaging modalities, or invasive procedures.

## BACKGROUND

- describe limitations of laser Doppler imaging

Laser Doppler imaging (LDI) has been widely adopted as a non-invasive tool for assessing cutaneous and subcutaneous blood flow by detecting Doppler shifts in reflected laser light from moving erythrocytes. However, LDI is fundamentally limited by its inability to distinguish between flow velocity and vessel density, resulting in measurements that reflect a composite of hemodynamic and structural variables rather than true perfusion rate. Furthermore, LDI is highly sensitive to probe positioning, tissue optical properties, and motion artifacts, leading to poor reproducibility across subjects and time points. Its spatial resolution is insufficient to resolve microvascular heterogeneity within a tissue region, and it provides no quantitative calibration against physiological standards, rendering inter-individual comparisons unreliable. In ischemic models, LDI often fails to detect subtle but clinically significant differences in perfusion that correlate with tissue outcome, thereby limiting its utility in prognostic applications.

- describe X-ray blood vessel imaging

X-ray-based blood vessel imaging, including digital subtraction angiography and micro-computed tomography (micro-CT), offers high-resolution structural visualization of vascular architecture. These techniques excel in delineating the morphology of arteries and veins, including collateral networks and stenotic lesions. However, they provide no direct information regarding the functional rate of blood flow through these vessels. The requirement for iodinated contrast agents, ionizing radiation, and complex post-processing renders these methods unsuitable for repeated measurements, especially in longitudinal studies. Moreover, X-ray imaging cannot differentiate between patent vessels that are functionally occluded due to low flow and those that actively perfuse surrounding tissue, thereby offering limited predictive value for tissue viability.

- describe ICG angiography

Indocyanine green (ICG) angiography has gained clinical traction for real-time visualization of vascular anatomy during surgical procedures, particularly in ophthalmology and neurosurgery. ICG, when excited by near-infrared light, emits fluorescence that allows for high-contrast imaging of vascular structures due to its albumin-binding properties and minimal extravasation in healthy tissues. However, traditional ICG angiography relies on qualitative assessment of fluorescence intensity and temporal patterns, without conversion into quantitative perfusion metrics. The resulting images are influenced by tissue depth, optical scattering, and systemic clearance kinetics, leading to significant inter-subject variability that precludes standardized interpretation. Without mathematical normalization, ICG angiography remains a descriptive tool incapable of supporting diagnostic or prognostic decision-making.

- describe ICG elimination test

The ICG elimination test, commonly used to assess hepatic function, measures the rate at which ICG is cleared from the bloodstream following intravenous injection, typically via spectrophotometric analysis of plasma samples. While this test provides a global estimate of liver metabolic capacity, it lacks spatial resolution and cannot be applied to peripheral tissue perfusion. It assumes uniform systemic distribution and hepatic extraction, ignoring regional variations in microvascular flow that are critical for predicting tissue ischemia. Consequently, the ICG elimination test is irrelevant for evaluating perfusion in extremities, organs, or tumors where local hemodynamics determine outcome.

- describe shortcomings of ICG dynamics

Although ICG fluorescence dynamics have been explored in prior studies for estimating cerebral blood flow and tissue oxygenation, these efforts have been hindered by the absence of a validated mathematical framework that accounts for systemic pharmacokinetic variability. Most approaches correlate peak fluorescence intensity or time-to-peak with perfusion without normalizing for individual differences in hepatic clearance, anesthetic depth, or cardiac output. As a result, these methods produce relative, rather than absolute, perfusion values that cannot be compared across subjects or time points. Furthermore, the lack of pixel-level computational modeling prevents the generation of spatially resolved perfusion maps, limiting utility to broad regional assessments.

- describe liver function test

Liver function tests, including serum bilirubin, transaminases, and synthetic markers, provide indirect and delayed indicators of hepatic health but offer no insight into peripheral tissue perfusion. These biochemical assays reflect systemic metabolic status rather than localized hemodynamic conditions and are not responsive to acute changes in microvascular flow. Their inability to correlate with tissue viability renders them unsuitable for guiding interventions in ischemic conditions such as critical limb ischemia or flap surgery.

- motivate need for new perfusion measurement method

There exists a critical unmet need for a non-invasive, quantitative, and spatially resolved method capable of measuring tissue perfusion with sufficient sensitivity to predict functional outcomes, distinguish microvascular from macrovascular contributions, and enable longitudinal monitoring of therapeutic interventions. Current techniques are either too qualitative, too invasive, too expensive, or too insensitive to capture the heterogeneity of perfusion that determines tissue survival. A method that translates ICG fluorescence dynamics into absolute perfusion rates, calibrated against systemic clearance kinetics and mapped across tissue regions, would revolutionize the diagnosis and management of vascular insufficiency, enhance the precision of preclinical research, and improve clinical decision-making in vascular surgery, oncology, and regenerative medicine.

## DISCLOSURE

### Technical Problem

- state problem of measuring perfusion rate

The fundamental challenge in measuring tissue perfusion lies in the inability of existing technologies to convert observable optical signals into absolute, spatially resolved, and physiologically meaningful perfusion rates that are independent of systemic variability and tissue optical properties. Perfusion, defined as the volume of blood delivered per unit time per unit mass of tissue, is a dynamic parameter that cannot be accurately inferred from static fluorescence intensity, Doppler shifts, or structural vascular imaging. Existing methods fail to account for inter-individual differences in ICG clearance kinetics, which are influenced by hepatic function, anesthesia, and hemodynamic status, leading to inconsistent and non-comparable measurements. Without a robust mathematical model that decouples local perfusion from systemic pharmacokinetics, it is impossible to generate reliable perfusion maps or predict tissue necrosis with clinical precision. This limitation impedes the development of personalized therapeutic strategies and the objective evaluation of interventions aimed at restoring microvascular function.

### Technical Solution

- propose tissue perfusion analysis apparatus

The invention provides a tissue perfusion analysis apparatus comprising a near-infrared fluorescence imaging system, a numerical conversion module, a perfusion rate calculation engine, and an output interface. The apparatus is configured to acquire time-series fluorescence images of indocyanine green (ICG) following intravenous administration, normalize these signals against a systemic reference region to account for hepatic clearance kinetics, and compute, on a pixel-by-pixel basis, the perfusion rate of each tissue region using a derived mathematical relationship between the time-to-peak (Tmax) of ICG fluorescence and the underlying perfusion rate. The apparatus further includes software for generating a spatially resolved perfusion map and a tissue necrosis probability map, both of which are displayed in color-coded formats for intuitive interpretation. The system enables real-time, non-invasive, quantitative assessment of perfusion in peripheral tissues with sub-millimeter resolution, overcoming the limitations of prior methods by integrating dynamic pharmacokinetic modeling with high-speed optical detection.

## BEST MODE

- introduce analysis apparatus

The tissue perfusion analysis apparatus is composed of a near-infrared fluorescence imaging system, a data acquisition and processing unit, and an output device. The imaging system includes a light source emitting in the 750–800 nm range, a band-pass optical filter centered at 830 nm to isolate ICG fluorescence emission, and a high-sensitivity charge-coupled device (CCD) camera capable of capturing images at temporal intervals of one second or less. The apparatus is coupled to a computer system running specialized software that processes the acquired fluorescence data to extract perfusion metrics.

- describe photodetector

The photodetector is a cooled CCD camera with quantum efficiency exceeding 60% at 830 nm, enabling detection of low-intensity ICG fluorescence signals even in deep tissue layers. The detector is calibrated using reference standards to ensure linearity of response across the dynamic range of fluorescence intensities observed during ICG kinetics. It captures sequential images at high temporal resolution to resolve the time-to-peak of ICG fluorescence in both normal and ischemic tissues.

- describe numerical conversion means

The numerical conversion means transforms raw fluorescence intensity values into normalized fluorescence units by subtracting background noise, correcting for tissue autofluorescence, and scaling each pixel’s signal relative to the systemic reference region, typically the trunk or abdominal vasculature. This normalization eliminates variability due to differences in injection dose, lighting conditions, and tissue optical properties, ensuring that the resulting values reflect true ICG concentration dynamics.

- describe perfusion rate calculation means

The perfusion rate calculation means implements a derived mathematical equation that relates the time-to-peak (Tmax) of ICG fluorescence in each pixel to the perfusion rate (P) of the corresponding tissue region, using the known systemic ICG half-life (t1/2) as a calibration parameter. The equation, derived from Fick’s principle and first-order pharmacokinetic modeling, enables the computation of absolute perfusion rates in percent per minute, independent of systemic variability.

- describe output means

The output means generates visual representations of the calculated perfusion rates in the form of pseudocolor maps, histograms, and probability overlays. These outputs are displayed on a monitor or printed for clinical or research use. The system also exports data in standard formats for integration with electronic medical records or statistical analysis platforms.

- describe ICG fluorescence dynamics

ICG fluorescence dynamics following intravenous injection exhibit a characteristic bell-shaped curve in perfused tissues, characterized by rapid uptake, a distinct time-to-peak (Tmax), and exponential decay due to hepatic clearance. In ischemic tissues, Tmax is significantly delayed, and the decay phase is prolonged, reflecting reduced blood flow and impaired washout. The shape and timing of this curve are directly modulated by the perfusion rate of the tissue.

- describe Tmax calculation

Tmax is calculated for each pixel by identifying the time point at which the fluorescence intensity reaches its maximum value within the acquisition window. This is achieved through automated peak detection algorithms that analyze the first derivative of the fluorescence time series, with the inflection point corresponding to Tmax.

- describe simulation of ICG dynamics

Computational simulations of ICG dynamics were performed using a three-compartment model representing the trunk, normal tissue, and ischemic tissue. These simulations confirmed that perfusion rate is inversely correlated with Tmax and directly proportional to the slope of fluorescence decline following the peak, validating the mathematical model across a range of physiological conditions.

- describe relationship between Tmax and perfusion rate

A robust inverse relationship exists between Tmax and perfusion rate: as perfusion increases, Tmax decreases, and vice versa. This relationship is governed by the rate at which ICG is delivered to and cleared from the tissue, and it remains consistent across subjects when normalized to systemic clearance kinetics.

- describe perfusion map

The perfusion map is a spatially resolved, pixel-by-pixel representation of calculated perfusion rates, displayed as a color gradient ranging from blue (low perfusion) to red (high perfusion). Each pixel’s color corresponds to its absolute perfusion value, enabling visualization of heterogeneity within a tissue region.

- describe color representation method

Color representation is achieved using a standardized lookup table that maps perfusion rates from 0 to 1000%/min to a continuous spectrum of hues, with blue representing values below 50%/min, green representing 50–300%/min, yellow representing 300–600%/min, and red representing values above 600%/min. This mapping is calibrated to physiological benchmarks derived from control tissue measurements.

- describe tissue necrosis probability map

The tissue necrosis probability map is a derived output that assigns to each pixel a probability of subsequent tissue necrosis based on the inverse sigmoidal relationship between measured perfusion rate and observed necrosis at seven days post-ischemia. Regions with perfusion below 20%/min are assigned a probability exceeding 80%, while those above 150%/min are assigned a probability below 10%.

- describe measurement apparatus

The measurement apparatus includes a near-infrared LED array, a band-pass filter, a CCD camera, and a computer-controlled imaging platform. The apparatus is mounted on a stable frame to ensure consistent positioning during serial imaging sessions. It is compatible with standard animal restraint systems and human limb positioning devices.

- describe light source

The light source consists of a high-intensity light-emitting diode array emitting at 760 nm, optimized for excitation of ICG with minimal tissue absorption and scattering. The irradiance is calibrated to ensure uniform illumination across the field of view.

- describe filter

The filter is a narrow-band interference filter with a center wavelength of 830 nm and a full-width at half-maximum of 30 nm, designed to transmit ICG fluorescence while blocking excitation light and ambient noise.

- describe detector

The detector is a cooled, scientific-grade CCD camera with 1024×1024 pixel resolution and 16-bit depth, capable of capturing images at 1 Hz or faster with negligible dark current and high signal-to-noise ratio.

- describe analysis apparatus

The analysis apparatus is a dedicated computer system running proprietary software that performs background subtraction, normalization, Tmax detection, perfusion rate calculation, and map generation. The software includes user interfaces for region-of-interest selection, calibration, and result visualization.

- describe ICG injection

ICG is administered as a single intravenous bolus at a concentration of 400 µmol/L, with a volume of 0.1 mL per 20 g body weight in murine models, or a clinically equivalent dose in humans. Injection is performed via tail vein or peripheral venous access, followed immediately by initiation of image acquisition.

- describe light radiation

Near-infrared light is directed at the tissue surface at a 45-degree angle to minimize specular reflection and maximize fluorescence capture. The irradiance is maintained below 10 mW/cm² to avoid photothermal effects.

- describe fluorescence detection

Fluorescence emission is captured by the CCD camera through the 830 nm band-pass filter. Images are acquired at 1-second intervals for 12 minutes, ensuring complete capture of the ICG kinetics curve.

- describe data processing

Data processing involves sequential steps: background subtraction, normalization to the trunk region, pixel-wise time-series extraction, Tmax identification, and application of the perfusion equation. All computations are performed in real time with sub-second latency.

- describe perfusion rate calculation

Perfusion rate is calculated using the equation P = k / (Tmax - t0), where k is a calibration constant derived from systemic ICG half-life and t0 is the time of injection. The constant k is determined empirically from control tissue measurements and validated across multiple animal models.

- describe output of perfusion rates

Perfusion rates are output as numerical values for each pixel, aggregated into regional averages, and displayed as color-coded maps. Data may be exported as CSV, DICOM, or MATLAB files for further analysis.

- describe perfusion map construction

The perfusion map is constructed by assigning each pixel a color based on its calculated perfusion rate using the standardized color scale. The map is overlaid on a white-light anatomical reference image for spatial correlation.

- describe tissue necrosis probability prediction

Tissue necrosis probability is predicted by applying a sigmoidal function fitted to experimental data from 20 murine models, relating perfusion rate to observed necrosis at day 7. The function outputs a probability score between 0 and 1 for each pixel.

- describe measurement method

The measurement method involves anesthetizing the subject, administering ICG intravenously, acquiring time-series fluorescence images for 12 minutes, processing the data to compute perfusion rates, and generating perfusion and necrosis probability maps. The entire procedure requires less than 20 minutes and is non-invasive.

- describe ICG blood vessel image diagram

An ICG blood vessel image diagram is generated by thresholding the maximum intensity frame of the time series, revealing the vascular architecture. This diagram is used to define regions of interest for perfusion analysis.

- describe Tmax acquisition

Tmax is acquired by analyzing the fluorescence intensity time curve for each pixel, identifying the peak value, and recording the corresponding time point with millisecond precision.

- describe perfusion rate calculation

Perfusion rate is calculated using the derived mathematical relationship between Tmax and systemic ICG half-life, ensuring that the result is independent of injection dose and hepatic variability.

- describe output of perfusion rates

Perfusion rates are output as a numerical matrix and visualized as a pseudocolor map, with each pixel’s value corresponding to its perfusion rate in %/min.

- describe perfusion map construction

The perfusion map is constructed by mapping each pixel’s calculated perfusion rate to a color in a predefined gradient, creating a spatial representation of tissue perfusion heterogeneity.

- describe tissue necrosis probability prediction

Tissue necrosis probability is predicted by applying a validated logistic regression model to the perfusion rate data, generating a probability map that correlates with histological outcomes at day 7.

- describe data processing

Data processing includes noise reduction, normalization, Tmax detection, perfusion calculation, and statistical mapping, all performed automatically by proprietary software without user intervention.

- describe software implementation

The software is implemented in C++ and MATLAB, with a graphical user interface for image acquisition, calibration, analysis, and export. The code is modular, allowing for integration with third-party imaging systems.

- describe output device

The output device is a high-resolution color monitor or printer capable of displaying the perfusion and necrosis probability maps in vivid pseudocolor. Output may also be transmitted to hospital information systems via DICOM protocol.

### MODE FOR INVENTION

- introduce embodiments

The invention may be embodied in a portable clinical device for use in operating rooms or wound care clinics, or as a research-grade system for preclinical studies. In one embodiment, the apparatus is integrated into a surgical microscope for real-time intraoperative perfusion assessment. In another, it is mounted on a robotic arm for automated imaging of multiple animals in high-throughput drug screening. In a third embodiment, the system is adapted for human use with a handheld probe and tablet interface for bedside assessment of diabetic foot ulcers or post-revascularization perfusion.

## Comparative Example 1

### Prediction for Tissue Necrosis through Doppler Imaging

- describe experimental data

In a cohort of 20 mice subjected to hindlimb ischemia, laser Doppler imaging was performed immediately after surgery to assess blood flow. The mean perfusion index measured by LDI in the ischemic limb was 28.4 ± 5.2% relative to the contralateral limb. However, animals with identical LDI values exhibited vastly different outcomes: some developed complete limb necrosis, while others showed minimal tissue loss.

- describe limitations of Doppler imaging

The LDI data failed to correlate with histological necrosis at day 7 (R² = 0.18, p = 0.12), demonstrating that Doppler-based flow indices are insufficient to predict tissue viability. The method’s inability to resolve microvascular flow, its sensitivity to probe pressure, and its lack of quantitative calibration rendered it incapable of distinguishing between animals with salvageable and non-salvageable ischemia.

## Example 1

### Establishment of Method of Measuring Perfusion Using ICG

- derive equation for perfusion measurement

An equation was derived from first-order pharmacokinetic principles and Fick’s law, relating the time-to-peak (Tmax) of ICG fluorescence to the perfusion rate (P) of tissue: P = (ln(2) × Vd) / (Tmax × t1/2), where Vd is vascular density and t1/2 is the systemic ICG half-life. This equation was validated through computational simulations.

- describe ICG fluorescence experiment

Time-series ICG fluorescence images were acquired from 30 mice at 1-second intervals following intravenous injection. Normal limbs exhibited a Tmax of 18–22 seconds, while ischemic limbs showed Tmax values ranging from 45 to 180 seconds.

- formulate normal tissue ICG dynamics

In normal tissue, ICG fluorescence followed a rapid rise to peak followed by exponential decay, consistent with a single-compartment model dominated by hepatic clearance.

- simulate ischemic tissue ICG dynamics

Simulations of ischemic tissue dynamics demonstrated that reduced perfusion delayed Tmax and prolonged the decay phase, matching experimental observations.

- introduce perfusion rate concept

Perfusion rate was defined as the percentage of vascular volume replaced per minute, enabling direct comparison across tissues and subjects.

- define Tmax correlation coefficient

The correlation coefficient between Tmax and perfusion rate was determined to be -0.94 (p < 0.001), indicating a strong inverse relationship.

- derive equation for Tmax

The equation Tmax = k / P was derived, where k is a constant dependent on systemic ICG half-life and vascular density.

- summarize Tmax and perfusion rate relationship

The relationship between Tmax and perfusion rate is inversely proportional and mathematically deterministic, allowing for accurate, quantitative perfusion measurement when systemic ICG kinetics are known.

## Example 2

### Measurement of Perfusion Using Indocyanine Green and the Construction of Perfusion Map and Tissue Necrosis Probability Map Based on Correlation Coefficient

- describe blood perfusion reduction model

A murine hindlimb ischemia model was established by ligation and excision of the femoral artery and vein. Perfusion was measured immediately post-surgery using the disclosed apparatus.

- measure perfusion rates in ischemic tissue

Perfusion rates in ischemic tissue ranged from 4.1 to 19.4%/min, with higher values correlating with preserved tissue viability.

- construct perfusion map

A pixel-level perfusion map was generated, revealing spatial heterogeneity within the ischemic limb, with regions of near-zero perfusion adjacent to areas of moderate flow.

- analyze relationship between perfusion rates and tissue necrosis

A sigmoidal relationship was identified: perfusion rates below 20%/min predicted necrosis with 92% sensitivity and 89% specificity.

- construct tissue necrosis probability map

A color-coded necrosis probability map was constructed, with red regions indicating >80% probability of necrosis. These regions matched histologically confirmed necrotic areas at day 7 with 94% spatial accuracy.

## INDUSTRIAL APPLICABILITY

- describe industrial applications of perfusion measurement method

The method has broad industrial applicability in pharmaceutical research, surgical planning, and clinical diagnostics. In drug development, it enables objective, quantitative evaluation of pro-angiogenic therapies in preclinical models. In vascular surgery, it guides revascularization decisions by predicting tissue viability prior to intervention. In wound care, it stratifies diabetic foot ulcers by perfusion status to determine amputation risk. In oncology, it monitors tumor perfusion changes during anti-angiogenic therapy. The system is compatible with existing hospital imaging infrastructure and requires no specialized training, making it suitable for integration into clinical workflows worldwide.