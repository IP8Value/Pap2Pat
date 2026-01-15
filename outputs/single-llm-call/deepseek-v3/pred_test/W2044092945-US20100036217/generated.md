Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the field of tissue perfusion analysis, specifically to apparatuses and methods for quantitatively measuring perfusion rates in peripheral tissues using indocyanine green (ICG) fluorescence dynamics. The invention provides a novel approach to assessing vascular insufficiency by analyzing the time-dependent behavior of ICG fluorescence in target tissues, enabling precise determination of perfusion rates and prediction of tissue necrosis probability.  

## BACKGROUND  

Laser Doppler imaging has traditionally been used to assess tissue perfusion but suffers from limitations in quantitative accuracy and spatial resolution. While this method provides relative blood flow measurements, it cannot precisely quantify perfusion rates or distinguish between macro- and microvascular contributions. X-ray blood vessel imaging techniques, such as micro-CT angiography, offer structural visualization of vasculature but fail to provide functional perfusion data. ICG angiography has been employed for vascular imaging due to its near-infrared fluorescence properties, but conventional approaches rely on static intensity measurements, which are influenced by tissue optical properties rather than true perfusion dynamics.  

The ICG elimination test, commonly used to assess liver function, measures systemic ICG clearance but does not provide localized perfusion data. Existing ICG dynamics analysis methods lack quantitative rigor, as they fail to account for variations in systemic ICG pharmacokinetics and tissue-specific perfusion characteristics. Current liver function tests based on ICG clearance are insufficient for peripheral tissue assessment, as they do not correlate with regional blood flow.  

A critical shortcoming of existing perfusion measurement techniques is their inability to provide spatially resolved, quantitative perfusion rates that can predict tissue viability. There remains an unmet need for a non-invasive, high-resolution method capable of accurately measuring perfusion rates and generating predictive tissue necrosis probability maps.  

## DISCLOSURE  

### Technical Problem  

The primary technical problem addressed by this invention is the lack of a reliable method for quantifying tissue perfusion rates in vivo. Existing techniques either provide qualitative assessments or require invasive procedures, making them unsuitable for clinical applications requiring precise perfusion measurements. The inability to correlate perfusion rates with tissue viability further limits their diagnostic utility.  

### Technical Solution  

The invention provides a tissue perfusion analysis apparatus that measures perfusion rates by analyzing ICG fluorescence dynamics. The apparatus comprises a photodetector for capturing time-series fluorescence data, numerical conversion means for processing fluorescence signals, perfusion rate calculation means employing a derived mathematical model, and output means for displaying perfusion maps and necrosis probability predictions. The method utilizes the time-to-peak (Tmax) of ICG fluorescence and its relationship with perfusion rate to generate quantitative perfusion maps.  

## BEST MODE  

The analysis apparatus integrates a light source emitting at 760 nm to excite ICG fluorescence and a detector equipped with an 830 nm band-pass filter to capture emitted fluorescence. The photodetector, preferably a CCD camera, acquires time-series images at intervals as short as 1 second to resolve rapid ICG kinetics in normal tissues. Numerical conversion means process the raw fluorescence data into intensity-time curves for each pixel within the region of interest.  

The perfusion rate calculation means implement a mathematical model relating Tmax to perfusion rate (P) through the equation P = (ln2)/Tmax, where Tmax is normalized by the systemic ICG half-life measured in a reference vascular region. The apparatus simulates ICG dynamics for different perfusion rates to validate the model and establish the inverse relationship between Tmax and perfusion rate.  

Output means generate color-coded perfusion maps where perfusion rates are represented by a continuous color scale, enabling visual assessment of perfusion heterogeneity. The apparatus further constructs tissue necrosis probability maps by applying a sigmoidal function that relates measured perfusion rates to necrosis likelihood based on experimental correlation data.  

The measurement apparatus operates by first administering an intravenous ICG bolus, then radiating the target tissue with excitation light while continuously detecting fluorescence emission. Data processing involves determining ICG fluorescence time courses for each pixel, calculating Tmax values, and converting these to perfusion rates using the derived equation. The system outputs both numerical perfusion rates and graphical perfusion maps, along with predictive necrosis probability maps.  

The measurement method involves acquiring an ICG blood vessel image sequence, determining Tmax for each pixel from the fluorescence time course, calculating perfusion rates using the model equation, and constructing perfusion maps. The tissue necrosis probability prediction is generated by applying the perfusion rate-necrosis probability correlation function to the calculated perfusion rates.  

Software implementation handles image acquisition, data processing, and visualization, while output devices display results in formats suitable for clinical interpretation. The complete system provides a comprehensive solution for quantitative perfusion assessment with predictive capability for tissue viability.  

### MODE FOR INVENTION  

Various embodiments of the invention include adaptations for different clinical applications. One embodiment integrates the apparatus with surgical microscopes for intraoperative perfusion monitoring. Another embodiment employs miniaturized detectors for endoscopic applications in deep tissue assessment. A portable version incorporates a handheld probe for point-of-care vascular assessments.  

## Comparative Example 1  

### Prediction for Tissue Necrosis through Doppler Imaging  

Experimental data comparing laser Doppler imaging (LDI) with the inventive method demonstrated LDI's limitations in necrosis prediction. In murine hindlimb ischemia models, LDI could only qualitatively distinguish ischemic from normal limbs but failed to provide the quantitative perfusion rates necessary for accurate necrosis prediction. The perfusion rate threshold for necrosis prediction established by the inventive method (20-60%/min) could not be resolved by LDI measurements, which showed poor correlation with actual necrosis outcomes.  

## Example 1  

### Establishment of Method of Measuring Perfusion Using ICG  

The equation for perfusion measurement was derived from Fick's law applied to ICG pharmacokinetics. Experimental ICG fluorescence data from normal murine hindlimbs showed rapid time-to-peak (~20 s) and exponential decay, while ischemic limbs exhibited delayed Tmax and clearance. Mathematical modeling formulated normal tissue ICG dynamics as FI(t) = Vd·e^(-t/τ), where τ is the systemic time constant. Ischemic tissue dynamics were described by dFIisc/dt = P(FIv - FIisc), yielding the perfusion rate equation P = (ln2)/Tmax.  

Simulations of ischemic tissue ICG dynamics confirmed the model's validity across perfusion rates from 5-500%/min. The correlation coefficient between Tmax and perfusion rate was established as R² = 0.98 in validation studies. The derived equation enables perfusion rate calculation from measured Tmax values normalized by systemic ICG half-life.  

## Example 2  

### Measurement of Perfusion Using Indocyanine Green and the Construction of Perfusion Map and Tissue Necrosis Probability Map Based on Correlation Coefficient  

A blood perfusion reduction model was created by femoral artery ligation in mice. Perfusion rates in ischemic tissue measured 56±6%/min compared to 475±19%/min in normal limbs. Perfusion maps revealed spatial heterogeneity, with some regions maintaining higher perfusion from collateral circulation.  

Analysis of the perfusion rate-necrosis relationship showed an inverse sigmoidal correlation: regions with <20%/min perfusion had >80% necrosis probability, while >60%/min perfusion resulted in <10% necrosis probability. Tissue necrosis probability maps accurately predicted actual necrosis patterns observed 7 days post-ischemia, demonstrating the method's predictive capability.  

## INDUSTRIAL APPLICABILITY  

The invention has broad industrial applications in medical diagnostics and therapeutic monitoring. Clinical applications include assessment of peripheral vascular disease in diabetic patients, prediction of tissue viability in critical limb ischemia, and intraoperative monitoring of tissue perfusion during vascular surgeries. Pharmaceutical applications include quantitative evaluation of angiogenic drug efficacy in preclinical studies. The method's non-invasive nature and quantitative output make it suitable for integration into various medical imaging platforms and point-of-care diagnostic devices.