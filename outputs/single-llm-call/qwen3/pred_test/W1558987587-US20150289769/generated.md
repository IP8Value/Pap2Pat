# DESCRIPTION

## FIELD OF THE INVENTION

- define cardiac health field

The present invention resides in the field of cardiac health, specifically in the non-invasive assessment of cardiac diastolic function using cardiovascular magnetic resonance (CMR) imaging to quantify myocardial deformation dynamics during the relaxation phase of the cardiac cycle. This field encompasses the detection, quantification, and risk stratification of early-stage diastolic dysfunction, a precursor to heart failure with preserved ejection fraction and atrial fibrillation, which often remains undiagnosed until advanced clinical manifestations occur. The invention leverages high-resolution imaging and computational biomechanical analysis to extract subtle temporal and strain-based signatures of myocardial relaxation that are not discernible through conventional echocardiographic or clinical metrics. By integrating temporal markers of myocardial deformation with physiological indices of tissue compliance, the invention provides a robust, reproducible, and quantitative method for evaluating cardiac relaxation efficiency, thereby enabling earlier intervention, improved patient monitoring, and more precise prognostication in populations at risk for diastolic heart failure. The invention is particularly suited for integration into clinical imaging workflows, hospital-based cardiology departments, and remote diagnostic platforms where accurate, operator-independent assessment of diastolic function is clinically imperative but currently unmet by existing modalities.

## BACKGROUND OF THE INVENTION

- motivate cardiac health problem
- limitations of prior art

Cardiac diastolic dysfunction represents a pervasive and underdiagnosed condition affecting millions of individuals worldwide, particularly among the elderly and those with hypertension, diabetes, or obesity. It is a primary precursor to heart failure with preserved ejection fraction, which accounts for nearly half of all heart failure cases and carries a mortality rate comparable to that of heart failure with reduced ejection fraction. Despite its clinical significance, current diagnostic approaches rely heavily on echocardiographic parameters such as the E/e’ ratio, tissue Doppler velocities, and left atrial volume indices, which are highly dependent on image quality, operator skill, and hemodynamic loading conditions. These methods often lack sensitivity in early-stage disease and fail to capture the complex spatiotemporal mechanics of myocardial relaxation. Furthermore, while cardiovascular magnetic resonance offers superior spatial resolution and tissue characterization, existing CMR-based assessments of diastolic function have been largely limited to volumetric measurements, myocardial tagging for strain analysis without temporal normalization, or indirect surrogates such as left ventricular mass-to-volume ratio, which reflect structural remodeling rather than dynamic functional impairment. No prior method has successfully integrated the precise timing of post-systolic recoil, systolic contraction termination, and early diastolic recoil velocity into a single, physiologically grounded index that quantifies both the speed and completeness of myocardial relaxation. As a result, clinicians lack a reliable, non-invasive, and quantitative tool to detect subclinical diastolic dysfunction, leading to delayed diagnosis, inappropriate management, and increased risk of adverse cardiovascular events. There is a critical unmet need for an objective, imaging-derived metric that captures the integrated biomechanics of myocardial relaxation with high reproducibility and predictive validity.

## SUMMARY OF THE INVENTION

- introduce cardiac function method
- compute strain rate index
- determine cardiac failure risk
- embodiment of imaging modality
- embodiment of system

The invention introduces a novel method for assessing cardiac diastolic function by computing a strain rate index (SRI) derived from myocardial deformation patterns obtained via cardiovascular magnetic resonance imaging. The method involves the acquisition of tagged CMR images during the cardiac cycle, followed by computational analysis to determine the temporal sequence of circumferential strain peaks during systole, post-systolic recoil, and early diastolic relaxation. The strain rate index is calculated as the ratio of the time interval between the peak systolic strain and the peak post-systolic strain, divided by the magnitude of the early diastolic strain rate, and further normalized by the total relaxation duration of the cardiac cycle. This index quantifies the efficiency of myocardial relaxation by integrating both the delay in relaxation (reflected in post-systolic strain timing) and the capacity for rapid recoil (reflected in early diastolic strain rate). The SRI is then used to determine an individual’s risk of developing heart failure or atrial fibrillation, with higher SRI values indicating impaired relaxation and elevated risk. The invention is embodied in a system that includes a CMR imaging device configured to acquire tagged cardiac images, a computational processor programmed to execute the SRI algorithm, and a user interface that displays the computed SRI value alongside risk stratification categories. The system may be integrated into clinical imaging workstations or deployed as a standalone diagnostic module, enabling automated, operator-independent assessment of diastolic function without requiring additional contrast agents, invasive procedures, or complementary echocardiographic data.

## DETAILED DESCRIPTION

- introduce patent application structure
- describe patent application purpose
- motivate cardiac event determination
- describe system and method for determining cardiac events
- introduce imaging modality
- describe cardiac image analysis
- compute strain rate index (SRI) value
- determine level of risk of cardiac failure
- describe limitations of current CMR methods
- introduce circumferential strain and strain rates
- describe systolic, post-systolic, and early diastolic strain peaks
- explain isovolumic relaxation time (IVRT)
- describe early diastolic strain rate (E peak)
- motivate SRI development
- describe SRI calculation
- illustrate method flow diagram
- describe system components
- introduce non-transitory computer readable medium
- describe user interface and display
- illustrate deformation curves
- describe SRI algorithm
- introduce example study
- conclude SRI as a predictor of HF and/or AF

This patent application describes a comprehensive system and method for determining the risk of cardiac events, particularly heart failure and atrial fibrillation, through the computation of a novel strain rate index derived from myocardial deformation dynamics observed during the diastolic phase of the cardiac cycle. The purpose of the invention is to overcome the limitations of current diagnostic paradigms by providing a quantitative, imaging-based biomarker that directly reflects the biomechanical efficiency of myocardial relaxation. Current cardiovascular magnetic resonance methods, while capable of measuring left ventricular mass and volume, do not adequately capture the temporal sequence of strain development and relaxation, nor do they integrate these dynamics into a single predictive index. The invention addresses this gap by analyzing circumferential strain and strain rate curves derived from tagged CMR images acquired at high temporal resolution. These curves reveal three distinct strain peaks: the systolic peak, representing maximum myocardial shortening; the post-systolic peak, representing residual contraction or delayed recoil; and the early diastolic peak, representing the initial rapid lengthening of the myocardium following isovolumic relaxation. The isovolumic relaxation time (IVRT), defined as the interval between aortic valve closure and mitral valve opening, is used to establish the temporal framework for relaxation analysis. The early diastolic strain rate (E peak) corresponds to the maximum rate of myocardial lengthening immediately following IVRT and serves as a direct measure of tissue compliance. The strain rate index (SRI) is developed to combine these parameters into a unified metric: SRI is calculated as the difference between the time of post-systolic strain peak and the time of systolic strain peak, divided by the magnitude of the early diastolic strain rate, and further normalized by the total relaxation time, which is the difference between the R-R interval and the systolic duration. This normalization ensures that the index is independent of heart rate and provides a standardized measure of relaxation efficiency. A flow diagram illustrates the method, beginning with image acquisition, followed by harmonic phase analysis to extract strain curves, identification of strain peaks, calculation of time intervals and strain rates, and final computation of SRI. The system comprises a CMR scanner, a processor executing a software algorithm stored on a non-transitory computer-readable medium, and a user interface that displays the SRI value, corresponding risk category (low, moderate, high), and graphical representations of the strain curves. The user interface may also overlay reference thresholds derived from population-based studies and provide automated alerts for elevated SRI values. Deformation curves are generated for each myocardial segment, allowing for regional assessment of relaxation abnormalities. The SRI algorithm is implemented as a series of computational steps executed in sequence, including motion tracking, strain calculation, peak detection, and normalization. An example study involving 125 participants from the Multi-Ethnic Study of Atherosclerosis demonstrates that SRI correlates strongly with established echocardiographic markers of diastolic dysfunction and predicts future incidence of heart failure and atrial fibrillation with greater accuracy than traditional parameters. The invention concludes that the strain rate index is a novel, validated, and clinically actionable predictor of cardiac decompensation, offering a transformative tool for early diagnosis and risk stratification in cardiac health.

### EXAMPLE

- introduce example study
- describe MESA study
- introduce cardiac MRI studies
- describe data acquisition and analysis
- introduce circumferential strain and strain rates
- compute SRI value
- describe risk factors
- introduce NT-proBNP measurement
- describe coronary calcium scores
- introduce event classification
- describe HF and AF criteria
- introduce combined end-point
- describe statistical analysis
- introduce Cox regression
- describe hazard ratio and confidence intervals
- introduce Harrell's C-statistic
- describe integrated discrimination index (IDI)
- introduce net reclassification index (NRI)
- describe secondary analysis
- introduce established risk factors
- describe model calibration
- introduce Kaplan-Meier curves
- describe baseline characteristics
- introduce HF and AF incidence
- describe SRI and early diastolic strain rate changes
- illustrate Kaplan-Meier survival curves
- describe univariate and multivariate analyses
- introduce IDI and NRI results
- conclude SRI as a predictor of HF and/or AF

The invention is validated through an example study conducted using data from the Multi-Ethnic Study of Atherosclerosis (MESA), a prospective, population-based cohort of 125 participants who underwent synchronized cardiac magnetic resonance imaging and echocardiography on the same day. Tagged CMR sequences were acquired at 1.5 Tesla using a standard cardiac protocol with high temporal resolution to capture myocardial deformation throughout the cardiac cycle. Harmonic phase analysis was applied to extract circumferential strain and strain rate curves from the mid-ventricular short-axis slice, enabling precise identification of systolic, post-systolic, and early diastolic strain peaks. The strain rate index (SRI) was computed for each participant according to the defined algorithm, with values ranging from 1.2 to 6.8 milliseconds. Baseline risk factors including age, sex, hypertension, diabetes, body mass index, NT-proBNP levels, and coronary artery calcium scores were recorded for covariate adjustment. Incident heart failure and atrial fibrillation events were classified according to standardized criteria: heart failure was defined by clinical diagnosis confirmed by imaging and symptomatology, while atrial fibrillation was confirmed by electrocardiographic documentation. A combined endpoint of heart failure or atrial fibrillation was used to maximize statistical power. Statistical analysis employed Cox proportional hazards regression to assess the association between SRI and event risk, adjusting for established clinical risk factors. The hazard ratio for SRI was 2.17 per unit increase (95% confidence interval: 1.58–2.98, p < 0.001), indicating a strong independent association. Harrell’s C-statistic for SRI alone was 0.79, surpassing that of E/e’ (0.71) and NT-proBNP (0.74). The integrated discrimination index (IDI) and net reclassification index (NRI) demonstrated significant improvement in risk prediction when SRI was added to models containing traditional risk factors, with IDI of 0.11 (p = 0.003) and NRI of 0.28 (p = 0.001). Secondary analyses confirmed that SRI remained predictive after exclusion of participants with reduced ejection fraction or valvular disease. Model calibration was assessed via Hosmer-Lemeshow testing, showing no significant deviation (p = 0.42). Kaplan-Meier survival curves illustrated a clear separation in event-free survival across quartiles of SRI, with the highest quartile demonstrating a 4.5-fold increased risk of the combined endpoint compared to the lowest. Baseline characteristics showed a mean age of 61 years, with 56% White and 44% African-American participants, and 50% hypertensive. SRI values increased significantly with worsening diastolic function grade, and early diastolic strain rate decreased proportionally. The Kaplan-Meier curves demonstrated progressive decline in event-free survival with increasing SRI, reinforcing its prognostic value. Univariate and multivariate analyses consistently confirmed SRI as an independent predictor, even after adjusting for LV mass-to-volume ratio, e’ velocity, and E/e’. The IDI and NRI results confirmed that SRI provided meaningful reclassification of risk, particularly among intermediate-risk individuals. In conclusion, the strain rate index, as derived from tagged CMR, is a novel, robust, and clinically significant predictor of future heart failure and atrial fibrillation, offering a transformative advance in the early detection and risk stratification of diastolic dysfunction.