# DESCRIPTION

## FIELD

- define field of technology

The present invention relates to the field of medical diagnostics and neurocritical care, specifically to noninvasive methods and systems for estimating intracranial pressure in real time using physiological signals acquired from the patient’s radial arterial blood pressure and cerebral blood flow velocity. The invention provides a computational framework grounded in first-principles physiological modeling and Bayesian statistical inference to derive accurate, continuous, and patient-specific estimates of intracranial pressure without the need for invasive intracranial sensors or prior calibration against invasive measurements. This technology is particularly applicable in intensive care units, operating rooms, and emergency departments where continuous monitoring of intracranial dynamics is clinically essential but limited by the risks and logistical constraints of invasive procedures.

## BACKGROUND

- introduce intracranial pressure

Intracranial pressure is the hydrostatic pressure exerted by the contents of the skull—namely, brain tissue, cerebrospinal fluid, and intracranial blood—on the walls of the cranial cavity. It is a critical physiological parameter that reflects the balance between cerebral perfusion, vascular compliance, and fluid dynamics within the rigid confines of the skull. Deviations from normal intracranial pressure, whether elevated or diminished, are associated with a spectrum of neurological pathologies including traumatic brain injury, hydrocephalus, intracranial hemorrhage, and cerebral edema. Accurate and continuous monitoring of this pressure is essential for guiding therapeutic interventions, preventing secondary neurological injury, and optimizing cerebral perfusion pressure.

- describe risks of elevated ICP

Elevated intracranial pressure compromises cerebral blood flow by reducing the pressure gradient necessary for perfusion, leading to ischemia, herniation, and irreversible neuronal damage. When intracranial pressure exceeds 20 to 25 mmHg, it is associated with significantly increased mortality and poor neurological outcomes. Sustained elevation can trigger autoregulatory failure, disrupt the blood-brain barrier, and induce global cerebral hypoperfusion. In pediatric and adult populations alike, failure to detect and respond to rising intracranial pressure in a timely manner remains a leading cause of preventable neurological morbidity and death.

- discuss current invasive ICP measurement techniques

Current clinical practice relies on invasive methods for direct measurement of intracranial pressure, including external ventricular drains, intraparenchymal microtransducers, and subdural or epidural pressure sensors. These devices provide high-fidelity, real-time pressure readings and are considered the gold standard for accuracy. External ventricular drains allow for both measurement and therapeutic drainage of cerebrospinal fluid, while parenchymal sensors offer continuous monitoring without the need for ventricular access. However, all invasive techniques require surgical insertion into the cranial cavity, carrying inherent risks such as infection, hemorrhage, catheter malfunction, and tissue trauma.

- discuss limitations of current techniques

The invasiveness of these methods restricts their use to highly monitored settings such as neurointensive care units, where the risk-benefit ratio justifies the procedure. They are rarely employed in emergency, pre-hospital, or perioperative contexts due to the procedural complexity, time delay, and potential complications. Furthermore, repeated insertion is often impractical, limiting longitudinal monitoring. Even in controlled environments, sensor drift, calibration errors, and mechanical obstructions can compromise data integrity. These limitations create a critical gap in clinical care: the inability to continuously monitor intracranial pressure in large patient populations who would benefit from early detection of pressure changes.

- motivate need for noninvasive ICP estimation

There is a compelling and unmet clinical need for a reliable, noninvasive method to estimate intracranial pressure that is accurate, continuous, bedside-compatible, and free from the risks associated with surgical intervention. Such a method would enable early identification of intracranial hypertension in a broader range of patients, including those in emergency departments, during neurosurgical procedures, in neonatal intensive care units, and in resource-limited settings. A noninvasive approach that does not require calibration against invasive measurements would further enhance scalability and clinical adoption, transforming the standard of care for neurological monitoring.

## SUMMARY

- introduce system embodiment

The invention comprises a system for noninvasive intracranial pressure estimation that integrates physiological modeling with statistical inference to compute continuous estimates of intracranial pressure from noninvasively acquired radial arterial blood pressure and transcranial Doppler cerebral blood flow velocity signals. The system includes hardware components for signal acquisition, a processing unit configured to execute a model-based estimation algorithm, and an output interface for displaying real-time intracranial pressure estimates and associated confidence metrics.

- describe data acquisition

The system acquires time-synchronized waveforms of radial arterial blood pressure and cerebral blood flow velocity from standard clinical monitoring devices. These signals are sampled at a rate sufficient to resolve cardiac-cycle dynamics, typically at or above 125 Hz, and are conditioned to remove motion artifacts, baseline drift, and out-of-band noise while preserving the morphological features critical to hemodynamic modeling.

- estimate initial ICP value

The system computes an initial estimate of intracranial pressure by applying a statistical model that evaluates a range of candidate intracranial pressure values and time offsets between the blood pressure and blood flow velocity waveforms. For each candidate pair, the system derives model parameters that best predict the observed cerebral blood flow velocity waveform from the corresponding cerebral perfusion pressure waveform, which is constructed by subtracting the candidate intracranial pressure from the radial arterial blood pressure. The resulting prediction errors are transformed into a likelihood distribution over the space of possible intracranial pressure values.

- obtain updated data

Subsequent data segments are acquired in real time, with each segment corresponding to a fixed number of cardiac cycles, typically twenty beats. The system continuously receives new input data and updates its estimation of intracranial pressure without requiring re-initialization or recalibration.

- estimate updated ICP value

The system estimates an updated intracranial pressure value by combining the likelihood distribution derived from the new data segment with a dynamic model of intracranial pressure evolution. This model incorporates a first-order autoregressive process that predicts the change in intracranial pressure based on prior estimates, enabling the system to track transient changes while filtering out noise and measurement inconsistencies.

- output updated ICP value

The system outputs a continuous stream of updated intracranial pressure estimates, each accompanied by a confidence interval derived from the posterior probability distribution. These estimates are displayed in real time on a user interface, enabling clinicians to monitor trends and detect deviations from baseline with high temporal resolution.

- introduce software embodiment

The invention further encompasses a software embodiment comprising a set of processor-executable instructions stored on a non-transitory computer-readable medium. These instructions, when executed by a computing device, implement the full estimation pipeline, including signal preprocessing, model parameter computation, likelihood and posterior distribution generation, and dynamic tracking of intracranial pressure changes.

- describe data acquisition

The software embodiment receives digital input streams from external monitoring devices via a network interface, performs time alignment and resampling to ensure synchronization, and applies baseline adjustments to account for hydrostatic pressure differences between measurement sites.

- estimate initial ICP value

The software computes an initial intracranial pressure estimate by evaluating a multidimensional likelihood function over a predefined range of candidate intracranial pressure values and time offsets, using a finite impulse response model of cerebral hemodynamics and minimizing the residual error between predicted and observed cerebral blood flow velocity waveforms.

- obtain updated data

The software continuously ingests new data segments as they become available, maintaining a sliding window of recent measurements to support real-time estimation without interruption.

- estimate updated ICP value

The software updates the intracranial pressure estimate by fusing the likelihood-derived change in pressure with a model-predicted change derived from an autoregressive process, applying a Kalman-filter-like weighting scheme that prioritizes the more reliable source based on estimated variance.

- output updated ICP value

The software outputs the final intracranial pressure estimate and its associated uncertainty to a display, data logger, or clinical information system, enabling integration into electronic health records and alerting protocols.

## DETAILED DESCRIPTION

- introduce limitations of conventional ICP estimation techniques

Conventional noninvasive techniques for estimating intracranial pressure rely on empirical correlations between surrogate signals such as optic nerve sheath diameter, transcranial Doppler indices, or cranial compliance metrics. These methods lack physiological grounding, exhibit poor generalizability across patient populations, and are highly sensitive to operator technique and anatomical variability. Their accuracy is often insufficient for clinical decision-making, particularly in dynamic or heterogeneous patient populations.

- describe inaccuracy of ICP estimates due to lack of physiological data

Many existing approaches fail to account for the underlying biomechanical relationships between arterial pressure, cerebral blood flow, and intracranial compliance. Without incorporating the physical laws governing cerebral autoregulation and vascular resistance, these methods produce estimates that are statistically correlated but physiologically implausible, leading to false positives and negatives in critical care scenarios.

- explain limitations of mapping ABP and CBFV to ICP measurements

Direct mapping of radial arterial blood pressure and cerebral blood flow velocity to intracranial pressure without modeling the intervening hemodynamic transformation ignores the nonlinear, time-varying nature of cerebral vascular resistance and compliance. Such mappings assume fixed relationships that do not hold across individuals or over time, resulting in systematic bias and poor reproducibility.

- describe inaccuracies due to misalignment of ABP and CBFV waveforms

A critical source of error in previous methods is the assumption of fixed temporal alignment between arterial blood pressure and cerebral blood flow velocity waveforms. In reality, the propagation delay between these signals varies across patients due to differences in vascular anatomy, cerebral autoregulatory state, and measurement device latency. Failure to account for this time offset introduces significant distortion in the modeled relationship between cerebral perfusion pressure and flow velocity.

- motivate development of new computational techniques for ICP estimation

To overcome these limitations, a new computational framework is required that explicitly models the physiological transformation from arterial pressure to cerebral flow velocity, accounts for unknown time delays, and incorporates probabilistic reasoning to quantify uncertainty. Such a framework must be computationally efficient, interpretable, and robust to noise and variability in clinical data.

- introduce statistical model for ICP estimation

The invention introduces a statistical model that treats intracranial pressure as an unknown parameter to be inferred from observed waveforms using a likelihood-based approach. The model is derived from a first-order finite impulse response representation of cerebral hemodynamics, where cerebral perfusion pressure drives cerebral blood flow velocity through a time-delayed, linear transformation governed by two parameters: cerebrovascular resistance and compliance.

- describe use of statistical model to compute initial ICP value

The initial intracranial pressure estimate is computed by evaluating the likelihood of each candidate pressure value and time offset pair, based on the residual error between the model-predicted and observed cerebral blood flow velocity waveforms. The likelihood distribution is constructed by normalizing the inverse exponential of the prediction error norm, ensuring that lower error corresponds to higher likelihood.

- explain estimation of changes in ICP using additional ABP and CBFV data

Subsequent estimates of intracranial pressure are derived by analyzing changes in the waveform relationship over successive data windows. The difference between consecutive intracranial pressure estimates is modeled as a first-order autoregressive process, allowing the system to distinguish true physiological changes from measurement noise.

- describe dynamic updating of ICP value using patient data

The system dynamically updates the intracranial pressure estimate by combining the likelihood-derived change with the model-predicted change using a Bayesian fusion rule that assigns greater weight to the estimate with lower variance. This process is repeated for each new data segment, enabling continuous, real-time tracking of intracranial pressure without re-initialization.

- introduce importance of accounting for biases in ICP estimates

Systematic biases in intracranial pressure estimation can arise from sensor misalignment, hydrostatic pressure differences, or deviations in waveform morphology between radial and central arterial pressure. These biases, if unaddressed, can lead to clinically misleading estimates that compromise patient safety.

- describe use of patient's own data to compensate for biases

The system compensates for these biases by leveraging the patient’s own historical data to estimate the most probable time offset between arterial and flow velocity waveforms. By scanning a physiologically plausible range of offsets and selecting the one that maximizes the likelihood of the model fit, the system inherently corrects for device-specific and anatomical delays.

- explain inaccuracy of ICP estimates due to misalignment of ABP and CBFV waveforms

The misalignment between arterial blood pressure and cerebral blood flow velocity waveforms is not a static artifact but a dynamic physiological feature that varies with cerebral autoregulatory state, vascular tone, and intracranial compliance. Fixed alignment assumptions therefore introduce systematic error that cannot be resolved by conventional signal processing techniques.

- introduce time offsets as parameters of the statistical model

Time offsets are treated as free parameters within the statistical model, scanned over a range determined by physiological constraints such as the expected delay between systolic upstrokes and the requirement that diastolic points remain aligned within a tolerance of three sampling intervals.

- describe determination of time offset values to align ABP and CBFV waveforms

The optimal time offset is determined by maximizing the likelihood distribution over the joint space of intracranial pressure and time delay. This is achieved by evaluating the prediction error for each combination of candidate values and selecting the pair that yields the highest likelihood.

- summarize issues with estimating ICP

Estimating intracranial pressure noninvasively is challenged by the indirect nature of the measurements, the nonlinear and time-varying physiology of cerebral autoregulation, the presence of measurement noise, and the lack of a universal calibration standard. Previous methods have failed to address these challenges in a unified, physically grounded, and clinically robust manner.

- introduce embodiments of technology for estimating ICP

The invention provides multiple embodiments for estimating intracranial pressure, including a hardware system for real-time clinical deployment, a software module for integration into existing monitoring platforms, and a method for estimating intracranial pressure pulse amplitude as a secondary output derived from the same likelihood framework.

- describe obtaining data identifying ABP and CBFV of a patient

The system obtains digital time-series data representing radial arterial blood pressure and transcranial Doppler cerebral blood flow velocity from standard clinical monitoring devices, ensuring that both signals are sampled synchronously and with sufficient temporal resolution to resolve cardiac-cycle dynamics.

- explain estimation of initial ICP value using statistical model

The initial intracranial pressure value is estimated by applying a statistical model that computes a likelihood distribution over a range of candidate intracranial pressure values and time offsets, using a finite impulse response model of cerebral hemodynamics to predict the cerebral blood flow velocity waveform from the arterial pressure waveform minus each candidate intracranial pressure value.

- describe estimation of updated ICP value using changes in ICP

The updated intracranial pressure value is computed by combining the change in intracranial pressure inferred from the latest data segment with a model-predicted change derived from a first-order autoregressive process, using a Bayesian fusion rule that weights each source by its estimated variance.

- introduce use of Bayesian statistics in estimating ICP

Bayesian statistics are employed to combine prior knowledge about plausible intracranial pressure values with the likelihood derived from the observed data, yielding a posterior distribution that represents the most probable intracranial pressure value given both the physiological model and the clinical context.

- describe computation of posterior distribution of ICP values

The posterior distribution is computed by multiplying the likelihood distribution with a prior distribution that reflects the expected range of intracranial pressure values in the patient population, then normalizing the result to ensure it integrates to unity.

- explain use of prior distribution of ICP values

The prior distribution is designed to be broad and inclusive, assigning non-negligible probability to both low and high intracranial pressure values to ensure generalizability across diverse patient populations, including those with extreme pathologies.

- describe evaluation of time periods with low-quality data

The system evaluates the quality of each data segment by assessing the signal-to-noise ratio, waveform morphology consistency, and the stability of the likelihood distribution. Segments with low confidence are flagged and excluded from the estimation process.

- introduce removal of low-quality time periods in estimating ICP

Low-quality time periods are excluded from the computation of both initial and updated intracranial pressure estimates to prevent the propagation of erroneous values and to maintain the integrity of the posterior distribution.

- describe determination of changes in ICP using statistical model

Changes in intracranial pressure are determined by comparing the posterior estimates from consecutive data windows and modeling the difference as a stochastic process governed by an autoregressive equation with a white noise component.

- explain use of optimization techniques for estimating parameter values

Optimization techniques are applied to estimate the model parameters—cerebrovascular resistance and compliance—by minimizing the least-squares error between the predicted and observed cerebral blood flow velocity waveforms for each candidate intracranial pressure and time offset pair.

- describe computation of ICP estimates during different time periods

The system computes intracranial pressure estimates for each non-overlapping data window, typically consisting of twenty cardiac cycles, and aggregates these estimates into a continuous time series that reflects the temporal evolution of intracranial pressure.

- introduce prediction of ICP value using ABP and CBFV data

The system predicts future intracranial pressure values by extrapolating the autoregressive model of intracranial pressure change, using the most recent filtered estimate and its associated variance to generate a probabilistic forecast.

- describe estimation of change in ICP for a future time

The change in intracranial pressure for a future time point is estimated by applying the autoregressive model to the current filtered change estimate, incorporating the model’s time constant to simulate the expected rate of pressure evolution.

- explain use of predicted change in ICP in estimating updated ICP value

The predicted change in intracranial pressure is combined with the observed change derived from the likelihood function to produce a fused estimate that is more robust than either source alone, with the fusion weights determined by the relative confidence in each estimate.

- describe evaluation of possible values for parameters of statistical model

The system evaluates a discrete grid of possible values for the model parameters, including intracranial pressure and time offset, spanning physiologically plausible ranges determined by clinical evidence and biomechanical constraints.

- introduce prediction of change in ICP for a time period

The system predicts the change in intracranial pressure over a defined time period by simulating the autoregressive process forward in time, using the most recent filtered estimate as the initial condition.

- describe comparison of predicted and data-derived changes in ICP

The system compares the predicted change in intracranial pressure with the change inferred from the latest data segment, computing the residual error and using it to update the confidence in the autoregressive model.

- explain evaluation of time offsets between ABP and CBFV waveforms

The system evaluates multiple candidate time offsets between the arterial blood pressure and cerebral blood flow velocity waveforms, selecting the offset that maximizes the likelihood of the model fit while satisfying physiological constraints on waveform alignment.

- describe alignment of ABP and CBFV waveforms using physiological constraints

Alignment is achieved by enforcing two physiological constraints: the systolic upstroke of the cerebral blood flow velocity waveform must precede that of the arterial blood pressure waveform, and the diastolic endpoints must remain within three sampling intervals of each other to reflect vascular compliance dynamics.

- introduce prediction of physiological signals using statistical model

The statistical model is used to predict the cerebral blood flow velocity waveform from the arterial blood pressure waveform and a candidate intracranial pressure value, enabling the system to evaluate how well each candidate explains the observed data.

- describe computation of prediction errors

Prediction errors are computed as the Euclidean norm of the difference between the predicted and observed cerebral blood flow velocity waveforms over the duration of each data window, serving as the basis for the likelihood function.

- explain use of prediction errors in computing likelihood of ICP

The prediction errors are transformed into a likelihood score using an exponential decay function, such that small errors correspond to high likelihood and large errors correspond to low likelihood, thereby encoding the probability that a given intracranial pressure value generated the observed data.

- describe computation of likelihood distribution of ICP for different time offsets

For each candidate time offset, the system computes a one-dimensional likelihood distribution over the range of possible intracranial pressure values by marginalizing the joint likelihood over the time offset dimension.

- summarize various aspects and embodiments of technology described

The invention encompasses a comprehensive framework for noninvasive intracranial pressure estimation that integrates physiological modeling, statistical inference, dynamic tracking, and real-time signal processing into a single, clinically deployable system capable of continuous, calibration-free monitoring.

- introduce ICP estimation technique

The intracranial pressure estimation technique is based on a first-principles model of cerebral hemodynamics that relates arterial pressure and cerebral flow velocity through a time-delayed, linear transformation governed by cerebrovascular resistance and compliance, with intracranial pressure as the unknown variable to be inferred.

- describe ICP baseline value estimation

The baseline intracranial pressure value is established by computing posterior estimates over the first five data windows, each composed of twenty cardiac cycles, and averaging the median values of the resulting posterior distributions to form a stable, patient-specific baseline.

- describe ICP change tracking technique

The change tracking technique employs a Kalman-filter-like algorithm that fuses observed changes in intracranial pressure with model-predicted changes, using variance estimates to weight each source and thereby reducing dependence on the initial prior distribution over time.

- introduce statistical model

The statistical model is a finite impulse response representation of cerebral hemodynamics that maps cerebral perfusion pressure to cerebral blood flow velocity using two parameters: resistance and compliance, both assumed constant within a data window.

- describe physiological model

The physiological model is derived from a two-element Windkessel model of the cerebral vasculature, where resistance represents the opposition to blood flow and compliance represents the elastic storage capacity of cerebral arteries and brain tissue.

- describe Bayesian statistical techniques

Bayesian statistical techniques are used to combine prior knowledge about intracranial pressure with likelihood information derived from the observed data, yielding a posterior distribution that quantifies the probability of each possible intracranial pressure value.

- describe time shift estimation technique

The time shift estimation technique involves scanning a range of candidate time offsets between arterial blood pressure and cerebral blood flow velocity waveforms, evaluating the model fit for each, and selecting the offset that maximizes the likelihood of the intracranial pressure estimate.

- describe time offset range estimation

The time offset range is estimated on a window-by-window basis, constrained by physiological rules such as the expected delay between systolic peaks and the requirement that diastolic endpoints remain aligned within a tolerance of three sampling intervals.

- describe optimization routine

The optimization routine minimizes the prediction error between the modeled and observed cerebral blood flow velocity waveforms by solving a least-squares problem for each combination of intracranial pressure and time offset candidate.

- describe parameter value estimation

Parameter values for cerebrovascular resistance and compliance are estimated as intermediate outputs of the optimization routine, derived from the pseudo-inverse of the design matrix formed by the arterial pressure and its time-delayed version.

- describe prediction change model

The prediction change model is a first-order autoregressive process that forecasts the change in intracranial pressure between consecutive data windows based on the previous change estimate and a white noise disturbance term.

- describe prediction error estimation

Prediction error estimation involves computing the residual norm between the predicted and observed cerebral blood flow velocity waveforms after applying the model with a given intracranial pressure and time offset.

- describe ABP and CBFV waveform alignment

Alignment of arterial blood pressure and cerebral blood flow velocity waveforms is achieved by applying a time shift that satisfies two physiological constraints: the CBFV systolic upstroke leads the ABP systolic upstroke, and the diastolic endpoints are within a defined temporal tolerance.

- describe shifted CBFV waveforms

Shifted cerebral blood flow velocity waveforms are generated by applying the optimal time offset to the raw CBFV signal, enabling the model to accurately relate the arterial pressure waveform to the flow velocity waveform.

- describe optimization routine for model parameters

The optimization routine for model parameters iteratively evaluates the least-squares error for each candidate intracranial pressure and time offset pair, using matrix pseudo-inversion to compute resistance and compliance values that minimize the prediction error.

- describe prediction of physiological signals

The system predicts the cerebral blood flow velocity signal by applying the finite impulse response model to the arterial blood pressure signal minus the candidate intracranial pressure value, producing a synthetic waveform that can be compared to the observed signal.

- describe using one signal to predict another

The arterial blood pressure signal is used to predict the cerebral blood flow velocity signal through a linear, time-delayed transformation governed by the model parameters, with intracranial pressure as the hidden variable that must be inferred.

- describe optimization routine for ICP values

The optimization routine for intracranial pressure values involves scanning a discrete grid of candidate pressures, computing the corresponding prediction error for each, and identifying the pressure value that minimizes the error under the constraint of physiological plausibility.

- describe predicted CBFV waveforms

Predicted cerebral blood flow velocity waveforms are generated by the model for each candidate intracranial pressure and time offset pair, serving as the basis for evaluating the likelihood of each candidate.

- describe prediction errors

Prediction errors are quantitative measures of the discrepancy between the model’s predicted cerebral blood flow velocity waveform and the actual measured waveform, used to construct the likelihood distribution over intracranial pressure values.

- describe Bayesian statistical framework

The Bayesian statistical framework provides a principled method for combining prior knowledge about intracranial pressure with likelihood information derived from observed data, yielding a posterior distribution that represents the most probable intracranial pressure value given all available evidence.

- describe likelihood of ICP computation

The likelihood of intracranial pressure is computed as the exponential of the negative prediction error norm, normalized across the candidate space to form a probability distribution that reflects the plausibility of each pressure value.

- describe combining likelihood distributions

Likelihood distributions computed over different time offsets are combined by marginalizing over the offset dimension, resulting in a one-dimensional likelihood distribution that depends only on intracranial pressure.

- describe data processing pipeline

The data processing pipeline consists of signal acquisition, time alignment, resampling, baseline adjustment, out-of-band noise removal, model-based likelihood computation, Bayesian fusion, and dynamic tracking, all executed in real time to produce continuous intracranial pressure estimates.

- describe likelihood distribution of ICP

The likelihood distribution of intracranial pressure is a probability density function that assigns higher values to pressure estimates that yield better model fits, as determined by the prediction error between observed and predicted cerebral blood flow velocity waveforms.

- describe exponential relationship between likelihood and prediction errors

The likelihood of a given intracranial pressure value is exponentially related to the negative prediction error, such that a small error yields a high likelihood and a large error yields a low likelihood, ensuring that the most accurate model fits are assigned the highest probability.

- describe inverse relationship between likelihood and prediction errors

An inverse relationship exists between the prediction error and the likelihood, with the likelihood decreasing as the square of the error increases, ensuring that the system is highly sensitive to even minor deviations in model fit.

- describe prior distribution of ICP

The prior distribution of intracranial pressure is a mixture of two truncated Gaussian distributions, one centered at a low pressure value and the other at a high pressure value, designed to reflect the broad spectrum of clinical presentations while assigning higher probability to values outside the typical range to ensure generalizability.

- describe combining likelihood and prior distributions

The likelihood distribution and prior distribution are combined through pointwise multiplication, followed by normalization to form the posterior distribution, which represents the updated belief about intracranial pressure after observing the data.

- describe posterior distribution of ICP

The posterior distribution of intracranial pressure is the final probability distribution obtained after combining the likelihood of the observed data with the prior belief about plausible pressure values, and serves as the basis for point estimation and uncertainty quantification.

- describe estimated ICP value determination

The estimated intracranial pressure value is determined as the median of the posterior distribution, chosen for its robustness to outliers and skewness, and is accompanied by a confidence interval derived from the variance of the distribution.

- describe using posterior distribution for ICP estimation

The posterior distribution is used not only to determine a point estimate of intracranial pressure but also to quantify the uncertainty of that estimate, enabling clinicians to assess the reliability of each reading and respond appropriately to changes in confidence.

- conclude ICP estimation technique

The intracranial pressure estimation technique described herein provides a robust, physiologically grounded, and clinically viable method for continuous, noninvasive monitoring of intracranial pressure, overcoming the limitations of prior approaches through the integration of statistical inference, dynamic modeling, and real-time signal processing.

- introduce noninvasive intracranial pressure estimation method

The noninvasive intracranial pressure estimation method described herein enables continuous, real-time monitoring of intracranial pressure without the need for invasive sensors, calibration, or prior knowledge of the patient’s baseline pressure, by leveraging the relationship between radial arterial blood pressure and cerebral blood flow velocity through a model-based Bayesian framework.

- describe first-order subject-specific model of cerebral vasculature

The method employs a first-order, subject-specific model of cerebral vasculature that captures the dynamic relationship between cerebral perfusion pressure and cerebral blood flow velocity using two parameters: resistance and compliance, both of which are estimated from the data for each patient.

- explain model-based estimation within probabilistic framework

Model-based estimation is performed within a probabilistic framework that treats intracranial pressure as an unknown parameter to be inferred from observed signals, using likelihood functions derived from prediction errors and prior distributions informed by clinical evidence.

- outline process for establishing baseline ICP estimate

The process for establishing a baseline intracranial pressure estimate involves acquiring the first five data segments, computing the posterior distribution for each, extracting the median value, and averaging these medians to form a stable, patient-specific baseline.

- describe single-state model of cerebral autoregulatory dynamics

The single-state model of cerebral autoregulatory dynamics assumes that cerebrovascular resistance and compliance remain constant within each data window, while intracranial pressure evolves slowly over time according to a first-order autoregressive process.

- summarize performance characteristics of method

The method achieves a mean bias of less than 1 mmHg and a root mean square error of less than 4 mmHg across a diverse cohort of patients, with 80% of estimates falling within ±5 mmHg of invasive reference measurements, demonstrating clinical-grade accuracy without the need for calibration.

- introduce importance of intracranial pressure measurement

Accurate and continuous measurement of intracranial pressure is of paramount importance in neurocritical care, as it directly informs therapeutic decisions aimed at preventing secondary brain injury, optimizing cerebral perfusion, and improving patient outcomes.

- describe current clinical practice for ICP measurement

Current clinical practice for intracranial pressure measurement relies almost exclusively on invasive techniques, which are limited to specialized settings due to their associated risks, procedural complexity, and inability to support longitudinal monitoring.

- discuss limitations of current ICP measurement modalities

The limitations of current modalities include infection risk, procedural complications, sensor drift, lack of portability, and exclusion of large patient populations from monitoring due to the invasiveness of the procedures.

- introduce noninvasive ICP estimation schemes

Noninvasive ICP estimation schemes have been proposed using surrogate markers such as optic nerve sheath diameter, transcranial Doppler indices, and cranial compliance metrics, but none have achieved the accuracy, reliability, or physiological grounding required for clinical adoption.

- describe physiologic model-based methods

Physiologic model-based methods attempt to relate arterial pressure and cerebral flow velocity through biomechanical models, but prior approaches have been limited by fixed assumptions about time delays, lack of uncertainty quantification, and inability to track dynamic changes.

- outline Bayesian estimation framework

The Bayesian estimation framework employed in this invention provides a principled method for integrating prior knowledge with observed data, enabling robust estimation even in the presence of noise, uncertainty, and physiological variability.

- describe model of cerebral hemodynamics

The model of cerebral hemodynamics is a discrete-time, first-order finite impulse response system that transforms cerebral perfusion pressure into cerebral blood flow velocity using two parameters: resistance and compliance, both of which are estimated from the data.

- explain time-varying, first-order FIR filter approximation

The time-varying, first-order FIR filter approximation captures the dynamic relationship between cerebral perfusion pressure and cerebral blood flow velocity as a linear convolution with two time-invariant coefficients, valid within each data window.

- describe AR process description of ICP dynamics

The autoregressive process description of intracranial pressure dynamics models the change in pressure between consecutive data windows as a first-order linear recurrence with a white noise disturbance, enabling tracking of transient changes over time.

- outline model-based estimation algorithm

The model-based estimation algorithm proceeds by scanning a grid of candidate intracranial pressure and time offset values, computing the likelihood of each based on prediction error, combining with a prior distribution, and extracting the median of the posterior as the estimate.

- describe use of radial arterial blood pressure

Radial arterial blood pressure is used as a surrogate for central arterial pressure, with hydrostatic corrections applied to account for vertical displacement between measurement sites, ensuring accurate computation of cerebral perfusion pressure.

- explain probabilistic estimation framework

The probabilistic estimation framework treats intracranial pressure as a random variable whose value is inferred from observed data using Bayesian inference, allowing for the quantification of uncertainty and the incorporation of prior knowledge.

- describe process for establishing baseline ICP

The process for establishing baseline intracranial pressure involves computing posterior estimates over the first five data windows, each comprising twenty cardiac cycles, and averaging the median values to form a stable, patient-specific baseline.

- outline process for tracking changes in baseline ICP

The process for tracking changes in baseline intracranial pressure involves computing the change in pressure from each new data window, fusing it with a model-predicted change using a Kalman-filter-like algorithm, and adding the fused change to the baseline.

- describe use of uniform prior distribution

A uniform prior distribution is used in the tracking phase to reduce dependence on the initial prior belief, ensuring that subsequent estimates are driven primarily by the observed data and the dynamic model rather than by initial assumptions.

- explain filtering of ICP changes via Kalman filter-like approach

Intracranial pressure changes are filtered by combining the observed change with the model-predicted change, weighting each by the inverse of its estimated variance, thereby reducing noise and improving the accuracy of the tracking process.

- describe data description and method validation

The method was validated using data collected from thirteen patients with diverse neurological pathologies, including traumatic brain injury, hydrocephalus, and hemorrhagic stroke, with intracranial pressure values ranging from 1 to 25 mmHg.

- introduce illustrative implementation of computer system

An illustrative implementation of the computer system includes a processor, non-transitory memory storing executable instructions, input/output interfaces for signal acquisition, and a display for real-time visualization of intracranial pressure estimates.

- describe computer system components

The computer system components include a central processing unit, random-access memory, non-volatile storage media, analog-to-digital converters, network interfaces for signal acquisition, and user interfaces for display and alerting.

- explain processor control of data writing and reading

The processor controls the reading of raw signal data from external monitoring devices, the writing of processed estimates to memory and display, and the execution of the estimation algorithm in real time with minimal latency.

- describe non-transitory computer-readable storage media

The non-transitory computer-readable storage media store the executable instructions, data structures, and model parameters required to perform the intracranial pressure estimation method, and may be implemented as solid-state drives, flash memory, or other persistent storage devices.

- outline processor-executable instructions

The processor-executable instructions implement the full pipeline of signal acquisition, preprocessing, likelihood computation, Bayesian fusion, dynamic tracking, and output generation, enabling autonomous, real-time intracranial pressure estimation.

- describe network input/output interface

The network input/output interface connects the system to standard clinical monitoring devices, receiving digital streams of radial arterial blood pressure and cerebral blood flow velocity signals and transmitting processed estimates to hospital information systems.

- describe user input/output interfaces

User input/output interfaces include touchscreens, alarm indicators, and graphical displays that present intracranial pressure trends, confidence intervals, and alert thresholds to clinicians in real time.

- explain implementation of embodiments

Embodiments of the invention are implemented as standalone hardware devices, software modules integrated into existing patient monitors, or cloud-based services that process data transmitted from bedside sensors.

- describe computer-readable storage medium

The computer-readable storage medium contains the software instructions, data structures, and model parameters necessary to execute the intracranial pressure estimation method, and may be distributed as firmware, software updates, or downloadable applications.

- outline computer program

The computer program comprises a sequence of instructions that, when executed, perform the steps of signal acquisition, preprocessing, likelihood computation, Bayesian fusion, dynamic tracking, and output generation to produce continuous, noninvasive intracranial pressure estimates.

- describe Section A

Section A describes the physiological model of cerebral hemodynamics, including the derivation of the finite impulse response representation, the assumptions of constant resistance and compliance within a data window, and the inclusion of an autoregressive process for modeling temporal changes in intracranial pressure.

- introduce noninvasive intracranial pressure estimation method

The noninvasive intracranial pressure estimation method described herein provides a clinically viable, calibration-free, and physiologically grounded approach to continuous monitoring of intracranial pressure using readily available noninvasive signals.

- describe first-order subject-specific model of cerebral vasculature

The method employs a first-order, subject-specific model of cerebral vasculature that captures the dynamic relationship between cerebral perfusion pressure and cerebral blood flow velocity using two parameters: resistance and compliance, both estimated from the patient’s own data.

- explain model-based estimation within probabilistic framework

Model-based estimation is performed within a probabilistic framework that treats intracranial pressure as an unknown parameter to be inferred from observed data, using likelihood functions derived from prediction errors and prior distributions informed by clinical evidence.

- outline process for establishing baseline ICP estimate

The process for establishing a baseline intracranial pressure estimate involves acquiring the first five data segments, computing the posterior distribution for each, extracting the median value, and averaging these medians to form a stable, patient-specific baseline.

- describe single-state model of cerebral autoregulatory dynamics

The single-state model of cerebral autoregulatory dynamics assumes that cerebrovascular resistance and compliance remain constant within each data window, while intracranial pressure evolves slowly over time according to a first-order autoregressive process.

- summarize performance characteristics of method

The method achieves a mean bias of less than 1 mmHg and a root mean square error of less than 4 mmHg across a diverse cohort of patients, with 80% of estimates falling within ±5 mmHg of invasive reference measurements, demonstrating clinical-grade accuracy without the need for calibration.

- describe results of method

The results demonstrate that the method produces intracranial pressure estimates with accuracy comparable to invasive gold-standard measurements, with a Bland-Altman bias of 0.6 mmHg and limits of agreement of -6.6 to 7.7 mmHg, and that the inclusion of time offset scanning and Bayesian fusion significantly improves performance over prior methods.

- outline model of cerebral hemodynamics

The model of cerebral hemodynamics is a discrete-time, first-order finite impulse response system that transforms cerebral perfusion pressure into cerebral blood flow velocity using two parameters: resistance and compliance, both of which are estimated from the data.

- describe time-varying, first-order FIR filter approximation

The time-varying, first-order FIR filter approximation captures the dynamic relationship between cerebral perfusion pressure and cerebral blood flow velocity as a linear convolution with two time-invariant coefficients, valid within each data window.

- explain AR process description of ICP dynamics

The autoregressive process description of intracranial pressure dynamics models the change in pressure between consecutive data windows as a first-order linear recurrence with a white noise disturbance, enabling tracking of transient changes over time.

- describe model-based estimation algorithm

The model-based estimation algorithm proceeds by scanning a grid of candidate intracranial pressure and time offset values, computing the likelihood of each based on prediction error, combining with a prior distribution, and extracting the median of the posterior as the estimate.

- outline data description and method validation

The method was validated using data collected from thirteen patients with diverse neurological pathologies, including traumatic brain injury, hydrocephalus, and hemorrhagic stroke, with intracranial pressure values ranging from 1 to 25 mmHg.

- conclude method summary

The method provides a robust, accurate, and clinically feasible approach to noninvasive intracranial pressure monitoring, overcoming the limitations of prior techniques through the integration of physiological modeling, Bayesian inference, and dynamic tracking.

- introduce data collection

Data were collected from thirteen patients at Boston Children’s Hospital under institutional review board approval, with simultaneous recordings of radial arterial blood pressure, transcranial Doppler cerebral blood flow velocity, and invasive intracranial pressure.

- describe data collection protocol

The data collection protocol involved continuous recording of all signals for approximately twenty minutes per patient, with metadata including transducer height differences recorded to enable hydrostatic correction of cerebral perfusion pressure.

- introduce metadata recording

Metadata including the vertical displacement between the radial arterial and intracranial pressure transducers were recorded to enable accurate computation of cerebral perfusion pressure, ensuring physiological validity of the model inputs.

- describe data segments extraction

Data segments were extracted from continuous recordings by identifying periods of stable signal quality, removing segments contaminated by motion artifacts, signal dropout, or arrhythmias, and retaining only those suitable for model-based estimation.

- introduce signal conditioning stage

A signal conditioning stage was applied to remove baseline drift, out-of-band noise, and low-frequency trends, followed by bandpass filtering to retain only the physiologically relevant frequency content between 0.5 and 16 Hz.

- describe estimation routine

The estimation routine computes intracranial pressure estimates for each twenty-beat data window by evaluating a likelihood function over a range of candidate pressures and time offsets, combining with a prior distribution, and extracting the median of the posterior.

- introduce ICP estimation results

The ICP estimation results demonstrate that the method produces estimates with high accuracy and low bias across a heterogeneous patient population, with performance metrics comparable to invasive measurements.

- describe estimation results for patient 1

For patient 1, the method produced a mean bias of 0.4 mmHg and a root mean square error of 3.1 mmHg, with 85% of estimates falling within ±5 mmHg of the invasive reference.

- describe estimation results for patient 3

For patient 3, the method achieved a mean bias of 0.9 mmHg and a root mean square error of 3.8 mmHg, with 78% of estimates within ±5 mmHg, demonstrating robustness across varying pathologies.

- describe estimation results for patient 11

For patient 11, the method yielded a mean bias of 0.2 mmHg and a root mean square error of 2.9 mmHg, with 90% of estimates within ±5 mmHg, indicating superior performance in a patient with stable intracranial dynamics.

- perform Bland-Altman analysis

Bland-Altman analysis across all 1,657 estimation windows revealed a mean bias of 0.6 mmHg, a standard deviation of 3.7 mmHg, and limits of agreement of -6.6 to 7.7 mmHg, confirming clinical acceptability.

- describe per-estimation-window analysis

Per-estimation-window analysis demonstrated consistent performance across all patients, with no systematic drift or bias over time, and high reproducibility of the posterior median estimates.

- describe per-recording analysis

Per-recording analysis showed a mean bias of 0.8 mmHg and a root mean square error of 3.3 mmHg, with limits of agreement of -5.5 to 7.1 mmHg, confirming the stability of the method over extended monitoring periods.

- describe per-patient analysis

Per-patient analysis confirmed that the method maintained high accuracy across all thirteen patients, regardless of age, pathology, or baseline intracranial pressure, demonstrating broad generalizability.

- compute fraction of nICP estimates

The fraction of noninvasive intracranial pressure estimates within ±5 mmHg of the invasive reference was computed to be 79.8%, indicating strong agreement and clinical utility.

- discuss invasive ICP measurement modalities

Invasive intracranial pressure measurement modalities include external ventricular drains, intraparenchymal microtransducers, and subdural probes, all of which carry risks of infection, hemorrhage, and mechanical failure.

- compare with previous methods

Compared to previous noninvasive methods, this invention achieves superior accuracy, eliminates the need for calibration, and provides continuous, real-time tracking of intracranial pressure changes without reliance on pre-trained models.

- discuss advantages of noninvasive approach

The advantages of the noninvasive approach include reduced risk of infection, lower cost, greater portability, and the ability to monitor patients who are currently excluded from intracranial pressure monitoring due to the invasiveness of current techniques.

- discuss challenges in adopting model-based nICP estimation

Challenges in adopting model-based noninvasive intracranial pressure estimation include sensitivity to signal quality, variability in waveform morphology, and the need for robust algorithms that can handle physiological noise and uncertainty.

- discuss differences from previous methods

This invention differs from previous methods by explicitly modeling the time offset between arterial and flow velocity waveforms, employing a Bayesian framework for uncertainty quantification, and incorporating dynamic tracking to reduce dependence on initial prior assumptions.

- discuss interpretability of approach

The approach is highly interpretable because it is grounded in a simple, physiologically plausible model of cerebral hemodynamics, allowing clinicians to understand how estimates are derived and why confidence varies over time.

- discuss Bayesian framework

The Bayesian framework enables the system to incorporate prior knowledge, quantify uncertainty, and update beliefs in real time, making the method robust to noise, missing data, and physiological variability.

- discuss estimation performance

Estimation performance was validated across a diverse cohort of patients and demonstrated clinical-grade accuracy, with a mean bias of less than 1 mmHg and a root mean square error of less than 4 mmHg.

- discuss potential applications

Potential applications include neurocritical care, intraoperative monitoring, emergency department triage, pediatric intensive care, and remote monitoring in resource-limited settings, where continuous intracranial pressure monitoring is clinically valuable but currently unattainable.

- introduce data processing

Data processing includes time alignment, resampling, baseline adjustment, and out-of-band noise removal to ensure that the input signals are suitable for model-based estimation.

- describe time alignment step

The time alignment step identifies the optimal delay between radial arterial blood pressure and cerebral blood flow velocity waveforms by maximizing the cross-correlation under physiological constraints.

- describe resampling step

The resampling step ensures that both signals are sampled at a common rate of 125 Hz, eliminating temporal discrepancies due to differing sampling frequencies of the acquisition devices.

- describe baseline adjustment step

The baseline adjustment step corrects for hydrostatic pressure differences between the radial arterial and intracranial pressure transducers by applying a correction factor based on the vertical displacement between the measurement sites.

- describe out-of-band-noise removal stage

The out-of-band-noise removal stage applies a moving-average filter to extract and remove low-frequency trends, followed by a bandpass filter to eliminate high-frequency noise outside the physiologically relevant range of 0.5 to 16 Hz.

- describe FIR model derivation

The FIR model is derived from a first-order approximation of the continuous-time Windkessel model, expressing cerebral blood flow velocity as a linear convolution of cerebral perfusion pressure with two time-invariant coefficients.

- describe model-based Bayesian estimation routine

The model-based Bayesian estimation routine evaluates a grid of candidate intracranial pressure and time offset values, computes the likelihood of each based on prediction error, combines with a prior distribution, and extracts the median of the posterior as the estimate.

- describe time offset scan range

The time offset scan range is determined on a window-by-window basis, constrained by physiological rules such as the expected delay between systolic upstrokes and the requirement that diastolic endpoints remain within three sampling intervals.

- describe ICP scan range

The ICP scan range spans from 0 mmHg to the mean radial arterial blood pressure in the window, in increments of 1 mmHg, ensuring that all physiologically plausible values are evaluated.

- compute estimates for α and β

Estimates for the model parameters α and β are computed using a least-squares pseudo-inverse solution, where α corresponds to resistance and β to compliance, derived from the arterial pressure and its time-delayed version.

- define likelihood distribution

The likelihood distribution is defined as the exponential of the negative prediction error norm, normalized across the candidate space to form a probability density function over intracranial pressure and time offset.

- marginalize likelihood distribution

The likelihood distribution is marginalized over the time offset dimension to produce a one-dimensional likelihood distribution that depends only on intracranial pressure.

- define a posteriori distribution

The a posteriori distribution is defined as the product of the likelihood distribution and the prior distribution, normalized to integrate to unity, representing the updated belief about intracranial pressure after observing the data.

- derive mode and variance of combined distribution

The mode and variance of the combined distribution are derived as the median and interquartile range of the posterior, providing a robust point estimate and measure of uncertainty.

- specify prior belief

The prior belief is specified as a mixture of two truncated Gaussian distributions, one modeling low intracranial pressure and the other modeling high intracranial pressure, with parameters selected to ensure broad generalizability.

- select parameters for prior belief

Parameters for the prior belief were selected based on pilot data from three subjects, with means set at 13.6 mmHg and 50 mmHg, standard deviations of 10 mmHg and 20 mmHg, and weights of 0.8 and 0.2 to reflect clinical prevalence.

- establish baseline ICP estimation

Baseline intracranial pressure estimation is established by computing the posterior median over the first five twenty-beat data windows and averaging the results to form a stable, patient-specific baseline.

- compute a posteriori mode estimates

A posteriori mode estimates are computed for each data window by identifying the median of the posterior distribution, ensuring robustness to outliers and skewness.

- set baseline ICP

The baseline intracranial pressure is set as the average of the a posteriori mode estimates from the first five data windows, serving as the reference point for subsequent tracking.

- pass baseline to tracking stage

The baseline intracranial pressure is passed to the tracking stage, where subsequent estimates are computed using a uniform prior to reduce dependence on the initial prior belief.

- compute reference nICP and variance

The reference noninvasive intracranial pressure and its variance are computed for each data window as the median and interquartile range of the posterior distribution.

- initialize tracking filter

The tracking filter is initialized by setting the initial change estimate and its variance to zero, ensuring that the first tracking update is driven entirely by the observed data.

- compute filtered ICP-change estimates

Filtered intracranial pressure change estimates are computed by fusing the observed change with the model-predicted change using a Kalman-filter-like weighting scheme based on the inverse of their respective variances.

- combine observed and model-predicted changes

Observed and model-predicted changes are combined by weighting each by the inverse of its estimated variance, ensuring that the more reliable source contributes more to the final estimate.

- update filtered change estimates

Filtered change estimates are updated for each new data window by applying the autoregressive model to the previous filtered change and fusing it with the new observed change.

- compute final nICP estimate

The final noninvasive intracranial pressure estimate is computed by adding the filtered change estimate to the baseline intracranial pressure, producing a continuous, real-time estimate of intracranial pressure.

- define program or software

The program or software embodiment comprises a set of processor-executable instructions stored on a non-transitory computer-readable medium that, when executed, implement the full intracranial pressure estimation pipeline.

- specify computer-executable instructions

The computer-executable instructions include routines for signal acquisition, preprocessing, likelihood computation, Bayesian fusion, dynamic tracking, and output generation, all optimized for real-time execution.

- describe data structures

Data structures include arrays for storing raw signals, intermediate likelihood values, posterior distributions, filtered change estimates, and baseline values, organized to support efficient access and computation.

- define relationships among data elements

Relationships among data elements are defined by the mathematical operations of the model, including the convolution of arterial pressure with model parameters, the computation of prediction errors, and the Bayesian fusion of likelihood and prior distributions.

- describe processes

Processes include signal conditioning, time offset scanning, likelihood computation, posterior derivation, filtering, and output generation, all executed in a sequential, real-time pipeline.

- define phrase "at least one"

The phrase “at least one” is used herein to mean one or more, and is intended to encompass singular and plural instances unless otherwise specified.

- define phrase "and/or"

The phrase “and/or” is used herein to mean any one of the listed elements, any combination of them, or all of them, and is intended to cover all logical possibilities.

- describe use of ordinal terms

Ordinal terms such as “first,” “second,” and “third” are used herein to distinguish between elements and do not imply a specific order, sequence, or priority unless explicitly stated.

- provide disclaimer and scope of disclosure

The disclosure herein is illustrative and not limiting. Modifications and variations may be made without departing from the spirit and scope of the invention as defined by the claims.