# DESCRIPTION

## FIELD

The field of the invention pertains to medical devices and methods for noninvasive estimation of intracranial pressure (ICP). Specifically, the invention relates to a method and system for estimating ICP using a combination of physiological models, Bayesian estimation techniques, and time-offset considerations to enhance the accuracy and reliability of noninvasive ICP monitoring.

## BACKGROUND

Intracranial pressure (ICP) monitoring is crucial in the management of various neurological conditions, including traumatic brain injury (TBI), hydrocephalus, and hemorrhagic strokes. Traditional methods for ICP monitoring involve invasive techniques such as external ventricular drains (EVDs) and parenchymal sensors, which carry significant risks of infection and other complications. Noninvasive methods for ICP estimation have been explored to mitigate these risks, but existing approaches often suffer from limitations in accuracy and robustness.

One of the key challenges in noninvasive ICP estimation is the accurate modeling of the relationship between cerebral blood flow velocity (CBFV) and cerebral perfusion pressure (CPP), which is the difference between arterial blood pressure (ABP) and ICP. Existing models often fail to account for the time offsets between these signals and the variability in ICP values, leading to suboptimal performance.

The present invention addresses these issues by providing a novel method for noninvasive ICP estimation that incorporates a physiologically inspired model of cerebral hemodynamics, Bayesian estimation techniques, and a systematic approach to handling time offsets. This method aims to provide accurate and reliable ICP estimates without the need for invasive procedures.

## SUMMARY

The invention provides a method and system for noninvasive estimation of intracranial pressure (ICP) using a combination of physiological modeling, Bayesian estimation, and time-offset considerations. The method involves the following steps:

1. **Modeling Cerebral Hemodynamics**: The method models the CBFV waveform as the output of a two-tap finite impulse response (FIR) filter, where the input is the cerebral perfusion pressure (CPP), defined as the difference between ABP and ICP. The model accounts for the time offset between the CBFV and CPP signals.

2. **Likelihood Distribution Generation**: For a range of candidate ICP values and time offsets, the method computes the FIR filter taps by minimizing the CBFV prediction error in a least-square error sense. The resulting prediction errors are transformed into a likelihood distribution of ICP values.

3. **Baseline Estimation**: The method establishes a baseline ICP estimate by combining the likelihood distribution with a prior belief of ICP values and taking the median of the resulting a posteriori distribution. This process is repeated for several data windows, and the estimates are averaged to yield the baseline.

4. **Tracking Changes**: Subsequent ICP estimates are computed using a uniform prior distribution to reduce dependence on the initial prior belief. Changes in estimated ICP are filtered using a Kalman filter-like approach, which combines model-predicted ICP changes with noninvasively determined changes.

5. **Pulse Pressure Estimation**: The method also estimates the ICP pulse pressure by determining the model parameters and using them to reconstruct the ICP waveform.

The invention further includes a system for implementing the method, comprising a processor configured to execute the steps of the method and a memory for storing the necessary data and model parameters.

## DETAILED DESCRIPTION

### Field of the Invention

The invention pertains to medical devices and methods for noninvasive estimation of intracranial pressure (ICP). Specifically, the invention relates to a method and system for estimating ICP using a combination of physiological models, Bayesian estimation techniques, and time-offset considerations to enhance the accuracy and reliability of noninvasive ICP monitoring.

### Background of the Invention

Intracranial pressure (ICP) monitoring is essential in the management of various neurological conditions, including traumatic brain injury (TBI), hydrocephalus, and hemorrhagic strokes. Traditional methods for ICP monitoring involve invasive techniques such as external ventricular drains (EVDs) and parenchymal sensors, which carry significant risks of infection and other complications. Noninvasive methods for ICP estimation have been explored to mitigate these risks, but existing approaches often suffer from limitations in accuracy and robustness.

One of the key challenges in noninvasive ICP estimation is the accurate modeling of the relationship between cerebral blood flow velocity (CBFV) and cerebral perfusion pressure (CPP), which is the difference between arterial blood pressure (ABP) and ICP. Existing models often fail to account for the time offsets between these signals and the variability in ICP values, leading to suboptimal performance.

The present invention addresses these issues by providing a novel method for noninvasive ICP estimation that incorporates a physiologically inspired model of cerebral hemodynamics, Bayesian estimation techniques, and a systematic approach to handling time offsets. This method aims to provide accurate and reliable ICP estimates without the need for invasive procedures.

### Summary of the Invention

The invention provides a method and system for noninvasive estimation of intracranial pressure (ICP) using a combination of physiological modeling, Bayesian estimation, and time-offset considerations. The method involves the following steps:

1. **Modeling Cerebral Hemodynamics**: The method models the CBFV waveform as the output of a two-tap finite impulse response (FIR) filter, where the input is the cerebral perfusion pressure (CPP), defined as the difference between ABP and ICP. The model accounts for the time offset between the CBFV and CPP signals.

2. **Likelihood Distribution Generation**: For a range of candidate ICP values and time offsets, the method computes the FIR filter taps by minimizing the CBFV prediction error in a least-square error sense. The resulting prediction errors are transformed into a likelihood distribution of ICP values.

3. **Baseline Estimation**: The method establishes a baseline ICP estimate by combining the likelihood distribution with a prior belief of ICP values and taking the median of the resulting a posteriori distribution. This process is repeated for several data windows, and the estimates are averaged to yield the baseline.

4. **Tracking Changes**: Subsequent ICP estimates are computed using a uniform prior distribution to reduce dependence on the initial prior belief. Changes in estimated ICP are filtered using a Kalman filter-like approach, which combines model-predicted ICP changes with noninvasively determined changes.

5. **Pulse Pressure Estimation**: The method also estimates the ICP pulse pressure by determining the model parameters and using them to reconstruct the ICP waveform.

The invention further includes a system for implementing the method, comprising a processor configured to execute the steps of the method and a memory for storing the necessary data and model parameters.

### Detailed Description of the Invention

#### Modeling Cerebral Hemodynamics

The method employs a discrete-time approximation of a two-element continuous-time model of cerebral hemodynamics. For the \(m\)-th estimation window, the continuous-time model is of the form:

\[
q(t) = \frac{1}{R_m} \left( p_a(t) - I(t) \right) + \frac{1}{C_m} \int_0^t \left( p_a(\tau) - I(\tau) \right) d\tau
\]

where:
- \(q(t)\) is the cerebral blood flow velocity (CBFV),
- \(p_a(t)\) is the cerebral arterial blood pressure (cABP),
- \(I(t)\) is the intracranial pressure (ICP),
- \(R_m\) is the cerebrovascular resistance,
- \(C_m\) is the cerebrovascular compliance.

The resistive element \(R_m\) models resistance to cerebrovascular blood flow, while the capacitive element \(C_m\) represents the aggregate arterial and brain tissue compliance. The cerebral autoregulatory processes that modulate the resistance and compliance tend to occur over timescales longer than the data window lengths considered here, and hence both \(R_m\) and \(C_m\) are assumed constant over the duration of a data window, chosen to be 20 beats throughout this work.

The model is simplified by assuming that the ICP waveform is constant over the duration of an estimation window, leading to the dynamic relationship:

\[
q[n] = \alpha_m \left( p_a[n] - I[m] \right) + \beta_m \left( p_a[n-1] - I[m] \right)
\]

where:
- \(q[n]\) is the discrete-time CBFV,
- \(p_a[n]\) is the discrete-time cABP,
- \(I[m]\) is the mean ICP in the \(m\)-th estimation window,
- \(\alpha_m\) and \(\beta_m\) are the model parameters.

The model parameters \(\alpha_m\) and \(\beta_m\) are related to the cerebrovascular resistance and compliance by:

\[
\alpha_m = \frac{1}{R_m}, \quad \beta_m = \frac{\Delta t}{C_m}
\]

where \(\Delta t\) is the sampling interval.

#### Likelihood Distribution Generation

The method generates a likelihood distribution of ICP values by solving the model for a range of candidate ICP and time offset pairs. The time offset range is selected on a window-by-window basis to ensure that the CBFV peaks lead the corresponding rABP systolic peaks while aligning the diastolic points of the two waveforms. The ICP scan range starts from 0 mmHg in increments of 1 mmHg and stops at the mean rABP in the estimation window.

For each ICP and time offset pair, the method computes estimates for \(\alpha\) and \(\beta\) in a least-square error sense:

\[
\begin{bmatrix} \alpha \\ \beta \end{bmatrix}_{I,d} = \left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{q}
\]

where:
- \(\mathbf{X}\) is the design matrix containing the shifted and delayed cABP signals,
- \(\mathbf{q}\) is the vector of CBFV values.

The corresponding residual-error norm is given by:

\[
e_{I,d} = \left\| \mathbf{q} - \mathbf{X} \begin{bmatrix} \alpha \\ \beta \end{bmatrix}_{I,d} \right\|
\]

The likelihood distribution \(L(I, d)\) over the ICP and time offsets is defined as:

\[
L(I, d) = \frac{\exp\left( -\frac{e_{I,d}^2}{2 \sigma_e^2} \right)}{S_L}
\]

where \(S_L\) is chosen so that \(L(I, d)\) normalizes to one. This formulation assigns a high likelihood to (I, d) pairs that result in a small residual CBFV prediction error and a low likelihood to pairs with large residual error norms.

#### Baseline Estimation

To establish a baseline, the method computes a posteriori ICP estimates \(I_C\) in the first \(M_b = 5\) twenty-beat data windows. These estimates are averaged to yield the baseline \(I_B\):

\[
I_B = \frac{1}{M_b} \sum_{m=1}^{M_b} I_C[m]
\]

The baseline ICP is then passed to the subsequent tracking stage. This stage uses the ICP estimates \(I_L\) derived from the likelihood distribution alone, which amounts to using a uniform prior belief. Using a uniform belief, however, also increases the chances of erroneous ICP estimates. Therefore, the method develops a tracking framework that filters the changes in ICP estimates computed with the uniform prior belief.

#### Tracking Changes

The method filters changes in ICP estimates by combining the changes in ICP estimates with model-predicted changes obtained with an autoregressive (AR) process model. The AR process is of the form:

\[
\Delta I[m] = \gamma_m \Delta I[m-1] + v_m
\]

where:
- \(\Delta I[m]\) is the window-by-window difference in mean ICP,
- \(\gamma_m\) is the autoregulatory state parameter,
- \(v_m\) is a white-noise sequence with variance \(\sigma_v^2\).

The model-predicted ICP change and its variance are computed as:

\[
\hat{\Delta I}[m] = \gamma_m \Delta I[m-1]
\]

\[
\sigma_{\hat{\Delta I}}^2[m] = \gamma_m^2 \sigma_{\Delta I}^2[m-1] + \sigma_v^2
\]

The noninvasively determined ICP change and its variance are:

\[
\Delta I_L[m] = I_L[m] - I_L[m-1]
\]

\[
\sigma_{\Delta I_L}^2[m] = \sigma_L^2[m] + \sigma_L^2[m-1]
\]

The filtered change \(\Delta I[m]\) is then computed as:

\[
\Delta I[m] = \frac{\sigma_{\Delta I_L}^2[m]}{\sigma_{\Delta I_L}^2[m] + \sigma_{\hat{\Delta I}}^2[m]} \Delta I_L[m] + \frac{\sigma_{\hat{\Delta I}}^2[m]}{\sigma_{\Delta I_L}^2[m] + \sigma_{\hat{\Delta I}}^2[m]} \hat{\Delta I}[m]
\]

The final ICP estimate is obtained by adding the filtered change to the previous ICP estimate:

\[
I[m] = I[m-1] + \Delta I[m]
\]

#### Pulse Pressure Estimation

The method also estimates the ICP pulse pressure by determining the model parameters and using them to reconstruct the ICP waveform. The pulse pressure estimation procedure can be applied independently to each data window. For each window, the expectation operator yields:

\[
R_m = \frac{1}{\alpha_m}, \quad C_m = \frac{\Delta t}{\beta_m}
\]

The mean-subtracted ABP, CBFV, and ICP waveforms are used to rewrite the model equation:

\[
q(t) = \frac{1}{R_m} \left( p_a(t) - p_i(t) \right) + \frac{1}{C_m} \int_0^t \left( p_a(\tau) - p_i(\tau) \right) d\tau
\]

Taking Fourier transforms and rearranging the equation yields:

\[
P_i(j\Omega) = \frac{Q(j\Omega) - \frac{1}{R_m} P_a(j\Omega)}{\frac{1}{C_m} j\Omega + \frac{1}{R_m}}
\]

where:
- \(Q(j\Omega)\) is the Fourier transform of the mean-subtracted CBFV,
- \(P_a(j\Omega)\) is the Fourier transform of the mean-subtracted ABP,
- \(P_i(j\Omega)\) is the Fourier transform of the mean-subtracted ICP.

The ICP wavelet is reconstructed using the inverse DFT of the selected harmonics.

### Conclusion

The invention provides a method and system for noninvasive estimation of intracranial pressure (ICP) using a combination of physiological modeling, Bayesian estimation techniques, and time-offset considerations. The method accurately models the relationship between CBFV and CPP, generates a likelihood distribution of ICP values, establishes a baseline ICP estimate, tracks changes in ICP, and estimates the ICP pulse pressure. The system is designed to be robust, accurate, and suitable for real-time monitoring, making it a valuable tool for improving neurocritical care.