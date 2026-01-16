Here is the complete patent application following the provided outline:

# DESCRIPTION  

## FIELD  

The present invention relates generally to medical monitoring systems and methods, and more particularly to systems and methods for noninvasive estimation of intracranial pressure (ICP) using cerebral blood flow velocity (CBFV) and radial arterial blood pressure (rABP) measurements. The invention provides a novel computational framework that combines physiological modeling with Bayesian estimation techniques to derive accurate, patient-specific ICP estimates without requiring invasive procedures. The disclosed system and method are particularly useful for continuous monitoring of patients with traumatic brain injury, hydrocephalus, stroke, and other neurological conditions where ICP monitoring is clinically indicated but current invasive methods pose infection risks or are otherwise contraindicated.  

## BACKGROUND  

Intracranial pressure monitoring is a critical component of neurocritical care, as elevated ICP can lead to devastating neurological outcomes if left untreated. Current clinical practice relies almost exclusively on invasive measurement techniques such as external ventricular drains or intraparenchymal microtransducers. These invasive methods carry substantial risks including infection (reported in 5-10% of cases), hemorrhage, and mechanical complications. Despite decades of research, no noninvasive ICP monitoring technique has achieved sufficient accuracy and reliability to replace invasive methods in clinical practice.  

Prior attempts at noninvasive ICP estimation have faced several fundamental challenges. First, the complex relationship between measurable physiological signals (such as CBFV and arterial blood pressure) and ICP is influenced by multiple patient-specific factors including cerebrovascular resistance, compliance, and autoregulatory status. Second, unknown time delays between different physiological signals (particularly between CBFV and arterial blood pressure measurements) introduce significant estimation errors. Third, most existing approaches either rely on oversimplified models that cannot capture patient-specific physiology or employ complex machine learning techniques that require extensive training datasets and lack physiological interpretability.  

There remains an unmet clinical need for a noninvasive ICP monitoring system that: (1) provides accurate, patient-specific estimates without requiring calibration to invasive measurements; (2) accounts for physiological variability between patients; (3) can track dynamic changes in ICP over time; and (4) offers computational efficiency suitable for real-time bedside monitoring. The present invention addresses these needs through a novel modeling and estimation framework as described below.  

## SUMMARY  

The present invention provides a system and method for noninvasive estimation of intracranial pressure (nICP) that overcomes limitations of prior approaches. At its core, the invention models the relationship between cerebral perfusion pressure (CPP - calculated as the difference between arterial blood pressure and ICP) and cerebral blood flow velocity (CBFV) using a two-tap finite impulse response (FIR) filter representation. This physiological model forms the basis for a two-stage estimation process that first establishes a baseline ICP estimate and then tracks dynamic changes in ICP over time.  

Key aspects of the invention include:  

1. A computational framework that treats CBFV as the output of a two-tap FIR filter whose input is CPP, with filter parameters representing cerebrovascular resistance and compliance.  

2. A systematic approach to estimating model parameters by scanning across physiologically plausible ICP values and time offsets between CBFV and arterial blood pressure signals, then selecting parameter sets that minimize CBFV prediction error.  

3. Generation of a likelihood distribution for ICP values based on prediction errors across the parameter scan space, enabling probabilistic estimation rather than point estimates.  

4. A two-stage estimation process comprising:  
   a) A baseline estimation phase that combines the likelihood distribution with a preset prior distribution of ICP values  
   b) A tracking phase that monitors ICP changes using a uniform prior distribution and filters estimates through an autoregressive (AR) model of ICP dynamics  

5. Capability to estimate both mean ICP and ICP pulse pressure noninvasively using the same underlying model framework.  

The invention provides several advantages over prior approaches. The FIR filter model offers sufficient complexity to capture patient-specific physiology while remaining computationally tractable for real-time implementation. The Bayesian framework incorporating likelihood distributions provides robustness against measurement noise and model mismatch. The two-stage approach with AR-based tracking enables accurate monitoring over extended periods while reducing dependence on initial assumptions. Importantly, the system requires no calibration to invasive ICP measurements, making it suitable for screening applications where invasive monitoring may not be justified.  

## DETAILED DESCRIPTION  

The following detailed description provides a complete enabling disclosure of the invention, including its underlying computational framework, estimation algorithms, and implementation considerations.  

**Physiological Model Foundation**  

The invention is based on a discrete-time approximation of cerebrovascular dynamics where cerebral blood flow velocity (q[n]) is modeled as the output of a two-tap finite impulse response (FIR) filter whose input is cerebral perfusion pressure (p_cpp[n]). This relationship is expressed as:  

q[n] = α·p_cpp[n] + β·p_cpp[n-1]  

where α and β are the filter tap weights representing combined effects of cerebrovascular resistance and compliance, and n is the discrete-time index. The cerebral perfusion pressure is calculated as:  

p_cpp[n] = p_a[n] - I  

where p_a[n] is the arterial blood pressure (ABP) and I is the mean intracranial pressure (ICP). This formulation assumes that pulsatility in the ICP waveform can initially be neglected when estimating mean ICP.  

The continuous-time equivalents of the filter parameters relate directly to physiological quantities:  

α = 1/R + f_s·C  
β = -f_s·C  

where R is cerebrovascular resistance, C is aggregate arterial and brain tissue compliance, and f_s is the sampling frequency. These relationships enable interpretation of the estimated parameters in physiological terms.  

**Model Parameter Estimation**  

The invention estimates the model parameters (α, β) and mean ICP (I) through a systematic scanning process that:  

1. Generates candidate CPP waveforms by subtracting a range of physiologically plausible ICP values from the measured ABP signal  
2. For each candidate ICP value, computes optimal FIR filter taps (α, β) that minimize the least-squares error between measured and predicted CBFV  
3. Evaluates multiple time offsets between ABP and CBFV signals to account for physiological delays  

Mathematically, for each candidate ICP value (I) and time offset (d), the optimal filter taps are computed as:  

[α, β]^T = (X^T X)^(-1) X^T q  

where X is a matrix containing time-shifted ABP samples and q is a vector of CBFV measurements. The residual prediction error for each (I,d) pair is calculated as:  

ε(I,d) = ||q - X[α, β]^T||^2  

**Likelihood Distribution Formation**  

The invention transforms prediction errors into a likelihood distribution over ICP values using:  

L(I,d) = exp(-ε(I,d)/2σ^2)  

where σ is a normalization parameter. This distribution assigns higher likelihood to ICP values that yield better CBFV prediction. The likelihood is marginalized over time offsets to produce a one-dimensional distribution L(I) across ICP values.  

**Baseline Estimation Phase**  

The baseline estimation stage combines the likelihood distribution with a preset prior distribution Pr(I) to form a posterior distribution:  

P(I) = Pr(I)·L(I)  

The prior distribution incorporates clinical knowledge about typical ICP ranges, modeled as a mixture of truncated Gaussian distributions covering both normal and pathological ranges. The baseline ICP estimate is taken as the median of this posterior distribution, averaged across multiple data windows to improve robustness.  

**Tracking Phase**  

After establishing the baseline, subsequent ICP estimates use a uniform prior distribution to reduce dependence on initial assumptions. To mitigate increased estimation variance, the invention employs an autoregressive (AR) model of ICP dynamics:  

I[m] = I[m-1] + ΔI[m]  
ΔI[m] = γ·ΔI[m-1] + v[m]  

where m indexes estimation windows, γ is an AR coefficient, and v[m] is white noise. Changes in nICP estimates are filtered by combining:  
1) Observed changes from the likelihood distribution  
2) Predicted changes from the AR model  

The fusion process resembles a Kalman filter, weighting each component by its inverse variance to produce optimal estimates.  

**ICP Pulse Pressure Estimation**  

The invention extends the framework to estimate ICP pulse pressure by:  
1) Computing frequency-domain representations of mean-subtracted ABP and CBFV waveforms  
2) Solving the model equations in the frequency domain to estimate ICP harmonic components  
3) Reconstructing the ICP pulse waveform via inverse Fourier transform  

This approach provides information about ICP pulsatility without requiring additional model parameters or training data.  

**Implementation Considerations**  

The disclosed methods can be implemented in dedicated hardware or in software running on standard computing hardware. Key implementation aspects include:  

- Signal preprocessing (filtering, artifact removal, time alignment)  
- Automated quality assessment of input signals  
- Real-time computation of estimates within clinically relevant timeframes  
- User interface for displaying trends and alerts  

The system requires only two input signals: CBFV (measured via transcranial Doppler) and ABP (measured invasively or noninvasively). No patient-specific calibration is needed, making the system suitable for both continuous monitoring and spot-check applications.  

The complete specification discloses all features necessary for implementation by those skilled in the art, including mathematical formulations, algorithmic steps, and practical considerations for clinical deployment. The invention represents a significant advance in noninvasive ICP monitoring by combining physiological modeling with robust estimation techniques in a computationally efficient framework.