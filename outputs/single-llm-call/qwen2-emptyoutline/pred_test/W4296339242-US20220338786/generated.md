# DESCRIPTION

## STATEMENT OF GOVERNMENT RIGHTS

The invention described herein was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## BACKGROUND

Deep brain stimulation (DBS) is a therapeutic technique that modulates neural activity by delivering electrical impulses through electrodes implanted in the brain. It is widely used to treat various neurological disorders, including Parkinson’s disease, epilepsy, and obsessive-compulsive disorder (OCD). Despite its effectiveness, optimizing DBS therapy remains challenging due to the need for precise control over stimulation parameters and the ability to monitor and adjust these parameters in real-time based on neural signals.

One critical aspect of DBS is the removal of stimulation artifacts from the recorded neural signals. These artifacts, caused by the high-amplitude and high-frequency electrical impulses, can severely distort the underlying neural signals, making it difficult to identify and analyze biomarkers. Traditional methods for artifact removal often fail to address the complexities introduced by unknown phase shifts, low sampling rates, and missing data, which are common in real-world DBS applications.

This background highlights the need for a robust and efficient method to remove stimulation artifacts in DBS, particularly one that can handle the aforementioned challenges. The invention described herein addresses these issues by providing a novel algorithm for periodic artifact removal and reconstruction, which can accurately estimate and remove stimulation artifacts even in the presence of unknown phase shifts and low sampling rates.

## SUMMARY

The present invention relates to a method and system for removing periodic artifacts from deep brain stimulation (DBS) recordings. The method involves an iterative algorithm that accurately estimates the artifact frequency and phase shifts, even in the presence of unknown phase shifts and low sampling rates. The algorithm uses harmonic regression to fit the observed signal and remove the artifact, while jointly estimating the frequency and phase shifts. An initialization algorithm is also provided to robustly estimate the initial parameters for the artifact removal algorithm.

Key features of the invention include:
- Accurate estimation of the artifact frequency and phase shifts.
- Handling of unknown phase shifts and low sampling rates.
- Efficient computation suitable for real-time applications.
- Potential for implementation in embedded systems for closed-loop DBS therapies.

The invention is particularly useful in the context of closed-loop DBS systems, where real-time artifact removal is essential for identifying and adjusting to biomarkers. The method can significantly improve the quality of neural signal recordings, enabling better understanding of the biological mechanisms underlying the success of DBS and facilitating the development of more effective therapeutic strategies.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

### Computer System

The invention can be implemented using a computer system comprising one or more processors, memory, and input/output interfaces. The computer system may include specialized hardware, such as digital signal processors (DSPs) or field-programmable gate arrays (FPGAs), to efficiently execute the artifact removal algorithms. The system may also include software modules for data acquisition, preprocessing, and post-processing of the neural signals.

### Period-Based Artifact Reconstruction and Removal for Deep Brain Stimulation

The core of the invention is a method for periodic artifact reconstruction and removal (PARRM) in deep brain stimulation (DBS) recordings. The method addresses the challenges of unknown phase shifts, low sampling rates, and missing data by using an iterative algorithm that jointly estimates the artifact frequency and phase shifts while fitting the observed signal using harmonic regression.

#### Model Formulation

Given \( n + 1 \) segments of data, the \( i \)-th segment of the observed signal \( S_i \) is modeled as:
\[ S_i(t) = A\left( t + \frac{\delta_i^*}{\xi^*} \right) + B_i(t) + \eta_i(t), \quad i = 0, \ldots, n, \]
where:
- \( \delta_i^* \) is the (unknown) phase shift between the 0-th and \( i \)-th segments.
- \( A \) is a periodic artifact with (unknown) period \( \frac{1}{\xi^*} \).
- \( B_i \) is the neural signal in segment \( i \).
- \( \eta_i \) is the noise in segment \( i \).

The goal is to estimate and remove \( A \) from each \( S_i \), thereby recovering the underlying signals \( B_i + \eta_i \).

#### Loss Function

To achieve this, the following loss function is used:
\[ \mathcal{L}(\xi, \delta_i, \mathbf{\theta}) = \sum_{i=0}^n \sum_{t \in T_i} \left( S_i(t) - a(t \mid \xi, \delta_i, \mathbf{\theta}) \right)^2, \]
where:
- \( a(t \mid \xi, \delta, \mathbf{\theta}) \approx A\left( t + \frac{\delta}{\xi^*} \right) \) is a model for the artifact.
- \( \mathbf{\theta} \) represents a set of parameters, including the amplitudes and harmonics of the artifact.

A parametric model for the artifact is given by:
\[ a(t \mid \xi, \delta, \alpha_0, \alpha_k, \beta_k, K) = \alpha_0 + \sum_{k=1}^K \left( \alpha_k \cos(2\pi k (\xi t + \delta)) + \beta_k \sin(2\pi k (\xi t + \delta)) \right). \]

### Comparison of PARRM to Conventional Filters

Conventional methods for removing DBS stimulation artifacts, such as those based on the discrete Fourier transform (DFT), often fail to handle unknown phase shifts and low sampling rates. DFT-based methods are limited in their accuracy for frequency estimation and cannot handle phase shifts effectively. In contrast, PARRM uses an iterative optimization approach that jointly estimates the frequency and phase shifts, leading to more accurate artifact removal.

### Periodic Estimation of Lost Packets From Deep Brain Stimulation Waveform Data

In DBS applications, data transmission can be unreliable, leading to missing packets or segments of data. PARRM addresses this issue by estimating the phase shifts between segments, allowing for the reconstruction of lost packets. The algorithm can handle multiple segments with unknown phase shifts, making it suitable for real-world scenarios where data integrity is compromised.

### Experimental Testing of the Period-Based Estimation of the Loss of Packets (PELP)

The performance of PARRM was evaluated through several numerical examples, including simulated and real-world data. The results demonstrate the algorithm's ability to accurately estimate and remove stimulation artifacts, even in the presence of unknown phase shifts and low sampling rates.

#### Example 1: Simulated Artifact with No Underlying Signal and No Phase Shifts

In this example, the underlying signals \( B_0 = \eta_0 = 0 \), and the sampling rate \( f_s = 1000 \) Hz with 10,000 samples. The algorithm successfully estimated the artifact frequency with a relative error of \( 3.7742 \times 10^{-14} \% \) and reconstructed the artifact with a relative RMSE of \( 1.7918 \times 10^{-10} \% \).

#### Example 2: Simulated Artifact with a Chirp and No Phase Shifts

In this example, the underlying signal \( B_0 \) is a chirp with frequencies ranging from 0 Hz to 500 Hz, and \( \eta_0 = 0 \). The algorithm accurately reconstructed the artifact with a relative RMSE of 0.5837%, and the recovered underlying signal matched the true signal with a relative RMSE of 5.5508%.

#### Example 3: Aliased Simulated Artifact with a Simulated Neural Signal, Noise, and Missing Data

In this example, the underlying signals \( B_i \) are simulated neural signals, and \( \eta_i \) is Gaussian noise. The sampling rate \( f_s = 250 \) Hz, and the data is segmented into 10 contiguous segments with random gaps. The algorithm estimated the artifact frequency with a relative error of \( 2.3023 \times 10^{-3} \% \) and reconstructed the artifact with a relative RMSE of 5.5521%.

#### Example 4: Human Local Field Potential (LFP) Recording with Missing Data

In this example, a real human LFP recording with DBS was used. The data was segmented into 10 contiguous segments with random gaps. The algorithm estimated the artifact frequency with a relative error of approximately \( 150.6093 \) Hz and successfully removed the artifact, as evidenced by the power spectral density plots.

### Period-Based Estimation of Electrical Stimulation Artifacts in the Presence of Phase Shifts

PARRM is designed to handle phase shifts between segments of data, which are common in DBS applications. The algorithm uses a modified Newton's descent method to solve the optimization problem, ensuring accurate estimation of the artifact frequency and phase shifts. The initialization algorithm provides robust initial estimates, enhancing the convergence of the main algorithm.

In summary, the invention provides a robust and efficient method for removing periodic artifacts from DBS recordings, addressing the challenges of unknown phase shifts, low sampling rates, and missing data. The method has the potential for real-time implementation in embedded systems, making it suitable for closed-loop DBS therapies and biomarker discovery.