# DESCRIPTION

## FEDERALLY-SPONSORED RESEARCH AND DEVELOPMENT

The present invention was made with government support under the Naval Innovative Science and Engineering Program at the Space and Naval Warfare Systems Center Pacific. The government has certain rights in the invention.

## BACKGROUND

### Field of the Invention

The present invention relates generally to the field of underwater acoustics and, more specifically, to methods and systems for localizing and tracking acoustic sources using passive sonar. The invention addresses the challenges associated with multipath propagation and model mismatch in shallow-water environments, providing a robust and efficient solution for real-time tracking of multiple acoustic sources.

### Description of Related Art

Underwater source localization and tracking are critical tasks for various applications, including environmental monitoring, surveillance, and military operations. Traditional methods, such as matched-field processing (MFP) and matched-field tracking (MFT), have been widely used but suffer from limitations due to the complex nature of underwater acoustic propagation. Multipath propagation, which causes constructive and destructive interference at hydrophones, exacerbates the localization problem. Additionally, mismatches between the true propagation environment and the acoustic models used can lead to artifacts in the ambiguity surfaces, obscuring the true source locations.

MFT algorithms, while effective in post-processing, are computationally intensive and not well-suited for real-time applications. They require constructing a sequence of ambiguity surfaces and connecting peaks to form tracks, which becomes infeasible with increasing numbers of sources and peaks. Bayesian approaches, though more accurate, are computationally prohibitive for real-time tracking due to the high-dimensional state space and the need for Markov Chain Monte Carlo methods.

### Problems with the Prior Art

1. **Computational Complexity**: Traditional MFT and Bayesian methods are computationally expensive, making them unsuitable for real-time applications.
2. **Model Mismatch**: Mismatch between the true propagation environment and the acoustic model used can lead to significant errors in localization and tracking.
3. **Scalability**: Existing methods struggle to handle multiple sources and large datasets efficiently.
4. **Temporal Dependency**: Most methods do not effectively capture the temporal dependency between consecutive time instances, leading to less accurate tracking.

## DETAILED DESCRIPTION OF SOME EMBODIMENTS

### Overview of the Invention

The present invention provides a sparsity-driven framework for broadband source localization and tracking using passive sonar. The invention constructs source localization maps (SLMs) for each frequency and enforces coherence across these maps to ensure that the support of the SLMs coincides. By leveraging the sparsity of the source locations and the temporal dependency between consecutive time instances, the invention offers a computationally efficient and robust solution for real-time tracking of multiple acoustic sources.

### Problem Statement

Consider \( K \) broadband acoustic sources radiating sound underwater at \( F \) frequencies \(\{\omega_f\}_{f=1}^F\). The sources can move, and their positions at time \( t \) are denoted by \(\{r_k(t) \in \mathbb{R}^d\}_{k=1}^K\), where \( d = 2 \) or \( d = 3 \). An array of \( N \) hydrophones collects a time series of acoustic pressure measurements per hydrophone. The acoustic time series data is transformed to the frequency domain via a discrete-time Fourier transform. The Fourier coefficients at frequencies \(\{\omega_f\}_{f=1}^F\) across all hydrophones are collected per frequency into vectors \( y_f(t) \), where \( y_n^f(t) \) denotes the Fourier coefficient estimate corresponding to \(\omega_f\) at time \( t \) obtained from data gathered by the \( n \)-th hydrophone.

A model characterizing the acoustic propagation in the environment is available. If the replicas \( p_{k,f} \) for each source location \( r_k(t) \) transmitting at \(\omega_f\) were known, one could use them to compute model-predicted Fourier coefficients at the array. Let \( p_{k,f} \) denote the normalized replica for a source located at \( r_k(t) \) transmitting at \(\omega_f\), where the normalization implies that \( \|p_{k,f}\|_2 = 1 \). Each \( y_f(t) \) can be modeled as:

\[ y_f(t) = \sum_{k=1}^K s_{k,f}(t) p_{k,f} + \epsilon_f(t) \]

where \( \epsilon_f(t) \in \mathbb{C}^N \) denotes a zero-mean additive noise component, and \( s_{k,f}(t) \) the Fourier coefficient at \(\omega_f\) of the spectrum corresponding to the \( k \)-th source acoustic signature at time \( t \).

Given \( K \) and \(\{y_f(t), \forall f\}_t\), the goal of the invention is to find estimates for \(\{r_k(t)\}_{k=1}^K\). Even if all \( s_{k,f}(t) \) in the model were known, finding estimates for the source locations is challenging due to the nonlinear relationship between \( r_k(t) \) and \( p_{k,f} \), which is often not available in closed form. To address this, a grid of tentative source locations \( G := \{r_g\}_{g=1}^G \) with \( G \geq KF \) is introduced. The \( y_f(t) \)'s at time \( t \) can be modeled as:

\[ y_f(t) = \sum_{g=1}^G s_{g,f}(t) p_{g,f} + \epsilon_f(t) \]

where \( p_{g,f} \) denotes the normalized replica corresponding to a source located at \( r_g \in G \), and \( s_{g,f}(t) \) the Fourier coefficient at \(\omega_f\) of the spectrum corresponding to the acoustic signature of the source located at \( r_g \in G \) at time \( t \).

Since \( G \gg KF \), most of the \( s_{g,f}(t) \)'s are expected to be zero. Only those few \( s_{g,f}(t) \)'s that correspond to the true source locations should take nonzero values. Let:

\[ S(t) = [s_1(t), s_2(t), \ldots, s_F(t)] \]

Once an estimate for \( S(t) \) is available, a broadband SLM can be obtained by plotting the pairs \((r_g, \|\zeta_g(t)\|_2^2)\) for all \( r_g \in G \), where \( \zeta_g(t) := [s_{g,1}(t), s_{g,2}(t), \ldots, s_{g,F}(t)] \in \mathbb{C}^F \) comprises the entries of the \( g \)-th row of \( S(t) \). Source location estimates \(\{r_k(t)\}\) correspond to the locations of the \( K \)-largest peaks in the broadband SLM.

### Sparsity-Driven Tracking of Acoustic Sources

An iterative estimator for \( S(t) \) is proposed in this invention. The estimator uses the previously estimated \( S(t-1) \) to capture the temporal dependency between source locations at consecutive time instances. Per time \( t \), an estimate \( \hat{S}(t) = [\hat{s}_1(t), \hat{s}_2(t), \ldots, \hat{s}_F(t)] \) for \( S(t) \) is obtained as:

\[ \hat{S}(t) = \arg \min_S \left\{ \frac{1}{2} \sum_{f=1}^F \| y_f(t) - P_f S_f \|_2^2 + \mu \sum_{g=1}^G \|\zeta_g\|_2 + \lambda \sum_{f=1}^F \| s_f(t) - s_f(t-1) \|_2^2 \right\} \]

where \( S = [s_1, s_2, \ldots, s_F] \), \( \zeta_g \) denotes the \( g \)-th row of \( S \), \( P_f = [p_{1,f}, p_{2,f}, \ldots, p_{G,f}] \in \mathbb{C}^{N \times G} \) is the matrix of replicas for \(\omega_f\), and \( \lambda, \mu > 0 \) are tuning parameters. Note that the problem is a regularized least squares regression problem. The regularization term scaled by \( \mu \) encourages group sparsity on the rows of \( \hat{S}(t) \), with \( \mu \) controlling the number of nonzero rows in \( \hat{S}(t) \). The regularization term scaled by \( \lambda \) encourages estimates \( \hat{s}_f(t) \) to be close to \( \hat{s}_f(t-1) \), with \( \lambda \) controlling the emphasis placed on \( \hat{s}_f(t-1) \) when estimating \( s_f(t) \).

### Proximal Gradient Solver for Sparse Tracking

To solve the optimization problem, a proximal gradient (PG) algorithm is developed. The problem can be written as a real-valued convex optimization problem by representing all complex-valued variables by the direct sum of their real and imaginary parts. Let:

\[ y_f(t) = [ \text{Re}\{y_f(t)\}, \text{Im}\{y_f(t)\} ] \]
\[ s_f = [ \text{Re}(s_f), \text{Im}(s_f) ] \]
\[ S = [s_1, s_2, \ldots, s_F] \]

Matrix \( S \) can be viewed in terms of its rows as \( S = [\zeta_1, \zeta_2, \ldots, \zeta_{2G}] \), where the first \( G \) rows correspond to the real parts and the last \( G \) rows to the imaginary parts of the rows of \( S \). The optimization problem is equivalent to:

\[ \min_S \left\{ \frac{1}{2} \sum_{f=1}^F \| y_f(t) - P_f S_f \|_2^2 + \mu \sum_{g=1}^G \| v_g \|_2 + \lambda \sum_{f=1}^F \| s_f(t) - s_f(t-1) \|_2^2 \right\} \]

where \( v_g = [ \zeta_g, \zeta_{g+G} ] \in \mathbb{R}^{2F} \) corresponds to the direct sum of the real and imaginary parts of \( \zeta_g \) and \( \zeta_{g+G} \).

The PG method can be interpreted as a majorization-minimization method relying on a majorizer \( H(S; Z) \) for the continuously differentiable portion of the cost. The majorizer \( H \) satisfies:

1. \( H(S; Z) \geq h(S), \forall S \)
2. \( H(S; Z) = h(S) \) for \( Z = S \)

The specific \( H \) used by the PG method is:

\[ H(S; Z) = h(Z) + \langle \nabla h(Z), S - Z \rangle + \frac{L_h}{2} \| S - Z \|_F^2 \]

where \( h(S) = \frac{1}{2} \sum_{f=1}^F \| y_f(t) - P_f S_f \|_2^2 + \lambda \sum_{f=1}^F \| s_f(t) - s_f(t-1) \|_2^2 \), and \( L_h = \max_{f=1,\ldots,F} \sigma_{\max}(P_f^* P_f + \lambda I_{2G}) \) is the Lipschitz constant of the gradient of \( h \).

The PG algorithm iteratively solves:

\[ S[i](t) = \arg \min_S \left\{ H(S; S[i-1](t)) + \mu \sum_{g=1}^G \| v_g \|_2 \right\} \]

From an algorithmic viewpoint, it is convenient to write \( H \) as a function of the \( v_g \)'s. After performing some algebraic manipulations and dropping all terms independent of \( S \), the PG update can be written as:

\[ v_g[i](t) = \text{prox}_{\frac{\mu}{L_h} \| \cdot \|_2} \left( v_g[i-1](t) - \frac{1}{L_h} d_g[i-1](t) \right) \]

where \( d_g[i-1](t) \) is a gradient descent step for the \( g \)-th row of \( S \), and the proximal operator is given by:

\[ \text{prox}_{\alpha \| \cdot \|_2}(v) = \left( 1 - \frac{\alpha}{\| v \|_2} \right)_+ v \]

The resulting PG algorithm is summarized as follows:

1. Initialize \( S[0](t) \) and choose tuning parameters \( \lambda \) and \( \mu \).
2. For \( i = 1, 2, \ldots \):
   - Compute \( d_g[i-1](t) \) for all \( g \).
   - Compute \( v_g[i-1](t) \) for all \( g \).
   - Update \( v_g[i](t) \) using the proximal operator.
   - Terminate when the change in \( S \) is below a small threshold \( \epsilon \).

### Numerical Tests on SWellEX-3 Data

The performance of the proposed algorithm was tested using data from the third Shallow-Water Evaluation Cell Experiment (SWellEX-3) dataset. The experiment involved a towed source transmitting at 10 frequencies \(\{53 + 16k\}_{k=0}^9\) (all in Hertz) and a vertical line array with 64 hydrophones collecting acoustic data. In this analysis, the array was subsampled to 9 hydrophones evenly spaced over the length of the array, with a total aperture of 90 meters and the bottom element 6 meters above the seafloor. A grid with \( G = 20,000 \) locations spanning radial distances 0-10 kilometers and depths 0-198 meters was used, with a radial and vertical spacing of 50 meters and 2 meters, respectively. All replicas were computed using the KRAKEN normal-mode program.

The proposed algorithm was compared against MFT, a traditional baseline method. Despite its high computational complexity, MFT yields accurate track estimates for the single-source case. Ambiguity surfaces generated via Bartlett MFP were used to construct partial linear trajectories, also known as tracklets. A total of 8 ambiguity surfaces were used to construct each tracklet, with each surface accounting for 13.65 seconds of recorded data. There was a 50% overlap between consecutive tracklets.

Figures 2a and 2b illustrate the depth and range tracks obtained by MFT, while Figures 2c and 2d show the tracks obtained using the proposed algorithm. The proposed algorithm provides a coarse approximation to the source trajectory, with gaps appearing due to the high value of \( \mu \) at certain time instances. Both MFT and the proposed method fail to track the source after \( t = 65 \) minutes due to the severe mismatch between the environment and the model used.

To simulate the presence of two sources, data corresponding to the portions of the trajectory between 0-25 minutes and 40-65 minutes were combined after being rescaled to compensate for the signal-to-noise ratio difference. The computational complexity of MFT increased dramatically, and it failed to distinguish the two sources. Figure 3 illustrates the performance of the proposed algorithm in tracking the two sources. While the trajectories of the two sources can be observed in range, it is difficult to separate them in depth. Dynamic adjustment of the parameters is expected to improve the quality of the tracks obtained.

### Conclusion

The present invention provides a robust and efficient method for real-time tracking of multiple acoustic sources using passive sonar. By leveraging the sparsity of the source locations and the temporal dependency between consecutive time instances, the invention offers a computationally efficient solution that outperforms traditional methods in terms of accuracy and computational complexity. The dynamic selection of tuning parameters remains an open challenge, and future work will focus on developing adaptive algorithms to improve the performance of the proposed method.