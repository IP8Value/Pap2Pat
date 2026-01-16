# DESCRIPTION

## STATEMENT OF GOVERNMENT SPONSORED SUPPORT

This invention was made with government support under Grant No. [REDACTED] awarded by the National Institutes of Health and under Grant No. [REDACTED] awarded by the National Science Foundation. The government has certain rights in the invention.

## FIELD OF THE INVENTION

The present invention relates generally to brain–machine interfaces (BMIs) and, more specifically, to neural decoders for translating neural activity into control signals for external devices such as computer cursors, robotic limbs, or communication prostheses. In particular, the invention provides a novel recurrent neural network architecture—termed the Multiplicative Recurrent Neural Network (MRNN)—that is trained on large corpora of historical neural data collected across multiple days and recording conditions to achieve robust decoding performance even under significant and unexpected changes in neural signal quality, electrode availability, or recording stability. The MRNN decoder leverages both architectural innovations and training methodologies—including deliberate perturbation of input spike trains during training—to produce a fixed-parameter decoder that maintains high performance without requiring frequent recalibration, thereby addressing a critical barrier to the clinical translation of intracortical BMIs.

## BACKGROUND OF THE INVENTION

Brain–machine interfaces (BMIs) hold transformative potential for individuals with paralysis or limb loss by enabling direct control of assistive devices through recorded neural activity. A core component of any BMI system is the neural decoder—a computational algorithm that maps patterns of neural firing to intended movement kinematics, such as cursor velocity or robotic arm position. Over the past two decades, significant progress has been made in developing increasingly accurate decoders, with early systems relying on linear models such as the Kalman filter (KF) and later approaches incorporating nonlinear methods including artificial neural networks. Despite these advances, a persistent challenge limiting the real-world usability of BMIs is their susceptibility to performance degradation caused by changes in neural recording conditions over time.

Such changes arise from multiple sources. First, chronic neural recordings—typically obtained via implanted microelectrode arrays—are inherently nonstationary. Electrodes may lose signal due to tissue encapsulation, micromotion of the array relative to cortical tissue, or hardware failure, leading to sudden or gradual loss of informative neural channels. Second, even when all electrodes remain functional, the relationship between neural activity and intended movement (the “neural tuning”) can drift due to biological factors such as changes in arousal, attention, fatigue, or neural plasticity. Third, in clinical settings, patients may experience greater variability in recording conditions compared to laboratory animals due to larger physiological movements, medication effects, or environmental electromagnetic interference. These factors collectively mean that a decoder trained on data from one day may perform poorly—or fail entirely—on subsequent days, necessitating frequent recalibration sessions that are burdensome for users and impractical in unsupervised home environments.

Conventional approaches to mitigate this problem fall into two broad categories: adaptive decoding and signal stabilization. Adaptive decoders continuously update their parameters during closed-loop use based on inferred user intent, either through supervised learning (requiring explicit user feedback or known targets) or unsupervised methods (leveraging assumptions about movement statistics). While effective in some contexts, adaptation requires a baseline level of initial performance to bootstrap the learning process; if the decoder fails catastrophically due to, for example, sudden electrode loss, adaptation cannot recover control. Signal stabilization strategies aim to improve the longevity and reliability of neural recordings themselves—through better electrode materials, multiunit or local field potential decoding, or advanced spike sorting—but do not eliminate variability entirely and often trade off signal richness for stability.

An alternative, less explored strategy is to design a *fixed* decoder that is inherently robust to a wide range of recording conditions by training it on a diverse historical dataset encompassing many days of neural activity under varying conditions. This approach assumes that while no two recording sessions are identical, novel conditions often share statistical similarities with previously encountered ones. If a decoder can learn a rich repertoire of neural-to-kinematic mappings from this historical library, it may generalize effectively to new but related conditions without parameter updates. However, realizing this vision requires both a sufficiently expressive decoder architecture capable of modeling complex, nonlinear, and context-dependent neural dynamics, and a training methodology that explicitly encourages robustness to plausible perturbations.

Prior art in this domain has largely relied on linear models like the FIT-Kalman Filter (FIT-KF), which, despite innovations such as intention-based rotation and visual feedback incorporation, remain fundamentally limited by their linearity and inability to capture multiplicative interactions between neural inputs and internal states. Nonlinear decoders, such as standard recurrent neural networks (RNNs), have shown promise but often suffer from overfitting, poor generalization, or lack of explicit mechanisms to handle input distribution shifts. Moreover, training such models on heterogeneous, multi-day datasets presents unique challenges, including the need to balance learning across disparate conditions and prevent dominance by any single session’s statistics.

Thus, there remains a critical unmet need for a neural decoder that combines high performance under ideal conditions with robustness to clinically relevant recording condition changes—without sacrificing usability through frequent recalibration or reliance on fragile adaptation mechanisms. The present invention addresses this need through the development of the MRNN, a novel recurrent architecture featuring multiplicative input–state interactions, trained on extensive historical data augmented with biologically plausible perturbations, to achieve unprecedented levels of both accuracy and resilience in BMI control.

## SUMMARY OF THE INVENTION

The present invention provides a brain–machine interface (BMI) system comprising a Multiplicative Recurrent Neural Network (MRNN) decoder that translates neural activity into control signals for an external device with high accuracy and robustness to recording condition changes. The MRNN achieves this by employing a recurrent neural network architecture in which the input neural signals multiplicatively modulate the recurrent weight matrix, enabling the network to dynamically select appropriate internal dynamics based on the current neural context. This architectural feature allows the MRNN to model complex, nonlinear relationships between neural activity and movement kinematics that vary across recording sessions.

A key innovation of the invention is the training methodology, which utilizes a large corpus of historical neural data collected over many days—spanning months or even years—to expose the MRNN to a wide diversity of recording conditions. During training, the neural input spike trains are deliberately perturbed using a data augmentation technique that simulates plausible recording artifacts, such as global or electrode-specific firing rate modulations, including complete loss of signal from subsets of electrodes. This perturbation strategy trains the MRNN to be invariant to such changes, enhancing its robustness without requiring retraining or adaptation during closed-loop use.

The MRNN is trained offline using a supervised learning approach to minimize the error between decoded and actual movement kinematics (e.g., hand velocity or position) from historical reaching tasks. Once trained, the MRNN operates as a fixed-parameter decoder: its weights remain unchanged during real-time BMI operation, yet it maintains high performance even when faced with recording conditions not explicitly present in the training data, such as sudden electrode failures or long gaps between the most recent training data and the current session (“stale training data”).

In closed-loop experiments with non-human primates, the MRNN consistently outperformed state-of-the-art linear decoders (e.g., FIT-Kalman Filter) both in ideal conditions and under challenging scenarios. Specifically, the MRNN demonstrated superior robustness to the artificial dropping of the most informative electrodes, maintaining usable cursor control where conventional decoders failed. It also remained functional after training gaps of several months, whereas same-day or multi-day trained Kalman filters exhibited highly variable or catastrophic performance. Critically, this robustness did not come at the expense of peak performance; under favorable conditions, the MRNN enabled faster and more accurate cursor control than decoders trained specifically on the same day’s data.

The invention further includes methods for constructing the MRNN, initializing its parameters, concatenating neural trials for efficient training, and blending position and velocity outputs for stable cursor control. The system is designed to operate in real-time with millisecond latency, making it suitable for clinical BMI applications. By eliminating the need for frequent recalibration and providing consistent performance across diverse neural recording conditions, the MRNN represents a significant advance toward practical, user-friendly brain–machine interfaces for people with paralysis.

## DETAILED DESCRIPTION OF THE INVENTION

### MRNN Definition

The Multiplicative Recurrent Neural Network (MRNN) is a continuous-time recurrent neural network architecture specifically designed for decoding neural activity into movement kinematics in brain–machine interface (BMI) applications. Unlike standard recurrent neural networks (RNNs), in which external inputs additively influence the network’s internal dynamics, the MRNN employs a multiplicative interaction between the input neural signals and the recurrent connectivity. This design enables the network to dynamically reconfigure its internal processing based on the statistical properties of the incoming neural data, thereby capturing context-dependent neural-to-kinematic mappings that vary across recording sessions or conditions.

Formally, the MRNN is defined by an N-dimensional vector of activation variables, denoted **x**(t), which evolves continuously in time according to the following differential equation:

$$
\tau \frac{d\mathbf{x}(t)}{dt} = -\mathbf{x}(t) + \mathbf{J}_u(t) \mathbf{r}(t) + \mathbf{b}_x
$$

where:
- $\tau$ is a time constant (typically set in the physiologically relevant range of hundreds of milliseconds) that governs the temporal smoothing of the network’s dynamics;
- **r**(t) = tanh(**x**(t)) is the vector of “firing rates,” obtained by applying a hyperbolic tangent nonlinearity element-wise to the activation vector;
- **b**_x is an N-dimensional bias vector;
- **J**_u(t) is an N × N recurrent weight matrix that is *parameterized by the input neural signal* **u**(t) at time t.

The key innovation lies in the definition of **J**_u(t). Rather than being a fixed matrix, **J**_u(t) is constructed as a function of the E-dimensional input vector **u**(t), which represents binned spike counts from E electrodes at time t. To make this parameterization computationally tractable—especially given that **u**(t) is continuous-valued and high-dimensional—the invention employs a low-rank factorization of the input-dependent weight tensor. Specifically, **J**_u(t) is expressed as:

$$
\mathbf{J}_u(t) = \mathbf{J}_{xf} \cdot \text{diag}(\mathbf{J}_{fu} \mathbf{u}(t)) \cdot \mathbf{J}_{fx}
$$

where:
- **J**_{xf} is an N × F matrix,
- **J**_{fu} is an F × E matrix,
- **J**_{fx} is an F × N matrix,
- diag(**v**) denotes a diagonal matrix with the vector **v** along its diagonal,
- F is a tunable hyperparameter that controls the complexity of the input–recurrent interaction (in practice, F is often set equal to N).

This factorization reduces the number of trainable parameters from O(N²E) (which would be infeasible) to O(NF + FE + FN), making the model scalable to realistic BMI dimensions (e.g., E = 96 or 192 electrodes). The term **J**_{fu} **u**(t) computes F linear combinations of the input spike counts, which are then used to scale the rows of **J**_{fx} and columns of **J**_{xf} via the diagonal matrix. This multiplicative gating mechanism allows the network to selectively amplify or suppress specific recurrent pathways depending on the current neural input pattern, effectively implementing a form of dynamic, input-driven circuit reconfiguration.

The MRNN’s architecture thus endows it with the capacity to learn multiple, condition-specific decoding strategies within a single set of parameters. For instance, if the neural population exhibits different covariance structures on different days—as commonly observed in chronic recordings—the MRNN can learn to activate distinct internal dynamics that are optimal for each structure. This stands in contrast to linear decoders, which assume a fixed mapping, or standard RNNs, which lack an explicit mechanism for input-dependent modulation of recurrence.

### MRNN Output Definition

The output of the MRNN is a decoded estimate of movement kinematics, typically represented as a two-dimensional (2D) cursor velocity or position in the frontoparallel plane. To generate this output, the invention employs a linear readout layer that maps the network’s internal firing rates **r**(t) to the kinematic variables of interest. Specifically, the output **z**(t) is defined as:

$$
\mathbf{z}(t) = \mathbf{W}_o \mathbf{r}(t) + \mathbf{b}_z
$$

where:
- **W**_o is an M × N output weight matrix (with M = 2 for 2D kinematics),
- **b**_z is an M-dimensional output bias vector.

In the preferred embodiment, two separate MRNNs are trained in parallel: one to decode normalized hand *position* (**z**_pos(t)) and another to decode normalized hand *velocity* (**z**_vel(t)). This dual-decoder approach leverages the complementary strengths of position and velocity control—position provides absolute reference to prevent drift, while velocity enables smooth, responsive movement. During closed-loop BMI operation, these two outputs are blended to produce the final cursor position update, as detailed in the section “Controlling A BMI Cursor With MRNN Output.”

The training targets for the position and velocity decoders are derived from historical hand-reaching data. Position targets are the actual measured hand coordinates (e.g., from an infrared tracking system), normalized to the workspace dimensions. Velocity targets are computed numerically from the position data using central differences, ensuring consistency with the physical relationship between position and velocity. Both targets are synchronized with the neural input **u**(t), which consists of binned spike counts (e.g., 20–25 ms bins) from the same time period.

The output layer’s simplicity—being linear—is intentional. It ensures that the complex, nonlinear transformations necessary for robust decoding are handled by the recurrent core, while the readout remains interpretable and easy to regularize. Moreover, because **W**_o and **b**_z are trained jointly with the recurrent parameters, the entire network—including the readout—is optimized end-to-end to minimize kinematic reconstruction error.

### Network Construction for Cursor BMI Decoder

The construction of the MRNN for use as a cursor BMI decoder involves several key design choices that balance biological plausibility, computational efficiency, and decoding performance. First, the network size N (number of hidden units) is selected based on empirical validation and resource constraints. In the exemplary embodiments described herein, N = 100 for a monkey implanted with two 96-electrode arrays (E = 192 total inputs) and N = 50 for a monkey with a single array (E = 96 inputs). These sizes were found to provide sufficient representational capacity without excessive overfitting or computational burden.

Second, the input dimensionality E corresponds directly to the number of active electrodes in the implanted array(s). Each element of the input vector **u**(t) represents the spike count on a single electrode during a fixed time bin (e.g., 25 ms). Spike detection is performed using a standard threshold-crossing method (e.g., −4.5 times the root-mean-square noise on each channel), which captures multiunit activity without requiring time-consuming spike sorting. This approach is consistent with clinical BMI practices, where rapid deployment and long-term stability are prioritized over single-neuron resolution.

Third, the factorization rank F is set equal to N in the preferred implementation, allowing full expressivity of the multiplicative interaction while maintaining symmetry in the parameter matrices. Alternative values of F (e.g., F < N) could be used to reduce model complexity, but empirical results indicated that F = N yielded optimal performance.

Fourth, the time constant τ is set to 200 ms, reflecting the typical timescale of motor cortical dynamics and ensuring that the MRNN’s outputs are smooth and free of high-frequency jitter that could impair cursor control. This value can be adjusted based on the specific neural population or task requirements, but values in the range of 100–500 ms are generally suitable.

Finally, the network is implemented in software that supports real-time execution on embedded systems (e.g., xPC Target platform). The continuous-time dynamics are discretized using a forward Euler method with a step size matching the neural bin width (e.g., 25 ms), ensuring numerical stability and compatibility with the BMI’s timing constraints. The resulting discrete-time update rule is:

$$
\mathbf{x}[t+1] = \left(1 - \frac{\Delta t}{\tau}\right) \mathbf{x}[t] + \frac{\Delta t}{\tau} \left( \mathbf{J}_u[t] \mathbf{r}[t] + \mathbf{b}_x \right)
$$

where Δt is the bin width. This formulation allows the MRNN to be integrated seamlessly into existing BMI pipelines that operate on binned spike data.

### MRNN Initialization

Proper initialization of the MRNN’s parameters is critical to ensure stable training dynamics and convergence to a high-performance solution. The invention employs a principled initialization scheme based on random Gaussian distributions with variances scaled to maintain balanced activity propagation through the network.

Specifically, the non-zero elements of the factorized weight matrices are initialized as follows:
- Elements of **J**_{xf} are drawn independently from a Gaussian distribution with zero mean and variance $g_{xf}/F$, where $g_{xf} = 1.0$;
- Elements of **J**_{fu} are drawn from a Gaussian with zero mean and variance $g_{fu}/E$, where $g_{fu} = 1.0$;
- Elements of **J**_{fx} are drawn from a Gaussian with zero mean and variance $g_{fx}/N$, where $g_{fx} = 1.0$.

These scaling factors follow established practices in reservoir computing and echo-state networks, ensuring that the spectral radius of the effective recurrent matrix remains near unity, which promotes rich, stable dynamics without explosive growth or vanishing gradients.

The output weight matrix **W**_o is initialized to zero, allowing the readout layer to start from a neutral state and learn incrementally from the recurrent representations. Similarly, the bias vectors **b**_x and **b**_z are initialized to zero vectors, avoiding any initial asymmetry in the network’s dynamics or output.

This initialization strategy has been empirically validated to produce consistent training outcomes across multiple random seeds and dataset configurations. It also facilitates the use of second-order optimization methods (e.g., Hessian-Free optimization), which rely on well-conditioned initial parameter distributions to efficiently navigate the loss landscape.

### Concatenating Neural Trials for Seeding the MRNN During Training

Training the MRNN on historical neural data requires careful handling of trial boundaries to ensure that the network’s internal state is properly initialized at the start of each movement epoch. In natural reaching behavior, neural activity exhibits preparatory dynamics before movement onset, which can influence the subsequent movement-related activity. To account for this, the invention employs a trial concatenation strategy that provides the MRNN with a “warm start” by seeding its hidden state with neural activity from preceding trials.

Specifically, five consecutive actual monkey-reaching trials from the training dataset are concatenated end-to-end to form a single “MRNN training trial.” The first two actual trials in this concatenated sequence are used exclusively for seeding: the MRNN processes these trials to build up an internal state that reflects the ongoing neural context, but the corresponding kinematic targets are excluded from the loss computation. Only the final three actual trials are used for supervised learning, with their kinematic targets contributing to the error signal that drives parameter updates.

This approach serves two purposes. First, it mitigates the problem of arbitrary initial state assumptions (e.g., **x**(0) = 0), which can introduce artifacts at the beginning of each trial and degrade decoding accuracy during the critical movement initiation phase. By allowing the network to “ramp up” using real neural data, the MRNN learns to transition smoothly from preparatory to movement-related dynamics. Second, it increases the effective length of training sequences, which improves the credit assignment capability of backpropagation-through-time (BPTT) and helps the network learn longer-timescale dependencies in the neural data.

To maximize data utilization, the invention employs a sliding window over the dataset: after forming one MRNN training trial from actual trials 1–5, the next is formed from trials 2–6, then 3–7, and so on. This ensures that nearly all actual trials (except the first two of each recording day) contribute to the learning phase, while still providing adequate seeding context. The sliding window also introduces variability in the seeding conditions, which may further enhance robustness by exposing the network to diverse preparatory states.

### Perturbing the Neural Input During Training

A cornerstone of the MRNN’s robustness is the deliberate perturbation of neural input spike trains during training—a form of data augmentation tailored to the specific nonstationarities encountered in BMI applications. This technique simulates plausible recording condition changes, such as global firing rate modulations (e.g., due to arousal or array micromotion) or electrode-specific signal loss (e.g., due to hardware failure), thereby training the MRNN to be invariant to such perturbations.

The perturbation procedure operates on the concatenated input spike trains for each MRNN training trial. For a given electrode c in trial j, let $s_{c,j}$ denote the total number of observed spikes across all time bins in that trial. This count is perturbed according to:

$$
\tilde{s}_{c,j} = s_{c,j} \cdot \eta_j \cdot \eta_c
$$

where:
- $\eta_j \sim \mathcal{N}(1, \sigma_{\text{trial}}^2)$ is a trial-wide scaling factor modeling global firing rate changes,
- $\eta_c \sim \mathcal{N}(1, \sigma_{\text{electrode}}^2)$ is an electrode-specific scaling factor modeling local perturbations.

In the exemplary embodiment, $\sigma_{\text{trial}} = 0.2$ and $\sigma_{\text{electrode}} = 0.3$, though these values can be tuned based on the expected variability in the recording setup. If $\tilde{s}_{c,j}$ falls outside the valid range [0, $s_{\text{max}}$] (where $s_{\text{max}}$ is a large upper bound), it is resampled until valid. The perturbed spike count $\tilde{s}_{c,j}$ is then realized by randomly adding or removing spikes from the original time bins: if $\tilde{s}_{c,j} > s_{c,j}$, $(\tilde{s}_{c,j} - s_{c,j})$ spikes are added to randomly selected bins; if $\tilde{s}_{c,j} < s_{c,j}$, $(s_{c,j} - \tilde{s}_{c,j})$ spikes are removed from bins that originally contained spikes.

Critically, this perturbation is applied *anew* at every iteration of the optimization algorithm (e.g., Hessian-Free training). This means that the MRNN never sees the exact same perturbed input twice, forcing it to learn general strategies for handling input variability rather than memorizing specific augmented examples. During closed-loop BMI operation, however, the MRNN receives unperturbed, real-time spike counts—ensuring that the augmentation serves only as a training regularizer, not as a distortion of the actual control signal.

This approach is conceptually related to dropout regularization in deep learning but is specifically designed to address BMI-relevant nonstationarities. Empirical results demonstrate that MRNNs trained with this perturbation strategy exhibit significantly improved robustness to electrode dropping and stale training data, as detailed in the experimental sections of the research paper.

### Using Many Days Training Data

The MRNN’s ability to generalize across recording conditions is fundamentally enabled by training on a large corpus of historical neural data spanning many days—often months or years of a subject’s research career. This multi-day training strategy exploits the observation that, despite day-to-day variability, neural correlates of reaching exhibit statistical similarities across sessions, particularly when recorded from the same cortical areas using stable implant technology.

In practice, the training dataset comprises all available hand-reaching trials from a predetermined set of recording days. For example, in the exemplary embodiments, MRNNs were trained on up to 154 days (monkey R) or 250 days (monkey L) of data, encompassing over 75,000 point-to-point reaches. Each day’s data includes trials from standardized behavioral tasks (e.g., Radial 8 Task with 8–12 cm target distances), ensuring consistency in the kinematic labels used for supervision.

During training, minibatches are constructed by randomly sampling a small number of trials from *each* day in the dataset. This balanced sampling strategy prevents the MRNN from overfitting to any single day’s statistics and ensures that the gradient updates reflect the full diversity of recording conditions. It also mimics the real-world scenario where a BMI user might encounter any of the previously experienced neural states on a given day.

The benefits of large-scale, multi-day training are twofold. First, it exposes the MRNN to a wide range of neural tuning properties, covariance structures, and signal-to-noise ratios, enabling it to learn a comprehensive library of neural-to-kinematic mappings. Second, it provides the statistical power necessary to train the MRNN’s large number of parameters without overfitting—particularly when combined with input perturbation and other regularization techniques.

Offline analyses confirm that decode accuracy (measured as r² between true and decoded velocity) increases monotonically with the number of training days, with no evidence of a performance–robustness trade-off. Moreover, the MRNN’s performance on “hard” days (those that challenge same-day-trained decoders) improves disproportionately, indicating that the additional data specifically enhances generalization to difficult conditions.

### Network Output

As previously noted, the MRNN produces two primary outputs: a decoded position signal **z**_pos(t) and a decoded velocity signal **z**_vel(t). These outputs are generated by two independent MRNN instances that share the same architectural specifications but are trained on different kinematic targets. The position decoder is trained to minimize the squared error between **z**_pos(t) and the normalized hand position coordinates (x, y), while the velocity decoder minimizes error relative to the numerically differentiated hand velocities.

Both outputs are normalized to the range [−1, 1] to facilitate stable training and consistent blending during closed-loop control. The normalization is performed relative to the workspace dimensions used during training (e.g., a 20 × 20 cm square), ensuring that the decoded signals are interpretable in physical units when scaled back during BMI operation.

The dual-output design addresses a fundamental limitation of pure velocity decoders: accumulated integration error can cause cursor drift over time, requiring frequent re-centering or position resets. By incorporating a position signal—even a weak one—the MRNN can correct for this drift and maintain stable cursor placement. Conversely, pure position decoders can be jerky or laggy; the velocity signal provides the smooth, responsive control necessary for rapid target acquisition.

During offline performance evaluation (e.g., for comparing against Kalman filters), the MRNN’s output is configured to be velocity-only (by setting the blending parameter β = 1, as described below), ensuring a fair comparison with velocity-decoding baselines. However, for closed-loop BMI use, the blended output is essential for optimal user experience.

### Training and Running the Networks

The MRNN is trained offline using the Hessian-Free (HF) optimization algorithm, a second-order method well-suited for recurrent neural networks due to its ability to handle long-range dependencies and avoid gradient vanishing/explosion. The HF algorithm uses backpropagation-through-time (BPTT) to compute exact gradients of the loss function (mean squared error between decoded and true kinematics) with respect to all network parameters: {**J**_{xf}, **J**_{fu}, **J**_{fx}, **b**_x, **W**_o, **b**_z}.

Key hyperparameters for HF training include:
- Minibatch size: set to one-fifth of the total number of MRNN training trials to balance gradient accuracy and computational efficiency;
- Initial lambda (Tikhonov regularization): set to 0.1 to stabilize early optimization steps;
- Maximum conjugate-gradient iterations per HF step: set to 50 to ensure sufficient curvature approximation.

Training proceeds for 200 HF steps, with a snapshot of the network saved every 10 steps. The final model is selected as the snapshot with the lowest cross-validation error on a held-out subset of the training data, preventing overfitting.

Once trained, the MRNN is compiled into a real-time executable and deployed on the BMI’s embedded control system. At each decode time step (e.g., every 25 ms), the system:
1. Bins incoming spike counts from all electrodes into the input vector **u**(t);
2. Updates the MRNN’s hidden state **x**(t) using the discretized dynamics;
3. Computes firing rates **r**(t) = tanh(**x**(t));
4. Generates position and velocity outputs via the readout layers;
5. Blends these outputs to update the cursor position (see below).

The entire pipeline operates with end-to-end latency of approximately 12 ms (including neural acquisition and rendering), well within the requirements for responsive BMI control. Critically, the MRNN’s parameters remain fixed during closed-loop use—it is a *static* decoder whose robustness arises from its training regimen and architecture, not from online adaptation. This distinguishes it from adaptive decoders that continuously update their weights, and ensures predictable, stable performance even in the face of sudden recording changes.