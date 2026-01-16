# DESCRIPTION

## FEDERAL FUNDING

The development of the Proportional Navigation with Line of Sight Curvature (PN-LOSC) guidance system was supported by federal funding from the Department of Defense, specifically through grants awarded by the Air Force Research Laboratory (AFRL) and the Defense Advanced Research Projects Agency (DARPA). The funding was instrumental in advancing the research and development of this innovative guidance technology, which aims to improve the accuracy and efficiency of missile systems against highly maneuvering targets.

## BACKGROUND

Proportional Navigation (PN) is a widely used guidance law in missile systems, designed to minimize the zero-effort miss (ZEM), which is the predicted miss distance if neither the missile nor the target accelerates for the remainder of the engagement. However, PN is susceptible to increased miss distances when the target performs maneuvers. To address this issue, Augmented Proportional Navigation (APN) was introduced, which incorporates an additional term based on the estimated target acceleration. While APN improves performance against step acceleration maneuvers, it requires accurate estimation of target acceleration, which can be challenging and prone to divergence, leading to large miss distances.

To overcome these limitations, there has been significant research into guidance laws that do not require estimates of target acceleration. Various approaches have been explored, including the use of sliding mode theory, geometric guidance, and reinforcement learning (RL). However, these methods often have limitations in terms of robustness and adaptability to a wide range of target maneuvers.

In this context, we propose a novel approach to improving the performance of any guidance law against maneuvering targets. Specifically, we optimize a line of sight (LOS) curvature policy \( c \) that maps navigation system outputs to a Euler 321 attitude parameterization \( \Delta \theta_{\text{LOSC}} \). This policy is then used to rotate the observed LOS unit vector, allowing for arbitrary curving of the LOS during an engagement. The LOS curvature policy is parameterized as a deep neural network with a recurrent layer and optimized using reinforcement meta-learning (meta-RL).

Reinforcement learning (RL) has been effectively applied to optimize integrated and adaptive guidance, navigation, and control (GNC) systems. Applications of RL in GNC include asteroid close proximity operations, planetary landing, exoatmospheric intercept, endoatmospheric intercept, and hypersonic vehicle guidance. In the RL framework, an agent learns to complete a task through episodic simulated experience with an environment. The policy is implemented as a deep neural network that maps observations to actions, and in our work, we use a customized version of proximal policy optimization (PPO) with a recurrent layer to allow actions to be generated using the history of observations.

## SUMMARY

The invention relates to a novel guidance system, Proportional Navigation with Line of Sight Curvature (PN-LOSC), which improves the performance of missile systems against highly maneuvering targets. The PN-LOSC guidance system optimizes a line of sight (LOS) curvature policy using reinforcement meta-learning (meta-RL). This policy maps navigation system outputs to a Euler 321 attitude parameterization, which is then used to rotate the observed LOS unit vector, allowing for arbitrary curving of the LOS during an engagement.

The key features of the PN-LOSC guidance system include:
1. **Line of Sight Curvature Policy**: A deep neural network with a recurrent layer that maps navigation system outputs to a Euler 321 attitude parameterization, enabling the arbitrary curving of the LOS.
2. **Reinforcement Meta-Learning**: The policy is optimized using a customized version of proximal policy optimization (PPO) with a recurrent layer, allowing the policy to infer properties of target maneuvers and adapt to novel environments.
3. **Improved Performance**: The PN-LOSC guidance system demonstrates improved accuracy and requires less control effort compared to traditional proportional navigation (PN) and augmented proportional navigation (APN) guidance laws, without requiring an estimate of target acceleration.
4. **Robustness to Radome Refraction**: The PN-LOSC guidance system is more robust to radome refraction, which can cause false indications of target motion, further enhancing its performance.

The PN-LOSC guidance system is applicable to a wide range of engagement scenarios, including high-altitude and low-speed targets, and can be integrated with any guidance law that takes inputs derived from the LOS unit vector pointing from the missile to the target. The invention represents a significant advancement in missile guidance technology, offering improved accuracy and efficiency against highly maneuvering targets.

## DETAILED DESCRIPTION

### Engagement Scenarios

The PN-LOSC guidance system is designed for a skewed head-on engagement scenario, where the missile is launched to intercept a highly maneuverable target. The target's initial velocity vector is randomly generated within a cone with an axis along the relative position vector \( \mathbf{r}_{TM} \) and a half-apex angle \( \alpha_{v_T} \). The engagement scenario is defined by the relative position vector \( \mathbf{r}_{TM} \), relative velocity vector \( \mathbf{v}_{TM} \), and the target's initial velocity vector \( \mathbf{v}_T \).

The target can perform a variety of maneuvers, including bang-bang, weave, and jinking, with varying acceleration levels and random start times, durations, and switching times. The maximum target acceleration is a function of dynamic pressure, reflecting the use of aerodynamic control surfaces. The engagement scenario parameters, including the initial conditions and vehicle parameters, are randomly generated within specified ranges to ensure robustness and adaptability.

### Radome and Seeker Model

The radome and seeker model is crucial for simulating the effects of radome refraction on the LOS measurement. The look angle is defined as the angle between the ground truth inertial frame LOS vector \( \mathbf{\hat{r}}_{TM} \) and the missile's velocity unit vector \( \mathbf{\hat{v}}_M \). The radome refraction angle \( \beta' \) is the angle between the ground truth LOS vector and the apparent LOS vector \( \mathbf{\hat{r}}_{\text{app}} \). The radome refraction angle is a function of the look angle and is modeled using a symmetrical radome with azimuthal and elevation refraction errors.

The radome refraction model includes a first-order low-pass noise filter to simulate the parasitic attitude loop, which can reduce guidance system performance. The seeker is assumed to be a strap-down implementation, and seeker gimbal lag is not modeled.

### Line of Sight Curvature

The LOS curvature policy \( c \) maps observations \( o \) to actions \( u \), which are interpreted as a Euler 321 attitude parameterization \( \Delta \theta_{\text{LOSC}} \). The action \( u \) is scaled and used to construct a direction cosine matrix (DCM) \( C(\Delta \theta_{\text{LOSC}}) \), which rotates the observed LOS unit vector \( \mathbf{\hat{r}}_{TM} \) to produce the shaped LOS direction vector \( \mathbf{\hat{r}}_{\text{LOSC}} \). By varying \( \Delta \theta_{\text{LOSC}} \) during the engagement, the LOS curvature policy can arbitrarily curve the LOS.

The LOS curvature policy is implemented as a deep recurrent neural network and optimized using reinforcement meta-learning. The recurrent layer allows the policy to infer properties of the target maneuvers and generate actions based on the history of observations.

### Guidance Law

The PN-LOSC guidance system combines the true proportional navigation (TPN) guidance law with the LOS curvature policy. The TPN guidance law is given by:

\[
\mathbf{a}_M = N \frac{\mathbf{\dot{\lambda}} \times \mathbf{v}_M}{\|\mathbf{v}_M\|}
\]

where \( \mathbf{a}_M \) is the commanded acceleration, \( N \) is the navigation gain, \( \mathbf{\dot{\lambda}} \) is the LOS rotation rate, and \( \mathbf{v}_M \) is the missile velocity vector. The APN benchmark uses a modified version of the TPN guidance law, incorporating an additional term based on the estimated target acceleration.

In the PN-LOSC guidance system, the LOS curvature policy is applied to the LOS unit vector before it is used in the guidance law. The shaped LOS direction vector \( \mathbf{\hat{r}}_{\text{LOSC}} \) is used to compute the LOS rotation rate \( \mathbf{\dot{\lambda}}_{\text{LOSC}} \), which is then used in the TPN guidance law. The PN-LOSC guidance law is given by:

\[
\mathbf{a}_M = N \frac{\mathbf{\dot{\lambda}}_{\text{LOSC}} \times \mathbf{v}_M}{\|\mathbf{v}_M\|}
\]

### Reinforcement Learning Framework

The reinforcement learning (RL) framework is used to optimize the LOS curvature policy. In the RL framework, an agent learns to complete a task through episodic interaction with an environment. The environment initializes an episode by generating a ground truth state, which is mapped to an observation and passed to the agent. The agent uses the observation to generate an action, which is sent to the environment. The environment then uses the action and the current state to generate the next state and a scalar reward signal. The process repeats until the episode terminates, and the trajectories collected over multiple episodes are used to update the policy and value functions.

The PN-LOSC guidance system uses proximal policy optimization (PPO) with a recurrent layer to optimize the LOS curvature policy. The PPO algorithm approximates the Trust Region Policy Optimization method by using a clipped objective function to ensure that the policy does not change drastically between updates. The objective function is given by:

\[
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
\]

where \( r_t(\theta) \) is the probability ratio, \( \hat{A}_t \) is the advantage function, and \( \epsilon \) is the clipping parameter. The value function is learned using a mean squared error loss function.

### RL Problem Formulation

The RL problem formulation for the PN-LOSC guidance system involves optimizing the LOS curvature policy over a range of engagement scenarios. The agent observation \( o \) includes the relative position vector \( \mathbf{r}_{TM} \), relative velocity vector \( \mathbf{v}_{TM} \), and other relevant navigation system outputs. The agent action \( u \) is interpreted as the Euler 321 attitude parameterization \( \Delta \theta_{\text{LOSC}} \).

The reward function is designed to encourage the agent to curve the LOS only when it results in higher terminal rewards. The reward function is given by:

\[
R_t = -U \|\Delta \theta_{\text{LOSC}}\| + V \exp(-\|\mathbf{r}_{TM}\| / A_{\text{lim}}) + W \exp(-\|\mathbf{r}_{TM}\| / B_{\text{lim}})
\]

where \( U \), \( V \), and \( W \) are reward coefficients, \( A_{\text{lim}} \) and \( B_{\text{lim}} \) are distance limits, and \( \|\mathbf{r}_{TM}\| \) is the relative distance between the missile and the target. The reward function penalizes the LOS curvature and encourages the agent to minimize the miss distance.

### Experiments

The PN-LOSC guidance system was optimized and tested over a wide range of engagement scenarios, including different levels of maximum target acceleration capability and target aerodynamic drag models. The performance of the PN-LOSC guidance system was compared to traditional proportional navigation (PN) and augmented proportional navigation (APN) guidance laws.

The results demonstrate that the PN-LOSC guidance system achieves improved accuracy with less control effort compared to both PN and APN against highly maneuvering targets. The performance advantage of the PN-LOSC guidance system is attributed to two factors: increased robustness to radome refraction and a more intelligent response to target maneuvers.

### Conclusion

The Proportional Navigation with Line of Sight Curvature (PN-LOSC) guidance system represents a significant advancement in missile guidance technology. By optimizing a line of sight curvature policy using reinforcement meta-learning, the PN-LOSC guidance system improves the accuracy and efficiency of missile systems against highly maneuvering targets. The system is robust to radome refraction and does not require an estimate of target acceleration, making it a valuable addition to modern missile guidance systems. Future work will investigate the use of line of sight curvature in higher fidelity six degrees-of-freedom models and explore additional applications of the technology.