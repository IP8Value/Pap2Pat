# DESCRIPTION

## FEDERAL FUNDING

- none

This invention was not made with any federal government support, and no rights are granted to the United States Government under this patent application.

## BACKGROUND

- describe limitations of existing missile guidance methods

Conventional missile guidance systems have long relied on proportional navigation (PN) as a foundational strategy for intercepting moving targets. While PN is computationally simple and effective against non-maneuvering or slowly maneuvering targets, its performance degrades significantly when confronted with highly agile adversaries capable of rapid acceleration changes. The core limitation of PN lies in its assumption that both the missile and target will maintain constant velocity for the remainder of the engagement—a premise invalidated by modern maneuvering threats such as supersonic cruise missiles or evasive aircraft. When a target executes abrupt lateral accelerations orthogonal to its velocity vector, the line-of-sight (LOS) rotation rate increases unpredictably, causing the PN-guided missile to over- or under-correct, often resulting in large miss distances. This vulnerability has motivated decades of research into more robust guidance laws, yet practical implementations remain constrained by assumptions about target behavior, sensor fidelity, or computational tractability.

- describe proportional navigation

Proportional navigation operates by commanding missile acceleration proportional to the LOS rotation rate, scaled by a navigation gain typically set between three and five. Mathematically, the commanded acceleration is perpendicular to the missile’s velocity vector and aligned with the direction of LOS rate change. This law can be interpreted as minimizing the zero-effort miss (ZEM)—the predicted miss distance if neither vehicle accelerates further. While elegant in theory, ZEM minimization fails when the target actively maneuvers, as the future trajectory deviates from the inertial extrapolation assumed by PN. Augmented proportional navigation (APN) attempts to address this by incorporating an estimate of target acceleration into the guidance command. However, accurate real-time estimation of arbitrary target maneuvers remains elusive; estimation errors can compound, leading to destabilizing feedback and degraded performance. Moreover, APN is provably optimal only for step-acceleration maneuvers, offering limited gains against complex evasion patterns like weaving or jinking.

- describe need for improved missile guidance

The increasing sophistication of aerial threats—exemplified by hypersonic vehicles, stealthy cruise missiles, and autonomous drones—demands guidance systems that adapt dynamically without relying on fragile target state estimates. Existing alternatives, such as switched bias proportional navigation (SBPN) or geometric guidance based on the Frenet-Serret frame, either impose restrictive assumptions (e.g., known target acceleration vectors) or demonstrate narrow effectiveness across limited maneuver sets. Furthermore, real-world effects like radome refraction—where aerodynamic shaping of the missile nose distorts seeker measurements based on look angle—introduce parasitic feedback loops that degrade guidance accuracy, particularly at high off-boresight angles. A guidance architecture that inherently compensates for both target agility and sensor imperfections, without explicit modeling of either, represents a critical unmet need. Such a system must operate within the constraints of onboard computation, maintain stability across diverse engagement geometries, and reduce control effort to preserve missile energy and extend kinematic reach.

## SUMMARY

- introduce curvature parameterization

The present invention introduces a novel line-of-sight (LOS) curvature parameterization that dynamically reshapes the observed LOS vector prior to its use in any conventional guidance law. Rather than treating the LOS as a fixed geometric quantity derived directly from seeker measurements, the invention applies a time-varying rotational transformation to the LOS unit vector, effectively curving the apparent trajectory of the target as perceived by the guidance system. This curvature is not heuristic but is governed by a policy function that maps real-time navigation observations—including relative position, velocity, LOS rate, and historical context—to a three-dimensional Euler attitude parameterization. The resulting curved LOS vector serves as a synthetic input to the underlying guidance law, enabling implicit compensation for target maneuvers and sensor distortions without requiring explicit estimation of either.

- describe application of deep learning network

The curvature policy is implemented as a deep neural network with recurrent layers, trained via reinforcement meta-learning to optimize interception performance across a broad distribution of engagement scenarios. The network ingests a feature vector comprising normalized LOS angles, closing velocity, relative range, and filtered derivatives, and outputs a compact attitude representation that defines a direction cosine matrix (DCM). This DCM rotates the true LOS vector into a shaped counterpart, which is then fed into a standard proportional navigation (or other) guidance law. By embedding memory through recurrent units, the policy infers latent characteristics of the target’s evasion strategy—such as weave frequency or jink timing—from temporal patterns in the observation stream. Training occurs in a simulated environment that randomizes initial conditions, target maneuver types (bang-bang, weave, jink), aerodynamic drag, radome refraction parameters, and dynamic pressure effects, ensuring robustness to real-world variability. The resulting guidance system, termed PN-LOSC (Proportional Navigation with Line-of-Sight Curvature), demonstrates superior accuracy and reduced control effort compared to both classical PN and augmented PN, particularly against high-g, unpredictable threats.

## DETAILED DESCRIPTION

- introduce navigation system

The navigation system of the missile integrates data from an onboard seeker, inertial measurement unit (IMU), and potentially external sources to generate a real-time estimate of the relative state between the missile and target. Central to this system is the line-of-sight (LOS) unit vector, denoted **ρ**, which points from the missile’s current position to the target’s estimated position in an inertial reference frame. This vector is continuously updated at a fixed sampling interval (e.g., 20 ms) and serves as the primary input to the guidance law. Additional navigation outputs include the relative position vector **r**<sub>TM</sub>, relative velocity **v**<sub>TM</sub>, missile velocity **v**<sub>M</sub>, and derived quantities such as closing velocity and look angle.

- describe line-of-sight unit vector

The LOS unit vector **ρ** is computed as **r**<sub>TM</sub> normalized by its Euclidean norm, i.e., **ρ** = **r**<sub>TM</sub>/‖**r**<sub>TM</sub>‖. It encapsulates the instantaneous bearing to the target and is fundamental to all homing guidance strategies. In ideal conditions, **ρ** evolves smoothly as the engagement progresses; however, in practice, it is corrupted by sensor noise, radome-induced refraction, and the target’s own maneuvers, which induce rapid rotations in **ρ**.

- derive line-of-sight rotation rate

The LOS rotation rate **ω** is derived as the cross product of **ρ** and its time derivative: **ω** = **ρ** × d**ρ**/dt. This vector quantifies the angular velocity of the LOS in three-dimensional space and is directly proportional to the commanded acceleration in proportional navigation guidance laws. Accurate computation of **ω** is essential, though challenging in discrete-time implementations due to noise amplification in numerical differentiation.

- introduce line-of-sight bias network

To overcome the limitations of raw LOS inputs, the invention employs a line-of-sight bias network—implemented as a deep recurrent neural network—that generates a corrective attitude adjustment. This network does not estimate target states but instead learns a mapping from observable engagement features to a curvature-inducing rotation.

- describe curvature parameterization

The curvature parameterization is defined by a 3-2-1 Euler angle triplet **θ**<sub>LOSC</sub> = [φ, θ, ψ]<sup>T</sup>, which is output by the neural policy and scaled to a bounded range (e.g., ±2 radians). These angles represent sequential rotations about the body-fixed axes and fully specify an orientation in SO(3).

- apply curvature parameterization to line-of-sight unit vector

The Euler angles **θ**<sub>LOSC</sub> are converted into a direction cosine matrix **C**(**θ**<sub>LOSC</sub>) using standard aerospace conventions. The shaped LOS vector is then computed as **ρ**<sub>LOSC</sub> = **C**(**θ**<sub>LOSC</sub>) **ρ**, effectively rotating the original LOS by the policy-determined attitude.

- output curved line-of-sight unit vector

The resulting **ρ**<sub>LOSC</sub> is a smoothed, anticipatory version of the true LOS, designed to preemptively counteract expected target motion and sensor artifacts. It replaces **ρ** as the input to the guidance law’s LOS rate calculation.

- describe reinforcement learning framework

The policy is trained within a reinforcement learning (RL) framework where an agent interacts with a high-fidelity simulation environment. Each interaction episode corresponds to a complete missile-target engagement, terminating upon intercept or miss.

- introduce agent

The agent embodies the LOS curvature policy and value function, both parameterized by neural networks. It observes the engagement state and selects actions (Euler angles) to minimize miss distance while conserving control energy.

- describe environment

The environment simulates six-degree-of-freedom dynamics (reduced to three for training efficiency), including variable aerodynamic drag, dynamic pressure-dependent acceleration limits, radome refraction, and stochastic target maneuvers.

- introduce engagement scenario generator

An engagement scenario generator randomizes initial conditions—range, aspect angle, heading error, altitude, velocities—according to operational envelopes, ensuring policy generalization across realistic combat geometries.

- describe dynamics model

The dynamics model integrates equations of motion for both missile and target, accounting for thrust, drag, gravity, and commanded accelerations. Missile acceleration is limited by dynamic pressure and structural load constraints (e.g., 40g max).

- introduce ensemble of target behavior models

The target employs one of three randomized maneuver types per episode: bang-bang (maximum lateral acceleration switching sign), sinusoidal weave, or jinking (random-duration pulses). Acceleration magnitude is sampled up to the target’s dynamic limit.

- describe radome model

The radome model introduces look-angle-dependent refraction: the apparent LOS is perturbed by azimuthal and elevation errors proportional to the cosine of the look angle, mimicking real-world optical distortion.

- introduce guidance law

The underlying guidance law is true proportional navigation (TPN), which commands acceleration perpendicular to missile velocity and proportional to LOS rate. The invention modifies TPN by replacing the raw LOS with **ρ**<sub>LOSC</sub>.

- describe reward function

The reward function penalizes LOS curvature magnitude (to discourage unnecessary shaping), rewards terminal proximity (inverse miss distance), and includes a small penalty for excessive control effort.

- introduce trajectory accumulation module

Collected trajectories (observation-action-reward sequences) are stored in a buffer and used to compute policy gradients via batch updates, ensuring stable learning.

- update policy function

The policy network is updated using proximal policy optimization (PPO), which maximizes a clipped surrogate objective to prevent destructive policy updates.

- update value function

A separate value network estimates expected cumulative reward and is trained via mean-squared-error regression against empirical returns.

- describe optimization

Optimization proceeds over 90,000 episodes, with hyperparameters tuned to maintain KL divergence near 0.001, ensuring gradual policy evolution.

- introduce terminal rewards

Terminal rewards dominate the learning signal, providing strong feedback on interception success, scaled inversely with final miss distance.

- describe curvature penalty

A curvature penalty term in the reward discourages aggressive LOS shaping unless justified by improved terminal outcomes, promoting energy-efficient guidance.

- introduce hyperparameters

Key hyperparameters include discount factor (0.95 for step rewards, 0.995 for terminal), PPO clipping epsilon (0.2), and neural network architecture (4 layers, 128 units, GRU recurrence).

- describe value function

The value function approximates the expected return from any state, serving as a baseline to reduce variance in policy gradient estimates.

- introduce cost function

The value network minimizes a Huber loss between predicted and actual returns, enhancing robustness to outlier rewards.

- describe advantage function

The advantage function, computed as return minus value estimate, indicates whether an action was better or worse than average, guiding policy improvement.

- introduce policy gradient method

Policy gradients are estimated using Generalized Advantage Estimation (GAE), balancing bias and variance in credit assignment.

- describe proximal policy optimization

PPO stabilizes training by clipping probability ratios, preventing large policy updates that could collapse performance.

- introduce clipped objective function

The clipped objective takes the minimum of the unclipped and clipped surrogate objectives, enforcing conservative updates.

- describe optimization objective

The overall objective is to maximize expected discounted reward while minimizing control usage and LOS distortion.

- implement policy function

The policy network uses tanh activations and a GRU layer to encode temporal dependencies, enabling inference of maneuver patterns.

- implement value function

The value network shares the same recurrent architecture, allowing joint adaptation to evolving engagement dynamics.

- introduce recurrent network layers

Gated Recurrent Units (GRUs) maintain a hidden state that accumulates evidence about target behavior, facilitating adaptive responses.

- describe inference of target maneuvers

Through recurrent processing, the policy learns to distinguish weave frequencies or jink timings from LOS history, adjusting curvature accordingly.

- describe adaptation to target behavior models

Meta-learning over diverse maneuver ensembles ensures the policy generalizes to unseen evasion tactics without retraining.

- describe application of optimized policy function

During deployment, the trained policy runs in real-time, transforming raw LOS into shaped LOS at each guidance cycle.

- describe performance of disclosed method

Experimental results show PN-LOSC reduces median miss distance by 30–50% versus PN and APN against 30g targets, with 20% less integrated control effort, demonstrating superior robustness and efficiency.