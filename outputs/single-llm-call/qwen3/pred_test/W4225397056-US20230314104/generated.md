# DESCRIPTION

## FEDERAL FUNDING

- none

## BACKGROUND

- describe limitations of existing missile guidance methods  
Existing missile guidance systems relying on proportional navigation (PN) exhibit significant performance degradation when confronted with highly maneuvering targets. These systems assume a constant or slowly varying line-of-sight rotation rate, which fails to account for abrupt, unpredictable accelerations exhibited by modern aerial threats. As a result, miss distances increase substantially under conditions where the target executes weaving, jinking, or bang-bang maneuvers at acceleration levels approaching that of the missile. While augmented proportional navigation (APN) attempts to mitigate this issue by incorporating an estimated target acceleration term, its efficacy is critically dependent on the accuracy and stability of that estimate. In practice, estimation errors—arising from sensor noise, dynamic pressure variations, or unmodeled aerodynamic effects—can cause the acceleration term to diverge, leading to oscillatory or destabilizing control inputs that worsen miss distance. Furthermore, APN requires precise knowledge of target dynamics, which is often unavailable in real-world engagements. Other geometric approaches have been proposed, but they either assume prior knowledge of the target’s acceleration profile or are restricted to low-speed, non-agile targets, rendering them inapplicable to high-performance air-to-air scenarios. The persistent challenge lies in designing a guidance law that adapts autonomously to arbitrary target behavior without requiring explicit estimation of target acceleration or reliance on idealized assumptions about target motion.

- describe proportional navigation  
Proportional navigation is a classical guidance law that commands missile acceleration perpendicular to the line-of-sight vector in proportion to the rate of change of the line-of-sight angle. This method operates under the principle that if the line-of-sight between the missile and target remains fixed in inertial space, a collision trajectory is achieved. The commanded acceleration is calculated as the product of a navigation gain and the line-of-sight rotation rate, which is derived from the relative velocity and position vectors between the missile and target. While computationally efficient and robust under ideal conditions, proportional navigation is fundamentally reactive, responding only to observed changes in the line-of-sight direction without anticipating or compensating for the underlying cause of those changes. Consequently, when the target executes maneuvers that induce rapid or non-linear variations in the line-of-sight rotation rate, the missile’s response becomes delayed and excessive, resulting in inefficient use of available control authority and increased miss distance. The law does not inherently distinguish between target-induced motion and measurement noise or environmental perturbations such as radome refraction, further limiting its adaptability in complex operational environments.

- describe need for improved missile guidance  
There exists a critical need for a guidance methodology that achieves superior interception accuracy against highly agile targets while simultaneously minimizing control effort and remaining robust to sensor imperfections, unmodeled dynamics, and environmental disturbances. Current approaches either require accurate target acceleration estimation—prone to error and instability—or rely on fixed control laws that lack adaptability to diverse maneuver patterns. A guidance system capable of learning and responding to the statistical structure of target behavior over time, without explicit modeling of target dynamics, would represent a significant advancement. Such a system must operate in real time using only passive sensor inputs, tolerate measurement noise and sensor biases, and dynamically adjust its trajectory in response to evolving engagement conditions. Furthermore, it must do so without increasing computational burden beyond practical limits for onboard implementation. The ideal solution would not merely improve performance marginally but would fundamentally reframe the problem by treating the line-of-sight vector not as a direct measurement to be followed, but as a malleable signal that can be shaped intelligently to enhance interception probability under uncertainty.

## SUMMARY

- introduce curvature parameterization  
A novel curvature parameterization is introduced to dynamically reshape the line-of-sight vector during missile engagement by applying a time-varying rotational transformation derived from a learned policy. This parameterization decouples the guidance command from direct measurement of target acceleration by instead modulating the orientation of the line-of-sight vector through a three-dimensional Euler rotation, thereby enabling the missile to follow a curved trajectory that implicitly compensates for target maneuvers. The rotation is parameterized as a continuous function of the history of navigation system outputs, allowing the system to integrate temporal context and infer latent target behavior without explicit estimation of target kinematics.

- describe application of deep learning network  
The curvature parameterization is implemented via a deep recurrent neural network that maps a sequence of past and current navigation observations—including line-of-sight rotation rate, relative position, velocity, and sensor noise estimates—to a three-dimensional Euler angle vector that defines the desired rotation of the line-of-sight frame. This network is trained using reinforcement learning in a simulated environment that replicates realistic engagement conditions, including variable target maneuvers, aerodynamic drag, radome refraction, and actuator dynamics. By leveraging recurrent layers, the network maintains an internal state that evolves over time, enabling it to recognize patterns in target motion and adapt its shaping strategy accordingly. The resulting policy generates a curved line-of-sight vector that is then fed into a conventional proportional navigation law, creating a hybrid guidance system that retains the simplicity of PN while achieving the performance of more complex adaptive methods.

## DETAILED DESCRIPTION

- introduce navigation system  
The navigation system comprises a suite of sensors and signal processing modules that continuously measure the relative position and velocity between the missile and the target, from which the line-of-sight unit vector is computed. These measurements are subject to inherent noise and biases introduced by the seeker, radome refraction, and atmospheric turbulence. The system also incorporates a low-pass filter to attenuate high-frequency sensor noise while preserving the essential dynamics of target motion. All outputs from the navigation system are temporally sampled at a fixed rate and provided as inputs to the curvature policy network.

- describe line-of-sight unit vector  
The line-of-sight unit vector is defined as the normalized vector pointing from the missile’s position to the target’s position in an inertial reference frame. This vector is the primary input to the guidance law and is used to compute the line-of-sight rotation rate, which drives the commanded acceleration in proportional navigation. In the disclosed system, this vector is not used directly but is instead transformed by a time-varying direction cosine matrix derived from the curvature policy, resulting in a modified line-of-sight vector that encodes a shaped trajectory.

- derive line-of-sight rotation rate  
The line-of-sight rotation rate is computed as the cross product of the line-of-sight unit vector and its time derivative, scaled by the inverse of the range between missile and target. This quantity represents the angular velocity of the line-of-sight vector in the missile’s reference frame and is used as a key input to both the conventional proportional navigation law and the curvature policy network. The derivation accounts for the relative motion of both missile and target, and is computed in real time using numerical differentiation of the position measurements.

- introduce line-of-sight bias network  
A deep recurrent neural network, referred to as the line-of-sight bias network, is introduced to generate a three-dimensional Euler angle adjustment that is applied to the line-of-sight unit vector. This network receives as input a history of navigation observations, including the line-of-sight rotation rate, relative velocity, and sensor noise estimates, and outputs a continuous rotation vector that biases the orientation of the line-of-sight vector in three dimensions. The network is trained to minimize miss distance while penalizing excessive control effort, thereby learning to curve the line-of-sight in ways that anticipate and counteract target maneuvers.

- describe curvature parameterization  
The curvature parameterization defines a continuous, time-dependent transformation of the line-of-sight vector through a sequence of Euler rotations applied in 3-2-1 order. This transformation is parameterized by the output of the bias network and is applied as a direction cosine matrix that rotates the original line-of-sight vector into a new, curved trajectory. The parameterization is designed to be smooth and differentiable, ensuring compatibility with the guidance law and avoiding discontinuities that could induce instability. The magnitude and direction of the curvature are not pre-defined but are learned adaptively through interaction with simulated engagements.

- apply curvature parameterization to line-of-sight unit vector  
The curvature parameterization is applied to the line-of-sight unit vector by multiplying it with the direction cosine matrix generated by the bias network. This operation yields a new, shaped line-of-sight unit vector that no longer points directly to the target’s current position but instead to a predicted point along a curved trajectory optimized for interception. This shaped vector is then used as the input to the proportional navigation law, effectively transforming the guidance system into a non-linear, adaptive controller.

- output curved line-of-sight unit vector  
The output of this process is a time-varying, curved line-of-sight unit vector that serves as the sole input to the proportional navigation guidance law. This vector is computed at each time step and is used to derive the commanded acceleration vector perpendicular to the curved line-of-sight. The curvature is not a fixed function of range or time but is dynamically adjusted based on the learned policy, enabling the missile to respond intelligently to target behavior without requiring explicit target acceleration estimation.

- describe reinforcement learning framework  
The curvature policy is optimized using a reinforcement learning framework in which an agent learns to maximize a cumulative reward signal by interacting with a simulated engagement environment. The agent’s policy is implemented as a neural network that maps sequences of navigation observations to curvature adjustments. The environment models the dynamics of missile and target motion, aerodynamic forces, sensor noise, and radome refraction. Over thousands of simulated engagements, the agent learns to select curvature profiles that minimize miss distance while conserving control effort.

- introduce agent  
The agent is a computational entity that implements the curvature policy and interacts with the simulation environment by selecting actions at each time step. The agent’s behavior is governed by a deep recurrent neural network that maintains an internal state representing the history of observations and actions. This recurrent structure enables the agent to infer the nature of the target’s maneuver from temporal patterns in the line-of-sight data, allowing it to respond to weaving, jinking, and step maneuvers with context-aware adjustments.

- describe environment  
The environment is a high-fidelity three-degree-of-freedom simulation that models the motion of both missile and target under realistic aerodynamic constraints, including dynamic pressure-dependent acceleration limits, drag forces, and actuator lag. The environment randomly generates initial conditions, target maneuvers, and sensor noise parameters for each episode, ensuring broad coverage of engagement scenarios. Termination occurs when the relative velocity vector reverses direction, indicating a miss or intercept.

- introduce engagement scenario generator  
An engagement scenario generator is integrated into the environment to randomly sample initial positions, velocities, heading errors, target acceleration profiles, and environmental parameters such as atmospheric density and radome refraction coefficients. This ensures that the policy is trained across a diverse and statistically representative set of scenarios, enhancing its generalization capability to novel engagements not seen during training.

- describe dynamics model  
The dynamics model integrates the equations of motion for both missile and target using fourth-order Runge-Kutta numerical integration, accounting for aerodynamic drag, control surface deflection limits, and time delays in the flight control system. The missile’s acceleration is constrained by dynamic pressure and maximum load limits, while the target’s acceleration is modeled as a piecewise-constant function with random onset, duration, and magnitude.

- introduce ensemble of target behavior models  
The training environment incorporates an ensemble of target behavior models, including bang-bang, sinusoidal weave, and random jinking maneuvers, each with randomized parameters such as frequency, amplitude, and switching time. This ensemble ensures that the policy learns to adapt to a wide spectrum of target motion patterns, rather than overfitting to any single maneuver type.

- describe radome model  
A radome refraction model is incorporated to simulate the optical distortion caused by the missile’s nose cone, which induces a bias in the measured line-of-sight vector as a function of look angle. The model includes azimuthal and elevation refraction errors that vary with the angle between the missile’s velocity vector and the true line-of-sight, introducing a parasitic attitude loop that degrades guidance performance. The curvature policy learns to compensate for this distortion implicitly through its shaping strategy.

- introduce guidance law  
The guidance law is based on true proportional navigation, which commands acceleration perpendicular to the curved line-of-sight vector in proportion to its rotation rate. The curvature-adjusted line-of-sight vector replaces the raw measurement in the traditional PN formulation, transforming the law into a non-linear, adaptive system. The navigation gain is fixed at a value optimized for conventional PN, ensuring compatibility with existing hardware.

- describe reward function  
The reward function is composed of a terminal reward based on miss distance, a curvature penalty that discourages excessive line-of-sight shaping, and a control effort penalty proportional to the square of the commanded acceleration. The terminal reward is weighted more heavily than intermediate penalties, encouraging the agent to prioritize interception accuracy while still favoring efficient trajectories.

- introduce trajectory accumulation module  
A trajectory accumulation module collects sequences of observations, actions, and rewards over multiple episodes to form training batches for policy and value function updates. Each batch contains data from 60 simulated engagements, ensuring sufficient statistical diversity for stable learning.

- update policy function  
The policy function is updated using a proximal policy optimization algorithm that employs a clipped objective function to constrain the magnitude of policy changes between updates. This ensures numerical stability and prevents catastrophic forgetting during training.

- update value function  
The value function, which estimates the expected cumulative reward from a given state, is updated using mean squared error minimization between predicted and empirical returns. It serves as a baseline to reduce variance in policy gradient estimates.

- describe optimization  
Optimization is performed over 90,000 training episodes, with policy and value function updates occurring after each batch of 60 episodes. Learning rates and clipping parameters are adaptively tuned to maintain a Kullback-Leibler divergence between successive policies below a threshold of 0.001.

- introduce terminal rewards  
Terminal rewards are assigned at the end of each episode and are inversely proportional to the miss distance, with a maximum reward given for intercepts within one meter. These rewards are discounted less heavily than intermediate rewards to emphasize the importance of successful interception.

- describe curvature penalty  
A curvature penalty is applied at each time step proportional to the squared magnitude of the Euler angle adjustment, discouraging unnecessary or excessive shaping of the line-of-sight vector. This ensures that the policy only curves the trajectory when it provides a clear benefit to interception performance.

- introduce hyperparameters  
Key hyperparameters include the discount factor for rewards, the clipping range for the policy objective, the learning rates for policy and value networks, the number of recurrent units, and the magnitude of the curvature penalty. These values are selected through grid search and validated across multiple training runs.

- describe value function  
The value function is implemented as a four-layer neural network with gated recurrent units in the second layer, allowing it to capture temporal dependencies in the state trajectory. It outputs a scalar estimate of the expected cumulative reward from the current observation history.

- introduce cost function  
The cost function is defined as the negative of the reward function and is minimized during training. It combines the terminal miss distance, control effort, and curvature penalty into a single scalar metric that guides the optimization process.

- describe advantage function  
The advantage function is computed as the difference between the empirical return and the value function estimate, providing a measure of how much better or worse an action was compared to the expected outcome. This signal is used to guide policy updates in a way that emphasizes actions leading to superior outcomes.

- introduce policy gradient method  
The policy gradient method is employed to directly optimize the parameters of the neural network policy by computing the gradient of the expected reward with respect to those parameters. This allows for end-to-end learning of the mapping from observations to curvature adjustments.

- describe proximal policy optimization  
Proximal policy optimization is used to stabilize training by constraining policy updates using a clipped probability ratio. This prevents large, destabilizing changes in the policy while still allowing for significant improvement over time.

- introduce clipped objective function  
The clipped objective function is defined as the minimum of the standard policy gradient term and a version of that term where the probability ratio is bounded within a specified range. This ensures that policy updates remain within a trust region, promoting convergence and robustness.

- describe optimization objective  
The optimization objective is to maximize the expected cumulative reward over all possible engagement scenarios, subject to constraints on control effort and curvature magnitude. This is achieved by iteratively improving the policy and value function using sampled trajectories from the environment.

- implement policy function  
The policy function is implemented as a neural network with three fully connected hidden layers and one recurrent layer using gated recurrent units. The input layer accepts 12-dimensional observation vectors, and the output layer produces a three-dimensional Euler angle adjustment.

- implement value function  
The value function is implemented as a parallel neural network with identical architecture to the policy function, sharing the same recurrent layer structure to maintain consistency in temporal state representation.

- introduce recurrent network layers  
Recurrent network layers enable the policy and value functions to maintain an internal memory of past observations, allowing them to infer the type, timing, and intensity of target maneuvers from historical patterns in the line-of-sight data. This memory is critical for distinguishing between noise and genuine maneuver signals.

- describe inference of target maneuvers  
Through training, the recurrent layers learn to infer the underlying target behavior model from the temporal evolution of the line-of-sight rotation rate and other navigation inputs. This enables the system to anticipate maneuvers before they fully manifest, allowing for preemptive shaping of the trajectory.

- describe adaptation to target behavior models  
The system adapts to novel target behavior models during testing by leveraging the recurrent state to update its internal representation of the target’s motion characteristics. This allows the policy to generalize to maneuvers not explicitly included in training, such as irregular jinking or multi-frequency weaving.

- describe application of optimized policy function  
The optimized policy function is deployed in real-time guidance systems by embedding the trained neural network into the missile’s onboard processor. At each time step, the network receives the current navigation observations and outputs the curvature adjustment, which is applied to the line-of-sight vector before being passed to the proportional navigation law.

- describe performance of disclosed method  
The disclosed method achieves superior interception accuracy compared to both proportional navigation and augmented proportional navigation across a wide range of target maneuvers, altitudes, and sensor conditions. It consistently requires less control effort than APN, even when the target’s acceleration capability approaches that of the missile. Performance gains are most pronounced for miss distances under one meter, and the system demonstrates robustness to radome refraction, sensor noise, and unmodeled aerodynamic effects.