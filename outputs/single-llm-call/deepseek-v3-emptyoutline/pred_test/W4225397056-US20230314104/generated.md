Here is the drafted patent application following the provided outline and research paper:

# DESCRIPTION  

## FEDERAL FUNDING  

The invention described herein was made with government support under Contract No. [Insert Contract Number] awarded by [Insert Agency Name]. The government has certain rights in the invention.  

## BACKGROUND  

Proportional navigation (PN) guidance systems have been widely employed in missile guidance applications due to their simplicity and effectiveness against non-maneuvering targets. However, conventional PN guidance exhibits significant performance degradation when intercepting highly maneuvering targets, as the guidance law does not account for target acceleration. While augmented proportional navigation (APN) attempts to address this limitation by incorporating estimated target acceleration, accurate estimation of arbitrary target maneuvers remains challenging and can lead to divergent estimates that degrade interception performance.  

Existing approaches to improving guidance against maneuvering targets suffer from several limitations. Some methods require precise knowledge of target acceleration vectors or are only effective against specific maneuver types. Others depend on large missile-to-target acceleration ratios that may not be achievable in practical engagements. Geometric guidance techniques often assume low-speed targets or perfect knowledge of target dynamics. There remains an unmet need for a guidance system that provides robust performance against highly maneuvering targets without requiring explicit target acceleration estimation or restrictive operational constraints.  

## SUMMARY  

The present invention provides a novel guidance system and method that dynamically modifies the line of sight (LOS) input to a guidance law using an optimized curvature policy. The system comprises a line of sight curvature module that implements a deep neural network policy trained through reinforcement meta-learning to generate optimal LOS curvature parameters during engagement. These parameters are transformed into a direction cosine matrix that rotates the observed LOS vector, effectively shaping the apparent target motion presented to the guidance law.  

In preferred embodiments, the system combines the LOS curvature module with a proportional navigation guidance law (PN-LOSC), though the approach can be integrated with any guidance system utilizing LOS-derived inputs. The curvature policy is implemented as a recurrent neural network that processes navigation system outputs to generate Euler 321 attitude parameters, which are then applied to continuously adjust the LOS vector throughout the engagement. This enables the guidance system to maintain effectiveness against arbitrary target maneuvers while requiring less control effort than conventional approaches.  

Key advantages of the invention include: (1) improved interception accuracy against highly maneuvering targets without requiring explicit target acceleration estimation; (2) reduced control effort compared to both PN and APN guidance systems; (3) inherent robustness to radome refraction effects; and (4) adaptability to various engagement scenarios through meta-reinforcement learning optimization. The system maintains effectiveness even when the target's acceleration capability approaches that of the interceptor missile.  

## DETAILED DESCRIPTION  

The present invention provides a missile guidance system that dynamically shapes the line of sight (LOS) input to improve performance against maneuvering targets. Referring to FIG. 1, the system architecture comprises several key components: a navigation system (101) providing state estimates, a LOS curvature module (102) implementing the optimized policy, a guidance law processor (103), and a flight control system (104) executing acceleration commands.  

The LOS curvature module (102) implements a deep recurrent neural network policy c: o → u that maps observations o to actions u. The observations include: relative position and velocity vectors between missile and target, missile acceleration history, and LOS rotation rate. The action output u is scaled and interpreted as Euler 321 attitude parameters θ_LOSC ∈ SO(3) according to the transformation θ_LOSC = κu, where κ is a scaling factor typically set to 2. These attitude parameters are converted to a direction cosine matrix C(θ_LOSC) that rotates the ground truth LOS unit vector λ to produce a shaped LOS vector λ_LOSC = C(θ_LOSC)λ.  

The guidance law processor (103) receives the shaped LOS vector λ_LOSC and implements true proportional navigation (TPN) according to:  

a_cmd = N'V_c × Ω_LOSC  

where N' is the navigation constant (typically 3), V_c is the closing velocity, and Ω_LOSC is the LOS rotation rate derived from λ_LOSC. The commanded acceleration is adjusted to remain perpendicular to the missile velocity vector. In alternative embodiments, the shaped LOS vector may be provided to other guidance laws such as augmented proportional navigation or geometric guidance methods.  

The flight control system (104) implements the acceleration commands with consideration of dynamic pressure limitations and actuator dynamics. A first-order lag filter models flight control system response (τ = 0.08s), while a second filter (τ = 0.02s) represents actuator dynamics. The achievable acceleration is constrained by both dynamic pressure and maximum load limits (typically 40g).  

The LOS curvature policy is optimized using reinforcement meta-learning (meta-RL) across an ensemble of engagement scenarios. The policy network architecture comprises four layers with tanh activations, including a gated recurrent unit (GRU) layer that enables adaptation to different target maneuver characteristics. Optimization employs proximal policy optimization (PPO) with a clipped objective function to ensure stable training.  

The reward function for policy optimization balances interception accuracy against control effort:  

R = -ζ‖θ_LOSC‖^2 - ‖a_M‖^2 + R_terminal  

where ζ controls the penalty for LOS curvature, ‖a_M‖^2 penalizes control effort, and R_terminal provides a large positive reward for successful intercepts with miss distances below 1m. Training proceeds over approximately 90,000 simulated episodes encompassing various target maneuvers (bang-bang, weave, and jinking) with randomized timing, duration, and acceleration levels.  

The resulting PN-LOSC guidance system demonstrates superior performance to both PN and APN across multiple metrics. In testing against targets with acceleration capabilities approaching the missile's own (30g maximum), PN-LOSC achieves:  

1. 38% lower mean miss distance than APN  
2. 52% lower mean miss distance than PN  
3. 22% less control effort than APN  
4. 17% less control effort than PN  

The system's effectiveness stems from two key factors: increased robustness to radome refraction effects and more intelligent response to target maneuvers. By dynamically adjusting LOS curvature based on engagement history, the system naturally attenuates parasitic attitude loops while maintaining responsiveness to genuine target motion.  

Alternative embodiments may incorporate additional constraints into the LOS curvature policy, such as impact angle requirements or thermal loading limits. The approach remains compatible with six degree-of-freedom implementations and may be extended to other aerospace guidance applications including planetary landing and proximity operations.  

While particular embodiments have been described, the invention is not limited to these examples. Various modifications incorporating the novel LOS curvature approach with different guidance laws and optimization methods fall within the scope of the claims.