# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates generally to methods and systems for reinforcement learning, and more specifically to improved experience replay techniques for off-policy deep reinforcement learning algorithms. The invention provides a novel framework for structuring and sampling from experience replay buffers using event-triggered trajectory segments, thereby enhancing sample efficiency, learning stability, and convergence speed in complex reinforcement learning environments.

### 2. Description of Prior Art and Related Information

Experience replay (ER) is a fundamental technique in modern reinforcement learning that enables agents to learn from previously collected experiences stored in a buffer, rather than solely from immediate interactions with the environment. This approach improves sample efficiency and stabilizes training by breaking temporal correlations in the data stream. However, conventional experience replay implementations typically employ uniform random sampling from the replay buffer, which becomes highly inefficient in environments where crucial learning events occur infrequently relative to the total number of state transitions.

For instance, in high-frequency simulation environments such as car racing simulators, an agent may execute thousands of steps to complete a single lap, while critical events like successful overtaking maneuvers might occur only a handful of times. Uniform sampling from a monolithic experience replay buffer containing all these transitions makes it statistically unlikely that the rare but informative events will be adequately represented in training batches, leading to slow convergence and poor learning outcomes.

Prioritized Experience Replay (PER) was introduced to address this limitation by skewing the sampling distribution toward experiences with higher temporal difference (TD) errors, under the assumption that these samples contain more valuable learning signals. While PER demonstrates improved performance in some domains, it suffers from several critical limitations. First, PER may overemphasize states that are unlikely to be visited under the optimal policy, particularly when the behavior policy differs significantly from the target policy. Second, experiences with low TD errors under one policy may never be sampled again even after the policy has changed, potentially missing important learning opportunities. Third, in environments with dense reward structures, TD errors can become volatile and unreliable as prioritization signals, diminishing the effectiveness of PER.

These limitations highlight the need for improved experience replay methods that can intelligently focus on experiences that are both aligned with optimal behavior and contain sufficient contextual information to enable effective value function bootstrapping. The present invention addresses this need by introducing a novel approach that leverages domain knowledge to partition experience replay buffers into event-triggered tables containing trajectory segments leading to significant events, combined with a stratified sampling strategy that ensures balanced representation of both common and rare but critical experiences during training.

## SUMMARY OF THE INVENTION

The present invention introduces Event Tables and the Stratified Sampling from Event Tables (SSET) algorithm as a novel framework for improving experience replay in off-policy reinforcement learning. Event Tables are specialized partitions within an experience replay buffer that store finite-length trajectory segments leading to predefined events of interest, rather than storing individual transitions or entire episodes. The SSET algorithm implements a stratified sampling approach that draws fixed proportions of training samples from each Event Table as well as from a default buffer containing all experiences, thereby ensuring that critical events and their contextual histories receive adequate representation during training.

The core intuition behind SSET is the "fast-lane" concept, where trajectory segments leading to significant events form efficient pathways for value function backups to propagate from terminal or high-value states back to initial states. By maintaining these contextual histories and sampling individual steps from them independently, SSET achieves both the stability benefits of uncorrelated mini-batch sampling and the efficiency advantages of trajectory-based learning approaches.

The theoretical foundation of SSET demonstrates that when event conditions are properly correlated with optimal policy behavior and history lengths are sufficiently long, the algorithm can dramatically improve sample complexity compared to uniform sampling or even PER. The invention provides formal guarantees quantifying these improvements under specific conditions, along with a bias correction procedure that preserves the Bellman target even when event conditions are suboptimal.

Empirical validation across multiple domains demonstrates the advantages of SSET. In MiniGrid environments, SSET shows superior sample efficiency and learning stability compared to uniform experience replay, reverse-sweep approaches, and PER. In continuous control benchmarks including LunarLander and MuJoCo domains, SSET consistently improves sample efficiency and policy stability, particularly in environments with dense reward structures where PER tends to underperform. Most significantly, in the highly realistic Gran Turismo Sport racing simulator, SSET enables agents to learn complex driving skills like slipstream-assisted overtaking more reliably and mitigates catastrophic forgetting in time-trial scenarios.

The invention further demonstrates that SSET is complementary to existing techniques. When combined with TD-error prioritization within each Event Table, SSET achieves the "best of both worlds" by focusing on trajectories aligned with optimal behavior while also prioritizing states along those trajectories that require the most significant value updates. Similarly, SSET can be effectively combined with potential-based reward shaping, where shaping guides exploration while SSET provides efficient backup pathways for value propagation.

Additional benefits include SSET's ability to mitigate catastrophic forgetting in multi-skill learning scenarios by maintaining separate Event Tables for different skill components, ensuring that all skills receive adequate training attention throughout the learning process. The framework also generalizes existing multi-table partitioning schemes by allowing dynamic assignment of experiences to tables based on event conditions rather than static episode-level partitioning, and extends beyond traditional goal-oriented settings to handle any state-based event specification.

The invention connects naturally to related concepts in reinforcement learning, including initial state selection and reward shaping, while providing a more direct mechanism for incorporating domain knowledge into the learning process. Unlike options frameworks that require explicit hierarchical policy structures, SSET operates purely at the experience replay level, making it compatible with any off-policy reinforcement learning algorithm without requiring architectural modifications.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS AND BEST MODE OF INVENTION

The present invention provides a comprehensive framework for improving experience replay in off-policy reinforcement learning through the introduction of Event Tables and the Stratified Sampling from Event Tables (SSET) algorithm. The following detailed description outlines the preferred embodiments and best mode of implementing this invention, including precise definitions of key terminology, hardware and software requirements, algorithmic specifications, and empirical validation results.

### Terminology

The invention operates within the standard framework of reinforcement learning, where a reinforcement learning agent interacts with an environment modeled as an episodic Markov Decision Process (MDP) M = ⟨S, A, R, P, γ, I, β⟩. Here, S represents the state space, A denotes the action space, R: S × A → Pr[ℝ] defines the reward function, P: S × A → Pr[S] specifies the transition kernel, γ ∈ [0, 1) is the discount factor, I: Pr[S] describes the initial state distribution, and β: S → {0, 1} indicates episode termination conditions.

The agent's objective is to learn an optimal policy π* that maximizes the expected discounted return, represented by the optimal action-value function Q*(s, a). In model-free off-policy methods, this function is learned directly from experience data through incremental updates, such as the Q-learning update rule: Q_{k+1}(s, a) = (1 - α)Q_k(s, a) + αδ, where α ∈ (0, 1] is the learning rate and δ = r(s, a) + γV_k(s') - Q_k(s, a) is the temporal difference (TD) error, with V_k(s') = max_{a'} Q_k(s', a').

In deep reinforcement learning applications, the state space S is typically continuous and high-dimensional, necessitating neural network representations for the value function and policy. Experience replay (ER) buffers store tuples of experience ⟨s, a, r, s'⟩ collected during agent-environment interactions, with mini-batches sampled from these buffers for gradient-based updates to network parameters.

A key innovation of the present invention is the formal definition of an event specification (event spec): ν = ⟨ω, τ⟩, comprising a Boolean event condition ω: S → {0, 1} and a history length τ. An event occurs in state s when ω(s) evaluates to true. The event condition should ideally identify states that are visited more frequently under the optimal policy than under the behavior collection policy, including terminal goal states, high-reward states, bottleneck states, or important rare states such as successful overtaking maneuvers in racing scenarios. The history length τ must be sufficiently long to ensure that trajectory segments stored in Event Tables can chain together from initial states to final states, forming effective "fast lanes" for value function backups.

### MiniGrid Experiments

The invention was extensively validated in MiniGrid environments using the DDQN reinforcement learning algorithm with dense neural network architectures and ε-greedy behavior policies. In a three-room grid world scenario, SSET demonstrated superior sample complexity and learning stability compared to uniform experience replay, reverse-sweep approaches, and PER. The implementation used two event conditions: one triggering at gap states between rooms and another at the goal state, with a history length of 200 steps.

Comparative analysis with potential-based reward shaping revealed that SSET outperformed shaping alone, while their combination yielded the best overall performance. This demonstrates that while reward shaping can guide exploration, SSET provides the essential mechanism for efficiently propagating value estimates back to initial states through its fast-lane trajectory segments.

The robustness of SSET to poorly designed event conditions was also evaluated. Even with suboptimal event specifications, SSET generally performed no worse than uniform sampling and often outperformed reward shaping baselines. Performance was found to be relatively insensitive to the default buffer's sampling probability (η₀) when using well-designed event conditions, though careful tuning could benefit mediocre event specifications.

In more complex obstacle course environments with randomized object placements, SSET successfully acquired and maintained multiple skills simultaneously, whereas uniform experience replay tended to focus on easier skills while neglecting more challenging ones. This capability to balance learning across diverse skill requirements represents a significant advantage in complex environments.

Catastrophic forgetting was explicitly addressed in extended learning scenarios, where SSET maintained consistent performance on perturbed initial states while uniform and PER-based approaches exhibited forgetting behavior. This demonstrates SSET's ability to preserve value estimates for critical transitions even as the behavior policy evolves during training.

The importance of intermediate events as waypoints was empirically confirmed, showing that SSET with multiple intermediate event conditions significantly outperformed approaches using only terminal goal events. Different sampling weight configurations were tested, revealing that equal weights across event tables generally provided robust performance, though task-specific tuning could yield marginal improvements.

Finally, the integration of Conflict-Averse Gradient Descent (CAGrad) with SSET was explored to address potential gradient conflicts between different event tables. Results indicated that while CAGrad could accelerate initial learning of difficult skills, it might compromise asymptotic performance, suggesting that dynamic scheduling of the conflict-aversion parameter could optimize both learning speed and final performance.

### Lunar Lander and Mujoco Experiments

The invention was further validated on continuous control benchmark tasks including LunarLanderContinuous-v3 and the MuJoCo suite from OpenAI Gym. These domains feature dense reward structures and, in the case of MuJoCo, lack predefined goal states, making them particularly challenging for traditional event-based approaches.

For LunarLander, event conditions were defined based on physical contact states (both legs making contact with landing flags) and positional proximity to the landing zone. In MuJoCo domains, event conditions were triggered when agents received rewards exceeding manually selected thresholds, with history lengths of 200 steps for all events.

Using the state-of-the-art SAC algorithm, SSET consistently demonstrated improved sample efficiency (achieving target performance in roughly half the epochs required by baselines) and greater policy stability across all tested domains. Notably, PER performed at best similarly to uniform experience replay in these dense reward environments, with performance degrading as the priority exponent increased, likely due to the volatility of TD errors in densely rewarded scenarios.

Ablation studies on history length variations in MuJoCo domains confirmed SSET's robustness to non-optimal history lengths while still benefiting from sufficiently long contextual histories. These results reinforce the recommendation to use longer history lengths when prior knowledge about optimal event spacing is limited, while also indicating that history lengths can be tuned for peak performance when domain expertise is available.

### Simulated Car Racing Experiments

The most compelling validation of the invention was conducted in the Gran Turismo Sport racing simulator, a highly realistic environment previously used to demonstrate reinforcement learning systems that outperformed human e-sports champions. Two distinct scenarios were investigated: a "slingshot passing" task on a straightaway section of the Circuit de la Sarthe track, and a time-trial scenario on the full Lago Maggiore GP track.

In the slingshot passing scenario, two event conditions were defined: a "slipstream" event triggered when the agent's slipstream feature exceeded a threshold of 0.7, and a "won" event triggered when the agent finished the section in first place. Both events used history lengths of 10-15 seconds and sampling proportions of 10%. SSET demonstrated consistently superior performance compared to uniform sampling, with lower variance across training runs and more reliable learning of the complex slipstream-assisted overtaking maneuver.

Robustness to event threshold variations was also tested, with results showing that SSET maintained advantages even when events were made harder to trigger (requiring slipstream values above 0.9), though with increased variance due to the rarity of early triggering events. This demonstrates SSET's adaptability to different event specifications while maintaining performance advantages over baseline approaches.

In the time-trial scenario, SSET was used to mitigate catastrophic forgetting of off-course avoidance behaviors. A "re-establish" event was defined to trigger when the agent returned to the track after leaving it, with a history length of 7 seconds to capture the full trajectory of leaving and returning to the course. Results showed that SSET policies maintained consistent on-course behavior throughout training, with 88.9% of policies incurring no penalties after epoch 1000, compared to only 74.7% for uniform sampling policies. Remarkably, even the worst-performing SSET run stayed on course more frequently than the best uniform sampling run, while maintaining comparable lap times.

### Recommendations on How to Pick Helpful Events

The invention provides practical guidelines for specifying effective event conditions based on empirical observations and theoretical insights. For users with limited domain knowledge, simple goal events (in goal-oriented environments) or reward-threshold events (in dense reward environments) with relatively long history lengths can provide substantial benefits without requiring deep understanding of optimal subgoals.

The robustness of SSET to poorly chosen events is another key advantage, as the technique typically performs no worse than uniform sampling even with suboptimal event specifications. Pathological cases that could actually hinder performance are easily avoidable in practice by setting reasonable caps on event table sizes (recommended to not exceed 30% of total buffer capacity).

From an implementation perspective, Event Tables require minimal additional memory overhead, as they can be implemented using unsigned integer indices pointing to data already stored in the main buffer. Efficient implementations are already available in packages like Reverb, making adoption straightforward for existing reinforcement learning systems.

While the current invention relies on user-specified event conditions, future extensions could explore automated event discovery through subgoal identification techniques or online learning of event weights, though such approaches would require additional theoretical guarantees to ensure convergence.

### Summary

The SSET algorithm represents a significant advancement in experience replay methodology for off-policy reinforcement learning. By combining domain knowledge through event specifications with stratified sampling from trajectory-segmented Event Tables, SSET provides improved sample efficiency, learning stability, and convergence speed across diverse reinforcement learning domains. The algorithm's compatibility with existing techniques like PER and reward shaping, along with its ability to mitigate catastrophic forgetting and balance multi-skill learning, makes it a versatile and powerful addition to the reinforcement learning toolkit.

### Definitions, Lemmas, Propositions Used Above

The theoretical foundation of the invention establishes formal guarantees for the performance improvements achievable with SSET. Event Tables are defined as multisets of transitions from trajectories of maximum length τ_i that lead to states satisfying event condition ω_i. The SSET algorithm implements weighted sampling between event tables and a default buffer containing all experiences, with sampling probabilities η_i for event tables and η_0 = 1 - Ση_i for the default buffer.

Key theoretical results include Lemma 1, which quantifies the over-sampling of experiences in event tables as m > 1/(1-η)^(2m) where m is asymptotically convergent to ln(τ_iμ) - ln ln(τ_iμ) + o(1) as μ → 0, with μ representing the threshold for state density disparity between optimal and behavior policies. Lemma 2 provides a bias correction procedure that preserves the Bellman target by computing weights for weighted importance sampling based on the probability of transitions not being included in any event table.

Theorem 1 establishes the main convergence result, showing that under appropriate conditions on event correlation with optimal behavior and sufficient history lengths, the sample complexity for achieving ε-optimal solutions is reduced compared to uniform sampling. Specifically, if τ_i ≤ (1-η)^m/((m+1)nμ) for all i ∈ [1,n], then the convergence rate improvement compounds across multiple outer iterations of target Q-learning, benefiting even transitions not included in event tables through the compounding effect of improved value estimates.

The theoretical analysis assumes a finite discrete episodic MDP with bounded rewards and a fixed stochastic behavior policy that can visit every state-action pair with non-zero probability. The results are derived using tabular Q-learning with target networks as the base algorithm, building upon existing finite-time convergence results from the literature. The constants C_B and L_B represent the minimum and maximum probabilities of sampling any state-action pair from the buffer, with C_B > 0 ensuring sufficient exploration coverage.

The invention's scope encompasses all equivalent implementations and modifications that incorporate the essential idea of stratified sampling from event-triggered trajectory segments, including alternative bias correction procedures, different event specification formats, and various sampling strategies that maintain the core fast-lane intuition for efficient value function bootstrapping.