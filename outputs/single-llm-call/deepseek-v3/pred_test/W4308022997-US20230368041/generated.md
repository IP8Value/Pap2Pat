Here is the patent application following your outline:

# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates generally to reinforcement learning methods and more specifically to improved techniques for experience replay in reinforcement learning systems. The invention provides novel methods for partitioning and sampling from experience replay buffers to accelerate learning while maintaining stability.

### 2. Description of Prior Art and Related Information

Current reinforcement learning systems commonly employ experience replay buffers to store and reuse past experiences during training. Standard uniform sampling from these buffers fails to adequately focus learning on crucial but rare events that are important for optimal performance. While prioritized experience replay methods attempt to address this limitation by skewing sampling based on temporal difference errors, these approaches still suffer from several drawbacks. Prioritized experience replay may focus learning on states that are irrelevant to the optimal policy while neglecting important states that currently have low temporal difference errors. There exists a need in the art for improved methods that can more effectively focus learning on experiences relevant to optimal performance while maintaining the stability benefits of experience replay.

## SUMMARY OF THE INVENTION

The present invention introduces a novel framework called Stratified Sampling from Event Tables (SSET) that partitions the experience replay buffer into multiple event tables based on predefined event conditions. Each event table stores not only the states where events occur but also the histories leading up to those events, creating "fast lanes" for propagating value updates through the state space. The SSET algorithm samples experiences from these event tables in a stratified manner, providing theoretical guarantees of improved sample complexity when events are correlated with optimal behavior.

The invention establishes that when event conditions occur in states more frequently visited by the optimal policy than the behavior policy, and when history lengths are sufficiently long to chain back to previous events or initial states, SSET can dramatically accelerate the convergence of off-policy reinforcement learning. Even when these ideal conditions are not fully met, the invention includes a bias correction procedure that preserves the Bellman target while still providing practical benefits.

Empirical results demonstrate that SSET outperforms both uniform sampling and prioritized experience replay across a range of domains including grid worlds, continuous control benchmarks, and high-fidelity racing simulators. The invention shows particular advantages in sample efficiency, learning stability, and mitigation of catastrophic forgetting. Furthermore, SSET can be effectively combined with both prioritized experience replay and reward shaping techniques to achieve even greater performance improvements.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS AND BEST MODE OF INVENTION

The present invention provides a comprehensive framework for implementing Stratified Sampling from Event Tables (SSET) in reinforcement learning systems. The following detailed description explains the components, algorithms, and implementation considerations that constitute the preferred embodiments of the invention.

### Terminology

A reinforcement learning agent according to the invention operates within an episodic Markov Decision Process characterized by a state space, action space, reward function, transition dynamics, discount factor, initial state distribution, and episode termination function. The agent maintains a value function representing the expected long-term discounted return from each state-action pair, which is updated through Q-learning updates based on temporal difference errors.

The invention introduces the concept of event specifications, each comprising an event condition (a Boolean function over states) and a history length. An event occurs when an agent enters a state satisfying the event condition. Event conditions are designed to identify states that are important for optimal performance, such as terminal goal states, high reward states, bottleneck states, or other strategically significant states. The history length determines how many preceding state transitions are stored in association with each event occurrence.

The invention employs an experience replay buffer that is partitioned into multiple event tables corresponding to different event specifications, along with a default table for storing general experiences. Each event table contains sequences of state transitions leading up to the associated events, enabling focused learning on these critical trajectories. The system implements deep reinforcement learning techniques where value functions and policies are represented by neural networks trained using mini-batches sampled from the partitioned buffer.

### MiniGrid Experiments

The invention has been empirically validated in MiniGrid environments using a Double Deep Q-Network (DDQN) reinforcement learning algorithm with dense neural network architectures. In a three-room grid world environment, the invention demonstrates significant improvements in sample complexity and learning stability compared to uniform experience replay and prioritized experience replay. SSET with event conditions at room gaps and the goal state learns optimal behavior more quickly and with lower variance across training runs.

Comparative experiments with potential-based reward shaping show that SSET provides complementary benefits, with the combination of both techniques yielding the best performance. While reward shaping guides exploration through modified rewards, SSET provides efficient value propagation through its event tables. The invention also demonstrates robustness to suboptimal event condition selection, with performance degrading gracefully rather than catastrophically when events are poorly chosen.

In more complex obstacle course environments with multiple challenges (spikes, lava, keys, and doors), SSET shows particular advantages in acquiring and maintaining diverse skills. The invention's event tables for different obstacle types enable effective learning without forgetting, outperforming both uniform and prioritized experience replay in multi-skill scenarios. Experiments varying history lengths and sampling weights demonstrate the invention's flexibility and robustness to parameter choices.

### Lunar Lander and Mujoco Experiments

The invention has been validated in continuous control benchmarks including LunarLanderContinuous-v3 and the MuJoCo suite. Using the Soft Actor-Critic (SAC) algorithm, SSET with event conditions based on reward thresholds demonstrates improved sample efficiency and stability compared to baseline methods. The invention shows particular advantages in these dense-reward environments where temporal difference errors may be volatile and less reliable for prioritization.

Ablation studies varying history lengths confirm that SSET maintains benefits across a range of parameter settings, with longer histories generally providing better performance when no prior knowledge is available. The invention's robustness to parameter choices makes it practical for real-world applications where optimal settings may not be known in advance.

### Simulated Car Racing Experiments

The invention has been tested in the highly realistic Gran Turismo Sport racing simulator, demonstrating practical advantages in challenging real-world scenarios. In a slingshot passing scenario, SSET with events for slipstream conditions and race outcomes learns effective passing strategies more consistently than uniform sampling. The invention shows robustness to variations in event threshold settings and maintains advantages even with less informative event definitions.

In time trial settings, SSET effectively mitigates catastrophic forgetting of off-course penalties, which become rare as learning progresses. By maintaining an event table for re-establishing on-course behavior, the invention produces more stable policies that consistently stay on track compared to standard experience replay approaches. These results demonstrate the invention's practical value in complex, high-fidelity environments.

### Recommendations on How to Pick Helpful Events

The invention provides guidelines for selecting effective event conditions in different domains. For users with limited domain knowledge, basic events like goal states or reward thresholds with sufficiently long histories can provide benefits. Even suboptimal event conditions typically perform no worse than uniform sampling, with performance degrading gracefully rather than catastrophically.

The invention suggests allocating no more than 30% of buffer capacity to any single event table to maintain robustness. Event tables can be implemented efficiently using existing buffer implementations with minimal memory overhead. While the current invention relies on user-specified events, the framework naturally accommodates future extensions for automatically learning event conditions.

### Summary

The Stratified Sampling from Event Tables (SSET) framework provides a novel approach to experience replay that accelerates reinforcement learning while maintaining stability. By partitioning the buffer based on event conditions and sampling strategically from these partitions, the invention provides theoretical guarantees of improved sample complexity when events correlate with optimal behavior. Empirical results across diverse domains demonstrate practical advantages in sample efficiency, policy stability, and skill retention.

### Definitions, Lemmas, Propositions Used Above

The theoretical foundations of the invention include formal definitions of event tables, event conditions, and event sections. Key results include Lemma 1 quantifying the over-sampling of experiences in event tables and Lemma 2 establishing the bias correction procedure. Proposition 3 derives the convergence rate improvement, while Theorem 1 provides the main theoretical guarantee that SSET improves sample complexity under appropriate conditions.

The mathematical analysis defines state probability distributions following policies, state density disparities from optimal behavior, and conditions under which events correlate with optimal trajectories. The proofs establish that when event conditions identify states more frequently visited by the optimal policy, and when history lengths suffice to chain back to previous events or initial states, SSET provides guaranteed improvements in convergence speed while preserving the Bellman target through appropriate bias correction.