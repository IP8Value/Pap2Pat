# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates to the field of reinforcement learning (RL), particularly to methods and systems for improving the efficiency and stability of off-policy RL algorithms through the use of Event Tables and a corresponding sampling algorithm, Stratified Sampling from Event Tables (SSET).

### 2. Description of Prior Art and Related Information

Recent advancements in deep reinforcement learning (RL) have relied heavily on Experience Replay (ER) and the corresponding Experience Replay Buffer (ERB) to store and reuse large amounts of data during training. However, traditional ER methods, such as uniform random sampling and Prioritized Experience Replay (PER), often struggle to focus on critical events that are crucial for learning optimal policies. Uniform random sampling from an ERB is inefficient as it treats all experiences equally, regardless of their importance. PER, which prioritizes experiences based on Temporal Difference (TD) errors, can also be suboptimal as it may focus on states that are unlikely to be encountered by the optimal policy.

To address these limitations, the present invention introduces Event Tables, which are partitions of the ERB that store sub-trajectories leading to specific events, and SSET, a sampling algorithm that builds training samples from these tables. This approach ensures that the RL algorithm focuses on crucial events, leading to faster convergence and more stable learning.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for improving the efficiency and stability of off-policy reinforcement learning (RL) algorithms. The invention introduces Event Tables, which are partitions of the Experience Replay Buffer (ERB) that store sub-trajectories leading to specific events, and a corresponding sampling algorithm, Stratified Sampling from Event Tables (SSET), which builds training samples from these tables.

Key features of the invention include:
1. **Event Tables**: Partitions of the ERB that store sub-trajectories leading to specific events, ensuring that the RL algorithm focuses on crucial states.
2. **Stratified Sampling from Event Tables (SSET)**: A sampling algorithm that constructs mini-batches by stratified sampling from the event tables and the default ERB, ensuring balanced and focused training.
3. **Bias Correction**: A mechanism to correct the bias introduced by the stratified sampling, preserving the Bellman target and ensuring the correctness of the learning process.
4. **Complementary Techniques**: The invention can be combined with other prioritization methods, such as TD-error PER, to further enhance learning efficiency and stability.

The invention is particularly useful in large and complex RL domains where traditional ER methods are inefficient. Empirical results demonstrate that SSET outperforms uniform sampling and PER in various environments, including MiniGrid, MuJoCo, Lunar Lander, and a highly realistic Gran Turismo Sport race-car simulator.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS AND BEST MODE OF INVENTION

### Terminology

In the context of reinforcement learning, the following terminology is used:

- **Reinforcement Learning (RL)**: A type of machine learning where an agent learns to make decisions by interacting with an environment to maximize a reward.
- **Markov Decision Process (MDP)**: A mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker.
- **Experience Replay Buffer (ERB)**: A data structure used in RL to store and reuse past experiences for training.
- **Temporal Difference (TD) Error**: The difference between the predicted and actual rewards, used to update the value function in RL.
- **Prioritized Experience Replay (PER)**: A method that prioritizes experiences in the ERB based on their TD errors.
- **Event Tables**: Partitions of the ERB that store sub-trajectories leading to specific events.
- **Stratified Sampling from Event Tables (SSET)**: A sampling algorithm that constructs mini-batches by stratified sampling from the event tables and the default ERB.

### MiniGrid Experiments

In the MiniGrid domain, the invention was tested using the Double DQN (DDQN) algorithm. The environment consists of a grid world where the agent must navigate through various obstacles to reach a goal. Two event conditions were used: one that occurs at the gap between rooms and another at the goal. The history length was set to 200 steps. The results showed that SSET significantly improved sample efficiency and learning stability compared to uniform sampling and PER.

### Lunar Lander and Mujoco Experiments

The invention was also tested on continuous control tasks using the Soft Actor-Critic (SAC) algorithm. In the Lunar Lander environment, two event conditions were used: one when the lander's legs make contact between the flags and another when the lander's position is close to the middle of the flags. For the MuJoCo suite, three event conditions were used based on reward thresholds. The results demonstrated that SSET improved sample efficiency and achieved stable policies by bootstrapping salient rewards more rapidly.

### Simulated Car Racing Experiments

In a highly realistic Gran Turismo Sport race-car simulator, the invention was tested in two scenarios: a slingshot passing scenario and a time trial scenario. In the slingshot passing scenario, two events were introduced: a "slipstream" event and a "won" event. The results showed that SSET significantly improved sample complexity and mitigated off-course driving. In the time trial scenario, an event was used to re-establish on-course behavior, which helped in maintaining consistent on-track laps and avoiding catastrophic forgetting.

### Recommendations on How to Pick Helpful Events

1. **Goal Events**: For environments with a clear goal, use a goal event with a long history to capture the trajectory leading to the goal.
2. **Reward-Threshold Events**: In environments with dense rewards, use reward-threshold events to capture significant progress.
3. **Intermediate Events**: Use intermediate events as waypoints to guide the learning process, especially in sparse reward environments.
4. **Avoid Pathological Events**: Ensure that the majority of the buffer is not used for incorrect events by setting reasonable caps on event table sizes.

### Summary

The present invention, Event Tables and SSET, provides a novel approach to improve the efficiency and stability of off-policy RL algorithms. By focusing on crucial events and using stratified sampling, the invention ensures that the RL algorithm learns more effectively and stably. Empirical results in various environments, including MiniGrid, MuJoCo, Lunar Lander, and a Gran Turismo Sport simulator, demonstrate the effectiveness of the invention.

### Definitions, Lemmas, Propositions Used Above

#### Definitions

1. **Event Condition**: A Boolean condition over states that defines when an event occurs.
2. **History Length**: The length of the sub-trajectory leading to an event that is stored in the event table.
3. **Event Table**: A partition of the ERB that stores sub-trajectories leading to specific events.
4. **Stratified Sampling from Event Tables (SSET)**: A sampling algorithm that constructs mini-batches by stratified sampling from the event tables and the default ERB.

#### Lemmas

1. **Lemma 1 (Over-Sampling of Experiences)**: The probability of sampling an event state from an event table is significantly higher than from the default ERB.
2. **Lemma 2 (Bias Correction)**: The bias introduced by the stratified sampling can be corrected using weighted importance sampling.

#### Propositions

1. **Proposition 1 (Sample Complexity Bound)**: The sample complexity for achieving an ε-optimal solution using SSET is reduced compared to uniform sampling.
2. **Proposition 2 (Event Section Coverage)**: All initial states, event states, and optimal terminal states belong to at least one event section.

These definitions, lemmas, and propositions provide the theoretical foundation for the invention, ensuring that the method is both effective and mathematically sound.