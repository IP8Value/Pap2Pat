# DESCRIPTION

## BACKGROUND

Reinforcement learning (RL) is a powerful paradigm for enabling agents to learn optimal behaviors through trial and error. One of the key challenges in RL is exploration, especially in environments with sparse or delayed rewards. Traditional exploration strategies often rely on random actions, which can be inefficient in complex and large state spaces. Curiosity-driven exploration has emerged as a promising solution to this challenge. It involves training an agent to explore the environment by optimizing intrinsic rewards derived from the discrepancy between the agent's predictions and actual outcomes. This approach encourages the agent to seek out novel and informative experiences, thereby improving its understanding of the environment and its ability to find rewarding states.

One of the most effective curiosity-driven exploration methods is the Bootstrap Your Own Latent (BYOL) approach, which has shown remarkable success in various domains, including computer vision and graph representation learning. BYOL-Explore extends this method to RL by using a self-supervised prediction loss to learn a world model and simultaneously drive exploration. The world model predicts future observations based on past experiences, and the prediction errors serve as intrinsic rewards that guide the agent's exploration.

## SUMMARY

The present invention, BYOL-Explore, is a curiosity-driven exploration algorithm designed to enhance the efficiency and effectiveness of reinforcement learning agents in complex environments. BYOL-Explore leverages a self-supervised prediction loss to learn a world model and uses the same loss to train a curiosity-driven policy. This unified approach simplifies the learning process and improves the quality of exploration by ensuring that the world model and the exploration policy are aligned.

Key features of BYOL-Explore include:
1. **Self-Supervised Learning**: The world model is trained using a self-supervised prediction loss, which predicts future observations based on past experiences.
2. **Intrinsic Rewards**: Prediction errors from the world model are used as intrinsic rewards to guide the agent's exploration.
3. **Unified Objective**: The same loss function is used for both learning the world model and training the exploration policy, ensuring a coherent and efficient learning process.
4. **Robust Performance**: BYOL-Explore has been tested on a variety of benchmark tasks, including the 10 hardest exploration Atari games and the DM-HARD-8 suite, demonstrating superior performance compared to existing methods.

## DETAILED DESCRIPTION

### 1. Overview of BYOL-Explore

BYOL-Explore is a curiosity-driven exploration algorithm that combines the strengths of self-supervised learning and intrinsic rewards to enhance the exploration capabilities of reinforcement learning agents. The algorithm consists of three main components:
1. **Latent-Predictive World Model**: A multi-step predictive model that operates at the latent level and uses a self-supervised prediction loss to learn a meaningful representation of the environment.
2. **Reward Normalization and Prioritization Scheme**: A mechanism to normalize and prioritize intrinsic rewards, ensuring stable and effective exploration.
3. **Generic RL Algorithm and Representation Sharing**: A flexible framework that can be integrated with any RL algorithm, with the option to share components of the world model with the RL model to further improve performance.

### 2. Latent-Predictive World Model

#### 2.1. Background and Notation

In a reinforcement learning setting, an agent interacts with an environment over discrete time steps. At each time step \( t \), the agent receives an observation \( o_t \) and generates an action \( a_t \). The environment transitions to a new state based on the action, and the agent receives a reward \( r_t \). The goal of the agent is to maximize the cumulative reward over time.

Formally, the environment dynamics are defined by a probability distribution \( p \) that maps a history of past observations and actions to a probability distribution over future observations:
\[ p : H \times A \rightarrow \Delta O \]
where \( H \) is the space of past observations and actions, \( A \) is the action space, and \( O \) is the observation space.

A policy \( \pi \) maps a history of past observations and actions to a probability distribution over actions:
\[ \pi : H \rightarrow \Delta A \]

An extrinsic reward function \( r_e \) maps a history of past observations and actions to a real number:
\[ r_e : H \times A \rightarrow \mathbb{R} \]

#### 2.2. Model Architecture

The BYOL-Explore world model is a multi-step predictive model that operates at the latent level. It consists of an encoder \( f_\theta \) and a recurrent neural network (RNN) cell \( h_c^\theta \).

1. **Encoder**: The encoder \( f_\theta \) transforms an observation \( o_t \) into a latent representation \( f_\theta(o_t) \in \mathbb{R}^N \), where \( N \) is the embedding size.
2. **RNN Cell**: The RNN cell \( h_c^\theta \) processes the latent representation and the previous action to compute a representation of the history \( b_t \in \mathbb{R}^M \), where \( M \) is the history representation size.

The target network is an observation encoder \( f_\phi \) whose parameters are an exponential moving average (EMA) of the online network's parameters \( \theta \). The target network outputs targets \( f_\phi(o_{t+k}) \in \mathbb{R}^N \) that are used to train the online network.

#### 2.3. Loss Function

The loss function \( L_{\text{BYOL-Explore}}(\theta) \) is defined as the average cosine distance between the open-loop future predictions \( g_\theta(b_{j,t,k}) \) and their respective targets \( f_\phi(o_{j,t+k}) \):
\[ L_{\text{BYOL-Explore}}(\theta) = \frac{1}{B(T-1)} \sum_{j=0}^{B-1} \sum_{t=0}^{T-2} \sum_{k=1}^{K(t)} \left( 1 - \frac{g_\theta(b_{j,t,k}) \cdot f_\phi(o_{j,t+k})}{\|g_\theta(b_{j,t,k})\| \|f_\phi(o_{j,t+k})\|} \right) \]
where \( K(t) = \min(K, T-1-t) \) is the valid open-loop horizon for a trajectory of length \( T \), and \( \text{sg} \) is the stop-gradient operator.

#### 2.4. World Model Uncertainties

The uncertainty associated with a transition \( (o_j^t, a_j^t, o_j^{t+1}) \) is the sum of the corresponding prediction losses:
\[ \mathcal{L}_j^t = \sum_{k=1}^{K(t)} \left( 1 - \frac{g_\theta(b_{j,t,k}) \cdot f_\phi(o_{j,t+k})}{\|g_\theta(b_{j,t,k})\| \|f_\phi(o_{j,t+k})\|} \right) \]

These uncertainties are used as intrinsic rewards to guide the agent's exploration.

### 3. Reward Normalization and Prioritization Scheme

#### 3.1. Reward Normalization

To counter the non-stationarity of the intrinsic rewards during training, a reward normalization scheme is used. The raw intrinsic rewards \( \mathcal{L}_j^t \) are divided by an EMA estimate of their standard deviation \( \sigma_r \):
\[ r_j^t = \frac{\mathcal{L}_j^t}{\sigma_r} \]

#### 3.2. Reward Prioritization

In addition to normalization, the intrinsic rewards can be prioritized to focus on the most uncertain parts of the environment. This is achieved by clipping the normalized rewards:
\[ r_j^t = \max\left( \frac{\mathcal{L}_j^t}{\sigma_r} - \mu_r, 0 \right) \]
where \( \mu_r \) is the adjusted EMA mean of the normalized rewards.

### 4. Generic RL Algorithm and Representation Sharing

#### 4.1. Integration with RL Algorithms

BYOL-Explore can be used in conjunction with any RL algorithm for training the policy. The intrinsic rewards generated by the world model are combined with the extrinsic rewards to form a mixed reward function:
\[ r_t = r_e^t + \lambda r_i^t \]
where \( \lambda \) is a mixing parameter that balances the importance of intrinsic and extrinsic rewards.

#### 4.2. Representation Sharing

To further improve performance, components of the BYOL-Explore world model can be shared with the RL model. Specifically, the encoder \( f_\theta \) and the RNN cell \( h_c^\theta \) can be shared, allowing the joint representation to be trained via both the RL loss and the BYOL-Explore loss.

### 5. Experimental Evaluation

#### 5.1. Benchmark Tasks

BYOL-Explore has been evaluated on a variety of benchmark tasks, including:
- **Atari Learning Environment**: A suite of 50 Atari games, with a focus on the 10 hardest exploration games.
- **DM-HARD-8**: A suite of 8 complex 3D tasks with sparse rewards, requiring efficient exploration to reach the final goal.

#### 5.2. Performance Metrics

Performance is evaluated using the agent score, defined as the undiscounted episode return. The highest agent score through training is denoted as:
\[ \text{Agent score} = \max_t \text{Agent score}(t) \]

Human Normalized Score (HNS) is defined as:
\[ \text{HNS}(t) = \frac{\text{Agent score}(t) - \text{Random score}}{\text{Human score} - \text{Random score}} \]
where \( \text{Human score} \) is the score achieved by a human player, and \( \text{Random score} \) is the score achieved by a random policy.

#### 5.3. Results

- **Atari Games**: BYOL-Explore achieves superhuman performance on the 10 hardest exploration games, outperforming other methods such as Random Network Distillation (RND) and Intrinsic Curiosity Module (ICM).
- **DM-HARD-8**: BYOL-Explore significantly outperforms other methods on the DM-HARD-8 tasks, achieving human-level performance on the majority of the tasks without the need for human demonstrations.

### 6. Conclusion

BYOL-Explore is a simple yet effective curiosity-driven exploration algorithm that leverages self-supervised learning to enhance the exploration capabilities of reinforcement learning agents. By using a unified loss function for both learning the world model and training the exploration policy, BYOL-Explore ensures a coherent and efficient learning process. The algorithm has demonstrated superior performance on a variety of benchmark tasks, making it a valuable tool for advancing the field of reinforcement learning.