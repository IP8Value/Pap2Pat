# DESCRIPTION

## CROSS REFERENCE

This application claims the benefit of U.S. Provisional Application No. 63/XXXXXXX, filed on [DATE], which is incorporated herein by reference in its entirety.

## FIELD

The present disclosure relates generally to the field of machine learning and, more specifically, to a novel method for modeling stochastic dynamical systems with interacting agents using Graph Deep State-Space Models (GDSSMs). The disclosed method provides a deterministic and efficient approach for generating expressive multi-modal predictive distributions over future trajectories, particularly useful in applications such as autonomous driving.

## BACKGROUND INFORMATION

Many dynamical systems, such as traffic flow, fluid dynamics, and human motion, involve interactions between agents. Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling these systems, allowing the learning of dynamics from data. However, for many real-world applications, predicting a single future trajectory for each agent is insufficient due to the inherent stochasticity in the system. For example, in autonomous driving, the driver's intention (e.g., overtaking, turning, lane changing) introduces significant variability in possible future trajectories.

Recent research has focused on developing methods to model deterministic complex systems, but these methods often struggle with the computational complexity and memory requirements when dealing with multi-agent systems. Probabilistic history-based methods, while reducing the sampling overhead, can be computationally intensive and may require complex models that are not suitable for embedded systems with limited memory capacity.

There is a need for a method that can efficiently and deterministically model stochastic dynamical systems with interacting agents, generating expressive multi-modal predictive distributions over future trajectories. The present invention addresses this need by introducing a novel Graph Deep State-Space Model (GDSSM) that leverages GNNs and deterministic moment matching to achieve this goal.

## SUMMARY

The present invention provides a method for modeling stochastic dynamical systems with interacting agents using Graph Deep State-Space Models (GDSSMs). The method includes the following steps:

1. **Model Definition**: Define a GDSSM where the shared dynamics of all agents are modeled in a joint latent space using GNNs. The model belongs to the family of recurrent neural networks, and the expensive Monte Carlo (MC) operations during training and testing are replaced by a novel deterministic moment matching scheme.

2. **Initial Latent State Distribution**: Place a Gaussian Mixture Model (GMM) over the initial latent states to result in multimodal predictive distributions over future trajectories. The initial latent state is often estimated from historical information and can provide information about the agents' intentions.

3. **Moment Matching Rules**: Derive output moments for GNN layers, making GNNs applicable to moment matching algorithms. This leads to the first deterministic inference scheme for deep state-space models for interacting systems.

4. **Structured Covariance Approximations**: Propose structured approximations to the GMM covariance matrices to reduce the computational complexity from \(O(M^3)\) to \(O(M^2)\), where \(M\) is the number of agents.

The method is particularly useful in applications such as autonomous driving, where the ability to predict multiple modes of future trajectories while accounting for interactions between traffic participants is crucial. The deterministic nature of the method ensures efficient and stable performance, making it suitable for real-time applications with limited computational resources.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

### Model Definition

The Graph Deep State-Space Model (GDSSM) is designed to model stochastic dynamical interactions between agents following complex behavioral patterns, such as road traffic interactions. The model extends deep state-space models to interacting systems by using graph neural networks (GNNs) in the transition model to capture interactions between agents.

#### State Representation

For \(M\) interacting agents, the state of agent \(m\) at time step \(t\) is denoted as \(x_m^t \in \mathbb{R}^{D_x}\), and the set of all state variables is \(x^t = \{x_m^t\}_{m=1}^M\). The dynamics of \(x^t\) follow the transition model:

\[
p(x^{t+1} | x^t, I) = \mathcal{N}(f(x^t, I), L(x^t, I))
\]

where:
- \(f(x^t, I)\) is the mean update function parameterized by a GNN.
- \(L(x^t, I)\) is the variance update function parameterized by another neural network.
- \(I\) is the context variable containing relational information and historical data.

#### Transition Model

The mean update function \(f(x^t, I)\) and variance update function \(L(x^t, I)\) are defined as:

\[
f(x^t, I) = \text{GNN}(x^t, I)
\]
\[
L(x^t, I) = \text{NN}(x^t, I)
\]

where \(\text{GNN}\) and \(\text{NN}\) are neural networks. The GNN operation updates the position and velocity information of each agent by taking information from adjacent traffic participants into account.

### Initial Latent State Distribution

The initial latent state \(x^0\) is modeled as a Gaussian Mixture Model (GMM):

\[
p(x^0 | I) = \sum_{v=1}^V \pi_v(I) \mathcal{N}(\mu_v(I), \Sigma_v(I))
\]

where:
- \(\pi_v(I)\) are the mixture weights.
- \(\mu_v(I)\) and \(\Sigma_v(I)\) are the mean and covariance of the \(v\)-th mixture component.
- \(V\) is the number of mixture components.

The mixture weights \(\pi_v(I)\) are determined by an embedding function \(h(I)\) that processes the context variable \(I\):

\[
\pi_v(I) = \text{softmax}(h(I))
\]

### Moment Matching Rules

To enable deterministic inference, the method derives output moments for GNN layers, allowing the propagation of moments through the neural network layers. The mean and covariance of the latent state at time step \(t\) are computed as:

\[
\mu_t(I) = E[f(x^{t-1}, I)]
\]
\[
\Sigma_t(I) = \text{Cov}[f(x^{t-1}, I)] + E[L(x^{t-1}, I)]
\]

The cross-covariance term \(\text{Cov}[x^{t-1}, f(x^{t-1}, I)]\) is approximated using Stein's Lemma:

\[
\text{Cov}[x^{t-1}, f(x^{t-1}, I)] = E[x^{t-1} \nabla_{x^{t-1}} f(x^{t-1}, I)^T]
\]

### Structured Covariance Approximations

For large numbers of agents \(M\), the computational complexity of the method can become prohibitive. To address this, the method proposes structured approximations to the GMM covariance matrices:

1. **Full Covariance**: Model the full covariance matrix.
2. **Main Diagonal**: Keep the diagonal entries in the covariance blocks.
3. **Main Blocks**: Keep the block-diagonal blocks in the covariance matrix.
4. **All Diagonals**: Structure the covariance matrix in blocks of shape \(M \times M\) and keep the diagonal entries in each block.

These approximations reduce the computational complexity from \(O(M^3)\) to \(O(M^2)\), making the method feasible for real-world applications with a large number of agents.

### Training and Inference

The model is trained by maximizing the predictive log-likelihood (PLL):

\[
\text{PLL}(y_1, \ldots, y_T | I) = \sum_{t=1}^T \log p(y_t | I)
\]

where \(y_t\) are the observed variables. The PLL is approximated using bidimensional moment matching (BMM), which combines horizontal moment matching along the time axis with vertical moment matching across the neural network layers.

During inference, the method propagates the latent state forward in time using the derived moment matching rules, resulting in a deterministic and efficient prediction process.

### Experimental Results

The method is evaluated on two challenging autonomous driving datasets: rounD and NGSIM. The results demonstrate that the deterministic model has strong empirical performance compared to state-of-the-art alternatives. The method is capable of predicting multiple modes of future trajectories, taking interactions into account using GNNs.

An ablation study examines the impact of individual contributions, such as the number of mixture components and the choice of covariance approximations. The findings indicate that sparse covariance approximations reduce the computational complexity by a factor of up to 100, making them favorable for applications with limited computational resources.

### Conclusion

The present invention provides a novel method for modeling stochastic dynamical systems with interacting agents using Graph Deep State-Space Models (GDSSMs). The method leverages GNNs and deterministic moment matching to generate expressive multi-modal predictive distributions over future trajectories efficiently and deterministically. The structured covariance approximations ensure that the method remains computationally feasible even for large numbers of agents. The strong empirical performance on challenging autonomous driving datasets demonstrates the effectiveness and practical utility of the proposed method.