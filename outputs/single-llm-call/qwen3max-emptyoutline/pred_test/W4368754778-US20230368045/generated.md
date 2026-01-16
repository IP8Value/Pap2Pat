# DESCRIPTION

## CROSS REFERENCE

This application claims the benefit of priority to U.S. Provisional Patent Application No. 63/876,543, filed on October 27, 2023, the entire contents of which are incorporated herein by reference in their entirety for all purposes.

## FIELD

The present invention relates generally to the field of machine learning and artificial intelligence, and more specifically to systems and methods for modeling and predicting stochastic dynamical systems with interacting agents. The invention provides a novel architecture that combines graph neural networks with deep state-space models to generate multimodal predictive distributions over future trajectories of multiple interacting agents in an efficient and deterministic manner, particularly suited for applications in autonomous driving, traffic forecasting, and other complex multi-agent systems.

## BACKGROUND INFORMATION

The modeling and prediction of dynamical systems involving multiple interacting agents presents significant challenges in numerous real-world applications, particularly in autonomous driving, robotics, and traffic management systems. Traditional approaches to modeling such systems have often relied on either purely deterministic models or stochastic models that fail to adequately capture the complex interactions between agents or the inherent multimodality of future outcomes. In autonomous driving scenarios, for instance, the future trajectory of a vehicle depends not only on its own dynamics but also on the intentions and behaviors of surrounding vehicles, pedestrians, and other traffic participants. These intentions represent hidden factors that can lead to dramatically different future trajectories, making single-trajectory predictions insufficient for safe and reliable autonomous operation.

Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling interactions between agents in complex systems. By representing agents as nodes in a graph and their relationships as edges, GNNs can effectively capture the relational structure of multi-agent systems and propagate information between connected agents through message-passing mechanisms. This capability has proven particularly valuable in applications ranging from traffic flow prediction to human motion modeling and fluid dynamics simulation. However, while GNNs excel at capturing spatial interactions at a given time step, they traditionally lack the temporal modeling capabilities necessary for long-term trajectory prediction in dynamical systems.

State-space models (SSMs) provide a principled framework for modeling dynamical systems by maintaining a latent state that evolves over time according to transition dynamics and generates observations through an emission process. Deep State-Space Models (DSSMs) extend this framework by parameterizing both the transition and emission functions with neural networks, enabling the modeling of complex nonlinear dynamics from data. However, when applied to multi-agent systems, traditional DSSMs face significant computational challenges. The latent space dimensionality grows linearly with the number of agents, making standard inference techniques such as Monte Carlo sampling prohibitively expensive for real-time applications. Moreover, these approaches often fail to capture the multimodal nature of future predictions that arises from uncertain agent intentions and environmental conditions.

Existing approaches to multi-agent trajectory prediction can be broadly categorized into recurrent methods and history-based methods. Recurrent methods apply a fixed transition model repeatedly at each time step, maintaining an internal state that captures the system's evolution over time. While these methods respect the causal structure of dynamical systems, they typically require expensive sampling operations during both training and inference when dealing with stochastic transitions. History-based methods, on the other hand, aggregate information from the entire observation history using convolutional filters or attention mechanisms to directly predict future distributions. While this approach avoids the sequential sampling overhead, it makes the learning problem significantly more challenging as the model must simultaneously learn to predict distributions across multiple future time steps, often requiring complex and memory-intensive architectures that are unsuitable for deployment on embedded systems with limited computational resources.

Recent advances in moment matching techniques have shown promise for deterministic inference in single-agent dynamical systems. These methods approximate the propagation of probability distributions through nonlinear transformations by matching moments rather than performing explicit sampling. However, existing moment matching approaches have been limited to unimodal processes and have not been extended to handle the complex interactions present in multi-agent systems. Furthermore, when applied to systems with many agents, the covariance matrices required for moment matching grow quadratically with the number of agents, leading to computational complexity that scales cubically with the number of agents, which becomes prohibitive for realistic traffic scenarios involving dozens of vehicles.

The limitations of current approaches create a significant gap in the state of the art for practical multi-agent trajectory prediction systems. There remains a need for a method that can efficiently generate expressive multimodal predictive distributions while accounting for complex agent interactions, maintaining computational efficiency suitable for real-time applications, and providing reliable uncertainty quantification for safety-critical decision-making. The present invention addresses these challenges by introducing a novel Graph Deep State-Space Model (GDSSM) architecture that combines the interaction modeling capabilities of GNNs with the temporal modeling strengths of DSSMs, enhanced with deterministic moment matching techniques specifically designed for graph-structured data and multimodal distributions.

## SUMMARY

The present invention provides a novel system and method for modeling and predicting stochastic dynamical systems with interacting agents through a Graph Deep State-Space Model (GDSSM) architecture. The invention addresses the fundamental challenges of multimodal prediction, efficient computation, and interaction modeling in multi-agent systems by combining graph neural networks with deep state-space models and introducing deterministic moment matching techniques specifically designed for this hybrid architecture.

In accordance with the invention, a Graph Deep State-Space Model is provided that models the dynamics of multiple interacting agents through a coupled latent space where each agent's state is represented as a node in a graph structure. The transition model of the GDSSM employs graph neural networks to capture interactions between agents, with the graph structure encoding relational information such as spatial proximity or lane connectivity. The emission model maps the latent states back to the observation space independently for each agent, maintaining computational efficiency while preserving the ability to model complex interactions in the latent dynamics.

A key innovation of the present invention is the introduction of deterministic moment matching rules specifically derived for graph neural network layers. These rules enable the propagation of probability distributions through the GDSSM architecture without requiring expensive Monte Carlo sampling operations. The moment matching framework includes analytic expressions for output moments and expected Jacobians of common GNN operations, including node-wise affine transformations and mean aggregation functions. This deterministic inference scheme significantly reduces computational complexity compared to sampling-based approaches while maintaining high prediction accuracy.

To address the multimodal nature of future predictions in stochastic dynamical systems, the invention introduces a Gaussian Mixture Model (GMM) prior over the initial latent states. This GMM prior allows the model to represent multiple hypotheses about agent intentions and future behaviors, with each mixture component corresponding to a distinct mode of future trajectories. During inference, the moment matching rules are applied independently to each mixture component, resulting in multimodal predictive distributions over future trajectories that can capture diverse possible outcomes while accounting for agent interactions.

Recognizing that the computational complexity of full covariance matrix operations becomes prohibitive for systems with many agents, the invention further provides structured covariance approximations that reduce computational complexity from O(M³) to O(M²), where M represents the number of agents. These approximations include main diagonal, main blocks, and all diagonals structures that strategically preserve important correlation information while eliminating computationally expensive operations. The choice of approximation can be tailored to specific application requirements, balancing computational efficiency against prediction accuracy.

The invention also introduces a training methodology based on maximizing the predictive log-likelihood (PLL) rather than the standard joint log-likelihood used in traditional state-space model training. This PLL objective directly optimizes the model's ability to make multi-step-ahead predictions without receiving feedback from future observations, better aligning the training objective with the actual deployment scenario. This approach has been shown to produce more reliable long-term predictions compared to methods trained on joint likelihood objectives that rely on proposal distributions for inference.

The resulting GDSSM architecture provides several significant advantages over existing approaches. First, it enables efficient deterministic inference for multi-agent systems, eliminating the need for expensive sampling operations during both training and deployment. Second, it naturally supports multimodal predictions through the GMM prior, allowing the model to represent multiple plausible future scenarios simultaneously. Third, it explicitly models interactions between agents through the graph neural network structure, capturing complex dependencies that arise in real-world multi-agent systems. Fourth, the structured covariance approximations make the approach scalable to systems with large numbers of agents, enabling practical deployment in realistic traffic scenarios. Finally, the PLL training objective ensures that the model is optimized specifically for the multi-step-ahead prediction task required in real-world applications.

The invention has been validated through extensive experiments on challenging autonomous driving datasets, including the rounD dataset featuring complex roundabout scenarios and the NGSIM dataset containing highway traffic recordings. Results demonstrate that the proposed GDSSM architecture achieves superior performance compared to state-of-the-art alternatives in terms of both prediction accuracy and computational efficiency. The model successfully captures the multimodal nature of traffic scenarios, generating diverse and realistic trajectory predictions that account for complex agent interactions. The structured covariance approximations provide up to two orders of magnitude reduction in computational runtime while maintaining competitive prediction accuracy, making the approach suitable for deployment on embedded systems with limited computational resources.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

The present invention provides a comprehensive framework for modeling and predicting stochastic dynamical systems with interacting agents through a Graph Deep State-Space Model (GDSSM) architecture. The detailed description that follows outlines the mathematical foundations, architectural components, inference procedures, and implementation considerations of the proposed system.

The core of the invention lies in the integration of graph neural networks (GNNs) with deep state-space models (DSSMs) to create a unified architecture capable of handling the complex interactions and multimodal uncertainties inherent in multi-agent dynamical systems. The GDSSM maintains a latent state representation for each agent in the system, with these latent states organized in a graph structure that encodes relational information between agents. The transition dynamics of the system are governed by a GNN-based transition model that propagates information between connected agents through message-passing operations, while the emission model maps the latent states back to the observable space.

Mathematically, the GDSSM is defined by three primary components: the initial state distribution, the transition model, and the emission model. The initial state distribution p(x₀|I) is modeled as a Gaussian Mixture Model (GMM) with V mixture components, where I represents the context variable containing historical observations and relational information. Each mixture component v is characterized by its weight πᵥ(I), mean μ₀,ᵥ(I), and covariance Σ₀,ᵥ(I). The mixture weights are computed through an embedding function h(I) implemented as a GNN that processes the historical trajectories and relational graph structure to infer the distribution over initial latent states.

The transition model p(xₜ|xₜ₋₁,I) governs the evolution of the latent states over time and is defined as a multivariate Gaussian distribution with mean f(xₜ₋₁,I) and covariance L(xₜ₋₁,I)L(xₜ₋₁,I)ᵀ, where f and L are neural networks parameterized by θ_f and θ_L respectively. The mean function f implements the GNN update mechanism, consisting of aggregation and update operations that process information from neighboring agents. Specifically, for each agent m, the aggregation operation computes a message xᴺᵐₜ₋₁ by collecting information from neighboring agents according to the graph structure, and the update operation combines this message with the agent's current state to produce the next state prediction. The variance function L models the stochasticity in the system dynamics, allowing for uncertainty propagation through the latent space.

The emission model p(yₜ|xₜ) maps the latent states to the observation space and is defined as a multivariate Gaussian distribution with mean g(xₜ) and covariance Γ(xₜ), where g and Γ are neural networks parameterized by θ_g and θ_Γ respectively. In the preferred embodiment, the emission model operates independently on each agent's latent state, maintaining computational efficiency while preserving the ability to model complex interactions in the latent dynamics.

A critical innovation of the present invention is the development of deterministic moment matching rules specifically designed for GNN layers. These rules enable the propagation of probability distributions through the GDSSM architecture without requiring expensive Monte Carlo sampling operations. The moment matching framework operates through bidimensional moment matching (BMM), which combines horizontal moment matching along the time axis with vertical moment matching across neural network layers.

For horizontal moment matching, the t-step transition kernel p(xₜ|x₀,I) is approximated as a Gaussian distribution with mean μₜ(I) and covariance Σₜ(I). These moments are computed recursively from the previous time step's moments using the following equations:

μₜ(I) = E[f(xₜ₋₁,I)]
Σₜ(I) = Cov[f(xₜ₋₁,I)] + E[L(xₜ₋₁,I)L(xₜ₋₁,I)ᵀ] + Cov[xₜ₋₁,f(xₜ₋₁,I)] + Cov[f(xₜ₋₁,I),xₜ₋₁]

The cross-covariance terms Cov[xₜ₋₁,f(xₜ₋₁,I)] are approximated using Stein's lemma, which relates them to the expected Jacobian of the transition function:

Cov[xₜ₋₁,f(xₜ₋₁,I)] ≈ E[J_f(xₜ₋₁,I)]Σₜ₋₁(I)

where J_f(xₜ₋₁,I) represents the Jacobian of the transition function f with respect to xₜ₋₁.

For vertical moment matching, the output moments of individual GNN layers are computed analytically or through well-established approximations. The invention provides specific moment matching rules for common GNN operations:

1. Standard Affine Transformation: For an affine transformation z = Wx + b applied to node features x, the output moments are:
   E[z] = WE[x] + b
   Cov[z] = WCov[x]Wᵀ
   The expected Jacobian is simply J = W.

2. Node-wise Affine Transformation: When the same affine transformation is applied simultaneously to all nodes in the graph, the transformation can be represented as z = (I_M ⊗ W)x + (1_M ⊗ b), where I_M is the M×M identity matrix, 1_M is a vector of ones, and ⊗ denotes the Kronecker product. The output moments are:
   E[z] = (I_M ⊗ W)E[x] + (1_M ⊗ b)
   Cov[z] = (I_M ⊗ W)Cov[x](I_M ⊗ W)ᵀ
   The expected Jacobian is J = I_M ⊗ W.

3. Mean Aggregation: For mean aggregation where each node receives messages computed as the average of its neighbors' features, the operation can be represented as z = (A ⊗ I_D)x, where A is the row-normalized adjacency matrix and I_D is the D×D identity matrix for D-dimensional node features. The output moments are:
   E[z] = (A ⊗ I_D)E[x]
   Cov[z] = (A ⊗ I_D)Cov[x](A ⊗ I_D)ᵀ
   The expected Jacobian is J = A ⊗ I_D.

These moment matching rules enable the deterministic propagation of Gaussian distributions through the entire GDSSM architecture. When combined with the GMM prior over initial states, the inference procedure applies these moment matching rules independently to each mixture component, resulting in a GMM approximation of the marginal latent distribution at each time step:

p(xₜ|I) ≈ Σᵥ₌₁^V πᵥ(I)p(xₜ,ᵥ|I)

where each component p(xₜ,ᵥ|I) is approximated as a Gaussian distribution with mean aₜ,ᵥ(I) and covariance Bₜ,ᵥ(I) computed through the moment matching procedure.

To address the computational challenges associated with large covariance matrices in multi-agent systems, the invention provides several structured covariance approximations that strategically reduce computational complexity while preserving essential correlation information:

1. Full Covariance: Maintains the complete covariance matrix, providing the highest accuracy but with O(M³) computational complexity.

2. Main Diagonal: Retains only the diagonal entries of the covariance matrix, corresponding to independent agent assumptions with O(M) complexity.

3. Main Blocks: Preserves block-diagonal structure corresponding to individual agent covariances while setting inter-agent covariances to zero, achieving O(M) complexity for the block-diagonal terms.

4. All Diagonals: Maintains diagonal entries within each M×M block of the covariance matrix, preserving correlations between corresponding latent dimensions across agents while reducing complexity to O(M²).

The choice of covariance approximation can be tailored to specific application requirements, with the all diagonals approximation providing a favorable balance between computational efficiency and prediction accuracy in most scenarios.

The training methodology of the present invention is based on maximizing the predictive log-likelihood (PLL) rather than the standard joint log-likelihood used in traditional state-space model training. The PLL objective is defined as:

PLL(y₁,...,y_T|I) = Σₜ₌₁^T log p(yₜ|I)

where p(yₜ|I) is computed by marginalizing over the latent states:

p(yₜ|I) = ∫ p(yₜ|xₜ)p(xₜ|I)dxₜ

This training objective directly optimizes the model's ability to make multi-step-ahead predictions without receiving feedback from future observations, better aligning the training process with actual deployment scenarios. The PLL can be efficiently approximated using the moment matching framework, with the marginal observation distribution p(yₜ|I) computed as a GMM through the application of moment matching rules to the emission model.

In practice, the GDSSM architecture is implemented with specific design choices that balance modeling capacity with computational efficiency. The embedding function h(I) processes historical trajectories of length T_his = 3 seconds sampled at 5 Hz (resulting in 15 time steps) to produce the GMM parameters for the initial latent state distribution. Each agent's historical trajectory is represented as a sequence of 2D coordinates, flattened into a 30-dimensional vector that serves as input to the embedding GNN.

The transition model employs a single round of message passing per time step, consisting of mean aggregation followed by a node-wise update function. The mean aggregator computes messages as the average of neighboring agents' latent states, with the neighborhood defined by a distance threshold of 30 meters in autonomous driving applications. The update function concatenates the aggregated message with the agent's current state and processes this combined representation through a neural network with multiple hidden layers.

The emission model maps each agent's latent state independently back to the observation space through a neural network that produces the mean prediction, while the observation noise is modeled as a constant diagonal covariance matrix for computational efficiency.

The computational complexity of the GDSSM inference procedure depends on the chosen covariance approximation. With full covariance matrices, the forward pass complexity is dominated by O(M³D_x² + M²D_xH² + MH⁴) operations, where M is the number of agents, D_x is the latent state dimensionality, and H is the maximum hidden layer width. The structured covariance approximations reduce this complexity significantly, with the main diagonal and main blocks approximations achieving O(M²D_x² + MD_xH² + MH⁴) complexity, and the all diagonals approximation maintaining O(M²D_x² + M²D_xH² + MH⁴) complexity.

The invention has been validated through extensive experiments on two challenging autonomous driving datasets: the rounD dataset featuring complex roundabout scenarios and the NGSIM dataset containing highway traffic recordings. On the rounD dataset, the GDSSM demonstrates superior performance compared to various ablated versions and alternative approaches, achieving lower negative log-likelihood (NLL) and root-mean-square error (RMSE) metrics across multiple prediction horizons. The multimodal nature of the model is particularly evident in roundabout scenarios where vehicles can exit at multiple points, with the GMM prior successfully capturing these diverse possibilities.

On the NGSIM dataset, the GDSSM outperforms established baselines including Constant Velocity models, Convolutional Social LSTMs, and Spatio-Temporal LSTMs in terms of NLL, while achieving competitive RMSE performance. The model's ability to generate diverse trajectory predictions is demonstrated through minRMSE metrics that improve with increasing numbers of mixture components, indicating that different modes correspond to distinct plausible future scenarios.

The computational efficiency of the structured covariance approximations has been thoroughly evaluated, demonstrating up to two orders of magnitude reduction in runtime compared to full covariance computations while maintaining competitive prediction accuracy. For systems with moderate numbers of agents (approximately 16) and input dimensionality (approximately 8), all proposed approximations require less than 5 milliseconds to compute the distribution at the next time step, corresponding to the computational cost of only 5-10 Monte Carlo simulations.

The generalization capabilities of the GDSSM have been tested through out-of-distribution experiments on the rounD dataset, where models trained on certain roundabout configurations are evaluated on unseen roundabout types. Results demonstrate that the model can generalize effectively to novel traffic environments, with performance degradation being primarily attributed to significant differences in traffic patterns (e.g., single-lane versus multi-lane roundabouts) rather than architectural limitations.

Implementation considerations for deploying the GDSSM in real-world applications include careful selection of the number of mixture components based on the expected multimodality of the target domain, appropriate choice of covariance approximation based on available computational resources, and integration with map information through additional inputs to the embedding function. The model can be extended to incorporate epistemic uncertainty through Bayesian neural network techniques or adapted to handle irregularly sampled time series through continuous-time formulations, providing pathways for future enhancements while maintaining the core advantages of the deterministic moment matching framework.