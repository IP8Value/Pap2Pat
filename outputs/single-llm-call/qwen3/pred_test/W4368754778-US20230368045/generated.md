## CROSS REFERENCE

- claim benefit of german patent application

This invention claims the benefit of priority under 35 U.S.C. §119 of German Patent Application No. DE 10 2023 000 000 000, filed on [Insert Filing Date], the entire disclosure of which is hereby incorporated by reference in its entirety. The present application is directed to a computer-implemented method for predicting the future behavior of interacting agents in dynamic systems, wherein the method leverages deterministic moment matching within a graph-based deep state-space framework to efficiently compute multimodal predictive distributions over future trajectories. The invention builds upon the technical disclosures and algorithmic innovations described in the referenced German application, particularly with respect to the recursive propagation of first and second moments through graph neural network layers, the structured approximation of covariance matrices under interaction constraints, and the integration of context-dependent initial latent state distributions modeled as Gaussian mixtures. The German application provided the foundational architecture for modeling latent dynamics in multi-agent environments using recurrent neural networks with graph-structured dependencies, and this patent application extends that disclosure by formalizing the computational procedures for determining predictive likelihoods without reliance on stochastic sampling, thereby enabling real-time deployment in resource-constrained systems such as autonomous vehicle control units. The technical solution disclosed herein is not merely an incremental improvement but represents a novel integration of deterministic inference rules with graph-based latent state modeling, which was neither explicitly taught nor suggested in the prior art referenced in the German filing. All claims herein are supported by the detailed descriptions, mathematical derivations, and experimental validations contained in the earlier application, and no new matter has been introduced beyond the scope of the original disclosure.

## FIELD

- relate to computer-implemented method

This invention relates to a computer-implemented method for predicting the future behavior of agents in a dynamic system characterized by interactions among multiple entities, wherein the prediction is performed using a deterministic, moment-based inference procedure over a latent state space modeled by graph neural networks. The method operates on observed trajectories of agents collected over time, processes these observations through a structured computational framework that encodes relational dependencies between agents as a graph, and computes probabilistic forecasts of future states without resorting to Monte Carlo sampling or particle filtering. The system is particularly suited for applications in autonomous driving, traffic flow modeling, crowd dynamics, and robotic multi-agent coordination, where accurate, real-time, and multimodal predictions of agent behavior are essential for safe and efficient decision-making. The method is implemented entirely in software running on one or more processors, with inputs received from sensor systems, map data, or historical trajectory logs, and outputs delivered as probability distributions over future positions, velocities, or accelerations of each agent. The core innovation lies in the replacement of stochastic approximation techniques with deterministic moment propagation rules that preserve the expressive power of multimodal distributions while maintaining computational tractability even in systems with large numbers of interacting agents. The method is not a general-purpose machine learning algorithm but a specialized inference engine tailored to the mathematical structure of interacting dynamical systems governed by latent state transitions and emission models parameterized by neural networks.

## BACKGROUND INFORMATION

- describe possibilities of predicting behavior

The prediction of future behavior in complex dynamical systems involving multiple interacting agents has long been a challenge in fields ranging from robotics to transportation engineering. Traditional approaches have relied on deterministic models that output a single predicted trajectory per agent, neglecting the inherent uncertainty and multimodal nature of human or vehicular decision-making. Other methods have employed probabilistic frameworks based on Monte Carlo sampling, which generate many possible futures by simulating stochastic transitions through latent state spaces. While these methods can capture uncertainty, they suffer from high computational cost, slow convergence, and poor scalability when the number of agents increases, as the dimensionality of the latent space grows linearly with the number of entities. Furthermore, many existing models fail to account for the relational structure between agents, treating them as independent entities or applying simplistic spatial proximity rules that do not reflect the nuanced nature of interactions such as lane-changing intent, yielding behavior, or cooperative maneuvers. Even advanced deep learning architectures that incorporate attention mechanisms or convolutional grids struggle to maintain accurate long-term predictions due to error accumulation over time and the absence of a principled mechanism for propagating uncertainty through nonlinear dynamics. The inability to efficiently compute full predictive distributions—particularly those with multiple modes corresponding to distinct behavioral hypotheses—limits the reliability of decision-making systems in safety-critical environments. Previous attempts to approximate these distributions using Gaussian assumptions have been insufficient for capturing the rich, context-dependent multimodality observed in real-world scenarios, such as vehicles deciding whether to enter or exit a roundabout based on the actions of neighboring agents. Consequently, there exists a significant gap between the theoretical expressiveness of probabilistic models and their practical feasibility in real-time embedded systems.

## SUMMARY

- achieve precise prediction of behavior

The invention achieves precise prediction of the future behavior of agents in a dynamic system by computing a multimodal probability distribution over their latent states and observed trajectories using a deterministic, recursive moment-matching procedure that avoids stochastic sampling. This is accomplished by modeling the system as a graph deep state-space model in which the evolution of latent states is governed by graph neural networks that encode interactions between agents, and the emission of observable quantities is modeled as a Gaussian process conditioned on these latent states. The method ensures that the predictive distribution accurately reflects the underlying uncertainty in agent intentions and environmental context, enabling reliable decision-making even under ambiguous conditions. The prediction is not a single trajectory but a full probability distribution over future positions, velocities, and accelerations, which can be used to assess risk, plan trajectories, and trigger safety interventions with confidence bounds.

- determine value of first moment of first distribution

The method determines the value of the first moment of the initial latent state distribution by applying a graph neural network to the historical trajectory data and contextual information, producing a Gaussian mixture model that represents the most probable latent states of each agent at the beginning of the prediction horizon. Each mixture component corresponds to a distinct behavioral hypothesis, such as a vehicle intending to exit a roundabout or remain within it, and the mean of each component is computed as a function of the aggregated information from neighboring agents via message-passing operations defined over the interaction graph.

- determine value of second moment of first distribution

The value of the second moment of the initial latent state distribution is determined simultaneously with the first moment, using the same graph neural network architecture to compute the covariance matrix of each Gaussian mixture component. The covariance captures the uncertainty in the estimated latent state for each agent under each behavioral hypothesis, and is structured to reflect the relational dependencies between agents as encoded in the graph, ensuring that correlations between agents who are connected by paths in the graph are preserved even in the initial distribution.

- determine expected value for first moment of second distribution

The expected value for the first moment of the second distribution is determined by recursively applying a deterministic moment-matching rule that propagates the mean of each Gaussian mixture component forward through the latent state transition model, which is implemented as a graph neural network. This propagation accounts for the deterministic component of the transition dynamics, which is parameterized by a neural network that maps the current latent state and context to the next latent state, and computes the expected value of the transformed mean using analytical expressions derived from the affine and aggregation layers of the graph neural network.

- determine second moment of third distribution

The second moment of the third distribution is determined by recursively propagating the covariance matrix of each Gaussian mixture component through the latent state transition model, incorporating both the deterministic transformation and the stochastic noise component. This involves computing the expected Jacobian of the transition function with respect to the latent state, and using it to update the covariance matrix according to the law of propagation of uncertainty through nonlinear functions, while preserving the structured sparsity of the covariance matrix to ensure computational efficiency.

- determine sum of third distributions

The sum of the third distributions is determined by combining the predictive Gaussian mixture components obtained after propagating each initial mixture component forward through the entire prediction horizon. The resulting distribution is a weighted sum of individual Gaussian distributions, each corresponding to a distinct behavioral mode, with weights derived from the initial mixture weights and updated through the transition dynamics. This sum represents the final predictive distribution over the observed variables at the target time point and is used to compute likelihoods, confidence intervals, and decision thresholds.

- determine prediction of behavior

The prediction of behavior is determined by evaluating the final predictive distribution over the observed variables, which includes the expected values and variances of position, velocity, and acceleration for each agent at future time points. This distribution is used to generate probabilistic forecasts that reflect the likelihood of different behavioral outcomes, such as lane changes, stops, or accelerations, and is output as a multimodal probability density function that can be sampled or thresholded for downstream control systems.

- recursively determine value of first moment

The value of the first moment is recursively determined by iteratively applying the moment-matching rule at each time step from the initial distribution to the final prediction horizon. At each step, the mean of each mixture component is updated using the expected value of the transition function applied to the previous mean, with the expectation computed analytically over the Gaussian distribution of the prior state, ensuring that the recursive propagation remains deterministic and computationally tractable.

- recursively determine value of second moment

The value of the second moment is recursively determined by iteratively updating the covariance matrix of each mixture component at every time step using the expected Jacobian of the transition function and the covariance of the previous state. This recursive update accounts for both the linearized dynamics of the neural network and the additive stochastic noise, ensuring that the uncertainty in the latent state is propagated forward in a manner consistent with the underlying probabilistic model.

- efficiently determine expected value

The expected value is efficiently determined by leveraging the structured sparsity of the graph neural network’s weight matrices and the analytical closed-form expressions for the moments of affine transformations and mean aggregation operations. This eliminates the need for numerical integration or Monte Carlo sampling, reducing the computational complexity from cubic to quadratic in the number of agents and enabling real-time inference on embedded hardware.

- determine covariance of first moment of second distribution

The covariance of the first moment of the second distribution is determined by computing the cross-covariance between the latent state and its transformed version under the transition function, using Stein’s lemma to express this term in terms of the expected Jacobian matrix. This allows the method to capture how changes in the latent state influence the expected next state, thereby refining the accuracy of the propagated mean.

- determine expected value for second moment of second distribution

The expected value for the second moment of the second distribution is determined by computing the expectation of the squared transition function applied to the latent state, using the known moments of the input distribution and the analytical properties of the neural network layers. This provides the second-order statistics necessary to update the covariance matrix and maintain the fidelity of the Gaussian approximation over time.

- determine second moment of third distribution

The second moment of the third distribution is determined by combining the propagated covariance matrix with the expected value of the emission model’s noise term, ensuring that the final predictive distribution over observable variables accurately reflects both the uncertainty in the latent state and the observation noise inherent in sensor measurements.

- consider context variable

The method considers a context variable that encodes auxiliary information such as historical trajectories, map topology, traffic signals, and relational connectivity between agents. This context is integrated at every stage of the latent state transition and emission models, influencing both the mean and variance of the predictive distribution, and enabling the model to adapt its predictions to environmental conditions and agent-specific histories.

- determine history of dynamic system

The history of the dynamic system is determined by aggregating observed trajectories over a fixed time window preceding the prediction horizon, encoding them as a sequence of position and velocity measurements for each agent, and feeding this sequence into the embedding function of the graph neural network to initialize the latent state distribution. This history serves as the sole input for the predictive model, eliminating the need for real-time sensor fusion during inference.

- consider neighborhood of agents

The neighborhood of agents is considered by constructing a dynamic graph in which each agent is a node and edges are established based on spatial proximity and topological relationships such as lane adjacency or intersection geometry. Message-passing operations over this graph allow information to flow between agents, enabling the model to capture indirect interactions and emergent behaviors that arise from the collective dynamics of the system.

- model latent states of agents

The latent states of agents are modeled as a joint high-dimensional vector that evolves over time according to a recurrent graph neural network, where each agent’s latent state is influenced not only by its own history but also by the latent states of its neighbors. This joint representation allows the model to capture inter-agent dependencies that are not directly observable, such as intent, attention, or coordination, and to propagate these hidden influences forward in time through deterministic moment matching.

- control agent depending on prediction

The agent is controlled depending on the prediction by using the multimodal predictive distribution to evaluate risk, plan trajectories, or trigger safety interventions. For example, if the probability of a collision exceeds a predefined threshold under any of the predicted modes, an emergency braking command is issued, or if the most likely mode indicates a lane change, the vehicle’s steering system is preemptively adjusted to accommodate the anticipated motion.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

- introduce device for predicting behavior of agents

A device for predicting the behavior of agents in a dynamic system comprises a processor configured to execute a computer program that implements a graph deep state-space model with deterministic moment propagation, a memory unit storing historical trajectory data and context information, and an interface for receiving sensor inputs and outputting predictive distributions. The device is operable in real-time environments such as autonomous vehicles, traffic management systems, or robotic fleets, and is designed to operate under strict computational constraints while maintaining high predictive accuracy.

- describe device components

The device components include a central processing unit capable of executing parallelized tensor operations, a non-volatile memory for storing trained neural network parameters and graph structures, a volatile memory for caching intermediate moments during recursive propagation, and input/output interfaces for connecting to external sensors such as LiDAR, radar, cameras, and map databases. The device further includes a communication module for exchanging predictive outputs with control systems or other agents in the environment.

- describe interface and sensor system

The interface and sensor system collect time-series data from multiple agents in the environment, including position, velocity, acceleration, heading, and lane position, at a sampling rate sufficient to resolve dynamic behavior. The sensor system is calibrated to provide measurements in a common coordinate frame, and the interface preprocesses these measurements into a structured format compatible with the graph neural network input, including the construction of adjacency matrices based on spatial proximity and topological constraints.

- describe sensor system capabilities

The sensor system is capable of detecting and tracking multiple agents simultaneously, even under occlusion or low-visibility conditions, by fusing data from heterogeneous sensors and applying temporal smoothing to reduce noise. The system maintains a history of observations for each agent over a fixed time window, typically three seconds, and updates this history continuously as new data arrives, ensuring that the predictive model always operates on the most recent context.

- describe actuator and control command

The actuator and control command module receives the predictive distribution from the processor and generates control signals for the vehicle or agent based on risk assessment criteria. If the probability of a collision exceeds a threshold under any predicted mode, the system issues an emergency brake command. If the most probable mode indicates a lane change, the system initiates a gradual steering maneuver. The control commands are generated in real-time and are constrained by physical limits of the actuation system.

- describe computer program

The computer program is stored in non-volatile memory and executed by the processor to implement the recursive moment-matching algorithm for graph deep state-space models. The program includes modules for initializing the Gaussian mixture distribution from historical data, propagating the first and second moments through the graph neural network layers, computing the expected Jacobian matrices, applying structured covariance approximations, and generating the final predictive distribution over observed variables. The program is optimized for low-latency execution and is designed to run on embedded hardware with limited memory and computational resources.

- introduce dynamic system example

The dynamic system example is a traffic intersection with multiple vehicles approaching from different directions, where each vehicle’s future behavior—such as stopping, turning, or proceeding—is uncertain and depends on the actions of neighboring vehicles. The system is modeled as a graph where each vehicle is a node, and edges connect vehicles that are within a predefined interaction radius or share a lane.

- describe agents in dynamic system

The agents in the dynamic system are vehicles, pedestrians, or other mobile entities whose motion is governed by latent states that encode intentions, attention, and decision-making processes not directly observable from sensor data. Each agent’s latent state evolves over time according to a neural network transition function that incorporates information from neighboring agents via message-passing operations.

- describe roundabout example

The roundabout example illustrates a scenario in which multiple vehicles approach, enter, and exit a circular intersection, with each vehicle’s behavior depending on the actions of others. The predictive model identifies distinct modes corresponding to different exit choices, and the covariance structure captures the correlation between vehicles that are mutually blocking each other’s paths.

- describe observed trajectories

The observed trajectories are sequences of position and velocity measurements recorded over time for each agent, used as input to the embedding function to initialize the latent state distribution. These trajectories are normalized and encoded into a fixed-length representation that preserves temporal order and spatial relationships.

- describe predicted trajectories

The predicted trajectories are not single paths but probability distributions over future positions, velocities, and accelerations for each agent at each future time step. These distributions are multimodal, reflecting the uncertainty in agent intentions, and are visualized as confidence ellipses or probability heatmaps.

- describe Gaussian mixture distribution

The Gaussian mixture distribution is used to model the initial latent state of the system, with each mixture component representing a distinct behavioral hypothesis. The weights of the components are learned from historical data and updated during propagation to reflect the likelihood of each hypothesis given the observed context.

- describe confidence intervals

The confidence intervals are derived from the predictive distribution at each time step by computing the quantiles of the Gaussian mixture, typically the 95% interval, and are used to quantify the uncertainty in the prediction. These intervals are dynamically adjusted based on the complexity of the interaction and the confidence of the model.

- describe prediction of distance, velocity, or acceleration

The prediction of distance, velocity, or acceleration is performed as part of the emission model, which maps the latent state to the observed space. The model outputs the mean and variance of each quantity for each agent, enabling the system to anticipate not only where an agent will be but also how fast it will be moving and whether it is accelerating or decelerating.

- describe autonomous vehicle example

In the autonomous vehicle example, the device is mounted on a self-driving car and uses sensor data from its surroundings to predict the behavior of nearby vehicles and pedestrians. The predictive distribution is used to plan a safe trajectory, adjust speed, and trigger emergency maneuvers when necessary, all while maintaining computational efficiency for real-time operation.

- describe machine learning model

The machine learning model is a graph deep state-space model composed of a graph neural network for the transition dynamics and a neural network for the emission model, both parameterized by learnable weights. The model is trained offline using historical trajectory data and optimized to maximize the predictive log-likelihood of future observations.

- introduce latent variable X

The latent variable X represents the unobserved internal state of each agent, encoding intentions, attention, and decision-making processes that influence future motion. The latent variable evolves over time according to a recurrent neural network and is inferred from observed trajectories.

- describe observed variable Y

The observed variable Y represents the measurable quantities such as position, velocity, and acceleration of each agent, which are emitted by the latent variable through a Gaussian emission model. The observed variable is the only input available to the system during inference.

- describe initial value for latent state

The initial value for the latent state is determined by applying a graph neural network to the historical trajectory data and context, producing a Gaussian mixture distribution that represents the most probable initial latent states of each agent.

- describe deterministic change in latent state

The deterministic change in the latent state is modeled by a neural network that maps the current latent state and context to the next latent state, representing the predictable, rule-based component of agent motion such as inertia or lane-following behavior.

- describe stochastic change in latent state

The stochastic change in the latent state is modeled by a neural network that outputs a noise covariance matrix, representing the unpredictable component of agent motion such as hesitation, distraction, or sudden maneuvers.

- describe context variable I

The context variable I encodes auxiliary information such as map topology, traffic signals, historical trajectories, and relational connectivity between agents, and is integrated into both the transition and emission models to condition the predictions on environmental and behavioral context.

- describe normal distribution for observed variable

The observed variable is modeled as a multivariate normal distribution whose mean is a function of the latent state and whose covariance is a constant or learned parameter, ensuring that the emission model is tractable and differentiable.

- describe neural network for deterministic change

The neural network for deterministic change is a graph neural network that performs message-passing over the interaction graph, aggregating information from neighboring agents and updating each agent’s latent state using a sequence of affine transformations and nonlinear activations.

- describe neural network for stochastic change

The neural network for stochastic change is a graph neural network that outputs a diagonal or structured covariance matrix for the latent state transition noise, ensuring that the uncertainty in the prediction is spatially and temporally coherent.

- describe neural network for observed variable

The neural network for the observed variable is a simple feedforward network that maps the latent state to the observed space, producing the mean and variance of position, velocity, and acceleration for each agent.

- describe update to deterministic change

The update to the deterministic change is performed recursively at each time step by applying the graph neural network to the current latent state and context, computing the expected value of the next state using analytical moment-matching rules.

- describe update to stochastic change

The update to the stochastic change is performed recursively by computing the expected Jacobian of the deterministic transition function and combining it with the noise covariance to update the latent state covariance matrix.

- describe message for agent

The message for each agent is computed by aggregating the latent states of its neighbors using a mean aggregation operation, and is concatenated with the agent’s own latent state before being passed through the update network.

- describe operation AGG

The operation AGG is a permutation-invariant aggregation function, such as the mean or sum, that collects information from neighboring agents and is applied uniformly across all nodes in the graph to ensure that the model is invariant to agent ordering.

- describe edges of graph

The edges of the graph are defined based on spatial proximity and topological relationships, such as lane adjacency or intersection geometry, and are dynamically updated at each time step to reflect changes in agent positions.

- describe prediction for prediction time point T

The prediction for prediction time point T is the final Gaussian mixture distribution over the observed variables, computed after recursively propagating the initial distribution forward through T time steps using deterministic moment matching.

- describe marginal probability p(yT|I)

The marginal probability p(yT|I) is the predictive distribution over the observed variables at time T, computed as the weighted sum of the emissions from each Gaussian mixture component, and represents the model’s best estimate of what will be observed at the prediction horizon.

- describe kernel p(xT|x0,I)

The kernel p(xT|x0,I) is the transition probability from the initial latent state to the latent state at time T, computed recursively using the moment-matching rules and representing the evolution of uncertainty over time.

- describe Gaussian mixture model p(x0|I)

The Gaussian mixture model p(x0|I) is the initial distribution over the latent state, parameterized by a graph neural network that maps the context variable I to a set of mixture weights, means, and covariances.

- describe mean value and covariance

The mean value and covariance of each mixture component are computed analytically at each time step using closed-form expressions derived from the affine and aggregation operations in the graph neural network, ensuring deterministic and efficient propagation.

- describe expected value and covariance

The expected value and covariance of the latent state at each time step are computed recursively using the law of total expectation and the law of total variance, ensuring that the predictive distribution remains calibrated and consistent with the underlying probabilistic model.

- describe cross-covariance

The cross-covariance between the latent state and its transformed version is computed using Stein’s lemma, which expresses the cross-covariance in terms of the expected Jacobian of the transition function, enabling accurate propagation of uncertainty.

- describe Jacobi matrix

The Jacobi matrix is the expected gradient of the transition function with respect to the latent state, computed analytically for each layer of the graph neural network, and is used to update the covariance matrix during propagation.

- describe method for prediction p(yT|I)

The method for prediction p(yT|I) involves initializing a Gaussian mixture distribution from historical data, recursively propagating each component forward using deterministic moment matching, and summing the resulting emissions to obtain the final predictive distribution.

- describe inner loop and outer loop

The inner loop performs the recursive propagation of moments through time, while the outer loop iterates over the mixture components, ensuring that each behavioral hypothesis is treated independently and its uncertainty is preserved throughout the prediction horizon.

- describe neural network for moments

The neural network for moments is a shared architecture that computes the mean, covariance, and Jacobian of the transition and emission functions, and is trained end-to-end to maximize the predictive log-likelihood of observed trajectories.

- describe context variable I and association Nm

The context variable I includes the association Nm, which defines the set of neighbors for each agent m, and is used to construct the graph structure and guide the message-passing operations.

- define trajectories

Trajectories are defined as sequences of observed position and velocity measurements over time for each agent, used as input to the embedding function and as ground truth for training the model.

- describe operation AGG

The operation AGG is implemented as a mean aggregation function that computes the average latent state of all neighbors for each agent, ensuring that information flows uniformly across the graph.

- motivate graph neural network

The graph neural network is motivated by the need to model interactions between agents in a way that is both expressive and computationally efficient, allowing the model to capture long-range dependencies and emergent behaviors without requiring a fully connected latent space.

- describe step 404

Step 404 involves recursively determining the first moment of the latent state distribution at each time step by applying the deterministic moment-matching rule to the previous mean and the expected value of the transition function.

- determine first moment recursively

The first moment is determined recursively by applying the analytical expression for the expected value of the neural network transition function to the previous mean, ensuring that the propagation remains deterministic and does not require sampling.

- describe tool for expected value and covariance

The tool for expected value and covariance is a set of closed-form mathematical expressions derived from the properties of affine transformations, mean aggregation, and Gaussian distributions, which enable the efficient computation of moments without numerical integration.

- implement operation AGG

The operation AGG is implemented as a matrix multiplication between the adjacency matrix and the latent state vector, followed by normalization, ensuring that the aggregation is permutation-invariant and scalable to varying numbers of agents.

- calculate expected value and covariance

The expected value and covariance are calculated using the analytical moment-matching rules for each layer of the graph neural network, ensuring that the propagation of uncertainty is consistent with the underlying probabilistic model.

- determine Jacobi matrix

The Jacobi matrix is determined by computing the expected gradient of the transition function with respect to the latent state, using the chain rule and the known Jacobians of the neural network layers.

- describe affine transformation

The affine transformation is a linear operation applied to the latent state, parameterized by a weight matrix and bias vector, and is used in the update function of the graph neural network to transform aggregated messages into new latent states.

- calculate expected value and covariance recursively

The expected value and covariance are calculated recursively by iterating over time steps and applying the moment-matching rules at each step, ensuring that the predictive distribution remains accurate over long horizons.

- determine value of first moment

The value of the first moment is determined as the mean of the Gaussian mixture component after propagation through the transition model, representing the most likely latent state under each behavioral hypothesis.

- describe step 406

Step 406 involves recursively determining the second moment of the latent state distribution at each time step by applying the moment-matching rule to the previous covariance and the expected Jacobian of the transition function.

- determine second moment recursively

The second moment is determined recursively by computing the updated covariance using the law of propagation of uncertainty, incorporating both the deterministic transformation and the stochastic noise.

- calculate covariance of deterministic change

The covariance of the deterministic change is calculated by applying the expected Jacobian to the previous covariance matrix, capturing how uncertainty evolves under the predictable component of motion.

- calculate expected value of stochastic change

The expected value of the stochastic change is calculated as the trace of the noise covariance matrix, representing the average uncertainty introduced at each time step.

- determine value of second moment

The value of the second moment is determined as the updated covariance matrix after propagation through the transition model, representing the uncertainty in the latent state under each behavioral hypothesis.

- describe step 408

Step 408 involves determining the expected value of the first moment of the emission distribution by applying the emission model to the propagated latent state mean.

- determine expected value of first moment

The expected value of the first moment is determined by applying the emission neural network to the propagated latent state mean, producing the expected observed value at the prediction horizon.

- determine covariance of first moment

The covariance of the first moment is determined by propagating the latent state covariance through the emission model using the Jacobian of the emission function, ensuring that the uncertainty in the latent state is reflected in the observed prediction.

- describe step 410

Step 410 involves determining the first moment of the normal distribution representing the emission model, which is the mean of the predicted observed variable.

- describe step 412

Step 412 involves determining the expected value of the second moment of the emission distribution by combining the propagated latent state covariance with the emission noise covariance.

- describe step 414

Step 414 involves determining the second moment of the normal distribution representing the emission model, which is the sum of the propagated latent state covariance and the emission noise covariance, yielding the final predictive covariance.

- describe outer loop and prediction

The outer loop iterates over the Gaussian mixture components, ensuring that each behavioral hypothesis is propagated independently, and the final prediction is obtained by summing the weighted emissions from each component, resulting in a multimodal predictive distribution.