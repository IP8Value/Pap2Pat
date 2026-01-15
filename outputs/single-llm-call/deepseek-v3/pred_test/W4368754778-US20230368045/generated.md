Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION

## CROSS REFERENCE

The present application claims the benefit of German Patent Application No. [application number], filed on [filing date], which is incorporated herein by reference in its entirety.

## FIELD

The present invention relates to computer-implemented methods for predicting behavior in dynamic systems, particularly in the context of autonomous vehicles and multi-agent systems. More specifically, the invention concerns a deterministic approach for modeling stochastic dynamical systems with interacting agents using graph neural networks and moment matching techniques.

## BACKGROUND INFORMATION

Predicting the behavior of dynamic systems with multiple interacting agents presents significant challenges in various technical fields, particularly in autonomous vehicle control systems. Current approaches for behavior prediction typically rely on either recurrent neural networks that apply fixed transition models or history-based methods that aggregate past information. However, these methods face limitations in accurately capturing the stochastic nature of real-world interactions between multiple agents.

Existing state-space models attempt to address these challenges by maintaining latent states that evolve over time. While these models work well for deterministic systems, they struggle with stochastic multi-agent systems due to computational limitations. Monte Carlo sampling approaches, while theoretically sound, become prohibitively slow as the number of agents increases. Similarly, probabilistic history-based methods require complex models that may be impractical for embedded systems with limited memory capacity.

The technical problem addressed by the present invention is therefore to provide an efficient and accurate method for predicting behavior in stochastic dynamical systems with multiple interacting agents, particularly suitable for real-time applications such as autonomous vehicle control systems.

## SUMMARY

The present invention provides a computer-implemented method for predicting behavior of agents in a dynamic system that overcomes the limitations of existing approaches. The method achieves precise prediction of behavior through a novel combination of graph neural networks and moment matching techniques.

The method determines the value of the first moment of a first distribution representing the initial state of the system. It further determines the value of the second moment of this first distribution. Based on these determinations, the method calculates an expected value for the first moment of a second distribution representing a subsequent state of the system.

The invention also determines the second moment of a third distribution that approximates the system state at a future time point. A sum of these third distributions across multiple time points enables accurate prediction of behavior over extended time horizons.

The method recursively determines the value of the first moment and the value of the second moment at each time step, allowing for efficient determination of expected values throughout the prediction horizon. It calculates the covariance of the first moment of the second distribution and determines the expected value for the second moment of this distribution.

Context variables representing environmental factors are incorporated into the predictions. The method considers the history of the dynamic system and the neighborhood of agents when modeling their latent states. This comprehensive approach enables accurate control of agents based on the prediction results.

The technical advantages of this invention include improved computational efficiency compared to Monte Carlo methods, better handling of multi-modal distributions through Gaussian mixture models, and more accurate predictions through structured covariance approximations. These improvements are particularly valuable in real-world applications such as autonomous vehicle control systems where both accuracy and computational efficiency are critical.

## DETAILED DESCRIPTION OF EXAMPLE EMBODIMENTS

The present invention introduces a device for predicting behavior of agents in dynamic systems. The device comprises several key components including processing units, memory modules, and specialized hardware accelerators for neural network computations. These components work together to implement the behavior prediction method described herein.

The device includes an interface and sensor system designed to receive real-time data about the dynamic system. The sensor system capabilities encompass various data acquisition modalities including but not limited to LIDAR, radar, cameras, and vehicle-to-vehicle communication systems. These sensors provide comprehensive information about the state of agents and their environment.

An actuator system receives control commands generated based on the prediction results. These commands may control various aspects of vehicle operation including steering, acceleration, and braking in autonomous driving applications. The device executes a computer program implementing the behavior prediction algorithm described below.

In one exemplary embodiment, the invention is applied to a dynamic system comprising multiple vehicles navigating a roundabout. The system observes trajectories of vehicles approaching and navigating the roundabout. Based on these observed trajectories, the method predicts future trajectories while accounting for interactions between vehicles.

The prediction results are represented as Gaussian mixture distributions with associated confidence intervals. The method predicts various aspects of vehicle behavior including distance, velocity, and acceleration profiles. In the autonomous vehicle example, these predictions inform decision-making processes for safe and efficient navigation.

The method employs a machine learning model that incorporates latent variables representing unobserved aspects of agent states. These latent variables evolve over time through both deterministic changes, modeled by neural networks, and stochastic changes accounting for uncertainty in behavior prediction.

Context variables capture relevant environmental information such as road layout, traffic signals, and weather conditions. The observed variables represent measurable aspects of agent states. The method initializes values for latent states and models their evolution through time.

Normal distributions characterize the observed variables, while neural networks parameterize both the deterministic and stochastic components of state transitions. The method updates these components based on new observations and refines predictions accordingly.

Messages between agents are exchanged according to a graph structure representing their interactions. An aggregation operation combines information from neighboring agents to inform predictions about each agent's behavior. The edges of the graph define which agents influence each other's predicted behavior.

For a given prediction time point T, the method calculates the marginal probability of observed states given the context information. This involves computing a kernel representing state transitions and a Gaussian mixture model representing initial state distributions.

The method calculates mean values and covariances for these distributions, along with expected values and cross-covariances between different time points. Jacobi matrices facilitate efficient computation of these statistical quantities.

An inner loop and outer loop structure organizes the computation, with neural networks calculating moments at each step. The context variable and its association with neighboring agents informs the aggregation operation that combines information across the system.

The method defines trajectories based on historical observations and predicted future states. The aggregation operation plays a crucial role in this process by enabling information flow between connected agents. This motivates the use of graph neural networks as they naturally capture these interactions.

In step 404, the method determines the first moment recursively by propagating state information through time. Tools for calculating expected values and covariances support this process. The aggregation operation is implemented efficiently to handle large numbers of agents.

Expected values and covariances are calculated recursively to maintain computational tractability. The value of the first moment is determined at each time step, enabling accurate prediction of system evolution. Affine transformations facilitate these calculations while preserving the statistical properties of the distributions.

In step 406, the method determines the second moment recursively by calculating the covariance of deterministic changes and the expected value of stochastic changes. These calculations yield the value of the second moment at each time step.

Step 408 involves determining the expected value of the first moment and its covariance structure. Step 410 calculates the first moment of a normal distribution approximating the system state. Step 412 determines the expected value of the second moment, while step 414 calculates the second moment of the normal distribution.

The outer loop combines these calculations to produce final predictions of agent behavior. This comprehensive approach enables accurate, efficient prediction of complex multi-agent systems while accounting for uncertainties and interactions between agents.

The technical implementation details described herein enable the invention to overcome limitations of prior approaches, particularly in handling stochasticity, multi-modality, and computational efficiency in complex dynamic systems. The combination of graph neural networks with moment matching techniques represents a significant advancement in the field of behavior prediction for autonomous systems.