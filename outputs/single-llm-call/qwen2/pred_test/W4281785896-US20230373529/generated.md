# DESCRIPTION

## BACKGROUND

Self-driving cars have emerged as a transformative technology with the potential to revolutionize urban transportation. Over the past decade, significant advancements have been made in the development of autonomous vehicles, driven by the creation of new datasets and the application of advanced machine learning techniques. These datasets have facilitated substantial improvements in object detection and motion forecasting, key components of self-driving systems. However, the critical aspects of motion planning and decision-making have lagged behind, primarily due to the reliance on hand-engineered algorithms that require extensive manual tuning.

Traditional planning and decision-making algorithms for self-driving cars are heavily dependent on manually crafted scoring functions that determine desirable behaviors. This process is labor-intensive and prone to unintended consequences, as improving performance in one area often leads to regressions in others. To address these challenges, the present invention introduces a novel learning-based planner that leverages inverse reinforcement learning (IRL) to automate the process of generating and scoring trajectories. This planner, referred to as DriveIRL, avoids the need for manual feature engineering and weight tuning by learning from expert demonstrations. The system generates, checks, and scores trajectories for the vehicle, ensuring that they are dynamically feasible, follow the route, and satisfy assumptions from the vehicle controller. Additionally, a lightweight safety filter ensures that each trajectory is safe, providing a recursive safety guarantee.

The primary contributions of this invention include:
1. The first learning-based planner to drive a car in dense, urban traffic using IRL.
2. A simple yet powerful modeling framework that focuses learning on the most challenging aspects of driving behavior.
3. Detailed evaluation of the planner on a real-world dataset, which will be made publicly available.

## DETAILED DESCRIPTION

### General Overview

The present invention, DriveIRL, is a learning-based planner designed to control a self-driving car in dense, urban traffic. The system operates by generating a diverse set of possible future trajectories for the vehicle, checking their safety, and scoring them based on expert demonstrations. This approach leverages the strengths of inverse reinforcement learning (IRL) to automate the process of trajectory generation and scoring, thereby reducing the need for manual feature engineering and weight tuning.

The DriveIRL system consists of three main stages: trajectory generation, safety filtering, and trajectory scoring. Each stage is designed to ensure that the generated trajectories are dynamically feasible, safe, and aligned with expert driving behavior. The system is particularly effective in handling challenging scenarios such as heavy traffic, aggressive cut-ins, and busy passenger pickup/dropoff zones, as demonstrated by its successful deployment on the Las Vegas Strip.

#### Trajectory Generation

The trajectory generation module synthesizes a diverse set of possible future motions for the ego vehicle. The module uses the current ego state, the desired route, and the high-definition map to create a set of trajectories that are dynamically feasible and comply with the map. The ego state includes the vehicle's speed, acceleration, and steering, while the route specifies the lanes the ego should traverse to reach its destination. The high-definition map provides lane centerlines, road boundaries, traffic light locations, pedestrian crosswalks, speed limits, and other semantic information.

The trajectory generator integrates a desired acceleration profile along the route ahead of the ego. In our experiments, we specified a range of constant acceleration profiles, from a hard brake (-5.0 m/s²) to a moderate acceleration (1.5 m/s²). The initial ego pose is smoothly connected with the route using Dubins paths, ensuring that the trajectories are continuous and feasible. The generator typically creates 50-150 trajectories depending on the ego state and route. The generated trajectories are then passed to the safety filter for further processing.

#### Safety Filter

The safety filter ensures that each candidate trajectory is safe by performing a series of checks under specified assumptions about the behavior of non-ego road users. The filter checks that the distance to the vehicle ahead never falls below 1.5 meters, assuming that all non-ego vehicles perform a hard brake at 3.5 m/s². The filter also performs a recursive safety check by applying a firm brake to the ego trajectory after the first second, ensuring that the vehicle remains safe even if it needs to stop abruptly. This recursive safety guarantee is implemented with minimal assumptions and checks, ensuring that the vehicle maintains a comfortable and safe driving experience.

#### Trajectory Scoring with Maximum Entropy IRL

The core challenge of the planning approach is appropriately scoring the generated trajectories. This is achieved using a deep neural network trained with a maximum entropy IRL loss. The network is trained on expert demonstrations collected from a skilled human driver, and the loss function favors trajectories that closely match the expert's behavior in feature space. The features used for scoring include time-to-collision (TTC), adaptive cruise control (ACC) information, maximum jerk, maximum lateral acceleration, past coupling, and speed limit adherence.

The time-to-collision feature calculates the minimum number of seconds before the ego would collide with another road user, evaluated at multiple points along the trajectory. The ACC information feature provides the ego's speed, the distance to the vehicle ahead, the speed of the vehicle ahead, and the relative speed. The maximum jerk and maximum lateral acceleration features measure the smoothness of the trajectory, while the past coupling feature ensures coherence between the past, present, and future trajectories. The speed limit feature ensures that the trajectory adheres to the posted speed limits.

The neural network processes these features through a series of layers, including batch normalization, LSTM modules, and a masked self-attention mechanism. The final score for each trajectory is a weighted combination of the feature scores, with the highest-scoring trajectory being selected as the reference for the vehicle's tracking and actuator controller.

### Experimental Evaluation

The DriveIRL planner was evaluated on a large-scale dataset capturing real-world urban driving in the center of Las Vegas. The dataset, part of the nuPlan dataset, includes object annotations and high-definition maps. Vehicles, pedestrians, and bicyclists are automatically annotated using an offline perception system, and the dataset contains 182,032 scenarios, each 11 seconds in duration (1 second past, 10 seconds future). The scenarios were filtered to focus on adaptive cruise control (ACC) behavior, and a 3:1:1 split was used for training, validation, and testing.

The evaluation metrics include safety, comfort, progress, and L2 error. Safety metrics measure the collision rate, time to collision, and tailgate rate. Comfort metrics assess the longitudinal and lateral accelerations, jerk, and yaw rate. Progress metrics evaluate the distance to the goal and deviation from the route. L2 error measures the distance between the executed trajectory and the expert's trajectory.

The DriveIRL planner outperformed baseline models, including an Intelligent Driver Model (IDM) and a constant speed (CS) lane follow model, in all safety-related metrics. The planner also demonstrated higher scores in comfort and progress metrics, indicating its ability to handle challenging urban driving scenarios effectively.

### Real-World Deployment

Prior to deployment on public roads, the DriveIRL planner was rigorously tested in simulation and on private, closed-course routes. The simulation tests involved a high-fidelity dynamics model for the ego vehicle and numerous actors exhibiting a wide variety of behaviors. When deployed on the Las Vegas Strip, the vehicle was piloted by a trained operator who could take over in unsafe situations or scenarios outside the operating domain.

The planner successfully handled challenging scenarios such as heavy traffic, aggressive cut-ins, and busy passenger pickup/dropoff zones. Without the safety filter, the vehicle remained in autonomous mode for 8.8 miles of the 11-mile route, with overrides occurring for mandatory takeover regions and twice for undesired behavior. With the safety filter, the vehicle remained in autonomous mode for 6.9 of 8.5 miles, with takeovers only occurring due to mandatory takeover regions.

### Conclusion

DriveIRL represents a significant advancement in learning-based planning for self-driving cars. By focusing on the most challenging aspects of driving behavior and leveraging expert demonstrations, the system achieves strong real-world performance in dense, urban traffic. The modular design of the planner, combining trajectory generation, safety filtering, and trajectory scoring, ensures that the vehicle operates safely and comfortably. The successful deployment on the Las Vegas Strip demonstrates the practical utility of the approach and paves the way for further advancements in autonomous driving technology.