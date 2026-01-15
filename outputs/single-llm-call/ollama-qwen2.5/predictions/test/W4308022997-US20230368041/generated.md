- **Event Conditions in MiniGrid**: Event conditions were defined as agent interactions with objects, the goal, or crossing between rooms. Shorter history lengths were used in obstacle courses due to denser events and more objects, while a larger buffer accommodated diverse skills and numerous event tables.

- **MiniGrid Experiment Resources**: Two asynchronous rollout workers each with 1.5 CPUs and 2GB memory collected experience data efficiently using Reverb. Training utilized two virtual CPUs and 3 GB of memory at a rate of 40 training steps per second.

- **Continuous Control Task Resources**: For continuous control tasks, a single rollout worker with 1 CPU and 2GB memory was used, while training was conducted asynchronously using 7.7 CPUs and 8GB of memory.

- **Gran Turismo Environment Setup**: The Gran Turismo Sport racing simulator on PlayStation 4 provided realistic dynamics for agent training. Experiments focused on isolated scenarios like a slingshot passing maneuver or time trials to study the impact of Event Tables.

- **Slingshot Passing Scenario**: In this scenario, the RL agent raced against a built-in AI opponent, starting behind and using slipstreaming to accelerate past. Parameters included randomization in agent spacing and Balance of Power adjustments for diversity.

- **Time Trial at Maggiore**: This experiment involved a time trial on a 6 km track with no opponents. Data was collected from 20 agents uniformly distributed around the track, yielding about 400 experiences per time step over 150-second training episodes.

- **Training and Hardware Setup**: Agents were controlled by rollout workers using two virtual CPUs and 3.3 GB of memory at a 10Hz frequency. Actions and observations were exchanged through a restricted API, with experience streamed back to the trainer and stored in an ERB via Reverb. Training used one NVIDIA V100 or half an A100 GPU, ∼8 virtual CPUs, and 55 GB of RAM.