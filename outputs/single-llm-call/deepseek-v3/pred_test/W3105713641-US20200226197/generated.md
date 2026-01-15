Here is the drafted patent application following the provided outline and research paper content:

# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to quantum computing optimization, specifically to hybrid quantum-classical computing systems and methods for solving mixed integer optimization problems. More particularly, the invention provides improved variational quantum algorithms that utilize conditional value-at-risk (CVaR) as an aggregation function for measurement outcomes, leading to faster convergence and better solutions for combinatorial optimization problems compared to existing approaches that use expected value minimization.

## BACKGROUND

Quantum mechanics principles form the foundation of quantum computing, which leverages quantum phenomena such as superposition and entanglement to perform computations. Classical computers operate using bits that represent either 0 or 1, while quantum computers use quantum bits (qubits) that can exist in superpositions of states. This fundamental difference enables quantum computers to potentially solve certain problems more efficiently than classical computers.

IBM has developed quantum processors that implement quantum computing principles using superconducting qubits. These processors operate as noisy intermediate-scale quantum (NISQ) devices that can perform limited quantum computations. Combinatorial optimization problems, which involve finding optimal solutions from finite sets of possibilities, represent an important class of problems that quantum computers may help solve more efficiently.

Hybrid quantum-classical optimization algorithms have been developed to leverage both quantum and classical computing resources. The Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA) represent two prominent examples of such hybrid approaches. These algorithms use quantum processors to prepare parameterized quantum states and measure outcomes, while classical processors optimize the parameters based on measurement results.

Current hybrid quantum-classical optimization methods face several limitations. Classical algorithms for mixed integer optimization often struggle with solution space limitations as problem size increases. Existing hybrid approaches typically minimize the expected value of measurement outcomes, which may not effectively focus on finding optimal solutions. There exists a need for novel methods that can overcome these limitations and provide better performance on NISQ devices.

## SUMMARY

The present invention provides a method for solving mixed integer optimization problems using hybrid quantum-classical computing systems. The method involves generating decision variables that represent potential solutions to the optimization problem. Quantum state parameters are derived based on these decision variables and problem constraints. A quantum processor is initialized according to these parameters to prepare quantum states representing potential solutions.

Intermediate quantum states are measured multiple times to obtain samples of potential solutions. These samples are evaluated using conditional value-at-risk (CVaR) as an aggregation function rather than expected value. The classical processor calculates updated parameters based on the CVaR evaluation and repeats the process until convergence. This approach provides improved performance for mixed integer optimization problems compared to existing methods.

The invention further provides a computer program product comprising computer readable instructions that, when executed, cause a hybrid quantum-classical computing system to perform the disclosed method. The computer program product may be delivered as software-as-a-service (SaaS) or implemented on computer readable storage media.

## DETAILED DESCRIPTION

The present invention addresses optimization problems that can be formulated as mixed integer programs. A classical optimization problem typically involves minimizing or maximizing an objective function subject to constraints. The hybrid classical-quantum computing system of the invention handles both equality and inequality constraints through the introduction of slack variables when necessary.

Current hybrid algorithms like VQE and QAOA have limitations when applied to combinatorial optimization problems. The invention extends these approaches by incorporating CVaR optimization instead of expected value minimization. This modification better aligns with the practical goal of finding optimal or near-optimal solutions, as it focuses on improving the best measurement outcomes rather than the average.

The system architecture includes both classical and quantum processing components. The classical processing system comprises one or more servers, storage units, and memory containing optimization software applications. The quantum processing system includes quantum processors capable of preparing and measuring quantum states according to variational forms. These components may communicate through networks in various configurations including client-server, service-oriented, or cloud computing architectures.

The hybrid optimization algorithm operates through an iterative process. The classical processor runs an optimization scheme to determine decision variables and derive parameters for quantum circuits. The quantum processor prepares quantum states according to these parameters, executes the quantum circuits, and measures the resulting states to obtain samples. The classical processor evaluates these samples using CVaR, updates the parameters accordingly, and repeats the process until convergence criteria are met.

Application components include a classical optimizer that interfaces with the quantum processor, a decision variable determination module, and a quantum circuit parameter derivation module. The quantum processing system components handle state preparation and measurement operations. The system can be applied to various optimization problems including the Quadratic Assignment Problem by appropriately parameterizing binary decision variables and handling inequality constraints.

The computer-implemented method may be delivered as a service through SaaS models or implemented on computer readable storage media containing executable instructions. The instructions may be downloaded over networks and executed on appropriate hardware systems. The invention encompasses various embodiments including systems, methods, and computer program products for hybrid quantum-classical optimization using CVaR aggregation of measurement outcomes.