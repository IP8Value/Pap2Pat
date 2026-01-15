# DESCRIPTION

## TECHNICAL FIELD

- relate to quantum computing optimization

The present invention relates to systems and methods for solving combinatorial optimization problems using hybrid quantum-classical computing architectures. More specifically, the invention concerns a novel approach to parameterizing and executing variational quantum algorithms—such as the Variational Quantum Eigensolver (VQE) and the Quantum Approximate Optimization Algorithm (QAOA)—by employing the Conditional Value-at-Risk (CVaR) as an aggregation function for measurement outcomes, thereby enhancing the efficiency and robustness of solution convergence in the presence of noise and limited qubit coherence. The invention is particularly applicable to noisy intermediate-scale quantum (NISQ) devices, where traditional expectation-value-based optimization suffers from poor convergence, low probability of sampling high-quality solutions, and sensitivity to hardware imperfections. By focusing optimization on the tail of the probability distribution of measurement outcomes rather than their arithmetic mean, the disclosed method enables faster convergence to near-optimal solutions for a broad class of NP-hard combinatorial problems, including but not limited to maximum cut, portfolio optimization, number partitioning, and maximum satisfiability. The invention further encompasses computer program products, data processing systems, and service-oriented architectures capable of implementing the method across cloud-based quantum computing platforms, enabling scalable deployment for industrial and scientific applications.

## BACKGROUND

- introduce quantum mechanics

Quantum mechanics provides a mathematical framework for describing the behavior of physical systems at atomic and subatomic scales, where classical notions of determinism and locality no longer hold. In this framework, the state of a quantum system is represented by a vector in a complex Hilbert space, and observable quantities are associated with Hermitian operators known as Hamiltonians. The evolution of such systems is governed by unitary transformations, and measurements yield probabilistic outcomes determined by the squared magnitudes of state amplitudes. These principles form the foundation for quantum computation, where information is encoded in quantum bits, or qubits, whose superposition and entanglement properties enable computational parallelism unattainable by classical means.

- describe classical computers

Classical computers operate based on Boolean logic, where information is represented as discrete binary states—0 or 1—processed through deterministic logic gates arranged in circuits. These systems execute algorithms sequentially or in parallel using fixed-precision arithmetic and memory architectures governed by von Neumann or Harvard models. While highly reliable and scalable for many computational tasks, classical computers face fundamental limitations when addressing combinatorial optimization problems, particularly those classified as NP-hard, where the solution space grows exponentially with problem size. Even with advanced heuristics and approximation algorithms, classical solvers often require prohibitive computational resources to find near-optimal solutions for large-scale instances.

- introduce quantum computing

Quantum computing leverages the principles of quantum mechanics to process information in ways that transcend classical computational limits. Unlike classical bits, qubits can exist in superpositions of states, enabling a quantum register of n qubits to represent 2^n simultaneous configurations. Through controlled entanglement and interference, quantum algorithms can explore vast solution spaces in parallel, offering potential speedups for specific classes of problems. Although universal fault-tolerant quantum computers remain under development, current-generation devices—referred to as noisy intermediate-scale quantum (NISQ) processors—offer limited qubit counts and high error rates, necessitating the design of algorithms that are resilient to noise and require shallow circuit depths.

- explain qubits

A qubit is the fundamental unit of quantum information, analogous to the classical bit but endowed with properties of superposition and entanglement. Mathematically, a qubit is described by a two-dimensional complex vector in Hilbert space, typically written as α|0⟩ + β|1⟩, where α and β are complex amplitudes satisfying |α|² + |β|² = 1. When measured, a qubit collapses probabilistically to either |0⟩ or |1⟩, with probabilities determined by the squared magnitudes of the amplitudes. Multiple qubits can be entangled, creating correlated states that cannot be described independently, enabling exponential scaling of representational capacity. These properties make qubits suitable for encoding combinatorial optimization problems through Hamiltonians whose eigenstates correspond to feasible solutions.

- describe IBM's quantum processor

IBM has developed a family of superconducting quantum processors, including the IBM Q Poughkeepsie and other NISQ-era devices, which utilize transmon qubits cooled to millikelvin temperatures and controlled via microwave pulses. These processors feature limited connectivity, typically constrained to nearest-neighbor interactions, and are subject to gate errors, decoherence, and readout inaccuracies. Despite these limitations, IBM’s quantum cloud platform provides public access to real quantum hardware, enabling experimental validation of hybrid quantum-classical algorithms. The architecture supports native gate sets including single-qubit rotations and two-qubit CNOT gates, which are essential for constructing variational quantum circuits used in algorithms such as VQE and QAOA.

- introduce combinatorial optimization problems

Combinatorial optimization problems involve selecting the best configuration from a finite set of discrete possibilities, subject to constraints and an objective function to be minimized or maximized. Examples include the maximum cut problem, where vertices of a graph are partitioned to maximize edge weights across the cut; the portfolio optimization problem, which seeks to allocate assets to maximize return while minimizing risk under budget constraints; and the maximum satisfiability problem, which determines variable assignments to satisfy the greatest number of logical clauses. These problems are often NP-hard, meaning that no known algorithm can solve them exactly in polynomial time for arbitrary instances, motivating the search for heuristic and approximate methods.

- describe hybrid quantum/classical optimization algorithms

Hybrid quantum-classical optimization algorithms combine the strengths of quantum processors and classical computing systems to solve problems intractable for either alone. In such frameworks, a parameterized quantum circuit prepares a trial quantum state, which is then measured to yield a set of classical bitstrings. Each bitstring corresponds to a candidate solution, and its associated objective value is computed classically. A classical optimizer then adjusts the quantum circuit’s parameters to improve the quality of subsequent measurements. This iterative loop—quantum state preparation, measurement, evaluation, and parameter update—enables the use of noisy quantum hardware without requiring full error correction. VQE and QAOA are two prominent examples of such algorithms, both relying on the expectation value of a problem Hamiltonian as the objective function for classical optimization.

- recognize limitations of current methods

Current hybrid quantum-classical methods, particularly those based on minimizing the expectation value of the Hamiltonian, suffer from several critical limitations. First, the expectation value may be dominated by high-probability, low-quality solutions, masking the presence of rare but optimal or near-optimal outcomes. Second, the resulting optimization landscape is often flat or ill-conditioned, especially for shallow circuits, leading to slow convergence or premature termination. Third, noise and decoherence in NISQ devices further degrade the fidelity of measurement outcomes, making the expectation value an unreliable proxy for solution quality. Finally, algorithms such as QAOA, with a fixed number of parameters proportional only to circuit depth, may produce quantum states with uniformly distributed amplitudes—termed “flat” states—that have negligible overlap with the true ground state, even when the average energy appears favorable.

- recognize classical algorithms for mixed integer optimization

Classical algorithms for mixed integer optimization, such as branch-and-bound, cutting planes, and metaheuristics like simulated annealing and genetic algorithms, have been extensively developed and deployed in industry. These methods are deterministic or stochastic, often leveraging problem-specific structure to prune the search space or guide sampling. However, they are fundamentally limited by computational complexity, requiring exponential time in the worst case, and struggle with high-dimensional, non-convex, or noisy objective landscapes. Furthermore, they lack the inherent parallelism offered by quantum superposition, making them unsuitable for problems where the solution space is too vast for exhaustive or even heuristic exploration within practical timeframes.

- recognize need for novel method

There exists a clear and unmet need for a novel method that improves the convergence behavior, robustness, and solution quality of hybrid quantum-classical optimization algorithms on NISQ devices. Such a method must be compatible with existing quantum hardware constraints, require no additional physical resources, and be implementable with minimal modification to current algorithmic frameworks. It must also be capable of prioritizing high-quality outcomes over average performance, even in the presence of noise, and must scale effectively with problem size without requiring deeper circuits or more qubits.

- recognize solution space limitations

The solution space of combinatorial optimization problems is inherently discrete and exponentially large, with the number of feasible configurations growing as 2^n for n binary variables. Traditional variational algorithms, by optimizing the expectation value, tend to distribute probability mass evenly across many suboptimal states, especially when the Hamiltonian’s energy landscape is flat or degenerate. This results in a low probability of sampling the optimal or near-optimal solution, even after many iterations. Moreover, the presence of local minima in the classical optimization landscape, combined with the statistical noise inherent in quantum measurements, further impedes the ability of current methods to navigate toward high-quality solutions. A method that explicitly targets the tail of the probability distribution—where the best solutions reside—is therefore necessary to overcome these intrinsic limitations.

## SUMMARY

- introduce method for solving mixed integer optimization

The invention introduces a novel method for solving mixed integer optimization problems using a hybrid quantum-classical computing system, wherein the classical optimization objective is defined not as the expected value of the problem Hamiltonian, but as the Conditional Value-at-Risk (CVaR) of the measurement outcomes. This approach shifts the focus of optimization from the average performance of the quantum state to the quality of its best-performing outcomes, thereby increasing the probability of sampling high-quality or optimal solutions. The method is applicable to any combinatorial optimization problem that can be mapped to a diagonal Hamiltonian and executed on a parameterized quantum circuit, including but not limited to binary quadratic optimization, maximum cut, portfolio optimization, and satisfiability problems.

- generate decision variables

The method begins by representing the combinatorial optimization problem in terms of binary decision variables, each corresponding to a qubit in the quantum processor. These variables encode feasible solutions as bitstrings, with each bit indicating the inclusion or exclusion of an element in the solution set. Constraints are encoded into the Hamiltonian as penalty terms, ensuring that only valid configurations contribute meaningfully to the objective function. The decision variables are then mapped to Pauli-Z operators acting on individual qubits, transforming the classical optimization problem into a quantum mechanical observable.

- derive quantum state parameters

A parameterized quantum circuit is constructed to prepare a trial quantum state dependent on a set of variational parameters. These parameters, which may include rotation angles and entangling gate durations, are optimized by a classical algorithm to minimize the CVaR of the Hamiltonian’s measurement outcomes. The circuit is designed to span the solution space sufficiently, with a number of parameters scaling at least linearly with the number of qubits to avoid flat state distributions. The parameters are initialized to a uniform or random configuration and updated iteratively based on feedback from quantum measurements.

- initiate quantum processor

The quantum processor is initialized to the all-zero state, and the parameterized quantum circuit is applied to generate the trial wavefunction. The circuit consists of single-qubit rotations and entangling gates, arranged in layers to ensure sufficient expressivity and connectivity. The number of layers, or circuit depth, is selected based on the problem size and hardware constraints, with the constraint that the total gate count remains within the coherence limits of the device.

- measure intermediate quantum states

Upon completion of the circuit, the qubits are measured in the computational basis, yielding a set of classical bitstrings. Each bitstring corresponds to a candidate solution, and its associated objective value is computed classically by evaluating the Hamiltonian’s diagonal elements. A predetermined number of measurements—typically thousands—are collected to form a statistical sample of possible outcomes.

- evaluate samples

The collected samples are sorted in ascending order of their objective values, and the Conditional Value-at-Risk is computed for a specified confidence level α ∈ (0,1]. The CVaR is defined as the average of the lowest α-fraction of sampled outcomes, effectively focusing optimization on the tail of the distribution where the best solutions reside. This metric replaces the traditional sample mean as the objective function for the classical optimizer, thereby biasing the search toward states with higher likelihood of containing optimal or near-optimal solutions.

- provide for classical processor calculation

The classical processor executes an optimization algorithm—such as COBYLA, L-BFGS-B, or gradient descent—to adjust the variational parameters in a direction that minimizes the computed CVaR. The classical optimizer evaluates the CVaR for each parameter update by re-running the quantum circuit, collecting new samples, and recalculating the metric. This iterative process continues until convergence criteria are met, such as a minimum change in CVaR over successive iterations or a maximum number of iterations.

- provide for mixed integer optimization

The final output of the method is a quantum state whose measurement outcomes exhibit a high probability of yielding solutions that are optimal or near-optimal for the original mixed integer optimization problem. The best solution observed across all iterations is selected as the final answer, and its quality is validated classically. The method is fully compatible with existing problem encodings, including quadratic unconstrained binary optimization (QUBO) and Ising models, and requires no modification to the underlying hardware architecture.

- provide for computer program product

The invention further encompasses a computer program product comprising non-transitory computer-readable storage media encoded with instructions that, when executed by a processing system, cause the system to perform the method described above. The program product may be distributed as software libraries, cloud-based services, or embedded firmware, and may be integrated into existing optimization software stacks. The instructions may be implemented in high-level programming languages, compiled into machine code, or directly executed by quantum control hardware, and may include modules for problem encoding, circuit generation, measurement sampling, CVaR computation, and classical optimization.

## DETAILED DESCRIPTION

- introduce optimization problem

The optimization problem addressed by the invention is a mixed integer optimization problem characterized by a set of binary decision variables, a linear or quadratic objective function, and a set of equality or inequality constraints. The problem is formulated as a minimization or maximization task over a discrete solution space, where each solution corresponds to a unique assignment of values to the decision variables. The goal is to identify a solution that optimizes the objective while satisfying all constraints, a task that becomes computationally intractable as the number of variables increases.

- describe classical optimization problem

Classically, such problems are solved using algorithms that explore the solution space through enumeration, relaxation, or heuristic search. These methods often rely on convex approximations, branch-and-bound trees, or stochastic sampling to navigate the non-convex and high-dimensional landscape of feasible solutions. However, the exponential growth of the solution space renders these approaches impractical for large-scale problems, motivating the need for alternative computational paradigms.

- define objective function

The objective function is defined as a real-valued function over the space of binary assignments, typically expressed as a sum of linear and quadratic terms involving the decision variables. For example, in a portfolio optimization problem, the objective may represent the trade-off between expected return and risk, penalized by a budget constraint. In the context of the invention, this classical objective function is transformed into a Hamiltonian operator whose diagonal elements correspond to the objective values of all possible bitstring solutions.

- explain hybrid classical-quantum computing system

The hybrid classical-quantum computing system comprises a classical processor, a quantum processor, and a communication interface enabling bidirectional data exchange. The classical processor is responsible for problem encoding, parameter initialization, CVaR computation, and optimization loop control. The quantum processor executes the parameterized circuit, performs measurements, and returns the sampled outcomes. The system operates in a closed-loop fashion, where each iteration of the classical optimizer triggers a new quantum execution, and the resulting measurements inform the next parameter update.

- recognize limitations of current hybrid algorithms

Current hybrid algorithms, which minimize the expectation value of the Hamiltonian, suffer from poor convergence due to the flatness of the optimization landscape, the dominance of low-quality solutions in the mean, and the inability to distinguish between states with similar average energy but vastly different tail behaviors. These limitations are exacerbated on NISQ devices, where noise obscures the true energy distribution and reduces the fidelity of measurement outcomes.

- describe linear quality constraints

Linear quality constraints are incorporated into the Hamiltonian as penalty terms that increase the energy of infeasible solutions. For example, a budget constraint requiring exactly B selected assets in a portfolio is enforced by adding a term proportional to the square of the deviation from B. These constraints do not alter the fundamental structure of the problem but ensure that the quantum state preferentially samples feasible solutions.

- introduce mixed integer optimization problems

Mixed integer optimization problems involve decision variables that may be continuous, binary, or integer-valued. In the context of this invention, the focus is on binary integer optimization, where all variables are constrained to {0,1}. These problems are naturally suited to quantum representation via qubits, as each variable maps directly to a qubit state. The problem Hamiltonian is constructed to encode both the objective and constraints into a single operator, enabling direct evaluation via quantum measurement.

- describe general classical optimization problem

The general classical optimization problem is defined as minimizing f(x) subject to g_i(x) ≤ 0 and h_j(x) = 0, where x is a vector of binary variables, f is the objective function, and g_i and h_j are inequality and equality constraints, respectively. This formulation is transformed into a quantum Hamiltonian H by replacing each variable x_i with the operator (I - σ_i^Z)/2, and each term in f and g_i with a corresponding tensor product of Pauli operators.

- add slack variables to equality constraints

Equality constraints are converted into inequality constraints by introducing slack variables, which are then encoded as additional qubits or absorbed into the penalty structure of the Hamiltonian. This transformation ensures compatibility with the QUBO or Ising model formalism, which only supports quadratic interactions between binary variables.

- handle inequality constraint

Inequality constraints are handled by augmenting the objective function with a penalty term proportional to the square of the constraint violation. The penalty weight is chosen to be sufficiently large to render infeasible solutions energetically unfavorable, while remaining small enough to avoid overwhelming the objective function’s natural scale.

- extend VQE hybrid classical-quantum approach

The Variational Quantum Eigensolver (VQE) is extended by replacing the expectation value of the Hamiltonian with the CVaR of the measurement outcomes as the objective function for the classical optimizer. This modification requires no change to the quantum circuit structure but fundamentally alters the optimization landscape, making it more conducive to convergence toward high-quality solutions.

- extend QAOA approach

The Quantum Approximate Optimization Algorithm (QAOA) is similarly extended by substituting the expectation value with CVaR as the metric to be minimized during parameter optimization. This change mitigates the tendency of QAOA to produce flat states by encouraging the classical optimizer to amplify the amplitudes of low-energy outcomes, even when their probability is low.

- describe illustrative embodiments

An illustrative embodiment involves solving a portfolio optimization problem with six assets, a fixed budget of three, and a risk-aversion parameter. The problem is encoded into a six-qubit Hamiltonian, and CVaR-VQE is executed on an IBM quantum processor with depth p=1 and α=10%. The method converges in fewer iterations than traditional VQE and achieves a ground state sampling probability exceeding 10%, matching the chosen confidence level.

- provide example configurations

Example configurations include a system with a classical server hosting the optimizer, a quantum processor connected via a network, and clients submitting optimization tasks through a web interface. The quantum processor may be located in a cloud environment, with results returned to the client upon completion. The system may support multiple users, concurrent jobs, and automated parameter tuning.

- describe data processing environments

The data processing environment includes a classical processing system, a quantum processing system, and a network connecting them. The classical system may be a server or workstation equipped with memory, storage, and a central processing unit. The quantum system may be a superconducting or trapped-ion processor with control electronics and measurement apparatus. The environment may be implemented as a cloud-based service, a local installation, or a hybrid architecture.

- describe network 102

The network facilitates communication between the classical and quantum systems, transmitting parameter sets from the classical processor to the quantum processor and returning measurement outcomes. The network may be a local area network, a wide area network, or the Internet, and may employ secure protocols for data transmission and authentication.

- describe classical processing system 104

The classical processing system executes the optimization algorithm, manages the variational parameters, computes the CVaR from measurement data, and coordinates the iterative loop. It may include a memory unit for storing problem instances, a storage unit for logging results, and application software for user interaction and result visualization.

- describe server 106

The server acts as a central hub for managing multiple optimization tasks, scheduling quantum circuit executions, and aggregating results from multiple users or problem instances. It may host a queue system, a database of optimized parameters, and an API for programmatic access.

- describe storage unit 108

The storage unit retains problem definitions, parameter histories, measurement samples, and optimization logs. It may be implemented as a solid-state drive, magnetic disk, or cloud-based storage, and may support versioning and backup capabilities.

- describe quantum processing system 140

The quantum processing system comprises qubits, control electronics, microwave generators, and readout circuits. It receives parameterized circuits from the classical system, executes them, measures the qubits, and returns the bitstring outcomes. The system may be calibrated regularly to maintain gate fidelity and reduce readout errors.

- describe clients 110, 112, and 114

Clients are end-user devices—such as laptops, tablets, or smartphones—that submit optimization problems to the server, monitor progress, and retrieve results. They may include graphical user interfaces for problem specification and result interpretation.

- describe device 132

The device refers to any physical or virtual machine capable of running the classical optimization software, including embedded systems, edge devices, or virtual machines in a cloud environment.

- describe memory 124

Memory 124 stores the classical optimization algorithm, problem data, and intermediate parameter values during execution. It may be volatile or non-volatile and may include cache, RAM, or register files.

- describe application 105

Application 105 is a software module that encapsulates the entire optimization workflow, including problem encoding, circuit generation, parameter update logic, and CVaR computation. It may be implemented as a library, a standalone executable, or a web service.

- describe memory 144

Memory 144 stores the quantum circuit definitions, control sequences, and measurement settings for the quantum processor. It may be part of the quantum control system and may be updated dynamically during execution.

- describe application 146

Application 146 is a firmware or software component on the quantum processor responsible for translating classical parameter inputs into pulse sequences, executing the circuit, and returning measurement results.

- describe data processing environment 100 as the Internet

The data processing environment 100 may be implemented as the Internet, enabling remote access to quantum computing resources via web portals or APIs. Users may submit problems from any location, and results are delivered over secure channels.

- describe client-server environment

The client-server environment enables scalable deployment, where multiple clients interact with a centralized server that manages quantum resources, schedules jobs, and distributes results. This architecture supports load balancing, authentication, and usage tracking.

- describe service oriented architecture

The service-oriented architecture decomposes the optimization workflow into discrete, interoperable services: problem encoding, circuit generation, quantum execution, CVaR evaluation, and parameter optimization. Each service may be independently deployed, scaled, and updated.

- describe cloud computing model

The cloud computing model allows users to access quantum processing as a service, paying only for usage. The invention may be deployed as a Software-as-a-Service (SaaS) offering, with subscription tiers based on problem size, execution frequency, or priority.

- describe data processing system 200

Data processing system 200 is a representative hardware architecture comprising a processor, memory, storage, input/output interfaces, and communication modules. It may be used to implement the classical component of the hybrid system.

- describe data processing system

The data processing system includes a central processing unit, system memory, storage devices, and buses connecting these components. It may run an operating system and programming environments capable of executing the optimization software.

- detail hardware components

Hardware components include processors, memory modules, storage drives, network interfaces, and power supplies. The system may be implemented as a single machine or distributed across multiple nodes.

- explain bus system

The bus system provides a communication pathway between the processor, memory, and peripherals, enabling data transfer and synchronization. It may be a shared bus, point-to-point link, or hierarchical interconnect.

- describe memory and storage devices

Memory devices include RAM, cache, and registers, while storage devices include hard drives, SSDs, and optical media. Both are used to retain program instructions, problem data, and historical results.

- detail operating system and programming systems

The operating system manages hardware resources and provides an interface for application execution. Programming systems include compilers, interpreters, and runtime environments for languages such as Python, C++, or Qiskit.

- describe instructions and code

Instructions and code refer to the software implementation of the optimization algorithm, including functions for circuit generation, parameter update, and CVaR computation. Code may be written in high-level languages and compiled into machine-executable form.

- explain downloading code over a network

Code may be downloaded from a remote server over a network, enabling updates, patches, or new problem templates to be deployed without physical access to the system.

- describe hardware variations

Hardware variations include different qubit technologies (superconducting, trapped ion, photonic), classical processors (CPU, GPU, FPGA), and network topologies. The invention is agnostic to specific implementations and may be adapted to any compatible architecture.

- introduce hybrid quantum/classical optimization algorithm

The hybrid quantum/classical optimization algorithm comprises a classical optimizer that iteratively adjusts parameters of a quantum circuit to minimize the CVaR of measurement outcomes. The algorithm is executed in a closed loop, with each iteration involving quantum state preparation, measurement, classical evaluation, and parameter update.

- detail classical processor and quantum processor

The classical processor performs all non-quantum computations, including problem encoding, CVaR calculation, and parameter optimization. The quantum processor executes the variational circuit and returns measurement samples. The two are connected via a communication interface that transmits parameters and results.

- describe classical optimization scheme

The classical optimization scheme employs a derivative-free or gradient-based algorithm to minimize the CVaR. It evaluates the objective function at each step by querying the quantum processor for a new set of samples and computing the average of the lowest α-fraction of outcomes.

- derive quantum angles and parameters

Quantum angles and parameters are derived from the classical optimizer’s output and correspond to rotation angles and entangling gate durations in the quantum circuit. These parameters are updated iteratively to improve the CVaR.

- prepare quantum states

Quantum states are prepared by applying a sequence of single-qubit rotations and entangling gates to the initial |0⟩ state, as defined by the current parameter set. The circuit is designed to be shallow enough to avoid decoherence but expressive enough to explore the solution space.

- execute quantum states and measure samples

The prepared quantum state is executed on the quantum processor, and multiple measurements are performed to collect a statistical sample of bitstrings. Each bitstring is converted into an objective value using the Hamiltonian mapping.

- evaluate samples and update parameters

Samples are sorted, and the CVaR is computed. The classical optimizer uses this value to compute a gradient or direction for parameter update, and the process repeats.

- repeat process until convergence

The iterative loop continues until a convergence criterion is satisfied, such as a minimal change in CVaR over successive iterations, a maximum number of iterations, or a target solution quality.

- describe application components

Application components include a decision variable determination module, a quantum circuit generator, a CVaR evaluator, and a classical optimizer engine. These components interact to automate the entire optimization workflow.

- detail classical optimizer component

The classical optimizer component selects the optimization algorithm, sets convergence thresholds, manages parameter history, and interfaces with the quantum processor. It may support multiple algorithms and adaptive step sizes.

- describe decision variable determination component

The decision variable determination component maps the problem’s variables to qubits, encodes constraints into the Hamiltonian, and ensures compatibility with the quantum processor’s native gate set.

- derive quantum circuit angles

Quantum circuit angles are derived from the classical optimizer’s output and represent the parameters of single-qubit rotations and entangling gate durations. These angles are updated in each iteration to improve the CVaR.

- derive classical parameters

Classical parameters include the confidence level α, the number of samples per iteration, the optimization algorithm, and convergence thresholds. These are user-configurable and may be tuned based on problem characteristics.

- evaluate quantum state samples

Quantum state samples are evaluated by computing their corresponding objective values using the Hamiltonian mapping and aggregating them into the CVaR metric.

- describe quantum processing system components

Quantum processing system components include qubit arrays, control electronics, microwave generators, and readout circuits. These components work in concert to prepare, manipulate, and measure quantum states.

- prepare quantum states

Quantum states are prepared by applying a sequence of calibrated pulses to the qubits, as defined by the parameter set received from the classical processor.

- measure quantum states

Quantum states are measured by projecting each qubit onto the computational basis and recording the resulting bitstring. Multiple measurements are performed to ensure statistical reliability.

- describe flowchart process for solving mixed integer optimization problems

The flowchart begins with problem encoding, followed by initialization of variational parameters. The quantum circuit is generated and executed, and samples are collected. The CVaR is computed, and the classical optimizer updates the parameters. The process repeats until convergence, at which point the best solution is returned.

- describe another flowchart process for solving mixed integer optimization problems

Another flowchart process includes an additional step for adaptive α tuning, where the confidence level is adjusted dynamically based on the progress of optimization. This enhances convergence in problems with highly skewed energy landscapes.

- describe application to Quadratic Assignment Problem

The method is applied to the Quadratic Assignment Problem by encoding facility-to-location assignments as binary variables, constructing a Hamiltonian that penalizes distance and flow mismatches, and optimizing the CVaR using a parameterized circuit with sufficient depth to avoid flat states.

- use binary decision variables

Binary decision variables are used to represent discrete choices, such as whether a facility is assigned to a location or whether an asset is included in a portfolio.

- include inequality constraints

Inequality constraints are included by augmenting the Hamiltonian with penalty terms that increase energy for violations, ensuring that feasible solutions dominate the low-energy tail of the distribution.

- parameterize decision variables

Decision variables are parameterized through the angles of single-qubit rotations in the variational circuit, allowing continuous optimization over discrete solution spaces.

- compute classical parameters

Classical parameters are computed based on problem size, desired solution quality, and hardware capabilities, and may include α, number of samples, and optimization algorithm settings.

- compute quantum parameters

Quantum parameters are computed by the classical optimizer as a function of the CVaR gradient and are used to update the quantum circuit’s rotation angles and entangling gate durations.

- describe computer implemented method

The computer-implemented method comprises a series of steps executed by a processing system, including encoding the problem, generating a quantum circuit, measuring outcomes, computing CVaR, updating parameters, and repeating until convergence.

- describe system or apparatus

The system or apparatus comprises a classical processor, a quantum processor, a communication interface, and software for orchestrating the optimization workflow. It may be implemented as a single device or distributed across multiple nodes.

- describe computer program product

The computer program product comprises non-transitory computer-readable storage media encoded with instructions that, when executed, cause a processing system to perform the method described herein. The product may be distributed as a software library, cloud API, or embedded firmware.

- describe delivery of application in SaaS model

The application may be delivered as a Software-as-a-Service (SaaS) platform, allowing users to submit optimization problems via a web interface, receive results via email or dashboard, and pay based on usage or subscription.

- describe computer readable storage medium

The computer-readable storage medium may be a magnetic disk, optical disc, solid-state drive, or flash memory, and may be removable or embedded within the system.

- describe computer readable program instructions

Computer-readable program instructions are sequences of code that, when executed by a processor, cause the system to perform the steps of the method. These instructions may be in source code, bytecode, or machine code.

- describe downloading instructions from network

Instructions may be downloaded from a remote server over a network, enabling automatic updates, new algorithm versions, or problem templates to be deployed without manual intervention.

- describe executing instructions on computer

Instructions are loaded into memory and executed by the processor, which performs the steps of problem encoding, parameter optimization, and result generation.

- describe computer readable program instructions as assembler instructions

The computer-readable program instructions may be implemented as assembler instructions, directly executable by the processor’s instruction set architecture.

- describe computer readable program instructions as object code

The instructions may be compiled into object code, which is linked with libraries and executed by the operating system.

- describe electronic circuitry executing instructions

Electronic circuitry, including application-specific integrated circuits (ASICs) or field-programmable gate arrays (FPGAs), may be designed to execute the optimization algorithm in hardware, enabling low-latency, high-throughput operation.