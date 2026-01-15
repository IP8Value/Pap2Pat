## FIELD

- define field of application

The present invention resides in the field of quantum computing and discrete optimization, specifically in the implementation of quantum algorithms for accelerating Markov chain Monte Carlo (MCMC) simulations through the use of quantized classical walks that avoid explicit arithmetic operations. The invention provides a novel circuit-level architecture for executing quantum walks derived from reversible classical transition dynamics, particularly those governed by the Metropolis-Hastings algorithm or Glauber dynamics, without reliance on oracle-based implementations of the transition matrix. This approach enables the efficient preparation of Boltzmann-distributed states over discrete configuration spaces, with direct applicability to combinatorial optimization problems, statistical physics simulations, and machine learning tasks involving energy minimization over high-dimensional binary or discrete state spaces. The disclosed methods are particularly suited for implementation on near-term quantum hardware with limited coherence times and gate fidelities, as they minimize circuit depth, eliminate costly quantum arithmetic, and reduce the number of required qubits through unary encoding and parallelizable gate structures. The invention further enables heuristic quantum optimization protocols that outperform classical MCMC methods in time-to-solution metrics, offering a scalable pathway toward quantum advantage in problems such as Ising spin glass minimization, constraint satisfaction, and probabilistic graphical model inference.

## SUMMARY

- introduce quantum walk implementation
- reformulate quantum walk to avoid arithmetic operations
- apply quantum walk to discrete optimization problems
- summarize numerical results

The invention introduces a fully circuit-based implementation of a quantum walk operator that emulates the behavior of a classical reversible Markov chain without requiring the explicit construction or inversion of the transition matrix. Unlike prior approaches that depend on quantum oracles to perform arithmetic evaluations of transition probabilities, this invention replaces such operations with a sequence of local, conditionally applied quantum gates that directly encode the acceptance criteria of the Metropolis-Hastings or Glauber dynamics rules. The quantum walk is constructed using four core components: a move preparation unitary that initializes a unary-encoded register representing possible spin-flip moves, a spin flip operator that applies conditional bit flips to the system register based on the selected move, a reflection operator that inverts the phase of the all-zero state in the move and coin registers, and a Boltzmann coin operator that applies a rotation to a single qubit conditioned on the local energy difference induced by the proposed move. These components are combined into a single unitary operator that, when iteratively applied, drives the system toward a coherent superposition of low-energy states corresponding to the Boltzmann distribution. The invention eliminates the need for quantum addition, comparison, or division operations, which are known to introduce substantial circuit depth and error accumulation. Instead, the Boltzmann coin is implemented using a series of multi-controlled single-qubit rotations whose complexity scales only with the sparsity of the interaction graph, not the full system size. Numerical simulations on one-dimensional and sparse random Ising models demonstrate that the resulting quantum algorithm achieves a polynomial speedup over classical MCMC, with observed time-to-solution scaling exponents as low as 0.42—exceeding the theoretical quadratic speedup predicted by standard quantum walk analyses. The invention further introduces a unitary heuristic that applies the quantum walk operator sequentially without intermediate measurement, achieving performance comparable to or better than Zeno-based adiabatic state preparation protocols, despite the absence of projective measurements. These results indicate that the proposed architecture not only circumvents the practical limitations of oracle-based quantum walks but also unlocks new regimes of quantum advantage in discrete optimization.

## DETAILED DESCRIPTION

### I. Introduction

- motivate MCMC simulations
- introduce quantum walk acceleration
- describe embodiments

Markov chain Monte Carlo methods are foundational tools in computational science for sampling from high-dimensional probability distributions, particularly in statistical physics, machine learning, and combinatorial optimization. These methods rely on constructing a reversible Markov chain whose stationary distribution matches the target distribution, such as the Boltzmann distribution over spin configurations. However, the mixing time of such chains often scales poorly with system size, limiting their utility for large-scale problems. Quantum walks offer a provable quadratic speedup over classical mixing times by leveraging quantum interference to coherently explore the state space. Yet, practical implementations have been hindered by the requirement of an oracle that computes transition probabilities via arithmetic operations, which are resource-intensive and incompatible with the constraints of noisy intermediate-scale quantum devices. The present invention overcomes this limitation by providing a direct, gate-based construction of the quantum walk operator that encodes the acceptance probability of the Metropolis-Hastings algorithm as a local rotation on a coin qubit, conditioned on the local energy change induced by a proposed move. This embodiment avoids any explicit computation of the transition matrix, instead using the structure of the underlying physical model—such as a sparse Ising Hamiltonian—to implement the acceptance condition through a sequence of constant-depth quantum gates. The invention further encompasses multiple variants of this core architecture, including parallelized implementations that exploit the locality of interactions to reduce circuit depth, and alternative encodings of the move register that minimize qubit overhead while preserving the unitarity of the walk operator.

### II. General Considerations

- define terminology
- clarify method descriptions

For the purposes of this disclosure, the term “system register” refers to a collection of qubits encoding the current configuration of a discrete system, such as a spin lattice where each qubit represents the state of a spin variable. The “move register” is a unary-encoded register of N qubits, where each qubit corresponds to a distinct possible move, such as a single-spin flip or a small cluster flip, and the presence of a single excitation in position j indicates that move j is selected. The “coin register” is a single qubit that controls the acceptance or rejection of the proposed move through a rotation whose angle is determined by the energy difference between the current and proposed states. The term “Boltzmann coin” denotes the unitary operator that applies a rotation to the coin qubit conditioned on the system state and the selected move, implementing the acceptance probability defined by the Metropolis-Hastings rule. The “reflection operator” is defined as a phase flip of the all-zero state in the move and coin registers, and is used to create interference between the proposed and unproposed branches of the quantum walk. The term “quantum walk operator” refers to the unitary composed of the move preparation, spin flip, reflection, and Boltzmann coin components, whose repeated application drives the system toward the desired stationary distribution. All operations are defined to be unitary and reversible, ensuring that the overall evolution preserves quantum coherence and enables interference effects that underlie the quantum speedup.

### III. Example Embodiments

- motivate quantum walks
- introduce Szegedy's method
- explain eigenvalues of unitary matrix
- define steady state of quantum walk
- describe spectral gap of quantum walk
- motivate quantum adiabatic algorithm
- introduce classical walk oracle
- define quantum walk operator U
- analyze quantum walk operator U
- define operator X
- analyze operator X
- define eigenvectors of X
- explain action of W†ΛW
- describe block diagonalization of W†ΛW
- introduce adiabatic state preparation
- explain measurement outcome of adiabatic state preparation
- describe complexity of adiabatic state preparation
- introduce Metropolis-Hastings algorithm
- define transition probability of Metropolis-Hastings algorithm
- explain detailed balance condition
- introduce Glauber dynamics
- describe Boltzmann distribution
- introduce circuit for walk operator
- explain implementation of walk operator
- describe unary representation of move register
- introduce components of walk operator
- explain move preparation V
- describe binary-tree implementation of V
- introduce coin flip operator B
- explain flip operator F
- introduce reflection operator R
- summarize complexity of walk operator components
- introduce quantum walk
- motivate quantum walk
- define quantum walk operator
- introduce spin flip F
- implement spin flip F
- introduce reflection R
- implement reflection R
- introduce Boltzmann coin B
- implement Boltzmann coin B
- describe Boltzmann coin complexity
- introduce heuristic use
- describe Metropolis-Hastings algorithm
- introduce total time to solution
- define total time to solution
- introduce Zeno with rewind
- describe Zeno with rewind
- introduce unitary implementation
- describe unitary implementation
- introduce numerical results
- describe numerical results for 1D Ising model
- describe numerical results for sparse random Ising model
- introduce cost analysis
- analyze cost of gate V
- analyze cost of applying proposed transition
- analyze cost of Boltzmann coin operation
- describe simplifications for sparse Ising model
- describe simplifications for discrete parameter Jl
- introduce example circuit layouts
- describe example circuit layouts
- introduce improved example circuit layouts
- describe improved example circuit layouts
- introduce quantum signal processing
- describe quantum signal processing
- introduce decomposition of exponential
- describe decomposition of exponential
- introduce Zeno algorithm
- describe Zeno algorithm
- conclude heuristic use

The invention is motivated by the observation that quantum walks, when properly structured, can accelerate the convergence of classical Markov chains by exploiting quantum interference to suppress transitions to high-energy states while enhancing those to low-energy ones. While Szegedy’s formalism provides a general framework for quantizing reversible classical walks, its reliance on an oracle to compute transition probabilities has rendered it impractical for real-world applications. This invention circumvents this limitation by constructing a quantum walk operator U that directly implements the Metropolis-Hastings acceptance rule without ever computing the transition matrix elements explicitly. The operator U is composed of four unitary components: V, F, R, and B. The move preparation operator V initializes the move register into an equal superposition over all possible moves using a binary-tree structure of √SWAP gates, achieving this in logarithmic depth. The spin flip operator F applies a controlled-NOT or Toffoli gate to the system register conditioned on the move register and the coin register, flipping the spins specified by the selected move. The reflection operator R performs a phase flip on the state where all qubits in the move and coin registers are zero, implemented via a binary-tree cascade of Toffoli gates with ancillae, requiring only 2 log N depth. The Boltzmann coin operator B applies a rotation to the coin qubit by an angle θ = βΔE/2, where ΔE is the energy difference induced by the proposed move, and β is the inverse temperature. This rotation is implemented using a sequence of multi-controlled single-qubit rotations, each acting on a constant-sized subset of the system register, and can be further optimized using quantum signal processing techniques that decompose the exponential acceptance function into a Fourier series of controlled rotations. The resulting quantum walk operator U has eigenvalues of the form e±iθk, where θk = arccos(λk) and λk are the eigenvalues of the classical transition matrix. The steady state of the walk corresponds to the eigenvalue θ = 0, which encodes the coherent superposition of all configurations weighted by their Boltzmann probabilities. The spectral gap δ = θ1 ∼ √Δ, where Δ is the classical spectral gap, enables a quadratic speedup in state preparation via quantum phase estimation. The invention further introduces a heuristic protocol that applies the quantum walk operator U sequentially across a sequence of increasing β values without measurement, yielding a final state that, upon measurement, collapses to a low-energy configuration with high probability. Numerical simulations on a one-dimensional Ising chain show that this unitary heuristic achieves a time-to-solution scaling exponent of 0.42, surpassing the classical scaling of 1.0 and the theoretical quantum limit of 0.5. On sparse random Ising models, the same protocol achieves an exponent of 0.75, consistently outperforming both classical MCMC and Zeno-based adiabatic protocols. The total time to solution is defined as the product of the number of quantum walk applications and the number of repetitions needed to achieve a success probability of at least 1−δ. The Zeno-with-rewind protocol, which iteratively projects onto the instantaneous stationary state by alternating measurements and partial reversals, reduces the total time compared to standard Zeno, but the unitary heuristic performs even better, suggesting that phase randomization induced by sequential unitary evolution mimics the effect of measurement without the overhead. The cost of the move preparation V scales as O(N) T gates, the spin flip F as O(Nc) Toffoli gates where c is the maximum number of spins flipped per move, the reflection R as O(N) Toffoli gates, and the Boltzmann coin B as O(N log(1/ε)) T gates for accuracy ε. For sparse models, where each spin interacts with only a constant number of neighbors, the Boltzmann coin complexity becomes independent of the system size, rendering the entire walk operator scalable. The invention further introduces improved circuit layouts that parallelize the application of the Boltzmann coin by copying the coin qubit into a conjugate basis, allowing non-overlapping rotations to be applied simultaneously. Quantum signal processing is employed to approximate the exponential acceptance function using a truncated Fourier series, reducing the number of required rotations from exponential to polynomial in the interaction size. The Zeno algorithm is implemented by preparing an initial state at β = 0, applying a sequence of L quantum walks with βj = jβ/L, and measuring the eigenvalue θ = 0 after each step. The unitary implementation, by contrast, applies the same sequence of unitaries without measurement, and the final state is measured directly. The numerical results demonstrate that this unitary heuristic consistently outperforms both classical and measurement-based quantum protocols, suggesting a new paradigm for quantum optimization that leverages coherent evolution rather than projective measurement.

### IV. Example Computing Environments

- introduce classical computing environment
- describe processing device
- describe memory
- describe software for generating/synthesizing/controlling quantum-circuit Markov chain Monte Carlo (MCMC) techniques
- describe storage
- describe input devices
- describe output devices
- describe communication connections
- describe interconnection mechanism
- describe operating system software
- describe storage of instructions for software
- describe input device types
- describe output device types
- describe communication medium
- describe modulated data signal
- describe computer-readable media
- describe program modules
- describe computer-executable instructions
- describe local or distributed computing environment
- describe network topology
- describe client-server network
- describe distributed computing environment
- describe computing device architecture
- describe network types
- describe computing device types
- describe network protocols
- describe network implementation

The invention may be implemented on a classical computing system configured to generate, optimize, and simulate the quantum circuits required to execute the disclosed quantum walk algorithm. The processing device comprises one or more central processing units (CPUs) or graphics processing units (GPUs) capable of executing software modules that synthesize the quantum circuit for a given Ising model, including the decomposition of the Boltzmann coin into elementary gates, the optimization of the binary-tree structure for move preparation, and the scheduling of the sequence of β values for the adiabatic or unitary heuristic. The memory includes volatile and non-volatile storage for holding the model parameters, the generated quantum circuit description, and intermediate simulation results. Software modules are stored on computer-readable media and include a circuit compiler that translates the Hamiltonian and acceptance rule into a sequence of quantum gates, a simulator that computes the evolution of the quantum state under the walk operator, and an optimizer that minimizes the total time to solution by adjusting the number of walk applications, the β schedule, and the circuit layout. Input devices such as keyboards, mice, or network interfaces allow users to specify problem instances, while output devices such as displays or printers present the results, including the final configuration, energy landscape, and time-to-solution statistics. Communication connections enable the transfer of quantum circuit descriptions to remote quantum processors via classical networks, and the modulated data signal carries encoded instructions for gate application timing, qubit mapping, and error correction parameters. The invention may be deployed in a local computing environment where all components reside on a single machine, or in a distributed environment where the circuit synthesis occurs on a classical server and the execution is offloaded to a quantum processing unit via a cloud-based quantum computing service. Network topologies may include client-server, peer-to-peer, or hybrid architectures, and the communication medium may consist of wired or wireless protocols such as Ethernet, Wi-Fi, or optical fiber. The operating system software manages resource allocation, memory paging, and task scheduling for both classical simulation and quantum control tasks. The program modules are implemented as computer-executable instructions stored in non-transitory memory and may be distributed across multiple computing devices in a distributed computing environment, enabling parallel optimization of multiple problem instances or adaptive circuit compilation based on real-time feedback from quantum hardware.

### V. Further Example Embodiments

- describe quantum walk procedure
- describe Metropolis-Hastings rotation
- describe Glauber dynamics rotation
- describe preparing move register
- describe copying state of left register onto right register
- describe conditioned flipping of bit
- describe evaluating A_(xy) and preparing coin register
- describe uncomputing move register
- describe implementing transformation
- describe quantum walk procedure variations

The quantum walk procedure begins with the system register initialized in a uniform superposition over all possible configurations, the move register in the all-zero state, and the coin register in the |+⟩ state. The move preparation unitary V is applied to entangle the move register into an equal superposition over all possible moves, encoded in unary. The spin flip operator F is then applied, which conditionally flips the spins of the system register according to the selected move, using Toffoli gates controlled by the move register and the coin register. The Boltzmann coin operator B is subsequently applied, rotating the coin qubit by an angle determined by the energy difference between the current and proposed states, as defined by the Metropolis-Hastings rule or the Glauber dynamics rule, the latter employing a different functional form for the rotation angle that depends on the ratio of forward and reverse transition probabilities. The reflection operator R is then applied to invert the phase of the all-zero state in the move and coin registers, creating interference that suppresses transitions to high-energy states. The move register is then uncomputed by reversing the move preparation operation, returning it to the all-zero state and disentangling it from the system and coin registers. The entire sequence constitutes a single application of the quantum walk operator U. Variations of the procedure include the use of a parallelized classical walk that updates all spins simultaneously with a tunable probability, enabling a direct quantization without violating reversibility; the use of quantum signal processing to approximate the Boltzmann coin using a Fourier decomposition of the exponential acceptance function; and the use of ancillary qubits to copy the coin state into the σy basis, enabling parallel application of non-overlapping rotations. The invention further encompasses embodiments where the system register is encoded in a binary representation rather than unary, at the cost of increased circuit depth, and where the reflection operator is implemented using a single ancilla qubit and a multi-controlled NOT gate, trading qubit count for depth. The transformation may also be applied in reverse to prepare the inverse Boltzmann distribution, enabling the sampling of high-energy states for thermodynamic integration or free energy estimation.

### VI. Concluding Remarks

- summarize disclosed technology

The disclosed technology provides a novel, practical, and scalable method for implementing quantum walks in discrete optimization problems by eliminating the need for arithmetic-based oracles and replacing them with local, conditionally applied quantum gates that directly encode the acceptance criteria of classical Markov chains. The invention enables the construction of a unitary quantum walk operator that achieves polynomial speedups over classical MCMC methods, as demonstrated on Ising models with both regular and sparse connectivity. The architecture is compatible with near-term quantum hardware due to its low circuit depth, minimal qubit overhead, and avoidance of costly quantum arithmetic. The invention further introduces a heuristic protocol based on sequential unitary evolution without measurement that outperforms both classical and measurement-based quantum approaches, suggesting a new paradigm for quantum optimization that leverages coherent dynamics rather than projective measurement. The method is broadly applicable to any problem that can be formulated as energy minimization over a discrete configuration space with local interactions, including spin glasses, constraint satisfaction, machine learning inference, and combinatorial optimization. The disclosed circuit designs, gate decompositions, and heuristic protocols constitute a foundational advancement in the practical realization of quantum advantage for real-world computational problems.