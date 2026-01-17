# DESCRIPTION

## FIELD

The present invention relates to the field of quantum computing, specifically to methods and systems for implementing quantum walks and their applications in solving discrete optimization problems. The invention provides a novel circuit design for the walk operator of a quantum walk, which is particularly useful in the context of the Metropolis-Hastings algorithm and other Markov Chain Monte Carlo (MCMC) methods. The invention also includes heuristic quantum algorithms that leverage these quantum walks to achieve significant speed-ups over classical counterparts.

## SUMMARY

The invention provides a method for implementing a quantum walk operator in a quantum computing system. The quantum walk operator is designed to efficiently simulate a classical walk, such as the Metropolis-Hastings algorithm, without the need for costly arithmetic operations. The method includes the preparation of a move register, the application of a spin flip operation, a reflection operation, and a Boltzmann coin operation. The invention further provides heuristic quantum algorithms that use the quantum walk operator to solve discrete optimization problems, demonstrating a polynomial speed-up over classical methods.

## DETAILED DESCRIPTION

### I. Introduction

Quantum computing offers the potential for significant speed-ups in solving complex computational problems, including discrete optimization tasks. One of the key techniques in this domain is the quantum walk, which can be used to simulate classical Markov Chain Monte Carlo (MCMC) methods such as the Metropolis-Hastings algorithm. However, the efficient implementation of quantum walks on a quantum computer is challenging, particularly due to the need for costly arithmetic operations. The present invention addresses this challenge by providing a novel circuit design for the walk operator of a quantum walk, which avoids these costly operations and enables efficient simulation of classical walks.

### II. General Considerations

The invention is based on the observation that the traditional approach to implementing a quantum walk, which relies on an oracle formulation of the walk operator, can be computationally expensive. Instead, the invention provides a detailed and simplified implementation of the walk operator, which circumvents the use of costly arithmetic operations. This is achieved by breaking down the walk operator into several components: move preparation, spin flip, reflection, and Boltzmann coin. Each component is designed to be efficient and scalable, making the overall implementation suitable for practical quantum computing applications.

### III. Example Embodiments

#### A. Move Preparation

The move preparation component of the walk operator prepares a move register in a state that represents a set of possible transitions in the classical walk. For a uniform distribution of moves, this can be achieved using a sequence of \(\sqrt{\text{SWAP}}\) gates arranged in a binary-tree fashion. This method is particularly efficient when the number of moves \(N\) is a power of 2. If \(N\) is not a power of 2, the distribution can be padded with additional states to maintain the efficiency of the circuit.

#### B. Spin Flip

The spin flip component applies a set of controlled-controlled-NOT (Toffoli) gates to flip a set of system spins conditioned on the coin qubit and the move register. This operation can be implemented in a sequential manner, but a parallel implementation using additional scratchpad qubits can reduce the circuit depth. The parallel implementation uses a binary-tree structure of CNOT gates to make multiple copies of the coin qubit, followed by the application of Toffoli gates in parallel.

#### C. Reflection

The reflection component is a reflection about the state \(|00...0\rangle\) in the move register and the coin qubit. This can be implemented using a single additional qubit and an open-control (N+1)-NOT gate. The open-control (N+1)-NOT gate can be realized using a binary tree of Toffoli gates, which minimizes the circuit depth.

#### D. Boltzmann Coin

The Boltzmann coin component applies a sequence of conditional rotations to the coin qubit, conditioned on the state of the system register and the move register. Each rotation is determined by the energy difference between the current state and the proposed state. The complexity of this component scales with the sparsity parameters of the model, but it can be optimized using quantum signal processing methods or by parallelizing the rotations.

### IV. Example Computing Environments

The invention can be implemented on a variety of quantum computing platforms, including superconducting qubits, ion traps, and photonic qubits. The specific hardware requirements will depend on the size and complexity of the problem being solved. For example, a small-scale problem might be solvable on a near-term quantum device with a few dozen qubits, while larger problems may require more advanced quantum computers with hundreds or thousands of qubits.

### V. Further Example Embodiments

#### A. Heuristic Use of Quantum Walks

The invention also provides heuristic quantum algorithms that use the quantum walk operator to solve discrete optimization problems. Two main heuristics are proposed: the Zeno with rewind algorithm and the unitary implementation algorithm.

1. **Zeno with Rewind**: This algorithm uses a sequence of quantum walks with gradually increasing parameters to prepare the eigenstate of the final walk operator. If a measurement fails, the algorithm rewinds to a previous state and continues. This method has been shown to offer significant speed-ups over classical methods, particularly for certain types of problems.

2. **Unitary Implementation**: This algorithm applies the quantum walk operators sequentially without the need for measurements. The final state is measured in the computational basis to obtain the solution. This method has been shown to be more efficient than the Zeno with rewind algorithm in some cases, achieving a polynomial speed-up over classical methods.

#### B. Numerical Results

Numerical simulations have been conducted to benchmark the performance of the heuristic quantum algorithms against classical methods. The results indicate a polynomial speed-up for both the Zeno with rewind and the unitary implementation algorithms, with the unitary implementation showing a particularly significant improvement over classical methods.

### VI. Concluding Remarks

The invention provides a novel and efficient method for implementing quantum walks on a quantum computer, which can be used to solve discrete optimization problems with significant speed-ups over classical methods. The detailed circuit design for the walk operator and the heuristic quantum algorithms presented in this invention pave the way for practical applications of quantum computing in fields such as statistical physics, machine learning, and combinatorial optimization. Future work will focus on optimizing the circuit implementations and exploring the broader class of problems that can benefit from these quantum speed-ups.