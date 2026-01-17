# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of quantum computing, specifically to hybrid quantum/classical variational algorithms designed for solving combinatorial optimization (CO) problems on noisy intermediate-scale quantum (NISQ) computers. The invention introduces a novel approach to optimizing the performance of these algorithms by utilizing the Conditional Value-at-Risk (CVaR) as an aggregation function for the samples obtained from trial wavefunctions, thereby improving the convergence to better solutions.

## BACKGROUND

Combinatorial optimization (CO) problems are prevalent in various fields, including business, science, and engineering. Despite being NP-hard, these problems are often solved using heuristic methods in industrial applications. Quantum computers offer the potential to solve CO problems more efficiently, particularly through hybrid quantum/classical algorithms such as the Variational Quantum Eigensolver (VQE) and the Quantum Approximate Optimization Algorithm (QAOA). These algorithms leverage parametrized quantum circuits to generate trial wavefunctions, which are then optimized using classical algorithms.

However, the performance of VQE and QAOA in the context of CO problems has been suboptimal. Traditional approaches minimize the expected value of the problem Hamiltonian, which may not align well with the practical goal of finding the best solution. The expected value is a statistical measure that averages over all possible outcomes, potentially diluting the focus on the best solutions. This limitation is particularly pronounced in the context of CO problems, where the Hamiltonian is diagonal, and the goal is to find the ground state with the lowest energy.

To address these limitations, the present invention proposes the use of the Conditional Value-at-Risk (CVaR) as an alternative aggregation function. CVaR is a risk management tool widely used in finance, which focuses on the tail of the probability distribution. By minimizing the CVaR of the samples obtained from the trial wavefunctions, the invention aims to improve the convergence to better solutions and enhance the robustness of VQE and QAOA in the context of CO problems.

## SUMMARY

The present invention provides a method for optimizing the performance of hybrid quantum/classical variational algorithms for solving combinatorial optimization (CO) problems on noisy intermediate-scale quantum (NISQ) computers. The method involves the following steps:

1. **Mapping the CO Problem to a Hamiltonian**: The CO problem is transformed into a Hamiltonian, which encodes the total energy of the system. This Hamiltonian is diagonal, reflecting the nature of CO problems.

2. **Generating Trial Wavefunctions**: Parametrized quantum circuits are used to generate trial wavefunctions. These circuits are designed to span all basis states, ensuring that the ground state can be reached.

3. **Aggregating Samples Using CVaR**: Instead of minimizing the expected value of the Hamiltonian, the method minimizes the Conditional Value-at-Risk (CVaR) of the samples obtained from the trial wavefunctions. CVaR is a statistical measure that focuses on the tail of the probability distribution, thereby emphasizing the best solutions.

4. **Classical Optimization**: A classical optimization algorithm is used to optimize the parameters of the trial wavefunctions. The objective function for the classical optimization is the CVaR of the samples, which guides the optimization process towards states with a higher probability of sampling the ground state.

5. **Convergence and Robustness**: The use of CVaR as the aggregation function leads to faster convergence to better solutions and enhances the robustness of the variational algorithms in the presence of noise and errors on NISQ computers.

The invention further provides a detailed analysis of the performance of the proposed method, including empirical evaluations using both classical quantum simulators and actual quantum hardware. The results demonstrate that the CVaR-based approach significantly improves the performance and robustness of VQE and QAOA, making them more practical for solving CO problems on NISQ computers.

## DETAILED DESCRIPTION

### Mapping the CO Problem to a Hamiltonian

Combinatorial optimization (CO) problems can be mapped to a Hamiltonian, which encodes the total energy of the system. For a quadratic unconstrained binary optimization (QUBO) problem on \( n \) variables, the problem can be expressed as:

\[
\min_{x \in \{0, 1\}^n} \left( \sum_{i=1}^n b_i x_i + \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j \right)
\]

Using the variable transformation \( x_i = \frac{1 - z_i}{2} \) for \( z_i \in \{-1, +1\} \), the QUBO problem can be transformed into an Ising spin glass model:

\[
\min_{z \in \{-1, +1\}^n} \left( c + \sum_{i=1}^n Q_{ii} z_i + \sum_{i=1}^n \sum_{j=1}^n Q_{ij} z_i z_j \right)
\]

The Ising model can be translated into a Hamiltonian for an \( n \)-qubit system by replacing \( z_i \) with the Pauli \( Z \)-matrix acting on the \( i \)-th qubit and each term of the form \( z_i z_j \) with \( \sigma_i^Z \otimes \sigma_j^Z \). The resulting Hamiltonian is diagonal, with eigenvalues corresponding to the objective function values of the CO problem.

### Generating Trial Wavefunctions

Parametrized quantum circuits are used to generate trial wavefunctions. For VQE, a standard variational form is employed, consisting of layers of single-qubit Y-rotations and controlled Z-gates. For \( n \) qubits and depth parameter \( p \), the variational form is defined as:

\[
U(\theta) = \prod_{k=1}^p \left( \prod_{i=1}^{n-1} \text{CZ}_{i,i+1} \right) \left( \prod_{i=1}^n R_Y(\theta_{k,i}) \right)
\]

For QAOA, the variational form is derived from the problem Hamiltonian \( H \) and consists of alternating layers of unitaries \( U_B \) and \( U_C \):

\[
U(\beta, \gamma) = \prod_{k=1}^p \left( e^{-i\beta_k H_B} e^{-i\gamma_k H_C} \right)
\]

where \( H_B = \sum_{i=1}^n \sigma_i^X \) and \( H_C = H \).

### Aggregating Samples Using CVaR

Instead of minimizing the expected value of the Hamiltonian, the method minimizes the Conditional Value-at-Risk (CVaR) of the samples obtained from the trial wavefunctions. CVaR is defined as the expected value of the lower \( \alpha \)-tail of the distribution of the samples. For a set of samples \( \{H_k\}_{k=1}^K \) and a confidence level \( \alpha \), the CVaR is given by:

\[
\text{CVaR}_\alpha = \frac{1}{\alpha K} \sum_{k=1}^{\lfloor \alpha K \rfloor} H_{(k)}
\]

where \( H_{(k)} \) are the ordered samples.

### Classical Optimization

A classical optimization algorithm, such as COBYLA, is used to optimize the parameters of the trial wavefunctions. The objective function for the classical optimization is the CVaR of the samples, which guides the optimization process towards states with a higher probability of sampling the ground state.

### Convergence and Robustness

Empirical evaluations using both classical quantum simulators and actual quantum hardware demonstrate that the CVaR-based approach leads to faster convergence to better solutions and enhances the robustness of the variational algorithms in the presence of noise and errors on NISQ computers. The results show that for small-depth circuits, the CVaR-based approach significantly outperforms the traditional expected value approach, particularly for CO problems.

### Implementation Details

The implementation of the proposed method involves the following steps:

1. **Constructing the Hamiltonian**: The CO problem is mapped to a Hamiltonian using the transformations described above.
2. **Generating Trial Wavefunctions**: Parametrized quantum circuits are constructed to generate trial wavefunctions.
3. **Sampling and Aggregation**: The quantum circuits are executed to obtain samples from the trial wavefunctions, and the CVaR of the samples is computed.
4. **Optimization**: The classical optimization algorithm is used to optimize the parameters of the trial wavefunctions, with the CVaR as the objective function.
5. **Evaluation**: The performance of the optimized trial wavefunctions is evaluated by measuring the probability of sampling the ground state and the convergence of the objective function values.

### Experimental Results

#### Simulation Results

The performance of the CVaR-based approach is evaluated using classical quantum simulators. The results show that for a variety of CO problems, including maximum stable set, maximum 3-satisfiability, number partitioning, maximum cut, market split, and portfolio optimization, the CVaR-based approach leads to faster convergence to better solutions. Specifically, for small-depth circuits, the CVaR-based approach achieves a higher probability of sampling the ground state compared to the traditional expected value approach.

#### Quantum Hardware Results

The CVaR-based approach is also tested on actual quantum hardware, specifically the IBM Q Poughkeepsie 20-qubit quantum computer. The results demonstrate that the CVaR-based approach is robust to noise and errors, and it consistently outperforms the traditional expected value approach in terms of the probability of sampling the ground state and the convergence of the objective function values.

### Conclusion

The present invention provides a novel method for optimizing the performance of hybrid quantum/classical variational algorithms for solving combinatorial optimization (CO) problems on noisy intermediate-scale quantum (NISQ) computers. By utilizing the Conditional Value-at-Risk (CVaR) as an aggregation function, the method improves the convergence to better solutions and enhances the robustness of the algorithms in the presence of noise and errors. The empirical evaluations using both classical quantum simulators and actual quantum hardware demonstrate the effectiveness of the proposed method, making it a promising approach for practical applications of quantum computing in CO.