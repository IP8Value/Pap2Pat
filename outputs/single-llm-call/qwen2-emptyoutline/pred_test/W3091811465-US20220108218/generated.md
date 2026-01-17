# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of quantum-assisted machine learning (QAML) and, more specifically, to methods and systems for compiling tensor network (TN) models, particularly matrix product states (MPS), for execution on noisy intermediate-scale quantum (NISQ) devices. The invention provides a comprehensive workflow for translating classical data into quantum states, optimizing TN models using classical techniques, and converting these models into resource-efficient sequential preparation schemes suitable for deployment on quantum hardware. The invention is particularly useful for generative unsupervised learning tasks, where the goal is to learn a probability distribution over data and generate new samples from this distribution.

## BACKGROUND

Quantum computing has emerged as a promising technology for solving complex computational problems that are infeasible for classical computers. Gate-based quantum computing platforms, such as those offered by IBM, Google, and Rigetti, have made significant strides in recent years, providing access to quantum devices with a few to dozens of qubits. However, these devices are still far from the millions of qubits required for canonical quantum computing tasks such as integer factorization with error correction. Current quantum devices, known as noisy intermediate-scale quantum (NISQ) devices, face challenges such as hardware noise, limited qubit connectivity, and restricted gate sets, which pose significant obstacles for demonstrating scalable universal quantum computation.

Machine learning (ML) has been proposed as a potential application area for NISQ devices. Quantum-assisted machine learning (QAML) leverages the unique properties of quantum systems to enhance classical ML algorithms. One promising approach is to use tensor networks (TNs), particularly matrix product states (MPS), to design parameterized quantum circuits that are resource-efficient and can be implemented on both classical and quantum hardware. TNs provide a robust means of designing such circuits and enable detailed benchmarking and optimization of QAML models.

However, translating TN models into operations for NISQ devices is a non-trivial task. The isometric operations defined by TN models must be compiled into native quantum gates, taking into account the hardware constraints such as limited coherence time, connectivity, and gate sets. Existing methods for quantum compilation often produce deep circuits with a high number of entangling gates, which can lead to increased noise and reduced fidelity. Therefore, there is a need for novel techniques that can efficiently compile TN models into shallow gate sequences that are robust to noise and hardware limitations.

## BRIEF SUMMARY

The present invention addresses the challenges of compiling tensor network (TN) models, particularly matrix product states (MPS), for execution on noisy intermediate-scale quantum (NISQ) devices. The invention provides a comprehensive workflow for quantum-assisted machine learning (QAML) that includes the following key components:

1. **Data Embedding**: A method for mapping classical data vectors into quantum states using a binary encoding scheme. This ensures that the quantum states are unentangled and can be prepared with high fidelity.

2. **Model Optimization**: A classical optimization procedure for training MPS models using a density matrix renormalization group (DMRG)-like algorithm with gradient descent. The optimization minimizes the negative log-likelihood of the training data, ensuring that the model accurately represents the probability distribution of the data.

3. **Sequential Preparation**: A technique for converting the optimized MPS model into a sequential preparation scheme that can be executed on quantum hardware. The scheme uses a single physical qubit and a χ-level ancilla to generate data samples, requiring only O(1) qubits for the physical data vector length N and O(log_2 χ) qubits for the bond dimension χ.

4. **Diagonal Gauge Transformation**: A method for transforming the MPS model into a diagonal gauge, which utilizes the inherent freedom in the model representation to reduce the complexity of the compiled model. The diagonal gauge ensures that the isometric operations are as diagonal as possible, minimizing the number of quantum operations required.

5. **Greedy Compilation Heuristics**: A set of greedy heuristics for compiling the isometric operations into shallow gate sequences that match the target hardware topology and allowed gate set. The heuristics use a cost function to select the most efficient gate sequences and include techniques for handling permutation and sign ambiguities in the model.

6. **Performance Assessment**: Methods for assessing the performance of the compiled QAML models on NISQ devices, including the use of hardware simulators to model the effects of depolarizing and readout noise. The performance is evaluated using metrics such as the Kullback-Leibler (KL) divergence between the ideal and estimated probability distributions.

The invention is particularly useful for generative unsupervised learning tasks, where the goal is to learn a probability distribution over data and generate new samples from this distribution. By providing a robust and efficient workflow for compiling TN models for NISQ devices, the invention enables the practical application of QAML in near-term quantum computing scenarios.

## DETAILED DESCRIPTION

### Data Embedding

The first step in the QAML workflow is to map classical data vectors into quantum states. The invention uses a binary encoding scheme where each element of the classical data vector \( x_i \) is mapped to a qubit state \( |x_i \rangle \). Specifically, for a binary data vector \( x \) of length \( N \), the encoding is given by:

\[ |x \rangle = |x_0 \rangle \otimes |x_1 \rangle \otimes \cdots \otimes |x_{N-1} \rangle \]

This encoding ensures that the quantum states are unentangled and can be prepared with high fidelity, making it suitable for NISQ devices.

### Model Optimization

The next step is to optimize the MPS model using a classical DMRG-like algorithm with gradient descent. The optimization minimizes the negative log-likelihood of the training data, which is defined as:

\[ L(T) = -\frac{1}{N_T} \sum_{j=1}^{N_T} \log P(x_j) \]

where \( P(x_j) \) is the probability of the data vector \( x_j \) according to the model, and \( N_T \) is the number of training data vectors. The optimization updates the MPS tensors using the gradient of the negative log-likelihood with respect to the tensor elements. The gradient is given by:

\[ \nabla_{A^{[i]}} L(T) = \sum_{j=1}^{N_T} \left( \langle x_j | A^{[i]} | x_j \rangle - \langle \psi | A^{[i]} | \psi \rangle \right) \]

where \( A^{[i]} \) is the MPS tensor at site \( i \), and \( |\psi \rangle \) is the wave function representing the model. The optimization proceeds by iteratively updating the tensors using a learning rate \( \eta \):

\[ A^{[i]} \rightarrow A^{[i]} - \eta \nabla_{A^{[i]}} L(T) \]

### Sequential Preparation

Once the MPS model is optimized, it is converted into a sequential preparation scheme that can be executed on quantum hardware. The scheme uses a single physical qubit and a χ-level ancilla to generate data samples. The isometric operations \( L^{[i]} \) are defined such that:

\[ L^{[i]} : |0 \rangle |0 \rangle \rightarrow \sum_{j=0}^{\chi-1} \sum_{q=0}^{1} L^{[i]}_{j,q} |j \rangle |q \rangle \]

where \( |0 \rangle \) is the initial state of the physical qubit and ancilla, and \( L^{[i]}_{j,q} \) are the elements of the isometric operator. The physical qubit is measured in the computational basis, and the outcome is recorded as the data sample element. The physical qubit is then reinitialized to the \( |0 \rangle \) state, and the process is repeated for the next site.

### Diagonal Gauge Transformation

To reduce the complexity of the compiled model, the invention introduces a diagonal gauge transformation. The transformation utilizes the inherent freedom in the representation of the ancilla states to make the isometric operations as diagonal as possible. The diagonal gauge is achieved by applying a permutation to the ancilla basis states, which is determined using the polar decomposition of the overlap matrix \( M^{[i]} \):

\[ M^{[i]} = U^{[i]} P^{[i]} \]

where \( U^{[i]} \) is a unitary matrix and \( P^{[i]} \) is a Hermitian and positive semidefinite matrix. The permutation is chosen to maximize the diagonal dominance of the isometric operations, reducing the number of quantum operations required.

### Greedy Compilation Heuristics

The invention provides a set of greedy heuristics for compiling the isometric operations into shallow gate sequences that match the target hardware topology and allowed gate set. The heuristics use a cost function to select the most efficient gate sequences and include techniques for handling permutation and sign ambiguities in the model. The cost function is defined as:

\[ C(L, U) = \sum_{(i,j) \in S} |L_{i,j} - U_{i,j}|^2 \]

where \( L \) is the isometric operator, \( U \) is the candidate unitary gate, and \( S \) is the set of indices where the elements of \( L \) are greater than a tolerance \( \delta \). The heuristics proceed by optimizing the root node (single-qubit gates) and then adding entangling gates and single-qubit rotations in a greedy manner, selecting the gates with the lowest cost function.

### Performance Assessment

The performance of the compiled QAML models is assessed using hardware simulators to model the effects of depolarizing and readout noise. The performance is evaluated using metrics such as the Kullback-Leibler (KL) divergence between the ideal and estimated probability distributions. The KL divergence is defined as:

\[ D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \]

where \( P(x) \) is the ideal probability distribution and \( Q(x) \) is the estimated probability distribution. The KL divergence provides a measure of the fidelity of the model and helps identify the impact of hardware noise on the performance.

### Example Applications

The invention is demonstrated using two example applications: an exactly solvable benchmark model and a generative model for the MNIST dataset. The benchmark model is a simple nontrivial example of a sequentially preparable QAML model, involving a single ancilla qubit. The MNIST dataset consists of grayscale images of handwritten digits, and the QAML model is used to learn the probability distribution of the data and generate new samples. The performance of the model is assessed on cloud-based NISQ devices, and the results show that the compiled models achieve high fidelity even in the presence of hardware noise.

### Conclusion

The present invention provides a comprehensive workflow for quantum-assisted machine learning (QAML) using tensor network (TN) models, particularly matrix product states (MPS). The workflow includes data embedding, model optimization, sequential preparation, diagonal gauge transformation, greedy compilation heuristics, and performance assessment. By addressing the challenges of compiling TN models for NISQ devices, the invention enables the practical application of QAML in near-term quantum computing scenarios. The invention is particularly useful for generative unsupervised learning tasks and lays the groundwork for further research in the field of quantum machine learning.