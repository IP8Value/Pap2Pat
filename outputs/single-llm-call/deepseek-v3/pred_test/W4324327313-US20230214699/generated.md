Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION

## BACKGROUND

### 1. Technical Field

The present invention relates generally to quantum computing systems and methods, and more particularly to systems and methods for simulating stochastic processes using quantum computing architectures. The technical field encompasses quantum algorithms for financial modeling, statistical mechanics, and other applications requiring efficient simulation of stochastic systems.

### 2. Description of Related Art

Stochastic process simulation represents a fundamental challenge across multiple scientific and engineering disciplines. Classical approaches to stochastic process simulation face inherent limitations in computational efficiency, particularly when modeling complex systems with high-dimensional state spaces or long time horizons. Current quantum computing approaches to stochastic process simulation typically involve direct quantization of classical algorithms, which fails to exploit the unique capabilities of quantum information processing.

Quantum state representation offers potential advantages for stochastic process simulation through inherent parallelism and efficient encoding of probability distributions. However, existing quantum simulation methods for stochastic processes generally rely on digital representations that require excessive quantum resources. These methods typically map classical probability distributions directly to quantum states without leveraging the analog properties of quantum systems.

The limitations of current simulation methods become particularly apparent when applied to financial instruments and risk analysis. Existing quantum Monte Carlo methods for financial applications suffer from prohibitive gate counts and circuit depths when attempting to achieve practical speedups. The application of stochastic process simulation to quantitative finance requires new approaches that can overcome these resource constraints while maintaining accuracy.

## SUMMARY

The present invention introduces a novel quantum simulation method for stochastic processes that achieves significant improvements in computational efficiency compared to existing approaches. The key innovation involves an analog representation of stochastic processes that exploits the exponential nature of quantum state space while minimizing quantum resource requirements.

The simulation method comprises several key components including a quantum state representation of stochastic processes that encodes trajectory information in quantum amplitudes rather than digital values. The method further includes a basis state definition for quantum computing systems that enables efficient encoding of process trajectories. An amplitude function maps price trajectories or other stochastic variables to quantum state amplitudes in a manner that preserves essential statistical properties.

The invention motivates and enables the use of quantum algorithms for financial instruments through efficient preparation of mixed quantum states representing stochastic processes. The method defines mixed states of stochastic processes that can be manipulated using quantum operations while maintaining required statistical properties. A discrete cosine transform (DCT) of the mixed state provides an efficient pathway for quantum state preparation and manipulation.

The DCT implementation represents the real part of a Quantum Fourier Transform, enabling the development of efficient algorithms for financial applications. The invention outlines an efficient algorithm for performing DCT on quantum computing systems with polynomial rather than exponential resource requirements. This algorithm enables preparation of state σ using DCT operations while maintaining linearity properties crucial for stochastic process simulation.

## DETAILED DESCRIPTION

The invention introduces a quantum state representation of stochastic processes that enables efficient analog simulation. This representation encodes process trajectories in quantum amplitudes while using quantum basis states to represent time steps or other process parameters. The basis state of the quantum computing system is defined as |t⟩ where t represents discrete time steps in the stochastic process simulation.

The amplitude function of price trajectory or other stochastic variables is encoded as ψ(t) = v_t where v_t represents the value of the stochastic process at time t. This analog encoding allows representation of T time steps using only O(log T) qubits by storing values in amplitudes rather than explicitly in quantum state labels. The invention motivates the use of quantum algorithms for financial instruments by demonstrating efficient preparation of these analog representations.

The mixed state of the stochastic process is defined as ρ = E[|ψ⟩⟨ψ|] where the expectation is taken over all possible trajectories of the process. This mixed state representation preserves the statistical properties of the stochastic process while enabling quantum manipulation. The invention introduces a discrete cosine transform (DCT) of the mixed state that facilitates efficient quantum simulation.

The DCT is implemented as the real part of a Quantum Fourier Transform (QFT), leveraging existing efficient quantum circuits for Fourier operations. The invention outlines an efficient algorithm for performing DCT on quantum computing systems that requires only polynomial resources in terms of qubits and gate count. This algorithm enables preparation of state σ using DCT operations while maintaining the linearity properties required for accurate stochastic process simulation.

For example, in the case of Brownian motion, the DCT series coefficients follow a known power law distribution that can be efficiently prepared using quantum state loading techniques. The invention describes truncation of the DCT series to a finite number of terms while maintaining simulation accuracy through careful error analysis. A probability distribution over DCT coefficients is defined that preserves the statistical properties of the original stochastic process.

The goal of preparing quantum state σ' is achieved through a quantum data loader algorithm that efficiently encodes the required probability distributions. The data loader algorithm A operates by recursively applying controlled rotations to generate the target quantum state. The invention describes preparation of state σ' using the data loader by appropriately initializing rotation angles based on the target distribution.

An inverse map of data loader angles enables efficient state preparation by working backward from desired amplitudes to required rotation parameters. For Brownian motion, the data loader angles can be computed explicitly from the known power law distribution of Fourier coefficients. The invention further describes a "data loading the data loader" algorithm that recursively applies the state preparation procedure to generate more complex distributions.

The algorithm creates state |D'⟩ by first preparing a quantum state encoding the parameters needed for the final state preparation. Application of data loader algorithm A to |D'⟩ generates the target state through a sequence of controlled operations. Tracing out the first register after state preparation yields the desired mixed state σ' representing the stochastic process.

For Brownian motion specifically, the invention describes an alternative method that takes advantage of special properties of Gaussian distributions. The method includes normalization procedures for vectors involved in state preparation to maintain proper quantum state normalization. The overall method for simulating stochastic processes is summarized as a sequence of quantum operations that prepare, transform, and measure appropriate quantum states.

FIG. 1 illustrates a quantum circuit diagram implementing the disclosed stochastic process simulation method. The circuit shows the sequence of quantum gates and operations used to prepare the analog encoding of stochastic process trajectories. FIG. 2 presents a flowchart of the method for simulating stochastic processes, detailing the algorithmic steps from process specification to final quantum state preparation.

### Example Method

An example method for simulating stochastic processes comprises several key steps. The method begins by receiving a description of a stochastic process with trajectories defined over time. This description includes statistical properties and evolution rules for the process. The method then determines a first quantum circuit to prepare mixed quantum state ρ' that represents the stochastic process in analog form.

The determination of the DCT series of the stochastic process follows, where the series coefficients capture the frequency domain representation of process trajectories. A probability distribution for coefficients in the DCT series is computed based on the statistical properties of the original process. The method then determines a probability distribution of angles D' for the data loader based on the DCT coefficient distribution.

Finally, the method executes the first quantum circuit to generate mixed quantum state ρ' that represents the stochastic process with desired accuracy. This execution involves preparing initial quantum states, applying quantum gates for the DCT and data loading operations, and performing measurements to extract relevant statistical properties.

### Description of a Computing System

The invention includes a description of computing system 400 capable of implementing the disclosed quantum simulation methods. System 400 comprises classical computing system 410 coupled to quantum computing system 420. The classical computing system handles pre-processing of stochastic process descriptions and post-processing of quantum computation results, while the quantum computing system performs the core simulation operations.

Classical computing system 410 includes standard computer components configured to interface with quantum computing resources. Quantum computing system 420 contains qubits 450 arranged in a qubit register and controlled by qubit controllers 440. The system architecture provides separate qubit controller 440 for each qubit 450 to enable precise individual control.

The execution of quantum routines on computing system 400 follows a defined workflow. First, a quantum program is generated based on the stochastic process description and desired simulation parameters. This program is then executed on the quantum computing hardware, with results computed through quantum measurements. The results are recorded in classical memory for further analysis and application.

The architecture of classical computing system 410 includes processor 502 for general computation tasks and chipset 504 for component interconnection. Memory 506 provides temporary storage, while storage device 508 offers persistent data retention. User interface components include keyboard 510, pointing device 514, and display 518. Graphics adapter 512 handles visual output, and network adapter 516 enables system communication.

### Additional Considerations

The embodiments described herein are provided for purposes of illustration and do not limit the scope of the invention. The essential features of the invention include the analog representation of stochastic processes and efficient quantum algorithms for preparing and manipulating these representations. The algorithmic processes disclosed can be implemented as software modules or hardware circuits in various computing environments.

In one embodiment, the invention is implemented as a computer program product containing instructions for programmable processors. The inclusive or construction is used throughout to indicate that any combination of the described features may be utilized. Approximate values and ranges are provided to account for implementation variations and practical considerations.

Alternative embodiments may employ different quantum state preparation techniques or modified circuit designs while maintaining the core inventive concepts. The data storage system requirements vary based on implementation scale and desired performance characteristics. Modifications and variations of the disclosed methods will be apparent to practitioners in the field while remaining within the scope of the invention.