# DESCRIPTION

## FIELD

The present invention relates to a scalable architecture for fault-tolerant measurement-based quantum computing using a hybrid resource state comprising Gottesman-Kitaev-Preskill (GKP) qubits and squeezed states of light. The architecture leverages the advantages of photonic technologies, including room-temperature operation, intrinsic compatibility with communication technology, and flexibility in error-correcting codes.

## BACKGROUND

Quantum computing holds the promise of solving problems that are intractable for classical computers. Photonic technologies offer several advantages for building scalable and fault-tolerant quantum computers, including the ability to operate at room temperature, compatibility with communication technology, and flexibility in error-correcting codes. However, current architectures for photonic quantum computing face significant challenges, particularly in the deterministic generation of GKP qubits, which are essential for non-Gaussian operations and error correction.

Traditional approaches to photonic quantum computing can be categorized into two extremes:
1. **Continuous-Variable (CV) Entangled Resource States**: These architectures use CV cluster states to implement computation on discrete-variable (DV) information encoded in bosonic modes. While CV resource states can be generated deterministically and scalably, they require DV resources to be generated on-demand and deterministically, which imposes infeasible hardware requirements.
2. **Discrete-Variable (DV) Resource States**: These architectures rely on generating entangled resource states made entirely out of high-quality bosonic qubits. While these states are resilient to noise, their generation and entanglement operations are probabilistic, leading to significant multiplexing requirements.

The present invention addresses these challenges by proposing a hybrid resource state that combines the benefits of both CV and DV approaches. Specifically, the invention uses a lattice of GKP qubits and squeezed states of light, where GKP qubits are generated using Gaussian Boson Sampling (GBS) devices. When GBS devices fail to produce a GKP qubit, the mode is guaranteed to be prepared in a squeezed state, which can still encode logical information and participate in the computation.

## SUMMARY

The invention provides a scalable architecture for fault-tolerant measurement-based quantum computing using a hybrid resource state. The key features of the architecture include:

- **Hybrid Resource State**: The resource state is a lattice of GKP qubits and squeezed states of light. GKP qubits are generated using GBS devices, and when these devices fail, the mode is prepared in a squeezed state.
- **Modular Design**: The architecture is modular, with different components dedicated to state preparation, multiplexing, cluster state generation, and measurement-based quantum computation.
- **Minimal Cryogenic Requirements**: Most components of the architecture can operate at room temperature, with only the state generation modules requiring cryogenic conditions.
- **Fast Clock Speeds**: The architecture supports fast clock speeds, enabling the use of short delay lines and reducing losses.
- **Error Correction**: The architecture includes a novel decoding procedure that accounts for the hybrid nature of the resource state and the correlated noise introduced by squeezed states.

The invention enables scalable fault-tolerant quantum computation with optically-generated GKP states or squeezed states of light, using a room-temperature, moderately-sized planar photonic chip. The architecture is designed to be compatible with existing silicon electronics and photonics technology, facilitating mass manufacturing and rapid scaling to large numbers of qubits.

## DETAILED DESCRIPTION

### INTRODUCTION

The invention proposes a scalable architecture for fault-tolerant measurement-based quantum computing that overcomes the limitations of existing photonic architectures. The architecture leverages a hybrid resource state comprising GKP qubits and squeezed states of light, generated using GBS devices. This approach combines the advantages of CV and DV resources, enabling efficient and scalable quantum computation.

### Overview of the System Configuration

The system configuration consists of four main modules:
1. **State-Preparation Module**: Generates high-quality GKP qubits using GBS devices.
2. **Multiplexing Module**: Boosts the qubit generation rates and substitutes momentum-squeezed vacuum modes when GBS devices fail.
3. **Computational Module**: Implements deterministic entangling operations to generate the hybrid resource state.
4. **Photonic Quantum Processing Unit (QPU)**: Performs homodyne measurements on the resource state to execute the quantum computation.

### Generation of Bosonic Qubits Using Multiplexed Gaussian Boson Sampling (GBS) Devices

GBS devices are used to generate GKP qubits probabilistically. The success probability of these devices can be boosted using spatial multiplexing, where multiple GBS devices are run in parallel. Active feed-forwarding of photon number resolving (PNR) detector outcomes is used to route successfully prepared states to the correct output port. If no GBS device successfully produces a GKP state, a momentum-squeezed state is substituted.

### Time Domain Generation of 1D Clusters

The first step in generating the resource state is to create one-dimensional hybrid cluster states that extend in the temporal domain. This is achieved using a linear cluster state generation setup involving a pair of fast actively switchable beam-splitters, controllable phase shifters, a delay line, and inline squeezers. The delay line is set to one clock period and is required to be phase stable, making integrated implementations preferable.

### GKP Cluster in 2+1 Dimensions

The 1D cluster states are extended to 2+1 dimensions by implementing additional CZ gates in the two spatial dimensions. A 2D spatial array of 1D time-domain cluster state sources is used, interspersed by additional state-preparation modules and connected in the spatial domain by a nearest-neighbor array of optical CZ gates. This results in a 3D lattice structure suitable for fault-tolerant quantum computation.

### Generation of a Hybrid Raussendorf-Harrington-Goyal (RHG) Lattice

The hybrid RHG lattice is generated by combining the 2+1D cluster state with the structure of the RHG model. The lattice is designed to ensure that any noise does not extend beyond a qubit's neighbors, which is crucial for fault tolerance. The lattice is composed of GKP qubits and squeezed states, with the latter substituting for failed GKP qubit generation.

### Passive Version of System Configuration

A passive version of the system configuration is also possible, where the CZ gates on the computational module are replaced by passive transformations. This simplifies the computational module and reduces experimental requirements. Techniques such as macronode-based architectures or methods demonstrated in the literature can be used to achieve this.

### 3D Hybrid Macronode Architecture

The 3D hybrid macronode architecture is designed to further simplify the computational module and reduce the experimental requirements. Macronodes are larger units that can be pre-fabricated and integrated into the system, reducing the complexity of the overall architecture. This approach is particularly useful for large-scale quantum computing.

## Technological Advantages

### Modularity

The architecture is modular, with different components dedicated to specific tasks. This modularity allows for easier fabrication, maintenance, and scaling. State preparation, multiplexing, cluster state generation, and measurement-based quantum computation are handled by separate modules, each optimized for its specific requirements.

### Minimal Cryogenic Requirements

Most components of the architecture can operate at room temperature, with only the state generation modules requiring cryogenic conditions. This significantly reduces the complexity and cost of the system, making it more practical for widespread deployment.

### Homodyne Detection Sets Timescales

The timescale of the cluster generation and manipulation is ultimately set by the timescales of homodyne detection, which can be much faster than PNR detectors used in the multiplexing procedure or threshold detectors used in other photonic encodings. This allows for the use of short delay lines and reduces losses, enabling very low-loss delay lines on the chip.

The invention provides a comprehensive and scalable solution for fault-tolerant measurement-based quantum computing, leveraging the unique advantages of photonic technologies. The hybrid resource state and modular design ensure that the architecture is both practical and efficient, paving the way for the realization of large-scale quantum computers.