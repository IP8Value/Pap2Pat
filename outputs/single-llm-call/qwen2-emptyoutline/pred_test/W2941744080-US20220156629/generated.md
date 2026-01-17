# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a surface code architecture for quantum computing using silicon spin (SS) qubits. More specifically, the invention addresses the design and implementation of a fault-tolerant quantum computing system that is robust against leakage errors, which can significantly impact the performance and reliability of quantum error correction codes.

## BACKGROUND

Quantum computing holds the promise of solving problems that are intractable for classical computers. However, the realization of practical quantum computers faces significant challenges, including the control and correction of errors in quantum states. Quantum error correction (QEC) codes, such as the surface code, are essential for achieving fault-tolerant quantum computing. The surface code is particularly attractive due to its high error threshold and local checking operations, making it suitable for implementation in various quantum computing platforms.

Silicon spin (SS) qubits are a promising candidate for scalable quantum computing due to their high coherence times and the potential for integration with existing semiconductor technology. However, scaling up SS qubits introduces practical challenges, such as the need for efficient qubit addressing, minimizing cross-talk, and managing heat dissipation. Additionally, leakage errors, where the quantum system escapes out of the computational subspace, pose a significant threat to the effectiveness of QEC codes.

Leakage errors in SS qubits can arise from charge migration, which can propagate through the quantum dot array and corrupt the surface code. Traditional QEC protocols are not designed to handle such correlated errors, making the development of leakage-resilient architectures crucial for the success of fault-tolerant quantum computing.

## SUMMARY

The present invention provides a surface code architecture for silicon spin qubits that is robust against leakage errors. The architecture includes elongated mediator dots to facilitate two-qubit gates while increasing the inter-qubit spacing, thereby accommodating measuring devices and classical control lines. The use of double-dot ancillae allows for parallel operations and reduces the complexity of the system. The architecture is designed to transfer charge leakage errors from the qubit dots to the mediator dots, which can be reset using charge reservoirs, effectively reducing leakage errors to the level of standard computational errors that can be corrected by the surface code.

Key features of the invention include:
1. **Elongated Mediator Dots**: These dots facilitate two-qubit gates and provide space for measuring devices and classical control lines.
2. **Double-Dot Ancillae**: These ancillae allow for parallel operations and reduce the complexity of the system.
3. **Charge Reservoirs**: These reservoirs are used to restore charge in the mediator dots, ensuring that charge leakage errors do not propagate through the qubit array.
4. **Robustness Against Spin and Charge Leakage Errors**: The architecture is designed to handle both spin and charge leakage errors, ensuring that the surface code remains effective.

The invention also includes methods for simulating the performance of the surface code under various error models, demonstrating the robustness of the architecture against leakage errors and other computational errors.

## DETAILED DESCRIPTION

### Physical Implementation

#### Data Qubits and Single-Qubit Gates

Each data qubit is represented by the spin state of an electron within an electrostatically-defined quantum dot. The lifting of the spin degeneracy via an applied magnetic field allows for electron-spin resonance (ESR) or electrically-driven spin resonance (EDSR) techniques, which have achieved control fidelities of up to 99.6-99.9% in silicon. The coherence times of these qubits can be extended using isotopically enriched silicon substrates and decoupling schemes, yielding long qubit operation times.

#### Ancilla Qubits and Read-out

The ancilla qubit is represented by the spin state of a pair of electrons distributed across two quantum dots. By initializing in a singlet state, a failed stabilizer check of neighboring data qubits transforms the ancilla spins into a triplet state. Pauli spin blockade (PSB) and its effect on interdot tunnelling can be used to determine the outcome of the stabilizer cycle. The ancilla qubits are initialized via the (0,2) electron occupation state of the double quantum dot, and the ground state is a singlet that can be rapidly prepared through 'hot-spot' relaxation near the (1,1):(0,2) charge transition.

#### Mediators and Two-Qubit Gates

Elongated quantum dots are used as mediators to extend the range of the exchange interaction between data and ancilla qubits. The mediators do not carry quantum information and are used to facilitate two-qubit gates. The strength of the exchange interaction can be controlled by tuning the on-site energy of the mediator dot, allowing for the implementation of √SWAP and S gates. The mediated exchange interaction is more readily switchable than direct exchange, leading to higher expected fidelity.

#### Realizations of the Ω ≫ J and Ω ≪ J Regimes

The RKKY exchange operation produced by the mediator dot can be utilized to construct the CZ operation either directly via the S gate when Ω ≪ J or indirectly via √SWAP operations when Ω ≫ J. The architecture can be operated in both regimes by embedding the device within a uniformly applied external magnetic field and using micromagnets to create local field gradients.

#### Charge Reservoirs and Initialisation

Charge reservoirs are integrated into the architecture to supply electrons to the quantum dots, facilitate spin-to-charge readout, and provide a relaxation path for rapid spin initialization. The mediator dots generate enough space for the integration of the reservoirs and the planar fan-out of metallic gate structures. The tunnel rate between the reservoir and the mediator dot can be tuned for rapid interaction during initial population or slow reset during periods of inactivity.

### Leakage Errors

#### Background

Leakage errors, where the state of the quantum system escapes out of the computational subspace, are not corrected by typical quantum error correction protocols. These errors can accumulate and eventually corrupt the logical qubits. In the context of SS qubits, leakage errors can arise from charge migration, which can propagate through the quantum dot array and corrupt the surface code.

#### Robustness Against Spin Leakage Errors

Spin leakage errors occur when the spin configuration of the ancilla qubits escapes out of the spin-zero subspace. These errors do not propagate to the data qubits and are removed in each new round of stabilizer checks when the ancilla is reinitialized. The effect of spin leakage errors on the double-dot ancillae is similar to the effect of computational errors in alternative schemes using single-dot ancillae.

#### Robustness Against Charge Leakage Errors

Charge leakage errors occur when the charge configuration of the qubit dots moves away from the ground charge configuration. These errors can be transferred from one qubit to another via gate operations and cannot be simply removed by reinitialization. The architecture uses mediator dots to transfer charge leakage errors from the qubit dots to the mediator dots, which can be reset using charge reservoirs. This approach ensures that charge leakage errors are effectively reduced to computational errors that can be corrected by the surface code.

### Surface Code Simulation

#### Surface Code Threshold and Stabilizer Check Circuit

The surface code is implemented by checking the X/Z parities of the data qubits spanned by each plaquette. The stabilizer-check circuit includes CZ gates, which are decomposed into √SWAP or S gates, and single-qubit Z rotations. The circuit is optimized to reduce the runtime and depth, and the stabilizer checks are partitioned to minimize spatial and temporal error correlations.

#### Stabilizer Cycle and Error Model

The stabilizer checks are divided into four disjoint partitions, performed in sequence. Within each partition, errors (including leakage errors) of one stabilizer check are independent of another, ensuring that the errors in each stabilizer check are Markovian. The error model includes two-qubit gate errors, readout errors, one-qubit gate errors, spin leakage errors, and charge leakage errors.

#### Surface Code Threshold Results

Simulations show that the surface code threshold for the computational errors arising from charge leakage errors is 0.66% in the absence of gate errors. Under a reasonable gate error rate of 0.5%, the charge leakage error threshold is 0.23-0.27%, demonstrating the robustness of the architecture against leakage errors even under gate noise.

#### Decoherence Errors

The finite decoherence time of spins in silicon can be tolerated by the surface code architecture. The stabilizer cycle time is designed to be fast enough to suppress environmental decoherence, and the probability of phase flip errors per stabilizer cycle is well within the per-gate error threshold.

### Conclusions and Outlook

The present invention introduces a surface code architecture for silicon spin qubits that is robust against spin and charge leakage errors. The architecture efficiently unifies the task of maintaining a proper charge distribution with the task of performing the stabilizer cycles required by the surface code. The use of mediator dots and double-dot ancillae, along with charge reservoirs, ensures that charge leakage errors are effectively reduced to computational errors that can be corrected by the surface code. Simulations demonstrate the robustness of the architecture against leakage errors and other computational errors, making it a promising approach for fault-tolerant quantum computing.