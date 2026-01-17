# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates to the field of quantum information processing and, more specifically, to a system and method for implementing a quantum memory using a hybrid spin system comprising a single 171 Yb ion coupled to a register of four 51 V nuclear spins. The invention provides a novel approach to storing and manipulating quantum information by leveraging the unique properties of the 171 Yb ion and the 51 V nuclear spins, enabling high-fidelity quantum state transfer and long-lived quantum memory.

### 2. Description of Related Art

Quantum information processing (QIP) holds the promise of revolutionizing computing and communication technologies by harnessing the principles of quantum mechanics. Central to QIP is the ability to store and manipulate quantum information reliably. Quantum memories, which are essential components of quantum networks and quantum computers, must be capable of storing quantum states for extended periods and performing operations on these states with high fidelity.

Several approaches have been proposed for implementing quantum memories, including the use of trapped ions, superconducting qubits, and solid-state defects. Each of these approaches has its own advantages and challenges. Trapped ions offer high coherence times and precise control but are limited by the complexity of scaling to large systems. Superconducting qubits provide fast operation speeds but suffer from short coherence times. Solid-state defects, such as nitrogen-vacancy centers in diamond, offer a balance between coherence and scalability but face challenges in achieving high-fidelity operations.

The use of rare-earth ions, such as 171 Yb, in combination with nuclear spins has emerged as a promising approach for quantum memory. 171 Yb ions have long-lived electronic states and can be precisely controlled using optical and microwave fields. However, the challenge lies in efficiently coupling the 171 Yb ion to a nuclear spin register that can serve as a quantum memory. Previous work has demonstrated the coupling of 171 Yb ions to single nuclear spins, but extending this to a multi-spin register has remained a significant challenge.

The present invention addresses these challenges by providing a system and method for implementing a quantum memory using a hybrid spin system. The invention leverages the unique properties of the 171 Yb ion and the 51 V nuclear spins to achieve high-fidelity state transfer and long-lived quantum memory.

## SUMMARY OF THE INVENTION

The present invention provides a system and method for implementing a quantum memory using a hybrid spin system comprising a single 171 Yb ion coupled to a register of four 51 V nuclear spins. The system includes a YVO4 crystal doped with 171 Yb ions, a nanophotonic cavity fabricated from the crystal, and a cryogenic environment for maintaining low temperatures. The method involves initializing the 171 Yb ion and the 51 V nuclear spins, performing state transfer operations using a ZenPol sequence, and implementing quantum memory protocols.

Key features of the invention include:
1. **High-Fidelity Initialization**: The 171 Yb ion and the 51 V nuclear spins are initialized into specific states with high fidelity using a two-stage protocol.
2. **State Transfer**: A ZenPol sequence is used to transfer quantum states between the 171 Yb ion and the 51 V nuclear spins with high fidelity.
3. **Quantum Memory**: The 51 V nuclear spins serve as a quantum memory, allowing for the storage of quantum states for extended periods.
4. **Bell State Generation**: The system can generate Bell states, which are essential for quantum communication and quantum computing.
5. **Supplementary Methods**: Additional methods and derivations are provided to enhance the performance and reliability of the system.

The invention offers significant advantages over existing quantum memory systems, including high-fidelity state transfer, long-lived quantum memory, and the ability to perform complex quantum operations.

## DETAILED DESCRIPTION OF THE INVENTION

### 1. First Example: System Implemented in Yb:YVO

The system of the present invention is implemented in a YVO4 crystal doped with 171 Yb ions. The crystal is cut and polished to ensure high quality and minimal impurities. A nanophotonic cavity is fabricated from the crystal using focused ion beam milling, resulting in a cavity with a high Q-factor of approximately 10,000. This high Q-factor leads to Purcell enhancement, reducing the excited state lifetime of the 171 Yb ion from 267 µs to 2.3 µs. The cavity is undercoupled, with κin/κ ≈ 0.14, ensuring that 14% of emitted light enters the waveguide mode. Waveguide-free space coupling is achieved via angled couplers with an efficiency of approximately 25%, resulting in an end-to-end system efficiency of approximately 1%.

The device is placed on the still-plate of a 3 He cryostat with a base temperature of 460 mK. Optical signals are fed into the cryostat through optical fibers and focused onto the device using an aspheric lens doublet mounted on a stack of x-y-z piezo nanopositioners. The device is tuned on-resonance with the 171 Yb optical transitions via nitrogen condensation. Residual magnetic fields are canceled along the crystal c ≡ z axis using home-built superconducting magnet coils.

### 2. Example Protocol for the First Example

The protocol for initializing the 171 Yb ion and the 51 V nuclear spins involves a two-stage process. Initially, the |aux g state of the 171 Yb ion is emptied using a series of 3 µs pulses applied to the optical F transition, each followed by a 3 µs wait period. When the 171 Yb ion is successfully excited from |aux g to |1 e, the population in |1 e will preferentially decay to |0 g during the wait time. Subsequently, the |1 g state is emptied by applying an optical π pulse to the A transition followed by a microwave π pulse to the fe transition, leading to excitation from |1 g to |1 e and decay into |0 g. This process is repeated several times to improve fidelity.

Readout of the 171 Yb |1 g state is performed by applying a series of 100 π pulses to the A transition, each followed by a 10 µs photon detection window. This process is enabled by the cyclic nature of the A transition. To read out the |0 g population, an additional π pulse is applied to swap the |0 g ↔ |1 g populations before performing the same optical readout procedure.

### 3. Example Implementation of the First Example as Quantum Memory

The 51 V nuclear spins serve as a quantum memory, allowing for the storage of quantum states for extended periods. The system is initialized by preparing the 171 Yb ion in |0 g and the 51 V register in |0 v = |±7/2 ⊗4. A series of ZenPol polarization operations are interleaved with 171 Yb re-initialization sequences and alternate between ωb and ωc transition control to sequentially polarize the 51 V register towards the |±7/2 level. After the initialization sequence, a single π/2 pulse is applied to the 171 Yb qubit to prepare a superposition state. Subsequently, the state is transferred to the 51 V register using a swap operation resonant with the ωc transition. After a variable wait time, the superposition state is retrieved with a second swap gate and measured in the x-basis via a π/2 pulse followed by optical readout on the A transition.

### 4. Example Bell State Generation Using the First Example

The system can generate Bell states, which are essential for quantum communication and quantum computing. The process involves preparing the 171 Yb ion in a superposition state and transferring this state to the 51 V register using a swap operation. The 51 V register, now in a superposition state, is entangled with the 171 Yb ion. The entangled state can be characterized using a series of measurements, including parity oscillations and population basis measurements.

### 5. Supplementary Example Methods for Implementing the First Example

Additional methods are provided to enhance the performance and reliability of the system. These methods include:
- **Dynamical Decoupling**: Periodic π pulses are applied to the 171 Yb qubit to decouple it from the nuclear Overhauser field, extending the coherence time of the 51 V register.
- **Direct Drive Gates**: Sinusoidal z-directed RF magnetic fields are applied to the 51 V register to induce Rabi oscillations, enabling local control of the register spins.
- **Population Basis Measurements**: A sequential tomography protocol is developed to read out the populations of the joint 171 Yb-51 V density matrix in the effective four-state basis.

### 6. Supplementary Example Derivations for Interactions and Hamiltonians Described Herein

The interactions and Hamiltonians described in the invention are derived from the effective Hamiltonian of the system. The effective Hamiltonian for the 171 Yb qubit and the 51 V register is given by:

\[
\hat{H}_{\text{eff}} = \frac{\omega_{01}}{2} \hat{S}_z + \frac{Q}{2\pi} \hat{I}_z^2 + \sum_{i=1}^{4} \left( a_x \hat{S}_z \hat{I}_x^{(i)} + a_z \hat{S}_z \hat{I}_z^{(i)} \right)
\]

where:
- \(\omega_{01}\) is the 171 Yb qubit transition frequency,
- \(Q\) is the 51 V nuclear quadrupole splitting,
- \(\hat{S}_z\) is the 171 Yb qubit operator along the z-axis,
- \(\hat{I}_x^{(i)}\) and \(\hat{I}_z^{(i)}\) are the 51 V spin-7/2 operators along the x- and z-axes for the i-th register spin,
- \(a_x\) and \(a_z\) are the effective coupling strengths between 171 Yb and 51 V along the x- and z-axes.

The ZenPol sequence is designed to generate an effective spin-exchange interaction between the 171 Yb qubit and the 51 V register. The average Hamiltonian in the rotating frame is given by:

\[
\hat{H}_{\text{avg}} = \frac{B_{\text{RF}}}{2} \left( \sqrt{7} a_x \hat{S}_x \hat{I}_x + \sqrt{7} a_y \hat{S}_y \hat{I}_y \right)
\]

where \(B_{\text{RF}}\) is the amplitude of the square-wave RF magnetic field.

## Definitions

- **171 Yb Ion**: A ytterbium ion with an atomic mass of 171, used as a qubit in the quantum memory system.
- **51 V Nuclear Spins**: Vanadium-51 nuclear spins, used as a register in the quantum memory system.
- **Nanophotonic Cavity**: A photonic structure fabricated from the YVO4 crystal to enhance the interaction between the 171 Yb ion and the 51 V nuclear spins.
- **ZenPol Sequence**: A pulse sequence designed to generate an effective spin-exchange interaction between the 171 Yb qubit and the 51 V register.
- **Swap Operation**: A quantum operation that transfers the state of one qubit to another.
- **Bell State**: An entangled state of two qubits, used in quantum communication and quantum computing.
- **Dynamical Decoupling**: A technique used to extend the coherence time of qubits by applying periodic pulses to decouple them from environmental noise.
- **Direct Drive Gates**: Control pulses applied to the 51 V register to induce Rabi oscillations and enable local control of the register spins.
- **Population Basis Measurements**: A protocol for measuring the populations of the joint 171 Yb-51 V density matrix in the effective four-state basis.

## CONCLUSION

The present invention provides a novel system and method for implementing a quantum memory using a hybrid spin system comprising a single 171 Yb ion coupled to a register of four 51 V nuclear spins. The system offers high-fidelity state transfer, long-lived quantum memory, and the ability to perform complex quantum operations, making it a significant advancement in the field of quantum information processing. The invention addresses the challenges of existing quantum memory systems and opens new avenues for the development of practical quantum technologies.