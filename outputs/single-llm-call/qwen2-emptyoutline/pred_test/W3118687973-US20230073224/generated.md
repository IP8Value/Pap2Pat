# DESCRIPTION

## BACKGROUND

Superconducting circuits have emerged as leading contenders for the development of quantum computers, owing to significant advancements in coherence times, gate fidelities, and processor sizes over the past two decades. These advancements have ushered in the noisy intermediate-scale quantum (NISQ) era and have enabled demonstrations of quantum advantage over classical computing. Despite these achievements, decoherence remains a central challenge for superconducting circuits. While the effects of environmental noise can be mitigated by improving material properties, this invention introduces an alternative approach to drastically reduce the noise spectral density by lowering the qubit frequency from gigahertz to megahertz. This reduction in frequency slows down all relevant timescales but still allows for the performance of single-qubit gates at speeds comparable to the state of the art in conventional qubits.

The invention leverages a heavy-fluxonium circuit, which features a low qubit frequency (approximately 14 MHz) and excited levels that are several gigahertz away. The large frequency separation enables the application of flux drives with amplitudes much larger than the qubit frequency, without any leakage outside of the computational subspace. Additionally, the gap to the excited levels exceeds the frequency scale set by the ambient temperature, allowing these levels to be used for initialization and readout of the qubit state. The invention develops a completely new set of protocols for operating the heavy fluxonium, demonstrating state-of-the-art performance that makes it a serious competitor to the transmon qubit, with the potential for further improvements.

## SUMMARY

The present invention relates to a method and apparatus for performing fast, high-fidelity single-qubit gates in a heavy-fluxonium circuit with a low qubit frequency. The heavy-fluxonium circuit comprises a small-area Josephson junction (JJ) shunted by a large inductance and a large capacitor. The qubit is initialized in a pure state using a reset protocol that utilizes the readout resonator and higher circuit levels to cool the qubit down to an effective temperature of approximately 190 μK. Despite the low qubit frequency, the invention achieves ultrafast-flux gate protocols that perform single-qubit gates at speeds similar to those of typical transmons.

The invention includes the following key features:
1. **Heavy-Fluxonium Circuit**: The circuit consists of a small-area Josephson junction shunted by a large inductance and a large capacitor, resulting in a Hamiltonian that supports a low qubit frequency.
2. **Qubit Initialization and Readout**: A reset protocol is developed to initialize the qubit in a pure state and a plasmon-assisted readout scheme is used to measure the qubit state.
3. **Fast Single-Cycle Flux Gates**: Nonadiabatic Landau-Zener transitions are utilized to realize ultrafast gates that occur within a single Larmor period, achieving high-fidelity single-qubit operations.
4. **Characterization of Device Coherence**: The coherence properties of the qubit are characterized, demonstrating long coherence times and high-fidelity gate operations.

## DETAILED DESCRIPTION

### Heavy-Fluxonium Circuit

The heavy-fluxonium circuit is designed to operate at a low qubit frequency, specifically around 14 MHz. The circuit comprises a small-area Josephson junction (JJ) with inductance \( L_J \) shunted by a large inductance \( L_{JA} \) and a large capacitor \( C_q \). The shunting inductance is realized by an array of 300 large-area JJs, each having a Josephson energy \( E_{JA} \) and charging energy \( E_{CA} \). The condition \( E_{JA}/E_{CA} \gg 1 \) ensures that the charge dispersion for each array junction is small, allowing the array to be treated as a linear inductor. The resulting Hamiltonian of the circuit is given by:

\[
H = \frac{4E_c n^2}{2} + E_L \left( \phi - \phi_{ext} \right)^2
\]

where \( E_c \) is the charging energy, \( E_L \) is the inductive energy, \( n \) is the charge operator, and \( \phi \) is the phase operator. The qubit is comprised of the lowest two energy levels \( |g\rangle \) and \( |e\rangle \), with the qubit transition being fluxon-like, characterized by a frequency of 14 MHz.

### Qubit Initialization and Readout

Due to the low transition frequency, the qubit starts in a nearly evenly mixed state in thermal equilibrium. To initialize the qubit in a pure state, a reset protocol is employed. This protocol involves simultaneously driving the \( |g0\rangle \rightarrow |h0\rangle \) and \( |h0\rangle \rightarrow |e1\rangle \) transitions for 15 μs. The high resonator frequency (5.7 GHz) and low resonator quality factor \( Q = 600 \) result in the rapid loss of a photon from \( |e1\rangle \), effectively removing entropy from the qubit. This process steers the system into a steady state with approximately 97% of the population settling in \( |e0\rangle \). An additional π pulse on the \( |g\rangle - |e\rangle \) transition initializes the system in the ground state \( |g0\rangle \). The reset is characterized by performing a Rabi rotation between the \( |e\rangle \leftrightarrow |f\rangle \) levels, demonstrating a 3 ± 2% error in state preparation.

Readout of the fluxonium levels is performed using circuit QED by capacitively coupling the fluxonium circuit to a readout resonator. The dispersive shift \( \chi \) of the resonator due to changes in the occupation of computational states is small (60 kHz). To overcome this, a π pulse on the \( |e\rangle - |f\rangle \) transition is performed before standard dispersive readout, transferring the population in \( |e\rangle \) to \( |f\rangle \). This plasmon-assisted readout scheme results in a 50% single-shot readout fidelity, which can be further improved with a parametric amplifier and by optimizing the resonator \( \kappa \) and dispersive shifts.

### Characterizing Device Coherence

The coherence properties of the qubit are characterized by measuring the relaxation time \( T_1 \) and dephasing time \( T_2 \). The experimentally measured \( T_1 \) as a function of the applied external flux shows a maximum value of 4.3 ± 0.2 ms at a specific flux point, consistent with previous heavy-fluxonium devices. The qubit relaxation times are measured over a wide range of external flux by driving the \( |g\rangle - |h\rangle \) transition for 120 μs and monitoring the subsequent decay. The dephasing is characterized using a Ramsey sequence with three echo π pulses, minimized at \( \Phi_{ext} = \Phi_0/2 \), where the qubit frequency is first-order insensitive to changes in flux. The dephasing rate near the flux-frustration point is separated into a frequency-independent term \( \Gamma_C \) and a term proportional to 1/f flux noise. The T2e values around the flux-frustration point are much higher than those for state-of-the-art transmons, demonstrating the improved coherence of the heavy-fluxonium qubit.

### Fast Single-Cycle Flux Gates

To maximize the advantage of the large anharmonicity of the heavy fluxonium, the invention rethinks the standard microwave-drive control of the circuit. Instead, the qubit is controlled through fast-flux pulses, utilizing nonadiabatic Landau-Zener transitions to realize ultrafast gates that occur within a single Larmor period. Near the flux-frustration point, the Hamiltonian within the computational space can be idealized as a spin-1/2 system:

\[
H = \frac{\hbar}{2} \left( A(\Phi_{ext}) \sigma_x + \Delta \sigma_z \right)
\]

where \( \Delta \approx 14 \) MHz is the splitting of \( |g\rangle \) and \( |e\rangle \) at the flux-frustration point, and \( A \) is the amplitude of the \( \sigma_x \) term, proportional to the flux offset \( \delta\Phi_{ext} \). The coefficient of the \( \sigma_x \) term can be much larger than the qubit frequency, with \( A \sim 300 \) MHz when \( \delta\Phi_{ext} = 0.06\Phi_0 \).

The protocol for a generic qubit pulse involves rapidly moving the flux-bias point away from the flux-frustration point in one direction and back, generating a rotation about the x-axis through a large \( \sigma_x \) term. The pulse is immune to shape distortions, and the total \( \sigma_x \) and \( \sigma_z \) amplitudes depend only on the area of the spike and the idling length \( \Delta t_z \). By sweeping the amplitude \( A \) of the triangular spike and the idling length \( \Delta t_z \), 2D Rabi patterns are obtained, providing a measure of the gate parameters. The fidelities of the single-qubit gates are characterized through randomized benchmarking (RB) and interleaved RB (IRB), demonstrating high-fidelity single-qubit operations.

## CONCLUSION

In conclusion, the invention realizes a heavy-fluxonium qubit with a 14 MHz transition frequency and coherence times exceeding those of state-of-the-art transmons. Protocols for plasmon-assisted reset and readout of the qubit and a new flux control scheme that performs fast high-fidelity gates are demonstrated. The invention explores a new frequency regime in superconducting qubits, providing a path for manipulating fluxonium qubits with computational frequencies in the range of several gigahertz at temperatures much higher than current dilution-refrigerator temperatures. The gate pulses can be directly synthesized with inexpensive digital-to-analog converters and are insensitive to shape distortions, making the heavy-fluxonium circuit a viable candidate for large-scale superconducting quantum computation.

## COMBINATION OF FEATURES

The combination of features in the present invention, including the heavy-fluxonium circuit design, the reset and readout protocols, and the fast single-cycle flux gates, collectively address the challenges of decoherence and low-frequency operation in superconducting qubits. The low qubit frequency and large anharmonicity enable the application of large-amplitude flux drives without leakage, while the nonadiabatic Landau-Zener transitions allow for ultrafast gate operations. The high-fidelity single-qubit gates and long coherence times demonstrate the potential of the heavy-fluxonium qubit for scalable quantum computing.