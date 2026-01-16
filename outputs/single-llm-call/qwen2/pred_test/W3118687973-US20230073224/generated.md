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

Readout of the fluxonium levels