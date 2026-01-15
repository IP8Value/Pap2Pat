### High-Level Summary of Key Points:

- **High-Quality Quantum Control**: The system demonstrates high-fidelity quantum control over both 171Yb and 51V qubits, achieving a swap gate fidelity of \( F_{sw,1} \) and a coherence time for the 51V register that can be extended using dynamical decoupling techniques.
- **Coherence Analysis**: Detailed analysis shows that the T*2 dephasing timescale is primarily limited by the Overhauser field and the 171Yb Knight field. Motional narrowing techniques significantly improve coherence times, extending them from 33 µs to 417 µs.
- **State Lifetime**: The |Wv state exhibits a faster decay compared to the |0v state due to dephasing effects. Dynamical decoupling extends the lifetime of the |Wv state from 39.5 µs to 127 µs.
- **Bell-State Coherence**: A method is derived to extract Bell-state coherence and fidelity from parity oscillation data, accounting for readout infidelities. The analysis involves a maximum likelihood approach to estimate the corrected coherence and its uncertainty.

### Detailed Breakdown:

- **High-Quality Quantum Control**:
  The system demonstrates high-fidelity quantum control over both 171Yb and 51V qubits, achieving a swap gate fidelity of \( F_{sw,1} \). This is crucial for implementing multi-qubit operations. Additionally, coherence times for the 51V register can be significantly extended using dynamical decoupling techniques, which flip the state of the 171Yb to cancel out phase accumulation.

- **Coherence Analysis**:
  The T*2 dephasing timescale is primarily limited by two magnetic interactions: the Overhauser field and the 171Yb Knight field. Numerical simulations show that when limited by the 171Yb Knight field, the coherence time is 33 µs. When decoupled from this field, the coherence time extends to 417 µs. Motional narrowing techniques further improve these times.

- **State Lifetime**:
  The |0v state exhibits a slow exponential decay with a T1 lifetime of 0.54 s, potentially limited by resonant population exchange with unpolarized dark spins or off-resonant exchanges with detuned bath spins. In contrast, the |Wv state decays much faster due to dephasing effects, with a T1 time of 39.5 µs. Dynamical decoupling extends this lifetime to 127 µs.

- **Bell-State Coherence**:
  A method is derived to extract Bell-state coherence and fidelity from parity oscillation data, accounting for readout infidelities. The analysis involves a maximum likelihood approach to estimate the corrected coherence and its uncertainty. This ensures that the measured coherence accurately reflects the true state of the system.

### Additional Insights:

- **Error Analysis**:
  The error in estimating g_vz could be up to 25%, potentially due to small discrepancies in the position of the two closest 51V bath spins. Further analysis is needed to refine these parameters.
- **Future Work**:
  Improving state initialization fidelity into |±3/2> via direct ωa transition polarisation may offer advantages for working with the {|0v, |1v} manifold. Additionally, exploring correlated and uncorrelated fields in more detail could provide deeper insights into coherence limitations.

### Conclusion:

The system demonstrates robust quantum control and coherence management over 171Yb and 51V qubits, with significant improvements achieved through dynamical decoupling techniques. Detailed analysis of dephasing mechanisms and error sources provides a foundation for further optimization and potential applications in quantum computing and sensing.