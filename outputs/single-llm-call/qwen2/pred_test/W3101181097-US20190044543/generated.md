# DESCRIPTION

## SUMMARY

The present invention relates to a method and system for fault-tolerant error correction (FTEC) in quantum computing systems. Specifically, the invention provides a generalized flag error correction protocol applicable to stabilizer codes of arbitrary distance. The protocol utilizes ancillary qubits, referred to as flag qubits, to detect high-weight errors that may arise during the measurement of stabilizers. The invention ensures that the error correction process remains fault-tolerant even in the presence of a limited number of faults, thereby extending the lifetime of encoded quantum information.

## DETAILED DESCRIPTION

### I. General Considerations

Quantum error correction (QEC) is essential for reliable quantum computing. Traditional fault-tolerant error correction (FTEC) schemes, such as those proposed by Shor, Steane, and Knill, have limitations in terms of qubit overhead and circuit depth. The present invention introduces a novel flag error correction protocol that addresses these limitations. The protocol is designed to be applicable to a wide range of stabilizer codes, including but not limited to surface codes, color codes, and quantum Reed-Muller codes. The key innovation lies in the use of flag qubits to detect and correct high-weight errors, ensuring fault tolerance even when a limited number of faults occur.

### II. Introduction and Formalism

Scalable quantum computers require robust error correction mechanisms to function reliably. Active error correction protocols, which involve measuring the check operators of an error-correcting code, are necessary for large-scale quantum computations. The present invention builds on existing FTEC schemes and introduces a generalized flag error correction protocol that can be applied to stabilizer codes of arbitrary distance.

#### Definitions

1. **Weight-t Pauli Operators**: A Pauli operator \( E \) is a tensor product of Pauli matrices \( X, Y, Z \), and the identity \( I \). The weight of \( E \), denoted \( \text{wt}(E) \), is the number of qubits on which \( E \) has non-trivial support.

2. **Stabilizer Error Correction**: Given a stabilizer group \( S = \langle g_1, \ldots, g_m \rangle \), the syndrome \( s(E) \) of an error \( E \) is a bit string where the \( i \)-th bit is 0 if \( g_i \) and \( E \) commute, and 1 otherwise. The minimal weight correction \( E_{\text{min}}(s) \) is a Pauli operator \( E \) such that \( s(E) = s \) and \( E \) has minimal weight. Two operators \( E \) and \( E' \) are logically equivalent, denoted \( E \sim E' \), if \( E \propto gE' \) for some \( g \in S \).

3. **Fault-Tolerant Error Correction**: An error correction protocol using a distance-\( d \) stabilizer code \( C \) is \( t \)-fault-tolerant if:
   - For an input codeword with an error of weight \( s_1 \), if \( s_2 \) faults occur during the protocol with \( s_1 + s_2 \leq t \), ideally decoding the output state gives the same codeword as ideally decoding the input state.
   - For \( s \) faults during the protocol with \( s \leq t \), no matter how many errors are present in the input state, the output state differs from a codeword by an error of at most weight \( s \).

### III. Flag Error Correction for Small Distance Codes

#### Definitions and Flag 1-FTEC with Distance-3 Codes

1. **Bad Locations**: A circuit location in which a single fault can result in a Pauli error \( E \) on the data block with \( \text{wt}(E) \geq 2 \) is referred to as a bad location.

2. **Flags and Measurements**: In a circuit for measuring a stabilizer generator that includes at least one flag ancilla, the ancilla used to infer the stabilizer outcome is referred to as the measurement qubit. The circuit has flagged if the eigenvalue of a flag qubit is measured as -1. If the eigenvalue of a measurement qubit is measured as -1, the measurement qubit flipped.

3. **t-Flag Circuit**: A circuit \( C(P) \) which, when fault-free, implements a projective measurement of a weight-\( w \) Pauli \( P \) without flagging is a \( t \)-flag circuit if for any set of \( v \) faults at up to \( t \) locations in \( C(P) \) resulting in an error \( E \) with \( \min(\text{wt}(E), \text{wt}(EP)) > v \), the circuit flags.

4. **Flag Error Set**: Let \( E(g_i) \) be the set of all errors caused by one fault which caused the circuit \( C(g_i) \) to flag.

#### Flag 1-FTEC Condition

A stabilizer code \( S = \langle g_1, g_2, \ldots, g_r \rangle \) and 1-flag circuits \( \{C(g_1), C(g_2), \ldots, C(g_r)\} \) satisfy the Flag 1-FTEC condition if for every generator \( g_i \), all pairs of elements \( E, E' \in E(g_i) \) satisfy:
\[ E \sim E' \quad \text{or} \quad s(E) \neq s(E') \]

#### Flag 1-FTEC Protocol

1. **Syndrome Measurement**: Repeat the syndrome measurement using flag circuits until one of the following is satisfied:
   - If the syndrome \( s \) is repeated twice in a row and there were no flags, apply the correction \( E_{\text{min}}(s) \).
   - If there were no flags and the syndromes \( s_1 \) and \( s_2 \) from two consecutive rounds differ, repeat the syndrome measurement using non-flag circuits yielding syndrome \( s \). Apply the correction \( E_{\text{min}}(s) \).
   - If a circuit \( C(g_i) \) flags, stop and repeat the syndrome measurement using non-flag circuits yielding syndrome \( s \). If there is an element \( E \in E(g_i) \) which satisfies \( s(E) = s \), then apply \( E \), otherwise apply \( E_{\text{min}}(s) \).

### IV. Flag Error Correction Protocol for Arbitrary Distance Codes

#### Conditions and Protocol

1. **Flag t-FTEC Condition**: Consider a stabilizer code \( S = \langle g_1, g_2, \ldots, g_r \rangle \) and \( t \)-flag circuits \( \{C(g_1), C(g_2), \ldots, C(g_r)\} \). The code satisfies the Flag t-FTEC condition if for any set of \( v \) faults at up to \( t \) locations in \( C(g_i) \) resulting in an error \( E \) with \( \min(\text{wt}(E), \text{wt}(EP)) > v \), the circuit flags.

2. **Update Rules**: Given a sequence of consecutive syndrome measurement outcomes \( s_k \) and \( s_{k+1} \):
   - If \( n_{\text{diff}} \) didn't increase in the previous round, and \( s_k = s_{k+1} \), increase \( n_{\text{diff}} \) by one.
   - If a flag occurs, reset \( n_{\text{same}} \) to zero.
   - If \( s_k = s_{k+1} \), increase \( n_{\text{same}} \) by one.

3. **Corrections**:
   - The same syndrome \( s \) is repeated \( t - n_{\text{diff}} + 1 \) times in a row and there are no flags, apply the correction \( E_{\text{min}}(s) \).
   - There were no flags and \( n_{\text{diff}} = t \). Repeat the syndrome measurement using non-flag circuits yielding the syndrome \( s \). Apply the correction \( E_{\text{min}}(s) \).
   - Some set of \( t \) circuits \( \{C(g_{i1}), \ldots, C(g_{it})\} \) have flagged. Repeat the syndrome measurement using non-flag circuits yielding the syndrome \( s \). Apply any correction from the set \( \tilde{E}_t^t(g_{i1}, \ldots, g_{it}, s) \).
   - Any circuit \( C(g_i) \) has flagged and \( n_{\text{diff}} = 1 \). Repeat the syndrome measurement using non-flag circuits yielding the syndrome \( s \). Apply any correction from the set \( \tilde{E}_1^t(g_i, s) \).
   - Any circuit \( C(g_i) \) has flagged and \( n_{\text{diff}} = 0 \) and \( n_{\text{same}} = 1 \). Use the measured syndrome \( s \) from the last round. Apply any correction from the set \( \tilde{E}_1^t(g_i, s) \cup \tilde{E}_2^t(g_i, s) \).

### V. Circuit Level Noise Analysis

The performance of the flag t-FTEC protocol is evaluated under a circuit-level noise model. The noise model assumes that each two-qubit gate is followed by a two-qubit Pauli error with probability \( p \), state preparations and measurements are subject to Pauli errors with probability \( \frac{2p}{3} \), and idle qubits are subject to Pauli errors with probability \( p \). The logical failure rate is estimated using Monte Carlo simulations, and the pseudo-threshold is defined as the value of \( p \) such that the logical failure rate is equal to the physical failure rate of an unencoded idle qubit.

### VI. Review

The flag t-FTEC protocol provides a robust method for fault-tolerant error correction in quantum computing systems. By utilizing flag qubits to detect high-weight errors, the protocol ensures that the error correction process remains fault-tolerant even in the presence of a limited number of faults. The protocol is applicable to a wide range of stabilizer codes, making it a versatile tool for extending the lifetime of encoded quantum information.

### VII. Proof that the Flag t-FTEC Protocol Satisfies the Fault-Tolerant Criteria of Definition (2)

To prove that the flag t-FTEC protocol satisfies the fault-tolerant criteria of Definition (2), we assume that there are at most \( t \) faults during the protocol. The protocol is designed to handle different scenarios based on the number of flags and the repetition of syndromes. The key steps in the proof are:

1. **Syndrome Repetition**: If the same syndrome is repeated \( t - n_{\text{diff}} + 1 \) times in a row and there are no flags, the protocol applies the minimal weight correction \( E_{\text{min}}(s) \). This ensures that the output state can differ from the input codeword by an error of at most weight \( t - n_{\text{diff}} \).

2. **No Flags and Maximum Differences**: If there are no flags and \( n_{\text{diff}} = t \), the protocol repeats the syndrome measurement using non-flag circuits and applies the minimal weight correction \( E_{\text{min}}(s) \). This ensures that the output state is a valid codeword.

3. **Multiple Flags**: If multiple circuits flag, the protocol applies a correction from the set \( \tilde{E}_t^t(g_{i1}, \ldots, g_{it}, s) \). This ensures that the output state is a valid codeword.

4. **Single Flag and Differences**: If a single circuit flags and \( n_{\text{diff}} = 1 \), the protocol applies a correction from the set \( \tilde{E}_1^t(g_i, s) \). This ensures that the output state is a valid codeword.

5. **Single Flag and No Differences**: If a single circuit flags and \( n_{\text{diff}} = 0 \) and \( n_{\text{same}} = 1 \), the protocol applies a correction from the set \( \tilde{E}_1^t(g_i, s) \cup \tilde{E}_2^t(g_i, s) \). This ensures that the output state is a valid codeword.

### VIII. Fault-Tolerant State Preparation and Measurement using Flag t-FTEC

1. **State Preparation**: To prepare a logical \( |0 \rangle \) state, perform a round of flag t-FTEC using the extended stabilizers \( g_1, \ldots, g_{n-1}, Z \). The output state is guaranteed to be the encoded \( |0 \rangle \) state with at most \( t \) single-qubit errors.

2. **Measurement**: To measure the eigenvalue of a logical operator \( P \), perform flag t-FTEC, measure the eigenvalue of \( P \) using a \( t \)-flag circuit, and repeat the process \( 2t + 1 \) times. Take the majority of the measured eigenvalues to obtain the correct result.

### IX. Quantum Reed-Muller Codes

The family of quantum Reed-Muller codes \( \text{QRM}(m) \) with code parameters \( [[2^m - 1, k = 1, d = 3]] \) satisfies the sufficient flag 1-FTEC condition. The X stabilizer generators are derived from shortened Reed-Muller codes, and the Z stabilizer generators are derived from the dual of higher-order Reed-Muller codes. The flag t-FTEC protocol can be applied to these codes to ensure fault-tolerant error correction.

---

This detailed description outlines the invention, providing a comprehensive framework for implementing the flag error correction protocol in quantum computing systems. The protocol is designed to be robust, efficient, and applicable to a wide range of stabilizer codes, making it a valuable tool for advancing the field of quantum error correction.