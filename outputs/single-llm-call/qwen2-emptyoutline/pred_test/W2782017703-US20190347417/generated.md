# DESCRIPTION

## BACKGROUND

In today's System-on-Chip (SoC) design landscape, the integration of Intellectual Property (IP) cores from third-party (3P) vendors is a common practice. This approach helps to reduce research and development (R&D) costs and accelerate the development cycle. However, it introduces a significant concern regarding the trustworthiness of these 3P IP cores (3PIPs). Third-party vendors can intentionally insert malicious components, known as hardware Trojans, into their IP cores. These Trojans can create backdoors in the design, leading to the leakage of sensitive information and enabling various attacks, such as denial of service and reduction in reliability.

Detecting Trojans in 3PIPs is a challenging task due to the lack of a golden reference model against which to compare the IP core during verification. Traditional methods, such as structural and functional analysis, logic testing, formal verification, information flow tracking, and runtime validation, have been proposed to address this issue. However, these techniques often fall short in ensuring the complete detection of Trojans. For instance, structural analysis techniques like unused circuit identification (UCI) and functional analysis techniques like FANCI and VeriTrust can be bypassed by sophisticated Trojans. Formal verification methods, while powerful, are limited by the number of clock cycles they can verify and can produce false positive results. Information flow tracking (IFT) techniques, though effective, can be evaded by Trojans that leverage valid paths or operate in scan mode.

This patent application introduces a novel framework for detecting Trojans in 3PIPs by leveraging information flow security (IFS) verification. The framework addresses the limitations of existing techniques by modeling assets (e.g., nets carrying secret keys) as faults and using automatic test pattern generation (ATPG) algorithms to detect the propagation of these assets. This approach ensures the detection of IFS violations caused by Trojans, even in complex and large-scale designs.

## DETAILED DESCRIPTION

### Overview of the IFS Verification Framework

The proposed IFS verification framework is designed to detect Trojans in 3PIPs by identifying violations of IFS policies. The framework operates on synthesized gate-level netlists and can be applied to both 3PIP and design-for-test (DfT) inserted netlists. The core idea is to model an asset (e.g., a net carrying a secret key) as a stuck-at-0 and stuck-at-1 fault and use ATPG algorithms to detect the propagation of these faults. The framework consists of two main components: Confidentiality Verification and Integrity Verification.

### Confidentiality Verification

Confidentiality policy ensures that sensitive information from a classified system does not leak to an unclassified domain. In the context of SoC design, this means that a secret key for data encryption should never flow to an unclassified output. A violation of the confidentiality policy indicates that an asset can be observed through an unauthorized observe point.

#### Algorithm for Confidentiality Verification

1. **Input**: The algorithm takes an asset (name of the port or net where the asset is located), the gate-level netlist of the design, and the technology library as inputs.
2. **Scan Capability**: Add scan capability to all flip-flops (FFs) in the design to make them controllable and observable.
3. **Identify Observe Points**: For each asset, find the observe points (primary outputs or scan FFs) in the fan-out cone of the asset.
4. **Capture Masks**: Add capture masks in these FFs to individually track asset propagation to each observe point.
5. **Stuck-at Fault**: Add the asset as the only stuck-at fault in the design.
6. **ATPG Analysis**: Use ATPG algorithms in sequential mode to find paths to propagate the asset to each observe point.
7. **Report Violations**: If both stuck-at-0 and stuck-at-1 faults are detected from an observe point, mark it as an observe point and report the propagation path and the control sequence required for asset propagation.
8. **Next Level Observe Points**: Find the next level of observe points by analyzing the fan-out cone of the current observe points.
9. **Repeat**: Continue the process until all observe points are primary outputs or the ATPG algorithm cannot generate patterns to propagate the fault to observe points.

### Integrity Verification

Integrity policy ensures that an untrusted system should never influence a trusted one. In SoC design, this means that an untrusted control point should never be able to influence a control pin of a trusted system. A violation of the integrity policy indicates that an asset can be influenced by unauthorized control points.

#### Algorithm for Integrity Verification

1. **Input**: The algorithm takes an asset (name of the register where the asset is located), the gate-level netlist of the design, and the technology library as inputs.
2. **Scan Capability**: Add scan capability to all FFs in the design to make them controllable and observable.
3. **Identify Control Points**: For each asset, find the control points (fan-in elements) in the fan-in cone of the asset.
4. **Stuck-at Fault**: Add the asset as the only stuck-at fault in the design.
5. **ATPG Analysis**: Use ATPG algorithms in sequential mode to activate the fault.
6. **Report Violations**: If the fault is detected, mark the control points and report the activation path and the control sequence required to activate the fault.
7. **Previous Level Control Points**: Find the previous level of control points by analyzing the fan-in cone of the current control points.
8. **Repeat**: Continue the process until all control points are primary inputs or the ATPG algorithm cannot generate patterns to activate the fault from control points.

### Trigger Condition Extraction

Once the IFS verification framework identifies IFS violations, the next step is to extract the input sequence that triggers the Trojan. This is particularly important for Trojans with complex trigger circuits, such as those involving finite state machines (FSMs).

#### Algorithm for Trigger Condition Extraction

1. **Identify State Registers**: Determine if the registers associated with the IFS violation are part of an FSM. This is done by checking if the output of a register feeds back to its input through a series of combinational circuits.
2. **FSM Extraction**: Use FSM extraction techniques to retrieve the functionality of the FSM. This involves determining the present states and input conditions that cause transitions to a particular state.
3. **Generate State Transition Graph (STG)**: Generate the STG of the overall FSM and extract the sequence of input patterns that trigger the Trojan.

### Experimental Validation

The proposed IFS verification framework was experimentally validated using benchmark circuits from the trust-hub. The framework successfully detected various types of Trojans, including those that leak sensitive information through valid output ports and those that operate in scan mode. The framework was also able to detect Trojans with complex trigger circuits, such as those involving FSMs and counters.

### Comparison with State-of-the-Art Techniques

The proposed IFS verification framework addresses the limitations of existing Trojan detection techniques. Unlike formal methods, which are limited by the number of clock cycles they can verify, the IFS framework can detect Trojans regardless of the number of cycles. Unlike IFT techniques, which can produce false positive results and cannot distinguish between valid and malicious paths, the IFS framework accurately tracks asset propagation paths. Additionally, the IFS framework can detect Trojans in DfT inserted netlists, which is a significant advantage over other techniques.

### Limitations of the Proposed Approach

While the IFS verification framework is robust, it has some limitations. The ATPG algorithm may fail to detect observe/control points if it takes too long to generate patterns for asset propagation/activation. Additionally, the framework cannot work with latches and uncontrollable FFs, although these components are rare in properly implemented designs. The framework issues warnings and reports the names and locations of such components, allowing for further analysis.

### Conclusion

The proposed IFS verification framework provides a comprehensive solution for detecting Trojans in 3PIPs. By modeling assets as faults and using ATPG algorithms, the framework ensures the detection of IFS violations caused by Trojans, even in complex and large-scale designs. The framework can be applied to both 3PIP and DfT inserted netlists, making it a versatile tool for SoC designers. While the framework has some limitations, it offers significant advantages over existing Trojan detection techniques and can be used to enhance the security of SoC designs.