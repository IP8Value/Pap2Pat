# DESCRIPTION  

## BACKGROUND  

The increasing reliance on third-party intellectual property (IP) cores in system-on-chip (SoC) designs has introduced significant security concerns, particularly with respect to the potential insertion of hardware Trojans by untrusted vendors. Hardware Trojans are malicious modifications embedded within IP cores that can compromise system integrity, leak sensitive information, or disrupt functionality under rare or specific conditions. Traditional verification techniques, such as structural analysis, functional testing, and formal verification, have been employed to detect such Trojans, but these methods suffer from inherent limitations, including false positives, limited verification depth, and an inability to distinguish between legitimate and malicious signal paths.  

Existing Trojan detection approaches, including unused circuit identification (UCI), functional analysis for nearly-unused circuit identification (FANCI), and information flow tracking (IFT), have demonstrated vulnerabilities that can be exploited by sophisticated adversaries. For instance, Trojans can be designed to evade detection by mimicking legitimate circuit behavior or by activating only under conditions that exceed the verification bounds of formal methods. Additionally, commercial tools such as Jasper Security Path Verification (SPV) and proof-carrying code (PCC) methodologies are constrained by their reliance on predefined security policies or golden reference models, which may not be available for third-party IPs.  

The limitations of current detection techniques highlight the need for a robust framework capable of identifying hardware Trojans without requiring a golden reference or exhaustive white-box knowledge of the IP. A novel approach leveraging information flow security (IFS) verification offers a promising solution by modeling assets as faults and utilizing automatic test pattern generation (ATPG) algorithms to detect unauthorized information propagation or control. This method overcomes the shortcomings of prior techniques by providing comprehensive coverage of both confidentiality and integrity violations, regardless of the Trojan’s activation conditions or structural complexity.  

## DETAILED DESCRIPTION  

The proposed framework for hardware Trojan detection is based on information flow security (IFS) verification, which systematically identifies violations of confidentiality and integrity policies introduced by malicious circuits. The framework operates on synthesized gate-level netlists and employs partial-scan ATPG techniques to dynamically analyze signal propagation and control paths without requiring full-scan or sequential ATPG, which are computationally intensive and limited in scope.  

### **Confidentiality Verification**  
Confidentiality verification ensures that sensitive information, such as cryptographic keys or privileged data, does not propagate to unauthorized observation points. The process begins by modeling an asset (e.g., a net carrying a secret key) as a stuck-at fault (both stuck-at-0 and stuck-at-1) and applying ATPG to detect whether the fault can be observed at any output or scan flip-flop. The algorithm proceeds as follows:  

1. **Asset Identification and Fault Injection**: The target asset is marked as a fault, and partial-scan ATPG is applied to determine if the fault can propagate to observable points.  
2. **Observation Point Analysis**: For each asset, the framework identifies all primary outputs and scan flip-flops within its fan-out cone. Capture masks are applied to isolate individual observation paths.  
3. **Path Sensitization**: ATPG generates test patterns to propagate the fault to each observation point. If both stuck-at-0 and stuck-at-1 conditions are detected, an information flow from the asset to the observation point is confirmed.  
4. **Malicious Path Identification**: Valid observation points (e.g., ciphertext outputs in an encryption module) are distinguished from malicious ones by analyzing propagation depth and signal path composition. A significantly shorter propagation depth or unexpected logic operations indicate a potential Trojan path.  
5. **Intersect Analysis**: To detect Type II Trojans (where leakage circuits are functionally isolated from legitimate logic), the framework cross-references observation points with the fan-in cones of valid outputs. Any observation point outside these cones is flagged as malicious.  

### **Integrity Verification**  
Integrity verification ensures that critical assets cannot be influenced by unauthorized control points. The methodology mirrors confidentiality verification but focuses on fan-in rather than fan-out analysis:  

1. **Control Point Identification**: The framework identifies all primary inputs and scan flip-flops capable of influencing the asset through sequential ATPG.  
2. **Fault Activation Analysis**: Test patterns are generated to determine if control points can alter the asset’s state. Successful activation confirms an integrity violation.  
3. **Malicious Control Detection**: Valid control points (e.g., pipeline registers in a microprocessor) are compared against identified control paths. Unauthorized control points (e.g., rogue counters or state machines) are flagged as malicious.  

### **Trigger Condition Extraction**  
For Trojans with finite state machine (FSM)-based triggers, the framework extracts activation sequences by reverse-engineering the FSM’s state transition graph (STG). This involves:  

1. **State Register Identification**: Registers with feedback loops are identified as potential state elements.  
2. **FSM Extraction**: The STG is reconstructed by analyzing present-state and input conditions that lead to Trojan activation states.  
3. **Trigger Sequence Derivation**: The input sequence required to reach the malicious state is extracted, enabling precise Trojan characterization.  

### **Experimental Validation**  
The framework was validated using Trust-Hub benchmark circuits, successfully detecting Trojans such as AES-T100 (key leakage via covert channels) and s35932-T100 (scan-chain exploitation). Comparative analysis demonstrated superiority over formal methods, GLIFT, and Jasper SPV, particularly in detecting deeply embedded or scan-mode Trojans.  

### **Limitations**  
The primary limitation lies in ATPG’s inability to handle latches or uncontrollable flip-flops, which may obscure certain observation/control points. However, such components are rare in properly synthesized designs, and warnings are issued for manual inspection.  

### **Conclusion**  
The proposed IFS verification framework provides a scalable, golden-reference-free solution for hardware Trojan detection, addressing critical gaps in existing methodologies. By leveraging ATPG and FSM extraction, it ensures comprehensive coverage of confidentiality and integrity violations, making it indispensable for securing third-party IPs in modern SoCs.