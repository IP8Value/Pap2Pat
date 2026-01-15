- The paper proposes an Information Flow Security (IFS) verification framework for detecting hardware Trojans without requiring white-box knowledge of intellectual property. It uses partial scan ATPG to identify all observe/control points that could be manipulated by a Trojan.

- Experimentally, the technique successfully detected various trust-hub Trojan benchmarks affecting confidentiality and integrity policies in encryption modules and microprocessors. The authors implemented AES-T100 payload in DES and PRESENT ciphers, which were also detected.

- Compared to formal methods, GLIFT, Jasper, and PCC techniques, this framework overcomes their limitations by not being constrained by clock cycles, distinguishing valid key paths from leakage paths, working with DfT-inserted netlists, and detecting power side-channel Trojans.

- The method can identify malicious observe/control points that violate IFS policies, even in designs where a designer lacks detailed knowledge of IP internals. It also successfully detected modified AES-T1100(M) Trojans designed to evade detection by formal methods.

- Limitations include potential undetected observe/control points if ATPG fails due to complexity or sequential depth, and inability to work with latches or uncontrollable flip-flops. However, the framework warns about such components for manual analysis.

- In conclusion, this IFS verification technique effectively detects both intentional hardware Trojans and unintentional design flaws that violate security policies. It provides a robust solution for ensuring IP integrity in complex SoC designs without requiring detailed internal knowledge.