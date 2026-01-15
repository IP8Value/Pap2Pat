# DESCRIPTION

## BACKGROUND

- introduce cryptographic systems  
Cryptographic systems are foundational to secure communication in digital environments, enabling entities to exchange information with confidentiality, integrity, and authenticity over untrusted channels. These systems rely on mathematical constructs that are computationally infeasible to reverse or predict without knowledge of a secret key. Symmetric key cryptography, in particular, employs a single shared key for both encryption and decryption, offering efficiency and speed suitable for high-throughput applications such as real-time data transmission, embedded systems, and military communications. The design of such systems must balance computational efficiency with resistance to a wide array of adversarial techniques, including brute force, differential cryptanalysis, linear cryptanalysis, and algebraic attacks. Modern cryptographic standards often prioritize algorithmic simplicity, hardware compatibility, and resistance to side-channel attacks, yet many existing solutions suffer from rigid architectures that limit adaptability or require multiple passes for authentication and encryption, thereby reducing performance and increasing implementation complexity.

- describe sponge functions  
Sponge functions represent a versatile cryptographic primitive that generalizes the concept of hash functions by allowing variable-length input and output. Unlike traditional hash functions constrained to fixed-size outputs, sponge functions operate by absorbing input data into an internal state and then squeezing out output bits of arbitrary length. This dual-phase mechanism—absorption followed by squeezing—enables a single construction to serve multiple roles, including hashing, stream ciphering, message authentication, and authenticated encryption. The security of a sponge function is governed by its internal state structure, which is partitioned into a rate portion accessible during input/output operations and a capacity portion that remains hidden and determines the resistance to collision and preimage attacks. The underlying function, typically a permutation or transformation, is iteratively applied to mix the state after each absorption or squeezing step, ensuring that small changes in input propagate throughout the entire state space.

- explain sponge construction  
The sponge construction operates by initializing a fixed-size internal state, typically composed of a rate and a capacity component, and then processing input data in blocks of size equal to the rate. Each input block is XORed with the corresponding portion of the state, after which the entire state is transformed by a deterministic, invertible function. This process continues until all input data is consumed, at which point the squeezing phase begins: output bits are extracted from the rate portion of the state, and the transformation function is reapplied after each extraction to ensure entropy dispersion. The construction is reinitialized between independent operations, ensuring isolation between distinct cryptographic tasks. The security of the sponge construction is formally bounded by the capacity of the internal state, with higher capacity values providing greater resistance to generic attacks. This framework has been successfully standardized in SHA-3, demonstrating its robustness and adaptability across diverse cryptographic applications.

- detail duplex constructions  
The duplex construction extends the sponge model by maintaining a persistent internal state across multiple sequential operations, eliminating the need for reinitialization between absorption and squeezing phases. This allows for simultaneous input absorption and output generation within a single call, enabling efficient, stateful cryptographic protocols such as authenticated encryption. In a duplex object, each invocation can both consume input and produce output, with the internal state evolving continuously throughout the sequence of operations. This design supports flexible use cases such as key derivation, stream ciphering with authentication, and session-based encryption, where the context must persist between message segments. The duplex construction retains the security guarantees of the sponge model but introduces additional complexity in state management, requiring careful handling of domain separation and input/output ordering to prevent ambiguity or replay vulnerabilities.

- limitations of prior art  
Prior cryptographic systems often require separate algorithms for encryption and authentication, leading to increased latency, key management overhead, and implementation complexity. Many authenticated encryption modes, such as CCM and GCM, rely on nested operations that necessitate multiple passes over the data, reducing throughput and increasing vulnerability to implementation errors. Other approaches, including stream ciphers with appended MACs, suffer from weak domain separation or predictable keystream generation, making them susceptible to replay and forgery attacks. Furthermore, existing permutations frequently employ small S-boxes, such as 8-bit substitutions, which limit non-linearity and algebraic complexity, forcing designers to compensate with additional rounds that degrade performance. These systems are also typically inflexible, offering no mechanism for customization without full reanalysis, rendering them unsuitable for applications requiring unique, proprietary implementations such as classified military or government communications.

## SUMMARY OF THE INVENTION

- motivate encryption systems  
Modern encryption systems must provide not only confidentiality but also strong authentication and resistance to adaptive attacks, particularly in environments where computational resources are constrained and adversaries possess significant analytical capabilities. The demand for high-performance, low-latency cryptographic primitives has intensified with the proliferation of IoT devices, real-time communication networks, and secure embedded systems. Traditional approaches that rely on separate encryption and authentication mechanisms are no longer sufficient due to their inefficiency and susceptibility to implementation flaws. A unified, stateful, and customizable cryptographic framework is required—one that minimizes computational overhead, avoids key proliferation, and enables tailored security profiles without compromising the underlying cryptographic strength.

- outline permutation function  
The invention introduces a novel permutation function designed for use within a duplex construction, characterized by a 512-bit internal state and composed of iterative rounds that incorporate 16×16 bijective substitution boxes, a bitwise permutation layer, a mixing layer based on finite field multiplication, and round constant addition. This permutation is non-linear, invertible, and entropy-preserving, ensuring that no information is lost during state transformation. Each round applies these layers in sequence to maximize diffusion and confusion, with the 16-bit S-boxes serving as the primary source of non-linearity. The permutation is structured to resist differential and linear cryptanalysis through a combination of high branch numbers, deranged bitwise diffusion, and unique round constants derived from cryptographic hash functions.

- describe keystream generation  
Keystream generation is achieved through the duplex construction by first absorbing a cryptographic key and optional initialization vector into the internal state via mute calls, followed by alternating absorption of plaintext blocks and squeezing of corresponding keystream blocks. Each keystream block is produced by extracting the rate portion of the state after applying the permutation function, with the output XORed directly with the plaintext to produce ciphertext. The state is maintained continuously between operations, ensuring that keystream generation is context-dependent and resistant to replay or reordering attacks. The rate of keystream production is fixed at 128 bits per squeeze, enabling efficient hardware pipelining and alignment with common data word sizes.

- application of sponge framework  
The invention leverages the sponge framework as the foundational architecture for authenticated encryption, utilizing its inherent ability to unify encryption and authentication within a single stateful process. By embedding the permutation function within a duplex construction, the system eliminates the need for separate MAC generation or additional passes over the data. The capacity portion of the state ensures security against generic attacks, while the rate enables high-speed throughput. The sponge’s reusability and modularity allow the same permutation to be applied across key derivation, message encryption, and tag generation, reducing code footprint and enhancing implementation consistency.

- embodiment of duplex framework  
The invention is embodied as a duplex object that maintains a persistent 512-bit state, partitioned into a 128-bit rate and a 384-bit capacity. The object accepts a key of either 128 or 256 bits, with the same rate and capacity preserved across both configurations to simplify implementation. The duplex object supports mute calls for key and header absorption, blank calls for tag generation, and interleaved input/output calls for encrypted message processing. Domain separation is implicitly managed by the sequence of operations, with no explicit frame bits required. The system is optimized for hardware deployment, with all operations designed for bitwise parallelism and minimal gate count, enabling deployment on FPGA and ASIC platforms with low power consumption and high throughput.

## DETAILED DESCRIPTION

- introduce patent scope  
This patent covers a cryptographic system and method for authenticated encryption based on a customizable duplex construction employing a novel permutation function with 16×16 S-boxes, designed for high-security, low-latency hardware implementation. The system provides confidentiality, integrity, and authenticity in a single pass, without requiring multiple keys or repeated data processing. The invention encompasses the structure, operation, and customization parameters of the permutation function, the duplex object, and the associated cryptographic protocols for key absorption, keystream generation, and authentication tag production.

- describe patent limitations  
The invention is limited to implementations that utilize the specified 512-bit state size, 128-bit rate, and 384-bit capacity. It does not extend to variations that alter the fundamental architecture of the duplex construction or replace the permutation function with a non-bijective transformation. The system assumes that padding and domain separation are handled at a higher protocol layer and does not prescribe specific bit-level encoding for input framing. The invention is not intended for software-only implementations that lack hardware-optimized S-box or mixer circuits, as its security and efficiency are predicated on the physical realization of the described components.

- define invention scope  
The scope of the invention includes all cryptographic systems that implement the duplex construction with the specified permutation function, including but not limited to hardware accelerators, embedded security modules, secure communication protocols, and cryptographic libraries. The invention further encompasses any system that employs the 16×16 S-box defined herein, the specified bitwise permutation, the GF(2¹⁶) mixer, or the round constant derivation method based on SHA-3. Customizations to the S-box, mixer, or round constants are included within the scope provided they meet the stated cryptographic constraints and do not reduce the minimum security margin of 2¹²⁸ for 128-bit keys or 2²⁵⁶ for 256-bit keys.

- discuss embodiment variations  
Embodiments of the invention may vary in the choice of 16×16 S-box, provided the substitution layer retains bijectivity, non-linearity, and hardware efficiency comparable to the disclosed design. Alternative bitwise permutations may be employed if they satisfy the constraints of derangement, high order, uniform bit diffusion, and affine definability. The mixer may be substituted with any invertible 2×2 matrix over GF(2¹⁶) that achieves a differential and linear branch number of three. Round constants may be generated from alternative pseudorandom sources, provided they are unique per round and exhibit sufficient asymmetry to prevent slide attacks. All such variations remain within the scope of the invention if they preserve the security properties and operational structure of the duplex framework.

- explain language usage  
For the purposes of this patent, the term “permutation” refers to a bijective function that maps a fixed-size state to itself without loss of entropy. The term “rate” denotes the portion of the internal state accessible during input and output operations. The term “capacity” refers to the hidden portion of the state that determines security against generic attacks. The term “duplex object” refers to a stateful entity that maintains internal context across multiple invocation cycles. The term “mute call” denotes a duplex operation that absorbs input without producing output. The term “blank call” denotes a duplex operation that produces output without absorbing input. All technical terms are used consistently with their established cryptographic meanings unless explicitly redefined herein.

- define technical terms  
“S-box” refers to a substitution box that performs a non-linear, invertible mapping from a 16-bit input to a 16-bit output. “Bitwise permutation” refers to a rearrangement of individual bits across the state, implemented via wire routing rather than logical operations. “Mixing layer” refers to a linear transformation applied to pairs of 16-bit words using finite field multiplication. “Round constant” refers to a unique, non-zero value XORed into the state at each round to disrupt symmetry. “Authentication tag” refers to a fixed-length output generated after all message data has been processed, used to verify integrity and authenticity. “Keystream” refers to a pseudorandom bit sequence generated by the duplex construction and XORed with plaintext to produce ciphertext.

- introduce cryptographic algorithm  
The cryptographic algorithm of the invention is a symmetric key authenticated encryption system based on the duplex construction, instantiated with a permutation function composed of iterative rounds that apply substitution, bitwise permutation, mixing, and round constant addition. The algorithm accepts a key of 128 or 256 bits and produces ciphertext and an authentication tag from plaintext and optional header data in a single, stateful pass. The algorithm is designed for hardware implementation with minimal latency and maximum throughput, utilizing large S-boxes to enhance non-linearity without increasing the number of rounds.

- describe sponge construction framework  
The sponge construction framework of the invention comprises a 512-bit internal state divided into a 128-bit rate and a 384-bit capacity. Input data is absorbed in 128-bit blocks, each XORed into the rate portion of the state, followed by application of the permutation function. Output is generated by extracting the rate portion after each permutation, with the state preserved between operations. The framework ensures that the security of the system is determined solely by the capacity, while the rate governs throughput. The sponge is used in duplex mode, eliminating reinitialization and enabling continuous state evolution.

- illustrate sponge construction architecture  
The sponge construction architecture consists of a state register of 512 bits, a rate mask for input/output access, and a permutation engine that applies the round function iteratively. The rate portion is accessible via XOR gates during absorption and extraction, while the capacity portion remains isolated. The permutation engine is triggered after each block absorption or squeezing operation, ensuring full state mixing. The architecture is symmetric, with identical hardware paths for absorption and squeezing phases, enabling pipelined operation and parallel processing.

- explain absorbing phase  
The absorbing phase begins with the initialization of the internal state to zero. The key is absorbed first, followed by any header data, using mute calls that XOR input blocks into the rate portion of the state and trigger the permutation function. Each input block is processed sequentially, with the state evolving incrementally. The absorbing phase concludes when all key and header material has been incorporated, and the system is ready to process the message body.

- describe XOR operations  
XOR operations are performed between incoming data blocks and the rate portion of the state using bitwise exclusive-or logic. These operations are implemented in hardware using parallel XOR gates, with no carry propagation or arithmetic dependencies. The XOR operation ensures that input data is linearly mixed into the state without introducing non-linearity, preserving the role of the S-box as the sole source of non-linearity. The operation is reversible and deterministic, allowing the same state to be reconstructed under identical input conditions.

- detail permutation functional block  
The permutation functional block executes a sequence of four operations per round: substitution via 32 identical 16×16 S-boxes, bitwise permutation of all 512 bits, mixing of adjacent 16-bit words using GF(2¹⁶) matrix multiplication, and addition of a round constant via XOR. These operations are applied in fixed order and repeated for a number of rounds determined by key size: 10 rounds for 128-bit keys and 16 rounds for 256-bit keys. The block is implemented as a combinational logic circuit with no memory elements, ensuring deterministic and timing-constant behavior.

- explain squeezing phase  
The squeezing phase begins after all input data has been absorbed. The rate portion of the state is extracted as a keystream block and XORed with the plaintext to produce ciphertext. After each extraction, the permutation function is applied to the entire state to ensure entropy dispersion. This process repeats until all plaintext blocks are encrypted. The final keystream block is not used for encryption but is retained to generate the authentication tag via a blank call.

- describe keystream block production  
Keystream blocks are produced by extracting the 128-bit rate portion of the state after each permutation application during the squeezing phase. Each block is immediately XORed with a corresponding plaintext block to generate ciphertext. The state is not reinitialized between extractions, ensuring that each keystream block is dependent on all prior inputs. The production rate is fixed at one keystream block per permutation cycle, enabling predictable throughput and hardware pipelining.

- detail message encryption  
Message encryption is performed by absorbing each plaintext block into the duplex object via a duplex call that simultaneously absorbs the block and produces a keystream block. The keystream is XORed with the plaintext to produce ciphertext, which is output immediately. This process continues until all message blocks are processed. The encryption is stateful, meaning that the same plaintext block encrypted at different times will produce different ciphertexts due to the evolving internal state.

- introduce duplex construction framework  
The duplex construction framework of the invention is a stateful cryptographic primitive that enables simultaneous absorption and squeezing operations without reinitialization. It is instantiated with a 512-bit state, 128-bit rate, and 384-bit capacity, and operates via a sequence of mute, blank, and duplex calls. The framework supports authentication by generating a tag after all data has been processed, and encryption by interleaving keystream generation with message absorption. The framework is designed to be implemented in hardware with minimal latency and maximal throughput.

- illustrate duplex construction architecture  
The duplex construction architecture consists of a 512-bit state register, a rate mask, a permutation engine, and control logic for managing mute, blank, and duplex calls. The state register is updated after every call, and the permutation engine is triggered unconditionally after each state update. Input and output are routed through the rate portion using multiplexers controlled by the call type. The architecture supports pipelined operation, with the permutation engine overlapping with data transfer to maximize throughput.

- explain duplex object operations  
A duplex object is initialized with a zero state and accepts three types of operations: mute calls, which absorb input without output; blank calls, which produce output without input; and duplex calls, which absorb input and produce output simultaneously. Each operation modifies the internal state by XORing the input into the rate, applying the permutation, and optionally extracting the rate as output. The object maintains state between operations, allowing for context-aware encryption and authentication.

- describe padding functional block  
Padding is not performed within the invention’s core algorithm but is assumed to be handled at a higher protocol layer. Input data is expected to be aligned to 128-bit boundaries, with no explicit padding bits required. If padding is used externally, it must not interfere with the duplex object’s state evolution or introduce ambiguity in data type identification.

- detail XOR operations  
XOR operations are implemented using parallel bitwise logic gates, with each bit of the input block XORed with the corresponding bit of the rate portion. These operations are performed in a single clock cycle and do not introduce latency or carry propagation. The XOR operation is used exclusively for state mixing and keystream generation, with no arithmetic or modular operations involved.

- explain permutation functional block  
The permutation functional block applies four sequential operations per round: substitution via 32 identical 16×16 S-boxes, a bitwise permutation that rearranges all 512 bits according to a deranged affine function, a mixing layer that applies GF(2¹⁶) matrix multiplication to adjacent word pairs, and a round constant addition via XOR. These operations are repeated for 10 or 16 rounds depending on key size, ensuring sufficient diffusion and confusion. The block is implemented as a combinational circuit with no feedback loops or memory elements.

- describe keystream block production  
Keystream blocks are produced by extracting the 128-bit rate portion of the state after each permutation application during the squeezing phase. Each block is immediately XORed with a corresponding plaintext block to produce ciphertext. The state is not reinitialized between extractions, ensuring that each keystream block is dependent on all prior inputs. The production rate is fixed at one keystream block per permutation cycle, enabling predictable throughput and hardware pipelining.

- detail message encryption  
Message encryption is performed by absorbing each plaintext block into the duplex object via a duplex call that simultaneously absorbs the block and produces a keystream block. The keystream is XORed with the plaintext to produce ciphertext, which is output immediately. This process continues until all message blocks are processed. The encryption is stateful, meaning that the same plaintext block encrypted at different times will produce different ciphertexts due to the evolving internal state.

- explain authentication data processing  
Authentication data is processed by first absorbing any header data via mute calls, followed by the message body via duplex calls. After all data has been absorbed, a blank call is issued to generate an authentication tag, which is extracted from the rate portion of the state. The tag is a fixed-length output that depends on the entire sequence of inputs and the internal state evolution, ensuring integrity and authenticity.

- describe duplex object operations  
A duplex object is initialized with a zero state and accepts three types of operations: mute calls, which absorb input without output; blank calls, which produce output without input; and duplex calls, which absorb input and produce output simultaneously. Each operation modifies the internal state by XORing the input into the rate, applying the permutation, and optionally extracting the rate as output. The object maintains state between operations, allowing for context-aware encryption and authentication.

- introduce permutation function ƒ  
The permutation function ƒ is a bijective transformation that operates on a 512-bit state and is composed of Nᵣ rounds, where Nᵣ is 10 for a 128-bit key and 16 for a 256-bit key. Each round applies a substitution layer, a bitwise permutation, a mixing layer, and a round constant addition. The function is designed to be implemented in hardware with minimal gate count and maximum throughput, and it is not invertible in practice due to the complexity of its components.

- illustrate permutation function architecture  
The permutation function architecture consists of a 512-bit state register, four functional units (S-boxes, bitwise permutor, mixer, and round constant adder), and a control sequencer that applies the rounds in sequence. The state register is updated after each round, and the functional units operate in parallel within each round. The architecture is fully combinational, with no memory elements, ensuring deterministic and timing-constant behavior.

- explain round function ƒround  
The round function ƒround applies the four layers of the permutation in fixed order: substitution via 16×16 S-boxes, bitwise permutation of all 512 bits, mixing of adjacent 16-bit word pairs via GF(2¹⁶) matrix multiplication, and addition of a round-specific constant via XOR. Each layer is designed to contribute to diffusion and confusion, with the S-box providing non-linearity, the permutation enabling long-range diffusion, the mixer enhancing local diffusion, and the round constant preventing symmetry.

- detail substitution layer  
The substitution layer applies 32 identical 16×16 bijective S-boxes in parallel to the 32 16-bit words of the state. Each S-box is based on multiplicative inversion in GF(2¹⁶) followed by an affine transformation, ensuring high non-linearity and resistance to algebraic attacks. The S-boxes are implemented in hardware using 1,238 XOR gates and 144 AND gates, providing a minimal footprint while maintaining cryptographic strength.

- describe permutation layer  
The permutation layer rearranges all 512 bits of the state according to a deranged affine function with order 32, ensuring that no bit remains fixed and that diffusion spans the entire state. The permutation is implemented via wire routing in hardware, requiring no logic gates and introducing zero latency. The function satisfies five critical properties: derangement, high order, uniform bit diffusion, affine definability, and no low-order bits.

- explain mixing layer  
The mixing layer applies a 2×2 matrix multiplication over GF(2¹⁶) to pairs of adjacent 16-bit words, with the matrix chosen to achieve a differential and linear branch number of three. The operation is implemented using rotations and XOR gates, requiring no multipliers or lookup tables. The mixer ensures that any input difference propagates to at least two output words, enhancing resistance to differential and linear cryptanalysis.

- describe mixing function  
The mixing function takes two 16-bit words A and B as input and produces outputs A′ and B′ via the matrix multiplication [1, x; x, 1], where x is a primitive element in GF(2¹⁶). The operation is implemented using three XOR gates and a single wire rotation per word, enabling high-speed, low-power execution. The function is invertible and has maximal branch number, ensuring optimal diffusion.

- describe mixer hardware implementation  
The mixer is implemented using a combinational circuit consisting of 16-bit shift registers and XOR gates arranged to perform multiplication by x in GF(2¹⁶). Each word is rotated left by one position, and three XOR gates combine the rotated bits with the original bits to produce the result. The implementation requires no multipliers, multi-bit adders, or memory elements, making it ideal for FPGA and ASIC deployment.

- describe round constant addition layer  
The round constant addition layer XORs a unique 512-bit value into the state at each round to disrupt symmetry and prevent slide attacks. The constants are derived from the SHA-3-512 hash of the ASCII representation of the round index, ensuring uniqueness and unpredictability. The layer is implemented using 512 parallel XOR gates and requires no additional storage.

- describe round constant calculation  
Round constants are calculated by hashing the ASCII string representation of the round number (e.g., “0”, “1”, ..., “15”) using SHA-3-512. The resulting 512-bit digest is used directly as the round constant. This method ensures that each constant is unique, pseudorandom, and resistant to prediction or correlation.

- provide round constant values  
The round constants for the 10-round and 16-round variants are derived from SHA-3-512 hashes of the strings “0” through “9” and “0” through “15”, respectively. These values are fixed and publicly specified, but may be replaced with other pseudorandom values as long as they remain unique per round and satisfy the asymmetry requirement.

- describe FPGA implementation  
The invention is implemented on FPGA platforms using a fully combinational architecture with pipelined rounds, parallel S-boxes, and wire-routed permutation. The design occupies less than 15,000 LUTs and operates at clock frequencies exceeding 500 MHz, achieving throughput of over 64 Gbps. The state register is implemented using distributed RAM, and the S-boxes are realized using optimized logic trees. The design supports dynamic key switching and is resistant to side-channel attacks.

- describe post-quantum security  
The invention provides post-quantum security through the use of a large capacity (384 bits), which ensures resistance to Grover’s algorithm with a security level of 2¹⁹² for 128-bit keys and 2²⁵⁶ for 256-bit keys. The permutation function’s non-linearity and algebraic complexity further impede quantum algebraic attacks, making the system secure against known quantum cryptanalytic techniques.

- describe customizability  
The invention is highly customizable through modifications to the S-box, bitwise permutation, mixer matrix, and round constants, provided that cryptographic constraints are maintained. Users may generate proprietary variants without requiring full cryptanalysis, as long as the minimum security margin is preserved. This enables tailored implementations for government, military, and industrial applications requiring unique cryptographic identities.

- describe flow diagram for generating encrypted data  
The flow diagram begins with initialization of the duplex object to zero state. The key is absorbed via mute calls, followed by header data if present. Plaintext blocks are processed via duplex calls, each producing a keystream block that is XORed with the plaintext to generate ciphertext. After all plaintext is processed, a blank call generates the authentication tag. The ciphertext and tag are output as the final encrypted result.

- describe optional key concatenation  
The key may be optionally concatenated with an initialization vector before absorption, provided the total length does not exceed the rate. This concatenation is performed externally and is not part of the core algorithm. The duplex object treats the concatenated value as a single input block, preserving state continuity.

- describe state initialization  
State initialization sets all 512 bits of the internal state to zero. This is the default configuration, but the capacity portion may be initialized to any 384-bit value to provide a unique system fingerprint. Initialization occurs once per session and precedes all key absorption operations.

- describe keystream generation  
Keystream generation occurs during the squeezing phase, where the rate portion of the state is extracted after each permutation application. Each 128-bit keystream block is XORed with a corresponding plaintext block to produce ciphertext. The state evolves continuously, ensuring that keystream blocks are dependent on all prior inputs and cannot be predicted without knowledge of the internal state.

- describe optional keystream truncation  
Keystream truncation is not permitted within the core algorithm. The full 128-bit keystream block must be used for each plaintext block. Truncation would reduce security and is therefore excluded from compliant implementations.

- describe data padding  
Data padding is not performed by the core algorithm. Input data must be aligned to 128-bit boundaries. If padding is required externally, it must be applied before absorption and must not interfere with the duplex object’s state evolution.

- describe data encryption  
Data encryption is performed by XORing each 128-bit plaintext block with a corresponding 128-bit keystream block generated by the duplex construction. The keystream is derived from the evolving internal state, ensuring that identical plaintext blocks produce different ciphertexts when encrypted at different times.

- describe sponge framework  
The sponge framework provides the foundational architecture for the invention, enabling variable-length input and output through absorption and squeezing phases. The framework is implemented in duplex mode, with persistent state and no reinitialization, allowing for efficient authenticated encryption in a single pass.

- describe duplex framework  
The duplex framework enables simultaneous absorption and squeezing operations within a single stateful object. It supports mute, blank, and duplex calls, allowing for flexible key absorption, header authentication, message encryption, and tag generation. The framework ensures that all operations are context-dependent and resistant to replay or reordering attacks.