# DESCRIPTION

## BACKGROUND

Cryptographic systems are foundational to secure digital communication, enabling parties to exchange information confidentially and verify its integrity in the presence of adversaries. Modern cryptographic protocols rely on symmetric key algorithms for efficiency, particularly in high-throughput or resource-constrained environments. These algorithms typically provide two core functionalities: encryption for confidentiality and authentication for data integrity and origin assurance. Historically, many cryptographic constructions have treated these functions separately, leading to inefficiencies such as multiple passes over the same data or the need for independent keys for encryption and authentication. Such designs complicate implementation, increase latency, and introduce potential vulnerabilities due to improper usage or key management errors. The demand for integrated authenticated encryption (AE) schemes—algorithms that simultaneously provide both confidentiality and authenticity—has grown significantly, especially in applications requiring high performance and provable security.

Sponge functions represent a significant advancement in cryptographic design, offering a flexible framework that generalizes hash functions to support variable-length inputs and outputs. Introduced prominently through the KECCAK algorithm, which became the SHA-3 standard, sponge constructions operate by iteratively applying a fixed underlying function to an internal state divided into two parts: a rate portion accessible to external input/output and a capacity portion hidden from direct observation. This structure enables a natural trade-off between performance (determined by the rate) and security (determined by the capacity). The sponge construction proceeds in two phases: an absorbing phase, during which input data is XORed into the rate portion of the state followed by application of the underlying function, and a squeezing phase, during which output is extracted from the rate portion, again interleaved with applications of the underlying function. This model has proven robust against generic attacks when instantiated with a secure permutation.

The sponge construction was later extended into the duplex construction, which eliminates the strict separation between absorption and squeezing. In the duplex model, input can be absorbed and output produced in a single operation, and the internal state is preserved across successive calls rather than reinitialized. This makes the duplex construction particularly well-suited for streaming applications such as authenticated encryption, where headers (authenticated but not encrypted), payloads (both encrypted and authenticated), and authentication tags must be processed incrementally. The duplex object maintains continuity of state, allowing for efficient processing of segmented data without buffering entire messages. Domain separation techniques—such as appending distinguishing bits to different types of input—are often employed to prevent ambiguity between key material, associated data, and plaintext.

Despite these advances, prior art in authenticated encryption exhibits several limitations. Many existing schemes require two separate cryptographic primitives—one for encryption and another for authentication—or necessitate two distinct keys, complicating key management and increasing the risk of misuse. Some algorithms, such as CCM or Phelix, suffer from structural weaknesses or operational constraints that limit their applicability in high-assurance environments like government or military communications. Furthermore, most standardized AE schemes are fixed in design, offering no mechanism for customization without compromising security. This lack of adaptability prevents organizations from deploying unique cryptographic variants tailored to specific threat models or operational requirements, thereby increasing systemic risk if a single algorithm is compromised. Consequently, there exists a pressing need for a secure, efficient, and customizable authenticated encryption algorithm built upon sound theoretical foundations and optimized for hardware implementation.

## SUMMARY OF THE INVENTION

The present invention addresses the shortcomings of prior cryptographic systems by providing a novel authenticated encryption algorithm based on the duplex construction, designed for high security, efficiency, and customizability. The invention is motivated by the need for a cryptographic primitive that simultaneously ensures confidentiality and authenticity while supporting flexible deployment across diverse security environments, including those requiring proprietary or user-specific instantiations. Central to the invention is a permutation function that serves as the core cryptographic engine within a sponge-based framework, enabling both keystream generation for encryption and tag generation for authentication through a unified process.

The permutation function is structured as an iterated round function comprising substitution, bitwise permutation, mixing, and round constant addition layers. It operates on a 512-bit internal state and employs large 16×16 bijective S-boxes derived from finite field inversion and affine transformation, which confer high nonlinearity and algebraic complexity with minimal hardware overhead. This design significantly raises the computational barrier against cryptanalytic attacks—particularly differential and linear cryptanalysis—without proportionally increasing the number of rounds, thereby preserving throughput.

Keystream generation is achieved by initializing the duplex object with a secret key (optionally concatenated with an initialization vector), absorbing associated data via mute calls, and then duplexing plaintext blocks to produce ciphertext through XOR with the generated keystream. Authentication tags are derived from a final blank call to the duplex object after all data has been processed. The entire process occurs in a single pass, eliminating the inefficiencies of two-pass schemes.

The invention applies the sponge framework in a keyed mode, where the internal state is partitioned into a 128-bit rate and a 384-bit capacity, ensuring a security level commensurate with 128-bit or 256-bit keys. By leveraging the generic security guarantees of the keyed sponge construction, the algorithm inherits provable resistance to generic attacks, provided the underlying permutation remains secure.

In a preferred embodiment, the invention is realized through the duplex framework, which allows seamless integration of key loading, associated data authentication, plaintext encryption, and tag generation within a continuous stateful process. This embodiment supports arbitrary-length inputs and outputs, accommodates optional domain separation, and facilitates hardware-efficient implementation due to the regular structure of its components. The resulting system provides a robust, high-performance solution for authenticated encryption that meets stringent security requirements while enabling safe customization for specialized applications.

## DETAILED DESCRIPTION

The scope of the present patent encompasses methods, systems, and apparatuses for performing authenticated encryption using a customizable cryptographic algorithm based on the duplex construction and a novel permutation function. The invention is not limited to any specific hardware or software platform but is particularly advantageous in environments where hardware efficiency, post-quantum readiness, and algorithmic uniqueness are valued. Patent protection extends to all implementations that embody the core architectural and functional principles disclosed herein, including variations that adhere to the defined security parameters.

Limitations of the invention include adherence to specified key sizes (128 or 256 bits), state size (512 bits), and minimum round counts (10 or 16, respectively), as deviations beyond these may compromise security. However, within these bounds, extensive customization is permitted without invalidating the security model.

The invention’s scope includes any cryptographic system that utilizes a duplex object initialized with a secret key, processes associated data and plaintext through sequential duplexing operations, and produces ciphertext and an authentication tag, wherein the underlying permutation employs 16-bit S-boxes, a bitwise derangement permutation, a 2×2 matrix mixer over GF(2¹⁶), and round constants derived from a cryptographically secure hash function.

Embodiment variations may include alternative initial inner state values, different but compliant S-boxes, other bitwise permutations satisfying the stated constraints, alternative invertible mixers with branch number three, and distinct round constant sequences, provided all maintain the required cryptographic properties.

Language used herein follows standard patent convention: “comprising” is open-ended, “consisting of” is exclusive, and terms like “means,” “step,” and “configured to” invoke functional claiming under 35 U.S.C. § 112(f) where appropriate. All technical terms are defined contextually or explicitly.

Technical terms include “duplex object” (a stateful cryptographic entity maintaining internal state across operations), “rate” (the externally accessible portion of the sponge state), “capacity” (the hidden portion determining security), “mute call” (a duplex operation with input but no output), and “blank call” (a duplex operation with no input but producing output).

The cryptographic algorithm begins with key initialization, followed by optional IV concatenation, state setup, and sequential processing of header and body data via the duplex framework.

The sponge construction framework divides the 512-bit state into 128-bit rate and 384-bit capacity. Input is padded to align with the rate, absorbed via XOR into the rate, and the permutation applied iteratively.

The sponge architecture comprises a register holding the full state, an XOR unit for absorption, and the permutation block for state evolution.

During the absorbing phase, input blocks are XORed into the rate portion, followed by invocation of the permutation function.

XOR operations are bitwise and occur between input data and the current rate segment of the state.

The permutation functional block executes a sequence of rounds, each consisting of substitution, bitwise permutation, mixing, and round constant addition.

The squeezing phase extracts output by reading the rate portion; after the first block, each subsequent r-bit output requires a permutation call.

Keystream blocks are produced during squeezing and used to encrypt plaintext via XOR.

Message encryption is performed by XORing plaintext blocks with corresponding keystream blocks generated via the sponge or duplex process.

The duplex construction framework extends the sponge by maintaining state across calls and allowing concurrent absorption and squeezing.

The duplex architecture includes a persistent state register, input buffer, output buffer, and control logic for managing mute, blank, and standard duplex calls.

Duplex object operations involve invoking the duplexing function with optional input σᵢ and optional output length ℓᵢ, updating internal state accordingly.

A padding functional block ensures inputs conform to rate boundaries, though padding may be delegated to higher-layer protocols.

XOR operations in the duplex mirror those in the sponge, applied during input absorption.

The permutation functional block in the duplex is identical to that in the sponge, ensuring consistency.

Keystream block production in the duplex occurs when body data is processed; the output of the duplex call is XORed with plaintext to yield ciphertext.

Message encryption in the duplex integrates encryption and authentication in one pass: headers are absorbed silently, body blocks generate keystream for encryption, and a final blank call yields the authentication tag.

Authentication data processing involves absorbing associated data (headers) via mute calls, ensuring inclusion in the final tag without encryption.

Duplex object operations manage the lifecycle of the cryptographic session, from key loading to tag output.

The permutation function ƒ is a bijective mapping on 512 bits, composed of Nᵣ rounds (10 or 16).

Its architecture treats the state as 32 words of 16 bits each, facilitating S-box and mixer operations.

Each round applies a round function ƒ_round comprising four layers.

The substitution layer applies 32 parallel 16×16 S-boxes, providing nonlinearity.

The permutation layer reroutes bits across the entire state via a fixed bitwise permutation.

The mixing layer applies a 2×2 matrix multiplication over GF(2¹⁶) to adjacent word pairs, enhancing diffusion.

The round constant addition layer XORs a unique 512-bit constant per round to break symmetry.

Customization of the permutation is supported through five dimensions: initial inner state, S-box selection, bitwise permutation choice, mixer matrix variation, and round constant generation, all within defined security constraints.

Round function iteration occurs Nᵣ times, with Nᵣ chosen to exceed the threshold for resistance to differential and linear attacks by a wide margin.

Substitution box operations map 16-bit inputs to 16-bit outputs via inversion in GF(2¹⁶)/p(x), p(x)=x¹⁶+x⁵+x³+x+1, followed by an affine transform.

S-box properties include bijectivity, high nonlinearity, low differential uniformity (max 2⁻¹⁴), and compact hardware implementation (1238 XOR, 144 AND gates).

S-box implementation avoids lookup tables by computing field operations directly in hardware.

The S-box function is defined as S(x) = A·x⁻¹ + b for x ≠ 0, S(0) = b, where A is a 16×16 binary matrix and b a constant vector.

The inverse S-box function is similarly defined for decryption, though not required in this AE mode.

Hardware implementation uses combinational logic for inversion and affine steps, optimized for gate count.

The permutation layer implements a bitwise derangement with order 32, satisfying constraints on diffusion and S-box dispersion.

The bitwise permutation function maps bit index i to (17i + 3) mod 512, ensuring no fixed points and maximal cycle length.

Permutation function properties include being an affine derangement, high order, and compliance with diffusion criteria.

The affine permutation function is defined by a linear transformation plus constant offset, though in this case it is purely linear.

The mixing layer applies a symmetric 2×2 matrix over GF(2¹⁶) modulo q(x)=x¹⁶+x⁵+x³+x²+1.

The mixing function takes two 16-bit words A, B and outputs A' = A + x·B, B' = x·A + (x+1)·B, where x· denotes multiplication by the field element x.

Mixer hardware implementation uses wire rotations and three XOR gates per multiplication by x.

The round constant addition layer XORs RCᵢ = SHA3-512(ASCII(i)) into the state at round i.

Round constants are calculated offline using SHA3-512 to ensure uniqueness and asymmetry.

Exemplary round constant values are deterministic outputs of the above formula for i=0 to Nᵣ−1.

FPGA implementation is efficient due to regular structure, parallel S-boxes, and simple bitwise operations, yielding low area and high throughput.

Post-quantum security is enhanced by large S-box algebraic complexity, which resists known quantum cryptanalytic techniques targeting small algebraic structures.

Customizability allows users to generate unique algorithm variants by altering non-cryptographic parameters, enabling proprietary deployments without revalidation.

The flow diagram for generating encrypted data includes: optional key||IV concatenation, state initialization to zero (or custom inner state), mute calls to absorb key, mute calls for associated data, duplex calls for plaintext producing ciphertext, and a blank call for the tag.

Optional key concatenation appends a nonce or IV to the key before absorption to support randomized encryption.

State initialization sets the 512-bit state to all zeros by default, but the inner 384 bits may be set to a user-defined constant.

Keystream generation occurs implicitly during duplex calls on body data; the output is the keystream block.

Optional keystream truncation may be applied if output length does not match plaintext block size.

Data padding aligns inputs to the 128-bit rate boundary using standard methods (e.g., 10*1 padding), though may be omitted if handled externally.

Data encryption XORs each plaintext block with the corresponding keystream block from the duplex output.

The sponge framework is used in keyed mode for standalone hashing or PRF applications.

The duplex framework is the primary mode for authenticated encryption, integrating all operations in a single stateful process.