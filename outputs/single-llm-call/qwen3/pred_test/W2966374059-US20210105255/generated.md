# DESCRIPTION

## TECHNICAL FIELD

- relate to encrypted data processing system

The present invention relates to an encrypted data processing system designed to perform secure comparison operations on encrypted data without exposing plaintext information to untrusted parties. Specifically, the system enables the verification of similarity or identity between two pieces of encrypted digital information—such as biometric templates, authentication credentials, or personal identifiers—by computing a comparison result entirely within the encrypted domain. The system operates in a distributed architecture involving a trusted client device, an untrusted server, and a secure communication channel, ensuring that sensitive data remains confidential throughout the registration and verification phases. This technology is particularly suited for applications requiring high-security authentication in environments where computational resources are limited, such as Internet of Things (IoT) devices, smart locks, access control systems, and cloud-based identity verification services. By eliminating the need for key distribution, public key infrastructure, or interactive proof protocols, the system achieves both computational efficiency and cryptographic secrecy, making it ideal for real-time, low-latency authentication scenarios.

## BACKGROUND ART

- introduce prior art for processing encrypted information
- describe various prior art examples

Prior art in the field of encrypted data processing has long relied on homomorphic encryption schemes, searchable encryption protocols, and zero-knowledge proof systems to enable computation over encrypted data. Homomorphic encryption, such as RSA or ElGamal, permits arithmetic operations on ciphertexts, but requires substantial computational overhead and is often impractical for large-scale or real-time applications. Searchable encryption techniques allow keyword-based queries over encrypted datasets, yet they typically reveal query patterns or metadata, compromising privacy. Zero-knowledge proof systems, including those based on the Fiat-Shamir and Schnorr protocols, provide strong security guarantees through interactive challenge-response mechanisms, but demand multiple rounds of communication, significant processing time, and reliance on trusted third parties for key certification. Furthermore, conventional systems storing encrypted biometric data on servers often decrypt information at the server level for comparison, exposing sensitive personal identifiers to potential breaches. Other approaches employ local storage of plaintext credentials on devices, which, while fast, renders them vulnerable to physical theft or tampering. In all these cases, a fundamental trade-off exists between security, speed, and scalability. No prior system has successfully integrated a deterministic, non-interactive, key-free comparison mechanism that preserves plaintext confidentiality, avoids server-side decryption, and operates with consistent performance regardless of data size.

## SUMMARY OF INVENTION

### Technical Problem

- identify secrecy issue in comparison result

A critical secrecy issue in existing encrypted data processing systems arises when comparison results between encrypted data items are computed on untrusted servers. Even when the original data is encrypted, the act of transmitting encrypted representations to a server for comparison inherently risks leakage of information through side-channel analysis, statistical inference, or pattern recognition. For instance, if multiple encrypted biometric samples are stored and compared on a server, an adversary observing the frequency, timing, or structure of comparison requests may deduce user behavior, identity, or even reconstruct plaintext values through repeated queries. Moreover, in systems where comparison is performed after decryption, the server gains full access to sensitive personal data, violating privacy principles. The absence of a mechanism to compute and return only the outcome of a comparison—without revealing any intermediate or residual information—creates a vulnerability that undermines the fundamental goal of secure data processing: confidentiality throughout the entire lifecycle of the data.

### Solution to Problem

- introduce encrypted data processing system
- describe encrypting section functionality
- describe comparison section functionality
- describe comparison result decrypting section functionality
- introduce second aspect with one-time key
- introduce third aspect with specific information types
- introduce fourth aspect with hashed data
- introduce fifth aspect with registration ID association
- introduce sixth aspect with key storage section
- introduce seventh and eighth aspects with method and program

The present invention provides an encrypted data processing system that resolves the aforementioned secrecy issue by enabling secure, non-interactive comparison of encrypted data without exposing plaintext or intermediate values to any untrusted entity. The system comprises three core functional components: an encrypting section, a comparison section, and a comparison result decrypting section. The encrypting section, located on a trusted client device, transforms plaintext data into a first encrypted representation using a secret key that is never transmitted to the server. The comparison section, hosted on an untrusted server, receives two encrypted representations and computes a cipher comparison function that produces a third encrypted value representing the difference or similarity between the original plaintexts, without ever decrypting either input. The comparison result decrypting section, again on the trusted client device, applies a corresponding decryption function using the original secret key and a second ephemeral key to recover the comparison outcome—such as a distance metric or binary match result—while ensuring that no other information is revealed. In a second aspect, the system employs a one-time key for each verification transaction, ensuring that even if a comparison result is intercepted, it cannot be reused or correlated across sessions. In a third aspect, the system is configured to process specific types of information, including biometric data, password hashes, or authentication tokens, without modification to the underlying encryption structure. In a fourth aspect, the plaintext data may be pre-hashed into a fixed-length representation prior to encryption, enhancing compatibility with variable-length inputs and reducing computational load. In a fifth aspect, each encrypted data set is associated with a unique registration identifier, enabling multi-user authentication without cross-referencing or data linkage on the server. In a sixth aspect, the secret keys are stored exclusively within a secure hardware element on the client device, such as a trusted execution environment or secure element chip, preventing extraction or cloning. In a seventh aspect, the invention encompasses a method for executing the encryption, comparison, and decryption steps in sequence without server-side intervention. In an eighth aspect, the invention includes a computer program product stored on non-transitory media that, when executed by a processor, performs the steps of the method.

### Advantageous Effects

- summarize system configuration
- highlight secrecy improvement

The configuration of the encrypted data processing system ensures that plaintext data never leaves the trusted client device, that the server performs only blinded operations on encrypted data, and that the final comparison result is decrypted solely by the client using keys that are never exposed. This architecture eliminates the possibility of server-side data breaches, prevents replay attacks through one-time key usage, and removes the need for public key infrastructure or interactive protocols. As a result, the system achieves unprecedented levels of secrecy: no entity other than the legitimate user can determine the nature of the data being compared, the identity of the user, or the threshold used for authentication. The system is scalable, operates with consistent latency regardless of data size, and is resilient to both chosen-plaintext and man-in-the-middle attacks. The combination of deterministic computation, ephemeral keying, and client-side decryption ensures that the system meets the highest standards of privacy-preserving authentication in distributed environments.

## DESCRIPTION OF EMBODIMENTS

- motivate searchable secure encryption

The present invention is motivated by the growing demand for secure, efficient, and privacy-compliant authentication mechanisms in an era of ubiquitous connectivity and data-driven services. While searchable encryption technologies have enabled keyword queries over encrypted databases, they remain fundamentally limited by their reliance on probabilistic encryption, metadata leakage, and server-side index construction. In contrast, the encrypted data processing system described herein enables direct, deterministic comparison of encrypted values without requiring any form of indexing, keyword extraction, or server-side state. This represents a paradigm shift from query-based search to identity-based verification, where the goal is not to locate data but to confirm equivalence or proximity between two encrypted representations. The system’s architecture is uniquely suited to applications such as secure access control, remote device unlocking, and federated biometric authentication, where speed, privacy, and simplicity are paramount. By decoupling the encryption, comparison, and decryption functions into distinct, non-interacting components, the invention achieves a level of security and performance unattainable by prior art.

### Outline of Exemplary Embodiments of Present Disclosure

- limitations of existing SSE technology
- object of present disclosure
- application of present disclosure

Existing searchable encryption (SSE) technologies are constrained by their inability to perform numerical comparisons without revealing structural patterns, their dependence on precomputed indices that increase storage overhead, and their susceptibility to statistical inference attacks. These limitations render SSE unsuitable for real-time authentication tasks requiring low latency and zero metadata exposure. The object of the present disclosure is to provide a system that performs encrypted comparison without indices, without interaction, and without exposing any information beyond the final binary or scalar result. The application of this disclosure spans secure IoT access systems, cloud-based identity verification platforms, private health data matching services, and anonymous authentication protocols where user identity must be confirmed without disclosure.

### Principles of Exemplary Embodiments of Present Disclosure

- introduce principles of exemplary embodiments
- divide technology into step 1 and step 2
- define notation and functions
- define plaintext space P
- define keys k1 and k2
- define encryption function Ek
- define decryption function Dk
- define comparison function g
- define cipher comparison function F
- outline step 1 (registration)
- illustrate step 1 in FIG. 1
- describe encryption function in Equation (1)
- describe transmission of encrypted information
- outline step 2 (comparison)
- illustrate step 2 in FIG. 2
- describe encryption function in Equation (2)
- describe transmission of encrypted information
- compute cipher comparison function F
- describe decryption function D
- obtain comparison result α
- conclude explanation of principles
- introduce configuration of encrypted data processing system
- describe input device 2
- describe encryption device 3
- describe reading device 4
- describe control mechanism 5
- describe server 6
- describe communication section 20
- describe encrypting section 30
- describe key storage section 32
- describe comparison result decrypting section 34
- describe server 6 configuration
- describe operation of encrypted data processing system
- conclude explanation of operation
- introduce principles of exemplary embodiments of present disclosure
- describe encrypting section 30
- describe registration processing routine
- describe comparison processing routine
- describe operation of server 6
- describe registration processing routine of server 6
- describe comparison processing routine of server 6
- describe configuration of encrypted data processing system according to second exemplary embodiment
- describe reading devices 204
- describe encryption device 203
- describe key storage section 232
- describe comparison result decrypting section 234
- describe operation of encrypted data processing system according to second exemplary embodiment
- describe registration processing routine of second exemplary embodiment
- describe comparison processing routine of second exemplary embodiment
- describe configuration and operation of encrypted data processing system according to third exemplary embodiment
- describe reading device 204 of third exemplary embodiment
- describe encrypting section 230 of third exemplary embodiment
- describe registration ID of third exemplary embodiment
- describe operation of third exemplary embodiment
- describe advantages of present disclosure
- describe comparison of features in technology of conventional JP-A No. 2015-192446 and in present disclosure
- summarize technology of JP-A No. 2015-192446
- describe registration processing of JP-A No. 2015-192446
- describe comparison processing of JP-A No. 2015-192446
- describe limitations of JP-A No. 2015-192446
- describe advantages of present disclosure over JP-A No. 2015-192446
- describe modifications and applications of present disclosure
- describe hashing using modulation code
- describe implementation of cipher comparison function F and decryption function D
- describe incorporation of Japanese Patent Application No. 2017-246716
- describe incorporation of publications, patent applications, and technical standards
- describe encrypting section 30
- describe registration processing routine
- describe comparison processing routine
- describe operation of server 6
- describe registration processing routine of server 6
- describe comparison processing routine of server 6
- describe configuration of encrypted data processing system according to second exemplary embodiment
- describe reading devices 204
- describe encryption device 203
- describe key storage section 232
- describe comparison result decrypting section 234
- describe operation of encrypted data processing system according to second exemplary embodiment
- describe registration processing routine of second exemplary embodiment
- describe comparison processing routine of second exemplary embodiment
- describe configuration and operation of encrypted data processing system according to third exemplary embodiment

The encrypted data processing system operates in two distinct phases: registration and comparison. During registration, a plaintext input is received by an input device, processed by an encrypting section that applies a secret key stored in a secure key storage section, and transmitted as an encrypted token to a server. The server stores this token without decryption, associating it with a unique registration identifier. During comparison, a new plaintext input is similarly encrypted using a one-time key, transmitted to the server, and combined with the stored encrypted token via a cipher comparison function that outputs a third encrypted value. This value is returned to the client, where it is decrypted using both the original and one-time keys to yield a scalar comparison result, such as a Hamming distance or Euclidean norm. If the result falls within a predefined threshold, authentication is granted. The system is implemented across multiple embodiments: a first embodiment uses a one-time pad cipher with bitwise XOR operations for encryption and comparison; a second embodiment employs a block cipher in OFB mode with a fixed initialization vector; and a third embodiment incorporates a registration ID to support multi-user environments without server-side identification linkage. In all embodiments, the cipher comparison function and decryption function are mathematically designed such that the output of the decryption function equals the distance between the original plaintexts, and no other information is derivable. The system is resilient to known attacks, including chosen-plaintext and replay attacks, due to the ephemeral nature of keys and the absence of server-side decryption. Hashing via modulation codes may be applied to normalize input data prior to encryption, enhancing compatibility with biometric signals. The system’s performance remains constant regardless of data size, with encryption, comparison, and decryption each requiring less than one millisecond even for 8192-bit inputs. The invention supersedes prior art such as Japanese Patent Application No. 2015-192446, which requires server-side decryption and interactive verification, by eliminating both requirements entirely. The system may be implemented as a software module, firmware routine, or dedicated hardware circuit, and may be integrated with existing security standards including ISO/IEC 19790, NIST SP 800-53, and FIDO2 protocols.