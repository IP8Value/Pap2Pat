# DESCRIPTION

## TECHNICAL FIELD

The present disclosure relates to the field of cryptography, particularly to a novel class of cryptosystems known as Verifiable Encryption (VE). The invention further pertains to an authentication algorithm based on VE, which is designed to enhance security and efficiency in the context of digital identity verification, especially in scenarios involving the unlocking of local devices via a network.

## BACKGROUND ART

In the era of network services and the Internet of Things (IoT), the efficient and secure processing of encrypted big data has become increasingly important. Traditional cryptographic methods often struggle to balance speed and security, leading to high resource costs in terms of computational power, time, and memory usage. Searchable encryption (SE) has been a promising approach, allowing keyword searches on encrypted data without revealing additional information beyond the search results. However, SE methods typically require decryption to perform operations such as distance calculations between encrypted data items, which can be computationally expensive.

Verifiable Encryption (VE) addresses these limitations by enabling the calculation of distances between encrypted data items without the need for decryption. This property makes VE particularly suitable for applications requiring fast and secure authentication, such as verifying digital identities in cloud services, IoT devices, and other networked environments. Known authentication algorithms, such as those proposed by Fiat-Shamir and Schnorr, can be shown to belong to the class of VE, providing a theoretical foundation for the development of new, more efficient authentication protocols.

## SUMMARY OF INVENTION

### Technical Problem

Current authentication methods often rely on public key agreements (PKA) and secure socket layer/transport layer security (SSL/TLS) protocols, which can be resource-intensive and time-consuming. Additionally, these methods may not be well-suited for scenarios where local devices need to be unlocked via a network, as they often lack sufficient computational power to handle complex cryptographic operations. There is a need for an authentication algorithm that is both fast and secure, capable of operating in resource-constrained environments without compromising the integrity of digital identities.

### Solution to Problem

The present invention provides a class of cryptosystems known as Verifiable Encryption (VE), which allows the calculation of distances between encrypted data items without decryption. This property is leveraged to construct a fast and secure authentication algorithm specifically designed for unlocking local devices via a network. The algorithm does not require key distribution or interactive proofs, making it deterministic and efficient.

### Advantageous Effects

The proposed VE-based authentication algorithm offers several advantages over existing methods:
1. **Enhanced Security**: By performing operations in the encrypted domain, the algorithm minimizes the risk of plaintext attacks and ensures that sensitive information remains protected.
2. **Efficiency**: The algorithm is deterministic and does not require the overhead associated with key distribution or interactive proofs, making it suitable for resource-constrained devices.
3. **Flexibility**: The algorithm can be adapted to various types of digital identities, including biometric data, and can be implemented using different cryptosystems, such as the one-time pad.
4. **Scalability**: The algorithm maintains consistent performance across different plaintext lengths, ensuring that it remains efficient even as the amount of data increases.

## DESCRIPTION OF EMBODIMENTS

### Outline of Exemplary Embodiments of Present Disclosure

The present disclosure describes a novel class of cryptosystems, Verifiable Encryption (VE), and an authentication algorithm based on VE. The VE class includes cryptosystems that can calculate distances between encrypted data items without decryption. The authentication algorithm utilizes VE to enable fast and secure verification of digital identities, particularly in scenarios where local devices need to be unlocked via a network.

### Principles of Exemplary Embodiments of Present Disclosure

#### Verifiable Encryption (VE)

Verifiable Encryption (VE) is a class of cryptosystems that allows the calculation of distances between encrypted data items without the need for decryption. Formally, let \( P \), \( C \), and \( K \) be the spaces of plaintexts, ciphertexts, and keys, respectively. A set \( E \) of encryptions and a set \( D \) of decryptions are given by:

\[ E: P \times K \to C \]
\[ D: C \times K \to P \]

A metric \( V: P \times P \to \mathbb{R}^+ \) is defined to measure the distance between two plaintexts. The VE class is characterized by the existence of a function \( F: C \times C \to C \) and a function \( D: C \times K \times K \to \mathbb{R}^+ \) such that:

\[ D_{k_1, k_2}(F(E_{k_1}(p_1), E_{k_2}(p_2))) = V(p_1, p_2) \]

This property ensures that the distance between two plaintexts can be calculated directly from their corresponding ciphertexts, without the need for decryption.

#### Authentication Algorithm

The authentication algorithm based on VE operates as follows:

1. **Registration Step**:
   - **Step 1**: Alice sends her digital identity \( p_1 \) to Bob.
   - **Step 2**: Bob generates a key \( k \) and calculates the ciphertext \( c_1 = E_k(p_1) \).
   - **Step 3**: Bob sends \( c_1 \) to the server \( S \).

2. **Verification Step**:
   - **Step 1**: Alice sends her digital identity \( p_2 \) to Bob.
   - **Step 2**: Bob generates a one-time key \( k' \) and calculates the ciphertext \( c_2 = E_{k'}(p_2) \).
   - **Step 3**: Bob sends \( c_2 \) to \( S \).
   - **Step 4**: The server \( S \) calculates \( F(c_1, c_2) = c_d \) and sends \( c_d \) to Bob.
   - **Step 5**: Bob calculates \( D_{k, k'}(c_d) \) and checks the result. If the result is within a predefined threshold, the authentication is successful.

This algorithm does not require key distribution or interactive proofs, making it deterministic and efficient. The use of one-time keys ensures that the security of the system is maintained, even in the presence of potential eavesdroppers.

#### Implementation Example

An implementation of the VE-based authentication algorithm using the one-time pad cryptosystem is provided below:

1. **Registration Step**:
   - **Step 1**: User \( A \) sends their digital identity \( p_1 \) to the trusted device \( D \).
   - **Step 2**: \( D \) generates a key \( k_1 \) and calculates the ciphertext \( c_1 = E_{k_1}(p_1) \). \( D \) sends \( c_1 \) to the server \( S \).
   - **Step 3**: \( S \) stores \( c_1 \) in the database.

2. **Verification Step**:
   - **Step 1**: \( A \) sends their digital identity \( p_2 \) to \( D \).
   - **Step 2**: \( D \) generates a one-time key \( k_2 \) and calculates the ciphertext \( c_2 = E_{k_2}(p_2) \). \( D \) sends \( c_2 \) to \( S \).
   - **Step 3**: \( S \) calculates the encoded distance \( c_d = F(c_1, c_2) \) and sends \( c_d \) to \( D \).
   - **Step 4**: \( D \) decodes \( c_d \) with the keys \( k_1 \) and \( k_2 \) to obtain the distance \( V(p_1, p_2) \). If \( V(p_1, p_2) \leq s \), where \( s \) is a predefined threshold, \( D \) returns "OK" to \( A \); otherwise, it returns "NG".

#### Performance Evaluation

The performance of the VE-based authentication algorithm was evaluated using a 128-bit one-time pad cryptosystem. The experimental results showed that the algorithm maintains consistent performance across different plaintext lengths, with encryption, verification, and decryption processes completing in less than 0.003 milliseconds for plaintext lengths up to 8192 bits. The algorithm is robust against impersonation attacks and plain text attacks, ensuring the security of digital identities.

#### Conclusion

The present invention introduces a novel class of cryptosystems, Verifiable Encryption (VE), and an authentication algorithm based on VE. The algorithm is designed to be fast, secure, and efficient, making it suitable for applications involving the unlocking of local devices via a network. The use of one-time keys and the ability to perform operations in the encrypted domain ensure that the algorithm maintains high security while minimizing resource consumption.