Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present disclosure relates to the field of cryptographic systems and authentication protocols, particularly to verifiable encryption (VE) methods that enable distance calculations between ciphertexts without decryption. More specifically, the invention pertains to a novel class of cryptosystems capable of performing secure and efficient authentication, particularly suitable for unlocking local devices via a network while maintaining computational efficiency and security against plaintext attacks.  

## BACKGROUND ART  

Conventional cryptographic systems, such as searchable encryption (SE), homomorphic encryption, and interactive proof-based authentication protocols like Fiat-Shamir and Schnorr, have been widely used for secure data processing and identity verification. However, these methods suffer from significant computational overhead, key distribution complexities, and inefficiencies when applied to resource-constrained environments such as IoT devices or cloud-based authentication systems.  

Existing authentication mechanisms often require decryption of ciphertexts to compare plaintexts, introducing latency and security vulnerabilities. Homomorphic encryption allows limited operations on ciphertexts but is computationally expensive and impractical for real-time applications. Zero-knowledge proofs, while secure, involve multiple rounds of communication, making them unsuitable for high-frequency authentication scenarios like unlocking devices via networks.  

There remains a need for a cryptographic system that balances speed, security, and computational efficiency, particularly for applications requiring frequent authentication without key distribution or decryption.  

## SUMMARY OF INVENTION  

### Technical Problem  

The primary technical problem addressed by this invention is the inefficiency and security limitations of existing cryptographic systems in performing distance calculations between encrypted data without decryption. Conventional methods require either costly homomorphic operations or interactive proofs, which are impractical for real-time authentication in networked environments, particularly for unlocking local devices via a network.  

### Solution to Problem  

The invention provides a class of cryptosystems termed verifiable encryption (VE), which enables distance calculations between ciphertexts without decryption. The VE system comprises two cryptosystems, \( C_1 = (P, C, K, E, D) \) and \( C_2 = (P, C, K', E', D') \), where:  
- \( P \) is the plaintext space,  
- \( C \) is the ciphertext space,  
- \( K, K' \) are key spaces,  
- \( E, E' \) are encryption functions, and  
- \( D, D' \) are decryption functions.  

A metric \( V: P \times P \rightarrow \mathbb{R}^+ \) is defined to measure the distance between plaintexts. The system includes a mapping \( F: C \times C \rightarrow C \) and a function \( D: C \rightarrow \mathbb{R}^+ \) such that \( D_{k,k'}(F(E_k(p_1), E'_{k'}(p_2))) = V(p_1, p_2) \). This allows distance calculations directly on ciphertexts, eliminating the need for decryption.  

### Advantageous Effects  

The invention offers the following advantages:  
1. **Efficiency**: Distance calculations are performed on ciphertexts without decryption, reducing computational overhead.  
2. **Security**: Eliminates the need for key distribution and protects against plaintext attacks.  
3. **Scalability**: Suitable for resource-constrained environments like IoT devices and cloud-based authentication.  
4. **Versatility**: Applicable to various cryptographic primitives, including one-time pads and stream ciphers.  

## DESCRIPTION OF EMBODIMENTS  

### Outline of Exemplary Embodiments of Present Disclosure  

Exemplary embodiments of the present disclosure include:  
1. **One-Time Pad Implementation**: A VE system using a one-time pad for secure authentication without key distribution.  
2. **Fiat-Shamir Adaptation**: A modified Fiat-Shamir protocol where \( x = r^2 \mod n \) and \( e = 1 \) to enable VE-compliant distance calculations.  
3. **Schnorr Adaptation**: A Schnorr-based VE system where \( x = r \cdot e \cdot p_1 \mod n \) and \( e = 1 \).  
4. **Block Cipher Exclusion**: Demonstration that ECB, CBC, and CFB modes do not belong to the VE class due to their inability to perform operations on ciphertexts.  

### Principles of Exemplary Embodiments of Present Disclosure  

#### One-Time Pad Implementation  

The one-time pad VE system operates as follows:  
1. **Registration**:  
   - A user sends plaintext \( p_1 \) to a trusted device.  
   - The device generates a key \( k \), computes \( c_1 = E_k(p_1) = p_1 \oplus k \), and sends \( c_1 \) to a server.  
2. **Verification**:  
   - The user sends plaintext \( p_2 \) to the device.  
   - The device generates a one-time key \( k' \), computes \( c_2 = E_{k'}(p_2) = p_2 \oplus k' \), and sends \( c_2 \) to the server.  
   - The server computes \( c_d = F(c_1, c_2) = c_1 \oplus c_2 \) and returns \( c_d \) to the device.  
   - The device computes \( D_{k,k'}(c_d) = c_d \oplus k \oplus k' = V(p_1, p_2) \) and verifies the result against a threshold \( s \).  

#### Fiat-Shamir Adaptation  

For \( x = r^2 \mod n \) and \( e = 1 \), the Fiat-Shamir protocol becomes VE-compliant:  
- The metric \( V(p_1, p_2) = (p_1 \oplus p_2)^2 \mod n \).  
- The mapping \( F(c_1, c_2) = c_1 \cdot c_2^{-1} \mod n \).  
- The decryption \( D_{r,x}(F(c_1, c_2)) = V(p_1, p_2) \).  

#### Schnorr Adaptation  

For \( x = r \cdot e \cdot p_1 \mod n \) and \( e = 1 \), the Schnorr protocol becomes VE-compliant:  
- The metric \( V(p_1, p_2) = g^{r(p_1 \oplus p_2)} \mod p \).  
- The mapping \( F(c_1, c_2) = c_1 \cdot c_2^{-1} \mod p \).  
- The decryption \( D_{r,x}(F(c_1, c_2)) = V(p_1, p_2) \).  

#### Block Cipher Exclusion  

ECB, CBC, and CFB modes do not belong to the VE class because their ciphertexts cannot be manipulated without decryption. However, OFB mode, functioning as a pseudorandom generator, can be adapted for VE.  

### Industrial Applicability  

The invention is industrially applicable in:  
1. **IoT Device Authentication**: Secure unlocking of smart devices via networks.  
2. **Cloud-Based Services**: Efficient and secure authentication for cloud storage and services.  
3. **Biometric Systems**: Fast and secure comparison of encrypted biometric data.  

### Experimental Results  

Implementation using a one-time pad showed:  
- **Speed**: Encryption, verification, and decryption each took less than 0.001 ms for plaintexts up to 8192 bits.  
- **Security**: Resistant to plaintext attacks due to one-time key usage.  
- **Flexibility**: Supports threshold-based authentication for noisy data like biometrics.  

### Conclusion  

The disclosed verifiable encryption system provides a secure, efficient, and scalable solution for authentication, particularly in networked environments. It eliminates key distribution, reduces computational overhead, and maintains security against common attacks.  

---  
This patent application fully describes the invention, its embodiments, and industrial applicability while adhering to the provided outline. Let me know if you need any modifications or additional details.