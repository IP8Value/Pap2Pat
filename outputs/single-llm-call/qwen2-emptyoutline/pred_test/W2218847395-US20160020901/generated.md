# DESCRIPTION

## BACKGROUND

The field of symmetric key cryptography encompasses methods and systems designed to enable secure communication over potentially insecure channels. Central to this field are the concepts of encryption and authentication. Encryption ensures that the content of a message remains confidential, while authentication guarantees the integrity of the message and confirms its origin. Existing authenticated encryption algorithms, although numerous, often fall short in terms of performance, security, or ease of use. Some algorithms require multiple passes to encrypt and authenticate data, leading to inefficiencies. Others have been found to be insecure or difficult to implement correctly. Additionally, many algorithms necessitate the use of two unique and unrelated keys, complicating key management.

In response to these challenges, there is a pressing need for a customizable authenticated encryption algorithm that can meet the stringent security requirements of government and military applications. Such algorithms are often kept out of the public domain to deter academic scrutiny and potential publication of vulnerabilities. The present invention addresses this need by providing a customizable authenticated encryption algorithm based on the duplex construction, which utilizes an iterated permutation with 16 × 16 bijective S-boxes. These large S-boxes introduce significant non-linearity and algebraic complexity, enhancing the algorithm's security without a substantial increase in hardware requirements.

## SUMMARY OF THE INVENTION

The present invention relates to a customizable authenticated encryption algorithm that combines the duplex construction with a permutation function. The algorithm is designed to provide robust security, efficient performance, and ease of customization. Key features of the invention include:

1. **Duplex Construction**: The algorithm leverages the duplex construction, a cryptographic primitive that allows for simultaneous absorption and squeezing of data. This construction is particularly well-suited for authenticated encryption, as it can handle both encryption and authentication in a single pass.

2. **Large S-boxes**: The permutation function at the core of the algorithm employs 16 × 16 bijective S-boxes. These large S-boxes significantly enhance the non-linearity and algebraic complexity of the permutation, making the algorithm more resistant to various cryptographic attacks.

3. **Customizability**: The algorithm is highly customizable, allowing different users or applications to generate unique, proprietary versions of the algorithm. Customizations can be made to the initial state, S-boxes, bitwise permutations, mixers, and round constants, ensuring that each instantiation is distinct and secure.

4. **Security Margins**: The number of rounds in the permutation is carefully chosen to provide a strong security margin against linear and differential cryptanalysis. The algorithm is also designed to resist algebraic attacks due to the high algebraic complexity introduced by the large S-boxes.

5. **Hardware Efficiency**: The permutation function is optimized for hardware implementation, with efficient S-box designs and bitwise permutations that minimize resource usage. This makes the algorithm suitable for deployment in resource-constrained environments.

## DETAILED DESCRIPTION

### Duplex Construction

The duplex construction is a cryptographic primitive that extends the sponge construction by maintaining its internal state between calls and eliminating the clear separation between the absorbing and squeezing phases. In the duplex construction, inputs and outputs are processed simultaneously, making it ideal for authenticated encryption.

#### Parameters

- **State Size (b)**: The total state size is 512 bits.
- **Rate (r)**: The rate is 128 bits, which is the portion of the state that is accessible externally.
- **Capacity (c)**: The capacity is 384 bits, which is the hidden portion of the state.

### Permutation Function

The permutation function \( f \) is a critical component of the algorithm, ensuring the security and efficiency of the duplex construction. The permutation consists of multiple rounds, each applying a series of operations to the state.

#### Substitution Step

The substitution step uses 32 identical, bijective 16 × 16 S-boxes. These S-boxes are the primary source of confusion and non-linearity in the permutation. The S-boxes are based on multiplicative inversion in the finite field \( \text{GF}(2^{16}) \) followed by an affine transformation. This design ensures efficient hardware implementation while maintaining high cryptographic strength.

#### Bitwise Permutation Step

The bitwise permutation step is designed to provide long-range diffusion across the entire state. The permutation is a derangement with no fixed points and has a high order, ensuring that it does not repeat within the number of rounds. The permutation is defined by an affine function, making it easy to implement in hardware.

#### Mix Step

The mix step provides local diffusion and increases the linear and differential branch numbers of a round from two to three. The mixer is based on multiplication by a 2 × 2 matrix in \( \text{GF}(2^{16}) \) modulo an irreducible polynomial. The matrix is chosen to be symmetric, ensuring that the differential and linear branch numbers are equal.

#### Add Round Constant Step

The add round constant step disrupts symmetry and prevents slide attacks. Each round constant is a 512-bit value derived from the ASCII representation of the round number and the SHA-3 hash function.

### Number of Rounds

The number of rounds is determined to provide resistance against linear and differential cryptanalysis. For a 128-bit key, the algorithm uses 10 rounds, and for a 256-bit key, it uses 16 rounds. This number of rounds ensures a strong security margin while maintaining efficient performance.

### Customization

The algorithm is highly customizable, allowing different users or applications to generate unique, proprietary versions. Possible customizations include:

1. **State Initialization**: The initial value of the inner state can be modified to any 384-bit value, providing a simple method of customization.
2. **S-boxes**: Users can choose from a variety of cryptographically secure 16-bit S-boxes, including those based on AES-like designs.
3. **Bitwise Permutations**: Users can select from a list of suitable bitwise permutations that satisfy the required constraints.
4. **Mixers**: Users can choose from a range of 2 × 2 matrices that meet the specified criteria for invertibility and branch numbers.
5. **Round Constants**: Users can generate their own unique set of round constants, ensuring that each instantiation is distinct.

### Cryptanalysis

The security of the algorithm is assessed through resistance to differential, linear, and algebraic attacks.

#### Differential Cryptanalysis

The maximum differential probability of the S-box is \( 2^{-14} \). The differential branch number of the mixer is three, ensuring that at least three S-boxes are differentially active between two rounds. The complexity of a differential attack exceeds the complexity of a brute-force attack at six rounds, leading to a requirement of 10 rounds for a 128-bit key and 16 rounds for a 256-bit key.

#### Linear Cryptanalysis

The linear branch number of the mixer is also three, ensuring resistance to linear attacks. The worst-case linear bias of the S-boxes is \( 2^{-8} \), and the complexity of a linear attack exceeds the complexity of a brute-force search at six rounds.

#### Algebraic Attacks

The high algebraic complexity introduced by the large S-boxes makes the algorithm resistant to algebraic attacks. The permutation is designed to be highly complex, making it impractical to solve the resulting algebraic system.

### Conclusion

The present invention provides a customizable authenticated encryption algorithm based on the duplex construction and a permutation function with 16 × 16 S-boxes. The algorithm offers robust security, efficient performance, and ease of customization, making it suitable for a wide range of applications, including government and military use. Further cryptanalysis and hardware implementation studies are recommended to fully validate the algorithm's security and efficiency.