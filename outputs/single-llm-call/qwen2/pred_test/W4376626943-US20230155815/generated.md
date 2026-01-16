# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and system for secure integer comparison using multi-party computation (MPC). Specifically, the invention provides a novel protocol for comparing two private integers without revealing any information other than the result of the comparison. The protocol is designed to be efficient and secure, leveraging homomorphic encryption (HE) techniques, and is particularly useful in privacy-preserving applications such as secure machine learning, benchmarking, and auctions.

## RELATED ART

Multi-party computation (MPC) is a cryptographic technique that enables multiple parties to jointly compute a function on their private inputs without revealing any information beyond the function's output. One of the fundamental problems in MPC is secure integer comparison, which is essential in various applications such as decision trees in machine learning, secure benchmarking, and secure auctions.

Several existing protocols for secure integer comparison have been proposed, including the seminal work by Yao [53] and the DGK protocol [18]. Yao's protocol introduced the concept of secure computation through garbled circuits, while the DGK protocol utilizes additively homomorphic encryption (AHE) to achieve secure comparison. Subsequent works have aimed to optimize and extend these protocols, often focusing on reducing computational and communication costs.

However, these existing protocols have limitations in terms of efficiency and scalability. For instance, the DGK protocol and its variants require multiple rounds of interaction and can be computationally intensive, especially for large integers. Recent work has explored the use of fully homomorphic encryption (FHE) to achieve more efficient and non-interactive solutions, but these approaches often suffer from high computational overhead due to the complexity of FHE operations.

## SUMMARY

The present invention addresses the limitations of existing secure integer comparison protocols by providing a novel protocol that is both efficient and secure. The protocol is based on the evaluation of a binary tree structure using homomorphic encryption (HE) and is designed to be non-interactive, meaning that the parties exchange messages only once.

The key contributions of the invention are as follows:

1. **Binary Tree Representation**: The server constructs a binary tree that represents the server's input integer \( y \). Each path in the tree corresponds to a possible integer value, and the leaves of the tree are labeled to indicate whether the path represents an integer greater than or equal to \( y \).

2. **Homomorphic Encryption**: The client encrypts its input integer \( x \) using a homomorphic encryption scheme and sends the encrypted input to the server. The server evaluates the binary tree using the encrypted input, performing homomorphic operations to compute the comparison result.

3. **Efficient Evaluation**: The protocol includes optimizations to reduce the computational complexity of the tree evaluation. For example, the use of pre-computed dependency lists for multiplication in the FHE case ensures that the multiplicative depth remains logarithmic, which is crucial for the efficiency of leveled FHE schemes.

4. **Security and Correctness**: The protocol is proven to be secure in the semi-honest model, ensuring that the parties learn only the comparison result and no additional information. The correctness of the protocol is guaranteed by the properties of the homomorphic encryption scheme and the structure of the binary tree.

5. **Extensions and Variants**: The protocol can be extended to handle various scenarios, including the case where both inputs are encrypted, the case where the comparison result is shared between the parties, and the case where the comparison is for "less than" instead of "greater than or equal to."

## DETAILED DESCRIPTION

### Introduction

Secure integer comparison is a fundamental building block in many privacy-preserving applications. The present invention provides a novel protocol for secure integer comparison that leverages homomorphic encryption (HE) and a binary tree structure to achieve efficient and secure computation. The protocol is designed to be non-interactive, meaning that the parties exchange messages only once, and is suitable for a wide range of applications.

### Technical Background

#### Homomorphic Encryption (HE)

Homomorphic encryption (HE) allows computations to be performed on encrypted data without the need to decrypt the data first. An HE scheme consists of the following algorithms:

- **Key Generation**: \( \text{pk}, \text{sk}, \text{ek} \leftarrow \text{KGen}(\lambda) \)
  - This probabilistic algorithm takes a security parameter \( \lambda \) and outputs public, private, and evaluation keys \( \text{pk} \), \( \text{sk} \), and \( \text{ek} \).

- **Encryption**: \( c \leftarrow \text{Enc}(\text{pk}, m) \)
  - This algorithm takes the public key \( \text{pk} \) and a message \( m \) and outputs a ciphertext \( c \).

- **Decryption**: \( m \leftarrow \text{Dec}(\text{sk}, c) \)
  - This algorithm takes the private key \( \text{sk} \) and a ciphertext \( c \) and outputs the plaintext \( m \).

- **Evaluation**: \( c' \leftarrow \text{Eval}(\text{ek}, f, c_1, \ldots, c_n) \)
  - This algorithm takes the evaluation key \( \text{ek} \), a function \( f \), and ciphertexts \( c_1, \ldots, c_n \) and outputs a ciphertext \( c' \) that encrypts \( f(m_1, \ldots, m_n) \).

#### Types of Homomorphic Encryption

- **Additively Homomorphic Encryption (AHE)**: Supports only addition operations on ciphertexts.
- **Fully Homomorphic Encryption (FHE)**: Supports both addition and multiplication operations on ciphertexts, allowing for the evaluation of arbitrary functions.

### Protocol Description

#### Intuition

The core idea of the protocol is to represent the comparison problem as a classification problem using a binary tree. The server constructs a binary tree where each path represents a possible integer value. The leaves of the tree are labeled to indicate whether the path represents an integer greater than or equal to the server's input \( y \). The client encrypts its input \( x \) and sends the encrypted input to the server. The server evaluates the tree using the encrypted input, performing homomorphic operations to compute the comparison result.

#### Data Structure

The data structure used in the protocol is a binary tree consisting of inner nodes and terminal nodes. Each inner node has two child nodes, and terminal nodes have no child nodes. The tree is constructed as follows:

- **Root Node**: The root node has no parent and represents the most significant bit of the server's input \( y \).
- **Inner Nodes**: Each inner node has two child nodes, representing the next bit in the binary representation of \( y \).
- **Leaf Nodes**: Leaf nodes are labeled with 0 or 1, indicating whether the path represents an integer greater than or equal to \( y \).

Each node in the tree has the following attributes:

- **parent**: Pointer to the parent node.
- **left**: Pointer to the left child node.
- **right**: Pointer to the right child node.
- **lEdge**: Bit representing the edge label to the left child node.
- **rEdge**: Bit representing the edge label to the right child node.
- **cLabel**: Value representing the node label (0 or 1 for terminal nodes).
- **cost**: Integer representing the cost on the path from the root.

#### Algorithms

##### Initialization

1. **Key Generation**: The client generates an appropriate triple \( (\text{pk}, \text{sk}, \text{ek}) \) of public, private, and evaluation keys for an HE scheme.
2. **Key Distribution**: The client sends \( (\text{pk}, \text{ek}) \) to the server.
3. **Input Encryption**: The client encrypts its input \( x \) using the public key \( \text{pk} \) and sends the encrypted input to the server.

##### Creating the Binary Tree

The server constructs the binary tree representing the server's input \( y \) as follows:

1. **Tree Construction**: The server creates a binary tree where each path represents a possible integer value. The leaves of the tree are labeled to indicate whether the path represents an integer greater than or equal to \( y \).
2. **Labeling**: The server labels the leaves of the tree such that:
   - Paths representing integers greater than or equal to \( y \) are labeled with 1.
   - Paths representing integers less than \( y \) are labeled with 0.
3. **Pruning**: The server prunes the tree by removing subtrees that are labeled with the same bit, reducing the number of nodes and improving efficiency.

##### Evaluating the Tree

The server evaluates the binary tree using the encrypted input \( x \) as follows:

1. **Node Evaluation**: For each inner node, the server computes the homomorphic comparison of the client's input bit \( x_i \) with the edge labels of the node. This is done using homomorphic operations (XOR or XNOR for AHE, subtraction for FHE).
2. **Path Evaluation**: The server aggregates the comparison results along the paths from the root to the leaf nodes. For FHE, this is done using homomorphic multiplication. For AHE, this is done using homomorphic addition.
3. **Leaf Evaluation**: The server aggregates the results at the leaf nodes to compute the final comparison result. For FHE, the server sums the results of all paths. For AHE, the server randomizes the encrypted costs at the leaves, permutes the list, and sends it to the client.

##### Decrypting the Result

The client decrypts the result of the evaluation to learn the final comparison result:

1. **Result Decryption**: For FHE, the client decrypts a single encrypted bit indicating the comparison result. For AHE, the client decrypts a list of ciphertexts and checks for the presence of an encryption of 0 to determine the comparison result.

### Extensions and Variants

#### Handling Encrypted Inputs

The protocol can be extended to handle the case where both inputs are encrypted. In this scenario, the server evaluates the tree using the encrypted inputs with the help of the client (or another server) that has the decryption key. The server constructs a normal comparison tree (normal cmp-tree) that is independent of the actual inputs, ensuring that the tree structure does not leak information about the inputs.

#### Shared Output Bit

The protocol can be modified to share the comparison bit between the client and the server. The server randomly chooses a bit \( b_s \) and computes the comparison bit \( b_c \) such that \( b = b_c \oplus b_s \). The server sends \( b_c \) to the client, and the client can reconstruct the comparison bit \( b \) using \( b = b_c \oplus b_s \).

#### Less Than (LT) Comparison

The protocol can be adapted to compute the "less than" (LT) comparison by using an inverse normal cmp-tree. The inverse normal cmp-tree is constructed such that the rightmost path represents the server's input \( y \), and the leaves are labeled to indicate whether the path represents an integer less than \( y \).

### Security and Correctness

The protocol is proven to be secure in the semi-honest model, ensuring that the parties learn only the comparison result and no additional information. The correctness of the protocol is guaranteed by the properties of the homomorphic encryption scheme and the structure of the binary tree.

### Applications

The secure integer comparison protocol has a wide range of applications, including:

- **Machine Learning**: Securely evaluating decision trees and other classifiers.
- **Benchmarking**: Securely comparing key performance indicators.
- **Auctions**: Securely determining the highest bid.
- **Biometrics**: Securely matching biometric data.

### Conclusion

The present invention provides a novel and efficient protocol for secure integer comparison using homomorphic encryption and a binary tree structure. The protocol is non-interactive, secure, and can be extended to handle various scenarios, making it suitable for a wide range of privacy-preserving applications.