# DESCRIPTION

## STATEMENT REGARDING GOVERNMENT SPONSORED RESEARCH AND DEVELOPMENT

The invention described herein was made in part under a grant from the National Science Foundation (NSF) CAREER Award CCF-0747415, NSF Grant CCF-1217944, a grant from Intellectual Ventures, the Israel Science Foundation (ISF) grant 480/08, and the Binational Science Foundation (BSF) grant 2010075. The government may have certain rights in the invention.

## TECHNICAL FIELD

The present invention relates generally to the field of data storage systems, particularly to methods and systems for coding in flash memories. More specifically, the invention pertains to a novel coding scheme that combines rewriting capabilities with error correction, thereby enhancing the longevity, speed, and power efficiency of flash memories.

## BACKGROUND

Flash memories are widely used in various data storage applications due to their non-volatility, high density, and low cost. However, these memories face significant challenges, including limited endurance and data retention issues. One approach to mitigate these challenges is through the use of write-once memory (WOM) codes, which allow data to be rewritten multiple times without requiring full block erasures. This reduces wear on the memory cells and improves overall performance.

Several techniques have been developed for designing WOM codes, including linear codes, tabular codes, codes based on projective geometry, and coset coding. Recent advancements have led to the discovery of WOM codes that achieve capacity, such as those based on polar coding. However, most existing WOM codes do not incorporate error correction, which is crucial for reliable data storage in practical applications.

Error correction in flash memories is typically handled by separate error-correcting codes (ECCs), which can correct a limited number of errors. Combining rewriting capabilities with robust error correction is essential for extending the lifespan and reliability of flash memories. Despite the importance of this combination, there has been limited progress in developing efficient codes that achieve both objectives simultaneously.

## SUMMARY

The present invention provides a novel coding scheme that integrates rewriting capabilities with error correction in flash memories. The scheme utilizes polar coding to construct a code that supports multiple rewrites and can correct a substantial number of errors. The code construction is based on the frozen sets corresponding to the WOM channel and the error channel, including their common degrading and common upgrading channels.

Key features of the invention include:
1. **Nested Structure**: For relatively small error probabilities, the frozen set for the binary symmetric channel (BSC) is often contained within the frozen set for the WOM channel, enabling a nested code structure.
2. **High Sum-Rate**: The code achieves high sum-rates, which are further optimized for various parameters.
3. **Scalability**: The code can be extended to multi-level cells (MLC) and more general noise models, making it versatile for different flash memory configurations.

The invention addresses the limitations of existing WOM codes by providing a robust solution that enhances the longevity, speed, and power efficiency of flash memories while ensuring data integrity through error correction.

## DETAILED DESCRIPTION

### I. Introduction

The invention pertains to a novel coding scheme for flash memories that combines rewriting capabilities with error correction. Flash memories are prone to wear and tear, leading to reduced lifespan and performance degradation. Write-once memory (WOM) codes have been proposed to mitigate these issues by allowing data to be rewritten multiple times without requiring full block erasures. However, most existing WOM codes lack error correction, which is essential for reliable data storage.

The present invention addresses this gap by introducing a coding scheme that uses polar coding to achieve both rewriting and error correction. The code construction leverages the frozen sets of the WOM channel and the error channel, including their common degrading and common upgrading channels. This results in a nested code structure for small error probabilities, which simplifies the implementation and improves performance.

### II. Basic Model

#### A. The Model for Rewriting

Consider a flash memory with \( N = 2^m \) cells, where each cell can store data at two levels: 0 and 1. The cell levels can only increase (from 0 to 1) but not decrease. A sequence of \( t \) messages \( M_1, M_2, \ldots, M_t \) will be written into the cells. Each message \( M_j \) has \( M_j \) bits, and the cells are initially at level 0 before the first write.

The rewriting process involves encoding functions \( E_j \) and decoding functions \( D_j \). For the \( j \)-th write, the encoding function \( E_j : \{0, 1\}^N \times \{0, 1\}^{M_j} \to \{0, 1\}^N \) changes the cell levels from \( s_j = (s_{1,j}, s_{2,j}, \ldots, s_{N,j}) \) to \( s_j' = (s_{1,j}', s_{2,j}', \ldots, s_{N,j}') \) given the initial cell state \( s_j \) and the message to store \( M_j \). The decoding function \( D_j : \{0, 1\}^N \to \{0, 1\}^{M_j} \) recovers the message \( M_j \) given the noisy cell state \( c_j = (c_{1,j}, c_{2,j}, \ldots, c_{N,j}) \).

Noise in the cell levels is modeled as a binary symmetric channel (BSC) with error probability \( p \). The error \( c_{i,j} \oplus s_{i,j} \) represents the error in the \( i \)-th cell caused by the noisy channel BSC(p).

#### B. Polar Codes

Polar codes, introduced by Arıkan, are linear block error-correcting codes that achieve the channel capacity of symmetric binary-input discrete memoryless channels (B-DMC). The encoder of a polar code transforms the input bits into a codeword using a generator matrix. The channels are polarized such that for large \( N \), the fraction of indices \( i \) for which the mutual information \( I(W_N^{(i)}) \) is nearly 1 approaches the capacity of the B-DMC, while the values of \( I(W_N^{(i)}) \) for the remaining indices \( i \) are nearly 0. The latter set of indices is called the frozen set.

For error correction, the bits in the frozen set take fixed values, and the other bits are used as information bits. A successive cancellation (SC) decoding algorithm achieves diminishing block error probability as \( N \) increases.

### III. Code Construction

#### A. Basic Code Construction with a Nested Structure

##### 1. Basic Concepts

Consider a single rewrite step. Let \( s = (s_1, s_2, \ldots, s_N) \) and \( s' = (s_1', s_2', \ldots, s_N') \) be the cell levels right before and after the rewrite, respectively. Let \( g = (g_1, g_2, \ldots, g_N) \) be a pseudo-random bit sequence with i.i.d. bits that are uniformly distributed. The value of \( g \) is known to both the encoder and the decoder, and \( g \) is called a dither.

The WOM channel for this rewrite is denoted by \( \text{WOM}(\alpha, \epsilon) \), where \( \alpha \in [0, 1] \) and \( \epsilon \in [0, 0.5] \) are given parameters. Here, \( \alpha = \frac{1}{N} \sum_{i=1}^N s_i \) represents the fraction of cells at level 0 before the rewrite, and \( \epsilon \) represents the fraction of cells that are changed from level 0 to level 1 by the rewrite. Let \( F_{\text{WOM}(\alpha, \epsilon)} \subseteq \{1, 2, \ldots, N\} \) be the frozen set of the polar code corresponding to this channel.

In this subsection, we assume \( F_{\text{BSC}(p)} \subseteq F_{\text{WOM}(\alpha, \epsilon)} \), which leads to a nested code structure. For any message \( M \in \{0, 1\}^M \), the set of cell values \( V_M \subseteq \{0, 1\}^N \) that represent the message \( M \) is a linear subspace of a linear error-correcting code (ECC) for the noisy channel BSC(p), and \( \{V_M | M \in \{0, 1\}^M\} \) form a partition of the ECC's codewords.

##### 2. The Encoder

Let \( E : \{0, 1\}^N \times \{0, 1\}^M \to \{0, 1\}^N \) be the encoder for this rewrite. Given the current cell state \( s \) and the message to write \( M \in \{0, 1\}^M \), the encoder finds a new cell state \( s' = E(s, M) \) that represents \( M \) and is above \( s \) (i.e., cell levels only increase).

The encoding process is similar to the WOM code encoder in [2], but with differences in how to assign bits to \( F_{\text{WOM}(\alpha, \epsilon)} \). The encoding function is presented in Algorithm 1. Here, \( y \) and \( u \) are vectors of length \( N \); \( u_{F_{\text{WOM}(\alpha, \epsilon)} - F_{\text{BSC}(p)}} \) are the bits in the frozen set \( F_{\text{WOM}(\alpha, \epsilon)} \) but not in \( F_{\text{BSC}(p)} \); \( u_{F_{\text{BSC}(p)}} \) are the bits in \( F_{\text{BSC}(p)} \); and \( u_{F_{\text{WOM}(\alpha, \epsilon)} - F_{\text{BSC}(p)}} \) are set to 0.

**Algorithm 1: The encoding function \( s' = E(s, M) \)**

1. Compute \( v = s \oplus g \).
2. Initialize \( u \) as a vector of length \( N \).
3. Set \( u_{F_{\text{BSC}(p)}} = 0 \).
4. Set \( u_{F_{\text{WOM}(\alpha, \epsilon)} - F_{\text{BSC}(p)}} = 0 \).
5. Set \( u_{\{1, 2, \ldots, N\} - F_{\text{WOM}(\alpha, \epsilon)}} = M \).
6. Compute \( y = u G_N \).
7. Compute \( s' = y \oplus g \).

##### 3. The Decoder

Let \( D : \{0, 1\}^N \to \{0, 1\}^M \) be the decoder for this rewrite. Given the noisy cell state \( c \), the decoder recovers the message \( M \) as \( M = D(c) \).

The decoding process is similar to the polar error-correcting code. The decoder is presented in Algorithm 2.

**Algorithm 2: The decoding function \( M = D(c) \)**

1. Compute \( c' = c \oplus g \).
2. Decode \( c' \) using the decoding algorithm of the polar error-correcting code, where the bits in the frozen set \( F_{\text{BSC}(p)} \) are set to 0.
3. Extract the message \( M \) from the decoded bits.

Both the encoding and decoding algorithms have a time complexity of \( O(N \log N) \).

##### 4. Nested Code for \( t \) Writes

The encoder and decoder for one rewrite can be naturally extended to a \( t \)-write error-correcting WOM code. For \( j = 1, 2, \ldots, t \), for the \( j \)-th write, replace \( \alpha, \epsilon, s, s', v, v', M, M', E, D, c, M', v' \) by \( \alpha_{j-1}, \epsilon_j, s_j, s_j', v_j, v_j', M_j, M_j', E_j, D_j, c_j, M_j', v_j' \), respectively, and apply the above encoder and decoder.

#### B. Extended Code Construction

The basic code construction assumes \( F_{\text{BSC}(p)} \subseteq F_{\text{WOM}(\alpha, \epsilon)} \). However, in the general case, \( F_{\text{BSC}(p)} \) is not necessarily a subset of \( F_{\text{WOM}(\alpha, \epsilon)} \).

To handle this, the encoder in Algorithm 1 is revised as follows. After all the steps in the algorithm, the bits in \( u_{F_{\text{BSC}(p)} - F_{\text{WOM}(\alpha, \epsilon)}} \) are stored using \( N_{\text{additional}, j} \) additional cells (for the \( j \)-th write). These additional cells are assumed to store the bits using an error-correcting code designed for the noisy channel BSC(p).

The decoder in Algorithm 2 is revised as follows. First, recover the bits in \( u_{F_{\text{BSC}(p)} - F_{\text{WOM}(\alpha, \epsilon)}} \) using the decoding algorithm of the ECC for the \( N_{\text{additional}, j} \) additional cells. Then, carry out all the steps in Algorithm 2, except that the bits in \( F_{\text{BSC}(p)} - F_{\text{WOM}(\alpha, \epsilon)} \) are known to the decoder as the recovered values instead of 0s.

### IV. Code Analysis for BSC

#### A. Correctness of the Code

The correctness of the code is proven by showing that the encoder and decoder work as intended. The encoder in Algorithm 1 successfully rewrites data in the same way as the code in [2], with the exception that the bits in \( F_{\text{WOM}(\alpha, \epsilon)} \cap F_{\text{BSC}(p)} \) are set to 0. The decoder in Algorithm 2 recovers the cell values from noise in the same way as the standard polar ECC. The stored message \( M \) is extracted from the decoded values.

Physical noise acts on cell levels \( s = (s_1, s_2, \ldots, s_N) \), but the ECC in the construction is for cell values \( v = (s_1 \oplus g_1, s_2 \oplus g_2, \ldots, s_N \oplus g_N) \). The dither \( g \) has independent and uniformly distributed elements, so when the noisy channel for \( s \) is BSC(p), the corresponding noisy channel for \( v \) is also BSC(p).

#### B. Achievable Sum-Rates

The performance of the code is analyzed by searching for the achievable sum-rates. Given the error probability \( p \), the parameters \( \epsilon_1, \epsilon_2, \ldots, \epsilon_t \) are optimized to maximize the sum-rate \( R_{\text{sum}} \).

Experimental results for \( t \)-write error-correcting WOM codes are shown in Figure 8. The x-axis is \( p \), and the y-axis is the maximum sum-rate found in the algorithmic search. The achievable sum-rate increases with the number of rewrites \( t \). In most cases, the achievable rate of the general code is very close to that of a nested code, indicating that the nested code performs well for this parameter range.

### VII. Additional Example Embodiments

#### A. Multi-Level Cells (MLC)

The code can be extended to multi-level cells (MLC) by considering more than two levels per cell. The basic principles of the code construction remain the same, but the encoding and decoding processes need to be adapted to handle multiple levels. The frozen sets for the MLC WOM channel and the error channel are determined, and the nested structure is maintained if possible.

#### B. General Noise Models

The code can be further extended to more general noise models beyond the binary symmetric channel (BSC). For example, the code can be adapted to handle asymmetric noise, where the error probabilities for different types of errors (e.g., 0 to 1 vs. 1 to 0) are different. The frozen sets for the general noise models are determined, and the encoding and decoding processes are adjusted accordingly.

#### C. Practical Implementation

In practical implementations, the code can be optimized for specific flash memory architectures and error profiles. The parameters \( \alpha \) and \( \epsilon \) can be fine-tuned based on empirical data to achieve the best performance. Additionally, the code can be integrated with existing error-correcting codes and data management schemes to enhance the overall reliability and efficiency of the flash memory system.

### Conclusion

The present invention provides a novel coding scheme that combines rewriting capabilities with error correction in flash memories. By leveraging polar coding and the nested structure of the code, the invention achieves high sum-rates and robust error correction, thereby enhancing the longevity, speed, and power efficiency of flash memories. The code can be extended to multi-level cells and more general noise models, making it versatile for various applications.