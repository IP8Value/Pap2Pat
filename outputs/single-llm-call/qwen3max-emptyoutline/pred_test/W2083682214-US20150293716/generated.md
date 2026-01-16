# DESCRIPTION

## STATEMENT REGARDING GOVERNMENT SPONSORED RESEARCH AND DEVELOPMENT

This invention was made with government support under Grant No. CCF-0747415 and Grant No. CCF-1217944 awarded by the National Science Foundation. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates generally to the field of non-volatile memory systems, and more specifically to error-correcting coding schemes for write-once memory (WOM) architectures, particularly as applied to flash memory technologies. The invention provides a novel method and apparatus for combining rewriting capabilities with robust error correction using polar coding techniques, enabling increased longevity, improved data reliability, and enhanced performance in solid-state storage devices. The disclosed technology addresses the fundamental challenge of maintaining data integrity across multiple write operations in memory systems where physical constraints limit the direction of state changes, while simultaneously correcting errors introduced by noise, interference, and device degradation mechanisms inherent in modern flash memory arrays.

## BACKGROUND

Non-volatile memory technologies, particularly flash memory, have become ubiquitous in modern computing and storage systems due to their high density, low power consumption, and mechanical robustness. However, flash memory cells suffer from several inherent limitations that impact their reliability and endurance. One critical limitation is that flash memory cells can only be programmed from a lower voltage state to a higher voltage state without performing an expensive and time-consuming block erase operation. This unidirectional programming constraint creates a natural write-once memory (WOM) model, where cell states can only increase over time until a block erase is performed. Block erasures are not only slow but also cause significant wear on the memory cells, ultimately limiting the total number of program/erase cycles that a flash memory device can endure before failure.

The WOM model has been extensively studied in information theory since its formal introduction by Rivest and Shamir in the early 1980s. In this model, a set of binary cells can be written multiple times, but with the constraint that once a cell is set to 1, it cannot be reset to 0 without external intervention (such as block erasure in flash memory). The primary goal of WOM coding is to maximize the total amount of information that can be stored across multiple write operations while respecting this unidirectional constraint. Various coding techniques have been developed for WOM, including linear codes, tabular codes, projective geometry-based codes, and coset coding approaches. More recently, capacity-achieving WOM codes based on polar coding have been discovered, representing a significant advancement in the field.

However, practical flash memory systems face additional challenges beyond the basic WOM constraint. Physical phenomena such as charge leakage, read/write disturbs, inter-cell interference, and manufacturing variations introduce errors that can corrupt stored data. These errors manifest as bit flips in the stored cell values, effectively creating a noisy channel between the intended stored values and the values that are actually read back. In particular, for many practical scenarios, these errors can be modeled as a binary symmetric channel (BSC) with some error probability p, where each stored bit is independently flipped with probability p during read operations.

The combination of the WOM constraint with channel noise creates a significantly more complex coding problem. While substantial progress has been made in developing high-rate WOM codes for noiseless scenarios, the development of WOM codes that can simultaneously correct a substantial number of errors has been much more limited. Existing approaches to error-correcting WOM codes have primarily focused on correcting only a small number of errors (typically 1, 2, or 3 errors), which is insufficient for practical flash memory applications where error rates can be much higher, especially as memory densities increase and cell geometries shrink.

Polar coding, introduced by Erdal Arıkan in 2009, represents a breakthrough in coding theory as the first explicit construction of codes that provably achieve the capacity of symmetric binary-input discrete memoryless channels (B-DMC) with low-complexity encoding and decoding algorithms. Polar codes work by transforming N independent copies of a given channel into N synthesized channels, some of which become nearly perfect (with capacity close to 1) and others that become nearly useless (with capacity close to 0) as N increases. The indices corresponding to the nearly useless channels form the "frozen set," where bits are fixed to known values, while the remaining indices carry information bits. The successive cancellation (SC) decoding algorithm enables efficient decoding with complexity O(N log N).

Recent work has demonstrated that polar coding techniques can be successfully applied to construct capacity-achieving WOM codes by treating the WOM constraint as a specific type of channel and applying polar coding principles to this channel model. However, the extension of these techniques to simultaneously handle both the WOM constraint and channel noise has remained an open challenge. The key insight needed to address this challenge lies in understanding the relationship between the frozen sets of the WOM channel and the noise channel (such as BSC), and leveraging this relationship to create a nested coding structure that can simultaneously satisfy both constraints.

The present invention addresses this critical gap by providing a novel coding scheme that combines rewriting capabilities with robust error correction using polar coding techniques. The invention leverages the mathematical relationship between the frozen sets of the WOM channel and the error channel to create either a nested coding structure (when the frozen set of the error channel is contained within the frozen set of the WOM channel) or an extended coding structure (for the general case) that can correct a substantial number of errors while maintaining high rewriting efficiency. This approach enables practical implementation of error-correcting WOM codes that can handle realistic error rates encountered in modern flash memory systems, thereby significantly improving both the reliability and endurance of these storage devices.

## SUMMARY

The present invention provides a novel method and apparatus for encoding and decoding data in write-once memory (WOM) systems that simultaneously supports multiple rewrites and corrects a substantial number of errors. The invention is based on a sophisticated application of polar coding techniques that leverages the mathematical relationship between the frozen sets of the WOM channel and the error channel to create an efficient coding structure.

In one aspect, the invention provides a method for encoding a message M_j during the j-th write operation in a WOM system comprising N binary cells, where each cell can only transition from state 0 to state 1. The method comprises receiving the current cell state s_j and the message M_j to be written, generating a pseudo-random dither sequence g of length N with uniformly distributed bits known to both encoder and decoder, and computing a polar code vector u of length N where information bits corresponding to the message M_j are placed in positions outside the frozen set of the WOM channel F_WOM(α_{j-1}, β_j), while bits in the intersection of the frozen sets F_WOM(α_{j-1}, β_j) ∩ F_BSC(p) are set to fixed values (typically all zeros), and bits in the difference set F_WOM(α_{j-1}, β_j) - F_BSC(p) are computed using successive cancellation encoding based on the current cell state s_j and the dither sequence g. The encoded cell state s'_j is then obtained by applying the inverse polar transformation to the vector u and XORing with the dither sequence g.

In another aspect, the invention provides a method for decoding a message M_j from noisy cell readings c_j obtained after the j-th write operation. The method comprises receiving the noisy cell state c_j, XORing with the dither sequence g to obtain a noisy polar codeword, and applying successive cancellation decoding for the polar error-correcting code designed for the binary symmetric channel BSC(p), where the frozen bits in F_BSC(p) are either known fixed values (in the nested case) or recovered from additional storage cells (in the general case).

The invention further provides a system for implementing the encoding and decoding methods, comprising a memory array of N binary cells, an encoder module configured to perform the encoding operations described above, a decoder module configured to perform the decoding operations, and a controller module for managing the write operations and parameter selection. The system may also include additional storage cells for handling the general case where the frozen set relationship does not permit a nested structure.

A key innovation of the invention is the recognition that for practical parameter ranges (relatively small error probabilities p and typical rewrite parameters), the frozen set of the BSC channel F_BSC(p) is often contained within the frozen set of the WOM channel F_WOM(α, β). This containment relationship enables a nested coding structure where the WOM code is embedded within the error-correcting code, significantly simplifying the implementation while maintaining high performance. When this containment relationship does not hold, the invention provides an extended coding structure that uses additional storage cells to handle the bits in F_BSC(p) - F_WOM(α, β), ensuring robust error correction capability in all cases.

The invention achieves sum-rates that are substantially higher than previous error-correcting WOM codes, particularly for scenarios involving multiple rewrites and realistic error rates. The analytical framework provided by the invention includes lower bounds on achievable sum-rates and demonstrates that the actual performance approaches these theoretical limits for practical code lengths. The invention is also extensible to multi-level cell (MLC) flash memories and more general noise models beyond the binary symmetric channel.

The computational complexity of both encoding and decoding operations is O(N log N), making the invention practical for real-world implementation in flash memory controllers. The invention thus provides a complete solution for high-performance, error-resilient rewriting in flash memory systems, addressing the critical need for improved reliability and endurance in modern storage devices.

## DETAILED DESCRIPTION

The present invention provides a comprehensive solution for combining rewriting capabilities with robust error correction in write-once memory (WOM) systems, particularly as applied to flash memory technologies. The invention is based on a novel application of polar coding techniques that leverages the mathematical relationship between the frozen sets of the WOM channel and the error channel to create an efficient coding structure. This detailed description provides a thorough explanation of the invention's components, operation, and implementation.

### System Architecture and Basic Model

The invention operates within a system comprising N = 2^m binary cells, where each cell can exist in one of two states: 0 or 1. The fundamental constraint of the system is that cells can only transition from state 0 to state 1, but not in the reverse direction. This constraint models the physical behavior of flash memory cells, where programming operations can only increase the threshold voltage of a cell, and decreasing the threshold voltage requires a separate block erase operation that is both time-consuming and causes wear on the memory cells.

A sequence of t messages M_1, M_2, ..., M_t is to be written into the memory cells sequentially. Each message M_j consists of M_j bits and belongs to the set {0, 1}^{M_j}. The system assumes that all cells are initialized to state 0 before the first write operation. After each write operation, the stored data is subject to noise that can corrupt the cell states. For the primary embodiment described herein, this noise is modeled as a binary symmetric channel (BSC) with error probability p, denoted as BSC(p). This means that each stored bit is independently flipped with probability p during read operations. Such errors can arise from various physical mechanisms in flash memory, including charge leakage, read/write disturbs, inter-cell interference, and manufacturing variations.

The coding system comprises t encoding functions E_j and t decoding functions D_j, where j ranges from 1 to t. Each encoding function E_j: {0, 1}^N × {0, 1}^{M_j} → {0, 1}^N takes as input the current cell state s_j and the message M_j to be written, and produces a new cell state s'_j that represents the message M_j while satisfying the constraint that s'_j ≥ s_j (i.e., no cell transitions from 1 back to 0). Each decoding function D_j: {0, 1}^N → {0, 1}^{M_j} takes as input the noisy cell state c_j (which represents the actual cell states after noise corruption) and recovers the original message M_j.

The rate of the j-th write operation is defined as R_j = M_j/N, and the sum-rate of the entire coding scheme is R_sum = Σ_{j=1}^t R_j. The goal of the invention is to maximize this sum-rate while ensuring reliable message recovery in the presence of noise.

### Polar Coding Foundation

The invention builds upon the foundation of polar coding, a breakthrough coding technique introduced by Erdal Arıkan. Polar codes work by transforming N independent copies of a given binary-input discrete memoryless channel (B-DMC) W into N synthesized channels W_N^{(i)} for i = 1, 2, ..., N. As N increases, these synthesized channels polarize such that a fraction approaching the channel capacity I(W) of the original channel become nearly perfect (with mutual information I(W_N^{(i)}) approaching 1), while the remaining fraction become nearly useless (with mutual information I(W_N^{(i)}) approaching 0).

The set of indices corresponding to the nearly useless channels forms the "frozen set" F, while the complement set carries the information bits. For error correction, the bits in the frozen set are fixed to known values (typically all zeros), while the information bits are placed in the remaining positions. The encoding process involves computing the polar transform of the combined information and frozen bits, and the decoding process uses successive cancellation (SC) decoding to recover the original bits.

The key property that makes polar codes suitable for the present invention is their ability to handle different channel models by appropriately selecting the frozen sets. The invention treats both the WOM constraint and the noise channel as distinct channel models and leverages the relationship between their respective frozen sets to create an efficient coding structure.

### Basic Code Construction with Nested Structure

The invention's core innovation lies in recognizing that for practical parameter ranges, the frozen set of the BSC channel F_BSC(p) is often contained within the frozen set of the WOM channel F_WOM(α, β). This containment relationship, expressed as F_BSC(p) ⊆ F_WOM(α, β), enables a nested coding structure that significantly simplifies implementation while maintaining high performance.

In this nested structure, the encoding process for a single write operation proceeds as follows. First, a pseudo-random dither sequence g of length N is generated, with each bit being independently and uniformly distributed. This dither sequence is known to both the encoder and decoder and serves to randomize the relationship between the physical cell states and the polar code bits.

Given the current cell state s and the message M to be written, the encoder computes a polar code vector u of length N. The bits of u are assigned as follows:
- Bits in positions outside the WOM frozen set F_WOM(α, β) carry the information bits corresponding to the message M.
- Bits in the intersection F_WOM(α, β) ∩ F_BSC(p) are set to fixed values (typically all zeros).
- Bits in the difference set F_WOM(α, β) - F_BSC(p) are computed using successive cancellation encoding based on the current cell state s and the dither sequence g.

The encoded cell state s' is then obtained by applying the inverse polar transformation to the vector u and XORing with the dither sequence g: s' = uG_N^{-1} ⊕ g, where G_N is the polar transformation matrix.

The decoding process leverages the nested structure by treating the problem as standard polar code decoding for the BSC channel. Given the noisy cell state c, the decoder first computes c ⊕ g to obtain a noisy polar codeword. This noisy codeword is then decoded using the standard successive cancellation decoding algorithm for the polar error-correcting code designed for BSC(p). The key advantage of the nested structure is that the frozen bits required for BSC decoding are already properly set during the encoding process, eliminating the need for additional storage or complex coordination between the WOM and error correction constraints.

### Extended Code Construction for General Cases

While the nested structure provides optimal performance when the frozen set containment relationship holds, the invention also provides an extended coding structure for the general case where F_BSC(p) is not necessarily a subset of F_WOM(α, β). This extended structure ensures robust error correction capability across all parameter ranges.

In the extended construction, the encoding process is modified to handle the bits in the set F_BSC(p) - F_WOM(α, β), which cannot be properly managed within the basic WOM encoding framework. After completing the basic encoding steps described above, the encoder stores the bits corresponding to F_BSC(p) - F_WOM(α, β) in additional storage cells. These additional cells are encoded using a standard error-correcting code designed specifically for the BSC(p) channel, ensuring that these critical frozen bits can be reliably recovered during decoding.

The number of additional cells required depends on the size of the set F_BSC(p) - F_WOM(α, β) and the efficiency of the error-correcting code used for these bits. While this approach requires additional storage overhead, it provides a complete solution that works for any combination of WOM and noise parameters.

The decoding process for the extended construction first recovers the bits in F_BSC(p) - F_WOM(α, β) from the additional storage cells using the appropriate error-correcting code decoder. These recovered bits are then used as known values during the main polar code decoding process, replacing the assumption that these bits are zero (which would be incorrect in the general case). The remainder of the decoding process proceeds as in the nested case, using successive cancellation decoding with the now-complete set of frozen bit values.

### Multi-Write Implementation

The single-write encoding and decoding procedures described above are naturally extended to support multiple write operations. For the j-th write operation (where j ranges from 1 to t), the parameters are updated to reflect the current state of the memory system:
- α_{j-1} represents the fraction of cells at level 0 before the j-th write
- β_j represents the fraction of cells that will be changed from level 0 to level 1 during the j-th write
- s_j represents the current cell state before the j-th write
- s'_j represents the new cell state after the j-th write
- M_j represents the message to be written during the j-th write
- c_j represents the noisy cell state after the j-th write

The encoding and decoding functions E_j and D_j are applied using the appropriate parameters for each write operation. The key insight is that the parameters α_{j-1} and β_j can be optimized to maximize the overall sum-rate R_sum across all t write operations. This optimization involves selecting the sequence β_1, β_2, ..., β_t that maximizes the sum of individual write rates while respecting the physical constraints of the memory system and the error correction requirements.

As the number of cells N approaches infinity, the achievable sum-rate approaches theoretical limits determined by the channel capacities and the frozen set relationships. For practical finite-length codes, the invention provides algorithms for selecting near-optimal parameter sequences that achieve high sum-rates while maintaining acceptable error performance.

### Performance Analysis and Optimization

The invention includes a comprehensive analytical framework for evaluating and optimizing performance. The key theoretical result is that when F_BSC(p) ⊆ F_WOM(α, β), the nested coding structure achieves a sum-rate that is the difference between the WOM channel capacity and the BSC channel capacity loss. Specifically, the achievable rate for a single write is approximately I(WOM(α, β)) - I(BSC(p)), where I(·) denotes the channel capacity.

For the general case, the achievable rate is reduced by the overhead required to store the additional frozen bits, but remains competitive with specialized solutions. The invention provides lower bounds on the achievable sum-rates and demonstrates through experimental evaluation that these bounds are approached by practical implementations.

The optimization process involves searching over the parameter space of β_1, β_2, ..., β_t to find the combination that maximizes the sum-rate R_sum. This search can be performed using dynamic programming or other optimization techniques, taking into account the dependencies between successive write operations (since α_j depends on the previous β values).

Experimental results demonstrate that the invention achieves substantial improvements over existing error-correcting WOM codes, particularly for scenarios involving multiple rewrites and realistic error rates. For example, with N = 8192 cells, target block error rate of 10^-5, and rate loss ΔR = 0.025, the invention achieves sum-rates that increase with the number of rewrites t and remain robust across a range of error probabilities p.

### Practical Implementation Considerations

The invention is designed for practical implementation in flash memory controllers and other storage systems. Both encoding and decoding operations have computational complexity O(N log N), making them suitable for real-time operation in modern storage devices. The memory requirements are also reasonable, with the main storage overhead being the additional cells required in the general case (which can be minimized through careful parameter selection).

The dither sequence g can be generated using a cryptographically secure pseudo-random number generator with a shared seed between encoder and decoder, eliminating the need to store the entire sequence. Alternatively, the dither sequence can be derived from system parameters or stored in a small dedicated memory area.

The invention is also compatible with existing flash memory management techniques, including wear leveling, bad block management, and error correction coding. The polar coding framework can be integrated with these techniques to provide comprehensive data protection and management capabilities.

### Extensions and Variations

The basic invention can be extended in several important directions. First, the invention can be adapted to multi-level cell (MLC) flash memories, where each cell can store more than one bit by using multiple voltage levels. This requires extending the WOM model to handle multiple levels and adapting the polar coding techniques accordingly.

Second, the invention can handle more general noise models beyond the binary symmetric channel. For example, asymmetric noise models, burst error models, or channel models that capture the specific characteristics of flash memory physics can be incorporated by appropriately modifying the frozen set selection and decoding algorithms.

Third, the invention can be combined with other advanced coding techniques, such as list decoding, belief propagation decoding, or concatenated coding schemes, to further improve performance or reduce complexity.

Finally, the invention can be applied to other types of memory systems that exhibit write-once characteristics, including phase-change memory (PCM), resistive RAM (ReRAM), and other emerging non-volatile memory technologies.

### VII. Additional Example Embodiments

In addition to the primary embodiment described above, the invention encompasses several alternative implementations that may be advantageous in specific application scenarios. One such alternative embodiment employs a hybrid coding approach that combines the polar coding framework with traditional algebraic coding techniques. In this hybrid approach, the polar code handles the majority of the error correction and rewriting functionality, while a secondary algebraic code (such as a BCH code or Reed-Solomon code) provides additional protection for critical metadata or handles residual errors that escape the polar code decoding.

Another alternative embodiment focuses on reducing the computational complexity of the decoding process by employing simplified successive cancellation decoding algorithms or by using pre-computed lookup tables for frequently encountered patterns. This approach is particularly valuable in resource-constrained embedded systems where processing power and memory are limited.

A third alternative embodiment implements adaptive parameter selection, where the encoding parameters (including the choice between nested and extended structures) are dynamically adjusted based on real-time monitoring of error rates and memory usage patterns. This adaptive approach can optimize performance across varying operating conditions and extend the useful life of the memory system.

A fourth alternative embodiment extends the invention to handle correlated errors, which are common in flash memory systems due to inter-cell interference and other physical coupling mechanisms. This extension involves modifying the polar code construction to account for error correlations and may employ techniques such as interleaving or specialized channel modeling.

A fifth alternative embodiment integrates the invention with advanced memory management techniques, such as dynamic wear leveling and garbage collection optimization. In this integrated approach, the coding parameters are coordinated with the memory management algorithms to achieve optimal overall system performance, balancing factors such as write amplification, endurance, and data reliability.

These alternative embodiments demonstrate the flexibility and adaptability of the core invention, enabling its application across a wide range of storage systems and operating conditions while maintaining the fundamental advantages of combining rewriting capabilities with robust error correction through polar coding techniques.