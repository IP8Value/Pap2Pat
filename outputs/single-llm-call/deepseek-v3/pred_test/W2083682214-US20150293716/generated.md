Here is the patent application following your outline and research paper:

# DESCRIPTION

## STATEMENT REGARDING GOVERNMENT SPONSORED RESEARCH AND DEVELOPMENT  

The invention described in this patent application was made with government support under NSF CAREER Award CCF-0747415, NSF Grant CCF-1217944, and ISF grant 480/08. The government has certain rights in the invention.

## TECHNICAL FIELD  

The present invention relates generally to data storage systems and more particularly to error-correcting write-once memory (WOM) codes for non-volatile memories such as flash memory devices. The invention specifically addresses the technical challenge of jointly implementing rewriting and error correction in memory systems through novel coding schemes based on polar codes.

## BACKGROUND  

Coding for rewriting has emerged as an important technology for enhancing the performance of non-volatile memories such as flash memory devices. The fundamental model for rewriting is the write-once memory (WOM) model where binary memory cells can only transition from a 0 state to a 1 state when data is rewritten. This constraint models the physical limitations of flash memory where cell levels can only increase until an expensive block erasure resets them. By delaying erasures through efficient rewriting codes, memory longevity, speed and power efficiency can be substantially improved.

Various techniques have been developed for designing WOM codes including linear codes, tabular codes, projective geometry-based codes, and coset coding approaches. Recent advances have produced WOM codes that achieve capacity, including notable constructions based on polar coding techniques. Polar codes themselves represent a significant breakthrough in coding theory as the first explicitly constructed codes proven to achieve capacity for symmetric binary-input discrete memoryless channels.

While WOM coding has seen extensive development, the integration of error correction capabilities with rewriting functionality remains underdeveloped. Existing error-correcting WOM codes are typically limited to correcting only a small number of errors. For practical deployment in memory systems, there exists a critical need for WOM codes capable of both multiple rewrites and substantial error correction capacity.

The WOM channel model considers N binary cells that can only transition from 0 to 1 states. A sequence of t messages is written to these cells, with each write operation updating the stored information without requiring retention of previous messages. Cell levels may be corrupted by noise modeled as a binary symmetric channel (BSC) with crossover probability p, representing common physical error mechanisms in flash memories such as read/write disturbs and charge leakage.

Polar codes provide the mathematical foundation for the present invention. These codes employ a channel polarization transformation that splits a given channel into N synthesized binary-input channels. For large N, a fraction of these channels become nearly perfect while the remainder become nearly useless. The information bits are transmitted through the good channels while the frozen channels carry fixed known values. This structure enables capacity-achieving performance with low-complexity successive cancellation decoding.

The concept of channel degradation plays a key role in the analysis of polar codes. A channel W is said to be degraded with respect to channel W' if W can be represented as the concatenation of W' with an additional channel. This relationship determines the ordering of frozen sets between different channels, which is fundamental to the nested code constructions employed in this invention.

## SUMMARY  

The present invention provides a novel coding scheme that jointly implements rewriting and error correction in memory systems. The invention employs polar coding techniques to construct WOM codes capable of multiple rewrites while simultaneously correcting substantial numbers of errors. The code construction utilizes the frozen sets corresponding to both the WOM channel and the error channel, including their common degrading and upgrading channels.

The encoding method involves generating new cell levels through a polar encoding process that incorporates both the message data and the current cell states. A dither sequence known to both encoder and decoder provides randomization of the cell values. The frozen bits for the WOM channel are carefully allocated to maintain both the rewriting functionality and error correction capability. The decoding process employs successive cancellation decoding adapted to recover the stored message from potentially noisy cell readings.

Analytical techniques are provided to determine lower bounds on the sum-rate achievable by the code. The sum-rate represents the total number of information bits that can be reliably written across all rewrite operations. The invention demonstrates that for practical error probabilities, the frozen set for the binary symmetric channel is often contained within the frozen set for the WOM channel, enabling an efficient nested code structure.

The coding scheme can be extended to support multiple-level cells (MLC) and more general noise models beyond the binary symmetric channel. The invention includes methods for rewriting memory contents, reading stored data, and managing the entire memory system incorporating these coding techniques. Specific embodiments provide implementations for both the nested code case and the general case where additional cells may be required to store certain frozen bits.

The encoding process determines current cell levels, generates new cell levels according to the message data and coding constraints, and writes the updated levels to memory. The rewriting operation minimizes cell level changes while ensuring reliable error correction. Two probabilistic models are provided for computing new cell levels, incorporating a polar code generating matrix and dither sequence.

The decoding method employs a polar code decoding algorithm to recover the stored data values from potentially corrupted cell readings. A list-decoding variant is provided to enhance performance, maintaining multiple candidate value assignments and selecting the most likely element based on computed probabilities. The decoding process recovers both the message data and any required frozen bits stored in additional cells.

The invention further provides analysis of the code's performance, including proofs of correctness and derivations of achievable sum-rates. Numerical results demonstrate the practical performance of the code for various error probabilities and rewrite counts. The code construction is shown to maintain good performance even when extended to multi-level cells and more general noise models.

## DETAILED DESCRIPTION  

The following detailed description provides specific implementations of the invention with reference to the accompanying drawings. It will be appreciated that the inventive concepts may be embodied in different forms and should not be construed as limited to the particular embodiments set forth herein.

The invention employs a coding scheme that combines rewriting and error correction for memory systems. Let there be N = 2^m binary cells used to store data, where each cell can exist in either a 0 or 1 state. The cells adhere to the WOM constraint whereby they may only transition from 0 to 1 states during write operations. A sequence of t messages M1, M2,..., Mt is written to these cells, with each message Mi containing Mi information bits.

The memory system is subject to noise modeled as a binary symmetric channel BSC(p) with crossover probability p. This represents physical errors that may flip cell states between 0 and 1. The coding scheme provides t encoding functions Ej and t decoding functions Dj that enable reliable storage and retrieval of the messages despite these errors.

The encoding function Ej for the j-th write operation takes as input the current cell state vector sj and the message Mj, producing an updated cell state vector sj' where each cell's level may only increase. The decoding function Dj recovers the message Mj from the potentially noisy cell state vector cj observed during reading.

The code construction utilizes polar coding techniques. For each rewrite operation, a WOM channel WOM(α, ε) is defined where α represents the fraction of cells at level 0 before the rewrite and ε represents the fraction of cells that change from 0 to 1 during the rewrite. The frozen set FWOM(α,ε) for this channel is determined based on polar code design principles.

Similarly, the frozen set FBSC(p) for the error channel BSC(p) is determined. When FBSC(p) is a subset of FWOM(α,ε), the code enjoys a nested structure where the WOM code's codewords form a subset of the error-correcting code's codewords. This allows efficient joint implementation of both functions.

The encoding process employs a polar encoder that incorporates a dither vector g of random bits known to both encoder and decoder. The cell values are computed as v = s ⊕ g where s represents the physical cell levels. This transformation ensures that the noise affecting v follows the same BSC(p) model as the physical noise affecting s.

The encoder determines the new cell levels by computing a vector u according to the polar code construction, where bits in the frozen sets take predetermined values and the remaining bits carry message information. The vector u is transformed through the polar code generator matrix to produce the new cell values v', which are then written to memory as s' = v' ⊕ g.

The decoding process treats the read cell values c as noisy observations of the written values s'. By computing c ⊕ g, the decoder obtains noisy versions of the polar codewords v' which can be decoded using successive cancellation decoding. The frozen bits in FWOM(α,ε) are used to recover the stored message while the frozen bits in FBSC(p) provide error correction capability.

When the nested condition FBSC(p) ⊆ FWOM(α,ε) does not hold, additional cells are employed to store the bits in FBSC(p) \ FWOM(α,ε). These additional cells use a separate error-correcting code to ensure reliable storage of these frozen bits. The decoder first recovers these additional frozen bits before proceeding with the main decoding process.

The code can be extended to support multiple rewrites by applying the encoding and decoding procedures sequentially for each write operation. The parameters α and ε are updated for each write to reflect the changing cell state distribution. The sum-rate across all writes is optimized by carefully choosing the ε values for each rewrite.

For multi-level cells (MLC), the invention employs a level-by-level approach where each level is treated as a binary channel. The coding scheme is applied independently to each level, with appropriate adjustments to account for interdependencies between levels. This allows the benefits of the invention to be realized in higher-density memory configurations.

The encoding and decoding processes can be implemented efficiently with O(N log N) complexity, making them practical for real-world memory systems. The invention includes specific embodiments where the encoding and decoding algorithms are implemented in hardware or software as part of a memory controller.

The analytical results demonstrate that the code achieves substantial sum-rates even in the presence of significant noise. Lower bounds on the achievable sum-rate are derived and shown to approach the noiseless WOM capacity as the error probability decreases. Numerical results for finite-length codes confirm these theoretical predictions.

The invention further provides methods for determining when the nested code structure is possible by analyzing the relationship between the frozen sets of the WOM channel and the error channel. This analysis shows that for practical error probabilities, the nested condition is frequently satisfied, enabling efficient implementations.

The coding scheme is shown to be correct through rigorous proofs establishing that the encoding and decoding procedures reliably store and retrieve messages while correcting errors. The proofs leverage the properties of polar codes and the channel degradation relationships between the WOM channel and error channel.

The invention includes embodiments where the coding scheme is implemented in a complete memory system comprising memory cells, a write device, a read device, and control logic implementing the encoding and decoding algorithms. The system may include buffers for temporary data storage and interfaces for communication with host devices.

## VII. ADDITIONAL EXAMPLE EMBODIMENTS  

The invention may be implemented in a data storage device constructed and configured to perform the described methods and operations. Such a device may include a memory array accessed by a memory controller operating under the control of a microcontroller. The memory controller manages communications with the memory via a write device and with a host device through a host interface.

The memory controller supervises data transfers between the host device and memory, including temporary storage of data values in buffers. An error correcting code (ECC) block maintains ECC data and performs error correction operations according to the described encoding and decoding scheme. The controller implements operations for reading data from the device and programming data into the memory cells.

The processing components may be implemented in hardware as control logic or in software as program instructions executed from program memory. The host device may comprise a computer apparatus including a processor and system memory connected via a memory bus. The processor may be a microprocessor, microcontroller, or digital signal processor incorporating multiple levels of caching, processor cores, and registers.

The system memory may include both volatile and non-volatile memory components storing an operating system, applications, and program data. The applications may include implementations of the described encoding and decoding algorithms as computer-readable instructions executed by the processor. Program data may include parameters and state information required for the coding operations.

The memory controller may be implemented as part of the processor or as a separate component. The system may include various interface controllers for facilitating communications with peripheral devices and networks. The complete computing device may range from small-form factor portable electronic devices to personal computers and enterprise storage systems.

The invention encompasses embodiments where the coding scheme is applied to various types of non-volatile memory including NAND flash, NOR flash, phase-change memory, and resistive RAM. The specific implementation details may vary according to the physical characteristics of the memory technology while maintaining the fundamental principles of the coding scheme.

Alternative embodiments may employ different variants of polar codes or other capacity-achieving codes in place of the standard polar code construction described. The frozen sets may be determined using alternative criteria while maintaining the essential relationships between the WOM channel and error channel frozen sets.

The invention may be implemented as a computer program product comprising non-transitory computer-readable media storing instructions that, when executed by a processor, perform the described encoding and decoding methods. Such media may include optical discs, magnetic storage devices, flash memory, or other machine-readable formats.

While particular embodiments have been described, it will be understood that various modifications may be made without departing from the spirit and scope of the invention. The scope of the invention should be determined with reference to the appended claims along with their full scope of equivalents.