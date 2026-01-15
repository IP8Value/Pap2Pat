# DESCRIPTION

## STATEMENT REGARDING GOVERNMENT SPONSORED RESEARCH AND DEVELOPMENT

This invention was made with government support under NSF CAREER Award CCF-0747415 and NSF Grant CCF-1217944. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates generally to the field of data storage systems, and more specifically to methods and systems for encoding and decoding data in write-once memory (WOM) devices that are subject to noise and physical degradation. The invention further pertains to error-correcting codes for non-volatile memory technologies, particularly flash memory, where cell levels can only be increased during rewriting operations. The disclosed techniques combine polar coding with joint rewriting and error correction capabilities to enhance memory endurance, reliability, and data retention in multi-write scenarios.

## BACKGROUND

Coding for rewriting is a critical technology for modern flash memory systems, enabling substantial improvements in device longevity, write speed, and power efficiency. In conventional flash memory, each memory block must undergo a costly and time-consuming erase operation before it can be rewritten, which accelerates wear and limits the number of write cycles. Coding for rewriting circumvents this limitation by allowing multiple writes to the same physical cells without requiring intermediate erasures, thereby extending the operational life of the memory device.

The foundational model for rewriting is the write-once memory (WOM) model, originally proposed by Rivest and Shamir. In this model, binary memory cells can only transition from a low state (0) to a high state (1), but never in reverse. This constraint mirrors the physical behavior of flash memory cells, where charge can be added to floating gates but cannot be selectively removed without erasing entire blocks. The WOM model provides a theoretical framework for designing codes that maximize the amount of information that can be stored over multiple write operations while respecting the unidirectional nature of cell transitions.

Numerous techniques have been developed for constructing WOM codes, including linear codes, tabular codes, codes based on projective geometry, and coset coding approaches. These early constructions achieved modest rates but laid the groundwork for more sophisticated designs. In recent years, significant advances have led to the discovery of WOM codes with substantially higher rates, culminating in capacity-achieving constructions. Notably, in 2012, capacity-achieving WOM codes were independently discovered using different approaches, with one particularly elegant construction leveraging polar coding theory.

Polar codes, introduced by Arıkan, represent a breakthrough in coding theory as the first explicitly constructed family of codes proven to achieve the capacity of symmetric binary-input discrete memoryless channels. Their application to WOM coding demonstrated that the theoretical limits of rewriting could be approached in practice, opening new avenues for memory system design.

However, practical memory systems are not ideal; they are subject to various noise sources that can corrupt stored data. These include read/write disturbs, inter-cell interference, and charge leakage, all of which can cause bit flips in stored values. While extensive research has focused on pure WOM codes, comparatively little attention has been paid to WOM codes that simultaneously provide robust error correction capabilities. Existing error-correcting WOM codes typically handle only a small number of errors (e.g., 1-3 errors), which is insufficient for real-world flash memory applications where error rates can be substantial.

This gap between theoretical rewriting capabilities and practical error resilience requirements motivates the need for joint rewriting and error correction schemes that can handle a substantial number of errors while maintaining high rewriting efficiency. Such codes must balance the competing constraints of unidirectional cell transitions and error correction redundancy, presenting unique challenges that cannot be addressed by simply combining existing WOM codes with conventional error-correcting codes.

## SUMMARY

The present invention provides a novel coding scheme that achieves joint rewriting and error correction for write-once memory devices. The invention supports any number of rewrite operations and can correct a substantial number of errors, making it suitable for practical flash memory applications. The core innovation lies in a code construction that utilizes polar coding theory to simultaneously address both rewriting constraints and error correction requirements.

The analytical foundation of the invention is based on the relationship between frozen sets corresponding to the WOM channel and the error channel, respectively. By analyzing common degrading and common upgrading channels, the invention establishes lower bounds on the sum-rate achievable by the proposed code construction. A key insight is that for relatively small error probabilities, the frozen set for the binary symmetric channel (BSC) is often contained within the frozen set for the WOM channel, enabling a nested code structure that simplifies implementation while maintaining high performance.

The invention extends beyond binary symmetric channels to multi-level cells (MLC) and more general noise models, providing a flexible framework for various memory technologies. The method includes specific procedures for rewriting memory cells, reading stored data, and operating complete memory systems that incorporate the disclosed coding techniques.

In particular, the invention introduces a method of rewriting a memory that determines current cell levels, generates next cell levels that respect the unidirectional constraint while encoding new information, and writes these next cell levels into memory. The method of reading a memory includes error correction operations that recover the originally written data despite noise-induced corruption. The memory system embodiment integrates these methods with appropriate hardware and software components to provide a complete solution for reliable, high-endurance data storage.

## DETAILED DESCRIPTION

The detailed description of the present invention begins with an introduction to the fundamental concepts and purposes underlying the disclosed techniques. The accompanying drawings, referenced throughout this description, illustrate various aspects of the invention including block diagrams of encoding and decoding processes, channel models, and system architectures. The illustrative embodiments described herein are provided to enable those skilled in the art to make and use the invention, but should not be construed as limiting the scope of the invention, which is defined by the appended claims.

The scope of disclosure encompasses methods, systems, and computer-readable media that implement joint rewriting and error correction for write-once memory devices. The invention is motivated by the practical need to extend memory endurance while maintaining data reliability in the presence of noise and physical degradation. The core coding scheme leverages polar coding theory to create a unified framework that addresses both rewriting constraints and error correction requirements simultaneously.

The coding scheme is initially analyzed for the binary symmetric channel (BSC) model, which represents a common noise model for flash memory systems. The analysis demonstrates that the proposed construction achieves provable lower bounds on sum-rate while providing robust error correction capabilities. The results are then extended to multi-level cells (MLC) and more general noise models, showing the versatility and adaptability of the invention to various memory technologies and operating conditions.

The code construction for error-correcting WOM codes begins with the definition of cell programming methods that respect the fundamental constraint that cell levels can only increase during rewriting operations. The constraints for cell programming include maintaining the unidirectional property while ensuring that sufficient redundancy is available for error correction. An example write process illustrates how these constraints are satisfied in practice.

The polar encoder and WOM channel devices work together to implement the encoding function. A block diagram illustrates the write process, showing the flow of data from the message input through the encoding operations to the final cell level assignments. The operations of the encoding method include determining current cell levels, generating next cell levels that encode the new message while respecting the unidirectional constraint, and writing these next cell levels into memory.

The rewriting operation is carefully designed to minimize unnecessary cell level changes, which helps preserve memory endurance. Two probabilistic models are introduced to compute new cell levels: a first probabilistic model that captures the statistical properties of the WOM channel, and a second probabilistic model that accounts for the error channel characteristics. A dither sequence, generated using a specified matrix A<sub>N×N</sub>, is used to randomize the encoding process and improve performance.

The vector y is computed as part of the encoding process, followed by the computation of u<sub>FWOM−FC</sub> and u<sub>FC</sub> components. The channel polarization process generates W<sub>N</sub><sup>(i)</sup> values, with base cases defined for the recursive computation. The likelihood ratios L<sub>N</sub><sup>(i)</sup> are computed to support the decoding process. The WOM cell levels are defined to represent the physical state of the memory cells after encoding.

The error correction method employs a polar code generating matrix to establish the relationship between information bits and encoded bits. The FWOM set is determined based on the WOM channel characteristics, while FER values are computed to evaluate performance. The error correction decoding process recovers the original data values from the noisy observations.

Binary vector properties are exploited to ensure proper encoding and decoding operations. Data value generation follows specific procedures that maintain the integrity of the stored information. The binary matrix A<sub>N×N</sub> defines the transformation applied during encoding. The subsets FWOM and FC are carefully chosen to balance rewriting efficiency and error correction capability.

The error distribution is characterized to guide the design of the error correction components. The polar code decoding algorithm recovers data values through a successive cancellation process. A list-decoding algorithm enhances performance by maintaining multiple candidate solutions during decoding. The ui values are computed iteratively, with list management ensuring that the most likely element is selected as the final output.

The stored data error-correcting code incorporates a list size parameter that controls the trade-off between complexity and performance. The matrix A<sub>N×N</sub> is specified to implement the desired transformation. The recover value action computes the final decoded bits, with WNi values supporting the decision process. The ui values are determined through the decoding algorithm, leading to the recovery of the original data bits.

The specification of the list-decoding algorithm includes detailed procedures for computing WNi values, updating the list of value assignments, and choosing the most likely element. The error-correcting code is implemented with careful attention to the computational requirements and performance characteristics. The WOM cells and their associated operations form the physical basis for the invention.

The rewriting method and error correction method work in concert to provide reliable data storage. The FC and FWOM sets are introduced to organize the encoding process. Embodiments of the encoding and decoding methods demonstrate practical implementation approaches. The introduction of N<sub>additional</sub> cells provides additional flexibility for handling cases where the nested structure assumption does not hold.

The modified encoding and decoding methods accommodate the additional cells while maintaining overall system performance. The extension to q-level cells enables application to multi-level cell technologies. The level-by-level approach provides a systematic method for handling multiple cell levels. Examples of encoder and decoder implementations illustrate the practical aspects of the invention.

The organization of the description follows a logical progression from basic concepts to advanced implementations. The basic model and notations establish the mathematical foundation for the invention. Embodiments of code construction demonstrate practical approaches to implementing the theoretical concepts. The embodiment of code extensions shows how the invention can be adapted to various scenarios.

The analysis of actual sum-rates achieved by code embodiments provides quantitative evidence of the invention's performance. Further example embodiments illustrate the versatility of the approach. Concluding remarks summarize the key contributions and advantages of the invention.

The model for rewriting establishes the fundamental constraints and objectives. Polar codes are introduced as the enabling technology for achieving high performance. Polar code properties are defined to support the analysis. The encoder transformation implements the encoding function. The decoding process recovers the original data.

The concept of upgrading and degrading channels provides analytical tools for understanding the relationship between different channel models. Channel degradation is defined precisely to support the theoretical analysis. The code construction with nested structure leverages the relationship between channel models to simplify implementation.

The WOM channel parameters characterize the rewriting process. The encoding function implements the rewriting operation. The decoding operation recovers the stored data. The time complexity of encoding and decoding is analyzed to ensure practical feasibility.

The extension to t-write error correcting WOM code demonstrates the scalability of the approach. The application of encoder and decoder for t writes shows how the invention handles multiple rewrite operations. Notes on computing α values for BSC(p) provide practical guidance for implementation.

The code construction is revised to handle general cases. The encoder and decoder are updated accordingly. The sum-rate equation is derived to quantify performance. The correctness of the code is analyzed to ensure reliability.

The nested structure of the code is discussed to highlight its advantages. Lemmas 1-4 are proved to establish theoretical foundations. The lower bound to sum-rate is analyzed to quantify performance limits. Equations for the number of bits written and sum-rate are derived.

Lemma 5 is proved to support the theoretical analysis. Theorem 6 is stated and proved to establish the main theoretical result. Numerical results demonstrate the practical performance of the invention. The erasure channel is introduced as an alternative noise model.

The handling of erasures is described to show the versatility of the approach. The extension to multi-level cells demonstrates applicability to advanced memory technologies. Achievable rates are discussed to quantify performance. BSCs satisfying the required condition are identified.

Achievable sum-rates for nested and general codes are shown to demonstrate performance. The lower bound to sum-rate is discussed to provide theoretical context. Concluding remarks on code performance summarize the key findings.

### VII. Additional Example Embodiments

The invention may be embodied in a data storage device that is specifically constructed and configured to perform the methods and operations disclosed herein. Such a device includes memory that is accessed by a memory controller, which operates under the control of a microcontroller to manage the overall operation of the storage system. The memory controller manages communications with the memory via a write device and coordinates communications with a host device through a host interface.

The memory controller supervises data transfers from the host to memory during write operations and manages data transfers from memory to the host during read operations. A data buffer is included for temporarily storing data values during these transfer operations. An Error Correcting Code (ECC) block maintains ECC data and performs error correction operations according to the encoding and decoding scheme disclosed herein.

The operations for operating the data storage device include specific procedures for reading data from the device and programming the data storage device. The encoding and decoding operations implement the joint rewriting and error correction functionality. The processing components may be implemented in software or hardware, or as control logic integrated into the device architecture.

Software program instructions are executed from program memory to implement the disclosed algorithms. The host device may be embodied as a computer apparatus that interfaces with the data storage device. An example computing device includes a basic configuration with a processor and system memory connected by a memory bus for communication.

The processor may be implemented as a microprocessor, microcontroller, or digital signal processor, and may include multiple levels of caching, processor cores, and registers. The memory controller may be integrated as part of the processor or implemented as a separate component. The system memory may be volatile or non-volatile memory that stores the operating system, applications, and program data.

Applications include the encoding and decoding algorithms disclosed herein, which operate with program data on the operating system. The WOM memory is written to and read from using the various features and methods disclosed in the invention. The applications include algorithms with computer-readable instructions that are executed by one or more processors to implement the disclosed functionality.

The program data includes various data structures and parameters used by the algorithms. A bus/interface controller facilitates communications between different components of the system. Data storage devices may be removable or non-removable, and include various examples of computer storage media.

The system memory, removable storage, and non-removable storage provide different tiers of storage capacity and performance. An interface bus facilitates communication from interface devices, which may include output devices such as graphics processing units and audio processing units. Peripheral interfaces may include serial interface controllers or parallel interface controllers.

Communication devices such as network controllers facilitate communications with other computing devices over networks. The computing device may be implemented as a portion of a small-form factor portable electronic device or as a personal computer, demonstrating the versatility and wide applicability of the disclosed invention.