# DESCRIPTION

## TECHNICAL FIELD

- introduce FPGA and ASIC devices

Field Programmable Gate Arrays (FPGA) and Application-Specific Integrated Circuits (ASIC) are reconfigurable or custom-designed digital hardware platforms capable of implementing complex logical operations with high parallelism and low latency. These devices consist of an array of programmable logic cells interconnected via configurable routing channels, enabling the creation of dedicated computational architectures tailored to specific data structures and algorithms. Unlike general-purpose processors that execute instructions sequentially, FPGAs allow simultaneous execution of multiple operations across spatially distributed processing elements, making them particularly suited for problems involving highly regular, repetitive, and data-parallel structures. ASICs, while less flexible, offer even greater performance and energy efficiency through permanent circuit optimization. Both technologies are increasingly employed in domains requiring intensive numerical computation, including bioinformatics, where the structure of genetic data—particularly pedigree relationships—exhibits inherent regularity that aligns with the architectural strengths of these devices. The present invention leverages the parallel processing capabilities of FPGAs and, where applicable, ASICs to accelerate the simulation and analysis of pedigree data by directly embedding the topological and probabilistic rules of inheritance into the hardware logic, thereby eliminating the overhead associated with software-based sequential execution.

## BACKGROUND ART

- define alleles and haplotypes
- explain pedigree data structures
- motivate efficient processing of pedigree data

In diploid organisms, each individual carries two copies of each gene, known as alleles, which may be identical or distinct and are inherited one from each parent during sexual reproduction. A haplotype refers to the specific combination of alleles along a single chromosome, representing the linear sequence of genetic variants inherited together from a single ancestor. Pedigree data structures encode the ancestral relationships among individuals in a population, typically represented as a directed acyclic graph in which each node corresponds to an individual and each edge denotes a parent-offspring relationship. In standard diploid pedigrees, every non-founder individual has exactly two parents, and the transmission of alleles from parents to offspring follows Mendelian principles, with one allele randomly selected from each parental pair during meiosis. The complete genotype of all individuals in a pedigree can be determined if the genotypes of the founders are known and the meiotic events along all inheritance paths are specified. However, in practice, only a subset of individuals may have observed genotypes, necessitating probabilistic inference to estimate unobserved genotypes, identity-by-descent (IBD) probabilities, or inbreeding coefficients. Traditional computational methods for such inference rely on sequential algorithms that iterate through possible combinations of meiotic events, often requiring exponential time in the number of individuals or loci. These methods become computationally prohibitive for large or inbred pedigrees due to the combinatorial explosion of possible allele transmissions. The need for efficient processing is further amplified in population genetics, medical genetics, and animal breeding, where accurate estimation of genetic parameters under uncertainty is critical for disease risk prediction, trait selection, and haplotype reconstruction. The invention addresses this challenge by replacing sequential computation with a massively parallel hardware architecture that simulates the entire pedigree’s allele flow in a single clock cycle per sample, drastically reducing the time required to generate statistically valid samples for probabilistic inference.

## DISCLOSURE OF INVENTION

- define device for representing pedigree data structures

The invention comprises a specialized hardware device configured to represent pedigree data structures as a layered network of interconnected processing modules, each corresponding to an individual in the pedigree and arranged according to generational hierarchy. Each module contains dedicated memory and logic circuitry to store and transmit alleles, evaluate inheritance consistency, and propagate signals to downstream descendants. The device is implemented on a Field Programmable Gate Array (FPGA) or, in alternative embodiments, an Application-Specific Integrated Circuit (ASIC), and is structured to mirror the topological relationships of the pedigree, ensuring that allele transmission occurs in synchrony with the natural flow of genetic inheritance across generations. The device enables the simultaneous simulation of all meiotic events within a pedigree during each clock cycle, producing a complete sample of genotypes for all individuals in a single temporal step, thereby transforming what would otherwise be an exponentially complex sequential computation into a fixed-time parallel operation.

- describe logic cells and electrical connections

The device is constructed from an array of reconfigurable logic cells, each capable of performing Boolean operations, arithmetic functions, and memory storage, interconnected via programmable routing channels that are configured to replicate the parent-offspring relationships defined in the pedigree. Each logic cell is assigned a specific role based on the individual it represents: founder, descendant, or holder. Electrical connections between cells are established during configuration to ensure that output signals from parent modules are routed exclusively to the input terminals of their direct offspring, preserving the integrity of the inheritance path. These connections are hardwired during the FPGA configuration phase, eliminating the need for dynamic address resolution or memory lookups during runtime, and enabling deterministic, low-latency data flow. The routing architecture is designed to minimize signal propagation delay and avoid contention, ensuring that all computations complete within a single clock cycle regardless of pedigree size.

- introduce input and output circuitry

The device incorporates dedicated input circuitry to receive founder genotype data, allele frequency parameters, and observed genotype constraints from an external host system, and output circuitry to transmit accumulated sample statistics, validity flags, and estimated probabilities back to the host. Input data is loaded into founder modules via parallel registers, while output data is collected from terminal layers containing counters and comparators that aggregate results across all completed samples. The output circuitry includes a master validity flag that is activated only when all descendant modules report consistent allele transmissions with observed data, ensuring that only valid samples contribute to statistical estimates.

- explain parallel processing of pedigree data

The device enables true parallel processing of pedigree data by assigning each individual a dedicated processing module that operates concurrently with all others during every clock cycle. Founder modules generate alleles simultaneously, descendant modules receive and combine parental alleles in parallel, and holder modules maintain allele continuity across multiple generations without disrupting synchronization. As a result, a complete and self-consistent sample of the entire pedigree’s genotype configuration is generated in a single clock cycle, with the number of samples produced per second determined solely by the clock frequency of the device, independent of pedigree size or complexity.

- define subset of pedigree data structure

The invention further permits the representation of a subset of the pedigree data structure by partitioning the full pedigree into independent subgraphs that can be processed in sequence or in parallel using multiple device instances. Each subset is configured as a self-contained processing unit with its own input, output, and internal logic, allowing large pedigrees to be analyzed in chunks that fit within the available hardware resources. The subsets are linked through boundary modules that preserve allele continuity between partitions, enabling the reconstruction of global inheritance patterns without requiring the entire pedigree to be resident in memory at once.

- describe embodiment with generation of pedigree

In one embodiment, the device is configured to generate a pedigree de novo by initializing founder modules with random or enumerated allele combinations and propagating inheritance through successive generations according to predefined mating rules. This embodiment is particularly useful for simulating population histories, testing inheritance models, or generating synthetic datasets for algorithm validation.

- describe embodiment with part of generation

In another embodiment, the device processes only a portion of the pedigree, such as a single lineage or a focal individual’s ancestry, by activating only the relevant modules and disabling others. This selective activation reduces resource utilization and accelerates computation for targeted analyses, such as estimating IBD probabilities for a specific individual without simulating the entire pedigree.

- describe embodiment with all generations

In a further embodiment, the device is fully populated with modules representing every individual across all generations of the pedigree, from founders to terminal descendants. This configuration enables comprehensive analysis of the entire genetic history, including the estimation of global inbreeding coefficients, haplotype reconstruction, and likelihood-based genotype inference under complex inheritance patterns.

- introduce duplicate copies of pedigree data structure

The invention includes the capability to instantiate multiple duplicate copies of the pedigree data structure within a single FPGA, each operating independently with its own set of random number generators and counters. These duplicates allow for concurrent sampling of multiple genetic scenarios, increasing throughput and enabling statistical aggregation across diverse inheritance paths without requiring additional hardware.

- describe sampling cycle

Each sampling cycle begins with the initialization of founder alleles, followed by the simultaneous transmission of one allele from each parent to each descendant module during a single clock pulse. Meiosis indicators are generated internally within each descendant module to determine which parental allele is inherited. Upon completion of transmission, validity checks are performed across all modules, and the master validity flag is set only if all observed genotype constraints are satisfied. Valid samples are then counted, and the cycle repeats, producing a new sample every clock cycle.

- introduce modules for representing individuals

Each individual in the pedigree is represented by a modular unit composed of memory registers, inheritance logic, and communication ports. Founder modules contain fixed or stochastically generated allele pairs, descendant modules contain comparators and selectors to choose parental alleles, and holder modules store and forward alleles without modification. These modules are identical in structure where functionally equivalent, enabling scalable and modular design.

- describe data counters and authenticator

Data counters are integrated into terminal layers of the device to accumulate counts of observed allele combinations, haplotypes, or IBD states across all valid samples. An authenticator circuit verifies the consistency of transmitted alleles against known genotype observations, rejecting samples that violate observed data constraints. The authenticator operates in parallel across all individuals, ensuring that only biologically plausible configurations contribute to statistical estimates.

- introduce filter for rejecting inconsistent data

A dedicated filter circuit is embedded within each descendant module to evaluate whether the inherited alleles are compatible with any observed phenotypic or genotypic data associated with that individual. If a mismatch is detected, the module generates a rejection signal that propagates to the master validity flag, preventing the sample from being counted. This filtering occurs in real time during each clock cycle, eliminating invalid samples before they consume downstream resources.

- describe generator for generating pedigree data

The device includes a configurable generator that produces founder alleles according to specified allele frequencies, mutation rates, or observed genotype distributions. This generator may be deterministic, enumerating all possible combinations, or stochastic, employing cellular automata-based pseudo-random number generators optimized for FPGA implementation to simulate random sampling from multinomial distributions.

- introduce inheritance generator

An inheritance generator within each descendant module selects one allele from each parent using a binary meiosis indicator, which is either pre-determined or generated via a pseudo-random process. The generator ensures that each meiosis event is independent and follows Mendelian segregation rules, with the selection mechanism implemented using simple multiplexers and flip-flops to minimize circuit complexity.

- describe weighting of generated data

In embodiments requiring likelihood-based inference, the device applies weights to each valid sample based on recombination rates, allele frequencies, or observed trait values. These weights are computed in parallel using fixed-point arithmetic units and accumulated in specialized counters, enabling estimation of posterior probabilities without the need for floating-point operations or external memory access.

- describe reconfiguration of FPGA

The FPGA is reconfigurable to adapt to different pedigree topologies, inheritance models, or analysis objectives. Configuration is performed offline using a hardware description language, allowing the device to be tailored for specific datasets without altering the underlying architecture. Reconfiguration is rapid and can be automated through software interfaces, enabling seamless transitions between different analytical tasks.

- describe method for processing pedigree data

The method involves initializing the FPGA with a configured pedigree topology, loading founder genotypes and constraints, and initiating a continuous sequence of clock cycles during which each cycle produces one complete sample of the pedigree’s genetic configuration. Valid samples are counted and aggregated in terminal layers, while invalid samples are discarded. Statistical estimates are derived from the accumulated counts, with precision improving as the number of samples increases.

- describe advantages of invention

The invention provides a dramatic reduction in computation time for pedigree analysis, achieving speedups of several orders of magnitude over sequential software implementations. It eliminates the combinatorial bottleneck inherent in traditional algorithms by exploiting the spatial parallelism of FPGA hardware. The device requires no dynamic memory allocation, avoids branching overhead, and operates deterministically, making it ideal for real-time or high-throughput genetic analysis. Its modular design enables scalability, and its energy efficiency surpasses that of CPU-based alternatives, making it suitable for deployment in resource-constrained environments.

## BEST MODE FOR CARRYING OUT THE INVENTION

- introduce FPGA structure

The preferred embodiment of the invention is implemented on a Xilinx Spartan 3 FPGA, comprising a grid of configurable logic blocks (CLBs), embedded memory blocks, and programmable interconnects. Each logic block contains lookup tables, flip-flops, and carry chains capable of implementing Boolean logic, arithmetic operations, and state storage. The interconnect fabric is configured to route signals between CLBs according to the pedigree’s parent-offspring relationships, forming a spatially embedded inheritance network.

- describe logic blocks and connections

Each logic block is assigned to represent a single individual, with its internal logic configured to perform allele selection, inheritance validation, and signal propagation. Connections between blocks are established during configuration to mirror the pedigree’s familial structure, ensuring that each descendant receives inputs only from its designated parents. The routing is optimized to minimize path delays and eliminate signal contention, enabling all operations to complete within a single clock cycle.

- illustrate pedigree data structure on FPGA

The pedigree is mapped onto the FPGA as a layered architecture, with founder individuals occupying the top layer, descendants arranged in subsequent layers, and terminal counters positioned below the final generation. Holder modules are inserted where individuals contribute to multiple descendant lines, preserving allele continuity across generations. The physical layout of the FPGA is arranged to reflect the depth and breadth of the pedigree, with modules aligned to facilitate synchronous data flow.

- define individuals and layers

Each individual is represented by a discrete module located in a specific layer corresponding to its generational position. Layer 0 contains founder modules, each with two allele registers. Subsequent layers contain descendant modules, each equipped with allele selectors, comparators, and validity flags. The number of layers equals the maximum number of generations separating any descendant from a founder.

- describe founders and descendants

Founders are initialized with known or sampled allele pairs and serve as the source of genetic variation. Descendants inherit one allele from each parent, selected via a meiosis indicator generated internally. Descendant modules contain logic to compare inherited alleles against observed genotypes and to propagate valid transmissions to their own offspring.

- explain allele transmission

Allele transmission occurs synchronously across all modules during each clock cycle. Each descendant module receives one allele from each parent module via dedicated wiring, selects one allele per locus using a binary switch, and updates its internal state. The transmission is deterministic and occurs without delay, ensuring that all individuals in the same generation process the same sample simultaneously.

- illustrate transformation of data through clock cycles

At clock cycle t, founder alleles are generated. At cycle t+1, first-generation descendants receive their alleles. At cycle t+2, second-generation descendants receive theirs, and so on. By cycle t+g, where g is the number of generations, the terminal layer receives the final genotype configuration. A new sample begins at cycle t+1, allowing g concurrent samples to propagate through the system.

- apply gene dropping algorithm

The device implements the gene dropping algorithm by stochastically assigning founder alleles and simulating meiotic segregation in parallel. Each valid sample contributes to the accumulation of allele counts, IBD states, or haplotype frequencies in terminal counters, enabling estimation of genetic probabilities without explicit likelihood calculations.

- describe random number generator

A cellular automata-based pseudo-random number generator is embedded within each founder and descendant module to produce independent meiosis indicators. The generator is implemented using simple shift registers and XOR logic, requiring minimal FPGA resources while producing high-quality random sequences suitable for genetic simulation.

- generate paternal and maternal allele pairs

Each parent module outputs two alleles, one designated as paternal and one as maternal, based on the sex of the parent. The descendant module selects one allele from each parent using a binary selector controlled by the meiosis indicator, ensuring Mendelian inheritance.

- illustrate allele transmission through generations

Alleles flow downward through the pedigree in a pipelined fashion, with each layer processing the same sample at a different time. A holder module retains an allele for multiple cycles, allowing it to be passed to multiple descendants without disrupting the synchronization of the system.

- describe holder modules

Holder modules contain no logic for allele selection but serve as memory buffers, storing a single allele pair received from a parent and forwarding it unchanged to multiple offspring. They are essential for maintaining temporal alignment in pedigrees where individuals have offspring across multiple generations.

- illustrate allele output to terminal layer

At the terminal layer, allele counters accumulate the frequency of each genotype, haplotype, or IBD state observed across all valid samples. The terminal layer is physically separated from the main pedigree structure to prevent interference with ongoing transmission.

- estimate allelic probabilities

Allelic probabilities are estimated by dividing the count of each observed genotype by the total number of valid samples. These estimates are computed in real time and output to the host system for downstream analysis.

- illustrate configuration of descendent module

Each descendant module contains two input ports for parental alleles, a meiosis selector, a comparator for observed genotype validation, and an output port to transmit the selected allele to offspring. The module is implemented using a lookup table, flip-flop, and multiplexer, requiring fewer than 100 logic cells.

- describe allele counter

The allele counter is a dedicated register that increments each time a specific allele combination is observed in a valid sample. Multiple counters are arranged in the terminal layer to track all possible genotypes, haplotypes, or IBD states.

- test and count valid allele configurations

Each module tests its inherited alleles against observed data. If consistent, it signals validity. Only when all modules signal validity is the master valid flag activated, allowing the counters to increment. Invalid samples are discarded without affecting the counters.

- illustrate configuration of founder module

Founder modules contain two fixed or stochastically generated allele registers, a random number generator for allele sampling, and an output port to transmit alleles to offspring. They require no input and operate independently.

- describe test experiment

A test experiment was conducted using pedigrees of 32 and 60 individuals, comparing the FPGA implementation to a sequential CPU-based gene dropping algorithm. The FPGA produced 50 million samples per second, while the CPU produced 300,000 samples per second, yielding a speedup of 166-fold.

- compare FPGA with general purpose CPU

The FPGA outperforms the CPU by eliminating instruction fetch cycles, branch prediction overhead, and memory latency. The CPU processes individuals sequentially, while the FPGA processes them in parallel. The FPGA’s throughput scales with the number of individuals, whereas the CPU’s time increases combinatorially.

- describe application to inbreeding coefficients

Inbreeding coefficients are estimated by counting the proportion of samples in which an individual inherits two alleles identical by descent. The FPGA computes this directly in parallel, updating a counter for each individual every cycle.

- illustrate structure of descendant module

The descendant module consists of two input registers, a selector multiplexer, a comparator, and an output register. All components are synchronized to a single clock, ensuring deterministic operation.

- describe comparator and counter

The comparator checks whether inherited alleles match observed genotypes. The counter increments only when all comparators in the pedigree return a positive result, ensuring that only valid samples are counted.

- illustrate allele transmission and comparison

Alleles are transmitted from parents to descendants in a single cycle. Comparison occurs immediately upon receipt, with validity signals propagated upward to the master flag. The entire process is completed within the duration of one clock pulse.

- describe test experiment

A second test experiment was conducted on a 20-individual pedigree with four known genotypes. The FPGA achieved a 495-fold speedup over the CPU, demonstrating scalability with increasing constraint density.

- describe alternative embodiment

In an alternative embodiment, the pedigree is partitioned into multiple subsets, each processed by a separate FPGA module. Subsets are linked via boundary modules that pass allele states between partitions, enabling analysis of pedigrees larger than the capacity of a single device.

- illustrate modules containing subsets of structure

Each subset is a self-contained pedigree fragment with its own founders, descendants, and terminal counters. Modules at the boundaries transmit allele states to adjacent subsets, preserving global consistency.

- describe operations on subsets

Operations on subsets include independent sampling, local validity checking, and conditional aggregation. Valid samples from each subset are combined at a central controller to form a global count.

- describe ready flag and data passing

Each subset module asserts a ready flag when its internal processing is complete. Data is passed only when all upstream modules signal readiness, ensuring synchronization across partitions.

- describe Metropolis-Hastings accept/reject step

In advanced embodiments, the device implements a Metropolis-Hastings algorithm by generating a proposed sample, computing a weight ratio based on likelihood, and accepting or rejecting the sample using a comparison against a random threshold. The ratio is computed using integer arithmetic to avoid floating-point operations.

- describe weighted samples

Weighted samples are accumulated in specialized counters that add the weight value rather than incrementing by one. Weights are derived from recombination rates, allele frequencies, or trait likelihoods.

- describe modified inheritance generators

Modified inheritance generators incorporate linkage disequilibrium, mutation rates, or non-Mendelian inheritance patterns by adjusting the probability distribution of meiosis indicators. These modifications are implemented as lookup tables within the selector logic.

- describe modified random generators

Modified random generators use multiple independent cellular automata to produce correlated random streams for linked loci, ensuring accurate modeling of recombination hotspots and haplotype blocks.

- illustrate alternative embodiment of pedigree data structure

In an alternative structure, the pedigree is represented as a circular graph with feedback loops to model inbreeding. The FPGA is reconfigured to allow allele transmission to return to earlier generations, with validity checks ensuring consistency with observed data.

- describe two FPGAs with pseudo-random number generators

Two FPGAs are used in tandem, each with an independent pseudo-random number generator, to produce uncorrelated samples for statistical validation. The outputs are compared to ensure convergence and reliability.

- illustrate validity tester and AND gate

Each descendant module contains a validity tester that outputs a binary signal. These signals are fed into a hierarchical AND gate network that produces the master valid flag only when all individual tests pass.

- describe Master Valid Flag

The Master Valid Flag is a single-bit register that is set only when every module in the pedigree reports a valid transmission. It controls whether the sample counters are updated, ensuring that only biologically plausible samples contribute to results.

- describe allele counter and Master Sample Valid Flag

The allele counter increments only when the Master Sample Valid Flag is high. This ensures that all statistical estimates are derived exclusively from samples that satisfy all observed constraints.

- illustrate configuration of descendent module

The descendant module is configured with dual-input registers, a selector multiplexer, a comparator, and a validity output. All components are clock-synchronized and occupy less than 120 logic cells.

- describe allele validity tester

The allele validity tester compares the inherited allele pair against a predefined genotype constraint. If the constraint is absent, the test passes. If a constraint is present, the test passes only if the inherited alleles match.

- describe allele counter

The allele counter is a multi-bit register that accumulates the frequency of each genotype observed in valid samples. Multiple counters are arranged in parallel to track all possible allele combinations.

- describe alternative configurations of components

Alternative configurations include replacing multiplexers with ROM-based selectors, using shift registers for allele storage, or integrating arithmetic units for likelihood weighting. These configurations are selected based on available FPGA resources and desired precision.

- illustrate central meiosis module

A central meiosis module is introduced to coordinate meiosis indicator generation across all descendant modules. It broadcasts a common random seed to ensure reproducibility while maintaining independence across loci.

- describe paternal and maternal selectors

Paternal and maternal selectors are separate multiplexers within each descendant module, each receiving one allele from the corresponding parent. They are controlled by independent meiosis indicators to model sex-specific inheritance patterns.

- illustrate allele selection table and counters

An allele selection table maps parental allele combinations to offspring genotypes. Counters track the frequency of each outcome across all valid samples, enabling direct estimation of genotype probabilities.

- describe specialized modules for sires, dams, and terminals

Specialized modules are designed for sires (male founders), dams (female founders), and terminals (individuals with no offspring). Sire and dam modules include sex-specific inheritance logic, while terminal modules contain only output ports and counters.

- describe Supervisor soft processor

A Picoblaze soft processor is embedded within the FPGA to manage external communication, load pedigree data, initiate sampling cycles, and retrieve results. It operates in parallel with the hardware logic, providing a user-friendly interface without compromising performance.

- describe acceleration of allele probability estimation

The invention accelerates allele probability estimation by eliminating sequential loops and performing all computations in parallel. The time required to estimate probabilities is independent of pedigree size and depends only on the clock frequency and number of samples required for statistical precision.