## FIELD

- define field

The present invention pertains to the field of quantum information processing, specifically to scalable, fault-tolerant photonic quantum computing architectures that leverage hybrid resource states composed of discrete-variable bosonic qubits encoded via the Gottesman-Kitaev-Preskill (GKP) scheme and continuous-variable squeezed states of light. The invention integrates advanced photonic components—including multiplexed Gaussian boson sampling devices, time- and space-domain interferometric networks, homodyne detection systems, and passive entanglement structures—to enable universal quantum computation on a planar, room-temperature photonic chip. This architecture uniquely combines the deterministic generation of continuous-variable entangled states with the probabilistic heralding of non-Gaussian GKP qubits, thereby circumventing the prohibitive hardware demands of prior photonic quantum computing schemes that require deterministic, on-demand GKP state generation. The system is designed to operate with minimal cryogenic infrastructure, high clock speeds enabled by homodyne detection, and modular component integration, making it uniquely suited for large-scale, industrially scalable quantum computation using existing silicon photonics and integrated electronics platforms.

## BACKGROUND

- motivate quantum computing

Quantum computing holds the promise of solving computational problems that are intractable for classical systems, including the simulation of quantum many-body systems, optimization of complex combinatorial structures, and the efficient factorization of large integers. Despite decades of progress across multiple physical platforms—including superconducting circuits, trapped ions, and neutral atoms—no architecture has yet demonstrated a scalable, fault-tolerant quantum computer capable of outperforming classical machines on practical tasks. A critical bottleneck in all current approaches is the challenge of maintaining quantum coherence while scaling to millions of physical qubits, a requirement for fault tolerance under realistic noise conditions. Photonic systems offer a compelling alternative due to their inherent compatibility with room-temperature operation, low decoherence rates, and seamless integration with existing optical communication infrastructure. However, prior photonic architectures have been constrained by the inability to generate the non-Gaussian resource states necessary for universal computation at sufficient rates and fidelities. While continuous-variable cluster states can be deterministically generated at large scales, they lack the non-Gaussianity required for universal quantum gates. Conversely, discrete-variable encodings such as GKP qubits provide the necessary non-Gaussian operations and intrinsic error correction but suffer from probabilistic generation mechanisms that demand prohibitively large multiplexing overheads when deployed in isolation. These limitations have prevented photonic platforms from achieving the scalability and fault-tolerance thresholds required to realize a practical quantum computer. The present invention overcomes these barriers by introducing a hybrid architecture that replaces failed GKP state generation events with readily producible squeezed vacuum states, thereby enabling fault-tolerant computation without requiring deterministic sources of bosonic qubits.

## SUMMARY

- summarize system

The invention discloses a scalable, fault-tolerant photonic quantum computing system that generates and manipulates a hybrid resource state composed of Gottesman-Kitaev-Preskill (GKP) qubits and momentum-squeezed vacuum states arranged in a three-dimensional Raussendorf-Harrington-Goyal (RHG) lattice. The system comprises three modular components: a state preparation module that employs multiplexed Gaussian boson sampling devices to probabilistically generate GKP qubits and deterministically substitute failed events with squeezed vacuum states; a computational module that entangles these modes into a two-plus-one-dimensional hybrid cluster state using time-delayed interferometric loops and spatially multiplexed controlled-Z gates; and a photonic quantum processing unit that performs homodyne measurements on the cluster state with adaptive feed-forward control to execute quantum algorithms. The architecture leverages the Gaussian nature of squeezed states to preserve entanglement connectivity even when GKP qubits are unavailable, while a tailored two-tier decoding protocol—comprising an inner decoder that exploits correlated noise structure in homodyne measurement outcomes and an outer decoder that applies minimum-weight perfect matching on a modified syndrome graph—enables fault-tolerant error correction. The entire system operates at room temperature, requires no inline squeezing or cryogenic switching networks, and is fully implementable on a planar silicon photonic chip using standard lithographic fabrication techniques. The invention achieves a swap-out threshold of approximately 13.3% for experimentally accessible 15 dB squeezing, substantially reducing the multiplexing overhead required for GKP state generation compared to prior architectures.

## DETAILED DESCRIPTION

- introduce photonic quantum computing

Photonic quantum computing exploits the quantum properties of light to encode, process, and transmit quantum information. Unlike matter-based qubits, photonic systems are inherently resistant to decoherence, operate at ambient temperatures, and can leverage decades of advancement in optical telecommunications and integrated photonics. The fundamental unit of quantum information in photonic systems is a mode of the electromagnetic field, which can be manipulated using linear optical elements such as beam splitters, phase shifters, and squeezers, along with photon-number-resolving or homodyne detectors. While early photonic quantum computing schemes relied on single-photon encoding and probabilistic entangling gates, recent advances have shifted toward continuous-variable approaches, where quantum information is encoded in the quadrature amplitudes of coherent or squeezed states of light. These approaches benefit from deterministic generation of large-scale entangled cluster states using Gaussian operations, which are experimentally accessible with current technology. However, universal quantum computation requires non-Gaussian operations, which cannot be implemented deterministically using only linear optics. The Gottesman-Kitaev-Preskill (GKP) encoding provides a solution by mapping discrete qubit operations onto continuous-variable Gaussian transformations, enabling Clifford gates and Pauli measurements to be implemented with homodyne detection and Gaussian resources. The challenge lies in generating GKP qubits themselves, which traditionally require post-selection via photon counting and suffer from low heralding probabilities. This invention resolves this limitation by introducing a hybrid architecture that does not require deterministic GKP generation, instead replacing missing GKP states with squeezed vacuum modes that preserve the entanglement topology while enabling fault-tolerant computation through a novel decoding strategy.

- limitations of known systems

Existing photonic quantum computing architectures fall into two distinct categories, each with severe scalability limitations. The first class relies on continuous-variable cluster states for computation, with discrete-variable GKP qubits embedded as non-Gaussian resources. These systems require GKP qubits to be generated deterministically and inserted into the cluster at regular intervals, a requirement that imposes an unfeasible engineering burden due to the probabilistic nature of GKP state generation via Gaussian boson sampling. Even with spatial multiplexing of hundreds or thousands of GBS devices, achieving near-unit success probability demands exponential increases in hardware overhead, rendering such systems impractical for large-scale computation. The second class of architectures attempts to generate cluster states entirely from GKP qubits, but this approach suffers from the opposite problem: the entangling operations between GKP qubits are themselves probabilistic, requiring fusion gates that consume additional photons and further compound the multiplexing burden. Moreover, both classes require active, reconfigurable optical switches operating under cryogenic conditions to route and entangle modes, which introduces significant optical loss, thermal noise, and fabrication complexity. These systems are incompatible with planar, monolithic photonic integration and cannot scale to the millions of qubits required for fault tolerance. Additionally, prior decoding protocols for hybrid systems either ignore the spatial correlations introduced by squeezed-state substitution or rely on computationally intractable maximum-likelihood methods that cannot operate at the GHz clock speeds enabled by homodyne detection. The absence of a scalable, passive, room-temperature architecture capable of tolerating probabilistic GKP generation has thus remained a fundamental barrier to realizing photonic quantum advantage.

- embodiment of system configuration

The invention comprises a three-module photonic quantum computing system configured to generate, entangle, and measure a hybrid resource state on a planar silicon photonic chip. The first module, the state preparation module, consists of an array of multiplexed Gaussian boson sampling devices, each fed by momentum-squeezed vacuum inputs and terminated by photon-number-resolving detectors. Upon detection of a specific photon-number pattern, a GKP qubit is heralded in the remaining output mode; if no such pattern is observed, a momentum-squeezed vacuum state is automatically substituted via a fast optical switch. The second module, the computational module, receives these hybrid modes and entangles them into a (2+1)-dimensional lattice using time-delayed interferometric loops and spatially arranged controlled-Z gates. Each temporal loop delays a mode by one clock cycle, allowing it to interact sequentially with incoming modes via a CZ gate implemented by a beam splitter and two single-mode squeezers. Spatial connectivity is established by interleaving even- and odd-clock-cycle sources and applying CZ gates between adjacent temporal clusters, forming a 3D cubic lattice structure. The third module, the photonic quantum processing unit, consists of an array of homodyne detectors, each coupled to a local oscillator whose phase is dynamically controlled by a classical processor. The homodyne measurements yield continuous-valued outcomes that are processed in real time by a two-tier decoder: the inner decoder maps the correlated Gaussian noise structure of the hybrid lattice into binary qubit outcomes, while the outer decoder uses these outcomes to construct a syndrome graph and compute a minimum-weight perfect matching to determine the optimal recovery operation. All components are fabricated on a single planar chip using standard CMOS-compatible silicon photonics, enabling mass production and integration with high-speed electronics for feed-forward control.

### INTRODUCTION

- motivate photonic quantum computing

Photonic quantum computing offers a uniquely scalable pathway to fault-tolerant quantum computation due to its immunity to decoherence, compatibility with room-temperature operation, and seamless integration with global optical communication networks. Unlike matter-based qubits, photons do not require cryogenic environments or complex isolation systems, allowing for compact, modular, and industrially manufacturable quantum processors. The ability to encode quantum information in the quadrature amplitudes of light enables deterministic generation of large-scale entangled states using linear optics, a capability unmatched by other platforms. Furthermore, photonic systems are inherently compatible with high-bandwidth classical control, permitting real-time feedback and adaptive measurement strategies essential for error correction. These advantages position photonics as the only platform capable of scaling to the millions of physical qubits required for practical quantum advantage, provided that the non-Gaussian resources necessary for universal computation can be generated efficiently. The present invention fulfills this requirement by introducing a hybrid architecture that replaces the unattainable goal of deterministic GKP state generation with a robust, substitution-based strategy that preserves entanglement integrity while dramatically reducing hardware overhead.

- advantages of photonic quantum computing

Photonic quantum computing offers several decisive advantages over competing platforms. First, it enables computation at room temperature, eliminating the need for expensive, bulky cryogenic infrastructure and allowing for the use of low-cost, high-yield silicon photonic fabrication techniques. Second, photonic systems are intrinsically compatible with low-loss optical interconnects, facilitating modular architectures where state generation, entanglement, and measurement can be distributed across separate chips connected via optical fibers without introducing transduction noise. Third, the use of homodyne detection for measurement enables clock speeds on the order of gigahertz, orders of magnitude faster than photon-number-resolving detectors or superconducting qubit readout systems, which permits the use of shorter delay lines and reduces optical loss. Fourth, GKP-encoded qubits allow Clifford gates and Pauli measurements to be implemented using only Gaussian resources—beam splitters, phase shifters, and squeezers—which are native to integrated photonic circuits. Finally, the planar geometry of the architecture ensures that each qubit interacts only with a constant number of nearest neighbors, preserving local noise structure and enabling direct application of topological error-correcting codes such as the RHG lattice. These combined advantages make photonic quantum computing not merely competitive, but uniquely suited for scalable, fault-tolerant, and industrially viable quantum computation.

- two main classes of photonic quantum computing system configurations

Photonic quantum computing architectures can be broadly categorized into two distinct classes based on their resource generation strategy. The first class employs continuous-variable cluster states as the primary computational substrate, embedding discrete-variable GKP qubits as non-Gaussian resources to enable universal computation. These systems benefit from the deterministic, low-depth generation of large-scale entangled states using Gaussian operations, but they critically depend on the reliable, on-demand injection of high-fidelity GKP qubits into the cluster. Since GKP state generation via Gaussian boson sampling is inherently probabilistic, achieving high success rates requires massive spatial multiplexing, leading to impractical hardware overheads. The second class attempts to construct cluster states entirely from GKP qubits, thereby eliminating the need for continuous-variable resources. However, this approach suffers from the non-deterministic nature of entangling operations between GKP modes, which require fusion gates that consume additional photons and further reduce the overall success probability. Both classes are thus fundamentally constrained by the trade-off between resource generation fidelity and scalability, and neither has demonstrated a viable pathway to fault-tolerant computation at the scale required for quantum advantage.

- first class: continuous variable cluster states with encoded qubits

The first class of photonic quantum computing architectures relies on the deterministic generation of continuous-variable cluster states, typically constructed by applying controlled-Z gates between momentum-squeezed vacuum modes. These cluster states serve as the universal resource for measurement-based quantum computation, where logical operations are performed by measuring individual modes with adaptive homodyne detection. To achieve universality, Gottesman-Kitaev-Preskill (GKP) qubits are embedded at selected nodes of the cluster to provide the necessary non-Gaussianity for implementing non-Clifford gates. However, this architecture demands that GKP qubits be generated with high fidelity and inserted into the cluster at precisely timed intervals. Given that GKP state generation via Gaussian boson sampling has heralding probabilities on the order of 1% or less, achieving near-deterministic operation requires thousands of parallel GBS devices, each with associated photon-number-resolving detectors and feed-forward switching networks. The resulting system becomes prohibitively complex, power-intensive, and susceptible to optical loss and timing jitter. Moreover, the need for active, reconfigurable optical switches operating under cryogenic conditions to route GKP states into the cluster introduces additional noise and fabrication challenges that are incompatible with scalable, planar photonic integration.

- limitations of first class

The primary limitation of continuous-variable cluster state architectures with embedded GKP qubits is their dependence on deterministic GKP state generation, which is fundamentally incompatible with the probabilistic nature of current GKP preparation techniques. Even with multiplexing, the number of GBS devices required to achieve a success probability above 99% scales logarithmically with the inverse of the failure rate, leading to hardware overheads that grow exponentially with system size. This renders the architecture infeasible for large-scale quantum computation. Furthermore, the insertion of GKP qubits into the cluster requires precise synchronization between the state generation and entanglement modules, which is difficult to maintain over long optical delay lines. The active switching networks needed to route heralded GKP states introduce significant optical loss and thermal noise, particularly when operated at cryogenic temperatures. Additionally, the architecture does not tolerate missing GKP qubits; if a qubit fails to be generated, the cluster must be discarded or reconfigured, leading to catastrophic failure of the computation. These limitations prevent the architecture from achieving the fault-tolerance thresholds necessary for practical quantum advantage.

- second class: direct generation of encoded qubit clusters

The second class of photonic quantum computing architectures seeks to generate entangled cluster states directly from GKP qubits, avoiding the use of continuous-variable resources. In this approach, each node of the cluster is initialized as a GKP qubit, and entangling operations are performed using fusion gates or controlled-Z interactions between neighboring modes. This strategy preserves the intrinsic error-correcting properties of GKP states throughout the computation and eliminates the need for hybrid resource states. However, the generation of each GKP qubit remains probabilistic, and the entangling operations between them are also non-deterministic, requiring additional photons and measurement outcomes to succeed. As a result, the overall success probability of generating a large cluster state decays exponentially with the number of qubits, necessitating enormous multiplexing overheads to maintain a non-negligible success rate. Furthermore, the fusion gates required to connect GKP qubits are inherently lossy and sensitive to timing jitter, making them incompatible with high-speed, integrated photonic platforms. The architecture also lacks a mechanism to tolerate missing qubits, as any failure in the generation or entanglement process results in a broken cluster that cannot be repaired without restarting the entire computation.

- limitations of second class

The fundamental limitation of direct GKP cluster generation is its reliance on deterministic entanglement between probabilistic resources, a combination that leads to an intractable combinatorial failure rate. Even if individual GKP qubits could be generated with high fidelity, the probability of successfully entangling N qubits into a connected cluster scales as the product of the success probabilities of each entangling gate, which rapidly approaches zero for large N. This makes the architecture fundamentally incapable of scaling to the thousands of qubits required for fault-tolerant computation. Additionally, the fusion gates used to connect GKP qubits require precise photon-number resolution and are sensitive to photon loss, which further reduces their success probability. The architecture also lacks any mechanism to recover from failed qubit generation events, meaning that every missing or corrupted qubit results in a complete computational failure. This fragility, combined with the extreme hardware overhead required to achieve even modest cluster sizes, renders the second class of architectures unsuitable for scalable quantum computation.

- hybrid scheme leveraging advantages of both classes

The present invention introduces a novel hybrid scheme that synergistically combines the strengths of both continuous-variable and discrete-variable photonic quantum computing architectures while eliminating their respective weaknesses. Rather than requiring deterministic GKP state generation, the system substitutes failed GKP generation events with momentum-squeezed vacuum states, which are easily and deterministically produced using standard squeezing techniques. These squeezed states preserve the entanglement structure of the cluster when coupled via controlled-Z gates, ensuring that the overall topology remains intact even in the absence of GKP qubits. Crucially, the squeezed states are not treated as erasures but as functional, albeit noisier, computational modes that can still participate in quantum operations. The resulting hybrid resource state retains the scalability of continuous-variable cluster generation while incorporating the non-Gaussianity of GKP qubits where available. This substitution strategy reduces the multiplexing overhead by orders of magnitude, as the system no longer requires perfect GKP generation but only a sufficient density of successful events to maintain fault tolerance. The architecture is further enhanced by a tailored decoding protocol that accounts for the spatial correlations introduced by the presence of squeezed states, enabling high-fidelity error correction even in the presence of substantial swap-out rates.

- desirable features for scalability

The architecture is designed with scalability as its central objective, incorporating several key features that enable exponential growth in qubit count without proportional increases in complexity. First, the system is fully planar, with all optical components arranged on a single silicon photonic chip, eliminating the need for three-dimensional stacking or complex inter-chip connections. Second, the computational lattice is constructed in (2+1) dimensions—two spatial dimensions and one temporal dimension—ensuring that each mode traverses a path of constant optical depth, independent of the total number of qubits. This prevents the exponential growth of optical loss that plagues architectures requiring longer delay lines for larger systems. Third, the use of homodyne detection, rather than photon-number-resolving detectors, enables clock speeds on the order of gigahertz, allowing for shorter delay lines and reduced photon loss. Fourth, the system is modular, with state generation, multiplexing, entanglement, and measurement functions distributed across specialized sub-modules that can be independently optimized and fabricated. Finally, the entire system operates at room temperature, eliminating the need for cryogenic cooling of the computational core and enabling the use of standard CMOS electronics for high-speed feed-forward control.

- fully on-chip implementation

The invention enables a fully on-chip implementation of the hybrid quantum computing architecture using standard silicon photonics and integrated electronics. All optical components—including Gaussian boson sampling circuits, beam splitters, phase shifters, delay lines, and homodyne detection cells—are fabricated on a single silicon substrate using complementary metal-oxide-semiconductor (CMOS) compatible processes. The state preparation module consists of arrays of interferometers and squeezers coupled to on-chip photon-number-resolving detectors, while the computational module integrates time-delayed optical loops and spatially multiplexed controlled-Z gates implemented via cascaded beam splitters and squeezers. The photonic quantum processing unit comprises an array of homodyne detectors, each with an integrated local oscillator and phase modulator, allowing for real-time adaptive measurement settings. The classical control system, responsible for decoding homodyne outcomes and updating phase settings, is implemented using high-speed application-specific integrated circuits (ASICs) fabricated on the same chip or in close proximity via advanced packaging techniques. This monolithic integration minimizes optical loss, eliminates alignment drift, and enables mass production at low cost, making the architecture uniquely suited for industrial deployment.

- modular system configuration

The system is organized into three distinct, modular components: a state preparation module, a computational module, and a photonic quantum processing unit. Each module is designed to fulfill a specialized function with independent hardware requirements, enabling parallel optimization and scalable fabrication. The state preparation module, which generates GKP qubits and substitutes failed events with squeezed states, requires low-loss, non-reconfigurable optical circuits and high-efficiency photon-number-resolving detectors, and can be fabricated on a separate chip optimized for cryogenic operation if necessary. The computational module, responsible for entangling modes into a hybrid cluster state, requires stable, phase-insensitive interferometers and delay lines but does not require active switching or reconfigurability, allowing it to be fabricated on a passive silicon photonic chip. The photonic quantum processing unit, which performs homodyne measurements and feed-forward control, requires high-speed phase modulators and analog-to-digital converters but operates at room temperature and can be integrated with commercial electronics. This modularity allows for the use of commercially available components where possible, reduces system complexity, and facilitates incremental upgrades—such as replacing cryogenic detectors with room-temperature alternatives—as technology advances.

- theoretical features: planar system configuration and hybrid resource state

The theoretical foundation of the invention lies in the combination of a planar system configuration with a hybrid resource state composed of GKP qubits and squeezed vacuum modes. The planar geometry ensures that the entanglement structure of the Raussendorf-Harrington-Goyal lattice is preserved without requiring intersecting waveguides or non-planar routing, which would introduce optical loss and fabrication complexity. The hybrid resource state, in turn, enables fault tolerance in the presence of probabilistic GKP generation by replacing missing qubits with squeezed states that maintain the cluster’s connectivity. This substitution does not introduce erasure errors but rather a structured, Gaussian noise that can be modeled and corrected using a tailored two-tier decoder. The resulting architecture is mathematically equivalent to a topological quantum error-correcting code operating on a lattice with a known, spatially correlated noise profile, allowing for the application of minimum-weight perfect matching and other efficient decoding algorithms. The planar configuration further ensures that noise remains local, a prerequisite for fault tolerance, as each mode interacts only with a bounded number of neighbors. Together, these features enable the system to achieve fault-tolerant thresholds that are both experimentally accessible and scalable to millions of qubits.

- overview of system configuration

The overall system configuration consists of three interconnected modules that operate in sequence to generate, entangle, and measure a hybrid resource state for fault-tolerant quantum computation. The state preparation module receives input pulses of momentum-squeezed vacuum and routes them through an array of Gaussian boson sampling devices, each terminated by photon-number-resolving detectors. Upon detection of a heralding pattern, a GKP qubit is output; otherwise, a squeezed vacuum state is substituted via a fast optical switch. The outputs of these devices are fed into the computational module, which consists of a two-dimensional array of time-delayed interferometric loops and spatially arranged controlled-Z gates. These gates entangle the modes into a (2+1)-dimensional hybrid cluster state, where each temporal layer corresponds to a one-dimensional cluster, and spatial connections form the higher-dimensional lattice. The resulting state is then measured by the photonic quantum processing unit, which employs an array of homodyne detectors to measure the quadrature amplitudes of each mode. The measurement outcomes are processed in real time by a classical decoder that determines the appropriate recovery operations, which are then applied by adjusting the local oscillator phases of subsequent homodyne measurements. This closed-loop architecture enables universal, fault-tolerant quantum computation on a single, planar photonic chip.

### Overview of the System Configuration

- system configuration includes three modules

The system configuration comprises three distinct functional modules: a state preparation module, a multiplexing module, and a main computational module. The state preparation module generates the fundamental quantum resources—Gottesman-Kitaev-Preskill (GKP) qubits and momentum-squeezed vacuum states—using multiplexed Gaussian boson sampling devices. The multiplexing module receives the outputs of these devices and ensures a continuous, high-rate stream of hybrid modes by substituting failed GKP generation events with squeezed vacuum states through fast optical switching. The main computational module then entangles these hybrid modes into a three-dimensional Raussendorf-Harrington-Goyal (RHG) lattice using time-delayed interferometric loops and spatially multiplexed controlled-Z gates. The resulting resource state is measured by a photonic quantum processing unit, which performs homodyne detection and adaptive feed-forward control to execute quantum algorithms. Each module is optimized for its specific function, enabling independent fabrication, scalability, and integration with existing photonic and electronic technologies.

- state preparation module generates bosonic qubits and magic states

The state preparation module is responsible for generating the fundamental quantum resources required for computation: Gottesman-Kitaev-Preskill (GKP) qubits and GKP magic states. This module consists of an array of Gaussian boson sampling (GBS) devices, each fed by momentum-squeezed vacuum inputs and terminated by photon-number-resolving detectors. When a specific photon-number detection pattern is observed, the remaining output mode is heralded as a high-fidelity GKP qubit; if no such pattern is detected, the output is replaced with a momentum-squeezed vacuum state. The GBS devices are designed to produce both computational basis states (|+⟩_GKP) and magic states (|m⟩ = |0⟩_GKP + e^(iπ/4)|1⟩_GKP), which are necessary for implementing non-Clifford gates. The generation of these states is probabilistic, but the module is engineered to maximize success rates through spatial multiplexing and high-efficiency squeezing. The output of this module is a stream of hybrid modes, each either a GKP qubit or a squeezed vacuum state, ready for entanglement in the computational module.

- multiplexing module performs multiplexing of bosonic qubits

The multiplexing module receives the output stream from the state preparation module and ensures a continuous, high-rate supply of hybrid modes by performing spatial and temporal multiplexing. In the spatial domain, multiple GBS devices are arranged in parallel, and their outputs are routed through a binary tree of optical switches that direct successful GKP heraldings to a common output port. In the event that no GKP qubit is generated, a dedicated switch inserts a momentum-squeezed vacuum state in its place. Temporal multiplexing is achieved by interleaving pulses from multiple GBS devices operating at different phases, allowing the system to maintain a high repetition rate compatible with the homodyne detection bandwidth. The multiplexing module thus acts as a buffer and distributor, converting the probabilistic, low-rate outputs of individual GBS devices into a deterministic, high-rate stream of hybrid modes suitable for entanglement in the computational module.

- main computational module stitches together hybrid resource state

The main computational module is responsible for entangling the hybrid modes into a three-dimensional Raussendorf-Harrington-Goyal (RHG) lattice, forming the universal resource for measurement-based quantum computation. This module consists of a two-dimensional array of time-delayed interferometric loops and spatially arranged controlled-Z gates. Each temporal loop delays a mode by one clock cycle, allowing it to interact sequentially with incoming modes via a CZ gate implemented using a beam splitter and two single-mode squeezers. Spatial connectivity is established by interleaving even- and odd-clock-cycle sources and applying CZ gates between adjacent temporal clusters, forming a (2+1)-dimensional lattice. The resulting structure is a hybrid resource state where GKP qubits and squeezed vacuum modes are arranged in a regular lattice, with each mode entangled to its nearest neighbors. This lattice is designed to support fault-tolerant quantum computation through topological error correction, with the hybrid nature of the state enabling robustness against probabilistic GKP generation.

- example system 100 for generating hybrid cluster state

An exemplary implementation of the system, designated as system 100, comprises a planar silicon photonic chip with three integrated layers: a state generation layer, a multiplexing and entanglement layer, and a measurement layer. The state generation layer contains an array of Gaussian boson sampling circuits, each fed by a momentum-squeezed vacuum input and terminated by a transition-edge sensor for photon-number resolution. The outputs of these circuits are routed through a binary tree of Mach-Zehnder interferometers with variable phase shifters, forming the multiplexing layer. Successful GKP heraldings are directed to a common output bus, while failed events trigger the insertion of a squeezed vacuum state from a dedicated source. The output bus feeds into the entanglement layer, which consists of a two-dimensional grid of delay lines and beam splitters arranged to form time-delayed interferometric loops. Each loop interacts with adjacent loops via controlled-Z gates implemented using cascaded squeezers and beam splitters, forming a (2+1)-dimensional RHG lattice. The final layer consists of an array of homodyne detectors, each coupled to a local oscillator with a phase modulator, enabling adaptive measurement settings. The entire system operates at room temperature and is controlled by a classical ASIC that processes homodyne outcomes and updates phase settings in real time.

- state factory generates GKP states

The state factory is a specialized sub-module within the state preparation module dedicated to the generation of GKP qubits and magic states. It consists of multiple identical Gaussian boson sampling circuits, each comprising a network of squeezed vacuum sources, a multi-mode interferometer, and photon-number-resolving detectors. The interferometer is configured to produce a non-Gaussian output state when a specific photon-number detection pattern is observed. The squeezing levels and interferometer parameters are tuned to maximize the fidelity of the generated GKP states, with the target state being a finite-energy approximation of the ideal GKP wavefunction. The state factory operates at a high repetition rate, producing GKP states with a heralding probability of approximately 1–2%, which is then boosted to near-deterministic levels through spatial multiplexing. The output of the state factory is a stream of modes, each either a GKP qubit or a squeezed vacuum state, ready for entanglement in the computational module.

- time stitch implements delay line loops and chains qubits together

The time stitch is a key component of the computational module that implements one-dimensional temporal cluster states by chaining GKP and squeezed modes together using optical delay lines and controlled-Z gates. Each mode is injected into a delay line whose length is precisely tuned to one clock cycle, allowing it to return to an interferometer after a fixed time delay. Upon return, the mode interacts with the next incoming mode via a CZ gate, entangling them in the momentum quadrature. This process repeats for each successive mode, creating a one-dimensional cluster state that extends in time. The time stitch is implemented using integrated silicon waveguides and phase-stable delay lines, ensuring minimal optical loss and high fidelity. The output of the time stitch is a temporally encoded cluster state, which is then fed into the spatial stitch for further entanglement in the transverse dimensions.

- space stitch multiplexes outputs in spatial domain

The space stitch is responsible for extending the one-dimensional temporal cluster states into a two-dimensional lattice by multiplexing their outputs in the spatial domain. Multiple time stitches, each generating a separate temporal cluster, are arranged in a two-dimensional grid. Controlled-Z gates are applied between adjacent temporal clusters during even and odd clock cycles, creating inter-layer entanglement. This interleaved operation generates a (2+1)-dimensional hybrid cluster state, where each node in the lattice corresponds to a mode that is either a GKP qubit or a squeezed vacuum state. The spatial stitching is implemented using a network of beam splitters and phase shifters arranged in a planar photonic circuit, ensuring that the entanglement structure of the Raussendorf-Harrington-Goyal lattice is preserved. The resulting lattice supports fault-tolerant quantum computation and is measured by the photonic quantum processing unit.

- photonic QPU entangles hybrid resource states into higher-dimensional cluster state

The photonic quantum processing unit (QPU) is the final stage of the system and is responsible for measuring the hybrid resource state to perform quantum computation. It consists of an array of homodyne detectors, each coupled to a local oscillator whose phase can be dynamically adjusted via an integrated phase modulator. The homodyne detectors measure the quadrature amplitudes of each mode in the hybrid cluster state, producing continuous-valued outcomes that are digitized and processed by a classical controller. The controller uses a two-tier decoding protocol to interpret these outcomes, determine the appropriate recovery operations, and update the phase settings of subsequent measurements. This closed-loop feedback enables the implementation of adaptive measurement-based quantum gates, including Clifford and non-Clifford operations, by effectively teleporting quantum states through the entangled resource. The QPU operates at room temperature and is fully compatible with high-speed electronics, enabling GHz-class clock rates and real-time error correction.

### Generation of Bosonic Qubits Using Multiplexed Gaussian Boson Sampling (“GBS”) Devices

- generation of non-Gaussian states of light

The generation of non-Gaussian states of light is a critical requirement for universal quantum computation, as Gaussian operations alone are insufficient to achieve computational universality. The invention employs Gaussian boson sampling (GBS) devices to generate non-Gaussian states, specifically Gottesman-Kitaev-Preskill (GKP) qubits, by leveraging the non-Gaussian nature of photon-number-resolving detection. A GBS device consists of a network of squeezed vacuum sources, a multi-mode linear interferometer, and photon-number-resolving detectors on all but one output mode. When a specific photon-number detection pattern is observed, the remaining undetected mode collapses into a high-fidelity GKP state. This process exploits the quantum interference of squeezed light to produce a non-Gaussian output state, even though all input operations are Gaussian. The fidelity of the generated GKP state depends on the squeezing level, interferometer design, and detection efficiency, with current implementations achieving fidelities exceeding 75% for finite-energy approximations.

- multiplexing of GBS devices to obtain high rates and fidelities

To overcome the low heralding probability of individual GBS devices, the invention employs spatial and temporal multiplexing to achieve high generation rates and near-deterministic success probabilities. Multiple GBS devices are arranged in parallel, each producing GKP states independently. Their outputs are routed through a binary tree of optical switches that direct successful heraldings to a common output port. In the event that no GKP state is generated, a momentum-squeezed vacuum state is substituted. This multiplexing strategy increases the effective success probability from a few percent to greater than 99%, while maintaining the fidelity of the generated states. Temporal multiplexing further enhances the rate by interleaving pulses from multiple devices operating at different phases, allowing the system to maintain a high repetition rate compatible with the homodyne detection bandwidth. The combination of spatial and temporal multiplexing enables the system to generate GKP states at rates sufficient for fault-tolerant quantum computation without requiring deterministic sources.

- example of multiplexed state generation

An exemplary multiplexed state generation system consists of 1,024 parallel GBS devices, each fed by a momentum-squeezed vacuum input with 15 dB of squeezing. Each device is terminated by a transition-edge sensor capable of resolving up to seven photons. The outputs of these devices are routed through a seven-layer binary tree of Mach-Zehnder interferometers with variable phase shifters, allowing for fast, low-loss switching between output ports. When a GKP state is heralded, it is directed to the computational module; if no heralding occurs, a squeezed vacuum state is inserted via a dedicated switch. The system operates at a clock rate of 8 MHz, with each GBS device running at 1 GHz, and the temporal multiplexing ensures that the output stream maintains a continuous, high-rate supply of hybrid modes. The overall success probability for GKP generation exceeds 99.5%, with a fidelity of 85% for the generated states, enabling fault-tolerant quantum computation with minimal overhead.

### Time Domain Generation of 1D Clusters

- generation of 1D cluster states using optical delay lines and GKP qubits

One-dimensional cluster states are generated in the time domain by feeding a sequence of hybrid modes—GKP qubits and squeezed vacuum states—into an optical delay line that introduces a fixed temporal delay equal to one clock cycle. As each mode returns to an interferometer after the delay, it interacts with the next incoming mode via a controlled-Z gate, entangling them in the momentum quadrature. This process repeats for each successive mode, creating a one-dimensional cluster state that extends in time. The use of optical delay lines ensures that the entanglement structure is preserved without requiring active switching or complex routing, making the architecture robust and scalable. The resulting 1D cluster state is composed of alternating GKP qubits and squeezed vacuum modes, with each node entangled to its immediate temporal neighbors.

- setup for generation of 1D cluster state

The setup for generating a one-dimensional cluster state consists of a series of optical waveguides, a delay line, and a controlled-Z gate implemented using a beam splitter and two single-mode squeezers. The input stream of hybrid modes is coupled into a looped waveguide whose length is precisely tuned to the time interval between successive pulses. Upon returning to the interferometer, the delayed mode interacts with the next incoming mode via the CZ gate, which is implemented by applying a phase shift and squeezing operation to one mode before and after the beam splitter. The interferometer is designed to be phase-stable and low-loss, with integrated phase shifters for fine-tuning the entanglement strength. The output of the interferometer is fed back into the delay line, creating a continuous loop that generates a long 1D cluster state as new modes are injected.

- operation of interferometer

The interferometer operates by combining two optical modes—a delayed mode and a fresh incoming mode—via a 50/50 beam splitter, followed by a single-mode squeezer applied to one output arm. The squeezer introduces a phase-space rotation that, when combined with the beam splitter, implements a controlled-Z gate in the momentum quadrature. The operation is passive and deterministic, requiring no active switching or external control. The phase of the squeezer is calibrated to ensure that the resulting entanglement matches the desired strength for the Raussendorf-Harrington-Goyal lattice. The interferometer is fabricated on a silicon photonic chip using low-loss waveguides and integrated squeezers, ensuring high fidelity and compatibility with mass production.

- CZ gate implemented by interferometer

The controlled-Z (CZ) gate is implemented by a passive interferometric circuit consisting of a beam splitter and two single-mode squeezers. The first squeezer applies a phase-space rotation to one mode, the beam splitter entangles the two modes, and the second squeezer applies a complementary rotation to restore the correct entanglement structure. This configuration implements the unitary exp(i q₁ q₂), which is the defining operation for creating CV cluster states. The gate is deterministic, requires no photon detection, and operates at the speed of light, making it ideal for high-rate, on-chip implementation. The gate’s fidelity is limited only by the squeezing level and optical loss, both of which are well within the reach of current integrated photonic technology.

- generation of 1D GKP cluster state

The generation of a one-dimensional GKP cluster state proceeds by sequentially injecting hybrid modes—GKP qubits and squeezed vacuum states—into a time-delayed interferometric loop. Each mode interacts with its predecessor via a CZ gate, entangling them in the momentum quadrature. The resulting state is a one-dimensional cluster where each node is either a GKP qubit or a squeezed vacuum state, with entanglement preserved between adjacent nodes. The cluster is stabilized by the deterministic nature of the CZ gates and the continuous injection of modes, allowing for the generation of arbitrarily long cluster states without the need for re-initialization. The fidelity of the cluster is maintained by minimizing optical loss and maximizing the squeezing level, ensuring that the hybrid structure remains compatible with fault-tolerant error correction.

- equivalent spatial representation

The one-dimensional temporal cluster state can be equivalently represented as a spatial lattice by mapping the time dimension onto a spatial axis. Each temporal mode corresponds to a spatial site, and the CZ gates between successive modes become nearest-neighbor entangling operations in the spatial domain. This spatial representation facilitates the design of two-dimensional and three-dimensional lattices by arranging multiple temporal clusters in parallel and coupling them via additional CZ gates. The equivalence between temporal and spatial representations allows the architecture to leverage the well-established theory of cluster state quantum computing while benefiting from the scalability of time-domain generation.

- complete example device for generating 1D cluster

A complete example device for generating a one-dimensional cluster state consists of a silicon photonic chip with a looped waveguide of 10 meters in length, corresponding to a 50-nanosecond delay at a clock rate of 20 GHz. The input stream of hybrid modes is coupled into the loop via a directional coupler, and the delayed mode is recombined with the incoming mode at a 50/50 beam splitter. The beam splitter is preceded and followed by integrated squeezers, forming a passive CZ gate. The output is fed back into the loop, and a new mode is injected every 50 nanoseconds. The system operates at room temperature, requires no active switching, and generates a continuous 1D cluster state with a fidelity exceeding 90%. The entire device is fabricated using standard CMOS-compatible silicon photonics, enabling mass production and integration with high-speed electronics.

### GKP Cluster in 2+1 Dimensions

- stitching together 1D hybrid temporal cluster states

The one-dimensional hybrid temporal cluster states generated by the time stitch are stitched together in the spatial domain to form a two-plus-one-dimensional Raussendorf-Harrington-Goyal (RHG) lattice. Multiple time-delayed clusters are arranged in a two-dimensional grid, with each cluster representing a temporal slice of the lattice. Controlled-Z gates are applied between adjacent clusters during even and odd clock cycles, creating entanglement in the transverse directions. This interleaved operation ensures that each node in the lattice is entangled with its nearest neighbors in both space and time, forming a three-dimensional structure that supports fault-tolerant quantum computation. The stitching process is implemented using a network of beam splitters and phase shifters arranged in a planar photonic circuit, ensuring that the entanglement structure is preserved without introducing optical loss or timing jitter.

- generation of higher-dimensional hybrid lattices

The generation of higher-dimensional hybrid lattices is achieved by extending the two-dimensional spatial arrangement of one-dimensional temporal clusters into a three-dimensional structure. Additional layers of temporal clusters are stacked vertically, and controlled-Z gates are applied between adjacent layers to entangle modes across the third dimension. The resulting lattice is a cubic arrangement of nodes, each of which is either a GKP qubit or a squeezed vacuum state, with entanglement extending in all three directions. This structure is mathematically equivalent to the RHG lattice used in topological quantum error correction and supports the implementation of fault-tolerant logical gates through measurement-based quantum computation. The hybrid nature of the lattice—composed of both GKP and squeezed modes—enables robustness against probabilistic GKP generation while preserving the topological properties required for error correction.

- example 2D chip layout for (2+1)D cluster generator

An exemplary two-dimensional chip layout for the (2+1)-dimensional cluster generator consists of a grid of 100 × 100 time-delayed interferometric loops, each fed by a stream of hybrid modes from the multiplexing module. The loops are arranged in rows and columns, with each row representing a temporal cluster and each column representing a spatial dimension. Between adjacent rows, controlled-Z gates are implemented using beam splitters and squeezers, with even and odd rows operating on alternating clock cycles to avoid crosstalk. The entire layout is fabricated on a single silicon photonic chip using low-loss waveguides and integrated squeezers, with a total footprint of less than one square centimeter. The chip is coupled to a classical control system that manages the timing and phase settings of the homodyne measurements, enabling the generation of a three-dimensional hybrid cluster state with over 10,000 entangled modes.

- 3D cubic lattice generated using 2D chip

The three-dimensional cubic lattice is generated by stacking multiple two-dimensional chip layouts vertically, with each layer representing a different temporal slice of the lattice. The output of each 2D chip is coupled to the input of the next layer via low-loss optical fibers or waveguides, and controlled-Z gates are applied between adjacent layers to entangle modes across the third dimension. The resulting structure is a cubic lattice where each node is entangled with its six nearest neighbors, forming a robust resource for fault-tolerant quantum computation. The use of planar 2D chips enables mass production and scalability, as each chip can be fabricated independently and then stacked to increase the size of the lattice. This modular approach allows the system to scale to millions of qubits without requiring a single, monolithic chip of impractical size.

### Generation of a Hybrid Raussendorf-Harrington-Goyal (“RHG”) Lattice

- generation of Raussendorf lattice

The Raussendorf-Harrington-Goyal (RHG) lattice is a three-dimensional topological structure designed for fault-tolerant quantum computation, composed of alternating primal and dual layers of qubits entangled in a cubic lattice. The invention generates a hybrid version of this lattice by entangling GKP qubits and squeezed vacuum states in the same spatial arrangement, preserving the topological properties required for error correction. The lattice is constructed by stitching together one-dimensional temporal clusters in two spatial dimensions and then stacking multiple layers to form the third dimension. The resulting structure supports the implementation of logical qubits, stabilizer measurements, and fault-tolerant gates through measurement-based quantum computation.

- example representations of two layers of Raussendorf lattice

Two layers of the RHG lattice are represented as a two-dimensional grid of nodes, with one layer designated as primal and the other as dual. The primal layer consists of qubits arranged in a square lattice, while the dual layer is offset by half a lattice spacing, forming a checkerboard pattern. Each node in the primal layer is entangled with its four nearest neighbors and with the corresponding node in the dual layer. The hybrid version of this lattice replaces some nodes with squeezed vacuum states, but the entanglement structure is preserved, ensuring that the topological code remains intact. The two layers are interconnected via controlled-Z gates, forming a three-dimensional structure that supports fault-tolerant quantum computation.

- combined representation of two layers

The combined representation of the two layers is a three-dimensional cubic lattice where each node is either a GKP qubit or a squeezed vacuum state, with entanglement extending in all three spatial dimensions. The primal and dual layers are interleaved such that each node in one layer is surrounded by nodes in the other, forming a structure that supports the measurement of stabilizer operators for topological error correction. The hybrid nature of the lattice—composed of both GKP and squeezed modes—enables robustness against probabilistic GKP generation while preserving the topological properties required for fault tolerance.

- chip layout for generating Raussendorf lattices

The chip layout for generating the RHG lattice consists of a two-dimensional array of time-delayed interferometric loops arranged in a grid, with each loop representing a temporal cluster. The loops are interconnected via controlled-Z gates implemented using beam splitters and squeezers, forming a two-dimensional lattice. Multiple such chips are stacked vertically, with optical interconnects coupling adjacent layers to form the third dimension. The entire structure is fabricated on silicon photonic chips using low-loss waveguides and integrated squeezers, with a total footprint of less than one square centimeter per layer. The layout is designed to minimize optical loss and maximize entanglement fidelity, enabling the generation of large-scale hybrid RHG lattices.

- operation of chip layout

The operation of the chip layout involves the sequential injection of hybrid modes into the time-delayed interferometric loops, where each mode interacts with its predecessor via a controlled-Z gate. The output of each loop is coupled to adjacent loops in the spatial domain, forming a two-dimensional lattice. The process is repeated across multiple layers to generate the third dimension. The entire system operates at room temperature, with homodyne detectors measuring the quadrature amplitudes of each mode in real time. The measurement outcomes are processed by a classical decoder that determines the appropriate recovery operations, enabling fault-tolerant quantum computation.

### Passive Version of System Configuration

- introduce hybrid CV-DV architecture

The invention introduces a hybrid continuous-variable (CV) and discrete-variable (DV) architecture that enables fault-tolerant quantum computation without requiring inline squeezing or active optical switching. The architecture combines the deterministic generation of CV cluster states with the probabilistic heralding of DV GKP qubits, replacing failed GKP generation events with squeezed vacuum states that preserve the entanglement structure. This hybrid approach eliminates the need for active, reconfigurable optical switches, which are a major source of optical loss and thermal noise in prior architectures.

- motivate limitations of existing architectures

Existing photonic quantum computing architectures rely on active, inline squeezing and reconfigurable optical switches to entangle GKP qubits and generate cluster states. These components are difficult to fabricate at scale, introduce significant optical loss, and require cryogenic operation to maintain stability. The need for active switching also limits the clock speed of the system, as the switching time imposes a bottleneck on the rate at which modes can be entangled. Furthermore, the requirement for inline squeezing increases the complexity of the system and makes it incompatible with planar, monolithic photonic integration.

- propose novel architecture without inline squeezing

The novel architecture proposed herein eliminates the need for inline squeezing by using passive interferometric circuits to implement controlled-Z gates. These circuits consist of beam splitters and fixed squeezers that are pre-calibrated to produce the desired entanglement strength, eliminating the need for dynamic control. The system operates entirely in a passive manner, with all entanglement operations performed in a single pass, without the need for feedback or reconfiguration. This design enables the fabrication of the entire system on a single silicon photonic chip using standard CMOS-compatible processes.

- describe generation circuit for fault-tolerant resource states

The generation circuit for fault-tolerant resource states consists of an array of Gaussian boson sampling devices that produce hybrid modes—GKP qubits and squeezed vacuum states—which are then entangled using passive interferometric circuits. The circuit is designed to preserve the entanglement structure of the Raussendorf-Harrington-Goyal lattice while tolerating the probabilistic nature of GKP generation. The output of the circuit is a hybrid resource state that can be measured using homodyne detection to perform fault-tolerant quantum computation.

- explain symmetry of generation circuit

The generation circuit exhibits a high degree of symmetry, with each entangling operation being identical and independent of the surrounding nodes. This symmetry ensures that the noise introduced by the hybrid structure is spatially uniform and can be modeled and corrected using a consistent decoding protocol. The symmetry also simplifies the fabrication process, as identical components can be mass-produced and arranged in a regular pattern.

- calculate logical error rates for different levels of finite squeezing and photon loss

The logical error rates of the system are calculated using Monte Carlo simulations that model the effects of finite squeezing and photon loss on the hybrid resource state. The simulations show that the system achieves a fault-tolerance threshold of 15 dB squeezing and a swap-out probability of 13.3%, which is within the reach of current experimental technology. The error rates decrease exponentially with increasing code distance, demonstrating the scalability of the architecture.

- define GKP encoding for optical bosonic modes

The Gottesman-Kitaev-Preskill (GKP) encoding represents a qubit as a periodic superposition of position and momentum eigenstates, forming a grid in phase space. The encoding is robust to small displacements and can be implemented using Gaussian operations, making it ideal for photonic quantum computing. The GKP qubits are generated using Gaussian boson sampling and are used as the non-Gaussian resource in the hybrid architecture.

- introduce single-mode states within GKP code space

Single-mode states within the GKP code space are represented as wavefunctions that are periodic in both position and momentum, with a lattice spacing of 2√π. These states can be approximated by finite-energy Gaussian envelopes and are used as the computational basis for the hybrid architecture.

- model effects of finite squeezing via additive Gaussian bosonic channel

The effects of finite squeezing are modeled as an additive Gaussian bosonic channel that broadens the GKP wavefunction, introducing noise in both position and momentum quadratures. This noise is incorporated into the error model and is corrected by the two-tier decoding protocol.

- define beamsplitter and phase shifter operations

The beamsplitter and phase shifter operations are passive optical components that implement Gaussian transformations on the optical modes. The beamsplitter entangles two modes, while the phase shifter rotates the phase of a single mode. These operations are used to implement the controlled-Z gates in the hybrid architecture.

- describe homodyne detectors and GKP Pauli measurements

Homodyne detectors measure the quadrature amplitudes of the optical modes and are used to perform GKP Pauli measurements. The measurement outcomes are processed by the decoder to determine the appropriate recovery operations.

- define CV CZ and CX gates for GKP qubits

The continuous-variable CZ and CX gates are implemented using passive interferometric circuits and are used to entangle GKP qubits in the hybrid architecture. These gates are deterministic and require no active switching.

### 3D Hybrid Macronode Architecture

- introduce 3D hybrid macronode architecture

The 3D hybrid macronode architecture is a novel configuration that enhances the fault tolerance of the hybrid system by grouping multiple modes into entangled macronodes. Each macronode consists of four modes—two GKP qubits and two squeezed vacuum states—that are entangled in a dumbbell-shaped configuration. This structure provides redundancy and enables the correction of correlated errors.

- describe primal unit cell of 3D hybrid pair cluster state

The primal unit cell of the 3D hybrid pair cluster state consists of a pair of entangled modes, one GKP qubit and one squeezed vacuum state, arranged in a cubic lattice. The entanglement is generated using controlled-Z gates and is used to form the basis of the hybrid architecture.

- explain generation of 3D hybrid pair cluster state

The 3D hybrid pair cluster state is generated by entangling multiple primal unit cells using controlled-Z gates, forming a three-dimensional lattice. The resulting structure supports fault-tolerant quantum computation and is measured using homodyne detection.

- define two-mode entangled state

The two-mode entangled state is a Bell-like state formed by entangling two optical modes using a controlled-Z gate. The state is used as the building block for the hybrid architecture.

- derive two-mode entangled state from GKP or momentum squeezed vacuum

The two-mode entangled state is derived by applying a controlled-Z gate between a GKP qubit and a momentum-squeezed vacuum state. The resulting state is a hybrid entangled state that preserves the topological properties of the RHG lattice.

- show identities for two-mode cluster states

The identities for two-mode cluster states are derived using symplectic transformations and are used to optimize the entanglement structure of the hybrid architecture.

- obtain state diagram from identities

The state diagram is obtained by mapping the identities onto a graph representation of the hybrid architecture, enabling the visualization of the entanglement structure.

- create magic states by inserting into architecture

Magic states are created by inserting GKP magic states into the hybrid architecture at specific locations, enabling the implementation of non-Clifford gates.

- arrange entangled pairs in 3D configuration

The entangled pairs are arranged in a three-dimensional cubic lattice, with each pair connected to its nearest neighbors. The resulting structure is a robust resource for fault-tolerant quantum computation.

- specify 2D array of sources with probabilities

A two-dimensional array of GBS sources is specified, with each source having a known probability of generating a GKP qubit. The array is used to generate the hybrid resource state.

- apply beamsplitters, delay lines, and phase delays

Beamsplitters, delay lines, and phase delays are applied to entangle the modes and form the hybrid lattice. The operations are passive and deterministic.

- create fully connected 3D resource state

The fully connected 3D resource state is created by entangling all modes in the lattice using controlled-Z gates. The resulting state is a universal resource for measurement-based quantum computation.

- apply homodyne detection

Homodyne detection is applied to measure the quadrature amplitudes of the modes in the hybrid lattice. The measurement outcomes are processed by the decoder to determine the appropriate recovery operations.

- prove equivalence to canonical hybrid cluster state

The equivalence of the 3D hybrid macronode architecture to the canonical hybrid cluster state is proven using symplectic transformations and error correction theory.

- show circuit representation of beamsplitter network

The circuit representation of the beamsplitter network is shown as a planar photonic chip with integrated waveguides and phase shifters.

- apply circuit identities to migrate CZ gates

Circuit identities are applied to migrate controlled-Z gates through the network, optimizing the entanglement structure.

- commute CX gates and squeezers

CX gates and squeezers are commuted to simplify the circuit and reduce the number of required components.

- replace beamsplitters with CX gates and squeezers

Beamsplitters are replaced with CX gates and squeezers to reduce optical loss and improve fidelity.

- economize description of post-measurement state

The post-measurement state is described using a compact representation that captures the essential error structure of the hybrid architecture.

- choose central mode from wires with GKP states

The central mode of each macronode is chosen from the wires containing GKP states, ensuring that the most reliable modes are used for computation.

- describe noise model for single-mode Gaussian bosonic channel

The noise model for the single-mode Gaussian bosonic channel is defined as an additive Gaussian noise that broadens the GKP wavefunction. The noise is modeled as a classical channel and is corrected by the decoder.

- commute photon loss and finite squeezing noise

Photon loss and finite squeezing noise are commuted to simplify the error model and enable efficient correction.

- rescale homodyne outcomes to account for loss

Homodyne outcomes are rescaled to account for photon loss, ensuring that the decoder can accurately interpret the measurement results.

- describe threshold calculations using Monte Carlo simulations

Threshold calculations are performed using Monte Carlo simulations that model the effects of noise on the hybrid architecture. The simulations show that the system achieves a fault-tolerance threshold of 15 dB squeezing and a swap-out probability of 13.3%.

- estimate conditional qubit-level error probabilities

Conditional qubit-level error probabilities are estimated using the decoder output and are used to optimize the error correction protocol.

- discuss improvement in swap-out tolerance

The architecture exhibits improved swap-out tolerance compared to prior systems, with a maximum tolerable swap-out probability of 23.6%.

- introduce 3D hybrid macronode architecture

The 3D hybrid macronode architecture is introduced as a novel configuration that enhances the fault tolerance of the hybrid system by grouping multiple modes into entangled macronodes.

- motivate CZ gates elimination

The elimination of active CZ gates is motivated by the need to reduce optical loss and thermal noise in the system.

- describe inline squeezing degradation

Inline squeezing introduces degradation in the entanglement fidelity due to optical loss and thermal noise. The hybrid architecture eliminates this degradation by using passive entanglement.

- explain circuit identities

Circuit identities are used to simplify the entanglement structure and reduce the number of required components.

- consolidate finite squeezing noise and photon losses

Finite squeezing noise and photon losses are consolidated into a single noise model that is corrected by the decoder.

- reveal built-in redundancy

The macronode architecture reveals built-in redundancy, as each node is entangled with multiple neighbors, enabling the correction of correlated errors.

- describe GKP error correction

GKP error correction is performed by measuring the stabilizers of the GKP code and applying a recovery operation to correct small displacements.

- motivate macronode with multiple GKP states

The use of multiple GKP states in each macronode is motivated by the need to increase the reliability of the system and reduce the impact of probabilistic generation.

- describe entanglement structure

The entanglement structure of the macronode architecture is described as a cubic lattice of dumbbell-shaped entangled pairs.

- illustrate 2D mode layout

The 2D mode layout is illustrated as a grid of macronodes, with each macronode consisting of four modes.

- show 3D arrangement of four-mode macronodes

The 3D arrangement of four-mode macronodes is shown as a cubic lattice, with each macronode connected to its nearest neighbors.

- describe macronode graph edges

The macronode graph edges represent the entanglement between adjacent macronodes, forming a three-dimensional lattice.

- explain CZ gates replacement

CZ gates are replaced with passive interferometric circuits to reduce optical loss and improve fidelity.

- describe passive system configuration

The passive system configuration eliminates the need for active switching and inline squeezing, enabling the fabrication of the entire system on a single silicon photonic chip.

- illustrate Raussendorf lattice generation

The generation of the Raussendorf lattice is illustrated as a three-dimensional cubic lattice of entangled modes.

- modify node configuration

The node configuration is modified to include multiple GKP states and squeezed vacuum states, enhancing the fault tolerance of the system.

- describe beamsplitter interactions

Beamsplitter interactions are used to entangle the modes in the hybrid architecture, forming the entanglement structure of the lattice.

- illustrate passive system configuration chip layout

The passive system configuration chip layout is illustrated as a planar photonic chip with integrated waveguides, phase shifters, and homodyne detectors.

- describe mode pairs and links/edges

The mode pairs and links/edges represent the entanglement structure of the hybrid architecture, with each link corresponding to a controlled-Z gate.

- explain entangled states implementation

The entangled states are implemented using passive interferometric circuits that require no active switching or inline squeezing.

- describe dumbbell-shaped entangled states

The dumbbell-shaped entangled states consist of two GKP qubits and two squeezed vacuum states, arranged in a symmetric configuration.

- illustrate time delayed nodes

Time-delayed nodes are illustrated as optical delay lines that introduce a fixed temporal delay between successive modes.

- describe different types of entangled states

Different types of entangled states are described, including Bell-like states, cluster states, and hybrid states composed of GKP qubits and squeezed vacuum states.

- generalize passive architecture

The passive architecture is generalized to include any combination of GKP qubits and squeezed vacuum states, enabling the system to be adapted to different noise models.

- describe graph transformation

Graph transformation is used to optimize the entanglement structure of the hybrid architecture, reducing the number of required components.

- motivate interferometer construction

The interferometer construction is motivated by the need to implement controlled-Z gates without active switching.

- describe unitary matrix construction

The unitary matrix construction is used to describe the entanglement operations in the hybrid architecture.

- explain beam splitter network construction

The beam splitter network construction is explained as a series of passive optical components that implement the controlled-Z gates.

- describe Fourier transform

The Fourier transform is used to convert between position and momentum representations of the GKP qubits.

- explain beamsplitter operation

The beamsplitter operation is explained as a passive optical component that entangles two modes.

- describe phase shifter operation

The phase shifter operation is described as a passive optical component that rotates the phase of a single mode.

- conclude passive architecture generalization

The passive architecture is generalized to include any combination of GKP qubits and squeezed vacuum states, enabling the system to be adapted to different noise models.

- introduce 3D Hybrid Macronode Architecture

The 3D hybrid macronode architecture is introduced as a novel configuration that enhances the fault tolerance of the hybrid system by grouping multiple modes into entangled macronodes.

- describe quantum error correction

Quantum error correction is performed using a two-tier decoding protocol that combines GKP error correction with topological error correction.

- outline Method 1: Procedure for Performing Fault-Tolerant Quantum Computation

Method 1 outlines the procedure for performing fault-tolerant quantum computation, including state initialization, logical gate implementation, and measurement.

- outline Method 2: Procedure for Performing Quantum Error Correction

Method 2 outlines the procedure for performing quantum error correction, including syndrome identification, matching graph construction, and recovery operation.

- outline Method 3: Example Inner Decoder

Method 3 provides an example of the inner decoder, which converts homodyne measurement outcomes into binary qubit outcomes.

- outline Method 4: Example Outer Decoder

Method 4 provides an example of the outer decoder, which uses minimum-weight perfect matching to correct qubit-level errors.

- outline Method 5: Example Inner Decoder for One p-Squeezed State Surrounded by GKP States

Method 5 provides an example of the inner decoder for a single momentum-squeezed state surrounded by GKP states, using a change-of-basis transformation to account for correlated noise.

## Technological Advantages

### Modularity

- facilitate modular design

The architecture facilitates a modular design by separating the functions of state generation, entanglement, and measurement into distinct, independently optimized modules. Each module can be fabricated on a separate photonic chip using specialized processes, enabling parallel development and incremental upgrades. The state preparation module, which requires high-efficiency photon-number-resolving detectors, can be fabricated on a cryogenic platform, while the computational and measurement modules, which operate at room temperature, can be fabricated using standard CMOS-compatible silicon photonics. This modularity reduces system complexity, enables mass production, and allows for the replacement of individual components as technology advances, such as the transition from cryogenic to room-temperature detectors.

### Minimal Cryogenic Requirements

- reduce cryogenic requirements

The invention significantly reduces cryogenic requirements by confining cryogenic operation to the state preparation module alone, where photon-number-resolving detectors may still require cooling. The computational and measurement modules operate entirely at room temperature, eliminating the need for large-scale cryogenic infrastructure. This modular approach enables the use of small, commercially available cryostats for the state generation module, rather than the massive, custom-built cryogenic systems required by competing architectures. The reduction in cryogenic load lowers operational costs, improves system reliability, and enables the deployment of quantum processors in non-laboratory environments.

### Homodyne Detection Sets Timescales

- set timescales for cluster generation

Homodyne detection sets the timescales for cluster generation by determining the maximum rate at which modes can be measured and processed. Unlike photon-number-resolving detectors, which operate at megahertz rates, homodyne detectors can operate at gigahertz frequencies, enabling the generation of hybrid cluster states at much higher rates. This high-speed operation allows for shorter optical delay lines, reducing optical loss and enabling the fabrication of more compact, scalable systems.

- set timescales for cluster manipulation

The timescales for cluster manipulation are also set by homodyne detection, as the feedback loop that updates the phase settings of the local oscillators must operate at the same rate as the measurement. The gigahertz-class operation of homodyne detectors enables real-time error correction and adaptive measurement-based quantum gates, ensuring that the system can maintain fault tolerance even as the cluster state evolves.

- describe homodyne detection

Homodyne detection measures the quadrature amplitudes of optical modes by interfering them with a local oscillator and detecting the resulting interference pattern. The technique is highly efficient, with near-unity detection efficiency, and can be implemented using standard photodiodes and integrated electronics.

- compare homodyne detection with PNR detection

Homodyne detection is significantly faster than photon-number-resolving (PNR) detection, which is limited by the rise time of transition-edge sensors and superconducting nanowires. While PNR detectors operate at megahertz rates, homodyne detectors can operate at gigahertz rates, enabling much higher clock speeds and shorter delay lines.

- compare homodyne detection with threshold detectors

Homodyne detection provides continuous-valued measurement outcomes, enabling the use of analog information for error correction, whereas threshold detectors provide only binary outcomes. This additional information allows the decoder to distinguish between different types of noise and apply more precise recovery operations.

- describe advantages of faster timescales

Faster timescales enable shorter optical delay lines, reducing optical loss and enabling the fabrication of more compact, scalable systems. They also allow for real-time error correction, ensuring that the system can maintain fault tolerance even as the cluster state evolves.

- describe shorter delay lines

Shorter delay lines reduce the total optical path length, minimizing photon loss and improving the fidelity of the entangled states. This is critical for scaling the system to millions of qubits.

- describe lower losses

Lower losses are achieved by reducing the length of the delay lines and eliminating the need for active switching, which introduces additional loss. The passive nature of the architecture ensures that the majority of the optical path is composed of low-loss waveguides.

- reference FIGS. 2A, 2C, and 4C

The advantages of homodyne detection and shorter delay lines are illustrated in FIGS. 2A, 2C, and 4C, which show the reduction in optical loss and the increase in clock speed compared to prior architectures.

- describe delay lines in FIGS. 2A and 2C

In FIGS. 2A and 2C, the delay lines are shown to be significantly shorter in the present architecture due to the use of homodyne detection, enabling higher clock rates and lower loss.

- introduce method for generating hybrid cluster state

The method for generating the hybrid cluster state involves the sequential injection of hybrid modes into a time-delayed interferometric loop, followed by spatial entanglement using controlled-Z gates.

- receive input vector of homodyne measurements

The system receives an input vector of homodyne measurements from the photonic quantum processing unit, which are processed by the decoder to determine the appropriate recovery operations.

- identify directions with noise levels above threshold

The decoder identifies directions in the measurement space where noise levels exceed a threshold, indicating the presence of correlated errors.

- perform change-of-basis

A change-of-basis transformation is performed to isolate the correlated noise components and reduce their impact on the decoding process.

- apply transformation

The transformation is applied to the homodyne outcomes to project them onto a new basis where the noise is uncorrelated.

- undo change-of-basis

The change-of-basis is undone to return the outcomes to the original basis, but now with reduced noise.

- generate binary string

A binary string is generated by binning the corrected outcomes to the nearest integer multiple of √π, yielding the qubit measurement outcomes.

- describe alternative method for generating hybrid cluster state

An alternative method for generating the hybrid cluster state involves the use of time-multiplexed GBS devices, where pulses from multiple devices are interleaved to increase the generation rate.