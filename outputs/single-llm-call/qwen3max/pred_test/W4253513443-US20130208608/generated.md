# DESCRIPTION

## STATEMENT OF FEDERALLY SPONSORED RESEARCH

This invention was made with government support under Grant No. CNS-0435426 awarded by the National Science Foundation. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates generally to wireless communication systems and, more particularly, to methods and systems for enhancing the performance of multiple-input multiple-output (MIMO), single-input multiple-output (SIMO), and multiple-input single-output (MISO) ad hoc networks through the use of reconfigurable antennas and intelligent antenna configuration selection techniques. Specifically, the invention provides novel architectures for compact pattern-reconfigurable antennas and distributed or centralized algorithms for selecting optimal antenna configurations in multi-link interference-limited environments to maximize network sum capacity without requiring excessive channel feedback or centralized control.

## BACKGROUND

Research in the domain of ad hoc networks has yielded significant advances in physical layer techniques aimed at improving spectral efficiency and mitigating interference. A substantial body of work has focused on the application of smart antennas and antenna diversity techniques to ad hoc networks, leveraging spatial degrees of freedom to enhance link reliability and throughput. Concurrently, medium access control (MAC) protocols tailored for MIMO ad hoc networks have been developed to coordinate transmissions and manage spatial resources effectively. Additionally, adaptive algorithms for antenna beamforming have been proposed to dynamically adjust radiation patterns based on channel state information, thereby optimizing signal reception and reducing co-channel interference.

Directional antennas—such as phased arrays and switchable parasitic element antennas—have been widely advocated as a means to suppress interference from adjacent nodes by focusing energy toward intended receivers. This directional selectivity can significantly improve overall network throughput by spatially isolating concurrent transmissions. To further boost spectral efficiency, MIMO spatial multiplexing and diversity techniques have been integrated into ad hoc network designs, enabling multiple data streams to be transmitted simultaneously over the same frequency band or providing robustness against fading through spatial redundancy.

However, these advanced antenna systems face a critical practical limitation: they are often incompatible with the form factor constraints of compact portable devices. Mounting multiple directional antennas or large MIMO arrays on small handheld or embedded devices is physically challenging due to space limitations, mutual coupling, and increased hardware complexity. This constraint has hindered the widespread adoption of high-performance antenna techniques in real-world mobile ad hoc networks.

In response, reconfigurable antennas have emerged as a promising solution. These antennas can dynamically alter their radiation characteristics—such as pattern, polarization, or resonance—using electronic switches (e.g., PIN diodes), thereby emulating the functionality of multiple fixed antennas within a single compact structure. Prior studies have demonstrated that reconfigurable antennas can enhance channel capacity while occupying minimal physical space, making them ideal for integration into mobile transceivers. Recent efforts have begun to explore configuration selection strategies for reconfigurable antennas in single-link scenarios, but no comprehensive framework exists for their deployment in multi-link MIMO ad hoc networks where interference coupling between links introduces complex interdependencies.

Critically, there remains a gap in the published literature regarding the practical implementation, field testing, and algorithmic optimization of reconfigurable antennas in realistic ad hoc network topologies. Existing approaches often assume idealized channel conditions or centralized control, which are infeasible in decentralized, rapidly changing wireless environments. Thus, there is a pressing need for novel antenna architectures and lightweight, distributed configuration selection methods that can harness the benefits of reconfigurability while operating under the constraints of real-world ad hoc networks.

## SUMMARY

The present invention introduces a MIMO/SIMO/MISO ad hoc network system that employs electrically reconfigurable antennas at network nodes to enhance spectral efficiency and mitigate interference. The core innovation lies in a method for selecting antenna configurations—either centrally or in a distributed manner—to maximize network sum capacity in interference-limited environments. This method accounts for the dynamic interplay between link-specific channel quality and cross-link interference, enabling intelligent adaptation of radiation patterns without requiring full network state knowledge.

The invention describes a configuration selection method wherein each node evaluates its local channel and interference conditions to choose an optimal antenna state from a finite set of predefined configurations. In the centralized approach, a network controller performs an exhaustive search over all possible configuration combinations across all nodes to identify the globally optimal assignment. In contrast, the distributed approach enables each link to independently optimize its own capacity based on locally available channel estimates, iteratively adapting to changes in interference caused by neighboring links.

The performance improvement achieved by the invention is substantial: empirical measurements and simulations demonstrate capacity gains of up to 75% over non-reconfigurable baseline systems. These gains stem from the ability of reconfigurable antennas to decorrelate desired signals from interference through spatial pattern adaptation, effectively creating “virtual” antenna diversity without increasing physical footprint.

An alternative configuration selection scheme restricts reconfigurability to only one side of the communication link—either the receiver (RXRA) or the transmitter (TXRA). The RXRA variant is particularly advantageous, as it eliminates the need for iterative updates and removes the requirement for feedback from receiver to transmitter, thereby reducing overhead and latency. In this mode, maximizing individual link capacity inherently maximizes network sum capacity, rendering the distributed and centralized approaches equivalent.

The invention encompasses various types of reconfigurable antennas, including a reconfigurable printed dipole array (RPDA) with four distinct length-based configurations and a reconfigurable circular patch antenna (RCPA) supporting two orthogonal higher-order modes. Both architectures provide full azimuthal coverage and are designed for operation in the 2.4–2.5 GHz ISM band, making them compatible with standard 802.11-like MIMO systems.

In summary, the invention provides a complete method for selecting antenna configurations in MIMO ad hoc networks using reconfigurable antennas, comprising steps of: (a) defining a finite set of antenna configurations per node; (b) estimating local channel and interference statistics; (c) selecting a configuration that maximizes either individual link capacity (distributed) or global network sum capacity (centralized); and (d) applying the selected configuration to the reconfigurable antenna elements. This method enables significant performance gains while remaining practical for real-world deployment.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

The present patent application discloses illustrative embodiments of a system and method for enhancing the performance of MIMO ad hoc networks through the use of reconfigurable antennas and intelligent configuration selection. The purpose of this detailed description is to enable any person skilled in the art to make and use the invention, and to set forth the best mode contemplated by the inventors for carrying out their invention.

The motivation for employing reconfigurable antennas in MIMO/SIMO/MISO ad hoc networks arises from the need to reconcile high spectral efficiency with compact device form factors. Traditional MIMO arrays require multiple spatially separated antennas, which is impractical in mobile devices. Reconfigurable antennas overcome this by offering multiple radiation states within a single aperture, thereby providing spatial diversity without physical expansion.

Two primary antenna configuration selection schemes are described: centralized and distributed. The centralized scheme assumes a global controller with full channel state information, performing an exhaustive search to maximize sum capacity. The distributed scheme operates autonomously at each link, using only local channel and interference estimates, and may employ iterative updates when transmitters are reconfigurable.

The invention introduces two compact pattern-reconfigurable antenna architectures: the reconfigurable printed dipole array (RPDA) and the reconfigurable circular patch antenna (RCPA). The RPDA uses PIN diodes to switch dipole lengths, yielding four configurations. The RCPA varies the radius of a circular patch to excite orthogonal TM₃₁ and TM₄₁ modes, providing two highly uncorrelated patterns.

Quantifiable benefits of these reconfigurable antennas include increased sum capacity, improved interference mitigation, and enhanced spatial diversity. Measurements in indoor environments show capacity gains of 30–75% over non-reconfigurable baselines, depending on the selection scheme and antenna type.

A distributed selection algorithm is disclosed wherein each link iteratively updates its antenna configuration to maximize its own capacity, responding to interference changes induced by neighbors. This algorithm converges rapidly in practice, typically within three iterations, and requires no inter-node coordination beyond standard channel estimation.

The centralized and distributed approaches are compared in terms of performance, complexity, and practicality. While centralized control offers optimal performance, it is infeasible in large-scale ad hoc networks. The distributed RXRA scheme achieves near-optimal gains with minimal overhead, making it the preferred embodiment for real-world deployment.

### I. RECONFIGURABLE ANTENNA ARCHITECTURES

The invention introduces two compact pattern reconfigurable antennas designed for 2.4–2.5 GHz operation: the reconfigurable printed dipole array (RPDA) and the reconfigurable circular patch antenna (RCPA). Both are engineered to provide multiple radiation states within a minimal footprint, suitable for integration into portable MIMO transceivers.

The RPDA structure consists of two quarter-wavelength-separated microstrip dipoles, each equipped with PIN diode switches that enable electrical reconfiguration of dipole length. Each dipole supports two states: “short” (switches off) and “long” (switches on), resulting in four array configurations: short-short (s-s), long-long (l-l), short-long (s-l), and long-short (l-s). Operation involves biasing the diodes to select the desired geometry, thereby altering mutual coupling and far-field radiation patterns.

The spatial correlation coefficient, denoted r<sub>j,k,l,m</sub>, quantifies the similarity between radiation patterns E<sub>j,k</sub>(Ω) and E<sub>l,m</sub>(Ω) over solid angle Ω, and is defined as the normalized inner product of the patterns. For the RPDA, measured spatial correlation coefficients between the two ports range from 0.45 to 0.70 across configurations, indicating sufficient diversity for MIMO operation. However, correlations between different configurations at the same port exceed 0.80, suggesting limited intra-port diversity.

Radiation efficiency for the RPDA is measured from 3D anechoic chamber data and varies significantly across configurations: s-s achieves 84% efficiency, while l-l drops to 48%, primarily due to PIN diode insertion losses. This efficiency imbalance affects received signal strength and must be accounted for in capacity calculations.

The RCPA structure comprises a single circular patch on a lossy FR4 substrate, with switches that simultaneously alter the effective patch radius to excite either the TM₃₁ (“Mode 3”) or TM₄₁ (“Mode 4”) resonant mode. The antenna is fed via two spatially orthogonal ports, ensuring >20 dB isolation and enabling true 2×2 MIMO operation from a single radiating element. Operation involves toggling all switches to transition between modes, each producing a distinct radiation pattern.

Measured radiation patterns of the RCPA show substantial differences between Mode 3 and Mode 4 in the azimuthal plane, enabling high pattern diversity. Spatial correlation coefficients between the two ports are near zero (<0.1) for both modes, confirming orthogonality. Moreover, the correlation between the two configurations at the same port is only 0.2, indicating exceptional inter-configuration diversity.

Radiation efficiency for the RCPA is lower than the RPDA due to higher-order mode losses and substrate dissipation: Mode 3 achieves 21% efficiency, while Mode 4 drops to 5%. Despite this, the superior pattern diversity compensates in rich-scattering environments.

Comparison of RPDA and RCPA reveals complementary design philosophies: RPDA offers more configurations (four vs. two) and higher efficiency, while RCPA provides greater pattern diversity and orthogonality. Both achieve full 360-degree azimuthal coverage, ensuring reliable signal reception regardless of node orientation.

The diversity gain of the RCPA stems from its near-uncorrelated patterns, which enhance channel matrix rank and capacity. Conversely, the RPDA’s higher radiation efficiency yields stronger received signals, potentially improving SNR but also increasing co-channel interference.

Both antennas guarantee full radiation coverage in the azimuth plane, eliminating blind spots and ensuring robust connectivity in arbitrary network geometries. Signal reception is maintained across all relative orientations of transmitter and receiver, a critical feature for mobile ad hoc networks.

In summary, the disclosed antenna designs balance trade-offs between number of states, pattern diversity, and radiation efficiency, providing flexible building blocks for reconfigurable MIMO ad hoc networks.

### II. SYSTEM MODEL AND NOTATION

The system model assumes an ad hoc network comprising L co-located, single-hop communication links, each consisting of a dedicated transmitter-receiver pair. All links operate simultaneously in the same frequency band, resulting in mutual interference. The network employs either RPDA or RCPA antennas at all nodes, with each antenna capable of switching among a finite set of configurations.

Notation is defined as follows: H<sub>i<sub>rc</sub>,j<sub>tc</sub></sub> denotes the channel matrix between the receiver of link i (using receive configuration i<sub>rc</sub>) and the transmitter of link j (using transmit configuration j<sub>tc</sub>). For RPDA, configurations range from 1 to 4; for RCPA, from 1 to 2. The input-output relationship for link l is given by y<sub>l</sub> = H<sub>l<sub>rc</sub>,l<sub>tc</sub></sub>x<sub>l</sub> + Σ<sub>i≠l</sub> H<sub>l<sub>rc</sub>,i<sub>tc</sub></sub>x<sub>i</sub> + n, where x<sub>l</sub> is the transmitted signal vector and n is additive white Gaussian noise.

The interference-plus-noise covariance matrix for link l is R<sub>l</sub> = Σ<sub>i≠l</sub> H<sub>l<sub>rc</sub>,i<sub>tc</sub></sub>H<sub>l<sub>rc</sub>,i<sub>tc</sub></sub><sup>H</sup> + σ²I, where σ² is the noise power and I is the identity matrix. This matrix depends on the receive configuration of link l and the transmit configurations of all other links.

Equal power allocation is employed, wherein each transmit antenna element is allocated equal power, eliminating the need for channel feedback from receiver to transmitter. Under this scheme, the capacity of link l is C<sub>l</sub> = log₂ det(I + (P<sub>T</sub>/2) H<sub>l<sub>rc</sub>,l<sub>tc</sub></sub><sup>H</sup> R<sub>l</sub>⁻¹ H<sub>l<sub>rc</sub>,l<sub>tc</sub></sub>), where P<sub>T</sub> is total transmit power.

The sum capacity of the network, C<sub>sum</sub> = Σ<sub>l=1</sub><sup>L</sup> C<sub>l</sub>, serves as the primary performance metric, reflecting the aggregate throughput of all interfering links.

Closed-loop MIMO power allocation algorithms, which use channel feedback to optimize power distribution, are noted as potential enhancements but are deemed impractical in this context due to the combinatorial explosion of channel estimates required across all antenna configurations and the dynamic interference landscape. Thus, the invention focuses on open-loop equal power allocation for simplicity and scalability.

### III. ANTENNA CONFIGURATION SELECTION METHODS

The invention considers three cases for reconfigurable antenna deployment: double-side reconfigurable antennas (DSRA), where both transmitter and receiver adapt configurations; receiver-side reconfigurable array (RXRA), where only the receiver adapts; and transmitter-side reconfigurable array (TXRA), where only the transmitter adapts. In RXRA and TXRA, the non-adaptive side is fixed to its most radiation-efficient configuration (s-s for RPDA, Mode 3 for RCPA).

The centralized configuration selection technique assumes a global controller with instantaneous knowledge of all channels H<sub>i<sub>rc</sub>,j<sub>tc</sub></sub>. The controller solves the optimization problem: max<sub>c</sub> C<sub>sum</sub>(c), where c is the vector of all node configurations. This is implemented via exhaustive search over all possible configuration combinations, providing an upper bound on performance.

The distributed configuration selection technique operates locally at each link. Each link l solves max<sub>l<sub>rc</sub>,l<sub>tc</sub></sub> C<sub>l</sub>, using only its own channel H<sub>l<sub>rc</sub>,l<sub>tc</sub></sub> and interference covariance R<sub>l</sub>. When transmitters are reconfigurable (DSRA, TXRA), this induces interference changes that require iterative updates across links, analogous to iterative waterfilling.

The iterative procedure for distributed selection proceeds as follows: each link initializes its configuration, computes its capacity, and updates to the best local configuration. Links repeat this process until convergence or a maximum iteration count (e.g., 10) is reached. In RXRA, no iteration is needed because receive-side changes do not affect other links’ interference.

The RXRA technique offers significant advantages: it eliminates iterative coordination, removes the need for receiver-to-transmitter feedback, and ensures that local optimization aligns with global sum capacity maximization. This makes RXRA highly practical for real-world deployment.

Configuration adaptation at a single side reduces the search space: for RPDA, DSRA has 16 combinations per link, while RXRA/TXRA have only 4. This reduction lowers channel training overhead, as fewer configurations need to be probed, which is beneficial in time-varying channels.

The invention assumes that even in single-side adaptation, both ends possess reconfigurable hardware, since ad hoc nodes may switch roles between transmitter and receiver. However, the non-adaptive role is constrained to the most efficient fixed configuration.

In summary, the antenna configuration selection methods balance performance, complexity, and practicality, with RXRA emerging as the preferred embodiment for scalable, low-overhead operation in decentralized networks.

### IV. DATA COLLECTION

Data collection involved both physical measurements and electromagnetic ray-tracing simulations to evaluate the performance of reconfigurable antennas in a realistic indoor environment. The measurement campaign was conducted on the third floor of the Bossone Research Center at Drexel University, using the HYDRA Software Defined Radio platform, a 2×2 MIMO system operating at 2.4 GHz with 64-subcarrier OFDM.

Two RCPAs and four RPDAs were fabricated with PIN diodes for real-time reconfiguration. Their 3D radiation patterns were measured in an anechoic chamber to characterize spatial correlation and efficiency.

The network topology comprised three transmitters (TX1–TX3) and three receivers (RX1–RX3), forming six distinct link pairings. To capture small-scale fading, receive antennas were mounted on a robotic positioner and moved in λ/10 increments across 40 positions per receiver. At each position, 100 channel estimates per subcarrier were averaged, and interference-limited scenarios were synthesized via superposition.

Sum capacity was computed per subcarrier and averaged over 52 data-carrying subcarriers. Channels were normalized per antenna type so that the maximum expected squared Frobenius norm of any channel matrix equaled 4, accounting for efficiency differences between RPDA and RCPA.

The simulation setup used the FASANT ray-tracing tool with a 3D model of the same hallway. Measured 3D radiation patterns of RPDA and RCPA were integrated into the simulation to accurately model antenna effects. A single tone at 2.484 GHz was used to compute channel matrices, which were then processed identically to measurement data.

Simulated channels underwent the same normalization procedure as measurements. Although the simulation model omitted fine environmental details (e.g., furniture, wall structures), it showed strong agreement with measurements, validating the methodology.

In summary, the data collection process combined empirical rigor with computational modeling to provide a comprehensive assessment of reconfigurable antenna performance under realistic conditions.

### V. RESULTS

Results for the reconfigurable circular patch array (RCPA) demonstrate significant capacity gains. Centralized configuration selection yields up to 75% increase in measured sum capacity over non-reconfigurable baselines, with simulations showing ~50% gains. Distributed RXRA achieves 31% (measured) and 14% (simulated) improvements, highlighting its practical value.

Centralized and distributed schemes exhibit similar performance trends, though measurements show RXRA outperforming TXRA, while simulations show the reverse—likely due to unmodeled environmental complexities.

Convergence for iterative distributed schemes (DSRA, TXRA) is rapid: over 99% of scenarios converge within 10 iterations, with average iterations below 2.5 in most cases.

For the reconfigurable printed dipole array (RPDA), capacity gains are even larger: centralized DSRA achieves up to 80% improvement, while distributed RXRA yields 31% (measured) and 24% (simulated) gains. RPDA consistently outperforms RCPA due to its greater number of configurations and higher efficiency.

However, RPDA’s distributed schemes require more iterations to converge (up to 26% non-convergence in measurements for DSRA), attributed to its larger configuration space.

Direct comparison shows RPDA provides higher absolute and relative capacity gains than RCPA, despite RCPA’s superior pattern diversity. This is because RPDA’s four configurations and balanced efficiency offer more adaptation flexibility.

Analysis of the number of configurations reveals that restricting RPDA to only two states (s-s, l-l) halves its capacity gain, underscoring the importance of configuration count—even with correlated patterns.

A new normalization procedure, which equalizes received power per configuration combination, isolates the effect of pattern correlation. Under this scheme, RCPA’s uncorrelated patterns yield significantly higher capacity than RPDA’s correlated ones, proving that low correlation is a key enabler of reconfigurable antenna gains.

In summary, results confirm that reconfigurable antennas substantially enhance ad hoc network capacity, with performance governed by a triad of factors: number of configurations, pattern correlation, and radiation efficiency balance. The RXRA distributed scheme offers the best trade-off for practical deployment.

### VI. SOFTWARE IMPLEMENTATION

The invention may be implemented in software executed on a general-purpose computing environment. The software comprises computer-executable instructions stored on a non-transitory medium, which, when run, cause a processor to perform the antenna configuration selection methods described herein.

The computing environment includes program modules such as a configuration selector, channel estimator, interference analyzer, and capacity calculator. Data structures store antenna configuration sets, channel matrices, and optimization parameters.

A typical computer system includes a processing unit, system memory (RAM/ROM), and a system bus coupling components. Input/output devices (e.g., network interfaces, displays) enable interaction with the wireless hardware. Storage devices (e.g., hard drives, SSDs) retain program code and data.

The system connects to a network via wired or wireless links, interfacing with remote computers or base stations. In a networked environment, program modules may be distributed across multiple machines, though the core selection logic resides at each node for distributed operation.

In summary, the software implementation enables scalable, real-time execution of the invention’s methods on standard computing hardware integrated with reconfigurable RF front-ends.

### VII. CONCLUSIONS

The performance of the disclosed reconfigurable antenna structures—RPDA and RCPA—is validated through extensive measurements and simulations, demonstrating substantial sum capacity gains in MIMO ad hoc networks. The RPDA’s multiple configurations and higher efficiency yield greater absolute performance, while the RCPA’s uncorrelated patterns offer superior diversity per state.

Key insights into reconfigurable antenna array design emerge: designers should aim for a relatively large number of available configurations, maximize pattern diversity (minimize spatial correlation), and maintain balanced radiation efficiency across states to avoid SNR disparities that degrade performance.

The distributed RXRA configuration selection scheme is shown to provide an excellent balance between performance and practicality, achieving significant capacity gains without requiring iterative coordination or channel feedback, thus making it ideal for real-world ad hoc networks.