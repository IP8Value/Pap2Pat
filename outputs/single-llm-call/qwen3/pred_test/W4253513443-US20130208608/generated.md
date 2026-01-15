# DESCRIPTION

## STATEMENT OF FEDERALLY SPONSORED RESEARCH

- acknowledge government support

The research and development efforts underlying this invention were supported in part by funding provided by the National Science Foundation under Grant Number CNS-0720981. The United States Government holds certain rights in this invention pursuant to the terms and conditions of the aforementioned grant agreement. The views, opinions, and findings contained in this disclosure are those of the inventors and do not necessarily reflect the official policy or position of the National Science Foundation or the United States Government.

## TECHNICAL FIELD

- define technical field

The present invention relates generally to wireless communication systems, and more specifically to multi-input multi-output (MIMO), single-input multi-output (SIMO), and multi-input single-output (MISO) ad hoc networks employing reconfigurable antenna architectures to enhance spectral efficiency, mitigate interference, and improve overall network capacity. The invention pertains to the design, implementation, and dynamic selection of antenna configurations in environments where nodes operate without centralized coordination, and where spatial diversity, radiation pattern adaptability, and efficient power utilization are critical to achieving high-performance communication under constrained physical dimensions and dynamic interference conditions.

## BACKGROUND

- summarize research in ad-hoc networks

Research in ad hoc networks has long focused on optimizing physical layer performance through advanced signal processing techniques, medium access control protocols, and interference management strategies. Early efforts emphasized the use of directional antennas to reduce co-channel interference and increase spatial reuse, particularly in dense network topologies. Subsequent work explored the integration of MIMO techniques to exploit multipath propagation for spatial multiplexing and diversity gains, enabling higher data rates without additional bandwidth or transmit power. These approaches, while effective in controlled environments, often require multiple physical antenna elements, which become impractical in compact, battery-powered devices such as mobile sensors, handheld terminals, and wearable communication systems.

- motivate smart antennas and antenna diversity

Smart antennas and antenna diversity techniques have been proposed as mechanisms to improve link reliability and network throughput by adaptively shaping radiation patterns in response to channel conditions. Phased arrays and switchable parasitic element antennas have demonstrated the ability to focus energy in desired directions and null interference from competing transmitters. However, their implementation typically demands large physical footprints, complex beamforming circuitry, and substantial power consumption, limiting their applicability in resource-constrained ad hoc networks.

- discuss physical layer techniques

Physical layer techniques such as spatial multiplexing, space-time coding, and adaptive modulation have been widely studied in the context of fixed infrastructure networks. Their extension to ad hoc settings has been hindered by the absence of centralized control, the dynamic nature of node mobility, and the lack of reliable channel state information across multiple links. Conventional approaches rely on feedback mechanisms that become prohibitively expensive when applied to systems with numerous interfering links and multiple possible antenna configurations.

- describe medium access control protocols

Medium access control protocols for MIMO ad hoc networks have been developed to coordinate transmission schedules and avoid collisions, often incorporating spatial separation and directional transmission rules. While these protocols improve throughput under ideal conditions, they remain sensitive to channel estimation errors, synchronization delays, and the inability to adapt to rapidly changing interference landscapes. Many existing protocols assume fixed antenna patterns, thereby neglecting the potential for dynamic reconfiguration to alter the effective channel matrix in real time.

- explain adaptive algorithms for antenna beamforming

Adaptive beamforming algorithms have been employed to optimize signal reception by adjusting weights applied to antenna array elements. These algorithms typically require accurate knowledge of the channel matrix and often assume a static or slowly varying environment. In ad hoc networks, where nodes frequently enter and exit the communication range, and where interference sources are unpredictable, such assumptions limit the effectiveness of traditional beamforming techniques.

- discuss limitations of directional antennas

Directional antennas, while effective at reducing interference, suffer from several critical limitations in mobile ad hoc environments. Their narrow beamwidths necessitate precise alignment between transmitter and receiver, which is difficult to maintain under mobility. Furthermore, directional antennas cannot simultaneously optimize both signal reception and interference suppression when multiple interferers are present from different angular directions. Their fixed radiation patterns also prevent adaptation to varying propagation conditions, such as multipath richness or shadowing effects.

- introduce reconfigurable antennas

Reconfigurable antennas offer a compelling alternative by enabling dynamic modification of radiation patterns through electrical tuning mechanisms such as PIN diodes, varactor loads, or microelectromechanical switches. These antennas can alter their resonant frequency, polarization, or directional characteristics without requiring additional physical elements, thereby achieving the benefits of multiple antenna arrays within a compact form factor. Recent studies have demonstrated their utility in single-link scenarios, where pattern diversity enhances capacity and reduces correlation between spatial channels.

- highlight lack of published work on reconfigurable antennas in ad-hoc networks

Despite these advances, no prior work has systematically investigated the deployment of reconfigurable antennas in multi-link MIMO ad hoc networks, where the configuration choice at one node directly influences the interference environment experienced by all other nodes. Existing literature has not addressed the distributed selection of antenna configurations under mutual interference, nor has it quantified the trade-offs between the number of available configurations, pattern diversity, radiation efficiency, and convergence behavior in a network-wide context. This gap in knowledge has hindered the practical adoption of reconfigurable antennas in real-world, decentralized wireless systems.

## SUMMARY

- introduce MIMO/SIMO/MISO ad-hoc network

The present invention introduces a novel framework for enhancing the spectral efficiency of MIMO, SIMO, and MISO ad hoc networks through the use of electrically reconfigurable antennas that dynamically adapt their radiation patterns in response to local channel conditions and network-wide interference. Unlike conventional systems that rely on fixed antenna geometries or centralized control, this invention enables each node to independently select its transmit and/or receive antenna configuration to maximize individual link capacity while contributing to an overall improvement in network sum capacity.

- describe configuration selection method

A configuration selection method is disclosed that determines the optimal antenna state for each node based on locally available channel information, including the estimated channel matrix and interference-plus-noise covariance. This method operates without requiring global knowledge of the network topology or coordinated signaling between nodes, making it suitable for decentralized, infrastructure-less environments.

- explain antenna configuration selection

Antenna configuration selection is performed by evaluating the capacity of a given link under each possible combination of transmit and receive antenna states, using a closed-form expression derived from the Shannon capacity theorem under equal power allocation. The selection process considers not only the direct channel strength but also the impact of interference generated by other nodes, enabling each node to make decisions that balance self-interest with network-wide performance.

- discuss performance improvement

The disclosed system achieves substantial improvements in network sum capacity compared to non-reconfigurable antenna systems, with measured gains exceeding 30% in realistic indoor environments. These improvements are attributed to the increased spatial diversity afforded by multiple distinct radiation patterns, reduced correlation between spatial channels, and the ability to suppress interference through pattern adaptation rather than spatial nulling.

- describe alternative configuration selection

An alternative configuration selection technique is disclosed in which only one end of a communication link—either the transmitter or the receiver—is permitted to reconfigure its antenna, while the other end remains fixed in its most radiation-efficient state. This approach eliminates the need for iterative convergence procedures and removes the requirement for feedback from receiver to transmitter, significantly reducing protocol overhead and implementation complexity.

- list types of reconfigurable antennas

Two distinct reconfigurable antenna architectures are employed in the invention: a reconfigurable printed dipole array and a reconfigurable circular patch array. The dipole array provides four distinct radiation patterns through the switching of PIN diodes along dipole arms, while the circular patch array offers two orthogonal patterns by exciting different electromagnetic modes via dual feedpoints. Both architectures are compact, operate in the 2.4–2.5 GHz band, and provide full azimuthal coverage.

- summarize method for selecting antenna configuration

The method for selecting antenna configuration comprises the steps of: (1) measuring or estimating the channel matrix between a transmitter and receiver pair; (2) computing the interference-plus-noise covariance matrix based on the current configurations of all other transmitters in the network; (3) evaluating the link capacity for each possible combination of transmit and receive antenna states; (4) selecting the configuration pair that maximizes the link capacity; and (5) updating the antenna configuration accordingly. This process may be executed in a centralized manner using global channel knowledge or in a distributed manner using only local channel information, with the distributed approach converging rapidly in practice.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

- introduce patent application

This detailed description presents the structural and operational components of a wireless communication system that employs reconfigurable antennas to dynamically optimize network performance in multi-link ad hoc environments. The system is designed to function without centralized coordination, relying instead on distributed decision-making at each node to adapt antenna configurations in real time based on local channel measurements and interference conditions.

- describe purpose of detailed description

The purpose of this detailed description is to provide a comprehensive and enabling disclosure of the invention, including the specific antenna architectures, system models, configuration selection algorithms, measurement methodologies, and implementation techniques necessary for a person skilled in the art to make and use the invention without undue experimentation. The embodiments described herein are illustrative and not intended to limit the scope of the invention as claimed.

- motivate reconfigurable antennas in MIMO/SIMO/MISO ad-hoc networks

The use of reconfigurable antennas in MIMO, SIMO, and MISO ad hoc networks addresses the fundamental challenge of achieving high spectral efficiency in spatially constrained, interference-limited environments. Traditional MIMO systems require multiple physical antenna elements, which are impractical in portable devices. Reconfigurable antennas overcome this limitation by emulating multiple antenna arrays through electrical tuning, thereby enabling spatial diversity and multiplexing gains without increasing the physical footprint of the device.

- describe antenna configuration selection schemes

Antenna configuration selection schemes are implemented in two primary modes: centralized and distributed. In the centralized mode, a global controller has access to the complete channel state information of all links in the network and exhaustively searches all possible configuration combinations to maximize the total network sum capacity. In the distributed mode, each node independently selects its configuration based solely on its own received channel estimates and the interference-plus-noise covariance matrix, leading to an iterative, self-organizing process that converges to a near-optimal state without requiring global coordination.

- introduce two reconfigurable antenna architectures

Two reconfigurable antenna architectures are disclosed: a reconfigurable printed dipole array and a reconfigurable circular patch array. The printed dipole array consists of two microstrip dipoles, each capable of being electrically lengthened or shortened via PIN diode switches, resulting in four distinct radiation patterns. The circular patch array consists of a single circular patch with two feedpoints, capable of exciting either the TM₃₁ or TM₄₁ mode by activating or deactivating a set of switching elements, yielding two orthogonal radiation patterns.

- quantify benefits of reconfigurable antennas

The benefits of reconfigurable antennas are quantified through measurements and simulations in a realistic indoor environment, demonstrating that the use of reconfigurable antennas increases network sum capacity by up to 75% compared to fixed-configuration systems. The dipole array achieves higher absolute capacity gains due to its greater number of configurations and superior radiation efficiency, while the circular patch array provides superior pattern diversity and lower spatial correlation, making it more effective in highly correlated channel environments.

- describe distributed selection algorithm

The distributed selection algorithm operates as an iterative process in which each node alternately updates its antenna configuration in response to changes in the interference environment caused by other nodes’ configuration updates. The algorithm converges rapidly, with over 99% of cases reaching stability within three iterations, even in scenarios where both transmitter and receiver are reconfigurable. The algorithm requires no feedback from receiver to transmitter, making it suitable for half-duplex operation and reducing protocol overhead.

- compare centralized and distributed approaches

The centralized approach achieves marginally higher sum capacity than the distributed approach, but at the cost of requiring complete and instantaneous knowledge of all channel states across the network, which is impractical in mobile, infrastructure-less environments. The distributed approach, while slightly suboptimal, achieves performance within 5–10% of the centralized solution while requiring only local channel information and no inter-node coordination, making it far more scalable and robust in dynamic settings.

### I. RECONFIGURABLE ANTENNA ARCHITECTURES

- introduce two compact pattern reconfigurable antennas

Two compact pattern reconfigurable antennas are disclosed for use in 2×2 MIMO ad hoc networks operating in the 2.4–2.5 GHz ISM band. Each antenna is designed to be fabricated on a standard printed circuit board, integrated with switching circuitry, and embedded within mobile communication devices without requiring additional space beyond the dimensions of a conventional antenna.

- describe RPDA structure and operation

The reconfigurable printed dipole array (RPDA) comprises two microstrip dipole elements, each approximately half a wavelength in length, separated by a quarter-wavelength distance. Each dipole is segmented into three sections, with PIN diodes placed at the junctions to enable electrical lengthening or shortening of the dipole. When the diodes are biased on, the dipole operates in a “long” configuration; when biased off, it operates in a “short” configuration. This yields four distinct radiation patterns: long-long, short-short, short-long, and long-short.

- define spatial correlation coefficient

The spatial correlation coefficient between two radiation patterns is defined as the normalized inner product of their complex far-field patterns over the full solid angle, accounting for the angular distribution of scattered energy in a rich multipath environment. This coefficient quantifies the degree of similarity between two patterns and is used to predict the level of channel decorrelation achievable between spatial streams.

- calculate spatial correlation coefficients for RPDA

For the RPDA, the spatial correlation coefficient between the two ports across all four configurations ranges from 0.25 to 0.70, indicating moderate to low correlation sufficient for spatial multiplexing. The correlation between different configurations at the same port exceeds 0.80, indicating limited pattern diversity within a single port, which constrains the diversity gain achievable through reconfiguration alone.

- measure radiation efficiency for RPDA

Radiation efficiency for each RPDA configuration was measured using the standard method of integrating the measured 3D far-field patterns over the sphere and comparing the radiated power to the input power. The short-short configuration exhibited the highest efficiency at 84%, while the long-long configuration showed the lowest at 48%, due to increased losses from PIN diode insertion and reduced radiation resistance.

- introduce RCPA structure and operation

The reconfigurable circular patch antenna (RCPA) consists of a single circular patch with a radius of approximately 15 mm, fed by two orthogonal feedpoints located symmetrically along the diameter. The patch is printed on an FR4 substrate with a dielectric constant of 4.4 and a loss tangent of 0.02. PIN diodes are arranged in a radial pattern around the patch perimeter and are used to alter the effective boundary conditions, enabling excitation of either the TM₃₁ or TM₄₁ mode.

- describe radiation patterns of RCPA

The radiation patterns generated by the TM₃₁ and TM₄₁ modes are significantly different in shape and directionality, with the TM₃₁ mode exhibiting a broad, omnidirectional pattern and the TM₄₁ mode producing a four-lobe structure with deep nulls. The patterns are nearly orthogonal in the azimuthal plane, with a measured correlation coefficient of 0.20, indicating high pattern diversity.

- calculate spatial correlation coefficients for RCPA

The spatial correlation coefficient between the two ports of the RCPA is less than 0.10 for both modes, indicating near-perfect spatial decorrelation. Between the two configurations at the same port, the correlation coefficient is 0.20, significantly lower than that of the RPDA, confirming superior pattern diversity.

- measure radiation efficiency for RCPA

The radiation efficiency of the RCPA was measured at 21% for the TM₄₁ mode and 5% for the TM₃₁ mode, primarily due to dielectric losses associated with higher-order mode excitation and substrate losses. The lower efficiency of the TM₃₁ mode is attributed to its larger surface current distribution and greater interaction with the lossy substrate.

- compare RPDA and RCPA

The RPDA and RCPA represent two distinct design philosophies: the RPDA prioritizes a larger number of configurations with moderate pattern diversity and high radiation efficiency, while the RCPA prioritizes high pattern diversity and low spatial correlation at the expense of lower overall efficiency. The RPDA is better suited for environments where signal strength is critical, while the RCPA excels in scenarios dominated by high spatial correlation or dense multipath.

- discuss diversity gain of RCPA

The RCPA provides significantly higher diversity gain than the RPDA due to the low correlation between its two configurations and the near-orthogonality of its radiation patterns. This allows the RCPA to achieve higher capacity gains in environments where channel matrices are highly correlated, such as in line-of-sight or confined indoor spaces.

- discuss radiation efficiency of RPDA

The RPDA’s superior radiation efficiency across all configurations enables stronger received signal power, which directly translates to higher signal-to-noise ratios and improved link capacity. This efficiency advantage compensates for its lower pattern diversity, making it the preferred choice in environments where interference is not the dominant constraint.

- describe full radiation coverage of both antennas

Both the RPDA and RCPA provide full 360-degree radiation coverage in the azimuthal plane, ensuring that communication links remain viable regardless of the relative orientation between transmitter and receiver. This characteristic is essential for mobile ad hoc networks where node positions and orientations are unpredictable.

- discuss signal reception of both antennas

The signal reception capability of both antennas is enhanced by their ability to adapt to varying angular distributions of incoming signals. The RPDA’s multiple configurations allow it to focus reception in directions of strong signal energy, while the RCPA’s orthogonal patterns enable it to capture signals arriving from multiple distinct angular directions simultaneously.

- summarize antenna designs

The two antenna designs presented herein offer complementary advantages: the RPDA delivers higher efficiency and more configuration options, while the RCPA provides superior pattern diversity and lower spatial correlation. Both are suitable for integration into compact MIMO devices and enable dynamic adaptation to changing channel conditions without requiring additional hardware.

### II. SYSTEM MODEL AND NOTATION

- introduce system model

The system model assumes a network of L co-located, single-hop communication links, each consisting of a transmitter and a receiver equipped with two-element reconfigurable antenna arrays. All links operate simultaneously in the same frequency band, resulting in mutual interference. The channel between any transmitter-receiver pair is modeled as a 2×2 complex matrix that varies with the antenna configuration at both ends.

- define notation for ad-hoc network

Let Hᵢᵣ꜀,ⱼₜ꜀ denote the 2×2 channel matrix between the receiver of link i using configuration r꜀ and the transmitter of link j using configuration t꜀. Let xᵢ be the 2×1 transmit vector for link i, and let n be the 2×1 additive white Gaussian noise vector with variance σ². The conjugate transpose operator is denoted by (·)ᴴ.

- describe input-output relationship for link l

The received signal vector at link l is given by yₗ = Hₗᵣ꜀,ₗₜ꜀ xₗ + Σᵢ∈L\l Hₗᵣ꜀,ᵢₜ꜀ xᵢ + n, where the summation term represents the aggregate interference from all other transmitters in the network.

- define interference plus noise covariance matrix

The interference-plus-noise covariance matrix for link l is defined as Rₗ = Σᵢ∈L\l Hₗᵣ꜀,ᵢₜ꜀ Qᵢ Hₗᵣ꜀,ᵢₜ꜀ᴴ + σ²I, where Qᵢ is the transmit covariance matrix for link i, assumed to be equal-power allocation such that Qᵢ = (Pₜ/2)I.

- introduce equal power allocation technique

Equal power allocation is employed to simplify the system model and avoid the complexity of dynamic power control. Under this technique, each transmit antenna element radiates the same power, and the power allocation matrix is diagonal with equal entries. This assumption eliminates the need for feedback from receiver to transmitter and enables closed-form capacity calculations.

- describe capacity of link l

The capacity of link l is given by Cₗ = log₂ det(I + (Pₜ/σ²) Hₗᵣ꜀,ₗₜ꜀ Hₗᵣ꜀,ₗₜ꜀ᴴ Rₗ⁻¹), where the determinant term captures the combined effect of desired signal strength, interference, and noise under the current antenna configuration pair.

- define sum capacity of network

The sum capacity of the network is the total capacity across all L links, defined as Cₛᵤₘ = Σₗ₌₁ᴸ Cₗ. This metric is used as the primary performance indicator for evaluating the effectiveness of different antenna configuration selection schemes.

- discuss closed loop MIMO power allocation algorithms

Closed-loop MIMO power allocation algorithms, which rely on channel state information feedback from receiver to transmitter, are not employed in this invention due to the prohibitive overhead associated with estimating and transmitting channel information for every possible antenna configuration. The equal power allocation scheme avoids this complexity and enables scalable operation in large networks.

### III. ANTENNA CONFIGURATION SELECTION METHODS

- introduce three cases for using reconfigurable antennas

Three operational cases are considered for the use of reconfigurable antennas: double-side reconfigurable antennas (DSRA), where both transmitter and receiver can change configuration; receiver-side reconfigurable antennas (RXRA), where only the receiver adapts; and transmitter-side reconfigurable antennas (TXRA), where only the transmitter adapts.

- describe centralized configuration selection technique

In the centralized configuration selection technique, a global controller has access to the complete channel state information of all links in the network. It exhaustively evaluates all possible combinations of transmit and receive configurations across all nodes and selects the combination that maximizes the total network sum capacity.

- formulate optimization problem for centralized technique

The optimization problem for the centralized technique is formulated as maximizing Cₛᵤₘ over all possible configuration vectors c = [r₁꜀, t₁꜀, r₂꜀, t₂꜀, ..., rₗ꜀, tₗ꜀], subject to the constraint that each configuration is selected from a finite set of discrete states.

- describe distributed configuration selection technique

In the distributed configuration selection technique, each node independently selects its antenna configuration based on locally measured channel information and the interference-plus-noise covariance matrix. The selection is performed iteratively, with each node updating its configuration in response to changes in the interference environment caused by other nodes’ updates.

- formulate optimization problem for distributed technique

The optimization problem for each link l in the distributed technique is to maximize its own capacity Cₗ, given the current configurations of all other links. This is a selfish optimization that does not consider the global sum capacity but converges to a stable state through iterative interaction.

- discuss iterative procedure for distributed technique

The iterative procedure for the distributed technique is analogous to iterative waterfilling, but instead of adjusting power allocation matrices, nodes adjust their antenna configurations. Each node updates its configuration once per iteration, and the process continues until the sum capacity changes by less than a predefined threshold or a maximum number of iterations is reached.

- introduce single side reconfigurable antennas

Single-side reconfigurable antennas refer to systems in which only one end of a link—either the transmitter or the receiver—is permitted to change its antenna configuration. The other end remains fixed in its most radiation-efficient state, simplifying the configuration space and eliminating the need for iterative convergence.

- discuss RXRA technique

The RXRA technique allows only the receiver to adapt its configuration, while the transmitter remains fixed. This approach has the advantage that a change in receiver configuration affects only the link’s own capacity, not the interference seen by other links, thereby eliminating the need for iterative updates.

- discuss advantages of RXRA technique

The RXRA technique offers several advantages: it requires no feedback from receiver to transmitter, it converges in a single step, it reduces the search space for centralized selection, and it ensures that the network sum capacity is maximized whenever each link maximizes its own capacity.

- compare distributed and centralized schemes for RXRA

In the RXRA case, the distributed and centralized schemes are equivalent, as maximizing individual link capacity also maximizes the total network sum capacity. This equivalence does not hold for DSRA or TXRA, where selfish behavior can lead to suboptimal network-wide outcomes.

- discuss configuration adaptation at single side of link

Configuration adaptation at a single side of the link reduces the number of possible configuration combinations from N² to N, where N is the number of available antenna states. For a four-state antenna, this reduces the search space from 16 to 4 configurations per link, significantly lowering computational and training overhead.

- restrict other link end to use most efficient configuration

The non-reconfigurable end of the link is restricted to operate in the configuration that yields the highest radiation efficiency. For the RPDA, this is the short-short configuration; for the RCPA, this is the TM₃₁ mode.

- summarize antenna configuration selection methods

The disclosed antenna configuration selection methods encompass centralized and distributed approaches, applicable to double-side, receiver-side, and transmitter-side reconfigurable antenna systems. The RXRA technique is identified as the most practical and efficient method for real-world deployment due to its simplicity, convergence speed, and equivalence between individual and network optimization.

### IV. DATA COLLECTION

- introduce measurement setup

The performance of the disclosed system was evaluated through a comprehensive measurement campaign conducted in a realistic indoor environment on the third floor of the Bossone Building at Drexel University. The HYDRA software-defined radio platform was used to capture channel responses between multiple transmitter-receiver pairs operating in the 2.4 GHz band.

- describe network topology

The network topology consisted of three transmitters and three receivers, each equipped with two-element reconfigurable antenna arrays. Six distinct link configurations were tested by varying the pairing between transmitters and receivers. Each receiver was mounted on a robotic positioner and moved to 40 distinct locations to capture small-scale fading effects.

- explain measurement procedure

At each location, 100 noisy channel estimates were collected per subcarrier over 52 OFDM subcarriers, averaged to reduce noise, and used to compute the channel matrix for each link. Interference was simulated by superposition of measured channel responses from multiple simultaneous transmissions.

- introduce simulation setup

Electromagnetic ray-tracing simulations were performed using the FAS-ANT software, which modeled the physical layout of the hallway, including walls, floor, and ceiling, using simplified material properties. The measured radiation patterns of the RPDA and RCPA were imported as antenna models.

- describe ray tracing simulation

The ray-tracing simulation transmitted a single tone at 2.484 GHz and computed the channel matrix for each transmitter-receiver pair by tracing the propagation paths of electromagnetic rays reflected and scattered by the environment. The resulting channel matrices were used to compute network sum capacity under each configuration selection scheme.

- explain normalization procedure for measurements

The measured channel matrices were normalized such that the maximum expected squared Frobenius norm of the channel matrix across all configurations and links equaled four. This normalization ensured comparability between the RPDA and RCPA, which differ significantly in radiation efficiency.

- explain normalization procedure for simulations

The simulated channel matrices were similarly normalized using the same criterion, ensuring that differences in performance between the two antenna architectures were attributable to pattern diversity and configuration count rather than absolute signal strength.

- summarize data collection process

The data collection process involved 240 independent measurements per antenna type and configuration scheme, combined with equivalent simulation runs, to generate statistically significant performance metrics for network sum capacity, convergence behavior, and configuration efficiency.

### V. RESULTS

- introduce results for reconfigurable circular patch array

The results for the reconfigurable circular patch array demonstrate that the RCPA achieves substantial gains in network sum capacity compared to fixed-configuration systems, with the highest gains observed under the RXRA configuration selection scheme.

- present sum capacity results for centralized configuration selection

Under centralized configuration selection, the RCPA achieved a 75% increase in expected sum capacity over the non-reconfigurable baseline in measurements, and a 50% increase in simulations.

- present sum capacity results for distributed configuration selection

Under distributed configuration selection, the RCPA achieved a 31% increase in measured sum capacity and a 14% increase in simulated sum capacity under the RXRA scheme, with convergence achieved in fewer than three iterations in over 99% of cases.

- compare results for centralized and distributed configuration selection

The centralized scheme consistently outperformed the distributed scheme, but the performance gap was small, particularly under RXRA, where the two schemes were nearly identical in performance.

- present convergence properties for distributed configuration selection

The distributed configuration selection converged rapidly, with the majority of cases reaching stability within two iterations. Less than 0.1% of cases failed to converge within ten iterations, confirming practical feasibility.

- introduce results for reconfigurable printed dipole array

The reconfigurable printed dipole array demonstrated higher absolute capacity gains than the RCPA, with a maximum measured increase of 80% under centralized DSRA.

- present sum capacity results for centralized configuration selection

For the RPDA, centralized configuration selection yielded an 80% increase in measured sum capacity and a 60% increase in simulated sum capacity.

- present sum capacity results for distributed configuration selection

Distributed configuration selection under RXRA yielded a 31% increase in measured sum capacity and a 24% increase in simulated sum capacity for the RPDA.

- compare results for centralized and distributed configuration selection

The performance gap between centralized and distributed schemes was wider for the RPDA than for the RCPA, due to the larger number of configurations and increased interference sensitivity.

- present convergence properties for distributed configuration selection

The RPDA required more iterations to converge than the RCPA, with 26% of distributed DSRA cases failing to converge within ten iterations in measurements, compared to less than 1% for the RCPA.

- compare results for reconfigurable circular patch array and reconfigurable printed dipole array

The RPDA outperformed the RCPA in absolute capacity gains due to its higher radiation efficiency and greater number of configurations, while the RCPA excelled in environments with high spatial correlation due to its superior pattern diversity.

- analyze effect of number of configurations

When the RPDA was restricted to only two configurations (short-short and long-long), its performance dropped by nearly 50%, demonstrating that the number of available configurations is a critical factor in achieving high capacity gains.

- analyze effect of correlation between patterns

When radiation efficiency was normalized to be equal across configurations, the RCPA outperformed the RPDA, confirming that low spatial correlation between patterns is a key enabler of capacity improvement.

- introduce new normalization procedure

A new normalization procedure was introduced that independently normalized each configuration pair to ensure equal received power, isolating the effect of pattern correlation from radiation efficiency.

- present results for reduced RPDA

Under the new normalization, the reduced RPDA (two configurations) achieved lower sum capacity than the RCPA, despite identical efficiency, due to higher pattern correlation.

- present results for RCPA

The RCPA showed a significant increase in mean sum capacity under the new normalization, confirming that pattern diversity is a dominant factor when efficiency is held constant.

- compare results for reduced RPDA and RCPA

The RCPA’s superior performance under equal-efficiency conditions demonstrates that low correlation between radiation patterns is more impactful than the number of configurations when efficiency is balanced.

- analyze effect of uncorrelated patterns

The analysis confirms that uncorrelated radiation patterns significantly enhance network capacity, even when the number of configurations is limited, and that this effect can outweigh the benefits of additional configurations with high correlation.

- summarize results for reconfigurable circular patch array

The RCPA provides high pattern diversity and low spatial correlation, making it ideal for environments with strong multipath and high interference correlation. Its performance is maximized under RXRA, and it achieves rapid convergence.

- summarize results for reconfigurable printed dipole array

The RPDA provides higher radiation efficiency and more configuration options, resulting in higher absolute capacity gains. It is best suited for environments where signal strength is the primary constraint, and its performance is optimized under centralized selection.

- summarize comparison between reconfigurable circular patch array and reconfigurable printed dipole array

The RPDA and RCPA represent complementary design approaches: the RPDA favors efficiency and configuration count, while the RCPA favors pattern diversity and low correlation. The choice between them depends on the dominant environmental constraints.

- summarize analysis of effect of number of configurations

The number of available configurations directly influences the achievable capacity gain, with diminishing returns observed beyond four configurations. The optimal number of configurations balances diversity, efficiency, and computational complexity.

- summarize analysis of effect of correlation between patterns

Low spatial correlation between radiation patterns is a critical design criterion for reconfigurable antennas in ad hoc networks. Even with fewer configurations, low correlation can outperform high-configuration systems with high correlation.

### VI. SOFTWARE IMPLEMENTATION

- introduce software implementation

The configuration selection algorithms described herein are implemented as computer-executable instructions stored on non-transitory computer-readable media and executed on embedded processing units within wireless communication nodes.

- describe computing environment

The computing environment comprises a microcontroller or digital signal processor capable of performing real-time matrix inversion, determinant computation, and iterative optimization at rates sufficient to update antenna configurations within the coherence time of the channel.

- introduce computer-executable instructions

Computer-executable instructions include modules for channel estimation, interference covariance computation, capacity evaluation for each configuration pair, and configuration selection logic that implements the distributed or centralized algorithm.

- describe program modules

Program modules include a channel estimator, a configuration optimizer, a pattern database storing pre-measured radiation patterns, a convergence monitor, and a control interface to the antenna switching circuitry.

- describe data structures

Data structures include arrays for storing channel matrices, configuration indices, interference covariance matrices, and historical performance metrics used to guide configuration selection.

- introduce computer system

The computer system includes a processing unit, system memory, system bus, input/output devices, storage devices, and network interfaces, all integrated into a compact mobile communication device.

- describe processing unit

The processing unit executes the configuration selection algorithm and performs real-time matrix operations using fixed-point arithmetic to minimize power consumption.

- describe system memory

System memory includes volatile RAM for temporary storage of channel estimates and non-volatile memory for storing antenna pattern databases and algorithm parameters.

- describe system bus

The system bus connects the processing unit to memory, input/output interfaces, and the antenna control circuitry, enabling low-latency configuration updates.

- describe input/output devices

Input/output devices include radio frequency front-ends, analog-to-digital converters, and digital-to-analog converters that interface with the antenna array and enable signal transmission and reception.

- describe storage devices

Storage devices include flash memory for persistent storage of the configuration selection software and pre-characterized radiation pattern data.

- describe network connections

Network connections include wireless interfaces compliant with IEEE 802.11 standards, enabling integration with existing ad hoc network protocols.

- describe remote computer

A remote computer may be used for offline optimization, pattern characterization, or firmware updates, but is not required for real-time operation.

- describe network environment

The network environment is a decentralized, infrastructure-less ad hoc network in which nodes operate independently and communicate without centralized coordination.

- summarize software implementation

The software implementation enables real-time, distributed, and scalable antenna configuration selection with minimal computational overhead, making it suitable for deployment in battery-powered mobile devices.

### VII. CONCLUSIONS

- summarize performance of reconfigurable antenna structures

The performance of reconfigurable antenna structures in MIMO ad hoc networks is significantly enhanced by the ability to dynamically adapt radiation patterns in response to channel conditions. Both the RPDA and RCPA demonstrate substantial improvements in network sum capacity compared to fixed-configuration systems, with gains exceeding 75% in favorable conditions.

- summarize insights into design of reconfigurable antenna arrays

The design of reconfigurable antenna arrays for ad hoc networks must balance three key factors: the number of available configurations, the spatial correlation between radiation patterns, and the radiation efficiency across configurations. High pattern diversity and low correlation are paramount, but must be complemented by balanced efficiency to avoid performance degradation. The RXRA configuration selection technique provides the optimal trade-off between performance, complexity, and practicality.