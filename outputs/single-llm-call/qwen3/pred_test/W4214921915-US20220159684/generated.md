# DESCRIPTION

## TECHNICAL FIELD

- define technical field of invention

The present invention relates to wireless communication systems, particularly to user equipment (UE) configured for millimeter wave (mmWave) frequency bands in fifth-generation (5G) and beyond mobile networks. More specifically, the invention pertains to a method and apparatus for dynamically adjusting the number of active antenna chains in a UE’s phased array antenna system to reduce power consumption and mitigate thermal stress while preserving beam correspondence between downlink and uplink transmission paths. The invention enables the UE to operate in a sub-chain beam mode, wherein only a subset of available antenna elements are activated for transmission, without requiring a full re-sweep of beam directions, thereby maintaining spectral efficiency and minimizing service interruption during thermal events or battery conservation scenarios.

## BACKGROUND

- motivate wireless communication improvements

In modern wireless communication systems operating in the mmWave spectrum, user equipment is typically equipped with large-scale antenna arrays to achieve high-gain directional beams necessary to overcome severe path loss. However, the activation of multiple radio frequency (RF) chains and power amplifiers in such arrays leads to substantial power consumption and localized heating, particularly during sustained uplink transmissions. This thermal burden often forces devices to fallback to sub-6 GHz cellular bands, resulting in significant reductions in data throughput, increased latency, and degraded user experience. Conventional approaches to power reduction, such as lowering transmit power across all active antennas, fail to address the base power draw inherent in each activated power amplifier, rendering such methods inefficient. Furthermore, the 5G standard assumes a strong correspondence between downlink and uplink beam directions, enabling efficient beam management without redundant beam training. Disrupting this correspondence by deactivating antenna elements without proper codebook design necessitates frequent and energy-intensive beam sweeping procedures, which further exacerbate power consumption and delay. There exists a critical need for a system that reduces power and thermal load by selectively deactivating antenna elements while preserving the spatial alignment between downlink reception and uplink transmission beams, thereby eliminating the necessity for re-acquisition of optimal beam pairs during mode transitions.

## SUMMARY

- introduce UE beam codebook design

The present invention introduces a novel user equipment beam codebook design methodology that enables seamless transitions between full-chain and sub-chain beam operation modes without compromising downlink-uplink beam correspondence. The codebook is structured to maintain spatial alignment between beams formed by different numbers of active antenna elements, ensuring that the optimal downlink beam direction corresponds to the optimal uplink beam direction even when the number of active chains is reduced. This alignment is achieved through three distinct design metrics—similarity score maximization, spherical coverage maximization, and beam correspondence spherical coverage maximization—each tailored to different operational priorities such as beam mapping fidelity, spatial coverage efficiency, or a balanced compromise between both.

- describe UE embodiment

The user equipment embodiment comprises a multi-antenna array system with dual-polarized elements arranged in two spatially separated arrays, each connected to a set of RF transceivers and phase shifters. The system is capable of operating in either a full-chain mode, wherein all antenna elements are activated, or a sub-chain mode, wherein only a subset of elements are enabled for transmission. The UE includes a controller that dynamically selects between these modes based on real-time environmental and operational parameters, including device temperature, battery level, signal quality, maximum permissible exposure limits, and network requirements. The controller selects an appropriate beam codebook from a precomputed set of codebooks, each corresponding to a distinct number of active antenna chains, and applies the corresponding beamforming weights to the transmit path without requiring a new beam sweeping procedure.

- describe UE components

The UE includes a plurality of antenna elements, each coupled to a phase shifter and a power amplifier, with the phase shifters constrained to discrete phase values determined by finite-resolution hardware. The RF front-end comprises multiple transmit and receive chains, each associated with a subset of antenna elements. A digital signal processor, coupled to memory storing the precomputed beam codebooks, generates beamforming vectors according to the selected mode. The UE further includes temperature sensors, battery level monitors, signal strength detectors, and a user interface for reporting operational states. The processor is configured to execute software routines that evaluate environmental triggers and select the optimal beam operation mode based on a composite decision function incorporating temperature, power consumption, signal quality, and regulatory constraints.

- describe UE functionality

The UE functionality centers on the ability to switch between full-chain and sub-chain transmission modes while preserving beam correspondence. When transitioning from full-chain to sub-chain mode, the UE selects a sub-chain beam from the codebook whose radiation pattern most closely aligns with the currently active downlink beam, as determined by the last successful beam measurement. This selection ensures that the uplink transmission direction remains spatially consistent with the downlink reception direction, eliminating the need for additional beam training. The UE continuously monitors thermal conditions and adjusts the number of active chains in response to rising temperatures, reducing the number of active chains incrementally to maintain safe operating limits. The system also adapts the number of uplink and downlink chains independently when network conditions permit, optimizing power efficiency without violating beam correspondence constraints.

- introduce method embodiment

The method embodiment comprises a procedure for designing a sub-chain beam codebook that preserves beam correspondence between full-chain and sub-chain transmission modes. The method involves generating a set of candidate beamforming vectors for each possible number of active antenna elements, evaluating each candidate against one or more predefined metrics, and selecting the optimal set of beams that maximizes the desired performance criterion. The method further includes a dynamic selection process executed by the UE to choose between codebooks based on real-time operational parameters, ensuring that beam transitions occur without service degradation or unnecessary signaling overhead.

- describe method steps

The method begins with the acquisition of electromagnetic field response data for each antenna element under realistic terminal housing conditions. A full-chain codebook is generated using a spherical coverage optimization algorithm such as K-Means clustering. Subsequently, for each target number of active antenna elements, a sub-chain codebook is generated using one of three optimization strategies: similarity score maximization, spherical coverage maximization, or beam correspondence spherical coverage maximization. The similarity score metric evaluates the spatial overlap between each sub-chain beam and its corresponding full-chain beam. The spherical coverage metric maximizes the average gain over the entire angular sphere. The beam correspondence spherical coverage metric restricts the optimization to the angular region covered by each full-chain beam, ensuring that the sub-chain beam retains directional alignment. The resulting codebooks are stored in memory and indexed by the number of active chains. During operation, the UE selects the appropriate codebook based on temperature, signal quality, and power constraints, and applies the corresponding beamforming weights to the transmit chain.

- mention other technical features

The invention further includes a temperature-based fallback mechanism that reduces the number of active chains in response to overheating, with the option to revert to full-chain operation when thermal conditions improve. The system supports asymmetric uplink and downlink chain configurations, allowing fewer chains to be used for uplink transmission than for downlink reception, thereby conserving power without compromising downlink performance. The codebook design accommodates dual-polarized antenna arrays with independent activation constraints per polarization. The method also supports dynamic adjustment of beam management parameters, such as reference signal resource configuration and beam indication intervals, to accommodate sub-chain operation without violating 3GPP beam management protocols.

- define certain words and phrases

For the purposes of this disclosure, the term “full-chain beam” refers to a beamformed transmission pattern generated using all available antenna elements and RF chains in the UE’s array. The term “sub-chain beam” refers to a beamformed transmission pattern generated using only a subset of the available antenna elements, with the remaining elements deactivated. The term “beam correspondence” refers to the spatial alignment between the optimal downlink reception beam and the optimal uplink transmission beam, such that the same angular direction yields maximum gain in both directions. The term “codebook” refers to a precomputed set of beamforming vectors, each associated with a specific number of active antenna elements and indexed for rapid selection during operation. The term “beam correspondence spherical coverage” refers to a design metric that optimizes the radiation pattern of a sub-chain beam within the angular region dominated by its corresponding full-chain beam, thereby preserving directional alignment while maximizing coverage efficiency.

## DETAILED DESCRIPTION

- introduce 5G communication systems

Fifth-generation wireless communication systems operate across a wide range of frequency bands, including millimeter wave frequencies from 24 GHz to 100 GHz, where high bandwidth enables multi-gigabit-per-second data rates. These systems rely on massive multiple-input multiple-output (MIMO) architectures and directional beamforming to overcome the high path loss inherent in mmWave propagation. The 3GPP standards define a comprehensive beam management framework that enables the UE and base station to identify, track, and switch between optimal beam pairs for reliable communication. This framework assumes a strong correspondence between downlink and uplink beam directions, allowing the UE to infer the optimal uplink beam from downlink measurements without requiring separate uplink beam training. Disruption of this correspondence due to hardware constraints, such as thermal throttling or power-saving modes, introduces inefficiencies that degrade system performance.

- motivate beam-specific operations

Beam-specific operations are essential in mmWave systems because the narrow beamwidths necessitate precise spatial alignment between transmitter and receiver. Unlike sub-6 GHz systems, where omnidirectional or wide beams suffice, mmWave systems require focused beams to achieve sufficient signal-to-noise ratio. The energy efficiency of these systems is heavily dependent on minimizing the number of active RF chains and power amplifiers, especially during uplink transmission, which consumes significantly more power than reception. Traditional approaches to power reduction, such as power scaling, do not reduce the base power consumption of active amplifiers, making them suboptimal. The present invention addresses this limitation by enabling selective deactivation of antenna elements, thereby reducing both transmitted and base power consumption while preserving beam alignment.

- describe frequency bands for 5G communication systems

The 5G communication systems operate in frequency bands ranging from below 6 GHz to millimeter wave bands above 24 GHz, with particular emphasis on the 24.25–29.5 GHz, 37–43.5 GHz, and 64–71 GHz bands for high-capacity applications. These bands are characterized by high atmospheric attenuation and limited diffraction, necessitating highly directional beamforming. The antenna arrays used in user equipment within these bands are typically composed of multiple dual-polarized patch elements arranged in linear or planar configurations, with each element connected to a phase shifter and power amplifier. The physical constraints of mobile devices limit the number of available RF chains, making it necessary to design codebooks that allow operation with fewer chains without sacrificing spatial performance.

- discuss beamforming and massive MIMO techniques

Beamforming in mmWave systems involves applying complex weights to the signals transmitted or received by each antenna element to steer the radiation pattern in a desired direction. Massive MIMO techniques extend this concept by utilizing large antenna arrays to form multiple simultaneous beams, increasing spectral efficiency and spatial multiplexing gains. In the context of this invention, beamforming is implemented using discrete-phase shifters with limited resolution, and the beamforming weights are constrained to binary activation states (on or off) for each antenna element. This simplifies the hardware design and reduces power consumption compared to continuous amplitude control, while still enabling high-gain directional transmission.

- introduce system network improvement developments

Recent developments in 5G network architecture have introduced dynamic beam management procedures that adapt to changing channel conditions, user mobility, and environmental obstructions. These procedures include beam sweeping, beam refinement, and beam failure recovery, all of which rely on the assumption of downlink-uplink beam correspondence. The present invention enhances these procedures by enabling the UE to operate in sub-chain modes without triggering beam failure or requiring additional signaling, thereby reducing network overhead and improving reliability.

- describe duplex method for DL and UL signaling

The 5G system employs both frequency division duplex (FDD) and time division duplex (TDD) schemes for downlink and uplink signaling. In TDD mode, the same frequency band is used for both directions, and beam correspondence is critical for efficient operation. The invention is particularly suited for TDD systems, where the assumption of reciprocity between downlink and uplink channels enables the UE to reuse downlink beam measurements for uplink transmission. By preserving beam correspondence during sub-chain operation, the invention ensures that this reciprocity remains valid, even when the number of active chains is reduced.

- discuss OFDM and OFDMA communication techniques

Orthogonal frequency division multiplexing (OFDM) and orthogonal frequency division multiple access (OFDMA) are the foundational modulation techniques used in 5G systems to divide the available bandwidth into multiple subcarriers, enabling robust transmission in multipath environments. The beamforming vectors generated by the invention are applied to the OFDM symbols prior to upconversion and transmission, ensuring that the spatial directionality of the signal is preserved across all subcarriers. The codebook design is independent of the modulation scheme, allowing seamless integration with existing OFDMA-based physical layer implementations.

- introduce various embodiments of the present disclosure

The present disclosure encompasses multiple embodiments, including a UE with dual-polarized antenna arrays, a method for generating sub-chain beam codebooks using similarity score, spherical coverage, or beam correspondence spherical coverage metrics, and a dynamic mode selection algorithm that adjusts the number of active chains based on temperature, battery level, and signal quality. Each embodiment is designed to operate within the constraints of 3GPP 5G standards while introducing novel power-saving and thermal-management capabilities.

- describe FIG. 1, an example wireless network

FIG. 1 illustrates a wireless communication network comprising a base station equipped with a large-scale antenna array and multiple user equipment devices, each with dual mmWave antenna arrays on opposing edges. The base station transmits downlink beams to the UEs, which receive and measure the beam quality. The UEs then transmit uplink beams back to the base station using a subset of their antenna elements, selected from a precomputed codebook to maintain beam correspondence. The figure demonstrates how the UE can transition between full-chain and sub-chain modes without requiring a new beam sweep, as the sub-chain beam aligns spatially with the previously identified downlink beam.

- introduce gNB and UE components

The gNodeB (gNB) includes a massive MIMO antenna array, multiple RF transceivers, a baseband processor, and a beam management controller. The UE includes a dual-polarized antenna array, a set of phase shifters, power amplifiers, RF front-end circuitry, a digital signal processor, memory storing precomputed codebooks, temperature sensors, and a power management unit. The UE’s processor is configured to select the appropriate beam codebook based on real-time environmental inputs and to apply the corresponding beamforming weights to the transmit chain.

- describe coverage areas of gNBs

Each gNB covers a specific geographic area, within which it maintains multiple directional beams to serve multiple UEs simultaneously. The coverage area is divided into angular sectors, each served by a distinct beam. The UE, when located within a sector, measures the downlink beam quality and selects the optimal beam direction. The invention ensures that the uplink beam selected by the UE corresponds to this same direction, even when operating in sub-chain mode.

- introduce 2D antenna arrays

The antenna arrays employed in the invention are two-dimensional, comprising multiple rows and columns of dual-polarized patch elements. This configuration allows for independent control of azimuth and elevation beam steering, enabling precise spatial focusing. The codebook design accounts for the two-dimensional radiation patterns of these arrays, ensuring that the sub-chain beams maintain directional fidelity in both planes.

- describe codebook design and structure

The codebook is structured as a hierarchical set of beamforming vectors, each indexed by the number of active antenna elements and polarization configuration. Each vector contains discrete phase values for each antenna element, constrained by the resolution of the phase shifters. The codebook is precomputed offline using electromagnetic simulations and stored in non-volatile memory within the UE. During operation, the UE selects the appropriate codebook based on the current operational mode and applies the corresponding weights to the transmit chain.

- introduce sub-chain beam codebook design and operation

The sub-chain beam codebook design enables the UE to operate with fewer active antenna elements than the full array, while maintaining spatial alignment with the downlink beam. The codebook is generated using one of three optimization methods: similarity score maximization, spherical coverage maximization, or beam correspondence spherical coverage maximization. Each method produces a distinct set of beamforming vectors optimized for different operational priorities. The operation of the sub-chain codebook involves selecting the appropriate beam from the codebook based on the last measured downlink beam direction and applying the corresponding weights to the transmit chain.

- describe FIG. 2, an example gNB

FIG. 2 depicts an example gNB with a large-scale antenna array composed of multiple antenna panels, each containing hundreds of dual-polarized elements. The gNB includes multiple RF transceivers, each connected to a subset of antenna elements, and a baseband processor that generates beamforming vectors for downlink transmission. The gNB also includes a beam management controller that receives feedback from UEs regarding beam quality and adjusts beam parameters accordingly.

- introduce multiple antennas and RF transceivers

The UE includes multiple antenna elements arranged in a dual-polarized configuration, with each polarization connected to a separate set of RF transceivers. The number of RF transceivers is less than the number of antenna elements, necessitating the use of sub-chain operation to reduce power consumption. The RF transceivers include low-noise amplifiers, mixers, filters, and power amplifiers, each contributing to the overall power budget of the device.

- describe TX and RX processing circuitry

The transmit processing circuitry includes a channel encoder, modulator, serial-to-parallel converter, inverse fast Fourier transform (IFFT) block, parallel-to-serial converter, cyclic prefix inserter, and up-converter. The receive processing circuitry includes a down-converter, cyclic prefix remover, fast Fourier transform (FFT) block, parallel-to-serial converter, demodulator, and channel decoder. The beamforming weights generated by the codebook are applied to the transmit and receive chains prior to upconversion and after downconversion, respectively.

- introduce controller/processor and memory components

The controller and processor are implemented as a system-on-chip (SoC) that executes firmware and software routines for beam codebook selection, thermal management, and power control. The memory stores the precomputed codebooks, beam measurement history, temperature thresholds, and operational parameters. The processor is configured to dynamically switch between codebooks based on real-time inputs from sensors and network commands.

- describe backhaul or network interface

The backhaul interface connects the UE to the core network via the gNB, enabling the exchange of control signals, beam measurement reports, and configuration commands. The interface supports 3GPP-defined protocols for beam management, including beam indication, beam failure recovery, and reference signal configuration.

- introduce beam forming or directional routing operations

Beam forming operations involve the application of complex weights to antenna elements to steer the radiation pattern in a desired direction. Directional routing refers to the process of selecting the optimal beam pair between the UE and gNB based on channel measurements. The invention enhances these operations by enabling seamless transitions between full-chain and sub-chain modes without requiring re-sweeping.

- describe sub-chain beam codebook design and operation

The sub-chain beam codebook design is based on three distinct optimization criteria: similarity score, spherical coverage, and beam correspondence spherical coverage. The similarity score maximizes the spatial overlap between each sub-chain beam and its corresponding full-chain beam. The spherical coverage maximizes the average gain over the entire angular sphere. The beam correspondence spherical coverage restricts the optimization to the angular region dominated by each full-chain beam, ensuring that the sub-chain beam remains aligned with the downlink beam direction. The operation of the codebook involves selecting the appropriate beam based on the last measured downlink beam and applying the corresponding weights to the transmit chain.

- introduce BIS algorithm and process

The Beam Index Selection (BIS) algorithm is a procedure used by the UE to determine the optimal beam index for transmission based on the last received downlink beam measurement. The algorithm compares the measured downlink beam index with the precomputed codebook entries and selects the corresponding sub-chain beam index. This process eliminates the need for additional beam training during mode transitions.

- describe FIG. 3, an example UE

FIG. 3 illustrates an example UE with dual mmWave antenna arrays on opposing edges, each connected to a set of RF transceivers and phase shifters. The UE includes a touchscreen display, microphone, speaker, I/O interface, processor, memory, temperature sensors, and battery level monitor. The processor executes the codebook selection algorithm and applies the appropriate beamforming weights to the transmit chain.

- introduce antenna, RF transceiver, and TX/RX processing circuitry

The antenna array consists of dual-polarized patch elements arranged in a linear configuration. Each antenna element is connected to a phase shifter and power amplifier, with the phase shifters constrained to discrete values. The RF transceivers include upconverters and downconverters for frequency translation, and the TX/RX processing circuitry applies the beamforming weights to the OFDM symbols prior to transmission and after reception.

- describe microphone, speaker, and I/O interface

The microphone and speaker enable voice communication, while the I/O interface supports peripheral connectivity such as USB, Bluetooth, and Wi-Fi. These components are managed by the processor, which prioritizes power consumption based on user activity and system state.

- introduce processor and memory components

The processor is a multi-core system-on-chip that executes the operating system, application software, and beam management firmware. The memory includes volatile and non-volatile storage for codebooks, beam measurement history, and system parameters. The processor is configured to dynamically adjust the number of active chains based on temperature, battery level, and signal quality.

- describe touchscreen and display components

The touchscreen and display provide user interaction and visual feedback on network status, battery life, and thermal conditions. The display may indicate when the device is operating in sub-chain mode due to high temperature or low battery.

- introduce OS and applications

The operating system manages hardware resources, including the antenna array, RF transceivers, and sensors. Applications such as video streaming, gaming, and augmented reality trigger high data rate demands, which may necessitate full-chain operation. The OS coordinates with the beam management firmware to balance performance and power consumption.

- describe UL transmission on uplink channel

Uplink transmission is performed using a subset of the available antenna elements, selected from the precomputed codebook to maintain beam correspondence with the downlink beam. The number of active elements is dynamically adjusted based on thermal and power constraints, with the beamforming weights applied prior to upconversion.

- introduce I/O interface and accessories

The I/O interface supports external accessories such as headsets, chargers, and diagnostic tools. The accessory detection circuitry may influence power management decisions, such as reducing transmit power when a wired headset is connected.

- describe FIG. 4A, transmit path circuitry

FIG. 4A illustrates the transmit path circuitry, including the channel encoder, modulator, serial-to-parallel converter, IFFT block, parallel-to-serial converter, cyclic prefix inserter, and up-converter. The beamforming weights are applied to the output of the IFFT block before upconversion.

- introduce channel coding and modulation block

The channel coding and modulation block encodes the data stream using LDPC or polar codes and modulates it using QPSK, 16-QAM, or 64-QAM. The output is a complex symbol stream that is processed by the subsequent blocks.

- describe S-to-P and IFFT blocks

The serial-to-parallel (S-to-P) block converts the serial symbol stream into parallel streams for processing by the IFFT block, which performs an inverse Fourier transform to convert the frequency-domain symbols into time-domain samples.

- introduce P-to-S and add cyclic prefix blocks

The parallel-to-serial (P-to-S) block converts the time-domain samples back into a serial stream, and the cyclic prefix inserter adds a guard interval to mitigate inter-symbol interference.

- describe up-converter block

The up-converter shifts the baseband signal to the mmWave carrier frequency using a local oscillator and mixer. The beamforming weights are applied prior to upconversion to ensure spatial directionality.

- introduce FIG. 4B, receive path circuitry

FIG. 4B illustrates the receive path circuitry, including the down-converter, cyclic prefix remover, FFT block, P-to-S block, channel decoder, and demodulator. The beamforming weights are applied to the received signal prior to downconversion.

- describe down-converter block

The down-converter shifts the received mmWave signal to baseband using a local oscillator and mixer. The beamforming weights are applied to the analog signal prior to downconversion to enhance signal reception.

- introduce remove cyclic prefix and S-to-P blocks

The cyclic prefix remover discards the guard interval, and the S-to-P block converts the serial signal into parallel streams for processing by the FFT block.

- describe Size N FFT block

The FFT block performs a fast Fourier transform on the time-domain samples to convert them into frequency-domain symbols for demodulation.

- introduce P-to-S and channel decoding and demodulation blocks

The P-to-S block converts the parallel symbols into a serial stream, which is then decoded and demodulated to recover the original data.

- conclude detailed description

The detailed description of the invention encompasses all components, methods, and operational procedures necessary to implement a user equipment system capable of dynamic sub-chain beam operation while preserving downlink-uplink beam correspondence. The invention integrates hardware, firmware, and algorithmic innovations to achieve significant power savings and thermal management without compromising communication performance.

- describe configurable hardware and software components

The hardware components, including antenna elements, phase shifters, and RF transceivers, are configurable to support multiple beam codebooks. The software components include firmware for codebook selection, thermal management, and beam measurement processing. The system is designed to be backward compatible with existing 3GPP standards.

- introduce FFT and IFFT blocks

The FFT and IFFT blocks are implemented using digital signal processing units integrated into the baseband processor. These blocks operate on complex symbols and are configured to support variable transform sizes depending on the number of active subcarriers.

- describe transmit path circuitry

The transmit path circuitry includes all components from channel encoding to upconversion, with beamforming weights applied after the IFFT stage to ensure spatial directionality.

- describe channel coding and modulation

Channel coding employs LDPC or polar codes for error correction, while modulation uses QPSK, 16-QAM, or 64-QAM depending on channel conditions. The modulation scheme is selected by the physical layer scheduler based on signal quality.

- describe serial-to-parallel conversion

Serial-to-parallel conversion divides the symbol stream into multiple parallel streams for simultaneous processing by the IFFT block, enabling efficient use of the available subcarriers.

- describe IFFT operation

The IFFT operation transforms frequency-domain symbols into time-domain samples, with the beamforming weights applied to each sample to steer the radiation pattern in the desired direction.

- describe parallel-to-serial conversion

Parallel-to-serial conversion recombines the time-domain samples into a single stream for transmission, with the cyclic prefix inserted to mitigate inter-symbol interference.

- describe cyclic prefix insertion

Cyclic prefix insertion adds a guard interval at the beginning of each symbol to prevent inter-symbol interference caused by multipath propagation.

- describe up-conversion

Up-conversion shifts the baseband signal to the mmWave carrier frequency using a local oscillator and mixer, with the beamforming weights applied prior to this stage to preserve spatial directionality.

- describe receive path circuitry

The receive path circuitry includes downconversion, cyclic prefix removal, FFT, and demodulation stages, with beamforming weights applied to the analog signal prior to downconversion to enhance reception.

- describe down-conversion

Down-conversion shifts the mmWave signal to baseband using a local oscillator and mixer, with the beamforming weights applied to the received signal to enhance signal-to-noise ratio.

- describe cyclic prefix removal

Cyclic prefix removal discards the guard interval added during transmission, leaving only the useful symbol data for processing.

- describe serial-to-parallel conversion

Serial-to-parallel conversion divides the received time-domain signal into multiple parallel streams for processing by the FFT block.

- describe FFT operation

The FFT operation transforms the time-domain samples into frequency-domain symbols for demodulation, with the beamforming weights applied prior to this stage to enhance signal reception.

- describe parallel-to-serial conversion

Parallel-to-serial conversion recombines the frequency-domain symbols into a single stream for channel decoding and demodulation.

- describe channel decoding and demodulation

Channel decoding corrects errors introduced during transmission using LDPC or polar codes, while demodulation recovers the original data bits from the modulated symbols.

- describe 5G communication system use cases

The invention is applicable to all 5G use cases, including enhanced mobile broadband (eMBB), ultra-reliable low-latency communication (URLLC), and massive machine-type communication (mMTC). In eMBB scenarios, the invention enables sustained high data rates without thermal throttling. In URLLC scenarios, it ensures reliable beam maintenance during critical transmissions. In mMTC scenarios, it extends battery life for low-power IoT devices.

- describe eMBB use case

In enhanced mobile broadband use cases, users demand high data rates for video streaming, gaming, and virtual reality. The invention enables continuous mmWave operation by dynamically reducing the number of active chains during thermal stress, avoiding fallback to sub-6 GHz and preserving throughput.

- describe URLL use case

In ultra-reliable low-latency communication use cases, such as autonomous driving and industrial automation, beam reliability is critical. The invention ensures that beam correspondence is maintained during mode transitions, preventing beam failure and minimizing latency.

- describe mMTC use case

In massive machine-type communication use cases, devices operate on battery power for extended periods. The invention reduces power consumption by activating only the necessary number of chains, extending battery life without compromising connectivity.

- describe communication system architecture

The communication system architecture includes the UE, gNB, core network, and backhaul links. The invention operates within the 3GPP-defined architecture, with no modifications required to the network infrastructure.

- describe downlink signals

Downlink signals are transmitted from the gNB to the UE using directional beams. The UE measures the quality of each downlink beam and reports the best beam index to the gNB.

- describe uplink signals

Uplink signals are transmitted from the UE to the gNB using a sub-chain beam selected from the precomputed codebook to maintain correspondence with the best downlink beam.

- describe resource allocation

Resource allocation is managed by the gNB scheduler, which assigns time-frequency resources to UEs based on channel conditions and QoS requirements. The invention does not alter the resource allocation mechanism but enhances its efficiency by reducing the need for beam retraining.

- describe antenna panel architecture

The antenna panel architecture consists of multiple antenna elements arranged in a linear or planar configuration, with each element connected to a phase shifter and power amplifier. The panels are mounted on opposing edges of the UE to provide spatial diversity.

- describe multi-beam operation

Multi-beam operation allows the gNB to serve multiple UEs simultaneously using different beams. The invention enables the UE to operate in sub-chain mode without disrupting multi-beam coordination.

- describe antenna block architecture

The antenna block architecture includes the physical antenna elements, phase shifters, power amplifiers, and RF front-end circuitry. The architecture is designed to support dynamic activation and deactivation of antenna elements.

- describe quasi co-located antenna ports

Quasi co-located antenna ports refer to antenna elements that share similar spatial characteristics, enabling the UE to infer beam properties across ports. The invention leverages this property to maintain beam correspondence during sub-chain operation.

- introduce UE configuration

The UE configuration includes the number of antenna elements, RF chains, phase shifter resolution, and codebook size. The configuration is fixed at manufacturing but supports dynamic codebook selection during operation.

- describe TCI-State configurations

TCI-State configurations define the relationship between downlink and uplink beams. The invention ensures that TCI-State configurations remain valid during sub-chain mode transitions.

- explain QCL relationship configuration

Quasi co-location (QCL) relationships define the spatial and temporal characteristics shared between reference signals. The invention preserves QCL relationships by maintaining beam correspondence during sub-chain operation.

- detail MAC-CE activation command

The MAC-CE activation command is used by the gNB to instruct the UE to change its beam configuration. The invention supports these commands and responds by selecting the appropriate sub-chain codebook.

- motivate multi-beam operation

Multi-beam operation increases system capacity by serving multiple users simultaneously. The invention enhances multi-beam operation by reducing the power consumption of each UE without compromising beam alignment.

- describe beam training and measurement procedure

Beam training involves the transmission of reference signals to measure beam quality. The invention reduces the frequency of beam training by maintaining beam correspondence during mode transitions.

- explain beam indication procedure

The beam indication procedure is used by the gNB to inform the UE of the optimal beam pair. The invention ensures that the indicated beam remains optimal even when the UE switches to sub-chain mode.

- define antenna panel

An antenna panel is a physical array of antenna elements connected to a common set of RF chains. The invention supports multiple antenna panels per UE.

- describe gNB transmit beam formation

The gNB forms transmit beams using a large-scale antenna array and applies beamforming weights to direct energy toward the UE. The invention does not alter gNB beam formation but enhances UE-side efficiency.

- explain beam sweeping procedure

Beam sweeping involves transmitting reference signals in multiple directions to identify the optimal beam. The invention reduces the need for beam sweeping by preserving beam correspondence during sub-chain operation.

- detail RS resource configuration

Reference signal (RS) resource configuration defines the time-frequency resources used for beam measurement. The invention supports standard RS configurations and does not require modification.

- describe UE measurement report feedback

The UE reports beam measurement results to the gNB, including the best beam index and signal quality. The invention ensures that these reports remain accurate during sub-chain operation.

- motivate beamforming in mmWave

Beamforming is essential in mmWave systems due to high path loss and limited diffraction. The invention enables efficient beamforming with reduced power consumption.

- describe antenna configuration on mobile terminal

The antenna configuration on the mobile terminal includes dual-polarized elements arranged in linear arrays on opposing edges. The configuration is optimized for spatial diversity and beam steering.

- detail power consumption modeling

Power consumption is modeled as the sum of base power from active amplifiers and transmitted power. The invention reduces both components by deactivating unnecessary chains.

- illustrate fallback process for terminal

When temperature exceeds a threshold, the terminal reduces the number of active chains instead of falling back to sub-6 GHz. This process is iterative and reversible.

- describe temperature check operation

Temperature is monitored periodically using embedded sensors. If the temperature exceeds a predefined threshold, the system initiates sub-chain operation.

- detail LTE fallback operation

LTE fallback is avoided by using sub-chain operation to reduce power and temperature. The invention eliminates the need for LTE fallback in most scenarios.

- motivate sub-chain operation

Sub-chain operation reduces power consumption by deactivating unnecessary antenna elements, thereby reducing both transmitted and base power.

- describe power consumption reduction

Power consumption is reduced by deactivating power amplifiers and phase shifters associated with inactive antenna elements.

- explain sub-chain beam transmission

Sub-chain beam transmission involves applying beamforming weights to a subset of antenna elements, with the remaining elements deactivated.

- list notations used throughout disclosure

Bold uppercase letters denote matrices, bold lowercase letters denote vectors, superscripts T, *, and H denote transpose, conjugate, and Hermitian, respectively. The L0 norm denotes the number of non-zero elements. The notation [a]m:n denotes a sub-vector from index m to n.

- define Nch(i)

Nch(i) denotes the number of active RF chains in the i-th operational mode.

- define NUL(i) and NDL(i)

NUL(i) denotes the number of active chains used for uplink transmission in mode i, and NDL(i) denotes the number used for downlink reception.

- define T(i)

T(i) denotes the temperature threshold for mode i.

- define γUL(i) and γDL(i)

γUL(i) and γDL(i) denote the minimum signal-to-noise ratio thresholds for uplink and downlink in mode i.

- define K(i) and G(0,0)

K(i) denotes the number of beams in the codebook for mode i, and G(0,0) denotes the gain of the beam at the boresight direction.

- illustrate downlink-uplink correspondence

Downlink-uplink correspondence is illustrated by the spatial alignment between the best downlink beam and the best uplink beam, even when the number of active chains differs.

- describe beam-sweeping operation

Beam-sweeping operation involves transmitting reference signals in multiple directions to identify the optimal beam. The invention reduces the frequency of beam-sweeping by maintaining beam correspondence.

- introduce sub-chain beam codebook design

The sub-chain beam codebook design generates a set of beamforming vectors for each possible number of active chains, optimized to preserve beam correspondence.

- define similarity score metric

The similarity score metric measures the spatial overlap between a sub-chain beam and its corresponding full-chain beam.

- describe spherical coverage metric

The spherical coverage metric measures the average beam gain over the entire angular sphere.

- describe beam correspondence spherical coverage metric

The beam correspondence spherical coverage metric measures the average gain of a sub-chain beam within the angular region dominated by its corresponding full-chain beam.

- illustrate codebook design procedure

The codebook design procedure involves generating a full-chain codebook, then generating sub-chain codebooks using one of three optimization methods.

- describe selection among three different design metrics

The selection among the three design metrics is based on operational priorities: similarity score for beam mapping fidelity, spherical coverage for spatial efficiency, and beam correspondence spherical coverage for a balanced compromise.

- illustrate codebook selection based on inter-chain beam correspondence

Codebook selection is based on the inter-chain beam correspondence metric, which evaluates the alignment between sub-chain and full-chain beams across multiple modes.

- describe codebook selection based on beam measurements

Codebook selection is based on the last measured downlink beam index, with the corresponding sub-chain beam selected from the codebook.

- illustrate codebook selection based on beam measurements

The UE selects the sub-chain beam whose radiation pattern most closely matches the direction of the last measured downlink beam.

- describe codebook selection based on terminal beam sweeping timing

Codebook selection is influenced by the timing of the last beam sweeping operation, with preference given to codebooks that minimize the need for future sweeping.

- illustrate codebook selection based on terminal beam sweeping timing

If beam sweeping was recently performed, the UE selects a codebook that preserves the current beam direction, avoiding unnecessary re-sweeping.

- introduce beam correspondence evaluation for sub-chain beam operation

Beam correspondence evaluation measures the probability that the best beam index remains unchanged when transitioning between full-chain and sub-chain modes.

- describe beam correspondence tolerance scheme

The beam correspondence tolerance scheme allows for minor deviations in beam direction, as long as the performance degradation remains below a threshold.

- describe procedure of beam correspondence tolerance scheme

The procedure involves measuring the angular deviation between the sub-chain and full-chain beams and accepting the match if the deviation is below a predefined tolerance.

- introduce sub-chain beam operation

Sub-chain beam operation involves activating only a subset of antenna elements for transmission while maintaining full reception capability.

- describe basic UE procedure for determining DL/UL beam operation scheme

The UE determines the DL/UL beam operation scheme based on temperature, battery level, signal quality, and network commands.

- describe scheme of NUL=NDL

In the NUL=NDL scheme, the same number of chains is used for uplink and downlink transmission.

- describe scheme of NUL≠NDL

In the NUL≠NDL scheme, fewer chains are used for uplink than for downlink to conserve power.

- introduce PMI feedback

Precoding Matrix Index (PMI) feedback is used by the UE to report the optimal beamforming matrix to the gNB.

- illustrate scheme selection based on PMI feedback configuration

If PMI feedback is enabled, the UE selects a codebook that aligns with the reported PMI. If PMI feedback is disabled, the UE relies on beam measurements.

- describe PMI feedback configuration without PMI feedback

Without PMI feedback, the UE selects the sub-chain beam based on the last measured downlink beam direction.

- describe PMI feedback configuration with PMI feedback

With PMI feedback, the UE selects the sub-chain beam that corresponds to the reported PMI, ensuring compatibility with gNB precoding.

- introduce temperature control

Temperature control is implemented by monitoring device temperature and reducing the number of active chains when thresholds are exceeded.

- describe power savings

Power savings are achieved by deactivating unnecessary power amplifiers and phase shifters, reducing both transmitted and base power.

- describe signal strength/quality

Signal strength and quality are monitored to ensure that sub-chain operation does not degrade communication performance below acceptable thresholds.

- describe maximum permissible exposure (MPE)

Maximum permissible exposure limits are enforced by reducing transmit power or the number of active chains when exposure thresholds are approached.

- describe precoding matrix index (PMI) feedback

PMI feedback is a 3GPP-defined mechanism for reporting the optimal precoding matrix to the gNB. The invention integrates PMI feedback into codebook selection.

- describe inter-chain beam correspondence requirement

Inter-chain beam correspondence requires that the optimal beam in one mode corresponds to the optimal beam in another mode, even when the number of active chains differs.

- describe sub-chain beam codebook design based on three different metrics

The sub-chain beam codebook is designed using similarity score, spherical coverage, or beam correspondence spherical coverage metrics, each optimized for different operational priorities.

- describe codebook selection based on operation requirement

Codebook selection is based on whether the priority is power saving, thermal management, or performance preservation.

- describe codebook selection based on beam measurements and inter-chain beam correspondence

Codebook selection combines beam measurement history with inter-chain correspondence metrics to ensure both accuracy and efficiency.

- describe codebook selection based on terminal beam sweeping timing and inter-chain beam correspondence

Codebook selection considers the timing of the last beam sweep and the correspondence between modes to minimize future sweeping.

- conclude sub-chain beam operation

Sub-chain beam operation enables significant power and thermal savings without compromising beam alignment or communication performance.

- illustrate sub-chain/full-chain operation dependent on temperature

Sub-chain operation is activated when temperature exceeds a threshold, and full-chain operation is restored when temperature falls below the threshold.

- check temperature to determine if sub-chain should be applied

Temperature is monitored periodically, and if it exceeds a predefined threshold, sub-chain operation is initiated.

- determine whether a temperature control trigger has been triggered

A temperature control trigger is activated when the temperature exceeds a threshold, initiating a mode transition.

- adopt sub-chain beam operation

When a temperature control trigger is activated, the UE adopts sub-chain beam operation by selecting the appropriate codebook.

- adopt full-chain beam operation

When temperature falls below the threshold, the UE reverts to full-chain beam operation.

- reduce number of chains as temperature increases

The number of active chains is reduced incrementally as temperature increases, with each reduction step corresponding to a lower codebook index.

- check temperature periodically

Temperature is checked at regular intervals, with the frequency increasing as temperature rises.

- determine whether a temperature control trigger has been triggered

The system continuously evaluates whether a temperature control trigger has been activated and responds accordingly.

- adopt Y-chain beam

The UE adopts a Y-chain beam when the temperature exceeds a threshold, where Y is less than the full-chain count.

- maintain X-chain operation

The UE maintains X-chain operation when temperature is within acceptable limits.

- check temperature again

After a mode transition, temperature is rechecked to determine if further adjustment is needed.

- determine whether a temperature control has been triggered

The system determines whether a new temperature control trigger has been activated since the last check.

- reduce to a lower sub-beam chain

If temperature continues to rise, the UE reduces to a lower sub-beam chain, further decreasing power consumption.

- set NUL=NDL based on temperature

The number of uplink and downlink chains is set equal when temperature is moderate, and unequal when temperature is high.

- try other antenna modules or fallback to sub-6 GHz LTE or 5G connection

If sub-chain operation cannot reduce temperature sufficiently, the UE attempts to switch to another antenna module or fallback to sub-6 GHz.

- reduce NUL to the minimum and then reduce NDL

The UE first reduces the number of uplink chains to the minimum, then reduces downlink chains if necessary.

- iteratively reduce NUL and NDL

The number of uplink and downlink chains is reduced iteratively until temperature is within limits.

- reduce the number of chains according to power consumption

The number of chains is reduced based on the total power consumption, with priority given to reducing uplink chains.

- determine the number of chains based on signal strength/quality

The number of chains is adjusted to maintain signal quality above a threshold, with fewer chains used only if quality remains acceptable.

- determine the number of chains based on battery level

The number of chains is reduced when battery level is low to extend operational time.

- determine the number of chains based on maximum permissible exposure

The number of chains is reduced when exposure levels approach regulatory limits.

- determine the number of chains based on upper layer requirement

The number of chains is adjusted based on requirements from higher-layer protocols, such as QoS or latency constraints.

- jointly consider temperature control, PMI feedback, signal strength, battery level, MPE, and other factors

The final decision on the number of active chains is based on a weighted combination of temperature, PMI feedback, signal strength, battery level, MPE, and other factors.

- check temperature, signal strength, battery level, or MPE

The system continuously monitors temperature, signal strength, battery level, and MPE to make real-time decisions.

- apply full-chain DL and UL operation

Full-chain operation is applied when all environmental and operational conditions are favorable.

- apply same number of sub chains for DL and UL

When symmetric operation is preferred, the same number of sub-chains is used for downlink and uplink.

- apply different number of chains for DL and UL

When asymmetric operation is beneficial, fewer chains are used for uplink than for downlink.

- apply a fewer number of chain for UL than DL

The system applies fewer chains for uplink transmission than for downlink reception to conserve power.

- perform antenna duty cycle reduction procedure

The antenna duty cycle is reduced by intermittently deactivating chains during periods of low data demand.

- determine if RX beam is used for beam measurement and not for data reception

If the receive beam is used only for measurement and not for data reception, the system may reduce transmit chains without affecting performance.

- apply full-chain DL operation

Full-chain downlink operation is maintained even when uplink chains are reduced, ensuring high downlink throughput.

- apply smaller number of sub chains for DL

In rare cases, the number of downlink chains is reduced if temperature is high and downlink demand is low.

- decide when to apply the antenna duty cycle reduction procedure according to the temperature

The antenna duty cycle reduction procedure is applied when temperature exceeds a threshold and data demand is low.

- perform beam sweeping during the change of chains

Beam sweeping is performed only during transitions between significantly different codebooks, minimizing overhead.

- adjust beam management parameter for sub-chain beam codebook

Beam management parameters such as reference signal period and beam indication interval are adjusted to accommodate sub-chain operation.