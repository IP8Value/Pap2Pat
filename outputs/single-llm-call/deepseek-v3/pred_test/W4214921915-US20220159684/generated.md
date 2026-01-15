Here is the patent application drafted according to the provided outline and research paper:

# DESCRIPTION  

## TECHNICAL FIELD  

The present disclosure relates generally to wireless communication systems. More specifically, the disclosure pertains to beamforming techniques in millimeter wave (mmWave) frequency bands for fifth generation (5G) and beyond wireless networks. The invention particularly focuses on user equipment (UE) beam codebook design and operation methods that enable power savings while maintaining beam correspondence between downlink and uplink transmissions.  

## BACKGROUND  

Modern wireless communication systems continue to face challenges in improving spectral efficiency, throughput, and reliability while managing power consumption. These challenges become particularly acute in mmWave frequency bands where high-gain beamforming using antenna arrays is essential to overcome severe path loss. Conventional approaches to managing UE power consumption and thermal issues in mmWave bands often involve falling back to sub-6 GHz frequencies, which results in reduced data rates and increased latency. There exists a need for improved techniques that allow continued operation in mmWave bands while effectively managing power consumption and thermal constraints without sacrificing beam management performance.  

## SUMMARY  

The present disclosure introduces novel UE beam codebook designs and operation methods that address these challenges. The invention describes a UE embodiment comprising antenna arrays with configurable activation states, where portions of the array can be selectively deactivated to form sub-chain beams while maintaining beam correspondence with full-chain operation. The UE components include multiple antenna panels, radio frequency (RF) transceivers, processing circuitry, and memory storing beam codebooks designed according to specific optimization criteria.  

The UE functionality encompasses dynamic switching between full-chain and sub-chain beam operation based on various conditions including temperature, power consumption, signal quality, and other operational parameters. A method embodiment is introduced comprising steps for determining appropriate beam operation schemes, performing beam measurements, and selecting codebooks that maintain beam correspondence between downlink and uplink directions.  

Technical features include novel codebook design metrics that balance beam similarity and spherical coverage, procedures for beam correspondence evaluation, and temperature-controlled operation modes. Certain words and phrases are defined, including "full-chain beam" referring to operation with all antenna elements active, "sub-chain beam" referring to operation with a subset of antenna elements active, and "beam correspondence" referring to the relationship between optimal downlink and uplink beams.  

## DETAILED DESCRIPTION  

5G communication systems utilize beamforming and massive multiple-input multiple-output (MIMO) techniques to achieve high throughput in mmWave frequency bands ranging from 24 GHz to 100 GHz. Beam-specific operations are motivated by the need to overcome high path loss while managing power consumption and thermal constraints at the UE.  

The system employs orthogonal frequency division multiplexing (OFDM) and orthogonal frequency division multiple access (OFDMA) techniques for downlink and uplink signaling. Duplex methods including time division duplex (TDD) and frequency division duplex (FDD) are supported. Recent network improvement developments have focused on enhancing beam management procedures while optimizing power efficiency.  

FIG. 1 illustrates an example wireless network comprising gNBs (next-generation NodeBs) and UEs operating in mmWave bands. The gNB components include multiple antennas arranged in two-dimensional (2D) arrays, RF transceivers, transmit (TX) and receive (RX) processing circuitry, controller/processor components, and memory. The coverage areas of gNBs are extended through beamforming techniques that focus energy in specific directions.  

The UE architecture, shown in FIG. 3, includes antenna arrays, RF transceivers, TX/RX processing circuitry, microphone and speaker components, input/output (I/O) interfaces, processors, memory, touchscreen displays, and operating systems supporting various applications. The UL transmission on uplink channels employs configurable antenna activation states to optimize power consumption while maintaining performance.  

FIGS. 4A and 4B depict transmit and receive path circuitry respectively. The transmit path includes channel coding and modulation blocks, serial-to-parallel (S-to-P) converters, inverse fast Fourier transform (IFFT) blocks, parallel-to-serial (P-to-S) converters, cyclic prefix insertion blocks, and up-converters. The receive path comprises down-converters, cyclic prefix removal blocks, S-to-P converters, FFT blocks, P-to-S converters, and channel decoding/demodulation blocks. These components are implemented through configurable hardware and software elements.  

5G communication systems support various use cases including enhanced mobile broadband (eMBB) for high data rates, ultra-reliable low latency (URLL) communications, and massive machine-type communications (mMTC). The system architecture facilitates downlink and uplink signaling with flexible resource allocation across frequency and time domains.  

Antenna panel architectures support multi-beam operation through quasi co-located antenna ports. UE configurations include transmission configuration indicator (TCI) states that define quasi-co-location (QCL) relationships between reference signals. Medium access control (MAC) control element (CE) activation commands configure these relationships for beam management.  

Beam training and measurement procedures involve beam sweeping operations where gNBs and UEs sequentially transmit and receive using different beam directions. Reference signal (RS) resource configurations enable UE measurement report feedback that guides beam selection. Beamforming in mmWave bands is particularly important due to the need for high antenna gain to overcome propagation challenges.  

The antenna configuration on mobile terminals considers power consumption modeling and thermal constraints. A fallback process to LTE operation is available but avoided through the disclosed sub-chain beam operation techniques. Temperature check operations trigger transitions between full-chain and sub-chain modes to manage thermal conditions while maintaining connectivity.  

Sub-chain beam operation provides power consumption reduction by activating only a portion of the antenna array. The notation Nch(i) represents the number of active chains in time interval i, while NUL(i) and NDL(i) denote uplink and downlink active chains respectively. The parameter T(i) indicates temperature, and γUL(i) and γDL(i) represent uplink and downlink power scaling factors.  

Downlink-uplink beam correspondence is evaluated through similarity score metrics, spherical coverage metrics, and beam correspondence spherical coverage metrics. The codebook design procedure selects among these three different metrics based on operational requirements. Codebook selection considers inter-chain beam correspondence, beam measurement results, and terminal beam sweeping timing.  

Beam correspondence evaluation for sub-chain beam operation includes tolerance schemes that define acceptable deviations from ideal correspondence. The basic UE procedure for determining DL/UL beam operation schemes supports both symmetric (NUL=NDL) and asymmetric (NUL≠NDL) configurations. Precoding matrix index (PMI) feedback configurations influence scheme selection, with options for operation with or without PMI feedback.  

Temperature control mechanisms coordinate power savings with signal strength/quality requirements and maximum permissible exposure (MPE) limits. The precoding matrix index feedback interacts with inter-chain beam correspondence requirements to optimize performance. Sub-chain beam codebook designs based on the three metrics (similarity score, spherical coverage, beam correspondence spherical coverage) are selected according to operational needs.  

Codebook selection procedures consider beam measurements, inter-chain beam correspondence, terminal beam sweeping timing, and other factors. The conclusion of sub-chain beam operation involves periodic temperature checks to determine appropriate chain configurations. FIG. 2 illustrates transitions between sub-chain and full-chain operation dependent on temperature conditions.  

The temperature control procedure checks whether triggers have been activated to adopt sub-chain beam operation. When triggered, the system reduces the number of active chains as temperature increases, periodically re-evaluating conditions. The process may adopt Y-chain beam configurations while maintaining X-chain operation for certain components, with iterative reductions in NUL and NDL values.  

Alternative procedures include trying other antenna modules or falling back to sub-6 GHz LTE or 5G connections when thermal conditions require. The number of active chains may be reduced according to power consumption needs, signal strength/quality metrics, battery levels, MPE considerations, and upper layer requirements. Joint consideration of temperature control, PMI feedback, signal strength, battery level, MPE, and other factors optimizes the operation mode selection.  

The system applies full-chain DL and UL operation when conditions permit, transitioning to same or different numbers of sub-chains for DL and UL as needed. Antenna duty cycle reduction procedures may be activated when RX beams are used for measurement rather than data reception. Beam management parameters are adjusted for sub-chain beam codebooks to maintain performance during mode transitions.  

Beam sweeping operations accompany changes in active chain configurations to ensure continued beam alignment. The comprehensive system of beam codebook design and temperature-aware operation provides significant improvements in power efficiency while maintaining beam correspondence and communication performance in mmWave bands.