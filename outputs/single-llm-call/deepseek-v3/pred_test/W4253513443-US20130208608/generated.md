Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT OF FEDERALLY SPONSORED RESEARCH  

The invention described herein was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Agency Name]. The government has certain rights in the invention.  

## TECHNICAL FIELD  

The present invention relates generally to wireless communication systems and more specifically to reconfigurable antenna systems for multiple-input multiple-output (MIMO), multiple-input single-output (MISO), and single-input multiple-output (SIMO) ad-hoc networks.  

## BACKGROUND  

Recent research in ad-hoc networks has yielded significant advances in physical layer techniques, particularly in the application of smart antennas and antenna diversity methods. Medium access control protocols suitable for MIMO ad-hoc networks have been developed, along with adaptive algorithms for antenna beamforming in such networks. Directional antennas, including phased arrays and switchable parasitic element antennas, have been proposed to reduce interference between adjacent nodes and maximize overall network throughput.  

MIMO spatial multiplexing techniques and diversity methods have been adopted to increase network spectral efficiency. However, integrating directional arrays and MIMO spatial multiplexing/diversity techniques on compact portable devices presents significant challenges due to space constraints. Electrically reconfigurable antennas have emerged as a solution to merge the benefits of MIMO techniques with those of directional antennas while minimizing space requirements.  

Previous studies have examined reconfigurable antennas in single-link communications, but no published work has addressed their implementation and testing in multi-link MIMO ad-hoc networks. The present invention fills this gap by providing systems and methods for utilizing reconfigurable antennas in ad-hoc networks, along with configuration selection algorithms to optimize network performance.  

## SUMMARY  

The present invention discloses a MIMO/SIMO/MISO ad-hoc network system employing reconfigurable antennas and methods for selecting optimal antenna configurations. The system includes two compact pattern reconfigurable antenna architectures: a reconfigurable printed dipole array (RPDA) and a reconfigurable circular patch antenna (RCPA).  

A configuration selection method is provided that may operate in either centralized or distributed modes. The centralized approach uses exhaustive search to assign optimal configurations across all network nodes, while the distributed approach allows each node to independently select configurations to maximize its own capacity. Performance improvements over conventional non-reconfigurable antenna systems are demonstrated through both measurements and simulations.  

Alternative configuration selection schemes are described where reconfigurability may be implemented at both ends of a link (DSRA), only at the transmitter (TXRA), or only at the receiver (RXRA). The RXRA scheme is particularly advantageous as it eliminates the need for iterative procedures and feedback loops between transmitters and receivers.  

The invention further describes methods for quantifying the benefits of reconfigurable antennas through parameters including the number of available configurations, spatial orthogonality between array elements, and antenna radiation efficiency. These parameters enable prediction of achievable performance with specific reconfigurable antenna designs in ad-hoc network environments.  

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS  

The following detailed description presents specific embodiments of the invention with reference to the accompanying drawings. These embodiments are provided to enable those skilled in the art to practice the invention, and it should be understood that other embodiments may be utilized and structural changes may be made without departing from the scope of the present invention.  

Reconfigurable antennas provide significant advantages in MIMO/SIMO/MISO ad-hoc networks by enabling dynamic adaptation to changing network conditions. The invention describes two specific reconfigurable antenna architectures along with methods for selecting optimal configurations in various network scenarios.  

### I. RECONFIGURABLE ANTENNA ARCHITECTURES  

Two compact pattern reconfigurable antenna designs are disclosed for use in 2×2 MIMO systems employing spatial multiplexing. The first architecture is a reconfigurable printed dipole array (RPDA) consisting of two microstrip dipoles separated by a quarter wavelength. Each dipole can be electrically reconfigured in length using PIN diode switches, providing four distinct configurations: long-long (l-l), short-short (s-s), short-long (s-l), and long-short (l-s).  

The RPDA exhibits spatial correlation coefficients ≤0.7 between radiation patterns at different ports, indicating significant diversity gain potential. Radiation efficiency varies between configurations from 48% to 84%, with the s-s configuration being most efficient.  

The second architecture is a reconfigurable circular patch antenna (RCPA) whose radius can be electrically varied to excite different electromagnetic modes. The RCPA has two configurations: Mode 3 (TM31 mode) and Mode 4 (TM41 mode). The RCPA demonstrates nearly orthogonal radiation patterns between configurations (spatial correlation coefficient ≈0.2), providing high pattern diversity. Radiation efficiency ranges from 5% to 21%, with Mode 3 being more efficient.  

Both antenna designs provide full radiation coverage in the azimuth plane, ensuring reliable signal reception regardless of transmitter-receiver orientation. The RPDA offers more configuration states (4 vs. 2), while the RCPA provides greater pattern diversity between states.  

### II. SYSTEM MODEL AND NOTATION  

The system model assumes an ad-hoc network with L co-located links that interfere with each other. All links are single-hop with predetermined transmit-receive pairs. The channel between the receiver of link i and transmitter of link j is denoted H_i^rc,j^tc, where rc and tc indicate receive and transmit configurations respectively.  

The input-output relationship for link l is given by:  
y_l = H_l^rc,l^tc x_l + Σ_i∈L\l H_l^rc,i^tc x_i + n  

where y_l is the received signal, x_l is the transmitted signal, and n represents noise. The interference plus noise covariance matrix for link l is:  
R_l = Σ_i∈L\l H_l^rc,i^tc (H_l^rc,i^tc)^H + σ^2 I  

Using equal power allocation, the capacity of link l is:  
C_l = log_2 det(I + (P_T/N_T) (H_l^rc,l^tc)^H R_l^-1 H_l^rc,l^tc)  

where P_T is total transmit power and N_T is number of transmit antennas. The sum network capacity is the primary performance metric:  
C_sum = Σ_l∈L C_l  

### III. ANTENNA CONFIGURATION SELECTION METHODS  

Three configuration adaptation cases are considered:  
1. Double-side reconfigurable antennas (DSRA) - both transmitter and receiver can adapt configurations  
2. Transmitter-side reconfigurable array (TXRA) - only transmitter adapts configurations  
3. Receiver-side reconfigurable array (RXRA) - only receiver adapts configurations  

For the centralized approach, a controller with global channel knowledge solves:  
c_opt = argmax_c C_sum(c)  
where c contains configurations for all network nodes.  

The distributed approach has each link l solve:  
c_l^opt = argmax_c_l C_l(c_l)  
This leads to an iterative procedure similar to iterative waterfilling, where nodes continually update configurations in response to interference changes.  

The RXRA scheme is particularly advantageous as it:  
1. Eliminates need for iterative procedures  
2. Removes requirement for transmitter-receiver feedback loops  
3. Makes distributed and centralized approaches equivalent  
4. Reduces search space and channel training requirements  

### IV. DATA COLLECTION  

Performance evaluation used both measurements and electromagnetic ray-tracing simulations in an indoor environment. The measurement setup employed a 2×2 MIMO platform operating at 2.4 GHz with OFDM across 64 subcarriers. Three receiver and three transmitter nodes created six network topologies, with small-scale fading captured through 40 position variations per topology.  

Simulations used a 3D ray tracer with measured antenna patterns as input. Both measurements and simulations were normalized to enable fair comparison between antenna architectures.  

### V. RESULTS  

For RCPAs, measured sum capacity increases ranged from 8.7% (distributed TXRA) to 75% (centralized DSRA) over non-reconfigurable systems. Distributed RXRA provided a 31% increase with simpler implementation.  

RPDAs showed even greater improvements, with measured capacity increases from 30% (distributed TXRA) to over 100% in some cases. Distributed RXRA provided a 31% increase.  

Analysis revealed that:  
1. More configuration states generally yield better performance  
2. Lower correlation between patterns improves performance  
3. Balanced radiation efficiency across configurations is desirable  

The distributed RXRA scheme provided an optimal balance between performance gains and implementation complexity.  

### VI. SOFTWARE IMPLEMENTATION  

The configuration selection algorithms may be implemented in software running on conventional computing systems. A typical implementation includes:  

- Processing unit (e.g., CPU)  
- System memory (RAM)  
- Storage devices (hard drives, SSDs)  
- Network interfaces for communication between nodes  
- Input/output devices for user interaction  

The software may comprise computer-executable instructions stored on non-transitory media that, when executed, perform the configuration selection methods described herein.  

### VII. CONCLUSIONS  

The invention demonstrates that reconfigurable antennas can significantly improve MIMO ad-hoc network performance. Key design insights include:  

1. More configuration states generally provide better performance  
2. Lower pattern correlation improves diversity gain  
3. Balanced radiation efficiency across configurations is desirable  
4. RXRA schemes offer excellent performance with simpler implementation  

These principles guide the design of effective reconfigurable antenna systems for ad-hoc networks. The disclosed architectures and methods provide substantial capacity improvements over conventional approaches while maintaining practical implementation requirements.