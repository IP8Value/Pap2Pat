Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT OF FEDERALLY SPONSORED RESEARCH  

This invention was made with government support under Grant No. [Grant Number] awarded by [Agency Name]. The government has certain rights in the invention.  

## TECHNICAL FIELD  

The present disclosure relates generally to wireless communication systems and, more particularly, to systems and methods employing reconfigurable antennas in multiple-input multiple-output (MIMO) ad hoc networks. Specifically, the invention pertains to novel architectures for reconfigurable antenna systems, distributed configuration selection algorithms, and methods for optimizing network capacity in interference-limited ad hoc network environments.  

## BACKGROUND  

Prior art wireless communication systems have employed various antenna technologies to improve network performance. Directional antennas, such as phased arrays and switchable parasitic element antennas, have been used to reduce interference between adjacent nodes. Multiple-input multiple-output (MIMO) systems employing spatial multiplexing and diversity techniques have been implemented to increase spectral efficiency. However, these conventional approaches present significant limitations when implemented in compact portable devices due to space constraints that prevent the mounting of multiple directional antennas.  

Existing solutions fail to adequately address the challenges of implementing effective antenna systems in ad hoc networks where multiple interfering links coexist. While some studies have examined reconfigurable antennas in single-link communications, there remains an unmet need for practical implementations that optimize network performance in multi-link MIMO ad hoc environments. Current systems lack efficient distributed algorithms for antenna configuration selection that can maximize network capacity without requiring centralized control or excessive channel feedback.  

## SUMMARY  

The present invention provides systems and methods for implementing reconfigurable antennas in MIMO ad hoc networks that overcome the limitations of prior art approaches. The disclosed invention encompasses several key aspects:  

First, the invention describes two novel reconfigurable antenna architectures specifically designed for MIMO ad hoc networks: a reconfigurable printed dipole array (RPDA) and a reconfigurable circular patch antenna (RCPA). The RPDA comprises two microstrip dipoles separated by a quarter wavelength, with each dipole capable of switching between "long" and "short" configurations using PIN diode switches. The RCPA features a circular patch with variable radius controlled through switching mechanisms, capable of exciting different electromagnetic modes.  

Second, the invention provides a comprehensive system model for analyzing reconfigurable antenna performance in interference-limited ad hoc networks. The model includes mathematical formulations for channel matrices, interference covariance matrices, and network capacity metrics that account for the unique characteristics of reconfigurable antennas.  

Third, the invention discloses both centralized and distributed antenna configuration selection algorithms. The centralized approach provides an upper performance bound by exhaustively searching all possible configurations, while the distributed approach enables practical implementation through localized decision-making at each node. Particularly innovative is the receiver-side reconfigurable array (RXRA) technique that simplifies implementation while maintaining performance advantages.  

Fourth, the invention includes methods for data collection and performance evaluation through both field measurements and electromagnetic ray-tracing simulations. These methods employ specific normalization procedures that enable fair comparison between different antenna architectures.  

Finally, the invention provides detailed analysis of performance results, demonstrating significant improvements in network sum capacity compared to non-reconfigurable antenna systems. The results quantify the effects of various design parameters including number of configurations, pattern diversity, and radiation efficiency.  

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS  

### I. RECONFIGURABLE ANTENNA ARCHITECTURES  

The present invention encompasses two principal reconfigurable antenna architectures optimized for MIMO ad hoc networks.  

The reconfigurable printed dipole array (RPDA) comprises two microstrip dipoles separated by a quarter wavelength (λ/4). Each dipole incorporates PIN diode switches that enable electrical reconfiguration between "long" and "short" states. This design yields four distinct configuration states for the array: long-long (l-l), short-short (s-s), long-short (l-s), and short-long (s-l). The switching mechanism alters the antenna geometry, affecting both mutual coupling characteristics and far-field radiation patterns.  

The reconfigurable circular patch antenna (RCPA) features a single circular patch with electrically variable radius controlled through switching elements. The antenna operates in two distinct electromagnetic modes: TM31 ("Mode 3") and TM41 ("Mode 4"), selected through switch activation. Two feed points are strategically positioned to ensure spatial orthogonality between radiation patterns while maintaining port isolation exceeding 20 dB.  

Both architectures operate in the 2.4-2.5 GHz frequency band suitable for 802.11-like MIMO networks. Key performance metrics include radiation pattern spatial correlation coefficients and radiation efficiency. The RPDA demonstrates superior radiation efficiency (48-84% across configurations) while the RCPA offers superior pattern diversity (correlation coefficient of 0.2 between modes).  

### II. SYSTEM MODEL AND NOTATION  

The invention provides a comprehensive system model for analyzing reconfigurable antenna performance in ad hoc networks with L co-located interfering links. The model employs the following notation:  

H_{i_{rc},j_{tc}} represents the channel matrix between the receiver of link i and transmitter of link j, where i_{rc} and j_{tc} denote the receiver and transmitter configurations respectively. For the RPDA, configuration indices range from 1 to 4, while for the RCPA they range from 1 to 2.  

The input-output relationship for link l is given by:  
y_l = H_{l_{rc},l_{tc}}x_l + Σ_{i∈L\l} H_{l_{rc},i_{tc}}x_i + n  
where the interference-plus-noise term has covariance matrix:  
R_l = Σ_{i∈L\l} H_{l_{rc},i_{tc}} P_i H_{l_{rc},i_{tc}}^H + σ^2I  

The network sum capacity is calculated as:  
C_{sum} = Σ_{l∈L} log_2 det(I + P_l H_{l_{rc},l_{tc}}^H R_l^{-1} H_{l_{rc},l_{tc}})  

The model assumes equal power allocation across antennas and incorporates the effects of both communication channels and interference channels that vary with antenna configuration selections.  

### III. ANTENNA CONFIGURATION SELECTION METHODS  

The invention discloses three configuration implementation scenarios:  

1. Double-Side Reconfigurable Arrays (DSRA): Both transmitter and receiver can adapt configurations  
2. Transmitter-Side Reconfigurable Arrays (TXRA): Only transmitter can adapt configurations  
3. Receiver-Side Reconfigurable Arrays (RXRA): Only receiver can adapt configurations  

For each scenario, both centralized and distributed selection algorithms are provided.  

The centralized algorithm solves:  
c_{opt} = argmax_{c∈C} C_{sum}(c)  
through exhaustive search over all possible configuration combinations.  

The distributed algorithm solves for each link l:  
c_l = argmax_{c_l∈C_l} log_2 det(I + P_l H_{l_{rc},l_{tc}}^H R_l^{-1} H_{l_{rc},l_{tc}})  

The RXRA approach is particularly advantageous as it eliminates the need for iterative convergence, requires no transmitter feedback, and naturally aligns individual link optimization with network-wide capacity maximization.  

### IV. DATA COLLECTION  

The invention includes methods for empirical performance evaluation through both measurements and simulations.  

Measurement procedures utilize a 2×2 MIMO software-defined radio platform operating in the 2.4 GHz band with OFDM modulation. Channel estimates are obtained at multiple spatial positions (λ/10 spacing) with 100 samples averaged per position.  

Simulation methods employ 3D ray-tracing using measured antenna patterns. The simulation environment models an indoor office space with appropriate material properties.  

Both methods incorporate specific normalization procedures to enable fair comparison between antenna architectures:  
max_{l,i∈L,rc,tc} E{||H_{l_{rc},i_{tc}}||_F^2} = 4  

### V. RESULTS  

Experimental results demonstrate significant capacity improvements:  

For RCPA:  
- Centralized DSRA: 50-75% capacity increase  
- Distributed RXRA: 14-31% capacity increase  

For RPDA:  
- Centralized DSRA: >50% capacity increase  
- Distributed RXRA: 24-31% capacity increase  

The distributed algorithms show rapid convergence (typically <3 iterations). The RPDA generally outperforms the RCPA due to its greater number of configurations and better efficiency balance, though the RCPA shows superior performance when pattern correlation is isolated as a factor.  

### VI. SOFTWARE IMPLEMENTATION  

The configuration selection algorithms may be implemented in software through the following process:  

1. Channel estimation module: Estimates H_{l_{rc},l_{tc}} and R_l for all relevant configurations  
2. Capacity calculation module: Computes log_2 det(I + P_l H_{l_{rc},l_{tc}}^H R_l^{-1} H_{l_{rc},l_{tc}})  
3. Optimization module: Performs argmax operation for centralized or distributed selection  
4. Configuration control module: Implements selected configurations through antenna switch control  

The software architecture supports both real-time operation for distributed implementations and batch processing for centralized analysis.  

### VII. CONCLUSIONS  

The disclosed invention provides comprehensive solutions for implementing reconfigurable antennas in MIMO ad hoc networks. Key advantages include:  

1. Compact antenna designs suitable for portable devices  
2. Significant capacity improvements over non-reconfigurable systems  
3. Practical distributed algorithms requiring minimal coordination  
4. Flexible implementation options (DSRA, TXRA, RXRA)  

The invention enables wireless system designers to optimize network performance through careful consideration of configuration count, pattern diversity, and efficiency balance in reconfigurable antenna design.