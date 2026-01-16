# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of millimeter wave (mmWave) communication systems, particularly to methods and apparatuses for managing power consumption and preventing overheating in user equipment (UE) by employing sub-chain beamforming techniques. The invention specifically addresses the design and implementation of sub-chain beam codebooks to maintain downlink-uplink (DL-UL) beam correspondence while reducing power consumption in mmWave 5G and beyond devices.

## BACKGROUND

In the millimeter wave band, antenna arrays are commonly employed by user equipment (UE) to generate high-gain beams, thereby achieving higher signal-to-noise ratio (SNR) and throughput compared to single antennas. For instance, various configurations such as 2 × 1, 4 × 1, or 2 × 2 arrays have been proposed for mmWave 5G phones. However, one of the significant challenges in 5G and beyond is the power consumption of UEs, which exacerbates issues related to battery life and temperature control, especially in the mmWave bands compared to the sub-6 GHz band.

When a phone heats up rapidly, a common solution is to fall back to the sub-6 GHz band and deactivate the mmWave array. This fallback, however, is undesirable for several reasons. First, the maximum data rate drops significantly from gigabits per second (Gbps) to a few hundred megabits per second (Mbps) or less. Second, the frequent turn-off and turn-on of the mmWave antenna module introduces additional latency, power consumption, and potential service disruptions.

An alternative approach to mitigate these issues is to reduce the number of activated antenna elements, a technique referred to as "subchain beamforming" in this context. In subchain beamforming, only a portion of the antenna array is activated for uplink (UL) transmission, while the entire array remains active for downlink (DL) reception. This strategy is chosen because transmission typically consumes more power than reception, and the downlink data rate requirements are generally higher than those for the uplink.

The 5G standard includes processes for identifying and maintaining suitable beam pairs for the base station (BS)-UE link, known as beam management (BM). The DL-UL beam correspondence, which assumes that the best beams in the downlink direction are also the best beams in the uplink direction, is a crucial design criterion. Disrupting this correspondence can necessitate additional beam management procedures, leading to increased complexity and resource usage.

This invention proposes methods to design sub-chain beam codebooks that maximize the maintenance of DL-UL beam correspondence, ensuring that the best downlink beam corresponds to the best uplink beam even when using sub-chain beams. Additionally, the invention explores the benefits of sub-chain beamforming over other power-saving techniques, such as scaling down the transmission power level of all antennas, which does not effectively reduce overall power consumption due to the base power required to activate power amplifiers (PAs).

## SUMMARY

The present invention provides a method and apparatus for designing sub-chain beam codebooks in millimeter wave (mmWave) communication systems to maintain downlink-uplink (DL-UL) beam correspondence while reducing power consumption. The invention addresses the challenge of overheating and high power consumption in user equipment (UE) by selectively deactivating portions of the antenna array during uplink transmission.

The invention includes the following key aspects:

1. **Sub-Chain Beam Codebook Design**: The invention proposes three methods for designing sub-chain beam codebooks:
   - **Similarity Score Maximization (Sim-Max)**: This method designs sub-chain beams to closely resemble the full-chain beams, ensuring high beam correspondence.
   - **Spherical Coverage Maximization (SC-Max)**: This method focuses on optimizing the spherical coverage of the sub-chain codebook, without considering beam correspondence.
   - **Beam Correspondence Spherical Coverage Maximization (BC-SC-Max)**: This method strikes a balance between similarity and spherical coverage, aiming to maintain beam correspondence while optimizing coverage.

2. **Beam Management**: The invention ensures that the UE can switch between full-chain and sub-chain operations without the need for a new round of beam sweeping, provided the beam correspondence is maintained. This reduces the overhead and latency associated with beam management.

3. **Performance Evaluation**: Extensive simulations are conducted to compare the performance of the proposed methods in terms of beam correspondence and spherical coverage. The results demonstrate that the BC-SC-Max method achieves superior performance, maintaining beam correspondence for over 90% of the time when switching among full-chain, 4-Ant, and 3-Ant sub-chain beam codebooks.

The invention is particularly useful in mmWave 5G and beyond devices, where high data rates and power efficiency are critical. By effectively managing power consumption and preventing overheating, the invention enhances the overall performance and reliability of mmWave communication systems.

## DETAILED DESCRIPTION

### System Model

In this invention, we consider a user equipment (UE) equipped with two arrays located on the left and right edges, respectively. Each array consists of \( L \) dual-polarization patch antennas. The \( 2L \) antenna elements are denoted as \( 1V, 2V, \ldots, LV, 1H, 2H, \ldots, LH \). The beamforming vector is defined as:

\[
\mathbf{w} = [w_1, w_2, \ldots, w_{2L}]^T
\]

where the magnitude of the beamforming weights \( |w_i| \) (for \( 1 \leq i \leq 2L \)) is either 0 or 1, indicating whether the antenna is on or off. The phase of \( w_i \) is restricted to a few discrete values due to the use of finite-resolution phase shifters. If \( b \)-bit phase shifters are used, the constraint on a non-zero beamforming weight is \( |w_i|^{2b} = 1 \).

The radiation pattern of a practical mmWave antenna, influenced by the terminal housing, is highly irregular. To address this, a data-driven method is employed for codebook design. The E-field response data of each antenna element is obtained through simulation or measurement, incorporating the effects of the terminal housing. The beam pattern is calculated as:

\[
B(\theta, \phi) = \left| \sum_{i=1}^{2L} w_i M_i(\theta, \phi) \right|^2
\]

where \( M(\theta, \phi) \) is the E-field response matrix of each antenna in the direction \( (\theta, \phi) \), typically a rank-2 matrix.

For sub-chain beams, additional design requirements may be imposed based on hardware implementation. In this invention, we consider a constraint that the number of activated antennas in the two polarizations is the same, i.e.,

\[
\|\mathbf{w}_{1:L}\|_0 = \|\mathbf{w}_{L+1:2L}\|_0
\]

However, the indices of the activated H and V-polarization antennas can differ. This constraint can be extended to other requirements, such as ensuring the same indices for activated antennas in both polarizations.

### Sub-Chain Beam Codebook Design

#### Similarity Score Maximization (Sim-Max)

In the first method, the sub-chain beams are designed to closely resemble the full-chain beams. The radiation pattern of each sub-chain beam is optimized to be similar to the corresponding full-chain beam, ensuring one-to-one mapping. The similarity score is defined as:

\[
\text{Sim}(i, j) = \frac{\sum_{n=1}^{N_p} G_i(\theta_n, \phi_n) B_j(\theta_n, \phi_n)}{\sqrt{\sum_{n=1}^{N_p} G_i^2(\theta_n, \phi_n)} \sqrt{\sum_{n=1}^{N_p} B_j^2(\theta_n, \phi_n)}}
\]

where \( G_i(\theta, \phi) \) is the \( i \)-th full-chain beam pattern, and \( B_j(\theta, \phi) \) is the sub-chain beam pattern. The candidate sub-chain beam with the highest similarity score is selected. This can be achieved by solving the following optimization problem:

\[
\max_{\mathbf{w}} \sum_{n=1}^{N_p} G_i(\theta_n, \phi_n) B_j(\theta_n, \phi_n)
\]

subject to:

\[
\|\mathbf{w}\|_0 = L_A
\]
\[
|w_i|^{2b} = 1 \quad \forall i
\]

Given the non-convex nature of the constraints, an exhaustive search over all possible activations is performed, followed by an iterative algorithm to optimize the phase cyclically. Multiple initial beams are used to ensure the selection of the best local optimum.

#### Spherical Coverage Maximization (SC-Max)

In this method, the sub-chain codebook is designed to maximize spherical coverage, without considering beam correspondence. This design is suitable if the UE operates with the same number of antennas for both transmission and reception, maintaining DL-UL beam correspondence as in the full-chain case. The optimization problem is:

\[
\max_{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_K} \sum_{k=1}^{K} \int_{\Omega} B_k(\theta, \phi) d\Omega
\]

subject to:

\[
\|\mathbf{w}_k\|_0 = L_A
\]
\[
|w_{ki}|^{2b} = 1 \quad \forall i
\]

The K-Means algorithm is used to solve this problem, iterating between assigning directions to the beam with the largest gain and optimizing the beams to serve the assigned directions. The beam optimization step involves solving \( L_A^2 K \) problems by exhausting all possible antenna activations.

#### Beam Correspondence Spherical Coverage Maximization (BC-SC-Max)

The third method combines the objectives of similarity and spherical coverage. The sub-chain beams are designed to maximize the radiation pattern over the full-chain beam's coverage region, ensuring that a fresh beam sweeping is not necessary when switching between full-chain and sub-chain operations. The design procedure is as follows:

1. **Partition the Unit-Sphere**: Divide the unit-sphere into \( K \) disjoint angular regions, each covered by a full-chain beam \( \mathbf{w}_k \).

2. **Design Sub-Chain Beams**: For each angular region, design the best sub-chain beam by solving:

\[
\max_{\mathbf{w}} \int_{D_k} B(\theta, \phi) d\Omega
\]

subject to:

\[
\|\mathbf{w}\|_0 = L_A
\]
\[
|w_i|^{2b} = 1 \quad \forall i
\]

The optimization problem is similar to the Sim-Max method but focuses on the main lobe region of the full-chain beam. The same iterative algorithm is used, with modifications to consider only the angular region and remove the weighting by the full-chain beam pattern.

### Simulation Results

Extensive simulations were conducted to evaluate the performance of the proposed methods. A 5G phone with two mmWave 1x5 arrays on the left and right edges was modeled using electromagnetic simulation software. E-field data was generated for each antenna element, and a 5-bit phase shifter resolution was used. The full-chain codebook was generated using the K-Means algorithm. The SC-Max method was initialized with a codebook obtained by a greedy algorithm.

Figures 2-4 illustrate the generated sub-chain beam codebooks from the Sim-Max, SC-Max, and BC-SC-Max methods, respectively. The composite beam patterns and best beam index distributions are shown for each method. The Sim-Max and BC-SC-Max codebooks exhibit high similarity in pattern shapes and best beam index distributions, indicating well-maintained beam correspondence. In contrast, the SC-Max codebooks show less similarity and different best beam index distributions across the codebooks.

The best beam matching rate, defined as the probability that the best beam index is the same between two codebooks over the unit-sphere, was quantified. The results indicate that the Sim-Max and BC-SC-Max codebooks preserve beam correspondence for over 90% of the time when switching among 5-Ant, 4-Ant, and 3-Ant codebooks. The BC-SC-Max method performs particularly well when switching to 1-Ant codebooks, maintaining beam correspondence for over 60% of the time.

To improve the matching rate of SC-Max codebooks, a repair procedure was applied, pairing beams based on the intersection of dominant sampling points. While this improved the matching rate, it still fell short of the Sim-Max and BC-SC-Max methods.

Finally, the spherical coverage of the codebooks was evaluated by checking the composite beam gain cumulative distribution function (CDF) on the unit-sphere. The results showed that the BC-SC-Max method achieved similar spherical coverage to the SC-Max method, while maintaining superior beam correspondence.

### Conclusion

The present invention provides a practical beam operation scheme for mmWave 5G devices to enhance the utilization of the mmWave band by managing power consumption and preventing overheating. The proposed sub-chain beamforming techniques, particularly the BC-SC-Max method, effectively maintain DL-UL beam correspondence while optimizing spherical coverage. This ensures that the UE can switch between full-chain and sub-chain operations with minimal overhead, improving the overall performance and reliability of mmWave communication systems.