# Patent Application: Sub-Chain Beam Operation for mmWave 5G Devices

## Background

### Field of Invention
The present invention relates to wireless communication systems and, more specifically, to methods and apparatuses for optimizing beam operations in millimeter-wave (mmWave) 5G devices to manage power consumption and heating issues.

### Description of Related Art
Millimeter-wave (mmWave) frequencies offer significant bandwidth for high-speed data transmission, but they also present challenges such as high power consumption and thermal management. Existing solutions often involve falling back to sub-6 GHz bands when the device overheats, which limits the utilization of mmWave capabilities. This invention proposes a sub-chain beam operation that deactivates part of the mmWave antenna array in uplink transmission to manage heat and power while maintaining communication efficiency.

## Summary of the Invention

### Overview
The invention provides a method for managing power consumption and thermal issues in mmWave 5G devices by selectively deactivating parts of the antenna array during uplink transmission. This sub-chain beam operation ensures that the device can continue to utilize mmWave frequencies without overheating, thereby maintaining high data rates and communication quality.

### Key Features
- **Sub-Chain Beam Operation**: Deactivates part of the mmWave antenna array in uplink transmission when the device is overheating.
- **Beam Correspondence Preservation**: Ensures that downlink-uplink beam correspondence is maintained to the extent possible, reducing the need for repeated beam sweeping.
- **Codebook Design Methods**: Proposes three methods—Sim-Max, SC-Max, and BC-SC-Max—for designing sub-chain codebooks to optimize performance.

## Detailed Description

### Sub-Chain Beam Operation
When a mmWave 5G device detects overheating or high power consumption, it can activate the sub-chain beam operation. This involves deactivating a portion of the antenna array in uplink transmission while keeping the full array active for downlink reception. The sub-chain beam operation helps manage heat and power without completely falling back to lower frequency bands.

### Beam Correspondence Preservation
The sub-chain beam operation could potentially disrupt the downlink-uplink beam correspondence, which is crucial for maintaining communication efficiency. To mitigate this issue, the invention proposes three methods for designing sub-chain codebooks:

1. **Sim-Max Method**: Generates sub-chain beams that closely resemble the shape of full-chain beams, ensuring high beam correspondence.
2. **SC-Max Method**: Optimizes the spherical coverage of sub-chain codebooks without considering one-to-one mapping with full-chain beams.
3. **BC-SC-Max Method**: Balances both similarity to full-chain beams and spherical coverage, providing a robust solution for maintaining beam correspondence.

### Codebook Design Methods
#### Sim-Max Method
The Sim-Max method generates sub-chain beams that closely match the shape of full-chain beams. This is achieved by solving an optimization problem that maximizes the similarity between the radiation patterns of full-chain and sub-chain beams over the entire sphere. The iterative algorithm used in this method efficiently optimizes the phase of the antennas to achieve the best local optimum.

#### SC-Max Method
The SC-Max method focuses on maximizing the spherical coverage of sub-chain codebooks without considering one-to-one mapping with full-chain beams. This is achieved by solving an optimization problem that maximizes the average beam gain over the entire sphere. The K-Means algorithm is used to iteratively assign directions to the best beam and optimize the beams for those directions.

#### BC-SC-Max Method
The BC-SC-Max method combines the benefits of both Sim-Max and SC-Max methods. It designs sub-chain beams that maximize the radiation pattern over the coverage region of full-chain beams, ensuring high beam correspondence while maintaining good spherical coverage. This is achieved by partitioning the unit-sphere into regions covered by full-chain beams and solving an optimization problem for each region.

### Simulation Results
Simulations were conducted to compare the performance of the three methods in a real 5G phone with two mmWave arrays. The results showed that the BC-SC-Max method provides the best balance between beam correspondence and spherical coverage. It achieves superior spherical coverage close to the SC-Max method while maintaining high beam correspondence, especially when switching between full-chain, 4-Ant, and 3-Ant sub-chain codebooks.

### Conclusion
The invention proposes a practical sub-chain beam operation for mmWave 5G devices that effectively manages power consumption and thermal issues. By deactivating part of the antenna array in uplink transmission, the device can continue to utilize mmWave frequencies without overheating. The BC-SC-Max method is particularly effective in maintaining high beam correspondence and spherical coverage, ensuring efficient communication even with reduced antenna activation.

## Claims
1. A method for managing power consumption and thermal issues in a millimeter-wave (mmWave) 5G device, comprising:
   - Detecting overheating or high power consumption in the device.
   - Activating a sub-chain beam operation that deactivates part of the mmWave antenna array in uplink transmission while keeping the full array active for downlink reception.

2. The method of claim 1, further comprising designing sub-chain codebooks using one of three methods:
   - Sim-Max method: generating sub-chain beams that closely resemble the shape of full-chain beams.
   - SC-Max method: optimizing the spherical coverage of sub-chain codebooks without considering one-to-one mapping with full-chain beams.
   - BC-SC-Max method: balancing both similarity to full-chain beams and spherical coverage.

3. The method of claim 2, wherein the BC-SC-Max method partitions the unit-sphere into regions covered by full-chain beams and designs sub-chain beams that maximize the radiation pattern over these regions.

4. A mmWave 5G device comprising:
   - An antenna array capable of operating in both full-chain and sub-chain modes.
   - A controller configured to detect overheating or high power consumption and activate a sub-chain beam operation by deactivating part of the antenna array in uplink transmission while keeping the full array active for downlink reception.

5. The device of claim 4, wherein the controller is further configured to design sub-chain codebooks using one of three methods: Sim-Max, SC-Max, and BC-SC-Max.

6. A non-transitory computer-readable medium storing instructions that, when executed by a processor, cause the processor to perform the method of claim 1.

7. The medium of claim 6, wherein the instructions further cause the processor to design sub-chain codebooks using one of three methods: Sim-Max, SC-Max, and BC-SC-Max.

## Drawings
[Figures illustrating the simulation results and beam patterns for the three methods]

---

This patent application provides a comprehensive solution for managing power consumption and thermal issues in mmWave 5G devices while maintaining high communication efficiency. The proposed sub-chain beam operation and codebook design methods offer practical and effective ways to utilize mmWave frequencies more efficiently.