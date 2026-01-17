# DESCRIPTION

## BACKGROUND

Protein structure prediction is a fundamental challenge in molecular biology and bioinformatics. Accurate prediction of protein structures is crucial for understanding the function and behavior of proteins, which play essential roles in various biological processes. Traditional methods for protein structure prediction, such as molecular dynamics (MD) simulations and homology modeling, often require significant computational resources and may not be feasible for proteins without suitable homologs or in unique environmental conditions.

This invention introduces a novel ab initio physical drift and diffusion-based protein structure prediction simulation that can run efficiently on a desktop PC. The method leverages first-principle forces and physical kinetics to predict protein structures, offering a computationally efficient and accurate alternative to existing techniques. The model accounts for multiple energy considerations, including electrostatic forces, displacement forces, thermal forces, and global entropy changes, to simulate the dynamic folding of proteins. This approach enables the prediction of both secondary and tertiary structures with high accuracy and speed, making it particularly useful for proteins where homologs are unavailable or where environmental conditions differ significantly from those used in experimental studies.

## SUMMARY

The invention provides a method and system for predicting protein structures using an ab initio physical drift and diffusion-based simulation. The method involves the following key steps:

1. **Initialization**: The amino acid sequence of the protein is input into the simulation. Each residue's hydrophobicity and charge are assigned to the nearest backbone atom.

2. **Secondary Structure Prediction**:
   - The algorithm scans the sequence to identify regions with a high propensity for forming secondary structures (alpha helices and beta sheets).
   - Hydrophobic and electrostatic forces are calculated for each residue, and the balance between these forces determines the formation of secondary structures.
   - Secondary structures are directly generated and inserted into the protein model using graphical software, ensuring that these structures are immutable during subsequent tertiary structure simulation.

3. **Tertiary Structure Prediction**:
   - The protein is allowed to drift and diffuse toward a lower energy state by simulating the relative motion of protein substructures.
   - Forces acting on each alpha carbon bond pair are calculated, and the bond pair with the largest net torque is allowed to move in each time step.
   - The simulation continues until the protein reaches a stable conformation, and the final structure is refined using molecular dynamics (MD) relaxation.

4. **Energy Minimization**:
   - The energy of the protein is tracked throughout the simulation by calculating the work done during each fold or motion.
   - The final structure is further optimized using MD relaxation to ensure energy minimization and structural accuracy.

The invention offers several advantages over existing methods:
- **Computational Efficiency**: The simulation can run on a desktop PC, making it accessible and cost-effective.
- **High Accuracy**: The method achieves near state-of-the-art accuracy in predicting both secondary and tertiary structures.
- **Environmental Sensitivity**: The model can simulate protein folding under various environmental conditions, providing insights into the effects of pH, temperature, and other factors on protein structure.
- **No Homolog Requirement**: The ab initio approach does not rely on homologous sequences, making it applicable to a broader range of proteins.

## DETAILED DESCRIPTION

### Initialization

The method begins by initializing the protein structure based on its amino acid sequence. Each residue's hydrophobicity and charge are assigned to the nearest backbone atom. This tagging process ensures that the physical properties of each residue are accurately represented in the simulation.

### Secondary Structure Prediction

#### Hydrophobic and Electrostatic Force Calculation

The algorithm scans the amino acid sequence to identify regions with a high propensity for forming secondary structures. The hydrophobic and electrostatic forces for each residue are calculated using the following equations:

1. **Electrostatic Force**:
   \[
   F_{\text{elect}} = \frac{q_1 q_2}{4\pi \epsilon r^2}
   \]
   where \( q_1 \) and \( q_2 \) are the charges of the residues, \( \epsilon \) is the dielectric constant, and \( r \) is the distance between the residues.

2. **Displacement Force**:
   \[
   F_{\text{disp}} = -\frac{4\beta |q|^2}{r^5}
   \]
   where \( \beta \) is a constant related to the nonpolar volume and other constants, and \( q \) is the charge of the residue.

3. **Thermal Force**:
   \[
   F_{\text{thermal}} = \gamma \sqrt{kT}
   \]
   where \( \gamma \) is a constant, \( k \) is the Boltzmann constant, and \( T \) is the temperature.

4. **Global Entropy Change**:
   The global entropy change is estimated by considering the proposed change in protein structure. The entropy change is used to determine a new mobility, which in turn affects the protein's motion.

#### Secondary Structure Formation

The balance between hydrophobic and electrostatic forces determines the formation of secondary structures. The algorithm follows a set of rules to predict the type and location of secondary structures:

1. **Alpha Helix Formation**:
   - If the hydrophobic character of the residues is strong and unopposed by dielectric displacement forces, an alpha helix will form.
   - The summation of charges and the product of charges are used to determine the magnitude of the electrostatic forces.

2. **Beta Sheet Formation**:
   - If alpha helix formation is blocked by the dielectric displacement force induced by intervening charged residues, a beta sheet region will form.
   - The summation of the magnitude of charges and the summation of the hydrophobic character of each residue in a 5-residue bracket are used to determine the formation of beta sheets.

The algorithm is sensitive to small changes in charge and hydrophobic characteristics, ensuring accurate secondary structure prediction.

### Tertiary Structure Prediction

#### Drift and Diffusion Simulation

The protein is allowed to drift and diffuse toward a lower energy state by simulating the relative motion of protein substructures. The relative motion of one part of the protein relative to the other parts is determined by allowing the alpha carbon bond pair with the largest net torque to move in each time step. The net torque is calculated as the sum of the actual force and the effective thermal force multiplied by the appropriate lever arm length.

The simulation uses a Markov process to allow the protein to move toward a lower energy state. The mobility of the protein is adjusted based on the global entropy change, with directions that increase entropy proceeding with higher mobility.

#### Energy Minimization

The energy of the protein is tracked throughout the simulation by calculating the work done during each fold or motion. The energy is calculated using the following equation:
\[
E = E_0 - \sum (F \cdot \theta \cdot r_0)
\]
where \( E_0 \) is the initial energy, \( F \) is the force applied, \( \theta \) is the calculated angle change, and \( r_0 \) is the average lever arm length of the protein.

The final structure is further optimized using molecular dynamics (MD) relaxation to ensure energy minimization and structural accuracy. The MD relaxation step employs nominal AMBER10 default parameters for dielectric constant and other parameters, and it can be carried out quickly on a supercomputer.

### Applications and Advantages

The invention has several applications and advantages:

1. **Protein Function Prediction**: The method can be used to predict the function of proteins by accurately simulating their structure and behavior under various environmental conditions.
2. **Drug Design**: The model can provide insights into the mechanisms of protein-protein interactions and the binding of small molecules, aiding in the design of new drugs.
3. **Biological Pathways**: The simulation can help elucidate the folding pathways of proteins, contributing to a better understanding of biological processes.
4. **High-Speed Simulation**: The method can run on a desktop PC, making it accessible and cost-effective for researchers and laboratories with limited computational resources.
5. **Environmental Sensitivity**: The model can simulate protein folding under various environmental conditions, providing insights into the effects of pH, temperature, and other factors on protein structure.
6. **No Homolog Requirement**: The ab initio approach does not rely on homologous sequences, making it applicable to a broader range of proteins.

In conclusion, the invention provides a robust and efficient method for predicting protein structures using an ab initio physical drift and diffusion-based simulation. The method offers high accuracy, computational efficiency, and environmental sensitivity, making it a valuable tool for protein structure prediction and functional analysis.