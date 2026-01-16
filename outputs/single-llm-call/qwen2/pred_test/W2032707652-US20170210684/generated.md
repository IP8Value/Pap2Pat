# DESCRIPTION

## GOVERNMENT FUNDING

This invention was made with government support under Award DE-FG02-12ER16362 awarded by the Department of Energy Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates to the field of materials science and chemical engineering, specifically to the identification and use of zeolites for challenging separations and chemical transformations. More particularly, the invention pertains to the discovery of optimal zeolites for the separation of ethanol from aqueous solutions and the transformation of long-chain hydrocarbons through predictive materials modeling.

## BACKGROUND

Crude oil remains the dominant source for transportation fuels and chemical feedstocks. Improving the efficiency of oil refining is crucial for extending the supply and reducing the cost of current petroleum products. In the 1950s, the emergence of crystalline zeolites with sub-2 nm internal pores revolutionized the petrochemical industry by providing shape and size-selective sorbents and catalysts. For instance, zeolites are used to catalyze the transformation of linear long-chain alkanes into slightly branched alkanes of similar molecular weight, which enhances the pour point and viscosity index of lubricant oils. Similar transformations are also beneficial for diesel and other fuel oils, where shorter alkanes are involved. These hydroisomerization reactions depend on a delicate balance between the degree of framework confinement and the size of alkane molecules. An optimal zeolite possesses a high affinity for linear alkanes but a low affinity for branched isomers, ensuring that the desired mono-branched products are not cracked into smaller species.

Separating ethanol from its aqueous solution, a critical process in biofuel production, currently relies on energy-intensive distillation. Nearly defect-free silicalite-1, an all-silica zeolite with the MFI framework type, has been proposed as an effective sorbent and membrane for this separation. All-silica zeolites are inherently hydrophobic, but the adsorption of ethanol can promote water co-adsorption through hydrogen bond formation, thereby lowering the selectivity. For this application, the desired zeolite should have a pore/channel system that accommodates ethanol molecules but disfavors hydrogen bonding with water molecules.

Experimental testing of all existing zeolites for a given application is time-consuming and labor-intensive, often infeasible when a synthesis protocol for the material with the desired composition is not yet developed. The number of potentially synthesizable zeolites is vast, and some structures from the predicted crystallography open database (PCOD) may possess superior characteristics. Predictive modeling to screen and select optimal candidate materials is therefore highly attractive. Prior screening studies have focused on single-component adsorption of small, rigid, non-hydrogen-bonding molecules such as short hydrocarbons, carbon dioxide, and hydrogen. However, screening sorbents and catalysts for complex mixtures composed of large, articulated molecules or polar, hydrogen-bonding molecules has been an intractable problem due to the need for advanced algorithms and accurate force fields.

## SUMMARY

The present invention addresses the need for efficient and accurate methods to identify optimal zeolites for challenging separations and chemical transformations. Specifically, the invention provides a method for high-throughput screening of zeolites to identify those with exceptional selectivities for ethanol purification from aqueous solutions and the transformation of long-chain hydrocarbons.

The method comprises the following steps:
1. **Data Collection**: Gathering framework structures from the Structure Commission of the International Zeolite Association (IZA-SC) and the Predicted Crystallography Open Database (PCOD).
2. **Force Field Selection**: Using transferable potentials for phase equilibria (TraPPE) and TIP4P force fields to model zeolites, hydrocarbons, ethanol, and water.
3. **Simulation Methods**: Employing configurational-bias Monte Carlo simulations in the grand-canonical ensemble (CB-GCMC) to compute sorbate loadings as a function of solution-phase concentrations or partial pressures.
4. **Performance Metrics**: Defining performance metrics for ethanol/water separation and hydrocarbon dewaxing, such as ethanol selectivity, loading, and performance score.
5. **Screening Workflow**: Implementing a two-step screening workflow to identify top-performing zeolites at various solution-phase concentrations and pressures.
6. **Validation**: Conducting experimental validation of the predicted zeolites to ensure their effectiveness in real-world applications.

The invention also includes the identified zeolites and their use in separation processes and chemical transformations, particularly for ethanol purification and hydroisomerization dewaxing.

## DETAILED DESCRIPTION

### Example

The invention is exemplified by the identification of optimal zeolites for ethanol/water separation and hydrocarbon dewaxing. The method involves a multi-step screening workflow, efficient sampling algorithms, accurate force fields, and a two-level parallel execution hierarchy utilizing up to 131,072 compute cores on a leadership-class supercomputer.

#### Ethanol/Water Separation

Sugar fermentation produces ethanol with solution-phase concentrations ranging from 5 to 15 wt% at temperatures between 298 and 323 K. The adsorption selectivity of ethanol over water generally decreases with solution-phase ethanol concentration and temperature. For separation processes based on equilibrium adsorption, the final concentration of the raffinate determines the composition of the adsorbed phase (retentate). The performance metric \( P_{\text{EtOH}} \) is defined as the product of ethanol selectivity \( S_{\text{EtOH}} \) and loading \( Q_{\text{EtOH}} \), which is robust and effective in identifying top structures at a given target concentration.

Monte Carlo simulations are conducted at a solution-phase concentration of 0.12 wt% and a temperature of 323 K to screen all framework structures available from the IZA-SC. The 64 structures with the highest \( P_{\text{EtOH}} \) values and satisfying the additional constraint \( Q_{\text{EtOH}} > Q_{\text{water}} \) are retained for simulations at five higher concentrations (0.43, 1.4, 4.5, 15, and 41 wt%).

FER is the top-ranked structure at 0.12 and 0.43 wt% due to its exceptionally high \( S_{\text{EtOH}} \), and remains among the top 10 structures at higher concentrations. By raising the concentration to 0.43 wt%, 18 structures can reach a sufficiently high \( S_{\text{EtOH}} \) to exceed the ethanol/water azeotropic point. The adsorption characteristics of the top five framework types at all six concentrations are compared with MFI, and FER consistently outperforms MFI.

#### Hydrocarbon Dewaxing

Hydroisomerization is a modern dewaxing technique that uses certain zeolites as bifunctional catalysts with incorporated group VIII metals to selectively convert linear long-chain alkanes to slightly branched isomers with a high viscosity index. The framework structures of high-performing zeolites for this application promote the adsorption of linear alkanes and have high separation factors over branched alkanes.

The method screens 433,000 zeolite structures in the IZA-SC and PCOD databases. An equimolar mixture of n-octadecane (C18), n-tetracosane (C24), n-triacontane (C30), 2-methyl and 4-methylheptadecane (2C17 and 4C17), and 2,2-dimethylhexadecane (22C16) is used to represent the complex hydrocarbon feed. Three performance indicators are constructed to characterize performance: (i) high affinity towards linear alkanes, (ii) high selectivity for linear-versus-branched alkanes, and (iii) low selectivity between linear alkanes of different lengths.

The first screening step aims to reduce the number of candidate materials using relatively short simulations performed at 573 K and the infinite-dilution limit. The top 64 structures in the IZA-SC database and the top 1,024 structures in the PCOD database are retained for longer simulations at the same conditions and at 3 MPa, a typical operation condition for the hydroisomerization process.

The top 10 structures from each database are compared, and the top 3 (ATO, MRE, and AFO) and another four (AEL, MTT, FER, and TON) from the top 10 IZA-SC structures are known to excel for this petrochemical application. The top 10 PCOD structures exhibit performance scores that are about two orders of magnitude higher than those for the high-performing IZA-SC structures, primarily due to their exceptionally high selectivity for linear alkanes.

### Methods

#### Framework Structures

The IZA-SC database consists of a set of idealized framework structures and other experimentally determined structures. The idealized structure for each framework type is obtained by geometric refinement with prescribed interatomic distances, assuming a (hypothetical) SiO2 composition, and in the highest possible symmetry space group of the framework type. The experimental structures are included for the screening if they contain only O, Si, Al, P, or H atoms after removal of any solvent molecules and ions and resolution of partial occupation numbers by proportional random assignments on the unit cell level, followed by replacement of Al and P atoms with Si atoms. Together, the idealized and experimental IZA-SC structures yield 402 unique sorbents. The larger PCOD database was constructed by enumerating space groups, unit cells, density, and sampling coordinates of Si atoms in the irreducible unit. The resulting 2.6 million candidate structures were geometry optimized, and based on an energetic criterion, 331,172 structures are considered as thermodynamically accessible.

#### Force Fields

The transferable potentials for phase equilibria (TraPPE) and TIP4P force fields are used to model zeolites, hydrocarbons, ethanol, and water. The TraPPE-zeo force field for all-silica zeolites treats dispersive and first-order electrostatic interactions in a balanced manner, allowing one to study a wide range of sorbate molecules. The combination of TraPPE-zeo and the sorbate models has been extensively validated in systems closely related to the two applications reported in this work, including the adsorption and diffusion of alkanes, CO2, alcohols, and H2O in different zeolite structures, across a wide range of temperatures and pressures.

#### Simulation Methods

Configurational-bias Monte Carlo simulations in the grand-canonical ensemble (CB-GCMC) are used to compute sorbate loadings as a function of either the concentration of the ethanol/water bulk solutions or partial pressures for the hydrocarbon mixtures. The chemical potentials required for these simulations are obtained from previous Gibbs ensemble simulations with explicit solution phases or determined from liquid-phase simulations in the isobaric-isothermal ensemble for the alkanes. The coupled-decoupled configurational-bias Monte Carlo (CD-CBMC) algorithm is used to enhance the sampling of intramolecular degrees of freedom and to improve the acceptance rates of GCMC insertion/deletion moves. In the infinite-dilution limit, these CB-GCMC simulations also yield directly Henry's constant and adsorption enthalpy.

To carry out the energy grid tabulation and CB-GCMC simulations in a high-throughput fashion, a two-level parallel execution hierarchy is implemented, exploring simultaneously 2^7 to 2^14 zeolite structures and accelerating the simulations for each structure by spreading the computational load over 2^2 to 2^8 compute cores. These massively parallel screening calculations are performed on Mira, a leadership-class supercomputer at Argonne National Laboratory.

### Exemplary Embodiments

#### Ethanol/Water Separation

1. **Initial Screening**: Monte Carlo simulations are performed at a solution-phase concentration of 0.12 wt% and a temperature of 323 K to screen all 402 IZA-SC structures. The 64 structures with the highest \( P_{\text{EtOH}} \) values and satisfying the additional constraint \( Q_{\text{EtOH}} > Q_{\text{water}} \) are retained.
2. **Secondary Screening**: Simulations are conducted at five higher concentrations (0.43, 1.4, 4.5, 15, and 41 wt%) to identify the top-performing zeolites.
3. **Validation**: Experimental validation is performed using high-quality samples of MFI (silicalite-1) and FER (ferrierite) to measure ethanol loading and compare with simulation predictions.

#### Hydrocarbon Dewaxing

1. **Initial Screening**: Short simulations are performed at 573 K and the infinite-dilution limit to screen 433,000 zeolite structures in the IZA-SC and PCOD databases. The top 64 structures in the IZA-SC database and the top 1,024 structures in the PCOD database are retained.
2. **Secondary Screening**: Longer simulations are conducted at the same conditions and at 3 MPa to obtain better statistics for the infinite-dilution properties and to compute the liquid-mixture properties.
3. **Validation**: Experimental validation is performed to confirm the predicted performance of the top-performing zeolites.

By following the method of the present invention, optimal zeolites for challenging separations and chemical transformations can be efficiently identified, leading to significant improvements in industrial processes.