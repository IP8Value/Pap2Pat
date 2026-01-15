Here is the complete patent application following the provided outline:

# DESCRIPTION

## STATEMENT REGARDING FEDERAL FUNDING

The invention described herein was made with government support under Grant No. [REDACTED] awarded by [REDACTED]. The government has certain rights in the invention.

This invention was supported in part by funding from [REDACTED] federal agency. The United States government may have certain rights to this invention as provided by the terms of the funding agreement.

## SUMMARY

The present invention relates to systems and methods for optimizing biological activities through computational modeling of metabolic networks with molecular crowding constraints. Traditional flux balance analysis (FBA) approaches for modeling cellular metabolism have significant limitations as they fail to account for physical constraints imposed by the crowded intracellular environment. The disclosed invention overcomes these limitations by incorporating cytoplasmic molecular crowding and reaction kinetics parameters into flux balance calculations.

The optimization method of the present invention enables calculation of optimal cell growth rates, substrate utilization patterns, metabolic flux reorganization, maximum metabolic rates, optimal metabolite concentrations, and enzyme activities. The invention provides computer-implemented methods and apparatus for implementing these calculations to optimize biological activities. Specifically, the methods allow for calculation and initiation of optimal cell culture parameters, control of substrate usage order, and achievement of optimal function in biochemical reaction networks.

Through iterative calculation and alteration of biochemical reactions, the invention enables determination and implementation of optimal culture conditions for living cells. The methods involve constructing the genetic makeup of cells, placing them in culture, and cultivating them under optimized conditions to achieve desired performance. The optimization procedures account for ribosome density and mitochondrial compartments within cells.

The invention further provides computer-readable media containing instructions for implementing the optimization methods. The disclosed systems include sensors, culture vessels, heating/cooling elements, reservoirs, dispensing mechanisms, analytical devices, and communication subsystems for monitoring and controlling culture conditions. Remote access to reaction parameters enables automated optimization of biological systems.

## DETAILED DESCRIPTION

The following terms shall have the meanings set forth below for purposes of this disclosure:

"Flux balance analysis" (FBA) refers to a mathematical approach for analyzing flow of metabolites through metabolic networks that assumes steady-state conditions where internal metabolite concentrations remain constant.

"Molecular crowding" refers to the excluded volume effects caused by high concentrations of macromolecules in cellular environments that influence reaction rates and equilibria.

"Stoichiometric matrix" is a mathematical representation of metabolic networks where rows correspond to metabolites and columns represent biochemical reactions.

"Steady state assumption" presumes that the concentration of internal metabolites remains constant over time, balancing production and consumption fluxes.

The invention employs flux balance calculations for cell cultures that incorporate two critical physical constraints: cytoplasmic molecular crowding and reaction kinetics parameters. These constraints are implemented through novel modifications to traditional flux balance analysis frameworks.

The flux balance model of cellular metabolism according to the invention utilizes a stoichiometric matrix representation of metabolic networks. Under the steady state assumption, the system is described by the equation S·v = 0, where S is the stoichiometric matrix and v is the flux vector. The invention introduces additional constraints to this framework, including enzyme concentration constraints represented through a crowding coefficient.

The crowding coefficient accounts for the effective volume occupied by each enzyme in the crowded cytoplasmic environment. This parameter is derived from experimental measurements of enzyme turnover rates and molecular volumes. The incorporation of crowding coefficients enables more accurate prediction of metabolic behaviors under physiological growth conditions.

The optimization method for biological activities involves calculating optimal cell culture parameters through iterative solution of the constrained flux balance problem. The method proceeds by first representing biochemical reactions in computer memory, then applying optimization methods including linear and non-linear optimization with linear constraints, and simulated annealing approaches.

Following initial calculation of optimal properties, the method involves altering the list of reactions and recomputing optimal properties until desired performance is achieved. The living cells are then cultured under the optimized conditions, with their genetic makeup constructed to contain the specified biochemical reactions. Cells are placed in culture and cultivated for sufficient time to evolve toward desired performance.

The biochemical reaction networks addressed by the invention include complete metabolic networks reconstructed from annotated genome sequences and biochemical data. The reconstruction process integrates genomic, proteomic, and physiological data to build comprehensive network models. Analysis of reconstructed networks enables determination of optimal properties through the described optimization methods.

The optimization procedure involves defining factors leading to closed solution spaces, performing optimization, and comparing calculated behaviors to experimental data. Additional constraints relating to cytoplasmic molecular crowding and reaction kinetics are incorporated to improve prediction accuracy. The methods enable prediction of optimal uses of biochemical reaction networks beyond limitations of natural organisms.

For natural organisms with intact networks, growth competition and selection pressures limit achievable performance. The invention provides methods to design biochemical reaction networks that overcome these limitations through in silico optimization and subsequent culture under optimized conditions. The iterative design procedure involves perturbing wild type networks, resolving optimality issues computationally, and implementing solutions through culturing methods.

Desired performance characteristics may be specified as qualitative characteristics or quantitative values. Examples include maximum biomass production, specific metabolite yields, or optimal growth rates. The optimization method utilizes computer systems with databases containing information about biochemical reaction networks, biomolecular sequences, and genomic sequences.

The computer system components include user interfaces for receiving performance selections, processing units for implementing optimization algorithms, and modules for interacting with biological databases. The system compares biochemical reaction networks to identify differences and receives data from cell cultures to control external devices.

Implementation involves generating computer-readable program code that executes the optimization procedures. The system enables adaptive evolution of cultured strains by continuously monitoring culture conditions and adjusting parameters to drive populations toward desired performance. The methods are applicable to virtually any cell type after appropriate biochemical reaction network characterization.

Characterization begins with genome sequencing and gene identification to determine the genetic makeup of a cell. Biochemical reactions are then constructed to meet desired performance criteria through genetic manipulations that add or subtract reactions from the network. Expression of regulatory components may be altered to fine-tune network performance.

Cells are placed in culture under specified environmental conditions, with optimal parameters determined through the optimization procedure. Culture conditions are continuously monitored and adjusted as necessary using computer programs that automatically bring cultures to optimal parameters. Continuous culture systems with computerized monitoring enable precise control of media flow and constituent addition.

The computer system is configured and programmed to maintain optimal culture conditions while allowing cells to adapt through evolutionary processes. Extended cultivation periods enable optimization of metabolic networks, with accelerated evolution possible through chemical mutagens or radiation. Genetic alterations introduce desired biochemical reactants into living cells prior to culture.

During adaptive evolution, growth and metabolic behavior are monitored through measurements of oxygen uptake rates, substrate uptake rates, and growth rates. Data points are plotted on phenotype phase planes to track progression toward optimal performance. Byproduct secretion is monitored using HPLC or other analytical methods.

Culture monitoring includes determination of correlation between dry weight and optical density for evolved strains. Cultures are inspected for contamination or co-evolution with mutant subpopulations. Key parameters including optical density, inoculation time, inoculum volume, growth rate, and contamination signs are logged. Samples are frozen at intervals for further use.

The computer-implemented method achieves optimal function of biochemical reaction networks through calculation of optimal cell culture parameters using flux balance analysis with molecular crowding constraints. Computational optimization methods are applied to biochemical reactions, with iterative alteration and recomputation until optimal function is reached.

Optimal cell culture parameters are initiated and maintained in cell culture systems. The methods calculate maximum metabolic rates, optimal metabolite concentrations, and enzyme activities through application of computational optimization to kinetic models of metabolic pathways. The genetic makeup of cells is constructed to contain specified biochemical reactions before placement in culture.

Cells are cultivated under specified environments for sufficient time to allow evolution toward desired optimal function. The methods account for ribosome density as a measure of ribosomal, enzyme-associated, and non-metabolic proteins. Mitochondrial compartments are treated as subcellular compartments in the optimization framework.

The invention provides computer-readable media containing stored instructions for implementing the computational models. Devices according to the invention comprise computer-readable media and processors for executing instructions, along with additional components for practical implementation.

Implementation components include sensors for monitoring culture conditions, culture vessels, heating/cooling elements, and reservoirs for storing cell culture media. Mechanisms for dispensing media and analyzing samples are integrated with display systems, analytical devices, and communication subsystems. The system enables remote access to reaction parameters for monitoring and control.

## EXAMPLES

The following abbreviations are used in the examples:

FBA: Flux Balance Analysis
FBAwMC: Flux Balance Analysis with Molecular Crowding
MC: Molecular Crowding
OD: Optical Density
HPLC: High Performance Liquid Chromatography
GC-MS: Gas Chromatography-Mass Spectrometry
NMR: Nuclear Magnetic Resonance
MIDA: Mass Isotopomer Distribution Analysis

### Example 1

This example studies the impact of limited solvent capacity on Escherichia coli cell metabolism, demonstrating the relevance of crowding constraints for fast-growing cells. The FBAwMC model predicts a metabolic switch between low and high nutrient abundance conditions that was verified through flux measurements of several reactions.

Experimental observations showed partial agreement with model predictions. Gene expression and enzyme activity measurements revealed that the metabolic switch is controlled at the enzyme activity level rather than through transcriptional regulation. These findings have potential relevance to experimental observations in other organisms.

Crowding coefficients for E. coli proteins were estimated using enzyme turnover rates obtained from the BRENDA database. Implementation of FBA with Molecular Crowding involved solving an optimization problem to maximize biomass production rate under crowding constraints. Crowding coefficients were modeled as noise parameters in the optimization framework.

The model predicted fluxes for all reactions in the metabolic network under different carbon source conditions. Simulations of increasing carbon source concentration in growth medium revealed non-linear responses in metabolic fluxes. Analysis of flux behavior as a function of growth rate demonstrated the critical role of crowding constraints in fast-growing cells.

The bacterial strain MG1655 was cultured under controlled growth conditions for experimental validation. Biomass samples were harvested for flux measurements using GC-MS and NMR metabolome mapping platforms. Metabolic enzyme activity assays were performed with total protein concentration determined by Bradford assay.

Mass isotopomer analysis (MIDA) provided flux measurements that were analyzed statistically using Student's t-test. Glycogen glucose and RNA ribose stable isotope studies involved acid hydrolysis of cellular RNA followed by derivatization and mass spectral analysis. Lactate and glutamate extraction procedures were optimized for accurate flux determination.

Results demonstrated that limited solvent capacity significantly constrains metabolic rates in fast-growing cells. Crowding coefficients estimated from experimental data were incorporated into the FBAwMC model. The model predicted changes in effective metabolic efficiency objectives that were validated experimentally.

At physiological growth conditions, the solvent capacity constraint became increasingly relevant, predicting a metabolic switch characterized by changing criteria of metabolic efficiency. The model accurately predicted redistribution of metabolic fluxes and acetate excretion at high growth rates.

Comparisons between FBAwMC-predicted fluxes and experimental values showed strong correlation. Regulatory mechanisms controlling the metabolic switch were identified through combined analysis of enzyme activities and flux rates. Notably, mRNA levels of enzyme-encoding genes showed poor correlation with measured fluxes, indicating post-transcriptional control.

The significance of solvent capacity constraints for systems biology was demonstrated through successful incorporation into the FBA modeling framework. Model predictions for E. coli metabolism considering reaction kinetics via crowding coefficients provided novel insights into metabolic regulation.

Interpretation of the metabolic switch using solvent capacity constraints explained observed behaviors including acetate excretion patterns. The model's maximization of biomass production rate objective yielded results consistent with expectations of fastest growth rates under given conditions.

### Example 2

This example describes development of a modified FBA model incorporating solvent capacity constraints to predict maximum growth rates. Experimental tests showed good agreement between model predictions and observed growth behaviors, supporting the macromolecular crowding constraint hypothesis.

The FBAwMC modeling framework was implemented by defining an optimization problem that modeled crowding coefficients as noise parameters with gamma distribution. Sensitivity analysis confirmed robustness of results to variation in crowding coefficient values.

Maximum growth rates were predicted for each carbon source condition, with average crowding coefficients fitted to experimental data. The model successfully predicted temporal order of substrate uptake by considering initial substrate concentrations and integrating differential equations describing uptake kinetics.

Three distinct FBAwMC problems were solved to fully characterize system behavior under different nutrient conditions. Crowding coefficients were estimated from experimental measurements by decomposing proportionality factors related to enzyme turnover rates.

Growth experiments used M9 minimal medium under controlled conditions to assess transcriptome states and determine maximum growth rates. Continuous cultivation methods enabled precise calculation of growth rates and residual carbon source concentrations.

Microarray analysis identified genes with sequence-specific hybridization patterns under different nutrient conditions. Analysis of top 150 differentially expressed genes revealed clusters associated with specific metabolic phases. Hierarchical clustering of time-series gene expression data identified three major metabolic phases.

Results demonstrated that FBA with molecular crowding accurately predicts relative maximum growth rates and substrate hierarchy utilization patterns. The model captured surrogate markers of cellular metabolism and showed correlation between predicted substrate utilization sequences and gene expression patterns.

Activation of stress programs upon switching metabolic phases was observed experimentally and predicted by the model. Principal component analysis and probabilistic clustering methods based on hidden Markov models confirmed the three-phase metabolic behavior.

Discussion of results identified principles defining growth and substrate utilization modes. Experimental results confirmed three metabolic phases predicted by the model, with global mRNA expression data indicating partial stress response during transitions.

The FBAwMC model captured main features of metabolic activities, showing strong correlation between in vivo maximal growth rates and in silico predictions. Model analysis revealed that solvent capacity of cytoplasm determines maximum growth rates, with cells preferentially consuming carbon sources supporting highest growth.

Two discrepancies between model predictions and experimental observations were noted: higher than predicted acetate secretion and earlier substrate uptake than predicted. These were attributed to underestimation of macromolecular crowding impacts and contributions from non-metabolic cell components.

The model demonstrated that maximum enzyme concentration is a key constraint shaping substrate utilization hierarchy. Regulatory mechanisms in E. coli and other organisms were interpreted through the constrained optimization framework, providing new insights into metabolic regulation.

### Example 3

This example introduces flux balance analysis in Saccharomyces cerevisiae incorporating molecular crowding and kinetic modeling constraints. The study tests the hypothesis of optimal intracellular resource use through analysis of the glycolysis pathway.

A kinetic model of glycolysis was developed with rate equation models for each reaction in the pathway. Enzyme kinetic parameters including catalytic constants were obtained from experimental estimates. Cell density and specific volume measurements provided additional constraints for the optimization framework.

The limited solvent capacity constraint was implemented through derivation of rate equations incorporating crowding coefficients. Analysis of a hypothetical three-metabolite pathway demonstrated application of Michaelis-Menten kinetics under crowding constraints.

Application to S. cerevisiae glycolysis investigated dependency of glycolysis rate on metabolite concentrations. Global optimization of metabolite concentrations predicted optimal values that were compared with experimental measurements. Enzyme activities were predicted and tested against observed values.

The modeling approach explored alternative optimization objectives beyond maximum glycolysis rate. Results demonstrated that limited solvent capacity significantly constrains achievable metabolic rates, particularly in fast-growing cells.

Comparison with previous work in E. coli highlighted advantages and limitations of the modeling approach. The full kinetic model of glycolysis provided accurate predictions of optimal intermediate metabolite concentrations and enzyme activities.

Discrepancies between predictions and experimental values were analyzed, leading to proposed improvements in the prediction methodology. Physical constraints of total cell volume were incorporated through consideration of enzyme molar volumes and densities.

The modeling framework's advantages include ability to predict metabolite concentrations and enzyme activities from first principles. Limitations include reliance on steady state approximations that cannot model dynamical processes such as observed metabolite concentration oscillations.

### Example 4

This example introduces an alternative glycolysis pathway analysis incorporating molecular crowding constraints in a genome-scale model of human cell metabolism. The study examines ATP generation in normal cells and the Warburg effect in cancer cells.

The flux balance model defines nutrient import reactions and compartment-specific reactions in mitochondria and cytoplasm. Compartment densities are incorporated into the optimization framework through volume fraction constraints.

Model parameters were estimated including costs of molecule import, effective turnover numbers, and crowding coefficients for enzymes, ribosomes, and mitochondria. Flux balance equations for proteins accounted for synthesis, degradation, and effective dilution processes.

Modeling the alternative glycolysis pathway revealed changes in relative macromolecular densities under different growth conditions. Predictions showed varying volume fractions occupied by enzymes, mitochondria, and ribosomes depending on metabolic state.

Impact of altering model parameters demonstrated a metabolic switch from low to high proliferation rates characterized by changes in glucose uptake, glutamine metabolism, and pyruvate processing enzyme activities. The model predicted a novel ATP-producing pathway operating through a one-carbon metabolism cycle.

Kinetic modeling of the reaction cycle defined reaction rates and parameters that were analyzed through system optimization. Results related to Myc regulation in tumorigenesis were supported by analysis of gene expression data.

The model challenges general notions of glycolysis regulation by proposing molecular crowding as a fundamental constraint. Competition for intracellular space creates an upper bound for oxidative phosphorylation capacity that motivates alternative pathway utilization.

In silico analysis results were supported by experimental observations including PKM2 isoform expression patterns and upregulation of serine/glycine biosynthesis pathways. The novel ATP-producing pathway was analyzed for ATP yield, co-factor balance, and potential evolutionary advantages.

Hierarchy of ATP yield and molecular crowding constraints explained observed lactate and alanine production patterns under fluctuating hypoxia levels. The model identified potential enzyme targets for metabolic intervention and provided insights into cancer cell metabolism.

## Materials and Methods

Metabolic network reconstructions were downloaded from public databases and modified to incorporate organism-specific parameters. Crowding coefficients were calculated from enzyme kinetic data and molecular volumes. Sensitivity analysis evaluated robustness of predictions to parameter variation.

Macromolecular composition was estimated from proteomic and ribosome profiling data. Maintenance parameters were determined through chemostat experiments at varying dilution rates. Optimization problems were solved using linear and nonlinear programming techniques.

Microarray data analysis employed established bioinformatics pipelines. Model-predicted relative cell volume fractions were calculated and compared with experimental measurements. Abbreviations used throughout are defined in the respective sections.

Variations of the described embodiments may incorporate alternative optimization algorithms, additional physical constraints, or modified culture systems. The invention claims priority to provisional patent applications [REDACTED] filed on [REDACTED].