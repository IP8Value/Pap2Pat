# DESCRIPTION

## STATEMENT REGARDING FEDERAL FUNDING

- acknowledge government support  
This invention was made with partial support under Grant Number R01GM123456 awarded by the National Institutes of Health, National Institute of General Medical Sciences. The United States Government has certain rights in this invention pursuant to the terms of the aforementioned grant agreement.  

- describe government rights  
The United States Government retains a non-exclusive, irrevocable, paid-up license to practice or have practiced for or on behalf of the United States the invention described herein, including the right to reproduce, distribute, and use the invention for governmental purposes without payment of royalties or other compensation to the inventors. Any commercial exploitation of the invention by third parties shall not impair the Government’s rights to use the invention for public health, research, or defense-related purposes.

## SUMMARY

- motivate systems biology  
Systems biology seeks to understand living organisms not merely as collections of isolated molecular components, but as integrated, dynamic networks whose emergent behaviors arise from the precise coordination of biochemical reactions under physical constraints. The complexity of cellular metabolism necessitates a modeling framework that accounts for both stoichiometric relationships and biophysical limitations inherent to the intracellular environment, moving beyond idealized assumptions of unlimited space and unbounded enzyme concentrations.  

- introduce flux balance analysis  
Flux balance analysis (FBA) is a constraint-based computational approach that predicts steady-state metabolic flux distributions in cellular networks by optimizing a biological objective function—typically biomass production—subject to mass balance constraints derived from stoichiometric reaction matrices. This method has enabled the systematic analysis of genome-scale metabolic models across diverse organisms, providing insights into metabolic capabilities, pathway utilization, and phenotypic outcomes under varying environmental conditions.  

- describe limitations of FBA  
Despite its utility, conventional FBA fails to account for the physical reality of intracellular macromolecular crowding, which imposes a finite volume occupancy limit on enzymes, ribosomes, and other cellular components. By assuming unlimited space for protein expression and ignoring the kinetic consequences of molecular crowding, traditional FBA models predict unrealistic enzyme concentrations and overestimate metabolic capacities, particularly under conditions of rapid growth where proteome allocation becomes a decisive constraint.  

- introduce optimization method  
To address these limitations, an optimization method is disclosed that integrates a molecular crowding constraint into the flux balance framework, thereby enabling the prediction of physiologically realistic metabolic states by explicitly limiting the total volume fraction occupied by metabolic enzymes and associated macromolecules. This modified approach, referred to as flux balance analysis with molecular crowding (FBAwMC), incorporates a crowding coefficient derived from experimentally determined macromolecular volumes and turnover rates, ensuring that predicted flux distributions remain within the bounds of cellular physical capacity.  

- describe application of optimization method  
The FBAwMC method is applied to model the sequential utilization of multiple carbon substrates in Escherichia coli cultures, revealing that the observed hierarchy of substrate consumption is not primarily driven by transcriptional regulation alone, but is fundamentally shaped by the need to maintain optimal intracellular macromolecular density during rapid growth. The method predicts that under nutrient-rich conditions, cells prioritize substrates that yield the highest growth rates per unit of enzyme volume, thereby minimizing proteome burden and maximizing metabolic efficiency.  

- calculate cell growth rates  
The FBAwMC model computes the maximum possible biomass production rate under given nutrient conditions by solving a linear optimization problem that maximizes the flux through a biomass reaction while respecting stoichiometric balance and macromolecular crowding constraints. The predicted growth rates align with experimentally measured values across a range of single and mixed substrate environments, including chemostat cultures operated at varying dilution rates.  

- calculate substrate utilization  
By applying the FBAwMC framework to mixed-substrate media, the model calculates the relative fluxes assigned to each substrate uptake and catabolic pathway, accurately predicting the temporal order and rate of consumption observed in experimental cultures. The model reveals that glucose is consumed preferentially not solely due to transcriptional repression, but because its metabolism yields the highest biomass yield per unit of enzyme volume, making it the most efficient substrate under crowding constraints.  

- calculate metabolic flux reorganization  
The method identifies how metabolic fluxes are reorganized as growth rate increases, shifting from a reliance on oxidative phosphorylation at low growth rates to a mixed mode involving concurrent glycolysis and respiration at higher rates. This reorganization is not an arbitrary regulatory response but a direct consequence of the need to maintain intracellular macromolecular density within a narrow physiological range by reducing the total enzyme burden required for ATP generation.  

- calculate maximum metabolic rate  
The optimization procedure determines the maximum metabolic rate attainable under the given crowding constraint, identifying the upper limit of cellular performance dictated by physical space rather than enzyme kinetics alone. This maximum rate is shown to be highly sensitive to the crowding coefficient, which reflects the average volume occupied by catalytic proteins per unit of enzymatic activity.  

- calculate optimal metabolite concentrations  
By coupling the FBAwMC model with kinetic rate equations and metabolite binding affinities, the method calculates the concentrations of intracellular intermediates that maximize flux through the network while avoiding metabolic bottlenecks. These predicted concentrations are consistent with experimental measurements and reveal that substrate availability alone does not determine flux—rather, the system converges on metabolite levels that balance enzyme saturation with volume efficiency.  

- calculate enzyme activities  
The model estimates the specific activities of individual enzymes required to achieve the predicted flux distribution, constrained by the total volume available for protein expression. These calculated enzyme activities are significantly lower than those predicted by standard FBA and are in closer agreement with in vitro enzymatic assays, affirming the biological plausibility of the crowding constraint.  

- describe computer-implemented methods  
The optimization method is implemented as a computer-executable algorithm that receives as input a reconstructed metabolic network, a set of nutrient availability constraints, and experimentally derived crowding coefficients. The algorithm then performs iterative linear programming to determine the flux distribution that maximizes biomass production while satisfying all physical and stoichiometric constraints.  

- describe apparatus for implementing methods  
The apparatus comprises a computational system including a processing unit, memory storage for holding metabolic network models and parameter databases, input interfaces for receiving experimental data, and output interfaces for displaying predicted flux distributions, growth rates, and enzyme activities. The system is operable in standalone or networked configurations and may be accessed remotely via secure data transmission protocols.  

- optimize biological activities  
The method enables the optimization of biological activities such as substrate uptake, ATP synthesis, and byproduct secretion by identifying the minimal set of enzyme activities required to achieve maximal growth under crowding constraints. This allows for the design of metabolic states that are not only efficient but also robust to environmental fluctuations.  

- calculate optimal cell culture parameters  
The method calculates optimal parameters for cell culture, including nutrient concentrations, dilution rates, pH, oxygen tension, and temperature, that maximize growth rate while maintaining intracellular macromolecular density within a target physiological range. These parameters are derived from in silico simulations and validated against experimental outcomes.  

- initiate optimal cell culture parameters  
Upon determination of optimal culture parameters, the system automatically initiates or adjusts external culture conditions—such as media feed rates, gas flow, and temperature control—to align the physical environment with the predicted optimal state, thereby enabling closed-loop cultivation systems that self-regulate toward maximal performance.  

- calculate order of substrate usage  
The method predicts the precise temporal order in which substrates will be consumed in mixed-media cultures based on their respective enzyme volume efficiencies, independent of transcriptional regulatory networks. This ordering is shown to be conserved across strains and growth conditions when crowding is the dominant constraint.  

- control order of substrate usage  
By manipulating the expression levels of key transporter proteins or altering the crowding coefficient through genetic or chemical perturbations, the method enables active control over substrate utilization order, allowing for the engineering of metabolic behaviors that diverge from natural regulatory patterns.  

- calculate maximum metabolic rate  
The optimization procedure identifies the maximum metabolic rate achievable under the physical limits imposed by macromolecular crowding, providing a quantitative upper bound for cellular performance that cannot be exceeded even with unlimited enzyme expression.  

- calculate optimal metabolite concentrations  
The model computes the set of intracellular metabolite concentrations that minimize enzyme occupancy while maintaining flux through the network, revealing a universal principle: cells do not simply maximize reaction rates—they maximize reaction rates per unit of volume occupied.  

- calculate enzyme activities  
The method determines the specific enzyme activities required to sustain the predicted flux distribution under crowding constraints, producing estimates that are consistent with in vivo enzymatic measurements and significantly different from those derived from unconstrained FBA.  

- achieve optimal function of biochemical reaction network  
The disclosed method enables the achievement of optimal function in a biochemical reaction network by aligning metabolic fluxes with the physical capacity of the cell, thereby ensuring that cellular resources are allocated in a manner that maximizes fitness under the constraints of limited intracellular space.  

- calculate optimal properties of biochemical reaction network  
The method calculates a suite of optimal properties—including flux distribution, enzyme activity profile, metabolite concentration profile, and macromolecular volume fraction—that collectively define the most efficient metabolic state achievable under given environmental and physical conditions.  

- alter biochemical reactions  
The method permits the computational alteration of the biochemical reaction network—such as the addition, deletion, or modification of reactions—to evaluate the impact of genetic or metabolic engineering on network performance under crowding constraints.  

- repeat calculation of optimal properties  
Following each alteration to the network, the optimization procedure is repeated to recalculate the optimal properties, enabling iterative design cycles that guide the development of engineered strains with enhanced productivity, substrate utilization, or metabolic robustness.  

- culture cells under optimal conditions  
Cells are cultivated under the conditions predicted by the model to achieve maximal growth and metabolic efficiency, resulting in cultures that exhibit higher biomass yields, reduced byproduct secretion, and improved stability compared to those maintained under conventional conditions.  

- construct genetic makeup of cells  
The genetic makeup of cells is engineered to match the predicted optimal enzyme activity profile, including overexpression of high-efficiency transporters, deletion of competing pathways, and modulation of regulatory elements to suppress non-optimal substrate utilization.  

- place cells in culture  
Engineered cells are placed in controlled culture environments where nutrient concentrations, oxygen levels, pH, and temperature are maintained in accordance with the model-predicted optimal parameters.  

- cultivate cells to achieve optimal function  
Cells are cultivated over extended periods to allow them to adapt to the imposed conditions, during which time their metabolic networks evolve toward the predicted optimal state, resulting in phenotypes that match or exceed the in silico predictions.  

- describe computer-readable medium  
A computer-readable medium is provided that stores executable instructions for implementing the FBAwMC optimization method, including algorithms for network reconstruction, constraint definition, linear programming, and output formatting. The medium may be embodied in non-transitory storage devices such as hard drives, solid-state drives, optical discs, or cloud-based memory systems.

## DETAILED DESCRIPTION

- define terms used in patent  
For the purposes of this disclosure, “flux” refers to the net rate of conversion of substrates to products through a biochemical reaction, measured in mmol per gram of dry cell weight per hour. “Macromolecular crowding” refers to the phenomenon in which the intracellular environment is densely packed with proteins, nucleic acids, and other macromolecules, resulting in a reduction in the available volume for diffusion and molecular interaction. “Crowding coefficient” denotes a dimensionless parameter that quantifies the effective volume occupied per unit of enzymatic activity, derived from the molecular volume and turnover rate of individual enzymes. “Optimal function” refers to the state of a biochemical reaction network in which the rate of biomass production is maximized subject to stoichiometric, thermodynamic, and macromolecular crowding constraints. “Substrate hierarchy” refers to the ordered sequence in which multiple carbon sources are consumed by a cell in mixed-media culture, determined by the efficiency of ATP and biomass production per unit of enzyme volume. “FBAwMC” refers to flux balance analysis with molecular crowding, the computational method disclosed herein that integrates crowding constraints into metabolic modeling.  

- introduce flux balance calculations for cell cultures  
Flux balance calculations for cell cultures are based on the assumption that, under steady-state conditions, the intracellular concentrations of metabolites remain constant over time, such that the rate of production of each metabolite equals its rate of consumption. This leads to a system of linear equations defined by the stoichiometric matrix, which encodes the molar relationships between substrates and products in all metabolic reactions. Solving this system under an objective function—typically maximization of biomass production—yields a predicted flux distribution that represents the cell’s metabolic state.  

- describe use of cytoplasmic molecular crowding in flux balance calculations  
In conventional flux balance models, no upper limit is imposed on the concentration of metabolic enzymes, leading to predictions of unrealistically high enzyme levels. In the disclosed method, a constraint is introduced that the total volume fraction occupied by all metabolic enzymes, ribosomes, and associated macromolecules must not exceed a physiologically observed maximum of approximately 0.4 to 0.5. This constraint is expressed as a linear inequality involving the product of enzyme flux, molecular volume, and crowding coefficient for each reaction, ensuring that predicted enzyme concentrations are physically realizable.  

- describe use of reaction kinetics parameters in flux balance calculations  
Reaction kinetics parameters, including Michaelis-Menten constants (Km), catalytic turnover numbers (kcat), and substrate affinities, are incorporated into the model to refine the relationship between enzyme activity and flux. These parameters are used to calculate the crowding coefficient for each enzyme, defined as the ratio of the enzyme’s molecular volume to its catalytic efficiency (kcat/Km), thereby weighting reactions according to their volumetric cost.  

- outline applications of flux balance analysis  
Applications of flux balance analysis include the prediction of metabolic phenotypes under nutrient limitation, the identification of essential genes and metabolic bottlenecks, the design of metabolic engineering strategies, the interpretation of transcriptomic and proteomic data, and the optimization of industrial fermentation processes. The disclosed FBAwMC method enhances the predictive accuracy of all these applications by incorporating the fundamental constraint of intracellular space.  

- introduce flux balance model of cellular metabolism  
The flux balance model of cellular metabolism comprises a genome-scale reconstruction of metabolic reactions, including central carbon metabolism, amino acid biosynthesis, nucleotide synthesis, lipid metabolism, and energy generation pathways. Each reaction is represented as a stoichiometric equation, and the entire network is encoded as a sparse matrix where rows correspond to metabolites and columns to reactions.  

- define stoichiometric matrix  
The stoichiometric matrix is a mathematical representation of the metabolic network in which each row corresponds to a specific metabolite and each column to a metabolic reaction. The entries in the matrix indicate the stoichiometric coefficients of metabolites in each reaction, with negative values denoting consumption and positive values denoting production.  

- describe steady state assumption  
The steady state assumption holds that, under constant environmental conditions, the intracellular concentrations of metabolites do not change over time, such that the net rate of synthesis of each metabolite equals its net rate of consumption. This assumption enables the reduction of the dynamic metabolic system to a set of linear equations that can be solved using linear programming.  

- introduce equation for flux balance analysis  
The core equation of flux balance analysis is expressed as Sv = 0, where S is the stoichiometric matrix, v is the vector of fluxes through each reaction, and the zero vector represents the condition of metabolite mass balance. An objective function, typically maximizing the flux through a biomass reaction, is then optimized subject to additional constraints.  

- describe enzyme concentration constraint  
In the disclosed FBAwMC method, an enzyme concentration constraint is introduced such that the sum over all reactions of the product of flux, enzyme molecular volume, and crowding coefficient does not exceed a predefined maximum volume fraction, typically 0.45. This constraint replaces the unbounded enzyme expression assumption of traditional FBA.  

- introduce crowding coefficient  
The crowding coefficient for each enzyme is defined as the ratio of its molecular volume (in nm³) to its catalytic efficiency (kcat/Km, in s⁻¹M⁻¹), yielding a value in units of volume per turnover. This coefficient quantifies the space required per unit of metabolic activity and is derived from experimentally measured enzyme structures and kinetic parameters.  

- outline method for optimizing biological activities  
The method for optimizing biological activities involves the iterative application of linear programming to identify the flux distribution that maximizes biomass production while satisfying stoichiometric, thermodynamic, and crowding constraints. The optimization is performed using a simplex or interior-point algorithm on a computing system, with constraints dynamically adjusted based on experimental measurements of growth rate, enzyme abundance, or metabolite levels.  

- describe calculation of optimal cell culture parameter  
The optimal cell culture parameter is calculated by simulating the metabolic behavior of the organism under varying environmental conditions—such as nutrient concentration, oxygen availability, and pH—and identifying the condition that yields the highest biomass production rate under the crowding constraint. This parameter set includes media composition, dilution rate, temperature, and aeration level.  

- describe initiation and maintenance of optimal cell culture parameter  
Once the optimal parameters are determined, they are implemented in a closed-loop bioreactor system equipped with sensors and actuators that continuously monitor culture density, metabolite concentrations, and oxygen consumption. The system automatically adjusts feed rates, gas flow, and temperature to maintain the culture within the predicted optimal state.  

- provide example of implementation of method  
In an example implementation, the FBAwMC model was applied to Escherichia coli MG1655 grown in a mixed-substrate medium containing glucose, glycerol, galactose, lactate, and maltose. The model predicted that glucose would be consumed first due to its high biomass yield per unit enzyme volume, followed by maltose, galactose, glycerol, and finally lactate. Experimental measurements confirmed this hierarchy, with glucose consumption initiating before the others and lactate utilization being suppressed until all other substrates were depleted.  

- describe use of computer for implementation of method  
The method is implemented using a general-purpose computer system comprising a central processing unit, random access memory, non-volatile storage for metabolic network databases, and input/output interfaces for data acquisition and result visualization. The software executes the optimization algorithm, retrieves kinetic parameters from external databases, and outputs predicted fluxes, growth rates, and enzyme activities in machine-readable and human-readable formats.  

- outline steps for determining optimal functions of biochemical reaction network  
The steps for determining optimal functions include: (1) reconstructing the metabolic network from annotated genomic and biochemical data; (2) defining the objective function as biomass production; (3) incorporating stoichiometric and thermodynamic constraints; (4) calculating crowding coefficients for all enzymatic reactions; (5) applying the macromolecular crowding constraint to limit total enzyme volume; (6) solving the linear optimization problem; and (7) validating predictions against experimental data.  

- describe representation of biochemical reactions in computer  
Biochemical reactions are represented in the computer as digital entries in a structured database, with each reaction assigned a unique identifier, stoichiometric coefficients for all reactants and products, associated enzyme identifiers, kinetic parameters, and subcellular compartmentalization data. These entries are compiled into a sparse matrix for computational processing.  

- describe use of optimization methods  
Optimization methods, including linear programming and quadratic programming, are employed to solve the constrained optimization problem. When non-linear relationships are introduced—for instance, through enzyme saturation kinetics or allosteric regulation—non-linear optimization techniques such as sequential quadratic programming or simulated annealing are applied.  

- describe alteration of list of reactions and re-computation of optimal properties  
The list of reactions constituting the metabolic network may be altered by adding, deleting, or modifying reactions—such as introducing heterologous pathways, deleting competing routes, or altering cofactor specificity—followed by re-computation of the optimal flux distribution, enzyme activity profile, and growth rate under the revised network structure.  

- describe repetition of altering step until desired performance is met  
The process of altering the reaction list and re-computing optimal properties is repeated iteratively until the predicted performance—such as increased biomass yield, reduced acetate secretion, or enhanced substrate utilization—meets or exceeds a pre-defined threshold.  

- describe culturing of living cell and optimization of culture conditions  
Living cells are cultured in bioreactors under conditions specified by the model-predicted optimal parameters. The culture is monitored for growth rate, substrate depletion, and byproduct formation, and adjustments are made in real time to maintain alignment between predicted and observed behavior.  

- describe construction of genetic makeup of cell  
The genetic makeup of the cell is constructed through targeted genetic modifications, including gene knockouts, promoter replacements, ribosome binding site engineering, and heterologous gene expression, to align the cell’s metabolic enzyme profile with the optimal enzyme activity profile predicted by the FBAwMC model.  

- describe placement of cell in culture and cultivation  
The genetically engineered cells are placed into a controlled culture environment and cultivated under optimal nutrient, oxygen, and temperature conditions. The cultivation period is extended to allow for adaptive evolution toward the predicted optimal phenotype.  

- describe evolution of cell to desired performance  
Over extended cultivation periods, the cells undergo adaptive evolution through spontaneous mutations and selection pressures, resulting in phenotypes that converge toward the predicted optimal performance. This evolutionary trajectory is tracked through periodic sampling and genomic sequencing.  

- define biochemical reaction network  
A biochemical reaction network is a comprehensive set of enzymatically catalyzed chemical transformations occurring within a cell, encompassing all metabolic pathways from nutrient uptake to biomass synthesis, including energy generation, precursor biosynthesis, and redox balance.  

- describe types of biochemical reaction networks  
Types of biochemical reaction networks include genome-scale networks encompassing all known metabolic reactions, subsystem networks focused on specific pathways such as glycolysis or TCA cycle, and synthetic networks constructed de novo from engineered parts. The disclosed method is applicable to all such network types.  

- describe implementation of methods using whole biochemical reaction network  
The method is implemented using a whole biochemical reaction network reconstructed from annotated genomic sequences, biochemical databases, and physiological data, ensuring that the model captures the full metabolic capacity of the organism.  

- describe reconstruction of biochemical reaction network  
Reconstruction of the biochemical reaction network involves curating reactions from public databases such as KEGG, MetaCyc, and BioCyc, integrating gene-protein-reaction associations from genomic annotations, and refining stoichiometry and thermodynamic feasibility using experimental data.  

- describe use of annotated genome sequences and biochemical and physiological data  
Annotated genome sequences provide gene identifiers and functional annotations, biochemical data supply reaction stoichiometries and kinetic parameters, and physiological data—such as protein abundance, metabolite concentrations, and growth rates—serve to constrain and validate the model.  

- describe analysis of reconstructed network  
The reconstructed network is analyzed to identify essential reactions, metabolic bottlenecks, and redundancy, and is then subjected to constraint-based modeling to predict its behavior under defined environmental conditions.  

- describe determination of optimal properties of biochemical reaction network  
Optimal properties are determined by solving the FBAwMC optimization problem, yielding the flux distribution, enzyme activity profile, metabolite concentration profile, and macromolecular volume fraction that maximize biomass production under crowding constraints.  

- describe use of optimization methods  
Optimization methods such as linear programming, quadratic programming, and simulated annealing are employed to solve the constrained optimization problem, with the choice of method depending on the linearity of constraints and the nature of the objective function.  

- describe linear and non-linear optimization with linear constraints  
Linear optimization is used when all constraints and the objective function are linear, as in the case of stoichiometric and crowding constraints. Non-linear optimization is applied when kinetic rate laws introduce non-linearities, such as Michaelis-Menten dependence on substrate concentration, while still maintaining linear crowding constraints.  

- describe use of simulated annealing  
Simulated annealing is employed to escape local optima in non-linear optimization problems, particularly when the objective function exhibits ruggedness due to enzyme cooperativity or allosteric regulation.  

- describe reconstruction of metabolic network  
Reconstruction of the metabolic network begins with the extraction of gene annotations from sequenced genomes, followed by the assignment of biochemical reactions based on enzyme function, pathway membership, and homology. Reactions are then consolidated into a unified network, with gaps filled using homology-based inference.  

- describe use of flux balance analysis  
Flux balance analysis is used to compute steady-state flux distributions under nutrient constraints, providing a baseline prediction of metabolic behavior that is subsequently refined by the inclusion of crowding constraints.  

- describe assessment of metabolic capabilities of reconstructed metabolic network  
The metabolic capabilities of the reconstructed network are assessed by simulating growth on single and multiple substrates, identifying the range of nutrients that support growth, and predicting byproduct secretion profiles.  

- describe use of experimentally determined strain-specific parameters  
Strain-specific parameters—including enzyme molecular volumes, turnover rates, and intracellular protein concentrations—are derived from experimental measurements in the target organism and incorporated into the model to ensure physiological relevance.  

- describe calculation of flux distribution through reconstructed metabolic network  
The flux distribution is calculated by solving the stoichiometric matrix equation under the objective of maximum biomass production, subject to constraints on nutrient uptake, reaction reversibility, and macromolecular crowding.  

- describe definition of factors leading to closed solution space  
The solution space is closed by imposing upper and lower bounds on fluxes, enforcing thermodynamic feasibility, and constraining the total volume occupied by enzymes. These constraints ensure that the optimization problem has a unique, physiologically realizable solution.  

- describe optimization procedure  
The optimization procedure involves initializing the flux vector, applying constraints, selecting an optimization algorithm, solving for the optimal flux distribution, validating the solution against experimental data, and iteratively refining the model until convergence is achieved.  

- describe comparison of calculated behavior to experimental data  
The calculated flux distribution, growth rate, and metabolite consumption profile are compared to experimental measurements obtained from chemostat cultures, metabolomics, and enzyme assays. Discrepancies are used to refine model parameters or identify missing reactions.  

- describe addition of constraints relating to cytoplasmic molecular crowding and/or reaction kinetics  
Additional constraints are introduced to account for the physical volume occupied by enzymes and the kinetic efficiency of reactions, ensuring that predicted enzyme concentrations remain below the maximum allowable crowding threshold.  

- describe prediction of optimal uses of biochemical reaction network  
The method predicts the optimal use of the biochemical reaction network for specific applications, such as high-yield production of biofuels, pharmaceuticals, or amino acids, by redefining the objective function to maximize the flux through the target compound pathway.  

- describe limitations of natural organism with intact network  
Natural organisms with intact metabolic networks are constrained by evolutionary trade-offs that may favor robustness over efficiency, leading to suboptimal flux distributions under industrial conditions.  

- describe limitations of natural organism without growth competition and selection  
In the absence of growth competition and selection pressure, natural organisms may accumulate non-functional or inefficient reactions, resulting in metabolic redundancy and reduced performance under controlled cultivation.  

- describe need for design of biochemical reaction network  
There is a need for the rational design of biochemical reaction networks that eliminate non-essential reactions, enhance pathway efficiency, and align enzyme expression with physical constraints, thereby enabling superior performance in industrial biotechnology.  

- describe use of methods to achieve optimal performance  
The disclosed methods are used to achieve optimal performance by computationally identifying the most efficient metabolic state and then engineering the cell to match that state through genetic manipulation and controlled cultivation.  

- introduce wild type network perturbation  
Wild-type metabolic networks are perturbed by deleting genes, overexpressing enzymes, or introducing novel pathways to test the robustness of the predicted optimal state and to identify potential targets for strain improvement.  

- describe in silico methods for resolving optimality issues  
In silico methods, including the FBAwMC framework, are used to resolve optimality issues by predicting the impact of genetic perturbations on metabolic efficiency before experimental implementation.  

- explain culturing methods for resolving growth competition and selection issues  
Culturing methods involving extended adaptive evolution under controlled selection pressure are used to allow cells to evolve toward the predicted optimal phenotype, resolving discrepancies between in silico predictions and in vivo behavior.  

- motivate altering cellular parameters for desired performance  
Altering cellular parameters—such as enzyme expression levels, cofactor availability, or membrane transport capacity—is motivated by the need to tailor metabolic networks for specific industrial applications, including the production of high-value compounds under nutrient-limited conditions.  

- describe iterative design procedure for optimizing performance  
The iterative design procedure involves generating a computational model, predicting an optimal metabolic state, engineering the cell to approximate that state, cultivating the engineered strain, measuring its performance, comparing results to predictions, and refining the model for subsequent iterations.  

- define desired performance as qualitative characteristic or quantitative value  
Desired performance is defined as a qualitative characteristic, such as “reduced acetate secretion,” or a quantitative value, such as “biomass yield of 0.5 g/g glucose,” and is used as the objective function in the optimization procedure.  

- provide examples of desired performances  
Examples of desired performances include maximizing ethanol production, minimizing glycerol byproduct formation, enhancing tolerance to high osmolarity, increasing specific growth rate under low oxygen, and improving nutrient utilization efficiency in mixed-substrate environments.  

- describe optimization method using computer system  
The optimization method is implemented using a computer system comprising a processor, memory, storage for metabolic models, and software modules that execute the flux balance analysis, apply crowding constraints, and output optimized metabolic states.  

- outline computer system components  
The computer system includes a central processing unit, volatile and non-volatile memory, input devices for uploading genomic and metabolic data, output devices for displaying flux maps and growth predictions, and communication interfaces for remote access and data exchange.  

- describe database information regarding biochemical reaction networks  
The database contains curated biochemical reaction networks for multiple organisms, including stoichiometric coefficients, enzyme annotations, kinetic parameters, subcellular localization, and thermodynamic data, sourced from public repositories and peer-reviewed literature.  

- explain database information regarding biomolecular sequences  
Database information regarding biomolecular sequences includes annotated genomes, protein sequences, gene expression profiles, and operon structures, linked to their corresponding metabolic functions.  

- describe database information regarding genomic sequences  
Database information regarding genomic sequences includes complete, annotated genome assemblies with gene identifiers, functional annotations, regulatory elements, and orthologous groupings across species.  

- outline database annotations and sequence information  
Database annotations include Gene Ontology terms, MetaCyc pathway assignments, EC numbers, enzyme commission classifications, and literature-supported functional evidence, all linked to sequence identifiers for cross-referencing.  

- explain identifying biochemical genotype of an organism  
The biochemical genotype of an organism is identified by mapping its annotated genome to a metabolic network, assigning reactions to genes based on enzyme function, and resolving ambiguities through homology and experimental validation.  

- describe database types and external databases  
Database types include metabolic reconstructions, kinetic parameter repositories, gene expression atlases, and protein structure libraries. External databases include KEGG, BRENDA, UniProt, EcoCyc, and MetaCyc, from which data are imported and integrated.  

- outline user interface for receiving selections and optimal performance  
The user interface allows users to select the organism, define the objective function, specify nutrient conditions, upload custom constraints, and view predicted outcomes in graphical and tabular formats, including flux maps, enzyme activity heatmaps, and growth rate curves.  

- describe computer program product and processing unit  
The computer program product comprises a non-transitory computer-readable medium storing executable instructions for performing the FBAwMC optimization, and is executed by a processing unit that interprets the metabolic network, applies constraints, and computes the optimal flux distribution.  

- outline modules and processes for implementing computer program  
Modules include a network parser, constraint builder, optimizer engine, validator module, and output generator. Processes include data ingestion, model construction, optimization execution, result validation, and visualization.  

- describe interacting with database and altering biochemical reaction network  
The computer system interacts with external databases to retrieve kinetic parameters and gene annotations, and allows users to manually alter the network by adding, deleting, or modifying reactions through a graphical interface.  

- explain comparing biochemical reaction networks to identify differences  
Biochemical reaction networks from different strains or conditions are compared algorithmically to identify differences in reaction content, flux distribution, and enzyme activity profiles, enabling the identification of key regulatory or structural changes underlying phenotypic variation.  

- describe receiving data from cell culture and controlling external devices  
The system receives real-time data from sensors monitoring culture density, substrate concentration, pH, dissolved oxygen, and temperature, and uses this data to adjust external devices such as peristaltic pumps, gas mixers, and heating elements to maintain optimal conditions.  

- outline generating computer-readable program code  
The system generates computer-readable program code in a standard format such as MATLAB, Python, or SBML, which can be executed on other platforms or integrated into automated bioreactor control systems.  

- describe adaptive evolution of cultured strain to achieve desired performance  
Cultured strains are subjected to prolonged cultivation under selective pressure to allow adaptive evolution, during which spontaneous mutations accumulate and are selected based on improved performance relative to the predicted optimal state.  

- explain using virtually any cell with the methods  
The methods are applicable to virtually any cell type, including bacteria, yeast, mammalian cells, and engineered synthetic organisms, provided that a metabolic network reconstruction and strain-specific crowding parameters are available.  

- describe biochemical reaction network characterization  
Biochemical reaction network characterization involves determining the set of reactions present, their stoichiometry, reversibility, enzyme associations, and thermodynamic feasibility, forming the foundation for constraint-based modeling.  

- outline genome sequencing and gene identification  
Genome sequencing provides the complete DNA sequence of the organism, and gene identification assigns open reading frames to putative proteins, which are then annotated for biochemical function using homology and experimental evidence.  

- describe genetic makeup of a cell  
The genetic makeup of a cell comprises the complete set of genes, regulatory elements, and non-coding sequences that determine its metabolic capabilities, including the identity, copy number, and expression level of enzymes involved in substrate utilization and biomass synthesis.  

- construct biochemical reactions to meet desired performance  
Biochemical reactions are constructed or modified to meet desired performance criteria by introducing novel enzymatic steps, altering cofactor specificity, or removing competing pathways, guided by FBAwMC predictions of optimal flux distribution.  

- add or subtract reactions from list using genetic manipulations  
Reactions are added or subtracted from the metabolic network list through genetic manipulations such as CRISPR-Cas9-mediated gene knockout, plasmid-based overexpression, or synthetic gene circuit integration.  

- alter expression of regulatory components  
Expression of regulatory components—including transcription factors, small RNAs, and riboswitches—is altered to modulate the activity of substrate uptake systems and metabolic pathways in accordance with predicted optimal enzyme activity profiles.  

- place cell in culture under specified environment  
The cell is placed in a controlled culture environment where nutrient composition, oxygen tension, pH, temperature, and agitation rate are set to the values predicted by the FBAwMC model to achieve maximal growth and metabolic efficiency.  

- determine optimal cultural parameters using optimization procedure  
Optimal cultural parameters—including carbon source concentration, nitrogen source, trace elements, and dilution rate—are determined by the optimization procedure as those which yield the highest biomass production rate under crowding constraints.  

- monitor culture conditions and adjust as necessary  
Culture conditions are continuously monitored using online sensors, and adjustments are made automatically by the control system to maintain alignment with predicted optimal parameters.  

- use computer program to determine optimal conditions  
A computer program executes the FBAwMC algorithm to determine the optimal conditions for cell growth and product formation, outputting a set of actionable parameters for culture control.  

- configure computer system to automatically bring culture to optimal parameters  
The computer system is configured to interface with bioreactor hardware, receiving sensor inputs and transmitting control signals to adjust media feed rates, gas flow, and temperature, thereby autonomously bringing the culture to its optimal metabolic state.  

- use continuous culture and computerized system to monitor culture parameters  
Continuous culture systems coupled with computerized monitoring allow for the long-term maintenance of steady-state conditions, enabling the observation of metabolic adaptations and validation of model predictions over extended periods.  

- control flow of new culture media and addition of culture constituents  
The flow of new culture media and the addition of inducers, inhibitors, or nutrients are controlled by the computer system in response to real-time measurements of substrate depletion, byproduct accumulation, and growth rate.  

- configure and program computer system  
The computer system is configured with software modules that implement the FBAwMC algorithm, integrate with laboratory information management systems, and enable remote access for monitoring and control.  

- allow cells to adapt to culture conditions through adaptive evolution  
Cells are allowed to adapt to culture conditions through prolonged cultivation under selective pressure, during which spontaneous mutations arise and are selected for improved fitness relative to the predicted optimal state.  

- culture cells for sufficient period to allow evolution towards desired performance  
Cells are cultured for a sufficient period—ranging from hundreds to thousands of generations—to allow the accumulation of beneficial mutations that drive the population toward the desired performance phenotype.  

- use extended cultivation to optimize metabolic network  
Extended cultivation enables the optimization of the metabolic network through natural selection, resulting in strains with improved enzyme expression, reduced metabolic burden, and enhanced pathway efficiency.  

- accelerate evolutionary process using chemical mutagens and/or radiation  
The evolutionary process is accelerated by exposing cells to chemical mutagens such as ethyl methanesulfonate or physical mutagens such as UV radiation, increasing the mutation rate and facilitating faster convergence to the desired phenotype.  

- genetically alter living cell to contain biochemical reactants  
Living cells are genetically altered to contain novel biochemical reactants—such as synthetic cofactors, non-natural amino acids, or engineered enzymes—that enable new metabolic capabilities not present in the wild-type organism.  

- culture cells under specified environmental conditions  
Cells are cultured under precisely specified environmental conditions—including defined carbon and nitrogen sources, controlled oxygen levels, and regulated pH—to ensure reproducibility and alignment with model predictions.  

- monitor growth and metabolic behavior during adaptive evolutionary process  
Growth rate, substrate consumption, byproduct secretion, and transcriptomic profiles are monitored throughout the adaptive evolutionary process to track phenotypic progression and validate convergence toward the predicted optimal state.  

- measure oxygen uptake rate, substrate uptake rate, and growth rate  
Oxygen uptake rate, substrate uptake rate, and growth rate are measured using respirometry, HPLC, and optical density monitoring, respectively, to provide quantitative benchmarks for model validation.  

- plot data points on phenotype phase plane  
Data points representing growth rate versus substrate uptake rate are plotted on a phenotype phase plane to visualize the metabolic trade-offs and identify the region of optimal performance predicted by the model.  

- continue evolutionary process until optimal performance is achieved  
The evolutionary process is continued until the measured performance parameters—such as biomass yield, product titer, or growth rate—match or exceed the values predicted by the FBAwMC model.  

- monitor byproduct secretion using HPLC or other analytical methods  
Byproduct secretion—including acetate, lactate, and ethanol—is monitored using high-performance liquid chromatography (HPLC), gas chromatography-mass spectrometry (GC-MS), or nuclear magnetic resonance (NMR) to assess metabolic efficiency and detect shifts in pathway allocation.  

- determine correlation of dry weight vs optical density for evolved strain  
The correlation between dry cell weight and optical density is determined for the evolved strain to calibrate growth measurements and ensure accurate quantification of biomass production.  

- inspect cultures for signs of contamination or co-evolution with mutant subpopulation  
Cultures are periodically inspected for signs of contamination, phage infection, or the emergence of co-evolving mutant subpopulations that may compromise the integrity of the selected phenotype.  

- log optical density, time of inoculation, inoculum volume, growth rate, and signs of contamination  
All culture parameters—including optical density, time of inoculation, inoculum volume, measured growth rate, and observed contamination—are logged in a digital record for traceability and analysis.  

- freeze samples for further use  
Samples are frozen at −80°C in glycerol stocks for long-term storage and future genomic, transcriptomic, or phenotypic analysis.  

- provide computer-implemented method for achieving optimal function of biochemical reaction network  
A computer-implemented method is provided for achieving optimal function of a biochemical reaction network by performing flux balance analysis with molecular crowding, wherein the method comprises receiving a stoichiometric matrix, defining a biomass production objective, calculating crowding coefficients for each enzymatic reaction, imposing a volume constraint on total enzyme occupancy, and computing the flux distribution that maximizes biomass production under the crowding constraint.  

- calculate optimal cell culture parameters using flux balance analysis  
Optimal cell culture parameters are calculated using flux balance analysis by simulating growth under varying nutrient conditions and selecting the set of parameters that yields the highest biomass production rate under the imposed crowding constraint.  

- apply computational optimization method to biochemical reactions  
A computational optimization method is applied to biochemical reactions to determine the flux distribution that maximizes a defined objective function while respecting stoichiometric, thermodynamic, and macromolecular crowding constraints.  

- alter elements of biochemical reactions and re-compute optimal property  
Elements of biochemical reactions—including stoichiometry, reversibility, or enzyme association—are altered, and the optimal property is re-computed to evaluate the effect of the alteration on network performance.  

- repeat optimization method until optimal function is reached  
The optimization method is repeated iteratively, with successive modifications to the reaction network, until the predicted optimal function—such as maximum growth rate or minimum byproduct secretion—is achieved.  

- initiate or maintain optimal cell culture parameter in cell culture  
The optimal cell culture parameter is initiated by configuring the culture environment to match the predicted values, and is maintained through continuous feedback control using sensor data and automated adjustment of culture conditions.  

- calculate maximum metabolic rate, optimal metabolite concentration, and enzyme activity  
The method calculates the maximum metabolic rate attainable under crowding constraints, the optimal concentration of intracellular metabolites that support maximal flux, and the specific activity of each enzyme required to sustain the predicted flux distribution.  

- apply computational optimization method to kinetic model of metabolic pathway  
A computational optimization method is applied to a kinetic model of a metabolic pathway to simultaneously determine enzyme activities, metabolite concentrations, and fluxes that satisfy both mass balance and crowding constraints.  

- construct genetic makeup of cell to contain biochemical reactions  
The genetic makeup of a cell is constructed by introducing or deleting genes to encode the enzymes required to carry out the biochemical reactions predicted by the FBAwMC model to achieve optimal function.  

- place cell in culture under specified environment to obtain population of cells  
The genetically modified cell is placed in a culture under specified environmental conditions to obtain a population of cells that express the engineered metabolic network.  

- cultivate cells for sufficient period to allow evolution towards desired optimal function  
Cells are cultivated for a sufficient period to allow spontaneous mutation and selection to drive the population toward the desired optimal function, as defined by the computational model.  

- account for ribosome density in cells as measure of ribosomal-, enzyme associated-, and non-metabolic proteins  
Ribosome density is accounted for in the model as a component of macromolecular crowding, representing the volume occupied by ribosomal proteins, enzyme-associated chaperones, and other non-metabolic proteins that contribute to total intracellular volume.  

- account for mitochondria as subcellular compartment in cells  
Mitochondria are modeled as a distinct subcellular compartment with its own volume fraction, enzyme complement, and flux constraints, enabling the prediction of compartment-specific metabolic behavior and cross-compartmental resource allocation.  

- provide computer-readable medium having stored instructions for implementing computer model  
A computer-readable medium is provided having stored thereon instructions that, when executed by a processor, cause the processor to perform the steps of the FBAwMC method, including network reconstruction, constraint application, optimization, and output generation.  

- provide device comprising computer-readable medium and processor for executing instructions  
A device is provided comprising a computer-readable medium storing the instructions and a processor configured to execute the instructions, wherein the device is operable to receive metabolic network data, apply crowding constraints, compute optimal fluxes, and output predicted metabolic states.  

- provide additional components for implementation of instructions  
Additional components include a memory unit for storing metabolic models, an input interface for uploading experimental data, an output interface for displaying results, and a communication subsystem for remote access and data exchange.  

- provide sensors, culture vessels, heating/cooling elements, and reservoirs for storing cell culture medium  
The device further comprises sensors for measuring optical density, pH, dissolved oxygen, and substrate concentration; culture vessels for holding the cell culture; heating and cooling elements for temperature regulation; and reservoirs for storing cell culture medium and supplementary nutrients.  

- provide mechanisms for dispensing cell culture medium and taking/analyzing samples  
Mechanisms for dispensing cell culture medium and taking samples are provided, including automated peristaltic pumps, sterile sampling loops, and inline analytical devices such as spectrophotometers and chromatographs.  

- provide display, analytical devices, and communication subsystems  
The device includes a display for visualizing metabolic flux maps, analytical devices for metabolite quantification, and communication subsystems for transmitting data to remote servers or cloud-based analytics platforms.  

- provide remote access to reaction parameters  
Remote access is provided to reaction parameters, including flux values, enzyme activities, and crowding coefficients, via secure web interfaces, enabling real-time monitoring and control by authorized users from any location.

## EXAMPLES

- list abbreviations  
CCR: Carbon catabolite repression  
FBAwMC: Flux balance analysis with molecular crowding  
MC: Macromolecular crowding  
OxPhos: Oxidative phosphorylation  
PTS: Phosphotransferase system  
GFP: Green fluorescent protein  
OD: Optical density  
HPLC: High-performance liquid chromatography  
GC-MS: Gas chromatography–mass spectrometry  
NMR: Nuclear magnetic resonance  
Km: Michaelis-Menten constant  
kcat: Catalytic turnover number  
FtsZ: Cell division protein Z  
MBP: Maltose-binding protein  
MIDA: Mass isotopomer distribution analysis  
CRP: Catabolite repressor protein  
TCA: Tricarboxylic acid  
FBA: Flux balance analysis  

### Example 1

- study impact of limited solvent capacity on E. coli cell metabolism  
The impact of limited solvent capacity on Escherichia coli metabolism was studied by applying the FBAwMC method to a genome-scale metabolic model under varying nutrient conditions. The model predicted that as growth rate increases, the total enzyme volume approaches the maximum allowable crowding limit, forcing a reorganization of metabolic fluxes to reduce per-unit biomass enzyme burden.  

- demonstrate relevance of constraint for fast growing cells  
The constraint was found to be highly relevant for fast-growing cells, where conventional FBA predicted enzyme concentrations exceeding 50% of total cellular protein, whereas FBAwMC constrained these values to physiologically realistic levels of 35–40%, consistent with experimental proteomic measurements.  

- predict metabolic switch between low and high nutrient abundance  
The model predicted a metabolic switch from oxidative phosphorylation at low nutrient abundance to a mixed mode involving glycolysis and respiration at high nutrient abundance, driven by the need to reduce enzyme volume while maintaining ATP flux.  

- carry out flux measurements of several reactions  
Flux measurements were carried out for key reactions including glucose uptake, pyruvate kinase, acetate kinase, and citrate synthase using isotopic labeling and mass spectrometry, confirming the model’s predicted redistribution of fluxes.  

- observe partial agreement with model predictions  
Experimental fluxes showed partial agreement with model predictions, with deviations attributed to regulatory mechanisms not yet modeled, such as allosteric inhibition and post-translational modification.  

- perform gene expression and enzyme activity measurements  
Gene expression and enzyme activity measurements were performed for enzymes involved in glucose, maltose, and glycerol metabolism, revealing that enzyme activity, not transcript abundance, was the primary determinant of flux differences.  

- find switch controlled at enzyme activity level  
The metabolic switch was found to be controlled at the enzyme activity level, with no consistent correlation between mRNA levels and flux, supporting the hypothesis that physical constraints—not transcriptional regulation—drive the reorganization of metabolism.  

- discuss potential relevance to experimental observations in other organisms  
The findings suggest that similar crowding-driven metabolic switches may occur in other rapidly dividing organisms, including cancer cells and industrial yeast strains, where high growth rates impose similar physical constraints.  

- estimate crowding coefficients for E. coli proteins  
Crowding coefficients were estimated for 892 enzymatic reactions in E. coli using experimentally determined molecular volumes and turnover rates from the BRENDA database, yielding a mean value of 1.8 × 10⁻⁴ nm³·s·M⁻¹.  

- obtain enzymes' turnover rates from BRENDAdatabase  
Enzyme turnover rates (kcat) were obtained from the BRENDA database, supplemented with literature values for poorly annotated enzymes, and corrected for temperature and pH to match experimental conditions.  

- implement Flux Balance analysis with Molecular Crowding  
The FBAwMC model was implemented in MATLAB using the COBRA toolbox, with the crowding constraint expressed as a linear inequality on the total volume occupied by all enzymatic reactions.  

- solve optimization problem to maximize biomass production rate  
The optimization problem was solved using the Gurobi solver to maximize biomass production rate subject to stoichiometric, nutrient uptake, and crowding constraints, yielding a unique solution that matched experimental growth rates.  

- model crowding coefficients as noise  
To account for biological variability, crowding coefficients were modeled as stochastic variables drawn from a gamma distribution with shape parameter β = 3 and scale parameter derived from experimental data, improving model robustness.  

- predict fluxes for all reactions  
The model predicted fluxes for all 1,173 reactions in the network, with the top 10% of reactions accounting for 95% of the total biomass production flux, consistent with the concept of metabolic economy.  

- make predictions for E. coli metabolic fluxes on different carbon sources  
Predictions for E. coli metabolic fluxes on glucose, maltose, galactose, glycerol, and lactate were made, and the predicted hierarchy of substrate utilization closely matched experimental consumption patterns.  

- model increase of carbon source concentration in growth medium  
The concentration of glucose in the growth medium was increased from 0.01% to 0.2%, and the model predicted a corresponding increase in growth rate until the crowding limit was reached, after which further increases in glucose had no effect.  

- compute fluxes that maximize biomass production rate  
Fluxes were computed using linear programming to maximize biomass production rate under fixed nutrient availability and crowding constraints, with the solution demonstrating that glucose utilization dominated due to its superior volume efficiency.  

- analyze behavior of metabolic fluxes as function of growth rate  
The behavior of metabolic fluxes was analyzed as a function of growth rate, revealing that flux through glycolysis increased linearly until the crowding limit was approached, after which flux was redistributed to less volume-intensive pathways.  

- describe bacterial strain and general growth conditions  
The bacterial strain used was Escherichia coli MG1655, grown in M9 minimal medium with 0.04% w/v each of glucose, glycerol, galactose, lactate, and maltose at 37°C with orbital shaking at 200 rpm.  

- harvest biomass samples for flux measurements  
Biomass samples were harvested during mid-exponential phase, snap-frozen in liquid nitrogen, and stored at −80°C until metabolite extraction and isotopic labeling analysis.  

- perform metabolic enzyme activity assays  
Enzyme activity assays were performed using cell lysates and spectrophotometric monitoring of NADH or NADPH consumption, with specific activities calculated per milligram of total protein.  

- determine total protein concentration in enzyme samples  
Total protein concentration in enzyme samples was determined using the Bradford assay with bovine serum albumin as standard, with measurements performed in triplicate.  

- define units of enzyme activity  
Enzyme activity was defined as micromoles of substrate converted per minute per milligram of total protein (μmol/min/mg), consistent with standard biochemical conventions.  

- describe Bradford's assay for protein concentration  
Bradford’s assay was performed by mixing 10 μL of lysate with 200 μL of Coomassie Brilliant Blue G-250 reagent, incubating for 5 minutes at room temperature, and measuring absorbance at 595 nm.  

- outline flux measurement and analysis procedure  
Flux measurement involved feeding cells with ¹³C-labeled glucose, extracting intracellular metabolites, and analyzing mass isotopomer distributions using GC-MS to infer fluxes via isotopomer balancing.  

- explain GC-MS and NMR metabolome mapping platform  
GC-MS and NMR metabolome mapping platforms were used to quantify intracellular metabolite concentrations and isotopic labeling patterns, enabling the reconstruction of metabolic flux distributions.  

- describe mass isotopomer analysis (MIDA)  
Mass isotopomer analysis (MIDA) was performed by detecting the relative abundance of isotopologues of key metabolites such as pyruvate, lactate, and glutamate, and using these to constrain flux solutions.  

- outline statistical analysis using Student's t-test  
Statistical analysis was performed using Student’s t-test to compare fluxes between wild-type and mutant strains, with significance defined as p < 0.05.  

- describe glycogen glucose and RNA ribose stable isotope studies  
Stable isotope studies of glycogen glucose and RNA ribose were performed to trace carbon flow through storage and nucleic acid biosynthesis pathways, confirming the model’s prediction of reduced anabolic flux under crowding.  

- outline acid hydrolysis of cellular RNA  
Cellular RNA was isolated and subjected to acid hydrolysis at 90°C for 2 hours to release ribose, which was then derivatized with heptafluorobutyryl imidazole for GC-MS analysis.  

- describe derivatization of ribose and glycogen glucose  
Ribose and glycogen glucose were derivatized using heptafluorobutyryl imidazole under anhydrous conditions to enhance volatility and detectability by GC-MS.  

- explain mass spectral analysis of ribose and glycogen glucose  
Mass spectral analysis of derivatized ribose and glycogen glucose was performed using electron ionization with full-scan mode, and isotopomer ratios were quantified using NIST library matching.  

- describe lactate extraction and derivatization  
Lactate was extracted from culture supernatant using solid-phase extraction and derivatized with N,O-bis(trimethylsilyl)trifluoroacetamide for GC-MS quantification.  

- outline mass spectral analysis of lactate  
Mass spectral analysis of lactate was performed in selected ion monitoring mode at m/z 217 for the derivatized species, with quantification against a calibrated standard curve.  

- describe glutamate extraction and derivatization  
Glutamate was extracted via perchloric acid precipitation, neutralized, and derivatized with o-phthaldialdehyde for HPLC-fluorescence detection.  

- explain mass spectral analysis of glutamate  
Mass spectral analysis of glutamate was performed using LC-MS/MS with multiple reaction monitoring of transitions from m/z 148 to 130 and 104, with quantification against isotopically labeled internal standards.  

- describe fatty acid extraction and derivatization  
Fatty acids were extracted using chloroform-methanol, methylated with BF₃-methanol, and analyzed as fatty acid methyl esters by GC-MS.  

- outline mass spectral analysis of fatty acids  
Mass spectral analysis of fatty acid methyl esters was performed in electron impact mode, with identification by retention time and fragmentation pattern matching against commercial libraries.  

- describe GC/MS and NMR settings  
GC/MS settings included a DB-5 column, 70 eV ionization energy, and scan range of 50–600 m/z; NMR settings used a 600 MHz spectrometer with D₂O as solvent and TSP as internal reference.  

- describe flux data analysis and statistical methods  
Flux data analysis was performed using the COBRA toolbox in MATLAB, with statistical validation performed using bootstrapping and two-tailed t-tests with Bonferroni correction.  

- detail RNA preparation for microarray analysis  
RNA was prepared using the MasterPure RNA isolation kit, treated with DNase I, and quality-checked using a Bioanalyzer prior to labeling and hybridization to Affymetrix E. coli GeneChips.  

- outline STEM clustering analysis  
STEM clustering analysis was applied to time-series gene expression data to identify co-regulated gene clusters associated with metabolic transitions during growth rate increases.  

- describe querying expression data to identify specific expression profiles  
Expression data were queried using a pattern-matching algorithm to identify genes whose expression correlated with the onset of CCR, revealing enrichment for PTS components and glucose transporters.  

- detail querying gene expression of operons in the central carbon metabolism  
Operons involved in central carbon metabolism—including glp, mal, mgl, and pts—were queried for coordinated expression patterns, revealing that their activation was tightly coupled to growth rate rather than substrate availability.  

- state results of limited solvent capacity constraining metabolic rate  
The results demonstrated that limited solvent capacity constrains metabolic rate not by limiting enzyme expression, but by forcing a reorganization of enzyme allocation to minimize volume burden, thereby enabling higher growth rates.  

- estimate crowding coefficients using data from experimental reports  
Crowding coefficients were estimated from experimental reports on protein molecular volumes and enzymatic turnover rates, with corrections applied for cellular hydration and excluded volume effects.  

- compute crowding coefficients for E. coli enzymes  
Crowding coefficients for 892 enzymatic reactions in E. coli were computed, with the highest values associated with low-efficiency enzymes such as galactose permease and low values for high-efficiency enzymes such as glucokinase.  

- predict change of effective metabolic efficiency objective using FBAwMC  
The FBAwMC model predicted that the effective metabolic efficiency objective shifts from maximizing ATP yield to maximizing ATP yield per unit enzyme volume as growth rate increases.  

- evaluate relevance of solvent capacity constraint at physiological growth conditions  
The solvent capacity constraint was evaluated at physiological growth conditions and found to be active at growth rates above 0.2 h⁻¹, coinciding with the onset of acetate excretion and CCR.  

- predict metabolic switch characterized by change in effective criteria of metabolic efficiency  
The model predicted a metabolic switch from oxidative phosphorylation to mixed-mode metabolism at a growth rate of 0.25 h⁻¹, characterized by a change in the effective criterion of metabolic efficiency from yield to rate.  

- predict redistribution of metabolic fluxes  
The model predicted a redistribution of fluxes away from low-efficiency pathways such as lactate utilization and toward high-efficiency pathways such as glucose oxidation, consistent with experimental observations.  

- predict excretion of acetate at high growth rates  
The model predicted the excretion of acetate at high growth rates as a consequence of the need to reduce enzyme volume by bypassing the TCA cycle and utilizing a more compact ATP-generating pathway.  

- compare FBAwMC-predicted metabolic fluxes with experimental values  
FBAwMC-predicted fluxes were compared to experimental values obtained from ¹³C-labeling studies, with a Pearson correlation coefficient of r = 0.87, indicating strong agreement.  

- identify regulatory mechanism(s) controlling metabolic switch  
The regulatory mechanism controlling the metabolic switch was found to be enzyme activity, not transcriptional regulation, as mRNA levels of key enzymes showed poor correlation with flux changes.  

- measure in vitro activity of selected enzymes  
In vitro activity of selected enzymes—including pyruvate kinase, acetate kinase, and phosphotransacetylase—was measured using cell-free lysates, and values were found to correlate with in vivo flux predictions.  

- correlate changes in enzyme activities with measured flux rates  
Changes in enzyme activities were found to correlate strongly with measured flux rates (r > 0.9), supporting the conclusion that enzyme allocation—not gene expression—drives metabolic reorganization.  

- analyze mRNA levels of enzyme-encoding genes  
mRNA levels of enzyme-encoding genes were analyzed using microarrays, revealing that while some genes were upregulated, many showed no change despite large flux increases, indicating post-transcriptional control.  

- discuss lack of correlation between measured metabolic fluxes and mRNA levels  
The lack of correlation between metabolic fluxes and mRNA levels suggests that classical transcriptional regulation is insufficient to explain metabolic behavior under crowding constraints.  

- discuss control of metabolic switch by enzyme activities  
Control of the metabolic switch is exerted at the level of enzyme activity, where the physical constraint of crowding forces cells to prioritize enzymes with the highest catalytic efficiency per unit volume.  

- discuss significance of solvent capacity constraint for systems biology  
The solvent capacity constraint provides a unifying physical principle for understanding metabolic regulation, unifying observations of CCR, acetate excretion, and growth rate dependency under a single mechanistic framework.  

- discuss incorporation of solvent capacity constraint into FBA modeling framework  
The incorporation of the solvent capacity constraint into FBA modeling transforms it from a purely stoichiometric tool into a biophysically grounded predictive framework capable of capturing metabolic trade-offs.  

- discuss flux predictions for E. coli metabolism  
Flux predictions for E. coli metabolism under varying nutrient conditions were highly accurate, with the model correctly predicting the order of substrate utilization and the timing of metabolic switches.  

- discuss consideration of reaction kinetics via crowding coefficients  
The consideration of reaction kinetics via crowding coefficients enables the model to distinguish between reactions that are stoichiometrically equivalent but physically inequivalent in terms of volume cost.  

- discuss interpretation of metabolic switch using solvent capacity constraint  
The metabolic switch observed in E. coli is interpreted as a physical optimization strategy to maintain intracellular crowding within a narrow physiological range, rather than a regulatory response to nutrient availability.  

- discuss maximization of biomass production rate objective  
The maximization of biomass production rate remains the primary objective, but the constraint of crowding ensures that this objective is pursued in a manner consistent with the physical laws governing intracellular space.  

- discuss consistency with expectation of fastest growth rates  
The model’s predictions are consistent with the expectation that cells evolve to achieve the fastest possible growth rates under physical constraints, with enzyme allocation serving as the key adaptive parameter.  

- discuss explanation of acetate excretion by solvent capacity constraint  
Acetate excretion is explained as a consequence of the need to generate ATP with fewer enzymes, as the pyruvate-to-acetate pathway requires fewer catalytic steps than the full TCA cycle.  

- discuss possibility of acetate excretion due to limited oxygen availability  
While limited oxygen availability can also cause acetate excretion, the model demonstrates that acetate production occurs even under fully aerobic conditions, implicating crowding as a primary driver.  

- discuss significance of results for systems biology  
The results establish macromolecular crowding as a fundamental constraint shaping metabolic evolution and regulation, providing a new paradigm for systems biology that integrates physics with biochemistry.  

- discuss development of modeling framework for quantitative description of cellular metabolism  
The developed modeling framework provides the first quantitative description of cellular metabolism that accounts for spatial limitations, enabling accurate prediction of metabolic behavior across growth conditions.  

- discuss uncovering physicochemical constraints influencing cellular metabolism  
The work uncovers the previously overlooked physicochemical constraint of intracellular crowding as a dominant factor influencing enzyme allocation, metabolic pathway selection, and growth rate optimization.  

- discuss significance of limited solvent capacity for fast growing E. coli cells  
Limited solvent capacity is shown to be a critical determinant of metabolic strategy in fast-growing E. coli cells, dictating not only what substrates are used but how efficiently they are converted into biomass.  

- discuss incorporation of solvent capacity constraint into FBAwMC model  
The incorporation of the solvent capacity constraint into the FBAwMC model enables it to predict metabolic behavior that is both quantitatively accurate and biologically plausible, surpassing the predictive power of traditional FBA.  

- discuss predictions of FBAwMC model  
The predictions of the FBAwMC model—including substrate utilization hierarchy, metabolic switch timing, and enzyme activity profiles—were validated across multiple independent experimental datasets.  

- discuss flux balance approximations  
Flux balance approximations remain valid under crowding constraints, provided that the objective function and constraints are appropriately modified to reflect physical limitations.  

- discuss reaction kinetics via crowding coefficients  
Reaction kinetics are effectively captured by crowding coefficients, which encode both catalytic efficiency and molecular volume, providing a unified metric for evaluating metabolic trade-offs.  

- discuss significance of results for understanding E. coli metabolism  
The results fundamentally alter the understanding of E. coli metabolism by demonstrating that its metabolic behavior is not governed by transcriptional logic alone, but by the physical imperative to optimize space usage.  

- discuss implications for systems biology and quantitative modeling  
The implications extend to systems biology broadly, suggesting that all cellular networks—metabolic, signaling, and regulatory—are shaped by the same physical constraints, and that accurate modeling must account for them.  

### Example 2

- develop modified FBA model  
A modified FBA model was developed by incorporating a macromolecular crowding constraint derived from experimentally determined enzyme volumes and turnover rates, enabling the prediction of physiologically realistic metabolic states.  

- incorporate solvent capacity constraint  
The solvent capacity constraint was incorporated as a linear inequality limiting the sum of enzyme volumes across all reactions to a maximum of 0.45, consistent with measured intracellular protein densities.  

- predict maximum growth rate  
The model predicted a maximum growth rate of 0.7 h⁻¹ for E. coli in glucose-rich medium, matching experimental measurements in chemostat cultures.  

- test model predictions  
Model predictions were tested against independent datasets from chemostat cultures, ¹³C-flux measurements, and proteomic analyses, with high concordance observed across all validation metrics.  

- obtain good agreement between model and experiment  
Good agreement was obtained between model predictions and experimental measurements of growth rate, substrate consumption rate, and metabolite secretion, with correlation coefficients exceeding 0.85.  

- suggest macromolecular crowding constraint  
The results strongly suggest that macromolecular crowding is the primary constraint governing metabolic strategy in rapidly growing cells, superseding transcriptional regulation in importance.  

- implement FBAwMC modeling framework  
The FBAwMC modeling framework was implemented in MATLAB using the COBRA toolbox, with custom constraint modules for crowding and kinetic efficiency.  

- define optimization problem  
The optimization problem was defined as maximizing biomass production rate subject to stoichiometric balance, nutrient uptake bounds, and a crowding constraint on total enzyme volume.  

- model crowding coefficients as noise  
Crowding coefficients were modeled as stochastic variables drawn from a gamma distribution with shape parameter β = 3, reflecting biological variability in enzyme packing and hydration.  

- assign random value to crowding coefficients  
Random values were assigned to crowding coefficients in 100 Monte Carlo simulations to assess the robustness of the predicted flux distribution.  

- use gamma distribution  
The gamma distribution was chosen for its ability to model positive-valued, skewed variables such as enzyme volumes, and provided superior fit to experimental data compared to normal or log-normal distributions.  

- obtain results for β=3  
For β = 3, the model predicted a stable, reproducible flux distribution with low variance across simulations, confirming the robustness of the crowding constraint.  

- test sensitivity of results  
Sensitivity analysis revealed that the predicted substrate utilization hierarchy was insensitive to ±20% variation in crowding coefficients, indicating that the overall metabolic strategy is robust.  

- obtain maximum growth rate for each carbon source  
Maximum growth rates were obtained for each carbon source, with glucose yielding the highest rate (0.7 h⁻¹), followed by maltose (0.62 h⁻¹), galactose (0.58 h⁻¹), glycerol (0.51 h⁻¹), and lactate (0.42 h⁻¹).  

- fit average crowding coefficient  
The average crowding coefficient across all reactions was fit to 1.8 × 10⁻⁴ nm³·s·M⁻¹, derived from the median of experimentally measured enzyme volumes and turnover rates.  

- model temporal order of substrate uptake  
The model successfully modeled the temporal order of substrate uptake, predicting glucose first, followed by maltose, galactose, glycerol, and lactate, matching experimental consumption profiles.  

- consider initial concentration of substrates  
Initial substrate concentrations were considered in the model, with uptake rates scaled according to availability, but the order of utilization remained invariant under all tested conditions.  

- integrate differential equations  
Differential equations describing substrate consumption and biomass growth were integrated numerically using the Runge-Kutta method over a 24-hour time course.  

- obtain maximum growth rate and fluxes  
Maximum growth rate and flux distributions were obtained for each substrate condition, with the model correctly predicting that glucose supports the highest flux through glycolysis and biomass synthesis.  

- solve three FBAwMC problems  
Three separate FBAwMC problems were solved: (1) single glucose culture, (2) mixed substrate culture, and (3) mutant strain lacking glucose transporter, with each yielding distinct, biologically consistent solutions.  

- estimate crowding coefficients from experimental measurements  
Crowding coefficients were estimated from experimental measurements of protein molecular volumes in E. coli and published turnover rates, with corrections applied for cellular hydration and excluded volume.  

- define crowding coefficients  
Crowding coefficients were defined as the ratio of molecular volume (nm³) to catalytic efficiency (kcat/Km, s⁻¹M⁻¹), yielding a dimensionless metric of volume cost per metabolic unit.  

- decompose proportionality factor  
The proportionality factor relating enzyme concentration to flux was decomposed into enzyme-specific parameters, enabling the identification of high-cost and low-cost reactions.  

- estimate crowding coefficients from turnover rate  
Crowding coefficients were estimated from turnover rate (kcat) and molecular volume, with low kcat and high volume resulting in high crowding cost, and vice versa.  

- describe growth experiments  
Growth experiments were performed in M9 minimal medium with 0.04% w/v of each of five carbon substrates, with cultures monitored for OD600, pH, and residual substrate concentration.  

- detail carbon substrate and microarray analyses  
Carbon substrate consumption was analyzed by HPLC, and microarray analyses were performed to assess gene expression changes across growth phases.  

- describe bacterial strains and growth conditions  
Wild-type E. coli MG1655 and ΔptsG mutant strains were grown in M9 minimal medium at 37°C with 200 rpm shaking, with growth monitored until stationary phase.  

- detail growth experiments using M9 minimal medium  
Growth experiments using M9 minimal medium were conducted in 50 mL flasks with 10 mL culture volume, inoculated at OD600 = 0.035, and sampled every 30 minutes for 8 hours.  

- assess transcriptome state  
Transcriptome state was assessed using Affymetrix microarrays, with data normalized using RMA and analyzed for differential expression across growth phases.  

- determine maximum growth rates  
Maximum growth rates were determined from the exponential phase slope of OD600 over time, with values averaged across triplicate cultures.  

- describe method of continuous cultivation  
Continuous cultivation was performed in a Labfors bioreactor using a chemostat mode with dilution rates ranging from 0.1 to 0.7 h⁻¹, with steady-state confirmed by stable OD and metabolite profiles.  

- calculate maximum growth rate  
Maximum growth rate was calculated as the highest observed exponential growth rate under saturating nutrient conditions, with the FBAwMC model predicting 0.7 h⁻¹, matching experimental values.  

- determine residual concentration of carbon sources  
Residual concentrations of carbon sources were determined by HPLC after centrifugation and filtration, with detection limits of 0.01 mM for all substrates.  

- analyze microarray samples  
Microarray samples were analyzed using the affy package in R, with background correction performed using MAS5 and normalization using qspline.  

- identify genes with sequence-specific hybridization  
Genes with sequence-specific hybridization were identified using stringent hybridization thresholds (signal-to-noise ratio > 5) and filtered for presence calls in all replicates.  

- analyze microarray data for individual carbon sources  
Microarray data for individual carbon sources were analyzed to identify genes upregulated in response to substrate induction, revealing that induction was strongest in single-substrate cultures.  

- examine top 150 genes  
The top 150 differentially expressed genes were examined for functional enrichment, revealing strong enrichment for transporters, catabolic enzymes, and stress response proteins.  

- introduce querying expression data  
Expression data were queried using pattern recognition algorithms to identify genes whose expression correlated with the onset of CCR, identifying ptsG, glk, and acs as key markers.  

- identify specific expression profiles  
Specific expression profiles were identified for operons involved in glucose, maltose, and galactose metabolism, revealing that their expression was tightly coupled to growth rate rather than substrate availability.  

- hierarchical clustering of time-series gene expression data  
Hierarchical clustering of time-series gene expression data revealed three distinct clusters: (1) genes induced at low growth, (2) genes induced at medium growth, and (3) genes induced at high growth.  

- probabilistic clustering of time-series data  
Probabilistic clustering using hidden Markov models identified transition points between metabolic phases, with high confidence in the switch from OxPhos to mixed metabolism at 0.25 h⁻¹.  

- stress response analysis  
Stress response analysis revealed upregulation of rpoS and chaperone genes at low growth rates, consistent with nutrient limitation, and downregulation at high growth rates.  

- biological functions of various genes  
Biological functions of various genes were annotated using GO terms, revealing that transporters and catabolic enzymes dominated the high-growth cluster, while biosynthesis and repair genes dominated the low-growth cluster.  

- results introduction  
Results demonstrate that substrate utilization hierarchy is governed by physical constraints rather than transcriptional regulation, and that the FBAwMC model accurately predicts this behavior.  

- FBA with molecular crowding predicts relative maximum growth  
FBA with molecular crowding predicts the relative maximum growth rates on different substrates with high accuracy, ranking glucose as the most efficient, followed by maltose, galactose, glycerol, and lactate.  

- substrate hierarchy utilization by E. coli cells  
E. coli cells utilize substrates in a strict hierarchy dictated by the crowding coefficient of their corresponding metabolic pathways, with glucose consumed first due to its low volume cost.  

- FBAwMC E. coli model on mixed-substrate conditions  
The FBAwMC E. coli model accurately predicts the sequential consumption of substrates in mixed-substrate cultures, with glucose consumed first and lactate last, matching experimental observations.  

- surrogate markers of cellular metabolism  
Surrogate markers of cellular metabolism—including acetate secretion, growth rate, and cell volume—were found to correlate strongly with predicted metabolic states, validating the model.  

- mode and sequence of substrate utilization correlate with gene expression  
The mode and sequence of substrate utilization correlate with gene expression patterns, but only for genes encoding enzymes with high crowding coefficients, not for regulatory genes.  

- activation of stress programs upon switching metabolic phases  
Activation of stress programs occurs upon switching from low- to high-growth metabolism, indicating that metabolic reorganization triggers a physiological transition.  

- hierarchical clustering with optimal leaf ordering  
Hierarchical clustering with optimal leaf ordering revealed that gene expression clusters correspond to metabolic phases, with clear boundaries between low, medium, and high growth states.  

- principal component analysis  
Principal component analysis identified the first principal component as explaining 82% of variance, corresponding to growth rate, and the second as explaining 10%, corresponding to substrate identity.  

- probabilistic clustering method based on hidden Markov models  
A probabilistic clustering method based on hidden Markov models successfully identified the transition points between metabolic phases with 95% accuracy.  

- discussion introduction  
The discussion introduces the hypothesis that carbon catabolite repression is not a regulatory mechanism per se, but a consequence of the need to maintain optimal intracellular crowding.  

- identification of principles that define growth and substrate utilization mode  
The principles defining growth and substrate utilization mode are physical: cells prioritize substrates that yield the highest growth rate per unit of enzyme volume.  

- experimental results indicate three major metabolic phases  
Experimental results indicate three major metabolic phases: low-growth OxPhos, intermediate mixed metabolism, and high-growth glycolytic flux, each with distinct enzyme allocation patterns.  

- global mRNA expression data indicate partial stress response  
Global mRNA expression data indicate a partial stress response during metabolic transitions, suggesting that metabolic reorganization imposes a physiological burden.  

- activation of foraging program upon exhaustion of substrates  
Activation of a foraging program upon exhaustion of preferred substrates is observed, consistent with a strategy to minimize metabolic downtime.  

- FBAwMC model captures main features of metabolic activities  
The FBAwMC model captures the main features of metabolic activities—including substrate hierarchy, metabolic switch timing, and acetate excretion—without invoking regulatory logic.  

- correlation between in vivo relative maximal growth rates and in silico predictions  
A strong correlation (r = 0.93) was found between in vivo relative maximal growth rates and in silico predictions, confirming the model’s predictive power.  

- FBAwMC model predicts three metabolic phases and hierarchical mode  
The FBAwMC model predicts three metabolic phases and a hierarchical substrate utilization mode that matches experimental data across multiple strains and conditions.  

- solvent capacity of cytoplasm determines growth rate  
The solvent capacity of the cytoplasm determines the maximum achievable growth rate by limiting the number of enzymes that can be expressed without exceeding physical volume bounds.  

- cells preferentially consume carbon source resulting in highest growth rate  
Cells preferentially consume the carbon source that yields the highest growth rate per unit of enzyme volume, as predicted by the crowding coefficient.  

- two discrepancies of FBAwMC model predictions  
Two discrepancies were observed: (1) higher than predicted acetate secretion, and (2) earlier substrate uptake than predicted, suggesting minor regulatory influences beyond crowding.  

- contribution of other cell components apart from metabolic enzymes  
The contribution of other cell components—such as ribosomes, membranes, and nucleic acids—was found to be significant, necessitating their inclusion in future model iterations.  

- FBAwMC model may underestimate impact of macromolecular crowding  
The FBAwMC model may underestimate the impact of macromolecular crowding due to simplifications in volume estimation and neglect of spatial heterogeneity.  

- acetate secretion correlated with increased carbon source uptake rate  
Acetate secretion was correlated with increased carbon source uptake rate, indicating that high flux demands trigger overflow metabolism even under aerobic conditions.  

- maximum enzyme concentration is key constraint shaping hierarchy of substrate utilization  
Maximum enzyme concentration, rather than gene expression or transporter affinity, is the key constraint shaping the hierarchy of substrate utilization.  

- regulatory mechanisms acting in E. coli and other organisms  
Regulatory mechanisms in E. coli and other organisms may have evolved not to control metabolism per se, but to enforce the physical constraints imposed by macromolecular crowding.  

- constrained optimization approaches help understand regulatory mechanisms  
Constrained optimization approaches help understand regulatory mechanisms by revealing that many “regulatory” behaviors are emergent properties of physical limits.  

- results of FBAwMC model on single carbon-limited media  
The results of the FBAwMC model on single carbon-limited media matched experimental growth rates and flux distributions, validating the model’s applicability beyond mixed-substrate conditions.  

- comparison of predicted and measured growth rates  
Comparison of predicted and measured growth rates yielded a correlation coefficient of r = 0.91, with mean absolute error of 0.04 h⁻¹.  

- FBAwMC model predicts maximal growth rate of E. coli MG1655 cells  
The FBAwMC model predicts a maximal growth rate of 0.70 h⁻¹ for E. coli MG1655 in glucose-rich medium, matching the highest observed experimental value.  

- substrate utilization of E. coli cells in mixed carbon-limited medium  
Substrate utilization in mixed carbon-limited medium follows a strict hierarchy: glucose > maltose > galactose > glycerol > lactate, as predicted by the model.  

- FBAwMC model predicts substrate uptake and consumption  
The FBAwMC model accurately predicts the timing and rate of substrate uptake and consumption, including the suppression of lactate utilization until glucose is exhausted.  

- changes in pH and oxygen concentrations in growth medium  
Changes in pH and oxygen concentrations were minimal and did not explain the observed substrate utilization hierarchy, supporting the crowding hypothesis.  

- expression of genes participating in uptake modules  
Expression of genes participating in uptake modules was highest during the transition phase, not during substrate consumption, suggesting preparatory regulation.  

- TimeSearcher identifies genes with similar expression patterns  
TimeSearcher identified genes with similar expression patterns across growth phases, revealing co-regulated clusters of transporters and catabolic enzymes.  

- genes displaying expression patterns similar to those of query genes  
Genes displaying expression patterns similar to query genes were enriched for metabolic functions, not regulatory ones, reinforcing the physical interpretation.  

- activation of stress programs upon switching metabolic phases  
Activation of stress programs upon switching metabolic phases suggests that metabolic reorganization is energetically costly and triggers a global stress response.  

- global state of E. coli transcriptome during metabolic phases  
The global state of the E. coli transcriptome shifts from a maintenance mode at low growth to a biosynthesis and flux mode at high growth, consistent with the model’s predictions.  

- hierarchical clustering with optimal leaf ordering  
Hierarchical clustering with optimal leaf ordering revealed clear transitions between metabolic states, with minimal overlap between phases.  

- principal component analysis  
Principal component analysis confirmed that growth rate is the dominant axis of variation, accounting for over 80% of transcriptome variance.  

- probabilistic clustering method based on hidden Markov models  
The probabilistic clustering method based on hidden Markov models identified transition points with 96% accuracy, validating the discrete nature of metabolic phases.  

- discussion of results and implications  
The results demonstrate that macromolecular crowding is a fundamental constraint shaping metabolic behavior, and that traditional regulatory frameworks are insufficient to explain observed phenotypes.  

### Example 3

- introduce flux balance in S. cerevisiae with molecular crowding and kinetic modeling  
Flux balance analysis with molecular crowding and kinetic modeling was introduced for Saccharomyces cerevisiae to predict glycolytic flux under varying nutrient conditions.  

- motivate hypothesis of optimal use of intracellular resources  
The hypothesis that cells optimize the use of intracellular resources under physical constraints was motivated by the observation that yeast cells exhibit metabolic switching behavior analogous to bacterial CCR.  

- describe glycolysis pathway in S. cerevisiae  
The glycolysis pathway in S. cerevisiae consists of ten enzymatic steps from glucose to pyruvate, with branching to ethanol and glycerol, and is highly conserved across eukaryotes.  

- present kinetic model of glycolysis  
A kinetic model of glycolysis was presented, incorporating Michaelis-Menten kinetics for each enzyme, allosteric regulation, and substrate inhibition.  

- illustrate glycolysis pathway with FIG. 21  
FIG. 21 illustrates the glycolysis pathway with enzyme names, reaction arrows, and metabolite nodes, annotated with kinetic parameters and crowding coefficients.  

- define optimization objective of glycolysis rate  
The optimization objective was defined as maximizing the rate of ATP production through glycolysis, subject to crowding and substrate availability constraints.  

- derive rate equation models for glycolysis reactions  
Rate equation models were derived for each glycolytic reaction using enzyme-specific kinetic parameters from the BRENDA database and in vivo metabolite concentrations.  

- describe glucose transport reaction model  
The glucose transport reaction was modeled as a facilitated diffusion with a Michaelis-Menten term for the Hxt transporters, with crowding coefficient derived from membrane protein volume.  

- describe hexokinase reaction model  
The hexokinase reaction was modeled as a reversible enzyme-catalyzed conversion of glucose to glucose-6-phosphate, with ATP as cofactor and crowding coefficient of 2.1 × 10⁻⁴ nm³·s·M⁻¹.  

- describe phosphoglucoisomerase reaction model  
The phosphoglucoisomerase reaction was modeled as an isomerization with low crowding cost due to small enzyme volume and high turnover rate.  

- describe phosphofructokinase-1 reaction model  
The phosphofructokinase-1 reaction was modeled with allosteric inhibition by ATP and activation by AMP, with high crowding cost due to large enzyme size and moderate turnover.  

- describe aldolase reaction model  
The aldolase reaction was modeled as a cleavage reaction with high catalytic efficiency and low volume cost, making it a low-cost flux channel.  

- describe triosephosphate isomerase reaction model  
The triosephosphate isomerase reaction was modeled with near-equilibrium kinetics and very low crowding cost, consistent with its high abundance and efficiency.  

- describe glyceraldehyde 3-phosphate dehydrogenase reaction model  
The glyceraldehyde 3-phosphate dehydrogenase reaction was modeled with NAD⁺ dependency and high crowding cost due to enzyme size and moderate turnover.  

- describe pyruvate kinase and glycerol 3-phosphate dehydrogenase reaction models  
Pyruvate kinase was modeled with high crowding cost and strong activation by fructose-1,6-bisphosphate; glycerol 3-phosphate dehydrogenase was modeled as a competing pathway with lower ATP yield but lower crowding cost.  

- introduce catalytic constants  
Catalytic constants (kcat and Km) were introduced for each enzyme, derived from in vitro assays and corrected for intracellular conditions of pH, ionic strength, and crowding.  

- obtain experimental estimates  
Experimental estimates of enzyme concentrations and metabolite levels were obtained from proteomic and metabolomic studies of S. cerevisiae under chemostat conditions.  

- describe cell density and specific volume  
Cell density was measured at 1.14 g/mL and specific volume at 120 fL/cell, providing the basis for calculating maximum allowable enzyme volume.  

- obtain optimal metabolite concentrations  
Optimal metabolite concentrations were obtained through global optimization, revealing that intermediates such as fructose-1,6-bisphosphate and phosphoenolpyruvate are maintained at levels that balance flux and enzyme saturation.  

- perform parameter sensitivity analysis  
Parameter sensitivity analysis revealed that the model is most sensitive to the crowding coefficient of phosphofructokinase and pyruvate kinase, and least sensitive to triosephosphate isomerase.  

- introduce limited solvent capacity constraint  
A limited solvent capacity constraint was introduced, limiting the total volume occupied by glycolytic enzymes to 15% of cytoplasmic volume.  

- derive equation for reaction rate  
The equation for reaction rate was derived as v = (kcat × [E] × [S]) / (Km + [S]), with [E] constrained by the crowding limit.  

- define crowding coefficients  
Crowding coefficients were defined as molecular volume divided by catalytic efficiency, with values ranging from 0.5 × 10⁻⁴ for small enzymes to 4.0 × 10⁻⁴ for large complexes.  

- analyze hypothetical three metabolites pathway  
A hypothetical three-metabolite pathway was analyzed to demonstrate that optimal flux is achieved not by maximizing enzyme concentration, but by allocating enzyme volume to the most efficient steps.  

- model reaction rates with Michaelis-Menten rate equations  
Reaction rates were modeled using Michaelis-Menten rate equations, with enzyme concentrations treated as decision variables subject to crowding constraints.  

- compute maximum metabolic rate  
Maximum metabolic rate was computed as the point at which increasing enzyme concentration no longer increases flux due to crowding saturation.  

- apply model to S. cerevisiae glycolysis  
The model was applied to S. cerevisiae glycolysis, predicting that ATP production is maximized when enzyme volume is allocated to high-efficiency steps such as aldolase and triosephosphate isomerase, and reduced at high-cost steps such as pyruvate kinase.  

- investigate dependency of glycolysis rate on metabolite concentrations  
The dependency of glycolysis rate on metabolite concentrations was investigated, revealing that flux is most sensitive to concentrations of fructose-1,6-bisphosphate and ATP.  

- perform global optimization of metabolite concentrations  
Global optimization of metabolite concentrations was performed using simulated annealing, yielding a set of concentrations that maximize ATP production under crowding constraints.  

- predict optimal metabolite concentrations  
Optimal metabolite concentrations were predicted to be: glucose 5 mM, fructose-6-phosphate 1.2 mM, fructose-1,6-bisphosphate 0.8 mM, pyruvate 2.1 mM, and ATP 2.5 mM.  

- predict enzyme activities  
Enzyme activities were predicted to be highest for triosephosphate isomerase and lowest for phosphofructokinase, consistent with their crowding costs and catalytic efficiencies.  

- test optimal metabolite concentration hypothesis  
The hypothesis was tested by measuring intracellular metabolite concentrations in chemostat cultures, finding strong agreement with predicted values (r = 0.94).  

- explore alternative optimization objectives  
Alternative optimization objectives—including ethanol yield, biomass yield, and ATP yield per glucose—were explored, revealing that each leads to a distinct enzyme allocation strategy.  

- compute maximum glycolysis rate  
Maximum glycolysis rate was computed as 18.7 mmol/gDW/h under optimal crowding conditions, matching experimental measurements in high-glucose cultures.  

- discuss implications of limited solvent capacity constraint  
The implications are profound: cells do not maximize enzyme expression—they maximize flux per unit volume, and this principle underlies metabolic evolution across domains of life.  

- summarize previous work on E. coli  
Previous work on E. coli demonstrated that macromolecular crowding governs substrate utilization hierarchy and metabolic switching, with similar principles extended here to eukaryotes.  

- discuss advantages of modeling approach  
The advantages include predictive accuracy, mechanistic insight, and the ability to guide metabolic engineering without relying on empirical trial-and-error.  

- discuss limitations of modeling approach  
Limitations include reliance on kinetic parameters from in vitro studies, potential underestimation of spatial heterogeneity, and lack of dynamic regulation in steady-state models.  

- introduce full kinetic model of glycolysis  
A full kinetic model of glycolysis was introduced, incorporating all ten reactions with allosteric regulators, cofactors, and enzyme isoforms.  

- predict optimal intermediate metabolite concentrations  
The model predicts that intermediate metabolite concentrations are not fixed, but dynamically optimized to balance flux, enzyme saturation, and crowding cost.  

- predict enzyme activities  
Enzyme activities are predicted to vary across growth conditions, with high-growth cells expressing more of low-cost enzymes and less of high-cost ones.  

- compare predictions with experimental values  
Predictions were compared with experimental values from proteomic and metabolomic studies, with Pearson correlation coefficients exceeding 0.90 for enzyme activities and 0.88 for metabolite concentrations.  

- discuss discrepancies between predictions and experimental values  
Discrepancies were attributed to post-translational modifications not modeled, such as phosphorylation of phosphofructokinase, and to mitochondrial shuttles not included in the cytoplasmic model.  

- propose method for predicting metabolite concentrations and enzyme activities  
The method proposes integrating crowding constraints with kinetic modeling and global optimization to predict both enzyme activities and metabolite concentrations simultaneously.  

- discuss applicability of method  
The method is applicable to any organism with a well-reconstructed metabolic network and experimentally derived kinetic parameters.  

- discuss physical constraint of total cell volume  
The physical constraint of total cell volume is a universal feature of all living cells, and its incorporation into metabolic models is essential for accurate prediction.  

- discuss enzyme molar volumes and density  
Enzyme molar volumes range from 50 to 200 nm³, and their density in the cytoplasm is approximately 150–200 mg/mL, consistent with observed crowding levels.  

- discuss advantages of modeling framework  
Advantages include biological plausibility, predictive power, and the ability to uncover hidden design principles of metabolism.  

- discuss limitations of modeling framework  
Limitations include computational intensity, dependency on parameter quality, and inability to model transient dynamics such as oscillations or pulses.  

- discuss steady state approximation  
The steady state approximation is valid for chemostat cultures and exponential growth phases, but not for transient responses to perturbation.  

- discuss inability to model dynamical processes  
The model cannot model dynamical processes such as diauxic shifts or metabolic oscillations, which require time-dependent kinetic equations.  

- discuss observed metabolite concentration oscillations  
Observed metabolite concentration oscillations in yeast are not captured by the model and represent a future extension to incorporate temporal dynamics.  

- conclude modeling framework  
The modeling framework provides a physically grounded, predictive, and engineering-ready platform for understanding and optimizing cellular metabolism.  

### Example 4

- introduce alternative glycolysis pathway  
An alternative glycolysis pathway was introduced, in which pyruvate is converted to alanine via alanine aminotransferase, bypassing pyruvate kinase and generating ATP through substrate-level phosphorylation.  

- describe ATP generation in normal cells  
In normal cells, ATP is generated through glycolysis, the TCA cycle, and oxidative phosphorylation, with glycolysis providing rapid but inefficient ATP production.  

- describe Warburg effect  
The Warburg effect describes the preference of cancer cells for glycolysis over oxidative phosphorylation even under aerobic conditions, a phenomenon long attributed to mitochondrial dysfunction.  

- explain molecular crowding constraint  
The molecular crowding constraint explains the Warburg effect not as a defect, but as an adaptation to maintain intracellular volume homeostasis by reducing the number of enzymes required for ATP production.  

- introduce genome-scale model of human cell metabolism  
A genome-scale model of human cell metabolism was introduced, incorporating 3,800 reactions, 2,700 metabolites, and 1,900 genes, with compartmentalization for cytoplasm, mitochondria, and nucleus.  

- describe flux balance model  
The flux balance model was described as a linear programming framework that maximizes biomass production subject to stoichiometric and crowding constraints.  

- define nutrient import reactions  
Nutrient import reactions were defined for glucose, glutamine, lactate, and fatty acids, with transporters assigned crowding coefficients based on membrane localization.  

- define reactions outside mitochondria  
Reactions outside mitochondria include glycolysis, pentose phosphate pathway, and amino acid biosynthesis, all assigned cytoplasmic volume constraints.  

- define reactions in mitochondria  
Reactions in mitochondria include TCA cycle, oxidative phosphorylation, and fatty acid oxidation, with volume constraints based on mitochondrial matrix density.  

- define compartment densities  
Compartment densities were defined as 0.38 for cytoplasm, 0.42 for mitochondrial matrix, and 0.25 for nucleus, based on electron microscopy measurements.  

- formulate optimization problem  
The optimization problem was formulated to maximize ATP production per unit of enzyme volume, subject to nutrient import limits and compartmental crowding constraints.  

- describe metabolic constraints  
Metabolic constraints included upper and lower bounds on fluxes, reaction reversibility, and thermodynamic feasibility based on Gibbs free energy calculations.  

- describe flux balance constraints  
Flux balance constraints ensured mass conservation for all metabolites, with zero net accumulation under steady-state conditions.  

- describe minimum/maximum flux constraints  
Minimum and maximum flux constraints were imposed based on enzyme abundance measurements from proteomic datasets.  

- describe minimum/maximum volume fraction constraints  
Minimum and maximum volume fraction constraints were imposed to ensure that enzyme volumes remained within physiologically observed ranges.  

- describe molecular crowding constraints  
Molecular crowding constraints were applied as a linear sum of enzyme volumes across all reactions, limited to 0.45 total volume fraction.  

- estimate model parameters  
Model parameters—including molecular volumes, turnover rates, and enzyme abundances—were estimated from published proteomic and metabolomic datasets.  

- discuss cost of importing molecules  
The cost of importing molecules was discussed as an additional crowding burden, with transporters contributing up to 12% of total cytoplasmic volume.  

- discuss effective turnover numbers  
Effective turnover numbers were discussed as the product of intrinsic kcat and the fraction of enzyme in active conformation, reduced by crowding-induced diffusion limitations.  

- discuss enzyme crowding coefficients  
Enzyme crowding coefficients were discussed as the dominant factor determining pathway preference, with high-efficiency, low-volume enzymes favored under crowding.  

- discuss ribosome crowding coefficient  
The ribosome crowding coefficient was discussed as a fixed volume component that sets the baseline for available space, limiting the maximum enzyme concentration.  

- discuss mitochondrial crowding coefficient  
The mitochondrial crowding coefficient was discussed as higher than cytoplasmic due to dense packing of respiratory complexes, limiting OxPhos capacity.  

- formulate flux balance equation for proteins  
The flux balance equation for proteins was formulated to account for synthesis and degradation, with crowding constraints limiting the rate of protein accumulation.  

- describe protein synthesis  
Protein synthesis was described as a process requiring ribosomes, tRNA, amino acids, and ATP, with crowding reducing translation efficiency.  

- describe protein degradation  
Protein degradation was described as a proteasome-dependent process, with degradation rates dependent on protein misfolding and crowding-induced aggregation.  

- describe effective protein dilution/degradation  
Effective protein dilution/degradation was described as the balance between synthesis rate and cell growth rate, with crowding reducing synthesis and increasing degradation.  

- model alternative glycolysis pathway  
The alternative glycolysis pathway was modeled by introducing alanine aminotransferase and alanine dehydrogenase, enabling ATP generation without pyruvate kinase.  

- describe changes in relative macromolecular densities  
The model predicted that the alternative pathway reduces enzyme volume by 22% compared to conventional glycolysis, with compensatory increase in mitochondrial volume.  

- show predicted relative volume fraction occupied by enzymes  
Predicted relative volume fractions showed that conventional glycolysis occupies 18% of cytoplasm, while the alternative pathway occupies only 14%.  

- show predicted relative volume fraction occupied by mitochondria  
Predicted mitochondrial volume fraction increased from 15% to 20% in the alternative pathway, consistent with increased reliance on OxPhos.  

- show predicted relative volume fraction occupied by ribosomes  
Predicted ribosome volume fraction remained constant at 12%, confirming that crowding limits are enforced at the level of total protein.  

- describe impact of altering model parameters  
Altering model parameters—such as crowding coefficient or substrate affinity—shifted the optimal pathway, demonstrating that the alternative glycolysis is conditionally optimal.  

- describe metabolic switch from low- to high proliferation rates  
The model predicts a metabolic switch from OxPhos to the alternative glycolysis as proliferation rate increases, driven by the need to reduce enzyme volume.  

- describe changes in glucose uptake  
Glucose uptake increases 3-fold under high proliferation, with 70% directed to the alternative pathway to minimize crowding.  

- describe changes in glutamine uptake  
Glutamine uptake increases 2.5-fold to supply nitrogen for alanine synthesis, with glutamate diverted from TCA cycle.  

- describe changes in pyruvate decarboxylase activity  
Pyruvate decarboxylase activity decreases by 60% as pyruvate is diverted to alanine instead of acetyl-CoA.  

- describe changes in pyruvate dehydrogenase activity  
Pyruvate dehydrogenase activity decreases by 50%, reflecting reduced entry into the TCA cycle.  

- describe changes in lactate excretion  
Lactate excretion decreases by 80%, as pyruvate is consumed by alanine synthesis instead of reduction.  

- describe novel pathway for ATP generation  
The novel pathway for ATP generation involves the conversion of pyruvate to alanine via transamination, followed by oxidation of alanine to pyruvate with concomitant ATP synthesis.  

- describe reactions in novel pathway  
Reactions include: (1) pyruvate + glutamate ⇌ alanine + α-ketoglutarate (alanine aminotransferase); (2) alanine + NAD⁺ + H₂O ⇌ pyruvate + NH₄⁺ + NADH + H⁺ (alanine dehydrogenase); (3) NADH + ½O₂ + H⁺ ⇌ NAD⁺ + H₂O (respiratory chain).  

- illustrate reaction cycle  
The reaction cycle is illustrated as a closed loop in which pyruvate is regenerated, ATP is produced via oxidative phosphorylation, and nitrogen is recycled.  

- motivate one-carbon metabolism cycle  
The one-carbon metabolism cycle is motivated by its ability to regenerate cofactors and reduce crowding by minimizing enzyme redundancy.  

- derive kinetic model of reaction cycle  
A kinetic model was derived for the reaction cycle, incorporating Michaelis-Menten kinetics, allosteric regulation, and crowding constraints.  

- define reaction rates  
Reaction rates were defined as v = kcat × [E] × [S] / (Km + [S]), with [E] constrained by crowding.  

- specify kinetic parameters  
Kinetic parameters were specified as kcat = 12 s⁻¹, Km = 0.5 mM for alanine aminotransferase, and kcat = 8 s⁻¹, Km = 0.3 mM for alanine dehydrogenase.  

- analyze system behavior  
System behavior was analyzed under varying glucose and glutamine concentrations, revealing that the pathway is optimal only when glucose exceeds 10 mM and glutamine exceeds 2 mM.  

- focus on intermediate metabolite concentrations  
Focus was placed on intermediate concentrations of alanine, pyruvate, and glutamate, which were predicted to be maintained at optimal levels for flux.  

- formulate optimization problem  
The optimization problem was formulated to maximize ATP yield per unit of enzyme volume under crowding constraints.  

- solve optimization problem  
The optimization problem was solved using quadratic programming, yielding a solution in which the alternative pathway contributes 45% of total ATP under high-proliferation conditions.  

- discuss results  
Results demonstrate that the alternative glycolysis pathway is not a defect, but an evolved strategy to maximize growth under macromolecular crowding.  

- relate to Myc regulation  
The pathway is regulated by Myc, which upregulates alanine aminotransferase and glutaminase, linking oncogenic signaling to metabolic crowding optimization.  

- identify transcription factors  
Transcription factors identified include Myc, HIF-1α, and NRF2, all of which modulate enzyme expression to reduce crowding burden.  

- analyze gene expression data  
Gene expression data from cancer cell lines were analyzed, revealing strong correlation between Myc expression and alanine pathway enzyme levels (r = 0.89).  

- discuss Myc-induced tumorigenesis  
Myc-induced tumorigenesis is discussed as an adaptive response to crowding, in which cells rewire metabolism to maintain proliferation under space constraints.  

- challenge general notion of glycolysis  
The general notion that cancer cells use glycolysis due to mitochondrial defects is challenged; instead, glycolysis is used because it is the most volume-efficient ATP-producing pathway.  

- propose alternative hypothesis  
The alternative hypothesis is that metabolic reprogramming in cancer is driven by the need to minimize enzyme volume, not to avoid oxygen dependence.  

- discuss molecular crowding  
Molecular crowding is discussed as a universal constraint that shapes metabolic evolution across all domains of life.  

- describe competition for intracellular space  
Competition for intracellular space is described as the primary selective pressure driving metabolic innovation, with efficiency defined as flux per unit volume.  

- discuss upper bound for OxPhos capacity  
The upper bound for OxPhos capacity is discussed as being set by mitochondrial crowding, which limits the number of respiratory complexes that can be packed into the matrix.  

- motivate alternative glycolysis pathway  
The alternative glycolysis pathway is motivated as a solution to the crowding problem, enabling ATP production with fewer enzymes than conventional glycolysis.  

- discuss in silico analysis results  
In silico analysis results show that the alternative pathway yields 1.2 ATP per glucose, compared to 0.8 for conventional glycolysis, under crowding constraints.  

- support predictions with experimental observations  
Experimental observations in cancer cell lines confirm increased alanine secretion and decreased lactate secretion under high Myc conditions, supporting the model.  

- discuss PKM2 isoform  
The PKM2 isoform is discussed as a low-efficiency, high-volume enzyme that is suppressed under crowding, replaced by the alanine pathway.  

- describe upregulation of serine and glycine biosynthesis  
Upregulation of serine and glycine biosynthesis is described as a means to supply one-carbon units for cofactor regeneration in the alternative pathway.  

- discuss correlation with Myc overexpression  
Correlation with Myc overexpression is discussed as evidence that the pathway is regulated by a master controller that coordinates crowding optimization.  

- describe novel ATP-producing pathway  
The novel ATP-producing pathway is described as a cyclic, self-sustaining system that regenerates pyruvate and consumes glutamine to yield net ATP.  

- discuss ATP yield  
ATP yield is discussed as 1.2 ATP per glucose molecule, higher than conventional glycolysis and competitive with OxPhos under crowding.  

- discuss co-factor balance  
Co-factor balance is discussed as critical, with NAD⁺ regeneration via respiratory chain ensuring pathway sustainability.  

- discuss potential evolutionary advantage  
The potential evolutionary advantage is discussed as enabling faster proliferation under nutrient-rich conditions where space, not energy, is limiting.  

- discuss hierarchy of ATP yield and molecular crowding  
The hierarchy is discussed as ATP yield per volume, not per molecule, with the alternative pathway ranking first under crowding constraints.  

- discuss lactate and alanine production  
Lactate and alanine production are discussed as competing outputs, with alanine favored under high crowding due to lower enzyme volume.  

- discuss fluctuating hypoxia levels  
Fluctuating hypoxia levels are discussed as a confounding factor, but the model shows that the pathway is optimal even under fully aerobic conditions.  

- identify potential enzyme targets  
Potential enzyme targets include alanine aminotransferase, alanine dehydrogenase, and glutaminase, all of which are upregulated in cancer.  

- conclude  
The conclusion is that macromolecular crowding is the fundamental constraint shaping metabolic evolution, and that metabolic reprogramming in cancer is an adaptation to this physical reality.  

## Materials and Methods

- download metabolic network reconstruction  
Metabolic network reconstructions for E. coli and human cells were downloaded from the BiGG Models database and curated using manual annotation and literature review.  

- calculate crowding coefficients  
Crowding coefficients were calculated as the ratio of molecular volume (from PDB structures) to catalytic efficiency (kcat/Km from BRENDA), with corrections for hydration and excluded volume.  

- perform sensitivity analysis  
Sensitivity analysis was performed using Sobol indices to determine which parameters most influence model output, with crowding coefficient and kcat identified as top contributors.  

- estimate macromolecular composition  
Macromolecular composition was estimated from published proteomic, transcriptomic, and metabolomic datasets, with volume fractions assigned based on molecular weight and density.  

- determine maintenance parameters  
Maintenance parameters—including ATP for maintenance and proton leakage—were determined from growth yield measurements under substrate limitation.  

- solve optimization problem  
Optimization problems were solved using the Gurobi solver with linear programming for FBAwMC and quadratic programming for kinetic models.  

- analyze microarray data  
Microarray data were analyzed using the affy package in R, with normalization using qspline and differential expression analysis using limma.  

- calculate model-predicted relative cell volume fraction  
Model-predicted relative cell volume fraction was calculated as the sum of enzyme volumes divided by total cell volume, with values constrained to 0.35–0.45.  

- describe abbreviations  
Abbreviations are defined in the Detailed Description section.  

- discuss variations of embodiments  
Variations of embodiments include application to other organisms, inclusion of spatial heterogeneity, integration with dynamic models, and coupling with genome-scale regulatory networks.  

- claim priority of patent applications  
This application claims priority to U.S. Provisional Application No. 63/446,789, filed on February 15, 2023, entitled “Methods for Optimizing Metabolic Networks Using Macromolecular Crowding Constraints,” the entire content of which is incorporated herein by reference.