# DESCRIPTION

## STATEMENT REGARDING FEDERAL FUNDING

This invention was made with government support under Grant No. [REDACTED] awarded by the National Institutes of Health. The government has certain rights in the invention.

The United States Government retains and the inventor hereby grants to the Government a nonexclusive, irrevocable, worldwide license to practice or have practiced the claimed invention throughout the world for or on behalf of the United States Government. This license includes the right to use, manufacture, and distribute the invention for governmental purposes, including but not limited to research, defense, and public health applications. Additionally, the Government reserves the right to require the patentee to license others upon reasonable terms if the invention is not reasonably available to the public or if action is necessary to meet public health or safety needs.

## SUMMARY

Systems biology seeks to understand the complex interactions within biological systems by integrating computational modeling with experimental data. A central challenge in this field is predicting how cells allocate their limited intracellular resources to achieve optimal growth and metabolic function under varying environmental conditions. Traditional approaches such as flux balance analysis (FBA) have provided valuable insights into cellular metabolism by assuming steady-state conditions and optimizing for objectives like biomass production. However, conventional FBA fails to account for critical biophysical constraints that govern enzyme allocation and reaction kinetics within the crowded cytoplasmic environment of living cells.

Flux balance analysis (FBA) is a constraint-based modeling technique that uses stoichiometric representations of metabolic networks to predict metabolic flux distributions under steady-state assumptions. While powerful, standard FBA neglects the finite capacity of the cell to accommodate enzymes and other macromolecules—a phenomenon known as macromolecular crowding (MC). This limitation prevents FBA from accurately capturing regulatory phenomena such as carbon catabolite repression (CCR), where cells preferentially utilize certain substrates over others in mixed-nutrient environments.

To address these shortcomings, the present invention introduces an advanced optimization method that integrates molecular crowding constraints into flux balance modeling. This method, termed flux balance analysis with molecular crowding (FBAwMC), incorporates the physical limit on total enzyme occupancy within the cytoplasm, thereby enabling more accurate predictions of metabolic behavior, substrate utilization hierarchies, and growth dynamics. By explicitly modeling the trade-off between enzyme concentration and catalytic efficiency, the invention provides a mechanistic explanation for CCR and other resource-allocation strategies employed by rapidly proliferating cells.

The application of this optimization method enables the calculation of key physiological parameters, including cell growth rates, substrate utilization profiles, and metabolic flux reorganizations across different nutrient conditions. It further allows for the determination of maximum metabolic rates achievable under given crowding constraints, as well as the computation of optimal metabolite concentrations and enzyme activities required to sustain those rates. These capabilities are implemented through computer-implemented methods that leverage genome-scale metabolic reconstructions and experimentally derived kinetic parameters.

Computer-implemented methods according to the invention execute algorithms that solve constrained optimization problems incorporating both stoichiometric and crowding constraints. An apparatus for implementing these methods includes a processor, memory, and input/output interfaces configured to receive biological data, perform calculations, and output optimized culture parameters or genetic designs. The invention also encompasses methods for optimizing biological activities by iteratively adjusting biochemical reaction networks and evaluating their performance under simulated or real-world conditions.

In particular, the invention enables the calculation of optimal cell culture parameters—such as dilution rates, nutrient compositions, and oxygen levels—that maximize desired outputs like biomass yield or product formation. These parameters can be initiated and maintained in bioreactors using automated control systems linked to the computational model. Furthermore, the method calculates the order of substrate usage in mixed-nutrient environments and provides mechanisms to control or manipulate this sequence through genetic or environmental interventions.

The invention also facilitates the calculation of maximum metabolic rates, optimal metabolite concentrations, and enzyme activities necessary to achieve peak performance of a biochemical reaction network. By achieving optimal function of such networks, the method supports the design of engineered microbial strains with enhanced productivity or novel metabolic capabilities. Alterations to the biochemical reaction network—through gene knockouts, insertions, or expression tuning—are followed by repeated calculation of optimal properties until a desired performance metric is met.

Cells cultured under these optimal conditions exhibit improved growth characteristics and metabolic efficiency. The invention further includes methods for constructing the genetic makeup of cells to encode specific biochemical reactions tailored to the intended application. Once genetically modified, cells are placed in culture and cultivated under specified environmental conditions to allow adaptive evolution toward the desired optimal function. A computer-readable medium storing instructions for performing these methods ensures reproducibility and scalability across diverse biological systems.

## DETAILED DESCRIPTION

For the purposes of this patent, several terms are defined as follows: “biochemical reaction network” refers to a set of interconnected enzymatic and transport reactions that constitute a cell’s metabolic, signaling, or regulatory pathways; “macromolecular crowding” denotes the physical constraint imposed by the high concentration of proteins, RNA, and other macromolecules in the cytoplasm, which limits the available volume for additional molecular components; “flux balance analysis” (FBA) is a mathematical approach that computes steady-state metabolic fluxes based on mass conservation and optimization principles; and “optimal function” signifies the achievement of a predefined biological objective—such as maximal growth rate or product yield—under given constraints.

Flux balance calculations for cell cultures traditionally rely on stoichiometric models that assume instantaneous equilibration of metabolites and unlimited enzyme availability. However, in reality, the cytoplasm is densely packed, and the synthesis of each enzyme consumes precious biosynthetic resources. The present invention improves upon classical FBA by introducing a constraint that accounts for cytoplasmic molecular crowding. This constraint limits the sum of enzyme concentrations weighted by their respective molar volumes, reflecting the finite solvent capacity of the cell.

Reaction kinetics parameters, such as turnover numbers (kcat) and Michaelis constants (Km), are incorporated into the model to relate enzyme concentrations to reaction rates. Specifically, the flux through each reaction is bounded by the product of its enzyme concentration and its effective turnover rate, adjusted for substrate availability. This integration of kinetic information enhances the predictive power of the model beyond what is possible with purely stoichiometric approaches.

Applications of this enhanced flux balance analysis include the rational design of microbial cell factories, prediction of metabolic shifts during adaptive laboratory evolution, and elucidation of regulatory mechanisms underlying substrate preference in natural isolates. The flux balance model of cellular metabolism begins with a genome-scale reconstruction that maps annotated genes to enzymatic functions and defines the corresponding stoichiometric matrix S, where rows represent metabolites and columns represent reactions.

Under the steady-state assumption, the time derivative of metabolite concentrations is zero, leading to the fundamental equation S·v = 0, where v is the vector of metabolic fluxes. In addition to this mass balance constraint, the invention imposes an enzyme concentration constraint: Σ(ci · αi) ≤ Vc, where ci is the concentration of enzyme i, αi is its crowding coefficient (representing its effective molar volume per unit activity), and Vc is the total available cytoplasmic volume for metabolic enzymes. The crowding coefficient αi is derived from experimental data or estimated from protein structural properties and turnover rates.

The optimization method proceeds by maximizing a biological objective—typically biomass production rate—subject to the stoichiometric and crowding constraints. This yields a unique flux distribution that reflects the optimal allocation of enzymatic resources under the given environmental conditions. From this solution, optimal cell culture parameters such as nutrient feed rates, pH, and temperature can be inferred and subsequently implemented in bioreactor systems.

Initiation and maintenance of these optimal parameters are achieved through feedback-controlled bioprocesses that monitor real-time culture metrics and adjust inputs accordingly. For example, in a continuous chemostat culture, the dilution rate can be tuned to match the predicted optimal growth rate, while substrate feeds are modulated to maintain desired uptake ratios. An exemplary implementation involves growing Escherichia coli in a mixed-substrate medium containing glucose, glycerol, galactose, lactate, and maltose, where the model correctly predicts the sequential utilization of substrates due to crowding-limited enzyme allocation.

The use of a computer for implementing the method is essential, as the optimization problem involves thousands of variables and constraints. The steps for determining optimal functions of a biochemical reaction network include: (1) reconstructing the network from genomic and biochemical data; (2) defining the stoichiometric matrix and associated constraints; (3) assigning crowding coefficients and kinetic parameters; (4) formulating the optimization objective; (5) solving the constrained optimization problem; and (6) interpreting the results in terms of biological performance.

Biochemical reactions are represented in the computer as a directed graph or sparse matrix, enabling efficient computation. Optimization methods employed include linear programming for cases where the objective and constraints are linear, and nonlinear programming or heuristic algorithms (e.g., simulated annealing) when kinetic dependencies introduce nonlinearity. Alteration of the reaction list—through deletion, addition, or modification of reactions—is followed by re-computation of optimal properties to assess the impact of genetic changes.

This process is repeated iteratively until the desired performance is achieved, whether that be maximal growth, minimal byproduct formation, or high titer of a target compound. Culturing of living cells under these optimized conditions allows empirical validation of model predictions and facilitates adaptive evolution toward the theoretical optimum. Construction of the genetic makeup of the cell may involve CRISPR-Cas9 editing, plasmid-based expression, or transposon mutagenesis to implement the designed network modifications.

Once placed in culture, cells are cultivated under specified environmental conditions that promote the emergence of the desired phenotype. Over time, spontaneous mutations or directed evolution protocols lead to strains that better approximate the predicted optimal state. The biochemical reaction network is characterized through genome sequencing, transcriptomics, and metabolomics to confirm the presence of intended modifications and identify compensatory adaptations.

Reconstruction of the biochemical reaction network relies on annotated genome sequences, biochemical databases (e.g., KEGG, MetaCyc), and physiological data from literature or experiments. Analysis of the reconstructed network includes topological assessment, gap-filling, and consistency checks to ensure biological plausibility. Determination of optimal properties involves solving the constrained optimization problem using solvers such as Gurobi, CPLEX, or COBRA Toolbox.

Linear and nonlinear optimization techniques with linear constraints are applied depending on the nature of the objective function. Simulated annealing may be used for global optimization when local minima are problematic. Reconstruction of the metabolic network specifically focuses on central carbon metabolism, energy generation, and biosynthetic pathways relevant to the application.

Flux balance analysis is then applied to assess the metabolic capabilities of the reconstructed network, including growth yields, substrate scopes, and knockout lethality. Experimentally determined strain-specific parameters—such as maintenance ATP requirements, membrane transporter efficiencies, and ribosomal allocation—are integrated to improve model accuracy. Calculation of flux distribution through the network reveals bottlenecks and redundancies that inform engineering strategies.

Factors leading to a closed solution space—such as insufficient degrees of freedom or conflicting constraints—are identified and addressed by relaxing bounds or adding auxiliary reactions. The optimization procedure compares calculated behaviors to experimental data, refining parameters until agreement is achieved. Additional constraints relating to cytoplasmic molecular crowding and/or reaction kinetics are added to enhance physiological realism.

The model predicts optimal uses of the biochemical reaction network, revealing why natural organisms often fail to achieve theoretical maxima due to evolutionary trade-offs and lack of selection pressure in non-competitive environments. Design of synthetic networks circumvents these limitations by directly encoding optimality criteria. In silico methods resolve optimality issues by simulating perturbations before wet-lab implementation.

Culturing methods address growth competition and selection issues by maintaining monocultures or using selective pressures to enrich desired phenotypes. Altering cellular parameters—such as promoter strength, ribosome binding site efficiency, or codon usage—enables fine-tuning of expression levels to match predicted optima. The iterative design procedure continues until the desired performance, defined as either a qualitative characteristic (e.g., substrate co-utilization) or quantitative value (e.g., 90% theoretical yield), is attained.

The optimization method is executed on a computer system comprising a central processing unit, memory, storage, and user interface. Database information includes curated biochemical reaction networks, biomolecular sequences (DNA, RNA, protein), genomic sequences, and functional annotations. External databases such as GenBank, UniProt, and BRENDA are accessed to populate internal repositories.

The user interface receives selections of target organisms, desired products, and performance metrics, then outputs optimal genetic designs and culture conditions. A computer program product stored on a non-transitory medium contains code for executing the described processes. Modules handle database interaction, network reconstruction, constraint definition, optimization, and result visualization.

Interacting with the database allows querying of homologous genes, pathway completeness, and enzyme kinetics. Comparing biochemical reaction networks identifies differences between wild-type and engineered strains. Data from cell culture—such as optical density, substrate concentrations, and byproduct levels—are fed back into the system to update model parameters and control external devices like pumps and sensors.

Computer-readable program code is generated for deployment on embedded systems controlling bioreactors. Adaptive evolution of cultured strains is accelerated using chemical mutagens or radiation, with periodic sampling to track progress. Virtually any cell type—bacterial, yeast, mammalian—can be modeled using the same framework, provided sufficient genomic and physiological data are available.

Characterization of the biochemical reaction network includes measurement of enzyme activities, metabolite pools, and fluxes via isotopic labeling. Genome sequencing and gene identification confirm the presence of intended modifications. Genetic makeup is constructed to contain only the necessary reactions for the desired performance, minimizing unnecessary metabolic burden.

Reactions are added or subtracted using genetic manipulations such as gene knockouts, integrations, or CRISPR interference. Expression of regulatory components—transcription factors, sRNAs, allosteric effectors—is altered to reshape flux distributions. Cells are placed in culture under specified environmental conditions that favor the emergence of the optimal phenotype.

Optimal cultural parameters are determined using the optimization procedure and monitored continuously. Adjustments are made automatically via computerized control systems that regulate media inflow, gas exchange, and temperature. Continuous culture modes like chemostats or turbidostats maintain cells in steady-state conditions ideal for model validation.

Sensors measure dissolved oxygen, pH, and metabolite levels, feeding data to the control system. Reservoirs store fresh media and waste collection vessels. Mechanisms for dispensing media and taking samples enable automated operation. Analytical devices such as HPLC, GC-MS, and flow cytometers provide real-time feedback.

Display subsystems present culture status and model predictions to users. Communication subsystems allow remote access to reaction parameters and control settings. The computer-implemented method achieves optimal function by repeatedly calculating and implementing improvements to the biochemical reaction network.

Maximum metabolic rates, optimal metabolite concentrations, and enzyme activities are computed using kinetic models integrated with crowding constraints. Genetic makeup is constructed to encode the necessary reactions, and cells are cultivated until they evolve toward the desired optimal function. Ribosome density is accounted for as a proxy for translational capacity and non-metabolic protein load.

Mitochondria and other subcellular compartments are explicitly modeled in eukaryotic cells, with separate crowding constraints for each organelle. A computer-readable medium stores instructions for implementing the computer model, and a device comprising this medium and a processor executes the instructions. Additional components include bioreactor hardware, sensors, and actuators.

Culture vessels, heating/cooling elements, and reservoirs for media storage are integrated into the system. Dispensing mechanisms add fresh media or inducers, while sampling systems extract aliquots for analysis. Displays, analytical devices, and communication subsystems complete the platform, enabling fully automated, model-driven bioprocessing.

## EXAMPLES

### Abbreviations

CCR: Carbon catabolite repression; FBAwMC: Flux balance analysis with molecular crowding; MC: Macromolecular crowding; OxPhos: Oxidative phosphorylation; PTS: Phosphotransferase system; kcat: Turnover number; Km: Michaelis constant; OD600: Optical density at 600 nm; GFP: Green fluorescent protein; TCA: Tricarboxylic acid cycle.

### Example 1

The impact of limited solvent capacity on E. coli cell metabolism was studied by implementing the FBAwMC framework. The relevance of the crowding constraint was demonstrated particularly for fast-growing cells, where enzyme allocation becomes limiting. The model successfully predicted a metabolic switch between low and high nutrient abundance regimes, characterized by a shift from full oxidative phosphorylation to mixed metabolism with acetate excretion. Flux measurements of central carbon metabolism reactions showed partial agreement with model predictions, validating the core hypothesis.

Gene expression and enzyme activity measurements revealed that the metabolic switch is primarily controlled at the enzyme activity level rather than transcriptional regulation. This finding aligns with the model’s emphasis on post-translational resource allocation. The potential relevance of these observations extends to other organisms, suggesting a universal role for macromolecular crowding in shaping metabolic strategies.

Crowding coefficients for E. coli proteins were estimated using data from the BRENDA database, which provides experimentally measured turnover rates. Enzymes' turnover rates were obtained from BRENDA, and crowding coefficients were calculated as the inverse of kcat normalized by protein density. The FBAwMC model was implemented by solving an optimization problem that maximizes biomass production rate subject to stoichiometric and crowding constraints.

Crowding coefficients were modeled as noise to account for uncertainty in parameter estimates, with random values drawn from a gamma distribution. Predictions were made for E. coli metabolic fluxes on different carbon sources, including glucose, glycerol, galactose, lactate, and maltose. The model simulated increasing carbon source concentration in the growth medium and computed the resulting fluxes that maximize biomass production rate.

Analysis of metabolic fluxes as a function of growth rate revealed a clear transition point above which acetate excretion becomes favorable. The bacterial strain used was E. coli MG1655, grown in M9 minimal medium supplemented with individual or mixed carbon sources. Biomass samples were harvested during exponential growth for flux measurements using isotopic tracers.

Metabolic enzyme activity assays were performed spectrophotometrically, with total protein concentration determined by Bradford’s assay. Enzyme activity units were defined as micromoles of substrate converted per minute per milligram of total protein. Flux measurement and analysis utilized a GC-MS and NMR metabolome mapping platform, with mass isotopomer distribution analysis (MIDA) to quantify labeling patterns.

Statistical analysis employed Student’s t-test to compare groups. Stable isotope studies tracked glycogen glucose and RNA ribose incorporation. Cellular RNA was subjected to acid hydrolysis, and ribose and glycogen glucose were derivatized for mass spectral analysis. Lactate, glutamate, and fatty acids were extracted, derivatized, and analyzed by GC-MS under standardized conditions.

RNA preparation for microarray analysis followed standard protocols, with STEM clustering used to identify co-regulated gene sets. Querying expression data revealed specific profiles for operons in central carbon metabolism. Results confirmed that limited solvent capacity constrains metabolic rate, particularly at high growth rates.

Crowding coefficients were estimated from experimental reports and computed for E. coli enzymes. The FBAwMC model predicted a change in the effective metabolic efficiency objective as growth rate increased, consistent with the observed metabolic switch. Evaluation of the solvent capacity constraint at physiological growth conditions supported its biological relevance.

The model predicted redistribution of metabolic fluxes and excretion of acetate at high growth rates. Comparison with experimental values showed good agreement for major pathways, though some discrepancies remained. Regulatory mechanisms controlling the metabolic switch were investigated by measuring in vitro enzyme activities and correlating them with flux rates.

mRNA levels of enzyme-encoding genes were analyzed, but no strong correlation was found with measured fluxes, indicating post-transcriptional control. The discussion emphasized the significance of solvent capacity as a physicochemical constraint influencing cellular metabolism. Incorporation of this constraint into FBA modeling represents a major advance in systems biology.

### Example 2

A modified FBA model incorporating solvent capacity constraint was developed and tested. The model predicted maximum growth rates that agreed well with experimental data, supporting the hypothesis that macromolecular crowding is a key constraint. The FBAwMC modeling framework was implemented by defining an optimization problem that maximizes growth rate subject to enzyme occupancy limits.

Crowding coefficients were modeled as random variables drawn from a gamma distribution with shape parameter β=3. Sensitivity analysis confirmed robustness of results to parameter variation. Maximum growth rates were obtained for each carbon source, and average crowding coefficients were fitted to match experimental data.

The model successfully predicted the temporal order of substrate uptake in mixed cultures, considering initial substrate concentrations and integrating differential equations describing consumption dynamics. Three FBAwMC problems were solved sequentially to simulate diauxic growth. Crowding coefficients were estimated from experimental measurements of enzyme turnover rates.

Growth experiments used M9 minimal medium with single or mixed carbon sources. Transcriptome states were assessed via microarray analysis. Maximum growth rates were determined from OD600 measurements in continuous culture. Residual carbon source concentrations were quantified to validate uptake predictions.

Microarray data analysis identified genes with sequence-specific hybridization signals. Hierarchical clustering and probabilistic methods based on hidden Markov models revealed expression patterns correlated with metabolic phases. Results showed that FBAwMC predicts relative maximum growth rates and substrate hierarchy utilization.

Surrogate markers of cellular metabolism, such as acetate secretion and oxygen consumption, aligned with model predictions. The mode and sequence of substrate utilization correlated with gene expression changes, particularly in stress response programs activated upon metabolic switching. Principal component analysis captured major sources of variation in transcriptome data.

Two discrepancies were noted: higher-than-predicted acetate secretion and earlier uptake of secondary substrates. These may reflect contributions from non-metabolic proteins or incomplete knowledge of regulatory interactions. Nevertheless, the model captured the essential features of CCR, demonstrating that maximum enzyme concentration is a key determinant of substrate utilization hierarchy.

### Example 3

Flux balance analysis in S. cerevisiae was extended to include molecular crowding and kinetic modeling. The hypothesis of optimal intracellular resource use was tested using a kinetic model of glycolysis. Rate equations for each glycolytic reaction were derived based on Michaelis-Menten kinetics, with parameters obtained from experimental estimates.

Cell density and specific volume were measured to constrain the crowding model. Global optimization of metabolite concentrations yielded predictions for optimal intermediate levels and enzyme activities. Parameter sensitivity analysis identified key control points in the pathway.

The limited solvent capacity constraint was introduced by bounding the total enzyme volume fraction. Reaction rates were expressed as functions of enzyme concentration and metabolite levels, with crowding coefficients defining the trade-off between enzyme amount and activity. Application to S. cerevisiae glycolysis revealed dependencies of glycolysis rate on metabolite concentrations.

Predicted optimal metabolite concentrations and enzyme activities were compared with experimental values, showing reasonable agreement despite simplifications in the model. Alternative optimization objectives were explored, including ATP yield and redox balance. The discussion highlighted the advantages of incorporating physical constraints into metabolic models, while acknowledging limitations such as the steady-state approximation.

### Example 4

An alternative glycolysis pathway was proposed in the context of human cell metabolism, motivated by the Warburg effect observed in cancer cells. A genome-scale model incorporating molecular crowding constraints was formulated, with separate compartments for cytosol and mitochondria. Nutrient import reactions, mitochondrial reactions, and cytosolic reactions were defined with associated volume fraction constraints.

The optimization problem included flux balance constraints, minimum/maximum flux bounds, and molecular crowding limits. Model parameters were estimated from literature, including enzyme crowding coefficients, ribosome density, and mitochondrial volume. Protein synthesis and degradation were modeled to account for effective protein dilution.

Simulations predicted a metabolic switch from low- to high-proliferation states, characterized by increased glucose uptake, lactate excretion, and altered pyruvate metabolism. A novel ATP-producing pathway involving one-carbon metabolism was identified, with potential implications for Myc-regulated tumorigenesis. Gene expression data supported the upregulation of serine and glycine biosynthesis in Myc-overexpressing cells.

The model challenged the conventional view of glycolysis as inefficient, proposing instead that it optimizes ATP production under crowding constraints. Competition for intracellular space was identified as a key driver of metabolic reprogramming. Potential enzyme targets for therapeutic intervention were suggested based on model predictions.

## Materials and Methods

The metabolic network reconstruction for E. coli was downloaded from the BiGG database. Crowding coefficients were calculated from BRENDA-derived turnover rates using the formula αi = 1 / (kcat,i · ρi), where ρi is the protein density. Sensitivity analysis was performed by varying parameters ±20% and assessing impact on predictions.

Macromolecular composition was estimated from literature values for protein, RNA, and lipid content. Maintenance parameters, such as ATP hydrolysis rate, were set based on experimental measurements. The optimization problem was solved using the COBRA Toolbox in MATLAB with the Gurobi solver.

Microarray data were processed using R/Bioconductor packages, with background correction and normalization as described. Model-predicted relative cell volume fractions were calculated from enzyme concentrations and crowding coefficients. Abbreviations used throughout are defined in the main text.

Variations of the disclosed embodiments include application to other organisms, incorporation of additional constraints (e.g., membrane area, redox balance), and use of alternative optimization algorithms. This patent application claims priority to U.S. Provisional Patent Application No. 62/XXXXXX, filed on [Date], the contents of which are incorporated herein by reference.