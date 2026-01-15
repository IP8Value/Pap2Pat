# DESCRIPTION

## FIELD OF THE INVENTION

- define field of invention

The present invention resides in the field of computational structural design and optimization, specifically in the domain of topology optimization for three-dimensional structural continua under deterministic and probabilistic failure conditions. It pertains to a computer-implemented method and system for generating structural designs that maintain structural integrity and load-bearing capacity following the occurrence of localized material loss, whether due to manufacturing defects, impact damage, fatigue-induced voids, or other forms of partial material failure. The invention integrates principles from finite element analysis, parallel computing, and reliability-based design to enable the synthesis of fail-safe architectures that satisfy stringent safety criteria without compromising mechanical efficiency. The method is particularly suited for applications in aerospace, automotive, and heavy machinery industries where catastrophic structural failure must be prevented even under extreme and unpredictable damage scenarios.

## BACKGROUND

- introduce topology optimization

Topology optimization is a computational methodology that determines the optimal distribution of material within a given design domain to achieve a specified performance objective, such as minimizing compliance or maximizing stiffness under prescribed loading conditions. Unlike traditional sizing or shape optimization, topology optimization permits the creation of complex, non-intuitive geometries by allowing material to be added or removed at the elemental level across the entire design space. This approach has revolutionized structural design by enabling the discovery of highly efficient load paths that often resemble natural biological structures or idealized trusses such as the Michell truss. The method relies on iterative numerical procedures that adjust material density variables at discrete finite element locations, guided by sensitivity analysis and constraint enforcement, to converge toward an optimal configuration.

- discuss reliability based design optimization

Reliability-based design optimization extends classical topology optimization by incorporating uncertainties in loading, material properties, boundary conditions, and geometric imperfections into the formulation. Rather than optimizing for a single nominal condition, reliability-based approaches seek designs that maintain acceptable performance across a range of possible realizations of these uncertainties. This is typically achieved through probabilistic modeling, where constraints are defined in terms of failure probabilities or reliability indices. While such methods have been successfully applied to problems involving stochastic loads or material variability, they are inherently limited in their ability to address discrete, deterministic failure modes such as the complete loss of a structural member or the formation of a localized void of significant size. These failure modes are not well captured by statistical distributions and require explicit, deterministic testing of structural integrity under specific damage conditions.

- motivate failsafe design

Fail-safe design represents a critical engineering philosophy, especially in safety-critical systems such as aircraft, bridges, and pressure vessels, where the consequences of structural failure are catastrophic. A fail-safe structure is one that retains sufficient load-carrying capacity even after the loss of one or more components, ensuring that failure remains non-catastrophic and that safe landing, controlled descent, or controlled shutdown remains possible. In aerospace applications, for instance, regulatory standards require that structures support at least 80–100% of limit loads following the failure of any single discrete member or the formation of a partial material void of a defined size. Traditional topology optimization, however, tends to produce highly efficient but minimally redundant designs that are vulnerable to such localized failures. The absence of a computationally tractable method to enforce fail-safe constraints during the optimization process has long been a barrier to integrating safety-driven design into automated structural synthesis. This invention addresses this gap by introducing a rigorous, scalable, and practical computational framework that embeds fail-safe requirements directly into the topology optimization process.

## SUMMARY

- introduce failsafe design challenge

The fundamental challenge in implementing fail-safe design within topology optimization lies in defining a mathematically consistent and computationally feasible failure test that applies to structures whose load-carrying members do not yet exist at the outset of the optimization. Unlike truss structures where members are predefined and failure can be tested by sequentially removing each member, continuum-based topologies generate material layouts dynamically, rendering traditional member-based failure tests inapplicable. The challenge, therefore, is to simulate the effect of an arbitrary, localized loss of material—whether as a spherical or cubic void—occurring at any location within the design domain, and to optimize the structure such that its performance remains acceptable under the worst-case scenario among all such possible damage events.

- introduce computational scheme

To address this challenge, the invention introduces a novel computational scheme that replaces the infinite continuum of possible damage locations with a finite, systematically generated population of damage volumes. These damage volumes are placed according to a hierarchical series of spatial distributions, each doubling the density of the prior level, thereby progressively refining the coverage of the design domain. By analyzing the structural response under each damage instance in this series, the method identifies the most critical damage configurations and uses them to guide the optimization toward designs with inherent redundancy. This approach transforms an intractable infinite-dimensional problem into a finite, parallelizable optimization task that can be executed on high-performance computing platforms.

- define failsafe design problem

The fail-safe design problem is formally defined as the minimization of the maximum compliance across all structural configurations resulting from the removal of a damage volume of specified size and shape, placed at any location within the design domain. The objective is to identify a material distribution that, even after the most detrimental damage event, retains sufficient stiffness to carry the applied loads without collapse. The problem is subject to a volume constraint that limits the total amount of material available and may include additional constraints on displacement, stress, or manufacturing feasibility. The damage volume is modeled as a geometric void—either spherical or cubic—that completely removes material within its boundaries, simulating the effect of a localized fracture, impact, or defect.

- introduce finite damage population

To render the problem computationally tractable, the invention introduces the concept of a finite damage population, wherein damage volumes are placed at discrete, predetermined locations rather than continuously throughout the domain. The damage population is constructed as a series of levels, beginning with a base level in which damage cubes fill the domain without gaps, and progressing to higher levels in which the density of damage placement doubles. This hierarchical construction ensures that the probability of a randomly occurring damage event being unaccounted for diminishes rapidly with each successive level, allowing the designer to select a level that balances computational cost with modeling fidelity.

- introduce simple formulae for material survival rate

The invention further introduces exact analytical formulae for calculating the maximum material survival rate within a representative structural section under a given damage population level. These formulae relate the geometry of intersecting damage cubes to the residual cross-sectional area of a structural member aligned with the damage orientation. For example, under the base damage population (Level 1), the maximum sectional survival rate for a member of size equal to the damage cube edge length is 75%, while under the second level (Level 2), this rate decreases to 43.75% for a full population and 50% for a partial population. These formulae provide a quantitative measure of the damage modeling accuracy and enable the designer to select the minimal damage population necessary to achieve a desired level of structural redundancy.

- introduce efficient computational scheme

The invention employs an efficient computational scheme that leverages the independence of each damage scenario to enable full parallelization of finite element analyses. Each damage configuration is analyzed independently, and the resulting compliances and sensitivities are aggregated by a master process to update the design variables. This approach eliminates the need for sequential evaluation and enables the solution of problems involving hundreds of damage instances in a fraction of the time required by traditional methods. The scheme is implemented using the Message Passing Interface (MPI) standard, allowing seamless scaling across thousands of computing cores.

- discuss MPI parallel implementation

The Message Passing Interface (MPI) parallel implementation enables the distribution of damage analysis tasks across multiple processors or computing nodes, with each node responsible for evaluating the structural response under one or more damage configurations. The master process coordinates the optimization loop, collects compliance and sensitivity data from all worker processes, and updates the design variables using a gradient-based optimizer. This architecture ensures near-linear speedup with increasing numbers of processors, making it feasible to solve large-scale 3D problems on commercially available high-performance computing clusters.

- introduce automatic damage cube placement

The invention includes an automated algorithm for generating damage cube placements within the design domain, ensuring complete coverage according to the selected damage population level. The algorithm respects domain boundaries, excludes damage volumes that intersect with point loads to preserve load integrity, and applies material volume thresholds to eliminate negligible damage instances. This automation eliminates manual intervention and ensures reproducibility across different design problems.

- discuss HPC resources

The method is designed to exploit modern high-performance computing (HPC) resources, including distributed memory systems with thousands of cores and high-bandwidth interconnects. The parallel nature of the algorithm allows it to scale efficiently on such systems, reducing solution times from weeks to hours for problems involving tens of thousands of elements and hundreds of damage instances. The use of HPC resources is not merely beneficial but essential for achieving practical turnaround times in industrial applications.

- introduce Level-1 damage population

The Level-1 damage population consists of a gapless, non-overlapping arrangement of damage cubes that completely tile the design domain. This configuration ensures that no region of the structure is left unprotected and provides the foundational level of redundancy. The Level-1 population is sufficient for many practical applications and serves as the baseline for higher-level damage series.

- introduce Level-2 damage population

The Level-2 damage population doubles the density of the Level-1 population by introducing a secondary layer of damage cubes centered at the junctions of the Level-1 cubes. This enrichment captures the previously untested “best hideout” locations where material might otherwise survive a single damage event. The Level-2 population significantly improves the fidelity of the damage model while requiring only twice the computational effort of Level-1, making it a cost-effective enhancement for critical applications.

- introduce partial damage population

The invention introduces a partial damage population, termed PB, which includes only the Level-2 enrichment layer without the full Level-1 base. This partial population retains the benefits of Level-2 coverage while reducing the total number of damage instances by a factor of four compared to the full Level-2 population. The PB population is shown to yield nearly identical fail-safe performance as the full Level-2 population, offering a superior trade-off between computational cost and design reliability.

- discuss added damage layers

Added damage layers are introduced in a systematic, geometrically consistent manner to ensure that each new layer targets previously unexposed regions of the design domain. The placement of each layer is determined by the spatial symmetry of the prior layer, ensuring uniform coverage and eliminating gaps. This layered approach enables incremental refinement of the damage model without requiring re-initialization of the entire optimization process.

- introduce computer-implemented method

The invention is embodied as a computer-implemented method executed by a digital processing system, wherein the method comprises the steps of defining a design domain, discretizing the domain into finite elements, specifying a damage volume size and shape, generating a finite damage population, performing finite element analysis for each damage configuration, aggregating compliance and sensitivity data, updating design variables, and iterating until convergence. The method is implemented as a software module integrated into commercial topology optimization platforms.

- define structural continuum

The structural continuum refers to the three-dimensional domain of material that is subject to topology optimization, where material density varies continuously across the domain and is represented by a discrete set of finite elements. The continuum is not assumed to contain predefined structural members; instead, load paths emerge dynamically through the optimization process as a function of the applied loads, constraints, and damage conditions.

- define damage volume

The damage volume is a geometric region of specified size and shape—spherical or cubic—that is removed from the structural continuum to simulate the effect of a localized material loss. The volume is defined by its dimensions and orientation, and its placement is determined by the damage population series. The damage volume completely eliminates material within its bounds, rendering any element intersecting it non-load-bearing for the purpose of analysis.

- introduce computational optimization

Computational optimization is performed using the Solid Isotropic Material with Penalization (SIMP) approach, wherein material density variables are penalized with a power law to encourage binary (0 or 1) solutions. The optimization iteratively adjusts these variables to minimize the maximum compliance across all damage scenarios, subject to a volume constraint and other design requirements. Sensitivity analysis is conducted using the adjoint method to compute the gradient of compliance with respect to each design variable.

- discuss finite damage population

The finite damage population is a discrete set of damage volume placements that approximates the infinite set of possible damage locations. The population is constructed hierarchically, with each level doubling the density of the prior level. The choice of population level determines the accuracy of the fail-safe model: higher levels provide greater coverage but at increased computational cost. The invention demonstrates that Level-1 or partial Level-2 populations are sufficient for most practical applications.

- introduce analysis models

Analysis models are finite element representations of the structural continuum under each damage configuration. Each model is identical except for the removal of elements intersecting the damage volume. The models are solved independently to compute compliance, displacement, and stress responses. The set of all such models constitutes the ensemble of failure scenarios used to guide the optimization.

- discuss computational structural analysis

Computational structural analysis is performed using linear static finite element analysis under prescribed loading conditions. The stiffness matrix for each damage configuration is modified by zeroing the stiffness contributions of elements within the damage volume. The resulting system of equations is solved to determine nodal displacements and element stresses. Sensitivities are computed using the adjoint method to determine how changes in material density affect compliance.

- introduce system

The system comprises a computing platform equipped with a central processing unit, memory, and storage, running a software application that implements the fail-safe topology optimization method. The system includes input modules for defining the design domain, loads, constraints, and damage parameters; a solver module for performing parallel finite element analyses; and an optimization module for updating design variables. The system outputs optimized structural geometries and performance metrics.

- introduce non-transitory computer readable medium

The invention further includes a non-transitory computer-readable medium storing program instructions that, when executed by a processor, cause the processor to perform the steps of the computer-implemented method. The medium may include hard drives, solid-state drives, optical discs, or other persistent storage devices, and the instructions may be distributed as software packages or cloud-based services.

## DETAILED DESCRIPTION

- introduce failsafe concept

The fail-safe concept underpinning this invention is rooted in the engineering principle that a structure must remain functional and non-catastrophic even after the loss of a critical component or the formation of a significant material void. Unlike traditional optimization that seeks maximum efficiency, fail-safe design prioritizes robustness and redundancy. The invention operationalizes this concept by embedding the requirement for post-damage survivability directly into the optimization loop, ensuring that every candidate design is evaluated not only under nominal conditions but under a comprehensive set of failure scenarios.

- define failure test for topology optimization

The failure test for topology optimization is defined as the removal of a damage volume of specified size and shape at every location within the design domain, with the structure’s performance assessed under each such condition. The worst-case compliance across all damage instances becomes the objective function to be minimized. This test is applied prior to the emergence of discrete structural members, ensuring that redundancy is not an afterthought but a fundamental design criterion.

- describe spherical damage model

The spherical damage model represents a three-dimensional void of uniform radius, simulating a crack or impact zone of isotropic extent. While mathematically ideal, the spherical model is computationally complex due to its curved boundaries and irregular intersection patterns with finite elements. For practical implementation, the spherical model is approximated by a cube-shaped damage volume with equivalent volume, which simplifies meshing and analysis while maintaining conservative behavior.

- illustrate damage model with FIG. 1A

FIG. 1A illustrates a three-dimensional structural domain with a spherical damage volume centered at an arbitrary location within the domain. The sphere is shown intersecting multiple finite elements, with those fully or partially enclosed by the sphere marked for material removal. The figure demonstrates the isotropic nature of the damage and its ability to affect structural members regardless of orientation.

- define damage volume

The damage volume is a geometric entity defined by its shape (cube or sphere), size (edge length or diameter), and spatial placement. It represents the complete absence of material within its boundaries and is used to simulate the effect of a localized structural defect. The volume is not a physical entity but a computational construct used to test the resilience of the design.

- describe effect of damage volume

The effect of the damage volume is to eliminate the load-bearing capacity of any finite element intersecting its boundaries. This results in a redistribution of stresses and strains throughout the structure, potentially leading to increased compliance or localized stress concentrations. The magnitude of this effect depends on the location, size, and orientation of the damage relative to the load paths in the structure.

- formulate topology optimization problem

The topology optimization problem is formulated as a minimization of the maximum compliance over all damage configurations, subject to a volume constraint and any additional design requirements. The objective function is expressed as the supremum of compliances across the finite damage population, and the design variables are the material densities of the finite elements.

- define objective function

The objective function is defined as the maximum compliance value among all structural configurations resulting from the removal of each damage volume in the population. This ensures that the optimization targets the most vulnerable condition, thereby guaranteeing that no single damage event can cause catastrophic performance degradation.

- define constraints

Constraints include a global volume fraction limit, which restricts the total amount of material used, and may include additional constraints such as maximum displacement, stress limits, symmetry requirements, or manufacturing constraints such as draw direction or minimum feature size.

- introduce SIMP topology optimization approach

The Solid Isotropic Material with Penalization (SIMP) approach is employed to model the relationship between material density and stiffness. In this approach, the stiffness of each element is proportional to the density raised to a penalization power, typically between 2 and 4. This encourages the optimizer to drive densities toward either zero (void) or one (solid), avoiding intermediate values that lead to numerical artifacts.

- describe power law penalty

The power law penalty is a mathematical function that nonlinearly amplifies the stiffness contribution of high-density elements while suppressing the stiffness of low-density elements. This promotes the formation of clear, distinct material boundaries and prevents the emergence of gray regions that are mechanically inefficient and difficult to manufacture.

- apply lower bound on density variables

A small lower bound, typically 0.01, is applied to all density variables to prevent singular stiffness matrices during finite element analysis. This bound ensures numerical stability while having negligible mechanical impact due to the power law penalty.

- discuss damage shape alternatives

While the cube-shaped damage volume is used for computational simplicity and geometric clarity, alternative shapes such as spheres, ellipsoids, or irregular polygons may be employed to better represent specific damage mechanisms. The cube is chosen as the default due to its alignment with finite element grids and its conservative nature in simulating material removal.

- introduce cube-shaped damage

Cube-shaped damage is a three-dimensional rectangular void with edges aligned to the coordinate axes, placed at discrete locations within the design domain. Its geometry is compatible with structured finite element meshes and allows for precise determination of intersecting elements. The cube’s orientation is fixed in the base implementation but may be varied to simulate directional vulnerability.

- illustrate cube damage with FIG. 1B

FIG. 1B illustrates a cubic damage volume intersecting a finite element mesh, with elements fully enclosed by the cube marked for removal. The figure highlights the regular, grid-aligned nature of the damage and its ability to cleanly eliminate material without introducing mesh distortion.

- discuss orientation of damage cubes

The orientation of damage cubes affects the severity of the damage inflicted on structural members. A cube aligned with the principal axes of a member may cut through its cross-section more severely than a rotated cube. For conservative design, the cube is assumed to be aligned with the global coordinate system, but the method allows for multiple orientations to be considered in advanced implementations.

- introduce solution strategy for failsafe topology optimization

The solution strategy involves the generation of a finite damage population, the independent analysis of each damage configuration, and the aggregation of results to guide the optimization. The strategy is iterative, with each cycle updating the design variables based on the worst-case compliance and its sensitivity. The process continues until convergence criteria are met.

- describe random placement of damage

Although the damage placements are deterministic in the implemented method, they are designed to approximate the statistical distribution of random damage locations. The hierarchical placement ensures that every region of the domain is sampled with increasing density, effectively covering the space of possible damage events.

- introduce Damage Series A (DS-A)

Damage Series A (DS-A) is a family of damage populations in which each level doubles the number of damage cubes by halving the spacing between them. Level 1 consists of a gapless tiling of cubes, Level 2 adds a secondary layer centered at the intersections of Level 1 cubes, and so on. DS-A provides a systematic means of refining the damage model.

- describe base damage population

The base damage population corresponds to Level 1 of DS-A, in which damage cubes are placed without gaps or overlaps to completely fill the design domain. This population ensures that no region of the structure is left untested and serves as the foundation for higher-level series.

- illustrate base population with FIG. 2A

FIG. 2A illustrates the base damage population for a two-dimensional domain, showing a grid of non-overlapping damage squares covering the entire design area. Each square represents a potential damage event, and the figure demonstrates the complete spatial coverage achieved at Level 1.

- describe damage population increase

Damage population increase occurs by introducing additional layers of damage cubes at half the spacing of the prior level. Each increase doubles the number of damage instances and refines the resolution of the damage model, capturing previously untested regions of the domain.

- formulate design problem for DS-A

The design problem for DS-A is formulated as the minimization of the maximum compliance across all damage configurations in the selected level of the series. The problem is solved using a gradient-based optimizer with parallel finite element analysis, and the solution converges to a design that is robust against the most critical damage events in the population.

- introduce Damage Series B (DS-B)

Damage Series B (DS-B) is a partial subset of DS-A that includes only the enrichment layers beyond Level 1, omitting the full Level 1 base. This reduces the total number of damage instances by a factor of four while retaining the critical coverage of high-risk regions.

- describe partial set of DS-A

The partial set of DS-A consists of the damage cubes introduced at Level 2 and above, excluding the Level 1 base. This set targets the “best hideout” locations where material would otherwise survive a single damage event, thereby improving the fidelity of the model without the full computational cost of a complete Level 2 population.

- formulate design problem for DS-B

The design problem for DS-B is identical in form to that of DS-A but uses a reduced set of damage configurations. The optimization seeks to minimize the maximum compliance over this partial set, resulting in a design that is nearly as robust as one derived from the full Level 2 population but at significantly lower computational cost.

- summarize population size in Table 1

Table 1 summarizes the number of damage cubes required for each level of DS-A and DS-B, demonstrating the exponential growth of DS-A and the efficiency of DS-B. For example, at Level 2, DS-A requires 64 cubes while DS-B requires only 16, yet both achieve comparable fail-safe performance.

- discuss reliability of model for capturing random failure

The reliability of the model in capturing random failure is quantified by the maximum material survival rate within a representative structural section. As the damage population increases, this rate decreases, approaching zero as the population approaches infinity. The invention demonstrates that Level 1 and DS-B populations achieve survival rates below 50%, which is sufficient for practical fail-safe design.

- introduce best hideout location concept

The best hideout location is the position within the design domain where a structural member can survive a damage event with maximum residual cross-section. These locations occur at the junctions between damage cubes and represent the most vulnerable regions of the structure. The invention identifies these locations and ensures they are tested by the damage population.

- calculate material survival rate

The material survival rate is calculated as the ratio of the residual volume of a representative structural section to its original volume, assuming the section is aligned with the damage cube orientation. Exact formulae are derived for each damage population level, providing a quantitative measure of the model’s fidelity.

- discuss sectional survival rate

The sectional survival rate is the ratio of the residual cross-sectional area of a structural member to its original area. This metric is more relevant than volumetric survival because structural load transfer occurs through cross-sections. The invention shows that the sectional survival rate under DS-B is consistently lower than under DS-A at higher levels, indicating superior damage coverage.

- illustrate best hideout locations with FIGS. 2B and 2C

FIGS. 2B and 2C illustrate the best hideout locations for DS-A and DS-B, respectively, in a two-dimensional domain. The figures show regions of maximum residual material that survive a single damage event, highlighting the effectiveness of DS-B in eliminating these regions through targeted enrichment.

- compare DS-A and DS-B

DS-A provides complete coverage but at high computational cost, while DS-B achieves nearly identical fail-safe performance with only a quarter of the damage instances. DS-B is therefore the preferred choice for practical applications where computational efficiency is paramount.

- discuss convergence of damage population series

The damage population series converges in the sense that the maximum material survival rate decreases monotonically with each level, approaching zero as the population density increases. The convergence is rapid, with Level 2 providing sufficient accuracy for most engineering applications.

- discuss practical application of failsafe analysis

The practical application of fail-safe analysis lies in its ability to generate structural designs that are inherently redundant and robust against localized damage. The method is applicable to any industry where structural failure must be prevented, including aerospace, automotive, civil infrastructure, and medical devices.

- introduce finite element analysis

Finite element analysis is the numerical technique used to compute the structural response under each damage configuration. The method discretizes the design domain into a mesh of elements, applies boundary conditions and loads, and solves the equilibrium equations to determine displacements and stresses.

- describe optimization problem in Eq. 1

The optimization problem is defined in Equation 1 as the minimization of the maximum compliance over the finite damage population, subject to a volume constraint. The objective function is expressed as the supremum of compliances, and the design variables are the material densities of the finite elements.

- discuss computational expense

The computational expense of the method arises from the need to perform multiple finite element analyses, one for each damage configuration. However, because each analysis is independent, the expense is mitigated through parallel processing, making the method scalable to large problems.

- introduce parallel processing

Parallel processing is employed to distribute the finite element analyses across multiple processors or computing nodes. Each processor handles one or more damage configurations, and results are aggregated by a master process to update the design variables.

- describe Message Passing Interface (MPI) parallel algorithm

The Message Passing Interface (MPI) parallel algorithm coordinates the distribution of damage analyses across a cluster of computing nodes. The master process manages the optimization loop, while worker processes perform the finite element analyses and return compliance and sensitivity data. The algorithm ensures efficient communication and load balancing.

- illustrate failsafe algorithm with FIG. 3

FIG. 3 illustrates the flow of the fail-safe algorithm, showing the master process initiating damage population generation, distributing tasks to worker processes, collecting results, updating design variables, and repeating until convergence. The figure highlights the modular, parallel architecture of the system.

- describe master process

The master process is responsible for orchestrating the entire optimization process. It generates the damage population, initializes the design, distributes tasks to worker processes, collects and aggregates results, updates the design variables using a gradient-based optimizer, and determines convergence.

- describe damage population generation

Damage population generation is performed automatically by the system based on user-defined parameters such as damage size, shape, and population level. The algorithm ensures complete coverage of the design domain and excludes damage volumes that intersect with point loads.

- describe analysis and sensitivity analysis

Analysis involves solving the finite element equations for each damage configuration to compute compliance. Sensitivity analysis computes the gradient of compliance with respect to each design variable using the adjoint method, enabling gradient-based optimization.

- describe convergence and constraint screening

Convergence is determined when the change in the objective function falls below a specified tolerance. Constraint screening ensures that all design constraints are satisfied at each iteration, and violations are corrected by adjusting the design variables.

- describe approximation and optimization process

The approximation process involves modeling the material distribution as a continuous density field, while the optimization process iteratively adjusts this field to minimize the maximum compliance. The process combines SIMP penalization, filtering, and gradient-based updates to converge to a binary solution.

- describe output and optimization end

The output of the optimization is a final material distribution that satisfies the volume constraint and demonstrates robust performance under all damage scenarios. The optimization ends when convergence criteria are met, and the result is exported as a 3D geometry for manufacturing or further analysis.

- discuss practical measures for damage zone generation

Practical measures include excluding damage zones that intersect with point loads, applying material volume thresholds to eliminate negligible damage instances, and restricting damage placement to within the structural domain. These measures ensure numerical stability and physical relevance.

- discuss reducing computation cost

Computation cost is reduced by using partial damage populations (DS-B), limiting the number of damage instances, and employing parallel processing. The use of coarse meshes for initial iterations and refinement for final convergence further reduces cost.

- discuss preserving load conditions

Load conditions are preserved by excluding damage volumes that intersect with point loads or critical load paths. Distributed loads may be partially removed, as their effect is less critical than point loads.

- introduce three-bar truss example

The three-bar truss example demonstrates the fail-safe concept in a simplified structure. The truss is optimized under a volume constraint, and the fail-safe design is compared to the standard optimal design. The fail-safe design retains all three bars, while the standard design eliminates the central bar, demonstrating the necessity of redundancy.

- formulate optimization problem for three-bar truss

The optimization problem for the three-bar truss minimizes the maximum compliance under three failure scenarios: failure of each individual bar. The problem is solved using standard finite element analysis and gradient-based optimization.

- discuss failsafe design results

The fail-safe design results show that redundancy is essential for survivability. The optimal fail-safe design retains all three bars, whereas the standard design collapses under any single member failure. This illustrates the fundamental difference between efficiency-driven and safety-driven design.

### EXAMPLES

- introduce 2D example 1: rectangular plate under shear force

The first two-dimensional example involves a rectangular plate subjected to a shear force applied at its center. The design domain is discretized into 20,000 quadratic elements, and a volume constraint of 20% is applied. The fail-safe optimization is performed using DS-A and DS-B damage populations.

- describe finite element model

The finite element model consists of a 100 × 50 rectangular domain with a thickness of 1.0, modeled using 200 × 100 quadratic elements. Material properties are assigned, and boundary conditions are applied to fix the top edge and apply a horizontal load at the bottom edge.

- illustrate damage population PB1

FIG. 11 illustrates the damage population PB1, consisting of 108 damage squares placed in a grid pattern across the domain. The squares are sized to represent a significant damage event, and their placement ensures complete coverage.

- illustrate damage population PB2

FIG. 11 also illustrates PB2, which adds a second layer of 193 damage squares centered at the intersections of PB1 squares. This layer targets previously untested regions and improves the fidelity of the damage model.

- show optimum for standard problem

The optimum for the standard problem is a two-bar truss-like structure that minimizes compliance under nominal loading. This design is highly efficient but vulnerable to localized damage.

- show failsafe designs with PB1 and PB2 damage populations

The fail-safe designs generated using PB1 and PB2 populations show significantly increased redundancy, with multiple load paths emerging around the damage zones. The PB2 design exhibits slightly more uniform material distribution than PB1.

- provide models for damage population PB1

Models for PB1 include the undamaged structure and the eight damaged configurations corresponding to each damage square. The compliance of each configuration is computed and compared to identify the worst-case scenario.

- list compliances of standard and failsafe designs

The compliance of the standard design is 58.72, while the compliances of the PB1 and PB2 fail-safe designs are 84.28 and 82.96, respectively. The increase in compliance reflects the added material required for redundancy.

- list compliances for damage population PB2

The compliances for PB2 under each damage configuration are listed, with the maximum compliance occurring at a central damage location. The active damage zones are identified as those yielding compliances within 2% of the maximum.

- discuss difference in compliances

The difference in compliances between the standard and fail-safe designs demonstrates the cost of redundancy. However, the small difference between PB1 and PB2 indicates that PB1 is sufficient for many applications.

- introduce example 2: rectangular plate under bending force

The second example involves a rectangular plate subjected to a bending load. The design domain is optimized under a 50% volume constraint, and the fail-safe performance is evaluated using PB1 and PB2 damage populations.

- describe design domain and FEA mesh

The design domain is a 100 × 50 rectangle with a mesh of 200 × 100 quadratic elements. The load is applied at the center of the bottom edge, and the top edge is fixed. The mesh is refined to capture fine features.

- optimize with 50% volume constraint

The optimization is performed with a 50% volume constraint, allowing more material than in Example 1. The resulting designs are more robust due to the increased material budget.

- show damage population for PB1 and PB2

The damage populations for PB1 and PB2 are shown, with PB2 adding a second layer of cubes at the intersections of PB1. The number of damage instances is 108 and 193, respectively.

- show final designs for standard and failsafe

The final designs show that the standard solution forms a single load path, while the fail-safe designs develop multiple redundant paths around the damage zones.

- list compliances of standard and failsafe designs

The standard design has a compliance of 47.60, while the fail-safe designs have compliances of 72.15 (PB1) and 71.80 (PB2). The small difference confirms the efficiency of PB2.

- show active damage zones for PB1 and PB2

Active damage zones are those that yield compliances within 2% of the maximum. For PB1, four zones are active; for PB2, five. The additional zone in PB2 confirms improved coverage.

- discuss results for base damage population PB1 and increased population PB2

The results demonstrate that PB1 is sufficient for achieving fail-safe performance in most cases. PB2 provides marginal improvement at increased computational cost, making it suitable for critical applications.

- introduce example 3: 3D control arm

The third example involves a three-dimensional control arm used in automotive suspension systems. The design domain is approximately 450 × 550 × 110 mm, discretized into 327,493 tetrahedral elements.

- describe dimensions and model

The control arm is subjected to two load cases representing combined bending and torsion. A 30% volume constraint is applied, and a casting constraint is enforced to ensure manufacturability.

- apply 30% volume fraction constraint

The volume fraction constraint limits the total material to 30% of the design domain, forcing the optimizer to distribute material efficiently while maintaining redundancy.

- consider two load cases

The two load cases represent different operational conditions: one with dominant bending and another with dominant torsion. The total compliance under both cases is used as the objective function.

- show base layer damage population PB1

The base layer damage population PB1 consists of 45 damage cubes placed throughout the control arm. The cubes are sized to represent a significant manufacturing defect or impact damage.

- show enrichment layer for PB2

The enrichment layer for PB2 adds 28 additional cubes centered at the intersections of PB1 cubes, targeting previously untested regions.

- show optimal designs for standard and failsafe

The standard design forms a thin, efficient structure with minimal redundancy. The fail-safe design adds thickened ribs and additional material near the bearing points, creating multiple load paths.

- list compliances of standard and failsafe designs

The standard design has a compliance of 162.3, while the fail-safe design has a compliance of 193.7. The maximum compliance under damage is 756.8, indicating a fourfold increase under worst-case conditions.

- discuss maximum compliance of damaged structure

The high maximum compliance under damage indicates that the structure is highly sensitive to damage in the vicinity of the bearings. This insight suggests that additional support or material should be added in those regions.

- discuss failsafe features

The fail-safe features include thickened ribs, expanded cross-sections near load paths, and material redistribution away from single-point vulnerabilities. These features ensure that no single damage event can cause catastrophic failure.

- discuss additional implementations

Additional implementations include the use of spherical damage, multiple damage sizes, and dynamic damage populations that adapt during optimization. The method may also be extended to include thermal, dynamic, or multiphysics constraints.

- discuss use in various design applications

The method is applicable to aerospace components, automotive chassis, biomedical implants, wind turbine blades, and civil infrastructure. Any structure where localized damage could lead to catastrophic failure can benefit from this approach.

- discuss functional operations

Functional operations include input parsing, mesh generation, damage population generation, finite element analysis, sensitivity computation, optimization iteration, and output generation. These operations are fully automated and integrated into a single workflow.

- discuss computer storage medium

The method is stored on a non-transitory computer-readable medium, such as a hard drive or solid-state drive, as a software application that can be executed on a computing system.

- discuss data processing apparatus

The data processing apparatus includes a central processing unit, memory, input/output interfaces, and storage, configured to execute the fail-safe topology optimization algorithm. The apparatus may be a single workstation or a distributed computing cluster.

- discuss computer program

The computer program comprises a set of executable instructions that implement the method, including modules for geometry definition, meshing, damage generation, analysis, optimization, and result visualization.

- discuss computer readable media

Computer readable media include magnetic, optical, and semiconductor storage devices capable of storing the program instructions. The media may be distributed via physical media or downloaded over a network.