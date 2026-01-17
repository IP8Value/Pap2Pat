# DESCRIPTION

## Example 1: Materials and Methods

The personalized in-silico surgical approach was based on graph theoretical analysis and brain network simulations. Preferentially, from the modularity analysis considering inoperable zones, brain regions and fiber tracts acting as hubs in the interaction between the modules were derived as target zones (TZs). Then, the obtained TZs were evaluated in terms of the effectiveness and the safety by personalized brain network simulations using The Virtual Brain (TVB), a platform to simulate the brain network dynamics. If the TZ did not satisfy the evaluation criteria, a new TZ was derived by feeding back the simulation results to the modularity analysis again. Through the feedback approach, the optimized TZ options that minimize seizure propagation while not affecting normal brain functions could be obtained. A detailed description of each step is provided below.

### Structural Brain Network Reconstruction

Neuroimaging data was obtained from seven drug-resistant epilepsy patients. The patients had epileptogenic zones (EZs) with different locations and underwent comprehensive presurgical evaluations. The clinical characteristics of each patient are provided in the supplementary materials. The structural brain network of each patient was reconstructed from diffusion MRI scans and T1-weighted images using the SCRIPTS pipeline. Each patient’s brain was divided into 84 regions, which included 68 cortical regions based on the Desikan-Killiany atlas, and 16 subcortical regions. Connection strengths between the brain regions were defined based on the number of streamlines (fiber tracts), and tract lengths to determine signal transmission delays between the regions were also derived.

### Target Zone Derivation Based on the Patient-Specific Modular Structure

#### Modularity Analysis

To analyze the modular structure of the brain network, a previously reported Matlab toolbox was utilized. The modularity analysis based on Newman’s spectral algorithm provided the non-overlapping modular structure that minimizes edges between modules and maximizes edges within modules. It computed the leading eigenvector of the modularity matrix \( B \) and divided the network nodes into two modules according to the signs of the elements in the eigenvector. The modularity matrix \( B \) is defined as:

\[ B_{ij} = A_{ij} - \alpha \frac{k_i k_j}{m} \]

where \( A_{ij} \) represents a weight value between node \( i \) and node \( j \), \( k_i \) and \( k_j \) indicate the degree of each node, \( m \) denotes the total number of edges in the network, and \( \alpha \) is a resolution parameter for the analysis. The classic value of \( \alpha \) is 1. The division was fine-tuned by the node moving method to obtain the maximal modularity coefficient \( Q \). The modularity coefficient has a value ranging from 0 to 1, with a value of 0.3 or higher generally indicating a good division. Each module, which was divided based on the eigenvector algorithm, was further divided into two modules until there was no effective division that resulted in a positive modularity coefficient.

In this study, a constraint was added to the existing toolbox to prevent inoperable nodes from being derived as TZs. First, the group membership variable values of the nodes classified by the eigenvector algorithm were identified. Then, if the inoperable node and its neighboring nodes (adjacent nodes based on the weight matrix) did not have the same value, the values of these nodes were set to the value that most of them had. In other words, the constraint limited the inoperable node and its neighbor nodes to belong to the same module, so that the inoperable node did not act as a hub connecting the modules. Meanwhile, the resolution parameter \( \alpha \) was swept from 0.5 to 1.5 with intervals of 0.25 to obtain multiple modular structures. The resolution parameter determines the size of each module, i.e., the number of modules, in dividing the network nodes into modules. A high parameter value derives a modular structure consisting of small modules (i.e., a large number of modules) and a low parameter value obtains a structure consisting of large modules (i.e., a small number of modules).

#### Target Node and Target Edge

To derive TZs from the modularity analysis, EZ and inoperable zones should be set first. The EZs were fixed according to the clinical evaluation of each patient, and the inoperable zones were arbitrarily set to all EZs, i.e., we assumed the worst-case scenario in which all EZs cannot be surgically removed. In detail, the goal was to obtain TZs excluding all EZs for resection surgery and excluding all fiber tracts connected to the EZs for disconnection surgery.

The strategy to suppress the seizure propagation is to divide each patient's brain network into multiple modules and then remove the connections (nodes or edges) from the module containing the EZ (EZ module) to the other modules. However, in the modularity analysis, when a low resolution parameter is used, a relatively large number of nodes may belong to the same module with the EZ, and eventually, quite a few nodes may still be seizure-recruited even if the TZs are eliminated. To control this issue (i.e., to prevent a significant number of nodes from becoming seizure-remained nodes), the strategy was to divide the EZ module into sub-modules once again and define the TZs as the nodes/edges that connect the submodule including the EZ (EZ submodule) to other submodules or modules. The nodes and the edges acquired for resection and disconnection surgery were named target nodes and target edges, respectively.

Since the resolution parameter in the modularity analysis was controlled, multiple modular structures were obtained for the same patient, thereby providing multiple intervention options for target nodes and target edges. All of the procedures described above were automatically performed by the Matlab model that was developed. The model could yield multiple TZ options according to the location of EZ and inoperable zones.

### Brain Network Simulation Using The Virtual Brain

#### Effectiveness Evaluation

Patient-specific network models were constructed using TVB to verify the effectiveness of derived TZs. The six-dimensional Epileptor model was specifically employed to describe a network node, and the reconstructed structural connectivity was used to connect the nodes. The Epileptor is a phenomenological neural population model reproducing seizure characteristics, which consists of five state variables and six parameters. Each Epileptor was coupled with others via the permittivity coupling of slow time scales variable \( z \) replicating extracellular effects. The equations governing the Epileptor model are as follows:

\[ \begin{aligned}
\dot{x}_{1,i} &= y_{1,i} - f_1(x_{1,i}, x_{2,i}) - z_i + I_1 \\
\dot{y}_{1,i} &= \varepsilon_1 (f_2(x_{2,i}) - y_{1,i}) \\
\dot{x}_{2,i} &= y_{2,i} - x_{1,i} - z_i + I_2 \\
\dot{y}_{2,i} &= \varepsilon_2 (x_{1,i} - y_{2,i}) \\
\dot{z}_i &= \mu (4 (x_{1,i} - x_0) - z_i)
\end{aligned} \]

where:
- \( f_1(x_{1,i}, x_{2,i}) = \begin{cases} 
x_{1,i}^3 - 3x_{1,i}^2 & \text{if } x_{1,i} < 0 \\
(x_{2,i} - 0.6(z_i - 4)^2)x_{1,i} & \text{if } x_{1,i} \geq 0 
\end{cases} \)
- \( f_2(x_{2,i}) = \begin{cases} 
0 & \text{if } x_{2,i} < -0.25 \\
6(x_{2,i} + 0.25) & \text{if } x_{2,i} \geq -0.25 
\end{cases} \)
- \( g(x_{1,i}) = \int_{t_0}^{t} e^{-\gamma(t - \tau)} x_{1,i}(\tau) d\tau \)

Clinically, degrees of epileptogenicity may be mapped upon the excitability parameter \( x_0 \) where we distinguish EZ that generates spontaneous seizure activities, propagation zone (PZ) that is recruited by seizure propagation from EZ, and other zones not recruited in the propagation. In this study, the excitability parameter \( x_0 \) was set to -1.6 for EZ, and a value between -2.150 and -2.095 corresponding to PZ for all other nodes depending on the structural connectivity of each patient, to simulate the worst-case scenario at which seizure activity originated from EZ propagates to most other brain nodes. For the other parameters in the equations, \( I_1 = 3.1 \), \( I_2 = 0.45 \), \( \gamma = 0.01 \), \( \tau_0 = 6667 \), and \( \tau_2 = 10 \). Zero mean white Gaussian noise with a standard deviation of 0.0003 was linearly added to the variables \( x_2 \) and \( y_2 \) in each Epileptor for stochastic simulations. These noise environments made each Epileptor excitable and thus produced interictal spikes as a baseline activity.

Using the patient-specific network model, the seizure propagation characteristics were simulated before and after eliminating target nodes or target edges. In particular, the suppression ratio of seizure propagation was quantified as:

\[ \text{Suppression Ratio (SR)} = 1 - \frac{\sum_{i=1}^{N} \text{Seizure-Recruited Nodes After Removal}}{\sum_{i=1}^{N} \text{Seizure-Recruited Nodes Before Removal}} \]

The \( x_1 + x_2 \) waveform of each Epileptor was observed to reproduce local field potential at each node.

#### Safety Evaluation

To assess normal brain function, a stimulation paradigm was adapted, in which the information transmission capacity of the network was quantified through the spatiotemporal properties of the trajectory leading to its resting state after a transient stimulation. Eight particular well-known resting state (RS) networks were tested, which include default mode, visual, auditory-phonological, somato-motor, memory, ventral stream, dorsal attention, and working memory. Previous work has shown that stimulating a specific brain region could reproduce dynamically responsive networks similar to brain activation patterns in RS networks. Spiegler and colleagues have reported the best-matched stimulation sites with each RS network in cortical and subcortical regions.

Based on the previous studies, an electrical pulse of 2.5 seconds was applied to a particular cortical region, and the response signals in all brain regions were observed. The stimulation sites to test each RS network are shown in Table 1, where the number in parentheses represents the node index. In this simulation, the patient-specific network models were used as before, with the neural mass model of the generic 2-dimensional oscillator rather than the Epileptor, to replicate damped oscillations due to the stimulation. The equations governing the generic 2-dimensional oscillator are as follows:

\[ \begin{aligned}
\dot{x}_i &= y_i \\
\dot{y}_i &= -a x_i - b y_i + c + d \sum_{j=1}^{N} K_{ij} (x_j - x_i) + e \sum_{j=1}^{N} K_{ij} (y_j - y_i) + f \eta_i(t)
\end{aligned} \]

For the parameters, \( \tau = 1 \), \( a = -0.5 \), \( b = -15.0 \), \( c = 0.0 \), \( d = 0.02 \), \( e = 3.0 \), \( f = 1.0 \), and \( g = 0.0 \). Each oscillator was coupled with other oscillators via difference coupling based on individual structural brain connectivity. Each oscillator (brain node) operated at a stable focus in proximity to the instability point, supercritical Andronov-Hopf bifurcation, but never reached the critical point. Each node showed no activity without stimulation, but when stimulated (or received input from other nodes through connectome), it generated a damped oscillation by operating closer to the critical point. Since the working distance to the critical point was determined depending on each node’s connectivity (connection weights and time delays), each node generated different damped oscillations (with different amplitudes and decay times), thereby producing a specific energy dissipation pattern (responsive activation pattern) according to the stimulation location and brain connectivity.

Then, the responsive spatiotemporal activation patterns before and after removing target nodes or target edges were compared. To do so, the subspace in which a trajectory evolves after stimulation was quantified by employing mode level cognitive subtraction (MLCS) analysis. From the principal component analysis (PCA) using response signals in all brain nodes before in-silico surgery, a reference coordinate system was derived, i.e., eigenvectors \( \phi_n \) of the covariance matrix of response signals were calculated. Then, three principal components (PC) were selected, and response signals in both cases (before and after removal of TZ, \( q_b \), \( q_a \)) were projected upon the PC, and reconstructed responsive signals \( q_{r,b} \), \( q_{r,a} \) were obtained at each brain node.

To compare the reconstructed responsive patterns, the amount of overlap between the powers of the reconstructed response signals before and after eliminating TZ was calculated for every brain node. The obtained value in each brain node was normalized by the overlap value using only the signal power before removal of TZ, and then defined as the similarity coefficient (defined as \( 1 - \) the deviation from 1, if the value > 1; thereby, the similarity coefficient has a value between 0 and 1). Here, it was considered that the derived TZ had a high risk if the mean value of similarity coefficients in all brain regions was below 0.75. In other words, it indicated that the elimination of the TZ could affect the corresponding RS network. The TZ with a high risk was referred to as an inoperable zone. If the TZs contained more than one node, the critical node that severely changed the responsive activation patterns due to stimulation was figured out, and then designated as an inoperable zone. The critical node was defined as a node that yielded the lowest similarity coefficients when the same simulation was repeated after removing each node belonging to the TZ. The updated inoperable zone (added the critical node) was applied to the modularity analysis again, which resulted in a new TZ. The effectiveness and safety of the newly obtained TZ were evaluated through network simulations again. These feedback procedures were iterated until the TZs that met the safety criteria were acquired.

## Example 2: Target Zone Derivation

### Modularity Analysis

The modularity analysis was performed to derive the target zones (TZs) for surgical intervention. The brain network of each patient was divided into modules using the Newman’s spectral algorithm. The resolution parameter \( \alpha \) was varied from 0.5 to 1.5 with intervals of 0.25 to obtain multiple modular structures. The constraint was added to ensure that inoperable nodes and their neighboring nodes belonged to the same module, preventing them from being derived as TZs. The modularity coefficient \( Q \) was calculated to assess the quality of the modular structure, with a value of 0.3 or higher indicating a good division.

### Target Node and Target Edge

The target nodes and target edges were derived from the modular structure. The EZ and inoperable zones were set first, and the brain network was divided into modules. The EZ module was further divided into sub-modules, and the TZs were defined as the nodes and edges connecting the EZ sub-module to other sub-modules or modules. The resolution parameter was controlled to obtain multiple TZ options, providing flexibility in surgical planning.

### Effectiveness and Safety Evaluation

The effectiveness of the derived TZs was evaluated by simulating seizure propagation characteristics before and after removing the TZs. The suppression ratio of seizure propagation was quantified to compare the removal effect of each TZ. The safety of the TZs was assessed by quantifying the information transmission capacity of the network through the spatiotemporal properties of the trajectory leading to its resting state after a transient stimulation. The similarity coefficient was calculated to compare the responsive activation patterns before and after removing the TZs. If the mean value of similarity coefficients was below 0.75, the TZ was considered to have a high risk and was designated as an inoperable zone. The critical node that severely changed the responsive activation patterns was identified, and the updated inoperable zone was applied to the modularity analysis again to derive new TZs. The feedback procedures were iterated until the TZs that met the safety criteria were obtained.

## Example 3: Systematic Analysis According to an Epileptogenic Zone Location

To demonstrate the robustness of the proposed method, additional simulation results were presented showing how TZs vary according to the location of the EZ. Systematic simulations were performed where one EZ was placed in all possible brain nodes, and the cumulative results of TZs were analyzed to identify the nodes and edges that are frequently used as TZs. These frequently derived nodes and edges play a crucial role in propagating seizure activity from the localized region to the entire brain and can effectively control seizure propagation by being removed.

The frequency of the target nodes initially acquired was positively correlated with the node strength, i.e., the nodes having high strength were frequently derived as TZs. However, the final target nodes obtained from the feedback procedure tended to be more concentrated at few nodes, and thus the frequency of the finally acquired target nodes was not noticeably relevant to the node strength.

The critical nodes, which were used for the feedback strategy to consider the safety for normal brain functions, were not significantly different across all patients. In particular, the superior-frontal cortex appeared often as the critical node, indicating that these nodes are effective in controlling seizure propagation but removing them may cause problems for normal brain function. The network simulation results identified that the elimination of these nodes severely distorted the RS networks corresponding to visual, working memory, and ventral stream as well as default mode.

The systematic simulations provided a reference to elicit reasonable surgical targets if there were several clinical hypotheses for the EZ location. The results can be used to identify major nodes and edges involved in seizure propagation, offering clinicians multiple options for surgical intervention.