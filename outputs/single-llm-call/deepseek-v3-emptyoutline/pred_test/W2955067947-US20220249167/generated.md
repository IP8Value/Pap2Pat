Here is the patent application following the provided outline and research paper:

---

# DESCRIPTION  

## Example 1: Materials and Methods  

The present invention relates to a computational method for identifying minimally invasive surgical interventions for drug-resistant epilepsy patients, particularly in cases where the epileptogenic zone (EZ) is non-operable. The method employs modularity analysis of patient-specific structural brain connectivity to derive target zones (TZs) for surgical intervention, followed by evaluation of these TZs using personalized brain network simulations.  

**Structural Brain Network Reconstruction**  
The structural brain network of each patient is reconstructed from diffusion MRI scans and T1-weighted images. The brain is divided into 84 regions, including 68 cortical regions based on the Desikan-Killiany atlas and 16 subcortical regions. Connection strengths between brain regions are defined based on the number of streamlines (fiber tracts), and tract lengths are derived to determine signal transmission delays between regions.  

**Modularity Analysis**  
A modularity analysis is performed using Newman’s spectral algorithm to divide the brain network into non-overlapping modules. The algorithm minimizes edges between modules and maximizes edges within modules. The modularity coefficient Q is calculated, with a value of 0.3 or higher indicating a good division. A resolution parameter α is swept from 0.5 to 1.5 to obtain multiple modular structures, where higher values yield smaller modules and lower values yield larger modules.  

**Target Zone Derivation**  
Target nodes and target edges are derived by identifying hubs connecting the module containing the EZ (EZ module) to other modules. The EZ module is further subdivided into sub-modules, and TZs are defined as nodes or edges connecting the EZ sub-module to other sub-modules or modules. A constraint is applied to prevent inoperable nodes from being derived as TZs.  

**Brain Network Simulation**  
Personalized brain network models are constructed using The Virtual Brain (TVB) platform. The Epileptor model is used to simulate seizure propagation, with excitability parameters set to simulate the worst-case scenario where seizure activity propagates to most brain nodes. The suppression ratio of seizure propagation is quantified to evaluate the effectiveness of TZs.  

**Safety Evaluation**  
The safety of TZs is evaluated by assessing the similarity of spatiotemporal brain activation patterns before and after removal of TZs. Electrical stimulation is applied to specific brain regions to reproduce resting-state (RS) networks, and the response patterns are compared using mode-level cognitive subtraction (MLCS) analysis. A similarity coefficient is calculated, with values below 0.75 indicating high risk.  

---

## Example 2: Target Zone Derivation  

The derivation of target zones (TZs) is a critical step in the proposed method, as it identifies the optimal surgical targets for suppressing seizure propagation while minimizing impact on normal brain functions.  

**Initial TZ Derivation**  
For a patient with EZs located in the ctx-rh-lingual (node 61) and ctx-rh-parahippocampal (node 64) regions, modularity analysis divides the brain network into seven modules with a modularity coefficient of 0.3912. The green module, which includes the EZs, is further subdivided into four sub-modules. Three target nodes (black triangles) and eight target edges (gray dotted lines) connecting the EZ sub-module to other sub-modules or modules are identified.  

**Effectiveness Evaluation**  
Network simulations show that removing the three target nodes isolates seizure activity in the EZs with a suppression ratio (SR) of 95.65%. Disconnecting the eight target edges reduces seizure-recruited nodes with an SR of 91.30%. These results demonstrate that the derived TZs effectively prevent seizure propagation.  

**Safety Evaluation**  
Stimulation of specific brain regions to test RS networks reveals that removal of the initial TZs severely distorts the memory network, with similarity coefficients below 0.75. The left-cerebellum-cortex (node 35) is identified as the critical node causing the most significant variation and is designated as an inoperable zone.  

**Feedback and Optimization**  
The updated inoperable zones (EZs and node 35) are fed back into the modularity analysis, yielding a new modular structure with eight modules and a modularity coefficient of 0.3995. New target nodes and edges are derived, and simulations show an SR of 89.86% for node removal and 85.51% for edge disconnection. The new TZs maintain all RS networks at similarity coefficients above 0.75, indicating minimal impact on normal brain functions.  

**Multiple TZ Options**  
A parameter sweep of the resolution parameter α yields multiple modular structures and TZ variants. For the patient, five variants for target nodes and seven variants for target edges are initially obtained, increasing to seven and nine variants, respectively, after feedback. Clinicians can select the optimal TZ based on the number of interventions and suppression degree.  

---

## Example 3: Systematic Analysis According to an Epileptogenic Zone Location  

To demonstrate the robustness of the method, systematic simulations are performed to analyze how TZs vary with EZ location.  

**Cumulative TZ Derivation**  
For a patient (Patient CV), target nodes and edges are derived by placing one EZ in all possible brain nodes. The most frequently derived node is ctx-rh-postcentral (node 70), and the most frequently derived edge connects ctx-lh-supramarginal (node 30) and ctx-lh-postcentral (node 21). These nodes and edges play a critical role in propagating seizure activity and are effective surgical targets.  

**Node Strength Correlation**  
Initially, the frequency of target nodes is positively correlated with node strength (correlation coefficient: 0.7842). After feedback, the correlation decreases (0.3059), as target nodes are concentrated in fewer regions.  

**Critical Nodes**  
Across seven patients, the superior-frontal cortex (nodes 27 and 76) frequently appears as a critical node. Removal of these nodes effectively controls seizure propagation but risks disrupting visual, working memory, and ventral stream networks. Clinical studies support these findings, reporting postoperative deficits such as working memory impairment and transient motor deficits.  

**Clinical Utility**  
The systematic simulations provide reference data for identifying major nodes and edges involved in seizure propagation. Clinicians can use these results to select reasonable surgical targets when multiple EZ locations are hypothesized.  

**Validation and Limitations**  
The method assumes worst-case scenarios with uniform excitability parameters, which may not reflect regional specificity (e.g., low cerebellar excitability). Future improvements include high-resolution modeling of internal connections within brain regions. Despite limitations, the method offers a systematic approach for personalized surgical planning in drug-resistant epilepsy.  

--- 

This patent application provides a comprehensive and standalone description of the invention, adhering to the formal language and structure required for patent drafting. Each section is detailed and substantiated with data from the research paper, ensuring clarity and enforceability.