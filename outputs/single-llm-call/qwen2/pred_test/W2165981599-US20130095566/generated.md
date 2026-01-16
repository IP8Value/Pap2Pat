# DESCRIPTION

## STATEMENT REGARDING FEDERAL FUNDING
This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## SUMMARY
The present invention relates to a method and system for optimizing the growth and substrate utilization of Escherichia coli (E. coli) in mixed substrate cultures through the regulation of carbon catabolite repression (CCR). Specifically, the invention provides a method for modulating CCR to maintain a near-constant intracellular macromolecular density, thereby enhancing cell growth and substrate utilization efficiency. The invention also includes a computational model, flux balance analysis with macromolecular crowding (FBAwMC), which predicts the substrate uptake order and rate in both single and mixed substrate cultures. Additionally, the invention discloses methods for characterizing the growth of E. coli in single and mixed substrate cultures, and the effects of CCR on cell density and volume.

## DETAILED DESCRIPTION
The present invention addresses the need for optimizing the growth and substrate utilization of E. coli in mixed substrate cultures by modulating carbon catabolite repression (CCR). CCR is a regulatory mechanism that ensures the preferential utilization of certain substrates over others, contributing to the efficient and rapid growth of E. coli. The invention provides a method for modulating CCR to maintain a near-constant intracellular macromolecular density, which is crucial for optimal cell growth and substrate utilization.

### Modulation of CCR
The invention involves modulating CCR in E. coli to optimize its growth and substrate utilization in mixed substrate cultures. This modulation can be achieved through genetic engineering, environmental manipulation, or the use of specific inducers. For example, the deletion of the glucose transporter gene (ptsG) can disrupt CCR, leading to altered substrate utilization patterns and growth defects. Conversely, the induction of specific regulons, such as the maltose regulon, can transiently disrupt CCR, affecting cell growth and density.

### Computational Modeling
The invention includes a computational model, flux balance analysis with macromolecular crowding (FBAwMC), which predicts the substrate uptake order and rate in both single and mixed substrate cultures. FBAwMC takes into account the total enzyme occupancy limit inside the cell due to the highly crowded nature of the cell’s cytoplasm. This model accurately predicts the observed sequential substrate uptake kinetics and provides insights into the role of CCR in maintaining intracellular macromolecular density.

### Characterization of Growth in Single and Mixed Substrate Cultures
The invention provides methods for characterizing the growth of E. coli in single and mixed substrate cultures. In single substrate cultures, E. coli displays different growth rates and substrate uptake patterns compared to mixed substrate cultures. For instance, glucose-limited cultures exhibit the fastest growth, while lactate-limited cultures grow the slowest. In mixed substrate cultures, E. coli preferentially utilizes glucose, followed by maltose, galactose, glycerol, and lactate. The activation of CCR in mixed substrate cultures correlates with the increasing rate of E. coli cell growth and proliferation.

### Effects of CCR on Cell Density and Volume
The invention also discloses the effects of CCR on cell density and volume. In mixed substrate cultures, the activation of CCR is associated with a near-constant intracellular macromolecular density, which is crucial for optimal cell growth. Perturbation of CCR, either through genetic modification or environmental manipulation, can lead to changes in cell density and volume. For example, the deletion of the ptsG gene results in lower buoyant density and larger cell volume, indicating that physiological cell density and volume regulation are intertwined with CCR.

### Applications
The methods and systems of the present invention have numerous applications in biotechnology and industrial microbiology. By optimizing the growth and substrate utilization of E. coli, the invention can enhance the production of biofuels, pharmaceuticals, and other valuable compounds. Additionally, the computational model FBAwMC can be used to predict and optimize the performance of E. coli in various bioprocesses.

## EXAMPLES

### Example 1
**Modulation of CCR through Genetic Engineering**
In this example, the ptsG gene, encoding the glucose transporter, was deleted in E. coli to disrupt CCR. The resulting ΔptsG mutant was grown in a mixed substrate chemostat culture. The growth rate and substrate utilization patterns of the mutant were compared to those of the wild-type strain. The ΔptsG mutant displayed growth defects and altered cell density and volume, indicating that CCR is essential for optimal cell growth in mixed substrate cultures.

### Example 2
**Induction of Maltose Regulon to Disrupt CCR**
In this example, the maltose regulon was induced in a mixed substrate culture by adding cAMP and maltotriose. The induction of the maltose regulon transiently disrupted CCR, leading to a slight inhibition of cell growth and an increase in cell volume. These results demonstrate that the activation of CCR is crucial for maintaining optimal intracellular macromolecular density and cell growth.

### Example 3
**Characterization of Growth in Single Substrate Cultures**
In this example, E. coli was grown in single substrate cultures containing glucose, glycerol, galactose, lactate, and maltose. The growth rates and substrate uptake patterns were characterized for each substrate. The results showed that glucose-limited cultures exhibited the fastest growth, while lactate-limited cultures grew the slowest. These findings highlight the importance of substrate preference in E. coli growth.

### Example 4
**Prediction of Substrate Uptake Kinetics Using FBAwMC**
In this example, the FBAwMC model was used to predict the substrate uptake kinetics in both single and mixed substrate cultures. The model accurately predicted the observed sequential substrate uptake kinetics in mixed substrate cultures, where glucose was preferentially utilized, followed by maltose, galactose, glycerol, and lactate. The model also predicted the substrate uptake rates in single substrate cultures, which correlated with the experimental data.

## Materials and Methods

### Bacterial Strains and Growth Conditions
The wild-type E. coli MG1655 strain and the CCR mutant, ΔptsG, were used throughout the study. Cells were cultured in Luria-Bertani broth (LB) at 37°C with agitation at 200 rpm. For single substrate cultures, the concentration of substrates was 0.2% w/vol. In mixed substrate cultures, five substrates (glucose, glycerol, galactose, lactate, and maltose) were added to a final concentration of 0.04% w/vol each. Continuous-feed chemostat cultures were grown in a Labfors bioreactor at different dilution rates (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7/hr).

### GFP Reporter System of Maltose Regulon Genes
The GFP reporter system was constructed by amplifying the promoter regions of maltose regulon genes (malEKSTPZ) using E. coli genomic DNA. The amplified sequences were cloned into a low-copy reporter plasmid, pCS21, and transformed into E. coli MG1655 cells. The promoter activities of the maltose regulon genes were measured by the ratio of GFP/OD600nm in the presence or absence of inducers (cAMP and maltotriose).

### Plasmids
High-copy reporter plasmids (pET28a) were used to express the UCH gene under the control of an inserted promoter. These plasmids were obtained from Dr. Hao Liu, University of Pittsburgh.

### Batch Culture Experiments with Inducer Induction
E. coli cells were cultured in LB overnight and then inoculated into 24-well plates containing mixed substrate media. Inducers (4 mM cAMP and 200 μM maltotriose) were added after 15 minutes of culture, and cell growth was monitored for 6 hours. Substrate consumption was measured by sampling the cell culture media at different time points.

### Cell Growth and Growth Rate Calculation
Cell growth data were collected at 30-minute intervals by measuring the optical density (OD600nm) using a photometer. The growth rate (GR) was calculated using the formula:
\[
\text{GR} = \left( \frac{\text{OD}_t / \text{OD}_{(t-1)}}{\Delta t} \right)
\]
where GR denotes the growth rate at time \( t \) (hr-1), \(\text{OD}_t\) is the OD600nm measured at time \( t \), and \(\Delta t\) is the sampling interval (hr).

### Cell Density Measurement with Ficoll Gradient
The Ficoll step gradient was prepared by dissolving Ficoll 400 powder in 1× PBS to achieve a concentration of ~60% w/v. The density of the solution was measured with a densitometer. The lower density solutions were prepared by diluting the Ficoll solution with PBS. The cells were centrifuged, resuspended in PBS, and layered onto the Ficoll gradient. The gradient was centrifuged at 16,000 g for 1 hour at 4°C. The cell density distribution (CDD) was calculated using the formula:
\[
\text{CDD}_{\rho i} = \frac{\text{OD}_{\rho i}}{\sum_{i=0}^{7} \text{OD}_{\rho i}}
\]

### Cell Volume Measurement
Cells in chemostat culture were sampled and diluted in a suspension solution. The cell volume was measured using a Multisizer 3 (Beckman Coulter). Each sample was measured twice.

### Western Blotting
Protein lysates were mixed with sample buffer, boiled, and loaded onto a 15% SDS-PAGE gel. The gel was transferred to a nitrocellulose membrane and blotted with a mouse monoclonal MalE antibody. The secondary HRP-anti mouse antibody was used to probe the membrane, and the membrane was processed for chemiluminescence development.

### Substrate Concentration Measurement
The sampled cell culture was centrifuged, and the supernatant was incubated at 80°C to deactivate enzymes. Substrate concentrations were determined using a commercial assay kit (R-Biopharm, Germany).

### Testing Maltose Regulon Promoter Activities
Promoter-GFP reporter plasmid-containing E. coli cells were used to measure the activities of the maltose regulon genes. The ratio of GFP/OD600nm was calculated to evaluate promoter activities.

### Microarray Experiments
Cells were sampled from chemostat cultures and processed for RNA isolation. RNA samples were treated with DNase to remove DNA contamination. The RNA samples were processed for transcriptome analysis using E. coli Affymetrix microarray chips. Microarray data were analyzed using R and the ‘affy’ package.

### FBAwMC Model
The FBAwMC model was used to simulate the substrate uptake kinetics in mixed and single substrate cultures. The model takes into account the total enzyme occupancy limit inside the cell due to macromolecular crowding.

### Statistics
Data were expressed as the mean ± SD and plotted using GraphPad Prism 5. Intergroup differences were assessed using the t-test.

## Availability of Supporting Data
Microarray data have been deposited in the Gene Expression Omnibus (GEO accession number: GSE51581).

# Abbreviations
CCR: Carbon catabolite repression; FBAwMC: Flux balance analysis with macromolecular crowding; MC: Macromolecular crowding; OxPhos: Oxidative phosphorylation; PTS: Phosphor-transferase system; UCHL1: Ubiquitin carboxy-terminal hydrolase-L1.

# Competing Interest
The authors declare that no competing financial and non-financial interests exist.

# Authors’ Contributions
YZ and ZNO designed the experiments. YZ, TW, and KW performed the experiments. AV and AW performed the modeling and microarray data analysis, respectively. YZ, AV, ZBJ, and ZNO wrote the manuscript. All authors read and approved the final manuscript.