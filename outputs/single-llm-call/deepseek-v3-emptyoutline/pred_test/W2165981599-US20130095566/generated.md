Here is the complete patent application following the provided outline and incorporating the research paper's invention:

# DESCRIPTION  

## STATEMENT REGARDING FEDERAL FUNDING  

The invention described herein was not made with federal funding.  

## SUMMARY  

The present invention relates to methods and systems for modulating carbon catabolite repression (CCR) in bacterial cells to optimize intracellular macromolecular crowding (MC) and enhance cell growth. The invention is based on the discovery that CCR is a regulatory mechanism that maintains near-constant intracellular macromolecular density in rapidly proliferating bacterial cells. The invention provides methods for controlling substrate uptake kinetics, growth rates, and metabolic pathways in bacterial cultures by modulating CCR activity.  

The invention further encompasses a computational model, Flux Balance Analysis with Macromolecular Crowding (FBAwMC), which predicts substrate uptake order and growth kinetics in bacterial cultures by accounting for intracellular crowding constraints. This model enables the optimization of culture conditions for enhanced biomass production.  

Additionally, the invention includes engineered bacterial strains with modified CCR activity, methods for transiently disrupting CCR to study cell growth dynamics, and applications in industrial fermentation and biotechnology.  

## DETAILED DESCRIPTION  

The present invention provides methods and systems for controlling bacterial cell growth by modulating carbon catabolite repression (CCR) to maintain optimal intracellular macromolecular crowding (MC). CCR is a regulatory mechanism that ensures selective substrate uptake in bacterial cells, such as *Escherichia coli*, when grown in mixed-substrate environments. The invention is based on the discovery that CCR activation correlates with cell growth rates and serves to stabilize intracellular MC during rapid proliferation.  

Key aspects of the invention include:  

1. **CCR and Growth Rate Correlation**: The invention demonstrates that CCR is inactive in slow-growing bacterial cells but becomes increasingly activated as growth rates increase. In mixed-substrate cultures, CCR ensures preferential glucose uptake at high growth rates while suppressing the uptake of less favorable substrates (e.g., lactate, glycerol).  

2. **Intracellular MC Maintenance**: The invention reveals that bacterial cells maintain near-constant buoyant density across varying growth rates, despite dynamic changes in cell volume. CCR contributes to this stability by regulating metabolic pathway utilization, such as the shift from oxidative phosphorylation (OxPhos) to aerobic glycolysis at high growth rates.  

3. **FBAwMC Model**: The invention incorporates a computational model, Flux Balance Analysis with Macromolecular Crowding (FBAwMC), which accurately predicts substrate uptake kinetics and growth behavior by accounting for intracellular crowding constraints. This model enables the design of optimized culture conditions for industrial applications.  

4. **Engineered Bacterial Strains**: The invention includes bacterial strains with modified CCR activity, such as Δ*ptsG* mutants, which exhibit altered substrate uptake profiles and growth defects at high proliferation rates. These strains are useful for studying CCR dynamics and optimizing fermentation processes.  

5. **Transient CCR Disruption**: The invention provides methods for transiently disrupting CCR, such as by inducing maltose regulon genes with cAMP and maltotriose, to study cell growth inhibition and MC adjustments.  

## EXAMPLES  

### Example 1  

**CCR Activation in Mixed-Substrate Cultures**  

*E. coli* cells were cultured in a medium containing glucose, glycerol, galactose, lactate, and maltose. Substrate uptake kinetics revealed sequential consumption, with glucose utilized first, followed by maltose, galactose, lactate, and glycerol. This behavior was predicted by the FBAwMC model, confirming that CCR is active in mixed-substrate environments.  

At low growth rates (0.1/hr dilution rate), CCR was absent, and all substrates were consumed simultaneously. As growth rates increased (0.2–0.7/hr), CCR became progressively more pronounced, with glucose dominating uptake at the highest rates.  

### Example 2  

**Transcriptional Regulation of CCR**  

Microarray analysis of *E. coli* cells grown at varying dilution rates revealed that transporter gene expression (e.g., *ptsG*, *malEFG*) was tightly regulated by growth rate. At low rates, genes for multiple substrates were highly expressed, while at high rates, only glucose transporter genes remained active. This transcriptional shift aligns with the observed CCR-mediated substrate hierarchy.  

### Example 3  

**Δ*ptsG* Mutant Phenotype**  

The Δ*ptsG* mutant, lacking the glucose transporter gene, displayed growth defects at high dilution rates (>0.5/hr) in mixed-substrate cultures. Substrate uptake was altered, with glycerol and lactate consumption increasing while glucose uptake remained suppressed. The mutant also exhibited lower cell density and larger cell volume, demonstrating the role of CCR in MC maintenance.  

### Example 4  

**Transient CCR Disruption**  

Addition of cAMP (4 mM) and maltotriose (200 μM) to a mixed-substrate culture induced maltose regulon genes, transiently disrupting CCR. This led to delayed growth, reduced glucose uptake, and increased cell volume, confirming that CCR is essential for optimal MC and proliferation.  

## Materials and Methods  

### Bacterial Strains and Growth Conditions  

Wild-type *E. coli* MG1655 and Δ*ptsG* mutant strains were cultured in M9-minimal salts media supplemented with single or mixed carbon substrates (glucose, glycerol, galactose, lactate, maltose). Batch and chemostat cultures were maintained at 37°C, with growth rates controlled by dilution rates (0.1–0.7/hr).  

### GFP Reporter System  

Maltose regulon promoter-GFP reporter plasmids were constructed to monitor gene expression dynamics. Promoter activities were quantified as GFP/OD600 ratios in induced vs. uninduced cultures.  

### Cell Density and Volume Measurements  

Cell buoyant density was assessed using Ficoll gradient centrifugation. Cell volume was measured with a Multisizer 3 particle counter.  

### Substrate Concentration Analysis  

Culture supernatants were assayed for substrate concentrations using enzymatic kits (R-Biopharm).  

### FBAwMC Modeling  

The FBAwMC model simulated metabolic fluxes and substrate uptake kinetics under crowding constraints. Predictions were validated against experimental data.  

### Microarray Experiments  

RNA was isolated from chemostat cultures and analyzed using *E. coli* Affymetrix microarrays. Data were processed with MAS5 and dChip for gene expression profiling.  

This patent application provides a comprehensive framework for leveraging CCR modulation to optimize bacterial cell growth and industrial fermentation processes.