# DESCRIPTION

## CROSS-REFERENCES AND RELATED APPLICATIONS

This application claims the benefit of U.S. Provisional Application No. 63/XXXXXXX, filed [DATE], which is hereby incorporated by reference in its entirety.

## BACKGROUND OF THE INVENTION

The present invention relates to the field of biotechnology and, more specifically, to the production of N-acetylglucosamine (GlcNAc) using engineered bacterial strains. GlcNAc is a valuable compound with numerous applications in pharmaceuticals, cosmetics, and food industries. Traditional methods of GlcNAc production, such as chemical synthesis and extraction from natural sources, are often costly and environmentally unfriendly. Therefore, there is a significant need for efficient and sustainable biotechnological processes to produce GlcNAc.

Recent advances in metabolic engineering have enabled the development of microbial strains capable of overproducing GlcNAc. However, these strains often face challenges related to low productivity and poor cell growth, particularly when grown in minimal media. One such challenge is the accumulation of GlcNAc-6-phosphate (GlcNAc-6-P), which can lead to metabolic imbalances and reduced cell viability. This invention addresses these issues by identifying and disrupting a futile cycle involving GlcNAc-6-P and GlcNAc, thereby enhancing the productivity and yield of GlcNAc in engineered bacterial strains.

## DETAILED DESCRIPTION

The present invention provides an engineered bacterial strain, specifically *Bacillus subtilis*, that is capable of efficiently producing N-acetylglucosamine (GlcNAc) in minimal glucose medium. The strain, designated as BSGNK, is derived from a previously engineered strain (BSGN) by disrupting a futile cycle involving GlcNAc-6-P and GlcNAc. This disruption significantly improves the growth rate and GlcNAc productivity of the strain.

### Construction of the Engineered Strain

The starting strain, BSGN, was previously constructed by overexpressing glucosamine-6-phosphate synthase (GlmS) under the control of an inducible promoter (PxylA) and GlcN-6-phosphate N-acetyltransferase (Gna1) under the control of a constitutive promoter (P43). Additionally, the strain lacks the genes involved in GlcNAc catabolism (nagP, gamP, nagA, nagB, and gamA) to prevent the breakdown of GlcNAc. Despite these modifications, BSGN exhibited a significantly reduced growth rate and GlcNAc productivity in minimal glucose medium.

To address these issues, we identified a metabolic bottleneck involving the accumulation of GlcNAc-6-P. Metabolomics analysis revealed that the high intracellular concentration of GlcNAc-6-P was due to a futile cycle where GlcNAc is re-phosphorylated to GlcNAc-6-P. This futile cycle consumes ATP and impairs cell growth and GlcNAc production.

To disrupt this futile cycle, we deleted the gene encoding glucose kinase (glcK) in BSGN, resulting in the strain BSGNK. The deletion of glcK eliminated the re-phosphorylation of GlcNAc, thereby reducing the intracellular concentration of GlcNAc-6-P and alleviating metabolic stress. As a result, BSGNK exhibited a more than doubled specific cell growth rate and a more than doubled GlcNAc productivity compared to BSGN.

### Metabolic Engineering Strategies

1. **Overexpression of Key Enzymes**: The strain BSGN was constructed by overexpressing GlmS and Gna1 to enhance the flux through the GlcNAc synthesis pathway. The use of inducible and constitutive promoters allows for precise control of enzyme expression levels.

2. **Knockout of Catabolic Genes**: The deletion of nagP, gamP, nagA, nagB, and gamA prevents the breakdown of GlcNAc, ensuring that the produced GlcNAc is not consumed by the cell.

3. **Disruption of the Futile Cycle**: The deletion of glcK in BSGN to create BSGNK eliminates the re-phosphorylation of GlcNAc, reducing the accumulation of GlcNAc-6-P and improving cellular energy balance.

### Metabolomics and Dynamic Analysis

To understand the metabolic limitations in BSGN, we performed steady-state and dynamic metabolomics analyses. These analyses revealed that the intracellular concentrations of key precursors (fructose-6-P, acetyl-CoA, and glutamine) were not limiting factors for GlcNAc production. Instead, the high concentration of GlcNAc-6-P suggested a metabolic bottleneck.

Dynamic metabolomics experiments further confirmed the presence of a futile cycle involving GlcNAc-6-P and GlcNAc. Isotopic tracer experiments using [U-13C]glucose demonstrated that the initial source of accumulating GlcNAc-6-P was primarily from unlabelled GlcNAc, indicating re-phosphorylation. The deletion of glcK in BSGNK abolished the formation of unlabelled GlcNAc-6-P and increased the formation of fully labelled GlcNAc-6-P, confirming the disruption of the futile cycle.

### Performance of the Engineered Strain

The engineered strain BSGNK exhibited several improvements over BSGN:

1. **Increased Growth Rate**: The specific cell growth rate of BSGNK was more than doubled compared to BSGN, indicating improved cellular health and metabolism.

2. **Enhanced GlcNAc Productivity**: The GlcNAc productivity of BSGNK was more than doubled, reaching 9.20 mg l⁻¹ h⁻¹. The GlcNAc yield on glucose was 2.3-fold higher in BSGNK compared to BSGN.

3. **Improved Energy Metabolism**: The energy charge in BSGNK increased from 0.68 ± 0.03 in BSGN to 0.81 ± 0.04, further confirming the positive impact of disrupting the futile cycle on cellular energy homeostasis.

### Applications and Advantages

The engineered strain BSGNK offers several advantages for the industrial production of GlcNAc:

1. **Cost-Effectiveness**: The use of minimal glucose medium reduces the overall production cost compared to complex media.

2. **Sustainability**: The strain is designed to minimize waste and maximize resource utilization, making it an environmentally friendly option for GlcNAc production.

3. **Scalability**: The strain can be easily scaled up for large-scale production, making it suitable for industrial applications.

4. **Versatility**: The strain can be further optimized by incorporating additional metabolic engineering strategies to further enhance GlcNAc production.

### EXAMPLES

#### Example 1: Construction of BSGN and BSGNK Strains

**Materials and Methods**

- **Strains and Plasmids**: The strains and plasmids used in this study are listed in Supplementary Table 6. The previously constructed GlcNAc production strain BSGN is characterized by (i) a block of GlcNAc catabolism through marker-free deletion of all relevant encoding genes and (ii) overexpression of the GlcNAc synthesis enzymes GlmS and Gna1.

- **Construction of BSGN-Pfk***: BSGN-Pfk* was constructed by introducing a site-directed mutation in the native pfk gene (Arg252Ala) via a markerless genome editing system. The front and back homology fragments with the mutation were amplified using primers AL-F/AL-R and AR-F/AR-R, respectively. The mazF cassette was amplified using primers AZ-F and AZ-R from the B. subtilis 168 genome. Fusion PCR was performed to fuse the front homology fragment, mazF cassette, and back homology fragment. The resulting DNA fragment was transformed into BSGN0, and transformants with the desired mutation were screened on LB plates with 2% xylose.

- **Construction of BSGN-GS**: BSGN-GS was constructed by overexpressing glutamine synthase (GS) in BSGN. The encoding sequences of GNA1 and GS were amplified using primers GNA1-F/GNA1-R and GS-F/GS-R, respectively. The vector sequence was amplified using primers V-F and V-R. The resulting encoding sequences of GNA1 and GS were fused via fusion PCR. Prolonged-overlap extension PCR was then performed to generate DNA multimer plasmids, which were transformed into BSGN0, yielding BSGN-GS.

- **Construction of BSGNK**: BSGNK was constructed by knocking out the glucose kinase encoding gene (glcK) in BSGN. The glcK disrupt cassette was amplified using primers GlcK-F/GlcK-R from B. subtilis 168 ΔglcK and transformed into BSGN, yielding BSGNK.

**Results**

- **Growth and GlcNAc Production**: The growth rate and GlcNAc productivity of BSGN, BSGN-Pfk*, BSGN-GS, and BSGNK were evaluated in minimal glucose medium. BSGN exhibited a significantly reduced growth rate and GlcNAc productivity compared to the wild-type strain. The introduction of the Pfk* mutation and overexpression of GS did not improve growth or GlcNAc production. However, the deletion of glcK in BSGNK resulted in a more than doubled specific cell growth rate and a more than doubled GlcNAc productivity.

#### Example 2: Metabolomics Analysis

**Materials and Methods**

- **Steady-State Metabolomics**: B. subtilis strains were cultured in M9 medium and harvested in the mid-exponential phase. Fast-filtration was used to collect cells, and the cells were quenched and extracted in acetonitrile/methanol/H2O (40:40:20) solution with 13C internal standard addition. The extract solution was dried and resuspended in H2O for UHPLC-MS/MS detection.

- **Metabolite Dynamics Analysis**: Cells were cultivated in LB medium and harvested via centrifugation. The cells were resuspended in M9 medium without glucose and incubated at 37°C with magnetic stirring. At t=0, glucose was added to a final concentration of 2 g l⁻¹. Samples were taken at various time points and immediately quenched and extracted for UHPLC-MS/MS detection.

- **Dynamic Labelling Experiment**: 100% [U-13C]glucose was used as the substrate. Data of mass isotopomers GlcNAc6P M+0 and M+8 were acquired via Xcalibur software version 2.07 SP1 (Thermo Fisher Scientific).

**Results**

- **Intracellular Metabolite Concentrations**: Steady-state metabolomics analysis revealed that the intracellular concentrations of key precursors (fructose-6-P, acetyl-CoA, and glutamine) were not limiting factors for GlcNAc production in BSGN. However, the concentration of GlcNAc-6-P was significantly higher in BSGN compared to the wild-type strain.

- **Metabolite Dynamics**: Dynamic metabolomics experiments showed that the concentrations of fructose-6-P and glutamine increased rapidly upon glucose addition, confirming the absence of precursor limitations. The concentration of GlcNAc-6-P equilibrated quickly, suggesting a metabolic bottleneck.

- **Isotopic Tracer Analysis**: The dynamic labelling experiment using [U-13C]glucose demonstrated that the initial source of accumulating GlcNAc-6-P was primarily from unlabelled GlcNAc, indicating re-phosphorylation. The deletion of glcK in BSGNK abolished the formation of unlabelled GlcNAc-6-P and increased the formation of fully labelled GlcNAc-6-P, confirming the disruption of the futile cycle.

#### Example 3: Dynamic Simulation

**Materials and Methods**

- **Model Development**: A linear pathway model was developed to simulate the dynamics of the GlcNAc synthesis pathway. The model consisted of four intracellular metabolites (x(1)–x(4)) and the extracellular product (x(5)). Reaction kinetics were described using Michaelis-Menten kinetics.

- **Simulation Scenarios**: The model was used to simulate different scenarios, including feedback inhibition, a limiting enzyme abundance, a futile cycle, and insufficient intracellular product exportation.

**Results**

- **Base Model**: The base model showed a continuous increase in the product and asymptotic equilibration of intermediates at the average Km of one.

- **Feedback Inhibition**: Simulations with feedback inhibition showed equilibration of intermediates at average inhibition constants and reduced overproduction.

- **Limiting Enzyme Abundance**: Simulations with a limiting enzyme abundance showed a continuously accumulating intermediate upstream of the bottleneck and a strong decrease in the downstream intermediate.

- **Futile Cycle**: Simulations with a futile cycle showed dynamics similar to a limiting reaction, with intermediates equilibrating faster than in the case of a limiting reaction.

- **Insufficient Intracellular Product Exportation**: Simulations with insufficient intracellular product exportation showed an accumulation of the intracellular product.

These simulations provided insights into the potential causes of the observed metabolic limitations and supported the hypothesis of a futile cycle involving GlcNAc-6-P and GlcNAc.

### Conclusion

The present invention provides an engineered *Bacillus subtilis* strain, BSGNK, that is capable of efficiently producing N-acetylglucosamine (GlcNAc) in minimal glucose medium. By disrupting a futile cycle involving GlcNAc-6-P and GlcNAc, BSGNK exhibits improved growth rate and GlcNAc productivity compared to the parental strain BSGN. This invention offers a cost-effective and sustainable method for the industrial production of GlcNAc, with potential applications in pharmaceuticals, cosmetics, and food industries.