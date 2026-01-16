Here is the complete patent application following the provided outline and research paper:

---

**DESCRIPTION**

### STATE OF THE ART  

Amadoriases, also known as Fructosyl Amino Acid Oxidases (FAOX), constitute a class of enzymes predominantly found in fungi and bacteria. These enzymes catalyze the cleavage of low molecular weight Amadori products, specifically glycated amino acids, to yield a free amine, glucosone, and hydrogen peroxide. In contemporary applications, variants of these enzymes, referred to as Fructosyl Peptide Oxidases (FPOX), are utilized in diagnostic kits for detecting glycated hemoglobin (HbA1c), a critical long-term marker for diabetes. The operational mechanism of these kits involves an initial proteolytic digestion of HbA1c to release glycated amino acids, followed by enzymatic oxidation by FPOX to produce hydrogen peroxide, which is subsequently quantified via a colorimetric assay.  

Despite their utility, a significant limitation of these biosensors is the inherent instability of their biological components during storage and transport, which adversely affects shelf-life and performance. Furthermore, Amadoriases have been explored as potential therapeutic agents for mitigating protein glycation in biological tissues. Glycation, a non-enzymatic and irreversible reaction between sugars and proteins, contributes to various pathological conditions, including arterial stiffening, atherosclerosis, and nephropathy. However, the therapeutic application of Amadoriases is hindered by their inability to act on intact proteins due to structural constraints, such as a buried active site and a narrow access tunnel.  

Additionally, Amadoriases hold promise in the food industry for reducing acrylamide formation, a carcinogenic byproduct of the Maillard reaction during high-temperature food processing. Current strategies to mitigate acrylamide formation are limited, and the development of thermostable Amadoriases could provide a viable solution. However, existing enzyme stabilization methods, such as directed evolution or rational design, are labor-intensive, costly, and often yield unpredictable outcomes. Thus, there remains a pressing need for an efficient and reliable method to engineer thermostable Amadoriase variants with retained or enhanced catalytic activity.  

### SUMMARY OF THE INVENTION  

The present invention addresses the aforementioned limitations by disclosing a novel computational method for the rational design of thermostable Amadoriase I mutants. The method employs Molecular Dynamics (MD) simulations to screen a library of potential disulfide bond mutations, enabling the identification of stabilizing mutations without extensive experimental validation. Specifically, the invention describes the design, production, and characterization of four Amadoriase I mutants (SS03, SS07, SS11, and SS17), two of which (SS03 and SS17) exhibit significantly enhanced thermal stability compared to the wild-type enzyme.  

The SS17 mutant, in particular, demonstrates an 8°C increase in the T50 value (the temperature at which the enzyme loses 50% of its activity) and retains detectable activity even after exposure to 95°C. Structural analysis via X-ray crystallography confirms the formation of the designed disulfide bonds in the SS03 and SS17 mutants without compromising the overall enzyme fold or active site architecture. The invention further encompasses applications of these stabilized enzymes in diabetes diagnostics, therapeutic protein deglycation, and food processing to reduce acrylamide formation.  

### DETAILED DESCRIPTION OF THE INVENTION  

**Computational Design and Screening of Disulfide Bonds**  
The invention utilizes a high-throughput computational screening approach to identify stabilizing disulfide bonds in Amadoriase I. The wild-type enzyme structure (PDB: 4WCT) was analyzed using the SSBOND software, which predicted 19 potential disulfide bond sites based on geometric and energetic criteria. Molecular models of each mutant were subjected to MD simulations at three temperatures (273K, 300K, and 340K) to evaluate their thermal stability. The Root Mean Square Fluctuation (RMSF) of each variant was calculated, and the average RMSF (avg-RMSF) was plotted against temperature. The slope (λ) of this linear relationship served as a stability index, with mutants exhibiting lower λ values than the wild-type being selected for experimental validation.  

Four mutants (SS03, SS07, SS11, and SS17) were identified as promising candidates. The SS03 mutant features a disulfide bond between residues S67 and P121, while the SS17 mutant introduces a disulfide bond between D295 and K303. Notably, the SS11 mutant, despite computational predictions, failed to express functionally due to misfolding, highlighting the method's ability to filter out unstable variants early in the design process.  

**Production and Biochemical Characterization**  
The selected mutants were expressed in *E. coli* and purified to homogeneity. Spectroscopic analysis confirmed the binding of the FAD cofactor in all variants except SS11, which exhibited no enzymatic activity. Steady-state kinetic assays revealed that the SS03 and SS17 mutants retained catalytic efficiency comparable to the wild-type enzyme, whereas the SS07 mutant showed reduced activity, likely due to steric hindrance near the substrate tunnel entrance.  

Thermal stability assays demonstrated that the SS03 and SS17 mutants exhibited T50 values of 55.25°C and 60.62°C, respectively, compared to 52.40°C for the wild-type. Remarkably, the SS17 mutant retained 50% activity at 60°C and displayed residual activity up to 95°C. Reduction of the disulfide bonds with DTT abolished the stabilizing effects, confirming their role in thermostability.  

**Structural Analysis**  
X-ray crystallography of the SS03 and SS17 mutants confirmed the formation of the designed disulfide bonds without significant alterations to the global enzyme fold. The SS03 mutant (2.15 Å resolution) superimposed closely with the wild-type structure (RMSD = 0.268 Å), while the SS17 mutant (2.85 Å resolution) exhibited minor conformational changes in a loop region distal to the active site. Importantly, neither mutation perturbed the catalytic pocket, ensuring retained substrate specificity and activity.  

**Applications**  
The stabilized Amadoriase I mutants of the invention have broad utility:  
1. **Diabetes Diagnostics**: Enhanced shelf-life and stability under storage conditions improve the reliability of HbA1c and glycated albumin detection kits.  
2. **Therapeutic Deglycation**: The SS17 mutant serves as a platform for further engineering to enable activity on intact glycated proteins, addressing unmet needs in diabetic complications.  
3. **Food Industry**: Thermostable variants can be incorporated into food processing workflows to reduce acrylamide formation during high-temperature treatments (e.g., frying, roasting).  

**Advantages Over Prior Art**  
The invention's computational screening method significantly reduces the experimental burden associated with enzyme stabilization. By leveraging MD simulations to predict stabilizing mutations, the approach minimizes false positives and accelerates the development of robust biocatalysts. In contrast to directed evolution, which requires extensive library screening, the disclosed method enables targeted design with higher success rates and lower costs.  

--- 

This patent application provides a comprehensive description of the invention, its technical advantages, and potential applications while adhering to formal patent language and structure. Let me know if you would like any modifications or additional details.