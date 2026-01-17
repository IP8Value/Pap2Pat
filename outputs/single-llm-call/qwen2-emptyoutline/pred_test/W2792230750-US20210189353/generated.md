# DESCRIPTION

## DESCRIPTION

### STATE OF THE ART

Amadoriases, also known as Fructosyl Amino Acid Oxidases (FAAOX), are a class of enzymes primarily found in fungi and bacteria. These enzymes are capable of cleaving low molecular weight Amadori products, such as glycated amino acids, to yield a free amine, glucosone, and hydrogen peroxide. The reaction mechanism involves the oxidation of the C-N bond between the nitrogen of the amino acid moiety of the Amadori product and C1 of the fructosyl portion, resulting in the formation of a Schiff base which is subsequently hydrolyzed to produce glucosone and a free amino acid. The reduced FAD is then oxidized by an oxygen molecule, releasing hydrogen peroxide.

Members of this enzyme class, commonly referred to as Fructosyl Peptide Oxidases (FPOX), are widely used in the detection of glycated hemoglobin (HbA1c), a long-term marker for diabetes. The detection process typically involves a proteolytic digestion of HbA1c, which releases single amino acids, including the N-terminal glycation-prone valine. The FPOX enzyme then binds to glycated valine and hydrolyzes it, producing hydrogen peroxide, which is measured in a colorimetric assay using horseradish peroxidase and a suitable chromophore. Similarly, Amadoriase I has potential applications in the detection of glycated albumin, a short to mid-term glycemic marker for diabetes.

However, a common challenge in the use of biosensors is the long-term stability of their biological components. Stabilized Amadoriase enzymes can significantly improve biosensor stability during transport and storage, thereby extending their shelf life. Additionally, Amadoriase enzymes are considered promising therapeutic tools for the prevention or reduction of protein glycation in biological tissues. Glycation, a spontaneous, non-enzymatic, and irreversible reaction between a sugar moiety and a protein, leads to the formation of covalent adducts that modify the chemistry of functional proteins. This cascade of adverse clinical outcomes includes arterial stiffening, atherosclerosis, nephropathy, retinopathy, and neuropathy. Despite their potential, the use of Amadoriases to prevent protein glycation is limited by their lack of significant activity on intact proteins due to the buried active site location and the narrow tunnel that provides access to the catalytic pocket.

Another potential application of thermostable Amadoriases is in the prevention of acrylamide formation in processed food. Thermal treatments used in food manufacturing, such as baking, toasting, frying, and roasting, accelerate the Maillard reaction between reducing sugars and amino acids, which imparts desirable flavors and aromas. However, the reaction between sugars and asparagine amino acids yields acrylamide, a carcinogenic compound. This is particularly relevant in fried potatoes, bakery products, and coffee, where there are currently no viable strategies to mitigate acrylamide formation while preserving the desired properties of processed food. A thermostable Amadoriase enzyme could serve as a potential tool to limit the Maillard reaction on single amino acids and reduce acrylamide formation.

### SUMMARY OF THE INVENTION

The present invention relates to a method for the rational design and stabilization of Amadoriase I, a deglycating enzyme, to enhance its thermal stability. The method involves a high-throughput computational screening approach based on Molecular Dynamics (MD) simulations to identify and produce thermostable mutants of Amadoriase I. The invention specifically discloses the identification, production, and enzymatic characterization of four Amadoriase I mutants, two of which exhibit a remarkable increase in thermal stability compared to the wild-type enzyme.

The computational design and screening method utilizes the SSBOND software to generate a list of potential disulfide bond sites. MD simulations are then performed to evaluate the stability of the wild-type enzyme and the proposed mutants at different temperatures. The root mean square fluctuation (RMSF) is calculated to determine the average flexibility of the protein at various temperatures, and the slope (λ) of the avg-RMSF versus temperature is used as a proxy for enzyme stability. Mutants with a lower λ value compared to the wild-type enzyme are selected for experimental production and characterization.

The invention further discloses the production and purification of the selected mutants, as well as their biochemical and structural characterization. The selected mutants, SS03 and SS17, show a significant increase in thermal stability, with SS17 retaining residual activity even at 95°C. The crystal structures of the SS03 and SS17 mutants confirm the formation of the designed disulfide bonds and demonstrate that the overall fold of the enzyme remains similar to the wild-type, with minimal structural perturbations in the catalytic pocket.

### DETAILED DESCRIPTION OF THE INVENTION

#### Computational Design and Screening of Disulfide Bonds

The invention begins with the identification of potential disulfide bond sites using the SSBOND software. The crystal structure of wild-type Amadoriase I (PDB code: 4WCT) serves as the template. The software generates a list of 19 possible disulfide bond sites, and molecular models of the wild-type and the 19 Amadoriase I variants are built. Each variant features a different disulfide bond, and MD simulations are performed to evaluate their stability at three different temperatures (270, 300, and 340 K).

The RMSF is calculated for the Cα atoms of the protein from residue 10 to 437, excluding the N- and C-termini due to their intrinsic high mobility. The avg-RMSF is then calculated at each temperature, and the slope (λ) of the avg-RMSF versus temperature is used to discriminate between stabilized and destabilized mutants. Mutants with a higher λ value compared to the wild-type enzyme are discarded, while those with a lower λ value are selected for experimental production.

#### Production, Purification, and Biochemical Characterization of the Amadoriase Variants

The selected mutants (SS03, SS07, SS11, and SS17) are produced and purified using standard molecular biology techniques. The point mutations required for the double-cysteine mutants are introduced by PCR in the pET3a vector using the QuikChange II site-directed mutagenesis kit. The resulting mutant plasmids are validated by DNA sequencing, and the proteins are expressed in E. coli BL21(DE3)pLysS cells. The expressed proteins are purified using Ni-NTA affinity chromatography and size exclusion chromatography.

The purified mutants are characterized for their enzymatic activity, thermal stability, and pH activity profiles. The SS03 and SS17 mutants show a significant increase in thermal stability, with SS17 retaining residual activity even at 95°C. The SS07 mutant, however, shows a decreased stability and catalytic efficiency, suggesting a detrimental effect of the introduced disulfide bond on the enzyme's structure and function. The SS11 mutant is poorly expressed and shows no binding to the FAD cofactor, indicating that the introduced mutations may affect the enzyme's folding.

#### Crystal Structures

The crystal structures of the SS03 and SS17 mutants are determined using X-ray crystallography. The structures are solved at high resolution (2.15 Å for SS03 and 2.85 Å for SS17) and confirm the formation of the designed disulfide bonds. The overall fold of the mutants is very similar to the wild-type enzyme, with minimal structural perturbations in the catalytic pocket. However, the SS17 mutant shows a significant conformational change in the region where the mutations were introduced, affecting the conformation of a loop that defines the boundaries of the tunnel leading to the catalytic pocket.

#### Applications and Future Directions

The stabilized Amadoriase I mutants, particularly SS17, have potential applications in the development of improved biosensors for the detection of diabetes, therapeutic tools for the prevention of protein glycation, and as a tool for the reduction of acrylamide formation in food processing. The increased thermal stability of these mutants makes them suitable for use in high-temperature processes, such as those involved in milk pasteurization and ultra-heat treatment.

Future work will focus on testing the computational screening method on other types of mutations and extending the validation of the screening strategy to other enzymes. Additionally, efforts will be made to introduce large yet sustainable structural changes in the enzyme to enhance its access to the catalytic site, which is necessary for its use in vivo to prevent protein glycation.