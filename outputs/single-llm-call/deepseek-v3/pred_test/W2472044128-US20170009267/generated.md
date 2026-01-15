Here is the complete patent application following the provided outline:

# DESCRIPTION  

## CROSS-REFERENCES AND RELATED APPLICATIONS  

This application claims priority to Chinese Patent Application No. [Application Number], filed [Filing Date], the contents of which are incorporated herein by reference in their entirety.  

## BACKGROUND OF THE INVENTION  

The present invention relates to the field of genetic engineering and microbial production of biochemical compounds. More specifically, it pertains to improved methods for producing N-acetylglucosamine (GlcNAc) through metabolic engineering of Bacillus subtilis.  

GlcNAc is an amino sugar that serves as a fundamental building block in many biological processes. It is a key component of bacterial cell walls, fungal cell walls, and human connective tissues. GlcNAc has wide applications in the pharmaceutical, nutraceutical, and cosmetic industries. Currently, GlcNAc is primarily produced through chemical synthesis from chitin or fermentation processes using engineered microorganisms.  

Existing microbial production methods suffer from several limitations. Conventional fermentation processes often exhibit low yields due to competing metabolic pathways and regulatory mechanisms that limit precursor availability. Additionally, many production strains experience growth inhibition when overproducing GlcNAc, resulting in reduced volumetric productivity. The current methods also frequently require complex media components, increasing production costs and complicating downstream purification processes.  

There exists a significant need for improved GlcNAc production methods that overcome these limitations. An ideal production system would achieve high yields while maintaining robust microbial growth in minimal media. The present invention addresses these needs through novel genetic modifications that eliminate metabolic bottlenecks in GlcNAc biosynthesis.  

## DETAILED DESCRIPTION  

The present invention provides a genetically modified Bacillus subtilis strain and associated methods for efficient GlcNAc production. The key innovation involves disrupting a previously unidentified futile cycle that was found to limit GlcNAc production in engineered strains.  

The invention begins with a base production strain designated BSGN6-PxylA-glmS-P43-GNA1 (abbreviated BSGN). This strain was engineered to overexpress glucosamine-6-phosphate synthase (GlmS) under the control of the inducible PxylA promoter and GlcN-6-phosphate N-acetyltransferase (Gna1) under the constitutive P43 promoter. Additionally, genes involved in GlcNAc catabolism (nagP, gamP, nagA, nagB, and gamA) were knocked out to prevent GlcNAc degradation.  

A critical discovery underlying the present invention was the identification of an ATP-dissipating futile cycle between GlcNAc-6-phosphate (GlcNAc6P) and GlcNAc. Through comprehensive metabolomic analysis and kinetic modeling, it was determined that the glucokinase GlcK, while primarily known for glucose phosphorylation, also phosphorylates GlcNAc, creating this energy-wasting cycle. This cycle was found to accumulate GlcNAc6P to toxic levels (over 300-fold higher than wild-type), impairing both cell growth and GlcNAc production.  

To overcome this limitation, the invention provides a modified strain (designated BSGNK) in which the glcK gene encoding glucokinase is knocked out. This genetic modification eliminates the futile cycle, resulting in several significant improvements:  

1. Intracellular GlcNAc6P levels are reduced to wild-type concentrations (0.06 mM compared to 33.71 mM in the parent strain)  
2. The specific growth rate more than doubles compared to the parent strain  
3. GlcNAc productivity increases by over 100% (9.20 mg/l/h compared to 4.35 mg/l/h)  
4. The GlcNAc yield on glucose improves 2.3-fold (147.5 mg/g glucose compared to 65.0 mg/g)  
5. Cellular energy charge increases from 0.68 to 0.81  

The production method involves culturing the engineered strain in a synthetic minimal medium containing glucose as the primary carbon source. The medium composition includes mineral salts, nitrogen sources, and trace elements, but avoids complex organic components, making the process cost-effective and suitable for industrial scale-up.  

Key advantages of the invention include:  
- Elimination of a major metabolic bottleneck that previously limited GlcNAc production  
- Improved cell growth and viability during production  
- Higher product yields and volumetric productivities  
- Reduced production costs through use of minimal media  
- Simplified downstream processing due to cleaner fermentation broths  

### EXAMPLES  

**Materials and Methods**  

Strain construction was performed using standard molecular biology techniques. The BSGN strain was derived from Bacillus subtilis 168 through sequential genetic modifications. The glcK knockout in BSGNK was achieved using a markerless genome editing system with appropriate selection protocols.  

**Seed Medium**  

The seed culture medium contained:  
- 10 g/l tryptone  
- 5 g/l yeast extract  
- 10 g/l NaCl  
- Appropriate antibiotics for selection  

**Fermentation Medium**  

The production medium consisted of M9 minimal salts:  
- 1 g/l NH4Cl  
- 0.5 g/l NaCl  
- 8.5 g/l Na2HPO4·H2O  
- 3 g/l KH2PO4  
- 1 ml/l 1 M MgSO4  
- 1 ml/l 0.1 M CaCl2  
- 1 ml/l 0.05 M FeCl3  
- 10 ml/l trace element solution (60 mg/l CoCl2·6H2O, 43 mg/l CuCl2·2H2O, 100 mg/l MnCl2·4H2O, 60 mg/l Na2MoO4·2H2O, 170 mg/l ZnCl2)  
- 20 g/l glucose  

**Cultivation Conditions**  

Cultures were grown at 37°C with agitation at 300 rpm. The pH was maintained at 7.0 through automatic addition of NaOH or HCl as needed. Dissolved oxygen was maintained above 30% saturation through adjustable aeration and agitation.  

**Knockout of glcK Gene**  

The glcK gene was inactivated using a markerless deletion strategy. Homologous recombination was employed to replace the glcK coding sequence with a selectable marker, which was subsequently excised to leave a clean deletion. Successful knockout was verified by PCR and sequencing.  

**Fermentation of GlcNAc**  

The production strain was first grown in seed medium to mid-exponential phase, then transferred to fermentation medium at an initial OD600 of 0.1. GlcNAc production was monitored over 48 hours through regular sampling and HPLC analysis.  

**Results of Fermentation**  

The BSGNK strain demonstrated superior performance compared to the parent BSGN strain:  
- Maximum specific growth rate: 0.35 h-1 (compared to 0.15 h-1)  
- Final GlcNAc titer: 4.2 g/l (compared to 1.8 g/l)  
- GlcNAc yield on glucose: 147.5 mg/g (compared to 65.0 mg/g)  
- Volumetric productivity: 9.20 mg/l/h (compared to 4.35 mg/l/h)  

These results demonstrate the significant improvements achieved through the genetic modifications of the present invention.