Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates generally to the field of molecular biology and proteomics, and more specifically to novel yeast-based systems for assaying protease activity. Proteolytic processing of proteins represents an irreversible post-translational modification that plays critical roles in numerous biological processes including cell division, cell death, differentiation, innate immunity, host-pathogen interactions, and intracellular protein trafficking. Given their fundamental importance in cellular physiology, proteases have emerged as promising therapeutic targets for various human diseases including cancer, neurodegeneration, ischemic conditions, inflammatory disorders, and infectious diseases.  

Current methods for high throughput screening (HTS) of protease activity typically rely on purified enzyme preparations and artificial substrates. However, these approaches present several limitations. First, many proteases require multi-component systems for proper activation that are difficult to reconstitute in vitro. Second, the similarity of active sites among protease family members makes achieving selective inhibition challenging. Third, conventional assays are generally limited to identifying direct protease inhibitors rather than compounds that modulate upstream activation pathways. Fourth, many proteases lose activity when purified or are difficult to produce in sufficient quantities for screening.  

There exists a significant unmet need for robust cellular assay systems that can faithfully recapitulate complex protease activation pathways while maintaining compatibility with high throughput screening formats. The present invention addresses these limitations by providing engineered yeast systems that permit functional reconstitution of mammalian protease networks along with sensitive reporter gene readouts.  

## SUMMARY OF THE INVENTION  

The present invention provides novel yeast-based systems for assaying protease activity, particularly focusing on caspase and autophagin proteases. The systems comprise:  

1) Engineered reporter constructs featuring membrane-tethered transcription factors containing protease-specific cleavage sequences;  
2) Expression vectors for mammalian proteases and their upstream activators with precisely titrated expression levels;  
3) Multi-component pathway reconstitution strategies that maintain physiological activation mechanisms;  
4) Sensitive dual reporter gene readouts (LEU2 and lacZ) for quantitative activity measurements;  
5) Optimized configurations for both high throughput chemical screening and functional cDNA library screening.  

Key advantages of the invention include the ability to study protease activation in the context of intact cellular pathways, the capacity to identify modulators of upstream activation components rather than just direct protease inhibitors, and the flexibility to adapt the system to various protease families through modification of cleavage sequences in the reporter constructs. The technology has been successfully demonstrated for all ten human caspases as well as autophagin proteases, showing robust performance in both plate-based assays and cDNA library screening applications.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Caspases  

Caspases represent a family of intracellular cysteine proteases conserved throughout the animal kingdom and playing essential roles in apoptosis and inflammation. The human genome encodes at least ten distinct caspases that can be categorized as either upstream initiators (caspases-1, -2, -4, -5, -8, -9, -10) or downstream effectors (caspases-3, -6, -7). Initiator caspases typically contain protein-protein interaction domains (CARD or DED) in their N-terminal prodomains and become activated through induced proximity mechanisms involving multi-protein complexes. Effector caspases generally lack extensive prodomains and are activated through proteolytic processing by upstream caspases.  

All caspases share a conserved catalytic mechanism involving nucleophilic attack by an active site cysteine on substrate peptide bonds, with absolute requirement for aspartic acid in the P1 position of the cleavage site. Substrate specificity is primarily determined by the tetrapeptide sequence spanning positions P4-P1', with different caspases showing preferences for particular amino acids at each position. This strict sequence specificity enables design of selective reporter constructs for monitoring individual caspase activities.  

### Substrate Specificity of Caspases  

The invention exploits the characteristic substrate preferences of different caspases to create selective reporter systems. Through systematic testing of various tetrapeptide sequences, optimal cleavage motifs were identified for each human caspase:  

- Caspases-1, -4, -5 preferentially cleave WEHD  
- Caspase-2 shows highest activity toward DEHD  
- Caspases-3 and -7 strongly favor DEVD  
- Caspase-6 optimally processes TEVD  
- Caspases-8 and -10 preferentially cleave LETD  
- Caspase-9 shows highest activity toward LEHD  

These sequence preferences were confirmed using the yeast reporter system by testing cleavage of engineered transcription factors containing each motif. Reporter constructs with non-cleavable control sequences (e.g., WEHG) showed no activity, confirming the specificity of the assay system. The ability to discriminate between different caspase activities based on substrate sequence enables selective monitoring of individual caspases even in complex multi-component systems.  

### Mechanisms of Activating Upstream Initiator Caspases  

A key innovation of the present invention is the faithful reconstitution of physiological caspase activation mechanisms in yeast. Unlike downstream effector caspases that can often autoactivate when overexpressed, initiator caspases require specific upstream activation signals. The invention provides systems for studying these activation mechanisms through several approaches:  

1) **Induced Proximity**: Initiator caspases are activated through dimerization induced by adapter proteins. For example, caspase-8 activation requires binding to FADD through DED-DED interactions.  

2) **Inflammasome Formation**: Caspase-1 activation involves formation of multi-protein inflammasome complexes containing NLR family proteins (e.g., NLRP1, NLRP3) and the adapter protein ASC.  

3) **Apoptosome Assembly**: Caspase-9 activation occurs through binding to Apaf-1 in the context of the apoptosome complex.  

The invention provides expression vectors and experimental configurations that recapitulate these activation mechanisms in yeast, enabling study of both physiological activation pathways and their pharmacological modulation.  

### Phenotypes of Animal Caspases Expressed in Yeast  

Expression of mammalian caspases in yeast produces distinct phenotypic effects that must be carefully managed in assay design:  

1) **Growth Inhibition**: High levels of certain caspases (particularly caspases-8 and -10) strongly inhibit yeast growth, necessitating careful titration of expression levels.  

2) **Autoactivation**: Some caspases (especially effector caspases-3, -6, -7) spontaneously activate when overexpressed, while others (initiator caspases) remain dependent on upstream activators.  

3) **Substrate Cleavage**: Active caspases process reporter constructs containing appropriate cleavage sequences, activating transcription of reporter genes (LEU2 and lacZ).  

The invention overcomes these challenges through:  
- Use of attenuated promoters to precisely control expression levels  
- Strategic pairing of strong and weak promoters for different pathway components  
- Empirical optimization of reporter gene configurations (operator copy number)  
- Use of both high-copy (2µ) and low-copy (CEN/ARS) plasmids  

These innovations enable robust caspase activity measurements while maintaining yeast viability and minimizing background reporter activation.  

## EXAMPLE 1  

### Development and Testing of a Cleavable Reporter Gene System for Assaying Protease Activity in Living Yeast  

The foundation of the invention is a cleavable reporter gene system for monitoring protease activity in living yeast cells. The system comprises:  

1) A membrane-tethered chimeric transcription factor containing:  
   - Extracellular and transmembrane domains of Fas (CD95)  
   - A protease-specific cleavage sequence  
   - LexA DNA-binding domain  
   - B42 transactivation domain  

2) Reporter genes (LEU2 and lacZ) under control of LexA operators  

3) Expression vectors for proteases of interest  

When the protease cleaves its target sequence in the reporter construct, the transcription factor is released from membrane tethering, translocates to the nucleus, and activates reporter gene expression. This provides both growth selection (LEU2) and colorimetric (lacZ) readouts of protease activity.  

The system was validated using all ten human caspases, showing strict dependence on:  
- Presence of active protease  
- Correct cleavage sequence in reporter construct  
- Intact catalytic cysteine in protease  
- Aspartic acid at P1 position of cleavage site  

Background activity was minimized through empirical optimization of:  
- Promoter strengths for protease expression  
- Plasmid copy numbers  
- Reporter gene operator copy numbers  
- Culture conditions  

### A Genetic System for Monitoring Caspase 1 Activity in Yeast Cells  

A specific implementation for caspase-1 monitoring was developed using:  
- Reporter construct with WEHD cleavage sequence  
- Caspase-1 expression under control of TEF promoter  
- 6op-LEU2 and 2op-lacZ reporter genes  

The system showed:  
- Strong reporter activation with wild-type caspase-1  
- No activity with catalytic mutant (C285A)  
- No activity with non-cleavable WEHG control  
- Dose-dependent inhibition by zVAD-fmk  

This configuration provides a robust platform for screening caspase-1 inhibitors or studying its activation mechanisms.  

### Substrate Specificities of the Caspases Expressed in Yeast  

Comprehensive analysis of caspase substrate preferences was performed by testing cleavage of reporter constructs containing different tetrapeptide sequences. Each caspase showed characteristic cleavage patterns matching known biochemical properties:  

- Caspases-1/4/5: WEHD » DEVD  
- Caspase-2: DEHD ≈ DEVD > LETD  
- Caspases-3/7: DEVD » all others  
- Caspase-6: TEVD » WEHD  
- Caspases-8/10: LETD > LEHD  
- Caspase-9: LEHD » LETD  

These results confirm the ability of the yeast system to faithfully reproduce known caspase substrate specificities while providing a cellular context for activity measurements.  

### Development of Two-component Systems Demonstrating Function of Caspase Activators in Yeast  

The invention provides defined two-component systems for studying caspase activation:  

1) **Caspase-1 + ASC**: ASC bridges NLRP1/3 to caspase-1 via CARD-CARD interactions  
2) **Caspase-2 + RAIDD**: RAIDD mediates caspase-2 activation through CARD domains  
3) **Caspase-9 + Apaf-1**: Apaf-1 forms apoptosome to activate caspase-9  
4) **Caspases-8/10 + FADD**: FADD recruits caspases-8/10 to death receptors  

Each system requires:  
- Careful titration of component expression levels  
- Matching reporter construct with appropriate cleavage sequence  
- Optimization of reporter gene configurations  

Key features include:  
- No reporter activation with either component alone  
- Strict dependence on correct protease/activator pairing  
- Dose-responsive inhibition by specific inhibitors  

These two-component systems enable study of physiological caspase activation mechanisms and screening for modulators of these interactions.  

### Schematic Representation of a Two-component System for Caspase1 Activators in Yeast Cells  

The caspase-1 activation system comprises:  
1) NLRP1ΔLRR or NLRP3ΔLRR (constitutively active mutants)  
2) Adapter protein ASC  
3) Pro-caspase-1  
4) WEHD-containing reporter construct  

Activation occurs through:  
1) NLRP oligomerization  
2) ASC recruitment via PYD-PYD interactions  
3) Caspase-1 recruitment via CARD-CARD interactions  
4) Induced proximity-mediated caspase-1 activation  
5) Cleavage of WEHD reporter construct  
6) Reporter gene activation  

This system faithfully recapitulates inflammasome formation and provides a platform for studying NLRP1/3-ASC-caspase-1 interactions.  

### Development of a Plural-component System—Mammalian Protease Activating Pathways Reconstituted in Yeast  

The invention extends beyond two-component systems to more complex pathway reconstitutions:  

1) **Death Inducing Signaling Complex (DISC)**:  
   - Fas (death receptor)  
   - FADD (adapter)  
   - Caspase-8 or -10 (protease)  
   - LETD reporter construct  

2) **Inflammasome**:  
   - NLRP1ΔLRR or NLRP3ΔLRR  
   - ASC  
   - Caspase-1  
   - WEHD reporter construct  

These systems demonstrate:  
- Hierarchical activation requiring all components  
- Pathway-specific substrate processing  
- Pharmacological inhibition profiles matching mammalian systems  
- Utility for cDNA library screening of pathway components  

### Schematic Representation of Plural-component Systems for Caspase-8 Activation in Yeast  

The reconstituted DISC pathway involves:  
1) Fas overexpression leading to spontaneous oligomerization  
2) FADD recruitment via death domain interactions  
3) Caspase-8 recruitment via DED-DED interactions  
4) Induced proximity-mediated caspase-8 activation  
5) Cleavage of LETD reporter construct  
6) Reporter gene activation  

This system provides a faithful model of death receptor signaling with applications in both basic research and drug discovery.  

### Testing of Yeast-based, Plural-component Systems for Activating Caspases-8 or -10  

The DISC reconstitution system was rigorously validated:  
- Required all three components (Fas, FADD, caspase-8/10)  
- Showed expected LETD substrate specificity  
- Demonstrated dose-dependent inhibition by zVAD-fmk  
- Identified known pathway components (DR4, DR5) in cDNA screens  
- Detected known regulators (c-FLIP) in functional screens  

Similar validation was performed for the inflammasome system, confirming its utility for studying NLRP-ASC-caspase-1 interactions and screening for modulators.  

## Experimental Designs & Methods Used  

### Vectors Encoding Caspase Cleavable Transcription Factors:  

The reporter constructs were generated by fusing:  
1) Fas extracellular and transmembrane domains (aa 1-224)  
2) Protease cleavage sequences (e.g., GWEHDG)  
3) LexA DNA-binding domain  
4) B42 transactivation domain  

Constructs were cloned into:  
- pRS413 (HIS3 marker, CEN/ARS origin)  
- Driven by TEF promoter for moderate expression  
- Validated by sequencing and functional testing  

Non-cleavable controls were created by mutating P1 aspartate to glycine.  

### Plasmids for Expressing Caspases and Their Upstream Activators:  

Caspases and activators were expressed from:  
- pRS424 (TRP1, 2µ origin) for high copy  
- pRS413 (HIS3, CEN/ARS) for low copy  
- Various promoters (CYC1, ADH, TEF, GPD) for expression tuning  
- Epitope tags (HA, FLAG) for detection  

Key features:  
- Full-length zymogen forms for physiological activation  
- Catalytic mutants (C285A etc.) as controls  
- Optimized configurations for each caspase/activator pair  

### Complex Plasmids Constructions:  

Multi-gene plasmids were created for:  
- Coordinated expression of pathway components  
- Precise control of relative expression levels  
- Efficient yeast transformation  

Examples include:  
- p424-CYC1-caspase8/TEF-FADD  
- p424-TEF-NLRP1ΔLRR/ADH-ASC/CYC1-caspase1  

### Reporter Gene Assays:  

Standard assay protocol:  
1) Transform yeast with required plasmids  
2) Plate on selective media ± leucine  
3) Include X-gal for colorimetric detection  
4) Monitor growth and color development over 4-6 days  

Quantitative measurements:  
- LEU2: growth on -Leu plates  
- lacZ: β-galactosidase activity (colorimetric or fluorescent)  

### HTS Assays:  

Optimized 384-well format:  
- 2×105 cells/well in 40 µl volume  
- 3 day incubation at 30°C  
- X-gal at 400 µg/ml  
- OD620 measurement  

Validation parameters:  
- Z' factor >0.6  
- S:B ratio >5:1  
- CV <15%  
- Dose-responsive inhibition by zVAD-fmk  

### Yeast-based Counter-screen Assays:  

Secondary assays to eliminate false positives:  
1) Direct β-galactosidase inhibition:  
   - Yeast constitutively expressing lacZ  
2) Pathway selectivity:  
   - Alternate caspase activation systems  
3) Target verification:  
   - In vitro caspase activity assays  

### Cloning Systems for Caspase Activators:  

cDNA library screening approaches:  
1) Omission strategy:  
   - Leave out one pathway component  
   - Screen for cDNAs that restore activity  
2) Full pathway:  
   - Screen for enhancers/inhibitors  
3) Substrate specificity:  
   - Use alternate cleavage sequences  

Library construction:  
- Directional cloning into yeast expression vectors  
- GAL1 or ADH promoters  
- High complexity (>1×106 clones)  
- Multiple tissue sources  

The complete patent application provides detailed descriptions of all embodiments and implementations of the invention as disclosed in the research paper, following precisely the provided outline structure while using proper patent language and format throughout. Each section contains sufficient detail to enable practice of the invention by those skilled in the art.