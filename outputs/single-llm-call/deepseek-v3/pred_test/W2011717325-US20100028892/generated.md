Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates to the field of molecular biology and drug discovery, specifically to novel genetic systems for monitoring protease activity in living cells. Proteases play critical roles in numerous biological processes including cell division, cell death, differentiation, immunity, and protein trafficking. Consequently, proteases represent important therapeutic targets for various diseases including cancer, neurodegeneration, inflammation, and infectious diseases.  

Current high-throughput screening (HTS) methods for identifying protease inhibitors face several limitations. Many assays require purified proteases, which can be challenging to produce in sufficient quantities and may lack stability. Furthermore, these assays typically only identify compounds that directly inhibit the protease active site, missing opportunities to discover allosteric modulators or compounds targeting upstream activators. The similarity of active sites among protease families also makes achieving inhibitor selectivity difficult.  

Caspases serve as a prime example where improved screening methods are needed. As intracellular cysteine proteases, caspases participate in complex proteolytic networks involving upstream initiators and downstream effectors. Initiator caspases typically require multi-protein complexes for activation, which are difficult to reconstitute in vitro. Given their roles in apoptosis and inflammation, caspases represent attractive drug targets, but current screening methods fail to capture the complexity of their native activation pathways.  

There exists a significant need for alternative screening approaches that can reconstitute complete protease activation pathways, enable identification of compounds targeting upstream regulators, and provide context-dependent selectivity. The present invention addresses these needs through novel yeast-based genetic systems that faithfully reproduce mammalian protease activation networks while enabling high-throughput screening applications.  

## SUMMARY OF THE INVENTION  

The invention provides novel genetic systems for monitoring protease activity in living yeast cells. These systems employ cleavable reporter gene activators where protease-mediated cleavage releases a transcription factor that activates reporter genes. The technology has been demonstrated for all ten human caspases and for autophagins, showing broad applicability across protease families.  

In one embodiment, the invention provides a Single Component system where an active protease is expressed along with a membrane-tethered transcription factor containing a protease-specific cleavage sequence. Cleavage releases the transcription factor, activating LEU2 and lacZ reporter genes. This system has been configured for caspases 1-10, with each caspase showing expected substrate specificity.  

Another embodiment provides a Two-Component system where an inactive pro-protease is co-expressed with its activator protein. Careful engineering of expression levels is achieved through: (1) using promoters of varying strengths; (2) employing low versus high copy plasmids; and (3) adjusting numbers of transcription factor binding sites in reporter genes. Exemplary two-component systems include pro-caspase-1 with ASC, pro-caspase-2 with RAIDD, and pro-caspase-9 with Apaf-1.  

A further embodiment provides a Plural-Component system that reconstitutes entire mammalian protease activation pathways. One example reconstitutes the extrinsic apoptosis pathway with Fas, FADD, and pro-caspase-8. Another reconstitutes the NLRP1 inflammasome with NLRP1ΔLRR, ASC, and pro-caspase-1. These systems enable screening for compounds targeting any component of the pathway.  

The invention further provides methods for chemical library screening using these genetic systems. The assays have been adapted to 384-well format with Z' factors >0.6, suitable for high-throughput screening. Automated screening processes include robotic liquid handling and data analysis pipelines. Optimization methods determine ideal screening concentrations and conditions.  

Additional embodiments include methods for identifying caspase activation protease networks through cDNA library screening. The systems have successfully identified known activators like DR4, DR5, and c-FLIP, validating the approach. The technology also enables discovery of novel protease regulators through functional screening.  

## DETAILED DESCRIPTION OF THE INVENTION  

Proteolytic processing represents an irreversible post-translational modification central to numerous biological processes. Proteases participate in cell division, death, differentiation, immunity, and protein trafficking, making them attractive therapeutic targets. However, current high-throughput screening methods for protease inhibitors face significant limitations. Producing and purifying sufficient quantities of active proteases remains challenging. Many assays only identify compounds that directly inhibit the protease active site, missing opportunities to discover allosteric modulators or compounds targeting upstream regulators. The similarity of active sites among protease families also complicates achieving inhibitor selectivity.  

These limitations highlight the need for alternative screening approaches that can: (1) identify both inhibitors and activators; (2) capture the complexity of native activation pathways; (3) enable discovery of compounds targeting upstream regulators; and (4) provide context-dependent selectivity. The present invention addresses these needs through reconstitution of complete protease activation networks in yeast.  

### Caspases  

Caspases represent a family of intracellular cysteine proteases conserved throughout the animal kingdom. The human genome encodes at least ten caspases, which can be categorized as upstream initiators or downstream effectors. Initiator caspases (e.g., caspases-1, -2, -4, -5, -8, -9, -10) typically contain protein-protein interaction domains (CARD or DED) in their N-terminal prodomains and are activated through multi-protein complexes. Effector caspases (e.g., caspases-3, -6, -7) are activated through proteolytic cleavage by initiator caspases.  

All caspases cleave after aspartic acid residues, with each family member showing distinct substrate specificities. The three-dimensional structures of most human caspases have been determined, revealing conserved active site geometries with variations in surrounding amino acids that confer specificity. Activity profiles have been characterized for most human caspases, along with known cleavage sites in endogenous protein substrates.  

### Substrate Specificity of Caspases  

Caspase substrate specificity derives from interactions between the protease active site and four amino acids (P4-P1) N-terminal to the cleavage site. Structural studies reveal that while the active site geometry is conserved, variations in surrounding residues create distinct specificity pockets. For example, caspase-1 prefers WEHD, caspase-3 prefers DEVD, and caspase-8 prefers LETD. These specificity profiles have been extensively characterized through peptide library screens and structural analyses.  

### Mechanisms of Activating Upstream Initiator Caspases  

Initiator caspases are activated through distinct mechanisms involving multi-protein complexes. In the extrinsic apoptosis pathway, death receptors like Fas recruit the adapter FADD through death domain interactions. FADD's death effector domain (DED) then recruits caspases-8 or -10, promoting their oligomerization and autoproteolytic activation.  

The intrinsic apoptosis pathway involves Apaf-1, the human homolog of C. elegans CED-4. Cytochrome c released from mitochondria binds Apaf-1, relieving autoinhibition and allowing oligomerization. The oligomerized Apaf-1 recruits and activates caspase-9 through CARD-CARD interactions.  

Inflammatory caspases like caspase-1 are activated through inflammasomes - complexes containing NLR family proteins (e.g., NLRP1, NLRP3) and the adapter ASC. These pathways demonstrate the complex protein networks required for initiator caspase activation.  

### Phenotypes of Animal Caspases Expressed in Yeast  

Expression of animal caspases in yeast has revealed several important phenomena. Many caspases can autoactivate when overexpressed, likely due to spontaneous dimerization. Some caspases inhibit yeast growth or cause lethality at high expression levels. These observations informed the development of cleavable reporter systems where caspase expression is carefully titrated to achieve activation without growth inhibition.  

## EXAMPLE 1  

### Development and Testing of a Cleavable Reporter Gene System for Assaying Protease Activity in Living Yeast  

A reporter system was developed where a Fas transmembrane domain is fused to a transcription factor (LexA-B42) through a caspase-cleavable linker. Various tetrapeptide sequences (e.g., WEHD for caspase-1, DEVD for caspase-3) were tested. Control constructs replaced the cleavage site aspartate with glycine.  

The system was tested in yeast strains EGY48 and EGY191 containing LEU2 and lacZ reporter genes. Caspases 1-10 were expressed from plasmids using ADH or TEF promoters. Caspases-8 and -10 required co-expression with FADD to avoid growth inhibition.  

Results showed perfect correlation between expected and observed cleavage specificities. For example, WEHD was cleaved by caspases-1, -4, -5 but not others, while DEVD was cleaved by caspases-2, -3, -7. Non-cleavable controls and active site mutants failed to activate reporters, validating the system.  

### Substrate Specificities of the Caspases Expressed in Yeast  

Versions of the Fas-LexA-B42 reporter containing different cleavage sites (WEHD, DEVD, TEVD, LETD, LEHD) were tested with caspases 1-10. Each caspase showed expected preferences, with inflammatory caspases (1,4,5) cleaving WEHD but not TEVD, and effector caspases (3,6,7) cleaving DEVD but not initiator caspase sites.  

### Development of Two-component Systems Demonstrating Function of Caspase Activators in Yeast  

Two-component systems were developed where pro-caspases are co-expressed with their activators. Examples include:  
- Pro-caspase-1 + ASC  
- Pro-caspase-2 + RAIDD  
- Pro-caspase-8 + FADD  

In each case, both components were required for reporter activation. Expression levels were carefully controlled using weak promoters (CYC1) for pro-caspases and strong promoters (TEF) for activators.  

### Development of a Plural-component System—Mammalian Protease Activating Pathways Reconstituted in Yeast  

Two three-component systems were reconstituted:  
1. Fas/FADD/pro-caspase-8 (extrinsic pathway)  
2. NLRP1ΔLRR/ASC/pro-caspase-1 (inflammasome)  

Both systems showed strict dependence on all three components for reporter activation. The Fas/FADD/caspase-8 system was used to screen cDNA libraries, successfully identifying known activators like DR4 and DR5.  

## Experimental Designs & Methods Used  

### Vectors Encoding Caspase Cleavable Transcription Factors  

Reporters were constructed by PCR and recombinant DNA techniques. Fas extracellular/transmembrane domains were fused to LexA-B42 through caspase cleavage sites (e.g., GWEHDG). Control constructs replaced the cleavage aspartate with glycine.  

### Plasmids for Expressing Caspases and Their Upstream Activators  

Plasmids were derived from pRS series vectors. Expression levels were controlled by:  
- Promoter strength (CYC1 < ADH < TEF)  
- Plasmid copy number (CEN/ARS vs 2µ)  
- Number of LexA operators in reporters  

### Reporter Gene Assays  

Transformants were plated on leucine-deficient medium with X-gal. Activation was assessed by growth and blue color development. Liquid assays in 384-well plates measured β-galactosidase activity at 620 nm.  

### HTS Assays  

Assays were performed in 384-well format with 2×105 cells/well in raffinose media. Z' factors >0.6 were achieved for both Fas/FADD/caspase-8 and NLRP1/ASC/caspase-1 systems. A pilot screen of 1280 compounds yielded hits with expected specificity profiles.  

This detailed description provides enabling disclosure for practicing the invention, including specific embodiments, experimental results, and methods. The examples demonstrate utility across multiple protease families and screening applications.