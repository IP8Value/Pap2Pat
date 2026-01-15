Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The purification of recombinant proteins represents a fundamental requirement for biochemical, structural, and pharmaceutical research. Conventional protein purification techniques often rely on affinity tags such as polyhistidine (His6) or glutathione S-transferase (GST) to facilitate isolation of target proteins from complex biological mixtures. However, these methods suffer from significant limitations, including low solubility of tagged proteins, interference of tags with protein function, and the necessity for additional enzymatic steps to remove tags post-purification.  

Existing tag removal strategies typically involve the addition of exogenous proteases such as thrombin or TEV protease, which introduce further complications. These proteases are expensive, exhibit variable activity under different buffer conditions, and may cleave target proteins at unintended secondary sites. Moreover, the requirement for subsequent purification steps to eliminate the exogenous protease increases processing time and reduces overall yield. There exists a pressing need for an improved protein purification system that combines high efficiency with minimal processing steps while maintaining target protein integrity.  

## SUMMARY OF THE INVENTION  

The present invention discloses a novel self-cleaving protease tag system for streamlined protein purification. At the core of this invention is the cysteine protease domain (CPD) derived from the Vibrio cholerae MARTX toxin, which serves as an autocatalytic fusion tag that can be selectively activated to cleave itself from the target protein. This system eliminates the need for exogenous proteases by incorporating the cleavage machinery directly into the fusion construct.  

The CPD tag provides multiple advantages over existing purification methods. First, it remains inactive during bacterial expression until specifically triggered by the small molecule inositol hexakisphosphate (InsP6). Second, the CPD cleaves with exceptional specificity at a defined leucine residue located at the junction between the target protein and the protease domain. Third, the system enables single-step purification whereby affinity capture, tag cleavage, and isolation of untagged target protein occur sequentially on the same resin matrix. Importantly, fusion to the CPD tag has been shown to enhance expression levels, solubility, and stability of diverse target proteins, addressing common challenges in recombinant protein production.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

### On-Bead Cleavage Purification System  

The invention provides an integrated on-bead cleavage purification system wherein a target protein is expressed as a fusion with the CPD followed by a His6 affinity tag. Following immobilization of the fusion protein on nickel-nitrilotriacetic acid (Ni-NTA) resin, addition of InsP6 induces autoprocessing at the target-CPD junction, releasing the untagged target protein into the supernatant while retaining the His6-tagged CPD on the resin. This approach condenses three purification steps - affinity capture, tag removal, and separation of cleaved components - into a single streamlined procedure.  

### Advantages of the System  

Key advantages of this system include its simplicity, cost-effectiveness, and high specificity. Unlike conventional methods requiring separate protease addition, the CPD system maintains a 1:1 stoichiometry between protease and substrate, ensuring efficient processing. The InsP6-inducible nature provides precise temporal control over cleavage initiation. Furthermore, the autoprocessing mechanism minimizes off-target cleavage events that could compromise target protein integrity.  

### Vibrio cholerae MARTX Toxin Cysteine Protease Domain (CPD)  

The CPD (amino acids 3440-3650 of V. cholerae MARTX toxin) serves as the catalytic core of this purification system. The CPD exhibits several unique properties that make it particularly suitable for this application: (1) strict specificity for cleavage after leucine residues; (2) absolute dependence on InsP6 for activation; and (3) preferential autoprocessing activity with minimal trans-cleavage capacity. Structural studies reveal that the P1 leucine fits precisely into the enzyme's S1 binding pocket, explaining the stringent sequence specificity.  

### Purification Method Using CPD-His6 Tag  

The standard purification protocol involves: (1) expression of CPD-His6 fusion proteins in E. coli; (2) affinity capture on Ni-NTA resin; (3) washing to remove contaminants; (4) InsP6-induced on-bead cleavage; and (5) collection of untagged target protein from the supernatant. The His6-tagged CPD remains bound to the resin and can be subsequently eluted with imidazole if desired. This method typically yields pure, untagged target protein within 4-6 hours from cell lysis.  

### Construction of pET Expression Vectors  

A series of pET-based expression vectors were engineered to accommodate different cloning strategies and cleavage outcomes. These include: pET-CPDSalI (introduces Val-Asp-Ala-Leu at the C-terminus of target protein); pET-CPDSacI (introduces Glu-Leu); and pET-CPDBamHI (introduces single Leu residue). Additional variants incorporate epitope tags such as HA between the target protein and CPD for specialized applications.  

### Feasibility Demonstration Using GFP  

The system's functionality was validated using green fluorescent protein (GFP) as a model substrate. GFP-CPD-His6 fusion protein was efficiently purified and released from Ni-NTA resin in an InsP6-dose-dependent manner. Intact GFP was recovered in the supernatant with no detectable cleavage at internal sites, while the CPD-His6 remained resin-bound. This confirmed the system's ability to produce pure, untagged target protein in a single step.  

### Fidelity of CPD-Mediated Processing  

To assess cleavage specificity, the intrinsically disordered intracellular domain (ICD) of gp130 containing multiple leucine residues was tested as a challenging substrate. Despite its unstructured nature and numerous potential cleavage sites, processing occurred exclusively at the designed ICD-CPD junction. This demonstrates the CPD's remarkable fidelity even with vulnerable target proteins.  

### Enhanced Expression and Solubility  

Unexpectedly, fusion to CPD-His6 consistently improved both expression levels and solubility of diverse target proteins. For example:  
- gp130(ICD) showed 3-fold higher expression as a CPD fusion compared to His6-tagged version  
- Biotin ligase (BirA) expression increased 3-fold versus GST-tagged construct  
- Soluble yield of mouse metalloelastase (MMP12) improved from <5% to >60% of total expressed protein  

### Protection from Proteolytic Degradation  

The CPD fusion system provided stabilization against proteolysis for sensitive domains like the STIM1 CRAC-activation domain (CAD), which proved difficult to purify using conventional methods. The CPD fusion approach enabled production of sufficient quantities for functional studies.  

### Comparison with Other Purification Systems  

The CPD system offers distinct advantages over alternative technologies:  
- Intein-based systems: Smaller tag size avoids solubility issues associated with large intein fusions  
- Sortase-based systems: Provides enhanced solubility and expression benefits lacking in sortase tags  
- ELP tags: Eliminates need for temperature or pH shifts that can destabilize proteins  

### Potential Applications  

This technology has broad applicability for:  
- High-throughput structural genomics  
- Production of therapeutic proteins  
- Proteomic studies requiring native protein forms  
- Commercial scale protein manufacturing  

### Generation of CPD Mutants  

The invention encompasses engineered CPD variants with altered properties, including:  
- Temperature-sensitive mutants for controlled activation  
- Modified specificity variants cleaving at alternative residues  
- Stability-enhanced mutants for harsh purification conditions  

## TERM DEFINITIONS  

### CPD  
The cysteine protease domain from Vibrio cholerae MARTX toxin (amino acids 3440-3650) that specifically cleaves after leucine residues upon InsP6 activation.  

### MARTX  
Multifunctional-autoprocessing repeats-in-toxin, a large virulence factor secreted by Vibrio cholerae containing the CPD domain.  

### Inducer  
Inositol hexakisphosphate (InsP6), the small molecule activator required for CPD protease activity.  

### FIG. 3  
Demonstrates dose-dependent release of GFP from Ni-NTA resin upon InsP6 addition.  

### FIG. 4  
Shows exclusive cleavage at the designed junction in gp130(ICD) fusion protein.  

### FIG. 5  
Compares enhanced expression of BirA as CPD versus GST fusion.  

### FIG. 6  
Illustrates improved SENP1 purification using CPD system versus traditional methods.  

### Tables  
Summarize quantitative improvements in expression levels, solubility, and purity for various target proteins.  

## METHODS  

### Bacterial Growth Conditions  
E. coli strains were grown in LB or 2YT media at 37°C with appropriate antibiotics. Protein expression was induced with 250 μM IPTG at OD600 0.6, typically followed by 3-4 hour incubation at 30°C.  

### Vector Construction  
All pET-CPD variants were constructed using standard molecular biology techniques. Key vectors include:  
- pET-CPDSalI: CPD cloned into SalI/XhoI sites of pET22b/pET28a  
- pET-HA-CPDSalI: Incorporates HA epitope tag upstream of CPD  
- pET-CPDBamHI: Enables minimal Leu residue addition to target protein  

### Protein Expression and Purification  
General protocol involves:  
1. Cell lysis in 500 mM NaCl, 50 mM Tris-HCl pH 7.5, 15 mM imidazole  
2. Batch binding to Ni-NTA resin for 2-4 hours at 4°C  
3. Washing with lysis buffer  
4. On-bead cleavage with 50-100 μM InsP6 for 1-2 hours  
5. Collection of untagged target protein in supernatant  

Special conditions were developed for challenging proteins:  
- MMP12: Included 1 mM TCEP in lysis buffer, expression at 16°C  
- SENP1: All steps performed at room temperature to prevent aggregation  

### Activity Assays  
Purified proteins were functionally validated:  
- GFP fluorescence measured at 511 nm excitation/488 nm emission  
- MMP12 activity monitored using fluorogenic substrate Mca-PLGLDL(Dpa)AR  

This comprehensive system represents a significant advance in recombinant protein purification technology, combining simplicity, efficiency, and enhanced protein quality in a single integrated platform.