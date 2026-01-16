Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The purification of recombinant proteins is a fundamental requirement for modern biological research, pharmaceutical development, and industrial biotechnology applications. Conventional protein purification strategies often rely on affinity tags such as polyhistidine (His6) or glutathione S-transferase (GST) sequences fused to target proteins, which enable selective binding to immobilized metal ions or glutathione resins, respectively. However, these tags frequently impair protein solubility, alter biological activity, or interfere with structural studies such as X-ray crystallography. Consequently, removal of these tags prior to downstream applications is often necessary.  

Current methods for tag removal involve the addition of exogenous proteases such as thrombin, factor Xa, or tobacco etch virus (TEV) protease, which recognize specific cleavage sites engineered between the tag and target protein. These proteases suffer from several limitations, including high cost, inefficient cleavage kinetics, sensitivity to buffer conditions, and non-specific cleavage of target proteins at secondary sites. Furthermore, the proteases themselves must be subsequently removed from the purified target protein through additional chromatography steps, increasing processing time and reducing final yields.  

There exists an unmet need for an improved protein purification system that combines high-yield affinity purification with precise, inducible tag removal in a single step, while simultaneously enhancing target protein solubility and stability during expression. The present invention addresses these needs through an innovative fusion tag system based on the Vibrio cholerae MARTX toxin cysteine protease domain (CPD), which enables simplified one-step purification of untagged recombinant proteins with superior efficiency and specificity compared to existing methods.  

## SUMMARY OF THE INVENTION  

The present invention provides a novel recombinant protein purification system comprising: (a) a fusion protein construct containing a target protein operably linked to an affinity-tagged Vibrio cholerae MARTX toxin cysteine protease domain (CPD); (b) methods for expressing said fusion protein in host cells; (c) methods for purifying said fusion protein using affinity chromatography; and (d) methods for inducing specific autocleavage of the fusion protein to release the untagged target protein while retaining the protease domain on the affinity resin.  

Key advantages of the invention include:  

1. The CPD fusion system enables single-step purification and tag removal through inducible autocleavage, eliminating the need for exogenous proteases and subsequent purification steps.  

2. The CPD is specifically activated by inositol hexakisphosphate (InsP6), allowing precise temporal control over cleavage while maintaining protease inactivity during bacterial expression.  

3. The CPD exhibits exquisite specificity for cleavage at a single leucine residue positioned at the fusion junction, preventing non-specific cleavage of target proteins.  

4. Fusion to CPD enhances expression levels, solubility, and stability of diverse target proteins compared to conventional His6 or GST tags.  

5. The system accommodates flexible cloning strategies allowing control over the number of residual amino acids remaining on the target protein after cleavage.  

The invention further provides expression vectors, host cells, and optimized protocols for implementing this purification system across a wide range of applications from basic research to industrial-scale protein production.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

### TERM DEFINITIONS  

As used throughout this specification, the following terms shall have the meanings specified:  

"CPD" refers to the cysteine protease domain derived from Vibrio cholerae MARTX toxin, comprising amino acids 3440-3650 of the full-length toxin or functional variants thereof retaining inducible autocleavage activity.  

"InsP6" refers to inositol hexakisphosphate, the specific small molecule activator of CPD autocleavage.  

"Fusion junction" refers to the peptide bond between the C-terminal residue of the target protein and the N-terminal residue of the CPD in the fusion construct.  

"Autocleavage" refers to the intramolecular proteolytic activity of the CPD resulting in specific cleavage at the fusion junction upon InsP6 activation.  

"Affinity tag" refers to a peptide sequence enabling purification by binding to a specific ligand, including but not limited to polyhistidine (His6), glutathione S-transferase (GST), or chitin-binding domains.  

### Methods  

The present invention provides comprehensive methods for the construction, expression, purification, and cleavage of CPD fusion proteins, as detailed below:  

**Vector Construction**  
The invention encompasses a series of expression vectors derived from pET plasmid backbones containing the CPD sequence cloned adjacent to multiple restriction enzyme sites (SalI, SacI, BamHI) to allow flexible fusion strategies. These vectors incorporate an N-terminal target protein cloning site and a C-terminal His6 tag for affinity purification. Specific embodiments include:  

1. pET-CPDSalI: Allows fusion leaving Val-Asp-Ala-Leu residues on the target protein  
2. pET-CPDSacI: Allows fusion leaving Glu-Leu residues on the target protein  
3. pET-CPDBamHI: Allows fusion leaving a single Leu residue on the target protein  
4. pET-HA-CPDSalI: Incorporates an HA epitope tag for detection  

**Protein Expression**  
The method comprises transforming appropriate expression vectors into E. coli host cells, culturing the cells to mid-log phase (OD600 ≈ 0.6), and inducing fusion protein expression with isopropyl β-D-1-thiogalactopyranoside (IPTG). Optimal expression conditions vary by target protein, with typical parameters including:  
- Induction temperature: 16-30°C  
- Induction duration: 3-8 hours  
- Culture media: 2YT or LB broth with appropriate antibiotics  

**Affinity Purification**  
The method comprises lysing cells by sonication in buffer containing 500 mM NaCl, 50 mM Tris-HCl (pH 7.5), 15 mM imidazole, and 10% glycerol, followed by clarification by centrifugation. The supernatant is incubated with Ni2+-NTA agarose beads for 2-4 hours at 4°C with gentle agitation. Beads are washed with lysis buffer to remove unbound proteins.  

**Inducible Cleavage**  
The method comprises resuspending beads containing immobilized fusion protein in lysis buffer and adding InsP6 to a final concentration of 50-100 μM. Cleavage proceeds for 1-2 hours at 4-25°C with gentle agitation. Key parameters include:  
- InsP6 concentration: 50-100 μM optimal  
- Temperature: 4-37°C effective  
- Time: 1-2 hours for complete cleavage  
- Buffer compatibility: Tolerant of various salts and additives  

Following cleavage, the supernatant containing the untagged target protein is separated from beads retaining the His6-tagged CPD. The target protein may be further purified by gel filtration chromatography if required.  

**Applications and Examples**  
The invention has been successfully applied to purify diverse proteins including:  
1. Green fluorescent protein (GFP) - Demonstrated dose-dependent InsP6 activation  
2. gp130 intracellular domain - Validated cleavage specificity with unstructured protein  
3. Biotin ligase (BirA) - Showed 3-fold increased expression versus GST fusion  
4. Plasmodium falciparum SENP1 - Achieved high purity without thrombin cleavage  
5. STIM1 CAD domain - Enabled purification of previously unstable protein  
6. MMP12 metalloprotease - Facilitated soluble expression of normally insoluble protein  

The CPD fusion system provides particular advantages for:  
- Proteins prone to insolubility or aggregation  
- Proteins requiring tag-free forms for structural studies  
- High-throughput purification pipelines  
- Proteins from difficult-to-express organisms  
- Industrial-scale protein production  

The complete system including vectors, host strains, and protocols represents a significant advance over existing protein purification technologies, combining the benefits of affinity tags with the convenience of inducible, specific tag removal in a single streamlined process.