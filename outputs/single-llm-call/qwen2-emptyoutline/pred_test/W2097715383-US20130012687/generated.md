# DESCRIPTION

## BACKGROUND OF THE INVENTION

The field of recombinant protein purification is critical for various applications in biotechnology, structural biology, and proteomics. Traditional methods often rely on the addition of affinity tags such as poly-histidine (His6) or glutathione S-transferase (GST) to facilitate protein purification. However, these tags can sometimes interfere with the biological activity of the target protein and complicate downstream applications such as protein crystallization. To address these issues, researchers have developed fusion proteins where small proteins like NusA or SUMO are used to enhance solubility, expression, and stability of the target protein. Despite these advancements, the removal of these tags often requires the use of exogenous proteases, which can be costly, exhibit poor solubility, and necessitate additional purification steps.

The present invention introduces a novel one-step purification system that utilizes a site-specific affinity-tagged protease, specifically the Vibrio cholerae MARTX toxin cysteine protease domain (CPD). This system condenses affinity purification, cleavage, and tag separation into a single step, thereby simplifying protein purification procedures and increasing yields. The CPD is highly specific, inducible by inositol hexakisphosphate (InsP6), and exhibits poor transcleavage efficiency, making it an ideal candidate for this application.

## SUMMARY OF THE INVENTION

The present invention provides a method for the purification of recombinant proteins using a one-step on-bead cleavage system. The method involves the fusion of a target protein to a site-specific protease, specifically the Vibrio cholerae MARTX toxin cysteine protease domain (CPD), which is itself fused to an affinity tag, such as a His6 tag. The fusion protein is expressed in a bacterial host, purified using affinity chromatography, and then cleaved on the resin by the addition of InsP6. This cleavage releases the untagged target protein into the supernatant while the CPD remains immobilized on the resin.

The invention offers several advantages over existing methods:
1. **Simplicity**: The entire process, including purification, cleavage, and tag separation, is completed in a single step.
2. **Cost-Effectiveness**: The use of InsP6 as an inducer is more economical compared to exogenous proteases.
3. **High Specificity**: The CPD cleaves exclusively at the fusion protein junction, ensuring the integrity of the target protein.
4. **Enhanced Expression and Solubility**: The CPD fusion can increase the expression and solubility of target proteins, particularly those that are difficult to express or purify.
5. **Versatility**: The system can be adapted to various target proteins and can be used in both individual research labs and high-throughput settings.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### TERM DEFINITIONS

- **Target Protein**: A protein of interest that is to be purified.
- **Affinity Tag**: A peptide or protein sequence that facilitates the purification of the target protein, such as a His6 tag.
- **Cysteine Protease Domain (CPD)**: The protease domain from the Vibrio cholerae MARTX toxin, which is inducible by inositol hexakisphosphate (InsP6).
- **Fusion Protein**: A protein consisting of the target protein fused to the CPD and an affinity tag.
- **Inositol Hexakisphosphate (InsP6)**: A small molecule that specifically activates the CPD.
- **Immobilized Metal Affinity Chromatography (IMAC)**: A chromatographic technique that uses metal ions, such as nickel, to bind to affinity tags like His6.

### Methods

#### Construction of CPD Expression Vectors

To develop the one-step CPD purification system, CPD expression vectors were constructed using the pET expression vector backbone. The DNA encoding the CPD was cloned into the SalI restriction site of the pET22b vector, ensuring that the fusion protein produced upon IPTG induction of E. coli carries the P2-P1 residues of the native CPD (Ala-Leu) and the P4-P3 residues encoded by the SalI site (Val-Asp). This design ensures that the untagged target protein is released from the resin upon InsP6-induced cleavage, leaving the His6-tagged CPD bound to the Ni2+-NTA resin.

#### Expression and Purification of CPD Fusion Proteins

1. **Bacterial Growth Conditions**:
   - Overnight cultures of E. coli strains harboring the pET-CPD vectors were grown at 37°C in Luria-Bertani (LB) broth supplemented with 100 µg/mL carbenicillin.
   - The cultures were diluted 1:500 into 1 L of 2YT media and grown at 37°C until an OD600 of 0.6 was reached.
   - IPTG was added to a final concentration of 250 µM, and the cultures were grown for 3-4 hours at 30°C.

2. **Protein Expression**:
   - Cultures were harvested by centrifugation at 4°C and resuspended in lysis buffer (500 mM NaCl, 50 mM Tris-HCl, pH 7.5, 15 mM imidazole, 10% glycerol).
   - Cells were lysed by sonication, and the lysates were cleared by centrifugation at 15,000×g for 30 minutes.

3. **Affinity Purification**:
   - The cleared lysates were incubated with Ni-NTA Agarose beads (Qiagen) for 2-4 hours at 4°C to bind the His6-tagged CPD fusion proteins.
   - The beads were washed three times with lysis buffer to remove non-specifically bound proteins.

4. **On-Bead Cleavage**:
   - The washed Ni2+-NTA beads containing the CPD-His6 fusion proteins were resuspended in lysis buffer and InsP6 was added to a final concentration of 50-100 µM.
   - The cleavage reaction was allowed to proceed for 1-2 hours at room temperature or 4°C.
   - The beads were pelleted, and the supernatant containing the untagged target protein was collected.
   - The beads were washed 3-4 times with lysis buffer to ensure complete removal of the target protein.

5. **Elution of CPD**:
   - The His6-tagged CPD remaining on the beads was eluted using high imidazole buffer (500 mM NaCl, 50 mM Tris-HCl, pH 7.5, 175 mM imidazole, 10% glycerol).

#### Application to Various Target Proteins

1. **Green Fluorescent Protein (GFP)**:
   - The pET22b-GFP-CPD construct was expressed and purified using the described method.
   - Addition of InsP6 resulted in the dose-dependent release of GFP from the Ni2+-NTA resin, demonstrating the feasibility of the system.

2. **Intracellular Domain of gp130 (ICD)**:
   - The pET22b-gp130(ICD)-CPD construct was expressed and purified.
   - Autoprocessing occurred exclusively at the ICD-CPD interdomain junction, releasing the untagged ICD into the supernatant.

3. **Biotin Ligase (BirA)**:
   - The pET22b-BirA-CPD construct was expressed and purified.
   - The CPD fusion increased BirA expression levels by three-fold compared to the GST-BirA construct.

4. **Plasmodium falciparum SENP1 (PfSENP1)**:
   - The pET22b-PfSENP1-CPD construct was expressed and purified.
   - The CPD system enhanced the expression and purity of PfSENP1, facilitating the rapid purification of untagged PfSENP1 suitable for crystallization.

5. **CRAC Activation Domain (CAD) of STIM1**:
   - The pET22b-STIM1(CAD)-CPD construct was expressed and purified.
   - The CPD fusion protected the CAD from proteolytic degradation, enabling the large-scale expression and purification of this important regulatory domain.

6. **Mouse Macrophage Metalloelastase (MMP12)**:
   - The pET22b-mMMP12-CPD construct was expressed and purified.
   - The CPD system increased the solubility of MMP12, allowing for the rapid purification of soluble, active MMP12 in approximately 7 hours.

#### Advantages and Applications

The CPD-based one-step purification system offers several advantages over traditional methods:
- **Simplified Workflow**: The entire purification process, including cleavage and tag separation, is completed in a single step.
- **Cost-Effective**: The use of InsP6 as an inducer is more economical compared to exogenous proteases.
- **High Specificity**: The CPD cleaves exclusively at the fusion protein junction, ensuring the integrity of the target protein.
- **Enhanced Expression and Solubility**: The CPD fusion can increase the expression and solubility of target proteins, particularly those that are difficult to express or purify.
- **Versatility**: The system can be adapted to various target proteins and can be used in both individual research labs and high-throughput settings.

This novel purification system is expected to have widespread utility in biological research, facilitating the rapid and efficient purification of recombinant proteins for various applications, including structural biology, proteomics, and biotechnology.