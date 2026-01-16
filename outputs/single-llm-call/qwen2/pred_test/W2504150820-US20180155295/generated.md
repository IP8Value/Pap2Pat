# DESCRIPTION

## STATEMENT REGARDING FEDERAL FUNDING
This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Agency Name]. The government has certain rights in the invention.

## FIELD
The present invention relates to the field of pharmaceuticals and more particularly to the discovery and characterization of novel small-molecule CXCR4 receptor agonists and antagonists. The invention provides methods for identifying and optimizing these compounds, as well as their use in treating various diseases and conditions associated with CXCR4 receptor activity.

## BACKGROUND
Chemokine receptor type 4 (CXCR4) is a G-protein-coupled receptor (GPCR) that plays a crucial role in various physiological and pathological processes, including stem cell homing, cancer metastasis, and immune responses. The natural ligand for CXCR4 is the chemokine stromal cell-derived factor-1 (SDF-1, also known as CXCL12). Activation of CXCR4 by SDF-1 leads to the activation of multiple signaling pathways, including the mitogen-activated protein kinase (MAPK) pathway and the phosphoinositide 3-kinase (PI3K) pathway, which are involved in cell migration, proliferation, and survival.

Despite the importance of CXCR4 in various diseases, the development of small-molecule modulators of this receptor has been challenging due to the complexity and size of the binding site. Most known CXCR4 modulators are either peptides or large cyclic peptides, which are not ideal for drug development due to their poor pharmacokinetic properties. Therefore, there is a significant need for small-molecule CXCR4 agonists and antagonists that can be used as therapeutic agents and research tools.

## SUMMARY
The present invention provides novel small-molecule CXCR4 receptor agonists and antagonists, methods for their identification, and their use in treating various diseases and conditions. The invention includes a dual in silico screening strategy combining ligand-based and structure-based approaches to identify and optimize these compounds.

In one aspect, the invention provides a method for identifying small-molecule CXCR4 receptor agonists and antagonists, comprising:
1. Generating a low-energy conformer database of a compound library.
2. Building a common-feature pharmacophore model using a training set of known CXCR4 antagonists.
3. Screening the conformer database using the pharmacophore model to identify potential hits.
4. Performing structure-based virtual screening using docking tools to further refine the hits.
5. Evaluating the biological activity of the identified compounds using in vitro assays, such as calcium imaging, ERK activation, receptor internalization, and chemotaxis assays.

In another aspect, the invention provides novel small-molecule CXCR4 receptor agonists and antagonists, including but not limited to compounds NUCC-388, NUCC-390, NUCC-392, NUCC-397, NUCC-398, NUCC-54118, NUCC-54120, and NUCC-54121.

In yet another aspect, the invention provides methods for treating diseases and conditions associated with CXCR4 receptor activity, comprising administering an effective amount of a small-molecule CXCR4 receptor agonist or antagonist to a subject in need thereof.

## DEFINITIONS
For the purposes of this invention, the following terms are defined as follows:

- **CXCR4**: Chemokine receptor type 4, a G-protein-coupled receptor (GPCR) that binds the chemokine stromal cell-derived factor-1 (SDF-1 or CXCL12).
- **Agonist**: A substance that binds to a receptor and activates it, producing a biological response.
- **Antagonist**: A substance that binds to a receptor and blocks its activation, preventing a biological response.
- **Pharmacophore**: A molecular framework that carries the essential features responsible for a drug's biological activity.
- **Docking**: A computational method used to predict the preferred orientation of one molecule to another when bound to each other to form a stable complex.
- **Conformer**: A specific spatial arrangement of atoms in a molecule that results from rotation about one or more single bonds.
- **Virtual Screening**: A computational technique used to search libraries of small molecules in order to identify those structures which are most likely to bind to a given protein target.

## DETAILED DESCRIPTION
The present invention provides novel small-molecule CXCR4 receptor agonists and antagonists, methods for their identification, and their use in treating various diseases and conditions. The invention combines ligand-based and structure-based approaches to identify and optimize these compounds, ensuring a comprehensive and robust screening process.

### EXPERIMENTAL
The experimental methods used to identify and characterize the novel small-molecule CXCR4 receptor agonists and antagonists are described in detail below.

#### Annotated Database Creation
To generate the low-energy conformer database, the ChemBridge GPCR-focused library containing approximately 13,000 compounds was used. Low-energy conformers were generated using the ConFirm/CatConf module from the Catalyst program implemented in Discovery Studio 3.1. The "best" mode was used to generate conformers, and a modified version of the CHARMm force field was employed along with a poling technique that biased the sampling of conformations towards geometries that were far from a local minimum but energetically near each other. This method generated approximately 100 conformers of each compound within an energy cutoff of 10 kcal/mol.

#### Common-Feature Pharmacophore Model Building And Database Screening
A set of 162 CXCR4 antagonists with IC50 values in the range of 1 nM to 10 μM was selected from ChEMBL (version 13). Cluster analysis protocols implemented in Discovery Studio 3.1 were used to group the compounds into 5 different clusters. Two molecules from each cluster were selected to build a training set, and the common feature pharmacophore modeling tool was used to build a set of 10 pharmacophores, called "hypotheses." The default parameters were used to build the hypotheses. Another 2 molecules from each of the 5 clusters (10 molecules total) were used as a test set. One 5-point pharmacophore model consisting of two hydrophobic (Hy), two Hydrogen Bond Acceptor (HBA), and one Positive Ionizable (PI) feature was found to fit well to all 10 compounds of the test set. This pharmacophore model was selected for screening the annotated GPCR compound database. Database screening produced 26 structures with >85% fit values along with conformational energy less than 5 kcal/mol. Based on availability and synthetic tractability, 6 compounds were purchased from this ligand-based hit set.

#### Structure-Based Virtual Screen
The CXCR4 crystal structures (accession codes 3ODU and 3OE0) were analyzed, and the 16-residue cyclic peptide was observed to fill a large ligand binding site, while the small molecule IT1t only occupied a small part of the pocket. To obtain consensus binding poses with flexible ligand docking tools, two docking engines built upon orthogonal algorithms were used: the Surflex docking engine implemented in Sybyl-X and the Glide docking tool version 6.5. The Surflex docking engine is based on a fragment-based algorithm, while the Glide docking tool is based on a grid-based technique.

The small-molecule bound CXCR4 crystal structure (pdb code 3ODU) was validated using Prime version 3.8 to correct for irrelevant side chains, missing atoms, and undesired orientation of Asn, Gln, or His residues. The 'Prot-Prep' module was used to prepare and refine the co-crystal structure to generate the receptor (protein) and the bound ligand. A 12 Å³ grid box was generated using the centroid of the bound ligand to prepare for Glide docking.

For Surflex docking, the ligand (IT1t) was extracted from the co-crystal structure, and the protein was subjected to the protein preparation panel in the Sybyl interface. Hydrogens were added in hydrogen bonding orientation, b-values were replaced by the Gasteiger charges, irrelevant torsions were eliminated, and the protonation states of the residues were fixed at pH 7.4. A ligand-based protomol was generated in the active site, representing the template for an ideal active-site ligand.

The 20 reported antagonists (Training and Test sets) were docked using the Glide-XP module with the standard sampling mode of maxkeep = 5000 and maxref = 400. The van der Waals radii for nonpolar ligand atoms were scaled to 0.8. After docking, the docked poses of the 20 compounds were analyzed, and the interactions of the antagonists with different protein residues were noted. The Lig-Prep module of the Schrödinger suite was used to prepare the GPCR-focused library for docking. Using the same docking protocols, the library structures were docked into the ligand-binding site of CXCR4. Compounds showing a Glide score of < -6.0 were considered for further analysis. The interacting residues identified from the known antagonist set guided the analysis of the docked poses of the unknown compounds from the library. Based on the docked scores and the interactions with critical residues, 52 compounds were selected as in silico hits.

A similar approach was used for the Surflex docking experiment. The 20 antagonist set was docked in the earlier-defined ligand-binding site of CXCR4 using the default set of run-time parameters and the GeomX docking mode. After docking the known antagonists, the docked poses were analyzed, and the critical interacting residues of the CXCR4 active site were identified. The GPCR ligand set was prepared using the ligand preparation panel implemented in the Sybyl interface. Using similar docking protocols, the library was docked, and 48 compounds showed good interactions with active site residues and had a total score > 6.0, where the total score is a function of -logKd. Twenty-two compounds with similar binding poses and favorable interactions with the active site residues were found in common between the Glide and Surflex docking experiments. These in silico hits underwent further evaluation for the presence of potentially toxic or metabolically unstable groups, reactive functional groups, non-drug-like features, synthetic feasibility, structural diversity, and commercial availability. Based on these criteria, 9 of these structure-based virtual hits were purchased.

#### Library Screening
The 15 vHTS hits were assayed at an initial single screening concentration of 10 μM using a calcium imaging assay. The aggressive human melanoma cell line C8161, which expresses numerous human CXCR4 receptors, was used. Cells were stimulated twice with SDF-1, resulting in two (Ca)i responses of similar magnitude, indicating little desensitization. To test a drug, the compound in question was added prior to the second stimulation with SDF-1. Several compounds showed significant biological activity, with some acting as antagonists and others as agonists. Antagonists NUCC-388, 392, 397, and 54120 had IC50 values of 0.3 μM, 3 μM, 1 μM, and 1 μM, respectively. Agonists NUCC-390, 398, 54118, 54121, and 54127 produced responses similar to those of SDF-1 and were inhibited by the selective CXCR4 antagonist AMD3100.

#### Calcium Imaging Assay
The calcium imaging assay was used to examine the activity of different molecules. The assay is based on the fact that activation of CXCR4 receptors produces an increase in the intracellular free Ca²⁺ concentration (Ca)i. This signal can be observed using a fluorescent Ca²⁺ sensing dye such as fura-2. The quantitative nature of this assay makes it ideal for screening purposes. The assay can also distinguish potential antagonists from potential agonists. The aggressive human melanoma cell line C8161, which expresses numerous human CXCR4 receptors, was used. Cells were usually stimulated twice with SDF-1, resulting in two (Ca)i responses of similar magnitude, indicating little desensitization. To test a drug, the compound in question was added prior to the second stimulation with SDF-1. At this point, it was possible to observe whether the compound itself acted as an agonist by giving its own response or if it reduced the magnitude of the second response to SDF-1.

#### Erk Activation By Agonist Nucc-390
To further explore the agonist potential of compound NUCC-390, changes in signaling downstream of CXCR4 were examined. Lysates from treated C8161 cells were collected and analyzed using Western blot. Activation of the CXCR4 receptor has been shown to indirectly mediate phosphorylation of ERK, a key signaling molecule in the MAP kinase pathway. As expected, cells treated with SDF-1 for 30 minutes displayed increased levels of phosphorylated ERK (pERK). Treatment with drug NUCC-390 also led to increased levels of pERK, further supporting the observation that NUCC-390 acts as a CXCR4 agonist.

#### Nucc-390 Induces Internalization Of Cxcr4 Receptors
Another characteristic feature of CXCR4 receptors and many other GPCRs is receptor internalization following agonist stimulation. To determine if NUCC-390 exhibited the ability to induce CXCR4 receptor internalization, the cellular localization of YFP-tagged CXCR4 receptors expressed in HEK293 cells following treatment with SDF-1 or NUCC-390 was assessed. Non-treated cells showed some diffuse expression of CXCR4-YFP throughout the cytosol and clear expression in the cell membrane. Treatment with SDF-1 for a period of 2 hours led to pronounced internalization of CXCR4-YFP, producing noticeable aggregates of the receptors in the cytosol but excluded from the nucleus. Similar effects were produced by NUCC-390. The effects of NUCC-390 were completely inhibited by AMD-3100 or NUCC-388, suggesting that NUCC-390 acts as a CXCR4 agonist.

#### Sdf-1 And Nucc-390 Mediate Chemotaxis
Chemokines are well known for their ability to stimulate chemotaxis of leukocytes and stem cells. To further establish the biological activity of the novel CXCR4 agonists, the ability of SDF-1 and NUCC-390 to produce chemotaxis of C8161 cells was compared using a Boyden chamber assay. SDF-1 produced robust chemotactic activity, which was matched by the effects of NUCC-390, demonstrating that this novel agonist can produce one of the major biological effects of chemokines.

#### 125I-Sdf-1Α Binding To The Cxcr4 Receptor
The interaction of NUCC-390 with CXCR4 receptors was assessed by examining the binding of 125I labelled SDF-1α to CXCR4 receptors in human Chem-1 cells. NUCC-390 showed no significant ability to inhibit binding of 125I-SDF-1α to CXCR4 in concentrations up to 10⁻⁵ M, indicating that its site of interaction is not identical to that of SDF-1.

### EXAMPLE 2
This example demonstrates the use of the novel small-molecule CXCR4 receptor agonist NUCC-390 in a calcium imaging assay to confirm its agonist activity.

### EXAMPLE 3
This example illustrates the use of the novel small-molecule CXCR4 receptor antagonist NUCC-388 in a calcium imaging assay to confirm its antagonist activity.

### EXAMPLE 4
This example shows the activation of ERK by the novel small-molecule CXCR4 receptor agonist NUCC-390, further confirming its agonist activity.

### EXAMPLE 5
This example demonstrates the induction of CXCR4 receptor internalization by the novel small-molecule CXCR4 receptor agonist NUCC-390.

### EXAMPLE 6
This example compares the chemotactic activity of the novel small-molecule CXCR4 receptor agonist NUCC-390 with that of SDF-1 using a Boyden chamber assay.

### EXAMPLE 7
This example assesses the binding of 125I labelled SDF-1α to CXCR4 receptors in the presence of the novel small-molecule CXCR4 receptor agonist NUCC-390 to confirm its non-competitive binding mechanism.