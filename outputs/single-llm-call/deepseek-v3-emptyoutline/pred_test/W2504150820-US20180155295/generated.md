Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT REGARDING FEDERAL FUNDING  

The invention described herein was not made with federal funding.  

## FIELD  

The present invention relates to the field of chemokine receptor modulation, specifically to novel small molecule compounds that function as agonists or antagonists of the CXCR4 receptor, methods for their identification through computational screening techniques, and their use in therapeutic applications.  

## BACKGROUND  

Chemokine receptors play critical roles in immune system regulation, stem cell migration, and cancer metastasis. The CXCR4 receptor, in particular, has been implicated in numerous physiological and pathological processes. Current CXCR4-targeting therapeutics are limited to antagonists such as AMD3100 (plerixafor), which is used for hematopoietic stem cell mobilization. There remains an unmet need for small molecule agonists of CXCR4 that could provide novel therapeutic approaches for conditions where CXCR4 activation is beneficial.  

Existing methods for identifying CXCR4 modulators rely primarily on high-throughput screening of compound libraries using biological assays. These approaches are costly, time-consuming, and often fail to identify drug-like molecules with suitable pharmacological properties. The development of computational screening methods combining both ligand-based and structure-based approaches could significantly improve the efficiency of identifying novel CXCR4 modulators with diverse pharmacological profiles.  

## SUMMARY  

The present invention provides novel small molecule compounds that modulate CXCR4 receptor activity, including both agonists and antagonists. The invention further encompasses methods for identifying these compounds through an integrated computational screening approach combining:  

1) Creation of an annotated database of compounds with calculated low-energy conformations;  
2) Development of a common-feature pharmacophore model based on known CXCR4 antagonists;  
3) Ligand-based virtual screening using the pharmacophore model;  
4) Structure-based virtual screening using multiple docking algorithms; and  
5) Biological validation of identified hits through calcium imaging, ERK activation, receptor internalization, and chemotaxis assays.  

The invention specifically discloses compounds including NUCC-390, NUCC-388, NUCC-397, NUCC-392, and NUCC-51420, which demonstrate either agonist or antagonist activity at CXCR4 receptors. These compounds represent the first known small molecule agonists of CXCR4 and provide new tools for studying CXCR4 biology and developing therapeutics for conditions involving CXCR4 signaling.  

## DEFINITIONS  

For purposes of the present invention, the following terms shall have the meanings specified:  

"CXCR4 agonist" refers to a compound that activates CXCR4 receptor signaling, as demonstrated by at least one of: calcium mobilization, ERK phosphorylation, receptor internalization, or chemotaxis in cells expressing CXCR4.  

"CXCR4 antagonist" refers to a compound that inhibits CXCR4 receptor signaling induced by the natural ligand SDF-1, as demonstrated by inhibition of calcium mobilization in cells expressing CXCR4.  

"Pharmacophore model" refers to an abstract representation of molecular features necessary for biological activity, including but not limited to hydrogen bond acceptors, hydrophobic regions, and charged groups.  

"Virtual screening" refers to computational methods for identifying compounds likely to bind to a target protein from large databases of chemical structures.  

"Conformational energy" refers to the potential energy associated with a particular three-dimensional arrangement of atoms in a molecule.  

## DETAILED DESCRIPTION  

The present invention provides a comprehensive approach for identifying novel small molecule modulators of CXCR4 through integrated computational and experimental methods. The detailed methodology is described below.  

### EXPERIMENTAL  

The experimental approach combines computational screening techniques with biological validation assays to identify and characterize novel CXCR4 modulators. The workflow includes:  

1) Preparation of an annotated compound database with calculated low-energy conformations;  
2) Development and validation of a pharmacophore model based on known CXCR4 antagonists;  
3) Parallel structure-based virtual screening using multiple docking algorithms;  
4) Intersection of hits from ligand-based and structure-based screening approaches;  
5) Biological characterization of selected compounds using calcium imaging, ERK activation, receptor internalization, and chemotaxis assays.  

### Annotated Database Creation  

The screening process began with creation of an annotated database using the ChemBridge GPCR-focused library containing approximately 13,000 compounds. Low-energy conformers were generated using the ConFirm/CatConf module from the Catalyst program implemented in Discovery Studio 3.1. Conformer generation employed a modified version of the CHARMm force field with a poling technique that biased sampling toward geometries distant from local minima but energetically similar. This method generated approximately 100 conformers per compound within an energy cutoff of 10 kcal/mol, providing a comprehensive representation of accessible conformational space for each molecule.  

### EXAMPLE 2  

### Common-Feature Pharmacophore Model Building And Database Screening  

A common-feature pharmacophore model was developed using 162 known CXCR4 antagonists from ChEMBL (version 13) with reported IC50 values ranging from 1 nM to 10 μM. Cluster analysis grouped these compounds into 5 distinct clusters based on structural similarity. From each cluster, 2 representative molecules were selected to form a training set for pharmacophore hypothesis generation.  

The common feature pharmacophore modeling tool in Discovery Studio 3.1 generated 10 pharmacophore hypotheses using default parameters. These hypotheses were validated using a test set of 10 additional compounds (2 from each cluster). One 5-point pharmacophore model demonstrated excellent fit to all test compounds, consisting of two hydrophobic (Hy) features, two hydrogen bond acceptor (HBA) features, and one positive ionizable (PI) feature. Each pharmacophoric feature was assigned equal weight, providing a maximum fit value of 100% for perfect alignment.  

Screening the annotated GPCR compound database with this pharmacophore model identified 26 structures with fit values >85% and conformational energies <5 kcal/mol. Six commercially available and synthetically tractable compounds from this set were selected for biological testing.  

### EXAMPLE 3  

### Structure-Based Virtual Screen  

Structure-based virtual screening employed two docking algorithms with orthogonal approaches: Surflex (fragment-based) and Glide (grid-based). The CXCR4 crystal structure (PDB code 3ODU) was prepared for docking by correcting side chains, adding missing atoms, optimizing residue orientations, assigning charges, and setting appropriate protonation states at physiological pH.  

For Glide docking, a 12 Å3 grid box was generated centered on the bound ligand (IT1t) from the crystal structure. Known CXCR4 antagonists were first docked to validate the approach, followed by docking of the GPCR-focused library. Compounds with Glide scores <-6.0 were considered hits.  

For Surflex docking, a ligand-based protomol was generated representing an ideal active-site template. After validating with known antagonists, the compound library was docked, with hits defined as having total scores >6.0 (where total score relates to -logKd).  

Intersection of hits from both docking approaches yielded 22 compounds with consistent binding poses and favorable interactions. After filtering for drug-like properties, synthetic feasibility, and commercial availability, 9 compounds were selected for biological testing.  

### EXAMPLE 4  

### Calcium Imaging Assay  

Initial biological validation employed a calcium imaging assay using C8161 human melanoma cells expressing high levels of CXCR4. Cells were stimulated with SDF-1 before and after compound addition to distinguish agonist and antagonist activity.  

Four compounds (NUCC-388, 392, 397, and 51420) demonstrated antagonist activity, inhibiting SDF-1-induced calcium mobilization with IC50 values of 0.3 μM, 1 μM, 3 μM, and 1 μM respectively. Five compounds (NUCC-390, 398, 54118, 54121, and 54127) showed agonist activity, inducing calcium mobilization similar to SDF-1. Agonist effects were specifically blocked by the CXCR4 antagonist AMD3100, confirming CXCR4 mediation.  

### EXAMPLE 5  

### Erk Activation By Agonist Nucc-390  

The agonist NUCC-390 was further characterized by examining ERK phosphorylation, a downstream signaling event of CXCR4 activation. Treatment of C8161 cells with NUCC-390 for 30 minutes increased phosphorylated ERK levels comparable to SDF-1 treatment, confirming agonist activity through an independent signaling pathway.  

### EXAMPLE 6  

### Nucc-390 Induces Internalization Of Cxcr4 Receptors  

CXCR4 receptor internalization was assessed using HEK293 cells expressing YFP-tagged CXCR4. Both SDF-1 and NUCC-390 treatment for 2 hours induced pronounced receptor internalization, with fluorescent signal shifting from the membrane to cytoplasmic aggregates. This effect was blocked by AMD3100, demonstrating specificity.  

### EXAMPLE 7  

### Sdf-1 And Nucc-390 Mediate Chemotaxis  

Chemotactic activity was evaluated using a Boyden chamber assay with C8161 cells. NUCC-390 induced robust cell migration comparable to SDF-1, confirming functional agonist activity in a physiologically relevant assay.  

The complete patent application provides detailed descriptions of the novel compounds, their identification methods, and biological characterization as CXCR4 modulators. This represents the first disclosure of small molecule CXCR4 agonists and an integrated computational/experimental approach for their discovery.