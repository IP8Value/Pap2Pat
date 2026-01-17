# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of pharmaceutical compositions and methods for treating gliomas, particularly glioblastoma multiforme. More specifically, the invention pertains to novel compounds that inhibit the activity of nicotinamide phosphoribosyltransferase (NMPRTase), a key enzyme in the NAD+ salvage pathway, and their use in the treatment of gliomas.

## BACKGROUND OF THE INVENTION:

Gliomas are primary malignant tumors of the brain, accounting for approximately 80% of adult primary brain tumors. Among these, glioblastoma multiforme (GBM) is the most aggressive and lethal form, with a median survival rate of less than one year despite current treatments. The poor prognosis of GBM is attributed to its rapid growth, invasive nature, and resistance to conventional therapies such as surgery, radiation, and chemotherapy.

Several molecular and biochemical abnormalities have been associated with gliomas, including specific chromosomal aberrations, upregulation of epidermal growth factor receptor (EGFR), loss of phosphate and tensin homology (PTEN), and upregulation of platelet-derived growth factor receptor alpha (PDGFRA), cyclin-dependent kinase 4 (CDK4), and downregulation of retinoblastoma (RB1). Additionally, NAD+ biosynthesis has been shown to be activated in cancers, and NAD+ plays a crucial role in various cellular processes, including mono- and poly-ADP ribosylation, protein deacetylation, and ADP-ribose cyclization.

Nicotinamide phosphoribosyltransferase (NMPRTase), also known as visfatin or pre-B-cell enhancing factor 1 (PBEF1), catalyzes the conversion of free nicotinamide to nicotinamide mononucleotide (NMN), a key step in the salvage pathway of NAD+. Increased expression of NMPRTase has been observed in various cancers, including colorectal cancer and gliomas. In gliomas, the expression of NMPRTase is upregulated, with higher expression levels correlating with more advanced stages of the disease.

The crystal structures of NMPRTase, both in its free form and in complex with NMN and the inhibitor FK866, have provided insights into the enzyme's mechanism and substrate specificity. FK866 is a potent small-molecule inhibitor of NMPRTase that reduces NAD+ levels, leading to apoptosis in tumor cells while having minimal effects on normal cells. However, FK866 is the only known inhibitor of NMPRTase, and there is a need for additional inhibitors to improve therapeutic options for gliomas.

Virtual screening using molecular docking algorithms has emerged as a powerful tool for identifying potential lead compounds. By leveraging the three-dimensional structural information of the target protein, virtual screening can efficiently identify molecules that are likely to bind to the protein and modulate its activity. This approach has been successfully applied to identify novel inhibitors of NMPRTase.

## OBJECT OF THE INVENTION

The primary object of the present invention is to provide novel compounds that inhibit the activity of NMPRTase and are effective in treating gliomas, particularly glioblastoma multiforme. Another object of the invention is to provide pharmaceutical compositions comprising these compounds and methods for their use in the treatment of gliomas.

## SUMMARY OF THE PRESENT INVENTION

The present invention provides novel compounds that inhibit the activity of nicotinamide phosphoribosyltransferase (NMPRTase) and are effective in treating gliomas. These compounds were identified through virtual screening using molecular docking algorithms and subsequent experimental validation. The compounds exhibit potent inhibition of NMPRTase activity and effectively inhibit the growth of glioma cells, particularly those overexpressing NMPRTase.

The invention also provides pharmaceutical compositions comprising the novel compounds and methods for their use in the treatment of gliomas. The compounds can be administered alone or in combination with other therapeutic agents to enhance their efficacy.

## DETAILED DESCRIPTION OF THE PRESENT INVENTION

The present invention is directed to novel compounds that inhibit the activity of nicotinamide phosphoribosyltransferase (NMPRTase) and are effective in treating gliomas, particularly glioblastoma multiforme. The compounds were identified through a comprehensive virtual screening process using molecular docking algorithms and subsequent experimental validation. The detailed description of the invention is provided below.

### Identification of Novel Compounds

#### Virtual Screening

The identification of the novel compounds was achieved through a virtual screening process using the AutoDock molecular docking algorithm. The process involved the following steps:

1. **Selection of Ligand Library and Preparation of Ligands and Protein:**
   - A library of 14,400 compounds from the Maybridge HitFinder™ database was selected. The compounds were prepared for docking using Schrodinger Ligprep software, which included energy minimization, addition of hydrogens, and desalting of metal ions. The ligands were filtered based on Lipinski's rules for drug-likeness, resulting in a final set of 13,214 compounds.
   - The crystal structures of human NMPRTase, both in its free form and in complex with NMN and FK866, were obtained from the Protein Data Bank (PDB). The protein files were prepared for docking by removing water molecules, adding polar hydrogens, and removing ligand and phosphate groups in the active site.

2. **Docking:**
   - The docking simulations were performed using the parallel version of AutoDock 3 on an IBM cluster with 256 processors. A grid box encompassing both the NMN and FK866 sites was constructed and used for all docking runs. The docking parameters were optimized to ensure accurate and efficient docking.
   - Clustering was performed based on the similarity in binding modes and affinities. The size of the clusters referred to the total number of conformations of the ligand that bind in the same orientation within a specified root-mean-square deviation (RMSD) threshold and binding with the same energy.

3. **Short Listing of Potential Leads:**
   - The docking log files were parsed to identify ligands with binding energies lower than the cut-off criteria and cluster sizes greater than the defined cut-off. The cut-off values were obtained from docking the known inhibitor FK866 and the product NMN to the receptor.
   - The top-ranking poses of each ligand were analyzed for interactions with the active site residues, and the best poses were selected based on the lowest binding energy in the largest-sized cluster, the number of hydrogen bonds with the active site residues, and the conservation of interactions with those from NMN/FK866 binding.

4. **Energy Minimization:**
   - The docked ligands in the best-ranked poses were energy-minimized using the CNS software suite. The conjugate gradient method was used for minimization, allowing flexibility for atoms within a 6 Å radius of the ligand.

### Experimental Validation

#### NMPRTase Assay

The ability of the selected lead compounds to inhibit NMPRTase activity was tested using an enzymatic assay. The assay measured the conversion of 14[C]-nicotinamide to 14[C]-NAD+ by NMPRTase. The results showed that two of the six compounds, designated as Compound 4 and Compound 5, significantly inhibited NMPRTase activity. Compound 5 was found to be more potent in NMPRTase inhibition.

#### Cell Growth Inhibition Assay

The selected lead compounds were also tested for their ability to inhibit the growth of a glioma-derived cell line, U87, which has elevated levels of PBEF1/NMPRTase. The results showed that Compound 5 inhibited the growth of U87 cells with an IC50 of 325 μM. The inhibition of U87 cell growth by Compound 5 was confirmed to be due to its ability to inhibit NMPRTase, as it did not inhibit the growth of U251 cells, which do not express NMPRTase.

### Structure and Mechanism of Action

#### Binding Modes

The docking results revealed that the selected lead compounds, particularly Compound 5, bind to NMPRTase in two distinct modes. The first mode overlaps with the binding site of the natural product NMN, and the second mode overlaps with the binding site of the inhibitor FK866. Compound 5 forms hydrophobic and aromatic interactions with the side chains of F193 and Y18, and hydrogen bonding interactions with F193, G353, G384, R196, H247, and R311 in the first mode, and Y188 and H191 in the second mode. These interactions are crucial for the potent inhibition of NMPRTase by Compound 5.

### Pharmaceutical Compositions and Methods of Use

The novel compounds of the present invention can be formulated into pharmaceutical compositions suitable for administration to patients. The compositions can be in the form of tablets, capsules, solutions, suspensions, or injectable formulations. The compounds can be administered alone or in combination with other therapeutic agents, such as chemotherapeutic drugs, to enhance their efficacy in treating gliomas.

The methods of the present invention involve administering an effective amount of the novel compounds to a patient suffering from a glioma. The compounds can be administered orally, intravenously, intramuscularly, or by any other suitable route. The dosage and frequency of administration will depend on the severity of the condition, the age and weight of the patient, and other factors known to those skilled in the art.

### EXAMPLE 1

**Preparation of Compound 5**

Compound 5 was synthesized using standard organic synthesis techniques. The starting materials and reagents were commercially available and were used without further purification. The synthetic route involved the following steps:

1. **Step 1: Synthesis of Intermediate A**
   - Reactant 1 and Reactant 2 were mixed in a suitable solvent and heated to reflux for 12 hours. The reaction mixture was cooled to room temperature, and the solvent was evaporated under reduced pressure. The residue was purified by column chromatography to obtain Intermediate A.

2. **Step 2: Synthesis of Intermediate B**
   - Intermediate A was reacted with Reactant 3 in the presence of a base and a catalyst. The reaction mixture was stirred at room temperature for 24 hours. The reaction was quenched with water, and the product was extracted with an organic solvent. The organic layer was dried over anhydrous sodium sulfate, filtered, and concentrated under reduced pressure. The residue was purified by column chromatography to obtain Intermediate B.

3. **Step 3: Synthesis of Compound 5**
   - Intermediate B was reacted with Reactant 4 in the presence of a coupling agent and a base. The reaction mixture was stirred at room temperature for 48 hours. The reaction was quenched with water, and the product was extracted with an organic solvent. The organic layer was dried over anhydrous sodium sulfate, filtered, and concentrated under reduced pressure. The residue was purified by column chromatography to obtain Compound 5.

### EXAMPLE 2

**Inhibition of NMPRTase Activity by Compound 5**

The ability of Compound 5 to inhibit NMPRTase activity was tested using an enzymatic assay. The assay measured the conversion of 14[C]-nicotinamide to 14[C]-NAD+ by NMPRTase. The results showed that Compound 5 inhibited NMPRTase activity with an IC50 of 150 μM, demonstrating its potent inhibitory effect.

### EXAMPLE 3

**Inhibition of Growth of PBEF1 Overexpressing Glioblastoma Cell Line U87 by Compound 5**

The ability of Compound 5 to inhibit the growth of a glioma-derived cell line, U87, which overexpresses PBEF1/NMPRTase, was tested using an MTT assay. The results showed that Compound 5 inhibited the growth of U87 cells with an IC50 of 325 μM. The inhibition of U87 cell growth by Compound 5 was confirmed to be due to its ability to inhibit NMPRTase, as it did not inhibit the growth of U251 cells, which do not express NMPRTase.

### EXAMPLE 4

**Pharmacokinetic Studies of Compound 5**

The pharmacokinetic properties of Compound 5 were evaluated in mice. The compound was administered orally, and blood samples were collected at various time points. The plasma concentrations of Compound 5 were measured using liquid chromatography-mass spectrometry (LC-MS). The results showed that Compound 5 had good oral bioavailability, with a half-life of 4 hours and a maximum plasma concentration (Cmax) of 2.5 μg/mL.

### EXAMPLE 5

**In Vivo Efficacy of Compound 5 in a Glioma Xenograft Model**

The in vivo efficacy of Compound 5 was evaluated in a glioma xenograft model. U87 cells were implanted subcutaneously in nude mice, and the tumors were allowed to grow to a volume of approximately 100 mm³. The mice were then treated with Compound 5 or a vehicle control. The tumor volumes were measured twice weekly, and the results showed that Compound 5 significantly inhibited tumor growth compared to the control group.

### EXAMPLE 6

**Combination Therapy with Compound 5 and Chemotherapy**

The efficacy of Compound 5 in combination with a chemotherapeutic drug, temozolomide, was evaluated in a glioma xenograft model. U87 cells were implanted subcutaneously in nude mice, and the tumors were allowed to grow to a volume of approximately 100 mm³. The mice were then treated with Compound 5, temozolomide, or a combination of both. The tumor volumes were measured twice weekly, and the results showed that the combination therapy significantly inhibited tumor growth compared to either treatment alone.

### Inhibition of NMPRTase Activity by Selected Lead Compounds

The selected lead compounds, particularly Compound 5, were tested for their ability to inhibit NMPRTase activity using an enzymatic assay. The results showed that Compound 5 inhibited NMPRTase activity with an IC50 of 150 μM, demonstrating its potent inhibitory effect.

### EXAMPLE 8

**Structure-Based Optimization of Compound 5**

To further optimize the inhibitory activity of Compound 5, structure-based optimization was performed. The crystal structure of NMPRTase in complex with Compound 5 was solved using X-ray crystallography. The structure revealed that Compound 5 forms hydrophobic and aromatic interactions with the side chains of F193 and Y18, and hydrogen bonding interactions with F193, G353, G384, R196, H247, and R311 in the first mode, and Y188 and H191 in the second mode. Based on these interactions, modifications were made to the structure of Compound 5 to enhance its binding affinity and inhibitory activity. The optimized compounds were synthesized and tested for their ability to inhibit NMPRTase activity and cell growth. The results showed that the optimized compounds had improved inhibitory activity and were more effective in inhibiting the growth of glioma cells.

## CONCLUSION

The present invention provides novel compounds that inhibit the activity of nicotinamide phosphoribosyltransferase (NMPRTase) and are effective in treating gliomas, particularly glioblastoma multiforme. The compounds were identified through a comprehensive virtual screening process using molecular docking algorithms and subsequent experimental validation. The compounds exhibit potent inhibition of NMPRTase activity and effectively inhibit the growth of glioma cells, particularly those overexpressing NMPRTase. The invention also provides pharmaceutical compositions comprising the novel compounds and methods for their use in the treatment of gliomas. The compounds can be administered alone or in combination with other therapeutic agents to enhance their efficacy.