# DESCRIPTION

The present application claims priority to U.S. Provisional Patent Application No. 63/456,789, filed on October 12, 2022, the entire contents of which are hereby incorporated by reference in their entirety for all purposes.

## TECHNICAL FIELD

The present invention relates generally to the field of medicinal chemistry and molecular oncology. More specifically, the invention pertains to novel small-molecule inhibitors of Signal Transducer and Activator of Transcription 3 (Stat3), methods for their identification and synthesis, pharmaceutical compositions comprising such inhibitors, and their use in the treatment of cancers and other diseases characterized by aberrant Stat3 signaling. The invention further encompasses combination therapies involving these Stat3 inhibitors with radiotherapy, chemotherapy, immunotherapy, or other targeted agents, particularly in the context of head and neck squamous cell carcinoma (HNSCC) and other malignancies driven by constitutive or ligand-induced Stat3 activation.

## BACKGROUND OF THE INVENTION

Signal transducer and activator of transcription 3 (Stat3) is a critical transcription factor belonging to the STAT family of proteins, which mediate intracellular signaling in response to cytokines and growth factors. Upon ligand binding to cell surface receptors such as those for interleukin-6 (IL-6) or epidermal growth factor (EGF), receptor-associated kinases phosphorylate specific tyrosine residues on the receptor cytoplasmic tails. Stat3 is recruited to these phosphotyrosine motifs via its Src homology 2 (SH2) domain, where it becomes phosphorylated at tyrosine 705 (Y705). This phosphorylation triggers Stat3 homodimerization through reciprocal SH2–pY705 interactions, nuclear translocation, DNA binding, and transcriptional activation of target genes involved in cell proliferation, survival, angiogenesis, immune evasion, and stemness.

Stat3 plays a central role in oncogenesis and is constitutively activated in approximately fifty percent of all human cancers, including breast, prostate, lung, colon, and head and neck cancers. In head and neck squamous cell carcinoma (HNSCC), Stat3 was the first transcription factor demonstrated to be essential for tumor cell growth and survival. Persistent Stat3 signaling supports tumor maintenance, metastasis, resistance to therapy, and the self-renewal of cancer stem cells—subpopulations implicated in tumor initiation, relapse, and therapeutic resistance. Consequently, Stat3 has emerged as a high-value molecular target for anticancer drug development.

Despite its therapeutic promise, the clinical translation of Stat3 inhibitors has been hindered by significant challenges. Existing small-molecule inhibitors—such as Stattic, STA-21, S3I-201, and BP-1-102—suffer from limitations including low potency, poor selectivity, inadequate pharmacokinetic properties, off-target effects, or covalent mechanisms that compromise safety. Many of these compounds exhibit IC50 values in the micromolar range, lack oral bioavailability, or demonstrate toxicity at effective doses. Furthermore, while some inhibitors show activity in vitro, they often fail to suppress Stat3 phosphorylation or tumor growth in vivo due to insufficient target engagement or metabolic instability. A critical unmet need remains for potent, selective, non-covalent, orally bioavailable Stat3 inhibitors with favorable safety profiles suitable for clinical development, either as monotherapy or in rational combinations with standard-of-care modalities such as radiation or chemotherapy.

## SUMMARY OF THE INVENTION

The present invention provides novel methods and compositions for inhibiting Stat3 activity through a rational drug discovery approach centered on virtual ligand screening (VLS) targeting the phosphotyrosine (pY)-binding pocket of the Stat3 SH2 domain. Using computational docking of nearly one million small molecules into the Stat3 SH2 domain structure, three initial hit compounds—designated Cpd3, Cpd30, and Cpd188—were identified as reversible inhibitors of Stat3–pY-peptide interaction. Among these, Cpd188 demonstrated promising activity in biochemical and cellular assays.

Building upon Cpd188 as a chemical scaffold, a hit-to-lead optimization program was executed involving two-dimensional (2D) fingerprint similarity screening of over 490,000 compounds followed by three-dimensional (3D) pharmacophore modeling and quantitative structure–activity relationship (QSAR) analysis. This integrated approach yielded second-generation derivatives with enhanced potency. Notably, compound Cpd188-9 emerged as a lead candidate exhibiting superior inhibitory activity against Stat3 across multiple assay platforms. Cpd188-9 binds to the Stat3 SH2 domain with high affinity (KD = 4.7 nM), effectively blocks IL-6- or G-CSF–mediated Stat3 phosphorylation (IC50 = 3.7 μM), and potently inhibits anchorage-dependent and -independent growth of Stat3-dependent cancer cell lines, including HNSCC models.

Surprisingly, detailed mechanistic studies revealed that Cpd188-9 also inhibits Stat1 activation with comparable potency, making it a dual Stat3/Stat1 inhibitor. This dual activity is therapeutically advantageous in cancers like HNSCC, where both Stat3 and Stat1 contribute to radioresistance and chemoresistance through overlapping and distinct transcriptional programs, including the interferon-related DNA damage resistance signature (IRDS). RNA sequencing of tumor xenografts treated with Cpd188-9 confirmed downregulation of numerous oncogenic, pro-survival, and therapy-resistance genes regulated by both Stat3 and Stat1.

The invention further discloses specific chemical structures of Stat3 inhibitors falling within defined general formulas. These compounds are characterized by a core N-naphthyl benzenesulfamide scaffold, with substituents at key positions—particularly the 3-position of the naphthyl ring—that confer high binding affinity and selectivity. Although initial design focused on Stat3 selectivity over Stat1, the unexpected yet beneficial dual inhibition profile of Cpd188-9 expands its therapeutic utility. The compounds inhibit nuclear translocation of phosphorylated Stat3, disrupt Stat3–DNA complex formation, and induce apoptosis in cancer cells without significant toxicity to normal hematopoietic progenitors at therapeutic doses.

Pharmaceutical compositions comprising these inhibitors, alone or in combination with radiation, chemotherapy (e.g., cisplatin, taxol), targeted agents (e.g., cetuximab, herceptin), or immunotherapies, are provided for the treatment of Stat3/Stat1-driven malignancies. The invention is particularly suited for treating HNSCC, breast cancer, and other solid tumors exhibiting constitutive or ligand-induced Stat3/Stat1 activation, especially those resistant to conventional therapies. Additionally, the compounds may be used in regenerative medicine contexts where modulation of Stat signaling is beneficial.

## DETAILED DESCRIPTION OF THE INVENTION

The present application incorporates by reference U.S. Patent Application Nos. 17/123,456 and 16/987,654, which describe earlier generations of Stat3 inhibitors and screening methodologies. The invention provides novel objects, features, and advantages over prior art, including the identification of high-affinity, non-covalent, orally bioavailable small molecules that directly target the Stat3 SH2 domain, thereby disrupting dimerization and transcriptional activity. Unlike peptidomimetics or oligonucleotide-based decoys, the disclosed compounds are drug-like, synthetically tractable, and amenable to medicinal chemistry optimization.

The Stat3 inhibitors of the invention were developed through structure-based virtual screening targeting the pY-binding pocket of the Stat3 SH2 domain (residues 595–688 of human Stat3, GeneID: 6774). This pocket is essential for phosphopeptide recognition and subsequent dimerization. Human Stat1 (GeneID: 6772) shares significant sequence and structural homology in its SH2 domain, presenting a challenge for achieving selectivity. However, subtle differences in key residues lining the binding cleft—such as Lys591 in Stat3 versus Glu612 in Stat1—were exploited during compound design and analysis.

In cellular assays, the lead compound Cpd188-9 demonstrated potent inhibition of Stat3 phosphorylation and function in multiple breast cancer and HNSCC cell lines, including UM-SCC-17B, SCC-35, and SCC-61. It induced apoptosis at low micromolar concentrations and suppressed tumor xenograft growth in vivo when administered orally or intraperitoneally. Importantly, Cpd188-9 exhibited a high maximum tolerated dose (>100 mg/kg/day in mice) and favorable pharmacokinetics, including tumor accumulation at levels nearly three-fold higher than plasma.

### I. DEFINITIONS

As used herein, the articles “a” and “an” are intended to mean one or more unless the context clearly indicates otherwise. The term “another” refers to at least a second or additional instance. The terms “having,” “including,” “containing,” and “comprising” are used interchangeably and are open-ended, allowing for the presence of additional elements. An “inhibitor” denotes any molecule that interferes with Stat3 activity, including but not limited to blocking SH2 domain–phosphopeptide interaction, dimerization, nuclear translocation, or DNA binding. A “therapeutically effective amount” is that quantity of a compound sufficient to produce a desired therapeutic effect in a subject without undue toxicity. “Pharmaceutically acceptable” refers to carriers, diluents, or excipients compatible with human or veterinary use. A subject “at risk for having cancer” includes individuals with genetic predispositions, premalignant lesions, or elevated biomarkers. “Binding affinity” quantifies the strength of molecular interaction, typically reported as dissociation constant (KD); association constants (KA) may be measured via surface plasmon resonance (SPR) or microscale thermophoresis (MST). “Chemotherapy-resistant cancer” denotes malignancies unresponsive to standard cytotoxic agents. A “domain” is a structurally and/or functionally distinct polypeptide segment; the “SH2 domain” mediates protein–protein interactions via phosphotyrosine recognition. The term “mammal” includes humans and other animals suitable for therapeutic intervention.

### II. DERIVATIVES

A “derivative” is a compound chemically derived from a parent structure, and a “functionally active derivative” retains the biological activity of the precursor. Derivatives of the disclosed Stat3 inhibitors include substitutions on aromatic rings, modifications of sulfonamide groups, or introduction of heterocycles. The invention includes methods of inhibiting Stat3 in a cell by administering such derivatives. Exemplary general formulas include Formula I: N-(naphthalen-1-yl)benzenesulfonamide derivatives wherein R1 and R2 independently represent hydrogen, halogen, alkyl, alkoxy, or aryl groups. Formula II encompasses compounds with R1, R2, R3, and R4 substituents on the benzenesulfonyl and naphthyl moieties, including triazole, binaphthyl, or mercapto groups at the 3-position of naphthalene. Formula III defines trisubstituted variants with R1–R3 selected from cyclic alkanes (e.g., cyclopropyl), ketones (e.g., acetophenone derivatives), monocyclic or polycyclic arenes (e.g., phenyl, naphthyl), or heteroarenes (e.g., pyridyl, thienyl). Synthetic routes involve sulfonation, nucleophilic aromatic substitution, or palladium-catalyzed cross-coupling reactions.

### III. EMBODIMENTS FOR TARGETING STAT3

The STAT family comprises seven members (Stat1–4, Stat5a, Stat5b, Stat6) with non-redundant roles. Stat1 mediates interferon responses and tumor suppression, whereas Stat3 drives oncogenesis. Both exist as isoforms: Stat3α (full-length, transforming) and Stat3β (C-terminal truncated, antagonistic to transformation). The invention specifically targets Stat3α while sparing Stat1 where possible, though dual inhibition proved beneficial in HNSCC. Virtual ligand screening identified first-generation inhibitors (Cpd3, Cpd30, Cpd188), followed by second-generation (e.g., Cpd188-9) and third-generation probes optimized via SAR and QSAR.

### IV. TARGETING CANCER STEM CELLS

Cancer stem cells (CSCs) drive tumor initiation, metastasis, and relapse. Stat3 is indispensable for CSC self-renewal in breast and head and neck cancers. The disclosed inhibitors, particularly Cpd188-9, eliminate CSCs in vitro and in vivo, as demonstrated by mammosphere formation and xenograft limiting dilution assays. Virtual screening strategies prioritized compounds capable of penetrating CSC niches and disrupting Stat3-dependent stemness pathways.

### V. COMBINATION THERAPY

The Stat3 inhibitors are administered with chemotherapy (cisplatin, carboplatin, taxol), targeted therapy (herceptin, tykerb, tamoxifen), radiotherapy, immunotherapy, or surgery. Protocols include concurrent or sequential dosing. For example, Cpd188-9 sensitizes radioresistant HNSCC to ionizing radiation by suppressing IRDS genes. Combinations with EGFR inhibitors (e.g., erlotinib) or anti-HER2 agents are particularly effective in tumors with RTK–Stat3 crosstalk.

### VI. PHARMACEUTICAL COMPOSITIONS

Pharmaceutical compositions comprise a Stat3 inhibitor and a pharmaceutically acceptable carrier (e.g., saline, polysorbate, cyclodextrin). Dosage ranges from 1–200 mg/kg/day, administered orally, intravenously, or intraperitoneally. Formulations include sterile injectables, tablets, or capsules, stabilized with antioxidants (e.g., ascorbate) and preservatives. Compounds may be used as free bases or salts (e.g., hydrochloride).

### VII. KITS OF THE INVENTION

Kits contain a Stat3 inhibitor in a container, optionally with instructions for use in cancer therapy. Syringeable formulations enable immediate clinical administration.

## EXAMPLES

### Exemplary Materials and Methods

The Stat3 SH2 domain structure (PDB: 1BG1) was converted to ICM format. Stat1 SH2 coordinates (PDB: 1YVL) were retrieved for comparative docking. Commercial databases (Life Chemicals, ChemBridge) were screened using ICM-Pro. The pY-binding pocket was defined around Arg609, Ser611, and Lys591. Flexible docking yielded 920,000 poses; top 100 compounds were purchased. Binding was assessed via SPR using biotinylated EGFR pY1068 peptide. Immunoblots detected pStat3 (Tyr705). Similarity screens used Unity/Sybyl. EMSA measured Stat3–DNA binding. Confocal microscopy tracked nuclear translocation. Apoptosis was quantified by Annexin V staining in MDA-MB-468 cells.

### Identification by VLS of Compounds that Blocked Stat3 Binding...

VLS identified Cpd3, Cpd30, and Cpd188 as blockers of Stat3–pY interaction. All inhibited IL-6–induced Stat3 phosphorylation in HepG2 cells.

### Compound-Mediated Inhibition... is Specific for Stat3 Vs. Stat1

Initially, compounds showed >10-fold selectivity for Stat3 over Stat1 in SPR and phosphoflow assays. However, Cpd188-9 later demonstrated equipotent inhibition of both.

### Sequence Analysis and Molecular Modeling...

Stat3 and Stat1 SH2 domains share 47% sequence identity. Molecular modeling revealed that Cpd188-9 forms van der Waals contacts with Leu706 and Lys591 in Stat3, and analogous residues in Stat1, explaining dual activity. Energy calculations confirmed favorable binding to both.

### Example 5

Cpd3, Cpd30, Cpd188, Cpd3-2, and Cpd3-7 inhibited Stat3 phosphorylation and nuclear accumulation in UM-SCC-17B cells. EMSA showed disrupted Stat3–hSIE binding. Apoptosis increased 3–5 fold in breast cancer lines. RNA-seq of xenografts revealed broad modulation of Stat3/Stat1 targets. Prior inhibitors (Stattic, S3I-201) were less potent. Stat3’s role in embryonic and leukemic stem cells underscores the need for selective probes. In AML with t(8;21), Stat3 supports leukemic stem cell survival. Cpd188-9 eliminated leukemic CFU-GM colonies without harming normal progenitors. In breast cancer, Stat3 maintains CSCs; Cpd188-9 reduced ALDH+ populations and suppressed xenograft initiation. Third-generation probes feature larger sulfonyl groups and modified naphthyl substituents, improving affinity 10-fold over Cpd188. SPR, HTFM, and MSFE assays confirmed efficacy. In vivo, Cpd188-9 eradicated residual CSCs post-chemotherapy, preventing relapse.

## PUBLICATIONS

The following publications are incorporated by reference: Akira et al., 1997; Al-Hajj et al., 2003; Becker et al., 1998; Bhasin et al., 2007; Brinkley, 1980; Bromberg et al., 1998, 1999, 2000, 2002; Cailleau et al., 1978; Caldenhoven et al., 1996; Catlett-Falcone et al., 1999; Chakraborty et al., 1996, 1999; Chapman et al., 1999; Chen et al., 1998; Cohen et al., 2005; Coleman et al., 2005; Costa-Pereira et al., 2002. Additional references describe constitutive Stat3 activation in breast carcinoma, intrinsic resistance of tumorigenic breast cancer cells, JAK3/STAT3 in colon cancer, Stat3 isoforms, small-molecule inhibitors, Stat1 knockout phenotypes, cancer statistics, Stat3 in prostate cancer, STAT5 structure, EGF-induced Stat3 activation, dimer stability, alternative Stat1/Stat3 activation, Stat1 gene regulation, Stat3-mediated chemoresistance, transcription factor targeting, phosphopeptide inhibitors, and Stat6 motifs.