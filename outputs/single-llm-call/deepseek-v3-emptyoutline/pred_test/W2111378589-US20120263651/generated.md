Here is the complete patent application following your outline and based on the provided research paper:

# DESCRIPTION

## STATEMENT OF FEDERALLY FUNDED RESEARCH

This invention was made with government support under Grant No. [INSERT GRANT NUMBER] awarded by [INSERT FUNDING AGENCY]. The government has certain rights in the invention.

## TECHNICAL FIELD OF THE INVENTION

The present invention relates generally to the field of molecular biology and immunology. More specifically, the invention pertains to novel single-stranded DNA (ssDNA) aptamers that specifically bind to the activated form of transcription factor RelA (p65 subunit of NF-κB), methods for their selection and characterization, and their use in detecting, isolating, and quantifying activated RelA in biological samples. The invention further relates to compositions and methods for modulating NF-κB mediated gene expression using these aptamers.

## BACKGROUND OF THE INVENTION

The nuclear factor kappa-light-chain-enhancer of activated B cells (NF-κB) family of transcription factors plays a central role in regulating immune responses, inflammation, cell proliferation, and survival. NF-κB exists as homo- or heterodimers of five related proteins: RelA (p65), RelB, c-Rel, p50, and p52. In resting cells, NF-κB is sequestered in the cytoplasm through interaction with inhibitory proteins called IκBs, particularly IκBα. Upon cellular stimulation by various inflammatory signals such as tumor necrosis factor alpha (TNFα), IκB kinase (IKK) phosphorylates IκBα, leading to its ubiquitination and proteasomal degradation. This releases NF-κB, allowing its translocation to the nucleus where it binds specific DNA sequences and regulates target gene expression.

RelA is the most abundant and transcriptionally active subunit of NF-κB. The activation status of RelA is critical for proper immune function and is dysregulated in many diseases including chronic inflammatory conditions, autoimmune disorders, and cancers. Current methods for detecting activated RelA rely primarily on immunoblotting or electrophoretic mobility shift assays, which are labor-intensive, semi-quantitative, and cannot distinguish between free (activated) and IκBα-bound (inactive) RelA. There remains an unmet need for specific, sensitive, and quantitative tools to detect and measure activated RelA in biological samples.

Aptamers are single-stranded oligonucleotides that bind specific molecular targets with high affinity and specificity. While RNA aptamers against NF-κB proteins have been reported, their utility is limited by RNA instability. DNA aptamers offer advantages including greater stability, easier synthesis, and lower cost of production. However, prior to this invention, no ssDNA aptamers specific for RelA had been developed, particularly ones capable of distinguishing between activated and inactive forms of RelA.

## SUMMARY OF THE INVENTION

The present invention provides isolated ssDNA aptamers that specifically bind to the activated form of RelA. In particular embodiments, the invention provides aptamer P028F4 having the sequence GGTAACCTTGAGTCACGAATTCAA-[30 base variable region]-CAGAAGCTGTAAGTTGGGTACCTT (SEQ ID NO: 1), wherein the 30 base variable region contains the sequence GGGAC (SEQ ID NO: 2). This aptamer binds RelA with high affinity (KD = 6.4 × 10^-10 M) and specifically recognizes the activated, IκBα-free form of RelA while not binding to IκBα-complexed RelA.

The invention further provides methods for selecting RelA-specific ssDNA aptamers through iterative rounds of binding, enrichment, and amplification. The selected aptamers are characterized using techniques including nitrocellulose filter binding assays, surface plasmon resonance, electrophoretic mobility shift assays, and confocal microscopy.

The invention also provides methods for using the RelA-specific aptamers to detect, isolate, and quantify activated RelA in biological samples. In particular embodiments, the aptamers are used in tandem affinity purification (ATAP) to enrich for activated RelA complexes followed by mass spectrometric quantification using selected reaction monitoring (SRM). This approach allows precise measurement of the fraction of activated RelA in cells under various conditions.

Additionally, the invention provides methods for modulating NF-κB mediated gene expression by administering the RelA-specific aptamers to cells. The aptamers compete with endogenous NF-κB binding sites, thereby inhibiting RelA's transcriptional activity. Pharmaceutical compositions containing the aptamers and methods for their therapeutic use in treating NF-κB-related disorders are also provided.

## DETAILED DESCRIPTION OF THE INVENTION

The present invention is based on the discovery and characterization of novel ssDNA aptamers that specifically recognize the activated form of transcription factor RelA. The detailed description that follows provides specific embodiments of the invention but is not intended to limit its scope, which is defined by the appended claims.

### Aptamer Selection and Characterization

The invention provides methods for selecting RelA-specific ssDNA aptamers through systematic evolution of ligands by exponential enrichment (SELEX). In a preferred embodiment, a DNA library consisting of a 30-base random sequence flanked by 25-base common primer binding sites is used for selection. The library is incubated with recombinant GST-RelA(1-313) protein in binding buffer (20 mM Tris-HCl pH 7.4, 150 mM NaCl, 1 mM MgCl2) for 30-60 minutes at room temperature. Bound DNA is separated from free DNA by filtration through nitrocellulose membranes, eluted, and amplified by PCR using a biotinylated reverse primer and unmodified forward primer. The amplified material is bound to streptavidin-paramagnetic beads, and the non-biotinylated single strand is eluted for subsequent selection rounds.

After multiple rounds of selection (typically 12 rounds), the enriched pool is cloned and individual clones are characterized for RelA binding. Positive clones are sequenced and analyzed. One particularly effective aptamer, designated P028F4, contains the sequence GGGAC in its variable region, matching half of the double-strand NF-κB natural binding site GGGACTTTCC. The 30-base variable region alone is sufficient for RelA binding.

Binding affinity is determined by surface plasmon resonance, with P028F4 showing a KD of 6.4 × 10^-10 M (ka = 1.8 × 10^6 1/Ms, kd = 0.0018 1/s). Nitrocellulose filter binding assays confirm these measurements. Competition experiments demonstrate that P028F4 effectively competes with natural NF-κB binding sites for RelA binding.

### Specificity for Activated RelA

A key feature of the invention is the aptamers' specificity for the activated form of RelA. In vitro binding studies show that P028F4 binds free RelA but not RelA complexed with IκBα. This specificity is maintained in cellular contexts, as demonstrated by:

1. Confocal microscopy showing P028F4 colocalization with nuclear EGFP-RelA after TNFα stimulation but not in unstimulated cells;
2. Western blot analysis of aptamer-captured material showing enrichment of Ser536-phosphorylated RelA and p300 complexes but absence of IκBα;
3. Quantitative mass spectrometry measurements demonstrating aptamer capture of activated but not IκBα-bound RelA from cellular extracts.

This specificity enables unique applications for detecting and quantifying the activated fraction of RelA in biological samples.

### Aptamer Tandem Affinity Purification (ATAP)

The invention provides methods for isolating activated RelA complexes using aptamer-based tandem affinity purification. In a preferred embodiment:

1. FLAG-tagged RelA is first purified from cellular extracts using anti-FLAG affinity gel;
2. The eluted material is incubated with biotinylated P028F4 aptamer;
3. Aptamer-bound complexes are captured on streptavidin magnetic beads;
4. After washing, the complexes are eluted for analysis.

This ATAP method enriches for activated RelA complexes containing post-translational modifications (e.g., Ser536 phosphorylation) and associated proteins (e.g., p300) while excluding IκBα-bound RelA. The method achieves 14-30 fold enrichment of activated RelA compared to control aptamers.

### Quantitative Measurement of Activated RelA

The invention provides sensitive and specific methods for quantifying activated RelA using selected reaction monitoring mass spectrometry (SRM-MS) combined with aptamer enrichment. Key aspects include:

1. Selection of high responding signature peptides that stoichiometrically represent RelA, based on uniqueness, length (8-25 residues), and absence of problematic amino acids;
2. Synthesis of stable isotope-labeled peptide standards (SIS) for precise quantification;
3. Optimization of SRM transitions and collision energies for maximum sensitivity;
4. On-bead tryptic digestion of aptamer-captured RelA followed by LC-SRM-MS analysis.

This approach provides a linear response over >1000-fold concentration range with a lower limit of quantification of 200 amol. Aptamer enrichment increases the signal-to-noise ratio for RelA detection by 36-fold compared to analysis of crude extracts.

Using this method, the invention enables determination that:
- Unstimulated cells contain ~5% activated RelA in the cytoplasm;
- TNFα stimulation increases cytoplasmic activated RelA ~6-fold;
- Nuclear activated Rela equals total nuclear RelA after stimulation;
- A typical cell contains ~200,000 total RelA molecules with ~50,000 activated molecules in the nucleus after stimulation.

### Modulation of NF-κB Activity

The invention provides methods for modulating NF-κB mediated gene expression using the RelA-specific aptamers. In cellular studies:

1. Transfection of P028F4 reduces TNFα-induced expression of NF-κB target genes (Groβ and TNFAIP3/A20) to less than 2-fold compared to 5-7 fold induction in controls;
2. Chromatin immunoprecipitation shows P028F4 reduces RelA binding to endogenous promoters;
3. The effects are specific to P028F4 compared to control aptamers.

These findings demonstrate the aptamers' utility as research tools and potential therapeutic agents for modulating NF-κB activity.

### Pharmaceutical Applications

The invention provides pharmaceutical compositions containing the RelA-specific aptamers for treating NF-κB-related disorders. The compositions may include:

1. Therapeutically effective amounts of one or more RelA-specific aptamers;
2. Pharmaceutically acceptable carriers;
3. Optional additional active agents;
4. Formulations for various routes of administration (e.g., intravenous, local).

Methods of treatment comprise administering these compositions to subjects with conditions involving aberrant NF-κB activation, including but not limited to:

1. Inflammatory diseases (e.g., rheumatoid arthritis, inflammatory bowel disease);
2. Autoimmune disorders;
3. Certain cancers;
4. Infectious diseases with inflammatory components.

The aptamers may be modified to enhance stability, delivery, or pharmacokinetics, including:
1. Chemical modifications (e.g., 2'-O-methyl, phosphorothioate);
2. Conjugation to carrier molecules;
3. Incorporation into delivery vehicles (e.g., nanoparticles, liposomes).

### EXAMPLES

The following examples illustrate specific embodiments of the invention but are not intended to limit its scope.

Example 1: Aptamer Selection
A DNA library with 30-base random regions was subjected to 12 rounds of selection against GST-RelA(1-313). The enriched pool was cloned and sequenced, identifying aptamer P028F4 with the variable region containing GGGAC. Binding studies showed KD = 6.4 × 10^-10 M.

Example 2: Specificity Testing
Recombinant RelA was pre-incubated with increasing amounts of IκBα. P028F4 bound free RelA but not IκBα-complexed RelA in EMSA and surface plasmon resonance assays.

Example 3: Cellular Localization
Confocal microscopy of TNFα-stimulated cells showed P028F4 colocalization with nuclear EGFP-RelA but not in unstimulated cells or with control aptamer P028A1.

Example 4: Gene Expression Modulation
A549 cells transfected with P028F4 showed reduced TNFα-induced Groβ and TNFAIP3/A20 expression compared to controls (2-fold vs 5-7 fold induction).

Example 5: Quantitative SRM-MS
ATAP combined with SID-SRM-MS quantified ~50,000 activated RelA molecules/nucleus after TNFα stimulation, representing ~25% of total cellular RelA.

### EQUIVALENTS

While the invention has been described with reference to specific embodiments, modifications and variations may be made without departing from the spirit and scope of the invention as defined by the claims. All publications, patents, and patent applications cited herein are incorporated by reference in their entirety.