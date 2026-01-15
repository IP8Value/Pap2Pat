# DESCRIPTION

## PRIORITY DATA

- claim priority of provisional patent application

This invention claims the benefit of priority under 35 U.S.C. § 119(e) to U.S. Provisional Patent Application No. 62/XXX,XXX, filed on [Insert Filing Date], entitled “Redox-Sensitive Conjugates for Real-Time Monitoring of Cellular Redox State,” the entire disclosure of which is hereby incorporated by reference in its entirety. The provisional application discloses the composition, synthesis, and functional validation of a disulfide-linked conjugate comprising a cell-penetrating peptide moiety covalently attached via a reducible bond to a cargo peptide bearing a fluorophore-quencher pair. The provisional application further describes the use of this conjugate to detect dynamic changes in intracellular redox potential in living cells, including its application in high-throughput screening for redox-modulating agents and its utility in optical imaging modalities. The present application expands upon these disclosures by providing detailed structural specifications, methodological protocols for quantifying cellular uptake and reduction kinetics, and novel applications in cardiac arrhythmia screening and scintigraphic imaging, all of which constitute non-obvious advancements beyond the scope of the provisional filing.

## BACKGROUND OF THE INVENTION

- introduce cell penetrating peptides
- describe properties of model amphipathic peptide
- discuss mechanisms of cell peptide internalization

Cell-penetrating peptides represent a class of short amino acid sequences capable of traversing the plasma membrane of mammalian cells without the need for endocytic machinery or exogenous transfection agents. These peptides have garnered significant interest in biomedical research due to their ability to deliver otherwise impermeable molecular cargoes—including nucleic acids, proteins, and small-molecule probes—into the cytosol and nucleus of living cells. Among the most extensively studied of these peptides is the model amphipathic peptide, a synthetic sequence composed of alternating lysine and leucine-alanine residues that adopts a stable alpha-helical conformation under physiological conditions. This structural arrangement results in a distinct spatial segregation of hydrophobic and cationic residues along opposing faces of the helix, enabling interactions with lipid bilayers through both electrostatic and hydrophobic forces. Unlike many other cell-penetrating peptides that rely on receptor-mediated endocytosis or membrane disruption, the model amphipathic peptide exhibits a unique capacity for direct translocation across the plasma membrane, even under conditions of metabolic inhibition or low temperature, suggesting a mechanism distinct from classical endocytic pathways. The internalization process is not driven by concentration gradients, nor is it reversible upon washing, indicating a stable integration into the intracellular compartment. While early studies proposed that uptake occurs via transient membrane perturbations or inverted micelle formation, more recent evidence suggests that the peptide’s net positive charge facilitates movement along the transmembrane electrical potential gradient, a phenomenon modulated by the redox state of membrane-associated ion channels. This interplay between peptide charge, membrane potential, and cellular redox environment has not been previously exploited for diagnostic or therapeutic purposes, despite its clear implications for targeted delivery systems. The model amphipathic peptide’s ability to bypass endosomal entrapment, coupled with its resistance to proteolytic degradation in the cytosol, renders it uniquely suited for applications requiring sustained intracellular delivery of redox-sensitive reporters.

## DETAILED DESCRIPTION

- provide definitions of terminology used

### Definitions

- define "substantially"

The term “substantially” as used herein refers to a degree of completeness, consistency, or correspondence that is sufficient to achieve the intended function or result without requiring absolute precision or total uniformity. In the context of this invention, “substantially” may describe the extent to which a conjugate retains its structural integrity, fluorescence properties, or cellular uptake efficiency under defined experimental conditions. For example, a reductide conjugate is considered substantially intact if it maintains its disulfide linkage and fluorophore-quencher pairing during storage, handling, and exposure to extracellular media for up to 24 hours, such that its redox-dependent fluorescence response remains distinguishable from background noise. Similarly, a cell population is described as substantially reduced if the majority of cells exhibit a fluorescence signal indicative of intracellular disulfide reduction, as determined by statistical comparison with control populations under identical conditions.

- define "about"

The term “about” as employed herein denotes a range of variation around a stated numerical value that accounts for normal experimental variability inherent in biological and chemical measurements. This variation is typically within ±10% of the recited value, unless otherwise specified, and encompasses minor deviations arising from instrument calibration, reagent purity, temperature fluctuations, or cell-to-cell heterogeneity. For instance, a concentration of “about 1 μM” reductide includes concentrations ranging from 0.9 μM to 1.1 μM, and a time period of “about 30 minutes” encompasses durations from 27 to 33 minutes. This term is used to ensure that claims are not unduly limited by minor, non-critical deviations that do not affect the inventive principle or functional outcome.

- describe presentation of lists

Lists presented in this disclosure, whether in the form of sequences, compounds, or method steps, are intended to be interpreted as inclusive and non-exhaustive unless explicitly stated otherwise. The use of terms such as “including,” “comprising,” or “such as” does not limit the scope of the invention to the enumerated elements but rather serves to provide illustrative examples. For example, a list of cell-penetrating peptides includes, but is not limited to, the model amphipathic peptide, penetratin, TAT, and transportan, and encompasses any peptide with analogous amphipathic structure and membrane translocation capability. Similarly, a list of redox-modifying agents includes, but is not limited to, N-acetylcysteine, hydrogen peroxide, diamide, and l-buthionine sulfoximine, and extends to any compound capable of altering the intracellular glutathione redox couple or thioredoxin system.

- interpret numerical ranges

Numerical ranges disclosed herein, whether expressed as intervals, minimum-maximum values, or percent deviations, are to be interpreted as continuous and inclusive of all intermediate values within the stated bounds. For example, a range of “from 1 μM to 10 μM” encompasses every concentration between and including 1 μM and 10 μM, including fractional values such as 2.3 μM or 7.8 μM. Such ranges are intended to cover all possible concentrations that yield the claimed functional effect, without requiring explicit enumeration of every possible value.

- provide examples of numerical ranges

Examples of numerical ranges used in this invention include, but are not limited to, incubation times of 5 to 60 minutes for cellular uptake, reductide concentrations of 0.1 to 10 μM for fluorescence detection, glutathione pool concentrations of 1 to 10 mM, and H₂O₂ treatment doses ranging from 50 to 800 μM. These ranges reflect empirically determined conditions under which the conjugate exhibits a measurable, reproducible, and statistically significant response to redox modulation.

- describe interpretation of numerical ranges

Numerical ranges in this disclosure are interpreted in light of the functional context in which they are applied. A concentration range is not merely a set of permissible values but represents a window of efficacy in which the conjugate reliably reports redox state without inducing cytotoxicity, non-specific quenching, or signal saturation. A time range reflects the duration necessary to achieve equilibrium between uptake and reduction kinetics, and is not intended to imply that values outside the range are ineffective, but rather that they may yield diminished signal-to-noise ratios or altered biological interpretations.

### Invention

- introduce conjugates for detecting cellular redox state

The present invention provides novel molecular conjugates designed to detect and quantify dynamic changes in the intracellular redox state of living cells through a mechanism that integrates cellular uptake kinetics with redox-dependent fluorescence activation. These conjugates consist of two covalently linked peptide segments connected by a disulfide bond, such that the fluorescence signal generated upon reduction serves as a direct readout of the reducing capacity of the intracellular environment. Unlike prior art systems that rely on passive diffusion or endocytic uptake, the conjugates of this invention exploit the redox-sensitive internalization properties of a model amphipathic peptide to enable both spatial and temporal resolution of redox changes in real time, without requiring genetic modification or invasive instrumentation.

- describe first segment of conjugate

The first segment of the conjugate comprises a cell-penetrating peptide, specifically a model amphipathic peptide sequence consisting of alternating lysine and leucine-alanine residues, which is covalently modified at its N-terminus with a fluorophore capable of emitting a stable, non-quenched signal regardless of redox state. This segment is responsible for mediating the translocation of the entire conjugate across the plasma membrane and for providing a consistent internalization marker that is independent of disulfide reduction. The sequence is designed to maintain its amphipathic helical structure under physiological conditions, ensuring consistent membrane interaction and uptake efficiency across diverse cell types.

- describe second segment of conjugate

The second segment of the conjugate comprises a non-cell-penetrating cargo peptide, which is covalently linked at its N-terminus to a second fluorophore that is quenched in proximity to the first fluorophore when the disulfide bond remains intact. This cargo peptide is selected for its inability to traverse the plasma membrane independently, ensuring that its intracellular detection is contingent upon successful delivery via the first segment. The sequence is optimized for solubility, stability, and minimal aggregation, and is designed to be released upon disulfide reduction, thereby enabling unquenching of the second fluorophore and generation of a detectable fluorescence signal.

- describe redox-sensitive linkage

The first and second segments are joined by a disulfide bond, a reversible covalent linkage that is selectively cleaved under reducing conditions within the intracellular milieu, particularly in environments rich in reduced glutathione or thioredoxin. This linkage is chemically stable in extracellular media and under oxidizing conditions, ensuring that fluorescence activation occurs exclusively upon internalization and reduction. The disulfide bond is positioned to maximize steric proximity between the two fluorophores, thereby ensuring efficient fluorescence resonance energy transfer (FRET) quenching prior to reduction and robust signal dequenching upon cleavage.

- provide example of fluorophore/quencher pair

In a preferred embodiment, the first fluorophore is 5(6)-carboxytetramethylrhodamine (TAMRA), which emits at approximately 590 nm upon excitation at 530 nm and remains fluorescent regardless of disulfide status, while the second fluorophore is fluorescein amidite (FAM), which emits at approximately 528 nm upon excitation at 485 nm and is quenched by TAMRA in the intact conjugate. Upon reduction of the disulfide bond, FAM fluorescence increases by more than tenfold, providing a highly sensitive and ratiometric readout of intracellular redox state.

- describe cell penetrating peptide

The cell-penetrating peptide is derived from the model amphipathic peptide sequence KLALKLALKALKAALKLA-NH₂, which has been modified to include an N-terminal cysteine residue for disulfide conjugation. This sequence is synthesized using standard Fmoc solid-phase peptide chemistry and is purified to homogeneity by reverse-phase high-performance liquid chromatography. The peptide exhibits no cytotoxicity at concentrations up to 10 μM and retains its membrane translocation capacity across a broad range of cell types, including human fibroblasts, cardiomyocytes, and cancer cell lines.

- provide examples of cell penetrating peptides

Other cell-penetrating peptides that may be substituted for the model amphipathic peptide include penetratin, TAT (transactivator of transcription), transportan, and polyarginine sequences of eight or more arginine residues. These peptides may be similarly modified with an N-terminal cysteine and conjugated to the same cargo-fluorophore pair, provided they retain the capacity for non-endocytic internalization and are not subject to rapid degradation in the cytosol.

- describe cargo peptide

The cargo peptide is a short, non-cell-penetrating sequence of six amino acids, specifically CLKANL, chosen for its lack of intrinsic membrane permeability and its compatibility with N-terminal labeling. This sequence is synthesized with an N-terminal cysteine residue to enable disulfide bonding and is labeled with FAM to serve as the redox-sensitive reporter. Alternative cargo peptides may include sequences of 4 to 10 amino acids that lack basic or hydrophobic motifs sufficient to mediate membrane translocation independently.

- provide examples of cargo peptides

Examples of alternative cargo peptides include CQKANL, CKLANK, and CRLANL, each of which has been tested and found to maintain the required quenching characteristics and reduction sensitivity when conjugated to the model amphipathic peptide via disulfide linkage. The cargo peptide may also be substituted with non-peptidic moieties such as small-molecule fluorophores or radiolabels, provided they remain quenched in the intact conjugate and are released upon disulfide reduction.

- describe method of detecting cellular redox state

The method of detecting cellular redox state involves incubating a living cell population with the conjugate for a period sufficient to allow uptake and intracellular reduction, followed by measurement of fluorescence emission at the wavelengths specific to the two fluorophores. The ratio of FAM to TAMRA fluorescence is calculated to generate a redox index that correlates with the intracellular glutathione redox potential. This index is independent of cell density, conjugate concentration, or instrument sensitivity, and provides a quantitative, real-time readout of redox state in single cells or populations.

- describe detection of uptake of conjugate by cell

Detection of conjugate uptake is achieved by monitoring the TAMRA fluorescence signal, which increases over time as the conjugate is internalized, regardless of disulfide reduction. This signal serves as a direct indicator of cellular internalization efficiency and can be used to normalize FAM fluorescence data across different cell types or treatment conditions. Uptake is confirmed by fluorescence microscopy, flow cytometry, or plate reader assays, and is distinguishable from background signal by its time-dependent increase and resistance to extracellular washout.

- describe measurement of redox state

Measurement of redox state is performed by quantifying the increase in FAM fluorescence following conjugate internalization, which occurs only upon reduction of the disulfide bond. The magnitude and kinetics of this increase are directly proportional to the reducing capacity of the intracellular environment, as demonstrated by correlation with established redox sensors such as roGFP. The measurement may be performed in live cells using fluorescence microscopy, plate readers, or flow cytometers, and is unaffected by cell viability, proliferation rate, or metabolic activity unrelated to redox balance.

- describe monitoring of cellular redox state

Monitoring of cellular redox state is achieved through repeated measurements of FAM and TAMRA fluorescence over time, enabling the construction of kinetic profiles that reflect dynamic changes in redox potential in response to stimuli such as oxidative stress, antioxidant treatment, or pharmacological intervention. This method allows for longitudinal tracking of redox shifts in individual cells or populations, providing insights into temporal patterns of redox signaling that are inaccessible to endpoint assays.

- provide examples of applications of conjugate and method

The conjugate and method find utility in a broad range of applications, including the high-throughput screening of compounds that modulate intracellular redox state, the identification of novel cardiac antiarrhythmic agents that restore redox homeostasis in cardiomyocytes, the real-time optical imaging of tumor microenvironments, and the scintigraphic detection of redox-altered tissues in vivo using radiolabeled versions of the conjugate.

- describe use in discovering redox modifying agents

The conjugate enables the discovery of novel redox-modifying agents by permitting high-throughput screening of chemical libraries in live cells, with fluorescence readouts serving as a direct functional readout of intracellular reduction potential. Compounds that increase FAM signal are identified as reducing agents, while those that decrease signal are classified as oxidizing agents, regardless of their known biochemical targets.

- describe use in discovering cardiac antiarrhythmic agents

In cardiomyocytes, aberrant redox state is a known contributor to arrhythmogenesis. The conjugate is used to screen for compounds that normalize redox imbalance in models of ischemia-reperfusion injury or oxidative stress-induced arrhythmia. Agents that restore FAM signal to baseline levels in stressed cells are prioritized as potential antiarrhythmic candidates, offering a functional, mechanism-based approach to drug discovery.

- describe use in optical imaging

The conjugate is suitable for optical imaging in live tissues, including skin, cornea, and explanted organs, where its cell-penetrating properties and redox-specific fluorescence enable visualization of regional redox heterogeneity. This application is particularly valuable in monitoring wound healing, tumor progression, or inflammatory responses.

- describe use in scintigraphic imaging

By replacing the fluorophores with radionuclides such as technetium-99m or iodine-123, the conjugate may be adapted for scintigraphic imaging, enabling non-invasive detection of redox-altered tissues in vivo. This application is especially relevant for identifying regions of oxidative stress in myocardial infarction, neurodegenerative disease, or cancer metastasis.

### Materials and Methods

- list reagents used

The reagents used in the synthesis and validation of the conjugate include N-acetylcysteine, 1-chloro-2,4-dinitrobenzene, l-buthionine sulfoximine, diamide, reduced and oxidized glutathione, Dulbecco’s Modified Eagle Medium, fetal bovine serum, puromycin, tris-buffered saline, acetic acid, and the Screen-Well™ REDOX library of 84 compounds.

- describe peptide synthesis and labeling

Peptides were synthesized using standard Fmoc solid-phase chemistry on Rink amide resin, with N-terminal cysteine residues protected by Acm groups. After cleavage and deprotection, peptides were purified by preparative HPLC and characterized by mass spectrometry. TAMRA and FAM were conjugated to the N-termini of the respective peptides via NHS ester chemistry, followed by oxidation to form the disulfide-linked conjugate.

- describe cell culture and transfection

Human fibroblasts (BJ, IMR90) and rat cardiomyocytes (H9c2) were cultured in DMEM supplemented with 10% fetal bovine serum and 2 mM L-glutamine. Stable expression of roGFP was achieved by retroviral transduction of PLAT-E cells, followed by selection with puromycin and validation by fluorescence microscopy and Western blot.

- describe reductide assay in GSH containing buffer

Reductide was dissolved in 3% acetic acid and diluted to 1 μM in TBS buffer containing varying ratios of GSH and GSSG. Fluorescence was measured at 485/528 nm (FAM) and 530/590 nm (TAMRA) using a Synergy HT plate reader over 60 minutes.

- describe fluorescence microscopy

Cells seeded on glass coverslips were incubated with 5 μM reductide in live-cell imaging medium at 37°C and 5% CO₂. Images were acquired every 15 minutes for 4.5 hours using an Olympus FV1000 confocal microscope with appropriate filter sets for TAMRA and FAM.

- compare reductide signal and roGFP

H9c2 cells expressing roGFP were treated with NAC or H₂O₂, imaged using a BD Pathway Bioimager, and subsequently incubated with reductide. Ratiometric roGFP signals were compared pixel-by-pixel with FAM intensities to determine correlation coefficients.

- describe reductide plate reader assay in cells

Cells were pretreated with redox-modifying agents, washed, and incubated with 1 μM reductide for 1 hour. Fluorescence was measured using a Synergy HT plate reader, with excitation and emission settings as above.

- describe effect of redox modifying agents

Treatment with NAC increased FAM signal in a dose-dependent manner, while CDNB, diamide, and BSO significantly suppressed signal. H₂O₂ induced biphasic responses, with low doses enhancing and high doses suppressing fluorescence.

- describe comparison with monochlorobimane

Monochlorobimane fluorescence was measured in parallel under identical conditions. Unlike reductide, monochlorobimane signal showed no significant correlation with pretreatment dose or agent type, demonstrating superior sensitivity and specificity of the conjugate.

- describe comparison with Alamar Blue

Alamar Blue fluorescence, a measure of metabolic activity, showed discordant trends with reductide signal in cells treated with 200–400 μM H₂O₂, confirming that reductide reports redox state independently of viability.

- describe flow cytometry

IMR90 cells were treated with NAC or H₂O₂, incubated with reductide for 3, 15, or 30 minutes, and analyzed by flow cytometry. TAMRA and FAM signals were measured in the FL2 and FL1 channels, respectively, with DAPI used for viability gating.

- describe statistical analysis

Data are presented as mean ± standard deviation. Comparisons between two groups were performed using Student’s t-test; multiple groups were analyzed by one-way ANOVA with Tukey’s post hoc test. P-values less than 0.05 were considered statistically significant.

- describe data presentation

All figures present representative data from at least three independent experiments, with n ≥ 6 biological replicates per condition. Error bars represent standard deviation, and asterisks denote statistical significance.

- describe statistical comparison

Statistical comparisons were performed using GraphPad Prism software. Normality was assessed using the Shapiro-Wilk test, and variance homogeneity was confirmed using Levene’s test.

- describe significance of p-values

P-values less than 0.05 indicate statistically significant differences, while p-values less than 0.01 indicate highly significant differences. All reported p-values are two-tailed.

### Statistical Analysis

- describe statistical analysis

Statistical analysis was conducted using standard parametric tests, with data meeting assumptions of normality and homogeneity of variance. For non-parametric comparisons, the Mann-Whitney U test was employed. Correlation coefficients were calculated using Pearson’s method, and linear regression analysis was used to assess the relationship between reductide signal and roGFP ratios. All analyses were performed in triplicate across independent biological replicates to ensure reproducibility and robustness.

## Results

### Effects of GSH/GSSG on Reductide Redox-Dependent Fluorescence

- introduce reductide

Reductide is a disulfide-linked conjugate composed of a model amphipathic peptide labeled with TAMRA and a non-cell-penetrating cargo peptide labeled with FAM. In its intact state, FAM fluorescence is quenched by proximity to TAMRA, but upon intracellular reduction, the disulfide bond is cleaved, resulting in a tenfold increase in FAM emission.

- describe fluorescence quenching

Fluorescence quenching is mediated by Förster resonance energy transfer between TAMRA and FAM, with an efficiency exceeding 90% in the intact conjugate. This quenching is stable in extracellular buffer and is only reversed upon reduction of the disulfide bond.

- describe GSH/GSSG buffer preparation

Buffers were prepared by dissolving GSH and GSSG in tris-buffered saline to yield a total glutathione concentration of 5 mM, with varying GSH/GSSG ratios to simulate different redox potentials. The pH was maintained at 7.4 throughout all experiments.

- show FAM emission intensity in GSH buffer

In buffer containing 5 mM GSH, FAM fluorescence increased rapidly over 20 minutes, reaching a maximum intensity that was 12-fold higher than baseline. This increase was linear with time and independent of TAMRA signal.

- show FAM emission intensity in GSH/GSSG buffer

When GSSG was added to the buffer, the rate of FAM signal development decreased proportionally to the GSSG concentration, and the maximum intensity achieved was reduced by up to 60%. This demonstrated that the reduction kinetics are sensitive to the glutathione redox potential.

- describe TAMRA emission intensity

TAMRA emission intensity increased with higher GSH concentrations but did not vary over time, indicating that TAMRA fluorescence is unaffected by disulfide reduction and serves as a stable internal control.

- show TAMRA emission intensity in GSH buffer

In 5 mM GSH, TAMRA fluorescence remained constant over 60 minutes, with no significant change in intensity.

- show TAMRA emission intensity in GSH/GSSG buffer

Similarly, in GSH/GSSG mixtures, TAMRA signal showed no time-dependent variation, confirming its utility as a reference for normalization.

- describe live cell imaging

Live cell imaging of BJ fibroblasts revealed that TAMRA signal accumulated uniformly in the cytoplasm, while FAM signal was initially quenched but became detectable after 15–30 minutes in reduced cells.

- show TAMRA and FAM signals in reduced cells

In cells pretreated with NAC, FAM signal emerged within 15 minutes and reached peak intensity by 45 minutes, coinciding with TAMRA accumulation.

- show TAMRA and FAM signals in oxidized cells

In cells pretreated with CDNB, FAM signal was delayed by over 60 minutes and remained significantly lower than in reduced cells, despite equivalent TAMRA uptake.

- describe exocytic vesicles

At later time points, FAM-labeled cargo was observed to be exported from cells via exocytic vesicles, leading to homogeneous extracellular fluorescence, while TAMRA remained intracellular.

- summarize cellular distribution of reductide

The conjugate distributes pan-cytosolically, with minimal nuclear localization of TAMRA and moderate nuclear accumulation of FAM following reduction. Exocytosis of the cargo peptide provides a mechanism for signal clearance and prevents signal saturation.

### Flow Cytometry

- describe flow cytometry results

Flow cytometry confirmed a time-dependent increase in both TAMRA and FAM fluorescence in cells incubated with reductide. TAMRA signal increased steadily over 30 minutes, while FAM signal exhibited a delayed but pronounced rise, consistent with the kinetics of disulfide reduction. Cells pretreated with NAC showed significantly higher FAM/TAMRA ratios than those pretreated with H₂O₂.

### Reductide Response to a Small Library of Redox Modifying Compounds

- describe experimental design

A library of 84 redox-modifying compounds was screened at 50 μM for 24 hours in BJ fibroblasts, followed by 4-hour incubation with 1 μM reductide. FAM fluorescence was measured to identify compounds that altered intracellular redox state.

- show FAM signal changes in response to redox modifying compounds

Of the 84 compounds, 65 (77.4%) significantly increased FAM signal, indicating a reducing effect, while 9 (10.7%) significantly decreased it, indicating an oxidizing effect. The remaining 10 compounds had no significant effect.

- summarize results

The screen identified multiple known antioxidants as potent reducers of the intracellular environment, as well as several unexpected oxidizing agents. Notably, some compounds that induced cell death still increased FAM signal, confirming that reductide reports redox state independently of viability.

## Discussion

### Reductide Uptake as Well as Reduction Depends on Cellular Redox State

- describe reductide uptake and reduction

The development of FAM fluorescence following reductide incubation is governed by two sequential processes: cellular uptake mediated by the amphipathic peptide and intracellular reduction of the disulfide bond. Both processes are influenced by the redox state of the cell.

- describe effect of redox state on reductide signal

In cells with a reduced intracellular environment, both uptake and reduction occur more rapidly, leading to earlier and stronger FAM signal. In oxidized cells, uptake is delayed and reduction is slower, resulting in diminished signal.

- describe TAMRA signal in reduced and oxidized cells

TAMRA signal, which reflects uptake alone, is consistently higher in reduced cells, indicating that the redox state modulates the internalization process independently of disulfide reduction.

- summarize redox-dependent differences in reductide signal

The combined dependence of uptake and reduction on redox state enables reductide to serve as a highly sensitive, dual-parameter reporter of intracellular redox dynamics, with superior resolution compared to single-mechanism sensors.

### Pro-Oxidants Activate an Antioxidative Response

- describe effect of H2O2 on FAM fluorescence

Low doses of H₂O₂ (200–400 μM) induced a delayed increase in FAM fluorescence after 24 hours, indicating an adaptive antioxidative response. Higher doses (≥600 μM) suppressed FAM signal, consistent with acute oxidative damage.

- describe antioxidative response to low dose H2O2

Low-dose H₂O₂ triggered upregulation of glutathione synthesis, thioredoxin reductase, and catalase, resulting in a net reductive shift that was detectable by reductide.

- describe effect of high dose H2O2 on FAM fluorescence

High-dose H₂O₂ overwhelmed endogenous antioxidant defenses, leading to depletion of reduced glutathione and suppression of FAM signal.

- describe published investigations on H2O2 effects

These findings align with prior studies showing that low-dose H₂O₂ activates the Nrf2 pathway and enhances antioxidant gene expression, while high-dose H₂O₂ induces Nrf2 nuclear exclusion and apoptosis.

- describe upregulation of antioxidative genes

Transcriptomic analysis confirmed upregulation of GCLC, GCLM, and TXNRD1 in cells treated with low-dose H₂O₂, supporting the mechanistic basis for the reductide signal increase.

- describe effect of H2O2 on apoptosis

Apoptosis was not observed at low doses but was prominent at high doses, yet reductide signal remained informative, demonstrating its independence from cell death pathways.

- summarize redox-dependent effects of H2O2

The biphasic response to H₂O₂ underscores the utility of reductide in distinguishing adaptive redox signaling from pathological oxidative stress.

## FIGURE LEGENDS

- describe FIG. 1A

Figure 1A shows the time-dependent increase in FAM fluorescence (485/528 nm) of reductide in tris-buffered saline containing 5 mM reduced glutathione, demonstrating rapid dequenching upon exposure to reducing conditions.

- describe FIG. 1B

Figure 1B illustrates the dose-dependent suppression of FAM fluorescence development in the presence of increasing concentrations of oxidized glutathione, confirming that the reduction kinetics are sensitive to the GSH/GSSG redox potential.

- describe FIG. 2A

Figure 2A presents confocal microscopy images of BJ fibroblasts incubated with reductide, showing TAMRA fluorescence (red) distributed throughout the cytoplasm and FAM fluorescence (green) appearing only after 30 minutes in untreated cells.

- describe FIG. 2B

Figure 2B shows enhanced FAM signal in cells pretreated with N-acetylcysteine, with earlier and more intense fluorescence compared to controls, demonstrating redox-dependent activation.

- describe FIG. 2C

Figure 2C displays FAM signal suppression in cells pretreated with CDNB, with minimal fluorescence even after 90 minutes, confirming that oxidation inhibits both uptake and reduction.

- describe FIG. 2D

Figure 2D reveals the exocytic export of FAM-labeled cargo into the extracellular space after 4 hours, while TAMRA remains intracellular, illustrating the dynamic redistribution of the conjugate components.

- describe FIG. 3

Figure 3 demonstrates a strong correlation between reductide FAM signal and roGFP ratiometric readings in H9c2 cardiomyocytes following H₂O₂ treatment, validating reductide as a quantitative redox sensor.

- describe FIG. 4

Figure 4 shows dose-dependent increases in FAM signal with NAC pretreatment and decreases with CDNB or H₂O₂, confirming the sensitivity of reductide to pharmacological redox modulation.

- describe FIG. 5A

Figure 5A illustrates the suppression of FAM signal in cells treated with l-buthionine sulfoximine, a glutathione synthesis inhibitor, confirming the dependence of reductide on intracellular GSH.

- describe FIG. 5B

Figure 5B shows similar suppression of FAM signal in cells treated with diamide, a thiol oxidizing agent, reinforcing the role of the glutathione redox couple in conjugate activation.

- describe FIG. 6

Figure 6 compares FAM signal in cells pretreated with redox agents before or after reductide incubation, demonstrating that redox state modulates uptake, not just reduction.

- describe FIG. 7A

Figure 7A compares reductide FAM signal with monochlorobimane fluorescence in IMR90 fibroblasts, showing that reductide is far more responsive to redox-modifying agents.

- describe FIG. 7B

Figure 7B confirms that monochlorobimane signal is largely insensitive to pretreatment with NAC or H₂O₂, highlighting the superior specificity of reductide.

- describe FIG. 8A

Figure 8A presents flow cytometry histograms showing TAMRA fluorescence intensity over time in IMR90 cells, with higher signal in NAC-pretreated cells.

- describe FIG. 8B

Figure 8B shows corresponding FAM fluorescence histograms, with delayed and lower signal in H₂O₂-treated cells, confirming redox-dependent reduction kinetics.

- describe FIG. 9

Figure 9 displays the biphasic response of reductide to the antioxidant carvedilol, which suppresses FAM signal at 50 μM but enhances it at 10 μM, illustrating the importance of concentration in redox modulation.

- describe FIG. 10

Figure 10 compares reductide signal with Alamar Blue viability assay in cells treated with H₂O₂, demonstrating that reductide reports redox state independently of metabolic activity.