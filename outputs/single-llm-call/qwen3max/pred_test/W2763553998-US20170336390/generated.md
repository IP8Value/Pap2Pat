# DESCRIPTION

## STATEMENT REGARDING FEDERALLY FUNDED RESEARCH

This invention was made with government support under Grant No. [REDACTED] awarded by the National Institutes of Health. The government has certain rights in the invention.

## BACKGROUND

The interrogation of biological systems to understand molecular structure, conformational dynamics, and interactions has long relied on a suite of analytical techniques that operate at different levels of biological organization. Among these, genome analysis provides a foundational map of an organism’s complete set of DNA, including all of its genes. This approach enables the identification of genetic variants, mutations, and regulatory elements but offers limited insight into functional protein states or dynamic structural changes that occur post-translationally. Exome analysis, which focuses specifically on the protein-coding regions of the genome, refines this view by targeting approximately 1–2% of the human genome that directly informs protein synthesis. While exome sequencing is efficient for diagnosing Mendelian disorders and identifying disease-associated coding variants, it similarly fails to capture the three-dimensional architecture or solvent accessibility of proteins in their native environments.

Transcriptome analysis extends the investigation to the RNA level, quantifying gene expression through methods such as RNA sequencing (RNA-seq). This technique reveals which genes are actively transcribed under specific conditions, offering clues about cellular states and responses. However, transcript abundance does not always correlate with protein abundance due to post-transcriptional regulation, translational control, and protein degradation rates. Consequently, transcriptomics cannot reliably predict protein conformation, folding, or interaction interfaces. Proteome analysis addresses this gap by cataloging the full complement of proteins in a biological sample, often using mass spectrometry (MS)-based platforms. While proteomics identifies protein presence, quantity, and some post-translational modifications, it typically requires denaturation and digestion, thereby losing critical information about higher-order structure and spatial relationships within intact macromolecules.

Despite advances in these omics technologies, a significant limitation persists: none provide direct, residue-level information about the solvent accessibility of amino acids in folded proteins under near-native conditions. Traditional structural biology methods such as X-ray crystallography and cryo-electron microscopy yield high-resolution static snapshots but often require non-physiological conditions like crystallization or freezing, which may distort dynamic conformations. Nuclear magnetic resonance (NMR) spectroscopy can probe solution-state structures but is limited by protein size, solubility, and complexity. To bridge this gap, protein footprinting techniques have emerged, wherein covalent or noncovalent labeling of solvent-exposed regions is coupled with MS readout to infer structural features. Hydrogen deuterium exchange (HDX), for instance, measures backbone amide hydrogen exchange rates as proxies for solvent exposure. Yet HDX suffers from technical challenges including rapid back-exchange, requirement for low-temperature quenching, acidic digestion conditions, and indirect inference of side-chain accessibility. Chemical labeling methods using hydroxyl radicals offer more direct side-chain modification but have been constrained by the need for synchrotron radiation facilities or laser-based systems requiring hazardous reagents like hydrogen peroxide. These limitations hinder widespread adoption in clinical, industrial, and academic settings where benchtop, reagent-free, and rapid structural interrogation is needed.

## SUMMARY

The present invention introduces a plasma-induced oxidation method for modifying biological molecules in a controlled and reproducible manner to assess their structural features, particularly solvent accessibility. This method involves generating plasma directly within or adjacent to a fluid sample containing biological molecules, thereby producing short bursts—on the order of microseconds—of reactive marker radicals, primarily hydroxyl radicals, derived from the solvent itself without the addition of exogenous chemical reagents. The marker radicals diffuse into the sample and selectively modify solvent-accessible regions of biological molecules, such as amino acid side chains in proteins or nucleotide bases in nucleic acids, in a dose-dependent fashion. By controlling plasma parameters, the extent and specificity of modification can be precisely tuned, enabling quantitative mapping of surface topology and conformational changes.

The invention further encompasses methods for modifying multiple biological molecules simultaneously within a single sample or across multiple samples, facilitating high-throughput structural profiling. A key application is the determination of solvent accessibility: regions of a biomolecule that are buried in the native folded state exhibit reduced oxidation compared to exposed regions, and this differential labeling can be quantified using analytical techniques such as mass spectrometry, gel electrophoresis, or sequencing. The method is also adaptable to comparative studies—for example, assessing structural differences between ligand-bound and unbound states of a receptor—by treating parallel subsamples and analyzing oxidation disparities.

In addition to methods, the invention includes integrated systems designed to implement plasma-induced oxidation. These systems feature a sample chamber, electrodes for plasma generation (including a plasma electrode and a ground electrode), a controllable power supply capable of delivering high-voltage pulses, and a feedback-controlled environment to regulate temperature, gas composition, and exposure duration. A plasma jet configuration is also disclosed, wherein a directed stream of ionized gas delivers radicals to the sample surface with spatial precision. The system may incorporate automation for sample loading, parameter optimization, and real-time monitoring via integrated analytical devices.

Compositions of matter are claimed, including a biological molecule in contact with a fluid medium during plasma exposure, thereby forming a transient complex with marker radicals; mixtures containing multiple marker radical precursors to enable multiplexed labeling; and synthetic biological molecules engineered to exhibit predictable oxidation responses for use as internal standards. Furthermore, kits are provided that include reference samples with known solvent accessibility profiles, along with instructions for calibrating and validating the plasma-induced oxidation system, ensuring reproducibility across laboratories and applications.

## DETAILED DESCRIPTION

### Methods

The scope of the invention encompasses methods for modifying biological molecules using plasma-generated marker radicals to derive structural and conformational information. As used herein, “biological molecule” refers to proteins, peptides, nucleic acids, lipids, carbohydrates, or complexes thereof, in purified form or within cellular contexts. “Marker radicals” are reactive species—predominantly hydroxyl radicals (•OH)—generated from the dissociation of water or other solvents under plasma conditions, which covalently modify susceptible residues in a diffusion-limited manner reflective of solvent exposure.

A representative method begins by placing a biological sample in a suitable container, such as a microcentrifuge tube or a dedicated sample chamber, optionally suspended in a buffer solution. The buffer may be selected to maintain physiological pH (e.g., phosphate-buffered saline, Tris-HCl, or ammonium acetate) and may include stabilizers to prevent aggregation or denaturation. The pH is typically maintained between 6.0 and 8.5 to preserve native conformation while allowing efficient radical chemistry. A plasma is then generated in proximity to or within the fluid sample using a dielectric-barrier discharge configuration, wherein a high-voltage pulse is applied across a gap between a powered electrode and a grounded counter-electrode, with the sample serving as part of the discharge path or being exposed to a plasma plume.

Plasma generation is controlled via precise electrical parameters: voltage amplitude (typically 1–31 kV), pulse width (microseconds to milliseconds), frequency (0–15 kHz), and total exposure duration (milliseconds to minutes). These settings determine the peak and average concentrations of marker radicals produced. For instance, a single microsecond pulse may generate a burst of ~350 nmol/sec of hydroxyl radicals, as calibrated using methionine oxidation kinetics. The system may deliver a single pulse or a sequence of pulses, with inter-pulse intervals sufficient to allow radical recombination and thermal equilibration, thereby minimizing cumulative heating. Temperature elevation is actively managed—via Peltier cooling, circulating coolant, or short duty cycles—to remain below thresholds that induce denaturation (typically <10°C above ambient).

Marker radical precursors may be introduced either via the gaseous feedgas (e.g., humidified argon or helium containing water vapor) or directly into the liquid sample (e.g., dissolved oxygen or trace peroxides). However, in the preferred embodiment, no exogenous precursors are added; radicals are generated solely from the aqueous solvent, ensuring minimal perturbation of the biological system. The radicals diffuse into the sample and react with electron-rich side chains (e.g., Met, Cys, Trp, Tyr, Phe, His) or nucleotide bases, forming stable oxidative adducts (+16 Da for sulfoxides, hydroxylated aromatics, etc.).

After a defined incubation period—ranging from seconds to minutes—the reaction is quenched by cooling or chemical scavenging. The modified biomolecules are then analyzed using mass spectrometry, where shifts in mass-to-charge ratio indicate site-specific modifications. Comparative analysis between native and denatured (e.g., predigested or chemically unfolded) samples reveals solvent-inaccessible regions: residues showing increased oxidation only after denaturation were originally buried. Alternatively, cleavage factors such as proteases or nucleases may be introduced post-labeling to expose protected domains, followed by secondary oxidation to map hidden epitopes.

Quantitative assessment involves measuring oxidation levels across replicate samples, normalizing to internal standards, and generating reports that highlight differentially accessible regions. Statistical analysis (e.g., t-tests) identifies significant changes, enabling conclusions about conformational shifts induced by ligands, mutations, or environmental conditions.

### Systems

Systems for implementing the invention comprise a sample chamber constructed of chemically inert materials (e.g., glass, quartz, or polypropylene) with dimensions optimized for plasma-sample interaction (e.g., 0.1–1 mL volume). The chamber is positioned between a plasma electrode—typically a sharpened metal needle or planar conductor made of tungsten, stainless steel, or platinum—and a ground electrode, which may double as a cooling plate. A high-voltage power supply, driven by a waveform generator and amplifier, delivers precisely timed pulses under software control. A dielectric barrier (e.g., ceramic or polymer coating) prevents arcing and ensures uniform discharge.

The system includes a control unit with a user interface for setting operational parameters (voltage, frequency, duration) and monitoring real-time feedback from temperature sensors, current probes, or optical emission detectors. Automated sample handling may be integrated via a hopper and robotic arm for high-throughput applications. Analytical devices such as inline mass spectrometers or capillary electrophoresis modules enable immediate readout. In a plasma jet variant, the discharge is confined to a nozzle, projecting a focused radical stream onto the sample surface; translation devices allow raster scanning for spatially resolved labeling. Protective housing with interlocked doors ensures operator safety, while gas manifolds regulate feedgas composition and flow. Optional oil layers over aqueous samples can modulate radical diffusion kinetics.

### Compositions of Matter

Compositions include a biological molecule transiently associated with plasma-generated marker radicals in an aqueous medium, forming a labeled adduct useful for structural probing. Mixtures may contain multiple radical precursors (e.g., H₂O, O₂, N₂) to generate diverse reactive species (•OH, •NO, O₃⁻) for multiplexed labeling. Synthetic biological molecules—such as engineered peptides with known solvent-exposed residues—are included as calibration standards exhibiting predictable oxidation kinetics.

### Kits

Kits comprise reference samples (e.g., cytochrome c, BSA) with documented oxidation profiles, along with protocols for system validation. A second kit type includes components for determining optimal plasma configuration based on sample type, enabling users to establish dose-response curves for new analytes.

## EXAMPLE 1

### Labeling Cytochrome C

A plasma generation system was assembled with a tungsten needle electrode positioned 1 mm above a 100 µL sample of 50 µM cytochrome c in 50 mM ammonium acetate (pH 7.0), cooled to 4°C. Plasma was generated using 10 kV pulses at 5 kHz for durations of 0–90 seconds. Mass spectrometry revealed progressive oxidation of methionine and tryptophan residues, with +16 Da adducts increasing linearly with exposure time, confirming dose-dependent labeling without protein fragmentation.

## EXAMPLE 2

### Labeling Bovine Serum Albumin

Bovine serum albumin (50 µM in PBS) was subjected to plasma treatment for 0, 15, 30, and 60 seconds. Intact protein analysis by SDS-PAGE showed no degradation, while tryptic digest MS/MS identified oxidation hotspots in solvent-exposed loops (e.g., residues 100–110) and minimal modification in hydrophobic cores (e.g., residues 387–399), consistent with known BSA structure.

## EXAMPLE 3

### Breakdown of DNA in Size-Dependent and Exposure-Dose-Dependent Fashion

Lambda DNA (50 ng/µL) was treated with plasma for 0–120 seconds. Agarose gel electrophoresis demonstrated progressive fragmentation, with shorter fragments accumulating at longer exposures, indicating radical-mediated strand scission proportional to both DNA length and plasma dose.

## EXAMPLE 4

### Protein Labeling in Intact/Live Cells

Live *E. coli* cells in LB broth were exposed to 5-second plasma pulses. Subsequent lysis and LC-MS/MS detected oxidation of outer membrane proteins (e.g., OmpA), confirming plasma penetration and selective labeling of surface-exposed domains in viable cells.

### EXAMPLE 5

Intact cytochrome c and urea-denatured cytochrome c were treated identically with 30-second plasma exposure. Denatured samples exhibited 3.2-fold higher average oxidation per residue, demonstrating that plasma labeling reflects conformational state.

### EXAMPLE 6

Native and trypsin-predigested bovine serum albumin were plasma-treated and analyzed. Predigested samples showed elevated oxidation in previously buried regions (e.g., residues 310–318), enabling mapping of solvent-inaccessible domains through comparative quantification.

### EXAMPLE 7

The ectodomain of EGFR was labeled with and without EGF binding. MS/MS revealed significantly reduced oxidation (p<0.05) in dimerization interface residues (e.g., Q193-C195, D563-H566) upon EGF addition, corroborating crystallographic models of ligand-induced conformational shielding.