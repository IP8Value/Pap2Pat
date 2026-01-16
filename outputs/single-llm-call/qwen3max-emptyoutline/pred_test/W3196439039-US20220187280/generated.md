# DESCRIPTION

## FIELD

The present invention relates generally to the field of molecular biosensors and their application in the detection and monitoring of cellular death, particularly in neurons. More specifically, the invention provides genetically encoded death indicators (GEDIs) that function as sensitive, early, and pathway-agnostic markers of neuronal death by detecting the irreversible loss of calcium homeostasis. These biosensors are engineered from modified genetically encoded calcium indicators (GECIs) and are designed to remain unresponsive to physiological calcium transients while exhibiting a robust and quantifiable fluorescence increase upon catastrophic calcium influx associated with plasma membrane rupture or failure of calcium buffering mechanisms. The GEDI platform is compatible with longitudinal live-cell imaging, high-throughput screening, and in vivo applications across multiple model organisms, including rodents and zebrafish. The invention further encompasses nucleic acid constructs, expression vectors, host cells, transgenic animals, kits, and methods for using GEDIs to study neurodegenerative diseases, screen neuroprotective compounds, identify subpopulations of death-resistant or death-sensitive neurons, and perform automated survival analysis in both in vitro and in vivo contexts. The technology enables precise temporal demarcation of the point of no return in neuronal degeneration, thereby facilitating the investigation of antecedent molecular and cellular events that drive neurodegeneration.

## BACKGROUND

Neurodegenerative diseases—including Parkinson’s disease (PD), Huntington’s disease (HD), Alzheimer’s disease (AD), amyotrophic lateral sclerosis (ALS), and frontotemporal dementia (FTD)—are characterized by the progressive dysfunction and eventual death of specific neuronal populations, leading to irreversible cognitive, motor, or behavioral deficits. While pathological hallmarks such as protein aggregates (e.g., Lewy bodies, amyloid plaques, or TDP-43 inclusions) are commonly used to define these diseases, accumulating evidence suggests that actual neuronal death—not merely the presence of aggregates—is the most reliable correlate of clinical symptom severity and progression. Consequently, the ability to accurately, sensitively, and non-invasively detect neuronal death in real time is critical for understanding disease mechanisms, evaluating therapeutic efficacy, and identifying protective or deleterious factors in neurodegeneration.

Traditional methods for assessing cell death in neurons have significant limitations. Vital dyes such as propidium iodide (PI), ethidium homodimer-1 (EthD-1), and 7-aminoactinomycin D (7-AAD) rely on the permeabilization of the plasma membrane to enter cells and bind nucleic acids, thereby labeling dead or dying cells. However, these dyes often exhibit delayed signal onset, require repeated application in longitudinal studies, and can be cytotoxic with prolonged exposure, thereby perturbing the very biological processes under investigation. Moreover, in thick tissue or in vivo settings, dye penetration is inconsistent, leading to unreliable quantification. Fluorescent protein-based reporters of apoptosis—such as those based on caspase activation—offer genetic encoding and reduced toxicity but are inherently pathway-specific. Since neurodegenerative diseases often involve a spectrum of cell death mechanisms—including apoptosis, necrosis, excitotoxicity, and autophagic cell death—reliance on a single death pathway marker can result in incomplete or biased detection of total neuronal death. Furthermore, cells may switch between death pathways when one is pharmacologically or genetically blocked, complicating interpretation.

Genetically encoded calcium indicators (GECIs), such as GCaMP and its derivatives, have revolutionized the study of neuronal activity by enabling real-time visualization of intracellular calcium dynamics. These sensors typically consist of a circularly permuted fluorescent protein fused to a calcium-binding domain (e.g., calmodulin and M13 peptide), resulting in fluorescence increases upon calcium binding. However, standard GECIs are optimized to detect transient, physiological calcium fluctuations associated with action potentials and synaptic activity, with dissociation constants (Kd) in the sub-micromolar range. In contrast, during neuronal death, cytosolic calcium levels rise dramatically—often exceeding tens to hundreds of micromolar—as extracellular calcium floods the cell due to loss of membrane integrity or failure of ion pumps. This pathological calcium surge is orders of magnitude higher than physiological transients and represents a terminal event in the degenerative cascade.

Recent advances have led to the development of organelle-targeted GECIs, such as CEPIA variants, which are engineered with higher Kd values to detect the elevated calcium concentrations found in the endoplasmic reticulum (ER) or mitochondria. These sensors are not responsive to normal neuronal firing but instead report on organellar calcium release. The inventors recognized that by relocalizing such high-Kd GECIs from intracellular organelles to the cytosol—through removal of ER retention signals—they could create a novel class of biosensors that remain silent during normal neuronal activity but fluoresce intensely only when cytosolic calcium reaches pathological, death-associated levels. This insight led to the development of genetically encoded death indicators (GEDIs), which provide a direct, real-time, and non-invasive readout of the irreversible commitment to neuronal death.

Despite the utility of existing tools, there remains a critical unmet need for a death indicator that is: (1) genetically encoded and thus suitable for long-term, non-toxic imaging; (2) agnostic to the specific cell death pathway; (3) capable of detecting death at the earliest possible moment—before morphological disintegration or DNA fragmentation; (4) quantifiable in a high-throughput, automated manner; and (5) adaptable to diverse experimental systems, including primary neurons, induced pluripotent stem cell (iPSC)-derived neurons, and whole organisms such as zebrafish. The present invention fulfills this need by providing a versatile family of GEDI biosensors that meet all these criteria and enable unprecedented precision in the study of neurodegeneration.

## SUMMARY

The present invention provides genetically encoded death indicators (GEDIs) and methods for their use in detecting and monitoring neuronal death in real time. GEDIs are engineered from modified genetically encoded calcium indicators (GECIs) that have been optimized to detect pathological, rather than physiological, elevations in cytosolic calcium concentration. Specifically, GEDIs are derived from high-affinity or high-Kd GECIs originally designed for organelle calcium sensing—such as RCEPIA or GCaMP6-150—by removing subcellular targeting sequences (e.g., ER retention signals) so that the sensor localizes to the cytosol. As a result, GEDIs exhibit minimal fluorescence response to normal neuronal activity or calcium transients but undergo a robust and sustained increase in fluorescence upon the catastrophic calcium influx that accompanies irreversible loss of plasma membrane integrity or calcium homeostasis, a hallmark of neuronal death.

In a preferred embodiment, GEDI constructs are designed as bicistronic expression cassettes comprising a GEDI fluorescent protein linked via a self-cleaving P2A peptide to a second, spectrally distinct fluorescent protein (e.g., EGFP, mApple, or TagBFP2). This configuration enables pseudo-ratiometric quantification, wherein the ratio of GEDI signal to the reference fluorescent protein signal (the “GEDI ratio”) serves as a normalized, expression-level-independent metric of cell death. A neuron is classified as dead when its GEDI ratio exceeds a predefined empirical threshold, which is determined from control populations of live and dead neurons. This approach minimizes variability due to differences in transfection efficiency, expression levels, or imaging conditions, thereby enabling high-throughput, automated survival analysis.

The invention further provides a family of GEDI variants with different spectral properties and subcellular localizations to accommodate diverse experimental needs. For example, RGEDI-P2A-EGFP uses red-shifted GEDI (RGEDI) and green reference (EGFP) for compatibility with standard filter sets; RGEDI-P2A-3xBFP employs blue fluorescent protein (BFP) as the reference to allow concurrent imaging of green biosensors; and nuclear-localized versions (e.g., RGEDI-NLS-P2A-EGFP-NLS) enhance signal resolution in dense tissues or whole-brain imaging. Additionally, GEDI variants based on GCaMP6-150 (termed GC150) offer higher calcium affinity and are particularly effective in in vivo settings where extracellular calcium availability may be limited.

Methods of use include: (1) longitudinal live-cell imaging of neuronal cultures to track individual neuron survival over time; (2) high-throughput screening of neuroprotective compounds in multi-well plates; (3) automated identification of subpopulations of neurons with differential sensitivity to toxins or disease-associated proteins; (4) survival analysis in iPSC-derived human neurons carrying disease-causing mutations; and (5) in vivo detection of neuronal death in transparent model organisms such as zebrafish larvae. The invention also enables the generation of cumulative risk-of-death (CRD) curves and Kaplan-Meier survival plots from single-cell data, providing statistical power comparable to clinical trials but at the cellular level.

Kits are provided comprising nucleic acid constructs encoding GEDI variants, along with instructions for use in cell culture, transfection, imaging, and data analysis. Transgenic animals expressing GEDI under neuron-specific promoters are also encompassed, facilitating in vivo studies of neurodegeneration without the need for repeated transfection or dye loading.

In summary, the GEDI platform offers an early, sensitive, quantitative, and pathway-agnostic method for detecting neuronal death with high temporal and spatial resolution. By providing a clear demarcation of the point of no return in neuronal degeneration, GEDIs enable researchers to investigate the molecular and cellular events that precede death, identify resilience factors, and accelerate the discovery of disease-modifying therapeutics for neurodegenerative disorders.

## DETAILED DESCRIPTION

### Definitions

As used herein, the term “genetically encoded death indicator” or “GEDI” refers to a recombinant polypeptide or nucleic acid construct that produces a fluorescent signal in response to pathological elevations in cytosolic calcium concentration associated with irreversible neuronal death. A GEDI is derived from a genetically encoded calcium indicator (GECI) that has been modified to localize to the cytosol and exhibit a dissociation constant (Kd) for calcium that renders it insensitive to physiological calcium transients but responsive to the high calcium levels (>10 µM) observed during cell death. GEDIs may be monomeric fluorescent proteins or part of a bicistronic expression system linked via a self-cleaving peptide to a reference fluorescent protein for ratiometric quantification.

The term “neuronal death” refers to the irreversible loss of neuronal viability, characterized by the failure of calcium homeostasis, plasma membrane permeabilization, and the inability to recover normal cellular function. Neuronal death may occur via multiple pathways, including apoptosis, necrosis, excitotoxicity, or autophagic cell death, and is distinguished from reversible states of stress or dysfunction.

“Pseudo-ratiometric” refers to a quantification method wherein the fluorescence intensity of a biosensor (e.g., GEDI) is normalized to the fluorescence intensity of a co-expressed, non-responsive reference protein (e.g., EGFP) within the same cell. This ratio corrects for variations in expression level, cell size, and imaging conditions, providing a robust metric for classification.

A “self-cleaving peptide” is a short amino acid sequence, such as porcine teschovirus-1 2A (P2A), that mediates ribosomal skipping during translation, resulting in the production of two separate polypeptides from a single open reading frame. This allows stoichiometric co-expression of GEDI and a reference fluorescent protein from a single transcript.

“Cytosolic localization” means that the GEDI polypeptide is distributed throughout the cytoplasm of the cell and is not targeted to specific organelles such as the endoplasmic reticulum, mitochondria, or nucleus, unless otherwise specified (e.g., in nuclear-localized variants).

“Pathological calcium elevation” refers to a sustained increase in intracellular calcium concentration to levels significantly above the physiological range (typically >5–10 µM), resulting from loss of membrane integrity, pump failure, or excessive glutamate receptor activation, and is indicative of an irreversible commitment to cell death.

“Automated survival analysis” denotes the use of computational image analysis pipelines to track individual cells over time, calculate GEDI ratios, apply a death threshold, and generate survival statistics such as Kaplan-Meier curves or cumulative risk-of-death plots without manual intervention.

### Vectors

The invention provides expression vectors encoding GEDI constructs for delivery into mammalian cells, neurons, or model organisms. These vectors comprise a promoter operably linked to a nucleic acid sequence encoding a GEDI polypeptide, optionally fused to a self-cleaving peptide and a reference fluorescent protein. In a preferred embodiment, the promoter is neuron-specific, such as the human synapsin 1 (hSyn1) promoter, to restrict expression to neurons and minimize off-target effects. Alternatively, constitutive promoters (e.g., CAG, CMV) or inducible systems (e.g., Tet-On) may be used depending on the experimental context.

Exemplary vectors include phSyn1:RGEDI-P2A-EGFP, wherein RGEDI is a red fluorescent GEDI derived from RCEPIA, P2A is the self-cleaving peptide, and EGFP is the green reference protein. Other variants include phSyn1:RGEDI-P2A-3xTagBFP2 (blue reference), phSyn1:GC150-P2A-mApple (green GEDI with red reference), and nuclear-localized versions such as phSyn1:RGEDI-NLS-P2A-EGFP-NLS, which incorporate nuclear localization signals (NLS) at both termini to concentrate the sensor in the nucleus for enhanced signal-to-noise in dense tissues.

For in vivo applications in zebrafish, vectors are constructed using the Tol2 transposon system for efficient genomic integration. These include neuroD:RGEDI-P2A-EGFP and neuroD:GC150-P2A-mApple, driven by the neurogenin D (neuroD) promoter for pan-neuronal expression. Co-expression with nitroreductase (NTR) under the same or a different promoter enables inducible ablation studies.

All vectors are codon-optimized for the target species, contain polyadenylation signals for mRNA stability, and may include antibiotic resistance genes (e.g., ampicillin, kanamycin) or fluorescent selection markers for cloning and screening. The nucleic acid sequences are verified by sequencing to ensure fidelity.

### Cells

The invention encompasses host cells transfected or transduced with GEDI-encoding vectors. Primary neurons—such as rat or mouse cortical neurons—are preferred for in vitro studies due to their relevance to human neurobiology and susceptibility to neurodegenerative insults. These cells are cultured under standard conditions and transfected at 4–5 days in vitro (DIV) using lipid-based reagents (e.g., Lipofectamine 2000) to achieve sparse labeling, facilitating single-cell tracking.

Human induced pluripotent stem cell (iPSC)-derived neurons are also within the scope, particularly for modeling genetic forms of neurodegeneration. For example, motor neurons differentiated from iPSCs carrying the SOD1 D90A mutation exhibit increased death rates detectable by GEDI, enabling patient-specific disease modeling and drug screening.

Non-neuronal cells, such as HEK293 cells, may also express GEDI and show differential fluorescence between live and dead states, confirming that the mechanism—loss of calcium homeostasis—is conserved across cell types. However, the primary utility of GEDI lies in neuronal systems where calcium dysregulation is a central feature of degeneration.

Stable cell lines expressing GEDI can be generated by lentiviral transduction or genomic integration, allowing long-term experiments without repeated transfection. Cells may be co-transfected with disease-associated proteins (e.g., mutant huntingtin, α-synuclein, TDP-43) to model specific neurodegenerative conditions.

### Animals

Transgenic animals expressing GEDI are provided for in vivo studies. Zebrafish (Danio rerio) are a preferred model due to their optical transparency, rapid development, and genetic tractability. Stable transgenic lines are created by co-injecting GEDI-encoding plasmids with Tol2 transposase mRNA into one-cell-stage embryos, followed by screening of founder fish and establishment of germline-transmitting lines. Expression under neuron-specific promoters (e.g., neuroD, mnx1) ensures labeling of relevant neuronal populations.

Rodent models, including mice and rats, can also be engineered to express GEDI using viral vectors (e.g., AAV, lentivirus) injected into specific brain regions or through germline transgenesis. These models enable the study of neurodegeneration in the context of intact neural circuits, behavior, and aging.

In all animal models, GEDI expression allows longitudinal, non-invasive imaging of neuronal death in response to genetic, toxic, or environmental challenges. For example, in zebrafish larvae expressing NTR in motor neurons, addition of metronidazole induces ablation, and GEDI signal provides an acute readout of death kinetics, surpassing traditional methods based on morphology or motility loss.

### Methods of Use

The invention provides methods for detecting and quantifying neuronal death using GEDI. In a typical protocol, neurons are transfected with a GEDI vector and imaged over time using automated microscopy. Fluorescence intensities of GEDI and reference channels are extracted for each cell, and the GEDI ratio is calculated. A death threshold is established from control populations (e.g., untreated vs. NaN3-treated) using the formula:  
GEDI ratio threshold = [(mean GEDI ratio_dead − mean GEDI ratio_live) × 0.25] + mean GEDI ratio_live.  
Cells exceeding this threshold are classified as dead.

Applications include:  
(1) **Toxicity screening**: Neurons exposed to glutamate, rotenone, or other neurotoxins are monitored to identify resistant subpopulations and dose-response relationships.  
(2) **Disease modeling**: Co-expression of GEDI with mutant proteins (e.g., HttEx1Q97, α-synuclein) reveals kinetics of degeneration and enables comparison of pathogenic vs. benign variants.  
(3) **Drug discovery**: Candidate therapeutics are tested for their ability to delay or prevent GEDI signal elevation, providing a functional readout of neuroprotection.  
(4) **In vivo imaging**: Immobilized zebrafish larvae expressing GEDI are imaged in 4D (x, y, z, time) to track single-neuron death in the intact brain.  
(5) **Survival statistics**: Single-cell death times are used to generate Kaplan-Meier curves and Cox proportional hazards models, analogous to clinical trial endpoints.

GEDIs are compatible with other biosensors (e.g., caspase reporters, mitochondrial dyes) for multimodal analysis of death pathways. Importantly, GEDI signal precedes morphological changes, TUNEL staining, and caspase activation, offering the earliest possible detection of irreversible commitment to death.

### Kits

Kits are provided for convenient implementation of GEDI technology. Each kit includes:  
- One or more purified plasmid DNAs encoding GEDI variants (e.g., RGEDI-P2A-EGFP, GC150-P2A-mApple);  
- Control plasmids (e.g., empty vector, disease-associated protein constructs);  
- Transfection reagents optimized for neurons;  
- Detailed protocols for cell culture, transfection, imaging, and data analysis;  
- Software scripts or access to cloud-based platforms for automated GEDI ratio calculation and survival analysis.

Optional components may include pre-coated multi-well plates, mounting media for zebrafish imaging, and positive/negative control compounds (e.g., NaN3, glutamate antagonists). Kits are tailored for specific applications, such as high-throughput screening or iPSC-neuron studies.

## EXAMPLES

### Example 1

Development and validation of RGEDI-P2A-EGFP as a genetically encoded death indicator in rat primary cortical neurons. Rat cortical neurons were isolated at embryonic day 20–21 and cultured in Neurobasal medium. At 4–5 DIV, neurons were transfected with phSyn1:RGEDI-P2A-EGFP using Lipofectamine 2000. After 24 hours, neurons were subjected to either electrical field stimulation (30 Hz for 3 seconds) or treatment with 2% sodium azide (NaN3), a metabolic toxin that induces rapid neuronal death. Fluorescence imaging was performed using a Nikon Ti-E inverted microscope equipped with appropriate filter sets for EGFP (excitation 490/20 nm, emission 535/50 nm) and RGEDI (excitation 543/22 nm, emission 617/73 nm). Time-lapse images were acquired every 30 seconds for 10 minutes post-stimulation or toxin addition.

Results showed that electrical stimulation induced a robust increase in GCaMP6f fluorescence (used as a positive control for physiological calcium transients) but no significant change in RGEDI fluorescence. In contrast, NaN3 treatment caused a rapid and sustained increase in both GCaMP6f and RGEDI signals. Quantification of the maximum ΔF/F response revealed that the ratio of stimulation-induced to death-induced signal was near zero for RGEDI, confirming its insensitivity to physiological activity. Removal of extracellular calcium from the imaging buffer abolished the NaN3-induced RGEDI response, demonstrating dependence on calcium influx from the extracellular space.

To enable ratiometric quantification, the RGEDI-P2A-EGFP construct was designed such that a single transcript produces equimolar amounts of RGEDI and EGFP. In live neurons, the overlay of red (RGEDI) and green (EGFP) channels appeared yellow in the soma, with green extending into neurites. Upon death, RGEDI fluorescence increased dramatically, shifting the overlay to red-yellow throughout the cell body and degenerating processes. Longitudinal imaging over 48 hours confirmed that the GEDI ratio remained stably elevated after death until cellular debris disappeared, with no instances of signal reversal. A GEDI threshold of 0.05 was established from control experiments and used to classify neurons as live or dead with >99% accuracy compared to manual curation, with GEDI often detecting death before morphological changes were evident.

### Example 2

Automated identification of glutamate-resistant neuronal subpopulations using GEDI. Rat primary cortical neurons transfected with hSyn1:RGEDI-P2A-EGFP were exposed to varying concentrations of glutamate (0, 0.01, 0.1, or 1 mM) in 96-well plates. Automated confocal microscopy was performed every 3 hours for 108 hours using a custom 4D imaging platform. For each well, composite red-green overlay images were generated, and individual neurons were segmented and tracked using custom Galaxy bioinformatics scripts.

Kaplan-Meier survival analysis revealed that 90% of neurons died within 3 hours of exposure to 0.1 or 1 mM glutamate, consistent with excitotoxic mechanisms. However, a small subpopulation (2–5%) exhibited low GEDI ratios throughout the imaging period, indicating resistance to glutamate-induced death. These resistant neurons maintained healthy morphology and EGFP fluorescence for the entire 108-hour window. To assess signal stability post-death, the decay rates of RGEDI and EGFP fluorescence were measured in neurons that died rapidly after glutamate exposure. Both fluorophores decayed with similar half-lives (~20.5 hours), confirming that the GEDI ratio remains a stable indicator of death over extended periods and does not produce false negatives due to differential photobleaching or degradation.

This example demonstrates GEDI’s utility in resolving heterogeneous responses to neurotoxic insults and identifying rare, resilient neuronal subtypes that may harbor protective genetic or epigenetic factors.

### Example 3

Application of GEDI in multiple neurodegenerative disease models and human iPSC-derived motor neurons. Rat cortical neurons were co-transfected with hSyn1:RGEDI-P2A-EGFP and plasmids encoding disease-associated proteins: HttEx1-Q97 (Huntington’s disease), α-synuclein (Parkinson’s disease), or TDP-43 (ALS/FTD). Control neurons expressed HttEx1-Q25 (non-pathogenic polyQ length) or empty vector. Longitudinal imaging over 168 hours showed progressive increases in GEDI ratio in disease-model neurons, with clear separation between live and dead populations by 48 hours. Cumulative risk-of-death (CRD) analysis using Cox proportional hazards models confirmed significantly elevated death risk in all disease models (HR = 1.73–1.83, p < 0.001).

In parallel, motor neurons were differentiated from iPSCs derived from a healthy donor and a patient carrying the SOD1 D90A mutation (associated with familial ALS). At day 19 of differentiation, neurons were transfected with hSyn1:RGEDI-P2A-EGFP and imaged every 12 hours for 96 hours. SOD1 D90A neurons exhibited a significantly higher CRD (HR = 1.26, p < 0.0001) compared to controls, validating GEDI’s applicability in human, physiologically relevant models. Notably, GEDI detected death in the absence of overt protein aggregation, underscoring its sensitivity to early, pre-morphological events in degeneration.

### Example 4

Engineering and characterization of an expanded GEDI sensor family. To broaden experimental flexibility, several GEDI variants were developed:  
- **RGEDI-P2A-3xBFP**: Uses triple-tagged BFP as the reference channel, enabling concurrent imaging of green biosensors (e.g., GCaMP).  
- **GC150-P2A-mApple**: Derived from GCaMP6-150, with higher calcium affinity (Kd ~150 µM) for improved sensitivity in low-calcium environments.  
- **Nuclear-localized GEDIs (e.g., RGEDI-NLS-P2A-EGFP-NLS)**: Incorporate SV40 nuclear localization signals to concentrate fluorescence in the nucleus, enhancing signal clarity in dense tissues.

All variants were tested in rat neurons exposed to NaN3. Peak GEDI ratios and response kinetics (τ) were comparable across constructs (ANOVA p = 0.44–0.65), confirming equivalent performance. Crucially, none responded to 30 Hz field stimulation, verifying specificity for pathological calcium. GC150-P2A-mApple was particularly effective in zebrafish (see Example 6), where extracellular calcium is limited.

### Example 6

In vivo detection of neuronal death in zebrafish larvae using GEDI. Zebrafish embryos were co-injected at the 1-cell stage with DNA encoding neuroD:NTR-BFP (for inducible ablation) and either neuroD:RGEDI-P2A-EGFP or neuroD:GC150-P2A-mApple. At 72 hours post-fertilization (hpf), larvae were immobilized in agarose-containing ZFplates and treated with 10 µM metronidazole (MTZ) to activate NTR-mediated cell death. Automated 4D confocal imaging was performed every 12 hours for 48 hours.

RGEDI-P2A-EGFP failed to show increased GEDI ratio in MTZ-treated neurons, likely due to insufficient extracellular calcium in the zebrafish brain to saturate its lower-affinity sensor. In contrast, GC150-P2A-mApple exhibited a significant GEDI ratio increase by 24 hours post-MTZ, with clear distinction from DMSO controls (ANOVA p < 0.0001). Individual neurons co-expressing NTR-BFP and GC150-mApple were tracked in 3D, and death was confirmed by morphological fragmentation at 48 hours. Importantly, GC150 did not respond to motor circuit activation in BTS-immobilized larvae, confirming its insensitivity to physiological calcium transients.

This example establishes GC150 as the preferred GEDI for in vivo applications and demonstrates the first platform for acute, longitudinal detection of neuronal death in a whole vertebrate organism.